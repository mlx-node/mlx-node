//! Generic paged decode stepper — retires the per-family hand-copied
//! [`DecodeStep`] impls the eager paged families duplicated.
//!
//! [`PagedStepModel`] is the per-family seam: one eager paged decode step
//! ([`PagedStepModel::paged_step`], returning the RAW `[1, 1, vocab]`
//! logits), the declarative policy consts ([`PagedStepModel::EVAL`],
//! [`PagedStepModel::PIPELINED`], [`PagedStepModel::FINAL_TOKEN_POLICY`]),
//! and optional hooks for the pieces a family genuinely extends (eval
//! bookkeeping, cache cadence, the lazy-token trio, the final-token
//! materialize, `end_decode`).
//!
//! [`PagedStepper`] is the single [`DecodeStep`] impl that owns a model and
//! supplies the shared machinery every family used to copy:
//!   * `forward` — extract the scalar (`item_at_int32`) and delegate to
//!     `forward_with_token` (trait parity; the loop drives the latter).
//!   * `forward_with_token` — `paged_step(token)` then `squeeze([1])` and
//!     `needs_squeeze = false`. This NORMALIZES the old qwen3/gemma4
//!     polarity (they returned `(raw [1,1,vocab], true)` and let the loop
//!     squeeze): `run_decode_loop` squeezes axis 1 exactly when the flag is
//!     set, so the produced `[1, vocab]` tensor is identical either way —
//!     the footgun is gone, not preserved.
//!   * `eval_step` — dispatch on `M::EVAL` (the measured per-family
//!     sync/async contract), or the model's own override.
//!   * `maintain_cache` — the paged cadence
//!     (`maybe_clear_cache_for_paged_step`) unless the model overrides.
//!   * the lazy-token pipeline trio — forwards to the model hooks; gated
//!     by `M::PIPELINED` through `supports_token_pipeline`.
//!   * `materialize_final` — gated on `M::FINAL_TOKEN_POLICY`:
//!     `AlwaysDrop` compiles to the `Ok(())` no-op the trait default
//!     always produced, so conv/GDN/mamba families can never reach a
//!     re-forward through this stepper.

use napi::bindgen_prelude::*;

use crate::array::MxArray;
use crate::engine::backend::DecodeStep;
use crate::engine::paged_epilogue::FinalTokenPolicy;

/// Per-step eval policy for a paged stepper — the measured sync/async
/// contract each family picked; the variants port the historical
/// `eval_step` bodies verbatim.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EvalPolicy {
    /// `MxArray::async_eval_arrays(&[next_token])` — sampled token only;
    /// the logits stay lazy.
    AsyncToken,
    /// `MxArray::async_eval_arrays(&[next_token, logits])` — both,
    /// unconditionally (qwen3, muse paged).
    AsyncTokenAndLogits,
    /// The [`DecodeStep`] default: token async always; logits async only
    /// when the think-budget force discarded the sample.
    AsyncTokenAndForcedLogits,
    /// `next_token.eval()` — one synchronous wait pulls the logits AND the
    /// paged K/V writes through the dependency chain, so the loop-top
    /// `y.eval()` no-ops (the measured ~5% win on fast paged decodes;
    /// k2/lfm2/nemotron paged).
    SyncToken,
    /// `next_token.eval()` plus `logits.eval()` when budget-forced — a
    /// forced host token has no dependency on the forward graph
    /// (qwen3_5 dense/MoE paged).
    SyncTokenAndForcedLogits,
}

impl EvalPolicy {
    /// The verbatim per-variant `eval_step` body.
    pub(crate) fn eval_step(self, next_token: &MxArray, logits: &MxArray, budget_forced: bool) {
        match self {
            Self::AsyncToken => MxArray::async_eval_arrays(&[next_token]),
            Self::AsyncTokenAndLogits => MxArray::async_eval_arrays(&[next_token, logits]),
            Self::AsyncTokenAndForcedLogits => {
                if budget_forced {
                    MxArray::async_eval_arrays(&[next_token, logits]);
                } else {
                    MxArray::async_eval_arrays(&[next_token]);
                }
            }
            Self::SyncToken => next_token.eval(),
            Self::SyncTokenAndForcedLogits => {
                next_token.eval();
                if budget_forced {
                    logits.eval();
                }
            }
        }
    }
}

/// Per-family seam consumed by [`PagedStepper`]'s [`DecodeStep`] impl.
///
/// The implementor is the family's per-turn paged decode state (the
/// `&mut *Inner` borrow plus any instrumentation fields) — the same
/// struct that used to implement [`DecodeStep`] by hand.
pub(crate) trait PagedStepModel {
    /// The eval policy [`PagedStepModel::eval_step`]'s default dispatches
    /// on. Required — the sync-vs-async choice is a measured per-family
    /// perf contract, never an accident.
    const EVAL: EvalPolicy;

    /// Whether the model tolerates the submit-ahead lazy-token pipeline:
    /// a placeholder record before the token id is known, plus
    /// commit/rollback for the speculative step. Pure-KV families only —
    /// conv/GDN/mamba state is not rewindable (K2 only today).
    const PIPELINED: bool = false;

    /// Whether the decode loop's final length-exit token may be
    /// re-forwarded to record its K/V — the SAME axis
    /// [`FinalTokenPolicy`] splits the epilogue's history trim on.
    /// `AlwaysDrop` (the default) compiles
    /// [`DecodeStep::materialize_final`] to the `Ok(())` no-op the trait
    /// default always produced: a family that does not opt in can never
    /// get a re-forward. `KeepAllOnLength` dispatches to
    /// [`PagedStepModel::materialize_final_token`].
    const FINAL_TOKEN_POLICY: FinalTokenPolicy = FinalTokenPolicy::AlwaysDrop;

    /// One eager paged decode step: record the token into the adapter and
    /// run the forward. Returns the RAW `[1, 1, vocab]` logits — the
    /// stepper collapses the sequence axis itself.
    fn paged_step(&mut self, token_id: u32) -> Result<MxArray>;

    /// Submit-ahead forward for the lazy-token pipeline — record the
    /// placeholder and build this step's forward from the caller's lazy
    /// `[1, 1]` ids (graph input, never drained). Only `PIPELINED` models
    /// override.
    fn paged_step_lazy(&mut self, _input_ids: &MxArray) -> Result<MxArray> {
        Err(Error::from_reason(
            "paged_step_lazy: model does not support the lazy-token pipeline",
        ))
    }

    /// Patch the pending placeholder record with the real sampled id.
    fn commit_placeholder(&mut self, _token_id: u32) -> Result<()> {
        Err(Error::from_reason(
            "commit_placeholder: model does not support the lazy-token pipeline",
        ))
    }

    /// Rewind the pending placeholder record (terminal/cancel path).
    fn rollback_placeholder(&mut self) -> Result<()> {
        Err(Error::from_reason(
            "rollback_placeholder: model does not support the lazy-token pipeline",
        ))
    }

    /// `DecodeStep::eval_step` — the default dispatches on
    /// [`Self::EVAL`]. A family with extra per-eval bookkeeping (gemma4's
    /// `pending_timing` reset on the forced path) overrides with its
    /// verbatim body.
    fn eval_step(&mut self, next_token: &MxArray, logits: &MxArray, budget_forced: bool) {
        Self::EVAL.eval_step(next_token, logits, budget_forced);
    }

    /// Body of `materialize_final` for `KeepAllOnLength` models: run ONE
    /// more `paged_step`-class forward for the final committed token and
    /// discard the logits, so the adapter's recorded token set equals the
    /// keep-all history. Err by default so a `KeepAllOnLength` model that
    /// forgets the override fails LOUDLY on a length exit instead of
    /// silently desyncing the adapter against the saved history — for an
    /// `AlwaysDrop` model the stepper never calls this.
    fn materialize_final_token(&mut self, _token_id: u32) -> Result<()> {
        Err(Error::from_reason(
            "materialize_final_token: FINAL_TOKEN_POLICY=KeepAllOnLength requires an impl",
        ))
    }

    /// `DecodeStep::maintain_cache` — the paged per-step cadence
    /// (`maybe_clear_cache_for_paged_step`). A family that extends it
    /// (gemma4's timing observe + sliding settle) or diverges (muse keeps
    /// the flat every-256 `clear_cache`) overrides with its verbatim body.
    fn maintain_cache(&mut self, step: i32) {
        crate::array::maybe_clear_cache_for_paged_step(step);
    }

    /// `DecodeStep::end_decode` — default `Ok(())`; gemma4 drains its
    /// deferred sliding-settle error here.
    fn end_decode(&mut self) -> Result<()> {
        Ok(())
    }
}

/// The single [`DecodeStep`] impl for every eager paged family. Owns the
/// family model `M` (the `&mut *Inner` borrow plus per-turn
/// instrumentation state) and supplies the shared stepper machinery; the
/// family's `PagedBackend::PagedDecode` GAT aliases
/// `PagedStepper<FamilyPagedDecode<'a>>`.
pub(crate) struct PagedStepper<M: PagedStepModel>(pub M);

impl<M: PagedStepModel> DecodeStep for PagedStepper<M> {
    /// Off the hot path — the engine drives `forward_with_token` (the
    /// scalar was already read once at the loop top). Kept for trait
    /// parity; extract then delegate.
    fn forward(&mut self, input_ids: &MxArray) -> Result<(MxArray, bool)> {
        let token_id = input_ids.item_at_int32(0)? as u32;
        self.forward_with_token(input_ids, token_id)
    }

    fn forward_with_token(
        &mut self,
        _input_ids: &MxArray,
        token_id: u32,
    ) -> Result<(MxArray, bool)> {
        // `paged_step` returns RAW `[1, 1, vocab]`; the `squeeze([1])`
        // collapses to `[1, vocab]` here so `needs_squeeze = false`. This
        // normalizes the old qwen3/gemma4 `true` polarity — the loop's
        // `needs_squeeze` arm produced the identical tensor, so the
        // contract downstream is unchanged.
        let logits = self.0.paged_step(token_id)?.squeeze(Some(&[1]))?;
        Ok((logits, false))
    }

    fn supports_token_pipeline(&self) -> bool {
        M::PIPELINED
    }

    fn forward_with_lazy_token(&mut self, input_ids: &MxArray) -> Result<(MxArray, bool)> {
        // Same squeeze polarity as `forward_with_token`: the stepper
        // collapses `[1, 1, vocab] -> [1, vocab]`, so `false`.
        let logits = self.0.paged_step_lazy(input_ids)?.squeeze(Some(&[1]))?;
        Ok((logits, false))
    }

    fn commit_lazy_token(&mut self, token_id: u32) -> Result<()> {
        self.0.commit_placeholder(token_id)
    }

    fn rollback_lazy_step(&mut self) -> Result<()> {
        self.0.rollback_placeholder()
    }

    fn eval_step(&mut self, next_token: &MxArray, logits: &MxArray, budget_forced: bool) {
        self.0.eval_step(next_token, logits, budget_forced);
    }

    fn maintain_cache(&mut self, step: i32) {
        self.0.maintain_cache(step);
    }

    fn materialize_final(&mut self, token_id: u32) -> Result<()> {
        // `AlwaysDrop` compiles to today's default `Ok(())` no-op — the
        // DO-NOT-override contract for conv/GDN/mamba families stays
        // machine-checked: a model that did not opt in can never reach a
        // re-forward through this stepper.
        match M::FINAL_TOKEN_POLICY {
            FinalTokenPolicy::KeepAllOnLength => self.0.materialize_final_token(token_id),
            FinalTokenPolicy::AlwaysDrop => Ok(()),
        }
    }

    fn end_decode(&mut self) -> Result<()> {
        self.0.end_decode()
    }
}
