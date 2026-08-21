//! Gemma4 draft speculative-decode wiring: the family-side
//! [`DsparkStepper`]/[`DsparkBackend`] implementation the engine-owned
//! [`crate::engine::dspark_turn::run_dspark_turn`] loop drives.
//!
//! The two drafts run on DIFFERENT physical target caches. DSpark verifies a
//! block against the target's PAGED pools through
//! [`crate::engine::spec_paged::SpecPagedCache`], so its turns are driven by
//! [`crate::engine::dspark_turn::run_paged_dspark_turn`] and share the paged
//! epilogue. The assistant drafter's Q-only attention reads the target's flat
//! [`super::layer_cache::Gemma4LayerCache`] K/V arrays directly, which the
//! pools cannot hand it, so it keeps the flat whole-turn core
//! (`flat_draft_chat_turn`) — the only remaining caller of that path.
//!
//! Split of responsibilities:
//!   * the DSpark DRAFT model (5-layer cross-attending transformer, markov
//!     head, confidence head, context K/V cache) lives in [`super::dspark`];
//!     the assistant draft + its stepper live in [`super::assistant`] /
//!     [`super::assistant_decode`];
//!   * the TARGET-side primitives (hidden tap, verify forward, shared-slot
//!     mask) live in [`super::model`] / [`super::layer_cache`];
//!   * the model-agnostic propose → verify → accept → stop-clamp → commit
//!     loop lives in [`crate::engine::dspark_turn`];
//!   * THIS module glues them together for gemma4: the DSpark stepper, the
//!     variant dispatch ([`Gemma4DraftTurnState`] / [`Gemma4DraftStepper`] /
//!     `begin_dspark_decode`), and the flat whole-turn core.

use std::time::Instant;

use napi::bindgen_prelude::*;

use crate::array::MxArray;
use crate::decode_profiler::DecodeProfiler;
use crate::engine::backend::{
    ChatBackend, DsparkBackend, DsparkProposal, DsparkStepper, DsparkVerifyOutput, FinalizeArgs,
    ResetScope, SpecFrontier, StreamEmitter, TurnOutput, WholeTurnArgs,
};
use crate::engine::decode::StreamingCtx;
use crate::engine::dspark_turn::{DsparkTurnArgs, PagedDsparkBackend, run_dspark_turn};
use crate::engine::finalize::compute_performance_metrics;
use crate::engine::params::{ChatParams, generated_capacity_hint};
use crate::engine::penalties::{ReasoningTracker, apply_all_penalties};
use crate::engine::spec_paged::{SpecPagedCache, VerifyTicket};
use crate::stream::{DeviceType, Stream, StreamContext};

use super::assistant_decode::{AssistantTurnState, Gemma4AssistantStepper};
use super::decoder_layer::Gemma4LayerKind;
use super::dspark::{DsparkContextCache, DsparkTap, truncate_by_confidence};
use super::model::{
    Gemma4Draft, Gemma4DsparkPrefillTap, Gemma4Inner, Gemma4SpecPagedCache,
    assistant_kv_source_indices, dspark_shared_slot_mask, eval_gemma4_caches, forward_inner,
};

/// Per-turn draft handoff from the whole-turn core's prefill to
/// [`DsparkBackend::begin_dspark_decode`], one variant per
/// [`super::model::Gemma4Draft`] variant.
///
/// The prefill-derived state travels through `Gemma4Inner::draft_turn_state`:
/// the whole-turn core stashes it right before calling `run_dspark_turn`,
/// and `begin_dspark_decode` TAKES it into the stepper (so it can never
/// leak across turns — fresh state is built every turn).
/// `begin_dspark_decode` hard-errors when the stashed variant disagrees
/// with the loaded draft variant.
pub(crate) enum Gemma4DraftTurnState {
    Dspark(DsparkTurnState),
    Assistant(AssistantTurnState),
}

/// DSpark's [`Gemma4DraftTurnState`] payload: the draft's fused-context
/// cache built by `paged_prefill_with_draft_state`, which taps the target's
/// paged prefill walk.
pub(crate) struct DsparkTurnState {
    /// The draft's fused-context K/V cache, holding one row per freshly
    /// prefilled prompt token (absolute positions
    /// `position_base .. position_base + rows`).
    pub(crate) ctx: DsparkContextCache,
    /// Absolute sequence position of the NEXT context row / target-cache
    /// slot — `cached_prefix_len + prefill_len` right after prefill, then
    /// advanced by `keep` on every commit.
    pub(crate) next_pos: i32,
}

/// Confidence-truncation threshold for drafted blocks, read ONCE at stepper
/// construction. Default `0.0` = keep-all (truncation disabled); invalid
/// values fall back to the default.
fn dspark_confidence_threshold_from_env() -> f32 {
    std::env::var("MLX_DSPARK_CONFIDENCE_THRESHOLD")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(0.0)
}

/// Per-turn gemma4 DSpark stepper ([`DsparkBackend::DsparkDecode`]).
///
/// Owns the turn's draft context cache and position cursor; borrows the
/// model for the whole decode loop. Every TARGET-side touch goes through
/// [`Gemma4SpecPagedCache`], so one cycle is that facade's checked
/// transaction: reserve the lookahead region, `record_verify` the block,
/// run the tapped layer loop over the rows it recorded, `commit_cycle` the
/// accepted prefix, then settle at the COMMITTED frontier — never inside
/// the open cycle and never at the write cursor (L-SETTLE).
///
/// The facade performs the verify WRITE (`record_verify`) rather than the
/// core, because gemma4's paged layer loop requires its rows to be recorded
/// first: `PagedKVCacheAdapter::update_keys_values` rejects a chunk whose
/// `first_logical_position` is not `current_token_count - chunk_len`.
///
/// The tapped target hiddens and the open cycle's ticket are stashed
/// between `verify` and `commit` — they never cross the engine trait (see
/// the invariant on [`DsparkStepper`]).
pub(crate) struct Gemma4DsparkStepper<'a> {
    inner: &'a mut Gemma4Inner,
    /// Paged sequence this turn owns.
    seq_id: u32,
    ctx: DsparkContextCache,
    /// Permanently use exact target-only AR for the rest of this turn after
    /// the measured break-even guard determines that speculation loses on
    /// the current hardware/context.
    ar_fallback: bool,
    /// Absolute position of the next verify block's anchor (== the committed
    /// paged frontier: prompt + anchor-exclusive generation).
    next_pos: i32,
    /// Draft `target_layer_ids` as decoder indices (strictly ascending).
    layer_ids: Vec<usize>,
    /// Per-layer paged routing, resolved once for the turn.
    layer_kinds: Vec<Gemma4LayerKind>,
    confidence_threshold: f32,
    /// The cycle `verify` opened, consumed by `commit`.
    open_cycle: Option<VerifyTicket>,
    /// Tapped `[1, 1+L, hidden]` hiddens from the last `verify` (one per
    /// `layer_ids` entry), consumed by `commit`.
    tapped: Option<Vec<MxArray>>,
    /// A one-token adaptive AR probe keeps every row it wrote, but still
    /// owns its cycle and its tap until `commit_ar_probe`.
    ar_probe_pending: bool,
}

impl Gemma4DsparkStepper<'_> {
    fn ensure_no_pending_verify(&self, op: &str) -> Result<()> {
        if self.open_cycle.is_some() || self.tapped.is_some() || self.ar_probe_pending {
            return Err(Error::from_reason(format!(
                "gemma4 DSpark {op}: previous verify was never committed"
            )));
        }
        Ok(())
    }

    fn paged_cache(&mut self) -> Gemma4SpecPagedCache<'_> {
        Gemma4SpecPagedCache::new(self.inner)
    }

    /// Close an open cycle that can no longer produce a valid row, keeping
    /// nothing. The original error is what the caller sees unless the
    /// retraction itself fails.
    fn retract_cycle(&mut self, ticket: VerifyTicket, op: &str, error: Error) -> Error {
        let seq_id = self.seq_id;
        match self.paged_cache().commit_cycle(seq_id, ticket, 0) {
            Ok(()) => error,
            Err(rollback) => Error::from_reason(format!(
                "gemma4 DSpark {op}: {error}; retracting the verify rows also failed: {rollback}"
            )),
        }
    }

    /// Record `verify_ids` at the committed frontier and run the tapped
    /// target forward over exactly those rows.
    ///
    /// A verify block is bit-for-bit an ordinary paged prefill chunk at
    /// those positions: same layer loop, same RoPE offset, same sliding
    /// mask, same slot mapping. Speculation adds no attention math.
    fn tapped_paged_verify(
        &mut self,
        verify_ids: &[u32],
        op: &str,
    ) -> Result<(MxArray, Vec<MxArray>, VerifyTicket)> {
        let seq_id = self.seq_id;
        let first_position = u32::try_from(self.next_pos).map_err(|_| {
            Error::from_reason(format!(
                "gemma4 DSpark {op}: verify anchor sits at negative position {}",
                self.next_pos
            ))
        })?;
        let ticket = self
            .paged_cache()
            .record_verify(seq_id, verify_ids)
            .map_err(Error::from_reason)?;

        let captured: Vec<MxArray>;
        let forward = {
            let mut tap = DsparkTap::new(&self.layer_ids);
            let forward = self.inner.run_paged_prefill_layer_loop(
                verify_ids,
                first_position,
                first_position,
                &self.layer_kinds,
                Some(&mut tap),
            );
            captured = tap.captured;
            forward
        };
        let hidden = match forward {
            Ok(hidden) => hidden,
            Err(error) => return Err(self.retract_cycle(ticket, op, error)),
        };
        if captured.len() != self.layer_ids.len() {
            let error = Error::from_reason(format!(
                "gemma4 DSpark {op}: tapped {} hiddens for {} configured target layers",
                captured.len(),
                self.layer_ids.len()
            ));
            return Err(self.retract_cycle(ticket, op, error));
        }
        // All-rows projection: row `i` is the target's distribution after
        // `verify_ids[i]`.
        match self.inner.project_paged_hidden(&hidden, false) {
            Ok(logits) => Ok((logits, captured, ticket)),
            Err(error) => Err(self.retract_cycle(ticket, op, error)),
        }
    }

    fn append_tapped_prefix(&mut self, tapped: &[MxArray], keep: usize, op: &str) -> Result<()> {
        let draft = self
            .inner
            .dspark_draft()
            .ok_or_else(|| Error::from_reason(format!("gemma4 DSpark {op}: no draft loaded")))?;
        let mut kept: Vec<MxArray> = Vec::with_capacity(tapped.len());
        for hidden in tapped {
            kept.push(hidden.slice_axis(1, 0, keep as i64)?);
        }
        let fused = draft.fuse_context(&kept)?;
        self.ctx.append(draft, &fused, self.next_pos)?;
        self.next_pos += keep as i32;
        Ok(())
    }

    /// Post-commit settle at the frontier the commit landed on: pending-write
    /// eval, the cold-checkpoint rung walk, and the committed-basis sliding
    /// prune. Rungs are selected by `boundary <= frontier`, so a cycle whose
    /// accept count steps the cursor straight over a rung still captures it.
    fn settle_at_committed_frontier(&mut self, op: &str) -> Result<()> {
        let seq_id = self.seq_id;
        let mut cache = self.paged_cache();
        let Some(frontier) = SpecPagedCache::frontier(&cache, seq_id) else {
            return Err(Error::from_reason(format!(
                "gemma4 DSpark {op}: the paged cache cannot name a frontier for sequence {seq_id}"
            )));
        };
        cache
            .settle_committed(seq_id, frontier.attn_tokens)
            .map_err(Error::from_reason)
    }
}

impl DsparkStepper for Gemma4DsparkStepper<'_> {
    fn supports_adaptive_ar_fallback(&self) -> bool {
        true
    }

    fn reserve_cycle_lookahead(&mut self, rows: usize) -> Result<bool> {
        if self.ar_fallback {
            // Target-only AR writes one row through the ordinary decode
            // path, which allocates for itself.
            return Ok(true);
        }
        let seq_id = self.seq_id;
        self.paged_cache()
            .reserve_lookahead(seq_id, rows)
            .map_err(Error::from_reason)
    }

    fn enter_ar_fallback(&mut self) -> Result<()> {
        if self.open_cycle.is_some() || self.tapped.is_some() || self.ar_probe_pending {
            return Err(Error::from_reason(
                "gemma4 DSpark AR fallback: cannot switch with an uncommitted verify",
            ));
        }
        let num_layers = self
            .inner
            .dspark_draft()
            .ok_or_else(|| Error::from_reason("gemma4 DSpark AR fallback: no draft loaded"))?
            .num_layers();
        // Drop the now-unused context arrays/graphs instead of retaining a
        // growing draft cache during the target-only remainder of the turn.
        self.ctx = DsparkContextCache::new(num_layers);
        self.ar_fallback = true;
        Ok(())
    }

    fn materialize_adaptive_state(&self) -> Result<()> {
        self.ctx.eval()
    }

    fn verify_ar_probe(&mut self, anchor_id: u32) -> Result<DsparkVerifyOutput> {
        if self.ar_fallback {
            return Err(Error::from_reason(
                "gemma4 DSpark AR probe: cannot calibrate after AR fallback",
            ));
        }
        self.ensure_no_pending_verify("AR probe")?;

        // One-token verify: the cycle is opened exactly as a speculative one
        // is, and its single row is kept in full. Keeping the hidden tap
        // conditions the next (speculative) calibration cycle on this anchor.
        // The engine times this whole call; fusion happens in
        // `commit_ar_probe` afterward.
        let (logits, tapped, ticket) = self.tapped_paged_verify(&[anchor_id], "AR probe")?;
        self.open_cycle = Some(ticket);
        self.tapped = Some(tapped);
        self.ar_probe_pending = true;
        Ok(DsparkVerifyOutput { logits })
    }

    fn commit_ar_probe(&mut self) -> Result<()> {
        if !self.ar_probe_pending {
            return Err(Error::from_reason(
                "gemma4 DSpark AR probe commit: no pending probe",
            ));
        }
        let tapped = self.tapped.take().ok_or_else(|| {
            Error::from_reason("gemma4 DSpark AR probe commit: no stashed tapped hiddens")
        })?;
        if let Some(first) = tapped.first()
            && first.shape_at(1)? != 1
        {
            return Err(Error::from_reason(format!(
                "gemma4 DSpark AR probe commit: tapped hiddens cover {} positions, expected 1",
                first.shape_at(1)?
            )));
        }
        let ticket = self.open_cycle.take().ok_or_else(|| {
            Error::from_reason("gemma4 DSpark AR probe commit: no pending verify cycle")
        })?;
        let seq_id = self.seq_id;
        self.paged_cache()
            .commit_cycle(seq_id, ticket, 1)
            .map_err(Error::from_reason)?;
        self.append_tapped_prefix(&tapped, 1, "AR probe commit")?;
        self.ar_probe_pending = false;
        self.settle_at_committed_frontier("AR probe commit")
    }

    fn propose(
        &mut self,
        anchor_id: u32,
        max_len: usize,
        params: &ChatParams,
        rng: &mut dyn rand::Rng,
    ) -> Result<DsparkProposal> {
        if self.ar_fallback {
            return Err(Error::from_reason(
                "gemma4 DSpark propose: engine called propose after AR fallback",
            ));
        }
        let draft = self
            .inner
            .dspark_draft()
            .ok_or_else(|| Error::from_reason("gemma4 DSpark propose: no draft model loaded"))?;
        if max_len == 0 {
            return Err(Error::from_reason(
                "gemma4 DSpark propose: engine contract violation (max_len == 0 cycles skip propose)",
            ));
        }

        // Draft block: `[anchor, MASK x (max_len - 1)]` at the block's
        // absolute positions (anchor sits at `next_pos`). ONE forward over
        // the persisted fused context; row k's logits draft the token at
        // absolute position `next_pos + k + 1`.
        let mask_id = draft.config.mask_token_id;
        let mut block_ids: Vec<i32> = Vec::with_capacity(max_len);
        block_ids.push(anchor_id as i32);
        block_ids.resize(max_len, mask_id);
        let block = MxArray::from_int32(&block_ids, &[1, max_len as i64])?;
        let (block_hidden, block_logits) = draft.forward_block(&block, self.next_pos, &self.ctx)?;

        // Sequential markov-chained sampling. Greedy detection INSIDE
        // `sample_block_sequential` uses the engine's
        // `sampling::is_greedy_temperature` predicate — the same predicate
        // `run_dspark_turn` keys its accept policy on — so the returned
        // `dists` are empty exactly when the engine expects them empty, and
        // at sampled temperature each row is the EXACT distribution the
        // draw came from.
        let cfg = params.sampling_config.unwrap_or_default();
        let (mut draft_ids, mut draft_dists) =
            draft.sample_block_sequential(&block_logits, anchor_id as i32, max_len, &cfg, rng)?;

        // Confidence truncation (opt-in via MLX_DSPARK_CONFIDENCE_THRESHOLD,
        // read once at stepper construction): keep the longest prefix whose
        // keep-probability clears the threshold. Returning FEWER tokens than
        // `max_len` is allowed by the engine contract (never more).
        if self.confidence_threshold > 0.0 {
            let mut prev_tokens: Vec<i32> = Vec::with_capacity(max_len);
            prev_tokens.push(anchor_id as i32);
            prev_tokens.extend_from_slice(&draft_ids[..max_len - 1]);
            let keep_probs = draft.confidence_keep_probs(&block_hidden, &prev_tokens)?;
            let keep = truncate_by_confidence(&keep_probs, self.confidence_threshold);
            draft_ids.truncate(keep);
            draft_dists.truncate(keep);
        }

        Ok(DsparkProposal {
            draft_ids,
            draft_dists,
        })
    }

    fn verify(&mut self, verify_ids: &[u32]) -> Result<DsparkVerifyOutput> {
        if verify_ids.is_empty() {
            return Err(Error::from_reason(
                "gemma4 DSpark verify: empty verify block",
            ));
        }
        // Commit-exactly-once defense: a second verify before the previous
        // cycle's commit would orphan its ticket (the cache would then hold
        // TWO uncommitted verify blocks).
        self.ensure_no_pending_verify("verify")?;

        if self.ar_fallback {
            if verify_ids.len() != 1 {
                return Err(Error::from_reason(format!(
                    "gemma4 DSpark AR fallback verify requires one anchor token, got {}",
                    verify_ids.len()
                )));
            }
            // Exact target-only AR: the ordinary paged decode step, which
            // records its own row and opens no cycle.
            let logits = self
                .inner
                .run_paged_decode_step_for(self.seq_id, verify_ids[0])?;
            return Ok(DsparkVerifyOutput { logits });
        }

        let (logits, tapped, ticket) = self.tapped_paged_verify(verify_ids, "verify")?;
        self.open_cycle = Some(ticket);
        self.tapped = Some(tapped);
        Ok(DsparkVerifyOutput { logits })
    }

    fn commit(&mut self, keep: usize, total_written: usize) -> Result<()> {
        if self.ar_fallback {
            if keep != 1 || total_written != 1 {
                return Err(Error::from_reason(format!(
                    "gemma4 DSpark AR fallback commit requires keep=1,total_written=1, got keep={keep},total_written={total_written}"
                )));
            }
            self.next_pos += 1;
            return self.settle_at_committed_frontier("AR fallback commit");
        }
        if self.ar_probe_pending {
            return Err(Error::from_reason(
                "gemma4 DSpark commit: pending AR probe requires commit_ar_probe",
            ));
        }
        if keep == 0 {
            return Err(Error::from_reason(
                "gemma4 DSpark commit: engine contract violation (keep must be >= 1 — the anchor's slot is unconditionally kept)",
            ));
        }
        {
            let tapped = self.tapped.as_ref().ok_or_else(|| {
                Error::from_reason("gemma4 DSpark commit: no stashed tapped hiddens")
            })?;
            if let Some(first) = tapped.first()
                && first.shape_at(1)? != total_written as i64
            {
                return Err(Error::from_reason(format!(
                    "gemma4 DSpark commit: stashed tapped hiddens cover {} positions but the engine reports a {}-token verify block",
                    first.shape_at(1)?,
                    total_written
                )));
            }
        }
        let ticket = self
            .open_cycle
            .take()
            .ok_or_else(|| Error::from_reason("gemma4 DSpark commit: no pending verify cycle"))?;
        let tapped = self
            .tapped
            .take()
            .ok_or_else(|| Error::from_reason("gemma4 DSpark commit: no stashed tapped hiddens"))?;

        // Target side: the facade keeps the first `keep` rows of the cycle
        // and derives the rollback from its own ticket.
        let seq_id = self.seq_id;
        self.paged_cache()
            .commit_cycle(seq_id, ticket, keep)
            .map_err(Error::from_reason)?;

        // Draft side: fuse the kept prefix of the tapped hiddens and append
        // it to the persisted context at the block's base position, then
        // advance the cursor. The boundary token has no slot on either side
        // — it re-enters as the next cycle's verify anchor.
        self.append_tapped_prefix(&tapped, keep, "commit")?;

        self.settle_at_committed_frontier("commit")
    }

    fn eval_boundary(&self, token: &MxArray) {
        // Schedule-only async eval of the next cycle's anchor (gemma4's
        // decode eval pattern: token only, never the logits).
        MxArray::async_eval_arrays(&[token]);
    }

    fn frontier(&self) -> Option<SpecFrontier> {
        // Pure-attention target: the frontier is the row count every KV
        // group agrees on; the drafter's private context cache is not target
        // state.
        self.inner
            .kv_cache_coordinator
            .as_ref()?
            .spec_frontier(self.seq_id)
    }
}

/// Per-turn stepper dispatch: [`DsparkBackend::DsparkDecode`] is ONE
/// associated type, so the two variant steppers ship behind this enum with
/// straight 4-method delegation. Constructed only by
/// [`DsparkBackend::begin_dspark_decode`], which hard-errors when the
/// stashed [`Gemma4DraftTurnState`] variant disagrees with the loaded
/// [`Gemma4Draft`] variant.
pub(crate) enum Gemma4DraftStepper<'a> {
    Dspark(Gemma4DsparkStepper<'a>),
    Assistant(Gemma4AssistantStepper<'a>),
}

impl DsparkStepper for Gemma4DraftStepper<'_> {
    fn supports_adaptive_ar_fallback(&self) -> bool {
        match self {
            Self::Dspark(stepper) => stepper.supports_adaptive_ar_fallback(),
            Self::Assistant(stepper) => stepper.supports_adaptive_ar_fallback(),
        }
    }

    fn reserve_cycle_lookahead(&mut self, rows: usize) -> Result<bool> {
        match self {
            Self::Dspark(stepper) => stepper.reserve_cycle_lookahead(rows),
            Self::Assistant(stepper) => stepper.reserve_cycle_lookahead(rows),
        }
    }

    fn enter_ar_fallback(&mut self) -> Result<()> {
        match self {
            Self::Dspark(stepper) => stepper.enter_ar_fallback(),
            Self::Assistant(stepper) => stepper.enter_ar_fallback(),
        }
    }

    fn materialize_adaptive_state(&self) -> Result<()> {
        match self {
            Self::Dspark(stepper) => stepper.materialize_adaptive_state(),
            Self::Assistant(stepper) => stepper.materialize_adaptive_state(),
        }
    }

    fn verify_ar_probe(&mut self, anchor_id: u32) -> Result<DsparkVerifyOutput> {
        match self {
            Self::Dspark(stepper) => stepper.verify_ar_probe(anchor_id),
            Self::Assistant(_) => Err(Error::from_reason(
                "gemma4 assistant draft does not support adaptive AR calibration",
            )),
        }
    }

    fn commit_ar_probe(&mut self) -> Result<()> {
        match self {
            Self::Dspark(stepper) => stepper.commit_ar_probe(),
            Self::Assistant(_) => Err(Error::from_reason(
                "gemma4 assistant draft does not support adaptive AR calibration",
            )),
        }
    }

    fn propose(
        &mut self,
        anchor_id: u32,
        max_len: usize,
        params: &ChatParams,
        rng: &mut dyn rand::Rng,
    ) -> Result<DsparkProposal> {
        match self {
            Self::Dspark(stepper) => stepper.propose(anchor_id, max_len, params, rng),
            Self::Assistant(stepper) => stepper.propose(anchor_id, max_len, params, rng),
        }
    }

    fn verify(&mut self, verify_ids: &[u32]) -> Result<DsparkVerifyOutput> {
        match self {
            Self::Dspark(stepper) => stepper.verify(verify_ids),
            Self::Assistant(stepper) => stepper.verify(verify_ids),
        }
    }

    fn commit(&mut self, keep: usize, total_written: usize) -> Result<()> {
        match self {
            Self::Dspark(stepper) => stepper.commit(keep, total_written),
            Self::Assistant(stepper) => stepper.commit(keep, total_written),
        }
    }

    fn eval_boundary(&self, token: &MxArray) {
        match self {
            Self::Dspark(stepper) => stepper.eval_boundary(token),
            Self::Assistant(stepper) => stepper.eval_boundary(token),
        }
    }

    fn frontier(&self) -> Option<SpecFrontier> {
        match self {
            Self::Dspark(stepper) => stepper.frontier(),
            Self::Assistant(stepper) => stepper.frontier(),
        }
    }
}

impl DsparkBackend for Gemma4Inner {
    type DsparkDecode<'a>
        = Gemma4DraftStepper<'a>
    where
        Self: 'a;

    fn begin_dspark_decode(&mut self, _block_size: usize) -> Result<Self::DsparkDecode<'_>> {
        let state = self.draft_turn_state.take().ok_or_else(|| {
            Error::from_reason(
                "gemma4 draft decode: begin_dspark_decode requires a prepared draft context \
                 (the draft whole-turn core's prefill must run first)",
            )
        })?;
        match state {
            Gemma4DraftTurnState::Dspark(state) => {
                let layer_ids: Vec<usize> = {
                    let draft = self.dspark_draft().ok_or_else(|| {
                        Error::from_reason(
                            "gemma4 draft decode: a DSpark turn state is stashed but the loaded \
                             draft is not the DSpark variant",
                        )
                    })?;
                    draft
                        .config
                        .target_layer_ids
                        .iter()
                        .map(|&id| id as usize)
                        .collect()
                };
                let confidence_threshold = dspark_confidence_threshold_from_env();
                let layer_kinds = self.compute_layer_kinds()?;
                let seq_id = self.active_paged_seq;
                Ok(Gemma4DraftStepper::Dspark(Gemma4DsparkStepper {
                    inner: self,
                    seq_id,
                    ctx: state.ctx,
                    ar_fallback: false,
                    next_pos: state.next_pos,
                    layer_ids,
                    layer_kinds,
                    confidence_threshold,
                    open_cycle: None,
                    tapped: None,
                    ar_probe_pending: false,
                }))
            }
            Gemma4DraftTurnState::Assistant(state) => {
                if self.assistant_draft().is_none() {
                    return Err(Error::from_reason(
                        "gemma4 draft decode: an assistant turn state is stashed but the loaded \
                         draft is not the assistant variant",
                    ));
                }
                let kv_sources = assistant_kv_source_indices(&self.config)?;
                let shared_slots = dspark_shared_slot_mask(&self.config);
                Ok(Gemma4DraftStepper::Assistant(
                    Gemma4AssistantStepper::from_turn_state(self, state, kv_sources, shared_slots),
                ))
            }
        }
    }
}

impl PagedDsparkBackend for Gemma4Inner {
    fn paged_prefill_with_draft_state(
        &mut self,
        suffix_tokens: &[u32],
        prefix: &Self::PrefixState,
        _stream: Stream,
    ) -> Result<MxArray> {
        let (layer_ids, draft_layers) = {
            let draft = self.dspark_draft().ok_or_else(|| {
                Error::from_reason("gemma4 DSpark paged prefill: no DSpark draft model loaded")
            })?;
            let layer_ids: Vec<usize> = draft
                .config
                .target_layer_ids
                .iter()
                .map(|&id| id as usize)
                .collect();
            (layer_ids, draft.num_layers())
        };
        let mut ctx = DsparkContextCache::new(draft_layers);
        // Diagnostic step -1 (prefill), matching `PagedBackend::paged_prefill`.
        crate::models::gemma4::diagnostic::set_step(-1);
        let last_logits = {
            let mut draft_tap = Gemma4DsparkPrefillTap {
                layer_ids: &layer_ids,
                ctx: &mut ctx,
            };
            self.run_paged_prefill_chunk(
                &prefix.full_tokens,
                suffix_tokens,
                prefix.effective_cached_prefix_len as u32,
                prefix.sliding_primed_prefix_len,
                prefix.cache_salt,
                Some(&mut draft_tap),
            )?
        };
        // Cached-prefix tokens have NO draft context rows: the drafter
        // cross-attends over whatever rows exist, and verification always
        // re-derives ground truth from the target, so a shorter context can
        // only depress acceptance, never correctness.
        self.draft_turn_state = Some(Gemma4DraftTurnState::Dspark(DsparkTurnState {
            ctx,
            next_pos: (prefix.effective_cached_prefix_len + suffix_tokens.len()) as i32,
        }));
        Ok(last_logits)
    }

    fn materialize_final_paged(&mut self, token_id: u32) -> Result<()> {
        let seq_id = self.active_paged_seq;
        let _logits = self.run_paged_decode_step_for(seq_id, token_id)?;
        Ok(())
    }

    fn dspark_block_size(&self) -> Result<usize> {
        Ok(self
            .dspark_draft()
            .ok_or_else(|| {
                Error::from_reason("gemma4 paged draft turn: no DSpark draft model loaded")
            })?
            .config
            .block_size)
    }

    fn paged_spec_seq_id(&self) -> u32 {
        self.active_paged_seq
    }
}

impl Gemma4Inner {
    /// FLAT-lane draft whole-turn core behind gemma4's
    /// `ChatBackend::run_speculative_turn` executor — the draft analog of the
    /// engine's generic `chat_turn_core` tail, sync AND streaming through
    /// the same body (`args.sink` presence selects the mode, mirroring
    /// `vision_chat_turn` / the MTP whole-turn cores).
    ///
    /// The ASSISTANT drafter is the only one that runs here: its Q-only
    /// attention reads the target's flat `Gemma4LayerCache` K/V arrays
    /// directly, which the block-paged pools cannot hand it. DSpark verifies
    /// against the paged pools and is routed to
    /// `engine::dspark_turn::run_paged_dspark_turn` instead.
    ///
    /// Flow: resolve params (+ `extra_eos_ids`) → prefix decision via the
    /// existing cache-prefix machinery → the assistant's chunked prefill
    /// (keeping only the last token's post-final-norm hidden) → anchor sample
    /// (byte-identical to the generic flow) → `run_dspark_turn` → save
    /// (AR-parity: stop exits drop the final token, length exits
    /// materialize its K/V and keep all — post-turn history AND cache
    /// offsets equal the AR flow's for every stop shape) → finalize
    /// (+ default `augment_performance`, which fills the `mtp_*`
    /// acceptance fields). Every error between prefill start and the save
    /// fails CLOSED (`draft_fail_closed`).
    pub(crate) fn flat_draft_chat_turn(
        &mut self,
        args: &mut WholeTurnArgs<'_>,
    ) -> Result<TurnOutput> {
        let tokenizer = args.tokenizer.clone();
        let eos_id = args.eos_id;
        let thinking = args.thinking;
        let is_delta = args.plan.is_delta;
        let tokens: Vec<u32> = args.tokens.to_vec();
        let is_streaming = args.sink.is_some();

        let think_end_id = tokenizer.think_end_id();
        let think_end_str = tokenizer.think_end_str().map(|s| s.to_string());

        // Owned params: re-resolve from the request config (deterministic —
        // identical to what the session core handed us in `args.params`),
        // then populate the stop-set the loop reads from
        // `ChatParams::extra_eos_ids` (the generic flow threads it as a
        // separate decode-loop argument instead).
        let mut p = ChatBackend::resolve_params(self, args.config);
        p.extra_eos_ids = ChatBackend::extra_eos_ids(self);
        let reuse_cache = p.reuse_cache;
        let report_perf = p.report_performance;
        let max_new_tokens = p.max_new_tokens;

        let generation_start = report_perf.then(Instant::now);
        let mut first_token_instant: Option<Instant> = None;

        // Prefix decision — the generic core's reset-or-delta split:
        //   Fresh: all-or-nothing `verify_cache_prefix`; strict-extend hits
        //   prefill only the tail, miss AND exact-match reset + re-prefill.
        //   Delta: strict extension by construction (`tokens` == cached
        //   history ++ delta); prefill exactly the tail.
        let prior_cached_len = if is_delta {
            self.cached_token_history.len()
        } else {
            0
        };
        let (prefill_tokens, cached_prefix_len): (Vec<u32>, usize) = if is_delta {
            (tokens[prior_cached_len..].to_vec(), prior_cached_len)
        } else {
            let hit = ChatBackend::verify_cache_prefix(self, &tokens, reuse_cache);
            if hit > 0 && hit < tokens.len() {
                tracing::info!(
                    "DSpark cache reuse: {} cached tokens, {} new tokens to prefill",
                    hit,
                    tokens.len() - hit,
                );
                (tokens[hit..].to_vec(), hit)
            } else {
                ChatBackend::reset_caches(self, ResetScope::PrefixMiss)?;
                (tokens.clone(), 0)
            }
        };

        let prompt_token_count = tokens.len();
        let mut token_history: Vec<u32> = tokens.clone();
        let mut generated_tokens: Vec<u32> =
            Vec::with_capacity(generated_capacity_hint(max_new_tokens));
        let mut finish_reason = String::from("length");

        let generation_stream = Stream::new(DeviceType::Gpu);

        let mut profiler = DecodeProfiler::new(
            ChatBackend::profiler_label(self, is_delta, is_streaming),
            ChatBackend::family_name(self),
        );
        profiler.set_prompt_tokens(prefill_tokens.len() as u32);
        profiler.snapshot_memory_before();

        let mut reasoning_tracker = ReasoningTracker::from_setup(&thinking, think_end_id);

        // Streaming decode state (mirrors the generic core; only the
        // streaming branch reads it).
        let stream_skip_special = ChatBackend::stream_skip_special_tokens(self);
        let mut decode_stream = tokenizer.inner().decode_stream(stream_skip_special);
        let mut streamed_text_len = 0usize;
        let mut last_is_reasoning = thinking.enabled;
        let mut emitter: Option<Box<dyn StreamEmitter>> =
            args.sink.map(|_| ChatBackend::stream_emitter(self));

        // --- variant prefill: target K/V + the draft's per-turn state ---
        // From here until the save runs, every error FAILS CLOSED
        // (`draft_fail_closed`): the target caches advance during
        // prefill/verify with nothing recorded in `cached_token_history`
        // yet, so an error abandoned mid-flight would leave a
        // history-vs-cache offset mismatch that a later prefix-reuse hit
        // could warm-start corrupt K/V from.
        profiler.begin_prefill();
        let (last_logits, turn_state) = match self.draft.as_ref() {
            Some(Gemma4Draft::Assistant(_)) => match self.assistant_prefill_with_hidden(
                &prefill_tokens,
                cached_prefix_len as i32,
                generation_stream,
            ) {
                Ok((logits, state)) => (logits, Gemma4DraftTurnState::Assistant(state)),
                Err(e) => return Err(self.draft_fail_closed(e)),
            },
            _ => {
                return Err(self.draft_fail_closed(Error::from_reason(
                    "gemma4 flat draft turn: the flat lane serves the assistant drafter only",
                )));
            }
        };
        profiler.end_prefill();

        // --- anchor sample: byte-identical to the generic flow ---
        let y = match apply_all_penalties(last_logits, &token_history, &p)
            .and_then(|logits| crate::sampling::sample(&logits, p.sampling_config))
        {
            Ok(y) => y,
            Err(e) => return Err(self.draft_fail_closed(e)),
        };
        y.eval();

        if let Err(e) = ChatBackend::eval_caches(self) {
            return Err(self.draft_fail_closed(e));
        }
        if report_perf {
            first_token_instant = Some(Instant::now());
        }

        // Per-cycle draft cap: the assistant drafts by chained AR steps, so
        // the resolved depth IS the cap.
        let block_size = p.mtp_depth;

        // Hand the prefill-built draft state to the stepper (taken by
        // `begin_dspark_decode` inside the loop).
        self.draft_turn_state = Some(turn_state);

        let mut rng = rand::rng();
        let mut last_in_cache;
        {
            let streaming_ctx = match (args.sink, args.cancelled, emitter.as_mut()) {
                (Some(sink), Some(cancelled), Some(em)) => Some(StreamingCtx {
                    callback: sink,
                    cancelled,
                    decode_stream: &mut decode_stream,
                    tokenizer: tokenizer.inner(),
                    streamed_text_len: &mut streamed_text_len,
                    last_is_reasoning: &mut last_is_reasoning,
                    emitter: em.as_mut(),
                }),
                _ => None,
            };
            let outcome = run_dspark_turn(
                self,
                &mut rng,
                DsparkTurnArgs {
                    y,
                    block_size,
                    params: &p,
                    reasoning_tracker: &mut reasoning_tracker,
                    profiler: &mut profiler,
                    max_new_tokens,
                    eos_id,
                    generated_tokens: &mut generated_tokens,
                    token_history: &mut token_history,
                    finish_reason: &mut finish_reason,
                    first_token_instant: &mut first_token_instant,
                    report_perf,
                    generation_stream,
                    // H2: on SYNC turns `args.cancelled` is the whole-turn
                    // cancel flag (streaming_ctx is None, so the loop's
                    // sync polls are the only cancellation path); on
                    // streaming turns it is the SAME flag StreamingCtx
                    // carries — the polls are idempotent.
                    cancel_flag: args.cancelled,
                    turn_token_observer: None,
                },
                streaming_ctx,
            );
            // Fail CLOSED on any loop error: verify advances the target
            // caches BEFORE commit resolves, so an error mid-cycle (or a
            // failed rollback/commit) can leave K/V rows the history knows
            // nothing about. The reset also clears the per-turn draft
            // stash, whether or not `begin_dspark_decode` consumed it.
            match outcome {
                Ok(o) => last_in_cache = o.last_in_cache,
                Err(e) => return Err(self.draft_fail_closed(e)),
            }
        }

        // --- save: AR-parity drop-last + length-exit materialization ---
        // The loop reports `last_in_cache == false` on EVERY stop-shaped
        // exit (in-cycle stop tokens are never committed — the loop's
        // AR-parity exclusion — and boundary stops never had a slot) and on
        // clean length exits (the final token is an unverified boundary).
        //   Stop exits (`finish_reason != "length"`): drop the final token
        //   from the persisted history — exactly the AR save, which never
        //   forwards its final token and drops it on every non-length stop.
        //   Length exits: the AR flow keeps ALL tokens and materializes the
        //   final token's K/V with one extra forward
        //   (`Gemma4Decode::materialize_final`); mirror it so the physical
        //   cache offsets equal the keep-all history.
        if finish_reason == "length"
            && !last_in_cache
            && let Some(&final_token) = generated_tokens.last()
        {
            if let Err(e) = self.flat_materialize_final(final_token, generation_stream) {
                return Err(self.draft_fail_closed(e));
            }
            last_in_cache = true;
        }
        let drop_last = !last_in_cache;
        let history_tokens: &[u32] = if drop_last && !generated_tokens.is_empty() {
            &generated_tokens[..generated_tokens.len() - 1]
        } else {
            &generated_tokens
        };
        let mut new_history = Vec::with_capacity(tokens.len() + history_tokens.len());
        new_history.extend_from_slice(&tokens);
        new_history.extend_from_slice(history_tokens);
        self.cached_token_history = new_history;
        if !is_delta {
            // Fresh text-only turn: clear any stale media keys (the draft
            // path is text-only by its speculative-plan gate).
            self.cached_image_key = None;
            self.cached_audio_key = None;
        }

        // --- finalize ---
        let performance = if report_perf {
            compute_performance_metrics(
                generation_start,
                first_token_instant,
                prefill_tokens.len(),
                generated_tokens.len(),
            )
            .map(|mut m| {
                // Default augmentation fills the mtp_* acceptance fields
                // from the profiler's recorded DSpark cycles.
                ChatBackend::augment_performance(self, &profiler, &mut m);
                m
            })
        } else {
            None
        };
        let reasoning_tokens = reasoning_tracker.reasoning_token_count();

        if let (Some(sink), Some(em)) = (args.sink, emitter.as_mut()) {
            // Residual flush through the emitter (same skip-special flag as
            // the in-loop DecodeStream so `streamed_text_len` accounting
            // stays consistent).
            //
            // CANCEL SEMANTICS (deliberate, verified AR/MTP parity): on a
            // cancelled turn the loop's clamp commits the cancel-observed
            // token to `generated_tokens` without step-streaming it; this
            // flush then delivers its text, so the TOTAL streamed text
            // equals `decode(generated_tokens)` — exactly the AR loop's
            // documented origin/main contract (`engine/decode.rs`
            // cancel-snapshot comment: the token is pushed at the loop top,
            // the break skips the detok, and the post-loop residual flush
            // re-streams the tail) and the MTP core's behavior (initial-arm
            // skip + unconditioned family flush). Suppressing the suffix
            // here would make a cancelled DSpark stream the ONLY path whose
            // streamed text cannot reconstruct the terminal chunk's
            // raw_text. The suffix is pinned to exactly ONE token by
            // `dspark_turn_streaming_cancel_in_clamp_commits_exactly_once`.
            let full_text = tokenizer
                .decode_sync(&generated_tokens, stream_skip_special)
                .unwrap_or_else(|e| {
                    tracing::warn!("Failed to decode generated tokens: {}", e);
                    String::new()
                });
            if full_text.len() > streamed_text_len {
                let residual = &full_text[streamed_text_len..];
                em.on_residual(residual, last_is_reasoning, p.include_reasoning, sink);
            }
        }

        let reported_prompt_tokens: u32 = if is_delta && is_streaming {
            let delta_len = prompt_token_count - prior_cached_len;
            ChatBackend::stream_delta_prompt_tokens(self, prompt_token_count, delta_len)
        } else {
            prompt_token_count as u32
        };

        let mut result = ChatBackend::finalize_turn(
            self,
            FinalizeArgs {
                tokenizer: &tokenizer,
                generated_tokens: &generated_tokens,
                finish_reason,
                think_end_id,
                think_end_str: think_end_str.as_deref(),
                performance,
                include_reasoning: p.include_reasoning,
                thinking_enabled: thinking.enabled,
                prompt_tokens: reported_prompt_tokens,
                reasoning_tokens,
            },
        )?;
        // `thinking_enabled` is replay provenance for the model-provided
        // template, not Gemma4's decode-time ThinkingSetup (which is disabled
        // by policy). Set it before both the terminal stream chunk and sync
        // return, matching the generic and paged engines.
        result.thinking_enabled =
            crate::engine::params::resolve_enable_thinking(args.config).unwrap_or(true);
        // cached_tokens mirrors the session core's overwrite: fresh turns
        // report the matched prefix, delta turns the prior history length.
        result.cached_tokens = if is_delta {
            prior_cached_len as u32
        } else {
            cached_prefix_len as u32
        };

        if let (Some(sink), Some(em)) = (args.sink, emitter.as_mut()) {
            em.finish(&result, sink);
            return Ok(TurnOutput::Streamed);
        }
        Ok(TurnOutput::Complete(Box::new(result)))
    }

    /// Fail CLOSED after a draft turn error that may have left the target
    /// caches advanced beyond `cached_token_history` (prefill and verify
    /// write K/V before the save records anything): drop the whole warm
    /// session via `reset_caches_sync` (caches → `None`, history/media
    /// keys/sliding checkpoints cleared) plus the per-turn draft stash, so
    /// no later turn can prefix-match into corrupt or misaligned K/V. The
    /// next fresh turn takes the cold path (full re-prefill); a delta turn
    /// on the dropped session is rejected by the live-session guard.
    /// Returns the error for `return Err(self.draft_fail_closed(e))`
    /// ergonomics.
    fn draft_fail_closed(&mut self, err: Error) -> Error {
        // Infallible today (`caches = None` + field clears); even if it
        // ever grows a fallible arm, nothing warm-reusable can survive it.
        let _ = self.reset_caches_sync();
        self.draft_turn_state = None;
        err
    }

    /// LENGTH-exit only: run ONE more flat forward for the final emitted
    /// token so its K/V lands in the live session caches, then DISCARD the
    /// logits — the flat-lane analog of `Gemma4Decode::materialize_final`
    /// (the AR flow keeps every token on length exits and materializes the
    /// final one; the save's keep-all history then equals the physical cache
    /// offsets). No sample / push / emit.
    ///
    /// The flat lane carries no paged adapter, so there is no cold tier for a
    /// sliding decode-boundary rung to describe and nothing to publish here.
    fn flat_materialize_final(&mut self, token_id: u32, stream: Stream) -> Result<()> {
        let caches = self
            .caches
            .as_mut()
            .ok_or_else(|| Error::from_reason("gemma4 DSpark materialize_final: caches missing"))?;
        let input_ids = MxArray::from_int32(&[token_id as i32], &[1, 1])?;
        let _stream_ctx = StreamContext::new(stream);
        crate::models::gemma4::diagnostic::set_step(-1);
        let _logits = forward_inner(
            &input_ids,
            &self.embed_tokens,
            &self.layers,
            caches,
            &self.final_norm,
            &self.lm_head,
            self.embed_weight_t.as_ref(),
            self.ple.as_ref(),
            &self.config,
        )?;
        eval_gemma4_caches(caches)?;
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::engine::plan::{
        DecoderPlan, MediaCapabilities, MediaInputs, SpeculativeKind, TurnPlan,
    };
    use crate::engine::types::ChatConfig;
    use crate::models::gemma4::assistant::{AssistantConfig, AssistantDraftModel};
    use crate::models::gemma4::config::Gemma4Config;
    use crate::models::gemma4::dspark::{DsparkConfig, DsparkDraftModel};
    use crate::models::gemma4::model::Gemma4Draft;

    /// Tiny flat-path Gemma4 config (paged OFF so `Gemma4Inner::new` builds
    /// no adapter): 4 hybrid layers, one KV-shared.
    pub(crate) fn tiny_target_config() -> Gemma4Config {
        serde_json::from_value(tiny_target_config_value())
            .expect("tiny Gemma4 config must deserialize")
    }

    /// [`tiny_target_config`] with an overridden sliding window — window 2
    /// makes any verify block over 2 rows violate the
    /// `snapshot_before_verify` rollback invariant (the fail-closed
    /// regression's REAL, unmocked mid-turn error).
    pub(crate) fn tiny_target_config_with_window(window: i64) -> Gemma4Config {
        let mut v = tiny_target_config_value();
        v["sliding_window"] = serde_json::json!(window);
        serde_json::from_value(v).expect("tiny Gemma4 config must deserialize")
    }

    fn tiny_target_config_value() -> serde_json::Value {
        serde_json::json!({
            "vocab_size": 16,
            "hidden_size": 8,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "intermediate_size": 16,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": true,
            "max_position_embeddings": 128,
            "sliding_window": 8,
            "layer_types": [
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention"
            ],
            "num_kv_shared_layers": 1,
            "use_block_paged_cache": false,
            // Explicitly EMPTY: the config default is [1], which is inside
            // the tiny 16-token vocab — random placeholder-weight turns
            // would then stop nondeterministically on token 1 instead of
            // running to their length budget (the whole-turn tests pass an
            // out-of-vocab session eos_id=999 as the only stop).
            "eos_token_ids": []
        })
    }

    /// [`tiny_target_config`] with the block-paged coordinator ON.
    ///
    /// The paged pools constrain geometry the flat fixture does not: head
    /// size >= 32 and a block size of 8/16/32. `paged_cache_memory_mb` keeps
    /// each group's Metal pool at ~8 MiB. The draft never reads target K/V
    /// (it cross-attends over fused residual hiddens), so the wider head is
    /// invisible to [`tiny_draft_config`].
    pub(crate) fn tiny_paged_target_config() -> Gemma4Config {
        let mut value = tiny_target_config_value();
        value["head_dim"] = serde_json::json!(32);
        value["use_block_paged_cache"] = serde_json::json!(true);
        value["paged_block_size"] = serde_json::json!(8);
        value["paged_cache_memory_mb"] = serde_json::json!(8);
        serde_json::from_value(value).expect("tiny paged Gemma4 config must deserialize")
    }

    /// A tiny PAGED target with the DSpark draft attached, or `None` when
    /// this build/host cannot back the paged pools (no Metal device) — the
    /// caller skips.
    pub(crate) fn tiny_paged_inner_with_draft() -> Option<Gemma4Inner> {
        let mut inner = Gemma4Inner::new(tiny_paged_target_config()).ok()?;
        inner.kv_cache_coordinator.as_ref()?;
        // The paged pools store 2-byte K/V; random-init weights are f32, so
        // the target must be cast before any row is written.
        crate::models::gemma4::model::tests::cast_paged_tiny_weights_to_bf16(&mut inner);
        let draft = DsparkDraftModel::new(tiny_draft_config()).ok()?;
        inner.draft = Some(Gemma4Draft::Dspark(draft));
        Some(inner)
    }

    /// Tiny draft config geometry-matched to [`tiny_target_config`]
    /// (hidden 8, vocab 16, 4 target layers, block_size 3).
    fn tiny_draft_config() -> DsparkConfig {
        serde_json::from_value(tiny_draft_config_value())
            .expect("tiny draft config must deserialize")
    }

    fn tiny_draft_config_value() -> serde_json::Value {
        serde_json::from_str(
            r#"{
                "architectures": ["Gemma4DSparkModel"],
                "model_type": "gemma4_text",
                "block_size": 3,
                "mask_token_id": 4,
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "global_head_dim": 4,
                "num_global_key_value_heads": 1,
                "rms_norm_eps": 1e-6,
                "final_logit_softcapping": 30.0,
                "vocab_size": 16,
                "target_layer_ids": [0, 2],
                "num_target_layers": 4,
                "markov_rank": 2,
                "markov_head_type": "vanilla",
                "enable_confidence_head": true,
                "attention_k_eq_v": true,
                "rope_parameters": {
                    "full_attention": {
                        "partial_rotary_factor": 0.5,
                        "rope_theta": 10000.0,
                        "rope_type": "proportional"
                    }
                }
            }"#,
        )
        .expect("tiny draft config JSON must parse")
    }

    pub(crate) fn tiny_inner_with_draft() -> Gemma4Inner {
        let mut inner =
            Gemma4Inner::new(tiny_target_config()).expect("tiny Gemma4Inner must construct");
        let draft =
            DsparkDraftModel::new(tiny_draft_config()).expect("tiny draft model must construct");
        inner.draft = Some(Gemma4Draft::Dspark(draft));
        inner
    }

    /// Tiny assistant draft config geometry-matched to
    /// [`tiny_target_config`]: backbone 8 / vocab 16 / head_dim 4 on both
    /// attention types (the tiny target sets no `global_head_dim`), one KV
    /// head each, `attention_k_eq_v` false (the target's serde default),
    /// window 8, and the target's default rope constants — so
    /// `AssistantConfig::validate(tiny_target_config())` passes.
    pub(crate) fn tiny_assistant_config() -> AssistantConfig {
        serde_json::from_str(
            r#"{
                "architectures": ["Gemma4UnifiedAssistantForCausalLM"],
                "model_type": "gemma4_unified_assistant",
                "backbone_hidden_size": 8,
                "use_ordered_embeddings": false,
                "tie_word_embeddings": true,
                "text_config": {
                    "hidden_size": 4,
                    "intermediate_size": 8,
                    "num_hidden_layers": 2,
                    "layer_types": ["sliding_attention", "full_attention"],
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "num_global_key_value_heads": 1,
                    "head_dim": 4,
                    "global_head_dim": null,
                    "attention_k_eq_v": false,
                    "sliding_window": 8,
                    "rms_norm_eps": 1e-6,
                    "vocab_size": 16,
                    "final_logit_softcapping": null,
                    "rope_parameters": {
                        "full_attention": {
                            "partial_rotary_factor": 0.25,
                            "rope_theta": 1000000.0,
                            "rope_type": "proportional"
                        },
                        "sliding_attention": {
                            "rope_theta": 10000.0,
                            "rope_type": "default"
                        }
                    }
                }
            }"#,
        )
        .expect("tiny assistant config must deserialize")
    }

    pub(crate) fn tiny_inner_with_assistant_draft() -> Gemma4Inner {
        let mut inner =
            Gemma4Inner::new(tiny_target_config()).expect("tiny Gemma4Inner must construct");
        let draft = AssistantDraftModel::new(tiny_assistant_config())
            .expect("tiny assistant draft model must construct");
        inner.draft = Some(Gemma4Draft::Assistant(draft));
        inner
    }

    pub(crate) fn chat_config(mtp_depth: Option<i32>) -> ChatConfig {
        ChatConfig {
            cache_salt: None,
            cache_owner_id: None,
            cache_root_owner_id: None,
            mtp_depth,
            ..ChatConfig::default()
        }
    }

    // ── resolve_params depth override ──────────────────────────────────

    /// With a draft loaded, an UNSET `mtpDepth` resolves to the draft's
    /// block_size (full blocks), bypassing the engine's default of 1.
    #[test]
    fn resolve_params_unset_depth_defaults_to_block_size() {
        let inner = tiny_inner_with_draft();
        let p = ChatBackend::resolve_params(&inner, &chat_config(None));
        assert_eq!(p.mtp_depth, 3, "unset depth must resolve to block_size");
        assert!(
            p.mtp_adaptive_depth,
            "an entirely unset DSpark config must enable measured AR fallback"
        );
    }

    /// An explicit depth is clamped to `[1, block_size]` from the RAW
    /// config value — the engine's central [1, 5] clamp must not cap a
    /// block_size wider than 5 (the real checkpoint's block_size is 7),
    /// and nonpositive values clamp up to 1 without wrapping.
    #[test]
    fn resolve_params_explicit_depth_clamps_to_block_size() {
        let inner = tiny_inner_with_draft();
        for (requested, expected) in [(1, 1), (2, 2), (3, 3), (99, 3), (0, 1), (-7, 1)] {
            let p = ChatBackend::resolve_params(&inner, &chat_config(Some(requested)));
            assert_eq!(
                p.mtp_depth, expected,
                "mtpDepth={requested} must resolve to {expected}"
            );
            assert!(
                !p.mtp_adaptive_depth,
                "an explicit mtpDepth pins DSpark unless adaptive mode is also explicit"
            );
        }
    }

    #[test]
    fn resolve_params_dspark_explicit_adaptive_override_wins() {
        let inner = tiny_inner_with_draft();

        let mut disabled = chat_config(None);
        disabled.mtp_adaptive_depth = Some(false);
        let p = ChatBackend::resolve_params(&inner, &disabled);
        assert!(!p.mtp_adaptive_depth);
        assert_eq!(p.mtp_depth, 3);

        let mut enabled_with_cap = chat_config(Some(2));
        enabled_with_cap.mtp_adaptive_depth = Some(true);
        let p = ChatBackend::resolve_params(&inner, &enabled_with_cap);
        assert!(p.mtp_adaptive_depth);
        assert_eq!(p.mtp_depth, 2);
    }

    /// Without a draft model the family override is inert: the engine's
    /// central [1, 5] clamp is untouched.
    #[test]
    fn resolve_params_without_draft_keeps_engine_clamp() {
        let inner = Gemma4Inner::new(tiny_target_config()).expect("tiny inner");
        let p = ChatBackend::resolve_params(&inner, &chat_config(None));
        assert_eq!(p.mtp_depth, 1);
        let p = ChatBackend::resolve_params(&inner, &chat_config(Some(99)));
        assert_eq!(p.mtp_depth, 5, "engine clamp caps at 5 without a draft");
    }

    /// The tiny assistant fixture must be a VALID pair with the tiny target
    /// — the assistant decode tests build on that geometry match.
    #[test]
    fn tiny_assistant_fixture_validates_against_tiny_target() {
        tiny_assistant_config()
            .validate(&tiny_target_config())
            .expect("tiny assistant draft must validate against the tiny target");
    }

    /// With an ASSISTANT draft loaded, an unset `mtpDepth` resolves to
    /// `ASSISTANT_DEFAULT_DEPTH` (no checkpoint-pinned block size).
    #[test]
    fn resolve_params_assistant_unset_depth_defaults() {
        let inner = tiny_inner_with_assistant_draft();
        let p = ChatBackend::resolve_params(&inner, &chat_config(None));
        assert_eq!(
            p.mtp_depth,
            crate::models::gemma4::assistant::ASSISTANT_DEFAULT_DEPTH,
            "unset depth must resolve to the assistant default (3)"
        );
        assert_eq!(p.mtp_depth, 3);
    }

    /// An explicit assistant depth is clamped to `[1, ASSISTANT_MAX_DEPTH]`
    /// from the RAW config value — wider than the engine's central [1, 5]
    /// clamp, and nonpositive values clamp up to 1 without wrapping.
    #[test]
    fn resolve_params_assistant_explicit_depth_clamps() {
        let inner = tiny_inner_with_assistant_draft();
        for (requested, expected) in [(1, 1), (8, 8), (99, 8), (0, 1), (-7, 1)] {
            let p = ChatBackend::resolve_params(&inner, &chat_config(Some(requested)));
            assert_eq!(
                p.mtp_depth, expected,
                "mtpDepth={requested} must resolve to {expected}"
            );
        }
    }

    // ── begin_dspark_decode plumbing ───────────────────────────────────

    /// The stepper derives its paged routing from the target config and its
    /// layer ids from the draft's `target_layer_ids`, adopts the model's
    /// active paged sequence, and starts with no open cycle; the per-turn
    /// context stash is TAKEN (single use), and calling begin without a
    /// stash is a hard error.
    #[test]
    fn begin_dspark_decode_takes_stash_and_derives_paged_state() {
        let Some(mut inner) = tiny_paged_inner_with_draft() else {
            eprintln!("skipping: this build cannot back the paged KV pools");
            return;
        };
        let num_draft_layers = inner
            .dspark_draft()
            .map(|d| d.num_layers())
            .expect("draft loaded");
        let num_target_layers = inner.config.num_hidden_layers as usize;
        inner.set_active_paged_owner(5);
        assert!(
            inner.begin_dspark_decode(3).is_err(),
            "begin without a prepared draft context must be a hard error"
        );

        inner.draft_turn_state = Some(Gemma4DraftTurnState::Dspark(DsparkTurnState {
            ctx: DsparkContextCache::new(num_draft_layers),
            next_pos: 7,
        }));
        {
            let stepper = match inner
                .begin_dspark_decode(3)
                .expect("begin with a stash must succeed")
            {
                Gemma4DraftStepper::Dspark(stepper) => stepper,
                Gemma4DraftStepper::Assistant(_) => {
                    panic!("a DSpark draft must yield the DSpark stepper")
                }
            };
            assert_eq!(stepper.layer_ids, vec![0, 2]);
            assert_eq!(
                stepper.layer_kinds.len(),
                num_target_layers,
                "the stepper resolves paged routing for every decoder layer once per turn"
            );
            assert_eq!(
                stepper.seq_id, 5,
                "the stepper must address the model's active paged sequence"
            );
            assert_eq!(stepper.next_pos, 7);
            assert!(!stepper.ar_fallback);
            assert!(
                stepper.open_cycle.is_none()
                    && stepper.tapped.is_none()
                    && !stepper.ar_probe_pending
            );
        }
        assert!(
            inner.draft_turn_state.is_none(),
            "the per-turn stash must be consumed by begin_dspark_decode"
        );
    }

    // ── paged speculative gates ────────────────────────────────────────

    /// PRIMARY ORACLE: at T=0 the paged DSpark turn must produce exactly the
    /// AR turn's tokens, and leave exactly the AR turn's paged rows.
    ///
    /// Both lanes run on identically seeded fixtures, so the only difference
    /// is the decoder. The second leg starts with a prompt LONGER than the
    /// sliding window, so every verify block lands on already-pruned sliding
    /// groups and the committed-basis prune runs between cycles.
    ///
    /// Mutation this catches: committing the whole verify block instead of
    /// the accepted prefix (`commit_cycle(.., ticket.rows())`), or settling
    /// at the write cursor instead of the committed frontier.
    #[test]
    fn paged_dspark_matches_paged_ar_at_greedy_temperature() {
        const SEED: u64 = 0x0D1_9A7E_0001;
        let Some(_probe) = seeded_tiny_paged_inner_with_draft(SEED) else {
            eprintln!("skipping: this build cannot back the paged KV pools");
            return;
        };
        let tokenizer = tiny_qwen_tokenizer();
        let window = 8usize;
        let epilogues_abandoned_before = crate::engine::spec_paged::abandoned_spec_turn_epilogues();

        for (label, prompt_len, budget) in [("sub_window", 4usize, 9i32), ("wrapped", 12, 24)] {
            let tokens: Vec<u32> = (0..prompt_len as u32).map(|i| i % 16).collect();
            let config = tiny_turn_config(None, budget);

            let mut ar_inner = seeded_tiny_paged_inner_with_draft(SEED).expect("seeded AR fixture");
            let ar_seq = ar_inner.active_paged_seq;
            let ar = run_tiny_paged_ar_turn(&mut ar_inner, &tokenizer, &tokens, &config)
                .expect("paged AR turn");
            let ar_rows = committed_paged_tokens(&ar_inner, ar_seq);

            let mut spec_inner =
                seeded_tiny_paged_inner_with_draft(SEED).expect("seeded speculative fixture");
            let spec_seq = spec_inner.active_paged_seq;
            let spec = run_tiny_paged_draft_turn(&mut spec_inner, &tokenizer, &tokens, &config)
                .expect("paged DSpark turn");
            let spec_rows = committed_paged_tokens(&spec_inner, spec_seq);

            assert_eq!(spec.raw_text, ar.raw_text, "[{label}] raw_text");
            assert_eq!(spec.num_tokens, ar.num_tokens, "[{label}] token count");
            assert_eq!(
                spec.finish_reason, ar.finish_reason,
                "[{label}] finish_reason"
            );
            assert_eq!(
                spec_inner.cached_token_history, ar_inner.cached_token_history,
                "[{label}] saved history"
            );
            assert_eq!(spec_rows, ar_rows, "[{label}] committed paged rows");
            assert_eq!(
                paged_free_blocks(&spec_inner),
                paged_free_blocks(&ar_inner),
                "[{label}] the post-commit settle must reclaim exactly the blocks the AR \
                 lane's per-step settle does"
            );
            assert_eq!(
                spec_rows.as_deref(),
                Some(spec_inner.cached_token_history.as_slice()),
                "[{label}] no drafted row may outlive its cycle"
            );

            let cycles = spec
                .performance
                .as_ref()
                .and_then(|p| p.mtp_cycles)
                .unwrap_or(0);
            assert!(
                cycles > 0,
                "[{label}] the speculative lane must actually run cycles (silent AR fallback?)"
            );
            assert_eq!(
                crate::engine::spec_paged::abandoned_spec_turn_epilogues(),
                epilogues_abandoned_before,
                "[{label}] a completed speculative turn must discharge its epilogue \
                 through SpecTurnEpilogue::finish (L-EPILOGUE)"
            );
            if label == "wrapped" {
                assert!(
                    prompt_len > window,
                    "the wrapped leg must start past the {window}-token sliding window"
                );
            }
        }
    }

    /// A stop INSIDE a speculative block registers only the tokens the turn
    /// emitted: the saved history drops the stop token (AR parity) and the
    /// paged rows equal that history exactly — the drafted rows past the cut
    /// were retracted by the cycle's commit.
    ///
    /// Mutation this catches: keeping the whole verify block on a stop cut
    /// (`keep = 1 + accepted_drafts_k` instead of the clamped count), which
    /// leaves rows past the saved history.
    #[test]
    fn paged_dspark_mid_cycle_stop_registers_only_emitted_tokens() {
        let Some(mut inner) = seeded_tiny_paged_inner_with_draft(0x5709_C107) else {
            eprintln!("skipping: this build cannot back the paged KV pools");
            return;
        };
        let tokenizer = tiny_qwen_tokenizer();
        let tokens: Vec<u32> = vec![0, 1, 2, 3];
        let seq_id = inner.active_paged_seq;

        // Every id in the tiny vocab is a stop token, so the FIRST emitted
        // token ends the turn — the earliest possible mid-block cut.
        let mut config = tiny_turn_config(None, 12);
        config.max_new_tokens = Some(12);
        let out = run_tiny_paged_turn(
            &mut inner, &tokenizer, &tokens, &config, true, 0, None, None,
        )
        .expect("stop-shaped speculative turn");
        let res = match out {
            TurnOutput::Complete(res) => *res,
            TurnOutput::Streamed => panic!("sync turn returned Streamed"),
        };
        assert_eq!(res.finish_reason, "stop", "the fixture must stop early");

        let history = inner.cached_token_history.clone();
        assert_eq!(
            history.len(),
            tokens.len() + res.num_tokens as usize - 1,
            "a stop exit drops the stop token from the saved history (AR parity)"
        );
        assert_eq!(
            committed_paged_tokens(&inner, seq_id).as_deref(),
            Some(history.as_slice()),
            "the committed paged rows must be exactly the saved history — no drafted \
             row past the stop may survive"
        );
    }

    /// REJECT-ALL: a cycle whose drafted rows are all rejected must leave
    /// the paged cache exactly as a cycle that drafted nothing does — same
    /// committed rows, same blocks — so no drafted row can reach the
    /// settle's cold-rung walk or the sliding prune.
    ///
    /// Both legs reserve the same lookahead, so the only difference is the
    /// three rows leg A writes and rolls back.
    ///
    /// Mutation this catches: committing `total_written` instead of `keep`
    /// (the rejected rows stay committed), or settling before the rollback
    /// (the checkpoint walk sees the write cursor).
    #[test]
    fn paged_dspark_reject_all_leaves_no_drafted_row_behind() {
        const SEED: u64 = 0x0D19_E7EC;
        const RESERVE_ROWS: usize = 3;

        fn cycle(verify_ids: &[u32]) -> Option<(Vec<u32>, Option<Vec<u32>>)> {
            let mut inner = seeded_tiny_paged_inner_with_draft(SEED)?;
            let tokenizer = tiny_qwen_tokenizer();
            let tokens: Vec<u32> = vec![0, 1, 2, 3];
            run_tiny_paged_draft_turn(&mut inner, &tokenizer, &tokens, &tiny_turn_config(None, 4))
                .expect("warm the sequence with a real speculative turn");
            let seq_id = inner.active_paged_seq;
            let warm = committed_paged_tokens(&inner, seq_id).expect("warm rows");
            let num_draft_layers = inner
                .dspark_draft()
                .map(DsparkDraftModel::num_layers)
                .expect("draft loaded");
            inner.draft_turn_state = Some(Gemma4DraftTurnState::Dspark(DsparkTurnState {
                ctx: DsparkContextCache::new(num_draft_layers),
                next_pos: i32::try_from(warm.len()).expect("warm rows fit i32"),
            }));
            {
                let Gemma4DraftStepper::Dspark(mut stepper) = inner
                    .begin_dspark_decode(2)
                    .expect("begin a synthetic cycle")
                else {
                    panic!("a DSpark draft must yield the DSpark stepper")
                };
                assert!(
                    stepper
                        .reserve_cycle_lookahead(RESERVE_ROWS)
                        .expect("reserve lookahead"),
                    "the tiny pool must admit {RESERVE_ROWS} lookahead rows"
                );
                stepper.verify(verify_ids).expect("verify block");
                // keep = 1: the anchor's own row only — every drafted row rejected.
                stepper.commit(1, verify_ids.len()).expect("commit");
            }
            let mut expected = warm;
            expected.push(verify_ids[0]);
            let rows = committed_paged_tokens(&inner, seq_id).expect("rows after the cycle");
            assert_eq!(
                rows, expected,
                "only the anchor's row may survive a cycle committed with keep = 1"
            );
            Some((rows, paged_free_blocks(&inner)))
        }

        const ANCHOR: u32 = 5;
        let Some((no_draft_rows, no_draft_blocks)) = cycle(&[ANCHOR]) else {
            eprintln!("skipping: this build cannot back the paged KV pools");
            return;
        };
        let (reject_all_rows, reject_all_blocks) = cycle(&[ANCHOR, 6, 7]).expect("reject-all leg");

        assert_eq!(
            reject_all_rows, no_draft_rows,
            "a reject-all cycle must commit exactly the rows a no-draft cycle commits"
        );
        assert_eq!(
            reject_all_blocks, no_draft_blocks,
            "a reject-all cycle must leave the same blocks a no-draft cycle does"
        );
    }

    // ── confidence threshold env ───────────────────────────────────────

    /// Threshold parsing: unset/invalid → 0.0 (keep-all). NOTE: reads the
    /// process env — no parallel test mutates this variable.
    #[test]
    fn confidence_threshold_env_parsing() {
        // SAFETY: test-local env mutation; no other test in this binary
        // touches MLX_DSPARK_CONFIDENCE_THRESHOLD.
        unsafe { std::env::remove_var("MLX_DSPARK_CONFIDENCE_THRESHOLD") };
        assert_eq!(dspark_confidence_threshold_from_env(), 0.0);
        unsafe { std::env::set_var("MLX_DSPARK_CONFIDENCE_THRESHOLD", "0.25") };
        assert_eq!(dspark_confidence_threshold_from_env(), 0.25);
        unsafe { std::env::set_var("MLX_DSPARK_CONFIDENCE_THRESHOLD", "not-a-number") };
        assert_eq!(dspark_confidence_threshold_from_env(), 0.0);
        unsafe { std::env::remove_var("MLX_DSPARK_CONFIDENCE_THRESHOLD") };
    }

    // ── fail-closed error path (whole-turn core) ───────────────────────

    /// WordLevel tokenizer covering the full tiny vocab (ids 0..16 as
    /// `t0`..`t15`) so every decode over tiny-model output succeeds,
    /// written to a temp `tokenizer.json` for `Qwen3Tokenizer::from_file`.
    pub(crate) fn tiny_qwen_tokenizer() -> Arc<crate::tokenizer::Qwen3Tokenizer> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "gemma4_dspark_tiny_tokenizer_{}_{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).expect("create temp tokenizer dir");
        let vocab = (0..16)
            .map(|i| format!("\"t{i}\": {i}"))
            .collect::<Vec<_>>()
            .join(", ");
        let json = format!(
            r#"{{
                "version": "1.0",
                "truncation": null,
                "padding": null,
                "added_tokens": [],
                "normalizer": null,
                "pre_tokenizer": null,
                "post_processor": null,
                "decoder": null,
                "model": {{
                    "type": "WordLevel",
                    "vocab": {{ {vocab} }},
                    "unk_token": "t0"
                }}
            }}"#
        );
        let path = dir.join("tokenizer.json");
        std::fs::write(&path, json).expect("write tiny tokenizer.json");
        Arc::new(
            crate::tokenizer::Qwen3Tokenizer::from_file(&path).expect("tiny tokenizer must load"),
        )
    }

    pub(crate) fn tiny_turn_config(mtp_depth: Option<i32>, max_new_tokens: i32) -> ChatConfig {
        ChatConfig {
            cache_salt: None,
            cache_owner_id: None,
            cache_root_owner_id: None,
            mtp_depth,
            // Whole-turn oracle/error tests below exercise the fixed-depth
            // speculative path; adaptive fallback has its own policy and
            // stepper tests.
            mtp_adaptive_depth: Some(false),
            max_new_tokens: Some(max_new_tokens),
            temperature: Some(0.0),
            reuse_cache: Some(true),
            report_performance: Some(true),
            include_reasoning: Some(false),
            ..ChatConfig::default()
        }
    }

    /// Drive one PAGED tiny turn through the real `ChatBackend` dispatch
    /// (sync unless a sink is supplied — no model thread), with an
    /// out-of-vocab `eos_id` so the tiny model can only exit "length".
    /// `speculative` picks the decoder the dispatch then routes on.
    pub(crate) fn run_tiny_paged_turn(
        inner: &mut Gemma4Inner,
        tokenizer: &Arc<crate::tokenizer::Qwen3Tokenizer>,
        tokens: &[u32],
        config: &ChatConfig,
        speculative: bool,
        eos_id: u32,
        sink: Option<&dyn crate::engine::backend::ChunkSink>,
        cancelled: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<TurnOutput> {
        let p = ChatBackend::resolve_params(inner, config);
        let thinking = ChatBackend::thinking_setup(inner, config);
        let mut args = WholeTurnArgs {
            tokens,
            tokenizer,
            eos_id,
            config,
            params: &p,
            thinking,
            plan: TurnPlan {
                is_delta: false,
                input_media: MediaCapabilities::NONE,
                context_media: MediaCapabilities::NONE,
                use_paged_attention: true,
                decoder: if speculative {
                    DecoderPlan::Speculative(SpeculativeKind::DraftModel)
                } else {
                    DecoderPlan::Autoregressive
                },
            },
            sink,
            cancelled,
            media: MediaInputs {
                images: &[],
                audio: &[],
            },
        };
        ChatBackend::run_paged_turn(inner, &mut args)
    }

    /// Drive `flat_draft_chat_turn` directly (sync — no model thread) for
    /// the FLAT lane the assistant drafter runs on, with an out-of-vocab
    /// `eos_id` so the tiny model can only exit "length".
    pub(crate) fn run_tiny_flat_draft_turn(
        inner: &mut Gemma4Inner,
        tokenizer: &Arc<crate::tokenizer::Qwen3Tokenizer>,
        tokens: &[u32],
        config: &ChatConfig,
    ) -> Result<crate::engine::types::ChatResult> {
        let p = ChatBackend::resolve_params(inner, config);
        let thinking = ChatBackend::thinking_setup(inner, config);
        let mut args = WholeTurnArgs {
            tokens,
            tokenizer,
            eos_id: 999,
            config,
            params: &p,
            thinking,
            plan: TurnPlan {
                is_delta: false,
                input_media: MediaCapabilities::NONE,
                context_media: MediaCapabilities::NONE,
                use_paged_attention: false,
                decoder: DecoderPlan::Speculative(SpeculativeKind::DraftModel),
            },
            sink: None,
            cancelled: None,
            media: MediaInputs {
                images: &[],
                audio: &[],
            },
        };
        match inner.flat_draft_chat_turn(&mut args)? {
            TurnOutput::Complete(r) => Ok(*r),
            TurnOutput::Streamed => panic!("sync draft turn returned TurnOutput::Streamed"),
        }
    }

    /// [`run_tiny_paged_turn`] on the speculative lane, sync.
    pub(crate) fn run_tiny_paged_draft_turn(
        inner: &mut Gemma4Inner,
        tokenizer: &Arc<crate::tokenizer::Qwen3Tokenizer>,
        tokens: &[u32],
        config: &ChatConfig,
    ) -> Result<crate::engine::types::ChatResult> {
        match run_tiny_paged_turn(inner, tokenizer, tokens, config, true, 999, None, None)? {
            TurnOutput::Complete(r) => Ok(*r),
            TurnOutput::Streamed => panic!("sync draft turn returned TurnOutput::Streamed"),
        }
    }

    /// The same turn on the plain paged AR lane — the T=0 oracle the
    /// speculative parity gates compare against.
    pub(crate) fn run_tiny_paged_ar_turn(
        inner: &mut Gemma4Inner,
        tokenizer: &Arc<crate::tokenizer::Qwen3Tokenizer>,
        tokens: &[u32],
        config: &ChatConfig,
    ) -> Result<crate::engine::types::ChatResult> {
        match run_tiny_paged_turn(inner, tokenizer, tokens, config, false, 999, None, None)? {
            TurnOutput::Complete(r) => Ok(*r),
            TurnOutput::Streamed => panic!("sync AR turn returned TurnOutput::Streamed"),
        }
    }

    /// Deterministic weights: MLX's global PRNG seeds `Linear::new`'s
    /// Xavier-uniform draw, so two fixtures built after the same seed carry
    /// byte-identical weights — the only way a two-instance A/B can compare
    /// trajectories.
    pub(crate) fn seeded_tiny_paged_inner_with_draft(seed: u64) -> Option<Gemma4Inner> {
        // SAFETY: test-local; the gemma4 tiny fixtures are the only users of
        // the global PRNG in this binary.
        unsafe { mlx_sys::mlx_seed(seed) };
        tiny_paged_inner_with_draft()
    }

    /// Per-group free-block counts — what the sliding prune half of the
    /// settle actually moves.
    pub(crate) fn paged_free_blocks(inner: &Gemma4Inner) -> Option<Vec<u32>> {
        let coordinator = inner.kv_cache_coordinator.as_ref()?;
        let mut group_ids: Vec<usize> = coordinator
            .routes()
            .iter()
            .map(|route| route.group_id)
            .collect();
        group_ids.sort_unstable();
        group_ids.dedup();
        group_ids
            .into_iter()
            .map(|group_id| {
                coordinator
                    .adapter(group_id)
                    .ok()?
                    .block_telemetry()
                    .ok()
                    .map(|telemetry| telemetry.free_blocks)
            })
            .collect()
    }

    /// The paged rows sequence `seq_id` has COMMITTED, or `None` when no
    /// live request names it.
    pub(crate) fn committed_paged_tokens(inner: &Gemma4Inner, seq_id: u32) -> Option<Vec<u32>> {
        inner
            .kv_cache_coordinator
            .as_ref()?
            .full_adapter()
            .request_tokens_for(seq_id)
            .map(<[u32]>::to_vec)
    }

    /// FAIL-CLOSED regression: a REAL (unmocked) error AFTER the prefill has
    /// advanced the paged cursor must release the request and drop the whole
    /// warm session — history, media keys, per-turn stash — so no later turn
    /// can prefix-match into rows the history knows nothing about; and the
    /// very next turn must succeed via the cold path.
    ///
    /// Error injection: `target_layer_ids` in DESCENDING order. The tap
    /// validator (`validate_paged_tap_layer_ids`) requires strictly
    /// ascending decoder indices and rejects the very first tapped chunk —
    /// after `record_tokens_all` advanced every group's cursor for it.
    #[test]
    fn dspark_paged_turn_error_fails_closed_then_cold_turn_recovers() {
        let Some(mut inner) = seeded_tiny_paged_inner_with_draft(0xD1_FA11_0001) else {
            eprintln!("skipping: this build cannot back the paged KV pools");
            return;
        };
        let mut broken = tiny_draft_config_value();
        broken["target_layer_ids"] = serde_json::json!([2, 0]);
        inner.draft = Some(Gemma4Draft::Dspark(
            DsparkDraftModel::new(
                serde_json::from_value(broken).expect("descending-tap draft config"),
            )
            .expect("tiny draft model"),
        ));
        let tokenizer = tiny_qwen_tokenizer();
        let tokens: Vec<u32> = vec![0, 1, 2, 3];
        let seq_id = inner.active_paged_seq;
        let epilogues_abandoned_before = crate::engine::spec_paged::abandoned_spec_turn_epilogues();

        let err =
            run_tiny_paged_draft_turn(&mut inner, &tokenizer, &tokens, &tiny_turn_config(None, 8))
                .expect_err("descending tap layer ids must be rejected by the paged layer loop");
        assert!(
            err.reason.contains("layer_ids"),
            "expected the paged tap layer-id guard, got: {}",
            err.reason
        );

        assert_eq!(
            crate::engine::spec_paged::abandoned_spec_turn_epilogues(),
            epilogues_abandoned_before,
            "a failed speculative turn must discharge its epilogue through \
             SpecTurnEpilogue::abort, not abandon it (L-EPILOGUE)"
        );

        // Fail CLOSED: nothing warm-reusable may survive the error.
        assert!(
            inner.cached_token_history.is_empty(),
            "cached_token_history must be cleared (it never covered the prefilled rows)"
        );
        assert!(
            inner.draft_turn_state.is_none(),
            "the per-turn draft stash must be cleared"
        );
        assert!(
            !ChatBackend::has_live_session(&inner),
            "the session must not be warm-reusable after a failed turn"
        );
        assert_eq!(
            ChatBackend::verify_cache_prefix(&inner, &tokens, true),
            0,
            "no prefix hit may match against the released request"
        );

        // Turn 2 on a sound draft: the turn must run cold end-to-end and land
        // fully consistent.
        inner.draft = Some(Gemma4Draft::Dspark(
            DsparkDraftModel::new(tiny_draft_config()).expect("tiny draft model"),
        ));
        let mut recovery_config = tiny_turn_config(Some(1), 3);
        recovery_config.reasoning_effort = Some("high".to_string());
        let res = run_tiny_paged_draft_turn(&mut inner, &tokenizer, &tokens, &recovery_config)
            .expect("the next turn after fail-closed must succeed via the cold path");
        assert_eq!(res.finish_reason, "length");
        assert!(
            res.thinking_enabled,
            "sync DSpark result must report effective template thinking provenance"
        );
        assert_eq!(
            res.cached_tokens, 0,
            "nothing may be warm-reused after fail-closed"
        );
        assert_eq!(res.num_tokens, 3, "budget-3 length exit");

        // Length-exit AR parity (keep-all + materialize): the saved history
        // holds prompt + ALL generated tokens, and the paged rows ARE that
        // history — no speculative row outlived its cycle.
        let history = inner.cached_token_history.clone();
        assert_eq!(history.len(), tokens.len() + res.num_tokens as usize);
        assert_eq!(&history[..tokens.len()], &tokens[..]);
        assert_eq!(
            committed_paged_tokens(&inner, seq_id).as_deref(),
            Some(history.as_slice()),
            "the committed paged rows must be exactly the saved history"
        );
    }

    // ── streaming cancellation (whole-turn) ────────────────────────────

    /// Records every chunk and flips the shared cancel flag once
    /// `flip_after` NON-TERMINAL chunks have arrived (the sink runs inline
    /// on the decode thread, so the flip lands mid-turn deterministically).
    pub(crate) struct CancelAfterSink {
        pub(crate) chunks: std::sync::Mutex<Vec<crate::engine::types::ChatStreamChunk>>,
        pub(crate) cancelled: Arc<std::sync::atomic::AtomicBool>,
        pub(crate) flip_after: usize,
    }

    impl crate::engine::backend::ChunkSink for CancelAfterSink {
        fn send(&self, chunk: Result<crate::engine::types::ChatStreamChunk>) {
            if let (Ok(c), Ok(mut v)) = (chunk, self.chunks.lock()) {
                v.push(c);
                if v.iter().filter(|c| !c.done).count() >= self.flip_after {
                    self.cancelled
                        .store(true, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }
    }

    /// WHOLE-TURN streaming cancellation through `run_paged_dspark_turn`: a
    /// cancel raised from the chunk sink must terminate the stream promptly
    /// ("cancelled", bounded block-granular overrun — never running on to
    /// the budget) and leave the paged session consistent (AR-parity
    /// drop-last: the final emitted token is persisted in NEITHER the
    /// history NOR the pool), with the next turn running normally.
    /// Chunk-vs-residual byte accounting for the cancel suffix is pinned at
    /// the engine seam
    /// (`dspark_turn_streaming_cancel_in_clamp_commits_exactly_once`), where
    /// the mid-clamp cancel point is injectable; a sink-driven flip lands at
    /// the next loop-top by construction.
    #[test]
    fn dspark_streaming_cancel_whole_turn_state_consistent() {
        let Some(mut inner) = seeded_tiny_paged_inner_with_draft(0xD5_9A4B_0001) else {
            eprintln!("skipping: this build cannot back the paged KV pools");
            return;
        };
        let tokenizer = tiny_qwen_tokenizer();
        let tokens: Vec<u32> = vec![0, 1, 2, 3];
        let seq_id = inner.active_paged_seq;
        // Budget 12 with a flip after 2 chunks: a broken cancel would run to
        // the length exit and emit ~12 chunks.
        let mut config = tiny_turn_config(Some(1), 12);
        config.include_reasoning = Some(true);

        let cancelled = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let sink = CancelAfterSink {
            chunks: std::sync::Mutex::new(Vec::new()),
            cancelled: Arc::clone(&cancelled),
            flip_after: 2,
        };
        let out = run_tiny_paged_turn(
            &mut inner,
            &tokenizer,
            &tokens,
            &config,
            true,
            999,
            Some(&sink),
            Some(&cancelled),
        )
        .expect("streaming cancelled turn must complete cleanly");
        assert!(
            matches!(out, TurnOutput::Streamed),
            "streaming turn must return TurnOutput::Streamed"
        );

        let chunks = sink.chunks.into_inner().expect("sink poisoned");
        let terminal = chunks
            .iter()
            .find(|c| c.done)
            .expect("stream must end with a terminal done-chunk");
        assert_eq!(
            terminal.finish_reason.as_deref(),
            Some("cancelled"),
            "sink-raised cancel must finish the turn as cancelled"
        );
        let n = terminal.num_tokens.expect("terminal must carry num_tokens") as usize;
        assert!(
            (1..=5).contains(&n),
            "cancel must stop within one depth-1 cycle of the flip \
             (seed + <= 2 cycles of <= 2 tokens), got {n} generated tokens"
        );

        // AR-parity cancelled save: the final emitted token is dropped from
        // the history AND never survives in the pool.
        let history = inner.cached_token_history.clone();
        assert_eq!(
            history.len(),
            tokens.len() + n - 1,
            "cancelled turn must persist prompt + generated minus the final token"
        );
        assert_eq!(&history[..tokens.len()], &tokens[..]);
        assert_eq!(
            committed_paged_tokens(&inner, seq_id).as_deref(),
            Some(history.as_slice()),
            "the committed paged rows must be exactly the saved history"
        );
        // Warm-reusability after a cancel is a PAGED-LANE property, not a
        // speculative one, so it is asserted against the plain paged AR lane
        // on an identically seeded fixture rather than against a constant —
        // a control that moves with the lane instead of rotting beside it.
        let ar_live = {
            let mut ar_inner =
                seeded_tiny_paged_inner_with_draft(0xD5_9A4B_0001).expect("seeded AR fixture");
            let ar_cancelled = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let ar_sink = CancelAfterSink {
                chunks: std::sync::Mutex::new(Vec::new()),
                cancelled: Arc::clone(&ar_cancelled),
                flip_after: 2,
            };
            run_tiny_paged_turn(
                &mut ar_inner,
                &tokenizer,
                &tokens,
                &config,
                false,
                999,
                Some(&ar_sink),
                Some(&ar_cancelled),
            )
            .expect("paged AR cancelled turn must complete cleanly");
            ChatBackend::has_live_session(&ar_inner)
        };
        assert_eq!(
            ChatBackend::has_live_session(&inner),
            ar_live,
            "a cancelled speculative turn must leave the session exactly as warm \
             (or as cold) as a cancelled AR turn does"
        );

        // The next turn runs normally (fresh prompt: the longer saved
        // history is a prefix-miss, so it takes the cold path).
        let res = run_tiny_paged_draft_turn(
            &mut inner,
            &tokenizer,
            &tokens,
            &tiny_turn_config(Some(1), 3),
        )
        .expect("the turn after a cancelled stream must succeed");
        assert_eq!(res.finish_reason, "length");
        assert_eq!(res.num_tokens, 3);
    }

    // ── EOS-accepted-as-draft AR state parity (real model, env-gated) ──

    /// EOS-ACCEPTED-AS-DRAFT regression: full post-turn STATE parity vs the
    /// AR flow — `cached_token_history` byte-equal AND physical cache
    /// offsets equal to the history length — across a 2-turn warm-continue,
    /// with the stop SHAPE (EOS cut INSIDE the accepted drafts, not at a
    /// cycle boundary) pinned via the mtp acceptance stats: on a boundary
    /// stop or clean cycles, generated == seed + Σk + cycles; a cut inside
    /// accepted drafts loses the cut cycle's boundary token (and any drafts
    /// past the EOS), so generated < seed + Σk + cycles.
    ///
    /// Run (single-threaded; both env vars required):
    ///
    /// ```shell
    /// PATH=/usr/bin:$PATH SDKROOT=$(xcrun --show-sdk-path) \
    /// MLX_TEST_GEMMA4_MODEL_PATH=... MLX_TEST_GEMMA4_DSPARK_PATH=... \
    ///     cargo test -p mlx-core --lib --release -- --ignored \
    ///     --test-threads=1 dspark_eos_accepted_draft_state_matches_ar_e2e
    /// ```
    #[test]
    #[ignore = "needs MLX_TEST_GEMMA4_MODEL_PATH + MLX_TEST_GEMMA4_DSPARK_PATH (real 12B + draft)"]
    fn dspark_eos_accepted_draft_state_matches_ar_e2e() {
        let (Ok(model_path), Ok(draft_path)) = (
            std::env::var("MLX_TEST_GEMMA4_MODEL_PATH"),
            std::env::var("MLX_TEST_GEMMA4_DSPARK_PATH"),
        ) else {
            eprintln!("skipping: set MLX_TEST_GEMMA4_MODEL_PATH + MLX_TEST_GEMMA4_DSPARK_PATH");
            return;
        };

        // Tie-screened fixture (see tests/gemma4_dspark.rs module doc);
        // measured shape on this checkpoint: the EOS is cut inside accepted
        // drafts on turn 1 (deficit >= 1 below).
        const PROMPT: &str = "What is the capital of France? Answer with just the city name.";
        const FOLLOW_UP: &str = "And of Italy? Same format.";

        fn cfg(enable_mtp: bool) -> ChatConfig {
            ChatConfig {
                cache_salt: None,
                cache_owner_id: None,
                cache_root_owner_id: None,
                max_new_tokens: Some(64),
                temperature: Some(0.0),
                include_reasoning: Some(false),
                report_performance: Some(true),
                reuse_cache: Some(true),
                enable_mtp: Some(enable_mtp),
                mtp_adaptive_depth: Some(false),
                ..ChatConfig::default()
            }
        }
        fn user(content: &str) -> crate::tokenizer::ChatMessage {
            crate::tokenizer::ChatMessage {
                role: "user".to_string(),
                content: content.to_string(),
                tool_calls: None,
                tool_call_id: None,
                is_error: None,
                reasoning_content: None,
                thinking_enabled: None,
                images: None,
                audio: None,
            }
        }
        fn assistant(result: &crate::engine::types::ChatResult) -> crate::tokenizer::ChatMessage {
            crate::tokenizer::ChatMessage {
                role: "assistant".to_string(),
                content: result.text.clone(),
                tool_calls: None,
                tool_call_id: None,
                is_error: None,
                reasoning_content: result.thinking.clone(),
                thinking_enabled: Some(result.thinking_enabled),
                images: None,
                audio: None,
            }
        }
        fn assert_offsets_match_history(inner: &Gemma4Inner, label: &str) {
            let h = inner.cached_token_history.len() as i32;
            let mask = dspark_shared_slot_mask(&inner.config);
            let caches = inner.caches.as_ref().expect("live caches");
            assert!(h > 0, "[{label}] saved history must be non-empty");
            for (i, cache) in caches.iter().enumerate() {
                let expected = if mask[i] { 0 } else { h };
                assert_eq!(
                    cache.get_offset(),
                    expected,
                    "[{label}] cache {i} physical offset diverged from the {h}-token history"
                );
            }
        }

        // ONE instance for both passes (the draft never touches the flat AR
        // path, and a fresh session start resets the prior session).
        let (mut inner, _weight_bytes) = Gemma4Inner::load_from_dir(&model_path, Some(&draft_path))
            .expect("12B + draft load failed");

        // Never-flipped whole-turn cancel flag — the sync session cores
        // now take one (H2); these turns are never cancelled.
        let no_cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

        // --- AR baseline: 2 turns, capturing history + offsets ---
        let ar1 = crate::engine::session::session_start(
            &mut inner,
            vec![user(PROMPT)],
            cfg(false),
            &no_cancel,
        )
        .expect("AR turn 1 failed");
        assert_eq!(ar1.finish_reason, "stop", "fixture must stop early on EOS");
        let ar_h1 = inner.cached_token_history.clone();
        assert_offsets_match_history(&inner, "ar_turn1");
        let ar2 = crate::engine::session::session_continue(
            &mut inner,
            vec![user(PROMPT), assistant(&ar1), user(FOLLOW_UP)],
            cfg(false),
            &no_cancel,
        )
        .expect("AR turn 2 failed");
        let ar_h2 = inner.cached_token_history.clone();
        assert_offsets_match_history(&inner, "ar_turn2");

        // --- DSpark pass: same 2 turns ---
        let sp1 = crate::engine::session::session_start(
            &mut inner,
            vec![user(PROMPT)],
            cfg(true),
            &no_cancel,
        )
        .expect("DSpark turn 1 failed");
        assert_eq!(sp1.finish_reason, "stop");
        // SHAPE fingerprint: the EOS must have been accepted as a DRAFT.
        let perf = sp1.performance.as_ref().expect("DSpark perf missing");
        let cycles = perf.mtp_cycles.expect("mtp_cycles missing") as i64;
        let mean_k = perf
            .mtp_mean_accepted_tokens
            .expect("mtp_mean_accepted_tokens missing");
        let total_k = (mean_k * cycles as f64).round() as i64;
        let full_emission = 1 + total_k + cycles;
        assert!(cycles > 0, "DSpark cycles must have run");
        assert!(
            (sp1.num_tokens as i64) < full_emission,
            "fixture no longer stops INSIDE accepted drafts (generated {} == seed + \u{03a3}k + \
             cycles = {full_emission}; the EOS landed on a cycle boundary) — re-screen the \
             prompt so the accepted-draft-EOS shape is actually exercised",
            sp1.num_tokens,
        );
        let sp_h1 = inner.cached_token_history.clone();
        assert_offsets_match_history(&inner, "dspark_turn1");

        let sp2 = crate::engine::session::session_continue(
            &mut inner,
            vec![user(PROMPT), assistant(&sp1), user(FOLLOW_UP)],
            cfg(true),
            &no_cancel,
        )
        .expect("DSpark turn 2 failed");
        assert!(
            sp2.cached_tokens > 0,
            "turn 2 must warm-continue on the saved session, got cached_tokens=0"
        );
        assert!(
            sp2.performance
                .as_ref()
                .and_then(|p| p.mtp_cycles)
                .unwrap_or(0)
                > 0,
            "the warm-continue turn must also run DSpark cycles"
        );
        let sp_h2 = inner.cached_token_history.clone();
        assert_offsets_match_history(&inner, "dspark_turn2");

        // --- Parity: transcript AND full logical/physical session state ---
        assert_eq!(sp1.text, ar1.text, "turn 1 text diverged from AR");
        assert_eq!(sp1.raw_text, ar1.raw_text, "turn 1 raw_text diverged");
        assert_eq!(sp1.num_tokens, ar1.num_tokens);
        assert_eq!(sp2.text, ar2.text, "turn 2 text diverged from AR");
        assert_eq!(sp2.raw_text, ar2.raw_text, "turn 2 raw_text diverged");
        assert_eq!(sp2.finish_reason, ar2.finish_reason);
        assert_eq!(sp2.num_tokens, ar2.num_tokens);
        assert_eq!(
            sp_h1, ar_h1,
            "post-turn-1 cached_token_history diverged from AR \
             (the accepted-draft EOS must be dropped from the persisted state)"
        );
        assert_eq!(
            sp_h2, ar_h2,
            "post-turn-2 cached_token_history diverged from AR"
        );
        println!(
            "[eos_accepted_draft_state] turn1: tokens={} cycles={cycles} \u{03a3}k={total_k} \
             deficit={} | turn2: tokens={} cached={} | history lens: {} / {}",
            sp1.num_tokens,
            full_emission - sp1.num_tokens as i64,
            sp2.num_tokens,
            sp2.cached_tokens,
            sp_h1.len(),
            sp_h2.len(),
        );
    }
}
