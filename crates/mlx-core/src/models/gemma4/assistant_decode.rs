//! Gemma4 assistant-draft speculative-decode wiring: the family-side
//! [`DsparkStepper`] implementation for the Google assistant draft, plus the
//! assistant prefill that seeds the per-turn state.
//!
//! Split of responsibilities (the assistant mirror of
//! [`super::dspark_decode`]):
//!   * the DRAFT model (Q-only transformer over the target's shared KV,
//!     backbone pre/post projections, tied lm_head) lives in
//!     [`super::assistant`];
//!   * the TARGET-side primitives (verify forward with exposed hidden,
//!     snapshot/commit rollback, KV-source mapping, shared-slot mask) live
//!     in [`super::model`] / [`super::layer_cache`];
//!   * the model-agnostic propose → verify → accept → stop-clamp → commit
//!     loop lives in [`crate::engine::dspark_turn`];
//!   * variant dispatch (`Gemma4DraftStepper`, `begin_dspark_decode`, the
//!     whole-turn core) lives in [`super::dspark_decode`];
//!   * THIS module implements the assistant stepper and prefill.
//!
//! Drafting per round follows mlx-vlm's `_mtp_rounds` semantics: chained
//! single-token AR steps, each attending the target's COMMITTED K/V with the
//! query RoPE'd at the round-constant position of the last committed token,
//! chaining the draft's own `h_prev` output locally. Round 1's `h_prev` is
//! the target's post-final-norm hidden of the last prompt token; after every
//! commit it becomes the verify hidden at the last kept slot.

use napi::bindgen_prelude::*;

use crate::array::{DType, MxArray};
use crate::engine::backend::{DsparkProposal, DsparkStepper, DsparkVerifyOutput};
use crate::engine::params::ChatParams;
use crate::sampling;
use crate::stream::{Stream, StreamContext};

use super::assistant::AssistantSharedKv;
use super::dspark::sample_index_from_probs;
use super::layer_cache::{Gemma4VerifyRollback, commit_after_verify, snapshot_before_verify};
use super::model::{
    AssistantKvSources, GEMMA4_PREFILL_STEP_SIZE, Gemma4Inner, assistant_verify_forward,
    compute_ple, eval_gemma4_caches, forward_body, lm_head_logits,
};

/// Assistant payload of [`super::dspark_decode::Gemma4DraftTurnState`]: the
/// per-turn handoff from `assistant_prefill_with_hidden` to
/// [`crate::engine::backend::DsparkBackend::begin_dspark_decode`].
pub(crate) struct AssistantTurnState {
    /// The target's post-final-norm hidden of the LAST PROMPT token,
    /// `[1, 1, backbone_hidden_size]` — round 1's chained `h_prev` seed
    /// (mlx-vlm's deliberate deviation from HF, which seeds from the
    /// pre-norm hidden).
    pub(crate) h_prev: MxArray,
    /// Absolute sequence position of the NEXT target-cache slot —
    /// `cached_prefix_len + prefill_len` right after prefill, then advanced
    /// by `keep` on every commit.
    pub(crate) next_pos: i32,
}

/// Per-turn gemma4 assistant stepper (the assistant arm of
/// [`super::dspark_decode::Gemma4DraftStepper`]).
///
/// Borrows the model for the whole decode loop. The verify hidden and the
/// rollback are stashed between `verify` and `commit` — they never cross the
/// engine trait (see the invariant on [`DsparkStepper`]).
pub(crate) struct Gemma4AssistantStepper<'a> {
    inner: &'a mut Gemma4Inner,
    /// Chained backbone hidden `[1, 1, B]` conditioning the next drafting
    /// round — the target's post-final-norm hidden at the LAST COMMITTED
    /// slot. Advanced ONLY by `commit` (propose chains a local copy).
    h_prev: MxArray,
    /// Absolute position of the next verify block's anchor (== the current
    /// committed sequence length: prompt + anchor-exclusive generation).
    next_pos: i32,
    /// Target-layer indices whose caches hold the draft's shared K/V, one
    /// per attention type ([`super::model::assistant_kv_source_indices`]).
    kv_sources: AssistantKvSources,
    /// Config-derived KV-shared slot mask
    /// ([`super::model::dspark_shared_slot_mask`]), passed to every
    /// `snapshot_before_verify`.
    shared_slots: Vec<bool>,
    /// Pending rollback from the last `verify`, consumed by `commit`.
    rollback: Option<Gemma4VerifyRollback>,
    /// `[1, 1+L, B]` post-final-norm hidden from the last `verify`,
    /// consumed by `commit`.
    verify_hidden: Option<MxArray>,
}

impl<'a> Gemma4AssistantStepper<'a> {
    /// Assemble the per-turn stepper from the prefill-derived state plus the
    /// config-derived KV routing (computed by `begin_dspark_decode`).
    pub(crate) fn from_turn_state(
        inner: &'a mut Gemma4Inner,
        state: AssistantTurnState,
        kv_sources: AssistantKvSources,
        shared_slots: Vec<bool>,
    ) -> Self {
        Self {
            inner,
            h_prev: state.h_prev,
            next_pos: state.next_pos,
            kv_sources,
            shared_slots,
            rollback: None,
            verify_hidden: None,
        }
    }
}

impl DsparkStepper for Gemma4AssistantStepper<'_> {
    fn propose(
        &mut self,
        anchor_id: u32,
        max_len: usize,
        params: &ChatParams,
        rng: &mut dyn rand::Rng,
    ) -> Result<DsparkProposal> {
        let draft = self
            .inner
            .assistant_draft()
            .ok_or_else(|| Error::from_reason("gemma4 assistant propose: no draft model loaded"))?;
        if max_len == 0 {
            return Err(Error::from_reason(
                "gemma4 assistant propose: engine contract violation (max_len == 0 cycles skip propose)",
            ));
        }
        let vocab = draft.config.text_config.vocab_size;
        if (anchor_id as i64) >= vocab {
            return Err(Error::from_reason(format!(
                "gemma4 assistant propose: anchor token {anchor_id} out of vocab range [0, {vocab})"
            )));
        }

        // Shared K/V handles, re-read from the target caches at the START of
        // every propose — always after the previous cycle's commit, so the
        // contents are exactly the committed positions. Never cached across
        // cycles.
        let kv = {
            let caches =
                self.inner.caches.as_ref().ok_or_else(|| {
                    Error::from_reason("gemma4 assistant propose: caches missing")
                })?;
            let fetch = |idx: usize, label: &str| -> Result<(MxArray, MxArray)> {
                caches
                    .get(idx)
                    .ok_or_else(|| {
                        Error::from_reason(format!(
                            "gemma4 assistant propose: {label} KV source layer {idx} out of range \
                             ({} caches)",
                            caches.len()
                        ))
                    })?
                    .get_cached_kv()
                    .ok_or_else(|| {
                        Error::from_reason(format!(
                            "gemma4 assistant propose: {label} KV source layer {idx} has no \
                             committed K/V (empty cache)"
                        ))
                    })
            };
            AssistantSharedKv {
                sliding: fetch(self.kv_sources.sliding, "sliding")?,
                full: fetch(self.kv_sources.full, "full")?,
            }
        };

        // Round-constant query position: the absolute position of the LAST
        // COMMITTED token, computed ONCE and held fixed across all chained
        // steps of this round.
        let q_pos = self.next_pos - 1;

        // Greedy detection uses the engine's `sampling::is_greedy_temperature`
        // predicate — the same predicate `run_dspark_turn` keys its accept
        // policy on — so the returned `dists` are empty exactly when the
        // engine expects them empty, and at sampled temperature each row is
        // the EXACT distribution the draw came from.
        let cfg = params.sampling_config.unwrap_or_default();
        let greedy = sampling::is_greedy_temperature(cfg.temperature.unwrap_or(1.0));

        // Chained single-token AR drafting. `h_prev` chains LOCALLY — the
        // stepper's field is advanced only by `commit` — and no target state
        // (caches, cursor) is touched, so propose is repeatable.
        let embed_scale = (self.inner.config.hidden_size as f64).sqrt();
        let mut draft_ids: Vec<i32> = Vec::with_capacity(max_len);
        let mut draft_dists: Vec<MxArray> = Vec::new();
        let mut h_prev = self.h_prev.clone();
        let mut prev_token = anchor_id as i32;
        for _ in 0..max_len {
            // Token embedding from the TARGET's table, scaled exactly like
            // the target's own `forward_body` step 1 (handles
            // packed-quantized tables), cast to the chained hidden's dtype
            // when the two differ.
            let ids = MxArray::from_int32(&[prev_token], &[1, 1])?;
            let token_embed = {
                let raw = self
                    .inner
                    .embed_tokens
                    .forward(&ids)?
                    .mul_scalar(embed_scale)?;
                let want = h_prev.dtype()?;
                if raw.dtype()? == want {
                    raw
                } else {
                    raw.astype(want)?
                }
            };
            let step = draft.forward_step(&token_embed, &h_prev, &kv, q_pos)?;
            let step_logits = step.logits.reshape(&[vocab])?;
            let token = if greedy {
                let idx = step_logits.argmax(0, Some(false))?.astype(DType::Int32)?;
                idx.eval();
                idx.item_at_int32(0)?
            } else {
                let dist = sampling::sampling_distribution(&step_logits, Some(cfg))?
                    .astype(DType::Float32)?;
                dist.eval();
                let probs = dist.to_float32()?;
                let token = sample_index_from_probs(&probs, rng)?;
                draft_dists.push(dist);
                token
            };
            draft_ids.push(token);
            prev_token = token;
            h_prev = step.h_prev_next;
        }

        Ok(DsparkProposal {
            draft_ids,
            draft_dists,
        })
    }

    fn verify(&mut self, verify_ids: &[u32]) -> Result<DsparkVerifyOutput> {
        if verify_ids.is_empty() {
            return Err(Error::from_reason(
                "gemma4 assistant verify: empty verify block",
            ));
        }
        // Commit-exactly-once defense: a second verify before the previous
        // cycle's commit would orphan its rollback (the caches would then
        // hold TWO uncommitted verify blocks).
        if self.rollback.is_some() || self.verify_hidden.is_some() {
            return Err(Error::from_reason(
                "gemma4 assistant verify: previous verify was never committed",
            ));
        }

        let inner = &mut *self.inner;
        let caches = inner
            .caches
            .as_mut()
            .ok_or_else(|| Error::from_reason("gemma4 assistant verify: caches missing"))?;
        let rb = snapshot_before_verify(caches, verify_ids.len(), &self.shared_slots)?;

        let ids_i32: Vec<i32> = verify_ids.iter().map(|&t| t as i32).collect();
        let block = MxArray::from_int32(&ids_i32, &[1, verify_ids.len() as i64])?;
        let (logits, hidden) = assistant_verify_forward(
            &block,
            &inner.embed_tokens,
            &inner.layers,
            caches,
            &inner.final_norm,
            &inner.lm_head,
            inner.embed_weight_t.as_ref(),
            inner.ple.as_ref(),
            &inner.config,
        )?;

        self.rollback = Some(rb);
        self.verify_hidden = Some(hidden);
        Ok(DsparkVerifyOutput { logits })
    }

    fn commit(&mut self, keep: usize, total_written: usize) -> Result<()> {
        let rb = self.rollback.take().ok_or_else(|| {
            Error::from_reason("gemma4 assistant commit: no pending verify rollback")
        })?;
        let hidden = self.verify_hidden.take().ok_or_else(|| {
            Error::from_reason("gemma4 assistant commit: no stashed verify hidden")
        })?;
        if keep == 0 {
            return Err(Error::from_reason(
                "gemma4 assistant commit: engine contract violation (keep must be >= 1 — the anchor's slot is unconditionally kept)",
            ));
        }
        if hidden.shape_at(1)? != total_written as i64 {
            return Err(Error::from_reason(format!(
                "gemma4 assistant commit: stashed verify hidden covers {} positions but the engine reports a {}-token verify block",
                hidden.shape_at(1)?,
                total_written
            )));
        }

        // Target side: keep the first `keep` of the verify block's K/V
        // slots, roll back the rest (validated against the shared-slot mask
        // on every commit, full keep included). `commit_after_verify` also
        // rejects `keep > total_written`, so the slice below is in range.
        {
            let caches = self
                .inner
                .caches
                .as_mut()
                .ok_or_else(|| Error::from_reason("gemma4 assistant commit: caches missing"))?;
            commit_after_verify(caches, &rb, keep)?;
        }

        // Draft side: the next round chains from the verify hidden at the
        // LAST KEPT slot; advance the cursor by the kept prefix. The
        // boundary token has no slot — it re-enters as the next cycle's
        // verify anchor.
        self.h_prev = hidden.slice_axis(1, keep as i64 - 1, keep as i64)?;
        self.next_pos += keep as i32;
        Ok(())
    }

    fn eval_boundary(&self, token: &MxArray) {
        // Schedule-only async eval of the next cycle's anchor (gemma4's
        // decode eval pattern: token only, never the logits).
        MxArray::async_eval_arrays(&[token]);
    }
}

impl Gemma4Inner {
    /// Chunked prefill for the assistant draft: target-side compute
    /// byte-identical to the AR path, plus the LAST prompt token's
    /// post-final-norm hidden.
    ///
    /// Mirrors `dspark_prefill_with_tap` MINUS all tap/fuse/context work
    /// (same upfront embedding/PLE, same 512-token chunking, same eval
    /// cadence and `clear_cache`, same last-token split) — the last token
    /// runs `forward_body` with the hidden kept, then the `lm_head_logits`
    /// tail, which composes to exactly `forward_inner`.
    ///
    /// Returns the sampling-ready last-token logits `[1, vocab]` plus the
    /// turn's [`AssistantTurnState`].
    #[allow(dead_code)] // wired into the whole-turn core with the draft turn generalization
    pub(crate) fn assistant_prefill_with_hidden(
        &mut self,
        prefill_tokens: &[u32],
        position_base: i32,
        stream: Stream,
    ) -> Result<(MxArray, AssistantTurnState)> {
        if prefill_tokens.is_empty() {
            return Err(Error::from_reason(
                "gemma4 assistant prefill: empty prefill token set",
            ));
        }
        if self.caches.is_none() {
            self.init_caches_sync()?;
        }

        let inner = &mut *self;
        let caches = inner
            .caches
            .as_mut()
            .ok_or_else(|| Error::from_reason("gemma4 assistant prefill: caches missing"))?;

        let n = prefill_tokens.len() as i64;
        let ids_i32: Vec<i32> = prefill_tokens.iter().map(|&t| t as i32).collect();
        let prompt = MxArray::from_int32(&ids_i32, &[1, n])?;

        {
            let _stream_ctx = StreamContext::new(stream);
            if n > 1 {
                // Body over tokens [0 .. n-1] — the last token is handled by
                // the hidden-keeping forward below (`prefill_body_gemma4`'s
                // split).
                let prefill_len = n - 1;
                let prefill_ids = prompt.slice_axis(1, 0, prefill_len)?;
                let all_embeds = {
                    let emb = inner.embed_tokens.forward(&prefill_ids)?;
                    emb.mul_scalar((inner.config.hidden_size as f64).sqrt())?
                };
                let all_ple: Option<MxArray> = match inner.ple.as_ref() {
                    Some(ple) => Some(compute_ple(&prefill_ids, &all_embeds, ple, prefill_len)?),
                    None => None,
                };
                let mut offset: i64 = 0;
                while offset < prefill_len {
                    let end = if prefill_len - offset > GEMMA4_PREFILL_STEP_SIZE {
                        offset + GEMMA4_PREFILL_STEP_SIZE
                    } else {
                        prefill_len
                    };
                    let chunk_embeds = all_embeds.slice_axis(1, offset, end)?;
                    let chunk_ple = all_ple
                        .as_ref()
                        .map(|p| p.slice_axis(1, offset, end))
                        .transpose()?;
                    let _hidden = forward_body(
                        None,
                        Some(chunk_embeds),
                        &inner.embed_tokens,
                        &inner.layers,
                        caches,
                        &inner.final_norm,
                        inner.ple.as_ref(),
                        chunk_ple.as_ref(),
                        &inner.config,
                        None,
                    )?;
                    // Eval cadence mirrors `prefill_body_gemma4`: full
                    // chunks eval+clear between iterations; the remainder
                    // chunk is covered by the post-body eval below.
                    if end < prefill_len {
                        eval_gemma4_caches(caches)?;
                        crate::array::clear_cache();
                    }
                    offset = end;
                }
            }
        }
        eval_gemma4_caches(caches)?;

        // Last token: `forward_body` keeping the `[1, 1, hidden]`
        // post-final-norm hidden (the turn's h_prev seed) + the
        // `lm_head_logits` tail — together exactly the AR prefill's
        // last-token `forward_inner`.
        let last = prompt.slice_axis(1, n - 1, n)?;
        let (last_logits, h_last) = {
            let _stream_ctx = StreamContext::new(stream);
            crate::models::gemma4::diagnostic::set_step(-1);
            let hidden = forward_body(
                Some(&last),
                None,
                &inner.embed_tokens,
                &inner.layers,
                caches,
                &inner.final_norm,
                inner.ple.as_ref(),
                None,
                &inner.config,
                None,
            )?;
            let logits = lm_head_logits(
                &hidden,
                &inner.embed_tokens,
                &inner.lm_head,
                inner.embed_weight_t.as_ref(),
                &inner.config,
            )?;
            (logits, hidden)
        };
        let last_logits = last_logits.squeeze(Some(&[1]))?;

        Ok((
            last_logits,
            AssistantTurnState {
                h_prev: h_last,
                next_pos: position_base + n as i32,
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::backend::{ChatBackend, DsparkBackend, DsparkTurnSetup};
    use crate::engine::types::ChatConfig;
    use crate::models::gemma4::dspark::DsparkContextCache;
    use crate::models::gemma4::dspark_decode::tests::{
        chat_config, tiny_inner_with_assistant_draft, tiny_inner_with_draft,
    };
    use crate::models::gemma4::dspark_decode::{
        DsparkTurnState, Gemma4DraftStepper, Gemma4DraftTurnState,
    };
    use crate::models::gemma4::model::dspark_shared_slot_mask;
    use crate::stream::DeviceType;

    fn to_vec_f32(a: &MxArray) -> Vec<f32> {
        a.eval();
        a.to_float32().expect("array must convert to f32").to_vec()
    }

    fn greedy_config(mtp_depth: Option<i32>) -> ChatConfig {
        ChatConfig {
            mtp_depth,
            temperature: Some(0.0),
            ..ChatConfig::default()
        }
    }

    /// Prefill the 4-token tiny prompt, stash the turn state, and return the
    /// assistant stepper's enum wrapper (via the REAL `begin_dspark_decode`
    /// dispatch).
    fn prefilled_assistant_stepper<'a>(
        inner: &'a mut Gemma4Inner,
        p: &ChatParams,
        block_size: usize,
    ) -> Gemma4DraftStepper<'a> {
        let tokens: Vec<u32> = vec![0, 1, 2, 3];
        let stream = Stream::new(DeviceType::Gpu);
        let (_logits, state) = inner
            .assistant_prefill_with_hidden(&tokens, 0, stream)
            .expect("tiny assistant prefill must succeed");
        assert_eq!(state.next_pos, 4, "prefill must report the prompt length");
        inner.draft_turn_state = Some(Gemma4DraftTurnState::Assistant(state));
        let setup = DsparkTurnSetup {
            params: p,
            block_size,
        };
        inner
            .begin_dspark_decode(&setup)
            .expect("begin with an assistant stash must succeed")
    }

    // ── begin_dspark_decode dispatch ───────────────────────────────────

    /// The assistant arm of `begin_dspark_decode`: KV sources computed from
    /// the target config, shared slots from the mask, cursor from the
    /// stash; the stash is consumed; no-stash and BOTH variant-mismatch
    /// directions are hard errors.
    #[test]
    fn begin_dspark_decode_dispatches_assistant() {
        let mut inner = tiny_inner_with_assistant_draft();
        let p = ChatBackend::resolve_params(&inner, &chat_config(None));
        let setup = DsparkTurnSetup {
            params: &p,
            block_size: 3,
        };

        // No stash → hard error naming the missing prefill.
        let err = inner
            .begin_dspark_decode(&setup)
            .err()
            .expect("begin without a stash must fail");
        assert!(
            err.reason.contains("prepared draft context"),
            "got: {}",
            err.reason
        );

        // Assistant stash → assistant stepper with config-derived routing.
        inner.draft_turn_state = Some(Gemma4DraftTurnState::Assistant(AssistantTurnState {
            h_prev: MxArray::zeros(&[1, 1, 8], None).expect("zeros"),
            next_pos: 7,
        }));
        {
            let stepper = match inner
                .begin_dspark_decode(&setup)
                .expect("begin with an assistant stash must succeed")
            {
                Gemma4DraftStepper::Assistant(stepper) => stepper,
                Gemma4DraftStepper::Dspark(_) => {
                    panic!("an assistant draft must yield the assistant stepper")
                }
            };
            assert_eq!(
                stepper.kv_sources,
                AssistantKvSources {
                    sliding: 2,
                    full: 1
                },
                "KV sources must be the last non-shared layer of each type"
            );
            assert_eq!(
                stepper.shared_slots,
                vec![false, false, false, true],
                "mask must come from dspark_shared_slot_mask(config)"
            );
            assert_eq!(stepper.next_pos, 7);
            assert!(stepper.rollback.is_none() && stepper.verify_hidden.is_none());
        }
        assert!(
            inner.draft_turn_state.is_none(),
            "the per-turn stash must be consumed by begin_dspark_decode"
        );

        // Mismatch 1: DSpark stash on an assistant draft.
        inner.draft_turn_state = Some(Gemma4DraftTurnState::Dspark(DsparkTurnState {
            ctx: DsparkContextCache::new(2),
            next_pos: 3,
        }));
        let err = inner
            .begin_dspark_decode(&setup)
            .err()
            .expect("a DSpark stash on an assistant draft must fail");
        assert!(
            err.reason.contains("DSpark turn state")
                && err.reason.contains("not the DSpark variant"),
            "got: {}",
            err.reason
        );

        // Mismatch 2: assistant stash on a DSpark draft.
        let mut dspark_inner = tiny_inner_with_draft();
        dspark_inner.draft_turn_state = Some(Gemma4DraftTurnState::Assistant(AssistantTurnState {
            h_prev: MxArray::zeros(&[1, 1, 8], None).expect("zeros"),
            next_pos: 7,
        }));
        let err = dspark_inner
            .begin_dspark_decode(&setup)
            .err()
            .expect("an assistant stash on a DSpark draft must fail");
        assert!(
            err.reason.contains("assistant turn state")
                && err.reason.contains("not the assistant variant"),
            "got: {}",
            err.reason
        );
    }

    // ── propose: committed KV, held position, no target mutation ──────

    /// After a real tiny prefill, a depth-3 propose returns 3 ids with
    /// EMPTY dists at greedy temperature, is deterministic on repeat, and
    /// mutates NO target state (cursor and every cache offset unchanged).
    #[test]
    fn propose_reads_committed_kv_and_holds_position() {
        // Pin MLX's global PRNG so the placeholder weights — and the drafted
        // ids — are identical on every run (the repo's `mlx_seed` pattern).
        unsafe { mlx_sys::mlx_seed(0xA551_0001) };
        let mut inner = tiny_inner_with_assistant_draft();
        let p = ChatBackend::resolve_params(&inner, &greedy_config(Some(3)));

        let offsets_before: Vec<i32>;
        let (prop1_ids, prop2_ids);
        {
            let mut stepper = match prefilled_assistant_stepper(&mut inner, &p, 3) {
                Gemma4DraftStepper::Assistant(stepper) => stepper,
                Gemma4DraftStepper::Dspark(_) => panic!("expected the assistant stepper"),
            };
            offsets_before = stepper
                .inner
                .caches
                .as_ref()
                .expect("caches live after prefill")
                .iter()
                .map(|c| c.get_offset())
                .collect();

            let mut rng = rand::rng();
            let prop1 = stepper
                .propose(3, 3, &p, &mut rng)
                .expect("first propose must succeed");
            assert_eq!(prop1.draft_ids.len(), 3, "depth-3 propose returns 3 ids");
            assert!(
                prop1.draft_dists.is_empty(),
                "greedy-temperature propose must ship empty dists"
            );
            for &id in &prop1.draft_ids {
                assert!((0..16).contains(&id), "drafted id {id} out of tiny vocab");
            }

            let prop2 = stepper
                .propose(3, 3, &p, &mut rng)
                .expect("second propose must succeed");
            assert_eq!(
                prop1.draft_ids, prop2.draft_ids,
                "propose must be deterministic on repeat (it mutates no state)"
            );
            assert_eq!(stepper.next_pos, 4, "propose must not advance the cursor");
            prop1_ids = prop1.draft_ids;
            prop2_ids = prop2.draft_ids;
        }
        assert_eq!(prop1_ids, prop2_ids);

        let offsets_after: Vec<i32> = inner
            .caches
            .as_ref()
            .expect("caches live")
            .iter()
            .map(|c| c.get_offset())
            .collect();
        assert_eq!(
            offsets_before, offsets_after,
            "propose must not touch the target caches"
        );
    }

    // ── verify + commit: h_prev chaining and cursor advance ───────────

    /// Verify anchor+2 then commit keep=2: the cursor advances by 2, h_prev
    /// becomes the verify hidden's row keep-1 bitwise, active caches roll
    /// back to the kept prefix — and both commit-without-verify and
    /// verify-before-commit are hard errors.
    #[test]
    fn commit_updates_h_prev_and_cursor() {
        unsafe { mlx_sys::mlx_seed(0xA551_0002) };
        let mut inner = tiny_inner_with_assistant_draft();
        let p = ChatBackend::resolve_params(&inner, &greedy_config(Some(2)));
        {
            let mut stepper = match prefilled_assistant_stepper(&mut inner, &p, 2) {
                Gemma4DraftStepper::Assistant(stepper) => stepper,
                Gemma4DraftStepper::Dspark(_) => panic!("expected the assistant stepper"),
            };

            // Commit without a verify → hard error.
            let err = stepper
                .commit(1, 1)
                .expect_err("commit without a pending verify must fail");
            assert!(
                err.reason.contains("no pending verify rollback"),
                "got: {}",
                err.reason
            );

            // Verify [anchor, d0, d1].
            let out = stepper
                .verify(&[3, 1, 2])
                .expect("3-token verify must succeed");
            assert_eq!(
                out.logits.shape().expect("logits shape").to_vec(),
                vec![1, 3, 16]
            );

            // A second verify before commit → hard error.
            let err = match stepper.verify(&[3, 1, 2]) {
                Ok(_) => panic!("verify before the previous commit must fail"),
                Err(err) => err,
            };
            assert!(
                err.reason.contains("never committed"),
                "got: {}",
                err.reason
            );

            let hidden = stepper
                .verify_hidden
                .as_ref()
                .expect("verify must stash the hidden")
                .clone();
            assert_eq!(
                hidden.shape().expect("hidden shape").to_vec(),
                vec![1, 3, 8]
            );

            stepper.commit(2, 3).expect("commit keep=2 must succeed");
            assert_eq!(
                stepper.next_pos, 6,
                "commit must advance the cursor by keep"
            );
            let expected = hidden.slice_axis(1, 1, 2).expect("slice row keep-1");
            assert_eq!(
                to_vec_f32(&stepper.h_prev),
                to_vec_f32(&expected),
                "h_prev must be the verify hidden at the LAST KEPT slot"
            );
            assert!(
                stepper.rollback.is_none() && stepper.verify_hidden.is_none(),
                "commit must consume the verify stash"
            );
        }

        // Physical rollback: active caches at prompt(4) + keep(2); the
        // KV-shared slot untouched.
        let mask = dspark_shared_slot_mask(&inner.config);
        let caches = inner.caches.as_ref().expect("caches live");
        for (i, cache) in caches.iter().enumerate() {
            let expected = if mask[i] { 0 } else { 6 };
            assert_eq!(
                cache.get_offset(),
                expected,
                "cache {i} offset after commit keep=2"
            );
        }
    }
}
