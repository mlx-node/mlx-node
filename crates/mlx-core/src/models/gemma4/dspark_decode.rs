//! Gemma4 draft speculative-decode wiring: the family-side
//! [`DsparkStepper`]/[`DsparkBackend`] implementation the engine-owned
//! [`crate::engine::dspark_turn::run_dspark_turn`] loop drives, plus the
//! variant-generic whole-turn core (`draft_chat_turn`) behind gemma4's
//! `ChatBackend::run_speculative_turn` executor.
//!
//! Split of responsibilities:
//!   * the DSpark DRAFT model (5-layer cross-attending transformer, markov
//!     head, confidence head, context K/V cache) lives in [`super::dspark`];
//!     the assistant draft + its stepper live in [`super::assistant`] /
//!     [`super::assistant_decode`];
//!   * the TARGET-side primitives (hidden tap, verify forward,
//!     snapshot/commit rollback, shared-slot mask) live in
//!     [`super::model`] / [`super::layer_cache`];
//!   * the model-agnostic propose → verify → accept → stop-clamp → commit
//!     loop lives in [`crate::engine::dspark_turn`];
//!   * THIS module glues them together for gemma4: the DSpark stepper, the
//!     variant dispatch ([`Gemma4DraftTurnState`] / [`Gemma4DraftStepper`] /
//!     `begin_dspark_decode`), and the whole-turn core.

use std::time::Instant;

use napi::bindgen_prelude::*;

use crate::array::MxArray;
use crate::decode_profiler::DecodeProfiler;
use crate::engine::backend::{
    ChatBackend, DsparkBackend, DsparkProposal, DsparkStepper, DsparkTurnSetup, DsparkVerifyOutput,
    FinalizeArgs, ResetScope, StreamEmitter, TurnOutput, WholeTurnArgs,
};
use crate::engine::decode::StreamingCtx;
use crate::engine::dspark_turn::{DsparkTurnArgs, run_dspark_turn};
use crate::engine::finalize::compute_performance_metrics;
use crate::engine::params::{ChatParams, generated_capacity_hint};
use crate::engine::penalties::{ReasoningTracker, apply_all_penalties};
use crate::stream::{DeviceType, Stream, StreamContext};

use super::assistant_decode::{AssistantTurnState, Gemma4AssistantStepper};
use super::dspark::{DsparkContextCache, DsparkTap, truncate_by_confidence};
use super::layer_cache::{Gemma4VerifyRollback, commit_after_verify, snapshot_before_verify};
use super::model::{
    GEMMA4_PREFILL_STEP_SIZE, Gemma4Draft, Gemma4Inner, assistant_kv_source_indices, compute_ple,
    dspark_shared_slot_mask, dspark_verify_forward, eval_gemma4_caches, forward_body,
    forward_inner,
};

/// Per-turn draft handoff from the whole-turn core's prefill to
/// [`DsparkBackend::begin_dspark_decode`], one variant per
/// [`super::model::Gemma4Draft`] variant.
///
/// The engine's [`DsparkTurnSetup`] carries only turn constants, so the
/// prefill-derived state travels through `Gemma4Inner::draft_turn_state`:
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
/// cache built by the whole-turn core's tapped prefill
/// (`dspark_prefill_with_tap`).
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
/// model for the whole decode loop. The tapped target hiddens and the
/// verify rollback are stashed between `verify` and `commit` — they never
/// cross the engine trait (see the invariant on [`DsparkStepper`]).
pub(crate) struct Gemma4DsparkStepper<'a> {
    inner: &'a mut Gemma4Inner,
    ctx: DsparkContextCache,
    /// Absolute position of the next verify block's anchor (== the current
    /// committed sequence length: prompt + anchor-exclusive generation).
    next_pos: i32,
    /// Draft `target_layer_ids` as decoder indices (strictly ascending).
    layer_ids: Vec<usize>,
    /// Config-derived KV-shared slot mask ([`dspark_shared_slot_mask`]),
    /// passed to every `snapshot_before_verify`.
    shared_slots: Vec<bool>,
    confidence_threshold: f32,
    /// Pending rollback from the last `verify`, consumed by `commit`.
    rollback: Option<Gemma4VerifyRollback>,
    /// Tapped `[1, 1+L, hidden]` hiddens from the last `verify` (one per
    /// `layer_ids` entry), consumed by `commit`.
    tapped: Option<Vec<MxArray>>,
}

impl DsparkStepper for Gemma4DsparkStepper<'_> {
    fn propose(
        &mut self,
        anchor_id: u32,
        max_len: usize,
        params: &ChatParams,
        rng: &mut dyn rand::Rng,
    ) -> Result<DsparkProposal> {
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
        // cycle's commit would orphan its rollback (the caches would then
        // hold TWO uncommitted verify blocks).
        if self.rollback.is_some() || self.tapped.is_some() {
            return Err(Error::from_reason(
                "gemma4 DSpark verify: previous verify was never committed",
            ));
        }

        let inner = &mut *self.inner;
        let caches = inner
            .caches
            .as_mut()
            .ok_or_else(|| Error::from_reason("gemma4 DSpark verify: caches missing"))?;
        let rb = snapshot_before_verify(caches, verify_ids.len(), &self.shared_slots)?;

        let ids_i32: Vec<i32> = verify_ids.iter().map(|&t| t as i32).collect();
        let block = MxArray::from_int32(&ids_i32, &[1, verify_ids.len() as i64])?;
        let mut tap = DsparkTap::new(&self.layer_ids);
        let logits = dspark_verify_forward(
            &block,
            &inner.embed_tokens,
            &inner.layers,
            caches,
            &inner.final_norm,
            &inner.lm_head,
            inner.embed_weight_t.as_ref(),
            inner.ple.as_ref(),
            &inner.config,
            &mut tap,
        )?;
        if tap.captured.len() != self.layer_ids.len() {
            return Err(Error::from_reason(format!(
                "gemma4 DSpark verify: tapped {} hiddens for {} configured target layers",
                tap.captured.len(),
                self.layer_ids.len()
            )));
        }

        self.rollback = Some(rb);
        self.tapped = Some(tap.captured);
        Ok(DsparkVerifyOutput { logits })
    }

    fn commit(&mut self, keep: usize, total_written: usize) -> Result<()> {
        let rb = self.rollback.take().ok_or_else(|| {
            Error::from_reason("gemma4 DSpark commit: no pending verify rollback")
        })?;
        let tapped = self
            .tapped
            .take()
            .ok_or_else(|| Error::from_reason("gemma4 DSpark commit: no stashed tapped hiddens"))?;
        if keep == 0 {
            return Err(Error::from_reason(
                "gemma4 DSpark commit: engine contract violation (keep must be >= 1 — the anchor's slot is unconditionally kept)",
            ));
        }
        if let Some(first) = tapped.first()
            && first.shape_at(1)? != total_written as i64
        {
            return Err(Error::from_reason(format!(
                "gemma4 DSpark commit: stashed tapped hiddens cover {} positions but the engine reports a {}-token verify block",
                first.shape_at(1)?,
                total_written
            )));
        }

        // Target side: keep the first `keep` of the verify block's K/V
        // slots, roll back the rest (validated against the shared-slot mask
        // on every commit, full keep included).
        {
            let caches = self
                .inner
                .caches
                .as_mut()
                .ok_or_else(|| Error::from_reason("gemma4 DSpark commit: caches missing"))?;
            commit_after_verify(caches, &rb, keep)?;
        }

        // Draft side: fuse the kept prefix of the tapped hiddens and append
        // it to the persisted context at the block's base position, then
        // advance the cursor. The boundary token has no slot on either side
        // — it re-enters as the next cycle's verify anchor.
        let draft = self
            .inner
            .dspark_draft()
            .ok_or_else(|| Error::from_reason("gemma4 DSpark commit: no draft model loaded"))?;
        let mut kept: Vec<MxArray> = Vec::with_capacity(tapped.len());
        for hidden in &tapped {
            kept.push(hidden.slice_axis(1, 0, keep as i64)?);
        }
        let fused = draft.fuse_context(&kept)?;
        self.ctx.append(draft, &fused, self.next_pos)?;
        self.next_pos += keep as i32;
        Ok(())
    }

    fn eval_boundary(&self, token: &MxArray) {
        // Schedule-only async eval of the next cycle's anchor (gemma4's
        // decode eval pattern: token only, never the logits).
        MxArray::async_eval_arrays(&[token]);
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
}

impl DsparkBackend for Gemma4Inner {
    type DsparkDecode<'a>
        = Gemma4DraftStepper<'a>
    where
        Self: 'a;

    fn begin_dspark_decode(
        &mut self,
        _setup: &DsparkTurnSetup<'_>,
    ) -> Result<Self::DsparkDecode<'_>> {
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
                let shared_slots = dspark_shared_slot_mask(&self.config);
                let confidence_threshold = dspark_confidence_threshold_from_env();
                Ok(Gemma4DraftStepper::Dspark(Gemma4DsparkStepper {
                    inner: self,
                    ctx: state.ctx,
                    next_pos: state.next_pos,
                    layer_ids,
                    shared_slots,
                    confidence_threshold,
                    rollback: None,
                    tapped: None,
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

impl Gemma4Inner {
    /// Draft whole-turn core (both [`Gemma4Draft`] variants) behind
    /// gemma4's `ChatBackend::run_speculative_turn` executor — the draft analog of the
    /// engine's generic `chat_turn_core` tail, sync AND streaming through
    /// the same body (`args.sink` presence selects the mode, mirroring
    /// `vision_chat_turn` / the MTP whole-turn cores).
    ///
    /// Flow: resolve params (+ `extra_eos_ids`) → prefix decision via the
    /// existing cache-prefix machinery → the VARIANT's prefill (DSpark:
    /// chunked prefill WITH the hidden tap, per-chunk capture → fuse →
    /// context-append → drop; assistant: the same chunked prefill keeping
    /// only the last token's post-final-norm hidden) → anchor sample
    /// (byte-identical to the generic flow) → `run_dspark_turn` → save
    /// (AR-parity: stop exits drop the final token, length exits
    /// materialize its K/V and keep all — post-turn history AND cache
    /// offsets equal the AR flow's for every stop shape) → finalize
    /// (+ default `augment_performance`, which fills the `mtp_*`
    /// acceptance fields). Every error between prefill start and the save
    /// fails CLOSED (`draft_fail_closed`).
    pub(crate) fn draft_chat_turn(&mut self, args: &mut WholeTurnArgs<'_>) -> Result<TurnOutput> {
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
            Some(Gemma4Draft::Dspark(_)) => match self.dspark_prefill_with_tap(
                &prefill_tokens,
                cached_prefix_len as i32,
                generation_stream,
            ) {
                Ok((logits, state)) => (logits, Gemma4DraftTurnState::Dspark(state)),
                Err(e) => return Err(self.draft_fail_closed(e)),
            },
            Some(Gemma4Draft::Assistant(_)) => match self.assistant_prefill_with_hidden(
                &prefill_tokens,
                cached_prefix_len as i32,
                generation_stream,
            ) {
                Ok((logits, state)) => (logits, Gemma4DraftTurnState::Assistant(state)),
                Err(e) => return Err(self.draft_fail_closed(e)),
            },
            None => {
                return Err(self.draft_fail_closed(Error::from_reason(
                    "gemma4 draft turn: no draft model loaded",
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

        // Per-cycle draft cap: DSpark blocks are checkpoint-pinned; the
        // assistant drafts by chained AR steps, so the resolved depth IS
        // the cap.
        let block_size = match self.draft.as_ref() {
            Some(Gemma4Draft::Dspark(draft)) => draft.config.block_size,
            Some(Gemma4Draft::Assistant(_)) => p.mtp_depth,
            None => {
                return Err(self.draft_fail_closed(Error::from_reason(
                    "gemma4 draft turn: no draft model loaded",
                )));
            }
        };

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
            if let Err(e) = self.dspark_materialize_final(final_token, generation_stream) {
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
            // Fresh text-only turn: clear any stale media keys (the DSpark
            // path is text-only by its `mtp_turn` gate).
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

    /// Chunked prefill WITH the DSpark hidden tap.
    ///
    /// Target-side compute is byte-identical to the AR path
    /// (`prefill_body_gemma4` + the last-token `forward_inner`): same
    /// upfront embedding/PLE, same 512-token chunking, same eval cadence
    /// and `clear_cache`, same last-token split — the tap only CLONES the
    /// residual-stream hiddens (tap purity is pinned by
    /// `dspark_tap_purity_and_verify_forward`).
    ///
    /// Per chunk: forward w/ a FRESH tap → `fuse_context` → context-append
    /// at the chunk's absolute base → drop the tap. Full-prompt tapped
    /// hiddens are never held. `position_base` is the cached-prefix length
    /// (the absolute position of `prefill_tokens[0]`); cached-prefix tokens
    /// have NO context rows — the draft cross-attends over whatever rows
    /// exist, and verification always re-derives ground truth from the
    /// target, so a shorter context can only depress acceptance, never
    /// correctness.
    ///
    /// Returns the sampling-ready last-token logits `[1, vocab]` plus the
    /// turn's [`DsparkTurnState`].
    fn dspark_prefill_with_tap(
        &mut self,
        prefill_tokens: &[u32],
        position_base: i32,
        stream: Stream,
    ) -> Result<(MxArray, DsparkTurnState)> {
        if prefill_tokens.is_empty() {
            return Err(Error::from_reason(
                "gemma4 DSpark prefill: empty prefill token set",
            ));
        }
        if self.caches.is_none() {
            self.init_caches_sync()?;
        }

        let inner = &mut *self;
        // Field-level borrow (not the whole-struct accessor) so the draft
        // borrow stays disjoint from the `caches` borrow below.
        let draft = match inner.draft.as_ref() {
            Some(Gemma4Draft::Dspark(draft)) => draft,
            _ => {
                return Err(Error::from_reason(
                    "gemma4 DSpark prefill: no DSpark draft model loaded",
                ));
            }
        };
        let caches = inner
            .caches
            .as_mut()
            .ok_or_else(|| Error::from_reason("gemma4 DSpark prefill: caches missing"))?;
        let layer_ids: Vec<usize> = draft
            .config
            .target_layer_ids
            .iter()
            .map(|&id| id as usize)
            .collect();
        let mut ctx = DsparkContextCache::new(draft.num_layers());

        let n = prefill_tokens.len() as i64;
        let ids_i32: Vec<i32> = prefill_tokens.iter().map(|&t| t as i32).collect();
        let prompt = MxArray::from_int32(&ids_i32, &[1, n])?;

        {
            let _stream_ctx = StreamContext::new(stream);
            if n > 1 {
                // Body over tokens [0 .. n-1] — the last token is handled by
                // the full forward below (`prefill_body_gemma4`'s split).
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
                    let mut tap = DsparkTap::new(&layer_ids);
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
                        Some(&mut tap),
                    )?;
                    // capture → fuse → append → drop, per chunk.
                    let fused = draft.fuse_context(&tap.captured)?;
                    ctx.append(draft, &fused, position_base + offset as i32)?;
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

        // Last token: full forward (lm_head + softcap) with one more tapped
        // context row, exactly the AR prefill's last-token split.
        let last = prompt.slice_axis(1, n - 1, n)?;
        let last_logits = {
            let _stream_ctx = StreamContext::new(stream);
            crate::models::gemma4::diagnostic::set_step(-1);
            let mut tap = DsparkTap::new(&layer_ids);
            let logits = forward_inner(
                &last,
                &inner.embed_tokens,
                &inner.layers,
                caches,
                &inner.final_norm,
                &inner.lm_head,
                inner.embed_weight_t.as_ref(),
                inner.ple.as_ref(),
                &inner.config,
                Some(&mut tap),
            )?;
            let fused = draft.fuse_context(&tap.captured)?;
            ctx.append(draft, &fused, position_base + (n - 1) as i32)?;
            logits
        };
        let last_logits = last_logits.squeeze(Some(&[1]))?;

        Ok((
            last_logits,
            DsparkTurnState {
                ctx,
                next_pos: position_base + n as i32,
            },
        ))
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

    /// LENGTH-exit only: run ONE more forward for the final emitted token
    /// so its K/V lands in the live session caches, then DISCARD the
    /// logits — the DSpark analog of `Gemma4Decode::materialize_final`
    /// (the AR flow keeps every token on length exits and materializes the
    /// final one; the save's keep-all history then equals the physical
    /// cache offsets). No sample / push / emit; like the AR steppers, this
    /// deliberately does NOT fire a sliding decode-boundary checkpoint.
    fn dspark_materialize_final(&mut self, token_id: u32, stream: Stream) -> Result<()> {
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
            None,
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

    /// Tiny draft config geometry-matched to [`tiny_target_config`]
    /// (hidden 8, vocab 16, 4 target layers, block_size 3).
    fn tiny_draft_config() -> DsparkConfig {
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
        .expect("tiny draft config must deserialize")
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
        }
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

    /// The stepper derives the shared-slot mask from the target config
    /// (`dspark_shared_slot_mask`) and the layer ids from the draft's
    /// `target_layer_ids`; the per-turn context stash is TAKEN (single
    /// use), and calling begin without a stash is a hard error.
    #[test]
    fn begin_dspark_decode_takes_stash_and_derives_mask() {
        let mut inner = tiny_inner_with_draft();
        let num_draft_layers = inner
            .dspark_draft()
            .map(|d| d.num_layers())
            .expect("draft loaded");

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

        // Stash → stepper carries the config-derived mask + draft layer ids
        // and the stash is consumed.
        inner.draft_turn_state = Some(Gemma4DraftTurnState::Dspark(DsparkTurnState {
            ctx: DsparkContextCache::new(num_draft_layers),
            next_pos: 7,
        }));
        {
            let stepper = match inner
                .begin_dspark_decode(&setup)
                .expect("begin with a stash must succeed")
            {
                Gemma4DraftStepper::Dspark(stepper) => stepper,
                Gemma4DraftStepper::Assistant(_) => {
                    panic!("a DSpark draft must yield the DSpark stepper")
                }
            };
            assert_eq!(
                stepper.shared_slots,
                vec![false, false, false, true],
                "mask must come from dspark_shared_slot_mask(config)"
            );
            assert_eq!(stepper.layer_ids, vec![0, 2]);
            assert_eq!(stepper.next_pos, 7);
            assert!(stepper.rollback.is_none() && stepper.tapped.is_none());
        }
        assert!(
            inner.draft_turn_state.is_none(),
            "the per-turn stash must be consumed by begin_dspark_decode"
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
            mtp_depth,
            max_new_tokens: Some(max_new_tokens),
            temperature: Some(0.0),
            reuse_cache: Some(true),
            report_performance: Some(true),
            include_reasoning: Some(false),
            ..ChatConfig::default()
        }
    }

    /// Drive `draft_chat_turn` directly (sync — no model thread), with an
    /// out-of-vocab `eos_id` so the tiny model can only exit "length".
    pub(crate) fn run_tiny_draft_turn(
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
        match inner.draft_chat_turn(&mut args)? {
            TurnOutput::Complete(r) => Ok(*r),
            TurnOutput::Streamed => panic!("sync draft turn returned TurnOutput::Streamed"),
        }
    }

    /// FAIL-CLOSED regression: a REAL (unmocked) stepper error AFTER
    /// prefill has advanced the target caches must drop the entire warm
    /// session — caches, `cached_token_history`, per-turn stash — so no
    /// later turn can prefix-match into K/V the history knows nothing
    /// about; and the very next turn must succeed via the cold path.
    ///
    /// Error injection: sliding_window 2 with the default depth (draft
    /// block_size 3) makes the first cycle's 4-row verify block violate
    /// `snapshot_before_verify`'s window >= block invariant — the error
    /// fires at the verify seam of cycle 1, after `dspark_prefill_with_tap`
    /// appended the whole prompt to every active cache.
    #[test]
    fn dspark_turn_error_fails_closed_then_cold_turn_recovers() {
        let mut inner = Gemma4Inner::new(tiny_target_config_with_window(2))
            .expect("tiny window-2 Gemma4Inner must construct");
        inner.draft = Some(Gemma4Draft::Dspark(
            DsparkDraftModel::new(tiny_draft_config()).expect("tiny draft model"),
        ));
        let tokenizer = tiny_qwen_tokenizer();
        let tokens: Vec<u32> = vec![0, 1, 2, 3];

        // Turn 1: unset depth resolves to block_size 3 → 1+3 verify rows
        // over a 2-token window → hard error mid-turn.
        let err = run_tiny_draft_turn(&mut inner, &tokenizer, &tokens, &tiny_turn_config(None, 8))
            .expect_err("a depth-3 verify block must violate the window-2 rollback invariant");
        assert!(
            err.reason.contains("sliding window") && err.reason.contains("verify block"),
            "expected the snapshot_before_verify window guard, got: {}",
            err.reason
        );
        // Fail CLOSED: nothing warm-reusable may survive the error.
        assert!(inner.caches.is_none(), "caches must be dropped");
        assert!(
            inner.cached_token_history.is_empty(),
            "cached_token_history must be cleared (it never covered the prefilled K/V)"
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
            "no prefix hit may match against the dropped session"
        );

        // Turn 2: depth 1 → verify blocks of <= 2 rows fit the window; the
        // turn must run cold end-to-end and land fully consistent.
        let res = run_tiny_draft_turn(
            &mut inner,
            &tokenizer,
            &tokens,
            &tiny_turn_config(Some(1), 3),
        )
        .expect("the next turn after fail-closed must succeed via the cold path");
        assert_eq!(res.finish_reason, "length");
        assert_eq!(
            res.cached_tokens, 0,
            "nothing may be warm-reused after fail-closed"
        );
        assert_eq!(res.num_tokens, 3, "budget-3 length exit");

        // Length-exit AR parity (keep-all + materialize): the saved history
        // holds prompt + ALL generated tokens and every ACTIVE cache offset
        // equals the history length — the final token's K/V was
        // materialized by `dspark_materialize_final`.
        let history = inner.cached_token_history.clone();
        assert_eq!(history.len(), tokens.len() + res.num_tokens as usize);
        assert_eq!(&history[..tokens.len()], &tokens[..]);
        let mask = dspark_shared_slot_mask(&inner.config);
        let caches = inner
            .caches
            .as_ref()
            .expect("live caches after a successful turn");
        for (i, cache) in caches.iter().enumerate() {
            let expected = if mask[i] { 0 } else { history.len() as i32 };
            assert_eq!(
                cache.get_offset(),
                expected,
                "cache {i} offset must equal the saved history length \
                 (length-exit materialize; shared slots stay unwritten)"
            );
        }
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

    /// WHOLE-TURN streaming cancellation through `draft_chat_turn`: a
    /// cancel raised from the chunk sink must terminate the stream promptly
    /// ("cancelled", bounded block-granular overrun — never running on to
    /// the budget) and leave the cached session state consistent (AR-parity
    /// drop-last: the final emitted token is persisted in NEITHER the
    /// history NOR the caches; offsets equal the history), with the next
    /// turn running normally. Chunk-vs-residual byte accounting for the
    /// cancel suffix is pinned at the engine seam
    /// (`dspark_turn_streaming_cancel_in_clamp_commits_exactly_once`),
    /// where the mid-clamp cancel point is injectable; a sink-driven flip
    /// lands at the next loop-top by construction.
    #[test]
    fn dspark_streaming_cancel_whole_turn_state_consistent() {
        // Placeholder weights come from MLX's global PRNG (`Linear::new` is
        // Xavier-uniform); pin it so the token stream — and with it the
        // chunk/flip timing the `n` bound below asserts on — is identical
        // on every run (the repo's established `mlx_seed` test pattern).
        unsafe { mlx_sys::mlx_seed(0xD5_9A4B_0001) };
        let mut inner =
            Gemma4Inner::new(tiny_target_config()).expect("tiny Gemma4Inner must construct");
        inner.draft = Some(Gemma4Draft::Dspark(
            DsparkDraftModel::new(tiny_draft_config()).expect("tiny draft model"),
        ));
        let tokenizer = tiny_qwen_tokenizer();
        let tokens: Vec<u32> = vec![0, 1, 2, 3];
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
        let p = ChatBackend::resolve_params(&inner, &config);
        let thinking = ChatBackend::thinking_setup(&inner, &config);
        let mut args = WholeTurnArgs {
            tokens: &tokens,
            tokenizer: &tokenizer,
            eos_id: 999,
            config: &config,
            params: &p,
            thinking,
            plan: TurnPlan {
                is_delta: false,
                input_media: MediaCapabilities::NONE,
                context_media: MediaCapabilities::NONE,
                use_paged_attention: false,
                decoder: DecoderPlan::Speculative(SpeculativeKind::DraftModel),
            },
            sink: Some(&sink),
            cancelled: Some(&cancelled),
            media: MediaInputs {
                images: &[],
                audio: &[],
            },
        };
        let out = inner
            .draft_chat_turn(&mut args)
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
        // the history AND was never committed to the caches.
        let history = inner.cached_token_history.clone();
        assert_eq!(
            history.len(),
            tokens.len() + n - 1,
            "cancelled turn must persist prompt + generated minus the final token"
        );
        assert_eq!(&history[..tokens.len()], &tokens[..]);
        let mask = dspark_shared_slot_mask(&inner.config);
        let caches = inner.caches.as_ref().expect("live caches after the turn");
        for (i, cache) in caches.iter().enumerate() {
            let expected = if mask[i] { 0 } else { history.len() as i32 };
            assert_eq!(
                cache.get_offset(),
                expected,
                "cache {i} offset must equal the saved history length after a cancel"
            );
        }
        assert!(
            ChatBackend::has_live_session(&inner),
            "a cleanly cancelled turn leaves a warm-reusable session (AR parity)"
        );

        // The next turn runs normally (fresh prompt: the longer saved
        // history is a prefix-miss, so it takes the cold path).
        let res = run_tiny_draft_turn(
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

        // --- AR baseline: 2 turns, capturing history + offsets ---
        let ar1 = crate::engine::session::session_start(&mut inner, vec![user(PROMPT)], cfg(false))
            .expect("AR turn 1 failed");
        assert_eq!(ar1.finish_reason, "stop", "fixture must stop early on EOS");
        let ar_h1 = inner.cached_token_history.clone();
        assert_offsets_match_history(&inner, "ar_turn1");
        let ar2 = crate::engine::session::session_continue(
            &mut inner,
            FOLLOW_UP.to_string(),
            None,
            None,
            cfg(false),
        )
        .expect("AR turn 2 failed");
        let ar_h2 = inner.cached_token_history.clone();
        assert_offsets_match_history(&inner, "ar_turn2");

        // --- DSpark pass: same 2 turns ---
        let sp1 = crate::engine::session::session_start(&mut inner, vec![user(PROMPT)], cfg(true))
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
            FOLLOW_UP.to_string(),
            None,
            None,
            cfg(true),
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
