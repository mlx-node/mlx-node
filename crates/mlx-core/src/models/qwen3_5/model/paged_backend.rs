//! `PagedBackend` for `Qwen35Inner` plus the paged decode/prefix adapters.

use super::*;
use crate::engine::paged_epilogue::FinalTokenPolicy;
use crate::engine::paged_stepper::{EvalPolicy, PagedStepModel, PagedStepper};
use crate::models::qwen3_5::paged_forward::PrefillTapSink;

/// Adapter giving the engine's [`ChunkSink`] the `.call()` shape the
/// `decode_loop!` macro and the engine's `run_mtp_turn` loop (and the
/// streaming cores behind the whole-turn probes) expect from a
/// `ThreadsafeFunction`-like callback.
///
/// The engine owns the channel and hands the probes a `&dyn ChunkSink`,
/// so the wrapper forwards `.call()` to [`ChunkSink::send`]; the call
/// mode is meaningless on the mpsc path and is dropped.
pub(super) struct StreamSender<'a>(pub(super) &'a dyn ChunkSink);

impl StreamSender<'_> {
    pub(super) fn call(
        &self,
        result: napi::Result<ChatStreamChunk>,
        _mode: ThreadsafeFunctionCallMode,
    ) {
        self.0.send(result);
    }

    /// Count one kept token on the request's progress.
    pub(super) fn count_token(&self) {
        if let Some(progress) = self.0.progress() {
            progress.add_generated(1);
        }
    }
}

/// Paged decode state for qwen3_5 dense (the paged analog of the FLAT
/// [`Qwen35Decode`]); wrapped by [`PagedStepper`] for the `DecodeStep`
/// impl. Drives [`crate::engine::decode::run_decode_loop`] through the
/// generic [`crate::engine::paged_turn::run_paged_turn`]: each
/// `paged_step` runs the pure-Rust eager paged step against the live
/// post-prefill adapter pools + GDN caches. Created by
/// `<Qwen35Inner as PagedBackend>::begin_paged_decode`, consumed across
/// the whole decode loop.
pub(crate) struct Qwen35PagedDecode<'a> {
    inner: &'a mut Qwen35Inner,
}

impl Drop for Qwen35PagedDecode<'_> {
    fn drop(&mut self) {
        self.inner.serial_decode = None;
    }
}

impl PagedStepModel for Qwen35PagedDecode<'_> {
    /// `SyncTokenAndForcedLogits`: isolated sync/stream comparisons did
    /// not resolve a benefit from async scheduling here — retain one
    /// completion boundary per token. The forced path also evals the
    /// logits: a forced host token does not depend on the target/cache
    /// writes.
    const EVAL: EvalPolicy = EvalPolicy::SyncTokenAndForcedLogits;
    /// `AlwaysDrop` — CRITICAL: dense paged drops the last token
    /// UNCONDITIONALLY (see `save_paged_history`). The adapter only
    /// advanced for the tokens the loop actually forwarded; re-running a
    /// decode step for the final length-exit token would record a token
    /// the GDN/adapter state never advanced → recurrent-state desync vs
    /// the saved drop-last history.
    const FINAL_TOKEN_POLICY: FinalTokenPolicy = FinalTokenPolicy::AlwaysDrop;

    fn paged_step(&mut self, token_id: u32) -> Result<MxArray> {
        #[cfg(test)]
        PAGED_STEPS.with(|steps| steps.set(steps.get() + 1));
        // Pure-Rust eager paged decode step.
        //
        // PERF: `token_id` is HANDED by the engine (already read once at
        // the loop top via `y.item_at_int32`), so we do NOT
        // re-`item_at_int32` the fresh `input_ids` reshape — that
        // redundant second per-step eval/sync measurably regressed decode.
        let _hadamard_scope = crate::quant::prism_hadamard::HadamardDecodeScope::enter(
            self.inner.prism_hadamard.is_some(),
        );
        let embed = self.inner.embedding.clone();
        crate::models::qwen3_5::dflash2_turn::paged_step_claimed(&mut *self.inner, |inner| {
            let caches_ref = inner.caches.as_mut().ok_or_else(|| {
                Error::from_reason("Qwen35PagedDecode::paged_step: caches dropped mid-decode")
            })?;
            let adapter = inner.paged_adapter.as_mut().ok_or_else(|| {
                Error::from_reason(
                    "Qwen35PagedDecode::paged_step: paged_adapter dropped mid-decode",
                )
            })?;
            crate::models::qwen3_5::paged_forward::run_paged_decode_step(
                token_id,
                &embed,
                &mut inner.layers,
                caches_ref,
                &inner.final_norm,
                &inner.lm_head,
                &inner.layer_kinds,
                adapter,
                inner.cached_rope_deltas.unwrap_or(0),
            )
        })
    }
}

/// qwen3_5 dense paged prefix state — the effective prefix/suffix split from
/// `prepare_turn_with_max_cache_hit_tokens`, PLUS the full prompt tokens and
/// the GDN-prime flag.
///
/// `full_tokens` is needed because the engine hands `paged_prefill` ONLY the
/// suffix (`tokens[effective_cached_prefix_len..]`), but
/// `run_paged_prefill_chunk` needs the FULL prompt for the GDN pre-pass over
/// the cached prefix. `gdn_prefix_already_primed` is the dense-specific bit the
/// prime resolves (the GDN recurrent state was already populated live / from a
/// checkpoint / via replay) and `paged_prefill` threads into
/// `run_paged_prefill_chunk` so the prefill skips re-priming the GDN prefix.
pub(crate) struct Qwen35PrefixState {
    pub(super) effective_cached_prefix_len: usize,
    pub(super) suffix_len: usize,
    pub(super) full_tokens: Vec<u32>,
    pub(super) cache_salt: u64,
    pub(super) gdn_prefix_already_primed: bool,
}

#[cfg(test)]
thread_local! {
    /// Eager paged decode steps run on this thread.
    pub(crate) static PAGED_STEPS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

impl Qwen35PrefixState {
    /// The turn's whole prompt.
    pub(crate) fn tokens(&self) -> &[u32] {
        &self.full_tokens
    }
}

impl PagedPrefix for Qwen35PrefixState {
    fn effective_cached_prefix_len(&self) -> usize {
        self.effective_cached_prefix_len
    }
    fn suffix_len(&self) -> usize {
        self.suffix_len
    }
}

impl Qwen35Inner {
    /// The `prepare_turn_…` + `prepare_dense_gdn_prefix_state` block that
    /// opens a dense paged turn. `dflash2_hit` is a DFlash2 turn's reusable
    /// prefix: the prepare then books nothing past it.
    pub(crate) fn prime_dense_prefix(
        &mut self,
        plan: &[u32],
        cache_salt: u64,
        dflash2_hit: Option<crate::models::qwen3_5::dflash2_turn::DFlash2Hit>,
    ) -> Result<Qwen35PrefixState> {
        self.paged_finalize_failed = false;
        let trace_enabled = inference_trace_enabled();
        let total_budget = plan.len() as u32;
        // vLLM exact-prefix cap: leave at least one prompt token to prefill so
        // the decoder always has something to consume.
        let max_cache_hit_tokens = total_budget.saturating_sub(1);
        // Whole-turn callers retain sequence 0. The continuous scheduler
        // activates a request-local recurrent row before entering this same
        // prime core, so K/V and GDN state share one sequence identity.
        let seq_id = self.active_scheduled_seq.unwrap_or(0);
        let block_size = {
            let adapter = self.paged_adapter.as_ref().ok_or_else(|| {
                Error::from_reason(
                    "prime_prefix_state: paged_adapter is None — this load has \
                     no block-paged pool",
                )
            })?;
            adapter.block_size()
        };
        let carries_image_lineage = self.cached_image_key.is_some()
            && !self.cached_paged_image_token_positions.is_empty()
            && !self.cached_token_history.is_empty()
            && plan.starts_with(&self.cached_token_history);
        let image_positions = if carries_image_lineage {
            self.cached_paged_image_token_positions.as_slice()
        } else {
            &[]
        };
        let lookup_extra_keys =
            engine::build_paged_extra_keys(plan.len(), block_size, image_positions);

        // Adapter-owned warm/cold lifecycle. The [MLX_TRACE] line below
        // reads the PRE-turn live state, so probe the adapter immutably FIRST
        // (prepare_turn mutates request_tokens via continue_turn/reset). The
        // adapter re-reads the same state internally, so live_* is identical to
        // what prepare_turn decides on. reuse_cache=true literal: continuation
        // eligibility carries no reuse term (the engine's reuse_cache drives
        // finalize/save instead). Suffix blocks are allocated inside
        // prepare_turn.
        let live_ready;
        let live_prefix_match;
        let live_tokens_len;
        let mut live_mismatch = TokenPrefixMismatchTrace::default();
        {
            let adapter = self
                .paged_adapter
                .as_ref()
                .ok_or_else(|| Error::from_reason("prime_prefix_state: paged_adapter is None"))?;
            live_ready = adapter.is_live_for_continue();
            let live_tokens = adapter.request_tokens();
            live_tokens_len = live_tokens.len();
            live_prefix_match = plan.starts_with(live_tokens);
            if trace_enabled && live_ready && !live_prefix_match {
                live_mismatch = token_prefix_mismatch_trace(plan, live_tokens);
            }
        }
        let keep = crate::models::qwen3_5::dflash2_turn::Keep {
            tokens: plan,
            keys: Some((&lookup_extra_keys, cache_salt)),
        };
        // A disk state's chain comes back with the KV lookup.
        let disk_state = dflash2_hit
            .is_none()
            .then(|| {
                crate::models::qwen3_5::dflash2_turn::disk_state_depth(
                    self,
                    plan,
                    &lookup_extra_keys,
                    cache_salt,
                    max_cache_hit_tokens,
                )
            })
            .flatten();
        let turn_plan = crate::models::qwen3_5::dflash2_turn::claimed(self, keep, |inner| {
            let adapter = inner
                .paged_adapter
                .as_mut()
                .ok_or_else(|| Error::from_reason("prime_prefix_state: paged_adapter is None"))?;
            let Some(hit) = dflash2_hit else {
                if let Some(depth) = disk_state {
                    adapter.arm_state_restore(depth);
                }
                // A scheduled lane books nothing past its hit: each prefill
                // slice's step claim books the slice (Splash
                // `Engine::workEndOf`).
                let prepared = if seq_id == 0 {
                    adapter.prepare_turn_per_block_with_max_cache_hit_tokens(
                        seq_id,
                        plan,
                        total_budget,
                        true,
                        &lookup_extra_keys,
                        cache_salt,
                        false,
                        max_cache_hit_tokens,
                    )
                } else {
                    adapter.prepare_turn_per_block_booking_hit_only(
                        seq_id,
                        plan,
                        total_budget,
                        &lookup_extra_keys,
                        cache_salt,
                        false,
                        max_cache_hit_tokens,
                    )
                };
                return prepared.map_err(Error::from_reason);
            };
            crate::models::qwen3_5::dflash2_turn::prepare_dflash2_turn(
                adapter,
                seq_id,
                plan,
                &lookup_extra_keys,
                cache_salt,
                hit,
            )
        })
        .inspect_err(|_| self.dflash2_states.state_read = None)?;
        let cached_prefix_len = turn_plan.cached_prefix_len;
        let continued_live_prefix = turn_plan.continued_live_prefix;
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] qwen3.5-dense paged_prefix_lookup prompt_tokens={} \
                 cached_prefix_tokens={} continued_live_prefix={} live_ready={} \
                 live_match={} live_tokens={} live_mismatch_at={} prompt_token={} live_token={}",
                plan.len(),
                cached_prefix_len,
                continued_live_prefix,
                live_ready,
                live_prefix_match,
                live_tokens_len,
                live_mismatch.index,
                live_mismatch.prompt_token,
                live_mismatch.cached_token
            ));
        }

        self.finish_dense_prime(
            plan,
            cache_salt,
            block_size,
            &lookup_extra_keys,
            carries_image_lineage,
            &turn_plan,
        )
    }

    /// Splash `replayTokens` for a suspended serial lane: the GDN prime
    /// over `plan`'s restored prefix, then the prefill of the rest of
    /// `tokens`, as at a turn start; a DFlash2 lane's prefill is tapped and
    /// leaves its draft window in the turn state. Returns the last row's
    /// logits, evaluated.
    pub(crate) fn replay_dense_serial_lane(
        &mut self,
        tokens: &[u32],
        cache_salt: u64,
        plan: &crate::transformer::paged_kv_cache_adapter::PagedTurnPlan,
        decode: crate::models::qwen3_5::dflash2_turn::SerialDecode,
        stored: Option<crate::models::qwen3_5::dflash2_state::store::EntryId>,
    ) -> Result<MxArray> {
        let block_size = self
            .paged_adapter
            .as_ref()
            .map(PagedKVCacheAdapter::block_size)
            .ok_or_else(|| Error::from_reason("a suspended lane lost its paged adapter"))?;
        let keys = engine::build_paged_extra_keys(tokens.len(), block_size, &[]);
        let prefix = self.finish_dense_prime(tokens, cache_salt, block_size, &keys, false, plan)?;
        let cached = prefix.effective_cached_prefix_len;
        let suffix = tokens
            .get(cached..)
            .ok_or_else(|| Error::from_reason("a replay's prefix passed its tokens"))?;
        let logits = match decode {
            crate::models::qwen3_5::dflash2_turn::SerialDecode::Autoregressive
                if self.dflash2.is_none() =>
            {
                self.open_dense_lane_turn(tokens, cached, cache_salt, true)?;
                self.dense_paged_prefill(suffix, &prefix, None)?.0
            }
            _ => {
                if cached == 0 {
                    self.dflash2_context = None;
                }
                crate::models::qwen3_5::dflash2_turn::admit_resumed(self, tokens, stored, cached)?;
                self.tapped_dflash2_prefill(suffix, &prefix)?
            }
        };
        logits.eval();
        crate::array::synchronize_and_clear_cache();
        Ok(logits)
    }

    /// Splash `Runtime::suspend` for the serial lane, before its sequence
    /// is released: the live GDN cell goes, the draft ring and context
    /// window go back to the idle lane, the GDN state use ends. Nothing is
    /// published.
    pub(crate) fn vacate_dense_lane(&mut self) {
        self.end_gdn_use(0);
        self.caches = None;
        self.idle_dflash2_lane();
    }

    /// The prime after the adapter's prepare: the GDN state over
    /// `turn_plan`'s cached prefix, the turn's session state, and the
    /// prefix the prefill continues from.
    pub(crate) fn finish_dense_prime(
        &mut self,
        plan: &[u32],
        cache_salt: u64,
        block_size: u32,
        lookup_extra_keys: &[Vec<u64>],
        carries_image_lineage: bool,
        turn_plan: &crate::transformer::paged_kv_cache_adapter::PagedTurnPlan,
    ) -> Result<Qwen35PrefixState> {
        let total_budget = plan.len() as u32;
        // GDN recurrent-state prime (live / checkpoint / replay). No qwen3/lfm2
        // analog — dense carries GDN recurrent state across turns.
        let gdn_prefix_preparation = self.prepare_dense_gdn_prefix_state(
            plan,
            turn_plan.cached_prefix_len,
            block_size,
            lookup_extra_keys,
            cache_salt,
            turn_plan.continued_live_prefix,
        )?;
        let cached_prefix_len = gdn_prefix_preparation.cached_prefix_len;
        let reconciled_prefix = crate::engine::recurrent_state::reconcile_hybrid_prefix(&[
            cached_prefix_len,
            gdn_prefix_preparation
                .restored_prefix_tokens
                .saturating_add(gdn_prefix_preparation.replayed_prefix_tokens),
        ]);
        debug_assert_eq!(
            reconciled_prefix, cached_prefix_len,
            "paged K/V and GDN must agree on one hybrid prefix boundary"
        );
        let gdn_prefix_already_primed = gdn_prefix_preparation.already_primed;
        // GDN recurrent state now covers exactly the cached prefix (installed
        // live, restored from a cold sidecar, or replayed just above). Discharge
        // the adapter's auxiliary-state obligation — set when the cold tier
        // carries a `ColdSidecarPolicy` and a HOT K/V hit handed back a prefix
        // no sidecar was gated against — before the turn's first `record_tokens`.
        // A no-op for a live continuation or a sidecar-backed restore.
        if let Some(adapter) = self.paged_adapter.as_mut() {
            adapter
                .confirm_aux_prefix_primed(cached_prefix_len)
                .map_err(Error::from_reason)?;
        }
        let preserves_image_lineage = carries_image_lineage && cached_prefix_len > 0;
        // Clear the per-turn session state here (history is re-set in
        // `save_paged_history`; image key is reset because the paged path does
        // not carry it across turns). The cross-turn M-RoPE delta is carried
        // only when this turn extends the live image sequence
        // (continued_live_prefix); a cold start or a non-live prefix-cache hit
        // (text-only prefix) drops a stale image delta so the text suffix
        // rotates at the raw physical slot.
        self.cached_token_history.clear();
        if !preserves_image_lineage {
            self.cached_image_key = None;
            self.cached_paged_image_token_positions.clear();
        }
        self.cached_rope_deltas = crate::models::qwen3_5::paged_forward::rope_delta_for_paged_turn(
            self.cached_rope_deltas,
            preserves_image_lineage,
        );

        let suffix_len = total_budget.checked_sub(cached_prefix_len).ok_or_else(|| {
            Error::from_reason("prime_prefix_state: cached_prefix_len > total_prompt_tokens")
        })? as usize;

        Ok(Qwen35PrefixState {
            effective_cached_prefix_len: cached_prefix_len as usize,
            suffix_len,
            full_tokens: plan.to_vec(),
            cache_salt,
            gdn_prefix_already_primed,
        })
    }
}

impl Qwen35Inner {
    /// An autoregressive whole turn on a draft load primes and prefills as a
    /// DFlash2 turn (Splash `configureDraftStatePlan` for every request): its
    /// states carry the draft window, and its lane is released at its end.
    pub(crate) fn draft_lane_turn(&self) -> bool {
        self.dflash2.is_some() && self.active_scheduled_seq.is_none()
    }

    /// Opens the lane's state points once the prime put it at `cached`, on
    /// a load without a draft (a draft load's turns open their own).
    pub(crate) fn open_dense_lane_turn(
        &mut self,
        tokens: &[u32],
        cached: usize,
        cache_salt: u64,
        reuse: bool,
    ) -> Result<()> {
        if self.dflash2.is_some() || self.paged_adapter.is_none() {
            return Ok(());
        }
        let seq = crate::models::qwen3_5::dflash2_turn::DFlash2Target::lane_seq(self);
        let restored = match self.gdn_in_use.get(&seq) {
            Some(crate::engine::governor::ledger::UnitKey::Store(id)) => Some(*id),
            _ => None,
        };
        crate::models::qwen3_5::dflash2_turn::open_lane_turn(
            self,
            tokens,
            &[],
            cached,
            cache_salt,
            reuse,
            restored,
        )
    }

    /// The paged prefill; with `draft`, the draft context advanced over the
    /// prompt from the same forward's taps. Without one, on a load without
    /// a draft, the turn's state points publish as the chunks reach them.
    pub(crate) fn dense_paged_prefill(
        &mut self,
        suffix_tokens: &[u32],
        prefix: &Qwen35PrefixState,
        draft: Option<crate::models::qwen3_5::dflash2::DFlash2ContextCache>,
    ) -> Result<(
        MxArray,
        Option<crate::models::qwen3_5::dflash2::DFlash2ContextCache>,
    )> {
        // Scheduled MTP also retains the final normalized prompt row as the
        // request-owned draft seed. The ordinary path only produces logits.
        let layer_kinds = crate::models::qwen3_5::decoder_layer::compute_layer_kinds(
            self.config.num_layers as usize,
            |i| self.config.is_linear_layer(i),
        );
        let embed = self.embedding.clone();
        // Cross-turn M-RoPE delta (0 unless this engine-driven text turn warm-
        // continues an image prefill); aligns the suffix keys with the
        // compressed-position image keys.
        let rope_deltas = self.cached_rope_deltas.unwrap_or(0);
        let chunk_size = crate::models::qwen3_5::paged_forward::prefill_chunk_rows(
            self.cold_gdn_prefill_chunk_size(),
        );
        // Cloned up front (cheap Option<Arc>) so the chunk-loop call below
        // can borrow `self.layers`/`self.caches` mutably at the same time.
        let turn_cancel = self.turn_cancel.clone();
        let mtp_seq = self
            .active_scheduled_seq
            .filter(|seq| self.scheduled_mtp.owners.contains_key(seq));
        if draft.is_some() && mtp_seq.is_some() {
            return Err(Error::from_reason(
                "a DFlash2 prefill never runs on a scheduled MTP sequence",
            ));
        }
        let mut mtp_hidden = None;
        if let Some(context) = draft {
            let (plan, captures) = crate::models::qwen3_5::dflash2_turn::turn_plan(
                self,
                &prefix.full_tokens,
                prefix.effective_cached_prefix_len,
                prefix.cache_salt,
                chunk_size,
            )?;
            // The draft leaves the model for the prefill so each chunk can book
            // through the take order on the whole model.
            let model = self
                .dflash2
                .take()
                .ok_or_else(|| Error::from_reason("Qwen3.8 has no loaded DFlash2 companion"))?;
            let lane = match self.dflash2_states.lane_window.take() {
                Some(lane) => lane,
                None => match self.dflash2_states.idle_lane.take_window(&model.config) {
                    Ok(lane) => lane,
                    Err(error) => {
                        self.dflash2 = Some(model);
                        return Err(error);
                    }
                },
            };
            let sink = crate::models::qwen3_5::dflash2_turn::DraftTapSink::new(
                &model,
                context,
                lane,
                &prefix.full_tokens,
                prefix.effective_cached_prefix_len,
                &plan,
                &captures,
            );
            let mut sink = match sink {
                Ok(sink) => sink,
                Err(error) => {
                    self.dflash2 = Some(model);
                    return Err(error);
                }
            };
            let prefilled =
                crate::models::qwen3_5::paged_forward::run_paged_prefill_chunk_with_size(
                    &prefix.full_tokens,
                    suffix_tokens,
                    prefix.effective_cached_prefix_len as u32,
                    prefix.gdn_prefix_already_primed,
                    &embed,
                    self,
                    &layer_kinds,
                    chunk_size,
                    rope_deltas,
                    turn_cancel.as_deref(),
                    Some(&mut sink as &mut dyn PrefillTapSink),
                );
            let (context, lane) = sink.into_parts();
            self.dflash2_states.idle_lane.give_window(lane);
            self.dflash2 = Some(model);
            let (logits, checkpoint) = match prefilled {
                Ok(prefilled) => prefilled,
                Err(error) => {
                    self.dflash2_states.idle_lane.give_ring(Some(context));
                    return Err(error);
                }
            };
            self.publish_dense_gdn_materialized_prefix_checkpoint(
                &prefix.full_tokens,
                prefix.cache_salt,
                checkpoint,
            );
            return Ok((logits, Some(context)));
        }
        let draft_lane = self.dflash2.is_some()
            && mtp_seq.is_none()
            && self
                .active_scheduled_seq
                .is_some_and(|seq| self.dflash2_states.turns.contains_key(&seq));
        if draft_lane {
            let cached = prefix.effective_cached_prefix_len;
            let (logits, checkpoint) =
                crate::models::qwen3_5::dflash2_turn::scheduled_draft_prefill(
                    self,
                    &prefix.full_tokens,
                    cached,
                    cached + suffix_tokens.len(),
                    chunk_size,
                    |inner| inner.dflash2.take(),
                    |inner, model| inner.dflash2 = Some(model),
                    |target, sink| {
                        crate::models::qwen3_5::paged_forward::run_paged_prefill_chunk_with_size(
                            &prefix.full_tokens,
                            suffix_tokens,
                            cached as u32,
                            prefix.gdn_prefix_already_primed,
                            &embed,
                            target,
                            &layer_kinds,
                            chunk_size,
                            rope_deltas,
                            turn_cancel.as_deref(),
                            Some(sink),
                        )
                    },
                )?;
            self.publish_dense_gdn_materialized_prefix_checkpoint(
                &prefix.full_tokens,
                prefix.cache_salt,
                checkpoint,
            );
            return Ok((logits, None));
        }
        // Splash `publishReachedStateBoundaries` runs for every request: an
        // MTP sequence's prefill reaches its points as a plain one does.
        let lane_states = self.dflash2.is_none()
            && self
                .dflash2_states
                .turns
                .contains_key(&self.active_scheduled_seq.unwrap_or(0));
        let cached = prefix.effective_cached_prefix_len;
        let captures = if lane_states {
            crate::models::qwen3_5::dflash2_turn::lane_command_breaks(
                self,
                cached,
                cached + suffix_tokens.len(),
                chunk_size,
            )
        } else {
            Vec::new()
        };
        let mut sink = crate::models::qwen3_5::dflash2_turn::LaneStateSink::new(&captures);
        let taps = lane_states.then_some(&mut sink as &mut dyn PrefillTapSink);
        let mut lane_target;
        let mut plain_target;
        let target: &mut dyn crate::models::qwen3_5::paged_forward::PrefillTarget<DecoderLayer> =
            if lane_states {
                lane_target = crate::models::qwen3_5::dflash2_turn::LanePrefill(&mut *self);
                &mut lane_target
            } else {
                plain_target = crate::models::qwen3_5::paged_forward::PrefillParts {
                    layers: &mut self.layers,
                    caches: self.caches.as_mut().ok_or_else(|| {
                        Error::from_reason("paged_prefill: caches not initialized")
                    })?,
                    final_norm: &self.final_norm,
                    lm_head: &self.lm_head,
                    adapter: self.paged_adapter.as_mut().ok_or_else(|| {
                        Error::from_reason("paged_prefill: paged_adapter dropped")
                    })?,
                };
                &mut plain_target
            };
        let (logits, checkpoint) = if mtp_seq.is_some() {
            let (logits, hidden, checkpoint) =
                crate::models::qwen3_5::paged_forward::run_paged_prefill_chunk_with_hidden_with_size(
                    &prefix.full_tokens,
                    suffix_tokens,
                    cached as u32,
                    prefix.gdn_prefix_already_primed,
                    &embed,
                    target,
                    &layer_kinds,
                    chunk_size,
                    Some(suffix_tokens.len()),
                    rope_deltas,
                    turn_cancel.as_deref(),
                    taps,
                )?;
            mtp_hidden = Some(hidden);
            (logits, checkpoint)
        } else {
            crate::models::qwen3_5::paged_forward::run_paged_prefill_chunk_with_size(
                &prefix.full_tokens,
                suffix_tokens,
                cached as u32,
                prefix.gdn_prefix_already_primed,
                &embed,
                target,
                &layer_kinds,
                chunk_size,
                rope_deltas,
                turn_cancel.as_deref(),
                taps,
            )?
        };
        if let Some(seq) = mtp_seq {
            crate::models::qwen3_5::scheduled_mtp::ScheduledMtpTarget::prefill_scheduled_mtp(
                self,
                seq,
                (prefix.effective_cached_prefix_len + suffix_tokens.len()) as u32,
                mtp_hidden.ok_or_else(|| Error::from_reason("MTP prefill omitted its seed"))?,
                suffix_tokens,
            )?;
        }
        self.publish_dense_gdn_materialized_prefix_checkpoint(
            &prefix.full_tokens,
            prefix.cache_salt,
            checkpoint,
        );
        Ok((logits, None))
    }
}

impl crate::models::qwen3_5::paged_forward::PrefillTarget<DecoderLayer> for Qwen35Inner {
    fn parts(
        &mut self,
    ) -> Result<crate::models::qwen3_5::paged_forward::PrefillParts<'_, DecoderLayer>> {
        Ok(crate::models::qwen3_5::paged_forward::PrefillParts {
            layers: &mut self.layers,
            caches: self
                .caches
                .as_mut()
                .ok_or_else(|| Error::from_reason("paged_prefill: caches not initialized"))?,
            final_norm: &self.final_norm,
            lm_head: &self.lm_head,
            adapter: self
                .paged_adapter
                .as_mut()
                .ok_or_else(|| Error::from_reason("paged_prefill: paged_adapter dropped"))?,
        })
    }

    fn record(&mut self, chunk: &[u32]) -> Result<()> {
        crate::models::qwen3_5::dflash2_turn::record_chunk_claimed(self, chunk)
    }

    fn prepare_capture(
        &mut self,
        row: u32,
    ) -> Result<crate::models::qwen3_5::paged_forward::CaptureGrant> {
        crate::models::qwen3_5::dflash2_turn::reserve_capture(self, row)
    }

    fn reached(
        &mut self,
        row: u32,
        outcome: crate::models::qwen3_5::paged_forward::Reached,
    ) -> Result<()> {
        crate::models::qwen3_5::dflash2_turn::reached(self, row, outcome)
    }
}

impl PagedBackend for Qwen35Inner {
    type PagedDecode<'a>
        = PagedStepper<Qwen35PagedDecode<'a>>
    where
        Self: 'a;
    type PrefixState = Qwen35PrefixState;

    fn set_generation_suffix(&mut self, suffix: Option<std::sync::Arc<[u32]>>) {
        self.dflash2_states.generation_suffix = suffix;
    }

    fn prime_prefix_state(
        &mut self,
        plan: &[u32],
        reuse_cache: bool,
        _block_size: usize,
        _extra_keys: &[u64],
        cache_salt: u64,
    ) -> Result<Self::PrefixState> {
        if self.draft_lane_turn() {
            return crate::engine::dspark_turn::PagedDsparkBackend::prime_dspark_prefix_state(
                self,
                plan,
                reuse_cache,
                cache_salt,
            );
        }
        if self.dflash2.is_some() {
            return crate::models::qwen3_5::dflash2_turn::prime_scheduled_draft_lane(
                self,
                plan,
                reuse_cache,
                cache_salt,
                |inner, hit| inner.prime_dense_prefix(plan, cache_salt, Some(hit)),
            );
        }
        let prefix = self.prime_dense_prefix(plan, cache_salt, None)?;
        self.open_dense_lane_turn(
            plan,
            prefix.effective_cached_prefix_len,
            cache_salt,
            reuse_cache,
        )?;
        Ok(prefix)
    }

    fn paged_prefill(
        &mut self,
        suffix_tokens: &[u32],
        prefix: &Self::PrefixState,
        stream: Stream,
    ) -> Result<MxArray> {
        if self.draft_lane_turn() {
            return crate::engine::dspark_turn::PagedDsparkBackend::paged_prefill_with_draft_state(
                self,
                suffix_tokens,
                prefix,
                stream,
            );
        }
        self.dense_paged_prefill(suffix_tokens, prefix, None)
            .map(|(logits, _)| logits)
    }

    fn begin_paged_decode(&mut self) -> Result<Self::PagedDecode<'_>> {
        // Pure-Rust eager paged decode: the stepper drives
        // `run_paged_decode_step` against the live post-prefill adapter pools +
        // GDN caches. No compiled-graph seeding / lifecycle locks needed.
        if self.active_scheduled_seq.is_none() {
            self.serial_decode =
                Some(crate::models::qwen3_5::dflash2_turn::SerialDecode::Autoregressive);
        }
        Ok(PagedStepper(Qwen35PagedDecode { inner: self }))
    }

    fn finalize_paged_turn(&mut self, reuse_cache: bool, cache_salt: u64) {
        // Terminal lifecycle block of a paged turn. Success: keep the request
        // live across turns when reuse is on, using PER-BLOCK extra keys (NOT
        // qwen3's empty `&[]`), so the next turn's continue builds on the
        // partial trailing block's live K/V; otherwise register full blocks
        // for reuse + release. The trait hook remains infallible, but an adapter
        // error must downgrade the session before the engine's unconditional
        // `save_paged_history` call can publish image-placeholder history.
        self.paged_finalize_failed = false;
        self.paged_turn_seq = self
            .paged_adapter
            .as_ref()
            .and_then(PagedKVCacheAdapter::active_seq_id);
        // A DFlash2 turn's lane is released at its end (Splash
        // `Engine::release`): its states are the published ones.
        let dflash2 = self.dflash2.is_some()
            && self
                .dflash2_states
                .turns
                .contains_key(&self.active_scheduled_seq.unwrap_or(0));
        let mut release_pending = false;
        let mut finalize_error = match self.paged_adapter.as_mut() {
            Some(adapter) if reuse_cache && !dflash2 => {
                let total_for_finalize = adapter.request_tokens().len();
                let block_size = adapter.block_size();
                let finalize_extra_keys = engine::build_paged_extra_keys(
                    total_for_finalize,
                    block_size,
                    &self.cached_paged_image_token_positions,
                );
                adapter
                    .finalize_turn_keep_live_per_block(&finalize_extra_keys, cache_salt)
                    .err()
            }
            Some(adapter) => {
                let total_for_finalize = adapter.request_tokens().len();
                let block_size = adapter.block_size();
                let finalize_extra_keys = engine::build_paged_extra_keys(
                    total_for_finalize,
                    block_size,
                    &self.cached_paged_image_token_positions,
                );
                // The release is attempted below, even when registration fails.
                release_pending = true;
                crate::models::qwen3_5::dflash2_turn::log_turn_end("register");
                adapter
                    .register_full_blocks_for_reuse_per_block(&finalize_extra_keys, cache_salt)
                    .err()
            }
            None => Some("paged_adapter is None during dense finalization".to_owned()),
        };
        // Always attempt the release for the non-reuse path, even when
        // registration failed.
        if release_pending && let Some(adapter) = self.paged_adapter.as_mut() {
            crate::models::qwen3_5::dflash2_turn::log_turn_end("release");
            finalize_error = finalize_error.or(adapter.release_request().err());
        }
        crate::models::qwen3_5::dflash2_turn::end_turn_states(self);
        if dflash2 {
            let fresh = self.fresh_dflash2_layer_caches();
            let lane = self.caches.replace(fresh);
            crate::models::qwen3_5::dflash2_turn::note_released_lane(lane);
            self.idle_dflash2_lane();
            self.dflash2_states.released = true;
            self.paged_gdn_state_dirty = false;
        }
        if let Some(error) = finalize_error {
            self.downgrade_failed_paged_finalize(&error);
        }
    }

    fn abort_paged_turn(&mut self) {
        // Hybrid error-path teardown: K/V and GDN may have advanced by
        // different amounts, so a bare adapter release would leave a false-live
        // recurrent session behind. Never register / keep live; invalidate all
        // model-local continuation state without masking the original error.
        self.invalidate_dense_paged_session("generic paged turn abort");
    }

    fn paged_decode_stream(&self, _generation_stream: Stream) -> Stream {
        // Run the paged DECODE on the canonical DEFAULT stream, NOT the
        // `generation_stream`. dense's paged forward + every
        // `y.eval()` run on the MLX DEFAULT stream; running the forward on a
        // queue separate from the shared loop's top-of-iteration `y.eval()`
        // (always on the default stream) would force a cross-queue
        // completion-wait every token.
        // `paged_prefill` still runs on `generation_stream`. See the
        // `PagedBackend::paged_decode_stream` doc for the full mechanism.
        Stream::default(crate::stream::DeviceType::Gpu)
    }

    fn save_paged_history(
        &mut self,
        save_tokens: &[u32],
        generated: &[u32],
        _keep_all: bool,
        reuse_cache: bool,
    ) -> Result<()> {
        let turn_seq = self.paged_turn_seq.take();
        let released = std::mem::take(&mut self.dflash2_states.released);
        // `finalize_paged_turn` is an infallible trait hook. When its adapter
        // operation failed it already invalidated the native caches and left
        // this one-turn latch so the engine's unconditional save cannot revive
        // the session from expanded image-placeholder ids.
        if std::mem::take(&mut self.paged_finalize_failed) {
            self.cached_token_history.clear();
            self.cached_image_key = None;
            self.cached_paged_image_token_positions.clear();
            self.cached_rope_deltas = None;
            self.gdn_last_history_checkpoint = None;
            self.caches = None;
            return Ok(());
        }
        // dense paged ALWAYS drops the last token, regardless of the engine's
        // `keep_all` (length-exit) signal — the paged decode loop NEVER forwards
        // the LAST sampled token (the engine's forward gate skips it AND
        // `materialize_final` is a no-op for dense), so the last `generated`
        // entry is NOT in the adapter / GDN caches and must be dropped to keep
        // the saved history aligned with the live cache state. Ordering:
        // finalize → set history (drop-last, last_token_in_cache=false) → GDN
        // checkpoint → clear image key. PRESERVE THIS EXACT ORDER — it is the
        // most delicate part for T=0 byte-equality.
        if !reuse_cache {
            self.cached_token_history.clear();
            self.cached_image_key = None;
            self.cached_paged_image_token_positions.clear();
            self.cached_rope_deltas = None;
            if let Some(seq_id) = turn_seq {
                self.end_gdn_use(seq_id);
            }
            return Ok(());
        }
        let mut full_history = save_tokens.to_vec();
        if !generated.is_empty() {
            // last_token_in_cache == false → drop-last UNCONDITIONAL.
            let upto = generated.len().saturating_sub(1);
            full_history.extend_from_slice(&generated[..upto]);
        }
        self.cached_token_history = full_history;
        // A released DFlash2 lane holds no live state to check or keep:
        // Splash forms no state at a turn's end.
        if released {
            if let Some(seq_id) = turn_seq {
                self.end_gdn_use(seq_id);
            }
            return Ok(());
        }
        // Frontier agreement before any GDN state is keyed on the history
        // just published. A disagreement arms the refuse-to-persist latch the
        // checkpoint store below consumes.
        let history_len = self.cached_token_history.len();
        self.check_dense_paged_frontier(history_len, "paged history save");
        // GDN history checkpoint — must run AFTER the history is set (it
        // snapshots the live recurrent caches keyed on `cached_token_history`),
        // BEFORE clearing the image key. A checkpoint/eval failure here
        // PROPAGATES (`?`) to abort the turn: a half-snapshotted or
        // failed-eval GDN state must NOT be published as a reusable
        // warm-continue checkpoint, or the next turn reads corrupt
        // recurrent state.
        let store = self.remember_dense_gdn_history_checkpoint()?;
        if let Some(seq_id) = turn_seq {
            self.end_gdn_use(seq_id);
        }
        if inference_trace_enabled() {
            write_inference_trace(format_args!(
                "[MLX_TRACE] qwen3.5-dense gdn_history_checkpoint stored={} tokens={} \
                 eval_ms={:.1} clone_ms={:.1} token_clone_ms={:.1} update_ms={:.1} \
                 total_ms={:.1}",
                store.stored,
                self.cached_token_history.len(),
                store.eval_ms,
                store.clone_ms,
                store.token_clone_ms,
                store.update_ms,
                store.total_ms
            ));
        }
        Ok(())
    }

    fn reconcile_paged_request_tokens(
        &mut self,
        prompt_len: usize,
        generated: &[u32],
        _keep_all: bool,
    ) -> bool {
        // dense ALWAYS drops the last token (see `save_paged_history`), so the
        // to-be-saved history length is `prompt_len + (generated.len() - 1)` (or
        // `prompt_len` when nothing was generated). Roll the adapter back to that
        // length so the next turn's warm-continue gate
        // (`prompt.starts_with(request_tokens())`) is not defeated by a trailing
        // token the pipelined loop recorded at the loop top before the
        // stop-check. `_keep_all` is intentionally ignored (qwen3 signal).
        //
        // Token accounting: on BOTH length and early-stop exits the to-be-saved
        // history equals the adapter cursor (the final/terminal forward was
        // skipped), so `surplus` is 0 and this is a true no-op for dense — but
        // the rollback is kept as the defensive contract the trait mandates.
        let Some(adapter) = self.paged_adapter.as_mut() else {
            return true;
        };
        if let Err((surplus, e)) = crate::engine::paged_epilogue::reconcile_paged_surplus(
            adapter.request_tokens().len(),
            prompt_len,
            generated.len(),
            _keep_all,
            crate::engine::paged_epilogue::FinalTokenPolicy::AlwaysDrop,
            |n| adapter.rollback_last_tokens(n),
        ) {
            tracing::warn!(
                target: "mlx_core::qwen3_5::paged",
                "reconcile_paged_request_tokens: rollback_last_tokens({surplus}) failed \
                 (finalize releases the request; next turn cold-prefills): {e}",
            );
            return false;
        }
        true
    }
}

impl Qwen35Inner {
    /// The tokenizer, params and perf flag an image turn decodes with: the
    /// request config under the checkpoint's generation defaults, with the
    /// planned decoder projected in.
    pub(super) fn vision_turn_params(
        &self,
        args: &WholeTurnArgs<'_>,
    ) -> Result<(Arc<Qwen3Tokenizer>, engine::ChatParams, bool)> {
        let mut config = args.config.clone();
        crate::engine::apply_generation_defaults(&mut config, &self.gen_defaults);
        apply_qwen35_dense_planned_decoder(&mut config, args.plan.decoder);
        let mut p = engine::extract_chat_params(&config);
        p.extra_eos_ids = self.gen_defaults.eos_token_ids.clone();
        let tokenizer = self
            .tokenizer
            .clone()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))?;
        Ok((tokenizer, p, config.report_performance.unwrap_or(false)))
    }

    /// A DFlash2 turn on the pool: the pool window clamps a copy of the
    /// params, then the paged DSpark driver runs it.
    pub(super) fn paged_dflash2_turn(
        &mut self,
        args: &mut WholeTurnArgs<'_>,
    ) -> Result<TurnOutput> {
        let mut constrained_params = args.params.clone();
        self.preflight_paged_context(args.tokens.len(), &mut constrained_params)?;
        let mut constrained_args = WholeTurnArgs {
            tokens: args.tokens,
            tokenizer: args.tokenizer,
            eos_id: args.eos_id,
            config: args.config,
            params: &constrained_params,
            thinking: args.thinking,
            plan: args.plan,
            sink: args.sink,
            cancelled: args.cancelled,
            media: args.media,
        };
        crate::engine::dspark_turn::run_paged_dspark_turn(self, &mut constrained_args)
    }

    /// Whole-turn block-paged dispatch behind [`ChatBackend::run_paged_turn`].
    ///
    /// Conditional router (dense differs from MoE here — `run_decode_loop` has
    /// no MTP gate, so planned MTP must use the native paged-MTP cores):
    ///   * planned MTP turns (sync or stream) use the matching eager paged-MTP
    ///     core, including text deltas over an image-derived live session;
    ///   * autoregressive turns (sync or stream) use the generic paged path via
    ///     `engine::paged_turn::run_paged_turn`, which drives the adapter
    ///     lifecycle through [`PagedBackend`] and reuses `run_decode_loop`.
    pub(super) fn paged_whole_turn(&mut self, args: &mut WholeTurnArgs<'_>) -> Result<TurnOutput> {
        // The MTP cores re-derive `p` from config (`extract_chat_params`). To
        // match the engine's default `resolve_params`, fold
        // generation_config.json defaults in first so the paged-MTP path
        // honors them too (no-op when the checkpoint ships none).
        let mut config = args.config.clone();
        crate::engine::apply_generation_defaults(&mut config, &self.gen_defaults);
        let mut planned_mtp = apply_qwen35_dense_planned_decoder(&mut config, args.plan.decoder);
        // MTP acceptance gate (same policy as the sync whole-turn core).
        if planned_mtp
            && !config.mtp_adaptive_depth.unwrap_or(false)
            && !self.mtp_gate_allows(config.mtp_depth.unwrap_or(1).max(1) as u32)
        {
            planned_mtp = false;
            config.enable_mtp = Some(false);
            // THIS turn leaves the speculative lane. The generic paged driver
            // below dispatches on `plan.decoder`, and dense implements no
            // paged speculative core (its MTP lives in the forked cores
            // above), so a stale `Speculative` would fail the turn on the
            // erroring default hook instead of decoding it autoregressively.
            args.plan.decoder = DecoderPlan::Autoregressive;
        }
        debug_assert!(!planned_mtp || self.has_mtp_weights());
        let mut p = extract_chat_params(&config);
        p.extra_eos_ids = self.gen_defaults.eos_token_ids.clone();
        if planned_mtp {
            self.dflash2_states.generation_suffix =
                args.tokenizer.generation_suffix(args.config).ok().flatten();
            let report_perf = args.config.report_performance.unwrap_or(false);
            let tokenizer = args.tokenizer.clone();
            let thinking = args.thinking;
            return match (args.sink, args.cancelled) {
                (Some(sink), Some(cancelled)) => {
                    let cb = StreamSender(sink);
                    self.paged_turn_stream_core(
                        args.tokens.to_vec(),
                        tokenizer,
                        args.eos_id,
                        p,
                        report_perf,
                        &cb,
                        cancelled,
                        thinking,
                    )?;
                    Ok(TurnOutput::Streamed)
                }
                _ => {
                    let result = self.paged_turn_sync_core(
                        args.tokens.to_vec(),
                        tokenizer,
                        args.eos_id,
                        p,
                        report_perf,
                        thinking,
                    )?;
                    Ok(TurnOutput::Complete(Box::new(result)))
                }
            };
        }

        // NON-MTP (sync or stream) → the generic AR+paged engine path.
        //
        // The specialized MTP/VLM paged cores perform their own hard capacity
        // preflight. The ordinary AR path delegates directly to the generic
        // paged engine, so constrain a copy of the already-resolved params here
        // and make that exact copy authoritative for prefill + decode. This is
        // the native backstop for callers that bypass the TypeScript
        // `ChatSession` preflight.
        let mut constrained_params = args.params.clone();
        self.preflight_paged_context(args.tokens.len(), &mut constrained_params)?;
        if !self.draft_lane_turn() {
            self.dflash2_context = None;
            self.dflash2_turn_state = None;
            self.dflash2_prefix
                .note_drop("paged_whole_turn", DropKind::Dirtied);
        }
        let mut constrained_args = WholeTurnArgs {
            tokens: args.tokens,
            tokenizer: args.tokenizer,
            eos_id: args.eos_id,
            config: args.config,
            params: &constrained_params,
            thinking: args.thinking,
            plan: args.plan,
            sink: args.sink,
            cancelled: args.cancelled,
            media: args.media,
        };
        crate::engine::paged_turn::run_paged_turn(self, &mut constrained_args)
    }
}
