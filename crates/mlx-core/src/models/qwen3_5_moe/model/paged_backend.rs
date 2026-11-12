//! `PagedBackend` for `Qwen35MoeInner` plus the paged decode/prefix adapters.

use super::*;
use crate::engine::paged_epilogue::FinalTokenPolicy;
use crate::engine::paged_stepper::{EvalPolicy, PagedStepModel, PagedStepper};

/// Adapter giving the engine's [`ChunkSink`] the `.call()` shape the
/// `decode_loop!` macro and the engine's `run_mtp_turn` loop (and the
/// streaming cores behind the whole-turn probes) expect from a
/// `ThreadsafeFunction`-like callback.
///
/// The engine owns the channel and hands the probes a `&dyn ChunkSink`,
/// so the wrapper forwards `.call()` to [`ChunkSink::send`]; the call
/// mode is meaningless on the mpsc path and is dropped. Mirrors the
/// dense `StreamSender` adapter.
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

/// Paged decode state for qwen3_5_moe (the paged analog of the FLAT
/// [`Qwen35MoeDecode`]); wrapped by [`PagedStepper`] for the `DecodeStep`
/// impl. Drives [`crate::engine::decode::run_decode_loop`] through the
/// generic [`crate::engine::paged_turn::run_paged_turn`]: each
/// `paged_step` runs the pure-Rust eager paged step against the live
/// post-prefill adapter pools + GDN caches. Created by
/// `<Qwen35MoeInner as PagedBackend>::begin_paged_decode`, consumed across
/// the whole decode loop.
pub(crate) struct Qwen35MoePagedDecode<'a> {
    inner: &'a mut Qwen35MoeInner,
}

impl Drop for Qwen35MoePagedDecode<'_> {
    fn drop(&mut self) {
        self.inner.serial_decode = None;
    }
}

impl PagedStepModel for Qwen35MoePagedDecode<'_> {
    /// `SyncTokenAndForcedLogits`: the measured single-completion cadence
    /// of the dense sibling — one synchronous `next_token.eval()` per
    /// step, plus `logits.eval()` on the budget-forced path (a forced host
    /// token does not depend on the target/cache writes).
    const EVAL: EvalPolicy = EvalPolicy::SyncTokenAndForcedLogits;
    /// `AlwaysDrop` — CRITICAL: moe paged drops the last token
    /// UNCONDITIONALLY (see `save_paged_history`). The adapter / GDN
    /// caches only advanced for the tokens the loop actually forwarded;
    /// re-running a decode step for the final length-exit token would
    /// record a token the GDN/adapter state never advanced →
    /// recurrent-state desync vs the saved drop-last history.
    const FINAL_TOKEN_POLICY: FinalTokenPolicy = FinalTokenPolicy::AlwaysDrop;

    fn paged_step(&mut self, token_id: u32) -> Result<MxArray> {
        // Pure-Rust eager paged decode step.
        //
        // PERF: `token_id` is HANDED by the engine (already read once at
        // the loop top via `y.item_at_int32`), so we do NOT
        // re-`item_at_int32` the fresh `input_ids` reshape — that
        // redundant second per-step eval/sync measurably regressed decode.
        let embed = self.inner.embedding.clone();
        crate::models::qwen3_5::dflash2_turn::paged_step_claimed(&mut *self.inner, |inner| {
            let caches_ref = inner.caches.as_mut().ok_or_else(|| {
                Error::from_reason("Qwen35MoePagedDecode::paged_step: caches dropped mid-decode")
            })?;
            let adapter = inner.paged_adapter.as_mut().ok_or_else(|| {
                Error::from_reason(
                    "Qwen35MoePagedDecode::paged_step: paged_adapter dropped mid-decode",
                )
            })?;
            crate::models::qwen3_5_moe::paged_forward::run_paged_decode_step(
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

/// qwen3_5_moe paged prefix state — the effective prefix/suffix split from
/// `prepare_turn_with_max_cache_hit_tokens`, PLUS the full prompt tokens and
/// the GDN-prime flag.
///
/// `full_tokens` is needed because the engine hands `paged_prefill` ONLY the
/// suffix (`tokens[effective_cached_prefix_len..]`), but
/// `run_paged_prefill_chunk` needs the FULL prompt for the GDN pre-pass over
/// the cached prefix. `gdn_prefix_already_primed` is the moe-specific bit the
/// prime resolves (the GDN recurrent state was already populated live / from a
/// checkpoint / via replay) and `paged_prefill` threads into
/// `run_paged_prefill_chunk` so the prefill skips re-priming the GDN prefix.
pub(crate) struct Qwen35MoePrefixState {
    pub(super) effective_cached_prefix_len: usize,
    pub(super) suffix_len: usize,
    pub(super) full_tokens: Vec<u32>,
    pub(super) gdn_prefix_already_primed: bool,
    pub(super) checkpoint_extra_keys: Vec<Vec<u64>>,
    pub(super) checkpoint_cache_salt: u64,
}

impl PagedPrefix for Qwen35MoePrefixState {
    fn effective_cached_prefix_len(&self) -> usize {
        self.effective_cached_prefix_len
    }
    fn suffix_len(&self) -> usize {
        self.suffix_len
    }
}

impl Qwen35MoeInner {
    /// The `prepare_turn_…` + `prepare_moe_gdn_prefix_state` block that
    /// opens a MoE paged turn. `dflash2_hit` is a DFlash2 turn's reusable
    /// prefix: the prepare then books nothing past it.
    pub(crate) fn prime_moe_prefix(
        &mut self,
        plan: &[u32],
        cache_salt: u64,
        dflash2_hit: Option<crate::models::qwen3_5::dflash2_turn::DFlash2Hit>,
    ) -> Result<Qwen35MoePrefixState> {
        self.paged_finalize_failed = false;
        let trace_enabled = inference_trace_enabled();
        let total_budget = plan.len() as u32;
        // vLLM exact-prefix cap: leave at least one prompt token to prefill so
        // the decoder always has something to consume.
        let max_cache_hit_tokens = total_budget.saturating_sub(1);
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
        let keep = crate::models::qwen3_5::dflash2_turn::prime_keep(
            plan,
            &lookup_extra_keys,
            cache_salt,
            dflash2_hit,
        );
        let turn_plan = crate::models::qwen3_5::dflash2_turn::claimed(self, keep, |inner| {
            let adapter = inner
                .paged_adapter
                .as_mut()
                .ok_or_else(|| Error::from_reason("prime_prefix_state: paged_adapter is None"))?;
            let Some(hit) = dflash2_hit else {
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
        })?;
        let cached_prefix_len = turn_plan.cached_prefix_len;
        let continued_live_prefix = turn_plan.continued_live_prefix;
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] qwen3.5-moe paged_prefix_lookup prompt_tokens={} \
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

        self.finish_moe_prime(
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
    /// `tokens` with its logits dropped, as at a turn start; a DFlash2 lane's
    /// prefill is tapped and leaves its draft window in the turn state.
    pub(crate) fn replay_moe_serial_lane(
        &mut self,
        tokens: &[u32],
        cache_salt: u64,
        plan: &crate::transformer::paged_kv_cache_adapter::PagedTurnPlan,
        decode: crate::models::qwen3_5::dflash2_turn::SerialDecode,
        stored: Option<crate::models::qwen3_5::dflash2_state::store::EntryId>,
    ) -> Result<()> {
        let block_size = self
            .paged_adapter
            .as_ref()
            .map(PagedKVCacheAdapter::block_size)
            .ok_or_else(|| Error::from_reason("a suspended lane lost its paged adapter"))?;
        let keys = engine::build_paged_extra_keys(tokens.len(), block_size, &[]);
        let prefix = self.finish_moe_prime(tokens, cache_salt, block_size, &keys, false, plan)?;
        let cached = prefix.effective_cached_prefix_len;
        let suffix = tokens
            .get(cached..)
            .ok_or_else(|| Error::from_reason("a replay's prefix passed its tokens"))?;
        let logits = match decode {
            crate::models::qwen3_5::dflash2_turn::SerialDecode::Autoregressive => {
                self.moe_paged_prefill(suffix, &prefix, None)?.0
            }
            crate::models::qwen3_5::dflash2_turn::SerialDecode::DFlash2 => {
                if cached == 0 {
                    self.dflash2_context = None;
                }
                crate::models::qwen3_5::dflash2_turn::use_resumed_replay_point(
                    self, tokens, stored, cached,
                );
                crate::engine::dspark_turn::PagedDsparkBackend::paged_prefill_with_draft_state(
                    self,
                    suffix,
                    &prefix,
                    Stream::generation(),
                )?
            }
        };
        logits.eval();
        crate::array::synchronize_and_clear_cache();
        Ok(())
    }

    /// The prime after the adapter's prepare: the GDN state over
    /// `turn_plan`'s cached prefix, the turn's session state, and the
    /// prefix the prefill continues from.
    pub(crate) fn finish_moe_prime(
        &mut self,
        plan: &[u32],
        cache_salt: u64,
        block_size: u32,
        lookup_extra_keys: &[Vec<u64>],
        carries_image_lineage: bool,
        turn_plan: &crate::transformer::paged_kv_cache_adapter::PagedTurnPlan,
    ) -> Result<Qwen35MoePrefixState> {
        let cached_prefix_len = turn_plan.cached_prefix_len;
        let continued_live_prefix = turn_plan.continued_live_prefix;
        let total_budget = plan.len() as u32;
        // GDN recurrent-state prime (live / checkpoint / replay). No qwen3/lfm2
        // analog — moe carries GDN recurrent state across turns.
        let gdn_prefix_preparation = self.prepare_moe_gdn_prefix_state(
            plan,
            cached_prefix_len,
            block_size,
            lookup_extra_keys,
            cache_salt,
            continued_live_prefix,
        )?;
        let gdn_prefix_already_primed = gdn_prefix_preparation.already_primed;
        // GDN recurrent state now covers exactly the cached prefix (installed
        // live, restored from a cold sidecar, or replayed just above). Discharge
        // the adapter's auxiliary-state obligation — set when the cold tier
        // carries a `ColdSidecarPolicy` and a HOT K/V hit handed back a prefix no
        // sidecar was gated against — before the turn's first `record_tokens`. A
        // no-op for a live continuation or a sidecar-backed restore.
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

        Ok(Qwen35MoePrefixState {
            effective_cached_prefix_len: cached_prefix_len as usize,
            suffix_len,
            full_tokens: plan.to_vec(),
            gdn_prefix_already_primed,
            checkpoint_extra_keys: lookup_extra_keys.to_vec(),
            checkpoint_cache_salt: cache_salt,
        })
    }

    /// The paged prefill; with `draft`, the draft context advanced over the
    /// prompt from the same forward's taps.
    pub(crate) fn moe_paged_prefill(
        &mut self,
        suffix_tokens: &[u32],
        prefix: &Qwen35MoePrefixState,
        draft: Option<crate::models::qwen3_5::dflash2::DFlash2ContextCache>,
    ) -> Result<(
        MxArray,
        Option<crate::models::qwen3_5::dflash2::DFlash2ContextCache>,
    )> {
        // The paged prefill block. `run_paged_prefill_chunk` writes K/V into
        // the adapter pool, populates the GDN linear caches, runs the GDN
        // pre-pass over the cached prefix from `full_tokens` (skipped when
        // `gdn_prefix_already_primed`), then the full forward over the suffix,
        // folding in the last-token slice (returns `[vocab]`). The engine
        // fires the post-prefill `synchronize_and_clear_cache` AFTER this
        // returns (NOT here).
        let layer_kinds = crate::models::qwen3_5::decoder_layer::compute_layer_kinds(
            self.config.num_layers as usize,
            |i| self.config.is_linear_layer(i),
        );
        let embed = self.embedding.clone();
        // Cross-turn M-RoPE delta (0 unless this engine-driven text turn warm-
        // continues an image prefill); aligns the suffix keys with the
        // compressed-position image keys.
        let rope_deltas = self.cached_rope_deltas.unwrap_or(0);
        // Split the prefill at the checkpoint ladder when persistence is on so
        // the GDN recurrent state is materialized at every boundary the cold
        // sidecar could anchor on. Byte-inert with no cold GDN policy — the
        // unchanged single-shot default. See `cold_gdn_prefill_chunk_size`.
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
            let capture = crate::models::qwen3_5::dflash2_turn::replay_capture_row(
                self,
                &prefix.full_tokens,
                prefix.effective_cached_prefix_len,
            );
            // The draft leaves the model for the prefill so each chunk can book
            // through the take order on the whole model.
            let model = self
                .dflash2
                .take()
                .ok_or_else(|| Error::from_reason("Qwen3.5 MoE has no loaded DFlash2 companion"))?;
            let mut sink = crate::models::qwen3_5::dflash2_turn::DraftTapSink::new(
                &model,
                context,
                &prefix.full_tokens,
                prefix.effective_cached_prefix_len,
                capture,
            );
            let prefilled =
                crate::models::qwen3_5_moe::paged_forward::run_paged_prefill_chunk_tapped(
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
                    None,
                    Some(
                        &mut sink as &mut dyn crate::models::qwen3_5::paged_forward::PrefillTapSink,
                    ),
                );
            let (context, anchor) = sink.into_parts();
            self.dflash2 = Some(model);
            let (logits, gdn_checkpoint) = prefilled?;
            self.publish_moe_gdn_materialized_prefix_checkpoint(
                &prefix.full_tokens,
                &prefix.checkpoint_extra_keys,
                prefix.checkpoint_cache_salt,
                gdn_checkpoint,
            );
            if let Some(anchor) = anchor {
                crate::models::qwen3_5::dflash2_turn::publish_replay_state(
                    self,
                    &prefix.full_tokens,
                    prefix.checkpoint_cache_salt,
                    anchor,
                );
            }
            return Ok((logits, Some(context)));
        }
        let (logits, gdn_checkpoint) = {
            let caches_ref = self
                .caches
                .as_mut()
                .ok_or_else(|| Error::from_reason("paged_prefill: caches not initialized"))?;
            let adapter = self
                .paged_adapter
                .as_mut()
                .ok_or_else(|| Error::from_reason("paged_prefill: paged_adapter dropped"))?;
            crate::models::qwen3_5_moe::paged_forward::run_paged_prefill_chunk_tapped(
                &prefix.full_tokens,
                suffix_tokens,
                prefix.effective_cached_prefix_len as u32,
                prefix.gdn_prefix_already_primed,
                &embed,
                &mut crate::models::qwen3_5::paged_forward::PrefillParts {
                    layers: &mut self.layers,
                    caches: caches_ref,
                    final_norm: &self.final_norm,
                    lm_head: &self.lm_head,
                    adapter,
                },
                &layer_kinds,
                chunk_size,
                rope_deltas,
                turn_cancel.as_deref(),
                mtp_seq.map(|_| &mut mtp_hidden),
                None,
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
        self.publish_moe_gdn_materialized_prefix_checkpoint(
            &prefix.full_tokens,
            &prefix.checkpoint_extra_keys,
            prefix.checkpoint_cache_salt,
            gdn_checkpoint,
        );
        Ok((logits, None))
    }
}

impl crate::models::qwen3_5::paged_forward::PrefillTarget<DecoderLayer> for Qwen35MoeInner {
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
}

impl PagedBackend for Qwen35MoeInner {
    type PagedDecode<'a>
        = PagedStepper<Qwen35MoePagedDecode<'a>>
    where
        Self: 'a;
    type PrefixState = Qwen35MoePrefixState;

    fn prime_prefix_state(
        &mut self,
        plan: &[u32],
        _reuse_cache: bool,
        _block_size: usize,
        _extra_keys: &[u64],
        cache_salt: u64,
    ) -> Result<Self::PrefixState> {
        self.invalidate_dflash2_context("prime_prefix_state");
        self.prime_moe_prefix(plan, cache_salt, None)
    }

    fn paged_prefill(
        &mut self,
        suffix_tokens: &[u32],
        prefix: &Self::PrefixState,
        _stream: Stream,
    ) -> Result<MxArray> {
        self.moe_paged_prefill(suffix_tokens, prefix, None)
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
        Ok(PagedStepper(Qwen35MoePagedDecode { inner: self }))
    }

    fn admit_paged_speculative_decode(&mut self, p: &engine::ChatParams) -> Result<bool> {
        // MTP acceptance gate: a previous turn whose draft head accepted below
        // break-even decodes THIS turn plain AR. It runs after prefill, where
        // the AR fallback below is still a plain loop rather than a failed
        // turn.
        if !p.mtp_adaptive_depth && !self.mtp_gate_allows(p.mtp_depth.max(1) as u32) {
            return Ok(false);
        }
        let Some(plan) = self.execution_plan().speculative else {
            // The plan is what selected a speculative decoder, so this is
            // unreachable; declining only restores plain AR decode.
            debug_assert!(
                false,
                "MoE paged speculative admission without a speculative plan"
            );
            return Ok(false);
        };
        // Pre-cycle lookahead reservation (I1) while AR fallback is still an
        // option — allocator exhaustion inside the verify loop would instead
        // surface as a turn error and invalidate the paged session.
        let rows = u32::try_from(crate::engine::mtp_turn::turn_lookahead_rows(plan, p))
            .unwrap_or(u32::MAX);
        let reserved = crate::models::qwen3_5::dflash2_turn::lookahead_claimed(self, |inner| {
            inner
                .paged_adapter
                .as_mut()
                .ok_or_else(|| {
                    Error::from_reason("MoE paged speculative admission: paged_adapter is None")
                })?
                .reserve_rows(rows)
                .map(drop)
                .map_err(Error::from_reason)
        })
        .map_err(|e| {
            Error::from_reason(format!(
                "MoE paged speculative lookahead reservation: {}",
                e.reason
            ))
        })?;
        if !reserved {
            tracing::warn!(
                target: "mlx_core::qwen3_5_moe::paged",
                lookahead_rows = rows,
                "MoE paged speculative lookahead reservation refused; falling back to \
                 autoregressive decode"
            );
        }
        Ok(reserved)
    }

    fn run_paged_speculative_decode(
        &mut self,
        args: crate::engine::backend::PagedSpeculativeArgs<'_, '_, '_>,
    ) -> Result<()> {
        // Pure-Rust eager paged MoE MTP. The main Step-A / verify forwards
        // route through the paged adapter (`run_paged_step_with_hidden` /
        // `run_paged_verify_step`); the GDN recurrent state stays FLAT in
        // `self.caches` Linear slots, so the tape replay (the rollback
        // keystone) is IDENTICAL to the flat eager arm. Full-attention K/V
        // lives in the paged pool, so the rollback rewinds it through the
        // facade cycle, NOT a `self.caches` KV trim.
        let crate::engine::backend::PagedSpeculativeArgs {
            y,
            prompt_len,
            params,
            reasoning_tracker,
            profiler,
            max_new_tokens,
            eos_id,
            generated_tokens,
            token_history,
            finish_reason,
            first_token_instant,
            report_perf,
            generation_stream,
            cancel_flag,
            streaming,
        } = args;
        MxArray::async_eval_arrays(&[&y]);
        let mut rng = rand::rng();
        let outcome = crate::engine::mtp_turn::run_mtp_turn(
            self,
            &mut rng,
            crate::engine::mtp_turn::MtpTurnArgs {
                y,
                depth: params.mtp_depth,
                params,
                reasoning_tracker,
                profiler,
                max_new_tokens,
                eos_id,
                // Reborrowed so the frontier check below can still read the
                // emitted count once the loop returns.
                generated_tokens: &mut *generated_tokens,
                token_history,
                finish_reason,
                first_token_instant,
                report_perf,
                generation_stream,
                // No MoE hidden-emitting prefill exists, so the drafter builds
                // cycle history from decode tokens alone.
                prompt_hidden: None,
                prompt_hidden_ids: None,
                cancel_flag,
            },
            streaming,
        )?;
        self.paged_mtp_last_rollback_unemitted = outcome.rollback_unemitted;
        debug_assert!(
            !outcome.desynced,
            "the paged MoE MTP stepper must rewind both state kinds itself and never \
             arm the flat desync latch"
        );
        // At the epilogue boundary: the adapter and the drop-last history the
        // epilogue is about to persist must sit at ONE frontier before any GDN
        // state is keyed on that history. Runs BEFORE `finish_paged_turn`
        // reconciles, because reconcile would roll the adapter back to the
        // history length without moving the recurrent state with it and hide
        // exactly the skew this catches.
        let expected_history_len = prompt_len + generated_tokens.len().saturating_sub(1);
        self.check_moe_paged_frontier(expected_history_len, "paged MTP epilogue");
        Ok(())
    }

    fn finalize_paged_turn(&mut self, reuse_cache: bool, cache_salt: u64) {
        // Terminal lifecycle block of a paged turn. Success: keep the request
        // live across turns when reuse is on, using PER-BLOCK extra keys (NOT
        // qwen3's empty `&[]`), so the next turn's continue builds on the
        // partial trailing block's live K/V; otherwise register full blocks
        // for reuse + release. The trait hook remains infallible, but an adapter
        // error must downgrade the session before the engine's unconditional
        // history save can publish image-placeholder history.
        self.paged_finalize_failed = false;
        self.paged_turn_seq = self
            .paged_adapter
            .as_ref()
            .and_then(PagedKVCacheAdapter::active_seq_id);
        let mut release_pending = false;
        let mut finalize_error = match self.paged_adapter.as_mut() {
            Some(adapter) if reuse_cache => {
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
                // The release is attempted below (even when registration fails),
                // but only AFTER the GDN sidecar capture — releasing resets the
                // adapter's cold-chain frontier to 0.
                release_pending = true;
                adapter
                    .register_full_blocks_for_reuse_per_block(&finalize_extra_keys, cache_salt)
                    .err()
            }
            None => Some("paged_adapter is None during MoE finalization".to_owned()),
        };
        // Persist the out-of-pool GDN recurrent state for the SAME chain the
        // adapter just captured (its `cold_captured_blocks` is now set), so a
        // later process can resume from the restored K/V prefix instead of
        // replaying GDN over it. Runs BEFORE the release below, which resets the
        // captured-chain frontier to 0. Skipped when the finalize failed: the
        // K/V chain the sidecar would anchor on was not published, so nothing
        // could ever select it.
        if finalize_error.is_none() {
            self.capture_moe_gdn_cold_sidecar(&self.cached_paged_image_token_positions, cache_salt);
        }
        // Always attempt the release for the non-reuse path, even when
        // registration failed (mirrors the prior `register_error.or(release)`).
        if release_pending && let Some(adapter) = self.paged_adapter.as_mut() {
            finalize_error = finalize_error.or(adapter.release_request().err());
        }
        crate::models::qwen3_5::dflash2_turn::settle_replay_state(self, finalize_error.is_none());
        if let Some(error) = finalize_error {
            self.downgrade_failed_paged_finalize(&error);
        }
    }

    fn abort_paged_turn(&mut self) {
        // Hybrid error-path teardown: K/V and GDN may have advanced by
        // different amounts, so a bare adapter release would leave a false-live
        // recurrent session behind. Never register / keep live; invalidate all
        // model-local continuation state without masking the original error.
        self.invalidate_moe_paged_session("generic paged turn abort");
    }

    fn paged_decode_stream(&self, _generation_stream: Stream) -> Stream {
        // DECODE runs on the canonical DEFAULT stream, NOT the per-turn
        // `generation_stream`: the shared loop's top-of-iteration `y.eval()` is
        // always on the default stream, so a separate queue would force a
        // cross-queue completion-wait every token. `paged_prefill` still runs on
        // `generation_stream`. See the `PagedBackend::paged_decode_stream` doc.
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
        if std::mem::take(&mut self.paged_finalize_failed) {
            self.cached_token_history.clear();
            self.cached_image_key = None;
            self.cached_paged_image_token_positions.clear();
            self.cached_rope_deltas = None;
            self.gdn_last_history_checkpoint = None;
            self.caches = None;
            return Ok(());
        }
        // moe paged ALWAYS drops the last token, regardless of the engine's
        // `keep_all` (length-exit) signal — the paged decode loop NEVER forwards
        // the LAST sampled token (the engine's forward gate skips it AND
        // `materialize_final` is a no-op for moe), so the last `generated` entry
        // is NOT in the adapter / GDN caches and must be dropped to keep the
        // saved history aligned with the live cache state. Ordering:
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
        // GDN history checkpoint — must run AFTER the history is set (it
        // snapshots the live recurrent caches keyed on `cached_token_history`),
        // BEFORE clearing the image key. A checkpoint/eval failure here PROPAGATES
        // (`?`) to abort the turn: a half-snapshotted or failed-eval GDN state
        // must NOT be published as a reusable warm-continue checkpoint, or the
        // next turn reads corrupt recurrent state.
        let store = self.remember_moe_gdn_history_checkpoint()?;
        if let Some(seq_id) = turn_seq {
            self.end_gdn_use(seq_id);
        }
        if inference_trace_enabled() {
            write_inference_trace(format_args!(
                "[MLX_TRACE] qwen3.5-moe gdn_history_checkpoint stored={} tokens={} \
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
        // moe ALWAYS drops the last token (see `save_paged_history`), so the
        // to-be-saved history length is `prompt_len + (generated.len() - 1)` (or
        // `prompt_len` when nothing was generated). Roll the adapter back to that
        // length so the next turn's warm-continue gate
        // (`prompt.starts_with(request_tokens())`) is not defeated by a trailing
        // token the pipelined loop recorded at the loop top before the
        // stop-check. `_keep_all` is intentionally ignored (qwen3 signal).
        //
        // Token accounting: on BOTH length and early-stop exits the to-be-saved
        // history equals the adapter cursor (the final/terminal forward was
        // skipped), so `surplus` is 0 and this is a true no-op for moe — but the
        // rollback is kept as the defensive contract the trait mandates.
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
                target: "mlx_core::qwen3_5_moe::paged",
                "reconcile_paged_request_tokens: rollback_last_tokens({surplus}) failed \
                 (finalize releases the request; next turn cold-prefills): {e}",
            );
            return false;
        }
        true
    }
}

impl Qwen35MoeInner {
    /// The tokenizer, params and perf flag an image turn decodes with: the
    /// request config under the checkpoint's generation defaults, with the
    /// planned decoder projected in.
    pub(super) fn vision_turn_params(
        &self,
        args: &WholeTurnArgs<'_>,
    ) -> Result<(Arc<Qwen3Tokenizer>, engine::ChatParams, bool)> {
        let mut config = args.config.clone();
        crate::engine::apply_generation_defaults(&mut config, &self.gen_defaults);
        apply_qwen35_moe_planned_decoder(&mut config, args.plan.decoder);
        let mut p = engine::extract_chat_params(&config);
        p.extra_eos_ids = self.gen_defaults.eos_token_ids.clone();
        let tokenizer = self
            .tokenizer
            .clone()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))?;
        Ok((tokenizer, p, config.report_performance.unwrap_or(false)))
    }
}
