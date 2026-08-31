//! Hand-written TEXT paged turn cores for Qwen3.5 MoE — the sync and
//! streaming pairs that own every planned-MTP paged turn.

use super::*;

impl Qwen35MoeInner {
    /// Block-paged variant of [`Self::vision_mtp_whole_turn_core`] for the MoE
    /// model. Mirrors the dense paged dispatch — see
    /// `Qwen35Inner::paged_turn_sync_core` for the full rationale.
    ///
    /// The paged decode loop runs the pure-Rust paged forward
    /// (`paged_forward::run_paged_decode_step`): it reads K/V from the
    /// adapter pool via `paged_kv_write` / `paged_attention` and reads GDN
    /// linear caches from the per-layer
    /// `Qwen3_5LayerCache::Linear(ArraysCache)`.
    pub(super) fn paged_turn_sync_core(
        &mut self,
        tokens: Vec<u32>,
        tokenizer: Arc<Qwen3Tokenizer>,
        eos_token_id: u32,
        mut p: engine::ChatParams,
        report_perf: bool,
        thinking: ThinkingSetup,
    ) -> Result<ChatResult> {
        if tokens.is_empty() {
            return Err(Error::from_reason("Empty prompt"));
        }
        self.preflight_paged_context(tokens.len(), &mut p)?;

        let prompt_token_count = tokens.len() as u32;
        let trace_enabled = inference_trace_enabled();
        let sampling_config = p.sampling_config;

        let think_end_id = tokenizer.think_end_id();
        let think_end_str = tokenizer.think_end_str().map(|s| s.to_string());
        // Thinking is resolved ONCE at turn entry and honors
        // `enable_thinking=false`.
        let thinking_enabled = thinking.enabled;
        let mut reasoning_tracker = engine::ReasoningTracker::from_setup(&thinking, think_end_id);

        let generation_start = if report_perf {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut first_token_instant: Option<std::time::Instant> = None;

        let seq_id = self.active_scheduled_seq.unwrap_or(0);
        // Lazy decode allocation: pass the prompt length only.
        let total_budget = tokens.len() as u32;
        // Per-block extra_keys: text-only paged dispatch builds an all-empty
        // per-block vec, bit-equal to passing `&[]` to the uniform API.
        // VLM-paged replaces the empty positions with real
        // (token_pos, image_hash) pairs.
        let block_size = {
            let adapter = self.paged_adapter.as_ref().ok_or_else(|| {
                Error::from_reason("MoE paged_turn_sync_core: paged_adapter is None")
            })?;
            adapter.block_size()
        };
        let carries_image_lineage = self.cached_image_key.is_some()
            && !self.cached_paged_image_token_positions.is_empty()
            && !self.cached_token_history.is_empty()
            && tokens.starts_with(&self.cached_token_history);
        let image_positions = if carries_image_lineage {
            self.cached_paged_image_token_positions.as_slice()
        } else {
            &[]
        };
        let lookup_extra_keys =
            engine::build_paged_extra_keys(tokens.len(), block_size, image_positions);
        let cache_salt = p.cache_salt;
        // vLLM exact-prefix cap: leave at least one prompt token to prefill.
        let max_cache_hit_tokens = total_budget.saturating_sub(1);
        let live_ready;
        let live_prefix_match;
        let live_tokens_len;
        let mut live_mismatch = TokenPrefixMismatchTrace::default();
        // Adapter-owned warm/cold lifecycle. The [MLX_TRACE] line below
        // reads the PRE-turn live state, so probe the adapter immutably FIRST
        // (prepare_turn mutates request_tokens via continue_turn/reset). The
        // adapter re-reads the same state internally, so live_* is identical to
        // what prepare_turn decides on. extra_keys=&[] (uniform API) is bit-equal
        // to `&lookup_extra_keys` for text-only dispatch (all-empty per-block
        // vec → identical hashes; see the block_size comment above).
        // reuse_cache=true: continuation eligibility carries no reuse term.
        // Suffix blocks are allocated inside prepare_turn.
        {
            let adapter = self.paged_adapter.as_ref().ok_or_else(|| {
                Error::from_reason("MoE paged_turn_sync_core: paged_adapter is None")
            })?;
            live_ready = adapter.is_live_for_continue();
            let live_tokens = adapter.request_tokens();
            live_tokens_len = live_tokens.len();
            live_prefix_match = tokens.starts_with(live_tokens);
            if trace_enabled && live_ready && !live_prefix_match {
                live_mismatch = token_prefix_mismatch_trace(&tokens, live_tokens);
            }
        }
        let plan_result = self
            .paged_adapter
            .as_mut()
            .ok_or_else(|| Error::from_reason("MoE paged_turn_sync_core: paged_adapter is None"))?
            .prepare_turn_per_block_with_max_cache_hit_tokens(
                seq_id,
                &tokens,
                total_budget,
                true,
                &lookup_extra_keys,
                cache_salt,
                false,
                max_cache_hit_tokens,
            )
            .map_err(Error::from_reason);
        let plan = match plan_result {
            Ok(plan) => plan,
            Err(error) => {
                self.invalidate_moe_paged_session("manual adapter preparation failure");
                return Err(error);
            }
        };
        let cached_prefix_len = plan.cached_prefix_len;
        let continued_live_prefix = plan.continued_live_prefix;
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] qwen3.5-moe paged_prefix_lookup prompt_tokens={} \
                 cached_prefix_tokens={} continued_live_prefix={} live_ready={} \
                 live_match={} live_tokens={} live_mismatch_at={} prompt_token={} live_token={}",
                tokens.len(),
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

        let gdn_prefix_preparation = match self.prepare_moe_gdn_prefix_state(
            &tokens,
            cached_prefix_len,
            block_size,
            &lookup_extra_keys,
            cache_salt,
            continued_live_prefix,
        ) {
            Ok(preparation) => preparation,
            Err(error) => {
                self.invalidate_moe_paged_session("manual GDN-prefix preparation failure");
                return Err(error);
            }
        };
        let gdn_prefix_already_primed = gdn_prefix_preparation.already_primed;
        // Discharge the adapter's auxiliary (GDN) prefix obligation before the
        // turn's first `record_tokens`. See `prime_prefix_state` for the full
        // rationale; a no-op unless a HOT K/V hit handed back a prefix the cold
        // tier's `ColdSidecarPolicy` was not gated against.
        let confirm_result = self
            .paged_adapter
            .as_mut()
            .map(|adapter| adapter.confirm_aux_prefix_primed(cached_prefix_len))
            .unwrap_or(Ok(()));
        if let Err(error) = confirm_result {
            self.invalidate_moe_paged_session("planned-MTP GDN auxiliary-prefix confirmation");
            return Err(Error::from_reason(error));
        }
        let preserves_image_lineage = carries_image_lineage && cached_prefix_len > 0;
        self.cached_token_history.clear();
        if !preserves_image_lineage {
            self.cached_image_key = None;
            self.cached_paged_image_token_positions.clear();
        }
        // Carry the cross-turn M-RoPE delta only when this turn extends the live
        // image sequence (continued_live_prefix); a cold start or a non-live
        // prefix-cache hit (text-only prefix) drops a stale image delta so the
        // text suffix prefill + decode rotate at the raw physical slot.
        self.cached_rope_deltas = crate::models::qwen3_5::paged_forward::rope_delta_for_paged_turn(
            self.cached_rope_deltas,
            preserves_image_lineage,
        );

        let suffix_len = match prompt_token_count.checked_sub(cached_prefix_len) {
            Some(suffix_len) => suffix_len,
            None => {
                self.invalidate_moe_paged_session("manual prefix length mismatch");
                return Err(Error::from_reason(
                    "MoE paged_turn_sync_core: cached_prefix_len > total_prompt_tokens",
                ));
            }
        };

        let forward_result = self.paged_turn_sync_core_inner(
            &tokens,
            cached_prefix_len,
            suffix_len,
            &p,
            eos_token_id,
            &sampling_config,
            &mut reasoning_tracker,
            report_perf,
            &mut first_token_instant,
            gdn_prefix_already_primed,
            &lookup_extra_keys,
            cache_salt,
        );

        let (generated_tokens, finish_reason) = match forward_result {
            Ok(t) => {
                let image_token_positions = self.cached_paged_image_token_positions.clone();
                self.finalize_moe_manual_paged_turn(&image_token_positions, cache_salt)?;
                t
            }
            Err(e) => {
                self.invalidate_moe_paged_session("manual sync prefill/decode failure");
                return Err(e);
            }
        };

        let last_token_in_cache = false;
        let mut full_history = tokens.clone();
        if !generated_tokens.is_empty() {
            let upto = if last_token_in_cache {
                generated_tokens.len()
            } else {
                generated_tokens.len().saturating_sub(1)
            };
            full_history.extend_from_slice(&generated_tokens[..upto]);
        }
        self.cached_token_history = full_history;
        let gdn_history_checkpoint_store = match self.remember_moe_gdn_history_checkpoint() {
            Ok(store) => store,
            Err(error) => {
                self.invalidate_moe_paged_session("manual sync GDN history checkpoint failure");
                return Err(error);
            }
        };
        if inference_trace_enabled() {
            write_inference_trace(format_args!(
                "[MLX_TRACE] qwen3.5-moe gdn_history_checkpoint stored={} tokens={} \
                 eval_ms={:.1} clone_ms={:.1} token_clone_ms={:.1} update_ms={:.1} total_ms={:.1}",
                gdn_history_checkpoint_store.stored,
                self.cached_token_history.len(),
                gdn_history_checkpoint_store.eval_ms,
                gdn_history_checkpoint_store.clone_ms,
                gdn_history_checkpoint_store.token_clone_ms,
                gdn_history_checkpoint_store.update_ms,
                gdn_history_checkpoint_store.total_ms
            ));
        }

        let performance = if report_perf {
            compute_performance_metrics(
                generation_start,
                first_token_instant,
                tokens.len() - cached_prefix_len as usize,
                generated_tokens.len(),
            )
        } else {
            None
        };

        let mut result = finalize_chat_result(
            &tokenizer,
            &generated_tokens,
            finish_reason,
            think_end_id,
            think_end_str.as_deref(),
            performance,
            p.include_reasoning,
            thinking_enabled,
            prompt_token_count,
            reasoning_tracker.reasoning_token_count(),
        )?;
        result.cached_tokens = cached_prefix_len;
        Ok(result)
    }

    #[allow(clippy::too_many_arguments)]
    fn paged_turn_sync_core_inner(
        &mut self,
        tokens: &[u32],
        cached_prefix_len: u32,
        suffix_len: u32,
        p: &engine::ChatParams,
        eos_token_id: u32,
        sampling_config: &Option<crate::sampling::SamplingConfig>,
        reasoning_tracker: &mut engine::ReasoningTracker,
        report_perf: bool,
        first_token_instant: &mut Option<std::time::Instant>,
        gdn_prefix_already_primed: bool,
        checkpoint_extra_keys: &[Vec<u64>],
        checkpoint_cache_salt: u64,
    ) -> Result<(Vec<u32>, String)> {
        // Invariant: caller-applied vLLM cap guarantees suffix_len > 0.
        debug_assert!(
            suffix_len > 0,
            "MoE paged_turn_sync_core_inner: caller must cap max_cache_hit_tokens at prompt.len() - 1"
        );

        // Clone the backend-installed per-turn cancel flag up front —
        // the decode loop below borrows `self` mutably.
        let turn_cancel = self.turn_cancel.clone();

        let suffix = &tokens[(cached_prefix_len as usize)..];
        let layer_kinds = crate::models::qwen3_5::decoder_layer::compute_layer_kinds(
            self.config.num_layers as usize,
            |i| self.config.is_linear_layer(i),
        );

        // Pure-Rust paged prefill: writes K/V into the adapter pool per
        // layer via `update_keys_values_native` (graph-native `PagedKVWrite`
        // primitive — the default; the synchronous raw-Metal
        // `update_keys_values` is the error/opt-out fallback) and populates
        // the GDN linear caches in `Qwen3_5LayerCache::Linear(ArraysCache)`.
        // Both are exactly what the pure-Rust paged decode steps
        // (`paged_forward::run_paged_decode_step`) read as inputs.
        let (last_logits, gdn_checkpoint) = self.run_moe_core_paged_prefill(
            tokens,
            suffix,
            cached_prefix_len,
            gdn_prefix_already_primed,
            &layer_kinds,
            "MoE paged_turn_sync_core_inner",
        )?;
        self.publish_moe_gdn_materialized_prefix_checkpoint(
            tokens,
            checkpoint_extra_keys,
            checkpoint_cache_salt,
            gdn_checkpoint,
        );

        let mut token_history: Vec<u32> = tokens.to_vec();
        let last_logits = apply_all_penalties(last_logits, &token_history, p)?;
        let mut y = sample(&last_logits, *sampling_config)?;
        y.eval();

        // Smooth memory peak: drop transient prefill buffers before decode
        // starts allocating. Prefill of long prompts builds a massive MLX
        // subgraph; once we have the last logits, those intermediates are
        // dead but MLX's cache holds them.
        crate::array::synchronize_and_clear_cache();

        if report_perf {
            *first_token_instant = Some(std::time::Instant::now());
        }

        let max_new_tokens = p.max_new_tokens;
        let mut generated_tokens: Vec<u32> =
            Vec::with_capacity(engine::generated_capacity_hint(max_new_tokens));
        let mut finish_reason = String::from("length");

        for step in 0..max_new_tokens {
            let token_id = y.item_at_int32(0)? as u32;
            generated_tokens.push(token_id);
            token_history.push(token_id);
            reasoning_tracker.observe_token(token_id);

            if token_id == eos_token_id || p.extra_eos_ids.contains(&token_id) {
                finish_reason = String::from("stop");
                break;
            }
            // Sync cancel poll — the SAME snapshot point as the MoE paged
            // streaming twin (`paged_turn_stream_core_inner`): after the EOS
            // check, before the repetition cutoff.
            if turn_cancel
                .as_deref()
                .is_some_and(|flag| flag.load(Ordering::Relaxed))
            {
                finish_reason = String::from("cancelled");
                break;
            }
            if let Some(reason) = crate::sampling::check_repetition_cutoff(
                &generated_tokens,
                p.max_consecutive_tokens,
                p.max_ngram_repeats,
                p.ngram_size,
            ) {
                finish_reason = reason.to_string();
                break;
            }
            if step + 1 >= max_new_tokens {
                break;
            }

            // Pure-Rust paged decode step.
            let next_logits = {
                let embed = self.embedding.clone();
                let caches_ref = self.caches.as_mut().ok_or_else(|| {
                    Error::from_reason("MoE paged_turn_sync_core_inner: caches dropped mid-decode")
                })?;
                let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                    Error::from_reason(
                        "MoE paged_turn_sync_core_inner: paged_adapter dropped mid-decode",
                    )
                })?;
                let logits = crate::models::qwen3_5_moe::paged_forward::run_paged_decode_step(
                    token_id,
                    &embed,
                    &mut self.layers,
                    caches_ref,
                    &self.final_norm,
                    &self.lm_head,
                    &layer_kinds,
                    adapter,
                    self.cached_rope_deltas.unwrap_or(0),
                )?;
                logits.squeeze(Some(&[1]))?
            };

            let next_logits = if reasoning_tracker.should_force_think_end() {
                let forced_id = reasoning_tracker.forced_token_id()? as i32;
                y = MxArray::from_int32(&[forced_id], &[1])?;
                y.eval();
                continue;
            } else {
                apply_all_penalties(next_logits, &token_history, p)?
            };

            y = sample(&next_logits, *sampling_config)?;
            y.eval();

            crate::array::maybe_clear_cache_for_paged_step(step);
        }

        Ok((generated_tokens, finish_reason))
    }

    /// Block-paged streaming variant for MoE — mirrors dense
    /// `paged_turn_stream_core`. See [`Self::paged_turn_sync_core`]
    /// for the paged dispatch rationale (pure-Rust paged prefill + decode
    /// against the adapter pool); the streaming path uses the same
    /// adapter lifecycle + prefix-reuse semantics.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn paged_turn_stream_core(
        &mut self,
        tokens: Vec<u32>,
        tokenizer: Arc<Qwen3Tokenizer>,
        eos_token_id: u32,
        mut p: engine::ChatParams,
        report_perf: bool,
        cb: &StreamSender<'_>,
        cancelled: &AtomicBool,
        thinking: ThinkingSetup,
    ) -> Result<()> {
        if tokens.is_empty() {
            return Err(Error::from_reason("Empty prompt"));
        }
        self.preflight_paged_context(tokens.len(), &mut p)?;

        let prompt_token_count = tokens.len() as u32;
        let trace_enabled = inference_trace_enabled();
        let request_trace_start = trace_enabled.then(std::time::Instant::now);
        let sampling_config = p.sampling_config;
        let include_reasoning = p.include_reasoning;

        let think_end_id = tokenizer.think_end_id();
        let think_end_str = tokenizer.think_end_str().map(|s| s.to_string());
        // Thinking is resolved ONCE at turn entry and honors
        // `enable_thinking=false`.
        let thinking_enabled = thinking.enabled;
        let mut reasoning_tracker = engine::ReasoningTracker::from_setup(&thinking, think_end_id);

        let generation_start = if report_perf {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut first_token_instant: Option<std::time::Instant> = None;

        let mut decode_stream = tokenizer.inner().decode_stream(true);
        let mut streamed_text_len = 0usize;
        let mut last_is_reasoning = thinking_enabled;

        let seq_id = self.active_scheduled_seq.unwrap_or(0);
        // Lazy decode allocation: pass the prompt length only.
        let total_budget = tokens.len() as u32;
        // Per-block extra_keys. See comments above.
        let block_size = {
            let adapter = self.paged_adapter.as_ref().ok_or_else(|| {
                Error::from_reason("MoE paged_turn_stream_core: paged_adapter is None")
            })?;
            adapter.block_size()
        };
        let carries_image_lineage = self.cached_image_key.is_some()
            && !self.cached_paged_image_token_positions.is_empty()
            && !self.cached_token_history.is_empty()
            && tokens.starts_with(&self.cached_token_history);
        let image_positions = if carries_image_lineage {
            self.cached_paged_image_token_positions.as_slice()
        } else {
            &[]
        };
        let lookup_extra_keys =
            engine::build_paged_extra_keys(tokens.len(), block_size, image_positions);
        let cache_salt = p.cache_salt;
        // See `paged_turn_sync_core` for the vLLM exact-prefix cap rationale.
        let max_cache_hit_tokens = total_budget.saturating_sub(1);
        let live_ready;
        let live_prefix_match;
        let live_tokens_len;
        let mut live_mismatch = TokenPrefixMismatchTrace::default();
        // Adapter-owned warm/cold lifecycle (see paged_turn_sync_core for
        // the full bit-identity rationale: pre-turn immutable probe for the
        // trace, extra_keys=&[] bit-equal to per-block for text-only,
        // reuse_cache=true, suffix blocks allocated internally).
        {
            let adapter = self.paged_adapter.as_ref().ok_or_else(|| {
                Error::from_reason("MoE paged_turn_stream_core: paged_adapter is None")
            })?;
            live_ready = adapter.is_live_for_continue();
            let live_tokens = adapter.request_tokens();
            live_tokens_len = live_tokens.len();
            live_prefix_match = tokens.starts_with(live_tokens);
            if trace_enabled && live_ready && !live_prefix_match {
                live_mismatch = token_prefix_mismatch_trace(&tokens, live_tokens);
            }
        }
        let plan_result = self
            .paged_adapter
            .as_mut()
            .ok_or_else(|| Error::from_reason("MoE paged_turn_stream_core: paged_adapter is None"))?
            .prepare_turn_per_block_with_max_cache_hit_tokens(
                seq_id,
                &tokens,
                total_budget,
                true,
                &lookup_extra_keys,
                cache_salt,
                false,
                max_cache_hit_tokens,
            )
            .map_err(Error::from_reason);
        let plan = match plan_result {
            Ok(plan) => plan,
            Err(error) => {
                self.invalidate_moe_paged_session("manual stream adapter preparation failure");
                return Err(error);
            }
        };
        let cached_prefix_len = plan.cached_prefix_len;
        let continued_live_prefix = plan.continued_live_prefix;
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] qwen3.5-moe paged_prefix_lookup prompt_tokens={} \
                 cached_prefix_tokens={} continued_live_prefix={} live_ready={} \
                 live_match={} live_tokens={} live_mismatch_at={} prompt_token={} live_token={}",
                tokens.len(),
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

        let prefill_trace_start = trace_enabled.then(std::time::Instant::now);
        let gdn_prefix_preparation = match self.prepare_moe_gdn_prefix_state(
            &tokens,
            cached_prefix_len,
            block_size,
            &lookup_extra_keys,
            cache_salt,
            continued_live_prefix,
        ) {
            Ok(preparation) => preparation,
            Err(error) => {
                self.invalidate_moe_paged_session("manual stream GDN-prefix preparation failure");
                return Err(error);
            }
        };
        let gdn_prefix_already_primed = gdn_prefix_preparation.already_primed;
        // Discharge the adapter's auxiliary (GDN) prefix obligation before the
        // turn's first `record_tokens`. See `prime_prefix_state` for the full
        // rationale; a no-op unless a HOT K/V hit handed back a prefix the cold
        // tier's `ColdSidecarPolicy` was not gated against.
        let confirm_result = self
            .paged_adapter
            .as_mut()
            .map(|adapter| adapter.confirm_aux_prefix_primed(cached_prefix_len))
            .unwrap_or(Ok(()));
        if let Err(error) = confirm_result {
            self.invalidate_moe_paged_session(
                "planned-MTP stream GDN auxiliary-prefix confirmation",
            );
            return Err(Error::from_reason(error));
        }
        let gdn_prefix_state = gdn_prefix_preparation.state;
        let preserves_image_lineage = carries_image_lineage && cached_prefix_len > 0;
        self.cached_token_history.clear();
        if !preserves_image_lineage {
            self.cached_image_key = None;
            self.cached_paged_image_token_positions.clear();
        }
        // Carry the cross-turn M-RoPE delta only when this turn extends the live
        // image sequence (continued_live_prefix); a cold start or a non-live
        // prefix-cache hit (text-only prefix) drops a stale image delta so the
        // text suffix prefill + decode rotate at the raw physical slot.
        self.cached_rope_deltas = crate::models::qwen3_5::paged_forward::rope_delta_for_paged_turn(
            self.cached_rope_deltas,
            preserves_image_lineage,
        );

        let suffix_len = match prompt_token_count.checked_sub(cached_prefix_len) {
            Some(suffix_len) => suffix_len,
            None => {
                self.invalidate_moe_paged_session("manual stream prefix length mismatch");
                return Err(Error::from_reason(
                    "MoE paged_turn_stream_core: cached_prefix_len > total_prompt_tokens",
                ));
            }
        };

        if trace_enabled {
            // The size this turn's prefill actually splits on, not the raw
            // `MLX_PAGED_PREFILL_CHUNK_SIZE`: under a GDN cold policy the two
            // differ, and reading 0 here while the same turn reports
            // `gdn_checkpoint_materialized=true` points at the wrong diagnosis.
            write_inference_trace(format_args!(
                "[MLX_TRACE] qwen3.5-moe stream_paged_start prompt_tokens={} \
                 cached_prefix_tokens={} suffix_tokens={} block_size={} \
                 prefill_chunk_size={} prefill_eval_interval={} decode_clear_interval={} \
                 gdn_prefix_state={}",
                prompt_token_count,
                cached_prefix_len,
                suffix_len,
                block_size,
                self.cold_gdn_prefill_chunk_size(),
                crate::array::paged_prefill_eval_interval(),
                crate::array::paged_decode_cache_clear_interval(),
                gdn_prefix_state
            ));
        }

        let result = self.paged_turn_stream_core_inner(
            &tokens,
            cached_prefix_len,
            suffix_len,
            &p,
            sampling_config,
            eos_token_id,
            &mut reasoning_tracker,
            report_perf,
            &mut first_token_instant,
            &tokenizer,
            &mut decode_stream,
            &mut streamed_text_len,
            &mut last_is_reasoning,
            cb,
            cancelled,
            gdn_prefix_already_primed,
            prefill_trace_start,
            &lookup_extra_keys,
            cache_salt,
        );

        if let Some(start) = request_trace_start {
            match &result {
                Ok((generated_tokens, finish_reason)) => {
                    write_inference_trace(format_args!(
                        "[MLX_TRACE] qwen3.5-moe stream_paged_done generated_tokens={} \
                         finish_reason={} elapsed_ms={:.1}",
                        generated_tokens.len(),
                        finish_reason,
                        elapsed_ms(start)
                    ));
                }
                Err(err) => {
                    write_inference_trace(format_args!(
                        "[MLX_TRACE] qwen3.5-moe stream_paged_error elapsed_ms={:.1} error={}",
                        elapsed_ms(start),
                        err
                    ));
                }
            }
        }

        let (generated_tokens, finish_reason) = match result {
            Ok(t) => {
                let image_token_positions = self.cached_paged_image_token_positions.clone();
                self.finalize_moe_manual_paged_turn(&image_token_positions, cache_salt)?;
                t
            }
            Err(e) => {
                self.invalidate_moe_paged_session("manual stream prefill/decode failure");
                return Err(e);
            }
        };

        let last_token_in_cache = false;
        let mut full_history = tokens.clone();
        if !generated_tokens.is_empty() {
            let upto = if last_token_in_cache {
                generated_tokens.len()
            } else {
                generated_tokens.len().saturating_sub(1)
            };
            full_history.extend_from_slice(&generated_tokens[..upto]);
        }
        self.cached_token_history = full_history;
        let gdn_history_checkpoint_store = match self.remember_moe_gdn_history_checkpoint() {
            Ok(store) => store,
            Err(error) => {
                self.invalidate_moe_paged_session("manual stream GDN history checkpoint failure");
                return Err(error);
            }
        };
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] qwen3.5-moe gdn_history_checkpoint stored={} tokens={} \
                 eval_ms={:.1} clone_ms={:.1} token_clone_ms={:.1} update_ms={:.1} total_ms={:.1}",
                gdn_history_checkpoint_store.stored,
                self.cached_token_history.len(),
                gdn_history_checkpoint_store.eval_ms,
                gdn_history_checkpoint_store.clone_ms,
                gdn_history_checkpoint_store.token_clone_ms,
                gdn_history_checkpoint_store.update_ms,
                gdn_history_checkpoint_store.total_ms
            ));
        }

        let full_text = tokenizer
            .decode_sync(&generated_tokens, true)
            .unwrap_or_else(|e| {
                tracing::warn!("Failed to decode generated tokens: {}", e);
                String::new()
            });
        if full_text.len() > streamed_text_len {
            let residual = full_text[streamed_text_len..].to_string();
            // Suppress residual when it is reasoning text and
            // include_reasoning == false.
            if include_reasoning || !last_is_reasoning {
                cb.call(
                    Ok(ChatStreamChunk {
                        text: residual,
                        done: false,
                        finish_reason: None,
                        tool_calls: None,
                        thinking: None,
                        thinking_enabled: None,
                        num_tokens: None,
                        prompt_tokens: None,
                        reasoning_tokens: None,
                        raw_text: None,
                        public_raw_text: None,
                        text_authoritative: None,
                        cached_tokens: None,
                        performance: None,
                        is_reasoning: Some(last_is_reasoning),
                    }),
                    ThreadsafeFunctionCallMode::NonBlocking,
                );
            }
        }

        let performance = if report_perf {
            compute_performance_metrics(
                generation_start,
                first_token_instant,
                tokens.len() - cached_prefix_len as usize,
                generated_tokens.len(),
            )
        } else {
            None
        };

        let reasoning_tokens = reasoning_tracker.reasoning_token_count();

        let mut result = finalize_chat_result(
            &tokenizer,
            &generated_tokens,
            finish_reason,
            think_end_id,
            think_end_str.as_deref(),
            performance,
            include_reasoning,
            thinking_enabled,
            prompt_token_count,
            reasoning_tokens,
        )?;
        result.cached_tokens = cached_prefix_len;

        cb.call(
            Ok(ChatStreamChunk {
                text: result.text.clone(),
                done: true,
                finish_reason: Some(result.finish_reason.clone()),
                tool_calls: Some(result.tool_calls.clone()),
                thinking: result.thinking.clone(),
                thinking_enabled: Some(result.thinking_enabled),
                num_tokens: Some(result.num_tokens),
                prompt_tokens: Some(result.prompt_tokens),
                reasoning_tokens: Some(result.reasoning_tokens),
                raw_text: Some(result.raw_text.clone()),
                public_raw_text: result.public_raw_text.clone(),
                text_authoritative: Some(true),
                cached_tokens: Some(cached_prefix_len),
                performance: result.performance.clone(),
                is_reasoning: None,
            }),
            ThreadsafeFunctionCallMode::NonBlocking,
        );

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn paged_turn_stream_core_inner<'a>(
        &mut self,
        tokens: &[u32],
        cached_prefix_len: u32,
        suffix_len: u32,
        p: &engine::ChatParams,
        sampling_config: Option<crate::sampling::SamplingConfig>,
        eos_token_id: u32,
        reasoning_tracker: &mut engine::ReasoningTracker,
        report_perf: bool,
        first_token_instant: &mut Option<std::time::Instant>,
        tokenizer: &'a Arc<Qwen3Tokenizer>,
        decode_stream: &mut tokenizers::DecodeStream<
            'a,
            tokenizers::ModelWrapper,
            tokenizers::NormalizerWrapper,
            tokenizers::PreTokenizerWrapper,
            tokenizers::PostProcessorWrapper,
            tokenizers::DecoderWrapper,
        >,
        streamed_text_len: &mut usize,
        last_is_reasoning: &mut bool,
        cb: &StreamSender<'_>,
        cancelled: &AtomicBool,
        gdn_prefix_already_primed: bool,
        prefill_trace_start: Option<std::time::Instant>,
        checkpoint_extra_keys: &[Vec<u64>],
        checkpoint_cache_salt: u64,
    ) -> Result<(Vec<u32>, String)> {
        // Invariant: caller-applied vLLM cap guarantees suffix_len > 0.
        debug_assert!(
            suffix_len > 0,
            "MoE paged_turn_stream_core_inner: caller must cap max_cache_hit_tokens at prompt.len() - 1"
        );

        let trace_enabled = inference_trace_enabled();
        let suffix = &tokens[(cached_prefix_len as usize)..];
        let layer_kinds = crate::models::qwen3_5::decoder_layer::compute_layer_kinds(
            self.config.num_layers as usize,
            |i| self.config.is_linear_layer(i),
        );

        // Pure-Rust paged prefill — see `paged_turn_sync_core_inner` for
        // the data-flow contract this populates (pool K/V + GDN linear
        // caches).
        let (last_logits, gdn_checkpoint) = self.run_moe_core_paged_prefill(
            tokens,
            suffix,
            cached_prefix_len,
            gdn_prefix_already_primed,
            &layer_kinds,
            "MoE paged_turn_stream_core_inner",
        )?;
        self.publish_moe_gdn_materialized_prefix_checkpoint(
            tokens,
            checkpoint_extra_keys,
            checkpoint_cache_salt,
            gdn_checkpoint,
        );

        let mut token_history: Vec<u32> = tokens.to_vec();
        let last_logits = apply_all_penalties(last_logits, &token_history, p)?;
        let mut y = sample(&last_logits, sampling_config)?;
        y.eval();

        // Smooth memory peak: drop transient prefill buffers before decode
        // starts allocating (see paged_turn_sync_core_inner for rationale).
        crate::array::synchronize_and_clear_cache();

        if let Some(start) = prefill_trace_start {
            write_inference_trace(format_args!(
                "[MLX_TRACE] qwen3.5-moe paged_first_token_ready prompt_tokens={} \
                 cached_prefix_tokens={} suffix_tokens={} prefill_to_first_token_ms={:.1}",
                tokens.len(),
                cached_prefix_len,
                suffix_len,
                elapsed_ms(start)
            ));
        }

        if report_perf {
            *first_token_instant = Some(std::time::Instant::now());
        }

        let max_new_tokens = p.max_new_tokens;
        let mut generated_tokens: Vec<u32> =
            Vec::with_capacity(engine::generated_capacity_hint(max_new_tokens));
        let mut finish_reason = String::from("length");
        let decode_trace_start = trace_enabled.then(std::time::Instant::now);
        let decode_progress_interval = if trace_enabled {
            crate::array::paged_decode_cache_clear_interval().max(1) as usize
        } else {
            usize::MAX
        };
        let mut decode_progress_last = decode_trace_start.unwrap_or_else(std::time::Instant::now);
        let mut decode_progress_last_count = 0usize;
        let decode_build_inputs_ms = 0.0;
        let mut decode_forward_ms = 0.0;
        let mut decode_sample_build_ms = 0.0;
        let mut decode_token_eval_ms = 0.0;
        let mut decode_cache_clear_ms = 0.0;

        for step in 0..max_new_tokens {
            let token_id = y.item_at_int32(0)? as u32;
            generated_tokens.push(token_id);
            token_history.push(token_id);
            let is_reasoning = reasoning_tracker.observe_token(token_id);
            *last_is_reasoning = is_reasoning;

            if token_id == eos_token_id || p.extra_eos_ids.contains(&token_id) {
                finish_reason = String::from("stop");
                break;
            }
            if cancelled.load(Ordering::Relaxed) {
                finish_reason = String::from("cancelled");
                break;
            }

            let token_text = Qwen3Tokenizer::step_decode_stream(
                decode_stream,
                tokenizer.inner(),
                token_id,
                &generated_tokens,
                *streamed_text_len,
            );
            *streamed_text_len += token_text.len();
            // Suppress reasoning deltas when include_reasoning == false.
            // Detokenize + length-advance above stay OUTSIDE this gate.
            if p.include_reasoning || !is_reasoning {
                cb.call(
                    Ok(ChatStreamChunk {
                        text: token_text,
                        done: false,
                        finish_reason: None,
                        tool_calls: None,
                        thinking: None,
                        thinking_enabled: None,
                        num_tokens: None,
                        prompt_tokens: None,
                        reasoning_tokens: None,
                        raw_text: None,
                        public_raw_text: None,
                        text_authoritative: None,
                        cached_tokens: None,
                        performance: None,
                        is_reasoning: Some(is_reasoning),
                    }),
                    ThreadsafeFunctionCallMode::NonBlocking,
                );
            }

            if let Some(reason) = crate::sampling::check_repetition_cutoff(
                &generated_tokens,
                p.max_consecutive_tokens,
                p.max_ngram_repeats,
                p.ngram_size,
            ) {
                finish_reason = reason.to_string();
                break;
            }
            if step + 1 >= max_new_tokens {
                break;
            }

            // Pure-Rust paged decode step.
            let next_logits = {
                let embed = self.embedding.clone();
                let caches_ref = self.caches.as_mut().ok_or_else(|| {
                    Error::from_reason(
                        "MoE paged_turn_stream_core_inner: caches dropped mid-decode",
                    )
                })?;
                let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                    Error::from_reason(
                        "MoE paged_turn_stream_core_inner: paged_adapter dropped mid-decode",
                    )
                })?;
                let forward_trace_start = trace_enabled.then(std::time::Instant::now);
                let logits = crate::models::qwen3_5_moe::paged_forward::run_paged_decode_step(
                    token_id,
                    &embed,
                    &mut self.layers,
                    caches_ref,
                    &self.final_norm,
                    &self.lm_head,
                    &layer_kinds,
                    adapter,
                    self.cached_rope_deltas.unwrap_or(0),
                )?;
                if let Some(start) = forward_trace_start {
                    decode_forward_ms += elapsed_ms(start);
                }
                logits.squeeze(Some(&[1]))?
            };

            let next_logits = if reasoning_tracker.should_force_think_end() {
                let forced_id = reasoning_tracker.forced_token_id()? as i32;
                y = MxArray::from_int32(&[forced_id], &[1])?;
                y.eval();
                continue;
            } else {
                apply_all_penalties(next_logits, &token_history, p)?
            };

            let sample_trace_start = trace_enabled.then(std::time::Instant::now);
            y = sample(&next_logits, sampling_config)?;
            if let Some(start) = sample_trace_start {
                decode_sample_build_ms += elapsed_ms(start);
            }
            let token_eval_trace_start = trace_enabled.then(std::time::Instant::now);
            y.eval();
            if let Some(start) = token_eval_trace_start {
                decode_token_eval_ms += elapsed_ms(start);
            }

            let cache_clear_trace_start = trace_enabled.then(std::time::Instant::now);
            crate::array::maybe_clear_cache_for_paged_step(step);
            if let Some(start) = cache_clear_trace_start {
                decode_cache_clear_ms += elapsed_ms(start);
            }
            if trace_enabled
                && generated_tokens
                    .len()
                    .is_multiple_of(decode_progress_interval)
            {
                let window_ms = elapsed_ms(decode_progress_last);
                let window_tokens = generated_tokens
                    .len()
                    .saturating_sub(decode_progress_last_count);
                let window_tok_s = if window_ms > 0.0 {
                    window_tokens as f64 / (window_ms / 1000.0)
                } else {
                    0.0
                };
                let elapsed_decode_ms = decode_trace_start.map(elapsed_ms).unwrap_or(0.0);
                let active_mib = crate::array::get_active_memory() / (1024.0 * 1024.0);
                let cache_mib = crate::array::get_cache_memory() / (1024.0 * 1024.0);
                let peak_mib = crate::array::get_peak_memory() / (1024.0 * 1024.0);
                write_inference_trace(format_args!(
                    "[MLX_TRACE] qwen3.5-moe paged_decode_progress generated_tokens={} \
                     context_tokens={} window_tokens={} window_ms={:.1} window_tok_s={:.2} \
                     elapsed_ms={:.1} cpp_ready={} build_inputs_ms={:.1} forward_ms={:.1} \
                     sample_ms={:.1} sample_build_ms={:.1} token_eval_ms={:.1} \
                     cache_clear_ms={:.1} active_mib={:.1} cache_mib={:.1} peak_mib={:.1}",
                    generated_tokens.len(),
                    token_history.len(),
                    window_tokens,
                    window_ms,
                    window_tok_s,
                    elapsed_decode_ms,
                    false,
                    decode_build_inputs_ms,
                    decode_forward_ms,
                    decode_sample_build_ms + decode_token_eval_ms,
                    decode_sample_build_ms,
                    decode_token_eval_ms,
                    decode_cache_clear_ms,
                    active_mib,
                    cache_mib,
                    peak_mib
                ));
                decode_progress_last = std::time::Instant::now();
                decode_progress_last_count = generated_tokens.len();
            }
        }

        if let Some(start) = decode_trace_start {
            write_inference_trace(format_args!(
                "[MLX_TRACE] qwen3.5-moe paged_decode_done generated_tokens={} finish_reason={} \
                 decode_loop_ms={:.1} build_inputs_ms={:.1} \
                 forward_ms={:.1} sample_ms={:.1} sample_build_ms={:.1} \
                 token_eval_ms={:.1} cache_clear_ms={:.1}",
                generated_tokens.len(),
                finish_reason,
                elapsed_ms(start),
                decode_build_inputs_ms,
                decode_forward_ms,
                decode_sample_build_ms + decode_token_eval_ms,
                decode_sample_build_ms,
                decode_token_eval_ms,
                decode_cache_clear_ms
            ));
        }

        Ok((generated_tokens, finish_reason))
    }
}
