//! The hand-written block-paged whole-turn cores (sync + stream, text + vision).

use super::*;

impl Qwen35Inner {
    /// Block-paged variant of [`Self::vision_mtp_whole_turn_core`].
    ///
    /// Mirrors the flat path's control flow (penalty stack, decode
    /// loop, EOS / repetition cutoff, performance timing, output
    /// post-processing) but routes full-attention layers through
    /// `forward_paged_or_flat` against the paged KV adapter. GDN
    /// (linear-attention) layers continue to use their existing
    /// `Qwen3_5LayerCache::Linear(ArraysCache)` storage and are
    /// reset+re-prefilled every turn (no cross-request prefix reuse —
    /// vLLM's `MambaManager` stance).
    ///
    /// Per-turn lifecycle:
    /// 1. Adapter lifecycle: warm-continue when the prior turn ended
    ///    via `finalize_turn_keep_live`; cold-start (reset →
    ///    find_cached_prefix → allocate_suffix) otherwise.
    /// 2. Prepare GDN prefix state from live/session checkpoints when
    ///    available; otherwise replay the cached prefix through GDN.
    /// 3. Prefill via `paged_forward::run_paged_prefill_chunk`.
    /// 4. Decode loop via `paged_forward::run_paged_decode_step`.
    /// 5. End-of-turn: `finalize_turn_keep_live` keeps the partial
    ///    trailing block live for the next turn's warm
    ///    `continue_turn` (mirrors LFM2 / Qwen3).
    ///
    /// Limitations:
    /// * VLM is rejected upstream — paged dispatch is text-only.
    /// * Cross-turn GDN prefix reuse is limited to live/history/prefix
    ///   checkpoints whose identity matches the paged KV prefix. Misses
    ///   fall back to GDN replay from token 0.
    /// * Pure-cache prompt (every prompt token already in the paged
    ///   pool) is rejected — same caveat as LFM2 / Qwen3 paged paths.
    /// * Paged turns run the pure-Rust
    ///   `DecoderLayer::forward_paged_or_flat`.
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

        // This paged turn writes full-attention K/V into the paged adapter
        // pool, NOT the flat `self.caches`, so the flat full-attention slots no
        // longer reflect the conversation. A later streaming dense-MTP fallback
        // must rebuild the flat caches before decoding. See
        // `paged_full_attn_caches_dirty`.
        self.paged_full_attn_caches_dirty = true;

        let prompt_token_count = tokens.len() as u32;
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
        let trace_enabled = inference_trace_enabled();

        // === Adapter lifecycle: warm continuation OR cold start ===
        let seq_id: u32 = 0;
        // Lazy decode allocation: pass the prompt length only.
        let total_budget = tokens.len() as u32;
        // Per-block extra_keys for prefix-cache lookup. Text-only dispatch
        // (image-bearing turns route to the flat path) yields all-empty
        // per-block vecs; the resulting hashes are bit-equal to passing `&[]`
        // to the uniform API. VLM-paged forward integration would swap in real
        // image-position pairs here to enable image-aware cache isolation.
        let block_size = {
            let adapter = self
                .paged_adapter
                .as_ref()
                .ok_or_else(|| Error::from_reason("paged_turn_sync_core: paged_adapter is None"))?;
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
        // vLLM exact-prefix cap — see qwen3/model.rs:paged_turn_sync_core.
        // Ensures every paged turn has at least one suffix token to prefill,
        // even when the live cache (or a prior request's residue) already
        // covers the entire new prompt.
        let max_cache_hit_tokens = total_budget.saturating_sub(1);
        let live_ready;
        let live_prefix_match;
        let live_tokens_len;
        let mut live_mismatch = TokenPrefixMismatchTrace::default();
        // Adapter-owned warm/cold lifecycle. The [MLX_TRACE] line below
        // reads the PRE-turn live state, so probe the adapter immutably FIRST
        // (prepare_turn mutates request_tokens via continue_turn/reset). The
        // adapter re-reads the same state internally, so live_* matches what
        // prepare_turn decides. extra_keys=&[] (uniform API) is bit-equal to
        // `&lookup_extra_keys` for text-only dispatch (all-empty per-block vec
        // → identical hashes; see the block_size comment above).
        // Suffix blocks are allocated inside prepare_turn.
        {
            let adapter = self.paged_adapter.as_ref().ok_or_else(|| {
                Error::from_reason(
                    "paged_turn_sync_core: paged_adapter is None — caller must check \
                     use_block_paged_cache before dispatch",
                )
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
            .ok_or_else(|| {
                Error::from_reason(
                    "paged_turn_sync_core: paged_adapter is None — caller must check \
                     use_block_paged_cache before dispatch",
                )
            })?
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
                self.invalidate_dense_paged_session("planned-MTP adapter preparation failure");
                return Err(error);
            }
        };
        let cached_prefix_len = plan.cached_prefix_len;
        let continued_live_prefix = plan.continued_live_prefix;
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] qwen3.5-dense paged_prefix_lookup prompt_tokens={} \
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

        let gdn_prefix_preparation = match self.prepare_dense_gdn_prefix_state(
            &tokens,
            cached_prefix_len,
            block_size,
            &lookup_extra_keys,
            cache_salt,
            continued_live_prefix,
        ) {
            Ok(preparation) => preparation,
            Err(error) => {
                self.invalidate_dense_paged_session("planned-MTP GDN-prefix preparation failure");
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
            self.invalidate_dense_paged_session("planned-MTP GDN auxiliary-prefix confirmation");
            return Err(Error::from_reason(error));
        }
        let preserves_image_lineage = carries_image_lineage && cached_prefix_len > 0;
        self.cached_token_history.clear();
        if !preserves_image_lineage {
            self.cached_image_key = None;
            self.cached_paged_image_token_positions.clear();
        }
        // Carry the cross-turn M-RoPE delta only when this turn extends the live
        // image sequence (continued_live_prefix). A cold start OR a non-live
        // prefix-cache hit (cached_prefix_len > 0 but not a live continuation)
        // can only restore pure-text prefix blocks, so a stale image delta is
        // dropped and the text suffix rotates at the raw physical slot.
        self.cached_rope_deltas = crate::models::qwen3_5::paged_forward::rope_delta_for_paged_turn(
            self.cached_rope_deltas,
            preserves_image_lineage,
        );

        let suffix_len = match prompt_token_count.checked_sub(cached_prefix_len) {
            Some(suffix_len) => suffix_len,
            None => {
                self.invalidate_dense_paged_session("planned-MTP prefix length mismatch");
                return Err(Error::from_reason(
                    "paged_turn_sync_core: cached_prefix_len > total_prompt_tokens",
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
        );

        let (generated_tokens, finish_reason, mtp_profiler) = match forward_result {
            Ok(t) => {
                let image_token_positions = self.cached_paged_image_token_positions.clone();
                // Drop-last history length published below (frontier check).
                let expected_history_len = tokens.len() + t.0.len().saturating_sub(1);
                self.finalize_dense_manual_paged_turn(
                    &image_token_positions,
                    cache_salt,
                    expected_history_len,
                )?;
                t
            }
            Err(e) => {
                self.invalidate_dense_paged_session("planned-MTP sync prefill/decode failure");
                return Err(e);
            }
        };

        // Persist the full token history so subsequent
        // `chat_session_continue` /
        // `chat_tokens_delta_sync` calls find an initialized session
        // to extend. The paged decode loop never feeds the LAST
        // sampled token through the model, so drop it from the
        // saved history (mirrors LFM2 / Qwen3 paged path).
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
        let gdn_history_checkpoint_store = match self.remember_dense_gdn_history_checkpoint() {
            Ok(store) => store,
            Err(error) => {
                self.invalidate_dense_paged_session(
                    "planned-MTP sync GDN history checkpoint failure",
                );
                return Err(error);
            }
        };
        if inference_trace_enabled() {
            write_inference_trace(format_args!(
                "[MLX_TRACE] qwen3.5-dense gdn_history_checkpoint stored={} tokens={} \
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
            .map(|mut m| {
                if let Some(prof) = mtp_profiler.as_ref() {
                    prof.fill_mtp_acceptance(&mut m);
                }
                m
            })
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

    /// Inner forward + decode loop for `paged_turn_sync_core`. Split
    /// out so the caller can wrap it with `release_request` on either
    /// path.
    ///
    /// The pure-Rust paged prefill populates the GDN linear caches and
    /// writes K/V into the adapter pool; decode steps then run through
    /// `paged_forward::run_paged_decode_step`.
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
    ) -> Result<(
        Vec<u32>,
        String,
        Option<crate::decode_profiler::DecodeProfiler>,
    )> {
        // Invariant: caller-applied vLLM cap guarantees suffix_len > 0.
        debug_assert!(
            suffix_len > 0,
            "paged_turn_sync_core_inner: caller must cap max_cache_hit_tokens at prompt.len() - 1"
        );

        // H2: clone the backend-installed per-turn cancel flag up front —
        // the decode loop below borrows `self` mutably.
        let turn_cancel = self.turn_cancel.clone();

        let suffix = &tokens[(cached_prefix_len as usize)..];
        let layer_kinds = crate::models::qwen3_5::decoder_layer::compute_layer_kinds(
            self.config.num_layers as usize,
            |i| self.config.is_linear_layer(i),
        );

        // Paged prompt-prefix MTP prefill. Mirrors the dense gate's
        // `want_prompt_hidden` predicate. Capturing the post-`final_norm`
        // hidden for every prompt token lets `begin_mtp_decode`'s
        // prompt-prefix seed commit the full prompt (advancing the stepper's
        // `committed_len` to N) before cycle 1, so
        // MTP drafts attend over the prompt (matches the dense MTP path). The
        // `cached_prefix_len == 0` clause matches dense: on a cache-reuse turn
        // the suffix-only prefill cannot produce the full prompt's hidden
        // tensor.
        let eager_mtp_paged = p.enable_mtp && self.has_mtp_weights();
        let want_prompt_hidden = eager_mtp_paged && cached_prefix_len == 0;
        let mut mtp_profiler = begin_paged_mtp_profiler(eager_mtp_paged, suffix_len);

        // === PREFILL ===
        let (last_logits, prompt_hidden, gdn_checkpoint) = self.run_dense_core_paged_prefill(
            tokens,
            suffix,
            cached_prefix_len,
            gdn_prefix_already_primed,
            want_prompt_hidden.then_some(tokens.len()),
            &layer_kinds,
            "paged_turn_sync_core_inner",
        )?;
        if let Some(profiler) = mtp_profiler.as_mut() {
            profiler.end_prefill();
        }
        self.publish_dense_gdn_materialized_prefix_checkpoint(tokens, p.cache_salt, gdn_checkpoint);

        // First-token sample.
        let mut token_history: Vec<u32> = tokens.to_vec();
        let last_logits = apply_all_penalties(last_logits, &token_history, p)?;
        let mut y = sample(&last_logits, *sampling_config)?;
        y.eval();

        // Smooth memory peak: drop transient prefill buffers before decode
        // starts allocating. Prefill builds a massive MLX subgraph; once
        // we have the last logits, those intermediates are dead but
        // MLX's caching allocator holds them.
        crate::array::synchronize_and_clear_cache();

        if let Some(profiler) = mtp_profiler.as_mut() {
            profiler.mark_first_token();
        }
        if report_perf {
            *first_token_instant = Some(std::time::Instant::now());
        }

        // === DECODE LOOP ===
        let max_new_tokens = p.max_new_tokens;
        let mut generated_tokens: Vec<u32> =
            Vec::with_capacity(engine::generated_capacity_hint(max_new_tokens));
        let mut finish_reason = String::from("length");

        // Pure-Rust ("eager") paged MTP gate. The paged adapter IS present
        // here (this is the paged core), so — unlike the flat eager gate —
        // the gate does NOT require `paged_adapter.is_none()`.
        info!(
            "Qwen3.5 MTP gate (paged): enable_mtp={} has_mtp_weights={} -> eager_mtp_paged={}",
            p.enable_mtp,
            self.has_mtp_weights(),
            eager_mtp_paged
        );

        // Pre-cycle lookahead reservation while AR fallback is still an
        // option — allocator exhaustion inside the verify loop would instead
        // surface as a turn error and invalidate the paged session.
        let eager_mtp_paged =
            eager_mtp_paged && self.reserve_paged_mtp_lookahead(p, "paged_turn_sync_core_inner")?;

        if eager_mtp_paged {
            // Pure-Rust ("eager") paged MTP.
            // The main Step-A / verify forwards route through the paged adapter
            // (`run_paged_step_with_hidden` / `run_paged_verify_step`); the GDN
            // recurrent state stays FLAT in `self.caches` Linear slots, so the
            // GDN tape replay (the rollback keystone) is IDENTICAL to the flat
            // eager arm. Full-attention K/V lives in the paged pool, so the
            // rollback rewinds it via `adapter.rollback_last_tokens(rejected)`,
            // NOT a `self.caches` KV trim.
            MxArray::async_eval_arrays(&[&y]);

            let mut profiler = mtp_profiler.take().ok_or_else(|| {
                Error::from_reason("Qwen3.5 paged MTP profiler was not initialized before decode")
            })?;

            let eos_id = eos_token_id;
            let generation_stream = crate::stream::Stream::new(crate::stream::DeviceType::Gpu);
            let model_size_bytes = self.config.estimate_memory_bytes() as usize;
            let _wired_ctx =
                crate::stream::WiredLimitContext::new(model_size_bytes, vec![generation_stream]);

            // Prompt-tail ids for the committed-history seed. `prompt_hidden`
            // is only `Some` when `want_prompt_hidden` held, which already
            // requires `cached_prefix_len == 0`, so the captured hiddens cover
            // the whole prompt.
            let prompt_hidden_ids: Vec<u32> = tokens.to_vec();

            let mut rng = rand::rng();

            // The propose/verify whole-turn loop is engine-owned
            // (`run_mtp_turn`) and drives the `DenseMtpStepper` in its PAGED
            // mode: `begin_mtp_decode` claims the adapter's active sequence as
            // the turn's owner, runs the committed-history prompt seed, and
            // routes the Step-A / verify forwards through the adapter, which
            // stays on `self` throughout so the paged-history save below finds
            // it. The paged path commits cache state through its own
            // paged-history save; the turn outcome's rewind tail is consumed
            // below (`last_in_cache` stays inert — dense paged is drop-last).
            let outcome = crate::engine::mtp_turn::run_mtp_turn(
                self,
                &mut rng,
                crate::engine::mtp_turn::MtpTurnArgs {
                    y: y.clone(),
                    depth: p.mtp_depth,
                    params: p,
                    reasoning_tracker,
                    profiler: &mut profiler,
                    max_new_tokens,
                    eos_id,
                    generated_tokens: &mut generated_tokens,
                    token_history: &mut token_history,
                    finish_reason: &mut finish_reason,
                    first_token_instant,
                    report_perf: p.report_performance,
                    generation_stream,
                    prompt_hidden,
                    prompt_hidden_ids: Some(prompt_hidden_ids),
                    // H2: sync paged MTP cancels through the engine loop's
                    // ungated polls (no StreamingCtx on this site).
                    cancel_flag: turn_cancel.as_deref(),
                },
                None,
            )?;
            // dense paged always saves drop-last, so `last_in_cache` is inert
            // here; the engine-computed rewind tail is recorded so the
            // epilogue's frontier check (and tests) can see the mid-cycle
            // stop was acted on rather than discarded.
            self.paged_mtp_last_rollback_unemitted = outcome.rollback_unemitted;

            // `self.caches` already holds the live GDN state (the eager paged
            // forwards wrote it directly) — nothing to export.
            return Ok((generated_tokens, finish_reason, Some(profiler)));
        }

        for step in 0..max_new_tokens {
            let token_id = y.item_at_int32(0)? as u32;
            generated_tokens.push(token_id);
            token_history.push(token_id);
            reasoning_tracker.observe_token(token_id);

            if token_id == eos_token_id || p.extra_eos_ids.contains(&token_id) {
                finish_reason = String::from("stop");
                break;
            }
            // H2 sync cancel poll — the SAME snapshot point as the paged
            // streaming twin (`paged_turn_stream_core_inner`): after the
            // EOS check, before the repetition cutoff.
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

            // Decode forward (pure-Rust paged step).
            let next_logits = {
                let embed = self.embedding.clone();
                let caches_ref = self.caches.as_mut().ok_or_else(|| {
                    Error::from_reason("paged_turn_sync_core_inner: caches dropped mid-decode")
                })?;
                let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                    Error::from_reason(
                        "paged_turn_sync_core_inner: paged_adapter dropped mid-decode",
                    )
                })?;
                let logits = crate::models::qwen3_5::paged_forward::run_paged_decode_step(
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

        Ok((generated_tokens, finish_reason, None))
    }

    /// Single-turn image-bearing block-paged dispatch (non-streaming).
    ///
    /// The paged sibling of the flat VLM prefill: it processes the images,
    /// merges the vision features into the token embeddings, computes M-RoPE
    /// positions, then prefills through the paged adapter via
    /// [`crate::models::qwen3_5::paged_forward::run_paged_vlm_prefill`] and runs the plain
    /// autoregressive decode loop.
    ///
    /// Same-image live histories continue in place; fresh histories look up
    /// full blocks using per-image content keys and prefill only the uncached
    /// suffix. Decode rotates at the physical token count plus the cached
    /// M-RoPE delta (`cached_rope_deltas`). This core runs plain autoregressive
    /// decode with no draft/verify; MTP weights are ignored, so an MTP-enabled
    /// session's image turns route here and decode as AR.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn vision_paged_turn_sync_core(
        &mut self,
        tokens: Vec<u32>,
        images: &[Vec<u8>],
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

        let (vision_encoder, img_proc) =
            match (self.vision_encoder.clone(), self.image_processor.as_ref()) {
                (Some(enc), Some(proc)) => (enc, proc),
                _ => {
                    return Err(Error::from_reason(
                        "VLM prefill requested but vision encoder/processor not loaded",
                    ));
                }
            };

        // This paged turn writes full-attention K/V into the paged adapter pool,
        // leaving the flat `self.caches` full-attention slots stale. A later
        // dense-MTP fallback must rebuild them first.
        self.paged_full_attn_caches_dirty = true;

        let think_end_id = tokenizer.think_end_id();
        let think_end_str = tokenizer.think_end_str().map(|s| s.to_string());
        let thinking_enabled = thinking.enabled;
        let mut reasoning_tracker = engine::ReasoningTracker::from_setup(&thinking, think_end_id);

        let generation_start = if report_perf {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut first_token_instant: Option<std::time::Instant> = None;
        let sampling_config = p.sampling_config;

        // === VLM image processing: expand placeholders + merge features ===
        let sms = self.spatial_merge_size.unwrap_or(2);
        let image_refs: Vec<&[u8]> = images.iter().map(|v| v.as_slice()).collect();
        let processed = img_proc.process_many(&image_refs)?;
        let per_image_token_counts =
            compute_image_token_counts_per_image(&processed.grid_thw(), sms)?;
        let expanded_tokens = inject_image_placeholders(&tokens, &per_image_token_counts)?;
        self.preflight_paged_context(expanded_tokens.len(), &mut p)?;
        let (image_cache_key, per_image_hashes) = engine::compute_image_cache_keys(images);
        let image_token_positions = engine::map_expanded_image_token_positions(
            &expanded_tokens,
            IMAGE_TOKEN_ID as u32,
            &per_image_token_counts,
            &per_image_hashes,
        )
        .map_err(Error::from_reason)?;
        let prompt_token_count = expanded_tokens.len() as u32;

        let embed = self.embedding.clone();
        let input_ids = MxArray::from_uint32(&expanded_tokens, &[1, expanded_tokens.len() as i64])?;

        let generation_stream = Stream::new(DeviceType::Gpu);
        let model_size_bytes = self.config.estimate_memory_bytes() as usize;
        let _wired_ctx =
            crate::stream::WiredLimitContext::new(model_size_bytes, vec![generation_stream]);

        let merge = vlm_prepare_vision_features(
            &input_ids,
            &per_image_hashes,
            &processed,
            &vision_encoder,
            sms,
            &embed,
            generation_stream,
            &self.vision_cache,
        )?;
        // `merge` only retains evaluated per-image features and newly-built
        // text/position arrays. Release the aggregate f32 pixel tensor before
        // language prefill instead of keeping every historical image alive for
        // the rest of the turn.
        drop(processed);
        crate::array::clear_cache();

        // === Image-aware paged-prefix lifecycle ===
        let total_budget = expanded_tokens.len() as u32;
        let block_size = self
            .paged_adapter
            .as_ref()
            .ok_or_else(|| {
                Error::from_reason("vision_paged_turn_sync_core: paged_adapter is None")
            })?
            .block_size();
        let lookup_extra_keys = engine::build_paged_extra_keys(
            expanded_tokens.len(),
            block_size,
            &image_token_positions,
        );
        let same_live_image = self.cached_image_key == Some(image_cache_key);
        let prefix_resolution = self.prepare_dense_vlm_paged_prefix(
            &expanded_tokens,
            total_budget,
            block_size,
            &lookup_extra_keys,
            p.reuse_cache,
            p.reuse_cache && same_live_image,
            image_cache_key,
            p.cache_salt,
        )?;
        let plan = prefix_resolution.effective_plan;
        let cached_prefix_len = plan.cached_prefix_len;
        tracing::info!(
            target: "mlx_core::inference",
            event = "vlm_prefix_plan",
            model = "qwen3_5_dense",
            prompt_tokens = expanded_tokens.len(),
            image_count = images.len(),
            image_tokens = image_token_positions.len(),
            candidate_cached_prefix_tokens = prefix_resolution.candidate_cached_prefix_len,
            effective_cached_prefix_tokens = cached_prefix_len,
            cached_prefix_tokens = cached_prefix_len,
            suffix_tokens = expanded_tokens.len() - cached_prefix_len as usize,
            continued_live_prefix = plan.continued_live_prefix,
            same_live_image,
            gdn_prefix_already_primed = prefix_resolution.gdn_prefix_already_primed,
            downgraded_to_cold = prefix_resolution.downgraded_to_cold,
            "image-aware paged prefix planned"
        );
        let gdn_prefix_already_primed = prefix_resolution.gdn_prefix_already_primed;
        self.cached_token_history.clear();
        self.cached_image_key = None;
        self.cached_paged_image_token_positions.clear();
        // Store the image prefill's compressed-position delta so a later text
        // warm-continuation rotates its queries at the same compressed M-RoPE
        // positions the image keys were written with.
        self.cached_rope_deltas = Some(merge.rope_deltas as i32);

        let layer_kinds = crate::models::qwen3_5::decoder_layer::compute_layer_kinds(
            self.config.num_layers as usize,
            |i| self.config.is_linear_layer(i),
        );

        let turn_cancel = self.turn_cancel.clone();
        let forward_result = (|| -> Result<(Vec<u32>, String)> {
            // === PREFILL ===
            let last_logits = {
                let _stream_ctx = StreamContext::new(generation_stream);
                let caches_ref = self.caches.as_mut().ok_or_else(|| {
                    Error::from_reason("vision_paged_turn_sync_core: caches not initialized")
                })?;
                let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                    Error::from_reason("vision_paged_turn_sync_core: paged_adapter dropped")
                })?;
                crate::models::qwen3_5::paged_forward::run_paged_vlm_prefill(
                    &expanded_tokens,
                    &merge,
                    cached_prefix_len,
                    gdn_prefix_already_primed,
                    &embed,
                    &mut self.layers,
                    caches_ref,
                    &self.final_norm,
                    &self.lm_head,
                    &layer_kinds,
                    adapter,
                    turn_cancel.as_deref(),
                )?
            };
            let (last_logits, gdn_checkpoint) = last_logits;
            self.publish_dense_gdn_materialized_prefix_checkpoint_with_keys(
                &expanded_tokens,
                &lookup_extra_keys,
                p.cache_salt,
                gdn_checkpoint,
            );

            let mut token_history: Vec<u32> = expanded_tokens.clone();
            let last_logits = apply_all_penalties(last_logits, &token_history, &p)?;
            let mut y = sample(&last_logits, sampling_config)?;
            y.eval();

            crate::array::synchronize_and_clear_cache();
            if report_perf {
                first_token_instant = Some(std::time::Instant::now());
            }

            // === DECODE LOOP (autoregressive, scalar-offset RoPE) ===
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
                // H2 sync cancel poll — the SAME snapshot point as the
                // vision paged streaming twin: after the EOS check, before
                // the repetition cutoff.
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

                let next_logits = {
                    let _stream_ctx = StreamContext::new(generation_stream);
                    let caches_ref = self.caches.as_mut().ok_or_else(|| {
                        Error::from_reason("vision_paged_turn_sync_core: caches dropped mid-decode")
                    })?;
                    let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                        Error::from_reason(
                            "vision_paged_turn_sync_core: paged_adapter dropped mid-decode",
                        )
                    })?;
                    let logits = crate::models::qwen3_5::paged_forward::run_paged_decode_step(
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

                if reasoning_tracker.should_force_think_end() {
                    let forced_id = reasoning_tracker.forced_token_id()? as i32;
                    y = MxArray::from_int32(&[forced_id], &[1])?;
                    y.eval();
                    continue;
                }
                let next_logits = apply_all_penalties(next_logits, &token_history, &p)?;

                y = sample(&next_logits, sampling_config)?;
                y.eval();

                crate::array::maybe_clear_cache_for_paged_step(step);
            }

            Ok((generated_tokens, finish_reason))
        })();

        // Terminal lifecycle, mirroring the text paged core
        // (`paged_turn_sync_core` / `finalize_paged_turn`). The error path
        // always releases the request and returns. The success path is
        // resolved below so the session ends in exactly one of two states,
        // never a partial one: FULLY continuable (keep-live registered AND GDN
        // checkpoint stored AND history + image key published) or
        // NON-continuable (request released AND history cleared AND image key
        // None) so a follow-up text continue is safely rejected instead of
        // cold-prefilling image-placeholder ids as ordinary tokens.
        let (generated_tokens, finish_reason) = match forward_result {
            Ok(t) => t,
            Err(e) => {
                self.invalidate_dense_paged_session("VLM sync prefill/decode failure");
                return Err(e);
            }
        };

        // Build the saved history: the EXPANDED (image-placeholder) prompt plus
        // all generated tokens except the last — the paged decode loop never
        // forwards the final sampled token into the cache, so it would not match
        // the live `request_tokens` (drop-last rule shared with the text paged
        // core).
        let mut full_history = expanded_tokens.clone();
        if !generated_tokens.is_empty() {
            full_history.extend_from_slice(&generated_tokens[..generated_tokens.len() - 1]);
        }

        // Keep-live registration must run before the GDN checkpoint, which
        // snapshots the live recurrent state; short-circuit `&&` preserves that
        // order. `remember_dense_gdn_history_checkpoint` snapshots from
        // `cached_token_history`, so publish the history first, then checkpoint.
        // Any failure downgrades to NON-continuable rather than discarding the
        // already-successful generation output.
        let keep_live_ok = p.reuse_cache
            && self
                .finalize_dense_manual_paged_turn(
                    &image_token_positions,
                    p.cache_salt,
                    full_history.len(),
                )
                .is_ok();
        let continuable = if keep_live_ok {
            self.cached_token_history = full_history;
            self.cached_image_key = Some(image_cache_key);
            self.cached_paged_image_token_positions = image_token_positions.clone();
            match self.remember_dense_gdn_history_checkpoint() {
                Ok(_) => true,
                Err(error) => {
                    tracing::warn!(
                        target: "mlx_core::qwen3_5::paged",
                        "dense VLM GDN history checkpoint failed: {error}",
                    );
                    self.invalidate_dense_paged_session("VLM sync GDN history checkpoint failure");
                    false
                }
            }
        } else {
            false
        };

        if continuable {
            // FULLY continuable: live KV + GDN recurrent state encode the image
            // context; `cached_image_key` records it (flat vision save contract).
        } else if self.caches.is_some() {
            // No-reuse, keep-live failure, or checkpoint failure: release the
            // request and reset to a pristine non-live state. `reset_caches_sync`
            // nulls `self.caches` (so `has_live_session()` reports false) and
            // clears token history, image key, rope deltas, and GDN checkpoints,
            // so a follow-up continue is rejected ("requires an initialized
            // session") instead of cold-prefilling image-placeholder ids.
            if p.reuse_cache {
                self.invalidate_dense_paged_session("non-continuable VLM sync completion");
            } else {
                self.discard_dense_paged_session();
            }
        }

        let performance = if report_perf {
            compute_performance_metrics(
                generation_start,
                first_token_instant,
                expanded_tokens.len() - cached_prefix_len as usize,
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

    /// Streaming twin of [`Self::vision_paged_turn_sync_core`].
    ///
    /// Single-turn image-bearing block-paged dispatch that emits each
    /// generated token through the streaming callback. Same prefill + decode
    /// spine; plain AR decode, MTP weights ignored.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn vision_paged_turn_stream_core(
        &mut self,
        tokens: Vec<u32>,
        images: &[Vec<u8>],
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

        let (vision_encoder, img_proc) =
            match (self.vision_encoder.clone(), self.image_processor.as_ref()) {
                (Some(enc), Some(proc)) => (enc, proc),
                _ => {
                    return Err(Error::from_reason(
                        "VLM prefill requested but vision encoder/processor not loaded",
                    ));
                }
            };

        self.paged_full_attn_caches_dirty = true;

        let include_reasoning = p.include_reasoning;
        let think_end_id = tokenizer.think_end_id();
        let think_end_str = tokenizer.think_end_str().map(|s| s.to_string());
        let thinking_enabled = thinking.enabled;
        let mut reasoning_tracker = engine::ReasoningTracker::from_setup(&thinking, think_end_id);

        let generation_start = if report_perf {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut first_token_instant: Option<std::time::Instant> = None;
        let sampling_config = p.sampling_config;

        let mut decode_stream = tokenizer.inner().decode_stream(true);
        let mut streamed_text_len = 0usize;
        let mut last_is_reasoning = thinking_enabled;

        // === VLM image processing: expand placeholders + merge features ===
        let sms = self.spatial_merge_size.unwrap_or(2);
        let image_refs: Vec<&[u8]> = images.iter().map(|v| v.as_slice()).collect();
        let processed = img_proc.process_many(&image_refs)?;
        let per_image_token_counts =
            compute_image_token_counts_per_image(&processed.grid_thw(), sms)?;
        let expanded_tokens = inject_image_placeholders(&tokens, &per_image_token_counts)?;
        self.preflight_paged_context(expanded_tokens.len(), &mut p)?;
        let (image_cache_key, per_image_hashes) = engine::compute_image_cache_keys(images);
        let image_token_positions = engine::map_expanded_image_token_positions(
            &expanded_tokens,
            IMAGE_TOKEN_ID as u32,
            &per_image_token_counts,
            &per_image_hashes,
        )
        .map_err(Error::from_reason)?;
        let prompt_token_count = expanded_tokens.len() as u32;

        let embed = self.embedding.clone();
        let input_ids = MxArray::from_uint32(&expanded_tokens, &[1, expanded_tokens.len() as i64])?;

        let generation_stream = Stream::new(DeviceType::Gpu);
        let model_size_bytes = self.config.estimate_memory_bytes() as usize;
        let _wired_ctx =
            crate::stream::WiredLimitContext::new(model_size_bytes, vec![generation_stream]);

        let merge = vlm_prepare_vision_features(
            &input_ids,
            &per_image_hashes,
            &processed,
            &vision_encoder,
            sms,
            &embed,
            generation_stream,
            &self.vision_cache,
        )?;
        drop(processed);
        crate::array::clear_cache();

        // === Image-aware paged-prefix lifecycle ===
        let total_budget = expanded_tokens.len() as u32;
        let block_size = self
            .paged_adapter
            .as_ref()
            .ok_or_else(|| {
                Error::from_reason("vision_paged_turn_stream_core: paged_adapter is None")
            })?
            .block_size();
        let lookup_extra_keys = engine::build_paged_extra_keys(
            expanded_tokens.len(),
            block_size,
            &image_token_positions,
        );
        let same_live_image = self.cached_image_key == Some(image_cache_key);
        let prefix_resolution = self.prepare_dense_vlm_paged_prefix(
            &expanded_tokens,
            total_budget,
            block_size,
            &lookup_extra_keys,
            p.reuse_cache,
            p.reuse_cache && same_live_image,
            image_cache_key,
            p.cache_salt,
        )?;
        let plan = prefix_resolution.effective_plan;
        let cached_prefix_len = plan.cached_prefix_len;
        tracing::info!(
            target: "mlx_core::inference",
            event = "vlm_prefix_plan",
            model = "qwen3_5_dense",
            prompt_tokens = expanded_tokens.len(),
            image_count = images.len(),
            image_tokens = image_token_positions.len(),
            candidate_cached_prefix_tokens = prefix_resolution.candidate_cached_prefix_len,
            effective_cached_prefix_tokens = cached_prefix_len,
            cached_prefix_tokens = cached_prefix_len,
            suffix_tokens = expanded_tokens.len() - cached_prefix_len as usize,
            continued_live_prefix = plan.continued_live_prefix,
            same_live_image,
            gdn_prefix_already_primed = prefix_resolution.gdn_prefix_already_primed,
            downgraded_to_cold = prefix_resolution.downgraded_to_cold,
            "image-aware paged prefix planned"
        );
        let gdn_prefix_already_primed = prefix_resolution.gdn_prefix_already_primed;
        self.cached_token_history.clear();
        self.cached_image_key = None;
        self.cached_paged_image_token_positions.clear();
        // Store the image prefill's compressed-position delta so a later text
        // warm-continuation rotates its queries at the same compressed M-RoPE
        // positions the image keys were written with.
        self.cached_rope_deltas = Some(merge.rope_deltas as i32);

        let layer_kinds = crate::models::qwen3_5::decoder_layer::compute_layer_kinds(
            self.config.num_layers as usize,
            |i| self.config.is_linear_layer(i),
        );

        let turn_cancel = self.turn_cancel.clone();
        let forward_result = (|| -> Result<(Vec<u32>, String)> {
            // === PREFILL ===
            let last_logits = {
                let _stream_ctx = StreamContext::new(generation_stream);
                let caches_ref = self.caches.as_mut().ok_or_else(|| {
                    Error::from_reason("vision_paged_turn_stream_core: caches not initialized")
                })?;
                let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                    Error::from_reason("vision_paged_turn_stream_core: paged_adapter dropped")
                })?;
                crate::models::qwen3_5::paged_forward::run_paged_vlm_prefill(
                    &expanded_tokens,
                    &merge,
                    cached_prefix_len,
                    gdn_prefix_already_primed,
                    &embed,
                    &mut self.layers,
                    caches_ref,
                    &self.final_norm,
                    &self.lm_head,
                    &layer_kinds,
                    adapter,
                    turn_cancel.as_deref(),
                )?
            };
            let (last_logits, gdn_checkpoint) = last_logits;
            self.publish_dense_gdn_materialized_prefix_checkpoint_with_keys(
                &expanded_tokens,
                &lookup_extra_keys,
                p.cache_salt,
                gdn_checkpoint,
            );

            let mut token_history: Vec<u32> = expanded_tokens.clone();
            let last_logits = apply_all_penalties(last_logits, &token_history, &p)?;
            let mut y = sample(&last_logits, sampling_config)?;
            y.eval();

            crate::array::synchronize_and_clear_cache();
            if report_perf {
                first_token_instant = Some(std::time::Instant::now());
            }

            let max_new_tokens = p.max_new_tokens;
            let mut generated_tokens: Vec<u32> =
                Vec::with_capacity(engine::generated_capacity_hint(max_new_tokens));
            let mut finish_reason = String::from("length");

            for step in 0..max_new_tokens {
                let token_id = y.item_at_int32(0)? as u32;
                generated_tokens.push(token_id);
                token_history.push(token_id);
                let is_reasoning = reasoning_tracker.observe_token(token_id);
                last_is_reasoning = is_reasoning;

                if token_id == eos_token_id || p.extra_eos_ids.contains(&token_id) {
                    finish_reason = String::from("stop");
                    break;
                }
                if cancelled.load(Ordering::Relaxed) {
                    finish_reason = String::from("cancelled");
                    break;
                }

                let token_text = Qwen3Tokenizer::step_decode_stream(
                    &mut decode_stream,
                    tokenizer.inner(),
                    token_id,
                    &generated_tokens,
                    streamed_text_len,
                );
                streamed_text_len += token_text.len();
                if include_reasoning || !is_reasoning {
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

                let next_logits = {
                    let _stream_ctx = StreamContext::new(generation_stream);
                    let caches_ref = self.caches.as_mut().ok_or_else(|| {
                        Error::from_reason(
                            "vision_paged_turn_stream_core: caches dropped mid-decode",
                        )
                    })?;
                    let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                        Error::from_reason(
                            "vision_paged_turn_stream_core: paged_adapter dropped mid-decode",
                        )
                    })?;
                    let logits = crate::models::qwen3_5::paged_forward::run_paged_decode_step(
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

                if reasoning_tracker.should_force_think_end() {
                    let forced_id = reasoning_tracker.forced_token_id()? as i32;
                    y = MxArray::from_int32(&[forced_id], &[1])?;
                    y.eval();
                    continue;
                }
                let next_logits = apply_all_penalties(next_logits, &token_history, &p)?;

                y = sample(&next_logits, sampling_config)?;
                y.eval();

                crate::array::maybe_clear_cache_for_paged_step(step);
            }

            Ok((generated_tokens, finish_reason))
        })();

        // Terminal lifecycle, mirroring the text paged core. The error path
        // always releases and returns. The success path is resolved below so
        // the session ends FULLY continuable (keep-live + GDN checkpoint +
        // history + image key) or NON-continuable (released + history cleared +
        // image key None), never partial — a follow-up text continue must never
        // cold-prefill image-placeholder ids as ordinary tokens.
        let (generated_tokens, finish_reason) = match forward_result {
            Ok(t) => t,
            Err(e) => {
                self.invalidate_dense_paged_session("VLM stream prefill/decode failure");
                return Err(e);
            }
        };

        // Saved history: expanded prompt + generated[..len-1] (drop-last rule
        // shared with the sync sibling / text paged core).
        let mut full_history = expanded_tokens.clone();
        if !generated_tokens.is_empty() {
            full_history.extend_from_slice(&generated_tokens[..generated_tokens.len() - 1]);
        }

        // Keep-live before the GDN checkpoint (which snapshots the live state);
        // checkpoint reads `cached_token_history`, so publish it first. Any
        // failure downgrades to NON-continuable rather than discarding output.
        let keep_live_ok = p.reuse_cache
            && self
                .finalize_dense_manual_paged_turn(
                    &image_token_positions,
                    p.cache_salt,
                    full_history.len(),
                )
                .is_ok();
        let continuable = if keep_live_ok {
            self.cached_token_history = full_history;
            self.cached_image_key = Some(image_cache_key);
            self.cached_paged_image_token_positions = image_token_positions.clone();
            match self.remember_dense_gdn_history_checkpoint() {
                Ok(_) => true,
                Err(error) => {
                    tracing::warn!(
                        target: "mlx_core::qwen3_5::paged",
                        "dense streaming VLM GDN history checkpoint failed: {error}",
                    );
                    self.invalidate_dense_paged_session(
                        "VLM stream GDN history checkpoint failure",
                    );
                    false
                }
            }
        } else {
            false
        };

        if continuable {
        } else if self.caches.is_some() {
            // Non-continuable: release the request and reset to a pristine
            // non-live state so a follow-up continue is rejected instead of
            // cold-prefilling image-placeholder ids. `reset_caches_sync` nulls
            // `self.caches` (so `has_live_session()` is false) and clears token
            // history, image key, rope deltas, and GDN checkpoints.
            if p.reuse_cache {
                self.invalidate_dense_paged_session("non-continuable VLM stream completion");
            } else {
                self.discard_dense_paged_session();
            }
        }

        // Flush residual buffered bytes (mirrors flat / text paged streaming).
        let full_text = tokenizer
            .decode_sync(&generated_tokens, true)
            .unwrap_or_else(|e| {
                tracing::warn!("Failed to decode generated tokens: {}", e);
                String::new()
            });
        if full_text.len() > streamed_text_len {
            let residual = full_text[streamed_text_len..].to_string();
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
                expanded_tokens.len() - cached_prefix_len as usize,
                generated_tokens.len(),
            )
        } else {
            None
        };

        let reasoning_tokens = reasoning_tracker.reasoning_token_count();
        let result = finalize_chat_result(
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

    /// Block-paged streaming variant of [`Self::chat_stream_sync_inner`].
    ///
    /// Mirrors `paged_turn_sync_core`'s adapter lifecycle and
    /// per-layer dispatch but emits each generated token through the
    /// streaming callback as it is produced.
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
        static NEXT_TRACE_TURN_ID: AtomicU64 = AtomicU64::new(1);
        if tokens.is_empty() {
            return Err(Error::from_reason("Empty prompt"));
        }
        let inference_info_enabled =
            tracing::enabled!(target: "mlx_core::inference", tracing::Level::INFO);
        let trace_turn_id = if inference_info_enabled {
            NEXT_TRACE_TURN_ID.fetch_add(1, Ordering::Relaxed)
        } else {
            0
        };
        let turn_trace_start = inference_info_enabled.then(std::time::Instant::now);
        if inference_info_enabled {
            tracing::info!(
                target: "mlx_core::inference",
                event = "turn_start",
                turn_id = trace_turn_id,
                path = "qwen3_5_dense_paged_stream",
                prompt_tokens = tokens.len(),
                mtp_requested = p.enable_mtp,
                mtp_depth = p.mtp_depth,
                report_performance = report_perf,
                "inference turn started"
            );
        }
        self.preflight_paged_context(tokens.len(), &mut p)?;

        // This paged turn writes full-attention K/V into the paged adapter
        // pool, NOT the flat `self.caches`, so a later streaming dense-MTP
        // fallback must rebuild the flat caches before decoding. See
        // `paged_full_attn_caches_dirty`.
        self.paged_full_attn_caches_dirty = true;

        let prompt_token_count = tokens.len() as u32;
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
        let trace_enabled = inference_trace_enabled();

        // Streaming decode state.
        let mut decode_stream = tokenizer.inner().decode_stream(true);
        let mut streamed_text_len = 0usize;
        let mut last_is_reasoning = thinking_enabled;
        let prefix_plan_start = inference_info_enabled.then(std::time::Instant::now);

        // === Adapter lifecycle: warm continue OR cold start ===
        let seq_id: u32 = 0;
        // Lazy decode allocation: pass the prompt length only.
        let total_budget = tokens.len() as u32;
        // Per-block extra_keys for prefix-cache lookup. See the matching
        // comment in `paged_turn_sync_core`.
        let block_size = {
            let adapter = self.paged_adapter.as_ref().ok_or_else(|| {
                Error::from_reason("paged_turn_stream_core: paged_adapter is None")
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
                Error::from_reason(
                    "paged_turn_stream_core: paged_adapter is None — caller must check \
                     use_block_paged_cache before dispatch",
                )
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
            .ok_or_else(|| {
                Error::from_reason(
                    "paged_turn_stream_core: paged_adapter is None — caller must check \
                     use_block_paged_cache before dispatch",
                )
            })?
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
                self.invalidate_dense_paged_session(
                    "planned-MTP stream adapter preparation failure",
                );
                return Err(error);
            }
        };
        let cached_prefix_len = plan.cached_prefix_len;
        let continued_live_prefix = plan.continued_live_prefix;
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] qwen3.5-dense paged_prefix_lookup prompt_tokens={} \
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

        let gdn_prefix_preparation = match self.prepare_dense_gdn_prefix_state(
            &tokens,
            cached_prefix_len,
            block_size,
            &lookup_extra_keys,
            cache_salt,
            continued_live_prefix,
        ) {
            Ok(preparation) => preparation,
            Err(error) => {
                self.invalidate_dense_paged_session(
                    "planned-MTP stream GDN-prefix preparation failure",
                );
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
            self.invalidate_dense_paged_session(
                "planned-MTP stream GDN auxiliary-prefix confirmation",
            );
            return Err(Error::from_reason(error));
        }
        let gdn_prefix_state = gdn_prefix_preparation.state;
        let gdn_restored_prefix_tokens = gdn_prefix_preparation.restored_prefix_tokens;
        let gdn_replayed_prefix_tokens = gdn_prefix_preparation.replayed_prefix_tokens;
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
                self.invalidate_dense_paged_session("planned-MTP stream prefix length mismatch");
                return Err(Error::from_reason(
                    "paged_turn_stream_core: cached_prefix_len > total_prompt_tokens",
                ));
            }
        };

        if inference_info_enabled {
            tracing::info!(
                target: "mlx_core::inference",
                event = "prefix_plan",
                turn_id = trace_turn_id,
                prompt_tokens = prompt_token_count,
                cached_prefix_tokens = cached_prefix_len,
                suffix_tokens = suffix_len,
                continued_live_prefix,
                live_ready,
                live_prefix_match,
                live_tokens = live_tokens_len,
                block_size,
                gdn_prefix_state,
                gdn_already_primed = gdn_prefix_already_primed,
                gdn_restored_prefix_tokens,
                gdn_replayed_prefix_tokens,
                elapsed_ms = prefix_plan_start.map(elapsed_ms).unwrap_or(0.0),
                "paged prefix plan resolved"
            );
        }

        if trace_enabled {
            // The size this turn's prefill actually splits on, not the raw
            // `MLX_PAGED_PREFILL_CHUNK_SIZE`: under a GDN cold policy the two
            // differ, and reading 0 here while the same turn reports
            // `gdn_checkpoint_materialized=true` points at the wrong diagnosis.
            write_inference_trace(format_args!(
                "[MLX_TRACE] qwen3.5-dense stream_paged_start prompt_tokens={} \
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
            trace_turn_id,
            turn_trace_start,
        );

        let (generated_tokens, finish_reason, decode_profiler) = match result {
            Ok(t) => {
                let image_token_positions = self.cached_paged_image_token_positions.clone();
                // Drop-last history length published below (frontier check).
                let expected_history_len = tokens.len() + t.0.len().saturating_sub(1);
                self.finalize_dense_manual_paged_turn(
                    &image_token_positions,
                    cache_salt,
                    expected_history_len,
                )?;
                t
            }
            Err(e) => {
                self.invalidate_dense_paged_session("planned-MTP stream prefill/decode failure");
                if inference_info_enabled {
                    tracing::info!(
                        target: "mlx_core::inference",
                        event = "turn_done",
                        turn_id = trace_turn_id,
                        status = "error",
                        stage = "prefill_or_decode",
                        elapsed_ms = turn_trace_start.map(elapsed_ms).unwrap_or(0.0),
                        "inference turn completed"
                    );
                }
                return Err(e);
            }
        };

        // Persist token history for subsequent session-continue calls.
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
        let gdn_history_checkpoint_store = match self.remember_dense_gdn_history_checkpoint() {
            Ok(store) => store,
            Err(error) => {
                self.invalidate_dense_paged_session(
                    "planned-MTP stream GDN history checkpoint failure",
                );
                return Err(error);
            }
        };
        if inference_info_enabled {
            tracing::info!(
                target: "mlx_core::inference",
                event = "cache_commit",
                turn_id = trace_turn_id,
                stored = gdn_history_checkpoint_store.stored,
                history_tokens = self.cached_token_history.len(),
                eval_ms = gdn_history_checkpoint_store.eval_ms,
                clone_ms = gdn_history_checkpoint_store.clone_ms,
                update_ms = gdn_history_checkpoint_store.update_ms,
                elapsed_ms = gdn_history_checkpoint_store.total_ms,
                "inference cache committed"
            );
        }
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] qwen3.5-dense gdn_history_checkpoint stored={} tokens={} \
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

        // Flush residual buffered bytes (mirrors flat streaming).
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
            .map(|mut metrics| {
                if let Some(profiler) = decode_profiler.as_ref() {
                    profiler.fill_mtp_acceptance(&mut metrics);
                }
                metrics
            })
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

        if inference_info_enabled {
            let (ttft_ms, prefill_tok_s, decode_tok_s, mtp_cycles, mtp_mean_total) = result
                .performance
                .as_ref()
                .map(|metrics| {
                    (
                        metrics.ttft_ms,
                        metrics.prefill_tokens_per_second,
                        metrics.decode_tokens_per_second,
                        metrics.mtp_cycles.unwrap_or(0),
                        metrics.mtp_mean_accepted_tokens_total.unwrap_or(0.0),
                    )
                })
                .unwrap_or((0.0, 0.0, 0.0, 0, 0.0));
            tracing::info!(
                target: "mlx_core::inference",
                event = "turn_done",
                turn_id = trace_turn_id,
                status = "ok",
                prompt_tokens = prompt_token_count,
                cached_prefix_tokens = cached_prefix_len,
                fresh_prefill_tokens = tokens.len().saturating_sub(cached_prefix_len as usize),
                generated_tokens = generated_tokens.len(),
                finish_reason = result.finish_reason.as_str(),
                elapsed_ms = turn_trace_start.map(elapsed_ms).unwrap_or(0.0),
                ttft_ms,
                prefill_tok_s,
                decode_tok_s,
                mtp_cycles,
                mtp_mean_total,
                "inference turn completed"
            );
        }

        // Terminal chunk.
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

    /// Inner forward + streaming decode loop for
    /// [`Self::paged_turn_stream_core`]. Mirrors LFM2's
    /// `paged_turn_stream_core_inner`.
    ///
    /// Runs the pure-Rust paged path — same dispatch as the sync sibling
    /// `paged_turn_sync_core_inner`: prefill populates the GDN linear
    /// caches and writes K/V into the adapter pool, then decode steps run
    /// through `paged_forward::run_paged_decode_step` (or the eager paged
    /// MTP arm when the MTP gate holds).
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
        trace_turn_id: u64,
        turn_trace_start: Option<std::time::Instant>,
    ) -> Result<(
        Vec<u32>,
        String,
        Option<crate::decode_profiler::DecodeProfiler>,
    )> {
        // Invariant: caller-applied vLLM cap guarantees suffix_len > 0.
        debug_assert!(
            suffix_len > 0,
            "paged_turn_stream_core_inner: caller must cap max_cache_hit_tokens at prompt.len() - 1"
        );

        let suffix = &tokens[(cached_prefix_len as usize)..];
        let layer_kinds = crate::models::qwen3_5::decoder_layer::compute_layer_kinds(
            self.config.num_layers as usize,
            |i| self.config.is_linear_layer(i),
        );

        // Eager paged MTP needs the prompt-tail hidden for the
        // committed-history v2 seed, same as the sync core. Only capture it
        // when the eager paged MTP arm will actually run.
        let eager_mtp_paged = p.enable_mtp && self.has_mtp_weights();
        let want_prompt_hidden = eager_mtp_paged && cached_prefix_len == 0;
        let inference_info_enabled =
            tracing::enabled!(target: "mlx_core::inference", tracing::Level::INFO);
        let prefill_trace_start = inference_info_enabled.then(std::time::Instant::now);
        if inference_info_enabled {
            tracing::info!(
                target: "mlx_core::inference",
                event = "prefill_start",
                turn_id = trace_turn_id,
                suffix_tokens = suffix_len,
                cached_prefix_tokens = cached_prefix_len,
                mtp = eager_mtp_paged,
                keep_prompt_hidden_tokens = if want_prompt_hidden { tokens.len() } else { 0 },
                "prefill started"
            );
        }

        let mut mtp_profiler = begin_paged_mtp_profiler(eager_mtp_paged, suffix_len);
        let (last_logits, prompt_hidden, gdn_checkpoint) = self.run_dense_core_paged_prefill(
            tokens,
            suffix,
            cached_prefix_len,
            gdn_prefix_already_primed,
            want_prompt_hidden.then_some(tokens.len()),
            &layer_kinds,
            "paged_turn_stream_core_inner",
        )?;
        if let Some(profiler) = mtp_profiler.as_mut() {
            profiler.end_prefill();
        }
        self.publish_dense_gdn_materialized_prefix_checkpoint(tokens, p.cache_salt, gdn_checkpoint);

        let mut token_history: Vec<u32> = tokens.to_vec();
        let last_logits = apply_all_penalties(last_logits, &token_history, p)?;
        let mut y = sample(&last_logits, sampling_config)?;
        y.eval();

        if inference_info_enabled {
            let prefill_elapsed = prefill_trace_start.map(elapsed_ms).unwrap_or(0.0);
            tracing::info!(
                target: "mlx_core::inference",
                event = "first_token_sampled",
                turn_id = trace_turn_id,
                suffix_tokens = suffix_len,
                elapsed_ms = prefill_elapsed,
                materialized_prefill_tok_s = if prefill_elapsed > 0.0 {
                    suffix_len as f64 / (prefill_elapsed / 1000.0)
                } else {
                    0.0
                },
                "first token sampled"
            );
        }

        // Smooth memory peak: drop transient prefill buffers before decode
        // starts allocating (see paged_turn_sync_core_inner for rationale).
        crate::array::synchronize_and_clear_cache();

        if let Some(profiler) = mtp_profiler.as_mut() {
            profiler.mark_first_token();
        }
        if report_perf {
            *first_token_instant = Some(std::time::Instant::now());
        }

        let max_new_tokens = p.max_new_tokens;
        let mut generated_tokens: Vec<u32> =
            Vec::with_capacity(engine::generated_capacity_hint(max_new_tokens));
        let mut finish_reason = String::from("length");
        let decode_progress_start = std::time::Instant::now();
        let mut decode_progress_window_start = decode_progress_start;
        let mut decode_progress_last_generated = 0usize;
        let mut decode_progress_next_generated = 32usize;

        // Pre-cycle lookahead reservation while AR fallback is still an
        // option — same rationale as the sync twin.
        let eager_mtp_paged = eager_mtp_paged
            && self.reserve_paged_mtp_lookahead(p, "paged_turn_stream_core_inner")?;

        if eager_mtp_paged {
            // Pure-Rust ("eager") paged MTP — streaming twin of the sync core's
            // `eager_mtp_paged` arm. Same stepper spine; the engine's
            // `run_mtp_turn` streaming path emits decoded text per token via `cb`.
            MxArray::async_eval_arrays(&[&y]);

            let mut profiler = mtp_profiler.take().ok_or_else(|| {
                Error::from_reason(
                    "Qwen3.5 streaming paged MTP profiler was not initialized before decode",
                )
            })?;

            let eos_id = eos_token_id;
            let generation_stream = crate::stream::Stream::new(crate::stream::DeviceType::Gpu);
            let model_size_bytes = self.config.estimate_memory_bytes() as usize;
            let _wired_ctx =
                crate::stream::WiredLimitContext::new(model_size_bytes, vec![generation_stream]);

            let prompt_hidden_ids: Vec<u32> = tokens.to_vec();

            let mut rng = rand::rng();

            if inference_info_enabled {
                tracing::info!(
                    target: "mlx_core::inference",
                    event = "mtp_decode_start",
                    turn_id = trace_turn_id,
                    depth = p.mtp_depth,
                    prompt_hidden_tokens = prompt_hidden_ids.len(),
                    elapsed_ms = turn_trace_start.map(elapsed_ms).unwrap_or(0.0),
                    "MTP decode started"
                );
            }

            // Streaming twin of the sync paged arm: the propose/verify whole-turn
            // loop is engine-owned (`run_mtp_turn`) and drives the
            // `DenseMtpStepper` in its PAGED mode. The streaming sink wires the
            // shared incremental detokenizer + the default ChatML emitter so
            // accepted tokens stream out `cb` per token; the stepper's `Drop`
            // restores `self.paged_adapter` before this call returns (the
            // paged-history save below relies on it). The turn outcome's rewind
            // tail is consumed below (`last_in_cache` stays inert — dense paged
            // is drop-last; the paged-history save owns cache state).
            let mut emitter = crate::engine::backend::DefaultStreamEmitter;
            let streaming = crate::engine::decode::StreamingCtx {
                callback: cb.0,
                cancelled,
                decode_stream,
                tokenizer: tokenizer.inner(),
                streamed_text_len,
                last_is_reasoning,
                emitter: &mut emitter,
            };

            let outcome = crate::engine::mtp_turn::run_mtp_turn(
                self,
                &mut rng,
                crate::engine::mtp_turn::MtpTurnArgs {
                    y: y.clone(),
                    depth: p.mtp_depth,
                    params: p,
                    reasoning_tracker,
                    profiler: &mut profiler,
                    max_new_tokens,
                    eos_id,
                    generated_tokens: &mut generated_tokens,
                    token_history: &mut token_history,
                    finish_reason: &mut finish_reason,
                    first_token_instant,
                    report_perf: p.report_performance,
                    generation_stream,
                    prompt_hidden,
                    prompt_hidden_ids: Some(prompt_hidden_ids),
                    // H2: the same flag StreamingCtx carries — the engine's
                    // ungated polls and the streaming reads are idempotent.
                    cancel_flag: Some(cancelled),
                },
                Some(streaming),
            )?;
            // dense paged always saves drop-last, so `last_in_cache` is inert
            // here; the engine-computed rewind tail is recorded so the
            // epilogue's frontier check (and tests) can see the mid-cycle
            // stop was acted on rather than discarded.
            self.paged_mtp_last_rollback_unemitted = outcome.rollback_unemitted;

            return Ok((generated_tokens, finish_reason, Some(profiler)));
        }

        for step in 0..max_new_tokens {
            let token_id = y.item_at_int32(0)? as u32;
            generated_tokens.push(token_id);
            token_history.push(token_id);
            let is_reasoning = reasoning_tracker.observe_token(token_id);
            *last_is_reasoning = is_reasoning;

            if inference_info_enabled && generated_tokens.len() >= decode_progress_next_generated {
                let now = std::time::Instant::now();
                let window_tokens = generated_tokens
                    .len()
                    .saturating_sub(decode_progress_last_generated);
                let window_elapsed_ms = now
                    .saturating_duration_since(decode_progress_window_start)
                    .as_secs_f64()
                    * 1000.0;
                tracing::info!(
                    target: "mlx_core::inference",
                    event = "decode_progress",
                    mode = "ar",
                    generated_tokens = generated_tokens.len(),
                    window_tokens,
                    window_elapsed_ms,
                    window_tok_s = if window_elapsed_ms > 0.0 {
                        window_tokens as f64 / (window_elapsed_ms / 1000.0)
                    } else {
                        0.0
                    },
                    elapsed_ms = now
                        .saturating_duration_since(decode_progress_start)
                        .as_secs_f64()
                        * 1000.0,
                    "decode progress"
                );
                decode_progress_last_generated = generated_tokens.len();
                decode_progress_window_start = now;
                decode_progress_next_generated = generated_tokens
                    .len()
                    .saturating_div(32)
                    .saturating_add(1)
                    .saturating_mul(32);
            }

            if token_id == eos_token_id || p.extra_eos_ids.contains(&token_id) {
                finish_reason = String::from("stop");
                break;
            }
            if cancelled.load(Ordering::Relaxed) {
                finish_reason = String::from("cancelled");
                break;
            }

            // Stream delta chunk.
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

            // Decode forward (pure-Rust paged step).
            let next_logits = {
                let embed = self.embedding.clone();
                let caches_ref = self.caches.as_mut().ok_or_else(|| {
                    Error::from_reason("paged_turn_stream_core_inner: caches dropped mid-decode")
                })?;
                let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                    Error::from_reason(
                        "paged_turn_stream_core_inner: paged_adapter dropped mid-decode",
                    )
                })?;
                let logits = crate::models::qwen3_5::paged_forward::run_paged_decode_step(
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

            y = sample(&next_logits, sampling_config)?;
            y.eval();

            crate::array::maybe_clear_cache_for_paged_step(step);
        }

        Ok((generated_tokens, finish_reason, None))
    }
}
