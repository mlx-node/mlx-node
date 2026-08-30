//! Load-time component setters and the image-bearing turn cores for
//! Qwen3.5 MoE (the VLM whole-turn entry plus its paged sync/stream cores).

use super::*;

impl Qwen35MoeInner {
    /// Set the tokenizer.
    pub(crate) fn set_tokenizer(&mut self, tokenizer: Arc<Qwen3Tokenizer>) {
        self.tokenizer = Some(tokenizer);
    }

    /// Set the vision encoder.
    ///
    /// Paged VLM checkpoints wire this alongside the image processor and
    /// adapter. Image-bearing turns then route through the MoE paged-vision
    /// cores; incomplete stacks stay backend-validated so the family reports
    /// its precise missing component. Text-only inputs still use scalar RoPE
    /// on both flat and paged paths.
    pub(crate) fn set_vision_encoder(&mut self, enc: Qwen3_5VisionEncoder) -> Result<()> {
        self.vision_encoder = Some(Arc::new(enc));
        self.vision_cache = Arc::new(Mutex::new(VisionCacheInner::new()));
        Ok(())
    }

    /// Set the image processor.
    pub(crate) fn set_image_processor(&mut self, proc: Qwen35VLImageProcessor) {
        self.image_processor = Some(Arc::new(proc));
        self.vision_cache = Arc::new(Mutex::new(VisionCacheInner::new()));
    }

    /// Set spatial merge size.
    pub(crate) fn set_spatial_merge_size(&mut self, size: i32) {
        self.spatial_merge_size = Some(size);
        self.vision_cache = Arc::new(Mutex::new(VisionCacheInner::new()));
    }

    /// Initialize M-RoPE on all full attention layers (VLM mode).
    pub(crate) fn init_mrope_layers(
        &mut self,
        mrope_section: Vec<i32>,
        rope_theta: f64,
        max_position_embeddings: i32,
    ) -> Result<()> {
        let rope_dims = self.config.rope_dims();
        for layer in self.layers.iter_mut() {
            if let crate::models::qwen3_5_moe::decoder_layer::AttentionType::Full(ref mut attn) =
                layer.attn
            {
                attn.init_mrope(
                    mrope_section.clone(),
                    rope_theta,
                    max_position_embeddings,
                    rope_dims,
                )?;
            }
        }
        Ok(())
    }

    /// Core chat implementation (runs on model thread).
    ///
    /// Whole-turn core for fresh SYNC turns reached through the engine's
    /// `vision_turn` (image-bearing) and `mtp_turn` (MTP-enabled)
    /// probes. The engine already rendered the prompt (`tokens`) and
    /// extracted the raw image payloads (`images`); everything from the
    /// paged dispatch onward runs the whole-turn pipeline.
    /// `eos_token_id` is the caller-supplied stop-on token id (typically
    /// `<|im_end|>`) so the cached history ends on a clean ChatML
    /// boundary, yielding a reusable prefix for subsequent session
    /// deltas.
    pub(super) fn vision_mtp_whole_turn_core(
        &mut self,
        tokens: Vec<u32>,
        images: &[Vec<u8>],
        config: ChatConfig,
        eos_token_id: u32,
        thinking: ThinkingSetup,
    ) -> Result<ChatResult> {
        let reuse_cache = config.reuse_cache.unwrap_or(true);
        let report_perf = config.report_performance.unwrap_or(false);

        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))?
            .clone();

        let has_images = !images.is_empty();

        let think_end_id = tokenizer.think_end_id();
        let think_end_str = tokenizer.think_end_str().map(|s| s.to_string());

        let mut p = extract_chat_params(&config);
        p.extra_eos_ids = self.gen_defaults.eos_token_ids.clone();
        let max_new_tokens = p.max_new_tokens;

        let generation_start = if report_perf {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut first_token_instant: Option<std::time::Instant> = None;

        // Block-paged dispatch — early-return onto the paged core.
        if self.paged_adapter.is_some() {
            if has_images {
                // All image turns (MTP or not) prefill through the paged-vision
                // core. The eager paged MTP stepper has no M-RoPE prefill seed,
                // so the core decodes plain autoregressively regardless of the
                // per-request MTP flag — it never reads the MTP head.
                return self.vision_paged_turn_sync_core(
                    tokens,
                    images,
                    tokenizer,
                    eos_token_id,
                    p,
                    report_perf,
                    thinking,
                );
            }
            return self.paged_turn_sync_core(
                tokens,
                tokenizer,
                eos_token_id,
                p,
                report_perf,
                thinking,
            );
        }

        // The flat fallback below is text-only. A MoE image turn requires the
        // block-paged backend; reaching here with images means the model was
        // loaded without a paged adapter (use_block_paged_cache=false or a
        // non-Metal build).
        if has_images {
            return Err(Error::from_reason(
                "qwen3.5 MoE image turns require the block-paged KV backend; the model was \
                 loaded without a paged adapter (use_block_paged_cache=false or non-Metal \
                 build)",
            ));
        }

        // Pure-Rust eager MTP. Active when the per-request flag is set and the
        // checkpoint carries an MTP head; runs the speculative-decode arm on
        // `Qwen35MoeInner`'s flat caches. Text-only flat turns only (the paged
        // dispatch already early-returned above).
        let eager_mtp =
            p.enable_mtp && self.has_mtp_weights() && self.paged_adapter.is_none() && !has_images;

        let embedding = self.embedding.clone();

        // Text-only from here: the `has_images` early-return above is the only
        // image path. These bindings preserve the shared cache-reuse / decode
        // plumbing (`has_images` is always false on this branch).
        let (expanded_tokens, current_image_cache_key) = (tokens.clone(), 0u64);

        // === Cache reuse: prefix verification ===
        let cached_prefix_len = if self.flat_mtp_caches_desynced {
            0
        } else {
            verify_cache_prefix_direct(
                reuse_cache,
                has_images,
                &tokens,
                &expanded_tokens,
                current_image_cache_key,
                &self.cached_token_history,
                &self.cached_image_key,
                self.caches.is_some(),
            )
        };

        let prefill_tokens = if cached_prefix_len > 0 {
            if has_images {
                info!(
                    "VLM cache reuse: {} cached tokens, {} new tokens to prefill",
                    cached_prefix_len,
                    expanded_tokens.len() - cached_prefix_len
                );
                expanded_tokens[cached_prefix_len..].to_vec()
            } else {
                info!(
                    "Cache reuse: {} cached tokens, {} new tokens to prefill",
                    cached_prefix_len,
                    tokens.len() - cached_prefix_len
                );
                tokens[cached_prefix_len..].to_vec()
            }
        } else {
            // Full reset
            if let Some(ref mut caches) = self.caches {
                for cache in caches.iter_mut() {
                    cache.reset();
                }
            }
            let new_caches = (0..self.config.num_layers as usize)
                .map(|i| {
                    if self.config.is_linear_layer(i) {
                        Qwen3_5LayerCache::new_linear()
                    } else {
                        Qwen3_5LayerCache::new_full_attention()
                    }
                })
                .collect();
            self.caches = Some(new_caches);
            tokens.clone()
        };

        // Zero-delta guard.
        //
        // Triggers when `cached_prefix_len == (expanded_)tokens.len()`, i.e.
        // the new prompt is byte-for-byte identical to the cached history
        // and there is literally no delta to prefill. We still need to
        // produce a `last_logits` for the decode loop, and the only safe
        // way to do that on the Qwen3.5 MoE hybrid stack is a full reset
        // + re-prefill. Trimming the cache by one token is infeasible
        // because the 30 GDN linear-attention layers carry a recurrent
        // state that cannot be rewound mid-sequence (see the invariant
        // doc on `verify_cache_prefix_direct`). In practice this branch
        // is a cold edge case — real agent turns always append at least
        // a user message, so the cached prefix is strictly shorter than
        // the new prompt.
        let (prefill_tokens, cached_prefix_len) = if prefill_tokens.is_empty() {
            info!("Zero-delta cache hit: resetting caches for full re-prefill");
            if let Some(ref mut caches) = self.caches {
                for cache in caches.iter_mut() {
                    cache.reset();
                }
            }
            let new_caches = (0..self.config.num_layers as usize)
                .map(|i| {
                    if self.config.is_linear_layer(i) {
                        Qwen3_5LayerCache::new_linear()
                    } else {
                        Qwen3_5LayerCache::new_full_attention()
                    }
                })
                .collect();
            self.caches = Some(new_caches);
            let tokens = if has_images {
                expanded_tokens.clone()
            } else {
                tokens.clone()
            };
            (tokens, 0)
        } else {
            (prefill_tokens, cached_prefix_len)
        };

        let eos_id = eos_token_id;
        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut finish_reason = String::from("length");

        // Track token history for repetition penalty
        let mut token_history: Vec<u32> = expanded_tokens.clone();

        let generation_stream = Stream::new(DeviceType::Gpu);
        let model_size_bytes = self.config.estimate_memory_bytes() as usize;
        let _wired_ctx =
            crate::stream::WiredLimitContext::new(model_size_bytes, vec![generation_stream]);

        // StreamContext created ONCE for entire prefill+decode
        let _stream_ctx = StreamContext::new(generation_stream);

        let fa_idx = self.fa_idx;

        // Profiler
        let mut profiler = crate::decode_profiler::DecodeProfiler::new("moe_chat", "qwen3_5_moe");
        profiler.set_prompt_tokens(prefill_tokens.len() as u32);
        profiler.snapshot_memory_before();

        // === Text prefill ===
        // Image turns never reach here — they early-return onto the paged-vision
        // core (or error when no paged adapter is present). This is the
        // text-only flat path.
        profiler.begin_prefill();
        let turn_cancel = self.turn_cancel.clone();
        let (mut last_logits, _seq_len) = {
            // Standard text prefill. Chunked to bound peak GPU memory for
            // long prompts (e.g. 40k+ tokens) — see `chunked_prefill` docs.
            let prompt = MxArray::from_uint32(&prefill_tokens, &[1, prefill_tokens.len() as i64])?;

            let prefill_result = chunked_prefill(
                &prompt,
                &embedding,
                &mut self.layers,
                &mut self.caches,
                &self.final_norm,
                &self.lm_head,
                fa_idx,
                generation_stream,
                turn_cancel.as_deref(),
            );
            // A partially advanced prefill (cancel or failure) must never be
            // continued: `self.caches` would hold the partial delta while
            // `cached_token_history` still describes the previous turn.
            let logits = match prefill_result {
                Ok(logits) => logits,
                Err(e) => {
                    self.invalidate_moe_paged_session("MTP whole-turn flat prefill failure");
                    return Err(e);
                }
            };

            let seq_len = logits.shape_at(1)?;
            let last_logits = logits.slice_axis(1, seq_len - 1, seq_len)?;
            let last_logits = last_logits.squeeze(Some(&[1]))?;
            (last_logits, tokens.len() as i64)
        };
        profiler.end_prefill();
        // caches now reflect the prefilled history
        self.flat_mtp_caches_desynced = false;

        last_logits = apply_all_penalties(last_logits, &token_history, &p)?;
        let mut y = sample(&last_logits, p.sampling_config)?;
        MxArray::async_eval_arrays(&[&y]);

        let mut reasoning_tracker = engine::ReasoningTracker::from_setup(&thinking, think_end_id);

        // Whether the final committed token reached the physical KV/GDN cache;
        // written by the decode driver so the save below drops it when it was
        // never forwarded (unforwarded stop token).
        let mut last_in_cache = true;

        if eager_mtp {
            // Pure-Rust eager MoE MTP — the propose/verify whole-turn loop is
            // engine-owned (`engine::run_mtp_turn`) and drives the
            // `MoeMtpStepper` (`MtpBackend::begin_mtp_decode`). Committed-
            // history v2 activates within-turn (persistent drafter cache
            // across cycles) when its opt-in flag is on; the prompt-prefix seed
            // itself stays inert here because this call site has no MoE
            // hidden-emitting prefill yet, so `prompt_hidden`/`prompt_hidden_ids`
            // stay `None`. The `profiler.set_label("moe_mtp_eager")` relabel
            // moved into `MoeMtpStepper::profiler_relabel`.
            let mut rng = rand::rng();
            MxArray::async_eval_arrays(&[&y]);

            let outcome = crate::engine::mtp_turn::run_mtp_turn(
                self,
                &mut rng,
                crate::engine::mtp_turn::MtpTurnArgs {
                    y: y.clone(),
                    depth: p.mtp_depth,
                    params: &p,
                    reasoning_tracker: &mut reasoning_tracker,
                    profiler: &mut profiler,
                    max_new_tokens,
                    eos_id,
                    generated_tokens: &mut generated_tokens,
                    token_history: &mut token_history,
                    finish_reason: &mut finish_reason,
                    first_token_instant: &mut first_token_instant,
                    report_perf: p.report_performance,
                    generation_stream,
                    prompt_hidden: None,
                    prompt_hidden_ids: None,
                    // H2: sync turns cancel through the engine loop's
                    // ungated polls (this site has no StreamingCtx).
                    cancel_flag: turn_cancel.as_deref(),
                },
                None,
            )?;

            last_in_cache = outcome.last_in_cache;
            // Propagate a mid-cycle stop: self.caches advanced past the emitted
            // history, so force a full re-prefill next turn.
            if outcome.desynced {
                self.flat_mtp_caches_desynced = true;
            }
        } else {
            // Rust fallback decode loop
            profiler.set_label("moe_chat_rust");

            let mut ops = mtp_decode::DecodeOps {
                forward: |ids: &MxArray, emb: &Embedding| -> Result<(MxArray, bool)> {
                    let logits = forward_inner(
                        ids,
                        emb,
                        &mut self.layers,
                        &mut self.caches,
                        &self.final_norm,
                        &self.lm_head,
                        fa_idx,
                    )?;
                    Ok((logits, true))
                },
                eval_step: |token: &MxArray, logits: &MxArray, _budget_forced: bool| {
                    MxArray::async_eval_arrays(&[token, logits]);
                },
            };
            mtp_decode::decode_loop!(
                ops: ops,
                y: y,
                embedding_weight: embedding,
                params: p,
                reasoning_tracker: reasoning_tracker,
                profiler: profiler,
                max_new_tokens: max_new_tokens,
                eos_id: eos_id,
                generated_tokens: generated_tokens,
                token_history: token_history,
                finish_reason: finish_reason,
                last_in_cache: last_in_cache,
                first_token_instant: first_token_instant,
                report_perf: p.report_performance,
                generation_stream: generation_stream,
                cancel: turn_cancel.as_deref()
            );
        }

        // Save cache state
        save_cache_state_direct(
            p.reuse_cache,
            has_images,
            &generated_tokens,
            &finish_reason,
            /* drop_last_always */ !last_in_cache,
            &tokens,
            Some(&expanded_tokens),
            current_image_cache_key,
            &mut self.cached_token_history,
            &mut self.cached_image_key,
            &mut self.cached_rope_deltas,
            &mut self.caches,
        );
        self.cached_paged_image_token_positions.clear();

        let performance = compute_performance_metrics(
            generation_start,
            first_token_instant,
            prefill_tokens.len(),
            generated_tokens.len(),
        )
        .map(|mut m| {
            profiler.fill_mtp_acceptance(&mut m);
            m
        });

        let mut result = finalize_chat_result(
            &tokenizer,
            &generated_tokens,
            finish_reason,
            think_end_id,
            think_end_str.as_deref(),
            performance,
            p.include_reasoning,
            thinking.enabled,
            if has_images {
                expanded_tokens.len() as u32
            } else {
                tokens.len() as u32
            },
            reasoning_tracker.reasoning_token_count(),
        )?;
        // Report the length of the reused cached prefix for observability.
        // `cached_prefix_len` is 0 on fresh/miss paths and the full cached
        // length on an exact-append hit — see the invariant doc on
        // `verify_cache_prefix_direct`.
        result.cached_tokens = cached_prefix_len as u32;
        Ok(result)
    }

    /// Single-turn image-bearing block-paged dispatch (non-streaming).
    ///
    /// The paged sibling of the flat MoE VLM prefill: it processes the images,
    /// merges the vision features into the token embeddings, computes M-RoPE
    /// positions, then prefills through the paged adapter via
    /// [`crate::models::qwen3_5_moe::paged_forward::run_paged_vlm_prefill_moe`] and runs the plain
    /// autoregressive decode loop.
    ///
    /// Same-image live histories continue in place; fresh histories look up
    /// full blocks using per-image content keys and prefill only the uncached
    /// suffix. Decode uses the image M-RoPE delta carried by the merged prompt.
    /// MTP is not supported here; image-bearing turns decode autoregressively.
    #[allow(clippy::too_many_arguments)]
    fn vision_paged_turn_sync_core(
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
        let prefix_resolution = self.prepare_moe_vlm_paged_prefix(
            &expanded_tokens,
            total_budget,
            block_size,
            &lookup_extra_keys,
            p.reuse_cache,
            p.reuse_cache && same_live_image,
            image_cache_key,
        )?;
        let plan = prefix_resolution.effective_plan;
        let cached_prefix_len = plan.cached_prefix_len;
        tracing::info!(
            target: "mlx_core::inference",
            event = "vlm_prefix_plan",
            model = "qwen3_5_moe",
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
                crate::models::qwen3_5_moe::paged_forward::run_paged_vlm_prefill_moe(
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
            self.publish_moe_gdn_materialized_prefix_checkpoint(
                &expanded_tokens,
                &lookup_extra_keys,
                0,
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
                // H2 sync cancel poll — the SAME snapshot point as the MoE
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
        // (`paged_turn_sync_core`). The error path always releases the request
        // and returns. The success path is resolved below so the session ends
        // in exactly one of two states, never partial: FULLY continuable
        // (keep-live registered AND GDN checkpoint stored AND history + image
        // key published) or NON-continuable (request released AND history
        // cleared AND image key None) so a follow-up text continue is rejected
        // instead of cold-prefilling image-placeholder ids as ordinary tokens.
        let (generated_tokens, finish_reason) = match forward_result {
            Ok(t) => t,
            Err(e) => {
                self.invalidate_moe_paged_session("VLM sync prefill/decode failure");
                return Err(e);
            }
        };

        // Saved history: expanded prompt + generated[..len-1] (drop-last rule
        // shared with the text paged core: the decode loop never forwards the
        // final sampled token into the cache).
        let mut full_history = expanded_tokens.clone();
        if !generated_tokens.is_empty() {
            full_history.extend_from_slice(&generated_tokens[..generated_tokens.len() - 1]);
        }

        // Keep-live before the GDN checkpoint (which snapshots the live
        // recurrent state); short-circuit `&&` preserves that order. The
        // checkpoint reads `cached_token_history`, so publish it first, then
        // checkpoint. Any failure downgrades to NON-continuable rather than
        // discarding the already-successful generation output.
        let keep_live_ok = p.reuse_cache
            && self
                .finalize_moe_manual_paged_turn(&image_token_positions, p.cache_salt)
                .is_ok();
        let continuable = if keep_live_ok {
            self.cached_token_history = full_history;
            self.cached_image_key = Some(image_cache_key);
            self.cached_paged_image_token_positions = image_token_positions.clone();
            match self.remember_moe_gdn_history_checkpoint() {
                Ok(_) => true,
                Err(error) => {
                    tracing::warn!(
                        target: "mlx_core::qwen3_5_moe::paged",
                        "MoE VLM GDN history checkpoint failed: {error}",
                    );
                    self.invalidate_moe_paged_session("VLM sync GDN history checkpoint failure");
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
                self.invalidate_moe_paged_session("non-continuable VLM sync completion");
            } else {
                self.discard_moe_paged_session();
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
    /// spine; MTP is rejected upstream.
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
        let prefix_resolution = self.prepare_moe_vlm_paged_prefix(
            &expanded_tokens,
            total_budget,
            block_size,
            &lookup_extra_keys,
            p.reuse_cache,
            p.reuse_cache && same_live_image,
            image_cache_key,
        )?;
        let plan = prefix_resolution.effective_plan;
        let cached_prefix_len = plan.cached_prefix_len;
        tracing::info!(
            target: "mlx_core::inference",
            event = "vlm_prefix_plan",
            model = "qwen3_5_moe",
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
                crate::models::qwen3_5_moe::paged_forward::run_paged_vlm_prefill_moe(
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
            self.publish_moe_gdn_materialized_prefix_checkpoint(
                &expanded_tokens,
                &lookup_extra_keys,
                0,
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
                self.invalidate_moe_paged_session("VLM stream prefill/decode failure");
                return Err(e);
            }
        };

        // Saved history: expanded prompt + generated[..len-1] (drop-last rule).
        let mut full_history = expanded_tokens.clone();
        if !generated_tokens.is_empty() {
            full_history.extend_from_slice(&generated_tokens[..generated_tokens.len() - 1]);
        }

        // Keep-live before the GDN checkpoint (which snapshots the live state);
        // checkpoint reads `cached_token_history`, so publish it first. Any
        // failure downgrades to NON-continuable rather than discarding output.
        let keep_live_ok = p.reuse_cache
            && self
                .finalize_moe_manual_paged_turn(&image_token_positions, p.cache_salt)
                .is_ok();
        let continuable = if keep_live_ok {
            self.cached_token_history = full_history;
            self.cached_image_key = Some(image_cache_key);
            self.cached_paged_image_token_positions = image_token_positions.clone();
            match self.remember_moe_gdn_history_checkpoint() {
                Ok(_) => true,
                Err(error) => {
                    tracing::warn!(
                        target: "mlx_core::qwen3_5_moe::paged",
                        "MoE streaming VLM GDN history checkpoint failed: {error}",
                    );
                    self.invalidate_moe_paged_session("VLM stream GDN history checkpoint failure");
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
                self.invalidate_moe_paged_session("non-continuable VLM stream completion");
            } else {
                self.discard_moe_paged_session();
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
}
