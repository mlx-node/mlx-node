//! Load-time component setters and the image-bearing paged turn cores
//! (sync and stream) for Qwen3.5 MoE.

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
    pub(crate) fn set_vision_encoder(&mut self, enc: QwenVisionEncoder) -> Result<()> {
        self.vision_encoder = Some(Arc::new(enc));
        self.vision_cache = Arc::new(Mutex::new(VisionCacheInner::new()));
        Ok(())
    }

    /// Set the image processor.
    pub(crate) fn set_image_processor(&mut self, proc: QwenImageProcessor) {
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
        let scaling = self.config.scaling();
        for layer in self.layers.iter_mut() {
            if let crate::models::qwen3_5_moe::decoder_layer::AttentionType::Full(ref mut attn) =
                layer.attn
            {
                attn.init_mrope(
                    mrope_section.clone(),
                    rope_theta,
                    max_position_embeddings,
                    rope_dims,
                    &scaling,
                )?;
            }
        }
        Ok(())
    }

    /// Single-turn image-bearing block-paged dispatch (non-streaming).
    ///
    /// It processes the images,
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

        let think_end_id = tokenizer.think_end_id();
        let think_end_str = tokenizer.think_end_str().map(|s| s.to_string());
        let thinking_enabled = thinking.enabled;
        let mut reasoning_tracker =
            engine::ReasoningTracker::from_setup(&thinking, &tokenizer.think_end());

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

        let generation_stream = Stream::generation();

        // Held until the prefill that reads `merge` has run: a reclaim takes
        // only features nothing holds (Splash `Runtime::prefillAsync` gives
        // an image's rows to the cache once they are in).
        let (merge, held_images) = vlm_prepare_vision_features(
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
                self.vlm_lane_prefill(
                    &expanded_tokens,
                    &lookup_extra_keys,
                    &merge,
                    cached_prefix_len,
                    gdn_prefix_already_primed,
                    p.reuse_cache,
                    &layer_kinds,
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
            drop(held_images);

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
                reasoning_tracker.enforce_next_token(&mut y)?;
                let token_id = y.item_at_int32(0)? as u32;
                generated_tokens.push(token_id);
                token_history.push(token_id);
                reasoning_tracker.observe_token(token_id);

                if token_id == eos_token_id || p.extra_eos_ids.contains(&token_id) {
                    finish_reason = String::from("stop");
                    break;
                }
                // Sync cancel poll — the SAME snapshot point as the MoE vision
                // paged streaming twin: after the EOS check, before the
                // repetition cutoff.
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
                    let logits =
                        crate::models::qwen3_5::dflash2_turn::paged_step_claimed(self, |inner| {
                            let caches_ref = inner.caches.as_mut().ok_or_else(|| {
                                Error::from_reason(
                                    "vision_paged_turn_sync_core: caches dropped mid-decode",
                                )
                            })?;
                            let adapter = inner.paged_adapter.as_mut().ok_or_else(|| {
                                Error::from_reason(
                                    "vision_paged_turn_sync_core: paged_adapter dropped mid-decode",
                                )
                            })?;
                            crate::models::qwen3_5_moe::paged_forward::run_paged_decode_step(
                                token_id,
                                &embed,
                                &mut inner.layers,
                                caches_ref,
                                &inner.final_norm,
                                &inner.lm_head,
                                &layer_kinds,
                                adapter,
                                inner.cached_rope_deltas.unwrap_or(0),
                            )
                        })?;
                    logits.squeeze(Some(&[1]))?
                };

                if let Some(forced_id) = reasoning_tracker.take_forced_token() {
                    let forced_id = forced_id as i32;
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
        let mut reasoning_tracker =
            engine::ReasoningTracker::from_setup(&thinking, &tokenizer.think_end());

        let generation_start = if report_perf {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut first_token_instant: Option<std::time::Instant> = None;
        let sampling_config = p.sampling_config;

        // One streaming bundle — detokenizer + cursors + emitter (MoE does
        // not override `stream_emitter`, so it is the default ChatML
        // emitter — byte-identical to the former inline emit).
        let mut turn_streaming = crate::engine::decode::TurnStreaming::new(
            self,
            tokenizer.inner(),
            thinking_enabled,
            true,
        );

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

        let generation_stream = Stream::generation();

        // Held until the prefill that reads `merge` has run: a reclaim takes
        // only features nothing holds (Splash `Runtime::prefillAsync` gives
        // an image's rows to the cache once they are in).
        let (merge, held_images) = vlm_prepare_vision_features(
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

        let forward_result = (|| -> Result<(Vec<u32>, String)> {
            // === PREFILL ===
            let last_logits = {
                let _stream_ctx = StreamContext::new(generation_stream);
                self.vlm_lane_prefill(
                    &expanded_tokens,
                    &lookup_extra_keys,
                    &merge,
                    cached_prefix_len,
                    gdn_prefix_already_primed,
                    p.reuse_cache,
                    &layer_kinds,
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
            drop(held_images);

            crate::array::synchronize_and_clear_cache();
            if report_perf {
                first_token_instant = Some(std::time::Instant::now());
            }

            let max_new_tokens = p.max_new_tokens;
            let mut generated_tokens: Vec<u32> =
                Vec::with_capacity(engine::generated_capacity_hint(max_new_tokens));
            let mut finish_reason = String::from("length");

            for step in 0..max_new_tokens {
                reasoning_tracker.enforce_next_token(&mut y)?;
                let token_id = y.item_at_int32(0)? as u32;
                generated_tokens.push(token_id);
                token_history.push(token_id);
                let is_reasoning = reasoning_tracker.observe_token(token_id);
                turn_streaming.last_is_reasoning = is_reasoning;

                if token_id == eos_token_id || p.extra_eos_ids.contains(&token_id) {
                    finish_reason = String::from("stop");
                    break;
                }
                if cancelled.load(Ordering::Relaxed) {
                    finish_reason = String::from("cancelled");
                    break;
                }

                let token_text = Qwen3Tokenizer::step_decode_stream(
                    &mut turn_streaming.decode_stream,
                    turn_streaming.tokenizer,
                    token_id,
                    &generated_tokens,
                    turn_streaming.streamed_text_len,
                );
                turn_streaming.streamed_text_len += token_text.len();
                cb.count_token();
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
                    let logits =
                        crate::models::qwen3_5::dflash2_turn::paged_step_claimed(self, |inner| {
                            let caches_ref = inner.caches.as_mut().ok_or_else(|| {
                                Error::from_reason(
                                    "vision_paged_turn_stream_core: caches dropped mid-decode",
                                )
                            })?;
                            let adapter = inner.paged_adapter.as_mut().ok_or_else(|| {
                        Error::from_reason(
                            "vision_paged_turn_stream_core: paged_adapter dropped mid-decode",
                        )
                    })?;
                            crate::models::qwen3_5_moe::paged_forward::run_paged_decode_step(
                                token_id,
                                &embed,
                                &mut inner.layers,
                                caches_ref,
                                &inner.final_norm,
                                &inner.lm_head,
                                &layer_kinds,
                                adapter,
                                inner.cached_rope_deltas.unwrap_or(0),
                            )
                        })?;
                    logits.squeeze(Some(&[1]))?
                };

                if let Some(forced_id) = reasoning_tracker.take_forced_token() {
                    let forced_id = forced_id as i32;
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

        // Flush residual buffered bytes (mirrors flat / text paged
        // streaming) through the bundle's emitter — the default ChatML
        // `on_residual` suppresses reasoning text when include_reasoning
        // == false and emits the same chunk shape the inline flush built.
        let full_text = tokenizer
            .decode_sync(&generated_tokens, true)
            .unwrap_or_else(|e| {
                tracing::warn!("Failed to decode generated tokens: {}", e);
                String::new()
            });
        turn_streaming.flush_residual(&full_text, include_reasoning, cb.0);

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

impl Qwen35MoeInner {
    /// The image prefill of `expanded_tokens` from `cached`, whose blocks
    /// carry `extra_keys` under the salt its lookup used (0): its chunks
    /// reach the turn's state points
    /// ([`crate::models::qwen3_5::dflash2_turn::image_lane_prefill`]).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn vlm_lane_prefill(
        &mut self,
        expanded_tokens: &[u32],
        extra_keys: &[Vec<u64>],
        merge: &crate::engine::vision::VisionMerge,
        cached: u32,
        gdn_prefix_already_primed: bool,
        reuse: bool,
        layer_kinds: &[super::super::decoder_layer::Qwen3_5LayerKind],
    ) -> Result<(
        MxArray,
        Vec<crate::models::qwen3_5::paged_forward::MaterializedGdnPrefixCheckpoint>,
    )> {
        let embed = self.embedding.clone();
        let turn_cancel = self.turn_cancel.clone();
        let seq = crate::models::qwen3_5::dflash2_turn::DFlash2Target::lane_seq(self);
        let restored = match self.gdn_in_use.get(&seq) {
            Some(crate::engine::governor::ledger::UnitKey::Store(id)) => Some(*id),
            _ => None,
        };
        let chunk_rows = crate::models::qwen3_5::paged_forward::prefill_chunk_rows(
            self.cold_gdn_prefill_chunk_size(),
        );
        crate::models::qwen3_5::dflash2_turn::image_lane_prefill(
            self,
            expanded_tokens,
            extra_keys,
            cached,
            0,
            reuse,
            restored,
            chunk_rows,
            |inner| inner.dflash2.take(),
            |inner, draft| inner.dflash2 = Some(draft),
            |target, sink| {
                crate::models::qwen3_5_moe::paged_forward::run_paged_vlm_prefill_moe(
                    expanded_tokens,
                    merge,
                    cached,
                    gdn_prefix_already_primed,
                    &embed,
                    target,
                    layer_kinds,
                    turn_cancel.as_deref(),
                    Some(sink),
                )
            },
        )
    }
}
