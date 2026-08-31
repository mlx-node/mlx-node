//! Flat (non-paged) whole-turn cores for Qwen3.5 MoE plus the raw generate
//! surface: the fresh streaming entry, the delta sync/stream cores, and
//! `generate_sync`.

use super::*;

impl Qwen35MoeInner {
    /// Core streaming chat implementation (runs on model thread).
    ///
    /// Whole-turn core for fresh STREAMING turns reached through the
    /// engine's `vision_turn` (image-bearing) and `mtp_turn`
    /// (MTP-enabled) probes. The engine already rendered the prompt
    /// (`tokens`) and extracted the raw image payloads (`images`);
    /// everything from the paged dispatch onward runs the whole-turn
    /// pipeline. `eos_token_id` is the caller-supplied
    /// stop-on token id (typically `<|im_end|>`) so the cached history
    /// ends on a clean ChatML boundary, yielding a reusable prefix for
    /// subsequent session deltas.
    pub(super) fn vision_mtp_whole_turn_stream_core(
        &mut self,
        tokens: Vec<u32>,
        images: &[Vec<u8>],
        config: ChatConfig,
        eos_token_id: u32,
        cb: &StreamSender<'_>,
        cancelled: &AtomicBool,
        thinking: ThinkingSetup,
    ) -> Result<()> {
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
        let tokenizer_for_decode = tokenizer.clone();

        let mut p = engine::extract_chat_params(&config);
        p.extra_eos_ids = self.gen_defaults.eos_token_ids.clone();

        let generation_start = if report_perf {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut first_token_instant: Option<std::time::Instant> = None;

        // Block-paged dispatch — early-return BEFORE the compile lock.
        if self.paged_adapter.is_some() {
            if has_images {
                // All image turns (MTP or not) prefill through the paged-vision
                // stream core. It decodes plain autoregressively regardless of
                // the per-request MTP flag — it never reads the MTP head.
                return self.vision_paged_turn_stream_core(
                    tokens,
                    images,
                    tokenizer_for_decode,
                    eos_token_id,
                    p,
                    report_perf,
                    cb,
                    cancelled,
                    thinking,
                );
            }
            return self.paged_turn_stream_core(
                tokens,
                tokenizer_for_decode,
                eos_token_id,
                p,
                report_perf,
                cb,
                cancelled,
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

        // Pure-Rust eager MTP. Text-only flat turns only (paged already
        // early-returned above).
        let eager_mtp =
            p.enable_mtp && self.has_mtp_weights() && self.paged_adapter.is_none() && !has_images;

        let embedding = self.embedding.clone();

        // Text-only from here: the `has_images` early-return above is the only
        // image path. These bindings preserve the shared cache-reuse / decode
        // plumbing (`has_images` is always false on this branch).
        let (expanded_tokens, current_image_cache_key) = (tokens.clone(), 0u64);

        // Cache reuse
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

        // Zero-delta guard. See the matching `vision_mtp_whole_turn_core` comment for
        // the design rationale — rewinding a GDN recurrent cache by one
        // token is not possible across Qwen3.5 MoE's 30 linear-attention
        // layers, so the only safe response to an exact-match prompt is
        // a full reset + re-prefill.
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
        let mut decode_stream = tokenizer_for_decode.inner().decode_stream(true);
        let mut streamed_text_len: usize = 0;

        let generation_stream = Stream::new(DeviceType::Gpu);
        let model_size_bytes = self.config.estimate_memory_bytes() as usize;
        let _wired_ctx =
            crate::stream::WiredLimitContext::new(model_size_bytes, vec![generation_stream]);

        let fa_idx = self.fa_idx;

        let mut profiler =
            crate::decode_profiler::DecodeProfiler::new("moe_chat_stream", "qwen3_5_moe");
        profiler.set_prompt_tokens(prefill_tokens.len() as u32);
        profiler.snapshot_memory_before();

        // Text prefill. Image turns never reach here — they early-return onto
        // the paged-vision stream core (or error when no paged adapter is
        // present). This is the text-only flat path.
        profiler.begin_prefill();
        let turn_cancel = self.turn_cancel.clone();
        let (mut last_logits, _seq_len) = {
            // Chunked to bound peak GPU memory for long prompts. See
            // `chunked_prefill` docs for the memory rationale.
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
                    self.invalidate_moe_paged_session("MTP stream flat prefill failure");
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

        let mut token_history: Vec<u32> = tokens.clone();
        last_logits = apply_all_penalties(last_logits, &token_history, &p)?;
        let mut y = sample(&last_logits, p.sampling_config)?;
        MxArray::async_eval_arrays(&[&y]);

        let starts_in_thinking = thinking.enabled;
        let mut last_is_reasoning = starts_in_thinking;
        let mut reasoning_tracker = engine::ReasoningTracker::from_setup(&thinking, think_end_id);

        // Whether the final committed token reached the physical KV/GDN cache;
        // written by the decode driver so the save below drops it when it was
        // never forwarded (unforwarded stop token).
        let mut last_in_cache = true;

        if eager_mtp {
            // Streaming eager MoE MTP — same engine-owned `run_mtp_turn` loop +
            // `MoeMtpStepper` as the sync site, with a `StreamingCtx` wired so
            // accepted tokens stream out the `cb` sink incrementally (qwen3_5
            // MoE does not override `stream_emitter`, so the default ChatML
            // emitter is byte-identical to the former inline emit).
            let mut rng = rand::rng();
            MxArray::async_eval_arrays(&[&y]);

            let mut emitter = crate::engine::backend::DefaultStreamEmitter;
            let streaming = crate::engine::decode::StreamingCtx {
                callback: cb.0,
                cancelled,
                decode_stream: &mut decode_stream,
                tokenizer: tokenizer_for_decode.inner(),
                streamed_text_len: &mut streamed_text_len,
                last_is_reasoning: &mut last_is_reasoning,
                emitter: &mut emitter,
            };

            let outcome = crate::engine::mtp_turn::run_mtp_turn(
                self,
                &mut rng,
                crate::engine::mtp_turn::MtpTurnArgs {
                    y: y.clone(),
                    depth: p.mtp_depth,
                    params: &p,
                    reasoning_tracker: &mut reasoning_tracker,
                    profiler: &mut profiler,
                    max_new_tokens: p.max_new_tokens,
                    eos_id,
                    generated_tokens: &mut generated_tokens,
                    token_history: &mut token_history,
                    finish_reason: &mut finish_reason,
                    first_token_instant: &mut first_token_instant,
                    report_perf: p.report_performance,
                    generation_stream,
                    prompt_hidden: None,
                    prompt_hidden_ids: None,
                    // The same flag StreamingCtx carries — the engine's ungated polls and
                    // the streaming reads are idempotent.
                    cancel_flag: Some(cancelled),
                },
                Some(streaming),
            )?;

            last_in_cache = outcome.last_in_cache;
            if outcome.desynced {
                self.flat_mtp_caches_desynced = true;
            }
        } else {
            profiler.set_label("moe_chat_stream_rust");

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
                max_new_tokens: p.max_new_tokens,
                eos_id: eos_id,
                generated_tokens: generated_tokens,
                token_history: token_history,
                finish_reason: finish_reason,
                last_in_cache: last_in_cache,
                first_token_instant: first_token_instant,
                report_perf: p.report_performance,
                generation_stream: generation_stream,
                streaming: {
                    callback: cb,
                    cancelled: cancelled,
                    decode_stream: decode_stream,
                    tokenizer: tokenizer_for_decode,
                    streamed_text_len: streamed_text_len,
                    last_is_reasoning: last_is_reasoning
                }
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

        let text = tokenizer_for_decode
            .decode_sync(&generated_tokens, true)
            .unwrap_or_else(|e| {
                warn!("Failed to decode generated tokens: {}", e);
                String::new()
            });

        // Flush residual bytes
        if text.len() > streamed_text_len {
            let residual = text[streamed_text_len..].to_string();
            // Suppress residual when it is reasoning text and
            // include_reasoning == false.
            if p.include_reasoning || !last_is_reasoning {
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

        let num_tokens = generated_tokens.len() as u32;
        let prompt_token_count = if has_images {
            expanded_tokens.len() as u32
        } else {
            tokens.len() as u32
        };

        let (clean_text, tool_calls, thinking) = engine::parse_thinking_and_tools(
            &text,
            &generated_tokens,
            starts_in_thinking,
            think_end_id,
            think_end_str.as_deref(),
            p.include_reasoning,
        );

        let finish_reason = if tool_calls.iter().any(|tc| tc.status == "ok") {
            "tool_calls".to_string()
        } else {
            finish_reason
        };

        let perf_metrics = compute_performance_metrics(
            generation_start,
            first_token_instant,
            prefill_tokens.len(),
            generated_tokens.len(),
        )
        .map(|mut m| {
            profiler.fill_mtp_acceptance(&mut m);
            m
        });

        // Send final done chunk
        cb.call(
            Ok(ChatStreamChunk {
                text: clean_text,
                done: true,
                finish_reason: Some(finish_reason),
                tool_calls: Some(tool_calls),
                thinking,
                thinking_enabled: Some(
                    crate::engine::resolve_enable_thinking(&config).unwrap_or(true),
                ),
                num_tokens: Some(num_tokens),
                prompt_tokens: Some(prompt_token_count),
                reasoning_tokens: Some(reasoning_tracker.reasoning_token_count()),
                raw_text: Some(engine::raw_text_with_reasoning_suppressed(
                    &text,
                    &generated_tokens,
                    starts_in_thinking,
                    think_end_id,
                    think_end_str.as_deref(),
                    p.include_reasoning,
                )),
                public_raw_text: Some(engine::raw_text_with_reasoning_suppressed(
                    &text,
                    &generated_tokens,
                    starts_in_thinking,
                    think_end_id,
                    think_end_str.as_deref(),
                    false,
                )),
                text_authoritative: Some(true),
                // Start path: report the matched prefix length from
                // `verify_cache_prefix_direct`. Zero on a miss, full
                // cached length on an exact-append hit.
                cached_tokens: Some(cached_prefix_len as u32),
                performance: perf_metrics,
                is_reasoning: None,
            }),
            ThreadsafeFunctionCallMode::NonBlocking,
        );

        Ok(())
    }

    /// Prefill a pre-tokenized delta on top of the existing KV caches and
    /// run the decode loop. Whole-turn core for SYNC delta turns reached
    /// through the engine's `mtp_turn` probe (MTP-enabled sessions;
    /// non-MTP sync deltas run the engine's generic flow or the paged
    /// probe).
    ///
    /// Uses `<|im_end|>` as the eos token (not `config.eos_token_id`) so
    /// the cached history continues to end on a clean ChatML boundary for
    /// the next turn. Cache save runs unconditionally at the end so the
    /// session stays consistent even on error. The engine's delta guards
    /// already enforce the session preconditions; the checks here are
    /// defense-in-depth for the `mtp_turn` caller.
    pub(crate) fn chat_tokens_delta_sync(
        &mut self,
        delta_tokens: Vec<u32>,
        config: ChatConfig,
        thinking: ThinkingSetup,
    ) -> Result<ChatResult> {
        // The delta path is a session-reuse operation by construction.
        if config.reuse_cache == Some(false) {
            return Err(Error::from_reason(
                "chat_tokens_delta_sync requires reuse_cache to be enabled; \
                 the delta path operates on session state by construction",
            ));
        }
        if self.caches.is_none() {
            return Err(Error::from_reason(
                "chat_tokens_delta_sync requires an initialized session (call chatSessionStart first)",
            ));
        }
        if delta_tokens.is_empty() {
            return Err(Error::from_reason(
                "chat_tokens_delta_sync requires a non-empty delta",
            ));
        }
        // Text-only delta on an image-bearing cache is intentional — the KV cache
        // retains the image attention state from the prior prefill. The engine's
        // `session_continue` gate filters real image-set changes with the
        // `IMAGE_CHANGE_REQUIRES_SESSION_RESTART:` prefix so the TS
        // `ChatSession` can route those through `chatSessionStart`.

        let report_perf = config.report_performance.unwrap_or(false);

        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))?
            .clone();

        // Session path: use <|im_end|> as eos, NOT config.eos_token_id.
        let eos_id = tokenizer
            .im_end_id()
            .ok_or_else(|| Error::from_reason("Tokenizer missing <|im_end|> special token"))?;

        let think_end_id = tokenizer.think_end_id();
        let think_end_str = tokenizer.think_end_str().map(|s| s.to_string());

        // Build full token history = cached_history + delta. Used for
        // penalty context AND as the running token history in the decode loop.
        // Snapshot the cached-prefix length before extending so we can
        // report it on the ChatResult for observability — the delta path
        // always reuses the full cached history by construction.
        let cached_prefix_len_for_result = self.cached_token_history.len() as u32;
        let mut full_token_history = self.cached_token_history.clone();
        full_token_history.extend(delta_tokens.iter().copied());

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
        // The delta path drives the paged core with the FULL token
        // history; the adapter's warm-continue path matches the cached
        // prefix automatically.
        if self.paged_adapter.is_some() {
            return self.paged_turn_sync_core(
                full_token_history.clone(),
                tokenizer.clone(),
                eos_id,
                p,
                report_perf,
                thinking,
            );
        }

        // Pure-Rust eager MTP. Delta turns are text-only by construction; paged
        // already early-returned.
        let eager_mtp = p.enable_mtp && self.has_mtp_weights() && self.paged_adapter.is_none();

        let embedding = self.embedding.clone();
        let generation_stream = Stream::new(DeviceType::Gpu);
        let model_size_bytes = self.config.estimate_memory_bytes() as usize;
        let _wired_ctx =
            crate::stream::WiredLimitContext::new(model_size_bytes, vec![generation_stream]);

        // StreamContext created ONCE for entire prefill+decode
        let _stream_ctx = StreamContext::new(generation_stream);

        let fa_idx = self.fa_idx;

        let mut profiler =
            crate::decode_profiler::DecodeProfiler::new("moe_chat_delta", "qwen3_5_moe");
        profiler.set_prompt_tokens(delta_tokens.len() as u32);
        profiler.snapshot_memory_before();

        // Text-only prefill of the delta on top of the existing caches.
        // Usually tiny (a single user turn), but chunked defensively so a
        // user pasting a long follow-up message doesn't blow memory.
        profiler.begin_prefill();
        let turn_cancel = self.turn_cancel.clone();
        let prefill_result = if self.flat_mtp_caches_desynced {
            // A prior eager-MTP turn stopped mid-cycle, leaving self.caches advanced
            // past the emitted history; GDN state cannot be rewound, so discard and
            // re-prefill the full conversation into fresh caches.
            self.caches = Some(fresh_moe_layer_caches(&self.config));
            profiler.set_prompt_tokens(full_token_history.len() as u32);
            let prompt =
                MxArray::from_uint32(&full_token_history, &[1, full_token_history.len() as i64])?;
            let logits = chunked_prefill(
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
            if logits.is_ok() {
                self.flat_mtp_caches_desynced = false;
            }
            logits
        } else {
            let prompt = MxArray::from_uint32(&delta_tokens, &[1, delta_tokens.len() as i64])?;
            chunked_prefill(
                &prompt,
                &embedding,
                &mut self.layers,
                &mut self.caches,
                &self.final_norm,
                &self.lm_head,
                fa_idx,
                generation_stream,
                turn_cancel.as_deref(),
            )
        };
        // A partial delta prefill (cancel or failure) leaves `self.caches`
        // ahead of `cached_token_history` — invalidate rather than let the
        // next delta extend poisoned caches.
        let logits = match prefill_result {
            Ok(logits) => logits,
            Err(e) => {
                self.invalidate_moe_paged_session("delta flat prefill failure");
                return Err(e);
            }
        };
        let prefill_out_seq_len = logits.shape_at(1)?;
        let mut last_logits = logits.slice_axis(1, prefill_out_seq_len - 1, prefill_out_seq_len)?;
        last_logits = last_logits.squeeze(Some(&[1]))?;
        profiler.end_prefill();

        let prompt_tokens_for_result = full_token_history.len() as u32;

        // Save snapshot for save_cache_state_direct (prior history + delta).
        let save_tokens = full_token_history.clone();

        // Decode setup.
        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut finish_reason = String::from("length");

        let mut token_history: Vec<u32> = full_token_history;
        last_logits = apply_all_penalties(last_logits, &token_history, &p)?;
        let mut y = sample(&last_logits, p.sampling_config)?;
        MxArray::async_eval_arrays(&[&y]);

        let mut reasoning_tracker = engine::ReasoningTracker::from_setup(&thinking, think_end_id);

        // Whether the final committed token reached the physical KV/GDN cache;
        // written by the decode driver so the save below drops it when it was
        // never forwarded (unforwarded stop token).
        let mut last_in_cache = true;

        if eager_mtp {
            // Delta-continuation eager MoE MTP — same engine-owned
            // `run_mtp_turn` loop + `MoeMtpStepper` as the fresh-prefill sync
            // site (prompt-prefix seed inert here — see the `MoeMtpStepper` struct doc).
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
                    // Sync turns cancel through the engine loop's ungated polls
                    // (this site has no StreamingCtx).
                    cancel_flag: turn_cancel.as_deref(),
                },
                None,
            )?;

            last_in_cache = outcome.last_in_cache;
            if outcome.desynced {
                self.flat_mtp_caches_desynced = true;
            }
        } else {
            profiler.set_label("moe_chat_delta_rust");

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

        // Save cache state. Delta continuations preserve
        // `cached_image_key` — the live KV cache still encodes the prior
        // prefill's image attention state even though this turn is
        // text-only, and a subsequent cache-prefix verify needs that
        // key to stay in place so a later image-bearing turn correctly
        // flags an image-set change instead of being accepted on the
        // delta path.
        engine::save_cache_state_after_delta(
            p.reuse_cache,
            &generated_tokens,
            &finish_reason,
            /* drop_last_always */ !last_in_cache,
            &save_tokens,
            &mut self.cached_token_history,
            &mut self.cached_image_key,
            &mut self.cached_rope_deltas,
            &mut self.caches,
        );

        let performance = compute_performance_metrics(
            generation_start,
            first_token_instant,
            delta_tokens.len(),
            generated_tokens.len(),
        )
        .map(|mut m| {
            profiler.fill_mtp_acceptance(&mut m);
            m
        });

        let _final_sampled_token = y;

        let mut result = finalize_chat_result(
            &tokenizer,
            &generated_tokens,
            finish_reason,
            think_end_id,
            think_end_str.as_deref(),
            performance,
            p.include_reasoning,
            thinking.enabled,
            prompt_tokens_for_result,
            reasoning_tracker.reasoning_token_count(),
        )?;
        // Delta path always reuses the full cached history — report it.
        result.cached_tokens = cached_prefix_len_for_result;
        Ok(result)
    }

    /// Prefill the delta tokens and run the streaming decode loop.
    ///
    /// Whole-turn core for STREAMING delta turns reached through the
    /// engine's `mtp_turn` probe (MTP-enabled sessions; non-MTP
    /// streaming deltas run the engine's generic flow or the paged
    /// probe). Mirrors [`Self::vision_mtp_whole_turn_stream_core`] but skips the
    /// message rendering + prefix verification stages — the caller owns
    /// cache coherence by construction. Uses `<|im_end|>` as eos so the
    /// cached history continues to end on a clean ChatML boundary after
    /// the reply is saved.
    pub(super) fn chat_stream_tokens_delta_sync_inner(
        &mut self,
        delta_tokens: Vec<u32>,
        config: ChatConfig,
        cb: &StreamSender<'_>,
        cancelled: &AtomicBool,
        thinking: ThinkingSetup,
    ) -> Result<()> {
        let report_perf = config.report_performance.unwrap_or(false);

        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))?
            .clone();

        // Session path: use <|im_end|> as eos, NOT config.eos_token_id.
        let eos_id = tokenizer
            .im_end_id()
            .ok_or_else(|| Error::from_reason("Tokenizer missing <|im_end|> special token"))?;

        let think_end_id = tokenizer.think_end_id();
        let think_end_str = tokenizer.think_end_str().map(|s| s.to_string());
        let tokenizer_for_decode = tokenizer.clone();

        // Build full token history = cached_history + delta.
        // Capture `prior_cached_len` BEFORE the extend — this is the
        // reused-prefix length reported on the terminal ChatStreamChunk's
        // `cached_tokens` field (mirrors the non-streaming delta path's
        // `cached_tokens_for_result`).
        let prior_cached_len = self.cached_token_history.len() as u32;
        let mut full_token_history = self.cached_token_history.clone();
        full_token_history.extend(delta_tokens.iter().copied());

        let mut p = extract_chat_params(&config);
        p.extra_eos_ids = self.gen_defaults.eos_token_ids.clone();

        let generation_start = if report_perf {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut first_token_instant: Option<std::time::Instant> = None;

        // Block-paged dispatch — early-return onto the paged core.
        if self.paged_adapter.is_some() {
            return self.paged_turn_stream_core(
                full_token_history.clone(),
                tokenizer_for_decode,
                eos_id,
                p,
                report_perf,
                cb,
                cancelled,
                thinking,
            );
        }

        // Pure-Rust eager MTP. Delta turns are text-only by construction; paged
        // already early-returned.
        let eager_mtp = p.enable_mtp && self.has_mtp_weights() && self.paged_adapter.is_none();

        let embedding = self.embedding.clone();
        let generation_stream = Stream::new(DeviceType::Gpu);
        let model_size_bytes = self.config.estimate_memory_bytes() as usize;
        let _wired_ctx =
            crate::stream::WiredLimitContext::new(model_size_bytes, vec![generation_stream]);

        let fa_idx = self.fa_idx;

        let mut profiler =
            crate::decode_profiler::DecodeProfiler::new("moe_chat_stream_delta", "qwen3_5_moe");
        profiler.set_prompt_tokens(delta_tokens.len() as u32);
        profiler.snapshot_memory_before();

        // Text-only prefill of the delta on top of the existing caches.
        // Chunked defensively — see the sync sibling for rationale.
        profiler.begin_prefill();
        let turn_cancel = self.turn_cancel.clone();
        let prefill_result = if self.flat_mtp_caches_desynced {
            // A prior eager-MTP turn stopped mid-cycle, leaving self.caches advanced
            // past the emitted history; GDN state cannot be rewound, so discard and
            // re-prefill the full conversation into fresh caches.
            self.caches = Some(fresh_moe_layer_caches(&self.config));
            profiler.set_prompt_tokens(full_token_history.len() as u32);
            let prompt =
                MxArray::from_uint32(&full_token_history, &[1, full_token_history.len() as i64])?;
            let logits = chunked_prefill(
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
            if logits.is_ok() {
                self.flat_mtp_caches_desynced = false;
            }
            logits
        } else {
            let prompt = MxArray::from_uint32(&delta_tokens, &[1, delta_tokens.len() as i64])?;
            chunked_prefill(
                &prompt,
                &embedding,
                &mut self.layers,
                &mut self.caches,
                &self.final_norm,
                &self.lm_head,
                fa_idx,
                generation_stream,
                turn_cancel.as_deref(),
            )
        };
        // A partial delta prefill (cancel or failure) leaves `self.caches`
        // ahead of `cached_token_history` — invalidate rather than let the
        // next delta extend poisoned caches.
        let logits = match prefill_result {
            Ok(logits) => logits,
            Err(e) => {
                self.invalidate_moe_paged_session("delta stream flat prefill failure");
                return Err(e);
            }
        };
        let prefill_out_seq_len = logits.shape_at(1)?;
        let mut last_logits = logits.slice_axis(1, prefill_out_seq_len - 1, prefill_out_seq_len)?;
        last_logits = last_logits.squeeze(Some(&[1]))?;
        profiler.end_prefill();

        // Save snapshot for save_cache_state_direct (prior history + delta).
        let save_tokens = full_token_history.clone();

        // Decode setup
        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut finish_reason = String::from("length");
        let mut decode_stream = tokenizer_for_decode.inner().decode_stream(true);
        let mut streamed_text_len: usize = 0;

        let mut token_history: Vec<u32> = full_token_history;
        last_logits = apply_all_penalties(last_logits, &token_history, &p)?;
        let mut y = sample(&last_logits, p.sampling_config)?;
        MxArray::async_eval_arrays(&[&y]);

        let starts_in_thinking = thinking.enabled;
        let mut last_is_reasoning = starts_in_thinking;
        let mut reasoning_tracker = engine::ReasoningTracker::from_setup(&thinking, think_end_id);

        // Whether the final committed token reached the physical KV/GDN cache;
        // written by the decode driver so the save below drops it when it was
        // never forwarded (unforwarded stop token).
        let mut last_in_cache = true;

        if eager_mtp {
            // Streaming delta-continuation eager MoE MTP — same engine-owned
            // `run_mtp_turn` loop + `MoeMtpStepper` + `StreamingCtx` as the
            // fresh-prefill stream site (prompt-prefix seed inert here — see the
            // `MoeMtpStepper` struct doc).
            let mut rng = rand::rng();
            MxArray::async_eval_arrays(&[&y]);

            let mut emitter = crate::engine::backend::DefaultStreamEmitter;
            let streaming = crate::engine::decode::StreamingCtx {
                callback: cb.0,
                cancelled,
                decode_stream: &mut decode_stream,
                tokenizer: tokenizer_for_decode.inner(),
                streamed_text_len: &mut streamed_text_len,
                last_is_reasoning: &mut last_is_reasoning,
                emitter: &mut emitter,
            };

            let outcome = crate::engine::mtp_turn::run_mtp_turn(
                self,
                &mut rng,
                crate::engine::mtp_turn::MtpTurnArgs {
                    y: y.clone(),
                    depth: p.mtp_depth,
                    params: &p,
                    reasoning_tracker: &mut reasoning_tracker,
                    profiler: &mut profiler,
                    max_new_tokens: p.max_new_tokens,
                    eos_id,
                    generated_tokens: &mut generated_tokens,
                    token_history: &mut token_history,
                    finish_reason: &mut finish_reason,
                    first_token_instant: &mut first_token_instant,
                    report_perf: p.report_performance,
                    generation_stream,
                    prompt_hidden: None,
                    prompt_hidden_ids: None,
                    // The same flag StreamingCtx carries — the engine's ungated polls and
                    // the streaming reads are idempotent.
                    cancel_flag: Some(cancelled),
                },
                Some(streaming),
            )?;

            last_in_cache = outcome.last_in_cache;
            if outcome.desynced {
                self.flat_mtp_caches_desynced = true;
            }
        } else {
            profiler.set_label("moe_chat_stream_delta_rust");

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
                max_new_tokens: p.max_new_tokens,
                eos_id: eos_id,
                generated_tokens: generated_tokens,
                token_history: token_history,
                finish_reason: finish_reason,
                last_in_cache: last_in_cache,
                first_token_instant: first_token_instant,
                report_perf: p.report_performance,
                generation_stream: generation_stream,
                streaming: {
                    callback: cb,
                    cancelled: cancelled,
                    decode_stream: decode_stream,
                    tokenizer: tokenizer_for_decode,
                    streamed_text_len: streamed_text_len,
                    last_is_reasoning: last_is_reasoning
                }
            );
        }

        // Save cache state unconditionally — even on cancellation, the
        // partial generated_tokens must be appended so the session stays
        // consistent for the next turn. Delta stream preserves
        // `cached_image_key` (see the sync sibling's rationale).
        engine::save_cache_state_after_delta(
            p.reuse_cache,
            &generated_tokens,
            &finish_reason,
            /* drop_last_always */ !last_in_cache,
            &save_tokens,
            &mut self.cached_token_history,
            &mut self.cached_image_key,
            &mut self.cached_rope_deltas,
            &mut self.caches,
        );

        // Decode the full reply text and emit the final done chunk.
        let text = tokenizer_for_decode
            .decode_sync(&generated_tokens, true)
            .unwrap_or_else(|e| {
                warn!("Failed to decode generated tokens: {}", e);
                String::new()
            });

        if text.len() > streamed_text_len {
            let residual = text[streamed_text_len..].to_string();
            // Suppress residual when it is reasoning text and
            // include_reasoning == false.
            if p.include_reasoning || !last_is_reasoning {
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

        let num_tokens = generated_tokens.len() as u32;
        let prompt_token_count = delta_tokens.len() as u32;

        let (clean_text, tool_calls, thinking) = engine::parse_thinking_and_tools(
            &text,
            &generated_tokens,
            starts_in_thinking,
            think_end_id,
            think_end_str.as_deref(),
            p.include_reasoning,
        );

        let finish_reason = if tool_calls.iter().any(|tc| tc.status == "ok") {
            "tool_calls".to_string()
        } else {
            finish_reason
        };

        let perf_metrics = compute_performance_metrics(
            generation_start,
            first_token_instant,
            delta_tokens.len(),
            generated_tokens.len(),
        )
        .map(|mut m| {
            profiler.fill_mtp_acceptance(&mut m);
            m
        });

        cb.call(
            Ok(ChatStreamChunk {
                text: clean_text,
                done: true,
                finish_reason: Some(finish_reason),
                tool_calls: Some(tool_calls),
                thinking,
                thinking_enabled: Some(
                    crate::engine::resolve_enable_thinking(&config).unwrap_or(true),
                ),
                num_tokens: Some(num_tokens),
                prompt_tokens: Some(prompt_token_count),
                reasoning_tokens: Some(reasoning_tracker.reasoning_token_count()),
                raw_text: Some(engine::raw_text_with_reasoning_suppressed(
                    &text,
                    &generated_tokens,
                    starts_in_thinking,
                    think_end_id,
                    think_end_str.as_deref(),
                    p.include_reasoning,
                )),
                public_raw_text: Some(engine::raw_text_with_reasoning_suppressed(
                    &text,
                    &generated_tokens,
                    starts_in_thinking,
                    think_end_id,
                    think_end_str.as_deref(),
                    false,
                )),
                text_authoritative: Some(true),
                // Delta path reuses the full prior history by construction
                // — report `prior_cached_len` (captured before the
                // `self.cached_token_history` extend above) as the
                // authoritative cached-prefix length.
                cached_tokens: Some(prior_cached_len),
                performance: perf_metrics,
                is_reasoning: None,
            }),
            ThreadsafeFunctionCallMode::NonBlocking,
        );

        Ok(())
    }

    /// Generate text from prompt tokens (synchronous, runs on model thread).
    pub(crate) fn generate_sync(
        &mut self,
        prompt_tokens: MxArray,
        config: Qwen3_5MoeGenerationConfig,
    ) -> Result<Qwen3_5MoeGenerationResult> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))?
            .clone();

        // Init caches
        self.init_caches_sync()?;

        let embedding = self.embedding.clone();
        let generation_stream = Stream::new(DeviceType::Gpu);
        let fa_idx = self.fa_idx;

        // Prefill. Chunked to bound peak GPU memory for long prompts —
        // see `chunked_prefill` docs. `chunked_prefill` internally manages
        // the StreamContext per chunk so we don't need an outer one here.
        let prompt = prompt_tokens.reshape(&[1, -1])?;
        let logits = chunked_prefill(
            &prompt,
            &embedding,
            &mut self.layers,
            &mut self.caches,
            &self.final_norm,
            &self.lm_head,
            fa_idx,
            generation_stream,
            None,
        )?;

        let seq_len = logits.shape_at(1)?;
        let last_logits = logits.slice_axis(1, seq_len - 1, seq_len)?;
        let last_logits = last_logits.squeeze(Some(&[1]))?;

        // Request value wins; otherwise fall back to the checkpoint's
        // generation_config.json default; otherwise the sampler's builtin.
        // When the request omits temperature, a `do_sample:false` in
        // generation_config.json forces greedy decoding (temperature 0),
        // overriding any gen-config temperature (HuggingFace transformers
        // semantics) — `effective_temperature()` folds that rule in.
        // This raw `generate` surface exposes only the four SamplingConfig
        // fields (no repetition/presence/frequency penalty), so a
        // generation_config repetition_penalty is honored on the ChatSession
        // path but intentionally not here. ChatSession is the full-parity surface.
        let sampling_config = Some(SamplingConfig {
            temperature: config
                .temperature
                .or(self.gen_defaults.effective_temperature()),
            top_k: config.top_k.or(self.gen_defaults.top_k),
            top_p: config.top_p.or(self.gen_defaults.top_p),
            min_p: config.min_p.or(self.gen_defaults.min_p),
        });

        let eos_id = self.config.eos_token_id as u32;
        // Extra stop ids from generation_config.json (e.g. a second EOS).
        // Captured before the loop so its `&mut self` borrows are unaffected.
        let extra_eos_ids = self.gen_defaults.eos_token_ids.clone();
        let is_eos = |t: u32| t == eos_id || extra_eos_ids.contains(&t);
        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut y = sample(&last_logits, sampling_config)?;

        for _step in 0..config.max_new_tokens {
            y.eval();
            let token_id = y.item_at_int32(0)? as u32;
            generated_tokens.push(token_id);

            if is_eos(token_id) {
                break;
            }

            let next_ids = y.reshape(&[1, 1])?;
            let logits = {
                let _stream_ctx = StreamContext::new(generation_stream);
                forward_inner(
                    &next_ids,
                    &embedding,
                    &mut self.layers,
                    &mut self.caches,
                    &self.final_norm,
                    &self.lm_head,
                    fa_idx,
                )?
            };

            let logits = logits.squeeze(Some(&[1]))?;
            y = sample(&logits, sampling_config)?;
            MxArray::async_eval_arrays(&[&y]);

            if (_step + 1) % 256 == 0 {
                crate::array::synchronize_and_clear_cache();
            }
        }

        self.reset_caches_sync()?;

        let finish_reason = if generated_tokens.last().is_some_and(|&t| is_eos(t)) {
            "stop"
        } else {
            "length"
        };

        let text = tokenizer
            .decode_sync(&generated_tokens, true)
            .unwrap_or_default();

        Ok(Qwen3_5MoeGenerationResult {
            tokens: generated_tokens.clone(),
            text,
            num_tokens: generated_tokens.len() as u32,
            finish_reason: finish_reason.to_string(),
        })
    }
}
