//! Save/setters plus the flat (non-paged) whole-turn and streaming cores.

use super::*;

impl Qwen35Inner {
    fn first_quantized_save_component(&self) -> Option<String> {
        if self.embedding.is_quantized() {
            return Some("embedding".to_string());
        }
        for (index, layer) in self.layers.iter().enumerate() {
            if layer.is_quantized() {
                return Some(format!("layers.{index}"));
            }
        }
        if self
            .lm_head
            .as_ref()
            .is_some_and(|head| head.is_quantized())
        {
            return Some("lm_head".to_string());
        }
        if self.mtp_weights_loaded
            && self
                .mtp
                .as_ref()
                .is_some_and(|mtp| mtp.has_quantized_weights())
        {
            return Some("mtp".to_string());
        }
        None
    }

    /// Save model weights and configuration to a directory (synchronous).
    pub(crate) fn save_model_sync(&self, save_path: &str) -> Result<()> {
        use crate::models::qwen3_5::decoder_layer::AttentionType;

        if let Some(component) = self.first_quantized_save_component() {
            return Err(Error::from_reason(format!(
                "Cannot save model: save_model is dense/BF16-only, but main-model component \
                 '{component}' contains quantized projections. Refusing before creating the \
                 destination because packed weights require sidecars and quantization metadata \
                 that this save format cannot preserve losslessly."
            )));
        }

        let mut params = HashMap::new();

        // Embedding
        params.insert("embedding.weight".to_string(), self.embedding.get_weight());

        // Layers
        for (i, layer) in self.layers.iter().enumerate() {
            let prefix = format!("layers.{}", i);
            match &layer.attn {
                AttentionType::Linear(gdn) => {
                    params.insert(
                        format!("{}.linear_attn.in_proj_qkvz.weight", prefix),
                        gdn.get_in_proj_qkvz_weight(),
                    );
                    params.insert(
                        format!("{}.linear_attn.in_proj_ba.weight", prefix),
                        gdn.get_in_proj_ba_weight(),
                    );
                    params.insert(
                        format!("{}.linear_attn.conv1d.weight", prefix),
                        gdn.get_conv1d_weight(),
                    );
                    params.insert(
                        format!("{}.linear_attn.norm.weight", prefix),
                        gdn.get_norm_weight(),
                    );
                    params.insert(
                        format!("{}.linear_attn.out_proj.weight", prefix),
                        gdn.get_out_proj_weight(),
                    );
                    params.insert(format!("{}.linear_attn.dt_bias", prefix), gdn.get_dt_bias());
                    params.insert(format!("{}.linear_attn.a_log", prefix), gdn.get_a_log());
                }
                AttentionType::Full(attn) => {
                    params.insert(
                        format!("{}.self_attn.q_proj.weight", prefix),
                        attn.get_q_proj_weight(),
                    );
                    params.insert(
                        format!("{}.self_attn.k_proj.weight", prefix),
                        attn.get_k_proj_weight(),
                    );
                    params.insert(
                        format!("{}.self_attn.v_proj.weight", prefix),
                        attn.get_v_proj_weight(),
                    );
                    params.insert(
                        format!("{}.self_attn.o_proj.weight", prefix),
                        attn.get_o_proj_weight(),
                    );
                    params.insert(
                        format!("{}.self_attn.q_norm.weight", prefix),
                        attn.get_q_norm_weight(),
                    );
                    params.insert(
                        format!("{}.self_attn.k_norm.weight", prefix),
                        attn.get_k_norm_weight(),
                    );
                }
            }
            params.insert(
                format!("{}.mlp.gate_proj.weight", prefix),
                layer.mlp.get_gate_proj_weight(),
            );
            params.insert(
                format!("{}.mlp.up_proj.weight", prefix),
                layer.mlp.get_up_proj_weight(),
            );
            params.insert(
                format!("{}.mlp.down_proj.weight", prefix),
                layer.mlp.get_down_proj_weight(),
            );
            params.insert(
                format!("{}.input_layernorm.weight", prefix),
                layer.get_input_layernorm_weight(),
            );
            params.insert(
                format!("{}.post_attention_layernorm.weight", prefix),
                layer.get_post_attention_layernorm_weight(),
            );
        }

        // Final norm
        params.insert(
            "final_norm.weight".to_string(),
            self.final_norm.get_weight(),
        );

        // LM head
        if !self.config.tie_word_embeddings
            && let Some(ref lm_head) = self.lm_head
        {
            params.insert("lm_head.weight".to_string(), lm_head.get_weight());
        }

        // Include vision encoder weights
        if let Some(ref vision_enc) = self.vision_encoder {
            let vision_params = vision_enc.get_parameters();
            params.extend(vision_params);
        }

        // Multi-Token Prediction head. `config.n_mtp_layers` round-trips
        // through config.json, so a reloaded checkpoint will reconstruct the
        // MTP module from config and expect the `mtp.*` tensors present —
        // without this block the loader would find them absent, set
        // `mtp_weights_loaded = false`, and silently disable speculative
        // decode (only a `warn!`). The `mtp_weights_loaded` guard is
        // essential: `mtp.is_some()` alone would serialize a random-init
        // module (constructed from config even when no weights were loaded).
        if self.mtp_weights_loaded
            && let Some(ref mtp) = self.mtp
        {
            // The early fail-closed quantized-state gate above guarantees this
            // MTP head is dense; partial omission would make the saved model
            // silently lose speculative decoding.
            params.extend(mtp.get_parameters());
        }

        // Validate for NaN/Inf
        for (name, param) in params.iter() {
            let data = param.to_float32()?;
            let invalid_count = data
                .iter()
                .filter(|v| v.is_nan() || v.is_infinite())
                .count();
            if invalid_count > 0 {
                return Err(napi::Error::new(
                    napi::Status::GenericFailure,
                    format!(
                        "Cannot save model: parameter '{}' contains {} NaN/Inf values.",
                        name, invalid_count
                    ),
                ));
            }
        }

        let mut params_clone: HashMap<String, MxArray> =
            params.iter().map(|(k, v)| (k.clone(), v.clone())).collect();

        // Weights metadata
        let mut weights_metadata = serde_json::Map::new();
        for (key, array) in params.iter() {
            let shape_data = array.shape()?;
            let shape: Vec<i64> = shape_data.as_ref().to_vec();
            let dtype = array.dtype()?;
            let mut param_info = serde_json::Map::new();
            param_info.insert("shape".to_string(), serde_json::json!(shape));
            param_info.insert("dtype".to_string(), serde_json::json!(dtype as i32));
            weights_metadata.insert(key.clone(), serde_json::Value::Object(param_info));
        }

        // Config JSON
        let mut config_value = serde_json::to_value(&self.config).map_err(|e| {
            napi::Error::new(
                napi::Status::GenericFailure,
                format!("Failed to serialize config: {e}"),
            )
        })?;
        if let serde_json::Value::Object(ref mut map) = config_value {
            map.insert("model_type".to_string(), serde_json::json!("qwen3_5"));
            // `parse_config` reads the MTP layer count ONLY from the
            // HF-convention keys `mtp_num_hidden_layers` /
            // `num_nextn_predict_layers`; the serde field name
            // `n_mtp_layers` is ignored on load. Without this, a saved MTP
            // checkpoint reloads with `n_mtp_layers = 0` and its head is
            // silently dropped. Mirrors the MoE saver
            // (`qwen3_5_moe::model::Qwen35MoeInner::save_model_sync`).
            map.insert(
                "mtp_num_hidden_layers".to_string(),
                serde_json::json!(self.config.n_mtp_layers),
            );
        }

        let weights_json = serde_json::json!({
            "version": "1.0",
            "config": config_value,
            "weights": weights_metadata,
            "note": "Full weights are in weights.safetensors"
        });

        let path = std::path::Path::new(save_path);
        std::fs::create_dir_all(path)?;

        info!("Saving model to {}", save_path);

        let config_path = path.join("config.json");
        let config_json = serde_json::to_string_pretty(&config_value)?;
        std::fs::write(&config_path, config_json)?;
        info!("Saved config.json");

        let safetensors_path = path.join("weights.safetensors");
        let metadata = Some(serde_json::json!({
            "format": "mlx-node",
            "version": "1.0"
        }));
        crate::utils::safetensors::save_safetensors(
            &safetensors_path,
            &mut params_clone,
            metadata,
        )?;
        info!("Saved weights.safetensors");

        let weights_str = serde_json::to_string_pretty(&weights_json)?;
        let weights_path = path.join("weights.mlx");
        std::fs::write(&weights_path, weights_str)?;
        info!("Saved weights.mlx metadata");

        Ok(())
    }

    /// Set the tokenizer.
    pub(crate) fn set_tokenizer(&mut self, tokenizer: Arc<Qwen3Tokenizer>) {
        self.tokenizer = Some(tokenizer);
    }

    /// Set the vision encoder.
    ///
    /// Paged VLM checkpoints wire this alongside the image processor and
    /// adapter. Image-bearing turns then route through the dedicated paged
    /// vision prefill/decode cores; incomplete stacks stay backend-validated
    /// so those cores can report the precise missing component.
    ///
    /// For text-only inputs M-RoPE collapses to standard scalar-offset
    /// RoPE — `Qwen3_5Attention::forward` uses `self.rope` whenever
    /// `position_ids` is `None`, which is the case for every text-only
    /// flat call. The paged forward (`Qwen3_5Attention::forward_paged`)
    /// also goes through `self.rope` unconditionally. Both paths share
    /// the same RoPE on text-only inputs, so byte-equal parity holds
    /// on VLM checkpoints provided no images are passed.
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
            if let crate::models::qwen3_5::decoder_layer::AttentionType::Full(ref mut attn) =
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

    /// Core synchronous chat implementation (runs on the model thread).
    ///
    /// Whole-turn core for fresh SYNC turns reached through the engine's
    /// `vision_turn` (image-bearing) and `mtp_turn` (MTP-enabled) probes.
    /// The engine already rendered the prompt (`tokens`) and extracted the
    /// raw image payloads (`images`); everything from the paged dispatch
    /// onward runs the whole-turn pipeline. `eos_token_id` is the
    /// caller-supplied stop-on token id (`<|im_end|>` for ChatML
    /// boundaries) so the cached history ends on a clean delimiter that
    /// subsequent session-delta turns can append to.
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

        let generation_start = if report_perf {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let first_token_instant: Option<std::time::Instant> = None;

        // Paged dispatch with native MTP support; the paged path self-handles
        // MTP via the gate inside `paged_turn_sync_core_inner`.
        if self.paged_adapter.is_some() {
            if has_images {
                // All image turns prefill through the paged-vision core, which
                // runs plain autoregressive decode. MTP weights are ignored
                // here (the core has no draft/verify), so an MTP-enabled
                // session decodes cleanly as AR.
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

        // The flat fallback below is text-only. A dense image turn requires the
        // block-paged backend; reaching here with images means the model was
        // loaded without a paged adapter (use_block_paged_cache=false, non-Metal
        // build, or a sym8 checkpoint).
        if has_images {
            return Err(Error::from_reason(
                "qwen3.5 dense image turns require the block-paged KV backend; the model was \
                 loaded without a paged adapter (use_block_paged_cache=false, non-Metal build, \
                 or sym8 checkpoint)",
            ));
        }

        let embedding = self.embedding.clone();

        // Text-only from here: the `has_images` early-return above is the only
        // image path. These bindings preserve the shared cache-reuse / decode
        // plumbing (`has_images` is always false on this branch).
        let (expanded_tokens, current_image_cache_key) = (tokens.clone(), 0u64);

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
        // and there is literally no delta to prefill. The decode loop still
        // needs a `last_logits` to sample from, so we must run *some*
        // forward pass. The GDN linear-attention layers hold a recurrent state
        // (`conv_state`, `recurrent_state`) that cannot be rewound mid-sequence,
        // so a one-token trim is not an option — only the full-attention layers
        // support `KVCache::trim`, and trimming the hybrid stack miscompiles
        // silently. Full reset + re-prefill is wasteful but always correct, and
        // this is a cold edge case.
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

        let generation_stream = Stream::new(DeviceType::Gpu);
        let model_size_bytes = self.config.estimate_memory_bytes() as usize;
        let _wired_ctx =
            crate::stream::WiredLimitContext::new(model_size_bytes, vec![generation_stream]);

        let mut profiler = crate::decode_profiler::DecodeProfiler::new("chat", "qwen3_5");
        profiler.set_prompt_tokens(prefill_tokens.len() as u32);
        profiler.snapshot_memory_before();

        // Prompt-prefix MTP prefill. When MTP is active for this turn the
        // prefill runs the hidden-emitting `chunked_prefill_with_hidden` so the
        // per-prompt-token hiddens can be committed into the MTP
        // committed-history cache; this raises draft acceptance (especially for
        // long prompts). The VLM / cached-prefix branches keep the cheaper
        // logits-only prefill — they do not feed the dense MTP
        // committed-history path.
        //
        // Cache-reuse turns: when `cached_prefix_len > 0` the prefill
        // only processes the uncached SUFFIX, so the captured hidden
        // tensor would cover the suffix — not the full prompt. The
        // prompt-prefill seed REQUIRES the full prompt's hiddens, so it
        // is skipped on cache-reuse turns; committed-history still runs
        // (it starts empty and builds from decode tokens — correct).
        let want_prompt_hidden = p.enable_mtp && self.has_mtp_weights() && cached_prefix_len == 0;

        profiler.begin_prefill();
        let mut prompt_hidden: Option<MxArray> = None;
        let (last_logits, seq_len) = {
            let prompt = MxArray::from_uint32(&prefill_tokens, &[1, prefill_tokens.len() as i64])?;
            let turn_cancel = self.turn_cancel.clone();
            let prefill_result = if want_prompt_hidden {
                chunked_prefill_with_hidden(
                    &prompt,
                    &embedding,
                    &mut self.layers,
                    &mut self.caches,
                    &self.final_norm,
                    &self.lm_head,
                    generation_stream,
                    Some(prefill_tokens.len()),
                    turn_cancel.as_deref(),
                )
                .map(|(logits, ph)| {
                    prompt_hidden = Some(ph);
                    logits
                })
            } else {
                chunked_prefill(
                    &prompt,
                    &embedding,
                    &mut self.layers,
                    &mut self.caches,
                    &self.final_norm,
                    &self.lm_head,
                    generation_stream,
                    turn_cancel.as_deref(),
                )
            };
            // A partially advanced prefill (cancel or failure) must never be
            // continued: `self.caches` would hold the partial delta while
            // `cached_token_history` still describes the previous turn.
            let last_logits = match prefill_result {
                Ok(logits) => logits,
                Err(e) => {
                    self.invalidate_dense_paged_session("MTP whole-turn flat prefill failure");
                    return Err(e);
                }
            };

            (last_logits, tokens.len() as i64)
        };
        profiler.end_prefill();
        // caches now reflect the prefilled history
        self.flat_mtp_caches_desynced = false;

        let prompt_tokens_for_result = if has_images {
            expanded_tokens.len() as u32
        } else {
            tokens.len() as u32
        };

        let save_expanded_tokens = if has_images {
            Some(expanded_tokens.clone())
        } else {
            None
        };

        self.chat_with_caches_inner(ChatDecodeInputs {
            last_logits,
            seq_len,
            is_delta: false,
            has_images,
            token_history_init: tokens.clone(),
            save_tokens: tokens,
            save_expanded_tokens,
            save_image_cache_key: current_image_cache_key,
            tokenizer,
            think_end_id,
            think_end_str,
            thinking,
            eos_id,
            profiler,
            generation_start,
            first_token_instant,
            prefill_tokens_len: prefill_tokens.len(),
            prompt_tokens_for_result,
            // Fresh prefill: report the matched prefix length.
            cached_tokens_for_result: cached_prefix_len as u32,
            embedding,
            generation_stream,
            params: p,
            // `prompt_hidden` is `Some` iff the hidden-emitting prefill ran,
            // and that prefill always covers the whole prompt.
            prompt_hidden_ids: prompt_hidden.as_ref().map(|_| prefill_tokens.clone()),
            prompt_hidden,
        })
    }

    /// Session-based chat continuation via a pre-tokenized delta.
    ///
    /// Runs a text-only prefill of `delta_tokens` on top of the existing KV
    /// caches and decodes the next reply. This path:
    /// - skips the jinja chat template entirely (caller produces the delta),
    /// - skips prefix verification (caller owns cache coherence by construction),
    /// - uses `<|im_end|>` (from the tokenizer vocab) as its stop token instead
    ///   of `config.eos_token_id`, yielding clean cache boundaries for the next
    ///   turn's delta,
    /// - resolves `enable_thinking` from `config.reasoning_effort` via
    ///   `engine::resolve_enable_thinking`,
    /// - is text-only: errors if the session has images.
    ///
    /// Requires a live session: `self.caches` must have been initialized by a
    /// prior session-start turn. Errors otherwise. (The engine's delta
    /// guards already enforce this; the checks here are defense-in-depth
    /// for the `mtp_turn` caller.)
    pub(crate) fn chat_tokens_delta_sync(
        &mut self,
        delta_tokens: Vec<u32>,
        config: ChatConfig,
        thinking: ThinkingSetup,
    ) -> Result<ChatResult> {
        // The delta path is a session-reuse operation by construction: it
        // prefills on top of the existing caches. `reuse_cache = Some(false)`
        // would make the post-decode `save_cache_state_direct` wipe those
        // caches + `cached_token_history`, making the delta turn both depend
        // on and then destroy the session — confusing and wrong. Reject early
        // so no state is mutated.
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
        // A populated `cached_image_key` means the live KV cache carries
        // attention state for images seen on the preceding prefill. The
        // delta path appends a text delta on top of that — the image
        // context stays intact and the model can keep reasoning about
        // it. We do NOT reject here; the outer `chat_session_continue_*`
        // gate already rejects IMAGE-SET CHANGES (non-empty new images
        // that don't match the cached key) with a prefixed error the TS
        // `ChatSession` can catch and route through `chatSessionStart`.

        let report_perf = config.report_performance.unwrap_or(false);

        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))?
            .clone();

        // Session path: use <|im_end|> as eos, NOT config.eos_token_id.
        // This yields clean cache boundaries.
        let eos_id = tokenizer
            .im_end_id()
            .ok_or_else(|| Error::from_reason("Tokenizer missing <|im_end|> special token"))?;

        let think_end_id = tokenizer.think_end_id();
        let think_end_str = tokenizer.think_end_str().map(|s| s.to_string());

        // Build full token history = cached_history + delta. Used for
        // penalty context AND as the running token history in the decode loop.
        // Also used as the snapshot we hand to `save_cache_state_direct` so
        // the saved `cached_token_history` correctly reflects the appended
        // delta plus the generated tokens.
        let mut full_token_history = self.cached_token_history.clone();
        full_token_history.extend(delta_tokens.iter().copied());

        let mut p = extract_chat_params(&config);
        p.extra_eos_ids = self.gen_defaults.eos_token_ids.clone();

        let generation_start = if report_perf {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let first_token_instant: Option<std::time::Instant> = None;

        // Paged dispatch with native MTP support inside
        // `paged_turn_sync_core_inner`.
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

        let embedding = self.embedding.clone();
        let generation_stream = Stream::new(DeviceType::Gpu);
        let model_size_bytes = self.config.estimate_memory_bytes() as usize;
        let _wired_ctx =
            crate::stream::WiredLimitContext::new(model_size_bytes, vec![generation_stream]);

        let mut profiler = crate::decode_profiler::DecodeProfiler::new("chat_delta", "qwen3_5");
        profiler.set_prompt_tokens(delta_tokens.len() as u32);
        profiler.snapshot_memory_before();

        // Text-only prefill of the delta on top of the existing caches.
        profiler.begin_prefill();
        let turn_cancel = self.turn_cancel.clone();
        let prefill_result = if self.flat_mtp_caches_desynced {
            // A prior eager-MTP turn stopped mid-cycle, leaving self.caches
            // advanced past the emitted history; GDN state cannot be rewound,
            // so discard and re-prefill the full conversation into fresh caches.
            self.caches = Some(fresh_dense_layer_caches(&self.config));
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
                generation_stream,
                turn_cancel.as_deref(),
            )
        };
        // A partial delta prefill (cancel or failure) leaves `self.caches`
        // ahead of `cached_token_history` — invalidate rather than let the
        // next delta extend poisoned caches.
        let last_logits = match prefill_result {
            Ok(logits) => logits,
            Err(e) => {
                self.invalidate_dense_paged_session("delta flat prefill failure");
                return Err(e);
            }
        };
        // Total context length post-prefill = full history length.
        let total_seq_len = full_token_history.len() as i64;
        profiler.end_prefill();

        let prompt_tokens_for_result = full_token_history.len() as u32;

        // For the delta path the caches already reflect the entire prior
        // history. For ChatResult observability we still report the
        // cached-prefix length so clients can see the session delta reused
        // the full history: `prior_cached_len` feeds the reported
        // `cached_tokens`.
        let prior_cached_len = full_token_history.len().saturating_sub(delta_tokens.len());

        // For cache save, pass the full token history (cached + delta) as
        // `save_tokens`; the helper / `save_cache_state_direct` will append
        // the generated tokens.
        let save_tokens = full_token_history.clone();

        self.chat_with_caches_inner(ChatDecodeInputs {
            last_logits,
            seq_len: total_seq_len,
            is_delta: true,
            has_images: false,
            token_history_init: full_token_history,
            save_tokens,
            save_expanded_tokens: None,
            save_image_cache_key: 0,
            tokenizer,
            think_end_id,
            think_end_str,
            thinking,
            eos_id,
            profiler,
            generation_start,
            first_token_instant,
            prefill_tokens_len: delta_tokens.len(),
            prompt_tokens_for_result,
            // Delta path reuses the full prior history by construction.
            cached_tokens_for_result: prior_cached_len as u32,
            embedding,
            generation_stream,
            params: p,
            // Delta path: the prefill runs on top of the live KV caches,
            // so there is no fresh full-prompt hidden to commit into the
            // MTP committed-history cache.
            prompt_hidden: None,
            prompt_hidden_ids: None,
        })
    }

    /// Shared post-prefill pipeline: penalty → sample → decode loop (eager
    /// MTP or AR) → save cache state → finalize result.
    ///
    /// Driven by both `vision_mtp_whole_turn_core` and the text-only session path
    /// (`chat_tokens_delta_sync`). `token_history_init` is the
    /// full pre-decode token sequence (used for penalty context and the decode
    /// loop's running history), and the decode loop mutates it in place.
    ///
    /// The caller is responsible for:
    /// - Creating a `WiredLimitContext` tied to `inputs.generation_stream` for
    ///   the lifetime of this call.
    /// - Running prefill and populating the resulting `last_logits` and
    ///   `seq_len` fields of `ChatDecodeInputs`.
    /// - Pre-starting the profiler (`set_prompt_tokens`, `snapshot_memory_before`,
    ///   `begin_prefill`, `end_prefill`).
    fn chat_with_caches_inner(&mut self, inputs: ChatDecodeInputs) -> Result<ChatResult> {
        let ChatDecodeInputs {
            last_logits,
            seq_len,
            is_delta,
            has_images,
            token_history_init,
            save_tokens,
            save_expanded_tokens,
            save_image_cache_key,
            tokenizer,
            think_end_id,
            think_end_str,
            thinking,
            eos_id,
            mut profiler,
            generation_start,
            mut first_token_instant,
            prefill_tokens_len,
            prompt_tokens_for_result,
            cached_tokens_for_result,
            embedding,
            generation_stream,
            params: p,
            prompt_hidden,
            prompt_hidden_ids,
        } = inputs;

        // Pure-Rust ("eager") dense MTP. Gated on the same per-request /
        // per-checkpoint preconditions (`enable_mtp`, MTP weights present),
        // restricted to the dense FLAT path (no live paged adapter, text-only
        // — the paged adapter has its own MTP gate and VLM routes through the
        // text decode path).
        let eager_mtp =
            p.enable_mtp && self.has_mtp_weights() && self.paged_adapter.is_none() && !has_images;

        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut finish_reason = String::from("length");
        let max_new_tokens = p.max_new_tokens;

        // Decode-entry trace. Snapshot all the inputs that decide the
        // AR-vs-MTP branch so MLX_NODE_LOG=info captures everything
        // needed to reconstruct a turn's control flow.
        {
            let prefill_len = seq_len as i32;
            let max_kv_len_estimate =
                engine::kv_capacity_round_up_saturating(prefill_len, max_new_tokens);
            let has_mtp = self.has_mtp_weights();
            let branch = if eager_mtp {
                "MTP (eager)"
            } else if !p.enable_mtp {
                "AR (enable_mtp=false)"
            } else if !has_mtp {
                "AR (no MTP weights on model)"
            } else {
                "AR"
            };
            info!(
                "Qwen3.5 chat_decode entry: prompt_len={} max_new_tokens={} enable_mtp={} \
                 mtp_depth={} prefill_seq_len={} max_kv_len={} has_mtp_weights={} \
                 is_delta={} has_images={} branch=\"{}\"",
                token_history_init.len(),
                max_new_tokens,
                p.enable_mtp,
                p.mtp_depth,
                prefill_len,
                max_kv_len_estimate,
                has_mtp,
                is_delta,
                has_images,
                branch,
            );
        }

        let last_logits = apply_all_penalties(last_logits, &token_history_init, &p)?;
        let mut y = sample(&last_logits, p.sampling_config)?;
        MxArray::async_eval_arrays(&[&y]);

        // Clone the backend-installed per-turn cancel flag up front —
        // the decode closures below borrow `self` mutably, so the flat AR
        // macro / eager-MTP loop read the clone, not `self.turn_cancel`.
        let turn_cancel = self.turn_cancel.clone();

        let mut token_history: Vec<u32> = token_history_init;

        let mut reasoning_tracker = engine::ReasoningTracker::from_setup(&thinking, think_end_id);

        // Whether the final committed token reached the physical KV/GDN cache.
        // The decode macros write `false` when they stop on an unforwarded
        // token so the save below can drop it from `cached_token_history`.
        let mut last_in_cache = true;

        if eager_mtp {
            // Pure-Rust eager dense MTP — the propose/verify whole-turn loop is
            // engine-owned (`engine::run_mtp_turn`) and drives the
            // `DenseMtpStepper` (`MtpBackend::begin_mtp_decode`). The stepper
            // captures the embedding table + config and runs the prompt-prefix
            // seed before the loop. The `profiler.set_label("mtp_eager")`
            // relabel lives in `DenseMtpStepper::profiler_relabel` (applied
            // once at turn entry by the engine).
            let mut rng = rand::rng();

            // Preserve the eager block's initial `async_eval_arrays(&[&y])`
            // (scheduling hint for the first sampled token) right before the
            // engine takes over.
            MxArray::async_eval_arrays(&[&y]);

            let outcome = crate::engine::mtp_turn::run_mtp_turn(
                self,
                &mut rng,
                crate::engine::mtp_turn::MtpTurnArgs {
                    // Cheap refcounted clone: `run_mtp_turn` consumes `y`, but the
                    // post-block `let _final_sampled_token = y;` discard (shared
                    // with the AR `decode_loop!` arm, which reassigns `y`) still
                    // reads it. The clone is the same lazy handle — byte-identical.
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
                    prompt_hidden,
                    prompt_hidden_ids,
                    // Sync turns cancel through the engine loop's
                    // ungated polls (this site has no StreamingCtx).
                    cancel_flag: turn_cancel.as_deref(),
                },
                // SYNC site: no streaming sink (the streaming flat site wires
                // its own `StreamingCtx` and shares this one loop).
                None,
            )?;

            last_in_cache = outcome.last_in_cache;
            self.flat_mtp_last_rollback_unemitted = outcome.rollback_unemitted;
            // Propagate a mid-cycle stop: self.caches advanced past the emitted
            // history, so force a full re-prefill next turn.
            if outcome.desynced {
                self.flat_mtp_caches_desynced = true;
            }
        } else {
            profiler.set_label("chat_rust");

            MxArray::async_eval_arrays(&[&y]);

            let mut ops = mtp_decode::DecodeOps {
                forward: |ids: &MxArray, emb: &Embedding| -> Result<(MxArray, bool)> {
                    let logits = forward_inner(
                        ids,
                        emb,
                        &mut self.layers,
                        &mut self.caches,
                        &self.final_norm,
                        &self.lm_head,
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
        // `cached_image_key` — the KV cache still holds the prior
        // prefill's image attention state even though this turn was
        // text-only. Prefill paths (re)set the key based on the fresh
        // turn's `has_images`.
        if is_delta {
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
        } else {
            save_cache_state_direct(
                p.reuse_cache,
                has_images,
                &generated_tokens,
                &finish_reason,
                /* drop_last_always */ !last_in_cache,
                &save_tokens,
                save_expanded_tokens.as_deref(),
                save_image_cache_key,
                &mut self.cached_token_history,
                &mut self.cached_image_key,
                &mut self.cached_rope_deltas,
                &mut self.caches,
            );
        }
        // This turn committed flat caches, not the paged adapter. Never carry
        // per-block image identities into a later unrelated paged lifecycle.
        self.cached_paged_image_token_positions.clear();

        let performance = compute_performance_metrics(
            generation_start,
            first_token_instant,
            prefill_tokens_len,
            generated_tokens.len(),
        )
        .map(|mut m| {
            profiler.fill_mtp_acceptance(&mut m);
            m
        });

        // `y` is the last sampled token from the decode loop. The
        // `decode_loop!` macro assigns to `y` each iteration and the final
        // assignment in the last iteration is never observed, which without
        // this explicit discard trips `clippy::unused_assignments` (the
        // macro repetition hides the usage pattern from the lint). Binding
        // here is cleaner than spraying `#[allow]` inside the macro body.
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
        // Report the length of the reused cached prefix for observability.
        // Driven by the caller-supplied `cached_tokens_for_result`:
        //   - fresh prefill: equals `cached_prefix_len` (0 miss / full hit)
        //   - session delta: equals the prior-history length (full reuse)
        // See the invariant doc on `verify_cache_prefix_direct`.
        result.cached_tokens = cached_tokens_for_result;
        Ok(result)
    }

    /// Shared pure-Rust eager-MTP decode loop for the dense FLAT STREAMING
    /// cores (`chat_stream_sync_inner` / `chat_stream_tokens_delta_sync_inner`).
    ///
    /// This is the streaming analogue of the `eager_mtp` arm of
    /// [`Self::chat_with_caches_inner`]: the propose/verify whole-turn loop is
    /// engine-owned ([`crate::engine::mtp_turn::run_mtp_turn`]) and drives the
    /// [`DenseMtpStepper`] ([`MtpBackend::begin_mtp_decode`]) — the SAME stepper
    /// and prompt-prefix committed-history seed the SYNC site uses. The only
    /// difference is the streaming sink: this site wires a
    /// [`crate::engine::decode::StreamingCtx`] (incremental detokenization plus
    /// the default [`crate::engine::backend::DefaultStreamEmitter`]) so accepted
    /// tokens stream out the `cb` sink incrementally, sharing ONE loop with the
    /// sync site. Caller owns prefill, sampling of the first `y`, the
    /// `WiredLimitContext`, and the post-loop save-cache / final-chunk tail.
    ///
    /// Preconditions (enforced by the callers' gate): `enable_mtp`,
    /// `has_mtp_weights()`, `paged_adapter.is_none()`, text-only. The body is
    /// byte-identical to the non-streaming eager MTP decode (same accept/rewind
    /// math, same GDN tape replay) — only the streamed deltas differ.
    #[allow(clippy::too_many_arguments)]
    fn run_flat_stream_eager_mtp<'a>(
        &mut self,
        y: MxArray,
        token_history: &mut Vec<u32>,
        generated_tokens: &mut Vec<u32>,
        finish_reason: &mut String,
        reasoning_tracker: &mut engine::ReasoningTracker,
        profiler: &mut crate::decode_profiler::DecodeProfiler,
        first_token_instant: &mut Option<std::time::Instant>,
        streamed_text_len: &mut usize,
        last_is_reasoning: &mut bool,
        decode_stream: &mut tokenizers::DecodeStream<
            'a,
            tokenizers::ModelWrapper,
            tokenizers::NormalizerWrapper,
            tokenizers::PreTokenizerWrapper,
            tokenizers::PostProcessorWrapper,
            tokenizers::DecoderWrapper,
        >,
        tokenizer: &'a Arc<Qwen3Tokenizer>,
        cb: &StreamSender<'_>,
        cancelled: &AtomicBool,
        p: &engine::ChatParams,
        eos_id: u32,
        max_new_tokens: i32,
        generation_stream: Stream,
        prompt_hidden: Option<MxArray>,
        prompt_hidden_ids: Option<Vec<u32>>,
        last_in_cache: &mut bool,
    ) -> Result<()> {
        MxArray::async_eval_arrays(&[&y]);

        let mut rng = rand::rng();

        // Wire the streaming sink: incremental detokenization through the
        // shared `step_decode_stream` + the default ChatML emitter (qwen3_5
        // does not override `stream_emitter`, so the macro's inline emit is
        // byte-identical to `DefaultStreamEmitter::on_token_text`). The
        // engine's `run_mtp_turn` routes the SAME three emit sites + the
        // pre-loop cancel break through this `StreamingCtx`.
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
                y,
                depth: p.mtp_depth,
                params: p,
                reasoning_tracker,
                profiler,
                max_new_tokens,
                eos_id,
                generated_tokens,
                token_history,
                finish_reason,
                first_token_instant,
                report_perf: p.report_performance,
                generation_stream,
                prompt_hidden,
                prompt_hidden_ids,
                // The same flag StreamingCtx carries — the engine's
                // ungated polls and the streaming reads are idempotent.
                cancel_flag: Some(cancelled),
            },
            Some(streaming),
        )?;

        *last_in_cache = outcome.last_in_cache;
        self.flat_mtp_last_rollback_unemitted = outcome.rollback_unemitted;
        // Propagate a mid-cycle stop: self.caches advanced past the emitted
        // history, so force a full re-prefill next turn.
        if outcome.desynced {
            self.flat_mtp_caches_desynced = true;
        }

        Ok(())
    }

    /// Turn-admission speculative reservation for the paged MTP arm: reserve
    /// the lookahead rows past the (post-prefill) prompt cursor, sized by
    /// `engine::mtp_turn::turn_lookahead_rows` off the model's
    /// [`SpeculativePlan`] (I1 — the plan property is the single source).
    /// This guarantees the FIRST cycle; the engine loop re-reserves the same
    /// margin before every later cycle through
    /// [`MtpStepper::reserve_cycle_lookahead`], where exhaustion degrades
    /// that cycle to a Step-A AR token instead of gating the whole turn.
    ///
    /// `Ok(false)` means the pool cannot hold even one verify cycle: the
    /// caller must run the turn autoregressively instead of erroring it
    /// (vLLM schedules such a request without its spec tokens rather than
    /// failing it). AR decode grows one row at a time against the same
    /// lazily-allocated tail, so it can still make progress where a
    /// `depth + 1`-row verify write could not. Non-capacity errors (missing
    /// adapter, poisoned allocator) still fail the turn.
    pub(super) fn reserve_paged_mtp_lookahead(
        &mut self,
        p: &engine::ChatParams,
        site: &str,
    ) -> Result<bool> {
        let Some(plan) = self.execution_plan().speculative else {
            // The MTP arms gate on `has_mtp_weights()`, which is exactly what
            // publishes the speculative plan — unreachable, but proceeding
            // without a reservation only restores lazy allocation.
            debug_assert!(
                false,
                "{site}: paged MTP admission without a speculative plan"
            );
            return Ok(true);
        };
        let rows = u32::try_from(crate::engine::mtp_turn::turn_lookahead_rows(plan, p))
            .unwrap_or(u32::MAX);
        let adapter = self
            .paged_adapter
            .as_mut()
            .ok_or_else(|| Error::from_reason(format!("{site}: paged_adapter is None")))?;
        match adapter.reserve_rows(rows) {
            Ok(_) => Ok(true),
            Err(e) if e.starts_with("context_length_exceeded:") => {
                tracing::warn!(
                    target: "mlx_core::qwen3_5::paged",
                    lookahead_rows = rows,
                    "{site}: speculative lookahead reservation exhausted the \
                     paged pool; falling back to autoregressive decode: {e}"
                );
                Ok(false)
            }
            Err(e) => Err(Error::from_reason(format!(
                "{site}: speculative lookahead reservation: {e}"
            ))),
        }
    }

    /// Prefill the delta tokens and run the streaming decode loop.
    ///
    /// Whole-turn core for STREAMING delta turns reached through the
    /// engine's `mtp_turn` probe (MTP-enabled sessions; non-MTP
    /// streaming deltas run the engine's generic flow or the paged
    /// probe). Mirrors [`Self::chat_stream_sync_inner`] but skips the
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
        let _reuse_cache = config.reuse_cache.unwrap_or(true);
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

        // MTP-on-paged delta streams fall through to the dense (flat)
        // streaming path; non-MTP paged streams take the paged
        // streaming core.
        let mtp_takes_dense_path =
            p.enable_mtp && self.has_mtp_weights() && self.paged_adapter.is_some();
        if mtp_takes_dense_path
            && let Some(ref mut adapter) = self.paged_adapter
            && let Err(e) = adapter.release_request()
        {
            tracing::warn!(
                target: "mlx_core::qwen3_5::paged",
                "MTP-on-paged dispatch (stream-delta): release_request failed (ignored): {e}",
            );
        }
        if self.paged_adapter.is_some() && !mtp_takes_dense_path {
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

        let embedding = self.embedding.clone();
        let generation_stream = Stream::new(DeviceType::Gpu);
        let model_size_bytes = self.config.estimate_memory_bytes() as usize;
        let _wired_ctx =
            crate::stream::WiredLimitContext::new(model_size_bytes, vec![generation_stream]);

        let mut profiler =
            crate::decode_profiler::DecodeProfiler::new("chat_stream_delta", "qwen3_5");
        // Prefill token count is reported below per-branch: the paged→dense
        // fallback re-prefills the FULL history (see `rebuild_full_flat_prefill`),
        // every other delta turn prefills only the delta.
        profiler.snapshot_memory_before();

        // Paged→dense-MTP cache-source transition.
        //
        // When `mtp_takes_dense_path` is true we arrived here off a PAGED
        // session (`self.paged_adapter.is_some()`) that fell through the
        // paged streaming core because it carries no MTP gate yet. On the
        // paged path the authoritative FULL-ATTENTION K/V lives in the
        // paged adapter's `LayerKVPool`, NOT in the flat `self.caches`
        // (which only ever received GDN linear conv/recurrent state). A
        // prior NON-streaming paged turn (`send()` → `chat_tokens_delta_sync`
        // → `paged_turn_sync_core`) therefore leaves `self.caches`'
        // full-attention slots EMPTY/STALE for the prior turn's tokens.
        // Delta-prefilling only `delta_tokens` on top of that and running
        // the eager MTP decode against `self.caches` would decode from an
        // incomplete flat prefix (missing the prior turn's attention KV).
        //
        // Fix: when taking this dense fallback AND the flat caches are dirty
        // (a paged-core turn ran since the last flat prefill —
        // `paged_full_attn_caches_dirty`), rebuild the flat caches from
        // scratch over the FULL token history: reset to fresh caches and
        // prefill `full_token_history` instead of just the delta. This
        // mirrors how the dense session-start path recovers from a
        // cache-prefix miss (full reset + full re-prefill). The GDN recurrent
        // state cannot be rewound mid-sequence, so a full reset + full
        // prefill is the only coherent way to seed both the GDN linear and
        // full-attention flat caches for the eager MTP decode.
        //
        // The dirty gate keeps this a ONE-TIME cost on the paged→dense
        // transition: the flag is cleared once at end-of-turn success (atomic
        // with the `cached_token_history` commit), so subsequent streaming MTP
        // turns delta-prefill on the now-authoritative flat caches (no O(n²)
        // full re-prefill every turn). It re-arms only if a later paged-core
        // turn runs again. The common non-paged dense
        // session (`paged_adapter.is_none()`, `mtp_takes_dense_path == false`,
        // flag never set) is untouched: it keeps the delta-on-existing-caches
        // prefill below, byte-identical.
        let rebuild_full_flat_prefill = (mtp_takes_dense_path && self.paged_full_attn_caches_dirty)
            || self.flat_mtp_caches_desynced;
        profiler.set_prompt_tokens(if rebuild_full_flat_prefill {
            full_token_history.len() as u32
        } else {
            delta_tokens.len() as u32
        });
        profiler.begin_prefill();
        let turn_cancel = self.turn_cancel.clone();
        let prefill_result = if rebuild_full_flat_prefill {
            // Discard the paged-session flat caches (full-attn slots are
            // stale, GDN state belongs to the released paged request) and
            // re-prefill the entire conversation into fresh flat caches.
            self.flat_full_reprefill_count += 1;
            self.caches = Some(fresh_dense_layer_caches(&self.config));
            let prompt =
                MxArray::from_uint32(&full_token_history, &[1, full_token_history.len() as i64])?;
            chunked_prefill(
                &prompt,
                &embedding,
                &mut self.layers,
                &mut self.caches,
                &self.final_norm,
                &self.lm_head,
                generation_stream,
                turn_cancel.as_deref(),
            )
        } else {
            // Text-only prefill of the delta on top of the existing caches.
            let prompt = MxArray::from_uint32(&delta_tokens, &[1, delta_tokens.len() as i64])?;
            chunked_prefill(
                &prompt,
                &embedding,
                &mut self.layers,
                &mut self.caches,
                &self.final_norm,
                &self.lm_head,
                generation_stream,
                turn_cancel.as_deref(),
            )
        };
        // A partial prefill (cancel or failure) leaves `self.caches` ahead of
        // `cached_token_history` — invalidate so the session goes cold instead
        // of extending poisoned caches. Full invalidation supersedes the
        // dirty-gate mitigation described below for this exit.
        let mut last_logits = match prefill_result {
            Ok(logits) => logits,
            Err(e) => {
                self.invalidate_dense_paged_session("delta stream flat prefill failure");
                return Err(e);
            }
        };
        // caches now reflect the prefilled history
        self.flat_mtp_caches_desynced = false;
        // The flat full-attention caches now cover the full history (rebuild
        // branch) or were already authoritative (delta branch). We do NOT
        // clear `paged_full_attn_caches_dirty` here: the clear is co-located
        // with the `cached_token_history` commit at the end-of-turn success
        // boundary (`save_cache_state_after_delta` below), so that ANY
        // mid-turn error — prefill OR decode — leaves the flag dirty. That
        // way the next paged→dense turn still performs the protective one-time
        // full rebuild over the authoritative history rather than trusting
        // half-advanced flat caches paired with a stale `cached_token_history`.
        let _seq_len = full_token_history.len() as i64;
        profiler.end_prefill();

        // Save snapshot for save_cache_state_direct (prior history + delta).
        let save_tokens = full_token_history.clone();

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

        // Pure-Rust ("eager") dense MTP gate for the FLAT streaming delta path.
        // Delta is text-only (no `has_images`) by construction; paged sessions
        // returned earlier. Continuations have a live cache prefix so the
        // committed-history builds from decode tokens with NO prompt seed
        // (mirrors the non-stream delta path's `None, None, 0`).
        let eager_mtp = p.enable_mtp && self.has_mtp_weights() && self.paged_adapter.is_none();

        // Whether the final committed token reached the physical KV/GDN cache;
        // written by the decode driver so the save below drops it when it was
        // never forwarded (unforwarded stop token).
        let mut last_in_cache = true;

        if eager_mtp {
            self.run_flat_stream_eager_mtp(
                y,
                &mut token_history,
                &mut generated_tokens,
                &mut finish_reason,
                &mut reasoning_tracker,
                &mut profiler,
                &mut first_token_instant,
                &mut streamed_text_len,
                &mut last_is_reasoning,
                &mut decode_stream,
                &tokenizer_for_decode,
                cb,
                cancelled,
                &p,
                eos_id,
                p.max_new_tokens,
                generation_stream,
                None,
                None,
                &mut last_in_cache,
            )?;
        } else {
            profiler.set_label("chat_stream_delta_rust");

            let mut ops = mtp_decode::DecodeOps {
                forward: |ids: &MxArray, emb: &Embedding| -> Result<(MxArray, bool)> {
                    let logits = forward_inner(
                        ids,
                        emb,
                        &mut self.layers,
                        &mut self.caches,
                        &self.final_norm,
                        &self.lm_head,
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
        // consistent for the next turn. Delta continuations preserve
        // `cached_image_key` so the next turn's cache-prefix verify
        // still sees the prior prefill's image state.
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
        // Clear the paged-dirty flag ATOMICALLY with the
        // `cached_token_history` commit above: a successful turn updates the
        // committed history and the flat caches together, so the one-time
        // protective rebuild is no longer needed until a later paged-core turn
        // re-dirties it. Placing the clear here (not after prefill) guarantees
        // any mid-turn `?`-error leaves the flag dirty.
        self.paged_full_attn_caches_dirty = false;

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

    /// Whole-turn core for fresh STREAMING turns reached through the
    /// engine's `vision_turn` (image-bearing) and `mtp_turn`
    /// (MTP-enabled) probes. The engine already rendered the prompt
    /// (`tokens`) and extracted the raw image payloads (`images`);
    /// everything from the MTP-on-paged dispatch onward runs the
    /// whole-turn pipeline.
    pub(super) fn chat_stream_sync_inner(
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

        // All image turns route to the paged-vision streaming core, which runs
        // plain autoregressive decode regardless of MTP (the core has no
        // draft/verify; MTP weights are ignored). This precedes the text-only
        // MTP-on-paged gate below so an image+MTP stream still reaches the
        // paged-vision core rather than the text dense fallback.
        if has_images && self.paged_adapter.is_some() {
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

        // Text-only paged dispatch. MTP-on-paged streams fall through to the
        // dense (flat) streaming path; non-MTP paged streams take the paged
        // streaming core.
        let mtp_takes_dense_path =
            p.enable_mtp && self.has_mtp_weights() && self.paged_adapter.is_some();
        if mtp_takes_dense_path
            && let Some(ref mut adapter) = self.paged_adapter
            && let Err(e) = adapter.release_request()
        {
            tracing::warn!(
                target: "mlx_core::qwen3_5::paged",
                "MTP-on-paged dispatch (stream-start): release_request failed (ignored): {e}",
            );
        }
        if self.paged_adapter.is_some() && !mtp_takes_dense_path {
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

        // The dense (flat) streaming fallback is text-only. A dense image turn
        // requires the block-paged backend; reaching here with images means the
        // model was loaded without a paged adapter (use_block_paged_cache=false,
        // non-Metal build, or a sym8 checkpoint).
        if has_images {
            return Err(Error::from_reason(
                "qwen3.5 dense image turns require the block-paged KV backend; the model was \
                 loaded without a paged adapter (use_block_paged_cache=false, non-Metal build, \
                 or sym8 checkpoint)",
            ));
        }

        let embedding = self.embedding.clone();

        // Text-only from here: the `has_images` early-return above is the only
        // image path. These bindings preserve the shared cache-reuse / decode
        // plumbing (`has_images` is always false on this branch).
        let (expanded_tokens, current_image_cache_key) = (tokens.clone(), 0u64);

        // Cache reuse
        let cached_prefix_len = verify_cache_prefix_direct(
            reuse_cache,
            has_images,
            &tokens,
            &expanded_tokens,
            current_image_cache_key,
            &self.cached_token_history,
            &self.cached_image_key,
            self.caches.is_some(),
        );

        // Same paged→dense-MTP stale-flat-cache hazard
        // as the streaming delta path, but for the stream-START dense
        // fallback. A prior paged-core turn wrote full-attention K/V into the
        // paged adapter pool, leaving the flat `self.caches` full-attention
        // slots empty/stale (only GDN linear state was imported). A prefix
        // hit from `verify_cache_prefix_direct` (matched against
        // `cached_token_history`) would then decode from an incomplete flat
        // prefix. When the flat caches are dirty, drop any prefix reuse so the
        // branch below does a full reset + re-prefill (cached_prefix_len = 0),
        // rebuilding the flat full-attention caches over the whole prompt.
        //
        // Reachability note: the flag is set ONLY by the two paged cores, and
        // a non-MTP paged turn returns earlier at the
        // `self.paged_adapter.is_some() && !mtp_takes_dense_path` branch above.
        // So whenever control reaches here the flag can be true only on the
        // paged+MTP (`mtp_takes_dense_path`) fallback; a non-paged dense start
        // (`paged_adapter.is_none()`) never sets it → this is a no-op and the
        // common path stays byte-identical. The ungated read is therefore
        // equivalent to gating on `mtp_takes_dense_path`.
        //
        // The flag is cleared at the END-OF-TURN success boundary, co-located
        // with the `cached_token_history` commit (`save_cache_state_direct`
        // below), NOT here and NOT right after prefill. This makes the clear
        // atomic with the history commit: ANY mid-turn `?`-error — prefill OR
        // decode — aborts the turn with the flat caches still un-rebuilt and
        // `cached_token_history` still holding the prior paged turn's tokens,
        // so the flag stays dirty and the NEXT paged→dense turn performs the
        // protective one-time full rebuild instead of decoding from an
        // incomplete flat prefix. Mirrors the reviewed delta path.
        let cached_prefix_len =
            if self.paged_full_attn_caches_dirty || self.flat_mtp_caches_desynced {
                0
            } else {
                cached_prefix_len
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
        // token is not possible, so the only safe response to an exact-
        // match prompt is a full reset + re-prefill.
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

        let mut profiler = crate::decode_profiler::DecodeProfiler::new("chat_stream", "qwen3_5");
        profiler.set_prompt_tokens(prefill_tokens.len() as u32);
        profiler.snapshot_memory_before();

        // Pure-Rust ("eager") dense MTP gate for the FLAT streaming path.
        // Same preconditions as `chat_with_caches_inner`: per-request /
        // per-checkpoint enablement, no live paged adapter (paged streams
        // returned earlier), text-only.
        let eager_mtp =
            p.enable_mtp && self.has_mtp_weights() && self.paged_adapter.is_none() && !has_images;
        // The committed-history v2 seed needs the prompt tail's hiddens, which
        // only the hidden-emitting prefill produces. Skip on cache-reuse turns
        // (the captured hidden would cover the SUFFIX, not the full prompt) —
        // committed-history still runs (it builds from decode tokens).
        let want_prompt_hidden = eager_mtp && cached_prefix_len == 0;
        let mut prompt_hidden: Option<MxArray> = None;

        // Text prefill
        profiler.begin_prefill();
        let (mut last_logits, _seq_len) = {
            let prompt = MxArray::from_uint32(&prefill_tokens, &[1, prefill_tokens.len() as i64])?;
            let turn_cancel = self.turn_cancel.clone();
            let prefill_result = if want_prompt_hidden {
                chunked_prefill_with_hidden(
                    &prompt,
                    &embedding,
                    &mut self.layers,
                    &mut self.caches,
                    &self.final_norm,
                    &self.lm_head,
                    generation_stream,
                    Some(prefill_tokens.len()),
                    turn_cancel.as_deref(),
                )
                .map(|(logits, ph)| {
                    prompt_hidden = Some(ph);
                    logits
                })
            } else {
                chunked_prefill(
                    &prompt,
                    &embedding,
                    &mut self.layers,
                    &mut self.caches,
                    &self.final_norm,
                    &self.lm_head,
                    generation_stream,
                    turn_cancel.as_deref(),
                )
            };
            // A partially advanced prefill (cancel or failure) must never be
            // continued: `self.caches` would hold the partial delta while
            // `cached_token_history` still describes the previous turn. Full
            // invalidation supersedes the dirty-gate mitigation described
            // below — the session goes cold, so the next turn prefills from
            // scratch rather than rebuilding.
            let last_logits = match prefill_result {
                Ok(logits) => logits,
                Err(e) => {
                    self.invalidate_dense_paged_session("MTP stream flat prefill failure");
                    return Err(e);
                }
            };

            (last_logits, tokens.len() as i64)
        };
        profiler.end_prefill();
        // caches now reflect the prefilled history
        self.flat_mtp_caches_desynced = false;

        // On a paged→dense-MTP transition the dirty gate
        // forced `cached_prefix_len = 0`, so the prefill above was a full reset
        // + full re-prefill and the flat full-attention caches now cover the
        // entire prompt. We do NOT clear `paged_full_attn_caches_dirty` here:
        // the clear is co-located with the `cached_token_history` commit at the
        // end-of-turn success boundary (`save_cache_state_direct` below), so it
        // is atomic with the history write. That way ANY mid-turn `?`-error —
        // prefill OR decode — leaves the flag dirty and the next paged→dense
        // turn still performs the protective one-time full rebuild rather than
        // trusting half-advanced flat caches against a stale committed history.
        // (No-op on the common non-paged path, where the flag is never set.)

        let mut token_history: Vec<u32> = tokens.clone();
        last_logits = apply_all_penalties(last_logits, &token_history, &p)?;
        let mut y = sample(&last_logits, p.sampling_config)?;
        MxArray::async_eval_arrays(&[&y]);

        let starts_in_thinking = thinking.enabled;
        let mut last_is_reasoning = starts_in_thinking;
        let mut reasoning_tracker = engine::ReasoningTracker::from_setup(&thinking, think_end_id);

        // The hidden-emitting prefill always covers the whole prompt, so the
        // captured hidden pairs with every prefill token.
        let prompt_hidden_ids: Option<Vec<u32>> =
            prompt_hidden.as_ref().map(|_| prefill_tokens.clone());

        // Whether the final committed token reached the physical KV/GDN cache;
        // written by the decode driver so the save below drops it when it was
        // never forwarded (unforwarded stop token).
        let mut last_in_cache = true;

        if eager_mtp {
            self.run_flat_stream_eager_mtp(
                y,
                &mut token_history,
                &mut generated_tokens,
                &mut finish_reason,
                &mut reasoning_tracker,
                &mut profiler,
                &mut first_token_instant,
                &mut streamed_text_len,
                &mut last_is_reasoning,
                &mut decode_stream,
                &tokenizer_for_decode,
                cb,
                cancelled,
                &p,
                eos_id,
                p.max_new_tokens,
                generation_stream,
                prompt_hidden,
                prompt_hidden_ids,
                &mut last_in_cache,
            )?;
        } else {
            profiler.set_label("chat_stream_rust");

            let mut ops = mtp_decode::DecodeOps {
                forward: |ids: &MxArray, emb: &Embedding| -> Result<(MxArray, bool)> {
                    let logits = forward_inner(
                        ids,
                        emb,
                        &mut self.layers,
                        &mut self.caches,
                        &self.final_norm,
                        &self.lm_head,
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
        // Clear the paged-dirty flag ATOMICALLY with the
        // `cached_token_history` commit above: a successful turn updates the
        // committed history and the rebuilt flat caches together, so the
        // one-time protective rebuild is no longer needed until a later
        // paged-core turn re-dirties it. Placing the clear here (not after
        // prefill) guarantees any mid-turn `?`-error — prefill OR decode —
        // leaves the flag dirty so the next paged→dense turn rebuilds.
        self.paged_full_attn_caches_dirty = false;

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
}
