//! Chat parameter resolution and `ChatBackend` for `Qwen35Inner`.

use super::*;

pub(super) fn resolve_qwen35_chat_params(
    config: &ChatConfig,
    defaults: &crate::engine::ModelGenerationDefaults,
    dflash_block_size: Option<usize>,
) -> crate::engine::params::ChatParams {
    // Mirror ChatBackend's default resolution before applying the
    // algorithm-specific DFlash knobs. Overriding `resolve_params` must not
    // discard generation_config.json defaults for ordinary Qwen turns.
    let mut merged = config.clone();
    crate::engine::apply_generation_defaults(&mut merged, defaults);
    let mut params = crate::engine::extract_chat_params(&merged);
    if let Some(block_size) = dflash_block_size {
        params.mtp_depth = config
            .mtp_depth
            .map(|depth| (depth.max(1) as usize).min(block_size))
            .unwrap_or(block_size);
        params.mtp_adaptive_depth = config.mtp_adaptive_depth.unwrap_or(false);
    }
    params
}

impl ChatBackend for Qwen35Inner {
    fn tokenizer(&self) -> Result<Arc<Qwen3Tokenizer>> {
        self.tokenizer
            .clone()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))
    }

    fn family_name(&self) -> &'static str {
        "qwen3_5"
    }

    fn set_turn_cancel_flag(&mut self, flag: Option<Arc<AtomicBool>>) {
        self.turn_cancel = flag;
    }

    fn set_cache_owner_id(&mut self, owner_id: &str, root_owner_id: Option<&str>) {
        self.active_cache_owner_id.clear();
        self.active_cache_owner_id.push_str(owner_id);
        if let Some(root_owner_id) = root_owner_id {
            self.gdn_root_cache_owner_id = Some(root_owner_id.to_owned());
            self.gdn_root_cache_owner_is_explicit = true;
        } else {
            self.gdn_root_cache_owner_id = Some(owner_id.to_owned());
            self.gdn_root_cache_owner_is_explicit = false;
        }
    }

    fn session_eos_id(&self, tok: &Qwen3Tokenizer) -> Result<u32> {
        tok.im_end_id()
            .ok_or_else(|| Error::from_reason("Tokenizer missing <|im_end|> special token"))
    }

    fn generation_defaults(&self) -> Option<&crate::engine::ModelGenerationDefaults> {
        Some(&self.gen_defaults)
    }

    fn resolve_params(&self, config: &ChatConfig) -> crate::engine::params::ChatParams {
        resolve_qwen35_chat_params(
            config,
            &self.gen_defaults,
            self.dflash2.as_ref().map(|draft| draft.config.block_size),
        )
    }

    fn extra_eos_ids(&self) -> Vec<u32> {
        self.gen_defaults.eos_token_ids.clone()
    }

    // thinking: engine default `policy()` == `ThinkingPolicy::TemplateHonoring`
    // → `thinking_setup` resolves to
    // `{enabled: resolve_enable_thinking(config).unwrap_or(true),
    //   budget: config.thinking_token_budget}`.

    fn cached_token_history(&self) -> &[u32] {
        &self.cached_token_history
    }

    fn reset_caches(&mut self, scope: ResetScope) -> Result<()> {
        self.dflash2_context = None;
        self.dflash2_turn_state = None;
        match scope {
            // Prefix-miss reset: reset each live layer cache, then install a
            // fresh hybrid cache vec. PRESERVES `cached_token_history` /
            // `cached_image_key` / `cached_rope_deltas` (the end-of-turn
            // save overwrites them) and the GDN checkpoints (paged-path
            // state the flat reset never touches).
            ResetScope::PrefixMiss => {
                if let Some(ref mut caches) = self.caches {
                    for cache in caches.iter_mut() {
                        cache.reset();
                    }
                }
                self.caches = Some(fresh_dense_layer_caches(&self.config));
                Ok(())
            }
            // Full clear including history, image key, rope deltas, GDN
            // checkpoints, via `reset_caches_sync`.
            ResetScope::Command => {
                self.reset_caches_sync()?;
                // The EXPLICIT command reset must restore a fully cold
                // state. `reset_caches_sync` clears the flat caches +
                // reuse/GDN state but leaves the paged request's FULL
                // blocks content-addressed in the per-instance
                // BlockAllocator's prefix cache, so a reset-then-rerun of
                // the same prompt would take the prefix-hit suffix-prefill
                // path (`verify_cache_prefix_direct` > 0) — a different
                // bf16 reduction order than the cold full prefill, enough
                // to flip a greedy near-tie (observed on the lfm2 sibling:
                // "says," vs "said" at token ~6; qwen3.5 shares the
                // identical adapter lifecycle).
                // Releasing the live request AND purging the prefix cache
                // makes the next turn replay the cold prefill byte-for-byte.
                if let Some(adapter) = self.paged_adapter.as_mut() {
                    adapter
                        .release_request_and_purge_prefix_cache()
                        .map_err(|e| {
                            Error::from_reason(format!(
                                "qwen3_5 reset_caches: paged prefix-cache purge failed: {e}"
                            ))
                        })?;
                }
                Ok(())
            }
        }
    }

    /// All-or-nothing prefix match (NO exact-match rewind — the GDN
    /// recurrent state cannot rewind one slot; the engine's
    /// exact-match-as-miss handling performs a full reset + re-prefill on a
    /// zero-delta hit). Text-only by construction: the
    /// generic flow never carries images (the vision probe owns those
    /// turns), so the expanded-token / image-key inputs collapse to the
    /// plain prompt.
    fn verify_cache_prefix(&self, tokens: &[u32], reuse_cache: bool) -> usize {
        verify_cache_prefix_direct(
            reuse_cache,
            false,
            tokens,
            tokens,
            0,
            &self.cached_token_history,
            &self.cached_image_key,
            self.caches.is_some(),
        )
    }

    fn flat_caches_desynced(&self) -> bool {
        self.flat_mtp_caches_desynced
    }

    fn clear_flat_caches_desynced(&mut self) {
        self.flat_mtp_caches_desynced = false;
    }

    fn save_cache_state(&mut self, args: SaveStateArgs<'_>) {
        // Delta continuations preserve `cached_image_key` — the KV cache
        // still holds the prior prefill's image attention state even
        // though this turn was text-only. Fresh turns (re)set the key
        // from the turn's (always-false here) `has_images`.
        //
        // `drop_last_always = true`: this is the generic `run_decode_loop`
        // flow (flat, non-MTP, non-image dense turns), which never forwards
        // the final committed token into the physical cache on ANY exit
        // kind. The GDN recurrent state is non-invertible, so we drop that
        // token (rather than materialize it) to keep
        // `cached_token_history.len() == physical_cache_len`.
        if args.is_delta {
            engine::save_cache_state_after_delta(
                args.reuse_cache,
                args.generated_tokens,
                args.finish_reason,
                /* drop_last_always */ true,
                args.save_tokens,
                &mut self.cached_token_history,
                &mut self.cached_image_key,
                &mut self.cached_rope_deltas,
                &mut self.caches,
            );
        } else {
            save_cache_state_direct(
                args.reuse_cache,
                args.has_images,
                args.generated_tokens,
                args.finish_reason,
                /* drop_last_always */ true,
                args.save_tokens,
                args.save_expanded_tokens,
                args.image_cache_key,
                &mut self.cached_token_history,
                &mut self.cached_image_key,
                &mut self.cached_rope_deltas,
                &mut self.caches,
            );
        }
        self.cached_paged_image_token_positions.clear();
    }

    fn eval_caches(&self) -> Result<()> {
        // No post-prefill cache sync on qwen3.5's reference paths:
        // `chunked_prefill` evals internally per chunk and the decode
        // loop schedules async evals. A blocking sync here would
        // introduce an unnecessary stall.
        Ok(())
    }

    fn prefill(&mut self, prompt_tokens: &[u32], stream: Stream) -> Result<MxArray> {
        // Text-only prefill block (the engine's reset-or-delta split already
        // ran; `self.caches` holds either fresh caches or the live session
        // state).
        let prompt = MxArray::from_uint32(prompt_tokens, &[1, prompt_tokens.len() as i64])?;
        chunked_prefill(
            &prompt,
            &self.embedding,
            &mut self.layers,
            &mut self.caches,
            &self.final_norm,
            &self.lm_head,
            stream,
            self.turn_cancel.as_deref(),
        )
    }

    type Decode<'a>
        = Qwen35Decode<'a>
    where
        Self: 'a;

    fn begin_decode(&mut self, turn: &TurnSetup<'_>) -> Result<Self::Decode<'_>> {
        let p = turn.params;

        let is_streaming = self.turn_is_streaming.get();

        // Decode-entry trace (sync paths only — the streaming cores never
        // logged it). `enable_mtp && has_mtp_weights` turns route through
        // `mtp_turn`, so the MTP branch string is unreachable here.
        if !is_streaming {
            let prefill_len = turn.total_seq_len as i32;
            let max_kv_len_estimate =
                engine::kv_capacity_round_up_saturating(prefill_len, p.max_new_tokens);
            let has_mtp = self.has_mtp_weights();
            let branch = if !p.enable_mtp {
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
                turn.total_seq_len,
                p.max_new_tokens,
                p.enable_mtp,
                p.mtp_depth,
                prefill_len,
                max_kv_len_estimate,
                has_mtp,
                turn.is_delta,
                turn.has_images,
                branch,
            );
        }

        let embedding = self.embedding.clone();

        let relabel = match (is_streaming, turn.is_delta) {
            (false, _) => "chat_rust",
            (true, false) => "chat_stream_rust",
            (true, true) => "chat_stream_delta_rust",
        };

        Ok(Qwen35Decode {
            inner: self,
            embedding,
            relabel,
        })
    }

    fn execution_plan(&self) -> ExecutionPlan {
        let speculative = if self.dflash2.is_some() {
            Some(SpeculativePlan {
                kind: SpeculativeKind::DraftModel,
                supported_input_media: MediaCapabilities::NONE,
                supported_context_media: MediaCapabilities::NONE,
                // DFlash2 verification currently uses the flat hybrid-cache
                // snapshot/tape replay path. The target's paged adapter stays
                // installed for ordinary AR turns but is not used here.
                supports_paged_attention: false,
                supports_streaming: true,
            })
        } else {
            self.has_mtp_weights().then_some(SpeculativePlan {
                kind: SpeculativeKind::NativeMtp,
                supported_input_media: MediaCapabilities::NONE,
                supported_context_media: MediaCapabilities::IMAGES,
                supports_paged_attention: true,
                supports_streaming: true,
            })
        };
        ExecutionPlan {
            media: qwen35_dense_media_plan(
                self.vision_encoder.is_some(),
                self.image_processor.is_some(),
                self.paged_adapter.is_some(),
            ),
            paged_attention: self.paged_adapter.as_ref().map(|_| PagedAttentionPlan {
                supports_delta: true,
            }),
            speculative,
        }
    }

    fn wired_limit_bytes(&self) -> Option<usize> {
        // Per-turn wired-memory limit = the model's estimated footprint.
        let draft = self
            .dflash2
            .as_ref()
            .map_or(0usize, |draft| draft.weight_bytes as usize);
        Some((self.config.estimate_memory_bytes() as usize).saturating_add(draft))
    }

    fn profiler_label(&self, is_delta: bool, is_streaming: bool) -> &'static str {
        // Record the turn's streaming-ness for `begin_decode`'s relabel
        // (`TurnSetup` does not carry it). The session core calls this
        // hook exactly once per generic-flow turn, before
        // `begin_decode`; specialized whole-turn paths return from their
        // planned executor earlier and never consult either hook. The labels
        // are the engine defaults.
        self.turn_is_streaming.set(is_streaming);
        match (is_streaming, is_delta) {
            (false, false) => "chat",
            (false, true) => "chat_delta",
            (true, false) => "chat_stream",
            (true, true) => "chat_stream_delta",
        }
    }

    fn has_live_session(&self) -> bool {
        // Delta guard: `self.caches.is_none()` means there is no
        // initialized session to continue.
        self.caches.is_some()
    }

    fn session_media(&self) -> MediaCapabilities {
        qwen35_dense_session_media(
            self.cached_image_key.is_some(),
            self.cached_rope_deltas.is_some(),
        )
    }

    fn session_media_matches_payloads(&self, images: &[Vec<u8>], audio: &[Vec<u8>]) -> bool {
        qwen35_dense_session_media_matches_payloads(self.cached_image_key, images, audio)
    }

    fn template_history_comparison_tokens<'a>(
        &self,
        tokens: &'a [u32],
    ) -> std::borrow::Cow<'a, [u32]> {
        engine::collapse_cached_media_placeholder_runs(
            tokens,
            IMAGE_TOKEN_ID as u32,
            &self.cached_paged_image_token_positions,
        )
    }

    fn run_paged_turn(&mut self, args: &mut WholeTurnArgs<'_>) -> Result<TurnOutput> {
        // Both SYNC and STREAMING turns take the paged core. The paged
        // cores self-handle MTP via the `eager_mtp_paged` arm
        // (`paged_turn_sync_core_inner` / `paged_turn_stream_core_inner`),
        // the streaming eager-MTP path.
        debug_assert!(args.plan.use_paged_attention);
        debug_assert!(self.paged_adapter.is_some());
        self.paged_whole_turn(args)
    }

    fn run_speculative_turn(&mut self, args: &mut WholeTurnArgs<'_>) -> Result<TurnOutput> {
        // The execution plan has already established request opt-in, loaded
        // MTP weights, flat-cache admission, and text-only input. The dense
        // core retains the algorithm-specific MTP gate and AR fallback.
        debug_assert!(args.media.is_empty());
        match args.plan.decoder {
            DecoderPlan::Speculative(SpeculativeKind::DraftModel) => self.dflash2_chat_turn(args),
            DecoderPlan::Speculative(SpeculativeKind::NativeMtp) => self.dense_whole_turn(args),
            _ => Err(Error::from_reason(
                "Qwen3.5 speculative turn received a non-speculative decoder plan",
            )),
        }
    }

    fn run_multimodal_turn(&mut self, args: &mut WholeTurnArgs<'_>) -> Result<TurnOutput> {
        // The dense cores own the full image pipeline (VLM prefill, M-RoPE
        // deltas, paged-backend validation, missing-encoder error).
        self.dense_whole_turn(args)
    }
}
