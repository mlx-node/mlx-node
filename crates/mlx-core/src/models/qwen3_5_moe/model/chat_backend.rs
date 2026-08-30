//! `ChatBackend` for `Qwen35MoeInner` plus the speculative plan it publishes.

use super::*;

impl ChatBackend for Qwen35MoeInner {
    fn set_turn_cancel_flag(&mut self, flag: Option<Arc<AtomicBool>>) {
        self.turn_cancel = flag;
    }

    fn tokenizer(&self) -> Result<Arc<Qwen3Tokenizer>> {
        self.tokenizer
            .clone()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))
    }

    fn family_name(&self) -> &'static str {
        "qwen3_5_moe"
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

    fn extra_eos_ids(&self) -> Vec<u32> {
        self.gen_defaults.eos_token_ids.clone()
    }

    // thinking: engine default `policy()` == `ThinkingPolicy::TemplateHonoring`
    // → `thinking_setup` resolves to the legacy
    // `{enabled: resolve_enable_thinking(config).unwrap_or(true),
    //   budget: config.thinking_token_budget}`.

    fn cached_token_history(&self) -> &[u32] {
        &self.cached_token_history
    }

    fn reset_caches(&mut self, scope: ResetScope) -> Result<()> {
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
                self.caches = Some(fresh_moe_layer_caches(&self.config));
                Ok(())
            }
            // Full clear including history, image key, rope deltas, GDN
            // checkpoints, via `reset_caches_sync`.
            //
            // The EXPLICIT command reset must additionally restore a
            // fully COLD paged state. `reset_caches_sync` does not touch
            // the paged adapter at all (it only clears the flat caches +
            // reuse state), so the prior turn's full blocks stay
            // content-addressed in the per-instance BlockAllocator's
            // prefix cache. A reset-then-rerun of the same prompt would
            // then take the prefix-hit 1-token-suffix prefill
            // (`find_cached_prefix_per_block_with_max_tokens` ->
            // `find_longest_cache_hit`) instead of the cold full prefill,
            // a different bf16 reduction order that can flip a greedy
            // near-tie (observed on the lfm2 sibling; qwen3_5_moe shares
            // the identical adapter lifecycle). One
            // call both releases the live request and purges every
            // prefix-cache entry. `ResetScope::PrefixMiss` (turn-internal)
            // keeps the prefix cache: cross-request block reuse after a
            // history miss is the paged design's entire point.
            ResetScope::Command => {
                self.reset_caches_sync()?;
                if let Some(adapter) = self.paged_adapter.as_mut() {
                    adapter
                        .release_request_and_purge_prefix_cache()
                        .map_err(|e| {
                            Error::from_reason(format!(
                                "qwen3.5-moe reset_caches: paged prefix-cache purge failed: {e}"
                            ))
                        })?;
                }
                Ok(())
            }
        }
    }

    /// All-or-nothing prefix match (NO exact-match rewind — the 30 GDN
    /// linear-attention layers carry a recurrent state that cannot
    /// rewind one slot; the engine's exact-match-as-miss handling
    /// performs a full reset + re-prefill on a zero-delta hit).
    /// Text-only by construction: the generic flow never
    /// carries images (the vision probe owns those turns), so the
    /// expanded-token / image-key inputs collapse to the plain prompt.
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
        // `drop_last_always = true`: generic `run_decode_loop` flow (flat,
        // non-MTP, non-image MoE turns) never forwards the final committed
        // token into the physical cache on any exit kind, and the GDN
        // recurrent state is non-invertible, so drop it to keep
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
        // No post-prefill cache sync on the MoE reference paths:
        // `chunked_prefill` evals internally per chunk and the decode
        // loop schedules async evals. A blocking sync here would
        // introduce an unnecessary stall.
        Ok(())
    }

    fn prefill(&mut self, prompt_tokens: &[u32], stream: Stream) -> Result<MxArray> {
        // Text-only prefill block (the engine's reset-or-delta split already
        // ran; `self.caches` holds either fresh caches or the live session
        // state). Unlike dense, the MoE `chunked_prefill` returns the
        // full `[1, seq, vocab]` logits, so the slice+squeeze to the last
        // position folds in here (the engine's prefill contract is
        // last-token logits).
        let prompt = MxArray::from_uint32(prompt_tokens, &[1, prompt_tokens.len() as i64])?;
        let fa_idx = self.fa_idx;
        let turn_cancel = self.turn_cancel.clone();
        let logits = chunked_prefill(
            &prompt,
            &self.embedding,
            &mut self.layers,
            &mut self.caches,
            &self.final_norm,
            &self.lm_head,
            fa_idx,
            stream,
            turn_cancel.as_deref(),
        )?;
        let seq_len = logits.shape_at(1)?;
        let last_logits = logits.slice_axis(1, seq_len - 1, seq_len)?;
        last_logits.squeeze(Some(&[1]))
    }

    type Decode<'a>
        = Qwen35MoeDecode<'a>
    where
        Self: 'a;

    fn begin_decode(&mut self, turn: &TurnSetup<'_>) -> Result<Self::Decode<'_>> {
        // NOTE: no decode-entry `info!` trace here — unlike dense, the
        // MoE path does not log a "chat_decode entry" line.
        let is_streaming = self.turn_is_streaming.get();

        let embedding = self.embedding.clone();

        let relabel = match (is_streaming, turn.is_delta) {
            (false, false) => "moe_chat_rust",
            (false, true) => "moe_chat_delta_rust",
            (true, false) => "moe_chat_stream_rust",
            (true, true) => "moe_chat_stream_delta_rust",
        };

        Ok(Qwen35MoeDecode {
            inner: self,
            embedding,
            relabel,
        })
    }

    fn execution_plan(&self) -> ExecutionPlan {
        ExecutionPlan {
            media: qwen35_moe_media_plan(
                self.vision_encoder.is_some(),
                self.image_processor.is_some(),
                self.paged_adapter.is_some(),
            ),
            paged_attention: self.paged_adapter.as_ref().map(|_| PagedAttentionPlan {
                supports_delta: true,
            }),
            speculative: self.has_mtp_weights().then(qwen35_moe_speculative_plan),
        }
    }

    fn wired_limit_bytes(&self) -> Option<usize> {
        // Per-turn wired-memory limit = the model's estimated footprint.
        Some(self.config.estimate_memory_bytes() as usize)
    }

    fn profiler_label(&self, is_delta: bool, is_streaming: bool) -> &'static str {
        // Record the turn's streaming-ness for `begin_decode`'s relabel
        // (`TurnSetup` does not carry it). The session core calls this
        // hook exactly once per generic-flow turn, before
        // `begin_decode`; specialized whole-turn paths return from their
        // planned executor earlier and never consult either hook.
        self.turn_is_streaming.set(is_streaming);
        match (is_streaming, is_delta) {
            (false, false) => "moe_chat",
            (false, true) => "moe_chat_delta",
            (true, false) => "moe_chat_stream",
            (true, true) => "moe_chat_stream_delta",
        }
    }

    fn has_live_session(&self) -> bool {
        // Delta guard: `self.caches.is_none()` means there is no
        // initialized session to continue.
        self.caches.is_some()
    }

    fn session_media(&self) -> MediaCapabilities {
        qwen35_moe_session_media(
            self.cached_image_key.is_some(),
            self.cached_rope_deltas.is_some(),
        )
    }

    fn session_media_matches_payloads(&self, images: &[Vec<u8>], audio: &[Vec<u8>]) -> bool {
        qwen35_moe_session_media_matches_payloads(self.cached_image_key, images, audio)
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
        // BOTH paged decoders run through the generic `run_paged_turn`, which
        // drives the adapter lifecycle via [`PagedBackend`]: autoregressive on
        // the shared decode loop, native MTP on the speculative branch
        // (`admit_paged_speculative_decode` + `run_paged_speculative_decode`).
        // Sharing one driver is what keeps the two decoders on one epilogue.
        debug_assert!(args.plan.use_paged_attention);
        debug_assert!(self.paged_adapter.is_some());
        debug_assert!(matches!(
            args.plan.decoder,
            DecoderPlan::Autoregressive | DecoderPlan::Speculative(SpeculativeKind::NativeMtp)
        ));
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
        crate::engine::paged_turn::run_paged_turn(self, &mut constrained_args)
    }

    fn run_speculative_turn(&mut self, args: &mut WholeTurnArgs<'_>) -> Result<TurnOutput> {
        // The execution plan has already established request opt-in, loaded
        // MTP weights, flat-cache admission, and text-only input.
        debug_assert!(args.media.is_empty());
        debug_assert!(matches!(
            args.plan.decoder,
            DecoderPlan::Speculative(SpeculativeKind::NativeMtp)
        ));
        self.moe_whole_turn(args)
    }

    fn run_multimodal_turn(&mut self, args: &mut WholeTurnArgs<'_>) -> Result<TurnOutput> {
        // The MoE cores own the full image pipeline (VLM prefill via
        // `vlm_prefill_moe`, M-RoPE deltas, paged-backend validation,
        // missing-encoder error).
        self.moe_whole_turn(args)
    }
}

/// The speculative plan this family publishes once its MTP head is loaded.
///
/// Named rather than inlined into [`ChatBackend::execution_plan`] so the
/// routing gate can compose the REAL published plan with `TurnPlan::resolve`
/// without standing up a 35B model — the flag and the core it promises then
/// cannot drift apart unnoticed.
///
/// Native MTP runs on BOTH lanes: the paged lane through the generic driver's
/// speculative branch (`PagedBackend::admit_paged_speculative_decode` +
/// `run_paged_speculative_decode`), the flat lane through
/// `ChatBackend::run_speculative_turn`. Media stays out of both — an
/// image-bearing turn has no MoE hidden-emitting prefill to seed a drafter.
pub(super) fn qwen35_moe_speculative_plan() -> SpeculativePlan {
    SpeculativePlan {
        kind: SpeculativeKind::NativeMtp,
        supported_input_media: MediaCapabilities::NONE,
        supported_context_media: MediaCapabilities::NONE,
        supports_paged_attention: true,
        supports_streaming: true,
    }
}
