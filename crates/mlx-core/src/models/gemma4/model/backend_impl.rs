//! The engine trait surface: the flat and paged decode steppers, the paged prefix state, and the PagedBackend / ChatBackend implementations for Gemma4Inner.

use super::*;

/// Eager flat decode stepper for one gemma4 turn
/// ([`ChatBackend::begin_decode`]). Runs the flat decode-loop step body:
/// `diagnostic::set_step(step)` before every forward (the
/// `MLX_DEBUG_GEMMA4_DUMP` per-step dump), `forward_inner` over the live
/// session caches, async-eval of the sampled token only (gemma4 never
/// async-evals the logits).
pub(crate) struct Gemma4Decode<'a> {
    inner: &'a mut Gemma4Inner,
    /// Diagnostic step counter. The engine loop has no step index in the
    /// `DecodeStep` seam, so the stepper carries its own 0-based sequence
    /// to feed `set_step`.
    step: i32,
}

impl DecodeStep for Gemma4Decode<'_> {
    fn forward(&mut self, input_ids: &MxArray) -> Result<(MxArray, bool)> {
        let inner = &mut *self.inner;
        let caches = inner
            .caches
            .as_mut()
            .ok_or_else(|| Error::from_reason("Gemma4 decode: caches missing"))?;
        crate::models::gemma4::diagnostic::set_step(self.step);
        self.step += 1;
        let logits = forward_inner(
            input_ids,
            &inner.embed_tokens,
            &inner.layers,
            caches,
            &inner.final_norm,
            &inner.lm_head,
            inner.embed_weight_t.as_ref(),
            inner.ple.as_ref(),
            &inner.config,
        )?;
        // `true` requests the engine's `squeeze(Some(&[1]))`: the eager
        // forward returns `[1, 1, vocab]`.
        Ok((logits, true))
    }

    fn eval_step(&mut self, next_token: &MxArray, _logits: &MxArray, _budget_forced: bool) {
        MxArray::async_eval_arrays(&[next_token]);
    }

    fn materialize_final(&mut self, token_id: u32) -> Result<()> {
        // LENGTH-exit only (the engine gates the call): run ONE more
        // `forward_inner` for the final committed token so its K/V lands in
        // the live session caches, then DISCARD the logits. This makes the
        // per-layer cache offsets equal the keep-all-on-length saved
        // history. No sample / push / emit. Like the paged override, this
        // deliberately does NOT fire a sliding decode-boundary checkpoint.
        let inner = &mut *self.inner;
        let caches = inner
            .caches
            .as_mut()
            .ok_or_else(|| Error::from_reason("Gemma4 materialize_final: caches missing"))?;
        let input_ids = MxArray::from_int32(&[token_id as i32], &[1, 1])?;
        crate::models::gemma4::diagnostic::set_step(self.step);
        self.step += 1;
        let _logits = forward_inner(
            &input_ids,
            &inner.embed_tokens,
            &inner.layers,
            caches,
            &inner.final_norm,
            &inner.lm_head,
            inner.embed_weight_t.as_ref(),
            inner.ple.as_ref(),
            &inner.config,
        )?;
        Ok(())
    }
}

/// Paged decode stepper for gemma4 (pure-eager — no compiled path, so no
/// lifecycle/reset guard fields). Drives
/// [`crate::engine::decode::run_decode_loop`] through
/// [`Gemma4Inner::run_paged_decode_step`], advancing every grouped adapter and
/// pruning physical sliding blocks after their lazy writes materialize.
pub(crate) struct Gemma4PagedDecode<'a> {
    /// Diagnostic step counter, fed to `set_step` before every paged
    /// forward. The engine loop has no step index in the `DecodeStep`
    /// seam, so the stepper carries its own.
    step: i32,
    pending_cache_error: Option<String>,
    inner: &'a mut Gemma4Inner,
}

impl DecodeStep for Gemma4PagedDecode<'_> {
    fn forward(&mut self, input_ids: &MxArray) -> Result<(MxArray, bool)> {
        // The loop hands the already-extracted token via
        // `forward_with_token`; recover it here from the `[1, 1]` input for
        // the bare `forward` contract (idempotent eval with the loop-top
        // `y.eval()`).
        let token_id = input_ids.item_at_int32(0)? as u32;
        self.forward_with_token(input_ids, token_id)
    }

    fn forward_with_token(
        &mut self,
        _input_ids: &MxArray,
        token_id: u32,
    ) -> Result<(MxArray, bool)> {
        crate::models::gemma4::diagnostic::set_step(self.step);
        self.step += 1;
        // `run_paged_decode_step` records the token in the adapter at its
        // top (BEFORE the forward), then returns `[1, 1, vocab]`.
        let logits = self.inner.run_paged_decode_step(token_id)?;
        // `run_paged_decode_step` returns `[1, 1, vocab]`; `true` requests
        // the engine's squeeze of axis 1 (the eager convention).
        Ok((logits, true))
    }

    fn eval_step(&mut self, next_token: &MxArray, _logits: &MxArray, _budget_forced: bool) {
        // Async-eval the sampled token only (gemma4 never async-evals the
        // logits); the loop-top `y.eval()` forces materialization next
        // iteration.
        MxArray::async_eval_arrays(&[next_token]);
    }

    fn maintain_cache(&mut self, step: i32) {
        // The loop has materialized the preceding sample before this hook.
        // That evaluation also completes the preceding KV writes, so blocks
        // wholly outside the sliding window can now be returned to the pool
        // without racing lazy Metal work.
        if self.pending_cache_error.is_none() {
            let seq_id = self.inner.active_paged_seq;
            if let Err(error) = self.inner.settle_grouped_kv_step(seq_id) {
                self.pending_cache_error = Some(error.reason.clone());
            }
        }
        // Paged cadence — the per-step
        // `maybe_clear_cache_for_paged_step(step)`.
        crate::array::maybe_clear_cache_for_paged_step(step);
    }

    fn end_decode(&mut self) -> Result<()> {
        if self.pending_cache_error.is_none() {
            let seq_id = self.inner.active_paged_seq;
            if let Err(error) = self.inner.settle_grouped_kv_step(seq_id) {
                self.pending_cache_error = Some(error.reason.clone());
            }
        }
        self.pending_cache_error
            .take()
            .map_or(Ok(()), |error| Err(Error::from_reason(error)))
    }

    fn materialize_final(&mut self, token_id: u32) -> Result<()> {
        // LENGTH-exit only (the engine gates the call): run ONE more
        // `run_paged_decode_step` for the final committed token so its K/V
        // lands in the paged adapter, then DISCARD the logits. The adapter's
        // `request_tokens()` / cursor advances by exactly 1 to equal the
        // saved keep-all history.
        //
        let _logits = self.inner.run_paged_decode_step(token_id)?;
        Ok(())
    }
}

/// Gemma4 paged prefix state: the common boundary owned by every physical
/// full/sliding group plus the suffix that still needs computation.
/// `full_tokens` retains the prompt so `run_paged_prefill_chunk` can verify that
/// the supplied suffix starts exactly at the common absolute cursor.
pub(crate) struct Gemma4PrefixState {
    pub(crate) effective_cached_prefix_len: usize,
    pub(super) suffix_len: usize,
    pub(crate) sliding_primed_prefix_len: u32,
    pub(crate) cache_salt: u64,
    pub(crate) full_tokens: Vec<u32>,
}

impl PagedPrefix for Gemma4PrefixState {
    fn effective_cached_prefix_len(&self) -> usize {
        self.effective_cached_prefix_len
    }
    fn suffix_len(&self) -> usize {
        self.suffix_len
    }
}

impl PagedBackend for Gemma4Inner {
    type PagedDecode<'a>
        = Gemma4PagedDecode<'a>
    where
        Self: 'a;
    type PrefixState = Gemma4PrefixState;

    fn prime_prefix_state(
        &mut self,
        plan: &[u32],
        reuse_cache: bool,
        _block_size: usize,
        _extra_keys: &[u64],
        cache_salt: u64,
    ) -> Result<Self::PrefixState> {
        let trace_enabled = inference_trace_enabled();
        let total_budget = plan.len() as u32;
        // Per-turn seq_id: the adapter is single-request and the prepare's
        // warm-continue / cold-reset arms make the previous seq_id
        // irrelevant.
        let seq_id = self.active_paged_seq;
        // The prepare runs the adapter's warm-continue / cold-reset arms,
        // applies the vLLM `max_cache_hit_tokens = total_budget - 1` cap,
        // and may ZERO the cached prefix mid-prepare when a large
        // sliding-prefix reuse is suppressed — so the EFFECTIVE
        // post-suppression length surfaces here (never the plan's raw
        // cached_len).
        let prep = self.prepare_gemma4_paged_turn(
            "paged",
            plan,
            reuse_cache,
            total_budget,
            seq_id,
            cache_salt,
            trace_enabled,
        )?;
        Ok(Gemma4PrefixState {
            effective_cached_prefix_len: prep.cached_prefix_len as usize,
            suffix_len: prep.suffix_len as usize,
            sliding_primed_prefix_len: prep.sliding_primed_prefix_len,
            cache_salt,
            // Sliding-layer re-prefill needs the FULL prompt, not just the
            // suffix the engine passes to `paged_prefill`.
            full_tokens: plan.to_vec(),
        })
    }

    fn paged_prefill(
        &mut self,
        suffix_tokens: &[u32],
        prefix: &Self::PrefixState,
        _stream: Stream,
    ) -> Result<MxArray> {
        // Mark the diagnostic step as -1 (prefill) before the forward
        // (diagnostic-only). The engine fires the post-prefill
        // `synchronize_and_clear_cache` AFTER this returns.
        crate::models::gemma4::diagnostic::set_step(-1);
        self.run_paged_prefill_chunk(
            &prefix.full_tokens,
            suffix_tokens,
            prefix.effective_cached_prefix_len as u32,
            prefix.sliding_primed_prefix_len,
            prefix.cache_salt,
            None,
        )
    }

    fn begin_paged_decode(&mut self) -> Result<Self::PagedDecode<'_>> {
        Ok(Gemma4PagedDecode {
            step: 0,
            pending_cache_error: None,
            inner: self,
        })
    }

    fn finalize_paged_turn(&mut self, reuse_cache: bool, cache_salt: u64) {
        self.paged_finalize_failed = false;
        let seq_id = self.active_paged_seq;
        let finalize_result = (|| -> std::result::Result<Vec<Vec<u64>>, String> {
            let coordinator = self.kv_cache_coordinator.as_mut().ok_or_else(|| {
                "Gemma4 hybrid KV coordinator missing during finalize".to_string()
            })?;
            coordinator.activate_request_all(seq_id)?;
            let total_tokens = coordinator.full_adapter().request_tokens().len();
            let extra_keys = engine::build_paged_extra_keys(
                total_tokens,
                coordinator.full_adapter().block_size(),
                &self.cached_paged_image_token_positions,
            );
            coordinator.eval_pending_pool_writes_all()?;
            if reuse_cache {
                coordinator.finalize_keep_live_all(seq_id, &extra_keys, cache_salt)?;
            } else {
                coordinator.register_full_for_cold_capture(seq_id, &extra_keys, cache_salt)?;
            }
            Ok(extra_keys)
        })();
        if let Ok(extra_keys) = finalize_result.as_ref() {
            self.capture_grouped_sliding_cold_sidecar(seq_id, extra_keys, cache_salt);
            if !reuse_cache
                && let Some(coordinator) = self.kv_cache_coordinator.as_mut()
                && let Err(error) = coordinator.release_request_all(seq_id)
            {
                tracing::warn!(
                    target: "mlx_core::gemma4::paged",
                    "Gemma4 paged release after cold capture failed: {error}"
                );
                self.paged_finalize_failed = true;
            }
            if !reuse_cache {
                self.grouped_sliding_cold_checkpoints.remove(&seq_id);
            }
        }
        let finalize_error = finalize_result.err();
        if let Some(error) = finalize_error {
            tracing::warn!(
                target: "mlx_core::gemma4::paged",
                "Gemma4 paged finalize failed: {error}"
            );
            self.paged_finalize_failed = true;
            if let Some(coordinator) = self.kv_cache_coordinator.as_mut() {
                let _ = coordinator.release_request_all(seq_id);
            }
            self.media_session_continuable = false;
        }
    }

    fn abort_paged_turn(&mut self) {
        // Error-path teardown: release the request fully — partial
        // block_table state is unsafe to keep around. Infallible (`let _ =`).
        if let Some(coordinator) = self.kv_cache_coordinator.as_mut() {
            let _ = coordinator.release_request_all(self.active_paged_seq);
        }
        self.caches = None;
        // The paged speculative prefill stashes the drafter's whole-prompt
        // context BEFORE the anchor sample, so an abort raised in that
        // window inherits a stash nothing downstream can consume.
        self.draft_turn_state = None;
        self.clear_reuse_state();
    }

    fn save_paged_history(
        &mut self,
        save_tokens: &[u32],
        generated: &[u32],
        keep_all: bool,
        reuse_cache: bool,
    ) -> Result<()> {
        if self.paged_finalize_failed {
            self.cached_token_history.clear();
            self.media_session_continuable = false;
            return Err(Error::from_reason(
                "Gemma4 paged finalize failed; refusing to publish reusable history",
            ));
        }
        // `run_paged_turn` snapshots the request planner's context here for
        // the duration of the executor. Empty means this text turn is a fresh
        // replacement; non-empty means it extended a media-derived session.
        let continued_media_context = self.paged_text_turn_context;
        // Save token history ONLY — the adapter's pool owns the K/V.
        // `keep_all` is the flat rule (engine: `finish_reason ==
        // "length"`); when it is false the terminal stop token is dropped
        // (DROP-LAST trim). The engine reconciles `request_tokens()` to this
        // same trimmed history via `reconcile_paged_request_tokens` BEFORE
        // finalize, so the adapter and the saved history stay aligned for
        // the next turn's warm-continue.
        if reuse_cache {
            let mut full_history = save_tokens.to_vec();
            let history_tokens = if keep_all || generated.is_empty() {
                generated
            } else {
                &generated[..generated.len() - 1]
            };
            full_history.extend_from_slice(history_tokens);
            self.cached_token_history = full_history;
            if continued_media_context.is_empty() {
                // A successful fresh text turn replaced any previous media
                // session. Its saved/live KV is now genuinely text-only.
                self.cached_image_key = None;
                self.cached_audio_key = None;
                self.cached_paged_image_token_positions.clear();
                self.media_session_context = MediaCapabilities::NONE;
                self.media_session_continuable = false;
            } else {
                // A warm text delta extended the same live media prefix.
                // Preserve the exact image key and ordered placeholder sidecar:
                // subsequent text blocks must keep registering under the same
                // image-aware lineage, and live continuation needs raw identity.
                self.cached_audio_key = None;
                debug_assert!(self.media_session_continuable);
                self.media_session_context = continued_media_context;
            }
            // The scheduler-owned sliding group remains live beside the full
            // group. There is no out-of-pool rotating state to snapshot.
        } else {
            self.cached_token_history.clear();
            // Fresh paged start: a text turn holds no media, so clear any media
            // key a prior turn on this reused model left set (mirrors the flat
            // `save_cache_state` fresh-turn clear). Without the audio clear a
            // text-only start over a model whose last turn was audio would leave
            // `cached_audio_key` stale and the delta image guard would wrongly
            // force an "audio state" restart on the text-only session.
            self.cached_image_key = None;
            self.cached_audio_key = None;
            self.cached_paged_image_token_positions.clear();
            self.media_session_context = MediaCapabilities::NONE;
            self.media_session_continuable = false;
        }
        Ok(())
    }

    fn reconcile_paged_request_tokens(
        &mut self,
        prompt_len: usize,
        generated: &[u32],
        keep_all: bool,
    ) -> bool {
        // Perf-parity warm-continue restore (see the trait doc). The
        // pipelined decode loop records the stop token into the adapter
        // (its forward ran at the loop top BEFORE the stop-check), but the
        // saved history DROPS it on a non-length exit. Roll the adapter back
        // to the to-be-saved history length so `request_tokens()` matches
        // the persisted history. `history_len` uses the EXACT same trim as
        // `save_paged_history`; `saturating_sub` makes it a no-op on a length
        // exit (`materialize_final` already recorded the final token) and on
        // a final-step stop (forward never ran).
        let Some(coordinator) = self.kv_cache_coordinator.as_mut() else {
            return true;
        };
        if let Err(error) = coordinator.activate_request_all(self.active_paged_seq) {
            tracing::warn!(
                target: "mlx_core::gemma4::paged",
                "reconcile_paged_request_tokens: activation failed: {error}",
            );
            return false;
        }
        let history_len = if keep_all || generated.is_empty() {
            generated.len()
        } else {
            generated.len() - 1
        };
        let target_len = prompt_len + history_len;
        let surplus = coordinator
            .full_adapter()
            .request_tokens()
            .len()
            .saturating_sub(target_len);
        if surplus > 0
            && let Err(e) =
                coordinator.rollback_last_tokens_all(self.active_paged_seq, surplus as u32)
        {
            tracing::warn!(
                target: "mlx_core::gemma4::paged",
                "reconcile_paged_request_tokens: rollback_last_tokens({surplus}) failed \
                 (finalize releases the request; next turn cold-prefills): {e}",
            );
            return false;
        }
        true
    }
}

impl Gemma4Inner {
    fn record_output_parser_prompt_state(
        &self,
        tok: &Qwen3Tokenizer,
        rendered_tokens: &[u32],
    ) -> Result<()> {
        let open_channel = tok.encode_sync("<|channel>thought\n", Some(false))?;
        self.output_starts_in_reasoning_channel.store(
            !open_channel.is_empty() && rendered_tokens.ends_with(&open_channel),
            Ordering::Relaxed,
        );
        Ok(())
    }

    pub(super) fn output_starts_in_reasoning_channel(&self) -> bool {
        self.output_starts_in_reasoning_channel
            .load(Ordering::Relaxed)
    }
}

impl ChatBackend for Gemma4Inner {
    fn tokenizer(&self) -> Result<Arc<Qwen3Tokenizer>> {
        self.tokenizer
            .clone()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))
    }

    fn family_name(&self) -> &'static str {
        "gemma4"
    }

    fn set_turn_cancel_flag(&mut self, flag: Option<Arc<AtomicBool>>) {
        self.turn_cancel = flag;
    }

    fn session_eos_id(&self, _tok: &Qwen3Tokenizer) -> Result<u32> {
        // Gemma4 stops on its `<turn|>` turn terminator, not `<|im_end|>`.
        self.turn_end_id()
    }

    fn policy(&self) -> engine::ThinkingPolicy {
        // Gemma4's selectable mode is a PROMPT capability (`<|think|>` in
        // the first system turn), not a Qwen-style `<think>...</think>`
        // decode region. Keep the generic tracker disabled: it has no
        // `<channel|>` end-token support and enabling it would incorrectly
        // classify every generated token as reasoning. Segmentation remains
        // downstream in `parse_gemma4_output` / `Gemma4StreamParser`, keyed
        // on `<|channel>` markers. Consequently Gemma4 still has no generic
        // think-budget forcing and reports `reasoning_tokens: 0`.
        engine::ThinkingPolicy::None
    }

    fn resolve_params(&self, config: &ChatConfig) -> ChatParams {
        let mut p = engine::extract_chat_params(config);
        // Fold the MODEL-config sampling defaults in; unset → T=0 greedy.
        // The engine's `sampling::sample` argmax fast path at T=0 is the
        // greedy argmax.
        p.sampling_config = make_sampling_config(config, &self.config);
        // gemma4 treats the penalty fields as no-ops. Neutralize so the
        // engine's `apply_all_penalties` skips all penalty work
        // structurally.
        p.repetition_penalty = 1.0;
        p.presence_penalty = 0.0;
        p.frequency_penalty = 0.0;
        // gemma4 ALWAYS returns Some(PerformanceMetrics), regardless of
        // `config.report_performance`.
        p.report_performance = true;
        // gemma4 never suppresses reasoning deltas at the loop level
        // (`include_reasoning` is a no-op here; the stream parser routes
        // channel segments itself). Defensive: pin `true` so the engine's
        // emitter gate can never suppress.
        p.include_reasoning = true;
        // Draft depth: with a draft model loaded, `mtpDepth` resolves per
        // variant — a family-local post-edit of the engine's central
        // `[1, 5]` clamp (an MTP-head contract that does not apply to
        // external drafts), always clamping from the RAW config value.
        //   * DSpark: unset runs full draft blocks (`block_size`, 7 on the
        //     v1 checkpoint) with the measured target-AR break-even guard;
        //     explicit depth pins the cap and disables that guard unless
        //     `mtpAdaptiveDepth: true` explicitly opts it back in.
        //   * Assistant: chained AR drafting has no checkpoint-pinned block
        //     size — unset resolves to `ASSISTANT_DEFAULT_DEPTH`; explicit
        //     values clamp to `[1, ASSISTANT_MAX_DEPTH]`.
        match self.draft.as_ref() {
            Some(Gemma4Draft::Dspark(draft)) => {
                let block_size = draft.config.block_size;
                p.mtp_depth = match config.mtp_depth {
                    Some(d) => (d.max(1) as usize).min(block_size),
                    None => block_size,
                };
                p.mtp_adaptive_depth = match config.mtp_adaptive_depth {
                    Some(enabled) => enabled,
                    None => config.mtp_depth.is_none(),
                };
            }
            Some(Gemma4Draft::Assistant(_)) => {
                p.mtp_depth = match config.mtp_depth {
                    Some(d) => (d.max(1) as usize)
                        .min(crate::models::gemma4::assistant::ASSISTANT_MAX_DEPTH),
                    None => crate::models::gemma4::assistant::ASSISTANT_DEFAULT_DEPTH,
                };
            }
            None => {}
        }
        p
    }

    /// Render every turn from the checkpoint-provided template. Gemma's
    /// stream parser also needs to know whether that template left the
    /// generation prompt inside an open thought channel.
    fn render_prompt(
        &self,
        tok: &Qwen3Tokenizer,
        messages: &[ChatMessage],
        config: &ChatConfig,
        preserve_thinking: bool,
    ) -> Result<Vec<u32>> {
        let tokens = tok.apply_chat_template_sync(
            messages,
            Some(true),
            config.tools.as_deref(),
            engine::resolve_enable_thinking(config),
            preserve_thinking,
        )?;
        self.record_output_parser_prompt_state(tok, &tokens)?;
        Ok(tokens)
    }

    fn cached_token_history(&self) -> &[u32] {
        &self.cached_token_history
    }

    fn reset_caches(&mut self, scope: ResetScope) -> Result<()> {
        // Legacy miss branch ran `reset_caches_sync()? +
        // init_caches_sync()?` back-to-back (the flat prefill needs live
        // caches); the explicit command reset only cleared (caches stay
        // `None` until the next turn's lazy init).
        self.reset_caches_sync()?;
        if scope == ResetScope::PrefixMiss {
            self.init_caches_sync()?;
        }
        // The EXPLICIT command reset must restore a fully cold state.
        // gemma4's flat reset path (`reset_caches_sync`) never touches the
        // paged adapter, so a prior turn's request stays live AND its full
        // blocks stay content-addressed in the per-instance BlockAllocator's
        // prefix cache. A reset-then-rerun of the same prompt would then take
        // the prefix-hit suffix-prefill path (via `find_longest_cache_hit`
        // inside `prepare_gemma4_paged_turn`) — a different bf16 reduction
        // order than the cold full prefill, enough to flip a greedy
        // near-tie.
        // `release_request_and_purge_prefix_cache` releases the live request
        // (the release gemma4's reset otherwise skips) AND purges every
        // prefix-cache entry. The turn-internal `PrefixMiss` reset keeps the
        // prefix cache (cross-request block reuse after a history miss is the
        // paged design's entire point).
        if scope == ResetScope::Command
            && let Some(coordinator) = self.kv_cache_coordinator.as_mut()
        {
            coordinator.release_all_and_purge().map_err(|e| {
                Error::from_reason(format!(
                    "gemma4 reset_caches: paged prefix-cache purge failed: {e}"
                ))
            })?;
        }
        Ok(())
    }

    /// Prefix-reuse check. The engine routes every media-bearing turn
    /// through the multimodal executor BEFORE this check, so only the
    /// session-side media gate (`session_media()` non-empty → miss) is needed
    /// here; there is no `has_images` parameter.
    ///
    /// All-or-nothing: returns `0` or `cached.len()` (exact-match falls
    /// through the `hit == tokens.len()` branch in the session core to
    /// the miss/reset path — gemma4's sliding-window cache has no "rewind
    /// by one" primitive).
    fn verify_cache_prefix(&self, tokens: &[u32], reuse_cache: bool) -> usize {
        if !reuse_cache {
            return 0;
        }
        // Text-only prefix reuse: force a miss whenever the cached
        // session holds image or audio state UNLESS the media turn is
        // continuable (kept-live + sliding checkpoint at the full prefix). This
        // keeps prefix reuse strictly aligned with text-only sessions and
        // sidesteps the media-key coordination the Qwen3.5 shared helper
        // handles, while letting a continuable media session reuse an
        // exactly-cached prefix. Held state is `session_media()` (raw keys ∪
        // persistent `media_session_context`), not the raw keys alone: after a
        // failed media prepare on a warm-continued session only the context
        // survives, and the media-expanded cached history must not seed a
        // text-only prefix hit.
        if !self.session_media().is_empty() && !self.media_session_continuable {
            return 0;
        }
        let cached = &self.cached_token_history;
        if cached.is_empty() {
            return 0;
        }
        if tokens.len() < cached.len() {
            return 0;
        }
        if tokens[..cached.len()] != cached[..] {
            return 0;
        }
        if self.active_flat_session {
            if self.caches.is_none() {
                return 0;
            }
        } else {
            let Some(coordinator) = self.kv_cache_coordinator.as_ref() else {
                return 0;
            };
            if !coordinator.can_continue_all(self.active_paged_seq, tokens) {
                return 0;
            }
        }
        cached.len()
    }

    fn save_cache_state(&mut self, args: SaveStateArgs<'_>) {
        // Flat save (identical on the fresh and delta paths): persist
        // `prompt + generated`, dropping the terminal turn-boundary token
        // when the decode terminated on stop so the cached history ends on
        // the `<turn|>` boundary the next delta re-renders itself.
        // Unconditional — there is no `reuse_cache` branch here (only the
        // paged core has one, and paged turns never reach this hook), and
        // the engine's session_start guard rejects `reuse_cache=Some(false)`
        // anyway.
        let history_tokens: &[u32] =
            if args.finish_reason != "length" && !args.generated_tokens.is_empty() {
                &args.generated_tokens[..args.generated_tokens.len() - 1]
            } else {
                args.generated_tokens
            };
        let mut new_history = Vec::with_capacity(args.save_tokens.len() + history_tokens.len());
        new_history.extend_from_slice(args.save_tokens);
        new_history.extend_from_slice(history_tokens);
        self.cached_token_history = new_history;
        if !args.is_delta {
            // Fresh text-only turn: clear any stale image/audio key (a
            // text-only turn has no multimodal key to set). Delta turns leave
            // them untouched — text-only by the delta image guard, so they are
            // structurally `None`.
            self.cached_image_key = None;
            self.cached_audio_key = None;
            self.media_session_context = MediaCapabilities::NONE;
            self.media_session_continuable = false;
        }
    }

    fn eval_caches(&self) -> Result<()> {
        // Materialize the prefill KV before entering the decode loop.
        eval_gemma4_caches(
            self.caches
                .as_ref()
                .ok_or_else(|| Error::from_reason("Gemma4 eval_caches: caches missing"))?,
        )
    }

    /// Flat prefill for the engine's generic flow. `prefill_body_gemma4`
    /// processes `tokens[0 .. N-1]` through the body (a no-op when
    /// `N == 1`), the per-layer KV evals materialize, then the last
    /// token runs the full forward for sampling-ready `[1, vocab]`
    /// logits. Serves the fresh path (full prompt or strict-extend
    /// tail) and the session-delta path identically.
    ///
    /// `diagnostic::set_step(-1)` marks the prefill forward for
    /// `MLX_DEBUG_GEMMA4_DUMP`, uniformly across entry points.
    fn prefill(&mut self, prompt_tokens: &[u32], stream: Stream) -> Result<MxArray> {
        // Defensive: caches must be live before the prefill runs. The
        // engine's miss-reset re-inits, and verify/`has_live_session`
        // check liveness — but if somebody cleared the caches
        // out-of-band between turns, re-init here.
        if self.caches.is_none() {
            self.init_caches_sync()?;
        }

        let prefill_slice: Vec<i32> = prompt_tokens.iter().map(|&t| t as i32).collect();
        let prefill_len = prefill_slice.len();
        let prompt = MxArray::from_int32(&prefill_slice, &[1, prefill_len as i64])?;

        {
            let _stream_ctx = StreamContext::new(stream);
            let caches = self
                .caches
                .as_mut()
                .ok_or_else(|| Error::from_reason("Gemma4 prefill: caches missing"))?;
            prefill_body_gemma4(
                &prompt,
                &self.embed_tokens,
                &self.layers,
                caches,
                &self.final_norm,
                self.ple.as_ref(),
                &self.config,
                self.turn_cancel.as_deref(),
            )?;
        }
        eval_gemma4_caches(
            self.caches
                .as_ref()
                .ok_or_else(|| Error::from_reason("Gemma4 prefill: caches missing"))?,
        )?;

        // Last token → logits. `prefill_body_gemma4` processed
        // `[0 .. prefill_len - 1]` and left the final token for us.
        let last_token = prompt.slice_axis(1, prefill_len as i64 - 1, prefill_len as i64)?;
        let logits = {
            let _stream_ctx = StreamContext::new(stream);
            let caches = self
                .caches
                .as_mut()
                .ok_or_else(|| Error::from_reason("Gemma4 prefill: caches missing"))?;
            crate::models::gemma4::diagnostic::set_step(-1);
            forward_inner(
                &last_token,
                &self.embed_tokens,
                &self.layers,
                caches,
                &self.final_norm,
                &self.lm_head,
                self.embed_weight_t.as_ref(),
                self.ple.as_ref(),
                &self.config,
            )?
        };
        logits.squeeze(Some(&[1]))
    }

    type Decode<'a>
        = Gemma4Decode<'a>
    where
        Self: 'a;

    fn begin_decode(&mut self, _turn: &TurnSetup<'_>) -> Result<Self::Decode<'_>> {
        // No compiled path, no turn-constant captures: gemma4's eager
        // decode threads everything through the live session caches.
        Ok(Gemma4Decode {
            inner: self,
            step: 0,
        })
    }

    /// Gemma4 output finalization: raw decode (`skip_special_tokens =
    /// false` so the channel/tool-call DSL markers survive) →
    /// `parse_gemma4_output` → `promote_channel_only_output` →
    /// tool-calls finish-reason promotion. `reasoning_tokens` arrives as
    /// 0 (thinking disabled) and `prompt_tokens` / `performance` are
    /// passed through unchanged. `cached_tokens` is overwritten by the
    /// session core.
    fn finalize_turn(&self, args: FinalizeArgs<'_>) -> Result<ChatResult> {
        let raw_text = args.tokenizer.decode_sync(args.generated_tokens, false)?;
        let starts_in_prompted_channel = self.output_starts_in_reasoning_channel();
        let mut parsed =
            crate::models::gemma4::output_parser::parse_gemma4_output_with_open_channel(
                &raw_text,
                starts_in_prompted_channel,
            );
        promote_channel_only_output(&mut parsed, starts_in_prompted_channel);
        let finish_reason = if parsed.tool_calls.iter().any(|tc| tc.status == "ok") {
            "tool_calls".to_string()
        } else {
            args.finish_reason
        };
        Ok(ChatResult {
            text: parsed.text,
            tool_calls: parsed.tool_calls,
            thinking: parsed.thinking,
            // Generic and paged session cores overwrite this with the
            // effective Jinja kwarg after family finalization.
            thinking_enabled: args.thinking_enabled,
            num_tokens: args.generated_tokens.len() as u32,
            prompt_tokens: args.prompt_tokens,
            reasoning_tokens: args.reasoning_tokens,
            finish_reason,
            raw_text,
            public_raw_text: None,
            cached_tokens: 0,
            performance: args.performance,
        })
    }

    fn execution_plan(&self) -> ExecutionPlan {
        // The scheduler selects one physical cache lane before entering the
        // model-neutral planner. An ASSISTANT-draft command temporarily
        // installs flat target caches, so the resident paged pools are hidden
        // for that command; every other command exposes them as usual.
        let paged_available = self.kv_cache_coordinator.is_some() && !self.active_flat_session;
        let image_components_loaded = self.image_path_loaded();
        let audio_embedder_loaded = self.embed_audio.is_some();
        let speculative = match self.draft.as_ref() {
            // DSpark proposes over tapped residual hiddens and verifies as a
            // block against the target's PAGED pools; there is no flat lane
            // for it, so it is offered only where those pools are visible.
            Some(Gemma4Draft::Dspark(_)) => paged_available.then_some(SpeculativePlan {
                kind: SpeculativeKind::DraftModel,
                supported_input_media: MediaCapabilities::NONE,
                supported_context_media: MediaCapabilities::NONE,
                supports_paged_attention: true,
                supports_streaming: true,
            }),
            // The assistant drafter reads the target's flat `Gemma4LayerCache`
            // K/V arrays directly for its Q-only attention, which the pools
            // cannot hand it — so it declares no paged support and the planner
            // routes it to the flat speculative handler.
            Some(Gemma4Draft::Assistant(_)) => Some(SpeculativePlan {
                kind: SpeculativeKind::DraftModel,
                supported_input_media: MediaCapabilities::NONE,
                supported_context_media: MediaCapabilities::NONE,
                supports_paged_attention: false,
                supports_streaming: true,
            }),
            None => None,
        };
        ExecutionPlan {
            media: gemma4_media_plan(
                image_components_loaded,
                audio_embedder_loaded,
                paged_available,
            ),
            paged_attention: paged_available.then_some(PagedAttentionPlan {
                supports_delta: true,
            }),
            speculative,
        }
    }

    fn extra_eos_ids(&self) -> Vec<u32> {
        // The MODEL-config eos list (`<eos>` / `<end_of_turn>`) honored
        // alongside the session `<turn|>` id. A negative config id can
        // never equal a `u32`-cast token, so filter those out instead of
        // wrapping.
        self.config
            .eos_token_ids
            .iter()
            .filter(|&&id| id >= 0)
            .map(|&id| id as u32)
            .collect()
    }

    fn stream_skip_special_tokens(&self) -> bool {
        // `decode_stream(false)`: the stream parser must see the
        // `<|channel>` / `<|tool_call>` markers. The residual flush then
        // decodes with the same flag (engine guarantee), keeping
        // `streamed_text_len` accounting consistent.
        false
    }

    fn stream_emitter(&self) -> Box<dyn StreamEmitter> {
        Box::new(Gemma4Emitter::new(
            self.output_starts_in_reasoning_channel(),
        ))
    }

    // `augment_performance` deliberately NOT overridden: the default
    // (`profiler.fill_mtp_acceptance`) fills the `mtp_*` acceptance fields
    // after a DSpark turn (and copies `profile_phases` when profiling is
    // enabled). AR turns record no MTP cycle, so their acceptance fields
    // stay `None` as before.

    fn has_live_session(&self) -> bool {
        !self.cached_token_history.is_empty()
            && if self.active_flat_session {
                self.caches.is_some()
            } else {
                self.kv_cache_coordinator
                    .as_ref()
                    .is_some_and(|coordinator| coordinator.is_live_all(self.active_paged_seq))
            }
    }

    fn session_media(&self) -> MediaCapabilities {
        // Keys cover a just-finalized media turn and direct/transitional test
        // states. `media_session_context` remains authoritative after warm
        // text saves clear those keys while preserving the same live media KV.
        self.media_session_context.union(MediaCapabilities {
            images: self.cached_image_key.is_some(),
            audio: self.cached_audio_key.is_some(),
        })
    }

    fn session_media_matches_payloads(&self, images: &[Vec<u8>], audio: &[Vec<u8>]) -> bool {
        gemma4_session_media_matches_payloads(
            self.media_session_continuable,
            self.cached_image_key,
            self.cached_audio_key,
            images,
            audio,
        )
    }

    const REASONING_CLOSE_TAG: &'static str =
        crate::models::gemma4::output_parser::reasoning_close_tag();

    fn template_history_comparison_tokens<'a>(
        &self,
        tokens: &'a [u32],
    ) -> std::borrow::Cow<'a, [u32]> {
        engine::collapse_cached_media_placeholder_runs(
            tokens,
            self.config.image_token_id.unwrap_or(258880) as u32,
            &self.cached_paged_image_token_positions,
        )
    }

    fn run_paged_turn(&mut self, args: &mut WholeTurnArgs<'_>) -> Result<TurnOutput> {
        // The execution plan admits every text turn shape (fresh + delta,
        // sync + streaming) when the adapter is loaded. Both decoders drive
        // the SAME paged lifecycle (prime → prefill → … → epilogue); they
        // differ only in what runs between the first sample and the epilogue.
        debug_assert!(args.plan.use_paged_attention);
        debug_assert!(self.kv_cache_coordinator.is_some());
        debug_assert!(self.paged_text_turn_context.is_empty());
        self.paged_text_turn_context = args.plan.context_media;
        let result = match args.plan.decoder {
            DecoderPlan::Speculative(_) => {
                debug_assert!(args.media.is_empty());
                crate::engine::dspark_turn::run_paged_dspark_turn(self, args)
            }
            DecoderPlan::Autoregressive => crate::engine::paged_turn::run_paged_turn(self, args),
        };
        self.paged_text_turn_context = MediaCapabilities::NONE;
        result
    }

    /// FLAT-lane draft speculative-decode whole-turn path. The execution
    /// plan admits this handler only after request opt-in, with the loaded
    /// draft the flat lane serves (the assistant), flat KV, and text-only
    /// input; DSpark declares paged support and reaches `run_paged_turn`.
    fn run_speculative_turn(&mut self, args: &mut WholeTurnArgs<'_>) -> Result<TurnOutput> {
        debug_assert!(args.media.is_empty());
        debug_assert!(matches!(
            args.plan.decoder,
            DecoderPlan::Speculative(SpeculativeKind::DraftModel)
        ));
        self.flat_draft_chat_turn(args)
    }

    fn run_multimodal_turn(&mut self, args: &mut WholeTurnArgs<'_>) -> Result<TurnOutput> {
        debug_assert!(!args.media.is_empty());
        self.multimodal_chat_turn(args)
    }
}
