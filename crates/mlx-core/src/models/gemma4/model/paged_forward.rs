//! The paged prefill/decode engine on Gemma4Inner: turn preparation, the chunked prefill layer loops (text and VLM), the decode steps, and the sliding-only prefill path.

use super::*;

impl Gemma4Inner {
    pub(super) fn prepare_gemma4_paged_turn(
        &mut self,
        trace_label: &str,
        tokens: &[u32],
        reuse_cache: bool,
        total_budget: u32,
        seq_id: u32,
        cache_salt: u64,
        _trace_enabled: bool,
    ) -> Result<Gemma4PagedTurnPreparation> {
        let carries_image_lineage = gemma4_carries_image_lineage(
            self.paged_text_turn_context,
            self.cached_image_key,
            &self.cached_paged_image_token_positions,
            &self.cached_token_history,
            tokens,
        );
        if reuse_cache
            && self
                .kv_cache_coordinator
                .as_ref()
                .is_some_and(|coordinator| coordinator.can_continue_all(seq_id, tokens))
        {
            let cached_prefix_len = self
                .kv_cache_coordinator
                .as_mut()
                .ok_or_else(|| {
                    Error::from_reason("Gemma4 hybrid continuation lost its KV coordinator")
                })?
                .continue_turn_all(seq_id, tokens, total_budget)
                .map_err(Error::from_reason)?;
            return Ok(Gemma4PagedTurnPreparation {
                cached_prefix_len,
                suffix_len: total_budget.saturating_sub(cached_prefix_len),
                sliding_primed_prefix_len: cached_prefix_len,
            });
        }
        // A media-derived prefix can only be continued in-place: replaying its
        // placeholder token IDs cannot recreate the scattered image features.
        if carries_image_lineage {
            return Err(Error::from_reason(format!(
                "{}{trace_label} lost the live hybrid media prefix",
                engine::IMAGE_CHANGE_RESTART_PREFIX
            )));
        }
        // Ownerless whole-turn and scheduler-owned text requests share the
        // same grouped prefix transaction. A full-group hot/SSD candidate is
        // usable only after every sliding group installs the validated
        // companion at that exact boundary; otherwise the helper resets all
        // groups to zero.
        let cached_prefix_len =
            self.prepare_scheduled_text_request(seq_id, tokens, cache_salt, reuse_cache)?;
        self.cached_image_key = None;
        self.cached_paged_image_token_positions.clear();
        Ok(Gemma4PagedTurnPreparation {
            cached_prefix_len,
            suffix_len: total_budget.saturating_sub(cached_prefix_len),
            sliding_primed_prefix_len: cached_prefix_len,
        })
    }

    /// [`project_paged_hidden_rows`] over this model's head — the shared
    /// projection tail of the paged forwards. `last_only == true` is the
    /// prefill tails' last-token projection; `last_only == false` is the
    /// all-rows verify shape.
    pub(crate) fn project_paged_hidden(
        &self,
        hidden_states: &MxArray,
        last_only: bool,
    ) -> Result<MxArray> {
        project_paged_hidden_rows(
            hidden_states,
            &self.final_norm,
            &self.embed_tokens,
            &self.lm_head,
            self.embed_weight_t.as_ref(),
            &self.config,
            last_only,
        )
    }

    /// Fuse one tapped chunk's residual hiddens and append one draft
    /// context row per token at `position_base`.
    fn append_dspark_prefill_rows(
        &self,
        draft_tap: Option<&mut Gemma4DsparkPrefillTap<'_>>,
        captured: &[MxArray],
        position_base: i32,
    ) -> Result<()> {
        let Some(draft_tap) = draft_tap else {
            return Ok(());
        };
        let draft = self.dspark_draft().ok_or_else(|| {
            Error::from_reason("Gemma4 tapped paged prefill: no DSpark draft model loaded")
        })?;
        let fused = draft.fuse_context(captured)?;
        draft_tap.ctx.append(draft, &fused, position_base)
    }

    /// Run a paged-attention prefill over the full prompt, dispatching
    /// per-layer between the adapter (global layers) and the existing
    /// flat path (sliding layers).
    ///
    /// `full_tokens` is the entire prompt (sliding layers re-prefill
    /// from token 0). `suffix_tokens` is the new portion beyond the
    /// paged prefix-cache hit (used by `record_tokens` +
    /// `update_keys_values` for global layers). `cached_prefix_len`
    /// is the paged-cache hit length.
    ///
    /// Returns the last position's logits squeezed to `[vocab]`.
    ///
    /// ## Prefill split (parity with the flat path)
    ///
    /// The final prompt token runs its OWN single-token forward, mirroring
    /// the flat path's `prefill_body_gemma4` split, so the K/V reduction
    /// order at the prefill→decode boundary matches between flat and paged.
    /// Merging it into the body lets BF16 SDPA drift flip argmax to a
    /// zero-embedding `<unused>` token: the `<turn|>` stop is missed and the
    /// decoder falls into the all-zero-input `mean(V)` → `id+1` cascade.
    ///
    /// `draft_tap` is `Some` on a DSpark speculative turn: each chunk's
    /// residual hiddens are cloned, fused, and appended to the draft's
    /// context before the chunk's graph is dropped, so the full prompt's
    /// tapped hiddens are never held at once. The target-side walk is the
    /// same either way — the tap only clones — which is what makes a
    /// speculative turn's prompt K/V bit-identical to the plain paged turn's.
    pub(crate) fn run_paged_prefill_chunk(
        &mut self,
        full_tokens: &[u32],
        suffix_tokens: &[u32],
        cached_prefix_len: u32,
        sliding_primed_prefix_len: u32,
        _cache_salt: u64,
        mut draft_tap: Option<&mut Gemma4DsparkPrefillTap<'_>>,
    ) -> Result<MxArray> {
        if suffix_tokens.is_empty() {
            return Err(Error::from_reason(
                "run_paged_prefill_chunk called with empty suffix",
            ));
        }
        if sliding_primed_prefix_len != cached_prefix_len {
            return Err(Error::from_reason(
                "Gemma4 hybrid paged prefill requires the same live boundary in every group",
            ));
        }
        let suffix_start = usize::try_from(cached_prefix_len).unwrap_or(usize::MAX);
        if full_tokens.get(suffix_start..) != Some(suffix_tokens) {
            return Err(Error::from_reason(
                "Gemma4 hybrid paged prefill suffix does not extend the live group boundary",
            ));
        }

        let layer_kinds = self.compute_layer_kinds()?;
        let final_index = suffix_tokens.len() - 1;
        let chunk_size = usize::try_from(gemma4_paged_prefill_group_max_chunk())
            .unwrap_or(usize::MAX)
            .max(1);
        let mut position = 0usize;
        while position < final_index {
            if self
                .turn_cancel
                .as_ref()
                .is_some_and(|flag| flag.load(Ordering::Relaxed))
            {
                return Err(Error::from_reason("prefill cancelled"));
            }
            let end = position.saturating_add(chunk_size).min(final_index);
            let chunk = &suffix_tokens[position..end];
            self.kv_cache_coordinator
                .as_mut()
                .ok_or_else(|| Error::from_reason("Gemma4 hybrid KV coordinator missing"))?
                .record_tokens_all(self.active_paged_seq, chunk)
                .map_err(Error::from_reason)?;
            let absolute_position = suffix_start.saturating_add(position);
            let mut tap = draft_tap
                .as_deref()
                .map(|draft_tap| DsparkTap::new(draft_tap.layer_ids));
            let hidden = self.run_paged_prefill_layer_loop(
                chunk,
                absolute_position as u32,
                absolute_position as u32,
                &layer_kinds,
                tap.as_mut(),
            )?;
            if let Some(tap) = tap {
                self.append_dspark_prefill_rows(
                    draft_tap.as_deref_mut(),
                    &tap.captured,
                    absolute_position as i32,
                )?;
            }
            hidden.eval();
            let coordinator = self
                .kv_cache_coordinator
                .as_mut()
                .ok_or_else(|| Error::from_reason("Gemma4 hybrid KV coordinator missing"))?;
            coordinator
                .eval_pending_pool_writes_all()
                .map_err(Error::from_reason)?;
            self.remember_grouped_sliding_cold_checkpoint(self.active_paged_seq)?;
            let coordinator = self
                .kv_cache_coordinator
                .as_mut()
                .ok_or_else(|| Error::from_reason("Gemma4 hybrid KV coordinator missing"))?;
            coordinator
                .prune_sliding_all(self.active_paged_seq)
                .map_err(Error::from_reason)?;
            crate::array::clear_cache();
            position = end;
        }

        let final_token = &suffix_tokens[final_index..];
        self.kv_cache_coordinator
            .as_mut()
            .ok_or_else(|| Error::from_reason("Gemma4 hybrid KV coordinator missing"))?
            .record_tokens_all(self.active_paged_seq, final_token)
            .map_err(Error::from_reason)?;
        let final_position = suffix_start.saturating_add(final_index);
        let mut tap = draft_tap
            .as_deref()
            .map(|draft_tap| DsparkTap::new(draft_tap.layer_ids));
        let hidden_states = self.run_paged_prefill_layer_loop(
            final_token,
            final_position as u32,
            final_position as u32,
            &layer_kinds,
            tap.as_mut(),
        )?;
        if let Some(tap) = tap {
            self.append_dspark_prefill_rows(draft_tap, &tap.captured, final_position as i32)?;
        }
        self.kv_cache_coordinator
            .as_mut()
            .ok_or_else(|| Error::from_reason("Gemma4 hybrid KV coordinator missing"))?
            .eval_pending_pool_writes_all()
            .map_err(Error::from_reason)?;
        self.remember_grouped_sliding_cold_checkpoint(self.active_paged_seq)?;

        self.project_paged_hidden(&hidden_states, true)
    }

    /// Execute one scheduler-pinned Gemma 4 prefill slice.
    ///
    /// Non-final slices are ordinary multi-token body chunks and do not run
    /// the vocabulary projection. The final slice keeps Gemma's load-bearing
    /// last-token split: its body is written first, then the prompt's final
    /// token is forwarded alone and projected. With pinned boundaries equal
    /// to the configured paged chunk size this is the same numerical shape as
    /// [`Self::run_paged_prefill_chunk`], merely interruptible between slices.
    pub(super) fn run_scheduled_paged_prefill_slice(
        &mut self,
        seq_id: u32,
        tokens: &[u32],
        first_logical_position: u32,
        final_prompt_slice: bool,
    ) -> Result<Option<MxArray>> {
        if tokens.is_empty() {
            return Err(Error::from_reason(
                "Gemma4 scheduled prefill slice must be non-empty",
            ));
        }
        if self
            .turn_cancel
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::Relaxed))
        {
            return Err(Error::from_reason("prefill cancelled"));
        }
        let layer_kinds = self.compute_layer_kinds()?;
        let body_len = if final_prompt_slice {
            tokens.len().saturating_sub(1)
        } else {
            tokens.len()
        };
        if body_len != 0 {
            let body = &tokens[..body_len];
            self.kv_cache_coordinator
                .as_mut()
                .ok_or_else(|| Error::from_reason("Gemma4 hybrid KV coordinator missing"))?
                .record_tokens_all(seq_id, body)
                .map_err(Error::from_reason)?;
            let hidden = self.run_paged_prefill_layer_loop(
                body,
                first_logical_position,
                first_logical_position,
                &layer_kinds,
                None,
            )?;
            hidden.eval();
            let coordinator = self
                .kv_cache_coordinator
                .as_mut()
                .ok_or_else(|| Error::from_reason("Gemma4 hybrid KV coordinator missing"))?;
            coordinator
                .eval_pending_pool_writes_all()
                .map_err(Error::from_reason)?;
            self.remember_grouped_sliding_cold_checkpoint(seq_id)?;
            let coordinator = self
                .kv_cache_coordinator
                .as_mut()
                .ok_or_else(|| Error::from_reason("Gemma4 hybrid KV coordinator missing"))?;
            coordinator
                .prune_sliding_all(seq_id)
                .map_err(Error::from_reason)?;
        }
        if !final_prompt_slice {
            crate::array::clear_cache();
            return Ok(None);
        }

        let final_token = &tokens[body_len..];
        let final_position = first_logical_position.saturating_add(body_len as u32);
        self.kv_cache_coordinator
            .as_mut()
            .ok_or_else(|| Error::from_reason("Gemma4 hybrid KV coordinator missing"))?
            .record_tokens_all(seq_id, final_token)
            .map_err(Error::from_reason)?;
        let hidden = self.run_paged_prefill_layer_loop(
            final_token,
            final_position,
            final_position,
            &layer_kinds,
            None,
        )?;
        self.kv_cache_coordinator
            .as_mut()
            .ok_or_else(|| Error::from_reason("Gemma4 hybrid KV coordinator missing"))?
            .eval_pending_pool_writes_all()
            .map_err(Error::from_reason)?;
        Ok(Some(self.project_paged_hidden(&hidden, true)?))
    }

    /// One forward pass through the embed → PLE → layer-loop pipeline
    /// for a single contiguous chunk of tokens. Returns the chunk's
    /// post-final-layer hidden state (NO final norm / lm_head / softcap
    /// — the caller decides whether to apply those).
    ///
    /// `chunk_tokens` is the slice being processed THIS call.
    /// `first_logical_position` is the absolute logical position of
    /// `chunk_tokens[0]` in the request (used as the RoPE offset and
    /// the slot-mapping anchor). `cached_prefix_len_for_chunk` is the
    /// number of K/V tokens already in the paged pool BEFORE this
    /// chunk's writes — when this is > 0 global attention adaptively chooses
    /// graph-native pool gather + SDPA or compact varlen PagedAttention while
    /// retaining the same physical paged storage. `layer_kinds` is the
    /// per-layer routing classification (Sliding / GlobalPaged /
    /// SharedOnGlobal / SharedOnSliding).
    ///
    /// Caller must have already called `record_tokens(chunk_tokens)`
    /// on the paged adapter so `update_keys_values`'s alignment check
    /// (`first_logical_position == current_token_count - chunk.len()`)
    /// passes.
    pub(crate) fn run_paged_prefill_layer_loop(
        &mut self,
        chunk_tokens: &[u32],
        first_logical_position: u32,
        cached_prefix_len_for_chunk: u32,
        layer_kinds: &[Gemma4LayerKind],
        mut tap: Option<&mut DsparkTap<'_>>,
    ) -> Result<MxArray> {
        let chunk_len = chunk_tokens.len() as u32;
        if chunk_len == 0 {
            return Err(Error::from_reason(
                "run_paged_prefill_layer_loop: chunk_tokens must be non-empty",
            ));
        }
        validate_paged_tap_layer_ids(tap.as_deref(), self.layers.len())?;
        let trace_enabled = inference_trace_enabled();
        let trace_start = trace_enabled.then(std::time::Instant::now);
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] gemma4 paged_prefill_layer_loop_enter first_position={} cached_prefix_for_chunk={} tokens={} layers={}",
                first_logical_position,
                cached_prefix_len_for_chunk,
                chunk_len,
                self.layers.len()
            ));
        }

        let input_ids = MxArray::from_uint32(chunk_tokens, &[1, chunk_len as i64])?;
        let mut hidden_states = self.embed_tokens.forward(&input_ids)?;
        // Apply Gemma4 embedding scaling (sqrt(hidden_size)).
        hidden_states = hidden_states.mul_scalar((self.config.hidden_size as f64).sqrt())?;

        // Compute PLE (per-layer embeddings) for the chunk's tokens.
        // Mirrors `forward_body`: PLE feeds an additive residual inside
        // every layer's `apply_ffn_ple_scalar` tail. For Gemma4 E2B/E4B
        // this is load-bearing — dropping it produces nonsense logits
        // because each layer is missing a critical residual
        // contribution. Sliding-only re-prefill of any cached prefix
        // doesn't propagate PLE through the global layers we'll touch
        // here (their stored K/V already accounts for it).
        let projected_ple: Option<MxArray> = if let Some(ref ple) = self.ple {
            let pre_layer_h = hidden_states.clone();
            Some(compute_ple(
                &input_ids,
                &pre_layer_h,
                ple,
                chunk_len as i64,
            )?)
        } else {
            None
        };

        // Build sliding masks against the bounded rotating-cache attention view,
        // not the absolute prompt offset. This mirrors mlx-lm's
        // RotatingKVCache.make_mask behavior and avoids huge long-context masks.
        let seq_len = chunk_len as i64;
        let sliding_offset = first_logical_position as i32;
        let sliding_window = self.config.sliding_window as i64;
        let sliding_mask_offset =
            sliding_mask_offset_for_chunk(seq_len, sliding_offset, sliding_window);
        if trace_enabled && (sliding_offset > 0 || sliding_mask_offset.is_some()) {
            write_inference_trace(format_args!(
                "[MLX_TRACE] gemma4 paged_prefill_sliding_mask seq_len={} cache_offset={} mask_offset={} window={} explicit_mask={}",
                seq_len,
                sliding_offset,
                sliding_mask_offset.unwrap_or(0),
                sliding_window,
                sliding_mask_offset.is_some()
            ));
        }
        let sliding_mask = sliding_mask_offset
            .map(|offset| create_sliding_mask(seq_len, offset, sliding_window))
            .transpose()?;

        let num_layers = self.layers.len();

        #[allow(clippy::needless_range_loop)]
        for layer_idx in 0..num_layers {
            crate::models::gemma4::diagnostic::set_layer(layer_idx);
            let kind = layer_kinds[layer_idx];
            let layer_trace_start = trace_enabled.then(std::time::Instant::now);
            if trace_enabled {
                write_inference_trace(format_args!(
                    "[MLX_TRACE] gemma4 paged_prefill_layer_start layer={} kind={:?} first_position={} cached_prefix_for_chunk={} tokens={}",
                    layer_idx, kind, first_logical_position, cached_prefix_len_for_chunk, chunk_len
                ));
            }
            let layer: &Gemma4DecoderLayer = unsafe {
                let ptr = self.layers.as_ptr().add(layer_idx);
                &*ptr
            };
            let mask: Option<&MxArray> = if kind.is_sliding() {
                sliding_mask.as_ref()
            } else {
                None
            };

            let adapter = self
                .kv_cache_coordinator
                .as_mut()
                .ok_or_else(|| {
                    Error::from_reason(
                        "run_paged_prefill_layer_loop: paged_adapter dropped mid-forward",
                    )
                })?
                .adapter_mut(kind.group_id())
                .map_err(Error::from_reason)?;

            // Build SharedKvInputs for shared layer kinds.
            let shared_inputs = match kind {
                Gemma4LayerKind::SharedOnGlobal { .. }
                | Gemma4LayerKind::SharedOnSliding { .. } => {
                    // Anchor's pool currently holds
                    // `cached_prefix_len_for_chunk + chunk_len` tokens
                    // for this layer (the anchor wrote its part of
                    // this chunk earlier in the same loop).
                    let total_ctx = cached_prefix_len_for_chunk + chunk_len;
                    Some(crate::models::gemma4::decoder_layer::SharedKvInputs {
                        cache_offset: first_logical_position as i32,
                        total_ctx,
                    })
                }
                _ => None,
            };

            // Slice the per-layer PLE input ([B, T, num_layers, ple_dim] →
            // [B, T, ple_dim]). Mirrors `forward_body`'s per-layer slice.
            let ple_input = projected_ple.as_ref().map(|p| {
                p.slice_axis(2, layer_idx as i64, layer_idx as i64 + 1)
                    .and_then(|s| s.squeeze(Some(&[2])))
            });
            let ple_input_ref = match &ple_input {
                Some(Ok(arr)) => Some(arr),
                _ => None,
            };

            let next_hidden_states = layer.forward_paged_or_flat(
                &hidden_states,
                kind,
                adapter,
                first_logical_position,
                cached_prefix_len_for_chunk,
                /* is_prefill */ true,
                mask,
                None,
                ple_input_ref,
                false,
                shared_inputs,
            )?;
            if trace_enabled {
                write_inference_trace(format_args!(
                    "[MLX_TRACE] gemma4 paged_prefill_layer_done layer={} kind={:?} elapsed_ms={:.1}",
                    layer_idx,
                    kind,
                    layer_trace_start.map(elapsed_ms).unwrap_or(0.0)
                ));
            }
            hidden_states = next_hidden_states;

            // Residual-stream hidden of layer `layer_idx` (post residual add,
            // pre final-norm) — the same capture point as `forward_body`'s
            // flat tap; the compute graph is otherwise unchanged.
            if let Some(t) = tap.as_deref_mut()
                && t.layer_ids.contains(&layer_idx)
            {
                t.captured.push(hidden_states.clone());
            }

            // Smooth the prefill memory peak: every K layers, materialize the
            // residual stream so MLX can release the upstream graph nodes
            // (embedding + every prior layer's attention/MLP/PLE intermediates)
            // from the cache pool. Without this the in-flight lazy graph
            // accumulates on long contexts before the post-prefill sync fires.
            // Cadence is `MLX_PAGED_PREFILL_EVAL_INTERVAL` (default 8).
            crate::array::maybe_eval_clear_for_paged_prefill_layer(layer_idx, &hidden_states)?;
        }

        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] gemma4 paged_prefill_layer_loop_exit first_position={} tokens={} elapsed_ms={:.1}",
                first_logical_position,
                chunk_len,
                trace_start.map(elapsed_ms).unwrap_or(0.0)
            ));
        }
        Ok(hidden_states)
    }

    /// Vision variant of [`Self::run_paged_prefill_layer_loop`]: drives one
    /// contiguous chunk of the merged image+text embeddings through the hybrid
    /// paged dispatch (global/sliding → their cache groups, KV-shared → the
    /// anchor group's physical slot).
    ///
    /// Identical layer routing to the text loop, with two image-aware seams:
    ///   * the residual stream is seeded from the supplied `chunk_embeds`
    ///     (the `masked_scatter` output for this chunk, ALREADY scaled by
    ///     `sqrt(hidden_size)` by the caller) instead of
    ///     `embed_tokens.forward(token_ids)`;
    ///   * PLE per-layer embeddings zero the image-token positions in
    ///     `chunk_token_ids` before `compute_ple`, because the image positions
    ///     carry vision features in the residual, not token PLE residuals.
    ///
    /// `chunk_token_ids` is the expanded token slice for this chunk (drives the
    /// PLE image mask and the sliding-mask sequence length).
    /// `chunk_embeds` is `[1, chunk_len, hidden]`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_paged_vlm_prefill_layer_loop(
        &mut self,
        chunk_token_ids: &[u32],
        chunk_embeds: &MxArray,
        first_logical_position: u32,
        cached_prefix_len_for_chunk: u32,
        layer_kinds: &[Gemma4LayerKind],
        overlay_type_ids: Option<&MxArray>,
        mut tap: Option<&mut DsparkTap<'_>>,
    ) -> Result<MxArray> {
        let chunk_len = chunk_token_ids.len() as u32;
        if chunk_len == 0 {
            return Err(Error::from_reason(
                "run_paged_vlm_prefill_layer_loop: chunk_token_ids must be non-empty",
            ));
        }
        validate_paged_tap_layer_ids(tap.as_deref(), self.layers.len())?;

        let input_ids = MxArray::from_uint32(chunk_token_ids, &[1, chunk_len as i64])?;
        let mut hidden_states = chunk_embeds.clone();

        // PLE over media-masked token ids: image AND audio positions hold
        // projected media features (not token embeddings), so their PLE
        // residual must be zero.
        let projected_ple: Option<MxArray> = if let Some(ref ple) = self.ple {
            let image_token_id = self.config.image_token_id.unwrap_or(258880);
            let image_token = MxArray::scalar_int(image_token_id)?;
            let mut media_mask = input_ids.equal(&image_token)?;
            if let Some(audio_token_id) = self.config.audio_token_id {
                let audio_token = MxArray::scalar_int(audio_token_id)?;
                let audio_mask = input_ids.equal(&audio_token)?;
                media_mask = media_mask.logical_or(&audio_mask)?;
            }
            let zero = MxArray::scalar_int(0)?;
            // Media positions (image and audio) are excluded from the PLE
            // residual because their embedding is the projected media feature,
            // not a learned token.
            let masked_ids = media_mask.where_(&zero, &input_ids)?;
            let pre_layer_h = hidden_states.clone();
            Some(compute_ple(
                &masked_ids,
                &pre_layer_h,
                ple,
                chunk_len as i64,
            )?)
        } else {
            None
        };

        // Sliding mask against the bounded rotating-cache attention view —
        // identical derivation to the text paged loop.
        let seq_len = chunk_len as i64;
        let sliding_offset = first_logical_position as i32;
        let sliding_window = self.config.sliding_window as i64;
        let sliding_mask_offset =
            sliding_mask_offset_for_chunk(seq_len, sliding_offset, sliding_window);
        let mut sliding_mask = sliding_mask_offset
            .map(|offset| create_sliding_mask(seq_len, offset, sliding_window))
            .transpose()?;

        // Unified-vision bidirectional overlay. Active only on the cold-start
        // single-chunk prefill (`overlay_type_ids` is Some and
        // `cached_prefix_len_for_chunk == 0`), where every mask key dimension
        // equals `seq_len`. Both layer types get an EXPLICIT materialized
        // boolean keep-mask (true=keep): the global layer's normal None/causal
        // fast path and the sliding layer's possibly-None window mask are
        // replaced by `base | same_image_block`.
        let overlay_active = overlay_type_ids.filter(|_| cached_prefix_len_for_chunk == 0);
        let overlay_global_mask: Option<MxArray> = if let Some(type_ids) = overlay_active {
            let base = create_causal_mask(seq_len as i32, None, None)?;
            let base = base.reshape(&[1, 1, seq_len, seq_len])?;
            Some(apply_bidirectional_vision_overlay(&base, type_ids)?)
        } else {
            None
        };
        if let Some(type_ids) = overlay_active {
            let base = create_causal_mask(seq_len as i32, None, Some(sliding_window as i32))?;
            let base = base.reshape(&[1, 1, seq_len, seq_len])?;
            sliding_mask = Some(apply_bidirectional_vision_overlay(&base, type_ids)?);
        }

        let num_layers = self.layers.len();

        #[allow(clippy::needless_range_loop)]
        for layer_idx in 0..num_layers {
            crate::models::gemma4::diagnostic::set_layer(layer_idx);
            let kind = layer_kinds[layer_idx];
            let layer: &Gemma4DecoderLayer = unsafe {
                let ptr = self.layers.as_ptr().add(layer_idx);
                &*ptr
            };
            let mask: Option<&MxArray> = if kind.is_sliding() {
                sliding_mask.as_ref()
            } else {
                // Global/full layers normally pass None (internal causal). When
                // the overlay is active they receive the explicit bidirectional
                // keep-mask, which `forward_paged` applies in the fresh-prefill
                // branch.
                overlay_global_mask.as_ref()
            };

            let adapter = self
                .kv_cache_coordinator
                .as_mut()
                .ok_or_else(|| {
                    Error::from_reason(
                        "run_paged_vlm_prefill_layer_loop: paged_adapter dropped mid-forward",
                    )
                })?
                .adapter_mut(kind.group_id())
                .map_err(Error::from_reason)?;

            let shared_inputs = match kind {
                Gemma4LayerKind::SharedOnGlobal { .. }
                | Gemma4LayerKind::SharedOnSliding { .. } => {
                    let total_ctx = cached_prefix_len_for_chunk + chunk_len;
                    Some(crate::models::gemma4::decoder_layer::SharedKvInputs {
                        cache_offset: first_logical_position as i32,
                        total_ctx,
                    })
                }
                _ => None,
            };

            let ple_input = projected_ple.as_ref().map(|p| {
                p.slice_axis(2, layer_idx as i64, layer_idx as i64 + 1)
                    .and_then(|s| s.squeeze(Some(&[2])))
            });
            let ple_input_ref = match &ple_input {
                Some(Ok(arr)) => Some(arr),
                _ => None,
            };

            let next_hidden_states = layer.forward_paged_or_flat(
                &hidden_states,
                kind,
                adapter,
                first_logical_position,
                cached_prefix_len_for_chunk,
                /* is_prefill */ true,
                mask,
                None,
                ple_input_ref,
                false,
                shared_inputs,
            )?;
            hidden_states = next_hidden_states;

            // Residual-stream hidden of layer `layer_idx` (post residual add,
            // pre final-norm) — the same capture point as `forward_body`'s
            // flat tap; the compute graph is otherwise unchanged.
            if let Some(t) = tap.as_deref_mut()
                && t.layer_ids.contains(&layer_idx)
            {
                t.captured.push(hidden_states.clone());
            }

            crate::array::maybe_eval_clear_for_paged_prefill_layer(layer_idx, &hidden_states)?;
        }

        Ok(hidden_states)
    }

    /// Cold-start paged prefill over the merged image+text embeddings.
    ///
    /// Single-shot only: both grouped adapters hold zero tokens, so
    /// `cached_prefix_len == 0` and there is no
    /// prefix-cache restore. Splits the merged-embedding body prefill from a
    /// last-token `forward_inner`, a split that is load-bearing — see
    /// [`Self::run_paged_prefill_chunk`] for why
    /// the final prompt token must run through the cache-hit branch separately
    /// (BF16 SDPA drift otherwise flips argmax to a zero-embedding `<unused>`
    /// token and the `<turn|>` stop is missed).
    ///
    /// `expanded_tokens` is the full `BOI + N×image + EOI` expanded sequence.
    /// `inputs_embeds` is `[1, prompt_len, hidden]`, ALREADY scaled by
    /// `sqrt(hidden_size)` and with vision features scattered at the image
    /// positions. Returns the final token's logits squeezed to `[vocab]`.
    pub(super) fn run_paged_vlm_prefill(
        &mut self,
        expanded_tokens: &[u32],
        suffix_embeds: &MxArray,
        layer_kinds: &[Gemma4LayerKind],
        cached_prefix_len: u32,
        _extra_keys_per_block: &[Vec<u64>],
        image_token_positions: &[(u32, u64)],
        _cache_salt: u64,
    ) -> Result<MxArray> {
        if expanded_tokens.is_empty() {
            return Err(Error::from_reason(
                "run_paged_vlm_prefill called with empty prompt",
            ));
        }
        let prompt_len = expanded_tokens.len() as u32;
        if cached_prefix_len >= prompt_len {
            return Err(Error::from_reason(format!(
                "run_paged_vlm_prefill requires a non-empty suffix: cached_prefix_len={cached_prefix_len}, prompt_len={prompt_len}"
            )));
        }
        let suffix_len = prompt_len - cached_prefix_len;
        if suffix_embeds.shape_at(1)? != suffix_len as i64 {
            return Err(Error::from_reason(format!(
                "run_paged_vlm_prefill suffix embedding length {} does not match suffix token length {suffix_len}",
                suffix_embeds.shape_at(1)?
            )));
        }

        // Sliding-window state covers the whole cached prefix by construction
        // on this path: `resolve_vlm_paged_prefix` either kept a candidate
        // whose `sliding_prefix_exact` was true, or restarted the turn cold
        // (`cached_prefix_len == 0`). Discharge the adapter's auxiliary-state
        // obligation before the first `record_tokens` of the turn — the same
        // ack `run_paged_prefill_chunk` makes for the text path.
        if let Some(adapter) = self.kv_cache_coordinator.as_mut() {
            adapter
                .confirm_aux_prefix_primed(cached_prefix_len)
                .map_err(Error::from_reason)?;
        }

        crate::models::gemma4::diagnostic::set_path("paged");
        crate::models::gemma4::diagnostic::set_step(-1);

        // Unified-vision bidirectional overlay gate: is_unified +
        // use_bidirectional_attention=="vision" + image tokens present + no audio
        // tokens + prefill (seq_len>1). Mixed image+audio prompts stay causal
        // (audio wins) — see `vision_overlay_active`. When active, the whole image
        // block must live in ONE prefill chunk so bidirectionality is not severed
        // by chunk boundaries.
        let image_token_id = self.config.image_token_id.unwrap_or(258880) as u32;
        let audio_token_id = self.config.audio_token_id.unwrap_or(258881) as u32;
        let has_image = expanded_tokens.contains(&image_token_id);
        let has_audio = expanded_tokens.contains(&audio_token_id);
        let overlay_full_type_ids: Option<MxArray> = if cached_prefix_len == 0
            && crate::models::gemma4::vision_mask::vision_overlay_active(
                self.config.is_unified,
                self.config.use_bidirectional_attention.as_deref() == Some("vision"),
                has_image,
                has_audio,
                prompt_len as usize,
            ) {
            Some(
                crate::models::gemma4::vision_mask::build_image_token_type_ids(
                    expanded_tokens,
                    image_token_id,
                )?,
            )
        } else {
            None
        };
        let overlay_active = overlay_full_type_ids.is_some();
        // The overlay only reaches GlobalPaged/Sliding layers. KV-shared layers
        // (SharedOnGlobal/SharedOnSliding) run forward_paged_shared, which takes
        // no mask and would silently stay causal — a half-applied overlay across
        // the stack. The 12B unified checkpoint has num_kv_shared_layers==0, so
        // this never fires; fail loudly rather than corrupt attention if a shared
        // unified checkpoint is ever loaded.
        if overlay_active && self.config.num_kv_shared_layers.is_some_and(|n| n > 0) {
            return Err(Error::from_reason(
                "Gemma4 unified-vision bidirectional overlay is unsupported with KV-shared layers \
                 (num_kv_shared_layers > 0): forward_paged_shared does not carry the overlay mask",
            ));
        }

        let block_size = self
            .kv_cache_coordinator
            .as_ref()
            .ok_or_else(|| Error::from_reason("run_paged_vlm_prefill: paged_adapter is None"))?
            .block_size();
        let prompt_checkpoint_boundary = prompt_len
            .saturating_sub(1)
            .checked_div(block_size)
            .map(|blocks| blocks.saturating_mul(block_size))
            .unwrap_or(0);
        let first_image_position = image_token_positions.first().map(|(position, _)| *position);
        let last_image_exclusive = image_token_positions
            .last()
            .map(|(position, _)| position.saturating_add(1));
        // Preserve the established prefill chunk boundaries.
        let leading_text_checkpoint_boundary = if overlay_active {
            0
        } else {
            first_image_position
                .and_then(|position| position.checked_div(block_size))
                .map(|blocks| blocks.saturating_mul(block_size))
                .unwrap_or(0)
        };

        // Pass 1: uncached suffix except its final token. Pass 2: the final
        // token alone, preserving the prefill/decode reduction boundary.
        let pass1_end = prompt_len - 1;
        let mut pass1_position = cached_prefix_len;
        if pass1_position < pass1_end {
            let configured_chunk_size = crate::array::paged_prefill_chunk_size();
            while pass1_position < pass1_end {
                // Cooperative-cancel checkpoint: abort at the chunk
                // boundary. Both VLM cores fail closed on Err via
                // `invalidate_gemma4_hybrid_session` — the request is
                // released, never finalized.
                if self
                    .turn_cancel
                    .as_ref()
                    .is_some_and(|f| f.load(Ordering::Relaxed))
                {
                    return Err(Error::from_reason("prefill cancelled"));
                }
                // The first unified chunk must include the complete image
                // overlay. Boundaries before the end of that span are ignored;
                // otherwise the later chunk would receive no overlay ids and
                // silently run half of the image causally.
                let chunk_end = gemma4_vlm_prefill_chunk_end(
                    pass1_position,
                    pass1_end,
                    configured_chunk_size,
                    overlay_active,
                    leading_text_checkpoint_boundary,
                    prompt_checkpoint_boundary,
                    last_image_exclusive,
                );
                let chunk_start = pass1_position as usize;
                let chunk_end_usize = chunk_end as usize;
                let chunk_tokens = &expanded_tokens[chunk_start..chunk_end_usize];
                let relative_start = pass1_position - cached_prefix_len;
                let relative_end = chunk_end - cached_prefix_len;
                let chunk_embeds =
                    suffix_embeds.slice_axis(1, relative_start as i64, relative_end as i64)?;
                let chunk_type_ids: Option<MxArray> = match &overlay_full_type_ids {
                    Some(ids) if pass1_position == 0 => {
                        Some(ids.slice_axis(1, 0, chunk_end as i64)?)
                    }
                    _ => None,
                };
                {
                    let coordinator = self.kv_cache_coordinator.as_mut().ok_or_else(|| {
                        Error::from_reason("run_paged_vlm_prefill: paged_adapter is None")
                    })?;
                    coordinator
                        .record_tokens_all(self.active_paged_seq, chunk_tokens)
                        .map_err(Error::from_reason)?;
                }
                let _hidden = self.run_paged_vlm_prefill_layer_loop(
                    chunk_tokens,
                    &chunk_embeds,
                    pass1_position,
                    pass1_position,
                    layer_kinds,
                    chunk_type_ids.as_ref(),
                    None,
                )?;
                _hidden.eval();
                if let Some(coordinator) = self.kv_cache_coordinator.as_mut() {
                    coordinator
                        .eval_pending_pool_writes_all()
                        .map_err(Error::from_reason)?;
                    coordinator
                        .prune_sliding_all(self.active_paged_seq)
                        .map_err(Error::from_reason)?;
                }
                if let Some(caches) = self.caches.as_ref() {
                    eval_gemma4_caches(caches)?;
                }
                crate::array::clear_cache();
                pass1_position = chunk_end;
            }
        }

        // Pass 2: the FINAL token (length 1).
        let last_idx = (prompt_len - 1) as usize;
        let pass2_tokens = &expanded_tokens[last_idx..];
        let pass2_relative_idx = last_idx - cached_prefix_len as usize;
        let pass2_embeds = suffix_embeds.slice_axis(
            1,
            pass2_relative_idx as i64,
            pass2_relative_idx as i64 + 1,
        )?;
        {
            let coordinator = self.kv_cache_coordinator.as_mut().ok_or_else(|| {
                Error::from_reason("run_paged_vlm_prefill: paged_adapter is None")
            })?;
            coordinator
                .record_tokens_all(self.active_paged_seq, pass2_tokens)
                .map_err(Error::from_reason)?;
        }
        let mut hidden_states = self.run_paged_vlm_prefill_layer_loop(
            pass2_tokens,
            &pass2_embeds,
            pass1_position,
            pass1_position,
            layer_kinds,
            // Pass 2 is the single final token (seq_len==1); the overlay never
            // applies to a single-token query.
            None,
            None,
        )?;
        if let Some(coordinator) = self.kv_cache_coordinator.as_mut() {
            coordinator
                .eval_pending_pool_writes_all()
                .map_err(Error::from_reason)?;
        }

        hidden_states = self.final_norm.forward(&hidden_states)?;
        crate::models::gemma4::diagnostic::dump_norm(0, "post_final_norm", &hidden_states, None);
        let logits = if let Some(ref head) = self.lm_head {
            head.forward(&hidden_states)?
        } else if self.embed_tokens.is_packed_quantized() {
            // Packed tied lm_head: project through the quantized matmul without
            // materializing the dense table.
            self.embed_tokens.as_linear(&hidden_states)?
        } else if let Some(ref w_t) = self.embed_weight_t {
            hidden_states.matmul(w_t)?
        } else {
            let weight = self.embed_tokens.get_weight();
            let weight_t = weight.transpose(Some(&[1, 0]))?;
            hidden_states.matmul(&weight_t)?
        };
        crate::models::gemma4::diagnostic::dump_logits("pre_softcap", &logits);
        let logits = if let Some(cap) = self.config.final_logit_softcapping {
            let cap_arr = MxArray::scalar_float_like(cap, &logits)?;
            let handle = unsafe { mlx_sys::mlx_logit_softcap(logits.handle.0, cap_arr.handle.0) };
            let capped = MxArray::from_handle(handle, "logit_softcap")?;
            crate::models::gemma4::diagnostic::dump_logits("post_softcap", &capped);
            capped
        } else {
            crate::models::gemma4::diagnostic::dump_logits("post_softcap", &logits);
            logits
        };

        let last_seq_len = logits.shape_at(1)?;
        logits
            .slice_axis(1, last_seq_len - 1, last_seq_len)?
            .squeeze(Some(&[0, 1]))
    }

    /// Run one paged decode step for a scheduler-owned sequence.
    pub(crate) fn run_paged_decode_step_for(
        &mut self,
        seq_id: u32,
        token_id: u32,
    ) -> Result<MxArray> {
        let first_logical_position = {
            let coordinator = self.kv_cache_coordinator.as_mut().ok_or_else(|| {
                Error::from_reason("run_paged_decode_step: paged_adapter is None")
            })?;
            coordinator
                .activate_request_all(seq_id)
                .map_err(Error::from_reason)?;
            coordinator.full_adapter().current_token_count()
        };
        {
            let coordinator = self.kv_cache_coordinator.as_mut().ok_or_else(|| {
                Error::from_reason("run_paged_decode_step: paged_adapter dropped")
            })?;
            coordinator
                .record_tokens_all(seq_id, &[token_id])
                .map_err(Error::from_reason)?;
        }

        let layer_kinds = self.compute_layer_kinds()?;

        let input_ids = MxArray::from_uint32(&[token_id], &[1, 1])?;
        let mut hidden_states = self.embed_tokens.forward(&input_ids)?;
        hidden_states = hidden_states.mul_scalar((self.config.hidden_size as f64).sqrt())?;

        // Compute PLE for the single decode token. Same load-bearing
        // residual contribution as the prefill path — see the comment in
        // `run_paged_prefill_chunk` for why dropping this destroys logits
        // on Gemma4 E2B/E4B.
        let projected_ple_step: Option<MxArray> = if let Some(ref ple) = self.ple {
            let pre_layer_h = hidden_states.clone();
            Some(compute_ple(&input_ids, &pre_layer_h, ple, 1)?)
        } else {
            None
        };

        let num_layers = self.layers.len();
        crate::models::gemma4::diagnostic::set_path("paged");
        #[allow(clippy::needless_range_loop)]
        for layer_idx in 0..num_layers {
            crate::models::gemma4::diagnostic::set_layer(layer_idx);
            let kind = layer_kinds[layer_idx];
            let layer: &Gemma4DecoderLayer = unsafe {
                let ptr = self.layers.as_ptr().add(layer_idx);
                &*ptr
            };
            let adapter = self
                .kv_cache_coordinator
                .as_mut()
                .ok_or_else(|| {
                    Error::from_reason("run_paged_decode_step: paged_adapter dropped mid-forward")
                })?
                .adapter_mut(kind.group_id())
                .map_err(Error::from_reason)?;

            let shared_inputs = match kind {
                Gemma4LayerKind::SharedOnGlobal { .. }
                | Gemma4LayerKind::SharedOnSliding { .. } => {
                    // Anchor's slot already has the new token (it ran
                    // its own forward_paged earlier in this loop, which
                    // wrote K/V via update_keys_values). Read full ctx.
                    let total_ctx = first_logical_position + 1;
                    Some(crate::models::gemma4::decoder_layer::SharedKvInputs {
                        cache_offset: first_logical_position as i32,
                        total_ctx,
                    })
                }
                _ => None,
            };

            // Slice the per-layer PLE input ([B, T, num_layers, ple_dim] →
            // [B, T, ple_dim]).
            let ple_input = projected_ple_step.as_ref().map(|p| {
                p.slice_axis(2, layer_idx as i64, layer_idx as i64 + 1)
                    .and_then(|s| s.squeeze(Some(&[2])))
            });
            let ple_input_ref = match &ple_input {
                Some(Ok(arr)) => Some(arr),
                _ => None,
            };

            hidden_states = layer.forward_paged_or_flat(
                &hidden_states,
                kind,
                adapter,
                first_logical_position,
                /* cached_prefix_len */ 0,
                /* is_prefill */ false,
                /* mask */ None,
                None,
                ple_input_ref,
                false,
                shared_inputs,
            )?;
        }

        hidden_states = self.final_norm.forward(&hidden_states)?;
        crate::models::gemma4::diagnostic::dump_norm(0, "post_final_norm", &hidden_states, None);
        let logits = if let Some(ref head) = self.lm_head {
            head.forward(&hidden_states)?
        } else if self.embed_tokens.is_packed_quantized() {
            // Packed tied lm_head: project through the quantized matmul without
            // materializing the dense table.
            self.embed_tokens.as_linear(&hidden_states)?
        } else if let Some(ref w_t) = self.embed_weight_t {
            hidden_states.matmul(w_t)?
        } else {
            let weight = self.embed_tokens.get_weight();
            let weight_t = weight.transpose(Some(&[1, 0]))?;
            hidden_states.matmul(&weight_t)?
        };
        crate::models::gemma4::diagnostic::dump_logits("pre_softcap", &logits);
        let logits = if let Some(cap) = self.config.final_logit_softcapping {
            let cap_arr = MxArray::scalar_float_like(cap, &logits)?;
            let handle = unsafe { mlx_sys::mlx_logit_softcap(logits.handle.0, cap_arr.handle.0) };
            let capped = MxArray::from_handle(handle, "logit_softcap")?;
            crate::models::gemma4::diagnostic::dump_logits("post_softcap", &capped);
            capped
        } else {
            crate::models::gemma4::diagnostic::dump_logits("post_softcap", &logits);
            logits
        };
        Ok(logits)
    }

    /// Run one uniform decode wave for multiple scheduler-owned sequences.
    /// Full and sliding groups advance atomically for every row, then each
    /// transformer layer executes once over `[N,1,H]` with request-specific
    /// RoPE offsets and block tables.
    pub(super) fn run_paged_decode_step_batched(&mut self, rows: &[(u32, u32)]) -> Result<MxArray> {
        if rows.is_empty() {
            return Err(Error::from_reason(
                "run_paged_decode_step_batched requires at least one row",
            ));
        }
        let coordinator = self.kv_cache_coordinator.as_ref().ok_or_else(|| {
            Error::from_reason("run_paged_decode_step_batched: KV coordinator is unavailable")
        })?;
        let mut seen = HashSet::with_capacity(rows.len());
        let mut planned_rows = Vec::with_capacity(rows.len());
        for &(seq_id, _) in rows {
            if !seen.insert(seq_id) {
                return Err(Error::from_reason(format!(
                    "run_paged_decode_step_batched received duplicate sequence {seq_id}"
                )));
            }
            let position = coordinator
                .request_token_count_all(seq_id)
                .map_err(Error::from_reason)?;
            planned_rows.push((seq_id, position));
        }

        let mut recorded = Vec::with_capacity(rows.len());
        for &(seq_id, token_id) in rows {
            let result = self
                .kv_cache_coordinator
                .as_mut()
                .ok_or_else(|| {
                    Error::from_reason("run_paged_decode_step_batched: KV coordinator disappeared")
                })?
                .record_tokens_all(seq_id, &[token_id]);
            if let Err(error) = result {
                for &recorded_seq in recorded.iter().rev() {
                    self.kv_cache_coordinator
                        .as_mut()
                        .ok_or_else(|| {
                            Error::from_reason(
                                "run_paged_decode_step_batched: KV coordinator disappeared during rollback",
                            )
                        })?
                        .rollback_last_tokens_all(recorded_seq, 1)
                        .map_err(|rollback| {
                            Error::from_reason(format!(
                                "Gemma4 batched record failed for sequence {seq_id}: {error}; \
                                 rollback for sequence {recorded_seq} failed: {rollback}"
                            ))
                        })?;
                }
                return Err(Error::from_reason(format!(
                    "Gemma4 batched record failed for sequence {seq_id}: {error}"
                )));
            }
            recorded.push(seq_id);
        }

        let token_ids = rows.iter().map(|&(_, token)| token).collect::<Vec<_>>();
        let batch = rows.len() as i64;
        let input_ids = MxArray::from_uint32(&token_ids, &[batch, 1])?;
        let mut hidden_states = self.embed_tokens.forward(&input_ids)?;
        hidden_states = hidden_states.mul_scalar((self.config.hidden_size as f64).sqrt())?;
        let projected_ple = if let Some(ref ple) = self.ple {
            Some(compute_ple(&input_ids, &hidden_states, ple, 1)?)
        } else {
            None
        };

        let layer_kinds = self.compute_layer_kinds()?;
        for (layer_idx, &kind) in layer_kinds.iter().enumerate() {
            let layer: &Gemma4DecoderLayer = unsafe { &*self.layers.as_ptr().add(layer_idx) };
            let adapter = self
                .kv_cache_coordinator
                .as_mut()
                .ok_or_else(|| {
                    Error::from_reason(
                        "run_paged_decode_step_batched: coordinator dropped mid-forward",
                    )
                })?
                .adapter_mut(kind.group_id())
                .map_err(Error::from_reason)?;
            let ple_input = projected_ple.as_ref().map(|ple| {
                ple.slice_axis(2, layer_idx as i64, layer_idx as i64 + 1)
                    .and_then(|slice| slice.squeeze(Some(&[2])))
            });
            let ple_input = match &ple_input {
                Some(Ok(input)) => Some(input),
                Some(Err(error)) => return Err(Error::from_reason(error.reason.clone())),
                None => None,
            };
            hidden_states = layer.forward_paged_batched(
                &hidden_states,
                kind,
                adapter,
                &planned_rows,
                ple_input,
            )?;
        }
        hidden_states = self.final_norm.forward(&hidden_states)?;
        lm_head_logits(
            &hidden_states,
            &self.embed_tokens,
            &self.lm_head,
            self.embed_weight_t.as_ref(),
            &self.config,
        )
    }

    /// Legacy single-session wrapper. Scheduled turns call the sequence-aware
    /// form directly.
    pub(super) fn run_paged_decode_step(&mut self, token_id: u32) -> Result<MxArray> {
        self.run_paged_decode_step_for(self.active_paged_seq, token_id)
    }

    /// Replay cached prefix tokens to reconstruct the flat sliding caches.
    /// Used to bring sliding-layer state up to the paged cache's
    /// `cached_prefix_len` boundary before the main `run_paged_prefill_chunk`
    /// continues with the suffix.
    ///
    /// Global layers run as read-only Q projections against their existing
    /// paged K/V. That keeps hidden states flowing into later sliding layers
    /// without rebuilding throwaway global K/V for the cached prefix.
    ///
    /// This body publishes ONE checkpoint (at `cached_prefix_len`, by its
    /// caller) and deliberately no [`gemma4_sliding_cold_anchor_rungs`], unlike
    /// `run_paged_prefill_chunk`'s pass-1 loop. Every rung it would cross is
    /// already in the store, because reaching this replay at all means the
    /// prefix below `first_logical_position` was already reconstructed from a
    /// source that published them:
    ///
    /// ```text
    ///   cold_sidecar arm   -> primed == cached, this body does not run
    ///   prefix_checkpoint  -> the store holds that entry AND every shallower
    ///                         rung of the same lineage (Ladder defers them)
    ///   replay arm         -> primed == 0. The pass-1 loop crosses the whole
    ///                         grid itself only when cached_prefix_len == 0 too.
    /// ```
    ///
    /// The only caller is the VLM leading-text replay; grouped text restore
    /// installs the sliding adapters directly and never goes through this flat
    /// replay path.
    pub(super) fn run_sliding_only_prefill(
        &mut self,
        prefix_tokens: &[u32],
        first_logical_position: u32,
        layer_kinds: &[Gemma4LayerKind],
    ) -> Result<()> {
        if prefix_tokens.is_empty() {
            return Ok(());
        }
        let configured_chunk_size = crate::array::paged_prefill_chunk_size();
        let num_query_heads = u32::try_from(self.config.num_attention_heads).map_err(|_| {
            Error::from_reason(format!(
                "Gemma4 sliding restore invalid num_attention_heads={}",
                self.config.num_attention_heads
            ))
        })?;
        let global_head_size =
            u32::try_from(self.config.effective_head_dim(true)).map_err(|_| {
                Error::from_reason(format!(
                    "Gemma4 sliding restore invalid global head_dim={}",
                    self.config.effective_head_dim(true)
                ))
            })?;
        let num_kv_heads = u32::try_from(self.config.effective_kv_heads(true)).map_err(|_| {
            Error::from_reason(format!(
                "Gemma4 sliding restore invalid global num_kv_heads={}",
                self.config.effective_kv_heads(true)
            ))
        })?;
        let route_policy = gemma4_paged_prefill_route_policy();
        let mut chunk_plan = gemma4_paged_prefill_body_chunk_plan(
            configured_chunk_size,
            prefix_tokens.len(),
            first_logical_position,
            num_query_heads,
            num_kv_heads,
            global_head_size,
            route_policy,
        )?;
        gemma4_coalesce_single_token_restore_chunks(&mut chunk_plan);

        let trace_enabled = inference_trace_enabled();
        let total_trace_start = trace_enabled.then(std::time::Instant::now);
        if trace_enabled {
            let first_chunk_size = chunk_plan.first().map(|chunk| chunk.len).unwrap_or(0);
            let min_chunk_size = chunk_plan.iter().map(|chunk| chunk.len).min().unwrap_or(0);
            let max_chunk_size = chunk_plan.iter().map(|chunk| chunk.len).max().unwrap_or(0);
            let aux_caps = chunk_plan
                .iter()
                .filter(|chunk| chunk.capped_by_v2_aux_limit)
                .count();
            write_inference_trace(format_args!(
                "[MLX_TRACE] gemma4 sliding_prefix_restore_start first_position={} prefix_tokens={} chunks={} chunk_size={} min_chunk_size={} max_chunk_size={} configured_chunk_size={} dynamic_v2_aux_caps={} path=paged_global_readonly",
                first_logical_position,
                prefix_tokens.len(),
                chunk_plan.len(),
                first_chunk_size,
                min_chunk_size,
                max_chunk_size,
                configured_chunk_size,
                aux_caps
            ));
        }

        let total_chunks = chunk_plan.len();
        for (chunk_idx, chunk_plan) in chunk_plan.iter().enumerate() {
            // Cooperative-cancel checkpoint (H1b): abort at the chunk
            // boundary. Both callers fail closed on Err — the VLM
            // leading-text replay invalidates the hybrid session, and the
            // paged chunk driver rides `abort_paged_turn`.
            if self
                .turn_cancel
                .as_ref()
                .is_some_and(|f| f.load(Ordering::Relaxed))
            {
                return Err(Error::from_reason("prefill cancelled"));
            }
            let chunk_end = chunk_plan
                .start
                .checked_add(chunk_plan.len)
                .ok_or_else(|| Error::from_reason("Gemma4 sliding restore chunk end overflow"))?;
            let chunk = prefix_tokens
                .get(chunk_plan.start..chunk_end)
                .ok_or_else(|| {
                    Error::from_reason("Gemma4 sliding restore chunk plan out of range")
                })?;
            let chunk_trace_start = trace_enabled.then(std::time::Instant::now);
            if trace_enabled {
                write_inference_trace(format_args!(
                    "[MLX_TRACE] gemma4 sliding_prefix_restore_chunk_start chunk={}/{} first_position={} tokens={} capped_by_v2_aux_limit={}",
                    chunk_idx + 1,
                    total_chunks,
                    chunk_plan.first_position,
                    chunk.len(),
                    chunk_plan.capped_by_v2_aux_limit
                ));
            }

            self.run_sliding_prefix_restore_layer_loop(
                chunk,
                chunk_plan.first_position,
                layer_kinds,
            )?;

            if let Some(caches) = self.caches.as_ref() {
                eval_gemma4_caches(caches)?;
            }
            crate::array::clear_cache();

            if trace_enabled {
                write_inference_trace(format_args!(
                    "[MLX_TRACE] gemma4 sliding_prefix_restore_chunk_done chunk={}/{} next_position={} elapsed_ms={:.1}",
                    chunk_idx + 1,
                    total_chunks,
                    chunk_plan.first_position + chunk.len() as u32,
                    chunk_trace_start.map(elapsed_ms).unwrap_or(0.0)
                ));
            }
        }

        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] gemma4 sliding_prefix_restore_done first_position={} prefix_tokens={} chunks={} elapsed_ms={:.1}",
                first_logical_position,
                prefix_tokens.len(),
                total_chunks,
                total_trace_start.map(elapsed_ms).unwrap_or(0.0)
            ));
        }
        Ok(())
    }

    fn run_sliding_prefix_restore_layer_loop(
        &mut self,
        chunk_tokens: &[u32],
        first_logical_position: u32,
        layer_kinds: &[Gemma4LayerKind],
    ) -> Result<()> {
        let _ = (chunk_tokens, first_logical_position, layer_kinds);
        Err(Error::from_reason(
            "Gemma4 hybrid paged groups do not replay private sliding state; \
             prefix admission must intersect all physical cache groups",
        ))
    }

    // Conversation structure is rendered by the checkpoint's chat template in the
    // shared engine; this family supplies only its terminal token policy.

    /// Resolve the token id for Gemma4's `<turn|>` turn terminator.
    ///
    /// Used as the `eos_token_id` in the session-start path so the
    /// decode loop stops at the model's turn boundary.
    pub(crate) fn turn_end_id(&self) -> Result<u32> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))?;
        let ids = tokenizer.encode_sync("<turn|>", Some(false))?;
        if ids.is_empty() {
            return Err(Error::from_reason(
                "Tokenizer encoded <turn|> to empty id vector",
            ));
        }
        if ids.len() != 1 {
            return Err(Error::from_reason(format!(
                "Tokenizer encoded <turn|> to {} tokens; expected 1",
                ids.len()
            )));
        }
        Ok(ids[0])
    }

    /// Multimodal whole-turn dispatch for the engine's
    /// [`ChatBackend::run_multimodal_turn`] handler. A complete structured
    /// continuation may include historical media, so the paged cores verify
    /// the rendered token prefix and media identity before deciding whether
    /// to reuse live state or cold-prefill.
    ///
    /// Image turns run ONLY on the block-paged KV backend. A model with
    /// no paged adapter (explicit `use_block_paged_cache: false`, a
    /// non-Metal build, or paged init failure) has no vision path and
    /// returns an error instead of silently falling back.
    pub(super) fn multimodal_chat_turn(
        &mut self,
        args: &mut WholeTurnArgs<'_>,
    ) -> Result<TurnOutput> {
        if self.kv_cache_coordinator.is_none() {
            return Err(Error::from_reason(
                "gemma4 image turns require the block-paged KV backend; the model was loaded \
                 without a paged adapter (use_block_paged_cache=false, non-Metal build, or paged \
                 init failed)",
            ));
        }
        let tokenizer = args.tokenizer.clone();
        match (args.sink, args.cancelled) {
            (Some(sink), Some(cancelled)) => {
                self.vision_paged_turn_stream_core(
                    args.tokens,
                    args.media.images,
                    args.media.audio,
                    &tokenizer,
                    args.config,
                    args.eos_id,
                    sink,
                    cancelled,
                )?;
                Ok(TurnOutput::Streamed)
            }
            _ => {
                let result = self.vision_paged_turn_sync_core(
                    args.tokens,
                    args.media.images,
                    args.media.audio,
                    &tokenizer,
                    args.config,
                    args.eos_id,
                )?;
                Ok(TurnOutput::Complete(Box::new(result)))
            }
        }
    }
}
