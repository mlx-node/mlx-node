//! The VLM/audio turn: token preparation, multimodal embed assembly, the paged vision prefix, the sync/stream vision cores, and the image/audio placeholder helpers they call.

use super::*;

impl Gemma4Inner {
    /// Decode + resize + patch raw image bytes and expand the rendered
    /// prompt's per-image `<|image|>` placeholders.
    ///
    /// The engine session core owns message-side image extraction
    /// (`engine::session::extract_images_from_messages`) and prompt
    /// rendering; the raw bytes arrive via [`WholeTurnArgs::media`].
    /// The "no vision support" rejection surfaces from INSIDE the vision
    /// turn (after render).
    fn prepare_vision_tokens(
        &self,
        rendered_tokens: &[u32],
        raw_images: &[Vec<u8>],
    ) -> Result<(
        Vec<u32>,
        Vec<ProcessedGemma4Image>,
        Option<u64>,
        Vec<(u32, u64)>,
    )> {
        let ip = self.image_processor.as_ref().ok_or_else(|| {
            Error::from_reason(
                "Images provided but model has no vision support (no vision_config in config.json)",
            )
        })?;
        let mut processed_images = Vec::with_capacity(raw_images.len());
        for bytes in raw_images {
            processed_images.push(ip.process_bytes(bytes)?);
        }

        // Compute the image cache key BEFORE the prefill so it can be
        // recorded on `self.cached_image_key` after the decode loop.
        // Session callers inspect this field to decide whether a
        // session-continue delta is allowed (text-only) or requires
        // a fresh `chat_session_start`.
        let (combined_image_key, per_image_hashes) = engine::compute_image_cache_keys(raw_images);
        let new_image_key = Some(combined_image_key);

        // Expand image tokens. Gemma4 uses: <|image>  (BOI) +
        // <|image|> × num_soft_tokens + <image|> (EOI). The chat template
        // inserts a single <|image|> per image; we expand it here.
        let image_token_id = self.config.image_token_id.unwrap_or(258880) as u32;
        let boi_token_id = self.config.boi_token_id.unwrap_or(255999) as u32;
        let eoi_token_id = self.config.eoi_token_id.unwrap_or(258882) as u32;
        let expanded = expand_image_tokens(
            rendered_tokens,
            &processed_images,
            image_token_id,
            boi_token_id,
            eoi_token_id,
        )?;

        let per_image_token_counts = processed_images
            .iter()
            .map(|image| image.num_soft_tokens as usize)
            .collect::<Vec<_>>();
        let image_token_positions = engine::map_expanded_image_token_positions(
            &expanded,
            image_token_id,
            &per_image_token_counts,
            &per_image_hashes,
        )
        .map_err(Error::from_reason)?;

        Ok((
            expanded,
            processed_images,
            new_image_key,
            image_token_positions,
        ))
    }

    /// Decode raw (encoded) audio bytes and expand the rendered prompt's
    /// per-clip `<|audio|>` placeholders into `boa + audio×n_frames + eoa`
    /// spans. The audio counterpart of [`Self::prepare_vision_tokens`].
    ///
    /// Each clip is decoded (`decode_wav_to_pcm`) into a mono 16 kHz f32
    /// waveform and framed (`frames_from_pcm`) into `[n_frames, 640]` raw
    /// windows; the per-clip frame counts drive `expand_audio_tokens`. All
    /// clips' frames are concatenated (axis 0) into a single
    /// `[total_frames, 640]` tensor so the merge scatter feeds them in order.
    /// `tokens` is the (possibly image-expanded) token stream; the audio
    /// expansion runs on top of it, leaving image spans untouched.
    fn prepare_audio_tokens(
        &self,
        tokens: &[u32],
        raw_audio: &[Vec<u8>],
    ) -> Result<(Vec<u32>, MxArray, Option<u64>)> {
        let spt = self.config.audio_samples_per_token.unwrap_or(640) as usize;
        let audio_token_id = self.config.audio_token_id.unwrap_or(258881) as u32;
        let boa_token_id = self.config.boa_token_id.unwrap_or(256000) as u32;
        let eoa_token_id = self.config.eoa_token_id.unwrap_or(258883) as u32;

        let mut per_clip_frames: Vec<MxArray> = Vec::with_capacity(raw_audio.len());
        let mut n_frames_per_clip: Vec<usize> = Vec::with_capacity(raw_audio.len());
        for bytes in raw_audio {
            let pcm = crate::models::gemma4::audio_processor::decode_wav_to_pcm(bytes)?;
            let frames = crate::models::gemma4::audio_processor::frames_from_pcm(&pcm, spt)?;
            let n = frames.shape_at(0)? as usize;
            n_frames_per_clip.push(n);
            per_clip_frames.push(frames);
        }

        let audio_frames = if per_clip_frames.len() == 1 {
            per_clip_frames.remove(0)
        } else {
            let refs: Vec<&MxArray> = per_clip_frames.iter().collect();
            MxArray::concatenate_many(refs, Some(0))?
        };

        let expanded = crate::models::gemma4::audio_processor::expand_audio_tokens(
            tokens,
            &n_frames_per_clip,
            audio_token_id,
            boa_token_id,
            eoa_token_id,
        )?;

        // Audio uses the same byte-identity cache key as images so an
        // audio-change cold-restarts the session server-side.
        let new_audio_key = Some(engine::compute_image_cache_key(raw_audio));

        Ok((expanded, audio_frames, new_audio_key))
    }

    /// Build the merged multimodal+text input embeddings for a prefill.
    ///
    /// Scatters image features (`@image_token_id`) AND audio features
    /// (`@audio_token_id`) into the SAME `sqrt(hidden)`-scaled text stream
    /// via chained `masked_scatter`s. Image-only turns skip the audio scatter
    /// (the image scatter math matches the prior vision-only prefill exactly);
    /// audio-only turns skip the image scatter. Returns `None` only when
    /// neither modality contributes features (text-only fallback).
    fn build_gemma4_multimodal_embeds(
        &self,
        prompt: &MxArray,
        processed_images: &[ProcessedGemma4Image],
        audio_frames: Option<&MxArray>,
    ) -> Result<Option<MxArray>> {
        let has_image_features = !processed_images.is_empty() && self.embed_vision.is_some();
        let has_audio_features = audio_frames.is_some() && self.embed_audio.is_some();
        if !has_image_features && !has_audio_features {
            return Ok(None);
        }

        // Base scaled text stream (built once; both scatters write into it).
        let text_embeds = self.embed_tokens.forward(prompt)?;
        let mut merged = text_embeds.mul_scalar((self.config.hidden_size as f64).sqrt())?;
        let embed_dtype = merged.dtype()?;

        // Image scatter @ image_token_id.
        if has_image_features {
            let ev = self.embed_vision.as_ref().ok_or_else(|| {
                Error::from_reason("Gemma4 image features require a vision projector")
            })?;
            let image_token_id = self.config.image_token_id.unwrap_or(258880);
            let mut all_features: Vec<MxArray> = Vec::new();
            for proc in processed_images {
                let features = if let Some(vt) = self.vision_tower.as_ref() {
                    vt.forward(&proc.pixel_values)?
                } else if let Some(ve) = self.unified_vision_embedder.as_ref() {
                    let positions = proc.position_ids.as_ref().ok_or_else(|| {
                        Error::from_reason(
                            "Unified vision embedder requires per-patch position ids, but none \
                             were produced by the image processor.",
                        )
                    })?;
                    ve.forward(&proc.pixel_values, positions)?.expand_dims(0)?
                } else {
                    return Err(Error::from_reason(
                        "Image features requested but no vision tower / unified embedder present",
                    ));
                };
                all_features.push(ev.forward(&features)?);
            }
            let image_features = if all_features.len() == 1 {
                all_features.remove(0)
            } else {
                let refs: Vec<&MxArray> = all_features.iter().collect();
                MxArray::concatenate_many(refs, Some(1))?
            };
            let image_features = image_features.astype(embed_dtype)?;

            let image_token = MxArray::scalar_int(image_token_id)?;
            let image_mask = prompt.equal(&image_token)?;
            let mask_count_arr = image_mask.astype(DType::Int32)?.sum(None, None)?;
            mask_count_arr.eval();
            let mask_count = mask_count_arr.item_at_int32(0)? as i64;
            let feature_count = image_features.shape_at(1)?;
            if mask_count != feature_count {
                return Err(Error::new(
                    Status::GenericFailure,
                    format!(
                        "Image token count ({mask_count}) does not match vision feature count ({feature_count}). \
                         Check that image token expansion produced the correct number of tokens."
                    ),
                ));
            }
            let image_mask_expanded = image_mask.expand_dims(-1)?.broadcast_to(&merged.shape()?)?;
            merged = masked_scatter(&merged, &image_mask_expanded, &image_features)?;
        }

        // Audio scatter @ audio_token_id (CAUSAL; audio features unscaled).
        if has_audio_features {
            let ea = self.embed_audio.as_ref().ok_or_else(|| {
                Error::from_reason("Gemma4 audio features require an audio projector")
            })?;
            let audio_token_id = self.config.audio_token_id.unwrap_or(258881);
            let audio_frames = audio_frames.ok_or_else(|| {
                Error::from_reason("Gemma4 audio features require prepared audio frames")
            })?;
            let audio_features = ea.forward(audio_frames)?.astype(embed_dtype)?;

            let audio_token = MxArray::scalar_int(audio_token_id)?;
            let audio_mask = prompt.equal(&audio_token)?;
            let mask_count_arr = audio_mask.astype(DType::Int32)?.sum(None, None)?;
            mask_count_arr.eval();
            let mask_count = mask_count_arr.item_at_int32(0)? as i64;
            let feature_count = audio_features.shape_at(0)?;
            if mask_count != feature_count {
                return Err(Error::new(
                    Status::GenericFailure,
                    format!(
                        "Audio token count ({mask_count}) does not match audio frame count ({feature_count}). \
                         Check that audio token expansion produced the correct number of frames."
                    ),
                ));
            }
            // Zero-frame audio has no scatter targets; leave the stream as-is
            // (a `masked_scatter` over an empty source would divide by zero).
            if feature_count > 0 {
                let audio_mask_expanded =
                    audio_mask.expand_dims(-1)?.broadcast_to(&merged.shape()?)?;
                merged = masked_scatter(&merged, &audio_mask_expanded, &audio_features)?;
            }
        }

        Ok(Some(merged))
    }

    /// Build only the embeddings the effective paged suffix will forward.
    /// When an image-aware hit already covers the complete image span, the
    /// suffix is text-only: avoid running SigLIP/unified vision entirely and
    /// embed just that suffix. Otherwise build the faithful full multimodal
    /// stream once and slice it at the effective cache boundary.
    fn build_gemma4_multimodal_suffix_embeds(
        &self,
        prompt: &MxArray,
        processed_images: &[ProcessedGemma4Image],
        audio_frames: Option<&MxArray>,
        cached_prefix_len: u32,
        last_image_exclusive: Option<u32>,
    ) -> Result<MxArray> {
        let prompt_len = u32::try_from(prompt.shape_at(1)?)
            .map_err(|_| Error::from_reason("Gemma4 prompt length exceeds u32"))?;
        if cached_prefix_len >= prompt_len {
            return Err(Error::from_reason(format!(
                "Gemma4 multimodal suffix is empty: cached_prefix_len={cached_prefix_len}, prompt_len={prompt_len}"
            )));
        }

        let image_span_fully_cached = audio_frames.is_none()
            && !processed_images.is_empty()
            && last_image_exclusive
                .is_some_and(|last_image_exclusive| cached_prefix_len >= last_image_exclusive);
        if image_span_fully_cached {
            let last_image_exclusive = last_image_exclusive
                .ok_or_else(|| Error::from_reason("Gemma4 cached image span has no endpoint"))?;
            tracing::info!(
                target: "mlx_core::inference",
                event = "vlm_vision_tower_skip",
                model = "gemma4",
                cached_prefix_tokens = cached_prefix_len,
                last_image_exclusive,
                suffix_tokens = prompt_len - cached_prefix_len,
                "skipping Gemma4 vision tower because the image span is fully cached"
            );
            if inference_trace_enabled() {
                write_inference_trace(format_args!(
                    "[MLX_TRACE] gemma4 vlm_vision_tower_skip cached_prefix_tokens={} last_image_exclusive={} suffix_tokens={}",
                    cached_prefix_len,
                    last_image_exclusive,
                    prompt_len - cached_prefix_len,
                ));
            }
            let suffix = prompt.slice_axis(1, cached_prefix_len as i64, prompt_len as i64)?;
            return self
                .embed_tokens
                .forward(&suffix)?
                .mul_scalar((self.config.hidden_size as f64).sqrt());
        }

        let merged =
            match self.build_gemma4_multimodal_embeds(prompt, processed_images, audio_frames)? {
                Some(merged) => merged,
                None => self
                    .embed_tokens
                    .forward(prompt)?
                    .mul_scalar((self.config.hidden_size as f64).sqrt())?,
            };
        merged.slice_axis(1, cached_prefix_len as i64, prompt_len as i64)
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_gemma4_multimodal_paged_turn(
        &mut self,
        tokens: &[u32],
        prompt: &MxArray,
        processed_images: &[ProcessedGemma4Image],
        audio_frames: Option<&MxArray>,
        new_image_key: Option<u64>,
        new_audio_key: Option<u64>,
        image_token_positions: &[(u32, u64)],
        reuse_cache: bool,
        cache_salt: u64,
    ) -> Result<Gemma4VlmTurnPreparation> {
        let layer_kinds = self.compute_layer_kinds()?;
        let total_budget = u32::try_from(tokens.len())
            .map_err(|_| Error::from_reason("Gemma4 multimodal prompt exceeds u32"))?;
        let block_size = self
            .kv_cache_coordinator
            .as_ref()
            .ok_or_else(|| {
                Error::from_reason("prepare_gemma4_multimodal_paged_turn: paged_adapter is None")
            })?
            .block_size();
        let image_only = new_image_key.is_some() && new_audio_key.is_none();
        let extra_keys_per_block = engine::build_paged_extra_keys(
            tokens.len(),
            block_size,
            if image_only {
                image_token_positions
            } else {
                &[]
            },
        );
        let last_image_exclusive = image_token_positions
            .last()
            .map(|(position, _)| position.saturating_add(1));

        let cached_prefix_len = if image_only {
            let image_token_id = self.config.image_token_id.unwrap_or(258880) as u32;
            let overlay_active = crate::models::gemma4::vision_mask::vision_overlay_active(
                self.config.is_unified,
                self.config.use_bidirectional_attention.as_deref() == Some("vision"),
                !image_token_positions.is_empty(),
                false,
                tokens.len(),
            );
            let allow_live_continue = reuse_cache
                && self.media_session_continuable
                && self.cached_image_key == new_image_key
                && self.cached_audio_key.is_none()
                && self.cached_paged_image_token_positions == image_token_positions;
            self.media_session_continuable = false;
            let resolution = self.prepare_gemma4_vlm_paged_prefix(
                tokens,
                total_budget,
                block_size,
                &extra_keys_per_block,
                image_token_positions,
                reuse_cache,
                allow_live_continue,
                cache_salt,
                if overlay_active {
                    last_image_exclusive
                } else {
                    None
                },
            )?;
            // The image placeholder is load-bearing for the image-only branch;
            // keep this check close to planning so a malformed expansion cannot
            // accidentally take the text-only tower-skip path.
            if !tokens.contains(&image_token_id) {
                self.invalidate_gemma4_hybrid_session(
                    "VLM image metadata had no expanded image placeholder",
                );
                return Err(Error::from_reason(
                    "Gemma4 image prompt contains no expanded image tokens",
                ));
            }
            resolution.effective_plan.cached_prefix_len
        } else {
            // Audio and mixed-media identity is not represented in the paged
            // block chain yet. Keep that path deliberately cold and do not
            // publish reusable prefix entries from its finalizer.
            self.media_session_continuable = false;
            let seq_id = self.active_paged_seq;
            let cold_plan = match self.kv_cache_coordinator.as_mut() {
                Some(coordinator) => coordinator
                    .prepare_scheduled_request(seq_id, tokens)
                    .map_err(Error::from_reason),
                None => Err(Error::from_reason(
                    "prepare_gemma4_multimodal_paged_turn: paged_adapter is None",
                )),
            };
            if let Err(error) = cold_plan {
                self.invalidate_gemma4_hybrid_session(
                    "audio/mixed VLM cold paged preparation failure",
                );
                return Err(error);
            }
            self.caches = None;
            self.cached_token_history.clear();
            self.cached_image_key = None;
            self.cached_audio_key = None;
            self.cached_paged_image_token_positions.clear();
            self.media_session_context = MediaCapabilities::NONE;
            self.paged_text_turn_context = MediaCapabilities::NONE;
            0
        };

        let suffix_embeds = match self.build_gemma4_multimodal_suffix_embeds(
            prompt,
            processed_images,
            audio_frames,
            cached_prefix_len,
            last_image_exclusive,
        ) {
            Ok(embeds) => embeds,
            Err(error) => {
                self.invalidate_gemma4_hybrid_session("VLM suffix embedding preparation failure");
                return Err(error);
            }
        };

        Ok(Gemma4VlmTurnPreparation {
            cached_prefix_len,
            suffix_embeds,
            layer_kinds,
            extra_keys_per_block,
        })
    }

    /// Prepare the merged multimodal prompt for a paged prefill: expand audio
    /// placeholders (when audio present) then image placeholders (when images
    /// present) on the rendered token stream, and decode/frame the audio.
    ///
    /// Each expansion replaces only the placeholder emitted by the checkpoint
    /// template. Missing or extra placeholders fail closed.
    ///
    /// Returns `(tokens, processed_images, audio_frames, new_image_key,
    /// new_audio_key, image_token_positions)`. Image-only turns never touch the audio path and leave
    /// `audio_frames`/`new_audio_key` as `None` (byte-identical to the old
    /// vision-only flow); audio-only turns never run the image processor and
    /// leave `processed_images` empty + `new_image_key` `None`.
    #[allow(clippy::type_complexity)]
    fn prepare_multimodal_tokens(
        &self,
        rendered_tokens: &[u32],
        raw_images: &[Vec<u8>],
        raw_audio: &[Vec<u8>],
    ) -> Result<(
        Vec<u32>,
        Vec<ProcessedGemma4Image>,
        Option<MxArray>,
        Option<u64>,
        Option<u64>,
        Vec<(u32, u64)>,
    )> {
        // Audio expansion first (only when audio present — keeps image-only
        // turns off the audio path and leaves `new_audio_key` None).
        let mut audio_frames: Option<MxArray> = None;
        let mut new_audio_key: Option<u64> = None;
        let tokens_after_audio = if raw_audio.is_empty() {
            rendered_tokens.to_vec()
        } else {
            let (expanded, frames, audio_key) =
                self.prepare_audio_tokens(rendered_tokens, raw_audio)?;
            audio_frames = Some(frames);
            new_audio_key = audio_key;
            expanded
        };

        // Image expansion only touches `<|image|>` ids, so audio spans are
        // inert to it.
        let (tokens, processed_images, new_image_key, image_token_positions) =
            if raw_images.is_empty() {
                (tokens_after_audio, Vec::new(), None, Vec::new())
            } else {
                self.prepare_vision_tokens(&tokens_after_audio, raw_images)?
            };

        Ok((
            tokens,
            processed_images,
            audio_frames,
            new_image_key,
            new_audio_key,
            image_token_positions,
        ))
    }

    /// Terminal media-state finalize shared by both vision cores (sync +
    /// stream), so the two stay byte-identical. Resolves the session into
    /// exactly ONE of two states, never partial:
    ///
    /// - **Continuable** (when `media_continuable` — currently a pure image
    ///   turn, including the unified bidirectional-vision image — AND
    ///   `reuse_cache`, AND every hybrid group finalizes successfully): both
    ///   full- and sliding-attention paged requests remain live at the same
    ///   absolute cursor. The next full-history text continuation forwards only
    ///   its suffix, so no media position is re-embedded from a raw
    ///   `<|image|>`/`<|audio|>` id.
    /// - **Non-continuable** (`reuse_cache=false` or any grouped finalize/live
    ///   check fails): release every group, keep history + media keys live
    ///   (marker stays false), and make the next continuation fail closed
    ///   instead of replaying placeholder ids as media embeddings.
    ///
    /// ## Why all-group liveness is the faithfulness gate
    /// KV-shared alias slots (E2B's `SharedOnSliding` and `SharedOnGlobal`)
    /// intentionally own no storage; their physical anchors carry the state
    /// for both layers. Every physical full/sliding group must therefore agree
    /// on the request cursor before the media lineage can remain continuable.
    /// A warm media→text continue is only numerically faithful when the media
    /// positions' sliding K/V can be reused IN PLACE: a text token's true
    /// embedding IS `embed_tokens.forward(id)` (replay-safe), but a media
    /// position's is a scattered SigLIP/audio feature that replay CANNOT rebuild
    /// from the raw special-token id. So the marker is armed ONLY after every
    /// physical group finalizes and remains live. Missing/misaligned anchors
    /// fail closed.
    ///
    /// ## R1 sliding-offset reconciliation (the length-finish materialize)
    /// The vision decode loop never forwards the final sampled token, so after
    /// the loop the live (non-shared) sliding caches AND the global paged KV sit
    /// at offset `prefill_len + G - 1`. The drop-last history rule yields
    /// `cached_token_history.len() == prefill_len + G - 1` on
    /// stop/repetition/cancelled (offsets MATCH) but `prefill_len + G` on a
    /// `"length"` finish (one short). On the continuable+`"length"` path we
    /// forward that final token once via `run_paged_decode_step` — exactly what
    /// the text path's `materialize_final` does (`paged_turn.rs` length gate →
    /// `Gemma4PagedDecode::materialize_final` → `run_paged_decode_step`) —
    /// advancing both caches to `prefill_len + G` so the kept-live global KV
    /// content-addresses against the saved history for the next delta's live
    /// restore. (Verified byte-exact by the non-unified-image warm==cold golden.)
    #[allow(clippy::too_many_arguments)]
    fn finalize_vision_turn_media_state(
        &mut self,
        expanded_tokens: &[u32],
        generated_tokens: &[u32],
        finish_reason: &str,
        new_image_key: Option<u64>,
        new_audio_key: Option<u64>,
        image_token_positions: &[(u32, u64)],
        media_continuable: bool,
        reuse_cache: bool,
        cache_salt: u64,
    ) -> Result<()> {
        let seq_id = self.active_paged_seq;
        let continuable_eligible = reuse_cache && media_continuable;
        let is_length = finish_reason == "length";

        // Drop-last history (mirrors the non-continuable save the vision cores
        // do today and the text path's `save_paged_history`): keep all tokens on
        // a `"length"` finish, otherwise drop the terminal token.
        let history_tokens: &[u32] = if !is_length && !generated_tokens.is_empty() {
            &generated_tokens[..generated_tokens.len() - 1]
        } else {
            generated_tokens
        };
        let mut full_history = Vec::with_capacity(expanded_tokens.len() + history_tokens.len());
        full_history.extend_from_slice(expanded_tokens);
        full_history.extend_from_slice(history_tokens);

        if continuable_eligible {
            // R1: align the sliding caches with the keep-all history before any
            // checkpoint. On a `"length"` finish the loop left the final token
            // unforwarded (offset == history.len() - 1); forward it now so both
            // the global paged KV and the sliding caches reach history.len().
            if is_length && let Some(&last_token) = generated_tokens.last() {
                // Forwards the token through the paged adapter + sliding caches.
                // A failure here aborts the turn before any state is published
                // (the request is still live; the caller's Err path releases it).
                let _logits = self.run_paged_decode_step(last_token)?;
            }

            let (keep_live_ok, live_for_continue) = match self.kv_cache_coordinator.as_mut() {
                Some(coordinator) => {
                    coordinator
                        .activate_request_all(seq_id)
                        .map_err(Error::from_reason)?;
                    let total = coordinator.full_adapter().request_tokens().len();
                    let bs = coordinator.full_adapter().block_size();
                    let extra = engine::build_paged_extra_keys(total, bs, image_token_positions);
                    let ok = match coordinator.eval_pending_pool_writes_all().and_then(|_| {
                        coordinator.finalize_keep_live_all(seq_id, &extra, cache_salt)
                    }) {
                        Ok(_) => true,
                        Err(error) => {
                            tracing::warn!(
                                target: "mlx_core::gemma4::paged",
                                "Gemma4 image per-block finalize failed: {error}"
                            );
                            false
                        }
                    };
                    (ok, coordinator.is_live_all(seq_id))
                }
                None => (false, false),
            };

            if keep_live_ok {
                // Both physical KV groups now share the same live boundary.
                // Media embeddings remain available in-place without a
                // request-private rotating-cache checkpoint.
                self.cached_token_history = full_history;
                self.publish_media_session_context(new_image_key, new_audio_key);
                self.cached_paged_image_token_positions = image_token_positions.to_vec();
                if live_for_continue {
                    self.media_session_continuable = true;
                    return Ok(());
                }
                if let Some(coordinator) = self.kv_cache_coordinator.as_mut() {
                    let _ = coordinator.release_request_all(seq_id);
                }
                self.media_session_continuable = false;
                return Ok(());
            }
            // keep-live failed: fall through to the non-continuable teardown.
            self.invalidate_gemma4_hybrid_session("VLM per-block finalize failure");
            return Err(Error::from_reason(
                "Gemma4 image paged finalize failed; reusable state was invalidated",
            ));
        }

        // Non-continuable: release every KV group but keep history + media keys
        // so full-history prefix verification forces the next continuation to
        // cold-restart (marker is false).
        if let Some(coordinator) = self.kv_cache_coordinator.as_mut() {
            let _ = coordinator.release_request_all(seq_id);
        }
        self.cached_token_history = full_history;
        self.publish_media_session_context(new_image_key, new_audio_key);
        self.cached_paged_image_token_positions.clear();
        self.media_session_continuable = false;
        Ok(())
    }

    /// Vision (VLM) whole-turn core over the BLOCK-PAGED backend,
    /// non-streaming.
    ///
    /// Shared multimodal prep (`prepare_multimodal_tokens` to expand
    /// `<|image|>` / `<|audio|>` placeholders, `build_gemma4_multimodal_embeds`
    /// to `masked_scatter` image+audio features into the residual) writes
    /// full- and sliding-attention K/V into their respective paged groups.
    ///
    /// Image-only prompts keep every hybrid KV group live at one exact cursor.
    /// Audio and mixed-media prompts remain deliberately cold.
    pub(super) fn vision_paged_turn_sync_core(
        &mut self,
        rendered_tokens: &[u32],
        raw_images: &[Vec<u8>],
        raw_audio: &[Vec<u8>],
        tokenizer: &Arc<Qwen3Tokenizer>,
        config: &ChatConfig,
        eos_token_id: u32,
    ) -> Result<ChatResult> {
        let max_new_tokens = config.max_new_tokens.unwrap_or(2048);
        let (
            tokens,
            processed_images,
            audio_frames,
            new_image_key,
            new_audio_key,
            image_token_positions,
        ) = self.prepare_multimodal_tokens(rendered_tokens, raw_images, raw_audio)?;
        if tokens.is_empty() {
            return Err(Error::from_reason("Empty prompt"));
        }
        let sampling_config = make_sampling_config(config, &self.config);
        let repetition_cutoff = repetition_cutoff_from_config(config);
        let eos_ids = self.config.eos_token_ids.clone();

        let prefill_slice: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let prefill_len = prefill_slice.len();
        let prompt = MxArray::from_int32(&prefill_slice, &[1, prefill_len as i64])?;
        let prompt_token_count = tokens.len();

        let generation_stream = Stream::new(DeviceType::Gpu);
        let _wired_ctx = crate::stream::WiredLimitContext::new(usize::MAX, vec![generation_stream]);

        let generation_start = std::time::Instant::now();
        let reuse_cache = config.reuse_cache.unwrap_or(true);
        let cache_salt = engine::cache_salt(config.cache_salt.as_deref());
        let turn = self.prepare_gemma4_multimodal_paged_turn(
            &tokens,
            &prompt,
            &processed_images,
            audio_frames.as_ref(),
            new_image_key,
            new_audio_key,
            &image_token_positions,
            reuse_cache,
            cache_salt,
        )?;
        let cached_prefix_len = turn.cached_prefix_len;

        // H2: clone the backend-installed per-turn cancel flag before the
        // closure — the decode loop inside borrows `self` mutably.
        let turn_cancel = self.turn_cancel.clone();

        let forward_result = (|| -> Result<(Vec<u32>, String)> {
            let last_logits = {
                let _stream_ctx = StreamContext::new(generation_stream);
                crate::models::gemma4::diagnostic::set_step(-1);
                self.run_paged_vlm_prefill(
                    &tokens,
                    &turn.suffix_embeds,
                    &turn.layer_kinds,
                    cached_prefix_len,
                    &turn.extra_keys_per_block,
                    &image_token_positions,
                    cache_salt,
                )?
            };

            crate::array::synchronize_and_clear_cache();

            let mut y = sample_next_token(&last_logits, sampling_config)?;
            y.eval();

            let mut generated_tokens: Vec<u32> = Vec::new();
            let mut finish_reason = String::from("length");

            for step in 0..max_new_tokens {
                let token_id = y.item_at_int32(0)? as u32;
                generated_tokens.push(token_id);

                // H2 sync cancel poll — the SAME snapshot point as the
                // gemma4 vision paged streaming twin
                // (`vision_paged_turn_stream_core`): right after the token
                // push, BEFORE the EOS check.
                if turn_cancel
                    .as_deref()
                    .is_some_and(|flag| flag.load(Ordering::Relaxed))
                {
                    finish_reason = "cancelled".to_string();
                    break;
                }

                if is_eos_token(token_id, &eos_ids, eos_token_id) {
                    finish_reason = String::from("stop");
                    break;
                }
                if let Some(reason) =
                    check_gemma4_repetition_cutoff(&generated_tokens, repetition_cutoff)
                {
                    finish_reason = reason.to_string();
                    break;
                }
                if step + 1 >= max_new_tokens {
                    break;
                }

                let next_logits = {
                    let _stream_ctx = StreamContext::new(generation_stream);
                    crate::models::gemma4::diagnostic::set_step(step);
                    self.run_paged_decode_step(token_id)?
                };
                let next_logits = next_logits.squeeze(Some(&[1]))?;
                y = sample_next_token(&next_logits, sampling_config)?;
                y.eval();

                crate::array::maybe_clear_cache_for_paged_step(step);
            }

            Ok((generated_tokens, finish_reason))
        })();

        // The Ok branch does NOT release the request here — the media-state
        // finalize decides between keep-live (continuable) and release
        // (non-continuable). The Err branch still releases fully.
        let (generated_tokens, finish_reason) = match forward_result {
            Ok(t) => t,
            Err(e) => {
                self.invalidate_gemma4_hybrid_session("VLM sync forward/decode failure");
                return Err(e);
            }
        };

        let first_token_instant = std::time::Instant::now();

        let raw_text = match tokenizer.decode_sync(&generated_tokens, false) {
            Ok(text) => text,
            Err(error) => {
                self.invalidate_gemma4_hybrid_session("VLM sync tokenizer decode failure");
                return Err(error);
            }
        };

        // Two-state media finalize: keep the global paged KV live + remember a
        // sliding history checkpoint when this is a pure image turn under
        // reuse, so a follow-up text delta
        // warm-continues; otherwise release + keep history/keys so the guard
        // rejects (single-shot, as today). A finalize Err means the live
        // request must be released before returning.
        let media_continuable =
            gemma4_media_continuable(new_image_key.is_some(), new_audio_key.is_some());
        if let Err(e) = self.finalize_vision_turn_media_state(
            &tokens,
            &generated_tokens,
            &finish_reason,
            new_image_key,
            new_audio_key,
            &image_token_positions,
            media_continuable,
            reuse_cache,
            cache_salt,
        ) {
            self.invalidate_gemma4_hybrid_session("VLM sync finalize failure");
            return Err(e);
        }

        let generation_end = std::time::Instant::now();
        let ttft_ms = first_token_instant
            .duration_since(generation_start)
            .as_secs_f64()
            * 1000.0;
        let decode_ms = generation_end
            .duration_since(first_token_instant)
            .as_secs_f64()
            * 1000.0;
        let gen_toks = generated_tokens.len() as f64;

        let performance = Some(crate::profiling::PerformanceMetrics {
            ttft_ms,
            prefill_tokens_per_second: if ttft_ms > 0.0 {
                (prefill_len.saturating_sub(cached_prefix_len as usize)) as f64 / (ttft_ms / 1000.0)
            } else {
                0.0
            },
            decode_tokens_per_second: if decode_ms > 0.0 && gen_toks > 1.0 {
                (gen_toks - 1.0) / (decode_ms / 1000.0)
            } else {
                0.0
            },
            mtp_mean_accepted_tokens: None,
            mtp_mean_accepted_tokens_total: None,
            mtp_acceptance_by_position: None,
            mtp_cycles: None,
            mtp_mean_depth: None,
            profile_phases: None,
        });

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
            finish_reason
        };

        Ok(ChatResult {
            text: parsed.text,
            tool_calls: parsed.tool_calls,
            thinking: parsed.thinking,
            thinking_enabled: engine::resolve_enable_thinking(config).unwrap_or(true),
            num_tokens: generated_tokens.len() as u32,
            prompt_tokens: prompt_token_count as u32,
            reasoning_tokens: 0,
            finish_reason,
            raw_text,
            public_raw_text: None,
            cached_tokens: cached_prefix_len,
            performance,
        })
    }

    /// Streaming twin of [`Self::vision_paged_turn_sync_core`]. Same paged
    /// prefill + decode spine; streams parser segments and emits the terminal
    /// chunk itself.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn vision_paged_turn_stream_core(
        &mut self,
        rendered_tokens: &[u32],
        raw_images: &[Vec<u8>],
        raw_audio: &[Vec<u8>],
        tokenizer: &Arc<Qwen3Tokenizer>,
        config: &ChatConfig,
        eos_token_id: u32,
        sink: &dyn ChunkSink,
        cancelled: &AtomicBool,
    ) -> Result<()> {
        let cb = StreamSender(sink);
        let max_new_tokens = config.max_new_tokens.unwrap_or(2048);
        let (
            tokens,
            processed_images,
            audio_frames,
            new_image_key,
            new_audio_key,
            image_token_positions,
        ) = self.prepare_multimodal_tokens(rendered_tokens, raw_images, raw_audio)?;
        if tokens.is_empty() {
            return Err(Error::from_reason("Empty prompt"));
        }
        let sampling_config = make_sampling_config(config, &self.config);
        let repetition_cutoff = repetition_cutoff_from_config(config);
        let eos_ids = self.config.eos_token_ids.clone();

        let prefill_slice: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let prefill_len = prefill_slice.len();
        let prompt = MxArray::from_int32(&prefill_slice, &[1, prefill_len as i64])?;
        let prompt_token_count = tokens.len();

        let generation_stream = Stream::new(DeviceType::Gpu);
        let _wired_ctx = crate::stream::WiredLimitContext::new(usize::MAX, vec![generation_stream]);

        let generation_start = std::time::Instant::now();
        let reuse_cache = config.reuse_cache.unwrap_or(true);
        let cache_salt = engine::cache_salt(config.cache_salt.as_deref());
        let turn = self.prepare_gemma4_multimodal_paged_turn(
            &tokens,
            &prompt,
            &processed_images,
            audio_frames.as_ref(),
            new_image_key,
            new_audio_key,
            &image_token_positions,
            reuse_cache,
            cache_salt,
        )?;
        let cached_prefix_len = turn.cached_prefix_len;

        let mut decode_stream = tokenizer.inner().decode_stream(false);
        let mut streamed_text_len = 0;
        let starts_in_prompted_channel = self.output_starts_in_reasoning_channel();
        let mut stream_parser =
            crate::models::gemma4::output_parser::Gemma4StreamParser::new_with_open_channel(
                starts_in_prompted_channel,
            );
        let mut stream_dispatch = Gemma4StreamDispatchState::new(starts_in_prompted_channel);

        let forward_result = (|| -> Result<(Vec<u32>, String)> {
            let last_logits = {
                let _stream_ctx = StreamContext::new(generation_stream);
                crate::models::gemma4::diagnostic::set_step(-1);
                self.run_paged_vlm_prefill(
                    &tokens,
                    &turn.suffix_embeds,
                    &turn.layer_kinds,
                    cached_prefix_len,
                    &turn.extra_keys_per_block,
                    &image_token_positions,
                    cache_salt,
                )?
            };

            crate::array::synchronize_and_clear_cache();

            let mut y = sample_next_token(&last_logits, sampling_config)?;
            y.eval();

            let mut generated_tokens: Vec<u32> = Vec::new();
            let mut finish_reason = String::from("length");

            for step in 0..max_new_tokens {
                let token_id = y.item_at_int32(0)? as u32;
                generated_tokens.push(token_id);

                if cancelled.load(Ordering::Relaxed) {
                    finish_reason = "cancelled".to_string();
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
                let segments = stream_parser.feed(&token_text);
                stream_dispatch.dispatch_segments(segments, &cb);

                if is_eos_token(token_id, &eos_ids, eos_token_id) {
                    finish_reason = "stop".to_string();
                    break;
                }
                if let Some(reason) =
                    check_gemma4_repetition_cutoff(&generated_tokens, repetition_cutoff)
                {
                    finish_reason = reason.to_string();
                    break;
                }
                if step + 1 >= max_new_tokens {
                    break;
                }

                let next_logits = {
                    let _stream_ctx = StreamContext::new(generation_stream);
                    crate::models::gemma4::diagnostic::set_step(step);
                    self.run_paged_decode_step(token_id)?
                };
                let next_logits = next_logits.squeeze(Some(&[1]))?;
                y = sample_next_token(&next_logits, sampling_config)?;
                y.eval();

                crate::array::maybe_clear_cache_for_paged_step(step);
            }

            Ok((generated_tokens, finish_reason))
        })();

        // The Ok branch does NOT release the request here — the media-state
        // finalize decides between keep-live (continuable) and release
        // (non-continuable). The Err branch still releases fully.
        let (generated_tokens, finish_reason) = match forward_result {
            Ok(t) => t,
            Err(e) => {
                self.invalidate_gemma4_hybrid_session("VLM stream forward/decode failure");
                return Err(e);
            }
        };

        let first_token_instant = std::time::Instant::now();

        let raw_text = match tokenizer.decode_sync(&generated_tokens, false) {
            Ok(text) => text,
            Err(error) => {
                self.invalidate_gemma4_hybrid_session("VLM stream tokenizer decode failure");
                return Err(error);
            }
        };

        // Flush residual bytes through the stream parser.
        if raw_text.len() > streamed_text_len {
            let residual = raw_text[streamed_text_len..].to_string();
            let mut segments = stream_parser.feed(&residual);
            segments.extend(stream_parser.flush());
            stream_dispatch.dispatch_segments(segments, &cb);
        } else {
            let tail = stream_parser.flush();
            stream_dispatch.dispatch_segments(tail, &cb);
        }
        stream_dispatch.finish(&cb);

        // Two-state media finalize (identical to the sync core via the shared
        // helper): keep every hybrid group live for a continuable pure-causal
        // media turn, else release + keep history/keys so the guard rejects.
        let media_continuable =
            gemma4_media_continuable(new_image_key.is_some(), new_audio_key.is_some());
        if let Err(e) = self.finalize_vision_turn_media_state(
            &tokens,
            &generated_tokens,
            &finish_reason,
            new_image_key,
            new_audio_key,
            &image_token_positions,
            media_continuable,
            reuse_cache,
            cache_salt,
        ) {
            self.invalidate_gemma4_hybrid_session("VLM stream finalize failure");
            return Err(e);
        }

        let generation_end = std::time::Instant::now();
        let ttft_ms = first_token_instant
            .duration_since(generation_start)
            .as_secs_f64()
            * 1000.0;
        let decode_ms = generation_end
            .duration_since(first_token_instant)
            .as_secs_f64()
            * 1000.0;
        let gen_toks = generated_tokens.len() as f64;

        let performance = Some(crate::profiling::PerformanceMetrics {
            ttft_ms,
            prefill_tokens_per_second: if ttft_ms > 0.0 {
                (prefill_len.saturating_sub(cached_prefix_len as usize)) as f64 / (ttft_ms / 1000.0)
            } else {
                0.0
            },
            decode_tokens_per_second: if decode_ms > 0.0 && gen_toks > 1.0 {
                (gen_toks - 1.0) / (decode_ms / 1000.0)
            } else {
                0.0
            },
            mtp_mean_accepted_tokens: None,
            mtp_mean_accepted_tokens_total: None,
            mtp_acceptance_by_position: None,
            mtp_cycles: None,
            mtp_mean_depth: None,
            profile_phases: None,
        });

        let parsed_tool_calls = stream_parser.tool_calls();
        let parsed_thinking = stream_parser.thinking();
        let finish_reason = if parsed_tool_calls.iter().any(|tc| tc.status == "ok") {
            "tool_calls".to_string()
        } else {
            finish_reason
        };

        cb.call(
            Ok(ChatStreamChunk {
                text: String::new(),
                done: true,
                finish_reason: Some(finish_reason),
                tool_calls: Some(parsed_tool_calls),
                thinking: parsed_thinking,
                thinking_enabled: Some(engine::resolve_enable_thinking(config).unwrap_or(true)),
                num_tokens: Some(generated_tokens.len() as u32),
                prompt_tokens: Some(prompt_token_count as u32),
                reasoning_tokens: Some(0),
                raw_text: Some(raw_text),
                public_raw_text: None,
                text_authoritative: Some(false),
                cached_tokens: Some(cached_prefix_len),
                performance,
                is_reasoning: None,
            }),
            ThreadsafeFunctionCallMode::NonBlocking,
        );

        Ok(())
    }

    // =================================================================
    // Block-paged dispatch (paged_turn_sync_core + helpers).
    //
    // Mirrors vLLM's hybrid KV coordinator: global and sliding layers route to
    // separate `PagedKVCacheAdapter` groups, while KV-shared layers alias their
    // anchor's physical group through routes derived from `LayerKVCacheSpec`.
    //
    // Lifecycle (mirrors Qwen3 / LFM2):
    // 1. Adapter cold-start (or warm-continue when previous turn
    //    finalize_turn_keep_live'd a strict-prefix request).
    // 2. Every group resumes the same owner cursor. Sliding adapters keep
    //    absolute logical positions while pruning expired physical blocks.
    // 3. Prefill via `run_paged_prefill_chunk` over the suffix.
    // 4. Decode loop via `run_paged_decode_step`.
    // 5. End-of-turn (success): `finalize_turn_keep_live` so the next
    //    turn's `continue_turn` can build on top of the partial trailing
    //    block's K/V (same partial-block carry trick as Qwen3 / LFM2).
    //
    // Caveats / scope:
    // * ordinary text rows fuse; media and speculative owners use scheduler
    //   barriers because their residual/draft shapes are request-specific;
    // * cross-owner prefix hits fail closed until every hybrid group can name
    //   the same reusable boundary. Same-owner live continuation is supported;
    // * exact live-prefix hits are capped at `prompt_len - 1` so the final
    //   prompt token is always recomputed to produce logits.
    // =================================================================

    pub(super) fn invalidate_gemma4_hybrid_session(&mut self, reason: &'static str) {
        tracing::warn!(
            target: "mlx_core::gemma4::paged",
            reason,
            "invalidating Gemma4 hybrid paged/sliding session"
        );
        if let Some(coordinator) = self.kv_cache_coordinator.as_mut() {
            let _ = coordinator.release_request_all(self.active_paged_seq);
        }
        self.grouped_sliding_cold_checkpoints
            .remove(&self.active_paged_seq);
        self.caches = None;
        self.clear_reuse_state();
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_gemma4_vlm_paged_prefix(
        &mut self,
        tokens: &[u32],
        total_budget: u32,
        block_size: u32,
        extra_keys_per_block: &[Vec<u64>],
        image_token_positions: &[(u32, u64)],
        reuse_cache: bool,
        allow_live_continue: bool,
        cache_salt: u64,
        unified_overlay_last_image_exclusive: Option<u32>,
    ) -> Result<engine::VlmPagedPrefixResolution> {
        let seq_id = self.active_paged_seq;
        if let Some(coordinator) = self.kv_cache_coordinator.as_mut() {
            let cached_prefix_len = if reuse_cache
                && allow_live_continue
                && coordinator.can_continue_all(seq_id, tokens)
            {
                coordinator
                    .continue_turn_all(seq_id, tokens, total_budget)
                    .map_err(Error::from_reason)?
            } else {
                coordinator
                    .prepare_scheduled_request(seq_id, tokens)
                    .map_err(Error::from_reason)?;
                0
            };
            let continued_live_prefix = cached_prefix_len != 0;
            let effective_plan = crate::transformer::paged_kv_cache_adapter::PagedTurnPlan {
                cached_prefix_len,
                continued_live_prefix,
                allocated_blocks: 0,
                cached_blocks: 0,
                total_budget,
                suffix_len: total_budget.saturating_sub(cached_prefix_len),
                reason: if continued_live_prefix {
                    crate::transformer::paged_kv_cache_adapter::PagedTurnPlanReason::ContinuedLivePrefix
                } else {
                    crate::transformer::paged_kv_cache_adapter::PagedTurnPlanReason::FreshReset
                },
            };
            if !continued_live_prefix {
                self.cached_token_history.clear();
                self.cached_image_key = None;
                self.cached_audio_key = None;
                self.media_session_context = MediaCapabilities::NONE;
                self.media_session_continuable = false;
            }
            self.cached_paged_image_token_positions = image_token_positions.to_vec();
            return Ok(engine::VlmPagedPrefixResolution {
                candidate_cached_prefix_len: cached_prefix_len,
                effective_plan,
                gdn_prefix_already_primed: continued_live_prefix,
                downgraded_to_cold: false,
            });
        }
        let max_cache_hit_tokens = total_budget.saturating_sub(1);
        let candidate_plan_result = match self.kv_cache_coordinator.as_mut() {
            Some(adapter) => adapter
                .prepare_turn_per_block_with_max_cache_hit_tokens(
                    0,
                    tokens,
                    total_budget,
                    false,
                    extra_keys_per_block,
                    cache_salt,
                    true,
                    max_cache_hit_tokens,
                )
                .map_err(Error::from_reason),
            None => Err(Error::from_reason(
                "prepare_gemma4_vlm_paged_prefix: paged_adapter is None",
            )),
        };
        let candidate_plan = match candidate_plan_result {
            Ok(plan) => plan,
            Err(error) => {
                self.invalidate_gemma4_hybrid_session("VLM paged-prefix preparation failure");
                return Err(error);
            }
        };
        self.kv_cache_coordinator
            .as_mut()
            .ok_or_else(|| Error::from_reason("Gemma4 hybrid KV coordinator missing"))?
            .begin_sliding_requests(0)
            .map_err(Error::from_reason)?;

        // Unified image K/V encodes the bidirectional overlay. The existing
        // attention API can consume an already-materialized overlay prefix only
        // when the cached boundary is after the complete expanded image run.
        // A candidate before/inside that run must be discarded, even if its
        // token/image block hash matches.
        let first_image_position = image_token_positions.first().map(|(position, _)| *position);
        let prefix_policy = gemma4_vlm_prefix_policy(
            candidate_plan.cached_prefix_len,
            first_image_position,
            unified_overlay_last_image_exclusive,
        );
        let sliding_preparation = if prefix_policy.unified_boundary_safe {
            self.prepare_gemma4_sliding_prefix_state_with_keys(
                tokens,
                candidate_plan.cached_prefix_len,
                candidate_plan.continued_live_prefix,
                extra_keys_per_block,
                image_token_positions,
                prefix_policy.require_exact_checkpoint,
                cache_salt,
            )
        } else {
            self.caches = Some(init_caches_for_config(&self.config));
            Ok(Gemma4SlidingPrefixPreparation {
                state: "unified_image_boundary_unsafe",
                primed_prefix_len: 0,
            })
        };
        let mut sliding_preparation = match sliding_preparation {
            Ok(preparation) => preparation,
            Err(error) => {
                self.invalidate_gemma4_hybrid_session("VLM sliding-prefix preparation failure");
                return Err(error);
            }
        };
        // A causal E2B hit ending before the first image token is pure text.
        // Reconstruct only that missing physical sliding prefix from token
        // embeddings; once a candidate includes an image position, replay is
        // forbidden because the placeholder embedding is not the vision feature.
        if prefix_policy.may_replay_leading_text
            && sliding_preparation.primed_prefix_len < candidate_plan.cached_prefix_len
        {
            let replay_result = (|| -> Result<()> {
                let replay = tokens
                    .get(
                        sliding_preparation.primed_prefix_len as usize
                            ..candidate_plan.cached_prefix_len as usize,
                    )
                    .ok_or_else(|| {
                        Error::from_reason("Gemma4 leading-text sliding replay range is invalid")
                    })?;
                let layer_kinds = self.compute_layer_kinds()?;
                self.run_sliding_only_prefill(
                    replay,
                    sliding_preparation.primed_prefix_len,
                    &layer_kinds,
                )?;
                self.remember_gemma4_sliding_materialized_prefix_checkpoint_with_keys(
                    tokens,
                    candidate_plan.cached_prefix_len,
                    block_size,
                    extra_keys_per_block,
                    cache_salt,
                )?;
                Ok(())
            })();
            if let Err(error) = replay_result {
                self.invalidate_gemma4_hybrid_session("VLM leading-text replay failure");
                return Err(error);
            }
            sliding_preparation.primed_prefix_len = candidate_plan.cached_prefix_len;
            sliding_preparation.state = "leading_text_replay";
        }
        let sliding_prefix_exact = prefix_policy.unified_boundary_safe
            && sliding_preparation.primed_prefix_len == candidate_plan.cached_prefix_len;

        let resolution =
            engine::resolve_vlm_paged_prefix(candidate_plan, sliding_prefix_exact, || {
                self.kv_cache_coordinator
                    .as_mut()
                    .ok_or_else(|| {
                        "prepare_gemma4_vlm_paged_prefix: adapter dropped before cold restart"
                            .to_string()
                    })?
                    .restart_prepared_turn_cold_per_block(
                        0,
                        tokens,
                        total_budget,
                        extra_keys_per_block,
                        0,
                    )
            });
        let resolution = match resolution {
            Ok(resolution) => resolution,
            Err(error) => {
                self.invalidate_gemma4_hybrid_session("VLM paged cold-restart failure");
                return Err(Error::from_reason(error));
            }
        };

        // The prepared request now owns the new turn. Clear only live/history
        // state; retain the bounded image-aware prefix checkpoints so A -> B -> A
        // can restore A after B displaced the live request.
        self.cached_token_history.clear();
        self.cached_image_key = None;
        self.cached_audio_key = None;
        self.cached_paged_image_token_positions = image_token_positions.to_vec();
        self.media_session_context = MediaCapabilities::NONE;
        self.paged_text_turn_context = MediaCapabilities::NONE;
        self.media_session_continuable = false;

        tracing::info!(
            target: "mlx_core::inference",
            event = "vlm_prefix_plan",
            model = "gemma4",
            prompt_tokens = tokens.len(),
            image_tokens = image_token_positions.len(),
            candidate_cached_prefix_tokens = resolution.candidate_cached_prefix_len,
            effective_cached_prefix_tokens = resolution.effective_plan.cached_prefix_len,
            continued_live_prefix = resolution.effective_plan.continued_live_prefix,
            sliding_prefix_exact,
            unified_boundary_safe = prefix_policy.unified_boundary_safe,
            downgraded_to_cold = resolution.downgraded_to_cold,
            "image-aware Gemma4 paged prefix planned"
        );

        Ok(resolution)
    }
}

// ---------------------------------------------------------------------------
// Vision helpers
// ---------------------------------------------------------------------------

/// Expand image tokens in a token sequence.
///
/// The chat template inserts a single `<|image|>` per image. This function
/// replaces each occurrence with: `boi_token + image_token × num_soft_tokens + eoi_token`.
///
/// The placeholder count must exactly match the processed image count. Prompt
/// structure belongs to the checkpoint template, so this processor never
/// inserts a fallback span at an invented position.
pub(super) fn expand_image_tokens(
    tokens: &[u32],
    processed_images: &[crate::models::gemma4::image_processor::ProcessedGemma4Image],
    image_token_id: u32,
    boi_token_id: u32,
    eoi_token_id: u32,
) -> Result<Vec<u32>> {
    let image_count = tokens.iter().filter(|&&t| t == image_token_id).count();
    if image_count != processed_images.len() {
        return Err(Error::from_reason(format!(
            "expand_image_tokens: {} image placeholder(s) but {} image(s) supplied",
            image_count,
            processed_images.len()
        )));
    }

    // Replace each <|image|> with the expanded BOI + N×image_token + EOI sequence
    let mut result = Vec::with_capacity(tokens.len() * 2);
    let mut img_idx = 0;
    for &t in tokens {
        if t == image_token_id && img_idx < processed_images.len() {
            let num_soft = processed_images[img_idx].num_soft_tokens;
            result.push(boi_token_id);
            for _ in 0..num_soft {
                result.push(image_token_id);
            }
            result.push(eoi_token_id);
            img_idx += 1;
        } else {
            result.push(t);
        }
    }
    Ok(result)
}

/// masked_scatter: replace positions where mask=true with values from source.
///
/// Matches Python: `mx.where(mask_flat, aligned, input_flat).reshape(input.shape)`
/// where `aligned = source.flatten()[(cumsum(mask_flat) - 1) % source.size]`
fn masked_scatter(input: &MxArray, mask: &MxArray, source: &MxArray) -> Result<MxArray> {
    let input_shape = input.shape()?;
    let mask_flat = mask.reshape(&[-1])?.astype(DType::Int32)?;
    let input_flat = input.reshape(&[-1])?;

    let source_flat = source.reshape(&[-1])?;
    let source_size = source_flat.shape_at(0)?;

    // cumsum of mask gives 1-based indices into source; subtract 1 for 0-based
    let indices = mask_flat.cumsum(0)?.sub(&MxArray::scalar_int(1)?)?;
    // Modulo source_size to handle wrap-around safely
    let source_size_arr = MxArray::scalar_int(source_size as i32)?;
    let safe_indices = indices.remainder(&source_size_arr)?;
    let aligned = source_flat.take(&safe_indices, 0)?;

    // where mask=1 use aligned (source), else keep input
    let result = mask_flat.where_(&aligned, &input_flat)?;
    result.reshape(&input_shape)
}

/// Reports whether `tokens` carry an image or audio placeholder id.
///
/// Used to decide whether a paged text turn may run a content-address prefix
/// lookup. Per-block prefix-cache hashes cover only token ids, not media
/// feature K/V, so a prompt that still holds media placeholders must skip the
/// lookup: otherwise a continue-turn-failure fallback could match the
/// token-only hash of media blocks registered by another session and reuse
/// that session's stale media K/V.
#[cfg(test)]
pub(super) fn prompt_holds_media_placeholders(
    tokens: &[u32],
    image_token_id: u32,
    audio_token_id: u32,
) -> bool {
    tokens.contains(&image_token_id) || tokens.contains(&audio_token_id)
}
