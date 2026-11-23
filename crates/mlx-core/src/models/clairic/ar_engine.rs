//! Clairic unified MLX execution engine (Path A grammar-constrained batched AR
//! decoding + Path B typed-head readout & answer-slot projection).
//!
//! Direct Rust/MLX port of upstream `clairic_runtime/mlx_engine.py::MLXEngine`
//! (lines 355-918), sharing one `Qwen35Inner` backbone across both paths.

use std::collections::HashMap;
use std::sync::{Arc, atomic::AtomicBool};

use image::RgbImage;
use mlx_clairic::ar::EOS as AR_EOS;
use mlx_clairic::embedded::RenderedPdfPageData;
use mlx_clairic::errors::check_cancel;
use mlx_clairic::grammar::{fill_batch_allowed_bitmasks, mask_and_score_row_bitmask};
use mlx_clairic::media::{hex_digest, rgb_image_digest};
use mlx_clairic::media_cache::{MediaCache, PreparedImages};
use mlx_clairic::orchestrator::{ClairicBackend, extract_json};
use mlx_clairic::profile::ClairicProfile;
use mlx_clairic::settings::SEGMENT_TOKENS;
use mlx_clairic::typed_prompt::{EncodedRecord, build_prediction_views_for_len};
pub use mlx_clairic::types::{ArGenOutput, ArGenTask, FinishReason};
use napi::Result;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::array::{DType, MxArray};
use crate::models::forward as model_fwd;
use crate::models::paddleocr_vl::processing::ProcessedImage;
use crate::models::qwen3_5::layer_cache::Qwen3_5LayerCache;
use crate::models::qwen3_5::model::Qwen35Inner;
use crate::stream::{Stream, StreamContext};
use crate::vision::qwen::processing::merged_image_token_count;
use crate::vision::qwen::prompt::{
    get_rope_index, inject_image_placeholders_with_token_id, merge_input_ids_with_image_features,
};

/// MLX-specialized prepared image cache entry holding `ProcessedImage` tensors.
pub type ClairicPreparedImages<M = ()> = PreparedImages<ProcessedImage, M>;

/// MLX-specialized LRU media cache holding `ProcessedImage` tensors.
pub type ClairicMediaCache<M = ()> = MediaCache<ProcessedImage, M>;

fn invalid_request(message: impl std::fmt::Display) -> napi::Error {
    super::to_napi_err(mlx_clairic::errors::invalid_request(message))
}

fn inference_error(message: impl std::fmt::Display) -> napi::Error {
    super::to_napi_err(mlx_clairic::errors::inference_error(message))
}

/// Threshold above which [`ClairicEngine::clear_request_caches`] drains MLX's
/// Metal allocator free pool (512 MiB, matching CLEF's cache drain threshold).
const CLEAR_CACHE_THRESHOLD_BYTES: u64 = 512 * 1024 * 1024;

/// Row chunk size when projecting answer-slot states through `lm_head`
/// (matches `mlx_engine.py:677`).
const ANSWER_SLOT_LM_HEAD_CHUNK: i64 = 256;

/// Stepwise per-layer eval threshold (`mlx_engine.py:600`).
const STEPWISE_EVAL_TOKEN_THRESHOLD: usize = 4096;

/// Runtime options for [`ClairicEngine`].
#[derive(Debug, Clone)]
pub struct EngineOptions {
    pub batch_size: usize,
    pub batch_token_budget: usize,
    pub segment_tokens: usize,
    pub prefix_cache: bool,
    pub memory_cap_bytes: Option<u64>,
}

impl Default for EngineOptions {
    fn default() -> Self {
        Self {
            batch_size: 16,
            batch_token_budget: 16_384,
            segment_tokens: SEGMENT_TOKENS,
            prefix_cache: true,
            memory_cap_bytes: None,
        }
    }
}

/// Cached `B=1` layer states after running a shared image prefix up to
/// `<|vision_end|>`.
struct PrefixSnapshot {
    caches: Vec<Qwen3_5LayerCache>,
    cache_len: i32,
    next_position: i32,
}

/// Live batched prefill/decode cache state across all layers of `Qwen35Inner`.
struct BatchCacheState {
    layers: Vec<Qwen3_5LayerCache>,
    /// `[B, cache_len]` bool mask when any row in the batch has right-padding.
    valid: Option<MxArray>,
    cache_len: i32,
    /// Per-row next rotary position (`[B]`) used for subsequent decode steps.
    next_positions: Vec<i32>,
}

impl BatchCacheState {
    fn empty(inner: &Qwen35Inner, batch: usize) -> Self {
        Self {
            layers: Qwen3_5LayerCache::fresh_layer_caches(&inner.config),
            valid: None,
            cache_len: 0,
            next_positions: vec![0; batch],
        }
    }

    fn eval_with_hidden(&self, hidden: &MxArray) -> Result<()> {
        let mut arrays: Vec<&MxArray> = Vec::with_capacity(1 + self.layers.len() * 2);
        arrays.push(hidden);
        for layer in &self.layers {
            layer.collect_arrays(&mut arrays);
        }
        MxArray::eval_arrays(&arrays)
    }

    fn eval_caches_only(&self) -> Result<()> {
        let mut arrays: Vec<&MxArray> = Vec::with_capacity(self.layers.len() * 2);
        for layer in &self.layers {
            layer.collect_arrays(&mut arrays);
        }
        if !arrays.is_empty() {
            MxArray::eval_arrays(&arrays)?;
        }
        Ok(())
    }
}

/// Embedded and right-padded batch ready for backbone forward.
struct EmbeddedBatch {
    inputs_embeds: MxArray,
    position_ids: MxArray,
    lengths: Vec<i32>,
}

/// Per-request Clairic execution engine borrowing the model-thread's
/// `&mut Qwen35Inner` backbone, typed head, vision tower, and grammar engine.
pub struct ClairicEngine<'a> {
    inner: &'a mut Qwen35Inner,
    options: EngineOptions,
    media_features: HashMap<String, MxArray>,
    media_grids: HashMap<String, [i32; 3]>,
    prefix_caches: HashMap<(String, String), PrefixSnapshot>,
}

impl<'a> ClairicEngine<'a> {
    /// Construct a request-scoped [`ClairicEngine`] with explicit options.
    pub(crate) fn new(inner: &'a mut Qwen35Inner, options: EngineOptions) -> Self {
        Self {
            inner,
            options: EngineOptions {
                batch_size: options.batch_size.max(1),
                batch_token_budget: options.batch_token_budget.max(1),
                segment_tokens: options.segment_tokens.max(1),
                prefix_cache: options.prefix_cache,
                memory_cap_bytes: options.memory_cap_bytes,
            },
            media_features: HashMap::new(),
            media_grids: HashMap::new(),
            prefix_caches: HashMap::new(),
        }
    }

    /// Construct a request-scoped [`ClairicEngine`] with default options.
    pub(crate) fn with_defaults(inner: &'a mut Qwen35Inner) -> Self {
        Self::new(inner, EngineOptions::default())
    }

    /// Active engine options.
    pub fn options(&self) -> &EngineOptions {
        &self.options
    }

    /// Drop all per-request vision feature caches and shared prefix snapshots,
    /// draining the Metal allocator pool only when it exceeds 512 MiB.
    pub fn clear_request_caches(&mut self) {
        self.media_features.clear();
        self.media_grids.clear();
        self.prefix_caches.clear();
        let cache_bytes = crate::array::get_cache_memory();
        if cache_bytes > CLEAR_CACHE_THRESHOLD_BYTES as f64 {
            crate::array::clear_cache();
        }
    }

    /// Path B: run segmented backbone prefill, evaluate `ClairicHead`, project
    /// optional answer-slot states through `lm_head` in 256-row chunks, and
    /// decode structured field answers (`mlx_engine.py::_read_typed`).
    pub fn read_typed(
        &mut self,
        item: &EncodedRecord,
        media: Option<&[Arc<RgbImage>]>,
        cancelled: Option<&Arc<AtomicBool>>,
    ) -> Result<Map<String, Value>> {
        check_cancel_opt(cancelled)?;
        if item.input_ids.is_empty() {
            return Err(invalid_request("read_typed received empty input_ids"));
        }
        if self.inner.clairic_head.is_none() {
            return Err(inference_error("ClairicHead is not loaded"));
        }
        if self.inner.tokenizer.is_none() {
            return Err(inference_error("Clairic tokenizer is not loaded"));
        }

        let _stream_ctx = StreamContext::new(Stream::generation());
        let pad_id = self.pad_token_id();
        let embedded = self.embed_batch(&[item.input_ids.as_slice()], &[media], pad_id)?;

        let (hidden_pre_norm, _) = self.run_segmented(
            embedded.inputs_embeds,
            embedded.position_ids,
            None,
            None,
            false,
            cancelled,
        )?;
        let normed = self.inner.final_norm.forward(&hidden_pre_norm)?;
        let states = normed.squeeze(Some(&[0]))?; // [L, H]
        MxArray::eval_arrays(&[&states])?;
        check_cancel_opt(cancelled)?;

        let states_len = usize::try_from(states.shape_at(0)?)
            .map_err(|_| inference_error("invalid hidden sequence length"))?;
        let states_dtype = states.dtype()?;
        let hidden_f32 = if states_dtype == DType::Float32 {
            states.clone()
        } else {
            states.astype(DType::Float32)?
        };

        let (input_start, stop, views, as_layout) =
            build_prediction_views_for_len(item, states_len).map_err(super::to_napi_err)?;

        // Scope the immutable borrow of `clairic_head` so `lm_head` projection
        // can run cleanly right afterwards.
        let (head_out, slot_states_opt) = {
            let head = self
                .inner
                .clairic_head
                .as_ref()
                .ok_or_else(|| inference_error("ClairicHead is not loaded"))?;
            let memory = head.memory(&hidden_f32, stop as i64)?;
            let head_out = head.forward(
                &hidden_f32,
                input_start as i64,
                stop as i64,
                &views,
                &memory,
            )?;
            let slot_states_opt = if !as_layout.is_empty() {
                let s = head.answer_slot_states(
                    &hidden_f32,
                    &as_layout.field_queries,
                    &as_layout.slot_kinds,
                    &as_layout.slot_positions,
                    &as_layout.slot_lengths,
                    stop as i64,
                    &memory,
                )?;
                Some(s)
            } else {
                None
            };
            (head_out, slot_states_opt)
        };

        let (slot_ids, slot_log_probs) = if let Some(slot_states) = slot_states_opt {
            let n_slot = usize::try_from(slot_states.shape_at(0)?)
                .map_err(|_| inference_error("invalid slot row count"))?;
            let width = usize::try_from(slot_states.shape_at(1)?)
                .map_err(|_| inference_error("invalid slot width"))?;
            let hidden_dim = slot_states.shape_at(2)?;
            let total_rows = (n_slot * width) as i64;
            let flat_states = slot_states
                .reshape(&[total_rows, hidden_dim])?
                .astype(states_dtype)?;

            let mut flat_ids: Vec<u32> = Vec::with_capacity(n_slot * width);
            let mut flat_lps: Vec<f32> = Vec::with_capacity(n_slot * width);

            let mut first = 0i64;
            while first < total_rows {
                check_cancel_opt(cancelled)?;
                let end = (first + ANSWER_SLOT_LM_HEAD_CHUNK).min(total_rows);
                let chunk = flat_states.slice_axis(0, first, end)?;
                let logits = self.project_logits_2d(&chunk)?.astype(DType::Float32)?;
                let best = logits.argmax(-1, Some(false))?.astype(DType::Uint32)?;
                let best_col = best.expand_dims(-1)?;
                let max_logit = logits
                    .take_along_axis(&best_col, -1)?
                    .squeeze(Some(&[-1]))?;
                let lse = logits.logsumexp(Some(&[-1]), Some(false))?;
                let lp = max_logit.sub(&lse)?;
                MxArray::eval_arrays(&[&best, &lp])?;

                flat_ids.extend_from_slice(&best.to_uint32()?);
                flat_lps.extend_from_slice(&lp.to_float32()?);
                first = end;
            }

            let mut ids_2d = Vec::with_capacity(n_slot);
            let mut lps_2d = Vec::with_capacity(n_slot);
            for r in 0..n_slot {
                let start = r * width;
                ids_2d.push(flat_ids[start..start + width].to_vec());
                lps_2d.push(flat_lps[start..start + width].to_vec());
            }
            (Some(ids_2d), Some(lps_2d))
        } else {
            (None, None)
        };

        let tokenizer = self
            .inner
            .tokenizer
            .as_ref()
            .ok_or_else(|| inference_error("Clairic tokenizer is not loaded"))?;
        super::head::decode_head_outputs(
            &head_out,
            item,
            Some(tokenizer.inner()),
            slot_ids,
            slot_log_probs,
        )
    }

    /// Path A: deterministic grammar-constrained batched generation with shared
    /// image-prefix caching and right-padded prefill/decode
    /// (`mlx_engine.py::_generate` + `_decode`).
    pub fn generate_batch(
        &mut self,
        tasks: &[ArGenTask],
        score_token_ids: &[u32],
        cancelled: Option<&Arc<AtomicBool>>,
    ) -> Result<Vec<ArGenOutput>> {
        check_cancel_opt(cancelled)?;
        if tasks.is_empty() {
            return Ok(Vec::new());
        }
        if self.inner.clairic_grammar.is_none() {
            return Err(inference_error("Clairic GrammarEngine is not loaded"));
        }
        if self.inner.tokenizer.is_none() {
            return Err(inference_error("Clairic tokenizer is not loaded"));
        }

        let _stream_ctx = StreamContext::new(Stream::generation());
        let default_eos = self.eos_token_id();
        let vision_end = self.vision_end_token_id();

        // Precompute canonical media key per task for grouping & prefix caching.
        let effective_keys: Vec<Option<String>> = tasks
            .iter()
            .map(|task| {
                task.media_key.clone().or_else(|| {
                    task.media
                        .as_deref()
                        .filter(|imgs| !imgs.is_empty())
                        .map(rgb_arcs_digest)
                })
            })
            .collect();

        // Group tasks by (max_new_tokens, resolved stop_token_ids), preserving first-seen order.
        let mut group_order: Vec<(usize, Vec<u32>)> = Vec::new();
        let mut group_map: HashMap<(usize, Vec<u32>), Vec<usize>> = HashMap::new();
        for (idx, task) in tasks.iter().enumerate() {
            if task.input_ids.is_empty() {
                return Err(invalid_request(format!(
                    "ArGenTask {:?} has empty input_ids",
                    task.id
                )));
            }
            let stops = if task.stop_token_ids.is_empty() {
                vec![default_eos]
            } else {
                task.stop_token_ids.clone()
            };
            let key = (task.max_new_tokens, stops);
            if let Some(members) = group_map.get_mut(&key) {
                members.push(idx);
            } else {
                group_order.push(key.clone());
                group_map.insert(key, vec![idx]);
            }
        }

        let mut results: Vec<Option<ArGenOutput>> = vec![None; tasks.len()];

        for group_key in group_order {
            let (max_new_tokens, stop_ids) = group_key.clone();
            let Some(mut indices) = group_map.remove(&group_key) else {
                continue;
            };
            indices.sort_by(|&a, &b| {
                let ka = effective_keys[a].as_deref().unwrap_or("");
                let kb = effective_keys[b].as_deref().unwrap_or("");
                ka.cmp(kb)
                    .then_with(|| tasks[a].input_ids.len().cmp(&tasks[b].input_ids.len()))
                    .then_with(|| a.cmp(&b))
            });

            // Pack into chunks respecting batch_size, media_key homogeneity,
            // and batch_token_budget (`mlx_engine.py:856-866`).
            let mut chunks: Vec<Vec<usize>> = Vec::new();
            let mut pending: Vec<usize> = Vec::new();
            for idx in indices {
                if !pending.is_empty() {
                    let key_differs = effective_keys[idx] != effective_keys[pending[0]];
                    let full = pending.len() >= self.options.batch_size;
                    let over_budget = if !key_differs && !full {
                        let mut cand: Vec<&[u32]> = Vec::with_capacity(pending.len() + 1);
                        for &p in &pending {
                            cand.push(tasks[p].input_ids.as_slice());
                        }
                        cand.push(tasks[idx].input_ids.as_slice());
                        batch_cost(
                            &cand,
                            effective_keys[idx].is_some(),
                            self.options.prefix_cache,
                            vision_end,
                        ) > self.options.batch_token_budget
                    } else {
                        false
                    };
                    if full || key_differs || over_budget {
                        chunks.push(pending);
                        pending = Vec::new();
                    }
                }
                pending.push(idx);
            }
            if !pending.is_empty() {
                chunks.push(pending);
            }

            for chunk in chunks {
                check_cancel_opt(cancelled)?;
                let chunk_tasks: Vec<&ArGenTask> = chunk.iter().map(|&i| &tasks[i]).collect();
                let chunk_keys: Vec<Option<&str>> = chunk
                    .iter()
                    .map(|&i| effective_keys[i].as_deref())
                    .collect();
                let decoded = self.decode_chunk(
                    &chunk_tasks,
                    &chunk_keys,
                    max_new_tokens,
                    &stop_ids,
                    score_token_ids,
                    cancelled,
                )?;
                let tokenizer = self
                    .inner
                    .tokenizer
                    .as_ref()
                    .ok_or_else(|| inference_error("Clairic tokenizer is not loaded"))?;

                for (&orig_idx, (output_ids, output_token_logprobs, first_step_token_logprobs)) in
                    chunk.iter().zip(decoded)
                {
                    let stopped_token = output_ids
                        .last()
                        .copied()
                        .filter(|tok| stop_ids.contains(tok));
                    let (finish_reason, decode_slice) = match stopped_token {
                        Some(tok) => (
                            FinishReason::Stop(tok),
                            &output_ids[..output_ids.len().saturating_sub(1)],
                        ),
                        None => (FinishReason::Length, output_ids.as_slice()),
                    };
                    let text = tokenizer.decode_sync(decode_slice, false)?;
                    results[orig_idx] = Some(ArGenOutput {
                        id: tasks[orig_idx].id.clone(),
                        output_ids,
                        text,
                        finish_reason,
                        output_token_logprobs,
                        first_step_token_logprobs,
                    });
                }
            }
        }

        results
            .into_iter()
            .enumerate()
            .map(|(i, opt)| {
                opt.ok_or_else(|| {
                    inference_error(format!("Missing ArGenOutput for batch index {i}"))
                })
            })
            .collect()
    }

    // =========================================================================
    // Internal helpers: vision features, embedding, segmented forward, decode
    // =========================================================================

    fn eos_token_id(&self) -> u32 {
        let vocab = self.inner.config.vocab_size.max(1) as u32;
        self.inner
            .tokenizer
            .as_ref()
            .and_then(|t| t.token_to_id("<|im_end|>".to_string()))
            .filter(|&id| id < vocab)
            .or_else(|| {
                self.inner
                    .tokenizer
                    .as_ref()
                    .map(|t| t.get_eos_token_id())
                    .filter(|&id| id < vocab)
            })
            .unwrap_or_else(|| (self.inner.config.eos_token_id.max(0) as u32).min(vocab - 1))
    }

    fn pad_token_id(&self) -> u32 {
        let vocab = self.inner.config.vocab_size.max(1) as u32;
        self.inner
            .tokenizer
            .as_ref()
            .map(|t| t.get_pad_token_id())
            .filter(|&id| id < vocab)
            .unwrap_or_else(|| {
                if self.inner.config.pad_token_id >= 0
                    && (self.inner.config.pad_token_id as u32) < vocab
                {
                    self.inner.config.pad_token_id as u32
                } else {
                    self.eos_token_id()
                }
            })
    }

    fn image_token_id(&self) -> i32 {
        self.inner.clairic_multimodal.image_token_id
    }

    fn vision_end_token_id(&self) -> u32 {
        self.inner.clairic_multimodal.vision_end_token_id.max(0) as u32
    }

    fn spatial_merge_size(&self) -> i32 {
        self.inner.spatial_merge_size.unwrap_or(2)
    }

    fn check_memory(&self) -> Result<()> {
        let Some(cap) = self.options.memory_cap_bytes else {
            return Ok(());
        };
        let cap_f = cap as f64;
        if crate::array::get_active_memory() > cap_f {
            crate::array::clear_cache();
            if crate::array::get_active_memory() > cap_f {
                return Err(inference_error(format!(
                    "request exceeds {:.1} GB active memory cap; shorten input or reduce batch_token_budget",
                    cap_f / 1e9
                )));
            }
        }
        Ok(())
    }

    /// Encode one `RgbImage` through `QwenImageProcessor` + `QwenVisionEncoder`
    /// and cache its `(features, grid_thw)` for the lifetime of this request.
    fn encode_image_cached(&mut self, img: &RgbImage) -> Result<(MxArray, [i32; 3])> {
        let key = rgb_image_digest(img);
        if let (Some(feats), Some(&grid)) =
            (self.media_features.get(&key), self.media_grids.get(&key))
        {
            return Ok((feats.clone(), grid));
        }
        let processor =
            self.inner.image_processor.as_ref().ok_or_else(|| {
                inference_error("QwenImageProcessor is not loaded for image input")
            })?;
        let encoder =
            self.inner.vision_encoder.as_ref().ok_or_else(|| {
                inference_error("QwenVisionEncoder is not loaded for image input")
            })?;

        let processed = processor.process_rgb_image(img)?;
        let grid_vec = processed.image_grid_thw();
        if grid_vec.len() != 3 {
            return Err(inference_error(format!(
                "Expected 3-element image_grid_thw, got {:?}",
                grid_vec
            )));
        }
        let grid = [grid_vec[0], grid_vec[1], grid_vec[2]];
        let pv = processed.pixel_values();
        let pv_shape = pv.shape()?;
        if pv_shape.len() != 4 {
            return Err(inference_error(format!(
                "Expected 4D processed pixel_values, got {:?}",
                pv_shape.as_ref()
            )));
        }
        let pixels_5d = pv.reshape(&[1, pv_shape[0], pv_shape[1], pv_shape[2], pv_shape[3]])?;
        let grid_arr = MxArray::from_int32(&grid, &[1, 3])?;
        let features = encoder.forward(&pixels_5d, &grid_arr)?;
        MxArray::eval_arrays(&[&features])?;

        self.media_features.insert(key.clone(), features.clone());
        self.media_grids.insert(key, grid);
        Ok((features, grid))
    }

    /// Embed a batch of sequences (with optional per-row images), right-padding
    /// to the maximum expanded sequence length and computing 3D M-RoPE
    /// `position_ids` `[3, B, T]`.
    fn embed_batch(
        &mut self,
        sequences: &[&[u32]],
        media: &[Option<&[Arc<RgbImage>]>],
        pad_id: u32,
    ) -> Result<EmbeddedBatch> {
        let batch = sequences.len();
        let image_token_id = self.image_token_id();
        let merge_size = self.spatial_merge_size();

        let mut expanded_sequences: Vec<Vec<u32>> = Vec::with_capacity(batch);
        let mut row_features: Vec<MxArray> = Vec::new();
        let mut row_grids: Vec<Option<Vec<i32>>> = Vec::with_capacity(batch);
        let mut any_media = false;

        for (seq, row_media) in sequences.iter().zip(media.iter()) {
            if let Some(imgs) = row_media
                && !imgs.is_empty()
            {
                any_media = true;
                let mut counts = Vec::with_capacity(imgs.len());
                let mut grids_flat = Vec::with_capacity(imgs.len() * 3);
                for img in *imgs {
                    let (feats, grid) = self.encode_image_cached(img)?;
                    let merged = merged_image_token_count(grid[0], grid[1], grid[2], merge_size)?;
                    counts.push(merged);
                    grids_flat.extend_from_slice(&grid);
                    row_features.push(feats);
                }
                let expanded =
                    inject_image_placeholders_with_token_id(seq, &counts, image_token_id)?;
                expanded_sequences.push(expanded);
                row_grids.push(Some(grids_flat));
            } else {
                expanded_sequences.push(seq.to_vec());
                row_grids.push(None);
            }
        }

        let max_len = expanded_sequences.iter().map(Vec::len).max().unwrap_or(0);
        if max_len == 0 {
            return Err(invalid_request("embed_batch received empty sequences"));
        }

        let lengths: Vec<i32> = expanded_sequences
            .iter()
            .map(|s| {
                i32::try_from(s.len())
                    .map_err(|_| invalid_request("sequence length exceeds i32::MAX"))
            })
            .collect::<Result<Vec<_>>>()?;

        let mut padded_ids = vec![pad_id; batch * max_len];
        for (row, seq) in expanded_sequences.iter().enumerate() {
            let dst = row * max_len;
            padded_ids[dst..dst + seq.len()].copy_from_slice(seq);
        }
        let ids_arr = MxArray::from_uint32(&padded_ids, &[batch as i64, max_len as i64])?;
        let text_embeds = self.inner.embedding.forward(&ids_arr)?;

        let (inputs_embeds, position_ids) = if any_media {
            let embed_dtype = text_embeds.dtype()?;
            let all_features = if row_features.len() == 1 {
                row_features.remove(0)
            } else {
                let refs: Vec<&MxArray> = row_features.iter().collect();
                MxArray::concatenate_many(refs, Some(0))?
            };
            let all_features = if all_features.dtype()? != embed_dtype {
                all_features.astype(embed_dtype)?
            } else {
                all_features
            };
            let merged = merge_input_ids_with_image_features(
                image_token_id,
                &all_features,
                &text_embeds,
                &ids_arr,
            )?;

            // Compute M-RoPE position_ids per row so each row receives only its
            // own image grids (`vision/qwen/prompt.rs::get_rope_index` expects
            // one row's grids).
            let mut per_row_pos: Vec<MxArray> = Vec::with_capacity(batch);
            for (row, grid_opt) in row_grids.iter().enumerate() {
                let row_ids = ids_arr.slice_axis(0, row as i64, row as i64 + 1)?;
                let grid_arr = match grid_opt {
                    Some(g) => Some(MxArray::from_int32(g, &[(g.len() / 3) as i64, 3])?),
                    None => None,
                };
                let (pos_row, _) =
                    get_rope_index(&row_ids, grid_arr.as_ref(), merge_size, image_token_id)?;
                per_row_pos.push(pos_row);
            }
            let pos_ids = if per_row_pos.len() == 1 {
                per_row_pos.remove(0)
            } else {
                let refs: Vec<&MxArray> = per_row_pos.iter().collect();
                MxArray::concatenate_many(refs, Some(1))?
            };
            (merged, pos_ids)
        } else {
            let pos = MxArray::arange(0.0, max_len as f64, Some(1.0), Some(DType::Int32))?
                .reshape(&[1, 1, max_len as i64])?;
            let pos_ids = MxArray::tile(&pos, &[3, batch as i32, 1])?;
            (text_embeds, pos_ids)
        };

        Ok(EmbeddedBatch {
            inputs_embeds,
            position_ids,
            lengths,
        })
    }

    /// Run one segment `[B, length, H]` through all backbone decoder layers,
    /// maintaining right-padded full-attention and GDN masks & caches
    /// (`mlx_engine.py::_forward`).
    fn forward_segment(
        &mut self,
        mut h: MxArray,
        positions: &MxArray,
        lengths: Option<&[i32]>,
        cache: Option<&mut BatchCacheState>,
        cancelled: Option<&Arc<AtomicBool>>,
    ) -> Result<MxArray> {
        let batch = usize::try_from(h.shape_at(0)?)
            .map_err(|_| inference_error("invalid batch dimension"))?;
        let length = i32::try_from(h.shape_at(1)?)
            .map_err(|_| inference_error("segment length exceeds i32::MAX"))?;

        let has_padding = lengths.is_some_and(|ls| ls.iter().any(|&l| l < length));
        let (valid_mask, linear_lengths) = if has_padding {
            let ls = lengths.ok_or_else(|| inference_error("missing lengths for padded batch"))?;
            (Some(build_valid_mask(ls, length)?), Some(ls))
        } else {
            (None, None)
        };

        let cache_len = cache.as_ref().map_or(0, |c| c.cache_len);
        let prev_valid = cache.as_ref().and_then(|c| c.valid.clone());

        let all_valid = if prev_valid.is_some() || valid_mask.is_some() {
            let new_valid = match &valid_mask {
                Some(v) => v.clone(),
                None => build_all_true_mask(batch, length)?,
            };
            if cache_len == 0 {
                Some(new_valid)
            } else {
                let before = match prev_valid {
                    Some(v) => v,
                    None => build_all_true_mask(batch, cache_len)?,
                };
                Some(MxArray::concatenate(&before, &new_valid, 1)?)
            }
        } else {
            None
        };

        let total = cache_len + length;
        let attn_mask = match &all_valid {
            Some(av) => Some(build_padded_attn_mask(av, batch, length, cache_len, total)?),
            None => None,
        };

        let stepwise = batch.saturating_mul(length as usize) > STEPWISE_EVAL_TOKEN_THRESHOLD;
        let mut cache_opt = cache;

        for (layer_idx, layer) in self.inner.layers.iter().enumerate() {
            check_cancel_opt(cancelled)?;
            let layer_cache = cache_opt.as_deref_mut().map(|c| &mut c.layers[layer_idx]);
            h = layer.forward_batched_flat(
                &h,
                attn_mask.as_ref(),
                valid_mask.as_ref(),
                linear_lengths,
                layer_cache,
                Some(positions),
            )?;
            if stepwise {
                if let Some(c) = cache_opt.as_deref() {
                    c.eval_with_hidden(&h)?;
                } else {
                    MxArray::eval_arrays(&[&h])?;
                }
                self.check_memory()?;
            }
        }

        if let Some(c) = cache_opt {
            c.valid = all_valid;
            c.cache_len = total;
        }
        Ok(h)
    }

    /// Run a whole sequence (or right-padded batch) through the backbone in
    /// `max(256, segment_tokens / batch)` chunks (`mlx_engine.py::_run`).
    ///
    /// When `keep == true`, returns `Some(BatchCacheState)` with `next_positions`
    /// set to each row's next rotary position (`max_axes(positions[:, b, len[b]-1]) + 1`).
    fn run_segmented(
        &mut self,
        h: MxArray,
        positions: MxArray,
        lengths: Option<&[i32]>,
        initial_cache: Option<BatchCacheState>,
        keep: bool,
        cancelled: Option<&Arc<AtomicBool>>,
    ) -> Result<(MxArray, Option<BatchCacheState>)> {
        let batch = usize::try_from(h.shape_at(0)?)
            .map_err(|_| inference_error("invalid batch dimension"))?;
        let length = usize::try_from(h.shape_at(1)?)
            .map_err(|_| inference_error("invalid sequence length"))?;
        let segment = (self.options.segment_tokens / batch.max(1)).max(256);

        let (out, mut cache_state) = if length <= segment {
            let mut state = match initial_cache {
                Some(c) => Some(c),
                None if keep => Some(BatchCacheState::empty(self.inner, batch)),
                None => None,
            };
            let out = self.forward_segment(h, &positions, lengths, state.as_mut(), cancelled)?;
            (out, state)
        } else {
            let mut state =
                initial_cache.unwrap_or_else(|| BatchCacheState::empty(self.inner, batch));
            let mut outputs: Vec<MxArray> = Vec::new();
            let mut start = 0usize;
            while start < length {
                check_cancel_opt(cancelled)?;
                let stop = (start + segment).min(length);
                let seg_len = (stop - start) as i32;
                let h_part = h.slice_axis(1, start as i64, stop as i64)?;
                let pos_part = positions.slice_axis(2, start as i64, stop as i64)?;
                let inside_vec: Option<Vec<i32>> = lengths.map(|ls| {
                    ls.iter()
                        .map(|&l| (l - start as i32).clamp(0, seg_len))
                        .collect()
                });
                let part = self.forward_segment(
                    h_part,
                    &pos_part,
                    inside_vec.as_deref(),
                    Some(&mut state),
                    cancelled,
                )?;
                state.eval_with_hidden(&part)?;
                self.check_memory()?;
                outputs.push(part);
                start = stop;
            }
            let refs: Vec<&MxArray> = outputs.iter().collect();
            let out = MxArray::concatenate_many(refs, Some(1))?;
            (out, if keep { Some(state) } else { None })
        };

        if keep && let Some(ref mut state) = cache_state {
            state.next_positions = compute_next_positions(&positions, lengths, batch, length)?;
        }

        Ok((out, cache_state))
    }

    /// Look up or compute the `B=1` prefix cache up to the last common
    /// `<|vision_end|>` token shared by every row in a batch, and broadcast it
    /// to `B = tasks.len()` (`mlx_engine.py::_image_prefix`).
    fn try_image_prefix(
        &mut self,
        tasks: &[&ArGenTask],
        media_keys: &[Option<&str>],
        cancelled: Option<&Arc<AtomicBool>>,
    ) -> Result<Option<(BatchCacheState, usize)>> {
        if !self.options.prefix_cache || tasks.is_empty() {
            return Ok(None);
        }
        let Some(first_key) = media_keys[0] else {
            return Ok(None);
        };
        if media_keys.iter().any(|k| *k != Some(first_key)) {
            return Ok(None);
        }

        let seq_slices: Vec<&[u32]> = tasks.iter().map(|t| t.input_ids.as_slice()).collect();
        let common_len = common_prefix_len(&seq_slices);
        if common_len == 0 {
            return Ok(None);
        }

        let vision_end = self.vision_end_token_id();
        let Some(ve_pos) = seq_slices[0][..common_len]
            .iter()
            .rposition(|&tok| tok == vision_end)
        else {
            return Ok(None);
        };
        let cut = ve_pos + 1;

        // Every row must retain at least 1 suffix token and contain no image
        // tokens after the cut.
        let image_token = self.image_token_id().max(0) as u32;
        if seq_slices
            .iter()
            .any(|s| s.len() <= cut || s[cut..].contains(&image_token))
        {
            return Ok(None);
        }

        let prefix_tokens = &seq_slices[0][..cut];
        let cache_key = (first_key.to_string(), hash_u32_slice(prefix_tokens));

        if !self.prefix_caches.contains_key(&cache_key) {
            let pad_id = self.pad_token_id();
            let embedded =
                self.embed_batch(&[prefix_tokens], &[tasks[0].media.as_deref()], pad_id)?;
            let prefix_expanded_len = embedded.lengths[0];
            let (_, state_opt) = self.run_segmented(
                embedded.inputs_embeds,
                embedded.position_ids,
                None,
                None,
                true,
                cancelled,
            )?;
            let state = state_opt
                .ok_or_else(|| inference_error("Prefix prefill returned no cache state"))?;
            state.eval_caches_only()?;
            let next_position = state
                .next_positions
                .first()
                .copied()
                .unwrap_or(prefix_expanded_len);
            self.prefix_caches.insert(
                cache_key.clone(),
                PrefixSnapshot {
                    caches: state.layers,
                    cache_len: state.cache_len,
                    next_position,
                },
            );
        }

        let snapshot = self
            .prefix_caches
            .get(&cache_key)
            .ok_or_else(|| inference_error("Prefix cache entry vanished"))?;
        let batch_i64 = tasks.len() as i64;
        let broadcast_layers = broadcast_layer_caches(&snapshot.caches, batch_i64)?;
        Ok(Some((
            BatchCacheState {
                layers: broadcast_layers,
                valid: None,
                cache_len: snapshot.cache_len,
                next_positions: vec![snapshot.next_position; tasks.len()],
            },
            cut,
        )))
    }

    /// Prefill one batch chunk, using the shared image-prefix cache when
    /// applicable, and gather each row's last real hidden state `[B, 1, H]`
    /// (`mlx_engine.py::_prefill`).
    fn prefill_chunk(
        &mut self,
        tasks: &[&ArGenTask],
        media_keys: &[Option<&str>],
        cancelled: Option<&Arc<AtomicBool>>,
    ) -> Result<(MxArray, BatchCacheState)> {
        let size = tasks.len();
        let pad_id = self.pad_token_id();

        let (h_in, positions, lengths, initial_cache) = if let Some((prefix_state, cut)) =
            self.try_image_prefix(tasks, media_keys, cancelled)?
        {
            let suffixes: Vec<&[u32]> = tasks.iter().map(|t| &t.input_ids[cut..]).collect();
            let max_suf = suffixes.iter().map(|s| s.len()).max().unwrap_or(0);
            let lengths: Vec<i32> = suffixes
                .iter()
                .map(|s| {
                    i32::try_from(s.len())
                        .map_err(|_| invalid_request("suffix length exceeds i32::MAX"))
                })
                .collect::<Result<Vec<_>>>()?;

            let mut padded_ids = vec![pad_id; size * max_suf];
            for (row, suf) in suffixes.iter().enumerate() {
                let dst = row * max_suf;
                padded_ids[dst..dst + suf.len()].copy_from_slice(suf);
            }
            let ids_arr = MxArray::from_uint32(&padded_ids, &[size as i64, max_suf as i64])?;
            let embeds = self.inner.embedding.forward(&ids_arr)?;

            let base_pos = prefix_state.next_positions[0];
            let pos_1d = MxArray::arange(
                f64::from(base_pos),
                f64::from(base_pos) + max_suf as f64,
                Some(1.0),
                Some(DType::Int32),
            )?
            .reshape(&[1, 1, max_suf as i64])?;
            let pos_3d = MxArray::tile(&pos_1d, &[3, size as i32, 1])?;

            (embeds, pos_3d, lengths, Some(prefix_state))
        } else {
            let seqs: Vec<&[u32]> = tasks.iter().map(|t| t.input_ids.as_slice()).collect();
            let medias: Vec<Option<&[Arc<RgbImage>]>> =
                tasks.iter().map(|t| t.media.as_deref()).collect();
            let embedded = self.embed_batch(&seqs, &medias, pad_id)?;
            (
                embedded.inputs_embeds,
                embedded.position_ids,
                embedded.lengths,
                None,
            )
        };

        let (h_out, cache_opt) = self.run_segmented(
            h_in,
            positions,
            Some(&lengths),
            initial_cache,
            true,
            cancelled,
        )?;
        let cache =
            cache_opt.ok_or_else(|| inference_error("Prefill did not return cache state"))?;

        // Extract each row's last valid hidden state -> [B, 1, H]
        let hidden_dim = h_out.shape_at(2)?;
        let last_indices: Vec<i32> = lengths.iter().map(|&l| (l - 1).max(0)).collect();
        let idx_arr = MxArray::from_int32(&last_indices, &[size as i64, 1, 1])?.broadcast_to(&[
            size as i64,
            1,
            hidden_dim,
        ])?;
        let last_h = h_out.take_along_axis(&idx_arr, 1)?;

        Ok((last_h, cache))
    }

    /// Build lazy FP32 step logits `[B, V]` from pre-norm `[B, 1, H]` hidden
    /// states through `final_norm` + `lm_head` without synchronizing Metal, so
    /// the caller can fuse logit evaluation with KV/GDN cache evaluation in a
    /// single `eval` barrier.
    fn build_step_logits_f32(&self, pre_norm_h: &MxArray) -> Result<(MxArray, usize)> {
        let normed = self.inner.final_norm.forward(pre_norm_h)?;
        let logits =
            model_fwd::project_logits(&normed, self.inner.lm_head.as_ref(), &self.inner.embedding)?
                .squeeze(Some(&[1]))?
                .astype(DType::Float32)?;
        let vocab = usize::try_from(logits.shape_at(1)?)
            .map_err(|_| inference_error("invalid logits vocab dimension"))?;
        Ok((logits, vocab))
    }

    /// Project post-norm 2D `[N, H]` states through `lm_head` (or tied embedding).
    fn project_logits_2d(&self, hidden_2d: &MxArray) -> Result<MxArray> {
        let h_3d = hidden_2d.expand_dims(0)?; // [1, N, H]
        model_fwd::project_logits(&h_3d, self.inner.lm_head.as_ref(), &self.inner.embedding)?
            .squeeze(Some(&[0]))
    }

    /// Greedy constrained decoding over one packed batch chunk
    /// (`mlx_engine.py::_decode`).
    #[allow(clippy::type_complexity)]
    fn decode_chunk(
        &mut self,
        tasks: &[&ArGenTask],
        media_keys: &[Option<&str>],
        max_new_tokens: usize,
        stop_ids: &[u32],
        score_token_ids: &[u32],
        cancelled: Option<&Arc<AtomicBool>>,
    ) -> Result<Vec<(Vec<u32>, Vec<(f32, u32)>, Vec<(f32, u32)>)>> {
        let size = tasks.len();
        if max_new_tokens == 0 {
            return Ok(vec![(Vec::new(), Vec::new(), Vec::new()); size]);
        }

        let (mut h, mut cache) = self.prefill_chunk(tasks, media_keys, cancelled)?;
        let (init_logits, logits_vocab) = self.build_step_logits_f32(&h)?;
        cache.eval_with_hidden(&init_logits)?;
        let mut scores = init_logits.to_float32()?.to_vec();

        let grammar_engine = self
            .inner
            .clairic_grammar
            .as_ref()
            .ok_or_else(|| inference_error("Clairic GrammarEngine is not loaded"))?
            .clone();
        let grammar_vocab = grammar_engine.vocab_size();
        if logits_vocab < grammar_vocab {
            return Err(inference_error(format!(
                "Model logits vocab {logits_vocab} is smaller than GrammarEngine vocab {grammar_vocab}"
            )));
        }

        let mut matchers = tasks
            .iter()
            .map(|t| {
                grammar_engine
                    .compile(&t.grammar)
                    .map_err(super::to_napi_err)
            })
            .collect::<Result<Vec<_>>>()?;

        let mut tokens: Vec<Vec<u32>> = vec![Vec::new(); size];
        let mut token_logprobs: Vec<Vec<(f32, u32)>> = vec![Vec::new(); size];
        let mut first_step_logprobs: Vec<Vec<(f32, u32)>> = vec![Vec::new(); size];
        let mut done: Vec<bool> = vec![false; size];
        let pad_id = self.pad_token_id();

        for step in 0..max_new_tokens {
            check_cancel_opt(cancelled)?;

            let active: Vec<bool> = (0..size).map(|row| !done[row] || step == 0).collect();
            let bitmasks =
                fill_batch_allowed_bitmasks(&mut matchers, &active).map_err(super::to_napi_err)?;
            let step_score_ids: &[u32] = if step == 0 { score_token_ids } else { &[] };

            for row in 0..size {
                if !active[row] {
                    continue;
                }
                let row_start = row * logits_vocab;
                let row_logits = &scores[row_start..row_start + grammar_vocab];
                let (choice, choice_lp, scored) =
                    mask_and_score_row_bitmask(row_logits, &bitmasks[row], step_score_ids);

                if step == 0 {
                    first_step_logprobs[row] = scored;
                }

                if done[row] {
                    continue;
                }

                tokens[row].push(choice);
                token_logprobs[row].push((choice_lp, choice));

                let is_stop = stop_ids.contains(&choice);
                if is_stop {
                    done[row] = true;
                }
                if !matchers[row].is_terminated() {
                    let accepted = matchers[row]
                        .accept_token(choice)
                        .map_err(super::to_napi_err)?;
                    if !accepted && !is_stop {
                        return Err(inference_error(format!(
                            "Generated token {choice} rejected by grammar for task {:?}",
                            tasks[row].id
                        )));
                    }
                } else if is_stop {
                    let _ = matchers[row].accept_token(choice);
                }
            }

            if done.iter().all(|&d| d) || step + 1 == max_new_tokens {
                break;
            }
            check_cancel_opt(cancelled)?;

            let feed_ids: Vec<u32> = (0..size)
                .map(|row| {
                    if done[row] {
                        pad_id
                    } else {
                        tokens[row].last().copied().unwrap_or(pad_id)
                    }
                })
                .collect();
            let feed_arr = MxArray::from_uint32(&feed_ids, &[size as i64, 1])?;
            let feed_embeds = self.inner.embedding.forward(&feed_arr)?;

            // Build [3, B, 1] rotary position_ids from cache.next_positions.
            let pos_1d = MxArray::from_int32(&cache.next_positions, &[1, size as i64, 1])?;
            let pos_3d = MxArray::tile(&pos_1d, &[3, 1, 1])?;

            h = self.forward_segment(feed_embeds, &pos_3d, None, Some(&mut cache), cancelled)?;
            for p in &mut cache.next_positions {
                *p += 1;
            }
            let (next_logits, _) = self.build_step_logits_f32(&h)?;
            cache.eval_with_hidden(&next_logits)?;
            scores = next_logits.to_float32()?.to_vec();
        }

        let mut out = Vec::with_capacity(size);
        for ((t, lp), first) in tokens
            .into_iter()
            .zip(token_logprobs)
            .zip(first_step_logprobs)
        {
            out.push((t, lp, first));
        }
        Ok(out)
    }
}

impl Drop for ClairicEngine<'_> {
    fn drop(&mut self) {
        self.clear_request_caches();
    }
}

impl ClairicBackend for ClairicEngine<'_> {
    fn tokenizer(&self) -> mlx_clairic::Result<Arc<tokenizers::Tokenizer>> {
        self.inner
            .tokenizer
            .as_ref()
            .map(|t| t.inner_arc())
            .ok_or_else(|| mlx_clairic::errors::inference_error("Clairic tokenizer is not loaded"))
    }

    fn eos_token_id(&self) -> mlx_clairic::Result<u32> {
        let tokenizer = self.inner.tokenizer.as_ref().ok_or_else(|| {
            mlx_clairic::errors::inference_error("Clairic tokenizer is not loaded")
        })?;
        let vocab = self.inner.config.vocab_size.max(1) as u32;
        Ok(tokenizer
            .token_to_id(AR_EOS.to_string())
            .filter(|&id| id < vocab)
            .unwrap_or_else(|| {
                let tok_eos = tokenizer.get_eos_token_id();
                if tok_eos < vocab {
                    tok_eos
                } else {
                    (self.inner.config.eos_token_id.max(0) as u32).min(vocab - 1)
                }
            }))
    }

    fn image_token_id(&self) -> u32 {
        self.inner.clairic_multimodal.image_token_id.max(0) as u32
    }

    fn vision_end_token_id(&self) -> u32 {
        ClairicEngine::vision_end_token_id(self)
    }

    fn profile(&self) -> ClairicProfile {
        self.inner.clairic_profile.unwrap_or(ClairicProfile::Flash)
    }

    fn build_vision_tokens(
        &self,
        images: &[Arc<RgbImage>],
    ) -> mlx_clairic::Result<(Vec<u32>, usize)> {
        let tokenizer = self.inner.tokenizer.as_ref().ok_or_else(|| {
            mlx_clairic::errors::inference_error("Clairic tokenizer is not loaded")
        })?;
        let processor = self.inner.image_processor.as_ref().ok_or_else(|| {
            mlx_clairic::errors::inference_error("QwenImageProcessor is not loaded for image input")
        })?;
        let merge_size = self.inner.spatial_merge_size.unwrap_or(2);
        let image_token_id = u32::try_from(self.inner.clairic_multimodal.image_token_id)
            .map_err(|_| mlx_clairic::errors::inference_error("invalid image_token_id"))?;
        let vision_end_id = u32::try_from(self.inner.clairic_multimodal.vision_end_token_id)
            .map_err(|_| mlx_clairic::errors::inference_error("invalid vision_end_token_id"))?;
        let vision_start_id = tokenizer.token_to_id("<|vision_start|>".to_string());

        let mut tokens: Vec<u32> = Vec::new();
        let mut total_image_tokens = 0usize;
        for img in images {
            let processed = processor
                .process_rgb_image(img)
                .map_err(super::napi_to_clairic_err)?;
            let grid = processed.image_grid_thw();
            if grid.len() != 3 {
                return Err(mlx_clairic::errors::inference_error(format!(
                    "Expected 3-element image_grid_thw, got {grid:?}"
                )));
            }
            let merged = merged_image_token_count(grid[0], grid[1], grid[2], merge_size)
                .map_err(super::napi_to_clairic_err)?;
            if let Some(vs) = vision_start_id {
                tokens.push(vs);
            }
            tokens.extend(std::iter::repeat_n(image_token_id, merged));
            tokens.push(vision_end_id);
            total_image_tokens += merged;
        }
        Ok((tokens, total_image_tokens))
    }

    fn read_typed(
        &mut self,
        item: &EncodedRecord,
        media: Option<&[Arc<RgbImage>]>,
        cancelled: Option<&Arc<AtomicBool>>,
    ) -> mlx_clairic::Result<Map<String, Value>> {
        ClairicEngine::read_typed(self, item, media, cancelled).map_err(super::napi_to_clairic_err)
    }

    fn generate_batch(
        &mut self,
        tasks: &[ArGenTask],
        score_token_ids: &[u32],
        cancelled: Option<&Arc<AtomicBool>>,
    ) -> mlx_clairic::Result<Vec<ArGenOutput>> {
        ClairicEngine::generate_batch(self, tasks, score_token_ids, cancelled)
            .map_err(super::napi_to_clairic_err)
    }

    fn clear_request_caches(&mut self) {
        ClairicEngine::clear_request_caches(self);
    }
}

/// Full dual-path Clairic request entry point running on `&mut Qwen35Inner`
/// via [`ClairicEngine`] and [`mlx_clairic::orchestrator::extract_json`].
pub(crate) fn extract_json_on_inner(
    inner: &mut Qwen35Inner,
    raw_request: &str,
    rendered_pdf_pages: Vec<RenderedPdfPageData>,
    cancelled: Option<&Arc<AtomicBool>>,
) -> Result<String> {
    let mut engine = ClairicEngine::with_defaults(inner);
    extract_json(&mut engine, raw_request, rendered_pdf_pages, cancelled)
        .map_err(super::to_napi_err)
}

// =============================================================================
// Pure helpers
// =============================================================================

fn check_cancel_opt(cancelled: Option<&Arc<AtomicBool>>) -> Result<()> {
    if let Some(flag) = cancelled {
        check_cancel(flag).map_err(super::to_napi_err)?;
    }
    Ok(())
}

fn rgb_arcs_digest(images: &[Arc<RgbImage>]) -> String {
    let mut hasher = Sha256::new();
    hasher.update((images.len() as u64).to_be_bytes());
    for img in images {
        hasher.update(img.width().to_be_bytes());
        hasher.update(img.height().to_be_bytes());
        hasher.update(img.as_raw());
    }
    hex_digest(&hasher.finalize())
}

fn hash_u32_slice(tokens: &[u32]) -> String {
    let mut hasher = Sha256::new();
    for &t in tokens {
        hasher.update(t.to_le_bytes());
    }
    hex_digest(&hasher.finalize())
}

fn common_prefix_len(sequences: &[&[u32]]) -> usize {
    let (Some(first), Some(last)) = (sequences.iter().min(), sequences.iter().max()) else {
        return 0;
    };
    first
        .iter()
        .zip(last.iter())
        .position(|(a, b)| a != b)
        .unwrap_or_else(|| first.len().min(last.len()))
}

fn batch_cost(sequences: &[&[u32]], has_media: bool, prefix_cache: bool, vision_end: u32) -> usize {
    let shared = if has_media && prefix_cache && sequences.len() > 1 {
        let common = common_prefix_len(sequences);
        sequences[0][..common]
            .iter()
            .rposition(|&tok| tok == vision_end)
            .map_or(0, |idx| idx + 1)
    } else {
        0
    };
    let max_len = sequences.iter().map(|s| s.len()).max().unwrap_or(0);
    max_len
        .saturating_sub(shared)
        .saturating_mul(sequences.len())
}

fn build_valid_mask(lengths: &[i32], width: i32) -> Result<MxArray> {
    let batch = lengths.len();
    let w = width as usize;
    let mut data = vec![0i32; batch * w];
    for (row, &len) in lengths.iter().enumerate() {
        let clamped = (len.max(0) as usize).min(w);
        for v in &mut data[row * w..row * w + clamped] {
            *v = 1;
        }
    }
    MxArray::from_int32(&data, &[batch as i64, width as i64])?.greater(&MxArray::scalar_int(0)?)
}

fn build_all_true_mask(batch: usize, width: i32) -> Result<MxArray> {
    let ones = MxArray::ones(&[batch as i64, width as i64], Some(DType::Int32))?;
    ones.greater(&MxArray::scalar_int(0)?)
}

/// Build the `[B, 1, Q, K]` boolean SDPA mask for right-padded batches matching
/// `mlx_engine.py:560-568`.
///
/// MLX SDPA boolean mask semantics: `true` = attend, `false` = mask out (`-inf`).
/// Padded query rows still attend to themselves (`diag`) so softmax never yields NaN.
fn build_padded_attn_mask(
    all_valid: &MxArray,
    batch: usize,
    q_len: i32,
    offset: i32,
    total_kv: i32,
) -> Result<MxArray> {
    let valid_4d = all_valid.reshape(&[batch as i64, 1, 1, total_kv as i64])?;
    if q_len <= 1 {
        return Ok(valid_4d);
    }
    let key_index = MxArray::arange(0.0, f64::from(total_kv), Some(1.0), Some(DType::Int32))?
        .reshape(&[1, 1, 1, total_kv as i64])?;
    let row_index = MxArray::arange(
        f64::from(offset),
        f64::from(offset + q_len),
        Some(1.0),
        Some(DType::Int32),
    )?
    .reshape(&[1, 1, q_len as i64, 1])?;

    let causal = key_index.less_equal(&row_index)?;
    let diag = key_index.equal(&row_index)?;
    let visible = valid_4d.logical_or(&diag)?;
    causal.logical_and(&visible)
}

/// Extract each row's next rotary position (`max_axes(positions[:, b, len[b]-1]) + 1`)
/// matching `mlx_engine.py:644-646`.
fn compute_next_positions(
    positions: &MxArray,
    lengths: Option<&[i32]>,
    batch: usize,
    width: usize,
) -> Result<Vec<i32>> {
    let last_indices: Vec<i32> = match lengths {
        Some(ls) => ls.iter().map(|&l| (l - 1).max(0)).collect(),
        None => vec![(width.saturating_sub(1)) as i32; batch],
    };
    let axes = positions.shape_at(0)?;
    let idx_arr = MxArray::from_int32(&last_indices, &[1, batch as i64, 1])?.broadcast_to(&[
        axes,
        batch as i64,
        1,
    ])?;
    let gathered = positions
        .take_along_axis(&idx_arr, 2)?
        .squeeze(Some(&[2]))?; // [3, B]
    MxArray::eval_arrays(&[&gathered])?;
    let vals = gathered.to_int32()?;
    let mut next_pos = Vec::with_capacity(batch);
    for b in 0..batch {
        let mut max_p = i32::MIN;
        for a in 0..(axes as usize) {
            let p = vals[a * batch + b];
            if p > max_p {
                max_p = p;
            }
        }
        next_pos.push(max_p + 1);
    }
    Ok(next_pos)
}

fn broadcast_layer_caches(
    prefix_caches: &[Qwen3_5LayerCache],
    batch: i64,
) -> Result<Vec<Qwen3_5LayerCache>> {
    prefix_caches
        .iter()
        .map(|slot| match slot {
            Qwen3_5LayerCache::Linear(ac) => {
                Ok(Qwen3_5LayerCache::Linear(ac.broadcast_batch(batch)?))
            }
            Qwen3_5LayerCache::FullAttention(kv) => {
                Ok(Qwen3_5LayerCache::FullAttention(kv.broadcast_batch(batch)?))
            }
        })
        .collect()
}

// =============================================================================
// Unit tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::clairic::head::{self, ClairicHead, ClairicHeadConfig};
    use crate::models::qwen3_5::attention::RopeMode;
    use crate::models::qwen3_5::config::{Qwen3_5Config, Qwen3_5MultimodalTokens};
    use crate::models::qwen3_5::decoder_layer::AttentionType;
    use crate::tokenizer::Qwen3Tokenizer;
    use mlx_clairic::grammar::{GrammarEngine, GrammarSpec};
    use mlx_clairic::profile::ClairicProfile;
    use mlx_clairic::typed_prompt::{EncodeRecordOptions, encode_record};
    use serde_json::json;
    use std::path::PathBuf;

    fn clef_tokenizer_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../__test__/fixtures/clef/tokenizer.json")
    }

    fn det_vec(seed: u64, len: usize, scale: f32) -> Vec<f32> {
        let mut x = seed ^ 0x9E37_79B9_7F4A_7C15;
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let u = ((x >> 40) as f32) / ((1u32 << 24) as f32);
            out.push((u * 2.0 - 1.0) * scale);
        }
        out
    }

    fn det_arr(seed: u64, shape: &[i64], scale: f32) -> MxArray {
        let n: usize = shape.iter().map(|&d| d as usize).product::<usize>().max(1);
        MxArray::from_float32(&det_vec(seed, n, scale), shape).unwrap()
    }

    fn synthetic_clairic_head(hidden_size: i64, bins: i64) -> ClairicHead {
        let cfg = ClairicHeadConfig {
            hidden_size,
            bins,
            pointer_width: 16,
            readout: "lexical".to_string(),
            route_width: 32,
            route_layers: 1,
            route_heads: 4,
            slot_rank: 8,
            estimate_lexical: true,
            span_trim: true,
            loop_feedback: false,
            answer_slots: true,
            answer_slot_routing: true,
            number_groups: true,
            number_pointer: true,
        };
        let mut weights = HashMap::new();
        let h = cfg.hidden_size;
        let w = cfg.route_width;
        let p = cfg.pointer_width;
        let r = cfg.slot_rank;
        let mut seed = 101u64;

        fn add(weights: &mut HashMap<String, MxArray>, seed: &mut u64, name: &str, shape: &[i64]) {
            let numel: usize = shape.iter().map(|&d| d as usize).product::<usize>().max(1);
            let data = if name.ends_with(".weight") && shape.len() == 1 {
                vec![1.0f32; numel]
            } else if name == "lexical_scale" {
                vec![2.0, 2.0, 1.5]
            } else {
                det_vec(*seed, numel, 0.2)
            };
            *seed += 1;
            weights.insert(
                name.to_string(),
                MxArray::from_float32(&data, shape).unwrap(),
            );
        }
        fn norm(weights: &mut HashMap<String, MxArray>, seed: &mut u64, prefix: &str, width: i64) {
            add(weights, seed, &format!("{prefix}.weight"), &[width]);
            add(weights, seed, &format!("{prefix}.bias"), &[width]);
        }
        fn linear(
            weights: &mut HashMap<String, MxArray>,
            seed: &mut u64,
            prefix: &str,
            inp: i64,
            out: i64,
            bias: bool,
        ) {
            add(weights, seed, &format!("{prefix}.weight"), &[out, inp]);
            if bias {
                add(weights, seed, &format!("{prefix}.bias"), &[out]);
            }
        }
        fn attn(weights: &mut HashMap<String, MxArray>, seed: &mut u64, prefix: &str, width: i64) {
            add(
                weights,
                seed,
                &format!("{prefix}.in_proj_weight"),
                &[3 * width, width],
            );
            add(
                weights,
                seed,
                &format!("{prefix}.in_proj_bias"),
                &[3 * width],
            );
            linear(
                weights,
                seed,
                &format!("{prefix}.out_proj"),
                width,
                width,
                true,
            );
        }

        add(&mut weights, &mut seed, "slot_route_scale", &[8]);
        norm(&mut weights, &mut seed, "norm", h);
        norm(&mut weights, &mut seed, "token_norm", h);
        linear(&mut weights, &mut seed, "null", h, 1, true);
        linear(&mut weights, &mut seed, "span_query", h, 2 * p, true);
        linear(&mut weights, &mut seed, "span_key", h, 2 * p, true);
        add(&mut weights, &mut seed, "length_bias", &[head::MAX_SPAN]);
        linear(&mut weights, &mut seed, "tag_query", h, 2 * p, true);
        linear(&mut weights, &mut seed, "tag_key", h, p, true);
        add(&mut weights, &mut seed, "tag_bias", &[2]);
        linear(
            &mut weights,
            &mut seed,
            "trim",
            h,
            2 * head::TRIM_CLASSES,
            true,
        );
        linear(&mut weights, &mut seed, "route_in", h, w, false);
        add(
            &mut weights,
            &mut seed,
            "kind_embedding.weight",
            &[head::KIND_EMBEDDING_ROWS, w],
        );
        linear(&mut weights, &mut seed, "memory_in", h, w, false);
        for n in ["norm1", "norm2", "norm3"] {
            norm(&mut weights, &mut seed, &format!("route_layers.0.{n}"), w);
        }
        attn(&mut weights, &mut seed, "route_layers.0.self_attn", w);
        attn(&mut weights, &mut seed, "route_layers.0.multihead_attn", w);
        linear(
            &mut weights,
            &mut seed,
            "route_layers.0.linear1",
            w,
            4 * w,
            true,
        );
        linear(
            &mut weights,
            &mut seed,
            "route_layers.0.linear2",
            4 * w,
            w,
            true,
        );
        norm(&mut weights, &mut seed, "route_norm", w);
        linear(&mut weights, &mut seed, "route_out", w, h, true);
        add(&mut weights, &mut seed, "route_gate", &[]);
        add(&mut weights, &mut seed, "digit_rows", &[10, h]);
        add(&mut weights, &mut seed, "month_rows", &[12, h]);
        add(&mut weights, &mut seed, "sign_rows", &[2, h]);
        add(&mut weights, &mut seed, "lexical_scale", &[3]);
        linear(
            &mut weights,
            &mut seed,
            "slot_down",
            h,
            (head::DIGIT_SLOTS + 2) * r,
            false,
        );
        add(
            &mut weights,
            &mut seed,
            "slot_up",
            &[head::DIGIT_SLOTS, r, 10],
        );
        add(&mut weights, &mut seed, "month_up", &[r, 12]);
        add(&mut weights, &mut seed, "sign_up", &[r, 2]);
        linear(
            &mut weights,
            &mut seed,
            "slot_residual",
            w,
            head::DIGIT_SLOTS * 10 + 12 + 2,
            true,
        );
        add(&mut weights, &mut seed, "residual_gate", &[]);
        linear(
            &mut weights,
            &mut seed,
            "exponent",
            h,
            head::EXPONENT_CLASSES,
            true,
        );
        linear(&mut weights, &mut seed, "item_end", h, 1, true);
        linear(&mut weights, &mut seed, "bins", h, cfg.bins, true);
        linear(&mut weights, &mut seed, "offset", h, cfg.bins, true);
        linear(&mut weights, &mut seed, "overflow", h, 2, true);
        linear(&mut weights, &mut seed, "estimate_gate", h, 1, true);
        linear(
            &mut weights,
            &mut seed,
            "estimate_spread",
            h,
            1 + head::LEXICAL_DIGITS,
            true,
        );
        linear(
            &mut weights,
            &mut seed,
            "number_group_down",
            h,
            head::GROUP_RANK,
            true,
        );
        linear(
            &mut weights,
            &mut seed,
            "number_group_up",
            head::GROUP_RANK,
            head::NUMBER_GROUPS * head::GROUP_CLASSES,
            true,
        );
        linear(&mut weights, &mut seed, "copy_query", h, 2 * p, true);
        linear(&mut weights, &mut seed, "copy_gate", h, 1, true);

        ClairicHead::from_weights(cfg, weights, hidden_size, false).expect("build synthetic head")
    }

    /// Build a deterministic 2-layer synthetic `Qwen35Inner` (1 GDN + 1 full
    /// attention, `hidden=64`, real `GrammarEngine` & `Qwen3Tokenizer` from the
    /// CLEF fixture, synthetic `ClairicHead`, `RopeMode::Temporal`).
    fn synthetic_clairic_inner() -> (Qwen35Inner, u32, u32) {
        let tok_path = clef_tokenizer_path();
        let tokenizer = Arc::new(
            Qwen3Tokenizer::load_from_file_sync(tok_path.to_str().unwrap())
                .expect("load clef fixture tokenizer"),
        );
        let vocab_size = tokenizer.vocab_size() as i32;
        let eos_id = tokenizer
            .token_to_id("<|im_end|>".to_string())
            .filter(|&id| (id as i32) < vocab_size)
            .unwrap_or_else(|| {
                let candidate = tokenizer.get_eos_token_id();
                if (candidate as i32) < vocab_size {
                    candidate
                } else {
                    (vocab_size - 1).max(0) as u32
                }
            });
        let pad_id = {
            let candidate = tokenizer.get_pad_token_id();
            if (candidate as i32) < vocab_size {
                candidate
            } else {
                0
            }
        };
        // Use a valid in-vocab token as synthetic vision_end_token_id.
        let vision_end_id: i32 = 7.min(vocab_size - 1);
        let image_token_id: i32 = 8.min(vocab_size - 1);

        let config = Qwen3_5Config {
            vocab_size,
            hidden_size: 64,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 2,
            intermediate_size: 128,
            rms_norm_eps: 1e-6,
            head_dim: 16,
            tie_word_embeddings: true,
            attention_bias: false,
            max_position_embeddings: 512,
            pad_token_id: pad_id as i32,
            eos_token_id: eos_id as i32,
            bos_token_id: 0,
            linear_num_value_heads: 4,
            linear_num_key_heads: 2,
            linear_key_head_dim: 16,
            linear_value_head_dim: 16,
            linear_conv_kernel_dim: 4,
            full_attention_interval: 2,
            partial_rotary_factor: 0.5,
            rope_theta: 10_000.0,
            paged_cache_memory_mb: None,
            paged_block_size: None,
            n_mtp_layers: 0,
            qwen35_gguf_gdn_layout: None,
            kv_format: Some("bf16".to_string()),
            rope_scaling: None,
        };

        let mut inner = Qwen35Inner::new(config.clone()).expect("Qwen35Inner::new");
        let h = config.hidden_size as i64;
        let v = config.vocab_size as i64;
        inner
            .embedding
            .set_weight(&det_arr(11, &[v, h], 0.15))
            .unwrap();
        inner
            .final_norm
            .set_weight(&MxArray::ones(&[h], Some(DType::Float32)).unwrap())
            .unwrap();

        let key_dim = config.linear_key_dim() as i64;
        let val_dim = config.linear_value_dim() as i64;
        let conv_dim = config.linear_conv_dim() as i64;
        let hv = config.linear_num_value_heads as i64;
        let nh = config.num_heads as i64;
        let nkv = config.num_kv_heads as i64;
        let hd = config.head_dim as i64;
        let inter = config.intermediate_size as i64;

        for (i, layer) in inner.layers.iter_mut().enumerate() {
            let s = (i as u64 + 1) * 31;
            match &mut layer.attn {
                AttentionType::Linear(gdn) => {
                    gdn.set_in_proj_qkvz_weight(&det_arr(
                        s + 1,
                        &[2 * key_dim + 2 * val_dim, h],
                        0.1,
                    ))
                    .unwrap();
                    gdn.set_in_proj_ba_weight(&det_arr(s + 2, &[2 * hv, h], 0.1))
                        .unwrap();
                    gdn.set_conv1d_weight(&det_arr(s + 3, &[conv_dim, 4, 1], 0.15), DType::Float32)
                        .unwrap();
                    gdn.set_out_proj_weight(&det_arr(s + 4, &[h, val_dim], 0.1))
                        .unwrap();
                }
                AttentionType::Full(attn) => {
                    attn.set_q_proj_weight(&det_arr(s + 5, &[2 * nh * hd, h], 0.1))
                        .unwrap();
                    attn.set_k_proj_weight(&det_arr(s + 6, &[nkv * hd, h], 0.1))
                        .unwrap();
                    attn.set_v_proj_weight(&det_arr(s + 7, &[nkv * hd, h], 0.1))
                        .unwrap();
                    attn.set_o_proj_weight(&det_arr(s + 8, &[h, nh * hd], 0.1))
                        .unwrap();
                }
            }
            if let crate::models::quantized_linear::MLPVariant::Standard(mlp) = &mut layer.mlp {
                mlp.set_gate_proj_weight(&det_arr(s + 9, &[inter, h], 0.08))
                    .unwrap();
                mlp.set_up_proj_weight(&det_arr(s + 10, &[inter, h], 0.08))
                    .unwrap();
                mlp.set_down_proj_weight(&det_arr(s + 11, &[h, inter], 0.08))
                    .unwrap();
            }
        }

        // rope_dims = 16 * 0.5 = 8 -> half_dim = 4, mrope_section sums to 4.
        inner
            .init_mrope_layers(
                vec![2, 1, 1],
                config.rope_theta,
                config.max_position_embeddings,
            )
            .unwrap();
        inner.set_rope_mode_layers(RopeMode::Temporal);

        inner.clairic_multimodal = Qwen3_5MultimodalTokens {
            image_token_id,
            vision_end_token_id: vision_end_id,
            mrope_section: vec![2, 1, 1],
        };
        let grammar_engine = GrammarEngine::new(
            tokenizer.inner_arc(),
            vocab_size as usize,
            eos_id,
            vec![eos_id],
        )
        .expect("build GrammarEngine");
        let head = synthetic_clairic_head(h, 256);
        inner.clairic_profile = Some(ClairicProfile::from_head_config(head.config()));
        inner.clairic_head = Some(head);
        inner.clairic_grammar = Some(Arc::new(grammar_engine));
        inner.set_tokenizer(tokenizer);

        (inner, eos_id, vision_end_id as u32)
    }

    #[test]
    fn ar_engine_constrained_batch_and_p_null() {
        let (mut inner, eos_id, _) = synthetic_clairic_inner();
        let grammar_engine = inner.clairic_grammar.as_ref().unwrap().clone();

        // Find a token id to score at step 0 (e.g. token for "null" or fallback).
        let null_toks = grammar_engine.tok_env().tokenize("null");
        let score_ids = vec![null_toks[0], eos_id];

        let tasks = vec![
            ArGenTask {
                id: "bool-field".to_string(),
                input_ids: vec![10, 11, 12, 13, 14], // len 5
                media: None,
                media_key: None,
                grammar: GrammarSpec::JsonSchema(json!({ "type": "boolean" })),
                max_new_tokens: 8,
                stop_token_ids: vec![eos_id],
            },
            ArGenTask {
                id: "num-field".to_string(),
                input_ids: vec![20, 21, 22, 23, 24, 25, 26, 27, 28], // len 9
                media: None,
                media_key: None,
                grammar: GrammarSpec::JsonSchema(json!({ "type": "number" })),
                max_new_tokens: 8,
                stop_token_ids: vec![eos_id],
            },
            ArGenTask {
                id: "grounded-field".to_string(),
                input_ids: vec![30, 31, 32, 33], // len 4
                media: None,
                media_key: None,
                grammar: GrammarSpec::GroundedText {
                    source: "Acme Corp Invoice".to_string(),
                    normalized_source: Some("Acme Corp Invoice".to_string()),
                    nullable: true,
                },
                max_new_tokens: 8,
                stop_token_ids: vec![eos_id],
            },
        ];

        // 1. Run as a single right-padded batch (B=3).
        let batched = {
            let mut engine = ClairicEngine::with_defaults(&mut inner);
            engine
                .generate_batch(&tasks, &score_ids, None)
                .expect("batched generate_batch")
        };
        assert_eq!(batched.len(), 3);

        // 2. Run each task independently (B=1) and assert bit-for-bit token parity
        //    and close logprob parity.
        for (i, task) in tasks.iter().enumerate() {
            let single = {
                let mut engine = ClairicEngine::with_defaults(&mut inner);
                engine
                    .generate_batch(std::slice::from_ref(task), &score_ids, None)
                    .expect("single generate_batch")
            };
            assert_eq!(batched[i].id, task.id);
            assert_eq!(
                batched[i].output_ids, single[0].output_ids,
                "row {i} ({}) output_ids diverged between B=3 and B=1",
                task.id
            );
            assert_eq!(batched[i].text, single[0].text);
            assert_eq!(batched[i].finish_reason, single[0].finish_reason);

            // Verify every step logprob is finite and matches B=1.
            assert_eq!(
                batched[i].output_token_logprobs.len(),
                batched[i].output_ids.len()
            );
            for ((blp, btok), (slp, stok)) in batched[i]
                .output_token_logprobs
                .iter()
                .zip(single[0].output_token_logprobs.iter())
            {
                assert_eq!(btok, stok);
                assert!(blp.is_finite(), "non-finite token logprob for row {i}");
                // B=1 takes the BF16 fused `mlx_qwen4_gdn_prepare` path while B>1 uses FP32 ops,
                // allowing ~1 BF16 ULP (~3e-3) logprob delta while emitted token IDs stay identical.
                assert!(
                    (blp - slp).abs() < 5e-3,
                    "row {i} logprob diff too large: {blp} vs {slp}"
                );
            }

            // Verify step-0 score_token_ids logprobs match B=1.
            assert_eq!(batched[i].first_step_token_logprobs.len(), score_ids.len());
            for ((blp, btok), (slp, stok)) in batched[i]
                .first_step_token_logprobs
                .iter()
                .zip(single[0].first_step_token_logprobs.iter())
            {
                assert_eq!(btok, stok);
                if blp.is_finite() || slp.is_finite() {
                    assert!(
                        (blp - slp).abs() < 5e-3,
                        "row {i} step-0 logprob diff: {blp} vs {slp}"
                    );
                }
            }

            // Verify emitted sequence obeys its grammar!
            let mut matcher = grammar_engine.compile(&task.grammar).unwrap();
            let mut allowed = vec![false; grammar_engine.vocab_size()];
            for &tok in &batched[i].output_ids {
                matcher.fill_allowed_mask(&mut allowed).unwrap();
                assert!(
                    allowed[tok as usize],
                    "row {i} emitted token {tok} disallowed by grammar"
                );
                assert!(matcher.accept_token(tok).unwrap());
            }
        }
        // For nullable GroundedText, step-0 p_null token ("null") must be finite.
        assert!(
            batched[2].first_step_token_logprobs[0].0.is_finite(),
            "expected finite step-0 p_null logprob for nullable GroundedText"
        );
    }

    #[test]
    fn ar_engine_shared_prefix_matches_uncached() {
        let (mut inner, eos_id, vision_end_id) = synthetic_clairic_inner();
        let score_ids = vec![eos_id];

        // 3 prompts sharing a 6-token prefix ending at `vision_end_id`, with
        // divergent suffixes of varying lengths (3, 6, 2).
        let shared = [10u32, 11, 12, 13, 14, vision_end_id];
        let mk_seq = |suf: &[u32]| {
            let mut v = shared.to_vec();
            v.extend_from_slice(suf);
            v
        };
        let tasks = vec![
            ArGenTask {
                id: "p0".to_string(),
                input_ids: mk_seq(&[21, 22, 23]),
                media: None,
                media_key: Some("doc-page-1".to_string()),
                grammar: GrammarSpec::JsonSchema(json!({ "type": "boolean" })),
                max_new_tokens: 6,
                stop_token_ids: vec![eos_id],
            },
            ArGenTask {
                id: "p1".to_string(),
                input_ids: mk_seq(&[24, 25, 26, 27, 28, 29]),
                media: None,
                media_key: Some("doc-page-1".to_string()),
                grammar: GrammarSpec::JsonSchema(json!({ "type": "number" })),
                max_new_tokens: 6,
                stop_token_ids: vec![eos_id],
            },
            ArGenTask {
                id: "p2".to_string(),
                input_ids: mk_seq(&[31, 32]),
                media: None,
                media_key: Some("doc-page-1".to_string()),
                grammar: GrammarSpec::JsonSchema(json!({ "type": "string" })),
                max_new_tokens: 6,
                stop_token_ids: vec![eos_id],
            },
        ];

        let cached = {
            let mut engine = ClairicEngine::new(
                &mut inner,
                EngineOptions {
                    prefix_cache: true,
                    ..EngineOptions::default()
                },
            );
            let out = engine
                .generate_batch(&tasks, &score_ids, None)
                .expect("cached generate_batch");
            assert_eq!(
                engine.prefix_caches.len(),
                1,
                "expected 1 cached shared prefix snapshot"
            );
            out
        };

        let uncached = {
            let mut engine = ClairicEngine::new(
                &mut inner,
                EngineOptions {
                    prefix_cache: false,
                    ..EngineOptions::default()
                },
            );
            let out = engine
                .generate_batch(&tasks, &score_ids, None)
                .expect("uncached generate_batch");
            assert!(engine.prefix_caches.is_empty());
            out
        };

        for i in 0..tasks.len() {
            assert_eq!(
                cached[i].output_ids, uncached[i].output_ids,
                "row {i} cached vs uncached tokens diverged"
            );
            assert_eq!(cached[i].text, uncached[i].text);
            assert_eq!(cached[i].finish_reason, uncached[i].finish_reason);
            for ((clp, ctok), (ulp, utok)) in cached[i]
                .output_token_logprobs
                .iter()
                .zip(uncached[i].output_token_logprobs.iter())
            {
                assert_eq!(ctok, utok);
                assert!(
                    (clp - ulp).abs() < 1e-4,
                    "row {i} logprob diff cached={clp} vs uncached={ulp}"
                );
            }
        }
    }

    #[test]
    fn ar_engine_read_typed_end_to_end() {
        let (mut inner, _, _) = synthetic_clairic_inner();
        let tok = inner.tokenizer.as_ref().unwrap().inner().clone();

        let record = json!({
            "input": "Invoice #42 dated 2025-05-01 at 14:30:00 for Acme Corp. Total: $123.45.",
            "fields": {
                "vendor": {"type": "text", "instructions": "Company name"},
                "total": {"type": "number", "decimals": 2},
                "date": {"type": "date"},
                "clock": {"type": "time"},
                "eta": {"type": "estimate", "min": 0, "max": 500},
                "tags": {"type": "list", "items": "text"}
            },
            "_meta": {"construction": "evidence-v1"}
        });
        let opts = EncodeRecordOptions {
            max_length: 4096,
            answer_slots: true,
            text_answer_slots: true,
            ..EncodeRecordOptions::default()
        };
        let encoded = encode_record(&tok, &record, &opts)
            .expect("encode_record")
            .expect("record fits");
        assert!(
            encoded.fields.iter().any(|f| !f.answer_slots.is_empty()),
            "test record must exercise answer_slots -> lm_head"
        );

        let preds = {
            let mut engine = ClairicEngine::new(
                &mut inner,
                EngineOptions {
                    // Use a small segment size to also exercise multi-segment prefill.
                    segment_tokens: 64,
                    ..EngineOptions::default()
                },
            );
            engine
                .read_typed(&encoded, None, None)
                .expect("read_typed succeeded")
        };

        for field in ["vendor", "total", "date", "clock", "eta", "tags"] {
            let obj = preds
                .get(field)
                .and_then(Value::as_object)
                .unwrap_or_else(|| panic!("missing decoded prediction object for field {field}"));
            assert!(
                obj.contains_key("value"),
                "field {field} missing 'value': {obj:?}"
            );
            if field == "eta" {
                assert!(
                    obj.contains_key("p10")
                        && obj.contains_key("p90")
                        && obj.contains_key("quantiles"),
                    "estimate field {field} missing distribution keys: {obj:?}"
                );
            } else {
                assert!(
                    obj.contains_key("confidence"),
                    "field {field} missing 'confidence': {obj:?}"
                );
            }
        }
    }
}
