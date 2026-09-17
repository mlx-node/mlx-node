//! Image placeholders and three-axis positions shared by Qwen chat models.
use super::processing::{QwenImageProcessor, merged_image_token_count};
use crate::array::MxArray;
use crate::engine::vision::extract_images_from_messages;
use crate::tokenizer::ChatMessage;
use napi::bindgen_prelude::*;
use std::sync::Arc;

pub(crate) const IMAGE_TOKEN_ID: i32 = 248056;

/// Compute the per-image merged-token count from a processed grid_thw
/// array. Each entry is the number of `IMAGE_TOKEN_ID` slots that image
/// must occupy in the prompt so the vision embeddings align 1:1 with the
/// corresponding token positions.
pub(crate) fn compute_image_token_counts_per_image(
    grid: &MxArray,
    spatial_merge_size: i32,
) -> Result<Vec<usize>> {
    grid.eval();
    let grid_data = grid.to_int32()?;
    let mut counts = Vec::with_capacity(grid_data.len() / 3);
    for i in 0..(grid_data.len() / 3) {
        let t = grid_data[i * 3];
        let h = grid_data[i * 3 + 1];
        let w = grid_data[i * 3 + 2];
        counts.push(merged_image_token_count(t, h, w, spatial_merge_size)?);
    }
    Ok(counts)
}

/// Return the exact length produced by [`inject_image_placeholders`] without
/// allocating the expanded token vector.
pub(crate) fn expanded_image_prompt_len(
    tokens: &[u32],
    per_image_token_counts: &[usize],
) -> Result<usize> {
    let total = per_image_token_counts
        .iter()
        .try_fold(0usize, |sum, count| {
            sum.checked_add(*count)
                .ok_or_else(|| Error::from_reason("expanded image prompt length overflow"))
        })?;
    if total == 0 {
        return Ok(tokens.len());
    }

    let existing = tokens
        .iter()
        .filter(|&&token| token == IMAGE_TOKEN_ID as u32)
        .count();
    if existing == per_image_token_counts.len() {
        return tokens
            .len()
            .checked_add(total)
            .and_then(|len| len.checked_sub(existing))
            .ok_or_else(|| Error::from_reason("expanded image prompt length overflow"));
    }
    if existing == total {
        return Ok(tokens.len());
    }

    Err(image_placeholder_shape_error(
        existing,
        per_image_token_counts.len(),
        total,
    ))
}

/// CPU-only prompt planner shared by Qwen chat wrappers.
///
/// It reads encoded image dimensions and applies the loaded Qwen processor's
/// smart-resize geometry, but never creates normalized pixel tensors, MLX
/// arrays, vision features, or KV state.
pub(crate) fn plan_expanded_image_prompt_len(
    image_processor: &QwenImageProcessor,
    spatial_merge_size: i32,
    tokens: &[u32],
    images: &[Vec<u8>],
) -> Result<usize> {
    let image_refs: Vec<&[u8]> = images.iter().map(Vec::as_slice).collect();
    let counts = image_processor.plan_merged_token_counts(&image_refs, spatial_merge_size)?;
    expanded_image_prompt_len(tokens, &counts)
}

/// Ensure the tokenized prompt contains the right number of
/// `IMAGE_TOKEN_ID` placeholders — one per vision patch, in the order
/// produced by the chat template.
///
/// Two input shapes are accepted:
///
/// 1. **Template emitted one `<|image_pad|>` per image** (the proper
///    Qwen VLM shape, produced by
///    `tokenizer::serialize_message_for_jinja` when the user turn
///    carries images). Each placeholder is expanded in-place to its
///    image's grid count. This keeps the vision tokens inside the user
///    turn — `get_rope_index` builds correct M-RoPE positions and the
///    model attends to the image in-context.
///
/// 2. **Template already emitted the fully expanded count** (non-Qwen
///    templates that inline the full patch run). Pass through unchanged.
///
///
/// Missing or mismatched markers are rejected. The checkpoint's chat
/// template owns marker placement; inserting a fallback run after BOS would
/// move vision tokens outside the user turn and produce invalid M-RoPE
/// positions.
pub(crate) fn inject_image_placeholders(
    tokens: &[u32],
    per_image_token_counts: &[usize],
) -> Result<Vec<u32>> {
    let expanded_len = expanded_image_prompt_len(tokens, per_image_token_counts)?;
    let existing = tokens
        .iter()
        .filter(|&&t| t == IMAGE_TOKEN_ID as u32)
        .count();
    if per_image_token_counts.iter().all(|&n| n == 0) || existing != per_image_token_counts.len() {
        return Ok(tokens.to_vec());
    }
    let mut expanded = Vec::with_capacity(expanded_len);
    let mut counts = per_image_token_counts.iter();
    for &token in tokens {
        if token == IMAGE_TOKEN_ID as u32 {
            // The length validator established one placeholder per image.
            let count = *counts.next().ok_or_else(|| {
                Error::from_reason("image placeholder has no matching image token count")
            })?;
            expanded.extend(std::iter::repeat_n(token, count));
        } else {
            expanded.push(token);
        }
    }
    Ok(expanded)
}

fn image_placeholder_shape_error(
    existing: usize,
    image_count: usize,
    expanded_count: usize,
) -> Error {
    if existing == 0 {
        Error::from_reason(format!(
            "model chat template emitted no image placeholder tokens for {image_count} image(s); \
expected {image_count} unexpanded marker(s) or {expanded_count} already-expanded marker(s)"
        ))
    } else {
        Error::from_reason(format!(
            "model chat template emitted {existing} image placeholder token(s) for {image_count} \
image(s); expected {image_count} unexpanded marker(s) or {expanded_count} already-expanded marker(s)"
        ))
    }
}

/// Compute M-RoPE position IDs for VLM
///
/// Text tokens get sequential positions [0, 1, 2, ...].
/// Image tokens get 2D spatial positions based on grid_thw.
///
/// Returns (position_ids [3, B, T], rope_deltas)
pub(crate) fn get_rope_index(
    input_ids: &MxArray,
    image_grid_thw: Option<&MxArray>,
    spatial_merge_size: i32,
    image_token_id: i32,
) -> Result<(MxArray, i64)> {
    let shape = input_ids.shape()?;
    let batch_size = shape[0];
    let seq_len = shape[1];

    // If no images, use simple sequential positions
    let Some(grid_thw) = image_grid_thw else {
        let pos = MxArray::arange(0.0, seq_len as f64, Some(1.0), None)?;
        let pos = pos.reshape(&[1, 1, seq_len])?;
        let position_ids = MxArray::tile(&pos, &[3, batch_size as i32, 1])?;
        return Ok((position_ids, 0));
    };
    let input_ids_data = input_ids.to_int32()?;
    grid_thw.eval();
    let grid_data = grid_thw.to_int32()?;

    let mut all_position_ids: Vec<Vec<i64>> = vec![Vec::new(); 3];

    for batch_idx in 0..batch_size as usize {
        let start = batch_idx * seq_len as usize;
        let end = start + seq_len as usize;
        let batch_tokens: Vec<i32> = input_ids_data[start..end].to_vec();

        // History can contain image runs separated by text. Retain every
        // gap when pairing the expanded placeholders with their image grids.
        let mut image_runs: Vec<(usize, usize)> = Vec::new();
        {
            let mut i = 0;
            while i < batch_tokens.len() {
                if batch_tokens[i] == image_token_id {
                    let start = i;
                    while i < batch_tokens.len() && batch_tokens[i] == image_token_id {
                        i += 1;
                    }
                    image_runs.push((start, i));
                } else {
                    i += 1;
                }
            }
        }

        if image_runs.is_empty() {
            for i in 0..seq_len {
                all_position_ids[0].push(i);
                all_position_ids[1].push(i);
                all_position_ids[2].push(i);
            }
            continue;
        }

        let num_images = grid_data.len() / 3;
        if num_images == 0 || grid_data.len() % 3 != 0 {
            return Err(Error::new(
                Status::InvalidArg,
                format!("grid_data must have 3N elements, got {}", grid_data.len()),
            ));
        }

        // Calculate token info for each image
        let mut image_token_info: Vec<(i64, i64, i64, usize)> = Vec::new();
        let mut total_expected_tokens = 0usize;

        for img_idx in 0..num_images {
            let t = grid_data[img_idx * 3] as i64;
            let h = grid_data[img_idx * 3 + 1] as i64;
            let w = grid_data[img_idx * 3 + 2] as i64;

            let llm_grid_t = t;
            let llm_grid_h = h / spatial_merge_size as i64;
            let llm_grid_w = w / spatial_merge_size as i64;
            let num_tokens = (llm_grid_t * llm_grid_h * llm_grid_w) as usize;

            image_token_info.push((llm_grid_t, llm_grid_h, llm_grid_w, num_tokens));
            total_expected_tokens += num_tokens;
        }

        let total_image_tokens: usize = image_runs.iter().map(|(s, e)| e - s).sum();
        if total_expected_tokens != total_image_tokens {
            return Err(Error::new(
                Status::GenericFailure,
                format!(
                    "Image token count mismatch: expected {} from grid, found {} in prompt",
                    total_expected_tokens, total_image_tokens,
                ),
            ));
        }

        // Two token layouts are valid here:
        //
        //  (a) N runs, one per image — the proper Qwen VLM shape after
        //      the tokenizer serialiser emits a `{type:"image"}` part
        //      per image and `inject_image_placeholders` expands each
        //      marker in place. Per-run length must match its grid.
        //
        //  (b) 1 big run whose length equals the grids' total — a
        //      checkpoint template may emit the fully expanded markers
        //      as one contiguous span. No text gap sits between images
        //      in this layout, so the position walk collapses consecutive
        //      sub-runs into one span without emitting interior text.
        //
        // We canonicalise both into a `per_image_offsets: Vec<(start,
        // grid_info)>` list of length `num_images` and feed it to the
        // position walk below. Any other shape is ambiguous (we'd have
        // to guess which grid goes with which run) — reject it.
        let per_image_offsets: Vec<(usize, (i64, i64, i64, usize))> = if image_runs.len()
            == num_images
        {
            // Case (a): validate per-run length, then pair by ordinal.
            for (run_idx, (run_start, run_end)) in image_runs.iter().enumerate() {
                let expected = image_token_info[run_idx].3;
                let actual = run_end - run_start;
                if expected != actual {
                    return Err(Error::new(
                        Status::GenericFailure,
                        format!(
                            "Image run {run_idx} has {actual} placeholder tokens but its grid expects {expected}",
                        ),
                    ));
                }
            }
            image_runs
                .iter()
                .zip(image_token_info.iter().copied())
                .map(|((start, _), info)| (*start, info))
                .collect()
        } else if image_runs.len() == 1 {
            // Case (b): already-expanded contiguous span — synthesise
            // per-image start offsets by walking `image_token_info`
            // lengths from the single run's start. Total was already
            // validated above.
            let big_start = image_runs[0].0;
            let mut offsets = Vec::with_capacity(num_images);
            let mut cursor = big_start;
            for info in image_token_info.iter().copied() {
                offsets.push((cursor, info));
                cursor += info.3;
            }
            offsets
        } else {
            return Err(Error::new(
                Status::GenericFailure,
                format!(
                    "Image run layout mismatch: prompt carries {} contiguous image-token runs but {} images \
                     were processed; expected either one run per image or a single contiguous fallback run \
                     containing every image's tokens.",
                    image_runs.len(),
                    num_images,
                ),
            ));
        };

        // End of the last image token in the token stream — everything
        // beyond is trailing text. For case (a) this is the last run's
        // end; for case (b) it's the shared run's end. In both cases
        // it equals the end of the validated non-empty `image_runs` list.
        let last_image_end = image_runs
            .last()
            .ok_or_else(|| Error::from_reason("image run validation lost its non-empty run"))?
            .1;

        // Emit positions by walking the sequence: text gap, image,
        // text gap, image, … final text gap. `current_pos` carries the
        // M-RoPE counter forward across both text and image segments so
        // every token gets a monotonically non-decreasing position id
        // in each axis. Synthesised case-(b) sub-runs sit back-to-back
        // so their text-gap loops iterate zero times between them —
        // the walk collapses naturally.
        let mut cursor: usize = 0;
        let mut current_pos: i64 = 0;

        for (run_start, info) in per_image_offsets.iter().copied() {
            // Text gap before this image run (zero-length for adjacent
            // case-(b) sub-runs after the first).
            for _ in cursor..run_start {
                all_position_ids[0].push(current_pos);
                all_position_ids[1].push(current_pos);
                all_position_ids[2].push(current_pos);
                current_pos += 1;
            }

            // Spatial positions for the image at this run
            let (llm_grid_t, llm_grid_h, llm_grid_w, count) = info;
            let image_base = current_pos;
            for t_idx in 0..llm_grid_t {
                for h_idx in 0..llm_grid_h {
                    for w_idx in 0..llm_grid_w {
                        all_position_ids[0].push(image_base + t_idx);
                        all_position_ids[1].push(image_base + h_idx);
                        all_position_ids[2].push(image_base + w_idx);
                    }
                }
            }
            let max_axis = std::cmp::max(
                llm_grid_t - 1,
                std::cmp::max(llm_grid_h - 1, llm_grid_w - 1),
            );
            current_pos = image_base + max_axis + 1;
            cursor = run_start + count;
        }

        // Trailing text after the last image (run in case (a), sub-run
        // end in case (b) — both resolve to `last_image_end`).
        debug_assert_eq!(cursor, last_image_end);
        let _ = last_image_end;
        for _ in cursor..seq_len as usize {
            all_position_ids[0].push(current_pos);
            all_position_ids[1].push(current_pos);
            all_position_ids[2].push(current_pos);
            current_pos += 1;
        }
    }

    // Convert to MxArray [3, batch, seq_len]
    let t_positions: Vec<i32> = all_position_ids[0].iter().map(|&x| x as i32).collect();
    let h_positions: Vec<i32> = all_position_ids[1].iter().map(|&x| x as i32).collect();
    let w_positions: Vec<i32> = all_position_ids[2].iter().map(|&x| x as i32).collect();

    let t_arr = MxArray::from_int32(&t_positions, &[batch_size, seq_len])?;
    let h_arr = MxArray::from_int32(&h_positions, &[batch_size, seq_len])?;
    let w_arr = MxArray::from_int32(&w_positions, &[batch_size, seq_len])?;

    let position_ids = MxArray::stack(vec![&t_arr, &h_arr, &w_arr], Some(0))?;

    // Decode offset must reference the GLOBAL max M-RoPE position, i.e. the max
    // over all three (t, h, w) axes — matching mlx-vlm's `llm_positions.max()`.
    // For an image the spatial (h, w) axes exceed the temporal one, so an
    // image-final prompt (no trailing text) would get a too-small delta if only
    // axis 0 were considered.
    let max_position = all_position_ids
        .iter()
        .flat_map(|axis| axis.iter().copied())
        .max()
        .unwrap_or(0);
    let rope_deltas = max_position + 1 - seq_len;

    Ok((position_ids, rope_deltas))
}

/// Merge image features into input embeddings at image token positions
pub(crate) fn merge_input_ids_with_image_features(
    image_token_id: i32,
    image_features: &MxArray,
    inputs_embeds: &MxArray,
    input_ids: &MxArray,
) -> Result<MxArray> {
    let input_shape = input_ids.shape()?;
    let batch_size = input_shape[0];

    let image_token = MxArray::scalar_int(image_token_id)?;
    let image_positions = input_ids.equal(&image_token)?;
    let inputs_embeds_shape = inputs_embeds.shape()?;
    let hidden_dim = inputs_embeds_shape[2];

    let mut batch_outputs: Vec<MxArray> = Vec::new();
    let mut feature_start_idx = 0i64;

    for batch_idx in 0..batch_size {
        let batch_mask = image_positions.slice_axis(0, batch_idx, batch_idx + 1)?;
        let batch_mask = batch_mask.squeeze(Some(&[0]))?;

        let mask_sum = batch_mask.sum(None, None)?;
        let num_positions = mask_sum.to_int32()?[0] as i64;

        if num_positions > 0 {
            let batch_features = image_features.slice_axis(
                0,
                feature_start_idx,
                feature_start_idx + num_positions,
            )?;

            let batch_embeds = inputs_embeds.slice_axis(0, batch_idx, batch_idx + 1)?;
            let batch_embeds = batch_embeds.squeeze(Some(&[0]))?;

            let mask_int = batch_mask.astype(crate::array::DType::Int32)?;
            let cumsum = mask_int.cumsum(0)?;

            let ones = MxArray::scalar_int(1)?;
            let feature_indices = cumsum.sub(&ones)?;
            let zeros =
                MxArray::zeros(&feature_indices.shape()?, Some(crate::array::DType::Int32))?;
            let feature_indices = batch_mask.where_(&feature_indices, &zeros)?;

            let gathered_features = batch_features.take(&feature_indices, 0)?;

            let mask_expanded = batch_mask.reshape(&[-1, 1])?;
            let mask_expanded =
                MxArray::broadcast_to(&mask_expanded, &[batch_mask.shape()?[0], hidden_dim])?;

            let batch_output = mask_expanded.where_(&gathered_features, &batch_embeds)?;
            batch_outputs.push(batch_output);
            feature_start_idx += num_positions;
        } else {
            let batch_embeds = inputs_embeds.slice_axis(0, batch_idx, batch_idx + 1)?;
            batch_outputs.push(batch_embeds.squeeze(Some(&[0]))?);
        }
    }

    let refs: Vec<&MxArray> = batch_outputs.iter().collect();
    MxArray::stack(refs, Some(0))
}

/// Optional admission budget, checked from encoded headers before image decode.
#[derive(Clone, Copy)]
pub(crate) struct ImageLimits {
    pub max_images: usize,
    pub max_pixels: u64,
    pub max_encoded_bytes: usize,
}
impl ImageLimits {
    pub fn validate(&self, images: &[Vec<u8>]) -> Result<()> {
        if images.len() > self.max_images {
            return Err(Error::from_reason(format!(
                "image count exceeds the {} image conversation budget",
                self.max_images
            )));
        }
        for bytes in images {
            if bytes.len() > self.max_encoded_bytes {
                return Err(Error::from_reason("image exceeds the encoded byte budget"));
            }
            let (w, h) = image::ImageReader::new(std::io::Cursor::new(bytes))
                .with_guessed_format()?
                .into_dimensions()
                .map_err(|e| Error::from_reason(e.to_string()))?;
            if u64::from(w) * u64::from(h) > self.max_pixels {
                return Err(Error::from_reason("image exceeds the pixel budget"));
            }
        }
        Ok(())
    }
}

/// Exact, non-mutating prompt planning for Qwen NAPI wrappers. Copy inputs
/// before entering the worker so no JS backing-store references cross threads.
pub(crate) async fn expanded_prompt_token_count(
    image_processor: Option<Arc<QwenImageProcessor>>,
    spatial_merge_size: i32,
    prompt_tokens: Uint32Array,
    messages: Vec<ChatMessage>,
    limits: Option<ImageLimits>,
) -> Result<u32> {
    let tokens = prompt_tokens.to_vec();
    let images = extract_images_from_messages(&messages);
    if images.is_empty() {
        return u32::try_from(tokens.len())
            .map_err(|_| Error::from_reason("rendered prompt token count exceeds u32"));
    }
    let image_processor = image_processor.ok_or_else(|| {
        Error::from_reason("cannot plan expanded image tokens: Qwen image processor is not loaded")
    })?;

    napi::bindgen_prelude::spawn_blocking(move || {
        if let Some(limits) = limits {
            limits.validate(&images)?;
        }
        let prompt_len =
            plan_expanded_image_prompt_len(&image_processor, spatial_merge_size, &tokens, &images)?;
        u32::try_from(prompt_len)
            .map_err(|_| Error::from_reason("expanded prompt token count exceeds u32"))
    })
    .await
    .map_err(|join_error| {
        Error::new(
            Status::GenericFailure,
            format!("Expanded prompt planning worker failed: {join_error}"),
        )
    })?
}

#[cfg(test)]
mod image_placeholder_tests;
#[cfg(test)]
mod rope_index_tests;
