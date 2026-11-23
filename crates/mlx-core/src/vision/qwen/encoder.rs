/// Qwen Vision Encoder
///
/// Assembles existing shared vision components (VisionAttention, VisionMLP,
/// VisionEncoderLayer, SpatialProjector, VisionRotaryEmbedding) into a full
/// vision encoder for Qwen VLMs.
///
/// Architecture: patch_embed → + pos_embed → 27 blocks with 2D RoPE → merger
use crate::array::MxArray;
use crate::vision::embeddings::PatchEmbedding;
#[cfg(test)]
use crate::vision::encoder::VisionAttention;
use crate::vision::encoder::{VisionEncoderLayer, build_vision_segment_boundaries};
use crate::vision::projector::SpatialProjector;
use crate::vision::rope_vision::{
    VisionRotaryEmbedding, compute_rot_pos_emb_2d, precompute_rotary_cos_sin,
};
use napi::bindgen_prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

/// Minimum patch count before materializing and trimming the allocator cache
/// after every vision transformer block on unified-memory or >=24 GiB CUDA
/// GPUs. Typical images (e.g. Clairic `max_pixels = 1.6M` → at most 6,250
/// patches, 53.8 MB BF16 FC1) fuse cleanly into one lazy graph evaluated by
/// the caller, avoiding 27 GPU-CPU syncs and allocator pool trims per image.
const CUDA_VISION_LAYER_BARRIER_MIN_PATCHES_LARGE: i64 = 32_768;
/// Lower per-layer materialization threshold for smaller (<24 GiB) discrete
/// CUDA GPUs to keep peak vision activation memory bounded.
const CUDA_VISION_LAYER_BARRIER_MIN_PATCHES_SMALL: i64 = 8_192;
/// Device memory threshold (24 GiB) selecting the larger CUDA vision layer
/// barrier cutoff.
const CUDA_VISION_LARGE_MEM_BYTES: u64 = 24u64 * 1024 * 1024 * 1024;

/// Decide whether to materialize the vision residual stream and clear the
/// allocator cache after every vision transformer layer.
///
/// Metal/CPU always materialize per layer to avoid peak-memory regressions on
/// macOS unified memory. CUDA defers per-layer barriers below the device
/// memory-scaled patch threshold so typical images run in one lazy graph.
fn should_materialize_vision_per_layer(
    total_seq_len: i64,
    cuda_ops: bool,
    is_unified_memory: bool,
    total_device_bytes: u64,
) -> bool {
    if !cuda_ops {
        return true;
    }
    let threshold = if is_unified_memory || total_device_bytes >= CUDA_VISION_LARGE_MEM_BYTES {
        CUDA_VISION_LAYER_BARRIER_MIN_PATCHES_LARGE
    } else {
        CUDA_VISION_LAYER_BARRIER_MIN_PATCHES_SMALL
    };
    total_seq_len > threshold
}

/// Qwen Vision Encoder configuration
#[derive(Debug, Clone)]
pub struct QwenVisionConfig {
    pub hidden_size: i32,        // 1152
    pub intermediate_size: i32,  // 4304
    pub num_heads: i32,          // 16
    pub num_layers: i32,         // 27
    pub patch_size: i32,         // 16
    pub spatial_merge_size: i32, // 2
    pub image_size: i32,         // 768 (reference)
    pub out_hidden_size: i32,    // 4096 (matches text hidden_size)
}

impl Default for QwenVisionConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1152,
            intermediate_size: 4304,
            num_heads: 16,
            num_layers: 27,
            patch_size: 16,
            spatial_merge_size: 2,
            image_size: 768,
            out_hidden_size: 4096,
        }
    }
}

/// Qwen Vision Encoder
///
/// Processes images into vision features compatible with the text model.
pub struct QwenVisionEncoder {
    config: QwenVisionConfig,
    /// Patch embedding: Conv2d extracting patches from images
    patch_embed: Arc<PatchEmbedding>,
    /// Learned position embeddings [num_pos, hidden_size]
    pos_embed: Option<Arc<MxArray>>,
    /// Transformer encoder layers
    layers: Vec<Arc<VisionEncoderLayer>>,
    /// 2D rotary position embeddings
    rotary_pos_emb: Arc<VisionRotaryEmbedding>,
    /// Spatial projector: merges patches and projects to text model dimension
    merger: Option<Arc<SpatialProjector>>,
}

impl QwenVisionEncoder {
    /// Create a new vision encoder (layers and weights added separately).
    pub fn new(config: QwenVisionConfig) -> Result<Self> {
        let head_dim = config.hidden_size / config.num_heads;
        let rotary_pos_emb = VisionRotaryEmbedding::new((head_dim / 2) as u32, None);

        // Create a dummy patch embedding (weights loaded later)
        let patch_size = config.patch_size as u32;
        let dummy_weight = MxArray::zeros(
            &[
                config.hidden_size as i64,
                patch_size as i64,
                patch_size as i64,
                3,
            ],
            None,
        )?;
        let patch_embed = PatchEmbedding::new(patch_size, &dummy_weight, None)?;

        Ok(Self {
            config,
            patch_embed: Arc::new(patch_embed),
            pos_embed: None,
            layers: Vec::new(),
            rotary_pos_emb: Arc::new(rotary_pos_emb),
            merger: None,
        })
    }

    /// Widths used by the vision transformer's largest per-patch
    /// intermediates. The request planner combines these with the live input
    /// dtype and allocator headroom to bound cache-miss microbatches.
    pub(crate) fn memory_widths(&self) -> (u64, u64, u64) {
        (
            u64::try_from(self.config.hidden_size).unwrap_or(0),
            u64::try_from(self.config.intermediate_size).unwrap_or(0),
            u64::try_from(self.config.out_hidden_size).unwrap_or(0),
        )
    }

    /// Set patch embedding weights (and optional Conv bias)
    pub fn set_patch_embed(&mut self, weight: &MxArray, bias: Option<&MxArray>) -> Result<()> {
        self.patch_embed = Arc::new(PatchEmbedding::new(
            self.config.patch_size as u32,
            weight,
            bias,
        )?);
        Ok(())
    }

    /// Set position embedding weights
    pub fn set_pos_embed(&mut self, weight: &MxArray) {
        self.pos_embed = Some(Arc::new(weight.clone()));
    }

    /// Add an encoder layer
    pub fn add_layer(&mut self, layer: &VisionEncoderLayer) {
        self.layers.push(Arc::new(layer.clone()));
    }

    /// Set the merger (spatial projector)
    pub fn set_merger(&mut self, merger: SpatialProjector) {
        self.merger = Some(Arc::new(merger));
    }

    /// Extract all vision encoder parameters as a flat map.
    ///
    /// Keys use the `visual.` prefix so they round-trip through save/load
    /// (load detects keys starting with `visual.`).
    /// Internal keys mirror the load convention: `patch_embed.proj.weight`,
    /// `pos_embed.weight`, `blocks.{i}.attn.qkv.weight`, etc.
    pub fn get_parameters(&self) -> HashMap<String, MxArray> {
        let mut params = HashMap::new();

        // Patch embedding
        params.insert(
            "visual.patch_embed.proj.weight".to_string(),
            self.patch_embed.weight(),
        );
        if let Some(b) = self.patch_embed.bias() {
            params.insert("visual.patch_embed.proj.bias".to_string(), b);
        }

        // Position embedding
        if let Some(ref pe) = self.pos_embed {
            params.insert("visual.pos_embed.weight".to_string(), pe.as_ref().clone());
        }

        // Encoder layers
        for (i, layer) in self.layers.iter().enumerate() {
            let prefix = format!("visual.blocks.{}", i);

            // Attention
            let attn = layer.self_attn();
            params.insert(format!("{}.attn.qkv.weight", prefix), attn.get_qkv_weight());
            if let Some(b) = attn.get_qkv_bias() {
                params.insert(format!("{}.attn.qkv.bias", prefix), b);
            }
            params.insert(
                format!("{}.attn.proj.weight", prefix),
                attn.get_out_proj_weight(),
            );
            if let Some(b) = attn.get_out_proj_bias() {
                params.insert(format!("{}.attn.proj.bias", prefix), b);
            }

            // MLP
            let mlp = layer.mlp();
            params.insert(
                format!("{}.mlp.linear_fc1.weight", prefix),
                mlp.get_fc1_weight(),
            );
            if let Some(b) = mlp.get_fc1_bias() {
                params.insert(format!("{}.mlp.linear_fc1.bias", prefix), b);
            }
            params.insert(
                format!("{}.mlp.linear_fc2.weight", prefix),
                mlp.get_fc2_weight(),
            );
            if let Some(b) = mlp.get_fc2_bias() {
                params.insert(format!("{}.mlp.linear_fc2.bias", prefix), b);
            }

            // Layer norms
            let ln1 = layer.layer_norm1();
            params.insert(format!("{}.norm1.weight", prefix), ln1.get_weight());
            params.insert(format!("{}.norm1.bias", prefix), ln1.get_bias());

            let ln2 = layer.layer_norm2();
            params.insert(format!("{}.norm2.weight", prefix), ln2.get_weight());
            params.insert(format!("{}.norm2.bias", prefix), ln2.get_bias());
        }

        // Merger (spatial projector)
        if let Some(ref merger) = self.merger {
            let norm = merger.pre_norm();
            params.insert("visual.merger.norm.weight".to_string(), norm.get_weight());
            params.insert("visual.merger.norm.bias".to_string(), norm.get_bias());

            let l1 = merger.linear_1();
            params.insert(
                "visual.merger.linear_fc1.weight".to_string(),
                l1.get_weight(),
            );
            if let Some(b) = l1.get_bias() {
                params.insert("visual.merger.linear_fc1.bias".to_string(), b);
            }

            let l2 = merger.linear_2();
            params.insert(
                "visual.merger.linear_fc2.weight".to_string(),
                l2.get_weight(),
            );
            if let Some(b) = l2.get_bias() {
                params.insert("visual.merger.linear_fc2.bias".to_string(), b);
            }
        }

        params
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `hidden_states` - Pixel patches [batch, seq_len, channels, patch_h, patch_w]
    /// * `grid_thw` - Grid dimensions [num_images, 3] with [t, h, w] per image
    ///
    /// # Returns
    /// Vision features [total_merged_patches, out_hidden_size]
    pub fn forward(&self, hidden_states: &MxArray, grid_thw: &MxArray) -> Result<MxArray> {
        let shape = hidden_states.shape()?;
        let batch = shape[0];
        let seq_len = shape[1];
        let channels = shape[2];
        let patch_h = shape[3];
        let patch_w = shape[4];

        // Reshape and transpose for Conv2d: [N, C, H, W] → [N, H, W, C] (NHWC)
        let flat_input = hidden_states.reshape(&[batch * seq_len, channels, patch_h, patch_w])?;
        let mut flat_input = flat_input.transpose(Some(&[0, 2, 3, 1]))?;
        if crate::accelerator::cuda_ops_available() {
            let weight_dtype = self.patch_embed.weight().dtype()?;
            if matches!(
                weight_dtype,
                crate::array::DType::BFloat16 | crate::array::DType::Float16
            ) && flat_input.dtype()? != weight_dtype
            {
                flat_input = flat_input.astype(weight_dtype)?;
            }
        }

        // Patch embedding
        let patch_embeds = self.patch_embed.forward(&flat_input)?;
        let embed_dim = patch_embeds.shape()?[2];
        let embeddings = patch_embeds.reshape(&[batch, seq_len, embed_dim])?;
        let embeddings = embeddings.squeeze(Some(&[0]))?;

        // Add position embeddings per image (with bilinear interpolation)
        let grid_data = grid_thw.to_int32()?;
        let num_images = grid_thw.shape()?[0] as usize;

        let mut h = if let Some(ref pos_embed) = self.pos_embed {
            // Add interpolated position embeddings
            let mut result_parts: Vec<MxArray> = Vec::new();
            let mut start = 0i64;

            for img_idx in 0..num_images {
                let t = grid_data[img_idx * 3] as i64;
                let h_dim = grid_data[img_idx * 3 + 1] as i64;
                let w_dim = grid_data[img_idx * 3 + 2] as i64;
                let end = start + t * h_dim * w_dim;

                let img_embeddings = embeddings.slice_axis(0, start, end)?;

                // Interpolate position embeddings for this image's grid size
                let pos = self.interpolate_pos_embed(pos_embed, h_dim as u32, w_dim as u32)?;
                let pos = if t > 1 {
                    let pos_refs: Vec<&MxArray> = (0..t).map(|_| &pos).collect();
                    MxArray::concatenate_many(pos_refs, Some(0))?
                } else {
                    pos
                };
                let embed_dtype = img_embeddings.dtype()?;
                let pos = if pos.dtype()? != embed_dtype {
                    pos.astype(embed_dtype)?
                } else {
                    pos
                };

                let img_with_pos = img_embeddings.add(&pos)?;
                result_parts.push(img_with_pos);
                start = end;
            }

            if result_parts.len() == 1 {
                result_parts.remove(0)
            } else {
                let refs: Vec<&MxArray> = result_parts.iter().collect();
                MxArray::concatenate_many(refs, Some(0))?
            }
        } else {
            embeddings
        };

        // Compute 2D rotary position embeddings and hoist (cos, sin) once for all layers
        let rotary_pos_emb = compute_rot_pos_emb_2d(&self.rotary_pos_emb, grid_thw)?;
        let (rot_cos, rot_sin) = precompute_rotary_cos_sin(&rotary_pos_emb)?;

        // Build cumulative image/frame boundaries. Qwen vision attention is
        // block-diagonal across these segments. Keep the boundaries on the CPU
        // and dispatch one fused SDPA per segment rather than allocating a
        // `[total_patches, total_patches]` additive mask.
        let total_seq_len = h.shape()?[0];
        let segment_boundaries = build_vision_segment_boundaries(grid_thw, total_seq_len)?;

        // Forward through encoder layers. MLX builds lazy graphs, so leaving all
        // 27 blocks attached to the final merger output retains every layer's
        // QKV and expanded MLP activations until the caller evaluates the final
        // feature tensor. On Metal/CPU, keep per-layer materialization enabled
        // unconditionally to avoid macOS peak-memory regressions. On CUDA, defer
        // per-layer materialization below the device memory-scaled patch cutoff
        // (32,768 on unified / >=24 GiB GPUs, 8,192 on smaller discrete GPUs)
        // so typical images fuse into one lazy graph evaluated by the caller.
        let cuda_ops = crate::accelerator::cuda_ops_available();
        let caps = crate::accelerator::caps();
        let materialize_per_layer = should_materialize_vision_per_layer(
            total_seq_len,
            cuda_ops,
            caps.is_unified_memory,
            caps.total_device_bytes,
        );
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            h = layer.forward_segmented_with_cos_sin(
                &h,
                &segment_boundaries,
                Some((&rot_cos, &rot_sin)),
            )?;
            if materialize_per_layer {
                let context = format!("qwen3_5_vision_layer_{layer_idx}");
                MxArray::eval_arrays_with_context(&[&h], &context).map_err(|error| {
                    Error::from_reason(format!(
                        "Qwen vision layer {layer_idx} materialization failed: {error}"
                    ))
                })?;
                crate::array::clear_cache();
                tracing::debug!(
                    target: "mlx_core::inference",
                    event = "vision_layer_materialized",
                    layer = layer_idx,
                    patch_count = total_seq_len,
                    "Qwen vision layer residual materialized"
                );
            }
        }

        // Spatial projector (reduces token count by merge_size^2)
        let merger = self
            .merger
            .as_ref()
            .ok_or_else(|| Error::from_reason("Vision encoder merger not initialized"))?;
        merger.forward(&h, grid_thw)
    }

    /// Interpolate position embeddings to target grid size
    fn interpolate_pos_embed(
        &self,
        pos_embed: &MxArray,
        target_h: u32,
        target_w: u32,
    ) -> Result<MxArray> {
        let pos_shape = pos_embed.shape()?;
        let num_positions = pos_shape[0];
        let embed_dim = pos_shape[1];

        let base_size = (num_positions as f64).sqrt() as i64;
        let target_patches = (target_h * target_w) as i64;

        if base_size * base_size == num_positions
            && target_h as i64 == base_size
            && target_w as i64 == base_size
        {
            return Ok(pos_embed.clone());
        }

        // Reshape to 2D grid and interpolate
        let pos_2d = pos_embed.reshape(&[base_size, base_size, embed_dim])?;
        let interpolated = crate::vision::interpolate::bilinear_interpolate(
            &pos_2d,
            target_h as i64,
            target_w as i64,
        )?;
        interpolated.reshape(&[target_patches, embed_dim])
    }
}

impl Clone for QwenVisionEncoder {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            patch_embed: self.patch_embed.clone(),
            pos_embed: self.pos_embed.clone(),
            layers: self.layers.clone(),
            rotary_pos_emb: self.rotary_pos_emb.clone(),
            merger: self.merger.clone(),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::nn::LayerNorm;
    use crate::vision::encoder::{BUILD_ATTENTION_MASK_CALLS, VisionMLP};

    fn random_array(shape: &[i64]) -> MxArray {
        let size: usize = shape.iter().map(|&s| s as usize).product();
        let data: Vec<f32> = (0..size)
            .map(|i| ((i % 100) as f32 - 50.0) / 100.0)
            .collect();
        MxArray::from_float32(&data, shape).unwrap()
    }

    /// Build a tiny encoder (small dims, `num_layers` transformer blocks, no
    /// spatial merge) sufficient to exercise a full `forward()` call.
    pub(crate) fn tiny_encoder(num_layers: usize) -> QwenVisionEncoder {
        let hidden = 8i64;
        let heads = 2u32;
        let patch = 2i64;
        let intermediate = 16i64;

        let config = QwenVisionConfig {
            hidden_size: hidden as i32,
            intermediate_size: intermediate as i32,
            num_heads: heads as i32,
            num_layers: num_layers as i32,
            patch_size: patch as i32,
            spatial_merge_size: 1,
            image_size: 4,
            out_hidden_size: hidden as i32,
        };

        let mut encoder = QwenVisionEncoder::new(config).unwrap();

        let patch_weight = random_array(&[hidden, patch, patch, 3]);
        encoder.set_patch_embed(&patch_weight, None).unwrap();

        for _ in 0..num_layers {
            let ln1 = LayerNorm::new(hidden as u32, None).unwrap();
            let ln2 = LayerNorm::new(hidden as u32, None).unwrap();
            let qkv_weight = random_array(&[hidden * 3, hidden]);
            let out_weight = random_array(&[hidden, hidden]);
            let attn =
                VisionAttention::new(hidden as u32, heads, &qkv_weight, None, &out_weight, None)
                    .unwrap();
            let fc1_weight = random_array(&[intermediate, hidden]);
            let fc2_weight = random_array(&[hidden, intermediate]);
            let mlp = VisionMLP::new(&fc1_weight, None, &fc2_weight, None).unwrap();
            let layer = VisionEncoderLayer::new(&ln1, &ln2, &attn, &mlp);
            encoder.add_layer(&layer);
        }

        let merger = SpatialProjector::new(
            1,
            &random_array(&[hidden]),
            &random_array(&[hidden]),
            &random_array(&[hidden, hidden]),
            &random_array(&[hidden]),
            &random_array(&[hidden, hidden]),
            &random_array(&[hidden]),
        )
        .unwrap();
        encoder.set_merger(merger);

        encoder
    }

    #[test]
    fn test_forward_segments_multi_image_attention_without_dense_mask() {
        let num_layers = 3;
        let encoder = tiny_encoder(num_layers);

        // Two images with 4 and 2 patches respectively. The production failure
        // came from concatenating several images into one global quadratic
        // mask; this path must not build that mask at all.
        let hidden_states = random_array(&[1, 6, 3, 2, 2]);
        let grid_thw = MxArray::from_int32(&[1, 2, 2, 1, 1, 2], &[2, 3]).unwrap();

        BUILD_ATTENTION_MASK_CALLS.with(|c| c.set(0));
        let output = encoder.forward(&hidden_states, &grid_thw).unwrap();
        let calls = BUILD_ATTENTION_MASK_CALLS.with(|c| c.get());

        assert_eq!(
            calls, 0,
            "segmented Qwen3.5 attention must not construct a global dense mask \
             across any of the {num_layers} encoder layers"
        );

        let shape: Vec<i64> = output.shape().unwrap().as_ref().to_vec();
        assert_eq!(shape, vec![6, 8]);
    }

    #[test]
    fn test_combined_and_per_image_microbatches_are_numerically_equivalent() {
        let encoder = tiny_encoder(2);
        // Distinct grids, including t=2 for the second image, exercise both
        // image ordering and per-frame attention boundaries.
        let hidden_states = random_array(&[1, 8, 3, 2, 2]);
        let combined_grid = MxArray::from_int32(&[1, 2, 2, 2, 1, 2], &[2, 3]).unwrap();
        let combined = encoder.forward(&hidden_states, &combined_grid).unwrap();

        let first_pixels = hidden_states.slice_axis(1, 0, 4).unwrap();
        let second_pixels = hidden_states.slice_axis(1, 4, 8).unwrap();
        let first_grid = MxArray::from_int32(&[1, 2, 2], &[1, 3]).unwrap();
        let second_grid = MxArray::from_int32(&[2, 1, 2], &[1, 3]).unwrap();
        let first = encoder.forward(&first_pixels, &first_grid).unwrap();
        let second = encoder.forward(&second_pixels, &second_grid).unwrap();
        let microbatched = MxArray::concatenate_many(vec![&first, &second], Some(0)).unwrap();

        MxArray::eval_arrays(&[&combined, &microbatched]).unwrap();
        let combined_values = combined.to_float32().unwrap().to_vec();
        let microbatched_values = microbatched.to_float32().unwrap().to_vec();
        assert_eq!(combined_values.len(), microbatched_values.len());
        let max_abs_diff = combined_values
            .iter()
            .zip(microbatched_values.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs_diff <= 1e-4,
            "combined/per-image vision output diverged: max abs diff {max_abs_diff}"
        );
    }

    #[test]
    fn vision_layer_barrier_gating_respects_backend_and_cuda_memory() {
        let gib = 1024u64 * 1024 * 1024;
        // Metal / CPU always materialize per layer regardless of patch count or memory.
        assert!(should_materialize_vision_per_layer(
            1,
            false,
            true,
            128 * gib
        ));
        assert!(should_materialize_vision_per_layer(
            4096,
            false,
            false,
            80 * gib
        ));

        // CUDA unified memory (e.g. GB10 Spark) uses 32,768 threshold.
        assert!(!should_materialize_vision_per_layer(
            32_768,
            true,
            true,
            16 * gib
        ));
        assert!(should_materialize_vision_per_layer(
            32_769,
            true,
            true,
            16 * gib
        ));

        // CUDA discrete >= 24 GiB uses 32,768 threshold.
        assert!(!should_materialize_vision_per_layer(
            32_768,
            true,
            false,
            24 * gib
        ));
        assert!(should_materialize_vision_per_layer(
            32_769,
            true,
            false,
            24 * gib
        ));

        // CUDA discrete < 24 GiB uses 8,192 threshold.
        assert!(!should_materialize_vision_per_layer(
            8_192,
            true,
            false,
            16 * gib
        ));
        assert!(should_materialize_vision_per_layer(
            8_193,
            true,
            false,
            16 * gib
        ));
    }
}
