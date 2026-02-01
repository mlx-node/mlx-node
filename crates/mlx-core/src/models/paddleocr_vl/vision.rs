/**
 * PaddleOCR-VL Vision Model
 *
 * Vision encoder for PaddleOCR-VL, including:
 * - Patch embeddings
 * - Vision transformer encoder
 * - Spatial projector for reducing token count
 */
use crate::array::MxArray;
use crate::models::paddleocr_vl::config::VisionConfig;
use crate::nn::LayerNorm;
use crate::vision::embeddings::{PatchEmbedding, VisionPositionEmbedding};
use crate::vision::encoder::VisionEncoderLayer;
use crate::vision::projector::SpatialProjector;
use crate::vision::rope_vision::VisionRotaryEmbedding;
use napi::bindgen_prelude::*;
use std::sync::Arc;

/// PaddleOCR Vision Embeddings (internal)
///
/// Combines patch embedding and position embedding for vision transformer.
///
/// Note: This is an internal implementation detail used by PaddleOCRVisionModel.
/// Not exposed to TypeScript - users interact with high-level VLModel API.
pub struct PaddleOCRVisionEmbeddings {
    patch_embedding: Arc<PatchEmbedding>,
    position_embedding: Arc<VisionPositionEmbedding>,
    patch_size: u32,
    embed_dim: u32,
}

impl PaddleOCRVisionEmbeddings {
    pub fn new(
        patch_size: u32,
        image_size: u32,
        embed_dim: u32,
        patch_weight: &MxArray,
        position_weight: &MxArray,
    ) -> Result<Self> {
        let patch_embedding = PatchEmbedding::new(patch_size, patch_weight)?;

        let num_patches = (image_size / patch_size).pow(2);
        let position_embedding =
            VisionPositionEmbedding::new(num_patches, embed_dim, position_weight)?;

        Ok(Self {
            patch_embedding: Arc::new(patch_embedding),
            position_embedding: Arc::new(position_embedding),
            patch_size,
            embed_dim,
        })
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `hidden_states` - Input pixel patches [batch, seq_len, channels, patch_h, patch_w]
    /// * `grid_thw` - Grid dimensions [num_images, 3] (temporal, height, width)
    ///
    /// # Returns
    /// * Embeddings [total_patches, embed_dim]
    pub fn forward(&self, hidden_states: &MxArray, grid_thw: &MxArray) -> Result<MxArray> {
        let shape = hidden_states.shape()?;

        // Input is [batch, seq_len, channels, patch_h, patch_w] (NCHW per patch)
        let batch = shape[0];
        let seq_len = shape[1];
        let channels = shape[2];
        let patch_h = shape[3];
        let patch_w = shape[4];

        // Reshape to [batch*seq, C, H, W] then transpose to [batch*seq, H, W, C] for Conv2d
        let flat_input = hidden_states.reshape(&[batch * seq_len, channels, patch_h, patch_w])?;
        let flat_input = flat_input.transpose(Some(&[0, 2, 3, 1]))?; // NCHW -> NHWC

        // Apply patch embedding conv
        let patch_embeds = self.patch_embedding.forward(&flat_input)?;

        // Reshape back: [batch, seq_len, embed_dim]
        let embed_dim = patch_embeds.shape()?[2];
        let embeddings = patch_embeds.reshape(&[batch, seq_len, embed_dim])?;

        // Squeeze batch dimension for processing
        let embeddings = embeddings.squeeze(Some(&[0]))?;

        // Add position embeddings per image
        let grid_data = grid_thw.to_int32()?;
        let num_images = grid_thw.shape()?[0] as usize;

        let mut result_parts: Vec<MxArray> = Vec::new();
        let mut start = 0i64;

        for img_idx in 0..num_images {
            let t = grid_data[img_idx * 3] as i64;
            let h = grid_data[img_idx * 3 + 1] as i64;
            let w = grid_data[img_idx * 3 + 2] as i64;

            let end = start + t * h * w;

            // Extract embeddings for this image
            let img_embeddings = embeddings.slice_axis(0, start, end)?;

            // Get interpolated position embeddings
            let pos_embed = self.position_embedding.forward(h as u32, w as u32)?;

            // Repeat position embeddings for each temporal frame when t > 1
            let pos_embed = if t > 1 {
                let pos_embeds: Vec<&MxArray> = (0..t).map(|_| &pos_embed).collect();
                MxArray::concatenate_many(pos_embeds, Some(0))?
            } else {
                pos_embed
            };

            // Add position embeddings
            let img_with_pos = img_embeddings.add(&pos_embed)?;

            result_parts.push(img_with_pos);
            start = end;
        }

        // Concatenate all image embeddings
        if result_parts.len() == 1 {
            Ok(result_parts.remove(0))
        } else {
            let refs: Vec<&MxArray> = result_parts.iter().collect();
            MxArray::concatenate_many(refs, Some(0))
        }
    }

    pub fn weight(&self) -> MxArray {
        self.patch_embedding.weight()
    }
}

/// PaddleOCR Vision Model (internal)
///
/// Complete vision encoder with embeddings, transformer layers, and projector.
///
/// Note: This is an internal implementation detail used by VLModel.
/// Not exposed to TypeScript - users interact with high-level VLModel API.
pub struct PaddleOCRVisionModel {
    config: VisionConfig,
    embeddings: Arc<PaddleOCRVisionEmbeddings>,
    rotary_pos_emb: Arc<VisionRotaryEmbedding>,
    layers: Vec<Arc<VisionEncoderLayer>>,
    post_layernorm: Arc<LayerNorm>,
    projector: Arc<SpatialProjector>,
}

impl PaddleOCRVisionModel {
    /// Create a new vision model
    ///
    /// Note: Layers should be added separately using `add_layer` after construction.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: VisionConfig,
        patch_weight: &MxArray,
        position_weight: &MxArray,
        post_norm_weight: &MxArray,
        post_norm_bias: &MxArray,
        projector_pre_norm_weight: &MxArray,
        projector_pre_norm_bias: &MxArray,
        projector_linear1_weight: &MxArray,
        projector_linear1_bias: &MxArray,
        projector_linear2_weight: &MxArray,
        projector_linear2_bias: &MxArray,
    ) -> Result<Self> {
        let embeddings = PaddleOCRVisionEmbeddings::new(
            config.patch_size as u32,
            config.image_size as u32,
            config.hidden_size as u32,
            patch_weight,
            position_weight,
        )?;

        let head_dim = config.hidden_size / config.num_attention_heads;
        let rotary_pos_emb = VisionRotaryEmbedding::new((head_dim / 2) as u32, None);

        let post_layernorm = LayerNorm::from_weights(
            post_norm_weight,
            Some(post_norm_bias),
            Some(config.layer_norm_eps),
        )?;

        let projector = SpatialProjector::new(
            config.spatial_merge_size as u32,
            projector_pre_norm_weight,
            projector_pre_norm_bias,
            projector_linear1_weight,
            projector_linear1_bias,
            projector_linear2_weight,
            projector_linear2_bias,
        )?;

        Ok(Self {
            config,
            embeddings: Arc::new(embeddings),
            rotary_pos_emb: Arc::new(rotary_pos_emb),
            layers: Vec::new(),
            post_layernorm: Arc::new(post_layernorm),
            projector: Arc::new(projector),
        })
    }

    /// Add an encoder layer
    pub fn add_layer(&mut self, layer: &VisionEncoderLayer) {
        // Clone the layer
        self.layers.push(Arc::new(layer.clone()));
    }

    /// Compute rotary position embeddings for the grid
    fn compute_rot_pos_emb(&self, grid_thw: &MxArray) -> Result<MxArray> {
        let grid_data = grid_thw.to_int32()?;
        let num_images = grid_thw.shape()?[0] as usize;

        let mut height_ids: Vec<i32> = Vec::new();
        let mut width_ids: Vec<i32> = Vec::new();

        for img_idx in 0..num_images {
            let t = grid_data[img_idx * 3];
            let h = grid_data[img_idx * 3 + 1];
            let w = grid_data[img_idx * 3 + 2];

            let num_patches = t * h * w;

            for idx in 0..num_patches {
                let local_idx = idx % (h * w);
                let h_id = local_idx / w;
                let w_id = local_idx % w;
                height_ids.push(h_id);
                width_ids.push(w_id);
            }
        }

        // Stack height and width position IDs
        let h_ids = MxArray::from_int32(&height_ids, &[height_ids.len() as i64])?;
        let w_ids = MxArray::from_int32(&width_ids, &[width_ids.len() as i64])?;
        let pos_ids = MxArray::stack(vec![&h_ids, &w_ids], Some(-1))?;

        // Get max grid size for computing frequencies
        let max_h = grid_data
            .iter()
            .skip(1)
            .step_by(3)
            .max()
            .copied()
            .unwrap_or(1);
        let max_w = grid_data
            .iter()
            .skip(2)
            .step_by(3)
            .max()
            .copied()
            .unwrap_or(1);
        let max_size = max_h.max(max_w) as u32;

        let freqs = self.rotary_pos_emb.forward(max_size)?;

        // Gather frequencies using position IDs
        // pos_ids: [total_patches, 2]
        // freqs: [max_size, dim]
        // Output: [total_patches, dim*2] (for h and w concatenated)

        // Gather h and w frequencies separately
        let h_ids_flat = pos_ids.slice_axis(1, 0, 1)?.reshape(&[-1])?;
        let w_ids_flat = pos_ids.slice_axis(1, 1, 2)?.reshape(&[-1])?;

        let h_freqs = freqs.take(&h_ids_flat, 0)?;
        let w_freqs = freqs.take(&w_ids_flat, 0)?;

        // Concatenate h and w frequencies
        MxArray::concatenate_many(vec![&h_freqs, &w_freqs], Some(-1))
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `hidden_states` - Pixel patches [batch, seq_len, channels, patch_h, patch_w]
    /// * `grid_thw` - Grid dimensions [num_images, 3]
    ///
    /// # Returns
    /// * Vision features [total_merged_patches, context_dim]
    pub fn forward(&self, hidden_states: &MxArray, grid_thw: &MxArray) -> Result<MxArray> {
        // Apply embeddings
        let mut h = self.embeddings.forward(hidden_states, grid_thw)?;

        // Compute rotary position embeddings
        let rotary_pos_emb = self.compute_rot_pos_emb(grid_thw)?;

        // Compute cumulative sequence lengths for attention
        let grid_data = grid_thw.to_int32()?;
        let num_images = grid_thw.shape()?[0] as usize;

        let mut cu_seqlens: Vec<i32> = vec![0];
        for img_idx in 0..num_images {
            let t = grid_data[img_idx * 3];
            let h = grid_data[img_idx * 3 + 1];
            let w = grid_data[img_idx * 3 + 2];
            for _ in 0..t {
                let prev = *cu_seqlens.last().unwrap();
                cu_seqlens.push(prev + h * w);
            }
        }
        let cu_seqlens_arr = MxArray::from_int32(&cu_seqlens, &[cu_seqlens.len() as i64])?;

        // Forward through encoder layers
        for layer in &self.layers {
            h = layer.forward(&h, &cu_seqlens_arr, Some(&rotary_pos_emb))?;
        }

        // Post layer norm
        h = self.post_layernorm.forward(&h)?;

        // Apply spatial projector (reduces token count)
        self.projector.forward(&h, grid_thw)
    }

    /// Get number of layers
    pub fn num_layers(&self) -> u32 {
        self.layers.len() as u32
    }
}

impl Clone for PaddleOCRVisionEmbeddings {
    fn clone(&self) -> Self {
        Self {
            patch_embedding: self.patch_embedding.clone(),
            position_embedding: self.position_embedding.clone(),
            patch_size: self.patch_size,
            embed_dim: self.embed_dim,
        }
    }
}

impl Clone for PaddleOCRVisionModel {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            embeddings: self.embeddings.clone(),
            rotary_pos_emb: self.rotary_pos_emb.clone(),
            layers: self.layers.clone(),
            post_layernorm: self.post_layernorm.clone(),
            projector: self.projector.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vision_config_defaults() {
        let config = VisionConfig::default();
        assert_eq!(config.hidden_size, 1152);
        assert_eq!(config.patch_size, 14);
        assert_eq!(config.spatial_merge_size, 2);
    }
}
