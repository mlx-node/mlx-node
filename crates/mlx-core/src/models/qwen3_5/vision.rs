/// Qwen3.5 Vision Encoder
///
/// Assembles existing shared vision components (VisionAttention, VisionMLP,
/// VisionEncoderLayer, SpatialProjector, VisionRotaryEmbedding) into a full
/// vision encoder for Qwen3.5-VL.
///
/// Architecture: patch_embed → + pos_embed → 27 blocks with 2D RoPE → merger
use crate::array::MxArray;
use crate::vision::embeddings::PatchEmbedding;
use crate::vision::encoder::VisionEncoderLayer;
use crate::vision::projector::SpatialProjector;
use crate::vision::rope_vision::VisionRotaryEmbedding;
use napi::bindgen_prelude::*;
use std::sync::Arc;

/// Qwen3.5 Vision Encoder configuration
#[derive(Debug, Clone)]
pub struct Qwen3_5VisionConfig {
    pub hidden_size: i32,        // 1152
    pub intermediate_size: i32,  // 4304
    pub num_heads: i32,          // 16
    pub num_layers: i32,         // 27
    pub patch_size: i32,         // 16
    pub spatial_merge_size: i32, // 2
    pub image_size: i32,         // 768 (reference)
    pub out_hidden_size: i32,    // 4096 (matches text hidden_size)
}

impl Default for Qwen3_5VisionConfig {
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

/// Qwen3.5 Vision Encoder
///
/// Processes images into vision features compatible with the text model.
pub struct Qwen3_5VisionEncoder {
    config: Qwen3_5VisionConfig,
    /// Patch embedding: Conv2d extracting patches from images
    patch_embed: Arc<PatchEmbedding>,
    /// Learned position embeddings [num_pos, hidden_size]
    pos_embed: Option<Arc<MxArray>>,
    /// 27 transformer encoder layers
    layers: Vec<Arc<VisionEncoderLayer>>,
    /// 2D rotary position embeddings
    rotary_pos_emb: Arc<VisionRotaryEmbedding>,
    /// Spatial projector: merges patches and projects to text model dimension
    merger: Option<Arc<SpatialProjector>>,
}

impl Qwen3_5VisionEncoder {
    /// Create a new vision encoder (layers and weights added separately).
    pub fn new(config: Qwen3_5VisionConfig) -> Result<Self> {
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
        let patch_embed = PatchEmbedding::new(patch_size, &dummy_weight)?;

        Ok(Self {
            config,
            patch_embed: Arc::new(patch_embed),
            pos_embed: None,
            layers: Vec::new(),
            rotary_pos_emb: Arc::new(rotary_pos_emb),
            merger: None,
        })
    }

    /// Set patch embedding weights
    pub fn set_patch_embed(&mut self, weight: &MxArray) -> Result<()> {
        self.patch_embed = Arc::new(PatchEmbedding::new(self.config.patch_size as u32, weight)?);
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

    /// Compute 2D rotary position embeddings for the grid
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
                height_ids.push(local_idx / w);
                width_ids.push(local_idx % w);
            }
        }

        let h_ids = MxArray::from_int32(&height_ids, &[height_ids.len() as i64])?;
        let w_ids = MxArray::from_int32(&width_ids, &[width_ids.len() as i64])?;
        let pos_ids = MxArray::stack(vec![&h_ids, &w_ids], Some(-1))?;

        // Max grid size for frequency computation
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

        // Gather h and w frequencies separately then concatenate
        let h_ids_flat = pos_ids.slice_axis(1, 0, 1)?.reshape(&[-1])?;
        let w_ids_flat = pos_ids.slice_axis(1, 1, 2)?.reshape(&[-1])?;

        let h_freqs = freqs.take(&h_ids_flat, 0)?;
        let w_freqs = freqs.take(&w_ids_flat, 0)?;

        MxArray::concatenate_many(vec![&h_freqs, &w_freqs], Some(-1))
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
        let flat_input = flat_input.transpose(Some(&[0, 2, 3, 1]))?;

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

        // Compute 2D rotary position embeddings
        let rotary_pos_emb = self.compute_rot_pos_emb(grid_thw)?;

        // Build cumulative sequence lengths for attention masking
        let mut cu_seqlens: Vec<i32> = vec![0];
        for img_idx in 0..num_images {
            let t = grid_data[img_idx * 3];
            let h_dim = grid_data[img_idx * 3 + 1];
            let w_dim = grid_data[img_idx * 3 + 2];
            for _ in 0..t {
                let prev = *cu_seqlens.last().unwrap();
                cu_seqlens.push(prev + h_dim * w_dim);
            }
        }
        let cu_seqlens_arr = MxArray::from_int32(&cu_seqlens, &[cu_seqlens.len() as i64])?;

        // Forward through encoder layers
        for layer in &self.layers {
            h = layer.forward(&h, &cu_seqlens_arr, Some(&rotary_pos_emb))?;
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

    /// Get spatial merge size
    pub fn spatial_merge_size(&self) -> i32 {
        self.config.spatial_merge_size
    }

    /// Get the config
    pub fn config(&self) -> &Qwen3_5VisionConfig {
        &self.config
    }
}

impl Clone for Qwen3_5VisionEncoder {
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
