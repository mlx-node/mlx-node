use napi::bindgen_prelude::*;

use crate::array::{DType, MxArray};
use crate::nn::{LayerNorm, Linear};

use super::unified_vision_config::UnifiedVisionConfig;

/// Encoder-free Gemma 4 unified vision embedder.
///
/// Mirrors `mlx-vlm`'s `gemma4_unified.VisionEmbedder`:
/// `patch_ln1` → `patch_dense` → `patch_ln2` → add 2D positional embedding →
/// `pos_norm`. Patches arrive already flattened to `[n, patch_dim]` where
/// `patch_dim = model_patch_size^2 * 3`. There is no transformer encoder; the
/// projected patches feed directly into `embed_vision`.
pub struct Gemma4UnifiedVisionEmbedder {
    patch_dim: i32,
    eps: f64,
    pub patch_ln1: LayerNorm,
    pub patch_dense: Linear,
    pub patch_ln2: LayerNorm,
    /// 2D positional embedding table `[mm_posemb_size, 2, mm_embed_dim]`.
    pub pos_embedding: MxArray,
    pub pos_norm: LayerNorm,
}

impl Gemma4UnifiedVisionEmbedder {
    pub fn new(config: &UnifiedVisionConfig) -> Result<Self> {
        let patch_dim = config.model_patch_size * config.model_patch_size * 3;
        let embed_dim = config.mm_embed_dim;
        let eps = Some(config.rms_norm_eps);
        Ok(Self {
            patch_dim,
            eps: config.rms_norm_eps,
            patch_ln1: LayerNorm::new(patch_dim as u32, eps)?,
            patch_dense: Linear::new(patch_dim as u32, embed_dim as u32, Some(true))?,
            patch_ln2: LayerNorm::new(embed_dim as u32, eps)?,
            pos_embedding: MxArray::zeros(
                &[config.mm_posemb_size as i64, 2, config.mm_embed_dim as i64],
                Some(DType::Float32),
            )?,
            pos_norm: LayerNorm::new(embed_dim as u32, eps)?,
        })
    }

    pub fn patch_dim(&self) -> i32 {
        self.patch_dim
    }

    pub fn eps(&self) -> f64 {
        self.eps
    }

    /// Forward over a single image's patches.
    ///
    /// * `pixel_values` — `[n, patch_dim]` flattened patches.
    /// * `position_ids` — `[n, 2]` int32 `(x, y)` grid coordinates. `-1` rows
    ///   mark padding; their positional contribution is masked to zero. In the
    ///   single-image Rust path every row is real (no padding), so the mask is a
    ///   no-op, but it is implemented faithfully to match the reference.
    ///
    /// Returns `[n, mm_embed_dim]`.
    pub fn forward(&self, pixel_values: &MxArray, position_ids: &MxArray) -> Result<MxArray> {
        let hidden = self.patch_ln1.forward(pixel_values)?;
        let hidden = self.patch_dense.forward(&hidden)?;
        let hidden = self.patch_ln2.forward(&hidden)?;

        let hidden = self.add_position_embeddings(&hidden, position_ids)?;
        self.pos_norm.forward(&hidden)
    }

    /// `hidden += pos_embedding[x, 0] * valid_x + pos_embedding[y, 1] * valid_y`
    /// where `(x, y) = position_ids[..., 0], position_ids[..., 1]`, clamped to
    /// `>= 0`, and `valid_* = (position_ids[..., *] != -1)`.
    fn add_position_embeddings(&self, hidden: &MxArray, position_ids: &MxArray) -> Result<MxArray> {
        let in_dtype = hidden.dtype()?;
        // position_ids: [n, 2] -> columns x (axis 1, idx 0) and y (idx 1).
        let x_idx = position_ids.take(&MxArray::scalar_int(0)?, 1)?; // [n]
        let y_idx = position_ids.take(&MxArray::scalar_int(1)?, 1)?; // [n]

        let neg_one = MxArray::scalar_int(-1)?;
        let valid_x = x_idx
            .not_equal(&neg_one)?
            .astype(in_dtype)?
            .expand_dims(-1)?; // [n, 1]
        let valid_y = y_idx
            .not_equal(&neg_one)?
            .astype(in_dtype)?
            .expand_dims(-1)?;

        let zero = MxArray::scalar_int(0)?;
        let x_clamped = x_idx.maximum(&zero)?.astype(DType::Int32)?;
        let y_clamped = y_idx.maximum(&zero)?.astype(DType::Int32)?;

        // pos_embedding: [P, 2, D]. take(axis=0, idx=x_clamped) -> [n, 2, D];
        // slice the per-axis plane (0 for x, 1 for y) and drop that axis.
        let pe = &self.pos_embedding;
        let d = pe.shape_at(2)?;

        let x_planes = pe.take(&x_clamped, 0)?; // [n, 2, D]
        let x_pos = x_planes.take(&zero, 1)?.reshape(&[-1, d])?; // [n, D]
        let y_planes = pe.take(&y_clamped, 0)?;
        let y_pos = y_planes
            .take(&MxArray::scalar_int(1)?, 1)?
            .reshape(&[-1, d])?;

        let x_pos = x_pos.astype(in_dtype)?.mul(&valid_x)?;
        let y_pos = y_pos.astype(in_dtype)?.mul(&valid_y)?;

        hidden.add(&x_pos)?.add(&y_pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> UnifiedVisionConfig {
        UnifiedVisionConfig {
            model_patch_size: 2,
            mm_embed_dim: 4,
            mm_posemb_size: 8,
            num_soft_tokens: 6,
            output_proj_dims: 4,
            patch_size: 1,
            pooling_kernel_size: 2,
            rms_norm_eps: 1e-6,
        }
    }

    #[test]
    fn embedder_forward_shapes() {
        let cfg = tiny_config();
        let ve = Gemma4UnifiedVisionEmbedder::new(&cfg).expect("build embedder");
        // patch_dim = 2*2*3 = 12. Two patches.
        let n = 2i64;
        let patches =
            MxArray::from_float32(&vec![0.1f32; (n * 12) as usize], &[n, 12]).expect("patches");
        // positions (x, y) in-range of mm_posemb_size.
        let positions = MxArray::from_int32(&[0, 0, 1, 0], &[n, 2]).expect("positions");
        let out = ve.forward(&patches, &positions).expect("forward");
        assert_eq!(out.shape().unwrap().as_ref(), &[n, cfg.mm_embed_dim as i64]);
        out.eval();
    }

    #[test]
    fn embedder_masks_padding_positions() {
        // A -1 position row must contribute zero positional embedding, so its
        // output equals the same patch run through the embedder with no pos.
        let cfg = tiny_config();
        let mut ve = Gemma4UnifiedVisionEmbedder::new(&cfg).expect("build embedder");
        // Non-uniform pos_embedding (varies per channel) so the positional add
        // survives the LayerNorm in pos_norm — a uniform shift would be
        // normalized away and could not distinguish masked from unmasked.
        let total = (cfg.mm_posemb_size * 2 * cfg.mm_embed_dim) as usize;
        let pos: Vec<f32> = (0..total).map(|i| (i % 7) as f32 * 0.3 - 0.6).collect();
        ve.pos_embedding = MxArray::from_float32(
            &pos,
            &[cfg.mm_posemb_size as i64, 2, cfg.mm_embed_dim as i64],
        )
        .unwrap();

        let patches = MxArray::from_float32(&[0.2f32; 12], &[1, 12]).unwrap();
        let padded_pos = MxArray::from_int32(&[-1, -1], &[1, 2]).unwrap();
        let valid_pos = MxArray::from_int32(&[0, 0], &[1, 2]).unwrap();

        let out_padded = ve.forward(&patches, &padded_pos).unwrap();
        // For valid (0,0) the contribution is pos[0,0]+pos[0,1] (both 0.5 each),
        // applied BEFORE pos_norm; padded masks both to zero. The two outputs
        // must therefore differ.
        let out_valid = ve.forward(&patches, &valid_pos).unwrap();
        out_padded.eval();
        out_valid.eval();
        let diff = out_padded
            .sub(&out_valid)
            .unwrap()
            .abs()
            .unwrap()
            .sum(None, None)
            .unwrap();
        diff.eval();
        // Padding masks the positional add → outputs differ (sanity that the
        // mask is wired, not a no-op stub that ignores positions entirely).
        assert!(
            diff.item_at_float32(0).unwrap() > 1e-3,
            "padded vs valid outputs should differ when pos_embedding is non-zero"
        );
    }
}
