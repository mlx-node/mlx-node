//! Vision Rotary Position Embedding
//!
//! Internal implementation - not exposed to TypeScript.
//! 2D rotary position embeddings for vision transformers.
//! Used for encoding spatial (height, width) positions in image patches.

use crate::array::{DType, MxArray};
use mlx_sys as sys;
use napi::bindgen_prelude::*;
use std::sync::Arc;

/// Vision Rotary Position Embedding (internal)
///
/// Generates 2D rotary embeddings for vision transformer patches.
/// Unlike text RoPE which uses 1D positions, vision RoPE encodes
/// both height and width positions.
pub struct VisionRotaryEmbedding {
    /// Dimension of the rotary embedding (typically head_dim // 2)
    dim: u32,
    /// Base theta for frequency computation
    theta: f32,
}

impl VisionRotaryEmbedding {
    /// Create a new Vision Rotary Embedding
    ///
    /// # Arguments
    /// * `dim` - Dimension of the embedding (typically head_dim // 2)
    /// * `theta` - Optional base theta, default 10000.0
    pub fn new(dim: u32, theta: Option<f64>) -> Self {
        Self {
            dim,
            theta: theta.unwrap_or(10000.0) as f32,
        }
    }

    /// Generate frequency tensor for a given sequence length
    ///
    /// # Arguments
    /// * `seq_len` - The sequence length to generate frequencies for
    ///
    /// # Returns
    /// * Frequency tensor of shape [seq_len, dim]
    pub fn forward(&self, seq_len: u32) -> Result<MxArray> {
        // Generate inverse frequencies: 1 / (theta^(2i/dim))
        let dim = self.dim as i64;
        let half_dim = dim / 2;

        // Create indices: [0, 2, 4, ..., dim-2] / dim
        let mut inv_freq_data = Vec::with_capacity(half_dim as usize);
        for i in 0..half_dim {
            let exp = (2 * i) as f32 / dim as f32;
            inv_freq_data.push(1.0 / self.theta.powf(exp));
        }
        let inv_freq = MxArray::from_float32(&inv_freq_data, &[half_dim])?;

        // Create sequence positions
        let seq = MxArray::arange(0.0, seq_len as f64, Some(1.0), None)?;

        // Compute outer product: [seq_len] x [half_dim] -> [seq_len, half_dim]
        let seq_expanded = seq.reshape(&[seq_len as i64, 1])?;
        let inv_freq_expanded = inv_freq.reshape(&[1, half_dim])?;
        let freqs = seq_expanded.mul(&inv_freq_expanded)?;

        Ok(freqs)
    }
}

/// Rotate half of the input tensor using pre-computed `ndim` and `last_dim`
/// to avoid redundant `.shape()` FFI calls in hot loops.
fn rotate_half_with_dims(x: &MxArray, ndim: usize, last_dim: i64) -> Result<MxArray> {
    let half_dim = last_dim / 2;

    let x1 = x.slice_axis(ndim - 1, 0, half_dim)?;
    let x2 = x.slice_axis(ndim - 1, half_dim, last_dim)?;

    let neg_x2 = x2.negative()?;
    MxArray::concatenate(&neg_x2, &x1, -1)
}

/// Rotate half of the input tensor
///
/// For input x = [x1, x2], returns [-x2, x1]
pub fn rotate_half(x: &MxArray) -> Result<MxArray> {
    let shape = x.shape()?;
    let ndim = shape.len();
    let last_dim = shape[ndim - 1];
    rotate_half_with_dims(x, ndim, last_dim)
}

/// Compute 2D rotary position embeddings for a multi-image/frame `grid_thw` tensor.
///
/// Shared by Qwen VLMs and PaddleOCR-VL vision encoders.
pub fn compute_rot_pos_emb_2d(
    rotary_pos_emb: &VisionRotaryEmbedding,
    grid_thw: &MxArray,
) -> Result<MxArray> {
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

    let freqs = rotary_pos_emb.forward(max_size)?;

    let h_ids_flat = pos_ids.slice_axis(1, 0, 1)?.reshape(&[-1])?;
    let w_ids_flat = pos_ids.slice_axis(1, 1, 2)?.reshape(&[-1])?;

    let h_freqs = freqs.take(&h_ids_flat, 0)?;
    let w_freqs = freqs.take(&w_ids_flat, 0)?;

    MxArray::concatenate_many(vec![&h_freqs, &w_freqs], Some(-1))
}

/// Multimodal Rotary Position Embedding (internal)
///
/// Unlike standard RoPE which uses 1D positions, mRoPE uses 3D positions
/// (temporal, height, width) to encode spatial relationships in vision tokens.
/// The head dimension is split into sections according to `mrope_section`.
pub struct MultimodalRoPE {
    /// mRoPE sections [temporal, height, width] (e.g., [16, 24, 24] or [11, 11, 10])
    mrope_section: [i32; 3],
    /// Pre-computed inverse frequencies, pre-shaped to [1, 1, half_dim, 1] for broadcasting
    inv_freq: Arc<MxArray>,
    /// Cached half_dim value to avoid FFI .shape() calls
    inv_freq_dim: i64,
    /// Attention scaling factor
    attention_scaling: f32,
}

impl MultimodalRoPE {
    /// Create a new Multimodal RoPE
    ///
    /// # Arguments
    /// * `dim` - Head dimension (e.g., 128)
    /// * `max_position_embeddings` - Maximum sequence length
    /// * `base` - Base theta (default 500000.0 for PaddleOCR-VL)
    /// * `mrope_section` - Section sizes [temporal, height, width]
    pub fn new(
        dim: i32,
        _max_position_embeddings: i32,
        base: Option<f64>,
        mrope_section: Vec<i32>,
    ) -> Result<Self> {
        let base = base.unwrap_or(500000.0) as f32;
        let inv_freq_data: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1.0 / base.powf(i as f32 / dim as f32))
            .collect();
        Self::with_inverse_frequencies(dim, mrope_section, &inv_freq_data, 1.0)
    }

    /// An M-RoPE over `dim` dims whose `dim / 2` inverse frequencies are
    /// `inverse`, with cos and sin scaled by `attention_scaling`.
    pub fn with_inverse_frequencies(
        dim: i32,
        mrope_section: Vec<i32>,
        inverse: &[f32],
        attention_scaling: f32,
    ) -> Result<Self> {
        if mrope_section.len() != 3 {
            return Err(Error::new(
                Status::InvalidArg,
                "mrope_section must have exactly 3 elements [t, h, w]",
            ));
        }

        let section_sum: i32 = mrope_section.iter().map(|&x| x * 2).sum();
        if section_sum != dim {
            return Err(Error::new(
                Status::InvalidArg,
                format!(
                    "mrope_section sum ({}) * 2 = {} must equal dim ({})",
                    mrope_section.iter().sum::<i32>(),
                    section_sum,
                    dim
                ),
            ));
        }
        let half_dim = dim / 2;
        if inverse.len() != half_dim as usize {
            return Err(Error::new(
                Status::InvalidArg,
                format!(
                    "M-RoPE takes {half_dim} inverse frequencies, got {}",
                    inverse.len()
                ),
            ));
        }

        let mrope_section_arr: [i32; 3] = [mrope_section[0], mrope_section[1], mrope_section[2]];
        let inv_freq_dim = half_dim as i64;
        // Pre-reshape to [1, 1, half_dim, 1] to avoid reshape+astype in every forward call.
        let inv_freq = MxArray::from_float32(inverse, &[1, 1, inv_freq_dim, 1])?;

        Ok(Self {
            mrope_section: mrope_section_arr,
            inv_freq: Arc::new(inv_freq),
            inv_freq_dim,
            attention_scaling,
        })
    }

    /// Compute cos and sin for rotary embeddings
    ///
    /// # Arguments
    /// * `x` - Input tensor (used only for dtype)
    /// * `position_ids` - Position IDs [3, batch, seq_len] for t, h, w
    ///
    /// # Returns
    /// * Tuple of (cos, sin) each of shape [3, batch, seq_len, head_dim]
    pub fn forward(&self, x: &MxArray, position_ids: &MxArray) -> Result<(MxArray, MxArray)> {
        let target_dtype = x.dtype()?;
        let pos_shape = position_ids.shape()?;

        // position_ids: [3, batch, seq_len]
        let batch_size = pos_shape[1];
        let seq_len = pos_shape[2];

        let inv_freq_expanded =
            MxArray::broadcast_to(&self.inv_freq, &[3, batch_size, self.inv_freq_dim, 1])?;

        let pos_expanded = position_ids
            .reshape(&[3, batch_size, 1, seq_len])?
            .astype(crate::array::DType::Float32)?;

        let freqs = inv_freq_expanded.matmul(&pos_expanded)?;
        let freqs = freqs.transpose(Some(&[0, 1, 3, 2]))?;
        let emb = MxArray::concatenate_many(vec![&freqs, &freqs], Some(-1))?;

        let (cos, sin) = if (self.attention_scaling - 1.0).abs() < 1e-8 {
            (emb.cos()?, emb.sin()?)
        } else {
            let cos = emb.cos()?.mul_scalar(self.attention_scaling as f64)?;
            let sin = emb.sin()?.mul_scalar(self.attention_scaling as f64)?;
            (cos, sin)
        };

        let cos = if target_dtype == crate::array::DType::Float32 {
            cos
        } else {
            cos.astype(target_dtype)?
        };
        let sin = if target_dtype == crate::array::DType::Float32 {
            sin
        } else {
            sin.astype(target_dtype)?
        };

        Ok((cos, sin))
    }

    /// Get raw inv_freq pointer for C++ forward pass
    pub fn get_inv_freq_ptr(&self) -> *mut sys::mlx_array {
        self.inv_freq.handle.0
    }

    /// Get mRoPE section array reference
    pub fn mrope_section_arr(&self) -> &[i32; 3] {
        &self.mrope_section
    }
}

/// Apply INTERLEAVED multimodal rotary position embedding to Q and K (internal).
///
/// Used by Qwen3.5-VL (`qwen3_5` / `qwen3_5_moe`). Unlike PaddleOCR-VL, whose
/// multimodal RoPE assigns the head_dim to spatial axes in CONTIGUOUS chunks,
/// Qwen3.5-VL assigns each inv_freq index to a spatial axis with a STRIDE-3
/// INTERLEAVE. This mirrors mlx-vlm's `_interleaved_position_selector`
/// (rope_utils.py):
///   selector[idx] = 1 (height) for idx in 1, 4, 7, ... up to `mrope_section[1] * 3`
///   selector[idx] = 2 (width)  for idx in 2, 5, 8, ... up to `mrope_section[2] * 3`
///   selector[idx] = 0 (temporal) otherwise
/// computed over the `half_dim` inv_freq axis, then mirrored across the doubled
/// cos/sin (`emb = concat([freqs, freqs])`).
///
/// `cos`/`sin` shape: `[3, batch, seq_len, head_dim]` (axis 0 = t/h/w).
/// `q`/`k` shape: `[batch, heads, seq_len, q_head_dim]` where `q_head_dim >=
/// head_dim` (the trailing `q_head_dim - head_dim` dims pass through unrotated).
pub fn apply_multimodal_rotary_pos_emb_interleaved(
    q: &MxArray,
    k: &MxArray,
    cos: &MxArray,
    sin: &MxArray,
    mrope_section: Vec<i32>,
) -> Result<(MxArray, MxArray)> {
    let (cos_final, sin_final) = select_interleaved_cos_sin(cos, sin, &mrope_section)?;
    apply_interleaved_rotary(q, k, &cos_final, &sin_final)
}

/// Position-only half of [`apply_multimodal_rotary_pos_emb_interleaved`].
///
/// Builds the stride-3 per-frequency axis selector and gathers the selected
/// `[batch, 1, seq_len, head_dim]` cos/sin rows out of the doubled
/// `[3, batch, seq_len, head_dim]` cos/sin. Safe to compute once per forward
/// pass and reuse across every full-attention layer via
/// [`apply_interleaved_rotary`].
pub fn select_interleaved_cos_sin(
    cos: &MxArray,
    sin: &MxArray,
    mrope_section: &[i32],
) -> Result<(MxArray, MxArray)> {
    let cos_shape = cos.shape()?;
    let batch_size = cos_shape[1];
    let seq_len = cos_shape[2];
    let head_dim = cos_shape[3];
    let half = head_dim / 2;
    let half_usize = half as usize;

    let mut sel_half = vec![0i32; half_usize];
    for (dim, offset) in [(1usize, 1usize), (2usize, 2usize)] {
        let limit = std::cmp::min(mrope_section[dim] as usize * 3, half_usize);
        let mut idx = offset;
        while idx < limit {
            sel_half[idx] = dim as i32;
            idx += 3;
        }
    }
    let mut sel_doubled = vec![0i32; head_dim as usize];
    for (j, slot) in sel_doubled.iter_mut().enumerate() {
        *slot = sel_half[j % half_usize];
    }

    let indices = MxArray::from_int32(&sel_doubled, &[1, 1, 1, head_dim])?;
    let indices = MxArray::broadcast_to(&indices, &[1, batch_size, seq_len, head_dim])?;

    let cos_sel = cos.take_along_axis(&indices, 0)?;
    let sin_sel = sin.take_along_axis(&indices, 0)?;

    let cos_final = cos_sel.reshape(&[batch_size, 1, seq_len, head_dim])?;
    let sin_final = sin_sel.reshape(&[batch_size, 1, seq_len, head_dim])?;
    Ok((cos_final, sin_final))
}

/// Q/K half of [`apply_multimodal_rotary_pos_emb_interleaved`]: applies
/// `rotate_half` using an already-selected `(cos_final, sin_final)` pair from
/// [`select_interleaved_cos_sin`].
pub fn apply_interleaved_rotary(
    q: &MxArray,
    k: &MxArray,
    cos_final: &MxArray,
    sin_final: &MxArray,
) -> Result<(MxArray, MxArray)> {
    let cos_shape = cos_final.shape()?;
    let rotary_dim = cos_shape[3];
    let q_shape = q.shape()?;
    let q_dim = q_shape[3];
    let q_ndim = q_shape.len();

    let q_rot = q.slice_axis(3, 0, rotary_dim)?;
    let q_pass = if rotary_dim < q_dim {
        Some(q.slice_axis(3, rotary_dim, q_dim)?)
    } else {
        None
    };

    let k_rot = k.slice_axis(3, 0, rotary_dim)?;
    let k_pass = if rotary_dim < q_dim {
        Some(k.slice_axis(3, rotary_dim, q_dim)?)
    } else {
        None
    };

    let q_rotated = rotate_half_with_dims(&q_rot, q_ndim, rotary_dim)?;
    let k_rotated = rotate_half_with_dims(&k_rot, q_ndim, rotary_dim)?;

    let q_embed = q_rot.mul(cos_final)?.add(&q_rotated.mul(sin_final)?)?;
    let k_embed = k_rot.mul(cos_final)?.add(&k_rotated.mul(sin_final)?)?;

    let q_out = if let Some(q_pass) = q_pass {
        MxArray::concatenate_many(vec![&q_embed, &q_pass], Some(-1))?
    } else {
        q_embed
    };

    let k_out = if let Some(k_pass) = k_pass {
        MxArray::concatenate_many(vec![&k_embed, &k_pass], Some(-1))?
    } else {
        k_embed
    };

    Ok((q_out, k_out))
}

/// Precompute FP32 `(cos, sin)` tensors from vision RoPE frequencies.
///
/// Computing `cos`/`sin` and tiling once per image avoids repeating trig and
/// tile kernels across Q and K in every vision transformer layer.
///
/// # Arguments
/// * `freqs` - Frequency tensor of shape `[seq_len, dim]` where `dim = head_dim / 2`
///
/// # Returns
/// * `(cos, sin)` each of shape `[seq_len, 1, head_dim]` in FP32
pub fn precompute_rotary_cos_sin(freqs: &MxArray) -> Result<(MxArray, MxArray)> {
    let freqs_f32 = if freqs.dtype()? != DType::Float32 {
        freqs.astype(DType::Float32)?
    } else {
        freqs.clone()
    };
    let shape = freqs_f32.shape()?;
    if shape.len() != 2 {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "precompute_rotary_cos_sin expects 2D freqs [seq_len, dim], got {:?}",
                shape.as_ref()
            ),
        ));
    }
    let freq_dim = shape[1];

    // Compute cos and sin from frequencies in FP32
    let cos = freqs_f32.cos()?;
    let sin = freqs_f32.sin()?;

    // Expand dims: [seq_len, dim] -> [seq_len, 1, dim]
    let cos = cos.reshape(&[-1, 1, freq_dim])?;
    let sin = sin.reshape(&[-1, 1, freq_dim])?;

    // Tile to double the dimension: [seq_len, 1, dim*2] (= [seq_len, 1, head_dim])
    let cos = MxArray::tile(&cos, &[1, 1, 2])?;
    let sin = MxArray::tile(&sin, &[1, 1, 2])?;

    Ok((cos, sin))
}

/// Apply rotary position embedding to a vision tensor using precomputed `(cos, sin)`.
///
/// # Arguments
/// * `tensor` - Input tensor of shape `[seq_len, num_heads, head_dim]` or
///   `[batch, seq_len, num_heads, head_dim]`
/// * `cos` - Precomputed cosine tensor of shape `[seq_len, 1, head_dim]` (FP32)
/// * `sin` - Precomputed sine tensor of shape `[seq_len, 1, head_dim]` (FP32)
///
/// # Returns
/// * Tensor with rotary embedding applied in `tensor`'s original dtype
pub fn apply_rotary_pos_emb_vision_precomputed(
    tensor: &MxArray,
    cos: &MxArray,
    sin: &MxArray,
) -> Result<MxArray> {
    let orig_dtype = tensor.dtype()?;

    // Apply rotary embedding: x * cos + rotate_half(x) * sin
    let rotated = rotate_half(tensor)?;
    let cos_term = tensor.mul(cos)?;
    let sin_term = rotated.mul(sin)?;
    let output = cos_term.add(&sin_term)?;

    // Cast back to original dtype
    output.astype(orig_dtype)
}

/// Apply rotary position embedding to vision tensor (internal)
///
/// # Arguments
/// * `tensor` - Input tensor of shape `[batch, seq_len, num_heads, head_dim]`
///   or `[seq_len, num_heads, head_dim]`
/// * `freqs` - Frequency tensor of shape `[seq_len, dim]` where `dim = head_dim/2`
///
/// # Returns
/// * Tensor with rotary embedding applied
pub fn apply_rotary_pos_emb_vision(tensor: &MxArray, freqs: &MxArray) -> Result<MxArray> {
    let (cos, sin) = precompute_rotary_cos_sin(freqs)?;
    apply_rotary_pos_emb_vision_precomputed(tensor, &cos, &sin)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vision_rope_creation() {
        let rope = VisionRotaryEmbedding::new(32, None);
        assert_eq!(rope.dim, 32);
        assert_eq!(rope.theta, 10000.0);
    }

    #[test]
    fn test_vision_rope_forward() {
        let rope = VisionRotaryEmbedding::new(32, None);
        let freqs = rope.forward(16).unwrap();

        let shape: Vec<i64> = freqs.shape().unwrap().as_ref().to_vec();
        // [seq_len=16, dim/2=16]
        assert_eq!(shape, vec![16, 16]);
    }

    #[test]
    fn test_rotate_half() {
        // Create tensor [2, 4] -> split into [2, 2] each
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let x = MxArray::from_float32(&data, &[2, 4]).unwrap();

        let rotated = rotate_half(&x).unwrap();
        let result = rotated.to_float32().unwrap();

        // Expected: [-x2, x1] = [[-3, -4, 1, 2], [-7, -8, 5, 6]]
        assert_eq!(result[0], -3.0);
        assert_eq!(result[1], -4.0);
        assert_eq!(result[2], 1.0);
        assert_eq!(result[3], 2.0);
    }

    #[test]
    fn test_mrope_forward_output_shapes() {
        let mrope = MultimodalRoPE::new(128, 131072, Some(500000.0), vec![16, 24, 24]).unwrap();

        let x = MxArray::zeros(&[1, 4, 128], Some(DType::Float32)).unwrap();
        let position_ids = MxArray::zeros(&[3, 1, 4], Some(DType::Float32)).unwrap();

        let (cos, sin) = mrope.forward(&x, &position_ids).unwrap();

        let cos_shape: Vec<i64> = cos.shape().unwrap().as_ref().to_vec();
        let sin_shape: Vec<i64> = sin.shape().unwrap().as_ref().to_vec();
        assert_eq!(cos_shape, vec![3, 1, 4, 128]);
        assert_eq!(sin_shape, vec![3, 1, 4, 128]);
    }

    #[test]
    fn test_mrope_invalid_section_length() {
        let result = MultimodalRoPE::new(128, 131072, Some(500000.0), vec![16, 24]);
        assert!(result.is_err());
    }

    #[test]
    fn test_mrope_getter() {
        let mrope = MultimodalRoPE::new(128, 131072, Some(500000.0), vec![16, 24, 24]).unwrap();
        assert_eq!(mrope.mrope_section_arr(), &[16, 24, 24]);
    }

    #[test]
    fn test_interleaved_selection_axis_assignment() {
        let head_dim = 12i64;
        let mut cos_data = Vec::with_capacity(3 * head_dim as usize);
        for axis in 0..3 {
            for _ in 0..head_dim {
                cos_data.push(axis as f32);
            }
        }
        let cos = MxArray::from_float32(&cos_data, &[3, 1, 1, head_dim]).unwrap();
        let sin = MxArray::zeros(&[3, 1, 1, head_dim], Some(DType::Float32)).unwrap();
        let q = MxArray::ones(&[1, 1, 1, head_dim], Some(DType::Float32)).unwrap();
        let k = MxArray::ones(&[1, 1, 1, head_dim], Some(DType::Float32)).unwrap();

        let (q_out, _k_out) =
            apply_multimodal_rotary_pos_emb_interleaved(&q, &k, &cos, &sin, vec![3, 2, 1]).unwrap();

        let expected: [f32; 12] = [
            0.0, 1.0, 2.0, 0.0, 1.0, 0.0, // sel_half
            0.0, 1.0, 2.0, 0.0, 1.0, 0.0, // mirrored
        ];
        let got = q_out.to_float32().unwrap();
        assert_eq!(got.as_ref(), &expected[..]);
    }

    #[test]
    fn test_interleaved_text_invariance() {
        let rope_dims = 64i32;
        let mrope =
            MultimodalRoPE::new(rope_dims, 4096, Some(100_000.0), vec![11, 11, 10]).unwrap();

        let seq_len = 5i64;
        let mut pos_data = Vec::with_capacity(3 * seq_len as usize);
        for _ in 0..3 {
            for p in 0..seq_len {
                pos_data.push(p as f32);
            }
        }
        let pos = MxArray::from_float32(&pos_data, &[3, 1, seq_len]).unwrap();

        let x = MxArray::zeros(&[1, seq_len, rope_dims as i64], Some(DType::Float32)).unwrap();
        let (cos, sin) = mrope.forward(&x, &pos).unwrap();

        let head_dim = 96i64;
        let heads = 2i64;
        let q = MxArray::random_uniform(
            &[1, heads, seq_len, head_dim],
            0.0,
            1.0,
            Some(DType::Float32),
        )
        .unwrap();
        let k = MxArray::random_uniform(
            &[1, heads, seq_len, head_dim],
            0.0,
            1.0,
            Some(DType::Float32),
        )
        .unwrap();

        let cos_t = cos
            .slice_axis(0, 0, 1)
            .unwrap()
            .reshape(&[1, 1, seq_len, rope_dims as i64])
            .unwrap();
        let sin_t = sin
            .slice_axis(0, 0, 1)
            .unwrap()
            .reshape(&[1, 1, seq_len, rope_dims as i64])
            .unwrap();
        let (q_ref, k_ref) = apply_interleaved_rotary(&q, &k, &cos_t, &sin_t).unwrap();
        let (q_int, k_int) =
            apply_multimodal_rotary_pos_emb_interleaved(&q, &k, &cos, &sin, vec![11, 11, 10])
                .unwrap();

        assert_eq!(
            q_ref.to_float32().unwrap().as_ref(),
            q_int.to_float32().unwrap().as_ref(),
            "interleaved Q must equal single-axis RoPE Q for text tokens (t==h==w)"
        );
        assert_eq!(
            k_ref.to_float32().unwrap().as_ref(),
            k_int.to_float32().unwrap().as_ref(),
            "interleaved K must equal single-axis RoPE K for text tokens (t==h==w)"
        );
    }

    #[test]
    fn test_precomputed_cos_sin_reused_across_layers_matches_per_layer_recompute() {
        let rope_dims = 64i32;
        let mrope_section = vec![11i32, 11, 10];
        let mrope =
            MultimodalRoPE::new(rope_dims, 4096, Some(100_000.0), mrope_section.clone()).unwrap();

        let seq_len = 5i64;
        let mut pos_data = Vec::with_capacity(3 * seq_len as usize);
        for axis in 0..3i64 {
            for p in 0..seq_len {
                pos_data.push((p + axis * 100) as f32);
            }
        }
        let pos = MxArray::from_float32(&pos_data, &[3, 1, seq_len]).unwrap();
        let x = MxArray::zeros(&[1, seq_len, rope_dims as i64], Some(DType::Float32)).unwrap();
        let (cos, sin) = mrope.forward(&x, &pos).unwrap();

        let head_dim = 96i64;
        let heads = 2i64;
        let q_layer1 = MxArray::random_uniform(
            &[1, heads, seq_len, head_dim],
            0.0,
            1.0,
            Some(DType::Float32),
        )
        .unwrap();
        let k_layer1 = MxArray::random_uniform(
            &[1, heads, seq_len, head_dim],
            0.0,
            1.0,
            Some(DType::Float32),
        )
        .unwrap();
        let q_layer2 = MxArray::random_uniform(
            &[1, heads, seq_len, head_dim],
            0.0,
            1.0,
            Some(DType::Float32),
        )
        .unwrap();
        let k_layer2 = MxArray::random_uniform(
            &[1, heads, seq_len, head_dim],
            0.0,
            1.0,
            Some(DType::Float32),
        )
        .unwrap();

        let (q1_ref, k1_ref) = apply_multimodal_rotary_pos_emb_interleaved(
            &q_layer1,
            &k_layer1,
            &cos,
            &sin,
            mrope_section.clone(),
        )
        .unwrap();
        let (q2_ref, k2_ref) = apply_multimodal_rotary_pos_emb_interleaved(
            &q_layer2,
            &k_layer2,
            &cos,
            &sin,
            mrope_section.clone(),
        )
        .unwrap();

        let (cos_final, sin_final) =
            select_interleaved_cos_sin(&cos, &sin, &mrope_section).unwrap();
        let (q1_opt, k1_opt) =
            apply_interleaved_rotary(&q_layer1, &k_layer1, &cos_final, &sin_final).unwrap();
        let (q2_opt, k2_opt) =
            apply_interleaved_rotary(&q_layer2, &k_layer2, &cos_final, &sin_final).unwrap();

        assert_eq!(
            q1_ref.to_float32().unwrap().as_ref(),
            q1_opt.to_float32().unwrap().as_ref(),
            "layer 1 Q must match the pre-fix per-layer recompute"
        );
        assert_eq!(
            k1_ref.to_float32().unwrap().as_ref(),
            k1_opt.to_float32().unwrap().as_ref(),
            "layer 1 K must match the pre-fix per-layer recompute"
        );
        assert_eq!(
            q2_ref.to_float32().unwrap().as_ref(),
            q2_opt.to_float32().unwrap().as_ref(),
            "layer 2 Q (reused cos/sin) must match the pre-fix per-layer recompute"
        );
        assert_eq!(
            k2_ref.to_float32().unwrap().as_ref(),
            k2_opt.to_float32().unwrap().as_ref(),
            "layer 2 K (reused cos/sin) must match the pre-fix per-layer recompute"
        );
    }

    #[test]
    fn test_precomputed_rotary_matches_direct_for_fp32_and_bf16() -> Result<()> {
        let seq_len = 6i64;
        let num_heads = 2i64;
        let head_dim = 8i64;
        let freq_dim = head_dim / 2;

        let freq_data: Vec<f32> = (0..(seq_len * freq_dim) as usize)
            .map(|i| (i as f32) * 0.13)
            .collect();
        let freqs = MxArray::from_float32(&freq_data, &[seq_len, freq_dim])?;

        let (cos, sin) = precompute_rotary_cos_sin(&freqs)?;
        assert_eq!(cos.dtype()?, DType::Float32);
        assert_eq!(sin.dtype()?, DType::Float32);
        assert_eq!(cos.shape()?.as_ref(), &[seq_len, 1, head_dim]);
        assert_eq!(sin.shape()?.as_ref(), &[seq_len, 1, head_dim]);

        let tensor_data: Vec<f32> = (0..(seq_len * num_heads * head_dim) as usize)
            .map(|i| ((i % 29) as f32 - 14.0) * 0.1)
            .collect();
        let tensor_4d_f32 =
            MxArray::from_float32(&tensor_data, &[1, seq_len, num_heads, head_dim])?;
        let tensor_3d_f32 = MxArray::from_float32(&tensor_data, &[seq_len, num_heads, head_dim])?;

        let direct_f32 =
            apply_rotary_pos_emb_vision(&tensor_4d_f32, &freqs)?.squeeze(Some(&[0]))?;
        let precomp_f32 = apply_rotary_pos_emb_vision_precomputed(&tensor_3d_f32, &cos, &sin)?;
        assert_eq!(precomp_f32.dtype()?, DType::Float32);
        assert_eq!(
            direct_f32.to_float32()?.as_ref(),
            precomp_f32.to_float32()?.as_ref()
        );

        let tensor_4d_bf16 = tensor_4d_f32.astype(DType::BFloat16)?;
        let tensor_3d_bf16 = tensor_3d_f32.astype(DType::BFloat16)?;
        let direct_bf16 =
            apply_rotary_pos_emb_vision(&tensor_4d_bf16, &freqs)?.squeeze(Some(&[0]))?;
        let precomp_bf16 = apply_rotary_pos_emb_vision_precomputed(&tensor_3d_bf16, &cos, &sin)?;
        assert_eq!(precomp_bf16.dtype()?, DType::BFloat16);
        assert_eq!(
            direct_bf16.to_float32()?.as_ref(),
            precomp_bf16.to_float32()?.as_ref()
        );

        Ok(())
    }
}
