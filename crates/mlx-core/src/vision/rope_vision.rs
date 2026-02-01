//! Vision Rotary Position Embedding
//!
//! Internal implementation - not exposed to TypeScript.
//! 2D rotary position embeddings for vision transformers.
//! Used for encoding spatial (height, width) positions in image patches.

use crate::array::MxArray;
use napi::bindgen_prelude::*;

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

/// Rotate half of the input tensor
///
/// For input x = [x1, x2], returns [-x2, x1]
pub fn rotate_half(x: &MxArray) -> Result<MxArray> {
    let shape = x.shape()?;
    let last_dim = shape[shape.len() - 1];
    let half_dim = last_dim / 2;

    // Split into two halves along the last dimension
    let x1 = x.slice_axis(shape.len() - 1, 0, half_dim)?;
    let x2 = x.slice_axis(shape.len() - 1, half_dim, last_dim)?;

    // Negate x2
    let neg_x2 = x2.mul_scalar(-1.0)?;

    // Concatenate [-x2, x1]
    MxArray::concatenate_many(vec![&neg_x2, &x1], Some(-1))
}

/// Apply rotary position embedding to vision tensor (internal)
///
/// # Arguments
/// * `tensor` - Input tensor of shape [batch, seq_len, num_heads, head_dim].
///   Only 4D input is supported; 3D input will fail.
/// * `freqs` - Frequency tensor of shape [seq_len, dim] where dim = head_dim/2
///
/// # Returns
/// * Tensor with rotary embedding applied
pub fn apply_rotary_pos_emb_vision(tensor: &MxArray, freqs: &MxArray) -> Result<MxArray> {
    let orig_dtype = tensor.dtype()?;

    // Compute cos and sin from frequencies
    let cos = freqs.cos()?;
    let sin = freqs.sin()?;

    // Expand dims: [seq_len, dim] -> [seq_len, 1, dim]
    let cos = cos.reshape(&[-1, 1, freqs.shape()?[1]])?;
    let sin = sin.reshape(&[-1, 1, freqs.shape()?[1]])?;

    // Tile to double the dimension: [seq_len, 1, dim*2]
    let cos = MxArray::tile(&cos, &[1, 1, 2])?;
    let sin = MxArray::tile(&sin, &[1, 1, 2])?;

    // Add batch dimension: [1, seq_len, 1, dim*2]
    let cos = cos.reshape(&[1, cos.shape()?[0], 1, cos.shape()?[2]])?;
    let sin = sin.reshape(&[1, sin.shape()?[0], 1, sin.shape()?[2]])?;

    // Apply rotary embedding: x * cos + rotate_half(x) * sin
    let rotated = rotate_half(tensor)?;
    let cos_term = tensor.mul(&cos)?;
    let sin_term = rotated.mul(&sin)?;
    let output = cos_term.add(&sin_term)?;

    // Cast back to original dtype
    output.astype(orig_dtype)
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
}
