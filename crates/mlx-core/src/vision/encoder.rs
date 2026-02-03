//! Vision Encoder
//!
//! Internal implementation - not exposed to TypeScript.
//! Transformer encoder layers for vision models.
//! Uses GELU activation and LayerNorm (not RMSNorm like language models).

use crate::array::MxArray;
use crate::array::attention::scaled_dot_product_attention;
use crate::nn::activations::Activations;
use crate::nn::{LayerNorm, Linear};
use crate::vision::rope_vision::apply_rotary_pos_emb_vision;
use napi::bindgen_prelude::*;
use std::sync::Arc;

/// Vision Attention (internal)
///
/// Self-attention for vision transformers with fused QKV projection.
pub struct VisionAttention {
    /// Fused Q/K/V projection [dim, dim * 3]
    qkv: Arc<Linear>,
    /// Output projection [dim, dim]
    out_proj: Arc<Linear>,
    /// Number of attention heads
    num_heads: u32,
    /// Dimension per head
    head_dim: u32,
    /// Attention scale factor
    scale: f32,
}

impl VisionAttention {
    /// Create a new Vision Attention layer
    ///
    /// # Arguments
    /// * `dim` - Model dimension
    /// * `num_heads` - Number of attention heads
    /// * `qkv_weight` - Fused QKV weight [dim * 3, dim]
    /// * `qkv_bias` - Optional QKV bias [dim * 3]
    /// * `out_weight` - Output projection weight [dim, dim]
    /// * `out_bias` - Optional output bias [dim]
    pub fn new(
        dim: u32,
        num_heads: u32,
        qkv_weight: &MxArray,
        qkv_bias: Option<&MxArray>,
        out_weight: &MxArray,
        out_bias: Option<&MxArray>,
    ) -> Result<Self> {
        if num_heads == 0 {
            return Err(Error::new(
                Status::InvalidArg,
                "num_heads must be greater than 0",
            ));
        }
        if !dim.is_multiple_of(num_heads) {
            return Err(Error::new(
                Status::InvalidArg,
                format!(
                    "dim ({}) must be divisible by num_heads ({})",
                    dim, num_heads
                ),
            ));
        }
        let head_dim = dim / num_heads;
        let scale = (head_dim as f32).powf(-0.5);

        let qkv = Linear::from_weights(qkv_weight, qkv_bias)?;
        let out_proj = Linear::from_weights(out_weight, out_bias)?;

        Ok(Self {
            qkv: Arc::new(qkv),
            out_proj: Arc::new(out_proj),
            num_heads,
            head_dim,
            scale,
        })
    }

    /// Build attention mask from cumulative sequence lengths.
    ///
    /// Creates a mask where positions within the same sub-sequence can attend
    /// to each other (mask=0) and positions in different sub-sequences are
    /// penalized (mask=1). This matches the Python mlx-vlm implementation.
    ///
    /// # Arguments
    /// * `cu_seqlens` - Cumulative sequence lengths, e.g. [0, 196, 392]
    /// * `seq_len` - Total sequence length
    /// * `dtype` - Output dtype for the mask
    fn build_attention_mask(
        cu_seqlens: &MxArray,
        seq_len: i64,
        dtype: crate::array::DType,
    ) -> Result<MxArray> {
        // Eval cu_seqlens so we can read values
        cu_seqlens.eval();
        let num_boundaries = cu_seqlens.size()? as usize;

        // positions = arange(0, seq_len) as int32: [seq_len]
        let positions = MxArray::arange(
            0.0,
            seq_len as f64,
            Some(1.0),
            Some(crate::array::DType::Int32),
        )?;

        // Build segment_ids: for each position, which segment it belongs to.
        // segment_ids[i] = number of boundaries in cu_seqlens[1..n-1] that are <= i
        // Start with zeros
        let mut segment_ids = MxArray::zeros(&[seq_len], Some(crate::array::DType::Int32))?;

        // For each interior boundary (skip first=0 and last=seq_len),
        // increment segment_ids where positions >= boundary
        for b in 1..num_boundaries.saturating_sub(1) {
            let boundary_val = cu_seqlens.item_at_int32(b)?;
            let boundary = MxArray::from_int32(&[boundary_val], &[1])?;
            // (positions >= boundary) broadcasts [seq_len] >= [1] -> [seq_len] bool
            let ge = positions.greater_equal(&boundary)?;
            // Cast bool to int32 and accumulate
            let ge_int = ge.astype(crate::array::DType::Int32)?;
            segment_ids = segment_ids.add(&ge_int)?;
        }

        // Build mask: segment_ids[i] != segment_ids[j] -> 1.0, else 0.0
        // row_ids: [seq_len, 1], col_ids: [1, seq_len] -> broadcast to [seq_len, seq_len]
        let row_ids = segment_ids.reshape(&[seq_len, 1])?;
        let col_ids = segment_ids.reshape(&[1, seq_len])?;
        let mask_bool = row_ids.not_equal(&col_ids)?;

        // Convert to additive mask: masked positions get large negative value
        // SDPA adds mask to scores, so -1e9 makes cross-segment attention near-zero
        let mask = mask_bool.astype(dtype)?;
        let neg_inf = MxArray::full(&[1], napi::Either::A(-1e9), Some(dtype))?;
        let mask = mask.mul(&neg_inf)?;
        mask.reshape(&[1, seq_len, seq_len])
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `x` - Input tensor [seq_len, dim]
    /// * `cu_seqlens` - Cumulative sequence lengths for variable-length batching
    /// * `rotary_pos_emb` - Rotary position embeddings [seq_len, head_dim/2]
    ///
    /// # Returns
    /// * Output tensor [seq_len, dim]
    pub fn forward(
        &self,
        x: &MxArray,
        cu_seqlens: &MxArray,
        rotary_pos_emb: Option<&MxArray>,
    ) -> Result<MxArray> {
        let shape = x.shape()?;
        let seq_len = shape[0];
        let _dim = shape[1];

        // QKV projection: [seq_len, dim] -> [seq_len, dim*3]
        let qkv = self.qkv.forward(x)?;

        // Reshape to [seq_len, 3, num_heads, head_dim]
        let qkv = qkv.reshape(&[seq_len, 3, self.num_heads as i64, self.head_dim as i64])?;

        // Transpose to [3, seq_len, num_heads, head_dim]
        let qkv = qkv.transpose(Some(&[1, 0, 2, 3]))?;

        // Split Q, K, V: each [1, seq_len, num_heads, head_dim] -> [seq_len, num_heads, head_dim]
        let q = qkv.slice_axis(0, 0, 1)?.squeeze(Some(&[0]))?;
        let k = qkv.slice_axis(0, 1, 2)?.squeeze(Some(&[0]))?;
        let v = qkv.slice_axis(0, 2, 3)?.squeeze(Some(&[0]))?;

        // Apply rotary position embeddings if provided
        let (q, k) = if let Some(freqs) = rotary_pos_emb {
            let q_expanded =
                q.reshape(&[1, seq_len, self.num_heads as i64, self.head_dim as i64])?;
            let k_expanded =
                k.reshape(&[1, seq_len, self.num_heads as i64, self.head_dim as i64])?;

            let q_rot = apply_rotary_pos_emb_vision(&q_expanded, freqs)?;
            let k_rot = apply_rotary_pos_emb_vision(&k_expanded, freqs)?;

            let q_rot = q_rot.squeeze(Some(&[0]))?;
            let k_rot = k_rot.squeeze(Some(&[0]))?;
            (q_rot, k_rot)
        } else {
            (q, k)
        };

        // Reshape to [1, seq_len, num_heads, head_dim] then transpose to
        // [1, num_heads, seq_len, head_dim] for SDPA
        let q = q
            .reshape(&[1, seq_len, self.num_heads as i64, self.head_dim as i64])?
            .transpose(Some(&[0, 2, 1, 3]))?;
        let k = k
            .reshape(&[1, seq_len, self.num_heads as i64, self.head_dim as i64])?
            .transpose(Some(&[0, 2, 1, 3]))?;
        let v = v
            .reshape(&[1, seq_len, self.num_heads as i64, self.head_dim as i64])?
            .transpose(Some(&[0, 2, 1, 3]))?;

        // Build attention mask from cu_seqlens
        let input_dtype = x.dtype()?;
        let attention_mask = Self::build_attention_mask(cu_seqlens, seq_len, input_dtype)?;

        // Use fused scaled dot-product attention (Metal kernel)
        // mask shape [1, seq_len, seq_len] broadcasts to [1, num_heads, seq_len, seq_len]
        let output =
            scaled_dot_product_attention(&q, &k, &v, self.scale as f64, Some(&attention_mask))?;

        // Transpose back: [1, num_heads, seq_len, head_dim] -> [1, seq_len, num_heads, head_dim]
        let output = output.transpose(Some(&[0, 2, 1, 3]))?;

        // Reshape: [seq_len, dim]
        let output = output.reshape(&[seq_len, (self.num_heads * self.head_dim) as i64])?;

        // Output projection
        self.out_proj.forward(&output)
    }
}

/// Vision MLP (internal)
///
/// Feed-forward network for vision transformers with GELU activation.
pub struct VisionMLP {
    /// First linear layer (expansion)
    fc1: Arc<Linear>,
    /// Second linear layer (projection)
    fc2: Arc<Linear>,
}

impl VisionMLP {
    /// Create a new Vision MLP
    ///
    /// # Arguments
    /// * `fc1_weight` - First layer weight [intermediate_size, dim]
    /// * `fc1_bias` - Optional first layer bias
    /// * `fc2_weight` - Second layer weight [dim, intermediate_size]
    /// * `fc2_bias` - Optional second layer bias
    pub fn new(
        fc1_weight: &MxArray,
        fc1_bias: Option<&MxArray>,
        fc2_weight: &MxArray,
        fc2_bias: Option<&MxArray>,
    ) -> Result<Self> {
        let fc1 = Linear::from_weights(fc1_weight, fc1_bias)?;
        let fc2 = Linear::from_weights(fc2_weight, fc2_bias)?;

        Ok(Self {
            fc1: Arc::new(fc1),
            fc2: Arc::new(fc2),
        })
    }

    /// Forward pass with GELU activation
    ///
    /// # Arguments
    /// * `x` - Input tensor
    ///
    /// # Returns
    /// * Output tensor
    pub fn forward(&self, x: &MxArray) -> Result<MxArray> {
        let hidden = self.fc1.forward(x)?;
        let activated = Activations::gelu(&hidden)?;
        self.fc2.forward(&activated)
    }
}

/// Vision Encoder Layer (internal)
///
/// Single transformer encoder layer for vision models.
/// Uses pre-norm architecture with LayerNorm.
pub struct VisionEncoderLayer {
    /// Pre-attention LayerNorm
    layer_norm1: Arc<LayerNorm>,
    /// Post-attention LayerNorm
    layer_norm2: Arc<LayerNorm>,
    /// Self-attention
    self_attn: Arc<VisionAttention>,
    /// Feed-forward MLP
    mlp: Arc<VisionMLP>,
}

impl VisionEncoderLayer {
    /// Create a new Vision Encoder Layer
    pub fn new(
        layer_norm1: &LayerNorm,
        layer_norm2: &LayerNorm,
        self_attn: &VisionAttention,
        mlp: &VisionMLP,
    ) -> Self {
        Self {
            layer_norm1: Arc::new(layer_norm1.clone()),
            layer_norm2: Arc::new(layer_norm2.clone()),
            self_attn: Arc::new(VisionAttention {
                qkv: self_attn.qkv.clone(),
                out_proj: self_attn.out_proj.clone(),
                num_heads: self_attn.num_heads,
                head_dim: self_attn.head_dim,
                scale: self_attn.scale,
            }),
            mlp: Arc::new(VisionMLP {
                fc1: mlp.fc1.clone(),
                fc2: mlp.fc2.clone(),
            }),
        }
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `hidden_states` - Input tensor [seq_len, dim]
    /// * `cu_seqlens` - Cumulative sequence lengths
    /// * `rotary_pos_emb` - Rotary position embeddings
    ///
    /// # Returns
    /// * Output tensor [seq_len, dim]
    pub fn forward(
        &self,
        hidden_states: &MxArray,
        cu_seqlens: &MxArray,
        rotary_pos_emb: Option<&MxArray>,
    ) -> Result<MxArray> {
        // Self-attention with residual
        let normed = self.layer_norm1.forward(hidden_states)?;
        let attn_output = self
            .self_attn
            .forward(&normed, cu_seqlens, rotary_pos_emb)?;
        let hidden_states = hidden_states.add(&attn_output)?;

        // MLP with residual
        let normed = self.layer_norm2.forward(&hidden_states)?;
        let mlp_output = self.mlp.forward(&normed)?;
        hidden_states.add(&mlp_output)
    }
}

impl Clone for VisionEncoderLayer {
    fn clone(&self) -> Self {
        Self {
            layer_norm1: self.layer_norm1.clone(),
            layer_norm2: self.layer_norm2.clone(),
            self_attn: self.self_attn.clone(),
            mlp: self.mlp.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_array(shape: &[i64]) -> MxArray {
        let size: usize = shape.iter().map(|&s| s as usize).product();
        let data: Vec<f32> = (0..size)
            .map(|i| ((i % 100) as f32 - 50.0) / 100.0)
            .collect();
        MxArray::from_float32(&data, shape).unwrap()
    }

    #[test]
    fn test_vision_mlp() {
        let dim = 32i64;
        let intermediate = 64i64;

        let fc1_weight = random_array(&[intermediate, dim]);
        let fc2_weight = random_array(&[dim, intermediate]);

        let mlp = VisionMLP::new(&fc1_weight, None, &fc2_weight, None).unwrap();

        let input = random_array(&[16, dim]);
        let output = mlp.forward(&input).unwrap();

        let shape: Vec<i64> = output.shape().unwrap().as_ref().to_vec();
        assert_eq!(shape, vec![16, dim]);
    }

    #[test]
    fn test_vision_attention() {
        let dim = 32u32;
        let num_heads = 4u32;
        let seq_len = 16i64;

        let qkv_weight = random_array(&[(dim * 3) as i64, dim as i64]);
        let qkv_bias_data = vec![0.0f32; (dim * 3) as usize];
        let qkv_bias = MxArray::from_float32(&qkv_bias_data, &[(dim * 3) as i64]).unwrap();
        let out_weight = random_array(&[dim as i64, dim as i64]);

        let attn = VisionAttention::new(
            dim,
            num_heads,
            &qkv_weight,
            Some(&qkv_bias),
            &out_weight,
            None,
        )
        .unwrap();

        let input = random_array(&[seq_len, dim as i64]);
        let cu_seqlens_data = vec![0i32, seq_len as i32];
        let cu_seqlens = MxArray::from_int32(&cu_seqlens_data, &[2]).unwrap();

        let output = attn.forward(&input, &cu_seqlens, None).unwrap();

        let shape: Vec<i64> = output.shape().unwrap().as_ref().to_vec();
        assert_eq!(shape, vec![seq_len, dim as i64]);
    }
}
