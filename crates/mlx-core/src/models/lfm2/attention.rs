use crate::array::MxArray;
use crate::array::attention::{scaled_dot_product_attention, scaled_dot_product_attention_causal};
use crate::nn::{Linear, RMSNorm, RoPE};
use crate::transformer::KVCache;
use napi::bindgen_prelude::*;

/// LFM2 multi-head attention with QK RMSNorm and RoPE.
///
/// Follows `lfm2.py:53-109` (Attention class).
///
/// Key features:
/// - GQA: 32 query heads, 8 KV heads (head_dim=64)
/// - Per-head RMSNorm on Q and K (not V)
/// - Standard RoPE (neox-style, base=1M)
/// - No bias on any projection
pub struct Lfm2Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    q_layernorm: RMSNorm,
    k_layernorm: RMSNorm,
    rope: RoPE,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    scale: f64,
}

impl Lfm2Attention {
    /// Create a new LFM2 attention layer.
    pub fn new(
        hidden_size: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        norm_eps: f64,
        rope_theta: f64,
    ) -> Result<Self> {
        let h = hidden_size as u32;
        let q_dim = (num_heads * head_dim) as u32;
        let kv_dim = (num_kv_heads * head_dim) as u32;

        let q_proj = Linear::new(h, q_dim, Some(false))?;
        let k_proj = Linear::new(h, kv_dim, Some(false))?;
        let v_proj = Linear::new(h, kv_dim, Some(false))?;
        let out_proj = Linear::new(q_dim, h, Some(false))?;

        let q_layernorm = RMSNorm::new(head_dim as u32, Some(norm_eps))?;
        let k_layernorm = RMSNorm::new(head_dim as u32, Some(norm_eps))?;

        let rope = RoPE::new(
            head_dim,
            Some(false), // traditional=False (neox-style)
            Some(rope_theta),
            None,
        );

        let scale = (head_dim as f64).powf(-0.5);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            q_layernorm,
            k_layernorm,
            rope,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
        })
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `x` - Input [B, T, hidden_size]
    /// * `mask` - Optional attention mask
    /// * `cache` - Optional KVCache for incremental decoding
    ///
    /// # Returns
    /// Output tensor [B, T, hidden_size]
    pub fn forward(
        &self,
        x: &MxArray,
        mask: Option<&MxArray>,
        cache: Option<&mut KVCache>,
    ) -> Result<MxArray> {
        let batch = x.shape_at(0)?;
        let seq_len = x.shape_at(1)?;

        // Q/K/V projections
        let queries = self.q_proj.forward(x)?;
        let keys = self.k_proj.forward(x)?;
        let values = self.v_proj.forward(x)?;

        // Reshape to [B, T, num_heads, head_dim] and apply per-head layernorm
        let queries =
            queries.reshape(&[batch, seq_len, self.num_heads as i64, self.head_dim as i64])?;
        let queries = self.q_layernorm.forward(&queries)?;
        // Transpose to [B, num_heads, T, head_dim]
        let queries = queries.transpose(Some(&[0, 2, 1, 3]))?;

        let keys = keys.reshape(&[
            batch,
            seq_len,
            self.num_kv_heads as i64,
            self.head_dim as i64,
        ])?;
        let keys = self.k_layernorm.forward(&keys)?;
        let keys = keys.transpose(Some(&[0, 2, 1, 3]))?;

        // V: reshape + transpose (no layernorm on V)
        let values = values.reshape(&[
            batch,
            seq_len,
            self.num_kv_heads as i64,
            self.head_dim as i64,
        ])?;
        let values = values.transpose(Some(&[0, 2, 1, 3]))?;

        // Apply RoPE with cache offset
        let offset = cache.as_ref().map_or(0, |c| c.get_offset());
        let queries = self.rope.forward(&queries, Some(offset))?;
        let keys = self.rope.forward(&keys, Some(offset))?;

        // Update KV cache
        let (keys, values) = if let Some(c) = cache {
            c.update_and_fetch(&keys, &values)?
        } else {
            (keys, values)
        };

        // Scaled dot-product attention
        let output = if let Some(m) = mask {
            scaled_dot_product_attention(&queries, &keys, &values, self.scale, Some(m))?
        } else if seq_len > 1 {
            scaled_dot_product_attention_causal(&queries, &keys, &values, self.scale)?
        } else {
            scaled_dot_product_attention(&queries, &keys, &values, self.scale, None)?
        };

        // Transpose back [B, H, T, D] -> [B, T, H*D]
        let output = output.transpose(Some(&[0, 2, 1, 3]))?;
        let output = output.reshape(&[batch, seq_len, (self.num_heads * self.head_dim) as i64])?;

        // Output projection
        self.out_proj.forward(&output)
    }

    // ========== Weight setters ==========

    pub fn set_q_proj_weight(&mut self, w: &MxArray) -> Result<()> {
        self.q_proj.set_weight(w)
    }

    pub fn set_k_proj_weight(&mut self, w: &MxArray) -> Result<()> {
        self.k_proj.set_weight(w)
    }

    pub fn set_v_proj_weight(&mut self, w: &MxArray) -> Result<()> {
        self.v_proj.set_weight(w)
    }

    pub fn set_out_proj_weight(&mut self, w: &MxArray) -> Result<()> {
        self.out_proj.set_weight(w)
    }

    pub fn set_q_layernorm_weight(&mut self, w: &MxArray) -> Result<()> {
        self.q_layernorm.set_weight(w)
    }

    pub fn set_k_layernorm_weight(&mut self, w: &MxArray) -> Result<()> {
        self.k_layernorm.set_weight(w)
    }
}
