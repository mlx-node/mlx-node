use crate::array::{MxArray, scaled_dot_product_attention};
use crate::nn::{Linear, RMSNorm, RoPE};
use crate::transformer::kv_cache::KVCache;
use napi::bindgen_prelude::*;
use napi_derive::napi;

/// Multi-head attention with fused QKV projection (Phi3/Llama style).
///
/// More efficient than separate Q/K/V projections - uses a single matrix multiplication.
/// Supports the same features as Attention: GQA, QK normalization, RoPE, KV caching.
#[napi(js_name = "FusedAttention")]
pub struct FusedAttention {
    qkv_proj: Linear,
    o_proj: Linear,
    q_norm: Option<RMSNorm>,
    k_norm: Option<RMSNorm>,
    rope: RoPE,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    scale: f64,
}

#[napi]
impl FusedAttention {
    /// Creates a new multi-head attention layer with fused QKV projection.
    ///
    /// # Arguments
    /// * `hidden_size` - Model dimension
    /// * `num_heads` - Number of query heads
    /// * `num_kv_heads` - Number of key/value heads (for GQA, typically < num_heads)
    /// * `head_dim` - Dimension per head (optional, defaults to hidden_size / num_heads)
    /// * `rope_theta` - RoPE base frequency (default: 10000)
    /// * `use_qk_norm` - Whether to use QK normalization (default: false)
    /// * `qk_norm_eps` - Epsilon for QK normalization (default: 1e-6)
    #[napi(constructor)]
    pub fn new(
        hidden_size: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: Option<u32>,
        rope_theta: Option<f64>,
        use_qk_norm: Option<bool>,
        qk_norm_eps: Option<f64>,
    ) -> Result<Self> {
        let head_dim = head_dim.unwrap_or(hidden_size / num_heads);
        let rope_theta = rope_theta.unwrap_or(10000.0);
        let use_qk_norm = use_qk_norm.unwrap_or(false);
        let qk_norm_eps = qk_norm_eps.unwrap_or(1e-6);

        // Combined QKV projection size = Q + K + V
        let qkv_size = num_heads * head_dim + 2 * num_kv_heads * head_dim;
        let qkv_proj = Linear::new(hidden_size, qkv_size, Some(false))?;
        let o_proj = Linear::new(num_heads * head_dim, hidden_size, Some(false))?;

        // Optional QK normalization
        let q_norm = if use_qk_norm {
            Some(RMSNorm::new(head_dim, Some(qk_norm_eps))?)
        } else {
            None
        };
        let k_norm = if use_qk_norm {
            Some(RMSNorm::new(head_dim, Some(qk_norm_eps))?)
        } else {
            None
        };

        // RoPE
        let rope = RoPE::new(head_dim as i32, Some(false), Some(rope_theta), Some(1.0))?;

        // Attention scale factor
        let scale = 1.0 / (head_dim as f64).sqrt();

        Ok(Self {
            qkv_proj,
            o_proj,
            q_norm,
            k_norm,
            rope,
            n_heads: num_heads,
            n_kv_heads: num_kv_heads,
            head_dim,
            scale,
        })
    }

    /// Forward pass of fused attention.
    ///
    /// # Arguments
    /// * `x` - Input tensor, shape: (batch, seq_len, hidden_size)
    /// * `mask` - Optional attention mask
    /// * `cache` - Optional KV cache for incremental generation
    ///
    /// # Returns
    /// Output tensor, shape: (batch, seq_len, hidden_size)
    #[napi]
    pub fn forward(
        &self,
        x: &MxArray,
        mask: Option<&MxArray>,
        cache: Option<&mut KVCache>,
    ) -> Result<MxArray> {
        // Use shape_at() to avoid allocating full shape vector
        let batch = x.shape_at(0)?;
        let seq_len = x.shape_at(1)?;

        // 1. Single fused QKV projection
        let qkv = self.qkv_proj.forward(x)?; // (B, L, qkv_size)

        // 2. Split into Q, K, V
        let q_size = (self.n_heads * self.head_dim) as i64;
        let kv_size = (self.n_kv_heads * self.head_dim) as i64;

        // Calculate split points
        let q_end = q_size;
        let k_end = q_end + kv_size;

        // Split using slice
        let queries = qkv.slice(&[0, 0, 0], &[batch, seq_len, q_end])?;
        let keys = qkv.slice(&[0, 0, q_end], &[batch, seq_len, k_end])?;
        let values = qkv.slice(&[0, 0, k_end], &[batch, seq_len, k_end + kv_size])?;

        // 3. Reshape to multi-head format: (B, L, n_heads, head_dim)
        let queries =
            queries.reshape(&[batch, seq_len, self.n_heads as i64, self.head_dim as i64])?;
        let keys = keys.reshape(&[batch, seq_len, self.n_kv_heads as i64, self.head_dim as i64])?;
        let values =
            values.reshape(&[batch, seq_len, self.n_kv_heads as i64, self.head_dim as i64])?;

        // 4. Apply QK normalization (if enabled)
        let queries = if let Some(ref q_norm) = self.q_norm {
            q_norm.forward(&queries)?
        } else {
            queries
        };
        let keys = if let Some(ref k_norm) = self.k_norm {
            k_norm.forward(&keys)?
        } else {
            keys
        };

        // 5. Transpose to (B, n_heads, L, head_dim) for attention
        let queries = queries.transpose(Some(&[0, 2, 1, 3]))?;
        let keys = keys.transpose(Some(&[0, 2, 1, 3]))?;
        let values = values.transpose(Some(&[0, 2, 1, 3]))?;

        // 6. Apply RoPE
        let offset = cache.as_ref().map(|c| c.get_offset()).unwrap_or(0);
        let queries = self.rope.forward(&queries, Some(offset))?;
        let keys = self.rope.forward(&keys, Some(offset))?;

        // 7. Update KV cache if provided
        let (keys, values) = if let Some(cache) = cache {
            let (keys, values) = cache.update_and_fetch(&keys, &values)?;
            (
                keys.add(&MxArray::full(&[1], Either::A(0.0), None)?)?,
                values.add(&MxArray::full(&[1], Either::A(0.0), None)?)?,
            )
        } else {
            (keys, values)
        };

        // 8. Scaled dot-product attention
        let output = scaled_dot_product_attention(&queries, &keys, &values, self.scale, mask)?;

        // 9. Transpose back to (B, L, n_heads, head_dim)
        let output = output.transpose(Some(&[0, 2, 1, 3]))?;

        // 10. Reshape to (B, L, n_heads * head_dim)
        let output = output.reshape(&[batch, seq_len, (self.n_heads * self.head_dim) as i64])?;

        // 11. Output projection
        self.o_proj.forward(&output)
    }

    // Weight setters for loading pretrained models

    #[napi]
    pub fn set_qkv_proj_weight(&mut self, weight: &MxArray) -> Result<()> {
        self.qkv_proj.set_weight(weight)?;
        Ok(())
    }

    #[napi]
    pub fn set_o_proj_weight(&mut self, weight: &MxArray) -> Result<()> {
        self.o_proj.set_weight(weight)?;
        Ok(())
    }

    // Weight getters for parameter extraction

    #[napi]
    pub fn get_qkv_proj_weight(&self) -> MxArray {
        self.qkv_proj.get_weight()
    }

    #[napi]
    pub fn get_o_proj_weight(&self) -> MxArray {
        self.o_proj.get_weight()
    }

    #[napi]
    pub fn get_q_norm_weight(&self) -> Option<MxArray> {
        self.q_norm.as_ref().map(|norm| norm.get_weight())
    }

    #[napi]
    pub fn get_k_norm_weight(&self) -> Option<MxArray> {
        self.k_norm.as_ref().map(|norm| norm.get_weight())
    }
}
