use crate::array::{
    DType, MxArray, scaled_dot_product_attention, scaled_dot_product_attention_causal,
};
use crate::nn::{Linear, RMSNorm, RoPE};
use crate::transformer::kv_cache::KVCache;
use mlx_sys as sys;
use napi::bindgen_prelude::*;
use std::ptr;

/// Multi-head attention with separate Q/K/V projections (Qwen3 style).
///
/// Supports:
/// - Grouped Query Attention (GQA) with different num_heads and num_kv_heads
/// - Optional QK normalization for training stability
/// - RoPE (Rotary Position Embeddings)
/// - KV caching for efficient inference
pub struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Option<RMSNorm>,
    k_norm: Option<RMSNorm>,
    rope: RoPE,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    scale: f64,
    // Config for fused forward
    rope_base: f32,
    qk_norm_eps: f32,
}

impl Attention {
    /// Creates a new multi-head attention layer.
    ///
    /// # Arguments
    /// * `hidden_size` - Model dimension
    /// * `num_heads` - Number of query heads
    /// * `num_kv_heads` - Number of key/value heads (for GQA, typically < num_heads)
    /// * `head_dim` - Dimension per head (optional, defaults to hidden_size / num_heads)
    /// * `rope_theta` - RoPE base frequency (default: 10000)
    /// * `use_qk_norm` - Whether to use QK normalization (Qwen3 feature, default: false)
    /// * `qk_norm_eps` - Epsilon for QK normalization (default: 1e-6)
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

        // Create projections (no bias)
        let q_proj = Linear::new(hidden_size, num_heads * head_dim, Some(false))?;
        let k_proj = Linear::new(hidden_size, num_kv_heads * head_dim, Some(false))?;
        let v_proj = Linear::new(hidden_size, num_kv_heads * head_dim, Some(false))?;
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
        let rope = RoPE::new(head_dim as i32, Some(false), Some(rope_theta), Some(1.0));

        // Attention scale factor
        let scale = 1.0 / (head_dim as f64).sqrt();

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            rope,
            n_heads: num_heads,
            n_kv_heads: num_kv_heads,
            head_dim,
            scale,
            rope_base: rope_theta as f32,
            qk_norm_eps: qk_norm_eps as f32,
        })
    }

    /// Forward pass of attention.
    ///
    /// # Arguments
    /// * `x` - Input tensor, shape: (batch, seq_len, hidden_size)
    /// * `mask` - Optional attention mask
    /// * `cache` - Optional KV cache for incremental generation
    ///
    /// # Returns
    /// Output tensor, shape: (batch, seq_len, hidden_size)
    pub fn forward(
        &self,
        x: &MxArray,
        mask: Option<&MxArray>,
        cache: Option<&mut KVCache>,
    ) -> Result<MxArray> {
        if self.is_quantized() || mask.is_some() {
            return self.forward_quantized(x, mask, cache);
        }

        // Use fused C++ implementation for better performance
        // This reduces ~15 FFI calls to 3 (qkv + cache + output)
        let seq_len = x.shape_at(1)?;

        // Get weight handles
        let w_q = self.q_proj.get_weight();
        let w_k = self.k_proj.get_weight();
        let w_v = self.v_proj.get_weight();
        let w_o = self.o_proj.get_weight();

        // Get optional QK norm weights
        let q_norm_w = self.q_norm.as_ref().map(|n| n.get_weight());
        let k_norm_w = self.k_norm.as_ref().map(|n| n.get_weight());

        // Get RoPE offset from cache BEFORE any updates
        let rope_offset = cache.as_ref().map(|c| c.get_offset()).unwrap_or(0);

        // 1. Fused Q/K/V projection with RoPE (single FFI call)
        // Returns Q, K, V in attention layout (B, n_heads, L, head_dim) with RoPE applied
        let mut q_out: *mut sys::mlx_array = ptr::null_mut();
        let mut k_out: *mut sys::mlx_array = ptr::null_mut();
        let mut v_out: *mut sys::mlx_array = ptr::null_mut();

        unsafe {
            sys::mlx_fused_attention_qkv(
                x.handle.0,
                w_q.handle.0,
                w_k.handle.0,
                w_v.handle.0,
                q_norm_w
                    .as_ref()
                    .map(|w| w.handle.0)
                    .unwrap_or(ptr::null_mut()),
                k_norm_w
                    .as_ref()
                    .map(|w| w.handle.0)
                    .unwrap_or(ptr::null_mut()),
                self.n_heads as i32,
                self.n_kv_heads as i32,
                self.head_dim as i32,
                self.rope_base,
                self.head_dim as i32, // rope_dims = head_dim
                self.qk_norm_eps,
                rope_offset,
                &mut q_out,
                &mut k_out,
                &mut v_out,
            );
        }

        // Check for null pointers (indicates C++ error)
        if q_out.is_null() || k_out.is_null() || v_out.is_null() {
            return Err(napi::Error::from_reason(
                "mlx_fused_attention_qkv returned null pointer",
            ));
        }

        let queries = MxArray::from_handle(q_out, "fused_attention_q")?;
        let keys = MxArray::from_handle(k_out, "fused_attention_k")?;
        let values = MxArray::from_handle(v_out, "fused_attention_v")?;

        // 2. Update KV cache if provided (kept in Rust for complex cache management)
        let (keys, values) = if let Some(cache) = cache {
            cache.update_and_fetch(&keys, &values)?
        } else {
            (keys, values)
        };

        // 3. Fused SDPA + output projection (single FFI call)
        // Determine mask mode based on sequence lengths
        let kv_len = keys.shape_at(2)?;
        // Use causal mode for prefill (seq_len > 1 and seq_len == kv_len)
        // Use "none" mode for generation (seq_len == 1)
        let use_causal = mask.is_none() && seq_len > 1 && seq_len == kv_len;

        let handle = unsafe {
            sys::mlx_fused_attention_output(
                queries.handle.0,
                keys.handle.0,
                values.handle.0,
                w_o.handle.0,
                self.n_heads as i32,
                self.head_dim as i32,
                self.scale as f32,
                use_causal,
            )
        };

        if handle.is_null() {
            return Err(napi::Error::from_reason(
                "mlx_fused_attention_output returned null pointer",
            ));
        }

        MxArray::from_handle(handle, "fused_attention_output")
    }

    /// Component path for packed projections or an explicit attention mask.
    /// The dense fused helper cannot consume an arbitrary mask, and passing a
    /// packed tensor as a raw weight would misinterpret its bytes or force a
    /// full dequantization. Keep every Q/K/V/O projection on `Linear::forward`,
    /// which dispatches packed weights to MLX's fused dequantize-matmul kernel.
    fn forward_quantized(
        &self,
        x: &MxArray,
        mask: Option<&MxArray>,
        cache: Option<&mut KVCache>,
    ) -> Result<MxArray> {
        let batch = x.shape_at(0)?;
        let seq_len = x.shape_at(1)?;
        let rope_offset = cache.as_ref().map(|cache| cache.get_offset()).unwrap_or(0);

        let mut queries = self.q_proj.forward(x)?.reshape(&[
            batch,
            seq_len,
            self.n_heads as i64,
            self.head_dim as i64,
        ])?;
        let mut keys = self.k_proj.forward(x)?.reshape(&[
            batch,
            seq_len,
            self.n_kv_heads as i64,
            self.head_dim as i64,
        ])?;
        let values = self.v_proj.forward(x)?.reshape(&[
            batch,
            seq_len,
            self.n_kv_heads as i64,
            self.head_dim as i64,
        ])?;
        if let Some(norm) = &self.q_norm {
            queries = norm.forward(&queries)?;
        }
        if let Some(norm) = &self.k_norm {
            keys = norm.forward(&keys)?;
        }
        queries = self
            .rope
            .forward(&queries.transpose(Some(&[0, 2, 1, 3]))?, Some(rope_offset))?;
        keys = self
            .rope
            .forward(&keys.transpose(Some(&[0, 2, 1, 3]))?, Some(rope_offset))?;
        let values = values.transpose(Some(&[0, 2, 1, 3]))?;

        let (keys, values) = if let Some(cache) = cache {
            cache.update_and_fetch(&keys, &values)?
        } else {
            (keys, values)
        };
        let kv_len = keys.shape_at(2)?;
        let attended = if let Some(mask) = mask {
            scaled_dot_product_attention(&queries, &keys, &values, self.scale, Some(mask))?
        } else if seq_len > 1 && seq_len == kv_len {
            scaled_dot_product_attention_causal(&queries, &keys, &values, self.scale)?
        } else {
            scaled_dot_product_attention(&queries, &keys, &values, self.scale, None)?
        };
        let attended = attended.transpose(Some(&[0, 2, 1, 3]))?.reshape(&[
            batch,
            seq_len,
            (self.n_heads * self.head_dim) as i64,
        ])?;
        self.o_proj.forward(&attended)
    }

    // Weight setters for loading pretrained models

    pub fn set_q_proj_weight(&mut self, weight: &MxArray) -> Result<()> {
        self.q_proj.set_weight(weight)?;
        Ok(())
    }

    pub fn set_k_proj_weight(&mut self, weight: &MxArray) -> Result<()> {
        self.k_proj.set_weight(weight)?;
        Ok(())
    }

    pub fn set_v_proj_weight(&mut self, weight: &MxArray) -> Result<()> {
        self.v_proj.set_weight(weight)?;
        Ok(())
    }

    pub fn set_o_proj_weight(&mut self, weight: &MxArray) -> Result<()> {
        self.o_proj.set_weight(weight)?;
        Ok(())
    }

    pub(crate) fn q_proj_mut(&mut self) -> &mut Linear {
        &mut self.q_proj
    }

    pub(crate) fn k_proj_mut(&mut self) -> &mut Linear {
        &mut self.k_proj
    }

    pub(crate) fn v_proj_mut(&mut self) -> &mut Linear {
        &mut self.v_proj
    }

    pub(crate) fn o_proj_mut(&mut self) -> &mut Linear {
        &mut self.o_proj
    }

    pub(crate) fn is_quantized(&self) -> bool {
        self.q_proj.is_quantized()
            || self.k_proj.is_quantized()
            || self.v_proj.is_quantized()
            || self.o_proj.is_quantized()
    }

    pub fn set_q_norm_weight(&mut self, weight: &MxArray) -> Result<()> {
        if let Some(ref mut norm) = self.q_norm {
            norm.set_weight(weight)?;
            Ok(())
        } else {
            Err(Error::from_reason(
                "Q normalization is not enabled for this attention layer".to_string(),
            ))
        }
    }

    pub fn set_k_norm_weight(&mut self, weight: &MxArray) -> Result<()> {
        if let Some(ref mut norm) = self.k_norm {
            norm.set_weight(weight)?;
            Ok(())
        } else {
            Err(Error::from_reason(
                "K normalization is not enabled for this attention layer".to_string(),
            ))
        }
    }

    // Weight getters for parameter extraction

    pub fn get_q_proj_weight(&self) -> MxArray {
        self.q_proj.get_weight()
    }

    pub fn get_k_proj_weight(&self) -> MxArray {
        self.k_proj.get_weight()
    }

    pub fn get_v_proj_weight(&self) -> MxArray {
        self.v_proj.get_weight()
    }

    pub fn get_o_proj_weight(&self) -> MxArray {
        self.o_proj.get_weight()
    }

    pub fn get_q_norm_weight(&self) -> Option<MxArray> {
        self.q_norm.as_ref().map(|n| n.get_weight())
    }

    pub fn get_k_norm_weight(&self) -> Option<MxArray> {
        self.k_norm.as_ref().map(|n| n.get_weight())
    }
}

/// Result of Q/K/V computation for paged attention
pub struct QKVResult {
    /// Query tensor: [num_tokens, num_heads, head_dim]
    pub queries: MxArray,
    /// Key tensor: [num_tokens, num_kv_heads, head_dim]
    pub keys: MxArray,
    /// Value tensor: [num_tokens, num_kv_heads, head_dim]
    pub values: MxArray,
}

impl Attention {
    /// Compute Q, K, V tensors for paged attention.
    ///
    /// This method computes the query, key, and value tensors with RoPE applied,
    /// formatted for use with paged attention kernels.
    ///
    /// # Arguments
    /// * `x` - Input tensor, shape: [batch, seq_len, hidden_size]
    /// * `rope_offset` - Position offset for RoPE (from cache position)
    ///
    /// # Returns
    /// QKVResult containing:
    /// - queries: [batch * seq_len, num_heads, head_dim]
    /// - keys: [batch * seq_len, num_kv_heads, head_dim]
    /// - values: [batch * seq_len, num_kv_heads, head_dim]
    pub fn compute_qkv(&self, x: &MxArray, rope_offset: i32) -> Result<QKVResult> {
        // Get weight handles
        let w_q = self.q_proj.get_weight();
        let w_k = self.k_proj.get_weight();
        let w_v = self.v_proj.get_weight();

        // Get optional QK norm weights
        let q_norm_w = self.q_norm.as_ref().map(|n| n.get_weight());
        let k_norm_w = self.k_norm.as_ref().map(|n| n.get_weight());

        // Use fused Q/K/V computation with RoPE
        let mut q_out: *mut sys::mlx_array = ptr::null_mut();
        let mut k_out: *mut sys::mlx_array = ptr::null_mut();
        let mut v_out: *mut sys::mlx_array = ptr::null_mut();

        unsafe {
            sys::mlx_fused_attention_qkv(
                x.handle.0,
                w_q.handle.0,
                w_k.handle.0,
                w_v.handle.0,
                q_norm_w
                    .as_ref()
                    .map(|w| w.handle.0)
                    .unwrap_or(ptr::null_mut()),
                k_norm_w
                    .as_ref()
                    .map(|w| w.handle.0)
                    .unwrap_or(ptr::null_mut()),
                self.n_heads as i32,
                self.n_kv_heads as i32,
                self.head_dim as i32,
                self.rope_base,
                self.head_dim as i32,
                self.qk_norm_eps,
                rope_offset,
                &mut q_out,
                &mut k_out,
                &mut v_out,
            );
        }

        if q_out.is_null() || k_out.is_null() || v_out.is_null() {
            return Err(napi::Error::from_reason(
                "mlx_fused_attention_qkv returned null pointer",
            ));
        }

        // Q, K, V are in attention layout: [B, num_heads, L, head_dim]
        // For paged attention, we need: [B * L, num_heads, head_dim]
        let queries_attn = MxArray::from_handle(q_out, "compute_qkv_q")?;
        let keys_attn = MxArray::from_handle(k_out, "compute_qkv_k")?;
        let values_attn = MxArray::from_handle(v_out, "compute_qkv_v")?;

        // Reshape from [B, num_heads, L, head_dim] to [B * L, num_heads, head_dim]
        let batch = queries_attn.shape_at(0)?;
        let seq_len = queries_attn.shape_at(2)?;
        let num_tokens = batch * seq_len;

        // Transpose from [B, num_heads, L, head_dim] to [B, L, num_heads, head_dim]
        // Then reshape to [B * L, num_heads, head_dim]
        let queries = queries_attn.transpose(Some(&[0, 2, 1, 3]))?;
        let queries = queries.reshape(&[num_tokens, self.n_heads as i64, self.head_dim as i64])?;

        let keys = keys_attn.transpose(Some(&[0, 2, 1, 3]))?;
        let keys = keys.reshape(&[num_tokens, self.n_kv_heads as i64, self.head_dim as i64])?;

        let values = values_attn.transpose(Some(&[0, 2, 1, 3]))?;
        let values = values.reshape(&[num_tokens, self.n_kv_heads as i64, self.head_dim as i64])?;

        Ok(QKVResult {
            queries,
            keys,
            values,
        })
    }

    /// Compute one decode token per batch row with row-specific RoPE offsets.
    ///
    /// `x` must have shape `[N, 1, hidden_size]` and `rope_offsets` must be an
    /// int32 array with shape `[N]`. The returned tensors use paged-attention
    /// layout `[N, heads, head_dim]`.
    pub fn compute_qkv_with_offsets(
        &self,
        x: &MxArray,
        rope_offsets: &MxArray,
    ) -> Result<QKVResult> {
        let x_shape = x.shape()?;
        if x_shape.len() != 3 || x_shape[1] != 1 {
            return Err(Error::from_reason(format!(
                "compute_qkv_with_offsets expects x shape [N, 1, hidden], got {:?}",
                x_shape.as_ref()
            )));
        }
        if x_shape[0] <= 0 {
            return Err(Error::from_reason(
                "compute_qkv_with_offsets requires at least one row",
            ));
        }
        let w_q = self.q_proj.get_weight();
        let hidden_size = w_q.shape_at(1)?;
        if x_shape[2] != hidden_size {
            return Err(Error::from_reason(format!(
                "compute_qkv_with_offsets hidden size mismatch: expected {}, got {}",
                hidden_size, x_shape[2]
            )));
        }
        let offsets_shape = rope_offsets.shape()?;
        if offsets_shape.as_ref() != [x_shape[0]] {
            return Err(Error::from_reason(format!(
                "compute_qkv_with_offsets expects rope_offsets shape [{}], got {:?}",
                x_shape[0],
                offsets_shape.as_ref()
            )));
        }
        if rope_offsets.dtype()? != DType::Int32 {
            return Err(Error::from_reason(format!(
                "compute_qkv_with_offsets expects int32 rope_offsets, got {:?}",
                rope_offsets.dtype()?
            )));
        }

        let w_k = self.k_proj.get_weight();
        let w_v = self.v_proj.get_weight();
        let q_norm_w = self.q_norm.as_ref().map(|n| n.get_weight());
        let k_norm_w = self.k_norm.as_ref().map(|n| n.get_weight());
        let mut q_out: *mut sys::mlx_array = ptr::null_mut();
        let mut k_out: *mut sys::mlx_array = ptr::null_mut();
        let mut v_out: *mut sys::mlx_array = ptr::null_mut();

        unsafe {
            sys::mlx_fused_attention_qkv_with_offsets(
                x.handle.0,
                w_q.handle.0,
                w_k.handle.0,
                w_v.handle.0,
                q_norm_w
                    .as_ref()
                    .map(|w| w.handle.0)
                    .unwrap_or(ptr::null_mut()),
                k_norm_w
                    .as_ref()
                    .map(|w| w.handle.0)
                    .unwrap_or(ptr::null_mut()),
                self.n_heads as i32,
                self.n_kv_heads as i32,
                self.head_dim as i32,
                self.rope_base,
                self.head_dim as i32,
                self.qk_norm_eps,
                rope_offsets.handle.0,
                &mut q_out,
                &mut k_out,
                &mut v_out,
            );
        }
        if q_out.is_null() || k_out.is_null() || v_out.is_null() {
            return Err(Error::from_reason(
                "mlx_fused_attention_qkv_with_offsets returned null pointer",
            ));
        }

        let queries = MxArray::from_handle(q_out, "compute_qkv_with_offsets_q")?
            .transpose(Some(&[0, 2, 1, 3]))?
            .reshape(&[x_shape[0], self.n_heads as i64, self.head_dim as i64])?;
        let keys = MxArray::from_handle(k_out, "compute_qkv_with_offsets_k")?
            .transpose(Some(&[0, 2, 1, 3]))?
            .reshape(&[x_shape[0], self.n_kv_heads as i64, self.head_dim as i64])?;
        let values = MxArray::from_handle(v_out, "compute_qkv_with_offsets_v")?
            .transpose(Some(&[0, 2, 1, 3]))?
            .reshape(&[x_shape[0], self.n_kv_heads as i64, self.head_dim as i64])?;
        Ok(QKVResult {
            queries,
            keys,
            values,
        })
    }

    /// Run output projection on attention output.
    ///
    /// # Arguments
    /// * `attn_output` - Attention output, shape: [batch * seq_len, num_heads, head_dim]
    /// * `batch` - Original batch size
    /// * `seq_len` - Original sequence length
    ///
    /// # Returns
    /// Output tensor, shape: [batch, seq_len, hidden_size]
    pub fn output_projection(
        &self,
        attn_output: &MxArray,
        batch: i64,
        seq_len: i64,
    ) -> Result<MxArray> {
        // attn_output: [batch * seq_len, num_heads, head_dim]
        // -> [batch, seq_len, num_heads * head_dim]
        let hidden_size = (self.n_heads * self.head_dim) as i64;
        let reshaped = attn_output.reshape(&[batch, seq_len, hidden_size])?;

        // Apply output projection
        self.o_proj.forward(&reshaped)
    }

    /// Get attention scale factor
    pub fn get_scale(&self) -> f64 {
        self.scale
    }
}

impl Clone for Attention {
    fn clone(&self) -> Self {
        Self {
            q_proj: self.q_proj.clone(),
            k_proj: self.k_proj.clone(),
            v_proj: self.v_proj.clone(),
            o_proj: self.o_proj.clone(),
            q_norm: self.q_norm.clone(),
            k_norm: self.k_norm.clone(),
            rope: self.rope.clone(),
            n_heads: self.n_heads,
            n_kv_heads: self.n_kv_heads,
            head_dim: self.head_dim,
            scale: self.scale,
            rope_base: self.rope_base,
            qk_norm_eps: self.qk_norm_eps,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_bit_equal(label: &str, actual: &MxArray, expected: &MxArray) {
        assert_eq!(
            actual.shape().expect("actual shape").as_ref(),
            expected.shape().expect("expected shape").as_ref(),
            "{label} shape"
        );
        let actual = actual.to_float32().expect("actual values");
        let expected = expected.to_float32().expect("expected values");
        assert_eq!(actual.as_ref(), expected.as_ref(), "{label} values");
    }

    fn assert_close(label: &str, actual: &MxArray, expected: &MxArray, tolerance: f32) {
        assert_eq!(
            actual.shape().expect("actual shape").as_ref(),
            expected.shape().expect("expected shape").as_ref(),
            "{label} shape"
        );
        let actual = actual.to_float32().expect("actual values");
        let expected = expected.to_float32().expect("expected values");
        let max_abs = actual
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs <= tolerance,
            "{label} max abs diff {max_abs} exceeds {tolerance}"
        );
    }

    #[test]
    fn array_offset_rope_is_bit_equal_to_scalar_rows() {
        let x = MxArray::from_float32(
            &[
                0.25, -0.5, 0.75, 1.0, -1.25, 1.5, -1.75, 2.0, 0.125, 0.375, -0.625, 0.875, 1.125,
                -1.375, 1.625, -1.875, -0.2, 0.4, -0.6, 0.8, -1.0, 1.2, -1.4, 1.6,
            ],
            &[3, 2, 1, 4],
        )
        .expect("input");
        let offsets = MxArray::from_int32(&[0, 7, 31], &[3]).expect("offsets");
        let batched_handle = unsafe {
            sys::mlx_fast_rope_with_freqs(
                x.handle.0,
                4,
                false,
                10_000.0,
                1.0,
                offsets.handle.0,
                ptr::null_mut(),
            )
        };
        let batched = MxArray::from_handle(batched_handle, "batched rope").expect("batched rope");
        let mut serial = Vec::new();
        for (row, offset) in [0, 7, 31].into_iter().enumerate() {
            let row = x
                .slice(&[row as i64, 0, 0, 0], &[row as i64 + 1, 2, 1, 4])
                .expect("row slice");
            let handle =
                unsafe { sys::mlx_fast_rope(row.handle.0, 4, false, 10_000.0, 1.0, offset) };
            serial.push(MxArray::from_handle(handle, "scalar rope").expect("scalar rope"));
        }
        let expected = MxArray::concatenate_many(serial.iter().collect(), Some(0))
            .expect("concat scalar rope");
        assert_bit_equal("rope", &batched, &expected);
    }

    #[test]
    fn array_offset_qkv_matches_scalar_rows_within_batched_gemm_tolerance() {
        let mut attention =
            Attention::new(8, 2, 1, Some(4), Some(10_000.0), Some(true), None).expect("attention");
        let weights = |len: usize| {
            (0..len)
                .map(|i| ((i as i32 % 17) - 8) as f32 * 0.03125)
                .collect::<Vec<_>>()
        };
        attention
            .q_proj
            .set_weight(&MxArray::from_float32(&weights(64), &[8, 8]).expect("q weight"))
            .expect("set q weight");
        attention
            .k_proj
            .set_weight(&MxArray::from_float32(&weights(32), &[4, 8]).expect("k weight"))
            .expect("set k weight");
        attention
            .v_proj
            .set_weight(&MxArray::from_float32(&weights(32), &[4, 8]).expect("v weight"))
            .expect("set v weight");
        let x = MxArray::from_float32(
            &[
                0.25, -0.5, 0.75, 1.0, -1.25, 1.5, -1.75, 2.0, 0.125, 0.375, -0.625, 0.875, 1.125,
                -1.375, 1.625, -1.875, -0.2, 0.4, -0.6, 0.8, -1.0, 1.2, -1.4, 1.6,
            ],
            &[3, 1, 8],
        )
        .expect("input");
        let offsets = MxArray::from_int32(&[0, 7, 31], &[3]).expect("offsets");

        let batched = attention
            .compute_qkv_with_offsets(&x, &offsets)
            .expect("batched qkv");
        let mut serial = Vec::new();
        for (row, offset) in [0, 7, 31].into_iter().enumerate() {
            let row = x
                .slice(&[row as i64, 0, 0], &[row as i64 + 1, 1, 8])
                .expect("row slice");
            serial.push(attention.compute_qkv(&row, offset).expect("scalar qkv"));
        }
        let expected_q =
            MxArray::concatenate_many(serial.iter().map(|qkv| &qkv.queries).collect(), Some(0))
                .expect("concat q");
        let expected_k =
            MxArray::concatenate_many(serial.iter().map(|qkv| &qkv.keys).collect(), Some(0))
                .expect("concat k");
        let expected_v =
            MxArray::concatenate_many(serial.iter().map(|qkv| &qkv.values).collect(), Some(0))
                .expect("concat v");

        // MLX chooses a different GEMM reduction for M=N than for N separate
        // M=1 projections. RoPE itself is bit-exact (the test above); this
        // bound isolates the expected projection-order difference without
        // permitting a row/offset mix-up.
        assert_close("queries", &batched.queries, &expected_q, 1e-3);
        assert_close("keys", &batched.keys, &expected_k, 1e-3);
        assert_close("values", &batched.values, &expected_v, 1e-3);
    }

    #[test]
    fn array_offset_qkv_rejects_non_decode_shapes_and_offsets() {
        let attention = Attention::new(8, 2, 1, Some(4), None, None, None).expect("attention");
        let prefill = MxArray::zeros(&[2, 3, 8], None).expect("prefill");
        let offsets = MxArray::from_int32(&[0, 1], &[2]).expect("offsets");
        let err = attention
            .compute_qkv_with_offsets(&prefill, &offsets)
            .err()
            .expect("multi-token rows must be rejected");
        assert!(err.to_string().contains("[N, 1, hidden]"));

        let decode = MxArray::zeros(&[2, 1, 8], None).expect("decode");
        let wrong_len = MxArray::from_int32(&[0], &[1]).expect("wrong offsets");
        let err = attention
            .compute_qkv_with_offsets(&decode, &wrong_len)
            .err()
            .expect("offset row mismatch must be rejected");
        assert!(err.to_string().contains("shape [2]"));

        let wrong_dtype = MxArray::from_float32(&[0.0, 1.0], &[2]).expect("float offsets");
        let err = attention
            .compute_qkv_with_offsets(&decode, &wrong_dtype)
            .err()
            .expect("float offsets must be rejected");
        assert!(err.to_string().contains("int32 rope_offsets"));
    }
}
