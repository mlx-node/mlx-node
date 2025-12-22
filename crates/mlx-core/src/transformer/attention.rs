use crate::array::{MxArray, scaled_dot_product_attention};
use crate::nn::{Linear, RMSNorm, RoPE};
use crate::transformer::kv_cache::KVCache;
use mlx_sys as sys;
use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::ptr;

/// Multi-head attention with separate Q/K/V projections (Qwen3 style).
///
/// Supports:
/// - Grouped Query Attention (GQA) with different num_heads and num_kv_heads
/// - Optional QK normalization for training stability
/// - RoPE (Rotary Position Embeddings)
/// - KV caching for efficient inference
#[napi(js_name = "Attention")]
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

#[napi]
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
        let rope = RoPE::new(head_dim as i32, Some(false), Some(rope_theta), Some(1.0))?;

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
    #[napi]
    pub fn forward(
        &self,
        x: &MxArray,
        mask: Option<&MxArray>,
        cache: Option<&mut KVCache>,
    ) -> Result<MxArray> {
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

    /// Forward pass with pre-computed Q/K/V tensors.
    ///
    /// This method skips the Q/K/V computation and uses provided tensors directly.
    /// Useful for paged attention prefill where Q/K/V are already computed for cache.
    ///
    /// # Arguments
    /// * `queries` - Pre-computed Q tensor, shape: (batch, n_heads, seq_len, head_dim)
    /// * `keys` - Pre-computed K tensor, shape: (batch, n_kv_heads, kv_len, head_dim)
    /// * `values` - Pre-computed V tensor, shape: (batch, n_kv_heads, kv_len, head_dim)
    /// * `use_causal` - Whether to use causal masking
    ///
    /// # Returns
    /// Output tensor, shape: (batch, seq_len, hidden_size)
    #[napi]
    pub fn forward_with_qkv(
        &self,
        queries: &MxArray,
        keys: &MxArray,
        values: &MxArray,
        use_causal: bool,
    ) -> Result<MxArray> {
        // Get output projection weight
        let w_o = self.o_proj.get_weight();

        // Fused SDPA + output projection
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

    /// Debug method: Forward pass with intermediate Q/K/V states captured
    #[napi]
    pub fn forward_debug(
        &self,
        x: &MxArray,
        mask: Option<&MxArray>,
        cache: Option<&mut KVCache>,
    ) -> Result<std::collections::HashMap<String, MxArray>> {
        use std::collections::HashMap;
        let mut states = HashMap::new();

        // Use shape_at() to avoid allocating full shape vector
        let batch = x.shape_at(0)?;
        let seq_len = x.shape_at(1)?;

        // 1. Project to Q, K, V
        let queries = self.q_proj.forward(x)?;
        let keys = self.k_proj.forward(x)?;
        let values = self.v_proj.forward(x)?;

        states.insert("after_q_proj".to_string(), queries.clone());
        states.insert("after_k_proj".to_string(), keys.clone());
        states.insert("after_v_proj".to_string(), values.clone());

        // 2. Reshape to multi-head format
        let mut queries =
            queries.reshape(&[batch, seq_len, self.n_heads as i64, self.head_dim as i64])?;
        let mut keys =
            keys.reshape(&[batch, seq_len, self.n_kv_heads as i64, self.head_dim as i64])?;
        let values =
            values.reshape(&[batch, seq_len, self.n_kv_heads as i64, self.head_dim as i64])?;

        // 3. Apply QK normalization BEFORE transpose
        queries = if let Some(ref q_norm) = self.q_norm {
            q_norm.forward(&queries)?
        } else {
            queries
        };
        keys = if let Some(ref k_norm) = self.k_norm {
            k_norm.forward(&keys)?
        } else {
            keys
        };

        // 4. Transpose
        let queries = queries.transpose(Some(&[0, 2, 1, 3]))?;
        let keys = keys.transpose(Some(&[0, 2, 1, 3]))?;
        let values = values.transpose(Some(&[0, 2, 1, 3]))?;

        states.insert("before_rope_q".to_string(), queries.clone());
        states.insert("before_rope_k".to_string(), keys.clone());

        // 5. Apply RoPE
        let offset = cache.as_ref().map(|c| c.get_offset()).unwrap_or(0);
        let queries = self.rope.forward(&queries, Some(offset))?;
        let keys = self.rope.forward(&keys, Some(offset))?;

        states.insert("after_rope_q".to_string(), queries.clone());
        states.insert("after_rope_k".to_string(), keys.clone());
        states.insert("values".to_string(), values.clone());

        // 6. KV cache (skip for debug)
        let (keys, values) = if let Some(cache) = cache {
            cache.update_and_fetch(&keys, &values)?
        } else {
            (keys, values)
        };

        // 7. Scaled dot-product attention
        let output = scaled_dot_product_attention(&queries, &keys, &values, self.scale, mask)?;

        states.insert("attn_output_before_transpose".to_string(), output.clone());

        // 8-10. Transpose, reshape, output projection
        let output = output.transpose(Some(&[0, 2, 1, 3]))?;
        let output = output.reshape(&[batch, seq_len, (self.n_heads * self.head_dim) as i64])?;
        let output = self.o_proj.forward(&output)?;

        states.insert("final_output".to_string(), output);

        Ok(states)
    }

    // Weight setters for loading pretrained models

    #[napi]
    pub fn set_q_proj_weight(&mut self, weight: &MxArray) -> Result<()> {
        self.q_proj.set_weight(weight)?;
        Ok(())
    }

    #[napi]
    pub fn set_k_proj_weight(&mut self, weight: &MxArray) -> Result<()> {
        self.k_proj.set_weight(weight)?;
        Ok(())
    }

    #[napi]
    pub fn set_v_proj_weight(&mut self, weight: &MxArray) -> Result<()> {
        self.v_proj.set_weight(weight)?;
        Ok(())
    }

    #[napi]
    pub fn set_o_proj_weight(&mut self, weight: &MxArray) -> Result<()> {
        self.o_proj.set_weight(weight)?;
        Ok(())
    }

    #[napi]
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

    #[napi]
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

    #[napi]
    pub fn get_q_proj_weight(&self) -> MxArray {
        self.q_proj.get_weight()
    }

    #[napi]
    pub fn get_k_proj_weight(&self) -> MxArray {
        self.k_proj.get_weight()
    }

    #[napi]
    pub fn get_v_proj_weight(&self) -> MxArray {
        self.v_proj.get_weight()
    }

    #[napi]
    pub fn get_o_proj_weight(&self) -> MxArray {
        self.o_proj.get_weight()
    }

    #[napi]
    pub fn get_q_norm_weight(&self) -> Option<MxArray> {
        self.q_norm.as_ref().map(|n| n.get_weight())
    }

    #[napi]
    pub fn get_k_norm_weight(&self) -> Option<MxArray> {
        self.k_norm.as_ref().map(|n| n.get_weight())
    }

    /// Forward pass with cached intermediates for backward pass.
    ///
    /// Returns a vector containing:
    /// - [0]: output (final result)
    /// - [1]: input x
    /// - [2]: queries (after projection, before reshape)
    /// - [3]: keys (after projection, before reshape)
    /// - [4]: values (after projection, before reshape)
    /// - [5]: queries (after reshape/norm/transpose/rope, ready for attention)
    /// - [6]: keys (after reshape/norm/transpose/rope, ready for attention)
    /// - [7]: values (after reshape/transpose, ready for attention)
    /// - [8]: attention output (before transpose back)
    /// - [9]: attention output (after transpose, before reshape)
    /// - [10]: attention output (after reshape, before o_proj)
    ///
    /// Note: This does NOT support KV caching - for training only
    #[napi]
    pub fn forward_with_cache(&self, x: &MxArray) -> Result<Vec<MxArray>> {
        // Use shape_at() to avoid allocating full shape vector
        let batch = x.shape_at(0)?;
        let seq_len = x.shape_at(1)?;

        // 1. Project to Q, K, V
        let queries = self.q_proj.forward(x)?; // (B, L, n_heads * head_dim)
        let keys = self.k_proj.forward(x)?; // (B, L, n_kv_heads * head_dim)
        let values = self.v_proj.forward(x)?; // (B, L, n_kv_heads * head_dim)

        // 2. Reshape to multi-head format: (B, L, n_heads, head_dim)
        let queries_reshaped =
            queries.reshape(&[batch, seq_len, self.n_heads as i64, self.head_dim as i64])?;
        let keys_reshaped =
            keys.reshape(&[batch, seq_len, self.n_kv_heads as i64, self.head_dim as i64])?;
        let values_reshaped =
            values.reshape(&[batch, seq_len, self.n_kv_heads as i64, self.head_dim as i64])?;

        // 3. Apply QK normalization (if enabled)
        let queries_normed = if let Some(ref q_norm) = self.q_norm {
            q_norm.forward(&queries_reshaped)?
        } else {
            queries_reshaped
        };
        let keys_normed = if let Some(ref k_norm) = self.k_norm {
            k_norm.forward(&keys_reshaped)?
        } else {
            keys_reshaped
        };

        // 4. Transpose to (B, n_heads, L, head_dim) for attention
        let queries_transposed = queries_normed.transpose(Some(&[0, 2, 1, 3]))?;
        let keys_transposed = keys_normed.transpose(Some(&[0, 2, 1, 3]))?;
        let values_transposed = values_reshaped.transpose(Some(&[0, 2, 1, 3]))?;

        // 5. Apply RoPE (no offset for training)
        let queries_rope = self.rope.forward(&queries_transposed, Some(0))?;
        let keys_rope = self.rope.forward(&keys_transposed, Some(0))?;

        // 6. Scaled dot-product attention (no mask for training)
        let attn_output = scaled_dot_product_attention(
            &queries_rope,
            &keys_rope,
            &values_transposed,
            self.scale,
            None,
        )?;

        // 7. Transpose back to (B, L, n_heads, head_dim)
        let attn_output_transposed = attn_output.transpose(Some(&[0, 2, 1, 3]))?;

        // 8. Reshape to (B, L, n_heads * head_dim)
        let attn_output_reshaped = attn_output_transposed.reshape(&[
            batch,
            seq_len,
            (self.n_heads * self.head_dim) as i64,
        ])?;

        // 9. Output projection
        let output = self.o_proj.forward(&attn_output_reshaped)?;

        Ok(vec![
            output,
            x.copy()?,
            queries,
            keys,
            values,
            queries_rope,
            keys_rope,
            values_transposed,
            attn_output,
            attn_output_transposed,
            attn_output_reshaped,
        ])
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

    /// Compute Q, K, V for paged attention decode with per-sequence RoPE.
    ///
    /// For decode, each sequence has exactly 1 token but different context lengths,
    /// so each needs a different RoPE position offset.
    ///
    /// # Arguments
    /// * `x` - Input tensor, shape: [num_seqs, 1, hidden_dim]
    /// * `positions` - Per-sequence positions, shape: [num_seqs]
    ///
    /// # Returns
    /// QKVResult with:
    /// - queries: [num_seqs, num_heads, head_dim]
    /// - keys: [num_seqs, num_kv_heads, head_dim]
    /// - values: [num_seqs, num_kv_heads, head_dim]
    pub fn compute_qkv_paged_decode(&self, x: &MxArray, positions: &MxArray) -> Result<QKVResult> {
        let num_seqs = x.shape_at(0)? as usize;
        let hidden_dim = x.shape_at(2)?;

        // Extract positions as Vec<i32>
        positions.eval();
        let positions_vec = positions.to_int32()?;

        if positions_vec.len() != num_seqs {
            return Err(napi::Error::from_reason(format!(
                "positions length {} doesn't match num_seqs {}",
                positions_vec.len(),
                num_seqs
            )));
        }

        // Process each sequence with its own RoPE offset
        let mut q_list = Vec::with_capacity(num_seqs);
        let mut k_list = Vec::with_capacity(num_seqs);
        let mut v_list = Vec::with_capacity(num_seqs);

        for i in 0..num_seqs {
            // Extract single sequence: [1, 1, hidden_dim]
            let seq_input = x.slice(&[i as i64, 0, 0], &[i as i64 + 1, 1, hidden_dim])?;
            let rope_offset = positions_vec[i];

            // Compute QKV with this sequence's RoPE offset
            let qkv = self.compute_qkv(&seq_input, rope_offset)?;

            // QKV shape is [1, num_heads, head_dim] for single token
            // Squeeze removes the batch dimension: -> [num_heads, head_dim]
            let q = qkv.queries.squeeze(Some(&[0]))?;
            let k = qkv.keys.squeeze(Some(&[0]))?;
            let v = qkv.values.squeeze(Some(&[0]))?;

            q_list.push(q);
            k_list.push(k);
            v_list.push(v);
        }

        // Stack results: [num_seqs, num_heads, head_dim]
        let q_refs: Vec<&MxArray> = q_list.iter().collect();
        let k_refs: Vec<&MxArray> = k_list.iter().collect();
        let v_refs: Vec<&MxArray> = v_list.iter().collect();

        let queries = MxArray::stack(q_refs, Some(0))?;
        let keys = MxArray::stack(k_refs, Some(0))?;
        let values = MxArray::stack(v_refs, Some(0))?;

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

    /// Get number of heads
    pub fn get_n_heads(&self) -> u32 {
        self.n_heads
    }

    /// Get number of KV heads
    pub fn get_n_kv_heads(&self) -> u32 {
        self.n_kv_heads
    }

    /// Get head dimension
    pub fn get_head_dim(&self) -> u32 {
        self.head_dim
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
