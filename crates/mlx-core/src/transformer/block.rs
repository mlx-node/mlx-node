use crate::array::MxArray;
use crate::nn::RMSNorm;
use crate::transformer::attention::Attention;
use crate::transformer::kv_cache::KVCache;
use crate::transformer::mlp::MLP;
use mlx_sys as sys;
use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::ptr;

/// Transformer block combining self-attention and MLP with pre-normalization.
///
/// Architecture (Qwen3/Llama style):
/// 1. x = x + self_attn(norm(x))  # Pre-norm + residual
/// 2. x = x + mlp(norm(x))        # Pre-norm + residual
#[napi(js_name = "TransformerBlock")]
pub struct TransformerBlock {
    pub(crate) self_attn: Attention,
    pub(crate) mlp: MLP,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
    // Config for fused forward
    n_heads: i32,
    n_kv_heads: i32,
    head_dim: i32,
    attn_scale: f32,
    rope_base: f32,
    rope_dims: i32,
    norm_eps: f32,
    #[allow(dead_code)] // Stored for future use in fused forward path
    use_qk_norm: bool,
}

#[napi]
impl TransformerBlock {
    /// Creates a new transformer block.
    ///
    /// # Arguments
    /// * `hidden_size` - Model dimension
    /// * `num_heads` - Number of attention heads
    /// * `num_kv_heads` - Number of key/value heads (for GQA)
    /// * `intermediate_size` - FFN hidden dimension
    /// * `rms_norm_eps` - Epsilon for RMSNorm
    /// * `rope_theta` - RoPE base frequency (optional)
    /// * `use_qk_norm` - Whether to use QK normalization (optional)
    /// * `head_dim` - Dimension per head (optional)
    #[napi(constructor)]
    pub fn new(
        hidden_size: u32,
        num_heads: u32,
        num_kv_heads: u32,
        intermediate_size: u32,
        rms_norm_eps: f64,
        rope_theta: Option<f64>,
        use_qk_norm: Option<bool>,
        head_dim: Option<u32>,
    ) -> Result<Self> {
        let head_dim_val = head_dim.unwrap_or(hidden_size / num_heads);
        let rope_theta_val = rope_theta.unwrap_or(10000.0);
        let use_qk_norm_val = use_qk_norm.unwrap_or(false);

        let self_attn = Attention::new(
            hidden_size,
            num_heads,
            num_kv_heads,
            Some(head_dim_val),
            Some(rope_theta_val),
            Some(use_qk_norm_val),
            Some(rms_norm_eps),
        )?;

        let mlp = MLP::new(hidden_size, intermediate_size)?;

        let input_layernorm = RMSNorm::new(hidden_size, Some(rms_norm_eps))?;
        let post_attention_layernorm = RMSNorm::new(hidden_size, Some(rms_norm_eps))?;

        // Store config for fused forward
        let attn_scale = 1.0 / (head_dim_val as f64).sqrt();

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            n_heads: num_heads as i32,
            n_kv_heads: num_kv_heads as i32,
            head_dim: head_dim_val as i32,
            attn_scale: attn_scale as f32,
            rope_base: rope_theta_val as f32,
            rope_dims: head_dim_val as i32,
            norm_eps: rms_norm_eps as f32,
            use_qk_norm: use_qk_norm_val,
        })
    }

    /// Forward pass through transformer block.
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
        // For cached inference, use component-based forward (cache handling is complex)
        // For non-cached (training/prefill), use fused C++ implementation
        if cache.is_some() {
            return self.forward_with_cache(x, mask, cache);
        }

        // Use fused C++ implementation for non-cached forward
        // This reduces ~40 FFI calls to 1 per block
        let input_norm_w = self.input_layernorm.get_weight();
        let post_attn_norm_w = self.post_attention_layernorm.get_weight();
        let w_q = self.self_attn.get_q_proj_weight();
        let w_k = self.self_attn.get_k_proj_weight();
        let w_v = self.self_attn.get_v_proj_weight();
        let w_o = self.self_attn.get_o_proj_weight();
        let q_norm_w = self.self_attn.get_q_norm_weight();
        let k_norm_w = self.self_attn.get_k_norm_weight();
        let w_gate = self.mlp.get_gate_proj_weight();
        let w_up = self.mlp.get_up_proj_weight();
        let w_down = self.mlp.get_down_proj_weight();

        // Determine if we should use causal masking
        let use_causal = mask.is_none(); // Use causal if no explicit mask provided

        let handle = unsafe {
            sys::mlx_fused_transformer_block_forward(
                x.handle.0,
                input_norm_w.handle.0,
                post_attn_norm_w.handle.0,
                w_q.handle.0,
                w_k.handle.0,
                w_v.handle.0,
                w_o.handle.0,
                q_norm_w.map(|w| w.handle.0).unwrap_or(ptr::null_mut()),
                k_norm_w.map(|w| w.handle.0).unwrap_or(ptr::null_mut()),
                w_gate.handle.0,
                w_up.handle.0,
                w_down.handle.0,
                self.n_heads,
                self.n_kv_heads,
                self.head_dim,
                self.attn_scale,
                self.rope_base,
                self.rope_dims,
                self.norm_eps,
                self.norm_eps, // qk_norm_eps (same as norm_eps for Qwen3)
                use_causal,
                0, // rope_offset for non-cached
            )
        };

        MxArray::from_handle(handle, "fused_transformer_block_forward")
    }

    /// Forward pass with KV cache (component-based for complex cache handling)
    fn forward_with_cache(
        &self,
        x: &MxArray,
        mask: Option<&MxArray>,
        cache: Option<&mut KVCache>,
    ) -> Result<MxArray> {
        // 1. Self-attention with pre-norm and residual
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward(&normed, mask, cache)?;
        let h = x.add(&attn_out)?; // Residual connection

        // 2. MLP with pre-norm and residual (uses fused MLP internally)
        let normed = self.post_attention_layernorm.forward(&h)?;
        let mlp_out = self.mlp.forward(&normed)?;
        let out = h.add(&mlp_out)?; // Residual connection

        Ok(out)
    }

    /// Debug method: Forward pass with intermediate states captured
    ///
    /// Returns a map of intermediate activations:
    /// - "after_input_norm": after input layer norm
    /// - "after_attn": attention output
    /// - "after_attn_residual": after attention residual connection
    /// - "after_post_norm": after post-attention layer norm
    /// - "after_mlp": MLP output
    /// - "output": final block output
    #[napi]
    pub fn forward_debug(
        &self,
        x: &MxArray,
        mask: Option<&MxArray>,
        cache: Option<&mut KVCache>,
    ) -> Result<std::collections::HashMap<String, MxArray>> {
        use std::collections::HashMap;
        let mut states = HashMap::new();

        // 1. Input layer norm
        let normed = self.input_layernorm.forward(x)?;
        states.insert("after_input_norm".to_string(), normed.clone());

        // 2. Self-attention
        let attn_out = self.self_attn.forward(&normed, mask, cache)?;
        states.insert("after_attn".to_string(), attn_out.clone());

        // 3. Attention residual
        let h = x.add(&attn_out)?;
        states.insert("after_attn_residual".to_string(), h.clone());

        // 4. Post-attention layer norm
        let normed = self.post_attention_layernorm.forward(&h)?;
        states.insert("after_post_norm".to_string(), normed.clone());

        // 5. MLP
        let mlp_out = self.mlp.forward(&normed)?;
        states.insert("after_mlp".to_string(), mlp_out.clone());

        // 6. Final residual
        let out = h.add(&mlp_out)?;
        states.insert("output".to_string(), out);

        Ok(states)
    }

    // Norm weight getters/setters for parameter management

    #[napi]
    pub fn get_input_layernorm_weight(&self) -> MxArray {
        self.input_layernorm.get_weight()
    }

    #[napi]
    pub fn get_post_attention_layernorm_weight(&self) -> MxArray {
        self.post_attention_layernorm.get_weight()
    }

    #[napi]
    pub fn set_input_layernorm_weight(&mut self, weight: &MxArray) -> Result<()> {
        self.input_layernorm.set_weight(weight)?;
        Ok(())
    }

    #[napi]
    pub fn set_post_attention_layernorm_weight(&mut self, weight: &MxArray) -> Result<()> {
        self.post_attention_layernorm.set_weight(weight)?;
        Ok(())
    }
}

impl Clone for TransformerBlock {
    fn clone(&self) -> Self {
        Self {
            self_attn: self.self_attn.clone(),
            mlp: self.mlp.clone(),
            input_layernorm: self.input_layernorm.clone(),
            post_attention_layernorm: self.post_attention_layernorm.clone(),
            n_heads: self.n_heads,
            n_kv_heads: self.n_kv_heads,
            head_dim: self.head_dim,
            attn_scale: self.attn_scale,
            rope_base: self.rope_base,
            rope_dims: self.rope_dims,
            norm_eps: self.norm_eps,
            use_qk_norm: self.use_qk_norm,
        }
    }
}
