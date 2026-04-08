use crate::array::MxArray;
use crate::array::attention::{scaled_dot_product_attention, scaled_dot_product_attention_causal};
use crate::nn::{Linear, RMSNorm, RoPE};
use mlx_sys as sys;
use napi::bindgen_prelude::*;

use super::config::Gemma4Config;
use super::layer_cache::Gemma4LayerCache;
use super::quantized_linear::{LinearProj, QuantizedLinear};

/// Trim mask to match K/V sequence length (e.g. after RotatingKVCache eviction).
fn trim_mask(mask: Option<&MxArray>, kv_len: i64) -> Result<Option<MxArray>> {
    match mask {
        Some(m) => {
            let mask_len = m.shape_at(3)?;
            if mask_len != kv_len {
                Ok(Some(m.slice_axis(3, mask_len - kv_len, mask_len)?))
            } else {
                Ok(Some(m.clone()))
            }
        }
        None => Ok(None),
    }
}

// ============================================
// Gemma4 Proportional RoPE (global layers)
// ============================================

/// Gemma4 proportional RoPE for global attention layers.
///
/// 1:1 port of mlx-lm `ProportionalRoPE` (rope_utils.py).
/// Uses inf-padded frequencies with a SINGLE `mx.fast.rope` call.
/// Non-rotated dimensions get `inf` frequency → no rotation (identity).
///
/// Key insight: exponent denominator = full `dims` (head_size), not `rotated_dims`.
/// Only `partial_rotary_factor` fraction of dims are actually rotated.
struct Gemma4ProportionalRoPE {
    /// Pre-computed frequencies for `mx.fast.rope`, shape [dims/2].
    /// First rotated_dims/2 entries: factor * base^(2i / dims)
    /// Remaining entries: inf (causes no rotation in mx.fast.rope)
    freqs: MxArray,
    /// Full head dimension (e.g. 512)
    dims: i32,
}

impl Gemma4ProportionalRoPE {
    /// Create proportional RoPE for global attention.
    ///
    /// Matches mlx-lm rope_utils.py:ProportionalRoPE.__init__
    ///
    /// # Arguments
    /// * `dims` - Full head dimension (e.g. 512)
    /// * `partial_rotary_factor` - Fraction of dims to rotate (e.g. 0.25)
    /// * `base` - RoPE theta (e.g. 1_000_000.0)
    fn new(dims: i32, partial_rotary_factor: f64, base: f64) -> Result<Self> {
        // rotated_dims = int(dims * partial_rotary_factor)
        let rotated_dims = (dims as f64 * partial_rotary_factor) as i32;
        let half_rotated = (rotated_dims / 2) as usize;
        let half_dims = (dims / 2) as usize;
        let nope_dims = half_dims - half_rotated;

        // freqs = concat([base^(arange(0,rotated_dims,2)/dims), full(inf, nope_dims)])
        let mut freqs_data: Vec<f32> = Vec::with_capacity(half_dims);
        for i in 0..half_rotated {
            let exponent = (2 * i) as f64 / dims as f64;
            freqs_data.push(base.powf(exponent) as f32);
        }
        // Pad with inf for non-rotated dimensions (identity rotation)
        freqs_data.extend(std::iter::repeat_n(f32::INFINITY, nope_dims));

        let freqs = MxArray::from_float32(&freqs_data, &[half_dims as i64])?;

        Ok(Self { freqs, dims })
    }

    /// Apply proportional RoPE to tensor in [B, H, T, D] format.
    ///
    /// Single fused `mx.fast.rope` call with inf-padded frequencies.
    /// No split/scatter needed — the kernel handles everything.
    fn forward(&self, x: &MxArray, offset: i32) -> Result<MxArray> {
        let offset_arr = MxArray::from_int32(&[offset], &[1])?;
        let handle = unsafe {
            sys::mlx_fast_rope_with_freqs(
                x.handle.0,
                self.dims, // full head dimension
                false,     // traditional=False (neox-style)
                0.0,       // base ignored when freqs provided
                1.0,       // scale=1.0
                offset_arr.handle.0,
                self.freqs.handle.0,
            )
        };
        MxArray::from_handle(handle, "proportional_rope")
    }
}

// ============================================
// Gemma4 RoPE dispatch (sliding vs global)
// ============================================

/// RoPE variant for Gemma4 attention layers.
enum Gemma4RoPE {
    /// Standard RoPE for sliding (local) attention layers.
    /// Uses `fast.rope(dims=head_dim, base=10K)` — correct because dims == head_size.
    Standard(RoPE),
    /// Proportional RoPE for global (full) attention layers.
    /// Uses mx.fast.rope with precomputed freqs on only the rotated dims.
    Proportional(Gemma4ProportionalRoPE),
}

impl Gemma4RoPE {
    fn forward(&self, x: &MxArray, offset: i32) -> Result<MxArray> {
        match self {
            Self::Standard(rope) => rope.forward(x, Some(offset)),
            Self::Proportional(rope) => rope.forward(x, offset),
        }
    }
}

// ============================================
// Gemma4 Attention
// ============================================

/// Gemma4 multi-head attention with QKV normalization and dual RoPE.
///
/// Key differences from Qwen3.5 attention:
/// 1. No gating (standard attention, not gated)
/// 2. Sliding layers: full RoPE rotation with theta=10K
/// 3. Global layers: proportional RoPE rotation with theta=1M (head_size denominator)
/// 4. Different head dimensions per layer type (sliding vs global)
/// 5. Optional K=V sharing (keys and values share projection weights)
/// 6. Values are also RMS-normalized (scale-free, no learnable weight)
/// 7. Attention scale = 1.0 (QK norm handles scaling; no query_pre_attn_scalar)
pub struct Gemma4Attention {
    q_proj: LinearProj,
    k_proj: LinearProj,
    v_proj: Option<LinearProj>, // None when attention_k_eq_v=true
    o_proj: LinearProj,

    q_norm: RMSNorm,
    k_norm: RMSNorm,
    /// V norm epsilon (scale-free: passes weight=None to rms_norm, matching Python RMSNormNoScale)
    v_norm_eps: f32,

    rope: Gemma4RoPE,

    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    k_is_v: bool,
}

impl Gemma4Attention {
    pub fn new(config: &Gemma4Config, layer_idx: usize) -> Result<Self> {
        let is_sliding = config.is_sliding_layer(layer_idx);
        let is_global = !is_sliding;

        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.effective_kv_heads(is_global);
        let head_dim = config.effective_head_dim(is_global);
        let has_bias = config.attention_bias;

        // K=V sharing only applies to global (full attention) layers.
        // vLLM: use_k_eq_v = self.is_full_attention and config.attention_k_eq_v
        let k_is_v = is_global && config.attention_k_eq_v;

        let q_proj = Linear::new(
            hidden_size as u32,
            (num_heads * head_dim) as u32,
            Some(has_bias),
        )?;
        let k_proj = Linear::new(
            hidden_size as u32,
            (num_kv_heads * head_dim) as u32,
            Some(has_bias),
        )?;

        // When k_is_v, we skip v_proj entirely and reuse k_proj output
        let v_proj = if k_is_v {
            None
        } else {
            Some(LinearProj::Standard(Linear::new(
                hidden_size as u32,
                (num_kv_heads * head_dim) as u32,
                Some(has_bias),
            )?))
        };

        let o_proj = Linear::new(
            (num_heads * head_dim) as u32,
            hidden_size as u32,
            Some(has_bias),
        )?;

        let q_norm = RMSNorm::new(head_dim as u32, Some(config.rms_norm_eps))?;
        let k_norm = RMSNorm::new(head_dim as u32, Some(config.rms_norm_eps))?;
        // V norm is scale-free: passes weight=None to rms_norm
        // Matches Python's RMSNormNoScale: mx.fast.rms_norm(x, None, eps)
        let v_norm_eps = config.rms_norm_eps as f32;

        // RoPE: sliding uses standard RoPE (theta=10K, dims=head_dim).
        // Global uses proportional RoPE (theta=1M, partial rotation via mx.fast.rope).
        let rope = if is_sliding {
            Gemma4RoPE::Standard(RoPE::new(
                config.rope_dims_sliding(),
                Some(false),
                Some(config.rope_local_base_freq),
                None,
            ))
        } else {
            Gemma4RoPE::Proportional(Gemma4ProportionalRoPE::new(
                head_dim,                     // full head dimension (e.g. 512)
                config.partial_rotary_factor, // fraction of dims to rotate (e.g. 0.25)
                config.rope_theta,            // 1M
            )?)
        };

        Ok(Self {
            q_proj: LinearProj::Standard(q_proj),
            k_proj: LinearProj::Standard(k_proj),
            v_proj,
            o_proj: LinearProj::Standard(o_proj),
            q_norm,
            k_norm,
            v_norm_eps,
            rope,
            num_heads,
            num_kv_heads,
            head_dim,
            k_is_v,
        })
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `x` - Input [B, T, hidden_size]
    /// * `mask` - Attention mask
    /// * `cache` - Layer cache (KVCache for global, RotatingKVCache for sliding)
    pub fn forward(
        &self,
        x: &MxArray,
        mask: Option<&MxArray>,
        cache: Option<&mut Gemma4LayerCache>,
        needs_stash: bool,
    ) -> Result<MxArray> {
        let batch = x.shape_at(0)?;
        let seq_len = x.shape_at(1)?;

        // Q/K/V projections
        let queries = self.q_proj.forward(x)?;
        let keys = self.k_proj.forward(x)?;
        let values = if self.k_is_v {
            keys.clone()
        } else {
            self.v_proj.as_ref().unwrap().forward(x)?
        };

        // Reshape to [B, T, H, D]
        let queries =
            queries.reshape(&[batch, seq_len, self.num_heads as i64, self.head_dim as i64])?;
        let keys = keys.reshape(&[
            batch,
            seq_len,
            self.num_kv_heads as i64,
            self.head_dim as i64,
        ])?;
        let values = values.reshape(&[
            batch,
            seq_len,
            self.num_kv_heads as i64,
            self.head_dim as i64,
        ])?;

        // QKV normalization
        let queries = self.q_norm.forward(&queries)?;
        let keys = self.k_norm.forward(&keys)?;
        // V norm: scale-free (weight=None), matching Python's RMSNormNoScale
        let values = {
            let handle = unsafe {
                sys::mlx_fast_rms_norm(values.handle.0, std::ptr::null_mut(), self.v_norm_eps)
            };
            MxArray::from_handle(handle, "v_norm")?
        };

        // Transpose to [B, H, T, D] BEFORE RoPE
        let queries = queries.transpose(Some(&[0, 2, 1, 3]))?;
        let keys = keys.transpose(Some(&[0, 2, 1, 3]))?;
        let values = values.transpose(Some(&[0, 2, 1, 3]))?;

        // Apply RoPE with cache offset
        let offset = cache.as_ref().map_or(0, |c| c.get_offset());
        let queries = self.rope.forward(&queries, offset)?;
        let keys = self.rope.forward(&keys, offset)?;

        // Update cache
        let (keys, values) = if let Some(c) = cache {
            if needs_stash {
                c.update_and_fetch_stash(&keys, &values)?
            } else {
                c.update_and_fetch(&keys, &values)?
            }
        } else {
            (keys, values)
        };

        let mask = trim_mask(mask, keys.shape_at(2)?)?;

        // Scaled dot-product attention with scale=1.0
        let output = if let Some(ref m) = mask {
            scaled_dot_product_attention(&queries, &keys, &values, 1.0, Some(m))?
        } else if seq_len > 1 {
            scaled_dot_product_attention_causal(&queries, &keys, &values, 1.0)?
        } else {
            scaled_dot_product_attention(&queries, &keys, &values, 1.0, None)?
        };

        // Transpose back [B, H, T, D] → [B, T, H*D]
        let output = output.transpose(Some(&[0, 2, 1, 3]))?;
        let output = output.reshape(&[batch, seq_len, (self.num_heads * self.head_dim) as i64])?;

        // Output projection
        self.o_proj.forward(&output)
    }

    /// Forward pass for KV-shared layers.
    ///
    /// Only computes queries; keys and values come from the anchor layer's cache.
    /// The anchor's K/V already have RoPE applied and are in [B, H, T, D] format.
    ///
    /// # Arguments
    /// * `x` - Input [B, T, hidden_size]
    /// * `mask` - Attention mask (may need to be adjusted for anchor's sequence length)
    /// * `shared_keys` - [B, H_kv, T_anchor, D] from anchor layer's cache (RoPE applied)
    /// * `shared_values` - [B, H_kv, T_anchor, D] from anchor layer's cache
    /// * `cache_offset` - RoPE offset for queries (total tokens seen so far, from anchor cache)
    pub fn forward_shared(
        &self,
        x: &MxArray,
        mask: Option<&MxArray>,
        shared_keys: &MxArray,
        shared_values: &MxArray,
        cache_offset: i32,
    ) -> Result<MxArray> {
        let batch = x.shape_at(0)?;
        let seq_len = x.shape_at(1)?;

        // Only compute queries
        let queries = self.q_proj.forward(x)?;
        let queries =
            queries.reshape(&[batch, seq_len, self.num_heads as i64, self.head_dim as i64])?;
        let queries = self.q_norm.forward(&queries)?;

        // Transpose to [B, H, T, D] before RoPE
        let queries = queries.transpose(Some(&[0, 2, 1, 3]))?;

        // Apply RoPE to queries using the anchor's cache offset
        let queries = self.rope.forward(&queries, cache_offset)?;

        let mask = trim_mask(mask, shared_keys.shape_at(2)?)?;

        // Use shared K/V directly (already [B, H_kv, T, D] with RoPE applied)
        let output = if let Some(ref m) = mask {
            scaled_dot_product_attention(&queries, shared_keys, shared_values, 1.0, Some(m))?
        } else if seq_len > 1 {
            scaled_dot_product_attention_causal(&queries, shared_keys, shared_values, 1.0)?
        } else {
            scaled_dot_product_attention(&queries, shared_keys, shared_values, 1.0, None)?
        };

        // Transpose back [B, H, T, D] -> [B, T, H*D]
        let output = output.transpose(Some(&[0, 2, 1, 3]))?;
        let output = output.reshape(&[batch, seq_len, (self.num_heads * self.head_dim) as i64])?;

        // Output projection
        self.o_proj.forward(&output)
    }

    // ========== Weight setters ==========

    pub fn set_q_proj_weight(&mut self, w: &MxArray) -> Result<()> {
        self.q_proj.set_weight(w, "q_proj")
    }
    pub fn set_k_proj_weight(&mut self, w: &MxArray) -> Result<()> {
        self.k_proj.set_weight(w, "k_proj")
    }
    pub fn set_v_proj_weight(&mut self, w: &MxArray) -> Result<()> {
        if let Some(ref mut vp) = self.v_proj {
            vp.set_weight(w, "v_proj")
        } else {
            // k_is_v mode: v_proj doesn't exist, ignore silently
            Ok(())
        }
    }
    pub fn set_o_proj_weight(&mut self, w: &MxArray) -> Result<()> {
        self.o_proj.set_weight(w, "o_proj")
    }
    pub fn set_q_proj_bias(&mut self, b: Option<&MxArray>) -> Result<()> {
        self.q_proj.set_bias(b, "q_proj")
    }
    pub fn set_k_proj_bias(&mut self, b: Option<&MxArray>) -> Result<()> {
        self.k_proj.set_bias(b, "k_proj")
    }
    pub fn set_v_proj_bias(&mut self, b: Option<&MxArray>) -> Result<()> {
        if let Some(ref mut vp) = self.v_proj {
            vp.set_bias(b, "v_proj")
        } else {
            Ok(())
        }
    }
    pub fn set_o_proj_bias(&mut self, b: Option<&MxArray>) -> Result<()> {
        self.o_proj.set_bias(b, "o_proj")
    }
    pub fn set_q_norm_weight(&mut self, w: &MxArray) -> Result<()> {
        self.q_norm.set_weight(w)
    }
    pub fn set_k_norm_weight(&mut self, w: &MxArray) -> Result<()> {
        self.k_norm.set_weight(w)
    }

    // ========== Quantized setters ==========

    pub fn set_quantized_q_proj(&mut self, ql: QuantizedLinear) {
        self.q_proj.set_quantized(ql);
    }
    pub fn set_quantized_k_proj(&mut self, ql: QuantizedLinear) {
        self.k_proj.set_quantized(ql);
    }
    pub fn set_quantized_v_proj(&mut self, ql: QuantizedLinear) {
        if let Some(ref mut vp) = self.v_proj {
            vp.set_quantized(ql);
        }
    }
    pub fn set_quantized_o_proj(&mut self, ql: QuantizedLinear) {
        self.o_proj.set_quantized(ql);
    }
}
