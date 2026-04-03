use crate::array::MxArray;
use crate::array::attention::{scaled_dot_product_attention, scaled_dot_product_attention_causal};
use crate::nn::{Linear, RMSNorm, RoPE};
use napi::bindgen_prelude::*;

use super::config::Gemma4Config;
use super::layer_cache::Gemma4LayerCache;
use super::quantized_linear::{LinearProj, QuantizedLinear};

// ============================================
// Gemma4 Proportional RoPE (global layers)
// ============================================

/// Gemma4 proportional RoPE for global attention layers.
///
/// Different from standard RoPE: frequency exponent denominator = head_size (not rotary_dim).
/// Non-rotated dims get identity rotation (cos=1, sin=0 via zero-padded inv_freq).
///
/// Matches HF `_compute_proportional_rope_parameters` and vLLM `Gemma4RotaryEmbedding`.
/// Uses neox-style `rotate_half`: `[-x2, x1]`.
struct Gemma4ProportionalRoPE {
    /// Pre-computed inverse frequencies [1, head_size/2], zero-padded for non-rotated dims.
    inv_freq: MxArray,
    half_dim: i64,
}

impl Gemma4ProportionalRoPE {
    /// Create proportional RoPE for global attention.
    ///
    /// # Arguments
    /// * `head_size` - Full head dimension for global layers (e.g. 512)
    /// * `rotary_dim` - Number of dims actually rotated (e.g. 64 = head_dim * partial_rotary_factor)
    /// * `base` - RoPE theta (e.g. 1_000_000.0)
    fn new(head_size: i32, rotary_dim: i32, base: f64) -> Result<Self> {
        let rope_angles = rotary_dim / 2; // rotation pairs
        let nope_angles = (head_size / 2) - rope_angles; // non-rotated pairs
        let half_dim = (head_size / 2) as i64;

        // Compute inv_freq with head_size as denominator (NOT rotary_dim).
        // Formula: inv_freq[i] = 1 / (base ^ (2i / head_size))  for i < rope_angles
        //          inv_freq[i] = 0                                for i >= rope_angles
        let total = (rope_angles + nope_angles) as usize;
        let mut inv_freq_data: Vec<f32> = Vec::with_capacity(total);
        for i in 0..rope_angles {
            let exponent = (2 * i) as f64 / head_size as f64;
            inv_freq_data.push((1.0 / base.powf(exponent)) as f32);
        }
        // Zero-pad for non-rotated dims (identity rotation: cos=1, sin=0)
        inv_freq_data.extend(std::iter::repeat_n(0.0f32, nope_angles as usize));

        let inv_freq = MxArray::from_float32(&inv_freq_data, &[1, half_dim])?;

        Ok(Self { inv_freq, half_dim })
    }

    /// Apply proportional RoPE to tensor in [B, H, T, D] format.
    ///
    /// Computes neox-style rotation: `x * cos + rotate_half(x) * sin`
    /// where `rotate_half(x) = [-x2, x1]` (x split at D/2).
    fn forward(&self, x: &MxArray, offset: i32) -> Result<MxArray> {
        let seq_len = x.shape_at(2)?; // T dimension in [B, H, T, D]

        // Position indices: [offset, offset+1, ..., offset+seq_len-1]
        let positions = MxArray::arange(
            offset as f64,
            (offset as i64 + seq_len) as f64,
            Some(1.0),
            None,
        )?;

        // freqs = positions[:, None] * inv_freq[None, :]  -> [T, half_dim]
        let positions = positions.reshape(&[seq_len, 1])?;
        let freqs = positions.mul(&self.inv_freq)?;

        // cos/sin: [T, half_dim]
        let cos_cache = freqs.cos()?;
        let sin_cache = freqs.sin()?;

        // Tile to full dim: [T, half_dim] -> [T, head_size] by repeating each value for both halves
        // neox-style needs [cos, cos] along last dim
        let cos_full = MxArray::concatenate(&cos_cache, &cos_cache, -1)?;
        let sin_full = MxArray::concatenate(&sin_cache, &sin_cache, -1)?;

        // Reshape for broadcasting: [1, 1, T, head_size]
        let head_size = self.half_dim * 2;
        let cos_b = cos_full.reshape(&[1, 1, seq_len, head_size])?;
        let sin_b = sin_full.reshape(&[1, 1, seq_len, head_size])?;

        // Cast to input dtype to avoid f32 promotion
        let x_dtype = x.dtype()?;
        let cos_b = cos_b.astype(x_dtype)?;
        let sin_b = sin_b.astype(x_dtype)?;

        // rotate_half: split x into [x1, x2] at D/2, return [-x2, x1]
        let x1 = x.slice_axis(3, 0, self.half_dim)?;
        let x2 = x.slice_axis(3, self.half_dim, head_size)?;
        let neg_x2 = x2.mul_scalar(-1.0)?;
        let rotated = MxArray::concatenate_many(vec![&neg_x2, &x1], Some(-1))?;

        // Apply rotation: x * cos + rotate_half(x) * sin
        x.mul(&cos_b)?.add(&rotated.mul(&sin_b)?)
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
    /// Frequency exponent denominator = head_size, not rotary_dim.
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
    v_norm: RMSNorm, // Scale-free RMSNorm (weight=ones, no learnable params)

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
        // V norm is scale-free: weight stays at ones, no learnable params.
        // Equivalent to x / sqrt(mean(x^2) + eps).
        let v_norm = RMSNorm::new(head_dim as u32, Some(config.rms_norm_eps))?;

        // RoPE: sliding uses standard RoPE (theta=10K, dims=head_dim).
        // Global uses proportional RoPE (theta=1M, freq denominator=head_size, zero-padded).
        let rope = if is_sliding {
            Gemma4RoPE::Standard(RoPE::new(
                config.rope_dims_sliding(),
                Some(false),
                Some(config.rope_local_base_freq),
                None,
            ))
        } else {
            let rotary_dim = config.rope_dims_global(); // head_dim * partial_rotary_factor (e.g. 64)
            Gemma4RoPE::Proportional(Gemma4ProportionalRoPE::new(
                head_dim,          // global head_size (e.g. 512)
                rotary_dim,        // actual rotation dims (e.g. 64)
                config.rope_theta, // 1M
            )?)
        };

        Ok(Self {
            q_proj: LinearProj::Standard(q_proj),
            k_proj: LinearProj::Standard(k_proj),
            v_proj,
            o_proj: LinearProj::Standard(o_proj),
            q_norm,
            k_norm,
            v_norm,
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
    ) -> Result<MxArray> {
        let batch = x.shape_at(0)?;
        let seq_len = x.shape_at(1)?;

        // Q/K/V projections
        let queries = self.q_proj.forward(x)?;
        let keys = self.k_proj.forward(x)?;
        let values = if self.k_is_v {
            // K=V sharing: use key projection output as values too
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

        // QKV normalization (operates on last dim D, layout-independent)
        // Q and K norms have learnable weights; V norm is scale-free (weight=ones)
        let queries = self.q_norm.forward(&queries)?;
        let keys = self.k_norm.forward(&keys)?;
        let values = self.v_norm.forward(&values)?;

        // Transpose to [B, H, T, D] BEFORE RoPE.
        // mx.fast.rope expects penultimate dim to be sequence positions (T).
        let queries = queries.transpose(Some(&[0, 2, 1, 3]))?;
        let keys = keys.transpose(Some(&[0, 2, 1, 3]))?;
        let values = values.transpose(Some(&[0, 2, 1, 3]))?;

        // Apply RoPE with cache offset (now [B, H, T, D] — T is penultimate)
        let offset = cache.as_ref().map_or(0, |c| c.get_offset());
        let queries = self.rope.forward(&queries, offset)?;
        let keys = self.rope.forward(&keys, offset)?;

        // Update cache, get full K/V sequence, and stash for KV sharing
        let (keys, values) = if let Some(c) = cache {
            c.update_and_fetch_stash(&keys, &values)?
        } else {
            (keys, values)
        };

        // Scaled dot-product attention with scale=1.0
        // Gemma4 uses QKV normalization instead of query_pre_attn_scalar scaling.
        let output = if let Some(m) = mask {
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

        // Use shared K/V directly (already [B, H_kv, T, D] with RoPE applied)
        let output = if let Some(m) = mask {
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
