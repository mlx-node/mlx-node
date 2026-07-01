/**
 * ERNIE Language Model for PaddleOCR-VL
 *
 * The language model component uses multimodal RoPE (mRoPE) which splits
 * the head dimension into sections for temporal, height, and width positions.
 */
use crate::array::MxArray;
use crate::models::paddleocr_vl::config::TextConfig;
use crate::nn::{Embedding, Linear, RMSNorm};
use mlx_sys as sys;
use napi::bindgen_prelude::*;
use std::sync::Arc;

/// Multimodal Rotary Position Embedding (internal)
///
/// Unlike standard RoPE which uses 1D positions, mRoPE uses 3D positions
/// (temporal, height, width) to encode spatial relationships in vision tokens.
/// The head dimension is split into sections according to `mrope_section`.
///
/// Note: This is an internal implementation detail used by ERNIELanguageModel.
/// Not exposed to TypeScript - users interact with high-level VLModel API.
pub struct MultimodalRoPE {
    /// mRoPE sections [temporal, height, width] (e.g., [16, 24, 24])
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

        let mrope_section_arr: [i32; 3] = [mrope_section[0], mrope_section[1], mrope_section[2]];

        // Compute inverse frequencies: 1 / (base^(2i/dim))
        let half_dim = dim / 2;
        let inv_freq_dim = half_dim as i64;
        let mut inv_freq_data = Vec::with_capacity(half_dim as usize);
        for i in (0..dim).step_by(2) {
            let exp = i as f32 / dim as f32;
            inv_freq_data.push(1.0 / base.powf(exp));
        }
        // Pre-reshape to [1, 1, half_dim, 1] to avoid reshape+astype in every forward call.
        // Already float32, so no astype needed.
        let inv_freq = MxArray::from_float32(&inv_freq_data, &[1, 1, inv_freq_dim, 1])?;

        Ok(Self {
            mrope_section: mrope_section_arr,
            inv_freq: Arc::new(inv_freq),
            inv_freq_dim,
            attention_scaling: 1.0,
        })
    }

    /// Compute cos and sin for rotary embeddings
    ///
    /// # Arguments
    /// * `x` - Input tensor (used only for dtype)
    /// * `position_ids` - Position IDs [3, batch, seq_len] for t, h, w
    ///
    /// # Returns
    /// * Tuple of (cos, sin) each of shape [batch, 1, seq_len, head_dim]
    pub fn forward(&self, x: &MxArray, position_ids: &MxArray) -> Result<(MxArray, MxArray)> {
        let target_dtype = x.dtype()?; // 1 FFI call
        let pos_shape = position_ids.shape()?; // 1 FFI call

        // position_ids: [3, batch, seq_len]
        let batch_size = pos_shape[1];
        let seq_len = pos_shape[2];

        // inv_freq is pre-shaped to [1, 1, half_dim, 1] and already float32.
        // Broadcast to [3, batch, half_dim, 1] — uses cached inv_freq_dim.
        let inv_freq_expanded =
            MxArray::broadcast_to(&self.inv_freq, &[3, batch_size, self.inv_freq_dim, 1])?; // 1 FFI call

        // Expand position_ids: [3, batch, 1, seq_len] and cast to float32
        let pos_expanded = position_ids
            .reshape(&[3, batch_size, 1, seq_len])? // 1 FFI call
            .astype(crate::array::DType::Float32)?; // 1 FFI call

        // Compute freqs: inv_freq @ position_ids -> [3, batch, half_dim, seq_len]
        let freqs = inv_freq_expanded.matmul(&pos_expanded)?; // 1 FFI call

        // Transpose: [3, batch, seq_len, half_dim]
        let freqs = freqs.transpose(Some(&[0, 1, 3, 2]))?; // 1 FFI call

        // Concatenate freqs with itself: [3, batch, seq_len, dim]
        let emb = MxArray::concatenate_many(vec![&freqs, &freqs], Some(-1))?; // 1 FFI call

        // Compute cos and sin.
        // Skip mul_scalar when attention_scaling == 1.0 (saves 2 FFI calls).
        let (cos, sin) = if (self.attention_scaling - 1.0).abs() < 1e-8 {
            (emb.cos()?, emb.sin()?) // 2 FFI calls
        } else {
            let cos = emb.cos()?.mul_scalar(self.attention_scaling as f64)?;
            let sin = emb.sin()?.mul_scalar(self.attention_scaling as f64)?;
            (cos, sin)
        };

        // Cast to target dtype only if needed (saves 2 FFI calls when already float32)
        let cos = if target_dtype == crate::array::DType::Float32 {
            cos
        } else {
            cos.astype(target_dtype)? // conditional FFI call
        };
        let sin = if target_dtype == crate::array::DType::Float32 {
            sin
        } else {
            sin.astype(target_dtype)? // conditional FFI call
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

/// Rotate half of the input tensor
///
/// Accepts pre-computed ndim and last_dim to avoid redundant .shape() FFI calls.
fn rotate_half(x: &MxArray, ndim: usize, last_dim: i64) -> Result<MxArray> {
    let half_dim = last_dim / 2;

    let x1 = x.slice_axis(ndim - 1, 0, half_dim)?;
    let x2 = x.slice_axis(ndim - 1, half_dim, last_dim)?;

    let neg_x2 = x2.mul_scalar(-1.0)?;
    MxArray::concatenate_many(vec![&neg_x2, &x1], Some(-1))
}

/// Apply multimodal rotary position embedding to Q and K (internal)
///
/// Uses split+concat pattern matching Python mlx-vlm for minimal FFI overhead.
/// cos/sin shape: [3, batch, seq_len, head_dim] where dim 0 = [temporal, height, width]
///
/// The mrope_section (e.g. [16, 24, 24]) defines how the head_dim is partitioned.
/// Each section i selects from spatial dimension i%3. After doubling (for sin/cos halves),
/// we split the last axis at cumulative indices and pick the right spatial row for each part,
/// then concatenate back. This mirrors Python's:
///   mrope_section = cumsum(mrope_section * 2)[:-1]
///   cos = concat([m[i%3] for i,m in enumerate(split(cos, mrope_section, axis=-1))], axis=-1)
///
/// Note: Internal implementation detail, not exposed to TypeScript.
pub fn apply_multimodal_rotary_pos_emb(
    q: &MxArray,
    k: &MxArray,
    cos: &MxArray,
    sin: &MxArray,
    mrope_section: Vec<i32>,
) -> Result<(MxArray, MxArray)> {
    // Compute cumulative section boundaries
    // e.g., [16, 24, 24] -> cumsum([32, 48, 48]) = [32, 80, 128]
    let mut boundaries: Vec<i64> = Vec::new();
    let mut cumsum = 0i64;
    for &s in &mrope_section {
        cumsum += (s * 2) as i64;
        boundaries.push(cumsum);
    }

    let cos_shape = cos.shape()?; // 1 FFI call — cache and reuse below
    let batch_size = cos_shape[1];
    let seq_len = cos_shape[2];
    let head_dim = cos_shape[3];

    // Split cos/sin by mRoPE sections and interleave
    // For each section i, take cos/sin from position i mod 3
    let mut cos_parts: Vec<MxArray> = Vec::new();
    let mut sin_parts: Vec<MxArray> = Vec::new();

    let mut start = 0i64;
    for (idx, &end) in boundaries.iter().enumerate() {
        let section_idx = idx % 3;

        // Extract section from cos/sin at position section_idx
        let cos_section = cos.slice_axis(0, section_idx as i64, section_idx as i64 + 1)?;
        let sin_section = sin.slice_axis(0, section_idx as i64, section_idx as i64 + 1)?;

        // Squeeze to remove the first dimension: [batch, seq_len, head_dim]
        let cos_section = cos_section.squeeze(Some(&[0]))?;
        let sin_section = sin_section.squeeze(Some(&[0]))?;

        // Slice the head_dim dimension
        let cos_slice = cos_section.slice_axis(2, start, end)?;
        let sin_slice = sin_section.slice_axis(2, start, end)?;

        cos_parts.push(cos_slice);
        sin_parts.push(sin_slice);

        start = end;
    }

    // Concatenate parts
    let cos_refs: Vec<&MxArray> = cos_parts.iter().collect();
    let sin_refs: Vec<&MxArray> = sin_parts.iter().collect();
    let cos_final = MxArray::concatenate_many(cos_refs, Some(-1))?;
    let sin_final = MxArray::concatenate_many(sin_refs, Some(-1))?;

    // Unsqueeze to [batch, 1, seq_len, head_dim] for broadcasting with heads
    let cos_final = cos_final.reshape(&[batch_size, 1, seq_len, head_dim])?;
    let sin_final = sin_final.reshape(&[batch_size, 1, seq_len, head_dim])?;

    // rotary_dim == head_dim (from cos_shape, already cached above)
    let rotary_dim = head_dim;
    let q_shape = q.shape()?; // 1 FFI call
    let q_dim = q_shape[3];
    let q_ndim = q_shape.len();

    // Split Q and K into rotary and pass-through parts
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

    // Apply rotary: q_rot * cos + rotate_half(q_rot) * sin
    // Pass pre-computed ndim and last_dim to avoid .shape() calls inside rotate_half
    let q_rotated = rotate_half(&q_rot, q_ndim, rotary_dim)?;
    let k_rotated = rotate_half(&k_rot, q_ndim, rotary_dim)?;

    let q_embed = q_rot.mul(&cos_final)?.add(&q_rotated.mul(&sin_final)?)?;
    let k_embed = k_rot.mul(&cos_final)?.add(&k_rotated.mul(&sin_final)?)?;

    // Concatenate rotary and pass-through parts
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

/// Apply INTERLEAVED multimodal rotary position embedding to Q and K (internal).
///
/// Used by Qwen3.5-VL (`qwen3_5` / `qwen3_5_moe`). Unlike PaddleOCR-VL, whose
/// multimodal RoPE assigns the head_dim to spatial axes in CONTIGUOUS chunks
/// (see [`apply_multimodal_rotary_pos_emb`]), Qwen3.5-VL assigns each inv_freq
/// index to a spatial axis with a STRIDE-3 INTERLEAVE. This mirrors mlx-vlm's
/// `_interleaved_position_selector` (rope_utils.py):
///   selector[idx] = 1 (height) for idx in 1, 4, 7, … up to `mrope_section[1] * 3`
///   selector[idx] = 2 (width)  for idx in 2, 5, 8, … up to `mrope_section[2] * 3`
///   selector[idx] = 0 (temporal) otherwise
/// computed over the `half_dim` inv_freq axis, then mirrored across the doubled
/// cos/sin (`emb = concat([freqs, freqs])`).
///
/// The per-frequency axis selection is performed with a `take_along_axis` gather
/// over axis 0 (the `[temporal, height, width]` axis) of the doubled cos/sin,
/// followed by the same `rotate_half` application as the sectioned path.
///
/// For TEXT tokens (temporal == height == width) every axis row of cos/sin is
/// bit-identical, so any selector picks identical values — this produces output
/// bit-identical to the sectioned [`apply_multimodal_rotary_pos_emb`]. See
/// `test_interleaved_text_invariance`.
///
/// `cos`/`sin` shape: `[3, batch, seq_len, head_dim]` (axis 0 = t/h/w).
/// `q`/`k` shape: `[batch, heads, seq_len, q_head_dim]` where `q_head_dim >=
/// head_dim` (the trailing `q_head_dim - head_dim` dims pass through unrotated,
/// matching qwen3_5's partial-rotary factor).
///
/// Split into [`select_interleaved_cos_sin`] (the `position_ids`/dtype-only
/// half — independent of q/k) and [`apply_interleaved_rotary`] (the q/k
/// half). Every full-attention layer in one Qwen3.5-VL forward pass shares
/// byte-identical `position_ids`/`mrope_section`/dtype (`init_mrope_layers`
/// seeds every layer from the same config), so a caller with multiple layers
/// — e.g. `Qwen3_5Attention::forward_paged`'s per-forward-pass `mrope_cache`
/// — computes `select_interleaved_cos_sin` ONCE and reuses it across layers
/// instead of recomputing the cos/sin table + `take_along_axis` gather per
/// layer. This wrapper stays byte-identical to the prior single-function
/// implementation for callers that only ever see one (q, k) pair.
///
/// Note: Internal implementation detail, not exposed to TypeScript.
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
/// Builds the stride-3 per-frequency axis selector (mirroring mlx-vlm's
/// `_interleaved_position_selector`) and gathers the selected
/// `[batch, 1, seq_len, head_dim]` cos/sin rows out of the doubled
/// `[3, batch, seq_len, head_dim]` cos/sin. Depends only on `cos`/`sin`
/// (themselves a pure function of `position_ids` and dtype) and
/// `mrope_section`/`head_dim` — NOT on q/k — so it is safe to compute ONCE
/// per forward pass and reuse across every full-attention layer via
/// [`apply_interleaved_rotary`].
pub fn select_interleaved_cos_sin(
    cos: &MxArray,
    sin: &MxArray,
    mrope_section: &[i32],
) -> Result<(MxArray, MxArray)> {
    let cos_shape = cos.shape()?; // 1 FFI call — cache and reuse below
    let batch_size = cos_shape[1];
    let seq_len = cos_shape[2];
    let head_dim = cos_shape[3];
    let half = head_dim / 2;
    let half_usize = half as usize;

    // Build the per-frequency axis selector over the half_dim inv_freq axis,
    // matching mlx-vlm's `_interleaved_position_selector`, then mirror it across
    // the doubled cos/sin (emb = concat([freqs, freqs]) => sel[j] = sel[j % half]).
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

    // Gather the selected axis per frequency from cos/sin [3, batch, seq, head_dim].
    // indices [1, 1, 1, head_dim] -> broadcast to [1, batch, seq, head_dim].
    let indices = MxArray::from_int32(&sel_doubled, &[1, 1, 1, head_dim])?;
    let indices = MxArray::broadcast_to(&indices, &[1, batch_size, seq_len, head_dim])?;

    let cos_sel = cos.take_along_axis(&indices, 0)?; // [1, batch, seq, head_dim]
    let sin_sel = sin.take_along_axis(&indices, 0)?;

    // [1, batch, seq, head_dim] -> [batch, 1, seq, head_dim] (flat order preserved).
    let cos_final = cos_sel.reshape(&[batch_size, 1, seq_len, head_dim])?;
    let sin_final = sin_sel.reshape(&[batch_size, 1, seq_len, head_dim])?;
    Ok((cos_final, sin_final))
}

/// Q/K half of [`apply_multimodal_rotary_pos_emb_interleaved`]: applies
/// `rotate_half` using an already-selected `(cos_final, sin_final)` pair from
/// [`select_interleaved_cos_sin`]. Must be called once PER LAYER (q/k differ
/// per layer) but does no cos/sin table build or gather work itself.
///
/// `cos_final`/`sin_final` shape: `[batch, 1, seq_len, head_dim]`.
/// `q`/`k` shape: `[batch, heads, seq_len, q_head_dim]` where `q_head_dim >=
/// head_dim` (the trailing `q_head_dim - head_dim` dims pass through
/// unrotated, matching qwen3_5's partial-rotary factor).
pub fn apply_interleaved_rotary(
    q: &MxArray,
    k: &MxArray,
    cos_final: &MxArray,
    sin_final: &MxArray,
) -> Result<(MxArray, MxArray)> {
    // Standard rotate_half application (identical to the sectioned path tail).
    let cos_shape = cos_final.shape()?; // 1 FFI call
    let rotary_dim = cos_shape[3];
    let q_shape = q.shape()?; // 1 FFI call
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

    let q_rotated = rotate_half(&q_rot, q_ndim, rotary_dim)?;
    let k_rotated = rotate_half(&k_rot, q_ndim, rotary_dim)?;

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

/// PaddleOCR Attention with mRoPE (internal)
///
/// Note: This is an internal implementation detail used by PaddleOCRDecoderLayer.
/// Not exposed to TypeScript - users interact with high-level VLModel API.
pub struct PaddleOCRAttention {
    q_proj: Arc<Linear>,
    k_proj: Arc<Linear>,
    v_proj: Arc<Linear>,
    o_proj: Arc<Linear>,
}

impl PaddleOCRAttention {
    /// Get raw weight pointers for C++ forward pass
    pub fn get_weight_ptrs(&self) -> [*mut sys::mlx_array; 4] {
        [
            self.q_proj.get_weight().handle.0,
            self.k_proj.get_weight().handle.0,
            self.v_proj.get_weight().handle.0,
            self.o_proj.get_weight().handle.0,
        ]
    }

    pub fn new(
        _config: TextConfig,
        q_weight: &MxArray,
        k_weight: &MxArray,
        v_weight: &MxArray,
        o_weight: &MxArray,
    ) -> Result<Self> {
        let q_proj = Linear::from_weights(q_weight, None)?;
        let k_proj = Linear::from_weights(k_weight, None)?;
        let v_proj = Linear::from_weights(v_weight, None)?;
        let o_proj = Linear::from_weights(o_weight, None)?;

        Ok(Self {
            q_proj: Arc::new(q_proj),
            k_proj: Arc::new(k_proj),
            v_proj: Arc::new(v_proj),
            o_proj: Arc::new(o_proj),
        })
    }
}

/// PaddleOCR Decoder Layer (internal)
///
/// Note: This is an internal implementation detail used by ERNIELanguageModel.
/// Not exposed to TypeScript - users interact with high-level VLModel API.
pub struct PaddleOCRDecoderLayer {
    self_attn: Arc<PaddleOCRAttention>,
    mlp_gate: Arc<Linear>,
    mlp_up: Arc<Linear>,
    mlp_down: Arc<Linear>,
    input_layernorm: Arc<RMSNorm>,
    post_attention_layernorm: Arc<RMSNorm>,
}

impl PaddleOCRDecoderLayer {
    /// Get all 9 weight pointers for this layer in the order expected by C++:
    /// [input_norm, post_attn_norm, q, k, v, o, gate, up, down]
    pub fn get_weight_ptrs(&self) -> [*mut sys::mlx_array; 9] {
        let attn_ptrs = self.self_attn.get_weight_ptrs();
        [
            self.input_layernorm.get_weight().handle.0,
            self.post_attention_layernorm.get_weight().handle.0,
            attn_ptrs[0], // q_proj
            attn_ptrs[1], // k_proj
            attn_ptrs[2], // v_proj
            attn_ptrs[3], // o_proj
            self.mlp_gate.get_weight().handle.0,
            self.mlp_up.get_weight().handle.0,
            self.mlp_down.get_weight().handle.0,
        ]
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: TextConfig,
        q_weight: &MxArray,
        k_weight: &MxArray,
        v_weight: &MxArray,
        o_weight: &MxArray,
        gate_weight: &MxArray,
        up_weight: &MxArray,
        down_weight: &MxArray,
        input_norm_weight: &MxArray,
        post_attn_norm_weight: &MxArray,
    ) -> Result<Self> {
        let self_attn =
            PaddleOCRAttention::new(config.clone(), q_weight, k_weight, v_weight, o_weight)?;

        let mlp_gate = Linear::from_weights(gate_weight, None)?;
        let mlp_up = Linear::from_weights(up_weight, None)?;
        let mlp_down = Linear::from_weights(down_weight, None)?;

        let input_layernorm = RMSNorm::from_weight(input_norm_weight, Some(config.rms_norm_eps))?;
        let post_attention_layernorm =
            RMSNorm::from_weight(post_attn_norm_weight, Some(config.rms_norm_eps))?;

        Ok(Self {
            self_attn: Arc::new(self_attn),
            mlp_gate: Arc::new(mlp_gate),
            mlp_up: Arc::new(mlp_up),
            mlp_down: Arc::new(mlp_down),
            input_layernorm: Arc::new(input_layernorm),
            post_attention_layernorm: Arc::new(post_attention_layernorm),
        })
    }
}

/// ERNIE Language Model (internal)
///
/// Note: This is an internal implementation detail used by VLModel.
/// Not exposed to TypeScript - users interact with high-level VLModel API.
pub struct ERNIELanguageModel {
    embed_tokens: Arc<Embedding>,
    layers: Vec<Arc<PaddleOCRDecoderLayer>>,
    norm: Arc<RMSNorm>,
    lm_head: Arc<Linear>,
    rotary_emb: Arc<MultimodalRoPE>,
    config: TextConfig,
    /// Stored position IDs for decode phase (computed during prefill with images)
    position_ids: Option<MxArray>,
    /// Stored rope deltas for decode phase offset calculation
    rope_deltas: Option<i64>,
    /// Fused C++ KV cache keys (one per layer)
    fused_kv_keys: Vec<Option<MxArray>>,
    /// Fused C++ KV cache values (one per layer)
    fused_kv_values: Vec<Option<MxArray>>,
    /// Fused C++ cache write position
    fused_cache_idx: i32,
}

impl ERNIELanguageModel {
    /// Create language model (layers will be set separately)
    pub fn new(
        config: TextConfig,
        embed_tokens_weight: &MxArray,
        final_norm_weight: &MxArray,
        lm_head_weight: &MxArray,
    ) -> Result<Self> {
        let embed_tokens = Embedding::from_weight(embed_tokens_weight)?;

        let norm = RMSNorm::from_weight(final_norm_weight, Some(config.rms_norm_eps))?;
        let lm_head = Linear::from_weights(lm_head_weight, None)?;

        let rotary_emb = MultimodalRoPE::new(
            config.head_dim,
            config.max_position_embeddings,
            Some(config.rope_theta),
            config.mrope_section.clone(),
        )?;

        Ok(Self {
            embed_tokens: Arc::new(embed_tokens),
            layers: Vec::new(),
            norm: Arc::new(norm),
            lm_head: Arc::new(lm_head),
            rotary_emb: Arc::new(rotary_emb),
            config,
            position_ids: None,
            rope_deltas: None,
            fused_kv_keys: Vec::new(),
            fused_kv_values: Vec::new(),
            fused_cache_idx: 0,
        })
    }

    /// Add a decoder layer
    pub fn add_layer(&mut self, layer: &PaddleOCRDecoderLayer) {
        self.layers.push(Arc::new(PaddleOCRDecoderLayer {
            self_attn: layer.self_attn.clone(),
            mlp_gate: layer.mlp_gate.clone(),
            mlp_up: layer.mlp_up.clone(),
            mlp_down: layer.mlp_down.clone(),
            input_layernorm: layer.input_layernorm.clone(),
            post_attention_layernorm: layer.post_attention_layernorm.clone(),
        }));
    }

    /// Set position IDs for the current generation sequence
    ///
    /// These are stored during prefill and used for proper position slicing during decode.
    pub fn set_position_state(&mut self, position_ids: MxArray, rope_deltas: i64) {
        self.position_ids = Some(position_ids);
        self.rope_deltas = Some(rope_deltas);
    }

    /// Reset position state (call when processing new image)
    pub fn reset_position_state(&mut self) {
        self.position_ids = None;
        self.rope_deltas = None;
    }

    /// Get stored rope deltas
    pub fn get_rope_deltas(&self) -> Option<i64> {
        self.rope_deltas
    }

    /// Get token embeddings without passing through the model
    pub fn get_embeddings(&self, input_ids: &MxArray) -> Result<MxArray> {
        self.embed_tokens.forward(input_ids)
    }

    /// Get number of layers
    pub fn num_layers(&self) -> u32 {
        self.layers.len() as u32
    }

    /// Get fused cache write position
    pub fn get_fused_cache_offset(&self) -> i32 {
        self.fused_cache_idx
    }

    /// Initialize fused KV cache state for C++ forward pass
    pub fn init_fused_kv_caches(&mut self) {
        let n = self.layers.len();
        self.fused_kv_keys = (0..n).map(|_| None).collect();
        self.fused_kv_values = (0..n).map(|_| None).collect();
        self.fused_cache_idx = 0;
    }

    /// Evaluate all fused KV cache arrays to materialize them.
    /// This is critical for chunked prefill - it forces the computation graph
    /// to be evaluated between chunks, preventing unbounded graph growth.
    pub fn eval_fused_kv_caches(&self) {
        for arr in self.fused_kv_keys.iter().flatten() {
            arr.eval();
        }
        for arr in self.fused_kv_values.iter().flatten() {
            arr.eval();
        }
    }

    /// Reset fused KV cache state
    pub fn reset_fused_kv_caches(&mut self) {
        for k in self.fused_kv_keys.iter_mut() {
            *k = None;
        }
        for v in self.fused_kv_values.iter_mut() {
            *v = None;
        }
        self.fused_cache_idx = 0;
    }

    /// Fused forward pass through C++ for minimal FFI overhead.
    ///
    /// This replaces the per-layer Rust forward_with_cache calls with a single
    /// C++ function that builds the entire computation graph in one FFI call.
    ///
    /// # Arguments
    /// * `input_embeds` - Input embeddings [batch, seq_len, hidden_size]
    /// * `position_ids` - Position IDs [3, batch, seq_len] for mRoPE
    ///
    /// # Returns
    /// * Logits [batch, seq_len, vocab_size]
    pub fn forward_fused(
        &mut self,
        input_embeds: &MxArray,
        position_ids: &MxArray,
    ) -> Result<MxArray> {
        let num_layers = self.layers.len() as i32;

        // Collect all layer weight pointers (9 per layer)
        let mut all_weight_ptrs: Vec<*mut sys::mlx_array> =
            Vec::with_capacity(num_layers as usize * 9);
        for layer in &self.layers {
            let ptrs = layer.get_weight_ptrs();
            all_weight_ptrs.extend_from_slice(&ptrs);
        }

        // Model-level weights
        let final_norm_ptr = self.norm.get_weight().handle.0;
        let lm_head_ptr = self.lm_head.get_weight().handle.0;
        let inv_freq_ptr = self.rotary_emb.get_inv_freq_ptr();
        let mrope_section = self.rotary_emb.mrope_section_arr();

        // Prepare KV cache input pointers
        let mut kv_keys_ptrs: Vec<*mut sys::mlx_array> = Vec::with_capacity(num_layers as usize);
        let mut kv_values_ptrs: Vec<*mut sys::mlx_array> = Vec::with_capacity(num_layers as usize);

        for i in 0..num_layers as usize {
            kv_keys_ptrs.push(
                self.fused_kv_keys[i]
                    .as_ref()
                    .map(|a| a.handle.0)
                    .unwrap_or(std::ptr::null_mut()),
            );
            kv_values_ptrs.push(
                self.fused_kv_values[i]
                    .as_ref()
                    .map(|a| a.handle.0)
                    .unwrap_or(std::ptr::null_mut()),
            );
        }

        // Prepare output buffers
        let mut out_logits: *mut sys::mlx_array = std::ptr::null_mut();
        let mut out_kv_keys: Vec<*mut sys::mlx_array> =
            vec![std::ptr::null_mut(); num_layers as usize];
        let mut out_kv_values: Vec<*mut sys::mlx_array> =
            vec![std::ptr::null_mut(); num_layers as usize];
        let mut out_cache_idx: i32 = 0;

        let config = &self.config;

        unsafe {
            sys::mlx_paddleocr_vl_forward_step(
                input_embeds.handle.0,
                all_weight_ptrs.as_ptr(),
                num_layers,
                final_norm_ptr,
                lm_head_ptr,
                inv_freq_ptr,
                position_ids.handle.0,
                mrope_section.as_ptr(),
                config.hidden_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                config.rms_norm_eps as f32,
                kv_keys_ptrs.as_ptr(),
                kv_values_ptrs.as_ptr(),
                self.fused_cache_idx,
                &mut out_logits,
                out_kv_keys.as_mut_ptr(),
                out_kv_values.as_mut_ptr(),
                &mut out_cache_idx,
            );
        }

        // Update fused KV cache state from output
        self.fused_cache_idx = out_cache_idx;
        for i in 0..num_layers as usize {
            if !out_kv_keys[i].is_null() {
                self.fused_kv_keys[i] = Some(MxArray::from_handle(out_kv_keys[i], "fused_kv_key")?);
            }
            if !out_kv_values[i].is_null() {
                self.fused_kv_values[i] =
                    Some(MxArray::from_handle(out_kv_values[i], "fused_kv_value")?);
            }
        }

        MxArray::from_handle(out_logits, "paddleocr_vl_forward_fused")
    }

    /// Run fused forward pass and extract KV cache arrays for batch merging.
    ///
    /// Runs prefill through the existing single-item FFI, then clones out the
    /// per-layer KV cache arrays and resets cache state for the next item.
    ///
    /// # Returns
    /// * (logits, kv_keys, kv_values, cache_idx) where kv_keys/values are per-layer
    pub fn forward_fused_extract_kv(
        &mut self,
        input_embeds: &MxArray,
        position_ids: &MxArray,
    ) -> Result<(MxArray, Vec<MxArray>, Vec<MxArray>, i32)> {
        let logits = self.forward_fused(input_embeds, position_ids)?;

        // Eval KV caches to materialize them before cloning
        self.eval_fused_kv_caches();

        // Clone out KV arrays
        let num_layers = self.layers.len();
        let mut kv_keys = Vec::with_capacity(num_layers);
        let mut kv_values = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            kv_keys.push(
                self.fused_kv_keys[i]
                    .as_ref()
                    .ok_or_else(|| {
                        Error::new(
                            Status::GenericFailure,
                            format!("KV key cache missing for layer {}", i),
                        )
                    })?
                    .clone(),
            );
            kv_values.push(
                self.fused_kv_values[i]
                    .as_ref()
                    .ok_or_else(|| {
                        Error::new(
                            Status::GenericFailure,
                            format!("KV value cache missing for layer {}", i),
                        )
                    })?
                    .clone(),
            );
        }

        let cache_idx = self.fused_cache_idx;

        // Reset cache state for next item
        self.reset_fused_kv_caches();

        Ok((logits, kv_keys, kv_values, cache_idx))
    }

    /// Batched fused forward pass through C++ with left-padding-aware attention.
    ///
    /// Like forward_fused but takes external KV cache arrays and left_padding
    /// for batched decode. Does NOT update internal cache state.
    ///
    /// # Arguments
    /// * `input_embeds` - Input embeddings [batch, seq_len, hidden_size]
    /// * `position_ids` - Position IDs [3, batch, seq_len] for mRoPE
    /// * `left_padding` - Left padding amounts [batch]
    /// * `kv_keys` - Per-layer KV keys [batch, heads, seq, dim]
    /// * `kv_values` - Per-layer KV values [batch, heads, seq, dim]
    /// * `cache_idx` - Current cache write position
    ///
    /// # Returns
    /// * (logits, updated_kv_keys, updated_kv_values, new_cache_idx)
    pub fn forward_fused_batched(
        &self,
        input_embeds: &MxArray,
        position_ids: &MxArray,
        left_padding: &MxArray,
        kv_keys: &[Option<MxArray>],
        kv_values: &[Option<MxArray>],
        cache_idx: i32,
    ) -> Result<(MxArray, Vec<Option<MxArray>>, Vec<Option<MxArray>>, i32)> {
        let num_layers = self.layers.len() as i32;

        // Collect layer weight pointers
        let mut all_weight_ptrs: Vec<*mut sys::mlx_array> =
            Vec::with_capacity(num_layers as usize * 9);
        for layer in &self.layers {
            let ptrs = layer.get_weight_ptrs();
            all_weight_ptrs.extend_from_slice(&ptrs);
        }

        let final_norm_ptr = self.norm.get_weight().handle.0;
        let lm_head_ptr = self.lm_head.get_weight().handle.0;
        let inv_freq_ptr = self.rotary_emb.get_inv_freq_ptr();
        let mrope_section = self.rotary_emb.mrope_section_arr();

        // KV cache input pointers
        let kv_keys_ptrs: Vec<*mut sys::mlx_array> = kv_keys
            .iter()
            .map(|k| {
                k.as_ref()
                    .map(|a| a.handle.0)
                    .unwrap_or(std::ptr::null_mut())
            })
            .collect();
        let kv_values_ptrs: Vec<*mut sys::mlx_array> = kv_values
            .iter()
            .map(|v| {
                v.as_ref()
                    .map(|a| a.handle.0)
                    .unwrap_or(std::ptr::null_mut())
            })
            .collect();

        // Output buffers
        let mut out_logits: *mut sys::mlx_array = std::ptr::null_mut();
        let mut out_kv_keys: Vec<*mut sys::mlx_array> =
            vec![std::ptr::null_mut(); num_layers as usize];
        let mut out_kv_values: Vec<*mut sys::mlx_array> =
            vec![std::ptr::null_mut(); num_layers as usize];
        let mut out_cache_idx: i32 = 0;

        let config = &self.config;

        unsafe {
            sys::mlx_paddleocr_vl_forward_step_batched(
                input_embeds.handle.0,
                all_weight_ptrs.as_ptr(),
                num_layers,
                final_norm_ptr,
                lm_head_ptr,
                inv_freq_ptr,
                position_ids.handle.0,
                mrope_section.as_ptr(),
                config.hidden_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                config.rms_norm_eps as f32,
                left_padding.handle.0,
                kv_keys_ptrs.as_ptr(),
                kv_values_ptrs.as_ptr(),
                cache_idx,
                &mut out_logits,
                out_kv_keys.as_mut_ptr(),
                out_kv_values.as_mut_ptr(),
                &mut out_cache_idx,
            );
        }

        // Convert output pointers to MxArrays
        let mut new_kv_keys: Vec<Option<MxArray>> = Vec::with_capacity(num_layers as usize);
        let mut new_kv_values: Vec<Option<MxArray>> = Vec::with_capacity(num_layers as usize);
        for i in 0..num_layers as usize {
            new_kv_keys.push(if !out_kv_keys[i].is_null() {
                Some(MxArray::from_handle(out_kv_keys[i], "batched_kv_key")?)
            } else {
                None
            });
            new_kv_values.push(if !out_kv_values[i].is_null() {
                Some(MxArray::from_handle(out_kv_values[i], "batched_kv_value")?)
            } else {
                None
            });
        }

        let logits = MxArray::from_handle(out_logits, "paddleocr_vl_forward_batched")?;
        Ok((logits, new_kv_keys, new_kv_values, out_cache_idx))
    }

    /// Get number of layers (needed for batch KV cache setup)
    pub fn num_layers_usize(&self) -> usize {
        self.layers.len()
    }

    /// Get the embedding layer (for external use during batch generation)
    pub fn get_embedding_layer(&self) -> &Embedding {
        &self.embed_tokens
    }
}

impl Clone for ERNIELanguageModel {
    fn clone(&self) -> Self {
        Self {
            embed_tokens: self.embed_tokens.clone(),
            layers: self.layers.clone(),
            norm: self.norm.clone(),
            lm_head: self.lm_head.clone(),
            rotary_emb: self.rotary_emb.clone(),
            config: self.config.clone(),
            position_ids: None, // Don't clone position state
            rope_deltas: None,
            fused_kv_keys: Vec::new(),
            fused_kv_values: Vec::new(),
            fused_cache_idx: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::DType;

    // Helper to create a test TextConfig with smaller dimensions
    fn test_text_config() -> TextConfig {
        TextConfig {
            model_type: "paddleocr_vl".to_string(),
            hidden_size: 256,
            num_hidden_layers: 2,
            intermediate_size: 512,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            rms_norm_eps: 1e-5,
            vocab_size: 1000,
            max_position_embeddings: 1024,
            rope_theta: 500000.0,
            rope_traditional: false,
            use_bias: false,
            head_dim: 64,
            mrope_section: vec![8, 12, 12],
        }
    }

    #[test]
    fn test_mrope_section_sum() {
        let config = TextConfig::default();
        // [16, 24, 24] * 2 = [32, 48, 48] -> total = 128 = head_dim
        let total: i32 = config.mrope_section.iter().map(|&x| x * 2).sum();
        assert_eq!(total, config.head_dim);
    }

    #[test]
    fn test_mrope_forward_output_shapes() {
        // Test MultimodalRoPE forward pass produces correct shapes
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
        // mRoPE section must have exactly 3 elements
        let result = MultimodalRoPE::new(128, 131072, Some(500000.0), vec![16, 24]);
        assert!(result.is_err());
    }

    #[test]
    fn test_mrope_getter() {
        // Test mRoPE section getter
        let mrope = MultimodalRoPE::new(128, 131072, Some(500000.0), vec![16, 24, 24]).unwrap();
        assert_eq!(mrope.mrope_section_arr(), &[16, 24, 24]);
    }

    #[test]
    fn test_apply_mrope_output_shapes() {
        // Test apply_multimodal_rotary_pos_emb preserves tensor shapes
        let q = MxArray::random_uniform(&[1, 4, 2, 32], 0.0, 1.0, Some(DType::Float32)).unwrap();
        let k = MxArray::random_uniform(&[1, 4, 2, 32], 0.0, 1.0, Some(DType::Float32)).unwrap();
        let cos = MxArray::ones(&[3, 1, 2, 32], Some(DType::Float32)).unwrap();
        let sin = MxArray::zeros(&[3, 1, 2, 32], Some(DType::Float32)).unwrap();

        let (q_out, k_out) =
            apply_multimodal_rotary_pos_emb(&q, &k, &cos, &sin, vec![4, 6, 6]).unwrap();

        let q_out_shape: Vec<i64> = q_out.shape().unwrap().as_ref().to_vec();
        let k_out_shape: Vec<i64> = k_out.shape().unwrap().as_ref().to_vec();
        assert_eq!(q_out_shape, vec![1, 4, 2, 32]);
        assert_eq!(k_out_shape, vec![1, 4, 2, 32]);
    }

    #[test]
    fn test_ernie_language_model_creation() {
        // Test creating ERNIELanguageModel
        let config = test_text_config();

        let embed_weight =
            MxArray::random_uniform(&[1000, 256], 0.0, 1.0, Some(DType::Float32)).unwrap();
        let norm_weight = MxArray::ones(&[256], Some(DType::Float32)).unwrap();
        let lm_head_weight =
            MxArray::random_uniform(&[1000, 256], 0.0, 1.0, Some(DType::Float32)).unwrap();

        let lm =
            ERNIELanguageModel::new(config, &embed_weight, &norm_weight, &lm_head_weight).unwrap();

        assert_eq!(lm.num_layers(), 0); // No layers added yet
    }

    #[test]
    fn test_ernie_language_model_add_layers() {
        // Test adding decoder layers to ERNIELanguageModel
        let config = test_text_config();

        let embed_weight =
            MxArray::random_uniform(&[1000, 256], 0.0, 1.0, Some(DType::Float32)).unwrap();
        let norm_weight = MxArray::ones(&[256], Some(DType::Float32)).unwrap();
        let lm_head_weight =
            MxArray::random_uniform(&[1000, 256], 0.0, 1.0, Some(DType::Float32)).unwrap();

        let mut lm =
            ERNIELanguageModel::new(config.clone(), &embed_weight, &norm_weight, &lm_head_weight)
                .unwrap();

        // Create decoder layer weights
        // Q: [num_heads * head_dim, hidden_size] = [4 * 64, 256] = [256, 256]
        // K/V: [num_kv_heads * head_dim, hidden_size] = [2 * 64, 256] = [128, 256]
        let q_weight =
            MxArray::random_uniform(&[256, 256], 0.0, 0.1, Some(DType::Float32)).unwrap();
        let k_weight =
            MxArray::random_uniform(&[128, 256], 0.0, 0.1, Some(DType::Float32)).unwrap();
        let v_weight =
            MxArray::random_uniform(&[128, 256], 0.0, 0.1, Some(DType::Float32)).unwrap();
        let o_weight =
            MxArray::random_uniform(&[256, 256], 0.0, 0.1, Some(DType::Float32)).unwrap();
        let gate_weight =
            MxArray::random_uniform(&[512, 256], 0.0, 0.1, Some(DType::Float32)).unwrap();
        let up_weight =
            MxArray::random_uniform(&[512, 256], 0.0, 0.1, Some(DType::Float32)).unwrap();
        let down_weight =
            MxArray::random_uniform(&[256, 512], 0.0, 0.1, Some(DType::Float32)).unwrap();
        let input_norm_weight = MxArray::ones(&[256], Some(DType::Float32)).unwrap();
        let post_attn_norm_weight = MxArray::ones(&[256], Some(DType::Float32)).unwrap();

        let layer = PaddleOCRDecoderLayer::new(
            config,
            &q_weight,
            &k_weight,
            &v_weight,
            &o_weight,
            &gate_weight,
            &up_weight,
            &down_weight,
            &input_norm_weight,
            &post_attn_norm_weight,
        )
        .unwrap();

        lm.add_layer(&layer);
        assert_eq!(lm.num_layers(), 1);

        // Add another layer
        lm.add_layer(&layer);
        assert_eq!(lm.num_layers(), 2);
    }

    #[test]
    fn test_ernie_language_model_get_embeddings() {
        // Test getting token embeddings
        let config = test_text_config();

        let embed_weight =
            MxArray::random_uniform(&[1000, 256], 0.0, 1.0, Some(DType::Float32)).unwrap();
        let norm_weight = MxArray::ones(&[256], Some(DType::Float32)).unwrap();
        let lm_head_weight =
            MxArray::random_uniform(&[1000, 256], 0.0, 1.0, Some(DType::Float32)).unwrap();

        let lm =
            ERNIELanguageModel::new(config, &embed_weight, &norm_weight, &lm_head_weight).unwrap();

        // Create input token IDs [batch=1, seq_len=4]
        let input_ids = MxArray::from_int32(&[1, 2, 3, 4], &[1, 4]).unwrap();

        let embeddings = lm.get_embeddings(&input_ids).unwrap();
        let shape: Vec<i64> = embeddings.shape().unwrap().as_ref().to_vec();

        // Output shape should be [1, 4, 256] (batch, seq_len, hidden_size)
        assert_eq!(shape, vec![1, 4, 256]);
    }

    #[test]
    fn test_ernie_language_model_position_state() {
        // Test position state management for multimodal generation
        let config = test_text_config();

        let embed_weight =
            MxArray::random_uniform(&[1000, 256], 0.0, 1.0, Some(DType::Float32)).unwrap();
        let norm_weight = MxArray::ones(&[256], Some(DType::Float32)).unwrap();
        let lm_head_weight =
            MxArray::random_uniform(&[1000, 256], 0.0, 1.0, Some(DType::Float32)).unwrap();

        let mut lm =
            ERNIELanguageModel::new(config, &embed_weight, &norm_weight, &lm_head_weight).unwrap();

        // Initially no rope_deltas
        assert!(lm.get_rope_deltas().is_none());

        // Set position state
        let pos_ids = MxArray::zeros(&[3, 1, 4], Some(DType::Float32)).unwrap();
        lm.set_position_state(pos_ids, 42);
        assert_eq!(lm.get_rope_deltas(), Some(42));

        // Reset position state
        lm.reset_position_state();
        assert!(lm.get_rope_deltas().is_none());
    }

    #[test]
    fn test_decoder_layer_creation() {
        // Test creating PaddleOCRDecoderLayer
        let config = test_text_config();

        let q_weight =
            MxArray::random_uniform(&[256, 256], 0.0, 0.1, Some(DType::Float32)).unwrap();
        let k_weight =
            MxArray::random_uniform(&[128, 256], 0.0, 0.1, Some(DType::Float32)).unwrap();
        let v_weight =
            MxArray::random_uniform(&[128, 256], 0.0, 0.1, Some(DType::Float32)).unwrap();
        let o_weight =
            MxArray::random_uniform(&[256, 256], 0.0, 0.1, Some(DType::Float32)).unwrap();
        let gate_weight =
            MxArray::random_uniform(&[512, 256], 0.0, 0.1, Some(DType::Float32)).unwrap();
        let up_weight =
            MxArray::random_uniform(&[512, 256], 0.0, 0.1, Some(DType::Float32)).unwrap();
        let down_weight =
            MxArray::random_uniform(&[256, 512], 0.0, 0.1, Some(DType::Float32)).unwrap();
        let input_norm_weight = MxArray::ones(&[256], Some(DType::Float32)).unwrap();
        let post_attn_norm_weight = MxArray::ones(&[256], Some(DType::Float32)).unwrap();

        let layer = PaddleOCRDecoderLayer::new(
            config,
            &q_weight,
            &k_weight,
            &v_weight,
            &o_weight,
            &gate_weight,
            &up_weight,
            &down_weight,
            &input_norm_weight,
            &post_attn_norm_weight,
        );

        assert!(layer.is_ok());
    }

    #[test]
    fn test_interleaved_selection_axis_assignment() {
        // Interleaved selector correctness (PRIMARY qwen3_5-VL fix).
        //
        // rotary head_dim = 12 -> half_dim = 6; mrope_section = [3, 2, 1].
        // mlx-vlm _interleaved_position_selector([3,2,1], freq_dim=6):
        //   dim=1 (height) offset 1: idx 1, 4 (range(1, min(2*3,6)=6, 3))      -> 1
        //   dim=2 (width)  offset 2: idx 2    (range(2, min(1*3,6)=3, 3))      -> 2
        //   everything else                                                    -> 0
        //   => sel_half = [0, 1, 2, 0, 1, 0]
        // Doubled across emb=concat([f,f]): sel_doubled = sel_half ++ sel_half.
        //
        // Mark each axis row with its index (cos[axis, .., j] = axis), set sin = 0
        // and q = ones, so q_out[j] = q_rot * cos_final = cos_final = sel_doubled[j].
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
        // Hard gate / safety net: for TEXT tokens (temporal == height == width)
        // the interleaved apply MUST be bit-identical to the existing sectioned
        // apply, proving the text path cannot regress.
        //
        // Realistic qwen3_5 rotary geometry: rope_dims = 64 -> half_dim = 32,
        // mrope_section = [11, 11, 10] (sum 32). cos/sin are built from a real
        // MultimodalRoPE forward over position_ids whose three axes are EQUAL.
        let rope_dims = 64i32;
        let mrope =
            MultimodalRoPE::new(rope_dims, 4096, Some(100_000.0), vec![11, 11, 10]).unwrap();

        let seq_len = 5i64;
        // position_ids [3, 1, seq] with all three (t, h, w) rows equal.
        let mut pos_data = Vec::with_capacity(3 * seq_len as usize);
        for _ in 0..3 {
            for p in 0..seq_len {
                pos_data.push(p as f32);
            }
        }
        let pos = MxArray::from_float32(&pos_data, &[3, 1, seq_len]).unwrap();

        // x supplies only the target dtype.
        let x = MxArray::zeros(&[1, seq_len, rope_dims as i64], Some(DType::Float32)).unwrap();
        let (cos, sin) = mrope.forward(&x, &pos).unwrap();

        // Q/K head_dim = 96 > rope_dims = 64 to exercise the partial-rotary
        // pass-through path that qwen3_5 uses.
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

        let (q_sec, k_sec) =
            apply_multimodal_rotary_pos_emb(&q, &k, &cos, &sin, vec![11, 11, 10]).unwrap();
        let (q_int, k_int) =
            apply_multimodal_rotary_pos_emb_interleaved(&q, &k, &cos, &sin, vec![11, 11, 10])
                .unwrap();

        assert_eq!(
            q_sec.to_float32().unwrap().as_ref(),
            q_int.to_float32().unwrap().as_ref(),
            "interleaved Q must equal sectioned Q for text tokens (t==h==w)"
        );
        assert_eq!(
            k_sec.to_float32().unwrap().as_ref(),
            k_int.to_float32().unwrap().as_ref(),
            "interleaved K must equal sectioned K for text tokens (t==h==w)"
        );
    }

    #[test]
    fn test_precomputed_cos_sin_reused_across_layers_matches_per_layer_recompute() {
        // Regression for the M-RoPE precompute-once optimization: Qwen3.5-VL's
        // paged prefill loop computes `select_interleaved_cos_sin` ONCE per
        // forward pass via `Qwen3_5Attention::forward_paged`'s `mrope_cache`
        // parameter and reuses the result across every full-attention layer,
        // instead of recomputing the cos/sin table + axis-selector gather per
        // layer (the pre-fix behaviour, still reachable through
        // `apply_multimodal_rotary_pos_emb_interleaved`). This proves the
        // precomputed pair is layer-invariant: applying it to TWO INDEPENDENT
        // (q, k) pairs (simulating two different full-attention layers
        // sharing one `position_ids`) must produce bit-identical results to
        // calling the combined, per-layer-recompute wrapper independently for
        // each layer.
        let rope_dims = 64i32;
        let mrope_section = vec![11i32, 11, 10];
        let mrope =
            MultimodalRoPE::new(rope_dims, 4096, Some(100_000.0), mrope_section.clone()).unwrap();

        let seq_len = 5i64;
        // Distinct per-axis positions (t != h != w) so this exercises genuine
        // image-style M-RoPE, not just the text-invariant (t==h==w) case.
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

        // Pre-fix behaviour: every "layer" independently recomputes the full
        // selector + gather from `cos`/`sin`.
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

        // Post-fix behaviour: compute the selected cos/sin ONCE, reuse for
        // both "layers" (mirrors `Qwen3_5Attention::forward_paged`'s
        // `mrope_cache` reuse).
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
    fn test_attention_creation() {
        // Test creating PaddleOCRAttention
        let config = test_text_config();

        let q_weight =
            MxArray::random_uniform(&[256, 256], 0.0, 0.1, Some(DType::Float32)).unwrap();
        let k_weight =
            MxArray::random_uniform(&[128, 256], 0.0, 0.1, Some(DType::Float32)).unwrap();
        let v_weight =
            MxArray::random_uniform(&[128, 256], 0.0, 0.1, Some(DType::Float32)).unwrap();
        let o_weight =
            MxArray::random_uniform(&[256, 256], 0.0, 0.1, Some(DType::Float32)).unwrap();

        let attn = PaddleOCRAttention::new(config, &q_weight, &k_weight, &v_weight, &o_weight);

        assert!(attn.is_ok());
    }
}
