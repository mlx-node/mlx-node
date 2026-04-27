use crate::array::MxArray;
use crate::array::attention::scaled_dot_product_attention_causal;
use crate::models::paddleocr_vl::language::{MultimodalRoPE, apply_multimodal_rotary_pos_emb};
use crate::nn::{Activations, Linear, RMSNorm, RoPE};
use crate::transformer::KVCache;
use napi::bindgen_prelude::*;

use super::config::Qwen3_5Config;
use super::quantized_linear::{LinearProj, QuantizedLinear};

#[cfg(target_family = "wasm")]
fn apply_qwen35_mrope(
    q: &MxArray,
    k: &MxArray,
    cos: &MxArray,
    sin: &MxArray,
    mrope_section: &[i32],
) -> Result<(MxArray, MxArray)> {
    let q_shape = q.shape()?;
    let batch = q_shape[0];
    let q_heads = q_shape[1];
    let seq_len = q_shape[2];
    let q_dim = q_shape[3];
    let k_shape = k.shape()?;
    let k_heads = k_shape[1];
    let k_dim = k_shape[3];
    let half_rotary: i64 = mrope_section.iter().map(|&s| s as i64).sum();
    let rope_dims = half_rotary * 2;

    let build_trig = |src: &MxArray| -> Result<MxArray> {
        let axes = [
            src.slice_axis(0, 0, 1)?.squeeze(Some(&[0]))?,
            src.slice_axis(0, 1, 2)?.squeeze(Some(&[0]))?,
            src.slice_axis(0, 2, 3)?.squeeze(Some(&[0]))?,
        ];

        let h_limit = mrope_section[1] as i64 * 3;
        let w_limit = mrope_section[2] as i64 * 3;
        let mut half_parts: Vec<MxArray> = Vec::with_capacity(half_rotary as usize);
        for dim in 0..half_rotary {
            let axis = if dim % 3 == 1 && dim < h_limit {
                1
            } else if dim % 3 == 2 && dim < w_limit {
                2
            } else {
                0
            };
            half_parts.push(axes[axis].slice_axis(2, dim, dim + 1)?);
        }

        let refs: Vec<&MxArray> = half_parts.iter().collect();
        let half = MxArray::concatenate_many(refs, Some(-1))?;
        MxArray::concatenate_many(vec![&half, &half], Some(-1))?
            .reshape(&[batch, 1, seq_len, rope_dims])
    };

    let cos_final = build_trig(cos)?;
    let sin_final = build_trig(sin)?;

    let rotate = |x: &MxArray, _heads: i64, dim: i64| -> Result<MxArray> {
        let x_rot = x.slice_axis(3, 0, rope_dims)?;
        let x1 = x_rot.slice_axis(3, 0, half_rotary)?;
        let x2 = x_rot.slice_axis(3, half_rotary, rope_dims)?;
        let neg_x2 = x2.mul_scalar(-1.0)?;
        let rotated = MxArray::concatenate_many(vec![&neg_x2, &x1], Some(-1))?;
        let embedded = x_rot.mul(&cos_final)?.add(&rotated.mul(&sin_final)?)?;
        if rope_dims < dim {
            let pass = x.slice_axis(3, rope_dims, dim)?;
            MxArray::concatenate_many(vec![&embedded, &pass], Some(-1))
        } else {
            Ok(embedded)
        }
    };

    Ok((rotate(q, q_heads, q_dim)?, rotate(k, k_heads, k_dim)?))
}

/// Qwen3.5 full attention with gating and partial RoPE.
///
/// Key differences from standard Qwen3 attention:
/// 1. q_proj outputs 2x width → split into queries + gate
/// 2. Partial RoPE: only rotates `head_dim * partial_rotary_factor` dimensions
/// 3. Output is gated: `o_proj(sdpa_output * sigmoid(gate))`
pub struct Qwen3_5Attention {
    q_proj: LinearProj, // hidden → num_heads * head_dim * 2 (queries + gate)
    k_proj: LinearProj, // hidden → num_kv_heads * head_dim
    v_proj: LinearProj, // hidden → num_kv_heads * head_dim
    o_proj: LinearProj, // num_heads * head_dim → hidden

    q_norm: RMSNorm, // [head_dim]
    k_norm: RMSNorm, // [head_dim]

    rope: RoPE,
    /// Optional M-RoPE for VLM mode (3D position encoding: temporal, height, width)
    mrope: Option<MultimodalRoPE>,

    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl Qwen3_5Attention {
    pub fn new(config: &Qwen3_5Config) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let has_bias = config.attention_bias;

        // q_proj outputs 2x for gating: queries + gate
        let q_proj = Linear::new(
            hidden_size as u32,
            (num_heads * head_dim * 2) as u32,
            Some(has_bias),
        )?;
        let k_proj = Linear::new(
            hidden_size as u32,
            (num_kv_heads * head_dim) as u32,
            Some(has_bias),
        )?;
        let v_proj = Linear::new(
            hidden_size as u32,
            (num_kv_heads * head_dim) as u32,
            Some(has_bias),
        )?;
        let o_proj = Linear::new(
            (num_heads * head_dim) as u32,
            hidden_size as u32,
            Some(has_bias),
        )?;

        let q_norm = RMSNorm::new(head_dim as u32, Some(config.rms_norm_eps))?;
        let k_norm = RMSNorm::new(head_dim as u32, Some(config.rms_norm_eps))?;

        // Partial RoPE: only rotate a fraction of dimensions
        let rope_dims = config.rope_dims();
        let rope = RoPE::new(rope_dims, Some(false), Some(config.rope_theta), None);

        let scale = (head_dim as f32).powf(-0.5);

        Ok(Self {
            q_proj: LinearProj::Standard(q_proj),
            k_proj: LinearProj::Standard(k_proj),
            v_proj: LinearProj::Standard(v_proj),
            o_proj: LinearProj::Standard(o_proj),
            q_norm,
            k_norm,
            rope,
            mrope: None,
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
    /// * `mask` - Attention mask (causal)
    /// * `cache` - Optional KVCache for incremental generation
    /// * `position_ids` - Optional [3, B, T] M-RoPE positions for VLM mode.
    ///   When None, uses scalar offset from KVCache (standard text-only behavior).
    ///
    /// # Returns
    /// Output [B, T, hidden_size]
    #[cfg_attr(not(target_family = "wasm"), allow(unused_variables))]
    pub fn forward(
        &self,
        x: &MxArray,
        _mask: Option<&MxArray>,
        cache: Option<&mut KVCache>,
        position_ids: Option<&MxArray>,
        rope_offset_delta: i32,
    ) -> Result<MxArray> {
        let batch = x.shape_at(0)?;
        let seq_len = x.shape_at(1)?;

        // Project queries (2x width for gating)
        let q_proj_output = self.q_proj.forward(x)?;

        // Split into queries and gate PER-HEAD (not flat):
        //   reshape to [B, T, num_heads, head_dim*2]
        //   split on last axis → queries [B,T,H,D] and gate [B,T,H,D]
        let q_per_head = q_proj_output.reshape(&[
            batch,
            seq_len,
            self.num_heads as i64,
            (self.head_dim * 2) as i64,
        ])?;
        let queries = q_per_head.slice_axis(3, 0, self.head_dim as i64)?;
        let gate = q_per_head.slice_axis(3, self.head_dim as i64, (self.head_dim * 2) as i64)?;
        // Flatten gate for later: [B, T, H, D] → [B, T, H*D]
        let gate = gate.reshape(&[batch, seq_len, (self.num_heads * self.head_dim) as i64])?;

        // Project keys and values
        let keys = self.k_proj.forward(x)?;
        let values = self.v_proj.forward(x)?;

        // Reshape to head format: [B, T, H, D]
        // queries already in [B, T, H, D] from per-head split above
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

        // Apply QK normalization (operates on last dim)
        let queries = self.q_norm.forward(&queries)?;
        let keys = self.k_norm.forward(&keys)?;

        // Apply RoPE: either M-RoPE (VLM) or standard scalar offset (text-only)
        let (queries, keys) = if let (Some(pos_ids), Some(mrope)) = (position_ids, &self.mrope) {
            // M-RoPE: compute cos/sin from 3D position IDs [3, B, T]
            let (cos, sin) = mrope.forward(&queries, pos_ids)?;
            // Transpose to [B, H, T, D] for apply_multimodal_rotary_pos_emb
            let q_t = queries.transpose(Some(&[0, 2, 1, 3]))?;
            let k_t = keys.transpose(Some(&[0, 2, 1, 3]))?;
            let (q_out, k_out) =
                apply_multimodal_rotary_pos_emb(&q_t, &k_t, &cos, &sin, mrope.mrope_section())?;
            // Transpose back to [B, T, H, D]
            let q_out = q_out.transpose(Some(&[0, 2, 1, 3]))?;
            let k_out = k_out.transpose(Some(&[0, 2, 1, 3]))?;
            (q_out, k_out)
        } else {
            // Standard scalar offset RoPE (text-only path, existing behavior)
            let base_offset = cache.as_ref().map_or(0, |c| c.get_offset());
            #[cfg(target_family = "wasm")]
            let offset = base_offset + rope_offset_delta;
            #[cfg(not(target_family = "wasm"))]
            let offset = base_offset;
            let queries = self.rope.forward(&queries, Some(offset))?;
            let keys = self.rope.forward(&keys, Some(offset))?;
            (queries, keys)
        };

        // Transpose to [B, H, T, D] for KVCache and SDPA
        let queries = queries.transpose(Some(&[0, 2, 1, 3]))?;
        let keys = keys.transpose(Some(&[0, 2, 1, 3]))?;
        let values = values.transpose(Some(&[0, 2, 1, 3]))?;

        // Update KV cache (expects [B, H, T, D])
        let (keys, values) = if let Some(c) = cache {
            c.update_and_fetch(&keys, &values)?
        } else {
            (keys, values)
        };

        // Fused scaled dot-product attention with causal masking.
        // The kernel handles GQA internally (no need to expand K/V heads).
        let output =
            scaled_dot_product_attention_causal(&queries, &keys, &values, self.scale as f64)?;

        // Transpose back: [B, H, T, D] → [B, T, H, D] → flatten to [B, T, H*D]
        let output = output.transpose(Some(&[0, 2, 1, 3]))?;
        let output = output.reshape(&[batch, seq_len, (self.num_heads * self.head_dim) as i64])?;

        // Apply gate: output * sigmoid(gate)
        // gate is already [B, T, H*D] from the per-head split above
        let gate_sigmoid = Activations::sigmoid(&gate)?;
        let gated_output = output.mul(&gate_sigmoid)?;

        // Output projection
        let result = self.o_proj.forward(&gated_output)?;
        Ok(result)
    }

    /// VLM prefill variant that returns K/V tensors instead of mutating KVCache.
    ///
    /// The browser WebGPU backend has a fragile path for large in-place
    /// slice-update graphs. VLM prefill can avoid that entirely: prefill starts
    /// from an empty cache, so the full K/V tensors are already the cache state.
    #[cfg(target_family = "wasm")]
    pub fn forward_vlm_prefill(
        &self,
        x: &MxArray,
        _mask: Option<&MxArray>,
        position_ids: &MxArray,
    ) -> Result<(MxArray, MxArray, MxArray)> {
        let batch = x.shape_at(0)?;
        let seq_len = x.shape_at(1)?;

        let q_proj_output = self.q_proj.forward(x)?;
        let q_per_head = q_proj_output.reshape(&[
            batch,
            seq_len,
            self.num_heads as i64,
            (self.head_dim * 2) as i64,
        ])?;
        let queries = q_per_head.slice_axis(3, 0, self.head_dim as i64)?;
        let gate = q_per_head.slice_axis(3, self.head_dim as i64, (self.head_dim * 2) as i64)?;
        let gate = gate.reshape(&[batch, seq_len, (self.num_heads * self.head_dim) as i64])?;

        let keys = self.k_proj.forward(x)?;
        let values = self.v_proj.forward(x)?;
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

        let queries = self.q_norm.forward(&queries)?;
        let keys = self.k_norm.forward(&keys)?;

        let (queries, keys) = if let Some(mrope) = &self.mrope {
            let (cos, sin) = mrope.forward(&queries, position_ids)?;
            let q_t = queries.transpose(Some(&[0, 2, 1, 3]))?;
            let k_t = keys.transpose(Some(&[0, 2, 1, 3]))?;
            let section = mrope.mrope_section();
            let (q_out, k_out) = apply_qwen35_mrope(&q_t, &k_t, &cos, &sin, &section)?;
            let q_out = q_out.transpose(Some(&[0, 2, 1, 3]))?;
            let k_out = k_out.transpose(Some(&[0, 2, 1, 3]))?;
            (q_out, k_out)
        } else {
            let queries = self.rope.forward(&queries, Some(0))?;
            let keys = self.rope.forward(&keys, Some(0))?;
            (queries, keys)
        };

        let queries = queries.transpose(Some(&[0, 2, 1, 3]))?;
        let keys = keys.transpose(Some(&[0, 2, 1, 3]))?;
        let values = values.transpose(Some(&[0, 2, 1, 3]))?;

        let output =
            scaled_dot_product_attention_causal(&queries, &keys, &values, self.scale as f64)?;
        let output = output.transpose(Some(&[0, 2, 1, 3]))?;
        let output = output.reshape(&[batch, seq_len, (self.num_heads * self.head_dim) as i64])?;

        let gate_sigmoid = Activations::sigmoid(&gate)?;
        let gated_output = output.mul(&gate_sigmoid)?;
        let result = self.o_proj.forward(&gated_output)?;
        Ok((result, keys, values))
    }

    /// Initialize M-RoPE for VLM mode.
    pub fn init_mrope(
        &mut self,
        mrope_section: Vec<i32>,
        rope_theta: f64,
        max_position_embeddings: i32,
        rope_dims: i32,
    ) -> Result<()> {
        // Use rope_dims (head_dim * partial_rotary_factor), not full head_dim
        self.mrope = Some(MultimodalRoPE::new(
            rope_dims,
            max_position_embeddings,
            Some(rope_theta),
            mrope_section,
        )?);
        Ok(())
    }

    // ========== Weight accessors (standard mode) ==========

    pub fn set_q_proj_weight(&mut self, w: &MxArray) -> Result<()> {
        self.q_proj.set_weight(w, "q_proj")
    }
    pub fn set_k_proj_weight(&mut self, w: &MxArray) -> Result<()> {
        self.k_proj.set_weight(w, "k_proj")
    }
    pub fn set_v_proj_weight(&mut self, w: &MxArray) -> Result<()> {
        self.v_proj.set_weight(w, "v_proj")
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
        self.v_proj.set_bias(b, "v_proj")
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
        self.v_proj.set_quantized(ql);
    }
    pub fn set_quantized_o_proj(&mut self, ql: QuantizedLinear) {
        self.o_proj.set_quantized(ql);
    }

    // ========== Weight getters (for training parameter extraction) ==========

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
    pub fn get_q_norm_weight(&self) -> MxArray {
        self.q_norm.get_weight()
    }
    pub fn get_k_norm_weight(&self) -> MxArray {
        self.k_norm.get_weight()
    }
}
