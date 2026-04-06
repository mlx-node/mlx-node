use crate::array::MxArray;
use crate::array::attention::{scaled_dot_product_attention, scaled_dot_product_attention_causal};
use crate::models::paddleocr_vl::language::{MultimodalRoPE, apply_multimodal_rotary_pos_emb};
use crate::nn::{Activations, Linear, RMSNorm, RoPE};
use crate::transformer::KVCache;
use napi::bindgen_prelude::*;

use super::config::Qwen3_5Config;
use super::quantized_linear::{LinearProj, QuantizedLinear};

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
    pub fn forward(
        &self,
        x: &MxArray,
        mask: Option<&MxArray>,
        cache: Option<&mut KVCache>,
        position_ids: Option<&MxArray>,
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
            let offset = cache.as_ref().map_or(0, |c| c.get_offset());
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

        // Scaled dot-product attention.
        // When no explicit mask is provided:
        //   - seq_len > 1 (prefill): use "causal" mode — MLX's fused Metal kernel handles
        //     causal masking internally without materializing an O(N²) mask array.
        //     This matches Python mlx-lm's `create_attention_mask` returning "causal".
        //   - seq_len == 1 (decode): no mask needed (single token only attends to past).
        // When an explicit mask is provided (e.g., sliding window): use it directly.
        // Manual SDPA implementation to bypass the fast::sdpa fallback
        // for debugging WebGPU divergence
        let output = {
            use crate::array::MxArray;
            // Cast scale to bf16 to maintain dtype throughout
            let scale_scalar = MxArray::scalar_float(self.scale as f64)?
                .astype(queries.dtype()?)?;
            let q_scaled = queries.mul(&scale_scalar)?;

            // GQA: expand q heads to match kv heads
            let n_rep = self.num_heads / self.num_kv_heads;
            // For GQA: repeat K/V heads to match Q heads count
            // This avoids 5D broadcast matmul which WebGPU doesn't handle correctly
            let (q_for_attn, k_for_attn, v_for_attn) = if n_rep > 1 {
                // Repeat K: [B, Hkv, T, D] → [B, Hkv, 1, T, D] → tile → [B, Hkv, n_rep, T, D] → reshape [B, Hq, T, D]
                let kv_len = keys.shape_at(2)?;
                let k_expanded = keys.reshape(&[
                    batch, self.num_kv_heads as i64, 1, kv_len, self.head_dim as i64
                ])?;
                let k_tiled = k_expanded.tile(&[1, 1, n_rep as i32, 1, 1])?;
                let k_flat = k_tiled.reshape(&[
                    batch, self.num_heads as i64, kv_len, self.head_dim as i64
                ])?;

                let v_expanded = values.reshape(&[
                    batch, self.num_kv_heads as i64, 1, kv_len, self.head_dim as i64
                ])?;
                let v_tiled = v_expanded.tile(&[1, 1, n_rep as i32, 1, 1])?;
                let v_flat = v_tiled.reshape(&[
                    batch, self.num_heads as i64, kv_len, self.head_dim as i64
                ])?;

                (q_scaled, k_flat, v_flat)
            } else {
                (q_scaled, keys.clone(), values.clone())
            };

            // scores = q @ k^T  (swap last 2 axes of k)
            let k_ndim = k_for_attn.ndim()? as i64;
            let mut perm: Vec<i32> = (0..k_ndim as i32).collect();
            let n = perm.len();
            perm.swap(n - 1, n - 2);
            let k_t = k_for_attn.transpose(Some(&perm))?;
            let mut scores = q_for_attn.matmul(&k_t)?;

            // Causal mask for prefill
            if seq_len > 1 {
                let kv_len = scores.shape_at(scores.ndim()? - 1)?;
                let q_len = scores.shape_at(scores.ndim()? - 2)?;
                let offset_val = kv_len - q_len;
                let q_idx = MxArray::arange(offset_val as f64, (q_len + offset_val) as f64, None, None)?;
                let k_idx = MxArray::arange(0.0, kv_len as f64, None, None)?;
                let q_idx_2d = q_idx.reshape(&[q_len, 1])?;
                let k_idx_2d = k_idx.reshape(&[1, kv_len])?;
                let mask_bool = q_idx_2d.greater_equal(&k_idx_2d)?;
                let neg_inf = MxArray::scalar_float(f64::NEG_INFINITY)?
                    .astype(scores.dtype()?)?;
                scores = mask_bool.where_(&scores, &neg_inf)?;
            }

            // softmax
            let attn_weights = crate::nn::Activations::softmax(&scores, Some(-1))?;

            // output = weights @ v
            let mut attn_out = attn_weights.matmul(&v_for_attn)?;

            // Already [B, Hq, T, D] — no reshape needed
            attn_out
        };


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
