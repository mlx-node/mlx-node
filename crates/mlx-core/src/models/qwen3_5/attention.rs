use std::sync::OnceLock;
use std::time::Instant;

use crate::array::MxArray;
use crate::array::attention::{scaled_dot_product_attention, scaled_dot_product_attention_causal};
use crate::array::mask::create_causal_mask;
use crate::inference_trace::{
    elapsed_ms, enabled as inference_trace_enabled, write as write_inference_trace,
};
use crate::inspector::InspectorRecorder;
use crate::models::paddleocr_vl::language::{MultimodalRoPE, apply_multimodal_rotary_pos_emb};
use crate::nn::{Activations, Linear, RMSNorm, RoPE};
use crate::transformer::KVCache;
#[cfg(feature = "full")]
use crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter;
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

fn paged_prefill_paged_attention_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        crate::inference_trace::env_flag_enabled_or_default(
            "MLX_PAGED_PREFILL_PAGED_ATTENTION",
            true,
        )
    })
}

fn native_kv_write_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("MLX_QWEN35_NATIVE_KV_WRITE")
            .or_else(|_| std::env::var("MLX_NATIVE_KV_WRITE"))
            .map(|value| crate::inference_trace::env_flag_value_enabled(&value))
            .unwrap_or(true)
    })
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
            let (q_out, k_out) = apply_multimodal_rotary_pos_emb(
                &q_t,
                &k_t,
                &cos,
                &sin,
                mrope.mrope_section_arr().to_vec(),
            )?;
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

        // Scaled dot-product attention using fast kernel.
        // When no explicit mask is provided:
        //   - seq_len > 1 (prefill): use "causal" mode; the fused kernel
        //     handles masking internally without materializing an O(N^2) mask.
        //   - seq_len == 1 (decode): no mask needed.
        // Explicit masks are still honored for sliding-window style paths.
        let output = if let Some(m) = _mask {
            scaled_dot_product_attention(&queries, &keys, &values, self.scale as f64, Some(m))?
        } else if seq_len > 1 {
            scaled_dot_product_attention_causal(&queries, &keys, &values, self.scale as f64)?
        } else {
            scaled_dot_product_attention(&queries, &keys, &values, self.scale as f64, None)?
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

    /// Inspector-aware forward.
    ///
    /// Behaviourally identical to [`Self::forward`] when `recorder` is
    /// `None` or when `recorder.should_capture_layer(layer_index)` is
    /// `false`. When capture is on, the post-RoPE / post-QK-norm Q and K
    /// tensors are materialized and passed to
    /// [`InspectorRecorder::capture_attention`] *after* the cache update
    /// (so K reflects the full attended context) but *before* the SDPA
    /// matmul with V — matching the contract on
    /// `inspector-types.ts::AttentionLayer.scores`.
    ///
    /// This is a separate method (rather than an extra arg on `forward`)
    /// to keep the chat hot path's call graph byte-identical. The
    /// inspector recompute is a side branch — the fused SDPA kernel
    /// still produces the actual layer output, so generation results
    /// stay bitwise-identical with or without the recorder.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_inspector(
        &self,
        x: &MxArray,
        mask: Option<&MxArray>,
        cache: Option<&mut KVCache>,
        position_ids: Option<&MxArray>,
        _rope_offset_delta: i32,
        recorder: Option<&mut InspectorRecorder>,
        layer_index: i32,
    ) -> Result<MxArray> {
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

        // Inspector runs in text-only mode — VLM / M-RoPE captures are
        // not in scope for this slice, so we mirror the standard scalar-
        // offset RoPE branch only.
        let (queries, keys) = if let (Some(pos_ids), Some(mrope)) = (position_ids, &self.mrope) {
            // Match the regular forward's M-RoPE branch for parity, even
            // though the inspector entry point won't drive it for the
            // current vertical slice.
            let (cos, sin) = mrope.forward(&queries, pos_ids)?;
            let q_t = queries.transpose(Some(&[0, 2, 1, 3]))?;
            let k_t = keys.transpose(Some(&[0, 2, 1, 3]))?;
            let (q_out, k_out) = apply_multimodal_rotary_pos_emb(
                &q_t,
                &k_t,
                &cos,
                &sin,
                mrope.mrope_section_arr().to_vec(),
            )?;
            let q_out = q_out.transpose(Some(&[0, 2, 1, 3]))?;
            let k_out = k_out.transpose(Some(&[0, 2, 1, 3]))?;
            (q_out, k_out)
        } else {
            let offset = cache.as_ref().map_or(0, |c| c.get_offset());
            let queries = self.rope.forward(&queries, Some(offset))?;
            let keys = self.rope.forward(&keys, Some(offset))?;
            (queries, keys)
        };

        let queries = queries.transpose(Some(&[0, 2, 1, 3]))?;
        let keys = keys.transpose(Some(&[0, 2, 1, 3]))?;
        let values = values.transpose(Some(&[0, 2, 1, 3]))?;

        // Capture the cache offset *before* update_and_fetch so the causal
        // mask can be built against the pre-update KV length when decoding
        // on top of a prefill (offset > 0 means the cache already has
        // history; the new query attends to all of it).
        let pre_update_offset = cache.as_ref().map_or(0, |c| c.get_offset());

        let (keys, values) = if let Some(c) = cache {
            c.update_and_fetch(&keys, &values)?
        } else {
            (keys, values)
        };

        // Side-branch capture. Runs only when the recorder asked for this
        // layer — otherwise no extra GPU work.
        if let Some(rec) = recorder {
            if rec.should_capture_layer(layer_index) {
                // Force Q and the (possibly cache-extended) K materialized
                // before the side branch reads them. Without this the
                // recorder's `.to_float32_vec()` would block on an
                // unrelated lazy graph.
                queries.eval();
                keys.eval();
                rec.capture_attention(
                    layer_index,
                    &queries,
                    &keys,
                    self.scale,
                    self.num_heads,
                    self.num_kv_heads,
                    pre_update_offset,
                )?;
            }
        }

        let output = if let Some(m) = mask {
            scaled_dot_product_attention(&queries, &keys, &values, self.scale as f64, Some(m))?
        } else if seq_len > 1 {
            scaled_dot_product_attention_causal(&queries, &keys, &values, self.scale as f64)?
        } else {
            scaled_dot_product_attention(&queries, &keys, &values, self.scale as f64, None)?
        };

        let output = output.transpose(Some(&[0, 2, 1, 3]))?;
        let output = output.reshape(&[batch, seq_len, (self.num_heads * self.head_dim) as i64])?;

        let gate_sigmoid = Activations::sigmoid(&gate)?;
        let gated_output = output.mul(&gate_sigmoid)?;

        self.o_proj.forward(&gated_output)
    }

    /// Forward pass routed through the block-paged KV adapter.
    ///
    /// This mirrors the flat full-attention path, but writes K/V into the
    /// paged pool and reads attention K/V back through the adapter. The path is
    /// text-only: image-bearing turns are rejected before dispatching here.
    #[cfg(feature = "full")]
    #[allow(clippy::too_many_arguments)]
    pub fn forward_paged(
        &self,
        x: &MxArray,
        adapter: &mut PagedKVCacheAdapter,
        attn_layer_idx: u32,
        first_logical_position: u32,
        cached_prefix_len: u32,
        is_prefill: bool,
    ) -> Result<MxArray> {
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

        let rope_offset = first_logical_position as i32;
        let queries = self.rope.forward(&queries, Some(rope_offset))?;
        let keys = self.rope.forward(&keys, Some(rope_offset))?;

        let queries_bhtd = queries.transpose(Some(&[0, 2, 1, 3]))?;
        let keys_bhtd = keys.transpose(Some(&[0, 2, 1, 3]))?;
        let values_bhtd = values.transpose(Some(&[0, 2, 1, 3]))?;

        let keys_paged = keys_bhtd.transpose(Some(&[0, 2, 1, 3]))?.reshape(&[
            batch * seq_len,
            self.num_kv_heads as i64,
            self.head_dim as i64,
        ])?;
        let values_paged = values_bhtd.transpose(Some(&[0, 2, 1, 3]))?.reshape(&[
            batch * seq_len,
            self.num_kv_heads as i64,
            self.head_dim as i64,
        ])?;

        let trace_enabled = inference_trace_enabled();
        let write_trace_start = trace_enabled.then(Instant::now);
        let write_path = if native_kv_write_enabled() {
            match adapter.update_keys_values_native(
                attn_layer_idx,
                &keys_paged,
                &values_paged,
                first_logical_position,
            ) {
                Ok(()) => "native",
                Err(err) => {
                    if trace_enabled {
                        write_inference_trace(format_args!(
                            "[MLX_TRACE] qwen3.5-attn paged_kv_write_fallback \
                             layer={} first_position={} seq_len={} error={}",
                            attn_layer_idx, first_logical_position, seq_len, err
                        ));
                    }
                    adapter
                        .update_keys_values(
                            attn_layer_idx,
                            &keys_paged,
                            &values_paged,
                            first_logical_position,
                        )
                        .map_err(napi::Error::from_reason)?;
                    "legacy"
                }
            }
        } else {
            adapter
                .update_keys_values(
                    attn_layer_idx,
                    &keys_paged,
                    &values_paged,
                    first_logical_position,
                )
                .map_err(napi::Error::from_reason)?;
            "legacy"
        };
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] qwen3.5-attn paged_kv_write_done \
                 layer={} first_position={} seq_len={} path={} elapsed_ms={:.1}",
                attn_layer_idx,
                first_logical_position,
                seq_len,
                write_path,
                write_trace_start.map(elapsed_ms).unwrap_or(0.0)
            ));
        }

        let attn_bhtd = if is_prefill {
            if cached_prefix_len == 0 {
                if seq_len > 1 {
                    scaled_dot_product_attention_causal(
                        &queries_bhtd,
                        &keys_bhtd,
                        &values_bhtd,
                        self.scale as f64,
                    )?
                } else {
                    scaled_dot_product_attention(
                        &queries_bhtd,
                        &keys_bhtd,
                        &values_bhtd,
                        self.scale as f64,
                        None,
                    )?
                }
            } else {
                let total_ctx = cached_prefix_len + (seq_len as u32);
                let maybe_paged_attn = if batch == 1 && paged_prefill_paged_attention_enabled() {
                    let paged_trace_start = trace_enabled.then(Instant::now);
                    let queries_paged =
                        queries.reshape(&[seq_len, self.num_heads as i64, self.head_dim as i64])?;
                    match adapter.gather_kv_for_prefill_chunk(
                        attn_layer_idx,
                        &queries_paged,
                        cached_prefix_len,
                        self.scale,
                    ) {
                        Ok(attn_t_h_d) => {
                            let target_dtype = x.dtype()?;
                            let attn_t_h_d = attn_t_h_d.astype(target_dtype)?;
                            let attn = attn_t_h_d.reshape(&[
                                batch,
                                seq_len,
                                self.num_heads as i64,
                                self.head_dim as i64,
                            ])?;
                            let attn = attn.transpose(Some(&[0, 2, 1, 3]))?;
                            if trace_enabled {
                                write_inference_trace(format_args!(
                                    "[MLX_TRACE] qwen3.5-attn cache_hit_prefill \
                                     layer={} suffix_tokens={} cached_prefix_tokens={} total_ctx={} \
                                     path=paged_attention bridge_ms={:.1} read_kv_range_ms=0.0 \
                                     mask_ms=0.0 sdpa_mode=none sdpa_graph_ms=0.0",
                                    attn_layer_idx,
                                    seq_len,
                                    cached_prefix_len,
                                    total_ctx,
                                    paged_trace_start.map(elapsed_ms).unwrap_or(0.0)
                                ));
                            }
                            Some(attn)
                        }
                        Err(err) => {
                            if trace_enabled {
                                write_inference_trace(format_args!(
                                    "[MLX_TRACE] qwen3.5-attn cache_hit_prefill_paged_fallback \
                                     layer={} suffix_tokens={} cached_prefix_tokens={} total_ctx={} \
                                     error={}",
                                    attn_layer_idx, seq_len, cached_prefix_len, total_ctx, err
                                ));
                            }
                            None
                        }
                    }
                } else {
                    None
                };

                match maybe_paged_attn {
                    Some(attn) => attn,
                    None => {
                        let read_trace_start = trace_enabled.then(Instant::now);
                        let (k_full, v_full) = adapter
                            .read_kv_range(attn_layer_idx, 0, total_ctx)
                            .map_err(napi::Error::from_reason)?;
                        let read_kv_range_ms = read_trace_start.map(elapsed_ms);
                        let mask_trace_start = trace_enabled.then(Instant::now);
                        let mask = create_causal_mask(
                            seq_len as i32,
                            Some(cached_prefix_len as i32),
                            None,
                        )?;
                        let mask_ms = mask_trace_start.map(elapsed_ms);
                        let sdpa_trace_start = trace_enabled.then(Instant::now);
                        let attn = scaled_dot_product_attention(
                            &queries_bhtd,
                            &k_full,
                            &v_full,
                            self.scale as f64,
                            Some(&mask),
                        )?;
                        if trace_enabled {
                            write_inference_trace(format_args!(
                                "[MLX_TRACE] qwen3.5-attn cache_hit_prefill \
                                 layer={} suffix_tokens={} cached_prefix_tokens={} total_ctx={} \
                                 path=read_kv_range read_kv_range_ms={:.1} mask_ms={:.1} \
                                 sdpa_mode=explicit_mask sdpa_graph_ms={:.1}",
                                attn_layer_idx,
                                seq_len,
                                cached_prefix_len,
                                total_ctx,
                                read_kv_range_ms.unwrap_or(0.0),
                                mask_ms.unwrap_or(0.0),
                                sdpa_trace_start.map(elapsed_ms).unwrap_or(0.0)
                            ));
                        }
                        attn
                    }
                }
            }
        } else {
            let queries_3d = queries_bhtd.squeeze(Some(&[2]))?.reshape(&[
                1,
                self.num_heads as i64,
                self.head_dim as i64,
            ])?;
            let gather_trace_start = trace_enabled.then(Instant::now);
            let attn_3d = match adapter.gather_kv_for_decode_graph(
                attn_layer_idx,
                &queries_3d,
                self.scale,
                1.0,
            ) {
                Ok(attn_3d) => {
                    if trace_enabled {
                        write_inference_trace(format_args!(
                            "[MLX_TRACE] qwen3.5-attn decode_gather_done \
                             layer={} path=graph total_ctx={} elapsed_ms={:.1}",
                            attn_layer_idx,
                            adapter.current_token_count(),
                            gather_trace_start.map(elapsed_ms).unwrap_or(0.0)
                        ));
                    }
                    attn_3d
                }
                Err(err) => {
                    if trace_enabled {
                        write_inference_trace(format_args!(
                            "[MLX_TRACE] qwen3.5-attn decode_gather_fallback \
                             layer={} path=raw total_ctx={} error={}",
                            attn_layer_idx,
                            adapter.current_token_count(),
                            err
                        ));
                    }
                    adapter
                        .gather_kv_for_decode(attn_layer_idx, &queries_3d, self.scale, 1.0)
                        .map_err(napi::Error::from_reason)?
                }
            };
            let target_dtype = x.dtype()?;
            let attn_3d = attn_3d.astype(target_dtype)?;
            attn_3d.reshape(&[1, self.num_heads as i64, 1, self.head_dim as i64])?
        };

        let output = attn_bhtd.transpose(Some(&[0, 2, 1, 3]))?;
        let output = output.reshape(&[batch, seq_len, (self.num_heads * self.head_dim) as i64])?;

        let gate_sigmoid = Activations::sigmoid(&gate)?;
        let gated_output = output.mul(&gate_sigmoid)?;

        self.o_proj.forward(&gated_output)
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
            let section = mrope.mrope_section_arr();
            let (q_out, k_out) = apply_qwen35_mrope(&q_t, &k_t, &cos, &sin, section)?;
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
