use std::sync::OnceLock;
use std::time::Instant;

use crate::array::MxArray;
use crate::array::attention::{scaled_dot_product_attention, scaled_dot_product_attention_causal};
use crate::array::mask::create_causal_mask;
use crate::inference_trace::{
    elapsed_ms, enabled as inference_trace_enabled, write as write_inference_trace,
};
use crate::models::paddleocr_vl::language::{MultimodalRoPE, apply_multimodal_rotary_pos_emb};
use crate::nn::{Activations, Linear, RMSNorm, RoPE};
use crate::transformer::KVCache;
use crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter;
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

        // Scaled dot-product attention using fast kernel.
        // When no explicit mask is provided:
        //   - seq_len > 1 (prefill): use "causal" mode — MLX's fused Metal kernel handles
        //     causal masking internally without materializing an O(N²) mask array.
        //     This matches Python mlx-lm's `create_attention_mask` returning "causal".
        //   - seq_len == 1 (decode): no mask needed (single token only attends to past).
        // When an explicit mask is provided (e.g., sliding window): use it directly.
        let output = if let Some(m) = mask {
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
        self.o_proj.forward(&gated_output)
    }

    /// Forward pass routed through the block-paged KV adapter.
    ///
    /// Mirrors [`Self::forward`] (Q-gating, partial RoPE, Q/K layernorm)
    /// but writes K/V into the paged pool instead of a flat `KVCache`
    /// and reads attention K/V back via either an explicit
    /// `read_kv_range` (cache-hit prefill) or a host-side
    /// `read_kv_range` followed by SDPA (decode). Decode uses
    /// `read_kv_range` instead of `gather_kv_for_decode` to keep BF16
    /// reduction order bit-equal to the flat path's SDPA — matches
    /// Qwen3 / Gemma4's paged decode strategy.
    ///
    /// **Caller contract** (mirrors LFM2 / Gemma4):
    /// 1. `adapter.record_tokens(&[...suffix])` BEFORE this call so the
    ///    adapter cursor is advanced by the chunk; `update_keys_values`
    ///    enforces alignment.
    /// 2. `attn_layer_idx` is the FULL-ATTENTION ORDINAL into the
    ///    adapter pool (NOT the absolute decoder index). Pool was sized
    ///    by `Qwen3_5Config::full_attention_layer_count()`.
    /// 3. The paged forward unconditionally uses standard scalar-offset
    ///    `self.rope` (no M-RoPE branching). Text-only inputs match the
    ///    flat path's behaviour (which uses `self.rope` whenever
    ///    `position_ids = None`); image-bearing turns are rejected
    ///    upstream at the chat-entry sites before reaching this fn.
    ///
    /// Returns `[B, T, hidden_size]` (post-output-projection,
    /// post-gate) so the layer's residual `h = x + r` matches the flat
    /// path.
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

        // Project queries (2x width for gating).
        let q_proj_output = self.q_proj.forward(x)?;

        // Per-head split of queries / gate (matches forward()).
        let q_per_head = q_proj_output.reshape(&[
            batch,
            seq_len,
            self.num_heads as i64,
            (self.head_dim * 2) as i64,
        ])?;
        let queries = q_per_head.slice_axis(3, 0, self.head_dim as i64)?;
        let gate = q_per_head.slice_axis(3, self.head_dim as i64, (self.head_dim * 2) as i64)?;
        let gate = gate.reshape(&[batch, seq_len, (self.num_heads * self.head_dim) as i64])?;

        // K/V projections + reshape to per-head layout.
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

        // QK normalization on the last dim.
        let queries = self.q_norm.forward(&queries)?;
        let keys = self.k_norm.forward(&keys)?;

        // Standard scalar-offset partial RoPE. Paged path is text-only;
        // image-bearing turns are rejected at the chat-entry sites
        // (`chat_sync_core` / `chat_stream_sync_inner` and the MoE
        // counterparts) before reaching this forward.
        let rope_offset = first_logical_position as i32;
        let queries = self.rope.forward(&queries, Some(rope_offset))?;
        let keys = self.rope.forward(&keys, Some(rope_offset))?;

        // Transpose to [B, H, T, D] for SDPA.
        let queries_bhtd = queries.transpose(Some(&[0, 2, 1, 3]))?;
        let keys_bhtd = keys.transpose(Some(&[0, 2, 1, 3]))?;
        let values_bhtd = values.transpose(Some(&[0, 2, 1, 3]))?;

        // Paged-pool layout: `[num_tokens, num_kv_heads, head_dim]`.
        // [B, H_kv, T, D] -> [B, T, H_kv, D] -> [B*T, H_kv, D].
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

        // Compute attention output.
        let attn_bhtd = if is_prefill {
            if cached_prefix_len == 0 {
                // Fresh prefill: SDPA over in-flight Q/K/V with internal
                // causal mask.
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
                // Cache-hit prefill: read full [0, total_ctx) K/V back
                // from the pool. The suffix was just written above. When
                // explicitly enabled, try the MLX paged-attention bridge first
                // so the suffix attends directly against the pool without
                // host-side K/V materialization.
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
            // Decode: prefer graph-native paged attention so native K/V
            // writes and attention reads remain in one MLX dependency graph.
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
                /* softcap */ 1.0,
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
                        .gather_kv_for_decode(
                            attn_layer_idx,
                            &queries_3d,
                            self.scale,
                            /* softcap */ 1.0,
                        )
                        .map_err(napi::Error::from_reason)?
                }
            };
            let target_dtype = x.dtype()?;
            let attn_3d = attn_3d.astype(target_dtype)?;
            attn_3d.reshape(&[1, self.num_heads as i64, 1, self.head_dim as i64])?
        };

        // Transpose back: [B, H, T, D] -> [B, T, H*D].
        let output = attn_bhtd.transpose(Some(&[0, 2, 1, 3]))?;
        let output = output.reshape(&[batch, seq_len, (self.num_heads * self.head_dim) as i64])?;

        // Apply gate: output * sigmoid(gate).
        let gate_sigmoid = Activations::sigmoid(&gate)?;
        let gated_output = output.mul(&gate_sigmoid)?;

        // Output projection.
        self.o_proj.forward(&gated_output)
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
