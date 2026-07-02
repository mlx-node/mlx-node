use std::sync::OnceLock;
use std::time::Instant;

use crate::array::MxArray;
use crate::array::attention::{scaled_dot_product_attention, scaled_dot_product_attention_causal};
use crate::array::mask::create_causal_mask;
use crate::inference_trace::{
    elapsed_ms, enabled as inference_trace_enabled, write as write_inference_trace,
};
use crate::models::paddleocr_vl::language::{
    MultimodalRoPE, apply_interleaved_rotary, apply_multimodal_rotary_pos_emb_interleaved,
    select_interleaved_cos_sin,
};
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

    /// Pre-transposed, OUTPUT-reordered `[hidden, 2*num_heads*head_dim]`
    /// q_proj weight: block order `[Q_h0..Q_h{H-1}, G_h0..G_h{H-1}]` instead
    /// of the checkpoint's per-head-interleaved order
    /// `[Q_h0,G_h0,Q_h1,G_h1,...]`. Populated once by
    /// `finalize_q_gate_block()` after `q_proj` is loaded; invalidated back
    /// to `None` by any `q_proj` setter.
    ///
    /// When present, `project_q_gate` slices queries/gate as two flat,
    /// row-contiguous halves of one matmul output instead of reshaping to
    /// `[B,T,H,2D]` and slicing per head. The per-head split's `gate` slice
    /// has a `2*head_dim` stride between heads, so `reshape([B,T,H*D])`
    /// fails MLX's `prepare_reshape` free-view check and dispatches a real
    /// strided `copy_gpu_inplace` (`CopyType::General`) Metal kernel on
    /// every call — this cache makes that copy a one-time load-time cost
    /// instead of a per-forward one. `None` (falls back to the unfused
    /// per-head path) when `q_proj` is quantized, mirroring
    /// `GatedDeltaNet::finalize_in_proj`.
    q_gate_block_t: Option<MxArray>,
    /// Reordered `[2*num_heads*head_dim]` q_proj bias matching
    /// `q_gate_block_t`'s column order. `None` when q_proj has no bias.
    q_gate_block_bias: Option<MxArray>,
}

fn paged_prefill_paged_attention_enabled() -> bool {
    // Without a Metal backend (CUDA/Linux build) the C++ paged-attention
    // kernel throws, so a flat-path cache-hit prefill must NOT dispatch it.
    // Hard-close the path here so reuse-turn prefills stay on the
    // device-agnostic SDPA fallback. (Single-turn fresh prompts never reach
    // this branch anyway, but this closes the multi-turn case too.)
    if !crate::engine::persistence::compiled_forward_backend_available() {
        return false;
    }
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
            q_gate_block_t: None,
            q_gate_block_bias: None,
        })
    }

    /// Project queries + gate, returning `(queries [B,T,H,D], gate
    /// [B,T,H*D])`.
    ///
    /// Fast path (`q_gate_block_t` present, i.e. `q_proj` is non-quantized
    /// and `finalize_q_gate_block()` has run): one matmul against the
    /// block-ordered weight, then two flat `slice_axis` calls — both
    /// already row-contiguous, so `queries`'s subsequent `[B,T,H,D]`
    /// reshape is a free view and `gate` needs no reshape at all.
    /// `MLX_DISABLE_QGATE_BLOCK_SPLIT=1` forces the fallback below (for
    /// same-binary A/B benchmarking), mirroring
    /// `MLX_DISABLE_E51_STACKED_GDN_IN_PROJ`.
    ///
    /// Fallback path (quantized `q_proj`, or the env override above):
    /// the original per-head reshape+slice, unchanged from before this
    /// split existed. `gate`'s reshape here pays a strided
    /// `copy_gpu_inplace` every call — see `q_gate_block_t`'s doc comment.
    fn project_q_gate(&self, x: &MxArray, batch: i64, seq_len: i64) -> Result<(MxArray, MxArray)> {
        let hd = (self.num_heads * self.head_dim) as i64;
        if let Some(w_block_t) = &self.q_gate_block_t
            && std::env::var("MLX_DISABLE_QGATE_BLOCK_SPLIT").is_err()
        {
            let flat = match &self.q_gate_block_bias {
                Some(bias) => x.addmm(bias, w_block_t, None, None)?,
                None => x.matmul(w_block_t)?,
            };
            let queries_flat = flat.slice_axis(2, 0, hd)?;
            let gate = flat.slice_axis(2, hd, 2 * hd)?;
            let queries = queries_flat.reshape(&[
                batch,
                seq_len,
                self.num_heads as i64,
                self.head_dim as i64,
            ])?;
            Ok((queries, gate))
        } else {
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
            let gate =
                q_per_head.slice_axis(3, self.head_dim as i64, (self.head_dim * 2) as i64)?;
            // Flatten gate for later: [B, T, H, D] → [B, T, H*D]
            let gate = gate.reshape(&[batch, seq_len, hd])?;
            Ok((queries, gate))
        }
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

        // Project queries (2x width for gating), split into per-head
        // queries [B,T,H,D] and flat gate [B,T,H*D]. See `project_q_gate`.
        let (queries, gate) = self.project_q_gate(x, batch, seq_len)?;

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
            // M-RoPE: compute cos/sin from 3D position IDs [3, B, T].
            // Qwen3.5-VL uses the INTERLEAVED (stride-3) per-frequency axis
            // selector, NOT PaddleOCR-VL's contiguous-chunk (sectioned) one.
            let (cos, sin) = mrope.forward(&queries, pos_ids)?;
            // Transpose to [B, H, T, D] for the rotary apply.
            let q_t = queries.transpose(Some(&[0, 2, 1, 3]))?;
            let k_t = keys.transpose(Some(&[0, 2, 1, 3]))?;
            let (q_out, k_out) = apply_multimodal_rotary_pos_emb_interleaved(
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
            // Standard scalar-offset RoPE (text-only path).
            //
            // `fast::rope` varies the rotation position along axis -2 of its
            // input, so it must see the [B, H, T, D] layout (token axis at
            // -2) — matching mlx-lm's `self.rope(x.transpose(0, 2, 1, 3),
            // offset)`. Applying it on [B, T, H, D] rotates along the HEAD
            // axis instead: every token in a multi-token forward gets the
            // same angle (offset + head_index), collapsing per-token
            // positions. Transpose in, rotate, transpose back (the extra
            // transposes are views; qwen3.5's partial rotary
            // (rope_dims < head_dim) takes the rope kernel's copying
            // `dims_ < D` branch either way, so the transposed input costs a
            // strided rather than vector copy — the same price mlx-lm pays).
            let offset = cache.as_ref().map_or(0, |c| c.get_offset());
            let q_t = queries.transpose(Some(&[0, 2, 1, 3]))?;
            let k_t = keys.transpose(Some(&[0, 2, 1, 3]))?;
            let q_rot = self.rope.forward(&q_t, Some(offset))?;
            let k_rot = self.rope.forward(&k_t, Some(offset))?;
            (
                q_rot.transpose(Some(&[0, 2, 1, 3]))?,
                k_rot.transpose(Some(&[0, 2, 1, 3]))?,
            )
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
    /// 3. RoPE selection mirrors [`Self::forward`]: when `position_ids`
    ///    is `Some` and this is a VLM checkpoint (`self.mrope` set), apply
    ///    3-row M-RoPE over those positions (the image-bearing prefill
    ///    path); otherwise use standard scalar-offset `self.rope` from
    ///    `first_logical_position` (the text-only path). The text-only
    ///    `position_ids = None` branch is byte-identical to the flat path's
    ///    `position_ids = None` behaviour.
    ///
    /// Returns `[B, T, hidden_size]` (post-output-projection,
    /// post-gate) so the layer's residual `h = x + r` matches the flat
    /// path.
    /// `mrope_cache` is a per-forward-pass scratch slot for the M-RoPE arm:
    /// every full-attention layer in one Qwen3.5-VL forward pass shares
    /// byte-identical `position_ids`/`mrope_section`/dtype
    /// (`init_mrope_layers` seeds every layer from the same config), so the
    /// FIRST layer to see `Some(position_ids)` computes the selected cos/sin
    /// and stores it here; every later layer in the same forward pass reuses
    /// it instead of recomputing the cos/sin table + `take_along_axis`
    /// gather. Callers outside the per-layer VLM prefill loop (decode / MTP
    /// steps, which always pass `position_ids = None`) can pass `&mut None`
    /// — it is never touched on that path.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_paged(
        &self,
        x: &MxArray,
        adapter: &mut PagedKVCacheAdapter,
        attn_layer_idx: u32,
        first_logical_position: u32,
        cached_prefix_len: u32,
        is_prefill: bool,
        position_ids: Option<&MxArray>,
        rope_position_offset: i32,
        mrope_cache: &mut Option<(MxArray, MxArray)>,
    ) -> Result<MxArray> {
        let batch = x.shape_at(0)?;
        let seq_len = x.shape_at(1)?;

        // Project queries (2x width for gating), split into per-head
        // queries / flat gate (matches forward(); see `project_q_gate`).
        let (queries, gate) = self.project_q_gate(x, batch, seq_len)?;

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

        // RoPE: 3-row M-RoPE over `position_ids` for image-bearing prefill,
        // standard scalar offset otherwise. The M-RoPE arm reproduces the flat
        // path's layout and transpose order exactly ([B,T,H,D] -> [B,H,T,D] ->
        // rotate -> [B,T,H,D]) so the rotation is bf16-bit-identical to flat;
        // the `None` (text-only) arm matches the flat path's scalar-offset
        // behaviour.
        let (queries, keys) = if let (Some(pos_ids), Some(mrope)) = (position_ids, &self.mrope) {
            // Qwen3.5-VL uses the INTERLEAVED (stride-3) per-frequency axis
            // selector, NOT PaddleOCR-VL's contiguous-chunk (sectioned) one.
            //
            // Every full-attention layer in one forward pass shares
            // byte-identical `pos_ids`/`mrope_section`/dtype, so the cos/sin
            // table build + axis-selector gather only needs to run once per
            // forward pass (see `mrope_cache`'s doc comment above), not once
            // per full-attention layer.
            let (cos_final, sin_final) = match mrope_cache {
                Some(cached) => cached.clone(),
                None => {
                    let (cos, sin) = mrope.forward(&queries, pos_ids)?;
                    let selected =
                        select_interleaved_cos_sin(&cos, &sin, mrope.mrope_section_arr())?;
                    *mrope_cache = Some(selected.clone());
                    selected
                }
            };
            let q_t = queries.transpose(Some(&[0, 2, 1, 3]))?;
            let k_t = keys.transpose(Some(&[0, 2, 1, 3]))?;
            let (q_out, k_out) = apply_interleaved_rotary(&q_t, &k_t, &cos_final, &sin_final)?;
            let q_out = q_out.transpose(Some(&[0, 2, 1, 3]))?;
            let k_out = k_out.transpose(Some(&[0, 2, 1, 3]))?;
            (q_out, k_out)
        } else {
            // Scalar-offset RoPE. `rope_position_offset` decouples the
            // rotation position from the physical KV slot: a turn that
            // warm-continues an image prefill rotates at the compressed
            // M-RoPE position (physical slot + a negative cross-turn delta)
            // while K/V still writes at the physical slot below. Text turns
            // pass `rope_position_offset == first_logical_position as i32`.
            //
            // `fast::rope` varies the rotation position along axis -2 of its
            // input, so it must see the [B, H, T, D] layout (token axis at
            // -2) — matching mlx-lm and the flat `forward` above. Applying
            // it on [B, T, H, D] rotates along the HEAD axis, collapsing
            // per-token positions within any multi-token chunk.
            let rope_offset = rope_position_offset;
            let q_t = queries.transpose(Some(&[0, 2, 1, 3]))?;
            let k_t = keys.transpose(Some(&[0, 2, 1, 3]))?;
            let q_rot = self.rope.forward(&q_t, Some(rope_offset))?;
            let k_rot = self.rope.forward(&k_t, Some(rope_offset))?;
            (
                q_rot.transpose(Some(&[0, 2, 1, 3]))?,
                k_rot.transpose(Some(&[0, 2, 1, 3]))?,
            )
        };

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
        self.q_gate_block_t = None; // invalidate block-order cache
        self.q_gate_block_bias = None;
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
        self.q_gate_block_t = None;
        self.q_gate_block_bias = None;
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

    /// Precompute the block-ordered `[hidden, 2*H*D]` q_proj weight (queries
    /// flat | gate flat) so `project_q_gate` can split queries vs. gate as
    /// two flat, row-contiguous slices instead of reshaping to
    /// `[B,T,H,2D]` and slicing per head. See `q_gate_block_t`'s doc
    /// comment for why the checkpoint's native per-head-interleaved column
    /// order forces a strided copy on every call.
    ///
    /// Safe to call repeatedly (idempotent). No-op when `q_proj` is
    /// quantized — mirrors `GatedDeltaNet::finalize_in_proj` (quantized
    /// checkpoints stay on the unfused path).
    pub fn finalize_q_gate_block(&mut self) -> Result<()> {
        let LinearProj::Standard(q_lin) = &self.q_proj else {
            return Ok(());
        };
        let h = self.num_heads as i64;
        let d = self.head_dim as i64;

        let weight = q_lin.get_weight(); // [2*H*D, hidden], per-head-interleaved
        let hidden = weight.shape_at(1)?;
        let w_per_head = weight.reshape(&[h, 2 * d, hidden])?;
        let w_q = w_per_head.slice_axis(1, 0, d)?.reshape(&[h * d, hidden])?;
        let w_g = w_per_head
            .slice_axis(1, d, 2 * d)?
            .reshape(&[h * d, hidden])?;
        let w_block_t = MxArray::concatenate(&w_q, &w_g, 0)?.transpose(Some(&[1, 0]))?;
        w_block_t.eval();
        self.q_gate_block_t = Some(w_block_t);

        self.q_gate_block_bias = match q_lin.get_bias() {
            Some(b) => {
                let b_per_head = b.reshape(&[h, 2 * d])?;
                let b_q = b_per_head.slice_axis(1, 0, d)?.reshape(&[h * d])?;
                let b_g = b_per_head.slice_axis(1, d, 2 * d)?.reshape(&[h * d])?;
                let b_block = MxArray::concatenate(&b_q, &b_g, 0)?;
                b_block.eval();
                Some(b_block)
            }
            None => None,
        };
        Ok(())
    }

    // ========== Quantized setters ==========

    pub fn set_quantized_q_proj(&mut self, ql: QuantizedLinear) {
        self.q_gate_block_t = None;
        self.q_gate_block_bias = None;
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

    /// Whether any of the q/k/v/o projections hold quantized weights.
    ///
    /// Used by the dense/bf16-only MTP save path to detect a quantized MTP
    /// head (loaded from a `--q-mtp all`/`cyankiwi` checkpoint) and refuse
    /// to serialize stale dense weights (see
    /// `Qwen3_5MTPModule::has_quantized_weights`).
    pub fn is_quantized(&self) -> bool {
        self.q_proj.is_quantized()
            || self.k_proj.is_quantized()
            || self.v_proj.is_quantized()
            || self.o_proj.is_quantized()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_cfg() -> Qwen3_5Config {
        Qwen3_5Config {
            vocab_size: 32,
            hidden_size: 32,
            num_layers: 1,
            num_heads: 4,
            num_kv_heads: 2,
            intermediate_size: 64,
            rms_norm_eps: 1e-6,
            head_dim: 8,
            tie_word_embeddings: true,
            attention_bias: false,
            max_position_embeddings: 128,
            pad_token_id: 0,
            eos_token_id: 0,
            bos_token_id: 0,
            linear_num_value_heads: 4,
            linear_num_key_heads: 2,
            linear_key_head_dim: 8,
            linear_value_head_dim: 8,
            linear_conv_kernel_dim: 4,
            full_attention_interval: 4,
            partial_rotary_factor: 0.5,
            rope_theta: 100_000.0,
            paged_cache_memory_mb: None,
            paged_block_size: None,
            use_block_paged_cache: None,
            n_mtp_layers: 0,
        }
    }

    /// RoPE token-axis regression test. Prefills one KV cache with a single
    /// 4-token forward (chunk) and another with 4 single-token forwards
    /// (stepwise), then runs the same probe token through both caches and
    /// compares the outputs.
    ///
    /// `fast::rope` varies the rotation position along axis -2 of its
    /// input. The scalar-offset arm used to rope on `[B, T, H, D]`, which
    /// rotates along the HEAD axis: in the 4-token chunk every token got
    /// the same angle (`offset + head_index`), while the stepwise path got
    /// per-token angles — so the two caches held O(1)-different keys and
    /// the probe outputs diverged (observed max_abs_diff 0.053 with this
    /// setup, vs 1.4e-4 with the fix). With the rotation on `[B, H, T, D]`
    /// the caches agree and the probe outputs match to f32-kernel noise.
    ///
    /// Everything is f32 on purpose: f32 matmuls take the non-NAX Metal
    /// path on gen-17 GPUs, so this test isolates rope-layout semantics
    /// from the half-precision NAX GEMM issues that poison bf16
    /// chunk-vs-stepwise comparisons on M5 hosts (see cleanup-G report).
    #[test]
    fn scalar_rope_rotates_along_token_axis() -> Result<()> {
        let cfg = tiny_cfg();
        let mut attn = Qwen3_5Attention::new(&cfg)?;

        let h = cfg.num_heads as i64;
        let d = cfg.head_dim as i64;
        let hidden = cfg.hidden_size as i64;
        let kv = cfg.num_kv_heads as i64;

        // Deterministic weights, scaled small so multi-layer products stay
        // O(1) in f32.
        let q_w: Vec<f32> = (0..(2 * h * d * hidden))
            .map(|i| ((i as f32) * 0.7391).sin() * 0.2)
            .collect();
        let k_w: Vec<f32> = (0..(kv * d * hidden))
            .map(|i| ((i as f32) * 0.5711 + 1.0).sin() * 0.2)
            .collect();
        let v_w: Vec<f32> = (0..(kv * d * hidden))
            .map(|i| ((i as f32) * 0.9173 + 2.0).sin() * 0.2)
            .collect();
        let o_w: Vec<f32> = (0..(hidden * h * d))
            .map(|i| ((i as f32) * 0.6133 + 3.0).sin() * 0.2)
            .collect();
        attn.set_q_proj_weight(&MxArray::from_float32(&q_w, &[2 * h * d, hidden])?)?;
        attn.set_k_proj_weight(&MxArray::from_float32(&k_w, &[kv * d, hidden])?)?;
        attn.set_v_proj_weight(&MxArray::from_float32(&v_w, &[kv * d, hidden])?)?;
        attn.set_o_proj_weight(&MxArray::from_float32(&o_w, &[hidden, h * d])?)?;

        let x_vals: Vec<f32> = (0..(4 * hidden))
            .map(|i| ((i as f32) * 0.8317).sin())
            .collect();
        let probe_vals: Vec<f32> = (0..hidden)
            .map(|i| ((i as f32) * 0.3719 + 5.0).sin())
            .collect();
        let probe = MxArray::from_float32(&probe_vals, &[1, 1, hidden])?;

        // Chunk prefill: one 4-token forward.
        let mut cache_chunk = KVCache::new();
        let x_full = MxArray::from_float32(&x_vals, &[1, 4, hidden])?;
        let _ = attn.forward(&x_full, None, Some(&mut cache_chunk), None)?;
        assert_eq!(cache_chunk.get_offset(), 4);

        // Stepwise prefill: four 1-token forwards.
        let mut cache_step = KVCache::new();
        for t in 0..4usize {
            let x_t = MxArray::from_float32(
                &x_vals[t * hidden as usize..(t + 1) * hidden as usize],
                &[1, 1, hidden],
            )?;
            let _ = attn.forward(&x_t, None, Some(&mut cache_step), None)?;
        }
        assert_eq!(cache_step.get_offset(), 4);

        // Same probe token through both caches.
        let out_chunk = attn.forward(&probe, None, Some(&mut cache_chunk), None)?;
        let out_step = attn.forward(&probe, None, Some(&mut cache_step), None)?;

        let a = out_chunk.to_float32()?;
        let b = out_step.to_float32()?;
        assert_eq!(a.len(), b.len());
        let mut max_diff = 0.0f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        // Observed ~1.4e-4 with the fix (chunk vs stepwise runs different
        // f32 GEMM/GEMV kernels and softmax reduction orders); the broken
        // head-axis rotation produced ~0.9. 1e-3 sits three orders of
        // magnitude below the failure signal.
        assert!(
            max_diff < 1e-3,
            "chunk-prefilled and stepwise-prefilled caches disagree \
             (max_abs_diff={max_diff}); scalar RoPE is not rotating along \
             the token axis"
        );
        Ok(())
    }

    /// Builds two `Qwen3_5Attention`s from byte-identical q/k/v/o weights:
    /// one with `finalize_q_gate_block()` called (block-order fast path in
    /// `project_q_gate`) and one without (the pre-fix per-head
    /// reshape+slice fallback). Asserts `forward()` produces numerically
    /// identical output on both — proving the q_proj row reorder is a pure
    /// layout change with no effect on the computed queries/gate values.
    #[test]
    fn q_gate_block_split_matches_unfused_fallback() -> Result<()> {
        let cfg = tiny_cfg();
        let mut fast = Qwen3_5Attention::new(&cfg)?;
        let mut slow = Qwen3_5Attention::new(&cfg)?;

        let h = cfg.num_heads as i64;
        let d = cfg.head_dim as i64;
        let hidden = cfg.hidden_size as i64;
        let kv = cfg.num_kv_heads as i64;

        // Deterministic, distinct-per-element weights (iota-derived) so any
        // column-reorder bug shows up as a numeric mismatch rather than
        // hiding behind a symmetric weight matrix.
        let q_w: Vec<f32> = (0..(2 * h * d * hidden))
            .map(|i| (i as f32) * 0.001)
            .collect();
        let k_w: Vec<f32> = (0..(kv * d * hidden))
            .map(|i| (i as f32) * 0.001 + 1.0)
            .collect();
        let v_w: Vec<f32> = (0..(kv * d * hidden))
            .map(|i| (i as f32) * 0.001 + 2.0)
            .collect();
        let o_w: Vec<f32> = (0..(hidden * h * d))
            .map(|i| (i as f32) * 0.001 + 3.0)
            .collect();

        let q_weight = MxArray::from_float32(&q_w, &[2 * h * d, hidden])?;
        let k_weight = MxArray::from_float32(&k_w, &[kv * d, hidden])?;
        let v_weight = MxArray::from_float32(&v_w, &[kv * d, hidden])?;
        let o_weight = MxArray::from_float32(&o_w, &[hidden, h * d])?;

        for attn in [&mut fast, &mut slow] {
            attn.set_q_proj_weight(&q_weight)?;
            attn.set_k_proj_weight(&k_weight)?;
            attn.set_v_proj_weight(&v_weight)?;
            attn.set_o_proj_weight(&o_weight)?;
        }
        fast.finalize_q_gate_block()?;
        // `slow` intentionally left un-finalized: `q_gate_block_t` stays
        // `None`, exercising the pre-fix per-head reshape+slice path.
        assert!(slow.q_gate_block_t.is_none());
        assert!(fast.q_gate_block_t.is_some());

        let x_data: Vec<f32> = (0..(2 * hidden))
            .map(|i| ((i as f32) * 0.01).sin())
            .collect();
        let x = MxArray::from_float32(&x_data, &[1, 2, hidden])?;

        let out_fast = fast.forward(&x, None, None, None)?;
        let out_slow = slow.forward(&x, None, None, None)?;

        let got = out_fast.to_float32()?;
        let want = out_slow.to_float32()?;
        assert_eq!(got.len(), want.len());
        // Empirically bit-identical (both paths compute the same per-column
        // dot products, just via differently-ordered matmul calls); keep a
        // tight-but-nonzero epsilon so the test isn't brittle to a future
        // MLX GEMM version choosing different tiling.
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (g - w).abs() < 1e-6,
                "mismatch at element {i}: fast={g} slow={w}"
            );
        }
        Ok(())
    }

    /// Same parity check as `q_gate_block_split_matches_unfused_fallback`,
    /// but WITH a q_proj bias loaded. This exercises the two bias-only
    /// branches the no-bias test above never reaches: the
    /// `q_lin.get_bias() => Some(b)` bias reorder in `finalize_q_gate_block`
    /// and the `Some(bias) => x.addmm(bias, ...)` fast path in
    /// `project_q_gate`. `Linear::set_bias` accepts a bias regardless of
    /// `attention_bias`, so the same tiny config is reused.
    #[test]
    fn q_gate_block_split_matches_unfused_fallback_with_bias() -> Result<()> {
        let cfg = tiny_cfg();
        let mut fast = Qwen3_5Attention::new(&cfg)?;
        let mut slow = Qwen3_5Attention::new(&cfg)?;

        let h = cfg.num_heads as i64;
        let d = cfg.head_dim as i64;
        let hidden = cfg.hidden_size as i64;
        let kv = cfg.num_kv_heads as i64;

        // Byte-identical iota-derived weights, same as the no-bias test.
        let q_w: Vec<f32> = (0..(2 * h * d * hidden))
            .map(|i| (i as f32) * 0.001)
            .collect();
        let k_w: Vec<f32> = (0..(kv * d * hidden))
            .map(|i| (i as f32) * 0.001 + 1.0)
            .collect();
        let v_w: Vec<f32> = (0..(kv * d * hidden))
            .map(|i| (i as f32) * 0.001 + 2.0)
            .collect();
        let o_w: Vec<f32> = (0..(hidden * h * d))
            .map(|i| (i as f32) * 0.001 + 3.0)
            .collect();
        // Distinct iota-derived q_proj bias, per-head-interleaved `[2*H*D]`
        // to match the checkpoint column order `finalize_q_gate_block`
        // reorders. Nonzero + distinct-per-element so a bias-reorder bug
        // surfaces as a numeric mismatch.
        let q_b: Vec<f32> = (0..(2 * h * d)).map(|i| (i as f32) * 0.01 + 4.0).collect();

        let q_weight = MxArray::from_float32(&q_w, &[2 * h * d, hidden])?;
        let k_weight = MxArray::from_float32(&k_w, &[kv * d, hidden])?;
        let v_weight = MxArray::from_float32(&v_w, &[kv * d, hidden])?;
        let o_weight = MxArray::from_float32(&o_w, &[hidden, h * d])?;
        let q_bias = MxArray::from_float32(&q_b, &[2 * h * d])?;

        for attn in [&mut fast, &mut slow] {
            attn.set_q_proj_weight(&q_weight)?;
            attn.set_k_proj_weight(&k_weight)?;
            attn.set_v_proj_weight(&v_weight)?;
            attn.set_o_proj_weight(&o_weight)?;
            // Load the q_proj bias BEFORE finalize: every q_proj setter
            // invalidates the block cache to `None`, so `finalize` must run
            // last to snapshot both the weight and the bias (matches the
            // production load order).
            attn.set_q_proj_bias(Some(&q_bias))?;
        }
        fast.finalize_q_gate_block()?;
        // `slow` intentionally left un-finalized: exercises the per-head
        // reshape+slice fallback (with `Linear::forward`'s own bias add).
        assert!(slow.q_gate_block_t.is_none());
        assert!(fast.q_gate_block_t.is_some());
        // Proves the bias-reorder branch actually ran (vs. silently taking
        // the `None` arm): `q_gate_block_bias` is populated only when
        // `q_proj` has a bias to reorder.
        assert!(
            fast.q_gate_block_bias.is_some(),
            "finalize_q_gate_block should have reordered the q_proj bias"
        );

        let x_data: Vec<f32> = (0..(2 * hidden))
            .map(|i| ((i as f32) * 0.01).sin())
            .collect();
        let x = MxArray::from_float32(&x_data, &[1, 2, hidden])?;

        let out_fast = fast.forward(&x, None, None, None)?;
        let out_slow = slow.forward(&x, None, None, None)?;

        let got = out_fast.to_float32()?;
        let want = out_slow.to_float32()?;
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (g - w).abs() < 1e-6,
                "mismatch at element {i}: fast={g} slow={w}"
            );
        }
        Ok(())
    }
}
