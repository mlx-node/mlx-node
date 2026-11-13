use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use crate::array::attention::{scaled_dot_product_attention, scaled_dot_product_attention_causal};
use crate::array::{DType, MxArray};
use crate::inference_trace::{
    elapsed_ms, enabled as inference_trace_enabled, write as write_inference_trace,
};
use crate::models::paddleocr_vl::language::{
    MultimodalRoPE, apply_interleaved_rotary, apply_multimodal_rotary_pos_emb_interleaved,
    select_interleaved_cos_sin,
};
use crate::nn::rope_scaling::{RopeScaling, rotary_inverse_frequencies};
use crate::nn::{Activations, Linear, RMSNorm, RoPE};
use crate::transformer::KVCache;
use crate::transformer::paged_flags::{graph_decode_gather_enabled, native_kv_write_enabled};
use crate::transformer::paged_kv_cache_adapter::{PagedKVCacheAdapter, SeqId};
use napi::bindgen_prelude::*;

use super::config::Qwen3_5Config;
use crate::models::quantized_linear::{LinearProj, QuantizedLinear};

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
    // Alternate load-captured operands; native norm parameters remain intact.
    source_qk: Option<(MxArray, MxArray, MxArray)>,

    rope: RoPE,
    /// Optional M-RoPE for VLM mode (3D position encoding: temporal, height, width)
    mrope: Option<MultimodalRoPE>,

    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    scale: f32,
    is_prism_model: bool,

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
    /// instead of a per-forward one. Affine and native GGUF K/IQ q_proj store
    /// the same block order directly in their packed operands instead of using
    /// this duplicate dense cache; other quantization modes retain the fallback.
    q_gate_block_t: Option<MxArray>,
    /// Reordered `[2*num_heads*head_dim]` q_proj bias matching
    /// `q_gate_block_t`'s column order. `None` when q_proj has no bias.
    q_gate_block_bias: Option<MxArray>,

    /// Packed row-merged `[w_k; w_v]` quantized projection plus the k-row
    /// split point, installed by `finalize_kv_proj()` when both projections
    /// are mergeable quantized linears. `k_proj`/`v_proj` are then swapped
    /// for zero-copy row-slice views (getters/set_weight unaffected), so the
    /// merged buffer is the only resident copy.
    kv_proj: Option<(LinearProj, i64)>,
}

static NATIVE_KV_FALLBACK_REPORTED: AtomicBool = AtomicBool::new(false);
static DECODE_GATHER_FALLBACK_REPORTED: AtomicBool = AtomicBool::new(false);

/// What a paged forward reads after storing its rows: one decode token, a
/// prefill chunk (Splash `PagedAttention::addPrefill`) or a verify block
/// (Splash `addVerify`, `kv::verifyAttentionSplits`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PagedPass {
    Decode,
    Prefill,
    Verify,
}

/// Store a chunk's rows into the pool, then attend over the pool (Splash
/// `QwenTarget::addPrefillMixer`: `addPrefillStore`, then `addPrefill` for
/// a fresh, first or cache-hit chunk alike). `queries` `[B, T, Hq, D]`,
/// `keys` / `values` `[B, T, Hkv, D]`; returns `[B, T, Hq * D]` in
/// `out_dtype`.
#[allow(clippy::too_many_arguments)]
pub(super) fn paged_store_attend(
    adapter: &mut PagedKVCacheAdapter,
    attn_layer_idx: u32,
    first_logical_position: u32,
    pass: PagedPass,
    queries: &MxArray,
    keys: &MxArray,
    values: &MxArray,
    scale: f32,
    out_dtype: DType,
) -> Result<MxArray> {
    let batch = queries.shape_at(0)?;
    let seq_len = queries.shape_at(1)?;
    let num_heads = queries.shape_at(2)?;
    let head_dim = queries.shape_at(3)?;
    let num_kv_heads = keys.shape_at(2)?;
    let keys_paged = keys.reshape(&[batch * seq_len, num_kv_heads, head_dim])?;
    let values_paged = values.reshape(&[batch * seq_len, num_kv_heads, head_dim])?;
    let prefill = pass != PagedPass::Decode;

    let trace_enabled = inference_trace_enabled();
    let inference_info_enabled =
        tracing::enabled!(target: "mlx_core::inference", tracing::Level::INFO);
    let inference_debug_enabled =
        tracing::enabled!(target: "mlx_core::inference", tracing::Level::DEBUG);
    let report_prefill = inference_info_enabled && prefill && attn_layer_idx == 0 && seq_len > 8;
    let write_trace_start = (trace_enabled || inference_debug_enabled).then(Instant::now);
    let write_info_start = report_prefill.then(Instant::now);
    // An int8 pool is written only by the native store (Splash
    // `PagedAttention::addPrefillStore`); its error is the turn's error.
    let int8_pool = adapter.cache_dtype() == mlx_paged_attn::metal::MetalDtype::Int8;
    let write_path = if native_kv_write_enabled() || int8_pool {
        match adapter.update_keys_values_native(
            attn_layer_idx,
            &keys_paged,
            &values_paged,
            first_logical_position,
        ) {
            Ok(()) => "native",
            Err(err) if int8_pool => return Err(napi::Error::from_reason(err)),
            Err(err) => {
                if trace_enabled {
                    write_inference_trace(format_args!(
                        "[MLX_TRACE] qwen3.5-attn paged_kv_write_fallback \
                         layer={} first_position={} seq_len={} error={}",
                        attn_layer_idx, first_logical_position, seq_len, err
                    ));
                }
                if inference_info_enabled && attn_layer_idx == 0 {
                    let first_report = !NATIVE_KV_FALLBACK_REPORTED.swap(true, Ordering::Relaxed);
                    if prefill || first_report || first_logical_position.is_multiple_of(32) {
                        tracing::warn!(
                            target: "mlx_core::inference",
                            event = "paged_kv_write_fallback",
                            layer = attn_layer_idx,
                            first_position = first_logical_position,
                            sequence_tokens = seq_len,
                            error = %err,
                            "native paged KV write failed; using legacy write path"
                        );
                    }
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
    if inference_debug_enabled {
        tracing::debug!(
            target: "mlx_core::inference",
            event = "paged_kv_write_done",
            layer = attn_layer_idx,
            first_position = first_logical_position,
            sequence_tokens = seq_len,
            path = write_path,
            elapsed_ms = write_trace_start.map(elapsed_ms).unwrap_or(0.0),
            "paged KV layer write completed"
        );
    }
    if report_prefill {
        tracing::info!(
            target: "mlx_core::inference",
            event = "paged_kv_write_done",
            layer = attn_layer_idx,
            first_position = first_logical_position,
            sequence_tokens = seq_len,
            path = write_path,
            elapsed_ms = write_info_start.map(elapsed_ms).unwrap_or(0.0),
            "paged prefill KV write completed"
        );
    }

    let attn = match pass {
        PagedPass::Prefill | PagedPass::Verify => {
            if batch != 1 {
                return Err(napi::Error::from_reason(format!(
                    "qwen3.5 paged prefill attention reads one sequence; got batch {batch}"
                )));
            }
            let plan_pass = match pass {
                PagedPass::Verify => mlx_paged_attn::prefill_plan::Pass::Verify,
                _ => mlx_paged_attn::prefill_plan::Pass::Prefill,
            };
            let start = (trace_enabled || report_prefill).then(Instant::now);
            let queries = queries.reshape(&[seq_len, num_heads, head_dim])?;
            let attn = adapter
                .paged_prefill_attention(
                    attn_layer_idx,
                    &queries,
                    first_logical_position,
                    plan_pass,
                    scale,
                )
                .map_err(napi::Error::from_reason)?;
            if trace_enabled {
                write_inference_trace(format_args!(
                    "[MLX_TRACE] qwen3.5-attn paged_prefill_attention \
                     layer={} rows={} committed={} pass={:?} graph_ms={:.1}",
                    attn_layer_idx,
                    seq_len,
                    first_logical_position,
                    plan_pass,
                    start.map(elapsed_ms).unwrap_or(0.0)
                ));
            }
            if report_prefill {
                tracing::info!(
                    target: "mlx_core::inference",
                    event = "paged_prefill_attention",
                    layer = attn_layer_idx,
                    rows = seq_len,
                    committed = first_logical_position,
                    pass = ?plan_pass,
                    graph_ms = start.map(elapsed_ms).unwrap_or(0.0),
                    "paged prefill attention graph constructed"
                );
            }
            attn
        }
        PagedPass::Decode => {
            // Graph-native paged attention keeps the native K/V write and
            // the attention read in one MLX dependency graph.
            let queries_3d = queries.reshape(&[1, num_heads, head_dim])?;
            let gather_trace_start = (trace_enabled || inference_debug_enabled).then(Instant::now);
            if !graph_decode_gather_enabled() {
                let attn_3d = adapter
                    .gather_kv_for_decode(
                        attn_layer_idx,
                        &queries_3d,
                        scale,
                        /* softcap */ 1.0,
                    )
                    .map_err(napi::Error::from_reason)?;
                if trace_enabled {
                    write_inference_trace(format_args!(
                        "[MLX_TRACE] qwen3.5-attn decode_gather_done \
                         layer={} path=legacy total_ctx={} elapsed_ms={:.1}",
                        attn_layer_idx,
                        adapter.current_token_count(),
                        gather_trace_start.map(elapsed_ms).unwrap_or(0.0)
                    ));
                }
                attn_3d
            } else {
                match adapter.gather_kv_for_decode_graph(
                    attn_layer_idx,
                    &queries_3d,
                    scale,
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
                        if inference_debug_enabled {
                            tracing::debug!(
                                target: "mlx_core::inference",
                                event = "paged_attention_gather_done",
                                layer = attn_layer_idx,
                                path = "graph",
                                context_tokens = adapter.current_token_count(),
                                elapsed_ms = gather_trace_start.map(elapsed_ms).unwrap_or(0.0),
                                "paged attention layer gather completed"
                            );
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
                        if inference_info_enabled && attn_layer_idx == 0 {
                            let context_tokens = adapter.current_token_count();
                            let first_report =
                                !DECODE_GATHER_FALLBACK_REPORTED.swap(true, Ordering::Relaxed);
                            if first_report || context_tokens.is_multiple_of(32) {
                                tracing::warn!(
                                    target: "mlx_core::inference",
                                    event = "paged_attention_gather_fallback",
                                    layer = attn_layer_idx,
                                    context_tokens,
                                    failed_path = "graph",
                                    fallback_path = "raw",
                                    error = %err,
                                    "graph paged-attention gather failed; using raw gather"
                                );
                            }
                        }
                        let attn_3d = adapter
                            .gather_kv_for_decode(
                                attn_layer_idx,
                                &queries_3d,
                                scale,
                                /* softcap */ 1.0,
                            )
                            .map_err(napi::Error::from_reason)?;
                        if inference_debug_enabled {
                            tracing::debug!(
                                target: "mlx_core::inference",
                                event = "paged_attention_gather_done",
                                layer = attn_layer_idx,
                                path = "raw_fallback",
                                context_tokens = adapter.current_token_count(),
                                elapsed_ms = gather_trace_start.map(elapsed_ms).unwrap_or(0.0),
                                "paged attention layer gather completed"
                            );
                        }
                        attn_3d
                    }
                }
            }
        }
    };
    attn.astype(out_dtype)?
        .reshape(&[batch, seq_len, num_heads * head_dim])
}

impl Qwen3_5Attention {
    pub(crate) fn validate_source_qk_weights(
        &self,
        q: &MxArray,
        k: &MxArray,
        frequencies: &MxArray,
    ) -> Result<()> {
        if self.num_heads != 16
            || self.num_kv_heads != 2
            || self.head_dim != 256
            || self.mrope.is_some()
            || self.rope.is_scaled()
            || self.rope.traditional
            || self.rope.dims != 64
            || self.rope.base.to_bits() != 10_000_000_f32.to_bits()
            || self.rope.scale.to_bits() != 1_f32.to_bits()
            || self.q_norm.eps_f32().to_bits() != 1e-6_f32.to_bits()
            || self.k_norm.eps_f32().to_bits() != 1e-6_f32.to_bits()
            || q.shape()?.as_ref() != [256]
            || k.shape()?.as_ref() != [256]
            || frequencies.shape()?.as_ref() != [32]
            || q.dtype()? != DType::Float32
            || k.dtype()? != DType::Float32
            || frequencies.dtype()? != DType::Float32
        {
            return Err(Error::from_reason(
                "source QK weights require measured text attention geometry and F32 operands",
            ));
        }
        Ok(())
    }

    // The loader validates all layers before this infallible publication.
    pub(crate) fn install_source_qk_weights(
        &mut self,
        q: MxArray,
        k: MxArray,
        frequencies: MxArray,
    ) {
        self.source_qk = Some((q, k, frequencies));
    }

    fn try_source_qk_norm_rope(
        &self,
        queries: &MxArray,
        keys: &MxArray,
        offsets: &MxArray,
    ) -> Result<Option<(MxArray, MxArray)>> {
        let Some((q_weight, k_weight, frequencies)) = &self.source_qk else {
            return Ok(None);
        };
        if self.mrope.is_some()
            || self.rope.is_scaled()
            || self.rope.traditional
            || queries.shape()?.as_ref() != [1, 8, 16, 256]
            || keys.shape()?.as_ref() != [1, 8, 2, 256]
            || queries.dtype()? != DType::BFloat16
            || keys.dtype()? != DType::BFloat16
            || !unsafe { mlx_sys::mlx_source_qk_norm_rope_available() }
        {
            return Ok(None);
        }
        let (mut q, mut k) = (std::ptr::null_mut(), std::ptr::null_mut());
        let ok = unsafe {
            mlx_sys::mlx_source_qk_norm_rope(
                queries.as_raw_ptr(),
                keys.as_raw_ptr(),
                q_weight.as_raw_ptr(),
                k_weight.as_raw_ptr(),
                frequencies.as_raw_ptr(),
                offsets.as_raw_ptr(),
                &mut q,
                &mut k,
            )
        };
        if !ok || q.is_null() || k.is_null() {
            unsafe {
                if !q.is_null() {
                    mlx_sys::mlx_array_delete(q);
                }
                if !k.is_null() {
                    mlx_sys::mlx_array_delete(k);
                }
            }
            // Selected source arithmetic never retries a native attention or
            // mutates a KV frontier a second time after factory failure.
            return Err(Error::from_reason(
                "source QK norm/RoPE factory rejected selected inputs",
            ));
        }
        let q = MxArray::from_handle(q, "source_qk:q")?;
        let k = MxArray::from_handle(k, "source_qk:k")?;
        Ok(Some((q, k)))
    }

    pub(super) fn paged_attention_operand(
        &self,
        x: &MxArray,
        cache_dtype: Option<DType>,
    ) -> Result<MxArray> {
        if (self.is_prism_model || self.q_proj.has_hadamard()) && x.dtype()? == DType::Float32 {
            let dtype = cache_dtype
                .filter(|dtype| matches!(dtype, DType::Float16 | DType::BFloat16))
                .ok_or_else(|| {
                    Error::from_reason("prism_hadamard paged attention requires a 16-bit KV cache")
                })?;
            x.astype(dtype)
        } else {
            Ok(x.clone())
        }
    }

    /// Queries, keys and values as the pool takes them: every operand in the
    /// pool's io dtype (Splash io is always bf16). An f16 or f32 residual
    /// stream, or a sym8 checkpoint's f32 norm weights (f32 Q and K beside a
    /// bf16 V), would otherwise reach a pool that writes one element type.
    pub(super) fn paged_attention_operands(
        &self,
        queries: &MxArray,
        keys: &MxArray,
        values: &MxArray,
        cache_dtype: Option<DType>,
    ) -> Result<(MxArray, MxArray, MxArray)> {
        let queries = self.paged_attention_operand(queries, cache_dtype)?;
        let keys = self.paged_attention_operand(keys, cache_dtype)?;
        let values = self.paged_attention_operand(values, cache_dtype)?;
        let Some(dtype) = cache_dtype else {
            let (q, k, v) = (queries.dtype()?, keys.dtype()?, values.dtype()?);
            if q == k && k == v {
                return Ok((queries, keys, values));
            }
            return Err(Error::from_reason(format!(
                "paged attention operands {q:?}/{k:?}/{v:?} need the pool's io dtype"
            )));
        };
        let cast = |x: MxArray| -> Result<MxArray> {
            if x.dtype()? == dtype {
                Ok(x)
            } else {
                x.astype(dtype)
            }
        };
        Ok((cast(queries)?, cast(keys)?, cast(values)?))
    }

    pub(super) fn set_prism_model(&mut self, enabled: bool) {
        self.is_prism_model = enabled;
    }

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
        let rope = RoPE::yarn(rope_dims, head_dim, config.rope_theta, &config.scaling())?;

        let scale = (head_dim as f32).powf(-0.5);

        Ok(Self {
            q_proj: LinearProj::Standard(q_proj),
            k_proj: LinearProj::Standard(k_proj),
            v_proj: LinearProj::Standard(v_proj),
            o_proj: LinearProj::Standard(o_proj),
            q_norm,
            k_norm,
            source_qk: None,
            rope,
            mrope: None,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
            is_prism_model: false,
            q_gate_block_t: None,
            q_gate_block_bias: None,
            kv_proj: None,
        })
    }

    /// Project queries + gate, returning `(queries [B,T,H,D], gate
    /// [B,T,H*D])`.
    ///
    /// Fast path (`finalize_q_gate_block()` installed either the dense cache or
    /// a packed quantized block layout): one projection followed by a single
    /// `split_sections` — zero-copy views instead of two `slice_axis` copy
    /// dispatches. Queries' `[B,T,H,D]` reshape stays a strided view and
    /// materializes only if a downstream kernel requires contiguous input;
    /// gate feeds elementwise ops directly.
    /// Fallback path (unsupported packed formats or an unfinalized projection):
    /// the original per-head reshape+slice, unchanged from before this
    /// split existed. `gate`'s reshape here pays a strided
    /// `copy_gpu_inplace` every call — see `q_gate_block_t`'s doc comment.
    fn project_q_gate(&self, x: &MxArray, batch: i64, seq_len: i64) -> Result<(MxArray, MxArray)> {
        let hd = (self.num_heads * self.head_dim) as i64;
        if self.q_gate_block_t.is_some() || self.q_proj.has_q_gate_block_layout() {
            let flat = match (&self.q_gate_block_t, &self.q_gate_block_bias) {
                (Some(w_block_t), Some(bias)) => x.addmm(bias, w_block_t, None, None)?,
                (Some(w_block_t), None) => x.matmul(w_block_t)?,
                (None, _) => self.q_proj.forward(x)?,
            };
            let qg = flat.split_sections(&[hd], 2)?;
            let queries_flat = &qg[0];
            let gate = qg[1].clone();
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

    /// Project keys and values. When `finalize_kv_proj` merged the two
    /// quantized projections this is ONE packed matmul plus two axis-2 slices;
    /// otherwise the original two matmuls for incompatible projections.
    fn project_kv(&self, x: &MxArray) -> Result<(MxArray, MxArray)> {
        if let Some((merged, k_rows)) = &self.kv_proj {
            let kv = merged.forward(x)?; // [B, T, k_dim + v_dim]
            let last = kv.ndim()? as usize - 1;
            let kv_split = kv.split_sections(&[*k_rows], last as i32)?;
            return Ok((kv_split[0].clone(), kv_split[1].clone()));
        }
        Ok((self.k_proj.forward(x)?, self.v_proj.forward(x)?))
    }

    /// One dispatch for heterogeneous quantized Q/gate, K and V. A merged
    /// K/V projection remains one segment, preserving its split-K reduction.
    fn project_mixer(
        &self,
        x: &MxArray,
        batch: i64,
        seq_len: i64,
    ) -> Result<(MxArray, MxArray, MxArray, MxArray)> {
        if self.q_proj.has_q_gate_block_layout() {
            let fused = if let Some((merged, _)) = &self.kv_proj {
                crate::models::quantized_linear::mixed_projection_forward(
                    x,
                    &[&self.q_proj, merged],
                )?
            } else {
                crate::models::quantized_linear::mixed_projection_forward(
                    x,
                    &[&self.q_proj, &self.k_proj, &self.v_proj],
                )?
            };
            if let Some(outputs) = fused {
                let hd = i64::from(self.num_heads * self.head_dim);
                let qg = outputs[0].split_sections(&[hd], 2)?;
                let queries = qg[0].reshape(&[
                    batch,
                    seq_len,
                    i64::from(self.num_heads),
                    i64::from(self.head_dim),
                ])?;
                let (keys, values) = if let Some((_, k_rows)) = &self.kv_proj {
                    let kv = outputs[1].split_sections(&[*k_rows], 2)?;
                    (kv[0].clone(), kv[1].clone())
                } else {
                    (outputs[1].clone(), outputs[2].clone())
                };
                return Ok((queries, qg[1].clone(), keys, values));
            }
        }
        let (queries, gate) = self.project_q_gate(x, batch, seq_len)?;
        let (keys, values) = self.project_kv(x)?;
        Ok((queries, gate, keys, values))
    }

    /// Merge `k_proj` + `v_proj` into one packed quantized projection when both
    /// are merge-compatible (`LinearProj::concat_rows`). The originals are
    /// replaced by row-slice views of the merged buffer, so per-projection
    /// getters and the unfused forwards stay correct. Dense or mixed-format
    /// pairs keep the two-matmul path. Idempotent.
    pub fn finalize_kv_proj(&mut self) -> Result<()> {
        if self.kv_proj.is_some() {
            return Ok(());
        }
        if let Some(merged) = self.k_proj.concat_rows(&self.v_proj)? {
            let k_rows = self.k_proj.packed_out_features()?;
            let v_rows = self.v_proj.packed_out_features()?;
            // Preserve calibration keys for consumers of individual views.
            let k_key = self.k_proj.amax_key().map(str::to_owned);
            let v_key = self.v_proj.amax_key().map(str::to_owned);
            self.k_proj = merged.slice_rows(0, k_rows)?.with_amax_key(k_key);
            self.v_proj = merged
                .slice_rows(k_rows, k_rows + v_rows)?
                .with_amax_key(v_key);
            self.kv_proj = Some((merged, k_rows));
        }
        Ok(())
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
        self.forward_impl(x, mask, cache, position_ids, None)
    }

    /// Returns residual-inclusive hidden state; callers must not add again.
    pub(crate) fn forward_source_residual(
        &self,
        x: &MxArray,
        mask: Option<&MxArray>,
        cache: Option<&mut KVCache>,
        position_ids: Option<&MxArray>,
        residual: &MxArray,
    ) -> Result<MxArray> {
        self.forward_impl(x, mask, cache, position_ids, Some(residual))
    }

    fn forward_impl(
        &self,
        x: &MxArray,
        mask: Option<&MxArray>,
        cache: Option<&mut KVCache>,
        position_ids: Option<&MxArray>,
        residual: Option<&MxArray>,
    ) -> Result<MxArray> {
        let batch = x.shape_at(0)?;
        let seq_len = x.shape_at(1)?;

        let (queries, gate, keys, values) = self.project_mixer(x, batch, seq_len)?;

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

        let source_pair =
            if self.source_qk.is_some() && position_ids.is_none() && batch == 1 && seq_len == 8 {
                let offset = cache.as_ref().map_or(0, |c| c.get_offset());
                let offsets = MxArray::from_int32(&[offset], &[1])?;
                self.try_source_qk_norm_rope(&queries, &keys, &offsets)?
            } else {
                None
            };
        let (queries, keys) = if let Some(pair) = source_pair {
            pair
        } else {
            // Apply QK normalization (operates on last dim)
            let queries = self.q_norm.forward(&queries)?;
            let keys = self.k_norm.forward(&keys)?;

            // Apply RoPE: either M-RoPE (VLM) or standard scalar offset (text-only)
            let (queries, keys) = if let (Some(pos_ids), Some(mrope)) = (position_ids, &self.mrope)
            {
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
            (queries, keys)
        };
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
            // The fused vector kernel launches 32×gqa×qL threads in its
            // 2-pass variant — qL·gqa > 32 exceeds the threadgroup limit and
            // MLX falls back to a ~15-op unfused graph (expand + scores matmul
            // + arange mask + where + softmax + matmul). Speculative verify
            // blocks land just over the bound (e.g. qL=7, gqa=6 → 42).
            // Splitting the query block keeps each call inside the fused
            // kernel: causal alignment follows from qL_off = kL − qL, so the
            // head chunk runs against keys truncated to kL − tail while the
            // tail chunk sees the full cache.
            let gqa = (self.num_heads / self.num_kv_heads.max(1)) as i64;
            let vector_dims = matches!(self.head_dim, 64 | 96 | 128 | 256);
            let tail = if (1..=32).contains(&gqa) {
                (32 / gqa).min(seq_len - 1)
            } else {
                0
            };
            if vector_dims && tail >= 1 && seq_len <= 8 && seq_len * gqa > 32 {
                let head_len = seq_len - tail;
                let kv_len = keys.shape_at(2)?;
                let q_parts = queries.split_sections(&[head_len], 2)?;
                let kv_split = keys.split_sections(&[kv_len - tail], 2)?;
                let vv_split = values.split_sections(&[kv_len - tail], 2)?;
                let out_head = if head_len > 1 {
                    scaled_dot_product_attention_causal(
                        &q_parts[0],
                        &kv_split[0],
                        &vv_split[0],
                        self.scale as f64,
                    )?
                } else {
                    // Single-row head: causal is a no-op, it attends to the
                    // whole truncated prefix.
                    scaled_dot_product_attention(
                        &q_parts[0],
                        &kv_split[0],
                        &vv_split[0],
                        self.scale as f64,
                        None,
                    )?
                };
                let out_tail = scaled_dot_product_attention_causal(
                    &q_parts[1],
                    &keys,
                    &values,
                    self.scale as f64,
                )?;
                MxArray::concatenate(&out_head, &out_tail, 2)?
            } else {
                scaled_dot_product_attention_causal(&queries, &keys, &values, self.scale as f64)?
            }
        } else {
            scaled_dot_product_attention(&queries, &keys, &values, self.scale as f64, None)?
        };

        // Transpose back: [B, H, T, D] → [B, T, H, D] → flatten to [B, T, H*D]
        let output = output.transpose(Some(&[0, 2, 1, 3]))?;
        let output = output.reshape(&[batch, seq_len, (self.num_heads * self.head_dim) as i64])?;

        // Apply gate: output * sigmoid(gate) — one compiled fusion instead
        // of a Sigmoid + Multiply pair.
        // gate is already [B, T, H*D] from the per-head split above
        let gated_output = Activations::sigmoid_mul_compiled(&gate, &output)?;

        // Output projection
        self.o_proj
            .forward_mixer_output(&gated_output, None, residual)
    }

    /// Forward pass routed through the block-paged KV adapter.
    ///
    /// Mirrors [`Self::forward`] (Q-gating, partial RoPE, Q/K layernorm)
    /// but stores K/V into the paged pool and then attends over the pool
    /// pages ([`paged_store_attend`]): a prefill chunk or verify block reads
    /// its own rows back from the pool, a decode token takes the paged
    /// decode kernel.
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
    pub(crate) fn forward_paged(
        &self,
        x: &MxArray,
        adapter: &mut PagedKVCacheAdapter,
        attn_layer_idx: u32,
        first_logical_position: u32,
        pass: PagedPass,
        position_ids: Option<&MxArray>,
        rope_position_offset: i32,
        mrope_cache: &mut Option<(MxArray, MxArray)>,
    ) -> Result<MxArray> {
        let batch = x.shape_at(0)?;
        let seq_len = x.shape_at(1)?;

        let (queries, gate, keys, values) = self.project_mixer(x, batch, seq_len)?;
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

        let (queries, keys, values) =
            self.paged_attention_operands(&queries, &keys, &values, adapter.paged_operand_dtype())?;
        let output = paged_store_attend(
            adapter,
            attn_layer_idx,
            first_logical_position,
            pass,
            &queries,
            &keys,
            &values,
            self.scale,
            x.dtype()?,
        )?;

        // Apply gate: output * sigmoid(gate) — one compiled fusion.
        let gated_output = Activations::sigmoid_mul_compiled(&gate, &output)?;

        // Output projection.
        self.o_proj.forward(&gated_output)
    }

    /// Uniform-width hybrid decode and MTP verification. GDN keeps [N,T,H];
    /// only attention packs the time axis into per-request paged query rows.
    pub(crate) fn forward_paged_batched(
        &self,
        x: &MxArray,
        adapter: &mut PagedKVCacheAdapter,
        attn_layer_idx: u32,
        rows: &[(SeqId, u32)],
        preserve_singleton_projection_graphs: bool,
    ) -> Result<MxArray> {
        let shape = x.shape()?;
        if rows.is_empty()
            || shape.as_ref().len() != 3
            || shape[0] != rows.len() as i64
            || shape[1] <= 0
        {
            return Err(Error::from_reason(format!(
                "Qwen3_5Attention::forward_paged_batched expects [N,T,H] for {} rows, got {:?}",
                rows.len(),
                shape.as_ref()
            )));
        }
        if !native_kv_write_enabled() {
            return Err(Error::from_reason(
                "Qwen3.5 batched decode requires native K/V writes",
            ));
        }

        let batch = rows.len() as i64;
        let seq_len = shape[1];
        let offsets = rows
            .iter()
            .map(|&(seq_id, position)| {
                i32::try_from(position).map_err(|_| {
                    Error::from_reason(format!(
                        "Qwen3.5 batched decode sequence {seq_id} position {position} exceeds i32::MAX"
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let offsets = MxArray::from_int32(&offsets, &[batch])?;
        let seq_ids = rows.iter().map(|&(seq_id, _)| seq_id).collect::<Vec<_>>();

        // K-quant projections retain a separate graph per owner.
        // Packed kernels may select a different reduction path for `B > 1`,
        // which can change greedy tokens even though the paged attention
        // operation itself is row-independent. Projection rows are cheap to
        // concatenate and K/V attention below remains one genuine batch.
        let (queries, gate, keys, values) = if preserve_singleton_projection_graphs {
            let mut query_rows = Vec::with_capacity(rows.len());
            let mut gate_rows = Vec::with_capacity(rows.len());
            let mut key_rows = Vec::with_capacity(rows.len());
            let mut value_rows = Vec::with_capacity(rows.len());
            for row in 0..rows.len() {
                let x_row = x.slice_axis(0, row as i64, row as i64 + 1)?;
                let (query, gate, key_raw, value_raw) = self.project_mixer(&x_row, 1, seq_len)?;
                query_rows.push(self.q_norm.forward(&query)?);
                gate_rows.push(gate);
                let key = key_raw.reshape(&[
                    1,
                    seq_len,
                    self.num_kv_heads as i64,
                    self.head_dim as i64,
                ])?;
                key_rows.push(self.k_norm.forward(&key)?);
                value_rows.push(value_raw.reshape(&[
                    1,
                    seq_len,
                    self.num_kv_heads as i64,
                    self.head_dim as i64,
                ])?);
            }
            (
                MxArray::concatenate_many(query_rows.iter().collect(), Some(0))?,
                MxArray::concatenate_many(gate_rows.iter().collect(), Some(0))?,
                MxArray::concatenate_many(key_rows.iter().collect(), Some(0))?,
                MxArray::concatenate_many(value_rows.iter().collect(), Some(0))?,
            )
        } else {
            let (queries, gate, key_raw, value_raw) = self.project_mixer(x, batch, seq_len)?;
            let queries = self.q_norm.forward(&queries)?;
            let keys = self.k_norm.forward(&key_raw.reshape(&[
                batch,
                seq_len,
                self.num_kv_heads as i64,
                self.head_dim as i64,
            ])?)?;
            let values = value_raw.reshape(&[
                batch,
                seq_len,
                self.num_kv_heads as i64,
                self.head_dim as i64,
            ])?;
            (queries, gate, keys, values)
        };
        let queries = queries.transpose(Some(&[0, 2, 1, 3]))?;
        let queries = self.rope.forward_with_offsets(&queries, &offsets)?;
        let keys = keys.transpose(Some(&[0, 2, 1, 3]))?;
        let keys = self.rope.forward_with_offsets(&keys, &offsets)?;
        let values = values.transpose(Some(&[0, 2, 1, 3]))?;
        let (queries, keys, values) =
            self.paged_attention_operands(&queries, &keys, &values, adapter.paged_operand_dtype())?;

        let output = if seq_len == 1 {
            let queries = queries.squeeze(Some(&[2]))?;
            let keys = keys.squeeze(Some(&[2]))?;
            let values = values.squeeze(Some(&[2]))?;
            adapter
                .update_keys_values_native_batched(attn_layer_idx, &keys, &values, rows)
                .map_err(Error::from_reason)?;
            adapter
                .gather_kv_for_decode_graph_batched(
                    attn_layer_idx,
                    &queries,
                    &seq_ids,
                    self.scale,
                    1.0,
                )
                .map_err(Error::from_reason)?
        } else {
            let packed = |array: &MxArray, heads: i32| -> Result<MxArray> {
                array.transpose(Some(&[0, 2, 1, 3]))?.reshape(&[
                    batch * seq_len,
                    heads as i64,
                    self.head_dim as i64,
                ])
            };
            let queries = packed(&queries, self.num_heads)?;
            let keys = packed(&keys, self.num_kv_heads)?;
            let values = packed(&values, self.num_kv_heads)?;
            let ragged = rows
                .iter()
                .map(|&(seq_id, position)| {
                    crate::transformer::paged_kv_cache_adapter::PagedRaggedRow {
                        seq_id,
                        first_logical_position: position,
                        query_len: seq_len as u32,
                    }
                })
                .collect::<Vec<_>>();
            adapter
                .update_keys_values_native_ragged(attn_layer_idx, &keys, &values, &ragged)
                .map_err(Error::from_reason)?;
            adapter
                .gather_kv_for_ragged_graph(attn_layer_idx, &queries, &ragged, self.scale, 1.0)
                .map_err(Error::from_reason)?
        }
        .astype(x.dtype()?)?
        .reshape(&[batch, seq_len, (self.num_heads * self.head_dim) as i64])?;
        let output = Activations::sigmoid_mul_compiled(&gate, &output)?;
        if preserve_singleton_projection_graphs {
            let projected = (0..rows.len())
                .map(|row| {
                    let row = output.slice_axis(0, row as i64, row as i64 + 1)?;
                    self.o_proj.forward(&row)
                })
                .collect::<Result<Vec<_>>>()?;
            MxArray::concatenate_many(projected.iter().collect(), Some(0))
        } else {
            self.o_proj.forward(&output)
        }
    }

    /// Initialize M-RoPE for VLM mode.
    pub fn init_mrope(
        &mut self,
        mrope_section: Vec<i32>,
        rope_theta: f64,
        max_position_embeddings: i32,
        rope_dims: i32,
        scaling: &RopeScaling,
    ) -> Result<()> {
        // Use rope_dims (head_dim * partial_rotary_factor), not full head_dim
        self.mrope = Some(if scaling.yarn() {
            // Splash `ROPE_BUILD_TABLES`: every M-RoPE axis takes the target table.
            MultimodalRoPE::with_inverse_frequencies(
                rope_dims,
                mrope_section,
                &rotary_inverse_frequencies(rope_theta as f32, (rope_dims / 2) as u32, scaling)?,
                scaling.attention_factor(),
            )?
        } else {
            MultimodalRoPE::new(
                rope_dims,
                max_position_embeddings,
                Some(rope_theta),
                mrope_section,
            )?
        });
        Ok(())
    }

    // ========== Weight accessors (standard mode) ==========

    pub fn set_q_proj_weight(&mut self, w: &MxArray) -> Result<()> {
        self.q_gate_block_t = None; // invalidate block-order cache
        self.q_gate_block_bias = None;
        self.q_proj.set_weight(w, "q_proj")
    }
    pub fn set_k_proj_weight(&mut self, w: &MxArray) -> Result<()> {
        self.kv_proj = None;
        self.k_proj.set_weight(w, "k_proj")
    }
    pub fn set_v_proj_weight(&mut self, w: &MxArray) -> Result<()> {
        self.kv_proj = None;
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
        self.kv_proj = None;
        self.k_proj.set_bias(b, "k_proj")
    }
    pub fn set_v_proj_bias(&mut self, b: Option<&MxArray>) -> Result<()> {
        self.kv_proj = None;
        self.v_proj.set_bias(b, "v_proj")
    }
    pub fn set_o_proj_bias(&mut self, b: Option<&MxArray>) -> Result<()> {
        self.o_proj.set_bias(b, "o_proj")
    }
    pub fn set_q_norm_weight(&mut self, w: &MxArray, compute_dtype: DType) -> Result<()> {
        if self.source_qk.is_some() {
            return Err(Error::from_reason(
                "source QK norm operands are immutable; load a new model to change weights",
            ));
        }
        self.q_norm
            .set_weight(&super::sidecar_to_compute_dtype(w, compute_dtype)?)
    }
    pub fn set_k_norm_weight(&mut self, w: &MxArray, compute_dtype: DType) -> Result<()> {
        if self.source_qk.is_some() {
            return Err(Error::from_reason(
                "source QK norm operands are immutable; load a new model to change weights",
            ));
        }
        self.k_norm
            .set_weight(&super::sidecar_to_compute_dtype(w, compute_dtype)?)
    }

    /// Precompute the block-ordered `[hidden, 2*H*D]` q_proj weight (queries
    /// flat | gate flat) so `project_q_gate` can split queries vs. gate as
    /// two flat, row-contiguous slices instead of reshaping to
    /// `[B,T,H,2D]` and slicing per head. See `q_gate_block_t`'s doc
    /// comment for why the checkpoint's native per-head-interleaved column
    /// order forces a strided copy on every call.
    ///
    /// Safe to call repeatedly (idempotent). Affine and native GGUF K/IQ
    /// q_proj consume their native-order packed operands and replace them with
    /// materialized block-order arrays. Other quantization modes remain on the
    /// fallback.
    pub fn finalize_q_gate_block(&mut self) -> Result<()> {
        let h = self.num_heads as i64;
        let d = self.head_dim as i64;

        if let LinearProj::Quantized(q_lin) = &mut self.q_proj {
            q_lin.finalize_packed_q_gate_block(self.num_heads, self.head_dim)?;
            return Ok(());
        }

        let LinearProj::Standard(q_lin) = &self.q_proj else {
            return Err(Error::from_reason(
                "Qwen3.5 q_proj changed variants while finalizing the block layout",
            ));
        };

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
        self.kv_proj = None;
        self.k_proj.set_quantized(ql);
    }
    pub fn set_quantized_v_proj(&mut self, ql: QuantizedLinear) {
        self.kv_proj = None;
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

    /// The per-tensor FP8 activation scale threaded onto the q_proj quantized
    /// backend at load time. Test-only read-back seam: proves a loader carried
    /// `PerLayerQuant::input_amax` from config through to the built
    /// `QuantizedLinear` on this attention block.
    #[cfg(test)]
    pub(crate) fn q_proj_input_amax(&self) -> Option<f32> {
        self.q_proj.input_amax()
    }

    /// Test-only: the q_proj quantization mode (`None` when dense) and
    /// whether its q/gate block was merged (packed reorder or dense cache).
    #[cfg(test)]
    pub(crate) fn q_proj_mode_and_block(&self) -> (Option<String>, bool) {
        let mode = match &self.q_proj {
            LinearProj::Quantized(ql) => Some(ql.mode().to_string()),
            LinearProj::Standard(_) => None,
        };
        (
            mode,
            self.q_gate_block_t.is_some() || self.q_proj.has_q_gate_block_layout(),
        )
    }

    #[cfg(test)]
    pub(crate) fn prism_hadamard_sites(&self) -> (bool, bool, bool, bool) {
        (
            self.q_proj.has_hadamard(),
            self.k_proj.has_hadamard(),
            self.v_proj.has_hadamard(),
            self.o_proj.has_hadamard(),
        )
    }
}

#[cfg(test)]
mod target_schedule_tests;

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_segmented_plan(
        q_len: i32,
        gqa: i32,
        two_pass: bool,
        partitions: i32,
        stage1_width: usize,
        stage1_max_threads: usize,
        stage1_static_memory: usize,
        device_max_memory: usize,
        stage2_width: usize,
        stage2_max_threads: usize,
        stage2_static_memory: usize,
    ) -> (bool, u32, u32) {
        let mut stage1_threads = 0;
        let mut stage2_threads = 0;
        let supported = unsafe {
            mlx_sys::mlx_segmented_sdpa_test_plan(
                q_len,
                gqa,
                two_pass,
                partitions,
                stage1_width,
                stage1_max_threads,
                stage1_static_memory,
                device_max_memory,
                stage2_width,
                stage2_max_threads,
                stage2_static_memory,
                &mut stage1_threads,
                &mut stage2_threads,
            )
        } == 1;
        (supported, stage1_threads, stage2_threads)
    }

    #[test]
    fn segmented_sdpa_capability_planner_accepts_only_supported_profiles() {
        let valid =
            synthetic_segmented_plan(5, 6, true, 128, 32, 1024, 4096, 32768, 32, 1024, 4096);
        assert_eq!(valid, (true, 960, 1024));

        // SIMD width 32 belongs to the reduction algorithm. Device capacities
        // are runtime inputs and must independently reject an unsafe launch.
        assert!(
            !synthetic_segmented_plan(5, 6, true, 128, 16, 1024, 4096, 32768, 32, 1024, 4096).0
        );
        assert!(!synthetic_segmented_plan(5, 6, true, 128, 32, 512, 4096, 32768, 32, 1024, 4096).0);
        assert!(
            !synthetic_segmented_plan(5, 6, true, 128, 32, 1024, 65536, 32768, 32, 1024, 4096).0
        );
        assert!(!synthetic_segmented_plan(5, 6, true, 128, 32, 1024, 4096, 32768, 32, 512, 4096).0);

        let one_pass = synthetic_segmented_plan(8, 6, false, 32, 32, 1024, 4096, 32768, 0, 0, 0);
        assert_eq!(one_pass, (true, 1024, 0));
    }

    #[cfg(target_os = "macos")]
    fn deterministic_bf16(len: usize, salt: u32) -> Vec<u16> {
        (0..len)
            .map(|i| {
                let x = (i as u32).wrapping_mul(0x9e37_79b9).wrapping_add(salt);
                let sign = ((x >> 23) as u16 & 1) << 15;
                sign | 0x3f00 | ((x >> 16) as u16 & 0x7f)
            })
            .collect()
    }

    #[cfg(target_os = "macos")]
    fn concat_sdpa_for_test(
        q: &MxArray,
        pk: &MxArray,
        pv: &MxArray,
        nk: &MxArray,
        nv: &MxArray,
        causal: bool,
    ) -> Result<MxArray> {
        let k = MxArray::concatenate(pk, nk, 2)?;
        let v = MxArray::concatenate(pv, nv, 2)?;
        if causal {
            scaled_dot_product_attention_causal(q, &k, &v, 0.0625)
        } else {
            scaled_dot_product_attention(q, &k, &v, 0.0625, None)
        }
    }

    #[cfg(target_os = "macos")]
    fn segmented_or_concat_split_for_test(
        q: &MxArray,
        pk: &MxArray,
        pv: &MxArray,
        nk: &MxArray,
        nv: &MxArray,
        segmented: bool,
    ) -> Result<MxArray> {
        let q_len = q.shape_at(2)?;
        let gqa = q.shape_at(1)? / pk.shape_at(1)?;
        let probed_max =
            (unsafe { mlx_sys::mlx_segmented_sdpa_max_query_length(gqa as i32) }) as i64;
        let max_q = if probed_max > 0 { probed_max } else { 32 / gqa };
        let call =
            |query: &MxArray, new_k: &MxArray, new_v: &MxArray, causal: bool| -> Result<MxArray> {
                if segmented && pk.shape_at(2)? > 0 {
                    let handle = unsafe {
                        mlx_sys::mlx_segmented_sdpa_test_forward(
                            query.as_raw_ptr(),
                            pk.as_raw_ptr(),
                            pv.as_raw_ptr(),
                            new_k.as_raw_ptr(),
                            new_v.as_raw_ptr(),
                            0.0625,
                            causal,
                        )
                    };
                    MxArray::from_handle(handle, "strict segmented SDPA test")
                } else {
                    concat_sdpa_for_test(query, pk, pv, new_k, new_v, causal)
                }
            };
        if q_len <= max_q {
            return call(q, nk, nv, q_len > 1);
        }
        let tail = max_q.min(q_len - 1);
        let head_len = q_len - tail;
        let q_parts = q.split_sections(&[head_len], 2)?;
        let head_k = nk.slice_axis(2, 0, head_len)?;
        let head_v = nv.slice_axis(2, 0, head_len)?;
        let head = call(&q_parts[0], &head_k, &head_v, head_len > 1)?;
        let tail = call(&q_parts[1], nk, nv, true)?;
        MxArray::concatenate(&head, &tail, 2)
    }

    /// This is intentionally an explicit GPU test: it covers every required
    /// verify width and prefix boundary, including the capability-derived
    /// split at q=8 (3/5 on the measured pipeline). Comparing f32 views is
    /// bit-exact for BF16 because every BF16 value has a unique f32
    /// representation.
    #[test]
    #[ignore = "requires coordinated Metal GPU validation"]
    #[cfg(target_os = "macos")]
    fn segmented_sdpa_matches_concat_bf16_exactly() -> Result<()> {
        if !unsafe { mlx_sys::mlx_metal_is_available() } {
            return Ok(());
        }
        // The dedicated test entry rejects unsupported pipelines instead of
        // falling back to concat SDPA, so eligible parity cases must exercise
        // SegmentedSdpa without changing process-global environment state.
        let max_query_length = unsafe { mlx_sys::mlx_segmented_sdpa_max_query_length(6) };
        eprintln!("segmented SDPA GQA=6 max_query_length={max_query_length}");
        if max_query_length < 1 {
            eprintln!(
                "SKIP segmented SDPA parity: selected Metal pipelines do not support any query width"
            );
            return Ok(());
        }
        // Production uses at most two chunks. A smaller legal pipeline can
        // support some widths without supporting the full eight-row split.
        let max_split_width = i64::from(max_query_length) * 2;
        if max_split_width < 8 {
            eprintln!(
                "SKIP segmented SDPA parity query widths {}..=8: production \
                 two-chunk route supports at most {max_split_width} rows",
                max_split_width + 1
            );
        }
        const D: i64 = 256;
        // Include segment and partition boundaries on either side of the
        // split traversal, including transitions crossed by an eight-row tail.
        for prefix in [
            0_i64, 1, 31, 32, 87, 1015, 1016, 1023, 1024, 4095, 4096, 6143, 6144, 6145, 32759,
            32760, 32767, 32768, 32769,
        ] {
            let prefix_elements = (prefix * D) as usize;
            let pk = MxArray::from_bfloat16(
                &deterministic_bf16(prefix_elements, 0x1234_5678),
                &[1, 1, prefix, D],
            )?;
            let pv = MxArray::from_bfloat16(
                &deterministic_bf16(prefix_elements, 0x8765_4321),
                &[1, 1, prefix, D],
            )?;
            for q_len in 1_i64..=8 {
                if q_len > max_split_width {
                    continue;
                }
                let q_elements = (6 * q_len * D) as usize;
                let kv_elements = (q_len * D) as usize;
                let q = MxArray::from_bfloat16(
                    &deterministic_bf16(q_elements, 0x1357_9bdf),
                    &[1, 6, q_len, D],
                )?;
                let nk = MxArray::from_bfloat16(
                    &deterministic_bf16(kv_elements, 0x2468_ace0),
                    &[1, 1, q_len, D],
                )?;
                let nv = MxArray::from_bfloat16(
                    &deterministic_bf16(kv_elements, 0xfdb9_7531),
                    &[1, 1, q_len, D],
                )?;
                let got = segmented_or_concat_split_for_test(&q, &pk, &pv, &nk, &nv, true)?
                    .to_float32()?;
                let expected = segmented_or_concat_split_for_test(&q, &pk, &pv, &nk, &nv, false)?
                    .to_float32()?;
                assert_eq!(
                    got.as_ref(),
                    expected.as_ref(),
                    "BF16 mismatch at prefix={prefix}, query_length={q_len}"
                );
            }
        }

        // Production layout coverage: projection outputs start as [B,T,H,D]
        // and are transposed into [B,H,T,D] views. Exercise non-unit batch,
        // multiple KV heads, the q=8 split, and both vector-SDPA routes.
        if max_split_width < 8 {
            eprintln!(
                "SKIP segmented SDPA eight-row strided cases: unsupported \
                 production split; supported contiguous widths were checked"
            );
            return Ok(());
        }
        for prefix in [87_i64, 4096] {
            let pk = MxArray::from_bfloat16(
                &deterministic_bf16((2 * 2 * prefix * D) as usize, 0x1020_3040),
                &[2, 2, prefix, D],
            )?;
            let pv = MxArray::from_bfloat16(
                &deterministic_bf16((2 * 2 * prefix * D) as usize, 0x5060_7080),
                &[2, 2, prefix, D],
            )?;
            let q_source = MxArray::from_bfloat16(
                &deterministic_bf16((2 * 8 * 12 * D) as usize, 0x90a0_b0c0),
                &[2, 8, 12, D],
            )?;
            let nk_source = MxArray::from_bfloat16(
                &deterministic_bf16((2 * 8 * 2 * D) as usize, 0xd0e0_f001),
                &[2, 8, 2, D],
            )?;
            let nv_source = MxArray::from_bfloat16(
                &deterministic_bf16((2 * 8 * 2 * D) as usize, 0x1234_abcd),
                &[2, 8, 2, D],
            )?;
            let q = q_source.transpose(Some(&[0, 2, 1, 3]))?;
            let nk = nk_source.transpose(Some(&[0, 2, 1, 3]))?;
            let nv = nv_source.transpose(Some(&[0, 2, 1, 3]))?;
            let got =
                segmented_or_concat_split_for_test(&q, &pk, &pv, &nk, &nv, true)?.to_float32()?;
            let expected =
                segmented_or_concat_split_for_test(&q, &pk, &pv, &nk, &nv, false)?.to_float32()?;
            assert_eq!(
                got.as_ref(),
                expected.as_ref(),
                "production-layout BF16 mismatch at prefix={prefix}, query_length=8"
            );
        }
        Ok(())
    }

    const VERIFY_UNIFIED: i32 = 2;

    /// Tile planner over synthetic limits: (supported, tile keys, threads,
    /// threadgroup bytes, partitions for `total` keys).
    fn tile_plan(
        rows: i32,
        gqa: i32,
        total: i32,
        blocks_override: i32,
        stage1: (usize, usize, usize),
        device_memory: usize,
        stage2: (usize, usize, usize),
    ) -> (bool, u32, u32, u32, u32) {
        let mut out = [0u32; 4];
        let supported = unsafe {
            mlx_sys::mlx_segmented_sdpa_test_verify_tile_plan(
                rows,
                gqa,
                total,
                blocks_override,
                stage1.0,
                stage1.1,
                stage1.2,
                device_memory,
                stage2.0,
                stage2.1,
                stage2.2,
                out.as_mut_ptr(),
            )
        } == 1;
        (supported, out[0], out[1], out[2], out[3])
    }

    /// The simdgroup-matrix verify kernel: 2 simdgroups per 8 queries of
    /// M = gqa x rows, a K/V tile of `tile_n x (256 + 8)` BF16 within three
    /// quarters of the device's threadgroup memory limit plus the score
    /// exchange (simdgroups x 32 x tile_n / 4 floats), and contiguous
    /// partitions of about 8 tiles (32..=1024, multiples of 32).
    #[test]
    fn segmented_verify_tile_planner_follows_device_limits() {
        let caps = (32, 1024, 0);
        let kib = 1024;
        // 24 q heads / 4 kv heads, 8 rows: M = 48 -> 12 simdgroups.
        assert_eq!(
            tile_plan(8, 6, 32776, 0, caps, 32 * kib, caps),
            (true, 32, 384, 32 * 264 * 2 + 12 * 32 * 8 * 4, 160)
        );
        assert_eq!(tile_plan(8, 6, 95, 0, caps, 32 * kib, caps).4, 32);
        assert_eq!(tile_plan(8, 6, 6227, 0, caps, 32 * kib, caps).4, 32);
        assert_eq!(tile_plan(8, 6, 16392, 0, caps, 32 * kib, caps).4, 96);
        assert_eq!(tile_plan(8, 6, 1 << 20, 0, caps, 32 * kib, caps).4, 1024);
        // MLX_SDPA_BLOCKS rounds up to 32, as the vector policy.
        assert_eq!(tile_plan(8, 6, 32776, 100, caps, 32 * kib, caps).4, 128);
        assert_eq!(tile_plan(8, 6, 32776, 4096, caps, 32 * kib, caps).4, 1024);
        // The tile follows the memory limit: 16 KiB -> 16 keys; 32 keys is
        // the ceiling (64 measured slower); 8 KiB fits nothing.
        assert_eq!(tile_plan(8, 6, 32776, 0, caps, 16 * kib, caps).1, 16);
        assert_eq!(tile_plan(8, 6, 32776, 0, caps, 64 * kib, caps).1, 32);
        assert!(!tile_plan(8, 6, 32776, 0, caps, 8 * kib, caps).0);
        // M must be a multiple of 8 and the block at least 2 rows.
        assert!(tile_plan(4, 6, 1000, 0, caps, 32 * kib, caps).0);
        assert!(!tile_plan(5, 6, 1000, 0, caps, 32 * kib, caps).0);
        assert!(!tile_plan(1, 8, 1000, 0, caps, 32 * kib, caps).0);
        assert!(tile_plan(2, 8, 1000, 0, caps, 32 * kib, caps).0);
        // 32 x 8 = 256 queries need 64 simdgroups: over 1024 threads. Wider
        // exchanges shrink the tile: 16 simdgroups (gqa 8) and 32 (gqa 16)
        // take 16 keys.
        assert!(!tile_plan(8, 32, 1000, 0, caps, 32 * kib, caps).0);
        assert_eq!(
            tile_plan(8, 16, 1000, 0, caps, 32 * kib, caps),
            (true, 16, 1024, 16 * 264 * 2 + 32 * 32 * 4 * 4, 32)
        );
        assert_eq!(tile_plan(8, 8, 1000, 0, caps, 32 * kib, caps).1, 16);
        assert_eq!(tile_plan(8, 8, 1000, 0, caps, 32 * kib, caps).2, 512);
        // The pipeline's own limits rule: width, threads, static memory.
        assert!(!tile_plan(8, 6, 1000, 0, (16, 1024, 0), 32 * kib, caps).0);
        assert!(!tile_plan(8, 6, 1000, 0, (32, 256, 0), 32 * kib, caps).0);
        assert_eq!(
            tile_plan(8, 6, 1000, 0, (32, 1024, 12 * kib), 32 * kib, caps).1,
            16
        );
        assert!(!tile_plan(8, 6, 1000, 0, (32, 1024, 26 * kib), 32 * kib, caps).0);
        assert!(!tile_plan(8, 6, 1000, 0, caps, 32 * kib, (32, 512, 0)).0);
    }

    /// Tensor-op planner over synthetic limits: (supported, M, tile keys,
    /// threads, threadgroup bytes, partitions for `total` keys).
    fn nax_plan(
        rows: i32,
        gqa: i32,
        total: i32,
        blocks_override: i32,
        stage1: (usize, usize, usize),
        device_memory: usize,
        stage2: (usize, usize, usize),
    ) -> (bool, u32, u32, u32, u32, u32) {
        let mut out = [0u32; 5];
        let supported = unsafe {
            mlx_sys::mlx_segmented_sdpa_test_verify_nax_plan(
                rows,
                gqa,
                total,
                blocks_override,
                stage1.0,
                stage1.1,
                stage1.2,
                device_memory,
                stage2.0,
                stage2.1,
                stage2.2,
                out.as_mut_ptr(),
            )
        } == 1;
        (supported, out[0], out[1], out[2], out[3], out[4])
    }

    /// The tensor-op verify kernel: 256 threads whatever M, M = gqa x rows a
    /// compiled multiple of 8 up to 64, the scratch (fp32 scores + BF16
    /// probabilities [M][tile_n], fp32 scale [M], two flags) within three
    /// quarters of the device's threadgroup memory limit choosing 64 or 32
    /// keys, and the tile kernel's partition policy at that tile size.
    #[test]
    fn segmented_verify_nax_planner_follows_device_limits() {
        // The tensor-op pipeline reports 256 threads; MLX's reduction 1024.
        let caps = (32, 256, 0);
        let red = (32, 1024, 0);
        let kib = 1024;
        let scratch = |m: u32, n: u32| m * n * 6 + m * 4 + 8;
        // 24 q heads / 4 kv heads, 8 rows: M = 48 -> 64 keys, 512 keys per
        // partition.
        assert_eq!(
            nax_plan(8, 6, 32776, 0, caps, 32 * kib, red),
            (true, 48, 64, 256, scratch(48, 64), 96)
        );
        assert_eq!(nax_plan(8, 6, 95, 0, caps, 32 * kib, red).5, 32);
        assert_eq!(nax_plan(8, 6, 16392, 0, caps, 32 * kib, red).5, 64);
        assert_eq!(nax_plan(8, 6, 1 << 20, 0, caps, 32 * kib, red).5, 1024);
        assert_eq!(nax_plan(8, 6, 32776, 100, caps, 32 * kib, red).5, 128);
        // M = 64 (gqa 8) exceeds the 64-key budget at 32 KiB: 32 keys and
        // 256 keys per partition; M = 56 still fits 64 keys.
        assert_eq!(
            nax_plan(8, 8, 32776, 0, caps, 32 * kib, red),
            (true, 64, 32, 256, scratch(64, 32), 160)
        );
        assert_eq!(nax_plan(7, 8, 1000, 0, caps, 32 * kib, red).2, 64);
        assert_eq!(nax_plan(8, 8, 1000, 0, caps, 64 * kib, red).2, 64);
        // 16 KiB: 64 keys up to M = 24, 32 keys to M = 56, nothing at 64;
        // 8 KiB: 32 keys up to M = 24.
        assert_eq!(nax_plan(4, 6, 1000, 0, caps, 16 * kib, red).2, 64);
        assert_eq!(nax_plan(8, 6, 1000, 0, caps, 16 * kib, red).2, 32);
        assert_eq!(nax_plan(7, 8, 1000, 0, caps, 16 * kib, red).2, 32);
        assert!(!nax_plan(8, 8, 1000, 0, caps, 16 * kib, red).0);
        assert!(!nax_plan(8, 6, 1000, 0, caps, 8 * kib, red).0);
        assert_eq!(nax_plan(4, 6, 1000, 0, caps, 8 * kib, red).2, 32);
        // Every compiled M: multiples of 8 from 8 (gqa 4 x 2) to 64.
        for rows in 2..=8 {
            for gqa in 1..=32 {
                let m = rows * gqa;
                let plan = nax_plan(rows, gqa, 1000, 0, caps, 32 * kib, red);
                assert_eq!(plan.0, m % 8 == 0 && m <= 64, "rows={rows} gqa={gqa}");
                if plan.0 {
                    assert_eq!(plan.1, m as u32);
                }
            }
        }
        assert!(!nax_plan(1, 8, 1000, 0, caps, 32 * kib, red).0, "rows < 2");
        // The pipeline's own limits rule: width, 256 threads, static memory.
        assert!(!nax_plan(8, 6, 1000, 0, (16, 256, 0), 32 * kib, red).0);
        assert!(!nax_plan(8, 6, 1000, 0, (32, 128, 0), 32 * kib, red).0);
        assert!(nax_plan(8, 6, 1000, 0, (32, 1024, 0), 32 * kib, red).0);
        assert_eq!(
            nax_plan(8, 6, 1000, 0, (32, 256, 8 * kib), 32 * kib, red).2,
            32
        );
        assert!(!nax_plan(8, 6, 1000, 0, (32, 256, 26 * kib), 32 * kib, red).0);
        assert!(!nax_plan(8, 6, 1000, 0, caps, 32 * kib, (32, 512, 0)).0);
    }

    /// `[B, H, prefix, D]` inputs for a verify block of `rows` over `prefix`
    /// keys from `[B, H, capacity, D]` caches with NaN past the prefix.
    #[cfg(target_os = "macos")]
    struct TileCase {
        q: MxArray,
        pk: MxArray,
        pv: MxArray,
        nk: MxArray,
        nv: MxArray,
    }

    #[cfg(target_os = "macos")]
    fn tile_case(
        base_k: &MxArray,
        base_v: &MxArray,
        q_heads: i64,
        rows: i64,
        prefix: i64,
        salt: u32,
        exponent_shift: u16,
    ) -> Result<TileCase> {
        const D: i64 = 256;
        let kv_heads = base_k.shape_at(1)?;
        let mut q_bits = deterministic_bf16((q_heads * rows * D) as usize, salt);
        // Adding to the exponent scales exactly: sharper softmax, larger
        // score magnitudes and more running-max changes between tiles.
        q_bits.iter_mut().for_each(|b| *b += exponent_shift << 7);
        Ok(TileCase {
            q: MxArray::from_bfloat16(&q_bits, &[1, q_heads, rows, D])?,
            pk: nan_tailed_prefix(base_k, prefix)?,
            pv: nan_tailed_prefix(base_v, prefix)?,
            nk: MxArray::from_bfloat16(
                &deterministic_bf16((kv_heads * rows * D) as usize, salt ^ 0x5a5a),
                &[1, kv_heads, rows, D],
            )?,
            nv: MxArray::from_bfloat16(
                &deterministic_bf16((kv_heads * rows * D) as usize, salt ^ 0xa5a5),
                &[1, kv_heads, rows, D],
            )?,
        })
    }

    #[cfg(target_os = "macos")]
    fn strict_tile_for_test(c: &TileCase) -> Result<MxArray> {
        let handle = unsafe {
            mlx_sys::mlx_segmented_sdpa_test_forward_tile(
                c.q.as_raw_ptr(),
                c.pk.as_raw_ptr(),
                c.pv.as_raw_ptr(),
                c.nk.as_raw_ptr(),
                c.nv.as_raw_ptr(),
                0.0625,
            )
        };
        MxArray::from_handle(handle, "strict tile segmented SDPA test")
    }

    #[cfg(target_os = "macos")]
    fn strict_nax_for_test(c: &TileCase) -> Result<MxArray> {
        let handle = unsafe {
            mlx_sys::mlx_segmented_sdpa_test_forward_nax(
                c.q.as_raw_ptr(),
                c.pk.as_raw_ptr(),
                c.pv.as_raw_ptr(),
                c.nk.as_raw_ptr(),
                c.nv.as_raw_ptr(),
                0.0625,
            )
        };
        MxArray::from_handle(handle, "strict tensor-op segmented SDPA test")
    }

    /// Whether this device plans the tensor-op kernel for a verify block of
    /// `rows` over `total` keys at `q_heads` / `kv_heads`; `plan` receives
    /// M, tile keys, threads, threadgroup bytes, partitions, pipeline max
    /// threads.
    #[cfg(target_os = "macos")]
    fn nax_supported(
        q_heads: i64,
        kv_heads: i64,
        rows: i64,
        total: i64,
        plan: &mut [u32; 6],
    ) -> bool {
        let supported = unsafe {
            mlx_sys::mlx_segmented_sdpa_test_nax_plan(
                q_heads as i32,
                kv_heads as i32,
                rows as i32,
                total as i32,
                plan.as_mut_ptr(),
            )
        };
        supported == 1
    }

    /// BF16 spacing at `x` (subnormals share the smallest normal spacing).
    fn bf16_ulp(x: f32) -> f32 {
        let exponent = x.abs().max(f32::from_bits(0x0080_0000)).log2().floor();
        2f32.powf(exponent - 7.0)
    }

    /// Max and mean |diff| in BF16 ulps of the output's magnitude (its
    /// largest |expected|). The tile route rounds P to BF16, so its error
    /// is absolute at that scale rather than relative per element.
    fn tile_error(got: &[f32], expected: &[f32]) -> (f32, f64) {
        assert_eq!(got.len(), expected.len());
        let magnitude = expected.iter().fold(0f32, |m, e| m.max(e.abs()));
        let ulp = bf16_ulp(magnitude);
        let mut max_diff = 0f32;
        let mut sum = 0f64;
        for (&g, &e) in got.iter().zip(expected) {
            assert!(g.is_finite(), "non-finite tile output {g}");
            let diff = (g - e).abs();
            max_diff = max_diff.max(diff);
            sum += f64::from(diff);
        }
        (max_diff / ulp, sum / got.len() as f64 / f64::from(ulp))
    }

    /// The block routes sum in another order (fp32 MMAs, BF16 P), so they
    /// are not bit-identical to the vector route; each must stay within 2
    /// BF16 ulps of it over every prefix boundary and block width, with
    /// random and large-magnitude (peaked softmax) scores, on NaN-tailed
    /// caches and the production [B, T, H, D] layout. The tensor-op (NAX)
    /// route is checked on the same blocks wherever this device plans it
    /// (gen-17+, M = gqa x rows <= 64) and must refuse a Q whose heads are
    /// not contiguous with its rows. Runs by default on Metal.
    #[test]
    #[cfg(target_os = "macos")]
    fn segmented_verify_tile_matches_vector_route_within_tolerance() -> Result<()> {
        if !unsafe { mlx_sys::mlx_metal_is_available() } {
            assert!(
                !crate::test_support::metal_required(),
                "MLX_TEST_REQUIRE_METAL=1 but no Metal device"
            );
            eprintln!("SKIP tile verify tolerance: no Metal device");
            return Ok(());
        }
        const D: i64 = 256;
        const HKV: i64 = 4;
        const CAPACITY: i64 = 32_768;
        let mut plan = [0u32; 5];
        let supported = unsafe {
            mlx_sys::mlx_segmented_sdpa_test_tile_plan(24, HKV as i32, 8, 32_776, plan.as_mut_ptr())
        };
        if supported != 1 {
            // Unlike the no-Metal arm above this is a device capability, not
            // an absent backend: a GPU without GPUFamilyApple7 cannot launch
            // the tile kernel, so MLX_TEST_REQUIRE_METAL=1 stays satisfied.
            eprintln!(
                "SKIP tile verify tolerance: this device cannot launch the tile kernel ({supported})"
            );
            return Ok(());
        }
        eprintln!(
            "tile plan 24/4 rows 8: tile_n={} threads={} tg_bytes={} partitions@32776={} \
             pipeline_max_threads={}",
            plan[0], plan[1], plan[2], plan[3], plan[4]
        );
        let base_k = MxArray::from_bfloat16(
            &deterministic_bf16((HKV * CAPACITY * D) as usize, 0x1234_5678),
            &[1, HKV, CAPACITY, D],
        )?;
        let base_v = MxArray::from_bfloat16(
            &deterministic_bf16((HKV * CAPACITY * D) as usize, 0x8765_4321),
            &[1, HKV, CAPACITY, D],
        )?;
        base_k.eval();
        base_v.eval();
        // The segmented primitive rejects an empty prefix (the Rust caller
        // concatenates there), so prefix 0 has no tile route to compare.
        let prefixes = [
            1_i64, 7, 31, 32, 33, 87, 1023, 1024, 4095, 4096, 6219, 32768,
        ];
        // gqa 6 tiles rows 4 and 8 (M % 8), gqa 8 every width 2..=8.
        let layouts: [(i64, Vec<i64>); 2] = [(24, vec![4, 8]), (32, (2..=8).collect())];
        let mut nax_plan = [0u32; 6];
        let nax_any = nax_supported(24, HKV, 8, 32_776, &mut nax_plan);
        if nax_any {
            eprintln!(
                "nax plan 24/4 rows 8: m={} tile_n={} threads={} tg_bytes={} partitions@32776={} \
                 pipeline_max_threads={}",
                nax_plan[0], nax_plan[1], nax_plan[2], nax_plan[3], nax_plan[4], nax_plan[5]
            );
        } else {
            eprintln!("nax route unsupported on this device; checking the tile route only");
        }
        let mut worst = (0f32, 0f64);
        let mut worst_nax = (0f32, 0f64);
        let mut checked = 0usize;
        let mut checked_nax = 0usize;
        for (q_heads, rows_list) in &layouts {
            for &rows in rows_list {
                for &prefix in &prefixes {
                    let nax_here =
                        nax_any && nax_supported(*q_heads, HKV, rows, prefix + rows, &mut nax_plan);
                    // Random, sharp (x8) and adversarial (x64: scores in
                    // the tens, a few keys own the softmax).
                    for (set, shift) in [0u16, 3, 6].into_iter().enumerate() {
                        let salt =
                            0x1357_9bdf ^ (prefix as u32 * 31) ^ (rows as u32) ^ (set as u32);
                        let c = tile_case(&base_k, &base_v, *q_heads, rows, prefix, salt, shift)?;
                        let got = strict_tile_for_test(&c)?.to_float32()?;
                        let expected = segmented_or_concat_split_for_test(
                            &c.q, &c.pk, &c.pv, &c.nk, &c.nv, true,
                        )?
                        .to_float32()?;
                        let (max_ulps, mean_ulps) = tile_error(got.as_ref(), expected.as_ref());
                        assert!(
                            max_ulps <= 2.0 && mean_ulps <= 0.25,
                            "q_heads={q_heads} rows={rows} prefix={prefix} set={set}: \
                             max {max_ulps} / mean {mean_ulps:.4} BF16 ulps of the output magnitude"
                        );
                        worst.0 = worst.0.max(max_ulps);
                        worst.1 = worst.1.max(mean_ulps);
                        checked += 1;
                        if nax_here {
                            let nax = strict_nax_for_test(&c)?.to_float32()?;
                            let (max_ulps, mean_ulps) = tile_error(nax.as_ref(), expected.as_ref());
                            assert!(
                                max_ulps <= 2.0 && mean_ulps <= 0.25,
                                "nax q_heads={q_heads} rows={rows} prefix={prefix} set={set}: \
                                 max {max_ulps} / mean {mean_ulps:.4} BF16 ulps of the output \
                                 magnitude"
                            );
                            worst_nax.0 = worst_nax.0.max(max_ulps);
                            worst_nax.1 = worst_nax.1.max(mean_ulps);
                            checked_nax += 1;
                        }
                    }
                }
            }
        }
        // Production layout: [B, T, H, D] projections transposed into
        // [B, H, T, D] views, batch 2, the prefix from a [B, P, H, D] cache.
        for prefix in [87_i64, 4096, 32766] {
            let q = MxArray::from_bfloat16(
                &deterministic_bf16((2 * 8 * 32 * D) as usize, 0x90a0_b0c0),
                &[2, 8, 32, D],
            )?
            .transpose(Some(&[0, 2, 1, 3]))?;
            let nk = MxArray::from_bfloat16(
                &deterministic_bf16((2 * 8 * 4 * D) as usize, 0xd0e0_f001),
                &[2, 8, 4, D],
            )?
            .transpose(Some(&[0, 2, 1, 3]))?;
            let nv = MxArray::from_bfloat16(
                &deterministic_bf16((2 * 8 * 4 * D) as usize, 0x1234_abcd),
                &[2, 8, 4, D],
            )?
            .transpose(Some(&[0, 2, 1, 3]))?;
            let pk = MxArray::from_bfloat16(
                &deterministic_bf16((2 * prefix * 4 * D) as usize, 0x1020_3040),
                &[2, prefix, 4, D],
            )?
            .transpose(Some(&[0, 2, 1, 3]))?;
            let pv = MxArray::from_bfloat16(
                &deterministic_bf16((2 * prefix * 4 * D) as usize, 0x5060_7080),
                &[2, prefix, 4, D],
            )?
            .transpose(Some(&[0, 2, 1, 3]))?;
            let c = TileCase { q, pk, pv, nk, nv };
            let got = strict_tile_for_test(&c)?.to_float32()?;
            let expected =
                segmented_or_concat_split_for_test(&c.q, &c.pk, &c.pv, &c.nk, &c.nv, true)?
                    .to_float32()?;
            let (max_ulps, mean_ulps) = tile_error(got.as_ref(), expected.as_ref());
            assert!(
                max_ulps <= 2.0 && mean_ulps <= 0.25,
                "production layout prefix={prefix}: max {max_ulps} / mean {mean_ulps:.4} BF16 ulps"
            );
            worst.0 = worst.0.max(max_ulps);
            worst.1 = worst.1.max(mean_ulps);
            checked += 1;
            if nax_any {
                // A [B, T, H, D] view interleaves the heads of a row: the
                // tensor-op route refuses it (the production entry takes the
                // tile kernel there) ...
                assert!(
                    strict_nax_for_test(&c).is_err(),
                    "prefix={prefix}: the nax route accepted a Q with interleaved heads"
                );
                // ... and takes the contiguous [B, H, T, D] Q that RoPE
                // produces in the verify path, with the same K/V views.
                let q = MxArray::from_bfloat16(
                    &deterministic_bf16((2 * 32 * 8 * D) as usize, 0x90a0_b0c0),
                    &[2, 32, 8, D],
                )?;
                let c = TileCase {
                    q,
                    pk: c.pk,
                    pv: c.pv,
                    nk: c.nk,
                    nv: c.nv,
                };
                let nax = strict_nax_for_test(&c)?.to_float32()?;
                let expected =
                    segmented_or_concat_split_for_test(&c.q, &c.pk, &c.pv, &c.nk, &c.nv, true)?
                        .to_float32()?;
                let (max_ulps, mean_ulps) = tile_error(nax.as_ref(), expected.as_ref());
                assert!(
                    max_ulps <= 2.0 && mean_ulps <= 0.25,
                    "nax production K/V layout prefix={prefix}: max {max_ulps} / mean \
                     {mean_ulps:.4} BF16 ulps"
                );
                worst_nax.0 = worst_nax.0.max(max_ulps);
                worst_nax.1 = worst_nax.1.max(mean_ulps);
                checked_nax += 1;
            }
        }
        eprintln!(
            "tile verify: {checked} blocks within tolerance; worst max {} / mean {:.4} BF16 ulps \
             of the output magnitude",
            worst.0, worst.1
        );
        if nax_any {
            assert!(checked_nax > 0, "nax supported but no block took it");
            eprintln!(
                "nax verify: {checked_nax} blocks within tolerance; worst max {} / mean {:.4} \
                 BF16 ulps of the output magnitude",
                worst_nax.0, worst_nax.1
            );
        }
        Ok(())
    }

    /// The strict vector entry never dispatches a block kernel; the tile and
    /// tensor-op entries each dispatch their own kernel exactly once per call
    /// (plus MLX's reduction) and never the other's.
    #[test]
    #[cfg(target_os = "macos")]
    fn segmented_verify_tile_route_is_counted() -> Result<()> {
        if !unsafe { mlx_sys::mlx_metal_is_available() } {
            return Ok(());
        }
        const D: i64 = 256;
        let mut plan = [0u32; 5];
        if unsafe { mlx_sys::mlx_segmented_sdpa_test_tile_plan(24, 4, 8, 1008, plan.as_mut_ptr()) }
            != 1
        {
            eprintln!("SKIP tile route counters: this device cannot launch the tile kernel");
            return Ok(());
        }
        let base_k = MxArray::from_bfloat16(
            &deterministic_bf16((4 * 1100 * D) as usize, 1),
            &[1, 4, 1100, D],
        )?;
        let base_v = MxArray::from_bfloat16(
            &deterministic_bf16((4 * 1100 * D) as usize, 2),
            &[1, 4, 1100, D],
        )?;
        let c = tile_case(&base_k, &base_v, 24, 8, 1000, 3, 0)?;
        let count = |family: &std::ffi::CStr| unsafe {
            mlx_sys::mlx_test_kquant_family_count(family.as_ptr())
        };
        // Enabling the counters clears them.
        unsafe { mlx_sys::mlx_test_kquant_counting(true) };
        let tile = strict_tile_for_test(&c).and_then(|o| o.to_float32());
        let tile_counts = [
            count(c"segmented_sdpa_route_tile"),
            count(c"segmented_sdpa_verify_tile_2pass_1"),
            count(c"segmented_sdpa_2pass_2"),
        ];
        let tile_nax_counts = [
            count(c"segmented_sdpa_route_nax"),
            count(c"segmented_sdpa_verify_nax_2pass_1"),
        ];
        unsafe { mlx_sys::mlx_test_kquant_counting(false) };
        tile?;
        assert_eq!(tile_counts, [1, 1, 1], "tile route counters");
        assert_eq!(tile_nax_counts, [0, 0], "tile entry touched the nax route");
        let mut nax_plan = [0u32; 6];
        if nax_supported(24, 4, 8, 1008, &mut nax_plan) {
            unsafe { mlx_sys::mlx_test_kquant_counting(true) };
            let nax = strict_nax_for_test(&c).and_then(|o| o.to_float32());
            let nax_counts = [
                count(c"segmented_sdpa_route_nax"),
                count(c"segmented_sdpa_verify_nax_2pass_1"),
                count(c"segmented_sdpa_2pass_2"),
                count(c"segmented_sdpa_route_tile"),
                count(c"segmented_sdpa_verify_tile_2pass_1"),
            ];
            unsafe { mlx_sys::mlx_test_kquant_counting(false) };
            nax?;
            assert_eq!(nax_counts, [1, 1, 1, 0, 0], "nax route counters");
        } else {
            eprintln!("nax route unsupported on this device; tile counters only");
        }
        // Its own counting window: the vector entry never reaches a block
        // kernel.
        let (_, route) = strict_segmented_routed(&c.q, &c.pk, &c.pv, &c.nk, &c.nv)?;
        let after_vector = [
            count(c"segmented_sdpa_route_tile"),
            count(c"segmented_sdpa_verify_tile_2pass_1"),
            count(c"segmented_sdpa_route_nax"),
            count(c"segmented_sdpa_verify_nax_2pass_1"),
        ];
        assert_eq!(
            after_vector,
            [0, 0, 0, 0],
            "vector entry (route {route}) touched a block route"
        );
        Ok(())
    }

    /// The crossover selector over synthetic timings (pure, no GPU): the
    /// smallest count from which the block route stays at least as fast
    /// through the largest count; one past the largest when it never is; a
    /// win followed by a loss is not a crossover; non-positive or NaN
    /// timings lose; the result is clamped to [256, 8192].
    #[test]
    fn segmented_block_min_keys_selection_follows_timings() {
        fn select(keys: &[i32], vector: &[f64], block: &[f64]) -> i32 {
            assert_eq!(keys.len(), vector.len());
            assert_eq!(keys.len(), block.len());
            unsafe {
                mlx_sys::mlx_segmented_sdpa_test_select_block_min_keys(
                    keys.as_ptr(),
                    vector.as_ptr(),
                    block.as_ptr(),
                    keys.len(),
                )
            }
        }
        let keys = [256, 512, 1024, 2048, 4096];
        let vector = [10.0, 12.0, 20.0, 40.0, 80.0];
        // Equal at 1024, block ahead above: 2048 (a tie keeps the exact
        // vector route; the block route must win by kSegmentedBlockMinGain).
        assert_eq!(
            select(&keys, &vector, &[13.0, 14.0, 20.0, 30.0, 60.0]),
            2048
        );
        // Ahead at 1024 by less than the required gain: still 2048.
        assert_eq!(
            select(&keys, &vector, &[13.0, 14.0, 19.5, 30.0, 60.0]),
            2048
        );
        // Block ahead by the gain everywhere: the floor.
        assert_eq!(select(&keys, &vector, &[9.0, 11.0, 19.0, 30.0, 60.0]), 256);
        // Never ahead: one past the largest count.
        assert_eq!(
            select(&keys, &vector, &[11.0, 13.0, 21.0, 41.0, 81.0]),
            4097
        );
        // Ahead at 512, behind at 1024: 2048 (a win followed by a loss).
        assert_eq!(
            select(&keys, &vector, &[11.0, 11.0, 21.0, 30.0, 60.0]),
            2048
        );
        // A NaN or zero block sample is a loss at that point.
        assert_eq!(
            select(&keys, &vector, &[9.0, 11.0, f64::NAN, 30.0, 60.0]),
            2048
        );
        assert_eq!(select(&keys, &vector, &[9.0, 11.0, 19.0, 30.0, 0.0]), 4097);
        // So is a zero vector sample.
        assert_eq!(
            select(
                &keys,
                &[10.0, 12.0, 20.0, 0.0, 80.0],
                &[9.0, 11.0, 19.0, 30.0, 60.0]
            ),
            4096
        );
        // Clamp below the floor and above the ceiling.
        assert_eq!(select(&[64, 128], &[1.0, 2.0], &[0.5, 1.0]), 256);
        assert_eq!(select(&[8192, 16384], &[1.0, 2.0], &[2.0, 3.0]), 8192);
        assert_eq!(select(&[16384], &[1.0], &[0.5]), 8192);
        assert_eq!(select(&[], &[], &[]), 8192);
        assert_eq!(
            unsafe {
                mlx_sys::mlx_segmented_sdpa_test_select_block_min_keys(
                    std::ptr::null(),
                    vector.as_ptr(),
                    vector.as_ptr(),
                    keys.len(),
                )
            },
            -1
        );
    }

    /// `strict_segmented_for_test` evaluated, and the verify route its
    /// `eval_gpu` dispatched, from the bridge kernel counters.
    #[cfg(target_os = "macos")]
    fn strict_segmented_routed(
        q: &MxArray,
        pk: &MxArray,
        pv: &MxArray,
        nk: &MxArray,
        nv: &MxArray,
    ) -> Result<(napi::bindgen_prelude::Float32Array, i32)> {
        let count = |family: &std::ffi::CStr| unsafe {
            mlx_sys::mlx_test_kquant_family_count(family.as_ptr())
        };
        unsafe { mlx_sys::mlx_test_kquant_counting(true) };
        let out = strict_segmented_for_test(q, pk, pv, nk, nv).and_then(|o| o.to_float32());
        let routes = [
            c"segmented_sdpa_route_single",
            c"segmented_sdpa_route_one_pass",
            c"segmented_sdpa_route_unified",
            c"segmented_sdpa_route_split",
        ]
        .map(count);
        let verify_kernels = count(c"segmented_sdpa_verify_2pass_1");
        unsafe { mlx_sys::mlx_test_kquant_counting(false) };
        let out = out?;
        assert_eq!(
            routes.iter().sum::<u64>(),
            1,
            "one verify dispatch, routes {routes:?}"
        );
        let route = routes.iter().position(|&n| n == 1).unwrap_or(0) as i32;
        assert_eq!(
            verify_kernels,
            u64::from(route == VERIFY_UNIFIED),
            "one-call verify kernel dispatches on route {route}"
        );
        Ok((out, route))
    }

    #[cfg(target_os = "macos")]
    fn strict_segmented_for_test(
        q: &MxArray,
        pk: &MxArray,
        pv: &MxArray,
        nk: &MxArray,
        nv: &MxArray,
    ) -> Result<MxArray> {
        let handle = unsafe {
            mlx_sys::mlx_segmented_sdpa_test_forward(
                q.as_raw_ptr(),
                pk.as_raw_ptr(),
                pv.as_raw_ptr(),
                nk.as_raw_ptr(),
                nv.as_raw_ptr(),
                0.0625,
                true,
            )
        };
        MxArray::from_handle(handle, "strict one-call segmented SDPA test")
    }

    /// `[B, H, prefix, D]` view of a `[B, H, prefix + 64, D]` cache whose rows
    /// past the prefix are NaN, as the compiled verify slices its KV cache.
    #[cfg(target_os = "macos")]
    fn nan_tailed_prefix(base: &MxArray, prefix: i64) -> Result<MxArray> {
        let batch = base.shape_at(0)?;
        let heads = base.shape_at(1)?;
        let d = base.shape_at(3)?;
        let nan = MxArray::from_bfloat16(
            &vec![0x7fc0_u16; (batch * heads * 64 * d) as usize],
            &[batch, heads, 64, d],
        )?;
        let cache = MxArray::concatenate(&base.slice_axis(2, 0, prefix)?, &nan, 2)?;
        cache.eval();
        cache.slice_axis(2, 0, prefix)
    }

    #[test]
    fn prism_hadamard_paged_operands_cast_only_rotated_fp32() -> Result<()> {
        use crate::quant::prism_hadamard::HadamardTransform;
        let cfg = Qwen3_5Config {
            hidden_size: 1024,
            ..tiny_cfg()
        };
        let mut attention = Qwen3_5Attention::new(&cfg)?;
        let x = MxArray::from_float32(&[0.1, -0.2, 0.3, 0.4, -0.5, 0.6], &[1, 2, 3])?;
        assert_eq!(
            attention
                .paged_attention_operand(&x, Some(DType::BFloat16))?
                .dtype()?,
            DType::Float32
        );
        let projection = QuantizedLinear::new(
            MxArray::zeros(&[64, 64], Some(DType::Uint32))?,
            MxArray::from_float16(&vec![0x3800; 64 * 8], &[64, 8])?,
            Some(MxArray::from_float16(&vec![0xb800; 64 * 8], &[64, 8])?),
            None,
            128,
            2,
            "affine".to_string(),
        )
        .with_hadamard(Some(HadamardTransform {
            signs: MxArray::from_float32(&vec![1.0; 1024], &[1024])?,
            block_size: 1024,
            gdn_permutation: None,
        }))?;
        attention.set_quantized_q_proj(projection);
        for dtype in [DType::Float16, DType::BFloat16] {
            let actual = attention.paged_attention_operand(&x, Some(dtype))?;
            assert_eq!(actual.dtype()?, dtype);
            assert_eq!(actual.shape()?.as_ref(), x.shape()?.as_ref());
            assert_eq!(
                actual.astype(DType::Float32)?.to_float32()?.as_ref(),
                x.astype(dtype)?
                    .astype(DType::Float32)?
                    .to_float32()?
                    .as_ref()
            );
        }
        let half = x.astype(DType::Float16)?;
        assert_eq!(
            attention
                .paged_attention_operand(&half, Some(DType::BFloat16))?
                .dtype()?,
            DType::Float16
        );
        assert!(attention.paged_attention_operand(&x, None).is_err());
        assert!(
            attention
                .paged_attention_operand(&x, Some(DType::Float32))
                .is_err()
        );
        Ok(())
    }

    /// Gate: a sym8 checkpoint's f32 Q/K beside a bf16 V, and a uniform f32
    /// or f16 residual stream, reach the pool as one element type, the
    /// pool's. Mutation N7 (mixed operands pass through uncast) → red;
    /// mutation F1 (uniform operands pass through uncast) → red.
    #[test]
    fn mixed_paged_operands_take_the_pool_dtype() -> Result<()> {
        let attention = Qwen3_5Attention::new(&tiny_cfg())?;
        let f32 = MxArray::from_float32(&[0.1, -0.2, 0.3, 0.4], &[1, 1, 4])?;
        let bf16 = f32.astype(DType::BFloat16)?;
        let f16 = f32.astype(DType::Float16)?;
        let (q, k, v) =
            attention.paged_attention_operands(&f32, &f32, &bf16, Some(DType::BFloat16))?;
        for operand in [&q, &k, &v] {
            assert_eq!(operand.dtype()?, DType::BFloat16);
        }
        for uniform in [&f32, &f16] {
            let (q, k, v) = attention.paged_attention_operands(
                uniform,
                uniform,
                uniform,
                Some(DType::BFloat16),
            )?;
            for operand in [&q, &k, &v] {
                assert_eq!(operand.dtype()?, DType::BFloat16, "{:?}", uniform.dtype()?);
            }
        }
        let (q, k, v) = attention.paged_attention_operands(&f32, &f32, &f32, None)?;
        for operand in [&q, &k, &v] {
            assert_eq!(operand.dtype()?, DType::Float32);
        }
        Ok(())
    }

    #[test]
    fn prism_hadamard_paged_operands_cast_for_dense_attention() -> Result<()> {
        let mut attention = Qwen3_5Attention::new(&tiny_cfg())?;
        let x = MxArray::from_float32(&[0.1, -0.2, 0.3, 0.4], &[1, 1, 4])?;
        assert_eq!(
            attention.prism_hadamard_sites(),
            (false, false, false, false)
        );
        attention.set_prism_model(true);
        for dtype in [DType::Float16, DType::BFloat16] {
            let actual = attention.paged_attention_operand(&x, Some(dtype))?;
            assert_eq!(actual.dtype()?, dtype);
            assert_eq!(actual.shape()?.as_ref(), x.shape()?.as_ref());
            assert_eq!(
                actual.astype(DType::Float32)?.to_float32()?.as_ref(),
                x.astype(dtype)?
                    .astype(DType::Float32)?
                    .to_float32()?
                    .as_ref()
            );
            let half = x.astype(dtype)?;
            assert_eq!(
                attention
                    .paged_attention_operand(&half, Some(dtype))?
                    .as_raw_ptr(),
                half.as_raw_ptr()
            );
        }
        assert!(attention.paged_attention_operand(&x, None).is_err());
        assert!(
            attention
                .paged_attention_operand(&x, Some(DType::Float32))
                .is_err()
        );
        attention.set_prism_model(false);
        assert_eq!(
            attention
                .paged_attention_operand(&x, Some(DType::BFloat16))?
                .as_raw_ptr(),
            x.as_raw_ptr()
        );
        Ok(())
    }

    pub(super) fn tiny_cfg() -> Qwen3_5Config {
        Qwen3_5Config {
            rope_scaling: None,
            qwen35_gguf_gdn_layout: None,
            kv_format: None,
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
            persist_paged_cache: None,
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

    fn quantize_affine(weight: &MxArray) -> Result<(MxArray, MxArray, MxArray)> {
        let mut out_q: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        let mut out_s: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        let mut out_b: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        let ok = unsafe {
            mlx_sys::mlx_quantize(
                weight.as_raw_ptr(),
                32,
                4,
                c"affine".as_ptr(),
                &mut out_q,
                &mut out_s,
                &mut out_b,
            )
        };
        if !ok {
            return Err(Error::from_reason("mlx_quantize affine failed"));
        }
        Ok((
            MxArray::from_handle(out_q, "q_gate_test_weight")?,
            MxArray::from_handle(out_s, "q_gate_test_scales")?,
            MxArray::from_handle(out_b, "q_gate_test_biases")?,
        ))
    }

    #[test]
    fn affine_quantized_q_gate_block_matches_native_h4_with_bias() -> Result<()> {
        let cfg = tiny_cfg();
        assert!(
            cfg.num_heads > 1,
            "fixture must cover the multi-head permutation"
        );
        let h = cfg.num_heads as i64;
        let d = cfg.head_dim as i64;
        let hidden = cfg.hidden_size as i64;
        let rows = 2 * h * d;

        let dense_values = (0..rows * hidden)
            .map(|i| ((i as f32) * 0.017 + 0.3).sin() * 0.5)
            .collect::<Vec<_>>();
        let dense =
            MxArray::from_float32(&dense_values, &[rows, hidden])?.astype(DType::BFloat16)?;
        let (weight, scales, quant_biases) = quantize_affine(&dense)?;
        let additive_values = (0..rows)
            .map(|i| ((i as f32) * 0.031 - 0.2).cos() * 0.1)
            .collect::<Vec<_>>();
        let additive_bias =
            MxArray::from_float32(&additive_values, &[rows])?.astype(DType::BFloat16)?;

        let make_q_proj = || {
            QuantizedLinear::new(
                weight.clone(),
                scales.clone(),
                Some(quant_biases.clone()),
                Some(additive_bias.clone()),
                32,
                4,
                crate::models::quantized_linear::DEFAULT_QUANT_MODE.to_string(),
            )
        };
        let mut block = Qwen3_5Attention::new(&cfg)?;
        let mut native = Qwen3_5Attention::new(&cfg)?;
        block.set_quantized_q_proj(make_q_proj());
        native.set_quantized_q_proj(make_q_proj());
        let block_original_weight = block.get_q_proj_weight().as_raw_ptr();
        let native_original_weight = native.get_q_proj_weight().as_raw_ptr();

        block.finalize_q_gate_block()?;
        // The independent reference keeps its original unfinalized layout.
        assert!(block.q_proj.has_q_gate_block_layout());
        assert!(!native.q_proj.has_q_gate_block_layout());
        assert!(
            block.q_gate_block_t.is_none(),
            "quantized block order must replace operands, not retain a dense cache"
        );
        assert_ne!(
            block.get_q_proj_weight().as_raw_ptr(),
            block_original_weight
        );
        assert_eq!(
            native.get_q_proj_weight().as_raw_ptr(),
            native_original_weight
        );

        let x_values = (0..2 * hidden)
            .map(|i| ((i as f32) * 0.023 + 0.7).sin())
            .collect::<Vec<_>>();
        let x = MxArray::from_float32(&x_values, &[1, 2, hidden])?.astype(DType::BFloat16)?;
        let (block_q, block_gate) = block.project_q_gate(&x, 1, 2)?;
        let (native_q, native_gate) = native.project_q_gate(&x, 1, 2)?;
        MxArray::eval_arrays(&[&block_q, &block_gate, &native_q, &native_gate])?;
        assert_eq!(
            block_q.to_uint16_native()?,
            native_q.to_uint16_native()?,
            "block-order affine queries must be bit-identical to native per-head splitting"
        );
        assert_eq!(
            block_gate.to_uint16_native()?,
            native_gate.to_uint16_native()?,
            "block-order affine gates must be bit-identical to native per-head splitting"
        );
        Ok(())
    }

    #[test]
    fn mixed_projection_dense_attention_hook_preserves_q_gate_and_kv_views() -> Result<()> {
        if !unsafe { mlx_sys::mlx_metal_is_available() } {
            return Ok(());
        }
        let mut cfg = tiny_cfg();
        cfg.hidden_size = 1024;
        cfg.head_dim = 64;
        cfg.num_heads = 4;
        cfg.num_kv_heads = 2;
        let make = |n: i64, seed: u32| -> Result<QuantizedLinear> {
            let k = 1024;
            let mut state = seed;
            let mut next = || {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                state
            };
            let w = (0..n * k / 8).map(|_| next()).collect::<Vec<_>>();
            let scales = (0..n * k / 16)
                .map(|_| (next() % 48 + 1) as u8)
                .collect::<Vec<_>>();
            let biases = vec![0x1800u16; (n * k / 128) as usize];
            let mut q = QuantizedLinear::new(
                MxArray::from_uint32(&w, &[n, k / 8])?,
                MxArray::from_uint8(&scales, &[n, k / 16])?,
                Some(MxArray::from_float16(&biases, &[n, k / 128])?),
                None,
                32,
                4,
                "q4k".to_string(),
            );
            assert!(q.tile_kquant_layout()?);
            Ok(q)
        };
        for merged_kv in [false, true] {
            let mut attention = Qwen3_5Attention::new(&cfg)?;
            attention.set_quantized_q_proj(make(512, 17)?);
            attention.set_quantized_k_proj(make(128, 23)?);
            attention.set_quantized_v_proj(make(128, 29)?);
            attention.finalize_q_gate_block()?;
            if merged_kv {
                attention.finalize_kv_proj()?;
            }
            let x = MxArray::from_float32(
                &(0..8192)
                    .map(|i| ((i as f32) * 0.017).sin())
                    .collect::<Vec<_>>(),
                &[1, 8, 1024],
            )?
            .astype(DType::BFloat16)?;
            let (q, g, k, v) = attention.project_mixer(&x, 1, 8)?;
            let (rq, rg) = attention.project_q_gate(&x, 1, 8)?;
            let (rk, rv) = attention.project_kv(&x)?;
            MxArray::eval_arrays(&[&q, &g, &k, &v, &rq, &rg, &rk, &rv])?;
            for (actual, expected) in [(&q, &rq), (&g, &rg), (&k, &rk), (&v, &rv)] {
                assert_eq!(actual.shape()?.as_ref(), expected.shape()?.as_ref());
                assert_eq!(actual.to_uint16_native()?, expected.to_uint16_native()?);
            }
        }
        Ok(())
    }

    #[test]
    fn q4k_quantized_q_gate_block_matches_native_h4_with_bias() -> Result<()> {
        use crate::utils::gguf_kquant::{KQuantFormat, KQuantScales, repack_kquant};

        let mut cfg = tiny_cfg();
        cfg.hidden_size = 256;
        cfg.head_dim = 32;
        cfg.num_heads = 4;
        cfg.num_kv_heads = 2;
        cfg.intermediate_size = 512;
        let h = cfg.num_heads as i64;
        let d = cfg.head_dim as i64;
        let hidden = cfg.hidden_size as i64;
        let rows = 2 * h * d;
        let format = KQuantFormat::Q4K;

        // One deterministic ggml Q4_K block per output row. Repack through the
        // same importer as a UD GGUF; keep d/dmin small and finite so the M=2
        // qmv_wide comparison is numerically well-conditioned.
        let mut state = 0x5147_4b21u64;
        let mut blocks = (0..rows as usize * format.block_bytes())
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (state >> 33) as u8
            })
            .collect::<Vec<_>>();
        for row in 0..rows as usize {
            let base = row * format.block_bytes();
            blocks[base] = 0x00;
            blocks[base + 1] = 0x24; // d = f16 0.015625
            blocks[base + 2] = 0x00;
            blocks[base + 3] = 0x20; // dmin = f16 0.0078125
        }
        let packed = repack_kquant(format, &blocks, rows as usize, hidden as usize)?;
        let weight = MxArray::from_uint32(
            &packed.weight,
            &[rows, format.weight_cols(hidden as usize) as i64],
        )?;
        let scales = match &packed.scales {
            KQuantScales::Unsigned(values) => {
                MxArray::from_uint8(values, &[rows, format.scales_cols(hidden as usize) as i64])?
            }
            KQuantScales::Signed(_) => panic!("Q4_K scales must be unsigned"),
        };
        let quant_biases = MxArray::from_float16(
            &packed.biases,
            &[rows, format.biases_cols(hidden as usize) as i64],
        )?;
        let additive_values = (0..rows)
            .map(|i| ((i as f32) * 0.031 - 0.2).cos() * 0.1)
            .collect::<Vec<_>>();
        let additive_bias =
            MxArray::from_float32(&additive_values, &[rows])?.astype(DType::BFloat16)?;

        let make_q_proj = || {
            QuantizedLinear::new(
                weight.clone(),
                scales.clone(),
                Some(quant_biases.clone()),
                Some(additive_bias.clone()),
                32,
                4,
                "q4k".to_string(),
            )
        };
        let mut block = Qwen3_5Attention::new(&cfg)?;
        let mut native = Qwen3_5Attention::new(&cfg)?;
        block.set_quantized_q_proj(make_q_proj());
        native.set_quantized_q_proj(make_q_proj());
        block.finalize_q_gate_block()?;
        // The independent reference keeps its original unfinalized layout.
        assert!(block.q_proj.has_q_gate_block_layout());
        assert!(!native.q_proj.has_q_gate_block_layout());
        assert!(block.q_gate_block_t.is_none());

        let x_values = (0..2 * hidden)
            .map(|i| ((i as f32) * 0.023 + 0.7).sin())
            .collect::<Vec<_>>();
        let x = MxArray::from_float32(&x_values, &[1, 2, hidden])?.astype(DType::BFloat16)?;
        let (block_q, block_gate) = block.project_q_gate(&x, 1, 2)?;
        let (native_q, native_gate) = native.project_q_gate(&x, 1, 2)?;
        MxArray::eval_arrays(&[&block_q, &block_gate, &native_q, &native_gate])?;
        assert_eq!(
            block_q.to_uint16_native()?,
            native_q.to_uint16_native()?,
            "block-order Q4_K queries must be bit-identical to native per-head splitting"
        );
        assert_eq!(
            block_gate.to_uint16_native()?,
            native_gate.to_uint16_native()?,
            "block-order Q4_K gates must be bit-identical to native per-head splitting"
        );
        Ok(())
    }

    #[test]
    fn source_mixer_attention_fallback_preserves_bf16_kv_frontier() -> Result<()> {
        let cfg = tiny_cfg();
        let mut attention = Qwen3_5Attention::new(&cfg)?;
        let bf = |shape: &[i64], seed: usize| -> Result<MxArray> {
            let n = shape.iter().product::<i64>() as usize;
            MxArray::from_float32(
                &(0..n)
                    .map(|i| ((i * 7 + seed) % 37) as f32 * 0.015625 - 0.25)
                    .collect::<Vec<_>>(),
                shape,
            )?
            .astype(DType::BFloat16)
        };
        attention.set_q_proj_weight(&bf(&[64, 32], 3)?)?;
        attention.set_k_proj_weight(&bf(&[16, 32], 7)?)?;
        attention.set_v_proj_weight(&bf(&[16, 32], 11)?)?;
        attention.set_o_proj_weight(&bf(&[32, 32], 17)?)?;
        let norm = MxArray::ones(&[8], Some(DType::BFloat16))?;
        attention.set_q_norm_weight(&norm, DType::BFloat16)?;
        attention.set_k_norm_weight(&norm, DType::BFloat16)?;
        let mut reference = KVCache::new();
        let mut fused = KVCache::new();
        for rows in [1, 7, 8, 33, 8] {
            let x = bf(&[1, rows, 32], rows as usize)?;
            let residual = bf(&[1, rows, 32], rows as usize + 19)?;
            let delta = attention.forward(&x, None, Some(&mut reference), None)?;
            let hidden =
                attention.forward_source_residual(&x, None, Some(&mut fused), None, &residual)?;
            assert_eq!(
                hidden.to_uint16_native()?,
                residual.add(&delta)?.to_uint16_native()?
            );
            assert_eq!(reference.get_offset(), fused.get_offset());
            for (a, b) in [
                (reference.keys_ref().unwrap(), fused.keys_ref().unwrap()),
                (reference.values_ref().unwrap(), fused.values_ref().unwrap()),
            ] {
                // Only the committed prefix is initialized in spare-capacity buffers.
                let end = reference.get_offset() as i64;
                assert_eq!(
                    a.slice_axis(2, 0, end)?.to_uint16_native()?,
                    b.slice_axis(2, 0, end)?.to_uint16_native()?
                );
            }
        }
        Ok(())
    }

    #[test]
    fn source_qk_operands_preserve_native_fallback_and_eager_kv_frontier() -> Result<()> {
        if !unsafe { mlx_sys::mlx_source_qk_norm_rope_available() } {
            return Ok(());
        }
        let mut cfg = tiny_cfg();
        cfg.head_dim = 256;
        cfg.num_heads = 16;
        cfg.num_kv_heads = 2;
        cfg.partial_rotary_factor = 0.25;
        cfg.rope_theta = 10_000_000.0;
        cfg.rms_norm_eps = 1e-6;
        let mut attn = Qwen3_5Attention::new(&cfg)?;
        let bf = |shape: &[i64], seed: usize| -> Result<MxArray> {
            let count = shape.iter().product::<i64>() as usize;
            MxArray::from_float32(
                &(0..count)
                    .map(|i| (((i * 7 + seed) % 41) as f32 - 20.0) * 0.0078125)
                    .collect::<Vec<_>>(),
                shape,
            )?
            .astype(DType::BFloat16)
        };
        attn.set_q_proj_weight(&bf(&[8192, 32], 3)?)?;
        attn.set_k_proj_weight(&bf(&[512, 32], 5)?)?;
        attn.set_v_proj_weight(&bf(&[512, 32], 7)?)?;
        attn.set_o_proj_weight(&bf(&[32, 4096], 11)?)?;
        let source_q = MxArray::from_float32(
            &(0..256)
                .map(|i| 1.0 + (i % 17) as f32 * 0.00031)
                .collect::<Vec<_>>(),
            &[256],
        )?;
        let source_k = MxArray::from_float32(
            &(0..256)
                .map(|i| 1.0 + (i % 23) as f32 * 0.00037)
                .collect::<Vec<_>>(),
            &[256],
        )?;
        let frequencies = MxArray::from_float32(
            &(0..32)
                .map(|i| 10_000_000_f32.powf(-(i as f32) / 32.0))
                .collect::<Vec<_>>(),
            &[32],
        )?;
        attn.set_q_norm_weight(&source_q, DType::BFloat16)?;
        attn.set_k_norm_weight(&source_k, DType::BFloat16)?;
        let original_q = attn.get_q_norm_weight().to_uint16_native()?;
        let original_k = attn.get_k_norm_weight().to_uint16_native()?;
        attn.validate_source_qk_weights(&source_q, &source_k, &frequencies)?;
        let mut native_cache = KVCache::new();
        let mut selected_cache = KVCache::new();
        for rows in [1, 7, 33] {
            let x = bf(&[1, rows, 32], rows as usize)?;
            attn.source_qk = None;
            let native = attn.forward(&x, None, Some(&mut native_cache), None)?;
            attn.install_source_qk_weights(source_q.clone(), source_k.clone(), frequencies.clone());
            let selected = attn.forward(&x, None, Some(&mut selected_cache), None)?;
            assert_eq!(native.to_uint16_native()?, selected.to_uint16_native()?);
            assert_eq!(native_cache.get_offset(), selected_cache.get_offset());
            let end = selected_cache.get_offset() as i64;
            assert_eq!(
                native_cache
                    .keys_ref()
                    .unwrap()
                    .slice_axis(2, 0, end)?
                    .to_uint16_native()?,
                selected_cache
                    .keys_ref()
                    .unwrap()
                    .slice_axis(2, 0, end)?
                    .to_uint16_native()?
            );
        }
        assert_eq!(attn.get_q_norm_weight().to_uint16_native()?, original_q);
        assert_eq!(attn.get_k_norm_weight().to_uint16_native()?, original_k);
        assert!(attn.set_q_norm_weight(&source_q, DType::BFloat16).is_err());
        assert!(attn.set_k_norm_weight(&source_k, DType::BFloat16).is_err());
        assert!(attn.source_qk.is_some());
        let x = bf(&[1, 8, 32], 29)?;
        let before = selected_cache.get_offset();
        let (queries, _, keys, _) = attn.project_mixer(&x, 1, 8)?;
        let keys = keys.reshape(&[1, 8, 2, 256])?;
        let offsets = MxArray::from_int32(&[before], &[1])?;
        let (_, expected_keys) = attn
            .try_source_qk_norm_rope(&queries, &keys, &offsets)?
            .expect("strict T8 must select source QK");
        let invalid = offsets.astype(DType::Float32)?;
        assert!(
            attn.try_source_qk_norm_rope(&queries, &keys, &invalid)
                .is_err()
        );
        let result = attn.forward(&x, None, Some(&mut selected_cache), None)?;
        result.eval();
        assert_eq!(selected_cache.get_offset(), before + 8);
        assert_eq!(
            selected_cache
                .keys_ref()
                .unwrap()
                .slice_axis(2, before as i64, (before + 8) as i64)?
                .to_uint16_native()?,
            expected_keys.to_uint16_native()?
        );
        assert_eq!(attn.get_q_norm_weight().to_uint16_native()?, original_q);
        Ok(())
    }

    /// The 27B target attention geometry for the YaRN gates.
    fn yarn_cfg(max_context: Option<u32>) -> Qwen3_5Config {
        let mut cfg = tiny_cfg();
        cfg.head_dim = 256;
        cfg.num_heads = 16;
        cfg.num_kv_heads = 2;
        cfg.partial_rotary_factor = 0.25;
        cfg.rope_theta = 10_000_000.0;
        cfg.max_position_embeddings = 262_144;
        cfg.rope_scaling = Some(
            RopeScaling::served_for("Qwen3.5", 262_144, RopeScaling::default(), max_context)
                .unwrap(),
        );
        cfg
    }

    fn bits_of(array: &MxArray) -> Vec<u32> {
        array
            .astype(DType::Float32)
            .unwrap()
            .to_float32()
            .unwrap()
            .iter()
            .map(|value| value.to_bits())
            .collect()
    }

    /// Gate: at or below the native window the attention keeps today's RoPE
    /// kernel. Mutation: `<` at the native window in
    /// `served_for` → red (262,144 turns YaRN on).
    #[test]
    fn below_native_attention_is_bit_equal() -> Result<()> {
        let x = MxArray::random_normal(&[1, 16, 3, 256], 0.0, 1.0, Some(DType::BFloat16))?;
        for max_context in [None, Some(102_400), Some(262_144)] {
            let cfg = yarn_cfg(max_context);
            assert_eq!(cfg.scaling(), RopeScaling::default(), "{max_context:?}");
            let mut attn = Qwen3_5Attention::new(&cfg)?;
            attn.q_norm
                .set_weight(&MxArray::ones(&[256], Some(DType::BFloat16))?)?;
            attn.k_norm
                .set_weight(&MxArray::ones(&[256], Some(DType::BFloat16))?)?;
            assert!(!attn.rope.is_scaled(), "{max_context:?}");
            let direct = MxArray::from_handle(
                unsafe { mlx_sys::mlx_fast_rope(x.handle.0, 64, false, 1e7, 1.0, 7) },
                "rope",
            )?;
            assert_eq!(bits_of(&attn.rope.forward(&x, Some(7))?), bits_of(&direct));
        }
        Ok(())
    }

    /// Under YaRN the source QK diagnostic (an unscaled table) steps aside.
    #[test]
    fn fused_kernels_step_aside_under_yarn() -> Result<()> {
        let mut scaled = Qwen3_5Attention::new(&yarn_cfg(Some(409_600)))?;
        scaled
            .q_norm
            .set_weight(&MxArray::ones(&[256], Some(DType::BFloat16))?)?;
        scaled
            .k_norm
            .set_weight(&MxArray::ones(&[256], Some(DType::BFloat16))?)?;
        let plain = Qwen3_5Attention::new(&yarn_cfg(None))?;
        assert!(scaled.rope.is_scaled());
        let weight = MxArray::from_float32(&[1.0; 256], &[256])?;
        let frequencies = MxArray::from_float32(&[1.0; 32], &[32])?;
        assert!(
            plain
                .validate_source_qk_weights(&weight, &weight, &frequencies)
                .is_ok()
        );
        assert!(
            scaled
                .validate_source_qk_weights(&weight, &weight, &frequencies)
                .is_err()
        );
        Ok(())
    }

    /// Image rows take the target's YaRN table and factor on every M-RoPE
    /// axis (Splash `ROPE_BUILD_TABLES`); without YaRN `init_mrope` is
    /// `MultimodalRoPE::new` bit for bit. Mutation: `init_mrope` ignores
    /// the scaling → red.
    #[test]
    fn mrope_takes_the_yarn_table() -> Result<()> {
        let section = vec![11, 11, 10];
        let positions: Vec<i32> = (0..3).flat_map(|_| [0, 3, 4093]).collect();
        let position_ids = MxArray::from_int32(&positions, &[3, 1, 3])?;
        let x = MxArray::zeros(&[1, 1, 3, 64], Some(DType::Float32))?;

        let scaling = yarn_cfg(Some(409_600)).scaling();
        let mut attn = Qwen3_5Attention::new(&yarn_cfg(Some(409_600)))?;
        attn.init_mrope(section.clone(), 1e7, 262_144, 64, &scaling)?;
        let (cos, sin) = attn.mrope.as_ref().unwrap().forward(&x, &position_ids)?;
        let inverse = rotary_inverse_frequencies(1e7, 32, &scaling)?;
        let factor = scaling.attention_factor();
        let reference =
            MultimodalRoPE::with_inverse_frequencies(64, section.clone(), &inverse, factor)?;
        let (want_cos, want_sin) = reference.forward(&x, &position_ids)?;
        assert_eq!(bits_of(&cos), bits_of(&want_cos));
        assert_eq!(bits_of(&sin), bits_of(&want_sin));
        // The f32 matmul of the angles is not exact on every GPU: the
        // small positions only, at its precision.
        let (cos, sin) = (cos.to_float32()?, sin.to_float32()?);
        assert_eq!(cos.len(), 3 * 3 * 64);
        for axis in 0..3 {
            for (t, position) in [0f32, 3.0].into_iter().enumerate() {
                for dim in 0..64 {
                    let angle = f64::from(inverse[dim % 32] * position);
                    let index = (axis * 3 + t) * 64 + dim;
                    let factor = f64::from(factor);
                    for (got, want) in [
                        (cos[index], angle.cos() * factor),
                        (sin[index], angle.sin() * factor),
                    ] {
                        assert!(
                            (f64::from(got) - want).abs() <= 3e-3,
                            "axis {axis} t {t} dim {dim}: {got} vs {want}"
                        );
                    }
                }
            }
        }

        let mut plain = Qwen3_5Attention::new(&yarn_cfg(None))?;
        plain.init_mrope(section.clone(), 1e7, 262_144, 64, &RopeScaling::default())?;
        let reference = MultimodalRoPE::new(64, 262_144, Some(1e7), section)?;
        let (a_cos, a_sin) = plain.mrope.as_ref().unwrap().forward(&x, &position_ids)?;
        let (b_cos, b_sin) = reference.forward(&x, &position_ids)?;
        assert_eq!(bits_of(&a_cos), bits_of(&b_cos));
        assert_eq!(bits_of(&a_sin), bits_of(&b_sin));
        Ok(())
    }

    /// `n` values uniform in ±1 from `seed`; with `outlier`, dimension
    /// `(row · 37) % dim` of each `dim`-wide row times 60.
    fn paged_attend_values(n: usize, dim: usize, seed: u32, outlier: bool) -> Vec<f32> {
        let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(11);
        (0..n)
            .map(|i| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let value = ((state >> 8) % 2001) as f32 / 1000.0 - 1.0;
                let row = i / dim;
                if outlier && i % dim == (row * 37) % dim {
                    value * 60.0
                } else {
                    value
                }
            })
            .collect()
    }

    fn bf16_array(values: &[f32], shape: &[i64]) -> MxArray {
        MxArray::from_float32(values, shape)
            .and_then(|a| a.astype(DType::BFloat16))
            .expect("bf16 array")
    }

    fn f32_values(array: &MxArray) -> Vec<f32> {
        array
            .astype(DType::Float32)
            .and_then(|a| a.to_float32())
            .expect("f32 values")
            .to_vec()
    }

    /// Layer 0's first `n` stored rows of `pool` over the block `ids` as f64
    /// `[Hkv][n][D]` (an int8 row times its scale, exactly).
    #[cfg(target_os = "macos")]
    fn stored_rows(
        pool: &mlx_paged_attn::LayerKVPool,
        ids: &[u32],
        n: usize,
        hkv: usize,
        d: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let read = |raw: std::result::Result<*mut mlx_sys::mlx_array, String>| {
            f32_values(&MxArray::from_handle(raw.expect("pool plane"), "pool plane").unwrap())
        };
        let int8 = pool.cache_dtype() == mlx_paged_attn::metal::MetalDtype::Int8;
        let (x, bs) = (if int8 { 16 } else { 8 }, pool.config().block_size as usize);
        let k = read(pool.key_cache_array_raw(0));
        let v = read(pool.value_cache_array_raw(0));
        let (ks, vs) = if int8 {
            (
                read(pool.key_scale_array_raw(0)),
                read(pool.value_scale_array_raw(0)),
            )
        } else {
            (Vec::new(), Vec::new())
        };
        let mut keys = vec![0f64; hkv * n * d];
        let mut values = vec![0f64; hkv * n * d];
        for h in 0..hkv {
            for t in 0..n {
                let block = ids[t / bs] as usize;
                let slot = t % bs;
                let scale = |plane: &[f32]| {
                    if int8 {
                        f64::from(plane[(block * hkv + h) * bs + slot])
                    } else {
                        1.0
                    }
                };
                for dim in 0..d {
                    let ki = (((block * hkv + h) * (d / x) + dim / x) * bs + slot) * x + dim % x;
                    let vi = ((block * hkv + h) * d + dim) * bs + slot;
                    keys[(h * n + t) * d + dim] = f64::from(k[ki]) * scale(&ks);
                    values[(h * n + t) * d + dim] = f64::from(v[vi]) * scale(&vs);
                }
            }
        }
        (keys, values)
    }

    /// The worst error of `got` `[rows, Hq, D]` over Splash's `checkOutput`
    /// bound (`mass[d] / 256 + ulp_bf16(ref)`) against an f64 softmax of
    /// `q` at `committed` over the stored rows `keys` / `values`
    /// `[Hkv][visible][D]`.
    #[cfg(target_os = "macos")]
    #[allow(clippy::too_many_arguments)]
    fn worst_of_bound(
        got: &[f32],
        q: &[f32],
        keys: &[f64],
        values: &[f64],
        committed: usize,
        rows: usize,
        (hq, hkv, d): (usize, usize, usize),
        scale: f32,
    ) -> f64 {
        let visible = committed + rows;
        let mut worst = 0f64;
        for row in 0..rows {
            let end = committed + row + 1;
            for head in 0..hq {
                let kv = head / (hq / hkv);
                let query = &q[(row * hq + head) * d..][..d];
                let scores: Vec<f64> = (0..end)
                    .map(|t| {
                        let key = &keys[(kv * visible + t) * d..][..d];
                        query
                            .iter()
                            .zip(key)
                            .map(|(a, b)| f64::from(*a) * b)
                            .sum::<f64>()
                            * f64::from(scale)
                    })
                    .collect();
                let max = scores.iter().cloned().fold(f64::MIN, f64::max);
                let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                let sum: f64 = weights.iter().sum();
                for dim in 0..d {
                    let (mut out, mut mass) = (0f64, 0f64);
                    for (t, w) in weights.iter().enumerate() {
                        let value = values[(kv * visible + t) * d + dim];
                        out += w / sum * value;
                        mass += w / sum * value.abs();
                    }
                    let ulp = f64::from(bf16_ulp(out as f32));
                    let error = (f64::from(got[(row * hq + head) * d + dim]) - out).abs();
                    worst = worst.max(error / (mass / 256.0 + ulp));
                }
            }
        }
        worst
    }

    /// Store `rows` rows at `committed` into `adapter`'s pool and attend
    /// them in `pass`; returns the bf16 queries and the output as f32.
    #[cfg(target_os = "macos")]
    fn store_attend_chunk(
        adapter: &mut PagedKVCacheAdapter,
        (committed, rows): (usize, usize),
        pass: PagedPass,
        (hq, hkv, d): (usize, usize, usize),
        seed: u32,
    ) -> (Vec<f32>, Vec<f32>) {
        adapter
            .record_tokens(&(committed as u32..(committed + rows) as u32).collect::<Vec<_>>())
            .unwrap();
        let q = paged_attend_values(rows * hq * d, d, seed + 1, false);
        let k = paged_attend_values(rows * hkv * d, d, seed + 2, true);
        let v = paged_attend_values(rows * hkv * d, d, seed + 3, true);
        let queries = bf16_array(&q, &[1, rows as i64, hq as i64, d as i64]);
        let keys = bf16_array(&k, &[1, rows as i64, hkv as i64, d as i64]);
        let values = bf16_array(&v, &[1, rows as i64, hkv as i64, d as i64]);
        let got = paged_store_attend(
            adapter,
            0,
            committed as u32,
            pass,
            &queries,
            &keys,
            &values,
            1.0 / (d as f32).sqrt(),
            DType::BFloat16,
        )
        .unwrap();
        (f32_values(&queries), f32_values(&got))
    }

    /// A paged prefill chunk stores its rows, then attends over the pool
    /// pages: every row, the chunk's own included, reads back as stored
    /// (int8 · scale on an int8 pool; Splash `addPrefillStore` then
    /// `addPrefill`). 24 / 4 heads, D256, 16-token blocks: a cold 40-row
    /// chunk at 0 and a 24-row chunk at 40, against an f64 softmax over
    /// `read_kv_range` within Splash's `checkOutput` bound (`mass[d] / 256 +
    /// ulp_bf16(ref)`). K / V rows carry one ×60 dimension each, so their
    /// int8 rows differ clearly from bf16. Mutation O1: a prefill chunk at
    /// position 0 attends its in-flight bf16 rows (MLX causal SDPA) → red
    /// (int8, P = 0).
    #[cfg(target_os = "macos")]
    #[test]
    fn paged_attention_reads_its_own_rows_back() {
        if crate::test_support::paged_prefill_unsupported("paged_attention_reads_its_own_rows_back")
        {
            return;
        }
        use crate::transformer::paged_int8_tests::adapter_for;
        use mlx_paged_attn::metal::MetalDtype;
        const HQ: usize = 24;
        const HKV: usize = 4;
        const D: usize = 256;
        let config = mlx_paged_attn::PagedAttentionConfig {
            block_size: 16,
            gpu_memory_mb: 256,
            head_size: D as u32,
            num_kv_heads: HKV as u32,
            num_layers: 1,
            max_seq_len: Some(1024),
            max_batch_size: Some(4),
        };
        let scale = 1.0 / (D as f32).sqrt();
        for dtype in [MetalDtype::Int8, MetalDtype::BFloat16] {
            let Some((mut adapter, _pool)) = adapter_for(config.clone(), 8, 8, dtype) else {
                return;
            };
            adapter.reset_for_new_request(1).unwrap();
            adapter.allocate_suffix_blocks(64).unwrap();
            for (seed, (committed, rows)) in [(0usize, 40usize), (40, 24)].into_iter().enumerate() {
                let (q, got) = store_attend_chunk(
                    &mut adapter,
                    (committed, rows),
                    PagedPass::Prefill,
                    (HQ, HKV, D),
                    seed as u32 * 10,
                );
                let ids = adapter.block_table().expect("table").block_ids();
                let (k_pool, v_pool) = stored_rows(&_pool, &ids, committed + rows, HKV, D);
                let worst = worst_of_bound(
                    &got,
                    &q,
                    &k_pool,
                    &v_pool,
                    committed,
                    rows,
                    (HQ, HKV, D),
                    scale,
                );
                assert!(
                    worst <= 1.0,
                    "{dtype:?} P {committed} T {rows}: worst error {worst:.3} of the bound"
                );
            }
        }
    }

    /// A GPU that cannot build the tensor-op prefill pipelines attends
    /// through the varlen kernel: the same stored rows, within the same
    /// `checkOutput` bound as the tile kernel, on int8 and bf16 pages, for
    /// groups 4 / 8 / 16 at D128 / D256, a cold chunk, a cache-hit chunk, a
    /// chunk past 512 tokens of context (the varlen kernel's partitioned
    /// reduce) and a verify block. The tile half runs where its pipelines
    /// build. Mutation PF2: the fallback attends at `committed - 1` → red.
    #[cfg(target_os = "macos")]
    #[test]
    fn portable_prefill_matches_the_tile_kernel() {
        use crate::transformer::paged_int8_tests::adapter_for;
        use mlx_paged_attn::metal::MetalDtype;
        let tile_builds = crate::transformer::paged_kv_cache_adapter::device_prefill_mpp_builds();
        // (committed, rows, pass, checked); the unchecked chunk only fills
        // the history the next chunks read.
        let chunks = [
            (0usize, 40usize, PagedPass::Prefill, true),
            (40, 24, PagedPass::Prefill, true),
            (64, 536, PagedPass::Prefill, false),
            (600, 9, PagedPass::Prefill, true),
            (609, 5, PagedPass::Verify, true),
        ];
        let mut largest_gap = 0f32;
        for (hq, hkv, d) in [(8usize, 2usize, 128usize), (16, 2, 256), (32, 2, 256)] {
            let config = mlx_paged_attn::PagedAttentionConfig {
                block_size: 16,
                gpu_memory_mb: 256,
                head_size: d as u32,
                num_kv_heads: hkv as u32,
                num_layers: 1,
                max_seq_len: Some(1024),
                max_batch_size: Some(4),
            };
            let scale = 1.0 / (d as f32).sqrt();
            for dtype in [MetalDtype::Int8, MetalDtype::BFloat16] {
                let mut lanes = Vec::new();
                for portable in [true, false] {
                    if !portable && !tile_builds {
                        continue;
                    }
                    let Some((mut adapter, pool)) = adapter_for(config.clone(), 48, 48, dtype)
                    else {
                        return;
                    };
                    if portable {
                        adapter.force_portable_prefill();
                    }
                    adapter.reset_for_new_request(1).unwrap();
                    adapter.allocate_suffix_blocks(614).unwrap();
                    lanes.push((portable, adapter, pool));
                }
                for (seed, &(committed, rows, pass, checked)) in chunks.iter().enumerate() {
                    let mut outputs = Vec::new();
                    for (portable, adapter, pool) in lanes.iter_mut() {
                        let (q, got) = store_attend_chunk(
                            adapter,
                            (committed, rows),
                            pass,
                            (hq, hkv, d),
                            seed as u32 * 10,
                        );
                        if !checked {
                            continue;
                        }
                        let ids = adapter.block_table().expect("table").block_ids();
                        let (keys, values) = stored_rows(pool, &ids, committed + rows, hkv, d);
                        let worst = worst_of_bound(
                            &got,
                            &q,
                            &keys,
                            &values,
                            committed,
                            rows,
                            (hq, hkv, d),
                            scale,
                        );
                        assert!(
                            worst <= 1.0,
                            "{dtype:?} {hq}/{hkv} D{d} P {committed} T {rows} {pass:?} \
                             (portable {portable}): worst error {worst:.3} of the bound"
                        );
                        outputs.push(got);
                    }
                    if let [portable, tile] = outputs.as_slice() {
                        let gap = portable
                            .iter()
                            .zip(tile)
                            .map(|(a, b)| (a - b).abs())
                            .fold(0f32, f32::max);
                        largest_gap = largest_gap.max(gap);
                    }
                }
            }
        }
        eprintln!(
            "portable vs tile prefill: largest |difference| {largest_gap} \
             (tile pipelines build: {tile_builds})"
        );
    }

    /// On a bf16 pool the portable prefill runs its rows in chunks of at
    /// most 2,048 row-partitions, the bound the varlen kernel tiles an int8
    /// pool to: 1,400 rows at 100 over 3 partitions run as 682 + 682 + 36,
    /// every row bit-equal to one unchunked launch. Mutations PT1 (a later
    /// piece attends at `committed + start - 1`) and PT2 (one launch) → red.
    #[cfg(target_os = "macos")]
    #[test]
    fn portable_prefill_chunks_a_bf16_pool_to_the_tile_bound() {
        use crate::transformer::paged_int8_tests::adapter_for;
        use mlx_paged_attn::metal::MetalDtype;
        const HQ: usize = 16;
        const HKV: usize = 2;
        const D: usize = 256;
        let config = mlx_paged_attn::PagedAttentionConfig {
            block_size: 16,
            gpu_memory_mb: 256,
            head_size: D as u32,
            num_kv_heads: HKV as u32,
            num_layers: 1,
            max_seq_len: Some(2048),
            max_batch_size: Some(4),
        };
        let scale = 1.0 / (D as f32).sqrt();
        let Some((mut adapter, _pool)) = adapter_for(config, 112, 112, MetalDtype::BFloat16) else {
            return;
        };
        adapter.force_portable_prefill();
        adapter.reset_for_new_request(1).unwrap();
        adapter.allocate_suffix_blocks(1500).unwrap();
        for (seed, (committed, rows)) in [(0usize, 100usize), (100, 1400)].into_iter().enumerate() {
            adapter.portable_prefill_launches.clear();
            let (q, got) = store_attend_chunk(
                &mut adapter,
                (committed, rows),
                PagedPass::Prefill,
                (HQ, HKV, D),
                seed as u32 * 10,
            );
            for &(launch_rows, context) in &adapter.portable_prefill_launches {
                assert!(
                    u64::from(launch_rows) * u64::from(context).div_ceil(512) <= 2048,
                    "P {committed} T {rows}: a launch of {launch_rows} rows over {context} \
                     tokens holds more than 2,048 row-partitions"
                );
            }
            if rows == 1400 {
                assert_eq!(
                    adapter.portable_prefill_launches,
                    [(682, 782), (682, 1464), (36, 1500)],
                    "the 1,400-row chunk's launches"
                );
            }
            let queries = bf16_array(&q, &[rows as i64, HQ as i64, D as i64]);
            let whole = adapter
                .gather_kv_for_prefill_chunk_varlen(0, &queries, committed as u32, scale)
                .unwrap();
            let whole = f32_values(&whole);
            let differing = got
                .iter()
                .zip(&whole)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(
                differing, 0,
                "P {committed} T {rows}: chunked values differ from one launch"
            );
        }
    }

    /// Every piece a bf16 portable prefill launches passes the varlen
    /// preflight and holds at most the tile kernel's priced workspace with a
    /// second copy of the chunk's output (the concatenate): 2,048-row chunks
    /// at histories up to 1,048,576 tokens (2,048 partitions), at 24 / 4 and
    /// 16 / 2 heads, D256. One launch of the chunk fails the preflight from
    /// 87,041 (24 heads) and 130,561 (16 heads) tokens. Mutation PT2 (one
    /// launch) → red.
    #[cfg(target_os = "macos")]
    #[test]
    fn portable_prefill_pieces_fit_the_priced_workspace() {
        use crate::transformer::paged_kv_cache_adapter::{
            PagedAttentionV2Layout, paged_attention_v2_aux_fits,
            paged_attention_v2_partition_upper_bound, portable_prefill_pieces,
        };
        use mlx_paged_attn::prefill_plan::{Pass, attention_bytes, workspace_bound_floats};
        const D: u32 = 256;
        const ROWS: u32 = 2048;
        const MOST: u32 = 1_048_576;
        let fits = |rows, hq, hkv, context| {
            paged_attention_v2_aux_fits(PagedAttentionV2Layout::Varlen, rows, hq, hkv, context, D)
        };
        for (hq, hkv, first_refused) in [(24u32, 4u32, 87_041u32), (16, 2, 130_561)] {
            assert!(fits(ROWS, hq, hkv, first_refused - 1));
            assert!(!fits(ROWS, hq, hkv, first_refused));
            let workspace = workspace_bound_floats(Pass::Prefill, hq, D) * 4;
            let priced = attention_bytes(ROWS, hq, hkv, D, Pass::Prefill);
            let duplicate_output = u64::from(ROWS) * u64::from(hq) * u64::from(D) * 2;
            let mut largest = 0u64;
            let edges = [ROWS, first_refused, 262_144, MOST];
            for total in (ROWS..=MOST).step_by(1021).chain(edges) {
                let committed = total - ROWS;
                let pieces = portable_prefill_pieces(committed, ROWS);
                assert_eq!(pieces.first().map(|p| p.0), Some(0));
                assert_eq!(pieces.last().map(|p| p.1), Some(ROWS));
                assert!(pieces.windows(2).all(|w| w[0].1 == w[1].0));
                for &(start, end) in &pieces {
                    let (rows, context) = (end - start, committed + end);
                    assert!(
                        fits(rows, hq, hkv, context),
                        "{hq}/{hkv} at {total}: piece {start}..{end} fails the preflight"
                    );
                    let partitions = paged_attention_v2_partition_upper_bound(
                        PagedAttentionV2Layout::Varlen,
                        rows,
                        hq,
                        hkv,
                        context,
                        D,
                    );
                    // bf16 partials, f32 exp sums and max logits.
                    let aux = u64::from(rows) * u64::from(hq) * partitions * (2 * u64::from(D) + 8);
                    largest = largest.max(aux);
                    assert!(
                        aux + duplicate_output <= workspace,
                        "{hq}/{hkv} at {total}: piece {start}..{end} holds {aux} B of partials"
                    );
                }
            }
            eprintln!(
                "{hq}/{hkv} D{D}: largest piece partials {largest} B, with the output copy {} B, \
                 workspace bound {workspace} B, attention_bytes {priced} B",
                largest + duplicate_output
            );
            assert!(largest + duplicate_output <= priced);
        }
    }

    /// A bf16 verify block of 8 rows at 131,072 tokens runs as 7 + 1 rows on
    /// the portable path. The 1-row piece is a view 14,336 bytes into a
    /// 16 KiB query buffer, which `contiguous` aliases rather than copies,
    /// and the varlen primitive refuses a nonzero offset. Every row must
    /// match one launch of the block bit for bit. Mutation F3 (no copy per
    /// piece) → the second piece is refused → red.
    #[cfg(target_os = "macos")]
    #[test]
    fn portable_verify_pieces_at_long_context_attend() {
        use crate::transformer::paged_int8_tests::adapter_for;
        use mlx_paged_attn::metal::MetalDtype;
        const HQ: usize = 8;
        const HKV: usize = 2;
        const D: usize = 128;
        const COMMITTED: usize = 131_072;
        const ROWS: usize = 8;
        let config = mlx_paged_attn::PagedAttentionConfig {
            block_size: 16,
            gpu_memory_mb: 256,
            head_size: D as u32,
            num_kv_heads: HKV as u32,
            num_layers: 1,
            max_seq_len: Some((COMMITTED + ROWS) as u32),
            max_batch_size: Some(1),
        };
        let blocks = ((COMMITTED + ROWS) / 16 + 1) as u32;
        let Some((mut adapter, _pool)) = adapter_for(config, blocks, blocks, MetalDtype::BFloat16)
        else {
            return;
        };
        adapter.force_portable_prefill();
        adapter.reset_for_new_request(1).unwrap();
        adapter
            .allocate_suffix_blocks((COMMITTED + ROWS) as u32)
            .unwrap();
        adapter
            .record_tokens(&(0..COMMITTED as u32).map(|t| t % 1000).collect::<Vec<_>>())
            .unwrap();
        let (q, got) = store_attend_chunk(
            &mut adapter,
            (COMMITTED, ROWS),
            PagedPass::Verify,
            (HQ, HKV, D),
            70,
        );
        assert_eq!(
            adapter.portable_prefill_launches,
            [(7, 131_079), (1, 131_080)],
            "the verify block's launches"
        );
        let queries = bf16_array(&q, &[ROWS as i64, HQ as i64, D as i64]);
        let whole = adapter
            .gather_kv_for_prefill_chunk_varlen(
                0,
                &queries,
                COMMITTED as u32,
                1.0 / (D as f32).sqrt(),
            )
            .unwrap();
        let whole = f32_values(&whole);
        let differing = got
            .iter()
            .zip(&whole)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(differing, 0, "the pieces differ from one launch");
    }

    /// The bf16 portable pieces of one chunk never hold their partials at
    /// the same time: 1,024 rows at 15,360 tokens run as 16 pieces of 64 rows
    /// over 32 partitions, each with 17,039,360 bytes of partials. MLX's peak
    /// over the call stays under two pieces' partials plus three copies of
    /// the output (pieces, concatenate, slack), where all 16 at once would
    /// hold 272 MB. Mutation F2 (no evaluation between pieces) → red.
    #[cfg(target_os = "macos")]
    #[test]
    fn portable_prefill_pieces_do_not_hold_their_partials_together() {
        use crate::transformer::paged_int8_tests::adapter_for;
        use mlx_paged_attn::metal::{MetalDtype, synchronize_mlx};
        const HQ: usize = 16;
        const HKV: usize = 2;
        const D: usize = 256;
        const COMMITTED: usize = 15_360;
        const ROWS: usize = 1024;
        let config = mlx_paged_attn::PagedAttentionConfig {
            block_size: 16,
            gpu_memory_mb: 256,
            head_size: D as u32,
            num_kv_heads: HKV as u32,
            num_layers: 1,
            max_seq_len: Some((COMMITTED + ROWS) as u32),
            max_batch_size: Some(1),
        };
        let blocks = ((COMMITTED + ROWS) / 16 + 1) as u32;
        let Some((mut adapter, _pool)) = adapter_for(config, blocks, blocks, MetalDtype::BFloat16)
        else {
            return;
        };
        adapter.force_portable_prefill();
        adapter.reset_for_new_request(1).unwrap();
        adapter
            .allocate_suffix_blocks((COMMITTED + ROWS) as u32)
            .unwrap();
        adapter
            .record_tokens(
                &(0..(COMMITTED + ROWS) as u32)
                    .map(|t| t % 1000)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let q = paged_attend_values(ROWS * HQ * D, D, 90, false);
        let queries = bf16_array(&q, &[ROWS as i64, HQ as i64, D as i64]);
        MxArray::eval_arrays(&[&queries]).unwrap();
        synchronize_mlx();
        let before = crate::array::get_active_memory();
        crate::array::reset_peak_memory();
        let out = adapter
            .paged_prefill_attention(
                0,
                &queries,
                COMMITTED as u32,
                mlx_paged_attn::prefill_plan::Pass::Prefill,
                1.0 / (D as f32).sqrt(),
            )
            .unwrap();
        MxArray::eval_arrays(&[&out]).unwrap();
        synchronize_mlx();
        let grew = crate::array::get_peak_memory() - before;
        assert_eq!(adapter.portable_prefill_launches.len(), 16);
        let partials = (64 * HQ * 32 * (2 * D + 8)) as f64;
        let output = (ROWS * HQ * D * 2) as f64;
        eprintln!(
            "portable prefill pieces: peak grew {grew} B; one piece's partials {partials} B, \
             the output {output} B"
        );
        assert!(
            grew <= 2.0 * partials + 3.0 * output,
            "peak grew {grew} B: the pieces' partials overlap"
        );
    }

    /// The device answer is made once per process, and a GPU that builds
    /// the tensor-op pipelines (every Apple GPU family) launches them for
    /// every prefill chunk. Mutations PF1 (the fallback always taken) and
    /// PF3 (the probe asked at every chunk) → red.
    #[cfg(target_os = "macos")]
    #[test]
    fn prefill_routes_once_and_to_mpp_where_it_builds() {
        if crate::test_support::paged_prefill_unsupported(
            "prefill_routes_once_and_to_mpp_where_it_builds",
        ) {
            return;
        }
        use crate::transformer::paged_int8_tests::adapter_for;
        let probe = |kind| unsafe { mlx_sys::mlx_paged_prefill_test_probe_count(kind) };
        let config = mlx_paged_attn::PagedAttentionConfig {
            block_size: 16,
            gpu_memory_mb: 256,
            head_size: 256,
            num_kv_heads: 2,
            num_layers: 1,
            max_seq_len: Some(1024),
            max_batch_size: Some(4),
        };
        let Some((mut adapter, _pool)) =
            adapter_for(config, 8, 8, mlx_paged_attn::metal::MetalDtype::Int8)
        else {
            return;
        };
        adapter.reset_for_new_request(1).unwrap();
        adapter.allocate_suffix_blocks(48).unwrap();
        let launches = probe(0);
        for (seed, chunk) in [(0usize, 40usize), (40, 8)].into_iter().enumerate() {
            store_attend_chunk(
                &mut adapter,
                chunk,
                PagedPass::Prefill,
                (16, 2, 256),
                seed as u32,
            );
        }
        assert_eq!(probe(0) - launches, 2, "one tile launch per chunk");
        assert!(crate::transformer::paged_kv_cache_adapter::device_prefill_mpp_builds());
        assert_eq!(probe(2), 1, "the device is asked once per process");
    }
}
