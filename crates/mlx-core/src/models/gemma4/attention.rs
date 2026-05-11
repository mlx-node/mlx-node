use std::sync::OnceLock;

use crate::array::MxArray;
use crate::array::attention::{scaled_dot_product_attention, scaled_dot_product_attention_causal};
use crate::array::mask::create_causal_mask;
use crate::inference_trace::{
    elapsed_ms, enabled as inference_trace_enabled, write as write_inference_trace,
};
use crate::nn::{Linear, RMSNorm, RoPE};
use crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter;
use mlx_sys as sys;
use napi::bindgen_prelude::*;

use super::config::Gemma4Config;
use super::layer_cache::Gemma4LayerCache;
use super::quantized_linear::{LinearProj, QuantizedLinear};

fn paged_prefill_paged_attention_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        crate::inference_trace::env_flag_enabled_or_default(
            "MLX_GEMMA4_PAGED_PREFILL_PAGED_ATTENTION",
            true,
        )
    })
}

fn native_kv_write_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        crate::inference_trace::env_flag_enabled_or_default("MLX_GEMMA4_NATIVE_KV_WRITE", true)
    })
}

/// Trim mask to match K/V sequence length (e.g. after RotatingKVCache eviction).
fn trim_mask(mask: Option<&MxArray>, kv_len: i64) -> Result<Option<MxArray>> {
    match mask {
        Some(m) => {
            let mask_len = m.shape_at(3)?;
            if mask_len == kv_len {
                Ok(Some(m.clone()))
            } else if mask_len > kv_len {
                Ok(Some(m.slice_axis(3, mask_len - kv_len, mask_len)?))
            } else {
                Err(Error::from_reason(format!(
                    "Gemma4 attention mask is shorter than K/V: mask_len={mask_len}, kv_len={kv_len}"
                )))
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
        let trace_enabled = inference_trace_enabled();

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
        if trace_enabled && offset > 0 && seq_len > 1 {
            write_inference_trace(format_args!(
                "[MLX_TRACE] gemma4 attention_flat_kv_ready offset_before={} seq_len={} kv_len={} mask_len={} needs_stash={}",
                offset,
                seq_len,
                keys.shape_at(2).unwrap_or(-1),
                mask.as_ref().and_then(|m| m.shape_at(3).ok()).unwrap_or(0),
                needs_stash
            ));
        }

        if trace_enabled && offset > 0 && seq_len > 1 {
            write_inference_trace(format_args!(
                "[MLX_TRACE] gemma4 attention_flat_sdpa_start offset_before={} seq_len={} q_heads={} kv_heads={} kv_len={} mask={}",
                offset,
                seq_len,
                queries.shape_at(1).unwrap_or(-1),
                keys.shape_at(1).unwrap_or(-1),
                keys.shape_at(2).unwrap_or(-1),
                if mask.is_some() { "explicit" } else { "causal" }
            ));
        }

        // Scaled dot-product attention with scale=1.0
        let output = if let Some(ref m) = mask {
            scaled_dot_product_attention(&queries, &keys, &values, 1.0, Some(m))?
        } else if seq_len > 1 {
            scaled_dot_product_attention_causal(&queries, &keys, &values, 1.0)?
        } else {
            scaled_dot_product_attention(&queries, &keys, &values, 1.0, None)?
        };
        if trace_enabled && offset > 0 && seq_len > 1 {
            write_inference_trace(format_args!(
                "[MLX_TRACE] gemma4 attention_flat_sdpa_done offset_before={} seq_len={} kv_len={}",
                offset,
                seq_len,
                keys.shape_at(2).unwrap_or(-1)
            ));
        }

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

    /// Forward pass driven by `PagedKVCacheAdapter` for global Gemma4
    /// attention layers.
    ///
    /// Mirrors `Lfm2Attention::forward_paged` but adapted to Gemma4's
    /// quirks:
    /// * Q/K/V are reshaped to `[B, T, H, D]` BEFORE per-head RMSNorm
    ///   (matches `forward`).
    /// * V receives a SCALE-FREE RMSNorm (`mlx_fast_rms_norm` with
    ///   `weight=null`), not a learned-scale norm.
    /// * RoPE dispatches between `Standard` (sliding) and
    ///   `Proportional` (global) — only global layers should call this
    ///   method, but the dispatch is uniform.
    /// * Optional K=V sharing collapses the V projection.
    /// * Attention scale is `1.0` (not `head_dim^-0.5`).
    ///
    /// Caller responsibilities (mirrors LFM2 / Qwen3 helper contracts):
    /// 1. `adapter.record_tokens(suffix)` BEFORE this call so the
    ///    cursor is advanced; `update_keys_values` enforces alignment.
    /// 2. `paged_idx` is the GLOBAL-LAYER ORDINAL into the adapter's
    ///    `LayerKVPool` (NOT the absolute decoder index). The pool is
    ///    sized for the global layer count.
    /// 3. `first_logical_position` is the first token's logical index
    ///    in the FULL request — used both as the RoPE offset and the
    ///    `update_keys_values` write position.
    /// 4. The decoder layer's `input_layernorm` is applied OUTSIDE this
    ///    method so `x` here is already pre-normalized (matches the
    ///    flat path's call site).
    ///
    /// Output: `[1, seq_len, hidden_size]` so the decoder layer's
    /// `apply_ffn_ple_scalar` tail can consume it unchanged.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_paged(
        &self,
        x: &MxArray,
        adapter: &mut PagedKVCacheAdapter,
        paged_idx: u32,
        first_logical_position: u32,
        cached_prefix_len: u32,
        is_prefill: bool,
    ) -> Result<MxArray> {
        let batch = x.shape_at(0)?;
        let seq_len = x.shape_at(1)?;
        let trace_enabled = inference_trace_enabled();

        // 1. Q/K/V projections (matches `forward`).
        let queries = self.q_proj.forward(x)?;
        let keys = self.k_proj.forward(x)?;
        let values = if self.k_is_v {
            keys.clone()
        } else {
            self.v_proj.as_ref().unwrap().forward(x)?
        };

        // 2. Reshape to [B, T, H, D] BEFORE per-head norm (matches `forward`).
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

        // 3. QKV normalization (Q/K learned-scale, V scale-free).
        let queries = self.q_norm.forward(&queries)?;
        let keys = self.k_norm.forward(&keys)?;
        let values = {
            let handle = unsafe {
                sys::mlx_fast_rms_norm(values.handle.0, std::ptr::null_mut(), self.v_norm_eps)
            };
            MxArray::from_handle(handle, "v_norm")?
        };

        // 4. Transpose to [B, H, T, D] BEFORE RoPE (matches `forward`).
        let queries = queries.transpose(Some(&[0, 2, 1, 3]))?;
        let keys = keys.transpose(Some(&[0, 2, 1, 3]))?;
        let values = values.transpose(Some(&[0, 2, 1, 3]))?;

        // 5. Apply RoPE using the request's logical offset.
        let rope_offset = first_logical_position as i32;
        let queries_bhtd = self.rope.forward(&queries, rope_offset)?;
        let keys_bhtd = self.rope.forward(&keys, rope_offset)?;
        let values_bhtd = values;

        // 6. Convert K/V into the paged layout `[num_tokens, n_kv_heads,
        //    head_dim]` expected by `update_keys_values`. Currently
        //    batch=1 so num_tokens = batch * seq_len = seq_len.
        //    [B, H_kv, T, D] -> [B, T, H_kv, D] -> [B*T, H_kv, D]
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

        let write_trace_start = trace_enabled.then(std::time::Instant::now);
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] gemma4 attention_paged_kv_write_start paged_idx={} first_position={} cached_prefix={} seq_len={} batch={} q_heads={} kv_heads={} head_dim={} input_dtype={:?} current_tokens={} blocks={}",
                paged_idx,
                first_logical_position,
                cached_prefix_len,
                seq_len,
                batch,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                x.dtype().ok(),
                adapter.current_token_count(),
                adapter.num_allocated_blocks()
            ));
        }
        let write_path = if native_kv_write_enabled() {
            match adapter.update_keys_values_native(
                paged_idx,
                &keys_paged,
                &values_paged,
                first_logical_position,
            ) {
                Ok(()) => "native",
                Err(err) => {
                    if trace_enabled {
                        write_inference_trace(format_args!(
                            "[MLX_TRACE] gemma4 attention_paged_kv_write_fallback paged_idx={} first_position={} seq_len={} error={}",
                            paged_idx, first_logical_position, seq_len, err
                        ));
                    }
                    adapter
                        .update_keys_values(
                            paged_idx,
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
                    paged_idx,
                    &keys_paged,
                    &values_paged,
                    first_logical_position,
                )
                .map_err(napi::Error::from_reason)?;
            "legacy"
        };
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] gemma4 attention_paged_kv_write_done paged_idx={} first_position={} seq_len={} path={} elapsed_ms={:.1}",
                paged_idx,
                first_logical_position,
                seq_len,
                write_path,
                write_trace_start.map(elapsed_ms).unwrap_or(0.0)
            ));
        }

        // 7. Compute attention output. Gemma4's attention scale is 1.0
        //    (the QK norm handles scaling).
        //
        // Single-token query path (`seq_len == 1`) ALWAYS takes the
        // mask=None branch regardless of `is_prefill` /
        // `cached_prefix_len`. The query is at logical position
        // `first_logical_position` and every key in
        // `[0, first_logical_position + 1)` is at a strictly-earlier
        // (or equal) position, so a causal mask never filters anything
        // out. But MLX dispatches `scaled_dot_product_attention` to
        // different kernels with vs. without an explicit mask, and the
        // mask-bearing kernel uses a different BF16 reduction order.
        // For Gemma4 paged-vs-flat parity at the prefill→decode
        // boundary (split-prefill pass 2 and every decode step) we
        // need the SAME kernel as flat decode (`scaled_dot_product_
        // attention(..., None)` at attention.rs:319) — without this,
        // the K/V written for the last prompt position diverges from
        // the flat cache by a few ULP per layer in BF16, which
        // compounds into an argmax flip on the first decode step.
        let attn_bhtd = if is_prefill && seq_len > 1 {
            if cached_prefix_len == 0 {
                // Fresh multi-token prefill: SDPA over in-flight Q/K/V
                // with internal causal mask.
                let sdpa_trace_start = trace_enabled.then(std::time::Instant::now);
                if trace_enabled {
                    write_inference_trace(format_args!(
                        "[MLX_TRACE] gemma4 attention_paged_sdpa_causal_start paged_idx={} seq_len={} cached_prefix=0",
                        paged_idx, seq_len
                    ));
                }
                let out = scaled_dot_product_attention_causal(
                    &queries_bhtd,
                    &keys_bhtd,
                    &values_bhtd,
                    1.0,
                )?;
                if trace_enabled {
                    write_inference_trace(format_args!(
                        "[MLX_TRACE] gemma4 attention_paged_sdpa_causal_done paged_idx={} seq_len={} elapsed_ms={:.1}",
                        paged_idx,
                        seq_len,
                        sdpa_trace_start.map(elapsed_ms).unwrap_or(0.0)
                    ));
                }
                out
            } else {
                // Cache-hit multi-token prefill: prefer the MLX paged-attention
                // bridge so each suffix token attends directly against the
                // on-GPU paged pool. The helper represents token i with
                // seq_len = cached_prefix_len + i + 1, matching the explicit
                // offset causal mask used by the materialized fallback without
                // copying full prefix K/V back through host memory.
                let total_ctx = cached_prefix_len + (seq_len as u32);
                let maybe_paged_attn = if batch == 1 && paged_prefill_paged_attention_enabled() {
                    let gather_trace_start = trace_enabled.then(std::time::Instant::now);
                    if trace_enabled {
                        write_inference_trace(format_args!(
                            "[MLX_TRACE] gemma4 attention_paged_prefill_gather_start paged_idx={} cached_prefix={} seq_len={} total_ctx={}",
                            paged_idx, cached_prefix_len, seq_len, total_ctx
                        ));
                    }

                    // [1, H, T, D] -> [T, H, D], matching
                    // PagedKVCacheAdapter::gather_kv_for_prefill_chunk.
                    let queries_3d = queries_bhtd
                        .squeeze(Some(&[0]))?
                        .transpose(Some(&[1, 0, 2]))?;
                    match adapter.gather_kv_for_prefill_chunk(
                        paged_idx,
                        &queries_3d,
                        cached_prefix_len,
                        1.0,
                    ) {
                        Ok(attn_3d) => {
                            let target_dtype = x.dtype()?;
                            let attn_3d = attn_3d.astype(target_dtype)?;
                            let out = attn_3d.transpose(Some(&[1, 0, 2]))?.reshape(&[
                                1,
                                self.num_heads as i64,
                                seq_len,
                                self.head_dim as i64,
                            ])?;
                            if trace_enabled {
                                write_inference_trace(format_args!(
                                    "[MLX_TRACE] gemma4 attention_paged_prefill_gather_done paged_idx={} cached_prefix={} seq_len={} total_ctx={} path=paged_attention elapsed_ms={:.1}",
                                    paged_idx,
                                    cached_prefix_len,
                                    seq_len,
                                    total_ctx,
                                    gather_trace_start.map(elapsed_ms).unwrap_or(0.0)
                                ));
                            }
                            Some(out)
                        }
                        Err(err) => {
                            if trace_enabled {
                                write_inference_trace(format_args!(
                                    "[MLX_TRACE] gemma4 attention_paged_prefill_gather_fallback paged_idx={} cached_prefix={} seq_len={} total_ctx={} error={}",
                                    paged_idx, cached_prefix_len, seq_len, total_ctx, err
                                ));
                            }
                            None
                        }
                    }
                } else {
                    None
                };

                match maybe_paged_attn {
                    Some(out) => out,
                    None => {
                        let read_trace_start = trace_enabled.then(std::time::Instant::now);
                        if trace_enabled {
                            write_inference_trace(format_args!(
                                "[MLX_TRACE] gemma4 attention_paged_read_kv_start paged_idx={} total_ctx={} cached_prefix={} seq_len={}",
                                paged_idx, total_ctx, cached_prefix_len, seq_len
                            ));
                        }
                        let (k_full, v_full) = adapter
                            .read_kv_range(paged_idx, 0, total_ctx)
                            .map_err(napi::Error::from_reason)?;
                        if trace_enabled {
                            write_inference_trace(format_args!(
                                "[MLX_TRACE] gemma4 attention_paged_read_kv_done paged_idx={} total_ctx={} elapsed_ms={:.1}",
                                paged_idx,
                                total_ctx,
                                read_trace_start.map(elapsed_ms).unwrap_or(0.0)
                            ));
                        }
                        let mask = create_causal_mask(
                            seq_len as i32,
                            Some(cached_prefix_len as i32),
                            None,
                        )?;
                        let sdpa_trace_start = trace_enabled.then(std::time::Instant::now);
                        if trace_enabled {
                            write_inference_trace(format_args!(
                                "[MLX_TRACE] gemma4 attention_paged_sdpa_masked_start paged_idx={} q_len={} kv_len={} cached_prefix={}",
                                paged_idx, seq_len, total_ctx, cached_prefix_len
                            ));
                        }
                        let out = scaled_dot_product_attention(
                            &queries_bhtd,
                            &k_full,
                            &v_full,
                            1.0,
                            Some(&mask),
                        )?;
                        if trace_enabled {
                            write_inference_trace(format_args!(
                                "[MLX_TRACE] gemma4 attention_paged_sdpa_masked_done paged_idx={} q_len={} kv_len={} elapsed_ms={:.1}",
                                paged_idx,
                                seq_len,
                                total_ctx,
                                sdpa_trace_start.map(elapsed_ms).unwrap_or(0.0)
                            ));
                        }
                        out
                    }
                }
            }
        } else {
            // Single-token path (decode OR split-prefill pass 2):
            // dispatch graph-native paged attention directly against the
            // on-GPU paged buffers. That keeps the native paged-kv-write
            // outputs and the attention read in one MLX dependency graph,
            // avoiding the per-layer sync that the raw gather fallback must
            // do before it reads the pool outside MLX's scheduler.
            //
            // `gather_kv_for_decode_graph` expects queries shape
            // `[1, num_query_heads, head_size]` (3-D). For seq_len=1
            // `queries_bhtd` is `[1, n_heads, 1, head_dim]`, so squeeze
            // axis 2 to land on the expected layout. Returns
            // `[1, n_heads, head_dim]`; reshape back to the standard
            // `[B, H, T, D]` tail with T=1.
            let queries_3d = queries_bhtd.squeeze(Some(&[2]))?.reshape(&[
                1,
                self.num_heads as i64,
                self.head_dim as i64,
            ])?;
            let gather_trace_start = trace_enabled.then(std::time::Instant::now);
            let attn_3d = match adapter.gather_kv_for_decode_graph(paged_idx, &queries_3d, 1.0, 1.0)
            {
                Ok(attn_3d) => {
                    if trace_enabled {
                        write_inference_trace(format_args!(
                            "[MLX_TRACE] gemma4 attention_paged_decode_gather_done paged_idx={} path=graph total_ctx={} elapsed_ms={:.1}",
                            paged_idx,
                            adapter.current_token_count(),
                            gather_trace_start.map(elapsed_ms).unwrap_or(0.0)
                        ));
                    }
                    attn_3d
                }
                Err(err) => {
                    if trace_enabled {
                        write_inference_trace(format_args!(
                            "[MLX_TRACE] gemma4 attention_paged_decode_gather_fallback paged_idx={} path=raw total_ctx={} error={}",
                            paged_idx,
                            adapter.current_token_count(),
                            err
                        ));
                    }
                    adapter
                        .gather_kv_for_decode(paged_idx, &queries_3d, 1.0, /* softcap */ 1.0)
                        .map_err(napi::Error::from_reason)?
                }
            };
            // Cast back to x's dtype so the residual stays homogeneous.
            let target_dtype = x.dtype()?;
            let attn_3d = attn_3d.astype(target_dtype)?;
            attn_3d.reshape(&[1, self.num_heads as i64, 1, self.head_dim as i64])?
        };

        // 8. Output: [B, H, T, D] -> [B, T, H*D] -> projection.
        let output = attn_bhtd.transpose(Some(&[0, 2, 1, 3]))?;
        let output = output.reshape(&[batch, seq_len, (self.num_heads * self.head_dim) as i64])?;
        self.o_proj.forward(&output)
    }

    // ========== Weight setters ==========

    /// Forward pass for KV-shared layers whose anchor is a global layer
    /// routed through the paged adapter.
    ///
    /// Only Q is computed; K and V are consumed directly from the anchor's
    /// paged slot (already RoPE-applied since the anchor wrote them
    /// post-RoPE during its own `forward_paged` call).
    ///
    /// Caller responsibilities:
    /// 1. `cache_offset` is the RoPE offset for the queries — equal to
    ///    the anchor's logical position when the anchor processed the
    ///    same chunk (i.e. `first_logical_position` for prefill,
    ///    `current_token_count - 1` for decode).
    /// 2. `total_ctx` is the number of K/V tokens available in the
    ///    anchor's paged slot. For prefill of a fresh suffix this is
    ///    `cached_prefix_len + seq_len`; for decode it is the live
    ///    token count.
    ///
    /// Output: `[1, seq_len, hidden_size]`, ready for the decoder
    /// layer's `apply_ffn_ple_scalar` tail.
    pub fn forward_paged_shared(
        &self,
        x: &MxArray,
        adapter: &mut PagedKVCacheAdapter,
        anchor_paged_idx: u32,
        cache_offset: i32,
        total_ctx: u32,
        is_prefill: bool,
    ) -> Result<MxArray> {
        let batch = x.shape_at(0)?;
        let seq_len = x.shape_at(1)?;
        let trace_enabled = inference_trace_enabled();

        // Q-only path (mirrors flat `forward_shared`).
        let queries = self.q_proj.forward(x)?;
        let queries =
            queries.reshape(&[batch, seq_len, self.num_heads as i64, self.head_dim as i64])?;
        let queries = self.q_norm.forward(&queries)?;
        let queries = queries.transpose(Some(&[0, 2, 1, 3]))?;
        let queries_bhtd = self.rope.forward(&queries, cache_offset)?;

        // SDPA. Same scale=1.0 as `forward_paged`. For cache-hit suffix
        // prefill, prefer the on-GPU paged prefill helper and fall back to
        // materialized K/V only if the bridge rejects this shape/request. For
        // decode (seq_len == 1), dispatch the on-GPU decode helper — every
        // cached key is at a strictly earlier position, so mask=None is
        // implicit.
        let attn_bhtd = if is_prefill && seq_len > 1 {
            let cached_prefix_len = total_ctx
                .checked_sub(seq_len as u32)
                .ok_or_else(|| Error::from_reason("forward_paged_shared: total_ctx < seq_len"))?;
            let maybe_paged_attn = if batch == 1 && paged_prefill_paged_attention_enabled() {
                let gather_trace_start = trace_enabled.then(std::time::Instant::now);
                if trace_enabled {
                    write_inference_trace(format_args!(
                        "[MLX_TRACE] gemma4 attention_paged_shared_prefill_gather_start anchor_paged_idx={} cached_prefix={} seq_len={} total_ctx={}",
                        anchor_paged_idx, cached_prefix_len, seq_len, total_ctx
                    ));
                }
                let queries_3d = queries_bhtd
                    .squeeze(Some(&[0]))?
                    .transpose(Some(&[1, 0, 2]))?;
                match adapter.gather_kv_for_prefill_chunk(
                    anchor_paged_idx,
                    &queries_3d,
                    cached_prefix_len,
                    1.0,
                ) {
                    Ok(attn_3d) => {
                        let target_dtype = x.dtype()?;
                        let attn_3d = attn_3d.astype(target_dtype)?;
                        let out = attn_3d.transpose(Some(&[1, 0, 2]))?.reshape(&[
                            1,
                            self.num_heads as i64,
                            seq_len,
                            self.head_dim as i64,
                        ])?;
                        if trace_enabled {
                            write_inference_trace(format_args!(
                                "[MLX_TRACE] gemma4 attention_paged_shared_prefill_gather_done anchor_paged_idx={} cached_prefix={} seq_len={} total_ctx={} path=paged_attention elapsed_ms={:.1}",
                                anchor_paged_idx,
                                cached_prefix_len,
                                seq_len,
                                total_ctx,
                                gather_trace_start.map(elapsed_ms).unwrap_or(0.0)
                            ));
                        }
                        Some(out)
                    }
                    Err(err) => {
                        if trace_enabled {
                            write_inference_trace(format_args!(
                                "[MLX_TRACE] gemma4 attention_paged_shared_prefill_gather_fallback anchor_paged_idx={} cached_prefix={} seq_len={} total_ctx={} error={}",
                                anchor_paged_idx, cached_prefix_len, seq_len, total_ctx, err
                            ));
                        }
                        None
                    }
                }
            } else {
                None
            };

            match maybe_paged_attn {
                Some(out) => out,
                None => {
                    let read_trace_start = trace_enabled.then(std::time::Instant::now);
                    if trace_enabled {
                        write_inference_trace(format_args!(
                            "[MLX_TRACE] gemma4 attention_paged_shared_read_kv_start anchor_paged_idx={} total_ctx={} cached_prefix={} seq_len={}",
                            anchor_paged_idx, total_ctx, cached_prefix_len, seq_len
                        ));
                    }
                    let (shared_keys, shared_values) = adapter
                        .read_kv_range(anchor_paged_idx, 0, total_ctx)
                        .map_err(napi::Error::from_reason)?;
                    if trace_enabled {
                        write_inference_trace(format_args!(
                            "[MLX_TRACE] gemma4 attention_paged_shared_read_kv_done anchor_paged_idx={} total_ctx={} elapsed_ms={:.1}",
                            anchor_paged_idx,
                            total_ctx,
                            read_trace_start.map(elapsed_ms).unwrap_or(0.0)
                        ));
                    }
                    let mask =
                        create_causal_mask(seq_len as i32, Some(cached_prefix_len as i32), None)?;
                    let sdpa_trace_start = trace_enabled.then(std::time::Instant::now);
                    if trace_enabled {
                        write_inference_trace(format_args!(
                            "[MLX_TRACE] gemma4 attention_paged_shared_sdpa_masked_start anchor_paged_idx={} q_len={} kv_len={} cached_prefix={}",
                            anchor_paged_idx, seq_len, total_ctx, cached_prefix_len
                        ));
                    }
                    let out = scaled_dot_product_attention(
                        &queries_bhtd,
                        &shared_keys,
                        &shared_values,
                        1.0,
                        Some(&mask),
                    )?;
                    if trace_enabled {
                        write_inference_trace(format_args!(
                            "[MLX_TRACE] gemma4 attention_paged_shared_sdpa_masked_done anchor_paged_idx={} q_len={} kv_len={} elapsed_ms={:.1}",
                            anchor_paged_idx,
                            seq_len,
                            total_ctx,
                            sdpa_trace_start.map(elapsed_ms).unwrap_or(0.0)
                        ));
                    }
                    out
                }
            }
        } else {
            // Single-token path (decode OR split-prefill pass 2):
            // dispatch graph-native paged attention against the anchor's
            // paged slot. Queries shape: `[1, num_heads, head_dim]`
            // (squeeze T=1).
            let queries_3d = queries_bhtd.squeeze(Some(&[2]))?.reshape(&[
                1,
                self.num_heads as i64,
                self.head_dim as i64,
            ])?;
            let gather_trace_start = trace_enabled.then(std::time::Instant::now);
            let attn_3d = match adapter.gather_kv_for_decode_graph(
                anchor_paged_idx,
                &queries_3d,
                1.0,
                1.0,
            ) {
                Ok(attn_3d) => {
                    if trace_enabled {
                        write_inference_trace(format_args!(
                            "[MLX_TRACE] gemma4 attention_paged_shared_decode_gather_done anchor_paged_idx={} path=graph total_ctx={} elapsed_ms={:.1}",
                            anchor_paged_idx,
                            total_ctx,
                            gather_trace_start.map(elapsed_ms).unwrap_or(0.0)
                        ));
                    }
                    attn_3d
                }
                Err(err) => {
                    if trace_enabled {
                        write_inference_trace(format_args!(
                            "[MLX_TRACE] gemma4 attention_paged_shared_decode_gather_fallback anchor_paged_idx={} path=raw total_ctx={} error={}",
                            anchor_paged_idx, total_ctx, err
                        ));
                    }
                    adapter
                        .gather_kv_for_decode(
                            anchor_paged_idx,
                            &queries_3d,
                            1.0,
                            /* softcap */ 1.0,
                        )
                        .map_err(napi::Error::from_reason)?
                }
            };
            let target_dtype = x.dtype()?;
            let attn_3d = attn_3d.astype(target_dtype)?;
            attn_3d.reshape(&[1, self.num_heads as i64, 1, self.head_dim as i64])?
        };

        // Output projection.
        let output = attn_bhtd.transpose(Some(&[0, 2, 1, 3]))?;
        let output = output.reshape(&[batch, seq_len, (self.num_heads * self.head_dim) as i64])?;
        self.o_proj.forward(&output)
    }

    // ========== Test-only weight getters ==========
    #[cfg(test)]
    pub(crate) fn q_proj_weight(&self) -> MxArray {
        self.q_proj.get_weight()
    }
    #[cfg(test)]
    pub(crate) fn k_proj_weight(&self) -> MxArray {
        self.k_proj.get_weight()
    }
    #[cfg(test)]
    pub(crate) fn v_proj_weight_opt(&self) -> Option<MxArray> {
        self.v_proj.as_ref().map(|p| p.get_weight())
    }
    #[cfg(test)]
    pub(crate) fn o_proj_weight(&self) -> MxArray {
        self.o_proj.get_weight()
    }
    #[cfg(test)]
    pub(crate) fn q_norm_weight(&self) -> MxArray {
        self.q_norm.get_weight()
    }
    #[cfg(test)]
    pub(crate) fn k_norm_weight(&self) -> MxArray {
        self.k_norm.get_weight()
    }

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trim_mask_rejects_mask_shorter_than_kv() {
        let mask = MxArray::zeros(&[1, 1, 2, 3], None).unwrap();
        let err = match trim_mask(Some(&mask), 4) {
            Ok(_) => panic!("expected trim_mask to reject a short mask"),
            Err(err) => err,
        };
        assert!(
            err.reason.contains("mask is shorter than K/V"),
            "unexpected error: {}",
            err.reason
        );
    }

    #[test]
    fn trim_mask_trims_longer_mask_to_kv_len() {
        let mask = MxArray::zeros(&[1, 1, 2, 5], None).unwrap();
        let trimmed = trim_mask(Some(&mask), 3).unwrap().unwrap();
        assert_eq!(trimmed.shape_at(0).unwrap(), 1);
        assert_eq!(trimmed.shape_at(1).unwrap(), 1);
        assert_eq!(trimmed.shape_at(2).unwrap(), 2);
        assert_eq!(trimmed.shape_at(3).unwrap(), 3);
    }
}
