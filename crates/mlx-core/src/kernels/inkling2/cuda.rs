//! Safe Rust facades for Inkling-2-Mini CUDA primitives
//! (`crates/mlx-sys/src/cuda/inkling2/{ops,sdpa,moe}.cpp`).
//!
//! All functions dispatch only when `active_backend() == ActiveBackend::CudaGpu`
//! (and return `Ok(None)` cleanly off CUDA or on off-contract shapes/dtypes so
//! callers can fall back without panicking).

use mlx_sys as sys;
use napi::bindgen_prelude::*;

use crate::accelerator::{cuda_ops_available, cuda_sym8_available};
use crate::array::MxArray;

/// Whether Inkling-2-Mini CUDA primitives can run on the active device.
#[inline]
pub fn available() -> bool {
    cuda_ops_available()
}

/// Whether Ampere+ (SM >= 8.0) Inkling-2-Mini Tensor Core primitives
/// (`lm_head_softcap_cuda`, `sdpa_*_cuda`, `moe_forward_cuda`) are available.
#[inline]
pub fn sm80_available() -> bool {
    cuda_sym8_available()
}

/// Quantization mode for Inkling-2-Mini MoE expert weight banks.
///
/// Discriminants match the C++ ABI in `crates/mlx-sys/src/cuda/inkling2/moe.cpp`.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoeQuantMode {
    /// NVFP4 (4-bit E2M1, group_size=16, uint8 FP8 E4M3FN scales, optional FP32 outer_scale).
    Nvfp4 = 0,
    /// Symmetric Q8_0 (8-bit XOR-0x80 signed, group_size=32, F16/BF16 scales, biases = None).
    SymQ8 = 1,
    /// Affine Q8_0 (8-bit unsigned, group_size=32, F16/BF16 scales + biases).
    AffineQ8 = 2,
    /// RowMajor Q5_K (5-bit, group_size=32, QK=256 super-blocks; `gate_up` only).
    Q5K = 3,
    /// MXFP8 (8-bit FP8 E4M3FN, group_size=32, uint8 UE8M0 scales).
    Mxfp8 = 4,
}

/// Output mode for [`lm_head_softcap_cuda`].
///
/// Discriminants match the C++ ABI in `crates/mlx-sys/src/cuda/inkling2/ops.cpp`.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LmHeadOutMode {
    /// Full `[M, unpadded_vocab]` FP32 logits (with softcap if `softcap > 0`).
    LogitsF32 = 0,
    /// Zero-logit-DRAM greedy argmax `[M]` Int32 (with softcap if `softcap > 0`).
    GreedyArgmax = 1,
    /// Full `[M, unpadded_vocab]` BF16 logits (uncapped when `softcap <= 0`).
    LogitsBf16 = 2,
}

/// Parameters for [`kv_conv_prep_store_cuda`].
pub struct KvConvPrepParams<'a> {
    /// `[T, 3072]` BF16 raw query projection output (`24 * 128`).
    pub q_raw: &'a MxArray,
    /// `[T, 384]` BF16 raw relative-bias query output (`24 * 16`).
    pub r_raw: &'a MxArray,
    /// `[T, 1024]` BF16 raw key projection output (`8 * 128`).
    pub k_raw: &'a MxArray,
    /// `[T, 1024]` BF16 raw value projection output (`8 * 128`).
    pub v_raw: &'a MxArray,
    /// `[1, 24, 128]` (or flat size 3072) FP32 Q L2-clip scale.
    pub q_clip: &'a MxArray,
    /// `[1, 24, 16]` (or flat size 384) FP32 R L2-clip scale.
    pub rel_clip: &'a MxArray,
    /// `[1024, 4]` FP32 static K ShortConv weights (oldest-to-newest).
    pub k_conv_w: &'a MxArray,
    /// `[1024, 4]` FP32 static V ShortConv weights (oldest-to-newest).
    pub v_conv_w: &'a MxArray,
    /// Global-only `[T, 8, 4]` or `[T, 32]` BF16/FP32 dynamic K ShortConv taps (newest-to-oldest).
    pub dyn_k: Option<&'a MxArray>,
    /// Global-only `[T, 8, 4]` or `[T, 32]` BF16/FP32 dynamic V ShortConv taps (newest-to-oldest).
    pub dyn_v: Option<&'a MxArray>,
    /// `[3, 1024]` BF16 rolling pre-conv K history (oldest at row 0).
    pub k_hist_in: &'a MxArray,
    /// `[3, 1024]` BF16 rolling pre-conv V history (oldest at row 0).
    pub v_hist_in: &'a MxArray,
    /// `[1, 8, cap, 128]` BF16 or Int8 preallocated K cache (mutated in-place).
    pub kv_k_cache: &'a MxArray,
    /// `[1, 8, cap, 128]` BF16 or Int8 preallocated V cache (mutated in-place).
    pub kv_v_cache: &'a MxArray,
    /// INT8-only `[1, 8, cap, 1]` or `[1, 8, cap]` FP32 K scale cache (mutated in-place).
    pub kv_k_scales: Option<&'a MxArray>,
    /// INT8-only `[1, 8, cap, 1]` or `[1, 8, cap]` FP32 V scale cache (mutated in-place).
    pub kv_v_scales: Option<&'a MxArray>,
    /// Optional `[T, 4]` Int32 tree ShortConv tap indices (`< 0` reads `hist_in[3 + idx]`).
    pub conv_taps: Option<&'a MxArray>,
    /// Optional `[T]` Int32 tree depths (`pos = start_pos + depths[t]` for Global `tau(p)`).
    pub depths: Option<&'a MxArray>,
    /// Absolute sequence position of token `0` (or tree root).
    pub start_pos: i32,
    /// Starting physical slot index in the KV cache for token `0`.
    pub write_offset: i32,
    /// Circular ring capacity (`0` = linear flat cache, `> 0` = circular ring modulo `ring_cap`).
    pub ring_cap: i32,
    /// Whether this layer is Global (enables `tau(p)` scaling on Q and R).
    pub is_global: bool,
    /// Global log-length scaling floor `n_floor` (e.g. `8192.0`).
    pub log_n_floor: f32,
    /// Global log-length scaling alpha (e.g. `0.1`).
    pub log_alpha: f32,
    /// When `true`, emits updated `[3, 1024]` rolling histories in `k_hist` / `v_hist`.
    pub update_hist: bool,
}

/// Outputs of [`kv_conv_prep_store_cuda`].
pub struct KvConvPrepOutput {
    /// `[1, 24, T, 128]` BF16 clipped (+ Global `tau`-scaled) Q ready for SDPA.
    pub q: MxArray,
    /// `[T, 24, 16]` FP32 clipped (+ Global `tau`-scaled) R ready for SDPA.
    pub r: MxArray,
    /// Updated `[3, 1024]` BF16 K ShortConv history when `update_hist == true`.
    pub k_hist: Option<MxArray>,
    /// Updated `[3, 1024]` BF16 V ShortConv history when `update_hist == true`.
    pub v_hist: Option<MxArray>,
    /// Graph-dependency alias of mutated `kv_k_cache` (must be passed into downstream SDPA).
    pub kv_k_alias: MxArray,
    /// Graph-dependency alias of mutated `kv_v_cache` (must be passed into downstream SDPA).
    pub kv_v_alias: MxArray,
    /// Graph-dependency alias of mutated `kv_k_scales` (present for INT8 KV).
    pub ks_alias: Option<MxArray>,
    /// Graph-dependency alias of mutated `kv_v_scales` (present for INT8 KV).
    pub vs_alias: Option<MxArray>,
}

/// Outputs of [`sconv_k4_residual_cuda`].
pub struct SconvK4Output {
    /// `[T, C]` BF16 output (post-ShortConv + optional residual).
    pub y: MxArray,
    /// Updated `[3, C]` BF16 rolling history when `update_hist == true`.
    pub hist: Option<MxArray>,
}

/// Parameters for [`sdpa_few_cuda`] and [`sdpa_prefill_cuda`].
pub struct SdpaParams<'a> {
    /// `[B, Hq, T, 128]` BF16 queries (typically `B=1, Hq=24`).
    pub q: &'a MxArray,
    /// `[B*T, Hq, 16]` or `[B, T, Hq, 16]` FP32 relative-bias queries.
    pub r: &'a MxArray,
    /// `[B, Hkv, cap, 128]` BF16 or Int8 K cache.
    pub k_cache: &'a MxArray,
    /// `[B, Hkv, cap, 128]` BF16 or Int8 V cache.
    pub v_cache: &'a MxArray,
    /// INT8-only `[B, Hkv, cap, 1]` or `[B, Hkv, cap]` FP32 K scales.
    pub k_scales: Option<&'a MxArray>,
    /// INT8-only `[B, Hkv, cap, 1]` or `[B, Hkv, cap]` FP32 V scales.
    pub v_scales: Option<&'a MxArray>,
    /// `[rel_extent, 16]` FP32 row-contiguous relative bias table.
    pub rel_bias_w: &'a MxArray,
    /// `[B*T, Hq*128]`, `[B, T, Hq*128]`, `[T, Hq, 128]`, or `[B, T, Hq, 128]` BF16 gate.
    pub gate: &'a MxArray,
    /// `[B*T, Hq]` or `[B, T, Hq]` BF16 or FP32 per-token sink logits.
    pub sink: &'a MxArray,
    /// Speculative tree depths `[T]` Int32 (`sdpa_few_cuda` only).
    pub tree_depths: Option<&'a MxArray>,
    /// Speculative tree ancestor bitmasks `[T]` Uint32 (`sdpa_few_cuda` only).
    pub tree_mask: Option<&'a MxArray>,
    /// Absolute sequence position of token `0` (or tree root).
    pub start_pos: i32,
    /// Number of prior prefix tokens already in the cache before the current `T` tokens.
    pub cache_len: i32,
    /// Absolute sequence position of logical prefix token `0`.
    pub kv_start_pos: i32,
    /// Physical ring slot of logical prefix token `0` (ignored when `ring_cap == 0`).
    pub kv_phys_base: i32,
    /// Physical slot where the current `T` tokens were written.
    pub write_offset: i32,
    /// Circular ring capacity (`0` = linear flat cache, `> 0` = circular ring modulo `ring_cap`).
    pub ring_cap: i32,
    /// Whether sliding-window attention (`dist < sliding_window`) is active.
    pub is_local: bool,
    /// Sliding window size (e.g. `1024`).
    pub sliding_window: i32,
}

/// Parameters for [`moe_forward_cuda`].
pub struct MoeForwardParams<'a> {
    /// `[M, 2304]` (or `[..., 2304]`) BF16 normalized MLP input (`M in 1..=65535`).
    pub x: &'a MxArray,
    /// `[128, 2304]` FP32 row-major router weight.
    pub router_w: &'a MxArray,
    /// Optional `[128]` FP32 router selection bias (`e_score_correction_bias`).
    pub router_bias: Option<&'a MxArray>,
    /// `[128, 2304, ...]` Uint32 packed `gate_up_proj` expert weights.
    pub gu_w: &'a MxArray,
    /// `[128, 2304, ...]` `gate_up_proj` group scales.
    pub gu_scales: &'a MxArray,
    /// Optional `[128, 2304, ...]` `gate_up_proj` group biases (`AffineQ8` / `Q5K`).
    pub gu_biases: Option<&'a MxArray>,
    /// Optional `[128]` FP32 per-expert `gate_up_proj` outer scale (NVFP4).
    pub gu_outer_scale: Option<&'a MxArray>,
    /// Quantization format of `gate_up_proj`.
    pub gu_mode: MoeQuantMode,
    /// `true` for interleaved `(g0, u0, g1, u1, ...)` (HF/SafeTensors); `false` for split `[gate, up]` (GGUF).
    pub gu_interleaved: bool,
    /// `[128, 2304, ...]` Uint32 packed `down_proj` expert weights.
    pub dn_w: &'a MxArray,
    /// `[128, 2304, ...]` `down_proj` group scales.
    pub dn_scales: &'a MxArray,
    /// Optional `[128, 2304, ...]` `down_proj` group biases (`AffineQ8`).
    pub dn_biases: Option<&'a MxArray>,
    /// Optional `[128]` FP32 per-expert `down_proj` outer scale (folded into router gamma).
    pub dn_outer_scale: Option<&'a MxArray>,
    /// Quantization format of `down_proj` (`Q5K` is not supported for `down_proj`).
    pub dn_mode: MoeQuantMode,
    /// Router scaling multiplier (`8.0` for Inkling-2-Mini).
    pub route_scale: f32,
    /// Optional `[2304, 4]` FP32 static `ffn_post_conv` weights (fuses ShortConv into combine).
    pub post_conv_w: Option<&'a MxArray>,
    /// `[3, 2304]` BF16 rolling pre-conv history (required when `post_conv_w` is `Some`).
    pub hist_in: Option<&'a MxArray>,
    /// Optional `[M, 2304]` BF16 residual to add after `post_conv` (requires `post_conv_w` is `Some`).
    pub residual: Option<&'a MxArray>,
    /// Optional `[M, 4]` Int32 tree ShortConv tap indices (requires `post_conv_w` is `Some`).
    pub conv_taps: Option<&'a MxArray>,
    /// When `true` (and `post_conv_w` is `Some`), writes updated `[3, 2304]` history to `out.hist`.
    pub update_hist: bool,
    /// When `true`, emits the pre-conv combined `[M, 2304]` BF16 rows in `out.pre_conv` (for rollback tape).
    pub emit_pre_conv: bool,
}

/// Outputs of [`moe_forward_cuda`].
pub struct MoeForwardOutput {
    /// `[M, 2304]` (same shape as `x`) BF16 MoE output (after optional `post_conv` + `residual`).
    pub y: MxArray,
    /// Updated `[3, 2304]` BF16 history (present iff `post_conv_w.is_some() && update_hist`).
    pub hist: Option<MxArray>,
    /// Pre-conv `[M, 2304]` BF16 combined MoE rows (present iff `emit_pre_conv`).
    pub pre_conv: Option<MxArray>,
}

#[inline]
unsafe fn delete_if_non_null(ptr: *mut sys::mlx_array) {
    if !ptr.is_null() {
        unsafe { sys::mlx_array_delete(ptr) };
    }
}

/// 1. Block AttnRes over `1..=8` `[T, D]` BF16/FP32 sources + optional fused sublayer RMSNorm.
///
/// - When `norm_weights` is `None`: returns FP32 (`[T, D]` if `Q == 1 && squeeze_q1`, else `[Q, T, D]`).
/// - When `norm_weights` is `Some`: returns BF16 (`[T, D]` if `Q == 1 && squeeze_q1`, else `[Q, T, D]`).
#[inline]
pub fn block_attn_res_norm_cuda(
    sources: &[&MxArray],
    queries: &MxArray,
    norm_weights: Option<&MxArray>,
    eps: f32,
    squeeze_q1: bool,
) -> Result<Option<MxArray>> {
    if !available() || sources.is_empty() || sources.len() > 8 {
        return Ok(None);
    }
    let mut ptrs = [std::ptr::null_mut(); 8];
    for (slot, src) in ptrs.iter_mut().zip(sources.iter()) {
        *slot = src.as_raw_ptr();
    }
    let nw_ptr = norm_weights.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    let handle = unsafe {
        sys::mlx_inkling2_block_attn_res_norm_cuda(
            ptrs.as_ptr(),
            sources.len(),
            queries.as_raw_ptr(),
            nw_ptr,
            eps,
            squeeze_q1,
        )
    };
    if handle.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(
            handle,
            "inkling2_block_attn_res_norm_cuda",
        )?))
    }
}

/// 2. Fused rank-64 LoopController + sublayer RMSNorm (`g_count` in `{1, 3}`).
///
/// Returns BF16 `[T, D]` when `g_count == 1 && squeeze_g1`, else `[G, T, D]`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn loop_ctrl_norm_cuda(
    dest: &MxArray,
    msg: &MxArray,
    down_w: &MxArray,
    gain_forget_w: &MxArray,
    norm_w: &MxArray,
    eps: f32,
    g_count: i32,
    squeeze_g1: bool,
) -> Result<Option<MxArray>> {
    if !available() || (g_count != 1 && g_count != 3) {
        return Ok(None);
    }
    let handle = unsafe {
        sys::mlx_inkling2_loop_ctrl_norm_cuda(
            dest.as_raw_ptr(),
            msg.as_raw_ptr(),
            down_w.as_raw_ptr(),
            gain_forget_w.as_raw_ptr(),
            norm_w.as_raw_ptr(),
            eps,
            g_count,
            squeeze_g1,
        )
    };
    if handle.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(
            handle,
            "inkling2_loop_ctrl_norm_cuda",
        )?))
    }
}

/// 3. Fused Q/R L2-clip + K/V ShortConv + K L2-clip + V RMSNorm + Global `tau(p)`
///    + in-place BF16/INT8 KV cache store (linear flat or circular ring, linear or tree).
pub fn kv_conv_prep_store_cuda(params: &KvConvPrepParams<'_>) -> Result<Option<KvConvPrepOutput>> {
    if !available() {
        return Ok(None);
    }
    let is_dyn = params.dyn_k.is_some() || params.dyn_v.is_some();
    if is_dyn && (params.dyn_k.is_none() || params.dyn_v.is_none()) {
        return Ok(None);
    }
    let is_int8 = params.kv_k_scales.is_some() || params.kv_v_scales.is_some();
    if is_int8 && (params.kv_k_scales.is_none() || params.kv_v_scales.is_none()) {
        return Ok(None);
    }

    let dyn_k_ptr = params
        .dyn_k
        .map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    let dyn_v_ptr = params
        .dyn_v
        .map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    let ks_ptr = params
        .kv_k_scales
        .map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    let vs_ptr = params
        .kv_v_scales
        .map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    let taps_ptr = params
        .conv_taps
        .map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    let depths_ptr = params
        .depths
        .map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);

    let mut out_q = std::ptr::null_mut();
    let mut out_r = std::ptr::null_mut();
    let mut out_k_hist = std::ptr::null_mut();
    let mut out_v_hist = std::ptr::null_mut();
    let mut out_kv_k = std::ptr::null_mut();
    let mut out_kv_v = std::ptr::null_mut();
    let mut out_ks = std::ptr::null_mut();
    let mut out_vs = std::ptr::null_mut();

    let ok = unsafe {
        sys::mlx_inkling2_kv_conv_prep_store_cuda(
            params.q_raw.as_raw_ptr(),
            params.r_raw.as_raw_ptr(),
            params.k_raw.as_raw_ptr(),
            params.v_raw.as_raw_ptr(),
            params.q_clip.as_raw_ptr(),
            params.rel_clip.as_raw_ptr(),
            params.k_conv_w.as_raw_ptr(),
            params.v_conv_w.as_raw_ptr(),
            dyn_k_ptr,
            dyn_v_ptr,
            params.k_hist_in.as_raw_ptr(),
            params.v_hist_in.as_raw_ptr(),
            params.kv_k_cache.as_raw_ptr(),
            params.kv_v_cache.as_raw_ptr(),
            ks_ptr,
            vs_ptr,
            taps_ptr,
            depths_ptr,
            params.start_pos,
            params.write_offset,
            params.ring_cap,
            params.is_global,
            params.log_n_floor,
            params.log_alpha,
            params.update_hist,
            &mut out_q,
            &mut out_r,
            if params.update_hist {
                &mut out_k_hist
            } else {
                std::ptr::null_mut()
            },
            if params.update_hist {
                &mut out_v_hist
            } else {
                std::ptr::null_mut()
            },
            &mut out_kv_k,
            &mut out_kv_v,
            if is_int8 {
                &mut out_ks
            } else {
                std::ptr::null_mut()
            },
            if is_int8 {
                &mut out_vs
            } else {
                std::ptr::null_mut()
            },
        )
    };

    if !ok
        || out_q.is_null()
        || out_r.is_null()
        || out_kv_k.is_null()
        || out_kv_v.is_null()
        || (params.update_hist && (out_k_hist.is_null() || out_v_hist.is_null()))
        || (is_int8 && (out_ks.is_null() || out_vs.is_null()))
    {
        unsafe {
            delete_if_non_null(out_q);
            delete_if_non_null(out_r);
            delete_if_non_null(out_k_hist);
            delete_if_non_null(out_v_hist);
            delete_if_non_null(out_kv_k);
            delete_if_non_null(out_kv_v);
            delete_if_non_null(out_ks);
            delete_if_non_null(out_vs);
        }
        return Ok(None);
    }

    Ok(Some(KvConvPrepOutput {
        q: MxArray::from_handle(out_q, "inkling2_kv_conv_prep_store:q")?,
        r: MxArray::from_handle(out_r, "inkling2_kv_conv_prep_store:r")?,
        k_hist: if params.update_hist {
            Some(MxArray::from_handle(
                out_k_hist,
                "inkling2_kv_conv_prep_store:k_hist",
            )?)
        } else {
            None
        },
        v_hist: if params.update_hist {
            Some(MxArray::from_handle(
                out_v_hist,
                "inkling2_kv_conv_prep_store:v_hist",
            )?)
        } else {
            None
        },
        kv_k_alias: MxArray::from_handle(out_kv_k, "inkling2_kv_conv_prep_store:kv_k")?,
        kv_v_alias: MxArray::from_handle(out_kv_v, "inkling2_kv_conv_prep_store:kv_v")?,
        ks_alias: if is_int8 {
            Some(MxArray::from_handle(
                out_ks,
                "inkling2_kv_conv_prep_store:ks",
            )?)
        } else {
            None
        },
        vs_alias: if is_int8 {
            Some(MxArray::from_handle(
                out_vs,
                "inkling2_kv_conv_prep_store:vs",
            )?)
        } else {
            None
        },
    }))
}

/// 4. Static ShortConv `k=4` + optional FP32/BF16 residual add (linear or tree `conv_taps`).
#[inline]
pub fn sconv_k4_residual_cuda(
    x: &MxArray,
    weight: &MxArray,
    hist_in: &MxArray,
    residual: Option<&MxArray>,
    conv_taps: Option<&MxArray>,
    update_hist: bool,
) -> Result<Option<SconvK4Output>> {
    if !available() {
        return Ok(None);
    }
    let res_ptr = residual.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    let taps_ptr = conv_taps.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    let mut out_y = std::ptr::null_mut();
    let mut out_hist = std::ptr::null_mut();
    let ok = unsafe {
        sys::mlx_inkling2_sconv_k4_residual_cuda(
            x.as_raw_ptr(),
            weight.as_raw_ptr(),
            hist_in.as_raw_ptr(),
            res_ptr,
            taps_ptr,
            update_hist,
            &mut out_y,
            if update_hist {
                &mut out_hist
            } else {
                std::ptr::null_mut()
            },
        )
    };
    if !ok || out_y.is_null() || (update_hist && out_hist.is_null()) {
        unsafe {
            delete_if_non_null(out_y);
            delete_if_non_null(out_hist);
        }
        return Ok(None);
    }
    Ok(Some(SconvK4Output {
        y: MxArray::from_handle(out_y, "inkling2_sconv_k4_residual:y")?,
        hist: if update_hist {
            Some(MxArray::from_handle(
                out_hist,
                "inkling2_sconv_k4_residual:hist",
            )?)
        } else {
            None
        },
    }))
}

/// 5. Fused SymQ8 / AffineQ8 / BF16 LM Head + Softcap (+ zero-DRAM greedy argmax).
///
/// - SymQ8: `scales = Some(...)`, `biases = None`
/// - AffineQ8: `scales = Some(...)`, `biases = Some(...)`
/// - Dense BF16: `scales = None`, `biases = None`
#[inline]
pub fn lm_head_softcap_cuda(
    x: &MxArray,
    w: &MxArray,
    scales: Option<&MxArray>,
    biases: Option<&MxArray>,
    unpadded_vocab: i32,
    softcap: f32,
    out_mode: LmHeadOutMode,
) -> Result<Option<MxArray>> {
    if !sm80_available() {
        return Ok(None);
    }
    if scales.is_none() && biases.is_some() {
        return Ok(None);
    }
    let scales_ptr = scales.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    let biases_ptr = biases.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    let handle = unsafe {
        sys::mlx_inkling2_lm_head_softcap_cuda(
            x.as_raw_ptr(),
            w.as_raw_ptr(),
            scales_ptr,
            biases_ptr,
            unpadded_vocab,
            softcap,
            out_mode as i32,
        )
    };
    if handle.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(
            handle,
            "inkling2_lm_head_softcap_cuda",
        )?))
    }
}

/// 6. Inkling-2-Mini Decode / Speculative Verify Split-K Tensor Core SDPA (`T in 1..=16`).
#[inline]
pub fn sdpa_few_cuda(params: &SdpaParams<'_>) -> Result<Option<MxArray>> {
    if !sm80_available() {
        return Ok(None);
    }
    if params.k_scales.is_some() != params.v_scales.is_some()
        || params.tree_depths.is_some() != params.tree_mask.is_some()
    {
        return Ok(None);
    }
    let ks_ptr = params
        .k_scales
        .map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    let vs_ptr = params
        .v_scales
        .map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    let td_ptr = params
        .tree_depths
        .map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    let tm_ptr = params
        .tree_mask
        .map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    let handle = unsafe {
        sys::mlx_inkling2_sdpa_few_cuda(
            params.q.as_raw_ptr(),
            params.r.as_raw_ptr(),
            params.k_cache.as_raw_ptr(),
            params.v_cache.as_raw_ptr(),
            ks_ptr,
            vs_ptr,
            params.rel_bias_w.as_raw_ptr(),
            params.gate.as_raw_ptr(),
            params.sink.as_raw_ptr(),
            td_ptr,
            tm_ptr,
            params.start_pos,
            params.cache_len,
            params.kv_start_pos,
            params.kv_phys_base,
            params.write_offset,
            params.ring_cap,
            params.is_local,
            params.sliding_window,
        )
    };
    if handle.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(
            handle,
            "inkling2_sdpa_few_cuda",
        )?))
    }
}

/// 7. Inkling-2-Mini FlashAttention Prefill Tensor Core SDPA (`T >= 1`, linear causal/SWA).
///
/// Declines cleanly with `Ok(None)` if tree mask/depths are supplied (use [`sdpa_few_cuda`]
/// for tree verification).
#[inline]
pub fn sdpa_prefill_cuda(params: &SdpaParams<'_>) -> Result<Option<MxArray>> {
    if !sm80_available() {
        return Ok(None);
    }
    if params.k_scales.is_some() != params.v_scales.is_some()
        || params.tree_depths.is_some()
        || params.tree_mask.is_some()
    {
        return Ok(None);
    }
    let ks_ptr = params
        .k_scales
        .map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    let vs_ptr = params
        .v_scales
        .map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    let handle = unsafe {
        sys::mlx_inkling2_sdpa_prefill_cuda(
            params.q.as_raw_ptr(),
            params.r.as_raw_ptr(),
            params.k_cache.as_raw_ptr(),
            params.v_cache.as_raw_ptr(),
            ks_ptr,
            vs_ptr,
            params.rel_bias_w.as_raw_ptr(),
            params.gate.as_raw_ptr(),
            params.sink.as_raw_ptr(),
            params.start_pos,
            params.cache_len,
            params.kv_start_pos,
            params.kv_phys_base,
            params.write_offset,
            params.ring_cap,
            params.is_local,
            params.sliding_window,
        )
    };
    if handle.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(
            handle,
            "inkling2_sdpa_prefill_cuda",
        )?))
    }
}

/// 8. Inkling-2-Mini Fused MoE (`E=128, top_k=8, H=2304, I=1152`): router + top-8 +
///    grouped `gate_up` + SiTU(4, 25) + grouped `down` + combine + optional fused
///    post ShortConv `k=4` (linear or tree) + residual + rollback pre-conv tape.
pub fn moe_forward_cuda(params: &MoeForwardParams<'_>) -> Result<Option<MoeForwardOutput>> {
    if !sm80_available() {
        return Ok(None);
    }
    if params.dn_mode == MoeQuantMode::Q5K {
        return Ok(None);
    }
    // Prevent silent drop of `residual` or `conv_taps` when `post_conv_w` is absent.
    if params.post_conv_w.is_none()
        && (params.hist_in.is_some()
            || params.residual.is_some()
            || params.conv_taps.is_some()
            || params.update_hist)
    {
        return Ok(None);
    }
    if params.post_conv_w.is_some() && params.hist_in.is_none() {
        return Ok(None);
    }

    let router_bias_ptr = params
        .router_bias
        .map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    let gu_biases_ptr = params
        .gu_biases
        .map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    let gu_outer_ptr = params
        .gu_outer_scale
        .map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    let dn_biases_ptr = params
        .dn_biases
        .map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    let dn_outer_ptr = params
        .dn_outer_scale
        .map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    let post_conv_ptr = params
        .post_conv_w
        .map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    let hist_in_ptr = params
        .hist_in
        .map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    let residual_ptr = params
        .residual
        .map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    let taps_ptr = params
        .conv_taps
        .map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);

    let want_hist = params.post_conv_w.is_some() && params.update_hist;
    let mut out_y = std::ptr::null_mut();
    let mut out_hist = std::ptr::null_mut();
    let mut out_pre_conv = std::ptr::null_mut();

    let ok = unsafe {
        sys::mlx_inkling2_moe_forward_cuda(
            params.x.as_raw_ptr(),
            params.router_w.as_raw_ptr(),
            router_bias_ptr,
            params.gu_w.as_raw_ptr(),
            params.gu_scales.as_raw_ptr(),
            gu_biases_ptr,
            gu_outer_ptr,
            params.gu_mode as i32,
            params.gu_interleaved,
            params.dn_w.as_raw_ptr(),
            params.dn_scales.as_raw_ptr(),
            dn_biases_ptr,
            dn_outer_ptr,
            params.dn_mode as i32,
            params.route_scale,
            post_conv_ptr,
            hist_in_ptr,
            residual_ptr,
            taps_ptr,
            params.update_hist,
            params.emit_pre_conv,
            &mut out_y,
            if want_hist {
                &mut out_hist
            } else {
                std::ptr::null_mut()
            },
            if params.emit_pre_conv {
                &mut out_pre_conv
            } else {
                std::ptr::null_mut()
            },
        )
    };

    if !ok
        || out_y.is_null()
        || (want_hist && out_hist.is_null())
        || (params.emit_pre_conv && out_pre_conv.is_null())
    {
        unsafe {
            delete_if_non_null(out_y);
            delete_if_non_null(out_hist);
            delete_if_non_null(out_pre_conv);
        }
        return Ok(None);
    }

    Ok(Some(MoeForwardOutput {
        y: MxArray::from_handle(out_y, "inkling2_moe_forward_cuda:y")?,
        hist: if want_hist {
            Some(MxArray::from_handle(
                out_hist,
                "inkling2_moe_forward_cuda:hist",
            )?)
        } else {
            None
        },
        pre_conv: if params.emit_pre_conv {
            Some(MxArray::from_handle(
                out_pre_conv,
                "inkling2_moe_forward_cuda:pre_conv",
            )?)
        } else {
            None
        },
    }))
}

/// 9. High-occupancy small-N BF16 projection (`N in 1..=128`, `M in 1..=16`, `K % 8 == 0`, `K >= 8`).
///
/// Replaces under-occupied 3-CTA `bf16_gemm_few` on `sink_proj [24, 2304]` and Global
/// `{k,v}_conv_proj [32, 2304]` with 1 CTA per output row, cache-streaming 128-bit weight
/// loads, and exact FP32 accumulation -> BF16.
#[inline]
pub fn small_bf16_proj_cuda(x: &MxArray, weight: &MxArray) -> Result<Option<MxArray>> {
    if !available() {
        return Ok(None);
    }
    let handle =
        unsafe { sys::mlx_inkling2_small_bf16_proj_cuda(x.as_raw_ptr(), weight.as_raw_ptr()) };
    if handle.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(
            handle,
            "inkling2_small_bf16_proj_cuda",
        )?))
    }
}

/// 10. Batched ShortConv `k=4` rollback tape commit (up to `128` streams; 1 CUDA launch per `<= 96` streams).
///
/// Reads potentially strided `rows_in[s]` (`[T, C]` BF16) and `hists_in[s]` (`[3, C]` BF16)
/// directly and writes contiguous `[3, C]` BF16 rolling histories for every stream:
/// - Tree mode (`accepted_nodes = Some(idx)`): commits `[keep]` arbitrary Int32 node indices.
/// - Linear / contiguous mode (`accepted_nodes = None`): commits prefix `0..contiguous_keep`.
pub fn commit_short_conv_tapes_cuda(
    hists_in: &[&MxArray],
    rows_in: &[&MxArray],
    accepted_nodes: Option<&MxArray>,
    contiguous_keep: usize,
) -> Result<Option<Vec<MxArray>>> {
    let n = hists_in.len();
    if !available() || n == 0 || n != rows_in.len() || n > 128 {
        return Ok(None);
    }
    let keep_i32 = if accepted_nodes.is_some() {
        0
    } else {
        match i32::try_from(contiguous_keep) {
            Ok(v) => v,
            Err(_) => return Ok(None),
        }
    };
    let h_ptrs: Vec<*mut sys::mlx_array> = hists_in.iter().map(|a| a.as_raw_ptr()).collect();
    let r_ptrs: Vec<*mut sys::mlx_array> = rows_in.iter().map(|a| a.as_raw_ptr()).collect();
    let nodes_ptr = accepted_nodes.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    let mut out_ptrs: Vec<*mut sys::mlx_array> = vec![std::ptr::null_mut(); n];

    let ok = unsafe {
        sys::mlx_inkling2_commit_short_conv_tapes_cuda(
            h_ptrs.as_ptr(),
            r_ptrs.as_ptr(),
            n,
            nodes_ptr,
            keep_i32,
            out_ptrs.as_mut_ptr(),
        )
    };
    if !ok || out_ptrs.iter().any(|p| p.is_null()) {
        for ptr in out_ptrs {
            unsafe { delete_if_non_null(ptr) };
        }
        return Ok(None);
    }

    let mut outs = Vec::with_capacity(n);
    for (idx, ptr) in out_ptrs.iter_mut().enumerate() {
        let handle = std::mem::replace(ptr, std::ptr::null_mut());
        match MxArray::from_handle(handle, "inkling2_commit_short_conv_tapes_cuda") {
            Ok(arr) => outs.push(arr),
            Err(err) => {
                for rem in &out_ptrs[idx + 1..] {
                    unsafe { delete_if_non_null(*rem) };
                }
                return Err(err);
            }
        }
    }
    Ok(Some(outs))
}

/// 11. Fused token embedding lookup + precise FP32 RMSNorm -> BF16.
///
/// Supports 1D `[T]` and 2D `[1, T]` Int32/Uint32 `token_ids` across:
/// - Dense BF16 (`scales = None`, `biases = None`)
/// - Symmetric Q8_0 (`bits = 8, group_size = 32`, `scales = Some(...)`, `biases = None`)
/// - Affine Q8_0 (`bits = 8, group_size = 32`, `scales = Some(...)`, `biases = Some(...)`)
#[inline]
pub fn embed_rmsnorm_cuda(
    token_ids: &MxArray,
    weight: &MxArray,
    scales: Option<&MxArray>,
    biases: Option<&MxArray>,
    norm_weight: &MxArray,
    eps: f32,
) -> Result<Option<MxArray>> {
    if !available() || (scales.is_none() && biases.is_some()) {
        return Ok(None);
    }
    let scales_ptr = scales.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    let biases_ptr = biases.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    let handle = unsafe {
        sys::mlx_inkling2_embed_rmsnorm_cuda(
            token_ids.as_raw_ptr(),
            weight.as_raw_ptr(),
            scales_ptr,
            biases_ptr,
            norm_weight.as_raw_ptr(),
            eps,
        )
    };
    if handle.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(
            handle,
            "inkling2_embed_rmsnorm_cuda",
        )?))
    }
}

/// 12. Batched race-free in-place KV cache compaction across up to `192` non-overlapping caches
///     (1 CUDA launch per `<= 96` caches; supports mixed BF16/F16/INT8/F32 and 4D `[1, H_kv, cap, D]`
///     or 3D `[1, H_kv, cap]` scale caches sharing `H_kv`).
///
/// Returns `caches.len()` graph-dependency aliases that share each input cache's GPU buffer and
/// must be propagated back into the cache state so downstream reads and `eval_boundary()` observe
/// the compacted rows.
pub fn compact_kv_batch_cuda(
    caches: &[&MxArray],
    write_offsets: &[i32],
    accepted_nodes: &MxArray,
) -> Result<Option<Vec<MxArray>>> {
    let n = caches.len();
    if !available() || n == 0 || n != write_offsets.len() || n > 192 {
        return Ok(None);
    }
    let cache_ptrs: Vec<*mut sys::mlx_array> = caches.iter().map(|c| c.as_raw_ptr()).collect();
    let mut out_ptrs: Vec<*mut sys::mlx_array> = vec![std::ptr::null_mut(); n];
    let ok = unsafe {
        sys::mlx_inkling2_compact_kv_batch_cuda(
            cache_ptrs.as_ptr(),
            n,
            write_offsets.as_ptr(),
            accepted_nodes.as_raw_ptr(),
            out_ptrs.as_mut_ptr(),
        )
    };
    if !ok || out_ptrs.iter().any(|p| p.is_null()) {
        for ptr in out_ptrs {
            unsafe { delete_if_non_null(ptr) };
        }
        return Ok(None);
    }
    let mut outs = Vec::with_capacity(n);
    for (idx, ptr) in out_ptrs.iter_mut().enumerate() {
        let handle = std::mem::replace(ptr, std::ptr::null_mut());
        match MxArray::from_handle(handle, "inkling2_compact_kv_batch_cuda") {
            Ok(arr) => outs.push(arr),
            Err(err) => {
                for rem in &out_ptrs[idx + 1..] {
                    unsafe { delete_if_non_null(*rem) };
                }
                return Err(err);
            }
        }
    }
    Ok(Some(outs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accelerator::host_has_cuda;
    use crate::array::DType;
    use crate::models::inkling2::attention::{global_log_scale_tau, l2_clip};
    use crate::models::inkling2::attn_res::{
        ATTN_RES_EPS, LoopController, LoopLayerControllers, apply_sublayer_rms_norm,
        block_attn_res_mix,
    };
    use crate::models::inkling2::moe::situ_activation;
    use crate::models::inkling2::short_conv::{dynamic_short_conv, static_short_conv};
    use crate::moe::sigmoid_biased_topk_from_logits;
    use crate::nn::Activations;
    use crate::nn::normalization::rms_norm_unscaled;
    use half::{bf16, f16};

    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0
        }
        fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
            let u = ((self.next_u64() >> 40) as f32) / 16_777_216.0;
            lo + (hi - lo) * u
        }
        fn vec(&mut self, len: usize, scale: f32) -> Vec<f32> {
            (0..len).map(|_| self.uniform(-scale, scale)).collect()
        }
    }

    fn bf16_arr(vals: &[f32], shape: &[i64]) -> MxArray {
        let bits: Vec<u16> = vals.iter().map(|&v| bf16::from_f32(v).to_bits()).collect();
        MxArray::from_bfloat16(&bits, shape).unwrap()
    }

    fn f32_arr(vals: &[f32], shape: &[i64]) -> MxArray {
        MxArray::from_float32(vals, shape).unwrap()
    }

    fn err_stats(got: &[f32], want: &[f32]) -> (f32, f32, f32) {
        assert_eq!(got.len(), want.len(), "length mismatch");
        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;
        let mut diff_sq = 0.0f64;
        let mut ref_sq = 0.0f64;
        for (&g, &w) in got.iter().zip(want.iter()) {
            assert!(g.is_finite(), "non-finite got={g} (want={w})");
            assert!(w.is_finite(), "non-finite want={w}");
            let d = (g - w).abs();
            max_abs = max_abs.max(d);
            max_rel = max_rel.max(d / w.abs().max(1.0));
            diff_sq += f64::from(d) * f64::from(d);
            ref_sq += f64::from(w) * f64::from(w);
        }
        let rel_l2 = (diff_sq / (ref_sq + 1e-12)).sqrt() as f32;
        (max_abs, max_rel, rel_l2)
    }

    fn assert_close(label: &str, got: &[f32], want: &[f32], max_abs_tol: f32, rel_l2_tol: f32) {
        let (max_abs, max_rel, rel_l2) = err_stats(got, want);
        if max_abs > max_abs_tol || rel_l2 > rel_l2_tol {
            let (wi, (&wg, &ww)) = got
                .iter()
                .zip(want.iter())
                .enumerate()
                .max_by(|(_, (a0, b0)), (_, (a1, b1))| {
                    (*a0 - *b0).abs().total_cmp(&(*a1 - *b1).abs())
                })
                .unwrap();
            panic!(
                "{label}: max_abs={max_abs:.6} (tol {max_abs_tol}) at idx={wi} (got={wg}, want={ww}), max_rel={max_rel:.6}, rel_l2={rel_l2:.6} (tol {rel_l2_tol})"
            );
        }
    }

    fn quantize_mlx(
        w: &MxArray,
        group_size: i32,
        bits: i32,
        mode: &std::ffi::CStr,
    ) -> (MxArray, MxArray, Option<MxArray>) {
        let mut out_q = std::ptr::null_mut();
        let mut out_s = std::ptr::null_mut();
        let mut out_b = std::ptr::null_mut();
        let ok = unsafe {
            sys::mlx_quantize(
                w.as_raw_ptr(),
                group_size,
                bits,
                mode.as_ptr(),
                &mut out_q,
                &mut out_s,
                &mut out_b,
            )
        };
        assert!(ok, "mlx_quantize failed for {mode:?}");
        let q = MxArray::from_handle(out_q, "q").unwrap();
        let s = MxArray::from_handle(out_s, "s").unwrap();
        let b = if out_b.is_null() {
            None
        } else {
            Some(MxArray::from_handle(out_b, "b").unwrap())
        };
        (q, s, b)
    }

    fn dequantize_mlx(
        w: &MxArray,
        scales: &MxArray,
        biases: Option<&MxArray>,
        group_size: i32,
        bits: i32,
        mode: &std::ffi::CStr,
    ) -> MxArray {
        // For affine mode, upcast F16/BF16 scales & biases to FP32 so mlx_dequantize
        // computes `q * scale + bias` in exact FP32 instead of truncating 8-bit
        // quantized weights down to 7-bit-mantissa BF16.
        let (s_owned, b_owned, out_dtype) = if mode == c"affine" {
            (
                Some(scales.astype(DType::Float32).unwrap()),
                biases.map(|b| b.astype(DType::Float32).unwrap()),
                -1,
            )
        } else {
            (None, None, DType::Float32 as i32)
        };
        let s_ref = s_owned.as_ref().unwrap_or(scales);
        let b_ptr = match (b_owned.as_ref(), biases) {
            (Some(b), _) => b.as_raw_ptr(),
            (None, Some(b)) => b.as_raw_ptr(),
            (None, None) => std::ptr::null_mut(),
        };
        let h = unsafe {
            sys::mlx_dequantize(
                w.as_raw_ptr(),
                s_ref.as_raw_ptr(),
                b_ptr,
                group_size,
                bits,
                out_dtype,
                mode.as_ptr(),
            )
        };
        assert!(!h.is_null(), "mlx_dequantize failed for {mode:?}");
        MxArray::from_handle(h, "deq")
            .unwrap()
            .astype(DType::Float32)
            .unwrap()
    }

    /// Exact group-32 symmetric Q8_0 quantization:
    /// `byte = (clamp(round(w / s), -127, 127) + 128) as u8`, `s = f16(amax / 127)`,
    /// `b_ref = f16(-128 * s)`. Returns `(w_u32 [rows, k/4], s_f16 [rows, k/32], b_ref [rows, k/32])`.
    fn quantize_sym_q8_f16(w_vals: &[f32], rows: usize, k: usize) -> (MxArray, MxArray, MxArray) {
        assert_eq!(k % 32, 0);
        let groups = k / 32;
        let mut w_u32 = vec![0u32; rows * (k / 4)];
        let mut s_f32 = vec![0.0f32; rows * groups];
        for r in 0..rows {
            for g in 0..groups {
                let base = r * k + g * 32;
                let grp = &w_vals[base..base + 32];
                let amax = grp.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                let s = f16::from_f32(if amax > 0.0 { amax / 127.0 } else { 0.0 }).to_f32();
                s_f32[r * groups + g] = s;
                let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
                for u in 0..8 {
                    let mut word = 0u32;
                    for lane in 0..4 {
                        let q_i8 = if s > 0.0 {
                            (grp[u * 4 + lane] * inv).round().clamp(-127.0, 127.0) as i32
                        } else {
                            0
                        };
                        let byte = (q_i8 + 128) as u32;
                        word |= (byte & 0xFF) << (lane * 8);
                    }
                    w_u32[r * (k / 4) + g * 8 + u] = word;
                }
            }
        }
        let w = MxArray::from_uint32(&w_u32, &[rows as i64, (k / 4) as i64]).unwrap();
        let s = f32_arr(&s_f32, &[rows as i64, groups as i64])
            .astype(DType::Float16)
            .unwrap();
        let b = s
            .astype(DType::Float32)
            .unwrap()
            .mul_scalar(-128.0)
            .unwrap()
            .astype(DType::Float16)
            .unwrap();
        (w, s, b)
    }

    /// Reference static k=4 ShortConv supporting arbitrary tree `conv_taps` (`[T, 4]` oldest-to-newest).
    fn ref_sconv_k4_tree(
        x_bf16: &[f32],
        weight: &[f32],
        hist_in: &[f32],
        residual: Option<&[f32]>,
        taps: &[[i32; 4]],
        c: usize,
    ) -> Vec<f32> {
        let t_len = taps.len();
        let scale = 1.0f32 / 5.0f32.sqrt();
        let mut out = vec![0.0f32; t_len * c];
        let load = |idx: i32, ch: usize| -> f32 {
            if idx < 0 {
                let hr = 3 + idx;
                if (0..3).contains(&hr) {
                    hist_in[(hr as usize) * c + ch]
                } else {
                    0.0
                }
            } else if (idx as usize) < t_len {
                x_bf16[(idx as usize) * c + ch]
            } else {
                0.0
            }
        };
        for (t, row_taps) in taps.iter().enumerate() {
            for ch in 0..c {
                let x3 = load(row_taps[0], ch);
                let x2 = load(row_taps[1], ch);
                let x1 = load(row_taps[2], ch);
                let x0 = load(row_taps[3], ch);
                let w = &weight[ch * 4..ch * 4 + 4];
                let acc = x0 + w[0] * x3 + w[1] * x2 + w[2] * x1 + w[3] * x0;
                let conv_bf = bf16::from_f32(acc * scale).to_f32();
                out[t * c + ch] = match residual {
                    Some(r) => bf16::from_f32(r[t * c + ch] + conv_bf).to_f32(),
                    None => conv_bf,
                };
            }
        }
        out
    }

    /// Reference dynamic k=4 ShortConv supporting arbitrary tree `conv_taps` (`[T, 4]`).
    fn ref_dyn_sconv_k4_tree(
        x_bf16: &[f32],
        weight: &[f32],
        dyn_taps: &[f32],
        hist_in: &[f32],
        taps: &[[i32; 4]],
        n_kv: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let c = n_kv * head_dim;
        let t_len = taps.len();
        let scale = 1.0f32 / 3.0f32;
        let mut out = vec![0.0f32; t_len * c];
        let load = |idx: i32, ch: usize| -> f32 {
            if idx < 0 {
                let hr = 3 + idx;
                if (0..3).contains(&hr) {
                    hist_in[(hr as usize) * c + ch]
                } else {
                    0.0
                }
            } else if (idx as usize) < t_len {
                x_bf16[(idx as usize) * c + ch]
            } else {
                0.0
            }
        };
        for (t, row_taps) in taps.iter().enumerate() {
            for h in 0..n_kv {
                let d_base = (t * n_kv + h) * 4;
                let d0 = dyn_taps[d_base];
                let d1 = dyn_taps[d_base + 1];
                let d2 = dyn_taps[d_base + 2];
                let d3 = dyn_taps[d_base + 3];
                for j in 0..head_dim {
                    let ch = h * head_dim + j;
                    let x3 = load(row_taps[0], ch);
                    let x2 = load(row_taps[1], ch);
                    let x1 = load(row_taps[2], ch);
                    let x0 = load(row_taps[3], ch);
                    let w = &weight[ch * 4..ch * 4 + 4];
                    let acc = x0
                        + (w[0] + d3) * x3
                        + (w[1] + d2) * x2
                        + (w[2] + d1) * x1
                        + (w[3] + d0) * x0;
                    out[t * c + ch] = bf16::from_f32(acc * scale).to_f32();
                }
            }
        }
        out
    }

    // =========================================================================
    // Test 1: Block AttnRes + optional fused sublayer RMSNorm
    // =========================================================================
    #[test]
    fn cuda_inkling2_block_attn_res_norm_parity() {
        if !host_has_cuda() || !available() {
            return;
        }
        let mut rng = Rng::new(0xA771_2E51);

        for &d in &[2304usize, 130usize] {
            for &t in &[1usize, 8, 37] {
                // Build 6 mixed BF16 / FP32 sources so we cover both input dtypes.
                let mut src_arrays = Vec::with_capacity(6);
                for s_idx in 0..6usize {
                    let vals = rng.vec(t * d, 1.5);
                    if s_idx % 2 == 0 {
                        src_arrays.push(bf16_arr(&vals, &[t as i64, d as i64]));
                    } else {
                        src_arrays.push(f32_arr(&vals, &[t as i64, d as i64]));
                    }
                }
                for &n in &[1usize, 2, 4, 6] {
                    let srcs: Vec<&MxArray> = src_arrays[..n].iter().collect();
                    for &q_cnt in &[1usize, 3, 4] {
                        let scale = 1.0f32 / (d as f32).sqrt();
                        let q_vals: Vec<f32> = rng
                            .vec(q_cnt * d, 1.0)
                            .into_iter()
                            .map(|v| v * scale)
                            .collect();
                        let queries = f32_arr(&q_vals, &[q_cnt as i64, d as i64]);
                        let ref_mix = block_attn_res_mix(&srcs, &queries).unwrap();
                        let ref_mix_f32 = ref_mix.to_float32().unwrap();

                        // Exact IEEE FP32 CPU reference (MLX GPU `matmul` uses TF32 on sm_80+,
                        // which has ~1e-3 rounding, whereas our CUDA kernel is true IEEE FP32).
                        let src_f32_vecs: Vec<Vec<f32>> = srcs
                            .iter()
                            .map(|s| s.to_float32().unwrap().as_ref().to_vec())
                            .collect();
                        let mut ieee_ref = vec![0.0f32; q_cnt * t * d];
                        for tok in 0..t {
                            let mut normed = vec![vec![0.0f32; d]; n];
                            for s_i in 0..n {
                                let row = &src_f32_vecs[s_i][tok * d..(tok + 1) * d];
                                let mean_sq = row
                                    .iter()
                                    .map(|&v| f64::from(v) * f64::from(v))
                                    .sum::<f64>()
                                    / (d as f64);
                                let inv = (1.0 / (mean_sq + f64::from(ATTN_RES_EPS)).sqrt()) as f32;
                                for j in 0..d {
                                    normed[s_i][j] = row[j] * inv;
                                }
                            }
                            for qi in 0..q_cnt {
                                let q_row = &q_vals[qi * d..(qi + 1) * d];
                                let mut scores = vec![0.0f64; n];
                                let mut max_s = f64::NEG_INFINITY;
                                for s_i in 0..n {
                                    let sc: f64 = normed[s_i]
                                        .iter()
                                        .zip(q_row)
                                        .map(|(&a, &b)| f64::from(a) * f64::from(b))
                                        .sum();
                                    scores[s_i] = sc;
                                    max_s = max_s.max(sc);
                                }
                                let exps: Vec<f64> =
                                    scores.iter().map(|&s| (s - max_s).exp()).collect();
                                let sum_e: f64 = exps.iter().sum();
                                let dst = &mut ieee_ref[(qi * t + tok) * d..(qi * t + tok + 1) * d];
                                for s_i in 0..n {
                                    let p = (exps[s_i] / sum_e) as f32;
                                    for j in 0..d {
                                        dst[j] += p * normed[s_i][j];
                                    }
                                }
                            }
                        }

                        // 1. Without norm_weights (FP32 output)
                        for &squeeze in &[false, true] {
                            let got = block_attn_res_norm_cuda(
                                &srcs,
                                &queries,
                                None,
                                ATTN_RES_EPS,
                                squeeze,
                            )
                            .unwrap()
                            .expect("block_attn_res_norm_cuda must run on CUDA");
                            let expected_shape: Vec<i64> = if q_cnt == 1 && squeeze {
                                vec![t as i64, d as i64]
                            } else {
                                vec![q_cnt as i64, t as i64, d as i64]
                            };
                            assert_eq!(got.shape().unwrap().as_ref(), expected_shape.as_slice());
                            assert_eq!(got.dtype().unwrap(), DType::Float32);
                            let got_f32 = got.to_float32().unwrap();
                            // Tight IEEE FP32 check (< 2e-5) proves kernel math is exact FP32!
                            assert_close(
                                &format!(
                                    "block_attn_res ieee_f32 D={d} T={t} N={n} Q={q_cnt} sq={squeeze}"
                                ),
                                got_f32.as_ref(),
                                &ieee_ref,
                                2e-5,
                                2e-5,
                            );
                            // Also verify close match against MLX `block_attn_res_mix` (TF32 matmul).
                            assert_close(
                                &format!(
                                    "block_attn_res mlx_ref D={d} T={t} N={n} Q={q_cnt} sq={squeeze}"
                                ),
                                got_f32.as_ref(),
                                ref_mix_f32.as_ref(),
                                2e-3,
                                1e-3,
                            );
                        }

                        // 2. With norm_weights: broadcast 1D [D] and per-query 2D [Q, D] (BF16 output)
                        let nw_1d_vals: Vec<f32> = (0..d).map(|_| rng.uniform(0.5, 1.5)).collect();
                        let nw_1d = f32_arr(&nw_1d_vals, &[d as i64]);
                        let nw_2d_vals: Vec<f32> =
                            (0..q_cnt * d).map(|_| rng.uniform(0.5, 1.5)).collect();
                        let nw_2d = f32_arr(&nw_2d_vals, &[q_cnt as i64, d as i64]);

                        for (nw_label, nw_arr, bcast) in
                            [("bcast1d", &nw_1d, true), ("per_q2d", &nw_2d, false)]
                        {
                            let mut ref_norm_all = Vec::with_capacity(q_cnt * t * d);
                            for qi in 0..q_cnt {
                                let slice_q = ref_mix
                                    .slice(
                                        &[qi as i64, 0, 0],
                                        &[(qi + 1) as i64, t as i64, d as i64],
                                    )
                                    .unwrap()
                                    .squeeze(Some(&[0]))
                                    .unwrap();
                                let w_q = if bcast {
                                    nw_1d.clone()
                                } else {
                                    nw_2d
                                        .slice(&[qi as i64, 0], &[(qi + 1) as i64, d as i64])
                                        .unwrap()
                                        .squeeze(Some(&[0]))
                                        .unwrap()
                                };
                                let n_q = apply_sublayer_rms_norm(
                                    &slice_q,
                                    &w_q,
                                    ATTN_RES_EPS,
                                    DType::BFloat16,
                                )
                                .unwrap();
                                ref_norm_all.extend_from_slice(n_q.to_float32().unwrap().as_ref());
                            }
                            let squeeze = q_cnt == 1;
                            let got = block_attn_res_norm_cuda(
                                &srcs,
                                &queries,
                                Some(nw_arr),
                                ATTN_RES_EPS,
                                squeeze,
                            )
                            .unwrap()
                            .expect("block_attn_res_norm_cuda(norm) must run on CUDA");
                            assert_eq!(got.dtype().unwrap(), DType::BFloat16);
                            let got_f32 = got.to_float32().unwrap();
                            assert_close(
                                &format!(
                                    "block_attn_res norm({nw_label}) D={d} T={t} N={n} Q={q_cnt}"
                                ),
                                got_f32.as_ref(),
                                &ref_norm_all,
                                2e-2,
                                5e-3,
                            );
                        }
                    }
                }
            }
        }
    }

    // =========================================================================
    // Test 2: Fused LoopController + Sublayer RMSNorm
    // =========================================================================
    #[test]
    fn cuda_inkling2_loop_ctrl_norm_parity() {
        if !host_has_cuda() || !available() {
            return;
        }
        let mut rng = Rng::new(0x100B_C721);
        let d = 2304usize;
        let rank = 64usize;

        for &w_bf16 in &[false, true] {
            let make_w = |vals: &[f32], shape: &[i64]| -> MxArray {
                if w_bf16 {
                    bf16_arr(vals, shape)
                } else {
                    f32_arr(vals, shape)
                }
            };
            let make_ctrl = |rng: &mut Rng| -> LoopController {
                let down = make_w(&rng.vec(rank * 2 * d, 0.04), &[rank as i64, (2 * d) as i64]);
                let gain = make_w(&rng.vec(d * rank, 0.04), &[d as i64, rank as i64]);
                let forget = make_w(&rng.vec(d * rank, 0.04), &[d as i64, rank as i64]);
                LoopController::new(down, gain, forget).unwrap()
            };
            let q_ctrl = make_ctrl(&mut rng);
            let k_ctrl = make_ctrl(&mut rng);
            let v_ctrl = make_ctrl(&mut rng);
            let mlp_ctrl = make_ctrl(&mut rng);
            let msg_q = f32_arr(&rng.vec(4 * d, 0.1), &[4, d as i64]);
            let layer_ctrls =
                LoopLayerControllers::new(msg_q, q_ctrl, k_ctrl, v_ctrl, mlp_ctrl).unwrap();

            let norm_1d = f32_arr(
                &(0..d).map(|_| rng.uniform(0.6, 1.4)).collect::<Vec<_>>(),
                &[d as i64],
            );
            let norm_3d = f32_arr(
                &(0..3 * d)
                    .map(|_| rng.uniform(0.6, 1.4))
                    .collect::<Vec<_>>(),
                &[3, d as i64],
            );

            for &t in &[1usize, 8, 32] {
                // --- G = 1 (mlp_ctrl, squeeze_g1 = true & false) ---
                let dest_2d = f32_arr(&rng.vec(t * d, 1.2), &[t as i64, d as i64]);
                let msg_2d = f32_arr(&rng.vec(t * d, 1.2), &[t as i64, d as i64]);
                let ref_g1 = layer_ctrls
                    .mlp_ctrl
                    .forward_fused_rms_norm(
                        &dest_2d,
                        &msg_2d,
                        &norm_1d,
                        ATTN_RES_EPS,
                        DType::BFloat16,
                    )
                    .unwrap()
                    .to_float32()
                    .unwrap();

                for &squeeze in &[true, false] {
                    let got = loop_ctrl_norm_cuda(
                        &dest_2d,
                        &msg_2d,
                        &layer_ctrls.mlp_ctrl.down_proj_w,
                        &layer_ctrls.mlp_ctrl.gain_forget_w,
                        &norm_1d,
                        ATTN_RES_EPS,
                        1,
                        squeeze,
                    )
                    .unwrap()
                    .expect("loop_ctrl_norm_cuda G=1 must run on CUDA");
                    let exp_shape: Vec<i64> = if squeeze {
                        vec![t as i64, d as i64]
                    } else {
                        vec![1, t as i64, d as i64]
                    };
                    assert_eq!(got.shape().unwrap().as_ref(), exp_shape.as_slice());
                    assert_eq!(got.dtype().unwrap(), DType::BFloat16);
                    let (abs_tol, rel_tol) = if w_bf16 { (0.04, 8e-3) } else { (0.02, 4e-3) };
                    assert_close(
                        &format!("loop_ctrl G=1 bf16={w_bf16} T={t} sq={squeeze}"),
                        got.to_float32().unwrap().as_ref(),
                        ref_g1.as_ref(),
                        abs_tol,
                        rel_tol,
                    );
                }

                // --- G = 3: broadcast 2D dest [T, D] + 1D norm AND 3D dest [3, T, D] + 2D norm ---
                let msgs_3d = f32_arr(&rng.vec(3 * t * d, 1.2), &[3, t as i64, d as i64]);
                let dest_3d = f32_arr(&rng.vec(3 * t * d, 1.2), &[3, t as i64, d as i64]);
                for (mode_lbl, d_arr, nw_arr) in [
                    ("bcast2d", &dest_2d, &norm_1d),
                    ("full3d", &dest_3d, &norm_3d),
                ] {
                    let (rq, rk, rv) = layer_ctrls
                        .forward_qkv_fused_rms_norm(
                            d_arr,
                            &msgs_3d,
                            nw_arr,
                            ATTN_RES_EPS,
                            DType::BFloat16,
                        )
                        .unwrap();
                    let mut ref_3 = Vec::with_capacity(3 * t * d);
                    ref_3.extend_from_slice(rq.to_float32().unwrap().as_ref());
                    ref_3.extend_from_slice(rk.to_float32().unwrap().as_ref());
                    ref_3.extend_from_slice(rv.to_float32().unwrap().as_ref());

                    let got = loop_ctrl_norm_cuda(
                        d_arr,
                        &msgs_3d,
                        &layer_ctrls.qkv_down_w,
                        &layer_ctrls.qkv_gain_forget_w,
                        nw_arr,
                        ATTN_RES_EPS,
                        3,
                        false,
                    )
                    .unwrap()
                    .expect("loop_ctrl_norm_cuda G=3 must run on CUDA");
                    assert_eq!(got.shape().unwrap().as_ref(), &[3, t as i64, d as i64]);
                    let (abs_tol, rel_tol) = if w_bf16 { (0.04, 8e-3) } else { (0.02, 4e-3) };
                    assert_close(
                        &format!("loop_ctrl G=3 {mode_lbl} bf16={w_bf16} T={t}"),
                        got.to_float32().unwrap().as_ref(),
                        &ref_3,
                        abs_tol,
                        rel_tol,
                    );
                }
            }
        }
    }

    // =========================================================================
    // Test 3: End-to-End kv_conv_prep_store + sdpa_few + sdpa_prefill
    //         (SWA ring_cap=1056 wrap-around, Global start_pos=9000 tau>1,
    //          BF16 KV & INT8 KV, linear & branching tree mode)
    // =========================================================================
    #[test]
    fn cuda_inkling2_kv_conv_prep_store_and_sdpa_parity() {
        if !host_has_cuda() || !sm80_available() {
            return;
        }
        let mut rng = Rng::new(0x5D9A_1234);
        let hq = 24usize;
        let hkv = 8usize;
        let gqa = hq / hkv;
        let hd = 128usize;
        let d_rel = 16usize;
        let rel_extent = 1024usize;
        let sliding_window = 1024usize;
        let qk_clip = 22.627_416_997_969_522_f64;
        let rel_clip_thresh = 8.0_f64;
        let log_n_floor = 8192usize;
        let log_alpha = 0.1_f64;

        let q_clip_vals: Vec<f32> = (0..hq * hd).map(|_| rng.uniform(0.7, 1.3)).collect();
        let rel_clip_vals: Vec<f32> = (0..hq * d_rel).map(|_| rng.uniform(0.7, 1.3)).collect();
        let k_conv_vals = rng.vec(hkv * hd * 4, 0.35);
        let v_conv_vals = rng.vec(hkv * hd * 4, 0.35);
        let rel_bias_vals = rng.vec(rel_extent * d_rel, 0.15);

        let q_clip = f32_arr(&q_clip_vals, &[1, hq as i64, hd as i64]);
        let rel_clip = f32_arr(&rel_clip_vals, &[1, hq as i64, d_rel as i64]);
        let k_conv_w = f32_arr(&k_conv_vals, &[(hkv * hd) as i64, 4]);
        let v_conv_w = f32_arr(&v_conv_vals, &[(hkv * hd) as i64, 4]);
        let rel_bias_w = f32_arr(&rel_bias_vals, &[rel_extent as i64, d_rel as i64]);
        let rel_bias_t = rel_bias_w.transpose(None).unwrap();

        // Reference SDPA over logical `[L, Hkv, D]` K/V (already dequantized when INT8)
        // supporting causal, SWA, and arbitrary tree `(depths, mask)`.
        #[allow(clippy::too_many_arguments)]
        let run_ref_sdpa = |q_1htd: &MxArray,
                            r_thr: &MxArray,
                            k_lhd: &MxArray,
                            v_lhd: &MxArray,
                            gate_2d: &MxArray,
                            sink_2d: &MxArray,
                            start_pos: usize,
                            cache_len: usize,
                            kv_start_pos: usize,
                            is_local: bool,
                            tree_depths: Option<&[i32]>,
                            tree_mask: Option<&[u32]>|
         -> Vec<f32> {
            let t = q_1htd.shape().unwrap()[2] as usize;
            let kv_len = k_lhd.shape().unwrap()[0] as usize;
            let q_htd = q_1htd
                .squeeze(Some(&[0]))
                .unwrap()
                .astype(DType::Float32)
                .unwrap(); // [Hq, T, D]
            let k_f32 = k_lhd.astype(DType::Float32).unwrap(); // [L, Hkv, D]
            let v_f32 = v_lhd.astype(DType::Float32).unwrap();
            let k_hld = k_f32
                .transpose(Some(&[1, 0, 2]))
                .unwrap()
                .reshape(&[hkv as i64, 1, kv_len as i64, hd as i64])
                .unwrap()
                .broadcast_to(&[hkv as i64, gqa as i64, kv_len as i64, hd as i64])
                .unwrap()
                .reshape(&[hq as i64, kv_len as i64, hd as i64])
                .unwrap();
            let v_hld = v_f32
                .transpose(Some(&[1, 0, 2]))
                .unwrap()
                .reshape(&[hkv as i64, 1, kv_len as i64, hd as i64])
                .unwrap()
                .broadcast_to(&[hkv as i64, gqa as i64, kv_len as i64, hd as i64])
                .unwrap()
                .reshape(&[hq as i64, kv_len as i64, hd as i64])
                .unwrap();

            let attn_scale = 1.0f64 / (hd as f64).sqrt();
            let scores_htl = q_htd
                .matmul(&k_hld.transpose(Some(&[0, 2, 1])).unwrap())
                .unwrap()
                .mul_scalar(attn_scale)
                .unwrap();
            let rel_logits = r_thr
                .matmul(&rel_bias_t)
                .unwrap()
                .transpose(Some(&[1, 0, 2]))
                .unwrap(); // [Hq, T, extent]

            // Build exact [T, L] dist + validity mask on host for clarity & tree generality.
            let mut dist_clamped = vec![0i32; t * kv_len];
            let mut in_rel = vec![0.0f32; t * kv_len];
            let mut valid = vec![0.0f32; t * kv_len];
            for qi in 0..t {
                let q_abs = start_pos as i32 + tree_depths.map_or(qi as i32, |d| d[qi]);
                let tmask = tree_mask.map_or(0u32, |m| m[qi]);
                for kj in 0..kv_len {
                    let is_new = kj >= cache_len;
                    let k_abs = if is_new {
                        let tj = kj - cache_len;
                        start_pos as i32 + tree_depths.map_or(tj as i32, |d| d[tj])
                    } else {
                        kv_start_pos as i32 + kj as i32
                    };
                    let dist = q_abs - k_abs;
                    let ok = if let Some(_) = tree_mask
                        && is_new
                    {
                        let tj = kj - cache_len;
                        ((tmask >> tj) & 1) != 0
                            && (!is_local || (dist >= 0 && dist < sliding_window as i32))
                    } else {
                        dist >= 0 && (!is_local || dist < sliding_window as i32)
                    };
                    if ok {
                        valid[qi * kv_len + kj] = 1.0;
                        if dist >= 0 && (dist as usize) < rel_extent {
                            dist_clamped[qi * kv_len + kj] = dist;
                            in_rel[qi * kv_len + kj] = 1.0;
                        }
                    }
                }
            }
            let dist_idx = MxArray::from_int32(&dist_clamped, &[1, t as i64, kv_len as i64])
                .unwrap()
                .broadcast_to(&[hq as i64, t as i64, kv_len as i64])
                .unwrap();
            let gathered = rel_logits.take_along_axis(&dist_idx, 2).unwrap();
            let rel_mask = f32_arr(&in_rel, &[1, t as i64, kv_len as i64]);
            let scores = scores_htl.add(&gathered.mul(&rel_mask).unwrap()).unwrap();

            let valid_mask = f32_arr(&valid, &[1, t as i64, kv_len as i64]);
            let neg_inf = MxArray::full(
                &[1, 1, 1],
                napi::Either::A(f64::NEG_INFINITY),
                Some(DType::Float32),
            )
            .unwrap();
            let masked = valid_mask
                .greater(&MxArray::scalar_float(0.5).unwrap())
                .unwrap()
                .where_(&scores, &neg_inf)
                .unwrap();

            let lse = masked.logsumexp(Some(&[-1]), Some(true)).unwrap(); // [Hq, T, 1]
            let probs = masked.sub(&lse).unwrap().exp().unwrap();
            let attn_o = probs
                .matmul(&v_hld)
                .unwrap()
                .transpose(Some(&[1, 0, 2]))
                .unwrap(); // [T, Hq, D]
            let lse_th = lse.squeeze(Some(&[-1])).unwrap().transpose(None).unwrap(); // [T, Hq]
            let sink_f32 = sink_2d.astype(DType::Float32).unwrap();
            let sink_scale = Activations::sigmoid(&lse_th.sub(&sink_f32).unwrap())
                .unwrap()
                .expand_dims(-1)
                .unwrap();
            let gate_scale = Activations::sigmoid(
                &gate_2d
                    .astype(DType::Float32)
                    .unwrap()
                    .reshape(&[t as i64, hq as i64, hd as i64])
                    .unwrap(),
            )
            .unwrap();
            let y = attn_o
                .mul(&gate_scale)
                .unwrap()
                .mul(&sink_scale)
                .unwrap()
                .astype(DType::BFloat16)
                .unwrap()
                .reshape(&[t as i64, (hq * hd) as i64])
                .unwrap();
            y.to_float32().unwrap().as_ref().to_vec()
        };

        // Helper to quantize `[L, Hkv, D]` BF16 -> `(int8 [L, Hkv, D], f32 scales [L, Hkv])`
        // matching `ink2_kv_conv_prep_store_cuda<true>`.
        let quantize_kv_i8 = |kv_lhd: &[f32], l: usize| -> (Vec<i8>, Vec<f32>, Vec<f32>) {
            let mut q = vec![0i8; l * hkv * hd];
            let mut sc = vec![0.0f32; l * hkv];
            let mut deq = vec![0.0f32; l * hkv * hd];
            for pos in 0..l {
                for h in 0..hkv {
                    let row = &kv_lhd[(pos * hkv + h) * hd..(pos * hkv + h + 1) * hd];
                    let amax = row.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                    let s = if amax > 0.0 { amax / 127.0 } else { 0.0 };
                    let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
                    sc[pos * hkv + h] = s;
                    for d in 0..hd {
                        let qi = if amax > 0.0 {
                            (row[d] * inv).round_ties_even().clamp(-127.0, 127.0) as i8
                        } else {
                            0
                        };
                        q[(pos * hkv + h) * hd + d] = qi;
                        deq[(pos * hkv + h) * hd + d] = f32::from(qi) * s;
                    }
                }
            }
            (q, sc, deq)
        };

        // Test matrix:
        //  1. SWA ring_cap=1056 (wrap-around!) with BF16 KV & INT8 KV, T=1, 8, 16 (few) & T=64 (prefill)
        //  2. Global linear with start_pos=9000 (tau(p) > 1.0!) with BF16 KV & INT8 KV, T=1, 8, 16 (few) & T=64 (prefill)
        //  3. Branching Tree mode (T=8) on BOTH SWA (ring_cap=1056) and Global (start_pos=9000), BF16 & INT8!
        #[derive(Clone, Copy)]
        struct Case {
            name: &'static str,
            is_local: bool,
            is_int8: bool,
            t: usize,
            use_prefill: bool,
            is_tree: bool,
            sink_f32: bool,
            start_pos: usize,
            cache_len: usize,
            kv_start_pos: usize,
            kv_phys_base: usize,
            write_offset: usize,
            ring_cap: usize,
            cap: usize,
        }

        let cases = [
            Case {
                name: "swa_bf16_few_t1",
                is_local: true,
                is_int8: false,
                t: 1,
                use_prefill: false,
                is_tree: false,
                sink_f32: false,
                start_pos: 1400,
                cache_len: 1024,
                kv_start_pos: 376,
                kv_phys_base: 26,
                write_offset: 1050,
                ring_cap: 1056,
                cap: 1056,
            },
            Case {
                name: "swa_bf16_few_t8_wrap",
                is_local: true,
                is_int8: false,
                t: 8,
                use_prefill: false,
                is_tree: false,
                sink_f32: true,
                start_pos: 1400,
                cache_len: 1024,
                kv_start_pos: 376,
                kv_phys_base: 26,
                write_offset: 1050,
                ring_cap: 1056,
                cap: 1056,
            },
            Case {
                name: "swa_i8_few_t16_wrap",
                is_local: true,
                is_int8: true,
                t: 16,
                use_prefill: false,
                is_tree: false,
                sink_f32: false,
                start_pos: 1400,
                cache_len: 1024,
                kv_start_pos: 376,
                kv_phys_base: 26,
                write_offset: 1050,
                ring_cap: 1056,
                cap: 1056,
            },
            Case {
                name: "swa_bf16_prefill_t64_wrap",
                is_local: true,
                is_int8: false,
                t: 64,
                use_prefill: true,
                is_tree: false,
                sink_f32: false,
                start_pos: 1500,
                cache_len: 960,
                kv_start_pos: 540,
                kv_phys_base: 80,
                write_offset: 1040,
                ring_cap: 1056,
                cap: 1056,
            },
            Case {
                name: "swa_i8_prefill_t64_wrap",
                is_local: true,
                is_int8: true,
                t: 64,
                use_prefill: true,
                is_tree: false,
                sink_f32: true,
                start_pos: 1500,
                cache_len: 960,
                kv_start_pos: 540,
                kv_phys_base: 80,
                write_offset: 1040,
                ring_cap: 1056,
                cap: 1056,
            },
            Case {
                name: "global_bf16_few_t1_tau",
                is_local: false,
                is_int8: false,
                t: 1,
                use_prefill: false,
                is_tree: false,
                sink_f32: false,
                start_pos: 9000,
                cache_len: 1200,
                kv_start_pos: 7800,
                kv_phys_base: 0,
                write_offset: 1200,
                ring_cap: 0,
                cap: 1300,
            },
            Case {
                name: "global_bf16_few_t16_tau",
                is_local: false,
                is_int8: false,
                t: 16,
                use_prefill: false,
                is_tree: false,
                sink_f32: true,
                start_pos: 9000,
                cache_len: 1200,
                kv_start_pos: 7800,
                kv_phys_base: 0,
                write_offset: 1200,
                ring_cap: 0,
                cap: 1300,
            },
            Case {
                name: "global_i8_few_t8_tau",
                is_local: false,
                is_int8: true,
                t: 8,
                use_prefill: false,
                is_tree: false,
                sink_f32: false,
                start_pos: 9000,
                cache_len: 1200,
                kv_start_pos: 7800,
                kv_phys_base: 0,
                write_offset: 1200,
                ring_cap: 0,
                cap: 1300,
            },
            Case {
                name: "global_bf16_prefill_t64_tau",
                is_local: false,
                is_int8: false,
                t: 64,
                use_prefill: true,
                is_tree: false,
                sink_f32: false,
                start_pos: 9000,
                cache_len: 1100,
                kv_start_pos: 7900,
                kv_phys_base: 0,
                write_offset: 1100,
                ring_cap: 0,
                cap: 1280,
            },
            Case {
                name: "global_i8_prefill_t64_tau",
                is_local: false,
                is_int8: true,
                t: 64,
                use_prefill: true,
                is_tree: false,
                sink_f32: true,
                start_pos: 9000,
                cache_len: 1100,
                kv_start_pos: 7900,
                kv_phys_base: 0,
                write_offset: 1100,
                ring_cap: 0,
                cap: 1280,
            },
            Case {
                name: "swa_bf16_tree_t8_wrap",
                is_local: true,
                is_int8: false,
                t: 8,
                use_prefill: false,
                is_tree: true,
                sink_f32: false,
                start_pos: 1400,
                cache_len: 1024,
                kv_start_pos: 376,
                kv_phys_base: 26,
                write_offset: 1050,
                ring_cap: 1056,
                cap: 1056,
            },
            Case {
                name: "global_i8_tree_t8_tau",
                is_local: false,
                is_int8: true,
                t: 8,
                use_prefill: false,
                is_tree: true,
                sink_f32: true,
                start_pos: 9000,
                cache_len: 1150,
                kv_start_pos: 7850,
                kv_phys_base: 0,
                write_offset: 1150,
                ring_cap: 0,
                cap: 1280,
            },
        ];

        // 8-node branching tree fixture
        let tree_taps_8: [[i32; 4]; 8] = [
            [-3, -2, -1, 0],
            [-2, -1, 0, 1],
            [-1, 0, 1, 2],
            [0, 1, 2, 3],
            [-1, 0, 1, 4],
            [0, 1, 4, 5],
            [-2, -1, 0, 6],
            [-1, 0, 6, 7],
        ];
        let tree_depths_8: [i32; 8] = [0, 1, 2, 3, 2, 3, 1, 2];
        let tree_mask_8: [u32; 8] = [
            0b0000_0001,
            0b0000_0011,
            0b0000_0111,
            0b0000_1111,
            0b0001_0011,
            0b0011_0011,
            0b0100_0001,
            0b1100_0001,
        ];
        let tree_taps_flat: Vec<i32> = tree_taps_8.iter().flatten().copied().collect();
        let tree_taps_arr = MxArray::from_int32(&tree_taps_flat, &[8, 4]).unwrap();
        let tree_depths_arr = MxArray::from_int32(&tree_depths_8, &[8]).unwrap();
        let tree_mask_arr = MxArray::from_uint32(&tree_mask_8, &[8]).unwrap();

        for c in cases {
            let t = c.t;
            let q_raw = bf16_arr(&rng.vec(t * hq * hd, 2.5), &[t as i64, (hq * hd) as i64]);
            let r_raw = bf16_arr(
                &rng.vec(t * hq * d_rel, 2.5),
                &[t as i64, (hq * d_rel) as i64],
            );
            let k_raw = bf16_arr(&rng.vec(t * hkv * hd, 2.0), &[t as i64, (hkv * hd) as i64]);
            let v_raw = bf16_arr(&rng.vec(t * hkv * hd, 2.0), &[t as i64, (hkv * hd) as i64]);
            let k_hist_in = bf16_arr(&rng.vec(3 * hkv * hd, 1.5), &[3, (hkv * hd) as i64]);
            let v_hist_in = bf16_arr(&rng.vec(3 * hkv * hd, 1.5), &[3, (hkv * hd) as i64]);
            let gate = bf16_arr(&rng.vec(t * hq * hd, 2.0), &[t as i64, (hq * hd) as i64]);
            let sink_vals = rng.vec(t * hq, 2.0);
            let sink = if c.sink_f32 {
                f32_arr(&sink_vals, &[t as i64, hq as i64])
            } else {
                bf16_arr(&sink_vals, &[t as i64, hq as i64])
            };
            let (dyn_k, dyn_v) = if !c.is_local {
                (
                    Some(bf16_arr(
                        &rng.vec(t * hkv * 4, 0.3),
                        &[t as i64, hkv as i64, 4],
                    )),
                    Some(f32_arr(
                        &rng.vec(t * hkv * 4, 0.3),
                        &[t as i64, hkv as i64, 4],
                    )),
                )
            } else {
                (None, None)
            };

            // Populate logical prefix K/V `[cache_len, Hkv, D]` and place into physical cache layout.
            let pref_k_raw = rng.vec(c.cache_len * hkv * hd, 1.0);
            let pref_v_raw = rng.vec(c.cache_len * hkv * hd, 1.0);
            let pref_k_bf: Vec<f32> = pref_k_raw
                .iter()
                .map(|&v| bf16::from_f32(v).to_f32())
                .collect();
            let pref_v_bf: Vec<f32> = pref_v_raw
                .iter()
                .map(|&v| bf16::from_f32(v).to_f32())
                .collect();

            let (kv_k_cache, kv_v_cache, kv_k_scales, kv_v_scales, pref_k_ref, pref_v_ref) = if !c
                .is_int8
            {
                let mut kc_host = vec![0u16; hkv * c.cap * hd];
                let mut vc_host = vec![0u16; hkv * c.cap * hd];
                for j in 0..c.cache_len {
                    let phys = if c.ring_cap > 0 {
                        (c.kv_phys_base + j) % c.ring_cap
                    } else {
                        j
                    };
                    for h in 0..hkv {
                        for d in 0..hd {
                            let src = (j * hkv + h) * hd + d;
                            let dst = (h * c.cap + phys) * hd + d;
                            kc_host[dst] = bf16::from_f32(pref_k_bf[src]).to_bits();
                            vc_host[dst] = bf16::from_f32(pref_v_bf[src]).to_bits();
                        }
                    }
                }
                let kc =
                    MxArray::from_bfloat16(&kc_host, &[1, hkv as i64, c.cap as i64, hd as i64])
                        .unwrap();
                let vc =
                    MxArray::from_bfloat16(&vc_host, &[1, hkv as i64, c.cap as i64, hd as i64])
                        .unwrap();
                (kc, vc, None, None, pref_k_bf, pref_v_bf)
            } else {
                let (pk_q, pk_s, pk_deq) = quantize_kv_i8(&pref_k_bf, c.cache_len);
                let (pv_q, pv_s, pv_deq) = quantize_kv_i8(&pref_v_bf, c.cache_len);
                let mut kc_host = vec![0i8; hkv * c.cap * hd];
                let mut vc_host = vec![0i8; hkv * c.cap * hd];
                let mut ks_host = vec![0.0f32; hkv * c.cap];
                let mut vs_host = vec![0.0f32; hkv * c.cap];
                for j in 0..c.cache_len {
                    let phys = if c.ring_cap > 0 {
                        (c.kv_phys_base + j) % c.ring_cap
                    } else {
                        j
                    };
                    for h in 0..hkv {
                        ks_host[h * c.cap + phys] = pk_s[j * hkv + h];
                        vs_host[h * c.cap + phys] = pv_s[j * hkv + h];
                        for d in 0..hd {
                            let src = (j * hkv + h) * hd + d;
                            let dst = (h * c.cap + phys) * hd + d;
                            kc_host[dst] = pk_q[src];
                            vc_host[dst] = pv_q[src];
                        }
                    }
                }
                let kc = MxArray::from_int8(&kc_host, &[1, hkv as i64, c.cap as i64, hd as i64])
                    .unwrap();
                let vc = MxArray::from_int8(&vc_host, &[1, hkv as i64, c.cap as i64, hd as i64])
                    .unwrap();
                let ks = f32_arr(&ks_host, &[1, hkv as i64, c.cap as i64, 1]);
                let vs = f32_arr(&vs_host, &[1, hkv as i64, c.cap as i64, 1]);
                (kc, vc, Some(ks), Some(vs), pk_deq, pv_deq)
            };

            // Run fused CUDA kv_conv_prep_store
            let update_hist = !c.is_tree;
            let prep = kv_conv_prep_store_cuda(&KvConvPrepParams {
                q_raw: &q_raw,
                r_raw: &r_raw,
                k_raw: &k_raw,
                v_raw: &v_raw,
                q_clip: &q_clip,
                rel_clip: &rel_clip,
                k_conv_w: &k_conv_w,
                v_conv_w: &v_conv_w,
                dyn_k: dyn_k.as_ref(),
                dyn_v: dyn_v.as_ref(),
                k_hist_in: &k_hist_in,
                v_hist_in: &v_hist_in,
                kv_k_cache: &kv_k_cache,
                kv_v_cache: &kv_v_cache,
                kv_k_scales: kv_k_scales.as_ref(),
                kv_v_scales: kv_v_scales.as_ref(),
                conv_taps: c.is_tree.then_some(&tree_taps_arr),
                depths: c.is_tree.then_some(&tree_depths_arr),
                start_pos: c.start_pos as i32,
                write_offset: c.write_offset as i32,
                ring_cap: c.ring_cap as i32,
                is_global: !c.is_local,
                log_n_floor: log_n_floor as f32,
                log_alpha: log_alpha as f32,
                update_hist,
            })
            .unwrap()
            .unwrap_or_else(|| panic!("kv_conv_prep_store_cuda returned None for {}", c.name));

            // Run fused CUDA SDPA (few or prefill) chained directly to prep aliases
            let sdpa_params = SdpaParams {
                q: &prep.q,
                r: &prep.r,
                k_cache: &prep.kv_k_alias,
                v_cache: &prep.kv_v_alias,
                k_scales: prep.ks_alias.as_ref(),
                v_scales: prep.vs_alias.as_ref(),
                rel_bias_w: &rel_bias_w,
                gate: &gate,
                sink: &sink,
                tree_depths: c.is_tree.then_some(&tree_depths_arr),
                tree_mask: c.is_tree.then_some(&tree_mask_arr),
                start_pos: c.start_pos as i32,
                cache_len: c.cache_len as i32,
                kv_start_pos: c.kv_start_pos as i32,
                kv_phys_base: c.kv_phys_base as i32,
                write_offset: c.write_offset as i32,
                ring_cap: c.ring_cap as i32,
                is_local: c.is_local,
                sliding_window: sliding_window as i32,
            };
            let sdpa_out = if c.use_prefill {
                sdpa_prefill_cuda(&sdpa_params)
            } else {
                sdpa_few_cuda(&sdpa_params)
            }
            .unwrap()
            .unwrap_or_else(|| panic!("sdpa_cuda returned None for {}", c.name));
            assert_eq!(
                sdpa_out.shape().unwrap().as_ref(),
                &[t as i64, (hq * hd) as i64]
            );
            assert_eq!(sdpa_out.dtype().unwrap(), DType::BFloat16);

            // --- Build exact Rust/MLX reference ---
            let mut ref_k_hist = k_hist_in.clone();
            let mut ref_v_hist = v_hist_in.clone();
            let (ref_k_conv, ref_v_conv) = if c.is_tree {
                let kr_f = k_raw.to_float32().unwrap();
                let vr_f = v_raw.to_float32().unwrap();
                let kh_f = k_hist_in.to_float32().unwrap();
                let vh_f = v_hist_in.to_float32().unwrap();
                if c.is_local {
                    let kc = ref_sconv_k4_tree(
                        kr_f.as_ref(),
                        &k_conv_vals,
                        kh_f.as_ref(),
                        None,
                        &tree_taps_8,
                        hkv * hd,
                    );
                    let vc = ref_sconv_k4_tree(
                        vr_f.as_ref(),
                        &v_conv_vals,
                        vh_f.as_ref(),
                        None,
                        &tree_taps_8,
                        hkv * hd,
                    );
                    (
                        bf16_arr(&kc, &[t as i64, (hkv * hd) as i64]),
                        bf16_arr(&vc, &[t as i64, (hkv * hd) as i64]),
                    )
                } else {
                    let dk_f = dyn_k.as_ref().unwrap().to_float32().unwrap();
                    let dv_f = dyn_v.as_ref().unwrap().to_float32().unwrap();
                    let kc = ref_dyn_sconv_k4_tree(
                        kr_f.as_ref(),
                        &k_conv_vals,
                        dk_f.as_ref(),
                        kh_f.as_ref(),
                        &tree_taps_8,
                        hkv,
                        hd,
                    );
                    let vc = ref_dyn_sconv_k4_tree(
                        vr_f.as_ref(),
                        &v_conv_vals,
                        dv_f.as_ref(),
                        vh_f.as_ref(),
                        &tree_taps_8,
                        hkv,
                        hd,
                    );
                    (
                        bf16_arr(&kc, &[t as i64, (hkv * hd) as i64]),
                        bf16_arr(&vc, &[t as i64, (hkv * hd) as i64]),
                    )
                }
            } else if c.is_local {
                (
                    static_short_conv(&k_raw, &k_conv_w, &mut ref_k_hist).unwrap(),
                    static_short_conv(&v_raw, &v_conv_w, &mut ref_v_hist).unwrap(),
                )
            } else {
                (
                    dynamic_short_conv(&k_raw, &k_conv_w, dyn_k.as_ref().unwrap(), &mut ref_k_hist)
                        .unwrap(),
                    dynamic_short_conv(&v_raw, &v_conv_w, dyn_v.as_ref().unwrap(), &mut ref_v_hist)
                        .unwrap(),
                )
            };

            if update_hist {
                assert_close(
                    &format!("{}: k_hist", c.name),
                    prep.k_hist.as_ref().unwrap().to_float32().unwrap().as_ref(),
                    ref_k_hist.to_float32().unwrap().as_ref(),
                    1e-6,
                    1e-6,
                );
                assert_close(
                    &format!("{}: v_hist", c.name),
                    prep.v_hist.as_ref().unwrap().to_float32().unwrap().as_ref(),
                    ref_v_hist.to_float32().unwrap().as_ref(),
                    1e-6,
                    1e-6,
                );
            }

            let q_3d = q_raw.reshape(&[t as i64, hq as i64, hd as i64]).unwrap();
            let r_3d = r_raw.reshape(&[t as i64, hq as i64, d_rel as i64]).unwrap();
            let k_3d = ref_k_conv
                .reshape(&[t as i64, hkv as i64, hd as i64])
                .unwrap();
            let v_3d = ref_v_conv
                .reshape(&[t as i64, hkv as i64, hd as i64])
                .unwrap();

            let mut ref_q = l2_clip(&q_3d, Some(&q_clip), qk_clip, DType::BFloat16).unwrap();
            let mut ref_r = l2_clip(&r_3d, Some(&rel_clip), rel_clip_thresh, DType::BFloat16)
                .unwrap()
                .astype(DType::Float32)
                .unwrap();
            if !c.is_local {
                let tau = if c.is_tree {
                    let vals: Vec<f32> = tree_depths_8
                        .iter()
                        .map(|&dp| {
                            let p = (c.start_pos as i32 + dp) as f64;
                            let ratio = ((p + 1.0) / (log_n_floor as f64)).max(1.0);
                            (1.0 + log_alpha * ratio.ln()) as f32
                        })
                        .collect();
                    f32_arr(&vals, &[t as i64, 1, 1])
                } else {
                    global_log_scale_tau(c.start_pos, t, log_n_floor, log_alpha).unwrap()
                };
                ref_q = ref_q
                    .astype(DType::Float32)
                    .unwrap()
                    .mul(&tau)
                    .unwrap()
                    .astype(DType::BFloat16)
                    .unwrap();
                ref_r = ref_r.mul(&tau).unwrap();
            }
            let ref_q_1htd = ref_q
                .transpose(Some(&[1, 0, 2]))
                .unwrap()
                .expand_dims(0)
                .unwrap();
            assert_close(
                &format!("{}: prep out_q", c.name),
                prep.q.to_float32().unwrap().as_ref(),
                ref_q_1htd.to_float32().unwrap().as_ref(),
                2e-2,
                3e-3,
            );
            assert_close(
                &format!("{}: prep out_r", c.name),
                prep.r.to_float32().unwrap().as_ref(),
                ref_r.to_float32().unwrap().as_ref(),
                2e-2,
                3e-3,
            );

            let ref_k_new = l2_clip(&k_3d, None, qk_clip, DType::BFloat16).unwrap();
            let ref_v_new = rms_norm_unscaled(&v_3d.astype(DType::Float32).unwrap(), ATTN_RES_EPS)
                .unwrap()
                .astype(DType::BFloat16)
                .unwrap();
            let ref_k_new_f = ref_k_new.to_float32().unwrap();
            let ref_v_new_f = ref_v_new.to_float32().unwrap();
            let (new_k_ref, new_v_ref) = if c.is_int8 {
                let (_, _, kd) = quantize_kv_i8(ref_k_new_f.as_ref(), t);
                let (_, _, vd) = quantize_kv_i8(ref_v_new_f.as_ref(), t);
                (kd, vd)
            } else {
                (ref_k_new_f.as_ref().to_vec(), ref_v_new_f.as_ref().to_vec())
            };

            let mut full_k = pref_k_ref;
            full_k.extend_from_slice(&new_k_ref);
            let mut full_v = pref_v_ref;
            full_v.extend_from_slice(&new_v_ref);
            let kv_len = c.cache_len + t;
            let k_all = f32_arr(&full_k, &[kv_len as i64, hkv as i64, hd as i64]);
            let v_all = f32_arr(&full_v, &[kv_len as i64, hkv as i64, hd as i64]);

            let ref_sdpa = run_ref_sdpa(
                &ref_q_1htd,
                &ref_r,
                &k_all,
                &v_all,
                &gate,
                &sink,
                c.start_pos,
                c.cache_len,
                c.kv_start_pos,
                c.is_local,
                c.is_tree.then_some(&tree_depths_8[..]),
                c.is_tree.then_some(&tree_mask_8[..]),
            );
            let got_sdpa = sdpa_out.to_float32().unwrap();
            // After sdpa_out evaluation, prep.kv_*_alias are evaluated and must alias the caches.
            assert!(
                unsafe {
                    sys::mlx_array_aliases(kv_k_cache.as_raw_ptr(), prep.kv_k_alias.as_raw_ptr())
                },
                "{}: kv_k_alias must alias kv_k_cache",
                c.name
            );
            assert!(
                unsafe {
                    sys::mlx_array_aliases(kv_v_cache.as_raw_ptr(), prep.kv_v_alias.as_raw_ptr())
                },
                "{}: kv_v_alias must alias kv_v_cache",
                c.name
            );
            let (abs_tol, rel_tol) = if c.is_int8 {
                (0.04, 1.5e-2)
            } else {
                (0.025, 8e-3)
            };
            assert_close(
                &format!("{}: sdpa_out", c.name),
                got_sdpa.as_ref(),
                &ref_sdpa,
                abs_tol,
                rel_tol,
            );
        }
    }

    // =========================================================================
    // Test 4: Static ShortConv k=4 + optional residual (linear & tree, C=2304)
    // =========================================================================
    #[test]
    fn cuda_inkling2_sconv_k4_residual_parity() {
        if !host_has_cuda() || !available() {
            return;
        }
        let mut rng = Rng::new(0x5C0B_4444);
        let c = 2304usize;
        let w_vals = rng.vec(c * 4, 0.4);
        let weight = f32_arr(&w_vals, &[c as i64, 4]);
        let hist_vals = rng.vec(3 * c, 1.2);
        let hist_in = bf16_arr(&hist_vals, &[3, c as i64]);
        let hist_bf = hist_in.to_float32().unwrap();

        // Linear mode: T = 1, 8, 16, 48; with and without residual (BF16 and FP32)
        for &t in &[1usize, 8, 16, 48] {
            let x = bf16_arr(&rng.vec(t * c, 1.2), &[t as i64, c as i64]);
            let res_vals = rng.vec(t * c, 1.2);
            let res_bf16 = bf16_arr(&res_vals, &[t as i64, c as i64]);
            let res_f32 = f32_arr(&res_vals, &[t as i64, c as i64]);

            let mut ref_hist = hist_in.clone();
            let ref_conv = static_short_conv(&x, &weight, &mut ref_hist).unwrap();
            let ref_conv_f = ref_conv.to_float32().unwrap();
            let ref_hist_f = ref_hist.to_float32().unwrap();

            for (res_lbl, res_opt) in [
                ("none", None),
                ("bf16", Some(&res_bf16)),
                ("f32", Some(&res_f32)),
            ] {
                let want = match res_opt {
                    None => ref_conv_f.as_ref().to_vec(),
                    Some(r) => r
                        .astype(DType::Float32)
                        .unwrap()
                        .add(&ref_conv.astype(DType::Float32).unwrap())
                        .unwrap()
                        .astype(DType::BFloat16)
                        .unwrap()
                        .to_float32()
                        .unwrap()
                        .as_ref()
                        .to_vec(),
                };
                for &update_hist in &[true, false] {
                    let out =
                        sconv_k4_residual_cuda(&x, &weight, &hist_in, res_opt, None, update_hist)
                            .unwrap()
                            .expect("sconv_k4_residual_cuda linear must run on CUDA");
                    assert_eq!(out.hist.is_some(), update_hist);
                    assert_close(
                        &format!("sconv linear T={t} res={res_lbl} uh={update_hist}"),
                        out.y.to_float32().unwrap().as_ref(),
                        &want,
                        0.02,
                        2e-3,
                    );
                    if let Some(h) = out.hist {
                        assert_close(
                            &format!("sconv linear hist T={t}"),
                            h.to_float32().unwrap().as_ref(),
                            ref_hist_f.as_ref(),
                            1e-6,
                            1e-6,
                        );
                    }
                }
            }
        }

        // Tree mode with branching taps + out-of-range (< -3) zero-padded tap
        let taps: [[i32; 4]; 8] = [
            [-4, -2, -1, 0],
            [-2, -1, 0, 1],
            [-1, 0, 1, 2],
            [0, 1, 2, 3],
            [-1, 0, 1, 4],
            [0, 1, 4, 5],
            [-3, -2, 0, 6],
            [-2, 0, 6, 7],
        ];
        let taps_flat: Vec<i32> = taps.iter().flatten().copied().collect();
        let taps_arr = MxArray::from_int32(&taps_flat, &[8, 4]).unwrap();
        let x = bf16_arr(&rng.vec(8 * c, 1.2), &[8, c as i64]);
        let x_bf = x.to_float32().unwrap();
        let res_bf = bf16_arr(&rng.vec(8 * c, 1.2), &[8, c as i64]);
        let res_bf_f = res_bf.to_float32().unwrap();

        for (lbl, res_arr, res_slice) in [
            ("tree_no_res", None, None),
            ("tree_bf16_res", Some(&res_bf), Some(res_bf_f.as_ref())),
        ] {
            let want = ref_sconv_k4_tree(
                x_bf.as_ref(),
                &w_vals,
                hist_bf.as_ref(),
                res_slice,
                &taps,
                c,
            );
            let out =
                sconv_k4_residual_cuda(&x, &weight, &hist_in, res_arr, Some(&taps_arr), false)
                    .unwrap()
                    .expect("sconv_k4_residual_cuda tree must run on CUDA");
            assert!(out.hist.is_none());
            assert_close(lbl, out.y.to_float32().unwrap().as_ref(), &want, 0.02, 2e-3);
        }
    }

    // =========================================================================
    // Test 5: Fused LM Head + Softcap (SymQ8, AffineQ8, BF16; M=1,2,8,16;
    //         out_mode=0 FP32, out_mode=1 GreedyArgmax, out_mode=2 BF16)
    // =========================================================================
    #[test]
    fn cuda_inkling2_lm_head_softcap_parity() {
        if !host_has_cuda() || !sm80_available() {
            return;
        }
        let mut rng = Rng::new(0x13AD_50F7);
        let k = 2304usize;
        let v_pad = 4096usize;
        let v_unpad = 3973usize;
        let softcap = 30.0f32;

        // Base dense BF16 weights [V_pad, K]
        let mut w_vals = rng.vec(v_pad * k, 0.08);
        // Make padded rows >= v_unpad huge so any out-of-bounds read would hijack argmax!
        for row in v_unpad..v_pad {
            for col in 0..k {
                w_vals[row * k + col] = 0.9;
            }
        }
        // Inject distinct unambiguous winners for up to 16 rows
        let winners: [usize; 16] = [
            17, 1024, 3972, 0, 513, 2047, 3001, 777, 123, 2500, 3971, 42, 888, 1999, 3333, 64,
        ];
        let mut x16_vals = rng.vec(16 * k, 0.35);
        for (m, &win) in winners.iter().enumerate() {
            // Boost orthogonal-ish 64-element block in both x[m] and w[win]
            let blk = (m * 64) % k;
            for j in 0..64 {
                x16_vals[m * k + blk + j] = 1.2;
                w_vals[win * k + blk + j] = 0.55;
            }
        }

        let w_bf16 = bf16_arr(&w_vals, &[v_pad as i64, k as i64]);
        let (w_q8, s_q8_raw, b_q8_raw) = quantize_mlx(&w_bf16, 32, 8, c"affine");
        let s_aff_bf16 = s_q8_raw.astype(DType::BFloat16).unwrap();
        let b_aff_bf16 = b_q8_raw.unwrap().astype(DType::BFloat16).unwrap();

        // True symmetric Q8_0 (group_size=32, F16 scales, byte = (round(w / s) + 128) as u8,
        // so bias = -128 * scale and SymQ8 kernel reads biases = None).
        let (w_sym_q8, s_sym_f16, b_sym_ref) = quantize_sym_q8_f16(&w_vals, v_pad, k);

        // Dequantize references once to compute exact FP32 logits [M, V_unpad]
        let dequant_q8 = |w: &MxArray, s: &MxArray, b: &MxArray| -> MxArray {
            let h = unsafe {
                sys::mlx_dequantize(
                    w.as_raw_ptr(),
                    s.as_raw_ptr(),
                    b.as_raw_ptr(),
                    32,
                    8,
                    -1,
                    c"affine".as_ptr(),
                )
            };
            assert!(!h.is_null(), "mlx_dequantize failed");
            MxArray::from_handle(h, "deq")
                .unwrap()
                .slice(&[0, 0], &[v_unpad as i64, k as i64])
                .unwrap()
                .astype(DType::Float32)
                .unwrap()
        };
        let w_sym_ref_t = dequant_q8(&w_sym_q8, &s_sym_f16, &b_sym_ref)
            .transpose(None)
            .unwrap();
        let w_aff_ref_t = dequant_q8(&w_q8, &s_aff_bf16, &b_aff_bf16)
            .transpose(None)
            .unwrap();
        let w_bf16_ref_t = w_bf16
            .slice(&[0, 0], &[v_unpad as i64, k as i64])
            .unwrap()
            .astype(DType::Float32)
            .unwrap()
            .transpose(None)
            .unwrap();

        let formats: [(&str, &MxArray, Option<&MxArray>, Option<&MxArray>, &MxArray); 3] = [
            ("SymQ8", &w_sym_q8, Some(&s_sym_f16), None, &w_sym_ref_t),
            (
                "AffineQ8",
                &w_q8,
                Some(&s_aff_bf16),
                Some(&b_aff_bf16),
                &w_aff_ref_t,
            ),
            ("BF16", &w_bf16, None, None, &w_bf16_ref_t),
        ];

        for &m in &[1usize, 2, 8, 16] {
            let x = bf16_arr(&x16_vals[..m * k], &[m as i64, k as i64]);
            let x_f32 = x.astype(DType::Float32).unwrap();

            for (fmt_name, w_arr, s_opt, b_opt, w_ref_t) in formats {
                let raw_ref = x_f32.matmul(w_ref_t).unwrap();
                let cap_ref = raw_ref
                    .mul_scalar(1.0 / f64::from(softcap))
                    .unwrap()
                    .tanh()
                    .unwrap()
                    .mul_scalar(f64::from(softcap))
                    .unwrap();
                let raw_ref_f = raw_ref.to_float32().unwrap();
                let cap_ref_f = cap_ref.to_float32().unwrap();
                let raw_ref_bf_f = raw_ref
                    .astype(DType::BFloat16)
                    .unwrap()
                    .to_float32()
                    .unwrap();

                // Expected greedy argmax per row
                let mut ref_argmax = Vec::with_capacity(m);
                for (r, &expected_winner) in winners.iter().enumerate().take(m) {
                    let row = &cap_ref_f.as_ref()[r * v_unpad..(r + 1) * v_unpad];
                    let (best_i, _) = row.iter().enumerate().fold(
                        (0usize, f32::NEG_INFINITY),
                        |(bi, bv), (i, &v)| {
                            if v > bv { (i, v) } else { (bi, bv) }
                        },
                    );
                    assert_eq!(
                        best_i, expected_winner,
                        "test fixture sanity: expected winner {expected_winner} got {best_i}",
                    );
                    ref_argmax.push(best_i as i32);
                }

                // out_mode = 0: softcapped FP32 [M, V_unpad]
                let out0 = lm_head_softcap_cuda(
                    &x,
                    w_arr,
                    s_opt,
                    b_opt,
                    v_unpad as i32,
                    softcap,
                    LmHeadOutMode::LogitsF32,
                )
                .unwrap()
                .expect("lm_head out_mode=0 must run on CUDA");
                assert_eq!(out0.dtype().unwrap(), DType::Float32);
                assert_eq!(out0.shape().unwrap().as_ref(), &[m as i64, v_unpad as i64]);
                assert_close(
                    &format!("lm_head {fmt_name} M={m} mode=LogitsF32"),
                    out0.to_float32().unwrap().as_ref(),
                    cap_ref_f.as_ref(),
                    0.05,
                    3e-3,
                );

                // out_mode = 1: zero-DRAM greedy argmax Int32 [M]
                let out1 = lm_head_softcap_cuda(
                    &x,
                    w_arr,
                    s_opt,
                    b_opt,
                    v_unpad as i32,
                    softcap,
                    LmHeadOutMode::GreedyArgmax,
                )
                .unwrap()
                .expect("lm_head out_mode=1 must run on CUDA");
                assert_eq!(out1.dtype().unwrap(), DType::Int32);
                assert_eq!(out1.shape().unwrap().as_ref(), &[m as i64]);
                assert_eq!(
                    out1.to_int32().unwrap().as_ref(),
                    ref_argmax.as_slice(),
                    "lm_head {fmt_name} M={m} mode=GreedyArgmax mismatch"
                );

                // out_mode = 2: uncapped BF16 [M, V_unpad] (softcap = 0.0)
                let out2 = lm_head_softcap_cuda(
                    &x,
                    w_arr,
                    s_opt,
                    b_opt,
                    v_unpad as i32,
                    0.0,
                    LmHeadOutMode::LogitsBf16,
                )
                .unwrap()
                .expect("lm_head out_mode=2 must run on CUDA");
                assert_eq!(out2.dtype().unwrap(), DType::BFloat16);
                assert_eq!(out2.shape().unwrap().as_ref(), &[m as i64, v_unpad as i64]);
                assert_close(
                    &format!("lm_head {fmt_name} M={m} mode=LogitsBf16"),
                    out2.to_float32().unwrap().as_ref(),
                    raw_ref_bf_f.as_ref(),
                    0.15,
                    6e-3,
                );
                let _ = raw_ref_f;
            }
        }
    }

    // =========================================================================
    // Test 6: Fused MoE across ALL 5 quant formats (NVFP4 split+interleaved+outer_scale,
    //         SymQ8, AffineQ8, Q5_K+SymQ8, MXFP8 interleaved), M=1, 8, 16, 48,
    //         with fused post_conv + residual + pre_conv tape + linear & tree conv_taps
    // =========================================================================
    #[test]
    fn cuda_inkling2_moe_forward_parity() {
        if !host_has_cuda() || !sm80_available() {
            return;
        }
        let mut rng = Rng::new(0x0E12_BEEF);
        let e = 128usize;
        let top_k = 8usize;
        let h = 2304usize;
        let inter = 1152usize;
        let n2 = 2 * inter;
        let route_scale = 8.0f32;
        let gate_cap = 4.0f64;
        let up_cap = 25.0f64;

        // Craft router_w [128, 2304] + router_bias [128] + x48 [48, 2304] so that:
        //  - Every token has an unambiguous top-8 (no near-ties -> identical expert IDs!)
        //  - At M=8: experts 0,1,2,3 have cnt >= 3 (TC MMA path!) AND unique experts
        //    including high IDs 112..127 have cnt <= 2 (streaming GEMV path!).
        //  - At M=48: exercises BM=32 multi-tile experts AND small-cnt experts in prefill!
        let max_m = 48usize;
        let mut x48_vals = rng.vec(max_m * h, 0.35);
        let mut rw_vals = rng.vec(e * h, 0.004);
        let rb_vals: Vec<f32> = (0..e).map(|i| -0.05 + (i % 13) as f32 * 0.008).collect();
        for tok in 0..max_m {
            // Orthogonal token indicator in columns 0..48
            for col in 0..max_m {
                x48_vals[tok * h + col] = if col == tok { 1.5 } else { 0.0 };
            }
            // Pick 4 hot experts (shared across tokens so cnt >= 3 at M=8 and cnt > 32 at M=48)
            // + 4 token-varied experts (including high IDs up to 127 so cnt = 1 or 2 at M=8).
            let hot = [0usize, 1, 2, 127];
            let cold = [
                10 + (tok * 3) % 90,
                11 + (tok * 3) % 90,
                100 + (tok / 2) % 12,
                112 + (tok % 15),
            ];
            for (rank_i, &exp_id) in hot.iter().chain(cold.iter()).enumerate() {
                // Distinct decreasing scores per rank so order & top-8 set are rock solid.
                rw_vals[exp_id * h + tok] = 3.5 - (rank_i as f32) * 0.25;
            }
        }
        let router_w = f32_arr(&rw_vals, &[e as i64, h as i64]);
        let router_w_t = router_w.transpose(None).unwrap();
        let router_bias = f32_arr(&rb_vals, &[e as i64]);

        let post_conv_vals = rng.vec(h * 4, 0.35);
        let post_conv_w = f32_arr(&post_conv_vals, &[h as i64, 4]);
        let hist_in = bf16_arr(&rng.vec(3 * h, 0.8), &[3, h as i64]);
        let hist_in_f = hist_in.to_float32().unwrap();
        let res48 = bf16_arr(&rng.vec(max_m * h, 0.8), &[max_m as i64, h as i64]);

        let tree_taps_8: [[i32; 4]; 8] = [
            [-3, -2, -1, 0],
            [-2, -1, 0, 1],
            [-1, 0, 1, 2],
            [0, 1, 2, 3],
            [-1, 0, 1, 4],
            [0, 1, 4, 5],
            [-2, -1, 0, 6],
            [-1, 0, 6, 7],
        ];
        let tree_taps_8_arr = MxArray::from_int32(
            &tree_taps_8.iter().flatten().copied().collect::<Vec<_>>(),
            &[8, 4],
        )
        .unwrap();
        // Prefill M=48 non-linear tree taps (6 independent 8-node subtrees)
        let mut tree_taps_48 = Vec::with_capacity(48);
        for blk in 0..6 {
            let base = blk * 8;
            for row in tree_taps_8 {
                tree_taps_48.push([
                    if row[0] >= 0 { base + row[0] } else { row[0] },
                    if row[1] >= 0 { base + row[1] } else { row[1] },
                    if row[2] >= 0 { base + row[2] } else { row[2] },
                    base + row[3],
                ]);
            }
        }
        let tree_taps_48_arr = MxArray::from_int32(
            &tree_taps_48.iter().flatten().copied().collect::<Vec<_>>(),
            &[48, 4],
        )
        .unwrap();

        let e_base = 8usize;
        let tile_reps = [(e / e_base) as i32, 1, 1];

        // Exact FP32 ground-truth MoE oracle over the 8 distinct base experts (tiled 16x -> 128),
        // bypassing MLX's `qmm_sm80` NVRTC kernels while verifying full 128-expert stride addressing.
        let split_base_t = |deq_3d: &MxArray, rows_dim: usize, cols_dim: usize| -> Vec<MxArray> {
            (0..e_base)
                .map(|be| {
                    deq_3d
                        .slice(
                            &[be as i64, 0, 0],
                            &[(be + 1) as i64, rows_dim as i64, cols_dim as i64],
                        )
                        .unwrap()
                        .squeeze(Some(&[0]))
                        .unwrap()
                        .transpose(None)
                        .unwrap()
                })
                .collect()
        };

        #[allow(clippy::too_many_arguments)]
        let run_ref_moe_pre_conv = |x: &MxArray,
                                    deq_gu_t: &[MxArray],
                                    deq_dn_t: &[MxArray],
                                    gu_outer: Option<&[f32]>,
                                    dn_outer: Option<&[f32]>,
                                    gu_interleaved: bool|
         -> MxArray {
            let m = x.shape().unwrap()[0] as usize;
            let x_f32 = x.astype(DType::Float32).unwrap();
            let logits = x_f32.matmul(&router_w_t).unwrap();
            let (gamma, topk) = sigmoid_biased_topk_from_logits(
                &logits,
                Some(&router_bias),
                top_k as i32,
                1e-20,
                f64::from(route_scale),
                true,
                true,
            )
            .unwrap();
            let gamma_f = gamma.to_float32().unwrap();
            let topk_i = topk.to_int32().unwrap();

            // Group (tok, exp_id, effective_gamma) by base expert `exp_id % e_base`.
            let mut buckets: Vec<Vec<(usize, usize, f32)>> = vec![Vec::new(); e_base];
            for tok in 0..m {
                for k_i in 0..top_k {
                    let slot = tok * top_k + k_i;
                    let exp_id = topk_i[slot] as usize;
                    let mut g_val = gamma_f[slot];
                    if let Some(dno) = dn_outer {
                        g_val *= dno[exp_id];
                    }
                    buckets[exp_id % e_base].push((tok, exp_id, g_val));
                }
            }

            let mut acc = vec![0.0f32; m * h];
            for (be, entries) in buckets.iter().enumerate() {
                if entries.is_empty() {
                    continue;
                }
                let cnt = entries.len();
                let tok_ids: Vec<i32> = entries.iter().map(|&(tok, _, _)| tok as i32).collect();
                let tok_idx = MxArray::from_int32(&tok_ids, &[cnt as i64]).unwrap();
                let x_be = x_f32.take(&tok_idx, 0).unwrap(); // [cnt, H]
                let mut gu_be = x_be
                    .matmul(&deq_gu_t[be])
                    .unwrap()
                    .astype(DType::BFloat16)
                    .unwrap()
                    .astype(DType::Float32)
                    .unwrap(); // [cnt, 2*I]
                if let Some(guo) = gu_outer {
                    let s_vec: Vec<f32> = entries.iter().map(|&(_, eid, _)| guo[eid]).collect();
                    let s_arr = f32_arr(&s_vec, &[cnt as i64, 1]);
                    gu_be = gu_be.mul(&s_arr).unwrap();
                }
                let (gate, up) = if gu_interleaved {
                    let paired = gu_be.reshape(&[cnt as i64, inter as i64, 2]).unwrap();
                    let g = paired
                        .slice(&[0, 0, 0], &[cnt as i64, inter as i64, 1])
                        .unwrap()
                        .squeeze(Some(&[-1]))
                        .unwrap();
                    let u = paired
                        .slice(&[0, 0, 1], &[cnt as i64, inter as i64, 2])
                        .unwrap()
                        .squeeze(Some(&[-1]))
                        .unwrap();
                    (g, u)
                } else {
                    let g = gu_be.slice(&[0, 0], &[cnt as i64, inter as i64]).unwrap();
                    let u = gu_be
                        .slice(&[0, inter as i64], &[cnt as i64, n2 as i64])
                        .unwrap();
                    (g, u)
                };
                let act = situ_activation(&gate, &up, gate_cap, up_cap)
                    .unwrap()
                    .astype(DType::BFloat16)
                    .unwrap()
                    .astype(DType::Float32)
                    .unwrap();
                let dn_be = act
                    .matmul(&deq_dn_t[be])
                    .unwrap()
                    .astype(DType::BFloat16)
                    .unwrap()
                    .to_float32()
                    .unwrap();
                for (row_i, &(tok, _, g_val)) in entries.iter().enumerate() {
                    let src_row = &dn_be.as_ref()[row_i * h..(row_i + 1) * h];
                    let dst_row = &mut acc[tok * h..(tok + 1) * h];
                    for d in 0..h {
                        dst_row[d] += src_row[d] * g_val;
                    }
                }
            }
            bf16_arr(&acc, &[m as i64, h as i64])
        };

        // Shared deterministic base unquantized weights for 8 distinct experts
        // (tiled 16x -> 128 experts on GPU); uniform(-0.14, 0.14) has std ~0.08.
        let base_gu_vals = rng.vec(e_base * n2 * h, 0.14);
        let base_dn_vals = rng.vec(e_base * h * inter, 0.14);
        let base_gu_bf16 = bf16_arr(&base_gu_vals, &[e_base as i64, n2 as i64, h as i64]);
        let base_dn_bf16 = bf16_arr(&base_dn_vals, &[e_base as i64, h as i64, inter as i64]);
        MxArray::eval_arrays(&[&base_gu_bf16, &base_dn_bf16]).unwrap();

        #[allow(clippy::too_many_arguments)]
        let verify_format = |label: &str,
                             gu_w: &MxArray,
                             gu_scales: &MxArray,
                             gu_biases: Option<&MxArray>,
                             dn_w: &MxArray,
                             dn_scales: &MxArray,
                             dn_biases: Option<&MxArray>,
                             deq_gu_t: &[MxArray],
                             deq_dn_t: &[MxArray],
                             gu_mode: MoeQuantMode,
                             dn_mode: MoeQuantMode,
                             gu_interleaved: bool,
                             gu_outer: Option<(&MxArray, &[f32])>,
                             dn_outer: Option<(&MxArray, &[f32])>,
                             ms: &[usize],
                             test_post_conv: bool,
                             abs_tol: f32,
                             rel_tol: f32| {
            let gu_outer_arr = gu_outer.map(|(a, _)| a);
            let gu_outer_sl = gu_outer.map(|(_, s)| s);
            let dn_outer_arr = dn_outer.map(|(a, _)| a);
            let dn_outer_sl = dn_outer.map(|(_, s)| s);

            for &m in ms {
                let x = bf16_arr(&x48_vals[..m * h], &[m as i64, h as i64]);
                let ref_pre = run_ref_moe_pre_conv(
                    &x,
                    deq_gu_t,
                    deq_dn_t,
                    gu_outer_sl,
                    dn_outer_sl,
                    gu_interleaved,
                );
                let ref_pre_f = ref_pre.to_float32().unwrap();

                // 1. Plain MoE (no post_conv)
                let out_plain = moe_forward_cuda(&MoeForwardParams {
                    x: &x,
                    router_w: &router_w,
                    router_bias: Some(&router_bias),
                    gu_w,
                    gu_scales,
                    gu_biases,
                    gu_outer_scale: gu_outer_arr,
                    gu_mode,
                    gu_interleaved,
                    dn_w,
                    dn_scales,
                    dn_biases,
                    dn_outer_scale: dn_outer_arr,
                    dn_mode,
                    route_scale,
                    post_conv_w: None,
                    hist_in: None,
                    residual: None,
                    conv_taps: None,
                    update_hist: false,
                    emit_pre_conv: false,
                })
                .unwrap()
                .unwrap_or_else(|| panic!("moe_forward_cuda returned None for {label} M={m}"));
                assert!(out_plain.hist.is_none() && out_plain.pre_conv.is_none());
                let out_plain_f = out_plain.y.to_float32().unwrap();
                assert_close(
                    &format!("moe {label} plain M={m}"),
                    out_plain_f.as_ref(),
                    ref_pre_f.as_ref(),
                    abs_tol,
                    rel_tol,
                );

                if !test_post_conv {
                    continue;
                }

                // 2. Fused post_conv + residual + update_hist + emit_pre_conv (linear)
                let res_m = res48.slice(&[0, 0], &[m as i64, h as i64]).unwrap();
                let mut ref_hist = hist_in.clone();
                let ref_conv = static_short_conv(&ref_pre, &post_conv_w, &mut ref_hist).unwrap();
                let ref_y_lin = res_m
                    .astype(DType::Float32)
                    .unwrap()
                    .add(&ref_conv.astype(DType::Float32).unwrap())
                    .unwrap()
                    .astype(DType::BFloat16)
                    .unwrap();

                let out_fused = moe_forward_cuda(&MoeForwardParams {
                    x: &x,
                    router_w: &router_w,
                    router_bias: Some(&router_bias),
                    gu_w,
                    gu_scales,
                    gu_biases,
                    gu_outer_scale: gu_outer_arr,
                    gu_mode,
                    gu_interleaved,
                    dn_w,
                    dn_scales,
                    dn_biases,
                    dn_outer_scale: dn_outer_arr,
                    dn_mode,
                    route_scale,
                    post_conv_w: Some(&post_conv_w),
                    hist_in: Some(&hist_in),
                    residual: Some(&res_m),
                    conv_taps: None,
                    update_hist: true,
                    emit_pre_conv: true,
                })
                .unwrap()
                .expect("moe_forward_cuda fused linear must run on CUDA");
                assert_close(
                    &format!("moe {label} fused_lin y M={m}"),
                    out_fused.y.to_float32().unwrap().as_ref(),
                    ref_y_lin.to_float32().unwrap().as_ref(),
                    abs_tol,
                    rel_tol,
                );
                assert_close(
                    &format!("moe {label} fused_lin pre_conv M={m}"),
                    out_fused
                        .pre_conv
                        .as_ref()
                        .unwrap()
                        .to_float32()
                        .unwrap()
                        .as_ref(),
                    ref_pre_f.as_ref(),
                    abs_tol,
                    rel_tol,
                );
                assert_close(
                    &format!("moe {label} fused_lin hist M={m}"),
                    out_fused
                        .hist
                        .as_ref()
                        .unwrap()
                        .to_float32()
                        .unwrap()
                        .as_ref(),
                    ref_hist.to_float32().unwrap().as_ref(),
                    abs_tol,
                    rel_tol,
                );

                // 3. Fused post_conv + residual + emit_pre_conv with branching tree conv_taps
                if m == 8 || m == 48 {
                    let (taps_arr, taps_slice): (&MxArray, &[[i32; 4]]) = if m == 8 {
                        (&tree_taps_8_arr, &tree_taps_8)
                    } else {
                        (&tree_taps_48_arr, &tree_taps_48)
                    };
                    let res_m_f = res_m.to_float32().unwrap();
                    let ref_y_tree = ref_sconv_k4_tree(
                        ref_pre_f.as_ref(),
                        &post_conv_vals,
                        hist_in_f.as_ref(),
                        Some(res_m_f.as_ref()),
                        taps_slice,
                        h,
                    );
                    let out_tree = moe_forward_cuda(&MoeForwardParams {
                        x: &x,
                        router_w: &router_w,
                        router_bias: Some(&router_bias),
                        gu_w,
                        gu_scales,
                        gu_biases,
                        gu_outer_scale: gu_outer_arr,
                        gu_mode,
                        gu_interleaved,
                        dn_w,
                        dn_scales,
                        dn_biases,
                        dn_outer_scale: dn_outer_arr,
                        dn_mode,
                        route_scale,
                        post_conv_w: Some(&post_conv_w),
                        hist_in: Some(&hist_in),
                        residual: Some(&res_m),
                        conv_taps: Some(taps_arr),
                        update_hist: false,
                        emit_pre_conv: true,
                    })
                    .unwrap()
                    .expect("moe_forward_cuda fused tree must run on CUDA");
                    assert!(out_tree.hist.is_none());
                    assert_close(
                        &format!("moe {label} fused_tree y M={m}"),
                        out_tree.y.to_float32().unwrap().as_ref(),
                        &ref_y_tree,
                        abs_tol,
                        rel_tol,
                    );
                    assert_close(
                        &format!("moe {label} fused_tree pre_conv M={m}"),
                        out_tree
                            .pre_conv
                            .as_ref()
                            .unwrap()
                            .to_float32()
                            .unwrap()
                            .as_ref(),
                        ref_pre_f.as_ref(),
                        abs_tol,
                        rel_tol,
                    );
                }
            }
        };

        // --- Format 1: NVFP4 (split & interleaved, WITH outer_scale on both gate_up & down) ---
        {
            let (gw_b, gs_b, _) = quantize_mlx(&base_gu_bf16, 16, 4, c"nvfp4");
            let (dw_b, ds_b, _) = quantize_mlx(&base_dn_bf16, 16, 4, c"nvfp4");
            let deq_gu_t =
                split_base_t(&dequantize_mlx(&gw_b, &gs_b, None, 16, 4, c"nvfp4"), n2, h);
            let deq_dn_t = split_base_t(
                &dequantize_mlx(&dw_b, &ds_b, None, 16, 4, c"nvfp4"),
                h,
                inter,
            );
            let gw = gw_b.tile(&tile_reps).unwrap();
            let gs = gs_b.tile(&tile_reps).unwrap();
            let dw = dw_b.tile(&tile_reps).unwrap();
            let ds = ds_b.tile(&tile_reps).unwrap();
            let gu_outer_v: Vec<f32> = (0..e).map(|i| 0.6 + 0.004 * i as f32).collect();
            let dn_outer_v: Vec<f32> = (0..e).map(|i| 1.1 - 0.003 * i as f32).collect();
            let gu_outer = f32_arr(&gu_outer_v, &[e as i64]);
            let dn_outer = f32_arr(&dn_outer_v, &[e as i64]);

            verify_format(
                "NVFP4_split",
                &gw,
                &gs,
                None,
                &dw,
                &ds,
                None,
                &deq_gu_t,
                &deq_dn_t,
                MoeQuantMode::Nvfp4,
                MoeQuantMode::Nvfp4,
                false,
                Some((&gu_outer, &gu_outer_v)),
                Some((&dn_outer, &dn_outer_v)),
                &[1, 8, 16, 48],
                true,
                0.08,
                1.5e-2,
            );
            verify_format(
                "NVFP4_interleaved",
                &gw,
                &gs,
                None,
                &dw,
                &ds,
                None,
                &deq_gu_t,
                &deq_dn_t,
                MoeQuantMode::Nvfp4,
                MoeQuantMode::Nvfp4,
                true,
                Some((&gu_outer, &gu_outer_v)),
                Some((&dn_outer, &dn_outer_v)),
                &[1, 8, 16, 48],
                false,
                0.08,
                1.5e-2,
            );
        }
        unsafe { sys::mlx_clear_cache() };

        // --- Format 2: AffineQ8 (BF16 sidecars) ---
        {
            let (gw_b, gs_b, gb_b) = quantize_mlx(&base_gu_bf16, 32, 8, c"affine");
            let (dw_b, ds_b, db_b) = quantize_mlx(&base_dn_bf16, 32, 8, c"affine");
            let gs_bf_b = gs_b.astype(DType::BFloat16).unwrap();
            let gb_bf_b = gb_b.unwrap().astype(DType::BFloat16).unwrap();
            let ds_bf_b = ds_b.astype(DType::BFloat16).unwrap();
            let db_bf_b = db_b.unwrap().astype(DType::BFloat16).unwrap();
            let deq_gu_t = split_base_t(
                &dequantize_mlx(&gw_b, &gs_bf_b, Some(&gb_bf_b), 32, 8, c"affine"),
                n2,
                h,
            );
            let deq_dn_t = split_base_t(
                &dequantize_mlx(&dw_b, &ds_bf_b, Some(&db_bf_b), 32, 8, c"affine"),
                h,
                inter,
            );
            let gw = gw_b.tile(&tile_reps).unwrap();
            let gs = gs_bf_b.tile(&tile_reps).unwrap();
            let gb = gb_bf_b.tile(&tile_reps).unwrap();
            let dw = dw_b.tile(&tile_reps).unwrap();
            let ds = ds_bf_b.tile(&tile_reps).unwrap();
            let db = db_bf_b.tile(&tile_reps).unwrap();
            verify_format(
                "AffineQ8",
                &gw,
                &gs,
                Some(&gb),
                &dw,
                &ds,
                Some(&db),
                &deq_gu_t,
                &deq_dn_t,
                MoeQuantMode::AffineQ8,
                MoeQuantMode::AffineQ8,
                false,
                None,
                None,
                &[1, 8, 16, 48],
                true,
                0.15,
                1.0e-2,
            );
        }
        unsafe { sys::mlx_clear_cache() };

        // --- Format 3: SymQ8 (F16 scales, biases = None) ---
        let (sym_dn_w, sym_dn_s, sym_deq_dn_t) = {
            let base_gu_f = base_gu_bf16.to_float32().unwrap();
            let base_dn_f = base_dn_bf16.to_float32().unwrap();
            let (gw_2d, gs_2d, gb_2d) = quantize_sym_q8_f16(base_gu_f.as_ref(), e_base * n2, h);
            let (dw_2d, ds_2d, db_2d) = quantize_sym_q8_f16(base_dn_f.as_ref(), e_base * h, inter);
            let gw_b = gw_2d
                .reshape(&[e_base as i64, n2 as i64, (h / 4) as i64])
                .unwrap();
            let gs_b = gs_2d
                .reshape(&[e_base as i64, n2 as i64, (h / 32) as i64])
                .unwrap();
            let gb_b = gb_2d
                .reshape(&[e_base as i64, n2 as i64, (h / 32) as i64])
                .unwrap();
            let dw_b = dw_2d
                .reshape(&[e_base as i64, h as i64, (inter / 4) as i64])
                .unwrap();
            let ds_b = ds_2d
                .reshape(&[e_base as i64, h as i64, (inter / 32) as i64])
                .unwrap();
            let db_b = db_2d
                .reshape(&[e_base as i64, h as i64, (inter / 32) as i64])
                .unwrap();
            let deq_gu_t = split_base_t(
                &dequantize_mlx(&gw_b, &gs_b, Some(&gb_b), 32, 8, c"affine"),
                n2,
                h,
            );
            let deq_dn_t = split_base_t(
                &dequantize_mlx(&dw_b, &ds_b, Some(&db_b), 32, 8, c"affine"),
                h,
                inter,
            );
            let gw = gw_b.tile(&tile_reps).unwrap();
            let gs = gs_b.tile(&tile_reps).unwrap();
            let dw = dw_b.tile(&tile_reps).unwrap();
            let ds = ds_b.tile(&tile_reps).unwrap();
            verify_format(
                "SymQ8",
                &gw,
                &gs,
                None,
                &dw,
                &ds,
                None,
                &deq_gu_t,
                &deq_dn_t,
                MoeQuantMode::SymQ8,
                MoeQuantMode::SymQ8,
                false,
                None,
                None,
                &[1, 8, 16, 48],
                true,
                0.15,
                1.0e-2,
            );
            (dw, ds, deq_dn_t)
        };
        unsafe { sys::mlx_clear_cache() };

        // --- Format 4: Q5_K gate_up + SymQ8 down (split & interleaved) ---
        {
            let n_groups = h / 32; // 72
            let n_super = h / 256; // 9
            let mut q5_w = vec![0u32; e_base * n2 * n_groups * 5];
            let mut q5_s = vec![0u8; e_base * n2 * n_groups * 2];
            let mut q5_b = vec![0.0f32; e_base * n2 * n_super * 2];
            let mut deq_q5 = vec![0.0f32; e_base * n2 * h];
            for row in 0..(e_base * n2) {
                for sb in 0..n_super {
                    let d_f16 = f16::from_f32(0.00012 + (rng.next_u64() % 8) as f32 * 0.000015);
                    let dmin_f16 = f16::from_f32(d_f16.to_f32() * 15.5);
                    let d_val = d_f16.to_f32();
                    let dmin_val = dmin_f16.to_f32();
                    q5_b[(row * n_super + sb) * 2] = d_val;
                    q5_b[(row * n_super + sb) * 2 + 1] = dmin_val;
                    for u in 0..8usize {
                        let g = sb * 8 + u;
                        let sc = 10u8 + (rng.next_u64() % 40) as u8;
                        q5_s[(row * n_groups + g) * 2] = sc;
                        q5_s[(row * n_groups + g) * 2 + 1] = sc;
                        let scale = d_val * f32::from(sc);
                        let min_v = dmin_val * f32::from(sc);
                        let mut words = [0u32; 5];
                        for i in 0..32usize {
                            let code = (rng.next_u64() & 31) as u32;
                            deq_q5[row * h + g * 32 + i] = scale * (code as f32) - min_v;
                            let bit = i * 5;
                            let wi = bit / 32;
                            let bo = bit % 32;
                            words[wi] |= code << bo;
                            if bo + 5 > 32 {
                                words[wi + 1] |= code >> (32 - bo);
                            }
                        }
                        let base = (row * n_groups + g) * 5;
                        q5_w[base..base + 5].copy_from_slice(&words);
                    }
                }
            }
            let deq_gu_t: Vec<MxArray> = (0..e_base)
                .map(|be| {
                    f32_arr(
                        &deq_q5[be * n2 * h..(be + 1) * n2 * h],
                        &[n2 as i64, h as i64],
                    )
                    .transpose(None)
                    .unwrap()
                })
                .collect();
            let gu_w =
                MxArray::from_uint32(&q5_w, &[e_base as i64, n2 as i64, (n_groups * 5) as i64])
                    .unwrap()
                    .tile(&tile_reps)
                    .unwrap();
            let gu_s =
                MxArray::from_uint8(&q5_s, &[e_base as i64, n2 as i64, (n_groups * 2) as i64])
                    .unwrap()
                    .tile(&tile_reps)
                    .unwrap();
            let gu_b = f32_arr(&q5_b, &[e_base as i64, n2 as i64, (n_super * 2) as i64])
                .astype(DType::Float16)
                .unwrap()
                .tile(&tile_reps)
                .unwrap();
            verify_format(
                "Q5K_SymQ8_split",
                &gu_w,
                &gu_s,
                Some(&gu_b),
                &sym_dn_w,
                &sym_dn_s,
                None,
                &deq_gu_t,
                &sym_deq_dn_t,
                MoeQuantMode::Q5K,
                MoeQuantMode::SymQ8,
                false,
                None,
                None,
                &[1, 8, 16, 48],
                true,
                0.25,
                2.0e-2,
            );
            verify_format(
                "Q5K_SymQ8_interleaved",
                &gu_w,
                &gu_s,
                Some(&gu_b),
                &sym_dn_w,
                &sym_dn_s,
                None,
                &deq_gu_t,
                &sym_deq_dn_t,
                MoeQuantMode::Q5K,
                MoeQuantMode::SymQ8,
                true,
                None,
                None,
                &[1, 8, 48],
                false,
                0.25,
                2.0e-2,
            );
        }
        drop((sym_dn_w, sym_dn_s, sym_deq_dn_t));
        unsafe { sys::mlx_clear_cache() };

        // --- Format 5: MXFP8 (interleaved, SafeTensors Inkling-2-Mini L20..23) ---
        {
            let (gw_b, gs_b, _) = quantize_mlx(&base_gu_bf16, 32, 8, c"mxfp8");
            let (dw_b, ds_b, _) = quantize_mlx(&base_dn_bf16, 32, 8, c"mxfp8");
            let deq_gu_t =
                split_base_t(&dequantize_mlx(&gw_b, &gs_b, None, 32, 8, c"mxfp8"), n2, h);
            let deq_dn_t = split_base_t(
                &dequantize_mlx(&dw_b, &ds_b, None, 32, 8, c"mxfp8"),
                h,
                inter,
            );
            let gw = gw_b.tile(&tile_reps).unwrap();
            let gs = gs_b.tile(&tile_reps).unwrap();
            let dw = dw_b.tile(&tile_reps).unwrap();
            let ds = ds_b.tile(&tile_reps).unwrap();
            verify_format(
                "MXFP8_interleaved",
                &gw,
                &gs,
                None,
                &dw,
                &ds,
                None,
                &deq_gu_t,
                &deq_dn_t,
                MoeQuantMode::Mxfp8,
                MoeQuantMode::Mxfp8,
                true,
                None,
                None,
                &[1, 8, 16, 48],
                true,
                0.08,
                1.5e-2,
            );
        }
        unsafe { sys::mlx_clear_cache() };
    }

    #[test]
    fn cuda_small_bf16_proj_matches_reference() {
        if !available() {
            assert!(!host_has_cuda(), "CUDA host must enable Inkling2 CUDA ops");
            return;
        }
        let mut rng = Rng::new(0x5A11_BF16);
        let k = 2304usize;
        for &n in &[1usize, 24, 32, 128] {
            let w_vals = rng.vec(n * k, 0.35);
            let w_bf16 = bf16_arr(&w_vals, &[n as i64, k as i64]);
            let w_f32_t = w_bf16
                .astype(DType::Float32)
                .unwrap()
                .transpose(None)
                .unwrap();

            for &m in &[1usize, 2, 4, 8, 12, 16] {
                let x_vals = rng.vec(m * k, 0.8);
                let x_bf16 = bf16_arr(&x_vals, &[m as i64, k as i64]);
                let got_arr = small_bf16_proj_cuda(&x_bf16, &w_bf16)
                    .unwrap()
                    .expect("small_bf16_proj_cuda should dispatch");
                assert_eq!(got_arr.dtype().unwrap(), DType::BFloat16);
                assert_eq!(got_arr.shape().unwrap().as_ref(), &[m as i64, n as i64]);

                let want_arr = x_bf16
                    .astype(DType::Float32)
                    .unwrap()
                    .matmul(&w_f32_t)
                    .unwrap()
                    .astype(DType::BFloat16)
                    .unwrap();
                let got = got_arr
                    .astype(DType::Float32)
                    .unwrap()
                    .to_float32()
                    .unwrap();
                let want = want_arr
                    .astype(DType::Float32)
                    .unwrap()
                    .to_float32()
                    .unwrap();
                assert_close(
                    &format!("small_bf16_proj M={m} N={n} K={k}"),
                    got.as_ref(),
                    want.as_ref(),
                    0.0625,
                    1.0e-3,
                );
                // Verify <= 1 BF16 ULP across all elements (FP32 FMA vs matmul accumulation order).
                let got_bits: Vec<u16> = got.iter().map(|&v| bf16::from_f32(v).to_bits()).collect();
                let want_bits: Vec<u16> =
                    want.iter().map(|&v| bf16::from_f32(v).to_bits()).collect();
                let max_ulp = got_bits
                    .iter()
                    .zip(&want_bits)
                    .map(|(&a, &b)| (a as i32 - b as i32).abs())
                    .max()
                    .unwrap_or(0);
                assert!(
                    max_ulp <= 1,
                    "small_bf16_proj M={m} N={n} K={k}: max_ulp={max_ulp} > 1"
                );
            }
        }

        // Verify 0-ULP batch invariance: M=1 row 0 must be bit-identical to M=8 and M=16 row 0.
        let w24 = bf16_arr(&rng.vec(24 * k, 0.35), &[24, k as i64]);
        let x16_vals = rng.vec(16 * k, 0.8);
        let x16 = bf16_arr(&x16_vals, &[16, k as i64]);
        let x8 = x16.slice(&[0, 0], &[8, k as i64]).unwrap();
        let x1 = x16.slice(&[0, 0], &[1, k as i64]).unwrap();
        let y1 = small_bf16_proj_cuda(&x1, &w24).unwrap().unwrap();
        let y8 = small_bf16_proj_cuda(&x8, &w24)
            .unwrap()
            .unwrap()
            .slice(&[0, 0], &[1, 24])
            .unwrap();
        let y16 = small_bf16_proj_cuda(&x16, &w24)
            .unwrap()
            .unwrap()
            .slice(&[0, 0], &[1, 24])
            .unwrap();
        let b1: Vec<u32> = y1
            .astype(DType::Float32)
            .unwrap()
            .to_float32()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let b8: Vec<u32> = y8
            .astype(DType::Float32)
            .unwrap()
            .to_float32()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let b16: Vec<u32> = y16
            .astype(DType::Float32)
            .unwrap()
            .to_float32()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(
            b1, b8,
            "small_bf16_proj M=1 vs M=8 row 0 must be 0-ULP bit-identical"
        );
        assert_eq!(
            b1, b16,
            "small_bf16_proj M=1 vs M=16 row 0 must be 0-ULP bit-identical"
        );

        // Off-contract decline checks: M=17 or N=129 must cleanly return Ok(None).
        let x17 = bf16_arr(&vec![0.1; 17 * 64], &[17, 64]);
        let w24_64 = bf16_arr(&vec![0.1; 24 * 64], &[24, 64]);
        assert!(small_bf16_proj_cuda(&x17, &w24_64).unwrap().is_none());
        let x1 = bf16_arr(&vec![0.1; 64], &[1, 64]);
        let w129 = bf16_arr(&vec![0.1; 129 * 64], &[129, 64]);
        assert!(small_bf16_proj_cuda(&x1, &w129).unwrap().is_none());
        unsafe { sys::mlx_clear_cache() };
    }

    #[test]
    fn cuda_commit_short_conv_tapes_matches_reference_contiguous_and_tree() {
        if !available() {
            assert!(!host_has_cuda(), "CUDA host must enable Inkling2 CUDA ops");
            return;
        }
        let mut rng = Rng::new(0xC044_1755);
        // Test both Inkling-2-Mini stream widths (`C=1024` for K/V, `C=2304` for attn/mlp)
        // at full 96-stream single-launch batch size, WITH strided rows_in slices!
        for &(n_streams, c, t) in &[(96usize, 1024usize, 12usize), (96usize, 2304usize, 8usize)] {
            let stride_cols = c + 128; // force non-contiguous strided slice view!
            let mut hists = Vec::with_capacity(n_streams);
            let mut rows_strided = Vec::with_capacity(n_streams);
            for _ in 0..n_streams {
                let h = bf16_arr(&rng.vec(3 * c, 2.0), &[3, c as i64]);
                let big = bf16_arr(
                    &rng.vec(t * stride_cols, 2.0),
                    &[t as i64, stride_cols as i64],
                );
                let sliced = big.slice(&[0, 64], &[t as i64, (64 + c) as i64]).unwrap();
                hists.push(h);
                rows_strided.push(sliced);
            }
            let h_refs: Vec<&MxArray> = hists.iter().collect();
            let r_refs: Vec<&MxArray> = rows_strided.iter().collect();

            // 1. Contiguous prefix commits across keep in {0, 1, 2, 3, 5, t}.
            for &keep in &[0usize, 1, 2, 3, 5, t] {
                let outs = commit_short_conv_tapes_cuda(&h_refs, &r_refs, None, keep)
                    .unwrap()
                    .expect("commit_short_conv_tapes_cuda contiguous should dispatch");
                assert_eq!(outs.len(), n_streams);
                for s in [0usize, 1, 47, 48, n_streams - 1] {
                    let tape = crate::models::inkling2::short_conv::StreamShortConvTape {
                        hist_before: hists[s].clone(),
                        rows: rows_strided[s].clone(),
                    };
                    let want = tape.commit_prefix(keep).unwrap();
                    let got_f = outs[s]
                        .astype(DType::Float32)
                        .unwrap()
                        .to_float32()
                        .unwrap();
                    let want_f = want.astype(DType::Float32).unwrap().to_float32().unwrap();
                    assert_eq!(
                        got_f.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        want_f.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        "contiguous mismatch C={c} keep={keep} stream={s}"
                    );
                }
            }

            // 2. Non-contiguous tree branch commits across keep in {1, 2, 3, 5}.
            let tree_paths: &[&[u32]] = &[&[0], &[0, 4], &[0, 2, 5], &[0, 2, 4, 6, 7]];
            for &path in tree_paths {
                let idx_i32: Vec<i32> = path.iter().map(|&v| v as i32).collect();
                let nodes_arr = MxArray::from_int32(&idx_i32, &[path.len() as i64]).unwrap();
                let outs = commit_short_conv_tapes_cuda(&h_refs, &r_refs, Some(&nodes_arr), 0)
                    .unwrap()
                    .expect("commit_short_conv_tapes_cuda tree should dispatch");
                assert_eq!(outs.len(), n_streams);
                for s in [0usize, 1, 47, 48, n_streams - 1] {
                    let tape = crate::models::inkling2::short_conv::StreamShortConvTape {
                        hist_before: hists[s].clone(),
                        rows: rows_strided[s].clone(),
                    };
                    let want = tape.commit_tree(path, false).unwrap();
                    let got_f = outs[s]
                        .astype(DType::Float32)
                        .unwrap()
                        .to_float32()
                        .unwrap();
                    let want_f = want.astype(DType::Float32).unwrap().to_float32().unwrap();
                    assert_eq!(
                        got_f.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        want_f.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        "tree mismatch C={c} path={path:?} stream={s}"
                    );
                }
            }
        }
        unsafe { sys::mlx_clear_cache() };
    }

    #[test]
    fn cuda_embed_rmsnorm_matches_reference_bf16_symq8_and_affineq8() {
        if !available() {
            assert!(!host_has_cuda(), "CUDA host must enable Inkling2 CUDA ops");
            return;
        }
        let mut rng = Rng::new(0xE4BE_D901);
        let vocab = 256usize;
        let d = 2304usize;
        let eps = 1e-6f32;
        let w_vals = rng.vec(vocab * d, 0.75);
        let w_bf16 = bf16_arr(&w_vals, &[vocab as i64, d as i64]);
        let nw_vals: Vec<f32> = (0..d).map(|_| rng.uniform(0.5, 1.5)).collect();
        let norm_w = f32_arr(&nw_vals, &[d as i64]);

        let ids_1d_vals: Vec<i32> = vec![0, 7, 42, 255, 13, 128, 199, 3];
        let ids_1d = MxArray::from_int32(&ids_1d_vals, &[ids_1d_vals.len() as i64]).unwrap();
        let ids_2d = MxArray::from_int32(&ids_1d_vals, &[1, ids_1d_vals.len() as i64]).unwrap();

        // 1. Dense BF16: 1D [T] and 2D [1, T], exact 0-ULP vs FP32 RMSNorm.
        for (label, ids, expected_shape) in [
            ("bf16_1d", &ids_1d, vec![ids_1d_vals.len() as i64, d as i64]),
            (
                "bf16_2d",
                &ids_2d,
                vec![1, ids_1d_vals.len() as i64, d as i64],
            ),
        ] {
            let got = embed_rmsnorm_cuda(ids, &w_bf16, None, None, &norm_w, eps)
                .unwrap()
                .expect("embed_rmsnorm_cuda BF16 should dispatch");
            assert_eq!(got.dtype().unwrap(), DType::BFloat16);
            assert_eq!(got.shape().unwrap().as_ref(), expected_shape.as_slice());

            let gathered = w_bf16.take(ids, 0).unwrap();
            let want = apply_sublayer_rms_norm(&gathered, &norm_w, eps, DType::BFloat16).unwrap();
            let got_f = got.astype(DType::Float32).unwrap().to_float32().unwrap();
            let want_f = want.astype(DType::Float32).unwrap().to_float32().unwrap();
            let max_ulp = got_f
                .iter()
                .zip(want_f.iter())
                .map(|(&a, &b)| {
                    (bf16::from_f32(a).to_bits() as i32 - bf16::from_f32(b).to_bits() as i32).abs()
                })
                .max()
                .unwrap_or(0);
            assert!(max_ulp <= 1, "{label}: max_ulp={max_ulp} > 1");
        }

        // 2. SymQ8 (F16 scales, biases = None) & AffineQ8 (F16 scales + F16 biases, GGUF layout).
        let (q_w, q_s_f16, q_b_f16) = quantize_sym_q8_f16(&w_vals, vocab, d);
        let deq_f32 = dequantize_mlx(&q_w, &q_s_f16, Some(&q_b_f16), 32, 8, c"affine");
        let want_q8 = apply_sublayer_rms_norm(
            &deq_f32.take(&ids_1d, 0).unwrap(),
            &norm_w,
            eps,
            DType::BFloat16,
        )
        .unwrap();
        let want_q8_f = want_q8
            .astype(DType::Float32)
            .unwrap()
            .to_float32()
            .unwrap();

        for (label, scales, biases) in [
            ("sym_q8_f16", &q_s_f16, None),
            ("affine_q8_f16", &q_s_f16, Some(&q_b_f16)),
        ] {
            let got = embed_rmsnorm_cuda(&ids_1d, &q_w, Some(scales), biases, &norm_w, eps)
                .unwrap()
                .expect("embed_rmsnorm_cuda Q8 should dispatch");
            assert_eq!(got.dtype().unwrap(), DType::BFloat16);
            assert_eq!(
                got.shape().unwrap().as_ref(),
                &[ids_1d_vals.len() as i64, d as i64]
            );
            let got_f = got.astype(DType::Float32).unwrap().to_float32().unwrap();
            assert_close(label, got_f.as_ref(), want_q8_f.as_ref(), 0.03125, 2.0e-3);
            let max_ulp = got_f
                .iter()
                .zip(want_q8_f.iter())
                .map(|(&a, &b)| {
                    (bf16::from_f32(a).to_bits() as i32 - bf16::from_f32(b).to_bits() as i32).abs()
                })
                .max()
                .unwrap_or(0);
            assert!(max_ulp <= 1, "{label}: max_ulp={max_ulp} > 1");
        }

        // 3. Asymmetric AffineQ8 from mlx_quantize (BF16 scales + biases) on 2D [1, T].
        let (aq_w, aq_s, aq_b_opt) = quantize_mlx(&w_bf16, 32, 8, c"affine");
        let aq_b = aq_b_opt.expect("affine biases");
        let got_aq = embed_rmsnorm_cuda(&ids_2d, &aq_w, Some(&aq_s), Some(&aq_b), &norm_w, eps)
            .unwrap()
            .expect("embed_rmsnorm_cuda AffineQ8 BF16 should dispatch");
        assert_eq!(
            got_aq.shape().unwrap().as_ref(),
            &[1, ids_1d_vals.len() as i64, d as i64]
        );
        let deq_aq = dequantize_mlx(&aq_w, &aq_s, Some(&aq_b), 32, 8, c"affine");
        let want_aq = apply_sublayer_rms_norm(
            &deq_aq.take(&ids_2d, 0).unwrap(),
            &norm_w,
            eps,
            DType::BFloat16,
        )
        .unwrap();
        let got_aq_f = got_aq.astype(DType::Float32).unwrap().to_float32().unwrap();
        let want_aq_f = want_aq
            .astype(DType::Float32)
            .unwrap()
            .to_float32()
            .unwrap();
        let max_ulp = got_aq_f
            .iter()
            .zip(want_aq_f.iter())
            .map(|(&a, &b)| {
                (bf16::from_f32(a).to_bits() as i32 - bf16::from_f32(b).to_bits() as i32).abs()
            })
            .max()
            .unwrap_or(0);
        assert!(max_ulp <= 1, "affine_q8_bf16_2d: max_ulp={max_ulp} > 1");

        unsafe { sys::mlx_clear_cache() };
    }

    #[test]
    fn cuda_compact_kv_batch_matches_reference_mixed_dtypes_and_shifts() {
        if !available() {
            assert!(!host_has_cuda(), "CUDA host must enable Inkling2 CUDA ops");
            return;
        }
        let mut rng = Rng::new(0xC049_AC75);
        let h_kv = 8usize;
        let d_kv = 128usize;

        // Exercise mixed BF16 + INT8 + F32 4D scales + F32 3D scales across >96 caches (104 caches
        // -> tests BOTH chunk 0 [96 caches] and chunk 1 [8 caches] in a single primitive!), with
        // distinct per-cache capacities and write_offsets, and overlapping/permuted node shifts!
        #[derive(Clone, Copy)]
        enum Kind {
            Bf16,
            Int8,
            Scale4D,
            Scale3D,
        }
        struct Entry {
            kind: Kind,
            cap: usize,
            write_off: i32,
            arr: MxArray,
            bf16_ref: Vec<u16>,
            i8_ref: Vec<i8>,
            f32_ref: Vec<f32>,
        }

        let specs: &[(Kind, usize, i32)] = &[
            (Kind::Bf16, 48, 5),
            (Kind::Bf16, 64, 17),
            (Kind::Int8, 40, 3),
            (Kind::Int8, 56, 21),
            (Kind::Scale4D, 40, 3),
            (Kind::Scale4D, 56, 21),
            (Kind::Scale3D, 48, 9),
            (Kind::Bf16, 32, 0),
        ];
        let mut entries: Vec<Entry> = Vec::with_capacity(104);
        for i in 0..104usize {
            let (kind, cap, write_off) = specs[i % specs.len()];
            match kind {
                Kind::Bf16 => {
                    let vals = rng.vec(h_kv * cap * d_kv, 3.0);
                    let bits: Vec<u16> =
                        vals.iter().map(|&v| bf16::from_f32(v).to_bits()).collect();
                    let arr =
                        MxArray::from_bfloat16(&bits, &[1, h_kv as i64, cap as i64, d_kv as i64])
                            .unwrap();
                    arr.eval();
                    entries.push(Entry {
                        kind,
                        cap,
                        write_off,
                        arr,
                        bf16_ref: bits,
                        i8_ref: Vec::new(),
                        f32_ref: Vec::new(),
                    });
                }
                Kind::Int8 => {
                    let vals: Vec<i8> = (0..h_kv * cap * d_kv)
                        .map(|_| ((rng.next_u64() % 251) as i32 - 125) as i8)
                        .collect();
                    let arr = MxArray::from_int8(&vals, &[1, h_kv as i64, cap as i64, d_kv as i64])
                        .unwrap();
                    arr.eval();
                    entries.push(Entry {
                        kind,
                        cap,
                        write_off,
                        arr,
                        bf16_ref: Vec::new(),
                        i8_ref: vals,
                        f32_ref: Vec::new(),
                    });
                }
                Kind::Scale4D => {
                    let vals: Vec<f32> = (0..h_kv * cap).map(|_| rng.uniform(0.01, 2.0)).collect();
                    let arr =
                        MxArray::from_float32(&vals, &[1, h_kv as i64, cap as i64, 1]).unwrap();
                    arr.eval();
                    entries.push(Entry {
                        kind,
                        cap,
                        write_off,
                        arr,
                        bf16_ref: Vec::new(),
                        i8_ref: Vec::new(),
                        f32_ref: vals,
                    });
                }
                Kind::Scale3D => {
                    let vals: Vec<f32> = (0..h_kv * cap).map(|_| rng.uniform(0.01, 2.0)).collect();
                    let arr = MxArray::from_float32(&vals, &[1, h_kv as i64, cap as i64]).unwrap();
                    arr.eval();
                    entries.push(Entry {
                        kind,
                        cap,
                        write_off,
                        arr,
                        bf16_ref: Vec::new(),
                        i8_ref: Vec::new(),
                        f32_ref: vals,
                    });
                }
            }
        }

        // Run two sequential overlapping tree compactions (including a reverse-order permutation
        // that would corrupt data if src reads weren't buffered before dst writes!).
        let shift_cases: &[&[i32]] = &[&[0, 2, 5, 7, 10], &[4, 2, 0, 3, 1, 8]];
        for &nodes in shift_cases {
            let nodes_arr = MxArray::from_int32(nodes, &[nodes.len() as i64]).unwrap();
            let refs: Vec<&MxArray> = entries.iter().map(|e| &e.arr).collect();
            let offs: Vec<i32> = entries.iter().map(|e| e.write_off).collect();
            let outs = compact_kv_batch_cuda(&refs, &offs, &nodes_arr)
                .unwrap()
                .expect("compact_kv_batch_cuda must admit 104 mixed caches");
            assert_eq!(outs.len(), entries.len());
            for (e, out) in entries.iter_mut().zip(outs) {
                e.arr = out;
                // Apply exact reference compaction on host buffers (load all first, then store).
                let base = e.write_off as usize;
                let cap = e.cap;
                let k_len = nodes.len();
                match e.kind {
                    Kind::Bf16 => {
                        for h in 0..h_kv {
                            let saved: Vec<Vec<u16>> = nodes
                                .iter()
                                .map(|&n| {
                                    let s = (h * cap + base + n as usize) * d_kv;
                                    e.bf16_ref[s..s + d_kv].to_vec()
                                })
                                .collect();
                            for (i, row) in saved.iter().enumerate().take(k_len) {
                                let d = (h * cap + base + i) * d_kv;
                                e.bf16_ref[d..d + d_kv].copy_from_slice(row);
                            }
                        }
                    }
                    Kind::Int8 => {
                        for h in 0..h_kv {
                            let saved: Vec<Vec<i8>> = nodes
                                .iter()
                                .map(|&n| {
                                    let s = (h * cap + base + n as usize) * d_kv;
                                    e.i8_ref[s..s + d_kv].to_vec()
                                })
                                .collect();
                            for (i, row) in saved.iter().enumerate().take(k_len) {
                                let d = (h * cap + base + i) * d_kv;
                                e.i8_ref[d..d + d_kv].copy_from_slice(row);
                            }
                        }
                    }
                    Kind::Scale4D | Kind::Scale3D => {
                        for h in 0..h_kv {
                            let saved: Vec<f32> = nodes
                                .iter()
                                .map(|&n| e.f32_ref[h * cap + base + n as usize])
                                .collect();
                            for (i, &val) in saved.iter().enumerate().take(k_len) {
                                e.f32_ref[h * cap + base + i] = val;
                            }
                        }
                    }
                }
            }
            let eval_refs: Vec<&MxArray> = entries.iter().map(|e| &e.arr).collect();
            MxArray::eval_arrays_with_context(&eval_refs, "test_compact_kv_batch_eval").unwrap();

            // Verify every checked slot in chunk 0 (0..96) AND chunk 1 (96..104) is bit-exact!
            for idx in [
                0usize, 1, 2, 3, 4, 5, 6, 7, 47, 95, 96, 97, 98, 100, 102, 103,
            ] {
                let e = &entries[idx];
                match e.kind {
                    Kind::Bf16 => {
                        let got = e.arr.to_uint16_native().unwrap();
                        assert_eq!(
                            got.as_slice(),
                            e.bf16_ref.as_slice(),
                            "BF16 compact mismatch at cache idx={idx}, nodes={nodes:?}"
                        );
                    }
                    Kind::Int8 => {
                        let got_f = e.arr.astype(DType::Float32).unwrap().to_float32().unwrap();
                        let want_f: Vec<f32> = e.i8_ref.iter().map(|&v| f32::from(v)).collect();
                        assert_eq!(
                            got_f.as_ref(),
                            want_f.as_slice(),
                            "INT8 compact mismatch at cache idx={idx}, nodes={nodes:?}"
                        );
                    }
                    Kind::Scale4D | Kind::Scale3D => {
                        let got = e.arr.to_float32().unwrap();
                        assert_eq!(
                            got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            e.f32_ref.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            "F32 scale compact mismatch at cache idx={idx}, nodes={nodes:?}"
                        );
                    }
                }
            }
        }

        // Guard checks: empty slice, mismatched offsets, out-of-bounds offset decline cleanly.
        let nodes_ok = MxArray::from_int32(&[0, 2], &[2]).unwrap();
        assert!(
            compact_kv_batch_cuda(&[], &[], &nodes_ok)
                .unwrap()
                .is_none()
        );
        assert!(
            compact_kv_batch_cuda(&[&entries[0].arr], &[0, 1], &nodes_ok)
                .unwrap()
                .is_none()
        );
        assert!(
            compact_kv_batch_cuda(&[&entries[0].arr], &[entries[0].cap as i32], &nodes_ok)
                .unwrap()
                .is_none()
        );

        unsafe { sys::mlx_clear_cache() };
    }

    #[test]
    fn cuda_compact_kv_batch_multichunk_cuda_graph_dependencies() {
        if !available() {
            assert!(!host_has_cuda(), "CUDA host must enable Inkling2 CUDA ops");
            return;
        }
        let mut rng = Rng::new(0xC049_68A9);
        let h_kv = 8usize;
        let d_kv = 128usize;
        let cap = 32usize;

        // 192 arrays (exact Inkling-2-Mini INT8 KV layout):
        //   chunk 0 (0..96):   48 INT8 K + 48 INT8 V caches
        //   chunk 1 (96..192): 48 F32 K scales + 48 F32 V scales
        let mut caches: Vec<MxArray> = Vec::with_capacity(192);
        let mut offs: Vec<i32> = Vec::with_capacity(192);
        let mut i8_refs: Vec<Vec<i8>> = Vec::with_capacity(96);
        let mut f32_refs: Vec<Vec<f32>> = Vec::with_capacity(96);

        for i in 0..96usize {
            let off = ((i * 3) % 12) as i32;
            let vals: Vec<i8> = (0..h_kv * cap * d_kv)
                .map(|_| ((rng.next_u64() % 241) as i32 - 120) as i8)
                .collect();
            let arr =
                MxArray::from_int8(&vals, &[1, h_kv as i64, cap as i64, d_kv as i64]).unwrap();
            arr.eval();
            caches.push(arr);
            offs.push(off);
            i8_refs.push(vals);
        }
        for i in 0..96usize {
            let off = ((i * 3) % 12) as i32;
            let vals: Vec<f32> = (0..h_kv * cap).map(|_| rng.uniform(0.05, 4.0)).collect();
            let arr = MxArray::from_float32(&vals, &[1, h_kv as i64, cap as i64, 1]).unwrap();
            arr.eval();
            caches.push(arr);
            offs.push(off);
            f32_refs.push(vals);
        }

        // Helper applying exact host compaction for two chained node sets.
        let apply_host = |i8_refs: &mut [Vec<i8>], f32_refs: &mut [Vec<f32>], nodes: &[i32]| {
            let k_len = nodes.len();
            for (idx, buf) in i8_refs.iter_mut().enumerate() {
                let base = offs[idx] as usize;
                for h in 0..h_kv {
                    let saved: Vec<Vec<i8>> = nodes
                        .iter()
                        .map(|&n| {
                            let s = (h * cap + base + n as usize) * d_kv;
                            buf[s..s + d_kv].to_vec()
                        })
                        .collect();
                    for (i, row) in saved.iter().enumerate().take(k_len) {
                        let d = (h * cap + base + i) * d_kv;
                        buf[d..d + d_kv].copy_from_slice(row);
                    }
                }
            }
            for (idx, buf) in f32_refs.iter_mut().enumerate() {
                let base = offs[96 + idx] as usize;
                for h in 0..h_kv {
                    let saved: Vec<f32> = nodes
                        .iter()
                        .map(|&n| buf[h * cap + base + n as usize])
                        .collect();
                    for (i, &val) in saved.iter().enumerate().take(k_len) {
                        buf[h * cap + base + i] = val;
                    }
                }
            }
        };

        // Repeat 3 rounds (cold CUDA graph capture + cached graph replay).
        // Each round captures into ONE single `eval_arrays_with_context`:
        //   lazy GPU `nodes1` -> compact #1 (chunk 0 + chunk 1)
        //   -> lazy GPU `nodes2` -> compact #2 (chunk 0 + chunk 1)
        //   -> downstream GPU slice/cast/add consumers on BOTH chunk 0 and chunk 1!
        let rounds: &[(&[i32], &[i32])] = &[
            (&[1, 3, 6, 9, 12], &[4, 2, 0, 3, 1]),
            (&[0, 4, 8, 11, 14], &[3, 1, 4, 0, 2]),
            (&[2, 5, 7, 10, 13], &[4, 3, 2, 1, 0]),
        ];
        let checked_idxs = [0usize, 1, 47, 95, 96, 97, 143, 191];
        let zero_f32 = MxArray::scalar_float(0.0).unwrap();
        zero_f32.eval();

        for (round, &(n1_target, n2_target)) in rounds.iter().enumerate() {
            // Build LAZY (unevaluated) GPU `nodes1` and `nodes2` via nontrivial GPU work so
            // missing CUDA graph edges from producer -> chunk 1 would immediately race!
            let work1 = MxArray::ones(&[1024, 64], Some(DType::Float32)).unwrap();
            let one_i32 = work1
                .sum(None, Some(false))
                .unwrap()
                .div_scalar(65536.0)
                .unwrap()
                .astype(DType::Int32)
                .unwrap();
            let base1: Vec<i32> = n1_target.iter().map(|&v| v - 1).collect();
            let base1_arr = MxArray::from_int32(&base1, &[base1.len() as i64]).unwrap();
            let lazy_nodes1 = base1_arr.add(&one_i32).unwrap();

            let work2 = MxArray::ones(&[1024, 64], Some(DType::Float32)).unwrap();
            let two_i32 = work2
                .sum(None, Some(false))
                .unwrap()
                .div_scalar(32768.0)
                .unwrap()
                .astype(DType::Int32)
                .unwrap();
            let base2: Vec<i32> = n2_target.iter().map(|&v| v - 2).collect();
            let base2_arr = MxArray::from_int32(&base2, &[base2.len() as i64]).unwrap();
            let lazy_nodes2 = base2_arr.add(&two_i32).unwrap();

            let refs1: Vec<&MxArray> = caches.iter().collect();
            let out1 = compact_kv_batch_cuda(&refs1, &offs, &lazy_nodes1)
                .unwrap()
                .expect("compact_kv_batch_cuda #1 must admit 192 caches");
            let refs2: Vec<&MxArray> = out1.iter().collect();
            let out2 = compact_kv_batch_cuda(&refs2, &offs, &lazy_nodes2)
                .unwrap()
                .expect("compact_kv_batch_cuda #2 must admit 192 caches");

            // Downstream GPU consumers reading BOTH chunk 0 (0..96) and chunk 1 (96..192)
            // in the SAME graph capture as the two chained multi-chunk compactions.
            let mut consumers: Vec<MxArray> = Vec::with_capacity(checked_idxs.len());
            for &idx in &checked_idxs {
                let start = offs[idx] as i64;
                let end = start + n2_target.len() as i64;
                let slice = out2[idx].slice_axis(2, start, end).unwrap();
                let f32_slice = if slice.dtype().unwrap() == DType::Float32 {
                    slice
                } else {
                    slice.astype(DType::Float32).unwrap()
                };
                consumers.push(f32_slice.add(&zero_f32).unwrap());
            }

            // Evaluate ALL outputs + downstream consumers in ONE SINGLE eval call!
            let mut eval_list: Vec<&MxArray> = Vec::with_capacity(out2.len() + consumers.len());
            eval_list.extend(out2.iter());
            eval_list.extend(consumers.iter());
            MxArray::eval_arrays_with_context(
                &eval_list,
                "test_compact_kv_batch_multichunk_cuda_graph",
            )
            .unwrap();
            caches = out2;

            apply_host(&mut i8_refs, &mut f32_refs, n1_target);
            apply_host(&mut i8_refs, &mut f32_refs, n2_target);

            for (c_pos, &idx) in checked_idxs.iter().enumerate() {
                let base = offs[idx] as usize;
                let k_len = n2_target.len();
                let got = consumers[c_pos].to_float32().unwrap();
                if idx < 96 {
                    let mut want = Vec::with_capacity(h_kv * k_len * d_kv);
                    for h in 0..h_kv {
                        for r in 0..k_len {
                            let s = (h * cap + base + r) * d_kv;
                            want.extend(i8_refs[idx][s..s + d_kv].iter().map(|&v| f32::from(v)));
                        }
                    }
                    assert_eq!(
                        got.as_ref(),
                        want.as_slice(),
                        "round={round} chunk0 INT8 consumer mismatch at idx={idx}"
                    );
                } else {
                    let mut want = Vec::with_capacity(h_kv * k_len);
                    for h in 0..h_kv {
                        for r in 0..k_len {
                            want.push(f32_refs[idx - 96][h * cap + base + r]);
                        }
                    }
                    assert_eq!(
                        got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        "round={round} chunk1 F32 scale consumer mismatch at idx={idx}"
                    );
                }
            }
        }

        unsafe { sys::mlx_clear_cache() };
    }

    #[test]
    fn cuda_commit_short_conv_tapes_multichunk_cuda_graph_dependencies() {
        if !available() {
            assert!(!host_has_cuda(), "CUDA host must enable Inkling2 CUDA ops");
            return;
        }
        let mut rng = Rng::new(0x5C04_68A9);
        // 128 streams (> 96) exercises BOTH chunk 0 (0..96) and chunk 1 (96..128) in one primitive!
        let n_streams = 128usize;
        let c = 256usize;
        let t = 10usize;
        let checked_streams = [0usize, 1, 47, 95, 96, 97, 111, 127];

        let mut hists: Vec<MxArray> = Vec::with_capacity(n_streams);
        let mut rows_base: Vec<MxArray> = Vec::with_capacity(n_streams);
        let mut rows_delta: Vec<MxArray> = Vec::with_capacity(n_streams);
        for _ in 0..n_streams {
            let h = bf16_arr(&rng.vec(3 * c, 2.0), &[3, c as i64]);
            let rb = bf16_arr(&rng.vec(t * c, 1.5), &[t as i64, c as i64]);
            let rd = bf16_arr(&rng.vec(t * c, 0.5), &[t as i64, c as i64]);
            h.eval();
            rb.eval();
            rd.eval();
            hists.push(h);
            rows_base.push(rb);
            rows_delta.push(rd);
        }

        let zero_f32 = MxArray::scalar_float(0.0).unwrap();
        zero_f32.eval();

        // 1. Tree mode: lazy GPU `rows` AND lazy GPU `accepted_nodes` + downstream GPU consumers
        //    on both chunk 0 and chunk 1 captured into ONE single `eval_arrays_with_context`.
        let tree_paths: &[&[u32]] = &[&[0, 2, 5, 8], &[1, 3, 6, 9], &[0, 4, 7]];
        for (round, &path) in tree_paths.iter().enumerate() {
            let lazy_rows: Vec<MxArray> = rows_base
                .iter()
                .zip(rows_delta.iter())
                .map(|(b, d)| b.add(d).unwrap())
                .collect();
            let work = MxArray::ones(&[1024, 64], Some(DType::Float32)).unwrap();
            let one_i32 = work
                .sum(None, Some(false))
                .unwrap()
                .div_scalar(65536.0)
                .unwrap()
                .astype(DType::Int32)
                .unwrap();
            let base_nodes: Vec<i32> = path.iter().map(|&v| v as i32 - 1).collect();
            let base_nodes_arr =
                MxArray::from_int32(&base_nodes, &[base_nodes.len() as i64]).unwrap();
            let lazy_nodes = base_nodes_arr.add(&one_i32).unwrap();

            let h_refs: Vec<&MxArray> = hists.iter().collect();
            let r_refs: Vec<&MxArray> = lazy_rows.iter().collect();
            let outs = commit_short_conv_tapes_cuda(&h_refs, &r_refs, Some(&lazy_nodes), 0)
                .unwrap()
                .expect("commit_short_conv_tapes_cuda must admit 128 streams");
            assert_eq!(outs.len(), n_streams);

            let consumers: Vec<MxArray> = checked_streams
                .iter()
                .map(|&s| {
                    outs[s]
                        .astype(DType::Float32)
                        .unwrap()
                        .add(&zero_f32)
                        .unwrap()
                })
                .collect();
            let eval_refs: Vec<&MxArray> = consumers.iter().collect();
            MxArray::eval_arrays_with_context(
                &eval_refs,
                "test_commit_short_conv_tapes_multichunk_tree_cuda_graph",
            )
            .unwrap();

            for (pos, &s) in checked_streams.iter().enumerate() {
                let ref_rows = rows_base[s].add(&rows_delta[s]).unwrap();
                let tape = crate::models::inkling2::short_conv::StreamShortConvTape {
                    hist_before: hists[s].clone(),
                    rows: ref_rows,
                };
                let want = tape.commit_tree(path, false).unwrap();
                let got_f = consumers[pos].to_float32().unwrap();
                let want_f = want.astype(DType::Float32).unwrap().to_float32().unwrap();
                assert_eq!(
                    got_f.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    want_f.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "tree round={round} path={path:?} mismatch at stream={s}"
                );
            }
        }

        // 2. Contiguous mode (`nodes = None`): lazy GPU `rows` + downstream GPU consumers on
        //    128 streams (both chunk 0 and chunk 1) in ONE single `eval_arrays_with_context`.
        for &keep in &[1usize, 3, 6] {
            let lazy_rows: Vec<MxArray> = rows_base
                .iter()
                .zip(rows_delta.iter())
                .map(|(b, d)| b.add(d).unwrap())
                .collect();
            let h_refs: Vec<&MxArray> = hists.iter().collect();
            let r_refs: Vec<&MxArray> = lazy_rows.iter().collect();
            let outs = commit_short_conv_tapes_cuda(&h_refs, &r_refs, None, keep)
                .unwrap()
                .expect("commit_short_conv_tapes_cuda contiguous must admit 128 streams");
            let consumers: Vec<MxArray> = checked_streams
                .iter()
                .map(|&s| {
                    outs[s]
                        .astype(DType::Float32)
                        .unwrap()
                        .add(&zero_f32)
                        .unwrap()
                })
                .collect();
            let eval_refs: Vec<&MxArray> = consumers.iter().collect();
            MxArray::eval_arrays_with_context(
                &eval_refs,
                "test_commit_short_conv_tapes_multichunk_contig_cuda_graph",
            )
            .unwrap();

            for (pos, &s) in checked_streams.iter().enumerate() {
                let ref_rows = rows_base[s].add(&rows_delta[s]).unwrap();
                let tape = crate::models::inkling2::short_conv::StreamShortConvTape {
                    hist_before: hists[s].clone(),
                    rows: ref_rows,
                };
                let want = tape.commit_prefix(keep).unwrap();
                let got_f = consumers[pos].to_float32().unwrap();
                let want_f = want.astype(DType::Float32).unwrap().to_float32().unwrap();
                assert_eq!(
                    got_f.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    want_f.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "contiguous keep={keep} mismatch at stream={s}"
                );
            }
        }

        unsafe { sys::mlx_clear_cache() };
    }
}
