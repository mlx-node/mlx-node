//! CUDA kernel facades for Qwen3.8-Flash-Next (`qwen4_exp`).
//!
//! All functions dispatch only when `active_backend() == ActiveBackend::CudaGpu`,
//! returning `Ok(None)` cleanly on Metal/CPU or off-contract shapes/dtypes so
//! callers can fall back to the Metal or generic MLX paths.

use mlx_sys as sys;
use napi::bindgen_prelude::*;

use super::common::{Qwen4HcInjectWeight, Qwen4MoeRoutedWeights, Qwen4MoeSharedWeights};
use crate::accelerator::{ActiveBackend, active_backend};
use crate::array::MxArray;

/// Fused CUDA MoE decode/verify primitive (`M in 1..=16`, `H=2560`, `D_FF=640`, `top_k <= 10`).
///
/// Runs GPU route planning, Q8_1 quantization, routed gate/up/SwiGLU, routed down,
/// optional fused shared expert, and weighted combine in a single `eval_gpu` call
/// with zero host sync. Pass `shared = None` for routed-only execution.
#[inline]
#[allow(dead_code)]
pub(crate) fn moe_routed_shared_few_cuda(
    x: &MxArray,
    ids: &MxArray,
    scores: &MxArray,
    routed: &Qwen4MoeRoutedWeights<'_>,
    shared: Option<&Qwen4MoeSharedWeights<'_>>,
) -> Result<Option<MxArray>> {
    if active_backend() != ActiveBackend::CudaGpu {
        return Ok(None);
    }
    let (swg, ssg, sbg, swu, ssu, sbu, swd, ssd, sbd, sgate) = match shared {
        Some(s) => (
            s.gate_w.as_raw_ptr(),
            s.gate_s.as_raw_ptr(),
            s.gate_b.as_raw_ptr(),
            s.up_w.as_raw_ptr(),
            s.up_s.as_raw_ptr(),
            s.up_b.as_raw_ptr(),
            s.down_w.as_raw_ptr(),
            s.down_s.as_raw_ptr(),
            s.down_b.as_raw_ptr(),
            s.shared_gate.as_raw_ptr(),
        ),
        None => (
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        ),
    };
    let handle = unsafe {
        sys::mlx_qwen4_moe_routed_shared_few_cuda(
            x.as_raw_ptr(),
            ids.as_raw_ptr(),
            scores.as_raw_ptr(),
            routed.gate_w.as_raw_ptr(),
            routed.gate_s.as_raw_ptr(),
            routed.gate_b.as_raw_ptr(),
            routed.up_w.as_raw_ptr(),
            routed.up_s.as_raw_ptr(),
            routed.up_b.as_raw_ptr(),
            routed.down_w.as_raw_ptr(),
            routed.down_s.as_raw_ptr(),
            routed.down_b.as_raw_ptr(),
            routed.mode_gu,
            routed.bits_dn,
            swg,
            ssg,
            sbg,
            swu,
            ssu,
            sbu,
            swd,
            ssd,
            sbd,
            sgate,
        )
    };
    if handle.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(
            handle,
            "qwen4_moe_routed_shared_few_cuda",
        )?))
    }
}

/// Fused CUDA MoE prefill primitive (`M >= 1`, designed for `M >= 17`, `H=2560`, `D_FF=640`).
///
/// Runs deterministic route sort + super-block INT8 activation quantization + grouped
/// gate/up/SwiGLU + grouped down + sorted combine (with optional `+ sigmoid(shared_gate) * shared_out`)
/// in a single `eval_gpu` call with zero host sync.
#[inline]
#[allow(dead_code)]
pub(crate) fn moe_prefill_cuda(
    x: &MxArray,
    ids: &MxArray,
    scores: &MxArray,
    routed: &Qwen4MoeRoutedWeights<'_>,
    shared: Option<(&MxArray, &MxArray)>,
) -> Result<Option<MxArray>> {
    if active_backend() != ActiveBackend::CudaGpu {
        return Ok(None);
    }
    let (shared_out, shared_gate) = match shared {
        Some((so, sg)) => (so.as_raw_ptr(), sg.as_raw_ptr()),
        None => (std::ptr::null_mut(), std::ptr::null_mut()),
    };
    let handle = unsafe {
        sys::mlx_qwen4_moe_prefill_cuda(
            x.as_raw_ptr(),
            ids.as_raw_ptr(),
            scores.as_raw_ptr(),
            routed.gate_w.as_raw_ptr(),
            routed.gate_s.as_raw_ptr(),
            routed.gate_b.as_raw_ptr(),
            routed.up_w.as_raw_ptr(),
            routed.up_s.as_raw_ptr(),
            routed.up_b.as_raw_ptr(),
            routed.down_w.as_raw_ptr(),
            routed.down_s.as_raw_ptr(),
            routed.down_b.as_raw_ptr(),
            routed.mode_gu,
            routed.bits_dn,
            shared_out,
            shared_gate,
        )
    };
    if handle.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(
            handle,
            "qwen4_moe_prefill_cuda",
        )?))
    }
}

/// Fused mHC v2 residual stream inject + 4-stream per-stream RMSNorm (`W=10240`, `H=2560`, `HC=4`).
///
/// When `inject` is `Some((branch, gate))`, returns `(x_out, normed)` where
/// `x_out[m, s, h] = x[m, s, h] + branch[m, h] * gate[m, s]` and `normed` is per-stream RMSNorm
/// of `x_out` with `(norm_w + weight_offset)`. When `inject` is `None`, returns `(x_alias, normed)`.
#[inline]
#[allow(dead_code)]
pub(crate) fn inject_norm_cuda(
    x: &MxArray,
    inject: Option<(&MxArray, &MxArray)>,
    norm_w: &MxArray,
    eps: f32,
    weight_offset: f32,
) -> Result<Option<(MxArray, MxArray)>> {
    if active_backend() != ActiveBackend::CudaGpu {
        return Ok(None);
    }
    let (branch_ptr, gate_ptr) = match inject {
        Some((branch, gate)) => (branch.as_raw_ptr(), gate.as_raw_ptr()),
        None => (std::ptr::null_mut(), std::ptr::null_mut()),
    };
    let mut out_x = std::ptr::null_mut();
    let mut out_normed = std::ptr::null_mut();
    let ok = unsafe {
        sys::mlx_qwen4_inject_norm_cuda(
            x.as_raw_ptr(),
            branch_ptr,
            gate_ptr,
            norm_w.as_raw_ptr(),
            eps,
            weight_offset,
            &mut out_x,
            &mut out_normed,
        )
    };
    if !ok || out_x.is_null() || out_normed.is_null() {
        return Ok(None);
    }
    Ok(Some((
        MxArray::from_handle(out_x, "qwen4_inject_norm_cuda:x")?,
        MxArray::from_handle(out_normed, "qwen4_inject_norm_cuda:normed")?,
    )))
}

/// Residual stream injection without RMSNorm on CUDA (`W=10240`, `H=2560`, `HC=4`):
/// `x_out[m, s, h] = bf16(x[m, s, h] + bf16(branch[m, h] * gate[m, s]))`.
#[inline]
#[allow(dead_code)]
pub(crate) fn inject_only_cuda(
    x: &MxArray,
    branch: &MxArray,
    gate: &MxArray,
) -> Result<Option<MxArray>> {
    if active_backend() != ActiveBackend::CudaGpu {
        return Ok(None);
    }
    let mut out_x = std::ptr::null_mut();
    let ok = unsafe {
        sys::mlx_qwen4_inject_norm_cuda(
            x.as_raw_ptr(),
            branch.as_raw_ptr(),
            gate.as_raw_ptr(),
            std::ptr::null_mut(),
            0.0,
            0.0,
            &mut out_x,
            std::ptr::null_mut(),
        )
    };
    if !ok || out_x.is_null() {
        return Ok(None);
    }
    Ok(Some(MxArray::from_handle(out_x, "qwen4_inject_only_cuda")?))
}

/// Fused mHC v2 `down_proj` (`[320, 10240]` Q8_0Bias) + `SiLU(0.25 * y)` and optional
/// `inject_proj` (`[4, 10240]`) + `2 * sigmoid(0.25 * y)` for `M in 1..=16`.
///
/// Returns `(down_act, Some(inject_gate))` when `inject` is not `None`, or
/// `(down_act, None)` when `inject` is `Qwen4HcInjectWeight::None`.
#[inline]
#[allow(dead_code)]
pub(crate) fn hc_down_act_cuda(
    normed: &MxArray,
    w_down: &MxArray,
    s_down: &MxArray,
    b_down: &MxArray,
    inject: Qwen4HcInjectWeight<'_>,
) -> Result<Option<(MxArray, Option<MxArray>)>> {
    if active_backend() != ActiveBackend::CudaGpu {
        return Ok(None);
    }
    let (w_inj, s_inj, b_inj, inj_mode) = match inject {
        Qwen4HcInjectWeight::None => (
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
        ),
        Qwen4HcInjectWeight::Q8_0Bias {
            weight,
            scales,
            biases,
        } => (
            weight.as_raw_ptr(),
            scales.as_raw_ptr(),
            biases.as_raw_ptr(),
            1,
        ),
        Qwen4HcInjectWeight::DenseBf16(w) => (
            w.as_raw_ptr(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            2,
        ),
        Qwen4HcInjectWeight::DenseFp16(w) => (
            w.as_raw_ptr(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            3,
        ),
        Qwen4HcInjectWeight::DenseFp32(w) => (
            w.as_raw_ptr(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            4,
        ),
    };
    let mut out_down: *mut sys::mlx_array = std::ptr::null_mut();
    let mut out_inj: *mut sys::mlx_array = std::ptr::null_mut();
    let out_inj_ptr = if inj_mode != 0 {
        &mut out_inj as *mut *mut sys::mlx_array
    } else {
        std::ptr::null_mut()
    };
    let ok = unsafe {
        sys::mlx_qwen4_hc_down_act_cuda(
            normed.as_raw_ptr(),
            w_down.as_raw_ptr(),
            s_down.as_raw_ptr(),
            b_down.as_raw_ptr(),
            w_inj,
            s_inj,
            b_inj,
            inj_mode,
            &mut out_down,
            out_inj_ptr,
        )
    };
    if !ok || out_down.is_null() || (inj_mode != 0 && out_inj.is_null()) {
        return Ok(None);
    }
    let down = MxArray::from_handle(out_down, "qwen4_hc_down_act_cuda:down")?;
    let inj = if inj_mode != 0 {
        Some(MxArray::from_handle(out_inj, "qwen4_hc_down_act_cuda:inj")?)
    } else {
        None
    };
    Ok(Some((down, inj)))
}

/// Fused mHC v2 `up_proj` (`[10240, 320]` Q8_0Bias) + `sigmoid` + 4-stream weighted mix
/// -> `[.., 2560]` BF16 (`M >= 1`).
#[inline]
#[allow(dead_code)]
pub(crate) fn hc_up_mix_cuda(
    down_act: &MxArray,
    normed: &MxArray,
    w_up: &MxArray,
    s_up: &MxArray,
    b_up: &MxArray,
) -> Result<Option<MxArray>> {
    if active_backend() != ActiveBackend::CudaGpu {
        return Ok(None);
    }
    let handle = unsafe {
        sys::mlx_qwen4_hc_up_mix_cuda(
            down_act.as_raw_ptr(),
            normed.as_raw_ptr(),
            w_up.as_raw_ptr(),
            s_up.as_raw_ptr(),
            b_up.as_raw_ptr(),
        )
    };
    if handle.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(handle, "qwen4_hc_up_mix_cuda")?))
    }
}

/// Fused prefill mHC mix (`sigmoid(up) * normed` + 4-stream `mean(-2)`, `W=10240 -> H=2560`).
#[inline]
#[allow(dead_code)]
pub(crate) fn prefill_hc_mix_cuda(up: &MxArray, normed: &MxArray) -> Result<Option<MxArray>> {
    if active_backend() != ActiveBackend::CudaGpu {
        return Ok(None);
    }
    let handle =
        unsafe { sys::mlx_qwen4_prefill_hc_mix_cuda(up.as_raw_ptr(), normed.as_raw_ptr()) };
    if handle.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(
            handle,
            "qwen4_prefill_hc_mix_cuda",
        )?))
    }
}

/// Fused 4-stream mean (`[.., 10240] -> [.., 2560]` BF16, FP32 sum in stream order 0..3 / 4).
#[inline]
#[allow(dead_code)]
pub(crate) fn stream_mean_cuda(x: &MxArray) -> Result<Option<MxArray>> {
    if active_backend() != ActiveBackend::CudaGpu {
        return Ok(None);
    }
    let handle = unsafe { sys::mlx_qwen4_stream_mean_cuda(x.as_raw_ptr()) };
    if handle.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(
            handle,
            "qwen4_stream_mean_cuda",
        )?))
    }
}

/// 1-pass affine Q8_0 (`bits=8, group_size=32`) -> BF16 dequantization on CUDA.
#[inline]
#[allow(dead_code)]
pub(crate) fn dequant_q8_bf16_cuda(
    w: &MxArray,
    scales: &MxArray,
    biases: &MxArray,
) -> Result<Option<MxArray>> {
    if active_backend() != ActiveBackend::CudaGpu {
        return Ok(None);
    }
    let handle = unsafe {
        sys::mlx_qwen4_dequant_q8_bf16_cuda(
            w.as_raw_ptr(),
            scales.as_raw_ptr(),
            biases.as_raw_ptr(),
        )
    };
    if handle.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(
            handle,
            "qwen4_dequant_q8_bf16_cuda",
        )?))
    }
}

/// Direct fused `Q8_0` (`bits=8, group_size=32, f16 scales/biases`) Tensor-Core
/// prefill GEMM on CUDA (eliminates BF16 dense weight cache allocation).
///
/// `gs_shift`: `0` = per-row INT8, `7` = GS=128 INT8, `8` = GS=256 INT8,
/// `-1` = auto-grouped (`8` if `K%256==0`, else `7` if `K%128==0`, else `0`),
/// `-2` = bit-exact BF16 MMA (zero X quantization).
#[inline]
#[allow(dead_code)]
pub(crate) fn q8_prefill_cuda(
    x: &MxArray,
    w: &MxArray,
    scales: &MxArray,
    biases: &MxArray,
    gs_shift: i32,
    general_bias: bool,
) -> Result<Option<MxArray>> {
    if active_backend() != ActiveBackend::CudaGpu {
        return Ok(None);
    }
    let handle = unsafe {
        sys::mlx_qwen4_q8_prefill_cuda(
            x.as_raw_ptr(),
            w.as_raw_ptr(),
            scales.as_raw_ptr(),
            biases.as_raw_ptr(),
            gs_shift,
            general_bias,
        )
    };
    if handle.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(handle, "qwen4_q8_prefill_cuda")?))
    }
}

/// Fused prefill mHC `up_proj` (`[10240, 320]` symmetric Q8_0) + `sigmoid` +
/// 4-stream weighted mean -> `[.., 2560]` BF16 (bit-exact BF16 MMA, zero
/// `[M, 10240]` DRAM materialization).
#[inline]
#[allow(dead_code)]
pub(crate) fn hc_up_mix_prefill_cuda(
    down_act: &MxArray,
    normed: &MxArray,
    w_up: &MxArray,
    s_up: &MxArray,
    b_up: &MxArray,
) -> Result<Option<MxArray>> {
    if active_backend() != ActiveBackend::CudaGpu {
        return Ok(None);
    }
    let handle = unsafe {
        sys::mlx_qwen4_hc_up_mix_prefill_cuda(
            down_act.as_raw_ptr(),
            normed.as_raw_ptr(),
            w_up.as_raw_ptr(),
            s_up.as_raw_ptr(),
            b_up.as_raw_ptr(),
        )
    };
    if handle.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(
            handle,
            "qwen4_hc_up_mix_prefill_cuda",
        )?))
    }
}

/// Fused Qwen4 Sparse Attention (QSA) indexer Top-512 + zero-gather Sparse SDPA on CUDA.
///
/// - `q`: `[1, Hq, Q_len, 256]` BF16 (`GQA = Hq / Hkv in 1..=16`)
/// - `k`, `v`: `[1, Hkv, K_cap, 256]` BF16 (strided cache views accepted directly)
/// - `iq`: `[1, 4, Q_len, 128]` or `[4, Q_len, 128]` BF16
/// - `pooled_k`: `[num_blocks_cap, 128]` or `[1, num_blocks_cap, 128]` BF16
/// - `tree_mask`: `Some(uint32[Q_len])` for `mask_mode = 1` (speculative tree, `Q_len <= 16`),
///   `None` for `mask_mode = 0` (causal)
/// - `kv_len`: active KV length (`>= Q_len`, `<= K_cap`); `base_pos = kv_len - Q_len`
/// - `num_blocks`: number of complete 4-token blocks in `pooled_k`
/// - `scale`: SDPA score scale (`1 / sqrt(256)`)
/// - `idx_scale`: positive indexer score scale (`1 / sqrt(128)`)
#[inline]
#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn qsa_sparse_sdpa_cuda(
    q: &MxArray,
    k: &MxArray,
    v: &MxArray,
    iq: &MxArray,
    pooled_k: &MxArray,
    tree_mask: Option<&MxArray>,
    kv_len: i32,
    num_blocks: i32,
    scale: f32,
    idx_scale: f32,
    mask_mode: i32,
) -> Result<Option<MxArray>> {
    if active_backend() != ActiveBackend::CudaGpu {
        return Ok(None);
    }
    let tree_mask_ptr = tree_mask.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    let handle = unsafe {
        sys::mlx_qwen4_qsa_sparse_sdpa_cuda(
            q.as_raw_ptr(),
            k.as_raw_ptr(),
            v.as_raw_ptr(),
            iq.as_raw_ptr(),
            pooled_k.as_raw_ptr(),
            tree_mask_ptr,
            kv_len,
            num_blocks,
            scale,
            idx_scale,
            mask_mode,
        )
    };
    if handle.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(
            handle,
            "qwen4_qsa_sparse_sdpa_cuda",
        )?))
    }
}

/// Fused Qwen4 PLE Layer-1 gate + 3x per-stream RMSNorm + dilated causal depthwise conv1d
/// (`k=4, d=3`, `keep=9`) + SiLU + add (+ optional residual add).
///
/// Returns `(out_x [same shape as x], next_history [9, 10240])` BF16.
#[inline]
#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn ple_gate_conv_cuda(
    x: &MxArray,
    key: &MxArray,
    value: &MxArray,
    norm_key_w: &MxArray,
    norm_query_w: &MxArray,
    norm_conv_w: &MxArray,
    conv_w: &MxArray,
    history: Option<&MxArray>,
    eps: f32,
    weight_offset: f32,
    fuse_residual: bool,
) -> Result<Option<(MxArray, MxArray)>> {
    if active_backend() != ActiveBackend::CudaGpu {
        return Ok(None);
    }
    let hist_ptr = history.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    let mut out_x = std::ptr::null_mut();
    let mut out_hist = std::ptr::null_mut();
    let ok = unsafe {
        sys::mlx_qwen4_ple_gate_conv_cuda(
            x.as_raw_ptr(),
            key.as_raw_ptr(),
            value.as_raw_ptr(),
            norm_key_w.as_raw_ptr(),
            norm_query_w.as_raw_ptr(),
            norm_conv_w.as_raw_ptr(),
            conv_w.as_raw_ptr(),
            hist_ptr,
            std::ptr::null_mut(),
            eps,
            weight_offset,
            fuse_residual,
            &mut out_x,
            &mut out_hist,
            std::ptr::null_mut(),
        )
    };
    if !ok || out_x.is_null() || out_hist.is_null() {
        return Ok(None);
    }
    Ok(Some((
        MxArray::from_handle(out_x, "qwen4_ple_gate_conv_cuda:x")?,
        MxArray::from_handle(out_hist, "qwen4_ple_gate_conv_cuda:hist")?,
    )))
}

/// Tree-aware fused Qwen4 PLE Layer-1 gate + 3x per-stream RMSNorm + dilated
/// depthwise conv1d (`k=4, d=3`, `keep=9`) + SiLU + add (+ optional residual add).
///
/// - `conv_taps`: optional `[T, 4]` `Int32` dilated tap indices (`>= 0` reads
///   `conv_in[idx]`, `-9 <= idx < 0` reads `history[9 + idx]`, `< -9` reads `0`).
///   When `None`, uses linear causal indices `[t-9, t-6, t-3, t]`.
/// - Returns `(out_x [same shape as x], conv_in [T, 10240])` BF16 so tree commit
///   can rebuild exact `[9, 10240]` `ple_conv` history along any accepted branch.
#[inline]
#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn ple_gate_conv_tree_cuda(
    x: &MxArray,
    key: &MxArray,
    value: &MxArray,
    norm_key_w: &MxArray,
    norm_query_w: &MxArray,
    norm_conv_w: &MxArray,
    conv_w: &MxArray,
    history: Option<&MxArray>,
    conv_taps: Option<&MxArray>,
    eps: f32,
    weight_offset: f32,
    fuse_residual: bool,
) -> Result<Option<(MxArray, MxArray)>> {
    if active_backend() != ActiveBackend::CudaGpu {
        return Ok(None);
    }
    let hist_ptr = history.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    let taps_ptr = conv_taps.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    let mut out_x = std::ptr::null_mut();
    let mut out_conv_in = std::ptr::null_mut();
    let ok = unsafe {
        sys::mlx_qwen4_ple_gate_conv_cuda(
            x.as_raw_ptr(),
            key.as_raw_ptr(),
            value.as_raw_ptr(),
            norm_key_w.as_raw_ptr(),
            norm_query_w.as_raw_ptr(),
            norm_conv_w.as_raw_ptr(),
            conv_w.as_raw_ptr(),
            hist_ptr,
            taps_ptr,
            eps,
            weight_offset,
            fuse_residual,
            &mut out_x,
            std::ptr::null_mut(),
            &mut out_conv_in,
        )
    };
    if !ok || out_x.is_null() || out_conv_in.is_null() {
        return Ok(None);
    }
    Ok(Some((
        MxArray::from_handle(out_x, "qwen4_ple_gate_conv_tree_cuda:x")?,
        MxArray::from_handle(out_conv_in, "qwen4_ple_gate_conv_tree_cuda:conv_in")?,
    )))
}

/// Fused CUDA router FP32 softmax + deterministic top-k + normalize
/// (`E <= 512`, `top_k in 1..=16`, `M >= 1`).
///
/// Returns `(ids: Uint32[..., top_k], scores: Float32[..., top_k])`.
#[inline]
#[allow(dead_code)]
pub(crate) fn router_topk_cuda(
    logits: &MxArray,
    top_k: usize,
) -> Result<Option<(MxArray, MxArray)>> {
    if active_backend() != ActiveBackend::CudaGpu || top_k == 0 || top_k > 16 {
        return Ok(None);
    }
    let mut out_ids = std::ptr::null_mut();
    let mut out_scores = std::ptr::null_mut();
    let ok = unsafe {
        sys::mlx_qwen4_router_topk_cuda(
            logits.as_raw_ptr(),
            top_k as i32,
            &mut out_ids,
            &mut out_scores,
        )
    };
    if !ok || out_ids.is_null() || out_scores.is_null() {
        return Ok(None);
    }
    Ok(Some((
        MxArray::from_handle(out_ids, "qwen4_router_topk_cuda:ids")?,
        MxArray::from_handle(out_scores, "qwen4_router_topk_cuda:scores")?,
    )))
}
