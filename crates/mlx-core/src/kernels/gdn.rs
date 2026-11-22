//! Backend-neutral Gated Delta Net (GDN) kernel dispatch facades.

use mlx_sys as sys;
use napi::bindgen_prelude::*;

use crate::accelerator::{ActiveBackend, active_backend};
use crate::array::{DType, MxArray};

/// Fused GDN prepare (`conv1d + SiLU + q/k RMSNorm + decay/beta + next_history`).
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare(
    qkv: &MxArray,
    a: &MxArray,
    b: &MxArray,
    conv_w: &MxArray,
    history: &MxArray,
    scale: &MxArray,
    dt: &MxArray,
    mean_eps: bool,
    beta_input_dtype: bool,
) -> Result<Option<[MxArray; 6]>> {
    let mut outputs = [std::ptr::null_mut(); 6];
    let ok = match active_backend() {
        ActiveBackend::MetalGpu => unsafe {
            sys::mlx_qwen4_gdn_prepare(
                qkv.handle.0,
                a.handle.0,
                b.handle.0,
                conv_w.handle.0,
                history.handle.0,
                scale.handle.0,
                dt.handle.0,
                mean_eps,
                beta_input_dtype,
                outputs.as_mut_ptr(),
            )
        },
        ActiveBackend::CudaGpu => unsafe {
            sys::mlx_gdn_prepare_cuda(
                qkv.handle.0,
                a.handle.0,
                b.handle.0,
                conv_w.handle.0,
                history.handle.0,
                scale.handle.0,
                dt.handle.0,
                mean_eps,
                beta_input_dtype,
                outputs.as_mut_ptr(),
            )
        },
        ActiveBackend::Cpu => false,
    };
    if !ok {
        return Ok(None);
    }
    Ok(Some([
        MxArray::from_handle(outputs[0], "gdn_prepare:q")?,
        MxArray::from_handle(outputs[1], "gdn_prepare:k")?,
        MxArray::from_handle(outputs[2], "gdn_prepare:v")?,
        MxArray::from_handle(outputs[3], "gdn_prepare:decay")?,
        MxArray::from_handle(outputs[4], "gdn_prepare:beta")?,
        MxArray::from_handle(outputs[5], "gdn_prepare:history")?,
    ]))
}

/// Fused GDN recurrence + gated RMSNorm + z-gate in one GPU dispatch (SiLU gate).
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn fused_step(
    q: &MxArray,
    k: &MxArray,
    v: &MxArray,
    g: &MxArray,
    beta: &MxArray,
    state: &MxArray,
    z: &MxArray,
    norm_weight: &MxArray,
    eps: f32,
    tiled_heads: bool,
    state_dst: Option<&MxArray>,
) -> Result<Option<(MxArray, MxArray)>> {
    fused_step_with_gate(
        q,
        k,
        v,
        g,
        beta,
        state,
        z,
        norm_weight,
        eps,
        tiled_heads,
        false,
        state_dst,
    )
}

/// Fused GDN recurrence + gated RMSNorm + z-gate with explicit gate selector:
/// `sigmoid_gate = false` uses SiLU (`z * sigmoid(z)`, Qwen3.5),
/// `sigmoid_gate = true` uses Sigmoid (`sigmoid(z)`, Qwen3.8-Flash-Next / `qwen4_exp`, CUDA).
/// On CUDA, both `beta` and `norm_weight` may be BF16 or FP32.
#[inline]
#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn fused_step_with_gate(
    q: &MxArray,
    k: &MxArray,
    v: &MxArray,
    g: &MxArray,
    beta: &MxArray,
    state: &MxArray,
    z: &MxArray,
    norm_weight: &MxArray,
    eps: f32,
    tiled_heads: bool,
    sigmoid_gate: bool,
    state_dst: Option<&MxArray>,
) -> Result<Option<(MxArray, MxArray)>> {
    let mut out = std::ptr::null_mut();
    let mut out_state = std::ptr::null_mut();
    let dst_ptr = state_dst
        .map(MxArray::as_raw_ptr)
        .unwrap_or(std::ptr::null_mut());
    let ok = match active_backend() {
        ActiveBackend::CudaGpu => unsafe {
            sys::mlx_gdn_fused_step_cuda_ex(
                q.as_raw_ptr(),
                k.as_raw_ptr(),
                v.as_raw_ptr(),
                g.as_raw_ptr(),
                beta.as_raw_ptr(),
                state.as_raw_ptr(),
                z.as_raw_ptr(),
                norm_weight.as_raw_ptr(),
                eps,
                i32::from(tiled_heads),
                sigmoid_gate,
                dst_ptr,
                &mut out,
                &mut out_state,
            )
        },
        ActiveBackend::MetalGpu if !sigmoid_gate => unsafe {
            sys::mlx_gdn_fused_step(
                q.as_raw_ptr(),
                k.as_raw_ptr(),
                v.as_raw_ptr(),
                g.as_raw_ptr(),
                beta.as_raw_ptr(),
                state.as_raw_ptr(),
                z.as_raw_ptr(),
                norm_weight.as_raw_ptr(),
                eps,
                dst_ptr,
                &mut out,
                &mut out_state,
            )
        },
        ActiveBackend::MetalGpu | ActiveBackend::Cpu => false,
    };
    if !ok || out.is_null() || out_state.is_null() {
        return Ok(None);
    }
    Ok(Some((
        MxArray::from_handle(out, "gated_delta_fused_step:out")?,
        MxArray::from_handle(out_state, "gated_delta_fused_step:state")?,
    )))
}

/// Deferred-fold CUDA fused GDN step (SiLU gate): folds `prev_keep` accepted rows from the
/// prior cycle's `(prev_k, prev_v, prev_g, prev_beta)` into `state_base` before
/// stepping the current verify window and applying gated RMSNorm + z-gate.
///
/// When `pending` is `None` (cycle 0), passes null pointers and `prev_keep = 0`,
/// reading `state_base` read-only and returning `(out_y, state_base_alias)`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn fused_step_deferred(
    q: &MxArray,
    k: &MxArray,
    v: &MxArray,
    g: &MxArray,
    beta: &MxArray,
    state_base: &MxArray,
    pending: Option<(&MxArray, &MxArray, &MxArray, &MxArray, i32)>,
    z: &MxArray,
    norm_weight: &MxArray,
    eps: f32,
    tiled_heads: bool,
) -> Result<Option<(MxArray, MxArray)>> {
    fused_step_deferred_with_gate(
        q,
        k,
        v,
        g,
        beta,
        state_base,
        pending,
        z,
        norm_weight,
        eps,
        tiled_heads,
        false,
    )
}

/// Deferred-fold CUDA fused GDN step with explicit `sigmoid_gate` selector
/// (`false` = SiLU, `true` = Qwen4 Sigmoid; BF16/FP32 `beta` and `norm_weight`).
#[inline]
#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn fused_step_deferred_with_gate(
    q: &MxArray,
    k: &MxArray,
    v: &MxArray,
    g: &MxArray,
    beta: &MxArray,
    state_base: &MxArray,
    pending: Option<(&MxArray, &MxArray, &MxArray, &MxArray, i32)>,
    z: &MxArray,
    norm_weight: &MxArray,
    eps: f32,
    tiled_heads: bool,
    sigmoid_gate: bool,
) -> Result<Option<(MxArray, MxArray)>> {
    if active_backend() != ActiveBackend::CudaGpu {
        return Ok(None);
    }
    let (prev_k, prev_v, prev_g, prev_beta, prev_keep) = match pending {
        Some((pk, pv, pg, pb, keep)) if keep > 0 => (
            pk.as_raw_ptr(),
            pv.as_raw_ptr(),
            pg.as_raw_ptr(),
            pb.as_raw_ptr(),
            keep,
        ),
        _ => (
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
        ),
    };
    let mut out_y = std::ptr::null_mut();
    let mut out_committed_state = std::ptr::null_mut();
    let ok = unsafe {
        sys::mlx_gdn_fused_step_deferred_cuda_ex(
            q.as_raw_ptr(),
            k.as_raw_ptr(),
            v.as_raw_ptr(),
            g.as_raw_ptr(),
            beta.as_raw_ptr(),
            state_base.as_raw_ptr(),
            prev_k,
            prev_v,
            prev_g,
            prev_beta,
            prev_keep,
            z.as_raw_ptr(),
            norm_weight.as_raw_ptr(),
            eps,
            i32::from(tiled_heads),
            sigmoid_gate,
            &mut out_y,
            &mut out_committed_state,
        )
    };
    if !ok || out_y.is_null() || out_committed_state.is_null() {
        return Ok(None);
    }
    Ok(Some((
        MxArray::from_handle(out_y, "gdn_fused_step_deferred:y")?,
        MxArray::from_handle(out_committed_state, "gdn_fused_step_deferred:state")?,
    )))
}

/// Tree-verify CUDA fused GDN step with deferred fold from the prior cycle (SiLU gate).
/// `parent` is `int32[T]` (`parent[0] = -1`, `parent[t] < t`).
/// `pending` optionally carries `(prev_k, prev_v, prev_g, prev_beta, prev_keep, prev_indices)`.
#[inline]
#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn tree_step_deferred(
    q: &MxArray,
    k: &MxArray,
    v: &MxArray,
    g: &MxArray,
    beta: &MxArray,
    state_base: &MxArray,
    pending: Option<(
        &MxArray,
        &MxArray,
        &MxArray,
        &MxArray,
        i32,
        Option<&MxArray>,
    )>,
    parent: &MxArray,
    z: &MxArray,
    norm_weight: &MxArray,
    eps: f32,
    tiled_heads: bool,
) -> Result<Option<(MxArray, MxArray)>> {
    tree_step_deferred_with_gate(
        q,
        k,
        v,
        g,
        beta,
        state_base,
        pending,
        parent,
        z,
        norm_weight,
        eps,
        tiled_heads,
        false,
    )
}

/// Tree-verify CUDA fused GDN step with deferred fold and explicit `sigmoid_gate`
/// selector (`false` = SiLU, `true` = Qwen4 Sigmoid; BF16/FP32 `beta` and `norm_weight`).
#[inline]
#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn tree_step_deferred_with_gate(
    q: &MxArray,
    k: &MxArray,
    v: &MxArray,
    g: &MxArray,
    beta: &MxArray,
    state_base: &MxArray,
    pending: Option<(
        &MxArray,
        &MxArray,
        &MxArray,
        &MxArray,
        i32,
        Option<&MxArray>,
    )>,
    parent: &MxArray,
    z: &MxArray,
    norm_weight: &MxArray,
    eps: f32,
    tiled_heads: bool,
    sigmoid_gate: bool,
) -> Result<Option<(MxArray, MxArray)>> {
    if active_backend() != ActiveBackend::CudaGpu {
        return Ok(None);
    }
    let (prev_k, prev_v, prev_g, prev_beta, prev_keep, prev_indices) = match pending {
        Some((pk, pv, pg, pb, keep, pidx)) if keep > 0 => (
            pk.as_raw_ptr(),
            pv.as_raw_ptr(),
            pg.as_raw_ptr(),
            pb.as_raw_ptr(),
            keep,
            pidx.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr),
        ),
        _ => (
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
            std::ptr::null_mut(),
        ),
    };
    let mut out_y = std::ptr::null_mut();
    let mut out_committed_state = std::ptr::null_mut();
    let ok = unsafe {
        sys::mlx_gdn_tree_step_cuda_ex(
            q.as_raw_ptr(),
            k.as_raw_ptr(),
            v.as_raw_ptr(),
            g.as_raw_ptr(),
            beta.as_raw_ptr(),
            state_base.as_raw_ptr(),
            prev_k,
            prev_v,
            prev_g,
            prev_beta,
            prev_keep,
            prev_indices,
            parent.as_raw_ptr(),
            z.as_raw_ptr(),
            norm_weight.as_raw_ptr(),
            eps,
            i32::from(tiled_heads),
            sigmoid_gate,
            &mut out_y,
            &mut out_committed_state,
        )
    };
    if !ok || out_y.is_null() || out_committed_state.is_null() {
        return Ok(None);
    }
    Ok(Some((
        MxArray::from_handle(out_y, "gdn_tree_step:y")?,
        MxArray::from_handle(out_committed_state, "gdn_tree_step:state")?,
    )))
}

/// Tree-conv1d variant of fused CUDA GDN preparation (`conv_taps` is `int32[T, 4]`).
#[inline]
#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn prepare_tree(
    qkv: &MxArray,
    a: &MxArray,
    b: &MxArray,
    conv_w: &MxArray,
    history: &MxArray,
    scale: &MxArray,
    dt: &MxArray,
    conv_taps: &MxArray,
    mean_eps: bool,
    beta_input_dtype: bool,
) -> Result<Option<[MxArray; 6]>> {
    if active_backend() != ActiveBackend::CudaGpu {
        return Ok(None);
    }
    let mut outputs = [std::ptr::null_mut(); 6];
    let ok = unsafe {
        sys::mlx_gdn_prepare_tree_cuda(
            qkv.as_raw_ptr(),
            a.as_raw_ptr(),
            b.as_raw_ptr(),
            conv_w.as_raw_ptr(),
            history.as_raw_ptr(),
            scale.as_raw_ptr(),
            dt.as_raw_ptr(),
            conv_taps.as_raw_ptr(),
            mean_eps,
            beta_input_dtype,
            outputs.as_mut_ptr(),
        )
    };
    if !ok {
        return Ok(None);
    }
    Ok(Some([
        MxArray::from_handle(outputs[0], "gdn_prepare_tree:q")?,
        MxArray::from_handle(outputs[1], "gdn_prepare_tree:k")?,
        MxArray::from_handle(outputs[2], "gdn_prepare_tree:v")?,
        MxArray::from_handle(outputs[3], "gdn_prepare_tree:decay")?,
        MxArray::from_handle(outputs[4], "gdn_prepare_tree:beta")?,
        MxArray::from_handle(outputs[5], "gdn_prepare_tree:history")?,
    ]))
}

/// Register-resident CUDA GDN prefill recurrence (B=1, Dk=Dv=128, BF16 q/k/v).
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn prefill_chain(
    q: &MxArray,
    k: &MxArray,
    v: &MxArray,
    g: &MxArray,
    beta: &MxArray,
    state: &MxArray,
    tiled_heads: bool,
) -> Result<Option<(MxArray, MxArray)>> {
    if active_backend() != ActiveBackend::CudaGpu {
        return Ok(None);
    }
    let mut out_y = std::ptr::null_mut();
    let mut out_state = std::ptr::null_mut();
    let ok = unsafe {
        sys::mlx_gdn_prefill_chain_cuda(
            q.as_raw_ptr(),
            k.as_raw_ptr(),
            v.as_raw_ptr(),
            g.as_raw_ptr(),
            beta.as_raw_ptr(),
            state.as_raw_ptr(),
            i32::from(tiled_heads),
            &mut out_y,
            &mut out_state,
        )
    };
    if !ok || out_y.is_null() || out_state.is_null() {
        return Ok(None);
    }
    Ok(Some((
        MxArray::from_handle(out_y, "gdn_prefill_chain:y")?,
        MxArray::from_handle(out_state, "gdn_prefill_chain:state")?,
    )))
}

/// Register-resident single-block-per-head CUDA GDN prefill recurrence with
/// fused strided-`z` `RMSNormGated` epilogue (`[1, T, hv * 128]` BF16 output, SiLU gate).
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn prefill_chain_gated(
    q: &MxArray,
    k: &MxArray,
    v: &MxArray,
    g: &MxArray,
    beta: &MxArray,
    state: &MxArray,
    z: &MxArray,
    norm_weight: &MxArray,
    eps: f32,
    tiled_heads: bool,
) -> Result<Option<(MxArray, MxArray)>> {
    prefill_chain_gated_with_gate(
        q,
        k,
        v,
        g,
        beta,
        state,
        z,
        norm_weight,
        eps,
        tiled_heads,
        false,
    )
}

/// Register-resident single-block-per-head CUDA GDN prefill recurrence with
/// fused strided-`z` `RMSNormGated` epilogue and explicit `sigmoid_gate` selector
/// (`false` = SiLU, `true` = Qwen4 Sigmoid; BF16/FP32 `beta` and `norm_weight`).
#[inline]
#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn prefill_chain_gated_with_gate(
    q: &MxArray,
    k: &MxArray,
    v: &MxArray,
    g: &MxArray,
    beta: &MxArray,
    state: &MxArray,
    z: &MxArray,
    norm_weight: &MxArray,
    eps: f32,
    tiled_heads: bool,
    sigmoid_gate: bool,
) -> Result<Option<(MxArray, MxArray)>> {
    if active_backend() != ActiveBackend::CudaGpu {
        return Ok(None);
    }
    let mut out_y = std::ptr::null_mut();
    let mut out_state = std::ptr::null_mut();
    let ok = unsafe {
        sys::mlx_gdn_prefill_chain_gated_cuda_ex(
            q.as_raw_ptr(),
            k.as_raw_ptr(),
            v.as_raw_ptr(),
            g.as_raw_ptr(),
            beta.as_raw_ptr(),
            state.as_raw_ptr(),
            z.as_raw_ptr(),
            norm_weight.as_raw_ptr(),
            eps,
            i32::from(tiled_heads),
            sigmoid_gate,
            &mut out_y,
            &mut out_state,
        )
    };
    if !ok || out_y.is_null() || out_state.is_null() {
        return Ok(None);
    }
    Ok(Some((
        MxArray::from_handle(out_y, "gdn_prefill_chain_gated:y")?,
        MxArray::from_handle(out_state, "gdn_prefill_chain_gated:state")?,
    )))
}

/// Fused accepted-prefix recurrent state replay for eager-MTP GDN rollback.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn replay(
    k: &MxArray,
    v: &MxArray,
    g: &MxArray,
    beta: &MxArray,
    start_state: &MxArray,
    accepted_steps: i32,
    tiled_heads: bool,
) -> Result<Option<MxArray>> {
    match active_backend() {
        ActiveBackend::MetalGpu => {
            let window_stride = k.shape_at(1)? as i32;
            let mut out_state = std::ptr::null_mut();
            let ok = unsafe {
                sys::mlx_gated_delta_replay(
                    k.as_raw_ptr(),
                    v.as_raw_ptr(),
                    g.as_raw_ptr(),
                    beta.as_raw_ptr(),
                    start_state.as_raw_ptr(),
                    accepted_steps,
                    window_stride,
                    &mut out_state,
                )
            };
            if ok && !out_state.is_null() {
                Ok(Some(MxArray::from_handle(
                    out_state,
                    "gated_delta_replay:state",
                )?))
            } else {
                Ok(None)
            }
        }
        ActiveBackend::CudaGpu => {
            if k.shape_at(3)? != 128
                || v.shape_at(3)? != 128
                || k.dtype()? != DType::BFloat16
                || v.dtype()? != DType::BFloat16
                || !matches!(beta.dtype()?, DType::BFloat16 | DType::Float32)
                || g.dtype()? != DType::Float32
                || start_state.dtype()? != DType::Float32
            {
                return Ok(None);
            }
            let mut out_state = std::ptr::null_mut();
            let ok = unsafe {
                sys::mlx_gdn_replay_cuda(
                    k.as_raw_ptr(),
                    v.as_raw_ptr(),
                    g.as_raw_ptr(),
                    beta.as_raw_ptr(),
                    start_state.as_raw_ptr(),
                    accepted_steps,
                    i32::from(tiled_heads),
                    &mut out_state,
                )
            };
            if ok && !out_state.is_null() {
                Ok(Some(MxArray::from_handle(
                    out_state,
                    "gdn_replay_cuda:state",
                )?))
            } else {
                Ok(None)
            }
        }
        ActiveBackend::Cpu => Ok(None),
    }
}

/// CUDA 64x64 strictly-lower-triangular `(I + A)^-1` fast path (non-differentiable).
#[inline]
pub(crate) fn invert64(a: &MxArray) -> Result<Option<MxArray>> {
    if active_backend() != ActiveBackend::CudaGpu {
        return Ok(None);
    }
    let handle = unsafe { sys::mlx_gdn_invert64(a.as_raw_ptr()) };
    if handle.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(handle, "gdn_invert64")?))
    }
}
