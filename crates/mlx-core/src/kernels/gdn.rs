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

/// Fused GDN recurrence + gated RMSNorm + z-gate in one GPU dispatch.
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
    let mut out = std::ptr::null_mut();
    let mut out_state = std::ptr::null_mut();
    let dst_ptr = state_dst
        .map(MxArray::as_raw_ptr)
        .unwrap_or(std::ptr::null_mut());
    let ok = match active_backend() {
        ActiveBackend::CudaGpu => unsafe {
            sys::mlx_gdn_fused_step_cuda(
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
                dst_ptr,
                &mut out,
                &mut out_state,
            )
        },
        ActiveBackend::MetalGpu => unsafe {
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
        ActiveBackend::Cpu => false,
    };
    if !ok || out.is_null() || out_state.is_null() {
        return Ok(None);
    }
    Ok(Some((
        MxArray::from_handle(out, "gated_delta_fused_step:out")?,
        MxArray::from_handle(out_state, "gated_delta_fused_step:state")?,
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
                || beta.dtype()? != DType::BFloat16
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
