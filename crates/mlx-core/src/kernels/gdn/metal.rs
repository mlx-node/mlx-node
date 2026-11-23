//! Metal Gated Delta Net (GDN) kernel wrappers.

use mlx_sys as sys;
use napi::bindgen_prelude::*;

use crate::array::MxArray;

#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn prepare(
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
    let ok = unsafe {
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

#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn fused_step(
    q: &MxArray,
    k: &MxArray,
    v: &MxArray,
    g: &MxArray,
    beta: &MxArray,
    state: &MxArray,
    z: &MxArray,
    norm_weight: &MxArray,
    eps: f32,
    state_dst: Option<&MxArray>,
) -> Result<Option<(MxArray, MxArray)>> {
    let mut out = std::ptr::null_mut();
    let mut out_state = std::ptr::null_mut();
    let dst_ptr = state_dst
        .map(MxArray::as_raw_ptr)
        .unwrap_or(std::ptr::null_mut());
    let ok = unsafe {
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
    };
    if !ok || out.is_null() || out_state.is_null() {
        return Ok(None);
    }
    Ok(Some((
        MxArray::from_handle(out, "gated_delta_fused_step:out")?,
        MxArray::from_handle(out_state, "gated_delta_fused_step:state")?,
    )))
}

#[inline]
pub(super) fn replay(
    k: &MxArray,
    v: &MxArray,
    g: &MxArray,
    beta: &MxArray,
    start_state: &MxArray,
    accepted_steps: i32,
) -> Result<Option<MxArray>> {
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
