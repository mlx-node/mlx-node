//! Metal attention & QK-Norm/RoPE kernel wrappers.

use mlx_sys as sys;
use napi::bindgen_prelude::*;

use crate::array::MxArray;

#[inline]
pub(super) fn sdpa_draft_masked(
    queries: &MxArray,
    keys: &MxArray,
    values: &MxArray,
    mask: Option<&MxArray>,
    scale: f32,
    tensor_attention_enabled: bool,
) -> Result<Option<MxArray>> {
    let mut tensor_op = std::ptr::null_mut();
    let status = unsafe {
        sys::mlx_dflash2_attention_with_policy(
            queries.as_raw_ptr(),
            keys.as_raw_ptr(),
            values.as_raw_ptr(),
            mask.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr),
            scale,
            tensor_attention_enabled,
            &mut tensor_op,
        )
    };
    match status {
        1 => Ok(Some(MxArray::from_handle(
            tensor_op,
            "dflash2 tensor-op attention",
        )?)),
        0 => Ok(None),
        _ => Err(Error::from_reason(
            "dflash2 tensor-op attention failed after admission",
        )),
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn qk_norm_rope_offsets(
    queries: &MxArray,
    keys: &MxArray,
    q_weight: &MxArray,
    k_weight: &MxArray,
    offsets: &MxArray,
    eps: f32,
    rope_base: f32,
    rope_scale: f32,
    rope_dims: i32,
) -> Result<Option<(MxArray, MxArray)>> {
    let mut out_q = std::ptr::null_mut();
    let mut out_k = std::ptr::null_mut();
    let ok = unsafe {
        sys::mlx_qk_norm_rope(
            queries.as_raw_ptr(),
            keys.as_raw_ptr(),
            q_weight.as_raw_ptr(),
            k_weight.as_raw_ptr(),
            offsets.as_raw_ptr(),
            eps,
            rope_base,
            rope_scale,
            rope_dims,
            &mut out_q,
            &mut out_k,
        )
    };
    if !ok || out_q.is_null() || out_k.is_null() {
        return Ok(None);
    }
    Ok(Some((
        MxArray::from_handle(out_q, "qk_norm_rope:q")?,
        MxArray::from_handle(out_k, "qk_norm_rope:k")?,
    )))
}
