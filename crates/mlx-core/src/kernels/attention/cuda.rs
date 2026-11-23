//! CUDA Tensor Core attention & QK-Norm/RoPE kernel wrappers.

use mlx_sys as sys;
use napi::bindgen_prelude::*;

use crate::array::{DType, MxArray};

#[inline]
pub(super) fn sdpa_causal_few_q1_to_16(
    queries: &MxArray,
    keys: &MxArray,
    values: &MxArray,
    head_dim: i64,
    gqa: i64,
    seq_len: i64,
    scale: f32,
) -> Result<Option<MxArray>> {
    let m = gqa * seq_len;
    let ok_single = (1..=96).contains(&m);
    let ok_two_tile = seq_len > 8 && gqa * 8 <= 96;
    if !(1..=16).contains(&seq_len)
        || head_dim != 256
        || gqa <= 1
        || (!ok_single && !ok_two_tile)
        || queries.dtype()? != DType::BFloat16
        || keys.dtype()? != DType::BFloat16
        || values.dtype()? != DType::BFloat16
    {
        return Ok(None);
    }
    let handle = unsafe {
        sys::mlx_sdpa_tc_few_cuda(
            queries.as_raw_ptr(),
            keys.as_raw_ptr(),
            values.as_raw_ptr(),
            scale,
        )
    };
    if handle.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(handle, "sdpa_tc_few_cuda")?))
    }
}

#[inline]
pub(super) fn sdpa_prefill_d256(
    queries: &MxArray,
    keys: &MxArray,
    values: &MxArray,
    head_dim: i64,
    seq_len: i64,
    scale: f32,
) -> Result<Option<MxArray>> {
    if seq_len < 9
        || head_dim != 256
        || queries.dtype()? != DType::BFloat16
        || keys.dtype()? != DType::BFloat16
        || values.dtype()? != DType::BFloat16
    {
        return Ok(None);
    }
    let handle = unsafe {
        sys::mlx_sdpa_prefill_d256_cuda(
            queries.as_raw_ptr(),
            keys.as_raw_ptr(),
            values.as_raw_ptr(),
            scale,
        )
    };
    if handle.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(
            handle,
            "sdpa_prefill_d256_cuda",
        )?))
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn sdpa_draft_masked(
    queries: &MxArray,
    keys: &MxArray,
    values: &MxArray,
    mask: Option<&MxArray>,
    head_dim: i64,
    num_heads: i64,
    num_kv_heads: i64,
    seq_len: i64,
    scale: f32,
) -> Result<Option<MxArray>> {
    let gqa = num_heads / num_kv_heads;
    let m = gqa * seq_len;
    let ok_single = (1..=96).contains(&m);
    let ok_two_tile = seq_len > 8 && gqa * 8 <= 96;
    if !(1..=16).contains(&seq_len)
        || !(head_dim == 128 || head_dim == 256)
        || (!ok_single && !ok_two_tile)
        || queries.dtype()? != DType::BFloat16
        || keys.dtype()? != DType::BFloat16
        || values.dtype()? != DType::BFloat16
    {
        return Ok(None);
    }
    let mask_ptr = match mask {
        Some(m) if m.ndim()? == 4 && m.shape_at(0)? == 1 && m.shape_at(1)? == 1 => m.as_raw_ptr(),
        None => std::ptr::null_mut(),
        _ => return Ok(None),
    };
    let handle = unsafe {
        sys::mlx_sdpa_tc_few_masked_cuda(
            queries.as_raw_ptr(),
            keys.as_raw_ptr(),
            values.as_raw_ptr(),
            mask_ptr,
            scale,
            false,
        )
    };
    if handle.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(
            handle,
            "dflash2 sdpa_tc_few_cuda",
        )?))
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
        sys::mlx_qk_norm_rope_cuda(
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

#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn sdpa_tree_few(
    queries: &MxArray,
    keys: &MxArray,
    values: &MxArray,
    tree_mask: &MxArray,
    cache_len: i32,
    head_dim: i64,
    gqa: i64,
    seq_len: i64,
    scale: f32,
) -> Result<Option<MxArray>> {
    let m = gqa * seq_len;
    let ok_single = (1..=96).contains(&m);
    let ok_two_tile = seq_len > 8 && gqa * 8 <= 96;
    if !(1..=16).contains(&seq_len)
        || !(head_dim == 128 || head_dim == 256)
        || cache_len < 0
        || (!ok_single && !ok_two_tile)
        || queries.dtype()? != DType::BFloat16
        || keys.dtype()? != DType::BFloat16
        || values.dtype()? != DType::BFloat16
        || tree_mask.dtype()? != DType::Uint32
    {
        return Ok(None);
    }
    let handle = unsafe {
        sys::mlx_sdpa_tc_tree_few_cuda(
            queries.as_raw_ptr(),
            keys.as_raw_ptr(),
            values.as_raw_ptr(),
            tree_mask.as_raw_ptr(),
            cache_len,
            scale,
        )
    };
    if handle.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(handle, "sdpa_tc_tree_few_cuda")?))
    }
}
