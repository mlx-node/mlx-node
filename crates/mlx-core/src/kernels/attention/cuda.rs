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
pub(super) fn sdpa_prefill_d256_gated(
    queries: &MxArray,
    keys: &MxArray,
    values: &MxArray,
    gate: &MxArray,
    head_dim: i64,
    seq_len: i64,
    scale: f32,
) -> Result<Option<MxArray>> {
    if seq_len < 9
        || head_dim != 256
        || queries.dtype()? != DType::BFloat16
        || keys.dtype()? != DType::BFloat16
        || values.dtype()? != DType::BFloat16
        || gate.dtype()? != DType::BFloat16
        || gate.ndim()? != 3
    {
        return Ok(None);
    }
    let handle = unsafe {
        sys::mlx_cuda_sdpa_prefill_d256_gated(
            queries.as_raw_ptr(),
            keys.as_raw_ptr(),
            values.as_raw_ptr(),
            gate.as_raw_ptr(),
            std::ptr::null_mut(),
            scale,
            true,
        )
    };
    if handle.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(
            handle,
            "sdpa_prefill_d256_gated_cuda",
        )?))
    }
}

#[inline]
pub(super) fn sdpa_unmasked_few(
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
        || !(head_dim == 128 || head_dim == 256)
        || gqa < 1
        || (!ok_single && !ok_two_tile)
        || queries.dtype()? != DType::BFloat16
        || keys.dtype()? != DType::BFloat16
        || values.dtype()? != DType::BFloat16
    {
        return Ok(None);
    }
    let handle = unsafe {
        sys::mlx_sdpa_tc_few_masked_cuda(
            queries.as_raw_ptr(),
            keys.as_raw_ptr(),
            values.as_raw_ptr(),
            std::ptr::null_mut(),
            scale,
            false,
        )
    };
    if handle.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(
            handle,
            "sdpa_tc_few_unmasked_cuda",
        )?))
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn sdpa_masked_few(
    queries: &MxArray,
    keys: &MxArray,
    values: &MxArray,
    mask: &MxArray,
    head_dim: i64,
    gqa: i64,
    seq_len: i64,
    scale: f32,
) -> Result<Option<MxArray>> {
    let m = gqa * seq_len;
    let ok_single = (1..=96).contains(&m);
    let ok_two_tile = seq_len > 8 && gqa * 8 <= 96;
    // Note: Rust `DType` has no `Bool` variant (`mlx_array_dtype` returns -1
    // for `bool_`, which `MxArray::dtype()` maps to `Err`), so C++ enforces
    // `mask.dtype() == bool_` and returns null on non-boolean masks.
    if !(1..=16).contains(&seq_len)
        || !(head_dim == 128 || head_dim == 256)
        || gqa < 1
        || (!ok_single && !ok_two_tile)
        || queries.dtype()? != DType::BFloat16
        || keys.dtype()? != DType::BFloat16
        || values.dtype()? != DType::BFloat16
        || mask.ndim()? != 4
    {
        return Ok(None);
    }
    let handle = unsafe {
        sys::mlx_sdpa_tc_few_masked_cuda(
            queries.as_raw_ptr(),
            keys.as_raw_ptr(),
            values.as_raw_ptr(),
            mask.as_raw_ptr(),
            scale,
            false,
        )
    };
    if handle.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(
            handle,
            "sdpa_tc_few_masked_cuda",
        )?))
    }
}

#[inline]
pub(super) fn sdpa_prefill_d256_masked(
    queries: &MxArray,
    keys: &MxArray,
    values: &MxArray,
    mask: &MxArray,
    head_dim: i64,
    seq_len: i64,
    scale: f32,
) -> Result<Option<MxArray>> {
    // C++ enforces `mask.dtype() == bool_` and returns null on non-boolean masks.
    if seq_len < 9
        || head_dim != 256
        || queries.dtype()? != DType::BFloat16
        || keys.dtype()? != DType::BFloat16
        || values.dtype()? != DType::BFloat16
        || mask.ndim()? != 4
    {
        return Ok(None);
    }
    let handle = unsafe {
        sys::mlx_sdpa_prefill_d256_masked_cuda(
            queries.as_raw_ptr(),
            keys.as_raw_ptr(),
            values.as_raw_ptr(),
            mask.as_raw_ptr(),
            scale,
            false,
        )
    };
    if handle.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(
            handle,
            "sdpa_prefill_d256_masked_cuda",
        )?))
    }
}

#[inline]
pub(super) fn sdpa_prefill_d256_causal_valid(
    queries: &MxArray,
    keys: &MxArray,
    values: &MxArray,
    valid_mask: &MxArray,
    scale: f32,
) -> Result<Option<MxArray>> {
    // C++ enforces `valid_mask.dtype() == bool_` (Rust `DType` has no `Bool`
    // variant, so do not call `valid_mask.dtype()`) and returns null on
    // non-boolean masks or unsupported shapes.
    if queries.ndim()? != 4
        || keys.ndim()? != 4
        || values.ndim()? != 4
        || valid_mask.ndim()? != 4
        || valid_mask.shape_at(2)? != 1
    {
        return Ok(None);
    }
    let seq_len = queries.shape_at(2)?;
    let head_dim = queries.shape_at(3)?;
    let kv_len = keys.shape_at(2)?;
    if seq_len < 9
        || kv_len < seq_len
        || head_dim != 256
        || queries.dtype()? != DType::BFloat16
        || keys.dtype()? != DType::BFloat16
        || values.dtype()? != DType::BFloat16
    {
        return Ok(None);
    }
    let handle = unsafe {
        sys::mlx_sdpa_prefill_d256_masked_cuda(
            queries.as_raw_ptr(),
            keys.as_raw_ptr(),
            values.as_raw_ptr(),
            valid_mask.as_raw_ptr(),
            scale,
            true,
        )
    };
    if handle.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(
            handle,
            "sdpa_prefill_d256_causal_valid_cuda",
        )?))
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn sdpa_prefill_d256_masked_gated(
    queries: &MxArray,
    keys: &MxArray,
    values: &MxArray,
    gate: &MxArray,
    mask: &MxArray,
    head_dim: i64,
    seq_len: i64,
    scale: f32,
) -> Result<Option<MxArray>> {
    // C++ enforces `mask.dtype() == bool_` and returns null on non-boolean masks.
    if seq_len < 9
        || head_dim != 256
        || queries.dtype()? != DType::BFloat16
        || keys.dtype()? != DType::BFloat16
        || values.dtype()? != DType::BFloat16
        || gate.dtype()? != DType::BFloat16
        || gate.ndim()? != 3
        || mask.ndim()? != 4
    {
        return Ok(None);
    }
    let handle = unsafe {
        sys::mlx_cuda_sdpa_prefill_d256_gated(
            queries.as_raw_ptr(),
            keys.as_raw_ptr(),
            values.as_raw_ptr(),
            gate.as_raw_ptr(),
            mask.as_raw_ptr(),
            scale,
            false,
        )
    };
    if handle.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(
            handle,
            "sdpa_prefill_d256_masked_gated_cuda",
        )?))
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn sdpa_prefill_d256_causal_valid_gated(
    queries: &MxArray,
    keys: &MxArray,
    values: &MxArray,
    gate: &MxArray,
    valid_mask: &MxArray,
    scale: f32,
) -> Result<Option<MxArray>> {
    // C++ enforces `valid_mask.dtype() == bool_` (Rust `DType` has no `Bool`
    // variant, so do not call `valid_mask.dtype()`) and returns null on
    // non-boolean masks or unsupported shapes.
    if queries.ndim()? != 4
        || keys.ndim()? != 4
        || values.ndim()? != 4
        || gate.ndim()? != 3
        || valid_mask.ndim()? != 4
        || valid_mask.shape_at(2)? != 1
    {
        return Ok(None);
    }
    let seq_len = queries.shape_at(2)?;
    let head_dim = queries.shape_at(3)?;
    let kv_len = keys.shape_at(2)?;
    if seq_len < 9
        || kv_len < seq_len
        || head_dim != 256
        || queries.dtype()? != DType::BFloat16
        || keys.dtype()? != DType::BFloat16
        || values.dtype()? != DType::BFloat16
        || gate.dtype()? != DType::BFloat16
    {
        return Ok(None);
    }
    let handle = unsafe {
        sys::mlx_cuda_sdpa_prefill_d256_gated(
            queries.as_raw_ptr(),
            keys.as_raw_ptr(),
            values.as_raw_ptr(),
            gate.as_raw_ptr(),
            valid_mask.as_raw_ptr(),
            scale,
            true,
        )
    };
    if handle.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(
            handle,
            "sdpa_prefill_d256_causal_valid_gated_cuda",
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
pub(super) fn qk_norm_rope_positions(
    queries: &MxArray,
    keys: &MxArray,
    q_norm_w: &MxArray,
    k_norm_w: &MxArray,
    positions: &MxArray,
    eps: f32,
    rope_dims: i32,
    rope_base: f32,
    rope_scale: f32,
    precise_f32: bool,
) -> Result<Option<(MxArray, MxArray)>> {
    let expected_w_dt = if precise_f32 {
        DType::Float32
    } else {
        DType::BFloat16
    };
    if queries.dtype()? != DType::BFloat16
        || keys.dtype()? != DType::BFloat16
        || q_norm_w.dtype()? != expected_w_dt
        || k_norm_w.dtype()? != expected_w_dt
        || positions.dtype()? != DType::Int32
        || positions.ndim()? != 2
    {
        return Ok(None);
    }
    let mut out_q = std::ptr::null_mut();
    let mut out_k = std::ptr::null_mut();
    let ok = unsafe {
        sys::mlx_qk_norm_rope_positions_cuda(
            queries.as_raw_ptr(),
            keys.as_raw_ptr(),
            q_norm_w.as_raw_ptr(),
            k_norm_w.as_raw_ptr(),
            positions.as_raw_ptr(),
            eps,
            rope_dims,
            rope_base,
            rope_scale,
            precise_f32,
            &mut out_q,
            &mut out_k,
        )
    };
    if !ok || out_q.is_null() || out_k.is_null() {
        return Ok(None);
    }
    Ok(Some((
        MxArray::from_handle(out_q, "qk_norm_rope_positions:q")?,
        MxArray::from_handle(out_k, "qk_norm_rope_positions:k")?,
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
