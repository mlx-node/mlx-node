//! CUDA KV cache store, fused QK-Norm/RoPE/KV-store, and compaction kernel wrappers.

use mlx_sys as sys;
use napi::bindgen_prelude::*;

use crate::array::MxArray;

#[inline]
pub(super) fn store_window_tail(
    window: [&MxArray; 4],
    keys: &MxArray,
    values: &MxArray,
) -> Result<Option<Vec<MxArray>>> {
    let args = [
        window[0].as_raw_ptr(),
        window[1].as_raw_ptr(),
        window[2].as_raw_ptr(),
        window[3].as_raw_ptr(),
        keys.as_raw_ptr(),
        values.as_raw_ptr(),
    ];
    let mut out = [std::ptr::null_mut(); 4];
    let stored = unsafe { sys::mlx_kv_store_window_tail_cuda(args.as_ptr(), out.as_mut_ptr()) };
    if !stored {
        return Ok(None);
    }
    let handles = out
        .into_iter()
        .map(|handle| MxArray::from_handle(handle, "compiled draft tail store"))
        .collect::<Result<Vec<_>>>()?;
    Ok(Some(handles))
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn qk_norm_rope_kv_store(
    q_in: &MxArray,
    k_in: &MxArray,
    v_in: &MxArray,
    q_norm_w: &MxArray,
    k_norm_w: &MxArray,
    k_cache: &MxArray,
    v_cache: &MxArray,
    eps: f32,
    rope_dims: i32,
    rope_base: f32,
    rope_scale: f32,
    offset: i32,
) -> Result<Option<(MxArray, MxArray, MxArray)>> {
    let mut out_q = std::ptr::null_mut();
    let mut out_k_cache = std::ptr::null_mut();
    let mut out_v_cache = std::ptr::null_mut();
    let ok = unsafe {
        sys::mlx_qk_norm_rope_kv_store_cuda(
            q_in.as_raw_ptr(),
            k_in.as_raw_ptr(),
            v_in.as_raw_ptr(),
            q_norm_w.as_raw_ptr(),
            k_norm_w.as_raw_ptr(),
            k_cache.as_raw_ptr(),
            v_cache.as_raw_ptr(),
            eps,
            rope_dims,
            rope_base,
            rope_scale,
            offset,
            &mut out_q,
            &mut out_k_cache,
            &mut out_v_cache,
        )
    };
    if !ok || out_q.is_null() || out_k_cache.is_null() || out_v_cache.is_null() {
        return Ok(None);
    }
    Ok(Some((
        MxArray::from_handle(out_q, "qk_norm_rope_kv_store:q")?,
        MxArray::from_handle(out_k_cache, "qk_norm_rope_kv_store:k_cache")?,
        MxArray::from_handle(out_v_cache, "qk_norm_rope_kv_store:v_cache")?,
    )))
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn qk_norm_rope_kv_store_tree(
    q_in: &MxArray,
    k_in: &MxArray,
    v_in: &MxArray,
    q_norm_w: &MxArray,
    k_norm_w: &MxArray,
    k_cache: &MxArray,
    v_cache: &MxArray,
    depths: &MxArray,
    eps: f32,
    rope_dims: i32,
    rope_base: f32,
    rope_scale: f32,
    offset: i32,
) -> Result<Option<(MxArray, MxArray, MxArray)>> {
    let mut out_q = std::ptr::null_mut();
    let mut out_k_cache = std::ptr::null_mut();
    let mut out_v_cache = std::ptr::null_mut();
    let ok = unsafe {
        sys::mlx_qk_norm_rope_kv_store_tree_cuda(
            q_in.as_raw_ptr(),
            k_in.as_raw_ptr(),
            v_in.as_raw_ptr(),
            q_norm_w.as_raw_ptr(),
            k_norm_w.as_raw_ptr(),
            k_cache.as_raw_ptr(),
            v_cache.as_raw_ptr(),
            depths.as_raw_ptr(),
            eps,
            rope_dims,
            rope_base,
            rope_scale,
            offset,
            &mut out_q,
            &mut out_k_cache,
            &mut out_v_cache,
        )
    };
    if !ok || out_q.is_null() || out_k_cache.is_null() || out_v_cache.is_null() {
        return Ok(None);
    }
    Ok(Some((
        MxArray::from_handle(out_q, "qk_norm_rope_kv_store_tree:q")?,
        MxArray::from_handle(out_k_cache, "qk_norm_rope_kv_store_tree:k_cache")?,
        MxArray::from_handle(out_v_cache, "qk_norm_rope_kv_store_tree:v_cache")?,
    )))
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn qk_norm_rope_kv_store_positions(
    q_in: &MxArray,
    k_in: &MxArray,
    v_in: &MxArray,
    q_norm_w: &MxArray,
    k_norm_w: &MxArray,
    k_cache: &MxArray,
    v_cache: &MxArray,
    positions: &MxArray,
    offset: i32,
    eps: f32,
    rope_dims: i32,
    rope_base: f32,
    rope_scale: f32,
    precise_f32: bool,
) -> Result<Option<(MxArray, MxArray, MxArray)>> {
    let mut out_q = std::ptr::null_mut();
    let mut out_k_cache = std::ptr::null_mut();
    let mut out_v_cache = std::ptr::null_mut();
    let ok = unsafe {
        sys::mlx_qk_norm_rope_kv_store_positions_cuda(
            q_in.as_raw_ptr(),
            k_in.as_raw_ptr(),
            v_in.as_raw_ptr(),
            q_norm_w.as_raw_ptr(),
            k_norm_w.as_raw_ptr(),
            k_cache.as_raw_ptr(),
            v_cache.as_raw_ptr(),
            positions.as_raw_ptr(),
            offset,
            eps,
            rope_dims,
            rope_base,
            rope_scale,
            precise_f32,
            &mut out_q,
            &mut out_k_cache,
            &mut out_v_cache,
        )
    };
    if !ok || out_q.is_null() || out_k_cache.is_null() || out_v_cache.is_null() {
        return Ok(None);
    }
    Ok(Some((
        MxArray::from_handle(out_q, "qk_norm_rope_kv_store_positions:q")?,
        MxArray::from_handle(out_k_cache, "qk_norm_rope_kv_store_positions:k_cache")?,
        MxArray::from_handle(out_v_cache, "qk_norm_rope_kv_store_positions:v_cache")?,
    )))
}

#[inline]
pub(super) fn kv_compact_rows(
    k_cache: &MxArray,
    v_cache: &MxArray,
    base_offset: i32,
    accepted_nodes: &MxArray,
) -> Result<Option<(MxArray, MxArray)>> {
    let mut out_k = std::ptr::null_mut();
    let mut out_v = std::ptr::null_mut();
    let ok = unsafe {
        sys::mlx_kv_compact_rows_cuda(
            k_cache.as_raw_ptr(),
            v_cache.as_raw_ptr(),
            base_offset,
            accepted_nodes.as_raw_ptr(),
            &mut out_k,
            &mut out_v,
        )
    };
    if !ok || out_k.is_null() || out_v.is_null() {
        return Ok(None);
    }
    Ok(Some((
        MxArray::from_handle(out_k, "kv_compact_rows:k")?,
        MxArray::from_handle(out_v, "kv_compact_rows:v")?,
    )))
}
