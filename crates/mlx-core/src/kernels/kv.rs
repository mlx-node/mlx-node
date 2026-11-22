//! Backend-neutral in-place KV cache row-store & ring-window tail-store facades.

use mlx_sys as sys;
use napi::bindgen_prelude::*;

use crate::accelerator::{ActiveBackend, active_backend};
use crate::array::MxArray;

/// Write 2 KV tensors (`[K, V]`) at `offsets` in place into preallocated cache buffers.
/// Returns `Ok(Some([new_k_buf, new_v_buf]))` when the GPU primitive ran, or `Ok(None)`
/// on CPU / unsupported geometry so the caller falls back to `slice_update`.
#[inline]
pub(crate) fn store_kv_pair(
    dst_k: &MxArray,
    dst_v: &MxArray,
    src_k: &MxArray,
    src_v: &MxArray,
    offset: i32,
) -> Result<Option<(MxArray, MxArray)>> {
    match active_backend() {
        ActiveBackend::MetalGpu | ActiveBackend::CudaGpu => {
            let dst = [dst_k.as_raw_ptr(), dst_v.as_raw_ptr()];
            let src = [src_k.as_raw_ptr(), src_v.as_raw_ptr()];
            let offsets = [offset; 2];
            let mut out = [std::ptr::null_mut(); 2];
            let ok = unsafe {
                sys::mlx_kv_store_rows(
                    2,
                    dst.as_ptr(),
                    src.as_ptr(),
                    offsets.as_ptr(),
                    out.as_mut_ptr(),
                )
            };
            if ok && !out[0].is_null() && !out[1].is_null() {
                Ok(Some((
                    MxArray::from_handle(out[0], "kv_store_rows:keys")?,
                    MxArray::from_handle(out[1], "kv_store_rows:values")?,
                )))
            } else {
                Ok(None)
            }
        }
        ActiveBackend::Cpu => Ok(None),
    }
}

/// Write new K/V rows into the compiled draft ring-window tail in place.
/// `window` holds `[full_k, full_v, view_k, view_v]`.
#[inline]
pub(crate) fn store_window_tail(
    window: &[MxArray],
    keys: &MxArray,
    values: &MxArray,
) -> Result<Option<Vec<MxArray>>> {
    if window.len() != 4 {
        return Err(Error::from_reason(
            "compiled draft window expects full K/V and two views",
        ));
    }
    let args = [
        window[0].as_raw_ptr(),
        window[1].as_raw_ptr(),
        window[2].as_raw_ptr(),
        window[3].as_raw_ptr(),
        keys.as_raw_ptr(),
        values.as_raw_ptr(),
    ];
    let mut out = [std::ptr::null_mut(); 4];
    let stored = match active_backend() {
        ActiveBackend::MetalGpu => unsafe {
            sys::mlx_kv_store_window_tail(args.as_ptr(), out.as_mut_ptr())
        },
        ActiveBackend::CudaGpu => unsafe {
            sys::mlx_kv_store_window_tail_cuda(args.as_ptr(), out.as_mut_ptr())
        },
        ActiveBackend::Cpu => false,
    };
    if !stored {
        return Ok(None);
    }
    let handles = out
        .into_iter()
        .map(|handle| MxArray::from_handle(handle, "compiled draft tail store"))
        .collect::<Result<Vec<_>>>()?;
    Ok(Some(handles))
}

/// Fused CUDA QK-Norm + non-traditional partial RoPE + in-place KV cache store.
///
/// Inputs `q_in` (`[B, T, Hq, D]`), `k_in` / `v_in` (`[B, T, Hkv, D]`),
/// `q_norm_w` / `k_norm_w` (`[D]`), and preallocated cache buffers
/// `k_cache` / `v_cache` (`[B, Hkv, cap, D]`) must all be BF16.
///
/// Returns `Ok(Some((q_out, new_k_cache, new_v_cache)))` where `q_out` is
/// `[B, Hq, T, D]` (SDPA layout) and `new_k_cache` / `new_v_cache` alias the
/// updated cache buffers, or `Ok(None)` on non-CUDA / unsupported geometry so
/// the caller falls back to the unfused path.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn qk_norm_rope_kv_store(
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
    if active_backend() != ActiveBackend::CudaGpu {
        return Ok(None);
    }
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

/// Tree-RoPE variant of fused CUDA QK-Norm + non-traditional partial RoPE +
/// in-place KV cache store. RoPE position for token `t` is `offset + depths[t]`,
/// while KV store writes at slot `offset + t`.
#[inline]
#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn qk_norm_rope_kv_store_tree(
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
    if active_backend() != ActiveBackend::CudaGpu {
        return Ok(None);
    }
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

/// Race-free in-place CUDA KV cache row compaction for tree acceptance (`keep <= 16`).
/// Copies `cache[:, :, base_offset + accepted_nodes[r], :] -> cache[:, :, base_offset + r, :]`
/// for both K and V, returning aliased cache handles.
#[inline]
#[allow(dead_code)]
pub(crate) fn kv_compact_rows(
    k_cache: &MxArray,
    v_cache: &MxArray,
    base_offset: i32,
    accepted_nodes: &MxArray,
) -> Result<Option<(MxArray, MxArray)>> {
    if active_backend() != ActiveBackend::CudaGpu {
        return Ok(None);
    }
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
