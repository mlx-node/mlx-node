//! Backend-neutral in-place KV cache row-store & ring-window tail-store facades.

mod cuda;
mod metal;

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
    let [full_k, full_v, view_k, view_v] = window else {
        return Err(Error::from_reason(
            "compiled draft window expects full K/V and two views",
        ));
    };
    let window = [full_k, full_v, view_k, view_v];
    match active_backend() {
        ActiveBackend::MetalGpu => metal::store_window_tail(window, keys, values),
        ActiveBackend::CudaGpu => cuda::store_window_tail(window, keys, values),
        ActiveBackend::Cpu => Ok(None),
    }
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
    match active_backend() {
        ActiveBackend::CudaGpu => cuda::qk_norm_rope_kv_store(
            q_in, k_in, v_in, q_norm_w, k_norm_w, k_cache, v_cache, eps, rope_dims, rope_base,
            rope_scale, offset,
        ),
        ActiveBackend::MetalGpu | ActiveBackend::Cpu => Ok(None),
    }
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
    match active_backend() {
        ActiveBackend::CudaGpu => cuda::qk_norm_rope_kv_store_tree(
            q_in, k_in, v_in, q_norm_w, k_norm_w, k_cache, v_cache, depths, eps, rope_dims,
            rope_base, rope_scale, offset,
        ),
        ActiveBackend::MetalGpu | ActiveBackend::Cpu => Ok(None),
    }
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
    match active_backend() {
        ActiveBackend::CudaGpu => {
            cuda::kv_compact_rows(k_cache, v_cache, base_offset, accepted_nodes)
        }
        ActiveBackend::MetalGpu | ActiveBackend::Cpu => Ok(None),
    }
}
