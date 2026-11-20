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
