//! Backend-neutral DFlash2 selector (top-k16 & greedy predecessor path) facades.

use mlx_sys as sys;

use crate::accelerator::{ActiveBackend, active_backend};
use crate::array::MxArray;

/// Fused top-16 candidate scan over `[1, rows, vocab]` logits.
/// Returns `(ids, f32_values)` on GPU backends, or `None` on decline/CPU.
#[inline]
pub(crate) fn topk16(logits: &MxArray) -> Option<(MxArray, MxArray)> {
    let mut ids = std::ptr::null_mut();
    let mut values = std::ptr::null_mut();
    let status = match active_backend() {
        ActiveBackend::MetalGpu => {
            (unsafe { sys::mlx_dflash2_topk16(logits.as_raw_ptr(), &mut ids, &mut values) }) as i32
        }
        ActiveBackend::CudaGpu => unsafe {
            sys::mlx_dflash2_topk16_cuda(logits.as_raw_ptr(), &mut ids, &mut values)
        },
        ActiveBackend::Cpu => return None,
    };
    if status != 1 || ids.is_null() || values.is_null() {
        return None;
    }
    let ids = MxArray::from_handle(ids, "dflash2_topk16:ids").ok()?;
    let values = MxArray::from_handle(values, "dflash2_topk16:values").ok()?;
    Some((ids, values))
}

/// Fused device-resident greedy predecessor path walk.
#[inline]
pub(crate) fn greedy_path(candidates: &MxArray, scores: &MxArray) -> Option<MxArray> {
    let mut path = std::ptr::null_mut();
    let ok = match active_backend() {
        ActiveBackend::MetalGpu => unsafe {
            sys::mlx_dflash2_greedy_path(candidates.as_raw_ptr(), scores.as_raw_ptr(), &mut path)
        },
        ActiveBackend::CudaGpu => unsafe {
            sys::mlx_dflash2_greedy_path_cuda(
                candidates.as_raw_ptr(),
                scores.as_raw_ptr(),
                &mut path,
            )
        },
        ActiveBackend::Cpu => return None,
    };
    if !ok || path.is_null() {
        return None;
    }
    MxArray::from_handle(path, "dflash2_greedy_path").ok()
}

/// Fused greedy predecessor path + temperature-1 confidence.
#[inline]
pub(crate) fn greedy_path_confidence(
    candidates: &MxArray,
    scores: &MxArray,
) -> Option<(MxArray, MxArray)> {
    let mut path = std::ptr::null_mut();
    let mut confidence = std::ptr::null_mut();
    let ok = match active_backend() {
        ActiveBackend::MetalGpu => unsafe {
            sys::mlx_dflash2_greedy_path_confidence(
                candidates.as_raw_ptr(),
                scores.as_raw_ptr(),
                &mut path,
                &mut confidence,
            )
        },
        ActiveBackend::CudaGpu => unsafe {
            sys::mlx_dflash2_greedy_path_confidence_cuda(
                candidates.as_raw_ptr(),
                scores.as_raw_ptr(),
                &mut path,
                &mut confidence,
            )
        },
        ActiveBackend::Cpu => return None,
    };
    if !ok || path.is_null() || confidence.is_null() {
        return None;
    }
    Some((
        MxArray::from_handle(path, "dflash2_greedy_path:path").ok()?,
        MxArray::from_handle(confidence, "dflash2_greedy_path:confidence").ok()?,
    ))
}
