//! Metal DFlash2 selector (top-k16 & greedy predecessor path) kernel wrappers.

use mlx_sys as sys;

use crate::array::MxArray;

#[inline]
pub(super) fn topk16(logits: &MxArray) -> Option<(MxArray, MxArray)> {
    let mut ids = std::ptr::null_mut();
    let mut values = std::ptr::null_mut();
    let ok = unsafe { sys::mlx_dflash2_topk16(logits.as_raw_ptr(), &mut ids, &mut values) };
    if !ok || ids.is_null() || values.is_null() {
        return None;
    }
    let ids = MxArray::from_handle(ids, "dflash2_topk16:ids").ok()?;
    let values = MxArray::from_handle(values, "dflash2_topk16:values").ok()?;
    Some((ids, values))
}

#[inline]
pub(super) fn greedy_path(candidates: &MxArray, scores: &MxArray) -> Option<MxArray> {
    let mut path = std::ptr::null_mut();
    let ok = unsafe {
        sys::mlx_dflash2_greedy_path(candidates.as_raw_ptr(), scores.as_raw_ptr(), &mut path)
    };
    if !ok || path.is_null() {
        return None;
    }
    MxArray::from_handle(path, "dflash2_greedy_path").ok()
}

#[inline]
pub(super) fn greedy_path_confidence(
    candidates: &MxArray,
    scores: &MxArray,
) -> Option<(MxArray, MxArray)> {
    let mut path = std::ptr::null_mut();
    let mut confidence = std::ptr::null_mut();
    let ok = unsafe {
        sys::mlx_dflash2_greedy_path_confidence(
            candidates.as_raw_ptr(),
            scores.as_raw_ptr(),
            &mut path,
            &mut confidence,
        )
    };
    if !ok || path.is_null() || confidence.is_null() {
        return None;
    }
    Some((
        MxArray::from_handle(path, "dflash2_greedy_path:path").ok()?,
        MxArray::from_handle(confidence, "dflash2_greedy_path:confidence").ok()?,
    ))
}
