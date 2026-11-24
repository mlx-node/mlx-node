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
#[allow(clippy::too_many_arguments)]
pub(super) fn selector_edges(
    candidates: &MxArray,
    unary: &MxArray,
    hidden: &MxArray,
    predecessor: &MxArray,
    successor: &MxArray,
    anchor: &MxArray,
    softcap: f32,
) -> Option<MxArray> {
    let scores = unsafe {
        sys::mlx_dflash2_selector_edges(
            candidates.as_raw_ptr(),
            unary.as_raw_ptr(),
            hidden.as_raw_ptr(),
            predecessor.as_raw_ptr(),
            successor.as_raw_ptr(),
            anchor.as_raw_ptr(),
            softcap,
        )
    };
    if scores.is_null() {
        return None;
    }
    MxArray::from_handle(scores, "dflash2_selector_edges").ok()
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
