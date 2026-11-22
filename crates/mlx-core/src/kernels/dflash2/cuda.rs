//! CUDA DFlash2 selector & speculative verification tree kernel wrappers.

use mlx_sys as sys;
use napi::bindgen_prelude::*;

use super::SelectTreeOutputs;
use crate::array::MxArray;

#[inline]
pub(super) fn topk16(logits: &MxArray) -> Option<(MxArray, MxArray)> {
    let mut ids = std::ptr::null_mut();
    let mut values = std::ptr::null_mut();
    let status =
        unsafe { sys::mlx_dflash2_topk16_cuda(logits.as_raw_ptr(), &mut ids, &mut values) };
    if status != 1 || ids.is_null() || values.is_null() {
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
        sys::mlx_dflash2_greedy_path_cuda(candidates.as_raw_ptr(), scores.as_raw_ptr(), &mut path)
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
        sys::mlx_dflash2_greedy_path_confidence_cuda(
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

#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn select_tree(
    candidates: &MxArray,
    scores: &MxArray,
    lookup_tokens: &MxArray,
    lookup_rates: &MxArray,
    budget_intercepts: &MxArray,
    anchor: u32,
    lookup_len: i32,
    max_nodes: i32,
    max_depth: i32,
    budget_slope: f32,
) -> Result<Option<SelectTreeOutputs>> {
    let mut outputs = [std::ptr::null_mut(); 6];
    let ok = unsafe {
        sys::mlx_dflash2_select_tree_cuda(
            candidates.as_raw_ptr(),
            scores.as_raw_ptr(),
            lookup_tokens.as_raw_ptr(),
            lookup_rates.as_raw_ptr(),
            budget_intercepts.as_raw_ptr(),
            anchor,
            lookup_len,
            max_nodes,
            max_depth,
            budget_slope,
            outputs.as_mut_ptr(),
        )
    };
    if !ok {
        return Ok(None);
    }
    Ok(Some(SelectTreeOutputs {
        tree_tokens: MxArray::from_handle(outputs[0], "dflash2_select_tree:tree_tokens")?,
        parent: MxArray::from_handle(outputs[1], "dflash2_select_tree:parent")?,
        depths: MxArray::from_handle(outputs[2], "dflash2_select_tree:depths")?,
        ancestors_mask: MxArray::from_handle(outputs[3], "dflash2_select_tree:ancestors_mask")?,
        conv_taps: MxArray::from_handle(outputs[4], "dflash2_select_tree:conv_taps")?,
        node_meta: MxArray::from_handle(outputs[5], "dflash2_select_tree:node_meta")?,
    }))
}
