//! Backend-neutral DFlash2 selector (top-k16 & greedy predecessor path) facades.

use mlx_sys as sys;
use napi::bindgen_prelude::*;

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

/// Outputs of GPU-resident [`select_tree_cuda`] (`T = max_nodes`).
pub(crate) struct SelectTreeOutputs {
    /// `[1, T]` Uint32 — DFS-preorder token IDs (`tree_tokens[0, 0] == anchor`).
    pub tree_tokens: MxArray,
    /// `[T]` Int32 — DFS-preorder parent index (`parent[0] == -1`, `0 <= parent[i] < i`).
    pub parent: MxArray,
    /// `[T]` Int32 — node depth (`depths[0] == 0`).
    pub depths: MxArray,
    /// `[T]` Uint32 — ancestor bitmask (`mask[0] == 1`).
    pub ancestors_mask: MxArray,
    /// `[T, 4]` Int32 — GDN conv1d tap indices (`tap3 == i`).
    pub conv_taps: MxArray,
    /// `[T, 4]` Float32 — `[draft_prob, estimate, cap, flags]` where `flags as u32` packs:
    /// - bit 0 (`0x001`): `on_chain`
    /// - bits 1..2 (`(f >> 1) & 0x3`): `LookupAgreement` (`0=Top, 1=Candidate, 2=Absent, 3=Unknown`)
    /// - bit 3 (`0x008`): `is_valid` (`i < actual_count`)
    /// - bits 4..8 (`(f >> 4) & 0x1F`): `actual_count` (`1..=T`)
    /// - bit 9 (`0x200`): `is_chain`
    pub node_meta: MxArray,
}

/// GPU-resident single-block DFlash2 speculative verification tree builder.
///
/// Builds the `T <= 16` DFS-preorder verification tree directly on CUDA from
/// device-resident `candidates` (`[1, L, 16]` i32) and `scores` (`[L, 16, 16]` f32),
/// eliminating the host sync bubble between `propose_tree_lattice` and `verify_tree`.
///
/// `lookup_rates` may be either `[16, 4]` f32 (indexed by `(depth, agreement)` for
/// exact parity with `verify_tree::build_verify_tree`) or `[16]` f32 (flat per-depth).
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn select_tree_cuda(
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
    if active_backend() != ActiveBackend::CudaGpu {
        return Ok(None);
    }
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
