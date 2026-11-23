//! Backend-neutral DFlash2 selector (top-k16 & greedy predecessor path) facades.

mod cuda;
mod metal;

use napi::bindgen_prelude::*;

use crate::accelerator::{ActiveBackend, active_backend};
use crate::array::MxArray;

/// Fused top-16 candidate scan over `[1, rows, vocab]` logits.
/// Returns `(ids, f32_values)` on GPU backends, or `None` on decline/CPU.
#[inline]
pub(crate) fn topk16(logits: &MxArray) -> Option<(MxArray, MxArray)> {
    match active_backend() {
        ActiveBackend::MetalGpu => metal::topk16(logits),
        ActiveBackend::CudaGpu => cuda::topk16(logits),
        ActiveBackend::Cpu => None,
    }
}

/// Fused device-resident greedy predecessor path walk.
#[inline]
pub(crate) fn greedy_path(candidates: &MxArray, scores: &MxArray) -> Option<MxArray> {
    match active_backend() {
        ActiveBackend::MetalGpu => metal::greedy_path(candidates, scores),
        ActiveBackend::CudaGpu => cuda::greedy_path(candidates, scores),
        ActiveBackend::Cpu => None,
    }
}

/// Fused greedy predecessor path + temperature-1 confidence.
#[inline]
pub(crate) fn greedy_path_confidence(
    candidates: &MxArray,
    scores: &MxArray,
) -> Option<(MxArray, MxArray)> {
    match active_backend() {
        ActiveBackend::MetalGpu => metal::greedy_path_confidence(candidates, scores),
        ActiveBackend::CudaGpu => cuda::greedy_path_confidence(candidates, scores),
        ActiveBackend::Cpu => None,
    }
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
    match active_backend() {
        ActiveBackend::CudaGpu => cuda::select_tree(
            candidates,
            scores,
            lookup_tokens,
            lookup_rates,
            budget_intercepts,
            anchor,
            lookup_len,
            max_nodes,
            max_depth,
            budget_slope,
        ),
        ActiveBackend::MetalGpu | ActiveBackend::Cpu => Ok(None),
    }
}
