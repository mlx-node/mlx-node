//! Shared token-dispatch primitives for Mixture-of-Experts forwards.
//!
//! Both Qwen3.5 MoE and the privacy-filter gpt-oss MoE FFN use the same
//! "sort-tokens-by-expert" trick before invoking `gather_mm` /
//! `gather_qmm`: sorting makes each expert's slice of the dispatch
//! contiguous, which lets `gather_mm(sorted=true)` stream over one
//! expert at a time instead of rebinding the RHS per slot.
//!
//! This module owns the canonical implementation. Model-specific call
//! sites (see `crate::models::privacy_filter::experts` and
//! `crate::models::qwen3_5_moe::switch_glu`) import these helpers
//! instead of carrying their own near-identical copies.
//!
//! The matching `do_sort = indices.size() >= 64` heuristic lives at the
//! call sites because it depends on shape introspection they already
//! perform.

use crate::array::MxArray;
use napi::bindgen_prelude::*;

/// Result of the gather-and-sort pass: tokens permuted so that every
/// expert's slice of the dispatch is contiguous, plus the inverse
/// permutation needed to undo the sort.
pub(crate) struct SortedDispatch {
    /// Tokens replicated and reordered to align with `idx_sorted`.
    /// Shape `[N*K, 1, d]`.
    pub(crate) x_sorted: MxArray,
    /// Per-slot expert ids, sorted ascending (so `gather_mm(..., true)`
    /// can stream over each expert's block in one pass). Shape `[N*K]`.
    pub(crate) idx_sorted: MxArray,
    /// Inverse permutation used by [`scatter_unsort`] to restore the
    /// original token ordering. Shape `[N*K]`.
    pub(crate) inv_order: MxArray,
}

/// Permute `x` so that tokens routed to the same expert are contiguous.
///
/// `indices` has shape `[..., K]` (any leading dims, trailing top-k
/// axis); `x` has shape `[..., d]` (the same leading token dims with a
/// trailing feature axis). Returns sorted tokens in the canonical
/// `[N*K, 1, d]` layout expected by `gather_mm` / `gather_qmm`.
///
/// Returns an error rather than panicking when either input has no
/// trailing dimension to read.
pub(crate) fn gather_sort(x: &MxArray, indices: &MxArray) -> Result<SortedDispatch> {
    let idx_shape = indices.shape()?;
    let m = *idx_shape
        .last()
        .ok_or_else(|| Error::from_reason("gather_sort: empty indices"))?;

    let flat_indices = indices.reshape(&[-1])?;
    let order = flat_indices.argsort(Some(-1))?;
    let inv_order = order.argsort(Some(-1))?;
    let idx_sorted = flat_indices.take(&order, 0)?;

    let x_shape = x.shape()?;
    let d = *x_shape
        .last()
        .ok_or_else(|| Error::from_reason("gather_sort: empty x"))?;
    // Collapse all leading dims into a single token axis; insert a unit
    // axis so the subsequent gather_mm sees the canonical
    // `[tokens, 1, d]` layout used by `mx.gather_mm`.
    let x_flat = x.reshape(&[-1, 1, d])?;
    let m_scalar = MxArray::scalar_int(m as i32)?;
    let token_indices = order.floor_divide(&m_scalar)?;
    let x_sorted = x_flat.take(&token_indices, 0)?;

    Ok(SortedDispatch {
        x_sorted,
        idx_sorted,
        inv_order,
    })
}

/// Undo [`gather_sort`]: reorder rows of `x` by `inv_order` and reshape
/// the leading flat-token axis back into `orig_shape` (the unflattened
/// original token shape, e.g. `(B, T, K)`).
pub(crate) fn scatter_unsort(
    x: &MxArray,
    inv_order: &MxArray,
    orig_shape: &[i64],
) -> Result<MxArray> {
    let unsorted = x.take(inv_order, 0)?;
    let x_shape = unsorted.shape()?;
    let mut new_shape = orig_shape.to_vec();
    for &dim in &x_shape[1..] {
        new_shape.push(dim);
    }
    unsorted.reshape(&new_shape)
}
