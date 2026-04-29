//! `PagedAttentionInputs` — standardized metadata bundle threaded into
//! every paged-attention model's compiled forward graph (Phase 3+).
//!
//! This is the Rust-side mirror of the C++ struct
//! `mlx::core::fast::paged::PagedAttentionInputs` declared in
//! `crates/mlx-sys/src/mlx_common.h`. Producers (the
//! `PagedKVCacheAdapter`) materialize the 6 MxArrays once per request via
//! [`PagedKVCacheAdapter::build_paged_attention_inputs`] and hand the
//! bundle into the model wrapper; the wrapper forwards each MxArray as
//! an input to the compiled C++ forward graph.
//!
//! # Why a dedicated struct?
//!
//! Phases 4-9 (one per model migration) all need the same 6 inputs in the
//! same shapes/dtypes. Bundling them into a struct that's the same
//! between all models means:
//!
//! 1. Each model's C++ forward graph has a uniform compile-cache key —
//!    re-tracing on shape changes (per-request data variation) is
//!    impossible because shapes are FIXED at compile time and only the
//!    contents change.
//! 2. Refactoring the metadata flow (e.g. adding FP8 scales as a 7th
//!    field, or splitting decode vs. prefill into separate bundles) only
//!    touches one definition.
//! 3. Test coverage is shared — every model wrapper that accepts
//!    `PagedAttentionInputs` benefits from the same struct-level
//!    invariants.
//!
//! # Field semantics
//!
//! See the Rustdoc on [`PagedKVCacheAdapter::build_paged_attention_inputs`]
//! and the comments on the C++ struct in `mlx_common.h` for full layout
//! details.

use crate::array::MxArray;

/// Standardized metadata bundle for paged-attention model forward graphs.
///
/// Each field is an `MxArray` of fixed compile-time shape; only contents
/// vary per request. See the module-level doc and the C++ mirror in
/// `mlx_common.h` for layout details.
///
/// `MxArray` doesn't implement `Debug` (its handle is an opaque MLX
/// pointer); we don't derive Debug on the bundle either. `Clone` IS
/// derived because `MxArray::clone` is `Arc::clone` of the handle —
/// cheap reference bumping that mirrors how MLX itself shares array
/// descriptors.
#[derive(Clone)]
pub struct PagedAttentionInputs {
    /// `[1]` int32 — global token position of the first new token.
    pub offset_arr: MxArray,
    /// `[1, max_blocks_per_seq]` int32, sentinel-padded with -1.
    pub block_table: MxArray,
    /// `[chunk_size_max]` int64, sentinel-padded with -1.
    pub slot_mapping: MxArray,
    /// `[1]` int32 — valid prefix length of `slot_mapping`.
    pub num_valid_tokens: MxArray,
    /// `[1]` int32 — valid prefix length of `block_table`.
    pub num_valid_blocks: MxArray,
    /// `[1]` int32 — total context length so far (= `block_table.num_tokens`
    /// after the chunk being written).
    pub seq_lens: MxArray,
}
