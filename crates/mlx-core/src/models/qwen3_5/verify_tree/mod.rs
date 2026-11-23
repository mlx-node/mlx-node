//! Pure-Rust speculative verification tree builder and suffix-automaton prompt
//! lookup for Qwen3.8 + DFlash2 tree decoding.
//!
//! Ports Splash's proven components to Rust while matching our Wave B CUDA
//! kernel contracts (`mlx_gdn_prepare_tree_cuda`, `mlx_gdn_tree_step_cuda`,
//! `mlx_qk_norm_rope_kv_store_tree_cuda`, `mlx_sdpa_tc_tree_few_cuda`, and
//! `mlx_kv_compact_rows_cuda`):
//! - [`LookupIndex`]: online suffix automaton (SAM) finding the most recent
//!   earlier occurrence of the longest suffix in `[3, 64]` tokens.
//! - [`LookupAcceptance`]: Splash's trace-fitted acceptance prior table plus
//!   exponentially decayed online counts per `(length, depth, agreement)`.
//! - [`VerifyBudget`]: Splash's logistic acceptance calibration
//!   `logit(p) = slope * logit(q) + intercept[pos]` with damped-Newton MAP refit.
//! - [`build_verify_tree`]: best-first speculative tree (up to 16 nodes)
//!   combining the DFlash2 Markov selector lattice and the lookup chain, with
//!   parent-child probability capping, `(parent, token)` deduplication, and
//!   highest-probability-first DFS preorder renumbering.

// Wave B2 wires this module into `dflash2_decode.rs`; allow unused until then.
#![allow(dead_code)]

mod budget;
mod lattice;
mod sam;
#[cfg(test)]
mod tests;
mod tree;

pub(crate) use budget::VerifyBudget;
#[cfg(test)]
pub(crate) use budget::{BudgetAcceptance, BudgetLabel};
pub(crate) use lattice::DraftLattice;
#[cfg(test)]
pub(crate) use sam::LookupMatchInfo;
pub(crate) use sam::{LookupAcceptance, LookupAgreement, LookupIndex};
pub(crate) use tree::{
    TreeCostModel, TreeDeviceViews, VerifyTree, build_verify_tree, build_verify_tree_cost_aware,
};

/// Maximum nodes in one verification tree (root anchor + up to 15 speculative
/// tokens). Fits in the `u32` ancestor bitmask and CUDA Wave B `T <= 16` limits.
pub(crate) const MAX_TREE_NODES: usize = 16;

/// Maximum lookup continuation length past the root anchor (depths `1..=15`).
pub(crate) const MAX_LOOKUP_CHAIN: usize = 15;

/// Top-K candidates per DFlash2 draft proposal position.
pub(crate) const DRAFT_CANDIDATES: usize = 16;

/// Number of DFlash2 draft proposal slots (`L = block_size - 1 = 7`, depths `1..=7`).
pub(crate) const DRAFT_SLOTS: usize = 7;

/// Maximum draft children expanded per tree parent (Splash `SPLASH_TREE_CHILDREN`).
pub(crate) const MAX_BRANCH_CHILDREN: usize = 4;
