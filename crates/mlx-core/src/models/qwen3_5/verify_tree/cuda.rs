//! Shared CUDA speculative tree-verification helpers used by both dense
//! Qwen3.8 (`qwen3_5::dflash2_decode`) and MoE Qwen4 (`qwen4_exp::decoder`).
//!
//! Deliberately leaves all GPU evaluation timing to callers (`qwen3_5` keeps
//! `SelectTreeOutputs` lazy until after verify; `qwen4_exp` evaluates it before
//! verify) and leaves cost-aware vs. fixed-width host tree selection to each
//! caller.

use napi::bindgen_prelude::*;

use crate::array::MxArray;
use crate::kernels::dflash2::SelectTreeOutputs;

use super::{
    DRAFT_SLOTS, DraftLattice, LookupAcceptance, LookupAgreement, LookupIndex, MAX_LOOKUP_CHAIN,
    MAX_TREE_NODES, TreeWalkResult, VerifyBudget, VerifyTree,
};

/// Snapshot of suffix-automaton lookup candidates and depth cap for one
/// speculative verification tree cycle.
pub(crate) struct TreeLookupQuery {
    pub eff_max_depth: usize,
    pub match_len: usize,
    pub proposals: Vec<u32>,
    pub lookup_len: usize,
}

/// Query `lookup_index` and compute the effective depth cap and usable lookup
/// prefix length for one speculative verification tree cycle.
pub(crate) fn prepare_tree_lookup(
    lookup_index: &LookupIndex,
    lookup_enabled: bool,
    max_depth: usize,
) -> TreeLookupQuery {
    let lookup_info = lookup_index.match_info();
    let eff_max_depth = max_depth.min(MAX_LOOKUP_CHAIN);
    let lookup_cap = MAX_LOOKUP_CHAIN.min(eff_max_depth);
    let proposals = if lookup_enabled && lookup_cap > 0 {
        lookup_index.proposals(lookup_cap)
    } else {
        Vec::new()
    };
    let lookup_len = if lookup_info.match_len >= LookupIndex::MIN_MATCH {
        proposals.len().min(MAX_LOOKUP_CHAIN)
    } else {
        0
    };
    TreeLookupQuery {
        eff_max_depth,
        match_len: lookup_info.match_len,
        proposals,
        lookup_len,
    }
}

/// Pack host lookup tables and budget calibration into device tensors and
/// invoke [`crate::kernels::dflash2::select_tree_cuda`].
///
/// Leaves all returned [`SelectTreeOutputs`] tensors lazy so callers retain
/// full control over GPU eval timing.
#[allow(clippy::too_many_arguments)]
pub(crate) fn select_tree_gpu(
    candidates: &MxArray,
    scores: &MxArray,
    lookup: &TreeLookupQuery,
    lookup_acceptance: &LookupAcceptance,
    verify_budget: &VerifyBudget,
    anchor: u32,
    node_budget: usize,
) -> Result<Option<SelectTreeOutputs>> {
    let mut lt = [0i32; MAX_TREE_NODES];
    for (dst, &tok) in lt
        .iter_mut()
        .zip(lookup.proposals.iter().take(lookup.lookup_len))
    {
        *dst = tok as i32;
    }
    let mut lr = [0.0f32; MAX_TREE_NODES * 4];
    if lookup.lookup_len > 0 {
        let agreements = [
            LookupAgreement::Top,
            LookupAgreement::Candidate,
            LookupAgreement::Absent,
            LookupAgreement::Unknown,
        ];
        for d in 0..lookup.lookup_len {
            for &ag in &agreements {
                lr[d * 4 + ag as usize] = lookup_acceptance.rate(false, lookup.match_len, d, ag);
            }
        }
    }
    let budget_acc = verify_budget.acceptance(false);
    let mut bi = [0.0f32; 8];
    bi[..DRAFT_SLOTS].copy_from_slice(&budget_acc.intercepts);

    let lookup_tokens_arr = MxArray::from_int32(&lt, &[MAX_TREE_NODES as i64])?;
    let lookup_rates_arr = MxArray::from_float32(&lr, &[MAX_TREE_NODES as i64, 4])?;
    let budget_intercepts_arr = MxArray::from_float32(&bi, &[8])?;
    crate::kernels::dflash2::select_tree_cuda(
        candidates,
        scores,
        &lookup_tokens_arr,
        &lookup_rates_arr,
        &budget_intercepts_arr,
        anchor,
        lookup.lookup_len as i32,
        node_budget as i32,
        lookup.eff_max_depth as i32,
        budget_acc.slope,
    )
}

/// Source of per-node [`LookupAgreement`] classifications for
/// [`finalize_tree_greedy_walk`].
pub(crate) enum TreeLookupSource<'a> {
    /// Host-built tree: compute agreements from the draft lattice.
    Lattice(&'a DraftLattice),
    /// GPU-selected tree: reuse agreements unpacked from `node_meta`.
    Agreements(&'a [LookupAgreement]),
}

/// Run [`VerifyTree::greedy_walk`] on `target_argmax` (length `tree.len()`),
/// update `verify_budget` and `lookup_acceptance` online estimators, and return
/// `(walk, lookup_accepted)`.
pub(crate) fn finalize_tree_greedy_walk(
    tree: &VerifyTree,
    target_argmax: &[u32],
    lookup: &TreeLookupQuery,
    source: TreeLookupSource<'_>,
    verify_budget: &mut VerifyBudget,
    lookup_acceptance: &mut LookupAcceptance,
) -> (TreeWalkResult, usize) {
    let walk = tree.greedy_walk(target_argmax);

    let labels = tree.budget_labels(&walk.emitted_tokens);
    verify_budget.observe(false, &labels);

    let mut lookup_accepted = 0usize;
    if lookup.lookup_len > 0 {
        let proposals = &lookup.proposals[..lookup.lookup_len];
        let outcome = match source {
            TreeLookupSource::Lattice(lattice) => {
                tree.lookup_outcome(proposals, &walk.emitted_tokens, lattice)
            }
            TreeLookupSource::Agreements(agreements) => {
                tree.lookup_outcome_with_agreements(proposals, &walk.emitted_tokens, agreements)
            }
        };
        lookup_acceptance.observe(false, lookup.match_len, &outcome.verified, outcome.accepted);
        lookup_accepted = outcome.accepted;
    }

    (walk, lookup_accepted)
}
