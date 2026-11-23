use std::cmp::Ordering;
use std::collections::BinaryHeap;

use napi::bindgen_prelude::*;

use crate::array::MxArray;

use super::budget::{BudgetAcceptance, BudgetLabel, VerifyBudget, logit_f64};
use super::lattice::DraftLattice;
use super::sam::{LookupAcceptance, LookupAgreement, LookupIndex, LookupOutcome};
use super::{DRAFT_SLOTS, MAX_LOOKUP_CHAIN, MAX_TREE_NODES};

// ---------------------------------------------------------------------------
// Verification Tree & Device Views
// ---------------------------------------------------------------------------

/// Device-resident tensors uploaded once per tree verification cycle and
/// consumed by our Wave B CUDA kernels:
/// - `depths_i32`: `[T]` Int32 -> `mlx_qk_norm_rope_kv_store_tree_cuda`
/// - `mask_u32`: `[T]` Uint32 -> `mlx_sdpa_tc_tree_few_cuda` (`mask_mode = 3`)
/// - `parent_i32`: `[T]` Int32 -> `mlx_gdn_tree_step_cuda`
/// - `conv_taps_i32`: `[T, 4]` Int32 -> `mlx_gdn_prepare_tree_cuda`
pub(crate) struct TreeDeviceViews {
    pub depths_i32: MxArray,
    pub mask_u32: MxArray,
    pub parent_i32: MxArray,
    pub conv_taps_i32: MxArray,
    pub tree_len: usize,
}

/// Result of greedy target verification over a [`VerifyTree`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TreeWalkResult {
    /// Accepted tree node indices along the verified path, always starting with
    /// root `0`. Length `K` (`1 <= K <= tree.len()`): `K` rows of target KV,
    /// GDN state, and draft taps are committed.
    pub accepted_nodes: Vec<usize>,
    /// Verified output tokens emitted after each node in `accepted_nodes`
    /// (`emitted_tokens.len() == accepted_nodes.len()`).
    pub emitted_tokens: Vec<u32>,
    /// Next cycle's anchor token (`*emitted_tokens.last().unwrap()`).
    pub next_anchor: u32,
    /// `true` iff `accepted_nodes == [0, 1, ..., K - 1]` (zero-copy KV/GDN commit).
    pub is_contiguous_prefix: bool,
}

/// DFS-preorder speculative verification tree (`T in 1..=MAX_TREE_NODES`).
///
/// Invariants:
/// - `tokens[0]` is the anchor token at depth `0`.
/// - `parent[0] == -1`; for every `i > 0`, `0 <= parent[i] < i`.
/// - `depth[0] == 0`; for every `i > 0`, `depth[i] == depth[parent[i]] + 1`.
/// - `ancestors_mask[i]` has bit `j` set iff `j == i` or `j` is an ancestor of `i`.
/// - `conv_taps[i]` holds `[tap0, tap1, tap2, tap3]` matching
///   `mlx_gdn_prepare_tree_cuda` (`tap3 = i`, negative values index
///   `history[3 + src]` where `-1 -> history[2]`, `-2 -> history[1]`, `-3 -> history[0]`).
/// - Because DFS visits highest-score children first, the highest-probability
///   path from root is always a contiguous prefix `[0, 1, 2, ...]`.
#[derive(Clone, Debug)]
pub(crate) struct VerifyTree {
    pub tokens: Vec<u32>,
    pub parent: Vec<i32>,
    pub depth: Vec<i32>,
    pub ancestors_mask: Vec<u32>,
    pub conv_taps: Vec<[i32; 4]>,
    /// Draft conditional probability `q` for each node (`1.0` for root, `-1.0`
    /// for lookup-only nodes never joined by a draft transition).
    pub draft_probs: Vec<f32>,
    /// Calibrated joint acceptance estimate for each node (`1.0` for root).
    pub estimates: Vec<f32>,
    /// Probability cap applied during tree expansion (`1.0` for root / lookup-only).
    pub caps: Vec<f32>,
    /// Whether each node lies on the lookup chain prefix from root.
    pub on_chain: Vec<bool>,
}

pub(super) fn compute_conv_taps(parent: &[i32]) -> Vec<[i32; 4]> {
    let step_back = |idx: i32| -> i32 {
        if idx >= 0 {
            parent[idx as usize]
        } else {
            idx - 1
        }
    };
    let mut conv_taps = Vec::with_capacity(parent.len());
    for i in 0..parent.len() {
        let t3 = i as i32;
        let t2 = step_back(t3);
        let t1 = step_back(t2);
        let t0 = step_back(t1);
        conv_taps.push([t0, t1, t2, t3]);
    }
    conv_taps
}

impl VerifyTree {
    /// Construct a 1D chain tree (`parent = [-1, 0, 1, ..., T-2]`) from `tokens`.
    pub(crate) fn from_chain(tokens: &[u32]) -> Result<Self> {
        let len = tokens.len();
        if len == 0 || len > MAX_TREE_NODES {
            return Err(Error::from_reason(format!(
                "VerifyTree chain length {len} outside 1..={MAX_TREE_NODES}"
            )));
        }
        let mut parent = Vec::with_capacity(len);
        let mut depth = Vec::with_capacity(len);
        let mut ancestors_mask = Vec::with_capacity(len);
        for i in 0..len {
            parent.push(if i == 0 { -1 } else { (i - 1) as i32 });
            depth.push(i as i32);
            ancestors_mask.push(((1u64 << (i + 1)) - 1) as u32);
        }
        let conv_taps = compute_conv_taps(&parent);
        Ok(Self {
            tokens: tokens.to_vec(),
            parent,
            depth,
            ancestors_mask,
            conv_taps,
            draft_probs: vec![1.0; len],
            estimates: vec![1.0; len],
            caps: vec![1.0; len],
            on_chain: vec![false; len],
        })
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.tokens.len()
    }

    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Returns `true` iff this tree is a pure 1D chain (`parent[i] == i - 1`
    /// for all `i > 0`).
    #[inline]
    pub(crate) fn is_chain(&self) -> bool {
        (1..self.parent.len()).all(|i| self.parent[i] == (i as i32) - 1)
    }

    /// Materialize the four tree metadata tensors required by the Wave B CUDA
    /// kernels (`depths_i32`, `mask_u32`, `parent_i32`, `conv_taps_i32`).
    pub(crate) fn to_device_views(&self) -> Result<TreeDeviceViews> {
        let t = self.len();
        if t == 0 || t > MAX_TREE_NODES {
            return Err(Error::from_reason(format!(
                "VerifyTree length {t} outside 1..={MAX_TREE_NODES}"
            )));
        }
        let t_i64 = t as i64;
        let depths_i32 = MxArray::from_int32(&self.depth, &[t_i64])?;
        let mask_u32 = MxArray::from_uint32(&self.ancestors_mask, &[t_i64])?;
        let parent_i32 = MxArray::from_int32(&self.parent, &[t_i64])?;
        let mut flat_taps = Vec::with_capacity(t * 4);
        for taps in &self.conv_taps {
            flat_taps.extend_from_slice(taps);
        }
        let conv_taps_i32 = MxArray::from_int32(&flat_taps, &[t_i64, 4])?;
        Ok(TreeDeviceViews {
            depths_i32,
            mask_u32,
            parent_i32,
            conv_taps_i32,
            tree_len: t,
        })
    }

    /// Compute dilated depthwise conv taps (`k=4, dilation=3`, Qwen4 Layer-1 PLE)
    /// for every node in this tree.
    ///
    /// For node `i`: `t3 = i`, `t2 = back3(t3)`, `t1 = back3(t2)`, `t0 = back3(t1)`
    /// where `step_back(x) = if x >= 0 { parent[x] } else { x - 1 }` applied 3 times.
    /// Negative values `-k` read `history[9 - k]` (`< -9` reads `0.0`).
    /// On a linear chain, `ple_conv_taps[i] == [i - 9, i - 6, i - 3, i]`.
    pub(crate) fn ple_conv_taps(&self) -> Vec<[i32; 4]> {
        let step_back = |idx: i32| -> i32 {
            if idx >= 0 {
                self.parent[idx as usize]
            } else {
                idx - 1
            }
        };
        let back3 = |idx: i32| -> i32 { step_back(step_back(step_back(idx))) };
        let mut out = Vec::with_capacity(self.parent.len());
        for i in 0..self.parent.len() {
            let t3 = i as i32;
            let t2 = back3(t3);
            let t1 = back3(t2);
            let t0 = back3(t1);
            out.push([t0, t1, t2, t3]);
        }
        out
    }

    /// Upload [`Self::ple_conv_taps`] as an `[T, 4]` `Int32` tensor for
    /// `kernels::qwen4::ple_gate_conv_tree_cuda`.
    pub(crate) fn ple_conv_taps_i32(&self) -> Result<MxArray> {
        let taps = self.ple_conv_taps();
        let t = taps.len();
        let mut flat = Vec::with_capacity(t * 4);
        for row in &taps {
            flat.extend_from_slice(row);
        }
        MxArray::from_int32(&flat, &[t as i64, 4])
    }

    /// Walk the tree greedily given `target_argmax[i]` (the target model's
    /// argmax next-token prediction at tree node `i`).
    pub(crate) fn greedy_walk(&self, target_argmax: &[u32]) -> TreeWalkResult {
        assert_eq!(
            target_argmax.len(),
            self.tokens.len(),
            "greedy_walk target_argmax length ({}) != tree length ({})",
            target_argmax.len(),
            self.tokens.len()
        );
        assert!(!self.tokens.is_empty(), "VerifyTree must contain root node");

        let mut cur = 0usize;
        let mut accepted_nodes = Vec::with_capacity(self.tokens.len());
        let mut emitted_tokens = Vec::with_capacity(self.tokens.len());
        accepted_nodes.push(0);

        let mut next_anchor;
        loop {
            let next_tok = target_argmax[cur];
            next_anchor = next_tok;
            emitted_tokens.push(next_tok);
            let child = ((cur + 1)..self.tokens.len())
                .find(|&c| self.parent[c] == cur as i32 && self.tokens[c] == next_tok);
            match child {
                Some(c) => {
                    cur = c;
                    accepted_nodes.push(c);
                }
                None => break,
            }
        }

        let is_contiguous_prefix = accepted_nodes
            .iter()
            .enumerate()
            .all(|(k, &node)| node == k);

        TreeWalkResult {
            accepted_nodes,
            emitted_tokens,
            next_anchor,
            is_contiguous_prefix,
        }
    }

    /// Extract [`VerifyBudget`] training labels for every draft node whose
    /// parent lay on the accepted path (Splash `VerifyBudget::treeLabels`).
    pub(crate) fn budget_labels(&self, emitted_tokens: &[u32]) -> Vec<BudgetLabel> {
        let mut place = vec![None::<usize>; self.len()];
        if self.is_empty() {
            return Vec::new();
        }
        place[0] = Some(0);
        let mut labels = Vec::new();
        for row in 1..self.len() {
            let parent = self.parent[row] as usize;
            let Some(at) = place[parent] else {
                continue;
            };
            if at >= emitted_tokens.len() || at >= DRAFT_SLOTS {
                continue;
            }
            let accepted = self.tokens[row] == emitted_tokens[at];
            if accepted {
                place[row] = Some(at + 1);
            }
            let prob = self.draft_probs[row];
            let cap = self.caps[row];
            if prob.is_finite()
                && (0.0..=1.0).contains(&prob)
                && cap.is_finite()
                && (0.0..=1.0).contains(&cap)
            {
                labels.push(BudgetLabel {
                    position: at,
                    feature: logit_f64(f64::from(prob)) as f32,
                    accepted,
                    cap,
                });
            }
        }
        labels
    }

    /// Unpack a [`VerifyTree`] and per-node [`LookupAgreement`] vector from
    /// [`crate::kernels::dflash2::select_tree_cuda`]'s compact device readback
    /// (`tree_tokens [M]`, `parent [M]`, `node_meta [M * 4]`).
    ///
    /// Returns `(tree, agreements, is_chain)` truncated to `actual_count <= M`.
    pub(crate) fn from_gpu_readback(
        raw_tokens: &[u32],
        raw_parent: &[i32],
        raw_meta: &[f32],
    ) -> Result<(Self, Vec<LookupAgreement>, bool)> {
        let cap_len = raw_tokens.len();
        if cap_len == 0
            || cap_len > MAX_TREE_NODES
            || raw_parent.len() != cap_len
            || raw_meta.len() != cap_len * 4
        {
            return Err(Error::from_reason(format!(
                "VerifyTree::from_gpu_readback shape mismatch: tokens={}, parent={}, meta={}",
                raw_tokens.len(),
                raw_parent.len(),
                raw_meta.len()
            )));
        }
        let flags0 = raw_meta[3] as u32;
        let actual_count = ((flags0 >> 4) & 0x1F) as usize;
        if actual_count == 0 || actual_count > cap_len {
            return Err(Error::from_reason(format!(
                "VerifyTree::from_gpu_readback invalid actual_count={actual_count} (cap={cap_len})"
            )));
        }
        let is_chain = (flags0 & 0x200) != 0;
        let tokens = raw_tokens[..actual_count].to_vec();
        let parent = raw_parent[..actual_count].to_vec();
        if parent[0] != -1 {
            return Err(Error::from_reason(format!(
                "VerifyTree::from_gpu_readback expected parent[0] == -1, got {}",
                parent[0]
            )));
        }
        let mut depth = Vec::with_capacity(actual_count);
        let mut ancestors_mask = Vec::with_capacity(actual_count);
        let mut draft_probs = Vec::with_capacity(actual_count);
        let mut estimates = Vec::with_capacity(actual_count);
        let mut caps = Vec::with_capacity(actual_count);
        let mut on_chain = Vec::with_capacity(actual_count);
        let mut agreements = Vec::with_capacity(actual_count);

        for (i, &p) in parent.iter().enumerate().take(actual_count) {
            if i == 0 {
                depth.push(0);
                ancestors_mask.push(1u32);
            } else {
                if p < 0 || (p as usize) >= i {
                    return Err(Error::from_reason(format!(
                        "VerifyTree::from_gpu_readback invalid parent[{i}] = {p}"
                    )));
                }
                let pu = p as usize;
                depth.push(depth[pu] + 1);
                ancestors_mask.push(ancestors_mask[pu] | (1u32 << i));
            }
            let base = i * 4;
            let flags = raw_meta[base + 3] as u32;
            let ag = match (flags >> 1) & 0x3 {
                0 => LookupAgreement::Top,
                1 => LookupAgreement::Candidate,
                2 => LookupAgreement::Absent,
                _ => LookupAgreement::Unknown,
            };
            draft_probs.push(raw_meta[base]);
            estimates.push(raw_meta[base + 1]);
            caps.push(raw_meta[base + 2]);
            on_chain.push((flags & 0x001) != 0);
            agreements.push(ag);
        }

        let conv_taps = compute_conv_taps(&parent);
        let tree = Self {
            tokens,
            parent,
            depth,
            ancestors_mask,
            conv_taps,
            draft_probs,
            estimates,
            caps,
            on_chain,
        };
        debug_assert_eq!(is_chain, tree.is_chain());
        Ok((tree, agreements, is_chain))
    }

    /// Extract [`LookupOutcome`] for the verified prefix of `lookup_proposals`
    /// against the target's `emitted_tokens` (Splash `lookupOutcome`).
    pub(crate) fn lookup_outcome(
        &self,
        lookup_proposals: &[u32],
        emitted_tokens: &[u32],
        lattice: &DraftLattice,
    ) -> LookupOutcome {
        let mut outcome = LookupOutcome::default();
        if self.is_empty() {
            return outcome;
        }
        let mut row = 0usize;
        let limit = lookup_proposals.len().min(emitted_tokens.len());
        for depth in 0..limit {
            let target_tok = lookup_proposals[depth];
            let child = ((row + 1)..self.len())
                .find(|&at| self.parent[at] == row as i32 && self.tokens[at] == target_tok);
            let Some(c) = child else {
                break;
            };
            outcome
                .verified
                .push(lattice.lookup_agreement(target_tok, depth));
            if emitted_tokens[depth] != target_tok {
                break;
            }
            outcome.accepted += 1;
            row = c;
        }
        outcome
    }

    /// Extract [`LookupOutcome`] using per-node [`LookupAgreement`] values
    /// already packed into GPU `node_meta` (avoids reading `candidates` back).
    pub(crate) fn lookup_outcome_with_agreements(
        &self,
        lookup_proposals: &[u32],
        emitted_tokens: &[u32],
        agreements: &[LookupAgreement],
    ) -> LookupOutcome {
        let mut outcome = LookupOutcome::default();
        if self.is_empty() || agreements.len() < self.len() {
            return outcome;
        }
        let mut row = 0usize;
        let limit = lookup_proposals.len().min(emitted_tokens.len());
        for depth in 0..limit {
            let target_tok = lookup_proposals[depth];
            let child = ((row + 1)..self.len())
                .find(|&at| self.parent[at] == row as i32 && self.tokens[at] == target_tok);
            let Some(c) = child else {
                break;
            };
            outcome.verified.push(agreements[c]);
            if emitted_tokens[depth] != target_tok {
                break;
            }
            outcome.accepted += 1;
            row = c;
        }
        outcome
    }
}

// ---------------------------------------------------------------------------
// Best-First Tree Builder + DFS Preorder Renumbering
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct HeapEntry {
    score: f32,
    order: u32,
    from_chain: bool,
    parent: u16,
    position: u16,
    cand_index: u16,
    cap: f32,
    q: f32,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.score.to_bits() == other.score.to_bits() && self.order == other.order
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap by score; ties broken by earliest push order.
        self.score
            .total_cmp(&other.score)
            .then_with(|| other.order.cmp(&self.order))
    }
}

struct GrownNode {
    token: u32,
    parent: i32,
    on_chain: bool,
    q: f32,
    estimate: f32,
    cap: f32,
}

#[allow(clippy::too_many_arguments)]
fn expand_draft_children(
    lattice: &DraftLattice,
    acceptance: &BudgetAcceptance,
    heap: &mut BinaryHeap<HeapEntry>,
    push_order: &mut u32,
    parent_row: usize,
    position: usize,
    predecessor: usize,
    parent_score: f32,
    chain_token: Option<u32>,
    chain_rate: f32,
) {
    if position >= lattice.slots || lattice.top_k == 0 {
        return;
    }
    let candidates = lattice.likeliest_slice(position, predecessor);
    let mut bound = parent_score;
    let mut reserved = chain_rate.clamp(0.0, 1.0);
    let mut left = (1.0 - reserved).max(0.0);

    for &cand_u8 in candidates {
        let cand = cand_u8 as usize;
        let q = lattice.prob(position, predecessor, cand);
        if !q.is_finite() || q <= 0.0 {
            break;
        }
        let tok = lattice.token(position, cand);
        let joins_chain = chain_token == Some(tok);
        let cap = if joins_chain {
            (left + reserved).min(1.0)
        } else {
            left
        };
        let own = acceptance.estimate(q, position).min(cap);
        if joins_chain {
            left = (cap - own.max(reserved)).max(0.0);
            reserved = 0.0;
        } else {
            left = (left - own).max(0.0);
        }
        bound = bound.min(parent_score * own);
        if !bound.is_finite() || bound <= 0.0 {
            continue;
        }
        let order = *push_order;
        *push_order += 1;
        heap.push(HeapEntry {
            score: bound,
            order,
            from_chain: false,
            parent: parent_row as u16,
            position: position as u16,
            cand_index: cand as u16,
            cap,
            q,
        });
    }
}

/// Linear per-cycle verify cost model `Cost(T) = base_ms + slope_ms * T` used
/// for MoE-cost-aware adaptive tree sizing (Splash `rising=true` step costs).
///
/// On 512-expert top-10 MoE (`qwen4_exp`), unique active experts grow almost
/// linearly with tree size `T`, so blindly verifying 16 nodes wastes bandwidth
/// on low-confidence prose while under-verifying `M=4` starves high-confidence
/// coding and prompt-lookup stretches. Selecting
/// `T* = argmax_{T in min_nodes..=max_nodes} E[T] / Cost(T)` over the best-first
/// prefix before DFS renumbering dynamically picks the throughput-optimal tree.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TreeCostModel {
    pub base_ms: f64,
    pub slope_ms: f64,
    pub min_nodes: usize,
    /// When true, restrict candidate `T*` to [`Self::CANONICAL_BUCKETS`] so
    /// CUDA NVRTC compiles at most 5 tree widths (`4, 6, 8, 12, 16`) instead
    /// of every integer `2..=16`.
    pub bucketed: bool,
}

impl TreeCostModel {
    /// Canonical verify widths aligned with CUDA `sdpa_tc_few` (`NUM_WARPS`)
    /// and `qwen4_moe_few` (`ALLOW_HI`) specializations.
    pub(crate) const CANONICAL_BUCKETS: [usize; 6] = [2, 4, 6, 8, 12, 16];

    /// Calibrated from live 125B-A6B (`qwen4_exp`) 512-expert top-10 MoE
    /// `M in {4, 6, 8, 12, 16}` sweeps at 9400 MHz (`Cost(T) ≈ 48.0 + 3.5 * T`).
    pub(crate) const QWEN4_MOE_DEFAULT: Self = Self {
        base_ms: 48.0,
        slope_ms: 3.5,
        min_nodes: 4,
        bucketed: true,
    };

    #[inline]
    pub(crate) fn cost_ms(&self, nodes: usize) -> f64 {
        (self.base_ms + self.slope_ms * (nodes as f64)).max(1.0e-3)
    }

    #[inline]
    fn is_allowed_size(&self, t: usize) -> bool {
        !self.bucketed || Self::CANONICAL_BUCKETS.contains(&t)
    }

    /// Pick `T* in min_nodes.min(n)..=n` maximizing expected tokens per ms
    /// `E[T] / Cost(T)` where `estimates[0..n]` are in best-first insertion order.
    pub(crate) fn choose_tree_size(&self, estimates: &[f32]) -> usize {
        let n = estimates.len();
        if n <= 1 {
            return n;
        }
        let lo = self.min_nodes.clamp(1, n);
        let mut expected = 0.0f64;
        let mut best_t = 0usize;
        let mut best_rate = f64::NEG_INFINITY;
        for (idx, &e) in estimates.iter().enumerate() {
            expected += f64::from(e.max(0.0));
            let t = idx + 1;
            if t < lo || !self.is_allowed_size(t) {
                continue;
            }
            let rate = expected / self.cost_ms(t);
            if rate > best_rate {
                best_rate = rate;
                best_t = t;
            }
        }
        if best_t == 0 {
            // Tail of turn where `n` falls below the smallest bucket >= `lo`.
            n
        } else {
            best_t
        }
    }
}

/// Build a DFS-preorder speculative verification tree (up to `max_nodes <= 16`)
/// combining DFlash2's candidate `lattice` and suffix-automaton
/// `lookup_proposals`.
///
/// - `anchor`: node 0 token ID at depth 0.
/// - `lattice`: DFlash2 candidate lattice (may be [`DraftLattice::empty`]).
/// - `lookup_proposals`: suffix-automaton continuation tokens (up to `MAX_LOOKUP_CHAIN`).
/// - `lookup_match_len`: match length from [`LookupIndex::match_info`] (`< 3` disables lookup).
/// - `max_nodes`: node budget clamped to `1..=MAX_TREE_NODES`.
/// - `max_depth`: optional maximum tree depth (e.g. `remaining_tokens - 1`);
///   defaults to `max_nodes - 1`.
/// - `samples`: `false` for greedy decoding, `true` for sampled decoding.
/// - `budget`: logistic calibration for draft probabilities.
/// - `lookup_acc`: prior + online acceptance rates for lookup proposals.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_verify_tree(
    anchor: u32,
    lattice: &DraftLattice,
    lookup_proposals: &[u32],
    lookup_match_len: usize,
    max_nodes: usize,
    max_depth: Option<usize>,
    samples: bool,
    budget: &VerifyBudget,
    lookup_acc: &LookupAcceptance,
) -> VerifyTree {
    build_verify_tree_inner(
        anchor,
        lattice,
        lookup_proposals,
        lookup_match_len,
        max_nodes,
        max_depth,
        samples,
        budget,
        lookup_acc,
        None,
    )
}

/// Like [`build_verify_tree`], but truncates the best-first candidate tree to
/// the throughput-optimal size `T* = argmax E[T] / Cost(T)` under `cost_model`
/// BEFORE DFS preorder renumbering.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_verify_tree_cost_aware(
    anchor: u32,
    lattice: &DraftLattice,
    lookup_proposals: &[u32],
    lookup_match_len: usize,
    max_nodes: usize,
    max_depth: Option<usize>,
    samples: bool,
    budget: &VerifyBudget,
    lookup_acc: &LookupAcceptance,
    cost_model: &TreeCostModel,
) -> VerifyTree {
    build_verify_tree_inner(
        anchor,
        lattice,
        lookup_proposals,
        lookup_match_len,
        max_nodes,
        max_depth,
        samples,
        budget,
        lookup_acc,
        Some(cost_model),
    )
}

#[allow(clippy::too_many_arguments)]
fn build_verify_tree_inner(
    anchor: u32,
    lattice: &DraftLattice,
    lookup_proposals: &[u32],
    lookup_match_len: usize,
    max_nodes: usize,
    max_depth: Option<usize>,
    samples: bool,
    budget: &VerifyBudget,
    lookup_acc: &LookupAcceptance,
    cost_model: Option<&TreeCostModel>,
) -> VerifyTree {
    let most = max_nodes.clamp(1, MAX_TREE_NODES);
    let deepest = max_depth
        .unwrap_or(most - 1)
        .min(MAX_LOOKUP_CHAIN)
        .min(most - 1);
    let chained = if lookup_match_len >= LookupIndex::MIN_MATCH {
        lookup_proposals.len().min(MAX_LOOKUP_CHAIN).min(deepest)
    } else {
        0
    };

    let acceptance = budget.acceptance(samples);

    // Precompute conditional acceptance rates for the lookup chain.
    // Keep full Splash priors while the chain agrees with DFlash2 (`Top`) or
    // comes from a long verbatim match (`base_absent >= 0.40`, i.e. match >= 32).
    // On shorter matches, cap non-`Top` reservations by the depth-0 prior and
    // stop extending past an `Absent` disagreement (or into `Unknown` depths
    // when the end of the draft window disagreed) so false/terminating matches
    // on long prompts do not starve DFlash2's own branches.
    let mut chain_rates = [0.0f32; MAX_LOOKUP_CHAIN];
    let base_cand = lookup_acc.rate(samples, lookup_match_len, 0, LookupAgreement::Candidate);
    let base_absent = lookup_acc.rate(samples, lookup_match_len, 0, LookupAgreement::Absent);
    let mut diverged = false;
    let mut last_draft_top = true;
    for d in 0..chained {
        let agreement = lattice.lookup_agreement(lookup_proposals[d], d);
        let mut rate = lookup_acc.rate(samples, lookup_match_len, d, agreement);
        if lattice.slots > 0 {
            if d < lattice.slots {
                last_draft_top = agreement == LookupAgreement::Top;
            }
            if (diverged
                && matches!(
                    agreement,
                    LookupAgreement::Absent | LookupAgreement::Unknown
                ))
                || (agreement == LookupAgreement::Unknown && !last_draft_top && base_absent < 0.40)
            {
                rate = 0.0;
            } else if agreement == LookupAgreement::Candidate {
                rate = rate.min(base_cand);
            } else if agreement == LookupAgreement::Absent {
                rate = rate.min(base_absent);
                if base_absent < 0.40 {
                    diverged = true;
                }
            }
        }
        chain_rates[d] = rate;
    }

    let mut grown = Vec::with_capacity(most);
    grown.push(GrownNode {
        token: anchor,
        parent: -1,
        on_chain: true,
        q: 1.0,
        estimate: 1.0,
        cap: 1.0,
    });

    let mut heap = BinaryHeap::new();
    let mut push_order = 0u32;

    if deepest > 0 && most > 1 {
        let root_chain_tok = (chained > 0).then(|| lookup_proposals[0]);
        let root_chain_rate = if chained > 0 { chain_rates[0] } else { 0.0 };
        expand_draft_children(
            lattice,
            acceptance,
            &mut heap,
            &mut push_order,
            0,
            0,
            0,
            1.0,
            root_chain_tok,
            root_chain_rate,
        );
        if chained > 0 && root_chain_rate > 0.0 && root_chain_rate.is_finite() {
            let order = push_order;
            push_order += 1;
            heap.push(HeapEntry {
                score: root_chain_rate,
                order,
                from_chain: true,
                parent: 0,
                position: 0,
                cand_index: 0,
                cap: 1.0,
                q: -1.0,
            });
        }
    }

    while grown.len() < most {
        let Some(entry) = heap.pop() else {
            break;
        };
        let parent = entry.parent as usize;
        let position = entry.position as usize;
        let cand_index = entry.cand_index as usize;
        let token = if entry.from_chain {
            lookup_proposals[position]
        } else {
            lattice.token(position, cand_index)
        };

        // Deduplicate identical `(parent, token)` paths.
        let existing = (1..grown.len())
            .find(|&other| grown[other].parent == parent as i32 && grown[other].token == token);

        let row = match existing {
            None => {
                let idx = grown.len();
                let on_chain = grown[parent].on_chain
                    && position < chained
                    && token == lookup_proposals[position];
                grown.push(GrownNode {
                    token,
                    parent: parent as i32,
                    on_chain,
                    q: if entry.from_chain { -1.0 } else { entry.q },
                    estimate: entry.score,
                    cap: if entry.from_chain { 1.0 } else { entry.cap },
                });
                idx
            }
            Some(idx) => {
                // If a draft step joins a node first created by the lookup chain,
                // attach its `(q, cap)` so `VerifyBudget` calibration learns from it.
                if !entry.from_chain && grown[idx].q < 0.0 {
                    grown[idx].q = entry.q;
                    grown[idx].cap = entry.cap;
                }
                idx
            }
        };

        let next_pos = position + 1;
        if next_pos >= deepest {
            continue;
        }
        let has_chain_child = grown[row].on_chain && next_pos < chained;
        if !entry.from_chain && next_pos < lattice.slots {
            expand_draft_children(
                lattice,
                acceptance,
                &mut heap,
                &mut push_order,
                row,
                next_pos,
                cand_index,
                entry.score,
                has_chain_child.then(|| lookup_proposals[next_pos]),
                if has_chain_child {
                    chain_rates[next_pos]
                } else {
                    0.0
                },
            );
        } else if entry.from_chain && next_pos < chained {
            let next_score = entry.score * chain_rates[next_pos];
            if next_score > 0.0 && next_score.is_finite() {
                let order = push_order;
                push_order += 1;
                heap.push(HeapEntry {
                    score: next_score,
                    order,
                    from_chain: true,
                    parent: row as u16,
                    position: next_pos as u16,
                    cand_index: 0,
                    cap: 1.0,
                    q: -1.0,
                });
            }
        }
    }

    // MoE-cost-aware truncation BEFORE DFS preorder renumbering: `grown` is in
    // strictly best-first order and every node's parent appears earlier, so any
    // prefix `grown[..best_t]` is a valid rooted subtree with highest `E[T]`.
    if let Some(cost) = cost_model {
        let best_estimates: Vec<f32> = grown.iter().map(|n| n.estimate).collect();
        let best_t = cost.choose_tree_size(&best_estimates);
        grown.truncate(best_t);
    }

    // DFS preorder renumbering with highest-probability child visited first.
    // This guarantees `0 <= parent[i] < i` and places the highest-probability
    // path from root at contiguous indices `[0, 1, 2, ...]`.
    let n = grown.len();
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (idx, node) in grown.iter().enumerate().skip(1) {
        children[node.parent as usize].push(idx);
    }
    for kids in &mut children {
        kids.sort_by(|&a, &b| {
            grown[b]
                .estimate
                .total_cmp(&grown[a].estimate)
                .then_with(|| a.cmp(&b))
        });
    }

    let mut order = Vec::with_capacity(n);
    let mut placed = vec![0usize; n];
    let mut stack = Vec::with_capacity(n);
    stack.push(0usize);
    while let Some(row) = stack.pop() {
        placed[row] = order.len();
        order.push(row);
        // Push lowest-score child first so highest-score child pops first.
        for &child in children[row].iter().rev() {
            stack.push(child);
        }
    }

    let mut tokens = Vec::with_capacity(n);
    let mut parent = Vec::with_capacity(n);
    let mut depth = Vec::with_capacity(n);
    let mut ancestors_mask = Vec::with_capacity(n);
    let mut draft_probs = Vec::with_capacity(n);
    let mut estimates = Vec::with_capacity(n);
    let mut caps = Vec::with_capacity(n);
    let mut on_chain = Vec::with_capacity(n);

    for (new_idx, &old_idx) in order.iter().enumerate() {
        let node = &grown[old_idx];
        let new_parent = if new_idx == 0 {
            -1
        } else {
            placed[node.parent as usize] as i32
        };
        let new_depth = if new_idx == 0 {
            0
        } else {
            depth[new_parent as usize] + 1
        };
        let new_mask = if new_idx == 0 {
            1u32
        } else {
            ancestors_mask[new_parent as usize] | (1u32 << new_idx)
        };
        tokens.push(node.token);
        parent.push(new_parent);
        depth.push(new_depth);
        ancestors_mask.push(new_mask);
        draft_probs.push(node.q);
        estimates.push(node.estimate);
        caps.push(node.cap);
        on_chain.push(node.on_chain);
    }

    let conv_taps = compute_conv_taps(&parent);

    VerifyTree {
        tokens,
        parent,
        depth,
        ancestors_mask,
        conv_taps,
        draft_probs,
        estimates,
        caps,
        on_chain,
    }
}
