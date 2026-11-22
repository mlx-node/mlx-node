use crate::array::MxArray;

use super::tree::compute_conv_taps;
use super::*;

fn assert_tree_invariants(tree: &VerifyTree) {
    let n = tree.len();
    assert!((1..=MAX_TREE_NODES).contains(&n));
    assert_eq!(tree.parent.len(), n);
    assert_eq!(tree.depth.len(), n);
    assert_eq!(tree.ancestors_mask.len(), n);
    assert_eq!(tree.conv_taps.len(), n);
    assert_eq!(tree.draft_probs.len(), n);
    assert_eq!(tree.estimates.len(), n);
    assert_eq!(tree.caps.len(), n);
    assert_eq!(tree.on_chain.len(), n);

    assert_eq!(tree.parent[0], -1);
    assert_eq!(tree.depth[0], 0);
    assert_eq!(tree.ancestors_mask[0], 1);
    assert_eq!(tree.conv_taps[0], [-3, -2, -1, 0]);

    for i in 1..n {
        let p = tree.parent[i];
        assert!(
            p >= 0 && (p as usize) < i,
            "parent[{i}] = {p} violates preorder"
        );
        let pu = p as usize;
        assert_eq!(tree.depth[i], tree.depth[pu] + 1);
        assert_eq!(
            tree.ancestors_mask[i],
            tree.ancestors_mask[pu] | (1u32 << i)
        );

        // Recompute conv taps independently by walking ancestors.
        let mut expected_taps = [0i32; 4];
        let mut cur = i as i32;
        for slot in (0..4).rev() {
            expected_taps[slot] = cur;
            cur = if cur >= 0 {
                tree.parent[cur as usize]
            } else {
                cur - 1
            };
        }
        assert_eq!(
            tree.conv_taps[i], expected_taps,
            "conv_taps mismatch at {i}"
        );

        // No duplicate `(parent, token)` siblings.
        for j in (i + 1)..n {
            assert!(
                !(tree.parent[j] == p && tree.tokens[j] == tree.tokens[i]),
                "duplicate sibling token {} under parent {p} at nodes {i} and {j}",
                tree.tokens[i]
            );
        }
    }
}

// --- LookupIndex tests ---

#[test]
fn lookup_index_ignores_matches_shorter_than_min_match() {
    let mut idx = LookupIndex::new();
    idx.append(&[10, 20, 30, 10, 20]);
    assert_eq!(idx.match_info(), LookupMatchInfo::default());
    assert!(idx.proposals(MAX_LOOKUP_CHAIN).is_empty());
}

#[test]
fn lookup_index_exact_repeat_and_continuation() {
    let mut idx = LookupIndex::new();
    // [1, 2, 3] followed by [99, 100, 101], then repeated [1, 2, 3].
    idx.append(&[1, 2, 3, 99, 100, 101, 1, 2, 3]);
    let info = idx.match_info();
    assert_eq!(info.match_len, 3);
    assert_eq!(info.match_end, 2);
    assert_eq!(info.available_after_match, 6);
    assert_eq!(idx.proposals(4), vec![99, 100, 101, 1]);
    assert_eq!(idx.proposals(2), vec![99, 100]);
}

#[test]
fn lookup_index_prefers_longest_suffix_then_most_recent() {
    let mut idx = LookupIndex::new();
    // Occurrence 1 (len 3 match): [10, 20, 30] -> 777
    // Occurrence 2 (len 4 match): [5, 10, 20, 30] -> 888
    // Occurrence 3 (len 3 match, more recent): [10, 20, 30] -> 999
    // Current suffix: [5, 10, 20, 30] -> must pick Occurrence 2 (len 4 > len 3).
    idx.append(&[
        10, 20, 30, 777, // ends at 2
        5, 10, 20, 30, 888, // ends at 7
        10, 20, 30, 999, // ends at 11
        5, 10, 20, 30,
    ]);
    let info = idx.match_info();
    assert_eq!(info.match_len, 4);
    assert_eq!(info.match_end, 7);
    assert_eq!(idx.proposals(1), vec![888]);

    // Now test equal-length preference: most recent wins.
    idx.reset();
    idx.append(&[
        1, 2, 3, 111, // first occurrence
        1, 2, 3, 222, // second (more recent) occurrence
        1, 2, 3,
    ]);
    let info = idx.match_info();
    assert_eq!(info.match_len, 3);
    assert_eq!(info.match_end, 6);
    assert_eq!(idx.proposals(1), vec![222]);
}

#[test]
fn lookup_index_prompt_plus_generated_continuation_and_reset() {
    let mut idx = LookupIndex::new();
    // Prompt has a repeating phrase.
    idx.append(&[42, 43, 44, 45, 46, 47, 50, 51, 42, 43]);
    assert_eq!(idx.match_info().match_len, 0);
    // Generated token completes the 3-token suffix [42, 43, 44].
    idx.push(44);
    assert_eq!(idx.match_info().match_len, 3);
    assert_eq!(idx.proposals(3), vec![45, 46, 47]);
    // Generated token extends match to 4 tokens [42, 43, 44, 45].
    idx.push(45);
    assert_eq!(idx.match_info().match_len, 4);
    assert_eq!(idx.proposals(2), vec![46, 47]);

    idx.reset();
    assert!(idx.is_empty());
    assert_eq!(idx.match_info(), LookupMatchInfo::default());
    assert!(idx.proposals(MAX_LOOKUP_CHAIN).is_empty());
}

#[test]
fn lookup_index_caps_repeated_run_at_max_match() {
    let mut idx = LookupIndex::new();
    let run = vec![7u32; 150];
    idx.append(&run);
    let info = idx.match_info();
    assert_eq!(info.match_len, LookupIndex::MAX_MATCH);
    assert_eq!(info.match_end, 148);
    assert_eq!(idx.proposals(MAX_LOOKUP_CHAIN), vec![7]);
}

// --- LookupAcceptance & VerifyBudget tests ---

#[test]
fn lookup_acceptance_buckets_and_online_observe() {
    assert_eq!(LookupAcceptance::length_bucket(3), 0);
    assert_eq!(LookupAcceptance::length_bucket(4), 1);
    assert_eq!(LookupAcceptance::length_bucket(7), 1);
    assert_eq!(LookupAcceptance::length_bucket(8), 2);
    assert_eq!(LookupAcceptance::length_bucket(15), 2);
    assert_eq!(LookupAcceptance::length_bucket(16), 3);
    assert_eq!(LookupAcceptance::length_bucket(31), 3);
    assert_eq!(LookupAcceptance::length_bucket(32), 4);
    assert_eq!(LookupAcceptance::length_bucket(63), 4);
    assert_eq!(LookupAcceptance::length_bucket(64), 5);

    assert_eq!(LookupAcceptance::depth_bucket(0), 0);
    assert_eq!(LookupAcceptance::depth_bucket(1), 1);
    assert_eq!(LookupAcceptance::depth_bucket(2), 2);
    assert_eq!(LookupAcceptance::depth_bucket(3), 2);
    assert_eq!(LookupAcceptance::depth_bucket(4), 3);
    assert_eq!(LookupAcceptance::depth_bucket(7), 3);
    assert_eq!(LookupAcceptance::depth_bucket(8), 4);
    assert_eq!(LookupAcceptance::depth_bucket(15), 4);
    assert_eq!(LookupAcceptance::depth_bucket(16), 5);

    let mut acc = LookupAcceptance::new();
    let before = acc.rate(false, 8, 0, LookupAgreement::Absent);
    assert!((before - 0.163).abs() < 1e-4);
    // Observe 10 consecutive accepts at (len=8, depth=0, Absent) -> rate rises.
    for _ in 0..10 {
        acc.observe(false, 8, &[LookupAgreement::Absent], 1);
    }
    let after = acc.rate(false, 8, 0, LookupAgreement::Absent);
    assert!(
        after > before + 0.3,
        "expected rate rise: {before} -> {after}"
    );
    // Sampled lane stayed untouched at prior.
    assert!((acc.rate(true, 8, 0, LookupAgreement::Absent) - 0.163).abs() < 1e-4);
}

#[test]
fn verify_budget_prior_and_refit() {
    let mut budget = VerifyBudget::new();
    let p_hi = budget.acceptance(false).estimate(0.95, 0);
    let p_lo = budget.acceptance(false).estimate(0.10, 0);
    assert!(p_hi > 0.8 && p_lo < 0.2);

    // Observing all-accepted labels shifts intercept upward on refit.
    let label = BudgetLabel {
        position: 2,
        feature: 0.0,
        accepted: true,
        cap: 1.0,
    };
    let before = budget.acceptance(false).intercepts[2];
    for _ in 0..VerifyBudget::REFIT_CYCLES {
        budget.observe(false, &[label]);
    }
    let after = budget.acceptance(false).intercepts[2];
    assert!(
        after > before,
        "intercept should move up: {before} -> {after}"
    );
}

// --- VerifyTree & build_verify_tree tests ---

#[test]
fn chain_tree_invariants_and_conv_taps() {
    let tokens: Vec<u32> = (100..108).collect();
    let tree = VerifyTree::from_chain(&tokens).unwrap();
    assert_tree_invariants(&tree);
    assert!(tree.is_chain());
    assert_eq!(tree.conv_taps[0], [-3, -2, -1, 0]);
    assert_eq!(tree.conv_taps[1], [-2, -1, 0, 1]);
    assert_eq!(tree.conv_taps[2], [-1, 0, 1, 2]);
    assert_eq!(tree.conv_taps[3], [0, 1, 2, 3]);
    assert_eq!(tree.conv_taps[7], [4, 5, 6, 7]);
    assert_eq!(tree.ancestors_mask[0], 0b0000_0001);
    assert_eq!(tree.ancestors_mask[3], 0b0000_1111);
    assert_eq!(tree.ancestors_mask[7], 0b1111_1111);
}

#[test]
fn build_tree_pure_draft_branching_and_contiguous_top_path() {
    let slots = DRAFT_SLOTS;
    let top_k = 4;
    let mut candidates = Vec::with_capacity(slots * top_k);
    let mut slot_probs = Vec::with_capacity(slots * top_k);
    for s in 0..slots {
        let base = (s as u32 + 1) * 100;
        // Put highest-prob token at index 2 (not 0) to test sorting/DFS ordering.
        candidates.extend_from_slice(&[base, base + 1, base + 2, base + 3]);
        slot_probs.extend_from_slice(&[0.25, 0.15, 0.55, 0.05]);
    }
    let lattice = DraftLattice::from_slot_probs(&candidates, &slot_probs, slots, top_k);
    let budget = VerifyBudget::new();
    let lookup_acc = LookupAcceptance::new();

    let tree = build_verify_tree(
        999,
        &lattice,
        &[],
        0,
        MAX_TREE_NODES,
        None,
        false,
        &budget,
        &lookup_acc,
    );
    assert_tree_invariants(&tree);
    assert_eq!(tree.len(), MAX_TREE_NODES);
    assert!(
        !tree.is_chain(),
        "branching draft tree should not be a 1D chain"
    );
    assert_eq!(tree.tokens[0], 999);

    // Highest-probability path from root must be a contiguous prefix [0, 1, 2, ...].
    assert_eq!(tree.tokens[1], 102);
    assert_eq!(tree.parent[1], 0);
    assert_eq!(tree.tokens[2], 202);
    assert_eq!(tree.parent[2], 1);

    // Verify branching node conv_taps: find a second child of root 0.
    let second_root_child = (2..tree.len())
        .find(|&i| tree.parent[i] == 0)
        .expect("root should branch with 0.55 vs 0.25 split");
    assert_eq!(tree.depth[second_root_child], 1);
    assert_eq!(
        tree.ancestors_mask[second_root_child],
        1u32 | (1u32 << second_root_child)
    );
    assert_eq!(
        tree.conv_taps[second_root_child],
        [-2, -1, 0, second_root_child as i32]
    );
}

#[test]
fn build_tree_pure_lookup_chain_depth_15() {
    let lookup: Vec<u32> = (1..=20).map(|x| x * 10).collect();
    let budget = VerifyBudget::new();
    let lookup_acc = LookupAcceptance::new();

    let tree = build_verify_tree(
        5,
        &DraftLattice::empty(),
        &lookup,
        16,
        MAX_TREE_NODES,
        None,
        false,
        &budget,
        &lookup_acc,
    );
    assert_tree_invariants(&tree);
    assert_eq!(tree.len(), MAX_TREE_NODES);
    assert!(tree.is_chain());
    assert_eq!(tree.depth[15], 15);
    assert_eq!(tree.tokens[0], 5);
    assert_eq!(&tree.tokens[1..], &lookup[..MAX_LOOKUP_CHAIN]);
    assert!(tree.on_chain.iter().all(|&c| c));
}

#[test]
fn build_tree_merges_and_deduplicates_draft_and_lookup() {
    let slots = DRAFT_SLOTS;
    let top_k = 4;
    let mut candidates = Vec::with_capacity(slots * top_k);
    let mut slot_probs = Vec::with_capacity(slots * top_k);
    for s in 0..slots {
        let base = (s as u32 + 1) * 10;
        candidates.extend_from_slice(&[base, base + 1, base + 2, base + 3]);
        slot_probs.extend_from_slice(&[0.60, 0.25, 0.10, 0.05]);
    }
    let lattice = DraftLattice::from_slot_probs(&candidates, &slot_probs, slots, top_k);
    // Lookup agrees with draft top-1 at depths 0..3 ([10, 20, 30]), then continues deeper.
    let lookup = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
    let budget = VerifyBudget::new();
    let lookup_acc = LookupAcceptance::new();

    let tree = build_verify_tree(
        1,
        &lattice,
        &lookup,
        32,
        MAX_TREE_NODES,
        None,
        false,
        &budget,
        &lookup_acc,
    );
    assert_tree_invariants(&tree);
    assert_eq!(tree.len(), MAX_TREE_NODES);

    // Node 1 is token 10 under root 0, present exactly once and joined by draft (`q >= 0`).
    let root_children_10: Vec<usize> = (1..tree.len())
        .filter(|&i| tree.parent[i] == 0 && tree.tokens[i] == 10)
        .collect();
    assert_eq!(root_children_10.len(), 1);
    let n1 = root_children_10[0];
    assert!(tree.on_chain[n1]);
    assert!(
        tree.draft_probs[n1] > 0.0,
        "merged lookup+draft node should carry draft probability for calibration"
    );
    // Lookup extends past draft depth 7 (depth 8+ present when match_len=32).
    assert!(tree.depth.iter().copied().max().unwrap() > DRAFT_SLOTS as i32);
}

#[test]
fn build_tree_respects_max_nodes_caps() {
    let slots = DRAFT_SLOTS;
    let top_k = 4;
    let candidates: Vec<u32> = (0..(slots * top_k) as u32).map(|x| x + 100).collect();
    let slot_probs: Vec<f32> = (0..slots).flat_map(|_| [0.55, 0.25, 0.15, 0.05]).collect();
    let lattice = DraftLattice::from_slot_probs(&candidates, &slot_probs, slots, top_k);
    let lookup: Vec<u32> = (1000..1015).collect();
    let budget = VerifyBudget::new();
    let lookup_acc = LookupAcceptance::new();

    for cap in [1usize, 4, 8, 16] {
        let tree = build_verify_tree(
            7,
            &lattice,
            &lookup,
            16,
            cap,
            None,
            false,
            &budget,
            &lookup_acc,
        );
        assert_tree_invariants(&tree);
        assert_eq!(tree.len(), cap);
    }
}

// --- greedy_walk tests ---

#[test]
fn greedy_walk_full_chain_partial_and_root_reject() {
    let tree = VerifyTree::from_chain(&[10, 20, 30, 40]).unwrap();

    // Full chain accept + bonus token 50.
    let full = tree.greedy_walk(&[20, 30, 40, 50]);
    assert_eq!(full.accepted_nodes, vec![0, 1, 2, 3]);
    assert_eq!(full.emitted_tokens, vec![20, 30, 40, 50]);
    assert_eq!(full.next_anchor, 50);
    assert!(full.is_contiguous_prefix);

    // Partial accept (accepts 20, rejects 30 -> emits 99).
    let partial = tree.greedy_walk(&[20, 99, 40, 50]);
    assert_eq!(partial.accepted_nodes, vec![0, 1]);
    assert_eq!(partial.emitted_tokens, vec![20, 99]);
    assert_eq!(partial.next_anchor, 99);
    assert!(partial.is_contiguous_prefix);

    // Immediate rejection at root -> 1 token emitted, only root kept.
    let reject = tree.greedy_walk(&[77, 30, 40, 50]);
    assert_eq!(reject.accepted_nodes, vec![0]);
    assert_eq!(reject.emitted_tokens, vec![77]);
    assert_eq!(reject.next_anchor, 77);
    assert!(reject.is_contiguous_prefix);
}

#[test]
fn greedy_walk_branch_switch_is_non_contiguous() {
    // Construct a branching DFS-preorder tree:
    //   0 (tok 10)
    //   ├── 1 (tok 20) -> 2 (tok 21)
    //   └── 3 (tok 30) -> 4 (tok 31)
    let parent = vec![-1, 0, 1, 0, 3];
    let depth = vec![0, 1, 2, 1, 2];
    let ancestors_mask = vec![0b00001, 0b00011, 0b00111, 0b01001, 0b11001];
    let conv_taps = compute_conv_taps(&parent);
    let tree = VerifyTree {
        tokens: vec![10, 20, 21, 30, 31],
        parent,
        depth,
        ancestors_mask,
        conv_taps,
        draft_probs: vec![1.0, 0.6, 0.5, 0.3, 0.4],
        estimates: vec![1.0, 0.6, 0.3, 0.3, 0.12],
        caps: vec![1.0; 5],
        on_chain: vec![true, false, false, false, false],
    };
    assert_tree_invariants(&tree);
    assert!(!tree.is_chain());
    assert_eq!(tree.conv_taps[3], [-2, -1, 0, 3]);
    assert_eq!(tree.conv_taps[4], [-1, 0, 3, 4]);

    // Target chooses second branch: 0 -> 3 (tok 30) -> 4 (tok 31) -> bonus 99.
    let walk = tree.greedy_walk(&[30, 999, 999, 31, 99]);
    assert_eq!(walk.accepted_nodes, vec![0, 3, 4]);
    assert_eq!(walk.emitted_tokens, vec![30, 31, 99]);
    assert_eq!(walk.next_anchor, 99);
    assert!(!walk.is_contiguous_prefix);

    // Verify calibration label extraction along the branch walk:
    // Under root (at=0, emitted=30): node 1 (20) rejected, node 3 (30) accepted.
    // Under node 3 (at=1, emitted=31): node 4 (31) accepted.
    // Node 2 (under rejected node 1) is off-path and produces no label.
    let labels = tree.budget_labels(&walk.emitted_tokens);
    assert_eq!(labels.len(), 3);
    assert_eq!((labels[0].position, labels[0].accepted), (0, false));
    assert_eq!((labels[1].position, labels[1].accepted), (0, true));
    assert_eq!((labels[2].position, labels[2].accepted), (1, true));
}

#[test]
fn draft_lattice_from_selector_scores_matches_markov_transitions() {
    // 2 slots, top_k = 2.
    // Slot 0 candidates: [10, 20]. After anchor (pred=0): cand 1 (20) wins strongly.
    // Slot 1 candidates: [30, 40]. After pred=0 (10): cand 0 (30) wins;
    //                              after pred=1 (20): cand 1 (40) wins.
    let candidates = vec![10, 20, 30, 40];
    let scores = vec![
        // pos 0, pred 0: [10 -> 0.0, 20 -> 4.0]
        0.0, 4.0, // pos 0, pred 1 (unused)
        0.0, 0.0, // pos 1, pred 0 (after 10): [30 -> 5.0, 40 -> 0.0]
        5.0, 0.0, // pos 1, pred 1 (after 20): [30 -> 0.0, 40 -> 5.0]
        0.0, 5.0,
    ];
    let lattice = DraftLattice::from_selector_scores(&candidates, &scores, 2, 2);
    assert_eq!(lattice.lookup_agreement(20, 0), LookupAgreement::Top);
    assert_eq!(lattice.lookup_agreement(10, 0), LookupAgreement::Candidate);
    assert_eq!(lattice.lookup_agreement(99, 0), LookupAgreement::Absent);
    assert_eq!(lattice.lookup_agreement(40, 1), LookupAgreement::Top);
    assert_eq!(lattice.lookup_agreement(40, 2), LookupAgreement::Unknown);

    let budget = VerifyBudget::new();
    let lookup_acc = LookupAcceptance::new();
    let tree = build_verify_tree(1, &lattice, &[], 0, 8, None, false, &budget, &lookup_acc);
    assert_tree_invariants(&tree);
    // Contiguous top path must follow 1 -> 20 -> 40!
    assert_eq!(tree.tokens[0], 1);
    assert_eq!(tree.tokens[1], 20);
    assert_eq!(tree.parent[1], 0);
    assert_eq!(tree.tokens[2], 40);
    assert_eq!(tree.parent[2], 1);
}

#[test]
fn to_device_views_matches_wave_b_cuda_contract() {
    use crate::array::DType;

    let parent = vec![-1, 0, 1, 0, 3];
    let depth = vec![0, 1, 2, 1, 2];
    let ancestors_mask = vec![0b00001u32, 0b00011, 0b00111, 0b01001, 0b11001];
    let conv_taps = compute_conv_taps(&parent);
    let tree = VerifyTree {
        tokens: vec![10, 20, 21, 30, 31],
        parent: parent.clone(),
        depth: depth.clone(),
        ancestors_mask: ancestors_mask.clone(),
        conv_taps: conv_taps.clone(),
        draft_probs: vec![1.0; 5],
        estimates: vec![1.0; 5],
        caps: vec![1.0; 5],
        on_chain: vec![false; 5],
    };

    let views = tree.to_device_views().unwrap();
    assert_eq!(views.tree_len, 5);
    assert_eq!(views.depths_i32.dtype().unwrap(), DType::Int32);
    assert_eq!(views.depths_i32.shape().unwrap().as_ref(), &[5]);
    assert_eq!(
        views.depths_i32.to_int32().unwrap().as_ref(),
        depth.as_slice()
    );

    assert_eq!(views.mask_u32.dtype().unwrap(), DType::Uint32);
    assert_eq!(views.mask_u32.shape().unwrap().as_ref(), &[5]);
    assert_eq!(
        views.mask_u32.to_uint32().unwrap().as_ref(),
        ancestors_mask.as_slice()
    );

    assert_eq!(views.parent_i32.dtype().unwrap(), DType::Int32);
    assert_eq!(views.parent_i32.shape().unwrap().as_ref(), &[5]);
    assert_eq!(
        views.parent_i32.to_int32().unwrap().as_ref(),
        parent.as_slice()
    );

    assert_eq!(views.conv_taps_i32.dtype().unwrap(), DType::Int32);
    assert_eq!(views.conv_taps_i32.shape().unwrap().as_ref(), &[5, 4]);
    let expected_flat: Vec<i32> = conv_taps.iter().flatten().copied().collect();
    assert_eq!(
        views.conv_taps_i32.to_int32().unwrap().as_ref(),
        expected_flat.as_slice()
    );
}

#[test]
fn lookup_outcome_and_max_depth_cap() {
    let lookup = vec![10, 20, 30, 40, 50];
    let budget = VerifyBudget::new();
    let lookup_acc = LookupAcceptance::new();

    // Cap depth at 2 even with max_nodes = 16 -> chain stops at depth 2 (3 nodes).
    let tree = build_verify_tree(
        1,
        &DraftLattice::empty(),
        &lookup,
        12,
        MAX_TREE_NODES,
        Some(2),
        false,
        &budget,
        &lookup_acc,
    );
    assert_tree_invariants(&tree);
    assert_eq!(tree.len(), 3);
    assert_eq!(tree.tokens, vec![1, 10, 20]);

    // Target accepts 10, rejects 20 (emits 99) -> 2 verified, 1 accepted.
    let outcome = tree.lookup_outcome(&lookup, &[10, 99], &DraftLattice::empty());
    assert_eq!(
        outcome.verified,
        vec![LookupAgreement::Unknown, LookupAgreement::Unknown]
    );
    assert_eq!(outcome.accepted, 1);
}

// --- GPU `select_tree_cuda` vs CPU `build_verify_tree` parity tests ---

fn assert_gpu_matches_cpu_tree(
    candidates: &[u32],
    scores: &[f32],
    slots: usize,
    anchor: u32,
    lookup_proposals: &[u32],
    lookup_match_len: usize,
    max_nodes: usize,
    max_depth: usize,
    budget: &VerifyBudget,
    lookup_acc: &LookupAcceptance,
) {
    if crate::accelerator::active_backend() != crate::accelerator::ActiveBackend::CudaGpu {
        return;
    }
    let lattice = DraftLattice::from_selector_scores(candidates, scores, slots, DRAFT_CANDIDATES);
    let cpu_tree = build_verify_tree(
        anchor,
        &lattice,
        lookup_proposals,
        lookup_match_len,
        max_nodes,
        Some(max_depth),
        false,
        budget,
        lookup_acc,
    );
    assert_tree_invariants(&cpu_tree);

    let cand_i32: Vec<i32> = candidates.iter().map(|&x| x as i32).collect();
    let cand_arr =
        MxArray::from_int32(&cand_i32, &[1, slots as i64, DRAFT_CANDIDATES as i64]).unwrap();
    let scores_arr = MxArray::from_float32(
        scores,
        &[
            slots as i64,
            DRAFT_CANDIDATES as i64,
            DRAFT_CANDIDATES as i64,
        ],
    )
    .unwrap();

    let lookup_len = if lookup_match_len >= LookupIndex::MIN_MATCH {
        lookup_proposals.len().min(MAX_LOOKUP_CHAIN)
    } else {
        0
    };
    let mut ltok = [0i32; 16];
    for (d, &tok) in lookup_proposals.iter().take(lookup_len).enumerate() {
        ltok[d] = tok as i32;
    }
    let ltok_arr = MxArray::from_int32(&ltok, &[16]).unwrap();

    let mut lrates = [0.0f32; 16 * 4];
    if lookup_len > 0 {
        let agreements = [
            LookupAgreement::Top,
            LookupAgreement::Candidate,
            LookupAgreement::Absent,
            LookupAgreement::Unknown,
        ];
        for d in 0..lookup_len {
            for (a_idx, &ag) in agreements.iter().enumerate() {
                lrates[d * 4 + a_idx] = lookup_acc.rate(false, lookup_match_len, d, ag);
            }
        }
    }
    let lrates_arr = MxArray::from_float32(&lrates, &[16, 4]).unwrap();

    let acc = budget.acceptance(false);
    let mut intercepts = [0.0f32; 8];
    intercepts[..DRAFT_SLOTS].copy_from_slice(&acc.intercepts);
    let intercepts_arr = MxArray::from_float32(&intercepts, &[8]).unwrap();

    let gpu = crate::kernels::dflash2::select_tree_cuda(
        &cand_arr,
        &scores_arr,
        &ltok_arr,
        &lrates_arr,
        &intercepts_arr,
        anchor,
        lookup_len as i32,
        max_nodes as i32,
        max_depth as i32,
        acc.slope,
    )
    .unwrap()
    .expect("select_tree_cuda must run on CUDA backend");

    MxArray::eval_arrays(&[
        &gpu.tree_tokens,
        &gpu.parent,
        &gpu.depths,
        &gpu.ancestors_mask,
        &gpu.conv_taps,
        &gpu.node_meta,
    ])
    .unwrap();

    let t = max_nodes;
    assert_eq!(gpu.tree_tokens.shape().unwrap().as_ref(), &[1, t as i64]);
    assert_eq!(gpu.parent.shape().unwrap().as_ref(), &[t as i64]);
    assert_eq!(gpu.depths.shape().unwrap().as_ref(), &[t as i64]);
    assert_eq!(gpu.ancestors_mask.shape().unwrap().as_ref(), &[t as i64]);
    assert_eq!(gpu.conv_taps.shape().unwrap().as_ref(), &[t as i64, 4]);
    assert_eq!(gpu.node_meta.shape().unwrap().as_ref(), &[t as i64, 4]);

    let g_tokens = gpu.tree_tokens.to_uint32().unwrap();
    let g_parent = gpu.parent.to_int32().unwrap();
    let g_depths = gpu.depths.to_int32().unwrap();
    let g_mask = gpu.ancestors_mask.to_uint32().unwrap();
    let g_taps = gpu.conv_taps.to_int32().unwrap();
    let g_meta = gpu.node_meta.to_float32().unwrap();

    let flags0 = g_meta[3] as u32;
    let actual_count = ((flags0 >> 4) & 0x1F) as usize;
    let gpu_is_chain = (flags0 & 0x200) != 0;
    assert_eq!(
        actual_count,
        cpu_tree.len(),
        "GPU actual_count ({actual_count}) != CPU tree.len() ({})",
        cpu_tree.len()
    );
    assert_eq!(gpu_is_chain, cpu_tree.is_chain());

    assert_eq!(&g_tokens[..actual_count], cpu_tree.tokens.as_slice());
    assert_eq!(&g_parent[..actual_count], cpu_tree.parent.as_slice());
    assert_eq!(&g_depths[..actual_count], cpu_tree.depth.as_slice());
    assert_eq!(&g_mask[..actual_count], cpu_tree.ancestors_mask.as_slice());

    for i in 0..actual_count {
        let taps = &g_taps[i * 4..(i + 1) * 4];
        assert_eq!(taps, cpu_tree.conv_taps[i].as_slice(), "conv_taps[{i}]");

        let q = g_meta[i * 4];
        let est = g_meta[i * 4 + 1];
        let cap = g_meta[i * 4 + 2];
        let flags = g_meta[i * 4 + 3] as u32;
        assert!(
            (q - cpu_tree.draft_probs[i]).abs() < 1e-5,
            "draft_prob[{i}]: gpu={q} cpu={}",
            cpu_tree.draft_probs[i]
        );
        assert!(
            (est - cpu_tree.estimates[i]).abs() < 1e-5,
            "estimate[{i}]: gpu={est} cpu={}",
            cpu_tree.estimates[i]
        );
        assert!(
            (cap - cpu_tree.caps[i]).abs() < 1e-5,
            "cap[{i}]: gpu={cap} cpu={}",
            cpu_tree.caps[i]
        );
        assert_eq!((flags & 1) != 0, cpu_tree.on_chain[i], "on_chain[{i}]");
        assert_ne!(flags & 8, 0, "is_valid[{i}] must be set");
        let expected_ag = if i == 0 {
            LookupAgreement::Unknown as u32
        } else {
            lattice.lookup_agreement(cpu_tree.tokens[i], (cpu_tree.depth[i] - 1) as usize) as u32
        };
        assert_eq!((flags >> 1) & 3, expected_ag, "agreement[{i}]");
    }

    // Verify padded trailing slots (if actual_count < max_nodes) satisfy Wave B invariants.
    for i in actual_count..t {
        assert_eq!(g_tokens[i], anchor);
        assert_eq!(g_parent[i], 0);
        assert_eq!(g_depths[i], 1);
        assert_eq!(g_mask[i], 1u32 | (1u32 << i));
        assert_eq!(&g_taps[i * 4..(i + 1) * 4], &[-2, -1, 0, i as i32]);
        let flags = g_meta[i * 4 + 3] as u32;
        assert_eq!(flags & 1, 0, "padded slot on_chain must be 0");
        assert_eq!(flags & 8, 0, "padded slot is_valid must be 0");
    }
}

fn make_markov_fixture(slots: usize) -> (Vec<u32>, Vec<f32>) {
    let k = DRAFT_CANDIDATES;
    let mut candidates = Vec::with_capacity(slots * k);
    for pos in 0..slots {
        for j in 0..k {
            // Ascending unary order: j = 15 is unary top-1, distinct tokens per slot.
            candidates.push(((pos as u32 + 1) * 1000) + j as u32);
        }
    }
    let mut scores = vec![-2.0f32; slots * k * k];
    for pos in 0..slots {
        let num_pred = if pos == 0 { 1 } else { k };
        for pred in 0..num_pred {
            let row = (pos * k + pred) * k;
            // Pick 4 strong candidates conditioned on predecessor so branching is non-trivial.
            let c0 = (15 - (pred % 5)) % k;
            let c1 = (c0 + 13) % k;
            let c2 = (c0 + 9) % k;
            let c3 = (c0 + 5) % k;
            for j in 0..k {
                scores[row + j] = -3.0 + (j as f32) * 0.05;
            }
            scores[row + c0] = 2.4 - (pos as f32) * 0.08;
            scores[row + c1] = 1.6 - (pos as f32) * 0.05;
            scores[row + c2] = 0.7;
            scores[row + c3] = 0.1;
        }
    }
    (candidates, scores)
}

#[test]
fn select_tree_cuda_parity_pure_draft_branching() {
    let (candidates, scores) = make_markov_fixture(DRAFT_SLOTS);
    let budget = VerifyBudget::new();
    let lookup_acc = LookupAcceptance::new();
    assert_gpu_matches_cpu_tree(
        &candidates,
        &scores,
        DRAFT_SLOTS,
        42,
        &[],
        0,
        16,
        15,
        &budget,
        &lookup_acc,
    );
}

#[test]
fn select_tree_cuda_parity_draft_plus_lookup_merge_dedup() {
    let (candidates, scores) = make_markov_fixture(DRAFT_SLOTS);
    let lattice =
        DraftLattice::from_selector_scores(&candidates, &scores, DRAFT_SLOTS, DRAFT_CANDIDATES);
    // Lookup agrees with draft top-1 at depths 0..2, hits a non-top candidate at depth 2,
    // is absent at depth 3, then extends past depth 7.
    let lookup = vec![
        lattice.token(0, lattice.top_candidate[0]),
        lattice.token(1, lattice.top_candidate[1]),
        lattice.token(2, (lattice.top_candidate[2] + 13) % DRAFT_CANDIDATES),
        999_001,
        lattice.token(4, lattice.top_candidate[4]),
        999_002,
        999_003,
        999_004,
    ];
    let budget = VerifyBudget::new();
    let lookup_acc = LookupAcceptance::new();
    assert_gpu_matches_cpu_tree(
        &candidates,
        &scores,
        DRAFT_SLOTS,
        77,
        &lookup,
        16,
        16,
        15,
        &budget,
        &lookup_acc,
    );
}

#[test]
fn select_tree_cuda_parity_long_lookup_chain_depth_15() {
    let (candidates, scores) = make_markov_fixture(DRAFT_SLOTS);
    let lattice =
        DraftLattice::from_selector_scores(&candidates, &scores, DRAFT_SLOTS, DRAFT_CANDIDATES);
    let mut lookup = Vec::with_capacity(15);
    for d in 0..15 {
        if d < DRAFT_SLOTS {
            lookup.push(lattice.token(d, lattice.top_candidate[d]));
        } else {
            lookup.push(50_000 + d as u32);
        }
    }
    let budget = VerifyBudget::new();
    let lookup_acc = LookupAcceptance::new();
    // High-confidence match_len = 64 extends the lookup chain all the way to depth 15.
    assert_gpu_matches_cpu_tree(
        &candidates,
        &scores,
        DRAFT_SLOTS,
        101,
        &lookup,
        64,
        16,
        15,
        &budget,
        &lookup_acc,
    );

    // Also test pure-lookup chain when draft scores are all -inf (no valid draft transitions).
    let neg_inf_scores = vec![f32::NEG_INFINITY; DRAFT_SLOTS * DRAFT_CANDIDATES * DRAFT_CANDIDATES];
    assert_gpu_matches_cpu_tree(
        &candidates,
        &neg_inf_scores,
        DRAFT_SLOTS,
        101,
        &lookup,
        32,
        16,
        15,
        &budget,
        &lookup_acc,
    );
}

#[test]
fn select_tree_cuda_parity_small_max_depth_and_padding() {
    let (candidates, scores) = make_markov_fixture(DRAFT_SLOTS);
    let budget = VerifyBudget::new();
    let lookup_acc = LookupAcceptance::new();

    // max_depth = 1, max_nodes = 8 -> only 1 + 4 = 5 real nodes, 3 padded tail slots!
    assert_gpu_matches_cpu_tree(
        &candidates,
        &scores,
        DRAFT_SLOTS,
        33,
        &[],
        0,
        8,
        1,
        &budget,
        &lookup_acc,
    );

    // max_depth = 2, max_nodes = 8 -> full 8 nodes.
    assert_gpu_matches_cpu_tree(
        &candidates,
        &scores,
        DRAFT_SLOTS,
        33,
        &[],
        0,
        8,
        2,
        &budget,
        &lookup_acc,
    );

    // max_depth = 1 with lookup (absent token -> 1 + 4 draft + 1 lookup = 6 real nodes, 2 padded).
    assert_gpu_matches_cpu_tree(
        &candidates,
        &scores,
        DRAFT_SLOTS,
        33,
        &[888_888, 888_889],
        8,
        8,
        1,
        &budget,
        &lookup_acc,
    );
}

#[test]
fn select_tree_cuda_flat_lookup_rates_shape() {
    if crate::accelerator::active_backend() != crate::accelerator::ActiveBackend::CudaGpu {
        return;
    }
    let (candidates, scores) = make_markov_fixture(DRAFT_SLOTS);
    let cand_i32: Vec<i32> = candidates.iter().map(|&x| x as i32).collect();
    let cand_arr =
        MxArray::from_int32(&cand_i32, &[1, DRAFT_SLOTS as i64, DRAFT_CANDIDATES as i64]).unwrap();
    let scores_arr = MxArray::from_float32(
        &scores,
        &[
            DRAFT_SLOTS as i64,
            DRAFT_CANDIDATES as i64,
            DRAFT_CANDIDATES as i64,
        ],
    )
    .unwrap();
    let mut ltok = [0i32; 16];
    for (i, slot) in ltok.iter_mut().enumerate().take(10) {
        *slot = 70_000 + i as i32;
    }
    let ltok_arr = MxArray::from_int32(&ltok, &[16]).unwrap();
    let flat_rates = [0.98f32; 16];
    let lrates_arr = MxArray::from_float32(&flat_rates, &[16]).unwrap();
    let acc = BudgetAcceptance::PRIOR;
    let mut intercepts = [0.0f32; 8];
    intercepts[..DRAFT_SLOTS].copy_from_slice(&acc.intercepts);
    let intercepts_arr = MxArray::from_float32(&intercepts, &[8]).unwrap();

    let gpu = crate::kernels::dflash2::select_tree_cuda(
        &cand_arr,
        &scores_arr,
        &ltok_arr,
        &lrates_arr,
        &intercepts_arr,
        55,
        10,
        16,
        15,
        acc.slope,
    )
    .unwrap()
    .expect("select_tree_cuda flat [16] rates should run on CUDA");
    MxArray::eval_arrays(&[&gpu.tree_tokens, &gpu.parent, &gpu.depths, &gpu.node_meta]).unwrap();
    let tokens = gpu.tree_tokens.to_uint32().unwrap();
    let depths = gpu.depths.to_int32().unwrap();
    assert_eq!(tokens[0], 55);
    assert!(
        depths.iter().copied().max().unwrap() >= 8,
        "high flat lookup rates should extend chain past draft depth 7"
    );
}

#[test]
fn from_gpu_readback_and_lookup_outcome_match_cpu() {
    if crate::accelerator::active_backend() != crate::accelerator::ActiveBackend::CudaGpu {
        return;
    }
    let (candidates, scores) = make_markov_fixture(DRAFT_SLOTS);
    let lattice =
        DraftLattice::from_selector_scores(&candidates, &scores, DRAFT_SLOTS, DRAFT_CANDIDATES);
    let lookup = vec![
        lattice.token(0, lattice.top_candidate[0]),
        lattice.token(1, (lattice.top_candidate[1] + 1) % DRAFT_CANDIDATES),
        999_001,
        999_002,
        999_003,
        999_004,
        999_005,
        999_006,
        999_007,
    ];
    let budget = VerifyBudget::new();
    let lookup_acc = LookupAcceptance::new();
    let cpu = build_verify_tree(
        42,
        &lattice,
        &lookup,
        16,
        16,
        Some(15),
        false,
        &budget,
        &lookup_acc,
    );

    let cand_i32: Vec<i32> = candidates.iter().map(|&x| x as i32).collect();
    let cand_arr =
        MxArray::from_int32(&cand_i32, &[1, DRAFT_SLOTS as i64, DRAFT_CANDIDATES as i64]).unwrap();
    let scores_arr = MxArray::from_float32(
        &scores,
        &[
            DRAFT_SLOTS as i64,
            DRAFT_CANDIDATES as i64,
            DRAFT_CANDIDATES as i64,
        ],
    )
    .unwrap();
    let mut ltok = [0i32; 16];
    for (i, &t) in lookup.iter().enumerate() {
        ltok[i] = t as i32;
    }
    let mut lrates = [0.0f32; 16 * 4];
    for d in 0..lookup.len() {
        for ag in [
            LookupAgreement::Top,
            LookupAgreement::Candidate,
            LookupAgreement::Absent,
            LookupAgreement::Unknown,
        ] {
            lrates[d * 4 + ag as usize] = lookup_acc.rate(false, 16, d, ag);
        }
    }
    let acc = budget.acceptance(false);
    let mut bi = [0.0f32; 8];
    bi[..DRAFT_SLOTS].copy_from_slice(&acc.intercepts);

    let gpu = crate::kernels::dflash2::select_tree_cuda(
        &cand_arr,
        &scores_arr,
        &MxArray::from_int32(&ltok, &[16]).unwrap(),
        &MxArray::from_float32(&lrates, &[16, 4]).unwrap(),
        &MxArray::from_float32(&bi, &[8]).unwrap(),
        42,
        lookup.len() as i32,
        16,
        15,
        acc.slope,
    )
    .unwrap()
    .expect("select_tree_cuda should run on CUDA");
    MxArray::eval_arrays(&[&gpu.tree_tokens, &gpu.parent, &gpu.node_meta]).unwrap();
    let (unpacked, agreements, is_chain) = VerifyTree::from_gpu_readback(
        &gpu.tree_tokens.to_uint32().unwrap(),
        &gpu.parent.to_int32().unwrap(),
        &gpu.node_meta.to_float32().unwrap(),
    )
    .unwrap();

    assert_eq!(unpacked.tokens, cpu.tokens);
    assert_eq!(unpacked.parent, cpu.parent);
    assert_eq!(unpacked.depth, cpu.depth);
    assert_eq!(unpacked.ancestors_mask, cpu.ancestors_mask);
    assert_eq!(unpacked.conv_taps, cpu.conv_taps);
    assert_eq!(unpacked.on_chain, cpu.on_chain);
    assert_eq!(is_chain, cpu.is_chain());

    let emitted = vec![lookup[0], lookup[1], 777_777];
    assert_eq!(
        unpacked.lookup_outcome_with_agreements(&lookup, &emitted, &agreements),
        cpu.lookup_outcome(&lookup, &emitted, &lattice)
    );
}

#[test]
fn diverging_short_lookup_does_not_starve_draft_and_matches_gpu() {
    let (candidates, scores) = make_markov_fixture(DRAFT_SLOTS);
    let lattice =
        DraftLattice::from_selector_scores(&candidates, &scores, DRAFT_SLOTS, DRAFT_CANDIDATES);
    // Simulate a false/terminating match (match_len=8): agrees at depths 0..3,
    // then disagrees (`Absent`) from depth 3 onward across 15 tokens.
    let mut lookup = Vec::with_capacity(15);
    for d in 0..15 {
        if d < 3 {
            lookup.push(lattice.token(d, lattice.top_candidate[d]));
        } else {
            lookup.push(800_000 + d as u32);
        }
    }
    let budget = VerifyBudget::new();
    let lookup_acc = LookupAcceptance::new();

    let tree = build_verify_tree(
        7,
        &lattice,
        &lookup,
        8,
        16,
        Some(15),
        false,
        &budget,
        &lookup_acc,
    );
    assert_tree_invariants(&tree);
    // Lookup must NOT extend past draft depth 7 after diverging at depth 3.
    assert!(
        tree.depth.iter().copied().max().unwrap() <= DRAFT_SLOTS as i32,
        "diverging short lookup should not waste depths > 7, got max depth {:?}",
        tree.depth.iter().copied().max()
    );
    // DFlash2's own depth-4 continuation after depth-3 top-1 must remain in the tree.
    let lkp_only = tree
        .draft_probs
        .iter()
        .skip(1)
        .filter(|&&q| q < 0.0)
        .count();
    assert!(
        lkp_only <= 2,
        "diverging short lookup should occupy at most 2 lookup-only slots, got {lkp_only}"
    );

    // Verify 100% GPU parity for this exact diverging case as well.
    assert_gpu_matches_cpu_tree(
        &candidates,
        &scores,
        DRAFT_SLOTS,
        7,
        &lookup,
        8,
        16,
        15,
        &budget,
        &lookup_acc,
    );
}

#[test]
fn ple_conv_taps_match_linear_chain_and_branching_ancestors() {
    // Linear chain: taps[i] == [i-9, i-6, i-3, i].
    let chain = VerifyTree::from_chain(&(0..12u32).collect::<Vec<_>>()).unwrap();
    let taps = chain.ple_conv_taps();
    assert_eq!(taps.len(), 12);
    for (i, &row) in taps.iter().enumerate() {
        let ii = i as i32;
        assert_eq!(row, [ii - 9, ii - 6, ii - 3, ii], "chain node {i}");
    }

    // Branching tree:
    // 0(-1) -> 1(0) -> 2(1) -> 3(2) -> 4(3)
    //       -> 5(0) -> 6(5) -> 7(6) -> 8(7)
    let parent = vec![-1i32, 0, 1, 2, 3, 0, 5, 6, 7];
    let tree = VerifyTree {
        tokens: (0..9u32).collect(),
        parent: parent.clone(),
        depth: vec![0, 1, 2, 3, 4, 1, 2, 3, 4],
        ancestors_mask: vec![1; 9],
        conv_taps: compute_conv_taps(&parent),
        draft_probs: vec![1.0; 9],
        estimates: vec![1.0; 9],
        caps: vec![1.0; 9],
        on_chain: vec![false; 9],
    };
    let btaps = tree.ple_conv_taps();
    assert_eq!(btaps[0], [-9, -6, -3, 0]);
    assert_eq!(btaps[3], [-6, -3, 0, 3]);
    assert_eq!(btaps[4], [-5, -2, 1, 4]);
    // Second branch node 8: path 0 -> 5 -> 6 -> 7 -> 8, so 3 back = 5, 6 back = -2, 9 back = -5.
    assert_eq!(btaps[5], [-8, -5, -2, 5]);
    assert_eq!(btaps[7], [-6, -3, 0, 7]);
    assert_eq!(btaps[8], [-5, -2, 5, 8]);
}

#[test]
fn cost_aware_tree_sizing_adapts_from_prose_to_high_confidence_lookup() {
    let cost = TreeCostModel::QWEN4_MOE_DEFAULT;
    let budget = VerifyBudget::new();
    let lookup_acc = LookupAcceptance::new();

    // 1. Low-confidence prose: flat softmax (high entropy) -> picks small T* <= 6.
    let slots = DRAFT_SLOTS;
    let top_k = DRAFT_CANDIDATES;
    let mut prose_cands = Vec::with_capacity(slots * top_k);
    let mut prose_scores = vec![0.0f32; slots * top_k * top_k];
    for pos in 0..slots {
        for c in 0..top_k {
            prose_cands.push((100 + pos * 32 + c) as u32);
        }
        let num_pred = if pos == 0 { 1 } else { top_k };
        for pred in 0..num_pred {
            let row = (pos * top_k + pred) * top_k;
            // Modest top-1 margin (~0.45 prob), flat tail.
            prose_scores[row] = 2.0;
            prose_scores[row + 1] = 1.0;
            for c in 2..top_k {
                prose_scores[row + c] = 0.0;
            }
        }
    }
    let prose_lattice =
        DraftLattice::from_selector_scores(&prose_cands, &prose_scores, slots, top_k);
    let prose_tree = build_verify_tree_cost_aware(
        1,
        &prose_lattice,
        &[],
        0,
        16,
        Some(15),
        false,
        &budget,
        &lookup_acc,
        &cost,
    );
    assert_tree_invariants(&prose_tree);
    assert!(
        (4..=6).contains(&prose_tree.len())
            && TreeCostModel::CANONICAL_BUCKETS.contains(&prose_tree.len()),
        "high-entropy prose should pick compact canonical bucket in 4..=6, got {}",
        prose_tree.len()
    );

    // 2. High-confidence coding + long verbatim SAM lookup match (match_len=40):
    //    picks wide T* >= 10 up to 16!
    let mut code_scores = vec![-10.0f32; slots * top_k * top_k];
    for pos in 0..slots {
        let num_pred = if pos == 0 { 1 } else { top_k };
        for pred in 0..num_pred {
            let row = (pos * top_k + pred) * top_k;
            code_scores[row] = 8.0; // ~0.999 probability on top-1 chain
        }
    }
    let code_lattice = DraftLattice::from_selector_scores(&prose_cands, &code_scores, slots, top_k);
    let mut lookup: Vec<u32> = (0..slots)
        .map(|d| code_lattice.token(d, code_lattice.top_candidate[d]))
        .collect();
    for extra in 0..8u32 {
        lookup.push(9000 + extra);
    }
    let code_tree = build_verify_tree_cost_aware(
        1,
        &code_lattice,
        &lookup,
        40,
        16,
        Some(15),
        false,
        &budget,
        &lookup_acc,
        &cost,
    );
    assert_tree_invariants(&code_tree);
    assert!(
        code_tree.len() >= 12 && TreeCostModel::CANONICAL_BUCKETS.contains(&code_tree.len()),
        "high-confidence coding + SAM lookup should expand to canonical bucket >= 12, got {}",
        code_tree.len()
    );
}
