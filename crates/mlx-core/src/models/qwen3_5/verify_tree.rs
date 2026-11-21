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

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use napi::bindgen_prelude::*;

use crate::array::MxArray;

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

// ---------------------------------------------------------------------------
// Suffix-Automaton Prompt Lookup (Splash `LookupProposals.{hpp,cpp}`)
// ---------------------------------------------------------------------------

/// Match metadata reported by [`LookupIndex::match_info`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct LookupMatchInfo {
    /// Length of the matched suffix (`0` if no match, else `MIN_MATCH..=MAX_MATCH`).
    pub match_len: usize,
    /// Zero-based token index of the earlier occurrence's last token (`0` if no match).
    pub match_end: usize,
    /// Number of recorded tokens strictly after `match_end` available as proposals.
    pub available_after_match: usize,
}

#[derive(Clone, Copy, Debug)]
struct SamState {
    length: u32,
    link: u32,
    edges: u32,
    recent: u32,
}

#[derive(Clone, Copy, Debug)]
struct SamEdge {
    from: u32,
    token: u32,
    to: u32,
    next: u32,
}

/// Online suffix automaton over a request's token history (prompt + emitted
/// tokens, ending at the current anchor).
///
/// Finds the most recent earlier occurrence of the longest suffix of length in
/// `[MIN_MATCH, MAX_MATCH]` in amortized O(1) per token (bounded walk of at
/// most `MAX_MATCH` states even on long runs of one repeated token).
#[derive(Clone, Debug)]
pub(crate) struct LookupIndex {
    tokens: Vec<u32>,
    states: Vec<SamState>,
    edges: Vec<SamEdge>,
    /// Open-addressed hash table keyed by `(from, token)`, storing `edge_idx + 1`
    /// (`0` = empty), kept at <= 50% load factor.
    slots: Vec<u32>,
    last: u32,
    capped: u32,
    match_len: usize,
    match_end: usize,
}

impl Default for LookupIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl LookupIndex {
    /// Minimum suffix length required to propose lookup continuations.
    pub(crate) const MIN_MATCH: usize = 3;
    /// Maximum tracked suffix length (bounds per-token walk on repeated tokens).
    pub(crate) const MAX_MATCH: usize = 64;

    const INITIAL_SLOTS: usize = 1024;
    const NONE: u32 = u32::MAX;

    pub(crate) fn new() -> Self {
        Self {
            tokens: Vec::new(),
            states: vec![SamState {
                length: 0,
                link: Self::NONE,
                edges: Self::NONE,
                recent: 0,
            }],
            edges: Vec::new(),
            slots: vec![0; Self::INITIAL_SLOTS],
            last: 0,
            capped: 0,
            match_len: 0,
            match_end: 0,
        }
    }

    /// Clear all indexed tokens and automaton states, keeping backing allocations
    /// bounded at the initial hash table size.
    pub(crate) fn reset(&mut self) {
        self.tokens.clear();
        self.states.clear();
        self.states.push(SamState {
            length: 0,
            link: Self::NONE,
            edges: Self::NONE,
            recent: 0,
        });
        self.edges.clear();
        if self.slots.len() > Self::INITIAL_SLOTS {
            self.slots.truncate(Self::INITIAL_SLOTS);
        }
        self.slots.fill(0);
        if self.slots.len() < Self::INITIAL_SLOTS {
            self.slots.resize(Self::INITIAL_SLOTS, 0);
        }
        self.last = 0;
        self.capped = 0;
        self.match_len = 0;
        self.match_end = 0;
    }

    pub(crate) fn len(&self) -> usize {
        self.tokens.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub(crate) fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    pub(crate) fn append(&mut self, tokens: &[u32]) {
        self.tokens.reserve(tokens.len());
        for &token in tokens {
            self.push(token);
        }
    }

    pub(crate) fn push(&mut self, token: u32) {
        let position = self.tokens.len() as u32;
        self.tokens.push(token);

        let current = self.states.len() as u32;
        let next_len = self.states[self.last as usize].length + 1;
        self.states.push(SamState {
            length: next_len,
            link: 0,
            edges: Self::NONE,
            recent: position,
        });

        let mut state = self.last;
        while state != Self::NONE && self.find(state, token) == Self::NONE {
            self.connect(state, token, current);
            state = self.states[state as usize].link;
        }

        if state != Self::NONE {
            let edge_idx = self.find(state, token);
            let next = self.edges[edge_idx as usize].to;
            if self.states[state as usize].length + 1 == self.states[next as usize].length {
                self.states[current as usize].link = next;
            } else {
                let clone = self.states.len() as u32;
                let clone_state = SamState {
                    length: self.states[state as usize].length + 1,
                    link: self.states[next as usize].link,
                    edges: Self::NONE,
                    recent: self.states[next as usize].recent,
                };
                self.states.push(clone_state);

                let mut edge = self.states[next as usize].edges;
                while edge != Self::NONE {
                    let e = self.edges[edge as usize];
                    self.connect(clone, e.token, e.to);
                    edge = e.next;
                }

                while state != Self::NONE {
                    let e_idx = self.find(state, token);
                    if e_idx == Self::NONE || self.edges[e_idx as usize].to != next {
                        break;
                    }
                    self.edges[e_idx as usize].to = clone;
                    state = self.states[state as usize].link;
                }

                self.states[next as usize].link = clone;
                self.states[current as usize].link = clone;
            }
        }
        self.last = current;

        // Track the state for the last `min(len, MAX_MATCH)` tokens.
        if (position as usize) < Self::MAX_MATCH {
            self.capped = current;
        } else {
            let mut held = self.capped;
            while self.states[held as usize].link != Self::NONE
                && self.states[self.states[held as usize].link as usize].length
                    >= Self::MAX_MATCH as u32
            {
                held = self.states[held as usize].link;
            }
            if self.states[held as usize].link != Self::NONE
                && self.states[self.states[held as usize].link as usize].length
                    == (Self::MAX_MATCH as u32 - 1)
            {
                held = self.states[held as usize].link;
            }
            let e_idx = self.find(held, token);
            debug_assert_ne!(e_idx, Self::NONE);
            self.capped = self.edges[e_idx as usize].to;
        }

        // Longest repeated suffix ending at `position`: `current`'s suffix link
        // (or `capped` when that suffix exceeds `MAX_MATCH`). Neither class's
        // `recent` has been updated to `position` yet.
        let repeated = self.states[current as usize].link;
        let rep_len = self.states[repeated as usize].length as usize;
        if rep_len < Self::MIN_MATCH {
            self.match_len = 0;
            self.match_end = 0;
        } else {
            let src = if rep_len > Self::MAX_MATCH {
                self.capped
            } else {
                repeated
            };
            self.match_len = rep_len.min(Self::MAX_MATCH);
            self.match_end = self.states[src as usize].recent as usize;
        }

        // Update `recent` on every suffix class in `[MIN_MATCH, MAX_MATCH]`.
        let mut s = self.capped;
        while s != Self::NONE && (self.states[s as usize].length as usize) >= Self::MIN_MATCH {
            self.states[s as usize].recent = position;
            s = self.states[s as usize].link;
        }
    }

    pub(crate) fn match_info(&self) -> LookupMatchInfo {
        if self.match_len == 0 {
            return LookupMatchInfo::default();
        }
        let begin = self.match_end.saturating_add(1);
        LookupMatchInfo {
            match_len: self.match_len,
            match_end: self.match_end,
            available_after_match: self.tokens.len().saturating_sub(begin),
        }
    }

    /// Borrow up to `max_count` predicted continuation tokens following the
    /// matched earlier occurrence. Empty when no match exists.
    pub(crate) fn proposal_slice(&self, max_count: usize) -> &[u32] {
        if self.match_len == 0 || max_count == 0 {
            return &[];
        }
        let begin = self.match_end.saturating_add(1);
        if begin >= self.tokens.len() {
            return &[];
        }
        let take = max_count.min(self.tokens.len() - begin);
        &self.tokens[begin..begin + take]
    }

    /// Up to `max_count` predicted continuation tokens following the current suffix.
    pub(crate) fn proposals(&self, max_count: usize) -> Vec<u32> {
        self.proposal_slice(max_count).to_vec()
    }

    #[inline]
    fn slot_hash(from: u32, token: u32) -> u64 {
        let mut key = (u64::from(from) << 32) | u64::from(token);
        key = (key ^ (key >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        key = (key ^ (key >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        key ^ (key >> 31)
    }

    #[inline]
    fn find(&self, state: u32, token: u32) -> u32 {
        let mask = self.slots.len() - 1;
        let mut slot = (Self::slot_hash(state, token) as usize) & mask;
        loop {
            let entry = self.slots[slot];
            if entry == 0 {
                return Self::NONE;
            }
            let edge = self.edges[(entry - 1) as usize];
            if edge.from == state && edge.token == token {
                return entry - 1;
            }
            slot = (slot + 1) & mask;
        }
    }

    #[inline]
    fn place(&mut self, edge_idx: u32) {
        let mask = self.slots.len() - 1;
        let edge = self.edges[edge_idx as usize];
        let mut slot = (Self::slot_hash(edge.from, edge.token) as usize) & mask;
        while self.slots[slot] != 0 {
            slot = (slot + 1) & mask;
        }
        self.slots[slot] = edge_idx + 1;
    }

    fn connect(&mut self, from: u32, token: u32, to: u32) {
        let edge_idx = self.edges.len() as u32;
        let prev_head = self.states[from as usize].edges;
        self.edges.push(SamEdge {
            from,
            token,
            to,
            next: prev_head,
        });
        self.states[from as usize].edges = edge_idx;
        if 2 * self.edges.len() > self.slots.len() {
            let new_len = self.slots.len() * 2;
            self.slots.clear();
            self.slots.resize(new_len, 0);
            for idx in 0..self.edges.len() as u32 {
                self.place(idx);
            }
        } else {
            self.place(edge_idx);
        }
    }
}

/// How a lookup proposal compares with the draft's top-K candidates at its depth.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum LookupAgreement {
    /// Matches the draft's most likely candidate at this depth.
    Top = 0,
    /// Appears among the draft's other top-K candidates at this depth.
    Candidate = 1,
    /// Absent from the draft's top-K candidates at this depth.
    Absent = 2,
    /// Draft has no proposal at this depth (e.g. depth >= `DRAFT_SLOTS`) or did not run.
    Unknown = 3,
}

/// Outcome of a lookup chain verified in one decode cycle, fed to
/// [`LookupAcceptance::observe`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct LookupOutcome {
    /// Agreement of each lookup chain token that was verified, in depth order.
    pub verified: Vec<LookupAgreement>,
    /// Number of verified lookup tokens accepted before the first rejection.
    pub accepted: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct AcceptanceCount {
    accepted: f64,
    labels: f64,
}

/// Conditional acceptance probability estimator for lookup-chain tokens,
/// combining Splash's trace-fitted prior table with exponentially decayed
/// online counts (greedy and sampled lanes tracked separately).
#[derive(Clone, Debug)]
pub(crate) struct LookupAcceptance {
    counts: [[[[AcceptanceCount; Self::AGREEMENTS]; Self::DEPTH_BUCKETS]; Self::LENGTH_BUCKETS]; 2],
}

impl Default for LookupAcceptance {
    fn default() -> Self {
        Self::new()
    }
}

impl LookupAcceptance {
    pub(crate) const LENGTH_BUCKETS: usize = 6;
    pub(crate) const DEPTH_BUCKETS: usize = 6;
    pub(crate) const AGREEMENTS: usize = 4;
    pub(crate) const PRIOR_LABELS: f64 = 4.0;
    pub(crate) const HALF_LIFE: f64 = 4096.0;

    /// Splash `LookupAcceptance::kPrior` (`[length_bucket][depth_bucket][agreement]`),
    /// fitted on 161,360 greedy decode labels across Qwen3.8-27B and Qwen3.6-35B-A3B.
    pub(crate) const PRIOR: [[[f32; Self::AGREEMENTS]; Self::DEPTH_BUCKETS]; Self::LENGTH_BUCKETS] = [
        // length = 3
        [
            [0.966, 0.214, 0.125, 0.648],
            [0.973, 0.307, 0.232, 0.709],
            [0.980, 0.516, 0.559, 0.787],
            [0.979, 0.652, 0.732, 0.830],
            [0.973, 0.569, 0.551, 0.856],
            [0.979, 0.634, 0.617, 0.911],
        ],
        // length = 4..=7
        [
            [0.975, 0.249, 0.153, 0.749],
            [0.980, 0.349, 0.276, 0.798],
            [0.985, 0.564, 0.614, 0.857],
            [0.984, 0.694, 0.775, 0.888],
            [0.980, 0.616, 0.607, 0.906],
            [0.984, 0.678, 0.670, 0.943],
        ],
        // length = 8..=15
        [
            [0.981, 0.307, 0.163, 0.812],
            [0.985, 0.419, 0.292, 0.851],
            [0.989, 0.634, 0.633, 0.897],
            [0.988, 0.753, 0.789, 0.920],
            [0.985, 0.682, 0.626, 0.933],
            [0.989, 0.738, 0.688, 0.960],
        ],
        // length = 16..=31
        [
            [0.987, 0.327, 0.220, 0.867],
            [0.990, 0.441, 0.374, 0.896],
            [0.992, 0.655, 0.714, 0.929],
            [0.992, 0.770, 0.844, 0.946],
            [0.990, 0.702, 0.708, 0.955],
            [0.992, 0.756, 0.761, 0.973],
        ],
        // length = 32..=63
        [
            [0.995, 0.648, 0.439, 0.935],
            [0.996, 0.750, 0.624, 0.950],
            [0.997, 0.878, 0.874, 0.967],
            [0.997, 0.927, 0.937, 0.975],
            [0.996, 0.899, 0.871, 0.979],
            [0.997, 0.921, 0.898, 0.988],
        ],
        // length >= 64
        [
            [0.999, 0.955, 0.859, 0.993],
            [1.000, 0.972, 0.928, 0.995],
            [1.000, 0.988, 0.982, 0.997],
            [1.000, 0.993, 0.992, 0.997],
            [1.000, 0.990, 0.981, 0.998],
            [1.000, 0.993, 0.986, 0.999],
        ],
    ];

    pub(crate) fn new() -> Self {
        Self {
            counts: [[[[AcceptanceCount::default(); Self::AGREEMENTS]; Self::DEPTH_BUCKETS];
                Self::LENGTH_BUCKETS]; 2],
        }
    }

    /// Bucket index for match length (`3`, `4..=7`, `8..=15`, `16..=31`, `32..=63`, `>=64`).
    #[inline]
    pub(crate) fn length_bucket(length: usize) -> usize {
        debug_assert!(length >= LookupIndex::MIN_MATCH);
        let bit_width = (usize::BITS - length.leading_zeros()) as usize;
        bit_width.saturating_sub(2).min(Self::LENGTH_BUCKETS - 1)
    }

    /// Bucket index for 0-based proposal depth (`0`, `1`, `2..=3`, `4..=7`, `8..=15`, `>=16`).
    #[inline]
    pub(crate) fn depth_bucket(depth: usize) -> usize {
        let bit_width = (usize::BITS - depth.leading_zeros()) as usize;
        bit_width.min(Self::DEPTH_BUCKETS - 1)
    }

    /// Estimated conditional acceptance rate for a lookup token at 0-based
    /// `depth` past the anchor, copied from a match of `match_len` tokens.
    pub(crate) fn rate(
        &self,
        samples: bool,
        match_len: usize,
        depth: usize,
        agreement: LookupAgreement,
    ) -> f32 {
        if match_len < LookupIndex::MIN_MATCH {
            return 0.0;
        }
        let l = Self::length_bucket(match_len);
        let d = Self::depth_bucket(depth);
        let a = agreement as usize;
        let c = self.counts[usize::from(samples)][l][d][a];
        ((c.accepted + Self::PRIOR_LABELS * f64::from(Self::PRIOR[l][d][a]))
            / (c.labels + Self::PRIOR_LABELS)) as f32
    }

    /// Update decayed online counts from one verified lookup chain outcome.
    pub(crate) fn observe(
        &mut self,
        samples: bool,
        match_len: usize,
        verified: &[LookupAgreement],
        accepted: usize,
    ) {
        if match_len < LookupIndex::MIN_MATCH || verified.is_empty() {
            return;
        }
        let accepted = accepted.min(verified.len());
        let labels = (accepted + 1).min(verified.len());
        let decay = (-(labels as f64) / Self::HALF_LIFE).exp2();
        let lane = &mut self.counts[usize::from(samples)];
        for depths in lane.iter_mut() {
            for agreements in depths.iter_mut() {
                for c in agreements.iter_mut() {
                    c.accepted *= decay;
                    c.labels *= decay;
                }
            }
        }
        let l = Self::length_bucket(match_len);
        for (depth, &agreement) in verified.iter().take(labels).enumerate() {
            let cells = &mut lane[l][Self::depth_bucket(depth)];
            let y = if depth < accepted { 1.0 } else { 0.0 };
            cells[agreement as usize].accepted += y;
            cells[agreement as usize].labels += 1.0;
            if agreement != LookupAgreement::Unknown {
                cells[LookupAgreement::Unknown as usize].accepted += y;
                cells[LookupAgreement::Unknown as usize].labels += 1.0;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Logistic Acceptance Calibration (Splash `VerifyBudget.{hpp,cpp}`)
// ---------------------------------------------------------------------------

#[inline]
fn logit_f64(p: f64) -> f64 {
    let p = p.clamp(1.0e-6, 1.0 - 1.0e-6);
    (p / (1.0 - p)).ln()
}

/// Logistic calibration parameters for draft proposal acceptance:
/// `logit(p) = slope * logit(q) + intercepts[position]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct BudgetAcceptance {
    pub slope: f32,
    pub intercepts: [f32; DRAFT_SLOTS],
}

impl BudgetAcceptance {
    /// Splash `VerifyBudget::kPrior` fitted on Qwen3.8 / Qwen3.6-35B-A3B decode cycles.
    pub(crate) const PRIOR: Self = Self {
        slope: 0.7,
        intercepts: [0.0, -0.35, -0.8, -0.75, -0.75, -0.85, -0.45],
    };

    /// Calibrated conditional acceptance probability for a draft candidate of
    /// probability `q` at 0-based proposal `position`.
    #[inline]
    pub(crate) fn estimate(&self, q: f32, position: usize) -> f32 {
        let q = q.clamp(1.0e-6, 1.0 - 1.0e-6);
        let intercept = self.intercepts.get(position).copied().unwrap_or(0.0);
        let z = self.slope * (q / (1.0 - q)).ln() + intercept;
        1.0 / (1.0 + (-z).exp())
    }
}

impl Default for BudgetAcceptance {
    fn default() -> Self {
        Self::PRIOR
    }
}

/// One labeled draft node from a completed verification walk, used to refit
/// [`VerifyBudget`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct BudgetLabel {
    pub position: usize,
    pub feature: f32,
    pub accepted: bool,
    pub cap: f32,
}

impl BudgetLabel {
    #[inline]
    fn is_valid(&self) -> bool {
        self.position < DRAFT_SLOTS
            && self.feature.is_finite()
            && self.cap.is_finite()
            && (0.0..=1.0).contains(&self.cap)
    }

    #[inline]
    fn is_capped(&self, z: f64) -> bool {
        self.cap < 1.0 && z > logit_f64(f64::from(self.cap))
    }
}

/// Online logistic calibrator for draft tree probabilities (Splash `VerifyBudget`).
#[derive(Clone, Debug)]
pub(crate) struct VerifyBudget {
    acceptance: [BudgetAcceptance; 2],
    labels: [Vec<BudgetLabel>; 2],
    next: [usize; 2],
    cycles: u64,
}

impl Default for VerifyBudget {
    fn default() -> Self {
        Self::new()
    }
}

impl VerifyBudget {
    pub(crate) const REFIT_CYCLES: u64 = 64;
    pub(crate) const MAX_LABELS: usize = 4096;
    pub(crate) const SLOPE_PRECISION: f64 = 50.0;
    pub(crate) const INTERCEPT_PRECISION: f64 = 4.0;
    const PARAMS: usize = 1 + DRAFT_SLOTS;
    const NEWTON_ITERS: usize = 16;

    pub(crate) fn new() -> Self {
        Self {
            acceptance: [BudgetAcceptance::PRIOR, BudgetAcceptance::PRIOR],
            labels: [Vec::new(), Vec::new()],
            next: [0, 0],
            cycles: 0,
        }
    }

    pub(crate) fn acceptance(&self, samples: bool) -> &BudgetAcceptance {
        &self.acceptance[usize::from(samples)]
    }

    /// Record a cycle's draft labels and trigger a MAP refit every `REFIT_CYCLES`.
    pub(crate) fn observe(&mut self, samples: bool, labels: &[BudgetLabel]) {
        let slot = usize::from(samples);
        let kept = &mut self.labels[slot];
        let next = &mut self.next[slot];
        for &label in labels {
            if !label.is_valid() {
                continue;
            }
            if kept.len() < Self::MAX_LABELS {
                kept.push(label);
            } else {
                kept[*next] = label;
                *next = (*next + 1) % Self::MAX_LABELS;
            }
        }
        self.cycles = self.cycles.wrapping_add(1);
        if self.cycles.is_multiple_of(Self::REFIT_CYCLES) {
            for s in 0..2 {
                if !self.labels[s].is_empty() {
                    let ordered = self.ordered_labels(s == 1);
                    self.acceptance[s] = Self::fit(&ordered, &self.acceptance[s]);
                }
            }
        }
    }

    /// Kept labels for greedy (`false`) or sampled (`true`) lanes, oldest first.
    pub(crate) fn ordered_labels(&self, samples: bool) -> Vec<BudgetLabel> {
        let slot = usize::from(samples);
        let kept = &self.labels[slot];
        let split = self.next[slot].min(kept.len());
        let mut out = Vec::with_capacity(kept.len());
        out.extend_from_slice(&kept[split..]);
        out.extend_from_slice(&kept[..split]);
        out
    }

    #[inline]
    fn prior_precision(param: usize) -> f64 {
        if param == 0 {
            Self::SLOPE_PRECISION
        } else {
            Self::INTERCEPT_PRECISION
        }
    }

    fn params_of(a: &BudgetAcceptance) -> [f64; Self::PARAMS] {
        let mut p = [0.0; Self::PARAMS];
        p[0] = f64::from(a.slope);
        for i in 0..DRAFT_SLOTS {
            p[1 + i] = f64::from(a.intercepts[i]);
        }
        p
    }

    fn objective(
        labels: &[BudgetLabel],
        theta: &[f64; Self::PARAMS],
        prior: &[f64; Self::PARAMS],
    ) -> f64 {
        let mut value = 0.0;
        for label in labels {
            if !label.is_valid() {
                continue;
            }
            let z = theta[0] * f64::from(label.feature) + theta[1 + label.position];
            if label.is_capped(z) {
                let cap = f64::from(label.cap).clamp(1.0e-6, 1.0 - 1.0e-6);
                value -= if label.accepted {
                    cap.ln()
                } else {
                    (-cap).ln_1p()
                };
                continue;
            }
            let margin = if label.accepted { z } else { -z };
            value += if margin > 0.0 {
                (-margin).exp().ln_1p()
            } else {
                -margin + margin.exp().ln_1p()
            };
        }
        for i in 0..Self::PARAMS {
            let diff = theta[i] - prior[i];
            value += 0.5 * Self::prior_precision(i) * diff * diff;
        }
        value
    }

    fn solve(
        mut h: [[f64; Self::PARAMS]; Self::PARAMS],
        mut g: [f64; Self::PARAMS],
    ) -> [f64; Self::PARAMS] {
        for col in 0..Self::PARAMS {
            let mut pivot = col;
            for row in (col + 1)..Self::PARAMS {
                if h[row][col].abs() > h[pivot][col].abs() {
                    pivot = row;
                }
            }
            h.swap(col, pivot);
            g.swap(col, pivot);
            let diag = h[col][col];
            if diag.abs() < 1.0e-30 {
                continue;
            }
            for row in (col + 1)..Self::PARAMS {
                let factor = h[row][col] / diag;
                for k in col..Self::PARAMS {
                    h[row][k] -= factor * h[col][k];
                }
                g[row] -= factor * g[col];
            }
        }
        let mut x = [0.0; Self::PARAMS];
        for row in (0..Self::PARAMS).rev() {
            let mut sum = g[row];
            for k in (row + 1)..Self::PARAMS {
                sum -= h[row][k] * x[k];
            }
            let diag = h[row][row];
            x[row] = if diag.abs() < 1.0e-30 {
                0.0
            } else {
                sum / diag
            };
        }
        x
    }

    /// Maximum a posteriori logistic calibration via damped Newton iterations.
    pub(crate) fn fit(labels: &[BudgetLabel], start: &BudgetAcceptance) -> BudgetAcceptance {
        let prior = Self::params_of(&BudgetAcceptance::PRIOR);
        let mut theta = Self::params_of(start);
        let mut current = Self::objective(labels, &theta, &prior);

        for _ in 0..Self::NEWTON_ITERS {
            let mut grad = [0.0; Self::PARAMS];
            let mut hess = [[0.0; Self::PARAMS]; Self::PARAMS];
            for label in labels {
                if !label.is_valid() {
                    continue;
                }
                let b = 1 + label.position;
                let f = f64::from(label.feature);
                let z = theta[0] * f + theta[b];
                if label.is_capped(z) {
                    continue;
                }
                let p = 1.0 / (1.0 + (-z).exp());
                let residual = p - if label.accepted { 1.0 } else { 0.0 };
                let weight = p * (1.0 - p);
                grad[0] += residual * f;
                grad[b] += residual;
                hess[0][0] += weight * f * f;
                hess[0][b] += weight * f;
                hess[b][0] += weight * f;
                hess[b][b] += weight;
            }
            for i in 0..Self::PARAMS {
                let prec = Self::prior_precision(i);
                grad[i] += prec * (theta[i] - prior[i]);
                hess[i][i] += prec;
            }
            let step = Self::solve(hess, grad);
            let mut next = [0.0; Self::PARAMS];
            let mut next_val = current;
            let mut scale = 1.0;
            let mut moved = false;
            for _ in 0..30 {
                for i in 0..Self::PARAMS {
                    next[i] = theta[i] - scale * step[i];
                }
                next_val = Self::objective(labels, &next, &prior);
                if next_val <= current {
                    moved = true;
                    break;
                }
                scale *= 0.5;
            }
            if !moved {
                break;
            }
            let mut largest = 0.0f64;
            for i in 0..Self::PARAMS {
                largest = largest.max((next[i] - theta[i]).abs());
            }
            theta = next;
            current = next_val;
            if largest < 1.0e-9 {
                break;
            }
        }

        let mut intercepts = [0.0f32; DRAFT_SLOTS];
        for (i, slot) in intercepts.iter_mut().enumerate() {
            *slot = theta[1 + i] as f32;
        }
        BudgetAcceptance {
            slope: theta[0] as f32,
            intercepts,
        }
    }
}

// ---------------------------------------------------------------------------
// DFlash2 Draft Candidate Lattice
// ---------------------------------------------------------------------------

/// Conditional candidate lattice produced by DFlash2's [`CandidateSelector`]
/// (or constructed from independent per-slot probabilities).
///
/// For `slots = L` and `top_k = K`:
/// - `ids[pos * K + j]` is candidate `j`'s vocabulary ID at 0-based slot `pos`.
/// - `probs[(pos * K + pred) * K + j]` is `P(slot pos = ids[pos, j] | slot pos-1 = ids[pos-1, pred])`.
///   For `pos == 0`, `pred == 0` conditions on the anchor token.
#[derive(Clone, Debug)]
pub(crate) struct DraftLattice {
    pub slots: usize,
    pub top_k: usize,
    pub ids: Vec<u32>,
    pub probs: Vec<f32>,
    /// Candidate index in `0..top_k` considered the draft's top-1 token at each
    /// slot (used to classify [`LookupAgreement::Top`]).
    pub top_candidate: Vec<usize>,
    /// Top `MAX_BRANCH_CHILDREN` candidate indices for each `(pos, pred)` sorted
    /// descending by probability (lowest index tie-break).
    likeliest: Vec<[u8; MAX_BRANCH_CHILDREN]>,
    likeliest_len: Vec<u8>,
}

impl DraftLattice {
    /// Empty lattice (for lookup-only tree construction without a draft step).
    pub(crate) fn empty() -> Self {
        Self {
            slots: 0,
            top_k: 0,
            ids: Vec::new(),
            probs: Vec::new(),
            top_candidate: Vec::new(),
            likeliest: Vec::new(),
            likeliest_len: Vec::new(),
        }
    }

    /// Build from DFlash2 `CandidateSelector` outputs:
    /// - `candidates`: `[slots * top_k]` token IDs
    /// - `scores`: `[slots * top_k * top_k]` transition scores (`unary + edge`)
    ///
    /// Computes temperature-1 softmax probabilities and top-`MAX_BRANCH_CHILDREN`
    /// transitions per `(pos, pred)` matching Splash's `draft_select_tree`.
    pub(crate) fn from_selector_scores(
        candidates: &[u32],
        scores: &[f32],
        slots: usize,
        top_k: usize,
    ) -> Self {
        if slots == 0 || top_k == 0 {
            return Self::empty();
        }
        assert_eq!(
            candidates.len(),
            slots * top_k,
            "DraftLattice candidates length mismatch"
        );
        assert_eq!(
            scores.len(),
            slots * top_k * top_k,
            "DraftLattice scores length mismatch"
        );
        assert!(
            top_k <= u8::MAX as usize,
            "DraftLattice top_k must fit in u8"
        );

        let mut probs = vec![0.0f32; slots * top_k * top_k];
        let mut likeliest = vec![[0u8; MAX_BRANCH_CHILDREN]; slots * top_k];
        let mut likeliest_len = vec![0u8; slots * top_k];

        let max_children = top_k.min(MAX_BRANCH_CHILDREN);
        for pos in 0..slots {
            let num_pred = if pos == 0 { 1 } else { top_k };
            for pred in 0..num_pred {
                let row_idx = pos * top_k + pred;
                let row_start = row_idx * top_k;
                let row_scores = &scores[row_start..row_start + top_k];
                let row_probs = &mut probs[row_start..row_start + top_k];

                let best = row_scores
                    .iter()
                    .copied()
                    .filter(|v| v.is_finite())
                    .fold(f32::NEG_INFINITY, f32::max);
                if best.is_finite() {
                    let mut mass = 0.0f32;
                    for (dst, &s) in row_probs.iter_mut().zip(row_scores.iter()) {
                        let w = if s.is_finite() { (s - best).exp() } else { 0.0 };
                        *dst = w;
                        mass += w;
                    }
                    if mass > 0.0 && mass.is_finite() {
                        let inv = 1.0 / mass;
                        for p in row_probs.iter_mut() {
                            *p *= inv;
                        }
                    } else {
                        row_probs.fill(0.0);
                    }
                }

                // Select up to `MAX_BRANCH_CHILDREN` likeliest candidates by
                // descending score (lowest index wins ties, matching Splash).
                let mut taken = [false; 256];
                let mut count = 0usize;
                for slot_out in likeliest[row_idx].iter_mut().take(max_children) {
                    let mut chosen: Option<usize> = None;
                    for i in 0..top_k {
                        if taken[i] || row_probs[i].is_nan() || row_probs[i] <= 0.0 {
                            continue;
                        }
                        let better = match chosen {
                            None => true,
                            Some(cur) => row_scores[i] > row_scores[cur],
                        };
                        if better {
                            chosen = Some(i);
                        }
                    }
                    let Some(idx) = chosen else {
                        break;
                    };
                    taken[idx] = true;
                    *slot_out = idx as u8;
                    count += 1;
                }
                likeliest_len[row_idx] = count as u8;
            }
        }

        // Default `top_candidate[pos]` follows the greedy Markov chain from anchor.
        let mut top_candidate = Vec::with_capacity(slots);
        let mut pred = 0usize;
        for pos in 0..slots {
            let row_idx = pos * top_k + pred;
            let best = if likeliest_len[row_idx] > 0 {
                likeliest[row_idx][0] as usize
            } else {
                0
            };
            top_candidate.push(best);
            pred = best;
        }

        Self {
            slots,
            top_k,
            ids: candidates.to_vec(),
            probs,
            top_candidate,
            likeliest,
            likeliest_len,
        }
    }

    /// Build from independent per-slot candidate IDs `[slots * top_k]` and
    /// probabilities `[slots * top_k]` (broadcast across all predecessor indices).
    pub(crate) fn from_slot_probs(
        candidates: &[u32],
        slot_probs: &[f32],
        slots: usize,
        top_k: usize,
    ) -> Self {
        if slots == 0 || top_k == 0 {
            return Self::empty();
        }
        assert_eq!(
            candidates.len(),
            slots * top_k,
            "DraftLattice candidates length mismatch"
        );
        assert_eq!(
            slot_probs.len(),
            slots * top_k,
            "DraftLattice slot_probs length mismatch"
        );
        assert!(
            top_k <= u8::MAX as usize,
            "DraftLattice top_k must fit in u8"
        );

        let mut probs = vec![0.0f32; slots * top_k * top_k];
        let mut likeliest = vec![[0u8; MAX_BRANCH_CHILDREN]; slots * top_k];
        let mut likeliest_len = vec![0u8; slots * top_k];
        let mut top_candidate = Vec::with_capacity(slots);
        let max_children = top_k.min(MAX_BRANCH_CHILDREN);

        for pos in 0..slots {
            let src = &slot_probs[pos * top_k..(pos + 1) * top_k];
            let mut norm = vec![0.0f32; top_k];
            let mut sum = 0.0f32;
            for (dst, &p) in norm.iter_mut().zip(src.iter()) {
                if p.is_finite() && p > 0.0 {
                    *dst = p;
                    sum += p;
                }
            }
            if sum > 0.0 {
                let inv = 1.0 / sum;
                for p in &mut norm {
                    *p *= inv;
                }
            }

            let mut taken = [false; 256];
            let mut row_like = [0u8; MAX_BRANCH_CHILDREN];
            let mut count = 0usize;
            for slot_out in row_like.iter_mut().take(max_children) {
                let mut chosen: Option<usize> = None;
                for i in 0..top_k {
                    if taken[i] || norm[i].is_nan() || norm[i] <= 0.0 {
                        continue;
                    }
                    let better = match chosen {
                        None => true,
                        Some(cur) => norm[i] > norm[cur],
                    };
                    if better {
                        chosen = Some(i);
                    }
                }
                let Some(idx) = chosen else {
                    break;
                };
                taken[idx] = true;
                *slot_out = idx as u8;
                count += 1;
            }
            top_candidate.push(if count > 0 { row_like[0] as usize } else { 0 });

            let num_pred = if pos == 0 { 1 } else { top_k };
            for pred in 0..num_pred {
                let row_idx = pos * top_k + pred;
                probs[row_idx * top_k..(row_idx + 1) * top_k].copy_from_slice(&norm);
                likeliest[row_idx] = row_like;
                likeliest_len[row_idx] = count as u8;
            }
        }

        Self {
            slots,
            top_k,
            ids: candidates.to_vec(),
            probs,
            top_candidate,
            likeliest,
            likeliest_len,
        }
    }

    /// Mark index `top_k - 1` as each slot's unary top-1 candidate (matching
    /// `fused_topk16`, which stores candidates ascending by unary logit).
    pub(crate) fn with_ascending_unary_top(mut self) -> Self {
        if self.top_k > 0 {
            self.top_candidate.fill(self.top_k - 1);
        }
        self
    }

    #[inline]
    pub(crate) fn token(&self, pos: usize, cand: usize) -> u32 {
        self.ids[pos * self.top_k + cand]
    }

    #[inline]
    pub(crate) fn prob(&self, pos: usize, pred: usize, cand: usize) -> f32 {
        let p = if pos == 0 { 0 } else { pred };
        self.probs[(pos * self.top_k + p) * self.top_k + cand]
    }

    #[inline]
    fn likeliest_slice(&self, pos: usize, pred: usize) -> &[u8] {
        let p = if pos == 0 { 0 } else { pred };
        let row_idx = pos * self.top_k + p;
        let len = self.likeliest_len[row_idx] as usize;
        &self.likeliest[row_idx][..len]
    }

    /// Classify how `token` at 0-based proposal `depth` agrees with this lattice.
    pub(crate) fn lookup_agreement(&self, token: u32, depth: usize) -> LookupAgreement {
        if depth >= self.slots || self.top_k == 0 {
            return LookupAgreement::Unknown;
        }
        let top_idx = self.top_candidate[depth];
        let slice = &self.ids[depth * self.top_k..(depth + 1) * self.top_k];
        if slice[top_idx] == token {
            return LookupAgreement::Top;
        }
        if slice.contains(&token) {
            LookupAgreement::Candidate
        } else {
            LookupAgreement::Absent
        }
    }
}

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

fn compute_conv_taps(parent: &[i32]) -> Vec<[i32; 4]> {
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

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
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
        let lattice =
            DraftLattice::from_selector_scores(candidates, scores, slots, DRAFT_CANDIDATES);
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
                lattice.lookup_agreement(cpu_tree.tokens[i], (cpu_tree.depth[i] - 1) as usize)
                    as u32
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
        let neg_inf_scores =
            vec![f32::NEG_INFINITY; DRAFT_SLOTS * DRAFT_CANDIDATES * DRAFT_CANDIDATES];
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
            MxArray::from_int32(&cand_i32, &[1, DRAFT_SLOTS as i64, DRAFT_CANDIDATES as i64])
                .unwrap();
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
        MxArray::eval_arrays(&[&gpu.tree_tokens, &gpu.parent, &gpu.depths, &gpu.node_meta])
            .unwrap();
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
            MxArray::from_int32(&cand_i32, &[1, DRAFT_SLOTS as i64, DRAFT_CANDIDATES as i64])
                .unwrap();
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
}
