use super::MAX_BRANCH_CHILDREN;
use super::sam::LookupAgreement;

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
    pub(super) fn likeliest_slice(&self, pos: usize, pred: usize) -> &[u8] {
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
