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
