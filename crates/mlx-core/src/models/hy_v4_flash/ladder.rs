//! RAM ladder: snapshots of one sequence at rows a later turn may go back
//! to (prefill state rows, the turn end, a cancelled prefill's last chunk
//! end), newest `RAM_STATES` kept.
//!
//! Every entry is a prefix of the sequence's current history: whoever cuts
//! the sequence below an entry drops it (`drop_above`).

/// Entries kept; the admission charges them in the kv term.
pub const RAM_STATES: usize = 8;

/// The length of the sequence a snapshot was taken at.
pub trait Snapshot {
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Why an entry was taken.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// A prefill checkpoint row.
    Checkpoint,
    /// The block row before the generation prompt.
    Prompt,
    /// The end of a turn.
    Turn,
    /// The last chunk end of a cancelled prefill.
    Cancel,
}

struct Entry<S> {
    kind: Kind,
    snap: S,
    /// Insertion order; the lowest goes first when full.
    stamp: u64,
}

/// Entries ascending by length, at most one per length.
pub struct Ladder<S> {
    entries: Vec<Entry<S>>,
    cap: usize,
    clock: u64,
}

impl<S: Snapshot> Default for Ladder<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: Snapshot> Ladder<S> {
    pub fn new() -> Self {
        Self::with_cap(RAM_STATES)
    }

    /// At most `cap` entries (at least one).
    pub fn with_cap(cap: usize) -> Self {
        Self {
            entries: Vec::new(),
            cap: cap.max(1),
            clock: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Entry lengths, ascending.
    pub fn lens(&self) -> Vec<usize> {
        self.entries.iter().map(|e| e.snap.len()).collect()
    }

    fn find(&self, len: usize) -> Result<usize, usize> {
        self.entries.binary_search_by_key(&len, |e| e.snap.len())
    }

    /// Adds `snap`, replacing an entry of the same length; the oldest entry
    /// goes when the ladder is full.
    pub fn put(&mut self, kind: Kind, snap: S) {
        self.clock += 1;
        let e = Entry {
            kind,
            snap,
            stamp: self.clock,
        };
        match self.find(e.snap.len()) {
            Ok(i) => self.entries[i] = e,
            Err(i) => self.entries.insert(i, e),
        }
        while self.entries.len() > self.cap {
            let oldest = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.stamp)
                .map(|(i, _)| i);
            match oldest {
                Some(i) => {
                    self.entries.remove(i);
                }
                None => break,
            }
        }
    }

    pub fn contains(&self, len: usize) -> bool {
        self.find(len).is_ok()
    }

    pub fn kind(&self, len: usize) -> Option<Kind> {
        self.find(len).ok().map(|i| self.entries[i].kind)
    }

    pub fn snap(&self, len: usize) -> Option<&S> {
        self.find(len).ok().map(|i| &self.entries[i].snap)
    }

    /// The longest entry no longer than `limit` and `held` (the sequence's
    /// length now).
    pub fn deepest(&self, limit: usize, held: usize) -> Option<usize> {
        let top = limit.min(held);
        self.entries
            .iter()
            .rev()
            .map(|e| e.snap.len())
            .find(|&l| l <= top)
    }

    /// `deepest` among the entries `ok` accepts.
    pub fn deepest_where(
        &self,
        limit: usize,
        held: usize,
        ok: impl Fn(&S) -> bool,
    ) -> Option<usize> {
        let top = limit.min(held);
        self.entries
            .iter()
            .rev()
            .find(|e| e.snap.len() <= top && ok(&e.snap))
            .map(|e| e.snap.len())
    }

    pub fn remove(&mut self, len: usize) {
        if let Ok(i) = self.find(len) {
            self.entries.remove(i);
        }
    }

    /// Drops the entries longer than `len` (the sequence was cut to `len`).
    pub fn drop_above(&mut self, len: usize) {
        let keep = self.entries.partition_point(|e| e.snap.len() <= len);
        self.entries.truncate(keep);
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }
}
