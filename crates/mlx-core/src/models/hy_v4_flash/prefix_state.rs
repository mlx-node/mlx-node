//! States of paged sequences at block rows, beside the pool's KV blocks:
//! what an owner holds outside the pool (`owner.rs`) at row `b`, in storage
//! of its own, so a later turn of any owner whose prompt shares `b` rows
//! (and whose pool holds their blocks) resumes there.
//!
//! ```text
//! state at block row b
//!   per trunk layer   sliding: the window ring │ indexed: the tok_k tail (keys of rows < b in the index plane)
//!   MTP (b - 1 rows)  closed 32-row segments (K/V [32, row] f16 + pooled keys) shared by the states of one lineage
//!                     + the open rows past them + the indexer tail + ONE seed row h_trunk(b - 1)
//! rows              P (block row before the generation prompt), every 8,192 below 32,768,
//!                   every 32,768 from there (all on the chunk grid), T (block row below the turn's KV end)
//! RAM               the newest RAM_STATES copies; a cut below a state drops it unless its blocks stay cached
//! ```

use std::collections::HashSet;
use std::sync::Arc;

use napi::{Error, Result};

use super::cache::WindowRing;
use super::indexer::TokTail;
use super::ladder::{Kind, RAM_STATES};
use super::prefill::{BLOCK_ROWS, Grid, StateRow, turn_row};
use crate::array::MxArray;
use crate::models::qwen3_5::dflash2_state::store::{
    HybridStateStore, Lineage, StateBytes, StateClass,
};

fn bad(msg: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("Hy4 prefix states: {msg}"))
}

/// MTP rows `[32k, 32k + 32)` of one lineage: K/V `[32, row]` f16 and the
/// pooled keys of their groups `[32 / kpool, d]` f16. Immutable.
pub struct Segment {
    pub kv: MxArray,
    pub pooled: MxArray,
}

impl Segment {
    pub fn nbytes(&self) -> usize {
        self.kv.nbytes() + self.pooled.nbytes()
    }
}

/// The MTP rows of a state past its whole segments: K/V `[r, row]` and the
/// pooled keys of their closed groups (`None` when they close none).
pub struct OpenRows {
    pub kv: MxArray,
    pub pooled: Option<MxArray>,
}

/// The MTP layer at a state's row `b`: its `b - 1` settled rows and the seed.
pub struct DraftState {
    pub segments: Vec<Arc<Segment>>,
    pub open: Option<OpenRows>,
    pub tail: TokTail,
    /// `h_trunk(b - 1)`, `[1, d]` f32.
    pub seed: MxArray,
}

impl DraftState {
    /// MTP rows held.
    pub fn rows(&self) -> Result<usize> {
        let open = match &self.open {
            Some(o) => o.kv.shape_at(0)? as usize,
            None => 0,
        };
        Ok(self.segments.len() * BLOCK_ROWS + open)
    }

    fn own_bytes(&self) -> usize {
        self.open.as_ref().map_or(0, |o| {
            o.kv.nbytes() + o.pooled.as_ref().map_or(0, MxArray::nbytes)
        }) + self.tail.nbytes()
            + self.seed.nbytes()
    }
}

/// One trunk layer's part of a state.
#[derive(Clone)]
pub enum StateLayer {
    Window(WindowRing),
    Tail(TokTail),
}

/// An owner's cell at block row `row`, in storage of its own.
pub struct PrefixState {
    pub row: usize,
    pub layers: Vec<StateLayer>,
    pub draft: Option<DraftState>,
}

impl PrefixState {
    fn own_bytes(&self) -> u64 {
        let layers: usize = self
            .layers
            .iter()
            .map(|l| match l {
                StateLayer::Window(r) => r.nbytes(),
                StateLayer::Tail(t) => t.nbytes(),
            })
            .sum();
        (layers + self.draft.as_ref().map_or(0, DraftState::own_bytes)) as u64
    }
}

/// What a stored state holds.
pub trait Stored {
    /// Every byte it holds, shared parts included.
    fn bytes(&self) -> u64;
    /// `(identity, bytes)` of the parts it may share with other states.
    fn shared(&self) -> Vec<(usize, u64)>;
}

impl Stored for PrefixState {
    fn bytes(&self) -> u64 {
        self.own_bytes() + self.shared().iter().map(|s| s.1).sum::<u64>()
    }

    fn shared(&self) -> Vec<(usize, u64)> {
        self.draft.as_ref().map_or_else(Vec::new, |d| {
            d.segments
                .iter()
                .map(|s| (Arc::as_ptr(s) as usize, s.nbytes() as u64))
                .collect()
        })
    }
}

/// A state as the store keeps it.
struct Held<S>(S);

impl<S: Stored> StateBytes for Held<S> {
    fn state_bytes(&self) -> u64 {
        self.0.bytes()
    }
}

/// The class a state of `kind` is stored with: a checkpoint goes first.
fn class_of(kind: Kind) -> StateClass {
    match kind {
        Kind::Checkpoint => StateClass::Checkpoint,
        Kind::Prompt | Kind::Turn | Kind::Cancel => StateClass::Ordinary,
    }
}

/// One owner's turn: the prompt's state rows past the row it resumed at,
/// and the newest block-row state past them (the T candidate).
pub struct Turn {
    rows: Vec<StateRow>,
    resume: usize,
    held: Option<PrefixState>,
}

impl Turn {
    /// A turn of `prompt` resumed at `resume` (0 when cold).
    pub fn new(grid: &Grid, resume: usize, prompt: &[u32]) -> Self {
        Self {
            rows: grid.states(resume, prompt),
            resume,
            held: None,
        }
    }

    /// The prompt's state rows, ascending.
    pub fn rows(&self) -> &[StateRow] {
        &self.rows
    }

    /// The kind of the prompt state at `row`, if `row` is one.
    pub fn prompt_kind(&self, row: usize) -> Option<Kind> {
        self.rows.iter().find(|s| s.row == row).map(|s| s.kind)
    }

    /// The deepest prompt state row, else the resume row.
    fn floor(&self) -> usize {
        self.rows.last().map_or(self.resume, |s| s.row)
    }

    /// Keeps `s` as the T candidate when it lies past the prompt's state
    /// rows and past the one held.
    pub fn hold(&mut self, s: PrefixState) {
        let newer = self.held.as_ref().is_none_or(|h| s.row > h.row);
        if s.row > self.floor() && newer {
            self.held = Some(s);
        }
    }

    /// The held candidate's row.
    pub fn held_row(&self) -> Option<usize> {
        self.held.as_ref().map(|h| h.row)
    }

    /// The sequence was cut to `len`: a held state past it goes.
    pub fn cut(&mut self, len: usize) {
        if self.held.as_ref().is_some_and(|h| h.row > len) {
            self.held = None;
        }
    }

    /// The turn state at its end, `kv_end` rows with KV written: the held
    /// state at `turn_row(kv_end)`, or the deepest one held below it when a
    /// multi-row commit stepped over that row. `None` when that row is a
    /// prompt state row or nothing was held.
    pub fn close(&mut self, kv_end: usize) -> Option<(StateRow, PrefixState)> {
        let t = turn_row(kv_end);
        let s = self.held.take().filter(|s| s.row <= t)?;
        Some((
            StateRow {
                row: s.row,
                kind: Kind::Turn,
            },
            s,
        ))
    }
}

/// A lineage: token ids, the extra keys of each block (blocks past the
/// list carry none) and the load's salt.
#[derive(Clone, Copy)]
pub struct Key<'a> {
    pub tokens: &'a [u32],
    pub keys: &'a [Vec<u64>],
    pub salt: u64,
}

impl<'a> Key<'a> {
    /// A text lineage with no extra keys.
    pub fn text(tokens: &'a [u32], salt: u64) -> Self {
        Self {
            tokens,
            keys: &[],
            salt,
        }
    }

    fn lineage(self) -> Lineage<'a> {
        Lineage {
            tokens: self.tokens,
            extra_keys: self.keys,
            salt: self.salt,
        }
    }
}

/// The states of one load, by lineage (token ids, per-block keys, salt).
/// Model thread only.
pub struct States<S = PrefixState> {
    store: HybridStateStore<Held<S>>,
    ram_cap: usize,
}

impl<S: Stored> Default for States<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: Stored> States<S> {
    pub fn new() -> Self {
        Self::with_cap(RAM_STATES)
    }

    /// At most `cap` RAM copies (at least one).
    pub fn with_cap(cap: usize) -> Self {
        Self {
            store: HybridStateStore::new(BLOCK_ROWS as u32),
            ram_cap: cap.max(1),
        }
    }

    /// States with a RAM copy.
    pub fn ram_len(&self) -> usize {
        self.store.entries().filter(|e| e.ram().is_some()).count()
    }

    /// Rows of the states with a RAM copy, ascending.
    pub fn ram_rows(&self) -> Vec<usize> {
        let mut rows: Vec<usize> = self
            .store
            .entries()
            .filter(|e| e.ram().is_some())
            .map(|e| e.depth())
            .collect();
        rows.sort_unstable();
        rows
    }

    /// Bytes the RAM copies hold, shared parts once.
    pub fn held_bytes(&self) -> u64 {
        let mut seen = HashSet::new();
        let mut total = 0u64;
        for Held(s) in self.store.entries().filter_map(|e| e.ram()) {
            let shared = s.shared();
            total += s.bytes().saturating_sub(shared.iter().map(|p| p.1).sum());
            for (id, bytes) in shared {
                if seen.insert(id) {
                    total += bytes;
                }
            }
        }
        total
    }

    /// The RAM copy that goes first: a checkpoint (oldest publication), else
    /// the least recently used; never one with a disk write in flight or in
    /// use.
    fn victim(&self) -> Option<u64> {
        let free = || {
            self.store.entries().filter(|e| {
                e.ram().is_some()
                    && e.disk().is_none_or(|d| d.landed)
                    && !self.store.in_use(e.mark())
            })
        };
        free()
            .filter(|e| e.checkpoint())
            .min_by_key(|e| e.publication())
            .or_else(|| free().min_by_key(|e| e.last_used()))
            .map(|e| e.id())
    }

    /// Gives up the RAM copy of `id`: a state whose disk copy landed keeps
    /// it alone, any other goes.
    fn evict(&mut self, id: u64) {
        let durable = self
            .store
            .get(id)
            .is_some_and(|e| e.disk().is_some_and(|d| d.landed));
        if durable {
            self.store.drop_ram(id);
        } else {
            self.store.remove(id);
        }
    }

    /// Gives up RAM copies in eviction order until `shortfall` bytes are
    /// free or none is left to give; the bytes freed.
    pub fn reclaim(&mut self, shortfall: u64) -> u64 {
        let before = self.held_bytes();
        while before.saturating_sub(self.held_bytes()) < shortfall {
            let Some(id) = self.victim() else {
                break;
            };
            self.evict(id);
        }
        before.saturating_sub(self.held_bytes())
    }

    /// Publishes `state` at the end of `lineage` (whole blocks). The oldest
    /// RAM copy goes when `RAM_STATES` are held; `None` when every held one
    /// is in use. A state already on that block is reused.
    pub fn publish(&mut self, key: Key<'_>, state: S, kind: Kind) -> Result<Option<u64>> {
        let lineage = key.lineage();
        if lineage.tokens.is_empty() || !lineage.tokens.len().is_multiple_of(BLOCK_ROWS) {
            return Err(bad(format!(
                "a state at row {} is not on a block row",
                lineage.tokens.len()
            )));
        }
        let (_, there) = self.store.locate(lineage);
        let adds = there.is_none_or(|id| self.store.get(id).is_some_and(|e| e.ram().is_none()));
        if adds {
            while self.ram_len() >= self.ram_cap {
                match self.victim() {
                    Some(id) => self.evict(id),
                    None => return Ok(None),
                }
            }
        }
        let ins = self.store.insert(lineage, Held(state), class_of(kind))?;
        Ok(Some(ins.id))
    }

    /// The deepest state with a RAM copy whose lineage is a strict prefix of
    /// `prompt` and every block of whose chain `cached` holds: `(id, row)`.
    pub fn lookup(&self, prompt: Key<'_>, cached: &dyn Fn(u64) -> bool) -> Option<(u64, usize)> {
        let prompt = prompt.lineage();
        let mut hit = self.store.lookup(prompt)?;
        loop {
            let usable = self
                .store
                .get(hit.id)
                .is_some_and(|e| e.ram().is_some() && e.chain().iter().all(|&h| cached(h)));
            if usable {
                return Some((hit.id, hit.depth));
            }
            hit = self
                .store
                .lookup_within(prompt, hit.depth.checked_sub(1)?)?;
        }
    }

    /// The RAM copy of `id`.
    pub fn get(&self, id: u64) -> Option<&S> {
        self.store.get(id).and_then(|e| e.ram()).map(|h| &h.0)
    }

    /// Stamps `id` newest (after a restore from it).
    pub fn touch(&mut self, id: u64) {
        self.store.touch(id);
    }

    /// A sequence of `tokens` (salt `salt`) was cut to `len`: its states
    /// deeper than `len` go, unless every block of their chain stays in
    /// `cached`.
    pub fn cut(&mut self, tokens: &[u32], salt: u64, len: usize, cached: &dyn Fn(u64) -> bool) {
        let stale: Vec<u64> = self
            .store
            .entries()
            .filter(|e| {
                e.salt() == salt
                    && e.depth() > len
                    && tokens.starts_with(e.tokens())
                    && !e.chain().iter().all(|&h| cached(h))
            })
            .map(|e| e.id())
            .collect();
        for id in stale {
            self.store.remove(id);
        }
    }

    /// Drops the RAM copy of every state with a block of its chain no longer
    /// in `cached`.
    pub fn prune(&mut self, cached: &dyn Fn(u64) -> bool) {
        let stale: Vec<u64> = self
            .store
            .entries()
            .filter(|e| e.ram().is_some() && !e.chain().iter().all(|&h| cached(h)))
            .map(|e| e.id())
            .collect();
        for id in stale {
            self.evict(id);
        }
    }

    pub fn clear(&mut self) {
        self.store.clear();
    }
}
