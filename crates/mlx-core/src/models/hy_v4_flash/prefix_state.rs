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
//! keys              a block's extra keys say how it was written: a cold prefill's pass key, else a
//!                   write-path key (decode rows, a prefill from a row no cold prefill cuts at) that no
//!                   cold prefill computes, so only the writing lineage resumes from it
//! ```

use std::collections::HashSet;
use std::sync::Arc;

use napi::{Error, Result};

use super::cache::WindowRingSnapshot;
use super::indexer::TokTail;
use super::ladder::{Kind, RAM_STATES};
use super::prefill::{BLOCK_ROWS, Grid, StateRow, turn_row};
use crate::array::MxArray;
use crate::disk_tier::store::{Entry, HybridStateStore, Lineage, StateBytes, StateClass};

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
    Window(WindowRingSnapshot),
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
pub struct Held<S> {
    pub state: S,
    pub kind: Kind,
}

impl<S: Stored> StateBytes for Held<S> {
    fn state_bytes(&self) -> u64 {
        self.state.bytes()
    }
}

/// The class a state of `kind` is stored with: a checkpoint goes first.
fn class_of(kind: Kind) -> StateClass {
    match kind {
        Kind::Checkpoint => StateClass::Checkpoint,
        Kind::Prompt | Kind::Turn | Kind::Cancel => StateClass::Ordinary,
    }
}

/// Merges an incoming publication kind onto an existing state at the same
/// block: `Kind::Prompt` always wins (never downgraded to `Checkpoint` or
/// `Turn`), and any ordinary kind upgrades a `Checkpoint`.
fn upgrade_kind(existing: Kind, incoming: Kind) -> Kind {
    match (existing, incoming) {
        (Kind::Prompt, _) | (_, Kind::Prompt) => Kind::Prompt,
        (Kind::Checkpoint, other) => other,
        (other, _) => other,
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

    /// Whether [`Self::hold`] would keep a state at `row`.
    pub fn wants(&self, row: usize) -> bool {
        row > self.floor() && self.held.as_ref().is_none_or(|h| row > h.row)
    }

    /// Keeps `s` as the T candidate when it lies past the prompt's state
    /// rows and past the one held.
    pub fn hold(&mut self, s: PrefixState) {
        if self.wants(s.row) {
            self.held = Some(s);
        }
    }

    /// The held candidate's row.
    pub fn held_row(&self) -> Option<usize> {
        self.held.as_ref().map(|h| h.row)
    }

    /// Bytes the held candidate keeps alive.
    pub fn held_bytes(&self) -> u64 {
        self.held.as_ref().map_or(0, Stored::bytes)
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
    /// States memory dropped, their disk copy not durable.
    dropped: u64,
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
            dropped: 0,
        }
    }

    pub(crate) fn store(&self) -> &HybridStateStore<Held<S>> {
        &self.store
    }

    pub(crate) fn store_mut(&mut self) -> &mut HybridStateStore<Held<S>> {
        &mut self.store
    }

    /// States memory dropped so far without a durable disk copy.
    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    /// Maximum RAM copies held at once.
    pub fn ram_cap(&self) -> usize {
        self.ram_cap
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

    /// Bytes `state` adds on top of shared parts already in `seen`.
    fn delta_bytes(seen: &mut HashSet<usize>, state: &S) -> u64 {
        let shared = state.shared();
        let own = state
            .bytes()
            .saturating_sub(shared.iter().map(|p| p.1).sum::<u64>());
        let mut extra = own;
        for (id, bytes) in shared {
            if seen.insert(id) {
                extra = extra.saturating_add(bytes);
            }
        }
        extra
    }

    /// Bytes the RAM copies hold, shared parts once.
    pub fn held_bytes(&self) -> u64 {
        let mut seen = HashSet::new();
        let mut total = 0u64;
        for Held { state: s, .. } in self.store.entries().filter_map(|e| e.ram()) {
            total = total.saturating_add(Self::delta_bytes(&mut seen, s));
        }
        total
    }

    /// Incremental RAM bytes adding `state` would add to [`Self::held_bytes`],
    /// counting shared parts only when not already held by another RAM entry.
    pub fn added_bytes(&self, state: &S) -> u64 {
        let mut seen = HashSet::new();
        for Held { state: s, .. } in self.store.entries().filter_map(|e| e.ram()) {
            for (id, _) in s.shared() {
                seen.insert(id);
            }
        }
        Self::delta_bytes(&mut seen, state)
    }

    /// Whether `e` holds a RAM copy eligible for eviction (not protected, not
    /// in use, and not waiting for a disk write to land).
    fn is_free_excluding(&self, e: &Entry<Held<S>>, protect: Option<u64>) -> bool {
        Some(e.id()) != protect
            && e.ram().is_some()
            && e.disk().is_none_or(|d| d.landed)
            && !self.store.in_use(e.mark())
    }

    /// Count, resident bytes (shared parts once), and shared part ids of RAM
    /// entries that cannot currently be evicted.
    fn pinned_ram_excluding(&self, protect: Option<u64>) -> (usize, u64, HashSet<usize>) {
        let mut count = 0usize;
        let mut bytes = 0u64;
        let mut seen = HashSet::new();
        for e in self.store.entries() {
            if let Some(Held { state: s, .. }) = e.ram()
                && !self.is_free_excluding(e, protect)
            {
                count += 1;
                bytes = bytes.saturating_add(Self::delta_bytes(&mut seen, s));
            }
        }
        (count, bytes, seen)
    }

    /// The RAM copy that goes first, skipping `protect`: a checkpoint (oldest
    /// publication), else the least recently used; never one with a disk write
    /// in flight or in use.
    fn victim_excluding(&self, protect: Option<u64>) -> Option<u64> {
        let free = || {
            self.store
                .entries()
                .filter(|e| self.is_free_excluding(e, protect))
        };
        free()
            .filter(|e| e.checkpoint())
            .min_by_key(|e| e.publication())
            .or_else(|| free().min_by_key(|e| e.last_used()))
            .map(|e| e.id())
    }

    fn victim(&self) -> Option<u64> {
        self.victim_excluding(None)
    }

    /// Gives up the RAM copy of `id`: a state whose disk copy and chain
    /// landed keeps it alone, any other goes.
    fn evict(&mut self, id: u64) {
        if self.store.get(id).is_some_and(durable) {
            self.store.drop_ram(id);
        } else if self.store.remove(id) {
            self.dropped += 1;
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

    /// Evicts eligible victims until `held_bytes() + incoming <= byte_cap`.
    /// Returns `false` without evicting when pinned RAM entries leave no way
    /// to fit `incoming`.
    pub fn reserve_bytes(&mut self, incoming: u64, byte_cap: u64) -> bool {
        let (_, pinned_bytes, _) = self.pinned_ram_excluding(None);
        if pinned_bytes.saturating_add(incoming) > byte_cap {
            return false;
        }
        while self.held_bytes().saturating_add(incoming) > byte_cap {
            let Some(id) = self.victim() else {
                break;
            };
            self.evict(id);
        }
        self.held_bytes().saturating_add(incoming) <= byte_cap
    }

    /// Evicts eligible victims (never `protect`) before decoding disk-only `id`
    /// until there is both a RAM slot and `incoming` bytes of headroom under
    /// `byte_cap`. Returns `false` without evicting when `id` is not disk-only
    /// or pinned/protected RAM entries leave no way to fit it.
    pub fn reserve_for_promote(
        &mut self,
        id: u64,
        incoming: u64,
        byte_cap: u64,
        protect: Option<u64>,
    ) -> bool {
        if !self.store.get(id).is_some_and(|e| e.ram().is_none()) {
            return false;
        }
        let (pinned_count, pinned_bytes, _) = self.pinned_ram_excluding(protect);
        if pinned_count >= self.ram_cap || pinned_bytes.saturating_add(incoming) > byte_cap {
            return false;
        }
        while self.ram_len() >= self.ram_cap
            || self.held_bytes().saturating_add(incoming) > byte_cap
        {
            match self.victim_excluding(protect) {
                Some(v) if v != id => self.evict(v),
                _ => return false,
            }
        }
        true
    }

    /// Evicts eligible victims (never `protect`) until there is a RAM slot and
    /// `state` fits within `byte_cap` (when given); returns `false` without
    /// evicting if pinned/protected RAM entries make fitting impossible.
    fn make_room_for_state(
        &mut self,
        exclude: Option<u64>,
        state: &S,
        byte_cap: Option<u64>,
        protect: Option<u64>,
    ) -> bool {
        let (pinned_count, pinned_bytes, mut pinned_seen) = self.pinned_ram_excluding(protect);
        if pinned_count >= self.ram_cap {
            return false;
        }
        if let Some(cap) = byte_cap {
            let min_bytes = pinned_bytes.saturating_add(Self::delta_bytes(&mut pinned_seen, state));
            if min_bytes > cap {
                return false;
            }
        }
        while self.ram_len() >= self.ram_cap
            || byte_cap
                .is_some_and(|cap| self.held_bytes().saturating_add(self.added_bytes(state)) > cap)
        {
            match self.victim_excluding(protect) {
                Some(v) if Some(v) != exclude => self.evict(v),
                _ => return false,
            }
        }
        true
    }

    /// Publishes `state` at the end of `lineage` (whole blocks). A victim RAM
    /// copy goes when `ram_cap` are held; `None` when every held one is in
    /// use. A state already on that block is reused (upgrading its kind when
    /// needed, never downgrading `Prompt` or ordinary states); one there on
    /// disk only takes this copy as its RAM copy.
    pub fn publish(&mut self, key: Key<'_>, state: S, kind: Kind) -> Result<Option<u64>> {
        self.publish_within(key, state, kind, None)
    }

    /// Like [`Self::publish`], also enforcing `byte_cap` on total resident
    /// state RAM bytes (shared parts counted once).
    pub fn publish_within(
        &mut self,
        key: Key<'_>,
        state: S,
        kind: Kind,
        byte_cap: Option<u64>,
    ) -> Result<Option<u64>> {
        let lineage = key.lineage();
        if lineage.tokens.is_empty() || !lineage.tokens.len().is_multiple_of(BLOCK_ROWS) {
            return Err(bad(format!(
                "a state at row {} is not on a block row",
                lineage.tokens.len()
            )));
        }
        let (_, there) = self.store.locate(lineage);
        let disk_only = there.filter(|&id| self.store.get(id).is_some_and(|e| e.ram().is_none()));
        if let Some(id) = there
            && disk_only.is_none()
            && let Some(prev) = self.store.get(id).and_then(|e| e.ram()).map(|h| h.kind)
        {
            let next_kind = upgrade_kind(prev, kind);
            if next_kind != prev
                && let Some(held) = self.store.ram_mut(id)
            {
                held.kind = next_kind;
            }
            self.store.reuse(id, next_kind == Kind::Checkpoint);
            return Ok(Some(id));
        }
        if (there.is_none() || disk_only.is_some())
            && !self.make_room_for_state(disk_only, &state, byte_cap, None)
        {
            return Ok(None);
        }
        if let Some(id) = disk_only
            && self.store.get(id).is_some()
        {
            self.store.promote(id, Held { state, kind });
            self.store.reuse(id, kind == Kind::Checkpoint);
            return Ok(Some(id));
        }
        let ins = self
            .store
            .insert(lineage, Held { state, kind }, class_of(kind))?;
        Ok(Some(ins.id))
    }

    /// The deepest state with a RAM copy that `prompt` resumes from
    /// ([`resumes`]) and every block of whose chain `cached` holds:
    /// `(id, row)`.
    pub fn lookup(&self, prompt: Key<'_>, cached: &dyn Fn(u64) -> bool) -> Option<(u64, usize)> {
        self.store
            .entries()
            .filter(|e| {
                e.ram().is_some() && resumes(e, prompt) && e.chain().iter().all(|&h| cached(h))
            })
            .max_by_key(|e| (e.depth(), e.last_used()))
            .map(|e| (e.id(), e.depth()))
    }

    /// The states on disk only that `prompt` resumes from, deeper than
    /// `floor`, the deepest first.
    pub fn disk_only_above(&self, prompt: Key<'_>, floor: usize) -> Vec<(u64, usize)> {
        let mut out: Vec<(u64, usize, u64)> = self
            .store
            .entries()
            .filter(|e| e.ram().is_none() && e.depth() > floor && resumes(e, prompt))
            .map(|e| (e.id(), e.depth(), e.last_used()))
            .collect();
        out.sort_unstable_by_key(|e| std::cmp::Reverse((e.1, e.2)));
        out.into_iter().map(|(id, row, _)| (id, row)).collect()
    }

    /// Gives disk-only `id` its RAM copy `state` back, a victim RAM copy
    /// going first when `ram_cap` are held; false when it is gone, holds
    /// one, or every held one is in use.
    pub fn promote(&mut self, id: u64, state: S) -> bool {
        self.promote_within(id, state, None, None)
    }

    /// Like [`Self::promote`], also enforcing `byte_cap` on total resident
    /// state RAM bytes (shared parts counted once) and never evicting
    /// `protect`.
    pub fn promote_within(
        &mut self,
        id: u64,
        state: S,
        byte_cap: Option<u64>,
        protect: Option<u64>,
    ) -> bool {
        let kind = match self.store.get(id) {
            Some(e) if e.ram().is_none() => {
                if e.checkpoint() {
                    Kind::Checkpoint
                } else if !any_written(e.extra_keys()) {
                    Kind::Prompt
                } else {
                    Kind::Turn
                }
            }
            _ => return false,
        };
        if !self.make_room_for_state(Some(id), &state, byte_cap, protect) {
            return false;
        }
        self.store.promote(id, Held { state, kind })
    }

    /// Drops the RAM copy of every state whose disk copy and chain landed.
    pub fn drop_durable_ram(&mut self) {
        let ids: Vec<u64> = self
            .store
            .entries()
            .filter(|e| e.ram().is_some() && durable(e) && !self.store.in_use(e.mark()))
            .map(|e| e.id())
            .collect();
        for id in ids {
            self.store.drop_ram(id);
        }
    }

    /// The extra keys of `id`'s blocks, one list per whole block.
    pub fn keys_of(&self, id: u64) -> Option<Vec<Vec<u64>>> {
        let e = self.store.get(id)?;
        let blocks = e.depth() / BLOCK_ROWS;
        Some(
            (0..blocks)
                .map(|i| e.extra_keys().get(i).cloned().unwrap_or_default())
                .collect(),
        )
    }

    /// The RAM copy of `id`.
    pub fn get(&self, id: u64) -> Option<&S> {
        self.store.get(id).and_then(|e| e.ram()).map(|h| &h.state)
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

    /// A command reset: every RAM copy goes; a state durable on disk stays
    /// there alone when `keep_durable`.
    pub fn reset(&mut self, keep_durable: bool) {
        if !keep_durable {
            self.store.clear();
            return;
        }
        let entries: Vec<(u64, bool)> =
            self.store.entries().map(|e| (e.id(), durable(e))).collect();
        for (id, kept) in entries {
            if kept {
                self.store.drop_ram(id);
            } else {
                self.store.remove(id);
            }
        }
    }
}

/// A landed disk copy whose whole chain landed too.
pub(crate) fn durable<S>(e: &Entry<S>) -> bool {
    e.disk().is_some_and(|d| d.landed && !d.chain_owed)
}

/// Whether a prompt resumes from `e`: the same salt, a strict prefix of
/// its tokens, and either write-path keys (the lineage that wrote them) or
/// the prompt's own keys on every block below.
pub(crate) fn resumes<S>(e: &Entry<S>, prompt: Key<'_>) -> bool {
    let depth = e.depth();
    depth > 0
        && depth < prompt.tokens.len()
        && e.salt() == prompt.salt
        && prompt.tokens[..depth] == *e.tokens()
        && (any_written(e.extra_keys()) || agrees(e.extra_keys(), prompt.keys, depth / BLOCK_ROWS))
}

fn agrees(a: &[Vec<u64>], b: &[Vec<u64>], blocks: usize) -> bool {
    (0..blocks).all(|i| block_key_agrees(block_key(a, i), block_key(b, i)))
}

/// Whether a stored block's extra key matches a prompt's cold pass key:
/// exact equality, or a pre-upgrade 1-element `[k]` key matching the
/// deterministic `[WRITE_TAG, k]` key now emitted for transient chunks.
fn block_key_agrees(stored: &[u64], prompt: &[u64]) -> bool {
    stored == prompt || matches!((stored, prompt), ([k], [WRITE_TAG, p]) if k == p)
}

fn block_key(keys: &[Vec<u64>], i: usize) -> &[u64] {
    keys.get(i).map_or(&[][..], Vec::as_slice)
}

/// The first extra key of a block a cold prefill never writes.
pub const WRITE_TAG: u64 = 0x6879_345f_7772_6974;

/// Whether one block's extra keys are a write-path key.
pub fn is_written(key: &[u64]) -> bool {
    matches!(key, [WRITE_TAG, _])
}

/// Whether any block of `keys` is a write-path block.
pub fn any_written(keys: &[Vec<u64>]) -> bool {
    keys.iter().any(|k| is_written(k))
}

/// The write-path key of block `i` of `tokens`, written by the turn of
/// `nonce`.
pub fn written_key(tokens: &[u32], i: usize, nonce: u64) -> Vec<u64> {
    let rows = &tokens[i * BLOCK_ROWS..(i + 1) * BLOCK_ROWS];
    vec![
        WRITE_TAG,
        mlx_paged_attn::hash_tokens(rows, nonce, &[i as u64]),
    ]
}

/// Per-block extra keys of `prompt` prefilled from row `from` on top of
/// rows whose whole blocks carry `line`: blocks below `from / 32` keep
/// `line`; where `from` is a cold chunk end of the prompt (`cold_end`, or
/// 0) the rest are the cold prefill's pass keys `pass`, else write-path
/// keys of `nonce` for every whole block from the one holding `from`.
/// Blocks `line` misses below `from` get write-path keys too.
pub fn prompt_keys(
    prompt: &[u32],
    from: usize,
    line: &[Vec<u64>],
    pass: &[Vec<u64>],
    cold_end: bool,
    nonce: u64,
) -> Vec<Vec<u64>> {
    let below = from / BLOCK_ROWS;
    let mut keys: Vec<Vec<u64>> = (0..below)
        .map(|i| match line.get(i) {
            Some(k) if !k.is_empty() => k.clone(),
            _ => written_key(prompt, i, nonce),
        })
        .collect();
    if from == 0 || cold_end {
        keys.extend(pass.iter().skip(below).cloned());
    } else {
        keys.extend((below..prompt.len() / BLOCK_ROWS).map(|i| written_key(prompt, i, nonce)));
    }
    keys
}

/// The keys of the whole blocks of `tokens` below `row` at a turn's end:
/// the prompt's `keys`, then write-path keys of `nonce` for every block they
/// do not cover (the prompt's last chunk and the decode rows).
pub fn turn_keys(tokens: &[u32], keys: &[Vec<u64>], row: usize, nonce: u64) -> Vec<Vec<u64>> {
    let blocks = row.min(tokens.len()) / BLOCK_ROWS;
    (0..blocks)
        .map(|i| match keys.get(i) {
            Some(k) if !k.is_empty() => k.clone(),
            _ => written_key(tokens, i, nonce),
        })
        .collect()
}
