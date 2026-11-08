//! Content-keyed resume states for DFlash2 turns.
//!
//! An entry holds a token lineage and up to two exact resume points on it: the
//! end-of-turn state E (any length) and a capture C at a block-aligned row.
//! Lookup keys on the paged allocator's chained block hashes plus an exact
//! token compare and the cache salt; the owner id plays no part. Bytes are
//! charged to two budgets: a fixed state budget for GDN cells and draft
//! windows, and the parked-KV budget, which is the live headroom the caller
//! passes in at every reserve.
//!
//! Every byte the store holds is an MLX array the allocator already counts, so
//! nothing here registers with `cache_limit::coordinator()`.

use std::sync::Arc;

use mlx_paged_attn::CacheRecency;
use napi::bindgen_prelude::*;

use super::trace::HitKind;
use crate::models::qwen3_5::gdn_checkpoint_store::compute_paged_prefix_block_hashes;

pub(crate) const STATE_BUDGET_ENV: &str = "MLX_DFLASH2_STATE_BUDGET_MB";
pub(crate) const DEFAULT_STATE_BUDGET_MB: u64 = 3072;

const MIB: u64 = 1024 * 1024;

fn parse_state_budget(raw: Option<&str>) -> u64 {
    raw.and_then(|raw| raw.trim().parse::<u64>().ok())
        .unwrap_or(DEFAULT_STATE_BUDGET_MB)
        .saturating_mul(MIB)
}

/// What shrinking a KV source to a row count would release, and what its copy
/// allocates while the old buffer is still alive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Shrink {
    pub(crate) frees: u64,
    pub(crate) transient: u64,
}

/// Target K/V rows `[0, rows())` an entry can resume on.
pub(crate) trait KvSource {
    fn rows(&self) -> usize;
    /// Bytes kept alive, charged to the parked-KV budget.
    fn bytes(&self) -> u64;
    /// `None` when shrinking to `rows` is not worth a copy. `rows == rows()`
    /// asks for compaction of capacity slack.
    fn shrink_plan(&self, rows: usize) -> Option<Shrink>;
    /// Keep only rows `[0, rows)`, releasing what the plan said.
    fn shrink(&mut self, rows: usize) -> Result<()>;
}

/// Bytes a parked state keeps alive, charged to the state budget.
pub(crate) trait StateBytes {
    fn state_bytes(&self) -> u64;
}

/// The concrete payload types of one lane (flat or paged).
pub(crate) trait Backing {
    type Kv: KvSource;
    /// GDN arrays and draft ring at the end of the turn.
    type End: StateBytes;
    /// GDN cell and draft window snapshot at a block-aligned row.
    type Anchor: StateBytes;
}

pub(crate) struct Anchor<A> {
    pub(crate) row: usize,
    pub(crate) state: A,
}

pub(crate) struct EntryParts<B: Backing> {
    pub(crate) end: Option<B::End>,
    pub(crate) anchor: Option<Anchor<B::Anchor>>,
    pub(crate) kv: Option<B::Kv>,
}

/// The prompt or history a key is computed from. Blocks past the end of
/// `extra_keys` carry no extra keys.
#[derive(Clone, Copy)]
pub(crate) struct Lineage<'a> {
    pub(crate) tokens: &'a [u32],
    pub(crate) extra_keys: &'a [Vec<u64>],
    pub(crate) salt: u64,
}

impl<'a> Lineage<'a> {
    pub(crate) fn text(tokens: &'a [u32], salt: u64) -> Self {
        Self {
            tokens,
            extra_keys: &[],
            salt,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct EntryKey {
    pub(crate) block_size: u32,
    pub(crate) depth: usize,
    /// Hash of the last whole block below `depth`; 0 when there is none.
    pub(crate) chain_hash: u64,
    pub(crate) salt: u64,
}

pub(crate) type EntryId = u64;

/// One lossy step of the eviction ladder on one entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct StoreStep {
    pub(crate) entry: EntryId,
    pub(crate) kind: StoreStepKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum StoreStepKind {
    /// Shrink an end state's KV to its anchor and drop the end.
    Demote {
        row: usize,
    },
    DropCapture,
    DropEnd,
}

pub(crate) struct Entry<B: Backing> {
    id: EntryId,
    block_size: u32,
    salt: u64,
    tokens: Vec<u32>,
    chain: Vec<u64>,
    end: Option<B::End>,
    anchor: Option<Anchor<B::Anchor>>,
    kv: Option<B::Kv>,
    state_bytes: u64,
    kv_bytes: u64,
    last_used: u64,
}

impl<B: Backing> Entry<B> {
    pub(crate) fn id(&self) -> EntryId {
        self.id
    }

    /// `End` while the end-of-turn state is held, `Capture` once only the
    /// anchor is left.
    pub(crate) fn kind(&self) -> HitKind {
        if self.end.is_some() {
            HitKind::End
        } else {
            HitKind::Capture
        }
    }

    /// Row of the deepest resume point the entry holds.
    pub(crate) fn depth(&self) -> usize {
        if self.end.is_some() {
            self.tokens.len()
        } else {
            self.anchor.as_ref().map_or(0, |anchor| anchor.row)
        }
    }

    pub(crate) fn key(&self) -> EntryKey {
        let depth = self.depth();
        let blocks = depth / self.block_size as usize;
        EntryKey {
            block_size: self.block_size,
            depth,
            chain_hash: blocks.checked_sub(1).map_or(0, |last| self.chain[last]),
            salt: self.salt,
        }
    }

    pub(crate) fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    pub(crate) fn anchor(&self) -> Option<&Anchor<B::Anchor>> {
        self.anchor.as_ref()
    }

    pub(crate) fn kv(&self) -> Option<&B::Kv> {
        self.kv.as_ref()
    }

    #[cfg(test)]
    pub(crate) fn end(&self) -> Option<&B::End> {
        self.end.as_ref()
    }

    pub(crate) fn state_bytes(&self) -> u64 {
        self.state_bytes
    }

    pub(crate) fn kv_bytes(&self) -> u64 {
        self.kv_bytes
    }

    pub(crate) fn into_parts(self) -> (Vec<u32>, EntryParts<B>) {
        (
            self.tokens,
            EntryParts {
                end: self.end,
                anchor: self.anchor,
                kv: self.kv,
            },
        )
    }

    fn resume_points(&self) -> impl Iterator<Item = (HitKind, usize)> + '_ {
        self.end
            .as_ref()
            .map(|_| (HitKind::End, self.tokens.len()))
            .into_iter()
            .chain(
                self.anchor
                    .as_ref()
                    .map(|anchor| (HitKind::Capture, anchor.row)),
            )
    }

    /// The entry's lineage equals `prompt[..depth]` and its KV covers it.
    fn resumes(&self, prompt: &PromptKey<'_>, depth: usize) -> bool {
        let blocks = depth / self.block_size as usize;
        depth > 0
            && depth < prompt.tokens.len()
            && depth <= self.tokens.len()
            && self.block_size == prompt.block_size
            && self.salt == prompt.salt
            && self.chain.get(..blocks).is_some()
            && self.chain.get(..blocks) == prompt.chain.get(..blocks)
            && self.tokens[..depth] == prompt.tokens[..depth]
            && self.kv.as_ref().is_some_and(|kv| kv.rows() >= depth)
    }

    /// The entry's whole lineage is a prefix of `lineage`.
    fn is_prefix_of(&self, lineage: &PromptKey<'_>) -> bool {
        let blocks = self.chain.len();
        self.block_size == lineage.block_size
            && self.salt == lineage.salt
            && self.tokens.len() <= lineage.tokens.len()
            && lineage.chain.get(..blocks) == Some(&self.chain[..])
            && lineage.tokens[..self.tokens.len()] == self.tokens[..]
    }

    fn refresh_bytes(&mut self) {
        self.state_bytes = self
            .end
            .as_ref()
            .map_or(0, StateBytes::state_bytes)
            .saturating_add(
                self.anchor
                    .as_ref()
                    .map_or(0, |anchor| anchor.state.state_bytes()),
            );
        self.kv_bytes = self.kv.as_ref().map_or(0, KvSource::bytes);
    }
}

struct PromptKey<'a> {
    tokens: &'a [u32],
    chain: Vec<u64>,
    block_size: u32,
    salt: u64,
}

fn block_chain(lineage: Lineage<'_>, block_size: u32) -> Vec<u64> {
    let bs = block_size as usize;
    let blocks = lineage.tokens.len() / bs;
    let Ok(prefix_len) = u32::try_from(blocks * bs) else {
        return Vec::new();
    };
    if blocks == 0 {
        return Vec::new();
    }
    let extra_keys: Vec<Vec<u64>> = (0..blocks)
        .map(|block| lineage.extra_keys.get(block).cloned().unwrap_or_default())
        .collect();
    compute_paged_prefix_block_hashes(
        lineage.tokens,
        prefix_len,
        block_size,
        &extra_keys,
        lineage.salt,
    )
    .unwrap_or_default()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StoreHit {
    pub(crate) id: EntryId,
    /// `End` or `Capture`.
    pub(crate) kind: HitKind,
    pub(crate) depth: usize,
}

impl StoreHit {
    /// The live state wins a tie.
    pub(crate) fn beats_live(&self, live_depth: Option<usize>) -> bool {
        live_depth.is_none_or(|live| self.depth > live)
    }
}

/// Bytes about to be reserved.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Reservation {
    pub(crate) state_bytes: u64,
    pub(crate) kv_bytes: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct EvictReport {
    /// Both budgets cover the reservation.
    pub(crate) satisfied: bool,
    pub(crate) compacted: usize,
    pub(crate) demoted: usize,
    pub(crate) dropped_captures: usize,
    pub(crate) dropped_ends: usize,
    pub(crate) freed_state_bytes: u64,
    pub(crate) freed_kv_bytes: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct StoreBytes {
    pub(crate) entries: usize,
    pub(crate) state_bytes: u64,
    pub(crate) kv_bytes: u64,
}

#[derive(Clone, Copy)]
struct Shortfall {
    state: bool,
    kv: bool,
}

impl Shortfall {
    fn any(self) -> bool {
        self.state || self.kv
    }
}

/// A resume point that eviction removed: whole-block chain hashes plus the
/// tokens past the last whole block, enough to tell a later prompt it would
/// have resumed there.
struct EvictedPoint {
    block_size: u32,
    salt: u64,
    depth: usize,
    chain: Vec<u64>,
    tail: Vec<u32>,
}

/// Evicted resume points remembered for miss attribution, newest last.
const EVICTED_POINTS: usize = 64;

pub(crate) struct HybridStateStore<B: Backing> {
    block_size: u32,
    state_budget: u64,
    entries: Vec<Entry<B>>,
    evicted: std::collections::VecDeque<EvictedPoint>,
    next_id: EntryId,
    recency: Arc<CacheRecency>,
}

impl<B: Backing> HybridStateStore<B> {
    /// State budget from `MLX_DFLASH2_STATE_BUDGET_MB` (default 3072).
    pub(crate) fn new(block_size: u32) -> Self {
        let raw = std::env::var(STATE_BUDGET_ENV).ok();
        Self::with_state_budget(block_size, parse_state_budget(raw.as_deref()))
    }

    pub(crate) fn with_state_budget(block_size: u32, state_budget: u64) -> Self {
        Self {
            block_size: block_size.max(1),
            state_budget,
            entries: Vec::new(),
            evicted: std::collections::VecDeque::new(),
            next_id: 1,
            recency: Arc::new(CacheRecency::new()),
        }
    }

    /// Stamp from `recency` from now on, the clock the model's other caches
    /// share.
    pub(crate) fn share_recency(&mut self, recency: Arc<CacheRecency>) {
        self.recency = recency;
    }

    pub(crate) fn recency(&self) -> Arc<CacheRecency> {
        Arc::clone(&self.recency)
    }

    pub(crate) fn block_size(&self) -> u32 {
        self.block_size
    }

    pub(crate) fn state_budget(&self) -> u64 {
        self.state_budget
    }

    pub(crate) fn bytes(&self) -> StoreBytes {
        StoreBytes {
            entries: self.entries.len(),
            state_bytes: self
                .entries
                .iter()
                .fold(0, |sum, entry| sum.saturating_add(entry.state_bytes)),
            kv_bytes: self
                .entries
                .iter()
                .fold(0, |sum, entry| sum.saturating_add(entry.kv_bytes)),
        }
    }

    #[cfg(test)]
    pub(crate) fn entries(&self) -> impl Iterator<Item = &Entry<B>> {
        self.entries.iter()
    }

    pub(crate) fn get(&self, id: EntryId) -> Option<&Entry<B>> {
        self.entries.iter().find(|entry| entry.id == id)
    }

    fn position(&self, id: EntryId) -> Option<usize> {
        self.entries.iter().position(|entry| entry.id == id)
    }

    fn prompt_key<'a>(&self, lineage: Lineage<'a>) -> PromptKey<'a> {
        PromptKey {
            tokens: lineage.tokens,
            chain: block_chain(lineage, self.block_size),
            block_size: self.block_size,
            salt: lineage.salt,
        }
    }

    fn validate_anchor(&self, row: usize, lineage_len: usize) -> Result<()> {
        if row == 0 || row > lineage_len || !row.is_multiple_of(self.block_size as usize) {
            return Err(Error::from_reason(format!(
                "DFlash2 state anchor at row {row} is not a block boundary of a \
                 {lineage_len}-token lineage (block size {})",
                self.block_size
            )));
        }
        Ok(())
    }

    /// Adds an entry. An entry with an end state drops every older end-state
    /// entry whose whole lineage it extends.
    pub(crate) fn insert(&mut self, lineage: Lineage<'_>, parts: EntryParts<B>) -> Result<EntryId> {
        let last_used = self.recency.next();
        self.insert_at(lineage, parts, last_used)
    }

    /// [`Self::insert`] keeping `last_used`: a parked live state keeps the
    /// stamp of the turn that ended it.
    pub(crate) fn insert_at(
        &mut self,
        lineage: Lineage<'_>,
        parts: EntryParts<B>,
        last_used: u64,
    ) -> Result<EntryId> {
        if parts.end.is_none() && parts.anchor.is_none() {
            return Err(Error::from_reason(
                "DFlash2 state entry holds neither an end state nor an anchor",
            ));
        }
        if lineage.tokens.is_empty() {
            return Err(Error::from_reason("DFlash2 state entry has no tokens"));
        }
        if let Some(anchor) = &parts.anchor {
            self.validate_anchor(anchor.row, lineage.tokens.len())?;
        }
        let key = self.prompt_key(lineage);
        if parts.end.is_some() {
            self.entries
                .retain(|old| old.end.is_none() || !old.is_prefix_of(&key));
        }
        let id = self.next_id;
        self.next_id += 1;
        let mut entry = Entry {
            id,
            block_size: self.block_size,
            salt: lineage.salt,
            tokens: lineage.tokens.to_vec(),
            chain: key.chain,
            end: parts.end,
            anchor: parts.anchor,
            kv: parts.kv,
            state_bytes: 0,
            kv_bytes: 0,
            last_used,
        };
        entry.refresh_bytes();
        self.entries.push(entry);
        Ok(id)
    }

    /// Publishes a capture as the anchor of entry `id`, replacing any older one.
    pub(crate) fn set_anchor(&mut self, id: EntryId, anchor: Anchor<B::Anchor>) -> Result<()> {
        let idx = self
            .position(id)
            .ok_or_else(|| Error::from_reason(format!("DFlash2 state entry {id} is gone")))?;
        self.validate_anchor(anchor.row, self.entries[idx].tokens.len())?;
        let last_used = self.recency.next();
        let entry = &mut self.entries[idx];
        entry.anchor = Some(anchor);
        entry.last_used = last_used;
        entry.refresh_bytes();
        Ok(())
    }

    /// Removes the entry so its buffers can move into the live state.
    pub(crate) fn take(&mut self, id: EntryId) -> Option<Entry<B>> {
        self.position(id).map(|idx| self.entries.remove(idx))
    }

    /// The deepest exact resume point `b < |P|` whose lineage equals
    /// `P[..b]`, whose KV covers `b`, and that `kv_ok` accepts. Ties go to an
    /// end state over a capture, then to the most recently used entry.
    pub(crate) fn lookup(
        &self,
        prompt: Lineage<'_>,
        mut kv_ok: impl FnMut(&Entry<B>) -> bool,
    ) -> Option<StoreHit> {
        let key = self.prompt_key(prompt);
        let mut best: Option<(StoreHit, u64)> = None;
        for entry in &self.entries {
            for (kind, depth) in entry.resume_points() {
                if !entry.resumes(&key, depth) || !kv_ok(entry) {
                    continue;
                }
                let rank = (depth, kind == HitKind::End, entry.last_used);
                let better = best
                    .is_none_or(|(hit, used)| rank > (hit.depth, hit.kind == HitKind::End, used));
                if better {
                    best = Some((
                        StoreHit {
                            id: entry.id,
                            kind,
                            depth,
                        },
                        entry.last_used,
                    ));
                }
            }
        }
        best.map(|(hit, _)| hit)
    }

    /// The deepest evicted resume point `b < |P|` whose lineage equals
    /// `P[..b]`, 0 when there is none. Whole blocks compare by chain hash.
    pub(crate) fn evicted_depth(&self, prompt: Lineage<'_>) -> usize {
        let key = self.prompt_key(prompt);
        self.evicted
            .iter()
            .filter(|point| {
                let floor = point.chain.len() * point.block_size as usize;
                point.depth < key.tokens.len()
                    && point.block_size == key.block_size
                    && point.salt == key.salt
                    && key.chain.get(..point.chain.len()) == Some(&point.chain[..])
                    && key.tokens[floor..point.depth] == point.tail[..]
            })
            .map(|point| point.depth)
            .max()
            .unwrap_or(0)
    }

    fn remember_evicted(&mut self, idx: usize, depth: usize) {
        let entry = &self.entries[idx];
        let blocks = depth / entry.block_size as usize;
        let point = EvictedPoint {
            block_size: entry.block_size,
            salt: entry.salt,
            depth,
            chain: entry.chain[..blocks.min(entry.chain.len())].to_vec(),
            tail: entry.tokens[blocks * entry.block_size as usize..depth].to_vec(),
        };
        if self.evicted.len() == EVICTED_POINTS {
            self.evicted.pop_front();
        }
        self.evicted.push_back(point);
    }

    fn shortfall(&self, need: Reservation, kv_free: u64) -> Shortfall {
        let bytes = self.bytes();
        Shortfall {
            state: bytes.state_bytes.saturating_add(need.state_bytes) > self.state_budget,
            kv: need.kv_bytes > kv_free,
        }
    }

    /// Entry ids of one kind, least recently used first, never `pinned`.
    fn lru_ids(&self, ends: bool, pinned: Option<EntryId>) -> Vec<EntryId> {
        let mut ids: Vec<(u64, EntryId)> = self
            .entries
            .iter()
            .filter(|entry| entry.end.is_some() == ends && Some(entry.id) != pinned)
            .map(|entry| (entry.last_used, entry.id))
            .collect();
        ids.sort_unstable();
        ids.into_iter().map(|(_, id)| id).collect()
    }

    /// Frees bytes until the state budget holds `need.state_bytes` more and
    /// the parked-KV headroom covers `need.kv_bytes`. `kv_headroom` is the
    /// live headroom net of everything the store already holds. Order:
    /// compact end-state KV slack, demote end states to their anchor, drop
    /// captures, drop end states, each least recently used first. `pinned` is
    /// never touched.
    pub(crate) fn evict_for(
        &mut self,
        need: Reservation,
        kv_headroom: u64,
        pinned: Option<EntryId>,
    ) -> Result<EvictReport> {
        let start = self.bytes();
        let kv_free = |store: &Self| {
            kv_headroom.saturating_add(start.kv_bytes.saturating_sub(store.bytes().kv_bytes))
        };
        let mut report = EvictReport::default();

        for id in self.lru_ids(true, pinned) {
            if !self.shortfall(need, kv_free(self)).kv {
                break;
            }
            let free = kv_free(self);
            let Some(idx) = self.position(id) else {
                continue;
            };
            let entry = &mut self.entries[idx];
            let Some(kv) = entry.kv.as_mut() else {
                continue;
            };
            let rows = kv.rows();
            if kv
                .shrink_plan(rows)
                .is_some_and(|plan| plan.frees > 0 && plan.transient <= free)
            {
                kv.shrink(rows)?;
                entry.refresh_bytes();
                report.compacted += 1;
            }
        }

        for id in self.lru_ids(true, pinned) {
            let short = self.shortfall(need, kv_free(self));
            if !short.any() {
                break;
            }
            let free = kv_free(self);
            let Some(idx) = self.position(id) else {
                continue;
            };
            let entry = &mut self.entries[idx];
            let Some(row) = entry.anchor.as_ref().map(|anchor| anchor.row) else {
                continue;
            };
            let kv_plan = entry
                .kv
                .as_ref()
                .and_then(|kv| kv.shrink_plan(row))
                .filter(|plan| plan.frees > 0 && plan.transient <= free);
            let frees_state = entry.end.as_ref().is_some_and(|end| end.state_bytes() > 0);
            if !((short.state && frees_state) || (short.kv && kv_plan.is_some())) {
                continue;
            }
            let end_depth = entry.tokens.len();
            self.remember_evicted(idx, end_depth);
            let entry = &mut self.entries[idx];
            entry.end = None;
            if kv_plan.is_some()
                && let Some(kv) = entry.kv.as_mut()
            {
                kv.shrink(row)?;
            }
            entry.tokens.truncate(row);
            entry.chain.truncate(row / entry.block_size as usize);
            entry.refresh_bytes();
            report.demoted += 1;
        }

        for ends in [false, true] {
            for id in self.lru_ids(ends, pinned) {
                let short = self.shortfall(need, kv_free(self));
                if !short.any() {
                    break;
                }
                let Some(idx) = self.position(id) else {
                    continue;
                };
                let entry = &self.entries[idx];
                if !((short.state && entry.state_bytes > 0) || (short.kv && entry.kv_bytes > 0)) {
                    continue;
                }
                let points: Vec<usize> = entry.resume_points().map(|(_, depth)| depth).collect();
                for depth in points {
                    self.remember_evicted(idx, depth);
                }
                self.entries.remove(idx);
                if ends {
                    report.dropped_ends += 1;
                } else {
                    report.dropped_captures += 1;
                }
            }
        }

        let end = self.bytes();
        report.freed_state_bytes = start.state_bytes.saturating_sub(end.state_bytes);
        report.freed_kv_bytes = start.kv_bytes.saturating_sub(end.kv_bytes);
        report.satisfied = !self.shortfall(need, kv_free(self)).any();
        Ok(report)
    }

    /// Frees parked KV by one step of the [`Self::evict_for`] ladder, for a
    /// caller that re-reads the live headroom and retries after each step:
    /// [`Self::compact_one`], else [`Self::next_lossy`]'s step. `None` when
    /// no step frees KV; `pinned` is never touched.
    pub(crate) fn evict_one(
        &mut self,
        kv_headroom: u64,
        pinned: Option<EntryId>,
    ) -> Result<Option<EvictReport>> {
        if let Some(report) = self.compact_one(kv_headroom, pinned)? {
            return Ok(Some(report));
        }
        match self.next_lossy(kv_headroom, pinned) {
            Some((_, step)) => self.apply_lossy(step).map(Some),
            None => Ok(None),
        }
    }

    /// Compacts one end state's KV slack, least recently used first, when
    /// `kv_headroom` covers the copy. Loses nothing.
    pub(crate) fn compact_one(
        &mut self,
        kv_headroom: u64,
        pinned: Option<EntryId>,
    ) -> Result<Option<EvictReport>> {
        let fits = |plan: &Shrink| plan.frees > 0 && plan.transient <= kv_headroom;
        let compact = self.lru_ids(true, pinned).into_iter().find_map(|id| {
            let idx = self.position(id)?;
            let kv = self.entries[idx].kv.as_ref()?;
            kv.shrink_plan(kv.rows()).filter(fits).map(|_| idx)
        });
        let Some(idx) = compact else {
            return Ok(None);
        };
        let start = self.bytes();
        let entry = &mut self.entries[idx];
        if let Some(kv) = entry.kv.as_mut() {
            let rows = kv.rows();
            kv.shrink(rows)?;
        }
        entry.refresh_bytes();
        let end = self.bytes();
        Ok(Some(EvictReport {
            compacted: 1,
            freed_state_bytes: start.state_bytes.saturating_sub(end.state_bytes),
            freed_kv_bytes: start.kv_bytes.saturating_sub(end.kv_bytes),
            ..EvictReport::default()
        }))
    }

    /// The next lossy step and its entry's `last_used`: demote one end
    /// state whose KV shrinks to its anchor, else drop one capture, else
    /// drop one end state, each least recently used first and holding KV.
    pub(crate) fn next_lossy(
        &self,
        kv_headroom: u64,
        pinned: Option<EntryId>,
    ) -> Option<(u64, StoreStep)> {
        let fits = |plan: &Shrink| plan.frees > 0 && plan.transient <= kv_headroom;
        let demote = self.lru_ids(true, pinned).into_iter().find_map(|id| {
            let entry = &self.entries[self.position(id)?];
            let row = entry.anchor.as_ref()?.row;
            let shrinks = entry
                .kv
                .as_ref()
                .and_then(|kv| kv.shrink_plan(row))
                .filter(fits)
                .is_some();
            shrinks.then_some((
                entry.last_used,
                StoreStep {
                    entry: id,
                    kind: StoreStepKind::Demote { row },
                },
            ))
        });
        demote.or_else(|| {
            [false, true].into_iter().find_map(|ends| {
                self.lru_ids(ends, pinned).into_iter().find_map(|id| {
                    let entry = &self.entries[self.position(id)?];
                    (entry.kv_bytes > 0).then_some((
                        entry.last_used,
                        StoreStep {
                            entry: id,
                            kind: if ends {
                                StoreStepKind::DropEnd
                            } else {
                                StoreStepKind::DropCapture
                            },
                        },
                    ))
                })
            })
        })
    }

    /// Applies a step [`Self::next_lossy`] returned.
    pub(crate) fn apply_lossy(&mut self, step: StoreStep) -> Result<EvictReport> {
        let idx = self.position(step.entry).ok_or_else(|| {
            Error::from_reason(format!("DFlash2 state entry {} is gone", step.entry))
        })?;
        let start = self.bytes();
        let mut report = EvictReport::default();
        match step.kind {
            StoreStepKind::Demote { row } => {
                let end_depth = self.entries[idx].tokens.len();
                self.remember_evicted(idx, end_depth);
                let entry = &mut self.entries[idx];
                entry.end = None;
                if let Some(kv) = entry.kv.as_mut() {
                    kv.shrink(row)?;
                }
                entry.tokens.truncate(row);
                entry.chain.truncate(row / entry.block_size as usize);
                entry.refresh_bytes();
                report.demoted = 1;
            }
            StoreStepKind::DropCapture | StoreStepKind::DropEnd => {
                let points: Vec<usize> = self.entries[idx]
                    .resume_points()
                    .map(|(_, depth)| depth)
                    .collect();
                for depth in points {
                    self.remember_evicted(idx, depth);
                }
                self.entries.remove(idx);
                if step.kind == StoreStepKind::DropEnd {
                    report.dropped_ends = 1;
                } else {
                    report.dropped_captures = 1;
                }
            }
        }
        let end = self.bytes();
        report.freed_state_bytes = start.state_bytes.saturating_sub(end.state_bytes);
        report.freed_kv_bytes = start.kv_bytes.saturating_sub(end.kv_bytes);
        Ok(report)
    }

    /// Drops every entry and every remembered eviction (`ResetScope::Command`).
    pub(crate) fn clear(&mut self) {
        self.entries.clear();
        self.evicted.clear();
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use super::*;

    type Log = Rc<RefCell<Vec<String>>>;

    struct TestKv {
        name: &'static str,
        rows: usize,
        capacity: usize,
        row_bytes: u64,
        log: Log,
    }

    impl KvSource for TestKv {
        fn rows(&self) -> usize {
            self.rows
        }

        fn bytes(&self) -> u64 {
            self.capacity as u64 * self.row_bytes
        }

        fn shrink_plan(&self, rows: usize) -> Option<Shrink> {
            let rows = rows.min(self.rows);
            (rows < self.capacity).then(|| Shrink {
                frees: (self.capacity - rows) as u64 * self.row_bytes,
                transient: rows as u64 * self.row_bytes,
            })
        }

        fn shrink(&mut self, rows: usize) -> Result<()> {
            self.rows = rows.min(self.rows);
            self.capacity = self.rows;
            self.log
                .borrow_mut()
                .push(format!("shrink {} {}", self.name, self.rows));
            Ok(())
        }
    }

    struct TestState {
        name: String,
        bytes: u64,
        log: Log,
    }

    impl StateBytes for TestState {
        fn state_bytes(&self) -> u64 {
            self.bytes
        }
    }

    impl Drop for TestState {
        fn drop(&mut self) {
            self.log.borrow_mut().push(format!("drop {}", self.name));
        }
    }

    struct TestBacking;

    impl Backing for TestBacking {
        type Kv = TestKv;
        type End = TestState;
        type Anchor = TestState;
    }

    type Store = HybridStateStore<TestBacking>;

    const BS: u32 = 16;
    const GDN: u64 = 100;
    const ANCHOR: u64 = 60;
    const ROW: u64 = 10;

    fn lineage(seed: u32, len: usize) -> Vec<u32> {
        (0..len as u32).map(|i| seed * 100_000 + i).collect()
    }

    fn fork(base: &[u32], at: usize, len: usize, seed: u32) -> Vec<u32> {
        let mut tokens = base[..at].to_vec();
        tokens.extend((0..(len - at) as u32).map(|i| seed * 100_000 + 50_000 + i));
        tokens
    }

    struct Fixture {
        log: Log,
    }

    impl Fixture {
        fn new() -> Self {
            Self { log: Rc::default() }
        }

        fn state(&self, name: String, bytes: u64) -> TestState {
            TestState {
                name,
                bytes,
                log: self.log.clone(),
            }
        }

        /// An entry with an end state over the whole lineage, flat KV with
        /// `slack` spare rows, and an optional anchor.
        fn end(
            &self,
            name: &'static str,
            tokens: &[u32],
            anchor: Option<usize>,
            slack: usize,
        ) -> EntryParts<TestBacking> {
            EntryParts {
                end: Some(self.state(format!("end {name}"), GDN)),
                anchor: anchor.map(|row| Anchor {
                    row,
                    state: self.state(format!("anchor {name}"), ANCHOR),
                }),
                kv: Some(self.kv(name, tokens.len(), slack)),
            }
        }

        fn capture(&self, name: &'static str, row: usize) -> EntryParts<TestBacking> {
            EntryParts {
                end: None,
                anchor: Some(Anchor {
                    row,
                    state: self.state(format!("anchor {name}"), ANCHOR),
                }),
                kv: Some(self.kv(name, row, 0)),
            }
        }

        fn kv(&self, name: &'static str, rows: usize, slack: usize) -> TestKv {
            TestKv {
                name,
                rows,
                capacity: rows + slack,
                row_bytes: ROW,
                log: self.log.clone(),
            }
        }

        fn drain(&self) -> Vec<String> {
            std::mem::take(&mut *self.log.borrow_mut())
        }
    }

    fn hit(store: &Store, prompt: &[u32], salt: u64) -> Option<(HitKind, usize)> {
        store
            .lookup(Lineage::text(prompt, salt), |_| true)
            .map(|hit| (hit.kind, hit.depth))
    }

    #[test]
    fn deepest_exact_resume_across_end_states_and_captures() {
        let fx = Fixture::new();
        let mut store = Store::with_state_budget(BS, u64::MAX);
        let a = lineage(1, 100);
        let b = lineage(2, 300);
        let a_id = store
            .insert(Lineage::text(&a, 0), fx.end("a", &a, Some(64), 0))
            .unwrap();
        store
            .insert(Lineage::text(&b, 0), fx.end("b", &b, Some(160), 0))
            .unwrap();
        let b_capture = lineage(2, 240);
        store
            .insert(Lineage::text(&b_capture, 0), fx.capture("b240", 240))
            .unwrap();

        let a_next = [a.clone(), lineage(9, 20)].concat();
        assert_eq!(hit(&store, &a_next, 0), Some((HitKind::End, 100)));
        // Re-render of A's generation: diverges past its anchor, before its end.
        assert_eq!(
            hit(&store, &fork(&a, 80, 120, 3), 0),
            Some((HitKind::Capture, 64))
        );
        // Diverges before A's anchor: nothing exact.
        assert_eq!(hit(&store, &fork(&a, 40, 120, 3), 0), None);
        // B lineage: the deeper standalone capture beats B's own anchor.
        assert_eq!(
            hit(&store, &fork(&b, 250, 400, 4), 0),
            Some((HitKind::Capture, 240))
        );
        assert_eq!(
            hit(&store, &fork(&b, 200, 400, 4), 0),
            Some((HitKind::Capture, 160))
        );
        assert_eq!(
            hit(&store, &[b.clone(), vec![7]].concat(), 0),
            Some((HitKind::End, 300))
        );
        // `b < |P|`: a prompt equal to a whole history cannot resume at its end.
        assert_eq!(hit(&store, &a, 0), Some((HitKind::Capture, 64)));
        // `kv_ok` vetoes an entry.
        let vetoed = store.lookup(Lineage::text(&a_next, 0), |entry| entry.id() != a_id);
        assert_eq!(vetoed, None);
    }

    #[test]
    fn equal_depth_prefers_end_then_recent_and_live_wins_ties() {
        let fx = Fixture::new();
        let mut store = Store::with_state_budget(BS, u64::MAX);
        let base = lineage(1, 64);
        let longer = lineage(1, 96);
        store
            .insert(Lineage::text(&longer, 0), fx.capture("c64", 64))
            .unwrap();
        let end_id = store
            .insert(Lineage::text(&base, 0), fx.end("e64", &base, None, 0))
            .unwrap();
        let prompt = [base.clone(), vec![1, 2]].concat();
        let found = store.lookup(Lineage::text(&prompt, 0), |_| true).unwrap();
        assert_eq!(
            (found.id, found.kind, found.depth),
            (end_id, HitKind::End, 64)
        );
        assert!(!found.beats_live(Some(64)));
        assert!(found.beats_live(Some(63)));
        assert!(found.beats_live(None));

        let newer = store.recency().next() + 10;
        let c1 = store
            .insert_at(Lineage::text(&longer, 0), fx.capture("c1", 32), newer)
            .unwrap();
        store
            .insert(Lineage::text(&longer, 0), fx.capture("c2", 32))
            .unwrap();
        let probe = fork(&longer, 40, 80, 5);
        assert_eq!(
            store.lookup(Lineage::text(&probe, 0), |_| true).unwrap().id,
            c1
        );
    }

    /// A parked live state keeps the stamp of the turn that ended it, so it
    /// ranks older than an entry inserted after that turn.
    /// Mutation: stamp `self.recency.next()` in `insert_at` → red.
    #[test]
    fn park_keeps_the_turn_end_stamp() {
        let fx = Fixture::new();
        let mut store = Store::with_state_budget(BS, u64::MAX);
        let parked_tokens = lineage(1, 64);
        let other_tokens = lineage(2, 64);
        let turn_end = store.recency().next();
        store
            .insert(
                Lineage::text(&other_tokens, 0),
                fx.end("other", &other_tokens, None, 0),
            )
            .unwrap();
        let parked = store
            .insert_at(
                Lineage::text(&parked_tokens, 0),
                fx.end("parked", &parked_tokens, None, 0),
                turn_end,
            )
            .unwrap();
        let (stamp, step) = store.next_lossy(0, None).expect("a lossy step");
        assert_eq!((stamp, step.entry), (turn_end, parked));
    }

    #[test]
    fn salt_isolates_lineages() {
        let fx = Fixture::new();
        let mut store = Store::with_state_budget(BS, u64::MAX);
        let a = lineage(1, 100);
        store
            .insert(Lineage::text(&a, 7), fx.end("a", &a, Some(64), 0))
            .unwrap();
        let prompt = [a.clone(), vec![1]].concat();
        assert_eq!(hit(&store, &prompt, 7), Some((HitKind::End, 100)));
        assert_eq!(hit(&store, &prompt, 0), None);
        assert_eq!(hit(&store, &prompt, 8), None);
    }

    #[test]
    fn extra_keys_split_lineages_with_equal_tokens() {
        let fx = Fixture::new();
        let mut store = Store::with_state_budget(BS, u64::MAX);
        let a = lineage(1, 100);
        let image = vec![vec![], vec![42]];
        let keyed = Lineage {
            tokens: &a,
            extra_keys: &image,
            salt: 0,
        };
        store.insert(keyed, fx.end("a", &a, Some(16), 0)).unwrap();
        let prompt = [a.clone(), vec![1]].concat();
        let probe = Lineage {
            tokens: &prompt,
            extra_keys: &image,
            salt: 0,
        };
        assert_eq!(
            store.lookup(probe, |_| true).map(|hit| hit.depth),
            Some(100)
        );
        // Without the image key only the block before it still lines up.
        assert_eq!(hit(&store, &prompt, 0), Some((HitKind::Capture, 16)));
    }

    #[test]
    fn lookup_ignores_which_session_parked_the_entry() {
        let fx = Fixture::new();
        let mut store = Store::with_state_budget(BS, u64::MAX);
        let header = lineage(1, 48);
        let main = fork(&header, 48, 200, 2);
        let sub = fork(&header, 48, 150, 3);
        let main_id = store
            .insert(Lineage::text(&main, 0), fx.end("main", &main, None, 0))
            .unwrap();
        let sub_id = store
            .insert(Lineage::text(&sub, 0), fx.end("sub", &sub, None, 0))
            .unwrap();
        // A fresh request (new owner) carrying either history resumes on it.
        let main_next = [main.clone(), vec![9]].concat();
        let sub_next = [sub.clone(), vec![9]].concat();
        assert_eq!(
            store
                .lookup(Lineage::text(&main_next, 0), |_| true)
                .unwrap()
                .id,
            main_id
        );
        assert_eq!(
            store
                .lookup(Lineage::text(&sub_next, 0), |_| true)
                .unwrap()
                .id,
            sub_id
        );
        assert_eq!(
            store
                .lookup(Lineage::text(&sub_next, 0), |_| true)
                .unwrap()
                .id,
            sub_id
        );
    }

    #[test]
    fn a_new_end_state_drops_the_end_states_it_extends() {
        let fx = Fixture::new();
        let mut store = Store::with_state_budget(BS, u64::MAX);
        let a1 = lineage(1, 100);
        let a2 = [a1.clone(), lineage(2, 50)].concat();
        let other = lineage(3, 100);
        let a1_id = store
            .insert(Lineage::text(&a1, 0), fx.end("a1", &a1, Some(64), 0))
            .unwrap();
        let other_id = store
            .insert(Lineage::text(&other, 0), fx.end("o", &other, None, 0))
            .unwrap();
        let shorter = lineage(1, 64);
        let capture_id = store
            .insert(Lineage::text(&shorter, 0), fx.capture("c", 64))
            .unwrap();
        fx.drain();

        let a2_id = store
            .insert(Lineage::text(&a2, 0), fx.end("a2", &a2, Some(128), 0))
            .unwrap();
        assert!(store.get(a1_id).is_none());
        assert_eq!(fx.drain(), ["drop end a1", "drop anchor a1"]);
        assert!(store.get(other_id).is_some());
        assert!(
            store.get(capture_id).is_some(),
            "a capture is not an end state"
        );
        assert!(store.get(a2_id).is_some());

        // A shorter end state on the same lineage dominates nothing.
        let a0 = lineage(1, 90);
        store
            .insert(Lineage::text(&a0, 0), fx.end("a0", &a0, None, 0))
            .unwrap();
        assert!(store.get(a2_id).is_some());
        assert_eq!(store.bytes().entries, 4);
    }

    #[test]
    fn state_budget_eviction_order() {
        let fx = Fixture::new();
        // Room for exactly the five states below.
        let budget = 3 * GDN + 2 * ANCHOR + ANCHOR;
        let mut store = Store::with_state_budget(BS, budget);
        let e_old = lineage(1, 100);
        let e_mid = lineage(2, 100);
        let e_new = lineage(3, 100);
        let c = lineage(4, 64);
        store
            .insert(
                Lineage::text(&e_old, 0),
                fx.end("old", &e_old, Some(64), 5000),
            )
            .unwrap();
        store
            .insert(Lineage::text(&c, 0), fx.capture("c", 64))
            .unwrap();
        store
            .insert(Lineage::text(&e_mid, 0), fx.end("mid", &e_mid, None, 0))
            .unwrap();
        store
            .insert(Lineage::text(&e_new, 0), fx.end("new", &e_new, Some(32), 0))
            .unwrap();
        assert_eq!(store.bytes().state_bytes, budget);
        fx.drain();

        // State pressure only: no compaction, demotions first, LRU first.
        let need = |state_bytes| Reservation {
            state_bytes,
            kv_bytes: 0,
        };
        let r = store.evict_for(need(GDN), u64::MAX, None).unwrap();
        assert_eq!(fx.drain(), ["drop end old", "shrink old 64"]);
        assert_eq!((r.satisfied, r.compacted, r.demoted), (true, 0, 1));
        assert_eq!(r.freed_state_bytes, GDN);

        let r = store
            .evict_for(need(2 * GDN + ANCHOR), u64::MAX, None)
            .unwrap();
        assert_eq!(
            fx.drain(),
            ["drop end new", "shrink new 32", "drop anchor old"]
        );
        assert_eq!((r.demoted, r.dropped_captures, r.dropped_ends), (1, 1, 0));
        assert!(r.satisfied);

        let r = store.evict_for(need(budget), u64::MAX, None).unwrap();
        assert_eq!(
            fx.drain(),
            ["drop anchor c", "drop anchor new", "drop end mid"]
        );
        assert_eq!((r.dropped_captures, r.dropped_ends), (2, 1));
        assert!(r.satisfied);
        assert_eq!(store.bytes(), StoreBytes::default());

        let r = store.evict_for(need(budget + 1), u64::MAX, None).unwrap();
        assert!(!r.satisfied, "nothing left to evict");
    }

    #[test]
    fn kv_headroom_eviction_order() {
        let fx = Fixture::new();
        let mut store = Store::with_state_budget(BS, u64::MAX);
        let e_old = lineage(1, 100);
        let e_new = lineage(2, 100);
        let c = lineage(3, 64);
        store
            .insert(
                Lineage::text(&e_old, 0),
                fx.end("old", &e_old, Some(64), 400),
            )
            .unwrap();
        store
            .insert(Lineage::text(&c, 0), fx.capture("c", 64))
            .unwrap();
        store
            .insert(Lineage::text(&e_new, 0), fx.end("new", &e_new, None, 0))
            .unwrap();
        fx.drain();
        let kv = |kv_bytes| Reservation {
            state_bytes: 0,
            kv_bytes,
        };

        // Compaction of old's slack (400 rows) covers it; states untouched.
        let r = store.evict_for(kv(300 * ROW), 100 * ROW, None).unwrap();
        assert_eq!(fx.drain(), ["shrink old 100"]);
        assert_eq!(
            (r.satisfied, r.compacted, r.freed_kv_bytes),
            (true, 1, 400 * ROW)
        );
        assert_eq!(r.freed_state_bytes, 0);

        // Next: demotion trims old's KV to its anchor before any capture goes.
        let r = store.evict_for(kv(100 * ROW), 64 * ROW, None).unwrap();
        assert_eq!(fx.drain(), ["drop end old", "shrink old 64"]);
        assert_eq!(
            (r.demoted, r.dropped_captures, r.freed_kv_bytes),
            (1, 0, 36 * ROW)
        );

        let r = store.evict_for(kv(100 * ROW), 0, None).unwrap();
        assert_eq!(fx.drain(), ["drop anchor old", "drop anchor c"]);
        assert_eq!((r.dropped_captures, r.dropped_ends), (2, 0));

        let r = store.evict_for(kv(100 * ROW), 0, None).unwrap();
        assert_eq!(fx.drain(), ["drop end new"]);
        assert!(r.satisfied);
    }

    /// Each `evict_one` call takes exactly one step of the ladder, in the
    /// order `evict_for` frees: compact, demote, drop captures, drop end
    /// states, least recently used first; `None` once nothing is left.
    /// Mutation: run `self.evict_for` with an unbounded need inside
    /// `evict_one` → red (the first call empties the store).
    #[test]
    fn store_evict_one_takes_one_lru_step() {
        let fx = Fixture::new();
        let mut store = Store::with_state_budget(BS, u64::MAX);
        let e_old = lineage(1, 100);
        let e_new = lineage(2, 100);
        let c = lineage(3, 64);
        store
            .insert(
                Lineage::text(&e_old, 0),
                fx.end("old", &e_old, Some(64), 400),
            )
            .unwrap();
        store
            .insert(Lineage::text(&c, 0), fx.capture("c", 64))
            .unwrap();
        store
            .insert(Lineage::text(&e_new, 0), fx.end("new", &e_new, None, 0))
            .unwrap();
        fx.drain();
        let mut step = |want: &[&str], entries: usize| {
            let report = store
                .evict_one(100 * ROW, None)
                .unwrap()
                .expect("a step frees bytes");
            assert_eq!(fx.drain(), want);
            assert_eq!(store.bytes().entries, entries, "{want:?}");
            report
        };
        let r = step(&["shrink old 100"], 3);
        assert_eq!((r.compacted, r.freed_kv_bytes), (1, 400 * ROW));
        let r = step(&["drop end old", "shrink old 64"], 3);
        assert_eq!((r.demoted, r.freed_kv_bytes), (1, 36 * ROW));
        let r = step(&["drop anchor old"], 2);
        assert_eq!(r.dropped_captures, 1);
        step(&["drop anchor c"], 1);
        let r = step(&["drop end new"], 0);
        assert_eq!(r.dropped_ends, 1);
        assert!(store.evict_one(u64::MAX, None).unwrap().is_none());
    }

    #[test]
    fn compaction_needs_headroom_for_its_copy() {
        let fx = Fixture::new();
        let mut store = Store::with_state_budget(BS, u64::MAX);
        let e = lineage(1, 100);
        store
            .insert(Lineage::text(&e, 0), fx.end("e", &e, None, 400))
            .unwrap();
        fx.drain();
        let need = Reservation {
            state_bytes: 0,
            kv_bytes: 200 * ROW,
        };
        // The 100-row copy does not fit 99 rows of headroom: drop instead.
        let r = store.evict_for(need, 99 * ROW, None).unwrap();
        assert_eq!((r.compacted, r.dropped_ends), (0, 1));
        assert_eq!(fx.drain(), ["drop end e"]);

        store
            .insert(Lineage::text(&e, 0), fx.end("e", &e, None, 400))
            .unwrap();
        let r = store.evict_for(need, 100 * ROW, None).unwrap();
        assert_eq!((r.compacted, r.dropped_ends, r.satisfied), (1, 0, true));
    }

    #[test]
    fn shrinking_headroom_evicts_end_kv_before_capture_states() {
        let fx = Fixture::new();
        let mut store = Store::with_state_budget(BS, u64::MAX);
        let a = lineage(1, 200);
        let b = lineage(2, 200);
        let c = lineage(3, 64);
        store
            .insert(Lineage::text(&a, 0), fx.end("a", &a, Some(64), 100))
            .unwrap();
        store
            .insert(Lineage::text(&c, 0), fx.capture("c", 64))
            .unwrap();
        store
            .insert(Lineage::text(&b, 0), fx.end("b", &b, Some(128), 100))
            .unwrap();
        fx.drain();
        let need = Reservation {
            state_bytes: 0,
            kv_bytes: 1000 * ROW,
        };
        let mut steps = Vec::new();
        for headroom_rows in (0..=1000).rev().step_by(100) {
            store.evict_for(need, headroom_rows * ROW, None).unwrap();
            steps.extend(fx.drain());
        }
        let first_capture_drop = steps.iter().position(|s| s.starts_with("drop anchor"));
        let last_end_kv = steps.iter().rposition(|s| s.starts_with("shrink"));
        assert!(
            first_capture_drop.is_none_or(|drop| last_end_kv.is_some_and(|kv| kv < drop)),
            "end-state KV must go before any capture state: {steps:?}"
        );
        assert_eq!(
            &steps[..6],
            [
                "shrink a 200",
                "shrink b 200",
                "drop end a",
                "shrink a 64",
                "drop end b",
                "shrink b 128"
            ]
        );
    }

    /// The KV budget is 9/10 of the headroom, so a freed byte raises it by
    /// 9/10 of a byte: `dflash2_kv_headroom_for` is the least headroom whose
    /// budget covers a need. Mutation: `budget / 9` for `budget.div_ceil(9)`
    /// in `dflash2_kv_headroom_for` → red.
    #[test]
    fn kv_headroom_for_is_the_least_covering_headroom() {
        use crate::models::qwen3_5::dflash2_decode::{
            dflash2_kv_budget_of, dflash2_kv_headroom_for,
        };
        for budget in (0..2000).chain([u64::MAX / 20, 9 * GDN * ROW + 1]) {
            let headroom = dflash2_kv_headroom_for(budget);
            assert!(dflash2_kv_budget_of(headroom) >= budget, "{budget}");
            assert!(
                headroom == 0 || dflash2_kv_budget_of(headroom - 1) < budget,
                "{budget}: {headroom} is not the least"
            );
        }
    }

    /// Eviction remembers each resume point it removes: a demotion the end
    /// state, a drop every point the entry held. A prompt on that lineage
    /// reports the deepest one; another lineage, another salt, or a prompt
    /// no longer than the point reports 0, and a command reset forgets them.
    #[test]
    fn evicted_resume_points_are_remembered_per_lineage() {
        let fx = Fixture::new();
        // Room for one end state and both anchors.
        let mut small = Store::with_state_budget(BS, GDN + 2 * ANCHOR);
        let a = lineage(1, 100);
        let b = lineage(2, 70);
        small
            .insert(Lineage::text(&a, 0), fx.end("a", &a, Some(64), 0))
            .unwrap();
        small
            .insert(Lineage::text(&b, 0), fx.end("b", &b, Some(48), 0))
            .unwrap();
        let next = |tokens: &[u32]| [tokens, &[7, 7]].concat();
        assert_eq!(small.evicted_depth(Lineage::text(&next(&a), 0)), 0);

        let state = |state_bytes| Reservation {
            state_bytes,
            kv_bytes: 0,
        };
        // a's end state goes first (LRU): its anchor stays.
        small.evict_for(state(0), u64::MAX, None).unwrap();
        assert_eq!(fx.drain(), ["drop end a", "shrink a 64"]);
        assert_eq!(small.evicted_depth(Lineage::text(&next(&a), 0)), 100);
        assert_eq!(small.evicted_depth(Lineage::text(&a, 0)), 0, "b < |P|");
        assert_eq!(small.evicted_depth(Lineage::text(&next(&a), 1)), 0, "salt");
        assert_eq!(small.evicted_depth(Lineage::text(&next(&b), 0)), 0);
        let rerender = fork(&a, 80, 100, 9);
        assert_eq!(small.evicted_depth(Lineage::text(&rerender, 0)), 0);

        // Dropping everything remembers b's end state and both anchors.
        small
            .evict_for(state(u64::MAX / 2), u64::MAX, None)
            .unwrap();
        assert_eq!(small.bytes().entries, 0);
        assert_eq!(small.evicted_depth(Lineage::text(&rerender, 0)), 64);
        assert_eq!(small.evicted_depth(Lineage::text(&next(&b), 0)), 70);
        assert_eq!(
            small.evicted_depth(Lineage::text(&fork(&b, 50, 70, 9), 0)),
            48
        );

        small.clear();
        assert_eq!(small.evicted_depth(Lineage::text(&next(&b), 0)), 0);

        for seed in 0..EVICTED_POINTS as u32 + 1 {
            let c = lineage(10 + seed, 40);
            small
                .insert(Lineage::text(&c, 0), fx.capture("c", 32))
                .unwrap();
            small
                .evict_for(state(u64::MAX / 2), u64::MAX, None)
                .unwrap();
        }
        assert_eq!(small.evicted.len(), EVICTED_POINTS);
        let oldest = [&lineage(10, 40)[..32], &[1, 2]].concat();
        assert_eq!(small.evicted_depth(Lineage::text(&oldest, 0)), 0);
        let newest = [&lineage(10 + EVICTED_POINTS as u32, 40)[..32], &[1, 2]].concat();
        assert_eq!(small.evicted_depth(Lineage::text(&newest, 0)), 32);
    }

    #[test]
    fn byte_accounting_tracks_inserts_and_removes() {
        let fx = Fixture::new();
        let mut store = Store::with_state_budget(BS, u64::MAX);
        let a = lineage(1, 100);
        let b = lineage(2, 64);
        let a_id = store
            .insert(Lineage::text(&a, 0), fx.end("a", &a, Some(64), 28))
            .unwrap();
        assert_eq!(
            store.bytes(),
            StoreBytes {
                entries: 1,
                state_bytes: GDN + ANCHOR,
                kv_bytes: 128 * ROW,
            }
        );
        let entry = store.get(a_id).unwrap();
        assert_eq!(
            (entry.state_bytes(), entry.kv_bytes()),
            (GDN + ANCHOR, 128 * ROW)
        );
        let b_id = store
            .insert(Lineage::text(&b, 0), fx.capture("b", 64))
            .unwrap();
        assert_eq!(store.bytes().state_bytes, GDN + 2 * ANCHOR);
        assert_eq!(store.bytes().kv_bytes, 192 * ROW);

        let a_entry = store.take(a_id).unwrap();
        assert_eq!(store.bytes().state_bytes, ANCHOR);
        assert_eq!(store.bytes().kv_bytes, 64 * ROW);
        let (tokens, parts) = a_entry.into_parts();
        assert_eq!(tokens, a);
        assert_eq!(parts.kv.unwrap().rows, 100);

        store
            .set_anchor(
                b_id,
                Anchor {
                    row: 48,
                    state: fx.state("b48".into(), 2 * ANCHOR),
                },
            )
            .unwrap();
        assert_eq!(store.bytes().state_bytes, 2 * ANCHOR);
        assert!(store.take(a_id).is_none());
        store.clear();
        assert_eq!(store.bytes(), StoreBytes::default());
    }

    #[test]
    fn pinned_entry_survives_eviction() {
        let fx = Fixture::new();
        let mut store = Store::with_state_budget(BS, 0);
        let a = lineage(1, 100);
        let b = lineage(2, 100);
        let a_id = store
            .insert(Lineage::text(&a, 0), fx.end("a", &a, Some(64), 500))
            .unwrap();
        let b_id = store
            .insert(Lineage::text(&b, 0), fx.end("b", &b, Some(64), 500))
            .unwrap();
        fx.drain();
        let need = Reservation {
            state_bytes: 1,
            kv_bytes: u64::MAX,
        };
        let r = store.evict_for(need, 0, Some(a_id)).unwrap();
        assert!(!r.satisfied);
        assert!(store.get(b_id).is_none());
        let pinned = store.get(a_id).unwrap();
        assert_eq!(pinned.kind(), HitKind::End);
        assert_eq!(pinned.kv().unwrap().capacity, 600);
        assert!(
            fx.drain()
                .iter()
                .all(|step| !step.ends_with(" a") && !step.contains(" a "))
        );
    }

    #[test]
    fn inserting_leaves_cache_limit_totals_unchanged() {
        let fx = Fixture::new();
        let before = crate::cache_limit::coordinator().registered_pool_bytes();
        let mut store = Store::with_state_budget(BS, u64::MAX);
        let a = lineage(1, 100);
        let mut parts = fx.end("a", &a, Some(64), 0);
        parts.kv.as_mut().unwrap().row_bytes = 1 << 30;
        store.insert(Lineage::text(&a, 0), parts).unwrap();
        assert!(store.bytes().kv_bytes >= 100 << 30);
        assert_eq!(
            crate::cache_limit::coordinator().registered_pool_bytes(),
            before
        );
    }

    #[test]
    fn block_size_32_keys_match_the_allocator_chain() {
        let fx = Fixture::new();
        let mut store = Store::with_state_budget(32, GDN + ANCHOR);
        assert_eq!(store.block_size(), 32);
        let a = lineage(1, 100);
        let id = store
            .insert(Lineage::text(&a, 5), fx.end("a", &a, Some(64), 0))
            .unwrap();
        let expected =
            compute_paged_prefix_block_hashes(&a, 96, 32, &vec![Vec::new(); 3], 5).unwrap();
        let entry = store.get(id).unwrap();
        assert_eq!(entry.chain, expected);
        assert_eq!(
            entry.key(),
            EntryKey {
                block_size: 32,
                depth: 100,
                chain_hash: expected[2],
                salt: 5,
            }
        );
        assert!(
            store
                .insert(Lineage::text(&a, 5), fx.capture("bad", 48))
                .is_err(),
            "48 is a 16-boundary, not a 32-boundary"
        );
        assert_eq!(
            hit(&store, &fork(&a, 70, 120, 2), 5),
            Some((HitKind::Capture, 64))
        );
        assert_eq!(hit(&store, &fork(&a, 60, 120, 2), 5), None);
        // Demotion keeps the chain consistent with the 32-row allocator chain.
        store
            .evict_for(
                Reservation {
                    state_bytes: GDN,
                    kv_bytes: 0,
                },
                0,
                None,
            )
            .unwrap();
        let entry = store.get(id).unwrap();
        assert_eq!(entry.key().chain_hash, expected[1]);
        assert_eq!(entry.chain, expected[..2]);
        assert_eq!(entry.tokens(), &a[..64]);
        assert_eq!(entry.kind(), HitKind::Capture);
    }

    #[test]
    fn lookup_needs_kv_covering_the_resume_row() {
        let fx = Fixture::new();
        let mut store = Store::with_state_budget(BS, u64::MAX);
        let a = lineage(1, 100);
        let mut short = fx.end("a", &a, Some(64), 0);
        short.kv.as_mut().unwrap().rows = 80;
        store.insert(Lineage::text(&a, 0), short).unwrap();
        let prompt = [a.clone(), vec![1]].concat();
        assert_eq!(hit(&store, &prompt, 0), Some((HitKind::Capture, 64)));
        store.clear();
        let mut missing = fx.end("a", &a, Some(64), 0);
        missing.kv = None;
        store.insert(Lineage::text(&a, 0), missing).unwrap();
        assert_eq!(hit(&store, &prompt, 0), None);
    }

    #[test]
    fn hash_collision_with_different_tokens_misses() {
        let fx = Fixture::new();
        let mut store = Store::with_state_budget(BS, u64::MAX);
        let a = lineage(1, 100);
        let b = lineage(2, 100);
        let id = store
            .insert(Lineage::text(&a, 0), fx.end("a", &a, Some(64), 0))
            .unwrap();
        let idx = store.position(id).unwrap();
        // Forge a collision: A's entry now carries B's block hashes.
        store.entries[idx].chain = block_chain(Lineage::text(&b, 0), BS);
        let prompt = [b.clone(), vec![1]].concat();
        assert_eq!(hit(&store, &prompt, 0), None);
    }

    #[test]
    fn inserts_reject_entries_without_a_resume_point() {
        let fx = Fixture::new();
        let mut store = Store::with_state_budget(BS, u64::MAX);
        let a = lineage(1, 100);
        let empty = EntryParts::<TestBacking> {
            end: None,
            anchor: None,
            kv: Some(fx.kv("a", 100, 0)),
        };
        assert!(store.insert(Lineage::text(&a, 0), empty).is_err());
        assert!(
            store
                .insert(Lineage::text(&[], 0), fx.end("a", &[], None, 0))
                .is_err()
        );
        assert!(
            store
                .insert(Lineage::text(&a, 0), fx.capture("past", 112))
                .is_err()
        );
        let id = store
            .insert(Lineage::text(&a, 0), fx.end("a", &a, None, 0))
            .unwrap();
        let anchor = |row| Anchor {
            row,
            state: fx.state("x".into(), 1),
        };
        assert!(store.set_anchor(id, anchor(0)).is_err());
        assert!(store.set_anchor(id, anchor(40)).is_err());
        assert!(store.set_anchor(id + 1, anchor(64)).is_err());
        store.set_anchor(id, anchor(96)).unwrap();
        assert_eq!(store.get(id).unwrap().anchor().unwrap().row, 96);
    }

    #[test]
    fn state_budget_reads_megabytes_with_a_3072_default() {
        assert_eq!(parse_state_budget(None), 3072 * MIB);
        assert_eq!(parse_state_budget(Some("512")), 512 * MIB);
        assert_eq!(parse_state_budget(Some(" 0 ")), 0);
        assert_eq!(parse_state_budget(Some("lots")), 3072 * MIB);
        let raw = std::env::var(STATE_BUDGET_ENV).ok();
        let store = Store::new(16);
        assert_eq!(store.state_budget(), parse_state_budget(raw.as_deref()));
        assert_eq!(store.block_size(), 16);
    }
}
