//! Shared governor-driven SSD state-disk engine (`StateDiskEngine<C>`) and
//! model-neutral state-store disk primitives (`Tick`, `HeldTier`, `adopt_states`,
//! `WriteBehind`, `label_of`, `lineage_keys`, `ReadWait`, `ChainCursor`).
//!
//! Shared by Pattern A families (`qwen3_5/dflash2_state` and `inkling2/state_disk`)
//! and used by `engine/hybrid_scheduler.rs` without depending on any model family.
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

use std::any::Any;
use std::collections::{HashSet, VecDeque};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use mlx_paged_attn::{
    BlockAllocator, ColdCacheFingerprint, ColdCacheKey, ColdGroup, ColdKeyHold, ColdStateLabel,
    ColdStateLayout, ColdStateObject, EnqueueRefused, Landed, LayerKVPool, MakeRoom,
    state_object_bytes,
};

use crate::disk_tier::DiskTier;
use crate::disk_tier::store::{
    DiskCopy, Entry, EntryId, HybridStateStore, Lineage, PART_PAGE_ROWS, StateBytes, StateClass,
};
use crate::transformer::paged_kv_cache_adapter::{
    ColdContextBacking, ColdTierContext, PagedKVCacheAdapter,
};

/// Cloned handles to a paged adapter's KV pool and block allocator so
/// [`StateDiskEngine`] can capture blocks without holding a borrow of the
/// enclosing model or adapter.
pub(crate) struct KvPoolHandle {
    pub(crate) pool: Arc<LayerKVPool>,
    pub(crate) allocator: Arc<Mutex<BlockAllocator>>,
}

impl KvPoolHandle {
    pub(crate) fn from_adapter(adapter: &PagedKVCacheAdapter) -> Self {
        Self {
            pool: Arc::clone(adapter.layer_kv_pool()),
            allocator: Arc::clone(adapter.allocator()),
        }
    }
}

/// What a tick of the state tier does (Splash `Engine::tick`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Tick {
    /// Landings only: no page read, no MLX eval (every allocation, every
    /// point a prefill reached).
    Poll,
    /// Landings, then the owed chain until `deadline` (a claim's wait).
    Wait { deadline: Instant },
    /// Landings, then what the writer's queue takes of the owed chain now
    /// (between commands), whose running lanes' next commands add
    /// `command_pages` KV pages beyond what they hold.
    Full { command_pages: u64 },
}

/// Opaque model-owned state tier with the states it holds, set aside while a
/// load's warm-up runs.
pub(crate) struct HeldTier(Box<dyn Any>);

impl HeldTier {
    pub(crate) fn new<T: 'static>(value: T) -> Self {
        Self(Box::new(value))
    }

    pub(crate) fn downcast<T: 'static>(self) -> Option<T> {
        self.0.downcast::<T>().ok().map(|boxed| *boxed)
    }
}

/// The state write in flight (Splash `StateCache::pending_`).
pub(crate) struct PendingWrite {
    pub(crate) id: EntryId,
    pub(crate) key: ColdCacheKey,
}

/// Replacement metadata for an in-flight in-place SSD overwrite of an
/// already-committed durable SSD anchor.
///
/// Owns the [`ColdKeyHold`] pinning both `key` and its complete Global KV chain
/// against competing `MakeRoom::Evict` writes so clearing the in-memory store
/// (e.g. `ResetCaches` / `clear_ram_and_checkpoints()`) neither drops the
/// eviction hold nor causes [`StateDiskEngine::poll_offload`] to remove the
/// freshly committed SSD anchor upon landing.
struct PendingReplacement {
    id: EntryId,
    key: ColdCacheKey,
    _fallback_hold: ColdKeyHold,
}

/// A disk state's chain on its way to the tier (Splash
/// `Cache::PersistCursor`): taken once, walked from the first block whose
/// copy has not landed.
pub(crate) struct ChainCursor {
    pub(crate) id: EntryId,
    /// The chain's KV objects, in block order.
    pub(crate) keys: Vec<ColdCacheKey>,
    /// Every block below it is on the tier.
    pub(crate) landed: usize,
    pub(crate) in_flight: HashSet<ColdCacheKey>,
    pub(crate) room: MakeRoom,
}

/// A parked restore whose state read waits for the staging buffer; while
/// one lives, new state writes yield to reads (reads before writes, Splash
/// `SlotFile`).
pub(crate) struct ReadWait(Arc<AtomicU32>);

impl Drop for ReadWait {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

/// What a reclaim of one state did (Splash `StateEviction`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Reclaimed {
    /// Its RAM copy left through a write that is now in flight.
    Written,
    /// Its RAM copy left; its disk copy stays.
    RamDropped,
    /// The state left with no copy.
    Erased,
    /// It frees nothing until a write in flight lands.
    Pending,
    /// No RAM copy to take.
    Nothing,
}

/// Result of [`StateDiskEngine::reclaim`], including whether state marks must
/// be resynced and whether a state offload was started or failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ReclaimStep {
    pub(crate) outcome: Reclaimed,
    pub(crate) resync: bool,
    pub(crate) staged: bool,
    pub(crate) encode_or_enqueue_failed: bool,
}

/// What [`StateDiskEngine::stage`] did with a write.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Staged {
    Started,
    /// The staging buffer or writer queue is busy: nothing changed.
    Busy,
    /// No write: the tier is closed or the state does not encode.
    Refused,
}

/// What one `chain_step` did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ChainStep {
    pub(crate) enqueued: usize,
    /// The chain is on the tier whole: its mark went.
    pub(crate) landed_all: bool,
    /// The point failed: its state's disk copy went.
    pub(crate) failed: bool,
}

/// Extended counters returned by [`StateDiskEngine::chain_step`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct EngineChainStep {
    pub(crate) step: ChainStep,
    pub(crate) invalidations: u64,
    pub(crate) offload_failures: u64,
    /// Entry ID whose chain failed (`lost || failed`) during this step, even when
    /// `self.chain` was `None` before the call.
    pub(crate) failed_id: Option<EntryId>,
}

/// Counters and flags returned by [`StateDiskEngine::poll_offload`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct PollOutcome {
    pub(crate) landed: bool,
    pub(crate) state_landed_ok: u64,
    pub(crate) offload_failures: u64,
    pub(crate) resync: bool,
    /// Entry ID whose in-flight state blob write failed during this poll.
    pub(crate) state_failed_id: Option<EntryId>,
    /// Entry ID whose in-flight KV chain block failed during this poll.
    pub(crate) chain_failed_id: Option<EntryId>,
}

/// Combined outcome returned by [`StateDiskEngine::tick`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct EngineTickOutcome {
    pub(crate) progressed: bool,
    pub(crate) resync: bool,
    pub(crate) state_landed_ok: u64,
    pub(crate) invalidations: u64,
    pub(crate) offload_failures: u64,
    /// Entry ID whose in-flight state blob write failed during this tick.
    pub(crate) state_failed_id: Option<EntryId>,
    /// Entry IDs whose KV chain failed during this tick (up to one per poll/chain sub-step).
    pub(crate) chain_failed_ids: [Option<EntryId>; 3],
}

impl EngineTickOutcome {
    fn record_chain_failed(&mut self, id: Option<EntryId>) {
        let Some(id) = id else {
            return;
        };
        if self.chain_failed_ids.contains(&Some(id)) {
            return;
        }
        if let Some(slot) = self.chain_failed_ids.iter_mut().find(|s| s.is_none()) {
            *slot = Some(id);
        }
    }
}

/// What `persist` did with a point (Splash `PersistStatus`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PersistStatus {
    /// The state and its whole chain are on the tier.
    Durable,
    /// Copies are in flight.
    Started,
    /// The tier serves another write first.
    Busy,
    /// The point is no longer its conversation's newest.
    Unneeded,
    /// The tier cannot keep it.
    Refused,
}

/// Extended counters returned by [`StateDiskEngine::persist`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PersistStep {
    pub(crate) status: PersistStatus,
    pub(crate) staged: bool,
    pub(crate) resync: bool,
    pub(crate) invalidations: u64,
    pub(crate) offload_failures: u64,
}

/// `entry`'s lineage borrowed as a [`Lineage`].
pub(crate) fn lineage_of<S>(entry: &Entry<S>) -> Lineage<'_> {
    Lineage {
        tokens: entry.tokens(),
        extra_keys: entry.extra_keys(),
        salt: entry.salt(),
    }
}

/// Chain keys of `lineage`'s whole `block_size`-token blocks in `group`.
pub(crate) fn lineage_keys(
    group: ColdGroup,
    fingerprint: ColdCacheFingerprint,
    lineage: Lineage<'_>,
    block_size: u32,
) -> Vec<ColdCacheKey> {
    let mut parent = None;
    lineage
        .tokens
        .chunks_exact(block_size.max(1) as usize)
        .enumerate()
        .map(|(index, tokens)| {
            let extra_keys = lineage.extra_keys.get(index).map_or(&[][..], Vec::as_slice);
            let key = ColdCacheKey::chain(
                group,
                fingerprint,
                parent,
                tokens,
                extra_keys,
                lineage.salt,
                index,
            );
            parent = Some(key);
            key
        })
        .collect()
}

/// The keys of `lineage` at `block` pages: one per whole page, and one for
/// the part page past the last whole one, chained on it. A key hashes its
/// token count, so the part page's key is never a whole page's.
pub(crate) fn part_page_keys(
    group: ColdGroup,
    fingerprint: ColdCacheFingerprint,
    lineage: Lineage<'_>,
    block: u32,
) -> (Vec<ColdCacheKey>, Option<ColdCacheKey>) {
    let whole = lineage_keys(group, fingerprint, lineage, block);
    let index = whole.len();
    let part = lineage
        .tokens
        .get(index * block.max(1) as usize..)
        .filter(|tokens| !tokens.is_empty())
        .map(|tokens| {
            ColdCacheKey::chain(
                group,
                fingerprint,
                whole.last().copied(),
                tokens,
                lineage.extra_keys.get(index).map_or(&[][..], Vec::as_slice),
                lineage.salt,
                index,
            )
        });
    (whole, part)
}

/// `entry`'s chain keys in `group`: one per whole block, with that block's
/// extra keys, the salt on block 0 (the adapter's cold walk keys a block
/// alike).
pub(crate) fn chain_keys<S>(
    group: ColdGroup,
    fingerprint: ColdCacheFingerprint,
    entry: &Entry<S>,
) -> Vec<ColdCacheKey> {
    lineage_keys(group, fingerprint, lineage_of(entry), entry.block_size())
}

pub(crate) fn missing_keys(tier: &DiskTier, keys: &[ColdCacheKey]) -> usize {
    keys.iter()
        .filter(|key| !tier.manager.contains_in(key, ColdGroup::Kv))
        .count()
}

pub(crate) fn missing_blocks<S>(tier: &DiskTier, entry: &Entry<S>) -> usize {
    missing_keys(tier, &chain_keys(ColdGroup::Kv, tier.kv_fingerprint, entry))
}

pub(crate) fn label_of<S>(entry: &Entry<S>) -> ColdStateLabel {
    ColdStateLabel {
        tokens: entry.tokens().to_vec(),
        extra_keys: entry.extra_keys().to_vec(),
        salt: entry.salt(),
        last_used: entry.last_used(),
        checkpoint: entry.checkpoint(),
    }
}

/// Splash `StateCache::discardDisk` with its slot: `id`'s disk copy leaves
/// the tier, and a state with no RAM copy goes with it.
pub(crate) fn drop_disk_copy<S: StateBytes>(
    store: &mut HybridStateStore<S>,
    tier: &DiskTier,
    id: EntryId,
) {
    if let Some(copy) = store.get(id).and_then(|entry| entry.disk().copied())
        && copy.landed
    {
        tier.manager.remove(&copy.key);
    }
    store.set_disk(id, None);
    if store.get(id).is_some_and(|entry| entry.ram().is_none()) {
        store.remove(id);
    }
}

/// Splash `Cache::newestPoint`: an ordinary state its conversation still
/// resumes from.
pub(crate) fn newest_point<S: StateBytes>(store: &HybridStateStore<S>, id: EntryId) -> bool {
    store.get(id).is_some_and(|entry| !entry.checkpoint()) && !store.superseded(id)
}

/// The adoption loop over any store: each state object of `spans` and
/// `geometry` whose own key and whole KV chain are on `tier` comes back as a
/// disk-only state. `copy` gives its `cached_bytes` and its copy's
/// `window_begin`, or `None` to leave it.
pub(crate) fn adopt_states<S: StateBytes>(
    store: &mut HybridStateStore<S>,
    tier: &DiskTier,
    spans: &[u64],
    geometry: &str,
    copy: impl FnMut(&ColdStateLabel) -> Option<(u64, u32)>,
) -> u64 {
    adopt_states_on(store, tier, spans, geometry, false, copy)
}

/// [`adopt_states`]; with `part_pages` a state the move of a 16-token store
/// left on a part page comes back too: its rows are whole blocks and
/// [`PART_PAGE_ROWS`] more, its key is chained once more over those rows,
/// and the fragment that holds them is on the tier with its whole chain.
pub(crate) fn adopt_states_on<S: StateBytes>(
    store: &mut HybridStateStore<S>,
    tier: &DiskTier,
    spans: &[u64],
    geometry: &str,
    part_pages: bool,
    mut copy: impl FnMut(&ColdStateLabel) -> Option<(u64, u32)>,
) -> u64 {
    let block_size = store.block_size();
    let manager = &tier.manager;
    // A lower limit than the earlier process had takes the oldest copies
    // first, so only chains that survive come back.
    manager.fit_quota();
    let mut newest = 0;
    let mut adopted = 0;
    for key in manager.keys_in(ColdGroup::DFlash2State) {
        let Some((layout, label)) = manager.state_label(key, tier.state_fingerprint) else {
            continue;
        };
        let depth = label.tokens.len();
        let lineage = Lineage {
            tokens: &label.tokens,
            extra_keys: &label.extra_keys,
            salt: label.salt,
        };
        let (own, own_part) = part_page_keys(
            ColdGroup::DFlash2State,
            tier.state_fingerprint,
            lineage,
            block_size,
        );
        let (chain, fragment) =
            part_page_keys(ColdGroup::Kv, tier.kv_fingerprint, lineage, block_size);
        let on_part_page = own_part.is_some();
        let own_key = if on_part_page {
            own_part
                .filter(|_| part_pages && own.len() * block_size as usize + PART_PAGE_ROWS == depth)
        } else {
            own.last()
                .copied()
                .filter(|_| own.len() * block_size as usize == depth)
        };
        if layout.spans != spans
            || layout.geometry != geometry
            || own_key != Some(key)
            || missing_keys(tier, &chain) > 0
            || fragment.is_some_and(|fragment| !manager.contains_in(&fragment, ColdGroup::Kv))
        {
            continue;
        }
        if u32::try_from(depth).is_err() {
            continue;
        }
        let Some((cached_bytes, window_begin)) = copy(&label) else {
            continue;
        };
        let copy = DiskCopy {
            key,
            bytes: state_object_bytes(&layout, &label),
            landed: true,
            window_begin,
            chain_owed: false,
        };
        let class = if label.checkpoint {
            StateClass::Checkpoint
        } else {
            StateClass::Ordinary
        };
        let taken = if on_part_page {
            store.adopt_disk_on_part_page(lineage, cached_bytes, class, copy, label.last_used)
        } else {
            store.adopt_disk(lineage, cached_bytes, class, copy, label.last_used)
        };
        if taken.is_ok() {
            newest = newest.max(label.last_used);
            adopted += 1;
        }
    }
    store.continue_after(newest);
    adopted
}

/// The instant `deadline_ms` names on the Unix-epoch millisecond clock
/// (`Date.now()`); now once it has passed.
pub(crate) fn instant_at(deadline_ms: f64) -> Instant {
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
        * 1000.0;
    let left = (deadline_ms - now_ms).max(0.0);
    Instant::now() + Duration::try_from_secs_f64(left / 1000.0).unwrap_or(Duration::ZERO)
}

/// The adapter's view of a state tier: keeps no per-turn KV on disk, so only
/// a state object backs a cold prefix.
pub(crate) fn state_context(tier: &DiskTier) -> ColdTierContext {
    ColdTierContext {
        manager: Arc::clone(&tier.manager),
        fingerprint: tier.kv_fingerprint,
        backing: ColdContextBacking::StateOnly,
    }
}

/// Splash `Engine::copyDemandBytes`: none unless a claim reached a state
/// that frees memory only once its copy lands within the copy demand's
/// window; then twice the pages the running lanes' next commands add, so
/// the next command finds them copied too.
pub(crate) fn copy_demand_bytes(
    governor: &crate::engine::governor::governor::Governor,
    now: f64,
    command_pages: u64,
    page_bytes: u64,
) -> u64 {
    if !governor.ledger(|ledger| ledger.copy_demand(now)) {
        return 0;
    }
    command_pages.saturating_mul(2).saturating_mul(page_bytes)
}

/// Per-model state encoding, key derivation, and adoption metadata for
/// [`StateDiskEngine`].
pub(crate) trait StateDiskCodec {
    type State: StateBytes;
    type EncodeCtx<'a>;

    /// Whether [`StateDiskEngine::chain_step`] breaks immediately when the
    /// block writer has no room or once `deadline` elapses after enqueuing a
    /// block. Models that allow a reclaim chain to wait for block room up to
    /// `deadline` (e.g. Qwen3.5) leave this `false`; Inkling2 sets `true`.
    const STOP_CHAIN_AT_DEADLINE: bool = false;

    /// Key of `entry`'s state object on `tier`. Defaults to the last whole-block
    /// key in `ColdGroup::DFlash2State`.
    fn state_key(&self, tier: &DiskTier, entry: &Entry<Self::State>) -> Option<ColdCacheKey> {
        chain_keys(ColdGroup::DFlash2State, tier.state_fingerprint, entry)
            .last()
            .copied()
    }

    /// Encodes `state` into the fixed-size staging buffer `into` and returns
    /// the state's `window_begin` on success, or `None` on failure.
    fn encode_into(
        &self,
        ctx: Self::EncodeCtx<'_>,
        spans: &[u64],
        entry: &Entry<Self::State>,
        state: &Self::State,
        into: &mut [u8],
    ) -> Option<u32>;

    /// `(cached_bytes, window_begin)` for an adopted state object's label, or
    /// `None` to skip it.
    fn adopt_copy(&self, spans: &[u64], label: &ColdStateLabel) -> Option<(u64, u32)>;
}

/// Shared governor-driven SSD state-disk engine: single reusable host staging
/// buffer, in-flight state write tracking, KV chain persistence cursor,
/// read-before-write waiter guard, and landing channel.
pub(crate) struct StateDiskEngine<C: StateDiskCodec> {
    pub(crate) tier: Arc<DiskTier>,
    pub(crate) spans: Vec<u64>,
    pub(crate) geometry: String,
    pub(crate) block_bytes: u64,
    pub(crate) codec: C,
    /// `None` while the write in flight holds it.
    pub(crate) staging: Option<Vec<u8>>,
    pub(crate) landed_tx: Sender<Landed>,
    pub(crate) landed_rx: Receiver<Landed>,
    /// The state write in flight (Splash `StateCache::pending_`).
    pub(crate) pending: Option<PendingWrite>,
    /// Replacement hold/flag for an in-flight in-place SSD overwrite of an
    /// already-committed durable SSD anchor (kept beside `pending` so external
    /// callers constructing [`PendingWrite`] remain untouched).
    pending_replacement: Option<PendingReplacement>,
    /// Optional entry ID whose store entry stays indexed on state-write failure even when its RAM
    /// copy was dropped (used for in-place SSD upgrades with a durable fallback copy to restore).
    pub(crate) preserve_entry_on_state_failure: Option<EntryId>,
    /// The chain the tier copies under a disk state (Splash
    /// `Cache::persisting_`).
    pub(crate) chain: Option<ChainCursor>,
    /// Parked restores whose state read waits for the staging buffer.
    pub(crate) read_waiters: Arc<AtomicU32>,
    /// The state writes that landed, in landing order.
    #[cfg(test)]
    pub(crate) landed_states: Vec<ColdCacheKey>,
}

impl<C: StateDiskCodec> StateDiskEngine<C> {
    pub(crate) fn new(
        tier: Arc<DiskTier>,
        spans: Vec<u64>,
        geometry: String,
        block_bytes: u64,
        codec: C,
    ) -> Option<Self> {
        let bytes = usize::try_from(spans.iter().sum::<u64>()).ok()?;
        let (landed_tx, landed_rx) = std::sync::mpsc::channel();
        Some(Self {
            tier,
            spans,
            geometry,
            block_bytes,
            codec,
            staging: Some(vec![0; bytes]),
            landed_tx,
            landed_rx,
            pending: None,
            pending_replacement: None,
            preserve_entry_on_state_failure: None,
            chain: None,
            read_waiters: Arc::new(AtomicU32::new(0)),
            #[cfg(test)]
            landed_states: Vec::new(),
        })
    }

    /// The writer wakes `model`'s thread after each tracked landing, so an
    /// idle model moves its owed chain with no request (Splash
    /// `Engine::tick` → `pollTransfers` → `signalResourceProgress`).
    pub(crate) fn wake_model_on_landing(&self, model: u64) {
        if model == 0 {
            return;
        }
        self.tier.manager.set_landing_wake(Some(Arc::new(move || {
            crate::engine::governor::monitor::wake(model);
        })));
    }

    /// The tier takes writes while open.
    pub(crate) fn writable(&self) -> bool {
        self.tier.is_open()
    }

    /// A state write is in flight (Splash `StateCache::writing`).
    pub(crate) fn writing(&self) -> bool {
        self.pending.is_some()
    }

    pub(crate) fn staging_free(&self) -> bool {
        self.staging.is_some()
    }

    /// Registers a parked restore whose state read waits for the staging buffer.
    pub(crate) fn acquire_read_wait(&self) -> ReadWait {
        self.read_waiters.fetch_add(1, Ordering::AcqRel);
        ReadWait(Arc::clone(&self.read_waiters))
    }

    pub(crate) fn has_read_waiters(&self) -> bool {
        self.read_waiters.load(Ordering::Acquire) > 0
    }

    #[cfg(test)]
    pub(crate) fn read_waiters(&self) -> u32 {
        self.read_waiters.load(Ordering::Acquire)
    }

    /// The staging buffer for a state write; none while a parked restore's
    /// state read waits for it (reads before writes, Splash `SlotFile`).
    pub(crate) fn staging_for_write(&mut self) -> Option<Vec<u8>> {
        if self.read_waiters.load(Ordering::Acquire) > 0 {
            return None;
        }
        self.staging.take()
    }

    /// Only a persistent tier makes room for a copy ahead (Splash
    /// `Cache::persist`); a reclaim's write always does.
    pub(crate) fn persist_room(&self) -> MakeRoom {
        if self.tier.persistent {
            MakeRoom::Evict
        } else {
            MakeRoom::Refuse
        }
    }

    /// A state object's layout at `depth`.
    pub(crate) fn layout(&self, depth: usize) -> Option<ColdStateLayout> {
        Some(ColdStateLayout {
            boundary_tokens: u32::try_from(depth).ok()?,
            spans: self.spans.clone(),
            geometry: self.geometry.clone(),
        })
    }

    /// The key of `entry`'s state object.
    pub(crate) fn state_key(&self, entry: &Entry<C::State>) -> Option<ColdCacheKey> {
        self.codec.state_key(&self.tier, entry)
    }

    /// The KV block chain keys for `entry`.
    #[allow(dead_code)]
    pub(crate) fn kv_chain_keys(&self, entry: &Entry<C::State>) -> Vec<ColdCacheKey> {
        chain_keys(ColdGroup::Kv, self.tier.kv_fingerprint, entry)
    }

    /// Every block of `entry`'s chain is on the tier.
    pub(crate) fn chain_on_disk(&self, entry: &Entry<C::State>) -> bool {
        missing_blocks(&self.tier, entry) == 0
    }

    /// Splash `Cache::pointFits`: a tier that makes room holds the state
    /// with its whole chain under its limit; one that does not has room
    /// left for what of them is not on it yet. Either way the volume holds
    /// what is not on the tier yet above the writer's reserve, after what
    /// making room frees (Splash `DiskBudget::reportFull`: a full volume
    /// refuses the write before it is made).
    pub(crate) fn point_fits(&self, entry: &Entry<C::State>, room: MakeRoom) -> bool {
        let Some(layout) = self.layout(entry.depth()) else {
            return false;
        };
        let state = state_object_bytes(&layout, &label_of(entry));
        let blocks = entry.chain().len() as u64;
        let limit = self.tier.limit;
        let on_disk = entry.disk().is_some_and(|copy| copy.landed);
        let lacking = (missing_blocks(&self.tier, entry) as u64)
            .saturating_mul(self.block_bytes)
            .saturating_add(if on_disk { 0 } else { state });
        let taken = self.tier.manager.total_bytes();
        let Some(free) = self.tier.manager.room_above_reserve() else {
            return false;
        };
        match room {
            MakeRoom::Evict => {
                blocks
                    .saturating_mul(self.block_bytes)
                    .saturating_add(state)
                    <= limit
                    && lacking <= free.saturating_add(taken)
            }
            MakeRoom::Refuse => taken <= limit && lacking <= (limit - taken).min(free),
        }
    }

    /// Splash `StateCache::reclaimWrites`: a RAM copy without a disk copy, on
    /// a tier that takes writes and holds it with its chain.
    pub(crate) fn reclaim_writes(&self, store: &HybridStateStore<C::State>, id: EntryId) -> bool {
        if !self.writable() {
            return false;
        }
        store.get(id).is_some_and(|entry| {
            entry.ram().is_some()
                && entry.disk().is_none()
                && self.point_fits(entry, MakeRoom::Evict)
        })
    }

    /// Splash `StateCache::startWrite`: encodes `id`'s RAM copy into the
    /// staging buffer and hands it to the tier's writer; its disk copy is in
    /// flight from then on. Without `keep_ram` the RAM copy goes at once.
    pub(crate) fn stage(
        &mut self,
        store: &mut HybridStateStore<C::State>,
        id: EntryId,
        room: MakeRoom,
        keep_ram: bool,
        ctx: C::EncodeCtx<'_>,
    ) -> Staged {
        self.stage_inner(store, id, room, keep_ram, ctx, None)
    }

    /// Like [`Self::stage`], for an in-place replacement of an already-committed
    /// durable SSD anchor. Transfers `fallback_hold` onto this engine until
    /// [`Self::poll_offload`] processes the landing (or until
    /// [`Self::cancel_pending_replacement`] / [`Self::strip_disk_copies`] or
    /// drop), so clearing the in-memory store while the write is in flight
    /// neither drops the eviction hold nor removes the committed replacement
    /// blob upon landing.
    pub(crate) fn stage_replacement(
        &mut self,
        store: &mut HybridStateStore<C::State>,
        id: EntryId,
        room: MakeRoom,
        keep_ram: bool,
        ctx: C::EncodeCtx<'_>,
        fallback_hold: ColdKeyHold,
    ) -> Staged {
        self.stage_inner(store, id, room, keep_ram, ctx, Some(fallback_hold))
    }

    /// Cancels replacement tracking (and releases its [`ColdKeyHold`]) for `id`
    /// when an in-flight replacement is explicitly discarded (e.g. urgent erase
    /// whose Global KV chain is incomplete), so a subsequent `Ok` landing
    /// removes any orphaned state blob.
    pub(crate) fn cancel_pending_replacement(&mut self, id: EntryId) {
        self.pending_replacement.take_if(|rep| rep.id == id);
    }

    #[cfg(test)]
    pub(crate) fn has_pending_replacement(&self) -> bool {
        self.pending_replacement.is_some()
    }

    fn stage_inner(
        &mut self,
        store: &mut HybridStateStore<C::State>,
        id: EntryId,
        room: MakeRoom,
        keep_ram: bool,
        ctx: C::EncodeCtx<'_>,
        fallback_hold: Option<ColdKeyHold>,
    ) -> Staged {
        let Some(entry) = store.get(id) else {
            return Staged::Refused;
        };
        let (Some(state), Some(layout), Some(key)) = (
            entry.ram(),
            self.layout(entry.depth()),
            self.state_key(entry),
        ) else {
            return Staged::Refused;
        };
        let Some(mut staging) = self.staging_for_write() else {
            return Staged::Busy;
        };
        let Some(window_begin) =
            self.codec
                .encode_into(ctx, &self.spans, entry, state, &mut staging)
        else {
            self.staging = Some(staging);
            return Staged::Refused;
        };
        let label = label_of(entry);
        let copy = DiskCopy {
            key,
            bytes: state_object_bytes(&layout, &label),
            landed: false,
            window_begin,
            chain_owed: !self.chain_on_disk(entry),
        };
        let object = ColdStateObject {
            key,
            fingerprint: self.tier.state_fingerprint,
            layout,
            label,
            payload: staging,
        };
        match self
            .tier
            .manager
            .enqueue_state(object, room, self.landed_tx.clone())
        {
            Ok(()) => {
                self.pending = Some(PendingWrite { id, key });
                self.pending_replacement = fallback_hold.map(|hold| PendingReplacement {
                    id,
                    key,
                    _fallback_hold: hold,
                });
                store.set_disk(id, Some(copy));
                if !keep_ram {
                    store.drop_ram(id);
                }
                Staged::Started
            }
            Err(EnqueueRefused::Busy(object)) => {
                self.staging = Some(object.payload);
                Staged::Busy
            }
            Err(EnqueueRefused::Gone(object)) => {
                self.staging = Some(object.payload);
                Staged::Refused
            }
        }
    }

    /// Splash `StateCache::reclaim`: a RAM copy whose own write is in flight
    /// waits for it; one with a landed disk copy goes; a writable one is
    /// written while the staging buffer is free, its RAM free at once, and
    /// waits while it is not; any other leaves with no copy.
    pub(crate) fn reclaim(
        &mut self,
        store: &mut HybridStateStore<C::State>,
        id: EntryId,
        ctx: C::EncodeCtx<'_>,
    ) -> ReclaimStep {
        let Some(entry) = store.get(id).filter(|entry| entry.ram().is_some()) else {
            return ReclaimStep {
                outcome: Reclaimed::Nothing,
                resync: false,
                staged: false,
                encode_or_enqueue_failed: false,
            };
        };
        if let Some(copy) = entry.disk() {
            if !copy.landed {
                return ReclaimStep {
                    outcome: Reclaimed::Pending,
                    resync: false,
                    staged: false,
                    encode_or_enqueue_failed: false,
                };
            }
            store.drop_ram(id);
            return ReclaimStep {
                outcome: Reclaimed::RamDropped,
                resync: true,
                staged: false,
                encode_or_enqueue_failed: false,
            };
        }
        let writable = self.reclaim_writes(store, id);
        if writable && self.writing() {
            return ReclaimStep {
                outcome: Reclaimed::Pending,
                resync: false,
                staged: false,
                encode_or_enqueue_failed: false,
            };
        }
        let mut failed = false;
        if writable {
            match self.stage(store, id, MakeRoom::Evict, false, ctx) {
                Staged::Started => {
                    return ReclaimStep {
                        outcome: Reclaimed::Written,
                        resync: true,
                        staged: true,
                        encode_or_enqueue_failed: false,
                    };
                }
                Staged::Busy => {
                    return ReclaimStep {
                        outcome: Reclaimed::Pending,
                        resync: false,
                        staged: false,
                        encode_or_enqueue_failed: false,
                    };
                }
                Staged::Refused => failed = true,
            }
        }
        store.remove(id);
        ReclaimStep {
            outcome: Reclaimed::Erased,
            resync: true,
            staged: false,
            encode_or_enqueue_failed: failed,
        }
    }

    /// Splash `StateCache::pollOffload` and `Cache::pollTransfers`: first the
    /// state objects the tier's evictions cleared, then the writer's landings.
    pub(crate) fn poll_offload(&mut self, store: &mut HybridStateStore<C::State>) -> PollOutcome {
        let mut out = PollOutcome::default();
        for key in self.tier.manager.take_evicted() {
            let Some(id) = store
                .entries()
                .find(|entry| {
                    entry
                        .disk()
                        .is_some_and(|copy| copy.landed && copy.key == key)
                })
                .map(Entry::id)
            else {
                continue;
            };
            store.set_disk(id, None);
            if store.get(id).is_some_and(|entry| entry.ram().is_none()) {
                store.remove(id);
            }
            if self.chain.as_ref().is_some_and(|cursor| cursor.id == id) {
                self.chain = None;
            }
            out.resync = true;
        }
        while let Ok(landing) = self.landed_rx.try_recv() {
            out.landed = true;
            let (key, ok, payload) = match landing {
                Landed::State { key, ok, payload } => (key, ok, payload),
                Landed::Block { key, ok } => {
                    let Some(id) = self
                        .chain
                        .as_mut()
                        .and_then(|cursor| cursor.in_flight.remove(&key).then_some(cursor.id))
                    else {
                        continue;
                    };
                    if ok {
                        continue;
                    }
                    out.offload_failures += 1;
                    drop_disk_copy(store, &self.tier, id);
                    self.chain = None;
                    out.chain_failed_id = Some(id);
                    out.resync = true;
                    continue;
                }
            };
            self.staging = Some(payload);
            let Some(write) = self.pending.take_if(|write| write.key == key) else {
                continue;
            };
            let replacement = self
                .pending_replacement
                .take_if(|rep| rep.id == write.id && rep.key == key);
            let is_replacement = replacement.is_some();
            let preserve_on_failure = self
                .preserve_entry_on_state_failure
                .take_if(|id| *id == write.id)
                .is_some();
            if ok {
                #[cfg(test)]
                self.landed_states.push(key);
                let chain_missing = store
                    .get(write.id)
                    .filter(|entry| entry.disk().is_some_and(|copy| copy.key == key))
                    .map(|entry| !self.chain_on_disk(entry));
                match (
                    chain_missing,
                    store.disk_mut(write.id).filter(|copy| copy.key == key),
                ) {
                    (Some(chain_missing), Some(copy)) => {
                        copy.landed = true;
                        if chain_missing && !copy.chain_owed {
                            copy.chain_owed = true;
                            out.resync = true;
                        }
                        out.state_landed_ok += 1;
                    }
                    // The entry vanished while its state write was in flight (e.g., `ResetCaches` /
                    // `clear_ram_and_checkpoints()` or urgent erase). For an in-place replacement
                    // of an already-committed durable SSD anchor, treat the landing as a clean
                    // bookkeeping no-op that preserves the freshly committed SSD anchor on disk
                    // (releasing `replacement`'s `ColdKeyHold` when `replacement` drops); for a
                    // non-replacement initial spill whose KV chain may never have been persisted,
                    // delete the orphaned blob.
                    (_, None) | (None, _) => {
                        if is_replacement {
                            out.state_landed_ok += 1;
                        } else {
                            self.tier.manager.remove(&key);
                        }
                    }
                }
                continue;
            }
            out.offload_failures += 1;
            out.state_failed_id = Some(write.id);
            out.resync = true;
            if self
                .chain
                .as_ref()
                .is_some_and(|cursor| cursor.id == write.id)
            {
                self.chain = None;
            }
            store.set_disk(write.id, None);
            if !preserve_on_failure
                && store
                    .get(write.id)
                    .is_some_and(|entry| entry.ram().is_none())
            {
                store.remove(write.id);
            }
        }
        out
    }

    /// Splash `Cache::persist` for the chain under a disk state, on the model
    /// thread: a reclaim's chain (the cursor's, else the oldest), else the
    /// cursor's copy ahead, else the oldest owed chain.
    pub(crate) fn chain_step(
        &mut self,
        store: &mut HybridStateStore<C::State>,
        kv: &KvPoolHandle,
        deadline: Instant,
    ) -> EngineChainStep {
        let mut out = EngineChainStep::default();
        let pool = &kv.pool;
        let allocator = &kv.allocator;
        let owed = |entry: &Entry<C::State>| entry.disk().is_some_and(|copy| copy.chain_owed);
        let reclaimed = |entry: &Entry<C::State>| owed(entry) && entry.ram().is_none();
        let oldest = |pick: &dyn Fn(&Entry<C::State>) -> bool| {
            store
                .entries()
                .filter(|entry| pick(entry))
                .min_by_key(|entry| entry.last_used())
                .map(Entry::id)
        };
        // A reclaim's chain frees memory once it lands: it goes before a copy
        // ahead (Splash `Cache::persist` yields while a reclaim's write is in
        // flight).
        let current = self.chain.as_ref().map(|cursor| cursor.id).filter(|&id| {
            store.get(id).is_some_and(|entry| {
                reclaimed(entry) || (owed(entry) && oldest(&reclaimed).is_none())
            })
        });
        let Some(id) = current
            .or_else(|| oldest(&reclaimed))
            .or_else(|| oldest(&owed))
        else {
            self.chain = None;
            return out;
        };
        let Some((entry, copy)) = store
            .get(id)
            .and_then(|entry| entry.disk().copied().map(|copy| (entry, copy)))
        else {
            return out;
        };
        if current.is_none() {
            // A reclaim's write makes room; a copy ahead beside its RAM copy
            // only on a persistent tier.
            let room = if entry.ram().is_none() || self.tier.persistent {
                MakeRoom::Evict
            } else {
                MakeRoom::Refuse
            };
            self.chain = Some(ChainCursor {
                id,
                keys: chain_keys(ColdGroup::Kv, self.tier.kv_fingerprint, entry),
                landed: 0,
                in_flight: HashSet::new(),
                room,
            });
        }
        let manager = &self.tier.manager;
        let gone = copy.landed && !manager.contains_in(&copy.key, ColdGroup::DFlash2State);
        let mut lost = gone;
        let mut failed = false;
        let mut missing = false;
        let mut in_flight = false;
        if let Some(cursor) = self.chain.as_mut().filter(|_| !gone) {
            let block_size = entry.block_size() as usize;
            let mut from = cursor.landed;
            'walk: loop {
                for index in from..cursor.keys.len() {
                    let key = cursor.keys[index];
                    if cursor.in_flight.contains(&key) {
                        in_flight = true;
                        continue;
                    }
                    if manager.contains_in(&key, ColdGroup::Kv) {
                        if index == cursor.landed {
                            cursor.landed += 1;
                        }
                        continue;
                    }
                    cursor.landed = cursor.landed.min(index);
                    missing = true;
                    let block = entry.chain().get(index).and_then(|&hash| {
                        allocator
                            .lock()
                            .ok()
                            .and_then(|allocator| allocator.cached_block(hash))
                    });
                    let (Some(block), Some(tokens)) = (
                        block,
                        entry
                            .tokens()
                            .get(index * block_size..(index + 1) * block_size),
                    ) else {
                        // Neither resident nor on the tier: the state cannot
                        // come back whole.
                        out.invalidations += 1;
                        lost = true;
                        break 'walk;
                    };
                    if !manager.has_block_room()
                        && (C::STOP_CHAIN_AT_DEADLINE || Instant::now() >= deadline)
                    {
                        break 'walk;
                    }
                    match manager.capture_and_enqueue_tracked(
                        pool,
                        &block,
                        key,
                        self.tier.kv_fingerprint,
                        tokens,
                        deadline,
                        cursor.room,
                        self.landed_tx.clone(),
                    ) {
                        Ok(true) => {
                            cursor.in_flight.insert(key);
                            in_flight = true;
                            out.step.enqueued += 1;
                            if C::STOP_CHAIN_AT_DEADLINE && Instant::now() >= deadline {
                                break 'walk;
                            }
                        }
                        Ok(false) => break 'walk,
                        Err(_) => {
                            out.offload_failures += 1;
                            failed = true;
                            break 'walk;
                        }
                    }
                }
                if in_flight || from == 0 {
                    break;
                }
                from = 0;
                missing = false;
            }
        }
        if lost || failed {
            drop_disk_copy(store, &self.tier, id);
            self.chain = None;
            out.step.failed = true;
            out.failed_id = Some(id);
        } else if copy.landed && !missing && !in_flight {
            if let Some(copy) = store.disk_mut(id) {
                copy.chain_owed = false;
            }
            self.chain = None;
            out.step.landed_all = true;
        }
        out
    }

    /// Runs the landing poll, chain step, and cache-root measure for one
    /// [`Tick`].
    pub(crate) fn tick(
        &mut self,
        store: &mut HybridStateStore<C::State>,
        kv: Option<&KvPoolHandle>,
        tick: Tick,
    ) -> EngineTickOutcome {
        let mut out = EngineTickOutcome::default();
        let first = self.poll_offload(store);
        out.progressed |= first.landed;
        out.resync |= first.resync;
        out.state_landed_ok += first.state_landed_ok;
        out.offload_failures += first.offload_failures;
        if first.state_failed_id.is_some() {
            out.state_failed_id = first.state_failed_id;
        }
        out.record_chain_failed(first.chain_failed_id);

        let deadline = match tick {
            Tick::Poll => None,
            Tick::Wait { deadline } => Some(deadline),
            Tick::Full { .. } => Some(Instant::now()),
        };
        if let Some(deadline) = deadline {
            if let Some(kv) = kv {
                let step = self.chain_step(store, kv, deadline);
                let settled = step.step.landed_all || step.step.failed;
                out.progressed |= settled;
                out.resync |= settled;
                out.invalidations += step.invalidations;
                out.offload_failures += step.offload_failures;
                out.record_chain_failed(step.failed_id);
            }
            if matches!(tick, Tick::Wait { .. }) {
                let second = self.poll_offload(store);
                out.progressed |= second.landed;
                out.resync |= second.resync;
                out.state_landed_ok += second.state_landed_ok;
                out.offload_failures += second.offload_failures;
                if second.state_failed_id.is_some() {
                    out.state_failed_id = second.state_failed_id;
                }
                out.record_chain_failed(second.chain_failed_id);
            }
        }
        if matches!(tick, Tick::Full { .. }) {
            self.tier.measure();
        }
        out
    }

    /// The blocks the chain's writes in flight free once they land: the marked
    /// chain of a disk-only state whose blocks are in flight (Splash
    /// `Returning::landingPages`).
    pub(crate) fn landing_pages(
        &self,
        store: &HybridStateStore<C::State>,
        adapter: Option<&PagedKVCacheAdapter>,
    ) -> u32 {
        let Some(entry) = self
            .chain
            .as_ref()
            .filter(|cursor| !cursor.in_flight.is_empty())
            .and_then(|cursor| store.get(cursor.id))
            .filter(|entry| entry.ram().is_none())
        else {
            return 0;
        };
        adapter
            .and_then(|adapter| adapter.freed_by_marks(&[entry.mark()]).first().copied())
            .unwrap_or(0)
    }

    /// What the RAM copies whose copies ahead are in flight free once those
    /// land (Splash `Cache::copyingAheadBytes`).
    pub(crate) fn copying_ahead_bytes(&self, store: &HybridStateStore<C::State>) -> u64 {
        self.pending
            .as_ref()
            .and_then(|write| store.get(write.id))
            .filter(|entry| entry.ram().is_some())
            .map_or(0, Entry::state_bytes)
    }

    /// Splash `Cache::persist` with `StateCache::persist`: a copy ahead of
    /// `id`'s RAM copy and its chain. When `head` is true, a persistent prompt
    /// head stays needed even after deeper ordinary turns supersede it.
    pub(crate) fn persist(
        &mut self,
        store: &mut HybridStateStore<C::State>,
        kv: Option<&KvPoolHandle>,
        id: EntryId,
        head: bool,
        ctx: C::EncodeCtx<'_>,
    ) -> PersistStep {
        let mut result = PersistStep {
            status: PersistStatus::Refused,
            staged: false,
            resync: false,
            invalidations: 0,
            offload_failures: 0,
        };
        let settle = |engine: &mut Self, mut res: PersistStep, status: PersistStatus| {
            if engine.chain.as_ref().is_some_and(|cursor| cursor.id == id) {
                engine.chain = None;
            }
            res.status = status;
            res
        };
        let needed = if head {
            store.get(id).is_some_and(|entry| !entry.checkpoint())
        } else {
            newest_point(store, id)
        };
        if !needed {
            return settle(self, result, PersistStatus::Unneeded);
        }
        if !self.writable() {
            return settle(self, result, PersistStatus::Refused);
        }
        let room = self.persist_room();
        let Some(entry) = store.get(id) else {
            return settle(self, result, PersistStatus::Unneeded);
        };
        if !self.point_fits(entry, room) {
            return settle(self, result, PersistStatus::Refused);
        }
        let yields = store.entries().any(|other| {
            other.id() != id
                && other.ram().is_none()
                && other.disk().is_some_and(|copy| copy.chain_owed)
        });
        if entry.disk().is_none() {
            if yields || self.writing() {
                result.status = PersistStatus::Busy;
                return result;
            }
            match self.stage(store, id, room, true, ctx) {
                Staged::Started => {
                    result.staged = true;
                    result.resync = true;
                }
                Staged::Busy => {
                    result.status = PersistStatus::Busy;
                    return result;
                }
                Staged::Refused => return settle(self, result, PersistStatus::Refused),
            }
        }
        let Some(copy) = store.get(id).and_then(|entry| entry.disk().copied()) else {
            return settle(self, result, PersistStatus::Refused);
        };
        if !copy.chain_owed {
            return if copy.landed {
                settle(self, result, PersistStatus::Durable)
            } else {
                result.status = PersistStatus::Started;
                result
            };
        }
        if yields {
            result.status = if copy.landed {
                PersistStatus::Busy
            } else {
                PersistStatus::Started
            };
            return result;
        }
        if self.chain.as_ref().is_none_or(|cursor| cursor.id != id)
            && let Some(entry) = store.get(id)
        {
            self.chain = Some(ChainCursor {
                id,
                keys: chain_keys(ColdGroup::Kv, self.tier.kv_fingerprint, entry),
                landed: 0,
                in_flight: HashSet::new(),
                room,
            });
        }
        let step = match kv {
            Some(kv) => self.chain_step(store, kv, Instant::now()),
            None => EngineChainStep::default(),
        };
        result.invalidations += step.invalidations;
        result.offload_failures += step.offload_failures;
        result.resync |= step.step.failed || step.step.landed_all;
        result.status = if step.step.failed {
            PersistStatus::Refused
        } else if step.step.landed_all {
            PersistStatus::Durable
        } else {
            PersistStatus::Started
        };
        result
    }

    /// Splash `QwenCompositeState::canSnapshotToDisk`: the tier takes writes
    /// and the staging buffer is free.
    pub(crate) fn can_snapshot_to_disk(&self) -> bool {
        self.writable() && self.staging.is_some()
    }

    /// Waits up to 50 ms, and never past `deadline`, for the writer's next
    /// landing, and leaves it on the channel for the next poll. False once
    /// `deadline` passed.
    pub(crate) fn await_landing(&self, deadline: Instant) -> bool {
        let now = Instant::now();
        if now >= deadline {
            return false;
        }
        if let Ok(landing) = self
            .landed_rx
            .recv_timeout((deadline - now).min(Duration::from_millis(50)))
        {
            let _ = self.landed_tx.send(landing);
        }
        true
    }

    /// Reads `id`'s state object into the single reusable staging buffer and
    /// passes the loaded bytes to `decode`. Caller must have ensured the
    /// staging buffer is free (via [`Self::await_landing`] + poll).
    pub(crate) fn read_into_staging<R>(
        &mut self,
        store: &HybridStateStore<C::State>,
        id: EntryId,
        decode: impl FnOnce(&C, &[u64], &Entry<C::State>, DiskCopy, &[u8]) -> Option<R>,
    ) -> Option<R> {
        let entry = store.get(id)?;
        let copy = *entry.disk()?;
        let layout = self.layout(entry.depth())?;
        let mut staging = self.staging.take()?;
        let read = (|| {
            self.tier.manager.load_state(
                copy.key,
                self.tier.state_fingerprint,
                &layout,
                entry.extra_keys(),
                &mut staging,
            )?;
            decode(&self.codec, &self.spans, entry, copy, &staging)
        })();
        self.staging = Some(staging);
        read
    }

    /// Adopts persistent state objects of this engine's layout into an empty
    /// store.
    pub(crate) fn adopt_all(
        &mut self,
        store: &mut HybridStateStore<C::State>,
        part_pages: bool,
    ) -> u64 {
        if !self.tier.persistent || store.entries().next().is_some() {
            return 0;
        }
        let tier = Arc::clone(&self.tier);
        let spans = &self.spans;
        let codec = &self.codec;
        adopt_states_on(store, &tier, spans, &self.geometry, part_pages, |label| {
            codec.adopt_copy(spans, label)
        })
    }

    /// Splash `StateCache::erase`: `id` leaves with its disk copy.
    pub(crate) fn erase(&self, store: &mut HybridStateStore<C::State>, id: EntryId) -> bool {
        if let Some(copy) = store.get(id).and_then(|entry| entry.disk().copied())
            && copy.landed
        {
            self.tier.manager.remove(&copy.key);
        }
        store.remove(id)
    }

    /// Splash `StateCache::discardState`: a state with a RAM copy loses only
    /// its disk copy, any other goes. Returns `true` when the entry existed.
    pub(crate) fn discard_state(
        &mut self,
        store: &mut HybridStateStore<C::State>,
        id: EntryId,
    ) -> bool {
        if store.get(id).is_none() {
            return false;
        }
        if self.chain.as_ref().is_some_and(|cursor| cursor.id == id) {
            self.chain = None;
        }
        drop_disk_copy(store, &self.tier, id);
        true
    }

    /// Strips every disk copy (dropping disk-only entries) and clears any
    /// active chain cursor after closing a temporary tier.
    pub(crate) fn strip_disk_copies(&mut self, store: &mut HybridStateStore<C::State>) {
        let copies: Vec<EntryId> = store
            .entries()
            .filter(|entry| entry.disk().is_some())
            .map(Entry::id)
            .collect();
        for id in copies {
            store.set_disk(id, None);
            if store.get(id).is_some_and(|entry| entry.ram().is_none()) {
                store.remove(id);
            }
        }
        self.chain = None;
        self.pending_replacement = None;
    }
}

// ---------------------------------------------------------------------------
// Governor-clocked write-behind scheduler (Splash `WriteBehind`).
// ---------------------------------------------------------------------------

/// Splash `WriteBehind::kDelayMilliseconds`.
pub(crate) const DELAY_SECONDS: f64 = 10.0;

/// Splash `WriteBehind::kMinimumTokens`: a shorter point recomputes faster
/// than its state writes.
#[cfg(not(test))]
pub(crate) const MINIMUM_TOKENS: u32 = 2048;
#[cfg(test)]
pub(crate) const MINIMUM_TOKENS: u32 = 32;

/// Splash `WriteBehindSnapshot`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct WriteBehindSnapshot {
    pub(crate) waiting: u32,
    pub(crate) durable: u64,
    pub(crate) unneeded: u64,
    pub(crate) refused: u64,
}

/// Why the last step stopped before the points due (Splash
/// `WriteBehind::Stop`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Stop {
    #[default]
    None,
    ClaimWaiting,
    Covered,
    /// The first point due and not in use waits for a landing.
    Stalled,
}

#[derive(Clone, Copy, Debug)]
struct Point {
    id: EntryId,
    due: f64,
    head: bool,
}

#[derive(Default)]
pub(crate) struct WriteBehind {
    waiting: VecDeque<Point>,
    stopped: Stop,
    /// Copies are made, as last asked.
    copying: bool,
    /// The demand the last step was asked with.
    demand_bytes: u64,
    counters: WriteBehindSnapshot,
}

impl WriteBehind {
    pub(crate) fn new(persistent: bool) -> Self {
        Self {
            copying: persistent,
            ..Self::default()
        }
    }

    pub(crate) fn snapshot(&self) -> WriteBehindSnapshot {
        WriteBehindSnapshot {
            waiting: u32::try_from(self.waiting.len()).unwrap_or(u32::MAX),
            ..self.counters
        }
    }

    #[cfg(test)]
    pub(crate) fn stopped(&self) -> Stop {
        self.stopped
    }

    /// Counts a step that settled its point; false while it is in progress.
    pub(crate) fn settle(&mut self, status: PersistStatus) -> bool {
        match status {
            PersistStatus::Durable => self.counters.durable += 1,
            PersistStatus::Unneeded => self.counters.unneeded += 1,
            PersistStatus::Refused => self.counters.refused += 1,
            PersistStatus::Started | PersistStatus::Busy => return false,
        }
        true
    }

    /// Splash `WriteBehind::nextWakeup`: when a step next has work no
    /// landing wakes the model for, the next point not in use falling due.
    pub(crate) fn next_wakeup<S: StateBytes>(
        &self,
        store: &HybridStateStore<S>,
        persistent: bool,
    ) -> Option<f64> {
        if !self.copying || matches!(self.stopped, Stop::ClaimWaiting | Stop::Covered) {
            return None;
        }
        let stalled = self.stopped == Stop::Stalled;
        self.waiting
            .iter()
            .find(|point| !point_in_use(store, persistent, point.id))
            .and_then(|point| (!stalled).then_some(point.due))
    }

    /// Splash `WriteBehind::published`: queues an ordinary point published or
    /// reused `tokens` deep at `now`. Persistent prompt heads (`head && persistent`)
    /// are queued even below `MINIMUM_TOKENS` and survive superseding turns.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_published<S: StateBytes>(
        &mut self,
        store: &HybridStateStore<S>,
        writable: bool,
        persistent: bool,
        id: EntryId,
        tokens: u32,
        now: f64,
        head: bool,
    ) {
        let head = head && persistent;
        if !writable
            || (tokens < MINIMUM_TOKENS && !head)
            || store.get(id).is_none_or(|entry| entry.checkpoint())
        {
            return;
        }
        if let Some(existing) = self.waiting.iter_mut().find(|point| point.id == id) {
            if head {
                existing.head = true;
            }
            return;
        }
        self.copying = persistent || self.demand_bytes != 0;
        if !self.copying {
            let mut unneeded = 0;
            self.waiting.retain(|point| {
                let keep = point.head || newest_point(store, point.id);
                unneeded += u64::from(!keep);
                keep
            });
            self.counters.unneeded += unneeded;
        }
        self.waiting.push_back(Point {
            id,
            due: now + DELAY_SECONDS,
            head,
        });
    }

    /// Splash `WriteBehind::run`: steps the points due at `now` and not in
    /// use, in publication order, unless a claim waits for memory on its way
    /// back.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn step<Ctx>(
        &mut self,
        ctx: &mut Ctx,
        now: f64,
        demand_bytes: u64,
        claim_waiting: bool,
        persistent: bool,
        is_in_use: impl Fn(&Ctx, EntryId) -> bool,
        is_demanded: impl Fn(&Ctx, u64) -> bool,
        mut persist: impl FnMut(&mut Ctx, EntryId, bool) -> PersistStatus,
    ) -> bool {
        self.demand_bytes = demand_bytes;
        self.copying = persistent || demand_bytes != 0;
        self.stopped = Stop::None;
        if !self.copying {
            return false;
        }
        if claim_waiting {
            self.stopped = Stop::ClaimWaiting;
            return false;
        }
        let mut settled = false;
        let mut index = 0;
        while let Some(point) = self.waiting.get(index).copied() {
            if point.due > now {
                break;
            }
            if is_in_use(ctx, point.id) {
                index += 1;
                continue;
            }
            if !persistent && !is_demanded(ctx, self.demand_bytes) {
                self.stopped = Stop::Covered;
                break;
            }
            if !self.settle(persist(ctx, point.id, point.head)) {
                self.stopped = Stop::Stalled;
                break;
            }
            self.waiting.remove(index);
            settled = true;
        }
        settled
    }

    /// Splash `WriteBehind::flush` at a clean stop: copies every waiting
    /// point, newest first, until each is durable or given up.
    pub(crate) fn flush<Ctx>(
        &mut self,
        ctx: &mut Ctx,
        mut persist: impl FnMut(&mut Ctx, EntryId, bool) -> PersistStatus,
    ) -> bool {
        loop {
            let Some(point) = self.waiting.back().copied() else {
                return true;
            };
            if !self.settle(persist(ctx, point.id, point.head)) {
                return false;
            }
            self.waiting.pop_back();
        }
    }
}

/// A temporary tier's point whose turn has not ended.
pub(crate) fn point_in_use<S: StateBytes>(
    store: &HybridStateStore<S>,
    persistent: bool,
    id: EntryId,
) -> bool {
    !persistent
        && store
            .get(id)
            .is_some_and(|entry| store.in_use(entry.mark()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::disk_tier::{CacheDisk, DiskTierOptions, PersistentIdentity, open_disk_tier};
    use mlx_paged_attn::ColdCacheFingerprint;

    #[derive(Clone)]
    struct DummyState(u64);

    impl StateBytes for DummyState {
        fn state_bytes(&self) -> u64 {
            self.0
        }
    }

    struct DummyCodec;

    impl StateDiskCodec for DummyCodec {
        type State = DummyState;
        type EncodeCtx<'a> = ();

        fn encode_into(
            &self,
            _ctx: (),
            _spans: &[u64],
            _entry: &Entry<DummyState>,
            _state: &DummyState,
            into: &mut [u8],
        ) -> Option<u32> {
            into.fill(0x5A);
            Some(0)
        }

        fn adopt_copy(&self, spans: &[u64], _label: &ColdStateLabel) -> Option<(u64, u32)> {
            Some((spans.iter().sum(), 0))
        }
    }

    #[test]
    fn persist_head_survives_superseding_ordinary_turn() {
        let temp_root =
            std::env::temp_dir().join(format!("state-disk-head-persist-{}", uuid::Uuid::new_v4()));
        let cache_dir = temp_root.join("cache");
        std::fs::create_dir_all(&cache_dir).expect("create cache dir");

        let opts = DiskTierOptions {
            cache_disk: CacheDisk::Bytes(64u64 << 20),
            persistent: true,
            cache_dir: Some(cache_dir),
            temp_root: Some(temp_root.clone()),
        };
        let fp = ColdCacheFingerprint::from_components([b"dummy-head-persist".as_slice()]);
        let id_fn = move || {
            Some(PersistentIdentity {
                namespace: fp.to_hex()[..32].to_string(),
                state: fp,
                kv: fp,
            })
        };
        let tier = open_disk_tier(&opts, id_fn, 64).expect("open persistent tier");
        let mut engine = StateDiskEngine::new(
            Arc::clone(&tier),
            vec![64],
            "dummy:v1".to_string(),
            64,
            DummyCodec,
        )
        .expect("engine");
        let mut store = HybridStateStore::<DummyState>::new(16);

        // Insert a short persistent head (16 tokens < MINIMUM_TOKENS = 32), a regular superseded
        // turn (32 tokens), and a deeper turn (48 tokens) extending the same conversation.
        let tokens: Vec<u32> = (100u32..148u32).collect();
        let head_id = store
            .insert(
                Lineage::text(&tokens[..16], 0),
                DummyState(64),
                StateClass::Ordinary,
            )
            .expect("insert head")
            .id;
        let mid_id = store
            .insert(
                Lineage::text(&tokens[..32], 0),
                DummyState(64),
                StateClass::Ordinary,
            )
            .expect("insert mid")
            .id;
        let tip_id = store
            .insert(
                Lineage::text(&tokens[..48], 0),
                DummyState(64),
                StateClass::Ordinary,
            )
            .expect("insert tip")
            .id;
        assert!(
            store.superseded(head_id),
            "head_id is superseded by deeper turns"
        );
        assert!(store.superseded(mid_id), "mid_id is superseded by tip_id");
        assert!(!store.superseded(tip_id), "tip_id is newest");

        // 1. WriteBehind queues a persistent head even below MINIMUM_TOKENS, and upgrades an
        //    existing queued point's head flag when re-published with `head = true`.
        let mut wb = WriteBehind::new(true);
        wb.record_published(&store, true, true, head_id, 16, 0.0, false);
        assert_eq!(
            wb.snapshot().waiting,
            0,
            "non-head below MINIMUM_TOKENS is skipped"
        );
        wb.record_published(&store, true, true, head_id, 16, 0.0, true);
        assert_eq!(
            wb.snapshot().waiting,
            1,
            "persistent head below MINIMUM_TOKENS is queued"
        );
        wb.record_published(&store, true, true, mid_id, 32, 0.0, false);
        wb.record_published(&store, true, true, tip_id, 48, 0.0, false);
        assert_eq!(wb.snapshot().waiting, 3);

        // 2. Directly verify `StateDiskEngine::persist(..., head)`:
        //    - non-head superseded `mid_id` returns `Unneeded` without writing.
        //    - persistent head `head_id` returns `Started` (persists despite `superseded`).
        let mid_res = engine.persist(&mut store, None, mid_id, false, ());
        assert_eq!(mid_res.status, PersistStatus::Unneeded);
        assert!(store.get(mid_id).unwrap().disk().is_none());

        let head_res = engine.persist(&mut store, None, head_id, true, ());
        assert_eq!(
            head_res.status,
            PersistStatus::Started,
            "persistent head must persist even when superseded"
        );
        assert!(store.get(head_id).unwrap().disk().is_some());

        // 3. Verify `WriteBehind::step` forwards `(id, head)` accurately.
        let mut seen = Vec::new();
        wb.step(
            &mut seen,
            DELAY_SECONDS + 1.0,
            0,
            false,
            true,
            |_, _| false,
            |_, _| true,
            |log, id, head| {
                log.push((id, head));
                if head || id == tip_id {
                    PersistStatus::Durable
                } else {
                    PersistStatus::Unneeded
                }
            },
        );
        assert_eq!(
            seen,
            vec![(head_id, true), (mid_id, false), (tip_id, false)]
        );
        assert_eq!(wb.snapshot().durable, 2);
        assert_eq!(wb.snapshot().unneeded, 1);
        assert_eq!(wb.snapshot().waiting, 0);

        let _ = tier.close(Instant::now() + Duration::from_secs(5));
        let _ = std::fs::remove_dir_all(&temp_root);
    }
}
