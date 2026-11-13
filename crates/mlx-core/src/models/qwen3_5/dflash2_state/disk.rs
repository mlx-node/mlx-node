//! A loaded model's DFlash2 states on its SSD tier (Splash `StateCache`
//! disk copies): the tier, the state object's shape and the one staging
//! buffer every state write and read goes through (Splash
//! `QwenStateStorage::staging_`).

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender};
use std::time::{Duration, Instant};

use mlx_paged_attn::{
    ColdCacheKey, ColdGroup, ColdStateLabel, ColdStateLayout, ColdStateObject, EnqueueRefused,
    Landed, MakeRoom, state_object_bytes,
};

use crate::array::DType;
use crate::array::memory::WeightsResident;
use crate::cold_tier::{CheckpointFiles, CheckpointLoadGuard, ShardIdentity};
use crate::disk_tier::{
    CacheNamespace, DiskTier, DiskTierOptions, DraftNamespace, PersistentIdentity, cache_namespace,
    open_disk_tier, state_geometry,
};
use crate::models::qwen3_5::config::Qwen3_5Config;
use crate::models::qwen3_5::dflash2::{DFlash2Config, DFlash2Model};
use crate::models::qwen3_5::dflash2_turn::DFlash2Target;
use crate::models::qwen3_5::draft_context::DraftContextWindow;
use crate::models::qwen3_5::gdn_sidecar::{self, GdnSidecarGeometry};
use crate::models::qwen3_5::layer_cache::Qwen3_5LayerCache;
use crate::transformer::paged_kv_cache_adapter::{
    ColdContextBacking, ColdTierContext, PagedKVCacheAdapter,
};

use super::paged::{DFlash2States, PagedAnchor};
use super::store::{DiskCopy, Entry, EntryId, HybridStateStore, Lineage, StateClass};
use super::write_behind::WriteBehind;

/// The model's state tier: one per model, so one state write in flight.
pub(crate) struct StateDisk {
    pub(crate) tier: Arc<DiskTier>,
    /// Splash `stateSpans`: the GDN cell, then the draft context window; the
    /// cell alone on a load without a draft.
    pub(crate) spans: Vec<u64>,
    /// The `gdn`, `draft` and `context` lines every object carries.
    pub(crate) geometry: String,
    /// The GDN cell the first span holds.
    gdn: GdnSidecarGeometry,
    /// The rows the context window holds at most.
    window_slots: u32,
    /// A KV block object's bytes on the tier, at most.
    block_bytes: u64,
    /// `None` while the write in flight holds it.
    staging: Option<Vec<u8>>,
    landed_tx: Sender<Landed>,
    landed_rx: Receiver<Landed>,
    /// The state write in flight (Splash `StateCache::pending_`).
    pending: Option<PendingWrite>,
    /// The chain the tier copies under a disk state (Splash
    /// `Cache::persisting_`).
    chain: Option<ChainCursor>,
    /// The landing pages the governor last read from this tier.
    published_landing: u32,
    pub(crate) counters: DiskCounters,
    pub(crate) write_behind: WriteBehind,
    /// The state writes that landed, in landing order.
    #[cfg(test)]
    pub(crate) landed_states: Vec<ColdCacheKey>,
}

struct PendingWrite {
    id: EntryId,
    key: ColdCacheKey,
}

/// A disk state's chain on its way to the tier (Splash
/// `Cache::PersistCursor`): taken once, walked from the first block whose
/// copy has not landed.
struct ChainCursor {
    id: EntryId,
    /// The chain's KV objects, in block order.
    keys: Vec<ColdCacheKey>,
    /// Every block below it is on the tier.
    landed: usize,
    in_flight: HashSet<ColdCacheKey>,
    room: MakeRoom,
}

/// Splash `StateCacheSnapshot` disk counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct DiskCounters {
    pub(crate) offloads: u64,
    pub(crate) offload_failures: u64,
    pub(crate) invalidations: u64,
    pub(crate) promotions: u64,
    pub(crate) promotions_skipped: u64,
    /// States a persistent tier took back at load (Splash
    /// `CacheAdoption::states`).
    pub(crate) adopted: u64,
}

impl StateDisk {
    pub(crate) fn new(
        tier: Arc<DiskTier>,
        gdn: GdnSidecarGeometry,
        window_bytes: Option<u64>,
        window_slots: u32,
        geometry: String,
        block_bytes: u64,
    ) -> Option<Self> {
        let mut spans = vec![gdn.cell_bytes()?];
        spans.extend(window_bytes);
        let bytes = usize::try_from(spans.iter().sum::<u64>()).ok()?;
        let (landed_tx, landed_rx) = std::sync::mpsc::channel();
        let write_behind = WriteBehind::new(tier.persistent);
        Some(Self {
            tier,
            spans,
            geometry,
            gdn,
            window_slots,
            block_bytes,
            staging: Some(vec![0; bytes]),
            landed_tx,
            landed_rx,
            pending: None,
            chain: None,
            published_landing: 0,
            counters: DiskCounters::default(),
            write_behind,
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

    /// The tier takes writes (Splash `QwenCompositeState::writable`: no write
    /// budget, so it takes every write while open).
    pub(crate) fn writable(&self) -> bool {
        self.tier.is_open()
    }

    /// A state write is in flight (Splash `StateCache::writing`).
    pub(crate) fn writing(&self) -> bool {
        self.pending.is_some()
    }

    /// Only a persistent tier makes room for a copy ahead (Splash
    /// `Cache::persist`); a reclaim's write always does.
    fn persist_room(&self) -> MakeRoom {
        if self.tier.persistent {
            MakeRoom::Evict
        } else {
            MakeRoom::Refuse
        }
    }

    /// A state object's layout at `entry`.
    fn layout(&self, depth: usize) -> Option<ColdStateLayout> {
        Some(ColdStateLayout {
            boundary_tokens: u32::try_from(depth).ok()?,
            spans: self.spans.clone(),
            geometry: self.geometry.clone(),
        })
    }

    /// The key of `entry`'s state object: the state group's chain over its
    /// blocks.
    fn state_key<S>(&self, entry: &Entry<S>) -> Option<ColdCacheKey> {
        chain_keys(ColdGroup::DFlash2State, self.tier.state_fingerprint, entry)
            .last()
            .copied()
    }

    /// Every block of `entry`'s chain is on the tier.
    fn chain_on_disk<S>(&self, entry: &Entry<S>) -> bool {
        missing_blocks(&self.tier, entry) == 0
    }

    /// Splash `Cache::pointFits`: a tier that makes room holds the state
    /// with its whole chain under its limit; one that does not has room
    /// left for what of them is not on it yet. Either way the volume holds
    /// what is not on the tier yet above the writer's reserve, after what
    /// making room frees (Splash `DiskBudget::reportFull`: a full volume
    /// refuses the write before it is made).
    fn point_fits<S>(&self, entry: &Entry<S>, room: MakeRoom) -> bool {
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
}

/// `entry`'s chain keys in `group`: one per whole block, with that block's
/// image keys, the salt on block 0 (the adapter's cold walk keys a block
/// alike).
fn chain_keys<S>(
    group: ColdGroup,
    fingerprint: mlx_paged_attn::ColdCacheFingerprint,
    entry: &Entry<S>,
) -> Vec<ColdCacheKey> {
    lineage_keys(
        group,
        fingerprint,
        Lineage {
            tokens: entry.tokens(),
            extra_keys: entry.extra_keys(),
            salt: entry.salt(),
        },
        entry.block_size(),
    )
}

fn lineage_keys(
    group: ColdGroup,
    fingerprint: mlx_paged_attn::ColdCacheFingerprint,
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

fn missing_blocks<S>(tier: &DiskTier, entry: &Entry<S>) -> usize {
    missing_keys(tier, &chain_keys(ColdGroup::Kv, tier.kv_fingerprint, entry))
}

fn missing_keys(tier: &DiskTier, keys: &[ColdCacheKey]) -> usize {
    keys.iter()
        .filter(|key| !tier.manager.contains_in(key, ColdGroup::Kv))
        .count()
}

fn label_of<S>(entry: &Entry<S>) -> ColdStateLabel {
    ColdStateLabel {
        tokens: entry.tokens().to_vec(),
        extra_keys: entry.extra_keys().to_vec(),
        salt: entry.salt(),
        last_used: entry.last_used(),
        checkpoint: entry.checkpoint(),
    }
}

/// Splash `StateCache::reclaimWrites`: a RAM copy without a disk copy, on a
/// tier that takes writes and holds it with its chain. The walk's listing
/// and [`reclaim`] both follow it (Splash `Cache::stateReclaim`).
pub(crate) fn reclaim_writes(states: &DFlash2States, id: EntryId) -> bool {
    let Some(disk) = states.disk.as_ref().filter(|disk| disk.writable()) else {
        return false;
    };
    states.store.get(id).is_some_and(|entry| {
        entry.ram().is_some() && entry.disk().is_none() && disk.point_fits(entry, MakeRoom::Evict)
    })
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

/// Splash `StateCache::reclaim`: a RAM copy whose own write is in flight
/// waits for it; one with a landed disk copy goes; a writable one is
/// written while the staging buffer is free, its RAM free at once (the
/// write reads its own copy), and waits while it is not; any other leaves
/// with no copy, as does one the tier cannot hold with its chain.
pub(crate) fn reclaim<T: DFlash2Target>(target: &mut T, id: EntryId) -> Reclaimed {
    let states = target.dflash2_states();
    let Some(entry) = states.store.get(id).filter(|entry| entry.ram().is_some()) else {
        return Reclaimed::Nothing;
    };
    if let Some(copy) = entry.disk() {
        if !copy.landed {
            return Reclaimed::Pending;
        }
        target.dflash2_states_mut().store.drop_ram(id);
        target.sync_gdn_state_marks();
        return Reclaimed::RamDropped;
    }
    let writable = reclaim_writes(states, id);
    if writable && states.disk.as_ref().is_some_and(StateDisk::writing) {
        return Reclaimed::Pending;
    }
    if writable {
        match stage(target, id, MakeRoom::Evict, false) {
            Staged::Started => return Reclaimed::Written,
            Staged::Busy => return Reclaimed::Pending,
            Staged::Refused => {}
        }
    }
    target.dflash2_states_mut().store.remove(id);
    target.sync_gdn_state_marks();
    Reclaimed::Erased
}

/// What [`stage`] did with a write.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Staged {
    Started,
    /// The staging buffer is out: nothing changed.
    Busy,
    /// No write: the tier is closed or the state does not encode.
    Refused,
}

/// Splash `StateCache::startWrite`: encodes `id`'s RAM copy into the
/// staging buffer and hands it to the tier's writer; its disk copy is in
/// flight from then on. Without `keep_ram` the RAM copy goes at once.
pub(crate) fn stage<T: DFlash2Target>(
    target: &mut T,
    id: EntryId,
    room: MakeRoom,
    keep_ram: bool,
) -> Staged {
    let config = target.dense_config().into_owned();
    let states = target.dflash2_states_mut();
    let Some(disk) = states.disk.as_mut() else {
        return Staged::Refused;
    };
    let Some(entry) = states.store.get(id) else {
        return Staged::Refused;
    };
    let (Some(anchor), Some(layout), Some(key)) = (
        entry.ram(),
        disk.layout(entry.depth()),
        disk.state_key(entry),
    ) else {
        return Staged::Refused;
    };
    let Some(mut staging) = disk.staging.take() else {
        return Staged::Busy;
    };
    let cell = (disk.spans[0] as usize).min(staging.len());
    let (cell_span, window_span) = staging.split_at_mut(cell);
    let encoded = gdn_sidecar::encode_into(&config, &disk.gdn, &anchor.linear, cell_span)
        .unwrap_or(false)
        && match anchor.window.as_ref() {
            Some(window) => window.write_bytes(window_span).unwrap_or(false),
            None => window_span.is_empty(),
        };
    if !encoded {
        disk.staging = Some(staging);
        return Staged::Refused;
    }
    let label = label_of(entry);
    let copy = DiskCopy {
        key,
        bytes: state_object_bytes(&layout, &label),
        landed: false,
        window_begin: anchor.window.as_ref().map_or_else(
            || u32::try_from(entry.depth()).unwrap_or(u32::MAX),
            DraftContextWindow::begin,
        ),
        chain_owed: !disk.chain_on_disk(entry),
    };
    let object = ColdStateObject {
        key,
        fingerprint: disk.tier.state_fingerprint,
        layout,
        label,
        payload: staging,
    };
    match disk
        .tier
        .manager
        .enqueue_state(object, room, disk.landed_tx.clone())
    {
        Ok(()) => {
            disk.pending = Some(PendingWrite { id, key });
            disk.counters.offloads += 1;
            states.store.set_disk(id, Some(copy));
            if !keep_ram {
                states.store.drop_ram(id);
            }
            target.sync_gdn_state_marks();
            Staged::Started
        }
        Err(EnqueueRefused::Busy(object)) => {
            disk.staging = Some(object.payload);
            Staged::Busy
        }
        Err(EnqueueRefused::Gone(object)) => {
            disk.staging = Some(object.payload);
            Staged::Refused
        }
    }
}

/// Splash `StateCache::pollOffload` and `Cache::pollTransfers`: first the
/// state objects the tier's evictions cleared (Splash `Cache::freeDiskSpace`:
/// a state with a RAM copy keeps it, one without goes), then the
/// writer's landings. A landed state's disk copy stands; a failed one
/// leaves, and a state left with no copy goes. A chain block that failed
/// fails its point: the state's disk copy leaves. True when anything
/// landed.
pub(crate) fn poll_offload<T: DFlash2Target>(target: &mut T) -> bool {
    let DFlash2States { store, disk, .. } = target.dflash2_states_mut();
    let Some(disk) = disk.as_mut() else {
        return false;
    };
    let mut landed = false;
    let mut resync = false;
    for key in disk.tier.manager.take_evicted() {
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
        if disk.chain.as_ref().is_some_and(|cursor| cursor.id == id) {
            disk.chain = None;
        }
        resync = true;
    }
    while let Ok(landing) = disk.landed_rx.try_recv() {
        landed = true;
        let (key, ok, payload) = match landing {
            Landed::State { key, ok, payload } => (key, ok, payload),
            Landed::Block { key, ok } => {
                let Some(id) = disk
                    .chain
                    .as_mut()
                    .and_then(|cursor| cursor.in_flight.remove(&key).then_some(cursor.id))
                else {
                    continue;
                };
                if ok {
                    continue;
                }
                disk.counters.offload_failures += 1;
                drop_disk_copy(store, &disk.tier, id);
                disk.chain = None;
                resync = true;
                continue;
            }
        };
        disk.staging = Some(payload);
        let Some(write) = disk.pending.take_if(|write| write.key == key) else {
            continue;
        };
        if ok {
            #[cfg(test)]
            disk.landed_states.push(key);
            match store.disk_mut(write.id).filter(|copy| copy.key == key) {
                Some(copy) => copy.landed = true,
                // The point failed while its state was in flight.
                None => {
                    disk.tier.manager.remove(&key);
                }
            }
            continue;
        }
        disk.counters.offload_failures += 1;
        resync = true;
        store.set_disk(write.id, None);
        if store
            .get(write.id)
            .is_some_and(|entry| entry.ram().is_none())
        {
            store.remove(write.id);
        }
    }
    if resync {
        target.sync_gdn_state_marks();
    }
    landed
}

/// Splash `StateCache::discardDisk` with its slot: `id`'s disk copy leaves
/// the tier, and a state with no RAM copy goes with it.
fn drop_disk_copy(store: &mut HybridStateStore<PagedAnchor>, tier: &DiskTier, id: EntryId) {
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

/// What one [`chain_step`] did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ChainStep {
    pub(crate) enqueued: usize,
    /// The chain is on the tier whole: its mark went.
    pub(crate) landed_all: bool,
    /// The point failed: its state's disk copy went.
    pub(crate) failed: bool,
}

/// Splash `Cache::persist` for the chain under a disk state, on the model
/// thread: a reclaim's chain (the cursor's, else the oldest), else the
/// cursor's copy ahead, else the oldest owed chain. A state whose object
/// left the tier fails its point (making room
/// gives up the oldest point, which may be this one). Each block without a
/// copy is read from its resident page and handed to the writer while the
/// queue has room or `deadline` allows; a walk that finds nothing in flight
/// walks the whole chain once more (a flat-LRU eviction may have taken a
/// block below the cursor). Once the state landed and every block is on
/// the tier, the chain is no longer owed and its mark goes.
pub(crate) fn chain_step<T: DFlash2Target>(target: &mut T, deadline: Instant) -> ChainStep {
    let mut step = ChainStep::default();
    let Some((pool, allocator)) = target.paged_adapter_ref().map(|adapter| {
        (
            Arc::clone(adapter.layer_kv_pool()),
            Arc::clone(adapter.allocator()),
        )
    }) else {
        return step;
    };
    let DFlash2States { store, disk, .. } = target.dflash2_states_mut();
    let Some(disk) = disk.as_mut() else {
        return step;
    };
    let owed = |entry: &Entry<PagedAnchor>| entry.disk().is_some_and(|copy| copy.chain_owed);
    let reclaimed = |entry: &Entry<PagedAnchor>| owed(entry) && entry.ram().is_none();
    let oldest = |pick: &dyn Fn(&Entry<PagedAnchor>) -> bool| {
        store
            .entries()
            .filter(|entry| pick(entry))
            .min_by_key(|entry| entry.last_used())
            .map(Entry::id)
    };
    // A reclaim's chain frees memory once it lands: it goes before a copy
    // ahead (Splash `Cache::persist` yields while a reclaim's write is in
    // flight).
    let current = disk.chain.as_ref().map(|cursor| cursor.id).filter(|&id| {
        store
            .get(id)
            .is_some_and(|entry| reclaimed(entry) || (owed(entry) && oldest(&reclaimed).is_none()))
    });
    let Some(id) = current
        .or_else(|| oldest(&reclaimed))
        .or_else(|| oldest(&owed))
    else {
        disk.chain = None;
        return step;
    };
    let Some((entry, copy)) = store
        .get(id)
        .and_then(|entry| entry.disk().copied().map(|copy| (entry, copy)))
    else {
        return step;
    };
    let StateDisk {
        tier,
        chain,
        landed_tx,
        counters,
        ..
    } = disk;
    if current.is_none() {
        // A reclaim's write makes room; a copy ahead beside its RAM copy
        // only on a persistent tier.
        let room = if entry.ram().is_none() || tier.persistent {
            MakeRoom::Evict
        } else {
            MakeRoom::Refuse
        };
        *chain = Some(ChainCursor {
            id,
            keys: chain_keys(ColdGroup::Kv, tier.kv_fingerprint, entry),
            landed: 0,
            in_flight: HashSet::new(),
            room,
        });
    }
    let manager = &tier.manager;
    let gone = copy.landed && !manager.contains_in(&copy.key, ColdGroup::DFlash2State);
    let mut lost = gone;
    let mut failed = false;
    let mut missing = false;
    let mut in_flight = false;
    if let Some(cursor) = chain.as_mut().filter(|_| !gone) {
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
                    counters.invalidations += 1;
                    lost = true;
                    break 'walk;
                };
                if !manager.has_block_room() && Instant::now() >= deadline {
                    break 'walk;
                }
                match manager.capture_and_enqueue_tracked(
                    &pool,
                    &block,
                    key,
                    tier.kv_fingerprint,
                    tokens,
                    deadline,
                    cursor.room,
                    landed_tx.clone(),
                ) {
                    Ok(true) => {
                        cursor.in_flight.insert(key);
                        in_flight = true;
                        step.enqueued += 1;
                    }
                    Ok(false) => break 'walk,
                    Err(_) => {
                        counters.offload_failures += 1;
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
        drop_disk_copy(store, tier, id);
        *chain = None;
        step.failed = true;
    } else if copy.landed && !missing && !in_flight {
        if let Some(copy) = store.disk_mut(id) {
            copy.chain_owed = false;
        }
        *chain = None;
        step.landed_all = true;
    }
    if step.failed || step.landed_all {
        target.sync_gdn_state_marks();
    }
    step
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

/// One tick of the model's state tier: its landings, the owed chain as
/// `tick` allows, the blocks the writes in flight return published, and
/// the model's progress signaled when anything landed or a chain settled
/// (Splash `signalResourceProgress`). True on progress.
pub(crate) fn tick<T: DFlash2Target>(target: &mut T, tick: Tick) -> bool {
    if target.dflash2_states().disk.is_none() {
        return false;
    }
    let mut progressed = poll_offload(target);
    let deadline = match tick {
        Tick::Poll => None,
        Tick::Wait { deadline } => Some(deadline),
        Tick::Full { .. } => Some(Instant::now()),
    };
    if let Some(deadline) = deadline {
        let step = chain_step(target, deadline);
        progressed |= step.landed_all || step.failed;
        if matches!(tick, Tick::Wait { .. }) {
            progressed |= poll_offload(target);
        }
    }
    if let Tick::Full { command_pages } = tick {
        if let Some(disk) = target.dflash2_states().disk.as_ref() {
            disk.tier.measure();
        }
        let governor = target
            .memory_registration()
            .map_or_else(crate::engine::governor::governor, |guard| guard.governor());
        let now = governor.now();
        let page_bytes = target
            .paged_adapter_ref()
            .and_then(|adapter| adapter.bytes_per_block().ok())
            .unwrap_or(0);
        let demand = copy_demand_bytes(governor, now, command_pages, page_bytes);
        let claim_waiting = governor.ledger(|ledger| ledger.claim_waiting());
        progressed |= super::write_behind::run(target, now, demand, claim_waiting);
        if let Some(guard) = target.memory_registration() {
            let states = target.dflash2_states();
            let wakeup = states.disk.as_ref().and_then(|disk| {
                disk.write_behind
                    .next_wakeup(&states.store, disk.tier.persistent)
            });
            crate::engine::governor::monitor::set_next_wakeup(guard.model_id(), wakeup);
        }
    }
    let pages = landing_pages(&*target);
    let changed = target
        .dflash2_states_mut()
        .disk
        .as_mut()
        .is_some_and(|disk| std::mem::replace(&mut disk.published_landing, pages) != pages);
    if let Some(guard) = target.memory_registration() {
        let governor = guard.governor();
        if changed {
            governor.publish_landing_pages(guard.model_id(), pages);
        }
        if progressed {
            governor.signal_progress(guard.model_id());
        }
    }
    progressed
}

/// The blocks the chain's writes in flight free once they land: the marked
/// chain of a disk-only state whose blocks are in flight (Splash
/// `Returning::landingPages`).
pub(crate) fn landing_pages<T: DFlash2Target>(target: &T) -> u32 {
    let states = target.dflash2_states();
    let Some(entry) = states
        .disk
        .as_ref()
        .and_then(|disk| disk.chain.as_ref())
        .filter(|cursor| !cursor.in_flight.is_empty())
        .and_then(|cursor| states.store.get(cursor.id))
        .filter(|entry| entry.ram().is_none())
    else {
        return 0;
    };
    target
        .paged_adapter_ref()
        .and_then(|adapter| adapter.freed_by_marks(&[entry.mark()]).first().copied())
        .unwrap_or(0)
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

/// The governor's clock, as the model's registration reads it.
pub(crate) fn clock<T: DFlash2Target>(target: &T) -> f64 {
    target
        .memory_registration()
        .map_or_else(crate::engine::governor::governor, |guard| guard.governor())
        .now()
}

/// What the RAM copies whose copies ahead are in flight free once those
/// land (Splash `Cache::copyingAheadBytes`).
pub(crate) fn copying_ahead_bytes(states: &DFlash2States) -> u64 {
    states
        .disk
        .as_ref()
        .and_then(|disk| disk.pending.as_ref())
        .and_then(|write| states.store.get(write.id))
        .filter(|entry| entry.ram().is_some())
        .map_or(0, Entry::state_bytes)
}

/// What [`persist`] did with a point (Splash `PersistStatus`).
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

/// Splash `Cache::newestPoint`: an ordinary state its conversation still
/// resumes from.
pub(crate) fn newest_point(store: &HybridStateStore<PagedAnchor>, id: EntryId) -> bool {
    store.get(id).is_some_and(|entry| !entry.checkpoint()) && !store.superseded(id)
}

/// Splash `Cache::persist` with `StateCache::persist`: a copy ahead of
/// `id`'s RAM copy and its chain. The state goes first, its RAM copy kept;
/// the chain follows over later calls from the first block whose copy has
/// not landed. Only a persistent tier makes room; a point the tier cannot
/// hold is refused (waiting for room would hold back the points after it).
/// While a reclaim's chain is owed, nothing starts unless the state is on
/// the tier already.
pub(crate) fn persist<T: DFlash2Target>(target: &mut T, id: EntryId) -> PersistStatus {
    let states = target.dflash2_states();
    let Some(disk) = states.disk.as_ref() else {
        return PersistStatus::Refused;
    };
    let store = &states.store;
    let settle = |target: &mut T, status: PersistStatus| {
        if let Some(disk) = target.dflash2_states_mut().disk.as_mut()
            && disk.chain.as_ref().is_some_and(|cursor| cursor.id == id)
        {
            disk.chain = None;
        }
        status
    };
    if !newest_point(store, id) {
        return settle(target, PersistStatus::Unneeded);
    }
    if !disk.writable() {
        return settle(target, PersistStatus::Refused);
    }
    let room = disk.persist_room();
    let Some(entry) = store.get(id) else {
        return settle(target, PersistStatus::Unneeded);
    };
    if !disk.point_fits(entry, room) {
        return settle(target, PersistStatus::Refused);
    }
    let yields = store.entries().any(|other| {
        other.id() != id
            && other.ram().is_none()
            && other.disk().is_some_and(|copy| copy.chain_owed)
    });
    if entry.disk().is_none() {
        if yields || disk.writing() {
            return PersistStatus::Busy;
        }
        match stage(target, id, room, true) {
            Staged::Started => {}
            Staged::Busy => return PersistStatus::Busy,
            Staged::Refused => return settle(target, PersistStatus::Refused),
        }
    }
    let Some(copy) = target
        .dflash2_states()
        .store
        .get(id)
        .and_then(|entry| entry.disk().copied())
    else {
        return settle(target, PersistStatus::Refused);
    };
    if !copy.chain_owed {
        return if copy.landed {
            settle(target, PersistStatus::Durable)
        } else {
            PersistStatus::Started
        };
    }
    if yields {
        return if copy.landed {
            PersistStatus::Busy
        } else {
            PersistStatus::Started
        };
    }
    let states = target.dflash2_states_mut();
    if let Some(disk) = states.disk.as_mut()
        && disk.chain.as_ref().is_none_or(|cursor| cursor.id != id)
        && let Some(entry) = states.store.get(id)
    {
        disk.chain = Some(ChainCursor {
            id,
            keys: chain_keys(ColdGroup::Kv, disk.tier.kv_fingerprint, entry),
            landed: 0,
            in_flight: HashSet::new(),
            room,
        });
    }
    let step = chain_step(target, Instant::now());
    if step.failed {
        PersistStatus::Refused
    } else if step.landed_all {
        PersistStatus::Durable
    } else {
        PersistStatus::Started
    }
}

/// Splash `QwenCompositeState::canSnapshotToDisk`: the tier takes writes
/// and the staging buffer is free. No write budget holds a state back, in
/// use or not.
pub(crate) fn can_snapshot_to_disk(states: &DFlash2States) -> bool {
    states
        .disk
        .as_ref()
        .is_some_and(|disk| disk.writable() && disk.staging.is_some())
}

/// Splash `StateCache::publishStateToDisk` with
/// `QwenStateStorage::snapshotToDisk`: the lane's GDN cell and context
/// window, both at the end of `lineage`, go through the staging buffer to
/// the tier as a state with no RAM copy, its write in flight. `None` when
/// nothing was published.
pub(crate) fn publish_from_lane<T: DFlash2Target>(
    target: &mut T,
    sink: &mut (dyn crate::models::qwen3_5::paged_forward::PrefillTapSink + '_),
    lineage: Lineage<'_>,
    class: StateClass,
) -> napi::Result<Option<EntryId>> {
    if !can_snapshot_to_disk(target.dflash2_states()) {
        return Ok(None);
    }
    let row = lineage.tokens.len();
    let config = target.dense_config().into_owned();
    let linear = target.snapshot_live_linear()?;
    let DFlash2States { store, disk, .. } = target.dflash2_states_mut();
    let Some(disk) = disk.as_mut() else {
        return Ok(None);
    };
    let (Some(layout), Some(mut staging)) = (disk.layout(row), disk.staging.take()) else {
        return Ok(None);
    };
    let cell = (disk.spans[0] as usize).min(staging.len());
    let (cell_span, window_span) = staging.split_at_mut(cell);
    let window_begin =
        if gdn_sidecar::encode_into(&config, &disk.gdn, &linear, cell_span).unwrap_or(false) {
            sink.write_lane_window(row, window_span).unwrap_or(None)
        } else {
            None
        };
    let Some(window_begin) = window_begin else {
        disk.staging = Some(staging);
        return Ok(None);
    };
    let block_size = store.block_size();
    let key = lineage_keys(
        ColdGroup::DFlash2State,
        disk.tier.state_fingerprint,
        lineage,
        block_size,
    )
    .last()
    .copied();
    let chain = lineage_keys(ColdGroup::Kv, disk.tier.kv_fingerprint, lineage, block_size);
    let Some(key) = key else {
        disk.staging = Some(staging);
        return Ok(None);
    };
    let copy = DiskCopy {
        key,
        bytes: 0,
        landed: false,
        window_begin,
        chain_owed: missing_keys(&disk.tier, &chain) > 0,
    };
    let cached_bytes = disk.spans.iter().sum();
    let Ok(id) = store.insert_disk(lineage, cached_bytes, class, copy) else {
        disk.staging = Some(staging);
        return Ok(None);
    };
    let label = store.get(id).map(label_of);
    let Some(label) = label else {
        disk.staging = Some(staging);
        return Ok(None);
    };
    if let Some(copy) = store.disk_mut(id) {
        copy.bytes = state_object_bytes(&layout, &label);
    }
    let object = ColdStateObject {
        key,
        fingerprint: disk.tier.state_fingerprint,
        layout,
        label,
        payload: staging,
    };
    match disk
        .tier
        .manager
        .enqueue_state(object, MakeRoom::Evict, disk.landed_tx.clone())
    {
        Ok(()) => {
            disk.pending = Some(PendingWrite { id, key });
            disk.counters.offloads += 1;
        }
        Err(EnqueueRefused::Busy(object) | EnqueueRefused::Gone(object)) => {
            disk.staging = Some(object.payload);
            store.remove(id);
            return Ok(None);
        }
    }
    target.sync_gdn_state_marks();
    Ok(Some(id))
}

/// Takes the staging buffer as a write in flight would.
#[cfg(test)]
pub(crate) fn hold_staging(states: &mut DFlash2States) -> Option<Vec<u8>> {
    states.disk.as_mut()?.staging.take()
}

/// The object a disk copy of `id` holds, read back through the staging
/// buffer's shape.
#[cfg(test)]
pub(crate) fn read_object(states: &DFlash2States, id: EntryId) -> Option<Vec<u8>> {
    let disk = states.disk.as_ref()?;
    let entry = states.store.get(id)?;
    let copy = entry.disk()?;
    let layout = disk.layout(entry.depth())?;
    let mut bytes = vec![0; usize::try_from(disk.spans.iter().sum::<u64>()).ok()?];
    disk.tier.manager.load_state(
        copy.key,
        disk.tier.state_fingerprint,
        &layout,
        entry.extra_keys(),
        &mut bytes,
    )?;
    Some(bytes)
}

/// `anchor` as the tier writes it: the GDN cell, then the window.
#[cfg(test)]
pub(crate) fn encode_anchor(
    states: &DFlash2States,
    config: &Qwen3_5Config,
    anchor: &PagedAnchor,
) -> Option<Vec<u8>> {
    let disk = states.disk.as_ref()?;
    let mut bytes = vec![0; usize::try_from(disk.spans.iter().sum::<u64>()).ok()?];
    let (cell, window) = bytes.split_at_mut(disk.spans[0] as usize);
    (gdn_sidecar::encode_into(config, &disk.gdn, &anchor.linear, cell).ok()?
        && match anchor.window.as_ref() {
            Some(lane) => lane.write_bytes(window).ok()?,
            None => window.is_empty(),
        })
    .then_some(bytes)
}

/// Splash `StateCache::erase`: `id` leaves with its disk copy. The caller
/// re-syncs the marks.
pub(crate) fn erase(states: &mut DFlash2States, id: EntryId) -> bool {
    let DFlash2States { store, disk, .. } = states;
    if let Some(disk) = disk.as_ref()
        && let Some(copy) = store.get(id).and_then(|entry| entry.disk().copied())
        && copy.landed
    {
        disk.tier.manager.remove(&copy.key);
    }
    store.remove(id)
}

/// Splash `StateCache::discardState` after a restore that cannot come
/// back: a state with a RAM copy loses only its disk copy, any other goes.
pub(crate) fn discard_state<T: DFlash2Target>(target: &mut T, id: EntryId) {
    let DFlash2States { store, disk, .. } = target.dflash2_states_mut();
    let Some(disk) = disk.as_mut() else {
        return;
    };
    if store.get(id).is_none() {
        return;
    }
    disk.counters.invalidations += 1;
    if disk.chain.as_ref().is_some_and(|cursor| cursor.id == id) {
        disk.chain = None;
    }
    drop_disk_copy(store, &disk.tier, id);
    target.sync_gdn_state_marks();
}

/// A turn can resume on `id`: a RAM copy, or a disk copy whose object is on
/// the tier or on its way there and whose chain comes back whole, its
/// leading blocks resident and every block past them on the tier.
pub(crate) fn restorable<T: DFlash2Target>(target: &T, id: EntryId) -> bool {
    let states = target.dflash2_states();
    let Some(entry) = states.store.get(id) else {
        return false;
    };
    if entry.ram().is_some() {
        return true;
    }
    let (Some(disk), Some(adapter), Some(copy)) = (
        states.disk.as_ref(),
        target.paged_adapter_ref(),
        entry.disk(),
    ) else {
        return false;
    };
    let manager = &disk.tier.manager;
    if copy.landed && !manager.contains_in(&copy.key, ColdGroup::DFlash2State) {
        return false;
    }
    let resident = entry
        .chain()
        .iter()
        .take_while(|&&hash| adapter.prefix_cached(hash))
        .count();
    chain_keys(ColdGroup::Kv, disk.tier.kv_fingerprint, entry)
        .get(resident..)
        .is_some_and(|keys| {
            keys.iter()
                .all(|key| manager.contains_in(key, ColdGroup::Kv))
        })
}

/// Takes the writer's landings until the write in flight hands the staging
/// buffer back, `deadline` passes or the turn is cancelled. False when the
/// buffer stays out.
fn wait_for_staging<T: DFlash2Target>(target: &mut T, deadline: Instant) -> bool {
    loop {
        let Some(disk) = target.dflash2_states().disk.as_ref() else {
            return false;
        };
        if disk.staging.is_some() {
            return true;
        }
        if !disk.writing() || crate::engine::governor::governor::cancelled() {
            return false;
        }
        if !await_landing(disk, deadline) {
            return false;
        }
        tick(target, Tick::Poll);
    }
}

/// Waits up to 50 ms, and never past `deadline`, for the writer's next
/// landing, and leaves it on the channel for the next poll. False once
/// `deadline` passed.
fn await_landing(disk: &StateDisk, deadline: Instant) -> bool {
    let now = Instant::now();
    if now >= deadline {
        return false;
    }
    if let Ok(landing) = disk
        .landed_rx
        .recv_timeout((deadline - now).min(Duration::from_millis(50)))
    {
        let _ = disk.landed_tx.send(landing);
    }
    true
}

/// Splash `QwenStateStorage::beginRestore` for a disk state: the object is
/// read through the staging buffer, once the write in flight hands it back
/// (no second buffer outside the plan), and checked against the state's
/// layout; its cell is decoded into GDN slots and, with `window`, its
/// context window taken into it. `None` when the object is missing,
/// damaged or of another layout.
pub(crate) fn read_state<T: DFlash2Target>(
    target: &mut T,
    id: EntryId,
    window: Option<&mut DraftContextWindow>,
) -> Option<Vec<Qwen3_5LayerCache>> {
    let deadline = Instant::now()
        + Duration::from_secs_f64(crate::engine::governor::wait::RESOURCE_WAIT_SECONDS);
    if !wait_for_staging(target, deadline) {
        return None;
    }
    let config = target.dense_config().into_owned();
    let DFlash2States { store, disk, .. } = target.dflash2_states_mut();
    let disk = disk.as_mut()?;
    let entry = store.get(id)?;
    let copy = *entry.disk()?;
    let layout = disk.layout(entry.depth())?;
    let depth = u32::try_from(entry.depth()).ok()?;
    let mut staging = disk.staging.take()?;
    let read = (|| {
        disk.tier.manager.load_state(
            copy.key,
            disk.tier.state_fingerprint,
            &layout,
            entry.extra_keys(),
            &mut staging,
        )?;
        let (cell, rest) = staging.split_at(usize::try_from(disk.spans[0]).ok()?);
        let linear = gdn_sidecar::decode_caches(&config, &disk.gdn, cell).ok()??;
        if let Some(window) = window
            && !window.read_bytes(rest, copy.window_begin, depth).ok()?
        {
            return None;
        }
        Some(linear)
    })();
    disk.staging = Some(staging);
    read
}

/// Splash `StateCache::promote`: disk-only `id` takes `anchor` as its RAM
/// copy, and its chain is marked again.
pub(crate) fn promote<T: DFlash2Target>(target: &mut T, id: EntryId, anchor: PagedAnchor) -> bool {
    let states = target.dflash2_states_mut();
    if !states.store.promote(id, anchor) {
        return false;
    }
    if let Some(disk) = states.disk.as_mut() {
        disk.counters.promotions += 1;
    }
    target.sync_gdn_state_marks();
    true
}

/// A promotion the ledger refused: the state stays on disk alone.
pub(crate) fn promotion_skipped(states: &mut DFlash2States) {
    if let Some(disk) = states.disk.as_mut() {
        disk.counters.promotions_skipped += 1;
    }
}

/// Splash `Cache::adopt` at a persistent tier's load: each state object of
/// this layout comes back as a state with no RAM copy, but only with its
/// whole KV chain on the tier, one per block, stamped as its earlier
/// process left it; new stamps continue after the newest. Its window ends
/// at its row and holds the rows before it the window has room for
/// (Splash `QwenStateStorage::adopt`). The tier first fits its limit (Splash
/// `Cache::adopt` → `fitDiskLimit`). An object that does not come back
/// stays on the tier for its quota to take.
pub(crate) fn adopt_all<T: DFlash2Target>(target: &mut T) -> u64 {
    let DFlash2States { store, disk, .. } = target.dflash2_states_mut();
    let Some(disk) = disk.as_mut() else {
        return 0;
    };
    if !disk.tier.persistent || store.entries().next().is_some() {
        return 0;
    }
    let block_size = store.block_size();
    let cached_bytes = disk.spans.iter().sum();
    let tier = Arc::clone(&disk.tier);
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
        let own = lineage_keys(
            ColdGroup::DFlash2State,
            tier.state_fingerprint,
            lineage,
            block_size,
        );
        let chain = lineage_keys(ColdGroup::Kv, tier.kv_fingerprint, lineage, block_size);
        if layout.spans != disk.spans
            || layout.geometry != disk.geometry
            || own.len() * block_size as usize != depth
            || own.last() != Some(&key)
            || missing_keys(&tier, &chain) > 0
        {
            continue;
        }
        let Ok(row) = u32::try_from(depth) else {
            continue;
        };
        let copy = DiskCopy {
            key,
            bytes: state_object_bytes(&layout, &label),
            landed: true,
            window_begin: row - row.min(disk.window_slots),
            chain_owed: false,
        };
        let class = if label.checkpoint {
            StateClass::Checkpoint
        } else {
            StateClass::Ordinary
        };
        if store
            .adopt_disk(lineage, cached_bytes, class, copy, label.last_used)
            .is_ok()
        {
            newest = newest.max(label.last_used);
            adopted += 1;
        }
    }
    store.continue_after(newest);
    disk.counters.adopted += adopted;
    adopted
}

/// Splash `closePersistentCache` with `FdTransport::runFlush` at a clean
/// stop: a persistent tier copies every restore point still waiting,
/// newest first, moving owed chains between landings, until `deadline`,
/// then drains its writer; a temporary tier's copies go with the process,
/// so its writer stops and its directory goes at once, and so does every
/// disk copy. Either way the tier takes no write afterwards. False when a
/// point or a queued write missed `deadline`.
pub(crate) fn close_cache<T: DFlash2Target>(target: &mut T, deadline: Instant) -> bool {
    let Some(tier) = target
        .dflash2_states()
        .disk
        .as_ref()
        .map(|disk| Arc::clone(&disk.tier))
    else {
        return true;
    };
    let mut flushed = true;
    if tier.persistent && tier.is_open() {
        flushed = loop {
            tick(
                target,
                Tick::Wait {
                    deadline: Instant::now(),
                },
            );
            if super::write_behind::flush(target) {
                break true;
            }
            let Some(disk) = target.dflash2_states().disk.as_ref() else {
                break false;
            };
            if !await_landing(disk, deadline) {
                break false;
            }
        };
    }
    let closed = tier.close(deadline);
    poll_offload(target);
    if !tier.persistent {
        let DFlash2States { store, disk, .. } = target.dflash2_states_mut();
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
        if let Some(disk) = disk.as_mut() {
            disk.chain = None;
        }
        target.sync_gdn_state_marks();
    }
    flushed && closed
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

/// What a target load hands its tier.
pub(crate) struct TierLoad<'a> {
    /// The cold fingerprint's family: `qwen3_5` or `qwen3_5_moe`.
    pub family: &'static str,
    /// The config text the KV fingerprint hashes
    /// (`crate::models::qwen3_5::model::cold_tier_config_json`).
    pub config_json: Vec<u8>,
    /// The dense-shaped config the GDN geometry reads.
    pub dense: &'a Qwen3_5Config,
    pub activation: DType,
    /// The DFlash2 draft and its files; `None` for a load without one.
    pub draft: Option<DraftTier<'a>>,
    pub adapter: &'a PagedKVCacheAdapter,
    pub files: CheckpointFiles<'a>,
    pub weights: &'a WeightsResident,
    pub guard: &'a CheckpointLoadGuard<'a>,
}

/// A draft load's draft, as its tier names it.
pub(crate) struct DraftTier<'a> {
    pub model: &'a DFlash2Model,
    pub files: &'a DraftLoadGuard<'a>,
}

/// The draft's files as Splash `assembly.verify` holds every source of the
/// namespace, the draft's with the target's: taken before the draft loads,
/// compared once its identity is read.
pub(crate) struct DraftLoadGuard<'a> {
    dir: &'a Path,
    before_read: Option<HashMap<String, ShardIdentity>>,
}

impl<'a> DraftLoadGuard<'a> {
    pub(crate) fn before_read(dir: &'a Path, enabled: bool) -> Self {
        Self {
            dir,
            before_read: enabled.then(|| draft_identities(dir)).flatten(),
        }
    }

    fn stable_after_fingerprint(&self) -> bool {
        self.before_read.is_some() && self.before_read == draft_identities(self.dir)
    }
}

fn draft_identities(dir: &Path) -> Option<HashMap<String, ShardIdentity>> {
    let mut files = crate::cold_tier::snapshot_shard_identities(dir)?;
    let config = std::fs::metadata(dir.join("config.json")).ok()?;
    files.insert(
        "config.json".to_string(),
        ShardIdentity::from_metadata(&config),
    );
    Some(files)
}

/// Splash `openDiskTier` for a paged load: the adapter's state-backed cold
/// context and the model's state tier, whose landings wake `model`'s
/// thread, or `None` when the tier stays off. A state is the GDN cell and
/// the draft window, the cell alone without a draft.
pub(crate) fn open_state_disk(
    load: &TierLoad<'_>,
    options: &DiskTierOptions,
    model: u64,
) -> Option<(ColdTierContext, StateDisk)> {
    let geometry = gdn_sidecar::geometry(load.dense, load.activation)?;
    let cell = geometry.cell_bytes()?;
    let draft = load.draft.as_ref().map(|draft| &draft.model.config);
    let window = draft.map(DFlash2Config::context_window_bytes);
    let state_bytes = cell.saturating_add(window.unwrap_or(0));
    let draft_geometry = draft.map(|draft| (draft.namespace_geometry(), draft.context_width()));
    let geometry_text = state_geometry(geometry.namespace_gdn(), draft_geometry);
    let pool = load.adapter.layer_kv_pool();
    let cache_dtype = format!("{:?}", pool.cache_dtype());
    let identity = || {
        let target = crate::cold_tier::build_model_fingerprint(
            load.family,
            load.files,
            Some(&load.config_json),
            &crate::cold_tier::ColdTierGeometry {
                block_size: pool.block_size() as u64,
                num_layers: pool.num_layers() as u64,
                num_kv_heads: pool.config().num_kv_heads as u64,
                head_size: pool.config().head_size as u64,
                cache_dtype: cache_dtype.clone(),
            },
            load.weights,
        )?;
        let draft_fingerprint = match load.draft.as_ref() {
            Some(draft) => {
                let mut components = vec![b"dflash2-draft\0".to_vec()];
                crate::cold_tier::push_dir_identity(draft.files.dir, &mut components)?;
                components.push(std::fs::read(draft.files.dir.join("config.json")).ok()?);
                Some(mlx_paged_attn::ColdCacheFingerprint::from_components(
                    components.iter().map(Vec::as_slice),
                ))
            }
            None => None,
        };
        let changed = if !load.guard.stable_after_fingerprint() {
            Some(load.files.label().to_string())
        } else {
            load.draft
                .as_ref()
                .filter(|draft| !draft.files.stable_after_fingerprint())
                .map(|draft| draft.files.dir.display().to_string())
        };
        if let Some(changed) = changed {
            tracing::warn!(
                "{changed}: the model directory changed during load; the SSD cache keeps \
                 temporary files"
            );
            return None;
        }
        let (namespace, state) = cache_namespace(&CacheNamespace {
            target,
            draft: draft_fingerprint
                .zip(draft)
                .map(|(fingerprint, draft)| DraftNamespace {
                    fingerprint,
                    geometry: draft.namespace_geometry(),
                    context_width: draft.context_width(),
                }),
            kv: (
                pool.block_size(),
                pool.num_layers() as u32,
                pool.config().num_kv_heads,
                pool.config().head_size,
                &cache_dtype,
            ),
            gdn: geometry.namespace_gdn(),
        });
        Some(PersistentIdentity {
            namespace,
            state,
            kv: target,
        })
    };
    let tier = open_disk_tier(options, identity, state_bytes)?;
    let block_bytes = mlx_paged_attn::block_object_bytes(pool);
    let context = state_context(&tier);
    let window_slots = match draft {
        Some(draft) => u32::try_from(draft.sliding_window).ok()?,
        None => 0,
    };
    let disk = StateDisk::new(
        tier,
        geometry,
        window,
        window_slots,
        geometry_text,
        block_bytes,
    )?;
    disk.wake_model_on_landing(model);
    Some((context, disk))
}

/// The adapter's view of a state tier: Splash keeps no per-turn KV on disk,
/// so only a state object backs a cold prefix.
pub(crate) fn state_context(tier: &DiskTier) -> ColdTierContext {
    ColdTierContext {
        manager: Arc::clone(&tier.manager),
        fingerprint: tier.kv_fingerprint,
        backing: ColdContextBacking::StateOnly,
    }
}

#[cfg(test)]
pub(crate) mod tests;
