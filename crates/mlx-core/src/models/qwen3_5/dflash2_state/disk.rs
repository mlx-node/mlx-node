//! A loaded model's DFlash2 states on its SSD tier (Splash `StateCache`
//! disk copies): Qwen3.5 adapter over the shared governor-driven
//! [`crate::disk_tier::state_disk::StateDiskEngine`].

use std::collections::{HashMap, HashSet};
use std::ops::{Deref, DerefMut};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[cfg(test)]
use mlx_paged_attn::Landed;
use mlx_paged_attn::{
    ColdCacheKey, ColdGroup, ColdStateLabel, ColdStateObject, EnqueueRefused, MakeRoom,
    state_object_bytes,
};

use crate::array::DType;
use crate::array::memory::WeightsResident;
use crate::cold_tier::{CheckpointFiles, CheckpointLoadGuard, ShardIdentity};
use crate::disk_tier::source_hold::open_disk_tier_with_source;
#[cfg(test)]
use crate::disk_tier::state_disk::{ChainCursor, chain_keys};
use crate::disk_tier::state_disk::{
    KvPoolHandle, PendingWrite, StateDiskCodec, StateDiskEngine, lineage_of, missing_keys,
};
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
use crate::transformer::paged_kv_cache_adapter::{ColdTierContext, PagedKVCacheAdapter};

use super::migrate::part_page_keys;
use super::paged::{DFlash2States, PagedAnchor};
#[cfg(test)]
use super::store::PART_PAGE_ROWS;
use super::store::{DiskCopy, Entry, EntryId, HybridStateStore, Lineage, StateClass};
use super::write_behind::WriteBehind;

#[cfg(test)]
pub(crate) use crate::disk_tier::state_disk::{ChainStep, Staged, adopt_states_on};
pub(crate) use crate::disk_tier::state_disk::{
    HeldTier, PersistStatus, ReadWait, Reclaimed, Tick, copy_demand_bytes, instant_at, label_of,
    lineage_keys, state_context,
};

pub(crate) struct Qwen35StateCodec {
    gdn: GdnSidecarGeometry,
    window_slots: u32,
}

impl StateDiskCodec for Qwen35StateCodec {
    type State = PagedAnchor;
    type EncodeCtx<'a> = &'a Qwen3_5Config;

    fn state_key(&self, tier: &DiskTier, entry: &Entry<PagedAnchor>) -> Option<ColdCacheKey> {
        let (whole, part) = part_page_keys(
            ColdGroup::DFlash2State,
            tier.state_fingerprint,
            lineage_of(entry),
            entry.block_size(),
        );
        part.or_else(|| whole.last().copied())
    }

    fn encode_into(
        &self,
        config: &Qwen3_5Config,
        spans: &[u64],
        entry: &Entry<PagedAnchor>,
        anchor: &PagedAnchor,
        into: &mut [u8],
    ) -> Option<u32> {
        let cell = (spans[0] as usize).min(into.len());
        let (cell_span, window_span) = into.split_at_mut(cell);
        let encoded = gdn_sidecar::encode_into(config, &self.gdn, &anchor.linear, cell_span)
            .unwrap_or(false)
            && match anchor.window.as_ref() {
                Some(window) => window.write_bytes(window_span).unwrap_or(false),
                None => window_span.is_empty(),
            };
        encoded.then(|| {
            anchor.window.as_ref().map_or_else(
                || u32::try_from(entry.depth()).unwrap_or(u32::MAX),
                DraftContextWindow::begin,
            )
        })
    }

    fn adopt_copy(&self, spans: &[u64], label: &ColdStateLabel) -> Option<(u64, u32)> {
        let cached_bytes = spans.iter().sum();
        let row = u32::try_from(label.tokens.len()).ok()?;
        Some((cached_bytes, row - row.min(self.window_slots)))
    }
}

/// The model's state tier: one per model, so one state write in flight.
pub(crate) struct StateDisk {
    pub(crate) engine: StateDiskEngine<Qwen35StateCodec>,
    /// The landing pages the governor last read from this tier.
    published_landing: u32,
    pub(crate) counters: DiskCounters,
    pub(crate) write_behind: WriteBehind,
}

impl Deref for StateDisk {
    type Target = StateDiskEngine<Qwen35StateCodec>;

    fn deref(&self) -> &Self::Target {
        &self.engine
    }
}

impl DerefMut for StateDisk {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.engine
    }
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
    pub(crate) fn writing(&self) -> bool {
        self.engine.writing()
    }

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
        let write_behind = WriteBehind::new(tier.persistent);
        let engine = StateDiskEngine::new(
            tier,
            spans,
            geometry,
            block_bytes,
            Qwen35StateCodec { gdn, window_slots },
        )?;
        Some(Self {
            engine,
            published_landing: 0,
            counters: DiskCounters::default(),
            write_behind,
        })
    }
}

/// Splash `StateCache::reclaimWrites`: a RAM copy without a disk copy, on a
/// tier that takes writes and holds it with its chain. The walk's listing
/// and [`reclaim`] both follow it (Splash `Cache::stateReclaim`).
pub(crate) fn reclaim_writes(states: &DFlash2States, id: EntryId) -> bool {
    states
        .disk
        .as_ref()
        .is_some_and(|disk| disk.engine.reclaim_writes(&states.store, id))
}

/// Splash `StateCache::reclaim`: a RAM copy whose own write is in flight
/// waits for it; one with a landed disk copy goes; a writable one is
/// written while the staging buffer is free, its RAM free at once (the
/// write reads its own copy), and waits while it is not; any other leaves
/// with no copy, as does one the tier cannot hold with its chain.
pub(crate) fn reclaim<T: DFlash2Target>(target: &mut T, id: EntryId) -> Reclaimed {
    let config = target.dense_config().into_owned();
    let DFlash2States { store, disk, .. } = target.dflash2_states_mut();
    let Some(disk) = disk.as_mut() else {
        let Some(entry) = store.get(id).filter(|entry| entry.ram().is_some()) else {
            return Reclaimed::Nothing;
        };
        if let Some(copy) = entry.disk() {
            if !copy.landed {
                return Reclaimed::Pending;
            }
            store.drop_ram(id);
            target.sync_gdn_state_marks();
            return Reclaimed::RamDropped;
        }
        store.remove(id);
        target.sync_gdn_state_marks();
        return Reclaimed::Erased;
    };
    let step = disk.engine.reclaim(store, id, &config);
    if step.staged {
        disk.counters.offloads += 1;
    }
    if step.resync {
        target.sync_gdn_state_marks();
    }
    step.outcome
}

/// Splash `StateCache::startWrite`: encodes `id`'s RAM copy into the
/// staging buffer and hands it to the tier's writer; its disk copy is in
/// flight from then on. Without `keep_ram` the RAM copy goes at once.
#[cfg(test)]
pub(crate) fn stage<T: DFlash2Target>(
    target: &mut T,
    id: EntryId,
    room: MakeRoom,
    keep_ram: bool,
) -> Staged {
    let config = target.dense_config().into_owned();
    let DFlash2States { store, disk, .. } = target.dflash2_states_mut();
    let Some(disk) = disk.as_mut() else {
        return Staged::Refused;
    };
    let staged = disk.engine.stage(store, id, room, keep_ram, &config);
    if staged == Staged::Started {
        disk.counters.offloads += 1;
        target.sync_gdn_state_marks();
    }
    staged
}

/// Splash `StateCache::pollOffload` and `Cache::pollTransfers`. True when
/// anything landed.
pub(crate) fn poll_offload<T: DFlash2Target>(target: &mut T) -> bool {
    let DFlash2States { store, disk, .. } = target.dflash2_states_mut();
    let Some(disk) = disk.as_mut() else {
        return false;
    };
    let out = disk.engine.poll_offload(store);
    disk.counters.offload_failures += out.offload_failures;
    if out.resync {
        target.sync_gdn_state_marks();
    }
    out.landed
}

/// Splash `Cache::persist` for the chain under a disk state, on the model
/// thread.
#[cfg(test)]
pub(crate) fn chain_step<T: DFlash2Target>(target: &mut T, deadline: Instant) -> ChainStep {
    let Some(kv) = target.paged_adapter_ref().map(KvPoolHandle::from_adapter) else {
        return ChainStep::default();
    };
    let DFlash2States { store, disk, .. } = target.dflash2_states_mut();
    let Some(disk) = disk.as_mut() else {
        return ChainStep::default();
    };
    let out = disk.engine.chain_step(store, &kv, deadline);
    disk.counters.invalidations += out.invalidations;
    disk.counters.offload_failures += out.offload_failures;
    if out.step.failed || out.step.landed_all {
        target.sync_gdn_state_marks();
    }
    out.step
}

/// One tick of the model's state tier: its landings, the owed chain as
/// `tick` allows, the blocks the writes in flight return published, and
/// the model's progress signaled when anything landed or a chain settled
/// (Splash `signalResourceProgress`). True on progress.
pub(crate) fn tick<T: DFlash2Target>(target: &mut T, tick: Tick) -> bool {
    if target.dflash2_states().disk.is_none() {
        return false;
    }
    let kv = target.paged_adapter_ref().map(KvPoolHandle::from_adapter);
    let out = {
        let DFlash2States { store, disk, .. } = target.dflash2_states_mut();
        let Some(disk) = disk.as_mut() else {
            return false;
        };
        let out = disk.engine.tick(store, kv.as_ref(), tick);
        disk.counters.invalidations += out.invalidations;
        disk.counters.offload_failures += out.offload_failures;
        out
    };
    if out.resync {
        target.sync_gdn_state_marks();
    }
    let mut progressed = out.progressed;
    if let Tick::Full { command_pages } = tick {
        let governor = target
            .memory_registration()
            .map_or_else(crate::engine::governor::governor, |guard| guard.governor());
        let now = governor.now();
        let page_bytes = target
            .paged_adapter_ref()
            .and_then(|adapter| adapter.bytes_per_block().ok())
            .unwrap_or(0);
        let demand = copy_demand_bytes(governor, now, command_pages, page_bytes);
        let read_waiting = target
            .dflash2_states()
            .disk
            .as_ref()
            .is_some_and(|disk| disk.engine.has_read_waiters());
        let claim_waiting = governor.ledger(|ledger| ledger.claim_waiting()) || read_waiting;
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
    let Some(disk) = states.disk.as_ref() else {
        return 0;
    };
    disk.engine
        .landing_pages(&states.store, target.paged_adapter_ref())
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
        .map_or(0, |disk| disk.engine.copying_ahead_bytes(&states.store))
}

/// Splash `Cache::persist` with `StateCache::persist`: a copy ahead of
/// `id`'s RAM copy and its chain.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn persist<T: DFlash2Target>(target: &mut T, id: EntryId) -> PersistStatus {
    persist_point(target, id, false)
}

/// [`persist`] with persistent prompt-head flag `head`.
pub(crate) fn persist_point<T: DFlash2Target>(
    target: &mut T,
    id: EntryId,
    head: bool,
) -> PersistStatus {
    let config = target.dense_config().into_owned();
    let kv = target.paged_adapter_ref().map(KvPoolHandle::from_adapter);
    let step = {
        let DFlash2States { store, disk, .. } = target.dflash2_states_mut();
        let Some(disk) = disk.as_mut() else {
            return PersistStatus::Refused;
        };
        let step = disk.engine.persist(store, kv.as_ref(), id, head, &config);
        if step.staged {
            disk.counters.offloads += 1;
        }
        disk.counters.invalidations += step.invalidations;
        disk.counters.offload_failures += step.offload_failures;
        step
    };
    if step.resync {
        target.sync_gdn_state_marks();
    }
    step.status
}

/// Splash `QwenCompositeState::canSnapshotToDisk`: the tier takes writes
/// and the staging buffer is free. No write budget holds a state back, in
/// use or not.
pub(crate) fn can_snapshot_to_disk(states: &DFlash2States) -> bool {
    states
        .disk
        .as_ref()
        .is_some_and(|disk| disk.engine.can_snapshot_to_disk())
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
    let (Some(layout), Some(mut staging)) = (disk.layout(row), disk.staging_for_write()) else {
        return Ok(None);
    };
    let cell = (disk.spans[0] as usize).min(staging.len());
    let (cell_span, window_span) = staging.split_at_mut(cell);
    let window_begin =
        if gdn_sidecar::encode_into(&config, &disk.engine.codec.gdn, &linear, cell_span)
            .unwrap_or(false)
        {
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

/// Gives back a buffer [`hold_staging`] took.
#[cfg(test)]
pub(crate) fn return_staging(states: &mut DFlash2States, staging: Vec<u8>) {
    if let Some(disk) = states.disk.as_mut() {
        disk.staging = Some(staging);
    }
}

/// The key of `id`'s state object on the tier: its disk copy's, else the
/// key a write gives it.
pub(crate) fn state_object_key(states: &DFlash2States, id: EntryId) -> Option<ColdCacheKey> {
    let disk = states.disk.as_ref()?;
    let entry = states.store.get(id)?;
    entry
        .disk()
        .map(|copy| copy.key)
        .or_else(|| disk.state_key(entry))
}

/// The staging buffer is free now, the writer's landings taken first; a
/// state read started now does not wait. True without a tier.
pub(crate) fn staging_free<T: DFlash2Target>(target: &mut T) -> bool {
    if target.dflash2_states().disk.is_none() {
        return true;
    }
    tick(target, Tick::Poll);
    target
        .dflash2_states()
        .disk
        .as_ref()
        .is_none_or(|disk| disk.staging_free())
}

pub(crate) fn read_wait(states: &DFlash2States) -> Option<ReadWait> {
    Some(states.disk.as_ref()?.engine.acquire_read_wait())
}

#[cfg(test)]
thread_local! {
    static STATE_READS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Test-only: the state reads this thread started.
#[cfg(test)]
pub(crate) fn state_reads() -> u64 {
    STATE_READS.with(std::cell::Cell::get)
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
    (gdn_sidecar::encode_into(config, &disk.engine.codec.gdn, &anchor.linear, cell).ok()?
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
    match disk.as_ref() {
        Some(disk) => disk.engine.erase(store, id),
        None => store.remove(id),
    }
}

/// Splash `StateCache::discardState` after a restore that cannot come
/// back: a state with a RAM copy loses only its disk copy, any other goes.
pub(crate) fn discard_state<T: DFlash2Target>(target: &mut T, id: EntryId) {
    let DFlash2States { store, disk, .. } = target.dflash2_states_mut();
    let Some(disk) = disk.as_mut() else {
        return;
    };
    if !disk.engine.discard_state(store, id) {
        return;
    }
    disk.counters.invalidations += 1;
    target.sync_gdn_state_marks();
}

/// A turn can resume on `id`: a RAM copy, or a disk copy whose object is on
/// the tier or on its way there and whose chain comes back whole, its
/// leading blocks resident and every block past them on the tier. The rows
/// of a part page are in no block: with either copy the state's fragment is
/// on the tier.
pub(crate) fn restorable<T: DFlash2Target>(target: &T, id: EntryId) -> bool {
    let states = target.dflash2_states();
    let Some(entry) = states.store.get(id) else {
        return false;
    };
    let on_part_page = !entry.depth().is_multiple_of(entry.block_size() as usize);
    if entry.ram().is_some() && !on_part_page {
        return true;
    }
    let (Some(disk), Some(adapter)) = (states.disk.as_ref(), target.paged_adapter_ref()) else {
        return false;
    };
    let manager = &disk.tier.manager;
    let (chain, fragment) = part_page_keys(
        ColdGroup::Kv,
        disk.tier.kv_fingerprint,
        lineage_of(entry),
        entry.block_size(),
    );
    if fragment.is_some_and(|key| !manager.contains_in(&key, ColdGroup::Kv)) {
        return false;
    }
    if entry.ram().is_some() {
        return true;
    }
    let Some(copy) = entry.disk() else {
        return false;
    };
    if copy.landed && !manager.contains_in(&copy.key, ColdGroup::DFlash2State) {
        return false;
    }
    let resident = entry
        .chain()
        .iter()
        .take_while(|&&hash| adapter.prefix_cached(hash))
        .count();
    chain.get(resident..).is_some_and(|keys| {
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
        if disk.staging_free() {
            return true;
        }
        if !disk.writing() || crate::engine::governor::governor::cancelled() {
            return false;
        }
        if !disk.engine.await_landing(deadline) {
            return false;
        }
        tick(target, Tick::Poll);
    }
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
    #[cfg(test)]
    STATE_READS.with(|reads| reads.set(reads.get() + 1));
    let deadline = Instant::now()
        + Duration::from_secs_f64(crate::engine::governor::wait::RESOURCE_WAIT_SECONDS);
    if !wait_for_staging(target, deadline) {
        return None;
    }
    let config = target.dense_config().into_owned();
    let DFlash2States { store, disk, .. } = target.dflash2_states_mut();
    let disk = disk.as_mut()?;
    disk.engine
        .read_into_staging(store, id, |codec, spans, entry, copy, staging| {
            let depth = u32::try_from(entry.depth()).ok()?;
            let (cell, rest) = staging.split_at(usize::try_from(spans[0]).ok()?);
            let linear = gdn_sidecar::decode_caches(&config, &codec.gdn, cell).ok()??;
            if let Some(window) = window
                && !window.read_bytes(rest, copy.window_begin, depth).ok()?
            {
                return None;
            }
            Some(linear)
        })
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

/// Splash `Cache::adopt` at a persistent tier's load.
pub(crate) fn adopt_all<T: DFlash2Target>(target: &mut T) -> u64 {
    // A fragment is 16 rows of a page with token-major keys: only a pool of
    // that page takes one.
    let part_pages = target
        .paged_adapter_ref()
        .is_some_and(|adapter| adapter.layer_kv_pool().layout().keys_token_major);
    let DFlash2States {
        store,
        disk,
        prompt_heads,
        ..
    } = target.dflash2_states_mut();
    let Some(disk) = disk.as_mut() else {
        return 0;
    };
    let adopted = disk.engine.adopt_all(store, part_pages);
    if adopted > 0 {
        for entry in store.entries() {
            if !entry.checkpoint() {
                prompt_heads.insert(entry.id());
            }
        }
    }
    disk.counters.adopted += adopted;
    adopted
}

/// Concrete payload stored inside the type-erased [`HeldTier`] while a load's
/// warm-up runs.
struct HeldDFlash2Tier {
    store: HybridStateStore<PagedAnchor>,
    prompt_heads: HashSet<EntryId>,
    disk: StateDisk,
    cold: Option<ColdTierContext>,
}

/// Sets the tier, its states and the adapter's view of it aside behind an
/// empty store: what a warm-up's cache reset clears is not theirs, and no
/// scratch turn can stage, write, adopt or evict one. `None` without a tier.
pub(crate) fn hold_tier<T: DFlash2Target>(target: &mut T) -> Option<HeldTier> {
    let states = target.dflash2_states_mut();
    let disk = states.disk.take()?;
    let empty = HybridStateStore::new(states.store.block_size());
    let store = std::mem::replace(&mut states.store, empty);
    let prompt_heads = std::mem::take(&mut states.prompt_heads);
    let cold = target
        .paged_adapter_mut_ref()
        .and_then(PagedKVCacheAdapter::take_cold_tier);
    target.sync_gdn_state_marks();
    Some(HeldTier::new(HeldDFlash2Tier {
        store,
        prompt_heads,
        disk,
        cold,
    }))
}

/// Puts a [`hold_tier`] back; the store that stood in for it goes.
pub(crate) fn release_tier<T: DFlash2Target>(target: &mut T, held: HeldTier) {
    let Some(HeldDFlash2Tier {
        store,
        prompt_heads,
        disk,
        cold,
    }) = held.downcast::<HeldDFlash2Tier>()
    else {
        return;
    };
    let states = target.dflash2_states_mut();
    states.store = store;
    states.prompt_heads = prompt_heads;
    states.disk = Some(disk);
    if let (Some(cold), Some(adapter)) = (cold, target.paged_adapter_mut_ref()) {
        adapter.set_cold_tier(cold);
    }
    target.sync_gdn_state_marks();
}

/// Splash `closePersistentCache` with `FdTransport::runFlush` at a clean
/// stop.
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
            if !disk.engine.await_landing(deadline) {
                break false;
            }
        };
    }
    let closed = tier.close(deadline);
    poll_offload(target);
    if !tier.persistent {
        let DFlash2States { store, disk, .. } = target.dflash2_states_mut();
        if let Some(disk) = disk.as_mut() {
            disk.engine.strip_disk_copies(store);
        }
        target.sync_gdn_state_marks();
    }
    flushed && closed
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
    let cache_dtype = load.adapter.cache_identity();
    let identity = || tier_identity(load, pool.block_size(), &cache_dtype);
    let moves = pool.layout().keys_token_major;
    let tier = if moves {
        open_disk_tier_with_source(
            options,
            identity,
            || super::migrate::source_identity(load),
            state_bytes,
        )?
    } else {
        open_disk_tier(options, identity, state_bytes)?
    };
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
    if moves {
        move_held_source(&disk, pool);
    }
    disk.wake_model_on_landing(model);
    Some((context, disk))
}

/// Moves the source store `disk`'s tier holds into it (the move of
/// [`super::migrate::move_source`]) and prints what it did; nothing when the
/// tier holds none.
fn move_held_source(disk: &StateDisk, pool: &mlx_paged_attn::LayerKVPool) {
    let Some(source_bytes) = disk.tier.with_source(|hold| hold.bytes) else {
        return;
    };
    let config = pool.config();
    let Some(source_layout) =
        super::migrate::source_layout(config.num_layers, config.num_kv_heads, config.head_size)
    else {
        return;
    };
    let started = Instant::now();
    let moved = super::migrate::move_source::move_source(
        &disk.tier,
        &super::migrate::move_source::MoveShape {
            spans: &disk.spans,
            geometry: &disk.geometry,
            source_layout,
        },
    );
    crate::utils::gguf_planes::load_line(format!(
        "Qwen3.5 state store move: source_bytes={} states={} pages={} fragments={} refused={} \
         unreadable={} dropped={} held_back={} stopped={} source_removed={} wall_ms={}",
        source_bytes,
        moved.states,
        moved.pages,
        moved.fragments,
        moved.refused,
        moved.unreadable,
        moved.dropped,
        moved.held_back,
        moved.stopped,
        moved.source_removed,
        started.elapsed().as_millis(),
    ));
}

/// The persistent identity [`open_state_disk`] keys `load`'s tier by when
/// its pool has `block_size` pages and the cache identity text
/// `cache_identity`: the same namespace, state and KV fingerprints, from the
/// same files, config bytes and geometry.
pub(crate) fn tier_identity(
    load: &TierLoad<'_>,
    block_size: u32,
    cache_identity: &str,
) -> Option<PersistentIdentity> {
    let geometry = gdn_sidecar::geometry(load.dense, load.activation)?;
    let draft = load.draft.as_ref().map(|draft| &draft.model.config);
    let pool = load.adapter.layer_kv_pool();
    let target = crate::cold_tier::build_model_fingerprint(
        load.family,
        load.files,
        Some(&load.config_json),
        &crate::cold_tier::ColdTierGeometry {
            block_size: u64::from(block_size),
            num_layers: pool.num_layers() as u64,
            num_kv_heads: u64::from(pool.config().num_kv_heads),
            head_size: u64::from(pool.config().head_size),
            cache_dtype: cache_identity.to_string(),
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
            block_size,
            pool.num_layers() as u32,
            pool.config().num_kv_heads,
            pool.config().head_size,
            cache_identity,
        ),
        gdn: geometry.namespace_gdn(),
    });
    Some(PersistentIdentity {
        namespace,
        state,
        kv: target,
    })
}

#[cfg(all(test, target_os = "macos"))]
mod part_page_tests;
#[cfg(test)]
pub(crate) mod tests;
