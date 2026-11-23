//! Shared Parker-thread prefix disk tier (`PrefixDisk<C>`, `Parker`, `Slot`,
//! `Wake`, `Closer`, `DiskTelemetry`, `Blocks`, `PoolBlocks`) for Pattern B
//! families (`qwen4_exp` and `hy_v4_flash`).
//!
//! ```text
//! model thread ─► host copies of blocks and segments ─► Slot.items (≤ QUEUE_BYTES)
//! parker       ─► offers them to the writer one at a time, measures the cache root
//!                 after a state write, drains once per plan, wakes the model thread
//!                 when the plan is due
//! closer       ─► any thread: the model thread writes the waiting states, the parker
//!                 hands over the copies, the tier drains and closes
//! ```
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, Weak};
use std::time::{Duration, Instant};

use mlx_paged_attn::{
    ColdCacheBlock, ColdCacheFingerprint, ColdCacheKey, ColdCacheLayout, ColdCacheManager,
    ColdGroup, ColdLayerBlock, ColdObject, ColdSidecar, ColdStateLayout, ColdStateObject,
    EnqueueRefused, Landed, MakeRoom, state_object_bytes,
};
use napi::{Error, Result};

use crate::disk_tier::binary_codec::{
    draft_segment_key, segment_sidecar_fits, segment_sidecar_layout,
};
use crate::disk_tier::state_disk::{adopt_states, label_of, lineage_keys};
use crate::disk_tier::store::{DiskCopy, EntryId, HybridStateStore, Lineage, StateBytes};
use crate::disk_tier::{DiskTier, TierHandle};
use crate::transformer::paged_kv_cache_adapter::{
    ColdContextBacking, ColdTierContext, PagedKVCacheAdapter, SeqId,
};

/// Splash `WriteBehind` hands a point over this long after publication.
pub const DELAY: Duration = Duration::from_secs(10);
/// Splash `WriteBehind::kMinimumTokens`: a shorter lineage prefills again
/// faster than its state is written and read back.
pub const MINIMUM_ROWS: usize = 2048;
/// A temporary tier copies while memory dropped a state this recently
/// (Splash `kCopyDemandSeconds`).
pub const DEMAND_WINDOW: Duration = Duration::from_secs(60);
/// A written state whose chain nothing can add to any more is dropped this
/// long after.
pub const GIVE_UP: Duration = Duration::from_secs(10);
/// The wait for a writer slot of one object on the parker thread (an object
/// that finds none stays first in line), for the staging buffer before a
/// read, and what a dispose gives the waiting states.
pub const SLOT_WAIT: Duration = Duration::from_secs(2);
/// The model thread's disk pass copies blocks out at most this often: a copy
/// waits for the GPU.
pub const STEP_TOP_UP: Duration = Duration::from_millis(250);
/// Host copies of blocks and segments waiting for the writer.
pub const QUEUE_BYTES: u64 = 128 << 20;

/// Counters of a prefix disk tier.
#[derive(Default)]
pub struct DiskTelemetry {
    /// 0 off, 1 persistent, 2 temporary.
    pub disk_enabled: AtomicU64,
    /// States waiting for their write, the one in flight included.
    pub waiting: AtomicU64,
    /// States written whose whole chain is on disk.
    pub durable: AtomicU64,
    /// Waiting states not written: their conversation went on, or they are
    /// gone.
    pub unneeded: AtomicU64,
    /// States not written or dropped (chain incomplete, room, encoding, a
    /// failed write, a closed tier), and copies the writer refused.
    pub refused: AtomicU64,
    pub disk_restored: AtomicU64,
    /// States a persistent tier took back at load, on disk only.
    pub adopted: AtomicU64,
    /// States on disk removed because their chain was not.
    pub dropped_chain: AtomicU64,
    /// Blocks and segments copied out and not handed to the writer yet.
    pub owed: AtomicU64,
    pub state_bytes_written: AtomicU64,
}

impl DiskTelemetry {
    pub fn bump(counter: &AtomicU64) {
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// The tier as a turn's row reports it: 0 off, 1 persistent, 2
    /// temporary; off once `tier` takes no write any more.
    pub(crate) fn mode(&self, tier: &TierHandle) -> u64 {
        match self.disk_enabled.load(Ordering::Relaxed) {
            mode if tier.is_open() => mode,
            _ => 0,
        }
    }
}

pub fn object_bytes(object: &ColdObject) -> u64 {
    match object {
        ColdObject::Block(block) => block
            .layers
            .iter()
            .map(|layer| (layer.keys.len() + layer.values.len()) as u64)
            .sum(),
        ColdObject::Sidecar(sidecar) => sidecar.tensors.iter().map(|t| t.len() as u64).sum(),
    }
}

/// What the model thread last left for the disk to do.
#[derive(Clone, Copy, Default)]
pub struct Plan {
    /// Bumped by every pass that stores a plan.
    pub epoch: u64,
    /// When the next waiting point or stuck chain is due.
    pub next_due: Option<Instant>,
    /// A chain is owed: blocks to copy, or a written state whose chain is
    /// not whole yet.
    pub owing: bool,
    /// The model thread's next pass that may copy.
    pub next_step: Option<Instant>,
}

#[derive(Default)]
pub struct Slot {
    /// Block and segment copies, first in line first.
    pub items: VecDeque<ColdObject>,
    pub item_bytes: u64,
    flush: u64,
    flushed: u64,
    flush_until: Option<Instant>,
    /// The last flush for which the thread waited out the copies it handed.
    drained: u64,
    pub stop: bool,
    pub plan: Plan,
    /// A close asks the model thread to write the waiting states by then.
    pub closing: Option<Instant>,
    /// The model thread's flush of the waiting states ended: true when no
    /// state waited any more and every copy and write it handed landed.
    pub points_flushed: Option<bool>,
    /// A state write went out: the parker measures the cache root.
    pub measure: bool,
    #[cfg(test)]
    pub measured: u64,
    /// The plan epoch whose wake went out.
    woke: u64,
    /// The plan epoch for which the writer was drained.
    drained_epoch: u64,
    #[cfg(test)]
    pub hold: bool,
    #[cfg(test)]
    pub slot_wait: Option<Duration>,
}

impl Slot {
    fn slot_wait(&self) -> Duration {
        #[cfg(test)]
        if let Some(wait) = self.slot_wait {
            return wait;
        }
        SLOT_WAIT
    }
}

pub struct Parker {
    slot: Mutex<Slot>,
    pub wake: Condvar,
    pub done: Condvar,
}

impl Parker {
    pub fn new() -> Self {
        Self {
            slot: Mutex::new(Slot::default()),
            wake: Condvar::new(),
            done: Condvar::new(),
        }
    }

    pub fn lock(&self) -> MutexGuard<'_, Slot> {
        self.slot
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    /// Hands every queued copy to the writer and waits until the writer is
    /// past them, until `deadline`. False when a copy was not handed over by
    /// then or the thread stopped first.
    pub fn flush(&self, deadline: Instant) -> bool {
        let mut slot = self.lock();
        slot.flush += 1;
        let target = slot.flush;
        slot.flush_until = Some(deadline);
        self.wake.notify_all();
        while slot.flushed < target && !slot.stop {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                break;
            }
            slot = self
                .done
                .wait_timeout(slot, left)
                .map(|(guard, _)| guard)
                .unwrap_or_else(|poison| poison.into_inner().0);
        }
        slot.flushed >= target && slot.items.is_empty()
    }
}

impl Default for Parker {
    fn default() -> Self {
        Self::new()
    }
}

/// Hands the queued copies to the writer one at a time, and measures the
/// cache root after a state write went out (that can remove another cache's
/// directory: never on the model thread). With no copy left and a chain
/// owed it drains the writer once per plan, then wakes the model thread once
/// per plan, at the earliest when the plan is due and the model thread's
/// next pass may copy.
pub(crate) fn run(
    parker: Arc<Parker>,
    tier: Arc<DiskTier>,
    telemetry: Arc<DiskTelemetry>,
    wake: Arc<Wake>,
) {
    let manager = Arc::clone(&tier.manager);
    let mut slot = parker.lock();
    loop {
        if slot.stop {
            break;
        }
        if std::mem::take(&mut slot.measure) {
            drop(slot);
            tier.measure();
            slot = parker.lock();
            #[cfg(test)]
            {
                slot.measured += 1;
            }
            continue;
        }
        let now = Instant::now();
        let flushing = slot.flush > slot.flushed;
        let until = slot.flush_until.filter(|_| flushing);
        if until.is_some_and(|until| now >= until) {
            slot.flushed = slot.flush;
            parker.done.notify_all();
            continue;
        }
        #[cfg(test)]
        let held = slot.hold;
        #[cfg(not(test))]
        let held = false;
        if !held && let Some(object) = slot.items.pop_front() {
            let bytes = object_bytes(&object);
            let wait = now + slot.slot_wait();
            drop(slot);
            let deadline = until.map_or(wait, |until| until.min(wait));
            let sent = manager.offer(object, deadline);
            slot = parker.lock();
            match sent {
                // No writer slot freed: it stays first in line, so the chain
                // keeps no hole.
                Ok(Some(object)) => slot.items.push_front(object),
                Ok(None) => {
                    slot.item_bytes = slot.item_bytes.saturating_sub(bytes);
                    telemetry.owed.fetch_sub(1, Ordering::Relaxed);
                }
                Err(_) => {
                    slot.item_bytes = slot.item_bytes.saturating_sub(bytes);
                    telemetry.owed.fetch_sub(1, Ordering::Relaxed);
                    DiskTelemetry::bump(&telemetry.refused);
                }
            }
            continue;
        }
        if flushing && slot.items.is_empty() {
            if slot.drained < slot.flush {
                slot.drained = slot.flush;
                drop(slot);
                if let Some(until) = until {
                    manager.drain(until.saturating_duration_since(Instant::now()));
                }
                slot = parker.lock();
                continue;
            }
            slot.flushed = slot.flush;
            parker.done.notify_all();
        }
        let plan = slot.plan;
        if slot.items.is_empty() && plan.owing && slot.drained_epoch != plan.epoch {
            slot.drained_epoch = plan.epoch;
            drop(slot);
            manager.drain(SLOT_WAIT);
            slot = parker.lock();
            continue;
        }
        let mut wait = Duration::from_secs(3600);
        let want = if plan.owing { Some(now) } else { plan.next_due };
        if let Some(want) = want
            && slot.woke != plan.epoch
        {
            let at = plan.next_step.map_or(want, |step| step.max(want));
            if now >= at {
                slot.woke = plan.epoch;
                drop(slot);
                wake.fire();
                slot = parker.lock();
                continue;
            }
            wait = wait.min(at - now);
        }
        if let Some(until) = until {
            wait = wait.min(until.saturating_duration_since(now));
        }
        slot = parker
            .wake
            .wait_timeout(slot, wait.max(Duration::from_millis(1)))
            .map(|(guard, _)| guard)
            .unwrap_or_else(|poison| poison.into_inner().0);
    }
}

type WakeCall = Arc<dyn Fn() + Send + Sync>;

/// The call that wakes the model thread for its disk work. A load installs
/// a weak command sender: it holds neither the closer nor the tier.
#[derive(Default)]
pub struct Wake {
    call: Mutex<Option<WakeCall>>,
}

impl Wake {
    fn slot(&self) -> MutexGuard<'_, Option<WakeCall>> {
        self.call
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    fn is_set(&self) -> bool {
        self.slot().is_some()
    }

    pub fn fire(&self) {
        let call = self.slot().clone();
        if let Some(call) = call {
            call();
        }
    }
}

/// Closes the tier from any thread: the model thread writes the waiting
/// states, the parker hands over the queued copies, then the tier drains
/// its writer and takes no write after.
pub struct Closer {
    pub parker: Arc<Parker>,
    pub(crate) tier: Arc<DiskTier>,
    pub telemetry: Arc<DiskTelemetry>,
    pub wake: Arc<Wake>,
    /// A close waits for the model thread's flush of the waiting states.
    pub closing: AtomicBool,
    /// The one close's outcome once it ended; `Some(None)` while it runs.
    outcome: Mutex<Option<Option<bool>>>,
    ended: Condvar,
}

impl Closer {
    pub(crate) fn new(
        parker: Arc<Parker>,
        tier: Arc<DiskTier>,
        telemetry: Arc<DiskTelemetry>,
        wake: Arc<Wake>,
    ) -> Self {
        Self {
            parker,
            tier,
            telemetry,
            wake,
            closing: AtomicBool::new(false),
            outcome: Mutex::new(None),
            ended: Condvar::new(),
        }
    }

    /// The call that wakes the model thread (a weak command sender).
    pub fn set_wake(&self, call: WakeCall) {
        *self.wake.slot() = Some(call);
    }

    /// A persistent tier: the model thread writes the waiting states newest
    /// first, given all of the time but the last second (at most half of
    /// it), then every queued copy goes to the writer and the tier closes,
    /// by `deadline`; false when the drain missed it. A temporary tier
    /// keeps nothing: its writer stops and its directory goes. One close
    /// runs: a call while it runs waits for its outcome until its own
    /// deadline (false when it has none by then), and a later call gets the
    /// same outcome.
    pub fn close(&self, deadline: Instant) -> bool {
        self.once(deadline, true)
    }

    /// [`Self::close`] without the model thread's flush: the model thread
    /// calls it itself.
    pub fn finish(&self, deadline: Instant) -> bool {
        self.once(deadline, false)
    }

    fn once(&self, deadline: Instant, ask_model: bool) -> bool {
        {
            let mut outcome = self
                .outcome
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            if outcome.is_some() {
                loop {
                    if let Some(Some(landed)) = *outcome {
                        return landed;
                    }
                    let left = deadline.saturating_duration_since(Instant::now());
                    if left.is_zero() {
                        return false;
                    }
                    outcome = self
                        .ended
                        .wait_timeout(outcome, left)
                        .map(|(guard, _)| guard)
                        .unwrap_or_else(|poison| poison.into_inner().0);
                }
            }
            *outcome = Some(None);
        }
        let mut flushed = true;
        if self.tier.persistent && self.tier.is_open() {
            if ask_model && self.wake.is_set() {
                flushed = self.flush_on_model_thread(deadline);
            } else if !ask_model {
                flushed = self.parker.lock().points_flushed != Some(false);
            }
        }
        if self.tier.persistent {
            flushed &= self.parker.flush(deadline);
        }
        let landed = self.tier.close(deadline) && flushed;
        self.closing.store(false, Ordering::Release);
        self.telemetry.disk_enabled.store(0, Ordering::Relaxed);
        *self
            .outcome
            .lock()
            .unwrap_or_else(|poison| poison.into_inner()) = Some(Some(landed));
        self.ended.notify_all();
        landed
    }

    /// Wakes the model thread to write its waiting states and waits for it
    /// until the last second of `deadline` (at most half of what is left).
    /// True when its flush left no state waiting and all it handed landed.
    fn flush_on_model_thread(&self, deadline: Instant) -> bool {
        let left = deadline.saturating_duration_since(Instant::now());
        let until = deadline
            .checked_sub((left / 2).min(Duration::from_secs(1)))
            .unwrap_or(deadline);
        {
            let mut slot = self.parker.lock();
            if slot.stop {
                return false;
            }
            slot.closing = Some(until);
            slot.points_flushed = None;
        }
        self.closing.store(true, Ordering::Release);
        self.wake.fire();
        let mut slot = self.parker.lock();
        while slot.points_flushed.is_none() && !slot.stop {
            let left = until.saturating_duration_since(Instant::now());
            if left.is_zero() {
                break;
            }
            slot = self
                .parker
                .done
                .wait_timeout(slot, left)
                .map(|(guard, _)| guard)
                .unwrap_or_else(|poison| poison.into_inner().0);
        }
        slot.points_flushed == Some(true)
    }
}

// ---------------------------------------------------------------------------
// Block sources.
// ---------------------------------------------------------------------------

/// Where a lineage's block bytes come from.
pub trait Blocks {
    /// Settles the writes before the first read of a pass; false when the
    /// source cannot be read now.
    fn begin(&mut self, lineage: &[u32]) -> bool;
    /// Block `i` (hot hash `hot`) of the lineage of the last `begin`: every
    /// pool layer's blobs; `None` when the source no longer holds it.
    fn block(&mut self, i: usize, hot: u64) -> Option<Vec<ColdLayerBlock>>;
    /// Optional block layout derived directly from the source (e.g. pool
    /// metadata) when the codec does not fix one up front.
    fn layout(&self, _layers: &[ColdLayerBlock]) -> Option<ColdCacheLayout> {
        None
    }
}

/// The paged KV pool's cached blocks.
pub struct PoolBlocks<'a>(pub &'a mut PagedKVCacheAdapter);

impl Blocks for PoolBlocks<'_> {
    fn begin(&mut self, _: &[u32]) -> bool {
        if self.0.eval_pending_pool_writes().is_err() {
            return false;
        }
        crate::array::memory::synchronize();
        true
    }

    fn block(&mut self, _: usize, hot: u64) -> Option<Vec<ColdLayerBlock>> {
        let block = self.0.cached_block(hot)?;
        let layers = self
            .0
            .layer_kv_pool()
            .read_block_all_layers(block.block_id)
            .ok()?;
        Some(
            layers
                .into_iter()
                .map(|(keys, values)| ColdLayerBlock { keys, values })
                .collect(),
        )
    }

    fn layout(&self, layers: &[ColdLayerBlock]) -> Option<ColdCacheLayout> {
        let first = layers.first()?;
        let pool = self.0.layer_kv_pool();
        Some(ColdCacheLayout {
            block_size: pool.block_size(),
            num_layers: pool.num_layers() as u32,
            num_kv_heads: pool.config().num_kv_heads,
            head_size: pool.config().head_size,
            cache_dtype: format!("{:?}", pool.cache_dtype()),
            key_bytes_per_layer: first.keys.len(),
            value_bytes_per_layer: first.values.len(),
        })
    }
}

// ---------------------------------------------------------------------------
// PrefixDisk<C> codec and engine.
// ---------------------------------------------------------------------------

/// How [`PrefixDisk::flush_points`] drains queued cursor blocks and judges
/// completion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FlushPolicy {
    /// Flush waiting points and hand/drain whatever is queued (`qwen4_exp`).
    PointsOnly,
    /// Also loop `top_up_all` until all cursor blocks/segments past one queue
    /// window are copied and require zero owed chains/items/new refusals
    /// (`hy_v4_flash`).
    DrainCursorsStrict,
}

/// What [`PrefixDisk::read_state`] gave back.
pub(crate) enum DiskRead<S> {
    State(S),
    /// The staging buffer did not come back in time: the object stays.
    Busy,
    /// It did not read back: the object is gone.
    Failed,
}

/// Per-model codec and policy for [`PrefixDisk`].
pub(crate) trait PrefixCodec {
    type StoreState: StateBytes;
    type DecodedState;
    type Segment: Clone;

    const BLOCK_ROWS: usize;
    const THREAD_NAME: &'static str;
    const LOG_PREFIX: &'static str;
    /// Preserve copied block/segment progress when `start` extends an existing
    /// cursor's lineage.
    const EXTEND_CURSOR: bool;
    const FLUSH_POLICY: FlushPolicy;
    /// Whether `Written::OnTier` counts `owed.bytes` toward
    /// `state_bytes_written` once its chain lands.
    const COUNT_ON_TIER_BYTES: bool;

    fn spans(&self) -> Vec<u64>;
    fn geometry(&self) -> String;
    fn block_layout(&self) -> Option<ColdCacheLayout> {
        None
    }
    fn has_draft(&self) -> bool;
    fn segment_stride(&self) -> usize;
    fn segment_format(&self) -> u32;

    fn state_matches_lane(&self, state: &Self::StoreState, row: usize) -> bool;
    fn state_segment_count(&self, state: &Self::StoreState) -> usize;
    fn expected_segment_count(&self, row: usize, stride: usize) -> usize;

    fn encode_state_into(&self, state: &Self::StoreState, into: &mut [u8]) -> Result<bool>;
    fn decode_state(
        &self,
        bytes: &[u8],
        row: usize,
        segments: Vec<Self::Segment>,
    ) -> Option<Self::DecodedState>;

    fn encode_segment(&self, stride: usize, segment: &Self::Segment) -> Result<Option<Vec<u8>>>;
    fn decode_segment(&self, bytes: &[u8]) -> Option<Self::Segment>;

    fn is_superseded(&self, store: &HybridStateStore<Self::StoreState>, id: EntryId) -> bool {
        store.superseded(id)
    }
}

/// What one turn of a sequence owes the disk: its lineage up to its last
/// point, keyed as the prefix cache keys it, and how far its blocks and
/// draft segments are copied out (or found on disk).
struct Cursor<Seg> {
    lineage: Vec<u32>,
    hot: Vec<u64>,
    cold: Vec<ColdCacheKey>,
    draft_chain: Vec<ColdCacheKey>,
    registered: usize,
    pending: Vec<Seg>,
    segment_stride: usize,
    blocks: usize,
    segments: usize,
    broken: bool,
    closed: bool,
}

/// The tier's one-slot state lane: the record shape and the one staging
/// buffer every state write and read goes through.
struct Lane {
    spans: Vec<u64>,
    geometry: String,
    staging: Option<Vec<u8>>,
    landed_tx: Sender<Landed>,
    landed_rx: Receiver<Landed>,
    pending: Option<(EntryId, ColdCacheKey)>,
}

impl Lane {
    fn new(spans: Vec<u64>, geometry: String) -> Option<Self> {
        let total = spans.iter().sum::<u64>();
        let bytes = usize::try_from(total).ok().filter(|&bytes| bytes > 0)?;
        let (landed_tx, landed_rx) = std::sync::mpsc::channel();
        Some(Self {
            spans,
            geometry,
            staging: Some(vec![0; bytes]),
            landed_tx,
            landed_rx,
            pending: None,
        })
    }

    fn layout(&self, row: usize) -> Option<ColdStateLayout> {
        Some(ColdStateLayout {
            boundary_tokens: u32::try_from(row).ok()?,
            spans: self.spans.clone(),
            geometry: self.geometry.clone(),
        })
    }
}

#[derive(Clone, Copy)]
struct Point {
    id: EntryId,
    due: Instant,
}

struct Owed {
    id: EntryId,
    key: ColdCacheKey,
    needs: Vec<(ColdCacheKey, ColdGroup)>,
    landed: usize,
    bytes: u64,
    stuck: Option<Instant>,
}

enum Written {
    Started,
    OnTier,
    Busy,
    Refused,
}

/// Shared Parker-thread prefix disk tier parameterized by [`PrefixCodec`].
pub(crate) struct PrefixDisk<C: PrefixCodec> {
    pub(crate) manager: Arc<ColdCacheManager>,
    pub(crate) closer: Arc<Closer>,
    pub(crate) fingerprint: ColdCacheFingerprint,
    pub(crate) codec: C,
    pub delay: Duration,
    /// A lineage shorter than this writes nothing.
    pub minimum: usize,
    /// A written state whose chain can no longer grow is dropped this long
    /// after.
    pub give_up: Duration,
    persistent: bool,
    demand_until: Option<Instant>,
    seen_drops: u64,
    parker: Arc<Parker>,
    thread: Option<std::thread::JoinHandle<()>>,
    cursors: HashMap<SeqId, Cursor<C::Segment>>,
    pub(crate) telemetry: Arc<DiskTelemetry>,
    skip_once: bool,
    step_after: Instant,
    lane: Lane,
    points: VecDeque<Point>,
    owed: Vec<Owed>,
    closing_by: Option<Instant>,
    #[cfg(test)]
    pub encoded: usize,
    #[cfg(test)]
    pub step_runs: usize,
    #[cfg(test)]
    pub landed_states: Vec<ColdCacheKey>,
}

impl<C: PrefixCodec> PrefixDisk<C> {
    pub(crate) fn open(
        tier: Arc<DiskTier>,
        telemetry: Arc<DiskTelemetry>,
        codec: C,
    ) -> Result<Self> {
        let lane = Lane::new(codec.spans(), codec.geometry()).ok_or_else(|| {
            Error::from_reason(format!("{}: no state record shape", C::LOG_PREFIX))
        })?;
        let manager = Arc::clone(&tier.manager);
        let fingerprint = tier.state_fingerprint;
        let persistent = tier.persistent;
        let trimmed = manager.fit_quota();
        if trimmed > 0 {
            tracing::info!(
                "{}: took {trimmed} stored objects off the SSD cache to fit its limit",
                C::LOG_PREFIX
            );
        }
        let parker = Arc::new(Parker::new());
        let wake = Arc::new(Wake::default());
        let thread = {
            let (parker, tier, telemetry, wake) = (
                Arc::clone(&parker),
                Arc::clone(&tier),
                Arc::clone(&telemetry),
                Arc::clone(&wake),
            );
            std::thread::Builder::new()
                .name(C::THREAD_NAME.to_string())
                .spawn(move || run(parker, tier, telemetry, wake))
                .map_err(|e| Error::from_reason(format!("{}: {e}", C::LOG_PREFIX)))?
        };
        telemetry
            .disk_enabled
            .store(if persistent { 1 } else { 2 }, Ordering::Relaxed);
        {
            let wake = Arc::clone(&wake);
            manager.set_landing_wake(Some(Arc::new(move || wake.fire())));
        }
        let closer = Arc::new(Closer::new(
            Arc::clone(&parker),
            tier,
            Arc::clone(&telemetry),
            wake,
        ));
        Ok(Self {
            manager,
            closer,
            fingerprint,
            codec,
            delay: DELAY,
            minimum: MINIMUM_ROWS,
            give_up: GIVE_UP,
            persistent,
            demand_until: None,
            seen_drops: 0,
            parker,
            thread: Some(thread),
            cursors: HashMap::new(),
            telemetry,
            skip_once: false,
            step_after: Instant::now(),
            lane,
            points: VecDeque::new(),
            owed: Vec::new(),
            closing_by: None,
            #[cfg(test)]
            encoded: 0,
            #[cfg(test)]
            step_runs: 0,
            #[cfg(test)]
            landed_states: Vec::new(),
        })
    }

    pub(crate) fn attach_kv_only(&self, adapter: &mut PagedKVCacheAdapter) {
        adapter.set_cold_tier(ColdTierContext {
            manager: Arc::clone(&self.manager),
            fingerprint: self.fingerprint,
            backing: ColdContextBacking::KvOnly,
        });
    }

    pub(crate) fn closer(&self) -> Weak<Closer> {
        Arc::downgrade(&self.closer)
    }

    pub(crate) fn tier_handle(&self) -> TierHandle {
        TierHandle::of(Some(&self.closer.tier))
    }

    #[cfg(test)]
    pub(crate) fn tier(&self) -> &Arc<DiskTier> {
        &self.closer.tier
    }

    #[cfg(test)]
    pub(crate) fn hold(&self, hold: bool) {
        self.parker.lock().hold = hold;
        self.parker.wake.notify_all();
    }

    #[cfg(test)]
    pub(crate) fn set_slot_wait(&self, wait: Duration) {
        self.parker.lock().slot_wait = Some(wait);
    }

    #[cfg(test)]
    pub(crate) fn discard_queue(&self) {
        let mut slot = self.parker.lock();
        let dropped = slot.items.len() as u64;
        slot.items.clear();
        slot.item_bytes = 0;
        self.telemetry.owed.fetch_sub(dropped, Ordering::Relaxed);
    }

    #[cfg(test)]
    pub(crate) fn idle(&self) -> bool {
        self.points.is_empty() && self.lane.pending.is_none() && self.owed.is_empty()
    }

    #[cfg(test)]
    pub(crate) fn owed_states(&self) -> usize {
        self.owed.len()
    }

    #[cfg(test)]
    pub(crate) fn plan_owing(&self) -> bool {
        self.parker.lock().plan.owing
    }

    #[cfg(test)]
    pub(crate) fn measured(&self) -> u64 {
        self.parker.lock().measured
    }

    #[cfg(test)]
    pub(crate) fn spans(&self) -> &[u64] {
        &self.lane.spans
    }

    fn copying(&self) -> bool {
        self.persistent
            || self
                .demand_until
                .is_some_and(|until| Instant::now() < until)
    }

    pub(crate) fn drops(&mut self, total: u64) {
        if total > self.seen_drops {
            self.seen_drops = total;
            self.demand_until = Some(Instant::now() + DEMAND_WINDOW);
        }
    }

    pub(crate) fn reset(&mut self) {
        self.cursors.clear();
        self.skip_once = true;
    }

    pub(crate) fn take_skip(&mut self) -> bool {
        std::mem::take(&mut self.skip_once)
    }

    pub(crate) fn skips(&self) -> bool {
        self.skip_once
    }

    pub(crate) fn remove_object(&self, key: &ColdCacheKey) {
        let manager = &self.manager;
        self.closer.tier.while_held(|| manager.remove(key));
    }

    fn sync_waiting(&self) {
        let waiting = self.points.len() + usize::from(self.lane.pending.is_some());
        self.telemetry
            .waiting
            .store(waiting as u64, Ordering::Relaxed);
    }

    pub(crate) fn start(
        &mut self,
        seq: SeqId,
        lineage: &[u32],
        keys: &[Vec<u64>],
        hot: &[u64],
        salt: u64,
        segment_stride: usize,
    ) {
        if lineage.len() < self.minimum {
            self.cursors.remove(&seq);
            return;
        }
        let blocks = hot.len().min(keys.len()).min(lineage.len() / C::BLOCK_ROWS);
        let lineage = lineage[..blocks * C::BLOCK_ROWS].to_vec();
        let fp = self.fingerprint;
        let line = Lineage {
            tokens: &lineage,
            extra_keys: keys,
            salt,
        };
        let cold = lineage_keys(ColdGroup::Kv, fp, line, C::BLOCK_ROWS as u32);
        let draft_chain = lineage_keys(ColdGroup::HybridDraft, fp, line, C::BLOCK_ROWS as u32);
        let kept = if C::EXTEND_CURSOR {
            self.cursors.remove(&seq).filter(|c| {
                !c.broken
                    && c.segment_stride == segment_stride
                    && c.cold.len() <= cold.len()
                    && cold[..c.cold.len()] == c.cold[..]
            })
        } else {
            None
        };
        let (registered, pending, done, segments) = kept.map_or((0, Vec::new(), 0, 0), |c| {
            (c.registered, c.pending, c.blocks, c.segments)
        });
        self.cursors.insert(
            seq,
            Cursor {
                hot: hot[..blocks].to_vec(),
                lineage,
                cold,
                draft_chain,
                registered,
                pending,
                segment_stride,
                blocks: done,
                segments,
                broken: false,
                closed: false,
            },
        );
    }

    pub(crate) fn published(
        &mut self,
        store: &HybridStateStore<C::StoreState>,
        seq: SeqId,
        row: usize,
        segments: Option<&[C::Segment]>,
        replay: Option<EntryId>,
    ) {
        if let Some(cursor) = self.cursors.get_mut(&seq) {
            cursor.registered = cursor.registered.max(row);
            if let Some(segments) = segments
                && segments.len() > cursor.pending.len()
            {
                cursor.pending = segments.to_vec();
            }
        }
        let Some(id) = replay else {
            return;
        };
        if !self.closer.tier.is_open()
            || row < self.minimum
            || store.get(id).is_none_or(|entry| entry.disk().is_some())
            || self.points.iter().any(|point| point.id == id)
            || self.lane.pending.is_some_and(|(pending, _)| pending == id)
        {
            return;
        }
        if !self.copying() {
            let points = std::mem::take(&mut self.points);
            for point in points {
                let gone = store.get(point.id).is_none_or(|entry| {
                    entry.checkpoint() || self.codec.is_superseded(store, point.id)
                });
                if gone {
                    DiskTelemetry::bump(&self.telemetry.unneeded);
                } else {
                    self.points.push_back(point);
                }
            }
        }
        self.points.push_back(Point {
            id,
            due: Instant::now() + self.delay,
        });
        self.sync_waiting();
    }

    pub(crate) fn cut(&mut self, seq: SeqId, len: usize) {
        if let Some(cursor) = self.cursors.get_mut(&seq)
            && cursor.registered > len
            && cursor.blocks < (cursor.registered / C::BLOCK_ROWS).min(cursor.cold.len())
        {
            cursor.broken = true;
        }
    }

    fn owes(&self, seq: SeqId) -> bool {
        self.cursors.get(&seq).is_some_and(|cursor| {
            !cursor.broken
                && (cursor.blocks < (cursor.registered / C::BLOCK_ROWS).min(cursor.cold.len())
                    || cursor.segments < cursor.pending.len())
        })
    }

    pub(crate) fn top_up(&mut self, seq: SeqId, source: &mut dyn Blocks) {
        if !self.closer.tier.is_open() || !self.copying() || !self.owes(seq) {
            return;
        }
        let mut bytes = self.parker.lock().item_bytes;
        if bytes >= QUEUE_BYTES {
            return;
        }
        let fp = self.fingerprint;
        let fixed_layout = self.codec.block_layout();
        let Some(cursor) = self.cursors.get_mut(&seq) else {
            return;
        };
        let target = (cursor.registered / C::BLOCK_ROWS).min(cursor.cold.len());
        let mut items = Vec::new();
        let mut begun = false;
        while cursor.blocks < target && bytes < QUEUE_BYTES {
            let i = cursor.blocks;
            if self.manager.contains(&cursor.cold[i]) {
                cursor.blocks += 1;
                continue;
            }
            if !begun {
                if !source.begin(&cursor.lineage) {
                    return;
                }
                begun = true;
            }
            let Some(layers) = source.block(i, cursor.hot[i]) else {
                cursor.broken = true;
                break;
            };
            let Some(layout) = fixed_layout.clone().or_else(|| source.layout(&layers)) else {
                cursor.broken = true;
                break;
            };
            let item = ColdObject::Block(ColdCacheBlock {
                key: cursor.cold[i],
                fingerprint: fp,
                tokens: cursor.lineage[i * C::BLOCK_ROWS..(i + 1) * C::BLOCK_ROWS].to_vec(),
                layout,
                layers,
            });
            bytes += object_bytes(&item);
            items.push(item);
            cursor.blocks += 1;
        }
        let stride = cursor.segment_stride;
        let seg_format = self.codec.segment_format();
        while !cursor.broken && cursor.segments < cursor.pending.len() && bytes < QUEUE_BYTES {
            let k = cursor.segments;
            let Some(key) = draft_segment_key(
                fp,
                &cursor.draft_chain,
                &cursor.lineage,
                C::BLOCK_ROWS,
                stride,
                k,
            ) else {
                cursor.broken = true;
                break;
            };
            if !self.manager.contains_in(&key, ColdGroup::HybridDraft) {
                let object = self
                    .codec
                    .encode_segment(stride, &cursor.pending[k])
                    .ok()
                    .flatten()
                    .and_then(|payload| {
                        Some(ColdSidecar {
                            key,
                            fingerprint: fp,
                            layout: segment_sidecar_layout(
                                ColdGroup::HybridDraft,
                                (k + 1) * stride,
                                seg_format,
                                k + 1,
                                stride,
                                payload.len(),
                            )?,
                            tensors: vec![payload],
                        })
                    });
                let Some(object) = object else {
                    cursor.broken = true;
                    break;
                };
                let item = ColdObject::Sidecar(object);
                bytes += object_bytes(&item);
                items.push(item);
            }
            cursor.segments += 1;
        }
        if items.is_empty() {
            return;
        }
        let mut slot = self.parker.lock();
        for item in items {
            slot.item_bytes += object_bytes(&item);
            slot.items.push_back(item);
            self.telemetry.owed.fetch_add(1, Ordering::Relaxed);
        }
        self.parker.wake.notify_all();
    }

    pub(crate) fn close(&mut self, seq: SeqId) {
        if let Some(cursor) = self.cursors.get_mut(&seq) {
            cursor.closed = true;
        }
    }

    fn top_up_all(&mut self, source: &mut dyn Blocks) {
        let owing: Vec<SeqId> = self
            .cursors
            .keys()
            .copied()
            .filter(|&seq| self.owes(seq))
            .collect();
        for seq in owing {
            self.top_up(seq, source);
        }
        let open = self.closer.tier.is_open() && self.copying();
        let done: Vec<SeqId> = self
            .cursors
            .iter()
            .filter(|(_, cursor)| cursor.closed)
            .map(|(&seq, _)| seq)
            .filter(|&seq| !open || !self.owes(seq))
            .collect();
        for seq in done {
            self.cursors.remove(&seq);
        }
    }

    pub(crate) fn step(
        &mut self,
        store: &mut HybridStateStore<C::StoreState>,
        source: &mut dyn Blocks,
    ) -> bool {
        let landed = self.take_landings(store);
        if let Some(deadline) = self.take_close() {
            self.flush_points(store, source, deadline);
            return landed;
        }
        let now = Instant::now();
        if landed || now >= self.step_after {
            self.pass(store, source, now);
        }
        landed
    }

    pub(crate) fn step_now(
        &mut self,
        store: &mut HybridStateStore<C::StoreState>,
        source: &mut dyn Blocks,
    ) {
        self.take_landings(store);
        if let Some(deadline) = self.take_close() {
            self.flush_points(store, source, deadline);
            return;
        }
        self.pass(store, source, Instant::now());
    }

    fn take_close(&mut self) -> Option<Instant> {
        if !self.closer.closing.swap(false, Ordering::AcqRel) {
            return None;
        }
        Some(self.parker.lock().closing.unwrap_or_else(Instant::now))
    }

    fn pass(
        &mut self,
        store: &mut HybridStateStore<C::StoreState>,
        source: &mut dyn Blocks,
        now: Instant,
    ) {
        self.step_after = now + STEP_TOP_UP;
        #[cfg(test)]
        {
            self.step_runs += 1;
        }
        self.take_evictions(store);
        self.judge_owed(store, now);
        self.due_points(store, now);
        self.top_up_all(source);
        self.store_plan();
    }

    fn take_evictions(&mut self, store: &mut HybridStateStore<C::StoreState>) {
        let evicted = self.manager.take_evicted();
        if evicted.is_empty() {
            return;
        }
        let pending = self.lane.pending.map(|(_, key)| key);
        let hit: Vec<(EntryId, bool)> = store
            .entries()
            .filter(|entry| {
                entry.disk().is_some_and(|copy| {
                    copy.landed && Some(copy.key) != pending && evicted.contains(&copy.key)
                })
            })
            .map(|entry| (entry.id(), entry.ram().is_some()))
            .collect();
        for (id, in_memory) in hit {
            if in_memory {
                store.set_disk(id, None);
            } else {
                store.remove(id);
            }
        }
        self.owed.retain(|owed| !evicted.contains(&owed.key));
    }

    fn store_plan(&mut self) {
        let owing = !self.owed.is_empty()
            || (self.copying() && self.cursors.keys().any(|&seq| self.owes(seq)));
        let mut next_due = None;
        if self.lane.staging.is_some() && self.copying() {
            next_due = self.points.front().map(|point| point.due);
        }
        for owed in &self.owed {
            if let Some(stuck) = owed.stuck {
                let due = stuck + self.give_up;
                next_due = Some(next_due.map_or(due, |next: Instant| next.min(due)));
            }
        }
        let mut slot = self.parker.lock();
        slot.plan = Plan {
            epoch: slot.plan.epoch + 1,
            next_due,
            owing,
            next_step: Some(self.step_after),
        };
        self.parker.wake.notify_all();
        drop(slot);
        self.sync_waiting();
    }

    fn take_landings(&mut self, store: &mut HybridStateStore<C::StoreState>) -> bool {
        let mut any = false;
        while let Ok(landing) = self.lane.landed_rx.try_recv() {
            self.settle_landing(store, landing);
            any = true;
        }
        if any {
            self.sync_waiting();
        }
        any
    }

    fn settle_landing(&mut self, store: &mut HybridStateStore<C::StoreState>, landing: Landed) {
        let Landed::State { key, ok, payload } = landing else {
            return;
        };
        self.lane.staging = Some(payload);
        let Some((id, pending)) = self.lane.pending else {
            return;
        };
        if pending != key {
            return;
        }
        self.lane.pending = None;
        let ours = store
            .get(id)
            .and_then(|entry| entry.disk())
            .is_some_and(|copy| copy.key == key);
        if ok {
            #[cfg(test)]
            self.landed_states.push(key);
            match store.disk_mut(id) {
                Some(copy) if ours => copy.landed = true,
                _ => self.remove_object(&key),
            }
        } else {
            if ours {
                store.set_disk(id, None);
            }
            self.owed.retain(|owed| owed.key != key);
            DiskTelemetry::bump(&self.telemetry.refused);
        }
    }

    fn judge_owed(&mut self, store: &mut HybridStateStore<C::StoreState>, now: Instant) {
        if self.owed.is_empty() {
            return;
        }
        let quiet = (!self.copying() || !self.cursors.keys().any(|&seq| self.owes(seq)))
            && self.parker.lock().item_bytes == 0;
        let pending = self.lane.pending.map(|(_, key)| key);
        let mut kept = Vec::with_capacity(self.owed.len());
        for mut owed in std::mem::take(&mut self.owed) {
            let copy = store
                .get(owed.id)
                .and_then(|entry| entry.disk())
                .filter(|copy| copy.key == owed.key)
                .copied();
            let Some(copy) = copy else {
                if pending != Some(owed.key) {
                    self.remove_object(&owed.key);
                }
                continue;
            };
            while let Some((key, group)) = owed.needs.get(owed.landed) {
                if !self.manager.contains_in(key, *group) {
                    break;
                }
                owed.landed += 1;
            }
            if owed.landed == owed.needs.len() {
                if copy.landed {
                    if let Some(copy) = store.disk_mut(owed.id) {
                        copy.chain_owed = false;
                    }
                    DiskTelemetry::bump(&self.telemetry.durable);
                    self.telemetry
                        .state_bytes_written
                        .fetch_add(owed.bytes, Ordering::Relaxed);
                    continue;
                }
                owed.stuck = None;
                kept.push(owed);
                continue;
            }
            if !quiet {
                owed.stuck = None;
                kept.push(owed);
                continue;
            }
            let since = *owed.stuck.get_or_insert(now);
            if now.saturating_duration_since(since) >= self.give_up {
                self.remove_object(&owed.key);
                store.set_disk(owed.id, None);
                DiskTelemetry::bump(&self.telemetry.refused);
                continue;
            }
            kept.push(owed);
        }
        self.owed = kept;
    }

    fn wanted(&self, store: &HybridStateStore<C::StoreState>, id: EntryId) -> bool {
        let Some(entry) = store.get(id) else {
            DiskTelemetry::bump(&self.telemetry.unneeded);
            return false;
        };
        if entry.checkpoint() || self.codec.is_superseded(store, id) {
            DiskTelemetry::bump(&self.telemetry.unneeded);
            return false;
        }
        entry.disk().is_none()
    }

    fn due_points(&mut self, store: &mut HybridStateStore<C::StoreState>, now: Instant) {
        let mut at = 0;
        while let Some(&point) = self.points.get(at) {
            if point.due > now {
                break;
            }
            if !self.wanted(store, point.id) {
                self.points.remove(at);
                continue;
            }
            if !self.persistent {
                let in_use = store
                    .get(point.id)
                    .is_some_and(|entry| store.in_use(entry.mark()));
                if in_use {
                    at += 1;
                    continue;
                }
                if !self.copying() {
                    break;
                }
            }
            if self.lane.staging.is_none() {
                break;
            }
            match self.write(store, point.id) {
                Written::Started => {
                    self.points.remove(at);
                    break;
                }
                Written::OnTier => {
                    self.points.remove(at);
                }
                Written::Refused => {
                    DiskTelemetry::bump(&self.telemetry.refused);
                    self.points.remove(at);
                }
                Written::Busy => break,
            }
        }
    }

    fn write(&mut self, store: &mut HybridStateStore<C::StoreState>, id: EntryId) -> Written {
        if !self.closer.tier.is_open() {
            return Written::Refused;
        }
        let Some(entry) = store.get(id) else {
            return Written::Refused;
        };
        let Some(state) = entry.ram() else {
            return Written::Refused;
        };
        let row = entry.depth();
        if !self.codec.state_matches_lane(state, row) {
            return Written::Refused;
        }
        let lineage = Lineage {
            tokens: entry.tokens(),
            extra_keys: entry.extra_keys(),
            salt: entry.salt(),
        };
        let keys = |group| lineage_keys(group, self.fingerprint, lineage, C::BLOCK_ROWS as u32);
        let Some(&key) = keys(ColdGroup::DFlash2State).last() else {
            return Written::Refused;
        };
        let draft_chain = keys(ColdGroup::HybridDraft);
        let mut needs: Vec<(ColdCacheKey, ColdGroup)> = keys(ColdGroup::Kv)
            .into_iter()
            .map(|key| (key, ColdGroup::Kv))
            .collect();
        let segments = self.codec.state_segment_count(state);
        let stride = self.codec.segment_stride();
        for k in 0..segments {
            let Some(segment) = draft_segment_key(
                self.fingerprint,
                &draft_chain,
                entry.tokens(),
                C::BLOCK_ROWS,
                stride,
                k,
            ) else {
                return Written::Refused;
            };
            needs.push((segment, ColdGroup::HybridDraft));
        }
        let label = label_of(entry);
        let Some(layout) = self.lane.layout(row) else {
            return Written::Refused;
        };
        let bytes = state_object_bytes(&layout, &label);
        let copy = |landed| DiskCopy {
            key,
            bytes,
            landed,
            window_begin: row as u32,
            chain_owed: true,
        };
        let owed = Owed {
            id,
            key,
            needs,
            landed: 0,
            bytes,
            stuck: None,
        };
        if self.manager.contains_in(&key, ColdGroup::DFlash2State) {
            store.set_disk(id, Some(copy(true)));
            let on_tier_bytes = if C::COUNT_ON_TIER_BYTES { bytes } else { 0 };
            self.owed.push(Owed {
                bytes: on_tier_bytes,
                ..owed
            });
            return Written::OnTier;
        }
        let Some(mut payload) = self.lane.staging.take() else {
            return Written::Busy;
        };
        if !matches!(self.codec.encode_state_into(state, &mut payload), Ok(true)) {
            self.lane.staging = Some(payload);
            return Written::Refused;
        }
        #[cfg(test)]
        {
            self.encoded += 1;
        }
        let chain: Vec<ColdCacheKey> = owed.needs.iter().map(|(key, _)| *key).collect();
        self.manager.touch(&chain);
        let room = if self.persistent {
            MakeRoom::Evict
        } else {
            MakeRoom::Refuse
        };
        let object = ColdStateObject {
            key,
            fingerprint: self.fingerprint,
            layout,
            label,
            payload,
        };
        match self
            .manager
            .enqueue_state(object, room, self.lane.landed_tx.clone())
        {
            Ok(()) => {
                self.lane.pending = Some((id, key));
                store.set_disk(id, Some(copy(false)));
                self.owed.push(owed);
                self.parker.lock().measure = true;
                self.parker.wake.notify_all();
                Written::Started
            }
            Err(EnqueueRefused::Busy(object)) => {
                self.lane.staging = Some(object.payload);
                Written::Busy
            }
            Err(EnqueueRefused::Gone(object)) => {
                self.lane.staging = Some(object.payload);
                Written::Refused
            }
        }
    }

    fn owed_items(&self) -> usize {
        self.cursors
            .values()
            .filter(|cursor| !cursor.broken)
            .map(|cursor| {
                let target = (cursor.registered / C::BLOCK_ROWS).min(cursor.cold.len());
                target.saturating_sub(cursor.blocks)
                    + cursor.pending.len().saturating_sub(cursor.segments)
            })
            .sum()
    }

    pub(crate) fn flush_points(
        &mut self,
        store: &mut HybridStateStore<C::StoreState>,
        source: &mut dyn Blocks,
        deadline: Instant,
    ) -> bool {
        let mut complete = true;
        if self.persistent {
            let refused = self.telemetry.refused.load(Ordering::Relaxed);
            loop {
                self.take_landings(store);
                self.judge_owed(store, Instant::now());
                let (room, before) = match C::FLUSH_POLICY {
                    FlushPolicy::DrainCursorsStrict => (
                        self.parker.lock().item_bytes < QUEUE_BYTES,
                        self.owed_items(),
                    ),
                    FlushPolicy::PointsOnly => (false, 0),
                };
                self.top_up_all(source);
                let after = match C::FLUSH_POLICY {
                    FlushPolicy::DrainCursorsStrict => self.owed_items(),
                    FlushPolicy::PointsOnly => 0,
                };
                let left = deadline.saturating_duration_since(Instant::now());
                if left.is_zero() {
                    break;
                }
                if self.lane.staging.is_none() {
                    if let Ok(landing) = self
                        .lane
                        .landed_rx
                        .recv_timeout(left.min(Duration::from_millis(50)))
                    {
                        self.settle_landing(store, landing);
                    }
                    continue;
                }
                let Some(point) = self.points.back().copied() else {
                    match C::FLUSH_POLICY {
                        FlushPolicy::PointsOnly => break,
                        FlushPolicy::DrainCursorsStrict => {
                            if after == 0 || (room && after >= before) {
                                break;
                            }
                            std::thread::sleep(left.min(Duration::from_millis(10)));
                            continue;
                        }
                    }
                };
                if !self.wanted(store, point.id) {
                    self.points.pop_back();
                    continue;
                }
                match self.write(store, point.id) {
                    Written::Started | Written::OnTier => {
                        self.points.pop_back();
                    }
                    Written::Refused => {
                        DiskTelemetry::bump(&self.telemetry.refused);
                        self.points.pop_back();
                    }
                    Written::Busy => {
                        std::thread::sleep(left.min(Duration::from_millis(10)));
                    }
                }
            }
            let handed = self.parker.flush(deadline);
            let drained = self
                .manager
                .drain(deadline.saturating_duration_since(Instant::now()));
            self.take_landings(store);
            self.judge_owed(store, Instant::now());
            self.sync_waiting();
            complete = match C::FLUSH_POLICY {
                FlushPolicy::PointsOnly => {
                    self.points.is_empty() && self.lane.pending.is_none() && handed && drained
                }
                FlushPolicy::DrainCursorsStrict => {
                    self.points.is_empty()
                        && self.lane.pending.is_none()
                        && handed
                        && drained
                        && self.owed.is_empty()
                        && self.owed_items() == 0
                        && self.telemetry.refused.load(Ordering::Relaxed) == refused
                }
            };
        }
        let mut slot = self.parker.lock();
        slot.points_flushed = Some(complete);
        self.parker.done.notify_all();
        complete
    }

    pub(crate) fn dispose(
        &mut self,
        store: &mut HybridStateStore<C::StoreState>,
        source: &mut dyn Blocks,
    ) {
        let by = Instant::now() + SLOT_WAIT;
        self.flush_points(store, source, by);
        self.closing_by = Some(by);
    }

    pub(crate) fn chain_whole(
        &self,
        tokens: &[u32],
        keys: &[Vec<u64>],
        salt: u64,
        row: usize,
        pad_keys: bool,
        cached: &dyn Fn(u64) -> bool,
        draft_window: Option<usize>,
    ) -> Option<Vec<ColdCacheKey>> {
        let n = row / C::BLOCK_ROWS;
        if n == 0 || !row.is_multiple_of(C::BLOCK_ROWS) || tokens.len() < row {
            return None;
        }
        if !pad_keys && keys.len() < n {
            return None;
        }
        let padded: Vec<Vec<u64>>;
        let block_keys: &[Vec<u64>] = if pad_keys {
            padded = (0..n)
                .map(|i| keys.get(i).cloned().unwrap_or_default())
                .collect();
            &padded
        } else {
            &keys[..n]
        };
        let line = Lineage {
            tokens: &tokens[..row],
            extra_keys: block_keys,
            salt,
        };
        let cold = lineage_keys(ColdGroup::Kv, self.fingerprint, line, C::BLOCK_ROWS as u32);
        let hot = mlx_paged_attn::chain_hashes_per_block(
            &tokens[..row],
            C::BLOCK_ROWS as u32,
            block_keys,
            salt,
        );
        let hit = hot.iter().take_while(|&&h| cached(h)).count();
        let kv_whole = hot.len() == n
            && cold.len() == n
            && cold[hit.min(n)..]
                .iter()
                .all(|key| self.manager.contains(key));
        if !kv_whole {
            return None;
        }
        match draft_window {
            Some(stride) if stride > 0 => {
                let count = self.codec.expected_segment_count(row, stride);
                let draft_chain = lineage_keys(
                    ColdGroup::HybridDraft,
                    self.fingerprint,
                    line,
                    C::BLOCK_ROWS as u32,
                );
                (0..count)
                    .map(|k| {
                        draft_segment_key(
                            self.fingerprint,
                            &draft_chain,
                            tokens,
                            C::BLOCK_ROWS,
                            stride,
                            k,
                        )
                    })
                    .collect::<Option<Vec<_>>>()
                    .filter(|keys| {
                        keys.iter()
                            .all(|key| self.manager.contains_in(key, ColdGroup::HybridDraft))
                    })
            }
            _ => Some(Vec::new()),
        }
    }

    pub(crate) fn discard(&mut self, store: &mut HybridStateStore<C::StoreState>, id: EntryId) {
        if let Some(copy) = store.get(id).and_then(|entry| entry.disk()).copied() {
            self.remove_object(&copy.key);
        }
        store.remove(id);
        DiskTelemetry::bump(&self.telemetry.dropped_chain);
    }

    pub(crate) fn adopt(
        &mut self,
        store: &mut HybridStateStore<C::StoreState>,
        cached_bytes_at_row: impl Fn(usize) -> u64,
    ) -> u64 {
        if !self.persistent || store.entries().next().is_some() {
            return 0;
        }
        let tier = Arc::clone(&self.closer.tier);
        let (spans, geometry) = (self.lane.spans.clone(), self.lane.geometry.clone());
        let has_draft = self.codec.has_draft();
        let stride = self.codec.segment_stride();
        let fp = self.fingerprint;
        let manager = Arc::clone(&self.manager);
        let codec = &self.codec;
        let adopted = adopt_states(store, &tier, &spans, &geometry, |label| {
            let row = label.tokens.len();
            if has_draft {
                let lineage = Lineage {
                    tokens: &label.tokens,
                    extra_keys: &label.extra_keys,
                    salt: label.salt,
                };
                let draft_chain =
                    lineage_keys(ColdGroup::HybridDraft, fp, lineage, C::BLOCK_ROWS as u32);
                let count = codec.expected_segment_count(row, stride);
                for k in 0..count {
                    let key = draft_segment_key(
                        fp,
                        &draft_chain,
                        &label.tokens,
                        C::BLOCK_ROWS,
                        stride,
                        k,
                    )?;
                    if !manager.contains_in(&key, ColdGroup::HybridDraft) {
                        return None;
                    }
                }
            }
            Some((cached_bytes_at_row(row), u32::try_from(row).ok()?))
        });
        self.telemetry.adopted.fetch_add(adopted, Ordering::Relaxed);
        if adopted > 0 {
            tracing::info!(
                "{}: took back {adopted} stored states from the SSD cache",
                C::LOG_PREFIX
            );
        }
        adopted
    }

    pub(crate) fn read_state(
        &mut self,
        store: &mut HybridStateStore<C::StoreState>,
        key: ColdCacheKey,
        row: usize,
        extra_keys: &[Vec<u64>],
        segment_keys: &[ColdCacheKey],
        bound: u64,
    ) -> DiskRead<C::DecodedState> {
        let deadline = Instant::now() + SLOT_WAIT;
        while self.lane.staging.is_none() {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return DiskRead::Busy;
            }
            if let Ok(landing) = self
                .lane
                .landed_rx
                .recv_timeout(left.min(Duration::from_millis(50)))
            {
                self.settle_landing(store, landing);
                self.sync_waiting();
            }
        }
        let Some(mut staging) = self.lane.staging.take() else {
            return DiskRead::Busy;
        };
        let stride = self.codec.segment_stride();
        let seg_format = self.codec.segment_format();
        let read = (|| {
            let layout = self.lane.layout(row)?;
            self.manager
                .load_state(key, self.fingerprint, &layout, extra_keys, &mut staging)?;
            let mut segments = Vec::with_capacity(segment_keys.len());
            for (k, seg_key) in segment_keys.iter().enumerate() {
                let object = self.manager.load_object(
                    *seg_key,
                    self.fingerprint,
                    ColdGroup::HybridDraft,
                    bound,
                )?;
                if !segment_sidecar_fits(
                    &object.layout,
                    ColdGroup::HybridDraft,
                    (k + 1) * stride,
                    seg_format,
                    k + 1,
                    stride,
                ) {
                    return None;
                }
                segments.push(self.codec.decode_segment(object.tensors.first()?)?);
            }
            self.codec.decode_state(&staging, row, segments)
        })();
        self.lane.staging = Some(staging);
        match read {
            Some(state) => DiskRead::State(state),
            None => {
                self.remove_object(&key);
                DiskTelemetry::bump(&self.telemetry.dropped_chain);
                DiskRead::Failed
            }
        }
    }

    pub(crate) fn restored(&self) {
        DiskTelemetry::bump(&self.telemetry.disk_restored);
    }
}

impl<C: PrefixCodec> Drop for PrefixDisk<C> {
    fn drop(&mut self) {
        self.closer.finish(
            self.closing_by
                .unwrap_or_else(|| Instant::now() + SLOT_WAIT),
        );
        {
            let mut slot = self.parker.lock();
            slot.stop = true;
            self.parker.wake.notify_all();
            self.parker.done.notify_all();
        }
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
