//! The process memory governor, after Splash `MemoryGovernor`: the limit,
//! the charge, admission of our own allocations, the host's rule and its
//! control pass.
//!
//! The books sit behind one mutex. The MLX and coordinator counters and the
//! host are read before it is taken (lock order: pool growth lock, then the
//! cache-limit coordinator, then the governor), and units are counted and
//! taken outside it, on the caller's model thread.

use std::cell::Cell;
use std::collections::VecDeque;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use tracing::{info, warn};

use super::host::{
    HOST_MARGIN_BYTES, HOST_SETTLE_SECONDS, HostMemory, HostRule, Pressure,
    available_reserve_bytes, query_host_memory, query_pressure, spare_headroom,
    spare_reserve_bytes,
};
use super::ledger::{
    Charge, ClaimClass, ClaimId, ClaimRank, ClaimSpec, FreeOutcome, FreePool, FreeResult, Ledger,
    LedgerSnapshot, Need, NeedTiming, ReachClass, VictimSource, WantedChains, count_available,
    decide, take_rights,
};
use super::metrics::{Gap, Metrics, TtftParts};

const MIB: u64 = 1 << 20;
/// Splash `MemoryStatusReporter`'s decision log length.
const DECISIONS_KEPT: usize = 64;
/// Splash `supervision::kStartRetryWindow` and `kStartupMemoryRecoveryPoll`.
const START_RETRY_SECONDS: f64 = 30.0;
const START_POLL_SECONDS: f64 = 1.0;

/// The counters `charged` is made of: MLX active and free-pool bytes, and
/// the paged pools registered with the cache-limit coordinator, which sit
/// outside MLX.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct MemoryStats {
    pub active: u64,
    pub pools: u64,
    pub cache: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct HostSample {
    pub memory: Option<HostMemory>,
    pub pressure: Option<Pressure>,
}

/// What the governor reads and sets outside its books. Every call is made
/// without the governor's mutex.
pub(crate) trait GovernorIo: Send + Sync {
    fn stats(&self) -> MemoryStats;
    fn host(&self) -> HostSample;
    /// Seconds on a monotonic clock.
    fn now(&self) -> f64;
    fn sleep(&self, seconds: f64);
    fn clear_cache(&self);
    /// MLX's memory limit; returns the previous one.
    fn set_memory_limit(&self, bytes: u64) -> u64;
    fn physical_memory(&self) -> u64;
}

pub(crate) struct SystemIo {
    start: Instant,
}

impl SystemIo {
    pub(crate) fn new() -> Self {
        Self {
            start: Instant::now(),
        }
    }
}

impl GovernorIo for SystemIo {
    fn stats(&self) -> MemoryStats {
        MemoryStats {
            active: crate::array::memory::active_memory_bytes(),
            pools: crate::cache_limit::coordinator().registered_pool_bytes(),
            cache: crate::array::memory::cache_memory_bytes(),
        }
    }

    fn host(&self) -> HostSample {
        let pressure = query_pressure();
        HostSample {
            memory: query_host_memory(pressure),
            pressure,
        }
    }

    fn now(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    fn sleep(&self, seconds: f64) {
        std::thread::sleep(Duration::from_secs_f64(seconds));
    }

    fn clear_cache(&self) {
        crate::array::memory::clear_cache();
    }

    fn set_memory_limit(&self, bytes: u64) -> u64 {
        crate::array::memory::set_memory_limit_bytes(bytes)
    }

    fn physical_memory(&self) -> u64 {
        crate::utils::host::physical_memory().unwrap_or(0)
    }
}

/// Why an admission was refused (Splash `AllocationFailure`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Failure {
    /// The limit, or the room claims ranked above hold.
    EngineBudget,
    HostPressure,
    /// Metal returned no buffer for bytes the governor admitted.
    DriverRejected,
    /// A defect: no claim's scope was entered.
    OutsideScope,
}

impl Failure {
    /// Splash `allocationFailureName`.
    pub(crate) fn name(failure: Option<Failure>) -> &'static str {
        match failure {
            None => "none",
            Some(Failure::EngineBudget) => "engine memory budget exceeded",
            Some(Failure::HostPressure) => "host memory reserve protected",
            Some(Failure::DriverRejected) => "Metal driver rejected allocation",
            Some(Failure::OutsideScope) => "allocated outside a claim's scope",
        }
    }
}

thread_local! {
    /// The scope's governor (by address), claim, and whether the scheduler
    /// judges its refusals.
    static SCOPE: Cell<Option<(usize, ClaimId, bool)>> = const { Cell::new(None) };
    /// The refusal an allocation under a scheduled lane's scope left for
    /// the scheduler to judge.
    static LANE_DENIAL: Cell<Option<super::allocate::Denial>> = const { Cell::new(None) };
    /// What refused this thread's latest pool load, and the stage it
    /// reached (Splash `RuntimeBootstrapReport`).
    static START_REFUSAL: Cell<Option<(Failure, u8)>> = const { Cell::new(None) };
    static CANCEL: std::cell::RefCell<Option<Arc<AtomicBool>>> =
        const { std::cell::RefCell::new(None) };
}

/// While one lives, a wait for memory on this thread ends when `flag` is
/// set: the whole turn's cancel.
pub(crate) struct CancelScope {
    previous: Option<Arc<AtomicBool>>,
    _thread: PhantomData<*const ()>,
}

impl CancelScope {
    pub(crate) fn enter(flag: Arc<AtomicBool>) -> Self {
        Self {
            previous: CANCEL.with(|cancel| cancel.replace(Some(flag))),
            _thread: PhantomData,
        }
    }
}

impl Drop for CancelScope {
    fn drop(&mut self) {
        let previous = self.previous.take();
        CANCEL.with(|cancel| *cancel.borrow_mut() = previous);
    }
}

/// The turn this thread runs was cancelled.
pub(crate) fn cancelled() -> bool {
    CANCEL.with(|cancel| {
        cancel
            .borrow()
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::Relaxed))
    })
}

/// While one lives, admissions on this thread are for its claim (Splash
/// `MemoryGovernor::Scope`). It restores the scope it replaced.
pub(crate) struct Scope {
    previous: Option<(usize, ClaimId, bool)>,
    _thread: PhantomData<*const ()>,
}

impl Scope {
    pub(crate) fn enter(claim: ClaimRef<'_>) -> Self {
        Self::enter_with(claim, false)
    }

    /// A scheduled lane's admission or resume (Splash `Engine::admit`): an
    /// allocation under it neither waits nor judges; it leaves its refusal
    /// for the scheduler ([`take_lane_denial`]), which judges it beside the
    /// resident lanes and defers the request.
    pub(crate) fn enter_lane(claim: ClaimRef<'_>) -> Self {
        LANE_DENIAL.with(|denial| denial.set(None));
        Self::enter_with(claim, true)
    }

    fn enter_with(claim: ClaimRef<'_>, lane: bool) -> Self {
        let entry = (std::ptr::from_ref(claim.governor) as usize, claim.id, lane);
        Self {
            previous: SCOPE.with(|scope| scope.replace(Some(entry))),
            _thread: PhantomData,
        }
    }
}

/// This thread's scope is a scheduled lane's admission or resume.
pub(crate) fn scheduler_judges() -> bool {
    SCOPE.with(Cell::get).is_some_and(|(_, _, lane)| lane)
}

/// An allocation under a scheduled lane's scope was refused.
pub(crate) fn note_lane_denial(denial: super::allocate::Denial) {
    LANE_DENIAL.with(|slot| slot.set(Some(denial)));
}

/// The refusal the latest allocation under a scheduled lane's scope left.
pub(crate) fn take_lane_denial() -> Option<super::allocate::Denial> {
    LANE_DENIAL.with(Cell::take)
}

impl Drop for Scope {
    fn drop(&mut self) {
        SCOPE.with(|scope| scope.set(self.previous));
    }
}

/// A claim that closes when dropped (Splash `ClaimHandle`).
pub(crate) struct ClaimHandle<'g> {
    governor: &'g Governor,
    id: ClaimId,
}

impl<'g> ClaimHandle<'g> {
    pub(crate) fn id(&self) -> ClaimId {
        self.id
    }

    pub(crate) fn claim(&self) -> ClaimRef<'g> {
        ClaimRef {
            governor: self.governor,
            id: self.id,
        }
    }
}

/// An open claim, owned by a [`ClaimHandle`] elsewhere (a request's, or the
/// thread scope's).
#[derive(Clone, Copy)]
pub(crate) struct ClaimRef<'g> {
    pub governor: &'g Governor,
    pub id: ClaimId,
}

impl Drop for ClaimHandle<'_> {
    fn drop(&mut self) {
        self.governor.ledger(|ledger| ledger.close(self.id));
    }
}

/// Bytes admitted while the allocation that needs them runs; released
/// when dropped.
pub(crate) struct Reservation<'g> {
    governor: &'g Governor,
    bytes: u64,
}

impl Drop for Reservation<'_> {
    fn drop(&mut self) {
        if self.bytes > 0 {
            let mut books = self.governor.lock();
            books.reserved = books.reserved.saturating_sub(self.bytes);
            books.retired = books.retired.wrapping_add(1);
        }
    }
}

/// A loaded model's terms: its plan's hard budget and untracked reserve
/// (pipeline + runtime overhead + lane transient), and the bytes of one KV
/// block of its pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ModelRegistration {
    pub model_id: u64,
    pub hard_budget: u64,
    pub untracked: u64,
    pub page_bytes: u64,
}

/// Unregisters its model when dropped.
pub(crate) struct ModelGuard<'g> {
    governor: &'g Governor,
    model_id: u64,
}

impl<'g> ModelGuard<'g> {
    pub(crate) fn model_id(&self) -> u64 {
        self.model_id
    }

    pub(crate) fn governor(&self) -> &'g Governor {
        self.governor
    }
}

impl Drop for ModelGuard<'_> {
    fn drop(&mut self) {
        self.governor.unregister(self.model_id);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HostPass {
    pub target: u64,
    pub result: FreeResult,
}

/// What a control pass did: the host's claim, when one ran.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ControlPass {
    pub pass: Option<HostPass>,
}

impl ControlPass {
    /// The host's claim waits for what lands: run the pass again at the
    /// next iteration.
    pub(crate) fn again(&self) -> bool {
        self.pass
            .is_some_and(|pass| pass.result.outcome == FreeOutcome::Pending)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HostStatus {
    pub measured: bool,
    pub available: u64,
    pub available_reserve: u64,
    pub spare: u64,
    pub spare_reserve: u64,
    pub headroom: u64,
    pub system_pressure: Pressure,
    pub growth_allowed: bool,
}

/// Splash `MemoryGovernorSnapshot`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GovernorSnapshot {
    pub limit: u64,
    pub charged: u64,
    pub headroom: u64,
    /// Critical; else warning while macOS warns or the host is in its
    /// band; else normal.
    pub pressure: Pressure,
    pub denied_reservations: u64,
    pub host: HostStatus,
    pub ledger: LedgerSnapshot,
}

/// A model's accounts (Splash `MemoryLedger::Account`): its KV pool, its
/// GDN and draft states, and its image features. Its thread publishes
/// them; the ledger reads them at each fill.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct MemoryAccounts {
    pub kv: u64,
    pub states: u64,
    pub embeddings: u64,
}

#[derive(Default)]
struct PublishedAccounts {
    kv: AtomicU64,
    states: AtomicU64,
    embeddings: AtomicU64,
}

/// Splash `ResourceWaitSnapshot` of one model.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct ResourceWaitStatus {
    pub memory: u32,
    pub concurrency: u32,
    pub suspended: u32,
    pub held_behind_refusal: u32,
    pub draining: bool,
    pub oldest_ms: f64,
}

impl ResourceWaitStatus {
    fn merge(&mut self, other: &Self) {
        self.memory += other.memory;
        self.concurrency += other.concurrency;
        self.suspended += other.suspended;
        self.held_behind_refusal += other.held_behind_refusal;
        self.draining |= other.draining;
        self.oldest_ms = self.oldest_ms.max(other.oldest_ms);
    }
}

/// Splash's engine counters, summed over the registered models and the
/// process.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Counters {
    pub replans: u64,
    pub resource_suspensions: u64,
    pub resource_resumptions: u64,
    pub resource_timeouts: u64,
    pub capacity_exhausted: u64,
    pub metal_out_of_memory: u64,
    pub publications_refused: u64,
}

struct ModelEntry {
    id: u64,
    hard_budget: u64,
    untracked: u64,
    /// Splash `Engine::resourceEpoch_`: moves at what may end a wait.
    epoch: u64,
    /// Splash `Engine::allocationFailed_`.
    allocation_failed: bool,
    /// Splash `Engine::drainEndMilliseconds_`: resident lanes drain
    /// before admission resumes, until then; 0.0 when unset.
    drain_end: f64,
    /// Splash `resourceSuspensions`.
    suspensions: u64,
    /// Splash `resourceResumptions`.
    resumptions: u64,
    /// Splash `Cache::wanted_`.
    wanted: Arc<WantedChains>,
    /// The next control pass's time (Splash `MemoryControl`'s period).
    next_control: f64,
    /// Splash `MemoryGovernor::nextHostPassMilliseconds_`, per model: a
    /// pass sees only its model's units, so one model's pass must not
    /// spend another's settle time.
    next_host_pass: f64,
    /// Publications a refused claim skipped.
    publications_refused: u64,
    /// A command exhausted Metal memory and the model's caches are not
    /// reset yet.
    gpu_oom: bool,
    /// Splash `metal_out_of_memory` restarts.
    metal_out_of_memory: u64,
    accounts: Arc<PublishedAccounts>,
    /// What the model thread last published.
    waits: ResourceWaitStatus,
    replans: u64,
}

struct Books {
    models: Vec<ModelEntry>,
    reserved: u64,
    denied: u64,
    system_pressure: Pressure,
    host_rule: HostRule,
    ledger: Ledger,
    /// Reservations dropped so far. A drop between a sample and the lock
    /// leaves its bytes in neither: the caller samples again.
    retired: u64,
    band_waived: bool,
    /// MLX's limit before the first model registered, put back when the
    /// last one leaves.
    limit_before: Option<u64>,
    spare_reserve: u64,
    available_reserve: u64,
}

impl Books {
    /// The smallest registered hard budget.
    fn limit(&self) -> u64 {
        self.models
            .iter()
            .map(|model| model.hard_budget)
            .min()
            .unwrap_or(u64::MAX)
    }

    fn untracked(&self) -> u64 {
        self.models
            .iter()
            .fold(0u64, |sum, model| sum.saturating_add(model.untracked))
    }

    fn charge(&self, stats: &MemoryStats) -> Charge {
        Charge {
            allocated: stats.active.saturating_add(stats.pools),
            untracked: self.untracked(),
            cache: stats.cache,
        }
    }

    /// Splash `MemoryGovernor::roomBytes`.
    fn room(&self, stats: &MemoryStats) -> u64 {
        self.limit()
            .saturating_sub(self.charge(stats).charged())
            .saturating_sub(self.reserved)
    }

    fn critical(&self) -> bool {
        self.host_rule.level() == Pressure::Critical
    }

    /// Splash `MemoryGovernor::sampleHostRule`.
    fn sample_host_rule(&mut self, host: &HostSample) {
        if let Some(pressure) = host.pressure {
            self.system_pressure = pressure;
        }
        let refused = self.ledger.host_refused_waiting_bytes();
        self.host_rule.sample(
            host.memory.as_ref(),
            self.spare_reserve,
            self.reserved,
            self.system_pressure,
            0,
            refused,
            self.band_waived,
        );
        // A waiver lasts the band it ended.
        if !self.host_rule.constrained() {
            self.band_waived = false;
        }
    }

    fn model(&mut self, id: u64) -> Option<&mut ModelEntry> {
        self.models.iter_mut().find(|model| model.id == id)
    }

    /// MLX collects its free pool at the ledger's limit and waits above it,
    /// less the pools MLX does not see.
    fn apply_memory_limit(&mut self, io: &dyn GovernorIo, pools: u64) {
        if self.models.is_empty() {
            if let Some(before) = self.limit_before.take() {
                io.set_memory_limit(before);
            }
            return;
        }
        let previous = io.set_memory_limit(self.limit().saturating_sub(pools));
        if self.limit_before.is_none() {
            self.limit_before = Some(previous);
        }
    }
}

/// The MLX free pool as the ledger's unsized unit.
struct MlxFreePool<'a> {
    io: &'a dyn GovernorIo,
    untracked: u64,
}

impl FreePool for MlxFreePool<'_> {
    fn charge(&self) -> Charge {
        let stats = self.io.stats();
        Charge {
            allocated: stats.active.saturating_add(stats.pools),
            untracked: self.untracked,
            cache: stats.cache,
        }
    }

    fn clear(&mut self) {
        self.io.clear_cache();
    }
}

/// A whole turn's claim, alone on its model (Splash `Engine::growthClass`
/// of a lone lane reaches what turns use). `request_id` 0: the engine's
/// own.
pub(crate) fn whole_turn_spec(request_id: u64) -> ClaimSpec {
    ClaimSpec {
        cls: ClaimClass::InService,
        request_id,
        rank: ClaimRank {
            priority: 1,
            decoding: false,
            completed_tokens: 0,
            earliness: u64::MAX - request_id,
        },
        reach: ReachClass::InUse,
        timing: NeedTiming::CanWait,
    }
}

/// A request id for a claim, from 1, never reused.
pub(crate) fn next_request_id() -> u64 {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

fn host_spec() -> ClaimSpec {
    ClaimSpec {
        cls: ClaimClass::Host,
        request_id: 0,
        rank: ClaimRank::default(),
        reach: ReachClass::Ordinary,
        timing: NeedTiming::Immediate,
    }
}

pub(crate) struct Governor {
    io: Arc<dyn GovernorIo>,
    /// Registered models, read without the mutex.
    registered: AtomicUsize,
    books: Mutex<Books>,
    /// After the books in lock order; never held while taking them.
    metrics: Mutex<Metrics>,
    decisions: Arc<Mutex<VecDeque<String>>>,
}

impl Governor {
    pub(crate) fn new(io: Arc<dyn GovernorIo>) -> Self {
        let physical = io.physical_memory();
        let mut ledger = Ledger::new();
        let decisions: Arc<Mutex<VecDeque<String>>> = Arc::default();
        let log = Arc::clone(&decisions);
        ledger.set_decision_log(Arc::new(move |line| {
            info!(target: "mlx_core::memory", "{line}");
            let mut log = log.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            if log.len() == DECISIONS_KEPT {
                log.pop_front();
            }
            log.push_back(line.to_string());
        }));
        Self {
            io,
            registered: AtomicUsize::new(0),
            metrics: Mutex::default(),
            decisions,
            books: Mutex::new(Books {
                models: Vec::new(),
                reserved: 0,
                denied: 0,
                system_pressure: Pressure::Normal,
                host_rule: HostRule::default(),
                ledger,
                retired: 0,
                band_waived: false,
                limit_before: None,
                spare_reserve: spare_reserve_bytes(physical),
                available_reserve: available_reserve_bytes(physical),
            }),
        }
    }

    fn lock(&self) -> MutexGuard<'_, Books> {
        self.books
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Stats and host sampled outside the books (lock order), then the
    /// books with no reservation retired since the sample: Splash
    /// `MemoryGovernor::tryReserve` reads charged and reserved at one
    /// instant.
    fn sample_locked(&self) -> (MemoryStats, HostSample, MutexGuard<'_, Books>) {
        loop {
            let seen = self.lock().retired;
            let stats = self.io.stats();
            let host = self.io.host();
            let books = self.lock();
            if books.retired == seen {
                return (stats, host, books);
            }
        }
    }

    /// The books, under the mutex. Never count or take units inside.
    pub(crate) fn ledger<R>(&self, f: impl FnOnce(&mut Ledger) -> R) -> R {
        f(&mut self.lock().ledger)
    }

    #[cfg(test)]
    /// Splash `MemoryGovernor`'s constructor checks: charged must fit the
    /// new limit, and macOS must have more than its reserve available. The
    /// "memory plan:" prefix is what the TS layer reads as a load refusal.
    pub(crate) fn register_model(
        &self,
        model: ModelRegistration,
    ) -> Result<ModelGuard<'_>, String> {
        self.try_register(model).map_err(|(_, text)| text)
    }

    fn try_register(&self, model: ModelRegistration) -> Result<ModelGuard<'_>, (Failure, String)> {
        let (stats, host, mut books) = self.sample_locked();
        let limit = books.limit().min(model.hard_budget);
        let charged = Charge {
            untracked: books.untracked().saturating_add(model.untracked),
            ..books.charge(&stats)
        }
        .charged();
        if charged > limit {
            return Err((
                Failure::EngineBudget,
                format!(
                    "memory plan: existing Metal allocations exceed memory governor limit ({} MiB charged, {} MiB limit)",
                    charged / MIB,
                    limit / MIB
                ),
            ));
        }
        let available = host.memory.map_or(0, |memory| memory.available);
        if available <= books.available_reserve {
            return Err((
                Failure::HostPressure,
                format!(
                    "memory plan: macOS is short of memory ({} MiB available, {} MiB kept for macOS); close memory-heavy applications",
                    available / MIB,
                    books.available_reserve / MIB
                ),
            ));
        }
        let accounts = Arc::new(PublishedAccounts::default());
        books.models.push(ModelEntry {
            id: model.model_id,
            hard_budget: model.hard_budget,
            untracked: model.untracked,
            epoch: 0,
            allocation_failed: false,
            drain_end: 0.0,
            suspensions: 0,
            resumptions: 0,
            wanted: Arc::default(),
            next_control: 0.0,
            next_host_pass: 0.0,
            publications_refused: 0,
            gpu_oom: false,
            metal_out_of_memory: 0,
            accounts: Arc::clone(&accounts),
            waits: ResourceWaitStatus::default(),
            replans: 0,
        });
        books
            .ledger
            .set_page_bytes(model.model_id, model.page_bytes);
        let read = |field: fn(&PublishedAccounts) -> &AtomicU64| {
            let accounts = Arc::clone(&accounts);
            Arc::new(move || field(&accounts).load(Ordering::Relaxed))
                as Arc<dyn Fn() -> u64 + Send + Sync>
        };
        books
            .ledger
            .add_account(model.model_id, "kv", read(|accounts| &accounts.kv));
        books
            .ledger
            .add_account(model.model_id, "states", read(|accounts| &accounts.states));
        books.ledger.add_account(
            model.model_id,
            "embeddings",
            read(|accounts| &accounts.embeddings),
        );
        self.registered.fetch_add(1, Ordering::AcqRel);
        books.apply_memory_limit(&*self.io, stats.pools);
        Ok(ModelGuard {
            governor: self,
            model_id: model.model_id,
        })
    }

    /// A planned pool load (Splash's start): the model registers, then its
    /// runway of `runway_bytes` is admitted as a Floor claim, by the host's
    /// rule and the limit, while `build` allocates it. A refusal is left for
    /// [`start_under_growth_lock`]'s retry.
    pub(crate) fn load_pool<R>(
        &self,
        model: ModelRegistration,
        runway_bytes: u64,
        build: impl FnOnce() -> Result<R, String>,
    ) -> Result<(ModelGuard<'_>, R), String> {
        let refused = |failure: Failure, stage: u8, text: String| {
            START_REFUSAL.with(|refusal| refusal.set(Some((failure, stage))));
            text
        };
        let registration = self
            .try_register(model)
            .map_err(|(failure, text)| refused(failure, START_REGISTER, text))?;
        let floor = self.open(
            ClaimSpec {
                cls: ClaimClass::Floor,
                request_id: 0,
                rank: ClaimRank::default(),
                reach: ReachClass::Ordinary,
                timing: NeedTiming::Immediate,
            },
            model.model_id,
        );
        let _scope = Scope::enter(floor.claim());
        let reservation = self.try_reserve(runway_bytes).map_err(|failure| {
            refused(
                failure,
                START_RUNWAY,
                format!(
                    "memory plan: the KV pool's runway ({} MiB) could not be allocated: {}",
                    runway_bytes / MIB,
                    Failure::name(Some(failure))
                ),
            )
        })?;
        let built = build()?;
        drop(reservation);
        Ok((registration, built))
    }

    fn unregister(&self, model_id: u64) {
        let stats = self.io.stats();
        let mut books = self.lock();
        let Some(at) = books.models.iter().position(|model| model.id == model_id) else {
            return;
        };
        books.models.remove(at);
        books.ledger.remove_model(model_id);
        self.registered.fetch_sub(1, Ordering::AcqRel);
        books.apply_memory_limit(&*self.io, stats.pools);
    }

    /// A pool's bytes changed: MLX's limit follows. Call only after the
    /// coordinator's lock is released.
    pub(crate) fn pools_changed(&self) {
        if self.registered.load(Ordering::Acquire) == 0 {
            return;
        }
        let stats = self.io.stats();
        self.lock().apply_memory_limit(&*self.io, stats.pools);
    }

    /// This thread's scope claim, when it is one of this governor's.
    pub(crate) fn scope_claim(&self) -> Option<ClaimId> {
        SCOPE
            .with(Cell::get)
            .filter(|(governor, _, _)| *governor == std::ptr::from_ref(self) as usize)
            .map(|(_, claim, _)| claim)
    }

    /// Opens a claim for `model`'s allocations (0: no registered model).
    pub(crate) fn open(&self, spec: ClaimSpec, model: u64) -> ClaimHandle<'_> {
        ClaimHandle {
            governor: self,
            id: self.ledger(|ledger| ledger.open(spec, model)),
        }
    }

    /// Splash `MemoryGovernor::tryReserve`: `bytes` under the limit, beside
    /// the room claims ranked above the scope's claim hold, if the host's
    /// rule admits them for its class. A refusal is the host's when the
    /// host refuses, and records what the claim lacks.
    pub(crate) fn try_reserve(&self, bytes: u64) -> Result<Reservation<'_>, Failure> {
        let Some(claim) = self.scope_claim() else {
            return Err(Failure::OutsideScope);
        };
        let (stats, host, mut books) = self.sample_locked();
        books.sample_host_rule(&host);
        let Some(state) = books.ledger.claim(claim) else {
            return Err(Failure::OutsideScope);
        };
        let (cls, exempt) = (state.spec.cls, state.host_exempt);
        let limit = books.limit();
        let observed = books.charge(&stats).charged();
        let requested = books.reserved.checked_add(bytes);
        let engine_fits =
            requested.is_some_and(|requested| observed <= limit && requested <= limit - observed);
        let held = books.ledger.held_bytes(claim);
        let fits = engine_fits && held <= limit - observed - requested.unwrap_or(0);
        let host_admits = books.host_rule.admits(cls, bytes, exempt);
        if !fits || !host_admits {
            let failure = if host_admits {
                Failure::EngineBudget
            } else {
                Failure::HostPressure
            };
            books.denied = books.denied.saturating_add(1);
            books.ledger.refused(
                claim,
                Need {
                    bytes,
                    host_refused: !host_admits,
                    ..Need::default()
                },
            );
            books.ledger.set_failure(claim, failure);
            return Err(failure);
        }
        books.reserved = requested.unwrap_or(books.reserved);
        Ok(Reservation {
            governor: self,
            bytes,
        })
    }

    /// Metal returned no buffer for `bytes` the scope's claim was admitted:
    /// a refusal of the same bytes (Splash `allocationAdmission` catching
    /// `MetalAllocationError`).
    pub(crate) fn driver_rejected(&self, bytes: u64) {
        let Some(claim) = self.scope_claim() else {
            return;
        };
        let mut books = self.lock();
        books.denied = books.denied.saturating_add(1);
        books.ledger.refused(
            claim,
            Need {
                bytes,
                ..Need::default()
            },
        );
        books.ledger.set_failure(claim, Failure::DriverRejected);
    }

    /// What refused the claim's latest attempt.
    pub(crate) fn failure(&self, claim: ClaimId) -> Option<Failure> {
        self.ledger(|ledger| ledger.claim(claim).and_then(|state| state.failure))
    }

    /// The claim's spec, while it is open.
    pub(crate) fn spec(&self, claim: ClaimId) -> Option<ClaimSpec> {
        self.ledger(|ledger| ledger.claim(claim).map(|state| state.spec))
    }

    /// The blocks held against an allocation in this thread's scope; 0
    /// outside one, or while no claim waits.
    pub(crate) fn held_blocks(&self) -> u32 {
        let Some(claim) = self.scope_claim() else {
            return 0;
        };
        if !self.ledger(|ledger| ledger.claim_waiting()) {
            return 0;
        }
        self.held_pages(claim)
    }

    /// `(limit, charged)` when this thread's scope is a registered model's
    /// claim.
    pub(crate) fn scope_budget(&self) -> Option<(u64, u64)> {
        let claim = self.scope_claim()?;
        let stats = self.io.stats();
        let books = self.lock();
        let model = books.ledger.claim(claim)?.model;
        books
            .models
            .iter()
            .any(|entry| entry.id == model)
            .then(|| (books.limit(), books.charge(&stats).charged()))
    }

    #[cfg(test)]
    pub(crate) fn reserved_bytes(&self) -> u64 {
        self.lock().reserved
    }

    /// Seconds on the governor's clock.
    pub(crate) fn now(&self) -> f64 {
        self.io.now()
    }

    pub(crate) fn sleep(&self, seconds: f64) {
        self.io.sleep(seconds);
    }

    /// The need its scope's refusals recorded, with the caller's in-place
    /// part (Splash `MemoryGovernor::need`).
    pub(crate) fn need(&self, claim: ClaimId, in_place: Need) -> Need {
        let refusal = self.ledger(|ledger| {
            ledger
                .claim(claim)
                .map(|state| state.refusal)
                .unwrap_or_default()
        });
        Need {
            bytes: refusal.bytes,
            host_refused: refusal.host_refused,
            ..in_place
        }
    }

    /// Splash `MemoryGovernor::heldPages`: the blocks held against
    /// `claim`'s, beside the room admission grants in-service work now.
    pub(crate) fn held_pages(&self, claim: ClaimId) -> u32 {
        let stats = self.io.stats();
        let host = self.io.host();
        let books = self.lock();
        let system = host.pressure.unwrap_or(books.system_pressure);
        let host_room = match host.memory {
            Some(memory) if system == Pressure::Normal => {
                spare_headroom(&memory, books.spare_reserve, books.reserved)
                    .saturating_sub(HOST_MARGIN_BYTES)
            }
            _ => 0,
        };
        books
            .ledger
            .held_pages(claim, books.room(&stats).min(host_room))
    }

    /// Splash `MemoryGovernor::free`: an InService claim the host refused
    /// passes the host once nothing in place meets it.
    pub(crate) fn free(
        &self,
        claim: ClaimId,
        need: Need,
        sources: &mut [&mut dyn VictimSource],
    ) -> FreeResult {
        self.free_with(claim, need, None, sources)
    }

    fn free_with(
        &self,
        claim: ClaimId,
        need: Need,
        room: Option<u64>,
        sources: &mut [&mut dyn VictimSource],
    ) -> FreeResult {
        let now = self.io.now();
        let stats = self.io.stats();
        let (inputs, room, exempts, untracked) = {
            let mut books = self.lock();
            books.ledger.set_now(now);
            let inputs = books.ledger.free_inputs(claim, need, books.critical());
            let room = room.unwrap_or_else(|| books.room(&stats));
            (
                inputs,
                room,
                inputs.spec.cls == ClaimClass::InService,
                books.untracked(),
            )
        };
        let mut pool = MlxFreePool {
            io: &*self.io,
            untracked,
        };
        let decision = decide(&inputs, room, &mut pool, sources);
        self.lock().ledger.record_free(&inputs, decision, exempts)
    }

    /// Splash `MemoryGovernor::hostPass`: the host's claim of what the
    /// pass finds, up to `target`, asked with no room.
    fn host_pass(
        &self,
        target: u64,
        critical: bool,
        sources: &mut [&mut dyn VictimSource],
    ) -> FreeResult {
        let stats = self.io.stats();
        let spec = host_spec();
        let need = Need {
            bytes: target,
            ..Need::default()
        };
        let (inputs, charge, claim) = {
            let mut books = self.lock();
            let inputs = books.ledger.available_inputs(&spec, need, critical);
            let charge = books.charge(&stats);
            (inputs, charge, books.ledger.open(spec, 0))
        };
        let found = count_available(&inputs, 0, charge, sources);
        let mut result = self.free_with(
            claim,
            Need {
                bytes: target.min(found),
                ..Need::default()
            },
            Some(0),
            sources,
        );
        let mut books = self.lock();
        books.ledger.close(claim);
        let waits = take_rights(&spec, &need, critical).waits_for_writes;
        if result.outcome == FreeOutcome::Met && result.now_bytes == 0 && result.later_bytes == 0 {
            result.outcome = if waits && books.ledger.returning().landing_bytes > 0 {
                FreeOutcome::Pending
            } else {
                FreeOutcome::Short
            };
        }
        result
    }

    /// Splash `MemoryGovernor::controlPass` over `model`'s sources, once
    /// the host has settled from that model's last pass. Under critical
    /// pressure: one host claim of the margin plus what host-refused claims
    /// wait for, once per settle time. Otherwise, with a deficit, the host
    /// claim frees the lesser of the margin and the deficit.
    pub(crate) fn control_pass(
        &self,
        model: u64,
        sources: &mut [&mut dyn VictimSource],
    ) -> ControlPass {
        let now = self.io.now();
        let host = self.io.host();
        let (target, critical) = {
            let mut books = self.lock();
            books.ledger.set_now(now);
            books.sample_host_rule(&host);
            let critical = books.critical();
            let refused = books.ledger.host_refused_waiting_bytes();
            let Some(next) = books.model(model).map(|entry| entry.next_host_pass) else {
                return ControlPass { pass: None };
            };
            let due = now >= next;
            let mut target = None;
            if critical {
                if due {
                    target = Some(HOST_MARGIN_BYTES.saturating_add(refused));
                    if let Some(entry) = books.model(model) {
                        entry.next_host_pass = now + HOST_SETTLE_SECONDS;
                    }
                }
            } else if due {
                let landing = books.ledger.returning().landing_bytes;
                let deficit = books.host_rule.deficit_bytes(refused, landing);
                if deficit > 0 {
                    target = Some(HOST_MARGIN_BYTES.min(deficit));
                }
            }
            (target, critical)
        };
        let Some(target) = target else {
            return ControlPass { pass: None };
        };
        let result = self.host_pass(target, critical, sources);
        let mut books = self.lock();
        // Only a pass while the host is short can waive the band: one that
        // found nothing to take.
        if books.host_rule.constrained() {
            books.band_waived = result.outcome == FreeOutcome::Short;
        }
        if (result.now_bytes > 0
            || result.later_bytes > 0
            || result.outcome == FreeOutcome::Pending)
            && let Some(entry) = books.model(model)
        {
            entry.next_host_pass = now + HOST_SETTLE_SECONDS;
        }
        ControlPass {
            pass: Some(HostPass { target, result }),
        }
    }

    #[cfg(test)]
    /// What the host lacks of M and of what host-refused claims wait for,
    /// beyond what lands, at a fresh sample.
    pub(crate) fn deficit_bytes(&self) -> u64 {
        let host = self.io.host();
        let mut books = self.lock();
        books.sample_host_rule(&host);
        let refused = books.ledger.host_refused_waiting_bytes();
        let landing = books.ledger.returning().landing_bytes;
        books.host_rule.deficit_bytes(refused, landing)
    }

    /// Splash `MemoryGovernor::snapshot`: a fresh sample read without
    /// moving the band.
    pub(crate) fn snapshot(&self) -> GovernorSnapshot {
        let stats = self.io.stats();
        let host = self.io.host();
        let books = self.lock();
        let system = host.pressure.unwrap_or(books.system_pressure);
        let mut rule = books.host_rule;
        rule.sample(
            host.memory.as_ref(),
            books.spare_reserve,
            books.reserved,
            system,
            0,
            books.ledger.host_refused_waiting_bytes(),
            books.band_waived,
        );
        let limit = books.limit();
        let charged = books.charge(&stats).charged();
        let pressure = if rule.level() == Pressure::Critical {
            Pressure::Critical
        } else if rule.constrained() {
            Pressure::Warning
        } else {
            rule.level()
        };
        GovernorSnapshot {
            limit,
            charged,
            headroom: limit.saturating_sub(charged.saturating_add(books.reserved)),
            pressure,
            denied_reservations: books.denied,
            host: HostStatus {
                measured: rule.measured(),
                available: host.memory.map_or(0, |memory| memory.available),
                available_reserve: books.available_reserve,
                spare: host.memory.map_or(0, |memory| memory.spare),
                spare_reserve: books.spare_reserve,
                headroom: rule.headroom(),
                system_pressure: system,
                growth_allowed: rule.admits(ClaimClass::Growth, 0, false),
            },
            ledger: books.ledger.fill(charged, &[("free_pool", stats.cache)]),
        }
    }

    pub(crate) fn registered_models(&self) -> usize {
        self.registered.load(Ordering::Acquire)
    }

    /// Splash `MemoryLedger::poolReleases`.
    pub(crate) fn pool_releases(&self) -> u64 {
        self.lock().ledger.pool_releases()
    }

    fn metrics_lock(&self) -> MutexGuard<'_, Metrics> {
        self.metrics
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// A request's first tokens went out (Splash
    /// `EngineEventSink::firstTokenParts`).
    pub(crate) fn first_tokens(&self, parts: &TtftParts) {
        self.metrics_lock().first_tokens(parts);
    }

    pub(crate) fn decode_gap(&self, gap: &Gap) {
        self.metrics_lock().gap(gap);
    }

    pub(crate) fn resource_timed_out(&self) {
        let mut metrics = self.metrics_lock();
        metrics.resource_timeouts = metrics.resource_timeouts.saturating_add(1);
    }

    pub(crate) fn capacity_exhausted(&self) {
        let mut metrics = self.metrics_lock();
        metrics.capacity_exhausted = metrics.capacity_exhausted.saturating_add(1);
    }

    pub(crate) fn metrics(&self) -> Metrics {
        self.metrics_lock().clone()
    }

    /// The last decision-log lines, oldest first.
    pub(crate) fn decisions(&self) -> Vec<String> {
        self.decisions
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .cloned()
            .collect()
    }

    /// The model thread's accounts, read at the next fill.
    pub(crate) fn publish_accounts(&self, model: u64, accounts: MemoryAccounts) {
        let Some(published) = self
            .lock()
            .model(model)
            .map(|entry| Arc::clone(&entry.accounts))
        else {
            return;
        };
        published.kv.store(accounts.kv, Ordering::Relaxed);
        published.states.store(accounts.states, Ordering::Relaxed);
        published
            .embeddings
            .store(accounts.embeddings, Ordering::Relaxed);
    }

    /// The model thread's waits and replans, as its last pass left them.
    pub(crate) fn publish_waits(&self, model: u64, waits: ResourceWaitStatus, replans: u64) {
        if let Some(entry) = self.lock().model(model) {
            entry.waits = waits;
            entry.replans = replans;
        }
    }

    /// The registered models' waits, merged.
    pub(crate) fn resource_waits(&self) -> ResourceWaitStatus {
        let books = self.lock();
        let mut waits = ResourceWaitStatus::default();
        for entry in &books.models {
            waits.merge(&entry.waits);
        }
        waits
    }

    pub(crate) fn counters(&self) -> Counters {
        let mut counters = {
            let books = self.lock();
            books
                .models
                .iter()
                .fold(Counters::default(), |sum, entry| Counters {
                    replans: sum.replans + entry.replans,
                    resource_suspensions: sum.resource_suspensions + entry.suspensions,
                    resource_resumptions: sum.resource_resumptions + entry.resumptions,
                    metal_out_of_memory: sum.metal_out_of_memory + entry.metal_out_of_memory,
                    publications_refused: sum.publications_refused + entry.publications_refused,
                    ..sum
                })
        };
        let metrics = self.metrics_lock();
        counters.resource_timeouts = metrics.resource_timeouts;
        counters.capacity_exhausted = metrics.capacity_exhausted;
        counters
    }

    /// The model's wait epoch.
    pub(crate) fn epoch(&self, model: u64) -> u64 {
        self.lock().model(model).map_or(0, |entry| entry.epoch)
    }

    /// Splash `Engine::signalResourceProgress`.
    pub(crate) fn signal_progress(&self, model: u64) {
        if let Some(entry) = self.lock().model(model) {
            entry.epoch = entry.epoch.saturating_add(1);
        }
    }

    #[cfg(test)]
    pub(crate) fn allocation_failed(&self, model: u64) -> bool {
        self.lock()
            .model(model)
            .is_some_and(|entry| entry.allocation_failed)
    }

    pub(crate) fn note_allocation_failed(&self, model: u64) {
        if let Some(entry) = self.lock().model(model) {
            entry.allocation_failed = true;
        }
    }

    /// Splash `Engine::suspendForGrowth`, after the lane was vacated: the
    /// resident lanes drain for the wait limit before admission resumes,
    /// and a limit the host did not set counts as a failure the drain
    /// waits out. The epoch does not move.
    pub(crate) fn lane_suspended(&self, model: u64, now: f64, failure: Option<Failure>) {
        if let Some(entry) = self.lock().model(model) {
            entry.drain_end = now + super::wait::RESOURCE_WAIT_SECONDS;
            entry.allocation_failed = failure != Some(Failure::HostPressure);
            entry.suspensions = entry.suspensions.saturating_add(1);
        }
    }

    #[cfg(test)]
    /// Splash `resourceSuspensions` of the model.
    pub(crate) fn suspensions(&self, model: u64) -> u64 {
        self.lock()
            .model(model)
            .map_or(0, |entry| entry.suspensions)
    }

    /// A suspended lane resumed (Splash `resourceResumptions`).
    pub(crate) fn lane_resumed(&self, model: u64) {
        if let Some(entry) = self.lock().model(model) {
            entry.resumptions = entry.resumptions.saturating_add(1);
        }
    }

    #[cfg(test)]
    pub(crate) fn resumptions(&self, model: u64) -> u64 {
        self.lock()
            .model(model)
            .map_or(0, |entry| entry.resumptions)
    }

    /// Splash `Cache::setWantedChains`: the chains the model's requests
    /// without a lane restore once they start.
    pub(crate) fn set_wanted_chains(&self, model: u64, wanted: WantedChains) {
        if let Some(entry) = self.lock().model(model) {
            entry.wanted = Arc::new(wanted);
        }
    }

    pub(crate) fn wanted_chains(&self, model: u64) -> Arc<WantedChains> {
        self.lock()
            .model(model)
            .map_or_else(Arc::default, |entry| Arc::clone(&entry.wanted))
    }

    /// The drain ends once no lane is suspended.
    pub(crate) fn end_drain(&self, model: u64) {
        if let Some(entry) = self.lock().model(model) {
            entry.drain_end = 0.0;
        }
    }

    /// Splash `Engine::drainingForRecovery`: a drain set by a suspension
    /// lasts while `lanes` (some lane exists) and the failure stands or
    /// the host refuses growth; it ends at its time.
    pub(crate) fn draining_for_recovery(&self, model: u64, now: f64, lanes: bool) -> bool {
        let mut books = self.lock();
        let growth = books.host_rule.admits(ClaimClass::Growth, 0, false);
        let Some(entry) = books.model(model) else {
            return false;
        };
        if entry.drain_end > 0.0 && now >= entry.drain_end {
            entry.drain_end = 0.0;
        }
        entry.drain_end > 0.0 && lanes && (entry.allocation_failed || !growth)
    }

    /// A lane gave its memory back (Splash `Engine::release`): what failed
    /// for want of it may fit now.
    pub(crate) fn lane_released(&self, model: u64) {
        if let Some(entry) = self.lock().model(model) {
            entry.allocation_failed = false;
            entry.epoch = entry.epoch.saturating_add(1);
        }
    }

    pub(crate) fn is_registered(&self, model: u64) -> bool {
        model != 0 && self.lock().models.iter().any(|entry| entry.id == model)
    }

    /// The host is short or macOS warns: the model threads run their
    /// control pass (Splash `MemoryPressureMonitor`).
    pub(crate) fn needs_control(&self) -> bool {
        if self.registered.load(Ordering::Acquire) == 0 {
            return false;
        }
        let host = self.io.host();
        let mut books = self.lock();
        books.sample_host_rule(&host);
        let refused = books.ledger.host_refused_waiting_bytes();
        let landing = books.ledger.returning().landing_bytes;
        books.system_pressure != Pressure::Normal
            || books.host_rule.deficit_bytes(refused, landing) > 0
    }

    /// Whether `model`'s control pass is due: once per monitor period, or
    /// at once when `forced`. The pass's own host passes keep their settle
    /// time.
    pub(crate) fn control_due(&self, model: u64, forced: bool) -> bool {
        let now = self.io.now();
        let mut books = self.lock();
        let Some(entry) = books.model(model) else {
            return false;
        };
        if !forced && now < entry.next_control {
            return false;
        }
        entry.next_control = now + super::monitor::MONITOR_PERIOD_SECONDS;
        true
    }

    /// The last pass waits for what lands: the next one runs at once.
    pub(crate) fn control_again(&self, model: u64) {
        if let Some(entry) = self.lock().model(model) {
            entry.next_control = 0.0;
        }
    }

    pub(crate) fn publication_refused(&self, model: u64) {
        if let Some(entry) = self.lock().model(model) {
            entry.publications_refused = entry.publications_refused.saturating_add(1);
        }
    }

    #[cfg(test)]
    pub(crate) fn publications_refused(&self, model: u64) -> u64 {
        self.lock()
            .model(model)
            .map_or(0, |entry| entry.publications_refused)
    }

    /// A command of `model` exhausted Metal memory: its caches reset at the
    /// model thread's next pass.
    pub(crate) fn note_gpu_oom(&self, model: u64) {
        if let Some(entry) = self.lock().model(model) {
            entry.gpu_oom = true;
        }
    }

    /// Takes the note; a taken note counts one reset.
    pub(crate) fn take_gpu_oom(&self, model: u64) -> bool {
        let mut books = self.lock();
        let Some(entry) = books.model(model) else {
            return false;
        };
        let noted = std::mem::take(&mut entry.gpu_oom);
        if noted {
            entry.metal_out_of_memory = entry.metal_out_of_memory.saturating_add(1);
        }
        noted
    }

    #[cfg(test)]
    pub(crate) fn metal_out_of_memory(&self, model: u64) -> u64 {
        self.lock()
            .model(model)
            .map_or(0, |entry| entry.metal_out_of_memory)
    }
}

/// The stages a start reaches, in order (Splash `RuntimeResourceStage`).
const START_REGISTER: u8 = 1;
const START_RUNWAY: u8 = 2;

/// Splash `StartupRetryWindow`: a start the host or the Metal driver
/// refused is retried for 30 s from its first refusal, and again from a
/// refusal at a later stage; one the engine's budget refused fails at once.
#[derive(Default)]
pub(crate) struct StartupRetryWindow {
    deadline: Option<f64>,
    stage: u8,
    logged: bool,
}

impl StartupRetryWindow {
    /// Whether the start refused `refusal` at `now` is retried.
    pub(crate) fn retry(&mut self, refusal: Option<(Failure, u8)>, now: f64, text: &str) -> bool {
        let Some((failure, stage)) = refusal else {
            return false;
        };
        if !matches!(failure, Failure::HostPressure | Failure::DriverRejected) {
            return false;
        }
        if self.deadline.is_none() || stage > self.stage {
            self.deadline = Some(now + START_RETRY_SECONDS);
            self.stage = stage;
        }
        if self.deadline.is_some_and(|deadline| now >= deadline) {
            return false;
        }
        if !self.logged {
            warn!("Waiting for sufficient available memory to start: {text}");
            self.logged = true;
        }
        true
    }
}

/// A pool load's start (Splash's bootstrap retried by
/// `StartupRetryWindow`): each attempt runs inside the process-wide
/// pool-growth lock, and the wait between attempts outside it, so the
/// other models' pools grow and shrink meanwhile.
pub(crate) fn start_under_growth_lock<R>(
    governor: &Governor,
    mut attempt: impl FnMut() -> napi::Result<R>,
) -> napi::Result<R> {
    let mut window = StartupRetryWindow::default();
    loop {
        START_REFUSAL.with(|refusal| refusal.set(None));
        let result = {
            let _growth = crate::cache_limit::pool_growth_lock()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            attempt()
        };
        let error = match result {
            Ok(started) => return Ok(started),
            Err(error) => error,
        };
        let refusal = START_REFUSAL.with(Cell::take);
        if !window.retry(refusal, governor.now(), &error.reason) {
            return Err(error);
        }
        governor.sleep(START_POLL_SECONDS);
    }
}

impl crate::transformer::paged_kv_cache_adapter::PoolAdmission for &'static Governor {
    fn admit_grow(&self, bytes: u64, grow: &mut dyn FnMut() -> Result<bool, String>) -> bool {
        let reservation = match self.try_reserve(bytes) {
            Ok(reservation) => reservation,
            Err(Failure::OutsideScope) => {
                tracing::error!(
                    target: "mlx_core::memory",
                    "logic: memory was allocated outside a claim's scope ({} MiB)",
                    bytes / MIB
                );
                return false;
            }
            Err(_) => return false,
        };
        let grew = match grow() {
            Ok(grew) => grew,
            Err(error) => {
                warn!(target: "mlx_core::memory", "{error}");
                self.driver_rejected(bytes);
                false
            }
        };
        drop(reservation);
        grew
    }

    fn held_blocks(&self) -> u32 {
        Governor::held_blocks(self)
    }
}

#[cfg(test)]
pub(crate) mod tests;
