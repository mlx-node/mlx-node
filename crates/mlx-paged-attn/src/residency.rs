//! Wired residency between commands (Splash `metal::Residency`).
//!
//! One process-wide [`Clock`] stands for Splash's single `lastUse_`: every
//! residency (each paged pool's own `MTLResidencySet`, each model's MLX wired
//! lease) is held while the clock is, and lapses with it. A command marks the
//! clock; a beat thread requests residency every [`BEAT`] while the clock is
//! within its keep-alive and ends every residency once, then dispatches one
//! kick, when it lapses. The sets are never attached to a command queue or a
//! command buffer.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, Weak};
use std::time::{Duration, Instant};

/// Splash `Residency::kBeat`: one request holds a set only about two seconds.
pub const BEAT: Duration = Duration::from_millis(500);
/// Splash `kResidencyKeepAliveSeconds`.
pub const DEFAULT_KEEP_ALIVE_SECS: f64 = 600.0;
const KEEP_ALIVE_ERROR: &str = "residency keep-alive must be positive";

/// One residency the beat holds or ends.
pub trait Resident: Send + Sync {
    /// The clock is within its keep-alive: request residency again.
    fn hold(&self);
    /// The clock lapsed: end residency without GPU work.
    fn lapse(&self);
}

#[derive(Debug)]
struct ClockState {
    open: u32,
    last_use: Instant,
    held: bool,
}

/// Splash's one "last command" clock.
#[derive(Debug)]
pub struct Clock {
    /// `f64` seconds; `INFINITY` holds residency for the process's life.
    keep_alive: AtomicU64,
    state: Mutex<ClockState>,
}

fn validate_keep_alive(seconds: f64) -> Result<f64, String> {
    if seconds > 0.0 {
        Ok(seconds)
    } else {
        Err(KEEP_ALIVE_ERROR.to_string())
    }
}

impl Clock {
    pub fn new(keep_alive_secs: f64) -> Result<Arc<Self>, String> {
        let seconds = validate_keep_alive(keep_alive_secs)?;
        Ok(Arc::new(Self {
            keep_alive: AtomicU64::new(seconds.to_bits()),
            state: Mutex::new(ClockState {
                open: 0,
                last_use: Instant::now(),
                held: false,
            }),
        }))
    }

    fn state(&self) -> std::sync::MutexGuard<'_, ClockState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn set_keep_alive(&self, seconds: f64) -> Result<(), String> {
        let seconds = validate_keep_alive(seconds)?;
        self.keep_alive.store(seconds.to_bits(), Ordering::Release);
        Ok(())
    }

    /// `None` when residency is held for the process's life (`off`).
    pub fn keep_alive(&self) -> Option<Duration> {
        let seconds = f64::from_bits(self.keep_alive.load(Ordering::Acquire));
        Duration::try_from_secs_f64(seconds).ok()
    }

    /// Splash `Residency::use`: stamps the last command; true when the clock
    /// had lapsed, so the residencies must be requested again.
    pub fn mark(&self, now: Instant) -> bool {
        let mut state = self.state();
        state.last_use = state.last_use.max(now);
        !std::mem::replace(&mut state.held, true)
    }

    /// A command is running: the clock cannot lapse until [`Self::close`].
    pub fn open(&self) {
        self.state().open += 1;
    }

    pub fn close(&self, stamp: Option<Instant>) {
        let mut state = self.state();
        state.open = state.open.saturating_sub(1);
        if let Some(now) = stamp {
            state.last_use = state.last_use.max(now);
        }
    }

    pub fn held(&self) -> bool {
        self.state().held
    }

    pub fn last_use(&self) -> Instant {
        self.state().last_use
    }

    pub fn open_count(&self) -> u32 {
        self.state().open
    }

    /// Splash `Residency::beat`. `Some(true)` = still held, `Some(false)` =
    /// lapsed at this beat, `None` = already lapsed earlier.
    fn beat(&self, now: Instant) -> Option<bool> {
        let keep_alive = self.keep_alive();
        let mut state = self.state();
        if !state.held {
            return None;
        }
        let lapsed = state.open == 0
            && keep_alive.is_some_and(|k| now.saturating_duration_since(state.last_use) >= k);
        state.held = !lapsed;
        Some(!lapsed)
    }
}

/// One beat over `residents`: within the keep-alive each is held again; at
/// the lapse each ends and `kick` runs once. Returns whether the clock is
/// still held.
pub fn beat_with(
    clock: &Clock,
    residents: &[Arc<dyn Resident>],
    now: Instant,
    kick: impl FnOnce(),
) -> bool {
    match clock.beat(now) {
        Some(true) => {
            for resident in residents {
                resident.hold();
            }
            true
        }
        Some(false) => {
            for resident in residents {
                resident.lapse();
            }
            kick();
            false
        }
        None => false,
    }
}

/// The process's clock; `--idle-release` sets its keep-alive.
pub fn clock() -> &'static Arc<Clock> {
    static CLOCK: OnceLock<Arc<Clock>> = OnceLock::new();
    CLOCK.get_or_init(|| {
        Clock::new(DEFAULT_KEEP_ALIVE_SECS).unwrap_or_else(|_| unreachable!("positive default"))
    })
}

/// `seconds` must be positive; `f64::INFINITY` keeps residency for the
/// process's life.
pub fn set_keep_alive(seconds: f64) -> Result<(), String> {
    clock().set_keep_alive(seconds)
}

struct Beats {
    residents: Vec<Weak<dyn Resident>>,
    woken: bool,
    started: bool,
}

fn beats() -> &'static (Mutex<Beats>, Condvar) {
    static BEATS: OnceLock<(Mutex<Beats>, Condvar)> = OnceLock::new();
    BEATS.get_or_init(|| {
        (
            Mutex::new(Beats {
                residents: Vec::new(),
                woken: false,
                started: false,
            }),
            Condvar::new(),
        )
    })
}

/// Joins `resident` to the process clock's beat, starting the beat thread
/// once.
pub fn register(resident: Weak<dyn Resident>) {
    let (lock, condvar) = beats();
    let mut beats = lock.lock().unwrap_or_else(|e| e.into_inner());
    beats.residents.push(resident);
    if !beats.started {
        match std::thread::Builder::new()
            .name("mlx-residency".into())
            .spawn(run_beats)
        {
            Ok(_) => beats.started = true,
            Err(error) => {
                #[cfg(target_os = "macos")]
                tracing::warn!("residency beat thread not started: {error}");
                #[cfg(not(target_os = "macos"))]
                let _ = error;
            }
        }
    }
    beats.woken = true;
    condvar.notify_one();
}

/// Splash `Residency::use`. On the process clock a lapse ended, the beat
/// wakes and requests every residency again off the caller's thread.
pub fn mark(clock: &Arc<Clock>, now: Instant) {
    if clock.mark(now) && Arc::ptr_eq(clock, self::clock()) {
        let (lock, condvar) = beats();
        lock.lock().unwrap_or_else(|e| e.into_inner()).woken = true;
        condvar.notify_one();
    }
}

fn run_beats() {
    let (lock, condvar) = beats();
    loop {
        let residents: Vec<Arc<dyn Resident>> = {
            let mut beats = lock.lock().unwrap_or_else(|e| e.into_inner());
            beats.woken = false;
            beats.residents.retain(|weak| weak.strong_count() > 0);
            beats.residents.iter().filter_map(Weak::upgrade).collect()
        };
        let held = beat_with(clock(), &residents, Instant::now(), kick);
        drop(residents);
        let mut beats = lock.lock().unwrap_or_else(|e| e.into_inner());
        if held {
            beats = condvar
                .wait_timeout(beats, BEAT)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        } else {
            while !beats.woken {
                beats = condvar.wait(beats).unwrap_or_else(|e| e.into_inner());
            }
        }
        drop(beats);
    }
}

/// Bytes the paged pools' residency sets hold, process-wide.
static POOL_SET_BYTES: AtomicU64 = AtomicU64::new(0);

pub fn pool_set_bytes() -> u64 {
    POOL_SET_BYTES.load(Ordering::Acquire)
}

/// Splash `Residency::kickResidencyEnd`: Metal applies an `endResidency` at
/// its next GPU operation, so one 1-thread dispatch on a queue of its own.
pub fn kick() {
    #[cfg(target_os = "macos")]
    if let Err(error) = metal_kick::kick() {
        tracing::warn!("residency kick failed: {error}");
    }
}

#[cfg(target_os = "macos")]
mod metal_kick {
    use crate::metal::MetalState;
    use metal::{Buffer, CommandQueue, ComputePipelineState, MTLResourceOptions, MTLSize};
    use std::sync::OnceLock;

    struct Kick {
        pipeline: ComputePipelineState,
        queue: CommandQueue,
        target: Buffer,
    }

    fn build() -> Result<Kick, String> {
        let state = MetalState::get()?;
        let pipeline = state.get_pipeline("residency_kick")?;
        let queue = state
            .device
            .try_new_command_queue()
            .ok_or("unable to create the Metal residency set")?;
        let target = state
            .device
            .try_new_buffer(4, MTLResourceOptions::StorageModePrivate)
            .ok_or("unable to create the Metal residency set")?;
        Ok(Kick {
            pipeline,
            queue,
            target,
        })
    }

    fn get() -> Result<&'static Kick, String> {
        static KICK: OnceLock<Result<Kick, String>> = OnceLock::new();
        KICK.get_or_init(build).as_ref().map_err(Clone::clone)
    }

    /// Built with the first residency so a lapse compiles nothing.
    pub(super) fn prebuild() -> Result<(), String> {
        get().map(|_| ())
    }

    pub(super) fn kick() -> Result<(), String> {
        let kick = get()?;
        let command = kick
            .queue
            .try_new_command_buffer()
            .ok_or("Metal command queue returned no command buffer")?;
        let encoder = command
            .try_new_compute_command_encoder()
            .ok_or("Metal command buffer returned no compute encoder")?;
        encoder.set_compute_pipeline_state(&kick.pipeline);
        encoder.set_buffer(0, Some(&kick.target), 0);
        encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        encoder.end_encoding();
        command.commit();
        Ok(())
    }
}

#[cfg(target_os = "macos")]
pub use pool::PoolResidency;

#[cfg(target_os = "macos")]
mod pool {
    use super::{Clock, POOL_SET_BYTES, Resident};
    use metal::{BufferRef, Device, ResidencySet};
    use std::sync::atomic::Ordering;
    use std::sync::{Arc, Mutex};
    use std::time::Instant;

    struct State {
        set: ResidencySet,
        held: bool,
        requests: u64,
        members_bytes: u64,
    }

    /// One paged pool's own residency set (Splash's KV extents in
    /// `metal::Residency`). Members are the pool's live layer buffers.
    pub struct PoolResidency {
        clock: Arc<Clock>,
        /// Every `ResidencySet` call runs under this lock.
        state: Mutex<State>,
    }

    impl PoolResidency {
        pub(crate) fn new(device: &Device, clock: Arc<Clock>) -> Result<Arc<Self>, String> {
            let set = device.try_new_residency_set()?;
            super::metal_kick::prebuild()?;
            Ok(Arc::new(Self {
                clock,
                state: Mutex::new(State {
                    set,
                    held: false,
                    requests: 0,
                    members_bytes: 0,
                }),
            }))
        }

        fn state(&self) -> std::sync::MutexGuard<'_, State> {
            self.state.lock().unwrap_or_else(|e| e.into_inner())
        }

        fn resize(state: &mut State, bytes: u64) {
            let before = std::mem::replace(&mut state.members_bytes, bytes);
            if bytes >= before {
                POOL_SET_BYTES.fetch_add(bytes - before, Ordering::AcqRel);
            } else {
                POOL_SET_BYTES.fetch_sub(before - bytes, Ordering::AcqRel);
            }
        }

        /// Splash `Residency::add`: one commit, then `use()`.
        pub(crate) fn add_all(&self, buffers: &[&BufferRef]) {
            {
                let mut state = self.state();
                let mut bytes = state.members_bytes;
                for buffer in buffers {
                    state.set.add(buffer);
                    bytes += buffer.length();
                }
                state.set.commit();
                Self::resize(&mut state, bytes);
            }
            self.used();
        }

        /// A grow or shrink replaced one layer's planes: Splash `add` of the
        /// new buffers and `remove` of the old (their last pool reference).
        pub(crate) fn swap(&self, old: &[&BufferRef], new: &[&BufferRef]) {
            {
                let mut state = self.state();
                let mut bytes = state.members_bytes;
                for buffer in new {
                    state.set.add(buffer);
                    bytes += buffer.length();
                }
                for buffer in old {
                    state.set.remove(buffer);
                    bytes = bytes.saturating_sub(buffer.length());
                }
                state.set.commit();
                Self::resize(&mut state, bytes);
            }
            self.used();
        }

        fn used(&self) {
            super::mark(&self.clock, Instant::now());
        }

        pub fn allocation_count(&self) -> u64 {
            self.state().set.allocation_count()
        }

        pub fn contains(&self, buffer: &BufferRef) -> bool {
            self.state().set.contains(buffer)
        }

        pub fn members_bytes(&self) -> u64 {
            self.state().members_bytes
        }

        pub fn requests(&self) -> u64 {
            self.state().requests
        }

        pub fn held(&self) -> bool {
            self.state().held
        }
    }

    impl Resident for PoolResidency {
        fn hold(&self) {
            let mut state = self.state();
            state.set.request_residency();
            state.requests += 1;
            state.held = true;
        }

        fn lapse(&self) {
            let mut state = self.state();
            if std::mem::replace(&mut state.held, false) {
                state.set.end_residency();
            }
        }
    }

    impl Drop for PoolResidency {
        fn drop(&mut self) {
            let state = self.state.get_mut().unwrap_or_else(|e| e.into_inner());
            if state.held {
                state.set.end_residency();
            }
            let bytes = state.members_bytes;
            POOL_SET_BYTES.fetch_sub(bytes, Ordering::AcqRel);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    #[derive(Default)]
    struct Counting {
        holds: AtomicU32,
        lapses: AtomicU32,
    }

    impl Resident for Counting {
        fn hold(&self) {
            self.holds.fetch_add(1, Ordering::SeqCst);
        }
        fn lapse(&self) {
            self.lapses.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// Splash `buffersStayResident`: within the keep-alive every beat
    /// requests again; at the keep-alive the residency ends with one kick;
    /// the next command requests it again.
    ///
    /// Mutation M7: the beat never ends a residency.
    #[test]
    fn residency_lapses_once_and_a_command_requests_again() {
        let clock = Clock::new(2.0).unwrap();
        let counting = Arc::new(Counting::default());
        let residents: Vec<Arc<dyn Resident>> = vec![counting.clone()];
        let t = Instant::now();
        assert!(clock.mark(t));
        let mut kicks = 0;
        for half in 1..=3 {
            let now = t + Duration::from_millis(500 * half);
            assert!(beat_with(&clock, &residents, now, || kicks += 1));
        }
        assert_eq!(counting.holds.load(Ordering::SeqCst), 3);
        assert!(!beat_with(
            &clock,
            &residents,
            t + Duration::from_secs(2),
            || kicks += 1
        ));
        assert_eq!((counting.lapses.load(Ordering::SeqCst), kicks), (1, 1));
        assert!(!beat_with(
            &clock,
            &residents,
            t + Duration::from_secs(3),
            || kicks += 1
        ));
        assert_eq!((counting.lapses.load(Ordering::SeqCst), kicks), (1, 1));
        assert!(clock.mark(t + Duration::from_secs(3)));
        assert!(beat_with(
            &clock,
            &residents,
            t + Duration::from_millis(3500),
            || kicks += 1
        ));
        assert_eq!(counting.holds.load(Ordering::SeqCst), 4);
    }

    #[test]
    fn an_open_command_never_lapses() {
        let clock = Clock::new(2.0).unwrap();
        let residents: Vec<Arc<dyn Resident>> = vec![Arc::new(Counting::default())];
        let t = Instant::now();
        clock.mark(t);
        clock.open();
        assert!(beat_with(
            &clock,
            &residents,
            t + Duration::from_secs(1_000_000),
            || ()
        ));
        clock.close(Some(t + Duration::from_secs(1_000_000)));
        assert!(beat_with(
            &clock,
            &residents,
            t + Duration::from_secs(1_000_001),
            || ()
        ));
        assert!(!beat_with(
            &clock,
            &residents,
            t + Duration::from_secs(1_000_002),
            || ()
        ));
    }

    /// Splash `infiniteKeepAliveHoldsResidency` and the
    /// `MetalBackend` keep-alive check.
    #[test]
    fn infinite_keep_alive_holds_residency() {
        let clock = Clock::new(f64::INFINITY).unwrap();
        assert_eq!(clock.keep_alive(), None);
        let residents: Vec<Arc<dyn Resident>> = vec![Arc::new(Counting::default())];
        let t = Instant::now();
        clock.mark(t);
        assert!(beat_with(
            &clock,
            &residents,
            t + Duration::from_secs(1_000_000_000),
            || ()
        ));
        for bad in [0.0, -1.0, f64::NAN] {
            assert_eq!(
                clock.set_keep_alive(bad).unwrap_err(),
                "residency keep-alive must be positive"
            );
            assert_eq!(
                Clock::new(bad).unwrap_err(),
                "residency keep-alive must be positive"
            );
        }
        clock.set_keep_alive(30.0).unwrap();
        assert_eq!(clock.keep_alive(), Some(Duration::from_secs(30)));
    }
}
