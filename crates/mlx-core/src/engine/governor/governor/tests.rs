use std::sync::{Arc, Mutex};

use super::super::host::{HOST_MARGIN_BYTES, HostMemory, Pressure, spare_reserve_bytes};
use super::super::ledger::{
    Charge, ClaimClass, ClaimRank, ClaimSpec, FreeOutcome, FreePool, Need, NeedTiming, ReachClass,
    Returning, Room, Stage, Taken, UnitKey, Victim, VictimQuery, VictimSource,
};
use super::*;

pub(crate) const G: u64 = 1 << 30;
pub(crate) const PHYSICAL: u64 = 32 * G;

pub(crate) struct FakeState {
    pub stats: MemoryStats,
    pub host: HostSample,
    pub now: f64,
    pub sleeps: u32,
    pub mlx_limit: u64,
    pub limits: Vec<u64>,
    pub physical: u64,
    /// Runs at every sleep, after the clock moved.
    pub on_sleep: Option<Box<dyn FnMut(&mut FakeState) + Send>>,
    /// Runs once, at the next stats read, after the sample was taken.
    pub after_stats: Option<Box<dyn FnOnce(&mut FakeState) + Send>>,
}

/// Counters, host and clock set by the test; a clear empties the cache, a
/// sleep moves the clock.
pub(crate) struct FakeIo {
    pub state: Mutex<FakeState>,
}

impl FakeIo {
    pub(crate) fn new(stats: MemoryStats, host: HostSample) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(FakeState {
                stats,
                host,
                now: 0.0,
                sleeps: 0,
                mlx_limit: 50 * G,
                limits: Vec::new(),
                physical: PHYSICAL,
                on_sleep: None,
                after_stats: None,
            }),
        })
    }

    pub(crate) fn with<R>(&self, f: impl FnOnce(&mut FakeState) -> R) -> R {
        f(&mut self.state.lock().unwrap())
    }
}

impl GovernorIo for FakeIo {
    fn stats(&self) -> MemoryStats {
        self.with(|state| {
            let sample = state.stats;
            if let Some(hook) = state.after_stats.take() {
                hook(state);
            }
            sample
        })
    }

    fn host(&self) -> HostSample {
        self.with(|state| state.host)
    }

    fn now(&self) -> f64 {
        self.with(|state| state.now)
    }

    fn sleep(&self, seconds: f64) {
        self.with(|state| {
            state.sleeps += 1;
            state.now += seconds;
            if let Some(mut hook) = state.on_sleep.take() {
                hook(state);
                state.on_sleep = Some(hook);
            }
        });
    }

    fn clear_cache(&self) {
        self.with(|state| state.stats.cache = 0);
    }

    fn set_memory_limit(&self, bytes: u64) -> u64 {
        self.with(|state| {
            state.limits.push(bytes);
            std::mem::replace(&mut state.mlx_limit, bytes)
        })
    }

    fn physical_memory(&self) -> u64 {
        self.with(|state| state.physical)
    }
}

/// A measured host whose headroom above the spare reserve is `headroom`.
pub(crate) fn host(headroom: u64, pressure: Pressure) -> HostSample {
    let spare = spare_reserve_bytes(PHYSICAL) + headroom;
    HostSample {
        memory: Some(HostMemory {
            available: spare,
            spare,
        }),
        pressure: Some(pressure),
    }
}

pub(crate) fn stats(active: u64) -> MemoryStats {
    MemoryStats {
        active,
        pools: 0,
        cache: 0,
    }
}

pub(crate) fn model(hard_budget: u64, untracked: u64) -> ModelRegistration {
    ModelRegistration {
        model_id: 7,
        hard_budget,
        untracked,
        page_bytes: G / 4,
    }
}

/// A model registered on a governor of its own whose limit is 0: every
/// reservation inside its claims' scope is refused.
pub(crate) fn refusing_registration() -> ModelGuard<'static> {
    let io = FakeIo::new(stats(0), host(32 * G, Pressure::Normal));
    let governor: &'static Governor = Box::leak(Box::new(Governor::new(io)));
    governor.register_model(model(0, 0)).expect("register")
}

/// Records a refusal of one byte on the scope's claim, as a refused pool
/// grow does.
pub(crate) fn refuse_one_byte(guard: &ModelGuard<'static>) {
    assert_eq!(
        guard.governor().try_reserve(1).err(),
        Some(Failure::EngineBudget)
    );
}

pub(crate) fn lane(cls: ClaimClass, request_id: u64, completed_tokens: u64) -> ClaimSpec {
    ClaimSpec {
        cls,
        request_id,
        rank: ClaimRank {
            priority: 1,
            decoding: false,
            completed_tokens,
            earliness: u64::MAX - request_id,
        },
        reach: ReachClass::InUse,
        timing: NeedTiming::CanWait,
    }
}

/// FreeNow units in order; a take removes the head and frees its bytes.
pub(crate) struct Units {
    pub units: Vec<Victim>,
}

impl Units {
    pub(crate) fn new(count: u64, bytes: u64) -> Self {
        Self {
            units: (0..count)
                .map(|at| Victim {
                    key: UnitKey::Store(at),
                    stamp: at,
                    bytes,
                    pages: 0,
                    stage: Stage::FreeNow,
                })
                .collect(),
        }
    }
}

impl VictimSource for Units {
    fn units(&self, _query: &VictimQuery) -> Vec<Victim> {
        self.units.clone()
    }

    fn take(&mut self, _query: &VictimQuery, victim: &Victim) -> Taken {
        let at = self
            .units
            .iter()
            .position(|unit| unit.key == victim.key)
            .expect("taken unit is listed");
        let unit = self.units.remove(at);
        Taken {
            outcome: FreeOutcome::Met,
            now_bytes: unit.bytes,
            later_bytes: 0,
            pages: 0,
        }
    }
}

struct NoPool;

impl FreePool for NoPool {
    fn charge(&self) -> Charge {
        Charge::default()
    }

    fn clear(&mut self) {}
}

#[test]
fn charged_is_the_larger_of_reserve_and_free_pool() {
    let io = FakeIo::new(
        MemoryStats {
            active: 10,
            pools: 4,
            cache: 1,
        },
        host(4 * G, Pressure::Normal),
    );
    let governor = Governor::new(io.clone());
    let _model = governor.register_model(model(100, 3)).unwrap();
    // max(10 + 4 + 3, 10 + 4 + 1)
    assert_eq!(governor.snapshot().charged, 17);
    io.with(|state| state.stats.cache = 5);
    // max(17, 10 + 4 + 5)
    let snapshot = governor.snapshot();
    assert_eq!(snapshot.charged, 19);
    assert_eq!(snapshot.headroom, 81);
    assert_eq!(snapshot.limit, 100);
    assert_eq!(snapshot.pressure, Pressure::Normal);
    assert!(snapshot.host.measured && snapshot.host.growth_allowed);
    assert_eq!(snapshot.host.headroom, 4 * G);
    assert_eq!(snapshot.host.system_pressure, Pressure::Normal);
    assert_eq!(snapshot.host.available, snapshot.host.spare);
    assert_eq!(snapshot.host.available_reserve, 2 * G);
    assert_eq!(snapshot.host.spare_reserve, spare_reserve_bytes(PHYSICAL));
    assert_eq!(snapshot.ledger.accounts[0].name, "fixed");
    // The model publishes nothing yet: charged less the MLX free pool.
    assert_eq!(snapshot.ledger.accounts[0].bytes, 14);
    let free_pool = snapshot
        .ledger
        .accounts
        .iter()
        .find(|row| row.name == "free_pool");
    assert_eq!(free_pool.map(|row| row.bytes), Some(5));
}

#[test]
fn try_reserve_refuses_held_room_and_records_it() {
    let io = FakeIo::new(stats(15), host(4 * G, Pressure::Normal));
    let governor = Governor::new(io);
    let _model = governor.register_model(model(20, 0)).unwrap();
    // A waits, ranked above B, for 3 bytes on their way back.
    let a = governor.open(lane(ClaimClass::InService, 1, 1000), 7);
    let b = governor.open(lane(ClaimClass::InService, 2, 0), 7);
    governor.ledger(|ledger| {
        ledger.set_returning(
            Arc::new(|| Returning {
                landing_bytes: 3,
                ..Returning::default()
            }),
            Arc::new(|_| Returning::default()),
        );
        let held = ledger.free(
            a.id(),
            Need {
                bytes: 3,
                ..Need::default()
            },
            Room::default(),
            &mut NoPool,
            &mut [],
        );
        assert_eq!(held.outcome, FreeOutcome::Pending);
    });
    assert_eq!(governor.try_reserve(3).err(), Some(Failure::OutsideScope));
    {
        let _scope = Scope::enter(b.claim());
        // Room 5 fits 3, but 3 of it is held: 3 > 20 - 15 - 3.
        assert_eq!(governor.try_reserve(3).err(), Some(Failure::EngineBudget));
        assert_eq!(governor.snapshot().denied_reservations, 1);
        let refusal = governor.ledger(|ledger| ledger.claim(b.id()).unwrap().refusal);
        assert_eq!(
            refusal,
            Need {
                bytes: 3,
                host_refused: false,
                ..Need::default()
            }
        );
        assert_eq!(
            governor.need(
                b.id(),
                Need {
                    pages: 2,
                    ..Need::default()
                }
            ),
            Need {
                bytes: 3,
                pages: 2,
                ..Need::default()
            }
        );
        // 2 leaves the held 3.
        let reservation = governor
            .try_reserve(2)
            .expect("2 bytes fit beside the hold");
        assert_eq!(governor.snapshot().headroom, 3);
        drop(reservation);
        assert_eq!(governor.snapshot().headroom, 5);
    }
    // A's scope: nothing held above it.
    let _scope = Scope::enter(a.claim());
    assert!(governor.try_reserve(3).is_ok());
    assert_eq!(governor.scope_claim(), Some(a.id()));
}

#[test]
fn host_passes_wait_for_the_host_to_settle() {
    fn passes(governor: &Governor, io: &FakeIo, units: &mut Units) -> Vec<u64> {
        let mut targets = Vec::new();
        for now in [0.0, 0.5, 1.0] {
            io.with(|state| state.now = now);
            let pass = governor.control_pass(7, &mut [units]);
            assert!(!pass.again());
            if let Some(pass) = pass.pass {
                targets.push(pass.target);
            }
        }
        targets
    }

    // Critical: one host claim of M per settled second, whatever it found.
    let io = FakeIo::new(stats(0), host(4 * G, Pressure::Critical));
    let governor = Governor::new(io.clone());
    let _model = governor.register_model(model(20 * G, 0)).unwrap();
    let mut none = Units::new(0, 0);
    assert_eq!(
        passes(&governor, &io, &mut none),
        vec![HOST_MARGIN_BYTES, HOST_MARGIN_BYTES]
    );
    assert_eq!(governor.snapshot().pressure, Pressure::Critical);

    // A deficit of 0.5 GiB; each pass frees it.
    let io = FakeIo::new(stats(0), host(G / 2, Pressure::Normal));
    let governor = Governor::new(io.clone());
    let _model = governor.register_model(model(20 * G, 0)).unwrap();
    let mut units = Units::new(8, G / 4);
    assert_eq!(governor.deficit_bytes(), G / 2);
    assert_eq!(passes(&governor, &io, &mut units), vec![G / 2, G / 2]);
    assert_eq!(units.units.len(), 4);
    assert_eq!(governor.snapshot().pressure, Pressure::Warning);

    // Unmeasured: no deficit, no pass.
    let io = FakeIo::new(stats(0), host(4 * G, Pressure::Normal));
    let governor = Governor::new(io.clone());
    let _model = governor.register_model(model(20 * G, 0)).unwrap();
    io.with(|state| state.host.memory = None);
    let mut units = Units::new(8, G / 4);
    assert!(passes(&governor, &io, &mut units).is_empty());
    assert_eq!(units.units.len(), 8);
}

/// Splash `MemoryGovernor::tryReserve` reads charged and reserved at one
/// instant. Here another model thread's allocation lands and its 80-byte
/// reservation drops between B's sample and B's lock: B must sample again
/// and see it. Mutation E1m (no resample) -> red (B is admitted: 160 > 100).
#[test]
fn a_reservation_retired_after_the_sample_is_still_charged() {
    let io = FakeIo::new(stats(0), host(4 * G, Pressure::Normal));
    let governor: &'static Governor = Box::leak(Box::new(Governor::new(io.clone())));
    let _model = governor.register_model(model(100, 0)).unwrap();
    let a = governor.open(lane(ClaimClass::InService, 1, 0), 7);
    let b = governor.open(lane(ClaimClass::InService, 2, 0), 7);
    let held = {
        let _scope = Scope::enter(a.claim());
        governor.try_reserve(80).expect("80 of 100")
    };
    let held = Mutex::new(Some(held));
    let held: &'static Mutex<Option<Reservation<'static>>> = Box::leak(Box::new(held));
    io.with(|state| {
        state.after_stats = Some(Box::new(move |state| {
            state.stats.active += 80;
            drop(held.lock().unwrap().take());
        }));
    });
    let _scope = Scope::enter(b.claim());
    assert_eq!(governor.try_reserve(80).err(), Some(Failure::EngineBudget));
    assert!(held.lock().unwrap().is_none());
    assert_eq!(governor.reserved_bytes(), 0);
    assert!(governor.try_reserve(20).is_ok());
}

/// Splash runs one engine per governor, so its host deadline and the units
/// a pass sees share one scope. Two models here: A holds nothing and runs
/// first at every tick; B holds 2 GiB. Under critical pressure B's own
/// passes still take a unit each settled second.
/// Mutation E3m (one deadline for every model) -> red (B never passes).
#[test]
fn an_empty_model_does_not_spend_another_models_host_pass() {
    let io = FakeIo::new(stats(0), host(4 * G, Pressure::Critical));
    let governor = Governor::new(io.clone());
    let _a = governor.register_model(model(20 * G, 0)).unwrap();
    let _b = governor
        .register_model(ModelRegistration {
            model_id: 8,
            ..model(20 * G, 0)
        })
        .unwrap();
    let mut empty = Units::new(0, 0);
    let mut full = Units::new(8, G / 4);
    let mut b_passes = 0;
    for now in [0.0, 0.5, 1.0, 1.5] {
        io.with(|state| state.now = now);
        assert!(governor.control_pass(7, &mut [&mut empty]).pass.is_some() == (now % 1.0 == 0.0));
        if governor.control_pass(8, &mut [&mut full]).pass.is_some() {
            b_passes += 1;
        }
    }
    assert_eq!(b_passes, 2);
    assert!(full.units.len() < 8, "B's units were taken");
}

#[test]
fn memory_limit_follows_registered_pools() {
    let io = FakeIo::new(
        MemoryStats {
            active: 0,
            pools: 4,
            cache: 0,
        },
        host(4 * G, Pressure::Normal),
    );
    let governor = Governor::new(io.clone());
    governor.pools_changed();
    assert!(
        io.with(|state| state.limits.is_empty()),
        "no model, no limit"
    );
    let guard = governor.register_model(model(20, 0)).unwrap();
    assert_eq!(io.with(|state| state.mlx_limit), 16);
    io.with(|state| state.stats.pools = 6);
    governor.pools_changed();
    assert_eq!(io.with(|state| state.mlx_limit), 14);
    drop(guard);
    assert_eq!(io.with(|state| state.mlx_limit), 50 * G);

    // The start gate: existing allocations beyond the limit.
    io.with(|state| state.stats.active = 30);
    let refused = governor.register_model(model(20, 0)).err().unwrap();
    assert!(
        refused.starts_with("memory plan: existing Metal allocations exceed memory governor limit"),
        "{refused}"
    );
    // macOS short of memory.
    io.with(|state| {
        state.stats.active = 0;
        state.host.memory = Some(HostMemory {
            available: G,
            spare: G,
        });
    });
    let refused = governor.register_model(model(20, 0)).err().unwrap();
    assert!(
        refused.starts_with(
            "memory plan: macOS is short of memory (1024 MiB available, 2048 MiB kept for macOS)"
        ),
        "{refused}"
    );
}

/// A pool load through [`start_under_growth_lock`], refused or not.
fn start_pool(governor: &Governor) -> napi::Result<()> {
    start_under_growth_lock(governor, || {
        governor
            .load_pool(model(20 * G, 0), G, || Ok(()))
            .map(drop)
            .map_err(napi::Error::from_reason)
    })
}

/// Splash `StartupRetryWindow::retryUntil`: a start the host refuses, at
/// its registration or at its runway, is retried every second for 30 s
/// with the pool-growth lock released between attempts, so the other
/// models' pools grow meanwhile; existing allocations beyond the limit
/// (the engine's budget) fail at once.
/// Mutation SR1 (every refusal retried) → red (the budget refusal waits
/// 30 s); SR2 (the wait holds the growth lock) → red.
#[test]
fn start_retries_host_refusals_outside_the_growth_lock() {
    let io = FakeIo::new(stats(0), host(4 * G, Pressure::Normal));
    let lock_free = Arc::new(Mutex::new(Vec::new()));
    {
        let lock_free = Arc::clone(&lock_free);
        io.with(|state| {
            state.on_sleep = Some(Box::new(move |_| {
                lock_free
                    .lock()
                    .unwrap()
                    .push(crate::cache_limit::pool_growth_lock().try_lock().is_ok());
            }));
        });
    }
    let governor = Governor::new(io.clone());

    // The engine's budget: no retry.
    io.with(|state| state.stats.active = 30 * G);
    let refused = start_pool(&governor).expect_err("over the limit");
    assert!(
        refused
            .reason
            .starts_with("memory plan: existing Metal allocations exceed memory governor limit"),
        "{}",
        refused.reason
    );
    assert_eq!(io.with(|state| state.sleeps), 0);

    // macOS short of memory at registration: 30 s of retries.
    io.with(|state| {
        state.stats.active = 0;
        state.host.memory = Some(HostMemory {
            available: G,
            spare: G,
        });
    });
    let refused = start_pool(&governor).expect_err("macOS stays short");
    assert!(
        refused
            .reason
            .starts_with("memory plan: macOS is short of memory"),
        "{}",
        refused.reason
    );
    assert_eq!(io.with(|state| state.sleeps), 30);
    assert!(lock_free.lock().unwrap().iter().all(|free| *free));

    // The runway the host refuses under critical pressure, which lifts at
    // the third retry.
    io.with(|state| {
        state.sleeps = 0;
        state.host = host(4 * G, Pressure::Critical);
        state.on_sleep = Some(Box::new(|state| {
            if state.sleeps == 3 {
                state.host = host(4 * G, Pressure::Normal);
            }
        }));
    });
    start_pool(&governor).expect("started once the host recovered");
    assert_eq!(io.with(|state| state.sleeps), 3);
}

/// Splash `MemoryGovernor`'s constructor check: on a 16 GiB machine macOS
/// keeps min(1/10, 2 GiB) = 1.6 GiB available; with 1 GiB available the
/// load is refused with the text the TS layer reads as a memory refusal.
/// Mutation G2 (no start gate) → red (the model registers).
#[test]
fn load_refused_when_macos_is_short() {
    let io = FakeIo::new(stats(0), host(4 * G, Pressure::Normal));
    io.with(|state| {
        state.physical = 16 * G;
        state.host.memory = Some(HostMemory {
            available: G,
            spare: G,
        });
    });
    let governor = Governor::new(io.clone());
    let refused = governor.register_model(model(8 * G, 0)).err();
    assert_eq!(
        refused.as_deref(),
        Some(
            "memory plan: macOS is short of memory (1024 MiB available, 1638 MiB kept for macOS); \
             close memory-heavy applications"
        )
    );
    io.with(|state| {
        state.host.memory = Some(HostMemory {
            available: 2 * G,
            spare: 2 * G,
        })
    });
    assert!(governor.register_model(model(8 * G, 0)).is_ok());
}
