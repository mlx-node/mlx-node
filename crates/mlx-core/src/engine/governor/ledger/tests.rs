use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use super::*;

const G: u64 = 1 << 30;
const M: u64 = 7;

type TakeLog = Rc<RefCell<Vec<UnitKey>>>;

/// Units in their take order; a take removes the unit, except keys in
/// `nothing`, whose take frees nothing.
struct FakeSource {
    units: Vec<Victim>,
    nothing: HashSet<UnitKey>,
    log: TakeLog,
    takes: usize,
}

impl FakeSource {
    fn new(units: Vec<Victim>, log: &TakeLog) -> Self {
        Self {
            units,
            nothing: HashSet::new(),
            log: log.clone(),
            takes: 0,
        }
    }
}

impl VictimSource for FakeSource {
    fn units(&self, _query: &VictimQuery) -> Vec<Victim> {
        self.units.clone()
    }

    fn take(&mut self, _query: &VictimQuery, victim: &Victim) -> Taken {
        self.takes += 1;
        assert!(self.takes <= 64, "step cap: the take loop does not end");
        self.log.borrow_mut().push(victim.key.clone());
        if self.nothing.contains(&victim.key) {
            return Taken::default();
        }
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
            pages: unit.pages,
        }
    }
}

/// Charged as the governor charges it; a clear empties the cache.
struct FakePool {
    charge: Charge,
    clears: usize,
}

impl FakePool {
    fn new(allocated: u64, untracked: u64, cache: u64) -> Self {
        Self {
            charge: Charge {
                allocated,
                untracked,
                cache,
            },
            clears: 0,
        }
    }

    fn empty() -> Self {
        Self::new(0, 0, 0)
    }
}

impl FreePool for FakePool {
    fn charge(&self) -> Charge {
        self.charge
    }

    fn clear(&mut self) {
        self.clears += 1;
        self.charge.cache = 0;
    }
}

fn unit(key: UnitKey, stamp: u64, bytes: u64, pages: u32, stage: Stage) -> Victim {
    Victim {
        key,
        stamp,
        bytes,
        pages,
        stage,
    }
}

fn rung() -> UnitKey {
    UnitKey::Ladder {
        owner_id: "turn".to_string(),
        prefix_len: 64,
        final_block_hash: 9,
    }
}

fn rank(decoding: bool, completed_tokens: u64) -> ClaimRank {
    ClaimRank {
        priority: 1,
        decoding,
        completed_tokens,
        earliness: u64::MAX - completed_tokens,
    }
}

fn lane(cls: ClaimClass, request_id: u64, rank: ClaimRank) -> ClaimSpec {
    ClaimSpec {
        cls,
        request_id,
        rank,
        reach: ReachClass::InUse,
        timing: NeedTiming::CanWait,
    }
}

fn room(bytes: u64) -> Room {
    Room {
        bytes,
        critical: false,
        exempts: false,
    }
}

fn bytes(bytes: u64) -> Need {
    Need {
        bytes,
        ..Need::default()
    }
}

fn pages(pages: u32) -> Need {
    Need {
        pages,
        ..Need::default()
    }
}

/// Splash `takeRights` transcribed per class (no request priority, no
/// pass command, conversion, store pools or lane-start buffers).
fn splash_rights(
    cls: ClaimClass,
    timing: NeedTiming,
    reach: ReachClass,
    critical: bool,
    host_refused: bool,
) -> TakeRights {
    let can_wait = timing == NeedTiming::CanWait && !host_refused;
    let reserves = if critical {
        Reserves::All
    } else {
        match cls {
            ClaimClass::InService | ClaimClass::Start => Reserves::NextExtent,
            _ => Reserves::None,
        }
    };
    match cls {
        ClaimClass::InService => TakeRights {
            units: true,
            waits_for_writes: can_wait,
            waits_for_landings: can_wait,
            in_use: can_wait && reach == ReachClass::InUse,
            certain_reuse: CertainReuse::All,
            reserves,
            rebuilt_pools: true,
        },
        ClaimClass::Start | ClaimClass::Optional => TakeRights {
            units: true,
            waits_for_writes: can_wait,
            waits_for_landings: can_wait,
            in_use: false,
            certain_reuse: CertainReuse::ResumePoint,
            reserves,
            rebuilt_pools: true,
        },
        ClaimClass::Growth => TakeRights {
            units: true,
            waits_for_writes: can_wait,
            waits_for_landings: can_wait,
            in_use: false,
            certain_reuse: CertainReuse::None,
            reserves,
            rebuilt_pools: false,
        },
        ClaimClass::Host => {
            let waits = !critical && !host_refused;
            TakeRights {
                units: true,
                waits_for_writes: waits,
                waits_for_landings: false,
                in_use: critical,
                certain_reuse: if critical {
                    CertainReuse::All
                } else {
                    CertainReuse::None
                },
                reserves,
                rebuilt_pools: true,
            }
        }
        ClaimClass::Floor => unreachable!(),
    }
}

#[test]
fn rights_follow_the_splash_table() {
    let classes = [
        ClaimClass::InService,
        ClaimClass::Start,
        ClaimClass::Optional,
        ClaimClass::Growth,
        ClaimClass::Host,
    ];
    for cls in classes {
        for timing in [NeedTiming::CanWait, NeedTiming::Immediate] {
            for reach in [ReachClass::Ordinary, ReachClass::InUse] {
                for critical in [false, true] {
                    for host_refused in [false, true] {
                        let spec = ClaimSpec {
                            cls,
                            request_id: 9,
                            rank: ClaimRank::default(),
                            reach,
                            timing,
                        };
                        let need = Need {
                            bytes: G,
                            host_refused,
                            ..Need::default()
                        };
                        assert_eq!(
                            take_rights(&spec, &need, critical),
                            splash_rights(cls, timing, reach, critical, host_refused),
                            "{cls:?} {timing:?} {reach:?} critical={critical} host_refused={host_refused}"
                        );
                    }
                }
            }
        }
    }
    let own = ClaimSpec {
        cls: ClaimClass::Optional,
        request_id: 0,
        rank: ClaimRank::default(),
        reach: ReachClass::InUse,
        timing: NeedTiming::CanWait,
    };
    assert_eq!(take_rights(&own, &bytes(G), true), TakeRights::default());
    let floor = ClaimSpec {
        cls: ClaimClass::Floor,
        ..own
    };
    assert!(std::panic::catch_unwind(|| take_rights(&floor, &bytes(G), false)).is_err());
}

#[test]
fn order_takes_free_now_then_rebuild_then_in_use() {
    let log = TakeLog::default();
    let mut model = FakeSource::new(
        vec![
            unit(UnitKey::Store(1), 5, G, 0, Stage::FreeNow),
            unit(rung(), 1, G, 0, Stage::InUse),
        ],
        &log,
    );
    let mut owners = FakeSource::new(vec![unit(UnitKey::Owner(2), 3, G, 0, Stage::FreeNow)], &log);
    let mut vision = FakeSource::new(
        vec![unit(UnitKey::Vision(4), 0, G, 0, Stage::Rebuild)],
        &log,
    );
    let mut ledger = Ledger::new();
    let claim = ledger.open(lane(ClaimClass::InService, 1, rank(true, 10)), M);
    let result = ledger.free(
        claim,
        bytes(4 * G),
        room(0),
        &mut FakePool::empty(),
        &mut [&mut model, &mut owners, &mut vision],
    );
    assert_eq!(result.outcome, FreeOutcome::Met);
    assert_eq!(
        *log.borrow(),
        vec![
            UnitKey::Owner(2),
            UnitKey::Store(1),
            UnitKey::Vision(4),
            rung()
        ]
    );

    // A stamp tie goes to the state (the model's source is listed first).
    let log = TakeLog::default();
    let mut model = FakeSource::new(vec![unit(UnitKey::History, 3, G, 0, Stage::FreeNow)], &log);
    let mut owners = FakeSource::new(vec![unit(UnitKey::Owner(2), 3, G, 0, Stage::FreeNow)], &log);
    let result = ledger.free(
        claim,
        bytes(G),
        room(0),
        &mut FakePool::empty(),
        &mut [&mut model, &mut owners],
    );
    assert_eq!(result.outcome, FreeOutcome::Met);
    assert_eq!(*log.borrow(), vec![UnitKey::History]);
}

#[test]
fn nothing_is_taken_unless_the_count_covers() {
    let log = TakeLog::default();
    let mut model = FakeSource::new(
        vec![
            unit(UnitKey::Store(1), 1, G, 3, Stage::FreeNow),
            unit(UnitKey::Store(2), 2, G, 4, Stage::FreeNow),
        ],
        &log,
    );
    let mut pool = FakePool::new(10 * G, 0, G);
    let mut ledger = Ledger::new();
    let claim = ledger.open(lane(ClaimClass::InService, 1, rank(true, 10)), M);
    let result = ledger.free(claim, pages(10), room(0), &mut pool, &mut [&mut model]);
    assert_eq!(result.outcome, FreeOutcome::Short);
    assert!(log.borrow().is_empty(), "took {:?}", log.borrow());
    assert_eq!(result.now_bytes, G);
}

#[test]
fn take_stops_once_covered() {
    let log = TakeLog::default();
    let mut model = FakeSource::new(
        vec![
            unit(UnitKey::Store(1), 1, G, 4, Stage::FreeNow),
            unit(UnitKey::Store(2), 2, G, 5, Stage::FreeNow),
        ],
        &log,
    );
    let mut ledger = Ledger::new();
    let claim = ledger.open(lane(ClaimClass::InService, 1, rank(true, 10)), M);
    let result = ledger.free(
        claim,
        pages(3),
        room(0),
        &mut FakePool::empty(),
        &mut [&mut model],
    );
    assert_eq!(result.outcome, FreeOutcome::Met);
    assert_eq!(*log.borrow(), vec![UnitKey::Store(1)]);
    assert_eq!(result.pages, 4);
}

#[test]
fn unsized_pool_goes_before_the_count() {
    // (a) the cache is above the untracked reserve: clearing it lowers
    // charged by cache - untracked, which meets the need.
    let log = TakeLog::default();
    let mut model = FakeSource::new(vec![unit(UnitKey::Store(1), 1, G, 0, Stage::FreeNow)], &log);
    let mut pool = FakePool::new(10 * G, G / 2, 2 * G);
    let mut ledger = Ledger::new();
    let spec = lane(ClaimClass::InService, 1, rank(true, 10));
    let claim = ledger.open(spec, M);
    assert_eq!(
        ledger.available(&spec, bytes(2 * G), room(0), pool.charge, &[&mut model]),
        2 * G
    );
    assert_eq!(
        ledger.available(&spec, bytes(3 * G), room(0), pool.charge, &[]),
        3 * G / 2
    );
    assert_eq!(pool.clears, 0);
    let result = ledger.free(claim, bytes(G), room(0), &mut pool, &mut [&mut model]);
    assert_eq!(result.outcome, FreeOutcome::Met);
    assert_eq!(result.now_bytes, 3 * G / 2);
    assert!(log.borrow().is_empty(), "took {:?}", log.borrow());
    assert_eq!(pool.clears, 1);

    // (b) the cache is within the reserve: clearing it credits nothing.
    let log = TakeLog::default();
    let mut model = FakeSource::new(vec![unit(UnitKey::Store(1), 1, G, 0, Stage::FreeNow)], &log);
    let mut pool = FakePool::new(10 * G, 3 * G / 2, G);
    assert_eq!(
        ledger.available(&spec, bytes(G / 2), room(0), pool.charge, &[]),
        0
    );
    let result = ledger.free(claim, bytes(G / 2), room(0), &mut pool, &mut [&mut model]);
    assert_eq!(result.outcome, FreeOutcome::Met);
    assert_eq!(*log.borrow(), vec![UnitKey::Store(1)]);
    let mut pool = FakePool::new(10 * G, 3 * G / 2, G);
    let result = ledger.free(claim, bytes(G / 2), room(0), &mut pool, &mut []);
    assert_eq!(result.outcome, FreeOutcome::Short);
    assert_eq!(result.now_bytes, 0);
}

/// Claim A (decoding) waits on 2 GiB landing and holds its 1 GiB need.
fn ledger_with_a_waiting(landing: &Arc<Mutex<Returning>>) -> (Ledger, ClaimId) {
    let mut ledger = Ledger::new();
    ledger.set_now(1.0);
    ledger.set_page_bytes(M, G / 4);
    let all = landing.clone();
    ledger.set_returning(
        Arc::new(move || *all.lock().unwrap()),
        Arc::new(|_| Returning::default()),
    );
    *landing.lock().unwrap() = Returning {
        landing_bytes: 2 * G,
        ..Returning::default()
    };
    let a = ledger.open(lane(ClaimClass::InService, 1, rank(true, 900)), M);
    let held = Need {
        bytes: G,
        pages: 3,
        ..Need::default()
    };
    let result = ledger.free(a, held, room(0), &mut FakePool::empty(), &mut []);
    assert_eq!(result.outcome, FreeOutcome::Pending);
    assert!(ledger.claim(a).unwrap().waiting);
    *landing.lock().unwrap() = Returning::default();
    (ledger, a)
}

#[test]
fn a_waiting_claim_holds_its_need_from_lower_ranks() {
    let landing = Arc::new(Mutex::new(Returning::default()));
    let (mut ledger, a) = ledger_with_a_waiting(&landing);
    let b = ledger.open(lane(ClaimClass::InService, 2, rank(false, 300)), M);
    let c = ledger.open(lane(ClaimClass::InService, 3, rank(false, 0)), M);
    // C decodes past A.
    ledger.update(c, lane(ClaimClass::InService, 3, rank(true, 1000)));
    assert_eq!(ledger.held_bytes(b), G);
    let result = ledger.free(
        b,
        bytes(G),
        room(3 * G / 2),
        &mut FakePool::empty(),
        &mut [],
    );
    assert_eq!(result.outcome, FreeOutcome::Pending);
    let result = ledger.free(
        c,
        bytes(G),
        room(3 * G / 2),
        &mut FakePool::empty(),
        &mut [],
    );
    assert_eq!(result.outcome, FreeOutcome::Met);
    // A's 3 blocks plus (1 GiB held - 0.5 GiB room) in 256 MiB blocks.
    assert_eq!(ledger.held_pages(b, G / 2), 5);
    assert_eq!(ledger.held_pages(a, 0), 0);

    ledger.set_now(3.0);
    let snapshot = ledger.fill(0, &[]);
    assert_eq!((snapshot.claims_waiting, snapshot.claims_bytes), (2, 2 * G));
    assert_eq!(snapshot.oldest_waiting_ms, 2000.0);
    // A's hold ends with its claim.
    ledger.close(a);
    ledger.set_waiting(b, false);
    assert!(!ledger.claim_waiting());
    assert_eq!(ledger.held_bytes(b), 0);
    assert_eq!(ledger.claims().count(), 2);
}

#[test]
fn short_only_by_held_room_waits() {
    let landing = Arc::new(Mutex::new(Returning::default()));
    let (mut ledger, _a) = ledger_with_a_waiting(&landing);
    let log = TakeLog::default();
    let mut model = FakeSource::new(
        vec![unit(UnitKey::Store(1), 1, G / 2, 0, Stage::FreeNow)],
        &log,
    );
    let b = ledger.open(lane(ClaimClass::InService, 2, rank(false, 300)), M);
    // Lacks 1 GiB + 1 GiB held - 0.75 GiB room; the unit gives 0.5 GiB.
    let result = ledger.free(
        b,
        bytes(G),
        room(3 * G / 4),
        &mut FakePool::empty(),
        &mut [&mut model],
    );
    assert_eq!(result.outcome, FreeOutcome::Pending);
    assert!(log.borrow().is_empty(), "took {:?}", log.borrow());
    assert!(ledger.claim(b).unwrap().waiting);
}

#[test]
fn accounts_sum_to_charged() {
    let mut ledger = Ledger::new();
    ledger.add_account(M, "kv", Arc::new(|| 4));
    ledger.add_account(M, "states", Arc::new(|| 2));
    ledger.add_account(M, "embeddings", Arc::new(|| 1));
    ledger.add_account(M, "free_pool", Arc::new(|| 3));
    ledger.add_account(M + 1, "states", Arc::new(|| 0));
    for (charged, fixed) in [(15, 5), (20, 10)] {
        let snapshot = ledger.fill(charged, &[]);
        let names: Vec<&str> = snapshot
            .accounts
            .iter()
            .map(|row| row.name.as_str())
            .collect();
        assert_eq!(names, ["fixed", "kv", "states", "embeddings", "free_pool"]);
        assert_eq!(snapshot.accounts[0].bytes, fixed);
        assert_eq!(
            snapshot.accounts.iter().map(|row| row.bytes).sum::<u64>(),
            charged
        );
    }
    ledger.remove_model(M);
    assert_eq!(ledger.fill(15, &[]).accounts[0].bytes, 15);
}

#[test]
fn host_refused_lane_is_met_past_the_host() {
    let log = TakeLog::default();
    let mut model = FakeSource::new(vec![unit(UnitKey::Owner(1), 1, 0, 2, Stage::FreeNow)], &log);
    let refused = Need {
        bytes: G,
        host_refused: true,
        ..Need::default()
    };
    let mut ledger = Ledger::new();
    let lane_claim = ledger.open(lane(ClaimClass::InService, 1, rank(true, 10)), M);
    let at = Room {
        bytes: 0,
        critical: false,
        exempts: true,
    };
    let result = ledger.free(
        lane_claim,
        refused,
        at,
        &mut FakePool::empty(),
        &mut [&mut model],
    );
    assert_eq!(result.outcome, FreeOutcome::Met);
    assert!(ledger.claim(lane_claim).unwrap().host_exempt);

    let start = ledger.open(lane(ClaimClass::Start, 2, rank(false, 0)), M);
    let result = ledger.free(
        start,
        refused,
        room(0),
        &mut FakePool::empty(),
        &mut [&mut model],
    );
    assert_eq!(result.outcome, FreeOutcome::Short);
    assert!(!ledger.claim(start).unwrap().host_exempt);
    assert!(log.borrow().is_empty());
    ledger.refused(start, refused);
    ledger.set_host_refused_waiting(start, true);
    assert_eq!(ledger.host_refused_waiting_bytes(), G);
    ledger.set_host_exempt(lane_claim, false);
    assert!(!ledger.claim(lane_claim).unwrap().host_exempt);
}

#[test]
fn a_unit_that_frees_nothing_is_passed_over() {
    let log = TakeLog::default();
    let mut model = FakeSource::new(
        vec![
            unit(UnitKey::Extent, 0, G, 0, Stage::FreeNow),
            unit(UnitKey::Store(2), 2, G, 0, Stage::FreeNow),
        ],
        &log,
    );
    model.nothing.insert(UnitKey::Extent);
    let mut ledger = Ledger::new();
    let claim = ledger.open(lane(ClaimClass::InService, 1, rank(true, 10)), M);
    let result = ledger.free(
        claim,
        bytes(G),
        room(0),
        &mut FakePool::empty(),
        &mut [&mut model],
    );
    assert_eq!(result.outcome, FreeOutcome::Met);
    assert_eq!(*log.borrow(), vec![UnitKey::Extent, UnitKey::Store(2)]);
}

#[test]
fn one_log_line_per_lack_and_outcome() {
    let lines = Arc::new(Mutex::new(Vec::<String>::new()));
    let mut ledger = Ledger::new();
    let sink = lines.clone();
    ledger.set_decision_log(Arc::new(move |line| {
        sink.lock().unwrap().push(line.to_string())
    }));
    let claim = ledger.open(lane(ClaimClass::InService, 7, rank(true, 10)), M);
    for _ in 0..5 {
        let result = ledger.free(claim, pages(4), room(0), &mut FakePool::empty(), &mut []);
        assert_eq!(result.outcome, FreeOutcome::Short);
    }
    let logged = lines.lock().unwrap().clone();
    assert_eq!(logged.len(), 1, "{logged:?}");
    ledger.granted(claim);
    let _ = ledger.free(claim, pages(4), room(0), &mut FakePool::empty(), &mut []);
    assert_eq!(lines.lock().unwrap().len(), 2);
    assert!(lines.lock().unwrap()[0].starts_with("Memory: in-service claim of request 7"));
}
