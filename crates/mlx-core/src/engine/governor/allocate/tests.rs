use super::super::governor::tests::{FakeIo, G, Units, host, lane, model, stats};
use super::super::governor::{Governor, MemoryStats, StartScope, take_start_refusal};
use super::super::host::Pressure;
use super::super::ledger::ClaimClass;
use super::*;

struct Ctx {
    tries: u32,
    units: Units,
}

impl UnitSources for Ctx {
    fn sources(&mut self) -> Vec<&mut dyn VictimSource> {
        vec![&mut self.units]
    }
}

#[test]
fn allocate_retries_after_met_and_stops_at_a_repeated_need() {
    let io = FakeIo::new(stats(18), host(4 * G, Pressure::Normal));
    let governor = Governor::new(io);
    let _model = governor.register_model(model(20, 0)).unwrap();
    let spec = lane(ClaimClass::InService, 1, 0);
    let claim = governor.open(spec, 7);
    // Room 2; the first unit's byte meets the 3-byte need, but what it
    // frees never reaches the counters, so the same need comes back.
    let mut ctx = Ctx {
        tries: 0,
        units: Units::new(2, 1),
    };
    let reserve = |ctx: &mut Ctx| {
        ctx.tries += 1;
        StateAdmission(governor.try_reserve(3))
    };
    let allocation = allocate(
        claim.claim(),
        spec,
        Refusal::Waits,
        false,
        &mut ctx,
        reserve,
    );
    assert_eq!(ctx.tries, 2);
    assert!(!allocation.admission.granted());
    assert!(allocation.freed);
    assert_eq!(
        allocation.denial,
        Denial {
            failure: Some(Failure::EngineBudget),
            pending: false,
            blocks: None,
            settled: false,
        }
    );
    assert_eq!(governor.epoch(7), 1);
    assert!(governor.allocation_failed(7));
    governor.lane_released(7);
    assert!(!governor.allocation_failed(7));
    assert_eq!(governor.epoch(7), 2);

    // A grant.
    let granted = allocate(
        claim.claim(),
        spec,
        Refusal::Settled,
        false,
        &mut ctx,
        |ctx: &mut Ctx| {
            ctx.tries += 1;
            StateAdmission(governor.try_reserve(1))
        },
    );
    assert!(granted.admission.granted());
    assert_eq!(granted.denial, Denial::default());
}

/// Under the start an allocation is the start's: granted at a macOS warning
/// as `Floor`; refused over the limit after one attempt, with no unit
/// freed, no failed allocation noted, a settled denial and no start refusal
/// written.
#[test]
fn a_start_claim_is_floor_and_tries_once() {
    let io = FakeIo::new(stats(18), host(4 * G, Pressure::Warning));
    let governor = Governor::new(io);
    let _model = governor.register_model(model(20, 0)).unwrap();
    let spec = lane(ClaimClass::InService, 1, 0);
    let claim = governor.open(spec, 7);
    let mut ctx = Ctx {
        tries: 0,
        units: Units::new(2, 1),
    };
    let _start = StartScope::enter();
    let granted = allocate(
        claim.claim(),
        spec,
        Refusal::Waits,
        false,
        &mut ctx,
        |ctx: &mut Ctx| {
            ctx.tries += 1;
            StateAdmission(governor.try_reserve(1))
        },
    );
    assert!(granted.admission.granted());
    assert_eq!(
        governor.spec(claim.id()).map(|spec| spec.cls),
        Some(ClaimClass::Floor)
    );
    drop(granted);
    ctx.tries = 0;
    let refused = allocate(
        claim.claim(),
        spec,
        Refusal::Waits,
        false,
        &mut ctx,
        |ctx: &mut Ctx| {
            ctx.tries += 1;
            StateAdmission(governor.try_reserve(3))
        },
    );
    assert_eq!(ctx.tries, 1);
    assert_eq!(ctx.units.units.len(), 2);
    assert!(!refused.admission.granted());
    assert_eq!(
        refused.denial,
        Denial {
            failure: Some(Failure::EngineBudget),
            pending: false,
            blocks: None,
            settled: true,
        }
    );
    assert!(!governor.allocation_failed(7));
    assert_eq!(take_start_refusal(), None);
}

/// A start claim the engine limit refuses only because MLX's free pool
/// holds bytes empties the pool and is granted; one the pool cannot help
/// is refused after that clear, and an empty pool adds no attempt.
#[test]
fn a_start_claim_clears_the_free_pool_before_its_refusal() {
    let io = FakeIo::new(
        MemoryStats {
            active: 10,
            pools: 0,
            cache: 9,
        },
        host(4 * G, Pressure::Normal),
    );
    let governor = Governor::new(io.clone());
    let _model = governor.register_model(model(20, 0)).unwrap();
    let spec = lane(ClaimClass::InService, 1, 0);
    let claim = governor.open(spec, 7);
    let mut ctx = Ctx {
        tries: 0,
        units: Units::new(2, 1),
    };
    let _start = StartScope::enter();
    let reserve = |bytes: u64, ctx: &mut Ctx| {
        allocate(
            claim.claim(),
            spec,
            Refusal::Waits,
            false,
            ctx,
            |ctx: &mut Ctx| {
                ctx.tries += 1;
                StateAdmission(governor.try_reserve(bytes))
            },
        )
    };
    let granted = reserve(3, &mut ctx);
    assert!(granted.admission.granted());
    assert_eq!(granted.denial, Denial::default());
    assert_eq!(
        (ctx.tries, io.with(|state| state.stats.cache)),
        (2, 0),
        "the pool was cleared and the claim tried again"
    );
    assert_eq!(ctx.units.units.len(), 2, "no unit was taken");
    drop(granted);

    io.with(|state| state.stats.cache = 9);
    ctx.tries = 0;
    let refused = reserve(11, &mut ctx);
    assert_eq!(ctx.tries, 2);
    assert!(!refused.admission.granted());
    assert_eq!(refused.denial.failure, Some(Failure::EngineBudget));
    assert!(refused.denial.settled);
    drop(refused);

    ctx.tries = 0;
    let refused = reserve(11, &mut ctx);
    assert_eq!(ctx.tries, 1, "an empty pool adds no attempt");
    assert!(refused.denial.settled);
    assert_eq!(ctx.units.units.len(), 2);
}

/// A serving claim in the same state is granted as before: its free takes
/// the pool first.
#[test]
fn a_serving_claim_still_takes_the_free_pool_through_its_free() {
    let io = FakeIo::new(
        MemoryStats {
            active: 10,
            pools: 0,
            cache: 9,
        },
        host(4 * G, Pressure::Normal),
    );
    let governor = Governor::new(io.clone());
    let _model = governor.register_model(model(20, 0)).unwrap();
    let spec = lane(ClaimClass::InService, 1, 0);
    let claim = governor.open(spec, 7);
    let mut ctx = Ctx {
        tries: 0,
        units: Units::new(2, 1),
    };
    let granted = allocate(
        claim.claim(),
        spec,
        Refusal::Waits,
        false,
        &mut ctx,
        |ctx: &mut Ctx| {
            ctx.tries += 1;
            StateAdmission(governor.try_reserve(3))
        },
    );
    assert!(granted.admission.granted());
    assert!(!granted.denial.settled);
    assert_eq!((ctx.tries, io.with(|state| state.stats.cache)), (2, 0));
}

/// A settled denial fails whatever else it holds; every other row is the
/// judge's as before.
#[test]
fn judge_fails_a_settled_denial() {
    let rows = [
        (false, true, Failure::EngineBudget, false, Verdict::Wait),
        (false, false, Failure::EngineBudget, true, Verdict::Yield),
        (false, false, Failure::HostPressure, false, Verdict::Yield),
        (false, false, Failure::EngineBudget, false, Verdict::Fail),
        (true, true, Failure::EngineBudget, false, Verdict::Fail),
        (true, false, Failure::EngineBudget, true, Verdict::Fail),
        (true, false, Failure::HostPressure, false, Verdict::Fail),
        (true, false, Failure::EngineBudget, false, Verdict::Fail),
    ];
    for (settled, pending, failure, another_resident, verdict) in rows {
        let denial = Denial {
            failure: Some(failure),
            pending,
            blocks: None,
            settled,
        };
        assert_eq!(
            judge(&denial, another_resident),
            verdict,
            "{denial:?}, another resident {another_resident}"
        );
    }
}
