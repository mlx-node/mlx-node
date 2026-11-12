use super::super::governor::Governor;
use super::super::governor::tests::{FakeIo, G, Units, host, lane, model, stats};
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
