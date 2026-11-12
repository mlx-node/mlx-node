use super::super::governor::tests::{FakeIo, G, host, lane, model, stats};
use super::super::governor::{Governor, Scope};
use super::super::host::{HOST_MARGIN_BYTES, Pressure};
use super::super::ledger::ClaimClass;
use super::*;

fn refused(failure: Failure) -> Denial {
    Denial {
        failure: Some(failure),
        pending: false,
        blocks: None,
    }
}

/// Refused at `t0` with 3 admissions counted, no lane.
fn waiting_since(t0: f64) -> ResourceWait {
    let mut wait = ResourceWait::default();
    defer_resource_retry(
        &mut wait,
        t0,
        &refused(Failure::EngineBudget),
        WaitReason::Memory,
        false,
        3,
        0,
    );
    wait
}

fn tick_one(wait: &mut ResourceWait, working: &[LaneWork], now: f64, draining: bool) -> bool {
    let mut waiters = [Waiter {
        wait,
        claim: None,
        submitted: 4,
        finalized: false,
        retried: true,
    }];
    !tick(&mut waiters, working, now, draining, 0).is_empty()
}

#[test]
fn resource_deadline_restarts_only_for_earlier_lanes() {
    let t0 = 100.0;
    // A lane admitted before the refusal works until t0 + 20.
    let mut wait = waiting_since(t0);
    assert_eq!(wait.admitted_before, Some(3));
    assert_eq!(resource_deadline(&wait, 0), t0 + RESOURCE_WAIT_SECONDS);
    let admitted_earlier = LaneWork {
        submitted: 9,
        admission: 2,
    };
    for at in 1..=20 {
        assert!(!tick_one(
            &mut wait,
            &[admitted_earlier],
            t0 + f64::from(at),
            false
        ));
    }
    assert_eq!(resource_deadline(&wait, 0), t0 + 50.0);
    assert!(!tick_one(&mut wait, &[], t0 + 49.0, false));
    assert!(tick_one(&mut wait, &[], t0 + 50.0, false));

    // Admitted after the refusal but submitted before the waiter: restart.
    let mut wait = waiting_since(t0);
    let submitted_earlier = LaneWork {
        submitted: 2,
        admission: 5,
    };
    tick_one(&mut wait, &[submitted_earlier], t0 + 10.0, false);
    assert_eq!(wait.earlier_lane_work, t0 + 10.0);

    // Admitted after the refusal and submitted after the waiter: none.
    let mut wait = waiting_since(t0);
    let later = LaneWork {
        submitted: 9,
        admission: 5,
    };
    tick_one(&mut wait, &[later], t0 + 10.0, false);
    assert_eq!(wait.earlier_lane_work, 0.0);
    assert_eq!(resource_deadline(&wait, 0), t0 + RESOURCE_WAIT_SECONDS);

    // While draining no wait times out.
    let mut wait = waiting_since(t0);
    assert!(!tick_one(&mut wait, &[], t0 + 1000.0, true));
    assert_eq!(resource_deadline(&wait, 0), 0.0);

    // A wait for a lane has no deadline and counts no admissions.
    let mut lane_wait = ResourceWait::default();
    defer_resource_retry(
        &mut lane_wait,
        t0,
        &Denial::default(),
        WaitReason::Concurrency,
        false,
        3,
        0,
    );
    assert_eq!((lane_wait.deadline, lane_wait.admitted_before), (0.0, None));
    assert!(!tick_one(&mut lane_wait, &[], t0 + 1000.0, false));

    // A pending wait whose epoch moved has no deadline until its next
    // refusal, which starts the limit again; it retries at once.
    let mut wait = waiting_since(t0);
    note_refusal(
        &mut wait,
        t0 + 1.0,
        &Denial {
            failure: Some(Failure::EngineBudget),
            pending: true,
            blocks: None,
        },
        0,
    );
    assert_eq!(wait.pending_since, Some(t0 + 1.0));
    assert!(!retry_ready(&wait, t0 + 0.05, 0));
    assert!(retry_ready(&wait, t0 + 0.05, 1));
    assert_eq!(resource_deadline(&wait, 1), 0.0);
    defer_resource_retry(
        &mut wait,
        t0 + 20.0,
        &refused(Failure::EngineBudget),
        WaitReason::Memory,
        false,
        8,
        1,
    );
    assert_eq!(wait.deadline, t0 + 20.0 + RESOURCE_WAIT_SECONDS);
    assert_eq!(wait.admitted_before, Some(8), "no lane: the latest refusal");
    assert_eq!(wait.retry_at, t0 + 20.0 + RETRY_BACKOFF_SECONDS);
    assert_eq!(wait.reason, WaitReason::Memory);
    // A lane in service keeps its first refusal's count.
    defer_resource_retry(
        &mut wait,
        t0 + 21.0,
        &refused(Failure::EngineBudget),
        WaitReason::Memory,
        true,
        9,
        1,
    );
    assert_eq!(wait.admitted_before, Some(8));
}

#[test]
fn host_refused_wait_raises_the_deficit() {
    let h = 9 * G / 10;
    let io = FakeIo::new(stats(0), host(h, Pressure::Normal));
    let governor = Governor::new(io);
    let _model = governor.register_model(model(100 * G, 0)).unwrap();
    let start = governor.open(lane(ClaimClass::Start, 1, 0), 7);
    let bytes = G / 20;
    {
        let _scope = Scope::enter(start.claim());
        assert_eq!(
            governor.try_reserve(bytes).err(),
            Some(Failure::HostPressure)
        );
    }
    let mut wait = ResourceWait::default();
    let epoch = governor.epoch(7);
    defer_resource_retry(
        &mut wait,
        0.0,
        &refused(Failure::HostPressure),
        WaitReason::Memory,
        false,
        0,
        epoch,
    );
    let sync = |wait: &mut ResourceWait, now: f64| {
        let mut waiters = [Waiter {
            wait,
            claim: Some(start.id()),
            submitted: 1,
            finalized: false,
            retried: true,
        }];
        sync_claims(&governor, &mut waiters, now, epoch);
    };
    assert_eq!(governor.deficit_bytes(), HOST_MARGIN_BYTES - h);
    sync(&mut wait, 1.0);
    assert!(governor.ledger(|ledger| ledger.claim(start.id()).unwrap().host_refused_waiting));
    assert_eq!(governor.deficit_bytes(), HOST_MARGIN_BYTES + bytes - h);
    // Past its deadline it no longer counts.
    sync(&mut wait, RESOURCE_WAIT_SECONDS);
    assert!(wait.hold_ended);
    assert!(!governor.ledger(|ledger| ledger.claim(start.id()).unwrap().host_refused_waiting));
    assert_eq!(governor.deficit_bytes(), HOST_MARGIN_BYTES - h);
    assert!(!governor.ledger(|ledger| ledger.claim(start.id()).unwrap().waiting));
}

/// Splash `Engine::syncClaims`: a pending claim holds what it waits for
/// until 30 s after it first went pending, even when a later refusal after
/// progress restarted its deadline.
/// Mutation T3 (the hold never ends) → red (still waiting at t0 + 30).
#[test]
fn hold_ends_30s_after_first_pending() {
    let io = FakeIo::new(stats(0), host(32 * G, Pressure::Normal));
    let governor = Governor::new(io);
    let _model = governor.register_model(model(100 * G, 0)).unwrap();
    let claim = governor.open(lane(ClaimClass::InService, 1, 0), 7);
    let pending = Denial {
        failure: Some(Failure::EngineBudget),
        pending: true,
        blocks: None,
    };
    let t0 = 10.0;
    let mut wait = ResourceWait::default();
    defer_resource_retry(&mut wait, t0, &pending, WaitReason::Memory, true, 0, 0);
    note_refusal(&mut wait, t0 + 20.0, &pending, 1);
    assert_eq!(wait.pending_since, Some(t0));
    assert_eq!(resource_deadline(&wait, 1), t0 + 50.0);
    let waiting = |wait: &mut ResourceWait, now: f64| {
        let mut waiters = [Waiter {
            wait,
            claim: Some(claim.id()),
            submitted: 1,
            finalized: false,
            retried: true,
        }];
        sync_claims(&governor, &mut waiters, now, 1);
        governor.ledger(|ledger| ledger.claim(claim.id()).unwrap().waiting)
    };
    assert!(waiting(&mut wait, t0 + 29.0));
    assert!(!wait.hold_ended);
    assert!(!waiting(&mut wait, t0 + 30.0));
    assert!(wait.hold_ended);
}
