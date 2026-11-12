//! Resource waits, after Splash `Engine::ResourceWait`,
//! `deferResourceRetry`, `noteRefusal`, `resourceDeadline`, `tick` and
//! `syncClaims`. Times are seconds on the governor's clock; the epoch is
//! the model's (`Governor::epoch`).

use super::allocate::Denial;
use super::governor::{Failure, Governor};
use super::ledger::ClaimId;

/// Splash `resourceWaitTimeoutMilliseconds`: the wait limit and the hold's.
pub(crate) const RESOURCE_WAIT_SECONDS: f64 = 30.0;
/// Splash `kResourceRetryBackoffMilliseconds`.
pub(crate) const RETRY_BACKOFF_SECONDS: f64 = 0.1;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum WaitReason {
    #[default]
    None,
    Memory,
    /// A wait for a lane, not for memory: no deadline.
    Concurrency,
}

/// Splash `Engine::ResourceWait`; 0.0 is an unset time.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct ResourceWait {
    pub reason: WaitReason,
    pub failure: Option<Failure>,
    pub started: Option<f64>,
    /// The admissions counted at the latest refusal of a request with no
    /// lane, or at the first refusal of a lane in service: lanes admitted
    /// up to then restart its limit while they work.
    pub admitted_before: Option<u64>,
    pub deadline: f64,
    /// The latest tick at which a lane submitted before the request, or
    /// one `admitted_before` counts, had work in flight.
    pub earlier_lane_work: f64,
    pub retry_at: f64,
    pub epoch: u64,
    pub pending: bool,
    pub pending_since: Option<f64>,
    /// Past its hold's limit the claim waits holding nothing.
    pub hold_ended: bool,
}

/// Splash `ResourceTimeout`, as `backend.py` words it: a 503 the client
/// retries.
pub(crate) fn resource_timeout(failure: Option<Failure>) -> String {
    let host = if failure == Some(Failure::HostPressure) {
        ": macOS is short of memory; close memory-heavy applications"
    } else {
        ""
    };
    format!(
        "resource_timeout: memory did not become available within the resource wait limit{host}"
    )
}

/// Splash `Engine::deferResourceRetry`.
pub(crate) fn defer_resource_retry(
    wait: &mut ResourceWait,
    now: f64,
    denial: &Denial,
    reason: WaitReason,
    has_lane: bool,
    admissions: u64,
    epoch: u64,
) {
    if wait.started.is_none() {
        wait.started = Some(now);
    }
    if reason != WaitReason::Concurrency && (!has_lane || wait.admitted_before.is_none()) {
        wait.admitted_before = Some(admissions);
    }
    note_refusal(wait, now, denial, epoch);
    wait.reason = reason;
    if reason == WaitReason::Concurrency {
        wait.deadline = 0.0;
    } else if wait.deadline <= 0.0 {
        wait.deadline = now + RESOURCE_WAIT_SECONDS;
    }
    wait.retry_at = now + RETRY_BACKOFF_SECONDS;
}

/// Splash `Engine::noteRefusal`: a pending wait that saw progress since
/// its last refusal starts its limit again.
pub(crate) fn note_refusal(wait: &mut ResourceWait, now: f64, denial: &Denial, epoch: u64) {
    if wait.pending && wait.epoch != epoch && wait.deadline > 0.0 {
        wait.deadline = now + RESOURCE_WAIT_SECONDS;
    }
    if denial.pending && wait.started.is_none() {
        wait.started = Some(now);
    }
    if denial.pending && wait.pending_since.is_none() {
        wait.pending_since = Some(now);
    }
    wait.failure = denial.failure;
    wait.pending = denial.pending;
    wait.epoch = epoch;
}

/// Splash `Engine::resourceDeadline`: none while a pending wait's epoch
/// moved; else the later of its own deadline and a limit after the latest
/// earlier lane's work.
pub(crate) fn resource_deadline(wait: &ResourceWait, epoch: u64) -> f64 {
    if (wait.pending && wait.epoch != epoch) || wait.deadline <= 0.0 {
        return 0.0;
    }
    wait.deadline
        .max(wait.earlier_lane_work + RESOURCE_WAIT_SECONDS)
}

/// Splash `Engine::resourceRetryReady`.
pub(crate) fn retry_ready(wait: &ResourceWait, now: f64, epoch: u64) -> bool {
    wait.retry_at <= 0.0 || wait.epoch != epoch || now >= wait.retry_at
}

/// A lane of the step that just ran: its submission order and admission
/// count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LaneWork {
    pub submitted: u64,
    pub admission: u64,
}

/// A request with a wait, as `tick` and `sync_claims` read it.
pub(crate) struct Waiter<'a> {
    pub wait: &'a mut ResourceWait,
    pub claim: Option<ClaimId>,
    pub submitted: u64,
    pub finalized: bool,
    /// A lane neither below the runnable ones nor passed over; a request
    /// with no lane that admission tries this pass.
    pub retried: bool,
}

/// Splash `Engine::tick`'s wait part over the lanes of the step that just
/// ran. Returns the waiters whose limit has passed.
pub(crate) fn tick(
    waiters: &mut [Waiter<'_>],
    working: &[LaneWork],
    now: f64,
    draining: bool,
    epoch: u64,
) -> Vec<usize> {
    let earliest_working = working
        .iter()
        .map(|lane| lane.submitted)
        .min()
        .unwrap_or(u64::MAX);
    let earliest_admission = working
        .iter()
        .map(|lane| lane.admission)
        .min()
        .unwrap_or(u64::MAX);
    let mut timed_out = Vec::new();
    for (at, waiter) in waiters.iter_mut().enumerate() {
        // Admission is paused while resident lanes drain: no wait times out.
        if draining {
            waiter.wait.deadline = 0.0;
            continue;
        }
        let submitted_before = waiter.submitted > earliest_working;
        let admitted_before = waiter
            .wait
            .admitted_before
            .is_some_and(|admitted| earliest_admission <= admitted);
        if submitted_before || admitted_before {
            waiter.wait.earlier_lane_work = now;
        }
        let deadline = resource_deadline(waiter.wait, epoch);
        if !waiter.finalized && deadline > 0.0 && now >= deadline {
            timed_out.push(at);
        }
    }
    timed_out
}

/// Splash `Engine::syncClaims`: a claim holds what it waits for only while
/// its request waits for what lands, is retried, and its hold has not
/// ended; a request the host refused, retried before its deadline, counts
/// in the host's deficit.
pub(crate) fn sync_claims(governor: &Governor, waiters: &mut [Waiter<'_>], now: f64, epoch: u64) {
    for waiter in waiters.iter_mut() {
        let Some(claim) = waiter.claim else {
            continue;
        };
        let wait = &mut *waiter.wait;
        let deadline = resource_deadline(wait, epoch);
        wait.hold_ended = wait
            .pending_since
            .is_some_and(|since| now >= since + RESOURCE_WAIT_SECONDS)
            || (deadline > 0.0 && now >= deadline);
        let waiting = !waiter.finalized
            && wait.pending
            && wait.pending_since.is_some()
            && waiter.retried
            && !wait.hold_ended;
        let host_refused = !waiter.finalized
            && !wait.pending
            && wait.failure == Some(Failure::HostPressure)
            && waiter.retried
            && (deadline <= 0.0 || now < deadline);
        governor.ledger(|ledger| {
            if ledger
                .claim(claim)
                .is_some_and(|state| state.waiting != waiting)
            {
                ledger.set_waiting(claim, waiting);
            }
            ledger.set_host_refused_waiting(claim, host_refused);
        });
    }
}

#[cfg(test)]
mod tests;
