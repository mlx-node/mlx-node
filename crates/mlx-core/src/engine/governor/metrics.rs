//! Splash's serving counters: what held each request before its first
//! token (`TtftPart`, `Engine::holds`, `Engine::holderOf`), the cause of
//! each decode gap past the target (`GapCause`, `Engine::decodeGap`) and
//! the gaps' log-2 histogram (`LatencyHistogram`). Times are seconds on the
//! governor's clock.

use std::cell::RefCell;
use std::marker::PhantomData;
use std::time::Instant;

use crate::engine::latency::LatencyHistogram;
use crate::engine::scheduler::DECODE_GAP_MS;
use crate::transformer::paged_kv_cache_adapter::SeqId;

use super::governor::Governor;

/// Splash `TtftPart`, without `Rest`, which is never accumulated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TtftPart {
    OwnPrefill,
    ClaimWait,
    Decode,
    OtherPrefill,
}

pub(crate) const TTFT_PARTS: usize = 4;

/// What one drive pass ran: Splash's command in flight.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct PassWork {
    /// The requests whose prefill rows ran.
    pub prefill: Vec<SeqId>,
    /// Decode rows ran, or a whole turn.
    pub decode: bool,
    /// The step's prefill-phase wall.
    pub prefill_seconds: f64,
}

impl PassWork {
    /// Splash `Engine::holderOf`: the request's own prefill, then its
    /// claim's wait, then the command's kind; `None` is Rest.
    pub(crate) fn holder_of(&self, request: Option<SeqId>, claim_waits: bool) -> Option<TtftPart> {
        if request.is_some_and(|id| self.prefill.contains(&id)) {
            return Some(TtftPart::OwnPrefill);
        }
        if claim_waits {
            return Some(TtftPart::ClaimWait);
        }
        if self.decode {
            return Some(TtftPart::Decode);
        }
        (!self.prefill.is_empty()).then_some(TtftPart::OtherPrefill)
    }

    /// The prefill units' time `lane` waited through without a row in them
    /// (Splash's batch completion adds a prefill command's wall to every
    /// lane outside it).
    pub(crate) fn prefill_beside(&self, lane: SeqId) -> f64 {
        if self.prefill.contains(&lane) {
            0.0
        } else {
            self.prefill_seconds
        }
    }
}

/// Seconds by [`TtftPart`].
pub(crate) type TtftParts = [f64; TTFT_PARTS];

/// A request's time before its first tokens (Splash `Request::held`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct TtftClock {
    held: TtftParts,
    sent: bool,
}

impl TtftClock {
    /// A clock that starts with what held the request while it queued.
    pub(crate) fn queued(held: TtftParts) -> Self {
        Self { held, sent: false }
    }

    pub(crate) fn hold(&mut self, seconds: f64, part: Option<TtftPart>) {
        if !self.sent
            && let Some(part) = part
        {
            self.held[part as usize] += seconds.max(0.0);
        }
    }

    /// The first tokens went out (Splash `Engine::firstTokens`): the parts,
    /// once.
    pub(crate) fn first_tokens(&mut self) -> Option<TtftParts> {
        (!std::mem::replace(&mut self.sent, true)).then_some(self.held)
    }
}

/// What held the requests that have no claim and no row yet (queued
/// commands: Decode, OtherPrefill or Rest), summed since the model thread
/// started. A queued request's parts are the difference between now and
/// its arrival's mark.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Timeline {
    held: TtftParts,
}

impl Timeline {
    pub(crate) fn hold(&mut self, seconds: f64, part: Option<TtftPart>) {
        if let Some(part) = part {
            self.held[part as usize] += seconds.max(0.0);
        }
    }

    pub(crate) fn since(&self, mark: &Timeline) -> TtftParts {
        std::array::from_fn(|part| (self.held[part] - mark.held[part]).max(0.0))
    }
}

/// Splash `GapCause`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GapCause {
    PrefillUnit,
    ClaimWait,
    SsdRead,
    PoolRelease,
    Replan,
    Other,
}

pub(crate) const GAP_CAUSES: usize = 6;

/// A gap between two token events of a lane.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Gap {
    pub seconds: f64,
    /// Past the target: the first cause that held in it.
    pub cause: Option<GapCause>,
    pub prefill_seconds: f64,
}

/// A lane's last token event and what happened since (Splash
/// `Request::gapSince`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct GapClock {
    last_token: Option<f64>,
    prefill_seconds: f64,
    claim_waited: bool,
    ssd_read: bool,
    pool_releases: u64,
    replans: u64,
}

impl GapClock {
    /// Prefill units ran for `prefill_seconds`, or the lane's claim waited.
    pub(crate) fn worked(&mut self, prefill_seconds: f64, claim_waited: bool) {
        self.prefill_seconds += prefill_seconds.max(0.0);
        self.claim_waited |= claim_waited;
    }

    /// The prefill units' time since the last token event.
    #[cfg(test)]
    pub(crate) fn prefill_seconds(&self) -> f64 {
        self.prefill_seconds
    }

    #[cfg(test)]
    /// A cold-tier restore poll ran.
    pub(crate) fn read_ssd(&mut self) {
        self.ssd_read = true;
    }

    /// Splash `Engine::decodeGap`: a token event at `now`. The gap since the
    /// lane's last one, past the target with the first cause that held in
    /// it, in Splash's order.
    pub(crate) fn token(&mut self, now: f64, pool_releases: u64, replans: u64) -> Option<Gap> {
        let since = *self;
        *self = Self {
            last_token: Some(now),
            pool_releases,
            replans,
            ..Self::default()
        };
        let last = since.last_token?;
        let seconds = (now - last).max(0.0);
        let first = if since.prefill_seconds > 0.0 {
            GapCause::PrefillUnit
        } else if since.claim_waited {
            GapCause::ClaimWait
        } else if since.ssd_read {
            GapCause::SsdRead
        } else if pool_releases != since.pool_releases {
            GapCause::PoolRelease
        } else if replans != since.replans {
            GapCause::Replan
        } else {
            GapCause::Other
        };
        let cause = (seconds * 1000.0 > DECODE_GAP_MS).then_some(first);
        Some(Gap {
            seconds,
            cause,
            prefill_seconds: if cause == Some(GapCause::PrefillUnit) {
                since.prefill_seconds
            } else {
                0.0
            },
        })
    }
}

/// The process's counters the model threads add to.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Metrics {
    pub ttft_parts: TtftParts,
    pub gaps: [u64; GAP_CAUSES],
    pub gap_seconds: [f64; GAP_CAUSES],
    /// Of the gaps a prefill unit caused, those units' time.
    pub gap_prefill_seconds: f64,
    pub gap_histogram: LatencyHistogram,
    pub resource_timeouts: u64,
    pub capacity_exhausted: u64,
}

impl Metrics {
    pub(crate) fn first_tokens(&mut self, parts: &TtftParts) {
        for (sum, part) in self.ttft_parts.iter_mut().zip(parts) {
            *sum += part;
        }
    }

    pub(crate) fn gap(&mut self, gap: &Gap) {
        self.gap_histogram.add(gap.seconds);
        if let Some(cause) = gap.cause {
            self.gaps[cause as usize] += 1;
            self.gap_seconds[cause as usize] += gap.seconds;
            self.gap_prefill_seconds += gap.prefill_seconds;
        }
    }
}

/// A whole turn the model thread runs: when its command arrived, when its
/// first token went out, and its decode gaps.
pub(crate) struct WholeTurn {
    pub enqueued_at: Instant,
    pub first_token: Option<Instant>,
    /// Seconds its claim waited before its first token.
    claim_wait: f64,
    gap: GapClock,
}

thread_local! {
    static WHOLE_TURN: RefCell<Option<WholeTurn>> = const { RefCell::new(None) };
}

/// While one lives, this thread runs one whole turn that arrived at
/// `enqueued_at`.
pub(crate) struct WholeTurnScope {
    _thread: PhantomData<*const ()>,
}

impl WholeTurnScope {
    pub(crate) fn enter(enqueued_at: Instant) -> Self {
        WHOLE_TURN.with(|turn| {
            *turn.borrow_mut() = Some(WholeTurn {
                enqueued_at,
                first_token: None,
                claim_wait: 0.0,
                gap: GapClock::default(),
            });
        });
        Self {
            _thread: PhantomData,
        }
    }

    /// The turn's first token, when one went out, and the seconds its
    /// claim waited before it.
    pub(crate) fn finish(self) -> Option<(Instant, f64)> {
        WHOLE_TURN.with(|turn| {
            turn.borrow_mut()
                .take()
                .and_then(|turn| turn.first_token.map(|first| (first, turn.claim_wait)))
        })
    }
}

impl Drop for WholeTurnScope {
    fn drop(&mut self) {
        WHOLE_TURN.with(|turn| turn.borrow_mut().take());
    }
}

/// The arrival of the whole turn this thread runs: its TTFT starts there.
pub(crate) fn whole_turn_enqueued_at() -> Option<Instant> {
    WHOLE_TURN.with(|turn| turn.borrow().as_ref().map(|turn| turn.enqueued_at))
}

pub(crate) fn whole_turn_first_token(at: Instant) {
    WHOLE_TURN.with(|turn| {
        if let Some(turn) = turn.borrow_mut().as_mut() {
            turn.first_token.get_or_insert(at);
        }
    });
}

/// The whole turn's claim waited `seconds` for memory.
pub(crate) fn whole_turn_claim_waited(seconds: f64) {
    WHOLE_TURN.with(|turn| {
        if let Some(turn) = turn.borrow_mut().as_mut() {
            turn.gap.worked(0.0, true);
            if turn.first_token.is_none() {
                turn.claim_wait += seconds;
            }
        }
    });
}

/// A whole turn's TTFT (Splash `holderOf` over a turn alone on its model):
/// what held it while it queued, then its claim's waits, then its own
/// prefill for the rest of the `run` seconds to its first token.
pub(crate) fn whole_turn_ttft(queued: TtftParts, run: f64, claim_wait: f64) -> TtftClock {
    let claim_wait = claim_wait.min(run);
    let mut clock = TtftClock::queued(queued);
    clock.hold(claim_wait, Some(TtftPart::ClaimWait));
    clock.hold(run - claim_wait, Some(TtftPart::OwnPrefill));
    clock
}

/// A decode step of the whole turn this thread runs: a token event.
pub(crate) fn whole_turn_token(governor: &Governor) {
    let now = governor.now();
    let releases = governor.pool_releases();
    let gap = WHOLE_TURN.with(|turn| {
        turn.borrow_mut()
            .as_mut()
            .and_then(|turn| turn.gap.token(now, releases, 0))
    });
    if let Some(gap) = gap {
        governor.decode_gap(&gap);
    }
}

#[cfg(test)]
mod tests;
