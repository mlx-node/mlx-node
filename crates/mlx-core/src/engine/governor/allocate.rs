//! The allocate loop, after Splash `Engine::allocate` (`EngineAllocate.hpp`):
//! an attempt in its claim's scope; a refusal asks the ledger to free what
//! the claim lacks, and the attempt runs again once that is met or memory
//! left now.

use super::governor::{ClaimRef, Failure, Reservation, Scope};
use super::ledger::{ClaimSpec, FreeOutcome, Need, VictimSource};
use crate::transformer::paged_kv_cache_adapter::BlockRefusal;

/// What a refused allocation leaves (Splash `Refusal`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Refusal {
    /// The request waits: its failure counts for the recovery drain.
    Waits,
    /// The caller goes on without it.
    Settled,
    /// The caller goes on without it, and nothing is freed for it: one
    /// attempt (a capture whose lane holds a state to take it from).
    SettledOnce,
}

/// An attempt's result as the loop reads it.
pub(crate) trait Admission {
    fn granted(&self) -> bool;
    fn failure(&self) -> Option<Failure>;
    /// Splash `inPlace`: the part of the need memory already allocated
    /// meets in place, with `held` blocks held for claims ranked above.
    fn in_place(&self, held: u32) -> Need;

    /// A refusal for memory, which what the ledger frees may end.
    fn memory_denied(&self) -> bool {
        matches!(
            self.failure(),
            Some(Failure::EngineBudget | Failure::HostPressure | Failure::DriverRejected)
        )
    }

    /// The blocks a refused KV reservation lacked.
    fn blocks(&self) -> Option<BlockRefusal> {
        None
    }
}

/// A KV step or prefill booking (Splash `TokenAdmission`): its result, the
/// adapter's refusal record, and what refused the scope's growth.
pub(crate) struct KvAdmission<R> {
    pub result: napi::Result<R>,
    pub refusal: Option<BlockRefusal>,
    pub failure: Option<Failure>,
}

impl<R> Admission for KvAdmission<R> {
    fn granted(&self) -> bool {
        self.result.is_ok()
    }

    fn failure(&self) -> Option<Failure> {
        self.failure
    }

    /// Splash `inPlace`: the blocks lacked beyond those the allocator could
    /// hand out, with the blocks held for claims ranked above left free.
    fn in_place(&self, held: u32) -> Need {
        let pages = self.refusal.map_or(0, |refusal| {
            refusal
                .additional
                .saturating_add(held)
                .saturating_sub(refusal.available)
        });
        Need {
            pages,
            kv: true,
            ..Need::default()
        }
    }

    /// Any refused block reservation, whatever refused its growth.
    fn memory_denied(&self) -> bool {
        self.result.is_err() && self.refusal.is_some()
    }

    fn blocks(&self) -> Option<BlockRefusal> {
        self.refusal
    }
}

/// An explicit reservation for a lazy MLX allocation (a GDN cell, draft
/// state, a publication's copy): bytes only. Granted, the caller keeps the
/// reservation until the eval that allocates the bytes has run.
pub(crate) struct StateAdmission<'g>(pub Result<Reservation<'g>, Failure>);

impl Admission for StateAdmission<'_> {
    fn granted(&self) -> bool {
        self.0.is_ok()
    }

    fn failure(&self) -> Option<Failure> {
        self.0.as_ref().err().copied()
    }

    fn in_place(&self, _held: u32) -> Need {
        Need::default()
    }
}

/// The caller's victim sources, borrowed from the same state the attempt
/// borrows, one after the other.
pub(crate) trait UnitSources {
    fn sources(&mut self) -> Vec<&mut dyn VictimSource>;
}

/// Splash `Denial`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Denial {
    pub failure: Option<Failure>,
    /// The ledger's last decision was to wait for what lands.
    pub pending: bool,
    /// What a KV reservation lacked.
    pub blocks: Option<BlockRefusal>,
}

/// Splash `Engine::Verdict`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Verdict {
    Wait,
    Yield,
    Fail,
}

/// Splash `Engine::judge`: memory on its way back is waited for; pages
/// resident lanes hold come back when they finish, and memory the host
/// refuses when its pressure lifts; only a lane that cannot fit on its own
/// has hit the capacity.
pub(crate) fn judge(denial: &Denial, another_resident: bool) -> Verdict {
    if denial.pending {
        return Verdict::Wait;
    }
    if another_resident || denial.failure == Some(Failure::HostPressure) {
        return Verdict::Yield;
    }
    Verdict::Fail
}

/// Splash `Engine::capacityExhausted`: the lone lane's judge failed.
pub(crate) fn capacity_exhausted(what: &str, failure: Option<Failure>, detail: &str) -> String {
    let detail = if detail.is_empty() {
        String::new()
    } else {
        format!(" ({detail})")
    };
    format!(
        "capacity_exhausted: could not allocate {what}: {}{detail}; the request does not fit in \
         the memory this server may use, even after every cached prefix was evicted: restart \
         the server with a smaller --max-context, or with a larger --max-memory where one \
         lowered it",
        Failure::name(failure)
    )
}

/// Splash `pageShortfall`.
pub(crate) fn page_shortfall(blocks: Option<BlockRefusal>) -> String {
    blocks.map_or_else(String::new, |blocks| {
        format!(
            "additional_pages={}, free_pages={}",
            blocks.additional, blocks.available
        )
    })
}

pub(crate) struct Allocation<A> {
    pub admission: A,
    pub denial: Denial,
    /// A free took memory for it, or the host's exemption let it pass.
    pub freed: bool,
}

/// Splash `Engine::allocate`. `claim` is the request's (or one the caller
/// opened for this allocation alone); `hold_ended` is the request's wait
/// past its hold's limit, whose claim waits holding nothing.
pub(crate) fn allocate<C: UnitSources, A: Admission>(
    claim: ClaimRef<'_>,
    spec: ClaimSpec,
    refusal: Refusal,
    hold_ended: bool,
    ctx: &mut C,
    mut attempt: impl FnMut(&mut C) -> A,
) -> Allocation<A> {
    let governor = claim.governor;
    let id = claim.id;
    let model = governor.ledger(|ledger| {
        ledger.update(id, spec);
        ledger.claim(id).map_or(0, |state| state.model)
    });
    // Each attempt runs in the claim's scope, from a need its refusal
    // records afresh.
    let mut try_once = |ctx: &mut C| {
        governor.ledger(|ledger| ledger.refused(id, Need::default()));
        let _scope = Scope::enter(claim);
        attempt(ctx)
    };
    let mut result = Allocation {
        admission: try_once(ctx),
        denial: Denial::default(),
        freed: false,
    };
    // The loop stops at a refusal the ledger cannot help: one that lacks
    // nothing it counts, one the host refuses after exempting the claim,
    // or a need already freed for once in this allocation.
    let mut met: Vec<Need> = Vec::new();
    while refusal != Refusal::SettledOnce && result.admission.memory_denied() {
        let part = result.admission.in_place(governor.held_pages(id));
        let need = governor.need(id, part);
        let exempt =
            governor.ledger(|ledger| ledger.claim(id).is_some_and(|state| state.host_exempt));
        if (need.bytes == 0 && !need.host_refused && need.pages == 0)
            || (need.host_refused && exempt)
            || met.contains(&need)
        {
            break;
        }
        let freed = {
            let mut sources = ctx.sources();
            governor.free(id, need, &mut sources)
        };
        result.denial.pending = freed.outcome == FreeOutcome::Pending;
        if result.denial.pending && spec.request_id != 0 && hold_ended {
            governor.ledger(|ledger| ledger.set_waiting(id, false));
        }
        if freed.outcome != FreeOutcome::Met && freed.now_bytes == 0 {
            if freed.later_bytes > 0 {
                result.admission = try_once(ctx);
            }
            break;
        }
        let exempt_now =
            governor.ledger(|ledger| ledger.claim(id).is_some_and(|state| state.host_exempt));
        result.freed = result.freed || exempt || !exempt_now;
        met.push(need);
        governor.signal_progress(model);
        result.admission = try_once(ctx);
    }
    // The host's exemption covers this allocation's retries alone.
    let granted = result.admission.granted();
    governor.ledger(|ledger| {
        ledger.set_host_exempt(id, false);
        if granted {
            ledger.granted(id);
        }
    });
    if !granted {
        result.denial.failure = result.admission.failure();
        result.denial.blocks = result.admission.blocks();
        if result.admission.memory_denied() && !result.denial.pending && refusal == Refusal::Waits {
            governor.note_allocation_failed(model);
        }
    }
    result
}

#[cfg(test)]
mod tests;
