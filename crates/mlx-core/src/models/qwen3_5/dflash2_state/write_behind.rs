//! Copies of the newest restore points on the SSD tier while they stay in
//! RAM, ahead of need (Splash `WriteBehind`): a reclaim then frees their
//! memory at once, and a persistent tier's next process takes them back.
//! An ordinary point published or reused at least [`MINIMUM_TOKENS`] deep
//! is copied once it waited [`DELAY_SECONDS`], unless its conversation went
//! on from it by then. A persistent tier copies every point, in use or not;
//! a temporary one only on the memory's demand for copies, and only points
//! whose turn ended. None is copied while a claim waits for memory on its
//! way back.

use std::collections::VecDeque;

use crate::engine::governor::ledger::{Need, Stage, TakeRights, VictimQuery};
use crate::models::qwen3_5::dflash2_turn::{DFlash2Target, Keep, model_units};

use super::disk::{PersistStatus, copying_ahead_bytes, newest_point, persist};
use super::paged::{DFlash2States, PagedAnchor};
use super::store::{EntryId, HybridStateStore};

/// Splash `WriteBehind::kDelayMilliseconds`.
pub(crate) const DELAY_SECONDS: f64 = 10.0;

/// Splash `WriteBehind::kMinimumTokens`: a shorter point recomputes faster
/// than its state writes.
#[cfg(not(test))]
pub(crate) const MINIMUM_TOKENS: u32 = 2048;
#[cfg(test)]
pub(crate) const MINIMUM_TOKENS: u32 = 32;

/// Splash `WriteBehindSnapshot`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct WriteBehindSnapshot {
    pub(crate) waiting: u32,
    pub(crate) durable: u64,
    pub(crate) unneeded: u64,
    pub(crate) refused: u64,
}

/// Why the last step stopped before the points due (Splash
/// `WriteBehind::Stop`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Stop {
    #[default]
    None,
    ClaimWaiting,
    Covered,
    /// The first point due and not in use waits for a landing.
    Stalled,
}

#[derive(Clone, Copy, Debug)]
struct Point {
    id: EntryId,
    due: f64,
}

#[derive(Default)]
pub(crate) struct WriteBehind {
    waiting: VecDeque<Point>,
    stopped: Stop,
    /// Copies are made, as last asked.
    copying: bool,
    /// The demand the last step was asked with.
    demand_bytes: u64,
    counters: WriteBehindSnapshot,
}

impl WriteBehind {
    pub(crate) fn new(persistent: bool) -> Self {
        Self {
            copying: persistent,
            ..Self::default()
        }
    }

    pub(crate) fn snapshot(&self) -> WriteBehindSnapshot {
        WriteBehindSnapshot {
            waiting: u32::try_from(self.waiting.len()).unwrap_or(u32::MAX),
            ..self.counters
        }
    }

    #[cfg(test)]
    pub(crate) fn stopped(&self) -> Stop {
        self.stopped
    }

    /// Counts a step that settled its point; false while it is in
    /// progress.
    fn settle(&mut self, status: PersistStatus) -> bool {
        match status {
            PersistStatus::Durable => self.counters.durable += 1,
            PersistStatus::Unneeded => self.counters.unneeded += 1,
            PersistStatus::Refused => self.counters.refused += 1,
            PersistStatus::Started | PersistStatus::Busy => return false,
        }
        true
    }

    /// Splash `WriteBehind::nextWakeup`: when a step next has work no
    /// landing wakes the model for, the next point not in use falling due.
    /// None while copies wait for demand, once a step stopped because a
    /// claim waits or the demand was covered, and while the first point
    /// waits for a landing (no write budget holds one back).
    pub(crate) fn next_wakeup(
        &self,
        store: &HybridStateStore<PagedAnchor>,
        persistent: bool,
    ) -> Option<f64> {
        if !self.copying || matches!(self.stopped, Stop::ClaimWaiting | Stop::Covered) {
            return None;
        }
        let stalled = self.stopped == Stop::Stalled;
        self.waiting
            .iter()
            .find(|point| !in_use(store, persistent, point.id))
            .and_then(|point| (!stalled).then_some(point.due))
    }
}

/// A temporary tier's point whose turn has not ended.
fn in_use(store: &HybridStateStore<PagedAnchor>, persistent: bool, id: EntryId) -> bool {
    !persistent
        && store
            .get(id)
            .is_some_and(|entry| store.in_use(entry.mark()))
}

/// Splash `WriteBehind::published`: an ordinary point published or reused
/// `tokens` deep at `now`. Nothing while the tier takes no writes. While
/// copies wait for demand, the queue keeps only the points their
/// conversations still resume from.
pub(crate) fn published(states: &mut DFlash2States, id: EntryId, tokens: u32, now: f64) {
    let DFlash2States { store, disk, .. } = states;
    let Some(disk) = disk.as_mut() else {
        return;
    };
    let persistent = disk.tier.persistent;
    if !disk.writable()
        || tokens < MINIMUM_TOKENS
        || store.get(id).is_none_or(|entry| entry.checkpoint())
    {
        return;
    }
    let behind = &mut disk.write_behind;
    if behind.waiting.iter().any(|point| point.id == id) {
        return;
    }
    behind.copying = persistent || behind.demand_bytes != 0;
    if !behind.copying {
        let mut unneeded = 0;
        behind.waiting.retain(|point| {
            let keep = newest_point(store, point.id);
            unneeded += u64::from(!keep);
            keep
        });
        behind.counters.unneeded += unneeded;
    }
    behind.waiting.push_back(Point {
        id,
        due: now + DELAY_SECONDS,
    });
}

/// Splash `WriteBehind::run`: steps the points due at `now` and not in
/// use, in publication order, unless a claim waits for memory on its way
/// back. A temporary tier copies only while `demand_bytes` exceeds what
/// the model's units that free at once hold, plus what the copies in
/// flight free once they land. True when a point became durable or was
/// given up.
pub(crate) fn run<T: DFlash2Target>(
    target: &mut T,
    now: f64,
    demand_bytes: u64,
    claim_waiting: bool,
) -> bool {
    let Some(disk) = target.dflash2_states_mut().disk.as_mut() else {
        return false;
    };
    let persistent = disk.tier.persistent;
    let mut behind = std::mem::take(&mut disk.write_behind);
    behind.demand_bytes = demand_bytes;
    behind.copying = persistent || demand_bytes != 0;
    let settled = step(target, &mut behind, now, claim_waiting, persistent);
    if let Some(disk) = target.dflash2_states_mut().disk.as_mut() {
        disk.write_behind = behind;
    }
    settled
}

/// Splash `WriteBehind::flush` at a clean stop: a persistent tier copies
/// every waiting point, newest first, whether due, in use or demanded,
/// until each is durable or given up. False while a copy is in flight; a
/// temporary tier has nothing to keep.
pub(crate) fn flush<T: DFlash2Target>(target: &mut T) -> bool {
    let Some(disk) = target.dflash2_states_mut().disk.as_mut() else {
        return true;
    };
    if !disk.tier.persistent {
        return true;
    }
    let mut behind = std::mem::take(&mut disk.write_behind);
    let flushed = loop {
        let Some(point) = behind.waiting.back().copied() else {
            break true;
        };
        if !behind.settle(persist(target, point.id)) {
            break false;
        }
        behind.waiting.pop_back();
    };
    if let Some(disk) = target.dflash2_states_mut().disk.as_mut() {
        disk.write_behind = behind;
    }
    flushed
}

fn step<T: DFlash2Target>(
    target: &mut T,
    behind: &mut WriteBehind,
    now: f64,
    claim_waiting: bool,
    persistent: bool,
) -> bool {
    behind.stopped = Stop::None;
    if !behind.copying {
        return false;
    }
    if claim_waiting {
        behind.stopped = Stop::ClaimWaiting;
        return false;
    }
    let mut settled = false;
    let mut index = 0;
    while let Some(point) = behind.waiting.get(index).copied() {
        if point.due > now {
            break;
        }
        if in_use(&target.dflash2_states().store, persistent, point.id) {
            index += 1;
            continue;
        }
        if !persistent && !demanded(&*target, behind.demand_bytes) {
            behind.stopped = Stop::Covered;
            break;
        }
        if !behind.settle(persist(target, point.id)) {
            behind.stopped = Stop::Stalled;
            break;
        }
        behind.waiting.remove(index);
        settled = true;
    }
    settled
}

/// Splash `WriteBehind::demanded`: what the model's units give a claim
/// that cannot wait now, waiting for no write to land (the state the idle
/// staging buffer writes among them), counted as the ledger counts them,
/// plus what the copies ahead in flight free once they land, is less than
/// `demand_bytes`.
fn demanded<T: DFlash2Target>(target: &T, demand_bytes: u64) -> bool {
    let mut covered = copying_ahead_bytes(target.dflash2_states());
    if covered >= demand_bytes {
        return false;
    }
    let query = VictimQuery {
        rights: TakeRights {
            units: true,
            ..TakeRights::default()
        },
        need: Need {
            bytes: demand_bytes,
            ..Need::default()
        },
    };
    for unit in model_units(target, Keep::default(), &query).into_units() {
        if unit.later || matches!(unit.stage, Stage::Rebuild | Stage::InUse) {
            continue;
        }
        covered = covered.saturating_add(unit.bytes);
        if covered >= demand_bytes {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests;
