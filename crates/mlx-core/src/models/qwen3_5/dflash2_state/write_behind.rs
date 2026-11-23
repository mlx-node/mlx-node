//! Copies of the newest restore points on the SSD tier while they stay in
//! RAM, ahead of need (Splash `WriteBehind`): thin Qwen3.5 adapter over the
//! shared [`crate::disk_tier::state_disk::WriteBehind`] scheduler.

use crate::disk_tier::state_disk::point_in_use;
use crate::engine::governor::ledger::{Need, Stage, TakeRights, VictimQuery};
use crate::models::qwen3_5::dflash2_turn::{DFlash2Target, Keep, model_units};

use super::disk::{copying_ahead_bytes, persist};
use super::paged::DFlash2States;
use super::store::EntryId;

pub(crate) use crate::disk_tier::state_disk::WriteBehind;
#[cfg(test)]
pub(crate) use crate::disk_tier::state_disk::{
    DELAY_SECONDS, MINIMUM_TOKENS, Stop, WriteBehindSnapshot,
};

/// Splash `WriteBehind::published`: an ordinary point published or reused
/// `tokens` deep at `now`. Nothing while the tier takes no writes.
pub(crate) fn published(states: &mut DFlash2States, id: EntryId, tokens: u32, now: f64) {
    let DFlash2States { store, disk, .. } = states;
    let Some(disk) = disk.as_mut() else {
        return;
    };
    let writable = disk.writable();
    let persistent = disk.tier.persistent;
    disk.write_behind
        .record_published(store, writable, persistent, id, tokens, now);
}

/// Splash `WriteBehind::run`: steps the points due at `now` and not in use,
/// in publication order, unless a claim waits for memory on its way back.
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
    let settled = behind.step(
        target,
        now,
        demand_bytes,
        claim_waiting,
        persistent,
        |target, id| point_in_use(&target.dflash2_states().store, persistent, id),
        |target, bytes| demanded(target, bytes),
        |target, id| persist(target, id),
    );
    if let Some(disk) = target.dflash2_states_mut().disk.as_mut() {
        disk.write_behind = behind;
    }
    settled
}

/// Splash `WriteBehind::flush` at a clean stop: a persistent tier copies
/// every waiting point, newest first, whether due, in use or demanded,
/// until each is durable or given up.
pub(crate) fn flush<T: DFlash2Target>(target: &mut T) -> bool {
    let Some(disk) = target.dflash2_states_mut().disk.as_mut() else {
        return true;
    };
    if !disk.tier.persistent {
        return true;
    }
    let mut behind = std::mem::take(&mut disk.write_behind);
    let flushed = behind.flush(target, |target, id| persist(target, id));
    if let Some(disk) = target.dflash2_states_mut().disk.as_mut() {
        disk.write_behind = behind;
    }
    flushed
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
