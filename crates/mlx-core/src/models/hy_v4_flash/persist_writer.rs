//! The disk tier's threads beside the model thread (a copy of Flash Next's
//! `qwen4_exp/prefix_disk.rs` parker, wake and closer):
//!
//! ```text
//! model thread ─► host copies of blocks and segments ─► Slot.items (≤ QUEUE_BYTES)
//! parker       ─► offers them to the writer one at a time, measures the cache root
//!                 after a state write, drains once per plan, wakes the model thread
//!                 when the plan is due
//! closer       ─► any thread: the model thread writes the waiting states, the parker
//!                 hands over the copies, the tier drains and closes
//! ```
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use mlx_paged_attn::ColdObject;

use crate::disk_tier::{DiskTier, TierHandle};

/// The wait for a writer slot of one object on the parker thread (an object
/// that finds none stays first in line), for the staging buffer before a
/// read, and what a dispose gives the waiting states.
pub const SLOT_WAIT: Duration = Duration::from_secs(2);

/// Counters of the stored states and of their disk tier.
#[derive(Default)]
pub struct Telemetry {
    /// 0 off, 1 persistent, 2 temporary.
    pub disk_enabled: AtomicU64,
    /// States waiting for their write, the one in flight included.
    pub waiting: AtomicU64,
    /// States written whose whole chain is on disk.
    pub durable: AtomicU64,
    /// Waiting states not written: their conversation went on, or they are
    /// gone.
    pub unneeded: AtomicU64,
    /// States not written or dropped (chain incomplete, room, encoding, a
    /// failed write, a closed tier), and copies the writer refused.
    pub refused: AtomicU64,
    pub disk_restored: AtomicU64,
    /// States a persistent tier took back at load, on disk only.
    pub adopted: AtomicU64,
    /// States on disk removed because their chain was not.
    pub dropped_chain: AtomicU64,
    /// Blocks and segments copied out and not handed to the writer yet.
    pub owed: AtomicU64,
    pub state_bytes_written: AtomicU64,
}

impl Telemetry {
    pub fn bump(counter: &AtomicU64) {
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// The tier as a turn's row reports it: 0 off, 1 persistent, 2
    /// temporary; off once `tier` takes no write any more.
    pub(crate) fn mode(&self, tier: &TierHandle) -> u64 {
        match self.disk_enabled.load(Ordering::Relaxed) {
            mode if tier.is_open() => mode,
            _ => 0,
        }
    }
}

pub fn object_bytes(object: &ColdObject) -> u64 {
    match object {
        ColdObject::Block(block) => block
            .layers
            .iter()
            .map(|layer| (layer.keys.len() + layer.values.len()) as u64)
            .sum(),
        ColdObject::Sidecar(sidecar) => sidecar.tensors.iter().map(|t| t.len() as u64).sum(),
    }
}

/// What the model thread last left for the disk to do.
#[derive(Clone, Copy, Default)]
pub struct Plan {
    /// Bumped by every pass that stores a plan.
    pub epoch: u64,
    /// When the next waiting point or stuck chain is due.
    pub next_due: Option<Instant>,
    /// A chain is owed: blocks to copy, or a written state whose chain is
    /// not whole yet.
    pub owing: bool,
    /// The model thread's next pass that may copy.
    pub next_step: Option<Instant>,
}

#[derive(Default)]
pub struct Slot {
    /// Block and segment copies, first in line first.
    pub items: VecDeque<ColdObject>,
    pub item_bytes: u64,
    flush: u64,
    flushed: u64,
    flush_until: Option<Instant>,
    /// The last flush for which the thread waited out the copies it handed.
    drained: u64,
    pub stop: bool,
    pub plan: Plan,
    /// A close asks the model thread to write the waiting states by then.
    pub closing: Option<Instant>,
    /// The model thread's flush of the waiting states ended: true when no
    /// state waited any more and every copy and write it handed landed.
    pub points_flushed: Option<bool>,
    /// A state write went out: the parker measures the cache root.
    pub measure: bool,
    #[cfg(test)]
    pub measured: u64,
    /// The plan epoch whose wake went out.
    woke: u64,
    /// The plan epoch for which the writer was drained.
    drained_epoch: u64,
    #[cfg(test)]
    pub hold: bool,
    #[cfg(test)]
    pub slot_wait: Option<Duration>,
}

impl Slot {
    fn slot_wait(&self) -> Duration {
        #[cfg(test)]
        if let Some(wait) = self.slot_wait {
            return wait;
        }
        SLOT_WAIT
    }
}

pub struct Parker {
    slot: Mutex<Slot>,
    pub wake: Condvar,
    pub done: Condvar,
}

impl Parker {
    pub fn new() -> Self {
        Self {
            slot: Mutex::new(Slot::default()),
            wake: Condvar::new(),
            done: Condvar::new(),
        }
    }

    pub fn lock(&self) -> MutexGuard<'_, Slot> {
        self.slot
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    /// Hands every queued copy to the writer and waits until the writer is
    /// past them, until `deadline`. False when a copy was not handed over by
    /// then or the thread stopped first.
    pub fn flush(&self, deadline: Instant) -> bool {
        let mut slot = self.lock();
        slot.flush += 1;
        let target = slot.flush;
        slot.flush_until = Some(deadline);
        self.wake.notify_all();
        while slot.flushed < target && !slot.stop {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                break;
            }
            slot = self
                .done
                .wait_timeout(slot, left)
                .map(|(guard, _)| guard)
                .unwrap_or_else(|poison| poison.into_inner().0);
        }
        slot.flushed >= target && slot.items.is_empty()
    }
}

impl Default for Parker {
    fn default() -> Self {
        Self::new()
    }
}

/// Hands the queued copies to the writer one at a time, and measures the
/// cache root after a state write went out (that can remove another cache's
/// directory: never on the model thread). With no copy left and a chain
/// owed it drains the writer once per plan, then wakes the model thread once
/// per plan, at the earliest when the plan is due and the model thread's
/// next pass may copy.
pub(crate) fn run(
    parker: Arc<Parker>,
    tier: Arc<DiskTier>,
    telemetry: Arc<Telemetry>,
    wake: Arc<Wake>,
) {
    let manager = Arc::clone(&tier.manager);
    let mut slot = parker.lock();
    loop {
        if slot.stop {
            break;
        }
        if std::mem::take(&mut slot.measure) {
            drop(slot);
            tier.measure();
            slot = parker.lock();
            #[cfg(test)]
            {
                slot.measured += 1;
            }
            continue;
        }
        let now = Instant::now();
        let flushing = slot.flush > slot.flushed;
        let until = slot.flush_until.filter(|_| flushing);
        if until.is_some_and(|until| now >= until) {
            slot.flushed = slot.flush;
            parker.done.notify_all();
            continue;
        }
        #[cfg(test)]
        let held = slot.hold;
        #[cfg(not(test))]
        let held = false;
        if !held && let Some(object) = slot.items.pop_front() {
            let bytes = object_bytes(&object);
            let wait = now + slot.slot_wait();
            drop(slot);
            let deadline = until.map_or(wait, |until| until.min(wait));
            let sent = manager.offer(object, deadline);
            slot = parker.lock();
            match sent {
                // No writer slot freed: it stays first in line, so the chain
                // keeps no hole.
                Ok(Some(object)) => slot.items.push_front(object),
                Ok(None) => {
                    slot.item_bytes = slot.item_bytes.saturating_sub(bytes);
                    telemetry.owed.fetch_sub(1, Ordering::Relaxed);
                }
                Err(_) => {
                    slot.item_bytes = slot.item_bytes.saturating_sub(bytes);
                    telemetry.owed.fetch_sub(1, Ordering::Relaxed);
                    Telemetry::bump(&telemetry.refused);
                }
            }
            continue;
        }
        if flushing && slot.items.is_empty() {
            if slot.drained < slot.flush {
                slot.drained = slot.flush;
                drop(slot);
                if let Some(until) = until {
                    manager.drain(until.saturating_duration_since(Instant::now()));
                }
                slot = parker.lock();
                continue;
            }
            slot.flushed = slot.flush;
            parker.done.notify_all();
        }
        let plan = slot.plan;
        if slot.items.is_empty() && plan.owing && slot.drained_epoch != plan.epoch {
            slot.drained_epoch = plan.epoch;
            drop(slot);
            manager.drain(SLOT_WAIT);
            slot = parker.lock();
            continue;
        }
        let mut wait = Duration::from_secs(3600);
        let want = if plan.owing { Some(now) } else { plan.next_due };
        if let Some(want) = want
            && slot.woke != plan.epoch
        {
            let at = plan.next_step.map_or(want, |step| step.max(want));
            if now >= at {
                slot.woke = plan.epoch;
                drop(slot);
                wake.fire();
                slot = parker.lock();
                continue;
            }
            wait = wait.min(at - now);
        }
        if let Some(until) = until {
            wait = wait.min(until.saturating_duration_since(now));
        }
        slot = parker
            .wake
            .wait_timeout(slot, wait.max(Duration::from_millis(1)))
            .map(|(guard, _)| guard)
            .unwrap_or_else(|poison| poison.into_inner().0);
    }
}

type WakeCall = Arc<dyn Fn() + Send + Sync>;

/// The call that wakes the model thread for its disk work. A load installs
/// a weak command sender: it holds neither the closer nor the tier.
#[derive(Default)]
pub struct Wake {
    call: Mutex<Option<WakeCall>>,
}

impl Wake {
    fn slot(&self) -> MutexGuard<'_, Option<WakeCall>> {
        self.call
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    fn is_set(&self) -> bool {
        self.slot().is_some()
    }

    pub fn fire(&self) {
        let call = self.slot().clone();
        if let Some(call) = call {
            call();
        }
    }
}

/// Closes the tier from any thread: the model thread writes the waiting
/// states, the parker hands over the queued copies, then the tier drains
/// its writer and takes no write after.
pub struct Closer {
    pub parker: Arc<Parker>,
    pub(crate) tier: Arc<DiskTier>,
    pub telemetry: Arc<Telemetry>,
    pub wake: Arc<Wake>,
    /// A close waits for the model thread's flush of the waiting states.
    pub closing: AtomicBool,
    /// The one close's outcome once it ended; `Some(None)` while it runs.
    outcome: Mutex<Option<Option<bool>>>,
    ended: Condvar,
}

impl Closer {
    pub(crate) fn new(
        parker: Arc<Parker>,
        tier: Arc<DiskTier>,
        telemetry: Arc<Telemetry>,
        wake: Arc<Wake>,
    ) -> Self {
        Self {
            parker,
            tier,
            telemetry,
            wake,
            closing: AtomicBool::new(false),
            outcome: Mutex::new(None),
            ended: Condvar::new(),
        }
    }

    /// The call that wakes the model thread (a weak command sender).
    pub fn set_wake(&self, call: WakeCall) {
        *self.wake.slot() = Some(call);
    }

    /// A persistent tier: the model thread writes the waiting states newest
    /// first, given all of the time but the last second (at most half of
    /// it), then every queued copy goes to the writer and the tier closes,
    /// by `deadline`; false when the drain missed it. A temporary tier
    /// keeps nothing: its writer stops and its directory goes. One close
    /// runs: a call while it runs waits for its outcome until its own
    /// deadline (false when it has none by then), and a later call gets the
    /// same outcome.
    pub fn close(&self, deadline: Instant) -> bool {
        self.once(deadline, true)
    }

    /// [`Self::close`] without the model thread's flush: the model thread
    /// calls it itself.
    pub fn finish(&self, deadline: Instant) -> bool {
        self.once(deadline, false)
    }

    fn once(&self, deadline: Instant, ask_model: bool) -> bool {
        {
            let mut outcome = self
                .outcome
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            if outcome.is_some() {
                loop {
                    if let Some(Some(landed)) = *outcome {
                        return landed;
                    }
                    let left = deadline.saturating_duration_since(Instant::now());
                    if left.is_zero() {
                        return false;
                    }
                    outcome = self
                        .ended
                        .wait_timeout(outcome, left)
                        .map(|(guard, _)| guard)
                        .unwrap_or_else(|poison| poison.into_inner().0);
                }
            }
            *outcome = Some(None);
        }
        let mut flushed = true;
        if self.tier.persistent && self.tier.is_open() {
            if ask_model && self.wake.is_set() {
                flushed = self.flush_on_model_thread(deadline);
            } else if !ask_model {
                flushed = self.parker.lock().points_flushed != Some(false);
            }
        }
        if self.tier.persistent {
            flushed &= self.parker.flush(deadline);
        }
        let landed = self.tier.close(deadline) && flushed;
        self.closing.store(false, Ordering::Release);
        self.telemetry.disk_enabled.store(0, Ordering::Relaxed);
        *self
            .outcome
            .lock()
            .unwrap_or_else(|poison| poison.into_inner()) = Some(Some(landed));
        self.ended.notify_all();
        landed
    }

    /// Wakes the model thread to write its waiting states and waits for it
    /// until the last second of `deadline` (at most half of what is left).
    /// True when its flush left no state waiting and all it handed landed.
    fn flush_on_model_thread(&self, deadline: Instant) -> bool {
        let left = deadline.saturating_duration_since(Instant::now());
        let until = deadline
            .checked_sub((left / 2).min(Duration::from_secs(1)))
            .unwrap_or(deadline);
        {
            let mut slot = self.parker.lock();
            if slot.stop {
                return false;
            }
            slot.closing = Some(until);
            slot.points_flushed = None;
        }
        self.closing.store(true, Ordering::Release);
        self.wake.fire();
        let mut slot = self.parker.lock();
        while slot.points_flushed.is_none() && !slot.stop {
            let left = until.saturating_duration_since(Instant::now());
            if left.is_zero() {
                break;
            }
            slot = self
                .parker
                .done
                .wait_timeout(slot, left)
                .map(|(guard, _)| guard)
                .unwrap_or_else(|poison| poison.into_inner().0);
        }
        slot.points_flushed == Some(true)
    }
}
