//! The process memory monitor, after Splash `MemoryPressureMonitor` and
//! `MemoryControl`: while the host is short or macOS warns, each registered
//! model's thread is woken to run its control pass, even when it is idle;
//! so is a model whose published wakeup fell due (Splash
//! `WriteBehind::nextWakeup`), and one whose SSD write landed.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use super::governor::Governor;

/// Splash `MemoryPressureMonitor`'s timer period.
pub(crate) const MONITOR_PERIOD_SECONDS: f64 = 0.5;

/// Sends a wake to a model thread; false once the thread is gone.
pub(crate) type Wake = Box<dyn Fn() -> bool + Send + Sync>;

struct Waker {
    model: u64,
    /// A wake is in flight: at most one per model.
    pending: Arc<AtomicBool>,
    wake: Wake,
    /// The governor-clock time the model thread asked to be woken at, as
    /// `f64` bits; 0 for none.
    next_wakeup: AtomicU64,
}

impl Waker {
    /// Sends the wake unless one is in flight; false once the thread is
    /// gone.
    fn send(&self) -> bool {
        if self.pending.swap(true, Ordering::AcqRel) {
            return true;
        }
        if (self.wake)() {
            return true;
        }
        self.pending.store(false, Ordering::Release);
        false
    }

    /// Takes the published wakeup when it is due at `now`.
    fn take_due(&self, now: f64) -> bool {
        let bits = self.next_wakeup.load(Ordering::Acquire);
        if bits == 0 || now < f64::from_bits(bits) {
            return false;
        }
        self.next_wakeup
            .compare_exchange(bits, 0, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }
}

fn wakers() -> &'static Mutex<Vec<Waker>> {
    static WAKERS: OnceLock<Mutex<Vec<Waker>>> = OnceLock::new();
    WAKERS.get_or_init(|| Mutex::new(Vec::new()))
}

fn lock() -> std::sync::MutexGuard<'static, Vec<Waker>> {
    wakers()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// `model`'s thread is woken by `wake` (it must not hold the thread's
/// command channel open). Starts the monitor on first use.
pub(crate) fn register_waker(model: u64, wake: Wake) {
    {
        let mut wakers = lock();
        wakers.retain(|waker| waker.model != model);
        wakers.push(Waker {
            model,
            pending: Arc::new(AtomicBool::new(false)),
            wake,
            next_wakeup: AtomicU64::new(0),
        });
    }
    static MONITOR: OnceLock<()> = OnceLock::new();
    MONITOR.get_or_init(|| {
        let spawned = std::thread::Builder::new()
            .name("mlx-memory-monitor".into())
            .spawn(|| {
                loop {
                    std::thread::sleep(Duration::from_secs_f64(MONITOR_PERIOD_SECONDS));
                    tick(super::governor());
                }
            });
        if let Err(error) = spawned {
            tracing::warn!(target: "mlx_core::memory", "memory monitor not started: {error}");
        }
    });
}

/// The model's thread took its wake: the next one may be sent.
pub(crate) fn wake_taken(model: u64) {
    if let Some(waker) = lock().iter().find(|waker| waker.model == model) {
        waker.pending.store(false, Ordering::Release);
    }
}

/// Wakes `model`'s thread now unless a wake is in flight (a tracked SSD
/// write landed: Splash `Engine::tick` polls its transfers).
pub(crate) fn wake(model: u64) {
    let wakers = lock();
    if let Some(waker) = wakers.iter().find(|waker| waker.model == model) {
        waker.send();
    }
}

/// The governor-clock time `model`'s thread wants its next tick at, `None`
/// for none (Splash `WriteBehind::nextWakeup`).
pub(crate) fn set_next_wakeup(model: u64, at: Option<f64>) {
    if let Some(waker) = lock().iter().find(|waker| waker.model == model) {
        let bits = at.map_or(0, |at| at.max(f64::MIN_POSITIVE).to_bits());
        waker.next_wakeup.store(bits, Ordering::Release);
    }
}

/// One monitor period: wakes every registered model whose wake is not in
/// flight while the host needs control, and each whose wakeup fell due;
/// forgets the threads that are gone.
pub(crate) fn tick(governor: &Governor) {
    let control = governor.needs_control();
    let now = governor.now();
    let models: Vec<u64> = lock().iter().map(|waker| waker.model).collect();
    let registered: Vec<u64> = models
        .into_iter()
        .filter(|&model| governor.is_registered(model))
        .collect();
    lock().retain(|waker| {
        if !registered.contains(&waker.model) {
            return true;
        }
        let due = waker.take_due(now);
        !(control || due) || waker.send()
    });
}

/// Drops `model`'s waker.
#[cfg(test)]
pub(crate) fn forget_waker(model: u64) {
    lock().retain(|waker| waker.model != model);
}

#[cfg(test)]
mod tests;
