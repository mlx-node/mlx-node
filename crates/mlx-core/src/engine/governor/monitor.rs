//! The process memory monitor, after Splash `MemoryPressureMonitor` and
//! `MemoryControl`: while the host is short or macOS warns, each registered
//! model's thread is woken to run its control pass, even when it is idle.

use std::sync::atomic::{AtomicBool, Ordering};
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

/// One monitor period: wakes every registered model whose wake is not in
/// flight while the host needs control; forgets the threads that are gone.
pub(crate) fn tick(governor: &Governor) {
    if !governor.needs_control() {
        return;
    }
    let models: Vec<u64> = lock().iter().map(|waker| waker.model).collect();
    let registered: Vec<u64> = models
        .into_iter()
        .filter(|&model| governor.is_registered(model))
        .collect();
    lock().retain(|waker| {
        if !registered.contains(&waker.model) || waker.pending.swap(true, Ordering::AcqRel) {
            return true;
        }
        if (waker.wake)() {
            return true;
        }
        waker.pending.store(false, Ordering::Release);
        false
    });
}
