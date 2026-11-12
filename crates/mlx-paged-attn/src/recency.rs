use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// One saturating use clock. Every cache that shares it orders its entries
/// on one timeline, so "older" means the same thing across them.
#[derive(Debug, Default)]
pub struct CacheRecency(AtomicU64);

impl CacheRecency {
    pub fn new() -> Self {
        Self::default()
    }

    /// The next stamp: strictly above every earlier one until the clock
    /// saturates at `u64::MAX`.
    pub fn next(&self) -> u64 {
        let mut current = self.0.load(Relaxed);
        loop {
            let next = current.saturating_add(1);
            match self
                .0
                .compare_exchange_weak(current, next, Relaxed, Relaxed)
            {
                Ok(_) => return next,
                Err(seen) => current = seen,
            }
        }
    }
}
