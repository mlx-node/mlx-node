//! Per-request streaming progress, written by the model thread and read by
//! JS through [`crate::engine::types::ChatStreamHandle::progress`].
//!
//! One [`TurnProgress`] lives behind an `Arc` shared by the handle and the
//! request's [`crate::model_thread::StreamTx`]. The model thread is the only
//! writer; every access is `Relaxed`, so a reader may see fields from
//! slightly different moments.
//!
//! Code that holds the sink reaches the progress through
//! [`crate::engine::backend::ChunkSink::progress`]. Whole-turn prefill loops
//! that cannot see the sink use the thread-local installed by
//! [`install_turn_progress`] for the duration of one whole-turn streaming
//! command. The scheduled lane never reads the thread-local.

use std::cell::RefCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU32, AtomicU64, Ordering};
use std::time::Instant;

use crate::engine::types::ChatStreamProgress;

const PHASE_QUEUED: u8 = 0;
const PHASE_PREFILL: u8 = 1;
const PHASE_DECODE: u8 = 2;
const PHASE_DONE: u8 = 3;

/// `prefill_total` / `prefill_done` / `cached_tokens` before they are known.
const UNKNOWN: u32 = u32::MAX;
/// `prefill_start_us` / `first_token_us` before they are set.
const UNSET: u64 = 0;

#[derive(Debug)]
pub(crate) struct TurnProgress {
    created: Instant,
    phase: AtomicU8,
    prefill_total: AtomicU32,
    prefill_done: AtomicU32,
    cached_tokens: AtomicU32,
    generated: AtomicU32,
    /// Microseconds since `created`, at least 1 once set.
    prefill_start_us: AtomicU64,
    /// Microseconds since `created`, at least 1 once set.
    first_token_us: AtomicU64,
}

impl Default for TurnProgress {
    fn default() -> Self {
        Self::new()
    }
}

impl TurnProgress {
    pub(crate) fn new() -> Self {
        Self {
            created: Instant::now(),
            phase: AtomicU8::new(PHASE_QUEUED),
            prefill_total: AtomicU32::new(UNKNOWN),
            prefill_done: AtomicU32::new(UNKNOWN),
            cached_tokens: AtomicU32::new(UNKNOWN),
            generated: AtomicU32::new(0),
            prefill_start_us: AtomicU64::new(UNSET),
            first_token_us: AtomicU64::new(UNSET),
        }
    }

    fn now_us(&self) -> u64 {
        (self.created.elapsed().as_micros() as u64).max(1)
    }

    fn phase(&self) -> u8 {
        self.phase.load(Ordering::Relaxed)
    }

    /// The model thread picked the request up; prefill totals are not known
    /// yet.
    pub(crate) fn begin_turn(&self) {
        if self.phase() != PHASE_QUEUED {
            return;
        }
        self.prefill_start_us
            .store(self.now_us(), Ordering::Relaxed);
        self.phase.store(PHASE_PREFILL, Ordering::Relaxed);
    }

    /// Prefill of `total` tokens starts on top of `cached` reused prefix
    /// tokens. Ignored once decode has started.
    pub(crate) fn begin_prefill(&self, total: u32, cached: u32) {
        if self.phase() >= PHASE_DECODE {
            return;
        }
        self.prefill_total.store(total, Ordering::Relaxed);
        self.prefill_done.store(0, Ordering::Relaxed);
        self.cached_tokens.store(cached, Ordering::Relaxed);
        self.prefill_start_us
            .store(self.now_us(), Ordering::Relaxed);
        self.phase.store(PHASE_PREFILL, Ordering::Relaxed);
    }

    /// `done` of the `total` prefill tokens are materialized. Clamped to the
    /// total; a smaller value than before (preemption replay) is kept.
    pub(crate) fn prefill_advanced(&self, done: u32) {
        if self.phase() != PHASE_PREFILL {
            return;
        }
        let total = self.prefill_total.load(Ordering::Relaxed);
        if total == UNKNOWN {
            return;
        }
        self.prefill_done.store(done.min(total), Ordering::Relaxed);
    }

    /// Prefill finished and the first token is sampled. Idempotent.
    pub(crate) fn first_token(&self) {
        if self.first_token_us.load(Ordering::Relaxed) == UNSET {
            self.first_token_us.store(self.now_us(), Ordering::Relaxed);
        }
        let total = self.prefill_total.load(Ordering::Relaxed);
        if total != UNKNOWN {
            self.prefill_done.store(total, Ordering::Relaxed);
        }
        if self.phase() < PHASE_DECODE {
            self.phase.store(PHASE_DECODE, Ordering::Relaxed);
        }
    }

    /// `n` more tokens were kept and streamed.
    pub(crate) fn add_generated(&self, n: u32) {
        self.first_token();
        self.generated.fetch_add(n, Ordering::Relaxed);
    }

    /// The terminal chunk or error was sent.
    pub(crate) fn finish(&self) {
        self.phase.store(PHASE_DONE, Ordering::Relaxed);
    }

    pub(crate) fn snapshot(&self) -> ChatStreamProgress {
        let phase = self.phase();
        let now_us = self.now_us();
        let known = |v: u32| (v != UNKNOWN).then_some(v);
        let prefill_start = self.prefill_start_us.load(Ordering::Relaxed);
        let first_token = self.first_token_us.load(Ordering::Relaxed);
        let ms = |us: u64| us as f64 / 1000.0;
        let prefill_elapsed_ms = (prefill_start != UNSET).then(|| {
            let end = if first_token != UNSET {
                first_token
            } else {
                now_us
            };
            ms(end.saturating_sub(prefill_start))
        });
        let decode_elapsed_ms =
            (first_token != UNSET).then(|| ms(now_us.saturating_sub(first_token)));
        ChatStreamProgress {
            phase: match phase {
                PHASE_QUEUED => "queued",
                PHASE_PREFILL => "prefill",
                PHASE_DECODE => "decode",
                _ => "done",
            }
            .to_string(),
            prefill_total: known(self.prefill_total.load(Ordering::Relaxed)),
            prefill_done: known(self.prefill_done.load(Ordering::Relaxed)),
            cached_tokens: known(self.cached_tokens.load(Ordering::Relaxed)),
            generated_tokens: (phase >= PHASE_DECODE)
                .then(|| self.generated.load(Ordering::Relaxed)),
            prefill_elapsed_ms,
            decode_elapsed_ms,
        }
    }
}

thread_local! {
    static TURN_PROGRESS: RefCell<Option<Arc<TurnProgress>>> = const { RefCell::new(None) };
}

/// Restores the previously installed turn progress on drop.
#[must_use]
pub(crate) struct TurnProgressGuard {
    previous: Option<Arc<TurnProgress>>,
}

impl Drop for TurnProgressGuard {
    fn drop(&mut self) {
        let previous = self.previous.take();
        TURN_PROGRESS.with(|slot| *slot.borrow_mut() = previous);
    }
}

/// Install `progress` as this thread's whole-turn progress until the guard
/// drops, and mark the turn as picked up.
pub(crate) fn install_turn_progress(progress: Option<Arc<TurnProgress>>) -> TurnProgressGuard {
    if let Some(progress) = &progress {
        progress.begin_turn();
    }
    let previous = TURN_PROGRESS.with(|slot| slot.replace(progress));
    TurnProgressGuard { previous }
}

/// Run `f` on this thread's whole-turn progress, if one is installed.
pub(crate) fn with_turn_progress(f: impl FnOnce(&TurnProgress)) {
    TURN_PROGRESS.with(|slot| {
        if let Some(progress) = slot.borrow().as_deref() {
            f(progress);
        }
    });
}

#[cfg(test)]
pub(crate) fn turn_progress_installed() -> bool {
    TURN_PROGRESS.with(|slot| slot.borrow().is_some())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_progress_is_queued_with_nothing_known() {
        let p = TurnProgress::new();
        let s = p.snapshot();
        assert_eq!(s.phase, "queued");
        assert_eq!(s.prefill_total, None);
        assert_eq!(s.prefill_done, None);
        assert_eq!(s.cached_tokens, None);
        assert_eq!(s.generated_tokens, None);
        assert_eq!(s.prefill_elapsed_ms, None);
        assert_eq!(s.decode_elapsed_ms, None);
    }

    #[test]
    fn begin_turn_starts_prefill_without_totals() {
        let p = TurnProgress::new();
        p.begin_turn();
        let s = p.snapshot();
        assert_eq!(s.phase, "prefill");
        assert_eq!(s.prefill_total, None);
        assert!(s.prefill_elapsed_ms.is_some());
    }

    #[test]
    fn prefill_then_decode_lifecycle() {
        let p = TurnProgress::new();
        // No totals yet: an advance has nothing to clamp against.
        p.prefill_advanced(5);
        assert_eq!(p.snapshot().prefill_done, None);

        p.begin_prefill(100, 40);
        let s = p.snapshot();
        assert_eq!(s.phase, "prefill");
        assert_eq!(
            (s.prefill_total, s.prefill_done, s.cached_tokens),
            (Some(100), Some(0), Some(40))
        );

        p.prefill_advanced(64);
        assert_eq!(p.snapshot().prefill_done, Some(64));
        p.prefill_advanced(500);
        assert_eq!(p.snapshot().prefill_done, Some(100), "clamped to total");
        p.prefill_advanced(32);
        assert_eq!(p.snapshot().prefill_done, Some(32), "may move back");

        p.first_token();
        let s = p.snapshot();
        assert_eq!(s.phase, "decode");
        assert_eq!(s.prefill_done, Some(100), "first token completes prefill");
        assert_eq!(s.generated_tokens, Some(0));
        assert!(s.decode_elapsed_ms.is_some());

        p.add_generated(1);
        p.add_generated(2);
        assert_eq!(p.snapshot().generated_tokens, Some(3));

        // Decode freezes the prefill counters.
        p.begin_prefill(7, 0);
        p.prefill_advanced(1);
        let s = p.snapshot();
        assert_eq!(s.phase, "decode");
        assert_eq!((s.prefill_total, s.prefill_done), (Some(100), Some(100)));

        p.finish();
        let s = p.snapshot();
        assert_eq!(s.phase, "done");
        assert_eq!(s.generated_tokens, Some(3));
    }

    #[test]
    fn add_generated_without_first_token_switches_to_decode() {
        let p = TurnProgress::new();
        p.begin_turn();
        p.add_generated(1);
        let s = p.snapshot();
        assert_eq!(s.phase, "decode");
        assert_eq!(s.generated_tokens, Some(1));
        assert!(s.decode_elapsed_ms.is_some());
    }

    #[test]
    fn guard_restores_previous_install_on_drop() {
        assert!(!turn_progress_installed());
        let outer = Arc::new(TurnProgress::new());
        {
            let _outer = install_turn_progress(Some(Arc::clone(&outer)));
            assert_eq!(outer.snapshot().phase, "prefill");
            {
                let _none = install_turn_progress(None);
                assert!(!turn_progress_installed());
            }
            assert!(turn_progress_installed());
            with_turn_progress(|p| p.begin_prefill(9, 1));
        }
        assert!(!turn_progress_installed());
        assert_eq!(outer.snapshot().prefill_total, Some(9));
        // Nothing installed: a write is a no-op.
        with_turn_progress(|_| panic!("no progress may be installed"));
    }
}
