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
//! command. The scheduled lane reads the thread-local only during its prefix
//! prepare ([`install_prepare_progress`]).
//!
//! Before its prefill a turn is `queued` (the thread works on something
//! else), `slot` (the scheduler could not start it for lanes or memory) or
//! `prepare` (the thread renders it, matches its prefix, restores or replays
//! its state). From `prefill` on the phase never moves back.

use std::cell::RefCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, AtomicU8, AtomicU32, AtomicU64, Ordering};
use std::time::Instant;

use crate::engine::types::ChatStreamProgress;

const PHASE_QUEUED: u8 = 0;
const PHASE_SLOT: u8 = 1;
const PHASE_PREPARE: u8 = 2;
const PHASE_PREFILL: u8 = 3;
const PHASE_DECODE: u8 = 4;
const PHASE_DONE: u8 = 5;

/// `prefill_total` / `prefill_done` / `cached_tokens` before they are known.
const UNKNOWN: u32 = u32::MAX;
/// A microsecond stamp before it is set.
const UNSET: u64 = 0;

/// What a turn waits on before its prefill.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Waiting {
    /// The thread runs other work.
    Queued,
    /// Lanes are full, a start ahead of it is held, or memory refused it.
    Slot,
    /// The thread prepares this turn.
    Prepare,
}

impl Waiting {
    fn phase(self) -> u8 {
        match self {
            Self::Queued => PHASE_QUEUED,
            Self::Slot => PHASE_SLOT,
            Self::Prepare => PHASE_PREPARE,
        }
    }
}

/// One waiting kind's time in one word, so a reader never pairs a closed sum
/// with the start of a span it already holds: `2·closed` while no span of
/// the kind runs, `2·(closed − start) + 1` while one opened at `start` runs
/// (microseconds).
#[derive(Debug, Default)]
struct WaitTime(AtomicI64);

impl WaitTime {
    fn open(&self, now: u64) {
        let word = self.0.load(Ordering::Relaxed);
        if word & 1 == 0 {
            self.0
                .store((((word >> 1) - now as i64) << 1) | 1, Ordering::Relaxed);
        }
    }

    fn close(&self, now: u64) {
        let word = self.0.load(Ordering::Relaxed);
        if word & 1 == 1 {
            self.0
                .store(((word >> 1) + now as i64) << 1, Ordering::Relaxed);
        }
    }

    fn read(&self, now: u64) -> u64 {
        let word = self.0.load(Ordering::Relaxed);
        let base = word >> 1;
        let us = if word & 1 == 1 {
            base + now as i64
        } else {
            base
        };
        us.max(0) as u64
    }
}

#[derive(Debug)]
pub(crate) struct TurnProgress {
    created: Instant,
    phase: AtomicU8,
    prefill_total: AtomicU32,
    prefill_done: AtomicU32,
    cached_tokens: AtomicU32,
    generated: AtomicU32,
    /// Microseconds since `created`, at least 1 once set.
    picked_up_us: AtomicU64,
    slot: WaitTime,
    prepare: WaitTime,
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
            picked_up_us: AtomicU64::new(UNSET),
            slot: WaitTime::default(),
            prepare: WaitTime::default(),
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

    /// The model thread took the request off its channel. Only the first
    /// call counts: the wait ends there.
    pub(crate) fn picked_up(&self) {
        if self.picked_up_us.load(Ordering::Relaxed) == UNSET {
            self.picked_up_us.store(self.now_us(), Ordering::Relaxed);
        }
    }

    fn wait_time(&self, phase: u8) -> Option<&WaitTime> {
        match phase {
            PHASE_SLOT => Some(&self.slot),
            PHASE_PREPARE => Some(&self.prepare),
            _ => None,
        }
    }

    /// Ends the open waiting span at `now`, adding it to the sum of its kind.
    fn close_span(&self, now: u64) {
        if let Some(time) = self.wait_time(self.phase()) {
            time.close(now);
        }
    }

    /// The thread now does `next` for this turn. A no-op from `prefill` on.
    pub(crate) fn enter(&self, next: Waiting) {
        if self.phase() >= PHASE_PREFILL {
            return;
        }
        self.picked_up();
        let now = self.now_us();
        self.close_span(now);
        if let Some(time) = self.wait_time(next.phase()) {
            time.open(now);
        }
        self.phase.store(next.phase(), Ordering::Relaxed);
    }

    /// `cached` prefix tokens are being restored. Kept only while preparing.
    pub(crate) fn restoring(&self, cached: u32) {
        if self.phase() == PHASE_PREPARE {
            self.cached_tokens.store(cached, Ordering::Relaxed);
        }
    }

    /// A whole turn starts: the prefill clock runs from here, prefill
    /// totals are not known yet.
    pub(crate) fn begin_turn(&self) {
        self.picked_up();
        if self.phase() >= PHASE_PREFILL {
            return;
        }
        let now = self.now_us();
        self.close_span(now);
        self.prefill_start_us.store(now, Ordering::Relaxed);
        self.phase.store(PHASE_PREFILL, Ordering::Relaxed);
    }

    /// Prefill of `total` tokens starts on top of `cached` reused prefix
    /// tokens. Ignored once decode has started. The prefill clock starts at
    /// the first call that brings the totals; a later call (a preemption
    /// replay) keeps it.
    pub(crate) fn begin_prefill(&self, total: u32, cached: u32) {
        if self.phase() >= PHASE_DECODE {
            return;
        }
        let now = self.now_us();
        if self.phase() < PHASE_PREFILL {
            self.picked_up();
            self.close_span(now);
        }
        let first = self.prefill_total.load(Ordering::Relaxed) == UNKNOWN;
        self.prefill_total.store(total, Ordering::Relaxed);
        self.prefill_done.store(0, Ordering::Relaxed);
        self.cached_tokens.store(cached, Ordering::Relaxed);
        if first {
            self.prefill_start_us.store(now, Ordering::Relaxed);
        }
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
        self.picked_up();
        if self.first_token_us.load(Ordering::Relaxed) == UNSET {
            self.first_token_us.store(self.now_us(), Ordering::Relaxed);
        }
        let total = self.prefill_total.load(Ordering::Relaxed);
        if total != UNKNOWN {
            self.prefill_done.store(total, Ordering::Relaxed);
        }
        if self.phase() < PHASE_DECODE {
            if self.phase() < PHASE_PREFILL {
                self.close_span(self.now_us());
            }
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
        self.picked_up();
        if self.phase() < PHASE_PREFILL {
            self.close_span(self.now_us());
        }
        self.phase.store(PHASE_DONE, Ordering::Relaxed);
    }

    pub(crate) fn snapshot(&self) -> ChatStreamProgress {
        let phase = self.phase();
        let now_us = self.now_us();
        let known = |v: u32| (v != UNKNOWN).then_some(v);
        let prefill_start = self.prefill_start_us.load(Ordering::Relaxed);
        let first_token = self.first_token_us.load(Ordering::Relaxed);
        let ms = |us: u64| us as f64 / 1000.0;
        let picked_up = self.picked_up_us.load(Ordering::Relaxed);
        let wait_ms = ms(if picked_up != UNSET {
            picked_up
        } else {
            now_us
        });
        let waited = |time: &WaitTime| (picked_up != UNSET).then(|| ms(time.read(now_us)));
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
        let prefill_total = known(self.prefill_total.load(Ordering::Relaxed));
        // Relaxed loads can pair a new (smaller) total with an older done.
        let prefill_done = known(self.prefill_done.load(Ordering::Relaxed))
            .map(|done| prefill_total.map_or(done, |total| done.min(total)));
        ChatStreamProgress {
            phase: match phase {
                PHASE_QUEUED => "queued",
                PHASE_SLOT => "slot",
                PHASE_PREPARE => "prepare",
                PHASE_PREFILL => "prefill",
                PHASE_DECODE => "decode",
                _ => "done",
            }
            .to_string(),
            wait_ms,
            slot_ms: waited(&self.slot),
            prepare_ms: waited(&self.prepare),
            prefill_total,
            prefill_done,
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

/// Install `progress` as this thread's progress until the guard drops,
/// without starting its turn: the scheduled lane's prefix prepare, whose
/// family code marks what it restores.
pub(crate) fn install_prepare_progress(progress: Option<Arc<TurnProgress>>) -> TurnProgressGuard {
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

    fn sleep_ms(ms: u64) {
        std::thread::sleep(std::time::Duration::from_millis(ms));
    }

    /// P-T1: queued -> slot -> prepare -> queued -> prepare -> prefill ->
    /// decode. The wait freezes at pickup, each kind sums its own spans (the
    /// open one included), nothing after `prefill` moves the phase or the
    /// sums back, and a second `begin_prefill` keeps the first start.
    /// Mutation P1 (`enter` adds the span to the new phase's sum) -> red.
    #[test]
    fn waiting_phases_sum_their_own_spans_and_prefill_is_final() {
        let p = TurnProgress::new();
        sleep_ms(5);
        let s = p.snapshot();
        assert_eq!(s.phase, "queued");
        assert_eq!((s.slot_ms, s.prepare_ms), (None, None), "not picked up");
        assert!(s.wait_ms >= 5.0, "{}", s.wait_ms);

        p.picked_up();
        let wait = p.snapshot().wait_ms;
        sleep_ms(10);
        let s = p.snapshot();
        assert_eq!(s.wait_ms, wait, "the wait is frozen at pickup");
        assert_eq!((s.slot_ms, s.prepare_ms), (Some(0.0), Some(0.0)));
        p.picked_up();
        assert_eq!(p.snapshot().wait_ms, wait, "only the first pickup counts");

        p.enter(Waiting::Slot);
        sleep_ms(30);
        let open = p.snapshot();
        assert_eq!(open.phase, "slot");
        assert!(open.slot_ms.is_some_and(|ms| ms >= 30.0), "{open:?}");
        sleep_ms(30);
        let first_prepare = std::time::Instant::now();
        p.enter(Waiting::Prepare);
        sleep_ms(10);
        p.enter(Waiting::Queued);
        let mut prepare_wall = first_prepare.elapsed();
        assert_eq!(p.snapshot().phase, "queued");
        sleep_ms(40);
        let second_prepare = std::time::Instant::now();
        p.enter(Waiting::Prepare);
        sleep_ms(10);
        p.restoring(48);
        let s = p.snapshot();
        assert_eq!(s.phase, "prepare");
        assert_eq!(s.cached_tokens, Some(48), "restoring is kept in prepare");
        p.begin_prefill(100, 48);
        prepare_wall += second_prepare.elapsed();

        let s = p.snapshot();
        assert_eq!(s.phase, "prefill");
        let slot = s.slot_ms.expect("picked up");
        let prepare = s.prepare_ms.expect("picked up");
        assert!(slot >= 60.0, "slot holds its own span: {s:?}");
        assert!(prepare >= 20.0, "two prepare spans add up: {s:?}");
        // A sleep can run long on a loaded machine, so the bound is this test's own clock around the two
        // prepare spans, not the length of another span. 1 ms covers the microsecond rounding of the sums.
        assert!(
            prepare <= prepare_wall.as_secs_f64() * 1000.0 + 1.0,
            "the queued span and the slot span are not prepare time: {s:?}"
        );
        assert_eq!(s.wait_ms, wait);

        p.enter(Waiting::Slot);
        sleep_ms(5);
        p.enter(Waiting::Prepare);
        p.restoring(7);
        p.enter(Waiting::Queued);
        let after = p.snapshot();
        assert_eq!(after.phase, "prefill", "never back from prefill");
        assert_eq!(
            (after.slot_ms, after.prepare_ms),
            (Some(slot), Some(prepare))
        );
        assert_eq!(after.cached_tokens, Some(48));

        let first = after.prefill_elapsed_ms.expect("prefill started");
        sleep_ms(10);
        p.begin_prefill(80, 20);
        let replay = p.snapshot();
        assert!(
            replay
                .prefill_elapsed_ms
                .is_some_and(|ms| ms >= first + 10.0),
            "a second begin_prefill keeps the first start: {first} vs {replay:?}"
        );
        assert_eq!(
            (replay.prefill_total, replay.cached_tokens),
            (Some(80), Some(20))
        );

        p.first_token();
        p.finish();
        let done = p.snapshot();
        assert_eq!(done.phase, "done");
        assert_eq!((done.slot_ms, done.prepare_ms), (Some(slot), Some(prepare)));
    }

    /// A turn that ends before the thread took it reads `done` with its
    /// wait frozen at the end. Mutation pf (`finish` marks no pickup) -> red.
    #[test]
    fn a_turn_ended_before_pickup_freezes_its_wait() {
        let p = TurnProgress::new();
        sleep_ms(5);
        p.finish();
        let done = p.snapshot();
        assert_eq!(done.phase, "done");
        sleep_ms(10);
        assert_eq!(p.snapshot().wait_ms, done.wait_ms);
    }

    /// A reader on another thread never counts a span twice: while the
    /// model thread moves between slot, prepare and queued as fast as it
    /// can, the slot and prepare time read stays within the turn's life.
    /// Mutation pt (a close writes the new sum before it clears the open
    /// mark) -> red.
    #[test]
    fn a_reader_never_counts_a_span_twice() {
        use std::sync::atomic::AtomicBool;
        use std::time::Duration;
        let p = Arc::new(TurnProgress::new());
        p.picked_up();
        let stop = Arc::new(AtomicBool::new(false));
        let writer = {
            let p = Arc::clone(&p);
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || {
                let kinds = [Waiting::Slot, Waiting::Prepare, Waiting::Queued];
                let mut i = 0;
                while !stop.load(Ordering::Relaxed) {
                    p.enter(kinds[i % 3]);
                    i += 1;
                }
            })
        };
        let started = Instant::now();
        let mut reads = 0u64;
        while started.elapsed() < Duration::from_millis(300) {
            let s = p.snapshot();
            let life = p.created.elapsed().as_secs_f64() * 1000.0;
            let waited = s.slot_ms.unwrap_or(0.0) + s.prepare_ms.unwrap_or(0.0);
            if life > 50.0 {
                reads += 1;
                assert!(
                    waited <= life + 5.0,
                    "slot + prepare {waited} ms in a life of {life} ms: {s:?}"
                );
            }
        }
        stop.store(true, Ordering::Relaxed);
        writer.join().expect("writer");
        assert!(reads > 0);
    }

    /// P-T7 (progress half): a whole turn whose core never calls
    /// `begin_prefill` reads `prefill` from pickup and no prepare time.
    /// Mutation P7 (`begin_turn` enters `prepare`) -> red.
    #[test]
    fn a_whole_turn_goes_from_queued_to_prefill() {
        let p = Arc::new(TurnProgress::new());
        sleep_ms(5);
        let _installed = install_turn_progress(Some(Arc::clone(&p)));
        let mut inside = None;
        with_turn_progress(|p| inside = Some(p.snapshot()));
        let inside = inside.expect("installed");
        assert_eq!(inside.phase, "prefill");
        assert!(inside.wait_ms >= 5.0);
        sleep_ms(5);
        let s = p.snapshot();
        assert_eq!(s.phase, "prefill");
        assert_eq!((s.slot_ms, s.prepare_ms), (Some(0.0), Some(0.0)));
        assert!(s.prefill_elapsed_ms.is_some_and(|ms| ms >= 5.0), "{s:?}");
        assert_eq!(s.wait_ms, inside.wait_ms);
    }

    /// The prepare install marks nothing: the phase is the scheduler's.
    #[test]
    fn the_prepare_install_starts_no_turn() {
        let p = Arc::new(TurnProgress::new());
        p.enter(Waiting::Prepare);
        {
            let _installed = install_prepare_progress(Some(Arc::clone(&p)));
            with_turn_progress(|p| p.restoring(32));
            assert_eq!(p.snapshot().phase, "prepare");
        }
        assert!(!turn_progress_installed());
        assert_eq!(p.snapshot().cached_tokens, Some(32));
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
    fn snapshot_never_reports_done_above_total() {
        let p = TurnProgress::new();
        p.begin_prefill(100, 0);
        // A reader racing a second `begin_prefill` can see this pair.
        p.prefill_done.store(500, Ordering::Relaxed);
        assert_eq!(p.snapshot().prefill_done, Some(100));
    }

    #[test]
    fn first_text_chunk_moves_a_silent_core_to_decode() {
        use crate::engine::backend::ChunkSink;
        use crate::engine::types::ChatStreamChunk;
        let chunk = |text: &str, done: bool| ChatStreamChunk {
            text: text.to_string(),
            done,
            finish_reason: None,
            tool_calls: None,
            thinking: None,
            thinking_enabled: None,
            num_tokens: None,
            prompt_tokens: None,
            reasoning_tokens: None,
            reasoning_close: None,
            reasoning_cut_at: None,
            reasoning_cycle_period: None,
            reasoning_stray_closes: None,
            raw_text: None,
            public_raw_text: None,
            text_authoritative: None,
            cached_tokens: None,
            performance: None,
            is_reasoning: None,
        };
        let p = Arc::new(TurnProgress::new());
        let (tx, mut rx) =
            crate::model_thread::stream_channel_with_progress::<ChatStreamChunk>(4, Arc::clone(&p));
        p.begin_turn();
        ChunkSink::send(&tx, Ok(chunk("", false)));
        assert_eq!(
            p.snapshot().phase,
            "prefill",
            "an empty chunk is not a token"
        );
        ChunkSink::send(&tx, Ok(chunk("hi", false)));
        let s = p.snapshot();
        assert_eq!(s.phase, "decode");
        assert!(s.decode_elapsed_ms.is_some());
        ChunkSink::send(&tx, Ok(chunk("", true)));
        assert_eq!(p.snapshot().phase, "done");
        drop(tx);
        while rx.blocking_recv().is_some() {}
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
