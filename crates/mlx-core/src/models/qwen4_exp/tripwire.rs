//! Residency facts shared with `residencyInfo()` and the mid-turn tripwire
//! that sends a thrashing mapped process to the slot path at its next turn.
use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

/// SSD bytes one turn may read before the pack is judged not to fit, as a
/// multiple of the mapped bytes: a cold turn reads the pack once (73-74 GiB
/// measured), a thrashing one 1.06-2.3 TB.
const SSD_FACTOR: u64 = 2;
/// A pass this many times slower than the running median of passes with
/// the same row count is a thrash symptom (2.4 s against 25-40 ms measured);
/// one such pass is also the ~1-1.4 s bank re-wire after an idle gap, so a
/// streak is required.
const SLOW_FACTOR: u32 = 10;
const SLOW_STREAK: u32 = 8;
/// Passes per row count the median is taken over; at least this many
/// before any pass is judged.
const SAMPLES: usize = 64;
const MIN_SAMPLES: usize = 8;

/// Why a process is on the slot path although the pack exists.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Fallback {
    /// The load gate: the live host left `budget` bytes for `pack` bytes of
    /// banks.
    LoadGate { pack: u64, budget: u64 },
    /// One turn read `read` bytes from SSD against `mapped` mapped bytes.
    SsdReads { read: u64, mapped: u64 },
    /// `steps` consecutive passes of `rows` rows took `step` against a
    /// `median` of their kind.
    SlowSteps {
        rows: usize,
        step: Duration,
        median: Duration,
        steps: u32,
    },
}

impl std::fmt::Display for Fallback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let gib = |bytes: u64| bytes as f64 / (1u64 << 30) as f64;
        match self {
            Self::LoadGate { pack, budget } => write!(
                f,
                "load gate: mapped banks need {:.1} GiB, the host leaves {:.1} GiB",
                gib(*pack),
                gib(*budget)
            ),
            Self::SsdReads { read, mapped } => write!(
                f,
                "one turn read {:.1} GiB from SSD against {:.1} GiB mapped",
                gib(*read),
                gib(*mapped)
            ),
            Self::SlowSteps {
                rows,
                step,
                median,
                steps,
            } => write!(
                f,
                "{steps} consecutive {rows}-row passes took {step:.1?} against a {median:.1?} median"
            ),
        }
    }
}

impl Fallback {
    pub fn read_bytes(&self) -> Option<u64> {
        match self {
            Self::SsdReads { read, .. } => Some(*read),
            _ => None,
        }
    }
    pub fn slow_steps(&self) -> Option<u32> {
        match self {
            Self::SlowSteps { steps, .. } => Some(*steps),
            _ => None,
        }
    }
}

/// What `residencyInfo()` reads from the JS thread while the model thread
/// may change the residency between turns.
#[derive(Default)]
pub struct Live {
    pub mapped: AtomicBool,
    pub mapped_bytes: AtomicU64,
    pub weight_budget: AtomicU64,
    pub partial_expert_slots: AtomicU64,
    pub mapped_bank_budget: AtomicU64,
    pub live_non_reclaimable: AtomicU64,
    pub fallback: Mutex<Option<Fallback>>,
}

impl Live {
    pub fn set_fallback(&self, fallback: Fallback) {
        *self.fallback.lock().unwrap_or_else(|p| p.into_inner()) = Some(fallback);
    }
    pub fn fallback(&self) -> Option<Fallback> {
        self.fallback
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }
    pub fn store(&self, plan: &super::memory::Plan) {
        self.mapped.store(plan.mapped, Ordering::Release);
        self.mapped_bytes
            .store(plan.mapped_bytes, Ordering::Release);
        self.weight_budget.store(plan.budget, Ordering::Release);
        self.partial_expert_slots
            .store(plan.partial_expert_slots as u64, Ordering::Release);
        self.mapped_bank_budget
            .store(plan.mapped_bank_budget, Ordering::Release);
        self.live_non_reclaimable
            .store(plan.live_non_reclaimable, Ordering::Release);
    }
}

/// Thrash symptoms of a mapped turn whose pack no longer fits beside the
/// host's live memory. The turn that shows them finishes as it is; the next
/// turn runs on slots.
#[derive(Default)]
pub(super) struct Tripwire {
    /// `process_disk_reads()` when the current turn started.
    turn_start_reads: u64,
    passes: HashMap<usize, VecDeque<Duration>>,
    slow_streak: u32,
    tripped: Option<Fallback>,
}

impl Tripwire {
    /// A turn starts after an idle gap, so its first main pass is the bank
    /// re-wire: one legitimately slow pass. A streak carried across turns
    /// would add those up over short answers (one judged pass each) and
    /// send a healthy session to slots for good.
    pub fn begin_turn(&mut self, reads_now: u64) {
        self.turn_start_reads = reads_now;
        self.slow_streak = 0;
    }

    #[cfg(test)]
    pub fn tripped(&self) -> Option<&Fallback> {
        self.tripped.as_ref()
    }

    pub fn take(&mut self) -> Option<Fallback> {
        self.tripped.take()
    }

    /// One main-network pass of `rows` rows took `elapsed`; `reads_now` is
    /// the process's disk read counter after it. Returns the symptom that
    /// tripped, once.
    pub fn observe(
        &mut self,
        rows: usize,
        elapsed: Duration,
        reads_now: u64,
        mapped_bytes: u64,
    ) -> Option<&Fallback> {
        if self.tripped.is_some() {
            return None;
        }
        let read = reads_now.saturating_sub(self.turn_start_reads);
        if mapped_bytes > 0 && read > mapped_bytes.saturating_mul(SSD_FACTOR) {
            self.tripped = Some(Fallback::SsdReads {
                read,
                mapped: mapped_bytes,
            });
            return self.tripped.as_ref();
        }
        let samples = self.passes.entry(rows).or_default();
        if samples.len() >= MIN_SAMPLES {
            let mut sorted: Vec<Duration> = samples.iter().copied().collect();
            sorted.sort_unstable();
            let median = sorted[sorted.len() / 2];
            if elapsed > median.saturating_mul(SLOW_FACTOR) {
                self.slow_streak += 1;
                if self.slow_streak >= SLOW_STREAK {
                    self.tripped = Some(Fallback::SlowSteps {
                        rows,
                        step: elapsed,
                        median,
                        steps: self.slow_streak,
                    });
                    return self.tripped.as_ref();
                }
                return None;
            }
            self.slow_streak = 0;
        }
        if samples.len() == SAMPLES {
            samples.pop_front();
        }
        samples.push_back(elapsed);
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MS: Duration = Duration::from_millis(1);

    #[test]
    fn a_streak_of_slow_steps_trips_but_one_rewire_does_not() {
        let mut wire = Tripwire::default();
        wire.begin_turn(1_000);
        for _ in 0..MIN_SAMPLES {
            assert!(wire.observe(1, 25 * MS, 1_000, 1 << 30).is_none());
        }
        // The first decode step after prefill re-wires the banks (~1.4 s):
        // one slow step, which a fast step forgets.
        assert!(wire.observe(1, 1_400 * MS, 1_000, 1 << 30).is_none());
        for _ in 0..SLOW_STREAK - 2 {
            assert!(wire.observe(1, 300 * MS, 1_000, 1 << 30).is_none());
        }
        assert!(wire.observe(1, 30 * MS, 1_000, 1 << 30).is_none());
        for _ in 0..SLOW_STREAK - 1 {
            assert!(wire.observe(1, 2_400 * MS, 1_000, 1 << 30).is_none());
        }
        let tripped = wire.observe(1, 2_400 * MS, 1_000, 1 << 30).cloned();
        assert_eq!(
            tripped,
            Some(Fallback::SlowSteps {
                rows: 1,
                step: 2_400 * MS,
                median: 25 * MS,
                steps: SLOW_STREAK,
            })
        );
        assert!(
            wire.observe(1, 2_400 * MS, 1_000, 1 << 30).is_none(),
            "reported once"
        );
        assert_eq!(wire.take(), tripped);
        assert!(wire.tripped().is_none());
    }

    /// Short turns: a prefill window of a new length (untrained) then the
    /// re-wire step, and the turn ends. Sixteen of them in a row, each
    /// reading the pack once, are cold starts, not a thrash.
    #[test]
    fn short_cold_turns_with_one_rewire_each_never_trip() {
        let mapped = 73u64 << 30;
        let mut wire = Tripwire::default();
        wire.begin_turn(0);
        for _ in 0..MIN_SAMPLES {
            assert!(wire.observe(1, 25 * MS, 0, mapped).is_none());
        }
        let mut reads = 0u64;
        for turn in 0..2 * SLOW_STREAK {
            wire.begin_turn(reads);
            reads += mapped;
            let rows = 1000 + 37 * turn as usize;
            assert!(wire.observe(rows, 900 * MS, reads, mapped).is_none());
            assert!(
                wire.observe(1, 1_400 * MS, reads, mapped).is_none(),
                "turn {turn}: the re-wire step is not a streak"
            );
        }
        assert!(wire.tripped().is_none());
        // Two slow stretches short of a streak, one per turn, do not add up.
        for _ in 0..2 {
            wire.begin_turn(reads);
            assert!(wire.observe(1, 1_400 * MS, reads, mapped).is_none());
            for _ in 0..SLOW_STREAK - 2 {
                assert!(wire.observe(1, 300 * MS, reads, mapped).is_none());
            }
        }
        assert!(wire.tripped().is_none());
        // Within one turn the streak still trips.
        wire.begin_turn(reads);
        for _ in 0..SLOW_STREAK - 1 {
            assert!(wire.observe(1, 2_400 * MS, reads, mapped).is_none());
        }
        assert_eq!(
            wire.observe(1, 2_400 * MS, reads, mapped)
                .map(Fallback::slow_steps),
            Some(Some(SLOW_STREAK))
        );
    }

    #[test]
    fn row_counts_keep_their_own_medians() {
        let mut wire = Tripwire::default();
        for _ in 0..MIN_SAMPLES {
            wire.observe(1, 25 * MS, 0, 1 << 30);
        }
        // Prefill windows of 1024 rows take 800 ms: not slow decode steps.
        for _ in 0..SLOW_STREAK + 2 {
            assert!(wire.observe(1024, 800 * MS, 0, 1 << 30).is_none());
        }
        // Verification passes of 4 rows have fewer than MIN_SAMPLES samples.
        for _ in 0..MIN_SAMPLES - 1 {
            assert!(wire.observe(4, 5_000 * MS, 0, 1 << 30).is_none());
        }
        assert!(wire.tripped().is_none());
    }

    #[test]
    fn ssd_reads_beyond_twice_the_pack_in_one_turn_trip() {
        let mapped = 73u64 << 30;
        let mut wire = Tripwire::default();
        wire.begin_turn(500 << 30);
        // A cold turn reads the pack once.
        assert!(
            wire.observe(1024, 800 * MS, (500 << 30) + mapped, mapped)
                .is_none()
        );
        assert!(
            wire.observe(1024, 800 * MS, (500 << 30) + 2 * mapped, mapped)
                .is_none()
        );
        let tripped = wire
            .observe(1024, 800 * MS, (500 << 30) + 2 * mapped + 1, mapped)
            .cloned();
        assert_eq!(
            tripped,
            Some(Fallback::SsdReads {
                read: 2 * mapped + 1,
                mapped,
            })
        );
        // The next turn starts its own count.
        let mut fresh = Tripwire::default();
        fresh.begin_turn(1_000 << 30);
        assert!(
            fresh
                .observe(1, 25 * MS, (1_000 << 30) + mapped, mapped)
                .is_none()
        );
        assert!(
            fresh.observe(1, 25 * MS, 1_000 << 30, 0).is_none(),
            "no mapped bytes, no SSD rule"
        );
    }

    #[test]
    fn fallback_reasons_name_their_numbers() {
        assert_eq!(
            Fallback::LoadGate {
                pack: 73 << 30,
                budget: 61 << 30
            }
            .to_string(),
            "load gate: mapped banks need 73.0 GiB, the host leaves 61.0 GiB"
        );
        let ssd = Fallback::SsdReads {
            read: 160 << 30,
            mapped: 73 << 30,
        };
        assert_eq!(
            ssd.to_string(),
            "one turn read 160.0 GiB from SSD against 73.0 GiB mapped"
        );
        assert_eq!(ssd.read_bytes(), Some(160 << 30));
        assert_eq!(ssd.slow_steps(), None);
        let slow = Fallback::SlowSteps {
            rows: 1,
            step: 2_400 * MS,
            median: 25 * MS,
            steps: 8,
        };
        assert_eq!(
            slow.to_string(),
            "8 consecutive 1-row passes took 2.4s against a 25.0ms median"
        );
        assert_eq!(slow.slow_steps(), Some(8));
        let live = Live::default();
        assert!(live.fallback().is_none());
        live.set_fallback(slow.clone());
        assert_eq!(live.fallback(), Some(slow));
    }
}
