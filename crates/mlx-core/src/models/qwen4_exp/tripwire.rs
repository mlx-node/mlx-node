//! Residency facts shared with `residencyInfo()` and the mid-turn tripwire
//! that sends a thrashing mapped process to the slot path at the pass that
//! showed the symptom.
use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

/// SSD bytes one turn may read before the pack is judged not to fit, as a
/// multiple of the mapped bytes: a cold turn reads the pack once (73-74 GiB
/// measured), a thrashing one 1.06-2.3 TB. The counter is the whole
/// process's, so another model loading beside the turn adds to it: the
/// rule trips only when a sample of the pack then finds it partly evicted;
/// a resident pack starts the count over from there.
const SSD_FACTOR: u64 = 2;
/// A pass this many times slower than the running median of passes with
/// the same row count is a thrash symptom (2.4 s against 25-40 ms measured);
/// one such pass is also the ~1-1.4 s bank re-wire after an idle gap, so a
/// streak is required.
const SLOW_FACTOR: u32 = 10;
const SLOW_STREAK: u32 = 8;
/// Passes per row count the median is taken over; at least this many
/// before any pass is judged against its own median.
const SAMPLES: usize = 64;
const MIN_SAMPLES: usize = 8;
/// A row count with fewer samples is judged against the fastest pass over
/// more rows seen so far, when that pass had at least this many rows: a
/// prefill window costs about 0.8 ms per row where a decode step costs 25-40
/// ms per row, so a smaller window would call healthy decode steps slow. A
/// 256-row window (>= 200 ms) leaves a healthy 1-row step 5-8x headroom but
/// a 2-4 row verify or batched step (60-160 ms) only 1.3-3x, which GPU
/// contention eats, so a pass judged against the seed is a symptom only with
/// `SEED_PASS_READ` of SSD reads behind it.
const SEED_MIN_ROWS: usize = 256;
/// SSD bytes a seed-judged slow pass must have read to count toward a
/// streak: a mapped step refaults about 1 GiB while the host evicts the pack
/// (115 GiB over 109 decode steps measured) and nothing once the prompt's
/// windows have cached it, so a slow pass without reads is contention and
/// becomes the row count's own pace.
const SEED_PASS_READ: u64 = 256 << 20;
/// `Live::fallback_token_index` while no request has left the mapped path
/// mid-turn.
pub const NO_TOKEN_INDEX: u64 = u64::MAX;

/// Why a process is on the slot path although the pack exists.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Fallback {
    /// The load gate: the live host left `budget` bytes for `pack` bytes of
    /// banks.
    LoadGate { pack: u64, budget: u64 },
    /// One turn read `read` bytes from SSD against `mapped` mapped bytes.
    SsdReads { read: u64, mapped: u64 },
    /// `steps` consecutive passes of `rows` rows took `step` against a
    /// `median` of their kind (the fastest larger pass while the row count
    /// has fewer than `MIN_SAMPLES` samples, each pass then with SSD reads
    /// behind it).
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
/// may change the residency between passes.
pub struct Live {
    pub mapped: AtomicBool,
    pub mapped_bytes: AtomicU64,
    pub weight_budget: AtomicU64,
    pub partial_expert_slots: AtomicU64,
    pub mapped_bank_budget: AtomicU64,
    pub live_non_reclaimable: AtomicU64,
    /// The prefill window the current plan admits (recomputed when a mapped
    /// process moves to slots).
    pub prefill_chunk_tokens: AtomicU64,
    pub fallback: Mutex<Option<Fallback>>,
    /// Tokens the active request held when it left the mapped path
    /// mid-turn; `NO_TOKEN_INDEX` for the load gate and a fallback applied
    /// between turns.
    pub fallback_token_index: AtomicU64,
}

impl Default for Live {
    fn default() -> Self {
        Self {
            mapped: AtomicBool::new(false),
            mapped_bytes: AtomicU64::new(0),
            weight_budget: AtomicU64::new(0),
            partial_expert_slots: AtomicU64::new(0),
            mapped_bank_budget: AtomicU64::new(0),
            live_non_reclaimable: AtomicU64::new(0),
            prefill_chunk_tokens: AtomicU64::new(0),
            fallback: Mutex::new(None),
            fallback_token_index: AtomicU64::new(NO_TOKEN_INDEX),
        }
    }
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
/// host's live memory. The store leaves the mapped path at the pass that
/// shows them, every open turn with it; a trip whose switch failed stays
/// armed for the next turn to open.
#[derive(Default)]
pub(super) struct Tripwire {
    /// `process_disk_reads()` when the current turn started.
    turn_start_reads: u64,
    /// `process_disk_reads()` after the last pass observed.
    last_reads: u64,
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
        self.last_reads = reads_now;
        self.slow_streak = 0;
    }

    /// A trip taken but not applied (the switch to slots failed) waits for
    /// the next turn to open.
    pub fn rearm(&mut self, why: Fallback) {
        self.tripped = Some(why);
    }

    /// The reference a pass of `rows` rows is judged against: the running
    /// median of its own kind once it has `MIN_SAMPLES`, else the fastest
    /// pass over more rows (`SEED_MIN_ROWS` or more) seen so far, so the
    /// slow-step rule can fire inside the first turn, whose decode steps
    /// have no history of their own.
    fn reference(&self, rows: usize) -> Option<(Duration, bool)> {
        let samples = self.passes.get(&rows).map_or(0, VecDeque::len);
        if samples >= MIN_SAMPLES {
            let mut sorted: Vec<Duration> = self.passes[&rows].iter().copied().collect();
            sorted.sort_unstable();
            return Some((sorted[sorted.len() / 2], true));
        }
        self.passes
            .iter()
            .filter(|&(&count, _)| count > rows && count >= SEED_MIN_ROWS)
            .filter_map(|(_, samples)| samples.iter().min().copied())
            .min()
            .map(|seed| (seed, false))
    }

    /// The symptom held and not yet applied.
    pub fn tripped(&self) -> Option<&Fallback> {
        self.tripped.as_ref()
    }

    #[cfg(test)]
    pub fn turn_start_reads(&self) -> u64 {
        self.turn_start_reads
    }

    #[cfg(test)]
    pub fn slow_streak(&self) -> u32 {
        self.slow_streak
    }

    pub fn take(&mut self) -> Option<Fallback> {
        self.tripped.take()
    }

    /// Whether the window has read more than `SSD_FACTOR` packs with the
    /// pack not resident behind them; `pack_resident` samples the pack only
    /// once the bytes are over.
    fn ssd_rule(
        &mut self,
        reads_now: u64,
        mapped_bytes: u64,
        pack_resident: impl FnOnce() -> bool,
    ) -> bool {
        let read = reads_now.saturating_sub(self.turn_start_reads);
        if mapped_bytes == 0 || read <= mapped_bytes.saturating_mul(SSD_FACTOR) {
            return false;
        }
        if pack_resident() {
            self.turn_start_reads = reads_now;
            return false;
        }
        self.tripped = Some(Fallback::SsdReads {
            read,
            mapped: mapped_bytes,
        });
        true
    }

    /// The SSD rule alone, between two layer sweeps of a prefill group:
    /// `reads_now` is the process's disk read counter now. Returns the
    /// symptom that tripped, once; the pass-level rules see the group's
    /// windows after it.
    pub fn observe_ssd(
        &mut self,
        reads_now: u64,
        mapped_bytes: u64,
        pack_resident: impl FnOnce() -> bool,
    ) -> Option<&Fallback> {
        if self.tripped.is_some() || !self.ssd_rule(reads_now, mapped_bytes, pack_resident) {
            return None;
        }
        self.tripped.as_ref()
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
        pack_resident: impl FnOnce() -> bool,
    ) -> Option<&Fallback> {
        if self.tripped.is_some() {
            return None;
        }
        let pass_read = reads_now.saturating_sub(self.last_reads);
        self.last_reads = reads_now;
        if self.ssd_rule(reads_now, mapped_bytes, pack_resident) {
            return self.tripped.as_ref();
        }
        if let Some((median, own)) = self.reference(rows) {
            let limit = if own {
                median.saturating_mul(SLOW_FACTOR)
            } else {
                median
            };
            if elapsed > limit && (own || pass_read >= SEED_PASS_READ) {
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
        let samples = self.passes.entry(rows).or_default();
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

    /// The pack sample of a turn whose pack is partly evicted.
    fn cold() -> bool {
        false
    }

    #[test]
    fn a_streak_of_slow_steps_trips_but_one_rewire_does_not() {
        let mut wire = Tripwire::default();
        wire.begin_turn(1_000);
        for _ in 0..MIN_SAMPLES {
            assert!(wire.observe(1, 25 * MS, 1_000, 1 << 30, cold).is_none());
        }
        // The first decode step after prefill re-wires the banks (~1.4 s):
        // one slow step, which a fast step forgets.
        assert!(wire.observe(1, 1_400 * MS, 1_000, 1 << 30, cold).is_none());
        for _ in 0..SLOW_STREAK - 2 {
            assert!(wire.observe(1, 300 * MS, 1_000, 1 << 30, cold).is_none());
        }
        assert!(wire.observe(1, 30 * MS, 1_000, 1 << 30, cold).is_none());
        for _ in 0..SLOW_STREAK - 1 {
            assert!(wire.observe(1, 2_400 * MS, 1_000, 1 << 30, cold).is_none());
        }
        let tripped = wire.observe(1, 2_400 * MS, 1_000, 1 << 30, cold).cloned();
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
            wire.observe(1, 2_400 * MS, 1_000, 1 << 30, cold).is_none(),
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
            assert!(wire.observe(1, 25 * MS, 0, mapped, cold).is_none());
        }
        let mut reads = 0u64;
        for turn in 0..2 * SLOW_STREAK {
            wire.begin_turn(reads);
            reads += mapped;
            let rows = 1000 + 37 * turn as usize;
            assert!(wire.observe(rows, 900 * MS, reads, mapped, cold).is_none());
            assert!(
                wire.observe(1, 1_400 * MS, reads, mapped, cold).is_none(),
                "turn {turn}: the re-wire step is not a streak"
            );
        }
        assert!(wire.tripped().is_none());
        // Two slow stretches short of a streak, one per turn, do not add up.
        for _ in 0..2 {
            wire.begin_turn(reads);
            assert!(wire.observe(1, 1_400 * MS, reads, mapped, cold).is_none());
            for _ in 0..SLOW_STREAK - 2 {
                assert!(wire.observe(1, 300 * MS, reads, mapped, cold).is_none());
            }
        }
        assert!(wire.tripped().is_none());
        // Within one turn the streak still trips.
        wire.begin_turn(reads);
        for _ in 0..SLOW_STREAK - 1 {
            assert!(wire.observe(1, 2_400 * MS, reads, mapped, cold).is_none());
        }
        assert_eq!(
            wire.observe(1, 2_400 * MS, reads, mapped, cold)
                .map(Fallback::slow_steps),
            Some(Some(SLOW_STREAK))
        );
    }

    #[test]
    fn row_counts_keep_their_own_medians() {
        let mut wire = Tripwire::default();
        for _ in 0..MIN_SAMPLES {
            wire.observe(1, 25 * MS, 0, 1 << 30, cold);
        }
        // Prefill windows of 1024 rows take 800 ms: not slow decode steps.
        for _ in 0..SLOW_STREAK + 2 {
            assert!(wire.observe(1024, 800 * MS, 0, 1 << 30, cold).is_none());
        }
        // Verification passes of 4 rows at 50 ms: healthy against their own
        // (absent) history and against the 1024-row windows.
        for _ in 0..MIN_SAMPLES - 1 {
            assert!(wire.observe(4, 50 * MS, 0, 1 << 30, cold).is_none());
        }
        assert!(wire.tripped().is_none());
    }

    /// The first turn of a process: a cold 6k prompt (six 1024-row windows,
    /// the first reading the pack) and then decode steps with no history of
    /// their own. A thrash that starts here (2.4 s steps, the pack refaulting
    /// behind them) trips against the fastest window; the re-wire step and
    /// healthy steps do not.
    ///
    /// MUTATION: `reference` returns `None` for a row count with
    /// fewer than `MIN_SAMPLES` samples; the thrash runs on.
    #[test]
    fn a_first_turn_thrash_trips_against_the_prompts_fastest_window() {
        let mapped = 73u64 << 30;
        let mut wire = Tripwire::default();
        wire.begin_turn(0);
        for window in [12_000, 900, 850, 800, 820, 810] {
            assert!(
                wire.observe(1024, window * MS, mapped, mapped, cold)
                    .is_none()
            );
        }
        // The first decode step re-wires the banks: one slow pass.
        assert!(wire.observe(1, 1_400 * MS, mapped, mapped, cold).is_none());
        for _ in 0..3 {
            assert!(wire.observe(1, 25 * MS, mapped, mapped, cold).is_none());
        }
        assert_eq!(wire.slow_streak(), 0);
        let mut reads = mapped;
        for _ in 0..SLOW_STREAK - 1 {
            reads += 1 << 30;
            assert!(wire.observe(1, 2_400 * MS, reads, mapped, cold).is_none());
        }
        reads += 1 << 30;
        assert_eq!(
            wire.observe(1, 2_400 * MS, reads, mapped, cold).cloned(),
            Some(Fallback::SlowSteps {
                rows: 1,
                step: 2_400 * MS,
                median: 800 * MS,
                steps: SLOW_STREAK,
            })
        );
        // Healthy decode after the same prompt never trips, and once the
        // steps have their own history they are judged against it.
        let mut healthy = Tripwire::default();
        healthy.begin_turn(0);
        for window in [12_000, 900, 850, 800, 820, 810] {
            assert!(
                healthy
                    .observe(1024, window * MS, mapped, mapped, cold)
                    .is_none()
            );
        }
        assert!(
            healthy
                .observe(1, 1_400 * MS, mapped, mapped, cold)
                .is_none()
        );
        for _ in 0..64 {
            assert!(healthy.observe(1, 30 * MS, mapped, mapped, cold).is_none());
        }
        assert_eq!(healthy.reference(1), Some((30 * MS, true)));
        assert!(healthy.tripped().is_none());
    }

    /// One 256-row window at 200 ms seeds the 4-row verify passes, which
    /// then run at 250 ms under GPU contention without a byte read from
    /// SSD: not a symptom. They build their own median within `MIN_SAMPLES`
    /// passes and are judged by it from then on; the same passes with the
    /// pack refaulting behind them trip, and one pass under the read floor
    /// breaks their streak.
    ///
    /// MUTATION: a seed-judged slow pass counts without
    /// `SEED_PASS_READ` of reads; the contended passes trip at pass eight.
    /// MUTATION: a seed-judged slow pass without reads is
    /// not recorded; the row count never owns its median.
    #[test]
    fn contended_passes_judged_against_the_seed_learn_their_own_pace() {
        let mapped = 73u64 << 30;
        let mut wire = Tripwire::default();
        wire.begin_turn(mapped);
        assert!(wire.observe(256, 200 * MS, mapped, mapped, cold).is_none());
        assert_eq!(wire.reference(4), Some((200 * MS, false)));
        for pass in 1..=MIN_SAMPLES {
            assert!(
                wire.observe(4, 250 * MS, mapped, mapped, cold).is_none(),
                "pass {pass}"
            );
            assert_eq!(wire.slow_streak(), 0, "pass {pass}");
        }
        assert_eq!(wire.reference(4), Some((250 * MS, true)));
        for _ in 0..2 * SLOW_STREAK {
            assert!(wire.observe(4, 250 * MS, mapped, mapped, cold).is_none());
        }
        assert!(wire.tripped().is_none());

        let mut thrash = Tripwire::default();
        thrash.begin_turn(mapped);
        assert!(
            thrash
                .observe(256, 200 * MS, mapped, mapped, cold)
                .is_none()
        );
        let mut reads = mapped;
        for _ in 0..SLOW_STREAK - 1 {
            reads += SEED_PASS_READ;
            assert!(thrash.observe(4, 250 * MS, reads, mapped, cold).is_none());
        }
        reads += SEED_PASS_READ;
        assert_eq!(
            thrash.observe(4, 250 * MS, reads, mapped, cold).cloned(),
            Some(Fallback::SlowSteps {
                rows: 4,
                step: 250 * MS,
                median: 200 * MS,
                steps: SLOW_STREAK,
            })
        );

        let mut broken = Tripwire::default();
        broken.begin_turn(mapped);
        assert!(
            broken
                .observe(256, 200 * MS, mapped, mapped, cold)
                .is_none()
        );
        let mut reads = mapped;
        for _ in 0..SLOW_STREAK - 1 {
            reads += SEED_PASS_READ;
            assert!(broken.observe(4, 250 * MS, reads, mapped, cold).is_none());
        }
        assert_eq!(broken.slow_streak(), SLOW_STREAK - 1);
        reads += SEED_PASS_READ - 1;
        assert!(broken.observe(4, 250 * MS, reads, mapped, cold).is_none());
        assert_eq!(broken.slow_streak(), 0);
    }

    /// A prompt shorter than `SEED_MIN_ROWS` seeds nothing: its per-row cost
    /// is a prefill cost, not a decode cost, so decode steps have no
    /// reference until they have their own history.
    #[test]
    fn a_short_prompt_seeds_no_decode_reference() {
        let mut wire = Tripwire::default();
        wire.begin_turn(0);
        assert!(wire.observe(40, 60 * MS, 0, 1 << 30, cold).is_none());
        assert_eq!(wire.reference(1), None);
        for _ in 0..2 * SLOW_STREAK {
            assert!(wire.observe(1, 100 * MS, 0, 1 << 30, cold).is_none());
        }
        assert!(wire.tripped().is_none());
        // A 512-row fallback window is judged against a 1024-row one.
        assert!(wire.observe(1024, 800 * MS, 0, 1 << 30, cold).is_none());
        assert_eq!(wire.reference(512), Some((800 * MS, false)));
        assert_eq!(wire.reference(1024), None, "no larger pass exists");
        // A failed switch re-arms the trip for the next turn.
        let why = Fallback::SsdReads {
            read: 3 << 30,
            mapped: 1 << 30,
        };
        wire.rearm(why.clone());
        assert_eq!(wire.take(), Some(why));
    }

    #[test]
    fn ssd_reads_beyond_twice_the_pack_in_one_turn_trip() {
        let mapped = 73u64 << 30;
        let mut wire = Tripwire::default();
        wire.begin_turn(500 << 30);
        // A cold turn reads the pack once.
        assert!(
            wire.observe(1024, 800 * MS, (500 << 30) + mapped, mapped, cold)
                .is_none()
        );
        assert!(
            wire.observe(1024, 800 * MS, (500 << 30) + 2 * mapped, mapped, cold)
                .is_none()
        );
        let tripped = wire
            .observe(1024, 800 * MS, (500 << 30) + 2 * mapped + 1, mapped, cold)
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
                .observe(1, 25 * MS, (1_000 << 30) + mapped, mapped, cold)
                .is_none()
        );
        assert!(
            fresh.observe(1, 25 * MS, 1_000 << 30, 0, cold).is_none(),
            "no mapped bytes, no SSD rule"
        );
    }

    /// The read counter is the process's: another model loading beside a
    /// turn reads past twice the pack while this pack stays resident. The
    /// pack is sampled only once the bytes are over, a resident sample
    /// starts the count over, and the same bytes over a partly evicted pack
    /// trip, between two passes and between two sweeps alike.
    ///
    /// MUTATION: `ssd_rule` ignores `pack_resident`; the unrelated reads
    /// send a resident model to slots.
    #[test]
    fn unrelated_reads_beside_a_resident_pack_never_trip() {
        let mapped = 73u64 << 30;
        let samples = std::cell::Cell::new(0u32);
        let resident = || {
            samples.set(samples.get() + 1);
            true
        };
        let mut wire = Tripwire::default();
        wire.begin_turn(0);
        assert!(
            wire.observe(1, 25 * MS, 2 * mapped, mapped, resident)
                .is_none()
        );
        assert_eq!(samples.get(), 0, "no sample under the bytes");
        assert!(
            wire.observe(1, 25 * MS, 3 * mapped, mapped, resident)
                .is_none()
        );
        assert_eq!(samples.get(), 1);
        assert_eq!(wire.turn_start_reads(), 3 * mapped, "the count starts over");
        assert!(
            wire.observe(1, 25 * MS, 5 * mapped, mapped, resident)
                .is_none()
        );
        assert_eq!(samples.get(), 1);
        assert!(wire.observe_ssd(5 * mapped + 1, mapped, resident).is_none());
        assert_eq!(samples.get(), 2);
        assert!(wire.tripped().is_none());
        // The host evicts the pack and the turn refaults it.
        assert_eq!(
            wire.observe(1, 25 * MS, 7 * mapped + 2, mapped, cold)
                .cloned(),
            Some(Fallback::SsdReads {
                read: 2 * mapped + 1,
                mapped,
            })
        );
        let mut sweep = Tripwire::default();
        sweep.begin_turn(0);
        assert!(sweep.observe_ssd(3 * mapped, mapped, resident).is_none());
        assert_eq!(
            sweep.observe_ssd(5 * mapped + 1, mapped, cold).cloned(),
            Some(Fallback::SsdReads {
                read: 2 * mapped + 1,
                mapped,
            })
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
