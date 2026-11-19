//! Opt-in wall-clock timeline of each DSpark cycle of the Qwen3.5 dense
//! DFlash2 whole turn and of each scheduled DFlash2 lane.
//!
//! `MLX_DFLASH2_CYCLE_TIMELINE=1` together with
//! `MLX_DFLASH2_CYCLE_TIMELINE_PATH=<file>` turns it on; both are read at
//! model load ([`configure_from_env`]). Marks only read the clock where the
//! code already is: no eval or sync is added. Rows are buffered on the model
//! thread and appended to the file when the stepper drops or the lane ends.
//! Off, every mark is one relaxed load and one branch.

use std::cell::{Cell, RefCell};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

pub(crate) const TIMELINE_ENV: &str = "MLX_DFLASH2_CYCLE_TIMELINE";
pub(crate) const TIMELINE_PATH_ENV: &str = "MLX_DFLASH2_CYCLE_TIMELINE_PATH";

static ON: AtomicBool = AtomicBool::new(false);
static PATH: Mutex<Option<PathBuf>> = Mutex::new(None);
static TURNS: AtomicU64 = AtomicU64::new(0);

/// Section edges of one cycle, in the order the cycle reaches them. Each
/// section runs from the previous present mark (or the cycle's start) to
/// its own mark.
#[derive(Clone, Copy)]
#[repr(usize)]
pub(crate) enum Mark {
    /// Loop-top stop checks: observer, EOS, cancel, repetition, length.
    StopChecks,
    /// Draft cap, lookahead row reservation, thinking-budget cut.
    Plan,
    /// Draft forward + head + selector (sampling) graph; the compiled greedy
    /// draft is one graph. GPU wait only on the eager first call and on the
    /// sampled host-selector path.
    DraftGraph,
    /// `async_eval` of the drafted path (encode + submit, no wait).
    DraftSubmit,
    /// Driver proposal checks, thinking-cut replace, verification budget.
    DraftPost,
    /// Device verify ids read to host before the verify input: only a device
    /// verify whose cycle draft is not on the device. GPU wait (draft).
    ProposalReadback,
    /// The 8-row verify input.
    VerifyInput,
    /// Row reservation (claim; a suspended lane replays here) and the
    /// write-cycle ticket.
    VerifyReserve,
    /// Target verify graph build (its first layers are submitted early).
    VerifyGraph,
    /// Device-id verify: `async_eval` of the verify, then the driver's ids
    /// and the draft read to host (GPU wait: draft, early layers).
    ProposalHost,
    /// Acceptance argmax eval: encode of the rest + GPU wait (verify).
    VerifyWait,
    /// Logits read-back and accept / sample.
    Accept,
    /// Pre-commit stop clamp (observer, EOS, repetition, budget).
    StopClamp,
    /// Paged K/V rows kept (host pool bookkeeping).
    CommitKv,
    /// GDN cells parity swap + pending tape.
    CommitGdn,
    /// Draft context fuse + ring append (graph only).
    CommitRing,
    /// Budget boundary sync and calibration materialize (GPU wait only on
    /// those cycles), acceptance stats.
    PostCommit,
    /// Emit: history, reasoning tracker, detokenize, stream send.
    Emit,
    /// `synchronize_and_clear_cache` every 256 tokens. GPU wait.
    CacheClear,
    /// `async_eval` of caches + ring + boundary (encode + submit, no wait).
    BoundarySubmit,
}

const SECTIONS: usize = Mark::BoundarySubmit as usize + 1;

const NAMES: [&str; SECTIONS] = [
    "stop_checks",
    "plan",
    "draft_graph",
    "draft_submit",
    "draft_post",
    "proposal_readback",
    "verify_input",
    "verify_reserve",
    "verify_graph",
    "proposal_host",
    "verify_wait",
    "accept",
    "stop_clamp",
    "commit_kv",
    "commit_gdn",
    "commit_ring",
    "post_commit",
    "emit",
    "cache_clear",
    "boundary_submit",
];

/// Rows a timeline that keeps rows holds before it grows.
const ROW_BUFFER: usize = 1024;

struct Cycle {
    start: Instant,
    marks: [Option<Instant>; SECTIONS],
    context: usize,
    drafted: usize,
    accepted: usize,
    emitted: usize,
}

impl Cycle {
    fn new(start: Instant, context: usize) -> Self {
        Cycle {
            start,
            marks: [None; SECTIONS],
            context,
            drafted: 0,
            accepted: 0,
            emitted: 0,
        }
    }

    fn end(&self) -> Instant {
        self.marks
            .iter()
            .rev()
            .flatten()
            .next()
            .copied()
            .unwrap_or(self.start)
    }
}

struct Row {
    gap_us: u64,
    cycle: Cycle,
}

struct Timeline {
    path: PathBuf,
    previous_end: Instant,
    open: Option<Cycle>,
    rows: Vec<Row>,
}

impl Timeline {
    fn new(path: PathBuf, rows: usize) -> Self {
        Timeline {
            path,
            previous_end: Instant::now(),
            open: None,
            rows: Vec::with_capacity(rows),
        }
    }

    fn close_open(&mut self) {
        if let Some(cycle) = self.open.take() {
            // A loop-top exit never planned a cycle.
            if cycle.marks[Mark::Plan as usize].is_none() {
                return;
            }
            let gap_us = micros(self.previous_end, cycle.start);
            self.previous_end = cycle.end();
            self.rows.push(Row { gap_us, cycle });
        }
    }
}

/// A scheduled lane's timeline. Lane cycles start in the shared scheduler
/// step; only a lane that a DFlash2 reservation claimed keeps its rows.
struct LaneTimeline {
    seq: u32,
    claimed: bool,
    timeline: Timeline,
}

thread_local! {
    static ACTIVE: RefCell<Option<Timeline>> = const { RefCell::new(None) };
    static LANES: RefCell<Vec<LaneTimeline>> = const { RefCell::new(Vec::new()) };
    /// The lane whose cycle takes the plain [`mark`]s (the draft graph, the
    /// acceptance waits) while the scheduler works on its row.
    static FOCUS: Cell<Option<u32>> = const { Cell::new(None) };
}

fn micros(from: Instant, to: Instant) -> u64 {
    to.saturating_duration_since(from).as_micros() as u64
}

/// Load-time read of the two env vars.
pub(crate) fn configure_from_env() {
    let on = std::env::var(TIMELINE_ENV).is_ok_and(|v| v == "1");
    let path = std::env::var_os(TIMELINE_PATH_ENV)
        .filter(|p| !p.is_empty())
        .map(PathBuf::from);
    let enabled = on && path.is_some();
    if on && !enabled {
        tracing::warn!(
            target: "mlx_core::inference",
            "{TIMELINE_ENV}=1 without {TIMELINE_PATH_ENV}: cycle timeline stays off"
        );
    }
    if let Ok(mut slot) = PATH.lock() {
        *slot = if enabled { path } else { None };
    }
    ON.store(enabled, Ordering::Relaxed);
}

/// Starts this thread's timeline for one decode turn.
#[inline]
pub(crate) fn begin_turn() {
    if ON.load(Ordering::Relaxed) {
        begin_turn_slow();
    }
}

fn path() -> Option<PathBuf> {
    PATH.lock().ok().and_then(|p| p.clone())
}

#[cold]
#[inline(never)]
fn begin_turn_slow() {
    #[cfg(test)]
    test_hooks::slow_call();
    let Some(path) = path() else {
        return;
    };
    ACTIVE.with(|active| {
        *active.borrow_mut() = Some(Timeline::new(path, ROW_BUFFER));
    });
}

#[inline]
pub(crate) fn cycle_start(context: usize) {
    if ON.load(Ordering::Relaxed) {
        cycle_start_slow(context);
    }
}

#[cold]
#[inline(never)]
fn cycle_start_slow(context: usize) {
    #[cfg(test)]
    test_hooks::slow_call();
    let now = Instant::now();
    ACTIVE.with(|active| {
        if let Some(t) = active.borrow_mut().as_mut() {
            t.close_open();
            t.open = Some(Cycle::new(now, context));
            #[cfg(test)]
            test_hooks::ORDER.with(|order| order.borrow_mut().push(Vec::new()));
        }
    });
}

#[inline]
pub(crate) fn mark(mark: Mark) {
    if ON.load(Ordering::Relaxed) {
        mark_slow(mark);
    }
}

#[cold]
#[inline(never)]
fn mark_slow(mark: Mark) {
    #[cfg(test)]
    test_hooks::slow_call();
    let now = Instant::now();
    if let Some(seq) = FOCUS.with(Cell::get)
        && lane_mark_at(seq, mark, now)
    {
        return;
    }
    ACTIVE.with(|active| {
        if let Some(cycle) = active.borrow_mut().as_mut().and_then(|t| t.open.as_mut()) {
            cycle.marks[mark as usize] = Some(now);
            #[cfg(test)]
            test_hooks::ORDER.with(|order| {
                if let Some(cycle) = order.borrow_mut().last_mut() {
                    cycle.push(NAMES[mark as usize]);
                }
            });
        }
    });
}

#[inline]
pub(crate) fn cycle_counts(drafted: usize, accepted: usize, emitted: usize) {
    if ON.load(Ordering::Relaxed) {
        cycle_counts_slow(drafted, accepted, emitted);
    }
}

#[cold]
#[inline(never)]
fn cycle_counts_slow(drafted: usize, accepted: usize, emitted: usize) {
    #[cfg(test)]
    test_hooks::slow_call();
    ACTIVE.with(|active| {
        if let Some(cycle) = active.borrow_mut().as_mut().and_then(|t| t.open.as_mut()) {
            cycle.drafted = drafted;
            cycle.accepted = accepted;
            cycle.emitted = emitted;
        }
    });
}

/// Ends this thread's timeline and appends it to the file.
#[inline]
pub(crate) fn end_turn() {
    if ON.load(Ordering::Relaxed) {
        end_turn_slow();
    }
}

#[cold]
#[inline(never)]
fn end_turn_slow() {
    #[cfg(test)]
    test_hooks::slow_call();
    let Some(t) = ACTIVE.with(|active| active.borrow_mut().take()) else {
        return;
    };
    write_block(t, None);
}

#[inline]
fn lane_on() -> bool {
    ON.load(Ordering::Relaxed)
}

/// Starts the next cycle of `seq`'s scheduled lane at its row of a step.
#[inline]
pub(crate) fn lane_cycle_start(seq: u32, context: usize) {
    if lane_on() {
        lane_cycle_start_slow(seq, context);
    }
}

#[cold]
#[inline(never)]
fn lane_cycle_start_slow(seq: u32, context: usize) {
    #[cfg(test)]
    test_hooks::slow_call();
    let now = Instant::now();
    LANES.with(|lanes| {
        let mut lanes = lanes.borrow_mut();
        let index = match lanes.iter().position(|lane| lane.seq == seq) {
            Some(index) => index,
            None => {
                let Some(path) = path() else {
                    return;
                };
                lanes.push(LaneTimeline {
                    seq,
                    claimed: false,
                    timeline: Timeline::new(path, 0),
                });
                lanes.len() - 1
            }
        };
        let lane = &mut lanes[index];
        if lane.claimed {
            lane.timeline.close_open();
        }
        lane.timeline.open = Some(Cycle::new(now, context));
        #[cfg(test)]
        test_hooks::LANE_ORDER.with(|order| order.borrow_mut().push((seq, Vec::new())));
    });
}

/// `seq`'s lane is a DFlash2 lane: its cycles keep their rows.
#[inline]
pub(crate) fn claim_lane(seq: u32) {
    if lane_on() {
        claim_lane_slow(seq);
    }
}

#[cold]
#[inline(never)]
fn claim_lane_slow(seq: u32) {
    #[cfg(test)]
    test_hooks::slow_call();
    LANES.with(|lanes| {
        if let Some(lane) = lanes.borrow_mut().iter_mut().find(|lane| lane.seq == seq)
            && !lane.claimed
        {
            lane.claimed = true;
            lane.timeline.rows.reserve(ROW_BUFFER);
        }
    });
}

#[inline]
pub(crate) fn lane_mark(seq: u32, mark: Mark) {
    if lane_on() {
        lane_mark_slow(seq, mark);
    }
}

#[cold]
#[inline(never)]
fn lane_mark_slow(seq: u32, mark: Mark) {
    #[cfg(test)]
    test_hooks::slow_call();
    lane_mark_at(seq, mark, Instant::now());
}

/// Marks `seq`'s open lane cycle; false when `seq` has no lane timeline.
fn lane_mark_at(seq: u32, mark: Mark, now: Instant) -> bool {
    LANES.with(|lanes| {
        let mut lanes = lanes.borrow_mut();
        let Some(lane) = lanes.iter_mut().find(|lane| lane.seq == seq) else {
            return false;
        };
        if let Some(cycle) = lane.timeline.open.as_mut() {
            cycle.marks[mark as usize] = Some(now);
            #[cfg(test)]
            test_hooks::LANE_ORDER.with(|order| {
                if let Some((_, cycle)) = order
                    .borrow_mut()
                    .iter_mut()
                    .rev()
                    .find(|(owner, _)| *owner == seq)
                {
                    cycle.push(NAMES[mark as usize]);
                }
            });
        }
        true
    })
}

#[inline]
pub(crate) fn lane_counts(seq: u32, drafted: usize, accepted: usize, emitted: usize) {
    if lane_on() {
        lane_counts_slow(seq, drafted, accepted, emitted);
    }
}

#[cold]
#[inline(never)]
fn lane_counts_slow(seq: u32, drafted: usize, accepted: usize, emitted: usize) {
    #[cfg(test)]
    test_hooks::slow_call();
    LANES.with(|lanes| {
        if let Some(cycle) = lanes
            .borrow_mut()
            .iter_mut()
            .find(|lane| lane.seq == seq)
            .and_then(|lane| lane.timeline.open.as_mut())
        {
            cycle.drafted = drafted;
            cycle.accepted = accepted;
            cycle.emitted = emitted;
        }
    });
}

/// Routes the plain [`mark`]s to `seq`'s lane cycle until the guard drops.
#[inline]
pub(crate) fn focus_lane(seq: u32) -> LaneFocus {
    if lane_on() {
        return focus_lane_slow(seq);
    }
    LaneFocus { set: false }
}

#[cold]
#[inline(never)]
fn focus_lane_slow(seq: u32) -> LaneFocus {
    #[cfg(test)]
    test_hooks::slow_call();
    FOCUS.with(|focus| focus.set(Some(seq)));
    LaneFocus { set: true }
}

pub(crate) struct LaneFocus {
    set: bool,
}

impl Drop for LaneFocus {
    fn drop(&mut self) {
        if self.set {
            FOCUS.with(|focus| focus.set(None));
        }
    }
}

/// Ends `seq`'s lane timeline; a claimed lane appends its rows to the file.
#[inline]
pub(crate) fn end_lane(seq: u32) {
    if lane_on() {
        end_lane_slow(seq);
    }
}

#[cold]
#[inline(never)]
fn end_lane_slow(seq: u32) {
    #[cfg(test)]
    test_hooks::slow_call();
    let lane = LANES.with(|lanes| {
        let mut lanes = lanes.borrow_mut();
        let index = lanes.iter().position(|lane| lane.seq == seq)?;
        Some(lanes.swap_remove(index))
    });
    if let Some(lane) = lane
        && lane.claimed
    {
        write_block(lane.timeline, Some(seq));
    }
}

fn write_block(mut t: Timeline, seq: Option<u32>) {
    t.close_open();
    let turn = TURNS.fetch_add(1, Ordering::Relaxed);
    let lane = seq.map_or_else(String::new, |seq| format!(" seq={seq}"));
    let mut out = String::with_capacity(64 + t.rows.len() * 160);
    out.push_str(&format!(
        "# dflash2-cycle-timeline v2 turn={turn} cycles={} unit=us{lane}\n# cycle\tctx\tdrafted\taccepted\temitted\tgap\ttotal",
        t.rows.len()
    ));
    for name in NAMES {
        out.push('\t');
        out.push_str(name);
    }
    out.push('\n');
    for (i, row) in t.rows.iter().enumerate() {
        let c = &row.cycle;
        out.push_str(&format!(
            "{i}\t{}\t{}\t{}\t{}\t{}\t{}",
            c.context,
            c.drafted,
            c.accepted,
            c.emitted,
            row.gap_us,
            micros(c.start, c.end())
        ));
        let mut last = c.start;
        for m in c.marks {
            match m {
                Some(at) => {
                    out.push_str(&format!("\t{}", micros(last, at)));
                    last = at;
                }
                None => out.push_str("\t-"),
            }
        }
        out.push('\n');
    }
    let written = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&t.path)
        .and_then(|mut f| f.write_all(out.as_bytes()));
    if let Err(error) = written {
        tracing::warn!(
            target: "mlx_core::inference",
            path = %t.path.display(),
            %error,
            "DFlash2 cycle timeline was not written"
        );
    }
}

#[cfg(test)]
pub(crate) mod test_hooks {
    use std::cell::{Cell, RefCell};
    use std::path::PathBuf;
    use std::sync::atomic::Ordering;

    thread_local! {
        /// The marks of each cycle of this thread, in the order reached.
        pub(crate) static ORDER: RefCell<Vec<Vec<&'static str>>> = const { RefCell::new(Vec::new()) };
        /// Each lane cycle of this thread: its lane and its marks in the
        /// order reached.
        pub(crate) static LANE_ORDER: RefCell<Vec<(u32, Vec<&'static str>)>> =
            const { RefCell::new(Vec::new()) };
        /// Calls of this thread that took a timeline's slow path.
        static SLOW_CALLS: Cell<usize> = const { Cell::new(0) };
    }

    pub(super) fn slow_call() {
        SLOW_CALLS.with(|calls| calls.set(calls.get() + 1));
    }

    /// The section names, in the order of a cycle.
    pub(crate) fn names() -> &'static [&'static str] {
        &super::NAMES
    }

    /// The slow-path calls of this thread since the last take.
    pub(crate) fn take_slow_calls() -> usize {
        SLOW_CALLS.with(|calls| calls.replace(0))
    }

    /// Turns the timeline on into `path`, as the two env vars do at load.
    pub(crate) fn enable(path: PathBuf) {
        if let Ok(mut slot) = super::PATH.lock() {
            *slot = Some(path);
        }
        ORDER.with(|order| order.borrow_mut().clear());
        LANE_ORDER.with(|order| order.borrow_mut().clear());
        super::ON.store(true, Ordering::Relaxed);
    }

    /// The row buffer of `seq`'s lane timeline on this thread.
    pub(crate) fn lane_row_capacity(seq: u32) -> Option<usize> {
        super::LANES.with(|lanes| {
            lanes
                .borrow()
                .iter()
                .find(|lane| lane.seq == seq)
                .map(|lane| lane.timeline.rows.capacity())
        })
    }

    pub(crate) fn disable() {
        super::ON.store(false, Ordering::Relaxed);
        if let Ok(mut slot) = super::PATH.lock() {
            *slot = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A lane that no DFlash2 reservation claimed keeps no row buffer: other
    /// families' lanes may end without their lane end. A claimed lane
    /// reserves its rows; an ended lane leaves no timeline.
    #[test]
    fn an_unclaimed_lane_holds_no_row_buffer() {
        let seq = 9_001;
        let path = std::env::temp_dir().join(format!("lane_buffer_{}.tsv", std::process::id()));
        test_hooks::enable(path.clone());
        lane_cycle_start(seq, 10);
        let unclaimed = test_hooks::lane_row_capacity(seq);
        claim_lane(seq);
        let claimed = test_hooks::lane_row_capacity(seq);
        end_lane(seq);
        let ended = test_hooks::lane_row_capacity(seq);
        test_hooks::disable();
        let _ = std::fs::remove_file(&path);
        assert_eq!(unclaimed, Some(0), "an unclaimed lane holds no row buffer");
        assert!(
            claimed.is_some_and(|rows| rows >= ROW_BUFFER),
            "a claimed lane reserves its rows: {claimed:?}"
        );
        assert_eq!(ended, None, "an ended lane leaves no timeline");
    }
}
