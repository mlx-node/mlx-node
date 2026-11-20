//! A load's warm-up (Splash's bootstrap warm-up): before the model serves,
//! real scheduled turns on a throwaway scheduler run the prefill row
//! classes, decode widths, speculative shapes and a stored state's restore
//! copy a first turn would otherwise compile pipelines for, each turn ended
//! by its cancel. The pool keeps its size, the SSD state tier and the
//! states it took back are set aside, no scratch turn (nor its replay after
//! a preemption) reuses or publishes a cache entry, the model's state must
//! read as it did before, and the model's governor books are put back. Its
//! claims are the start's (`Floor`, one attempt, nothing freed): a refused
//! optional step is skipped; a refused required step fails the load, after
//! the start's retry for a host or driver refusal; a claim left open or any
//! other error fails the load. A load that serves no chat turn runs none.

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;

use napi::bindgen_prelude::{Error, Result};
use napi_derive::napi;

use crate::engine::backend::ResetScope;
use crate::engine::governor::governor::{
    Failure, ModelBooks, START_POLL_SECONDS, START_WARMUP, StartScope, StartupRetryWindow,
    take_start_refusal,
};
use crate::engine::hybrid_scheduler::{
    HybridSchedulerBackend, HybridSchedulerState, ScratchRequest, ScratchStep,
    scheduler_max_num_seqs_for,
};
use crate::engine::load_progress::LoadProgressCell;
use crate::engine::model_command::ModelCommand;
use crate::engine::scheduler::PREFILL_CHUNK_ROWS;
use crate::engine::session::CHAT_SESSION_CANCELLED;
use crate::transformer::paged_kv_cache_adapter::SeqId;

#[cfg(test)]
thread_local! {
    /// The warm-up's first prefix preparation fails with this text.
    pub(crate) static FAIL_PREFIX: std::cell::Cell<Option<&'static str>> =
        const { std::cell::Cell::new(None) };
    /// The warm-up's first step that plans rows fails in its executor with
    /// this text.
    pub(crate) static FAIL_STEP: std::cell::Cell<Option<&'static str>> =
        const { std::cell::Cell::new(None) };
    /// The first decode row's turn is preempted once.
    pub(crate) static PREEMPT_FIRST_DECODE: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
    /// The preempted turn's blocks leave the prefix cache before its replay.
    pub(crate) static PURGE_AFTER_PREEMPT: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
    /// The next scratch turn ends after one token instead of by its cancel.
    pub(crate) static FINISH_NEXT_TURN: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
    /// Step A admits its full rows without checking that they fit.
    pub(crate) static OVERSIZE_MAX_PREFILL: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
    /// Accounts no serving model has are published before the warm-up's
    /// end.
    pub(crate) static STALE_ACCOUNTS: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
    /// A claim for the model is opened after the steps and never closed.
    pub(crate) static LEAK_CLAIM: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
    /// A lane of the model is suspended after the steps.
    pub(crate) static SUSPEND_AFTER_STEPS: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
    /// The model's next control pass time, read after the steps.
    pub(crate) static CONTROL_AFTER_STEPS: std::cell::Cell<Option<f64>> =
        const { std::cell::Cell::new(None) };
    /// Runs as each step begins, with the step.
    pub(crate) static BEFORE_STEP: std::cell::RefCell<Option<Box<dyn FnMut(Step)>>> =
        const { std::cell::RefCell::new(None) };
}

fn oversize_max_prefill() -> bool {
    #[cfg(test)]
    {
        OVERSIZE_MAX_PREFILL.take()
    }
    #[cfg(not(test))]
    {
        false
    }
}

/// Decode steps (or verify cycles) each width runs.
const GOAL_STEPS: usize = 2;
/// Drive passes one goal may take before its step counts as memory-limited.
const MAX_PASSES: usize = 256;
const LONG_DRAFTS: u32 = 7;
const SHORT_DRAFTS: u32 = 1;
/// Rows past a 64-row multiple a ragged row class runs.
const RAGGED_ROWS: u32 = 32;

/// What a load's warm-up did.
#[napi(object, js_name = "WarmupReport")]
#[derive(Clone, Debug, PartialEq)]
pub struct WarmupReport {
    /// Wall time of the warm-up.
    pub ms: f64,
    pub steps: u32,
    /// Steps not run, or run smaller than planned, by the attempt that
    /// warmed the model (a retry runs a refused attempt's steps again).
    pub skipped: u32,
    /// 1 when a cache reset, in any attempt, made the model's state read as
    /// before.
    pub reset: u32,
    /// Pipelines and source libraries created during the warm-up.
    pub pipelines: u32,
    /// Time inside those creations.
    pub compile_ms: f64,
    /// The process allocator's peak over every attempt, read after the
    /// warm-up.
    pub peak_bytes: f64,
}

/// One progress step of the plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Step {
    /// One turn of `slice_rows` prompt rows in one slice, then two decode
    /// steps.
    MaxPrefill,
    /// A prefix hit's state restore copy on that turn's GDN cell.
    RestoreCopy,
    /// One more 1-row turn: two decode steps with `width` rows.
    Widen(u32),
    /// Lanes cancelled down to `width`: two decode steps at short history.
    Narrow(u32),
    /// One prefill slice of this many rows.
    Class(u32),
    /// Two verify cycles of `width` speculative rows at `drafts` drafts.
    Speculative { width: u32, drafts: u32 },
}

impl Step {
    /// A step the serving model needs (Splash's required warm-up): the
    /// largest prefill, each row class and width 1. A refusal of its claim
    /// fails the load; any other step's is skipped.
    pub(crate) fn required(self) -> bool {
        matches!(
            self,
            Step::MaxPrefill
                | Step::Class(_)
                | Step::Narrow(1)
                | Step::Speculative {
                    width: 1,
                    drafts: LONG_DRAFTS,
                }
        )
    }
}

/// The steps of one load's warm-up.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct WarmupPlan {
    pub lanes: u32,
    pub slice_rows: u32,
    pub steps: Vec<Step>,
}

impl WarmupPlan {
    /// The plan for `inner`: its granted lanes, its row classes up to
    /// `slice_rows`, and the speculative widths when it drafts on the
    /// scheduled lane.
    pub(crate) fn for_backend<B: HybridSchedulerBackend>(inner: &B, slice_rows: u32) -> Self {
        let lanes = Self::lanes(
            inner.scheduler_capacity(),
            HybridSchedulerState::<B>::force_serial(),
        );
        Self::new(lanes, slice_rows, inner.supports_scheduled_speculation())
    }

    pub(crate) fn minimal_for_backend<B: HybridSchedulerBackend>(
        inner: &B,
        slice_rows: u32,
    ) -> Self {
        let lanes = Self::lanes(
            inner.scheduler_capacity(),
            HybridSchedulerState::<B>::force_serial(),
        );
        Self::minimal(lanes, slice_rows, inner.supports_scheduled_speculation())
    }

    pub(crate) fn minimal(lanes: u32, slice_rows: u32, speculative: bool) -> Self {
        let mut steps = vec![Step::MaxPrefill, Step::RestoreCopy];
        steps.extend((2..=lanes).map(Step::Widen));
        if speculative {
            steps.push(Step::Speculative {
                width: 1,
                drafts: SHORT_DRAFTS,
            });
        }
        Self {
            lanes,
            slice_rows,
            steps,
        }
    }

    /// The granted lanes; one when every turn runs whole.
    pub(crate) fn lanes(capacity: usize, force_serial: bool) -> u32 {
        if force_serial {
            return 1;
        }
        scheduler_max_num_seqs_for(capacity).max(1) as u32
    }

    pub(crate) fn new(lanes: u32, slice_rows: u32, speculative: bool) -> Self {
        let mut steps = vec![Step::MaxPrefill, Step::RestoreCopy];
        steps.extend((2..=lanes).map(Step::Widen));
        if lanes > 1 {
            steps.extend((1..lanes).rev().map(Step::Narrow));
        } else {
            steps.push(Step::Narrow(1));
        }
        steps.extend((1..=33).filter(|&rows| rows <= slice_rows).map(Step::Class));
        for rows in [64, 128, 256, 512, 1024] {
            if rows > slice_rows {
                break;
            }
            steps.push(Step::Class(rows));
            // A slice past 64 rows that leaves a ragged gemm tile selects
            // its own pipelines; serving cuts slices at block multiples.
            let ragged = if rows + RAGGED_ROWS <= slice_rows {
                rows + RAGGED_ROWS
            } else {
                rows - RAGGED_ROWS
            };
            if ragged > 64 {
                steps.push(Step::Class(ragged));
            }
        }
        if speculative {
            steps.extend((1..=lanes).map(|width| Step::Speculative {
                width,
                drafts: LONG_DRAFTS,
            }));
            steps.push(Step::Speculative {
                width: 1,
                drafts: SHORT_DRAFTS,
            });
        }
        Self {
            lanes,
            slice_rows,
            steps,
        }
    }

    pub(crate) fn classes(&self) -> Vec<u32> {
        self.steps
            .iter()
            .filter_map(|step| match step {
                Step::Class(rows) => Some(*rows),
                _ => None,
            })
            .collect()
    }
}

/// The model's state a warm-up must leave as it found it.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Snapshot {
    pub total_blocks: u32,
    pub allocated_blocks: u32,
    pub evictable_blocks: u32,
    pub reclaimable_blocks: u32,
    pub cold_enqueued: u64,
    pub cold_bytes_written: u64,
    pub cold_total_bytes: u64,
    pub recurrent_bytes: u64,
    pub retained_states: usize,
    pub per_sequence_entries: usize,
    pub store: Option<[u64; 5]>,
    /// The SSD state tier: its mode, limits, the states it took back and
    /// its write-behind, the disk copies and the writes the model counted.
    pub state_tier: Option<(
        crate::engine::governor::governor::DFlash2DiskStatus,
        [u64; 8],
    )>,
    pub speculation_owners: usize,
    pub books: ModelBooks,
}

impl Snapshot {
    pub(crate) fn take<B: HybridSchedulerBackend>(inner: &B) -> Self {
        let adapter = inner.paged_adapter();
        let blocks = adapter.and_then(|adapter| adapter.block_telemetry().ok());
        let cold = adapter
            .and_then(|adapter| adapter.cold_tier())
            .map(|tier| tier.manager.stats());
        let status = inner.dflash2_state_status();
        Self {
            total_blocks: blocks.as_ref().map_or(0, |blocks| blocks.total_blocks),
            allocated_blocks: blocks.as_ref().map_or(0, |blocks| blocks.allocated_blocks),
            evictable_blocks: blocks.as_ref().map_or(0, |blocks| blocks.evictable_blocks),
            reclaimable_blocks: blocks
                .as_ref()
                .map_or(0, |blocks| blocks.reclaimable_blocks),
            cold_enqueued: cold.as_ref().map_or(0, |cold| cold.enqueued),
            cold_bytes_written: cold.as_ref().map_or(0, |cold| cold.bytes_written),
            cold_total_bytes: adapter
                .and_then(|adapter| adapter.cold_tier())
                .map_or(0, |tier| tier.manager.total_bytes()),
            recurrent_bytes: inner.scheduled_recurrent_bytes(),
            retained_states: inner.retained_state_entries(),
            per_sequence_entries: inner.per_sequence_entries(),
            store: status.map(|store| {
                [
                    store.entries,
                    store.bytes,
                    store.checkpoint_entries,
                    store.checkpoint_bytes,
                    store.in_use,
                ]
            }),
            state_tier: status.map(|status| {
                (
                    status.disk,
                    [
                        status.disk_entries,
                        status.disk_bytes,
                        status.disk_state_publications,
                        status.offloads,
                        status.offload_failures,
                        status.invalidations,
                        status.disk_promotions,
                        status.disk_promotions_skipped,
                    ],
                )
            }),
            speculation_owners: inner.scheduled_speculation_owners(),
            books: inner
                .memory_governor()
                .map(|governor| governor.model_books(inner.memory_model()))
                .unwrap_or_default(),
        }
    }

    /// The rows of the model's state that differ from `before`; the
    /// governor's books are not rows: the warm-up puts them back.
    fn differs(&self, before: &Self) -> Vec<&'static str> {
        let mut rows = Vec::new();
        let mut row = |name: &'static str, same: bool| {
            if !same {
                rows.push(name);
            }
        };
        row("pool size", self.total_blocks == before.total_blocks);
        row(
            "allocated blocks",
            self.allocated_blocks == before.allocated_blocks,
        );
        row(
            "prefix cache",
            self.evictable_blocks == before.evictable_blocks
                && self.reclaimable_blocks == before.reclaimable_blocks,
        );
        row(
            "disk cache",
            self.cold_enqueued == before.cold_enqueued
                && self.cold_bytes_written == before.cold_bytes_written
                && self.cold_total_bytes == before.cold_total_bytes,
        );
        row(
            "recurrent state",
            self.recurrent_bytes == before.recurrent_bytes
                && self.retained_states == before.retained_states,
        );
        row(
            "per-sequence tables",
            self.per_sequence_entries == before.per_sequence_entries,
        );
        row("state store", self.store == before.store);
        row("state tier", self.state_tier == before.state_tier);
        row(
            "speculation owners",
            self.speculation_owners == before.speculation_owners,
        );
        rows
    }
}

/// Puts back what the scratch turns moved in `inner`'s governor books and
/// republishes the serving model's accounts and states over the last
/// scratch pass's.
fn put_back_books<B: HybridSchedulerBackend>(inner: &B, attached: &ModelBooks, after: &ModelBooks) {
    let model = inner.memory_model();
    let Some(governor) = inner.memory_governor().filter(|_| model != 0) else {
        return;
    };
    #[cfg(test)]
    if STALE_ACCOUNTS.take() {
        governor.publish_accounts(
            model,
            crate::engine::governor::governor::MemoryAccounts {
                kv: 1 << 40,
                states: 1 << 40,
                embeddings: 1 << 40,
            },
        );
    }
    if after != attached {
        governor.restore_model_books(model, attached);
    }
    if let Some(accounts) = inner.memory_accounts() {
        governor.publish_accounts(model, accounts);
    }
    if let Some(states) = inner.dflash2_state_status() {
        governor.publish_dflash2_states(model, states);
    }
}

/// What the warm-up ran, for its tests.
#[derive(Clone, Debug, Default)]
pub(crate) struct WarmupTrace {
    /// The prefill slices the row-class turns ran, in order.
    pub class_rows: Vec<u32>,
    /// Each step's decode rows.
    pub decode_widths: Vec<usize>,
    /// Every sequence a scratch turn ran as.
    pub seq_ids: Vec<SeqId>,
    /// The steps counted skipped, in order.
    pub skipped: Vec<Step>,
}

/// A warmed model, the sequence id serving continues from, and the report.
pub(crate) struct Warmed<B> {
    pub inner: B,
    pub next_seq_id: SeqId,
    pub report: WarmupReport,
    pub trace: WarmupTrace,
}

/// The load's error when the model's state reads changed in `rows`.
fn left_changed(when: &str, rows: &[&str]) -> Error {
    Error::from_reason(format!(
        "load warm-up left the model changed {when}: {}",
        rows.join(", ")
    ))
}

/// A memory limit ends the turn: its step is skipped, or fails the load
/// when it is required and a claim was refused; anything else fails the
/// load.
fn memory_limited(error: &Error) -> bool {
    error.reason.starts_with("capacity_exhausted:")
        || error.reason.starts_with("context_length_exceeded:")
}

/// The serving scheduler of a load: a paged load that will serve chat
/// turns (`warm`) warms up first; a load with no pool, or one whose model a
/// tool drives directly, starts at once and reports no warm-up.
pub(crate) fn serving_scheduler<B: HybridSchedulerBackend>(
    inner: B,
    progress: &LoadProgressCell,
    warm: bool,
) -> Result<(HybridSchedulerState<B>, Option<WarmupReport>)> {
    if !warm || !inner.scheduler_cache_available() {
        return Ok((HybridSchedulerState::new(inner)?, None));
    }
    finish_serving_scheduler(run(inner, progress)?)
}

pub(crate) fn serving_scheduler_minimal<B: HybridSchedulerBackend>(
    inner: B,
    progress: &LoadProgressCell,
    warm: bool,
    slice_rows: u32,
) -> Result<(HybridSchedulerState<B>, Option<WarmupReport>)> {
    if !warm || !inner.scheduler_cache_available() {
        return Ok((HybridSchedulerState::new(inner)?, None));
    }
    let plan = WarmupPlan::minimal_for_backend(&inner, slice_rows);
    finish_serving_scheduler(run_plan(inner, progress, plan)?)
}

fn finish_serving_scheduler<B: HybridSchedulerBackend>(
    warmed: Warmed<B>,
) -> Result<(HybridSchedulerState<B>, Option<WarmupReport>)> {
    let report = &warmed.report;
    tracing::info!(
        target: "mlx_core::inference",
        steps = report.steps,
        skipped = report.skipped,
        reset = report.reset,
        turns = warmed.trace.seq_ids.len(),
        pipelines = report.pipelines,
        compile_ms = report.compile_ms,
        ms = report.ms,
        "load warm-up"
    );
    Ok((
        HybridSchedulerState::resume(warmed.inner, warmed.next_seq_id)?,
        Some(warmed.report),
    ))
}

/// Warms `inner` up on [`PREFILL_CHUNK_ROWS`]-row prefills.
pub(crate) fn run<B: HybridSchedulerBackend>(
    inner: B,
    progress: &LoadProgressCell,
) -> Result<Warmed<B>> {
    run_with(inner, progress, PREFILL_CHUNK_ROWS)
}

/// What one attempt of the warm-up ended with.
enum Attempt<B> {
    Warmed(Warmed<B>),
    /// A claim of a required step was refused: the model reads as loaded
    /// and its sequences continue from `next_seq_id`.
    Refused {
        inner: B,
        next_seq_id: SeqId,
        step: Step,
        failure: Failure,
        reset: u32,
    },
}

/// The warm-up under the start's claims. A refused required step runs the
/// warm-up again after a poll while the start's window retries its
/// failure (the host or the Metal driver, 30 s); otherwise it fails the
/// load.
pub(crate) fn run_with<B: HybridSchedulerBackend>(
    inner: B,
    progress: &LoadProgressCell,
    slice_rows: u32,
) -> Result<Warmed<B>> {
    let plan = WarmupPlan::for_backend(&inner, slice_rows);
    run_plan(inner, progress, plan)
}

pub(crate) fn run_plan<B: HybridSchedulerBackend>(
    mut inner: B,
    progress: &LoadProgressCell,
    plan: WarmupPlan,
) -> Result<Warmed<B>> {
    let started = Instant::now();
    let _start = StartScope::enter();
    let _loading = crate::engine::pipeline_counter::loading_scope();
    let pipelines_before = crate::engine::pipeline_counter::read();
    let mut window = StartupRetryWindow::default();
    let mut next_seq_id: SeqId = 1;
    let mut reset = 0;
    loop {
        match attempt(inner, progress, &plan, next_seq_id)? {
            Attempt::Warmed(mut warmed) => {
                warmed.report.reset = warmed.report.reset.max(reset);
                let pipelines_after = crate::engine::pipeline_counter::read();
                let (pipelines, compile_ms) = match (&pipelines_before, &pipelines_after) {
                    (Some(before), Some(after)) => (
                        after.created_loading.saturating_sub(before.created_loading),
                        (after.loading_ms - before.loading_ms).max(0.0),
                    ),
                    _ => (0, 0.0),
                };
                if pipelines_before.is_some()
                    && let Some(after) = &pipelines_after
                {
                    crate::engine::pipeline_counter::check_first_warmup(
                        !warmed.trace.seq_ids.is_empty(),
                        after.created_loading.saturating_add(after.created_serving),
                    );
                }
                warmed.report.pipelines = pipelines;
                warmed.report.compile_ms = compile_ms;
                warmed.report.ms = started.elapsed().as_secs_f64() * 1000.0;
                return Ok(warmed);
            }
            Attempt::Refused {
                inner: back,
                next_seq_id: next,
                step,
                failure,
                reset: refused_reset,
            } => {
                reset = reset.max(refused_reset);
                let text = format!(
                    "memory plan: the load's warm-up ({step:?}) could not be allocated: {}",
                    Failure::name(Some(failure))
                );
                let Some(governor) = back.memory_governor() else {
                    return Err(Error::from_reason(text));
                };
                if !window.retry(Some((failure, START_WARMUP)), governor.now(), &text) {
                    return Err(Error::from_reason(text));
                }
                governor.sleep(START_POLL_SECONDS);
                inner = back;
                next_seq_id = next;
            }
        }
    }
}

/// One run of the plan on a scratch scheduler whose sequences start at
/// `next_seq_id`, then the checks that the model reads as loaded.
fn attempt<B: HybridSchedulerBackend>(
    mut inner: B,
    progress: &LoadProgressCell,
    plan: &WarmupPlan,
    next_seq_id: SeqId,
) -> Result<Attempt<B>> {
    progress.begin_kernels(plan.steps.len() as u32);
    let attached = Snapshot::take(&inner);
    let held_tier = inner.hold_state_tier();
    let before = Snapshot::take(&inner);
    let block_size = inner.scheduler_block_size()?.max(1);
    let vocab = inner.token_vocab_size();
    if let Some(adapter) = inner.paged_adapter_mut() {
        adapter.set_growth_hold(true);
    }
    let (sender, idle) = tokio::sync::mpsc::unbounded_channel::<ModelCommand<B::FamilyCommand>>();
    drop(sender);
    let mut run = Run {
        scratch: HybridSchedulerState::scratch_from(inner, next_seq_id)?,
        idle,
        live: Vec::new(),
        progress,
        plan,
        block_size,
        vocab,
        salt: format!("warmup-{:016x}", rand::random::<u64>()),
        turns: 0,
        finalized: false,
        limited: false,
        failure: None,
        refused: None,
        trace: WarmupTrace::default(),
    };
    #[cfg(test)]
    if let Some(reason) = FAIL_PREFIX.take() {
        run.scratch.fail_next_prefix(reason);
    }
    #[cfg(test)]
    if let Some(reason) = FAIL_STEP.take() {
        run.scratch.fail_next_step(reason);
    }
    run.all_steps()?;
    #[cfg(test)]
    run.after_steps_hooks();
    let preempted = run.scratch.preemptions() > 0;
    let (finalized, refused, trace) = (run.finalized, run.refused, std::mem::take(&mut run.trace));
    let (mut inner, next_seq_id) = run.scratch.into_parts();
    if let Some(adapter) = inner.paged_adapter_mut() {
        adapter.set_growth_hold(false);
    }
    crate::array::memory::synchronize_and_clear_cache_checked("load warm-up")?;
    let peak_bytes = crate::array::memory::get_peak_memory();
    let rows = Snapshot::take(&inner).differs(&before);
    let mut reset = 0;
    if finalized || preempted || !rows.is_empty() {
        inner.reset_caches(ResetScope::Command)?;
        reset = 1;
        let rows = Snapshot::take(&inner).differs(&before);
        if !rows.is_empty() {
            return Err(left_changed("after a cache reset", &rows));
        }
    }
    if let Some(held) = held_tier {
        inner.release_state_tier(held);
    }
    let after = Snapshot::take(&inner);
    let rows = after.differs(&attached);
    if !rows.is_empty() {
        return Err(left_changed("with its state tier back", &rows));
    }
    if after.books.open_claims != attached.books.open_claims {
        return Err(Error::from_reason("load warm-up left a memory claim open"));
    }
    put_back_books(&inner, &attached.books, &after.books);
    if let Some((step, failure)) = refused {
        return Ok(Attempt::Refused {
            inner,
            next_seq_id,
            step,
            failure,
            reset,
        });
    }
    Ok(Attempt::Warmed(Warmed {
        inner,
        next_seq_id,
        report: WarmupReport {
            ms: 0.0,
            steps: plan.steps.len() as u32,
            skipped: trace.skipped.len() as u32,
            reset,
            pipelines: 0,
            compile_ms: 0.0,
            peak_bytes,
        },
        trace,
    }))
}

struct Live {
    seq_id: SeqId,
    reply: tokio::sync::oneshot::Receiver<Result<crate::engine::types::ChatResult>>,
    cancel: Arc<std::sync::atomic::AtomicBool>,
}

struct Run<'a, B: HybridSchedulerBackend> {
    scratch: HybridSchedulerState<B>,
    idle: tokio::sync::mpsc::UnboundedReceiver<ModelCommand<B::FamilyCommand>>,
    live: Vec<Live>,
    progress: &'a LoadProgressCell,
    plan: &'a WarmupPlan,
    block_size: u32,
    vocab: u32,
    salt: String,
    turns: u64,
    /// A scratch turn finalized: its blocks may be in the prefix cache.
    finalized: bool,
    /// A scratch turn met a memory limit since the last step began.
    limited: bool,
    /// What failed a turn of the step in hand.
    failure: Option<Failure>,
    /// The first required step a claim refused.
    refused: Option<(Step, Failure)>,
    trace: WarmupTrace,
}

impl<B: HybridSchedulerBackend> Run<'_, B> {
    #[cfg(test)]
    fn after_steps_hooks(&self) {
        let model = self.scratch.inner.memory_model();
        let Some(governor) = self.scratch.inner.memory_governor() else {
            return;
        };
        CONTROL_AFTER_STEPS.set(Some(governor.model_books(model).next_control));
        if SUSPEND_AFTER_STEPS.take() {
            governor.lane_suspended(model, governor.now(), None);
        }
        if LEAK_CLAIM.take() {
            use crate::engine::governor::ledger::{
                ClaimClass, ClaimRank, ClaimSpec, NeedTiming, ReachClass,
            };
            let claim = governor.open(
                ClaimSpec {
                    cls: ClaimClass::InService,
                    request_id: 1,
                    rank: ClaimRank::default(),
                    reach: ReachClass::InUse,
                    timing: NeedTiming::CanWait,
                },
                model,
            );
            std::mem::forget(claim);
        }
    }

    fn all_steps(&mut self) -> Result<()> {
        let reached = self.max_prefill()?;
        self.done();
        if self.refused.is_some() {
            return self.cancel_all();
        }
        if reached {
            self.restore_copy()?;
        } else {
            self.skip(Step::RestoreCopy);
        }
        self.done();
        for width in 2..=self.plan.lanes {
            self.widen(width)?;
            self.done();
        }
        let narrow: Vec<u32> = self
            .plan
            .steps
            .iter()
            .filter_map(|step| match step {
                Step::Narrow(width) => Some(*width),
                _ => None,
            })
            .collect();
        if !narrow.is_empty() {
            self.cancel_oldest()?;
        }
        for width in narrow {
            if !self.narrow(width)? {
                self.skip(Step::Narrow(width));
            }
            self.done();
            if self.refused.is_some() {
                return self.cancel_all();
            }
        }
        self.cancel_all()?;
        self.classes()?;
        self.cancel_all()?;
        if self.refused.is_some() {
            return Ok(());
        }
        let speculative: Vec<(u32, u32)> = self
            .plan
            .steps
            .iter()
            .filter_map(|step| match step {
                Step::Speculative { width, drafts } => Some((*width, *drafts)),
                _ => None,
            })
            .collect();
        let mut limited = false;
        let mut drafts_before = None;
        for (width, drafts) in speculative {
            if drafts_before.is_some_and(|before| before != drafts) {
                self.cancel_all()?;
            }
            drafts_before = Some(drafts);
            limited = limited || !self.speculative(width, drafts)?;
            if limited {
                self.skip(Step::Speculative { width, drafts });
            }
            self.done();
            if self.refused.is_some() {
                return self.cancel_all();
            }
        }
        self.cancel_all()
    }

    fn done(&self) {
        self.progress.step_done();
    }

    /// A step begins: no turn of it has met a memory limit yet.
    fn begin(&mut self, _step: Step) {
        #[cfg(test)]
        BEFORE_STEP.with_borrow_mut(|hook| {
            if let Some(hook) = hook {
                hook(_step);
            }
        });
        self.limited = false;
        self.failure = None;
        take_start_refusal();
    }

    /// What refused a turn of `step` fails the load when `step` is
    /// required.
    fn record_refusal(&mut self, step: Step) {
        if let Some(failure) = self.failure.take()
            && step.required()
        {
            self.refused.get_or_insert((step, failure));
        }
    }

    fn skip(&mut self, step: Step) {
        self.record_refusal(step);
        self.trace.skipped.push(step);
    }

    fn free_blocks(&self) -> u32 {
        self.scratch
            .inner
            .paged_adapter()
            .and_then(|adapter| adapter.block_telemetry().ok())
            .map_or(0, |blocks| blocks.free_blocks)
    }

    fn blocks(&self, rows: u32) -> u32 {
        rows.div_ceil(self.block_size) + 1
    }

    /// Step A: whether it ran. A pool too small for its rows runs it on the
    /// rows that fit, counted skipped.
    fn max_prefill(&mut self) -> Result<bool> {
        self.begin(Step::MaxPrefill);
        let growth = (2 * self.plan.lanes).div_ceil(self.block_size);
        let free = self.free_blocks();
        let mut rows = self.plan.slice_rows;
        let mut counted = false;
        if !oversize_max_prefill() && self.blocks(rows) + growth > free {
            rows = free.saturating_sub(1 + growth) * self.block_size;
            self.skip(Step::MaxPrefill);
            counted = true;
            if rows == 0 {
                return Ok(false);
            }
        }
        let reached = match self.admit(rows, Vec::new(), None)? {
            Some(seq_id) => self.run_until(|step| step.decode == [seq_id])?,
            None => false,
        };
        if !reached {
            if counted {
                self.record_refusal(Step::MaxPrefill);
            } else {
                self.skip(Step::MaxPrefill);
            }
        }
        Ok(reached)
    }

    /// A prefix hit's state restore copy, on the live lane's GDN cell;
    /// counted skipped when the lane holds none.
    fn restore_copy(&mut self) -> Result<()> {
        let restored = match self.live.first().map(|live| live.seq_id) {
            Some(seq_id) => self.scratch.inner.warm_state_restore(seq_id)?,
            None => false,
        };
        if !restored {
            self.skip(Step::RestoreCopy);
        }
        Ok(())
    }

    /// Step W at `width`: one more 1-row lane, two decode steps of `width`
    /// rows.
    fn widen(&mut self, width: u32) -> Result<bool> {
        self.begin(Step::Widen(width));
        if self.live.len() as u32 + 1 != width || self.blocks(1) > self.free_blocks() {
            self.skip(Step::Widen(width));
            return Ok(false);
        }
        if self.admit(1, Vec::new(), None)?.is_none() {
            self.skip(Step::Widen(width));
            return Ok(false);
        }
        let reached = self.run_until(|step| step.decode.len() == width as usize)?;
        if !reached {
            self.skip(Step::Widen(width));
        }
        Ok(reached)
    }

    /// Step N at `width`: the oldest lanes cancelled until `width` decode;
    /// with none left, one 1-row lane.
    fn narrow(&mut self, width: u32) -> Result<bool> {
        self.begin(Step::Narrow(width));
        while self.live.len() as u32 > width {
            self.cancel_oldest()?;
        }
        if self.live.is_empty()
            && width == 1
            && (self.blocks(1) > self.free_blocks() || self.admit(1, Vec::new(), None)?.is_none())
        {
            return Ok(false);
        }
        if self.live.len() as u32 != width {
            return Ok(false);
        }
        self.run_until(|step| step.decode.len() == width as usize)
    }

    /// Steps C: the row classes as pinned slices, packed into turns of at
    /// most the slice rows that fit; one decode step after each turn's last
    /// slice.
    fn classes(&mut self) -> Result<()> {
        let mut pending: std::collections::VecDeque<u32> = self.plan.classes().into();
        let mut limited = false;
        while let Some(&first) = pending.front() {
            self.begin(Step::Class(first));
            let fit = self.free_blocks().saturating_sub(1) * self.block_size;
            let limit = self.plan.slice_rows.min(fit);
            let mut turn = Vec::new();
            let mut total = 0;
            while let Some(&rows) = pending.front()
                && !limited
                && total + rows <= limit
            {
                total += rows;
                turn.push(rows);
                pending.pop_front();
            }
            if turn.is_empty() {
                if let Some(rows) = pending.pop_front() {
                    self.skip(Step::Class(rows));
                }
                self.done();
                continue;
            }
            let mut ends = Vec::with_capacity(turn.len());
            let mut end = 0;
            for &rows in &turn {
                end += rows;
                ends.push(end);
            }
            let breaks = ends[..ends.len() - 1].to_vec();
            let ran = match self.admit(total, breaks, None)? {
                Some(seq_id) => self.class_slices(seq_id, &ends)?,
                None => 0,
            };
            for _ in 0..ran {
                self.done();
            }
            for &rows in &turn[ran..] {
                self.skip(Step::Class(rows));
                self.done();
            }
            limited = ran < turn.len();
            self.cancel_all()?;
            if self.refused.is_some() {
                return Ok(());
            }
        }
        Ok(())
    }

    /// Drives `seq_id`'s slices; how many of `ends` its prefill reached
    /// before a decode step ran.
    fn class_slices(&mut self, seq_id: SeqId, ends: &[u32]) -> Result<usize> {
        let mut reached = 0u32;
        let mut decoded = false;
        for _ in 0..MAX_PASSES {
            for step in self.drive()? {
                for &(seq, rows) in &step.prefill {
                    if seq == seq_id {
                        reached += rows;
                        self.trace.class_rows.push(rows);
                    }
                }
                decoded |= step.decode.contains(&seq_id);
            }
            if decoded || self.limited || !self.is_live(seq_id) {
                break;
            }
        }
        Ok(ends.iter().filter(|&&end| end <= reached).count())
    }

    /// Step E: one more speculative 1-row lane at `drafts` drafts, two
    /// verify cycles of `width` rows.
    fn speculative(&mut self, width: u32, drafts: u32) -> Result<bool> {
        self.begin(Step::Speculative { width, drafts });
        let rows = 1 + GOAL_STEPS as u32 * (drafts + 1) + 1;
        if self.live.len() as u32 + 1 != width || self.blocks(rows) + 1 > self.free_blocks() {
            return Ok(false);
        }
        if self.admit(1, Vec::new(), Some(drafts))?.is_none() {
            return Ok(false);
        }
        self.run_until(|step| step.decode.len() == width as usize)
    }

    fn is_live(&self, seq_id: SeqId) -> bool {
        self.live.iter().any(|live| live.seq_id == seq_id)
    }

    /// Admits a scratch turn of `rows` tokens; `None` when memory held it
    /// back (it is cancelled again).
    fn admit(
        &mut self,
        rows: u32,
        breaks: Vec<u32>,
        mtp_depth: Option<u32>,
    ) -> Result<Option<SeqId>> {
        self.limited = false;
        self.turns += 1;
        let vocab = u64::from(self.vocab);
        let tokens = (0..u64::from(rows))
            .map(|i| ((1 + 7919 * i) % vocab) as u32)
            .collect();
        let turn = self.scratch.admit_scratch(ScratchRequest {
            tokens,
            breaks,
            mtp_depth,
            owner_id: format!("warmup-{}", self.turns),
            cache_salt: self.salt.clone(),
        })?;
        let seq_id = turn.seq_id;
        self.trace.seq_ids.push(seq_id);
        self.live.push(Live {
            seq_id,
            reply: turn.reply,
            cancel: turn.cancel,
        });
        self.check_replies()?;
        if self.limited || !self.is_live(seq_id) {
            return Ok(None);
        }
        if self.scratch.admission_waits_for(seq_id) {
            self.cancel(seq_id)?;
            return Ok(None);
        }
        Ok(Some(seq_id))
    }

    /// One drive pass; the rows each step granted. A pass that stopped the
    /// scheduler fails the load with its reason.
    fn drive(&mut self) -> Result<Vec<ScratchStep>> {
        self.scratch.drive(&mut self.idle);
        if let Some(error) = self.scratch.take_stop() {
            return Err(error);
        }
        self.check_replies()?;
        let steps = self.scratch.take_scratch_steps();
        #[cfg(test)]
        if PREEMPT_FIRST_DECODE.get()
            && let Some(seq_id) = steps
                .iter()
                .flat_map(|step| step.decode.iter().copied())
                .find(|&seq_id| self.is_live(seq_id))
        {
            PREEMPT_FIRST_DECODE.set(false);
            assert!(self.scratch.preempt_running(seq_id), "preempt {seq_id}");
            if PURGE_AFTER_PREEMPT.take()
                && let Some(adapter) = self.scratch.inner.paged_adapter()
            {
                adapter
                    .allocator()
                    .lock()
                    .expect("allocator")
                    .purge_prefix_cache();
            }
        }
        self.trace
            .decode_widths
            .extend(steps.iter().map(|step| step.decode.len()));
        Ok(steps)
    }

    /// Takes the replies that arrived: a cancel's is expected, a memory
    /// limit marks the step limited, any other error fails the load.
    fn check_replies(&mut self) -> Result<()> {
        let mut index = 0;
        while index < self.live.len() {
            let live = &mut self.live[index];
            match live.reply.try_recv() {
                Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {
                    index += 1;
                    continue;
                }
                Ok(Ok(_)) => self.finalized = true,
                Ok(Err(error))
                    if live.cancel.load(Ordering::Relaxed)
                        && error.reason == CHAT_SESSION_CANCELLED => {}
                Ok(Err(error)) if memory_limited(&error) => {
                    self.limited = true;
                    self.failure = self
                        .failure
                        .or(take_start_refusal().map(|(failure, _)| failure));
                }
                Ok(Err(error)) => return Err(error),
                Err(tokio::sync::oneshot::error::TryRecvError::Closed) => {
                    return Err(Error::from_reason(format!(
                        "load warm-up turn {} ended without a reply",
                        live.seq_id
                    )));
                }
            }
            self.live.remove(index);
        }
        Ok(())
    }

    /// Drives until [`GOAL_STEPS`] steps met `goal`; false when a memory
    /// limit or the pass bound came first.
    fn run_until(&mut self, mut goal: impl FnMut(&ScratchStep) -> bool) -> Result<bool> {
        let mut met = 0;
        for _ in 0..MAX_PASSES {
            met += self.drive()?.iter().filter(|step| goal(step)).count();
            if met >= GOAL_STEPS {
                return Ok(true);
            }
            if self.limited || self.live.is_empty() {
                return Ok(false);
            }
        }
        Ok(false)
    }

    fn cancel(&mut self, seq_id: SeqId) -> Result<()> {
        if let Some(live) = self.live.iter().find(|live| live.seq_id == seq_id) {
            live.cancel.store(true, Ordering::Relaxed);
        }
        self.settle(|run| !run.is_live(seq_id))
    }

    fn cancel_oldest(&mut self) -> Result<()> {
        match self.live.first().map(|live| live.seq_id) {
            Some(seq_id) => self.cancel(seq_id),
            None => Ok(()),
        }
    }

    fn cancel_all(&mut self) -> Result<()> {
        for live in &self.live {
            live.cancel.store(true, Ordering::Relaxed);
        }
        self.settle(|run| run.live.is_empty() && !run.scratch.has_turns())
    }

    fn settle(&mut self, ended: impl Fn(&Self) -> bool) -> Result<()> {
        for _ in 0..MAX_PASSES {
            if ended(self) {
                return Ok(());
            }
            self.drive()?;
        }
        if ended(self) {
            return Ok(());
        }
        Err(Error::from_reason(
            "load warm-up: a cancelled scratch turn did not end",
        ))
    }
}

#[cfg(all(test, target_os = "macos"))]
#[path = "warmup/tests.rs"]
mod tests;
