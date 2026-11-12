//! Model-neutral step scheduler for continuous batching.
//!
//! The scheduler owns request progress and ordering only. Model execution is
//! behind [`StepExecutor`], so scheduling tests do not load MLX or a model.
//! Prefill versus decode is derived on every plan from token progress; there
//! is deliberately no phase queue.
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use crate::engine::governor::allocate::Denial;
use crate::engine::governor::ledger::ClaimRank;
use crate::engine::governor::wait::{ResourceWait, retry_ready};
use crate::engine::latency::{FineLatencyHistogram, LatencyStats};
use crate::transformer::paged_kv_cache_adapter::{BlockRefusal, SeqId};
use napi_derive::napi;

/// Largest output budget a `prompt_tokens` prompt leaves in a `capacity`
/// window, or the `context_length_exceeded` rejection when it leaves none.
/// Splash's budget (frontend.py:1212-1222): a prompt must leave room for one
/// new token, and `prompt + max_new` never passes the window.
pub(crate) fn context_output_budget(prompt_tokens: usize, capacity: u32) -> napi::Result<u32> {
    let prompt = u32::try_from(prompt_tokens).unwrap_or(u32::MAX);
    if prompt >= capacity {
        return Err(napi::Error::from_reason(format!(
            "context_length_exceeded: rendered prompt has {prompt} tokens, effective active \
             context is {capacity} tokens (at most {} prompt tokens)",
            capacity.saturating_sub(1)
        )));
    }
    Ok(capacity - prompt)
}

/// Clamp a scheduled turn's requested output to the fixed per-sequence
/// context window. With `rest_of_window` the budget is
/// [`context_output_budget`]. Otherwise the final sampled token is returned
/// without another model forward, so `N` outputs consume only `N - 1`
/// additional cache positions and the budget is `window - prompt + 1`.
pub(crate) fn clamp_scheduled_output_tokens(
    prompt_tokens: u32,
    requested_max_new_tokens: u32,
    per_seq_context: u32,
    rest_of_window: bool,
) -> Result<u32, String> {
    if rest_of_window {
        return context_output_budget(prompt_tokens as usize, per_seq_context)
            .map(|max_output| requested_max_new_tokens.min(max_output))
            .map_err(|error| error.reason);
    }
    if prompt_tokens > per_seq_context {
        return Err(format!(
            "context_length_exceeded: prompt ({prompt_tokens}) exceeds scheduler per-sequence context {per_seq_context}"
        ));
    }
    let max_output = per_seq_context
        .saturating_sub(prompt_tokens)
        .saturating_add(1);
    Ok(requested_max_new_tokens.min(max_output))
}

/// Physical cache positions consumed by a scheduled turn at its output cap.
pub(crate) fn scheduled_materialized_tokens(prompt_tokens: u32, max_new_tokens: u32) -> u32 {
    prompt_tokens.saturating_add(max_new_tokens.saturating_sub(1))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TurnStatus {
    Waiting,
    Running,
    Preempted,
    WaitingForSsd,
    Draining,
}

/// Request-owned state that survives across model steps.
pub(crate) struct TurnState<P> {
    pub seq_id: SeqId,
    pub num_computed_tokens: u32,
    pub num_tokens: u32,
    pub prompt_tokens: u32,
    /// Maximum additional hypothetical query rows; never part of token_history.
    pub decode_draft_allowance: u32,
    pub pinned_prefill_breaks: Vec<u32>,
    /// The family prefills this prompt as one group of windows in one step
    /// (`HybridSchedulerBackend::scheduler_whole_prompt_prefill`); re-pinned
    /// with the breaks on every restore and replay.
    pub whole_prompt_prefill: bool,
    pub token_history: Vec<u32>,
    pub status: TurnStatus,
    pub cancelled: Option<Arc<AtomicBool>>,
    pub cancel_snapshot: bool,
    /// Full-ISL KV reservation for this request. Zero means the model does
    /// not participate in block-watermark admission.
    pub block_reservation_total: u32,
    pub block_materialized_blocks: u32,
    pub block_size: u32,
    /// Whether this row's future-growth reservation is currently charged.
    /// Preemption releases the charge together with the live block table; it
    /// is restored only after the model-specific resume preparation succeeds.
    pub block_reservation_active: bool,
    /// Constant-size recurrent/sliding state charged in the same byte budget
    /// as future paged-block growth. Zero means this row has no hybrid state.
    pub recurrent_state_bytes: u64,
    /// Preemption drops both state charges (`recurrent_state_bytes` and
    /// `scheduled_state_bytes`) until resume preparation succeeds.
    pub recurrent_state_reservation_active: bool,
    /// Off-pool state (recurrent row, draft snapshots and K/V) this turn's
    /// sequence holds once fully materialized. Admission charges the part the
    /// family does not hold yet, so a queued MTP turn keeps its draft
    /// reservation until its owner exists.
    pub scheduled_state_bytes: u64,
    pub preemptions: u32,
    /// Parked for an SSD restore: it holds the restore's blocks and runs once
    /// they land, so it is another lane to a blocked turn. A victim holds none.
    pub ssd_lane: bool,
    /// The KV a yielded lane resumes to at once (Splash
    /// `resumeKvTargetTokens`); 0 when none.
    pub resume_target: u32,
    /// Splash `Request::resourceWait`.
    pub resource_wait: ResourceWait,
    /// The engine's admission count when this turn was admitted (Splash
    /// `Request::admission`).
    pub admission: u64,
    /// Splash `Request::overtaken`: prefill units in a row that served
    /// later arrivals and not it, up to [`MAX_OVERTAKES`].
    pub overtaken: u32,
    /// The submission order a start the scheduler held before admission
    /// keeps (Splash `Request::order`).
    pub submitted_order: Option<u64>,
    pub payload: P,
    arrival_order: u64,
}

/// Model-owned replay description installed after a victim releases its live
/// cache. Decode preemption rebuilds only the already-computed history and then
/// returns to the pending token; prefill preemption rebuilds the original
/// prompt and still samples its first output token.
pub(crate) struct PreemptionReplay {
    pub tokens: Vec<u32>,
    pub cached_prefix: u32,
    pub suppress_sample: bool,
}

/// Rewind scheduler-visible progress after a victim releases its live cache.
/// The final known decode token remains pending in `token_history`; replay only
/// rebuilds the tokens that were already computed and never emits them again.
pub(crate) fn install_preemption_replay<P>(
    turn: &mut TurnState<P>,
    original_prompt: &[u32],
    long_prefill_tokens: u32,
) -> Result<PreemptionReplay, String> {
    if long_prefill_tokens == 0 {
        return Err("preemption replay chunk size must be positive".to_string());
    }
    let original_prompt_len = u32::try_from(original_prompt.len())
        .map_err(|_| "preemption prompt length exceeds u32::MAX".to_string())?;
    let (tokens, target, suppress_sample) = if turn.num_computed_tokens < original_prompt_len {
        (original_prompt.to_vec(), original_prompt_len, false)
    } else {
        let target = turn.num_computed_tokens;
        let target_usize = usize::try_from(target)
            .map_err(|_| "preemption replay target exceeds usize::MAX".to_string())?;
        if target_usize > turn.token_history.len() {
            return Err(format!(
                "request {}: preemption replay target {target} exceeds token history {}",
                turn.seq_id,
                turn.token_history.len()
            ));
        }
        (turn.token_history[..target_usize].to_vec(), target, true)
    };
    if target == 0 {
        return Err(format!(
            "request {}: preemption replay cannot target an empty prefix",
            turn.seq_id
        ));
    }

    turn.num_computed_tokens = 0;
    turn.prompt_tokens = target;
    // Replay prefills from scratch: the family's plain slice grid, no
    // cached-prefix start and no extra breaks (those re-pin on restore).
    turn.pinned_prefill_breaks =
        super::hybrid_scheduler::prefill_slice_ends(0, target, long_prefill_tokens, []);
    turn.block_materialized_blocks = 0;
    turn.block_reservation_active = false;
    turn.recurrent_state_reservation_active = false;
    Ok(PreemptionReplay {
        tokens,
        cached_prefix: 0,
        suppress_sample,
    })
}

impl<P> TurnState<P> {
    pub fn try_new_recover_payload(
        seq_id: SeqId,
        token_history: Vec<u32>,
        num_computed_tokens: u32,
        pinned_prefill_breaks: Vec<u32>,
        cancelled: Option<Arc<AtomicBool>>,
        payload: P,
    ) -> Result<Self, (String, P)> {
        let prompt_tokens = match u32::try_from(token_history.len()) {
            Ok(prompt_tokens) => prompt_tokens,
            Err(_) => {
                return Err(("token history length exceeds u32::MAX".to_string(), payload));
            }
        };
        if num_computed_tokens > prompt_tokens {
            return Err((
                format!(
                    "request {seq_id}: computed tokens {num_computed_tokens} exceed prompt tokens {prompt_tokens}"
                ),
                payload,
            ));
        }
        let mut prior = 0;
        for &boundary in &pinned_prefill_breaks {
            if boundary <= prior || boundary > prompt_tokens {
                return Err((
                    format!(
                        "request {seq_id}: invalid pinned prefill boundary {boundary} after {prior} for prompt length {prompt_tokens}"
                    ),
                    payload,
                ));
            }
            prior = boundary;
        }
        Ok(Self {
            seq_id,
            num_computed_tokens,
            num_tokens: prompt_tokens,
            prompt_tokens,
            decode_draft_allowance: 0,
            pinned_prefill_breaks,
            whole_prompt_prefill: false,
            token_history,
            status: TurnStatus::Waiting,
            cancelled,
            cancel_snapshot: false,
            block_reservation_total: 0,
            block_materialized_blocks: 0,
            block_size: 0,
            block_reservation_active: true,
            recurrent_state_bytes: 0,
            recurrent_state_reservation_active: true,
            scheduled_state_bytes: 0,
            preemptions: 0,
            ssd_lane: false,
            resume_target: 0,
            resource_wait: ResourceWait::default(),
            admission: 0,
            overtaken: 0,
            submitted_order: None,
            payload,
            arrival_order: 0,
        })
    }

    #[cfg(test)]
    pub fn new(
        seq_id: SeqId,
        token_history: Vec<u32>,
        num_computed_tokens: u32,
        pinned_prefill_breaks: Vec<u32>,
        cancelled: Option<Arc<AtomicBool>>,
        payload: P,
    ) -> Result<Self, String> {
        Self::try_new_recover_payload(
            seq_id,
            token_history,
            num_computed_tokens,
            pinned_prefill_breaks,
            cancelled,
            payload,
        )
        .map_err(|(error, _)| error)
    }

    /// Splash `Engine::rankOf`: no request priority; a decoding lane
    /// ranks above a prefilling one, then by the tokens it completed, then
    /// by arrival.
    pub fn claim_rank(&self) -> ClaimRank {
        ClaimRank {
            priority: 1,
            decoding: self.num_computed_tokens >= self.prompt_tokens,
            completed_tokens: u64::from(self.num_computed_tokens),
            earliness: u64::MAX - self.arrival_order,
        }
    }

    /// Its submission order (Splash `Engine::submissionOrder`).
    pub fn arrival_order(&self) -> u64 {
        self.arrival_order
    }

    /// Splash `Engine::yieldsBefore`: compared without earliness.
    pub fn yields_before(&self, other: &Self) -> bool {
        let (left, right) = (self.claim_rank(), other.claim_rank());
        (left.priority, left.decoding, left.completed_tokens)
            < (right.priority, right.decoding, right.completed_tokens)
    }

    fn snapshot_cancel(&mut self) {
        self.cancel_snapshot = self
            .cancelled
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::Relaxed));
    }

    fn next_prefill_boundary(&self) -> u32 {
        self.pinned_prefill_breaks
            .iter()
            .copied()
            .find(|&boundary| boundary > self.num_computed_tokens)
            .unwrap_or(self.prompt_tokens)
    }

    pub fn with_block_reservation(
        mut self,
        total_blocks: u32,
        materialized_blocks: u32,
        block_size: u32,
    ) -> Self {
        self.block_reservation_total = total_blocks;
        self.block_materialized_blocks = materialized_blocks.min(total_blocks);
        self.block_size = block_size;
        self
    }

    pub fn with_recurrent_state_reservation(mut self, bytes: u64) -> Self {
        self.recurrent_state_bytes = bytes;
        self.recurrent_state_reservation_active = true;
        self
    }

    pub fn with_scheduled_state_reservation(mut self, bytes: u64) -> Self {
        self.scheduled_state_bytes = bytes;
        self
    }

    fn unallocated_reserved_blocks(&self) -> u32 {
        if !self.block_reservation_active
            || self.block_reservation_total == 0
            || self.block_size == 0
        {
            return 0;
        }
        self.block_reservation_total
            .saturating_sub(self.block_materialized_blocks)
    }

    fn reserved_recurrent_state_bytes(&self) -> u64 {
        if self.recurrent_state_reservation_active {
            self.recurrent_state_bytes
        } else {
            0
        }
    }

    fn outstanding_scheduled_state_bytes(&self, held: u64) -> u64 {
        if self.recurrent_state_reservation_active {
            self.scheduled_state_bytes.saturating_sub(held)
        } else {
            0
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct BlockTelemetry {
    pub total_blocks: u32,
    pub free_blocks: u32,
    /// Cache-only blocks `allocate` may take; marked chains out.
    pub reclaimable_blocks: u32,
    pub allocated_blocks: u32,
}

/// Unified-memory view used by hybrid admission. Paged blocks and recurrent
/// state are deliberately expressed in bytes so one cannot be admitted by a
/// budget that ignores the other.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct MemoryTelemetry {
    pub capacity_bytes: u64,
    pub free_bytes: u64,
    pub reclaimable_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct MemoryAdmission {
    pub admitted: bool,
    pub watermark_bytes: u64,
    pub reserved_block_bytes: u64,
    pub reserved_state_bytes: u64,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn memory_admission_decision(
    telemetry: MemoryTelemetry,
    existing_unallocated_blocks: u32,
    candidate_blocks: u32,
    bytes_per_block: u64,
    existing_state_bytes: u64,
    candidate_state_bytes: u64,
    has_live_rows: bool,
    watermark_fraction: f64,
) -> MemoryAdmission {
    let fraction = if watermark_fraction.is_finite() {
        watermark_fraction.clamp(0.0, 1.0)
    } else {
        0.05
    };
    let watermark_bytes = if has_live_rows {
        ((telemetry.capacity_bytes as f64) * fraction).ceil() as u64
    } else {
        0
    };
    let reserved_block_bytes =
        u64::from(existing_unallocated_blocks.saturating_add(candidate_blocks))
            .saturating_mul(bytes_per_block);
    let reserved_state_bytes = existing_state_bytes.saturating_add(candidate_state_bytes);
    let available = telemetry
        .free_bytes
        .saturating_add(telemetry.reclaimable_bytes);
    MemoryAdmission {
        admitted: reserved_block_bytes
            .saturating_add(reserved_state_bytes)
            .saturating_add(watermark_bytes)
            <= available,
        watermark_bytes,
        reserved_block_bytes,
        reserved_state_bytes,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum StepKind {
    Prefill,
    Decode,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct StepRow {
    pub seq_id: SeqId,
    pub kind: StepKind,
    pub token_start: u32,
    pub num_tokens: u32,
    pub cancel_snapshot: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct StepPlan {
    pub global_step: u64,
    pub rows: Vec<StepRow>,
    /// A lane decodes beside the step's prefill rows, planned or not
    /// (Splash `Scheduler::contended`).
    pub contended: bool,
    /// The leader the measured cut gave fewer rows (Splash
    /// `Planned::boundCuts`).
    pub leader_cut: Option<SeqId>,
}

/// Splash `ExecutionLimits::prefillTokenBudget`.
pub(crate) const PREFILL_CHUNK_ROWS: u32 = 2048;
/// Splash `kFloorRows`: smaller contended cuts lose prefill throughput.
pub(crate) const PREFILL_FLOOR_ROWS: u32 = 512;
/// Splash `kMinimumPrefillRows`: the pair bound halves a leader no lower.
pub(crate) const PREFILL_MIN_ROWS: u32 = 64;
/// Splash `ExecutionLimits::maximumBatchWidth`: lanes in one prefill unit.
pub(crate) const PREFILL_MAX_LANES: usize = 4;
/// Splash `kMaximumOvertakes`.
pub(crate) const MAX_OVERTAKES: u32 = PREFILL_MAX_LANES as u32 - 1;
/// Splash `SchedulerConfig::decodeGapMilliseconds` (G); also the decode-gap
/// classifier's threshold (Splash `Engine::decodeGap`).
pub(crate) const DECODE_GAP_MS: f64 = 500.0;
/// Splash `EngineConfig::decodeShare`.
pub(crate) const DEFAULT_DECODE_SHARE: f64 = 0.5;

/// Splash `model::denseAttentionPairs`.
pub(crate) fn dense_attention_pairs(position: u64, rows: u64) -> u64 {
    rows * position + rows * (rows + 1) / 2
}

/// A family whose prefill rows are cut beside a decoding lane.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MeasuredConfig {
    /// The native context window the attention-pair bound is sized from
    /// (Splash `commandLimits`).
    pub native_window: u32,
    /// Splash `SchedulerConfig::decodeShare`: the part of a prefill unit's
    /// wall it owes the decoding lanes.
    pub decode_share: f64,
}

/// How a step's prefill rows are sized.
enum PrefillUnit {
    /// One token budget shared by the decode rows, their draft rows and the
    /// prefill rows, each prefill row also ending at its pinned break.
    Fixed { budget: u32 },
    /// Splash `Scheduler::planPrefill`: one unit of up to
    /// [`PREFILL_MAX_LANES`] prefill lanes, cut to the rows the last timed
    /// unit ran in G while a lane decodes (`cutRows`).
    Measured {
        pair_bound: u64,
        measured: Option<MeasuredUnit>,
        decode_share: f64,
        /// Splash `decodeDebtMilliseconds_`.
        decode_debt_ms: f64,
    },
}

/// Splash `SchedulerSnapshot`'s counters and `contendedPrefill_`, shared
/// with the status reader and read live.
pub(crate) struct SchedulerCounters {
    pub prefill_batches: AtomicU64,
    pub prefill_rows: AtomicU64,
    pub prefill_units: AtomicU64,
    pub bound_cuts: AtomicU64,
    pub decode_batches: AtomicU64,
    /// Decode batches of width `index + 1`.
    pub decode_batches_by_width: Box<[AtomicU64]>,
    pub queued: AtomicU64,
    pub waiting_resources: AtomicU64,
    pub waiting_prefix: AtomicU64,
    pub prefilling: AtomicU64,
    pub decoding: AtomicU64,
    /// The walls of the prefill units a lane decoded beside.
    pub contended_prefill: FineLatencyHistogram,
}

/// A copy of [`SchedulerCounters`].
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct SchedulerCounterSnapshot {
    pub prefill_batches: u64,
    pub prefill_rows: u64,
    pub prefill_units: u64,
    pub bound_cuts: u64,
    pub decode_batches: u64,
    pub decode_batches_by_width: Vec<u64>,
    pub queued: u64,
    pub waiting_resources: u64,
    pub waiting_prefix: u64,
    pub prefilling: u64,
    pub decoding: u64,
    pub contended_prefill: LatencyStats,
}

impl SchedulerCounters {
    pub(crate) fn new(max_num_seqs: usize) -> Self {
        Self {
            prefill_batches: AtomicU64::new(0),
            prefill_rows: AtomicU64::new(0),
            prefill_units: AtomicU64::new(0),
            bound_cuts: AtomicU64::new(0),
            decode_batches: AtomicU64::new(0),
            decode_batches_by_width: (0..max_num_seqs.max(1))
                .map(|_| AtomicU64::new(0))
                .collect(),
            queued: AtomicU64::new(0),
            waiting_resources: AtomicU64::new(0),
            waiting_prefix: AtomicU64::new(0),
            prefilling: AtomicU64::new(0),
            decoding: AtomicU64::new(0),
            contended_prefill: FineLatencyHistogram::new(),
        }
    }

    pub(crate) fn snapshot(&self) -> SchedulerCounterSnapshot {
        let load = |value: &AtomicU64| value.load(Ordering::Relaxed);
        SchedulerCounterSnapshot {
            prefill_batches: load(&self.prefill_batches),
            prefill_rows: load(&self.prefill_rows),
            prefill_units: load(&self.prefill_units),
            bound_cuts: load(&self.bound_cuts),
            decode_batches: load(&self.decode_batches),
            decode_batches_by_width: self.decode_batches_by_width.iter().map(load).collect(),
            queued: load(&self.queued),
            waiting_resources: load(&self.waiting_resources),
            waiting_prefix: load(&self.waiting_prefix),
            prefilling: load(&self.prefilling),
            decoding: load(&self.decoding),
            contended_prefill: self.contended_prefill.snapshot(),
        }
    }
}

/// Splash `Scheduler::Measured`: the last timed unit.
#[derive(Clone, Copy, Debug, PartialEq)]
struct MeasuredUnit {
    rows: u32,
    pairs: u64,
    millis: f64,
}

/// A prefill lane as planning weighs it (Splash `Scheduler::Lane`).
#[derive(Clone, Copy, Debug)]
struct PrefillLane {
    seq_id: SeqId,
    arrival_order: u64,
    processed: u32,
    remaining: u32,
    overdue: bool,
}

/// Splash `Scheduler::leadsBefore` without priority: overdue lanes first,
/// oldest first; then the fewest remaining rows, then submission order.
fn leads_before(a: &PrefillLane, b: &PrefillLane) -> bool {
    if a.overdue != b.overdue {
        return a.overdue;
    }
    if !a.overdue && a.remaining != b.remaining {
        return a.remaining < b.remaining;
    }
    a.arrival_order < b.arrival_order
}

/// Splash `Scheduler::cutRows`: the rows the last timed unit ran in G at
/// its time a row, fewer where this lane reaches more pairs; a rest below
/// the floor runs with the unit, up to a chunk.
fn cut_rows(leader: &PrefillLane, measured: Option<MeasuredUnit>) -> u32 {
    let floor = PREFILL_FLOOR_ROWS;
    let mut rows = floor;
    if let Some(m) = measured {
        let pairs = dense_attention_pairs(u64::from(leader.processed), u64::from(m.rows)) as f64;
        let deeper = if m.pairs != 0 && pairs > m.pairs as f64 {
            m.pairs as f64 / pairs
        } else {
            1.0
        };
        rows = (f64::from(m.rows) * DECODE_GAP_MS / m.millis * deeper)
            .clamp(f64::from(floor), f64::from(floor.max(PREFILL_CHUNK_ROWS)))
            as u32;
    }
    let remaining = leader.remaining;
    if remaining > rows && remaining - rows < floor {
        rows.max(remaining.min(PREFILL_CHUNK_ROWS))
    } else {
        rows
    }
}

/// The prefill unit of a step: each lane's start and rows, the leader
/// first.
struct PlannedUnit {
    lanes: Vec<(PrefillLane, u32)>,
    leader_cut: bool,
}

/// The clock a waiter's retry is judged by (Splash
/// `Engine::resourceRetryReady`): a lane whose retry is not due is planned
/// without (`PlanInputs::excluded`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct RetryClock {
    pub now: f64,
    pub epoch: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SpeculativeRowResult {
    pub verified_tokens: u32,
    pub generated_tokens: Vec<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RowStepResult {
    pub seq_id: SeqId,
    pub num_computed_tokens: u32,
    pub generated_token: Option<u32>,
    pub speculative: Option<SpeculativeRowResult>,
    pub finished: bool,
    /// The row made no progress because its lazy block growth could not be
    /// satisfied. This is scheduler policy, not a terminal model error.
    pub allocation_blocked: bool,
    /// Wall time attributable to a successful prefill/recompute slice.
    pub prefill_micros: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct StepResult {
    pub rows: Vec<RowStepResult>,
    /// Largest decode forward the executor actually fused in this step.
    /// This is executor-reported rather than inferred from the plan so a
    /// silent N-times-scalar fallback cannot fake occupancy telemetry.
    pub executed_decode_batch: usize,
    /// Rows whose next tokens were selected by one successful batched greedy
    /// argmax. Zero means the executor used scalar sampling or the optional
    /// epilogue fell back.
    pub executed_greedy_epilogue_batch: usize,
    pub rows_alloc_evicted: u32,
    /// Wall of the step's prefill rows, from before the first to after the
    /// last; 0 without one.
    pub prefill_wall_micros: u64,
    /// Wall of the step's decode rows; 0 without one.
    pub decode_wall_micros: u64,
}

pub(crate) trait StepExecutor<P> {
    type Error;

    fn execute(
        &mut self,
        plan: &StepPlan,
        running: &mut [TurnState<P>],
    ) -> Result<StepResult, Self::Error>;
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct SchedulerStats {
    pub global_steps: u64,
    pub max_batch_occupancy: usize,
    pub decode_batch_occupancy_hist: BTreeMap<usize, u64>,
    pub fused_greedy_epilogue_steps: u64,
    pub admitted: u64,
    pub completed: u64,
    pub admission_deferred_blocks: u64,
    pub rows_alloc_evicted: u64,
    pub block_capacity: u32,
    pub free_blocks: u32,
    pub reclaimable_blocks: u32,
    pub allocated_blocks: u32,
    pub watermark_blocks: u32,
    pub reserved_blocks: u32,
    pub memory_capacity_bytes: u64,
    pub memory_watermark_bytes: u64,
    pub reserved_block_bytes: u64,
    pub reserved_state_bytes: u64,
    pub admission_deferred_state: u64,
    pub ssd_restore_waiting: u32,
    pub ssd_restore_bytes: u64,
    pub ssd_restore_wait_micros: u64,
    pub preemptions: u64,
    pub preemptions_recompute: u64,
    pub preemptions_ssd: u64,
    pub prefill_tokens: u64,
    pub prefill_micros: u64,
    /// Splash `replans`: steps that ran some rows and refused others.
    pub replans: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PreemptionMode {
    Recompute,
    Ssd,
}

pub(crate) fn is_paged_allocation_blocked(reason: &str) -> bool {
    reason.contains("paged decode could not reserve")
        || reason.contains("paged cache could not reserve")
}

const SSD_PREEMPT_MIN_BLOCKS: u64 = 64;
const SSD_PREEMPT_FIXED_MICROS: u128 = 2_000;
const SSD_PREEMPT_MARGIN_NUMERATOR: u128 = 5;
const SSD_PREEMPT_MARGIN_DENOMINATOR: u128 = 4;

/// Select SSD escalation only for a long prefix and only when live measurements
/// predict a clear win over recomputation. Missing measurements fail closed to
/// vLLM-v1-style recompute preemption.
pub(crate) fn select_preemption_mode(
    prefix_tokens: u32,
    block_size: u32,
    bytes_per_block: u64,
    prefill_tokens: u64,
    prefill_micros: u64,
    restore_bytes: u64,
    restore_micros: u64,
) -> PreemptionMode {
    if block_size == 0
        || bytes_per_block == 0
        || prefill_tokens == 0
        || prefill_micros == 0
        || restore_bytes == 0
        || restore_micros == 0
    {
        return PreemptionMode::Recompute;
    }
    let full_blocks = u64::from(prefix_tokens / block_size);
    if full_blocks < SSD_PREEMPT_MIN_BLOCKS {
        return PreemptionMode::Recompute;
    }
    let restore_payload = u128::from(full_blocks).saturating_mul(u128::from(bytes_per_block));
    let predicted_restore = restore_payload
        .saturating_mul(u128::from(restore_micros))
        .div_ceil(u128::from(restore_bytes))
        .saturating_add(SSD_PREEMPT_FIXED_MICROS);
    let predicted_recompute = u128::from(prefix_tokens)
        .saturating_mul(u128::from(prefill_micros))
        .div_ceil(u128::from(prefill_tokens));
    if predicted_restore.saturating_mul(SSD_PREEMPT_MARGIN_NUMERATOR)
        < predicted_recompute.saturating_mul(SSD_PREEMPT_MARGIN_DENOMINATOR)
    {
        PreemptionMode::Ssd
    } else {
        PreemptionMode::Recompute
    }
}

#[napi(object, js_name = "DecodeBatchOccupancyBucket")]
pub struct DecodeBatchOccupancyBucketJs {
    pub occupancy: u32,
    pub steps: f64,
}

/// Read-only NAPI/dashboard mirror. Counters use `f64`, matching the other
/// native metrics snapshots and avoiding BigInt round-trips in JavaScript.
#[napi(object, js_name = "SchedulerStats")]
pub struct SchedulerStatsJs {
    pub global_steps: f64,
    pub max_batch_occupancy: u32,
    pub decode_batch_occupancy_hist: Vec<DecodeBatchOccupancyBucketJs>,
    pub fused_greedy_epilogue_steps: f64,
    pub admitted: f64,
    pub completed: f64,
    pub admission_deferred_blocks: f64,
    pub rows_alloc_evicted: f64,
    pub block_capacity: u32,
    pub free_blocks: u32,
    pub reclaimable_blocks: u32,
    pub allocated_blocks: u32,
    pub watermark_blocks: u32,
    pub reserved_blocks: u32,
    pub memory_capacity_bytes: f64,
    pub memory_watermark_bytes: f64,
    pub reserved_block_bytes: f64,
    pub reserved_state_bytes: f64,
    pub admission_deferred_state: f64,
    pub ssd_restore_waiting: u32,
    pub ssd_restore_bytes: f64,
    pub ssd_restore_wait_ms: f64,
    pub preemptions: f64,
    pub preemptions_recompute: f64,
    pub preemptions_ssd: f64,
}

impl SchedulerStats {
    pub fn to_js(&self) -> SchedulerStatsJs {
        SchedulerStatsJs {
            global_steps: self.global_steps as f64,
            max_batch_occupancy: self.max_batch_occupancy as u32,
            decode_batch_occupancy_hist: self
                .decode_batch_occupancy_hist
                .iter()
                .map(|(&occupancy, &steps)| DecodeBatchOccupancyBucketJs {
                    occupancy: occupancy as u32,
                    steps: steps as f64,
                })
                .collect(),
            fused_greedy_epilogue_steps: self.fused_greedy_epilogue_steps as f64,
            admitted: self.admitted as f64,
            completed: self.completed as f64,
            admission_deferred_blocks: self.admission_deferred_blocks as f64,
            rows_alloc_evicted: self.rows_alloc_evicted as f64,
            block_capacity: self.block_capacity,
            free_blocks: self.free_blocks,
            reclaimable_blocks: self.reclaimable_blocks,
            allocated_blocks: self.allocated_blocks,
            watermark_blocks: self.watermark_blocks,
            reserved_blocks: self.reserved_blocks,
            memory_capacity_bytes: self.memory_capacity_bytes as f64,
            memory_watermark_bytes: self.memory_watermark_bytes as f64,
            reserved_block_bytes: self.reserved_block_bytes as f64,
            reserved_state_bytes: self.reserved_state_bytes as f64,
            admission_deferred_state: self.admission_deferred_state as f64,
            ssd_restore_waiting: self.ssd_restore_waiting,
            ssd_restore_bytes: self.ssd_restore_bytes as f64,
            ssd_restore_wait_ms: self.ssd_restore_wait_micros as f64 / 1_000.0,
            preemptions: self.preemptions as f64,
            preemptions_recompute: self.preemptions_recompute as f64,
            preemptions_ssd: self.preemptions_ssd as f64,
        }
    }
}

#[derive(Debug)]
pub(crate) enum SchedulerError<E> {
    Executor(E),
    InvalidResult(String),
}

/// How a step's blocked rows are settled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum JudgePolicy {
    /// The newest running turn is preempted; a lone blocked turn is
    /// reported in `blocked_alone`.
    Legacy,
    /// Rows claimed their memory before the step (Splash
    /// `Engine::prepare`): blocked rows and their denials are reported and
    /// no turn moves; the caller judges them.
    Ledger,
}

/// A step's claims per plan row: `None` granted, else the row's denial.
pub(crate) type RowGrant = Option<Denial>;

/// What [`Scheduler::prepare_step`] found to do.
pub(crate) enum Prepared<P, Exclusive, Barrier> {
    Action(SchedulerAction<P, Exclusive, Barrier>),
    Plan(StepPlan),
}

pub(crate) enum SchedulerAction<P, Exclusive, Barrier> {
    Idle,
    Exclusive(Exclusive),
    Barrier(Barrier),
    Stepped {
        #[cfg(test)]
        plan: StepPlan,
        completed: Vec<TurnState<P>>,
        preempted: Option<Box<PreemptedTurn<P>>>,
        /// The step's only turn blocked: no other turn holds memory that
        /// could come back, so it stays running, unpreempted, for the model
        /// to reclaim for or fail (Splash `Engine::judge`).
        blocked_alone: Option<SeqId>,
        /// [`JudgePolicy::Ledger`]: each blocked row and its denial.
        blocked: Vec<(SeqId, Denial)>,
        /// Some row of the step ran.
        progressed: bool,
        planned: Vec<SeqId>,
        /// The step's prefill-phase wall (`StepResult`).
        prefill_wall_micros: u64,
    },
}

/// A step's preemption victim, and whether its own row was the one that
/// blocked. A victim that did not block resumes to the KV it held, not to
/// the blocked row's step (Splash `Engine.cpp` resume target).
pub(crate) struct PreemptedTurn<P> {
    pub turn: TurnState<P>,
    pub blocked: bool,
}

struct Ordered<T> {
    order: u64,
    value: T,
}

/// FCFS scheduler with a single token budget shared by prefix-hit prefills,
/// cold prefills, and decode rows, or with a measured prefill unit beside
/// the decode rows (Splash `Scheduler::planPrefill`).
pub(crate) struct Scheduler<P, Exclusive, Barrier> {
    waiting: VecDeque<TurnState<P>>,
    exclusive_lane: VecDeque<Ordered<Exclusive>>,
    barriers: VecDeque<Ordered<Barrier>>,
    running: Vec<TurnState<P>>,
    max_num_seqs: usize,
    prefill: PrefillUnit,
    /// The lanes the last plan was made without (Splash `excluded_`).
    plan_excluded: Vec<SeqId>,
    counters: Arc<SchedulerCounters>,
    next_order: u64,
    /// Splash `Scheduler::complete`'s `youngestServed` of the last prefill
    /// unit, for the starts held outside the scheduler.
    youngest_served: Option<u64>,
    global_step: u64,
    stats: SchedulerStats,
}

impl<P, Exclusive, Barrier> Scheduler<P, Exclusive, Barrier> {
    pub fn new(max_num_seqs: usize, budget: u32) -> Result<Self, String> {
        if max_num_seqs == 0 {
            return Err("max_num_seqs must be positive".to_string());
        }
        if budget == 0 {
            return Err("the step token budget must be positive".to_string());
        }
        Ok(Self {
            waiting: VecDeque::new(),
            exclusive_lane: VecDeque::new(),
            barriers: VecDeque::new(),
            running: Vec::new(),
            max_num_seqs,
            prefill: PrefillUnit::Fixed { budget },
            plan_excluded: Vec::new(),
            counters: Arc::new(SchedulerCounters::new(max_num_seqs)),
            next_order: 0,
            youngest_served: None,
            global_step: 0,
            stats: SchedulerStats::default(),
        })
    }

    /// Plans prefill as Splash `Scheduler::planPrefill`; the attention-pair
    /// bound is Splash `commandLimits`' `attentionPairs`.
    pub fn with_measured_prefill_cut(mut self, config: MeasuredConfig) -> Self {
        self.prefill = PrefillUnit::Measured {
            pair_bound: dense_attention_pairs(
                u64::from(config.native_window.saturating_sub(PREFILL_CHUNK_ROWS)),
                u64::from(PREFILL_CHUNK_ROWS),
            ),
            measured: None,
            decode_share: config.decode_share,
            decode_debt_ms: 0.0,
        };
        self
    }

    /// The counters the status reads live.
    pub fn counters(&self) -> Arc<SchedulerCounters> {
        Arc::clone(&self.counters)
    }

    #[cfg(test)]
    pub fn measures_prefill_cut(&self) -> bool {
        matches!(self.prefill, PrefillUnit::Measured { .. })
    }

    /// The rows of the last timed unit.
    #[cfg(test)]
    pub fn measured_rows(&self) -> Option<u32> {
        match self.prefill {
            PrefillUnit::Measured { measured, .. } => measured.map(|unit| unit.rows),
            PrefillUnit::Fixed { .. } => None,
        }
    }

    fn take_order(&mut self) -> u64 {
        let order = self.next_order;
        self.next_order = self.next_order.saturating_add(1);
        order
    }

    /// The submission order of a start held before it has a turn.
    pub fn reserve_order(&mut self) -> u64 {
        self.take_order()
    }

    /// The youngest lane the last measured prefill unit served, once.
    pub fn take_youngest_served(&mut self) -> Option<u64> {
        self.youngest_served.take()
    }

    #[cfg(test)]
    pub fn enqueue_turn(&mut self, turn: TurnState<P>) -> Result<(), String> {
        self.try_enqueue_turn(turn)
            .map_err(|error_and_turn| error_and_turn.0)
    }

    pub fn try_enqueue_turn(
        &mut self,
        mut turn: TurnState<P>,
    ) -> Result<(), Box<(String, TurnState<P>)>> {
        if self
            .waiting
            .iter()
            .chain(self.running.iter())
            .any(|existing| existing.seq_id == turn.seq_id)
        {
            return Err(Box::new((
                format!("duplicate scheduler sequence {}", turn.seq_id),
                turn,
            )));
        }
        turn.status = TurnStatus::Waiting;
        turn.arrival_order = match turn.submitted_order.take() {
            Some(order) => order,
            None => self.take_order(),
        };
        self.waiting.push_back(turn);
        Ok(())
    }

    pub fn enqueue_exclusive(&mut self, value: Exclusive) {
        let order = self.take_order();
        self.exclusive_lane.push_back(Ordered { order, value });
    }

    pub fn enqueue_barrier(&mut self, value: Barrier) {
        let order = self.take_order();
        self.barriers.push_back(Ordered { order, value });
    }

    pub fn has_work(&self) -> bool {
        !self.waiting.is_empty()
            || !self.running.is_empty()
            || !self.exclusive_lane.is_empty()
            || !self.barriers.is_empty()
    }

    /// Whether an ordered control command is waiting for the rows ahead of it
    /// to drain. Model drivers must not perform admission side effects while
    /// this is true: a barrier may invalidate every cache object an admission
    /// would otherwise attach to.
    pub fn has_pending_control(&self) -> bool {
        !self.exclusive_lane.is_empty() || !self.barriers.is_empty()
    }

    pub fn waiting_len(&self) -> usize {
        self.waiting.len()
    }

    pub fn running_len(&self) -> usize {
        self.running.len()
    }

    pub fn max_num_seqs(&self) -> usize {
        self.max_num_seqs
    }

    pub fn contains_seq(&self, seq_id: SeqId) -> bool {
        self.waiting
            .iter()
            .chain(self.running.iter())
            .any(|turn| turn.seq_id == seq_id)
    }

    pub fn has_live_turns(&self) -> bool {
        !self.waiting.is_empty() || !self.running.is_empty()
    }

    /// Splash `Engine::lanes_` is not empty: a running turn, or a restore
    /// a turn waits for.
    pub fn has_resident(&self) -> bool {
        !self.running.is_empty() || self.waiting.iter().any(|turn| turn.ssd_lane)
    }

    pub fn has_preempted(&self) -> bool {
        self.waiting
            .iter()
            .any(|turn| turn.status == TurnStatus::Preempted)
    }

    /// Whether a resuming victim may wait for memory: a running row, or a
    /// waiter that arrived before it and so may run ahead of it, can still
    /// free memory. Later waiters and other victims cannot run first, so
    /// waiting on them never ends.
    pub fn resume_may_wait(&self, resuming: &TurnState<P>) -> bool {
        !self.running.is_empty()
            || self.waiting.iter().any(|turn| {
                turn.status != TurnStatus::Preempted
                    && (turn.ssd_lane || turn.arrival_order < resuming.arrival_order)
            })
    }

    fn unallocated_reserved_blocks(&self) -> u32 {
        self.waiting
            .iter()
            .chain(self.running.iter())
            .fold(0u32, |sum, turn| {
                sum.saturating_add(turn.unallocated_reserved_blocks())
            })
    }

    fn reserved_recurrent_state_bytes(&self) -> u64 {
        self.waiting
            .iter()
            .chain(self.running.iter())
            .fold(0u64, |sum, turn| {
                sum.saturating_add(turn.reserved_recurrent_state_bytes())
            })
    }

    /// Off-pool state the waiting and running turns reserved beyond what
    /// their sequences hold now (`held`).
    pub fn outstanding_scheduled_state_bytes(&self, held: impl Fn(SeqId) -> u64) -> u64 {
        self.waiting
            .iter()
            .chain(self.running.iter())
            .fold(0u64, |sum, turn| {
                sum.saturating_add(turn.outstanding_scheduled_state_bytes(held(turn.seq_id)))
            })
    }

    pub fn try_reserve_memory(
        &mut self,
        telemetry: MemoryTelemetry,
        candidate_blocks: u32,
        bytes_per_block: u64,
        resident_state_bytes: u64,
        candidate_state_bytes: u64,
        watermark_fraction: f64,
    ) -> MemoryAdmission {
        let decision = memory_admission_decision(
            telemetry,
            self.unallocated_reserved_blocks(),
            candidate_blocks,
            bytes_per_block,
            self.reserved_recurrent_state_bytes()
                .max(resident_state_bytes),
            candidate_state_bytes,
            self.has_live_turns(),
            watermark_fraction,
        );
        self.stats.memory_capacity_bytes = telemetry.capacity_bytes;
        self.stats.memory_watermark_bytes = decision.watermark_bytes;
        self.stats.reserved_block_bytes = decision.reserved_block_bytes;
        self.stats.reserved_state_bytes = decision.reserved_state_bytes;
        if bytes_per_block > 0 {
            let blocks =
                |bytes: u64| bytes.div_euclid(bytes_per_block).min(u64::from(u32::MAX)) as u32;
            self.stats.block_capacity = blocks(telemetry.capacity_bytes);
            self.stats.free_blocks = blocks(telemetry.free_bytes);
            self.stats.reclaimable_blocks = blocks(telemetry.reclaimable_bytes);
            self.stats.allocated_blocks = self
                .stats
                .block_capacity
                .saturating_sub(self.stats.free_blocks);
            self.stats.watermark_blocks = decision
                .watermark_bytes
                .div_ceil(bytes_per_block)
                .min(u64::from(u32::MAX)) as u32;
            self.stats.reserved_blocks = blocks(decision.reserved_block_bytes);
        }
        if !decision.admitted {
            self.stats.admission_deferred_state =
                self.stats.admission_deferred_state.saturating_add(1);
        }
        decision
    }

    pub fn observe_hybrid_memory(
        &mut self,
        blocks: BlockTelemetry,
        bytes_per_block: u64,
        resident_state_bytes: u64,
        watermark_fraction: f64,
    ) -> MemoryAdmission {
        let telemetry = MemoryTelemetry {
            capacity_bytes: u64::from(blocks.total_blocks).saturating_mul(bytes_per_block),
            free_bytes: u64::from(blocks.free_blocks).saturating_mul(bytes_per_block),
            reclaimable_bytes: u64::from(blocks.reclaimable_blocks).saturating_mul(bytes_per_block),
        };
        let decision = memory_admission_decision(
            telemetry,
            self.unallocated_reserved_blocks(),
            0,
            bytes_per_block,
            self.reserved_recurrent_state_bytes()
                .max(resident_state_bytes),
            0,
            self.has_live_turns(),
            watermark_fraction,
        );
        self.stats.block_capacity = blocks.total_blocks;
        self.stats.free_blocks = blocks.free_blocks;
        self.stats.reclaimable_blocks = blocks.reclaimable_blocks;
        self.stats.allocated_blocks = blocks.allocated_blocks;
        self.stats.watermark_blocks = if bytes_per_block == 0 {
            0
        } else {
            decision
                .watermark_bytes
                .div_ceil(bytes_per_block)
                .min(u64::from(u32::MAX)) as u32
        };
        self.stats.reserved_blocks = self.unallocated_reserved_blocks();
        self.stats.memory_capacity_bytes = telemetry.capacity_bytes;
        self.stats.memory_watermark_bytes = decision.watermark_bytes;
        self.stats.reserved_block_bytes = decision.reserved_block_bytes;
        self.stats.reserved_state_bytes = decision.reserved_state_bytes;
        decision
    }

    pub fn park_waiting_for_ssd(&mut self, seq_id: SeqId) -> Result<(), String> {
        let turn = self
            .waiting
            .iter_mut()
            .find(|turn| turn.seq_id == seq_id)
            .ok_or_else(|| format!("cannot park unknown waiting sequence {seq_id}"))?;
        if turn.status != TurnStatus::Waiting {
            return Err(format!("sequence {seq_id} is not admission-waiting"));
        }
        turn.status = TurnStatus::WaitingForSsd;
        turn.ssd_lane = true;
        self.stats.ssd_restore_waiting = self.stats.ssd_restore_waiting.saturating_add(1);
        Ok(())
    }

    pub fn wake_from_ssd(
        &mut self,
        seq_id: SeqId,
        bytes_restored: u64,
        wait: Duration,
    ) -> Result<(), String> {
        let turn = self
            .waiting
            .iter_mut()
            .find(|turn| turn.seq_id == seq_id)
            .ok_or_else(|| format!("cannot wake unknown waiting sequence {seq_id}"))?;
        if turn.status != TurnStatus::WaitingForSsd {
            return Err(format!("sequence {seq_id} is not waiting for SSD"));
        }
        turn.status = TurnStatus::Waiting;
        self.stats.ssd_restore_waiting = self.stats.ssd_restore_waiting.saturating_sub(1);
        self.stats.ssd_restore_bytes = self.stats.ssd_restore_bytes.saturating_add(bytes_restored);
        self.stats.ssd_restore_wait_micros = self
            .stats
            .ssd_restore_wait_micros
            .saturating_add(wait.as_micros().try_into().unwrap_or(u64::MAX));
        Ok(())
    }

    pub fn waiting_turn_mut(&mut self, seq_id: SeqId) -> Option<&mut TurnState<P>> {
        self.waiting.iter_mut().find(|turn| turn.seq_id == seq_id)
    }

    /// Remove every cancelled admission waiter before planning. This avoids
    /// spending a prefill slice on a dead ordinary waiter, and lets a model
    /// drop an SSD waiter's restore ticket immediately instead of waiting for
    /// storage I/O to finish before observing the same flag.
    pub fn take_cancelled_waiters(&mut self) -> Vec<TurnState<P>> {
        let mut cancelled = Vec::new();
        let mut retained = VecDeque::with_capacity(self.waiting.len());
        while let Some(turn) = self.waiting.pop_front() {
            let is_cancelled_waiter =
                matches!(turn.status, TurnStatus::Waiting | TurnStatus::WaitingForSsd)
                    && turn
                        .cancelled
                        .as_ref()
                        .is_some_and(|flag| flag.load(Ordering::Relaxed));
            if is_cancelled_waiter {
                if turn.status == TurnStatus::WaitingForSsd {
                    self.stats.ssd_restore_waiting =
                        self.stats.ssd_restore_waiting.saturating_sub(1);
                }
                cancelled.push(turn);
            } else {
                retained.push_back(turn);
            }
        }
        self.waiting = retained;
        cancelled
    }

    fn preempt_running_at(&mut self, index: usize) -> TurnState<P> {
        let mut victim = self.running.remove(index);
        victim.status = TurnStatus::Preempted;
        victim.ssd_lane = false;
        victim.block_reservation_active = false;
        victim.recurrent_state_reservation_active = false;
        victim.preemptions = victim.preemptions.saturating_add(1);
        self.stats.preemptions = self.stats.preemptions.saturating_add(1);
        self.drop_stale_decode_debt();
        victim
    }

    /// Undo [`Self::preempt_running_at`] for a victim whose cache was never
    /// released: memory was found without preempting it.
    pub fn reinstate_running(&mut self, mut turn: TurnState<P>) {
        turn.status = TurnStatus::Running;
        turn.block_reservation_active = true;
        turn.recurrent_state_reservation_active = true;
        turn.preemptions = turn.preemptions.saturating_sub(1);
        self.stats.preemptions = self.stats.preemptions.saturating_sub(1);
        self.running.push(turn);
    }

    pub fn preempt_running(&mut self, seq_id: SeqId) -> Option<TurnState<P>> {
        let index = self.running.iter().position(|turn| turn.seq_id == seq_id)?;
        Some(self.preempt_running_at(index))
    }

    /// Every turn the scheduler holds, waiting and running.
    pub fn turns_mut(&mut self) -> impl Iterator<Item = &mut TurnState<P>> {
        self.waiting.iter_mut().chain(self.running.iter_mut())
    }

    /// Every turn the scheduler holds, waiting and running.
    pub fn turns(&self) -> impl Iterator<Item = &TurnState<P>> {
        self.waiting.iter().chain(self.running.iter())
    }

    /// The suspended turn admission tries next (Splash `admitQueued`
    /// resumes one per pass).
    pub fn first_preempted(&self) -> Option<SeqId> {
        self.waiting
            .iter()
            .find(|turn| turn.status == TurnStatus::Preempted)
            .map(|turn| turn.seq_id)
    }

    pub fn running_turn_mut(&mut self, seq_id: SeqId) -> Option<&mut TurnState<P>> {
        self.running.iter_mut().find(|turn| turn.seq_id == seq_id)
    }

    /// Splash `Engine::anotherResident`: a lane other than `seq_id`'s holds
    /// memory that comes back when it ends, a running turn's or a restore's
    /// it waits for.
    pub fn another_resident(&self, seq_id: SeqId) -> bool {
        self.running.iter().any(|turn| turn.seq_id != seq_id)
            || self.waiting.iter().any(|turn| turn.ssd_lane)
    }

    /// Remove a running turn that ends without completing.
    pub fn take_running(&mut self, seq_id: SeqId) -> Option<TurnState<P>> {
        let index = self.running.iter().position(|turn| turn.seq_id == seq_id)?;
        let turn = self.running.remove(index);
        self.drop_stale_decode_debt();
        Some(turn)
    }

    #[cfg(test)]
    pub fn turn_mut(&mut self, seq_id: SeqId) -> Option<&mut TurnState<P>> {
        self.waiting
            .iter_mut()
            .chain(self.running.iter_mut())
            .find(|turn| turn.seq_id == seq_id)
    }

    /// Remove the oldest preempted row for model-specific cache re-preparation.
    /// Preempted rows are deliberately invisible to `admit_waiting`; the model
    /// must first establish a fresh adapter request (or an SSD restore ticket).
    pub fn take_preempted(&mut self) -> Option<TurnState<P>> {
        let index = self
            .waiting
            .iter()
            .position(|turn| turn.status == TurnStatus::Preempted)?;
        self.waiting.remove(index)
    }

    /// Put a victim at the head of the FCFS queue without assigning a new
    /// arrival order. It cannot be admitted until `ready_preempted` succeeds.
    pub fn prepend_preempted(&mut self, mut turn: TurnState<P>) {
        turn.status = TurnStatus::Preempted;
        turn.ssd_lane = false;
        turn.block_reservation_active = false;
        turn.recurrent_state_reservation_active = false;
        self.waiting.push_front(turn);
    }

    /// Publish successful model-specific resume preparation.
    pub fn ready_preempted(&mut self, mut turn: TurnState<P>, waiting_for_ssd: bool) {
        turn.block_reservation_active = true;
        turn.recurrent_state_reservation_active = true;
        turn.ssd_lane = waiting_for_ssd;
        if waiting_for_ssd {
            turn.status = TurnStatus::WaitingForSsd;
            self.stats.ssd_restore_waiting = self.stats.ssd_restore_waiting.saturating_add(1);
        } else {
            turn.status = TurnStatus::Waiting;
        }
        self.waiting.push_front(turn);
    }

    pub fn preemption_mode(
        &self,
        prefix_tokens: u32,
        block_size: u32,
        bytes_per_block: u64,
    ) -> PreemptionMode {
        select_preemption_mode(
            prefix_tokens,
            block_size,
            bytes_per_block,
            self.stats.prefill_tokens,
            self.stats.prefill_micros,
            self.stats.ssd_restore_bytes,
            self.stats.ssd_restore_wait_micros,
        )
    }

    pub fn record_preemption_mode(&mut self, mode: PreemptionMode) {
        match mode {
            PreemptionMode::Recompute => {
                self.stats.preemptions_recompute =
                    self.stats.preemptions_recompute.saturating_add(1);
            }
            PreemptionMode::Ssd => {
                self.stats.preemptions_ssd = self.stats.preemptions_ssd.saturating_add(1);
            }
        }
    }

    pub fn take_waiting(&mut self, seq_id: SeqId) -> Option<TurnState<P>> {
        let index = self.waiting.iter().position(|turn| turn.seq_id == seq_id)?;
        let turn = self.waiting.remove(index)?;
        if turn.status == TurnStatus::WaitingForSsd {
            self.stats.ssd_restore_waiting = self.stats.ssd_restore_waiting.saturating_sub(1);
        }
        Some(turn)
    }

    pub fn running(&self) -> &[TurnState<P>] {
        &self.running
    }

    pub fn stats(&self) -> &SchedulerStats {
        &self.stats
    }

    fn earliest_control_order(&self) -> Option<u64> {
        self.exclusive_lane
            .front()
            .map(|item| item.order)
            .into_iter()
            .chain(self.barriers.front().map(|item| item.order))
            .min()
    }

    fn pop_control_if_due(&mut self) -> Option<SchedulerAction<P, Exclusive, Barrier>> {
        if !self.running.is_empty() {
            return None;
        }
        let waiting_order = self.waiting.iter().map(|turn| turn.arrival_order).min();
        let exclusive_order = self.exclusive_lane.front().map(|item| item.order);
        let barrier_order = self.barriers.front().map(|item| item.order);
        let control_order = exclusive_order.into_iter().chain(barrier_order).min()?;
        if waiting_order.is_some_and(|order| order < control_order) {
            return None;
        }
        if exclusive_order == Some(control_order) {
            return self
                .exclusive_lane
                .pop_front()
                .map(|item| SchedulerAction::Exclusive(item.value));
        }
        debug_assert!(
            self.running.is_empty(),
            "barriers require an empty running set"
        );
        self.barriers
            .pop_front()
            .map(|item| SchedulerAction::Barrier(item.value))
    }

    fn admit_waiting(&mut self) -> Result<(), String> {
        let control_order = self.earliest_control_order();
        // A preempted row keeps its original FCFS position while the model
        // rebuilds or restores its cache. Later arrivals may not leapfrog it:
        // doing so can keep the victim permanently unable to reacquire its
        // reservation and, transitively, hold an ordered barrier forever.
        let preempted_order = self
            .waiting
            .iter()
            .filter(|turn| turn.status == TurnStatus::Preempted)
            .map(|turn| turn.arrival_order)
            .min();
        while self.running.len() < self.max_num_seqs {
            let candidate = self.waiting.iter().position(|turn| {
                turn.status == TurnStatus::Waiting
                    && (turn.ssd_lane
                        || !preempted_order.is_some_and(|order| turn.arrival_order > order))
                    && !control_order.is_some_and(|order| turn.arrival_order > order)
            });
            let Some(candidate) = candidate else {
                break;
            };
            let mut turn = self.waiting.remove(candidate).ok_or_else(|| {
                format!(
                    "scheduler admission selected missing waiting index {candidate} (len={})",
                    self.waiting.len()
                )
            })?;
            turn.status = TurnStatus::Running;
            self.running.push(turn);
            self.stats.admitted = self.stats.admitted.saturating_add(1);
        }
        Ok(())
    }

    fn build_plan(&mut self, clock: Option<RetryClock>) -> StepPlan {
        for turn in &mut self.running {
            turn.snapshot_cancel();
        }
        let mut budget = match self.prefill {
            PrefillUnit::Fixed { budget } => budget,
            PrefillUnit::Measured { .. } => return self.build_measured_plan(clock),
        };
        self.plan_excluded.clear();
        let mut rows: Vec<StepRow> = Vec::with_capacity(self.running.len());
        // Decode-first is a priority within ONE phase-free plan, not a decode
        // queue. Every row's kind is still derived from token progress. This
        // matches vLLM v1's running-first budget shape and prevents a long
        // newly admitted prefill from consuming the step budget ahead of an
        // already interactive decode row.
        for planned_kind in [StepKind::Decode, StepKind::Prefill] {
            if planned_kind == StepKind::Prefill {
                for row in &mut rows {
                    let Some(turn) = self.running.iter().find(|turn| turn.seq_id == row.seq_id)
                    else {
                        continue;
                    };
                    let extra = turn.decode_draft_allowance.min(budget);
                    row.num_tokens += extra;
                    budget -= extra;
                }
            }
            for turn in &self.running {
                if budget == 0 || turn.status != TurnStatus::Running {
                    continue;
                }
                let pending = turn.num_tokens.saturating_sub(turn.num_computed_tokens);
                if pending == 0 {
                    continue;
                }
                let kind = if turn.num_computed_tokens < turn.prompt_tokens {
                    StepKind::Prefill
                } else {
                    StepKind::Decode
                };
                if kind != planned_kind {
                    continue;
                }
                let row_limit = match kind {
                    StepKind::Prefill => turn
                        .next_prefill_boundary()
                        .saturating_sub(turn.num_computed_tokens),
                    StepKind::Decode => 1,
                };
                // A whole-prompt prefill row ignores the token budget and is
                // the step's only prefill row; it waits while another turn's
                // prefill row is planned ahead of it.
                let whole = kind == StepKind::Prefill && turn.whole_prompt_prefill;
                if whole && rows.iter().any(|row| row.kind == StepKind::Prefill) {
                    continue;
                }
                let num_tokens = if whole {
                    pending.min(row_limit)
                } else {
                    pending.min(row_limit).min(budget)
                };
                if num_tokens == 0 {
                    continue;
                }
                rows.push(StepRow {
                    seq_id: turn.seq_id,
                    kind,
                    token_start: turn.num_computed_tokens,
                    num_tokens,
                    cancel_snapshot: turn.cancel_snapshot,
                });
                budget = if whole { 0 } else { budget - num_tokens };
            }
        }
        StepPlan {
            global_step: self.global_step,
            rows,
            contended: self.contended(),
            leader_cut: None,
        }
    }

    /// Splash `Scheduler::contended`: a lane decodes, planned or not. A
    /// cancelled lane is not decoding.
    fn contended(&self) -> bool {
        self.running.iter().any(|turn| {
            turn.status == TurnStatus::Running
                && turn.num_computed_tokens >= turn.prompt_tokens
                && !turn.cancel_snapshot
        })
    }

    /// Splash `Scheduler::next` + `planPrefill`: the decode rows, outside
    /// the unit; the prefill unit; then the cancelled prefill turns, one
    /// row each, which the executor fails.
    fn build_measured_plan(&mut self, clock: Option<RetryClock>) -> StepPlan {
        let excluded: Vec<SeqId> = match clock {
            Some(clock) => self
                .running
                .iter()
                .chain(self.waiting.iter())
                .filter(|turn| !retry_ready(&turn.resource_wait, clock.now, clock.epoch))
                .map(|turn| turn.seq_id)
                .collect(),
            _ => Vec::new(),
        };
        let mut rows: Vec<StepRow> = Vec::with_capacity(self.running.len());
        for turn in &self.running {
            if turn.status != TurnStatus::Running
                || turn.num_computed_tokens < turn.prompt_tokens
                || turn.num_tokens <= turn.num_computed_tokens
                || (!turn.cancel_snapshot && excluded.contains(&turn.seq_id))
            {
                continue;
            }
            rows.push(StepRow {
                seq_id: turn.seq_id,
                kind: StepKind::Decode,
                token_start: turn.num_computed_tokens,
                num_tokens: 1 + turn.decode_draft_allowance,
                cancel_snapshot: turn.cancel_snapshot,
            });
        }
        let contended = self.contended();
        let mut leader_cut = None;
        // Splash `Scheduler::next`: decode runs while prefill owes it time.
        let owed = matches!(self.prefill, PrefillUnit::Measured { decode_debt_ms, .. } if decode_debt_ms > 0.0)
            && rows.iter().any(|row| !row.cancel_snapshot);
        if !owed && let Some(unit) = self.plan_unit(self.ready_lanes(&excluded), contended) {
            if unit.leader_cut {
                leader_cut = unit.lanes.first().map(|(lane, _)| lane.seq_id);
            }
            rows.extend(unit.lanes.iter().map(|&(lane, num_tokens)| StepRow {
                seq_id: lane.seq_id,
                kind: StepKind::Prefill,
                token_start: lane.processed,
                num_tokens,
                cancel_snapshot: false,
            }));
        }
        for turn in &self.running {
            if turn.status == TurnStatus::Running
                && turn.num_computed_tokens < turn.prompt_tokens
                && turn.cancel_snapshot
            {
                rows.push(StepRow {
                    seq_id: turn.seq_id,
                    kind: StepKind::Prefill,
                    token_start: turn.num_computed_tokens,
                    num_tokens: 1,
                    cancel_snapshot: true,
                });
            }
        }
        self.plan_excluded = excluded;
        StepPlan {
            global_step: self.global_step,
            rows,
            contended,
            leader_cut,
        }
    }

    /// Splash `Scheduler::readyLanes`: the running prefill lanes, neither
    /// cancelled nor excluded.
    fn ready_lanes(&self, excluded: &[SeqId]) -> Vec<PrefillLane> {
        self.running
            .iter()
            .filter(|turn| {
                turn.status == TurnStatus::Running
                    && turn.num_computed_tokens < turn.prompt_tokens
                    && turn.num_tokens > turn.num_computed_tokens
                    && !turn.cancel_snapshot
                    && !excluded.contains(&turn.seq_id)
            })
            .map(|turn| {
                debug_assert!(!turn.whole_prompt_prefill);
                PrefillLane {
                    seq_id: turn.seq_id,
                    arrival_order: turn.arrival_order,
                    processed: turn.num_computed_tokens,
                    remaining: turn.prompt_tokens - turn.num_computed_tokens,
                    overdue: turn.overtaken >= MAX_OVERTAKES,
                }
            })
            .collect()
    }

    /// Splash `Scheduler::planPrefill`: the leader halves under the pair
    /// bound and takes the cut while a lane decodes; joiners take the rows
    /// that spare them one of their own units, when they fit.
    fn plan_unit(&self, mut ready: Vec<PrefillLane>, contended: bool) -> Option<PlannedUnit> {
        let PrefillUnit::Measured {
            pair_bound,
            measured,
            ..
        } = self.prefill
        else {
            return None;
        };
        ready.sort_by(|a, b| {
            if leads_before(a, b) {
                std::cmp::Ordering::Less
            } else if leads_before(b, a) {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        });
        let (&leader, joiners) = ready.split_first()?;
        let pairs =
            |position: u32, rows: u32| dense_attention_pairs(u64::from(position), u64::from(rows));
        let mut own = leader.remaining.min(PREFILL_CHUNK_ROWS);
        while own > PREFILL_MIN_ROWS && pairs(leader.processed, own) > pair_bound {
            own /= 2;
        }
        let unit = if contended {
            cut_rows(&leader, measured)
        } else {
            PREFILL_CHUNK_ROWS
        };
        let leader_cut = unit < own;
        let leader_rows = own.min(unit);
        let mut lanes = vec![(leader, leader_rows)];
        let mut unit_rows = leader_rows;
        let mut unit_pairs = pairs(leader.processed, leader_rows);
        for &lane in joiners {
            if lanes.len() == PREFILL_MAX_LANES {
                break;
            }
            let rows = (lane.remaining - 1) % PREFILL_CHUNK_ROWS + 1;
            let with = unit_pairs + pairs(lane.processed, rows);
            if unit_rows + rows > unit || with > pair_bound {
                continue;
            }
            lanes.push((lane, rows));
            unit_rows += rows;
            unit_pairs = with;
        }
        Some(PlannedUnit { lanes, leader_cut })
    }

    /// Splash `Scheduler::prefillAdmissionOrder` for one candidate of
    /// `prompt_tokens`, `cached` of them reused: whether the first unit it
    /// would join serves it, i.e. holds it and either ends its prompt or
    /// leaves it waiting for no other lane's prompt. A start holds its memory from its
    /// first unit to its last, so one that would join and then wait out
    /// another's prompt stays queued until it leads.
    /// A held start ranks by its `submitted` order and `overtaken` count.
    pub fn serves_candidate(
        &self,
        seq_id: SeqId,
        cached: u32,
        prompt_tokens: u32,
        submitted: Option<u64>,
        overtaken: u32,
    ) -> bool {
        let processed = cached.min(prompt_tokens.saturating_sub(1));
        if !matches!(self.prefill, PrefillUnit::Measured { .. }) || prompt_tokens == 0 {
            return true;
        }
        let candidate = PrefillLane {
            seq_id,
            arrival_order: submitted.unwrap_or(self.next_order),
            processed,
            remaining: prompt_tokens - processed,
            overdue: overtaken >= MAX_OVERTAKES,
        };
        let mut ready = self.ready_lanes(&[]);
        ready.push(candidate);
        let Some(unit) = self.plan_unit(ready.clone(), self.contended()) else {
            return true;
        };
        let rows_of = |lane: &PrefillLane| {
            unit.lanes
                .iter()
                .find(|(planned, _)| planned.seq_id == lane.seq_id)
                .map(|&(_, rows)| rows)
        };
        let Some(rows) = rows_of(&candidate) else {
            return false;
        };
        rows == candidate.remaining
            || !ready.iter().any(|other| {
                other.seq_id != seq_id
                    && leads_before(other, &candidate)
                    && rows_of(other) != Some(other.remaining)
            })
    }

    /// Splash `Engine::prepare`'s first half: a control command that is
    /// due, or the admitted turns' plan. With `clock`, a measured plan is
    /// made without the waiters whose retry is not due.
    pub fn prepare_step(
        &mut self,
        clock: Option<RetryClock>,
    ) -> Result<Prepared<P, Exclusive, Barrier>, String> {
        if let Some(action) = self.pop_control_if_due() {
            return Ok(Prepared::Action(action));
        }
        self.admit_waiting()?;
        if self.running.is_empty() {
            return Ok(Prepared::Action(SchedulerAction::Idle));
        }
        let plan = self.build_plan(clock);
        if plan.rows.is_empty() {
            return Ok(Prepared::Action(SchedulerAction::Idle));
        }
        Ok(Prepared::Plan(plan))
    }

    /// [`Self::prepare_step`] and [`Self::finish_step`] with every row
    /// granted.
    #[cfg(test)]
    pub fn drive_once<E>(
        &mut self,
        executor: &mut E,
    ) -> Result<SchedulerAction<P, Exclusive, Barrier>, SchedulerError<E::Error>>
    where
        E: StepExecutor<P>,
    {
        match self
            .prepare_step(None)
            .map_err(SchedulerError::InvalidResult)?
        {
            Prepared::Action(action) => Ok(action),
            Prepared::Plan(plan) => self.finish_step(plan, &[], executor, JudgePolicy::Legacy),
        }
    }

    /// Runs the granted rows of `plan` and applies the result; a denied
    /// row reports zero progress (Splash: a refused lane leaves the plan).
    pub fn finish_step<E>(
        &mut self,
        plan: StepPlan,
        grants: &[RowGrant],
        executor: &mut E,
        policy: JudgePolicy,
    ) -> Result<SchedulerAction<P, Exclusive, Barrier>, SchedulerError<E::Error>>
    where
        E: StepExecutor<P>,
    {
        let denied: HashMap<SeqId, Denial> = plan
            .rows
            .iter()
            .zip(grants)
            .filter_map(|(row, grant)| grant.map(|denial| (row.seq_id, denial)))
            .collect();
        let executed = StepPlan {
            global_step: plan.global_step,
            rows: plan
                .rows
                .iter()
                .filter(|row| !denied.contains_key(&row.seq_id))
                .cloned()
                .collect(),
            contended: plan.contended,
            leader_cut: plan.leader_cut,
        };
        let result = if executed.rows.is_empty() {
            StepResult {
                rows: Vec::new(),
                executed_decode_batch: 0,
                executed_greedy_epilogue_batch: 0,
                rows_alloc_evicted: 0,
                prefill_wall_micros: 0,
                decode_wall_micros: 0,
            }
        } else {
            executor
                .execute(&executed, &mut self.running)
                .map_err(SchedulerError::Executor)?
        };
        if result.rows.len() != executed.rows.len() {
            return Err(SchedulerError::InvalidResult(format!(
                "executor returned {} rows for a {}-row plan",
                result.rows.len(),
                executed.rows.len()
            )));
        }
        let executed_decode_batch = result.executed_decode_batch;
        let executed_greedy_epilogue_batch = result.executed_greedy_epilogue_batch;
        let rows_alloc_evicted = result.rows_alloc_evicted;
        let (prefill_wall_micros, decode_wall_micros) =
            (result.prefill_wall_micros, result.decode_wall_micros);
        let mut ran = result.rows.into_iter();
        let rows: Vec<RowStepResult> = plan
            .rows
            .iter()
            .filter_map(|row| {
                if denied.contains_key(&row.seq_id) {
                    Some(RowStepResult {
                        seq_id: row.seq_id,
                        num_computed_tokens: 0,
                        generated_token: None,
                        speculative: None,
                        finished: false,
                        allocation_blocked: true,
                        prefill_micros: 0,
                    })
                } else {
                    ran.next()
                }
            })
            .collect();
        let result = StepResult {
            rows,
            executed_decode_batch,
            executed_greedy_epilogue_batch,
            rows_alloc_evicted,
            prefill_wall_micros,
            decode_wall_micros,
        };
        let blocked_rows: HashSet<SeqId> = result
            .rows
            .iter()
            .filter(|row| row.allocation_blocked)
            .map(|row| row.seq_id)
            .collect();
        let progressed = result.rows.iter().any(|row| !row.allocation_blocked);
        let planned_decode_rows = executed
            .rows
            .iter()
            .filter(|row| row.kind == StepKind::Decode)
            .count();
        if executed_decode_batch > planned_decode_rows {
            return Err(SchedulerError::InvalidResult(format!(
                "executor reported decode occupancy {executed_decode_batch} for {planned_decode_rows} planned decode rows"
            )));
        }
        if executed_greedy_epilogue_batch > executed_decode_batch {
            return Err(SchedulerError::InvalidResult(format!(
                "executor reported greedy epilogue occupancy {executed_greedy_epilogue_batch} for {executed_decode_batch} executed decode rows"
            )));
        }
        self.apply_result(&plan, result)
            .map_err(SchedulerError::InvalidResult)?;
        self.global_step = self.global_step.saturating_add(1);
        self.stats.global_steps = self.global_step;
        self.stats.rows_alloc_evicted = self
            .stats
            .rows_alloc_evicted
            .saturating_add(u64::from(rows_alloc_evicted));
        let decode_occupancy = executed_decode_batch;
        if decode_occupancy != 0 {
            self.stats.max_batch_occupancy = self.stats.max_batch_occupancy.max(decode_occupancy);
            *self
                .stats
                .decode_batch_occupancy_hist
                .entry(decode_occupancy)
                .or_default() += 1;
        }
        if executed_greedy_epilogue_batch != 0 {
            self.stats.fused_greedy_epilogue_steps =
                self.stats.fused_greedy_epilogue_steps.saturating_add(1);
        }

        let mut preempted = None;
        let mut blocked_alone = None;
        let mut blocked = Vec::new();
        match policy {
            JudgePolicy::Ledger => {
                if !denied.is_empty() && progressed {
                    self.stats.replans = self.stats.replans.saturating_add(1);
                }
                // A row the executor blocked after its claim was granted
                // lacked KV blocks.
                let executor_kv = Denial {
                    blocks: Some(BlockRefusal::default()),
                    ..Denial::default()
                };
                blocked = plan
                    .rows
                    .iter()
                    .filter(|row| blocked_rows.contains(&row.seq_id))
                    .map(|row| {
                        (
                            row.seq_id,
                            denied.get(&row.seq_id).copied().unwrap_or(executor_kv),
                        )
                    })
                    .collect();
            }
            JudgePolicy::Legacy => {
                if !blocked_rows.is_empty()
                    && let Some(index) = self
                        .running
                        .iter()
                        .rposition(|turn| turn.status == TurnStatus::Running)
                {
                    let another_lane =
                        self.running.len() > 1 || self.waiting.iter().any(|turn| turn.ssd_lane);
                    if !another_lane {
                        blocked_alone = Some(self.running[index].seq_id);
                    } else {
                        let turn = self.preempt_running_at(index);
                        let blocked = blocked_rows.contains(&turn.seq_id);
                        preempted = Some(Box::new(PreemptedTurn { turn, blocked }));
                    }
                }
            }
        }

        let mut completed = Vec::new();
        let mut retained = Vec::with_capacity(self.running.len());
        for turn in self.running.drain(..) {
            if turn.status == TurnStatus::Draining {
                self.stats.completed = self.stats.completed.saturating_add(1);
                completed.push(turn);
            } else {
                retained.push(turn);
            }
        }
        self.running = retained;
        Ok(SchedulerAction::Stepped {
            planned: plan.rows.iter().map(|row| row.seq_id).collect(),
            prefill_wall_micros,
            #[cfg(test)]
            plan,
            completed,
            preempted,
            blocked_alone,
            blocked,
            progressed,
        })
    }

    /// Splash `Engine::laneToYield`: the running lane of the lowest rank.
    pub fn lane_to_yield(&self) -> Option<SeqId> {
        self.running
            .iter()
            .filter(|turn| turn.status == TurnStatus::Running)
            .min_by_key(|turn| turn.claim_rank())
            .map(|turn| turn.seq_id)
    }

    /// Splash `Scheduler::commit`'s overtakes and `complete`'s measure,
    /// counters and decode-debt work-off, over the unit's rows that ran: a
    /// blocked, denied or cancelled row is one the unit ran without, and a
    /// row that ran untimed sizes nothing. A lane refused its row is passed
    /// over as Splash's replan leaves it (`Engine::tick` commits with the
    /// excluded lanes less the refused). Whether a prefill row ran.
    fn observe_prefill_unit(&mut self, plan: &StepPlan, result: &StepResult) -> bool {
        let PrefillUnit::Measured {
            measured,
            decode_debt_ms,
            ..
        } = &mut self.prefill
        else {
            return false;
        };
        let mut served: Vec<SeqId> = Vec::new();
        let mut cancelled: Vec<SeqId> = Vec::new();
        let (mut rows, mut pairs, mut untimed) = (0u32, 0u64, false);
        let mut decoded = 0usize;
        for (planned, output) in plan.rows.iter().zip(&result.rows) {
            if planned.kind != StepKind::Prefill {
                if !output.allocation_blocked && !planned.cancel_snapshot {
                    decoded += 1;
                }
                continue;
            }
            if planned.cancel_snapshot {
                cancelled.push(planned.seq_id);
            }
            if output.allocation_blocked || planned.cancel_snapshot {
                continue;
            }
            if output.prefill_micros == 0 {
                untimed = true;
                continue;
            }
            served.push(planned.seq_id);
            rows += planned.num_tokens;
            pairs += dense_attention_pairs(
                u64::from(planned.token_start),
                u64::from(planned.num_tokens),
            );
        }
        if decoded > 0 && result.decode_wall_micros > 0 {
            *decode_debt_ms =
                (*decode_debt_ms - result.decode_wall_micros as f64 / 1000.0).max(0.0);
        }
        if rows > 0 && !untimed && result.prefill_wall_micros > 0 {
            *measured = Some(MeasuredUnit {
                rows,
                pairs,
                millis: result.prefill_wall_micros as f64 / 1000.0,
            });
        }
        let counters = &self.counters;
        if decoded > 0 {
            counters.decode_batches.fetch_add(1, Ordering::Relaxed);
            let width = decoded.min(counters.decode_batches_by_width.len()) - 1;
            counters.decode_batches_by_width[width].fetch_add(1, Ordering::Relaxed);
        }
        if rows > 0 {
            counters.prefill_units.fetch_add(1, Ordering::Relaxed);
            counters.prefill_batches.fetch_add(1, Ordering::Relaxed);
            counters
                .prefill_rows
                .fetch_add(u64::from(rows), Ordering::Relaxed);
            if plan
                .leader_cut
                .is_some_and(|leader| served.contains(&leader))
            {
                counters.bound_cuts.fetch_add(1, Ordering::Relaxed);
            }
            if plan.contended && result.prefill_wall_micros > 0 {
                counters
                    .contended_prefill
                    .add(result.prefill_wall_micros.saturating_mul(1000));
            }
        }
        let Some(youngest) = self
            .running
            .iter()
            .filter(|turn| served.contains(&turn.seq_id))
            .map(|turn| turn.arrival_order)
            .max()
        else {
            return false;
        };
        self.youngest_served = Some(youngest);
        let excluded = &self.plan_excluded;
        let waiting = self
            .waiting
            .iter_mut()
            .filter(|turn| matches!(turn.status, TurnStatus::Waiting | TurnStatus::WaitingForSsd));
        let prefilling = self.running.iter_mut().filter(|turn| {
            turn.status == TurnStatus::Running && turn.num_computed_tokens < turn.prompt_tokens
        });
        for turn in prefilling.chain(waiting) {
            if excluded.contains(&turn.seq_id) || cancelled.contains(&turn.seq_id) {
                continue;
            }
            if served.contains(&turn.seq_id) {
                turn.overtaken = 0;
            } else if turn.arrival_order < youngest && turn.overtaken < MAX_OVERTAKES {
                turn.overtaken += 1;
            }
        }
        true
    }

    /// Splash `Scheduler::complete` after progress: a prefill unit owes the
    /// lanes that decode beside it now (those it finished included, none it
    /// was planned without) its decode share of its wall; then the stale
    /// debt drops.
    fn settle_prefill_unit(&mut self, prefill_ran: bool, prefill_wall_micros: u64) {
        let PrefillUnit::Measured {
            decode_share,
            decode_debt_ms,
            ..
        } = &mut self.prefill
        else {
            return;
        };
        let excluded = &self.plan_excluded;
        let owes = self.running.iter().any(|turn| {
            turn.status == TurnStatus::Running
                && turn.num_computed_tokens >= turn.prompt_tokens
                && !turn.cancel_snapshot
                && !excluded.contains(&turn.seq_id)
        });
        if prefill_ran && prefill_wall_micros > 0 && owes && *decode_share > 0.0 {
            *decode_debt_ms += *decode_share * prefill_wall_micros as f64 / 1000.0;
        }
        self.drop_stale_decode_debt();
    }

    /// Splash `Scheduler::snapshot`'s phase counts for the status: the
    /// turns here by phase (a resident lane waiting for its retry is still
    /// prefilling or decoding, as `noteRefusal` leaves its phase), plus the
    /// requests the caller holds before admission, `queued` waiting for
    /// scheduling and `waiting_resources` refused a lane or memory.
    pub fn publish_gauges(&self, mut queued: u64, mut waiting_resources: u64) {
        let mut waiting_prefix = 0u64;
        for turn in &self.waiting {
            match turn.status {
                TurnStatus::Waiting => queued += 1,
                TurnStatus::Preempted => waiting_resources += 1,
                TurnStatus::WaitingForSsd => waiting_prefix += 1,
                _ => {}
            }
        }
        let (mut prefilling, mut decoding) = (0u64, 0u64);
        for turn in &self.running {
            if turn.status != TurnStatus::Running {
                continue;
            }
            if turn.num_computed_tokens < turn.prompt_tokens {
                prefilling += 1;
            } else {
                decoding += 1;
            }
        }
        let counters = &self.counters;
        counters.queued.store(queued, Ordering::Relaxed);
        counters
            .waiting_resources
            .store(waiting_resources, Ordering::Relaxed);
        counters
            .waiting_prefix
            .store(waiting_prefix, Ordering::Relaxed);
        counters.prefilling.store(prefilling, Ordering::Relaxed);
        counters.decoding.store(decoding, Ordering::Relaxed);
    }

    /// Splash `Scheduler::dropStaleDecodeDebt`: no lane decodes, so none is
    /// owed.
    fn drop_stale_decode_debt(&mut self) {
        let PrefillUnit::Measured { decode_debt_ms, .. } = &mut self.prefill else {
            return;
        };
        let decoding = self.running.iter().any(|turn| {
            turn.status == TurnStatus::Running
                && turn.num_computed_tokens >= turn.prompt_tokens
                && !turn.cancel_snapshot
        });
        if !decoding {
            *decode_debt_ms = 0.0;
        }
    }

    fn apply_result(&mut self, plan: &StepPlan, result: StepResult) -> Result<(), String> {
        if result.rows.len() != plan.rows.len() {
            return Err(format!(
                "executor returned {} rows for a {}-row plan",
                result.rows.len(),
                plan.rows.len()
            ));
        }
        let mut seen = HashSet::with_capacity(result.rows.len());
        for (planned, output) in plan.rows.iter().zip(&result.rows) {
            if planned.seq_id != output.seq_id {
                return Err(format!(
                    "executor row order mismatch: planned {}, returned {}",
                    planned.seq_id, output.seq_id
                ));
            }
            if !seen.insert(output.seq_id) {
                return Err(format!("executor returned duplicate row {}", output.seq_id));
            }
            if output.allocation_blocked {
                if output.num_computed_tokens != 0
                    || output.generated_token.is_some()
                    || output.finished
                    || output.speculative.is_some()
                {
                    return Err(format!(
                        "blocked executor row {} must report zero progress and remain live",
                        output.seq_id
                    ));
                }
            } else if let Some(spec) = &output.speculative {
                if planned.kind != StepKind::Decode
                    || spec.verified_tokens > planned.num_tokens
                    || output.num_computed_tokens > spec.verified_tokens
                    || output.generated_token.is_some()
                    || (!output.finished
                        && (spec.generated_tokens.len() != output.num_computed_tokens as usize
                            || output.num_computed_tokens == 0))
                {
                    return Err(format!(
                        "invalid speculative progress for request {}",
                        output.seq_id
                    ));
                }
            } else if output.num_computed_tokens != planned.num_tokens {
                return Err(format!(
                    "executor row {} computed {} tokens for a {}-token slice",
                    output.seq_id, output.num_computed_tokens, planned.num_tokens
                ));
            }
            let turn = self
                .running
                .iter()
                .find(|turn| turn.seq_id == output.seq_id)
                .ok_or_else(|| format!("executor returned unknown row {}", output.seq_id))?;
            if output.speculative.is_some()
                && turn.num_tokens != turn.num_computed_tokens.saturating_add(1)
            {
                return Err(format!(
                    "speculative request {} must have exactly one pending anchor",
                    turn.seq_id
                ));
            }
            if output.generated_token.is_some()
                && turn
                    .num_computed_tokens
                    .checked_add(output.num_computed_tokens)
                    != Some(turn.num_tokens)
            {
                return Err(format!(
                    "request {} generated before all {} known tokens were computed",
                    turn.seq_id, turn.num_tokens
                ));
            }
        }
        let prefill_ran = self.observe_prefill_unit(plan, &result);
        let prefill_wall_micros = result.prefill_wall_micros;
        for output in result.rows {
            let turn = self
                .running
                .iter_mut()
                .find(|turn| turn.seq_id == output.seq_id)
                .ok_or_else(|| format!("executor returned unknown row {}", output.seq_id))?;
            turn.num_computed_tokens = turn
                .num_computed_tokens
                .checked_add(output.num_computed_tokens)
                .ok_or_else(|| format!("request {} computed-token overflow", turn.seq_id))?;
            // Splash `Scheduler::promptCompleted`.
            if turn.num_computed_tokens >= turn.prompt_tokens {
                turn.overtaken = 0;
            }
            if turn.block_size != 0 {
                turn.block_materialized_blocks = turn
                    .block_materialized_blocks
                    .max(turn.num_computed_tokens.div_ceil(turn.block_size));
            }
            if let Some(token) = output.generated_token {
                turn.token_history.push(token);
                turn.num_tokens = turn
                    .num_tokens
                    .checked_add(1)
                    .ok_or_else(|| format!("request {} token-count overflow", turn.seq_id))?;
            }
            if let Some(spec) = output.speculative {
                turn.num_tokens = turn
                    .num_tokens
                    .checked_add(
                        u32::try_from(spec.generated_tokens.len())
                            .map_err(|_| "speculative output length overflow")?,
                    )
                    .ok_or_else(|| format!("request {} token-count overflow", turn.seq_id))?;
                turn.token_history.extend(spec.generated_tokens);
            }
            if output.finished {
                turn.status = TurnStatus::Draining;
            }
            if output.prefill_micros != 0 {
                self.stats.prefill_tokens = self
                    .stats
                    .prefill_tokens
                    .saturating_add(u64::from(output.num_computed_tokens));
                self.stats.prefill_micros = self
                    .stats
                    .prefill_micros
                    .saturating_add(output.prefill_micros);
            }
        }
        self.settle_prefill_unit(prefill_ran, prefill_wall_micros);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Families without the rest-of-window budget keep `+ 1`: the last
    /// sampled token is never written, so a prompt that fills the window still
    /// gets one token. Mutation: Splash's clamp regardless of
    /// `rest_of_window` → red (Ok(16) ≠ Ok(17)).
    #[test]
    fn scheduled_output_hint_clamps_to_remaining_context() {
        assert_eq!(clamp_scheduled_output_tokens(48, 100, 64, false), Ok(17));
        assert_eq!(clamp_scheduled_output_tokens(48, 10, 64, false), Ok(10));
        assert_eq!(clamp_scheduled_output_tokens(64, 1, 64, false), Ok(1));
        assert_eq!(scheduled_materialized_tokens(48, 17), 64);
        assert_eq!(scheduled_materialized_tokens(64, 1), 64);
    }

    #[test]
    fn scheduled_output_hint_rejects_an_oversized_prompt() {
        let oversized = clamp_scheduled_output_tokens(65, 1, 64, false).unwrap_err();
        assert!(oversized.starts_with("context_length_exceeded: prompt (65)"));
    }

    /// Splash's scheduled clamp (frontend.py:1212-1222): `prompt >= window`
    /// is rejected and the budget is `window - prompt`.
    /// Mutations: `.saturating_add(1)` on the rest-of-window budget → red
    /// (Ok(17) ≠ Ok(16)); `prompt >= capacity` → `prompt > capacity` → red
    /// (a full-window prompt is admitted).
    #[test]
    fn rest_of_window_scheduled_clamp_matches_splash() {
        assert_eq!(clamp_scheduled_output_tokens(48, 100, 64, true), Ok(16));
        assert_eq!(clamp_scheduled_output_tokens(48, 10, 64, true), Ok(10));
        assert_eq!(clamp_scheduled_output_tokens(63, 100, 64, true), Ok(1));
        assert_eq!(scheduled_materialized_tokens(48, 16), 63);
        let full = clamp_scheduled_output_tokens(64, 1, 64, true).unwrap_err();
        assert!(full.starts_with("context_length_exceeded: rendered prompt has 64 tokens"));
        let oversized = clamp_scheduled_output_tokens(65, 1, 64, true).unwrap_err();
        assert!(oversized.starts_with("context_length_exceeded:"));
    }

    #[derive(Default)]
    struct MockStep {
        plans: Vec<StepPlan>,
        finish: HashSet<SeqId>,
        allocation_blocked: HashSet<SeqId>,
        next_token: u32,
        executed_greedy_epilogue_batch: usize,
    }

    impl StepExecutor<&'static str> for MockStep {
        type Error = String;

        fn execute(
            &mut self,
            plan: &StepPlan,
            _running: &mut [TurnState<&'static str>],
        ) -> Result<StepResult, Self::Error> {
            self.plans.push(plan.clone());
            let rows = plan
                .rows
                .iter()
                .map(|row| {
                    let generated_token = (row.kind == StepKind::Decode
                        || row.token_start + row.num_tokens >= 4)
                        .then(|| {
                            self.next_token += 1;
                            self.next_token
                        });
                    RowStepResult {
                        seq_id: row.seq_id,
                        num_computed_tokens: if self.allocation_blocked.contains(&row.seq_id) {
                            0
                        } else {
                            row.num_tokens
                        },
                        generated_token: (!self.allocation_blocked.contains(&row.seq_id))
                            .then_some(generated_token)
                            .flatten(),
                        speculative: None,
                        finished: !self.allocation_blocked.contains(&row.seq_id)
                            && (self.finish.contains(&row.seq_id) || row.cancel_snapshot),
                        allocation_blocked: self.allocation_blocked.contains(&row.seq_id),
                        prefill_micros: u64::from(row.kind == StepKind::Prefill),
                    }
                })
                .collect();
            Ok(StepResult {
                rows,
                executed_decode_batch: plan
                    .rows
                    .iter()
                    .filter(|row| row.kind == StepKind::Decode)
                    .count(),
                executed_greedy_epilogue_batch: self.executed_greedy_epilogue_batch,
                rows_alloc_evicted: 0,
                prefill_wall_micros: 0,
                decode_wall_micros: 0,
            })
        }
    }

    fn turn(
        seq_id: SeqId,
        computed: u32,
        breaks: &[u32],
        cancel: Option<Arc<AtomicBool>>,
    ) -> TurnState<&'static str> {
        TurnState::new(
            seq_id,
            vec![1, 2, 3, 4],
            computed,
            breaks.to_vec(),
            cancel,
            "payload",
        )
        .expect("turn")
    }

    #[test]
    fn fcfs_admission_and_one_budget_cover_prefix_prefill_and_decode() {
        let mut scheduler = Scheduler::<_, (), ()>::new(3, 3).expect("scheduler");
        scheduler
            .enqueue_turn(turn(1, 0, &[2, 4], None))
            .expect("enqueue 1");
        scheduler
            .enqueue_turn(turn(2, 3, &[4], None))
            .expect("enqueue 2");
        scheduler
            .enqueue_turn(turn(3, 4, &[4], None))
            .expect("enqueue 3");
        scheduler.running.push({
            let mut decode = scheduler.waiting.pop_back().expect("decode waiting");
            decode.status = TurnStatus::Running;
            decode.token_history.push(9);
            decode.num_tokens += 1;
            decode
        });
        let mut step = MockStep::default();
        let SchedulerAction::Stepped { plan, .. } = scheduler.drive_once(&mut step).expect("step")
        else {
            panic!("expected step");
        };
        assert_eq!(
            plan.rows,
            vec![
                StepRow {
                    seq_id: 3,
                    kind: StepKind::Decode,
                    token_start: 4,
                    num_tokens: 1,
                    cancel_snapshot: false,
                },
                StepRow {
                    seq_id: 1,
                    kind: StepKind::Prefill,
                    token_start: 0,
                    num_tokens: 2,
                    cancel_snapshot: false,
                },
            ]
        );
        assert_eq!(scheduler.running()[0].seq_id, 3);
        assert_eq!(scheduler.running()[1].seq_id, 1);
        assert_eq!(scheduler.running()[2].seq_id, 2);
    }

    #[test]
    fn speculative_queries_share_budget_but_only_committed_rows_advance_history() {
        let mut scheduler = Scheduler::<_, (), ()>::new(3, 7).unwrap();
        for seq_id in 1..=3 {
            let mut row = turn(seq_id, 4, &[4], None);
            row.status = TurnStatus::Running;
            row.token_history.push(8);
            row.num_tokens += 1;
            row.decode_draft_allowance = 4;
            scheduler.running.push(row);
        }
        let plan = scheduler.build_plan(None);
        assert_eq!(
            plan.rows
                .iter()
                .map(|row| row.num_tokens)
                .collect::<Vec<_>>(),
            vec![5, 1, 1]
        );
        assert_eq!(plan.rows.iter().map(|row| row.num_tokens).sum::<u32>(), 7);
        let result = StepResult {
            rows: plan
                .rows
                .iter()
                .map(|row| RowStepResult {
                    seq_id: row.seq_id,
                    num_computed_tokens: 1,
                    generated_token: None,
                    speculative: Some(SpeculativeRowResult {
                        verified_tokens: row.num_tokens,
                        generated_tokens: vec![9],
                    }),
                    finished: false,
                    allocation_blocked: false,
                    prefill_micros: 0,
                })
                .collect(),
            executed_decode_batch: 3,
            executed_greedy_epilogue_batch: 3,
            rows_alloc_evicted: 0,
            prefill_wall_micros: 0,
            decode_wall_micros: 0,
        };
        scheduler.apply_result(&plan, result).unwrap();
        for row in scheduler.running() {
            assert_eq!(row.num_computed_tokens, 5);
            assert_eq!(row.num_tokens, 6);
            assert_eq!(&row.token_history[4..], &[8, 9]);
        }
        let plan = scheduler.build_plan(None);
        let before = scheduler.running()[0].token_history.clone();
        let invalid = StepResult {
            rows: plan
                .rows
                .iter()
                .map(|row| RowStepResult {
                    seq_id: row.seq_id,
                    num_computed_tokens: 1,
                    generated_token: None,
                    speculative: Some(SpeculativeRowResult {
                        verified_tokens: row.num_tokens,
                        generated_tokens: vec![9, 10],
                    }),
                    finished: false,
                    allocation_blocked: false,
                    prefill_micros: 0,
                })
                .collect(),
            executed_decode_batch: 3,
            executed_greedy_epilogue_batch: 0,
            rows_alloc_evicted: 0,
            prefill_wall_micros: 0,
            decode_wall_micros: 0,
        };
        assert!(scheduler.apply_result(&plan, invalid).is_err());
        assert_eq!(scheduler.running()[0].token_history, before);
    }

    #[test]
    fn no_phase_queue_progress_derives_kind_and_honors_pinned_breaks() {
        let mut scheduler = Scheduler::<_, (), ()>::new(1, 8).expect("scheduler");
        scheduler
            .enqueue_turn(turn(7, 0, &[2, 4], None))
            .expect("enqueue");
        let mut step = MockStep::default();
        for expected in [
            (StepKind::Prefill, 0, 2),
            (StepKind::Prefill, 2, 2),
            (StepKind::Decode, 4, 1),
        ] {
            let SchedulerAction::Stepped { plan, .. } =
                scheduler.drive_once(&mut step).expect("step")
            else {
                panic!("expected step");
            };
            assert_eq!(
                (
                    plan.rows[0].kind,
                    plan.rows[0].token_start,
                    plan.rows[0].num_tokens
                ),
                expected
            );
        }
    }

    #[test]
    fn exclusive_and_barrier_wait_for_running_rows_and_preserve_order() {
        let mut scheduler = Scheduler::<_, &str, &str>::new(1, 4).expect("scheduler");
        scheduler
            .enqueue_turn(turn(1, 4, &[4], None))
            .expect("enqueue running");
        scheduler.running.push({
            let mut active = scheduler.waiting.pop_front().expect("active");
            active.status = TurnStatus::Running;
            active.token_history.push(8);
            active.num_tokens += 1;
            active
        });
        scheduler.enqueue_exclusive("generic");
        scheduler.enqueue_barrier("reset");
        scheduler
            .enqueue_turn(turn(2, 0, &[4], None))
            .expect("enqueue after controls");
        let mut step = MockStep::default();
        step.finish.insert(1);
        assert!(matches!(
            scheduler.drive_once(&mut step).expect("drain active"),
            SchedulerAction::Stepped { .. }
        ));
        assert!(matches!(
            scheduler.drive_once(&mut step).expect("exclusive"),
            SchedulerAction::Exclusive("generic")
        ));
        assert!(matches!(
            scheduler.drive_once(&mut step).expect("barrier"),
            SchedulerAction::Barrier("reset")
        ));
        assert!(scheduler.running().is_empty());
        assert_eq!(scheduler.waiting_len(), 1);
    }

    #[test]
    fn pending_control_is_visible_before_it_becomes_executable() {
        let mut scheduler = Scheduler::<_, &str, &str>::new(1, 4).expect("scheduler");
        scheduler
            .enqueue_turn(turn(1, 0, &[4], None))
            .expect("enqueue older turn");
        scheduler.enqueue_barrier("reset");
        assert!(
            scheduler.has_pending_control(),
            "model admission must pause even while an older row keeps the barrier from executing"
        );
        let mut step = MockStep::default();
        assert!(matches!(
            scheduler.drive_once(&mut step).expect("older turn runs"),
            SchedulerAction::Stepped { .. }
        ));
        assert!(scheduler.has_pending_control());
    }

    /// MUTATION: drop the `recurrent_state_reservation_active` gate in
    /// `TurnState::outstanding_scheduled_state_bytes`; the preempted row keeps
    /// reporting 70 and the `0` assertion fails.
    #[test]
    fn outstanding_scheduled_state_skips_preempted_rows_and_held_state() {
        let mut scheduler = Scheduler::<_, (), ()>::new(2, 4).expect("scheduler");
        scheduler
            .enqueue_turn(turn(1, 0, &[4], None).with_scheduled_state_reservation(100))
            .expect("enqueue 1");
        scheduler
            .enqueue_turn(turn(2, 0, &[4], None).with_scheduled_state_reservation(50))
            .expect("enqueue 2");
        let held = |seq_id: SeqId| if seq_id == 1 { 30 } else { 80 };
        assert_eq!(scheduler.outstanding_scheduled_state_bytes(held), 70);
        let victim = scheduler.take_waiting(1).expect("take victim");
        scheduler.prepend_preempted(victim);
        assert_eq!(scheduler.outstanding_scheduled_state_bytes(held), 0);
        let victim = scheduler.take_preempted().expect("resume victim");
        scheduler.ready_preempted(victim, false);
        assert_eq!(scheduler.outstanding_scheduled_state_bytes(held), 70);
    }

    #[test]
    fn later_waiter_cannot_leapfrog_preempted_fcfs_head_or_release_barrier() {
        let mut scheduler = Scheduler::<_, (), &str>::new(1, 4).expect("scheduler");
        scheduler
            .enqueue_turn(turn(1, 0, &[4], None))
            .expect("enqueue victim");
        let victim = scheduler.take_waiting(1).expect("take victim");
        scheduler.prepend_preempted(victim);
        scheduler
            .enqueue_turn(turn(2, 0, &[4], None))
            .expect("enqueue later turn");
        scheduler.enqueue_barrier("reset");

        let mut step = MockStep::default();
        assert!(matches!(
            scheduler.drive_once(&mut step).expect("parked head"),
            SchedulerAction::Idle
        ));
        assert!(scheduler.running().is_empty());
        assert_eq!(scheduler.waiting_len(), 2);
        assert!(scheduler.has_pending_control());

        let victim = scheduler.take_preempted().expect("resume victim first");
        scheduler.ready_preempted(victim, false);
        let SchedulerAction::Stepped { plan, .. } = scheduler
            .drive_once(&mut step)
            .expect("resumed victim runs")
        else {
            panic!("expected resumed victim step");
        };
        assert_eq!(plan.rows[0].seq_id, 1);
    }

    #[test]
    fn cancelled_ssd_waiter_is_reaped_without_waiting_for_io() {
        let cancelled = Arc::new(AtomicBool::new(false));
        let mut scheduler = Scheduler::<_, (), ()>::new(2, 4).expect("scheduler");
        scheduler
            .enqueue_turn(turn(1, 0, &[4], Some(Arc::clone(&cancelled))))
            .expect("enqueue restore waiter");
        scheduler
            .park_waiting_for_ssd(1)
            .expect("park restore waiter");
        scheduler
            .enqueue_turn(turn(2, 0, &[4], None))
            .expect("enqueue healthy peer");
        cancelled.store(true, Ordering::Relaxed);

        let reaped = scheduler.take_cancelled_waiters();
        assert_eq!(reaped.len(), 1);
        assert_eq!(reaped[0].seq_id, 1);
        assert_eq!(scheduler.stats().ssd_restore_waiting, 0);
        assert!(scheduler.contains_seq(2));
        assert!(!scheduler.contains_seq(1));
    }

    #[test]
    fn cancel_is_snapshotted_once_before_execute_and_membership_rebuilds() {
        let cancel = Arc::new(AtomicBool::new(false));
        let mut scheduler = Scheduler::<_, (), ()>::new(2, 2).expect("scheduler");
        for seq_id in [1, 2] {
            let flag = (seq_id == 1).then(|| cancel.clone());
            let mut request = turn(seq_id, 4, &[4], flag);
            request.token_history.push(9);
            request.num_tokens += 1;
            scheduler.enqueue_turn(request).expect("enqueue");
        }
        cancel.store(true, Ordering::Relaxed);
        let mut step = MockStep::default();
        let SchedulerAction::Stepped {
            plan, completed, ..
        } = scheduler.drive_once(&mut step).expect("step")
        else {
            panic!("expected step");
        };
        assert_eq!(
            plan.rows
                .iter()
                .map(|row| row.cancel_snapshot)
                .collect::<Vec<_>>(),
            [true, false]
        );
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].seq_id, 1);
        assert_eq!(scheduler.running().len(), 1);
        assert_eq!(scheduler.running()[0].seq_id, 2);
    }

    #[test]
    fn global_step_cadence_and_decode_occupancy_histogram_are_non_vacuous() {
        let mut scheduler = Scheduler::<_, (), ()>::new(2, 2).expect("scheduler");
        for seq_id in [1, 2] {
            let mut request = turn(seq_id, 4, &[4], None);
            request.token_history.push(9);
            request.num_tokens += 1;
            scheduler.enqueue_turn(request).expect("enqueue");
        }
        let mut step = MockStep::default();
        for _ in 0..4 {
            assert!(matches!(
                scheduler.drive_once(&mut step).expect("step"),
                SchedulerAction::Stepped { .. }
            ));
        }
        assert_eq!(scheduler.stats().global_steps, 4);
        assert_eq!(scheduler.stats().max_batch_occupancy, 2);
        assert_eq!(
            scheduler.stats().decode_batch_occupancy_hist.get(&2),
            Some(&4)
        );
        let js = scheduler.stats().to_js();
        assert_eq!(js.max_batch_occupancy, 2);
        assert_eq!(js.decode_batch_occupancy_hist[0].occupancy, 2);
        assert_eq!(js.decode_batch_occupancy_hist[0].steps, 4.0);
    }

    #[test]
    fn fused_greedy_epilogue_stats_require_executor_reported_engagement() {
        let mut scheduler = Scheduler::<_, (), ()>::new(2, 2).expect("scheduler");
        for seq_id in [1, 2] {
            let mut request = turn(seq_id, 4, &[4], None);
            request.token_history.push(9);
            request.num_tokens += 1;
            scheduler.enqueue_turn(request).expect("enqueue");
        }
        let mut step = MockStep {
            executed_greedy_epilogue_batch: 2,
            ..MockStep::default()
        };
        assert!(matches!(
            scheduler.drive_once(&mut step).expect("fused step"),
            SchedulerAction::Stepped { .. }
        ));
        assert_eq!(scheduler.stats().fused_greedy_epilogue_steps, 1);
        assert_eq!(scheduler.stats().to_js().fused_greedy_epilogue_steps, 1.0);
    }

    #[test]
    fn greedy_epilogue_occupancy_cannot_exceed_executed_decode_occupancy() {
        let mut scheduler = Scheduler::<_, (), ()>::new(1, 1).expect("scheduler");
        let mut request = turn(1, 4, &[4], None);
        request.token_history.push(9);
        request.num_tokens += 1;
        scheduler.enqueue_turn(request).expect("enqueue");
        let mut step = MockStep {
            executed_greedy_epilogue_batch: 2,
            ..MockStep::default()
        };
        assert!(matches!(
            scheduler.drive_once(&mut step),
            Err(SchedulerError::InvalidResult(message))
                if message.contains("greedy epilogue occupancy 2 for 1 executed decode rows")
        ));
    }

    /// A resuming victim waits only on rows that can run before it: an
    /// earlier waiter (here parked for SSD) or a running row; a later waiter
    /// is stuck behind the victim, so waiting on it never ends.
    /// Mutation: `resume_may_wait` counts every waiter → red.
    #[test]
    fn resume_waits_only_on_turns_that_can_run_first() {
        let mut scheduler = Scheduler::<_, (), ()>::new(2, 8).expect("scheduler");
        scheduler
            .enqueue_turn(turn(1, 0, &[4], None))
            .expect("enqueue earlier waiter");
        scheduler
            .park_waiting_for_ssd(1)
            .expect("park earlier waiter");
        scheduler
            .enqueue_turn(turn(2, 0, &[4], None))
            .expect("enqueue victim");
        let mut step = MockStep::default();
        assert!(matches!(
            scheduler.drive_once(&mut step).expect("step"),
            SchedulerAction::Stepped { .. }
        ));
        let victim = scheduler.preempt_running(2).expect("preempt victim");
        assert!(scheduler.resume_may_wait(&victim));
        scheduler
            .enqueue_turn(turn(3, 0, &[4], None))
            .expect("enqueue later waiter");
        scheduler.take_waiting(1).expect("earlier waiter leaves");
        assert!(!scheduler.resume_may_wait(&victim));
        assert!(!scheduler.has_preempted());
        scheduler.prepend_preempted(victim);
        assert!(scheduler.has_preempted());
    }

    /// A blocked step beside another running turn preempts the newest one.
    /// Mutation: `reinstate_running` keeps the preemption count → red.
    /// Mutation: report every victim as not blocked → red.
    /// Mutation: treat every blocked step as alone → red (no victim).
    #[test]
    fn reinstated_victim_runs_as_if_never_preempted() {
        let mut scheduler = Scheduler::<_, (), ()>::new(2, 8).expect("scheduler");
        scheduler
            .enqueue_turn(turn(1, 0, &[4], None))
            .expect("enqueue peer");
        scheduler
            .enqueue_turn(turn(2, 0, &[4], None).with_block_reservation(2, 0, 4))
            .expect("enqueue");
        let mut step = MockStep {
            allocation_blocked: HashSet::from([2]),
            ..MockStep::default()
        };
        let SchedulerAction::Stepped {
            preempted: Some(preempted),
            blocked_alone: None,
            ..
        } = scheduler.drive_once(&mut step).expect("blocked step")
        else {
            panic!("expected a preempted victim");
        };
        let PreemptedTurn {
            turn: victim,
            blocked,
        } = *preempted;
        assert_eq!(victim.seq_id, 2);
        assert!(blocked, "the victim blocked its own step");
        assert_eq!(scheduler.stats().preemptions, 1);
        scheduler.reinstate_running(victim);
        assert_eq!(scheduler.stats().preemptions, 0);
        let turn = &scheduler.running()[1];
        assert_eq!(turn.seq_id, 2);
        assert_eq!(turn.status, TurnStatus::Running);
        assert_eq!(turn.preemptions, 0);
        assert!(turn.block_reservation_active && turn.recurrent_state_reservation_active);
        assert_eq!(turn.unallocated_reserved_blocks(), 2);
        step.allocation_blocked.clear();
        assert!(matches!(
            scheduler.drive_once(&mut step).expect("retried step"),
            SchedulerAction::Stepped {
                preempted: None,
                blocked_alone: None,
                ..
            }
        ));
    }

    /// The only running turn blocked: no turn holds memory that could come
    /// back, so it is not preempted; it stays running and the step names it.
    /// Mutation: preempt the lone turn as before → red (a victim, 1
    /// preemption).
    #[test]
    fn lone_blocked_turn_is_reported_not_preempted() {
        let mut scheduler = Scheduler::<_, (), ()>::new(2, 8).expect("scheduler");
        scheduler
            .enqueue_turn(turn(1, 0, &[4], None).with_block_reservation(2, 0, 4))
            .expect("enqueue");
        let mut step = MockStep {
            allocation_blocked: HashSet::from([1]),
            ..MockStep::default()
        };
        let SchedulerAction::Stepped {
            preempted,
            blocked_alone,
            ..
        } = scheduler.drive_once(&mut step).expect("blocked step")
        else {
            panic!("expected a step");
        };
        assert!(preempted.is_none(), "a lone turn has no one to yield to");
        assert_eq!(blocked_alone, Some(1));
        assert_eq!(scheduler.stats().preemptions, 0);
        assert_eq!(scheduler.running().len(), 1);
        assert_eq!(scheduler.running()[0].status, TurnStatus::Running);
        assert!(scheduler.take_running(1).is_some());
        assert!(scheduler.running().is_empty());
        assert_eq!(scheduler.stats().preemptions, 0);
    }

    /// A lone blocked turn beside a turn parked for SSD yields: the restore
    /// holds blocks that come back when that turn ends, so it is not alone.
    /// The victim waits on the SSD lane, and the lane, once woken, runs ahead.
    /// Mutation M1: drop `ssd_lane` from drive_once's `another_lane` → red.
    /// Mutation M2: drop `turn.ssd_lane ||` from resume_may_wait → red.
    /// Mutation M3: drop `turn.ssd_lane ||` from admit_waiting → red.
    #[test]
    fn lone_blocked_turn_beside_an_ssd_lane_yields() {
        let mut scheduler = Scheduler::<_, (), ()>::new(2, 8).expect("scheduler");
        scheduler
            .enqueue_turn(turn(1, 0, &[4], None).with_block_reservation(2, 0, 4))
            .expect("enqueue running turn");
        let mut step = MockStep::default();
        scheduler.drive_once(&mut step).expect("admit turn 1");
        scheduler
            .enqueue_turn(turn(2, 0, &[4], None).with_block_reservation(4, 4, 4))
            .expect("enqueue SSD turn");
        scheduler.park_waiting_for_ssd(2).expect("park SSD turn");
        step.allocation_blocked.insert(1);
        let SchedulerAction::Stepped {
            preempted: Some(preempted),
            blocked_alone: None,
            ..
        } = scheduler.drive_once(&mut step).expect("blocked step")
        else {
            panic!("a turn beside an SSD lane must yield, not fail alone");
        };
        let PreemptedTurn {
            turn: victim,
            blocked,
        } = *preempted;
        assert_eq!(victim.seq_id, 1);
        assert!(blocked);
        assert!(!victim.ssd_lane);
        assert!(
            scheduler.resume_may_wait(&victim),
            "the victim waits on the SSD lane"
        );
        scheduler.prepend_preempted(victim);
        scheduler
            .wake_from_ssd(2, 0, Duration::from_millis(1))
            .expect("wake");
        step.allocation_blocked.clear();
        let SchedulerAction::Stepped { plan, .. } = scheduler.drive_once(&mut step).expect("step")
        else {
            panic!("the woken SSD lane must run ahead of the victim");
        };
        assert_eq!(
            plan.rows.iter().map(|row| row.seq_id).collect::<Vec<_>>(),
            vec![2]
        );
    }

    #[test]
    fn ssd_wait_does_not_create_a_phase_queue_or_block_later_runnable_work() {
        let mut scheduler = Scheduler::<_, (), ()>::new(1, 4).expect("scheduler");
        scheduler
            .enqueue_turn(turn(1, 0, &[4], None))
            .expect("enqueue SSD row");
        scheduler
            .enqueue_turn(turn(2, 0, &[4], None))
            .expect("enqueue runnable row");
        scheduler.park_waiting_for_ssd(1).expect("park SSD row");
        let mut step = MockStep::default();
        let SchedulerAction::Stepped { plan, .. } =
            scheduler.drive_once(&mut step).expect("runnable step")
        else {
            panic!("expected runnable step");
        };
        assert_eq!(plan.rows[0].seq_id, 2);
        scheduler
            .wake_from_ssd(1, 4096, Duration::from_millis(3))
            .expect("wake SSD row");
        assert_eq!(scheduler.stats().ssd_restore_waiting, 0);
        assert_eq!(scheduler.stats().ssd_restore_bytes, 4096);
        assert!(scheduler.stats().ssd_restore_wait_micros >= 3_000);
    }

    #[test]
    fn control_barrier_cannot_overtake_an_older_ssd_wait() {
        let mut scheduler = Scheduler::<_, (), &'static str>::new(1, 4).expect("scheduler");
        scheduler
            .enqueue_turn(turn(1, 0, &[4], None))
            .expect("enqueue SSD row");
        scheduler.park_waiting_for_ssd(1).expect("park SSD row");
        scheduler.enqueue_barrier("reset");

        let mut step = MockStep::default();
        assert!(matches!(
            scheduler.drive_once(&mut step).expect("idle behind SSD"),
            SchedulerAction::Idle
        ));
        scheduler
            .wake_from_ssd(1, 0, Duration::from_millis(1))
            .expect("wake SSD row");
        assert!(matches!(
            scheduler
                .drive_once(&mut step)
                .expect("older turn runs first"),
            SchedulerAction::Stepped { .. }
        ));
    }

    /// Mutation: report every victim as blocked → red.
    #[test]
    fn allocation_squeeze_preempts_exactly_one_lifo_victim_without_same_step_readmit() {
        let mut scheduler = Scheduler::<_, (), ()>::new(3, 3).expect("scheduler");
        for seq_id in [1, 2, 3] {
            let mut request = turn(seq_id, 4, &[4], None).with_block_reservation(8, 4, 1);
            request.token_history.push(9);
            request.num_tokens += 1;
            scheduler.enqueue_turn(request).expect("enqueue");
        }
        let mut step = MockStep::default();
        step.allocation_blocked.insert(1);
        let SchedulerAction::Stepped {
            plan, preempted, ..
        } = scheduler.drive_once(&mut step).expect("squeeze step")
        else {
            panic!("expected step");
        };
        assert_eq!(
            plan.rows.iter().map(|row| row.seq_id).collect::<Vec<_>>(),
            [1, 2, 3]
        );
        let PreemptedTurn {
            turn: victim,
            blocked,
        } = *preempted.expect("one victim");
        assert_eq!(victim.seq_id, 3, "newest running row is the LIFO victim");
        assert!(!blocked, "row 1 blocked, not the victim");
        assert_eq!(victim.status, TurnStatus::Preempted);
        assert!(!victim.block_reservation_active);
        assert_eq!(victim.preemptions, 1);
        assert_eq!(
            scheduler
                .running()
                .iter()
                .map(|turn| turn.seq_id)
                .collect::<Vec<_>>(),
            [1, 2]
        );
        assert_eq!(
            scheduler.waiting_len(),
            0,
            "victim is returned to the model before requeue"
        );
        assert_eq!(scheduler.stats().preemptions, 1);

        scheduler.prepend_preempted(victim);
        step.allocation_blocked.clear();
        let SchedulerAction::Stepped { plan, .. } =
            scheduler.drive_once(&mut step).expect("survivor retry")
        else {
            panic!("expected survivor step");
        };
        assert_eq!(
            plan.rows.iter().map(|row| row.seq_id).collect::<Vec<_>>(),
            [1, 2],
            "preempted row cannot re-enter until model-specific resume preparation"
        );
        let mut victim = scheduler.take_preempted().expect("preempted victim");
        victim.num_computed_tokens = 4; // deepest verified hot-prefix hit
        victim.block_materialized_blocks = 4;
        scheduler.ready_preempted(victim, false);
        assert_eq!(scheduler.waiting_len(), 1);
    }

    #[test]
    fn ssd_escalation_requires_a_long_prefix_and_measured_cost_win() {
        let fast_restore = (64 * 1024 * 1024, 20_000);
        let slow_prefill = (4096, 400_000);
        assert_eq!(
            select_preemption_mode(
                32 * 16,
                16,
                64 * 1024,
                slow_prefill.0,
                slow_prefill.1,
                fast_restore.0,
                fast_restore.1,
            ),
            PreemptionMode::Recompute,
            "short prefixes do not pay SSD fixed/capture overhead"
        );
        assert_eq!(
            select_preemption_mode(
                128 * 16,
                16,
                64 * 1024,
                slow_prefill.0,
                slow_prefill.1,
                fast_restore.0,
                fast_restore.1,
            ),
            PreemptionMode::Ssd,
            "a long prefix escalates when measured restore throughput wins"
        );
        assert_eq!(
            select_preemption_mode(128 * 16, 16, 64 * 1024, 0, 0, 0, 0),
            PreemptionMode::Recompute,
            "missing measurements fail closed to recompute"
        );
        assert_eq!(
            select_preemption_mode(
                128 * 16,
                16,
                64 * 1024,
                4096,
                40_000,
                64 * 1024 * 1024,
                800_000,
            ),
            PreemptionMode::Recompute,
            "the cost branch rejects a measured SSD loss"
        );
    }

    #[test]
    fn prefill_preemption_replays_the_original_prompt_and_still_samples() {
        let mut request = turn(9, 2, &[2, 4], None).with_block_reservation(8, 2, 1);
        let replay =
            install_preemption_replay(&mut request, &[1, 2, 3, 4], 3).expect("prefill replay");
        assert_eq!(replay.tokens, [1, 2, 3, 4]);
        assert_eq!(replay.cached_prefix, 0);
        assert!(!replay.suppress_sample);
        assert_eq!(request.num_computed_tokens, 0);
        assert_eq!(request.num_tokens, 4);
        assert_eq!(request.prompt_tokens, 4);
        assert_eq!(request.pinned_prefill_breaks, [3, 4]);
        assert_eq!(request.block_materialized_blocks, 0);
        assert!(!request.block_reservation_active);
    }

    #[test]
    fn decode_preemption_replays_only_computed_history_and_keeps_pending_token() {
        let mut request = turn(10, 4, &[4], None).with_block_reservation(8, 4, 1);
        request.token_history.push(99);
        request.num_tokens += 1;
        let replay =
            install_preemption_replay(&mut request, &[1, 2, 3, 4], 2).expect("decode replay");
        assert_eq!(replay.tokens, [1, 2, 3, 4]);
        assert!(replay.suppress_sample);
        assert_eq!(request.token_history, [1, 2, 3, 4, 99]);
        assert_eq!(request.num_computed_tokens, 0);
        assert_eq!(request.num_tokens, 5);
        assert_eq!(request.prompt_tokens, 4);
        assert_eq!(request.pinned_prefill_breaks, [2, 4]);
    }

    #[test]
    fn hybrid_memory_admission_charges_blocks_and_state_to_one_budget() {
        const MIB: u64 = 1024 * 1024;
        let telemetry = MemoryTelemetry {
            capacity_bytes: 1024 * MIB,
            free_bytes: 800 * MIB,
            reclaimable_bytes: 0,
        };
        let admitted =
            memory_admission_decision(telemetry, 2, 1, 16 * MIB, 335 * MIB, 335 * MIB, true, 0.05);
        assert!(admitted.admitted);
        assert_eq!(admitted.reserved_block_bytes, 48 * MIB);
        assert_eq!(admitted.reserved_state_bytes, 670 * MIB);
        assert_eq!(
            admitted.watermark_bytes,
            ((1024 * MIB) as f64 * 0.05).ceil() as u64
        );

        let blocks_only_would_fit_but_state_does_not =
            memory_admission_decision(telemetry, 2, 1, 16 * MIB, 670 * MIB, 335 * MIB, true, 0.05);
        assert!(!blocks_only_would_fit_but_state_does_not.admitted);
    }

    #[test]
    fn resident_recurrent_table_bytes_survive_an_empty_scheduler_queue() {
        const MIB: u64 = 1024 * 1024;
        let telemetry = MemoryTelemetry {
            capacity_bytes: 1024 * MIB,
            free_bytes: 800 * MIB,
            reclaimable_bytes: 0,
        };
        let mut scheduler = Scheduler::<(), (), ()>::new(2, 2).expect("scheduler");
        let decision = scheduler.try_reserve_memory(telemetry, 0, 0, 335 * MIB, 335 * MIB, 0.05);
        assert!(decision.admitted);
        assert_eq!(decision.reserved_state_bytes, 670 * MIB);
        assert_eq!(scheduler.stats().reserved_state_bytes, 670 * MIB);
    }

    #[test]
    fn unified_memory_admission_preserves_block_telemetry() {
        let telemetry = MemoryTelemetry {
            capacity_bytes: 100 * 4096,
            free_bytes: 70 * 4096,
            reclaimable_bytes: 5 * 4096,
        };
        let mut scheduler = Scheduler::<(), (), ()>::new(2, 2).expect("scheduler");
        let decision = scheduler.try_reserve_memory(telemetry, 3, 4096, 0, 0, 0.05);

        assert!(decision.admitted);
        assert_eq!(scheduler.stats().block_capacity, 100);
        assert_eq!(scheduler.stats().free_blocks, 70);
        assert_eq!(scheduler.stats().reclaimable_blocks, 5);
        assert_eq!(scheduler.stats().allocated_blocks, 30);
        assert_eq!(scheduler.stats().watermark_blocks, 0);
        assert_eq!(scheduler.stats().reserved_blocks, 3);
    }

    #[test]
    fn hybrid_observation_replaces_stale_candidate_reservation() {
        let blocks = BlockTelemetry {
            total_blocks: 100,
            free_blocks: 70,
            reclaimable_blocks: 5,
            allocated_blocks: 30,
        };
        let mut scheduler = Scheduler::<(), (), ()>::new(2, 2).expect("scheduler");
        assert!(
            scheduler
                .try_reserve_memory(
                    MemoryTelemetry {
                        capacity_bytes: 100 * 4096,
                        free_bytes: 70 * 4096,
                        reclaimable_bytes: 5 * 4096,
                    },
                    3,
                    4096,
                    0,
                    0,
                    0.05,
                )
                .admitted
        );
        assert_eq!(scheduler.stats().reserved_blocks, 3);

        let observed = scheduler.observe_hybrid_memory(blocks, 4096, 0, 0.05);
        assert!(observed.admitted);
        assert_eq!(scheduler.stats().reserved_blocks, 0);
        assert_eq!(scheduler.stats().allocated_blocks, 30);
    }

    #[test]
    fn eight_gemma_state_candidates_admit_two_and_defer_six() {
        const MIB: u64 = 1024 * 1024;
        const GEMMA_STATE: u64 = 335 * MIB;
        let telemetry = MemoryTelemetry {
            capacity_bytes: 1024 * MIB,
            free_bytes: 760 * MIB,
            reclaimable_bytes: 0,
        };
        let mut scheduler = Scheduler::<_, (), ()>::new(8, 8).expect("scheduler");
        let mut admitted = Vec::new();
        let mut deferred = Vec::new();
        for seq_id in 1..=8 {
            let decision = scheduler.try_reserve_memory(telemetry, 0, 0, 0, GEMMA_STATE, 0.05);
            if decision.admitted {
                scheduler
                    .enqueue_turn(
                        turn(seq_id, 1, &[1], None).with_recurrent_state_reservation(GEMMA_STATE),
                    )
                    .unwrap();
                admitted.push(seq_id);
            } else {
                deferred.push(seq_id);
            }
        }
        assert_eq!(admitted, [1, 2]);
        assert_eq!(deferred, [3, 4, 5, 6, 7, 8]);
        assert_eq!(scheduler.waiting_len(), 2);
        assert_eq!(scheduler.stats().admission_deferred_state, 6);
        assert_eq!(scheduler.stats().reserved_state_bytes, 3 * GEMMA_STATE);
    }

    /// A whole-prompt prefill row (a cold Flash Next turn in group mode)
    /// ignores the token budget, is its step's only prefill row, and waits
    /// while another turn's prefill row is planned ahead of it; every other
    /// turn keeps the 1024-slice grid under the shared budget.
    /// MUTATION: `build_plan` subtracts the whole row from the budget ->
    /// u32 underflow (debug panic) on the 6219-token row.
    #[test]
    fn a_whole_prompt_row_ignores_the_budget_and_is_the_steps_only_prefill_row() {
        let running = |seq_id: SeqId, prompt: usize, breaks: Vec<u32>, whole: bool| {
            let mut turn =
                TurnState::new(seq_id, vec![1; prompt], 0, breaks, None, "payload").expect("turn");
            turn.status = TurnStatus::Running;
            turn.whole_prompt_prefill = whole;
            turn
        };
        let rows = |plan: &StepPlan| {
            plan.rows
                .iter()
                .map(|row| (row.seq_id, row.kind, row.num_tokens))
                .collect::<Vec<_>>()
        };
        // A decode peer runs its row first; the cold turn's whole prompt
        // follows in one row, past the 2048 budget.
        let mut scheduler = Scheduler::<_, (), ()>::new(4, 2048).unwrap();
        let mut peer = running(1, 4, vec![4], false);
        peer.num_computed_tokens = 4;
        peer.token_history.push(9);
        peer.num_tokens += 1;
        scheduler.running.push(peer);
        scheduler.running.push(running(2, 6219, vec![6219], true));
        assert_eq!(
            rows(&scheduler.build_plan(None)),
            [(1, StepKind::Decode, 1), (2, StepKind::Prefill, 6219)]
        );
        // Two warm turns: 1024 each, 2048 per step.
        let mut scheduler = Scheduler::<_, (), ()>::new(4, 2048).unwrap();
        scheduler
            .running
            .push(running(3, 3000, vec![1024, 2048, 3000], false));
        scheduler
            .running
            .push(running(4, 3000, vec![1024, 2048, 3000], false));
        assert_eq!(
            rows(&scheduler.build_plan(None)),
            [(3, StepKind::Prefill, 1024), (4, StepKind::Prefill, 1024)]
        );
        // A warm prefill row ahead of the cold turn: the cold turn waits.
        let mut scheduler = Scheduler::<_, (), ()>::new(4, 2048).unwrap();
        scheduler
            .running
            .push(running(5, 3000, vec![1024, 2048, 3000], false));
        scheduler.running.push(running(6, 6219, vec![6219], true));
        assert_eq!(
            rows(&scheduler.build_plan(None)),
            [(5, StepKind::Prefill, 1024)]
        );
        // The cold turn ahead: no other prefill row joins its step.
        let mut scheduler = Scheduler::<_, (), ()>::new(4, 2048).unwrap();
        scheduler.running.push(running(7, 6219, vec![6219], true));
        scheduler
            .running
            .push(running(8, 3000, vec![1024, 2048, 3000], false));
        assert_eq!(
            rows(&scheduler.build_plan(None)),
            [(7, StepKind::Prefill, 6219)]
        );
    }

    /// Splash `Engine::laneToYield`: the running lane of the lowest rank
    /// yields; a prefilling lane before a decoding one, then fewer
    /// completed tokens, then the newest.
    /// Mutation Y1: the newest running turn (`rposition`) → red (3, the
    /// newest, decoding with 900 completed, instead of 2).
    #[test]
    fn yield_order_follows_rank() {
        // (seq, prompt, computed, arrival)
        let lanes = |turns: &[(SeqId, u32, u32, u64)]| {
            let mut scheduler = Scheduler::<_, (), ()>::new(4, 64).expect("scheduler");
            for &(seq, prompt, computed, arrival) in turns {
                let mut turn = TurnState::new(
                    seq,
                    vec![1; prompt as usize],
                    0,
                    Vec::new(),
                    None,
                    "payload",
                )
                .expect("turn");
                turn.num_computed_tokens = computed;
                turn.num_tokens = computed.max(prompt) + u32::from(computed >= prompt);
                turn.arrival_order = arrival;
                turn.status = TurnStatus::Running;
                scheduler.running.push(turn);
            }
            scheduler.lane_to_yield()
        };
        assert_eq!(
            lanes(&[(1, 800, 900, 0), (2, 1000, 300, 1), (3, 1000, 100, 2)]),
            Some(3)
        );
        assert_eq!(
            lanes(&[(1, 100, 500, 0), (2, 100, 200, 1), (3, 100, 900, 2)]),
            Some(2)
        );
        assert_eq!(lanes(&[(1, 1000, 100, 0), (2, 1000, 100, 1)]), Some(2));
    }

    type Measured = Scheduler<&'static str, (), ()>;

    fn measured_scheduler(native_window: u32, decode_share: f64) -> Measured {
        let scheduler = Scheduler::new(10, PREFILL_CHUNK_ROWS)
            .expect("scheduler")
            .with_measured_prefill_cut(MeasuredConfig {
                native_window,
                decode_share,
            });
        assert!(scheduler.measures_prefill_cut());
        scheduler
    }

    /// A running turn with an `n`-token prompt, `computed` of it done; its
    /// arrival is the push order.
    fn push_prefill(scheduler: &mut Measured, seq: SeqId, n: u32, computed: u32) {
        let mut turn = TurnState::new(seq, vec![1; n as usize], computed, Vec::new(), None, "p")
            .expect("turn");
        turn.status = TurnStatus::Running;
        turn.arrival_order = scheduler.take_order();
        scheduler.running.push(turn);
    }

    /// A running turn decoding past its 4-token prompt.
    fn push_decoder(scheduler: &mut Measured, seq: SeqId) {
        push_prefill(scheduler, seq, 4, 4);
        let turn = scheduler.running.last_mut().expect("decoder");
        turn.token_history.push(9);
        turn.num_tokens += 1;
    }

    fn running_mut(scheduler: &mut Measured, seq: SeqId) -> &mut TurnState<&'static str> {
        scheduler.running_turn_mut(seq).expect("running turn")
    }

    fn measured(scheduler: &Measured) -> Option<MeasuredUnit> {
        match scheduler.prefill {
            PrefillUnit::Measured { measured, .. } => measured,
            PrefillUnit::Fixed { .. } => None,
        }
    }

    fn decode_debt(scheduler: &Measured) -> f64 {
        match scheduler.prefill {
            PrefillUnit::Measured { decode_debt_ms, .. } => decode_debt_ms,
            PrefillUnit::Fixed { .. } => 0.0,
        }
    }

    /// Each row as (seq, start, end).
    fn spans(plan: &StepPlan) -> Vec<(SeqId, u32, u32)> {
        plan.rows
            .iter()
            .map(|row| {
                (
                    row.seq_id,
                    row.token_start,
                    row.token_start + row.num_tokens,
                )
            })
            .collect()
    }

    /// Prefill rows take `num_tokens × micros_per_row` (none for `untimed`
    /// lanes) and sample at their prompt's end; decode rows sample one
    /// token in `decode_micros` together; a cancelled row fails as the
    /// hybrid executor fails it.
    #[derive(Default)]
    struct TimedStep {
        micros_per_row: u64,
        decode_micros: u64,
        untimed: HashSet<SeqId>,
        /// Decoders whose next row ends their turn.
        finishing: HashSet<SeqId>,
    }

    impl StepExecutor<&'static str> for TimedStep {
        type Error = String;

        fn execute(
            &mut self,
            plan: &StepPlan,
            running: &mut [TurnState<&'static str>],
        ) -> Result<StepResult, Self::Error> {
            let mut prefill_wall_micros = 0;
            let mut decode_rows = 0;
            let rows = plan
                .rows
                .iter()
                .map(|row| {
                    let mut result = RowStepResult {
                        seq_id: row.seq_id,
                        num_computed_tokens: row.num_tokens,
                        generated_token: None,
                        speculative: None,
                        finished: row.cancel_snapshot,
                        allocation_blocked: false,
                        prefill_micros: 0,
                    };
                    if row.cancel_snapshot {
                        return result;
                    }
                    match row.kind {
                        StepKind::Decode => {
                            decode_rows += 1;
                            result.generated_token = Some(9);
                            result.finished = self.finishing.contains(&row.seq_id);
                        }
                        StepKind::Prefill => {
                            let turn = running
                                .iter()
                                .find(|turn| turn.seq_id == row.seq_id)
                                .expect("planned turn");
                            if row.token_start + row.num_tokens == turn.prompt_tokens {
                                result.generated_token = Some(9);
                            }
                            if !self.untimed.contains(&row.seq_id) {
                                result.prefill_micros =
                                    u64::from(row.num_tokens) * self.micros_per_row;
                            }
                            prefill_wall_micros += result.prefill_micros;
                        }
                    }
                    result
                })
                .collect();
            Ok(StepResult {
                rows,
                executed_decode_batch: decode_rows,
                executed_greedy_epilogue_batch: 0,
                rows_alloc_evicted: 0,
                prefill_wall_micros,
                decode_wall_micros: if decode_rows > 0 {
                    self.decode_micros
                } else {
                    0
                },
            })
        }
    }

    fn timed(micros_per_row: u64) -> TimedStep {
        TimedStep {
            micros_per_row,
            ..TimedStep::default()
        }
    }

    fn step(scheduler: &mut Measured, executor: &mut TimedStep) -> StepPlan {
        let SchedulerAction::Stepped { plan, .. } = scheduler.drive_once(executor).expect("step")
        else {
            panic!("expected a step");
        };
        plan
    }

    /// Splash `cutRows`: a unit that took 2048 ms for 2048 rows sizes the
    /// next contended one to 166 rows deeper in the prompt, so the floor.
    /// Mutation Q1: the measured unit is always a chunk → red (P 2048 rows).
    #[test]
    fn a_slow_unit_cuts_the_next_to_the_floor() {
        let mut scheduler = measured_scheduler(262_144, DEFAULT_DECODE_SHARE);
        push_prefill(&mut scheduler, 1, 40_000, 0);
        let mut executor = timed(1000);
        let plan = step(&mut scheduler, &mut executor);
        assert_eq!(spans(&plan), vec![(1, 0, 2048)]);
        assert!(!plan.contended);
        assert_eq!(
            measured(&scheduler),
            Some(MeasuredUnit {
                rows: 2048,
                pairs: 2_098_176,
                millis: 2048.0,
            })
        );
        push_decoder(&mut scheduler, 2);
        let plan = step(&mut scheduler, &mut executor);
        assert_eq!(spans(&plan), vec![(2, 4, 5), (1, 2048, 2560)]);
        assert!(plan.contended);
        assert_eq!(plan.leader_cut, Some(1));
    }

    /// 2048 rows in 204.8 ms run 5000 in G; at depth 2048 a row reaches
    /// 2,098,176 / 6,292,480 of the pairs: 1667 rows.
    /// Mutation Q3: `deeper` = 1 → red (5000 → the 2048 ceiling).
    #[test]
    fn the_cut_follows_the_measured_rate_and_depth() {
        let mut scheduler = measured_scheduler(262_144, DEFAULT_DECODE_SHARE);
        push_prefill(&mut scheduler, 1, 40_000, 0);
        let mut executor = timed(100);
        step(&mut scheduler, &mut executor);
        push_decoder(&mut scheduler, 2);
        let plan = step(&mut scheduler, &mut executor);
        assert_eq!(spans(&plan), vec![(2, 4, 5), (1, 2048, 3715)]);
        assert_eq!(plan.leader_cut, Some(1));
    }

    /// A rest of 388 rows behind a 512-row cut runs with it.
    /// Mutation Q2: no rest rule → red (512 rows).
    #[test]
    fn a_rest_below_the_floor_runs_with_the_cut() {
        let mut scheduler = measured_scheduler(262_144, DEFAULT_DECODE_SHARE);
        push_prefill(&mut scheduler, 1, 2948, 0);
        let mut executor = timed(1000);
        step(&mut scheduler, &mut executor);
        push_decoder(&mut scheduler, 2);
        let plan = step(&mut scheduler, &mut executor);
        assert_eq!(spans(&plan), vec![(2, 4, 5), (1, 2048, 2948)]);
        assert_eq!(plan.leader_cut, None);
    }

    /// A unit with an untimed row sizes nothing: the contended unit after
    /// it is the floor.
    /// Mutation Q4: untimed rows count as timed → red (measured from
    /// 1388 rows in 30 ms: 2048 rows).
    #[test]
    fn only_timed_units_size_the_cut() {
        let mut scheduler = measured_scheduler(262_144, 0.0);
        push_prefill(&mut scheduler, 1, 300, 0);
        push_prefill(&mut scheduler, 2, 40_000, 0);
        let mut executor = timed(100);
        executor.untimed.insert(2);
        let plan = step(&mut scheduler, &mut executor);
        assert_eq!(spans(&plan), vec![(1, 0, 300), (2, 0, 1088)]);
        assert_eq!(measured(&scheduler), None);
        executor.untimed.clear();
        let plan = step(&mut scheduler, &mut executor);
        assert!(plan.contended);
        assert_eq!(spans(&plan), vec![(1, 300, 301), (2, 1088, 1600)]);
    }

    /// Splash `planPrefill` joiners: the fewest rows that spare a lane one
    /// of its own units, skipped when they do not fit.
    /// Mutation J1: a joiner takes what is left of the unit → red (C joins
    /// with 1748).
    #[test]
    fn uncontended_joiners_take_the_rows_that_spare_a_chunk() {
        let mut scheduler = measured_scheduler(262_144, DEFAULT_DECODE_SHARE);
        push_prefill(&mut scheduler, 1, 300, 0);
        push_prefill(&mut scheduler, 2, 5000, 0);
        push_prefill(&mut scheduler, 3, 4096, 0);
        let plan = scheduler.build_plan(None);
        assert_eq!(spans(&plan), vec![(1, 0, 300), (2, 0, 904)]);
        assert!(!plan.contended);
    }

    /// Decode rows and their draft rows are their own work, outside the
    /// prefill unit.
    /// Mutation B1: the decode rows are charged to the unit → red (P 497).
    #[test]
    fn decode_rows_sit_outside_the_prefill_unit() {
        let mut scheduler = measured_scheduler(262_144, DEFAULT_DECODE_SHARE);
        for seq in 1..=3 {
            push_decoder(&mut scheduler, seq);
            running_mut(&mut scheduler, seq).decode_draft_allowance = 4;
        }
        push_prefill(&mut scheduler, 4, 40_000, 0);
        let plan = scheduler.build_plan(None);
        assert_eq!(
            spans(&plan),
            vec![(1, 4, 9), (2, 4, 9), (3, 4, 9), (4, 0, 512)]
        );
    }

    /// Splash `leadsBefore`: the fewest remaining rows lead.
    /// Mutation L1: FCFS leader → red (A leads with 512).
    #[test]
    fn the_fewest_remaining_lead_beside_a_decoder() {
        let mut scheduler = measured_scheduler(262_144, DEFAULT_DECODE_SHARE);
        push_decoder(&mut scheduler, 1);
        push_prefill(&mut scheduler, 2, 40_000, 0);
        push_prefill(&mut scheduler, 3, 300, 0);
        let plan = scheduler.build_plan(None);
        assert_eq!(spans(&plan), vec![(1, 4, 5), (3, 0, 300)]);
        assert_eq!(plan.leader_cut, None);
    }

    /// Splash `commit` overtakes: a lane passed over by three units that
    /// served later arrivals leads the fourth.
    /// Mutation O1: no overtake count → red (F leads the fourth unit).
    #[test]
    fn an_overtaken_lane_leads_after_three() {
        let mut scheduler = measured_scheduler(262_144, 0.0);
        push_decoder(&mut scheduler, 1);
        push_prefill(&mut scheduler, 2, 40_000, 0);
        for seq in 3..=6 {
            push_prefill(&mut scheduler, seq, 300, 0);
        }
        let mut executor = timed(1000);
        for (served, overtaken) in [(3, 1), (4, 2), (5, 3)] {
            let plan = step(&mut scheduler, &mut executor);
            let prefill: Vec<_> = plan
                .rows
                .iter()
                .filter(|row| row.kind == StepKind::Prefill)
                .map(|row| (row.seq_id, row.token_start, row.num_tokens))
                .collect();
            assert_eq!(prefill, vec![(served, 0, 300)]);
            assert_eq!(running_mut(&mut scheduler, 2).overtaken, overtaken);
        }
        let plan = step(&mut scheduler, &mut executor);
        let prefill: Vec<_> = plan
            .rows
            .iter()
            .filter(|row| row.kind == StepKind::Prefill)
            .map(|row| (row.seq_id, row.token_start, row.num_tokens))
            .collect();
        assert_eq!(prefill, vec![(2, 0, 512)]);
        assert_eq!(running_mut(&mut scheduler, 2).overtaken, 0);
    }

    /// Splash `maximumBatchWidth`: a unit holds at most four lanes.
    /// Mutation W1: no width break → red (six rows).
    #[test]
    fn a_unit_holds_at_most_four_lanes() {
        let mut scheduler = measured_scheduler(262_144, DEFAULT_DECODE_SHARE);
        for seq in 1..=6 {
            push_prefill(&mut scheduler, seq, 100, 0);
        }
        let plan = scheduler.build_plan(None);
        assert_eq!(
            spans(&plan),
            vec![(1, 0, 100), (2, 0, 100), (3, 0, 100), (4, 0, 100)]
        );
    }

    /// Splash `withinPairs`: at depth 16,384 of an 8192 native window the
    /// leader halves to 512 rows; the halving is not a cut.
    /// Mutation P1: no halving → red (2048 rows).
    #[test]
    fn a_deep_leader_halves_under_the_pair_bound() {
        let mut scheduler = measured_scheduler(8192, DEFAULT_DECODE_SHARE);
        push_prefill(&mut scheduler, 1, 40_000, 16_384);
        let plan = scheduler.build_plan(None);
        assert_eq!(spans(&plan), vec![(1, 16_384, 16_896)]);
        assert_eq!(plan.leader_cut, None);
    }

    /// Splash `readyLanes` never plans a cancelled lane: it neither leads
    /// nor sizes the unit, and fails in a row of its own.
    /// Mutation C1: cancelled turns stay ready lanes → red (A leads).
    #[test]
    fn a_cancelled_prefill_never_leads_or_sizes() {
        let mut scheduler = measured_scheduler(262_144, DEFAULT_DECODE_SHARE);
        push_decoder(&mut scheduler, 1);
        push_prefill(&mut scheduler, 2, 300, 0);
        running_mut(&mut scheduler, 2).cancelled = Some(Arc::new(AtomicBool::new(true)));
        push_prefill(&mut scheduler, 3, 40_000, 0);
        let expected = vec![(1, 4, 5), (3, 0, 512), (2, 0, 1)];
        assert_eq!(spans(&scheduler.build_plan(None)), expected);
        let plan = step(&mut scheduler, &mut timed(1000));
        assert_eq!(spans(&plan), expected);
        assert_eq!(plan.leader_cut, Some(3));
        assert_eq!(
            measured(&scheduler),
            Some(MeasuredUnit {
                rows: 512,
                pairs: 131_328,
                millis: 512.0,
            })
        );
    }

    /// Splash `PlanInputs::excluded`: a waiter whose retry is not due is
    /// planned without; a waiting decoder still contends.
    /// Mutation E1: the measured plan ignores the retry → red (A leads).
    #[test]
    fn a_waiting_lane_is_left_out_of_the_unit() {
        let mut scheduler = measured_scheduler(262_144, DEFAULT_DECODE_SHARE);
        push_decoder(&mut scheduler, 1);
        push_prefill(&mut scheduler, 2, 300, 0);
        push_prefill(&mut scheduler, 3, 40_000, 0);
        running_mut(&mut scheduler, 2).resource_wait.retry_at = 1000.0;
        let clock = Some(RetryClock { now: 0.0, epoch: 0 });
        let plan = scheduler.build_plan(clock);
        assert_eq!(spans(&plan), vec![(1, 4, 5), (3, 0, 512)]);
        assert!(plan.contended);
        running_mut(&mut scheduler, 1).resource_wait.retry_at = 1000.0;
        let plan = scheduler.build_plan(clock);
        assert_eq!(spans(&plan), vec![(3, 0, 512)]);
        assert!(plan.contended);
    }

    /// Splash `complete`: every unit counts, a cut counts once it ran, and
    /// only the units a lane decoded beside are in the contended histogram;
    /// a step whose prefill row was refused ran no unit.
    /// Mutation C2: every unit goes into the histogram → red (2 samples).
    #[test]
    fn contended_units_are_counted_apart() {
        let mut scheduler = measured_scheduler(262_144, 0.0);
        let counters = scheduler.counters();
        push_prefill(&mut scheduler, 1, 40_000, 0);
        let mut executor = timed(1000);
        step(&mut scheduler, &mut executor);
        push_decoder(&mut scheduler, 2);
        step(&mut scheduler, &mut executor);
        let check = |snapshot: SchedulerCounterSnapshot| {
            assert_eq!(snapshot.prefill_units, 2);
            assert_eq!(snapshot.prefill_rows, 2048 + 512);
            assert_eq!(snapshot.bound_cuts, 1);
            assert_eq!(snapshot.contended_prefill.samples, 1);
            assert_eq!(snapshot.contended_prefill.max_ms, 512.0);
        };
        check(counters.snapshot());
        let Prepared::Plan(plan) = scheduler.prepare_step(None).expect("prepare") else {
            panic!("expected a plan");
        };
        assert_eq!(spans(&plan), vec![(2, 5, 6), (1, 2560, 3072)]);
        let grants = [None, Some(Denial::default())];
        scheduler
            .finish_step(plan, &grants, &mut executor, JudgePolicy::Ledger)
            .expect("step");
        scheduler.publish_gauges(0, 0);
        let snapshot = counters.snapshot();
        assert_eq!(snapshot.decode_batches, 2);
        assert_eq!(snapshot.decode_batches_by_width[0], 2);
        assert_eq!((snapshot.prefilling, snapshot.decoding), (1, 1));
        check(snapshot);
    }

    /// Splash `next` / `complete`: a contended unit owes the decoders half
    /// its wall, decode-only steps work it off with theirs, and the debt
    /// goes with the last decoder.
    /// Mutation S2: the share is 0 → red (the second step plans P).
    #[test]
    fn prefill_owes_decoders_half_its_wall() {
        let mut scheduler = measured_scheduler(262_144, DEFAULT_DECODE_SHARE);
        push_decoder(&mut scheduler, 1);
        push_prefill(&mut scheduler, 2, 40_000, 0);
        let mut executor = timed(1000);
        executor.decode_micros = 30_000;
        let plan = step(&mut scheduler, &mut executor);
        assert_eq!(spans(&plan), vec![(1, 4, 5), (2, 0, 512)]);
        assert_eq!(decode_debt(&scheduler), 256.0);
        for (at, debt) in [226.0, 196.0, 166.0, 136.0, 106.0, 76.0, 46.0, 16.0, 0.0]
            .into_iter()
            .enumerate()
        {
            let plan = step(&mut scheduler, &mut executor);
            let start = 5 + at as u32;
            assert_eq!(spans(&plan), vec![(1, start, start + 1)]);
            assert_eq!(decode_debt(&scheduler), debt);
        }
        let plan = step(&mut scheduler, &mut executor);
        assert_eq!(spans(&plan), vec![(1, 14, 15), (2, 512, 1024)]);
        assert_eq!(decode_debt(&scheduler), 256.0);
        executor.finishing.insert(1);
        let plan = step(&mut scheduler, &mut executor);
        assert_eq!(spans(&plan), vec![(1, 15, 16)]);
        assert_eq!(decode_debt(&scheduler), 0.0);
    }

    /// Splash `prefillAdmissionOrder`: a candidate starts only where its
    /// first unit holds it and leaves it waiting for no other lane's
    /// prompt.
    /// Mutation A1: every candidate is served → red (the first case).
    #[test]
    fn admission_waits_for_a_command_that_serves_it() {
        let mut scheduler = measured_scheduler(262_144, DEFAULT_DECODE_SHARE);
        assert!(scheduler.serves_candidate(9, 0, 50_000, None, 0));
        push_prefill(&mut scheduler, 1, 40_000, 0);
        assert!(!scheduler.serves_candidate(9, 0, 50_000, None, 0));
        assert!(scheduler.serves_candidate(9, 0, 300, None, 0));
        let mut scheduler = measured_scheduler(262_144, DEFAULT_DECODE_SHARE);
        push_prefill(&mut scheduler, 1, 400, 0);
        assert!(scheduler.serves_candidate(9, 0, 50_000, None, 0));
        let mut fixed = Scheduler::<&'static str, (), ()>::new(10, PREFILL_CHUNK_ROWS).unwrap();
        fixed.running.push({
            let mut turn = TurnState::new(1, vec![1; 40_000], 0, Vec::new(), None, "p").unwrap();
            turn.status = TurnStatus::Running;
            turn
        });
        assert!(fixed.serves_candidate(9, 0, 50_000, None, 0));
    }

    /// Splash `Engine::tick` commits a unit with the excluded lanes less
    /// those `prepare` refused, so `commit` passes over an older lane whose
    /// row was refused while a younger one ran.
    /// Mutation K1: a refused lane is skipped → red (overtaken 0).
    #[test]
    fn a_refused_older_lane_is_passed_over() {
        let mut scheduler = measured_scheduler(262_144, DEFAULT_DECODE_SHARE);
        push_prefill(&mut scheduler, 1, 40_000, 0);
        push_prefill(&mut scheduler, 2, 300, 0);
        let Prepared::Plan(plan) = scheduler.prepare_step(None).expect("prepare") else {
            panic!("expected a plan");
        };
        assert_eq!(spans(&plan), vec![(2, 0, 300), (1, 0, 1088)]);
        let grants = [None, Some(Denial::default())];
        scheduler
            .finish_step(plan, &grants, &mut timed(1000), JudgePolicy::Ledger)
            .expect("step");
        assert_eq!(running_mut(&mut scheduler, 1).overtaken, 1);
        assert_eq!(running_mut(&mut scheduler, 1).num_computed_tokens, 0);
    }

    /// Splash `Scheduler::snapshot` counts by phase, and `noteRefusal`
    /// leaves a resident lane's phase: a decoder planned without for its
    /// retry still decodes, beside a lane that ended its prompt and a
    /// request the caller holds.
    /// Mutation W3: an excluded lane counts as waiting for resources → red.
    #[test]
    fn a_waiting_lane_keeps_its_phase_in_the_gauges() {
        let mut scheduler = measured_scheduler(262_144, DEFAULT_DECODE_SHARE);
        let counters = scheduler.counters();
        push_decoder(&mut scheduler, 1);
        push_prefill(&mut scheduler, 2, 300, 0);
        running_mut(&mut scheduler, 1).resource_wait.retry_at = 1000.0;
        let clock = Some(RetryClock { now: 0.0, epoch: 0 });
        let Prepared::Plan(plan) = scheduler.prepare_step(clock).expect("prepare") else {
            panic!("expected a plan");
        };
        assert_eq!(spans(&plan), vec![(2, 0, 300)]);
        scheduler
            .finish_step(plan, &[None], &mut timed(1000), JudgePolicy::Ledger)
            .expect("step");
        scheduler.publish_gauges(1, 0);
        let snapshot = counters.snapshot();
        assert_eq!(
            (
                snapshot.queued,
                snapshot.waiting_resources,
                snapshot.prefilling,
                snapshot.decoding
            ),
            (1, 0, 0, 2)
        );
    }
}
