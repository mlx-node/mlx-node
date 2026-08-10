//! Model-neutral step scheduler for continuous batching.
//!
//! The scheduler owns request progress and ordering only. Model execution is
//! behind [`StepExecutor`], so scheduling tests do not load MLX or a model.
//! Prefill versus decode is derived on every plan from token progress; there
//! is deliberately no phase queue.
#![allow(dead_code)] // B4 defines the seam; B5 wires the first production caller.

use std::collections::{BTreeMap, HashSet, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crate::transformer::paged_kv_cache_adapter::SeqId;
use napi_derive::napi;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TurnStatus {
    Waiting,
    Running,
    WaitingForSsd,
    Draining,
}

/// Request-owned state that survives across model steps.
pub(crate) struct TurnState<P> {
    pub seq_id: SeqId,
    pub owner_id: u64,
    pub num_computed_tokens: u32,
    pub num_tokens: u32,
    pub prompt_tokens: u32,
    pub pinned_prefill_breaks: Vec<u32>,
    pub token_history: Vec<u32>,
    pub status: TurnStatus,
    pub cancelled: Option<Arc<AtomicBool>>,
    pub cancel_snapshot: bool,
    /// Full-ISL KV reservation for this request. Zero means the model does
    /// not participate in block-watermark admission.
    pub block_reservation_total: u32,
    pub block_materialized_blocks: u32,
    pub block_size: u32,
    pub payload: P,
    arrival_order: u64,
}

impl<P> TurnState<P> {
    pub fn new(
        seq_id: SeqId,
        owner_id: u64,
        token_history: Vec<u32>,
        num_computed_tokens: u32,
        pinned_prefill_breaks: Vec<u32>,
        cancelled: Option<Arc<AtomicBool>>,
        payload: P,
    ) -> Result<Self, String> {
        let prompt_tokens = u32::try_from(token_history.len())
            .map_err(|_| "token history length exceeds u32::MAX".to_string())?;
        if num_computed_tokens > prompt_tokens {
            return Err(format!(
                "request {seq_id}: computed tokens {num_computed_tokens} exceed prompt tokens {prompt_tokens}"
            ));
        }
        let mut prior = 0;
        for &boundary in &pinned_prefill_breaks {
            if boundary <= prior || boundary > prompt_tokens {
                return Err(format!(
                    "request {seq_id}: invalid pinned prefill boundary {boundary} after {prior} for prompt length {prompt_tokens}"
                ));
            }
            prior = boundary;
        }
        Ok(Self {
            seq_id,
            owner_id,
            num_computed_tokens,
            num_tokens: prompt_tokens,
            prompt_tokens,
            pinned_prefill_breaks,
            token_history,
            status: TurnStatus::Waiting,
            cancelled,
            cancel_snapshot: false,
            block_reservation_total: 0,
            block_materialized_blocks: 0,
            block_size: 0,
            payload,
            arrival_order: 0,
        })
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

    fn unallocated_reserved_blocks(&self) -> u32 {
        if self.block_reservation_total == 0 || self.block_size == 0 {
            return 0;
        }
        self.block_reservation_total
            .saturating_sub(self.block_materialized_blocks)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct BlockTelemetry {
    pub total_blocks: u32,
    pub free_blocks: u32,
    pub reclaimable_blocks: u32,
    pub allocated_blocks: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct BlockAdmission {
    pub admitted: bool,
    pub watermark_blocks: u32,
    pub reserved_blocks: u32,
}

pub(crate) fn block_admission_decision(
    telemetry: BlockTelemetry,
    existing_unallocated: u32,
    candidate_blocks: u32,
    has_live_rows: bool,
    watermark_fraction: f64,
) -> BlockAdmission {
    let fraction = if watermark_fraction.is_finite() {
        watermark_fraction.clamp(0.0, 1.0)
    } else {
        0.05
    };
    let watermark_blocks = if has_live_rows {
        ((telemetry.total_blocks as f64) * fraction).ceil() as u32
    } else {
        0
    };
    let reserved_blocks = existing_unallocated.saturating_add(candidate_blocks);
    BlockAdmission {
        admitted: reserved_blocks.saturating_add(watermark_blocks)
            <= telemetry
                .free_blocks
                .saturating_add(telemetry.reclaimable_blocks),
        watermark_blocks,
        reserved_blocks,
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
    pub token_budget: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RowStepResult {
    pub seq_id: SeqId,
    pub num_computed_tokens: u32,
    pub generated_token: Option<u32>,
    pub finished: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct StepResult {
    pub rows: Vec<RowStepResult>,
    /// Largest decode forward the executor actually fused in this step.
    /// This is executor-reported rather than inferred from the plan so a
    /// silent N-times-scalar fallback cannot fake occupancy telemetry.
    pub executed_decode_batch: usize,
    pub rows_alloc_evicted: u32,
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
    pub ssd_restore_waiting: u32,
    pub ssd_restore_bytes: u64,
    pub ssd_restore_wait_micros: u64,
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
    pub ssd_restore_waiting: u32,
    pub ssd_restore_bytes: f64,
    pub ssd_restore_wait_ms: f64,
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
            ssd_restore_waiting: self.ssd_restore_waiting,
            ssd_restore_bytes: self.ssd_restore_bytes as f64,
            ssd_restore_wait_ms: self.ssd_restore_wait_micros as f64 / 1_000.0,
        }
    }
}

#[derive(Debug)]
pub(crate) enum SchedulerError<E> {
    Executor(E),
    InvalidResult(String),
}

pub(crate) enum SchedulerAction<P, Exclusive, Barrier> {
    Idle,
    Exclusive(Exclusive),
    Barrier(Barrier),
    Stepped {
        plan: StepPlan,
        completed: Vec<TurnState<P>>,
    },
}

struct Ordered<T> {
    order: u64,
    value: T,
}

/// FCFS scheduler with a single token budget shared by prefix-hit prefills,
/// cold prefills, and decode rows.
pub(crate) struct Scheduler<P, Exclusive, Barrier> {
    waiting: VecDeque<TurnState<P>>,
    exclusive_lane: VecDeque<Ordered<Exclusive>>,
    barriers: VecDeque<Ordered<Barrier>>,
    running: Vec<TurnState<P>>,
    max_num_seqs: usize,
    max_num_batched_tokens: u32,
    next_order: u64,
    global_step: u64,
    stats: SchedulerStats,
}

impl<P, Exclusive, Barrier> Scheduler<P, Exclusive, Barrier> {
    pub fn new(max_num_seqs: usize, max_num_batched_tokens: u32) -> Result<Self, String> {
        if max_num_seqs == 0 {
            return Err("max_num_seqs must be positive".to_string());
        }
        if max_num_batched_tokens == 0 {
            return Err("max_num_batched_tokens must be positive".to_string());
        }
        Ok(Self {
            waiting: VecDeque::new(),
            exclusive_lane: VecDeque::new(),
            barriers: VecDeque::new(),
            running: Vec::new(),
            max_num_seqs,
            max_num_batched_tokens,
            next_order: 0,
            global_step: 0,
            stats: SchedulerStats::default(),
        })
    }

    fn take_order(&mut self) -> u64 {
        let order = self.next_order;
        self.next_order = self.next_order.saturating_add(1);
        order
    }

    pub fn enqueue_turn(&mut self, mut turn: TurnState<P>) -> Result<(), String> {
        if self
            .waiting
            .iter()
            .chain(self.running.iter())
            .any(|existing| existing.seq_id == turn.seq_id)
        {
            return Err(format!("duplicate scheduler sequence {}", turn.seq_id));
        }
        turn.status = TurnStatus::Waiting;
        turn.arrival_order = self.take_order();
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

    fn unallocated_reserved_blocks(&self) -> u32 {
        self.waiting
            .iter()
            .chain(self.running.iter())
            .fold(0u32, |sum, turn| {
                sum.saturating_add(turn.unallocated_reserved_blocks())
            })
    }

    pub fn observe_blocks(
        &mut self,
        telemetry: BlockTelemetry,
        watermark_fraction: f64,
    ) -> BlockAdmission {
        let decision = block_admission_decision(
            telemetry,
            self.unallocated_reserved_blocks(),
            0,
            self.has_live_turns(),
            watermark_fraction,
        );
        self.stats.block_capacity = telemetry.total_blocks;
        self.stats.free_blocks = telemetry.free_blocks;
        self.stats.reclaimable_blocks = telemetry.reclaimable_blocks;
        self.stats.allocated_blocks = telemetry.allocated_blocks;
        self.stats.watermark_blocks = decision.watermark_blocks;
        self.stats.reserved_blocks = decision.reserved_blocks;
        decision
    }

    pub fn try_reserve_blocks(
        &mut self,
        telemetry: BlockTelemetry,
        candidate_blocks: u32,
        watermark_fraction: f64,
    ) -> BlockAdmission {
        let existing_unallocated = self.unallocated_reserved_blocks();
        let decision = block_admission_decision(
            telemetry,
            existing_unallocated,
            candidate_blocks,
            self.has_live_turns(),
            watermark_fraction,
        );
        self.stats.block_capacity = telemetry.total_blocks;
        self.stats.free_blocks = telemetry.free_blocks;
        self.stats.reclaimable_blocks = telemetry.reclaimable_blocks;
        self.stats.allocated_blocks = telemetry.allocated_blocks;
        self.stats.watermark_blocks = decision.watermark_blocks;
        self.stats.reserved_blocks = decision.reserved_blocks;
        if !decision.admitted {
            self.stats.admission_deferred_blocks =
                self.stats.admission_deferred_blocks.saturating_add(1);
        }
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

    pub fn maintenance_due(&self, cadence: u64) -> bool {
        cadence != 0 && self.global_step != 0 && self.global_step.is_multiple_of(cadence)
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

    fn admit_waiting(&mut self) {
        let control_order = self.earliest_control_order();
        while self.running.len() < self.max_num_seqs {
            let candidate = self.waiting.iter().position(|turn| {
                turn.status == TurnStatus::Waiting
                    && !control_order.is_some_and(|order| turn.arrival_order > order)
            });
            let Some(candidate) = candidate else {
                break;
            };
            let mut turn = self
                .waiting
                .remove(candidate)
                .expect("candidate checked above");
            turn.status = TurnStatus::Running;
            self.running.push(turn);
            self.stats.admitted = self.stats.admitted.saturating_add(1);
        }
    }

    fn build_plan(&mut self) -> StepPlan {
        for turn in &mut self.running {
            turn.snapshot_cancel();
        }
        let mut budget = self.max_num_batched_tokens;
        let mut rows = Vec::with_capacity(self.running.len());
        // Decode-first is a priority within ONE phase-free plan, not a decode
        // queue. Every row's kind is still derived from token progress. This
        // matches vLLM v1's running-first budget shape and prevents a long
        // newly admitted prefill from consuming the step budget ahead of an
        // already interactive decode row.
        for planned_kind in [StepKind::Decode, StepKind::Prefill] {
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
                let num_tokens = pending.min(row_limit).min(budget);
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
                budget -= num_tokens;
            }
        }
        StepPlan {
            global_step: self.global_step,
            rows,
            token_budget: self.max_num_batched_tokens,
        }
    }

    pub fn drive_once<E>(
        &mut self,
        executor: &mut E,
    ) -> Result<SchedulerAction<P, Exclusive, Barrier>, SchedulerError<E::Error>>
    where
        E: StepExecutor<P>,
    {
        if let Some(action) = self.pop_control_if_due() {
            return Ok(action);
        }
        self.admit_waiting();
        if self.running.is_empty() {
            return Ok(SchedulerAction::Idle);
        }
        let plan = self.build_plan();
        if plan.rows.is_empty() {
            return Ok(SchedulerAction::Idle);
        }
        let result = executor
            .execute(&plan, &mut self.running)
            .map_err(SchedulerError::Executor)?;
        let executed_decode_batch = result.executed_decode_batch;
        let rows_alloc_evicted = result.rows_alloc_evicted;
        let planned_decode_rows = plan
            .rows
            .iter()
            .filter(|row| row.kind == StepKind::Decode)
            .count();
        if executed_decode_batch > planned_decode_rows {
            return Err(SchedulerError::InvalidResult(format!(
                "executor reported decode occupancy {executed_decode_batch} for {planned_decode_rows} planned decode rows"
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
        Ok(SchedulerAction::Stepped { plan, completed })
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
            if output.num_computed_tokens != planned.num_tokens {
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
            if output.finished {
                turn.status = TurnStatus::Draining;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct MockStep {
        plans: Vec<StepPlan>,
        finish: HashSet<SeqId>,
        next_token: u32,
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
                        num_computed_tokens: row.num_tokens,
                        generated_token,
                        finished: self.finish.contains(&row.seq_id) || row.cancel_snapshot,
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
                rows_alloc_evicted: 0,
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
            u64::from(seq_id),
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
        let SchedulerAction::Stepped { plan, completed } =
            scheduler.drive_once(&mut step).expect("step")
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
        assert!(scheduler.maintenance_due(2));
        assert!(!scheduler.maintenance_due(3));
        let js = scheduler.stats().to_js();
        assert_eq!(js.max_batch_occupancy, 2);
        assert_eq!(js.decode_batch_occupancy_hist[0].occupancy, 2);
        assert_eq!(js.decode_batch_occupancy_hist[0].steps, 4.0);
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

    #[test]
    fn block_admission_reserves_growth_and_watermark_before_allocation() {
        let mut scheduler = Scheduler::<_, (), ()>::new(2, 8).expect("scheduler");
        scheduler
            .enqueue_turn(turn(1, 4, &[4], None).with_block_reservation(6, 2, 2))
            .expect("enqueue reserved row");

        // A has four unallocated blocks left. B asks for five. Both the
        // existing reservation and the 10% watermark are individually
        // decisive: ignoring either would incorrectly admit B into ten free
        // blocks (5 + 2 <= 10, or 4 + 5 <= 10).
        let deferred = scheduler.try_reserve_blocks(
            BlockTelemetry {
                total_blocks: 20,
                free_blocks: 10,
                reclaimable_blocks: 0,
                allocated_blocks: 10,
            },
            5,
            0.10,
        );
        assert_eq!(
            deferred,
            BlockAdmission {
                admitted: false,
                watermark_blocks: 2,
                reserved_blocks: 9,
            }
        );
        assert_eq!(scheduler.stats().admission_deferred_blocks, 1);
        assert_eq!(scheduler.stats().free_blocks, 10);

        let admitted = scheduler.try_reserve_blocks(
            BlockTelemetry {
                total_blocks: 20,
                free_blocks: 10,
                reclaimable_blocks: 0,
                allocated_blocks: 10,
            },
            4,
            0.10,
        );
        assert!(admitted.admitted);
        assert_eq!(admitted.reserved_blocks + admitted.watermark_blocks, 10);
    }

    #[test]
    fn first_block_admission_has_no_watermark_but_still_must_fit() {
        assert_eq!(
            block_admission_decision(
                BlockTelemetry {
                    total_blocks: 100,
                    free_blocks: 8,
                    reclaimable_blocks: 0,
                    allocated_blocks: 92,
                },
                0,
                8,
                false,
                0.05,
            ),
            BlockAdmission {
                admitted: true,
                watermark_blocks: 0,
                reserved_blocks: 8,
            }
        );
        assert!(
            !block_admission_decision(
                BlockTelemetry {
                    total_blocks: 100,
                    free_blocks: 8,
                    reclaimable_blocks: 0,
                    allocated_blocks: 92,
                },
                0,
                9,
                false,
                0.05,
            )
            .admitted
        );
    }
}
