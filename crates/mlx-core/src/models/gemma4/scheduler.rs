use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use napi::bindgen_prelude::{Error, Result};

use super::*;
use crate::array::{MxArray, synchronize_and_clear_cache};
use crate::decode_profiler::DecodeProfiler;
use crate::engine::backend::{ChatBackend, ChunkSink, StreamEmitter, TurnOutput};
use crate::engine::cmd::{ChatCmd, FromChatCmd, handle_chat_cmd};
use crate::engine::scheduler::{
    PreemptionReplay, RowStepResult, Scheduler, SchedulerAction, StepExecutor, StepKind, StepPlan,
    StepResult, TurnState, install_preemption_replay, is_paged_allocation_blocked,
};
use crate::engine::types::{ChatResult, ChatStreamChunk};
use crate::engine::{self};
use crate::model_thread::{LoopControl, ResponseTx, StreamTx};
use crate::sampling::{check_repetition_cutoff, sample};
use crate::stream::{DeviceType, Stream, StreamContext};
use crate::transformer::paged_kv_cache_adapter::SeqId;

fn scheduler_max_num_seqs(inner: &Gemma4Inner) -> usize {
    let physical = inner
        .kv_cache_coordinator
        .as_ref()
        .map_or(1, |coordinator| coordinator.max_concurrent_sequences());
    std::env::var("MLX_SCHED_MAX_NUM_SEQS")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(8)
        .min(32)
        .min(physical) as usize
}

fn scheduler_max_batched_tokens() -> u32 {
    std::env::var("MLX_SCHED_MAX_BATCHED_TOKENS")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(2048)
}

fn scheduler_prefill_slice_tokens() -> u32 {
    super::gemma4_paged_prefill_group_max_chunk().max(1)
}

/// Commands owned by the Gemma 4 scheduler thread. Optional draft and media
/// capabilities remain ordinary chat commands; request classification decides
/// whether they enter the scheduled or exclusive lane.
pub(crate) enum Gemma4Cmd {
    Chat(Box<ChatCmd>),
    SchedulerStats {
        reply: ResponseTx<engine::SchedulerStatsJs>,
    },
}

impl FromChatCmd for Gemma4Cmd {
    fn from_chat(command: ChatCmd) -> Self {
        Self::Chat(Box::new(command))
    }
}

enum ScheduledReply {
    Sync(ResponseTx<ChatResult>),
    Stream(StreamTx<ChatStreamChunk>),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum OwnerLane {
    Paged,
    Flat,
}

impl ScheduledReply {
    fn sink(&self) -> Option<&dyn ChunkSink> {
        match self {
            Self::Sync(_) => None,
            Self::Stream(stream) => Some(stream),
        }
    }

    fn send_error(self, error: Error, cancelled: &AtomicBool) {
        match self {
            Self::Sync(reply) => {
                let error = if cancelled.load(Ordering::Relaxed) {
                    Error::from_reason(engine::session::CHAT_SESSION_CANCELLED)
                } else {
                    error
                };
                let _ = reply.send(Err(error));
            }
            Self::Stream(stream) => ChunkSink::send(&stream, Err(error)),
        }
    }
}

struct ScheduledTurn {
    owner_id: String,
    tokenizer: Arc<Qwen3Tokenizer>,
    eos_id: u32,
    config: ChatConfig,
    params: engine::params::ChatParams,
    thinking: engine::backend::ThinkingSetup,
    prompt_tokens: Vec<u32>,
    prefix: Gemma4PrefixState,
    is_delta: bool,
    reuse_cache: bool,
    response: ScheduledReply,
    generated_tokens: Vec<u32>,
    finish_reason: String,
    reasoning_tracker: engine::penalties::ReasoningTracker,
    extra_eos_ids: Vec<u32>,
    generation_start: Option<Instant>,
    first_token_instant: Option<Instant>,
    generation_stream: Stream,
    profiler: DecodeProfiler,
    emitter: Option<Box<dyn StreamEmitter>>,
    stream_skip_special: bool,
    decode_ids: Vec<u32>,
    decode_prefix: String,
    decode_prefix_index: usize,
    streamed_text_len: usize,
    last_is_reasoning: bool,
    failure: Option<Error>,
    allocation_failed: bool,
    preemption_replay: Option<PreemptionReplay>,
}

struct PreparedTurn {
    admitted: engine::session::AdmittedPagedTurn,
    response: ScheduledReply,
    cancelled: Arc<AtomicBool>,
    owner_id: String,
    seq_id: SeqId,
    newly_assigned: bool,
}

struct Gemma4StepExecutor<'a> {
    inner: &'a mut Gemma4Inner,
}

struct Gemma4PreparedDecodeRow {
    plan_index: usize,
    seq_id: SeqId,
    token_id: u32,
    terminal: bool,
    at_length: bool,
    stops_at_eos: bool,
    cancelled: bool,
    repetition: Option<&'static str>,
    batch_index: Option<usize>,
}

impl Gemma4StepExecutor<'_> {
    fn blocked(row: &engine::scheduler::StepRow) -> RowStepResult {
        RowStepResult {
            seq_id: row.seq_id,
            num_computed_tokens: 0,
            generated_token: None,
            finished: false,
            allocation_blocked: true,
            prefill_micros: 0,
        }
    }

    fn elapsed_micros(started: Instant) -> u64 {
        started
            .elapsed()
            .as_micros()
            .try_into()
            .unwrap_or(u64::MAX)
            .max(1)
    }

    fn fail(
        turn: &mut TurnState<ScheduledTurn>,
        row: &engine::scheduler::StepRow,
        error: Error,
    ) -> RowStepResult {
        turn.payload.allocation_failed |= is_paged_allocation_blocked(&error.reason);
        turn.payload.failure = Some(error);
        turn.payload.profiler.snapshot_memory_after();
        turn.payload.profiler.report();
        RowStepResult {
            seq_id: row.seq_id,
            num_computed_tokens: row.num_tokens,
            generated_token: None,
            finished: true,
            allocation_blocked: false,
            prefill_micros: 0,
        }
    }

    fn stream_token(turn: &mut TurnState<ScheduledTurn>, token_id: u32, is_reasoning: bool) {
        let payload = &mut turn.payload;
        let text = match tokenizers::tokenizer::step_decode_stream(
            payload.tokenizer.inner(),
            vec![token_id],
            payload.stream_skip_special,
            &mut payload.decode_ids,
            &mut payload.decode_prefix,
            &mut payload.decode_prefix_index,
        ) {
            Ok(Some(text)) => text,
            Ok(None) => String::new(),
            Err(_) => {
                payload.decode_ids.clear();
                payload.decode_prefix.clear();
                payload.decode_prefix_index = 0;
                let mut replayed = String::new();
                for &token in &payload.generated_tokens {
                    if let Ok(Some(text)) = tokenizers::tokenizer::step_decode_stream(
                        payload.tokenizer.inner(),
                        vec![token],
                        payload.stream_skip_special,
                        &mut payload.decode_ids,
                        &mut payload.decode_prefix,
                        &mut payload.decode_prefix_index,
                    ) {
                        replayed.push_str(&text);
                    }
                }
                replayed
                    .get(payload.streamed_text_len..)
                    .unwrap_or_default()
                    .to_string()
            }
        };
        payload.streamed_text_len = payload.streamed_text_len.saturating_add(text.len());
        if let (Some(sink), Some(emitter)) = (payload.response.sink(), payload.emitter.as_mut()) {
            emitter.on_token_text(&text, is_reasoning, payload.params.include_reasoning, sink);
        }
    }

    fn finish_decode_row(turn: &mut TurnState<ScheduledTurn>, prepared: &Gemma4PreparedDecodeRow) {
        turn.payload.profiler.mark_first_token();
        let is_reasoning = turn
            .payload
            .reasoning_tracker
            .observe_token(prepared.token_id);
        turn.payload.last_is_reasoning = is_reasoning;
        if prepared.cancelled {
            turn.payload.finish_reason = String::from("cancelled");
        } else {
            if turn.payload.response.sink().is_some() {
                Self::stream_token(turn, prepared.token_id, is_reasoning);
            }
            if prepared.stops_at_eos {
                turn.payload.finish_reason = String::from("stop");
            } else if let Some(reason) = prepared.repetition {
                turn.payload.finish_reason = reason.to_string();
            }
        }
    }

    fn execute_prefill(
        &mut self,
        row: &engine::scheduler::StepRow,
        turn: &mut TurnState<ScheduledTurn>,
    ) -> Result<RowStepResult> {
        if row.cancel_snapshot {
            return Ok(Self::fail(
                turn,
                row,
                Error::from_reason(engine::session::CHAT_SESSION_CANCELLED),
            ));
        }
        let source = turn
            .payload
            .preemption_replay
            .as_ref()
            .map_or(turn.payload.prompt_tokens.as_slice(), |replay| {
                replay.tokens.as_slice()
            });
        let start = row.token_start as usize;
        let end = start.saturating_add(row.num_tokens as usize);
        let Some(slice) = source.get(start..end) else {
            return Ok(Self::fail(
                turn,
                row,
                Error::from_reason("Gemma4 scheduler prefill slice exceeds prompt"),
            ));
        };
        self.inner.set_active_paged_owner(row.seq_id);
        self.inner
            .set_turn_cancel_flag(turn.cancelled.as_ref().map(Arc::clone));
        let final_slice = end == source.len();
        let started = Instant::now();
        turn.payload.profiler.begin_prefill();
        let logits = match self.inner.run_scheduled_paged_prefill_slice(
            row.seq_id,
            slice,
            row.token_start,
            final_slice,
        ) {
            Ok(logits) => logits,
            Err(error) if is_paged_allocation_blocked(&error.reason) => {
                turn.payload.profiler.end_prefill();
                return Ok(Self::blocked(row));
            }
            Err(error) => {
                turn.payload.profiler.end_prefill();
                return Ok(Self::fail(turn, row, error));
            }
        };
        turn.payload.profiler.end_prefill();
        if !final_slice {
            return Ok(RowStepResult {
                seq_id: row.seq_id,
                num_computed_tokens: row.num_tokens,
                generated_token: None,
                finished: false,
                allocation_blocked: false,
                prefill_micros: Self::elapsed_micros(started),
            });
        }
        if turn
            .payload
            .preemption_replay
            .as_ref()
            .is_some_and(|replay| replay.suppress_sample)
        {
            turn.payload.preemption_replay = None;
            synchronize_and_clear_cache();
            return Ok(RowStepResult {
                seq_id: row.seq_id,
                num_computed_tokens: row.num_tokens,
                generated_token: None,
                finished: false,
                allocation_blocked: false,
                prefill_micros: Self::elapsed_micros(started),
            });
        }
        turn.payload.preemption_replay = None;
        let Some(logits) = logits else {
            return Ok(Self::fail(
                turn,
                row,
                Error::from_reason("Gemma4 final prefill slice produced no logits"),
            ));
        };
        let logits = match engine::penalties::apply_all_penalties(
            logits,
            &turn.token_history,
            &turn.payload.params,
        ) {
            Ok(logits) => logits,
            Err(error) => return Ok(Self::fail(turn, row, error)),
        };
        let sampled = match sample(&logits, turn.payload.params.sampling_config) {
            Ok(sampled) => sampled,
            Err(error) => return Ok(Self::fail(turn, row, error)),
        };
        sampled.eval();
        if turn.payload.params.report_performance {
            turn.payload.first_token_instant = Some(Instant::now());
        }
        synchronize_and_clear_cache();
        if turn.payload.params.max_new_tokens == 0 {
            turn.payload.profiler.snapshot_memory_after();
            turn.payload.profiler.report();
            return Ok(RowStepResult {
                seq_id: row.seq_id,
                num_computed_tokens: row.num_tokens,
                generated_token: None,
                finished: true,
                allocation_blocked: false,
                prefill_micros: Self::elapsed_micros(started),
            });
        }
        let token = match sampled.item_at_int32(0) {
            Ok(token) => token as u32,
            Err(error) => return Ok(Self::fail(turn, row, error)),
        };
        Ok(RowStepResult {
            seq_id: row.seq_id,
            num_computed_tokens: row.num_tokens,
            generated_token: Some(token),
            finished: false,
            allocation_blocked: false,
            prefill_micros: Self::elapsed_micros(started),
        })
    }

    fn execute_decode(
        &mut self,
        row: &engine::scheduler::StepRow,
        turn: &mut TurnState<ScheduledTurn>,
    ) -> Result<RowStepResult> {
        let Some(&token_id) = turn.token_history.last() else {
            return Ok(Self::fail(
                turn,
                row,
                Error::from_reason("Gemma4 scheduler decode row has no current token"),
            ));
        };
        self.inner.set_active_paged_owner(row.seq_id);
        let step_index = turn.payload.generated_tokens.len();
        turn.payload.generated_tokens.push(token_id);
        crate::array::maybe_clear_cache_for_paged_step(step_index as i32);
        let stops_at_eos =
            token_id == turn.payload.eos_id || turn.payload.extra_eos_ids.contains(&token_id);
        let repetition = check_repetition_cutoff(
            &turn.payload.generated_tokens,
            turn.payload.params.max_consecutive_tokens,
            turn.payload.params.max_ngram_repeats,
            turn.payload.params.ngram_size,
        );
        let terminal = stops_at_eos || row.cancel_snapshot || repetition.is_some();
        let at_length = turn.payload.generated_tokens.len()
            >= turn.payload.params.max_new_tokens.max(0) as usize;

        let forward_logits = if !terminal {
            let _stream_context = StreamContext::new(turn.payload.generation_stream);
            turn.payload.profiler.begin("forward");
            let logits = match self
                .inner
                .run_paged_decode_step_for(row.seq_id, token_id)
                .and_then(|logits| logits.squeeze(Some(&[1])))
            {
                Ok(logits) => logits,
                Err(error) if is_paged_allocation_blocked(&error.reason) => {
                    turn.payload.profiler.end();
                    let popped = turn.payload.generated_tokens.pop();
                    debug_assert_eq!(popped, Some(token_id));
                    return Ok(Self::blocked(row));
                }
                Err(error) => return Ok(Self::fail(turn, row, error)),
            };
            turn.payload.profiler.end();
            Some(logits)
        } else {
            None
        };

        let next_token = if !at_length && let Some(mut logits) = forward_logits {
            let sampled = if turn.payload.reasoning_tracker.should_force_think_end() {
                let forced = match turn.payload.reasoning_tracker.forced_token_id() {
                    Ok(token) => token as i32,
                    Err(error) => return Ok(Self::fail(turn, row, error)),
                };
                match MxArray::from_int32(&[forced], &[1]) {
                    Ok(sampled) => sampled,
                    Err(error) => return Ok(Self::fail(turn, row, error)),
                }
            } else {
                logits = match engine::penalties::apply_all_penalties(
                    logits,
                    &turn.token_history,
                    &turn.payload.params,
                ) {
                    Ok(logits) => logits,
                    Err(error) => return Ok(Self::fail(turn, row, error)),
                };
                match sample(&logits, turn.payload.params.sampling_config) {
                    Ok(sampled) => sampled,
                    Err(error) => return Ok(Self::fail(turn, row, error)),
                }
            };
            MxArray::async_eval_arrays(&[&sampled]);
            sampled.eval();
            match sampled.item_at_int32(0) {
                Ok(token) => Some(token as u32),
                Err(error) => return Ok(Self::fail(turn, row, error)),
            }
        } else {
            if let Some(logits) = forward_logits {
                logits.eval();
            }
            None
        };
        if !terminal {
            if let Some(coordinator) = self.inner.kv_cache_coordinator.as_mut()
                && let Err(error) = coordinator.eval_pending_pool_writes_all()
            {
                return Ok(Self::fail(turn, row, Error::from_reason(error)));
            }
            if let Err(error) = self
                .inner
                .remember_grouped_sliding_cold_checkpoint(row.seq_id)
            {
                return Ok(Self::fail(turn, row, error));
            }
            if let Some(coordinator) = self.inner.kv_cache_coordinator.as_mut()
                && let Err(error) = coordinator.prune_sliding_all(row.seq_id)
            {
                return Ok(Self::fail(turn, row, Error::from_reason(error)));
            }
        }

        turn.payload.profiler.step();
        Self::finish_decode_row(
            turn,
            &Gemma4PreparedDecodeRow {
                plan_index: 0,
                seq_id: row.seq_id,
                token_id,
                terminal,
                at_length,
                stops_at_eos,
                cancelled: row.cancel_snapshot,
                repetition,
                batch_index: None,
            },
        );
        let finished = terminal || at_length;
        if finished {
            turn.payload.profiler.snapshot_memory_after();
            turn.payload.profiler.report();
        }
        Ok(RowStepResult {
            seq_id: row.seq_id,
            num_computed_tokens: row.num_tokens,
            generated_token: (!finished).then_some(next_token).flatten(),
            finished,
            allocation_blocked: false,
            prefill_micros: 0,
        })
    }

    fn execute_decode_batch(
        &mut self,
        plan: &StepPlan,
        running: &mut [TurnState<ScheduledTurn>],
    ) -> (Vec<(usize, RowStepResult)>, usize, usize) {
        let mut work = Vec::new();
        let mut results = Vec::new();
        let mut batch_rows = Vec::new();
        for (plan_index, row) in plan
            .rows
            .iter()
            .enumerate()
            .filter(|(_, row)| row.kind == StepKind::Decode)
        {
            let turn = running
                .iter_mut()
                .find(|turn| turn.seq_id == row.seq_id)
                .expect("scheduler validated Gemma4 decode row");
            let Some(&token_id) = turn.token_history.last() else {
                results.push((
                    plan_index,
                    Self::fail(
                        turn,
                        row,
                        Error::from_reason("Gemma4 scheduler decode row has no current token"),
                    ),
                ));
                continue;
            };
            turn.payload.generated_tokens.push(token_id);
            let stops_at_eos =
                token_id == turn.payload.eos_id || turn.payload.extra_eos_ids.contains(&token_id);
            let repetition = check_repetition_cutoff(
                &turn.payload.generated_tokens,
                turn.payload.params.max_consecutive_tokens,
                turn.payload.params.max_ngram_repeats,
                turn.payload.params.ngram_size,
            );
            let terminal = stops_at_eos || row.cancel_snapshot || repetition.is_some();
            let at_length = turn.payload.generated_tokens.len()
                >= turn.payload.params.max_new_tokens.max(0) as usize;
            // Gemma's paged finalize saves the generated history only after
            // the last sampled token is materialized in K/V. Therefore an
            // at-length non-terminal row still participates in this forward,
            // but its logits are not sampled.
            let batch_index = (!terminal).then(|| {
                let index = batch_rows.len();
                batch_rows.push((row.seq_id, token_id));
                index
            });
            work.push(Gemma4PreparedDecodeRow {
                plan_index,
                seq_id: row.seq_id,
                token_id,
                terminal,
                at_length,
                stops_at_eos,
                cancelled: row.cancel_snapshot,
                repetition,
                batch_index,
            });
        }

        crate::array::maybe_clear_cache_for_paged_step(plan.global_step as i32);
        for prepared in &work {
            if prepared.batch_index.is_some() {
                running
                    .iter_mut()
                    .find(|turn| turn.seq_id == prepared.seq_id)
                    .expect("prepared Gemma4 decode row remains running")
                    .payload
                    .profiler
                    .begin("forward");
            }
        }
        let executed_decode_batch = batch_rows.len();
        let batched_logits = if batch_rows.is_empty() {
            Ok(None)
        } else {
            let _stream_context = StreamContext::new(Stream::default(DeviceType::Gpu));
            self.inner
                .run_paged_decode_step_batched(&batch_rows)
                .and_then(|logits| {
                    self.inner
                        .kv_cache_coordinator
                        .as_mut()
                        .ok_or_else(|| {
                            Error::from_reason("Gemma4 batched decode lost its KV coordinator")
                        })?
                        .eval_pending_pool_writes_all()
                        .map_err(Error::from_reason)?;
                    for &(seq_id, _) in &batch_rows {
                        self.inner
                            .remember_grouped_sliding_cold_checkpoint(seq_id)?;
                    }
                    let coordinator =
                        self.inner.kv_cache_coordinator.as_mut().ok_or_else(|| {
                            Error::from_reason("Gemma4 batched decode lost its KV coordinator")
                        })?;
                    for &(seq_id, _) in &batch_rows {
                        coordinator
                            .prune_sliding_all(seq_id)
                            .map_err(Error::from_reason)?;
                    }
                    Ok(Some(logits))
                })
        };
        for prepared in &work {
            if prepared.batch_index.is_some() {
                running
                    .iter_mut()
                    .find(|turn| turn.seq_id == prepared.seq_id)
                    .expect("prepared Gemma4 decode row remains running")
                    .payload
                    .profiler
                    .end();
            }
        }

        let allocation_blocked = batched_logits
            .as_ref()
            .is_err_and(|error| is_paged_allocation_blocked(&error.reason));
        let greedy_tokens = match &batched_logits {
            Ok(Some(logits)) => {
                let sampling_rows = work
                    .iter()
                    .filter(|prepared| prepared.batch_index.is_some() && !prepared.at_length)
                    .collect::<Vec<_>>();
                if engine::batch_sampling::can_batch_greedy_wave(sampling_rows.iter().map(
                    |prepared| {
                        let turn = running
                            .iter()
                            .find(|turn| turn.seq_id == prepared.seq_id)
                            .expect("prepared Gemma4 decode row remains running");
                        (
                            &turn.payload.params,
                            turn.payload.reasoning_tracker.force_think_end_pending(),
                        )
                    },
                )) {
                    engine::batch_sampling::batch_greedy_tokens_or_fallback(logits)
                } else {
                    None
                }
            }
            _ => None,
        };
        let executed_greedy_epilogue_batch = greedy_tokens.as_ref().map_or(0, |_| {
            work.iter()
                .filter(|prepared| prepared.batch_index.is_some() && !prepared.at_length)
                .count()
        });
        for prepared in work {
            let planned = &plan.rows[prepared.plan_index];
            let turn = running
                .iter_mut()
                .find(|turn| turn.seq_id == prepared.seq_id)
                .expect("prepared Gemma4 decode row remains running");
            let greedy_next = if prepared.at_length {
                None
            } else {
                prepared.batch_index.and_then(|batch_index| {
                    greedy_tokens.as_ref().map(|tokens| tokens[batch_index])
                })
            };
            let logits = match (&batched_logits, prepared.batch_index) {
                (Err(_), Some(_)) if allocation_blocked => {
                    let popped = turn.payload.generated_tokens.pop();
                    debug_assert_eq!(popped, Some(prepared.token_id));
                    results.push((prepared.plan_index, Self::blocked(planned)));
                    continue;
                }
                (Err(error), Some(_)) => {
                    results.push((
                        prepared.plan_index,
                        Self::fail(turn, planned, Error::from_reason(error.reason.clone())),
                    ));
                    continue;
                }
                (Ok(Some(_)), Some(_)) if greedy_next.is_some() => None,
                (Ok(Some(logits)), Some(batch_index)) => match logits
                    .slice_axis(0, batch_index as i64, batch_index as i64 + 1)
                    .and_then(|row| row.squeeze(Some(&[1])))
                {
                    Ok(logits) => Some(logits),
                    Err(error) => {
                        results.push((prepared.plan_index, Self::fail(turn, planned, error)));
                        continue;
                    }
                },
                _ => None,
            };
            let next_token = if let Some(token) = greedy_next {
                Some(token)
            } else if !prepared.at_length {
                if let Some(mut logits) = logits {
                    let sampled = if turn.payload.reasoning_tracker.should_force_think_end() {
                        let forced = match turn.payload.reasoning_tracker.forced_token_id() {
                            Ok(token) => token as i32,
                            Err(error) => {
                                results
                                    .push((prepared.plan_index, Self::fail(turn, planned, error)));
                                continue;
                            }
                        };
                        match MxArray::from_int32(&[forced], &[1]) {
                            Ok(sampled) => sampled,
                            Err(error) => {
                                results
                                    .push((prepared.plan_index, Self::fail(turn, planned, error)));
                                continue;
                            }
                        }
                    } else {
                        logits = match engine::penalties::apply_all_penalties(
                            logits,
                            &turn.token_history,
                            &turn.payload.params,
                        ) {
                            Ok(logits) => logits,
                            Err(error) => {
                                results
                                    .push((prepared.plan_index, Self::fail(turn, planned, error)));
                                continue;
                            }
                        };
                        match sample(&logits, turn.payload.params.sampling_config) {
                            Ok(sampled) => sampled,
                            Err(error) => {
                                results
                                    .push((prepared.plan_index, Self::fail(turn, planned, error)));
                                continue;
                            }
                        }
                    };
                    MxArray::async_eval_arrays(&[&sampled, &logits]);
                    sampled.eval();
                    match sampled.item_at_int32(0) {
                        Ok(token) => Some(token as u32),
                        Err(error) => {
                            results.push((prepared.plan_index, Self::fail(turn, planned, error)));
                            continue;
                        }
                    }
                } else {
                    None
                }
            } else {
                if let Some(logits) = logits {
                    logits.eval();
                }
                None
            };
            turn.payload.profiler.step();
            Self::finish_decode_row(turn, &prepared);
            let finished = prepared.terminal || prepared.at_length;
            if finished {
                turn.payload.profiler.snapshot_memory_after();
                turn.payload.profiler.report();
            }
            results.push((
                prepared.plan_index,
                RowStepResult {
                    seq_id: prepared.seq_id,
                    num_computed_tokens: planned.num_tokens,
                    generated_token: (!finished).then_some(next_token).flatten(),
                    finished,
                    allocation_blocked: false,
                    prefill_micros: 0,
                },
            ));
        }
        (
            results,
            executed_decode_batch,
            executed_greedy_epilogue_batch,
        )
    }
}

impl StepExecutor<ScheduledTurn> for Gemma4StepExecutor<'_> {
    type Error = std::convert::Infallible;

    fn execute(
        &mut self,
        plan: &StepPlan,
        running: &mut [TurnState<ScheduledTurn>],
    ) -> std::result::Result<StepResult, Self::Error> {
        let mut results = Vec::with_capacity(plan.rows.len());
        results.resize_with(plan.rows.len(), || None);
        let decode_count = plan
            .rows
            .iter()
            .filter(|row| row.kind == StepKind::Decode)
            .count();
        let mut executed_decode_batch = 0;
        let mut executed_greedy_epilogue_batch = 0;
        let mut batched_decode_blocked = false;
        if decode_count > 1 {
            let (batch_results, actual_batch_rows, greedy_epilogue_rows) =
                self.execute_decode_batch(plan, running);
            executed_decode_batch = actual_batch_rows;
            executed_greedy_epilogue_batch = greedy_epilogue_rows;
            batched_decode_blocked = batch_results
                .iter()
                .any(|(_, result)| result.allocation_blocked);
            for (index, result) in batch_results {
                results[index] = Some(result);
            }
        }
        for (index, planned) in plan.rows.iter().enumerate() {
            if results[index].is_some() {
                continue;
            }
            let turn = running
                .iter_mut()
                .find(|turn| turn.seq_id == planned.seq_id)
                .expect("scheduler validated Gemma4 row");
            let result = match planned.kind {
                StepKind::Prefill => self.execute_prefill(planned, turn),
                StepKind::Decode => self.execute_decode(planned, turn),
            }
            .unwrap_or_else(|error| Self::fail(turn, planned, error));
            results[index] = Some(result);
        }
        let rows = results
            .into_iter()
            .map(|result| result.expect("every planned Gemma4 row executed"))
            .collect::<Vec<_>>();
        let scalar_decode_occupancy = plan
            .rows
            .iter()
            .zip(&rows)
            .filter(|(planned, result)| {
                planned.kind == StepKind::Decode && !result.allocation_blocked
            })
            .count();
        Ok(StepResult {
            rows,
            executed_decode_batch: if batched_decode_blocked {
                0
            } else if decode_count > 1 {
                executed_decode_batch
            } else {
                scalar_decode_occupancy
            },
            executed_greedy_epilogue_batch,
            rows_alloc_evicted: running
                .iter()
                .filter(|turn| turn.payload.allocation_failed)
                .count() as u32,
        })
    }
}

/// Scheduler-owned Gemma 4 state. Every sequence advances the full-attention
/// and sliding-window paged groups atomically. Media and draft requests are
/// ordered exclusive commands, so they can use their specialized target cores
/// without globally disabling ordinary text batching.
pub(crate) struct Gemma4SchedulerState {
    inner: Gemma4Inner,
    scheduler: Scheduler<ScheduledTurn, Gemma4Cmd, Gemma4Cmd>,
    pending: VecDeque<Gemma4Cmd>,
    owner_sequences: HashMap<String, SeqId>,
    owner_metadata: HashMap<String, Gemma4OwnerMetadata>,
    flat_owner_caches: HashMap<String, Vec<Gemma4LayerCache>>,
    next_seq_id: SeqId,
}

impl Gemma4SchedulerState {
    pub(crate) fn configured_capacity(inner: &Gemma4Inner) -> u32 {
        u32::try_from(scheduler_max_num_seqs(inner)).unwrap_or(u32::MAX)
    }

    pub(crate) fn new(inner: Gemma4Inner) -> Self {
        let max_num_seqs = Self::configured_capacity(&inner) as usize;
        Self {
            inner,
            scheduler: Scheduler::new(max_num_seqs, scheduler_max_batched_tokens())
                .expect("validated Gemma4 scheduler limits"),
            pending: VecDeque::new(),
            owner_sequences: HashMap::new(),
            owner_metadata: HashMap::new(),
            flat_owner_caches: HashMap::new(),
            next_seq_id: 1,
        }
    }

    pub(crate) fn force_serial() -> bool {
        std::env::var("MLX_SERVE_FORCE_SERIAL").is_ok_and(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
    }

    fn chat_parts(command: &ChatCmd) -> Option<(&[ChatMessage], &ChatConfig)> {
        match command {
            ChatCmd::SessionStart {
                messages, config, ..
            }
            | ChatCmd::SessionContinue {
                messages, config, ..
            }
            | ChatCmd::SessionContinueTool {
                messages, config, ..
            }
            | ChatCmd::StreamSessionStart {
                messages, config, ..
            }
            | ChatCmd::StreamSessionContinue {
                messages, config, ..
            }
            | ChatCmd::StreamSessionContinueTool {
                messages, config, ..
            } => Some((messages, config)),
            ChatCmd::ResetCaches { .. } | ChatCmd::ReleaseCacheOwner { .. } => None,
        }
    }

    fn owner_id(command: &ChatCmd) -> Option<&str> {
        Self::chat_parts(command)
            .and_then(|(_, config)| config.cache_owner_id.as_deref())
            .filter(|owner| !owner.is_empty())
    }

    fn is_session_start(command: &ChatCmd) -> bool {
        matches!(
            command,
            ChatCmd::SessionStart { .. } | ChatCmd::StreamSessionStart { .. }
        )
    }

    fn owner_lane(&self, command: &ChatCmd) -> Option<OwnerLane> {
        Self::chat_parts(command).map(|(_, config)| {
            if config.enable_mtp == Some(true) || self.inner.kv_cache_coordinator.is_none() {
                OwnerLane::Flat
            } else {
                OwnerLane::Paged
            }
        })
    }

    fn send_exclusive_error(command: ChatCmd, message: impl Into<String>) {
        let error = Error::from_reason(message.into());
        match command {
            ChatCmd::SessionStart {
                reply, cancelled, ..
            }
            | ChatCmd::SessionContinue {
                reply, cancelled, ..
            }
            | ChatCmd::SessionContinueTool {
                reply, cancelled, ..
            } => ScheduledReply::Sync(reply).send_error(error, cancelled.as_ref()),
            ChatCmd::StreamSessionStart {
                stream_tx,
                cancelled,
                ..
            }
            | ChatCmd::StreamSessionContinue {
                stream_tx,
                cancelled,
                ..
            }
            | ChatCmd::StreamSessionContinueTool {
                stream_tx,
                cancelled,
                ..
            } => ScheduledReply::Stream(stream_tx).send_error(error, cancelled.as_ref()),
            ChatCmd::ResetCaches { reply } => {
                let _ = reply.send(Err(error));
            }
            ChatCmd::ReleaseCacheOwner { reply, .. } => {
                let _ = reply.send(Err(error));
            }
        }
    }

    fn handle_exclusive_chat(&mut self, command: ChatCmd) {
        let Some(lane) = self.owner_lane(&command) else {
            handle_chat_cmd(&mut self.inner, command);
            return;
        };
        let Some(owner_id) = Self::owner_id(&command).map(str::to_owned) else {
            self.inner.select_ownerless_lane(lane == OwnerLane::Flat);
            handle_chat_cmd(&mut self.inner, command);
            return;
        };

        let start = Self::is_session_start(&command);
        if start {
            if let Err(error) = self.release_cache_owner_now(&owner_id) {
                Self::send_exclusive_error(command, error.reason);
                return;
            }
        } else {
            let matching_lane = match lane {
                OwnerLane::Paged => self.owner_sequences.contains_key(&owner_id),
                OwnerLane::Flat => self.flat_owner_caches.contains_key(&owner_id),
            };
            if !matching_lane {
                let conflicting_lane = match lane {
                    OwnerLane::Paged => self.flat_owner_caches.contains_key(&owner_id),
                    OwnerLane::Flat => self.owner_sequences.contains_key(&owner_id),
                };
                let message = if conflicting_lane {
                    "Gemma4 cannot switch a live cache owner between autoregressive/media and MTP/DSpark cache layouts; start a new session"
                } else {
                    "chat session continuation requires an initialized cache owner (call chatSessionStart first)"
                };
                Self::send_exclusive_error(command, message);
                return;
            }
        }

        let metadata = self
            .owner_metadata
            .get(&owner_id)
            .cloned()
            .unwrap_or_default();
        match lane {
            OwnerLane::Flat => {
                let caches = self.flat_owner_caches.remove(&owner_id);
                self.inner.install_flat_owner_caches(caches);
                self.inner.install_owner_metadata(metadata);
                handle_chat_cmd(&mut self.inner, command);
                let live = ChatBackend::has_live_session(&self.inner);
                let metadata = self.inner.owner_metadata();
                let caches = self.inner.take_flat_owner_caches();
                if live {
                    if let Some(caches) = caches {
                        self.flat_owner_caches.insert(owner_id.clone(), caches);
                    }
                    self.owner_metadata.insert(owner_id, metadata);
                } else {
                    self.owner_metadata.remove(&owner_id);
                    self.flat_owner_caches.remove(&owner_id);
                }
            }
            OwnerLane::Paged => {
                let seq_id = if let Some(&seq_id) = self.owner_sequences.get(&owner_id) {
                    seq_id
                } else {
                    let seq_id = self.next_seq_id;
                    self.next_seq_id = self.next_seq_id.saturating_add(1);
                    self.owner_sequences.insert(owner_id.clone(), seq_id);
                    seq_id
                };
                self.inner.set_active_paged_owner(seq_id);
                self.inner.install_owner_metadata(metadata);
                handle_chat_cmd(&mut self.inner, command);
                if ChatBackend::has_live_session(&self.inner) {
                    self.owner_metadata
                        .insert(owner_id, self.inner.owner_metadata());
                } else {
                    self.owner_metadata.remove(&owner_id);
                    self.owner_sequences.remove(&owner_id);
                }
            }
        }
    }

    fn scheduled_eligible(&self, command: &ChatCmd) -> bool {
        let Some((messages, config)) = Self::chat_parts(command) else {
            return false;
        };
        self.inner.kv_cache_coordinator.is_some()
            && Self::owner_id(command).is_some()
            && config.enable_mtp != Some(true)
            && engine::session::media_capabilities_from_messages(messages).is_empty()
    }

    fn release_cache_owner_now(&mut self, owner_id: &str) -> Result<()> {
        let paged_release = self
            .owner_sequences
            .remove(owner_id)
            .and_then(|seq_id| {
                self.inner
                    .kv_cache_coordinator
                    .as_mut()
                    .map(|coordinator| coordinator.release_request_all(seq_id))
            })
            .transpose();
        self.flat_owner_caches.remove(owner_id);
        self.owner_metadata.remove(owner_id);
        paged_release.map(|_| ()).map_err(Error::from_reason)
    }

    fn reply_error(response: ScheduledReply, cancelled: &AtomicBool, error: Error) {
        response.send_error(error, cancelled);
    }

    fn prepare_chat(&mut self, command: ChatCmd) -> Option<PreparedTurn> {
        let (messages, config, response, cancelled, turn_kind, guard_name, stream_guard_name) =
            match command {
                ChatCmd::SessionStart {
                    messages,
                    config,
                    reply,
                    cancelled,
                } => (
                    messages,
                    config,
                    ScheduledReply::Sync(reply),
                    cancelled,
                    engine::session::TurnKind::Start,
                    "chat_session_start",
                    None,
                ),
                ChatCmd::SessionContinue {
                    messages,
                    config,
                    reply,
                    cancelled,
                } => (
                    messages,
                    config,
                    ScheduledReply::Sync(reply),
                    cancelled,
                    engine::session::TurnKind::Continue,
                    "chat_session_continue",
                    None,
                ),
                ChatCmd::SessionContinueTool {
                    messages,
                    config,
                    reply,
                    cancelled,
                } => (
                    messages,
                    config,
                    ScheduledReply::Sync(reply),
                    cancelled,
                    engine::session::TurnKind::Continue,
                    "chat_session_continue_tool",
                    None,
                ),
                ChatCmd::StreamSessionStart {
                    messages,
                    config,
                    stream_tx,
                    cancelled,
                } => (
                    messages,
                    config,
                    ScheduledReply::Stream(stream_tx),
                    cancelled,
                    engine::session::TurnKind::Start,
                    "chat_stream_session_start",
                    Some("chat_stream_session_start"),
                ),
                ChatCmd::StreamSessionContinue {
                    messages,
                    config,
                    stream_tx,
                    cancelled,
                } => (
                    messages,
                    config,
                    ScheduledReply::Stream(stream_tx),
                    cancelled,
                    engine::session::TurnKind::Continue,
                    "chat_stream_session_continue",
                    Some("chat_stream_session_continue"),
                ),
                ChatCmd::StreamSessionContinueTool {
                    messages,
                    config,
                    stream_tx,
                    cancelled,
                } => (
                    messages,
                    config,
                    ScheduledReply::Stream(stream_tx),
                    cancelled,
                    engine::session::TurnKind::Continue,
                    "chat_stream_session_continue_tool",
                    Some("chat_stream_session_continue_tool"),
                ),
                ChatCmd::ResetCaches { .. } | ChatCmd::ReleaseCacheOwner { .. } => {
                    unreachable!("control commands do not enter Gemma4 preparation")
                }
            };
        let owner_id = config.cache_owner_id.clone().unwrap_or_default();
        self.inner
            .set_active_paged_owner(self.owner_sequences.get(&owner_id).copied().unwrap_or(0));
        self.inner.install_owner_metadata(
            self.owner_metadata
                .get(&owner_id)
                .cloned()
                .unwrap_or_default(),
        );
        if cancelled.load(Ordering::Relaxed) {
            let message = stream_guard_name.map_or_else(
                || engine::session::CHAT_SESSION_CANCELLED.to_string(),
                |name| format!("{name} cancelled before start"),
            );
            Self::reply_error(response, cancelled.as_ref(), Error::from_reason(message));
            return None;
        }
        if config.reuse_cache == Some(false) {
            Self::reply_error(
                response,
                cancelled.as_ref(),
                Error::from_reason(format!(
                    "{guard_name} requires reuse_cache=true (leave as None or set to true). The session API only makes sense with cache reuse enabled."
                )),
            );
            return None;
        }
        if turn_kind == engine::session::TurnKind::Continue
            && (self.inner.cached_token_history.is_empty()
                || !self.owner_sequences.get(&owner_id).is_some_and(|&seq_id| {
                    self.inner
                        .kv_cache_coordinator
                        .as_ref()
                        .is_some_and(|coordinator| coordinator.is_live_all(seq_id))
                }))
        {
            Self::reply_error(
                response,
                cancelled.as_ref(),
                Error::from_reason(format!(
                    "{guard_name} requires an initialized session (call chatSessionStart first)"
                )),
            );
            return None;
        }
        let mut admitted =
            match engine::session::admit_paged_turn(&mut self.inner, messages, config, turn_kind) {
                Ok(admitted) => admitted,
                Err(error) => {
                    Self::reply_error(response, cancelled.as_ref(), error);
                    return None;
                }
            };
        if admitted.plan.path() != engine::plan::TurnPath::Paged
            || !matches!(admitted.plan.decoder, DecoderPlan::Autoregressive)
        {
            Self::reply_error(
                response,
                cancelled.as_ref(),
                Error::from_reason("Gemma4 scheduler admitted a non-paged text turn"),
            );
            return None;
        }
        let prompt_tokens = admitted.tokens.len() as u32;
        let requested_max_new_tokens = admitted.params.max_new_tokens.max(0) as u32;
        let context = u32::try_from(self.inner.config.max_position_embeddings)
            .unwrap_or(1)
            .max(1);
        let max_new_tokens = match engine::scheduler::clamp_scheduled_output_tokens(
            prompt_tokens,
            requested_max_new_tokens,
            context,
        ) {
            Ok(value) => value,
            Err(error) => {
                Self::reply_error(response, cancelled.as_ref(), Error::from_reason(error));
                return None;
            }
        };
        admitted.params.max_new_tokens = max_new_tokens as i32;
        let (seq_id, newly_assigned) = if let Some(&seq_id) = self.owner_sequences.get(&owner_id) {
            (seq_id, false)
        } else {
            let seq_id = self.next_seq_id;
            self.next_seq_id = self.next_seq_id.saturating_add(1);
            self.owner_sequences.insert(owner_id.clone(), seq_id);
            (seq_id, true)
        };
        if self.scheduler.contains_seq(seq_id) {
            if newly_assigned {
                self.owner_sequences.remove(&owner_id);
            }
            Self::reply_error(
                response,
                cancelled.as_ref(),
                Error::from_reason("chat session already has an in-flight scheduled turn"),
            );
            return None;
        }
        Some(PreparedTurn {
            admitted,
            response,
            cancelled,
            owner_id,
            seq_id,
            newly_assigned,
        })
    }

    fn enqueue_prepared(&mut self, prepared: PreparedTurn) {
        let PreparedTurn {
            admitted,
            response,
            cancelled,
            owner_id,
            seq_id,
            newly_assigned,
        } = prepared;
        self.inner.set_active_paged_owner(seq_id);
        self.inner.install_owner_metadata(
            self.owner_metadata
                .get(&owner_id)
                .cloned()
                .unwrap_or_default(),
        );
        self.inner
            .set_turn_cancel_flag(Some(Arc::clone(&cancelled)));
        let cached_prefix = match self.inner.prepare_scheduled_text_request(
            seq_id,
            &admitted.tokens,
            admitted.params.cache_salt,
            true,
        ) {
            Ok(prefix) if prefix < admitted.tokens.len() as u32 => prefix,
            Ok(_) => {
                let reset = self
                    .inner
                    .kv_cache_coordinator
                    .as_mut()
                    .ok_or_else(|| "Gemma4 scheduled route lost its KV coordinator".to_string())
                    .and_then(|coordinator| coordinator.reset_scheduled_request(seq_id));
                match reset {
                    Ok(()) => 0,
                    Err(error) => {
                        if newly_assigned {
                            self.owner_sequences.remove(&owner_id);
                        }
                        self.inner.set_turn_cancel_flag(None);
                        Self::reply_error(response, cancelled.as_ref(), Error::from_reason(error));
                        return;
                    }
                }
            }
            Err(error) => {
                if newly_assigned {
                    self.owner_sequences.remove(&owner_id);
                }
                self.inner.set_turn_cancel_flag(None);
                Self::reply_error(response, cancelled.as_ref(), error);
                return;
            }
        };
        let suffix_len = admitted.tokens.len().saturating_sub(cached_prefix as usize);
        let prefix = Gemma4PrefixState {
            effective_cached_prefix_len: cached_prefix as usize,
            suffix_len,
            sliding_primed_prefix_len: cached_prefix,
            cache_salt: admitted.params.cache_salt,
            full_tokens: admitted.tokens.clone(),
        };
        let is_streaming = response.sink().is_some();
        let generation_stream = Stream::new(DeviceType::Gpu);
        let mut profiler = DecodeProfiler::new(
            self.inner
                .profiler_label(admitted.plan.is_delta, is_streaming),
            self.inner.family_name(),
        );
        profiler.set_prompt_tokens(suffix_len as u32);
        profiler.snapshot_memory_before();
        let payload = ScheduledTurn {
            owner_id,
            tokenizer: admitted.tokenizer,
            eos_id: admitted.eos_id,
            config: admitted.config,
            params: admitted.params,
            thinking: admitted.thinking,
            prompt_tokens: admitted.tokens.clone(),
            prefix,
            is_delta: admitted.plan.is_delta,
            reuse_cache: true,
            response,
            generated_tokens: Vec::new(),
            finish_reason: String::from("length"),
            reasoning_tracker: engine::penalties::ReasoningTracker::from_setup(
                &admitted.thinking,
                admitted.think_end_id,
            ),
            extra_eos_ids: self.inner.extra_eos_ids(),
            generation_start: Some(Instant::now()),
            first_token_instant: None,
            generation_stream,
            profiler,
            emitter: is_streaming.then(|| self.inner.stream_emitter()),
            stream_skip_special: self.inner.stream_skip_special_tokens(),
            decode_ids: Vec::new(),
            decode_prefix: String::new(),
            decode_prefix_index: 0,
            streamed_text_len: 0,
            last_is_reasoning: admitted.thinking.enabled,
            failure: None,
            allocation_failed: false,
            preemption_replay: None,
        };
        let mut pinned_prefill_breaks = Vec::new();
        let mut boundary = cached_prefix;
        while boundary < prompt_tokens_len(&payload.prompt_tokens) {
            boundary = boundary
                .saturating_add(scheduler_prefill_slice_tokens())
                .min(prompt_tokens_len(&payload.prompt_tokens));
            pinned_prefill_breaks.push(boundary);
        }
        pinned_prefill_breaks.extend(self.inner.scheduled_cold_anchor_rungs().into_iter().filter(
            |&rung| rung > cached_prefix && rung < prompt_tokens_len(&payload.prompt_tokens),
        ));
        pinned_prefill_breaks.sort_unstable();
        pinned_prefill_breaks.dedup();
        let turn = TurnState::new(
            seq_id,
            admitted.tokens,
            cached_prefix,
            pinned_prefill_breaks,
            Some(cancelled),
            payload,
        )
        .expect("Gemma4 scheduler turn must satisfy progress invariants");
        self.scheduler
            .enqueue_turn(turn)
            .expect("Gemma4 sequence uniqueness was validated before enqueue");
    }

    fn finish_completed(&mut self, mut turn: TurnState<ScheduledTurn>) {
        let cancelled = turn.cancelled.take().expect("Gemma4 turn cancel flag");
        self.inner.set_active_paged_owner(turn.seq_id);
        self.inner.install_owner_metadata(
            self.owner_metadata
                .get(&turn.payload.owner_id)
                .cloned()
                .unwrap_or_default(),
        );
        if let Some(error) = turn.payload.failure.take() {
            self.inner.abort_paged_turn();
            self.inner.set_turn_cancel_flag(None);
            self.owner_metadata.remove(&turn.payload.owner_id);
            self.owner_sequences.remove(&turn.payload.owner_id);
            turn.payload.response.send_error(error, cancelled.as_ref());
            return;
        }
        let sink = turn.payload.response.sink();
        let outcome = engine::paged_turn::finish_paged_turn(
            &mut self.inner,
            engine::paged_turn::FinishPagedTurnArgs {
                tokenizer: &turn.payload.tokenizer,
                params: &turn.payload.params,
                config: &turn.payload.config,
                thinking: turn.payload.thinking,
                is_delta: turn.payload.is_delta,
                reuse_cache: turn.payload.reuse_cache,
                prompt_tokens: &turn.payload.prompt_tokens,
                effective_cached_prefix_len: turn.payload.prefix.effective_cached_prefix_len,
                suffix_len: turn.payload.prefix.suffix_len,
                generated_tokens: &turn.payload.generated_tokens,
                finish_reason: std::mem::take(&mut turn.payload.finish_reason),
                generation_start: turn.payload.generation_start,
                first_token_instant: turn.payload.first_token_instant,
                reasoning_tokens: turn.payload.reasoning_tracker.reasoning_token_count(),
                profiler: &turn.payload.profiler,
                stream_skip_special: turn.payload.stream_skip_special,
                streamed_text_len: turn.payload.streamed_text_len,
                last_is_reasoning: turn.payload.last_is_reasoning,
                sink,
                emitter: turn.payload.emitter.take(),
            },
        );
        if outcome.is_ok() && turn.payload.reuse_cache {
            self.owner_metadata
                .insert(turn.payload.owner_id.clone(), self.inner.owner_metadata());
        } else if outcome.is_err() {
            self.owner_metadata.remove(&turn.payload.owner_id);
            self.owner_sequences.remove(&turn.payload.owner_id);
        }
        self.inner.set_turn_cancel_flag(None);
        match turn.payload.response {
            ScheduledReply::Sync(reply) => {
                let result = if cancelled.load(Ordering::Relaxed) {
                    Err(Error::from_reason(engine::session::CHAT_SESSION_CANCELLED))
                } else {
                    match outcome {
                        Ok(TurnOutput::Complete(result)) => Ok(*result),
                        Ok(TurnOutput::Streamed) => Err(Error::from_reason(
                            "Gemma4 scheduler returned streamed output on sync turn",
                        )),
                        Err(error) => Err(error),
                    }
                };
                let _ = reply.send(result);
            }
            ScheduledReply::Stream(stream) => {
                if let Err(error) = outcome {
                    ChunkSink::send(&stream, Err(error));
                }
            }
        }
    }

    fn handle_preempted(&mut self, mut turn: TurnState<ScheduledTurn>) {
        let prompt = turn.payload.prompt_tokens.clone();
        match install_preemption_replay(&mut turn, &prompt, scheduler_prefill_slice_tokens()) {
            Ok(replay) => turn.payload.preemption_replay = Some(replay),
            Err(error) => {
                turn.payload.failure = Some(Error::from_reason(error));
                self.finish_completed(turn);
                return;
            }
        }
        if let Some(coordinator) = self.inner.kv_cache_coordinator.as_mut() {
            let _ = coordinator.release_request_all(turn.seq_id);
        }
        self.scheduler
            .record_preemption_mode(engine::scheduler::PreemptionMode::Recompute);
        self.scheduler.prepend_preempted(turn);
    }

    fn try_resume_preempted(&mut self) {
        let Some(mut turn) = self.scheduler.take_preempted() else {
            return;
        };
        if turn
            .cancelled
            .as_ref()
            .is_some_and(|cancelled| cancelled.load(Ordering::Relaxed))
        {
            turn.payload.failure =
                Some(Error::from_reason(engine::session::CHAT_SESSION_CANCELLED));
            self.finish_completed(turn);
            return;
        }
        let Some(replay) = turn.payload.preemption_replay.as_mut() else {
            turn.payload.failure = Some(Error::from_reason(
                "Gemma4 preempted turn is missing replay state",
            ));
            self.finish_completed(turn);
            return;
        };
        let result = self.inner.prepare_scheduled_text_request(
            turn.seq_id,
            &replay.tokens,
            turn.payload.params.cache_salt,
            true,
        );
        match result {
            Ok(cached) => {
                replay.cached_prefix = cached;
                turn.num_computed_tokens = cached;
                turn.pinned_prefill_breaks.clear();
                let target = replay.tokens.len() as u32;
                let mut boundary = cached;
                while boundary < target {
                    boundary = boundary
                        .saturating_add(scheduler_prefill_slice_tokens())
                        .min(target);
                    turn.pinned_prefill_breaks.push(boundary);
                }
                turn.pinned_prefill_breaks.extend(
                    self.inner
                        .scheduled_cold_anchor_rungs()
                        .into_iter()
                        .filter(|&rung| rung > cached && rung < target),
                );
                turn.pinned_prefill_breaks.sort_unstable();
                turn.pinned_prefill_breaks.dedup();
                self.scheduler.ready_preempted(turn, false);
            }
            Err(error) if is_paged_allocation_blocked(&error.reason) => {
                self.scheduler.prepend_preempted(turn);
            }
            Err(error) => {
                turn.payload.failure = Some(error);
                self.finish_completed(turn);
            }
        }
    }

    fn reap_cancelled_waiters(&mut self) {
        for mut turn in self.scheduler.take_cancelled_waiters() {
            turn.payload.failure =
                Some(Error::from_reason(engine::session::CHAT_SESSION_CANCELLED));
            self.finish_completed(turn);
        }
    }

    pub(crate) fn drive(
        &mut self,
        receiver: &mut tokio::sync::mpsc::UnboundedReceiver<Gemma4Cmd>,
    ) -> LoopControl {
        if !self.scheduler.has_work() && self.pending.is_empty() {
            match receiver.blocking_recv() {
                Some(command) => self.pending.push_back(command),
                None => return LoopControl::Break,
            }
        }
        while let Ok(command) = receiver.try_recv() {
            self.pending.push_back(command);
        }
        self.reap_cancelled_waiters();

        while !self.scheduler.has_pending_control()
            && let Some(command) = self.pending.pop_front()
        {
            if Self::force_serial() {
                self.scheduler.enqueue_exclusive(command);
                break;
            }
            match command {
                Gemma4Cmd::Chat(chat)
                    if matches!(chat.as_ref(), ChatCmd::ReleaseCacheOwner { .. }) =>
                {
                    let ChatCmd::ReleaseCacheOwner { owner_id, reply } = *chat else {
                        unreachable!("guarded cache-owner release")
                    };
                    if self
                        .owner_sequences
                        .get(&owner_id)
                        .is_some_and(|&seq_id| self.scheduler.contains_seq(seq_id))
                    {
                        self.pending.push_front(Gemma4Cmd::Chat(Box::new(
                            ChatCmd::ReleaseCacheOwner { owner_id, reply },
                        )));
                        break;
                    }
                    let _ = reply.send(self.release_cache_owner_now(&owner_id));
                }
                Gemma4Cmd::Chat(chat) if matches!(chat.as_ref(), ChatCmd::ResetCaches { .. }) => {
                    self.scheduler.enqueue_barrier(Gemma4Cmd::Chat(chat));
                    break;
                }
                Gemma4Cmd::Chat(chat) if self.scheduled_eligible(chat.as_ref()) => {
                    if let Some(prepared) = self.prepare_chat(*chat) {
                        self.enqueue_prepared(prepared);
                    }
                }
                Gemma4Cmd::Chat(chat) => {
                    self.scheduler.enqueue_exclusive(Gemma4Cmd::Chat(chat));
                    break;
                }
                stats @ Gemma4Cmd::SchedulerStats { .. } => {
                    self.scheduler.enqueue_barrier(stats);
                    break;
                }
            }
        }

        let action = {
            let mut executor = Gemma4StepExecutor {
                inner: &mut self.inner,
            };
            self.scheduler.drive_once(&mut executor)
        };
        let scheduler_idle = matches!(&action, Ok(SchedulerAction::Idle));
        let mut may_resume_preempted = true;
        match action {
            Ok(SchedulerAction::Idle) => {}
            Ok(SchedulerAction::Exclusive(command) | SchedulerAction::Barrier(command)) => {
                if let Gemma4Cmd::SchedulerStats { reply } = command {
                    let _ = reply.send(Ok(self.scheduler.stats().to_js()));
                    return LoopControl::Continue;
                }
                let reset = matches!(
                    command,
                    Gemma4Cmd::Chat(ref chat)
                        if matches!(chat.as_ref(), ChatCmd::ResetCaches { .. })
                );
                match command {
                    Gemma4Cmd::Chat(chat) if reset => {
                        self.inner.set_active_paged_owner(0);
                        handle_chat_cmd(&mut self.inner, *chat);
                    }
                    Gemma4Cmd::Chat(chat) => self.handle_exclusive_chat(*chat),
                    Gemma4Cmd::SchedulerStats { .. } => {
                        unreachable!("scheduler stats returned above")
                    }
                }
                if reset {
                    self.owner_sequences.clear();
                    self.owner_metadata.clear();
                    self.flat_owner_caches.clear();
                }
            }
            Ok(SchedulerAction::Stepped {
                completed,
                preempted,
                ..
            }) => {
                for turn in completed {
                    self.finish_completed(turn);
                }
                if let Some(turn) = preempted {
                    self.handle_preempted(turn);
                    may_resume_preempted = false;
                }
            }
            Err(error) => panic!("Gemma4 scheduler invariant failure: {error:?}"),
        }
        if may_resume_preempted {
            self.try_resume_preempted();
        }
        if scheduler_idle && self.scheduler.has_work() {
            std::thread::sleep(Duration::from_millis(1));
        }
        LoopControl::Continue
    }
}

fn prompt_tokens_len(tokens: &[u32]) -> u32 {
    u32::try_from(tokens.len()).unwrap_or(u32::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_paged_inner() -> Gemma4Inner {
        let config = Gemma4Config {
            vocab_size: 32,
            hidden_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 32,
            intermediate_size: 128,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: true,
            max_position_embeddings: 64,
            sliding_window: 16,
            layer_types: vec![
                "sliding_attention".to_string(),
                "full_attention".to_string(),
            ],
            rope_theta: 1_000_000.0,
            rope_local_base_freq: 10_000.0,
            partial_rotary_factor: 0.5,
            paged_cache_memory_mb: Some(16),
            paged_block_size: Some(8),
            use_block_paged_cache: Some(true),
            ..Gemma4Config::default()
        };
        Gemma4Inner::new(config).expect("tiny Gemma4 scheduler fixture")
    }

    #[test]
    fn live_owner_cannot_continue_across_paged_and_flat_cache_layouts() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        let inner = tiny_paged_inner();
        assert!(
            Gemma4SchedulerState::configured_capacity(&inner) > 1,
            "shared full-attention blocks must not statically partition one max context per slot"
        );
        let mut state = Gemma4SchedulerState::new(inner);
        state.owner_sequences.insert("session-a".to_string(), 7);
        let config = ChatConfig {
            cache_owner_id: Some("session-a".to_string()),
            enable_mtp: Some(true),
            ..ChatConfig::default()
        };
        let (reply, result) = tokio::sync::oneshot::channel();
        state.handle_exclusive_chat(ChatCmd::SessionContinue {
            messages: Vec::new(),
            config,
            reply,
            cancelled: Arc::new(AtomicBool::new(false)),
        });
        let error = result
            .blocking_recv()
            .expect("mode-switch reply")
            .expect_err("live paged-to-draft continuation must fail closed");
        assert!(error.reason.contains("cannot switch a live cache owner"));
        assert_eq!(state.owner_sequences.get("session-a"), Some(&7));
        assert!(!state.flat_owner_caches.contains_key("session-a"));
    }

    #[test]
    fn releasing_an_owner_drops_both_physical_lane_registries_and_metadata() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        let mut state = Gemma4SchedulerState::new(tiny_paged_inner());
        state.owner_sequences.insert("session-a".to_string(), 7);
        state.inner.init_caches_sync().expect("flat caches");
        state.flat_owner_caches.insert(
            "session-a".to_string(),
            state
                .inner
                .take_flat_owner_caches()
                .expect("flat caches live"),
        );
        state.owner_metadata.insert(
            "session-a".to_string(),
            Gemma4OwnerMetadata {
                cached_token_history: vec![1, 2, 3],
                ..Gemma4OwnerMetadata::default()
            },
        );

        state
            .release_cache_owner_now("session-a")
            .expect("release owner");
        assert!(!state.owner_sequences.contains_key("session-a"));
        assert!(!state.flat_owner_caches.contains_key("session-a"));
        assert!(!state.owner_metadata.contains_key("session-a"));
    }
}
