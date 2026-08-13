//! Engine-owned continuous-batching lifecycle for hybrid paged-KV plus
//! recurrent-state model families.
//!
//! Plain text autoregressive turns share one scheduler step. Native MTP,
//! multimodal turns, raw generation, calibration, persistence, and training
//! remain ordered barriers and execute through the existing command handler.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use napi::bindgen_prelude::{Error, Result};

use crate::array::MxArray;
use crate::decode_profiler::DecodeProfiler;
use crate::engine::backend::{
    ChunkSink, PagedBackend, PagedPrefix, StreamEmitter, ThinkingSetup, TurnOutput,
};
use crate::engine::cmd::{ChatCmd, FromChatCmd};
use crate::engine::scheduler::{
    BlockTelemetry, PreemptionMode, PreemptionReplay, RowStepResult, Scheduler, SchedulerAction,
    SchedulerError, StepExecutor, StepKind, StepPlan, StepResult, TurnState,
    install_preemption_replay, is_paged_allocation_blocked,
};
use crate::engine::types::{ChatConfig, ChatResult, ChatStreamChunk};
use crate::engine::{self, SchedulerStatsJs};
use crate::model_thread::{LoopControl, ResponseTx, StreamTx};
use crate::sampling::{check_repetition_cutoff, sample};
use crate::stream::{DeviceType, Stream, StreamContext};
use crate::tokenizer::Qwen3Tokenizer;
use crate::transformer::paged_kv_cache_adapter::{PagedKVCacheAdapter, SeqId};

/// Uninhabited restore ticket used by cache managers whose prefix restore is
/// synchronous on the model thread.
pub(crate) enum NoRestoreTicket {}

pub(crate) enum ScheduledPrefixAdmission<P, R> {
    Ready(P),
    Waiting { provisional: P, restore: R },
}

pub(crate) struct ScheduledRestoreResult<P> {
    pub prefix: P,
    pub bytes_restored: u64,
    pub wait: Duration,
    pub materialized_blocks: u32,
    pub profiler_prefill_tokens: u32,
    pub extra_prefill_breaks: Vec<u32>,
}

#[derive(Clone, Copy)]
pub(crate) struct SchedulerCacheSnapshot {
    pub blocks: BlockTelemetry,
    pub bytes_per_block: u64,
}

pub(crate) struct SchedulerOwnerContext<'a, S> {
    pub owner_sequences: &'a mut HashMap<String, SeqId>,
    pub owner_states: &'a mut HashMap<String, S>,
    pub next_seq_id: &'a mut SeqId,
}

/// Command surface consumed by the engine-owned hybrid scheduler.
///
/// Families keep their complete command enum so raw generation, MTP,
/// multimodal, persistence, and training operations remain typed ordered
/// barriers. The scheduler only needs model-neutral views of chat and stats
/// commands; it never matches family variants itself.
pub(crate) trait HybridSchedulerCommand: FromChatCmd + Sized {
    fn as_chat(&self) -> Option<&ChatCmd>;
    fn into_chat(self) -> std::result::Result<ChatCmd, Self>;
    fn into_scheduler_stats(self) -> std::result::Result<ResponseTx<SchedulerStatsJs>, Self>;
}

/// Family capabilities required by the shared hybrid scheduler lifecycle.
///
/// This mirrors vLLM's scheduler/cache-manager boundary: the engine owns
/// request state and policy, while a model runner supplies cache access and
/// executes the planned prefill/decode work.
pub(crate) trait HybridSchedulerBackend: PagedBackend + Sized {
    type Command: HybridSchedulerCommand;
    type RestoreTicket;
    type OwnerState: Default;
    type StepExecutor<'a>: StepExecutor<ScheduledTurn<Self::PrefixState>, Error = Error>
    where
        Self: 'a;

    const SCHEDULER_NAME: &'static str;
    const ENABLED_BY_DEFAULT: bool = true;
    const CANCEL_PRECEDES_EOS: bool = false;
    const STREAM_EOS_TOKEN: bool = false;

    fn paged_adapter(&self) -> Option<&PagedKVCacheAdapter>;
    fn paged_adapter_mut(&mut self) -> Option<&mut PagedKVCacheAdapter>;
    fn scheduler_cache_available(&self) -> bool {
        self.paged_adapter().is_some()
    }
    fn scheduler_capacity(&self) -> usize {
        32
    }
    fn scheduler_block_size(&self) -> Result<u32> {
        self.paged_adapter()
            .map(PagedKVCacheAdapter::block_size)
            .ok_or_else(|| {
                Error::from_reason(format!(
                    "{} cache manager is unavailable",
                    Self::SCHEDULER_NAME
                ))
            })
    }
    fn scheduler_cache_snapshot(&self) -> Result<Option<SchedulerCacheSnapshot>> {
        let Some(adapter) = self.paged_adapter() else {
            return Ok(None);
        };
        let blocks = adapter.block_telemetry().map_err(Error::from_reason)?;
        let bytes_per_block = adapter.bytes_per_block().map_err(Error::from_reason)?;
        Ok(Some(SchedulerCacheSnapshot {
            blocks: BlockTelemetry {
                total_blocks: blocks.total_blocks,
                free_blocks: blocks.free_blocks,
                reclaimable_blocks: blocks.reclaimable_blocks,
                allocated_blocks: blocks.allocated_blocks,
            },
            bytes_per_block,
        }))
    }
    fn scheduler_materialized_blocks(&self, seq_id: SeqId) -> u32 {
        self.paged_adapter()
            .and_then(|adapter| adapter.block_table_for(seq_id))
            .map(|table| table.num_blocks() as u32)
            .unwrap_or(0)
    }
    fn scheduler_has_cold_tier(&self) -> bool {
        self.paged_adapter()
            .is_some_and(|adapter| adapter.cold_tier().is_some())
    }
    fn scheduler_prefill_slice_tokens(&self) -> u32 {
        scheduler_long_prefill_tokens()
    }
    fn reset_owner_on_session_start(&self) -> bool {
        false
    }
    fn release_scheduled_cache(&mut self, seq_id: SeqId) -> Result<()> {
        let Some(adapter) = self.paged_adapter_mut() else {
            return Ok(());
        };
        if adapter.block_table_for(seq_id).is_some() {
            adapter
                .release_request_for(seq_id)
                .map(|_| ())
                .map_err(Error::from_reason)?;
        }
        Ok(())
    }
    fn preempt_scheduled_cache(
        &mut self,
        seq_id: SeqId,
        cache_salt: u64,
        mode: PreemptionMode,
    ) -> Result<()> {
        let adapter = self.paged_adapter_mut().ok_or_else(|| {
            Error::from_reason(format!(
                "{} cache manager is unavailable during preemption",
                Self::SCHEDULER_NAME
            ))
        })?;
        adapter
            .register_full_blocks_for_reuse_for(
                seq_id,
                &[],
                cache_salt,
                mode == PreemptionMode::Ssd,
            )
            .and_then(|_| adapter.release_request_for(seq_id).map(|_| ()))
            .map_err(Error::from_reason)
    }
    fn max_position_embeddings(&self) -> i32;
    fn recurrent_state_bytes(&self) -> u64 {
        0
    }
    fn scheduled_recurrent_bytes(&self) -> u64 {
        0
    }
    fn has_scheduled_recurrent(&self, _seq_id: SeqId) -> bool {
        false
    }
    fn can_activate_scheduled_recurrent(&self, _seq_id: SeqId) -> bool {
        true
    }
    fn activate_scheduled_recurrent(&mut self, _seq_id: SeqId) -> Result<()> {
        Ok(())
    }
    fn activate_paged_seq(&mut self, seq_id: SeqId) -> Result<()>;
    fn park_active_scheduled_recurrent(&mut self) -> Result<()> {
        Ok(())
    }
    fn release_scheduled_recurrent_for(&mut self, _seq_id: SeqId) {}
    fn run_paged_decode_step_batched(&mut self, rows: &[(SeqId, u32)]) -> Result<MxArray>;
    fn replace_cached_token_history(&mut self, history: Vec<u32>);
    fn owner_tokens(state: &Self::OwnerState) -> &[u32];
    fn install_owner_state(&mut self, seq_id: SeqId, state: &Self::OwnerState) {
        let _ = seq_id;
        self.replace_cached_token_history(Self::owner_tokens(state).to_vec());
    }
    fn capture_owner_state(&mut self, seq_id: SeqId) -> Self::OwnerState;
    fn release_owner_resources(
        &mut self,
        seq_id: Option<SeqId>,
        _state: Option<&Self::OwnerState>,
    ) -> Result<()> {
        if let Some(seq_id) = seq_id {
            self.release_scheduled_cache(seq_id)?;
            self.release_scheduled_recurrent_for(seq_id);
        }
        Ok(())
    }
    fn build_scheduled_prefix(
        &self,
        base: &Self::PrefixState,
        effective_cached_prefix_len: usize,
        suffix_len: usize,
        full_tokens: Vec<u32>,
        first_chunk: bool,
    ) -> Self::PrefixState;
    fn run_scheduled_prefill_slice(
        &mut self,
        _seq_id: SeqId,
        source: &[u32],
        base: &Self::PrefixState,
        start: usize,
        end: usize,
        generation_stream: Stream,
        first_chunk: bool,
    ) -> Result<Option<MxArray>> {
        let prefix = self.build_scheduled_prefix(
            base,
            start,
            end.saturating_sub(start),
            source.to_vec(),
            first_chunk,
        );
        self.paged_prefill(&source[start..end], &prefix, generation_stream)
            .map(Some)
    }
    fn finish_scheduled_decode_batch(&mut self, _rows: &[(SeqId, u32)]) -> Result<()> {
        Ok(())
    }
    fn prepare_scheduled_prefix(
        &mut self,
        seq_id: SeqId,
        tokens: &[u32],
        _owner_history: &[u32],
        reuse_cache: bool,
        cache_salt: u64,
        block_size: u32,
    ) -> Result<ScheduledPrefixAdmission<Self::PrefixState, Self::RestoreTicket>> {
        self.activate_paged_seq(seq_id)?;
        self.prime_prefix_state(tokens, reuse_cache, block_size as usize, &[], cache_salt)
            .map(ScheduledPrefixAdmission::Ready)
    }
    fn poll_scheduled_restore(
        &mut self,
        _seq_id: SeqId,
        _restore: &mut Self::RestoreTicket,
        _prompt_tokens: &[u32],
        _owner_history: &[u32],
        _is_preemption_replay: bool,
    ) -> Result<Option<ScheduledRestoreResult<Self::PrefixState>>> {
        Ok(None)
    }
    fn restore_reserved_blocks(_restore: &Self::RestoreTicket) -> u32 {
        0
    }
    fn extra_prefill_breaks(&self, _prompt_tokens: u32, _cached_prefix: u32) -> Vec<u32> {
        Vec::new()
    }
    fn profiler_prefill_tokens(&self, prefix: &Self::PrefixState, _prompt_tokens: u32) -> u32 {
        prefix.suffix_len() as u32
    }
    fn step_executor(&mut self) -> Self::StepExecutor<'_>;
    fn execute_barrier(
        &mut self,
        command: Self::Command,
        owners: SchedulerOwnerContext<'_, Self::OwnerState>,
    );
}

/// Engine-owned scheduler policy. Model families declare execution/cache
/// capabilities; they do not proxy global scheduling configuration.
fn configured_scheduler_capacity(physical_capacity: usize) -> usize {
    static VALUE: OnceLock<usize> = OnceLock::new();
    (*VALUE.get_or_init(|| {
        std::env::var("MLX_SCHED_MAX_NUM_SEQS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|&value| value > 0)
            .unwrap_or(8)
            .min(32)
    }))
    .min(physical_capacity.max(1))
}

pub(crate) fn scheduler_max_num_seqs_for(physical_capacity: usize) -> usize {
    configured_scheduler_capacity(physical_capacity)
}

/// Public Qwen3.5 capacity query. The hybrid GDN row store has two live units;
/// other families pass their own physical capacity to the same engine policy.
pub(crate) fn scheduler_max_num_seqs() -> usize {
    configured_scheduler_capacity(crate::engine::recurrent_state::HYBRID_LIVE_STATE_UNITS)
}

fn scheduler_max_batched_tokens() -> u32 {
    static VALUE: OnceLock<u32> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("MLX_SCHED_MAX_BATCHED_TOKENS")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .filter(|&value| value > 0)
            .unwrap_or(2048)
    })
}

fn scheduler_long_prefill_tokens() -> u32 {
    static VALUE: OnceLock<u32> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("MLX_SCHED_LONG_PREFILL_TOKENS")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .filter(|&value| value > 0)
            .unwrap_or(2048)
    })
}

pub(crate) fn scheduler_per_seq_context() -> u32 {
    static VALUE: OnceLock<u32> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("MLX_PAGED_PER_SEQ_CTX")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .filter(|&value| value > 0)
            .unwrap_or(32_768)
    })
}

fn scheduler_watermark_fraction() -> f64 {
    static VALUE: OnceLock<f64> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("MLX_SCHED_WATERMARK_FRACTION")
            .ok()
            .and_then(|value| value.parse::<f64>().ok())
            .filter(|value| value.is_finite() && (0.0..=1.0).contains(value))
            .unwrap_or(0.05)
    })
}

fn scheduler_reserve_full_isl() -> bool {
    static VALUE: OnceLock<bool> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("MLX_SCHED_RESERVE_FULL_ISL").map_or(true, |value| {
            !matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "no" | "off"
            )
        })
    })
}

pub(crate) enum ScheduledReply {
    Sync(ResponseTx<ChatResult>),
    Stream(StreamTx<ChatStreamChunk>),
}

impl ScheduledReply {
    pub(crate) fn sink(&self) -> Option<&dyn ChunkSink> {
        match self {
            Self::Sync(_) => None,
            Self::Stream(stream) => Some(stream),
        }
    }

    pub(crate) fn send_error(self, error: Error, cancelled: &AtomicBool) {
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

pub(crate) struct ScheduledTurn<P> {
    pub(crate) owner_id: String,
    pub(crate) tokenizer: Arc<Qwen3Tokenizer>,
    pub(crate) eos_id: u32,
    pub(crate) config: ChatConfig,
    pub(crate) params: engine::params::ChatParams,
    pub(crate) thinking: ThinkingSetup,
    pub(crate) prompt_tokens: Vec<u32>,
    pub(crate) prefix: P,
    pub(crate) is_delta: bool,
    pub(crate) reuse_cache: bool,
    pub(crate) response: ScheduledReply,
    pub(crate) generated_tokens: Vec<u32>,
    pub(crate) finish_reason: String,
    pub(crate) reasoning_tracker: engine::penalties::ReasoningTracker,
    pub(crate) extra_eos_ids: Vec<u32>,
    pub(crate) generation_start: Option<Instant>,
    pub(crate) first_token_instant: Option<Instant>,
    pub(crate) generation_stream: Stream,
    pub(crate) profiler: crate::decode_profiler::DecodeProfiler,
    pub(crate) emitter: Option<Box<dyn StreamEmitter>>,
    pub(crate) stream_skip_special: bool,
    pub(crate) decode_ids: Vec<u32>,
    pub(crate) decode_prefix: String,
    pub(crate) decode_prefix_index: usize,
    pub(crate) streamed_text_len: usize,
    pub(crate) last_is_reasoning: bool,
    pub(crate) failure: Option<Error>,
    pub(crate) allocation_failed: bool,
    pub(crate) preemption_replay: Option<PreemptionReplay>,
}

struct PreparedTurn {
    admitted: engine::session::AdmittedPagedTurn,
    response: ScheduledReply,
    cancelled: Arc<AtomicBool>,
    owner_id: String,
    seq_id: SeqId,
    newly_assigned: bool,
    reservation_blocks: u32,
    block_size: u32,
}

struct PreparedDecodeRow {
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DeferredCleanupProgress {
    None,
    OwnerReleaseProcessed,
}

pub(crate) struct HybridStepExecutor<'a, B: HybridSchedulerBackend> {
    inner: &'a mut B,
}

impl<B: HybridSchedulerBackend> HybridStepExecutor<'_, B> {
    pub(crate) fn new(inner: &mut B) -> HybridStepExecutor<'_, B> {
        HybridStepExecutor { inner }
    }

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
        turn: &mut TurnState<ScheduledTurn<B::PrefixState>>,
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

    fn stream_token(
        turn: &mut TurnState<ScheduledTurn<B::PrefixState>>,
        token_id: u32,
        is_reasoning: bool,
    ) {
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

    fn execute_prefill(
        &mut self,
        row: &engine::scheduler::StepRow,
        turn: &mut TurnState<ScheduledTurn<B::PrefixState>>,
    ) -> Result<RowStepResult> {
        if row.cancel_snapshot {
            return Ok(Self::fail(
                turn,
                row,
                Error::from_reason(engine::session::CHAT_SESSION_CANCELLED),
            ));
        }
        let start = row.token_start as usize;
        let end = start.saturating_add(row.num_tokens as usize);
        let source_len = turn
            .payload
            .preemption_replay
            .as_ref()
            .map_or(turn.payload.prompt_tokens.len(), |replay| {
                replay.tokens.len()
            });
        if end > source_len {
            return Ok(Self::fail(
                turn,
                row,
                Error::from_reason(format!(
                    "{} scheduler prefill slice exceeds prompt",
                    B::SCHEDULER_NAME
                )),
            ));
        }
        if let Err(error) = self.inner.activate_paged_seq(row.seq_id) {
            return Ok(Self::fail(turn, row, error));
        }
        self.inner
            .set_turn_cancel_flag(turn.cancelled.as_ref().map(Arc::clone));
        let started = Instant::now();
        turn.payload.profiler.begin_prefill();
        let first_chunk = start
            == turn.payload.preemption_replay.as_ref().map_or(
                turn.payload.prefix.effective_cached_prefix_len(),
                |replay| replay.cached_prefix as usize,
            );
        let source = turn
            .payload
            .preemption_replay
            .as_ref()
            .map_or(turn.payload.prompt_tokens.as_slice(), |replay| {
                replay.tokens.as_slice()
            });
        let logits = match self.inner.run_scheduled_prefill_slice(
            row.seq_id,
            source,
            &turn.payload.prefix,
            start,
            end,
            turn.payload.generation_stream,
            first_chunk,
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
        if end < source_len {
            crate::array::synchronize_and_clear_cache();
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
            crate::array::synchronize_and_clear_cache();
            turn.payload.preemption_replay = None;
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
                Error::from_reason(format!(
                    "{} final prefill slice produced no logits",
                    B::SCHEDULER_NAME
                )),
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
        crate::array::synchronize_and_clear_cache();
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

    fn finish_decode_row(
        turn: &mut TurnState<ScheduledTurn<B::PrefixState>>,
        row: &PreparedDecodeRow,
    ) {
        turn.payload.profiler.mark_first_token();
        let is_reasoning = turn.payload.reasoning_tracker.observe_token(row.token_id);
        turn.payload.last_is_reasoning = is_reasoning;
        if B::CANCEL_PRECEDES_EOS && row.cancelled {
            turn.payload.finish_reason = String::from("cancelled");
        } else if row.stops_at_eos && !B::STREAM_EOS_TOKEN {
            turn.payload.finish_reason = String::from("stop");
        } else if row.cancelled {
            turn.payload.finish_reason = String::from("cancelled");
        } else {
            if turn.payload.response.sink().is_some() {
                Self::stream_token(turn, row.token_id, is_reasoning);
            }
            if row.stops_at_eos {
                turn.payload.finish_reason = String::from("stop");
            } else if let Some(reason) = row.repetition {
                turn.payload.finish_reason = reason.to_string();
            }
        }
    }

    fn sample_next(
        turn: &mut TurnState<ScheduledTurn<B::PrefixState>>,
        mut logits: MxArray,
    ) -> Result<u32> {
        let sampled = if turn.payload.reasoning_tracker.should_force_think_end() {
            MxArray::from_int32(
                &[turn.payload.reasoning_tracker.forced_token_id()? as i32],
                &[1],
            )?
        } else {
            turn.payload.profiler.begin("rep_penalty");
            logits = engine::penalties::apply_all_penalties(
                logits,
                &turn.token_history,
                &turn.payload.params,
            )?;
            turn.payload.profiler.end();
            turn.payload.profiler.begin("sample");
            let sampled = sample(&logits, turn.payload.params.sampling_config)?;
            turn.payload.profiler.end();
            sampled
        };
        turn.payload.profiler.begin("schedule_eval");
        MxArray::async_eval_arrays(&[&sampled, &logits]);
        turn.payload.profiler.end();
        sampled.eval();
        Ok(sampled.item_at_int32(0)? as u32)
    }

    fn prepare_decode_rows(
        plan: &StepPlan,
        running: &mut [TurnState<ScheduledTurn<B::PrefixState>>],
    ) -> Result<(
        Vec<PreparedDecodeRow>,
        Vec<(usize, RowStepResult)>,
        Vec<(SeqId, u32)>,
    )> {
        let mut work = Vec::new();
        let mut early = Vec::new();
        let mut batch = Vec::new();
        for (plan_index, planned) in plan
            .rows
            .iter()
            .enumerate()
            .filter(|(_, row)| row.kind == StepKind::Decode)
        {
            let turn = running
                .iter_mut()
                .find(|turn| turn.seq_id == planned.seq_id)
                .ok_or_else(|| {
                    Error::from_reason(format!(
                        "{} scheduler planned missing decode sequence {}",
                        B::SCHEDULER_NAME,
                        planned.seq_id
                    ))
                })?;
            let Some(&token_id) = turn.token_history.last() else {
                early.push((
                    plan_index,
                    Self::fail(
                        turn,
                        planned,
                        Error::from_reason("scheduler decode row has no current token"),
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
            let terminal = stops_at_eos || planned.cancel_snapshot || repetition.is_some();
            let at_length = turn.payload.generated_tokens.len()
                >= turn.payload.params.max_new_tokens.max(0) as usize;
            // GDN state is not rewindable. A terminal/length token is never
            // forwarded merely to materialize K/V.
            let batch_index = (!terminal && !at_length).then(|| {
                let index = batch.len();
                batch.push((planned.seq_id, token_id));
                index
            });
            work.push(PreparedDecodeRow {
                plan_index,
                seq_id: planned.seq_id,
                token_id,
                terminal,
                at_length,
                stops_at_eos,
                cancelled: planned.cancel_snapshot,
                repetition,
                batch_index,
            });
        }
        Ok((work, early, batch))
    }

    fn execute_decode_rows(
        &mut self,
        plan: &StepPlan,
        running: &mut [TurnState<ScheduledTurn<B::PrefixState>>],
    ) -> Result<(Vec<(usize, RowStepResult)>, usize, usize)> {
        let (work, mut results, batch_rows) = Self::prepare_decode_rows(plan, running)?;
        let executed_decode_batch = batch_rows.len();
        crate::array::maybe_clear_cache_for_paged_step(plan.global_step as i32);
        for row in &work {
            if row.batch_index.is_some() {
                let turn = running
                    .iter_mut()
                    .find(|turn| turn.seq_id == row.seq_id)
                    .ok_or_else(|| {
                        Error::from_reason(format!(
                            "{} decode sequence {} disappeared before forward",
                            B::SCHEDULER_NAME,
                            row.seq_id
                        ))
                    })?;
                turn.payload.profiler.begin("forward");
            }
        }
        let logits = if batch_rows.is_empty() {
            Ok(None)
        } else {
            let _stream_context = StreamContext::new(Stream::default(DeviceType::Gpu));
            self.inner
                .run_paged_decode_step_batched(&batch_rows)
                .and_then(|logits| {
                    self.inner.finish_scheduled_decode_batch(&batch_rows)?;
                    Ok(Some(logits))
                })
        };
        for row in &work {
            if row.batch_index.is_some() {
                let turn = running
                    .iter_mut()
                    .find(|turn| turn.seq_id == row.seq_id)
                    .ok_or_else(|| {
                        Error::from_reason(format!(
                            "{} decode sequence {} disappeared after forward",
                            B::SCHEDULER_NAME,
                            row.seq_id
                        ))
                    })?;
                turn.payload.profiler.end();
            }
        }
        let greedy_wave = work
            .iter()
            .filter(|row| row.batch_index.is_some())
            .map(|row| {
                let turn = running
                    .iter()
                    .find(|turn| turn.seq_id == row.seq_id)
                    .ok_or_else(|| {
                        Error::from_reason(format!(
                            "{} decode sequence {} disappeared before sampling",
                            B::SCHEDULER_NAME,
                            row.seq_id
                        ))
                    })?;
                Ok((
                    &turn.payload.params,
                    turn.payload.reasoning_tracker.force_think_end_pending(),
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let greedy_tokens: std::result::Result<Option<Vec<u32>>, String> = match &logits {
            Ok(Some(logits)) if engine::batch_sampling::can_batch_greedy_wave(greedy_wave) => Ok(
                engine::batch_sampling::batch_greedy_tokens_or_fallback(logits),
            ),
            _ => Ok(None),
        };
        let executed_greedy_epilogue_batch = greedy_tokens
            .as_ref()
            .ok()
            .and_then(Option::as_ref)
            .map_or(0, Vec::len);
        for row in work {
            let planned = plan.rows.get(row.plan_index).ok_or_else(|| {
                Error::from_reason(format!(
                    "{} decode result index {} exceeds the step plan",
                    B::SCHEDULER_NAME,
                    row.plan_index
                ))
            })?;
            let turn = running
                .iter_mut()
                .find(|turn| turn.seq_id == row.seq_id)
                .ok_or_else(|| {
                    Error::from_reason(format!(
                        "{} decode sequence {} disappeared before result handling",
                        B::SCHEDULER_NAME,
                        row.seq_id
                    ))
                })?;
            let greedy_next = match (&greedy_tokens, row.batch_index) {
                (Ok(Some(tokens)), Some(index)) => Some(*tokens.get(index).ok_or_else(|| {
                    Error::from_reason(format!(
                        "{} greedy batch omitted row {index} of {}",
                        B::SCHEDULER_NAME,
                        tokens.len()
                    ))
                })?),
                (Err(error), Some(_)) => {
                    results.push((
                        row.plan_index,
                        Self::fail(turn, planned, Error::from_reason(error.clone())),
                    ));
                    continue;
                }
                _ => None,
            };
            let row_logits = match (&logits, row.batch_index) {
                (Err(error), Some(_)) if is_paged_allocation_blocked(&error.reason) => {
                    let rolled_back = turn.payload.generated_tokens.pop();
                    debug_assert_eq!(rolled_back, Some(row.token_id));
                    results.push((row.plan_index, Self::blocked(planned)));
                    continue;
                }
                (Err(error), Some(_)) => {
                    results.push((
                        row.plan_index,
                        Self::fail(turn, planned, Error::from_reason(error.reason.clone())),
                    ));
                    continue;
                }
                (Ok(Some(_)), Some(_)) if greedy_next.is_some() => None,
                (Ok(Some(logits)), Some(index)) => match logits
                    .slice_axis(0, index as i64, index as i64 + 1)
                    .and_then(|row| row.squeeze(Some(&[1])))
                {
                    Ok(logits) => Some(logits),
                    Err(error) => {
                        results.push((row.plan_index, Self::fail(turn, planned, error)));
                        continue;
                    }
                },
                _ => None,
            };
            let next = if greedy_next.is_some() {
                greedy_next
            } else {
                match row_logits {
                    Some(logits) => match Self::sample_next(turn, logits) {
                        Ok(token) => Some(token),
                        Err(error) => {
                            results.push((row.plan_index, Self::fail(turn, planned, error)));
                            continue;
                        }
                    },
                    None => None,
                }
            };
            turn.payload.profiler.step();
            Self::finish_decode_row(turn, &row);
            let finished = row.terminal || row.at_length;
            if finished {
                turn.payload.profiler.snapshot_memory_after();
                turn.payload.profiler.report();
            }
            results.push((
                row.plan_index,
                RowStepResult {
                    seq_id: row.seq_id,
                    num_computed_tokens: planned.num_tokens,
                    generated_token: (!finished).then_some(next).flatten(),
                    finished,
                    allocation_blocked: false,
                    prefill_micros: 0,
                },
            ));
        }
        Ok((
            results,
            executed_decode_batch,
            executed_greedy_epilogue_batch,
        ))
    }
}

impl<B: HybridSchedulerBackend> StepExecutor<ScheduledTurn<B::PrefixState>>
    for HybridStepExecutor<'_, B>
{
    type Error = Error;

    fn execute(
        &mut self,
        plan: &StepPlan,
        running: &mut [TurnState<ScheduledTurn<B::PrefixState>>],
    ) -> std::result::Result<StepResult, Self::Error> {
        let mut rows = Vec::with_capacity(plan.rows.len());
        rows.resize_with(plan.rows.len(), || None);
        let (decode_results, executed_decode_batch, executed_greedy_epilogue_batch) =
            self.execute_decode_rows(plan, running)?;
        let batched_decode_blocked = decode_results
            .iter()
            .any(|(_, result)| result.allocation_blocked);
        for (index, result) in decode_results {
            rows[index] = Some(result);
        }
        for (index, planned) in plan.rows.iter().enumerate() {
            if rows[index].is_some() {
                continue;
            }
            let turn = running
                .iter_mut()
                .find(|turn| turn.seq_id == planned.seq_id)
                .ok_or_else(|| {
                    Error::from_reason(format!(
                        "{} scheduler planned missing sequence {}",
                        B::SCHEDULER_NAME,
                        planned.seq_id
                    ))
                })?;
            rows[index] = Some(
                self.execute_prefill(planned, turn)
                    .unwrap_or_else(|error| Self::fail(turn, planned, error)),
            );
        }
        Ok(StepResult {
            rows: rows
                .into_iter()
                .enumerate()
                .map(|(index, row)| {
                    row.ok_or_else(|| {
                        Error::from_reason(format!(
                            "{} scheduler produced no result for plan row {index}",
                            B::SCHEDULER_NAME
                        ))
                    })
                })
                .collect::<Result<Vec<_>>>()?,
            executed_decode_batch: if batched_decode_blocked {
                0
            } else {
                executed_decode_batch
            },
            executed_greedy_epilogue_batch: if batched_decode_blocked {
                0
            } else {
                executed_greedy_epilogue_batch
            },
            rows_alloc_evicted: running
                .iter()
                .filter(|turn| turn.payload.allocation_failed)
                .count() as u32,
        })
    }
}

pub(crate) struct HybridSchedulerState<B: HybridSchedulerBackend> {
    pub(crate) inner: B,
    enabled: bool,
    scheduler: Scheduler<ScheduledTurn<B::PrefixState>, B::Command, B::Command>,
    pending: VecDeque<B::Command>,
    prepared_waiting: Option<Box<PreparedTurn>>,
    pending_restores: HashMap<SeqId, B::RestoreTicket>,
    pub(crate) owner_sequences: HashMap<String, SeqId>,
    pub(crate) owner_states: HashMap<String, B::OwnerState>,
    next_seq_id: SeqId,
}

impl<B: HybridSchedulerBackend> HybridSchedulerState<B> {
    pub(crate) fn continuous_batching_enabled() -> bool {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var("MLX_CONTINUOUS_BATCHING").is_ok_and(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
        })
    }

    pub(crate) fn new(inner: B) -> Result<Self> {
        let enabled = inner.scheduler_cache_available()
            && (B::ENABLED_BY_DEFAULT || Self::continuous_batching_enabled());
        let max_num_seqs = configured_scheduler_capacity(inner.scheduler_capacity());
        let max_batched_tokens = scheduler_max_batched_tokens();
        Ok(Self {
            inner,
            enabled,
            scheduler: Scheduler::new(max_num_seqs, max_batched_tokens)
                .map_err(Error::from_reason)?,
            pending: VecDeque::new(),
            prepared_waiting: None,
            pending_restores: HashMap::new(),
            owner_sequences: HashMap::new(),
            owner_states: HashMap::new(),
            next_seq_id: 1,
        })
    }

    pub(crate) fn force_serial() -> bool {
        std::env::var("MLX_SERVE_FORCE_SERIAL").is_ok_and(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
    }

    fn chat_config(command: &ChatCmd) -> Option<&ChatConfig> {
        match command {
            ChatCmd::SessionStart { config, .. }
            | ChatCmd::SessionContinue { config, .. }
            | ChatCmd::SessionContinueTool { config, .. }
            | ChatCmd::StreamSessionStart { config, .. }
            | ChatCmd::StreamSessionContinue { config, .. }
            | ChatCmd::StreamSessionContinueTool { config, .. } => Some(config),
            ChatCmd::ResetCaches { .. } | ChatCmd::ReleaseCacheOwner { .. } => None,
        }
    }

    fn chat_has_explicit_owner(command: &ChatCmd) -> bool {
        Self::chat_config(command)
            .and_then(|config| config.cache_owner_id.as_deref())
            .is_some_and(|owner| !owner.is_empty())
    }

    fn chat_requires_legacy_owner_drain(command: &ChatCmd) -> bool {
        !matches!(command, ChatCmd::ReleaseCacheOwner { .. })
            && !Self::chat_has_explicit_owner(command)
    }

    fn chat_requires_barrier(&self, command: &ChatCmd) -> bool {
        if Self::chat_config(command).is_some_and(|config| config.enable_mtp == Some(true)) {
            return true;
        }
        let (messages, is_continue) = match command {
            ChatCmd::SessionStart { messages, .. }
            | ChatCmd::StreamSessionStart { messages, .. } => (messages, false),
            ChatCmd::SessionContinue { messages, .. }
            | ChatCmd::SessionContinueTool { messages, .. }
            | ChatCmd::StreamSessionContinue { messages, .. }
            | ChatCmd::StreamSessionContinueTool { messages, .. } => (messages, true),
            ChatCmd::ResetCaches { .. } | ChatCmd::ReleaseCacheOwner { .. } => return true,
        };
        let has_media = messages.iter().any(|message| {
            message
                .images
                .as_ref()
                .is_some_and(|images| !images.is_empty())
                || message
                    .audio
                    .as_ref()
                    .is_some_and(|audio| !audio.is_empty())
        });
        let owner_known = Self::chat_config(command)
            .and_then(|config| config.cache_owner_id.as_deref())
            .is_some_and(|owner| self.owner_sequences.contains_key(owner));
        has_media || (is_continue && !owner_known)
    }

    fn cache_owner_release_blocked(&self, owner_id: &str) -> bool {
        let Some(&seq_id) = self.owner_sequences.get(owner_id) else {
            return false;
        };
        self.scheduler.contains_seq(seq_id)
            || self
                .prepared_waiting
                .as_ref()
                .is_some_and(|prepared| prepared.owner_id == owner_id)
    }

    fn release_cache_owner_with<F>(&mut self, owner_id: &str, release: F) -> Result<()>
    where
        F: FnOnce(&mut B, Option<SeqId>, Option<&B::OwnerState>) -> Result<()>,
    {
        let seq_id = self.owner_sequences.get(owner_id).copied();
        release(&mut self.inner, seq_id, self.owner_states.get(owner_id))?;
        if let Some(seq_id) = seq_id {
            self.pending_restores.remove(&seq_id);
        }
        self.owner_sequences.remove(owner_id);
        self.owner_states.remove(owner_id);
        Ok(())
    }

    pub(crate) fn release_cache_owner_now(&mut self, owner_id: &str) -> Result<()> {
        self.release_cache_owner_with(owner_id, B::release_owner_resources)
    }

    /// Run only owner cleanup that can make a memory-blocked prepared
    /// turn admissible. Ordinary requests remain FIFO in `pending`; an idle
    /// owner's release is independent of them. Global resets stay in FIFO:
    /// overtaking the prepared request would change which turns the reset
    /// orders after.
    fn process_deferred_cleanup(&mut self, blocked_owner_id: &str) -> DeferredCleanupProgress {
        let mut release_index = None;
        let mut owners_with_earlier_turns = HashSet::new();
        for (index, command) in self.pending.iter().enumerate() {
            match command.as_chat() {
                Some(ChatCmd::ReleaseCacheOwner { owner_id, .. })
                    if owner_id != blocked_owner_id
                        && !owners_with_earlier_turns.contains(owner_id)
                        && !self.cache_owner_release_blocked(owner_id) =>
                {
                    release_index = Some(index);
                    break;
                }
                // A reset or family-specific barrier orders every later
                // command after itself, so cleanup must not cross it.
                Some(ChatCmd::ResetCaches { .. }) | None => break,
                Some(chat) => {
                    if let Some(owner_id) = Self::chat_config(chat)
                        .and_then(|config| config.cache_owner_id.as_deref())
                        .filter(|owner_id| !owner_id.is_empty())
                    {
                        owners_with_earlier_turns.insert(owner_id.to_owned());
                    }
                }
            }
        }
        let Some(index) = release_index else {
            return DeferredCleanupProgress::None;
        };
        let Some(command) = self.pending.remove(index) else {
            return DeferredCleanupProgress::None;
        };
        match command.into_chat() {
            Ok(ChatCmd::ReleaseCacheOwner { owner_id, reply }) => {
                let _ = reply.send(self.release_cache_owner_now(&owner_id));
                DeferredCleanupProgress::OwnerReleaseProcessed
            }
            Ok(chat) => {
                self.pending.insert(index, B::Command::from_chat(chat));
                DeferredCleanupProgress::None
            }
            Err(command) => {
                self.pending.insert(index, command);
                DeferredCleanupProgress::None
            }
        }
    }

    fn prepare_chat(&mut self, command: ChatCmd) -> Option<Box<PreparedTurn>> {
        let (messages, config, response, cancelled, kind, guard) = match command {
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
            ),
            ChatCmd::SessionContinue {
                messages,
                config,
                reply,
                cancelled,
            }
            | ChatCmd::SessionContinueTool {
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
            ),
            ChatCmd::StreamSessionContinue {
                messages,
                config,
                stream_tx,
                cancelled,
            }
            | ChatCmd::StreamSessionContinueTool {
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
            ),
            ChatCmd::ResetCaches { reply } => {
                self.scheduler
                    .enqueue_barrier(B::Command::from_chat(ChatCmd::ResetCaches { reply }));
                return None;
            }
            ChatCmd::ReleaseCacheOwner { .. } => {
                return None;
            }
        };
        if cancelled.load(Ordering::Relaxed) {
            response.send_error(
                Error::from_reason(engine::session::CHAT_SESSION_CANCELLED),
                cancelled.as_ref(),
            );
            return None;
        }
        if config.reuse_cache == Some(false) {
            response.send_error(
                Error::from_reason(format!(
                    "{guard} requires reuse_cache=true (leave as None or set to true). The session API only makes sense with cache reuse enabled."
                )),
                cancelled.as_ref(),
            );
            return None;
        }
        let owner_id = config.cache_owner_id.clone().unwrap_or_default();
        if kind == engine::session::TurnKind::Start && self.inner.reset_owner_on_session_start() {
            if self.cache_owner_release_blocked(&owner_id) {
                response.send_error(
                    Error::from_reason("chat session already has an in-flight scheduled turn"),
                    cancelled.as_ref(),
                );
                return None;
            }
            if let Err(error) = self.release_cache_owner_now(&owner_id) {
                response.send_error(error, cancelled.as_ref());
                return None;
            }
        }
        if kind == engine::session::TurnKind::Continue && !self.owner_states.contains_key(&owner_id)
        {
            response.send_error(
                Error::from_reason(format!(
                    "{guard} requires an initialized session (call chatSessionStart first)"
                )),
                cancelled.as_ref(),
            );
            return None;
        }
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
            response.send_error(
                Error::from_reason("chat session already has an in-flight scheduled turn"),
                cancelled.as_ref(),
            );
            return None;
        }
        let owner_state = self.owner_states.entry(owner_id.clone()).or_default();
        self.inner.install_owner_state(seq_id, owner_state);
        let mut admitted =
            match engine::session::admit_paged_turn(&mut self.inner, messages, config, kind) {
                Ok(admitted) => admitted,
                Err(error) => {
                    if newly_assigned {
                        self.owner_sequences.remove(&owner_id);
                        self.owner_states.remove(&owner_id);
                    }
                    response.send_error(error, cancelled.as_ref());
                    return None;
                }
            };
        if admitted.plan.path() != engine::plan::TurnPath::Paged
            || !admitted.images.is_empty()
            || !admitted.audio.is_empty()
            || admitted.params.enable_mtp
        {
            if newly_assigned {
                self.owner_sequences.remove(&owner_id);
                self.owner_states.remove(&owner_id);
            }
            response.send_error(
                Error::from_reason(format!(
                    "{} scheduler only admits plain text paged autoregressive turns",
                    B::SCHEDULER_NAME
                )),
                cancelled.as_ref(),
            );
            return None;
        }
        let prompt_tokens = admitted.tokens.len() as u32;
        let requested_max_new_tokens = admitted.params.max_new_tokens.max(0) as u32;
        let trained_context = u32::try_from(self.inner.max_position_embeddings())
            .unwrap_or(1)
            .max(1);
        let context = trained_context.min(scheduler_per_seq_context());
        let max_new_tokens = match engine::scheduler::clamp_scheduled_output_tokens(
            prompt_tokens,
            requested_max_new_tokens,
            context,
        ) {
            Ok(max_new_tokens) => max_new_tokens,
            Err(error) => {
                if newly_assigned {
                    self.owner_sequences.remove(&owner_id);
                    self.owner_states.remove(&owner_id);
                }
                response.send_error(Error::from_reason(error), cancelled.as_ref());
                return None;
            }
        };
        admitted.params.max_new_tokens = max_new_tokens as i32;
        let requested_tokens =
            engine::scheduler::scheduled_materialized_tokens(prompt_tokens, max_new_tokens);
        if !self.inner.scheduler_cache_available() {
            if newly_assigned {
                self.owner_sequences.remove(&owner_id);
                self.owner_states.remove(&owner_id);
            }
            response.send_error(
                Error::from_reason(format!(
                    "{} cache manager disappeared during admission",
                    B::SCHEDULER_NAME
                )),
                cancelled.as_ref(),
            );
            return None;
        }
        let block_size = match self.inner.scheduler_block_size() {
            Ok(block_size) => block_size,
            Err(error) => {
                if newly_assigned {
                    self.owner_sequences.remove(&owner_id);
                    self.owner_states.remove(&owner_id);
                }
                response.send_error(error, cancelled.as_ref());
                return None;
            }
        };
        let full_blocks = requested_tokens.div_ceil(block_size);
        let reservation_blocks = if scheduler_reserve_full_isl() {
            full_blocks
        } else {
            prompt_tokens
                .div_ceil(block_size)
                .saturating_add(u32::from(max_new_tokens != 0))
                .min(full_blocks)
        };
        Some(Box::new(PreparedTurn {
            admitted,
            response,
            cancelled,
            owner_id,
            seq_id,
            newly_assigned,
            reservation_blocks,
            block_size,
        }))
    }

    fn cleanup_rejected_prepared(&mut self, prepared: &PreparedTurn) {
        if prepared.newly_assigned {
            self.owner_states.remove(&prepared.owner_id);
            self.owner_sequences.remove(&prepared.owner_id);
            self.inner.release_scheduled_recurrent_for(prepared.seq_id);
        }
    }

    /// Keep the two-unit residency cap as a queueing boundary. An idle warm
    /// row may be discarded because its owner history can rebuild the exact
    /// GDN state through the ordinary prefix-prime path; a scheduler-owned row
    /// is never evicted here.
    fn ensure_recurrent_slot(&mut self, seq_id: SeqId) -> bool {
        if self.inner.can_activate_scheduled_recurrent(seq_id) {
            return true;
        }
        let idle_victim = self
            .owner_sequences
            .values()
            .copied()
            .filter(|&candidate| {
                candidate != seq_id
                    && self.inner.has_scheduled_recurrent(candidate)
                    && !self.scheduler.contains_seq(candidate)
            })
            .min();
        let Some(victim) = idle_victim else {
            return false;
        };
        self.inner.release_scheduled_recurrent_for(victim);
        self.inner.can_activate_scheduled_recurrent(seq_id)
    }

    fn admit_prepared(
        &mut self,
        prepared: Box<PreparedTurn>,
    ) -> std::result::Result<Option<TurnState<ScheduledTurn<B::PrefixState>>>, Box<PreparedTurn>>
    {
        if prepared.cancelled.load(Ordering::Relaxed) {
            self.cleanup_rejected_prepared(&prepared);
            let PreparedTurn {
                response,
                cancelled,
                ..
            } = *prepared;
            response.send_error(
                Error::from_reason(engine::session::CHAT_SESSION_CANCELLED),
                cancelled.as_ref(),
            );
            return Ok(None);
        }
        let cache_snapshot = match self.inner.scheduler_cache_snapshot() {
            Ok(snapshot) => snapshot,
            Err(error) => {
                self.cleanup_rejected_prepared(&prepared);
                prepared
                    .response
                    .send_error(error, prepared.cancelled.as_ref());
                return Ok(None);
            }
        };
        if cache_snapshot
            .is_some_and(|snapshot| prepared.reservation_blocks > snapshot.blocks.total_blocks)
        {
            self.cleanup_rejected_prepared(&prepared);
            let total_blocks = cache_snapshot.map_or(0, |snapshot| snapshot.blocks.total_blocks);
            prepared.response.send_error(
                Error::from_reason(format!(
                    "context_length_exceeded: request requires {} paged blocks but the pool has {}",
                    prepared.reservation_blocks, total_blocks
                )),
                prepared.cancelled.as_ref(),
            );
            return Ok(None);
        }
        if !self.ensure_recurrent_slot(prepared.seq_id) {
            return Err(prepared);
        }
        let candidate_state_bytes = if self.inner.has_scheduled_recurrent(prepared.seq_id) {
            0
        } else {
            self.inner.recurrent_state_bytes()
        };
        if let Some(snapshot) = cache_snapshot {
            let decision = self.scheduler.try_reserve_memory(
                engine::scheduler::MemoryTelemetry {
                    capacity_bytes: u64::from(snapshot.blocks.total_blocks)
                        .saturating_mul(snapshot.bytes_per_block),
                    free_bytes: u64::from(snapshot.blocks.free_blocks)
                        .saturating_mul(snapshot.bytes_per_block),
                    reclaimable_bytes: u64::from(snapshot.blocks.reclaimable_blocks)
                        .saturating_mul(snapshot.bytes_per_block),
                },
                prepared.reservation_blocks,
                snapshot.bytes_per_block,
                self.inner.scheduled_recurrent_bytes(),
                candidate_state_bytes,
                scheduler_watermark_fraction(),
            );
            if !decision.admitted {
                return Err(prepared);
            }
        }

        let PreparedTurn {
            admitted,
            response,
            cancelled,
            owner_id,
            seq_id,
            newly_assigned,
            reservation_blocks,
            block_size,
        } = *prepared;
        if cancelled.load(Ordering::Relaxed) {
            if newly_assigned {
                self.owner_states.remove(&owner_id);
                self.owner_sequences.remove(&owner_id);
                self.inner.release_scheduled_recurrent_for(seq_id);
            }
            response.send_error(
                Error::from_reason(engine::session::CHAT_SESSION_CANCELLED),
                cancelled.as_ref(),
            );
            return Ok(None);
        }
        if let Err(error) = self.inner.activate_scheduled_recurrent(seq_id) {
            if newly_assigned {
                self.owner_states.remove(&owner_id);
                self.owner_sequences.remove(&owner_id);
                self.inner.release_scheduled_recurrent_for(seq_id);
            }
            response.send_error(error, cancelled.as_ref());
            return Ok(None);
        }
        let owner_state = self.owner_states.entry(owner_id.clone()).or_default();
        self.inner.install_owner_state(seq_id, owner_state);
        self.inner.set_cache_owner_id(
            &admitted.params.cache_owner_id,
            admitted.params.cache_root_owner_id.as_deref(),
        );
        self.inner
            .set_turn_cancel_flag(Some(Arc::clone(&cancelled)));
        let owner_history = self
            .owner_states
            .get(&owner_id)
            .map(B::owner_tokens)
            .unwrap_or_default()
            .to_vec();
        let prefix_admission = match self.inner.prepare_scheduled_prefix(
            seq_id,
            &admitted.tokens,
            &owner_history,
            admitted.plan.is_delta || admitted.params.reuse_cache,
            admitted.params.cache_salt,
            block_size,
        ) {
            Ok(prefix) => prefix,
            Err(error) => {
                self.inner.abort_paged_turn();
                self.inner.release_scheduled_recurrent_for(seq_id);
                self.inner.set_turn_cancel_flag(None);
                self.owner_states.remove(&owner_id);
                self.owner_sequences.remove(&owner_id);
                response.send_error(error, cancelled.as_ref());
                return Ok(None);
            }
        };
        let (prefix, restore) = match prefix_admission {
            ScheduledPrefixAdmission::Ready(prefix) => (prefix, None),
            ScheduledPrefixAdmission::Waiting {
                provisional,
                restore,
            } => (provisional, Some(restore)),
        };
        if prefix.suffix_len() == 0 {
            self.inner.abort_paged_turn();
            self.inner.release_scheduled_recurrent_for(seq_id);
            self.inner.set_turn_cancel_flag(None);
            self.owner_states.remove(&owner_id);
            self.owner_sequences.remove(&owner_id);
            response.send_error(
                Error::from_reason(format!(
                    "{} scheduler produced an empty prefill suffix",
                    B::SCHEDULER_NAME
                )),
                cancelled.as_ref(),
            );
            return Ok(None);
        }
        let materialized_blocks = self
            .inner
            .scheduler_materialized_blocks(seq_id)
            .saturating_add(restore.as_ref().map_or(0, B::restore_reserved_blocks));
        let is_streaming = response.sink().is_some();
        let generation_stream = Stream::new(DeviceType::Gpu);
        let mut profiler = DecodeProfiler::new(
            self.inner
                .profiler_label(admitted.plan.is_delta, is_streaming),
            self.inner.family_name(),
        );
        profiler.set_prompt_tokens(
            self.inner
                .profiler_prefill_tokens(&prefix, admitted.tokens.len() as u32),
        );
        profiler.snapshot_memory_before();
        let generation_start = admitted.params.report_performance.then(Instant::now);
        let reuse_cache = admitted.plan.is_delta || admitted.params.reuse_cache;
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
            reuse_cache,
            response,
            generated_tokens: Vec::new(),
            finish_reason: String::from("length"),
            reasoning_tracker: engine::penalties::ReasoningTracker::from_setup(
                &admitted.thinking,
                admitted.think_end_id,
            ),
            extra_eos_ids: self.inner.extra_eos_ids(),
            generation_start,
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
        let prompt_len = admitted.tokens.len() as u32;
        let mut breaks = Vec::new();
        let mut boundary = payload.prefix.effective_cached_prefix_len() as u32;
        while boundary < prompt_len {
            boundary = boundary
                .saturating_add(self.inner.scheduler_prefill_slice_tokens())
                .min(prompt_len);
            breaks.push(boundary);
        }
        breaks.extend(
            self.inner
                .extra_prefill_breaks(
                    prompt_len,
                    payload.prefix.effective_cached_prefix_len() as u32,
                )
                .into_iter()
                .filter(|&boundary| {
                    boundary > payload.prefix.effective_cached_prefix_len() as u32
                        && boundary < prompt_len
                }),
        );
        breaks.sort_unstable();
        breaks.dedup();
        let turn_cancelled = Arc::clone(&cancelled);
        let turn = match TurnState::try_new_recover_payload(
            seq_id,
            admitted.tokens,
            payload.prefix.effective_cached_prefix_len() as u32,
            breaks,
            Some(cancelled),
            payload,
        ) {
            Ok(turn) => turn,
            Err((error, payload)) => {
                let failed_owner_id = payload.owner_id.clone();
                self.inner.abort_paged_turn();
                self.inner.release_scheduled_recurrent_for(seq_id);
                self.inner.set_turn_cancel_flag(None);
                self.owner_states.remove(&failed_owner_id);
                self.owner_sequences.remove(&failed_owner_id);
                payload
                    .response
                    .send_error(Error::from_reason(error), turn_cancelled.as_ref());
                return Ok(None);
            }
        }
        .with_block_reservation(reservation_blocks, materialized_blocks, block_size)
        .with_recurrent_state_reservation(self.inner.recurrent_state_bytes());
        if let Some(restore) = restore {
            self.pending_restores.insert(seq_id, restore);
            if let Err(error) = self.scheduler.try_enqueue_turn(turn) {
                let (_, mut turn) = *error;
                self.pending_restores.remove(&seq_id);
                turn.payload.failure = Some(Error::from_reason(
                    "duplicate sequence while parking SSD restore",
                ));
                self.finish_completed(turn);
                return Ok(None);
            }
            if let Err(error) = self.scheduler.park_waiting_for_ssd(seq_id) {
                self.pending_restores.remove(&seq_id);
                if let Some(mut turn) = self.scheduler.take_waiting(seq_id) {
                    turn.payload.failure = Some(Error::from_reason(error));
                    self.finish_completed(turn);
                }
            }
            Ok(None)
        } else {
            Ok(Some(turn))
        }
    }

    fn fail_preempted(
        &mut self,
        mut turn: TurnState<ScheduledTurn<B::PrefixState>>,
        error: String,
    ) {
        turn.payload.failure = Some(Error::from_reason(error));
        self.finish_completed(turn);
    }

    fn enqueue_turn_or_reject(&mut self, turn: TurnState<ScheduledTurn<B::PrefixState>>) {
        if let Err(error_and_turn) = self.scheduler.try_enqueue_turn(turn) {
            let (error, mut turn) = *error_and_turn;
            tracing::error!("{} scheduler admission failed: {error}", B::SCHEDULER_NAME);
            turn.payload.failure = Some(Error::from_reason(error));
            self.finish_completed(turn);
        }
    }

    fn reap_cancelled_waiters(&mut self) {
        for mut turn in self.scheduler.take_cancelled_waiters() {
            self.pending_restores.remove(&turn.seq_id);
            turn.payload.failure =
                Some(Error::from_reason(engine::session::CHAT_SESSION_CANCELLED));
            self.finish_completed(turn);
        }
    }

    fn poll_restores(&mut self) {
        let seq_ids = self.pending_restores.keys().copied().collect::<Vec<_>>();
        for seq_id in seq_ids {
            let Some((prompt_tokens, owner_history, preemption_replay)) =
                self.scheduler.waiting_turn_mut(seq_id).map(|turn| {
                    (
                        turn.payload.prompt_tokens.clone(),
                        self.owner_states
                            .get(&turn.payload.owner_id)
                            .map(B::owner_tokens)
                            .unwrap_or_default()
                            .to_vec(),
                        turn.payload.preemption_replay.is_some(),
                    )
                })
            else {
                self.pending_restores.remove(&seq_id);
                let _ = self.inner.release_scheduled_cache(seq_id);
                continue;
            };
            let outcome = {
                let Some(restore) = self.pending_restores.get_mut(&seq_id) else {
                    continue;
                };
                self.inner.poll_scheduled_restore(
                    seq_id,
                    restore,
                    &prompt_tokens,
                    &owner_history,
                    preemption_replay,
                )
            };
            match outcome {
                Ok(None) => {}
                Ok(Some(restored)) => {
                    self.pending_restores.remove(&seq_id);
                    let Some(turn) = self.scheduler.waiting_turn_mut(seq_id) else {
                        let _ = self.inner.release_scheduled_cache(seq_id);
                        continue;
                    };
                    turn.num_computed_tokens = restored.prefix.effective_cached_prefix_len() as u32;
                    turn.block_materialized_blocks = restored.materialized_blocks;
                    if let Some(replay) = turn.payload.preemption_replay.as_mut() {
                        replay.cached_prefix = turn.num_computed_tokens;
                    }
                    turn.payload.prefix = restored.prefix;
                    turn.payload
                        .profiler
                        .set_prompt_tokens(restored.profiler_prefill_tokens);
                    turn.pinned_prefill_breaks.clear();
                    let mut boundary = turn.num_computed_tokens;
                    while boundary < turn.prompt_tokens {
                        boundary = boundary
                            .saturating_add(self.inner.scheduler_prefill_slice_tokens())
                            .min(turn.prompt_tokens);
                        turn.pinned_prefill_breaks.push(boundary);
                    }
                    turn.pinned_prefill_breaks.extend(
                        restored.extra_prefill_breaks.into_iter().filter(|&value| {
                            value > turn.num_computed_tokens && value < turn.prompt_tokens
                        }),
                    );
                    turn.pinned_prefill_breaks.sort_unstable();
                    turn.pinned_prefill_breaks.dedup();
                    if let Err(error) =
                        self.scheduler
                            .wake_from_ssd(seq_id, restored.bytes_restored, restored.wait)
                    {
                        tracing::error!(
                            "{} failed to wake SSD restore {seq_id}: {error}",
                            B::SCHEDULER_NAME
                        );
                    }
                }
                Err(error) => {
                    self.pending_restores.remove(&seq_id);
                    if let Some(mut turn) = self.scheduler.take_waiting(seq_id) {
                        turn.payload.failure = Some(error);
                        self.finish_completed(turn);
                    } else {
                        let _ = self.inner.release_scheduled_cache(seq_id);
                    }
                }
            }
        }
    }

    /// Release both halves of a hybrid victim. Full-attention blocks remain
    /// reusable through their verified hashes; request-local GDN arrays are
    /// deliberately dropped because they cannot be rewound. Resume enters
    /// through `prime_prefix_state`, whose right-to-left checkpoint/sidecar
    /// lookup reconstructs the deepest GDN boundary that agrees with K/V.
    fn handle_preempted(&mut self, mut turn: TurnState<ScheduledTurn<B::PrefixState>>) {
        let prefix_tokens = turn.num_computed_tokens;
        let prompt_tokens = turn.payload.prompt_tokens.clone();
        let replay = match install_preemption_replay(
            &mut turn,
            &prompt_tokens,
            self.inner.scheduler_prefill_slice_tokens(),
        ) {
            Ok(replay) => replay,
            Err(error) => {
                self.fail_preempted(turn, error);
                return;
            }
        };
        let bytes_per_block = match self.inner.scheduler_cache_snapshot() {
            Ok(Some(snapshot)) => snapshot.bytes_per_block,
            Ok(None) => 1,
            Err(error) => {
                self.fail_preempted(turn, error.reason);
                return;
            }
        };
        let mut mode =
            self.scheduler
                .preemption_mode(prefix_tokens, turn.block_size, bytes_per_block);
        if mode == PreemptionMode::Ssd && !self.inner.scheduler_has_cold_tier() {
            mode = PreemptionMode::Recompute;
        }
        let lifecycle =
            self.inner
                .preempt_scheduled_cache(turn.seq_id, turn.payload.params.cache_salt, mode);
        self.inner.release_scheduled_recurrent_for(turn.seq_id);
        if let Err(error) = lifecycle {
            let _ = self.inner.release_scheduled_cache(turn.seq_id);
            self.fail_preempted(turn, error.reason);
            return;
        }
        turn.payload.preemption_replay = Some(replay);
        self.scheduler.record_preemption_mode(mode);
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
            self.fail_preempted(turn, engine::session::CHAT_SESSION_CANCELLED.to_string());
            return;
        }
        let cache_snapshot = match self.inner.scheduler_cache_snapshot() {
            Ok(snapshot) => snapshot,
            Err(error) => {
                self.fail_preempted(turn, error.reason);
                return;
            }
        };
        if !self.ensure_recurrent_slot(turn.seq_id) {
            self.scheduler.prepend_preempted(turn);
            return;
        }
        if let Some(snapshot) = cache_snapshot {
            let decision = self.scheduler.try_reserve_memory(
                engine::scheduler::MemoryTelemetry {
                    capacity_bytes: u64::from(snapshot.blocks.total_blocks)
                        .saturating_mul(snapshot.bytes_per_block),
                    free_bytes: u64::from(snapshot.blocks.free_blocks)
                        .saturating_mul(snapshot.bytes_per_block),
                    reclaimable_bytes: u64::from(snapshot.blocks.reclaimable_blocks)
                        .saturating_mul(snapshot.bytes_per_block),
                },
                turn.block_reservation_total,
                snapshot.bytes_per_block,
                self.inner.scheduled_recurrent_bytes(),
                turn.recurrent_state_bytes,
                scheduler_watermark_fraction(),
            );
            if !decision.admitted {
                self.scheduler.prepend_preempted(turn);
                return;
            }
        }
        let Some(replay) = turn.payload.preemption_replay.as_ref() else {
            self.fail_preempted(
                turn,
                format!(
                    "{} preempted turn is missing replay state",
                    B::SCHEDULER_NAME
                ),
            );
            return;
        };
        let replay_tokens = replay.tokens.clone();
        let target = replay_tokens.len() as u32;
        let owner_state = self
            .owner_states
            .entry(turn.payload.owner_id.clone())
            .or_default();
        self.inner.install_owner_state(turn.seq_id, owner_state);
        self.inner.set_cache_owner_id(
            &turn.payload.params.cache_owner_id,
            turn.payload.params.cache_root_owner_id.as_deref(),
        );
        if let Err(error) = self.inner.activate_scheduled_recurrent(turn.seq_id) {
            self.fail_preempted(turn, error.reason);
            return;
        }
        let prefix_admission = match self.inner.prepare_scheduled_prefix(
            turn.seq_id,
            &replay_tokens,
            self.owner_states
                .get(&turn.payload.owner_id)
                .map(B::owner_tokens)
                .unwrap_or_default(),
            true,
            turn.payload.params.cache_salt,
            turn.block_size,
        ) {
            Ok(prefix) => prefix,
            Err(error) if is_paged_allocation_blocked(&error.reason) => {
                let _ = self.inner.release_scheduled_cache(turn.seq_id);
                self.inner.release_scheduled_recurrent_for(turn.seq_id);
                self.scheduler.prepend_preempted(turn);
                return;
            }
            Err(error) => {
                let _ = self.inner.release_scheduled_cache(turn.seq_id);
                self.inner.release_scheduled_recurrent_for(turn.seq_id);
                self.fail_preempted(turn, error.reason);
                return;
            }
        };
        let (prefix, restore) = match prefix_admission {
            ScheduledPrefixAdmission::Ready(prefix) => (prefix, None),
            ScheduledPrefixAdmission::Waiting {
                provisional,
                restore,
            } => (provisional, Some(restore)),
        };
        let materialized_blocks = self
            .inner
            .scheduler_materialized_blocks(turn.seq_id)
            .saturating_add(restore.as_ref().map_or(0, B::restore_reserved_blocks));
        turn.num_computed_tokens = prefix.effective_cached_prefix_len() as u32;
        turn.block_materialized_blocks = materialized_blocks;
        turn.pinned_prefill_breaks.clear();
        let mut boundary = turn.num_computed_tokens;
        while boundary < target {
            boundary = boundary
                .saturating_add(self.inner.scheduler_prefill_slice_tokens())
                .min(target);
            turn.pinned_prefill_breaks.push(boundary);
        }
        turn.pinned_prefill_breaks.extend(
            self.inner
                .extra_prefill_breaks(target, turn.num_computed_tokens)
                .into_iter()
                .filter(|&boundary| boundary > turn.num_computed_tokens && boundary < target),
        );
        turn.pinned_prefill_breaks.sort_unstable();
        turn.pinned_prefill_breaks.dedup();
        let Some(replay) = turn.payload.preemption_replay.as_mut() else {
            self.fail_preempted(
                turn,
                format!(
                    "{} preempted turn lost replay state after prefix restore",
                    B::SCHEDULER_NAME
                ),
            );
            return;
        };
        replay.cached_prefix = turn.num_computed_tokens;
        turn.payload.prefix = prefix;
        if let Some(restore) = restore {
            let seq_id = turn.seq_id;
            self.pending_restores.insert(seq_id, restore);
            self.scheduler.ready_preempted(turn, true);
        } else {
            self.scheduler.ready_preempted(turn, false);
        }
    }

    fn finish_completed(&mut self, mut turn: TurnState<ScheduledTurn<B::PrefixState>>) {
        let Some(cancelled) = turn.cancelled.take() else {
            let owner_id = turn.payload.owner_id.clone();
            self.inner.abort_paged_turn();
            self.inner.release_scheduled_recurrent_for(turn.seq_id);
            self.inner.set_turn_cancel_flag(None);
            self.owner_states.remove(&owner_id);
            self.owner_sequences.remove(&owner_id);
            turn.payload.response.send_error(
                Error::from_reason(format!(
                    "{} scheduled turn lost its cancellation state",
                    B::SCHEDULER_NAME
                )),
                &AtomicBool::new(false),
            );
            return;
        };
        if let Some(owner_state) = self.owner_states.get(&turn.payload.owner_id) {
            self.inner.install_owner_state(turn.seq_id, owner_state);
        }
        if let Err(error) = self.inner.activate_paged_seq(turn.seq_id) {
            self.inner.release_scheduled_recurrent_for(turn.seq_id);
            self.owner_states.remove(&turn.payload.owner_id);
            self.owner_sequences.remove(&turn.payload.owner_id);
            turn.payload.response.send_error(error, cancelled.as_ref());
            return;
        }
        if let Some(error) = turn.payload.failure.take() {
            self.inner.abort_paged_turn();
            self.inner.release_scheduled_recurrent_for(turn.seq_id);
            self.inner.set_turn_cancel_flag(None);
            self.owner_states.remove(&turn.payload.owner_id);
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
                effective_cached_prefix_len: turn.payload.prefix.effective_cached_prefix_len(),
                suffix_len: turn.payload.prefix.suffix_len(),
                generated_tokens: &turn.payload.generated_tokens,
                finish_reason: std::mem::take(&mut turn.payload.finish_reason),
                retain_final_length_token: false,
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
        let parked = if outcome.is_ok() && turn.payload.reuse_cache {
            self.owner_states.insert(
                turn.payload.owner_id.clone(),
                self.inner.capture_owner_state(turn.seq_id),
            );
            self.inner.park_active_scheduled_recurrent()
        } else {
            self.inner.release_scheduled_recurrent_for(turn.seq_id);
            Ok(())
        };
        self.inner.set_turn_cancel_flag(None);
        if outcome.is_err() || parked.is_err() {
            self.owner_states.remove(&turn.payload.owner_id);
            self.owner_sequences.remove(&turn.payload.owner_id);
        }
        let outcome = match (outcome, parked) {
            (Ok(output), Ok(())) => Ok(output),
            (Err(error), _) | (Ok(_), Err(error)) => Err(error),
        };
        match turn.payload.response {
            ScheduledReply::Sync(reply) => {
                let result = if cancelled.load(Ordering::Relaxed) {
                    Err(Error::from_reason(engine::session::CHAT_SESSION_CANCELLED))
                } else {
                    match outcome {
                        Ok(TurnOutput::Complete(result)) => Ok(*result),
                        Ok(TurnOutput::Streamed) => Err(Error::from_reason(format!(
                            "{} scheduler returned streamed output for a sync turn",
                            B::SCHEDULER_NAME
                        ))),
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

    pub(crate) fn drive(
        &mut self,
        receiver: &mut tokio::sync::mpsc::UnboundedReceiver<B::Command>,
    ) -> LoopControl {
        if !self.scheduler.has_work() && self.pending.is_empty() && self.prepared_waiting.is_none()
        {
            match receiver.blocking_recv() {
                Some(command) => self.pending.push_back(command),
                None => return LoopControl::Break,
            }
        }
        while let Ok(command) = receiver.try_recv() {
            self.pending.push_back(command);
        }
        self.reap_cancelled_waiters();
        self.poll_restores();

        match self.inner.scheduler_cache_snapshot() {
            Ok(Some(snapshot)) => {
                self.scheduler.observe_hybrid_memory(
                    snapshot.blocks,
                    snapshot.bytes_per_block,
                    self.inner.scheduled_recurrent_bytes(),
                    scheduler_watermark_fraction(),
                );
            }
            Ok(None) => {}
            Err(error) => tracing::warn!(
                "{} scheduler telemetry unavailable: {error:?}",
                B::SCHEDULER_NAME
            ),
        }

        let mut deferred = false;
        if !self.scheduler.has_pending_control()
            && let Some(prepared) = self.prepared_waiting.take()
        {
            match self.admit_prepared(prepared) {
                Ok(Some(turn)) => {
                    self.enqueue_turn_or_reject(turn);
                }
                Ok(None) => {}
                Err(prepared) => {
                    self.prepared_waiting = Some(prepared);
                    deferred = true;
                }
            }
        }
        if deferred
            && let Some(blocked_owner_id) = self
                .prepared_waiting
                .as_ref()
                .map(|prepared| prepared.owner_id.clone())
        {
            self.process_deferred_cleanup(&blocked_owner_id);
        }
        while !deferred
            && !self.scheduler.has_pending_control()
            && let Some(command) = self.pending.front()
        {
            let must_wait_for_legacy_owner = command.as_chat().is_some_and(|chat| {
                Self::chat_requires_legacy_owner_drain(chat) && self.scheduler.has_work()
            });
            if must_wait_for_legacy_owner {
                break;
            }
            if command
                .as_chat()
                .is_some_and(|chat| !matches!(chat, ChatCmd::ReleaseCacheOwner { .. }))
                && self.scheduler.waiting_len() + self.scheduler.running_len()
                    >= self.scheduler.max_num_seqs()
            {
                break;
            }
            let Some(command) = self.pending.pop_front() else {
                break;
            };
            if (Self::force_serial() || !self.enabled)
                && !command
                    .as_chat()
                    .is_some_and(|chat| matches!(chat, ChatCmd::ReleaseCacheOwner { .. }))
            {
                self.scheduler.enqueue_exclusive(command);
                break;
            }
            match command.into_chat() {
                Ok(ChatCmd::ReleaseCacheOwner { owner_id, reply }) => {
                    if self.cache_owner_release_blocked(&owner_id) {
                        self.pending.push_front(B::Command::from_chat(
                            ChatCmd::ReleaseCacheOwner { owner_id, reply },
                        ));
                        break;
                    }
                    let _ = reply.send(self.release_cache_owner_now(&owner_id));
                }
                Ok(chat) => {
                    let is_reset = matches!(chat, ChatCmd::ResetCaches { .. });
                    let mut exclusive = false;
                    if !self.inner.scheduler_cache_available()
                        || !Self::chat_has_explicit_owner(&chat)
                        || self.chat_requires_barrier(&chat)
                    {
                        self.scheduler
                            .enqueue_exclusive(B::Command::from_chat(chat));
                        exclusive = true;
                    } else if let Some(prepared) = self.prepare_chat(chat) {
                        match self.admit_prepared(prepared) {
                            Ok(Some(turn)) => {
                                self.enqueue_turn_or_reject(turn);
                            }
                            Ok(None) => {}
                            Err(prepared) => {
                                self.prepared_waiting = Some(prepared);
                                deferred = true;
                            }
                        }
                    }
                    if is_reset || exclusive {
                        break;
                    }
                }
                Err(barrier) => {
                    self.scheduler.enqueue_barrier(barrier);
                    break;
                }
            }
        }

        let action = {
            let mut executor = self.inner.step_executor();
            self.scheduler.drive_once(&mut executor)
        };
        let scheduler_idle = matches!(&action, Ok(SchedulerAction::Idle));
        let mut may_resume_preempted = true;
        match action {
            Ok(SchedulerAction::Idle) => {}
            Ok(SchedulerAction::Exclusive(command) | SchedulerAction::Barrier(command)) => {
                let reset = command
                    .as_chat()
                    .is_some_and(|chat| matches!(chat, ChatCmd::ResetCaches { .. }));
                let command = match command.into_scheduler_stats() {
                    Ok(reply) => {
                        let _ = reply.send(Ok(self.scheduler.stats().to_js()));
                        return LoopControl::Continue;
                    }
                    Err(command) => command,
                };
                self.inner.execute_barrier(
                    command,
                    SchedulerOwnerContext {
                        owner_sequences: &mut self.owner_sequences,
                        owner_states: &mut self.owner_states,
                        next_seq_id: &mut self.next_seq_id,
                    },
                );
                if reset {
                    self.owner_sequences.clear();
                    self.owner_states.clear();
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
            Err(SchedulerError::Executor(error)) => {
                tracing::error!(
                    "{} scheduler execution failed: {error:?}",
                    B::SCHEDULER_NAME
                );
                return LoopControl::Break;
            }
            Err(SchedulerError::InvalidResult(message)) => {
                tracing::error!(
                    "{} scheduler returned an invalid result: {message}",
                    B::SCHEDULER_NAME
                );
                return LoopControl::Break;
            }
        }
        if may_resume_preempted {
            self.try_resume_preempted();
        }
        if scheduler_idle && (self.scheduler.has_work() || self.prepared_waiting.is_some()) {
            std::thread::sleep(Duration::from_millis(1));
        }
        LoopControl::Continue
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::qwen3_5::config::Qwen3_5Config;
    use crate::models::qwen3_5::model::{Qwen35Cmd, Qwen35Inner};
    use crate::models::qwen3_5_moe::config::Qwen3_5MoeConfig;
    use crate::models::qwen3_5_moe::model::Qwen35MoeInner;

    fn tiny_config() -> Qwen3_5Config {
        Qwen3_5Config {
            vocab_size: 1024,
            hidden_size: 64,
            num_layers: 8,
            num_heads: 4,
            num_kv_heads: 2,
            intermediate_size: 128,
            rms_norm_eps: 1e-6,
            head_dim: 16,
            tie_word_embeddings: true,
            attention_bias: false,
            max_position_embeddings: 1024,
            pad_token_id: 0,
            eos_token_id: 0,
            bos_token_id: 0,
            linear_num_value_heads: 4,
            linear_num_key_heads: 2,
            linear_key_head_dim: 16,
            linear_value_head_dim: 16,
            linear_conv_kernel_dim: 4,
            full_attention_interval: 4,
            partial_rotary_factor: 0.25,
            rope_theta: 100_000.0,
            paged_cache_memory_mb: Some(64),
            paged_block_size: Some(16),
            use_block_paged_cache: None,
            persist_paged_cache: None,
            n_mtp_layers: 0,
        }
    }

    fn tiny_moe_config() -> Qwen3_5MoeConfig {
        Qwen3_5MoeConfig {
            vocab_size: 1024,
            hidden_size: 64,
            num_layers: 8,
            num_heads: 4,
            num_kv_heads: 2,
            intermediate_size: 128,
            rms_norm_eps: 1e-6,
            head_dim: 16,
            tie_word_embeddings: true,
            attention_bias: false,
            max_position_embeddings: 1024,
            pad_token_id: 0,
            eos_token_id: 0,
            bos_token_id: 0,
            linear_num_value_heads: 4,
            linear_num_key_heads: 2,
            linear_key_head_dim: 16,
            linear_value_head_dim: 16,
            linear_conv_kernel_dim: 4,
            full_attention_interval: 4,
            partial_rotary_factor: 0.25,
            rope_theta: 100_000.0,
            num_experts: 4,
            num_experts_per_tok: 2,
            decoder_sparse_step: 1,
            shared_expert_intermediate_size: None,
            moe_intermediate_size: None,
            norm_topk_prob: true,
            mlp_only_layers: None,
            paged_cache_memory_mb: Some(64),
            paged_block_size: Some(16),
            use_block_paged_cache: None,
            persist_paged_cache: None,
            n_mtp_layers: 0,
        }
    }

    #[test]
    fn dense_and_moe_construct_the_same_engine_scheduler() {
        let dense = Qwen35Inner::new(tiny_config()).expect("construct tiny dense model");
        let moe = Qwen35MoeInner::new(tiny_moe_config()).expect("construct tiny MoE model");
        let dense_state: HybridSchedulerState<Qwen35Inner> =
            HybridSchedulerState::new(dense).expect("construct dense scheduler");
        let moe_state: HybridSchedulerState<Qwen35MoeInner> =
            HybridSchedulerState::new(moe).expect("construct MoE scheduler");

        assert_eq!(
            dense_state.scheduler.max_num_seqs(),
            moe_state.scheduler.max_num_seqs()
        );
        assert_eq!(dense_state.scheduler.waiting_len(), 0);
        assert_eq!(moe_state.scheduler.waiting_len(), 0);
        assert!(
            !dense_state.enabled,
            "dense Qwen3.5 must keep the generic scheduler opt-in"
        );
        assert!(
            !moe_state.enabled,
            "Qwen3.5 MoE must keep the generic scheduler opt-in"
        );
    }

    #[test]
    fn recurrent_slot_eviction_skips_history_only_owners() {
        let inner = Qwen35Inner::new(tiny_config()).expect("construct tiny dense model");
        let mut state = HybridSchedulerState::new(inner).expect("construct scheduler");
        state.owner_sequences.insert("history-only".into(), 1);
        state.owner_sequences.insert("resident-a".into(), 2);
        state.owner_sequences.insert("resident-b".into(), 3);
        state
            .inner
            .activate_scheduled_recurrent(2)
            .expect("activate resident a");
        state
            .inner
            .activate_scheduled_recurrent(3)
            .expect("park resident a and activate resident b");

        assert!(state.ensure_recurrent_slot(4));
        assert!(
            !state.inner.has_scheduled_recurrent(2),
            "the oldest actual idle resident is evicted"
        );
        assert!(state.inner.has_scheduled_recurrent(3));
        assert!(state.inner.can_activate_scheduled_recurrent(4));
    }

    #[test]
    fn cache_owner_release_drops_history_sequence_and_recurrent_state() {
        let inner = Qwen35Inner::new(tiny_config()).expect("construct tiny dense model");
        let mut state = HybridSchedulerState::new(inner).expect("construct scheduler");
        state.owner_sequences.insert("stateless-owner".into(), 9);
        state
            .owner_states
            .insert("stateless-owner".into(), vec![7, 8]);
        state
            .inner
            .activate_scheduled_recurrent(9)
            .expect("activate owner recurrent state");

        state
            .release_cache_owner_now("stateless-owner")
            .expect("release owner");

        assert!(!state.owner_sequences.contains_key("stateless-owner"));
        assert!(!state.owner_states.contains_key("stateless-owner"));
        assert!(!state.inner.has_scheduled_recurrent(9));
    }

    #[test]
    fn failed_cache_owner_release_preserves_registries_for_retry() {
        let inner = Qwen35Inner::new(tiny_config()).expect("construct tiny dense model");
        let mut state = HybridSchedulerState::new(inner).expect("construct scheduler");
        state.owner_sequences.insert("retry-owner".into(), 11);
        state
            .owner_states
            .insert("retry-owner".into(), vec![3, 5, 8]);

        let error = state
            .release_cache_owner_with("retry-owner", |_inner, seq_id, owner_state| {
                assert_eq!(seq_id, Some(11));
                assert_eq!(owner_state.map(Vec::as_slice), Some(&[3, 5, 8][..]));
                Err(Error::from_reason("injected release failure"))
            })
            .expect_err("injected release must fail");

        assert_eq!(error.reason, "injected release failure");
        assert_eq!(state.owner_sequences.get("retry-owner"), Some(&11));
        assert_eq!(
            state.owner_states.get("retry-owner").map(Vec::as_slice),
            Some(&[3, 5, 8][..])
        );

        state
            .release_cache_owner_with("retry-owner", |_inner, seq_id, owner_state| {
                assert_eq!(seq_id, Some(11));
                assert_eq!(owner_state.map(Vec::as_slice), Some(&[3, 5, 8][..]));
                Ok(())
            })
            .expect("retry release succeeds");
        assert!(!state.owner_sequences.contains_key("retry-owner"));
        assert!(!state.owner_states.contains_key("retry-owner"));
    }

    #[test]
    fn deferred_admission_processes_idle_owner_release_behind_ordinary_work() {
        let inner = Qwen35Inner::new(tiny_config()).expect("construct tiny dense model");
        let mut state = HybridSchedulerState::new(inner).expect("construct scheduler");
        state.owner_sequences.insert("completed-owner".into(), 9);
        state
            .owner_states
            .insert("completed-owner".into(), vec![7, 8]);
        state
            .inner
            .activate_scheduled_recurrent(9)
            .expect("activate completed owner recurrent state");

        let (ordinary_reply, _ordinary_result) = tokio::sync::oneshot::channel();
        state
            .pending
            .push_back(Qwen35Cmd::from_chat(ChatCmd::SessionStart {
                messages: Vec::new(),
                config: ChatConfig::default(),
                reply: ordinary_reply,
                cancelled: Arc::new(AtomicBool::new(false)),
            }));
        let (release_reply, mut release_result) = tokio::sync::oneshot::channel();
        state
            .pending
            .push_back(Qwen35Cmd::from_chat(ChatCmd::ReleaseCacheOwner {
                owner_id: "completed-owner".into(),
                reply: release_reply,
            }));

        assert_eq!(
            state.process_deferred_cleanup("blocked-owner"),
            DeferredCleanupProgress::OwnerReleaseProcessed
        );
        assert_eq!(
            state.pending.len(),
            1,
            "ordinary work stays queued in place"
        );
        assert!(state.pending.front().is_some_and(|command| {
            command
                .as_chat()
                .is_some_and(|chat| matches!(chat, ChatCmd::SessionStart { .. }))
        }));
        assert!(matches!(release_result.try_recv(), Ok(Ok(()))));
        assert!(!state.owner_sequences.contains_key("completed-owner"));
        assert!(!state.owner_states.contains_key("completed-owner"));
        assert!(!state.inner.has_scheduled_recurrent(9));
    }

    #[test]
    fn deferred_cleanup_does_not_cross_an_earlier_global_reset() {
        let inner = Qwen35Inner::new(tiny_config()).expect("construct tiny dense model");
        let mut state = HybridSchedulerState::new(inner).expect("construct scheduler");
        state.owner_sequences.insert("completed-owner".into(), 9);
        state
            .owner_states
            .insert("completed-owner".into(), vec![7, 8]);
        let (ordinary_reply, _ordinary_result) = tokio::sync::oneshot::channel();
        state
            .pending
            .push_back(Qwen35Cmd::from_chat(ChatCmd::SessionStart {
                messages: Vec::new(),
                config: ChatConfig::default(),
                reply: ordinary_reply,
                cancelled: Arc::new(AtomicBool::new(false)),
            }));
        let (reset_reply, _reset_result) = tokio::sync::oneshot::channel();
        state
            .pending
            .push_back(Qwen35Cmd::from_chat(ChatCmd::ResetCaches {
                reply: reset_reply,
            }));
        let (release_reply, mut release_result) = tokio::sync::oneshot::channel();
        state
            .pending
            .push_back(Qwen35Cmd::from_chat(ChatCmd::ReleaseCacheOwner {
                owner_id: "completed-owner".into(),
                reply: release_reply,
            }));

        assert_eq!(
            state.process_deferred_cleanup("blocked-owner"),
            DeferredCleanupProgress::None
        );
        assert_eq!(state.pending.len(), 3);
        assert!(matches!(
            release_result.try_recv(),
            Err(tokio::sync::oneshot::error::TryRecvError::Empty)
        ));
        assert_eq!(state.owner_sequences.get("completed-owner"), Some(&9));
    }

    #[test]
    fn deferred_cleanup_does_not_overtake_an_earlier_turn_for_the_same_owner() {
        let inner = Qwen35Inner::new(tiny_config()).expect("construct tiny dense model");
        let mut state = HybridSchedulerState::new(inner).expect("construct scheduler");
        state.owner_sequences.insert("continued-owner".into(), 9);
        state
            .owner_states
            .insert("continued-owner".into(), vec![7, 8]);
        let (turn_reply, _turn_result) = tokio::sync::oneshot::channel();
        state
            .pending
            .push_back(Qwen35Cmd::from_chat(ChatCmd::SessionContinue {
                messages: Vec::new(),
                config: ChatConfig {
                    cache_owner_id: Some("continued-owner".into()),
                    ..ChatConfig::default()
                },
                reply: turn_reply,
                cancelled: Arc::new(AtomicBool::new(false)),
            }));
        let (release_reply, mut release_result) = tokio::sync::oneshot::channel();
        state
            .pending
            .push_back(Qwen35Cmd::from_chat(ChatCmd::ReleaseCacheOwner {
                owner_id: "continued-owner".into(),
                reply: release_reply,
            }));

        assert_eq!(
            state.process_deferred_cleanup("blocked-owner"),
            DeferredCleanupProgress::None
        );
        assert_eq!(state.pending.len(), 2);
        assert!(matches!(
            release_result.try_recv(),
            Err(tokio::sync::oneshot::error::TryRecvError::Empty)
        ));
        assert_eq!(state.owner_sequences.get("continued-owner"), Some(&9));
        assert_eq!(
            state.owner_states.get("continued-owner").map(Vec::as_slice),
            Some(&[7, 8][..])
        );
    }
}
