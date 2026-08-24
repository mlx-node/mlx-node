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
    TurnTokenObserver,
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
    pub max_total_blocks: u32,
    pub bytes_per_block: u64,
}

/// Result of unified-memory reservation after reclaiming idle parked rows.
#[derive(Debug)]
enum MemoryReserveOutcome {
    Admitted,
    Defer,
    Reject { total_blocks: u32 },
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
            max_total_blocks: blocks.max_total_blocks,
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
    *VALUE.get_or_init(|| scheduler_per_seq_context_override().unwrap_or(32_768))
}

pub(crate) fn scheduler_per_seq_context_override() -> Option<u32> {
    static VALUE: OnceLock<Option<u32>> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("MLX_PAGED_PER_SEQ_CTX")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .filter(|&value| value > 0)
    })
}

pub(crate) fn scheduled_turn_context(
    trained_context: u32,
    pool_tokens: u32,
    override_cap: Option<u32>,
) -> u32 {
    let trained = trained_context.max(1);
    let pool = pool_tokens.max(1);
    let mut cap = trained.min(pool);
    if let Some(limit) = override_cap.filter(|&value| value > 0) {
        cap = cap.min(limit);
    }
    cap
}

/// Tokens one sequence can actually use after charging recurrent state
/// (Mamba/GDN) against the same unified-memory pool as paged KV blocks.
pub(crate) fn pool_tokens_after_recurrent(
    pool_tokens: u32,
    block_size: u32,
    bytes_per_block: u64,
    recurrent_state_bytes: u64,
) -> u32 {
    if bytes_per_block == 0 || block_size == 0 {
        return pool_tokens.max(1);
    }
    let rec_blocks = recurrent_state_bytes
        .div_ceil(bytes_per_block)
        .min(u64::from(u32::MAX)) as u32;
    let rec_tokens = rec_blocks.saturating_mul(block_size);
    pool_tokens.saturating_sub(rec_tokens).max(1)
}

/// Whether a reservation plus one row of recurrent state can ever fit in an
/// empty pool. If not, admission must error — not defer — or the row wedges
/// `prepared_waiting` forever.
pub(crate) fn reservation_fits_empty_pool(
    reservation_blocks: u32,
    total_blocks: u32,
    bytes_per_block: u64,
    recurrent_state_bytes: u64,
) -> bool {
    let need = u64::from(reservation_blocks)
        .saturating_mul(bytes_per_block)
        .saturating_add(recurrent_state_bytes);
    let cap = u64::from(total_blocks).saturating_mul(bytes_per_block);
    need <= cap
}

/// Max-pool admission ceiling: a reservation that cannot fit even the
/// pool's maximum (grow-target) block count must hard-error at admission —
/// no grow path can reach past the max pool. A reservation below the max
/// but above CURRENT free capacity is handled by
/// `try_reserve_reclaiming_idle`, which may grow the pool once on its
/// no-live-turns Reject edge before rejecting.
pub(crate) fn exceeds_max_pool_ceiling(
    snapshot: &SchedulerCacheSnapshot,
    reservation_blocks: u32,
    candidate_state_bytes: u64,
) -> bool {
    !reservation_fits_empty_pool(
        reservation_blocks,
        snapshot.max_total_blocks,
        snapshot.bytes_per_block,
        candidate_state_bytes,
    )
}

/// Block shortfall a one-shot pool grow must add on the no-live-turns
/// Reject edge of `try_reserve_reclaiming_idle_with`, or `None` to keep the
/// existing Reject path. The shortfall is measured in BYTES — reservation
/// blocks plus candidate and already-scheduled recurrent state — because
/// admission is a byte inequality over one shared budget, and each grown
/// block adds `bytes_per_block` of budget. `None` when the pool already
/// sits at its max, the reservation exceeds the max-pool ceiling, or the
/// byte shortfall is zero.
fn grow_needed_blocks(
    snapshot: &SchedulerCacheSnapshot,
    reservation_blocks: u32,
    candidate_state_bytes: u64,
    existing_scheduled_state_bytes: u64,
) -> Option<u32> {
    if snapshot.blocks.total_blocks >= snapshot.max_total_blocks {
        return None;
    }
    if exceeds_max_pool_ceiling(snapshot, reservation_blocks, candidate_state_bytes) {
        return None;
    }
    let bytes_per_block = snapshot.bytes_per_block.max(1);
    let available_bytes = u64::from(snapshot.blocks.free_blocks)
        .saturating_add(u64::from(snapshot.blocks.reclaimable_blocks))
        .saturating_mul(bytes_per_block);
    let need_bytes = u64::from(reservation_blocks)
        .saturating_mul(bytes_per_block)
        .saturating_add(candidate_state_bytes)
        .saturating_add(existing_scheduled_state_bytes);
    let shortfall_bytes = need_bytes.saturating_sub(available_bytes);
    let shortfall_blocks = shortfall_bytes
        .div_ceil(bytes_per_block)
        .min(u64::from(u32::MAX)) as u32;
    (shortfall_blocks > 0).then_some(shortfall_blocks)
}

/// Keep-live blocks already allocated for `seq_id` that a continuation
/// will reuse. Subtracted from `reservation_blocks` at admit so the
/// full-prompt ISL is not charged on top of KV that is neither free nor
/// reclaimable. Mismatch or `!reuse` credits 0 — prefix prepare rebuilds.
fn reusable_keep_live_blocks(
    adapter: Option<&PagedKVCacheAdapter>,
    seq_id: SeqId,
    prompt: &[u32],
    reuse: bool,
) -> u32 {
    if !reuse {
        return 0;
    }
    let Some(adapter) = adapter else {
        return 0;
    };
    if !adapter.is_live_for_continue_for(seq_id) {
        return 0;
    }
    let Some(held) = adapter.request_tokens_for(seq_id) else {
        return 0;
    };
    if !prompt.starts_with(held) {
        return 0;
    }
    adapter
        .block_table_for(seq_id)
        .map(|table| table.num_blocks() as u32)
        .unwrap_or(0)
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
    pub(crate) turn_token_observer: Option<Box<dyn TurnTokenObserver>>,
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
    observer_stopped: bool,
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
            emitter.on_token_id(
                token_id,
                &text,
                is_reasoning,
                payload.params.include_reasoning,
                sink,
            );
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
            if row.observer_stopped || row.stops_at_eos {
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
            let observer_stopped = !planned.cancel_snapshot
                && !(stops_at_eos && !B::STREAM_EOS_TOKEN)
                && turn
                    .payload
                    .turn_token_observer
                    .as_mut()
                    .is_some_and(|observer| observer.observe_token_id(token_id));
            let terminal =
                stops_at_eos || planned.cancel_snapshot || repetition.is_some() || observer_stopped;
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
                observer_stopped,
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
                    if rolled_back.is_some()
                        && let Some(observer) = turn.payload.turn_token_observer.as_mut()
                    {
                        observer.rollback_last_token();
                    }
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

    const fn chat_is_streaming(command: &ChatCmd) -> bool {
        matches!(
            command,
            ChatCmd::StreamSessionStart { .. }
                | ChatCmd::StreamSessionContinue { .. }
                | ChatCmd::StreamSessionContinueTool { .. }
        )
    }

    /// Whether a speculative decoder this model would actually admit for a
    /// turn of this shape needs the exclusive lane. Both the barrier routing
    /// gate and the scheduled-lane admission reject consult this; neither
    /// derives the answer from `enable_mtp` alone, so a request the planner
    /// will resolve autoregressive — no decoder loaded, or no streaming arm —
    /// keeps the scheduled lane instead of stalling the whole model.
    ///
    /// Media and live-context narrowing stay in `TurnPlan::resolve`, which
    /// runs after prompt rendering: this gate only ever admits MORE turns to
    /// the barrier than the planner will speculate on, which costs a lane and
    /// never correctness.
    fn speculation_requires_exclusive_lane(&self, streaming: bool) -> bool {
        self.inner
            .execution_plan()
            .speculative
            .is_some_and(|speculative| {
                speculative.admits_streaming(streaming)
                    && speculative.lane() != engine::plan::SpeculativeLane::Scheduled
            })
    }

    fn chat_requires_barrier(&self, command: &ChatCmd) -> bool {
        if Self::chat_config(command).is_some_and(|config| config.enable_mtp == Some(true))
            && self.speculation_requires_exclusive_lane(Self::chat_is_streaming(command))
        {
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
        let streaming = matches!(response, ScheduledReply::Stream(_));
        let mut admitted = match engine::session::admit_paged_turn(
            &mut self.inner,
            messages,
            config,
            kind,
            streaming,
        ) {
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
        // `chat_requires_barrier` already routes Barrier-lane speculation to
        // the exclusive lane; admission re-consults the SAME lane decision
        // against the decoder the planner actually resolved, so a routing gap
        // can never execute speculation on the scheduled lane.
        if admitted.plan.path() != engine::plan::TurnPath::Paged
            || !admitted.images.is_empty()
            || !admitted.audio.is_empty()
            || (matches!(
                admitted.plan.decoder,
                engine::plan::DecoderPlan::Speculative(_)
            ) && self.speculation_requires_exclusive_lane(streaming))
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
        let adapter = self.inner.paged_adapter();
        let pool_tokens = adapter
            .map(crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter::max_capacity_tokens)
            .unwrap_or(trained_context);
        let bytes_per_block = adapter
            .map(|adapter| adapter.bytes_per_block().unwrap_or(0))
            .unwrap_or(0);
        let usable_pool = pool_tokens_after_recurrent(
            pool_tokens,
            block_size,
            bytes_per_block,
            self.inner.recurrent_state_bytes(),
        );
        let context = scheduled_turn_context(
            trained_context,
            usable_pool,
            scheduler_per_seq_context_override(),
        );
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

    /// Idle parked recurrent row that is not the candidate and is not in the
    /// scheduler's running/waiting set. Unit-cap eviction only.
    fn idle_recurrent_victim(&self, seq_id: SeqId) -> Option<SeqId> {
        self.owner_sequences
            .values()
            .copied()
            .filter(|&candidate| {
                candidate != seq_id
                    && self.inner.has_scheduled_recurrent(candidate)
                    && !self.scheduler.contains_seq(candidate)
            })
            .min()
    }

    /// Idle owner that holds a recurrent row or a keep-live/cache table.
    /// Used by memory-denial reclaim only; unit-cap stays rec-only.
    fn idle_memory_victim(&self, seq_id: SeqId) -> Option<SeqId> {
        let mut candidates: Vec<SeqId> = self.owner_sequences.values().copied().collect();
        if let Some(adapter) = self.inner.paged_adapter() {
            candidates.extend(adapter.live_seq_ids());
        }
        candidates.sort_unstable();
        candidates.dedup();
        candidates.into_iter().find(|&candidate| {
            candidate != seq_id
                && !self.scheduler.contains_seq(candidate)
                && (self.inner.has_scheduled_recurrent(candidate)
                    || self.inner.scheduler_materialized_blocks(candidate) > 0
                    || self
                        .inner
                        .paged_adapter()
                        .is_some_and(|adapter| adapter.block_table_for(candidate).is_some()))
        })
    }

    /// Keep the two-unit residency cap as a queueing boundary. An idle warm
    /// row may be discarded because its owner history can rebuild the exact
    /// GDN state through the ordinary prefix-prime path; a scheduler-owned row
    /// is never evicted here.
    fn ensure_recurrent_slot(&mut self, seq_id: SeqId) -> bool {
        if self.inner.can_activate_scheduled_recurrent(seq_id) {
            return true;
        }
        let Some(victim) = self.idle_recurrent_victim(seq_id) else {
            return false;
        };
        self.inner.release_scheduled_recurrent_for(victim);
        self.inner.can_activate_scheduled_recurrent(seq_id)
    }

    /// Drop one idle parked recurrent row and/or unpin its keep-live KV so
    /// published full blocks drop to prefix-cache `ref_count == 1` (evictable).
    /// `Ok(false)` means no victim or the same victim is still eligible.
    /// Production admit uses [`Self::reclaim_one_idle_scheduled_with`] so tests
    /// can inject a failing unpin; this wrapper is test-only.
    #[cfg(test)]
    fn reclaim_one_idle_scheduled(&mut self, seq_id: SeqId) -> Result<bool> {
        self.reclaim_one_idle_scheduled_with(seq_id, B::release_scheduled_cache)
    }

    fn reclaim_one_idle_scheduled_with<F>(
        &mut self,
        seq_id: SeqId,
        release_cache: F,
    ) -> Result<bool>
    where
        F: FnOnce(&mut B, SeqId) -> Result<()>,
    {
        let Some(victim) = self.idle_memory_victim(seq_id) else {
            return Ok(false);
        };
        self.inner.release_scheduled_recurrent_for(victim);
        release_cache(&mut self.inner, victim)?;
        if !self.idle_reclaim_made_progress(seq_id, victim) {
            return Ok(false);
        }
        Ok(true)
    }

    fn idle_reclaim_made_progress(&self, seq_id: SeqId, victim: SeqId) -> bool {
        self.idle_memory_victim(seq_id) != Some(victim)
    }

    fn try_reserve_reclaiming_idle(
        &mut self,
        seq_id: SeqId,
        reservation_blocks: u32,
        candidate_state_bytes: u64,
        snapshot: SchedulerCacheSnapshot,
    ) -> Result<MemoryReserveOutcome> {
        self.try_reserve_reclaiming_idle_with(
            seq_id,
            reservation_blocks,
            candidate_state_bytes,
            snapshot,
            B::release_scheduled_cache,
        )
    }

    /// Unified-memory admission decision against one cache snapshot.
    fn reserve_memory_decision(
        &mut self,
        snapshot: &SchedulerCacheSnapshot,
        reservation_blocks: u32,
        candidate_state_bytes: u64,
    ) -> engine::scheduler::MemoryAdmission {
        self.scheduler.try_reserve_memory(
            engine::scheduler::MemoryTelemetry {
                capacity_bytes: u64::from(snapshot.blocks.total_blocks)
                    .saturating_mul(snapshot.bytes_per_block),
                free_bytes: u64::from(snapshot.blocks.free_blocks)
                    .saturating_mul(snapshot.bytes_per_block),
                reclaimable_bytes: u64::from(snapshot.blocks.reclaimable_blocks)
                    .saturating_mul(snapshot.bytes_per_block),
            },
            reservation_blocks,
            snapshot.bytes_per_block,
            self.inner.scheduled_recurrent_bytes(),
            candidate_state_bytes,
            scheduler_watermark_fraction(),
        )
    }

    fn try_reserve_reclaiming_idle_with<F>(
        &mut self,
        seq_id: SeqId,
        reservation_blocks: u32,
        candidate_state_bytes: u64,
        mut snapshot: SchedulerCacheSnapshot,
        mut release_cache: F,
    ) -> Result<MemoryReserveOutcome>
    where
        F: FnMut(&mut B, SeqId) -> Result<()>,
    {
        loop {
            let decision =
                self.reserve_memory_decision(&snapshot, reservation_blocks, candidate_state_bytes);
            if decision.admitted {
                return Ok(MemoryReserveOutcome::Admitted);
            }
            if !self.reclaim_one_idle_scheduled_with(seq_id, &mut release_cache)? {
                if self.scheduler.has_live_turns() {
                    return Ok(MemoryReserveOutcome::Defer);
                }
                // No live turns: one pool grow may still make the
                // reservation fit, so grow-and-retry once. The fit math
                // itself stays on CURRENT free + reclaimable counts —
                // capacity derived from the max pool would admit against
                // phantom free blocks. A declined or failed grow keeps the
                // existing Reject path byte-identical.
                if let Some(grown) = self.try_grow_reservation_pool(
                    &snapshot,
                    reservation_blocks,
                    candidate_state_bytes,
                )? {
                    snapshot = grown;
                    let decision = self.reserve_memory_decision(
                        &snapshot,
                        reservation_blocks,
                        candidate_state_bytes,
                    );
                    if decision.admitted {
                        return Ok(MemoryReserveOutcome::Admitted);
                    }
                }
                return Ok(MemoryReserveOutcome::Reject {
                    total_blocks: snapshot.blocks.total_blocks,
                });
            }
            if let Some(updated) = self.inner.scheduler_cache_snapshot()? {
                snapshot = updated;
            }
        }
    }

    /// One-shot pool grow for the no-live-turns Reject edge of
    /// `try_reserve_reclaiming_idle_with`: only when the pool sits below
    /// its max and the reservation still fits the max-pool ceiling does the
    /// pool grow by the byte shortfall (blocks plus recurrent state) over
    /// current free + reclaimable. Returns the fresh snapshot so the caller
    /// can re-run the reserve decision once. `Ok(None)` covers every
    /// keep-the-Reject-path case: no paged adapter, pool already at max,
    /// reservation above the max ceiling, zero shortfall, or a
    /// declined/failed grow. Admission runs on the model thread, so the
    /// adapter grow contract holds.
    fn try_grow_reservation_pool(
        &mut self,
        snapshot: &SchedulerCacheSnapshot,
        reservation_blocks: u32,
        candidate_state_bytes: u64,
    ) -> Result<Option<SchedulerCacheSnapshot>> {
        let Some(needed) = grow_needed_blocks(
            snapshot,
            reservation_blocks,
            candidate_state_bytes,
            self.inner.scheduled_recurrent_bytes(),
        ) else {
            return Ok(None);
        };
        let Some(adapter) = self.inner.paged_adapter_mut() else {
            return Ok(None);
        };
        if !adapter.try_grow_pool(needed) {
            return Ok(None);
        }
        self.inner.scheduler_cache_snapshot()
    }

    fn context_length_exceeded(reservation_blocks: u32, total_blocks: u32) -> Error {
        Error::from_reason(format!(
            "context_length_exceeded: request requires {} paged blocks but the pool has {}",
            reservation_blocks, total_blocks
        ))
    }

    fn context_length_exceeded_max(reservation_blocks: u32, max_total_blocks: u32) -> Error {
        Error::from_reason(format!(
            "context_length_exceeded: request requires {} paged blocks but the max pool holds {}",
            reservation_blocks, max_total_blocks
        ))
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
        let candidate_state_bytes = if self.inner.has_scheduled_recurrent(prepared.seq_id) {
            0
        } else {
            self.inner.recurrent_state_bytes()
        };
        if cache_snapshot.as_ref().is_some_and(|snapshot| {
            exceeds_max_pool_ceiling(snapshot, prepared.reservation_blocks, candidate_state_bytes)
        }) {
            self.cleanup_rejected_prepared(&prepared);
            let max_total_blocks = cache_snapshot.map_or(0, |snapshot| snapshot.max_total_blocks);
            prepared.response.send_error(
                Self::context_length_exceeded_max(prepared.reservation_blocks, max_total_blocks),
                prepared.cancelled.as_ref(),
            );
            return Ok(None);
        }
        if !self.ensure_recurrent_slot(prepared.seq_id) {
            return Err(prepared);
        }
        if let Some(snapshot) = cache_snapshot {
            let reuse = prepared.admitted.plan.is_delta || prepared.admitted.params.reuse_cache;
            let charge = prepared
                .reservation_blocks
                .saturating_sub(reusable_keep_live_blocks(
                    self.inner.paged_adapter(),
                    prepared.seq_id,
                    &prepared.admitted.tokens,
                    reuse,
                ));
            match self.try_reserve_reclaiming_idle(
                prepared.seq_id,
                charge,
                candidate_state_bytes,
                snapshot,
            ) {
                Ok(MemoryReserveOutcome::Admitted) => {}
                Ok(MemoryReserveOutcome::Defer) => return Err(prepared),
                Ok(MemoryReserveOutcome::Reject { total_blocks }) => {
                    self.cleanup_rejected_prepared(&prepared);
                    prepared.response.send_error(
                        Self::context_length_exceeded(prepared.reservation_blocks, total_blocks),
                        prepared.cancelled.as_ref(),
                    );
                    return Ok(None);
                }
                Err(error) => {
                    self.cleanup_rejected_prepared(&prepared);
                    prepared
                        .response
                        .send_error(error, prepared.cancelled.as_ref());
                    return Ok(None);
                }
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
            turn_token_observer: self.inner.turn_token_observer(),
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
                self.fail_preempted(turn, error.reason.clone());
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
            self.fail_preempted(turn, error.reason.clone());
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
                self.fail_preempted(turn, error.reason.clone());
                return;
            }
        };
        if !self.ensure_recurrent_slot(turn.seq_id) {
            self.scheduler.prepend_preempted(turn);
            return;
        }
        if let Some(snapshot) = cache_snapshot {
            let materialized = self.inner.scheduler_materialized_blocks(turn.seq_id);
            let charge = if materialized > 0 {
                turn.block_reservation_total.saturating_sub(materialized)
            } else {
                turn.block_reservation_total
            };
            match self.try_reserve_reclaiming_idle(
                turn.seq_id,
                charge,
                turn.recurrent_state_bytes,
                snapshot,
            ) {
                Ok(MemoryReserveOutcome::Admitted) => {}
                Ok(MemoryReserveOutcome::Defer) => {
                    self.scheduler.prepend_preempted(turn);
                    return;
                }
                Ok(MemoryReserveOutcome::Reject { total_blocks }) => {
                    let reservation_blocks = turn.block_reservation_total;
                    self.fail_preempted(
                        turn,
                        Self::context_length_exceeded(reservation_blocks, total_blocks)
                            .reason
                            .clone(),
                    );
                    return;
                }
                Err(error) => {
                    self.fail_preempted(turn, error.reason.clone());
                    return;
                }
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
            self.fail_preempted(turn, error.reason.clone());
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
                self.fail_preempted(turn, error.reason.clone());
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
    use crate::engine::backend::ChatBackend;
    use crate::engine::plan::{DecoderPlan, MediaCapabilities, TurnPath, TurnPlan, TurnRequest};
    use crate::models::nemotron_h::config::NemotronHConfig;
    use crate::models::nemotron_h::model::NemotronHInner;
    use crate::models::qwen3_5::config::Qwen3_5Config;
    use crate::models::qwen3_5::model::{Qwen35Cmd, Qwen35Inner};
    use crate::models::qwen3_5_moe::config::Qwen3_5MoeConfig;
    use crate::models::qwen3_5_moe::model::Qwen35MoeInner;

    fn tiny_config() -> Qwen3_5Config {
        Qwen3_5Config {
            qwen35_gguf_gdn_layout: None,
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
            paged_cache_initial_memory_mb: None,
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
            paged_cache_initial_memory_mb: None,
            paged_block_size: Some(16),
            use_block_paged_cache: None,
            persist_paged_cache: None,
            n_mtp_layers: 0,
            qwen35_gguf_gdn_layout: None,
        }
    }

    fn tiny_nemotron_paged_config() -> NemotronHConfig {
        NemotronHConfig {
            vocab_size: 32,
            hidden_size: 256,
            num_hidden_layers: 3,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 128,
            max_position_embeddings: 512,
            layer_norm_epsilon: 1e-5,
            layers_block_type: vec![
                "linear_attention".into(),
                "moe".into(),
                "full_attention".into(),
            ],
            mamba_num_heads: 2,
            mamba_head_dim: 2,
            ssm_state_size: 2,
            n_groups: 1,
            conv_kernel: 4,
            chunk_size: 4,
            time_step_min: 0.001,
            time_step_limit: None,
            n_routed_experts: 4,
            num_experts_per_tok: 1,
            routed_scaling_factor: 1.0,
            norm_topk_prob: true,
            intermediate_size: 6,
            moe_shared_expert_intermediate_size: 8,
            eos_token_ids: vec![2],
            mtp_layers_block_type: Vec::new(),
            n_mtp_layers: 0,
            paged_cache_memory_mb: Some(256),
            paged_block_size: Some(16),
            use_block_paged_cache: Some(true),
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

    /// Paged adapter + a complete native MTP head whose flat core has no
    /// streaming arm: the shape whose lane and decoder answers used to be
    /// derived in two different places.
    fn tiny_nemotron_paged_mtp_config() -> NemotronHConfig {
        NemotronHConfig {
            vocab_size: 32,
            hidden_size: 256,
            num_hidden_layers: 3,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 128,
            max_position_embeddings: 512,
            layer_norm_epsilon: 1e-5,
            layers_block_type: vec![
                "linear_attention".into(),
                "moe".into(),
                "full_attention".into(),
            ],
            mamba_num_heads: 2,
            mamba_head_dim: 2,
            ssm_state_size: 2,
            n_groups: 1,
            conv_kernel: 4,
            chunk_size: 4,
            time_step_min: 0.001,
            time_step_limit: None,
            n_routed_experts: 4,
            num_experts_per_tok: 1,
            routed_scaling_factor: 1.0,
            norm_topk_prob: true,
            intermediate_size: 6,
            moe_shared_expert_intermediate_size: 8,
            eos_token_ids: vec![2],
            mtp_layers_block_type: vec!["full_attention".into(), "moe".into()],
            n_mtp_layers: 1,
            paged_cache_memory_mb: Some(256),
            paged_block_size: Some(16),
            use_block_paged_cache: Some(true),
        }
    }

    fn mtp_config() -> ChatConfig {
        ChatConfig {
            enable_mtp: Some(true),
            ..ChatConfig::default()
        }
    }

    fn sync_start(
        config: ChatConfig,
    ) -> (ChatCmd, tokio::sync::oneshot::Receiver<Result<ChatResult>>) {
        let (reply, result) = tokio::sync::oneshot::channel();
        (
            ChatCmd::SessionStart {
                messages: Vec::new(),
                config,
                reply,
                cancelled: Arc::new(AtomicBool::new(false)),
            },
            result,
        )
    }

    fn streaming_start(
        config: ChatConfig,
    ) -> (
        ChatCmd,
        tokio::sync::mpsc::Receiver<Result<crate::engine::types::ChatStreamChunk>>,
    ) {
        let (stream_tx, rx) = crate::model_thread::stream_channel(4);
        (
            ChatCmd::StreamSessionStart {
                messages: Vec::new(),
                config,
                stream_tx,
                cancelled: Arc::new(AtomicBool::new(false)),
            },
            rx,
        )
    }

    /// L-LANE, the seam this whole gate exists for: whenever the scheduler
    /// declines to barrier a turn, the PLANNER must resolve that same turn to
    /// an autoregressive decoder. A model that speculates on the scheduled
    /// lane has no per-row drafter state, so a gap here is a correctness bug,
    /// not a perf one.
    ///
    /// Driven against a REAL NemotronH plan, whose speculative decoder is
    /// admitted for sync turns and refused for streaming ones — the one
    /// family where the two answers can differ.
    ///
    /// MUTATIONS:
    ///   * barrier on `config.enable_mtp` alone (drop the
    ///     `speculation_requires_exclusive_lane` conjunct) — the streaming leg
    ///     here reports a barrier the planner does not need. This one kills
    ///     BOTH this test and
    ///     `enable_mtp_without_a_loaded_decoder_keeps_the_scheduled_lane`,
    ///     whose fixture carries no head at all;
    ///   * drop `admits_streaming` from `speculation_requires_exclusive_lane`
    ///     — fails exactly this test: the streaming leg barriers while the
    ///     plan says autoregressive.
    #[test]
    fn the_barrier_gate_never_declines_a_turn_the_planner_will_speculate_on() {
        let mut inner =
            NemotronHInner::new(tiny_nemotron_paged_mtp_config()).expect("construct tiny nemotron");
        inner.mtp_weights_loaded = true;
        assert!(
            inner.paged_adapter.is_some(),
            "the seam needs the paged adapter exposed"
        );
        let state = HybridSchedulerState::new(inner).expect("construct scheduler");

        for streaming in [false, true] {
            let (command, _keepalive): (ChatCmd, Box<dyn std::any::Any>) = if streaming {
                let (command, rx) = streaming_start(mtp_config());
                (command, Box::new(rx))
            } else {
                let (command, rx) = sync_start(mtp_config());
                (command, Box::new(rx))
            };
            let barriers = state.chat_requires_barrier(&command);
            let plan = TurnPlan::resolve(
                ChatBackend::execution_plan(&state.inner),
                TurnRequest {
                    is_delta: false,
                    input_media: MediaCapabilities::NONE,
                    context_media: MediaCapabilities::NONE,
                    speculative_requested: true,
                    streaming,
                },
            );
            let speculates = matches!(plan.decoder, DecoderPlan::Speculative(_));
            assert_eq!(
                barriers, speculates,
                "lane gate and planner disagree (streaming={streaming}): \
                 barriers={barriers}, plan={plan:?}"
            );
        }

        // The two answers, spelled out, so a mutation that flips BOTH in step
        // cannot pass on the equality above alone.
        let (sync, _sync_rx) = sync_start(mtp_config());
        assert!(
            state.chat_requires_barrier(&sync),
            "a sync MTP turn runs the flat speculative core and needs the exclusive lane"
        );
        let (streamed, _stream_rx) = streaming_start(mtp_config());
        assert!(
            !state.chat_requires_barrier(&streamed),
            "the flat MTP core has no streaming arm, so a streaming MTP turn is plain \
             paged AR and must not stall the whole model"
        );
        assert_eq!(
            TurnPlan::resolve(
                ChatBackend::execution_plan(&state.inner),
                TurnRequest {
                    is_delta: false,
                    input_media: MediaCapabilities::NONE,
                    context_media: MediaCapabilities::NONE,
                    speculative_requested: true,
                    streaming: true,
                },
            )
            .path(),
            TurnPath::Paged,
            "the refused streaming turn belongs to the scheduler's own paged AR lane"
        );
    }

    /// A model with NO speculative decoder must not barrier on `enable_mtp`:
    /// the flag is a request, and the plan for it is plain autoregressive.
    ///
    /// MUTATION: barrier on `config.enable_mtp` alone — the first assertion
    /// fails, since this dense fixture carries no MTP head.
    #[test]
    fn enable_mtp_without_a_loaded_decoder_keeps_the_scheduled_lane() {
        let inner = Qwen35Inner::new(tiny_config()).expect("construct tiny dense model");
        let state = HybridSchedulerState::new(inner).expect("construct scheduler");
        assert!(
            ChatBackend::execution_plan(&state.inner)
                .speculative
                .is_none(),
            "the fixture must carry no MTP head"
        );

        let (mtp_start, _mtp_result) = sync_start(mtp_config());
        assert!(
            !state.chat_requires_barrier(&mtp_start),
            "an enable_mtp turn on a model with nothing to speculate with resolves \
             autoregressive and belongs on the scheduled lane"
        );
        assert_eq!(
            TurnPlan::resolve(
                ChatBackend::execution_plan(&state.inner),
                TurnRequest {
                    is_delta: false,
                    input_media: MediaCapabilities::NONE,
                    context_media: MediaCapabilities::NONE,
                    speculative_requested: true,
                    streaming: false,
                },
            )
            .decoder,
            DecoderPlan::Autoregressive,
        );

        let (plain_start, _plain_result) = sync_start(ChatConfig::default());
        assert!(!state.chat_requires_barrier(&plain_start));
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
    fn memory_denial_reclaims_idle_parked_row_unit_cap_would_keep() {
        let inner = Qwen35Inner::new(tiny_config()).expect("construct tiny dense model");
        let mut state = HybridSchedulerState::new(inner).expect("construct scheduler");
        state.owner_sequences.insert("history-only".into(), 3);
        state.owner_sequences.insert("parked".into(), 1);
        state
            .inner
            .activate_scheduled_recurrent(1)
            .expect("activate parked owner");
        state
            .inner
            .park_active_scheduled_recurrent()
            .expect("park seq 1");
        if let Some(adapter) = state.inner.paged_adapter_mut() {
            adapter
                .begin_request(1)
                .expect("parked owner can hold a keep-live request slot");
            assert!(adapter.block_table_for(1).is_some());
        }

        assert!(state.inner.has_scheduled_recurrent(1));
        assert!(
            state.ensure_recurrent_slot(2),
            "one parked idle row does not fill the two-unit cap"
        );
        assert!(
            state.inner.has_scheduled_recurrent(1),
            "unit-cap eviction must not fire with a single parked owner"
        );
        assert_eq!(state.idle_recurrent_victim(2), Some(1));
        assert_ne!(
            state.idle_recurrent_victim(2),
            Some(3),
            "history-only owners are not reclaim victims"
        );

        assert!(
            state.reclaim_one_idle_scheduled(2).expect("cache release"),
            "memory-denial reclaim must evict the idle parked row"
        );
        assert!(
            !state.inner.has_scheduled_recurrent(1),
            "reclaim drops the parked GDN/Mamba row"
        );
        assert!(
            state
                .inner
                .paged_adapter()
                .and_then(|adapter| adapter.block_table_for(1))
                .is_none(),
            "reclaim unpins keep-live KV so published full blocks become evictable"
        );
        assert!(state.inner.can_activate_scheduled_recurrent(2));
        assert!(!state.inner.has_scheduled_recurrent(3));
    }

    #[test]
    fn reclaiming_parked_recurrent_admits_usable_pool_reservation() {
        let inner = Qwen35Inner::new(tiny_config()).expect("construct tiny dense model");
        let mut state = HybridSchedulerState::new(inner).expect("construct scheduler");
        state.owner_sequences.insert("parked".into(), 1);
        state
            .inner
            .activate_scheduled_recurrent(1)
            .expect("activate parked owner");
        state
            .inner
            .park_active_scheduled_recurrent()
            .expect("park seq 1");

        let rec = state.inner.recurrent_state_bytes();
        assert!(rec > 0, "tiny Qwen3.5 fixture must have GDN state");
        assert_eq!(state.inner.scheduled_recurrent_bytes(), rec);
        // Qwen3.5 unit constructors leave the paged adapter unbuilt; size a
        // synthetic pool the same way admit does: KV at usable tokens after
        // one recurrent row.
        let block_size = 16u32;
        let bytes_per_block = 1024u64;
        let rec_tokens = rec
            .div_ceil(bytes_per_block)
            .saturating_mul(u64::from(block_size))
            .min(u64::from(u32::MAX)) as u32;
        let pool_tokens = rec_tokens.saturating_add(256);
        let usable = pool_tokens_after_recurrent(pool_tokens, block_size, bytes_per_block, rec);
        let reservation_blocks = usable.div_ceil(block_size);
        let total_blocks = pool_tokens / block_size;
        let snapshot = SchedulerCacheSnapshot {
            blocks: BlockTelemetry {
                total_blocks,
                free_blocks: total_blocks,
                reclaimable_blocks: 0,
                allocated_blocks: 0,
            },
            max_total_blocks: total_blocks,
            bytes_per_block,
        };
        assert!(reservation_fits_empty_pool(
            reservation_blocks,
            total_blocks,
            bytes_per_block,
            rec,
        ));

        let telemetry = engine::scheduler::MemoryTelemetry {
            capacity_bytes: u64::from(total_blocks).saturating_mul(bytes_per_block),
            free_bytes: u64::from(total_blocks).saturating_mul(bytes_per_block),
            reclaimable_bytes: 0,
        };
        assert!(
            !engine::scheduler::memory_admission_decision(
                telemetry,
                0,
                reservation_blocks,
                bytes_per_block,
                rec,
                rec,
                false,
                0.0,
            )
            .admitted,
            "parked row plus candidate overflows a usable-pool KV reservation"
        );

        let outcome = state
            .try_reserve_reclaiming_idle(2, reservation_blocks, rec, snapshot)
            .expect("reserve after reclaim");
        assert!(
            matches!(outcome, MemoryReserveOutcome::Admitted),
            "idle reclaim must admit, not defer: {outcome:?}"
        );
        assert!(!state.inner.has_scheduled_recurrent(1));
        assert!(
            engine::scheduler::memory_admission_decision(
                telemetry,
                0,
                reservation_blocks,
                bytes_per_block,
                state.inner.scheduled_recurrent_bytes(),
                rec,
                false,
                0.0,
            )
            .admitted
        );
    }

    #[test]
    fn idle_memory_denial_without_idle_victims_rejects() {
        let inner = Qwen35Inner::new(tiny_config()).expect("construct tiny dense model");
        let mut state = HybridSchedulerState::new(inner).expect("construct scheduler");
        assert!(state.idle_recurrent_victim(2).is_none());
        assert!(!state.scheduler.has_live_turns());
        let snapshot = SchedulerCacheSnapshot {
            blocks: BlockTelemetry {
                total_blocks: 100,
                free_blocks: 0,
                reclaimable_blocks: 0,
                allocated_blocks: 100,
            },
            max_total_blocks: 100,
            bytes_per_block: 4096,
        };
        let outcome = state
            .try_reserve_reclaiming_idle(2, 10, 1, snapshot)
            .expect("no snapshot refresh");
        assert!(
            matches!(outcome, MemoryReserveOutcome::Reject { total_blocks: 100 }),
            "idle scheduler must hard-error, not spin in prepared_waiting: {outcome:?}"
        );
    }

    #[test]
    fn reclaim_without_idle_victim_returns_ok_false() {
        let inner = Qwen35Inner::new(tiny_config()).expect("construct tiny dense model");
        let mut state = HybridSchedulerState::new(inner).expect("construct scheduler");
        assert!(state.idle_memory_victim(2).is_none());
        let progress = state
            .reclaim_one_idle_scheduled(2)
            .expect("no victim does not release cache");
        assert!(!progress, "no idle victim is not reclaim progress");
    }

    #[test]
    fn idle_reclaim_makes_no_progress_while_same_victim_eligible() {
        let inner = Qwen35Inner::new(tiny_config()).expect("construct tiny dense model");
        let mut state = HybridSchedulerState::new(inner).expect("construct scheduler");
        state.owner_sequences.insert("parked".into(), 1);
        state
            .inner
            .activate_scheduled_recurrent(1)
            .expect("activate parked owner");
        state
            .inner
            .park_active_scheduled_recurrent()
            .expect("park seq 1");
        assert_eq!(state.idle_memory_victim(2), Some(1));
        assert!(
            !state.idle_reclaim_made_progress(2, 1),
            "same victim still eligible is no progress"
        );
        state.inner.release_scheduled_recurrent_for(1);
        assert!(
            state.idle_memory_victim(2).is_none(),
            "rec release drops the Qwen3.5 victim"
        );
        assert!(
            state.idle_reclaim_made_progress(2, 1),
            "cleared victim is progress"
        );
    }

    #[test]
    fn reclaim_propagates_cache_release_error() {
        let inner = Qwen35Inner::new(tiny_config()).expect("construct tiny dense model");
        let mut state = HybridSchedulerState::new(inner).expect("construct scheduler");
        state.owner_sequences.insert("parked".into(), 1);
        state
            .inner
            .activate_scheduled_recurrent(1)
            .expect("activate parked owner");
        state
            .inner
            .park_active_scheduled_recurrent()
            .expect("park seq 1");
        let error = state
            .reclaim_one_idle_scheduled_with(2, |_inner, victim| {
                assert_eq!(victim, 1);
                Err(Error::from_reason("injected cache release failure"))
            })
            .expect_err("cache release Err must propagate");
        assert_eq!(error.reason, "injected cache release failure");
    }

    #[test]
    fn try_reserve_propagates_cache_release_error() {
        let inner = Qwen35Inner::new(tiny_config()).expect("construct tiny dense model");
        let mut state = HybridSchedulerState::new(inner).expect("construct scheduler");
        state.owner_sequences.insert("parked".into(), 1);
        state
            .inner
            .activate_scheduled_recurrent(1)
            .expect("activate parked owner");
        state
            .inner
            .park_active_scheduled_recurrent()
            .expect("park seq 1");
        let snapshot = SchedulerCacheSnapshot {
            blocks: BlockTelemetry {
                total_blocks: 100,
                free_blocks: 0,
                reclaimable_blocks: 0,
                allocated_blocks: 100,
            },
            max_total_blocks: 100,
            bytes_per_block: 4096,
        };
        let error = state
            .try_reserve_reclaiming_idle_with(2, 10, 1, snapshot, |_inner, victim| {
                assert_eq!(victim, 1);
                Err(Error::from_reason("injected cache release failure"))
            })
            .expect_err("deny + failed unpin must not spin");
        assert_eq!(error.reason, "injected cache release failure");
    }

    #[test]
    fn continuation_credits_keep_live_blocks_against_reservation() {
        let inner = Qwen35Inner::new(tiny_config()).expect("construct tiny dense model");
        let mut state = HybridSchedulerState::new(inner).expect("construct scheduler");
        assert!(state.idle_recurrent_victim(2).is_none());
        assert!(state.idle_memory_victim(2).is_none());
        assert!(!state.scheduler.has_live_turns());
        let snapshot = SchedulerCacheSnapshot {
            blocks: BlockTelemetry {
                total_blocks: 100,
                free_blocks: 40,
                reclaimable_blocks: 0,
                allocated_blocks: 60,
            },
            max_total_blocks: 100,
            bytes_per_block: 4096,
        };
        let reservation_blocks = 70u32;
        let already_materialized = 60u32;
        assert!(reservation_fits_empty_pool(
            reservation_blocks,
            snapshot.blocks.total_blocks,
            snapshot.bytes_per_block,
            0,
        ));
        let without_credit = state
            .try_reserve_reclaiming_idle(2, reservation_blocks, 0, snapshot)
            .expect("no snapshot refresh");
        assert!(
            matches!(
                without_credit,
                MemoryReserveOutcome::Reject { total_blocks: 100 }
            ),
            "70 vs free 40 without keep-live credit must Reject: {without_credit:?}"
        );
        let charge = reservation_blocks.saturating_sub(already_materialized);
        assert_eq!(charge, 10);
        let with_credit = state
            .try_reserve_reclaiming_idle(2, charge, 0, snapshot)
            .expect("no snapshot refresh");
        assert!(
            matches!(with_credit, MemoryReserveOutcome::Admitted),
            "charge 10 vs free 40 must admit: {with_credit:?}"
        );
    }

    #[test]
    fn grow_needed_blocks_measures_max_fitting_shortfall() {
        let snapshot = SchedulerCacheSnapshot {
            blocks: BlockTelemetry {
                total_blocks: 40,
                free_blocks: 10,
                reclaimable_blocks: 5,
                allocated_blocks: 25,
            },
            max_total_blocks: 100,
            bytes_per_block: 4096,
        };
        assert_eq!(
            grow_needed_blocks(&snapshot, 70, 0, 0),
            Some(55),
            "70 reserved - (10 free + 5 reclaimable) = 55 blocks to grow"
        );
        // No existing state: the byte formula matches the block formula.
        // Zero shortfall means the reservation fits current free +
        // reclaimable bytes, so there is nothing to grow for.
        assert_eq!(grow_needed_blocks(&snapshot, 15, 0, 0), None);
        // A pool already at its max cannot grow.
        let at_max = SchedulerCacheSnapshot {
            blocks: BlockTelemetry {
                total_blocks: 100,
                free_blocks: 10,
                reclaimable_blocks: 0,
                allocated_blocks: 90,
            },
            max_total_blocks: 100,
            bytes_per_block: 4096,
        };
        assert_eq!(grow_needed_blocks(&at_max, 60, 0, 0), None);
        // Above the max ceiling, admission must hard-error, not grow.
        assert_eq!(grow_needed_blocks(&snapshot, 101, 0, 0), None);
        // Candidate recurrent state can push the reservation past the max.
        assert_eq!(grow_needed_blocks(&snapshot, 70, 400_000, 0), None);
    }

    #[test]
    fn grow_needed_blocks_covers_state_byte_overflow() {
        let snapshot = SchedulerCacheSnapshot {
            blocks: BlockTelemetry {
                total_blocks: 40,
                free_blocks: 15,
                reclaimable_blocks: 0,
                allocated_blocks: 25,
            },
            max_total_blocks: 100,
            bytes_per_block: 4096,
        };
        // Blocks fit free exactly, but the candidate GDN state overflows the
        // shared byte budget: grow ceil(10_000 / 4096) = 3 blocks.
        assert_eq!(grow_needed_blocks(&snapshot, 15, 10_000, 0), Some(3));
        // Already-scheduled state charges the same budget.
        assert_eq!(grow_needed_blocks(&snapshot, 15, 5_000, 5_000), Some(3));
        // Positive block shortfall plus state: the grow must cover the byte
        // sum, not just the block count. 20*4096 + 10_000 - 15*4096 =
        // 30_480 bytes -> ceil = 8 blocks (block-only math would say 5).
        let needed = grow_needed_blocks(&snapshot, 20, 10_000, 0);
        assert_eq!(needed, Some(8));
        assert!(
            u64::from(needed.expect("grow blocks")) * 4096 >= 30_480,
            "grown blocks must cover the full byte shortfall"
        );
        // Recurrent state alone can push a block-fitting reservation past
        // the max-pool ceiling: hard-error, no grow.
        let fits_max_blocks = SchedulerCacheSnapshot {
            blocks: BlockTelemetry {
                total_blocks: 40,
                free_blocks: 15,
                reclaimable_blocks: 0,
                allocated_blocks: 25,
            },
            max_total_blocks: 100,
            bytes_per_block: 4096,
        };
        assert_eq!(
            grow_needed_blocks(&fits_max_blocks, 50, 300_000, 0),
            None,
            "50 blocks fit the 100-block max, but +300_000 state bytes do not"
        );
    }

    /// Real adapter over a 4-block pool with a 16-block max so a scheduler
    /// grow is observable. Panics without Metal; callers must skip first.
    fn tiny_growable_paged_adapter() -> PagedKVCacheAdapter {
        let cfg = mlx_paged_attn::PagedAttentionConfig {
            block_size: 16,
            num_kv_heads: 1,
            head_size: 128,
            num_layers: 1,
            gpu_memory_mb: 256,
            use_fp8_cache: Some(false),
            max_seq_len: Some(512),
            max_batch_size: Some(32),
        };
        let pool = mlx_paged_attn::LayerKVPool::new(
            cfg,
            4,
            16,
            mlx_paged_attn::metal::MetalDtype::BFloat16,
        )
        .expect("growable test LayerKVPool");
        let allocator = Arc::new(std::sync::Mutex::new(mlx_paged_attn::BlockAllocator::new(
            4, 16, 16,
        )));
        PagedKVCacheAdapter::new(allocator, Arc::new(pool), 16).expect("growable test adapter")
    }

    /// Nemotron-backed scheduler whose paged adapter is the growable
    /// fixture (4 live blocks, 16 max) plus its real cache snapshot.
    /// Returns `None` to skip when Metal is unavailable.
    fn growable_scheduler_state()
    -> Option<(HybridSchedulerState<NemotronHInner>, SchedulerCacheSnapshot)> {
        let mut inner = match NemotronHInner::new(tiny_nemotron_paged_config()) {
            Ok(inner) => inner,
            Err(_) => {
                eprintln!("skipping grow-path scheduler test: inner failed");
                return None;
            }
        };
        if inner.paged_adapter().is_none() {
            eprintln!("skipping grow-path scheduler test: Metal unavailable");
            return None;
        }
        inner.paged_adapter = Some(tiny_growable_paged_adapter());
        let state = HybridSchedulerState::new(inner).expect("construct nemotron scheduler");
        let snapshot = state
            .inner
            .scheduler_cache_snapshot()
            .expect("scheduler snapshot")
            .expect("paged adapter present");
        Some((state, snapshot))
    }

    #[test]
    fn no_live_turns_reservation_grows_pool_once_and_admits() {
        let Some((mut state, snapshot)) = growable_scheduler_state() else {
            return;
        };
        assert_eq!(snapshot.blocks.total_blocks, 4);
        assert_eq!(snapshot.blocks.free_blocks, 4);
        assert_eq!(snapshot.max_total_blocks, 16);
        assert!(!state.scheduler.has_live_turns());
        assert!(state.idle_memory_victim(2).is_none());
        // Force the headroom probe to pass so the one-shot grow outcome
        // does not depend on the live Metal budget.
        state
            .inner
            .paged_adapter_mut()
            .expect("paged adapter")
            .set_grow_headroom_probe_override(Some(Arc::new(Ok)));
        let charge = snapshot.blocks.free_blocks + 1;
        let outcome = state
            .try_reserve_reclaiming_idle(2, charge, 0, snapshot)
            .expect("grow + re-reserve");
        assert!(
            matches!(outcome, MemoryReserveOutcome::Admitted),
            "a max-fitting reservation above free+reclaimable must grow and admit: {outcome:?}"
        );
        let grown = state
            .inner
            .scheduler_cache_snapshot()
            .expect("post-grow snapshot")
            .expect("paged adapter present");
        assert_eq!(
            grown.blocks.total_blocks, 8,
            "grow target min(16, max(2*4, 4+1)) is 8 blocks"
        );
    }

    #[test]
    fn state_overflow_reservation_grows_pool_and_admits() {
        let Some((mut state, snapshot)) = growable_scheduler_state() else {
            return;
        };
        state
            .inner
            .paged_adapter_mut()
            .expect("paged adapter")
            .set_grow_headroom_probe_override(Some(Arc::new(Ok)));
        let bytes_per_block = snapshot.bytes_per_block;
        // The reservation blocks fit current free exactly; only the
        // candidate recurrent state overflows the shared byte budget.
        let reservation_blocks = snapshot.blocks.free_blocks;
        let candidate_state_bytes = u64::from(snapshot.blocks.free_blocks) * bytes_per_block + 1;
        let outcome = state
            .try_reserve_reclaiming_idle(2, reservation_blocks, candidate_state_bytes, snapshot)
            .expect("grow + re-reserve");
        assert!(
            matches!(outcome, MemoryReserveOutcome::Admitted),
            "a state-byte overflow within the max pool must grow and admit: {outcome:?}"
        );
        let grown = state
            .inner
            .scheduler_cache_snapshot()
            .expect("post-grow snapshot")
            .expect("paged adapter present");
        let needed = candidate_state_bytes.div_ceil(bytes_per_block) as u32;
        assert_eq!(
            grown.blocks.total_blocks,
            4 + needed,
            "grow target min(16, max(2*4, 4+{needed})) covers blocks plus state"
        );
    }

    #[test]
    fn declined_grow_preserves_no_live_turns_reject() {
        let Some((mut state, snapshot)) = growable_scheduler_state() else {
            return;
        };
        // Probe caps the pool below the shortfall target: try_grow_pool
        // declines. (One below the DOUBLING target is partial headroom and
        // grows — the decline edge only fires below current + needed.)
        state
            .inner
            .paged_adapter_mut()
            .expect("paged adapter")
            .set_grow_headroom_probe_override(Some(Arc::new(|requested| {
                let _ = requested;
                Ok(4)
            })));
        let charge = snapshot.blocks.free_blocks + 1;
        let outcome = state
            .try_reserve_reclaiming_idle(2, charge, 0, snapshot)
            .expect("declined grow keeps Reject");
        assert!(
            matches!(outcome, MemoryReserveOutcome::Reject { total_blocks: 4 }),
            "a declined grow must keep the Reject path naming current counts: {outcome:?}"
        );
        assert_eq!(
            state
                .inner
                .scheduler_cache_snapshot()
                .expect("post-decline snapshot")
                .expect("paged adapter present")
                .blocks
                .total_blocks,
            4,
            "a declined grow must not grow the pool"
        );
    }

    #[test]
    fn partial_headroom_grows_to_selected_and_admits() {
        let Some((mut state, snapshot)) = growable_scheduler_state() else {
            return;
        };
        // Probe caps the 4 -> 8 doubling target at 7 blocks: above the
        // shortfall (current 4 + needed 1 = 5), below the target. The grow
        // must land at the selected count, not decline.
        state
            .inner
            .paged_adapter_mut()
            .expect("paged adapter")
            .set_grow_headroom_probe_override(Some(Arc::new(|requested| {
                Ok(requested.saturating_sub(1))
            })));
        let charge = snapshot.blocks.free_blocks + 1;
        let outcome = state
            .try_reserve_reclaiming_idle(2, charge, 0, snapshot)
            .expect("partial grow + re-reserve");
        assert!(
            matches!(outcome, MemoryReserveOutcome::Admitted),
            "a partial-headroom grow covering the shortfall must admit: {outcome:?}"
        );
        assert_eq!(
            state
                .inner
                .scheduler_cache_snapshot()
                .expect("post-grow snapshot")
                .expect("paged adapter present")
                .blocks
                .total_blocks,
            7,
            "the grow must land at the probe-selected 7 blocks"
        );
    }

    #[test]
    fn above_max_reservation_rejects_without_grow() {
        let Some((mut state, snapshot)) = growable_scheduler_state() else {
            return;
        };
        state
            .inner
            .paged_adapter_mut()
            .expect("paged adapter")
            .set_grow_headroom_probe_override(Some(Arc::new(Ok)));
        let outcome = state
            .try_reserve_reclaiming_idle(2, 17, 0, snapshot)
            .expect("above-max Reject");
        assert!(
            matches!(outcome, MemoryReserveOutcome::Reject { total_blocks: 4 }),
            "a reservation above the max pool must Reject, not grow: {outcome:?}"
        );
        assert_eq!(
            state
                .inner
                .scheduler_cache_snapshot()
                .expect("post-reject snapshot")
                .expect("paged adapter present")
                .blocks
                .total_blocks,
            4,
            "an above-max reservation must not grow the pool"
        );
    }

    /// Admission's must-fit hard-reject is measured against the MAX pool the
    /// grow hook can reach during turn execution, not the CURRENT block
    /// count. A reservation above the current pool but below the max must
    /// NOT be hard-rejected at admission (the live free-blocks logic then
    /// admits or defers it exactly as before); only a reservation above the
    /// max is rejected, with the `context_length_exceeded` family marker
    /// naming the max pool size.
    #[test]
    fn admission_ceiling_uses_max_pool_not_current_pool() {
        let snapshot = SchedulerCacheSnapshot {
            blocks: BlockTelemetry {
                total_blocks: 40,
                free_blocks: 40,
                reclaimable_blocks: 0,
                allocated_blocks: 0,
            },
            max_total_blocks: 100,
            bytes_per_block: 4096,
        };
        let reservation_blocks = 70u32;
        // Guards the regression: the old current-pool ceiling would have
        // hard-rejected 70 blocks against a 40-block live pool.
        assert!(
            !reservation_fits_empty_pool(
                reservation_blocks,
                snapshot.blocks.total_blocks,
                snapshot.bytes_per_block,
                0,
            ),
            "70 blocks do not fit the CURRENT 40-block pool"
        );
        assert!(
            !exceeds_max_pool_ceiling(&snapshot, reservation_blocks, 0),
            "70 blocks fit the 100-block MAX pool: admission must not hard-reject, \
             so the turn can defer/admit while the grow hook reaches it"
        );
        assert!(
            exceeds_max_pool_ceiling(&snapshot, 101, 0),
            "101 blocks exceed the max pool: admission must reject"
        );
        let error =
            HybridSchedulerState::<Qwen35Inner>::context_length_exceeded_max(101, 100).reason;
        assert!(
            error.starts_with("context_length_exceeded"),
            "max-ceiling reject keeps the family marker: {error}"
        );
        assert!(
            error.contains("max pool holds 100"),
            "max-ceiling reject states the max pool size, not the current one: {error}"
        );
    }

    #[test]
    fn reusable_keep_live_credits_prefix_match_not_mismatch() {
        assert_eq!(
            reusable_keep_live_blocks(None, 1, &[1, 2, 3], true),
            0,
            "no adapter credits 0"
        );
        let inner = match NemotronHInner::new(tiny_nemotron_paged_config()) {
            Ok(inner) => inner,
            Err(_) => {
                eprintln!(
                    "skipping reusable_keep_live_credits_prefix_match_not_mismatch: inner failed"
                );
                return;
            }
        };
        let mut state = HybridSchedulerState::new(inner).expect("construct nemotron scheduler");
        let Some(adapter) = state.inner.paged_adapter_mut() else {
            eprintln!(
                "skipping reusable_keep_live_credits_prefix_match_not_mismatch: Metal unavailable"
            );
            return;
        };
        let held: Vec<u32> = (1..33).collect();
        adapter.begin_request(1).expect("begin keep-live request");
        let _ = adapter
            .find_cached_prefix(&[], &[], 0, false)
            .expect("prefix lookup");
        adapter
            .allocate_suffix_blocks(held.len() as u32)
            .expect("allocate held blocks");
        adapter.record_tokens(&held).expect("record held tokens");
        adapter
            .finalize_turn_keep_live(&[], 0)
            .expect("keep-live finalize");
        let num_blocks = adapter
            .block_table_for(1)
            .expect("keep-live table")
            .num_blocks() as u32;
        assert!(num_blocks > 0, "held tokens must occupy blocks");
        let mut continued = held.clone();
        continued.extend_from_slice(&[99, 100]);
        let mut mismatch = held.clone();
        mismatch[0] = 0;
        mismatch.extend_from_slice(&[99, 100]);
        let adapter = state.inner.paged_adapter();
        assert_eq!(
            reusable_keep_live_blocks(adapter, 1, &continued, true),
            num_blocks,
            "prompt starting with held credits num_blocks"
        );
        assert_eq!(
            reusable_keep_live_blocks(adapter, 1, &mismatch, true),
            0,
            "mismatch credits 0; prefix will rebuild"
        );
        assert_eq!(
            reusable_keep_live_blocks(adapter, 1, &continued, false),
            0,
            "!reuse credits 0"
        );
    }

    #[test]
    fn cache_only_idle_owner_is_memory_victim_not_unit_cap_victim() {
        let inner = match NemotronHInner::new(tiny_nemotron_paged_config()) {
            Ok(inner) => inner,
            Err(_) => {
                eprintln!(
                    "skipping cache_only_idle_owner_is_memory_victim_not_unit_cap_victim: inner failed"
                );
                return;
            }
        };
        if inner.paged_adapter().is_none() {
            eprintln!(
                "skipping cache_only_idle_owner_is_memory_victim_not_unit_cap_victim: Metal unavailable"
            );
            return;
        }
        let mut state = HybridSchedulerState::new(inner).expect("construct nemotron scheduler");
        state.owner_sequences.insert("cache-only".into(), 1);
        state
            .inner
            .paged_adapter_mut()
            .expect("paged adapter")
            .begin_request(1)
            .expect("begin cache-only request");
        state.inner.release_scheduled_recurrent_for(1);
        assert!(
            !state.inner.has_scheduled_recurrent(1),
            "recurrent row is gone"
        );
        assert!(
            state
                .inner
                .paged_adapter()
                .and_then(|adapter| adapter.block_table_for(1))
                .is_some(),
            "block table remains after rec release"
        );
        assert!(
            state.idle_recurrent_victim(2).is_none(),
            "cache-only owner is not a unit-cap victim"
        );
        assert_eq!(
            state.idle_memory_victim(2),
            Some(1),
            "cache-only keep-live table is a memory victim"
        );
        assert!(
            state.reclaim_one_idle_scheduled(2).expect("cache release"),
            "memory-denial reclaim must unpin the cache-only table"
        );
        assert!(
            state
                .inner
                .paged_adapter()
                .and_then(|adapter| adapter.block_table_for(1))
                .is_none(),
            "reclaim unpins the keep-live table"
        );
        assert!(
            state.idle_memory_victim(2).is_none(),
            "successful cache-only reclaim must drop the victim"
        );
    }

    #[test]
    fn exclusive_seq_zero_keep_live_is_a_memory_victim() {
        let inner = match NemotronHInner::new(tiny_nemotron_paged_config()) {
            Ok(inner) => inner,
            Err(_) => {
                eprintln!("skipping exclusive_seq_zero_keep_live_is_a_memory_victim: inner failed");
                return;
            }
        };
        if inner.paged_adapter().is_none() {
            eprintln!(
                "skipping exclusive_seq_zero_keep_live_is_a_memory_victim: Metal unavailable"
            );
            return;
        }
        let mut state = HybridSchedulerState::new(inner).expect("construct nemotron scheduler");
        state
            .inner
            .paged_adapter_mut()
            .expect("paged adapter")
            .begin_request(0)
            .expect("exclusive lane keep-live");
        assert!(
            state.owner_sequences.is_empty(),
            "seq 0 is not entered in owner_sequences"
        );
        assert!(
            state.idle_recurrent_victim(1).is_none(),
            "seq 0 without rec is not a unit-cap victim"
        );
        assert_eq!(
            state.idle_memory_victim(1),
            Some(0),
            "exclusive keep-live table must be reclaimable"
        );
        assert!(
            state.reclaim_one_idle_scheduled(1).expect("unpin seq 0"),
            "memory-denial reclaim must unpin sequence 0"
        );
        assert!(
            state
                .inner
                .paged_adapter()
                .and_then(|adapter| adapter.block_table_for(0))
                .is_none(),
            "seq 0 table is gone after reclaim"
        );
    }

    #[test]
    fn cache_only_unpin_that_leaves_victim_is_not_progress() {
        let inner = match NemotronHInner::new(tiny_nemotron_paged_config()) {
            Ok(inner) => inner,
            Err(_) => {
                eprintln!(
                    "skipping cache_only_unpin_that_leaves_victim_is_not_progress: inner failed"
                );
                return;
            }
        };
        if inner.paged_adapter().is_none() {
            eprintln!(
                "skipping cache_only_unpin_that_leaves_victim_is_not_progress: Metal unavailable"
            );
            return;
        }
        let mut state = HybridSchedulerState::new(inner).expect("construct nemotron scheduler");
        state.owner_sequences.insert("cache-only".into(), 1);
        state
            .inner
            .paged_adapter_mut()
            .expect("paged adapter")
            .begin_request(1)
            .expect("begin cache-only request");
        state.inner.release_scheduled_recurrent_for(1);
        assert_eq!(state.idle_memory_victim(2), Some(1));
        let mut unpin_calls = 0u32;
        let progress = state
            .reclaim_one_idle_scheduled_with(2, |_inner, victim| {
                unpin_calls += 1;
                assert_eq!(victim, 1);
                Ok(())
            })
            .expect("noop unpin is Ok");
        assert_eq!(unpin_calls, 1);
        assert!(
            !progress,
            "same idle victim after a no-op unpin is not progress"
        );
        assert_eq!(state.idle_memory_victim(2), Some(1));
    }

    #[test]
    fn try_reserve_rejects_when_cache_only_unpin_makes_no_progress() {
        let inner = match NemotronHInner::new(tiny_nemotron_paged_config()) {
            Ok(inner) => inner,
            Err(_) => {
                eprintln!(
                    "skipping try_reserve_rejects_when_cache_only_unpin_makes_no_progress: inner failed"
                );
                return;
            }
        };
        if inner.paged_adapter().is_none() {
            eprintln!(
                "skipping try_reserve_rejects_when_cache_only_unpin_makes_no_progress: Metal unavailable"
            );
            return;
        }
        let mut state = HybridSchedulerState::new(inner).expect("construct nemotron scheduler");
        state.owner_sequences.insert("cache-only".into(), 1);
        state
            .inner
            .paged_adapter_mut()
            .expect("paged adapter")
            .begin_request(1)
            .expect("begin cache-only request");
        state.inner.release_scheduled_recurrent_for(1);
        assert_eq!(state.idle_memory_victim(2), Some(1));
        let snapshot = SchedulerCacheSnapshot {
            blocks: BlockTelemetry {
                total_blocks: 100,
                free_blocks: 0,
                reclaimable_blocks: 0,
                allocated_blocks: 100,
            },
            max_total_blocks: 100,
            bytes_per_block: 4096,
        };
        let mut unpin_calls = 0u32;
        let outcome = state
            .try_reserve_reclaiming_idle_with(2, 10, 0, snapshot, |_inner, victim| {
                unpin_calls += 1;
                assert!(
                    unpin_calls <= 2,
                    "reclaim loop must stop when the victim remains"
                );
                assert_eq!(victim, 1);
                Ok(())
            })
            .expect("noop unpin is Ok");
        assert_eq!(
            unpin_calls, 1,
            "no-progress reclaim must not retry the same victim"
        );
        assert!(
            matches!(outcome, MemoryReserveOutcome::Reject { total_blocks: 100 }),
            "no-progress reclaim must Reject, not spin: {outcome:?}"
        );
        assert_eq!(state.idle_memory_victim(2), Some(1));
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

#[cfg(test)]
mod scheduled_context_tests {
    use super::{pool_tokens_after_recurrent, reservation_fits_empty_pool, scheduled_turn_context};

    #[test]
    fn admit_uses_pool_when_env_unset() {
        // Incident numbers: 33611 prompt, 32768 old cap, ~349520 pool, 1M trained.
        let cap = scheduled_turn_context(1_048_576, 349_520, None);
        assert_eq!(cap, 349_520);
        assert!(crate::engine::scheduler::clamp_scheduled_output_tokens(33_611, 1024, cap).is_ok());
        assert!(
            crate::engine::scheduler::clamp_scheduled_output_tokens(33_611, 1024, 32_768).is_err()
        );
    }

    #[test]
    fn admit_never_exceeds_trained() {
        assert_eq!(scheduled_turn_context(40_960, 262_144, None), 40_960);
    }

    #[test]
    fn explicit_env_cap_still_clips() {
        assert_eq!(
            scheduled_turn_context(1_048_576, 349_520, Some(32_768)),
            32_768
        );
    }

    #[test]
    fn zero_or_missing_override_is_ignored() {
        // Some(0) is treated as unset; None means no extra clip beyond min(trained, pool).
        assert_eq!(scheduled_turn_context(64, 64, Some(0)), 64);
        assert_eq!(scheduled_turn_context(1_048_576, 349_520, None), 349_520);
    }

    #[test]
    fn incident_prompt_fits_nemotron_pool() {
        let trained = 1_048_576;
        let pool = 349_520;
        let cap = scheduled_turn_context(trained, pool, None);
        // Output still clamped to remaining window; prompt accepted against the pool.
        assert!(crate::engine::scheduler::clamp_scheduled_output_tokens(33_611, 1024, cap).is_ok());
    }

    #[test]
    fn recurrent_bytes_leave_kv_headroom() {
        // 16-token blocks, 1024 bytes/block: 48 MiB recurrent → 49152 blocks → 786432 tokens.
        let pool = 1_048_576;
        let usable = pool_tokens_after_recurrent(pool, 16, 1024, 48 * 1024 * 1024);
        assert_eq!(usable, 1_048_576 - 786_432);
        assert!(usable < pool);
    }

    #[test]
    fn filling_the_pool_with_nonzero_recurrent_cannot_fit() {
        // Equality used to skip `reservation_blocks > total_blocks` and then
        // defer forever once recurrent bytes were charged.
        assert!(reservation_fits_empty_pool(100, 100, 4096, 0));
        assert!(!reservation_fits_empty_pool(100, 100, 4096, 1));
        assert!(!reservation_fits_empty_pool(101, 100, 4096, 0));
    }

    #[test]
    fn one_recurrent_row_fits_usable_pool_two_do_not() {
        // KV sized to `pool_tokens_after_recurrent` (one row). Empty-pool + that
        // row fits; parked row + candidate row does not.
        let pool_tokens = 1_048_576;
        let block_size = 16;
        let bytes_per_block = 1024;
        let rec_bytes = 48 * 1024 * 1024;
        let usable =
            pool_tokens_after_recurrent(pool_tokens, block_size, bytes_per_block, rec_bytes);
        let total_blocks = pool_tokens / block_size;
        let reservation_blocks = usable.div_ceil(block_size);

        assert!(reservation_fits_empty_pool(
            reservation_blocks,
            total_blocks,
            bytes_per_block,
            rec_bytes,
        ));

        let telemetry = crate::engine::scheduler::MemoryTelemetry {
            capacity_bytes: u64::from(total_blocks).saturating_mul(bytes_per_block),
            free_bytes: u64::from(total_blocks).saturating_mul(bytes_per_block),
            reclaimable_bytes: 0,
        };
        let two_rows = crate::engine::scheduler::memory_admission_decision(
            telemetry,
            0,
            reservation_blocks,
            bytes_per_block,
            rec_bytes,
            rec_bytes,
            false,
            0.0,
        );
        assert!(
            !two_rows.admitted,
            "parked + candidate recurrent rows must overflow usable-pool KV"
        );

        let one_row = crate::engine::scheduler::memory_admission_decision(
            telemetry,
            0,
            reservation_blocks,
            bytes_per_block,
            0,
            rec_bytes,
            false,
            0.0,
        );
        assert!(one_row.admitted);
    }
}
