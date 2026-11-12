//! Engine-owned continuous-batching lifecycle for hybrid paged-KV plus
//! recurrent-state model families.
//!
//! Plain text autoregressive turns share one scheduler step. Native MTP,
//! multimodal turns, raw generation, calibration, persistence, and training
//! remain ordered barriers and execute through the existing command handler.

mod speculative;

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use napi::bindgen_prelude::{Error, Result};

use crate::array::MxArray;
use crate::decode_profiler::DecodeProfiler;
use crate::engine;
use crate::engine::backend::{
    ChunkSink, PagedBackend, PagedPrefix, ResetScope, StreamEmitter, ThinkingSetup, TurnOutput,
    TurnTokenObserver,
};
use crate::engine::cmd::{ChatCmd, handle_chat_cmd};
use crate::engine::governor::allocate::{
    Admission, Denial, KvAdmission, Refusal, StateAdmission, UnitSources, Verdict, allocate,
    capacity_exhausted, judge, page_shortfall,
};
use crate::engine::governor::governor::{
    ClaimHandle, ClaimRef, Failure, Governor, MemoryAccounts, Reservation, ResourceWaitStatus,
    Scope, next_request_id,
};
use crate::engine::governor::ledger::{
    CertainReuse, ClaimClass, ClaimRank, ClaimSpec, Need, NeedTiming, ReachClass, Stage,
    Taken as LedgerTaken, UnitKey as LedgerKey, UnitRuns, Victim, VictimQuery, VictimSource,
    WantedChains,
};
use crate::engine::governor::metrics::{GapClock, PassWork, Timeline, TtftClock, WholeTurnScope};
use crate::engine::governor::wait::{
    LaneWork, ResourceWait, WaitReason, Waiter, defer_resource_retry, defer_wait, note_refusal,
    resource_timeout, retry_ready, sync_claims, tick,
};
use crate::engine::model_command::{FamilyCommand, ModelCommand, handle_model_command};
use crate::engine::scheduler::{
    BlockTelemetry, JudgePolicy, MeasuredConfig, PREFILL_CHUNK_ROWS, PreemptedTurn, PreemptionMode,
    PreemptionReplay, Prepared, RetryClock, RowGrant, RowStepResult, Scheduler, SchedulerAction,
    SchedulerError, StepExecutor, StepKind, StepPlan, StepResult, TurnState, TurnStatus,
    install_preemption_replay, is_paged_allocation_blocked,
};
use crate::engine::types::{ChatConfig, ChatResult, ChatStreamChunk};
use crate::model_thread::{LoopControl, ResponseTx, StreamTx};
use crate::sampling::{check_repetition_cutoff, sample};
use crate::stream::{DeviceType, Stream, StreamContext, WiredResidency};
use crate::tokenizer::Qwen3Tokenizer;
use crate::transformer::paged_kv_cache_adapter::BlockRefusal;
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

/// Per-sequence state a family keeps outside the paged pool (recurrent rows,
/// draft snapshots): what it may hold at once and what it holds now.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ScheduledStateCapacity {
    pub budget: u64,
    pub held: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ScheduledStateFit {
    Fits,
    AfterReclaim,
    Never,
}

/// `outstanding_bytes` is what queued and running turns reserved but their
/// sequences do not hold yet; it counts like held state. A candidate above
/// the whole budget must error, never defer: no reclaim can make room for it.
pub(crate) fn scheduled_state_fit(
    capacity: Option<ScheduledStateCapacity>,
    outstanding_bytes: u64,
    candidate_bytes: u64,
) -> ScheduledStateFit {
    let Some(capacity) = capacity else {
        return ScheduledStateFit::Fits;
    };
    if candidate_bytes > capacity.budget {
        ScheduledStateFit::Never
    } else if capacity
        .held
        .saturating_add(outstanding_bytes)
        .saturating_add(candidate_bytes)
        > capacity.budget
    {
        ScheduledStateFit::AfterReclaim
    } else {
        ScheduledStateFit::Fits
    }
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

/// Host scheduling metadata for one contiguous verifier slice. Proposed tokens
/// are not visible in committed history until the matching batch commit.
pub(crate) struct ScheduledVerifyRow {
    pub seq_id: SeqId,
    pub first_position: u32,
    pub tokens: Vec<u32>,
    pub speculative: bool,
}

pub(crate) struct ScheduledVerifyCommit {
    pub seq_id: SeqId,
    pub keep: usize,
}

/// Family capabilities required by the shared hybrid scheduler lifecycle.
///
/// This mirrors vLLM's scheduler/cache-manager boundary: the engine owns
/// request state and policy, while a model runner supplies cache access and
/// executes the planned prefill/decode work.
pub(crate) trait HybridSchedulerBackend: PagedBackend + Sized {
    type FamilyCommand: FamilyCommand<Self>;
    type RestoreTicket;
    type OwnerState: Default;
    type StepExecutor<'a>: StepExecutor<ScheduledTurn<Self::PrefixState>, Error = Error>
    where
        Self: 'a;

    const SCHEDULER_NAME: &'static str;
    const ENABLED_BY_DEFAULT: bool = true;
    const CANCEL_PRECEDES_EOS: bool = false;
    const STREAM_EOS_TOKEN: bool = false;

    /// The model's wired residency, held across every drive iteration.
    fn wired_residency(&self) -> Option<Arc<WiredResidency>> {
        None
    }
    /// `seq_id` was admitted; called before the turn's cancel flag and
    /// prefill slices are read, so a backend that changes its
    /// residency per turn does it here. Every opened turn is closed once, on
    /// whichever path finishes or rejects it.
    fn scheduled_turn_opened(&mut self, _seq_id: SeqId) {}
    fn scheduled_turn_closed(&mut self, _seq_id: SeqId) {}
    /// The family prefills this prompt (`prompt_tokens`, `cached_prefix` of
    /// them cached) as one group of windows: its prefill is one slice and
    /// one step, the step's only prefill row, outside the token budget.
    /// Read with the slice grid at admission and at every re-pin.
    fn scheduler_whole_prompt_prefill(&self, _prompt_tokens: u32, _cached_prefix: u32) -> bool {
        false
    }

    fn paged_adapter(&self) -> Option<&PagedKVCacheAdapter>;
    fn paged_adapter_mut(&mut self) -> Option<&mut PagedKVCacheAdapter>;
    /// The family's memory is the governor's ledger: its rows claim what
    /// their steps allocate and its blocked rows are judged as Splash's
    /// (`JudgePolicy::Ledger`).
    fn reclaims_in_order(&self) -> bool {
        false
    }
    /// The family's prefill rows are cut beside a decoding lane (Splash
    /// `Scheduler::planPrefill`); `None` plans the fixed step budget.
    fn measured_prefill_cut(&self) -> Option<MeasuredConfig> {
        None
    }
    /// The model's id with the memory governor; 0 when not registered.
    fn memory_model(&self) -> u64 {
        0
    }
    fn sync_state_marks(&mut self) {}
    /// The memory governor the model registered with.
    fn memory_governor(&self) -> Option<&'static Governor> {
        None
    }
    /// The model's units on the ledger's order, in its take order (Splash's
    /// cache walk), none that `kept` restores through; the scheduler adds
    /// the idle owners.
    fn victim_units(&self, _query: &VictimQuery, _kept: &[u32]) -> UnitRuns {
        UnitRuns::default()
    }
    fn take_victim(
        &mut self,
        _query: &VictimQuery,
        _victim: &Victim,
        _kept: &[u32],
    ) -> LedgerTaken {
        LedgerTaken::default()
    }
    /// A GPU command exhausted Metal memory: every cache its writes may
    /// have reached goes (Splash restarts the engine).
    /// The model's accounts on the governor's books.
    fn memory_accounts(&self) -> Option<MemoryAccounts> {
        None
    }
    /// The model's DFlash2 states for the governor's status.
    fn dflash2_state_status(
        &self,
    ) -> Option<crate::engine::governor::governor::DFlash2StateStatus> {
        None
    }
    fn reset_after_gpu_oom(&mut self) -> Result<()> {
        self.reset_caches(ResetScope::Command)
    }
    /// Each planned row's claim of what its step allocates (Splash
    /// `Engine::prepare`); every row is granted unclaimed by default.
    fn claim_step_rows(state: &mut HybridSchedulerState<Self>, plan: &StepPlan) -> StepClaims {
        let _ = state;
        StepClaims::granted(plan)
    }
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
                reclaimable_blocks: blocks.evictable_blocks,
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
    /// The turn books its prompt plus one block and grows KV per step; a
    /// blocked step reclaims, then preempts, and a lone blocked row fails.
    /// Otherwise the whole turn is booked at admission.
    fn scheduler_books_on_demand(&self) -> bool {
        false
    }
    /// The window a scheduled turn clamps on when the family's memory plan
    /// sets it. A plan that prices the draft state into its lane answers one
    /// window for both kinds; otherwise `speculative` turns also pay the
    /// draft state they hold. `None` keeps the pool's own window and
    /// `MLX_PAGED_PER_SEQ_CTX`.
    fn planned_turn_window(&self, _speculative: bool) -> Option<u32> {
        None
    }
    /// The scheduled output budget is the rest of the window: a prompt must
    /// leave room for one new token and the budget is `window - prompt`. The
    /// family publishes the same rule to the host
    /// (`Qwen3_5ContextLimits::rest_of_window_output_budget`). Otherwise the
    /// budget is `window - prompt + 1`.
    fn scheduler_output_budget_is_rest_of_window(&self) -> bool {
        false
    }
    fn reset_owner_on_session_start(&self) -> bool {
        false
    }
    /// The exclusive lane keeps the ownerless session in the scheduler's owner
    /// tables under the empty owner id, so the scheduled lane can serve that
    /// same session. Without this, an ownerless turn must stay exclusive: its
    /// session lives outside the tables the scheduled lane reads.
    fn ownerless_session_in_owner_tables(&self) -> bool {
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
    fn supports_scheduled_speculation(&self) -> bool {
        false
    }
    fn supports_adaptive_scheduled_speculation(&self) -> bool {
        false
    }
    fn scheduled_verification_budget(
        &mut self,
    ) -> Option<&mut engine::verification_budget::ScheduledVerificationBudget> {
        None
    }
    fn complete_scheduled_draft_state(&self, _seq_ids: &[SeqId]) -> Result<()> {
        Ok(())
    }
    fn scheduled_draft_state_bytes(&self, _total_tokens: u32) -> u64 {
        0
    }
    /// Draft state a running speculative turn holds at `total_tokens`;
    /// [`Self::scheduled_draft_state_bytes`] also prices what only its
    /// prefill holds.
    fn scheduled_draft_state_held_bytes(&self, total_tokens: u32) -> u64 {
        self.scheduled_draft_state_bytes(total_tokens)
    }
    /// A family whose recurrent rows and draft snapshots live outside the
    /// paged pool budgets them here and charges nothing to the pool through
    /// `recurrent_state_bytes` / `scheduled_draft_state_bytes`.
    fn scheduled_state_capacity(&self) -> Option<ScheduledStateCapacity> {
        None
    }
    /// Give back memory the state budget is charged for (free pool extents)
    /// toward `shortfall` bytes; true when anything was freed.
    fn reclaim_for_scheduled_state(&mut self, _shortfall: u64) -> bool {
        false
    }
    /// Off-pool state this turn adds beside what the family already holds.
    fn scheduled_state_candidate_bytes(
        &self,
        _seq_id: SeqId,
        _total_tokens: u32,
        _speculative: bool,
    ) -> u64 {
        0
    }
    /// Off-pool state `seq_id` holds now; a live turn's reservation is charged
    /// only beyond this.
    fn scheduled_state_held_bytes(&self, _seq_id: SeqId) -> u64 {
        0
    }
    /// Return false when this prefix cannot seed complete draft state. The
    /// request keeps its paged target cache and continues as scheduled AR.
    fn begin_scheduled_speculation(&mut self, _seq_id: SeqId, _position: u32) -> Result<bool> {
        Err(Error::from_reason("scheduled speculation is unavailable"))
    }
    fn reserve_scheduled_speculation(&mut self, _seq_id: SeqId, _queries: usize) -> Result<bool> {
        Ok(false)
    }
    fn propose_scheduled(
        &mut self,
        _seq_id: SeqId,
        _anchor: u32,
        _max_drafts: usize,
        _params: &engine::params::ChatParams,
        _rng: &mut dyn rand::Rng,
        _confidence: bool,
    ) -> Result<engine::backend::DsparkProposal> {
        Err(Error::from_reason("scheduled proposals are unavailable"))
    }
    fn run_scheduled_verify(&mut self, _rows: &[ScheduledVerifyRow]) -> Result<MxArray> {
        Err(Error::from_reason("scheduled verification is unavailable"))
    }
    fn commit_scheduled_verify(
        &mut self,
        _rows: &[ScheduledVerifyCommit],
    ) -> Result<Vec<Result<()>>> {
        Err(Error::from_reason(
            "scheduled verification commit is unavailable",
        ))
    }
    fn release_scheduled_speculation(&mut self, _seq_id: SeqId) {}

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
    /// Called after the sampled GPU values have materialized, before host
    /// result handling. Forced tokens and failed samples do not qualify.
    fn observe_scheduled_decode_batch(&mut self, _completed: bool) {}

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
    /// Default chat barrier behavior. Override only when an exclusive turn
    /// must install or reconcile family-specific owner state.
    fn execute_chat_barrier(
        &mut self,
        command: ChatCmd,
        _owners: SchedulerOwnerContext<'_, Self::OwnerState>,
    ) {
        handle_chat_cmd(self, command);
    }

    fn execute_barrier(
        &mut self,
        command: ModelCommand<Self::FamilyCommand>,
        owners: SchedulerOwnerContext<'_, Self::OwnerState>,
    ) {
        match command {
            ModelCommand::Chat(chat) => self.execute_chat_barrier(*chat, owners),
            other => handle_model_command(self, other),
        }
    }
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

/// Pinned prefill slice ends for one prompt: the family's slice grid
/// (`scheduler_prefill_slice_tokens`) walked from `start` (the cached prefix)
/// to `target`, plus the family's `extra_prefill_breaks` strictly inside, in
/// ascending order without repeats. Each slice's attention and K/V round per
/// slice, so a whole-turn lane that must give the scheduled lane's bits walks
/// the same ends.
pub(crate) fn prefill_slice_ends(
    start: u32,
    target: u32,
    slice_tokens: u32,
    extra_breaks: impl IntoIterator<Item = u32>,
) -> Vec<u32> {
    let mut ends = Vec::new();
    let mut boundary = start;
    while boundary < target {
        boundary = boundary.saturating_add(slice_tokens.max(1)).min(target);
        ends.push(boundary);
    }
    ends.extend(
        extra_breaks
            .into_iter()
            .filter(|&boundary| boundary > start && boundary < target),
    );
    ends.sort_unstable();
    ends.dedup();
    ends
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

/// The context a scheduled turn clamps on: the planned window when the
/// family's memory plan sets one, else the pool window under
/// `override_cap`.
pub(crate) fn scheduled_window(
    trained_context: u32,
    planned: Option<u32>,
    pool_tokens: u32,
    override_cap: Option<u32>,
) -> u32 {
    match planned {
        Some(window) => scheduled_turn_context(trained_context, window, None),
        None => scheduled_turn_context(trained_context, pool_tokens, override_cap),
    }
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
    if live_held_prefix(adapter, seq_id, prompt, reuse).is_none() {
        return 0;
    }
    adapter
        .and_then(|adapter| adapter.block_table_for(seq_id))
        .map(|table| table.num_blocks() as u32)
        .unwrap_or(0)
}

/// The tokens of `prompt` a live continuation of `seq_id` keeps: the held
/// request's, when the prompt continues it.
fn live_held_prefix(
    adapter: Option<&PagedKVCacheAdapter>,
    seq_id: SeqId,
    prompt: &[u32],
    reuse: bool,
) -> Option<usize> {
    if !reuse {
        return None;
    }
    let adapter = adapter?;
    if !adapter.is_live_for_continue_for(seq_id) {
        return None;
    }
    let held = adapter.request_tokens_for(seq_id)?;
    prompt.starts_with(held).then_some(held.len())
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

/// Blocks a scheduled turn books at admission, and the tokens they cover.
/// On demand: the prompt plus one growth block, never more than the whole
/// turn. Otherwise the whole turn.
pub(crate) fn scheduled_booking(
    on_demand: bool,
    prompt_tokens: u32,
    max_new_tokens: u32,
    block_size: u32,
) -> (u32, u32) {
    let block_size = block_size.max(1);
    let requested_tokens =
        engine::scheduler::scheduled_materialized_tokens(prompt_tokens, max_new_tokens);
    let full_blocks = requested_tokens.div_ceil(block_size);
    if !on_demand {
        return (full_blocks, requested_tokens);
    }
    let blocks = prompt_tokens
        .div_ceil(block_size)
        .saturating_add(u32::from(max_new_tokens != 0))
        .min(full_blocks);
    (
        blocks,
        requested_tokens.min(blocks.saturating_mul(block_size)),
    )
}

/// Blocks a preempted on-demand turn that blocked books to resume, never
/// less than it already booked. A prefill replay books its prompt: its first
/// token is sampled from the prefill logits and writes no row. A decode
/// replay books its history plus the step it blocked at (the pending token
/// and `draft_rows`), never past the whole turn; the running step caps its
/// drafts at the output left the same way.
pub(crate) fn resume_booking_blocks(
    booked_blocks: u32,
    history_tokens: u32,
    draft_rows: u32,
    requested_tokens: u32,
    block_size: u32,
    prefill_replay: bool,
) -> u32 {
    let rows = if prefill_replay {
        history_tokens
    } else {
        let next_step = history_tokens.saturating_add(1);
        next_step
            .saturating_add(draft_rows)
            .min(requested_tokens.max(next_step))
    };
    booked_blocks.max(rows.div_ceil(block_size.max(1)))
}

/// State a running on-demand speculative turn holds at `tokens` past the
/// charge it was admitted or resumed with; 0 for any other turn.
fn uncharged_draft_state_bytes<B: HybridSchedulerBackend>(
    inner: &B,
    turn: &TurnState<ScheduledTurn<B::PrefixState>>,
    tokens: u32,
) -> u64 {
    if !inner.scheduler_books_on_demand()
        || !turn.recurrent_state_reservation_active
        || turn.payload.scheduled_speculation.is_none()
    {
        return 0;
    }
    inner
        .recurrent_state_bytes()
        .saturating_add(inner.scheduled_draft_state_held_bytes(tokens))
        .saturating_sub(turn.recurrent_state_bytes)
}

/// Off-pool state a running turn holds this step at `tokens`, plus its
/// booked blocks the pool has not handed out yet. An on-demand speculative
/// turn is priced at what it holds once decoding (its prefill hidden rows
/// are freed) and never below its charge while prefilling; any other turn
/// at its charge.
fn step_state_bytes<B: HybridSchedulerBackend>(
    inner: &B,
    turn: &TurnState<ScheduledTurn<B::PrefixState>>,
    tokens: u32,
    bytes_per_block: u64,
) -> u64 {
    let state = if !turn.recurrent_state_reservation_active {
        0
    } else if inner.scheduler_books_on_demand() && turn.payload.scheduled_speculation.is_some() {
        let held = inner
            .recurrent_state_bytes()
            .saturating_add(inner.scheduled_draft_state_held_bytes(tokens));
        if turn.num_computed_tokens >= turn.prompt_tokens {
            held
        } else {
            held.max(turn.recurrent_state_bytes)
        }
    } else {
        turn.recurrent_state_bytes
    };
    let booked = if turn.block_reservation_active
        && turn.block_reservation_total != 0
        && turn.block_size != 0
    {
        turn.block_reservation_total
            .saturating_sub(inner.scheduler_materialized_blocks(turn.seq_id))
    } else {
        0
    };
    state.saturating_add(u64::from(booked).saturating_mul(bytes_per_block))
}

/// Blocks a pool grow must add before `decision` admits, or `None` when no
/// grow below the max pool can. Live turns' unallocated bookings and the
/// watermark are in `decision`; the watermark grows with the pool by
/// `watermark_fraction` of each added block.
fn grow_needed_blocks_for(
    snapshot: &SchedulerCacheSnapshot,
    decision: &engine::scheduler::MemoryAdmission,
    watermark_fraction: f64,
) -> Option<u32> {
    if snapshot.blocks.total_blocks >= snapshot.max_total_blocks {
        return None;
    }
    let bytes_per_block = snapshot.bytes_per_block.max(1);
    let available = u64::from(snapshot.blocks.free_blocks)
        .saturating_add(u64::from(snapshot.blocks.reclaimable_blocks))
        .saturating_mul(bytes_per_block);
    let shortfall = decision
        .reserved_block_bytes
        .saturating_add(decision.reserved_state_bytes)
        .saturating_add(decision.watermark_bytes)
        .saturating_sub(available);
    if shortfall == 0 {
        return None;
    }
    let fraction = if decision.watermark_bytes > 0 {
        watermark_fraction.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let per_block = bytes_per_block as f64 * (1.0 - fraction);
    if per_block <= 0.0 {
        return None;
    }
    let blocks = (shortfall as f64 / per_block).ceil();
    let room = snapshot
        .max_total_blocks
        .saturating_sub(snapshot.blocks.total_blocks);
    (blocks <= f64::from(room)).then_some(blocks as u32)
}

fn is_cancelled<P>(turn: &TurnState<P>) -> bool {
    turn.cancelled
        .as_ref()
        .is_some_and(|flag| flag.load(Ordering::Relaxed))
}

fn stop_requested<P>(turn: &TurnState<ScheduledTurn<P>>) -> bool {
    match &turn.payload.response {
        ScheduledReply::Stream(stream) => stream.stop_requested(),
        ScheduledReply::Sync(_) => false,
    }
}

/// A stop reached a turn that generated tokens: it ends as a completion
/// (Splash `Engine::stop` → `settle` → `finish(Stop)`), never as a refusal.
fn stopped_with_tokens<P>(turn: &TurnState<ScheduledTurn<P>>) -> bool {
    stop_requested(turn) && !turn.payload.generated_tokens.is_empty()
}

/// A running turn's stop counts the token its last prefill slice sampled
/// (Splash `Engine::stop` reads `generatedTokens`): its decode row commits
/// it and ends `Stop`.
fn stopped_after_sampling<P>(turn: &TurnState<ScheduledTurn<P>>) -> bool {
    stop_requested(turn) && turn.token_history.len() > turn.prompt_tokens as usize
}

/// A failed prefill slice of a stopped turn ends `Stop`, unless the GPU ran
/// out of memory: that failure must reach the restart (Splash restarts on
/// `metal_out_of_memory` however the request ended).
fn failed_slice_vacates<P>(turn: &mut TurnState<ScheduledTurn<P>>, error: &Error) -> bool {
    !crate::engine::governor::is_gpu_out_of_memory(&error.reason) && stop_vacates(turn)
}

/// A stopped turn that generated tokens and holds no pages ends as a
/// completion; marks it and returns true.
pub(crate) fn stop_vacates<P>(turn: &mut TurnState<ScheduledTurn<P>>) -> bool {
    if !stopped_with_tokens(turn) {
        return false;
    }
    turn.payload.vacated_stop = true;
    turn.payload.finish_reason = "cancelled".to_string();
    true
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
    pub(crate) scheduled_speculation: Option<rand::rngs::StdRng>,
    pub(crate) pending_token_emitted: bool,
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
    /// A stop reached the turn while it held no pages (preempted, waiting, or
    /// replaying after preemption): it ends as a completion on what it
    /// generated (Splash `Engine::stop` → `settle` → `finish(Stop)`).
    pub(crate) vacated_stop: bool,
    pub(crate) allocation_failed: bool,
    pub(crate) preemption_replay: Option<PreemptionReplay>,
    /// Token bound the off-pool state reservation was sized for. An
    /// on-demand block reservation covers fewer tokens, so a resume must
    /// not derive the bound from it.
    pub(crate) requested_tokens: u32,
    /// The request's memory claim (Splash `Request::claim`), from its
    /// admission to its end.
    pub(crate) claim: Option<ClaimHandle<'static>>,
    /// The GDN cell its admission reserved, held until its first step ran
    /// (the cell's arrays are allocated lazily).
    pub(crate) cell: Option<Reservation<'static>>,
    pub(crate) clock: TurnClock,
}

/// A scheduled request's clocks: what held it before its first tokens
/// (Splash `Request::held`) and its token events, for decode gaps.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct TurnClock {
    ttft: TtftClock,
    gap: GapClock,
    token_events: u64,
    seen_events: u64,
}

impl TurnClock {
    /// Tokens went out to the client.
    pub(crate) fn token_event(&mut self) {
        self.token_events += 1;
    }
}

/// When a command reached the model thread (Splash counts a request's TTFT
/// from its submission) and the claimless timeline then.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Arrival {
    at: Instant,
    mark: Timeline,
}

/// A command off the channel, with its arrival.
pub(crate) struct Queued<C> {
    command: ModelCommand<C>,
    arrival: Arrival,
}

impl<C> From<ModelCommand<C>> for Queued<C> {
    fn from(command: ModelCommand<C>) -> Self {
        Self {
            command,
            arrival: Arrival {
                at: Instant::now(),
                mark: Timeline::default(),
            },
        }
    }
}

impl<C> std::ops::Deref for Queued<C> {
    type Target = ModelCommand<C>;

    fn deref(&self) -> &ModelCommand<C> {
        &self.command
    }
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
    reservation_state_bytes: u64,
    requested_tokens: u32,
    /// The request's claim, from its first admission attempt.
    claim: Option<ClaimHandle<'static>>,
    /// Splash `Request::resourceWait` before the request has a lane.
    resource_wait: ResourceWait,
    enqueued_at: Instant,
    ttft: TtftClock,
    /// Splash `Phase::Queued`: the scheduler did not want it in its last
    /// admission attempt.
    held: bool,
    /// Splash `Request::order`, taken when it is first held.
    order: Option<u64>,
    /// Splash `Request::overtaken` while it is held.
    overtaken: u32,
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

    /// A row's work is its request's (Splash `Engine::stateClaim`): what it
    /// publishes claims with its rank and rights.
    fn request_scope(turn: &TurnState<ScheduledTurn<B::PrefixState>>) -> Option<Scope> {
        turn.payload
            .claim
            .as_ref()
            .map(|claim| Scope::enter(claim.claim()))
    }

    fn blocked(row: &engine::scheduler::StepRow) -> RowStepResult {
        RowStepResult {
            seq_id: row.seq_id,
            num_computed_tokens: 0,
            generated_token: None,
            speculative: None,
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
        Self::end_row(turn, row)
    }

    fn end_row(
        turn: &mut TurnState<ScheduledTurn<B::PrefixState>>,
        row: &engine::scheduler::StepRow,
    ) -> RowStepResult {
        turn.payload.profiler.snapshot_memory_after();
        turn.payload.profiler.report();
        RowStepResult {
            seq_id: row.seq_id,
            num_computed_tokens: row.num_tokens,
            generated_token: None,
            speculative: None,
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
            if stop_vacates(turn) {
                return Ok(Self::end_row(turn, row));
            }
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
        let cached_prefix = turn.payload.preemption_replay.as_ref().map_or(
            turn.payload.prefix.effective_cached_prefix_len(),
            |replay| replay.cached_prefix as usize,
        );
        let first_chunk = start == cached_prefix;
        let progress = turn
            .payload
            .response
            .sink()
            .and_then(|sink| sink.progress());
        if first_chunk && let Some(progress) = progress {
            progress.begin_prefill(
                source_len.saturating_sub(cached_prefix) as u32,
                cached_prefix as u32,
            );
        }
        if first_chunk && turn.payload.scheduled_speculation.is_some() {
            match self
                .inner
                .begin_scheduled_speculation(row.seq_id, start as u32)
            {
                Ok(true) => {}
                Ok(false) => {
                    self.inner.release_scheduled_speculation(row.seq_id);
                    turn.payload.scheduled_speculation = None;
                    turn.decode_draft_allowance = 0;
                    turn.recurrent_state_bytes = self.inner.recurrent_state_bytes();
                    turn.scheduled_state_bytes = 0;
                }
                Err(error) => return Ok(Self::fail(turn, row, error)),
            }
        }
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
                // A stop that lands inside a replay slice cancels the slice
                // from the live flag; the turn still ends `Stop` on what it
                // generated (Splash `Engine::stop` → `settle`).
                if failed_slice_vacates(turn, &error) {
                    return Ok(Self::end_row(turn, row));
                }
                return Ok(Self::fail(turn, row, error));
            }
        };
        turn.payload.profiler.end_prefill();
        if end < source_len {
            crate::array::synchronize_and_clear_cache();
            if let Some(progress) = turn
                .payload
                .response
                .sink()
                .and_then(|sink| sink.progress())
            {
                progress.prefill_advanced(end.saturating_sub(cached_prefix) as u32);
            }
            return Ok(RowStepResult {
                seq_id: row.seq_id,
                num_computed_tokens: row.num_tokens,
                generated_token: None,
                speculative: None,
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
                speculative: None,
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
        let config = turn.payload.params.sampling_config;
        let sampled = if let Some(rng) = turn.payload.scheduled_speculation.as_mut()
            && !crate::sampling::is_greedy_temperature(
                config.and_then(|config| config.temperature).unwrap_or(1.0),
            ) {
            crate::sampling::sampling_distribution(&logits, config)
                .and_then(|distribution| distribution.astype(crate::array::DType::Float32))
                .and_then(|distribution| {
                    crate::sampling::sample_dense_distribution_array(&distribution, rng)
                })
        } else {
            sample(&logits, config)
        };
        let mut sampled = match sampled {
            Ok(sampled) => sampled,
            Err(error) => return Ok(Self::fail(turn, row, error)),
        };
        if let Err(error) = turn
            .payload
            .reasoning_tracker
            .enforce_next_token(&mut sampled)
        {
            return Ok(Self::fail(turn, row, error));
        }
        sampled.eval();
        crate::array::synchronize_and_clear_cache();
        if let Some(progress) = turn
            .payload
            .response
            .sink()
            .and_then(|sink| sink.progress())
        {
            progress.first_token();
        }
        if turn.payload.params.max_new_tokens == 0 {
            turn.payload.clock.token_event();
            if turn.payload.params.report_performance {
                turn.payload.first_token_instant = Some(Instant::now());
            }
            turn.payload.profiler.snapshot_memory_after();
            turn.payload.profiler.report();
            return Ok(RowStepResult {
                seq_id: row.seq_id,
                num_computed_tokens: row.num_tokens,
                generated_token: None,
                speculative: None,
                finished: true,
                allocation_blocked: false,
                prefill_micros: Self::elapsed_micros(started),
            });
        }
        let token = match sampled.item_at_int32(0) {
            Ok(token) if token >= 0 => token as u32,
            Ok(_) => {
                return Ok(Self::fail(
                    turn,
                    row,
                    Error::from_reason("prefill sampling has no valid probability mass"),
                ));
            }
            Err(error) => return Ok(Self::fail(turn, row, error)),
        };
        Ok(RowStepResult {
            seq_id: row.seq_id,
            num_computed_tokens: row.num_tokens,
            generated_token: Some(token),
            speculative: None,
            finished: false,
            allocation_blocked: false,
            prefill_micros: Self::elapsed_micros(started),
        })
    }

    fn finish_decode_row(
        turn: &mut TurnState<ScheduledTurn<B::PrefixState>>,
        row: &PreparedDecodeRow,
        is_reasoning: bool,
    ) {
        turn.payload.clock.token_event();
        turn.payload.profiler.mark_first_token();
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
        // The prefill-sampled token is emitted only here, one decode forward
        // later; stamping TTFT at the sample hides that forward (seconds on a
        // cold request) inside decode time.
        if turn.payload.params.report_performance && turn.payload.first_token_instant.is_none() {
            turn.payload.first_token_instant = Some(Instant::now());
        }
    }

    fn prepare_sample_next(
        turn: &mut TurnState<ScheduledTurn<B::PrefixState>>,
        mut logits: MxArray,
    ) -> Result<engine::batch_sampling::PendingSample> {
        let sampled = if let Some(forced) = turn.payload.reasoning_tracker.take_forced_token() {
            MxArray::from_int32(&[forced as i32], &[1])?
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
        Ok(engine::batch_sampling::PendingSample {
            token: sampled,
            logits,
        })
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
        if running.iter().any(|turn| {
            turn.payload.scheduled_speculation.is_some() || turn.payload.pending_token_emitted
        }) {
            return self.execute_speculative_rows(plan, running);
        }
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
            if row.batch_index.is_some() {
                turn.payload.profiler.end();
            }
            // Count the current token before either sampling path chooses the
            // next one. Blocked forwards retry this token, so leave their
            // reasoning state untouched until the forward succeeds.
            if row.batch_index.is_none() || logits.is_ok() {
                turn.payload.last_is_reasoning =
                    turn.payload.reasoning_tracker.observe_token(row.token_id);
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
                    turn.payload.reasoning_tracker.ending(),
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let sampled_forward = greedy_wave.iter().all(|(_, forced)| !forced);
        let greedy_tokens = match &logits {
            Ok(Some(logits)) if engine::batch_sampling::can_batch_greedy_wave(greedy_wave) => {
                engine::batch_sampling::batch_greedy_tokens_or_fallback(logits)
            }
            _ => None,
        };
        let executed_greedy_epilogue_batch = greedy_tokens.as_ref().map_or(0, Vec::len);
        let mut sampling_eval_time = None;
        let mut next_tokens: Vec<_> = if let Some(tokens) = greedy_tokens {
            tokens.into_iter().map(|token| Some(Ok(token))).collect()
        } else if let Ok(Some(logits)) = &logits {
            let mut pending = Vec::with_capacity(batch_rows.len());
            for row in &work {
                let Some(index) = row.batch_index else {
                    continue;
                };
                debug_assert_eq!(index, pending.len());
                let turn = running
                    .iter_mut()
                    .find(|turn| turn.seq_id == row.seq_id)
                    .ok_or_else(|| {
                        Error::from_reason("decode sequence disappeared before sampling")
                    })?;
                let sample = logits
                    .slice_axis(0, index as i64, index as i64 + 1)
                    .and_then(|row| row.squeeze(Some(&[1])))
                    .and_then(|logits| Self::prepare_sample_next(turn, logits));
                pending.push(sample);
            }
            let started = Instant::now();
            let sampled = engine::batch_sampling::evaluate_pending_samples(pending);
            sampling_eval_time = Some(started.elapsed());
            sampled.into_iter().map(Some).collect()
        } else {
            Vec::new()
        };
        self.inner.observe_scheduled_decode_batch(
            sampled_forward
                && !next_tokens.is_empty()
                && next_tokens.iter().all(|token| matches!(token, Some(Ok(_)))),
        );
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
            let next = match (&logits, row.batch_index) {
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
                (Ok(Some(_)), Some(index)) => {
                    match next_tokens.get_mut(index).and_then(Option::take) {
                        Some(Ok(token)) => Some(token),
                        Some(Err(error)) => {
                            results.push((row.plan_index, Self::fail(turn, planned, error)));
                            continue;
                        }
                        None => {
                            return Err(Error::from_reason(format!(
                                "{} sampling omitted batch row {index}",
                                B::SCHEDULER_NAME,
                            )));
                        }
                    }
                }
                _ => None,
            };
            if row.batch_index.is_some()
                && let Some(elapsed) = sampling_eval_time
            {
                // Each request waited on the same wave. This is per-request
                // latency, not additive GPU time across the batch.
                turn.payload
                    .profiler
                    .record_duration("sample_eval", elapsed);
            }
            turn.payload.profiler.step();
            Self::finish_decode_row(turn, &row, turn.payload.last_is_reasoning);
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
                    speculative: None,
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
        let decode_started = Instant::now();
        let (decode_results, executed_decode_batch, executed_greedy_epilogue_batch) =
            self.execute_decode_rows(plan, running)?;
        let decode_wall_micros = if plan.rows.iter().any(|row| row.kind == StepKind::Decode) {
            Self::elapsed_micros(decode_started)
        } else {
            0
        };
        let mut prefill_started = None;
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
            let _request = Self::request_scope(turn);
            prefill_started.get_or_insert_with(Instant::now);
            rows[index] = Some(
                self.execute_prefill(planned, turn)
                    .unwrap_or_else(|error| Self::fail(turn, planned, error)),
            );
        }
        let prefill_wall_micros = prefill_started.map_or(0, Self::elapsed_micros);
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
            prefill_wall_micros,
            decode_wall_micros,
        })
    }
}

/// A step's claims: each plan row's grant, and the bytes admitted for the
/// lazy allocations its eval makes, held until the step has run.
pub(crate) struct StepClaims {
    pub(crate) grants: Vec<RowGrant>,
    reservations: Vec<Reservation<'static>>,
}

impl StepClaims {
    pub(crate) fn granted(plan: &StepPlan) -> Self {
        Self {
            grants: vec![None; plan.rows.len()],
            reservations: Vec::new(),
        }
    }
}

/// The scheduler's units (Splash's cache walk): the model's beside the
/// idle owners' kept tables and parked rows, in one stamp order, a state
/// first on a tie. The claimant's own owner is no unit.
pub(crate) struct SchedUnits<'a, B: HybridSchedulerBackend> {
    pub(crate) state: &'a mut HybridSchedulerState<B>,
    pub(crate) claimant: SeqId,
    /// The prompt the claim restores through (Splash `VictimQuery::kept`).
    pub(crate) kept: &'a [u32],
}

impl<B: HybridSchedulerBackend> SchedUnits<'_, B> {
    /// An owner's blocks count for a blocks need, its parked row for a
    /// bytes need; a need the host refused walks only what meets it in
    /// place (Splash `Cache::walkFor`). Without certain reuse a claim
    /// keeps the owners whose chains a waiting request restores.
    fn owner_units(&self, query: &VictimQuery) -> Vec<Victim> {
        let state = &*self.state;
        let host_refused = query.need.host_refused;
        let pages_only = host_refused && query.need.pages > 0;
        let adapter = state.inner.paged_adapter();
        let mut owners = state.idle_owners(self.claimant);
        if query.rights.certain_reuse != CertainReuse::All {
            let wanted = state
                .memory_governor()
                .wanted_chains(state.inner.memory_model());
            owners.retain(|&seq_id| {
                !state
                    .owner_sequences
                    .iter()
                    .find(|(_, owner_seq)| **owner_seq == seq_id)
                    .and_then(|(owner_id, _)| state.owner_states.get(owner_id))
                    .is_some_and(|owner| wanted.keeps(B::owner_tokens(owner)))
            });
        }
        let freed = if query.need.pages > 0 {
            adapter
                .map(|adapter| adapter.releasable_blocks_of(&owners))
                .unwrap_or_default()
        } else {
            HashSet::new()
        };
        owners
            .into_iter()
            .map(|seq_id| {
                let pages = adapter
                    .and_then(|adapter| adapter.block_table_for(seq_id))
                    .map_or(0, |table| {
                        table
                            .blocks()
                            .iter()
                            .filter(|block| freed.contains(&block.block_id))
                            .count() as u32
                    });
                let bytes = if pages_only || !state.inner.has_scheduled_recurrent(seq_id) {
                    0
                } else {
                    state.inner.recurrent_state_bytes()
                };
                let stamp = state.owner_stamp(seq_id);
                Victim {
                    key: LedgerKey::Owner(seq_id),
                    stamp,
                    bytes,
                    pages,
                    stage: Stage::FreeNow,
                }
            })
            .filter(|unit| unit.bytes > 0 || unit.pages > 0)
            .collect()
    }
}

impl<B: HybridSchedulerBackend> VictimSource for SchedUnits<'_, B> {
    /// The model's checkpoints go first (Splash `Phase::Checkpoints`),
    /// ahead of an older idle owner.
    fn units(&self, query: &VictimQuery) -> Vec<Victim> {
        let UnitRuns { mut ahead, merged } = self.state.inner.victim_units(query, self.kept);
        let mut units: Vec<(Victim, bool)> = merged
            .into_iter()
            .map(|unit| (unit, false))
            .chain(self.owner_units(query).into_iter().map(|unit| (unit, true)))
            .collect();
        units.sort_by_key(|(unit, owner)| (unit.stage, unit.stamp, *owner));
        ahead.extend(units.into_iter().map(|(unit, _)| unit));
        ahead
    }

    /// An owner's take credits the measured drop in MLX's active bytes and
    /// the blocks it made takeable.
    fn take(&mut self, query: &VictimQuery, victim: &Victim) -> LedgerTaken {
        let LedgerKey::Owner(seq_id) = victim.key else {
            return self.state.inner.take_victim(query, victim, self.kept);
        };
        let takeable = |state: &HybridSchedulerState<B>| {
            state
                .inner
                .paged_adapter()
                .map_or(0, PagedKVCacheAdapter::takeable_blocks)
        };
        let active = crate::array::memory::active_memory_bytes();
        let blocks = takeable(self.state);
        if !self.state.release_owner(seq_id) {
            return LedgerTaken::default();
        }
        let now_bytes = active.saturating_sub(crate::array::memory::active_memory_bytes());
        let pages = takeable(self.state).saturating_sub(blocks);
        LedgerTaken {
            outcome: if now_bytes > 0 || pages > 0 {
                crate::engine::governor::ledger::FreeOutcome::Met
            } else {
                crate::engine::governor::ledger::FreeOutcome::Short
            },
            now_bytes,
            later_bytes: 0,
            pages,
        }
    }
}

impl<B: HybridSchedulerBackend> UnitSources for SchedUnits<'_, B> {
    fn sources(&mut self) -> Vec<&mut dyn VictimSource> {
        vec![self]
    }
}

/// A row's step claim (Splash `growLane`): its KV to the work end, then
/// its draft state's growth, a bytes claim.
struct RowAdmission {
    kv: KvAdmission<u32>,
    state: Option<StateAdmission<'static>>,
}

impl Admission for RowAdmission {
    fn granted(&self) -> bool {
        self.kv.granted() && self.state.as_ref().is_none_or(Admission::granted)
    }

    fn failure(&self) -> Option<Failure> {
        if self.kv.granted() {
            self.state.as_ref().and_then(Admission::failure)
        } else {
            self.kv.failure()
        }
    }

    fn in_place(&self, held: u32) -> Need {
        if self.kv.granted() {
            Need::default()
        } else {
            self.kv.in_place(held)
        }
    }

    fn memory_denied(&self) -> bool {
        if self.kv.granted() {
            self.state.as_ref().is_some_and(Admission::memory_denied)
        } else {
            self.kv.memory_denied()
        }
    }

    fn blocks(&self) -> Option<BlockRefusal> {
        if self.kv.granted() {
            None
        } else {
            self.kv.blocks()
        }
    }
}

/// Splash `Cache::ensureTokens`: `seq_id`'s KV up to `end` tokens, capped
/// at the pool's capacity, as an allocation of `claim`.
fn ensure_tokens(
    adapter: Option<&mut PagedKVCacheAdapter>,
    seq_id: SeqId,
    end: u32,
    claim: ClaimRef<'_>,
) -> KvAdmission<u32> {
    let Some(adapter) = adapter else {
        return KvAdmission {
            result: Ok(0),
            refusal: None,
            failure: None,
        };
    };
    adapter.clear_refusals();
    let current = adapter.current_token_count_for(seq_id).unwrap_or(0);
    let end = end.min(adapter.max_capacity_tokens());
    let result = adapter
        .reserve_rows_for(seq_id, end.saturating_sub(current))
        .map_err(Error::from_reason);
    let refusal = if result.is_err() {
        adapter.take_block_refusal()
    } else {
        None
    };
    KvAdmission {
        result,
        refusal,
        failure: claim.governor.failure(claim.id),
    }
}

/// What a request's admission claim decided (Splash `Engine::admit`).
enum Admitted {
    /// The GDN cell's reservation, when the request had none live.
    Granted(Option<Reservation<'static>>),
    Wait,
    Fail(Error),
}

/// A running or preempted scheduled turn.
type Turn<B> = TurnState<ScheduledTurn<<B as PagedBackend>::PrefixState>>;

type Action<B> = SchedulerAction<
    ScheduledTurn<<B as PagedBackend>::PrefixState>,
    Queued<<B as HybridSchedulerBackend>::FamilyCommand>,
    ModelCommand<<B as HybridSchedulerBackend>::FamilyCommand>,
>;

pub(crate) struct HybridSchedulerState<B: HybridSchedulerBackend> {
    pub(crate) inner: B,
    enabled: bool,
    scheduler: Scheduler<
        ScheduledTurn<B::PrefixState>,
        Queued<B::FamilyCommand>,
        ModelCommand<B::FamilyCommand>,
    >,
    pending: VecDeque<Queued<B::FamilyCommand>>,
    prepared_waiting: Option<Box<PreparedTurn>>,
    /// The starts the scheduler holds back (Splash `Engine::admitQueued`'s
    /// `queue`), in submission order: they hold back nothing.
    held: VecDeque<Box<PreparedTurn>>,
    pending_restores: HashMap<SeqId, B::RestoreTicket>,
    pub(crate) owner_sequences: HashMap<String, SeqId>,
    pub(crate) owner_states: HashMap<String, B::OwnerState>,
    /// Turn-end stamp of each idle owner's kept table, on the model's one
    /// cache clock.
    owner_last_used: HashMap<SeqId, u64>,
    /// The work end each row of the latest step claimed (Splash
    /// `Denied::workEnd`): a lane that yields resumes to its own.
    row_work_ends: HashMap<SeqId, u32>,
    /// Splash `Engine::admissions_`.
    admissions: u64,
    /// The lanes the latest step ran (Splash's command in flight), read by
    /// the next wait tick.
    worked_lanes: Vec<LaneWork>,
    /// The requests the published wanted chains were built from.
    wanted_signature: Vec<(SeqId, usize)>,
    /// What a step's GPU command that exhausted Metal memory reported.
    gpu_oom: Option<String>,
    /// The next lane admission or resume's prefix preparation fails with
    /// this text.
    #[cfg(test)]
    prefix_failure: Option<&'static str>,
    /// The current drive pass: its start on the governor's clock and what
    /// it ran.
    pass_start: f64,
    pass_work: PassWork,
    /// What held the requests that have no claim and no row yet.
    timeline: Timeline,
    /// When the model's accounts are next published.
    accounts_due: f64,
    next_seq_id: SeqId,
}

/// The tokens a planned row's step writes up to (Splash
/// `Engine::workEndOf`): a prefill row's own end, wherever it was cut, or
/// a decode row's pending token (already in `num_tokens`) and its draft
/// rows.
fn row_work_end<P>(row: &engine::scheduler::StepRow, turn: &TurnState<P>) -> u32 {
    match row.kind {
        StepKind::Prefill => row.token_start.saturating_add(row.num_tokens),
        StepKind::Decode => turn.num_tokens.saturating_add(turn.decode_draft_allowance),
    }
}

/// The pinned prefill ends of one prompt and whether the family runs it as
/// one whole-prompt step (then the grid is the one end `prompt`).
fn prefill_breaks<B: HybridSchedulerBackend>(
    inner: &B,
    cached_prefix: u32,
    prompt_tokens: u32,
    extra_breaks: Vec<u32>,
) -> (Vec<u32>, bool) {
    let whole = inner.scheduler_whole_prompt_prefill(prompt_tokens, cached_prefix);
    let slice = if whole {
        u32::MAX
    } else {
        inner.scheduler_prefill_slice_tokens()
    };
    (
        prefill_slice_ends(cached_prefix, prompt_tokens, slice, extra_breaks),
        whole,
    )
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
        let mut scheduler =
            Scheduler::new(max_num_seqs, PREFILL_CHUNK_ROWS).map_err(Error::from_reason)?;
        if let Some(config) = inner.measured_prefill_cut() {
            scheduler = scheduler.with_measured_prefill_cut(config);
        }
        if let Some(governor) = inner.memory_governor() {
            governor.register_scheduler(inner.memory_model(), scheduler.counters());
        }
        Ok(Self {
            inner,
            enabled,
            scheduler,
            pending: VecDeque::new(),
            prepared_waiting: None,
            held: VecDeque::new(),
            pending_restores: HashMap::new(),
            owner_sequences: HashMap::new(),
            owner_states: HashMap::new(),
            owner_last_used: HashMap::new(),
            row_work_ends: HashMap::new(),
            admissions: 0,
            worked_lanes: Vec::new(),
            wanted_signature: Vec::new(),
            gpu_oom: None,
            #[cfg(test)]
            prefix_failure: None,
            pass_start: 0.0,
            pass_work: PassWork::default(),
            timeline: Timeline::default(),
            accounts_due: 0.0,
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

    /// An ownerless speculative request takes the scheduled lane as the empty
    /// owner when the family keeps its ownerless session in the owner tables.
    /// The legacy-owner drain still admits it only into an idle scheduler, so
    /// the empty owner never has two turns in flight.
    fn ownerless_speculation_is_scheduled(&self, command: &ChatCmd) -> bool {
        self.inner.ownerless_session_in_owner_tables()
            && self.inner.supports_scheduled_speculation()
            && Self::chat_config(command).is_some_and(|config| config.enable_mtp == Some(true))
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
                    && speculative.lane(self.inner.supports_scheduled_speculation())
                        != engine::plan::SpeculativeLane::Scheduled
            })
    }

    fn adaptive_requires_exclusive_lane(
        &self,
        params: &engine::params::ChatParams,
        streaming: bool,
    ) -> bool {
        self.inner
            .execution_plan()
            .speculative
            .is_some_and(|speculative| speculative.admits_streaming(streaming))
            && params.mtp_adaptive_depth
            && (!self.inner.supports_adaptive_scheduled_speculation()
                || !crate::sampling::is_greedy_temperature(
                    params
                        .sampling_config
                        .and_then(|config| config.temperature)
                        .unwrap_or(1.0),
                ))
    }

    fn chat_requires_barrier(&self, command: &ChatCmd) -> bool {
        if Self::chat_config(command).is_some_and(|config| {
            config.enable_mtp == Some(true)
                && (self.speculation_requires_exclusive_lane(Self::chat_is_streaming(command))
                    || self.adaptive_requires_exclusive_lane(
                        &self.inner.resolve_params(config),
                        Self::chat_is_streaming(command),
                    ))
        }) {
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
            .or_else(|| self.inner.ownerless_session_in_owner_tables().then_some(""))
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
                .iter()
                .chain(&self.held)
                .any(|prepared| prepared.owner_id == owner_id)
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
        let Some(Queued { command, arrival }) = self.pending.remove(index) else {
            return DeferredCleanupProgress::None;
        };
        match command.into_chat() {
            Ok(ChatCmd::ReleaseCacheOwner { owner_id, reply }) => {
                let _ = reply.send(self.release_cache_owner_now(&owner_id));
                DeferredCleanupProgress::OwnerReleaseProcessed
            }
            Ok(chat) => {
                self.pending.insert(
                    index,
                    Queued {
                        command: ModelCommand::from_chat(chat),
                        arrival,
                    },
                );
                DeferredCleanupProgress::None
            }
            Err(command) => {
                self.pending.insert(index, Queued { command, arrival });
                DeferredCleanupProgress::None
            }
        }
    }

    fn prepare_chat(&mut self, command: ChatCmd, arrival: Arrival) -> Option<Box<PreparedTurn>> {
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
                    .enqueue_barrier(ModelCommand::from_chat(ChatCmd::ResetCaches { reply }));
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
        if !(admitted.plan.path() == engine::plan::TurnPath::Paged
            || (admitted.plan.path() == engine::plan::TurnPath::Speculative
                && self.inner.supports_scheduled_speculation()))
            || !admitted.images.is_empty()
            || !admitted.audio.is_empty()
            || (matches!(
                admitted.plan.decoder,
                engine::plan::DecoderPlan::Speculative(_)
            ) && (self.speculation_requires_exclusive_lane(streaming)
                || self.adaptive_requires_exclusive_lane(&admitted.params, streaming)))
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
        let speculative = matches!(
            admitted.plan.decoder,
            engine::plan::DecoderPlan::Speculative(_)
        );
        let max_new_tokens = match self.scheduled_output_budget(
            prompt_tokens,
            requested_max_new_tokens,
            block_size,
            speculative,
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
        let on_demand = self.inner.scheduler_books_on_demand();
        let (reservation_blocks, booked_tokens) =
            scheduled_booking(on_demand, prompt_tokens, max_new_tokens, block_size);
        let reservation_state_bytes =
            self.inner
                .recurrent_state_bytes()
                .saturating_add(if speculative {
                    self.inner.scheduled_draft_state_bytes(if on_demand {
                        booked_tokens
                    } else {
                        requested_tokens
                    })
                } else {
                    0
                });
        admitted.params.extra_eos_ids = self.inner.extra_eos_ids();
        Some(Box::new(PreparedTurn {
            reservation_state_bytes,
            admitted,
            response,
            cancelled,
            owner_id,
            seq_id,
            newly_assigned,
            reservation_blocks,
            block_size,
            requested_tokens,
            claim: None,
            resource_wait: ResourceWait::default(),
            enqueued_at: arrival.at,
            ttft: TtftClock::queued(self.timeline.since(&arrival.mark)),
            held: false,
            order: None,
            overtaken: 0,
        }))
    }

    /// Tokens one scheduled turn can use: the family's planned window, else
    /// the max pool after recurrent state. An on-demand speculative turn's
    /// draft state grows outside the pool with every row, so its planned
    /// window also pays the draft state it holds at its last row.
    #[cfg(test)]
    fn scheduled_turn_window(&self, block_size: u32, speculative: bool) -> u32 {
        self.inner
            .planned_turn_window(speculative && self.inner.scheduler_books_on_demand())
            .unwrap_or_else(|| self.pool_turn_window(block_size))
    }

    /// Tokens one scheduled turn can use in the max pool after recurrent
    /// state.
    fn pool_turn_window(&self, block_size: u32) -> u32 {
        let trained_context = u32::try_from(self.inner.max_position_embeddings())
            .unwrap_or(1)
            .max(1);
        let adapter = self.inner.paged_adapter();
        let pool_tokens = adapter
            .map(crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter::max_capacity_tokens)
            .unwrap_or(trained_context);
        let bytes_per_block = adapter
            .map(|adapter| adapter.bytes_per_block().unwrap_or(0))
            .unwrap_or(0);
        pool_tokens_after_recurrent(
            pool_tokens,
            block_size,
            bytes_per_block,
            self.inner.recurrent_state_bytes(),
        )
    }

    /// A scheduled turn's output budget: the request clamped to the turn's
    /// window by the family's rule
    /// ([`HybridSchedulerBackend::scheduler_output_budget_is_rest_of_window`]).
    fn scheduled_output_budget(
        &self,
        prompt_tokens: u32,
        requested_max_new_tokens: u32,
        block_size: u32,
        speculative: bool,
    ) -> std::result::Result<u32, String> {
        let trained_context = u32::try_from(self.inner.max_position_embeddings())
            .unwrap_or(1)
            .max(1);
        let planned = self
            .inner
            .planned_turn_window(speculative && self.inner.scheduler_books_on_demand());
        let context = scheduled_window(
            trained_context,
            planned,
            planned.unwrap_or_else(|| self.pool_turn_window(block_size)),
            scheduler_per_seq_context_override(),
        );
        engine::scheduler::clamp_scheduled_output_tokens(
            prompt_tokens,
            requested_max_new_tokens,
            context,
            self.inner.scheduler_output_budget_is_rest_of_window(),
        )
    }

    fn cleanup_rejected_prepared(&mut self, prepared: &PreparedTurn) {
        if prepared.newly_assigned {
            self.owner_states.remove(&prepared.owner_id);
            self.owner_sequences.remove(&prepared.owner_id);
            self.inner.release_scheduled_recurrent_for(prepared.seq_id);
            self.inner.release_scheduled_speculation(prepared.seq_id);
        }
    }

    /// Idle parked recurrent row that is not the candidate and is not in the
    /// scheduler's running/waiting set. Unit-cap eviction only. A family on
    /// the ledger evicts the oldest by its owner's stamp.
    fn idle_recurrent_victim(&self, seq_id: SeqId) -> Option<SeqId> {
        let idle = self.owner_sequences.values().copied().filter(|&candidate| {
            candidate != seq_id
                && self.inner.has_scheduled_recurrent(candidate)
                && !self.scheduler.contains_seq(candidate)
        });
        if self.inner.reclaims_in_order() {
            idle.min_by_key(|&candidate| (self.owner_stamp(candidate), candidate))
        } else {
            idle.min()
        }
    }

    /// The turn-end stamp of an idle owner's kept table, on the model's
    /// cache clock.
    fn owner_stamp(&self, seq_id: SeqId) -> u64 {
        self.owner_last_used
            .get(&seq_id)
            .copied()
            .or_else(|| {
                self.inner
                    .paged_adapter()
                    .and_then(|adapter| adapter.turn_end_stamp(seq_id))
            })
            .unwrap_or(0)
    }

    /// Idle owner that holds a recurrent row or a keep-live/cache table.
    /// Used by memory-denial reclaim only; unit-cap stays rec-only.
    fn idle_memory_victim(&self, seq_id: SeqId) -> Option<SeqId> {
        self.idle_owners(seq_id).first().copied()
    }

    /// Every owner [`Self::idle_memory_victim`] may pick, lowest first.
    fn idle_owners(&self, seq_id: SeqId) -> Vec<SeqId> {
        let mut candidates: Vec<SeqId> = self.owner_sequences.values().copied().collect();
        if let Some(adapter) = self.inner.paged_adapter() {
            candidates.extend(adapter.live_seq_ids());
        }
        candidates.sort_unstable();
        candidates.dedup();
        candidates.retain(|&candidate| {
            candidate != seq_id
                && !self.scheduler.contains_seq(candidate)
                && (self.inner.has_scheduled_recurrent(candidate)
                    || self.inner.scheduler_materialized_blocks(candidate) > 0
                    || self
                        .inner
                        .paged_adapter()
                        .is_some_and(|adapter| adapter.block_table_for(candidate).is_some()))
        });
        candidates
    }

    /// Releases an idle owner's row, draft state and table.
    fn release_owner(&mut self, seq_id: SeqId) -> bool {
        self.inner.release_scheduled_recurrent_for(seq_id);
        self.inner.release_scheduled_speculation(seq_id);
        if let Err(error) = self.inner.release_scheduled_cache(seq_id) {
            tracing::warn!(
                "{} could not release idle owner {seq_id} for a blocked step: {error:?}",
                B::SCHEDULER_NAME
            );
            return false;
        }
        self.owner_last_used.remove(&seq_id);
        true
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
        self.inner.release_scheduled_speculation(victim);
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

    #[cfg(test)]
    pub(crate) fn live_turns(&self) -> usize {
        self.scheduler.waiting_len() + self.scheduler.running_len()
    }

    #[cfg(test)]
    pub(crate) fn has_prepared_waiting(&self) -> bool {
        self.prepared_waiting.is_some()
    }

    #[cfg(test)]
    pub(crate) fn turn_mut(
        &mut self,
        seq_id: SeqId,
    ) -> Option<&mut TurnState<ScheduledTurn<B::PrefixState>>> {
        self.scheduler.turn_mut(seq_id)
    }

    #[cfg(test)]
    pub(crate) fn preempt_running(&mut self, seq_id: SeqId) -> bool {
        let Some(turn) = self.scheduler.preempt_running(seq_id) else {
            return false;
        };
        self.handle_preempted(PreemptedTurn {
            turn,
            blocked: false,
        });
        true
    }

    #[cfg(test)]
    pub(crate) fn resume_preempted(&mut self) {
        self.try_resume_preempted();
    }

    /// Runs one scheduler step whose rows execute after `before_execute`,
    /// i.e. after the step's cancel snapshot, and settles it.
    #[cfg(test)]
    pub(crate) fn step_after_snapshot(&mut self, before_execute: impl FnMut()) -> bool {
        struct Hooked<'a, B: HybridSchedulerBackend, F> {
            executor: HybridStepExecutor<'a, B>,
            before_execute: F,
        }
        impl<B: HybridSchedulerBackend, F: FnMut()> StepExecutor<ScheduledTurn<B::PrefixState>>
            for Hooked<'_, B, F>
        {
            type Error = Error;

            fn execute(
                &mut self,
                plan: &StepPlan,
                running: &mut [TurnState<ScheduledTurn<B::PrefixState>>],
            ) -> std::result::Result<StepResult, Self::Error> {
                (self.before_execute)();
                self.executor.execute(plan, running)
            }
        }
        self.begin_scheduled_step();
        let Ok(Ok((plan, claims))) = self.plan_and_claim() else {
            return false;
        };
        let policy = self.judge_policy();
        let mut hooked = Hooked {
            executor: HybridStepExecutor::new(&mut self.inner),
            before_execute,
        };
        let Ok(action) = self
            .scheduler
            .finish_step(plan, &claims.grants, &mut hooked, policy)
        else {
            return false;
        };
        if !matches!(action, SchedulerAction::Stepped { .. }) {
            return false;
        }
        self.settle_step(action);
        true
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
        self.inner.release_scheduled_speculation(victim);
        release_cache(&mut self.inner, victim)?;
        if !self.idle_reclaim_made_progress(seq_id, victim) {
            return Ok(false);
        }
        Ok(true)
    }

    /// Idle owners, one at a time, until one gives back pool blocks or a
    /// parked row (a speculative step's fit counts idle rows). True only
    /// when free plus reclaimable blocks grew or the parked rows shrank, so
    /// the blocked rows retry and the loop is bounded by what is idle.
    fn reclaim_idle_owners(&mut self, victim: SeqId) -> bool {
        let snapshot = |state: &Self| state.inner.scheduler_cache_snapshot().ok().flatten();
        let available = |snapshot: &SchedulerCacheSnapshot| {
            u64::from(snapshot.blocks.free_blocks) + u64::from(snapshot.blocks.reclaimable_blocks)
        };
        let Some(current) = snapshot(self) else {
            return false;
        };
        let before = available(&current);
        let rows_before = self.inner.scheduled_recurrent_bytes();
        while let Some(owner) = self.idle_memory_victim(victim) {
            if !self.release_owner(owner) {
                return false;
            }
            if snapshot(self).is_some_and(|after| available(&after) > before)
                || self.inner.scheduled_recurrent_bytes() < rows_before
            {
                return true;
            }
            if self.idle_memory_victim(victim) == Some(owner) {
                return false;
            }
        }
        false
    }

    /// Ends the step's completed turns and settles its blocked rows. False
    /// when a row blocked: no victim resumes in the same round.
    fn settle_step(&mut self, step: Action<B>) -> bool {
        let SchedulerAction::Stepped {
            completed,
            preempted,
            blocked_alone,
            blocked,
            progressed,
            planned,
            ..
        } = step
        else {
            return true;
        };
        let ledger = self.judge_policy() == JudgePolicy::Ledger;
        let mut completed = completed;
        if ledger {
            for turn in &mut completed {
                self.note_gpu_oom(&mut turn.payload.failure);
            }
            // Splash restarts on `metal_out_of_memory` however the step's
            // requests ended: MLX may raise one row's failed command in a
            // peer's, so no row of the step completes.
            if let Some(detail) = self.gpu_oom.clone() {
                let error = if detail.starts_with("metal_out_of_memory:") {
                    detail
                } else {
                    crate::engine::governor::metal_out_of_memory(&detail)
                };
                for turn in &mut completed {
                    if turn.payload.failure.is_none() {
                        turn.payload.failure = Some(Error::from_reason(error.clone()));
                    }
                }
            }
        }
        for turn in completed {
            self.finish_completed(turn);
        }
        if ledger {
            return self.settle_blocked(blocked, progressed, &planned);
        }
        if let Some(seq_id) = blocked_alone {
            self.handle_blocked_alone(seq_id);
            return false;
        }
        let Some(mut preempted) = preempted else {
            return true;
        };
        if self.inner.scheduler_books_on_demand() {
            // Its caches stay resident until handle_preempted releases them.
            preempted.turn.block_reservation_active = true;
            preempted.turn.recurrent_state_reservation_active = true;
            if self.reclaim_idle_owners(preempted.turn.seq_id) {
                self.scheduler.reinstate_running(preempted.turn);
                return false;
            }
        }
        self.handle_preempted(*preempted);
        false
    }

    /// A family whose rows claim their memory is judged as Splash's;
    /// every other family keeps the preempt-the-newest policy.
    fn judge_policy(&self) -> JudgePolicy {
        if self.inner.reclaims_in_order() {
            JudgePolicy::Ledger
        } else {
            JudgePolicy::Legacy
        }
    }

    fn memory_governor(&self) -> &'static Governor {
        self.inner
            .memory_governor()
            .unwrap_or_else(crate::engine::governor::governor)
    }

    /// The KV a lane holds now, in tokens (Splash `Engine::kvCapacity`).
    fn kv_capacity(&self, seq_id: SeqId) -> u32 {
        self.inner
            .scheduler_materialized_blocks(seq_id)
            .saturating_mul(
                self.inner
                    .paged_adapter()
                    .map_or(0, |adapter| adapter.block_size()),
            )
    }

    fn yields_before(&self, a: SeqId, b: SeqId) -> bool {
        let running = self.scheduler.running();
        let turn = |seq_id: SeqId| running.iter().find(|turn| turn.seq_id == seq_id);
        match (turn(a), turn(b)) {
            (Some(a), Some(b)) => a.yields_before(b),
            _ => false,
        }
    }

    /// Splash `Engine::prepare` after its claims, and `Engine::refuse`. A
    /// refusal no transfer returns and that is not KV yields the lane to
    /// yield first when it holds its memory outside the step and yields
    /// before the refused row. Otherwise a step that ran some rows replans
    /// the refused ones at the next step, and a step that ran none waits
    /// for memory on its way back, or judges the row that yields first.
    fn settle_blocked(
        &mut self,
        blocked: Vec<(SeqId, Denial)>,
        progressed: bool,
        planned: &[SeqId],
    ) -> bool {
        if blocked.is_empty() {
            return true;
        }
        let yielding = self.scheduler.lane_to_yield();
        if let Some(yielding) = yielding
            && !planned.contains(&yielding)
            && let Some((_, denial)) = blocked.iter().find(|(seq_id, denial)| {
                denial.blocks.is_none() && !denial.pending && self.yields_before(yielding, *seq_id)
            })
        {
            let failure = denial.failure;
            let target = self.kv_capacity(yielding);
            self.yield_lane(yielding, target, failure);
            return false;
        }
        if progressed {
            return false;
        }
        if blocked.iter().any(|(_, denial)| denial.pending) {
            self.defer_blocked(&blocked);
            return false;
        }
        let Some(&(victim, denial)) = blocked.iter().reduce(|left, right| {
            if self.yields_before(right.0, left.0) {
                right
            } else {
                left
            }
        }) else {
            return false;
        };
        let yielding = yielding.unwrap_or(victim);
        match judge(&denial, self.scheduler.another_resident(yielding)) {
            Verdict::Wait => self.defer_blocked(&blocked),
            Verdict::Yield => {
                let target = if blocked.iter().any(|(seq_id, _)| *seq_id == yielding) {
                    self.row_work_ends
                        .get(&yielding)
                        .copied()
                        .unwrap_or_else(|| self.kv_capacity(yielding))
                } else {
                    self.kv_capacity(yielding)
                };
                self.yield_lane(yielding, target, denial.failure);
            }
            Verdict::Fail => {
                let what = if denial.blocks.is_some() {
                    "KV target"
                } else {
                    "draft state"
                };
                let error =
                    capacity_exhausted(what, denial.failure, &page_shortfall(denial.blocks));
                self.memory_governor().capacity_exhausted();
                if let Some(turn) = self.scheduler.take_running(victim) {
                    self.fail_preempted(turn, error);
                }
            }
        }
        false
    }

    /// Splash `refuse`'s wait: every refused lane takes a retry deadline.
    fn defer_blocked(&mut self, blocked: &[(SeqId, Denial)]) {
        let governor = self.memory_governor();
        let model = self.inner.memory_model();
        let (now, epoch) = (governor.now(), governor.epoch(model));
        let admissions = self.admissions;
        for (seq_id, denial) in blocked {
            if let Some(turn) = self.scheduler.running_turn_mut(*seq_id) {
                defer_resource_retry(
                    &mut turn.resource_wait,
                    now,
                    denial,
                    WaitReason::Memory,
                    true,
                    admissions,
                    epoch,
                );
            }
        }
    }

    /// Splash `Engine::admit`'s KV refusal for a scheduled prepare (the
    /// chain it restores, the pool grown for it): the scheduler judges what
    /// the prepare's claim left beside the resident lanes. `None`: the
    /// prepare failed for anything but memory. `Some(Err)`: the request
    /// fails `capacity_exhausted`. `Some(Ok)`: it waits, deferred.
    fn prefix_refused(
        &mut self,
        seq_id: SeqId,
        wait: &mut ResourceWait,
        has_lane: bool,
    ) -> Option<std::result::Result<(), Error>> {
        let denial = crate::engine::governor::governor::take_lane_denial()?;
        let governor = self.memory_governor();
        let model = self.inner.memory_model();
        if judge(&denial, self.scheduler.another_resident(seq_id)) == Verdict::Fail {
            governor.capacity_exhausted();
            return Some(Err(Error::from_reason(capacity_exhausted(
                "KV target",
                denial.failure,
                &page_shortfall(denial.blocks),
            ))));
        }
        defer_resource_retry(
            wait,
            governor.now(),
            &denial,
            WaitReason::Memory,
            has_lane,
            self.admissions,
            governor.epoch(model),
        );
        Some(Ok(()))
    }

    /// Splash `Engine::suspendForGrowth` → `suspendLane`: the lane is
    /// vacated (its KV released, its history to replay), takes `target`
    /// tokens of KV at once when it resumes, and waits as a memory
    /// refusal; the resident lanes drain first.
    fn yield_lane(&mut self, seq_id: SeqId, target: u32, failure: Option<Failure>) {
        let Some(mut turn) = self.scheduler.preempt_running(seq_id) else {
            return;
        };
        let governor = self.memory_governor();
        let model = self.inner.memory_model();
        let (now, epoch) = (governor.now(), governor.epoch(model));
        // A lane vacated before its first step drops its admission's cell
        // bytes: its resume reserves them again.
        turn.payload.cell = None;
        turn.resume_target = target;
        defer_resource_retry(
            &mut turn.resource_wait,
            now,
            &Denial {
                failure,
                ..Denial::default()
            },
            WaitReason::Memory,
            true,
            self.admissions,
            epoch,
        );
        governor.lane_suspended(model, now, failure);
        let blocked = self.row_work_ends.contains_key(&seq_id);
        self.handle_preempted(PreemptedTurn { turn, blocked });
    }

    /// Splash `Engine::growthClass`: the lane that yields first reaches
    /// only ordinary units while another lane is resident.
    fn growth_class(&self, seq_id: SeqId) -> ReachClass {
        if self.scheduler.lane_to_yield() == Some(seq_id) && self.scheduler.another_resident(seq_id)
        {
            ReachClass::Ordinary
        } else {
            ReachClass::InUse
        }
    }

    /// Splash `Engine::commit` ends a lane at its stop or length token, so
    /// a decode row whose un-emitted pending token ends the turn writes
    /// nothing: its step finishes it without a forward. The observer stop
    /// is left out (observing mutates it).
    fn pending_ends_turn(turn: &mut Turn<B>) -> bool {
        if turn.payload.pending_token_emitted {
            return false;
        }
        let Some(&token) = turn.token_history.last() else {
            return false;
        };
        if token == turn.payload.eos_id
            || turn.payload.extra_eos_ids.contains(&token)
            || turn.payload.generated_tokens.len() + 1
                >= turn.payload.params.max_new_tokens.max(0) as usize
        {
            return true;
        }
        turn.payload.generated_tokens.push(token);
        let repetition = check_repetition_cutoff(
            &turn.payload.generated_tokens,
            turn.payload.params.max_consecutive_tokens,
            turn.payload.params.max_ngram_repeats,
            turn.payload.params.ngram_size,
        );
        turn.payload.generated_tokens.pop();
        repetition.is_some()
    }

    /// Splash `Engine::prepare`'s claims: in plan order each row grows its
    /// lane to its work end under its request's InService claim
    /// (`growLane`), a speculative row's draft-state growth (a prefill
    /// slice's draft K/V too) a bytes claim under the same one. A granted
    /// row's wait ends; a denied row notes its refusal. A row refused for
    /// anything but memory runs and fails in its step. A row whose pending
    /// token ends its turn claims nothing.
    pub(crate) fn claim_rows_in_order(&mut self, plan: &StepPlan) -> StepClaims {
        self.inner.sync_state_marks();
        self.row_work_ends.clear();
        let governor = self.memory_governor();
        let model = self.inner.memory_model();
        let mut claims = StepClaims {
            grants: Vec::with_capacity(plan.rows.len()),
            reservations: Vec::new(),
        };
        for row in &plan.rows {
            // A cancelled or stopped row ends in its step and grows nothing.
            if row.cancel_snapshot {
                claims.grants.push(None);
                continue;
            }
            let reach = self.growth_class(row.seq_id);
            let Some(turn) = self.scheduler.running_turn_mut(row.seq_id) else {
                claims.grants.push(None);
                continue;
            };
            if row.kind == StepKind::Decode && Self::pending_ends_turn(turn) {
                claims.grants.push(None);
                continue;
            }
            let work_end = row_work_end(row, turn);
            let hold_ended = turn.resource_wait.hold_ended;
            let draft_from = match row.kind {
                StepKind::Prefill => row.token_start,
                StepKind::Decode => turn.num_computed_tokens,
            };
            let speculative = turn.payload.scheduled_speculation.is_some();
            let mut spec = ClaimSpec {
                cls: ClaimClass::InService,
                request_id: 0,
                rank: turn.claim_rank(),
                reach,
                timing: NeedTiming::CanWait,
            };
            let claim = match &turn.payload.claim {
                Some(handle) => handle.claim(),
                None => {
                    spec.request_id = crate::engine::governor::governor::next_request_id();
                    let handle = governor.open(spec, model);
                    let claim = handle.claim();
                    turn.payload.claim = Some(handle);
                    claim
                }
            };
            spec.request_id = governor.spec(claim.id).map_or(0, |open| open.request_id);
            let draft_bytes = if speculative {
                self.inner
                    .scheduled_draft_state_held_bytes(work_end)
                    .saturating_sub(self.inner.scheduled_draft_state_held_bytes(draft_from))
            } else {
                0
            };
            let seq_id = row.seq_id;
            let mut units = SchedUnits {
                state: &mut *self,
                claimant: seq_id,
                kept: &[],
            };
            let allocation = allocate(
                claim,
                spec,
                Refusal::Waits,
                hold_ended,
                &mut units,
                |units| {
                    let kv = ensure_tokens(
                        units.state.inner.paged_adapter_mut(),
                        seq_id,
                        work_end,
                        claim,
                    );
                    let state = (kv.granted() && draft_bytes > 0)
                        .then(|| StateAdmission(governor.try_reserve(draft_bytes)));
                    RowAdmission { kv, state }
                },
            );
            let admission = allocation.admission;
            let denied = !admission.granted() && admission.memory_denied();
            let (now, epoch) = (governor.now(), governor.epoch(model));
            let Some(turn) = self.scheduler.running_turn_mut(row.seq_id) else {
                claims.grants.push(None);
                continue;
            };
            if denied {
                note_refusal(&mut turn.resource_wait, now, &allocation.denial, epoch);
                self.row_work_ends.insert(row.seq_id, work_end);
                claims.grants.push(Some(allocation.denial));
            } else {
                turn.resource_wait = Default::default();
                governor.ledger(|ledger| ledger.set_waiting(claim.id, false));
                if let Some(Ok(reservation)) = admission.state.map(|state| state.0) {
                    claims.reservations.push(reservation);
                }
                claims.reservations.extend(turn.payload.cell.take());
                claims.grants.push(None);
            }
        }
        claims
    }

    /// Splash `Engine::admit`'s claim class: a request starts beside a
    /// resident lane (`Start`, which reaches no state a turn uses) or alone
    /// in service.
    fn admission_spec(&self, seq_id: SeqId, request_id: u64, rank: ClaimRank) -> ClaimSpec {
        let start = self.scheduler.another_resident(seq_id);
        ClaimSpec {
            cls: if start {
                ClaimClass::Start
            } else {
                ClaimClass::InService
            },
            request_id,
            rank,
            reach: if start {
                ReachClass::Ordinary
            } else {
                ReachClass::InUse
            },
            timing: NeedTiming::CanWait,
        }
    }

    /// The request's claim, opened at its first admission attempt.
    fn request_claim(
        governor: &'static Governor,
        model: u64,
        slot: &mut Option<ClaimHandle<'static>>,
        spec: ClaimSpec,
    ) -> ClaimRef<'static> {
        slot.get_or_insert_with(|| governor.open(spec, model))
            .claim()
    }

    /// Splash `Engine::admit`'s state claim: the GDN cell `seq_id`
    /// activates and, for a speculative lane, its draft state's fixed part
    /// (Splash `CompositeStateLayout::laneBytes`), a bytes reservation (the
    /// arrays are allocated lazily). `None` when the request's row is live
    /// already.
    fn claim_cell(
        &mut self,
        seq_id: SeqId,
        kept: &[u32],
        claim: ClaimRef<'static>,
        spec: ClaimSpec,
        hold_ended: bool,
        speculative: bool,
    ) -> Option<crate::engine::governor::allocate::Allocation<StateAdmission<'static>>> {
        if self.inner.has_scheduled_recurrent(seq_id) {
            return None;
        }
        let draft = if speculative {
            self.inner.scheduled_draft_state_held_bytes(0)
        } else {
            0
        };
        let bytes = self.inner.recurrent_state_bytes().saturating_add(draft);
        let governor = claim.governor;
        let mut units = SchedUnits {
            state: &mut *self,
            claimant: seq_id,
            kept,
        };
        Some(allocate(
            claim,
            spec,
            Refusal::Waits,
            hold_ended,
            &mut units,
            |_| StateAdmission(governor.try_reserve(bytes)),
        ))
    }

    /// Splash `Engine::admitQueued` + `Engine::admit` for a new request on
    /// the ledger. Nothing new starts while a lane is suspended or the
    /// resident lanes drain; the unit cap waits for a lane without a
    /// deadline. A refusal the judge fails gives up the request's own kept
    /// table first (`droppedLease`): the prepare looks its chain up again.
    fn admit_claimed(&mut self, prepared: &mut PreparedTurn) -> Admitted {
        let governor = self.memory_governor();
        let model = self.inner.memory_model();
        let seq_id = prepared.seq_id;
        let now = governor.now();
        if self.scheduler.has_preempted()
            || governor.draining_for_recovery(model, now, self.scheduler.has_resident())
            || !retry_ready(&prepared.resource_wait, now, governor.epoch(model))
        {
            return Admitted::Wait;
        }
        if !self.ensure_recurrent_slot(seq_id) {
            defer_resource_retry(
                &mut prepared.resource_wait,
                now,
                &Denial::default(),
                WaitReason::Concurrency,
                false,
                self.admissions,
                governor.epoch(model),
            );
            return Admitted::Wait;
        }
        let request_id = prepared
            .claim
            .as_ref()
            .and_then(|claim| governor.spec(claim.id()))
            .map_or_else(next_request_id, |spec| spec.request_id);
        let rank = ClaimRank {
            priority: 1,
            decoding: false,
            completed_tokens: 0,
            earliness: u64::MAX - request_id,
        };
        let spec = self.admission_spec(seq_id, request_id, rank);
        let claim = Self::request_claim(governor, model, &mut prepared.claim, spec);
        let hold_ended = prepared.resource_wait.hold_ended;
        let speculative = matches!(
            prepared.admitted.plan.decoder,
            engine::plan::DecoderPlan::Speculative(_)
        );
        let mut dropped_lease = false;
        loop {
            let Some(allocation) = self.claim_cell(
                seq_id,
                &prepared.admitted.tokens,
                claim,
                spec,
                hold_ended,
                speculative,
            ) else {
                return Admitted::Granted(None);
            };
            if let Ok(reservation) = allocation.admission.0 {
                prepared.resource_wait = ResourceWait::default();
                governor.ledger(|ledger| ledger.set_waiting(claim.id, false));
                return Admitted::Granted(Some(reservation));
            }
            let denial = allocation.denial;
            let verdict = judge(&denial, self.scheduler.another_resident(seq_id));
            if verdict == Verdict::Fail
                && !dropped_lease
                && self.inner.scheduler_materialized_blocks(seq_id) > 0
            {
                dropped_lease = true;
                if self.inner.release_scheduled_cache(seq_id).is_ok() {
                    continue;
                }
            }
            if verdict == Verdict::Fail {
                governor.capacity_exhausted();
                return Admitted::Fail(Error::from_reason(capacity_exhausted(
                    "request state",
                    denial.failure,
                    "",
                )));
            }
            defer_resource_retry(
                &mut prepared.resource_wait,
                governor.now(),
                &denial,
                WaitReason::Memory,
                false,
                self.admissions,
                governor.epoch(model),
            );
            return Admitted::Wait;
        }
    }

    /// Splash `Engine::admitQueued` + `Engine::admit` for a suspended
    /// request on the ledger: retried once its retry time came or the
    /// epoch moved, never while the resident lanes drain; its state claim
    /// covers its GDN cell.
    fn resume_cell(&mut self, turn: &mut Turn<B>) -> Admitted {
        let governor = self.memory_governor();
        let model = self.inner.memory_model();
        let seq_id = turn.seq_id;
        let (now, epoch) = (governor.now(), governor.epoch(model));
        if !retry_ready(&turn.resource_wait, now, epoch)
            || governor.draining_for_recovery(model, now, self.scheduler.has_resident())
        {
            return Admitted::Wait;
        }
        if !self.ensure_recurrent_slot(seq_id) {
            defer_resource_retry(
                &mut turn.resource_wait,
                now,
                &Denial::default(),
                WaitReason::Concurrency,
                false,
                self.admissions,
                epoch,
            );
            return Admitted::Wait;
        }
        let kept = turn.payload.preemption_replay.as_ref().map_or_else(
            || turn.payload.prompt_tokens.clone(),
            |replay| replay.tokens.clone(),
        );
        let (claim, spec) = self.resume_claim(turn);
        let hold_ended = turn.resource_wait.hold_ended;
        let speculative = turn.payload.scheduled_speculation.is_some();
        let Some(allocation) = self.claim_cell(seq_id, &kept, claim, spec, hold_ended, speculative)
        else {
            return Admitted::Granted(None);
        };
        if let Ok(reservation) = allocation.admission.0 {
            turn.resource_wait = ResourceWait::default();
            governor.ledger(|ledger| ledger.set_waiting(claim.id, false));
            return Admitted::Granted(Some(reservation));
        }
        let denial = allocation.denial;
        if judge(&denial, self.scheduler.another_resident(seq_id)) == Verdict::Fail {
            self.memory_governor().capacity_exhausted();
            return Admitted::Fail(Error::from_reason(capacity_exhausted(
                "request state",
                denial.failure,
                "",
            )));
        }
        defer_resource_retry(
            &mut turn.resource_wait,
            governor.now(),
            &denial,
            WaitReason::Memory,
            false,
            self.admissions,
            governor.epoch(model),
        );
        Admitted::Wait
    }

    /// A suspended request's claim, Start beside a resident lane.
    fn resume_claim(&self, turn: &mut Turn<B>) -> (ClaimRef<'static>, ClaimSpec) {
        let governor = self.memory_governor();
        let request_id = turn
            .payload
            .claim
            .as_ref()
            .and_then(|claim| governor.spec(claim.id()))
            .map_or_else(next_request_id, |spec| spec.request_id);
        let spec = self.admission_spec(turn.seq_id, request_id, turn.claim_rank());
        let claim = Self::request_claim(
            governor,
            self.inner.memory_model(),
            &mut turn.payload.claim,
            spec,
        );
        (claim, spec)
    }

    /// Splash `Engine::admit`: a lane that resumes takes the KV target it
    /// was suspended with at once, after the chain it restored.
    fn claim_resume_target(&mut self, turn: &mut Turn<B>, kept: &[u32]) -> Admitted {
        let target = turn.resume_target;
        if target == 0 {
            return Admitted::Granted(None);
        }
        let governor = self.memory_governor();
        let model = self.inner.memory_model();
        let seq_id = turn.seq_id;
        let (claim, spec) = self.resume_claim(turn);
        let hold_ended = turn.resource_wait.hold_ended;
        let mut units = SchedUnits {
            state: &mut *self,
            claimant: seq_id,
            kept,
        };
        let allocation = allocate(
            claim,
            spec,
            Refusal::Waits,
            hold_ended,
            &mut units,
            |units| ensure_tokens(units.state.inner.paged_adapter_mut(), seq_id, target, claim),
        );
        if allocation.admission.granted() {
            turn.resource_wait = ResourceWait::default();
            governor.ledger(|ledger| ledger.set_waiting(claim.id, false));
            return Admitted::Granted(None);
        }
        let denial = allocation.denial;
        if !allocation.admission.memory_denied() {
            return Admitted::Fail(
                allocation
                    .admission
                    .result
                    .err()
                    .unwrap_or_else(|| Error::from_reason("resume target refused")),
            );
        }
        if judge(&denial, self.scheduler.another_resident(seq_id)) == Verdict::Fail {
            self.memory_governor().capacity_exhausted();
            return Admitted::Fail(Error::from_reason(capacity_exhausted(
                "KV target",
                denial.failure,
                &page_shortfall(denial.blocks),
            )));
        }
        defer_resource_retry(
            &mut turn.resource_wait,
            governor.now(),
            &denial,
            WaitReason::Memory,
            false,
            self.admissions,
            governor.epoch(model),
        );
        Admitted::Wait
    }

    /// One scheduler step: the plan, its rows' claims, then the granted
    /// rows run.
    fn step(&mut self) -> std::result::Result<Action<B>, SchedulerError<Error>> {
        let (plan, claims) = match self.plan_and_claim()? {
            Ok(planned) => planned,
            Err(action) => return Ok(action),
        };
        let policy = self.judge_policy();
        // The step's batched forward is the engine's own (Splash's engine
        // claim); each row's work is its request's (`HybridStepExecutor`).
        let model = self.inner.memory_model();
        let engine = (model != 0).then(|| {
            self.memory_governor()
                .open(crate::engine::governor::governor::whole_turn_spec(0), model)
        });
        let _engine_scope = engine.as_ref().map(|claim| Scope::enter(claim.claim()));
        let mut executor = self.inner.step_executor();
        self.scheduler
            .finish_step(plan, &claims.grants, &mut executor, policy)
    }

    /// The step's plan without the rows whose wait has not reached its
    /// retry (Splash `Engine::prepare` plans no deferred lane), and their
    /// claims. The granted rows are the lanes this step works.
    #[allow(clippy::type_complexity)]
    fn plan_and_claim(
        &mut self,
    ) -> std::result::Result<
        std::result::Result<(StepPlan, StepClaims), Action<B>>,
        SchedulerError<Error>,
    > {
        let clock = (self.judge_policy() == JudgePolicy::Ledger).then(|| {
            let governor = self.memory_governor();
            RetryClock {
                now: governor.now(),
                epoch: governor.epoch(self.inner.memory_model()),
            }
        });
        let mut plan = match self
            .scheduler
            .prepare_step(clock)
            .map_err(SchedulerError::InvalidResult)?
        {
            Prepared::Action(action) => return Ok(Err(action)),
            Prepared::Plan(plan) => plan,
        };
        if let Some(RetryClock { now, epoch }) = clock {
            let scheduler = &mut self.scheduler;
            plan.rows.retain(|row| {
                row.cancel_snapshot
                    || scheduler
                        .running_turn_mut(row.seq_id)
                        .is_none_or(|turn| retry_ready(&turn.resource_wait, now, epoch))
            });
            if plan.rows.is_empty() {
                return Ok(Err(SchedulerAction::Idle));
            }
        }
        let claims = B::claim_step_rows(self, &plan);
        for (row, grant) in plan.rows.iter().zip(&claims.grants) {
            if grant.is_none() {
                match row.kind {
                    StepKind::Prefill => self.pass_work.prefill.push(row.seq_id),
                    StepKind::Decode => self.pass_work.decode = true,
                }
            }
        }
        self.worked_lanes = plan
            .rows
            .iter()
            .zip(&claims.grants)
            .filter(|(_, grant)| grant.is_none())
            .filter_map(|(row, _)| {
                self.scheduler
                    .running()
                    .iter()
                    .find(|turn| turn.seq_id == row.seq_id)
                    .map(|turn| LaneWork {
                        submitted: turn.arrival_order(),
                        admission: turn.admission,
                    })
            })
            .collect();
        Ok(Ok((plan, claims)))
    }

    /// Splash `Engine::collectLandings` → `Cache::setWantedChains`: the
    /// prompts of the requests without a lane, the one admission waits on
    /// and the suspended or queued turns' replays.
    fn refresh_wanted_chains(&mut self) {
        if self.judge_policy() != JudgePolicy::Ledger {
            return;
        }
        let tokens = |turn: &Turn<B>| -> Vec<u32> {
            turn.payload.preemption_replay.as_ref().map_or_else(
                || turn.payload.prompt_tokens.clone(),
                |replay| replay.tokens.clone(),
            )
        };
        let signature: Vec<(SeqId, usize)> = self
            .prepared_waiting
            .iter()
            .chain(&self.held)
            .map(|prepared| (prepared.seq_id, prepared.admitted.tokens.len()))
            .chain(
                self.scheduler
                    .turns()
                    .filter(|turn| turn.status != TurnStatus::Running)
                    .map(|turn| (turn.seq_id, tokens(turn).len())),
            )
            .collect();
        if signature == self.wanted_signature {
            return;
        }
        let prompts = self
            .prepared_waiting
            .iter()
            .chain(&self.held)
            .map(|prepared| prepared.admitted.tokens.clone())
            .chain(
                self.scheduler
                    .turns()
                    .filter(|turn| turn.status != TurnStatus::Running)
                    .map(tokens),
            )
            .collect();
        self.wanted_signature = signature;
        self.memory_governor()
            .set_wanted_chains(self.inner.memory_model(), WantedChains::new(prompts));
    }

    /// Splash `Engine::syncClaims` and `Engine::tick`'s waits, once per
    /// drive pass on the ledger: a claim holds what it waits for only while
    /// its request is retried; a wait ends `resource_timeout` at its
    /// deadline, which a lane submitted or admitted before it restarts
    /// while it works; nothing times out while the resident lanes drain.
    fn tick_waits(&mut self) {
        if self.judge_policy() != JudgePolicy::Ledger {
            return;
        }
        let governor = self.memory_governor();
        let model = self.inner.memory_model();
        let (now, epoch) = (governor.now(), governor.epoch(model));
        let draining = governor.draining_for_recovery(model, now, self.scheduler.has_resident());
        let worked = std::mem::take(&mut self.worked_lanes);
        let has_preempted = self.scheduler.has_preempted();
        let first_preempted = self.scheduler.first_preempted();
        let mut keys: Vec<Option<SeqId>> = Vec::new();
        let mut waiters: Vec<Waiter<'_>> = Vec::new();
        if let Some(prepared) = self.prepared_waiting.as_mut() {
            keys.push(None);
            waiters.push(Waiter {
                claim: prepared.claim.as_ref().map(ClaimHandle::id),
                wait: &mut prepared.resource_wait,
                // Every turn the scheduler holds was submitted before it.
                submitted: u64::MAX,
                finalized: false,
                retried: !has_preempted && !draining,
            });
        }
        for turn in self.scheduler.turns_mut() {
            let retried = match turn.status {
                TurnStatus::Preempted => !draining && first_preempted == Some(turn.seq_id),
                _ => true,
            };
            // A cancel or stop is the turn's end (Splash `Engine::deferEnd`
            // keeps the first): its wait no longer times out.
            let finalized = is_cancelled(turn);
            keys.push(Some(turn.seq_id));
            waiters.push(Waiter {
                claim: turn.payload.claim.as_ref().map(ClaimHandle::id),
                submitted: turn.arrival_order(),
                wait: &mut turn.resource_wait,
                finalized,
                retried,
            });
        }
        sync_claims(governor, &mut waiters, now, epoch);
        let timed_out: Vec<(Option<SeqId>, Option<Failure>)> =
            tick(&mut waiters, &worked, now, draining, epoch)
                .into_iter()
                .map(|at| (keys[at], waiters[at].wait.failure))
                .collect();
        drop(waiters);
        for (key, failure) in timed_out {
            let error = resource_timeout(failure);
            governor.resource_timed_out();
            match key {
                None => {
                    if let Some(prepared) = self.prepared_waiting.take() {
                        self.cleanup_rejected_prepared(&prepared);
                        prepared
                            .response
                            .send_error(Error::from_reason(error), prepared.cancelled.as_ref());
                    }
                }
                Some(seq_id) => {
                    let turn = match self.scheduler.take_running(seq_id) {
                        Some(turn) => Some(turn),
                        None => {
                            self.pending_restores.remove(&seq_id);
                            self.scheduler.take_waiting(seq_id)
                        }
                    };
                    if let Some(turn) = turn {
                        self.fail_preempted(turn, error);
                    }
                }
            }
        }
    }

    /// A command off the channel: the memory monitor's wake is taken here,
    /// before anything waits behind it.
    fn receive(&mut self, command: ModelCommand<B::FamilyCommand>) -> bool {
        match command {
            ModelCommand::MemoryControl => {
                crate::engine::governor::monitor::wake_taken(self.inner.memory_model());
                true
            }
            command => {
                self.pending.push_back(Queued {
                    command,
                    arrival: Arrival {
                        at: Instant::now(),
                        mark: self.timeline,
                    },
                });
                false
            }
        }
    }

    /// Splash `MemoryControl::run` on the model thread: the governor's
    /// control pass over the model's units and idle owners, once per
    /// monitor period, at once on the monitor's wake, and again while the
    /// pass waits for what lands.
    fn run_control_pass(&mut self, woke: bool) {
        if self.judge_policy() != JudgePolicy::Ledger {
            return;
        }
        let model = self.inner.memory_model();
        let governor = self.memory_governor();
        if model == 0 || !governor.control_due(model, woke) {
            return;
        }
        let mut units = SchedUnits {
            state: &mut *self,
            claimant: SeqId::MAX,
            kept: &[],
        };
        if governor.control_pass(model, &mut [&mut units]).again() {
            governor.control_again(model);
        }
    }

    /// Runs a command whole on the model thread: a barrier, or a turn the
    /// exclusive lane holds. `Some` ends the pass.
    fn run_command(
        &mut self,
        command: ModelCommand<B::FamilyCommand>,
        arrival: Option<Arrival>,
        wired: Option<&mut crate::stream::WiredUse>,
    ) -> Option<LoopControl> {
        let reset = command
            .as_chat()
            .is_some_and(|chat| matches!(chat, ChatCmd::ResetCaches { .. }));
        let command = match command.into_scheduler_stats() {
            Ok(reply) => {
                let _ = reply.send(Ok(self.scheduler.stats().to_js()));
                self.close_pass(&mut []);
                return Some(LoopControl::Continue);
            }
            Err(command) => command,
        };
        if let Some(wired) = wired {
            wired.mark();
        }
        let model = self.inner.memory_model();
        // A whole turn is its request's one claim, alone on the
        // model, and a wait for memory ends at its cancel.
        let cancel = command.as_chat().and_then(ChatCmd::cancel_flag);
        let whole_turn = model != 0 && cancel.is_some();
        let turn = whole_turn.then(|| {
            let governor = crate::engine::governor::governor();
            governor.open(
                crate::engine::governor::governor::whole_turn_spec(
                    crate::engine::governor::governor::next_request_id(),
                ),
                model,
            )
        });
        let _turn_scope = turn
            .as_ref()
            .map(|claim| crate::engine::governor::governor::Scope::enter(claim.claim()));
        let _cancel_scope = cancel.map(crate::engine::governor::governor::CancelScope::enter);
        // The turn holds the model as a command does: what waits behind it
        // is held by its decode. Its own TTFT runs from its arrival.
        self.pass_work.decode |= whole_turn;
        let timed = arrival.filter(|_| whole_turn).map(|arrival| {
            (
                self.timeline.since(&arrival.mark),
                Instant::now(),
                WholeTurnScope::enter(arrival.at),
            )
        });
        self.inner.execute_barrier(
            command,
            SchedulerOwnerContext {
                owner_sequences: &mut self.owner_sequences,
                owner_states: &mut self.owner_states,
                next_seq_id: &mut self.next_seq_id,
            },
        );
        if let Some((queued, started, scope)) = timed
            && let Some((first, claim_wait)) = scope.finish()
        {
            let mut clock = crate::engine::governor::metrics::whole_turn_ttft(
                queued,
                first.saturating_duration_since(started).as_secs_f64(),
                claim_wait,
            );
            if let Some(parts) = clock.first_tokens() {
                self.memory_governor().first_tokens(&parts);
            }
        }
        if reset {
            self.owner_sequences.clear();
            self.owner_states.clear();
        }
        self.close_pass(&mut []);
        None
    }

    /// Splash `Engine::holds` at the end of a drive pass, on the ledger: the
    /// pass's time goes to what held each request without its first tokens
    /// (`holderOf` over the pass's work); a lane's token events are timed
    /// for decode gaps (`decodeGap`); the model's waits, replans and
    /// accounts are published for the governor's status.
    fn close_pass(&mut self, finished: &mut [Turn<B>]) {
        if self.judge_policy() != JudgePolicy::Ledger {
            return;
        }
        let governor = self.memory_governor();
        let model = self.inner.memory_model();
        let now = governor.now();
        let seconds = (now - self.pass_start).max(0.0);
        let work = std::mem::take(&mut self.pass_work);
        let releases = governor.pool_releases();
        let replans = self.scheduler.stats().replans;
        let waits = |claim: Option<&ClaimHandle<'static>>| {
            claim.is_some_and(|claim| {
                governor.ledger(|ledger| {
                    ledger
                        .claim(claim.id())
                        .is_some_and(|state| state.waiting || state.host_refused_waiting)
                })
            })
        };
        self.timeline.hold(seconds, work.holder_of(None, false));
        for prepared in self.prepared_waiting.iter_mut().chain(&mut self.held) {
            let part = work.holder_of(Some(prepared.seq_id), waits(prepared.claim.as_ref()));
            prepared.ttft.hold(seconds, part);
        }
        for turn in self.scheduler.turns_mut().chain(finished.iter_mut()) {
            let claim_waits = waits(turn.payload.claim.as_ref());
            let part = work.holder_of(Some(turn.seq_id), claim_waits);
            let clock = &mut turn.payload.clock;
            clock.ttft.hold(seconds, part);
            clock
                .gap
                .worked(work.prefill_beside(turn.seq_id), claim_waits);
            if clock.token_events > clock.seen_events {
                clock.seen_events = clock.token_events;
                if let Some(parts) = clock.ttft.first_tokens() {
                    governor.first_tokens(&parts);
                }
                if let Some(gap) = clock.gap.token(now, releases, replans) {
                    governor.decode_gap(&gap);
                }
            }
        }
        if model == 0 {
            return;
        }
        let draining = governor.draining_for_recovery(model, now, self.scheduler.has_resident());
        let mut status = ResourceWaitStatus {
            draining,
            ..ResourceWaitStatus::default()
        };
        let mut count = |wait: &ResourceWait, suspended: bool| {
            let Some(started) = wait.started else {
                return false;
            };
            match wait.reason {
                WaitReason::Memory => status.memory += 1,
                WaitReason::Concurrency => status.concurrency += 1,
                WaitReason::None => return false,
            }
            status.suspended += u32::from(suspended);
            status.oldest_ms = status.oldest_ms.max((now - started).max(0.0) * 1000.0);
            wait.reason == WaitReason::Memory
        };
        let refused = self
            .prepared_waiting
            .as_ref()
            .is_some_and(|prepared| count(&prepared.resource_wait, false));
        for turn in self.scheduler.turns() {
            if turn.status == TurnStatus::Preempted {
                count(&turn.resource_wait, true);
            }
        }
        // The chats behind a start refused memory wait without being tried
        // (Splash `heldBehindRefusal`).
        if refused {
            status.held_behind_refusal = self
                .pending
                .iter()
                .filter(|queued| {
                    queued
                        .as_chat()
                        .is_some_and(|chat| !matches!(chat, ChatCmd::ReleaseCacheOwner { .. }))
                })
                .count() as u32;
        }
        governor.publish_waits(model, status, replans);
        if now >= self.accounts_due
            && let Some(accounts) = self.inner.memory_accounts()
        {
            governor.publish_accounts(model, accounts);
            if let Some(states) = self.inner.dflash2_state_status() {
                governor.publish_dflash2_states(model, states);
            }
            self.accounts_due = now + crate::engine::governor::monitor::MONITOR_PERIOD_SECONDS;
        }
    }

    /// A finished row's failure that a GPU command exhausting Metal memory
    /// caused becomes `metal_out_of_memory:`, and the model's caches reset
    /// after the step.
    fn note_gpu_oom(&mut self, failure: &mut Option<Error>) {
        let Some(error) = failure.as_mut() else {
            return;
        };
        if error.reason.starts_with("metal_out_of_memory:") {
            self.gpu_oom.get_or_insert_with(|| error.reason.clone());
        } else if crate::engine::governor::is_gpu_out_of_memory(&error.reason) {
            self.gpu_oom = Some(error.reason.clone());
            *error =
                Error::from_reason(crate::engine::governor::metal_out_of_memory(&error.reason));
        }
    }

    /// Splash `NativeRuntime::executionFailed` around `Engine::tick`'s
    /// admission: a lane's prefix preparation that a GPU command exhausting
    /// Metal memory failed restarts the model like a failed step. The
    /// request's error, or `None` for any other failure.
    fn prefix_gpu_oom(&mut self, error: &Error) -> Option<String> {
        if self.judge_policy() != JudgePolicy::Ledger
            || !crate::engine::governor::is_gpu_out_of_memory(&error.reason)
        {
            return None;
        }
        let _ = crate::engine::governor::governor::take_lane_denial();
        self.gpu_oom = Some(error.reason.clone());
        Some(if error.reason.starts_with("metal_out_of_memory:") {
            error.reason.clone()
        } else {
            crate::engine::governor::metal_out_of_memory(&error.reason)
        })
    }

    /// The step failed as a whole. On the ledger a GPU command that
    /// exhausted Metal memory fails the model's requests and resets its
    /// caches, and the thread lives; any other failure ends the thread.
    fn executor_failed(&mut self, error: Error) -> LoopControl {
        if self.judge_policy() == JudgePolicy::Ledger
            && crate::engine::governor::is_gpu_out_of_memory(&error.reason)
        {
            self.gpu_oom = Some(error.reason);
            self.recover_if_gpu_oom();
            return LoopControl::Continue;
        }
        tracing::error!(
            "{} scheduler execution failed: {error:?}",
            B::SCHEDULER_NAME
        );
        LoopControl::Break
    }

    /// Splash's engine restart after `metal_out_of_memory`, the thread
    /// kept: MLX reports a command's failure at a later command, so what
    /// any running command wrote may be garbage. Every request with a lane
    /// or a prepared admission fails, the idle owners and every cache of
    /// the model go, and the MLX free pool is returned. Commands not yet
    /// started stay queued.
    fn recover_if_gpu_oom(&mut self) {
        let model = self.inner.memory_model();
        if model == 0 && self.gpu_oom.is_none() {
            return;
        }
        let governor = self.memory_governor();
        if self.gpu_oom.is_some() {
            governor.note_gpu_oom(model);
        }
        let noted = governor.take_gpu_oom(model);
        let detail = self.gpu_oom.take();
        if !noted && detail.is_none() {
            return;
        }
        let detail = detail.unwrap_or_else(|| "a whole turn's command".to_string());
        let error = if detail.starts_with("metal_out_of_memory:") {
            detail.clone()
        } else {
            crate::engine::governor::metal_out_of_memory(&detail)
        };
        tracing::error!(
            target: "mlx_core::memory",
            "{} reset after a GPU command exhausted Metal memory: {detail}",
            B::SCHEDULER_NAME
        );
        let seq_ids: Vec<SeqId> = self.scheduler.turns().map(|turn| turn.seq_id).collect();
        for seq_id in seq_ids {
            let turn = match self.scheduler.take_running(seq_id) {
                Some(turn) => Some(turn),
                None => self.scheduler.take_waiting(seq_id),
            };
            if self.pending_restores.remove(&seq_id).is_some() {
                self.restore_landed();
            }
            if let Some(turn) = turn {
                self.fail_preempted(turn, error.clone());
            }
        }
        let held = std::mem::take(&mut self.held);
        for prepared in self.prepared_waiting.take().into_iter().chain(held) {
            self.cleanup_rejected_prepared(&prepared);
            prepared.response.send_error(
                Error::from_reason(error.clone()),
                prepared.cancelled.as_ref(),
            );
        }
        for seq_id in self.idle_owners(SeqId::MAX) {
            self.release_owner(seq_id);
        }
        self.owner_sequences.clear();
        self.owner_states.clear();
        self.owner_last_used.clear();
        if let Err(reset) = self.inner.reset_after_gpu_oom() {
            tracing::warn!(
                target: "mlx_core::memory",
                "{} cache reset after a GPU memory failure: {reset}",
                B::SCHEDULER_NAME
            );
        }
        crate::array::memory::clear_cache();
    }

    /// [`Self::step`] with a test executor.
    #[cfg(test)]
    fn step_with<E>(
        &mut self,
        executor: &mut E,
    ) -> std::result::Result<Action<B>, SchedulerError<Error>>
    where
        E: StepExecutor<ScheduledTurn<B::PrefixState>, Error = Error>,
    {
        let (plan, claims) = match self.plan_and_claim()? {
            Ok(planned) => planned,
            Err(action) => return Ok(action),
        };
        let policy = self.judge_policy();
        self.scheduler
            .finish_step(plan, &claims.grants, executor, policy)
    }

    /// The step's only running turn blocked. A family that books on demand
    /// first releases idle owners; with none left the turn cannot fit even
    /// alone, so it fails with `capacity_exhausted` and is released once,
    /// never preempted: a replay would block at the same step (Splash
    /// `Engine::judge` fails a lone lane). Other families book the whole
    /// turn and keep the preempt-and-resume path.
    fn handle_blocked_alone(&mut self, seq_id: SeqId) {
        if !self.inner.scheduler_books_on_demand() {
            if let Some(turn) = self.scheduler.preempt_running(seq_id) {
                self.handle_preempted(PreemptedTurn {
                    turn,
                    blocked: true,
                });
            }
            return;
        }
        if self.reclaim_idle_owners(seq_id) {
            return;
        }
        let error = self.capacity_exhausted(seq_id);
        if let Some(turn) = self.scheduler.take_running(seq_id) {
            self.fail_preempted(turn, error.reason.clone());
        }
    }

    fn capacity_exhausted(&self, seq_id: SeqId) -> Error {
        let held = self.inner.scheduler_materialized_blocks(seq_id);
        let pool = match self.inner.scheduler_cache_snapshot() {
            Ok(Some(snapshot)) => format!(
                "it holds {held} paged blocks; the pool has {} of at most {} blocks, {} free",
                snapshot.blocks.total_blocks,
                snapshot.max_total_blocks,
                u64::from(snapshot.blocks.free_blocks)
                    + u64::from(snapshot.blocks.reclaimable_blocks)
            ),
            _ => format!("it holds {held} paged blocks"),
        };
        Error::from_reason(format!(
            "capacity_exhausted: could not allocate KV or draft state for this turn's next step \
             ({pool}); the request does not fit in the memory the page pool may use, even alone \
             after every idle cache owner was released"
        ))
    }

    fn idle_reclaim_made_progress(&self, seq_id: SeqId, victim: SeqId) -> bool {
        self.idle_memory_victim(seq_id) != Some(victim)
    }

    fn try_reserve_reclaiming_idle(
        &mut self,
        seq_id: SeqId,
        reservation_blocks: u32,
        candidate_state_bytes: u64,
        state_candidate_bytes: u64,
        snapshot: SchedulerCacheSnapshot,
    ) -> Result<MemoryReserveOutcome> {
        self.try_reserve_reclaiming_idle_with(
            seq_id,
            reservation_blocks,
            candidate_state_bytes,
            state_candidate_bytes,
            snapshot,
            B::release_scheduled_cache,
        )
    }

    fn scheduled_state_exceeded(&self, candidate_bytes: u64) -> Error {
        let capacity = self
            .inner
            .scheduled_state_capacity()
            .unwrap_or(ScheduledStateCapacity { budget: 0, held: 0 });
        let mib = |bytes: u64| bytes as f64 / (1u64 << 20) as f64;
        Error::from_reason(format!(
            "{} working memory: this turn needs {:.0} MiB of recurrent and draft state beside \
             {:.0} MiB already held, but only {:.0} MiB remain after the page pool and the \
             prefill window; release idle cache owners or retry without enableMtp",
            B::SCHEDULER_NAME,
            mib(candidate_bytes),
            mib(capacity.held),
            mib(capacity.budget)
        ))
    }

    /// Draft state live speculative turns hold past the charge they were
    /// admitted with. An on-demand turn is charged its booked rows' draft
    /// state, and its draft K/V grows with every decoded row outside the
    /// pool, so what it holds now (and the next step's draft rows) is
    /// charged as it grows.
    fn grown_draft_state_bytes(&self) -> u64 {
        self.scheduler.running().iter().fold(0u64, |sum, turn| {
            sum.saturating_add(uncharged_draft_state_bytes(
                &self.inner,
                turn,
                turn.num_tokens.saturating_add(turn.decode_draft_allowance),
            ))
        })
    }

    /// Unified-memory admission decision against one cache snapshot.
    fn reserve_memory_decision(
        &mut self,
        snapshot: &SchedulerCacheSnapshot,
        reservation_blocks: u32,
        candidate_state_bytes: u64,
    ) -> engine::scheduler::MemoryAdmission {
        let candidate_state_bytes =
            candidate_state_bytes.saturating_add(self.grown_draft_state_bytes());
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
        state_candidate_bytes: u64,
        snapshot: SchedulerCacheSnapshot,
        release_cache: F,
    ) -> Result<MemoryReserveOutcome>
    where
        F: FnMut(&mut B, SeqId) -> Result<()>,
    {
        let may_wait = self.scheduler.has_live_turns();
        self.try_reserve_with_policy(
            seq_id,
            reservation_blocks,
            candidate_state_bytes,
            state_candidate_bytes,
            snapshot,
            may_wait,
            release_cache,
        )
    }

    /// `may_wait`: other turns can still free memory, so a reservation that
    /// does not fit defers instead of rejecting.
    #[allow(clippy::too_many_arguments)]
    fn try_reserve_with_policy<F>(
        &mut self,
        seq_id: SeqId,
        reservation_blocks: u32,
        candidate_state_bytes: u64,
        state_candidate_bytes: u64,
        mut snapshot: SchedulerCacheSnapshot,
        may_wait: bool,
        mut release_cache: F,
    ) -> Result<MemoryReserveOutcome>
    where
        F: FnMut(&mut B, SeqId) -> Result<()>,
    {
        let mut grown_for_live_turns = false;
        let mut grown_before_reclaim = false;
        loop {
            let decision =
                self.reserve_memory_decision(&snapshot, reservation_blocks, candidate_state_bytes);
            let outstanding_bytes = {
                let inner = &self.inner;
                self.scheduler
                    .outstanding_scheduled_state_bytes(|seq| inner.scheduled_state_held_bytes(seq))
            };
            let capacity = self.inner.scheduled_state_capacity();
            let state = scheduled_state_fit(capacity, outstanding_bytes, state_candidate_bytes);
            if decision.admitted && state == ScheduledStateFit::Fits {
                return Ok(MemoryReserveOutcome::Admitted);
            }
            // Splash `Cache::takeReclaim`: free pool extents first, then idle
            // owners; state is refused only once neither frees anything.
            if state != ScheduledStateFit::Fits
                && let Some(capacity) = capacity
                && self.inner.reclaim_for_scheduled_state(
                    capacity
                        .held
                        .saturating_add(outstanding_bytes)
                        .saturating_add(state_candidate_bytes)
                        .saturating_sub(capacity.budget),
                )
            {
                if let Some(updated) = self.inner.scheduler_cache_snapshot()? {
                    snapshot = updated;
                }
                continue;
            }
            // An on-demand family's pool starts at a runway: ask for the
            // grow once before an idle owner's cache is given up, as Splash
            // takes a new KV extent before it reclaims. A refused grow
            // frees nothing: the idle owner's in-pool blocks go first, and
            // parked state beside the pool only once none is left.
            if self.inner.scheduler_books_on_demand()
                && !grown_before_reclaim
                && state == ScheduledStateFit::Fits
                && self.idle_memory_victim(seq_id).is_some()
                && let Some(needed) =
                    grow_needed_blocks_for(&snapshot, &decision, scheduler_watermark_fraction())
            {
                grown_before_reclaim = true;
                if self
                    .inner
                    .paged_adapter_mut()
                    .is_some_and(|adapter| adapter.try_grow_pool(needed))
                    && let Some(grown) = self.inner.scheduler_cache_snapshot()?
                {
                    snapshot = grown;
                    continue;
                }
            }
            if !self.reclaim_one_idle_scheduled_with(seq_id, &mut release_cache)? {
                if state == ScheduledStateFit::Never {
                    return Err(self.scheduled_state_exceeded(state_candidate_bytes));
                }
                if may_wait {
                    if self.inner.scheduler_books_on_demand()
                        && !grown_for_live_turns
                        && state == ScheduledStateFit::Fits
                        && let Some(needed) = grow_needed_blocks_for(
                            &snapshot,
                            &decision,
                            scheduler_watermark_fraction(),
                        )
                    {
                        grown_for_live_turns = true;
                        if let Some(grown) = self.try_grow_pool(needed)? {
                            snapshot = grown;
                            continue;
                        }
                    }
                    return Ok(MemoryReserveOutcome::Defer);
                }
                if state != ScheduledStateFit::Fits {
                    return Err(self.scheduled_state_exceeded(state_candidate_bytes));
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
        self.try_grow_pool(needed)
    }

    /// Grows the pool by `needed` blocks; the fresh snapshot, or `None`
    /// when it did not grow.
    fn try_grow_pool(&mut self, needed: u32) -> Result<Option<SchedulerCacheSnapshot>> {
        if self
            .inner
            .paged_adapter_mut()
            .is_some_and(|adapter| adapter.try_grow_pool(needed))
        {
            self.inner.scheduler_cache_snapshot()
        } else {
            Ok(None)
        }
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

    /// Blocks an admission charges: the reservation less the kept live
    /// blocks it continues on.
    fn admission_charge(
        &self,
        reservation_blocks: u32,
        seq_id: SeqId,
        tokens: &[u32],
        reuse: bool,
    ) -> u32 {
        reservation_blocks.saturating_sub(reusable_keep_live_blocks(
            self.inner.paged_adapter(),
            seq_id,
            tokens,
            reuse,
        ))
    }

    /// Whether the scheduler holds the prepared start back
    /// (`Scheduler::prefillAdmissionOrder`), which Splash ranks by its
    /// cached tokens: a live continuation's are the ones it keeps.
    fn admission_waits(&self, prepared: &PreparedTurn) -> bool {
        let tokens = &prepared.admitted.tokens;
        let reuse = prepared.admitted.plan.is_delta || prepared.admitted.params.reuse_cache;
        let cached = live_held_prefix(self.inner.paged_adapter(), prepared.seq_id, tokens, reuse)
            .unwrap_or(0);
        !self.scheduler.serves_candidate(
            prepared.seq_id,
            u32::try_from(cached).unwrap_or(u32::MAX),
            u32::try_from(tokens.len()).unwrap_or(u32::MAX),
            prepared.order,
            prepared.overtaken,
        )
    }

    /// Splash `Engine::admitQueued`'s `queue` (`deferWait`): a start the
    /// scheduler holds back waits for scheduling, and its claim holds no
    /// room meanwhile. A start backing off from a lane wait is left as it
    /// is.
    fn defer_admission(&self, prepared: &mut PreparedTurn) {
        if self.judge_policy() != JudgePolicy::Ledger {
            return;
        }
        let governor = self.memory_governor();
        let wait = &mut prepared.resource_wait;
        let epoch = governor.epoch(self.inner.memory_model());
        if wait.reason == WaitReason::Concurrency && !retry_ready(wait, governor.now(), epoch) {
            return;
        }
        defer_wait(wait);
        if let Some(claim) = prepared.claim.as_ref() {
            let id = claim.id();
            // `sync_claims` never visits a held start, so its host refusal
            // is cleared here (Splash `Engine::deferWait` resets the
            // allocation failure `Engine::syncClaims` reads).
            governor.ledger(|ledger| {
                if ledger.claim(id).is_some_and(|state| state.waiting) {
                    ledger.set_waiting(id, false);
                }
                ledger.set_host_refused_waiting(id, false);
            });
        }
    }

    fn admit_prepared(
        &mut self,
        mut prepared: Box<PreparedTurn>,
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
        // Splash `Scheduler::prefillAdmissionOrder`: it takes no memory
        // until the unit it would join serves it.
        prepared.held = self.admission_waits(&prepared);
        if prepared.held {
            self.defer_admission(&mut prepared);
            if prepared.order.is_none() {
                prepared.order = Some(self.scheduler.reserve_order());
            }
            return Err(prepared);
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
        let candidate_state_bytes = prepared.reservation_state_bytes.saturating_sub(
            if self.inner.has_scheduled_recurrent(prepared.seq_id) {
                self.inner.recurrent_state_bytes()
            } else {
                0
            },
        );
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
        let ledger = self.judge_policy() == JudgePolicy::Ledger;
        let mut cell = None;
        if ledger {
            match self.admit_claimed(&mut prepared) {
                Admitted::Granted(reservation) => cell = reservation,
                Admitted::Wait => return Err(prepared),
                Admitted::Fail(error) => {
                    self.cleanup_rejected_prepared(&prepared);
                    prepared
                        .response
                        .send_error(error, prepared.cancelled.as_ref());
                    return Ok(None);
                }
            }
        } else {
            // A preempted row resumes before any later turn books memory: a
            // booking taken now can keep the victim from ever fitting again,
            // while the victim blocks the later turn from running.
            if self.inner.scheduler_books_on_demand() && self.scheduler.has_preempted() {
                return Err(prepared);
            }
            if !self.ensure_recurrent_slot(prepared.seq_id) {
                return Err(prepared);
            }
        }
        let mut scheduled_state_bytes = 0u64;
        if !ledger && let Some(snapshot) = cache_snapshot {
            let reuse = prepared.admitted.plan.is_delta || prepared.admitted.params.reuse_cache;
            let charge = self.admission_charge(
                prepared.reservation_blocks,
                prepared.seq_id,
                &prepared.admitted.tokens,
                reuse,
            );
            let state_candidate_bytes = self.inner.scheduled_state_candidate_bytes(
                prepared.seq_id,
                prepared.requested_tokens,
                matches!(
                    prepared.admitted.plan.decoder,
                    engine::plan::DecoderPlan::Speculative(_)
                ),
            );
            scheduled_state_bytes = state_candidate_bytes
                .saturating_add(self.inner.scheduled_state_held_bytes(prepared.seq_id));
            let reserved = self.try_reserve_reclaiming_idle(
                prepared.seq_id,
                charge,
                candidate_state_bytes,
                state_candidate_bytes,
                snapshot,
            );
            match reserved {
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

        if prepared.cancelled.load(Ordering::Relaxed) {
            if prepared.newly_assigned {
                self.owner_states.remove(&prepared.owner_id);
                self.owner_sequences.remove(&prepared.owner_id);
                self.inner.release_scheduled_recurrent_for(prepared.seq_id);
                self.inner.release_scheduled_speculation(prepared.seq_id);
            }
            prepared.response.send_error(
                Error::from_reason(engine::session::CHAT_SESSION_CANCELLED),
                prepared.cancelled.as_ref(),
            );
            return Ok(None);
        }
        let seq_id = prepared.seq_id;
        if let Err(error) = self.inner.activate_scheduled_recurrent(seq_id) {
            if prepared.newly_assigned {
                self.owner_states.remove(&prepared.owner_id);
                self.owner_sequences.remove(&prepared.owner_id);
                self.inner.release_scheduled_recurrent_for(seq_id);
                self.inner.release_scheduled_speculation(seq_id);
            }
            prepared
                .response
                .send_error(error, prepared.cancelled.as_ref());
            return Ok(None);
        }
        let owner_state = self
            .owner_states
            .entry(prepared.owner_id.clone())
            .or_default();
        self.inner.install_owner_state(seq_id, owner_state);
        self.inner.set_cache_owner_id(
            &prepared.admitted.params.cache_owner_id,
            prepared.admitted.params.cache_root_owner_id.as_deref(),
        );
        self.inner.scheduled_turn_opened(seq_id);
        self.inner
            .set_turn_cancel_flag(Some(Arc::clone(&prepared.cancelled)));
        let owner_history = self
            .owner_states
            .get(&prepared.owner_id)
            .map(B::owner_tokens)
            .unwrap_or_default()
            .to_vec();
        let scope = prepared
            .claim
            .as_ref()
            .map(|claim| Scope::enter_lane(claim.claim()));
        let prefix_admission = self.inner.prepare_scheduled_prefix(
            seq_id,
            &prepared.admitted.tokens,
            &owner_history,
            prepared.admitted.plan.is_delta || prepared.admitted.params.reuse_cache,
            prepared.admitted.params.cache_salt,
            prepared.block_size,
        );
        drop(scope);
        #[cfg(test)]
        let prefix_admission = match self.prefix_failure.take() {
            Some(reason) => Err(Error::from_reason(reason)),
            None => prefix_admission,
        };
        let prefix_admission = match prefix_admission {
            Ok(prefix) => prefix,
            Err(error) => {
                self.inner.abort_paged_turn();
                self.inner.release_scheduled_recurrent_for(seq_id);
                self.inner.release_scheduled_speculation(seq_id);
                self.inner.scheduled_turn_closed(seq_id);
                self.inner.set_turn_cancel_flag(None);
                let oom = self.prefix_gpu_oom(&error);
                let error = match oom.clone() {
                    Some(reason) => Error::from_reason(reason),
                    None => match self.prefix_refused(seq_id, &mut prepared.resource_wait, false) {
                        Some(Err(failed)) => failed,
                        Some(Ok(())) => return Err(prepared),
                        None => error,
                    },
                };
                self.owner_states.remove(&prepared.owner_id);
                self.owner_sequences.remove(&prepared.owner_id);
                prepared
                    .response
                    .send_error(error, prepared.cancelled.as_ref());
                if oom.is_some() {
                    self.recover_if_gpu_oom();
                }
                return Ok(None);
            }
        };
        let PreparedTurn {
            admitted,
            response,
            cancelled,
            owner_id,
            seq_id,
            reservation_blocks,
            block_size,
            reservation_state_bytes,
            requested_tokens,
            claim,
            enqueued_at,
            ttft,
            order,
            overtaken,
            ..
        } = *prepared;
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
            self.inner.release_scheduled_speculation(seq_id);
            self.inner.scheduled_turn_closed(seq_id);
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
        let generation_stream = Stream::generation();
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
        // On the ledger a request's TTFT runs from its arrival (Splash counts
        // from submission).
        let generation_start = admitted.params.report_performance.then(|| {
            if self.judge_policy() == JudgePolicy::Ledger {
                enqueued_at
            } else {
                Instant::now()
            }
        });
        let reuse_cache = admitted.plan.is_delta || admitted.params.reuse_cache;
        let payload = ScheduledTurn {
            owner_id,
            scheduled_speculation: matches!(
                admitted.plan.decoder,
                engine::plan::DecoderPlan::Speculative(_)
            )
            .then(|| {
                use rand::SeedableRng;
                rand::rngs::StdRng::from_rng(&mut rand::rng())
            }),
            pending_token_emitted: false,
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
                &admitted.close,
            ),
            extra_eos_ids: self.inner.extra_eos_ids(),
            generation_start,
            first_token_instant: None,
            generation_stream,
            profiler,
            emitter: is_streaming.then(|| -> Box<dyn StreamEmitter> {
                Box::new(engine::backend::ProgressEmitter(
                    self.inner.stream_emitter(),
                ))
            }),
            turn_token_observer: self.inner.turn_token_observer(),
            stream_skip_special: self.inner.stream_skip_special_tokens(),
            decode_ids: Vec::new(),
            decode_prefix: String::new(),
            decode_prefix_index: 0,
            streamed_text_len: 0,
            last_is_reasoning: admitted.thinking.enabled,
            failure: None,
            vacated_stop: false,
            allocation_failed: false,
            preemption_replay: None,
            requested_tokens,
            claim,
            cell,
            clock: TurnClock {
                ttft,
                ..TurnClock::default()
            },
        };
        let prompt_len = admitted.tokens.len() as u32;
        let cached_prefix = payload.prefix.effective_cached_prefix_len() as u32;
        let extra_breaks = self.inner.extra_prefill_breaks(prompt_len, cached_prefix);
        let (breaks, whole_prompt_prefill) =
            prefill_breaks(&self.inner, cached_prefix, prompt_len, extra_breaks);
        let turn_cancelled = Arc::clone(&cancelled);
        let draft_allowance = if payload.scheduled_speculation.is_some() {
            payload.params.mtp_depth as u32
        } else {
            0
        };
        let mut turn = match TurnState::try_new_recover_payload(
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
                self.inner.release_scheduled_speculation(seq_id);
                self.inner.scheduled_turn_closed(seq_id);
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
        .with_recurrent_state_reservation(reservation_state_bytes)
        .with_scheduled_state_reservation(scheduled_state_bytes);
        turn.decode_draft_allowance = draft_allowance;
        turn.whole_prompt_prefill = whole_prompt_prefill;
        turn.submitted_order = order;
        turn.overtaken = overtaken;
        if ledger {
            self.admissions = self.admissions.saturating_add(1);
            turn.admission = self.admissions;
        }
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

    /// Cancelled requests end between steps (Splash `Engine::cancel`): the
    /// waiting ones, and on the ledger the suspended ones (not in flight,
    /// Splash `Engine::settle` ends them at once) and the running ones too,
    /// whose refused rows would otherwise keep their memory and their
    /// claim's takes until they ran again.
    fn reap_cancelled_waiters(&mut self) {
        let (cancelled, held): (Vec<_>, Vec<_>) = std::mem::take(&mut self.held)
            .into_iter()
            .partition(|prepared| prepared.cancelled.load(Ordering::Relaxed));
        self.held = held.into();
        for prepared in cancelled {
            self.cleanup_rejected_prepared(&prepared);
            prepared.response.send_error(
                Error::from_reason(engine::session::CHAT_SESSION_CANCELLED),
                prepared.cancelled.as_ref(),
            );
        }
        let mut reaped = self.scheduler.take_cancelled_waiters();
        if self.judge_policy() == JudgePolicy::Ledger {
            let suspended: Vec<SeqId> = self
                .scheduler
                .turns()
                .filter(|turn| turn.status == TurnStatus::Preempted && is_cancelled(turn))
                .map(|turn| turn.seq_id)
                .collect();
            reaped.extend(
                suspended
                    .into_iter()
                    .filter_map(|seq_id| self.scheduler.take_waiting(seq_id)),
            );
        }
        for mut turn in reaped {
            if self.pending_restores.remove(&turn.seq_id).is_some() {
                self.restore_landed();
            }
            if !stop_vacates(&mut turn) {
                turn.payload.failure =
                    Some(Error::from_reason(engine::session::CHAT_SESSION_CANCELLED));
            }
            self.finish_completed(turn);
        }
        if self.judge_policy() != JudgePolicy::Ledger {
            return;
        }
        // A stopped running turn ends in its step: its prefill row vacates,
        // its decode row commits what it generated.
        let cancelled: Vec<SeqId> = self
            .scheduler
            .running()
            .iter()
            .filter(|turn| is_cancelled(turn) && !stopped_after_sampling(turn))
            .map(|turn| turn.seq_id)
            .collect();
        for seq_id in cancelled {
            if let Some(turn) = self.scheduler.take_running(seq_id) {
                self.fail_preempted(turn, engine::session::CHAT_SESSION_CANCELLED.to_string());
            }
        }
    }

    /// A cold-tier restore landed or was cancelled (Splash
    /// `collectLandings` signals progress): what waited for its memory may
    /// fit now.
    fn restore_landed(&self) {
        if self.judge_policy() == JudgePolicy::Ledger {
            self.memory_governor()
                .signal_progress(self.inner.memory_model());
        }
    }

    fn poll_restores(&mut self) {
        let seq_ids = self.pending_restores.keys().copied().collect::<Vec<_>>();
        for seq_id in seq_ids {
            let Some(turn) = self.scheduler.waiting_turn_mut(seq_id) else {
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
                    &turn.payload.prompt_tokens,
                    self.owner_states
                        .get(&turn.payload.owner_id)
                        .map(B::owner_tokens)
                        .unwrap_or_default(),
                    turn.payload.preemption_replay.is_some(),
                )
            };
            match outcome {
                Ok(None) => {}
                Ok(Some(restored)) => {
                    self.pending_restores.remove(&seq_id);
                    self.restore_landed();
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
                    (turn.pinned_prefill_breaks, turn.whole_prompt_prefill) = prefill_breaks(
                        &self.inner,
                        turn.num_computed_tokens,
                        turn.prompt_tokens,
                        restored.extra_prefill_breaks,
                    );
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
    fn handle_preempted(&mut self, preempted: PreemptedTurn<ScheduledTurn<B::PrefixState>>) {
        let PreemptedTurn { mut turn, blocked } = preempted;
        let held_blocks = turn.block_materialized_blocks;
        let prefix_tokens = turn.num_computed_tokens;
        let prompt_tokens = turn.payload.prompt_tokens.clone();
        let replay = match install_preemption_replay(
            &mut turn,
            &prompt_tokens,
            self.inner.scheduler_prefill_slice_tokens(),
        ) {
            Ok(replay) => {
                (turn.pinned_prefill_breaks, turn.whole_prompt_prefill) =
                    prefill_breaks(&self.inner, 0, turn.prompt_tokens, Vec::new());
                replay
            }
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
        self.inner.release_scheduled_speculation(turn.seq_id);
        if let Err(error) = lifecycle {
            let _ = self.inner.release_scheduled_cache(turn.seq_id);
            self.fail_preempted(turn, error.reason.clone());
            return;
        }
        if self.inner.scheduler_books_on_demand() && !self.inner.reclaims_in_order() {
            self.book_resume(
                &mut turn,
                replay.tokens.len() as u32,
                blocked,
                held_blocks,
                !replay.suppress_sample,
            );
        }
        turn.payload.preemption_replay = Some(replay);
        self.scheduler.record_preemption_mode(mode);
        self.scheduler.prepend_preempted(turn);
    }

    /// A victim that blocked resumes only into room for its replayed history
    /// plus the step it was preempted at (the pending token and its draft
    /// rows); a victim preempted in prefill resumes into its prompt.
    /// Booking less re-admits it into the same shortage, so it replays
    /// and blocks again every step. A victim that did not block resumes to
    /// the blocks it held (`held_blocks`, read before replay zeroes them).
    /// A speculative victim's replay rebuilds draft state for its whole
    /// history, so it is charged what a running turn holds at its booked
    /// rows, never less than its admission charge. Pricing hidden rows for
    /// the whole history instead makes a lone victim reject a resume whose
    /// rows still fit the max pool.
    fn book_resume(
        &self,
        turn: &mut TurnState<ScheduledTurn<B::PrefixState>>,
        history_tokens: u32,
        blocked: bool,
        held_blocks: u32,
        prefill_replay: bool,
    ) {
        turn.block_reservation_total = if blocked {
            resume_booking_blocks(
                turn.block_reservation_total,
                history_tokens,
                turn.decode_draft_allowance,
                turn.payload.requested_tokens,
                turn.block_size.max(1),
                prefill_replay,
            )
        } else {
            turn.block_reservation_total.max(held_blocks)
        };
        if turn.payload.scheduled_speculation.is_some() {
            let tokens = turn.payload.requested_tokens.min(
                turn.block_reservation_total
                    .saturating_mul(turn.block_size.max(1)),
            );
            turn.recurrent_state_bytes = turn.recurrent_state_bytes.max(
                self.inner
                    .recurrent_state_bytes()
                    .saturating_add(self.inner.scheduled_draft_state_held_bytes(tokens)),
            );
        }
    }

    fn try_resume_preempted(&mut self) {
        let ledger = self.judge_policy() == JudgePolicy::Ledger;
        let Some(mut turn) = self.scheduler.take_preempted() else {
            if ledger {
                self.memory_governor().end_drain(self.inner.memory_model());
            }
            return;
        };
        if turn
            .cancelled
            .as_ref()
            .is_some_and(|cancelled| cancelled.load(Ordering::Relaxed))
        {
            if stop_vacates(&mut turn) {
                self.finish_completed(turn);
            } else {
                self.fail_preempted(turn, engine::session::CHAT_SESSION_CANCELLED.to_string());
            }
            return;
        }
        let cache_snapshot = match self.inner.scheduler_cache_snapshot() {
            Ok(snapshot) => snapshot,
            Err(error) => {
                self.fail_preempted(turn, error.reason.clone());
                return;
            }
        };
        let mut cell = None;
        if ledger {
            match self.resume_cell(&mut turn) {
                Admitted::Granted(reservation) => cell = reservation,
                Admitted::Wait => {
                    self.scheduler.prepend_preempted(turn);
                    return;
                }
                Admitted::Fail(error) => {
                    self.fail_preempted(turn, error.reason.clone());
                    return;
                }
            }
        } else if !self.ensure_recurrent_slot(turn.seq_id) {
            self.scheduler.prepend_preempted(turn);
            return;
        }
        if !ledger && let Some(snapshot) = cache_snapshot {
            let materialized = self.inner.scheduler_materialized_blocks(turn.seq_id);
            let charge = if materialized > 0 {
                turn.block_reservation_total.saturating_sub(materialized)
            } else {
                turn.block_reservation_total
            };
            let state_candidate_bytes = self.inner.scheduled_state_candidate_bytes(
                turn.seq_id,
                turn.payload.requested_tokens,
                turn.payload.scheduled_speculation.is_some(),
            );
            let may_wait = if self.inner.scheduler_books_on_demand() {
                self.scheduler.resume_may_wait(&turn)
            } else {
                self.scheduler.has_live_turns()
            };
            let reserved = self.try_reserve_with_policy(
                turn.seq_id,
                charge,
                turn.recurrent_state_bytes,
                state_candidate_bytes,
                snapshot,
                may_wait,
                B::release_scheduled_cache,
            );
            match reserved {
                Ok(MemoryReserveOutcome::Admitted) => {
                    turn.scheduled_state_bytes = state_candidate_bytes
                        .saturating_add(self.inner.scheduled_state_held_bytes(turn.seq_id));
                }
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
        let scope = turn
            .payload
            .claim
            .as_ref()
            .map(|claim| Scope::enter_lane(claim.claim()));
        let prefix_admission = self.inner.prepare_scheduled_prefix(
            turn.seq_id,
            &replay_tokens,
            self.owner_states
                .get(&turn.payload.owner_id)
                .map(B::owner_tokens)
                .unwrap_or_default(),
            true,
            turn.payload.params.cache_salt,
            turn.block_size,
        );
        #[cfg(test)]
        let prefix_admission = match self.prefix_failure.take() {
            Some(reason) => Err(Error::from_reason(reason)),
            None => prefix_admission,
        };
        let prefix_admission = match prefix_admission {
            Ok(prefix) => prefix,
            Err(error) => {
                drop(scope);
                let _ = self.inner.release_scheduled_cache(turn.seq_id);
                self.inner.release_scheduled_recurrent_for(turn.seq_id);
                self.inner.release_scheduled_speculation(turn.seq_id);
                if let Some(oom) = self.prefix_gpu_oom(&error) {
                    self.fail_preempted(turn, oom);
                    self.recover_if_gpu_oom();
                    return;
                }
                let seq_id = turn.seq_id;
                match self.prefix_refused(seq_id, &mut turn.resource_wait, true) {
                    Some(Err(failed)) => self.fail_preempted(turn, failed.reason.clone()),
                    Some(Ok(())) => self.scheduler.prepend_preempted(turn),
                    None if is_paged_allocation_blocked(&error.reason) => {
                        self.scheduler.prepend_preempted(turn);
                    }
                    None => self.fail_preempted(turn, error.reason.clone()),
                }
                return;
            }
        };
        drop(scope);
        let (prefix, restore) = match prefix_admission {
            ScheduledPrefixAdmission::Ready(prefix) => (prefix, None),
            ScheduledPrefixAdmission::Waiting {
                provisional,
                restore,
            } => (provisional, Some(restore)),
        };
        if ledger {
            let refused = match self.claim_resume_target(&mut turn, &replay_tokens) {
                Admitted::Granted(_) => None,
                Admitted::Wait => Some(None),
                Admitted::Fail(error) => Some(Some(error)),
            };
            if let Some(failure) = refused {
                let _ = self.inner.release_scheduled_cache(turn.seq_id);
                self.inner.release_scheduled_recurrent_for(turn.seq_id);
                self.inner.release_scheduled_speculation(turn.seq_id);
                match failure {
                    Some(error) => self.fail_preempted(turn, error.reason.clone()),
                    None => self.scheduler.prepend_preempted(turn),
                }
                return;
            }
            let governor = self.memory_governor();
            governor.lane_resumed(self.inner.memory_model());
            turn.resume_target = 0;
            turn.payload.cell = cell;
            self.admissions = self.admissions.saturating_add(1);
            turn.admission = self.admissions;
        }
        let materialized_blocks = self
            .inner
            .scheduler_materialized_blocks(turn.seq_id)
            .saturating_add(restore.as_ref().map_or(0, B::restore_reserved_blocks));
        turn.num_computed_tokens = prefix.effective_cached_prefix_len() as u32;
        turn.block_materialized_blocks = materialized_blocks;
        let extra_breaks = self
            .inner
            .extra_prefill_breaks(target, turn.num_computed_tokens);
        (turn.pinned_prefill_breaks, turn.whole_prompt_prefill) =
            prefill_breaks(&self.inner, turn.num_computed_tokens, target, extra_breaks);
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

    /// A step's refusals are its own: one left by a reservation outside the
    /// step (admission, a prefix prepare) must not reach its reclaim.
    fn begin_scheduled_step(&mut self) {
        if let Some(adapter) = self.inner.paged_adapter() {
            adapter.clear_refusals();
        }
    }

    fn finish_completed(&mut self, mut turn: TurnState<ScheduledTurn<B::PrefixState>>) {
        // What the turn's end publishes is its request's.
        let claim = turn.payload.claim.take();
        let _request = claim.as_ref().map(|claim| Scope::enter(claim.claim()));
        // Every arm below ends the turn, the error returns included.
        self.inner.scheduled_turn_closed(turn.seq_id);
        // Splash `Engine::release`: a lane, or a suspended request, gives its
        // memory back and the waits retry; only a lane ends the failure.
        if self.inner.reclaims_in_order() {
            let model = self.inner.memory_model();
            if turn.status == TurnStatus::Preempted {
                self.memory_governor().signal_progress(model);
            } else {
                self.memory_governor().lane_released(model);
            }
        }
        let Some(cancelled) = turn.cancelled.take() else {
            let owner_id = turn.payload.owner_id.clone();
            self.inner.abort_paged_turn();
            self.inner.release_scheduled_recurrent_for(turn.seq_id);
            self.inner.release_scheduled_speculation(turn.seq_id);
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
        let vacated = turn.payload.vacated_stop;
        if vacated {
            // Splash `vacateLane` → `finish(Stop)`: the stop completes on what
            // the turn generated and takes no lane back. Activating here
            // would claim a recurrent unit that peers may hold and fail the
            // turn; `abort_paged_turn` would tear down the active peer.
            let _ = self.inner.release_scheduled_cache(turn.seq_id);
            if let Some(error) = turn.payload.failure.take() {
                turn.payload.response.send_error(error, cancelled.as_ref());
                return;
            }
        } else {
            if let Some(owner_state) = self.owner_states.get(&turn.payload.owner_id) {
                self.inner.install_owner_state(turn.seq_id, owner_state);
            }
            if let Err(error) = self.inner.activate_paged_seq(turn.seq_id) {
                self.inner.release_scheduled_recurrent_for(turn.seq_id);
                self.inner.release_scheduled_speculation(turn.seq_id);
                self.owner_states.remove(&turn.payload.owner_id);
                self.owner_sequences.remove(&turn.payload.owner_id);
                turn.payload.response.send_error(error, cancelled.as_ref());
                return;
            }
            if let Some(error) = turn.payload.failure.take() {
                self.inner.abort_paged_turn();
                self.inner.release_scheduled_recurrent_for(turn.seq_id);
                self.inner.release_scheduled_speculation(turn.seq_id);
                self.inner.set_turn_cancel_flag(None);
                self.owner_states.remove(&turn.payload.owner_id);
                self.owner_sequences.remove(&turn.payload.owner_id);
                turn.payload.response.send_error(error, cancelled.as_ref());
                return;
            }
        }
        let sink = turn.payload.response.sink();
        let finish = if vacated {
            engine::paged_turn::emit_paged_turn_result::<B>
        } else {
            engine::paged_turn::finish_paged_turn::<B>
        };
        let outcome = finish(
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
        self.inner.release_scheduled_speculation(turn.seq_id);
        let parked = if outcome.is_ok() && turn.payload.reuse_cache && !vacated {
            self.owner_states.insert(
                turn.payload.owner_id.clone(),
                self.inner.capture_owner_state(turn.seq_id),
            );
            self.inner.park_active_scheduled_recurrent()
        } else {
            self.inner.release_scheduled_recurrent_for(turn.seq_id);
            self.inner.release_scheduled_speculation(turn.seq_id);
            Ok(())
        };
        self.inner.set_turn_cancel_flag(None);
        if outcome.is_err() || parked.is_err() || vacated {
            self.owner_states.remove(&turn.payload.owner_id);
            self.owner_sequences.remove(&turn.payload.owner_id);
        }
        if self.owner_sequences.get(&turn.payload.owner_id) == Some(&turn.seq_id)
            && let Some(adapter) = self.inner.paged_adapter()
        {
            let stamp = adapter
                .turn_end_stamp(turn.seq_id)
                .unwrap_or_else(|| adapter.recency().next());
            self.owner_last_used.insert(turn.seq_id, stamp);
        }
        let live: HashSet<SeqId> = self.owner_sequences.values().copied().collect();
        self.owner_last_used
            .retain(|seq_id, _| live.contains(seq_id));
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

    /// Splash `Engine::admitQueued` for one start: admitted, held for
    /// scheduling (`queue`), which holds back nothing, or refused, which
    /// holds back every start after it. Whether it was refused.
    fn start_prepared(&mut self, prepared: Box<PreparedTurn>) -> bool {
        match self.admit_prepared(prepared) {
            Ok(Some(turn)) => self.enqueue_turn_or_reject(turn),
            Ok(None) => {}
            Err(prepared) if prepared.held => self.held.push_back(prepared),
            Err(prepared) => {
                self.prepared_waiting = Some(prepared);
                return true;
            }
        }
        false
    }

    /// A command that must not pass the held starts: a later turn of a held
    /// owner, and anything that orders the turns after it (a reset, a
    /// barrier, the exclusive lane), which would otherwise run before them.
    fn waits_behind_held(&self, command: &Queued<B::FamilyCommand>) -> bool {
        if self.held.is_empty() {
            return false;
        }
        let Some(chat) = command.as_chat() else {
            return true;
        };
        match chat {
            ChatCmd::ReleaseCacheOwner { .. } => false,
            ChatCmd::ResetCaches { .. } => true,
            _ => {
                let owner = Self::chat_config(chat)
                    .and_then(|config| config.cache_owner_id.as_deref())
                    .unwrap_or_default();
                Self::force_serial()
                    || !self.enabled
                    || !self.inner.scheduler_cache_available()
                    || !(Self::chat_has_explicit_owner(chat)
                        || self.ownerless_speculation_is_scheduled(chat))
                    || self.chat_requires_barrier(chat)
                    || self.held.iter().any(|prepared| prepared.owner_id == owner)
            }
        }
    }

    /// Splash `Scheduler::complete`: a prefill unit that served a later
    /// submission overtakes each held start, up to overdue.
    fn count_held_overtakes(&mut self) {
        let Some(youngest) = self.scheduler.take_youngest_served() else {
            return;
        };
        for prepared in &mut self.held {
            if prepared.order.is_some_and(|order| order < youngest)
                && prepared.overtaken < engine::scheduler::MAX_OVERTAKES
            {
                prepared.overtaken += 1;
            }
        }
    }

    /// Splash `Engine::admitQueued` over the held starts, in order. Each
    /// start stays in `self.held` until it is tried: a start's GPU OOM
    /// restarts the model, which must fail the starts not yet tried.
    /// Whether one was refused.
    fn retry_held(&mut self, mut wired: Option<&mut crate::stream::WiredUse>) -> bool {
        let mut deferred = false;
        for _ in 0..self.held.len() {
            let Some(prepared) = self.held.pop_front() else {
                break;
            };
            if deferred
                || self.scheduler.waiting_len() + self.scheduler.running_len()
                    >= self.scheduler.max_num_seqs()
            {
                self.held.push_back(prepared);
                continue;
            }
            if let Some(wired) = wired.as_mut() {
                wired.mark();
            }
            deferred = self.start_prepared(prepared);
        }
        deferred
    }

    pub(crate) fn drive(
        &mut self,
        receiver: &mut tokio::sync::mpsc::UnboundedReceiver<ModelCommand<B::FamilyCommand>>,
    ) -> LoopControl {
        let mut woke = false;
        if !self.scheduler.has_work()
            && self.pending.is_empty()
            && self.prepared_waiting.is_none()
            && self.held.is_empty()
        {
            match receiver.blocking_recv() {
                Some(command) => woke |= self.receive(command),
                None => return LoopControl::Break,
            }
        }
        while let Ok(command) = receiver.try_recv() {
            woke |= self.receive(command);
        }
        self.pass_work = PassWork::default();
        self.pass_start = self.memory_governor().now();
        let mut wired = self.inner.wired_residency().map(|r| r.begin());
        self.reap_cancelled_waiters();
        self.poll_restores();
        self.tick_waits();
        self.refresh_wanted_chains();
        self.run_control_pass(woke);

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
            if let Some(wired) = wired.as_mut() {
                wired.mark();
            }
            deferred = self.start_prepared(prepared);
        }
        // Splash `Engine::admitQueued`'s recovery tries no start of a
        // suspended lane's priority: a held start's earlier order would
        // otherwise pass the suspended lane in `admit_waiting`.
        if !deferred && !self.scheduler.has_pending_control() && !self.scheduler.has_preempted() {
            deferred = self.retry_held(wired.as_mut());
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
            if must_wait_for_legacy_owner || self.waits_behind_held(command) {
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
            let Some(Queued { command, arrival }) = self.pending.pop_front() else {
                break;
            };
            if (Self::force_serial() || !self.enabled)
                && !command
                    .as_chat()
                    .is_some_and(|chat| matches!(chat, ChatCmd::ReleaseCacheOwner { .. }))
            {
                self.scheduler
                    .enqueue_exclusive(Queued { command, arrival });
                break;
            }
            match command.into_chat() {
                Ok(ChatCmd::ReleaseCacheOwner { owner_id, reply }) => {
                    if self.cache_owner_release_blocked(&owner_id) {
                        self.pending.push_front(Queued {
                            command: ModelCommand::from_chat(ChatCmd::ReleaseCacheOwner {
                                owner_id,
                                reply,
                            }),
                            arrival,
                        });
                        break;
                    }
                    let _ = reply.send(self.release_cache_owner_now(&owner_id));
                }
                Ok(chat) => {
                    let is_reset = matches!(chat, ChatCmd::ResetCaches { .. });
                    let mut exclusive = false;
                    if !self.inner.scheduler_cache_available()
                        || !(Self::chat_has_explicit_owner(&chat)
                            || self.ownerless_speculation_is_scheduled(&chat))
                        || self.chat_requires_barrier(&chat)
                    {
                        self.scheduler.enqueue_exclusive(Queued {
                            command: ModelCommand::from_chat(chat),
                            arrival,
                        });
                        exclusive = true;
                    } else if let Some(prepared) = {
                        if let Some(wired) = wired.as_mut() {
                            wired.mark();
                        }
                        self.prepare_chat(chat, arrival)
                    } {
                        deferred = self.start_prepared(prepared);
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

        self.begin_scheduled_step();
        let action = self.step();
        self.count_held_overtakes();
        let scheduler_idle = matches!(&action, Ok(SchedulerAction::Idle));
        let mut may_resume_preempted = true;
        match action {
            Ok(SchedulerAction::Idle) => self.close_pass(&mut []),
            Ok(SchedulerAction::Exclusive(Queued { command, arrival })) => {
                if let Some(control) = self.run_command(command, Some(arrival), wired.as_mut()) {
                    return control;
                }
            }
            Ok(SchedulerAction::Barrier(command)) => {
                if let Some(control) = self.run_command(command, None, wired.as_mut()) {
                    return control;
                }
            }
            Ok(mut step @ SchedulerAction::Stepped { .. }) => {
                if let Some(wired) = wired.as_mut() {
                    wired.used();
                }
                self.close_step_pass(&mut step);
                may_resume_preempted = self.settle_step(step);
            }
            Err(SchedulerError::Executor(error)) => return self.executor_failed(error),
            Err(SchedulerError::InvalidResult(message)) => {
                tracing::error!(
                    "{} scheduler returned an invalid result: {message}",
                    B::SCHEDULER_NAME
                );
                return LoopControl::Break;
            }
        }
        self.recover_if_gpu_oom();
        if may_resume_preempted {
            self.refresh_wanted_chains();
            self.try_resume_preempted();
        }
        self.publish_gauges();
        if scheduler_idle
            && (self.scheduler.has_work()
                || self.prepared_waiting.is_some()
                || !self.held.is_empty())
        {
            std::thread::sleep(Duration::from_millis(1));
        }
        LoopControl::Continue
    }

    /// The pass of a step that ran: its prefill phase is time the lanes
    /// outside it waited (Splash's batch completion).
    fn close_step_pass(&mut self, step: &mut Action<B>) {
        if let SchedulerAction::Stepped {
            completed,
            prefill_wall_micros,
            ..
        } = step
        {
            self.pass_work.prefill_seconds = *prefill_wall_micros as f64 / 1e6;
            self.close_pass(completed);
        }
    }

    /// The status gauges after a pass (Splash `Scheduler::snapshot`): the
    /// scheduler's turns by phase, the chats not yet prepared as queued,
    /// and the prepared start as waiting for resources once refused a lane
    /// or memory (`waitForResources`), else queued (`deferAdmission`).
    fn publish_gauges(&self) {
        let mut queued = self
            .pending
            .iter()
            .filter(|queued| {
                queued
                    .as_chat()
                    .is_some_and(|chat| !matches!(chat, ChatCmd::ReleaseCacheOwner { .. }))
            })
            .count() as u64;
        queued += self.held.len() as u64;
        let mut waiting_resources = 0;
        if let Some(prepared) = self.prepared_waiting.as_ref() {
            if prepared.resource_wait.reason == WaitReason::None {
                queued += 1;
            } else {
                waiting_resources += 1;
            }
        }
        self.scheduler.publish_gauges(queued, waiting_resources);
    }
}

#[cfg(test)]
pub(crate) mod tests {
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
            rope_scaling: None,
            qwen35_gguf_gdn_layout: None,
            kv_format: None,
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
            persist_paged_cache: None,
            n_mtp_layers: 0,
        }
    }

    fn tiny_moe_config() -> Qwen3_5MoeConfig {
        Qwen3_5MoeConfig {
            rope_scaling: None,
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
            persist_paged_cache: None,
            n_mtp_layers: 0,
            qwen35_gguf_gdn_layout: None,
            kv_format: None,
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
    fn prefill_slice_ends_walks_one_grid_plus_extra_breaks() {
        // Plain grid from 0, and the same grid anchored at a cached prefix.
        assert_eq!(
            prefill_slice_ends(0, 2048, 512, []),
            [512, 1024, 1536, 2048]
        );
        assert_eq!(
            prefill_slice_ends(100, 2048, 512, []),
            [612, 1124, 1636, 2048]
        );
        // Extra breaks land strictly inside the range, merge with the grid,
        // and survive dedup: a rung coinciding with a grid point adds nothing.
        assert_eq!(
            prefill_slice_ends(0, 2048, 512, [256, 1024, 64, 4096, 100]),
            [64, 100, 256, 512, 1024, 1536, 2048]
        );
        // A degenerate or zero-length range still terminates at the target.
        assert_eq!(prefill_slice_ends(0, 7, 512, []), [7]);
        assert_eq!(prefill_slice_ends(128, 128, 512, [64]), Vec::<u32>::new());
        // A whole-prompt step: the saturating slice gives the one end.
        assert_eq!(prefill_slice_ends(0, 6219, u32::MAX, []), [6219]);
        assert_eq!(
            prefill_slice_ends(100, 6219, u32::MAX, [3000]),
            [3000, 6219]
        );
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
    fn shared_command_dispatch_orders_family_operations_stats_and_reset() {
        use crate::models::qwen3_5::model::Qwen35FamilyCommand;

        let mut inner = Qwen35Inner::new(tiny_config()).expect("construct tiny dense model");
        inner.cached_token_history = vec![1, 2, 3];
        let mut state = HybridSchedulerState::new(inner).expect("construct scheduler");
        state.owner_sequences.insert("owner".into(), 9);
        state.owner_states.insert("owner".into(), vec![7, 8]);
        let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
        let (force_reply, mut force_result) = tokio::sync::oneshot::channel();
        let (stats_reply, mut stats_result) = tokio::sync::oneshot::channel();
        let (before_reply, mut before_result) = tokio::sync::oneshot::channel();
        let (reset_reply, mut reset_result) = tokio::sync::oneshot::channel();
        let (after_reply, mut after_result) = tokio::sync::oneshot::channel();
        for command in [
            Qwen35FamilyCommand::ForcePagedGdnMismatchForTest { reply: force_reply }.into(),
            Qwen35Cmd::scheduler_stats(stats_reply),
            Qwen35FamilyCommand::MtpPagedGdnStateForTest {
                reply: before_reply,
            }
            .into(),
            Qwen35Cmd::from_chat(ChatCmd::ResetCaches { reply: reset_reply }),
            Qwen35FamilyCommand::MtpPagedGdnStateForTest { reply: after_reply }.into(),
        ] {
            assert!(sender.send(command).is_ok());
        }

        assert!(matches!(state.drive(&mut receiver), LoopControl::Continue));
        assert!(matches!(force_result.try_recv(), Ok(Ok(()))));
        assert!(matches!(
            stats_result.try_recv(),
            Err(tokio::sync::oneshot::error::TryRecvError::Empty)
        ));
        assert!(matches!(state.drive(&mut receiver), LoopControl::Continue));
        assert!(matches!(stats_result.try_recv(), Ok(Ok(_))));
        assert!(matches!(
            before_result.try_recv(),
            Err(tokio::sync::oneshot::error::TryRecvError::Empty)
        ));
        assert!(matches!(state.drive(&mut receiver), LoopControl::Continue));
        assert!(before_result.try_recv().unwrap().unwrap().history_len > 0);
        assert!(state.owner_states.contains_key("owner"));
        assert!(matches!(
            reset_result.try_recv(),
            Err(tokio::sync::oneshot::error::TryRecvError::Empty)
        ));
        assert!(matches!(state.drive(&mut receiver), LoopControl::Continue));
        assert!(matches!(reset_result.try_recv(), Ok(Ok(()))));
        assert!(state.owner_sequences.is_empty());
        assert!(state.owner_states.is_empty());
        assert!(matches!(
            after_result.try_recv(),
            Err(tokio::sync::oneshot::error::TryRecvError::Empty)
        ));
        assert!(matches!(state.drive(&mut receiver), LoopControl::Continue));
        assert_eq!(after_result.try_recv().unwrap().unwrap().history_len, 0);
    }

    /// A drive iteration that runs GPU work stamps the model's residency
    /// clock and closes its guard; a stats query stamps nothing (Splash
    /// `Residency::use` runs only at command submit).
    ///
    /// Mutation M4: `drive` takes no guard.
    #[test]
    fn drive_marks_the_wired_residency_for_gpu_work_only() {
        use crate::models::qwen3_5::model::Qwen35FamilyCommand;

        let clock = mlx_paged_attn::residency::Clock::new(600.0).unwrap();
        let mut inner = Qwen35Inner::new(tiny_config()).expect("construct tiny dense model");
        inner.wired = Some(WiredResidency::for_test(clock.clone()));
        let mut state = HybridSchedulerState::new(inner).expect("construct scheduler");
        let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
        let created = clock.last_use();
        let (stats_reply, mut stats_result) = tokio::sync::oneshot::channel();
        assert!(sender.send(Qwen35Cmd::scheduler_stats(stats_reply)).is_ok());
        assert!(matches!(state.drive(&mut receiver), LoopControl::Continue));
        assert!(matches!(stats_result.try_recv(), Ok(Ok(_))));
        assert_eq!(clock.last_use(), created);
        let (force_reply, mut force_result) = tokio::sync::oneshot::channel();
        assert!(
            sender
                .send(
                    Qwen35FamilyCommand::ForcePagedGdnMismatchForTest { reply: force_reply }.into()
                )
                .is_ok()
        );
        assert!(matches!(state.drive(&mut receiver), LoopControl::Continue));
        assert!(matches!(force_result.try_recv(), Ok(Ok(()))));
        assert!(clock.last_use() > created);
        assert_eq!(clock.open_count(), 0);
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
    fn moe_dflash_uses_exclusive_lane_while_disabled_draft_keeps_paged_ar() {
        if !unsafe { mlx_sys::mlx_metal_is_available() } {
            return;
        }
        let mut inner = crate::models::qwen3_5_moe::model::scheduled_mtp::seeded_inner_with_config(
            0xDFA5_3504,
            |config| config.n_mtp_layers = 0,
        );
        inner.dflash2 = Some(
            crate::models::qwen3_5::dflash2::tiny_dflash2_model_for_stepper_test(
                &inner.config.to_dense_config(),
            )
            .expect("tiny compatible draft"),
        );
        assert!(
            !inner.supports_scheduled_speculation(),
            "external draft is never native MTP"
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
            assert!(state.chat_requires_barrier(&command));
            let plan = TurnPlan::resolve(
                state.inner.execution_plan(),
                TurnRequest {
                    is_delta: false,
                    input_media: MediaCapabilities::NONE,
                    context_media: MediaCapabilities::NONE,
                    speculative_requested: true,
                    streaming,
                },
            );
            assert_eq!(
                plan.decoder,
                DecoderPlan::Speculative(engine::plan::SpeculativeKind::DraftModel)
            );
            assert!(plan.use_paged_attention);
        }
        let (command, _rx) = sync_start(ChatConfig {
            enable_mtp: Some(false),
            ..ChatConfig::default()
        });
        assert!(
            !state.chat_requires_barrier(&command),
            "plain AR remains schedulable"
        );
    }

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
    fn adaptive_barrier_requires_a_decoder_that_admits_the_turn_shape() {
        fn check<B: HybridSchedulerBackend>(inner: B, sync_speculates: bool) {
            let state = HybridSchedulerState::new(inner).expect("construct scheduler");
            for streaming in [false, true] {
                for temperature in [0.0, 0.7] {
                    let config = ChatConfig {
                        enable_mtp: Some(true),
                        mtp_adaptive_depth: Some(true),
                        temperature: Some(temperature),
                        ..ChatConfig::default()
                    };
                    let (command, _keepalive): (ChatCmd, Box<dyn std::any::Any>) = if streaming {
                        let (command, rx) = streaming_start(config);
                        (command, Box::new(rx))
                    } else {
                        let (command, rx) = sync_start(config);
                        (command, Box::new(rx))
                    };
                    let plan = TurnPlan::resolve(
                        state.inner.execution_plan(),
                        TurnRequest {
                            is_delta: false,
                            input_media: MediaCapabilities::NONE,
                            context_media: MediaCapabilities::NONE,
                            speculative_requested: true,
                            streaming,
                        },
                    );
                    let expected = sync_speculates && !streaming;
                    assert_eq!(
                        matches!(plan.decoder, DecoderPlan::Speculative(_)),
                        expected
                    );
                    assert_eq!(
                        state.chat_requires_barrier(&command),
                        expected,
                        "adaptive request routed incorrectly: streaming={streaming}, temperature={temperature}"
                    );
                }
            }
        }
        check(Qwen35Inner::new(tiny_config()).unwrap(), false);
        let mut nemotron = NemotronHInner::new(tiny_nemotron_paged_mtp_config()).unwrap();
        nemotron.mtp_weights_loaded = true;
        check(nemotron, true);
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
            .try_reserve_reclaiming_idle(2, reservation_blocks, rec, 0, snapshot)
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
            .try_reserve_reclaiming_idle(2, 10, 1, 0, snapshot)
            .expect("no snapshot refresh");
        assert!(
            matches!(outcome, MemoryReserveOutcome::Reject { total_blocks: 100 }),
            "idle scheduler must hard-error, not spin in prepared_waiting: {outcome:?}"
        );
    }

    /// Off-pool state admission: a candidate above the whole budget errors
    /// instead of deferring behind rows no reclaim can free enough of, and a
    /// queued turn's unmaterialized reservation counts like held state.
    /// MUTATION: compare the candidate with `>=` in `scheduled_state_fit`; the
    /// exact-budget row below reports `Never` and its assertion fails.
    /// MUTATION: drop `outstanding_bytes` from the sum; `(20, 21)` reports
    /// `Fits` and its assertion fails.
    #[test]
    fn scheduled_state_fit_orders_fits_reclaim_and_never() {
        assert_eq!(
            scheduled_state_fit(None, u64::MAX, u64::MAX),
            ScheduledStateFit::Fits
        );
        let capacity = Some(ScheduledStateCapacity {
            budget: 100,
            held: 60,
        });
        assert_eq!(
            scheduled_state_fit(capacity, 0, 40),
            ScheduledStateFit::Fits
        );
        assert_eq!(
            scheduled_state_fit(capacity, 0, 41),
            ScheduledStateFit::AfterReclaim
        );
        assert_eq!(
            scheduled_state_fit(capacity, 0, 100),
            ScheduledStateFit::AfterReclaim
        );
        assert_eq!(
            scheduled_state_fit(capacity, 0, 101),
            ScheduledStateFit::Never
        );
        assert_eq!(
            scheduled_state_fit(capacity, 20, 20),
            ScheduledStateFit::Fits
        );
        assert_eq!(
            scheduled_state_fit(capacity, 20, 21),
            ScheduledStateFit::AfterReclaim
        );
        assert_eq!(
            scheduled_state_fit(capacity, 20, 100),
            ScheduledStateFit::AfterReclaim
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
            .try_reserve_reclaiming_idle_with(2, 10, 1, 0, snapshot, |_inner, victim| {
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
            .try_reserve_reclaiming_idle(2, reservation_blocks, 0, 0, snapshot)
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
            .try_reserve_reclaiming_idle(2, charge, 0, 0, snapshot)
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
        tiny_paged_adapter_with(Arc::new(mlx_paged_attn::CacheRecency::new()), 4, 16)
    }

    /// Real adapter over an `initial`-block pool growing to `max` blocks of
    /// 16 tokens, its allocator on `recency`.
    fn tiny_paged_adapter_with(
        recency: Arc<mlx_paged_attn::CacheRecency>,
        initial: u32,
        max: u32,
    ) -> PagedKVCacheAdapter {
        let cfg = mlx_paged_attn::PagedAttentionConfig {
            block_size: 16,
            num_kv_heads: 1,
            head_size: 128,
            num_layers: 1,
            gpu_memory_mb: 256,
            max_seq_len: Some(512),
            max_batch_size: Some(32),
        };
        let pool = mlx_paged_attn::LayerKVPool::new(
            cfg,
            initial,
            max,
            mlx_paged_attn::metal::MetalDtype::BFloat16,
        )
        .expect("growable test LayerKVPool");
        let allocator = Arc::new(std::sync::Mutex::new(
            mlx_paged_attn::BlockAllocator::with_recency(initial, max, 16, recency),
        ));
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
            .try_reserve_reclaiming_idle(2, charge, 0, 0, snapshot)
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
            .try_reserve_reclaiming_idle(2, reservation_blocks, candidate_state_bytes, 0, snapshot)
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
            .try_reserve_reclaiming_idle(2, charge, 0, 0, snapshot)
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
        // A 7-block ceiling caps the 4 -> 8 doubling target at 7: above the
        // shortfall (current 4 + needed 1 = 5), below the target. The
        // re-probe of 7 fits, so the grow must land at 7, not decline.
        state
            .inner
            .paged_adapter_mut()
            .expect("paged adapter")
            .set_grow_headroom_probe_override(Some(Arc::new(|requested| Ok(requested.min(7)))));
        let charge = snapshot.blocks.free_blocks + 1;
        let outcome = state
            .try_reserve_reclaiming_idle(2, charge, 0, 0, snapshot)
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
            .try_reserve_reclaiming_idle(2, 17, 0, 0, snapshot)
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

    /// Qwen3.5 dense scheduler over the growable fixture (4 live blocks, 16
    /// max). Returns `None` to skip when Metal is unavailable.
    fn growable_qwen35_state() -> Option<(HybridSchedulerState<Qwen35Inner>, SchedulerCacheSnapshot)>
    {
        let mut inner = match Qwen35Inner::new(tiny_config()) {
            Ok(inner) => inner,
            Err(_) => {
                eprintln!("skipping Qwen3.5 grow-path scheduler test: inner failed");
                return None;
            }
        };
        if !crate::models::qwen3_5::dflash2_decode::tests::metal_is_default_device() {
            eprintln!("skipping Qwen3.5 grow-path scheduler test: Metal unavailable");
            return None;
        }
        inner.paged_adapter = Some(tiny_paged_adapter_with(
            Arc::clone(&inner.cache_recency),
            4,
            16,
        ));
        let state = HybridSchedulerState::new(inner).expect("construct Qwen3.5 scheduler");
        let snapshot = state
            .inner
            .scheduler_cache_snapshot()
            .expect("scheduler snapshot")
            .expect("paged adapter present");
        Some((state, snapshot))
    }

    /// An omitted budget on the M3 Pro pool window (78,464) books its 1,000
    /// prompt tokens plus one block, not the whole window.
    /// Mutation: `scheduled_booking` ignores `on_demand` → red (4904 ≠ 64).
    #[test]
    fn on_demand_booking_charges_the_prompt_not_the_window() {
        let window = 78_464;
        let (blocks, booked) = scheduled_booking(true, 1_000, window - 1_000, 16);
        assert_eq!((blocks, booked), (64, 1_024));
        assert_eq!(
            scheduled_booking(false, 1_000, window - 1_000, 16),
            (4_904, 78_463)
        );
        assert_eq!(scheduled_booking(true, 1_000, 0, 16), (63, 1_000));
        assert_eq!(scheduled_booking(true, 1_000, 1, 16), (63, 1_000));
    }

    /// Mutation: the dense (or MoE) override returns `false` → red.
    #[test]
    fn qwen35_families_clamp_scheduled_output_to_the_rest_of_the_window() {
        let dense = Qwen35Inner::new(tiny_config()).expect("construct tiny dense model");
        let moe = Qwen35MoeInner::new(tiny_moe_config()).expect("construct tiny MoE model");
        assert!(dense.scheduler_output_budget_is_rest_of_window());
        assert!(moe.scheduler_output_budget_is_rest_of_window());
        if let Ok(nemotron) = NemotronHInner::new(tiny_nemotron_paged_config()) {
            assert!(!nemotron.scheduler_output_budget_is_rest_of_window());
        }
    }

    /// The window an AR scheduled turn of `state` clamps on.
    fn scheduled_ar_window<B: HybridSchedulerBackend>(state: &HybridSchedulerState<B>) -> u32 {
        let block_size = state.inner.scheduler_block_size().expect("block size");
        scheduled_turn_context(
            u32::try_from(state.inner.max_position_embeddings()).expect("trained window"),
            state.scheduled_turn_window(block_size, false),
            scheduler_per_seq_context_override(),
        )
    }

    /// Admission's budget for a Qwen3.5 scheduled turn is the rest of the
    /// window and a prompt that fills it is refused; a family without that
    /// rule keeps `window - prompt + 1`.
    /// Mutation: pass `false` for the backend's rule in
    /// `scheduled_output_budget` → red (window - 48 + 1 ≠ window - 48).
    #[test]
    fn scheduled_admission_budget_follows_the_family_rule() {
        let Some((state, _)) = growable_qwen35_state() else {
            return;
        };
        let block_size = state.inner.scheduler_block_size().expect("block size");
        let window = scheduled_ar_window(&state);
        assert!(window > 48);
        assert_eq!(
            state.scheduled_output_budget(48, u32::MAX, block_size, false),
            Ok(window - 48)
        );
        let full = state
            .scheduled_output_budget(window, 1, block_size, false)
            .unwrap_err();
        assert!(full.starts_with("context_length_exceeded:"), "{full}");

        let Ok(nemotron) = NemotronHInner::new(tiny_nemotron_paged_config()) else {
            return;
        };
        if nemotron.paged_adapter.is_none() {
            return;
        }
        let state = HybridSchedulerState::new(nemotron).expect("construct nemotron scheduler");
        let block_size = state.inner.scheduler_block_size().expect("block size");
        let window = scheduled_ar_window(&state);
        assert_eq!(
            state.scheduled_output_budget(48, u32::MAX, block_size, false),
            Ok(window - 48 + 1)
        );
        assert_eq!(
            state.scheduled_output_budget(window, 1, block_size, false),
            Ok(1)
        );
    }

    /// Mutation: the dense override returns `false` → red.
    #[test]
    fn qwen35_families_book_kv_on_demand() {
        let dense = Qwen35Inner::new(tiny_config()).expect("construct tiny dense model");
        let moe = Qwen35MoeInner::new(tiny_moe_config()).expect("construct tiny MoE model");
        assert!(dense.scheduler_books_on_demand());
        assert!(moe.scheduler_books_on_demand());
        if let Ok(nemotron) = NemotronHInner::new(tiny_nemotron_paged_config()) {
            assert!(!nemotron.scheduler_books_on_demand());
        }
    }

    /// With live turns an on-demand reservation grows the runway pool before
    /// it defers; a whole-turn family still defers.
    /// Mutation: drop the live-turn grow → red (Defer). Mutation: drop the
    /// `scheduler_books_on_demand` gate on it → the Nemotron twin is red.
    #[test]
    fn live_turn_reservation_grows_the_runway_pool_before_deferring() {
        let Some((mut state, snapshot)) = growable_qwen35_state() else {
            return;
        };
        state
            .inner
            .paged_adapter_mut()
            .expect("paged adapter")
            .set_grow_headroom_probe_override(Some(Arc::new(Ok)));
        let charge = snapshot.blocks.free_blocks + 1;
        let outcome = state
            .try_reserve_with_policy(
                2,
                charge,
                0,
                0,
                snapshot,
                true,
                <Qwen35Inner as HybridSchedulerBackend>::release_scheduled_cache,
            )
            .expect("grow + re-reserve");
        assert!(
            matches!(outcome, MemoryReserveOutcome::Admitted),
            "an on-demand reservation must grow the runway beside live turns: {outcome:?}"
        );
        assert_eq!(
            state
                .inner
                .scheduler_cache_snapshot()
                .expect("snapshot")
                .expect("adapter")
                .blocks
                .total_blocks,
            8
        );

        let Some((mut state, snapshot)) = growable_scheduler_state() else {
            return;
        };
        state
            .inner
            .paged_adapter_mut()
            .expect("paged adapter")
            .set_grow_headroom_probe_override(Some(Arc::new(Ok)));
        let outcome = state
            .try_reserve_with_policy(
                2,
                snapshot.blocks.free_blocks + 1,
                0,
                0,
                snapshot,
                true,
                <NemotronHInner as HybridSchedulerBackend>::release_scheduled_cache,
            )
            .expect("defer");
        assert!(
            matches!(outcome, MemoryReserveOutcome::Defer),
            "a whole-turn family must keep deferring beside live turns: {outcome:?}"
        );
    }

    /// Reserves one block more than is free for turn 2 while turn 1 is live
    /// in the scheduler and holds pool blocks. Returns the outcome, the pool
    /// size after it, and whether turn 1's block table is unchanged.
    #[cfg(target_os = "macos")]
    fn reserve_beside_a_live_turn<B: HybridSchedulerBackend>(
        mut state: HybridSchedulerState<B>,
    ) -> (MemoryReserveOutcome, u32, bool) {
        state
            .inner
            .paged_adapter_mut()
            .expect("paged adapter")
            .set_grow_headroom_probe_override(Some(Arc::new(Ok)));
        let prompt: Vec<u32> = (0..40).map(|token| token % 16).collect();
        let live = speculative::tests::request(
            &mut state.inner,
            1,
            prompt,
            false,
            Arc::new(AtomicBool::new(false)),
        );
        let materialized = state.inner.scheduler_materialized_blocks(1);
        assert!(materialized > 0, "the live turn holds pool blocks");
        let live_blocks = |state: &HybridSchedulerState<B>| {
            state
                .inner
                .paged_adapter()
                .and_then(|adapter| adapter.block_table_for(1))
                .map(|table| table.block_ids())
        };
        let before = live_blocks(&state);
        state
            .scheduler
            .enqueue_turn(live.with_block_reservation(materialized + 1, materialized, 16))
            .expect("enqueue the live turn");
        assert!(state.scheduler.has_live_turns());
        let snapshot = state
            .inner
            .scheduler_cache_snapshot()
            .expect("snapshot")
            .expect("adapter");
        assert_eq!(snapshot.blocks.total_blocks, 4);
        let outcome = state
            .try_reserve_reclaiming_idle(2, snapshot.blocks.free_blocks + 1, 0, 0, snapshot)
            .expect("reserve beside the live turn");
        let total = state
            .inner
            .scheduler_cache_snapshot()
            .expect("snapshot")
            .expect("adapter")
            .blocks
            .total_blocks;
        (outcome, total, live_blocks(&state) == before)
    }

    /// A second turn arrives while the first is live on the runway pool: the
    /// on-demand family grows the pool and admits it, with the watermark on
    /// and the live turn's blocks kept; a whole-turn family still defers.
    /// Mutation: drop the live-turn grow in `try_reserve_with_policy` → red
    /// (Defer). Mutation: drop its `scheduler_books_on_demand` gate → red
    /// (the Nemotron turn is admitted).
    #[cfg(target_os = "macos")]
    #[test]
    fn second_turn_admits_beside_a_live_turn_by_growing_the_runway_pool() {
        let Some((state, _)) = growable_qwen35_state() else {
            return;
        };
        let (outcome, total, kept) = reserve_beside_a_live_turn(state);
        assert!(
            matches!(outcome, MemoryReserveOutcome::Admitted),
            "the runway pool must grow for a turn beside a live one: {outcome:?}"
        );
        assert!(total > 4, "pool stayed at {total} blocks");
        assert!(kept, "the grow moved the live turn's blocks");

        let Some((state, _)) = growable_scheduler_state() else {
            return;
        };
        let (outcome, total, kept) = reserve_beside_a_live_turn(state);
        assert!(
            matches!(outcome, MemoryReserveOutcome::Defer),
            "a whole-turn family must defer beside a live turn: {outcome:?}"
        );
        assert_eq!(total, 4);
        assert!(kept);
    }

    #[cfg(target_os = "macos")]
    type GrowProbe = Arc<dyn Fn(u32) -> std::result::Result<u32, String> + Send + Sync>;

    /// Reserves one block more than is free for turn 2 while idle owner 1
    /// holds two pool blocks and no turn is live, the pool's grow probe
    /// answering `probe`. Returns the outcome and whether owner 1 still
    /// holds its block table.
    #[cfg(target_os = "macos")]
    fn reserve_beside_an_idle_owner<B: HybridSchedulerBackend>(
        state: &mut HybridSchedulerState<B>,
        probe: GrowProbe,
    ) -> (MemoryReserveOutcome, bool) {
        state.owner_sequences.insert("idle".into(), 1);
        let adapter = state.inner.paged_adapter_mut().expect("paged adapter");
        adapter.set_grow_headroom_probe_override(Some(probe));
        adapter.begin_request(1).expect("begin idle request");
        adapter
            .allocate_suffix_blocks_for(1, 32)
            .expect("idle owner holds two blocks");
        let snapshot = state
            .inner
            .scheduler_cache_snapshot()
            .expect("snapshot")
            .expect("adapter");
        assert!(!state.scheduler.has_live_turns());
        let outcome = state
            .try_reserve_reclaiming_idle(2, snapshot.blocks.free_blocks + 1, 0, 0, snapshot)
            .expect("reserve beside the idle owner");
        let kept = state
            .inner
            .paged_adapter()
            .and_then(|adapter| adapter.block_table_for(1))
            .is_some();
        (outcome, kept)
    }

    /// Below its max the runway pool grows for a new turn before an idle
    /// owner's cache is reclaimed, as Splash allocates a KV extent before it
    /// reclaims; a whole-turn family keeps reclaiming first.
    /// Mutation: drop the grow before `reclaim_one_idle_scheduled_with` in
    /// `try_reserve_with_policy` → red (the idle owner's table is released).
    /// Mutation: drop its `scheduler_books_on_demand` gate → red (the
    /// Nemotron owner keeps its table).
    #[cfg(target_os = "macos")]
    #[test]
    fn admission_grows_the_runway_pool_before_reclaiming_an_idle_owner() {
        let Some((mut state, _)) = growable_qwen35_state() else {
            return;
        };
        let (outcome, kept) = reserve_beside_an_idle_owner(&mut state, Arc::new(Ok));
        assert!(
            matches!(outcome, MemoryReserveOutcome::Admitted),
            "{outcome:?}"
        );
        assert!(kept, "the idle owner's cache was reclaimed before a grow");

        let Some((mut state, _)) = growable_scheduler_state() else {
            return;
        };
        let (outcome, kept) = reserve_beside_an_idle_owner(&mut state, Arc::new(Ok));
        assert!(
            matches!(outcome, MemoryReserveOutcome::Admitted),
            "{outcome:?}"
        );
        assert!(!kept, "a whole-turn family must reclaim before it grows");
    }

    /// A running MTP turn decoded to 40,000 tokens holds draft K/V far past
    /// its admission charge (its booked prompt rows), but only the few
    /// hidden rows a verify keeps; admission charges what it holds, so a
    /// one-block turn that fits beside the admission charge waits.
    /// Mutations: return 0 from `grown_draft_state_bytes` → red (Admitted);
    /// price it with `scheduled_draft_state_bytes` → red (40,000 hidden rows
    /// charged).
    #[cfg(target_os = "macos")]
    #[test]
    fn running_mtp_turn_charges_its_grown_draft_state() {
        let Some((mut state, live)) = growable_qwen35_state() else {
            return;
        };
        state.inner.config.n_mtp_layers = 1;
        let prompt: Vec<u32> = (0..40).map(|token| token % 16).collect();
        let recurrent = state.inner.recurrent_state_bytes();
        let admission = recurrent + state.inner.scheduled_draft_state_bytes(56);
        let mut turn = speculative::tests::request(
            &mut state.inner,
            1,
            prompt,
            true,
            Arc::new(AtomicBool::new(false)),
        )
        .with_block_reservation(0, 0, 16)
        .with_recurrent_state_reservation(admission);
        turn.num_tokens = 40_000;
        let tokens = 40_000 + turn.decode_draft_allowance;
        let hidden_row = state.inner.config.hidden_size as u64 * 8;
        let kept_hidden_rows = (crate::models::qwen3_5::scheduled_mtp::MAX_DRAFTS + 3) as u64;
        let held = recurrent + state.inner.scheduled_draft_state_bytes(tokens)
            - hidden_row * (u64::from(tokens) - kept_hidden_rows);
        state.scheduler.reinstate_running(turn);
        assert_eq!(state.grown_draft_state_bytes(), held - admission);
        let bytes_per_block = live.bytes_per_block;
        let existing = admission.max(state.inner.scheduled_recurrent_bytes());
        let blocks = ((existing + bytes_per_block) as f64 / (0.95 * bytes_per_block as f64)).ceil()
            as u32
            + 2;
        assert!(
            held - admission > u64::from(blocks) * bytes_per_block,
            "the grown draft state must not fit the pool"
        );
        let snapshot = SchedulerCacheSnapshot {
            blocks: BlockTelemetry {
                total_blocks: blocks,
                free_blocks: blocks,
                reclaimable_blocks: 0,
                allocated_blocks: 0,
            },
            max_total_blocks: blocks,
            bytes_per_block,
        };
        let outcome = state
            .try_reserve_with_policy(
                2,
                1,
                0,
                0,
                snapshot,
                true,
                <Qwen35Inner as HybridSchedulerBackend>::release_scheduled_cache,
            )
            .expect("admission decision");
        assert!(
            matches!(outcome, MemoryReserveOutcome::Defer),
            "{outcome:?}"
        );
    }

    /// Qwen3.5 scheduler with one MTP layer, a pool of `initial` (up to
    /// `max`) 8,192 B blocks, and the admission charge of a turn booked at
    /// 56 rows (435,968 B). The pool grow is declined unless
    /// `set_mtp_step_probe`.
    #[cfg(target_os = "macos")]
    pub(in crate::engine::hybrid_scheduler) fn mtp_step_state_with(
        initial: u32,
        max: u32,
    ) -> Option<(HybridSchedulerState<Qwen35Inner>, u64)> {
        let (mut state, _) = growable_qwen35_state()?;
        state.inner.config.n_mtp_layers = 1;
        let cfg = mlx_paged_attn::PagedAttentionConfig {
            block_size: 16,
            num_kv_heads: 1,
            head_size: 128,
            num_layers: 1,
            gpu_memory_mb: 256,
            max_seq_len: Some(8_192),
            max_batch_size: Some(32),
        };
        let pool = mlx_paged_attn::LayerKVPool::new(
            cfg,
            initial,
            max,
            mlx_paged_attn::metal::MetalDtype::BFloat16,
        )
        .expect("MTP step LayerKVPool");
        let allocator = Arc::new(std::sync::Mutex::new(mlx_paged_attn::BlockAllocator::new(
            initial, max, 16,
        )));
        state.inner.paged_adapter = Some(
            PagedKVCacheAdapter::new(allocator, Arc::new(pool), 16).expect("MTP step adapter"),
        );
        let admission =
            state.inner.recurrent_state_bytes() + state.inner.scheduled_draft_state_bytes(56);
        assert_eq!(admission, 435_968);
        let snapshot = state
            .inner
            .scheduler_cache_snapshot()
            .expect("scheduler snapshot")
            .expect("paged adapter present");
        assert_eq!(snapshot.blocks.total_blocks, initial);
        assert_eq!(snapshot.bytes_per_block, 8_192);
        set_mtp_step_probe(&mut state, false);
        Some((state, admission))
    }

    #[cfg(target_os = "macos")]
    fn set_mtp_step_probe(state: &mut HybridSchedulerState<Qwen35Inner>, grows: bool) {
        let probe: Arc<dyn Fn(u32) -> std::result::Result<u32, String> + Send + Sync> = if grows {
            Arc::new(Ok)
        } else {
            Arc::new(|_| Err("no headroom".to_string()))
        };
        state
            .inner
            .paged_adapter_mut()
            .expect("paged adapter")
            .set_grow_headroom_probe_override(Some(probe));
    }

    /// Every planned row blocks, as a step whose KV or draft state did not fit.
    #[cfg(target_os = "macos")]
    struct BlockEveryRow;

    #[cfg(target_os = "macos")]
    impl<P> StepExecutor<P> for BlockEveryRow {
        type Error = Error;

        fn execute(
            &mut self,
            plan: &StepPlan,
            _running: &mut [TurnState<P>],
        ) -> std::result::Result<StepResult, Self::Error> {
            Ok(StepResult {
                rows: plan
                    .rows
                    .iter()
                    .map(HybridStepExecutor::<Qwen35Inner>::blocked)
                    .collect(),
                executed_decode_batch: 0,
                executed_greedy_epilogue_batch: 0,
                rows_alloc_evicted: 0,
                prefill_wall_micros: 0,
                decode_wall_micros: 0,
            })
        }
    }

    /// Admits `turn`, runs one step in which it blocks, and settles it.
    #[cfg(target_os = "macos")]
    fn settle_blocked_step<B: HybridSchedulerBackend>(
        state: &mut HybridSchedulerState<B>,
        turn: TurnState<ScheduledTurn<B::PrefixState>>,
    ) {
        state.scheduler.enqueue_turn(turn).expect("enqueue");
        let step = state.step_with(&mut BlockEveryRow).expect("a step");
        assert!(matches!(step, SchedulerAction::Stepped { .. }));
        assert!(!state.settle_step(step));
    }

    /// Every planned row runs one decode or prefill slice.
    #[cfg(target_os = "macos")]
    struct RunEveryRow;

    #[cfg(target_os = "macos")]
    impl<P> StepExecutor<P> for RunEveryRow {
        type Error = Error;

        fn execute(
            &mut self,
            plan: &StepPlan,
            _running: &mut [TurnState<P>],
        ) -> std::result::Result<StepResult, Self::Error> {
            Ok(StepResult {
                rows: plan
                    .rows
                    .iter()
                    .map(|row| RowStepResult {
                        seq_id: row.seq_id,
                        num_computed_tokens: row.num_tokens,
                        generated_token: (row.kind == StepKind::Decode).then_some(7),
                        speculative: None,
                        finished: false,
                        allocation_blocked: false,
                        prefill_micros: 0,
                    })
                    .collect(),
                executed_decode_batch: plan
                    .rows
                    .iter()
                    .filter(|row| row.kind == StepKind::Decode)
                    .count(),
                executed_greedy_epilogue_batch: 0,
                rows_alloc_evicted: 0,
                prefill_wall_micros: 0,
                decode_wall_micros: 0,
            })
        }
    }

    /// A running turn decoding past its `16`-token prompt of `seq`'s own
    /// tokens, which holds one block.
    #[cfg(target_os = "macos")]
    fn decoding_turn(state: &mut HybridSchedulerState<Qwen35Inner>, seq: u32) -> Turn<Qwen35Inner> {
        let mut turn = speculative::tests::request(
            &mut state.inner,
            seq,
            (0..16).map(|token| seq * 100 + token).collect(),
            false,
            Arc::new(AtomicBool::new(false)),
        );
        turn.num_computed_tokens = 16;
        turn.token_history.push(3);
        turn.num_tokens = 17;
        turn
    }

    /// Splash `Engine::growLane` claims a decode row's write end: at a
    /// block boundary (31 rows written, the pending token in `num_tokens`)
    /// the row writes into the two blocks it holds, so with no budget left
    /// to grow the pool it is granted, not refused.
    /// Mutation DC1: the decode claim adds one token past `num_tokens` →
    /// red (a third block is claimed and refused).
    #[cfg(target_os = "macos")]
    #[test]
    fn a_decode_row_at_a_block_boundary_claims_only_its_rows() {
        use crate::engine::governor::governor::tests::{G, model};
        let Some((mut state, governor, io)) = ledger_state(16 * G, 0) else {
            return;
        };
        let mut adapter = tiny_paged_adapter_with(Arc::clone(&state.inner.cache_recency), 2, 8);
        adapter.set_grow_headroom_probe_override(Some(passing()));
        adapter.set_extent_growth(1);
        let pools = adapter.layer_kv_pool().total_bytes();
        io.with(|fake| fake.stats.pools = pools);
        let notified = Arc::clone(&io);
        adapter.set_pool_growth_notifier(Some(Arc::new(move |bytes| {
            notified.with(|fake| fake.stats.pools = bytes);
        })));
        state.inner.paged_adapter = Some(adapter);
        state.inner.memory_registration = None;
        state.inner.memory_registration =
            Some(governor.register_model(model(pools, 0)).expect("register"));
        state
            .inner
            .paged_adapter_mut()
            .expect("paged adapter")
            .set_admission(Arc::new(governor));
        let prompt: Vec<u32> = (0..31).map(|token| 100 + token).collect();
        let mut turn = speculative::tests::request(
            &mut state.inner,
            1,
            prompt.clone(),
            false,
            Arc::new(AtomicBool::new(false)),
        );
        state
            .inner
            .paged_adapter_mut()
            .expect("paged adapter")
            .record_tokens_for(1, &prompt)
            .expect("the prompt's rows");
        turn.num_computed_tokens = 31;
        turn.token_history.push(3);
        turn.num_tokens = 32;
        state.scheduler.reinstate_running(turn);
        assert_eq!(state.inner.scheduler_materialized_blocks(1), 2);
        let plan = StepPlan {
            global_step: 0,
            rows: vec![engine::scheduler::StepRow {
                seq_id: 1,
                kind: StepKind::Decode,
                token_start: 31,
                num_tokens: 1,
                cancel_snapshot: false,
            }],
            contended: false,
            leader_cut: None,
        };
        let claims = state.claim_rows_in_order(&plan);
        assert!(claims.grants[0].is_none(), "{:?}", claims.grants[0]);
        assert!(!state.row_work_ends.contains_key(&1));
        assert_eq!(state.inner.scheduler_materialized_blocks(1), 2);
    }

    /// Splash `Engine::commit` ends a lane at its stop or length token: a
    /// decode row whose pending token is EOS, or reaches `max_new_tokens`,
    /// at a block boundary (32 rows written) with no budget to grow the
    /// pool is granted without a claim, so its step finishes it.
    /// Mutation TB1: the terminal row claims its pending token's row → red
    /// (a third block is claimed and refused).
    #[cfg(target_os = "macos")]
    #[test]
    fn a_terminal_decode_row_claims_nothing() {
        use crate::engine::governor::governor::tests::{G, model};
        for at_length in [false, true] {
            let Some((mut state, governor, io)) = ledger_state(16 * G, 0) else {
                return;
            };
            let mut adapter = tiny_paged_adapter_with(Arc::clone(&state.inner.cache_recency), 2, 8);
            adapter.set_grow_headroom_probe_override(Some(passing()));
            adapter.set_extent_growth(1);
            let pools = adapter.layer_kv_pool().total_bytes();
            io.with(|fake| fake.stats.pools = pools);
            state.inner.paged_adapter = Some(adapter);
            state.inner.memory_registration = None;
            state.inner.memory_registration =
                Some(governor.register_model(model(pools, 0)).expect("register"));
            state
                .inner
                .paged_adapter_mut()
                .expect("paged adapter")
                .set_admission(Arc::new(governor));
            let prompt: Vec<u32> = (0..32).map(|token| 100 + token).collect();
            let mut turn = speculative::tests::request(
                &mut state.inner,
                1,
                prompt.clone(),
                false,
                Arc::new(AtomicBool::new(false)),
            );
            state
                .inner
                .paged_adapter_mut()
                .expect("paged adapter")
                .record_tokens_for(1, &prompt)
                .expect("the prompt's rows");
            turn.num_computed_tokens = 32;
            if at_length {
                turn.payload.params.max_new_tokens = 1;
                turn.token_history.push(3);
            } else {
                turn.token_history.push(turn.payload.eos_id);
            }
            turn.num_tokens = 33;
            state.scheduler.reinstate_running(turn);
            assert_eq!(state.inner.scheduler_materialized_blocks(1), 2);
            let plan = StepPlan {
                global_step: 0,
                rows: vec![engine::scheduler::StepRow {
                    seq_id: 1,
                    kind: StepKind::Decode,
                    token_start: 32,
                    num_tokens: 1,
                    cancel_snapshot: false,
                }],
                contended: false,
                leader_cut: None,
            };
            let claims = state.claim_rows_in_order(&plan);
            assert!(
                claims.grants[0].is_none(),
                "at_length={at_length}: {:?}",
                claims.grants[0]
            );
            assert!(!state.row_work_ends.contains_key(&1));
            assert_eq!(state.inner.scheduler_materialized_blocks(1), 2);
        }
    }

    /// Two decode rows (the live GDN unit cap) at the max pool with room for
    /// one: the row that claims first runs, the other is refused with its
    /// denial and replans at the next step, and no turn is preempted
    /// (Splash `Engine::prepare`: a refused lane leaves the plan).
    /// Mutation P2m: the hook refuses every row when one is refused → red
    /// (no row runs; replans 0).
    #[cfg(target_os = "macos")]
    #[test]
    fn one_refused_row_does_not_stall_the_wave() {
        let Some((mut state, _)) = growable_qwen35_state() else {
            return;
        };
        state.inner.paged_adapter = Some(tiny_paged_adapter_with(
            Arc::clone(&state.inner.cache_recency),
            3,
            3,
        ));
        for seq in 1..=2 {
            let turn = decoding_turn(&mut state, seq);
            state.scheduler.reinstate_running(turn);
        }
        let snapshot = state
            .inner
            .scheduler_cache_snapshot()
            .expect("snapshot")
            .expect("adapter");
        assert_eq!(
            snapshot.blocks.free_blocks + snapshot.blocks.reclaimable_blocks,
            1
        );
        let step = state.step_with(&mut RunEveryRow).expect("a step");
        let SchedulerAction::Stepped {
            blocked,
            progressed,
            ..
        } = &step
        else {
            panic!("expected a step");
        };
        assert!(*progressed);
        assert_eq!(blocked.len(), 1);
        assert_eq!(blocked[0].0, 2);
        assert!(blocked[0].1.blocks.is_some());
        assert_eq!(state.scheduler.stats().replans, 1);
        assert!(!state.settle_step(step));
        assert_eq!(state.scheduler.stats().preemptions, 0);
        assert_eq!(state.scheduler.running_len(), 2);
        let computed: Vec<u32> = state
            .scheduler
            .running()
            .iter()
            .map(|turn| turn.num_computed_tokens)
            .collect();
        assert_eq!(computed, vec![17, 16]);
    }

    /// A Qwen3.5 state registered on a governor of its own, with `lanes`
    /// running, each with its reply; lane 1 has prefilled 8 tokens, so a
    /// later lane yields first.
    #[cfg(target_os = "macos")]
    #[allow(clippy::type_complexity)]
    fn judged_state(
        lanes: &[u32],
    ) -> Option<(
        HybridSchedulerState<Qwen35Inner>,
        &'static Governor,
        Vec<tokio::sync::oneshot::Receiver<Result<ChatResult>>>,
    )> {
        use crate::engine::governor::governor::tests::{FakeIo, G, host, model, stats};
        let (mut state, _) = growable_qwen35_state()?;
        let io = FakeIo::new(
            stats(0),
            host(32 * G, crate::engine::governor::host::Pressure::Normal),
        );
        let governor: &'static Governor = Box::leak(Box::new(Governor::new(io)));
        state.inner.memory_registration =
            Some(governor.register_model(model(16 * G, 0)).expect("register"));
        let mut replies = Vec::new();
        for &seq in lanes {
            let mut turn = speculative::tests::request(
                &mut state.inner,
                seq,
                (0..16).map(|token| seq * 100 + token).collect(),
                false,
                Arc::new(AtomicBool::new(false)),
            );
            if seq == 1 {
                turn.num_computed_tokens = 8;
            }
            let (reply, result) = tokio::sync::oneshot::channel();
            turn.payload.response = ScheduledReply::Sync(reply);
            replies.push(result);
            state.scheduler.reinstate_running(turn);
        }
        Some((state, governor, replies))
    }

    fn kv_denial(failure: Failure) -> Denial {
        Denial {
            failure: Some(failure),
            pending: false,
            blocks: Some(BlockRefusal {
                additional: 3,
                available: 1,
            }),
        }
    }

    /// Splash `Engine::refuse` / `judge` when no row ran: a pending denial
    /// waits (every refused row deferred, nothing preempted); beside
    /// another lane the lane to yield first is suspended to the KV it
    /// holds; a lone lane the host refused is suspended to its own work
    /// end; a lone lane the limit refused fails `capacity_exhausted`.
    /// Mutation J1: a lone lane the host refused fails → red (the third
    /// case's lane fails instead of yielding).
    #[cfg(target_os = "macos")]
    #[test]
    fn judge_waits_yields_or_fails() {
        // Pending → Wait.
        let Some((mut state, _, _replies)) = judged_state(&[1, 2]) else {
            return;
        };
        let pending = Denial {
            pending: true,
            ..Denial::default()
        };
        let blocked = vec![(1, pending), (2, kv_denial(Failure::EngineBudget))];
        assert!(!state.settle_blocked(blocked, false, &[1, 2]));
        assert_eq!(state.scheduler.stats().preemptions, 0);
        assert_eq!(state.scheduler.running_len(), 2);
        for turn in state.scheduler.running() {
            assert!(
                turn.resource_wait.retry_at > 0.0,
                "{} deferred",
                turn.seq_id
            );
        }

        // Another resident → Yield the lane to yield first, to its KV.
        let Some((mut state, governor, _replies)) = judged_state(&[1, 2]) else {
            return;
        };
        assert_eq!(state.scheduler.lane_to_yield(), Some(2));
        let blocked = vec![(1, kv_denial(Failure::EngineBudget))];
        assert!(!state.settle_blocked(blocked, false, &[1, 2]));
        assert_eq!(state.scheduler.stats().preemptions, 1);
        assert_eq!(
            state
                .scheduler
                .running()
                .iter()
                .map(|turn| turn.seq_id)
                .collect::<Vec<_>>(),
            vec![1]
        );
        let yielded = state.scheduler.turn_mut(2).expect("suspended turn");
        assert_eq!(yielded.resume_target, 16);
        assert!(yielded.resource_wait.retry_at > 0.0);
        assert_eq!(governor.suspensions(7), 1);
        assert!(governor.draining_for_recovery(7, governor.now(), true));

        // Lone + HostPressure → Yield to its own work end.
        let Some((mut state, governor, _replies)) = judged_state(&[1]) else {
            return;
        };
        state.row_work_ends.insert(1, 33);
        let blocked = vec![(1, kv_denial(Failure::HostPressure))];
        assert!(!state.settle_blocked(blocked, false, &[1]));
        assert_eq!(state.scheduler.stats().preemptions, 1);
        assert_eq!(state.scheduler.running_len(), 0);
        assert_eq!(
            state
                .scheduler
                .turn_mut(1)
                .expect("suspended turn")
                .resume_target,
            33
        );
        assert_eq!(governor.suspensions(7), 1);

        // Lone + EngineBudget → Fail.
        let Some((mut state, _, mut replies)) = judged_state(&[1]) else {
            return;
        };
        let blocked = vec![(1, kv_denial(Failure::EngineBudget))];
        assert!(!state.settle_blocked(blocked, false, &[1]));
        let error = replies[0]
            .try_recv()
            .expect("the turn has ended")
            .expect_err("the turn fails");
        assert!(
            error.reason.starts_with(
                "capacity_exhausted: could not allocate KV target: engine memory budget exceeded \
                 (additional_pages=3, free_pages=1)"
            ),
            "{}",
            error.reason
        );
        assert_eq!(state.scheduler.stats().preemptions, 0);
        assert_eq!(state.live_turns(), 0);
    }

    /// Splash `Engine::prepare`: before a step that ran no row refuses its
    /// lanes, a refusal that is not KV and that nothing returns yields the
    /// lane to yield first when that lane is outside the plan and yields
    /// before the refused row, even beside a pending refusal.
    /// Mutation F8 (the exception only when a row ran) → red (the pending
    /// denial defers every row, nothing yields).
    #[cfg(target_os = "macos")]
    #[test]
    fn a_draft_refusal_yields_the_outside_lane_when_no_row_ran() {
        let Some((mut state, governor, _replies)) = judged_state(&[1, 2]) else {
            return;
        };
        // A third lane past the live GDN unit cap: lane 1's cell goes, as
        // nothing here runs a forward.
        state.inner.release_scheduled_recurrent_for(1);
        let mut third = speculative::tests::request(
            &mut state.inner,
            3,
            (0..16).map(|token| 300 + token).collect(),
            false,
            Arc::new(AtomicBool::new(false)),
        );
        third.num_computed_tokens = 4;
        state.scheduler.reinstate_running(third);
        assert_eq!(state.scheduler.lane_to_yield(), Some(2));
        let pending = Denial {
            pending: true,
            ..kv_denial(Failure::EngineBudget)
        };
        let draft = Denial {
            failure: Some(Failure::EngineBudget),
            pending: false,
            blocks: None,
        };
        let blocked = vec![(1, pending), (3, draft)];
        assert!(!state.settle_blocked(blocked, false, &[1, 3]));
        assert_eq!(state.scheduler.stats().preemptions, 1);
        assert_eq!(
            state
                .scheduler
                .running()
                .iter()
                .map(|turn| turn.seq_id)
                .collect::<Vec<_>>(),
            vec![1, 3]
        );
        assert_eq!(governor.suspensions(7), 1);
    }

    /// A cancelled lane whose rows the ledger refuses ends between steps
    /// (Splash `Engine::cancel`), its claim and memory with it.
    /// Mutation RC1 (only waiting turns are reaped) → red (the lane stays
    /// running, its reply open).
    #[cfg(target_os = "macos")]
    #[test]
    fn a_cancelled_refused_lane_ends_between_steps() {
        let Some((mut state, _governor, mut replies)) = judged_state(&[1, 2]) else {
            return;
        };
        let turn = state.scheduler.running_turn_mut(2).expect("lane 2");
        turn.resource_wait.retry_at = f64::MAX;
        turn.cancelled
            .as_ref()
            .expect("cancel flag")
            .store(true, Ordering::Relaxed);
        state.reap_cancelled_waiters();
        assert_eq!(
            state
                .scheduler
                .running()
                .iter()
                .map(|turn| turn.seq_id)
                .collect::<Vec<_>>(),
            vec![1]
        );
        let error = replies[1]
            .try_recv()
            .expect("the turn has ended")
            .expect_err("cancelled");
        assert_eq!(error.reason, engine::session::CHAT_SESSION_CANCELLED);
        assert!(replies[0].try_recv().is_err(), "lane 1 runs on");
    }

    /// A lane suspended before its first step was granted gives back its
    /// admission's GDN cell bytes: its resume reserves them again.
    /// Mutation CL1 (the suspended lane keeps its cell reservation) → red
    /// (the bytes stay reserved).
    #[cfg(target_os = "macos")]
    #[test]
    fn a_lane_vacated_before_its_first_step_drops_its_cell() {
        let Some((mut state, governor, _replies)) = judged_state(&[1, 2]) else {
            return;
        };
        let claim = governor.open(crate::engine::governor::governor::whole_turn_spec(9), 7);
        let cell = {
            let _scope = Scope::enter(claim.claim());
            governor.try_reserve(1 << 20).expect("the cell's bytes")
        };
        state
            .scheduler
            .running_turn_mut(2)
            .expect("lane 2")
            .payload
            .cell = Some(cell);
        assert_eq!(governor.reserved_bytes(), 1 << 20);
        let blocked = vec![(1, kv_denial(Failure::EngineBudget))];
        assert!(!state.settle_blocked(blocked, false, &[1, 2]));
        assert_eq!(state.scheduler.stats().preemptions, 1);
        assert_eq!(governor.reserved_bytes(), 0);
    }

    /// A row's work runs in its request's claim (Splash
    /// `Engine::stateClaim`), so what it publishes claims with the lane's
    /// request and rank, not the engine's.
    /// Mutation RQ1 (no row scope) → red.
    #[cfg(target_os = "macos")]
    #[test]
    fn a_rows_work_runs_in_its_requests_claim() {
        let Some((mut state, governor, _replies)) = judged_state(&[1]) else {
            return;
        };
        let claim = governor.open(crate::engine::governor::governor::whole_turn_spec(11), 7);
        let id = claim.id();
        let turn = state.scheduler.running_turn_mut(1).expect("lane 1");
        turn.payload.claim = Some(claim);
        let _row = HybridStepExecutor::<Qwen35Inner>::request_scope(turn);
        assert_eq!(governor.scope_claim(), Some(id));
        assert_eq!(governor.spec(id).map(|spec| spec.request_id), Some(11));
    }

    /// A Qwen3.5 scheduler registered on a governor of its own whose limit
    /// is `hard_budget`, its counters reading `pools` bytes of pools.
    #[cfg(target_os = "macos")]
    #[allow(clippy::type_complexity)]
    fn ledger_state(
        hard_budget: u64,
        pools: u64,
    ) -> Option<(
        HybridSchedulerState<Qwen35Inner>,
        &'static Governor,
        Arc<crate::engine::governor::governor::tests::FakeIo>,
    )> {
        use crate::engine::governor::governor::tests::{FakeIo, G, host, model, stats};
        let (mut state, _) = growable_qwen35_state()?;
        let io = FakeIo::new(
            stats(0),
            host(32 * G, crate::engine::governor::host::Pressure::Normal),
        );
        io.with(|fake| fake.stats.pools = pools);
        let governor: &'static Governor = Box::leak(Box::new(Governor::new(io.clone())));
        state.inner.memory_registration = Some(
            governor
                .register_model(model(hard_budget, 0))
                .expect("register"),
        );
        Some((state, governor, io))
    }

    /// A plain text turn of `prompt` for owner `seq_id`, prepared for
    /// admission, and its reply.
    #[cfg(target_os = "macos")]
    fn prepared_turn(
        state: &mut HybridSchedulerState<Qwen35Inner>,
        seq_id: SeqId,
        prompt: Vec<u32>,
    ) -> (
        Box<PreparedTurn>,
        tokio::sync::oneshot::Receiver<Result<ChatResult>>,
    ) {
        use crate::models::gemma4::dspark_decode::tests::{tiny_qwen_tokenizer, tiny_turn_config};
        let config = tiny_turn_config(None, 20);
        let params = state.inner.resolve_params(&config);
        let thinking = state.inner.thinking_setup(&config);
        let prompt_tokens = prompt.len() as u32;
        let owner_id = format!("owner-{seq_id}");
        state.owner_sequences.insert(owner_id.clone(), seq_id);
        let (reply, result) = tokio::sync::oneshot::channel();
        let prepared = PreparedTurn {
            admitted: engine::session::AdmittedPagedTurn {
                tokenizer: tiny_qwen_tokenizer(),
                eos_id: 999,
                close: engine::backend::ThinkEndResolution {
                    think_end_id: None,
                    think_end_str: None,
                    thinking_end: Arc::from(Vec::new()),
                    think_end_extra_ids: Vec::new(),
                },
                config,
                params,
                thinking,
                template_thinking_enabled: false,
                tokens: prompt,
                images: Vec::new(),
                audio: Vec::new(),
                plan: engine::plan::TurnPlan {
                    is_delta: false,
                    input_media: engine::plan::MediaCapabilities::NONE,
                    context_media: engine::plan::MediaCapabilities::NONE,
                    use_paged_attention: true,
                    decoder: engine::plan::DecoderPlan::Autoregressive,
                },
            },
            response: ScheduledReply::Sync(reply),
            cancelled: Arc::new(AtomicBool::new(false)),
            owner_id,
            seq_id,
            newly_assigned: false,
            reservation_blocks: prompt_tokens.div_ceil(16) + 1,
            block_size: 16,
            reservation_state_bytes: state.inner.recurrent_state_bytes(),
            requested_tokens: prompt_tokens + 19,
            claim: None,
            resource_wait: ResourceWait::default(),
            enqueued_at: Instant::now(),
            ttft: TtftClock::default(),
            held: false,
            order: None,
            overtaken: 0,
        };
        (Box::new(prepared), result)
    }

    /// Splash `Engine::admit`: a request alone claims its GDN cell in
    /// service, one beside a resident lane as a Start. With room below the
    /// cell and nothing to take, the cell's reservation is refused and the
    /// lone start fails `capacity_exhausted`.
    /// Mutation S1: the admission claim is always InService → red (the
    /// second turn's claim is InService). Mutation S2: the cell is not
    /// reserved (its arrays are allocated lazily) → red (the lone start is
    /// admitted).
    #[cfg(target_os = "macos")]
    #[test]
    fn admission_is_start_beside_a_lane() {
        use crate::engine::governor::governor::tests::G;
        let Some((mut state, governor, _io)) = ledger_state(16 * G, 0) else {
            return;
        };
        let cell = state.inner.recurrent_state_bytes();
        assert!(cell > 0);
        let class = |turn: &Turn<Qwen35Inner>| {
            governor
                .spec(turn.payload.claim.as_ref().expect("a claim").id())
                .expect("an open claim")
                .cls
        };
        let (prepared, _alone) = prepared_turn(&mut state, 1, tokens_of(1, 1));
        let turn = state
            .admit_prepared(prepared)
            .ok()
            .flatten()
            .expect("admitted alone");
        assert_eq!(class(&turn), ClaimClass::InService);
        assert!(turn.payload.cell.is_some(), "the cell is reserved");
        assert_eq!(turn.admission, 1);
        state.scheduler.reinstate_running(turn);
        let (prepared, _beside) = prepared_turn(&mut state, 2, tokens_of(2, 1));
        let turn = state
            .admit_prepared(prepared)
            .ok()
            .flatten()
            .expect("admitted beside a lane");
        assert_eq!(class(&turn), ClaimClass::Start);
        assert_eq!(turn.admission, 2);

        let Some((mut state, _, _io)) = ledger_state(cell - 1, 0) else {
            return;
        };
        let (prepared, mut result) = prepared_turn(&mut state, 1, tokens_of(1, 1));
        assert!(matches!(state.admit_prepared(prepared), Ok(None)));
        let error = result
            .try_recv()
            .expect("the turn has ended")
            .expect_err("the turn fails");
        assert!(
            error.reason.starts_with(
                "capacity_exhausted: could not allocate request state: engine memory budget \
                 exceeded"
            ),
            "{}",
            error.reason
        );
    }

    /// Splash `Engine::admit` claims a speculative lane's whole state at
    /// its start (`CompositeStateLayout::laneBytes`): an MTP turn's
    /// admission reserves its GDN cell and its draft state's fixed part, so
    /// with room one byte below both the lone start fails
    /// `capacity_exhausted`, and with room for both it is admitted.
    /// Mutation MB1: the admission reserves the GDN cell only → red (the
    /// start is admitted one byte short).
    #[cfg(target_os = "macos")]
    #[test]
    fn a_speculative_start_claims_its_draft_state() {
        use crate::engine::governor::governor::tests::G;
        let Some((state, _, _io)) = ledger_state(16 * G, 0) else {
            return;
        };
        let lane =
            state.inner.recurrent_state_bytes() + state.inner.scheduled_draft_state_held_bytes(0);
        assert!(state.inner.scheduled_draft_state_held_bytes(0) > 0);
        drop(state);
        for (budget, admitted) in [(lane - 1, false), (lane, true)] {
            let Some((mut state, _, _io)) = ledger_state(budget, 0) else {
                return;
            };
            let (mut prepared, mut result) = prepared_turn(&mut state, 1, tokens_of(1, 1));
            prepared.admitted.plan.decoder =
                engine::plan::DecoderPlan::Speculative(engine::plan::SpeculativeKind::NativeMtp);
            let turn = state.admit_prepared(prepared).ok().flatten();
            assert_eq!(turn.is_some(), admitted, "budget {budget}");
            if !admitted {
                let error = result
                    .try_recv()
                    .expect("the turn has ended")
                    .expect_err("the turn fails");
                assert!(
                    error
                        .reason
                        .starts_with("capacity_exhausted: could not allocate request state"),
                    "{}",
                    error.reason
                );
            }
        }
    }

    /// Splash `Engine::suspendLane` → `admit`: a lone lane the host refused
    /// at its slice end E is suspended to E and, once its retry time came,
    /// resumes holding E's KV at once. A lane suspended beside another
    /// keeps the KV it held as its target and waits out the drain while
    /// the other runs.
    /// Mutation V1: the resume claims its chain only → red (no block past
    /// the chain). Mutation V2: a waiting resume is retried every
    /// iteration → red (resumed before its retry time).
    #[cfg(target_os = "macos")]
    #[test]
    fn a_victim_resumes_to_its_suspended_target() {
        let Some((mut state, governor, _replies)) = judged_state(&[1]) else {
            return;
        };
        state.row_work_ends.insert(1, 33);
        let blocked = vec![(1, kv_denial(Failure::HostPressure))];
        assert!(!state.settle_blocked(blocked, false, &[1]));
        assert_eq!(
            state
                .scheduler
                .turn_mut(1)
                .expect("suspended")
                .resume_target,
            33
        );
        state.resume_preempted();
        assert!(state.scheduler.has_preempted(), "not before its retry time");
        governor.sleep(0.2);
        state.resume_preempted();
        assert!(!state.scheduler.has_preempted(), "resumed");
        assert_eq!(state.inner.scheduler_materialized_blocks(1), 3);
        let resumed = state.scheduler.turn_mut(1).expect("resumed");
        assert_eq!(resumed.resume_target, 0);
        assert!(resumed.payload.cell.is_some());
        assert_eq!(governor.resumptions(7), 1);

        let Some((mut state, governor, _replies)) = judged_state(&[1, 2]) else {
            return;
        };
        let blocked = vec![(1, kv_denial(Failure::EngineBudget))];
        assert!(!state.settle_blocked(blocked, false, &[1, 2]));
        assert_eq!(
            state
                .scheduler
                .turn_mut(2)
                .expect("suspended")
                .resume_target,
            16
        );
        governor.sleep(0.2);
        state.resume_preempted();
        assert!(
            state.scheduler.has_preempted(),
            "not while the resident lane drains"
        );
        let lane = state.scheduler.take_running(1).expect("lane 1");
        state.fail_preempted(lane, "done".into());
        state.resume_preempted();
        assert!(!state.scheduler.has_preempted(), "resumed");
        assert_eq!(state.inner.scheduler_materialized_blocks(2), 1);
        assert_eq!(governor.resumptions(7), 1);
    }

    /// Splash `Engine::tick`: a start the host refused, with no lane
    /// working before it, waits out the 30 s limit and then fails
    /// `resource_timeout`, not before.
    /// Mutation T1 (no deadline) → red (never times out).
    #[cfg(target_os = "macos")]
    #[test]
    fn start_times_out_at_30s_without_competitor_work() {
        use crate::engine::governor::governor::tests::{G, host};
        use crate::engine::governor::host::Pressure;
        let Some((mut state, _, io)) = ledger_state(16 * G, 0) else {
            return;
        };
        io.with(|fake| fake.host = host(32 * G, Pressure::Critical));
        let (prepared, mut result) = prepared_turn(&mut state, 1, tokens_of(1, 1));
        let Err(prepared) = state.admit_prepared(prepared) else {
            panic!("the host refused it: it waits");
        };
        assert_eq!(prepared.resource_wait.failure, Some(Failure::HostPressure));
        state.prepared_waiting = Some(prepared);
        for at in 1..30 {
            io.with(|fake| fake.now = f64::from(at));
            state.tick_waits();
            assert!(result.try_recv().is_err(), "ended at {at} s");
            assert!(state.has_prepared_waiting());
        }
        io.with(|fake| fake.now = 30.0);
        state.tick_waits();
        assert!(!state.has_prepared_waiting());
        let error = result
            .try_recv()
            .expect("the turn has ended")
            .expect_err("the turn fails");
        assert_eq!(
            error.reason,
            "resource_timeout: memory did not become available within the resource wait limit: \
             macOS is short of memory; close memory-heavy applications"
        );
    }

    /// Splash `Engine::admit`'s `droppedLease`: a lone start whose own kept
    /// table pins the pool's top extent, with nothing else to take, gives
    /// the table up, takes the extent its blocks leave, and starts.
    /// Mutation D2: no dropped-lease retry → red (`capacity_exhausted`).
    #[cfg(target_os = "macos")]
    #[test]
    fn a_lone_start_drops_its_own_chain_and_starts() {
        use crate::engine::governor::governor::tests::G;
        let Some((mut state, governor, io)) = ledger_state(G, 0) else {
            return;
        };
        let mut adapter = tiny_paged_adapter_with(Arc::clone(&state.inner.cache_recency), 2, 8);
        adapter.set_grow_headroom_probe_override(Some(passing()));
        adapter.set_extent_growth(2);
        assert!(adapter.try_grow_pool(2));
        adapter.begin_request(9).expect("begin");
        adapter
            .allocate_suffix_blocks_for(9, 32)
            .expect("the runway");
        let chain = tokens_of(40, 2);
        adapter.begin_request(1).expect("begin");
        adapter
            .allocate_suffix_blocks_for(1, 32)
            .expect("the extent");
        adapter.record_tokens(&chain).expect("record");
        adapter
            .register_full_blocks_for_reuse_for(1, &[], 0, false)
            .expect("publish the chain");
        adapter.release_request_for(9).expect("release");
        assert_eq!(
            adapter.free_extent_bytes(false),
            0,
            "the kept table pins it"
        );
        let pools = adapter.layer_kv_pool().total_bytes();
        io.with(|fake| fake.stats.pools = pools);
        let notified = Arc::clone(&io);
        adapter.set_pool_growth_notifier(Some(Arc::new(move |bytes| {
            notified.with(|fake| fake.stats.pools = bytes);
        })));
        state.inner.paged_adapter = Some(adapter);
        let cell = state.inner.recurrent_state_bytes();
        let model = crate::engine::governor::governor::tests::model(pools + cell - 1, 0);
        state.inner.memory_registration = None;
        state.inner.memory_registration = Some(governor.register_model(model).expect("register"));
        let mut prompt = chain.clone();
        prompt.push(3);
        let (prepared, mut result) = prepared_turn(&mut state, 1, prompt);
        let turn = state.admit_prepared(prepared).ok().flatten();
        assert!(turn.is_some(), "not admitted: {:?}", result.try_recv());
        assert!(turn.expect("admitted").payload.cell.is_some());
        let snapshot = state
            .inner
            .scheduler_cache_snapshot()
            .expect("snapshot")
            .expect("adapter");
        assert_eq!(snapshot.blocks.total_blocks, 2, "the extent was cut");
    }

    /// The only running turn blocks at the max pool with nothing idle to
    /// release: it fails with `capacity_exhausted`, its KV and row are
    /// released, and nothing is preempted.
    /// Mutation: preempt the lone turn in `drive_once` as before → red (no
    /// reply: the victim waits to replay; 1 preemption).
    #[cfg(target_os = "macos")]
    #[test]
    fn lone_blocked_turn_fails_capacity_exhausted_without_a_preempt() {
        let Some((mut state, _)) = mtp_step_state_with(64, 64) else {
            return;
        };
        let mut turn = speculative::tests::request(
            &mut state.inner,
            1,
            (0..40).map(|token| token % 16).collect(),
            false,
            Arc::new(AtomicBool::new(false)),
        );
        let (reply, mut result) = tokio::sync::oneshot::channel();
        turn.payload.response = ScheduledReply::Sync(reply);
        assert_eq!(state.inner.scheduler_materialized_blocks(1), 3);
        assert_eq!(
            state.inner.scheduled_recurrent_bytes(),
            state.inner.recurrent_state_bytes()
        );
        settle_blocked_step(&mut state, turn);
        let error = result
            .try_recv()
            .expect("the turn has ended")
            .expect_err("the turn fails");
        assert!(
            error.reason.starts_with("capacity_exhausted:"),
            "{}",
            error.reason
        );
        assert_eq!(state.scheduler.stats().preemptions, 0);
        assert_eq!(state.live_turns(), 0);
        assert_eq!(state.inner.scheduler_materialized_blocks(1), 0);
        assert_eq!(state.inner.scheduled_recurrent_bytes(), 0);
    }

    /// Every turn takes the pool lane's one window: the plan prices the MTP
    /// state and the DFlash2 state into that lane, so an MTP turn, an AR
    /// turn, a DFlash2 turn, the limits the model reports and the window it
    /// publishes agree. On a 64-block pool of 2 MiB blocks in the 27B's bf16
    /// pages (two 64 MiB extents, 2,041 rows after the scratch). A DFlash2
    /// turn one token past the window fails before it books a block.
    /// Mutation O2: the dense `planned_turn_window` prices the held draft
    /// state on top of the pool window for a speculative turn → red
    /// (mtp < ar). Mutation O3: the DFlash2 arm of `run_paged_turn` skips
    /// `preflight_paged_context` → red (the prompt reaches the prime).
    #[test]
    fn every_turn_takes_the_pool_window() {
        use crate::engine::backend::{ChatBackend, WholeTurnArgs};
        use crate::engine::plan::{
            DecoderPlan, MediaCapabilities, MediaInputs, SpeculativeKind, TurnPlan,
        };
        let Some((mut state, _)) = growable_qwen35_state() else {
            return;
        };
        state.inner.config.n_mtp_layers = 1;
        state.inner.config.max_position_embeddings = 4_096;
        state.inner.dflash2 = Some(
            crate::models::qwen3_5::dflash2::tiny_dflash2_model_for_stepper_test(
                &state.inner.config,
            )
            .expect("tiny DFlash2 companion"),
        );
        state.inner.memory_plan = Some(crate::engine::memory_plan::MemoryPlan::with_pool_window(
            64,
            2 << 20,
            0,
            2_041,
        ));
        let ar = state.scheduled_turn_window(16, false);
        let mtp = state.scheduled_turn_window(16, true);
        assert_eq!(ar, 2_041);
        assert_eq!(mtp, ar);
        assert_eq!(state.inner.paged_context_limits().1, ar);
        assert_eq!(
            state
                .inner
                .context_limits_snapshot()
                .effective_window_tokens,
            ar
        );

        let tokenizer = crate::models::gemma4::dspark_decode::tests::tiny_qwen_tokenizer();
        let tokens: Vec<u32> = (0..=ar).map(|i| i % 15 + 1).collect();
        let config = crate::engine::types::ChatConfig {
            temperature: Some(0.0),
            max_new_tokens: Some(2),
            ..Default::default()
        };
        let params = ChatBackend::resolve_params(&state.inner, &config);
        let thinking = ChatBackend::thinking_setup(&state.inner, &config);
        let mut args = WholeTurnArgs {
            tokens: &tokens,
            tokenizer: &tokenizer,
            eos_id: u32::MAX,
            config: &config,
            params: &params,
            thinking,
            plan: TurnPlan {
                is_delta: false,
                input_media: MediaCapabilities::NONE,
                context_media: MediaCapabilities::NONE,
                use_paged_attention: true,
                decoder: DecoderPlan::Speculative(SpeculativeKind::DraftModel),
            },
            sink: None,
            cancelled: None,
            media: MediaInputs {
                images: &[],
                audio: &[],
            },
        };
        let err = match ChatBackend::run_paged_turn(&mut state.inner, &mut args) {
            Ok(_) => panic!("a DFlash2 prompt past the window must fail"),
            Err(err) => err,
        };
        assert!(
            err.reason.contains("context_length_exceeded")
                && err
                    .reason
                    .contains(&format!("effective active context is {ar} tokens")),
            "{}",
            err.reason
        );
        let adapter = state.inner.paged_adapter.as_ref().expect("pool");
        assert_eq!(adapter.current_token_count(), 0);
        assert!(
            adapter
                .block_table()
                .is_none_or(|table| table.num_blocks() == 0),
            "no block booked"
        );
    }

    /// A planned window is not capped by `MLX_PAGED_PER_SEQ_CTX`; without a
    /// plan the pool window is.
    /// Mutation: pass `override_cap` on the planned arm of `scheduled_window`
    /// → red (1).
    #[test]
    fn planned_window_ignores_the_per_seq_cap() {
        assert_eq!(
            scheduled_window(262_144, Some(50_585), 1_000, Some(1)),
            50_585
        );
        assert_eq!(scheduled_window(262_144, None, 1_000, Some(1)), 1);
        assert_eq!(scheduled_window(40_960, Some(50_585), 1_000, None), 40_960);
    }

    /// 40 of 100 blocks, 10 free, 20 booked, 5% watermark (2 blocks'
    /// worth): every grown block adds 0.95 of a block, so 13 blocks fit and
    /// 12 do not.
    /// Mutation: size the grow as the plain shortfall → red (12 ≠ 13).
    #[test]
    fn live_turn_grow_covers_the_watermark_it_raises() {
        let snapshot = SchedulerCacheSnapshot {
            blocks: BlockTelemetry {
                total_blocks: 40,
                free_blocks: 10,
                reclaimable_blocks: 0,
                allocated_blocks: 30,
            },
            max_total_blocks: 100,
            bytes_per_block: 1_000,
        };
        let decision = engine::scheduler::MemoryAdmission {
            admitted: false,
            watermark_bytes: 2_000,
            reserved_block_bytes: 20_000,
            reserved_state_bytes: 0,
        };
        assert_eq!(grow_needed_blocks_for(&snapshot, &decision, 0.05), Some(13));
        let fits = |grown: u64| {
            let capacity = (40 + grown) * 1_000;
            let watermark = (capacity as f64 * 0.05).ceil() as u64;
            20_000 + watermark <= (10 + grown) * 1_000
        };
        assert!(fits(13) && !fits(12));
        let near_max = SchedulerCacheSnapshot {
            max_total_blocks: 45,
            ..snapshot
        };
        assert_eq!(grow_needed_blocks_for(&near_max, &decision, 0.05), None);
    }

    type Probe = Arc<dyn Fn(u32) -> std::result::Result<u32, String> + Send + Sync>;

    // ---- the ledger's units on the scheduled lane ----

    /// Qwen3.5 dense scheduler (no draft) over an `initial`-block pool
    /// growing to `max`, on the model's clock, with `probe`.
    fn qwen35_pool_state(
        initial: u32,
        max: u32,
        probe: Probe,
    ) -> Option<HybridSchedulerState<Qwen35Inner>> {
        let (mut state, _) = growable_qwen35_state()?;
        let recency = Arc::clone(&state.inner.cache_recency);
        let mut adapter = tiny_paged_adapter_with(recency, initial, max);
        adapter.set_grow_headroom_probe_override(Some(probe));
        state.inner.paged_adapter = Some(adapter);
        Some(state)
    }

    fn passing() -> Probe {
        Arc::new(Ok)
    }

    fn tokens_of(seed: u32, blocks: u32) -> Vec<u32> {
        (0..blocks * 16)
            .map(|i| (seed * 11 + i * 7) % 15 + 1)
            .collect()
    }

    /// Caches `tokens` as a chain nothing holds; returns its block hashes.
    fn cache_dead_chain<B: HybridSchedulerBackend>(
        state: &HybridSchedulerState<B>,
        tokens: &[u32],
    ) -> Vec<u64> {
        let adapter = state.inner.paged_adapter().expect("paged adapter");
        let mut allocator = adapter.allocator().lock().expect("allocator");
        let blocks: Vec<_> = (0..tokens.len() / 16)
            .map(|_| allocator.allocate().expect("a chain block"))
            .collect();
        allocator
            .cache_full_blocks(tokens, &blocks, 16, &[], 0)
            .expect("cache the chain");
        for block in blocks {
            allocator.free(block);
        }
        mlx_paged_attn::chain_hashes(tokens, 16, &[], 0)
    }

    /// A ladder rung marking a cached `blocks`-block chain.
    fn rung(
        state: &mut HybridSchedulerState<Qwen35Inner>,
        seed: u32,
        blocks: u32,
        last_used: u64,
    ) -> Vec<u64> {
        let tokens = tokens_of(seed, blocks);
        let hashes = cache_dead_chain(state, &tokens);
        state.inner.push_ladder_entry_for_test(
            tokens.len() as u32,
            hashes.clone(),
            tokens,
            Vec::new(),
            last_used,
        );
        hashes
    }

    /// A finished owned turn records its table's turn-end stamp for its
    /// owner, and drops the stamps of owners no longer kept.
    /// Mutations: drop the `owner_last_used` insert in `finish_completed` →
    /// red (no stamp for owner 1); drop the `retain` there → red (seq 7's
    /// stamp stays).
    #[cfg(target_os = "macos")]
    #[test]
    fn finished_turn_stamps_its_owner_and_drops_stale_stamps() {
        let Some((mut state, _)) = growable_qwen35_state() else {
            return;
        };
        let mut turn = speculative::tests::request(
            &mut state.inner,
            1,
            (0..20).map(|token| token % 15 + 1).collect(),
            false,
            Arc::new(AtomicBool::new(false)),
        );
        turn.payload.reuse_cache = true;
        let (reply, mut result) = tokio::sync::oneshot::channel();
        turn.payload.response = ScheduledReply::Sync(reply);
        state
            .owner_sequences
            .insert(turn.payload.owner_id.clone(), 1);
        state.owner_last_used.insert(7, 3);
        state.finish_completed(turn);
        result
            .try_recv()
            .expect("the turn replied")
            .expect("the turn completes");
        let stamp = state.owner_last_used.get(&1).copied();
        assert!(stamp.is_some(), "owner 1 is stamped");
        let table = state
            .inner
            .paged_adapter()
            .and_then(|adapter| adapter.turn_end_stamp(1));
        if table.is_some() {
            assert_eq!(stamp, table);
        }
        assert!(!state.owner_last_used.contains_key(&7));
    }

    /// A step whose GPU command exhausted Metal memory.
    #[cfg(target_os = "macos")]
    struct OutOfMemoryStep;

    #[cfg(target_os = "macos")]
    impl<P> StepExecutor<P> for OutOfMemoryStep {
        type Error = Error;

        fn execute(
            &mut self,
            _plan: &StepPlan,
            _running: &mut [TurnState<P>],
        ) -> std::result::Result<StepResult, Self::Error> {
            Err(Error::from_reason(
                "[METAL] Command buffer execution failed: Insufficient Memory \
                 (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory).",
            ))
        }
    }

    /// Splash's engine restart after `metal_out_of_memory`, the thread
    /// kept: a step that fails with MLX's command-buffer OOM fails both
    /// running turns `metal_out_of_memory:`, empties the ladder and the
    /// prefix cache, and the loop goes on; a new turn on the rung's prompt
    /// admits with no prefix hit.
    /// Mutations K4 (the loop ends) → red; K5 (no cache reset) → red (the
    /// new turn hits the cached prefix).
    #[cfg(target_os = "macos")]
    #[test]
    fn metal_oom_fails_running_rows_and_resets_caches() {
        let Some((mut state, governor, mut replies)) = judged_state(&[1, 2]) else {
            return;
        };
        let r = *rung(&mut state, 30, 2, 0).last().expect("R");
        let Err(SchedulerError::Executor(error)) = state.step_with(&mut OutOfMemoryStep) else {
            panic!("the step fails");
        };
        assert!(matches!(
            state.executor_failed(error),
            LoopControl::Continue
        ));
        for result in &mut replies {
            let error = result
                .try_recv()
                .expect("the turn replied")
                .expect_err("the turn fails");
            assert!(
                error.reason.starts_with(
                    "metal_out_of_memory: a GPU command exhausted Metal memory ([METAL] Command \
                     buffer execution failed: Insufficient Memory"
                ),
                "{}",
                error.reason
            );
        }
        assert_eq!(state.scheduler.running_len(), 0);
        assert!(!has_rung(&state, r), "the ladder is reset");
        assert_eq!(state.inner.dflash2_states.store.entries().count(), 0);
        assert_eq!(governor.metal_out_of_memory(7), 1);
        let mut prompt = tokens_of(30, 2);
        prompt.push(3);
        let (prepared, mut result) = prepared_turn(&mut state, 3, prompt);
        let turn = state
            .admit_prepared(prepared)
            .ok()
            .flatten()
            .unwrap_or_else(|| panic!("not admitted: {:?}", result.try_recv()));
        assert_eq!(
            turn.payload.prefix.effective_cached_prefix_len(),
            0,
            "no prefix hit after the reset"
        );
    }

    /// A step whose every row ends: row 2 with MLX's allocation failure,
    /// the others as completed turns.
    #[cfg(target_os = "macos")]
    struct OneRowOutOfMemory;

    #[cfg(target_os = "macos")]
    impl<P> StepExecutor<ScheduledTurn<P>> for OneRowOutOfMemory {
        type Error = Error;

        fn execute(
            &mut self,
            plan: &StepPlan,
            running: &mut [TurnState<ScheduledTurn<P>>],
        ) -> std::result::Result<StepResult, Self::Error> {
            for turn in running.iter_mut().filter(|turn| turn.seq_id == 2) {
                turn.payload.failure = Some(Error::from_reason(
                    "[malloc] Unable to allocate 1073741824 bytes.",
                ));
            }
            Ok(StepResult {
                rows: plan
                    .rows
                    .iter()
                    .map(|row| RowStepResult {
                        seq_id: row.seq_id,
                        num_computed_tokens: row.num_tokens,
                        generated_token: None,
                        speculative: None,
                        finished: true,
                        allocation_blocked: false,
                        prefill_micros: 0,
                    })
                    .collect(),
                executed_decode_batch: 0,
                executed_greedy_epilogue_batch: 0,
                rows_alloc_evicted: 0,
                prefill_wall_micros: 0,
                decode_wall_micros: 0,
            })
        }
    }

    /// Splash restarts on `metal_out_of_memory` however the step's requests
    /// ended: one row's GPU allocation failure fails the peer the same step
    /// completed too, and the caches reset.
    /// Mutation E4 (the step's completed peers finish before the OOM is
    /// seen) → red (row 1 replies a result).
    #[cfg(target_os = "macos")]
    #[test]
    fn one_rows_oom_fails_the_peer_its_step_completed() {
        let Some((mut state, governor, mut replies)) = judged_state(&[1, 2]) else {
            return;
        };
        let step = state
            .step_with(&mut OneRowOutOfMemory)
            .unwrap_or_else(|_| panic!("the step runs"));
        let SchedulerAction::Stepped { completed, .. } = &step else {
            panic!("the step runs");
        };
        assert_eq!(completed.len(), 2, "both rows end in the step");
        state.settle_step(step);
        state.recover_if_gpu_oom();
        for (seq, result) in replies.iter_mut().enumerate() {
            let error = result
                .try_recv()
                .expect("the turn replied")
                .expect_err("the turn fails");
            assert!(
                error.reason.starts_with(
                    "metal_out_of_memory: a GPU command exhausted Metal memory ([malloc]"
                ),
                "row {}: {}",
                seq + 1,
                error.reason
            );
        }
        assert_eq!(state.scheduler.running_len(), 0);
        assert_eq!(governor.metal_out_of_memory(7), 1);
    }

    /// The reply of a request a GPU memory restart ended.
    #[cfg(target_os = "macos")]
    fn assert_metal_oom(
        result: &mut tokio::sync::oneshot::Receiver<Result<ChatResult>>,
        who: &str,
    ) {
        let error = result
            .try_recv()
            .unwrap_or_else(|_| panic!("{who} replied"))
            .expect_err("the request fails");
        assert!(
            error.reason.starts_with(
                "metal_out_of_memory: a GPU command exhausted Metal memory ([METAL] Command \
                 buffer execution failed: Insufficient Memory"
            ),
            "{who}: {}",
            error.reason
        );
    }

    /// Splash `NativeRuntime::executionFailed` around `Engine::tick`'s
    /// admission: a new turn whose prefix preparation MLX failed for Metal
    /// memory, and a suspended lane whose resume did, fail
    /// `metal_out_of_memory:` and restart the model at once: the resident
    /// lanes fail and the ladder resets.
    /// Mutation E7 (a prefix preparation's error is not classified) → red
    /// (the new turn gets MLX's text); E7b (the resume's is not) → red (the
    /// resumed lane gets MLX's text).
    #[cfg(target_os = "macos")]
    #[test]
    fn a_prefix_preparation_oom_restarts_the_model() {
        const OOM: &str = "[METAL] Command buffer execution failed: Insufficient Memory \
                           (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory).";
        let Some((mut state, governor, mut replies)) = judged_state(&[1]) else {
            return;
        };
        let r = *rung(&mut state, 30, 2, 0).last().expect("R");
        state.prefix_failure = Some(OOM);
        let mut prompt = tokens_of(40, 2);
        prompt.push(3);
        let (prepared, mut result) = prepared_turn(&mut state, 3, prompt);
        assert!(matches!(state.admit_prepared(prepared), Ok(None)));
        assert!(state.prefix_failure.is_none(), "the prefix was prepared");
        assert_metal_oom(&mut result, "the new turn");
        assert_metal_oom(&mut replies[0], "the resident lane");
        assert_eq!(state.scheduler.running_len(), 0);
        assert!(!has_rung(&state, r), "the ladder is reset");
        assert_eq!(governor.metal_out_of_memory(7), 1);

        let Some((mut state, governor, mut replies)) = judged_state(&[1]) else {
            return;
        };
        state.row_work_ends.insert(1, 33);
        let blocked = vec![(1, kv_denial(Failure::HostPressure))];
        assert!(!state.settle_blocked(blocked, false, &[1]));
        let r = *rung(&mut state, 30, 2, 0).last().expect("R");
        governor.sleep(0.2);
        state.prefix_failure = Some(OOM);
        state.resume_preempted();
        assert!(state.prefix_failure.is_none(), "the resume was prepared");
        assert!(!state.scheduler.has_preempted());
        assert_metal_oom(&mut replies[0], "the resumed lane");
        assert!(!has_rung(&state, r), "the ladder is reset");
        assert_eq!(governor.metal_out_of_memory(7), 1);
    }

    /// Splash's engine restart fails every queued start: a held start's
    /// prefix preparation that MLX failed for Metal memory restarts the
    /// model, and a later held start the retry has not tried yet fails
    /// `metal_out_of_memory:` too, never admitted without its owner.
    /// Mutation HR1 (the retry takes the held starts out first) → red (the
    /// later start is admitted).
    #[cfg(target_os = "macos")]
    #[test]
    fn a_restart_fails_the_held_starts_not_yet_tried() {
        const OOM: &str = "[METAL] Command buffer execution failed: Insufficient Memory \
                           (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory).";
        let Some((mut state, _governor, mut replies)) = judged_state(&[1]) else {
            return;
        };
        let mut prompt = tokens_of(40, 2);
        prompt.push(3);
        let (first, mut first_result) = prepared_turn(&mut state, 2, prompt);
        let (second, mut second_result) = prepared_turn(&mut state, 3, tokens_of(3, 1));
        state.held.push_back(first);
        state.held.push_back(second);
        state.prefix_failure = Some(OOM);
        assert!(!state.retry_held(None));
        assert!(state.prefix_failure.is_none(), "the first was prepared");
        assert_metal_oom(&mut first_result, "the first held start");
        assert_metal_oom(&mut second_result, "the second held start");
        assert_metal_oom(&mut replies[0], "the resident lane");
        assert!(state.held.is_empty());
        assert!(state.prepared_waiting.is_none());
        assert_eq!(
            state.scheduler.running_len() + state.scheduler.waiting_len(),
            0
        );
        assert!(!state.owner_sequences.contains_key("owner-3"));
    }

    /// A step that tells the test it runs, then waits for its release.
    #[cfg(target_os = "macos")]
    struct HeldStep {
        entered: std::sync::mpsc::Sender<Option<&'static Governor>>,
        governor: &'static Governor,
        release: std::sync::mpsc::Receiver<()>,
    }

    #[cfg(target_os = "macos")]
    impl<P> StepExecutor<P> for HeldStep {
        type Error = Error;

        fn execute(
            &mut self,
            plan: &StepPlan,
            running: &mut [TurnState<P>],
        ) -> std::result::Result<StepResult, Self::Error> {
            let _ = self.entered.send(Some(self.governor));
            let _ = self.release.recv();
            RunEveryRow.execute(plan, running)
        }
    }

    /// The governor's status is read from the governor alone: it returns
    /// while a scheduled turn's step runs on the model thread, and its
    /// accounts (the model's published ones, the MLX free pool and
    /// "fixed") sum to charged.
    /// Mutation Z3: the step holds what the status reads (as a status sent
    /// to the model thread would wait for it) → red (no status before the
    /// step is released).
    #[cfg(target_os = "macos")]
    #[test]
    fn status_reads_while_a_turn_runs() {
        use crate::engine::governor::governor::tests::{FakeIo, G, host, model, stats};
        use std::sync::mpsc;
        let (entered, step_running) = mpsc::channel();
        let (release, held) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            let Some((mut state, _)) = growable_qwen35_state() else {
                let _ = entered.send(None);
                return None;
            };
            let io = FakeIo::new(
                stats(4 * G),
                host(32 * G, crate::engine::governor::host::Pressure::Normal),
            );
            io.with(|fake| fake.stats.cache = G);
            let governor: &'static Governor = Box::leak(Box::new(Governor::new(io)));
            state.inner.memory_registration =
                Some(governor.register_model(model(16 * G, 0)).expect("register"));
            let accounts = state.inner.memory_accounts().expect("accounts");
            assert!(accounts.kv > 0, "{accounts:?}");
            governor.publish_accounts(7, accounts);
            let mut turn = speculative::tests::request(
                &mut state.inner,
                1,
                (0..16).collect(),
                false,
                Arc::new(AtomicBool::new(false)),
            );
            let (reply, result) = tokio::sync::oneshot::channel();
            turn.payload.response = ScheduledReply::Sync(reply);
            state.scheduler.reinstate_running(turn);
            let mut step = HeldStep {
                entered,
                governor,
                release: held,
            };
            state.step_with(&mut step).expect("the step runs");
            drop(result);
            Some(accounts)
        });
        let Some(governor) = step_running.recv().expect("the worker reports") else {
            return;
        };
        let (status_tx, status_rx) = mpsc::channel();
        std::thread::spawn(move || {
            let _ = status_tx.send(crate::engine::governor::status::status(governor));
        });
        let status = status_rx.recv_timeout(Duration::from_secs(5));
        release.send(()).expect("release the step");
        let accounts = worker.join().expect("the worker ends").expect("accounts");
        let status = status
            .expect("the status returns while the step runs")
            .expect("a model is registered");
        let rows = &status.snapshot.ledger.accounts;
        let current = |name: &str| {
            rows.iter()
                .find(|row| row.name == name)
                .map_or(0, |row| row.bytes)
        };
        assert_eq!(current("kv"), accounts.kv, "{rows:?}");
        assert_eq!(current("free_pool"), G, "{rows:?}");
        assert!(current("fixed") > 0, "{rows:?}");
        assert_eq!(
            rows.iter().map(|row| row.bytes).sum::<u64>(),
            status.snapshot.charged,
            "{rows:?}"
        );
    }

    // ---- the lone blocked turn reaches in-use points (Splash `growthClass`) ----

    fn has_rung(state: &HybridSchedulerState<Qwen35Inner>, final_hash: u64) -> bool {
        state
            .inner
            .ladder_entries_for_test()
            .iter()
            .any(|&(_, hash, _)| hash == final_hash)
    }

    /// Puts `turn` (its 40-token prompt recorded) at its 49th token, so its
    /// next decode row writes into a 4th block, then runs and settles one
    /// step.
    #[cfg(target_os = "macos")]
    fn claim_lone_step<B: HybridSchedulerBackend>(
        state: &mut HybridSchedulerState<B>,
        mut turn: TurnState<ScheduledTurn<B::PrefixState>>,
    ) {
        turn.token_history.extend([3; 9]);
        turn.num_computed_tokens = 48;
        turn.num_tokens = 49;
        state.scheduler.reinstate_running(turn);
        let step = state.step_with(&mut RunEveryRow).expect("a step");
        state.settle_step(step);
    }

    /// A running turn over a 40-token prompt (3 blocks) whose reply is
    /// `result`, on a pool of exactly `max` blocks.
    #[cfg(target_os = "macos")]
    fn running_turn<B: HybridSchedulerBackend>(
        state: &mut HybridSchedulerState<B>,
        seq: u32,
    ) -> (
        TurnState<ScheduledTurn<B::PrefixState>>,
        tokio::sync::oneshot::Receiver<Result<ChatResult>>,
    ) {
        let mut turn = speculative::tests::request(
            &mut state.inner,
            seq,
            (0..40).map(|token| (token + seq) % 16).collect(),
            false,
            Arc::new(AtomicBool::new(false)),
        );
        let (reply, result) = tokio::sync::oneshot::channel();
        turn.payload.response = ScheduledReply::Sync(reply);
        assert_eq!(state.inner.scheduler_materialized_blocks(seq), 3);
        (turn, result)
    }

    /// The step's only running turn T (3 blocks) claims one more block at
    /// the max pool; the rest of the pool is the 2-block cache-only chain
    /// of rung R, which suspended turn 7 restored from. The lone lane's
    /// claim reaches R (Splash `Engine::growthClass`), takes it, and runs.
    /// Mutation GC1: `growth_class` is always Ordinary → red (T fails
    /// `capacity_exhausted`).
    #[cfg(target_os = "macos")]
    #[test]
    fn lone_turn_takes_a_suspended_turns_replay_point_last() {
        let Some(mut state) = qwen35_pool_state(5, 5, passing()) else {
            return;
        };
        let (turn, mut result) = running_turn(&mut state, 1);
        let r = *rung(&mut state, 30, 2, 0).last().expect("R");
        state.inner.use_ladder_entry_for_test(7, r);
        let snapshot = state
            .inner
            .scheduler_cache_snapshot()
            .expect("snapshot")
            .expect("adapter");
        assert_eq!(snapshot.blocks.total_blocks, snapshot.max_total_blocks);
        let adapter = state.inner.paged_adapter().expect("paged adapter");
        assert_eq!(adapter.takeable_blocks(), 0);
        claim_lone_step(&mut state, turn);
        assert!(
            matches!(
                result.try_recv(),
                Err(tokio::sync::oneshot::error::TryRecvError::Empty)
            ),
            "the turn is still running"
        );
        assert_eq!(state.scheduler.running().len(), 1);
        assert!(!has_rung(&state, r), "R was taken");
    }

    /// The kept prompt wins over the in-use reach: R, the replay point
    /// suspended turn 7 uses, is an `InUse` unit of every claim but the one
    /// whose prompt restores through it.
    /// Mutation KP1: drop the `kept.keeps` filter in dense
    /// `gdn_state_units` → red (R listed for its own prompt).
    #[test]
    fn kept_prompt_beats_in_use_reach() {
        let Some(mut state) = qwen35_pool_state(4, 4, passing()) else {
            return;
        };
        let r = *rung(&mut state, 30, 2, 0).last().expect("R");
        state.inner.use_ladder_entry_for_test(7, r);
        let mut prompt = tokens_of(30, 2);
        prompt.push(1);
        let query = VictimQuery {
            rights: crate::engine::governor::ledger::TakeRights {
                in_use: true,
                certain_reuse: CertainReuse::All,
                ..Default::default()
            },
            need: Need {
                pages: 1,
                kv: true,
                ..Need::default()
            },
        };
        let listed = |kept: &[u32]| {
            state.inner.victim_units(&query, kept).into_units().into_iter().any(|unit| {
                unit.stage == Stage::InUse
                    && matches!(unit.key, LedgerKey::Ladder { final_block_hash, .. } if final_block_hash == r)
            })
        };
        assert!(!listed(&prompt));
        assert!(listed(&[]));
    }

    /// Stored checkpoints go first among the model's units (Splash
    /// `Cache::nextUnit`: `Phase::Checkpoints` before `Phase::Ordinary`),
    /// ahead of an older idle owner; the rest stays in stamp order. A
    /// bytes need one unit covers takes the checkpoint. A pages-only walk
    /// lists no checkpoint.
    /// Mutation CF1 (`SchedUnits` merges checkpoints into the stamp order)
    /// → red (the owner is listed first).
    #[cfg(target_os = "macos")]
    #[test]
    fn checkpoints_go_before_ordinary_states() {
        use crate::engine::governor::ledger::TakeRights;
        use crate::models::qwen3_5::dflash2_state::paged::PagedAnchor;
        use crate::models::qwen3_5::dflash2_state::store::{Lineage, StateClass};
        use crate::models::qwen3_5::dflash2_turn::{Keep, model_units};
        use crate::models::qwen3_5::draft_context::DraftContextWindow;
        let Some(mut state) = qwen35_pool_state(8, 8, passing()) else {
            return;
        };
        drop(speculative::tests::request(
            &mut state.inner,
            2,
            (0..20).map(|token| 400 + token).collect(),
            false,
            Arc::new(AtomicBool::new(false)),
        ));
        state.owner_sequences.insert("idle".into(), 2);
        state.owner_last_used.insert(2, 0);
        let anchor = || PagedAnchor {
            linear: Vec::new(),
            window: DraftContextWindow::new(64, 8).expect("window"),
        };
        let (ordinary, checkpoint) = (tokens_of(41, 1), tokens_of(42, 1));
        let store = &mut state.inner.dflash2_states.store;
        let o = store
            .insert(Lineage::text(&ordinary, 0), anchor(), StateClass::Ordinary)
            .expect("ordinary")
            .id;
        let c = store
            .insert(
                Lineage::text(&checkpoint, 0),
                anchor(),
                StateClass::Checkpoint,
            )
            .expect("checkpoint")
            .id;
        store.touch(c);
        let stamp = |id| store.get(id).expect("stored").last_used();
        let (o_stamp, c_stamp) = (stamp(o), stamp(c));
        assert!(o_stamp + 1 < c_stamp);
        let r = rung(&mut state, 43, 1, o_stamp + 1)[0];
        let r_key = LedgerKey::Ladder {
            owner_id: String::new(),
            prefix_len: 16,
            final_block_hash: r,
        };
        let query = |need: Need| VictimQuery {
            rights: TakeRights {
                units: true,
                certain_reuse: CertainReuse::All,
                ..TakeRights::default()
            },
            need,
        };
        let bytes = query(Need {
            bytes: 1,
            ..Need::default()
        });
        let keys = |units: Vec<Victim>| -> Vec<LedgerKey> {
            units
                .into_iter()
                .map(|unit| match unit.key {
                    LedgerKey::Ladder {
                        prefix_len,
                        final_block_hash,
                        ..
                    } => LedgerKey::Ladder {
                        owner_id: String::new(),
                        prefix_len,
                        final_block_hash,
                    },
                    key => key,
                })
                .filter(|key| *key != LedgerKey::Extent)
                .collect()
        };
        assert_eq!(
            keys(model_units(&state.inner, Keep::default(), &bytes).into_units()),
            vec![LedgerKey::Store(c), LedgerKey::Store(o), r_key.clone()]
        );
        let pages = query(Need {
            pages: 1,
            kv: true,
            host_refused: true,
            ..Need::default()
        });
        assert!(
            !keys(model_units(&state.inner, Keep::default(), &pages).into_units())
                .contains(&LedgerKey::Store(c)),
            "a pages-only walk skips checkpoints"
        );
        let mut units = SchedUnits {
            state: &mut state,
            claimant: 1,
            kept: &[],
        };
        let listed = keys(units.units(&bytes));
        assert_eq!(
            listed,
            vec![
                LedgerKey::Store(c),
                LedgerKey::Owner(2),
                LedgerKey::Store(o),
                r_key
            ]
        );
        let head = units
            .units(&bytes)
            .into_iter()
            .find(|unit| unit.key != LedgerKey::Extent)
            .expect("a unit");
        units.take(&bytes, &head);
        let after = keys(units.units(&bytes));
        assert!(!after.contains(&LedgerKey::Store(c)), "the checkpoint went");
        assert!(after.contains(&LedgerKey::Owner(2)), "the owner stays");
    }

    /// MoE twin of `lone_turn_takes_a_suspended_turns_replay_point_last`.
    /// Mutation GC1: `growth_class` is always Ordinary → red (T fails
    /// `capacity_exhausted`).
    #[cfg(target_os = "macos")]
    #[test]
    fn moe_lone_turn_takes_a_suspended_turns_replay_point_last() {
        if !crate::models::qwen3_5::dflash2_decode::tests::metal_is_default_device() {
            return;
        }
        let mut inner = Qwen35MoeInner::new(tiny_moe_config()).expect("construct tiny MoE");
        let mut adapter = tiny_paged_adapter_with(Arc::clone(&inner.cache_recency), 5, 5);
        adapter.set_grow_headroom_probe_override(Some(passing()));
        inner.paged_adapter = Some(adapter);
        let mut state = HybridSchedulerState::new(inner).expect("construct MoE scheduler");
        let (turn, mut result) = running_turn(&mut state, 1);
        let tokens = tokens_of(30, 2);
        let hashes = cache_dead_chain(&state, &tokens);
        let r = *hashes.last().expect("R");
        state
            .inner
            .push_ladder_entry_for_test(32, hashes, tokens, Vec::new(), 0);
        state.inner.use_ladder_entry_for_test(7, r);
        let snapshot = state
            .inner
            .scheduler_cache_snapshot()
            .expect("snapshot")
            .expect("adapter");
        assert_eq!(snapshot.blocks.total_blocks, snapshot.max_total_blocks);
        let adapter = state.inner.paged_adapter().expect("paged adapter");
        assert_eq!(adapter.takeable_blocks(), 0);
        claim_lone_step(&mut state, turn);
        assert!(
            matches!(
                result.try_recv(),
                Err(tokio::sync::oneshot::error::TryRecvError::Empty)
            ),
            "the turn is still running"
        );
        assert_eq!(state.scheduler.running().len(), 1);
        assert!(
            !state
                .inner
                .ladder_entries_for_test()
                .iter()
                .any(|&(_, hash, _)| hash == r),
            "R was taken"
        );
    }

    /// Marked chains are out of the snapshot's reclaimable blocks.
    /// Mutation: fill it from `reclaimable_blocks` (reusable) → red (2).
    #[test]
    fn scheduler_snapshot_counts_only_takeable_blocks() {
        let Some(state) = qwen35_pool_state(2, 2, passing()) else {
            return;
        };
        let hashes = cache_dead_chain(&state, &tokens_of(12, 2));
        let reclaimable = |state: &HybridSchedulerState<Qwen35Inner>| {
            state
                .inner
                .scheduler_cache_snapshot()
                .expect("snapshot")
                .expect("adapter")
                .blocks
                .reclaimable_blocks
        };
        let adapter = state.inner.paged_adapter().expect("paged adapter");
        adapter.set_state_marks([hashes[1]]);
        assert_eq!(reclaimable(&state), 0);
        state
            .inner
            .paged_adapter()
            .expect("paged adapter")
            .set_state_marks([]);
        assert_eq!(reclaimable(&state), 2);
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
            .try_reserve_reclaiming_idle_with(2, 10, 0, 0, snapshot, |_inner, victim| {
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
        state.pending.push_back(
            Qwen35Cmd::from_chat(ChatCmd::SessionStart {
                messages: Vec::new(),
                config: ChatConfig::default(),
                reply: ordinary_reply,
                cancelled: Arc::new(AtomicBool::new(false)),
            })
            .into(),
        );
        let (release_reply, mut release_result) = tokio::sync::oneshot::channel();
        state.pending.push_back(
            Qwen35Cmd::from_chat(ChatCmd::ReleaseCacheOwner {
                owner_id: "completed-owner".into(),
                reply: release_reply,
            })
            .into(),
        );

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
        state.pending.push_back(
            Qwen35Cmd::from_chat(ChatCmd::SessionStart {
                messages: Vec::new(),
                config: ChatConfig::default(),
                reply: ordinary_reply,
                cancelled: Arc::new(AtomicBool::new(false)),
            })
            .into(),
        );
        let (reset_reply, _reset_result) = tokio::sync::oneshot::channel();
        state
            .pending
            .push_back(Qwen35Cmd::from_chat(ChatCmd::ResetCaches { reply: reset_reply }).into());
        let (release_reply, mut release_result) = tokio::sync::oneshot::channel();
        state.pending.push_back(
            Qwen35Cmd::from_chat(ChatCmd::ReleaseCacheOwner {
                owner_id: "completed-owner".into(),
                reply: release_reply,
            })
            .into(),
        );

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
        state.pending.push_back(
            Qwen35Cmd::from_chat(ChatCmd::SessionContinue {
                messages: Vec::new(),
                config: ChatConfig {
                    cache_owner_id: Some("continued-owner".into()),
                    ..ChatConfig::default()
                },
                reply: turn_reply,
                cancelled: Arc::new(AtomicBool::new(false)),
            })
            .into(),
        );
        let (release_reply, mut release_result) = tokio::sync::oneshot::channel();
        state.pending.push_back(
            Qwen35Cmd::from_chat(ChatCmd::ReleaseCacheOwner {
                owner_id: "continued-owner".into(),
                reply: release_reply,
            })
            .into(),
        );

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

    /// Word-level `t0`..`t15` plus `<|im_end|>` (16, outside the 16-token
    /// head) and a fixed 8-token chat template.
    fn vacated_stop_tokenizer() -> Arc<Qwen3Tokenizer> {
        let vocab = (0..16)
            .map(|i| format!("\"t{i}\": {i}"))
            .collect::<Vec<_>>()
            .join(", ");
        let json = format!(
            r#"{{
                "version": "1.0",
                "truncation": null,
                "padding": null,
                "added_tokens": [],
                "normalizer": null,
                "pre_tokenizer": {{ "type": "Whitespace" }},
                "post_processor": null,
                "decoder": null,
                "model": {{
                    "type": "WordLevel",
                    "vocab": {{ {vocab}, "<|im_end|>": 16 }},
                    "unk_token": "t0"
                }}
            }}"#
        );
        let tokenizer = tokenizers::Tokenizer::from_bytes(json.as_bytes())
            .unwrap_or_else(|error| panic!("fixture tokenizer: {error}"));
        Arc::new(
            Qwen3Tokenizer::from_tokenizer(tokenizer, Some("t1 t2 t3 t4 t5 t6 t7 t8".to_string()))
                .expect("tokenizer"),
        )
    }

    fn vacated_stop_config(owner: &str) -> ChatConfig {
        ChatConfig {
            max_new_tokens: Some(40),
            temperature: Some(0.0),
            enable_mtp: Some(false),
            max_consecutive_tokens: Some(0),
            max_ngram_repeats: Some(0),
            reasoning_effort: Some("none".to_string()),
            cache_owner_id: Some(owner.to_string()),
            ..ChatConfig::default()
        }
    }

    fn vacated_stop_message() -> Vec<crate::tokenizer::ChatMessage> {
        vec![
            serde_json::from_value(serde_json::json!({"role": "user", "content": "t1"}))
                .expect("message"),
        ]
    }

    fn sync_peer(owner: &str) -> (ChatCmd, tokio::sync::oneshot::Receiver<Result<ChatResult>>) {
        let (reply, result) = tokio::sync::oneshot::channel();
        let command = ChatCmd::SessionStart {
            messages: vacated_stop_message(),
            config: ChatConfig {
                max_new_tokens: Some(2),
                ..vacated_stop_config(owner)
            },
            reply,
            cancelled: Arc::new(AtomicBool::new(false)),
        };
        (command, result)
    }

    /// Splash `Engine::stop` on a suspended request: `settle` → `finish(Stop)`
    /// takes no lane back. Turn A generates beside owner B, which finishes and
    /// parks its recurrent row; A is preempted (its row dropped, its full
    /// blocks registered) and owner C holds the second recurrent unit. A stop
    /// on A ends it "stop" on the tokens it had, the full blocks through the
    /// cut stay in the cache, and B and C keep their rows. `in_reap`: the
    /// pass's reap settles it at once, not its resume.
    fn a_vacated_stop_completes_beside_two_resident_peers<B: HybridSchedulerBackend>(
        mut state: HybridSchedulerState<B>,
        in_reap: bool,
    ) {
        state.enabled = true;
        let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
        let (plumbing, mut rx) = crate::engine::napi_glue::chat_stream_plumbing();
        let mut chunks = Vec::new();
        let command = ChatCmd::StreamSessionStart {
            messages: vacated_stop_message(),
            config: vacated_stop_config("a"),
            stream_tx: plumbing.stream_tx.clone(),
            cancelled: plumbing.cancelled.clone(),
        };
        assert!(
            sender
                .send(ModelCommand::<B::FamilyCommand>::from_chat(command))
                .is_ok()
        );
        let (command, mut b_result) = sync_peer("b");
        assert!(sender.send(ModelCommand::from_chat(command)).is_ok());
        let mut b_done = None;
        for _ in 0..64 {
            assert!(matches!(state.drive(&mut receiver), LoopControl::Continue));
            while let Ok(chunk) = rx.try_recv() {
                chunks.push(chunk);
            }
            if b_done.is_none() {
                b_done = b_result.try_recv().ok();
            }
            let seq = state.owner_sequences["a"];
            if b_done.is_some()
                && state
                    .turn_mut(seq)
                    .is_some_and(|turn| turn.payload.generated_tokens.len() >= 20)
            {
                break;
            }
        }
        let b = b_done
            .expect("B did not finish")
            .unwrap_or_else(|error| panic!("B failed: {}", error.reason));
        assert_eq!(b.num_tokens, 2);
        let seq = state.owner_sequences["a"];
        let seq_b = state.owner_sequences["b"];
        let (prompt, cut, salt) = {
            let turn = state.turn_mut(seq).expect("A still runs");
            (
                turn.payload.prompt_tokens.clone(),
                turn.payload.generated_tokens.clone(),
                turn.payload.params.cache_salt,
            )
        };
        assert!(cut.len() >= 20, "{} tokens", cut.len());
        assert!(state.preempt_running(seq));
        assert!(!state.inner.has_scheduled_recurrent(seq));

        // Admission waits behind a preempted turn (on-demand booking), so the
        // second resident row is pinned for owner C directly.
        let seq_c = state.next_seq_id;
        state.next_seq_id += 1;
        state.owner_sequences.insert("c".to_string(), seq_c);
        state
            .inner
            .activate_scheduled_recurrent(seq_c)
            .expect("C takes the second unit");
        state
            .inner
            .park_active_scheduled_recurrent()
            .expect("C parks");
        assert!(state.inner.has_scheduled_recurrent(seq_b), "B stays parked");
        assert!(
            !state.inner.can_activate_scheduled_recurrent(seq),
            "B and C hold both recurrent units"
        );

        let model = state.inner.memory_model();
        assert_ne!(model, 0, "the fixture registers its memory");
        let epoch = state.memory_governor().epoch(model);
        plumbing.handle.stop();
        if in_reap {
            state.reap_cancelled_waiters();
        } else {
            state.resume_preempted();
        }
        assert!(
            state.memory_governor().epoch(model) > epoch,
            "the suspended request's end wakes the waits"
        );
        while let Ok(chunk) = rx.try_recv() {
            chunks.push(chunk);
        }
        drop(plumbing);
        let done = chunks
            .into_iter()
            .map(|chunk| {
                chunk.unwrap_or_else(|error| panic!("stopped turn failed: {}", error.reason))
            })
            .find(|chunk| chunk.done)
            .expect("the stopped turn sent no done chunk");
        assert_eq!(done.finish_reason.as_deref(), Some("stop"));
        assert_eq!(done.num_tokens, Some(cut.len() as u32));
        assert!(!state.owner_sequences.contains_key("a"));

        let mut history = prompt;
        history.extend_from_slice(&cut[..cut.len() - 1]);
        let adapter = state.inner.paged_adapter().expect("adapter");
        let block_size = adapter.block_size();
        let (_, hit) = adapter.allocator().lock().unwrap().find_longest_cache_hit(
            &history,
            block_size,
            &[],
            salt,
        );
        assert_eq!(
            hit,
            history.len() / block_size as usize * block_size as usize
        );
        assert!(hit >= block_size as usize);

        assert!(state.inner.has_scheduled_recurrent(seq_b));
        assert!(state.inner.has_scheduled_recurrent(seq_c));
    }

    /// A governor of its own with room for the fixtures, and its clock.
    fn roomy_governor() -> (
        Arc<crate::engine::governor::governor::tests::FakeIo>,
        &'static crate::engine::governor::governor::Governor,
    ) {
        use crate::engine::governor::governor::Governor;
        use crate::engine::governor::governor::tests::{FakeIo, G, host, stats};
        let io = FakeIo::new(
            stats(0),
            host(32 * G, crate::engine::governor::host::Pressure::Normal),
        );
        let governor: &'static Governor = Box::leak(Box::new(Governor::new(io.clone())));
        (io, governor)
    }

    /// Mutation F4: `finish_completed` activates the vacated turn before it
    /// reads `vacated_stop` → red (the recurrent-unit cap error). SP1 (a
    /// suspended request's end signals no progress) → red. PS1 (the reap
    /// leaves a stopped suspended turn to its resume) → red (no wake, no
    /// done chunk).
    #[cfg(target_os = "macos")]
    #[test]
    fn a_vacated_stop_completes_beside_two_resident_qwen35_peers() {
        use crate::engine::governor::governor::tests::{G, model};
        if !crate::models::qwen3_5::dflash2_decode::tests::metal_is_default_device() {
            return;
        }
        for in_reap in [false, true] {
            let mut inner = crate::models::qwen3_5::model::scheduled_mtp::seeded_inner(0x5730_0002);
            inner.tokenizer = Some(vacated_stop_tokenizer());
            inner.memory_registration = Some(
                roomy_governor()
                    .1
                    .register_model(model(16 * G, 0))
                    .expect("register"),
            );
            a_vacated_stop_completes_beside_two_resident_peers(
                HybridSchedulerState::new(inner).expect("scheduler"),
                in_reap,
            );
        }
    }

    /// Mutation F4, SP1, PS1: as the dense test → red.
    #[cfg(target_os = "macos")]
    #[test]
    fn a_vacated_stop_completes_beside_two_resident_qwen35_moe_peers() {
        use crate::engine::governor::governor::tests::{G, model};
        if !crate::models::qwen3_5::dflash2_decode::tests::metal_is_default_device() {
            return;
        }
        for in_reap in [false, true] {
            let mut inner =
                crate::models::qwen3_5_moe::model::scheduled_mtp::seeded_inner(0x5730_0003);
            inner.tokenizer = Some(vacated_stop_tokenizer());
            inner.memory_registration = Some(
                roomy_governor()
                    .1
                    .register_model(model(16 * G, 0))
                    .expect("register"),
            );
            a_vacated_stop_completes_beside_two_resident_peers(
                HybridSchedulerState::new(inner).expect("scheduler"),
                in_reap,
            );
        }
    }

    /// When a running turn is stopped.
    #[derive(Clone, Copy, Debug, PartialEq)]
    enum StopAt {
        /// Its memory wait has not reached its retry.
        Waiting,
        /// Its memory wait is past its limit too.
        Expired,
        /// Its last prefill slice sampled a token no decode row took yet.
        FirstToken,
    }

    /// Splash `Engine::stop` on a running lane: `settle` ends it at the pass
    /// boundary as `finish(Stop)`, and a lane whose claim waits is no
    /// exception. A running turn whose memory wait has not reached its
    /// retry, or has passed its limit, is stopped: the next pass ends it in
    /// its step, done "stop" on the tokens it generated (Splash
    /// `generatedTokens` counts the prefill's sampled token), its owner state
    /// parked (a completion, not the ordinary cancel's failure, nor
    /// `resource_timeout`).
    fn a_stopped_running_turn_ends_in_its_step<B: HybridSchedulerBackend>(
        mut state: HybridSchedulerState<B>,
        io: &crate::engine::governor::governor::tests::FakeIo,
        at: StopAt,
    ) {
        state.enabled = true;
        let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
        let (plumbing, mut rx) = crate::engine::napi_glue::chat_stream_plumbing();
        let command = ChatCmd::StreamSessionStart {
            messages: vacated_stop_message(),
            config: vacated_stop_config("a"),
            stream_tx: plumbing.stream_tx.clone(),
            cancelled: plumbing.cancelled.clone(),
        };
        assert!(
            sender
                .send(ModelCommand::<B::FamilyCommand>::from_chat(command))
                .is_ok()
        );
        let mut chunks = Vec::new();
        for _ in 0..32 {
            assert!(matches!(state.drive(&mut receiver), LoopControl::Continue));
            while let Ok(chunk) = rx.try_recv() {
                chunks.push(chunk);
            }
            let Some(&seq) = state.owner_sequences.get("a") else {
                continue;
            };
            if state.turn_mut(seq).is_some_and(|turn| match at {
                StopAt::FirstToken => turn.token_history.len() > turn.prompt_tokens as usize,
                _ => turn.payload.generated_tokens.len() >= 3,
            }) {
                break;
            }
        }
        let seq = state.owner_sequences["a"];
        let governor = state.memory_governor();
        let model = state.inner.memory_model();
        let (now, epoch) = (governor.now(), governor.epoch(model));
        {
            let turn = state.turn_mut(seq).expect("A runs");
            if at == StopAt::FirstToken {
                assert!(turn.payload.generated_tokens.is_empty());
                assert_eq!(turn.token_history.len(), turn.prompt_tokens as usize + 1);
            } else {
                assert!(turn.payload.generated_tokens.len() >= 3);
                turn.resource_wait.retry_at = now + 1_000.0;
                turn.resource_wait.epoch = epoch;
            }
            if at == StopAt::Expired {
                turn.resource_wait.pending = false;
                turn.resource_wait.deadline = now + 1.0;
                turn.resource_wait.earlier_lane_work = 0.0;
            }
        }
        if at == StopAt::Expired {
            io.with(|state| state.now = now + 100.0);
        }
        plumbing.handle.stop();
        for _ in 0..4 {
            if state.turn_mut(seq).is_none() {
                break;
            }
            assert!(matches!(state.drive(&mut receiver), LoopControl::Continue));
        }
        while let Ok(chunk) = rx.try_recv() {
            chunks.push(chunk);
        }
        drop(plumbing);
        let done = chunks
            .into_iter()
            .map(|chunk| {
                chunk.unwrap_or_else(|error| panic!("stopped turn failed: {}", error.reason))
            })
            .find(|chunk| chunk.done)
            .expect("the stopped turn sent no done chunk");
        assert_eq!(done.finish_reason.as_deref(), Some("stop"), "{at:?}");
        if at == StopAt::FirstToken {
            assert_eq!(done.num_tokens, Some(1));
        } else {
            assert!(done.num_tokens.is_some_and(|tokens| tokens >= 3), "{at:?}");
        }
        assert_eq!(
            state.owner_sequences.get("a"),
            Some(&seq),
            "{at:?}: the stop completes and parks its owner"
        );
    }

    /// Mutation RS1 (the ledger's reap fails a stopped running turn as a
    /// cancel) → red (the cancel error); RS2 (the step plans no stopped row
    /// whose wait has not reached its retry) → red (no done chunk); FS1 (the
    /// wait of a stopped turn still times out) → red (`resource_timeout`);
    /// FT1 (the reap counts only decoded tokens) → red (the cancel error at
    /// the first token).
    #[cfg(target_os = "macos")]
    #[test]
    fn a_stopped_running_turn_ends_in_its_qwen35_step() {
        use crate::engine::governor::governor::tests::{G, model};
        if !crate::models::qwen3_5::dflash2_decode::tests::metal_is_default_device() {
            return;
        }
        for at in [StopAt::Waiting, StopAt::Expired, StopAt::FirstToken] {
            let mut inner = crate::models::qwen3_5::model::scheduled_mtp::seeded_inner(0x5730_0004);
            inner.tokenizer = Some(vacated_stop_tokenizer());
            let (io, governor) = roomy_governor();
            inner.memory_registration =
                Some(governor.register_model(model(16 * G, 0)).expect("register"));
            a_stopped_running_turn_ends_in_its_step(
                HybridSchedulerState::new(inner).expect("scheduler"),
                &io,
                at,
            );
        }
    }

    /// Mutation RS1, RS2, FS1, FT1: as the dense test → red.
    #[cfg(target_os = "macos")]
    #[test]
    fn a_stopped_running_turn_ends_in_its_qwen35_moe_step() {
        use crate::engine::governor::governor::tests::{G, model};
        if !crate::models::qwen3_5::dflash2_decode::tests::metal_is_default_device() {
            return;
        }
        for at in [StopAt::Waiting, StopAt::Expired, StopAt::FirstToken] {
            let mut inner =
                crate::models::qwen3_5_moe::model::scheduled_mtp::seeded_inner(0x5730_0005);
            inner.tokenizer = Some(vacated_stop_tokenizer());
            let (io, governor) = roomy_governor();
            inner.memory_registration =
                Some(governor.register_model(model(16 * G, 0)).expect("register"));
            a_stopped_running_turn_ends_in_its_step(
                HybridSchedulerState::new(inner).expect("scheduler"),
                &io,
                at,
            );
        }
    }

    /// Splash restarts on `metal_out_of_memory` however the request ended: a
    /// stopped turn whose prefill slice fails with a GPU OOM keeps the
    /// failure that `settle_step` reads for the restart; any other slice
    /// failure of the stopped turn ends it `Stop`.
    /// Mutation OS1 (the stop vacates an OOM slice) → red.
    #[cfg(target_os = "macos")]
    #[test]
    fn a_stopped_slice_keeps_its_gpu_oom() {
        let Some((mut state, _)) = growable_qwen35_state() else {
            return;
        };
        let oom = Error::from_reason(
            "[METAL] Command buffer execution failed: Insufficient Memory \
             (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory).",
        );
        for (error, vacates) in [
            (oom, false),
            (Error::from_reason("prefill cancelled"), true),
        ] {
            let (plumbing, _rx) = crate::engine::napi_glue::chat_stream_plumbing();
            let mut turn = decoding_turn(&mut state, 1);
            turn.payload.response = ScheduledReply::Stream(plumbing.stream_tx.clone());
            turn.payload.generated_tokens.push(3);
            plumbing.handle.stop();
            assert_eq!(
                failed_slice_vacates(&mut turn, &error),
                vacates,
                "{}",
                error.reason
            );
            assert_eq!(turn.payload.vacated_stop, vacates, "{}", error.reason);
        }
    }

    /// Only Qwen3.5 dense and MoE plan the measured prefill cut; every other
    /// family keeps the fixed step budget.
    /// Mutation M1: `new` never applies the hook → red (dense false).
    #[test]
    fn only_qwen35_measures_the_prefill_cut() {
        let dense = Qwen35Inner::new(tiny_config()).expect("construct tiny dense model");
        let dense = HybridSchedulerState::new(dense).expect("dense scheduler");
        assert!(dense.scheduler.measures_prefill_cut());
        let moe = Qwen35MoeInner::new(tiny_moe_config()).expect("construct tiny MoE model");
        let moe = HybridSchedulerState::new(moe).expect("MoE scheduler");
        assert!(moe.scheduler.measures_prefill_cut());
        if let Ok(nemotron) = NemotronHInner::new(tiny_nemotron_paged_config()) {
            let nemotron = HybridSchedulerState::new(nemotron).expect("nemotron scheduler");
            assert!(!nemotron.scheduler.measures_prefill_cut());
        }
    }

    /// Splash `Engine::Engine` refuses a negative or non-finite decode
    /// share and defaults it to 0.5; the share a load takes is the one its
    /// measured cut owes decoders.
    /// Mutation D2: the dense hook ignores the load's share → red (0.5).
    #[test]
    fn a_load_decode_share_reaches_the_cut() {
        use crate::models::qwen3_5::load_plan::decode_share_option;
        assert_eq!(
            decode_share_option("Qwen3.5", None).expect("unset"),
            engine::scheduler::DEFAULT_DECODE_SHARE
        );
        for refused in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(
                decode_share_option("Qwen3.5", Some(refused)).is_err(),
                "{refused}"
            );
        }
        let share = decode_share_option("Qwen3.5", Some(0.0)).expect("zero");
        let mut dense = Qwen35Inner::new(tiny_config()).expect("construct tiny dense model");
        dense.decode_share = share;
        let cut = dense.measured_prefill_cut().expect("dense measures");
        assert_eq!(cut.decode_share, 0.0);
        let mut moe = Qwen35MoeInner::new(tiny_moe_config()).expect("construct tiny MoE model");
        moe.decode_share = share;
        let cut = moe.measured_prefill_cut().expect("MoE measures");
        assert_eq!(cut.decode_share, 0.0);
    }

    /// Splash `Engine::workEndOf`: a prefill row claims to its own end, cut
    /// or past a pinned break, not to the next break.
    /// Mutation W2: the prefill claim ends at the next pinned break → red
    /// (2048 and 2048).
    #[test]
    fn a_prefill_claim_ends_at_its_row_end() {
        let prefill = TurnState::new(1, vec![1; 8192], 512, vec![2048, 4096], None, ())
            .expect("prefill turn");
        let row = |kind, token_start, num_tokens| engine::scheduler::StepRow {
            seq_id: 1,
            kind,
            token_start,
            num_tokens,
            cancel_snapshot: false,
        };
        assert_eq!(
            row_work_end(&row(StepKind::Prefill, 512, 512), &prefill),
            1024
        );
        assert_eq!(
            row_work_end(&row(StepKind::Prefill, 512, 2048), &prefill),
            2560
        );
        let mut decoder =
            TurnState::new(2, vec![1; 16], 16, Vec::new(), None, ()).expect("decoder");
        decoder.token_history.push(3);
        decoder.num_tokens = 17;
        decoder.decode_draft_allowance = 4;
        assert_eq!(row_work_end(&row(StepKind::Decode, 16, 5), &decoder), 21);
    }

    /// The executor's step result carries what it ran.
    #[cfg(target_os = "macos")]
    struct Recorded<'a, B: HybridSchedulerBackend> {
        executor: HybridStepExecutor<'a, B>,
        result: Option<StepResult>,
    }

    #[cfg(target_os = "macos")]
    impl<B: HybridSchedulerBackend> StepExecutor<ScheduledTurn<B::PrefixState>> for Recorded<'_, B> {
        type Error = Error;

        fn execute(
            &mut self,
            plan: &StepPlan,
            running: &mut [TurnState<ScheduledTurn<B::PrefixState>>],
        ) -> std::result::Result<StepResult, Self::Error> {
            let result = self.executor.execute(plan, running)?;
            self.result = Some(result.clone());
            Ok(result)
        }
    }

    /// One step of `scheduler` on `inner`'s executor: its plan and result.
    #[cfg(target_os = "macos")]
    fn timed_step<B: HybridSchedulerBackend>(
        scheduler: &mut Scheduler<ScheduledTurn<B::PrefixState>, (), ()>,
        inner: &mut B,
    ) -> (StepPlan, StepResult) {
        let mut recorded = Recorded {
            executor: HybridStepExecutor::new(inner),
            result: None,
        };
        let SchedulerAction::Stepped { plan, .. } = scheduler.drive_once(&mut recorded).unwrap()
        else {
            panic!("expected a step");
        };
        (plan, recorded.result.expect("the step ran"))
    }

    /// The step's prefill phase wall covers its prefill rows and its decode
    /// phase wall its decode rows, and it times the measured unit; a
    /// decode-only step has no prefill wall. No decode share, so the unit
    /// owes the first lane no decode-only step.
    /// Mutation T1m: the prefill wall is left 0 → red.
    #[cfg(target_os = "macos")]
    #[test]
    fn the_step_times_its_prefill_phase() {
        use crate::models::gemma4::dspark_decode::tests::seeded_tiny_paged_inner_with_draft;
        let mut inner =
            seeded_tiny_paged_inner_with_draft(1729).expect("the tiny paged gemma4 fixture");
        let mut scheduler = Scheduler::<_, (), ()>::new(2, 12)
            .unwrap()
            .with_measured_prefill_cut(MeasuredConfig {
                native_window: 262_144,
                decode_share: 0.0,
            });
        let first = speculative::tests::request(
            &mut inner,
            41,
            (0..7).collect(),
            false,
            Arc::new(AtomicBool::new(false)),
        );
        scheduler.enqueue_turn(first).unwrap();
        let (plan, _) = timed_step(&mut scheduler, &mut inner);
        assert_eq!(plan.rows.len(), 1);
        assert_eq!(plan.rows[0].kind, StepKind::Prefill);
        assert_eq!(scheduler.measured_rows(), Some(7));
        let second = speculative::tests::request(
            &mut inner,
            42,
            (4..13).collect(),
            false,
            Arc::new(AtomicBool::new(false)),
        );
        scheduler.enqueue_turn(second).unwrap();
        let (plan, result) = timed_step(&mut scheduler, &mut inner);
        let kinds: Vec<_> = plan.rows.iter().map(|row| (row.seq_id, row.kind)).collect();
        assert_eq!(kinds, vec![(41, StepKind::Decode), (42, StepKind::Prefill)]);
        assert!(result.rows[1].prefill_micros > 0, "{result:?}");
        assert!(result.prefill_wall_micros >= result.rows[1].prefill_micros);
        assert!(result.decode_wall_micros > 0);
        assert!(plan.contended);
        assert_eq!(scheduler.measured_rows(), Some(9));
        let (plan, result) = timed_step(&mut scheduler, &mut inner);
        assert!(plan.rows.iter().all(|row| row.kind == StepKind::Decode));
        assert_eq!(result.prefill_wall_micros, 0);
        assert!(result.decode_wall_micros > 0);
    }

    /// Every planned row runs; the prefill rows take `0` µs each, which is
    /// also the step's prefill phase wall.
    #[cfg(target_os = "macos")]
    struct PrefillWall(u64);

    #[cfg(target_os = "macos")]
    impl<P> StepExecutor<P> for PrefillWall {
        type Error = Error;

        fn execute(
            &mut self,
            plan: &StepPlan,
            running: &mut [TurnState<P>],
        ) -> std::result::Result<StepResult, Self::Error> {
            let mut result = RunEveryRow.execute(plan, running)?;
            for (row, output) in plan.rows.iter().zip(&mut result.rows) {
                if row.kind == StepKind::Prefill {
                    output.prefill_micros = self.0;
                    result.prefill_wall_micros = self.0;
                }
            }
            Ok(result)
        }
    }

    /// A ledger state with a running prefill of `prompt_tokens`, none of
    /// them computed.
    #[cfg(target_os = "macos")]
    fn state_prefilling(
        prompt_tokens: u32,
    ) -> (HybridSchedulerState<Qwen35Inner>, &'static Governor) {
        use crate::engine::governor::governor::tests::G;
        let (mut state, governor, _io) = ledger_state(16 * G, 0).expect("a ledger state");
        let mut lead = speculative::tests::request(
            &mut state.inner,
            1,
            tokens_of(1, 1),
            false,
            Arc::new(AtomicBool::new(false)),
        );
        lead.prompt_tokens = prompt_tokens;
        lead.num_tokens = prompt_tokens;
        state.scheduler.reinstate_running(lead);
        (state, governor)
    }

    /// Splash's batch completion adds a prefill command's wall to every lane
    /// outside it: the step's prefill phase reaches the decoding lane's gap
    /// clock, not the prefilling lane's.
    /// Mutation G2: the pass keeps no prefill wall → red (0 s).
    #[cfg(target_os = "macos")]
    #[test]
    fn a_step_prefill_wall_reaches_the_lanes_beside_it() {
        use crate::engine::governor::governor::tests::G;
        let (mut state, _governor, _io) = ledger_state(16 * G, 0).expect("a ledger state");
        let decoder = decoding_turn(&mut state, 1);
        state.scheduler.reinstate_running(decoder);
        let prefill = speculative::tests::request(
            &mut state.inner,
            2,
            tokens_of(2, 1),
            false,
            Arc::new(AtomicBool::new(false)),
        );
        state.scheduler.reinstate_running(prefill);
        let mut step = state.step_with(&mut PrefillWall(600_000)).expect("a step");
        let SchedulerAction::Stepped { plan, .. } = &step else {
            panic!("expected a step");
        };
        let kinds: Vec<_> = plan.rows.iter().map(|row| (row.seq_id, row.kind)).collect();
        assert_eq!(kinds, vec![(1, StepKind::Decode), (2, StepKind::Prefill)]);
        state.close_step_pass(&mut step);
        let gap = |seq_id: SeqId| {
            state
                .scheduler
                .running()
                .iter()
                .find(|turn| turn.seq_id == seq_id)
                .map(|turn| turn.payload.clock.gap.prefill_seconds())
        };
        assert_eq!(gap(1), Some(0.6));
        assert_eq!(gap(2), Some(0.0));
    }

    /// Splash `Scheduler::prefillAdmissionOrder` gates admission: a 50,000
    /// token start beside a 40,000 token prefill would join a unit it does
    /// not end and then wait out the other's prompt, so it stays prepared
    /// and takes no memory.
    /// Mutation S3: admission skips the gate → red (the start is tried).
    #[cfg(target_os = "macos")]
    #[test]
    fn admission_holds_a_start_its_unit_would_not_serve() {
        let (mut state, _governor) = state_prefilling(40_000);
        let (prepared, _reply) = prepared_turn(&mut state, 2, vec![1; 50_000]);
        let Err(prepared) = state.admit_prepared(prepared) else {
            panic!("the start was tried");
        };
        assert!(prepared.claim.is_none());
    }

    /// Splash `Engine::admitQueued` queues a start the scheduler does not
    /// want, and it holds back nothing: beside a 40,000 token prefill a held
    /// 50,000 token start leaves a later short start to be admitted at once.
    /// Mutation AQ1: a held start is kept as a refused one → red (deferred).
    #[cfg(target_os = "macos")]
    #[test]
    fn a_held_start_holds_back_no_later_start() {
        let (mut state, _governor) = state_prefilling(40_000);
        let (long, _long_reply) = prepared_turn(&mut state, 2, vec![1; 50_000]);
        assert!(!state.start_prepared(long));
        let (short, _short_reply) = prepared_turn(&mut state, 3, tokens_of(3, 1));
        assert!(!state.start_prepared(short));
        assert!(state.prepared_waiting.is_none());
        assert_eq!(
            state
                .held
                .iter()
                .map(|held| held.seq_id)
                .collect::<Vec<_>>(),
            vec![2]
        );
        assert!(state.scheduler.contains_seq(3));
    }

    /// Splash `Scheduler::complete` overtakes a queued request with each
    /// prefill unit that serves a later submission; once overdue it leads,
    /// so a held start is not starved by the short starts that pass it.
    /// Mutation AQ2: held starts are not overtaken → red (0).
    #[cfg(target_os = "macos")]
    #[test]
    fn a_passed_held_start_turns_overdue() {
        let (mut state, _governor) = state_prefilling(40_000);
        let (long, _long_reply) = prepared_turn(&mut state, 2, vec![1; 50_000]);
        assert!(!state.start_prepared(long));
        let (short, _short_reply) = prepared_turn(&mut state, 3, tokens_of(3, 1));
        assert!(!state.start_prepared(short));
        let step = state.step_with(&mut PrefillWall(1_000)).expect("a step");
        assert!(matches!(step, SchedulerAction::Stepped { .. }));
        state.count_held_overtakes();
        let held = state.held.front().expect("held");
        assert_eq!(held.overtaken, 1);
        assert!(state.admission_waits(held));
        state.held[0].overtaken = engine::scheduler::MAX_OVERTAKES;
        assert!(!state.admission_waits(&state.held[0]));
    }

    /// Splash ranks a candidate by its cached tokens: a live continuation
    /// that keeps 32 of its 2,050 tokens has fewer left than a 2,047 token
    /// prefill, so it leads and is served at once.
    /// Mutation H1: the kept tokens are not counted → red (it waits).
    #[cfg(target_os = "macos")]
    #[test]
    fn a_live_continuation_is_ranked_by_what_it_keeps() {
        use crate::engine::governor::governor::tests::G;
        let (mut state, _governor, _io) = ledger_state(16 * G, 0).expect("a ledger state");
        let held = tokens_of(2, 2);
        let adapter = state.inner.paged_adapter_mut().expect("a paged adapter");
        adapter.begin_request(2).expect("begin keep-live request");
        adapter
            .find_cached_prefix(&[], &[], 0, false)
            .expect("prefix lookup");
        adapter
            .allocate_suffix_blocks(held.len() as u32)
            .expect("allocate held blocks");
        adapter.record_tokens(&held).expect("record held tokens");
        adapter
            .finalize_turn_keep_live(&[], 0)
            .expect("keep-live finalize");
        let mut lead = speculative::tests::request(
            &mut state.inner,
            1,
            tokens_of(1, 1),
            false,
            Arc::new(AtomicBool::new(false)),
        );
        lead.prompt_tokens = 2_047;
        lead.num_tokens = 2_047;
        state.scheduler.reinstate_running(lead);
        let mut prompt = held;
        prompt.extend((0..2_018).map(|token| token % 15 + 1));
        let (mut prepared, _reply) = prepared_turn(&mut state, 2, prompt);
        prepared.admitted.params.reuse_cache = true;
        assert!(!state.admission_waits(&prepared));
    }

    /// Splash `Engine::admitQueued` queues a start the scheduler does not
    /// want (`deferWait`): its claim stops waiting, and its wait keeps only
    /// when it began.
    /// Mutation D3: the held start keeps its wait → red (claim waiting).
    #[cfg(target_os = "macos")]
    #[test]
    fn a_held_start_stops_its_memory_wait() {
        let (mut state, governor) = state_prefilling(40_000);
        let (mut prepared, _reply) = prepared_turn(&mut state, 2, vec![1; 50_000]);
        let claim = governor.open(
            crate::engine::governor::governor::whole_turn_spec(0),
            state.inner.memory_model(),
        );
        let id = claim.id();
        governor.ledger(|ledger| ledger.set_waiting(id, true));
        prepared.claim = Some(claim);
        prepared.resource_wait = ResourceWait {
            reason: WaitReason::Memory,
            started: Some(1.0),
            pending: true,
            pending_since: Some(1.0),
            retry_at: 5.0,
            deadline: 31.0,
            ..ResourceWait::default()
        };
        let Err(prepared) = state.admit_prepared(prepared) else {
            panic!("the start was tried");
        };
        let waiting = governor.ledger(|ledger| ledger.claim(id).map(|state| state.waiting));
        assert_eq!(waiting, Some(false));
        assert_eq!(
            prepared.resource_wait,
            ResourceWait {
                started: Some(1.0),
                pending_since: Some(1.0),
                ..ResourceWait::default()
            }
        );
    }

    /// Splash `Engine::deferWait` resets a held start's allocation failure,
    /// so `Engine::syncClaims` no longer counts it: a start the host refused
    /// leaves the host deficit once held.
    /// Mutation HD1: the held start keeps its host refusal → red (1 G).
    #[cfg(target_os = "macos")]
    #[test]
    fn a_held_start_leaves_the_host_deficit() {
        use crate::engine::governor::governor::tests::G;
        let (mut state, governor) = state_prefilling(40_000);
        let (mut prepared, _reply) = prepared_turn(&mut state, 2, vec![1; 50_000]);
        let claim = governor.open(
            crate::engine::governor::governor::whole_turn_spec(0),
            state.inner.memory_model(),
        );
        let id = claim.id();
        governor.ledger(|ledger| {
            ledger.refused(
                id,
                crate::engine::governor::ledger::Need {
                    bytes: G,
                    host_refused: true,
                    ..Default::default()
                },
            );
            ledger.set_host_refused_waiting(id, true);
        });
        assert_eq!(
            governor.ledger(|ledger| ledger.host_refused_waiting_bytes()),
            G
        );
        prepared.claim = Some(claim);
        let Err(prepared) = state.admit_prepared(prepared) else {
            panic!("the start was tried");
        };
        assert!(prepared.held);
        assert_eq!(
            governor.ledger(|ledger| ledger.host_refused_waiting_bytes()),
            0
        );
    }

    /// Splash `Scheduler::snapshot` counts the requests not yet started: a
    /// prepared start is queued, and waits for resources once refused.
    /// Mutation P1: the prepared start is left out → red.
    #[cfg(target_os = "macos")]
    #[test]
    fn the_prepared_start_is_in_the_gauges() {
        let (mut state, _governor) = state_prefilling(40_000);
        let counters = state.scheduler.counters();
        let (prepared, _reply) = prepared_turn(&mut state, 2, tokens_of(2, 1));
        state.prepared_waiting = Some(prepared);
        let gauges = |state: &HybridSchedulerState<Qwen35Inner>| {
            state.publish_gauges();
            let snapshot = counters.snapshot();
            (
                snapshot.queued,
                snapshot.waiting_resources,
                snapshot.prefilling,
            )
        };
        assert_eq!(gauges(&state), (1, 0, 1));
        state
            .prepared_waiting
            .as_mut()
            .expect("prepared")
            .resource_wait
            .reason = WaitReason::Memory;
        assert_eq!(gauges(&state), (0, 1, 1));
    }

    /// A model on the governor hands it its scheduler's counters when its
    /// scheduler is built.
    /// Mutation R1: the scheduler is not registered → red (none).
    #[cfg(target_os = "macos")]
    #[test]
    fn a_registered_model_publishes_its_scheduler() {
        use crate::engine::governor::governor::tests::{FakeIo, G, host, model, stats};
        let io = FakeIo::new(
            stats(0),
            host(32 * G, crate::engine::governor::host::Pressure::Normal),
        );
        let governor: &'static Governor = Box::leak(Box::new(Governor::new(io)));
        let mut inner = Qwen35Inner::new(tiny_config()).expect("construct tiny dense model");
        inner.memory_registration =
            Some(governor.register_model(model(16 * G, 0)).expect("register"));
        let model_id = inner.memory_model();
        let _state = HybridSchedulerState::new(inner).expect("construct Qwen3.5 scheduler");
        let schedulers = governor.schedulers();
        assert_eq!(schedulers.len(), 1);
        assert_eq!(schedulers[0].model_id, model_id);
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
        assert!(
            crate::engine::scheduler::clamp_scheduled_output_tokens(33_611, 1024, cap, true)
                .is_ok()
        );
        assert!(
            crate::engine::scheduler::clamp_scheduled_output_tokens(33_611, 1024, 32_768, true)
                .is_err()
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
        assert!(
            crate::engine::scheduler::clamp_scheduled_output_tokens(33_611, 1024, cap, false)
                .is_ok()
        );
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
