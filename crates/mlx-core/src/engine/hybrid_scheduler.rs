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
    ChunkSink, PagedBackend, PagedPrefix, StreamEmitter, ThinkingSetup, TurnOutput,
    TurnTokenObserver,
};
use crate::engine::cmd::{ChatCmd, handle_chat_cmd};
use crate::engine::model_command::{FamilyCommand, ModelCommand, handle_model_command};
use crate::engine::reclaim::{
    Freed, KvNeed, KvReclaim, MetBefore, Reach, Unit, UnitKey, free_for_kv,
};
use crate::engine::scheduler::{
    BlockTelemetry, PreemptedTurn, PreemptionMode, PreemptionReplay, RowStepResult, Scheduler,
    SchedulerAction, SchedulerError, StepExecutor, StepKind, StepPlan, StepResult, TurnState,
    install_preemption_replay, is_paged_allocation_blocked,
};
use crate::engine::types::{ChatConfig, ChatResult, ChatStreamChunk};
use crate::model_thread::{LoopControl, ResponseTx, StreamTx};
use crate::sampling::{check_repetition_cutoff, sample};
use crate::stream::{DeviceType, Stream, StreamContext, WiredLimitContext};
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

    /// Optional independently admitted residency bound for scheduled turns.
    /// Families opt in explicitly; ordinary turn policies remain unchanged.
    fn scheduler_wired_bytes(&self) -> Option<usize> {
        None
    }
    /// `seq_id` was admitted; called before the turn's cancel flag, wired
    /// context and prefill slices are read, so a backend that changes its
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
    /// The family takes a refused KV reservation's units in one stamp order
    /// (`crate::engine::reclaim`), at the pool's cap too.
    fn reclaims_in_order(&self) -> bool {
        false
    }
    /// Moves an idle live state beside the pool where the take order sees it.
    fn park_idle_live_state(&mut self) {}
    fn sync_state_marks(&mut self) {}
    /// Bytes every state unit could free (an upper bound); the states turns
    /// use count only with `in_use`.
    fn state_reach_bytes(&self, _in_use: bool) -> u64 {
        0
    }
    fn compact_one_state(&mut self) -> bool {
        false
    }
    fn oldest_state_unit(
        &self,
        _at_cap: bool,
        _skip: &HashSet<UnitKey>,
        _in_use: bool,
    ) -> Option<Unit> {
        None
    }
    /// Takes one state unit; false when it freed nothing.
    fn take_state_unit(&mut self, _key: &UnitKey) -> bool {
        false
    }
    /// The take order for `need`, keeping `keep`'s own entries.
    fn reclaim_kv(&mut self, _need: KvNeed, _keep: &[u32]) -> Freed {
        Freed::Short
    }
    /// Marked cache-only blocks `prompt` hits: its admission is charged no
    /// block for them.
    fn marked_hit_blocks(&self, _prompt: &[u32]) -> u32 {
        0
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
        !scheduler_reserve_full_isl()
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

/// Blocks free plus evictable blocks lack for `need_bytes` at the pool's
/// cap, where only a reclaim adds takeable blocks. `None` below the cap or
/// when nothing is short.
fn cap_shortfall_blocks(snapshot: &SchedulerCacheSnapshot, need_bytes: u64) -> Option<u32> {
    if snapshot.blocks.total_blocks < snapshot.max_total_blocks {
        return None;
    }
    let bytes_per_block = snapshot.bytes_per_block.max(1);
    let available = u64::from(snapshot.blocks.free_blocks)
        .saturating_add(u64::from(snapshot.blocks.reclaimable_blocks))
        .saturating_mul(bytes_per_block);
    let short = need_bytes
        .saturating_sub(available)
        .div_ceil(bytes_per_block);
    (short > 0).then(|| short.min(u64::from(u32::MAX)) as u32)
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

/// Largest turn length a lone speculative turn can reach in
/// `usable_pool_tokens` when its draft state, held outside the pool, grows
/// with every row: its blocks plus `held(T)` fit the usable pool's bytes.
pub(crate) fn speculative_window_tokens(
    usable_pool_tokens: u32,
    block_size: u32,
    bytes_per_block: u64,
    held: impl Fn(u32) -> u64,
) -> u32 {
    if bytes_per_block == 0 || block_size == 0 {
        return usable_pool_tokens.max(1);
    }
    let budget = u64::from(usable_pool_tokens / block_size).saturating_mul(bytes_per_block);
    let fits = |tokens: u32| {
        u64::from(tokens.div_ceil(block_size))
            .saturating_mul(bytes_per_block)
            .saturating_add(held(tokens))
            <= budget
    };
    let (mut low, mut high) = (0u32, usable_pool_tokens);
    while low < high {
        let mid = low + (high - low).div_ceil(2);
        if fits(mid) {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    low.max(1)
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
    pub(crate) _wired_context: Option<WiredLimitContext>,
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
    /// Token bound the off-pool state reservation was sized for. The block
    /// reservation may cover fewer tokens (`MLX_SCHED_RESERVE_FULL_ISL=0`),
    /// so a resume must not derive the bound from it.
    pub(crate) requested_tokens: u32,
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
                    turn.payload.reasoning_tracker.force_think_end_pending(),
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

/// The scheduler side of the take order for a blocked step: the family's
/// states beside the idle owners, each by its own stamp.
struct SchedReclaim<'a, B: HybridSchedulerBackend> {
    state: &'a mut HybridSchedulerState<B>,
    victim: SeqId,
}

impl<B: HybridSchedulerBackend> KvReclaim for SchedReclaim<'_, B> {
    fn at_cap(&self) -> bool {
        self.state
            .inner
            .paged_adapter()
            .is_none_or(PagedKVCacheAdapter::at_cap)
    }

    fn takeable(&self) -> u32 {
        self.state
            .inner
            .paged_adapter()
            .map_or(0, PagedKVCacheAdapter::takeable_blocks)
    }

    fn ungrown(&self) -> u32 {
        self.state
            .inner
            .paged_adapter()
            .map_or(0, PagedKVCacheAdapter::ungrown_blocks)
    }

    fn bytes_per_block(&self) -> u64 {
        self.state
            .inner
            .paged_adapter()
            .and_then(|adapter| adapter.bytes_per_block().ok())
            .unwrap_or(0)
    }

    fn met(&mut self, need: KvNeed) -> bool {
        self.state
            .inner
            .paged_adapter_mut()
            .is_some_and(|adapter| adapter.kv_need_met(need.blocks, need.over_takeable))
    }

    fn grow_room(&self) -> u32 {
        self.state
            .inner
            .paged_adapter()
            .map_or(0, PagedKVCacheAdapter::last_grow_room)
    }

    fn before_count(&mut self, at_cap: bool) {
        if !at_cap {
            self.state.inner.park_idle_live_state();
        }
    }

    fn sync_marks(&mut self) {
        self.state.inner.sync_state_marks();
    }

    fn reach(&self, at_cap: bool, in_use: bool) -> Reach {
        let owners = self.state.idle_owners(self.victim);
        let adapter = self.state.inner.paged_adapter();
        let releasable =
            adapter.map_or(0, |adapter| adapter.releasable_blocks_of(&owners).len()) as u64;
        let rows = (owners
            .iter()
            .filter(|&&seq_id| self.state.inner.has_scheduled_recurrent(seq_id))
            .count() as u64)
            .saturating_mul(self.state.inner.recurrent_state_bytes());
        Reach {
            blocks: u64::from(adapter.map_or(0, PagedKVCacheAdapter::marked_blocks))
                .saturating_add(releasable),
            bytes: if at_cap {
                0
            } else {
                self.state
                    .inner
                    .state_reach_bytes(in_use)
                    .saturating_add(rows)
            },
        }
    }

    fn compact_one(&mut self) -> bool {
        self.state.inner.compact_one_state()
    }

    fn oldest_state(&self, at_cap: bool, skip: &HashSet<UnitKey>, in_use: bool) -> Option<Unit> {
        self.state.inner.oldest_state_unit(at_cap, skip, in_use)
    }

    fn oldest_kv(&self, at_cap: bool, skip: &HashSet<UnitKey>) -> Option<Unit> {
        self.state.oldest_idle_owner(self.victim, at_cap, skip)
    }

    fn take(&mut self, key: &UnitKey) -> bool {
        match key {
            UnitKey::Owner(seq_id) => self.state.release_owner(*seq_id),
            _ => self.state.inner.take_state_unit(key),
        }
    }
}

pub(crate) struct HybridSchedulerState<B: HybridSchedulerBackend> {
    pub(crate) inner: B,
    enabled: bool,
    scheduler: Scheduler<
        ScheduledTurn<B::PrefixState>,
        ModelCommand<B::FamilyCommand>,
        ModelCommand<B::FamilyCommand>,
    >,
    pending: VecDeque<ModelCommand<B::FamilyCommand>>,
    prepared_waiting: Option<Box<PreparedTurn>>,
    pending_restores: HashMap<SeqId, B::RestoreTicket>,
    pub(crate) owner_sequences: HashMap<String, SeqId>,
    pub(crate) owner_states: HashMap<String, B::OwnerState>,
    /// Turn-end stamp of each idle owner's kept table, on the model's one
    /// cache clock.
    owner_last_used: HashMap<SeqId, u64>,
    /// Splash `Engine::allocate`'s met list for the blocked step's retries.
    met_before: MetBefore<(SeqId, u32)>,
    /// The prompt an admission grow keeps (Splash `VictimQuery::kept`).
    reclaim_keep: Vec<u32>,
    next_seq_id: SeqId,
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
            owner_last_used: HashMap::new(),
            met_before: MetBefore::default(),
            reclaim_keep: Vec::new(),
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
                self.pending.insert(index, ModelCommand::from_chat(chat));
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
        }))
    }

    /// Tokens one scheduled turn can use in the max pool after recurrent
    /// state. An on-demand speculative turn's draft state grows outside the
    /// pool with every row and each step must fit it, so its window also
    /// pays the draft state it holds at its last row.
    fn scheduled_turn_window(&self, block_size: u32, speculative: bool) -> u32 {
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
        let usable_pool = pool_tokens_after_recurrent(
            pool_tokens,
            block_size,
            bytes_per_block,
            self.inner.recurrent_state_bytes(),
        );
        if !(speculative && self.inner.scheduler_books_on_demand()) {
            return usable_pool;
        }
        speculative_window_tokens(usable_pool, block_size, bytes_per_block, |tokens| {
            self.inner.scheduled_draft_state_held_bytes(tokens)
        })
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
        let context = scheduled_turn_context(
            trained_context,
            self.scheduled_turn_window(block_size, speculative),
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

    /// The idle owner with the oldest turn-end stamp that frees something: a
    /// block at the cap, a block or a parked row below it. A block counts
    /// when releasing every idle owner would free it. An owner the
    /// scheduler never finished goes by its table's turn-end stamp.
    fn oldest_idle_owner(
        &self,
        victim: SeqId,
        at_cap: bool,
        skip: &HashSet<UnitKey>,
    ) -> Option<Unit> {
        let adapter = self.inner.paged_adapter();
        let owners = self.idle_owners(victim);
        let freed = adapter
            .map(|adapter| adapter.releasable_blocks_of(&owners))
            .unwrap_or_default();
        owners
            .into_iter()
            .filter(|&seq_id| !skip.contains(&UnitKey::Owner(seq_id)))
            .filter(|&seq_id| {
                adapter.is_some_and(|adapter| adapter.table_holds_any(seq_id, &freed))
                    || (!at_cap && self.inner.has_scheduled_recurrent(seq_id))
            })
            .map(|seq_id| {
                let stamp = self
                    .owner_last_used
                    .get(&seq_id)
                    .copied()
                    .or_else(|| adapter.and_then(|adapter| adapter.turn_end_stamp(seq_id)))
                    .unwrap_or(0);
                (stamp, seq_id)
            })
            .min()
            .map(|(stamp, seq_id)| Unit {
                stamp,
                key: UnitKey::Owner(seq_id),
            })
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

    /// Frees memory for a blocked step without preempting `victim`. A
    /// family that reclaims in order sends the step's refused KV need
    /// through the take order (Splash `MemoryLedger::free`): met in place, a
    /// count that cannot cover it takes nothing, else states and idle owners
    /// go oldest first until it is met; a need met or freed for once already
    /// stops (Splash `Engine::allocate`). `reach_in_use`: the step's turn
    /// runs alone, so its need goes on to the states suspended and running
    /// turns use, last (Splash `Engine::growthClass`); a turn that yields
    /// beside another keeps them. A blocked step that left no KV refusal,
    /// and every other family, releases idle owners.
    fn reclaim_for_blocked_step(&mut self, victim: SeqId, reach_in_use: bool) -> bool {
        if !self.inner.reclaims_in_order() {
            return self.reclaim_idle_owners(victim);
        }
        let Some(blocks) = self
            .inner
            .paged_adapter()
            .and_then(PagedKVCacheAdapter::take_refusal)
        else {
            return self.reclaim_idle_owners(victim);
        };
        let need = (victim, blocks);
        let retry = !self.met_before.seen(&need) && {
            let freed = free_for_kv(
                &mut SchedReclaim {
                    state: self,
                    victim,
                },
                KvNeed {
                    reach_in_use,
                    ..KvNeed::in_place(blocks)
                },
            );
            self.met_before.retry(need, freed)
        };
        if !retry {
            self.met_before.clear();
        }
        retry
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

    /// Ends the step's completed turns and handles its blocked one. False
    /// when a turn blocked: no victim resumes in the same round.
    fn settle_step(
        &mut self,
        completed: Vec<TurnState<ScheduledTurn<B::PrefixState>>>,
        preempted: Option<PreemptedTurn<ScheduledTurn<B::PrefixState>>>,
        blocked_alone: Option<SeqId>,
    ) -> bool {
        for turn in completed {
            self.finish_completed(turn);
        }
        if let Some(seq_id) = blocked_alone {
            self.handle_blocked_alone(seq_id);
            return false;
        }
        let Some(preempted) = preempted else {
            self.met_before.clear();
            return true;
        };
        if self.inner.scheduler_books_on_demand()
            && self.reclaim_for_blocked_step(preempted.turn.seq_id, false)
        {
            self.scheduler.reinstate_running(preempted.turn);
        } else {
            self.handle_preempted(preempted);
        }
        false
    }

    /// The step's only running turn blocked. A family that books on demand
    /// first reclaims idle memory; with none left the turn cannot fit even
    /// alone, so it fails with `capacity_exhausted` and is released once,
    /// never preempted: a replay would block at the same step (Splash
    /// `Engine::judge` fails a lone lane). Other
    /// families book the whole turn and keep the preempt-and-resume path.
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
        if self.reclaim_for_blocked_step(seq_id, true) {
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
        let tried = if self.inner.reclaims_in_order() {
            "even alone: the parked states and idle cache owners could not free what the step \
             lacks (none is taken when their count falls short)"
        } else {
            "even alone after every idle cache owner was released"
        };
        Error::from_reason(format!(
            "capacity_exhausted: could not allocate KV or draft state for this turn's next step \
             ({pool}); the request does not fit in the memory the page pool may use, {tried}"
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
        let mut met_before = MetBefore::default();
        loop {
            let decision =
                self.reserve_memory_decision(&snapshot, reservation_blocks, candidate_state_bytes);
            let outstanding_bytes = {
                let inner = &self.inner;
                self.scheduler
                    .outstanding_scheduled_state_bytes(|seq| inner.scheduled_state_held_bytes(seq))
            };
            let state = scheduled_state_fit(
                self.inner.scheduled_state_capacity(),
                outstanding_bytes,
                state_candidate_bytes,
            );
            if state == ScheduledStateFit::Never {
                return Err(self.scheduled_state_exceeded(state_candidate_bytes));
            }
            if decision.admitted && state == ScheduledStateFit::Fits {
                return Ok(MemoryReserveOutcome::Admitted);
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
                if may_wait {
                    if self.inner.scheduler_books_on_demand()
                        && !grown_for_live_turns
                        && state == ScheduledStateFit::Fits
                        && let Some(needed) = grow_needed_blocks_for(
                            &snapshot,
                            &decision,
                            scheduler_watermark_fraction(),
                        )
                        .or_else(|| {
                            self.inner
                                .reclaims_in_order()
                                .then(|| {
                                    cap_shortfall_blocks(
                                        &snapshot,
                                        decision
                                            .reserved_block_bytes
                                            .saturating_add(decision.reserved_state_bytes)
                                            .saturating_add(decision.watermark_bytes),
                                    )
                                })
                                .flatten()
                        })
                    {
                        grown_for_live_turns = true;
                        if let Some(grown) = self.try_grow_pool_by(needed, &mut met_before)? {
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
                    &mut met_before,
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
        met_before: &mut MetBefore<u32>,
    ) -> Result<Option<SchedulerCacheSnapshot>> {
        let existing_state_bytes = self.inner.scheduled_recurrent_bytes();
        let Some(needed) = grow_needed_blocks(
            snapshot,
            reservation_blocks,
            candidate_state_bytes,
            existing_state_bytes,
        )
        .or_else(|| {
            (self.inner.reclaims_in_order()
                && !exceeds_max_pool_ceiling(snapshot, reservation_blocks, candidate_state_bytes))
            .then(|| {
                cap_shortfall_blocks(
                    snapshot,
                    u64::from(reservation_blocks)
                        .saturating_mul(snapshot.bytes_per_block.max(1))
                        .saturating_add(candidate_state_bytes)
                        .saturating_add(existing_state_bytes),
                )
            })
            .flatten()
        }) else {
            return Ok(None);
        };
        self.try_grow_pool_by(needed, met_before)
    }

    /// Grows the pool by `needed` blocks. A family that reclaims in order
    /// sends the need through the take order (at the cap too, where only a
    /// take adds blocks), keeping the admitted prompt's entries; a `Short`
    /// take or a need `met_before` saw is not grown. Returns the fresh
    /// snapshot, or `None` when the pool did not grow.
    fn try_grow_pool_by(
        &mut self,
        needed: u32,
        met_before: &mut MetBefore<u32>,
    ) -> Result<Option<SchedulerCacheSnapshot>> {
        let Some(adapter) = self.inner.paged_adapter() else {
            return Ok(None);
        };
        let grown = if self.inner.reclaims_in_order() {
            let blocks = adapter.takeable_blocks().saturating_add(needed);
            !met_before.seen(&blocks) && {
                let freed = self.inner.reclaim_kv(
                    KvNeed {
                        blocks,
                        over_takeable: true,
                        checked: false,
                        reach_in_use: false,
                    },
                    &self.reclaim_keep,
                );
                met_before.retry(blocks, freed)
            }
        } else {
            self.inner
                .paged_adapter_mut()
                .is_some_and(|adapter| adapter.try_grow_pool(needed))
        };
        if grown {
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
    /// blocks it continues on and the marked chain its prompt hits.
    fn admission_charge(
        &self,
        reservation_blocks: u32,
        seq_id: SeqId,
        tokens: &[u32],
        reuse: bool,
    ) -> u32 {
        reservation_blocks
            .saturating_sub(reusable_keep_live_blocks(
                self.inner.paged_adapter(),
                seq_id,
                tokens,
                reuse,
            ))
            .saturating_sub(self.inner.marked_hit_blocks(tokens))
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
        // A preempted row resumes before any later turn books memory: a
        // booking taken now can keep the victim from ever fitting again,
        // while the victim blocks the later turn from running.
        if self.inner.scheduler_books_on_demand() && self.scheduler.has_preempted() {
            return Err(prepared);
        }
        if !self.ensure_recurrent_slot(prepared.seq_id) {
            return Err(prepared);
        }
        let mut scheduled_state_bytes = 0u64;
        if let Some(snapshot) = cache_snapshot {
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
            self.reclaim_keep.clone_from(&prepared.admitted.tokens);
            let reserved = self.try_reserve_reclaiming_idle(
                prepared.seq_id,
                charge,
                candidate_state_bytes,
                state_candidate_bytes,
                snapshot,
            );
            self.reclaim_keep.clear();
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

        let PreparedTurn {
            admitted,
            response,
            cancelled,
            owner_id,
            seq_id,
            newly_assigned,
            reservation_blocks,
            block_size,
            reservation_state_bytes,
            requested_tokens,
        } = *prepared;
        if cancelled.load(Ordering::Relaxed) {
            if newly_assigned {
                self.owner_states.remove(&owner_id);
                self.owner_sequences.remove(&owner_id);
                self.inner.release_scheduled_recurrent_for(seq_id);
                self.inner.release_scheduled_speculation(seq_id);
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
                self.inner.release_scheduled_speculation(seq_id);
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
        self.inner.scheduled_turn_opened(seq_id);
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
                self.inner.release_scheduled_speculation(seq_id);
                self.inner.scheduled_turn_closed(seq_id);
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
        let generation_start = admitted.params.report_performance.then(Instant::now);
        let wired_context = self
            .inner
            .scheduler_wired_bytes()
            .map(|bytes| WiredLimitContext::bounded(bytes, vec![generation_stream]));
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
            reasoning_tracker: engine::penalties::ReasoningTracker::from_setup_multi(
                &admitted.thinking,
                admitted.think_end_id,
                admitted.think_end_extra_ids,
            ),
            extra_eos_ids: self.inner.extra_eos_ids(),
            generation_start,
            first_token_instant: None,
            generation_stream,
            _wired_context: wired_context,
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
            allocation_failed: false,
            preemption_replay: None,
            requested_tokens,
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
        if self.inner.scheduler_books_on_demand() {
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
            }
            .saturating_sub(self.inner.marked_hit_blocks(&turn.payload.prompt_tokens));
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
            self.reclaim_keep.clone_from(&turn.payload.prompt_tokens);
            let reserved = self.try_reserve_with_policy(
                turn.seq_id,
                charge,
                turn.recurrent_state_bytes,
                state_candidate_bytes,
                snapshot,
                may_wait,
                B::release_scheduled_cache,
            );
            self.reclaim_keep.clear();
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
                self.inner.release_scheduled_speculation(turn.seq_id);
                self.scheduler.prepend_preempted(turn);
                return;
            }
            Err(error) => {
                let _ = self.inner.release_scheduled_cache(turn.seq_id);
                self.inner.release_scheduled_recurrent_for(turn.seq_id);
                self.inner.release_scheduled_speculation(turn.seq_id);
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
        // Every arm below ends the turn, the error returns included.
        self.inner.scheduled_turn_closed(turn.seq_id);
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
        self.inner.release_scheduled_speculation(turn.seq_id);
        let parked = if outcome.is_ok() && turn.payload.reuse_cache {
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
        if outcome.is_err() || parked.is_err() {
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

    pub(crate) fn drive(
        &mut self,
        receiver: &mut tokio::sync::mpsc::UnboundedReceiver<ModelCommand<B::FamilyCommand>>,
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
                        self.pending.push_front(ModelCommand::from_chat(
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
                        || !(Self::chat_has_explicit_owner(&chat)
                            || self.ownerless_speculation_is_scheduled(&chat))
                        || self.chat_requires_barrier(&chat)
                    {
                        self.scheduler
                            .enqueue_exclusive(ModelCommand::from_chat(chat));
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

        self.begin_scheduled_step();
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
                blocked_alone,
                ..
            }) => {
                may_resume_preempted = self.settle_step(completed, preempted, blocked_alone);
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

        let inner = Qwen35Inner::new(tiny_config()).expect("construct tiny dense model");
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
            Qwen35FamilyCommand::ForceFlatMtpDesyncForTest { reply: force_reply }.into(),
            Qwen35Cmd::scheduler_stats(stats_reply),
            Qwen35FamilyCommand::MtpFlatStateForTest {
                reply: before_reply,
            }
            .into(),
            Qwen35Cmd::from_chat(ChatCmd::ResetCaches { reply: reset_reply }),
            Qwen35FamilyCommand::MtpFlatStateForTest { reply: after_reply }.into(),
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
        assert!(before_result.try_recv().unwrap().unwrap().1);
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
        assert!(!after_result.try_recv().unwrap().unwrap().1);
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
        let mut inner = Qwen35MoeInner::new(tiny_moe_config()).expect("construct tiny MoE");
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
            assert!(!plan.use_paged_attention);
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
            use_fp8_cache: Some(false),
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

    /// Swaps `inner`'s paged adapter for the growable fixture, makes the
    /// headroom probe decline its first call only, and reserves one block
    /// past the free pool with no live turns. Returns whether admission
    /// admitted, the pool's blocks after, and `inner`.
    pub(crate) fn reserve_past_a_declined_grow<B: HybridSchedulerBackend>(
        mut inner: B,
    ) -> (bool, u32, B) {
        let recency = inner.paged_adapter().map_or_else(
            || Arc::new(mlx_paged_attn::CacheRecency::new()),
            PagedKVCacheAdapter::recency,
        );
        let mut adapter = tiny_paged_adapter_with(recency, 4, 16);
        let probes = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let calls = Arc::clone(&probes);
        adapter.set_grow_headroom_probe_override(Some(Arc::new(move |requested| {
            if calls.fetch_add(1, Ordering::Relaxed) == 0 {
                Ok(4)
            } else {
                Ok(requested)
            }
        })));
        *inner.paged_adapter_mut().expect("paged adapter") = adapter;
        let mut state = HybridSchedulerState::new(inner).expect("construct scheduler");
        let snapshot = state
            .inner
            .scheduler_cache_snapshot()
            .expect("scheduler snapshot")
            .expect("paged adapter present");
        assert_eq!(snapshot.blocks.total_blocks, 4);
        assert!(!state.scheduler.has_live_turns());
        let charge = snapshot.blocks.free_blocks + 1;
        let outcome = state
            .try_reserve_reclaiming_idle(2, charge, 0, 0, snapshot)
            .expect("reserve");
        let total = state
            .inner
            .scheduler_cache_snapshot()
            .expect("post-reserve snapshot")
            .expect("paged adapter present")
            .blocks
            .total_blocks;
        (
            matches!(outcome, MemoryReserveOutcome::Admitted),
            total,
            state.inner,
        )
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
        if std::env::var_os("MLX_SCHED_RESERVE_FULL_ISL").is_none()
            && let Ok(nemotron) = NemotronHInner::new(tiny_nemotron_paged_config())
        {
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

        if std::env::var_os("MLX_SCHED_RESERVE_FULL_ISL").is_some() {
            return;
        }
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

    /// A refused grow before the idle owner is reclaimed frees nothing
    /// beside the pool: the idle owner's in-pool blocks admit the turn, and
    /// the live and parked DFlash2 states stay. Mutation: grow through
    /// `try_grow_pool_by` before `reclaim_one_idle_scheduled_with` in
    /// `try_reserve_with_policy` → red (the states are evicted, 0 ≠ 2).
    #[cfg(target_os = "macos")]
    #[test]
    fn refused_grow_reclaims_the_idle_owner_before_any_dflash2_state() {
        let Some(mut state) = qwen35_state_with_parked_dflash2_states(3) else {
            return;
        };
        let _no_compaction = no_compaction_headroom();
        assert_eq!(parked_dflash2_states(&state.inner), 2);
        let evictions = pool_grow_evictions();
        let (outcome, kept) = reserve_beside_an_idle_owner(&mut state, Arc::new(|_| Ok(0)));
        assert!(
            matches!(outcome, MemoryReserveOutcome::Admitted),
            "{outcome:?}"
        );
        assert!(!kept, "the idle owner's blocks admit the turn");
        assert_eq!(pool_grow_evictions(), evictions);
        assert_eq!(parked_dflash2_states(&state.inner), 2);
        assert!(
            state.inner.dflash2_context.is_some(),
            "the live state stays live"
        );
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
    fn mtp_step_state_with(
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
            use_fp8_cache: Some(false),
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

    /// A running turn over a `prompt`-token prompt at `num_tokens`, charged
    /// `charge`; its prompt blocks go back to the pool so every case reads
    /// the whole pool as room.
    #[cfg(target_os = "macos")]
    fn mtp_step_turn_with(
        state: &mut HybridSchedulerState<Qwen35Inner>,
        seq: u32,
        prompt: u32,
        speculative: bool,
        num_tokens: u32,
        charge: u64,
    ) -> TurnState<ScheduledTurn<<Qwen35Inner as PagedBackend>::PrefixState>> {
        let mut turn = speculative::tests::request(
            &mut state.inner,
            seq,
            (0..prompt).map(|token| token % 16).collect(),
            speculative,
            Arc::new(AtomicBool::new(false)),
        )
        .with_block_reservation(0, 0, 16)
        .with_recurrent_state_reservation(charge);
        turn.num_tokens = num_tokens;
        state
            .inner
            .release_scheduled_cache(seq)
            .expect("return the prompt blocks");
        let snapshot = state
            .inner
            .scheduler_cache_snapshot()
            .expect("scheduler snapshot")
            .expect("paged adapter present");
        assert_eq!(
            snapshot.blocks.free_blocks + snapshot.blocks.reclaimable_blocks,
            snapshot.blocks.total_blocks
        );
        turn
    }

    /// A decoding MTP turn over a 40-token prompt at `num_tokens`.
    #[cfg(target_os = "macos")]
    fn mtp_step_turn(
        state: &mut HybridSchedulerState<Qwen35Inner>,
        seq: u32,
        num_tokens: u32,
        admission: u64,
    ) -> TurnState<ScheduledTurn<<Qwen35Inner as PagedBackend>::PrefixState>> {
        let mut turn = mtp_step_turn_with(state, seq, 40, true, num_tokens, admission);
        turn.num_computed_tokens = num_tokens - 1;
        turn
    }

    /// Each step prices every running MTP turn at the state it holds, its
    /// charge included, against the 64-block pool (524,288 B): alone at
    /// 254 rows 463,104 B fits, at 504 rows 527,104 B does not, and two
    /// turns at 250 (925,952 B) do not, though their state past the
    /// charges is only 54,016 B.
    /// Mutations: `draft_growth_fits` returns `Ok(true)` → red (504 rows);
    /// price turns past their charge (`uncharged_draft_state_bytes`) → red
    /// (pair); drop the peers → red (pair).
    #[cfg(target_os = "macos")]
    #[test]
    fn running_mtp_step_prices_every_turns_held_state() {
        let Some((mut state, admission)) = mtp_step_state_with(64, 64) else {
            return;
        };
        let short = vec![mtp_step_turn(&mut state, 1, 250, admission)];
        assert_eq!(
            step_state_bytes(&state.inner, &short[0], 254, 8_192),
            463_104
        );
        assert!(
            HybridStepExecutor::new(&mut state.inner)
                .draft_growth_fits(&short, 0, 4)
                .expect("short step")
        );
        let lone = vec![mtp_step_turn(&mut state, 2, 500, admission)];
        assert_eq!(
            step_state_bytes(&state.inner, &lone[0], 504, 8_192),
            527_104
        );
        assert!(
            !HybridStepExecutor::new(&mut state.inner)
                .draft_growth_fits(&lone, 0, 4)
                .expect("lone step")
        );
        state.inner.release_scheduled_recurrent_for(1);
        state.inner.release_scheduled_recurrent_for(2);
        let pair = vec![
            mtp_step_turn(&mut state, 3, 250, admission),
            mtp_step_turn(&mut state, 4, 250, admission),
        ];
        assert_eq!(pair[1].decode_draft_allowance, 3);
        assert_eq!(
            step_state_bytes(&state.inner, &pair[1], 253, 8_192),
            462_848
        );
        assert_eq!(
            uncharged_draft_state_bytes(&state.inner, &pair[0], 254)
                + uncharged_draft_state_bytes(&state.inner, &pair[1], 253),
            54_016
        );
        assert!(
            !HybridStepExecutor::new(&mut state.inner)
                .draft_growth_fits(&pair, 0, 4)
                .expect("pair step")
        );
    }

    /// A 16-block pool (131,072 B free) that may grow to 64 blocks has
    /// 524,288 B left for state. A lone MTP turn at 254 rows holds
    /// 463,104 B: it fits in the ungrown room and the pool stays at 16
    /// blocks, since state lives beside the pool. At 504 rows it holds
    /// 527,104 B and blocks.
    /// Mutation: grow the pool by the shortfall over the free bytes in
    /// `draft_growth_fits` → red (57 blocks ≠ 16).
    #[cfg(target_os = "macos")]
    #[test]
    fn running_mtp_step_counts_the_ungrown_pool_without_growing_it() {
        let Some((mut state, admission)) = mtp_step_state_with(16, 64) else {
            return;
        };
        set_mtp_step_probe(&mut state, true);
        let total_blocks = |state: &HybridSchedulerState<Qwen35Inner>| {
            state
                .inner
                .scheduler_cache_snapshot()
                .expect("scheduler snapshot")
                .expect("paged adapter present")
                .blocks
                .total_blocks
        };
        let lone = vec![mtp_step_turn(&mut state, 1, 250, admission)];
        assert_eq!(
            step_state_bytes(&state.inner, &lone[0], 254, 8_192),
            463_104
        );
        assert!(
            HybridStepExecutor::new(&mut state.inner)
                .draft_growth_fits(&lone, 0, 4)
                .expect("step in the ungrown room")
        );
        assert_eq!(total_blocks(&state), 16);
        state.inner.release_scheduled_recurrent_for(1);
        let past = vec![mtp_step_turn(&mut state, 2, 500, admission)];
        assert_eq!(
            step_state_bytes(&state.inner, &past[0], 504, 8_192),
            527_104
        );
        assert!(
            !HybridStepExecutor::new(&mut state.inner)
                .draft_growth_fits(&past, 0, 4)
                .expect("step past the max pool")
        );
        assert_eq!(total_blocks(&state), 16);
    }

    /// A lone MTP turn admitted over a 128-token prompt (9 booked blocks)
    /// passes its step at the last row of its window (159 rows, 10 blocks:
    /// 438,784 B of state in 442,368 B free), priced at what it holds, not
    /// at its admission charge's prefill hidden rows; at 176 rows (11
    /// blocks) it is past the window and blocks.
    /// Mutation: price the own turn at `max(charge, held)` → red (503,552 B).
    #[cfg(target_os = "macos")]
    #[test]
    fn lone_mtp_turn_passes_every_step_in_its_window() {
        let Some((mut state, _)) = mtp_step_state_with(64, 64) else {
            return;
        };
        assert_eq!(state.scheduled_turn_window(16, true), 159);
        let charge =
            state.inner.recurrent_state_bytes() + state.inner.scheduled_draft_state_bytes(144);
        let mut turn = mtp_step_turn_with(&mut state, 1, 128, true, 155, charge)
            .with_block_reservation(9, 9, 16);
        turn.num_computed_tokens = 154;
        let adapter = state.inner.paged_adapter_mut().expect("paged adapter");
        adapter.reserve_rows_for(1, 159).expect("window KV");
        assert_eq!(adapter.block_table_for(1).expect("table").num_blocks(), 10);
        let lone = vec![turn];
        assert_eq!(
            step_state_bytes(&state.inner, &lone[0], 159, 8_192),
            438_784
        );
        assert!(
            HybridStepExecutor::new(&mut state.inner)
                .draft_growth_fits(&lone, 0, 4)
                .expect("last window step")
        );
        let mut past = lone;
        past[0].num_tokens = 172;
        past[0].num_computed_tokens = 171;
        state
            .inner
            .paged_adapter_mut()
            .expect("paged adapter")
            .reserve_rows_for(1, 176)
            .expect("past-window KV");
        assert!(
            !HybridStepExecutor::new(&mut state.inner)
                .draft_growth_fits(&past, 0, 4)
                .expect("past-window step")
        );
    }

    /// A peer's booked blocks the pool has not handed out yet count: an AR
    /// peer still prefilling with 8 unbooked blocks (65,536 B) and its
    /// recurrent charge (29,184 B) beside a lone MTP turn at 254 rows
    /// (463,104 B) is 557,824 B, over the 64-block pool's 524,288 B.
    /// Mutation: drop the bookings from `step_state_bytes` → red (29,184 B ≠ 94,720 B).
    #[cfg(target_os = "macos")]
    #[test]
    fn running_mtp_step_counts_unmaterialized_bookings() {
        let Some((mut state, admission)) = mtp_step_state_with(64, 64) else {
            return;
        };
        let recurrent = state.inner.recurrent_state_bytes();
        let mtp = mtp_step_turn(&mut state, 1, 250, admission);
        let ar = mtp_step_turn_with(&mut state, 2, 40, false, 40, recurrent)
            .with_block_reservation(8, 0, 16);
        let pair = vec![mtp, ar];
        assert_eq!(
            step_state_bytes(&state.inner, &pair[1], 40, 8_192),
            recurrent + 65_536
        );
        assert!(
            !HybridStepExecutor::new(&mut state.inner)
                .draft_growth_fits(&pair, 0, 4)
                .expect("step beside a booked peer")
        );
        assert!(
            HybridStepExecutor::new(&mut state.inner)
                .draft_growth_fits(&pair[..1], 0, 4)
                .expect("step alone")
        );
    }

    /// An idle owner's parked row counts: beside it, the lone MTP turn at
    /// the last row of its window (438,784 B in 442,368 B free) needs
    /// 467,968 B and blocks; with the row released it fits.
    /// Mutations: drop the idle rows from `draft_growth_fits` → red (blocks
    /// beside the row); do not subtract the running turns' own rows → red
    /// (fits alone).
    #[cfg(target_os = "macos")]
    #[test]
    fn mtp_step_counts_idle_parked_rows() {
        let Some((mut state, _)) = mtp_step_state_with(64, 64) else {
            return;
        };
        let recurrent = state.inner.recurrent_state_bytes();
        drop(mtp_step_turn_with(&mut state, 2, 40, false, 40, recurrent));
        let charge = recurrent + state.inner.scheduled_draft_state_bytes(144);
        let mut turn = mtp_step_turn_with(&mut state, 1, 128, true, 155, charge)
            .with_block_reservation(9, 9, 16);
        turn.num_computed_tokens = 154;
        let adapter = state.inner.paged_adapter_mut().expect("paged adapter");
        adapter.reserve_rows_for(1, 159).expect("window KV");
        assert_eq!(adapter.block_table_for(1).expect("table").num_blocks(), 10);
        let lone = vec![turn];
        assert_eq!(state.inner.scheduled_recurrent_bytes(), 58_368);
        assert!(
            !HybridStepExecutor::new(&mut state.inner)
                .draft_growth_fits(&lone, 0, 4)
                .expect("step beside an idle row")
        );
        state.inner.release_scheduled_recurrent_for(2);
        assert_eq!(state.inner.scheduled_recurrent_bytes(), 29_184);
        assert!(
            HybridStepExecutor::new(&mut state.inner)
                .draft_growth_fits(&lone, 0, 4)
                .expect("step with the idle row released")
        );
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
        let Ok(SchedulerAction::Stepped {
            completed,
            preempted,
            blocked_alone,
            ..
        }) = state.scheduler.drive_once(&mut BlockEveryRow)
        else {
            panic!("expected a step");
        };
        assert!(!state.settle_step(completed, preempted, blocked_alone));
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

    /// A lone MTP turn at the last row of its window blocks beside an idle
    /// owner that holds only its parked row (no blocks): the blocked step
    /// frees that row and the turn's next step fits.
    /// Mutation: pick only idle owners with blocks in
    /// `reclaim_for_blocked_step` (the old `materialized_blocks > 0`
    /// filter) → red (the turn fails).
    #[cfg(target_os = "macos")]
    #[test]
    fn lone_mtp_turn_frees_an_idle_row_only_owner() {
        let Some((mut state, _)) = mtp_step_state_with(64, 64) else {
            return;
        };
        let recurrent = state.inner.recurrent_state_bytes();
        drop(mtp_step_turn_with(&mut state, 2, 40, false, 40, recurrent));
        state.owner_sequences.insert("idle".into(), 2);
        assert_eq!(state.inner.scheduler_materialized_blocks(2), 0);
        let charge = recurrent + state.inner.scheduled_draft_state_bytes(144);
        let mut turn = mtp_step_turn_with(&mut state, 1, 128, true, 155, charge)
            .with_block_reservation(9, 9, 16);
        turn.num_computed_tokens = 154;
        let (reply, mut result) = tokio::sync::oneshot::channel();
        turn.payload.response = ScheduledReply::Sync(reply);
        state
            .inner
            .paged_adapter_mut()
            .expect("paged adapter")
            .reserve_rows_for(1, 159)
            .expect("window KV");
        assert_eq!(state.inner.scheduled_recurrent_bytes(), 2 * recurrent);
        assert!(
            !HybridStepExecutor::new(&mut state.inner)
                .draft_growth_fits(std::slice::from_ref(&turn), 0, 4)
                .expect("step beside the idle row")
        );
        settle_blocked_step(&mut state, turn);
        assert!(
            matches!(
                result.try_recv(),
                Err(tokio::sync::oneshot::error::TryRecvError::Empty)
            ),
            "the turn is still running"
        );
        assert_eq!(state.scheduler.running().len(), 1);
        assert_eq!(state.scheduler.stats().preemptions, 0);
        assert_eq!(state.inner.scheduled_recurrent_bytes(), recurrent);
        assert!(
            HybridStepExecutor::new(&mut state.inner)
                .draft_growth_fits(state.scheduler.running(), 0, 4)
                .expect("step after the release")
        );
    }

    /// 256 usable tokens in 16-token blocks of 8,192 B hold 131,072 B; with
    /// `1,000 + 256·T` held beside them, T = 160 fits (123,880 B) and
    /// T = 161 needs 11 blocks (132,328 B).
    /// Mutation: drop `held(T)` from the fit → red (256 ≠ 160).
    #[test]
    fn speculative_window_prices_the_held_draft_state() {
        let held = |tokens: u32| 1_000 + 256 * u64::from(tokens);
        assert_eq!(speculative_window_tokens(256, 16, 8_192, held), 160);
        assert_eq!(speculative_window_tokens(256, 16, 0, held), 256);
        assert_eq!(speculative_window_tokens(256, 16, 8_192, |_| u64::MAX), 1);
    }

    /// An MTP turn's window pays the draft state it holds at its last row,
    /// so a lone turn there passes every step check; an AR turn keeps the
    /// pool window.
    /// Mutation: `scheduled_turn_window` ignores `speculative` → red.
    #[test]
    fn mtp_turn_window_is_the_speculative_window() {
        let Some((mut state, _)) = growable_qwen35_state() else {
            return;
        };
        state.inner.config.n_mtp_layers = 1;
        let cfg = mlx_paged_attn::PagedAttentionConfig {
            block_size: 16,
            num_kv_heads: 1,
            head_size: 128,
            num_layers: 1,
            gpu_memory_mb: 256,
            use_fp8_cache: Some(false),
            max_seq_len: Some(8_192),
            max_batch_size: Some(32),
        };
        let pool = mlx_paged_attn::LayerKVPool::new(
            cfg,
            4,
            512,
            mlx_paged_attn::metal::MetalDtype::BFloat16,
        )
        .expect("MTP window LayerKVPool");
        let allocator = Arc::new(std::sync::Mutex::new(mlx_paged_attn::BlockAllocator::new(
            4, 512, 16,
        )));
        state.inner.paged_adapter = Some(
            PagedKVCacheAdapter::new(allocator, Arc::new(pool), 16).expect("MTP window adapter"),
        );
        let ar = state.scheduled_turn_window(16, false);
        let mtp = state.scheduled_turn_window(16, true);
        assert_eq!(
            mtp,
            speculative_window_tokens(ar, 16, 8_192, |tokens| {
                state.inner.scheduled_draft_state_held_bytes(tokens)
            })
        );
        assert!(1 < mtp && mtp < ar, "{mtp} vs {ar}");
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

    /// A victim preempted at history 63 resumes into 64 rows (4 blocks), 67
    /// with three draft rows (5 blocks); alone in a pool that cannot grow
    /// past 4 blocks, the 5-block resume rejects instead of re-admitting.
    /// Mutation: `resume_booking_blocks` keeps the booked blocks → red.
    /// Mutation: drop the draft rows → red (4 ≠ 5).
    #[test]
    fn lone_victim_resumes_only_into_its_blocked_step() {
        assert_eq!(resume_booking_blocks(2, 63, 0, 1_000, 16, false), 4);
        assert_eq!(resume_booking_blocks(2, 63, 3, 1_000, 16, false), 5);
        assert_eq!(resume_booking_blocks(9, 63, 3, 1_000, 16, false), 9);
        let Some((mut state, snapshot)) = growable_qwen35_state() else {
            return;
        };
        state
            .inner
            .paged_adapter_mut()
            .expect("paged adapter")
            .set_grow_headroom_probe_override(Some(Arc::new(|_| Ok(4))));
        let outcome = state
            .try_reserve_with_policy(
                2,
                resume_booking_blocks(2, 63, 3, 1_000, 16, false),
                0,
                0,
                snapshot,
                false,
                <Qwen35Inner as HybridSchedulerBackend>::release_scheduled_cache,
            )
            .expect("reject");
        assert!(
            matches!(outcome, MemoryReserveOutcome::Reject { total_blocks: 4 }),
            "{outcome:?}"
        );
    }

    /// A victim at history 63 with one output token left needs 64 rows: its
    /// three draft rows run past the whole turn, so it books 4 blocks, not 5,
    /// and resumes alone in a pool that cannot grow past 4 blocks.
    /// Mutation: drop the `.min(requested_tokens…)` cap in
    /// `resume_booking_blocks` → red (5 blocks, Reject).
    #[cfg(target_os = "macos")]
    #[test]
    fn lone_victim_with_one_token_left_resumes_in_its_last_block() {
        let Some((mut state, _)) = growable_qwen35_state() else {
            return;
        };
        let mut turn = speculative::tests::request(
            &mut state.inner,
            2,
            (0..40).map(|token| token % 16).collect(),
            true,
            Arc::new(AtomicBool::new(false)),
        )
        .with_block_reservation(2, 0, 16);
        turn.payload.requested_tokens = 64;
        assert_eq!(turn.decode_draft_allowance, 3);
        state.book_resume(&mut turn, 63, true, 0, false);
        state.inner.release_scheduled_recurrent_for(2);
        state.inner.release_scheduled_speculation(2);
        state
            .inner
            .release_scheduled_cache(2)
            .expect("release the victim");
        let snapshot = state
            .inner
            .scheduler_cache_snapshot()
            .expect("scheduler snapshot")
            .expect("paged adapter present");
        state
            .inner
            .paged_adapter_mut()
            .expect("paged adapter")
            .set_grow_headroom_probe_override(Some(Arc::new(|_| Ok(4))));
        let outcome = state
            .try_reserve_with_policy(
                2,
                turn.block_reservation_total,
                0,
                0,
                snapshot,
                false,
                <Qwen35Inner as HybridSchedulerBackend>::release_scheduled_cache,
            )
            .expect("resume decision");
        assert!(
            matches!(outcome, MemoryReserveOutcome::Admitted),
            "{outcome:?}"
        );
        assert_eq!(resume_booking_blocks(2, 63, 3, 64, 16, false), 4);
        assert_eq!(resume_booking_blocks(2, 63, 3, 66, 16, false), 5);
        assert_eq!(resume_booking_blocks(2, 63, 3, 67, 16, false), 5);
        assert_eq!(resume_booking_blocks(2, 63, 3, 10, 16, false), 4);
    }

    /// A 64-token prompt with one output token books 4 blocks; preempted
    /// mid-prefill, it resumes into the same 4, not 5 for a pending row a
    /// prefill replay does not have, and a lone resume fits a 4-block pool.
    /// Mutation: drop the `prefill_replay` branch in `resume_booking_blocks`
    /// → red (5 blocks, Reject).
    #[cfg(target_os = "macos")]
    #[test]
    fn prefill_victim_resumes_into_its_admission_blocks() {
        assert_eq!(scheduled_booking(true, 64, 1, 16), (4, 64));
        assert_eq!(resume_booking_blocks(4, 64, 3, 64, 16, true), 4);
        assert_eq!(resume_booking_blocks(4, 64, 0, 64, 16, true), 4);
        assert_eq!(resume_booking_blocks(4, 64, 0, 65, 16, false), 5);
        let Some((mut state, _)) = growable_qwen35_state() else {
            return;
        };
        let prompt: Vec<u32> = (0..64).map(|token| token % 16).collect();
        let mut turn = speculative::tests::request(
            &mut state.inner,
            2,
            prompt.clone(),
            true,
            Arc::new(AtomicBool::new(false)),
        )
        .with_block_reservation(4, 2, 16);
        turn.payload.requested_tokens = 64;
        turn.num_computed_tokens = 32;
        let replay = install_preemption_replay(&mut turn, &prompt, 32).expect("prefill replay");
        assert!(!replay.suppress_sample);
        state.book_resume(
            &mut turn,
            replay.tokens.len() as u32,
            true,
            0,
            !replay.suppress_sample,
        );
        assert_eq!(turn.block_reservation_total, 4);
        state.inner.release_scheduled_recurrent_for(2);
        state.inner.release_scheduled_speculation(2);
        state
            .inner
            .release_scheduled_cache(2)
            .expect("release the victim");
        let snapshot = state
            .inner
            .scheduler_cache_snapshot()
            .expect("scheduler snapshot")
            .expect("paged adapter present");
        state
            .inner
            .paged_adapter_mut()
            .expect("paged adapter")
            .set_grow_headroom_probe_override(Some(Arc::new(|_| Ok(4))));
        let outcome = state
            .try_reserve_with_policy(
                2,
                turn.block_reservation_total,
                0,
                0,
                snapshot,
                false,
                <Qwen35Inner as HybridSchedulerBackend>::release_scheduled_cache,
            )
            .expect("resume decision");
        assert!(
            matches!(outcome, MemoryReserveOutcome::Admitted),
            "{outcome:?}"
        );
    }

    /// A victim that did not block resumes to the blocks it held, not to the
    /// blocked row's step; one that blocked books its own step.
    /// Mutation: ignore `blocked` in `book_resume` → red (5 ≠ 4).
    #[cfg(target_os = "macos")]
    #[test]
    fn unblocked_victim_resumes_to_held_blocks() {
        let Some((mut state, _)) = growable_qwen35_state() else {
            return;
        };
        let mut turn = speculative::tests::request(
            &mut state.inner,
            2,
            (0..40).map(|token| token % 16).collect(),
            true,
            Arc::new(AtomicBool::new(false)),
        )
        .with_block_reservation(3, 0, 16);
        turn.payload.requested_tokens = 1_000;
        state.book_resume(&mut turn, 63, false, 4, false);
        assert_eq!(turn.block_reservation_total, 4);
        state.book_resume(&mut turn, 63, false, 2, false);
        assert_eq!(turn.block_reservation_total, 4);
        state.book_resume(&mut turn, 63, true, 4, false);
        assert_eq!(turn.block_reservation_total, 5);
    }

    /// A blocked step frees an idle owner's blocks and keeps its victim
    /// running; with nothing idle left it preempts.
    /// Mutation: never offer idle owners (`oldest_kv` returns `None` in
    /// `SchedReclaim`) → red.
    #[test]
    fn blocked_step_releases_an_idle_owner_before_preempting() {
        let Some((mut state, _)) = growable_qwen35_state() else {
            return;
        };
        state.owner_sequences.insert("idle".into(), 1);
        let adapter = state.inner.paged_adapter_mut().expect("paged adapter");
        adapter.set_grow_headroom_probe_override(Some(Arc::new(|_| Ok(4))));
        adapter.begin_request(1).expect("begin idle request");
        adapter
            .allocate_suffix_blocks_for(1, 32)
            .expect("idle owner holds two blocks");
        assert_eq!(state.idle_memory_victim(2), Some(1));
        let refuse = |state: &HybridSchedulerState<Qwen35Inner>, blocks| {
            state
                .inner
                .paged_adapter()
                .expect("paged adapter")
                .refuse_for_test(2, blocks);
        };
        refuse(&state, 3);
        assert!(state.reclaim_for_blocked_step(2, false));
        assert!(
            state
                .inner
                .paged_adapter()
                .and_then(|adapter| adapter.block_table_for(1))
                .is_none(),
            "the idle owner's table is released"
        );
        refuse(&state, 5);
        assert!(!state.reclaim_for_blocked_step(2, false));
    }

    /// Qwen3.5 scheduler over the growable pool fixture (4 live blocks, 16
    /// max, a passing grow probe) with one parked DFlash2 state; `None`
    /// without Metal.
    fn qwen35_state_with_a_parked_dflash2_state() -> Option<HybridSchedulerState<Qwen35Inner>> {
        qwen35_state_with_parked_dflash2_states(2)
    }

    /// The growable fixture after `turns` DFlash2 turns of distinct prompts:
    /// `turns - 1` parked states and the last one live.
    fn qwen35_state_with_parked_dflash2_states(
        turns: u32,
    ) -> Option<HybridSchedulerState<Qwen35Inner>> {
        qwen35_dflash2_state(turns, 4, 16)
    }

    /// [`qwen35_state_with_parked_dflash2_states`] over an `initial`-block
    /// pool growing to `max`.
    fn qwen35_dflash2_state(
        turns: u32,
        initial: u32,
        max: u32,
    ) -> Option<HybridSchedulerState<Qwen35Inner>> {
        use crate::models::qwen3_5::dflash2_decode::tests::{
            metal_is_default_device, run_tiny_dflash_turn, tiny_int8_dflash_inner,
        };
        if !metal_is_default_device() {
            return None;
        }
        let tokenizer = crate::models::gemma4::dspark_decode::tests::tiny_qwen_tokenizer();
        let mut inner = tiny_int8_dflash_inner(0xDFA5_1702);
        for seed in 4..4 + turns {
            let body: Vec<u32> = (0..40).map(|i| (seed + i * 7) % 9 + 1).collect();
            run_tiny_dflash_turn(&mut inner, &tokenizer, &body, true, 12)
                .expect("tiny DFlash2 turn");
        }
        assert_eq!(parked_dflash2_states(&inner), turns as usize - 1);
        inner.paged_adapter = Some(tiny_paged_adapter_with(
            Arc::clone(&inner.cache_recency),
            initial,
            max,
        ));
        let mut state = HybridSchedulerState::new(inner).expect("construct Qwen3.5 scheduler");
        state
            .inner
            .paged_adapter_mut()
            .expect("paged adapter")
            .set_grow_headroom_probe_override(Some(Arc::new(Ok)));
        Some(state)
    }

    /// Zero live headroom on this thread until dropped, so a store step
    /// drops a state instead of compacting one.
    struct NoCompactionHeadroom;
    impl Drop for NoCompactionHeadroom {
        fn drop(&mut self) {
            crate::models::qwen3_5::dflash2_turn::KV_BUDGET_OVERRIDE
                .with(|budget| budget.set(None));
        }
    }
    fn no_compaction_headroom() -> NoCompactionHeadroom {
        crate::models::qwen3_5::dflash2_turn::KV_BUDGET_OVERRIDE.with(|budget| budget.set(Some(0)));
        NoCompactionHeadroom
    }

    fn pool_grow_evictions() -> usize {
        crate::models::qwen3_5::dflash2_turn::grow_reclaim_log().len()
    }

    fn parked_dflash2_states(inner: &Qwen35Inner) -> usize {
        inner.dflash2_states.store.bytes().entries
    }

    fn parked_bytes(state: &HybridSchedulerState<Qwen35Inner>) -> u64 {
        let bytes = state.inner.dflash2_states.store.bytes();
        bytes.kv_bytes + bytes.state_bytes
    }

    fn bytes_per_block(state: &HybridSchedulerState<Qwen35Inner>) -> u64 {
        state
            .inner
            .paged_adapter()
            .expect("paged adapter")
            .bytes_per_block()
            .expect("bytes per block")
    }

    fn clear_reclaim_log() {
        crate::models::qwen3_5::dflash2_turn::clear_grow_reclaim_log();
    }

    fn reclaim_log() -> Vec<crate::models::qwen3_5::dflash2_turn::GrowReclaim> {
        crate::models::qwen3_5::dflash2_turn::grow_reclaim_log()
    }

    type Probe = Arc<dyn Fn(u32) -> std::result::Result<u32, String> + Send + Sync>;

    /// Declines with no room (the 4-block pool's own size) until the reclaim
    /// log on this thread has `taken` entries.
    fn decline_until_taken(taken: usize) -> Probe {
        Arc::new(move |blocks| {
            Ok(if reclaim_log().len() >= taken {
                blocks
            } else {
                4
            })
        })
    }

    fn pool_total_blocks(state: &HybridSchedulerState<Qwen35Inner>) -> u32 {
        state
            .inner
            .scheduler_cache_snapshot()
            .expect("snapshot")
            .expect("adapter")
            .blocks
            .total_blocks
    }

    /// A blocked step at the max pool evicts no parked DFlash2 state: no
    /// grow can use the freed memory, and the state sits on no block.
    /// Mutation: park the live state at the cap too (drop `!at_cap` in
    /// `SchedReclaim::before_count`) → red (the live state parks).
    #[test]
    fn blocked_step_at_the_max_pool_keeps_parked_dflash2_states() {
        let Some(mut state) = qwen35_state_with_a_parked_dflash2_state() else {
            return;
        };
        let adapter = state.inner.paged_adapter_mut().expect("paged adapter");
        while adapter.try_grow_pool(16) {}
        adapter.refuse_for_test(2, 17);
        assert_eq!(pool_total_blocks(&state), 16);
        assert!(!state.reclaim_for_blocked_step(2, false));
        assert!(
            state.inner.dflash2_context.is_some(),
            "the live state stays"
        );
        assert_eq!(parked_dflash2_states(&state.inner), 1);
    }

    /// A blocked step below the max pool parks the idle live DFlash2 state,
    /// evicts one parked state, and grows as soon as the grow passes: the
    /// other states stay.
    /// Mutation: drop every parked state at once (`store.clear()` in place
    /// of `apply_lossy` in `take_state_unit`) → red (0 ≠ 2).
    /// Mutation: skip the park in `park_idle_live_state` → red (the live
    /// context stays).
    #[test]
    fn blocked_scheduled_step_evicts_one_then_grows() {
        let Some(mut state) = qwen35_state_with_parked_dflash2_states(3) else {
            return;
        };
        let _no_compaction = no_compaction_headroom();
        assert_eq!(parked_dflash2_states(&state.inner), 2);
        assert!(parked_bytes(&state) >= bytes_per_block(&state));
        let before = pool_total_blocks(&state);
        let adapter = state.inner.paged_adapter_mut().expect("paged adapter");
        adapter.set_grow_headroom_probe_override(Some(decline_until_taken(1)));
        adapter.refuse_for_test(2, 5);
        clear_reclaim_log();
        let evictions = pool_grow_evictions();
        assert!(state.reclaim_for_blocked_step(2, false));
        assert_eq!(pool_grow_evictions() - evictions, 1);
        assert!(pool_total_blocks(&state) > before);
        assert!(
            state.inner.dflash2_context.is_none(),
            "the live state parked"
        );
        assert_eq!(parked_dflash2_states(&state.inner), 2);
    }

    /// A declined admission grow takes one unit at a time and stops at the
    /// first grant: two declines cost the live state's park plus two parked
    /// states, and one stays.
    /// Mutation: re-check the need only once nothing is left in
    /// `free_for_kv` → red (0 ≠ 1).
    /// Mutation: drop every parked state at once (`store.clear()` in place
    /// of `apply_lossy` in `take_state_unit`) → red (0 ≠ 1).
    #[test]
    fn admission_grow_evicts_one_per_retry() {
        use std::sync::atomic::AtomicUsize;
        let Some(mut state) = qwen35_state_with_parked_dflash2_states(3) else {
            return;
        };
        let _no_compaction = no_compaction_headroom();
        let calls = Arc::new(AtomicUsize::new(0));
        let probe_calls = Arc::clone(&calls);
        state
            .inner
            .paged_adapter_mut()
            .expect("paged adapter")
            .set_grow_headroom_probe_override(Some(Arc::new(move |blocks| {
                Ok(if probe_calls.fetch_add(1, Ordering::SeqCst) < 2 {
                    0
                } else {
                    blocks
                })
            })));
        assert!(parked_bytes(&state) >= 4 * bytes_per_block(&state));
        clear_reclaim_log();
        let evictions = pool_grow_evictions();
        let grown = state
            .try_grow_pool_by(4, &mut MetBefore::default())
            .expect("grow")
            .expect("the third grow passes");
        assert!(grown.blocks.total_blocks > 4);
        assert_eq!(calls.load(Ordering::SeqCst), 3);
        assert_eq!(pool_grow_evictions() - evictions, 2);
        assert_eq!(parked_dflash2_states(&state.inner), 1);
    }

    // ---- take order (`crate::engine::reclaim`) on the scheduled lane ----

    use crate::models::qwen3_5::dflash2_turn::GrowReclaim;

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

    fn hold(state: &mut HybridSchedulerState<Qwen35Inner>, seq_id: SeqId, blocks: u32) {
        let adapter = state.inner.paged_adapter_mut().expect("paged adapter");
        adapter.begin_request(seq_id).expect("begin");
        adapter
            .allocate_suffix_blocks_for(seq_id, blocks * 16)
            .expect("hold blocks");
    }

    /// `seq_id` already holds `held` blocks and is refused `more`.
    fn refused(state: &mut HybridSchedulerState<Qwen35Inner>, seq_id: SeqId, held: u32, more: u32) {
        let adapter = state.inner.paged_adapter_mut().expect("paged adapter");
        if held == 0 {
            adapter.begin_request(seq_id).expect("begin");
        }
        assert!(
            adapter
                .reserve_rows_for(seq_id, (held + more) * 16)
                .is_err(),
            "the reservation is refused"
        );
    }

    fn table_of(state: &HybridSchedulerState<Qwen35Inner>, seq_id: SeqId) -> bool {
        state
            .inner
            .paged_adapter()
            .is_some_and(|adapter| adapter.block_table_for(seq_id).is_some())
    }

    fn rungs(state: &HybridSchedulerState<Qwen35Inner>) -> usize {
        state.inner.ladder_entries_for_test().len()
    }

    /// A real KV refusal goes to the older of an idle owner and a parked
    /// state, by their own stamps.
    /// Mutations: offer owners before states always → red (case b); states
    /// before owners always → red (case a).
    #[test]
    fn kv_denial_takes_the_older_of_an_idle_owner_and_a_state() {
        for owner_older in [true, false] {
            let Some(mut state) = qwen35_dflash2_state(2, 4, 16) else {
                return;
            };
            let _no_compaction = no_compaction_headroom();
            state
                .inner
                .paged_adapter_mut()
                .expect("paged adapter")
                .set_grow_headroom_probe_override(Some(decline_until_taken(1)));
            state.owner_sequences.insert("idle".into(), 1);
            hold(&mut state, 1, 2);
            state
                .owner_last_used
                .insert(1, if owner_older { 0 } else { u64::MAX });
            refused(&mut state, 9, 0, 3);
            clear_reclaim_log();
            assert!(
                state.reclaim_for_blocked_step(9, false),
                "owner older {owner_older}"
            );
            if owner_older {
                assert!(!table_of(&state, 1));
                assert!(reclaim_log().is_empty());
            } else {
                assert!(table_of(&state, 1));
                assert_eq!(reclaim_log(), vec![GrowReclaim::Store]);
            }
        }
    }

    /// An idle owner whose table frees no block (the blocked turn holds its
    /// chain too) is passed over though it is older.
    /// Mutation: drop the "frees something" filter in `oldest_idle_owner` →
    /// red (owner 2's table is released).
    #[test]
    fn idle_owners_that_free_nothing_are_skipped() {
        let Some(mut state) = qwen35_pool_state(4, 4, passing()) else {
            return;
        };
        let shared = tokens_of(1, 1);
        {
            let adapter = state.inner.paged_adapter_mut().expect("paged adapter");
            adapter.begin_request(3).expect("begin");
            adapter.allocate_suffix_blocks_for(3, 16).expect("allocate");
            adapter.record_tokens(&shared).expect("record");
            adapter
                .finalize_turn_keep_live(&[], 0)
                .expect("keep the shared block held");
            adapter.begin_request(2).expect("owner 2");
            let mut prompt = shared.clone();
            prompt.push(1);
            let hit = adapter
                .find_cached_prefix(&prompt, &[], 0, false)
                .expect("owner 2 hits the shared block");
            assert_eq!(hit.cached_token_count, 16);
        }
        hold(&mut state, 1, 1);
        state.owner_sequences.insert("one".into(), 1);
        state.owner_sequences.insert("two".into(), 2);
        state.owner_last_used.insert(1, 20);
        state.owner_last_used.insert(2, 10);
        let adapter = state.inner.paged_adapter_mut().expect("paged adapter");
        assert!(
            adapter.reserve_rows_for(3, 48).is_err(),
            "the blocked turn is refused 3 blocks"
        );
        assert!(state.reclaim_for_blocked_step(3, false));
        assert!(!table_of(&state, 1));
        assert!(table_of(&state, 2));
    }

    /// Two idle owners share a cached block: neither frees it alone, both
    /// together do, so the count covers a need of the free blocks plus both
    /// tables and the blocked turn runs.
    /// Mutation: count each owner's blocks of ref at most 2 apart (the old
    /// per-owner `releasable_blocks_for`) in `SchedReclaim::reach` → red
    /// (`Short`: the step fails `capacity_exhausted`).
    #[test]
    fn idle_owners_sharing_a_block_free_it_together() {
        let Some(mut state) = qwen35_pool_state(4, 4, passing()) else {
            return;
        };
        let shared = tokens_of(2, 1);
        {
            let adapter = state.inner.paged_adapter_mut().expect("paged adapter");
            adapter.begin_request(1).expect("owner 1");
            adapter.allocate_suffix_blocks_for(1, 32).expect("allocate");
            let mut owned = shared.clone();
            owned.extend(tokens_of(3, 1));
            adapter.record_tokens(&owned).expect("record");
            adapter
                .finalize_turn_keep_live(&[], 0)
                .expect("keep owner 1's table");
            adapter.begin_request(2).expect("owner 2");
            let mut prompt = shared.clone();
            prompt.push(1);
            let hit = adapter
                .find_cached_prefix(&prompt, &[], 0, false)
                .expect("owner 2 hits the shared block");
            assert_eq!(hit.cached_token_count, 16);
        }
        state.owner_sequences.insert("one".into(), 1);
        state.owner_sequences.insert("two".into(), 2);
        state.owner_last_used.insert(1, 10);
        state.owner_last_used.insert(2, 20);
        refused(&mut state, 9, 0, 4);
        assert_eq!(
            state
                .inner
                .paged_adapter()
                .expect("paged adapter")
                .takeable_blocks(),
            2
        );
        assert!(state.reclaim_for_blocked_step(9, false));
        assert!(!table_of(&state, 1));
        assert!(!table_of(&state, 2));
        assert_eq!(
            state
                .inner
                .paged_adapter()
                .expect("paged adapter")
                .takeable_blocks(),
            4
        );
    }

    /// An idle table the scheduler never finished goes by its turn-end
    /// stamp, newer than the parked state, so the state goes first.
    /// Mutation: stamp an owner with no `owner_last_used` entry 0 in
    /// `oldest_idle_owner` → red (the table is released).
    #[test]
    fn an_unfinished_idle_table_goes_by_its_turn_end_stamp() {
        let Some(mut state) = qwen35_dflash2_state(2, 4, 16) else {
            return;
        };
        let _no_compaction = no_compaction_headroom();
        state
            .inner
            .paged_adapter_mut()
            .expect("paged adapter")
            .set_grow_headroom_probe_override(Some(decline_until_taken(1)));
        state.owner_sequences.insert("idle".into(), 1);
        hold(&mut state, 1, 2);
        state
            .inner
            .paged_adapter_mut()
            .expect("paged adapter")
            .finalize_turn_keep_live_no_prefix()
            .expect("keep the table");
        assert!(
            state
                .inner
                .paged_adapter()
                .expect("paged adapter")
                .turn_end_stamp(1)
                .is_some()
        );
        assert!(!state.owner_last_used.contains_key(&1));
        refused(&mut state, 9, 0, 3);
        clear_reclaim_log();
        assert!(state.reclaim_for_blocked_step(9, false));
        assert!(table_of(&state, 1));
        assert_eq!(reclaim_log(), vec![GrowReclaim::Store]);
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

    /// At the cap a ladder rung and an idle owner go by their own stamps:
    /// the rung's block was cached after the owner's turn ended, but the
    /// rung itself is older.
    /// Mutation: stamp ladder units with their final block's stamp → red
    /// (the owner is released).
    #[test]
    fn cap_reclaim_orders_ladder_entries_and_owners_by_their_own_stamps() {
        let Some(mut state) = qwen35_pool_state(4, 4, passing()) else {
            return;
        };
        let hashes = rung(&mut state, 2, 1, 0);
        let block_stamp = state
            .inner
            .paged_adapter()
            .expect("paged adapter")
            .allocator()
            .lock()
            .expect("allocator")
            .block_stamp(hashes[0])
            .expect("block stamp");
        assert!(block_stamp >= 2);
        state.owner_sequences.insert("idle".into(), 1);
        hold(&mut state, 1, 1);
        state.owner_last_used.insert(1, block_stamp - 1);
        hold(&mut state, 9, 2);
        refused(&mut state, 9, 2, 1);
        clear_reclaim_log();
        assert!(state.reclaim_for_blocked_step(9, false));
        assert_eq!(reclaim_log(), vec![GrowReclaim::Ladder]);
        assert!(table_of(&state, 1));
    }

    /// A need no take could cover takes nothing.
    /// Mutation: drop the count in `free_for_kv` → red (the rung and the
    /// owner are both taken).
    #[test]
    fn cap_reclaim_evicts_nothing_it_cannot_cover() {
        let Some(mut state) = qwen35_pool_state(4, 4, passing()) else {
            return;
        };
        rung(&mut state, 3, 1, 0);
        state.owner_sequences.insert("idle".into(), 1);
        hold(&mut state, 1, 1);
        hold(&mut state, 9, 2);
        state
            .inner
            .paged_adapter()
            .expect("paged adapter")
            .refuse_for_test(9, 5);
        assert!(!state.reclaim_for_blocked_step(9, false));
        assert_eq!(rungs(&state), 1);
        assert!(table_of(&state, 1));
    }

    /// An admission grow at the cap takes the ladder rung, which frees
    /// blocks, and passes over the older parked state, which frees none.
    /// Mutation: drop the at-cap store skip in `oldest_state_unit` → red
    /// (`[Store, Ladder]`).
    #[test]
    fn admission_grow_at_the_cap_skips_older_store_states() {
        let Some(mut state) = qwen35_dflash2_state(2, 4, 4) else {
            return;
        };
        let stamp = state.inner.cache_recency.next();
        rung(&mut state, 4, 2, stamp);
        clear_reclaim_log();
        let grown = state
            .try_grow_pool_by(1, &mut MetBefore::default())
            .expect("grow");
        assert!(grown.is_some());
        assert_eq!(reclaim_log(), vec![GrowReclaim::Ladder]);
        assert_eq!(parked_dflash2_states(&state.inner), 1);
    }

    /// A reservation refused outside the step leaves no refusal for the
    /// step's reclaim.
    /// Mutation: drop `clear_refusals` from `begin_scheduled_step` → red
    /// (the rung is dropped).
    #[test]
    fn a_failed_reserve_outside_the_step_leaves_no_refusal() {
        let Some(mut state) = qwen35_dflash2_state(2, 4, 4) else {
            return;
        };
        rung(&mut state, 5, 2, 0);
        hold(&mut state, 9, 2);
        refused(&mut state, 9, 2, 1);
        state.begin_scheduled_step();
        assert!(!state.reclaim_for_blocked_step(9, false));
        assert_eq!(rungs(&state), 1);
    }

    /// Below the cap with every grow declined, dropping the oldest rung
    /// makes its chain takeable: the need is met, the newer state stays.
    /// Mutation: read takeable blocks as free plus reusable (marked chains
    /// in) in `kv_need_met` → red (the store state goes too).
    #[test]
    fn kv_denial_counts_a_freed_ladder_chain_as_progress() {
        let Some(mut state) = qwen35_dflash2_state(2, 4, 16) else {
            return;
        };
        let _no_compaction = no_compaction_headroom();
        state
            .inner
            .paged_adapter_mut()
            .expect("paged adapter")
            .set_grow_headroom_probe_override(Some(Arc::new(|_| Ok(4))));
        rung(&mut state, 6, 2, 0);
        hold(&mut state, 9, 2);
        refused(&mut state, 9, 2, 1);
        clear_reclaim_log();
        assert!(state.reclaim_for_blocked_step(9, false));
        assert_eq!(reclaim_log(), vec![GrowReclaim::Ladder]);
        assert_eq!(parked_dflash2_states(&state.inner), 2);
    }

    /// A decode wave at the cap records each row's own need, so the gate
    /// sees the wave's sum. Row 2, the one other table, shares its block
    /// with the blocked row 1, so no idle owner frees a block.
    /// Mutation: record only the first row's need in `check_wave_fits` →
    /// red (the record is 1 and the rung is dropped).
    #[test]
    fn wave_need_feeds_the_cap_gate() {
        let Some(mut state) = qwen35_pool_state(3, 3, passing()) else {
            return;
        };
        let shared = tokens_of(7, 1);
        {
            let adapter = state.inner.paged_adapter_mut().expect("paged adapter");
            adapter.begin_request(1).expect("row 1");
            adapter.allocate_suffix_blocks_for(1, 32).expect("allocate");
            let mut owned = shared.clone();
            owned.extend(tokens_of(8, 1));
            adapter.record_tokens(&owned).expect("record");
            adapter
                .finalize_turn_keep_live(&[], 0)
                .expect("keep row 1's table");
            adapter.begin_request(2).expect("row 2");
            let mut prompt = shared.clone();
            prompt.push(1);
            adapter
                .find_cached_prefix(&prompt, &[], 0, false)
                .expect("row 2 hits the shared block");
        }
        rung(&mut state, 9, 1, 0);
        let wave = |state: &mut HybridSchedulerState<Qwen35Inner>| {
            crate::transformer::paged_policy::record_decode_wave(
                state.inner.paged_adapter_mut().expect("paged adapter"),
                &[(1, 5), (2, 5)],
                "test",
            )
        };
        assert!(wave(&mut state).is_err());
        assert_eq!(
            state
                .inner
                .paged_adapter()
                .expect("paged adapter")
                .take_refusal(),
            Some(2)
        );
        assert!(wave(&mut state).is_err());
        assert!(!state.reclaim_for_blocked_step(1, false));
        assert_eq!(rungs(&state), 1);
    }

    /// A lone turn whose need no take covers keeps every cache: the count
    /// stops it after one cache clear.
    /// Mutations: drop the count in `free_for_kv` → red (the store
    /// empties); clear the cache only after the count → red (+0).
    #[test]
    fn doomed_lone_turn_keeps_the_cache() {
        let Some(mut state) = qwen35_dflash2_state(2, 4, 16) else {
            return;
        };
        let _no_compaction = no_compaction_headroom();
        state
            .inner
            .paged_adapter_mut()
            .expect("paged adapter")
            .set_grow_headroom_probe_override(Some(Arc::new(|_| Ok(4))));
        let dead = tokens_of(10, 1);
        cache_dead_chain(&state, &dead);
        let adapter = state.inner.paged_adapter().expect("paged adapter");
        let need = adapter.takeable_blocks() + adapter.ungrown_blocks() + 1;
        adapter.refuse_for_test(9, need);
        let clears = crate::array::memory::TEST_CLEAR_CACHE_CALLS.with(std::cell::Cell::get);
        state.handle_blocked_alone(9);
        assert_eq!(
            crate::array::memory::TEST_CLEAR_CACHE_CALLS.with(std::cell::Cell::get) - clears,
            1
        );
        assert_eq!(parked_dflash2_states(&state.inner), 2);
        let adapter = state.inner.paged_adapter().expect("paged adapter");
        let mut allocator = adapter.allocator().lock().expect("allocator");
        let (hit, cached) = allocator.find_longest_cache_hit(&dead, 16, &[], 0);
        assert_eq!(cached, 16);
        for block in hit {
            allocator.free(block);
        }
    }

    /// A need one state covers takes one state.
    /// Mutation: re-check the need only once nothing is left in
    /// `free_for_kv` → red (`[Store, Store]`).
    #[test]
    fn covered_need_stops_when_covered() {
        let Some(mut state) = qwen35_dflash2_state(2, 4, 16) else {
            return;
        };
        let _no_compaction = no_compaction_headroom();
        assert!(parked_bytes(&state) >= bytes_per_block(&state));
        state
            .inner
            .paged_adapter_mut()
            .expect("paged adapter")
            .set_grow_headroom_probe_override(Some(decline_until_taken(1)));
        hold(&mut state, 9, 4);
        refused(&mut state, 9, 4, 1);
        clear_reclaim_log();
        assert!(state.reclaim_for_blocked_step(9, false));
        assert_eq!(reclaim_log(), vec![GrowReclaim::Store]);
        assert_eq!(parked_dflash2_states(&state.inner), 1);
    }

    /// An admission grow no take could cover takes nothing.
    /// Mutation: drop the count in `free_for_kv` → red (0 entries).
    #[test]
    fn admission_grow_takes_nothing_it_cannot_cover() {
        let Some(mut state) = qwen35_dflash2_state(2, 4, 16) else {
            return;
        };
        let _no_compaction = no_compaction_headroom();
        state
            .inner
            .paged_adapter_mut()
            .expect("paged adapter")
            .set_grow_headroom_probe_override(Some(Arc::new(|_| Ok(4))));
        let needed = state
            .inner
            .paged_adapter()
            .expect("paged adapter")
            .ungrown_blocks()
            + 1;
        let grown = state
            .try_grow_pool_by(needed, &mut MetBefore::default())
            .expect("grow");
        assert!(grown.is_none());
        assert_eq!(parked_dflash2_states(&state.inner), 2);
    }

    /// A need a finished turn's released table already covers is met in
    /// place: nothing is taken.
    /// Mutation: drop the met check in `free_for_kv` → red (one `Store`).
    #[test]
    fn met_need_takes_nothing() {
        let Some(mut state) = qwen35_dflash2_state(2, 4, 16) else {
            return;
        };
        let _no_compaction = no_compaction_headroom();
        state
            .inner
            .paged_adapter_mut()
            .expect("paged adapter")
            .set_grow_headroom_probe_override(Some(Arc::new(|_| Ok(4))));
        hold(&mut state, 1, 2);
        hold(&mut state, 2, 2);
        refused(&mut state, 2, 2, 2);
        state
            .inner
            .paged_adapter_mut()
            .expect("paged adapter")
            .release_request_for(1)
            .expect("the finished turn's table");
        clear_reclaim_log();
        assert!(state.reclaim_for_blocked_step(2, false));
        assert!(reclaim_log().is_empty());
        assert_eq!(parked_dflash2_states(&state.inner), 1);
    }

    /// A lone admission at the cap whose only takeable blocks are a ladder
    /// chain drops the rung and admits.
    /// Mutations: fill the snapshot's `reclaimable_blocks` from
    /// `num_reusable_blocks` → red (admitted with the rung kept); drop
    /// `cap_shortfall_blocks` from `try_grow_reservation_pool` → red
    /// (`Reject`).
    #[test]
    fn admission_at_the_cap_drops_a_ladder_entry_to_admit() {
        let Some(mut state) = qwen35_pool_state(4, 4, passing()) else {
            return;
        };
        rung(&mut state, 11, 4, 0);
        let snapshot = state
            .inner
            .scheduler_cache_snapshot()
            .expect("snapshot")
            .expect("adapter");
        let outcome = state
            .try_reserve_reclaiming_idle(2, 2, 0, 0, snapshot)
            .expect("reserve");
        assert!(
            matches!(outcome, MemoryReserveOutcome::Admitted),
            "{outcome:?}"
        );
        assert_eq!(rungs(&state), 0);
    }

    /// A lone admission of 5 blocks on a 4-block pool, max 8, whose 4
    /// blocks are a ladder chain: the grow clamps at the max and adds 4,
    /// still short, so the take goes on to the rung and admits.
    /// Mutation: count a passing grow as met in `kv_need_met` → red
    /// (`Reject`: the grow came back `Met` with nothing taken).
    #[test]
    fn capped_admission_grow_takes_the_ladder() {
        let Some(mut state) = qwen35_pool_state(4, 8, passing()) else {
            return;
        };
        rung(&mut state, 16, 4, 0);
        let snapshot = state
            .inner
            .scheduler_cache_snapshot()
            .expect("snapshot")
            .expect("adapter");
        assert_eq!(
            snapshot.blocks.free_blocks + snapshot.blocks.reclaimable_blocks,
            0
        );
        clear_reclaim_log();
        let outcome = state
            .try_reserve_reclaiming_idle(2, 5, 0, 0, snapshot)
            .expect("reserve");
        assert!(
            matches!(outcome, MemoryReserveOutcome::Admitted),
            "{outcome:?}"
        );
        assert!(reclaim_log().contains(&GrowReclaim::Ladder));
        assert_eq!(rungs(&state), 0);
        assert_eq!(pool_total_blocks(&state), 8);
    }

    // ---- the lone blocked turn reaches in-use points (Splash `growthClass`) ----

    fn has_rung(state: &HybridSchedulerState<Qwen35Inner>, final_hash: u64) -> bool {
        state
            .inner
            .ladder_entries_for_test()
            .iter()
            .any(|&(_, hash, _)| hash == final_hash)
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

    /// The step's only running turn T (3 blocks) blocks for one block at the
    /// max pool; the rest of the pool is the 2-block cache-only chain of
    /// rung R, which suspended turn 7 restored from. The lone turn's need
    /// reaches R, takes it, and runs on.
    /// Mutation: `handle_blocked_alone` reclaims with `reach_in_use` false
    /// → red (T fails `capacity_exhausted`).
    #[cfg(target_os = "macos")]
    #[test]
    fn lone_turn_takes_a_suspended_turns_replay_point_last() {
        let Some(mut state) = qwen35_pool_state(5, 5, passing()) else {
            return;
        };
        let (turn, mut result) = running_turn(&mut state, 1);
        let r = *rung(&mut state, 30, 2, 0).last().expect("R");
        state.inner.use_ladder_entry_for_test(7, r);
        let adapter = state.inner.paged_adapter().expect("paged adapter");
        assert!(adapter.at_cap());
        assert_eq!(adapter.takeable_blocks(), 0);
        adapter.refuse_for_test(1, 1);
        settle_blocked_step(&mut state, turn);
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

    /// As above beside an ordinary rung Q (1 block) newer than R: the
    /// need takes Q and keeps R and its use.
    /// Mutation: take in-use points in the ordinary loop by stamp in
    /// `free_for_kv` → red (R, the older, goes first).
    #[test]
    fn ordinary_units_go_before_an_in_use_point() {
        let Some(mut state) = qwen35_pool_state(6, 6, passing()) else {
            return;
        };
        hold(&mut state, 9, 3);
        let r = *rung(&mut state, 30, 2, 0).last().expect("R");
        let q = *rung(&mut state, 31, 1, 100).last().expect("Q");
        state.inner.use_ladder_entry_for_test(7, r);
        refused(&mut state, 9, 3, 1);
        assert!(state.reclaim_for_blocked_step(9, true));
        assert!(!has_rung(&state, q), "Q was taken");
        assert!(has_rung(&state, r), "R stays");
        assert!(matches!(
            state.inner.gdn_in_use_for_test(7),
            Some(UnitKey::Ladder { final_block_hash, .. }) if *final_block_hash == r
        ));
    }

    /// Two running turns at the max pool beside rung R, which the later
    /// one (2) restored from; both block, so the later one yields: its
    /// reclaim keeps R and it is preempted, still using R.
    #[cfg(target_os = "macos")]
    fn yielding_pair() -> Option<(HybridSchedulerState<Qwen35Inner>, u64)> {
        let mut state = qwen35_pool_state(8, 8, passing())?;
        let (first, _) = running_turn(&mut state, 1);
        let (second, _) = running_turn(&mut state, 2);
        let r = *rung(&mut state, 30, 2, 0).last().expect("R");
        state.inner.use_ladder_entry_for_test(2, r);
        let adapter = state.inner.paged_adapter().expect("paged adapter");
        assert!(adapter.at_cap());
        assert_eq!(adapter.takeable_blocks(), 0);
        adapter.refuse_for_test(2, 1);
        state.scheduler.enqueue_turn(first).expect("enqueue");
        state.scheduler.enqueue_turn(second).expect("enqueue");
        let Ok(SchedulerAction::Stepped {
            completed,
            preempted,
            blocked_alone,
            ..
        }) = state.scheduler.drive_once(&mut BlockEveryRow)
        else {
            panic!("expected a step");
        };
        assert!(blocked_alone.is_none());
        assert_eq!(preempted.as_ref().map(|turn| turn.turn.seq_id), Some(2));
        assert!(!state.settle_step(completed, preempted, blocked_alone));
        Some((state, r))
    }

    /// Mutation: `settle_step` reclaims with `reach_in_use` true → red (R
    /// taken, no preemption).
    #[cfg(target_os = "macos")]
    #[test]
    fn yielding_lane_does_not_take_in_use() {
        let Some((state, r)) = yielding_pair() else {
            return;
        };
        assert!(has_rung(&state, r), "R stays");
        assert_eq!(state.scheduler.stats().preemptions, 1);
        assert_eq!(state.scheduler.running().len(), 1);
    }

    /// A preempted turn keeps its replay point in use, suspended or not
    /// (Splash `Engine::Request::replayPoint`), through a resume that waits.
    /// Mutation: end the use in `handle_preempted`
    /// (`scheduled_turn_closed`) → red (`None`).
    #[cfg(target_os = "macos")]
    #[test]
    fn preemption_keeps_the_use() {
        let Some((mut state, r)) = yielding_pair() else {
            return;
        };
        let used = |state: &HybridSchedulerState<Qwen35Inner>| {
            matches!(
                state.inner.gdn_in_use_for_test(2),
                Some(UnitKey::Ladder { final_block_hash, .. }) if *final_block_hash == r
            )
        };
        assert!(used(&state));
        state.try_resume_preempted();
        assert_eq!(state.scheduler.running().len(), 1, "the resume waits");
        assert!(used(&state));
    }

    /// The kept prompt wins over the in-use reach: a prompt that restores
    /// R keeps it out of the in-use units too.
    /// Mutation: drop `kept.keeps` from `gdn_unit_excluded` for in-use
    /// units → red (R offered).
    #[test]
    fn kept_prompt_beats_in_use_reach() {
        let Some(mut state) = qwen35_pool_state(4, 4, passing()) else {
            return;
        };
        let r = *rung(&mut state, 30, 2, 0).last().expect("R");
        state.inner.use_ladder_entry_for_test(7, r);
        let mut prompt = tokens_of(30, 2);
        prompt.push(1);
        let skip = HashSet::new();
        let offered = |keep: crate::models::qwen3_5::dflash2_turn::Keep<'_>, in_use| {
            state
                .inner
                .oldest_gdn_state_unit(true, &skip, keep, in_use)
                .map(|unit| unit.key)
        };
        let keep = crate::models::qwen3_5::dflash2_turn::Keep {
            tokens: &prompt,
            keys: None,
        };
        assert_eq!(offered(keep, true), None);
        assert_eq!(offered(keep, false), None);
        assert!(offered(Default::default(), true).is_some());
        assert_eq!(offered(Default::default(), false), None);
    }

    /// A taken in-use point keeps its use, as Splash's `StateUse` outlives
    /// the state: a rung rebuilt on the same chain is in use again until
    /// the turn ends.
    /// Mutation: drop the turn's record in `drop_gdn_state` → red (`None`).
    #[test]
    fn taken_in_use_point_keeps_its_use() {
        let Some(mut state) = qwen35_pool_state(5, 5, passing()) else {
            return;
        };
        hold(&mut state, 9, 3);
        let r = *rung(&mut state, 30, 2, 0).last().expect("R");
        state.inner.use_ladder_entry_for_test(7, r);
        refused(&mut state, 9, 3, 1);
        assert!(state.reclaim_for_blocked_step(9, true));
        assert!(!has_rung(&state, r));
        assert!(state.inner.gdn_in_use_for_test(7).is_some());
        let rebuilt = *rung(&mut state, 30, 2, 0).last().expect("R again");
        assert_eq!(rebuilt, r);
        let skip = HashSet::new();
        let offered = |state: &HybridSchedulerState<Qwen35Inner>, in_use| {
            state
                .inner
                .oldest_gdn_state_unit(false, &skip, Default::default(), in_use)
                .map(|unit| unit.key)
        };
        assert_eq!(offered(&state, false), None, "the rebuilt rung is in use");
        assert!(offered(&state, true).is_some());
        state.inner.end_gdn_use(7);
        assert!(offered(&state, false).is_some());
    }

    /// MoE twin of `lone_turn_takes_a_suspended_turns_replay_point_last`.
    /// Mutation: `handle_blocked_alone` reclaims with `reach_in_use` false
    /// → red (T fails `capacity_exhausted`).
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
        let adapter = state.inner.paged_adapter().expect("paged adapter");
        assert!(adapter.at_cap());
        assert_eq!(adapter.takeable_blocks(), 0);
        adapter.refuse_for_test(1, 1);
        settle_blocked_step(&mut state, turn);
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

    /// An admission charges no block for the marked chain its prompt hits:
    /// the free blocks hold the rest, so nothing is taken.
    /// Mutation: drop `- marked_hit_blocks` from `admission_charge` → red
    /// (not admitted).
    #[test]
    fn admission_charges_no_block_for_its_marked_hit() {
        let Some(mut state) = qwen35_pool_state(7, 7, passing()) else {
            return;
        };
        rung(&mut state, 13, 2, 0);
        let own = tokens_of(14, 3);
        let hashes = cache_dead_chain(&state, &own);
        let stamp = state.inner.cache_recency.next();
        state.inner.push_ladder_entry_for_test(
            own.len() as u32,
            hashes,
            own.clone(),
            Vec::new(),
            stamp,
        );
        let mut prompt = own;
        prompt.extend(tokens_of(15, 2));
        let charge = state.admission_charge(5, 2, &prompt, false);
        assert_eq!(charge, 2);
        let snapshot = state
            .inner
            .scheduler_cache_snapshot()
            .expect("snapshot")
            .expect("adapter");
        assert_eq!(snapshot.blocks.free_blocks, 2);
        state.reclaim_keep.clone_from(&prompt);
        let outcome = state
            .try_reserve_reclaiming_idle(2, charge, 0, 0, snapshot)
            .expect("reserve");
        state.reclaim_keep.clear();
        assert!(
            matches!(outcome, MemoryReserveOutcome::Admitted),
            "{outcome:?}"
        );
        assert_eq!(rungs(&state), 2);
    }

    /// The MoE admission charges no block for the marked chain its prompt
    /// hits.
    /// Mutation: the MoE `HybridSchedulerBackend::marked_hit_blocks` returns
    /// 0 → red (5 ≠ 2).
    #[test]
    fn moe_admission_charges_no_block_for_its_marked_hit() {
        if !crate::models::qwen3_5::dflash2_decode::tests::metal_is_default_device() {
            return;
        }
        let Ok(mut inner) = Qwen35MoeInner::new(tiny_moe_config()) else {
            return;
        };
        let recency = Arc::clone(&inner.cache_recency);
        inner.paged_adapter = Some(tiny_paged_adapter_with(recency, 7, 7));
        let own = tokens_of(14, 3);
        let hashes = {
            let adapter = inner.paged_adapter.as_ref().expect("paged adapter");
            let mut allocator = adapter.allocator().lock().expect("allocator");
            let blocks: Vec<_> = (0..3)
                .map(|_| allocator.allocate().expect("a chain block"))
                .collect();
            allocator
                .cache_full_blocks(&own, &blocks, 16, &[], 0)
                .expect("cache the chain");
            for block in blocks {
                allocator.free(block);
            }
            mlx_paged_attn::chain_hashes(&own, 16, &[], 0)
        };
        inner.push_ladder_entry_for_test(own.len() as u32, hashes, own.clone(), Vec::new(), 0);
        let state = HybridSchedulerState::new(inner).expect("construct MoE scheduler");
        let mut prompt = own;
        prompt.extend(tokens_of(15, 2));
        assert_eq!(state.admission_charge(5, 2, &prompt, false), 2);
    }

    /// A lone MTP turn at the cap fits its draft over a marked chain, as it
    /// did before marks: the draft fit is a byte estimate.
    /// Mutation: drop the `marked` term in `draft_growth_fits` → red.
    #[cfg(target_os = "macos")]
    #[test]
    fn lone_mtp_turn_at_the_cap_fits_its_draft_over_a_marked_chain() {
        let Some((mut state, admission)) = mtp_step_state_with(64, 64) else {
            return;
        };
        let lone = vec![mtp_step_turn(&mut state, 1, 250, admission)];
        assert_eq!(
            step_state_bytes(&state.inner, &lone[0], 254, 8_192),
            463_104
        );
        let chain: Vec<u32> = (0..57 * 16).map(|i| i % 13 + 1).collect();
        let hashes = cache_dead_chain(&state, &chain);
        hold(&mut state, 99, 7);
        let adapter = state.inner.paged_adapter().expect("paged adapter");
        adapter.set_state_marks([*hashes.last().expect("final")]);
        let snapshot = state
            .inner
            .scheduler_cache_snapshot()
            .expect("snapshot")
            .expect("adapter");
        assert_eq!(
            (
                snapshot.blocks.free_blocks,
                snapshot.blocks.reclaimable_blocks
            ),
            (0, 0)
        );
        assert!(
            HybridStepExecutor::new(&mut state.inner)
                .draft_growth_fits(&lone, 0, 4)
                .expect("lone step")
        );
    }

    /// The MoE family takes a refused step's need in the same order: at the
    /// cap a ladder rung goes and the step's need is met.
    /// Mutation: the MoE `reclaims_in_order` returns `false` → red (the
    /// owner loop finds nothing).
    #[test]
    fn moe_cap_reclaim_takes_a_ladder_rung() {
        if !crate::models::qwen3_5::dflash2_decode::tests::metal_is_default_device() {
            return;
        }
        let Ok(mut inner) = Qwen35MoeInner::new(tiny_moe_config()) else {
            return;
        };
        let recency = Arc::clone(&inner.cache_recency);
        inner.paged_adapter = Some(tiny_paged_adapter_with(recency, 4, 4));
        let tokens = tokens_of(16, 2);
        let hashes = {
            let adapter = inner.paged_adapter.as_ref().expect("paged adapter");
            let mut allocator = adapter.allocator().lock().expect("allocator");
            let blocks: Vec<_> = (0..2)
                .map(|_| allocator.allocate().expect("a chain block"))
                .collect();
            allocator
                .cache_full_blocks(&tokens, &blocks, 16, &[], 0)
                .expect("cache the chain");
            for block in blocks {
                allocator.free(block);
            }
            mlx_paged_attn::chain_hashes(&tokens, 16, &[], 0)
        };
        inner.push_ladder_entry_for_test(32, hashes, tokens, Vec::new(), 0);
        let mut state = HybridSchedulerState::new(inner).expect("construct MoE scheduler");
        let adapter = state.inner.paged_adapter_mut().expect("paged adapter");
        adapter.begin_request(9).expect("begin");
        adapter
            .allocate_suffix_blocks_for(9, 32)
            .expect("hold two blocks");
        assert!(adapter.reserve_rows_for(9, 48).is_err());
        clear_reclaim_log();
        assert!(state.reclaim_for_blocked_step(9, false));
        assert_eq!(reclaim_log(), vec![GrowReclaim::Ladder]);
    }

    /// Contention preempts an MTP victim at a long history; alone, it
    /// resumes into a max pool its replayed rows and its admission state
    /// charge fill exactly, which draft state re-priced on the whole history
    /// overflows.
    /// Mutation: re-price `recurrent_state_bytes` with
    /// `scheduled_draft_state_bytes` of the booked tokens in
    /// `book_resume` → red (Reject).
    #[cfg(target_os = "macos")]
    #[test]
    fn lone_resume_after_contention_fits_the_max_pool() {
        let Some((mut state, live)) = growable_qwen35_state() else {
            return;
        };
        let prompt: Vec<u32> = (0..40).map(|token| token % 16).collect();
        let recurrent = state.inner.recurrent_state_bytes();
        let admission = recurrent + state.inner.scheduled_draft_state_bytes(56);
        let mut turn = speculative::tests::request(
            &mut state.inner,
            2,
            prompt,
            true,
            Arc::new(AtomicBool::new(false)),
        )
        .with_block_reservation(4, 0, 16)
        .with_recurrent_state_reservation(admission);
        turn.payload.requested_tokens = 78_464;
        state.book_resume(&mut turn, 1_000, true, 0, false);
        let blocks = turn.block_reservation_total;
        let bytes_per_block = live.bytes_per_block;
        let rows_bytes =
            u64::from(blocks) * bytes_per_block + state.inner.scheduled_recurrent_bytes();
        let max_blocks = (rows_bytes + admission).div_ceil(bytes_per_block) as u32;
        let repriced = recurrent + state.inner.scheduled_draft_state_bytes(blocks * 16);
        assert!(
            rows_bytes + repriced > u64::from(max_blocks) * bytes_per_block,
            "the re-priced charge overflows the max pool"
        );
        let snapshot = SchedulerCacheSnapshot {
            blocks: BlockTelemetry {
                total_blocks: max_blocks,
                free_blocks: max_blocks,
                reclaimable_blocks: 0,
                allocated_blocks: 0,
            },
            max_total_blocks: max_blocks,
            bytes_per_block,
        };
        let outcome = state
            .try_reserve_with_policy(
                2,
                blocks,
                turn.recurrent_state_bytes,
                0,
                snapshot,
                false,
                <Qwen35Inner as HybridSchedulerBackend>::release_scheduled_cache,
            )
            .expect("resume decision");
        assert!(
            matches!(outcome, MemoryReserveOutcome::Admitted),
            "{outcome:?}"
        );
    }

    /// A cold replay rebuilds an MTP victim's draft K/V for its whole
    /// history, so its resume is charged what a running turn holds at the
    /// booked rows, blocked or not, and a max pool that fits only the
    /// admission charge beside the rows rejects it.
    /// Mutation: delete the `scheduled_speculation` re-charge in
    /// `book_resume` → red (charge 435,968 ≠ 10,642,176).
    #[cfg(target_os = "macos")]
    #[test]
    fn resumed_mtp_victim_charges_its_rebuilt_draft_state() {
        let Some((mut state, live)) = growable_qwen35_state() else {
            return;
        };
        state.inner.config.n_mtp_layers = 1;
        let prompt: Vec<u32> = (0..40).map(|token| token % 16).collect();
        let recurrent = state.inner.recurrent_state_bytes();
        let admission = recurrent + state.inner.scheduled_draft_state_bytes(56);
        let victim = |state: &mut HybridSchedulerState<Qwen35Inner>, seq: u32| {
            let mut turn = speculative::tests::request(
                &mut state.inner,
                seq,
                prompt.clone(),
                true,
                Arc::new(AtomicBool::new(false)),
            )
            .with_block_reservation(4, 0, 16)
            .with_recurrent_state_reservation(admission);
            turn.payload.requested_tokens = 78_464;
            turn
        };
        let rebuilt = recurrent + state.inner.scheduled_draft_state_held_bytes(40_016);
        let mut turn = victim(&mut state, 2);
        state.book_resume(&mut turn, 40_000, true, 0, false);
        assert_eq!(turn.block_reservation_total, 2_501);
        assert_eq!(turn.recurrent_state_bytes, rebuilt);
        let mut unblocked = victim(&mut state, 3);
        state.book_resume(&mut unblocked, 40_000, false, 2_501, false);
        assert_eq!(unblocked.recurrent_state_bytes, rebuilt);
        let blocks = turn.block_reservation_total;
        let bytes_per_block = live.bytes_per_block;
        let rows_bytes =
            u64::from(blocks) * bytes_per_block + state.inner.scheduled_recurrent_bytes();
        let max_blocks = (rows_bytes + admission).div_ceil(bytes_per_block) as u32;
        let snapshot = SchedulerCacheSnapshot {
            blocks: BlockTelemetry {
                total_blocks: max_blocks,
                free_blocks: max_blocks,
                reclaimable_blocks: 0,
                allocated_blocks: 0,
            },
            max_total_blocks: max_blocks,
            bytes_per_block,
        };
        let outcome = state
            .try_reserve_with_policy(
                2,
                blocks,
                turn.recurrent_state_bytes,
                0,
                snapshot,
                false,
                <Qwen35Inner as HybridSchedulerBackend>::release_scheduled_cache,
            )
            .expect("resume decision");
        assert!(
            matches!(outcome, MemoryReserveOutcome::Reject { .. }),
            "{outcome:?}"
        );
    }

    /// A preempted MTP victim books blocks for its replayed history; its
    /// state charge is never less than its admission charge and prices no
    /// hidden rows for the whole history.
    /// Mutation: re-price `recurrent_state_bytes` with
    /// `scheduled_draft_state_bytes` of the booked tokens in
    /// `book_resume` → red.
    #[cfg(target_os = "macos")]
    #[test]
    fn resumed_victim_keeps_its_admission_state_charge() {
        let Some((mut state, _)) = growable_qwen35_state() else {
            return;
        };
        let prompt: Vec<u32> = (0..40).map(|token| token % 16).collect();
        let admission =
            state.inner.recurrent_state_bytes() + state.inner.scheduled_draft_state_bytes(56);
        let mut turn = speculative::tests::request(
            &mut state.inner,
            2,
            prompt,
            true,
            Arc::new(AtomicBool::new(false)),
        )
        .with_block_reservation(4, 0, 16)
        .with_recurrent_state_reservation(admission);
        turn.payload.requested_tokens = 78_464;
        state.book_resume(&mut turn, 40_000, true, 0, false);
        assert_eq!(
            turn.block_reservation_total,
            resume_booking_blocks(4, 40_000, 3, 78_464, 16, false)
        );
        assert_eq!(turn.recurrent_state_bytes, admission);
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
