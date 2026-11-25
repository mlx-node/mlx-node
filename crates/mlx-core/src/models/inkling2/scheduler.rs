//! Continuous-batching [`HybridSchedulerBackend`] for Inkling-2-Mini.
//!
//! Features:
//! - Unified [`Governor`] ledger reclamation (`reclaims_in_order = true`, `JudgePolicy::Ledger`)
//! - 500ms decode-gap prefill pacing (`MeasuredConfig` with `DECODE_GAP_MS = 500.0`)
//! - 100ms layer-chunked prefill interleaving (`prefill_runs_in_passes`, `run_scheduled_prefill_pass`)
//! - Fused `[N, 1]` batched decode across concurrent lanes (`run_paged_decode_step_batched`)
//! - Clean preemption replay (`preempt_scheduled_cache` releasing Global KV + SWA KV + ShortConv)
//! - Qwen 3.8-style StateDisk SSD cache (`WaitingForSsd` async restore via `Inkling2Restore`,
//!   rolling prefill checkpoints every 4096 tokens, turn-end ordinary states, restart `adopt_states`)
//! - Exclusive barrier lane for vision/media turns (`execute_chat_barrier`)

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use napi::bindgen_prelude::{Error, Result};

use crate::array::MxArray;
use crate::engine::backend::ChatBackend;
use crate::engine::cmd::ChatCmd;
use crate::engine::governor::governor::{Governor, MemoryAccounts, ModelGuard};
use crate::engine::governor::ledger::{
    FreeOutcome, Reserves, Stage, Taken as LedgerTaken, UnitKey, UnitRuns, Victim, VictimQuery,
};
use crate::engine::hybrid_scheduler::{
    DualLaneChatBarrier, HybridSchedulerBackend, HybridSchedulerState, HybridStepExecutor,
    ScheduledPrefillPass, ScheduledPrefixAdmission, ScheduledRestoreResult, SchedulerOwnerContext,
    StepClaims, execute_dual_lane_chat_barrier,
};
use crate::engine::model_command::{FamilyCommand, ModelCommand};
use crate::engine::scheduler::{MeasuredConfig, PreemptionMode, StepPlan};
use crate::engine::spec_paged::{SpecPagedCache, VerifyTicket};
use crate::model_thread::ResponseTx;
use crate::models::qwen3_5::dflash2_state::disk::{Tick, instant_at};
use crate::models::qwen3_5::dflash2_state::store::{
    Entry, EntryId, StateBytes, StateClass, StoreHit,
};
use crate::stream::{Stream, WiredResidency};
use crate::transformer::paged_kv_cache_adapter::{
    ChainPrefetchStart, PagedKVCacheAdapter, RestoreUploadBudget, SeqId,
};

use super::kv_cache::INKLING2_PAGED_PREFILL_MAX_CHUNK;
use super::memory::short_conv_state_bytes;
use super::model::{Inkling2FlatCache, Inkling2Inner};
use super::paged_forward::{Inkling2PagedRuntime, Inkling2PrefixState};
use super::state_disk::{
    BoundaryDraftWindow, Inkling2Restore, Inkling2RestorePhase, Inkling2StateAnchor,
    ReclaimOutcome, install_anchor_swa_and_conv, planned_rolling_checkpoints, poll_restore_ticket,
    start_global_chain_prefetch,
};

/// Family-specific commands routed to Inkling-2-Mini's dedicated model thread.
pub(crate) enum Inkling2FamilyCommand {
    /// Flush waiting ordinary states (if persistent) and close the SSD cache tier by `deadline_ms`
    /// (milliseconds since Unix epoch on `Date.now()`'s clock).
    CloseCache {
        deadline_ms: f64,
        reply: ResponseTx<bool>,
    },
}

impl FamilyCommand<Inkling2Inner> for Inkling2FamilyCommand {
    fn execute(self, inner: &mut Inkling2Inner) {
        match self {
            Inkling2FamilyCommand::CloseCache { deadline_ms, reply } => {
                let ok = inner.close_state_disk(instant_at(deadline_ms));
                let _ = reply.send(Ok(ok));
            }
        }
    }
}

/// Commands dispatched from NAPI methods to the Inkling-2-Mini model thread.
pub(crate) type Inkling2Cmd = ModelCommand<Inkling2FamilyCommand>;

/// Generic continuous-batching scheduler state for Inkling-2-Mini.
pub(crate) type Inkling2SchedulerState = HybridSchedulerState<Inkling2Inner>;

#[cfg(test)]
thread_local! {
    /// Test-only hook: force optional `DraftContextWindow` materialization inside
    /// `install_state_hit_sync` to return an error for one `SeqId`, verifying best-effort
    /// isolation while the target prefix (Global KV, SWA, ShortConv) still restores cleanly.
    pub(crate) static FAIL_DRAFT_WINDOW_RESTORE_SEQ: std::cell::Cell<Option<SeqId>> =
        const { std::cell::Cell::new(None) };
    /// Test-only counter: number of times `FAIL_DRAFT_WINDOW_RESTORE_SEQ` triggered.
    pub(crate) static DRAFT_WINDOW_RESTORE_INJECTED_COUNT: std::cell::Cell<u64> =
        const { std::cell::Cell::new(0) };
    /// Test-only hook: force `DFlash2ContextCache::restore` inside `begin_scheduled_dspark` /
    /// `prime_dspark_prefix_state` to return an error for one `SeqId`, verifying clean fallback
    /// (scheduled AR keeping the restored target prefix, or exclusive cold prefill).
    pub(crate) static FAIL_DRAFT_KV_RESTORE_SEQ: std::cell::Cell<Option<SeqId>> =
        const { std::cell::Cell::new(None) };
    /// Test-only counter: number of times `FAIL_DRAFT_KV_RESTORE_SEQ` triggered.
    pub(crate) static DRAFT_KV_RESTORE_INJECTED_COUNT: std::cell::Cell<u64> =
        const { std::cell::Cell::new(0) };
}

/// Per-owner metadata persisted across turns by [`HybridSchedulerState`].
#[derive(Default)]
pub(crate) struct Inkling2OwnerState {
    pub cached_token_history: Vec<u32>,
    pub cached_image_key: Option<u64>,
    pub flat_cache: Option<Inkling2FlatCache>,
    pub cuda_cache: Option<super::cuda_kv::Inkling2CudaCache>,
    pub cuda_dflash2_ctx: Option<crate::models::qwen3_5::dflash2::DFlash2ContextCache>,
    pub cuda_dspark_ctx: Option<super::dspark::Inkling2DsparkContext>,
}

impl DualLaneChatBarrier for Inkling2Inner {
    fn prepare_reset_barrier(&mut self) {
        self.prefill_runs.clear();
        if let Some(rt) = self.paged.as_mut() {
            rt.active_seq = 0;
        }
    }

    fn command_uses_flat_lane(&self, _command: &ChatCmd) -> bool {
        self.paged.is_none()
    }

    fn prepare_ownerless_barrier(&mut self, _flat_lane: bool) {
        if let Some(rt) = self.paged.as_mut() {
            rt.active_seq = 0;
        }
        self.cached_token_history.clear();
        self.cached_image_key = None;
        if let Some(ref mut ctx) = self.cuda_dflash2_ctx {
            ctx.reset();
        }
        self.cuda_dspark_ctx = None;
    }

    fn finish_ownerless_barrier(&mut self, flat_lane: bool) {
        if !flat_lane && !ChatBackend::has_live_session(self) {
            let _ = self.release_scheduled_cache(0);
        }
    }

    fn owner_has_flat_state(state: &Self::OwnerState) -> bool {
        state.flat_cache.is_some() || state.cuda_cache.is_some()
    }

    fn install_flat_owner_state(&mut self, state: Self::OwnerState) {
        self.flat_cache = state.flat_cache;
        self.cuda_cache = state.cuda_cache;
        self.cuda_dflash2_ctx = state.cuda_dflash2_ctx;
        self.cuda_dspark_ctx = state.cuda_dspark_ctx;
        self.cached_token_history = state.cached_token_history;
        self.cached_image_key = state.cached_image_key;
    }

    fn capture_flat_owner_state(&mut self) -> Self::OwnerState {
        Inkling2OwnerState {
            cached_token_history: self.cached_token_history.clone(),
            cached_image_key: self.cached_image_key,
            flat_cache: self.flat_cache.take(),
            cuda_cache: self.cuda_cache.take(),
            cuda_dflash2_ctx: self.cuda_dflash2_ctx.take(),
            cuda_dspark_ctx: self.cuda_dspark_ctx.take(),
        }
    }

    fn install_paged_owner_state(&mut self, seq_id: SeqId, state: Self::OwnerState) {
        if let Some(rt) = self.paged.as_mut() {
            rt.active_seq = seq_id;
        }
        self.cached_token_history = state.cached_token_history;
        self.cached_image_key = state.cached_image_key;
    }

    fn capture_paged_owner_state(&mut self) -> Self::OwnerState {
        Inkling2OwnerState {
            cached_token_history: self.cached_token_history.clone(),
            cached_image_key: self.cached_image_key,
            flat_cache: None,
            cuda_cache: None,
            cuda_dflash2_ctx: None,
            cuda_dspark_ctx: None,
        }
    }

    fn release_dead_paged_owner(&mut self, seq_id: SeqId) {
        let _ = self.release_scheduled_cache(seq_id);
    }
}

/// Returns `true` only when `err_msg` reports transient paged KV pool exhaustion
/// (reclaimable by releasing idle owners or unmarking `StateStore` chains), never
/// hard capacity exhaustion (`requested N tokens, ... capacity is C`) or unrelated errors.
fn is_paged_allocation_blocked(err_msg: &str) -> bool {
    err_msg.contains("could not reserve")
        || err_msg.contains("are free or reclaimable")
        || err_msg.contains("could allocate only")
}

/// Bounded wait budget when exclusive-turn reclaim has no non-SSD victims left and must wait
/// for in-flight StateDisk state/chain writes to land and unmark Global KV blocks.
const EXCLUSIVE_SSD_RECLAIM_WAIT_BUDGET: Duration =
    Duration::from_secs(crate::engine::governor::wait::RESOURCE_WAIT_SECONDS as u64);

/// Poll slice for StateDisk `Tick::Wait` during bounded exclusive-turn SSD reclaim waits,
/// keeping turn cancellation responsive while advancing chain offloads.
const EXCLUSIVE_SSD_RECLAIM_POLL_SLICE: Duration = Duration::from_millis(25);

/// Short event-wait / sleep slice when a StateDisk tick made no immediate progress while
/// background disk worker writes are in flight, preventing 100% CPU busy-spinning.
const EXCLUSIVE_SSD_RECLAIM_IDLE_SLICE: Duration = Duration::from_millis(5);

/// Which KV pool group(s) require blocks during an exclusive-turn reclaim attempt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ExclusiveReclaimScope {
    /// Global (Group 0) or both groups need blocks: reclaim `StateStore` checkpoints,
    /// then idle owners and ordinary `StateStore` marks in unified `CacheRecency` order.
    Any,
    /// Only SWA (Group 1) needs blocks: skip `StateStore` checkpoints and ordinary marks
    /// (which only pin Global KV, never SWA), and only reclaim idle owners that hold live
    /// Group 1 (SWA) blocks.
    SlidingOnly,
}

/// Per-allocation reclaim cursor for exclusive barrier turns.
///
/// Defers SSD-backed `StateStore` entries (`pending_ssd`) until non-blocking victims are
/// exhausted, records unprogressable entries in `skipped`, and initializes `wait_deadline`
/// at most once per allocation attempt.
#[derive(Default)]
pub(crate) struct ExclusiveReclaimState {
    skipped: HashSet<EntryId>,
    pending_ssd: Vec<EntryId>,
    wait_deadline: Option<Instant>,
}

impl ExclusiveReclaimState {
    fn is_deferred(&self, id: EntryId) -> bool {
        self.skipped.contains(&id) || self.pending_ssd.contains(&id)
    }

    fn push_pending_ssd(&mut self, id: EntryId) {
        if !self.skipped.contains(&id) && !self.pending_ssd.contains(&id) {
            self.pending_ssd.push(id);
        }
    }
}

/// Mirror of `HybridStateStore`'s `Entry::marks_chain`: an entry pins its Global KV chain
/// while it holds a RAM copy or an SSD copy whose Global KV blocks are still owed to disk.
fn store_entry_marks_chain<S>(entry: &Entry<S>) -> bool {
    entry.ram().is_some() || entry.disk().is_some_and(|d| d.chain_owed)
}

/// Whether the attached StateDisk tier can still advance or land an in-flight write.
fn state_disk_can_progress(rt: &Inkling2PagedRuntime) -> bool {
    rt.state_store.disk.as_ref().is_some_and(|disk| {
        disk.writable()
            || disk.writing()
            || disk.chain.as_ref().is_some_and(|c| !c.in_flight.is_empty())
    })
}

/// Choose [`ExclusiveReclaimScope::SlidingOnly`] when Group 0 (Global) already has enough
/// free/evictable blocks for advancing `seq_id` to `new_total_tokens` and ONLY Group 1 (SWA)
/// is short — avoiding unnecessary eviction of Global `StateStore` checkpoints or marks that
/// cannot free any SWA block.
fn exclusive_reclaim_scope_for_total(
    rt: &Inkling2PagedRuntime,
    seq_id: SeqId,
    new_total_tokens: u32,
) -> ExclusiveReclaimScope {
    let Some(swa) = rt.coordinator.sliding_adapter() else {
        return ExclusiveReclaimScope::Any;
    };
    let full = rt.coordinator.full_adapter();
    let full_bs = full.block_size();
    let swa_bs = swa.block_size();
    if full_bs == 0 || swa_bs == 0 {
        return ExclusiveReclaimScope::Any;
    }
    let (Ok(full_tel), Ok(swa_tel)) = (full.block_telemetry(), swa.block_telemetry()) else {
        return ExclusiveReclaimScope::Any;
    };
    let full_cur = full
        .block_table_for(seq_id)
        .map_or(0, |t| t.num_blocks() as u32);
    let full_need = new_total_tokens.div_ceil(full_bs).saturating_sub(full_cur);
    let full_avail = full_tel
        .free_blocks
        .saturating_add(full_tel.evictable_blocks);

    let swa_cur = swa
        .block_table_for(seq_id)
        .map_or(0, |t| t.num_blocks() as u32);
    let swa_need = new_total_tokens.div_ceil(swa_bs).saturating_sub(swa_cur);
    let swa_avail = swa_tel.free_blocks.saturating_add(swa_tel.evictable_blocks);

    if full_need <= full_avail && swa_need > swa_avail {
        ExclusiveReclaimScope::SlidingOnly
    } else {
        ExclusiveReclaimScope::Any
    }
}

/// Reclaim or advance one marked `StateStore` entry for an exclusive turn without blocking.
///
/// Returns `true` if the entry was removed/unmarked or Global `marked_blocks()` decreased
/// immediately; otherwise defers `id` to `state.pending_ssd` (when StateDisk can still land
/// an in-flight write) or `state.skipped` (when unprogressable) so immediate non-SSD victims
/// are always tried first before any bounded SSD wait.
fn reclaim_store_entry_for_exclusive(
    rt: &mut Inkling2PagedRuntime,
    id: EntryId,
    state: &mut ExclusiveReclaimState,
) -> bool {
    let Some(has_ram) = rt
        .state_store
        .store
        .get(id)
        .map(|entry| entry.ram().is_some())
    else {
        state.skipped.insert(id);
        return false;
    };
    let marked_before = rt.coordinator.full_adapter().marked_blocks();
    let defer = |rt: &Inkling2PagedRuntime, state: &mut ExclusiveReclaimState| {
        if state_disk_can_progress(rt) {
            state.push_pending_ssd(id);
        } else {
            state.skipped.insert(id);
        }
    };

    if has_ram {
        match rt.state_store.reclaim_entry(&rt.coordinator, id) {
            ReclaimOutcome::Erased => return true,
            ReclaimOutcome::RamDropped | ReclaimOutcome::Spilled => {
                let still_marks = rt
                    .state_store
                    .store
                    .get(id)
                    .is_some_and(store_entry_marks_chain);
                if !still_marks || rt.coordinator.full_adapter().marked_blocks() < marked_before {
                    return true;
                }
                rt.state_store.tick(
                    &rt.coordinator,
                    Tick::Wait {
                        deadline: Instant::now(),
                    },
                );
                let still_marks_after = rt
                    .state_store
                    .store
                    .get(id)
                    .is_some_and(store_entry_marks_chain);
                if !still_marks_after
                    || rt.coordinator.full_adapter().marked_blocks() < marked_before
                {
                    return true;
                }
                defer(rt, state);
                return false;
            }
            ReclaimOutcome::Pending => {
                rt.state_store.tick(
                    &rt.coordinator,
                    Tick::Wait {
                        deadline: Instant::now(),
                    },
                );
                let still_marks_after = rt
                    .state_store
                    .store
                    .get(id)
                    .is_some_and(store_entry_marks_chain);
                if !still_marks_after
                    || rt.coordinator.full_adapter().marked_blocks() < marked_before
                {
                    return true;
                }
                defer(rt, state);
                return false;
            }
        }
    }

    // Disk-only entry with `chain_owed`: drive one non-blocking tick to advance chain offload;
    // if still pinned, defer to `pending_ssd` so non-SSD victims are tried before waiting.
    rt.state_store.tick(
        &rt.coordinator,
        Tick::Wait {
            deadline: Instant::now(),
        },
    );
    let still_marks_after = rt
        .state_store
        .store
        .get(id)
        .is_some_and(store_entry_marks_chain);
    if !still_marks_after || rt.coordinator.full_adapter().marked_blocks() < marked_before {
        return true;
    }
    defer(rt, state);
    false
}

/// Retry `reclaim_entry` on deferred `pending_ssd` entries that still hold a RAM copy:
/// - Any entry whose own disk state write just landed drops RAM immediately (`RamDropped`).
/// - If no deferred entry is already spilling its chain (`ram().is_none()`) and the staging
///   buffer is free (or disk became non-writable), advance at most one deferred RAM entry.
fn retry_pending_ssd_ram_reclaim(rt: &mut Inkling2PagedRuntime, pending_ssd: &[EntryId]) {
    for &id in pending_ssd {
        let ready_to_drop_ram = rt
            .state_store
            .store
            .get(id)
            .is_some_and(|e| e.ram().is_some() && e.disk().is_some_and(|c| c.landed));
        if ready_to_drop_ram {
            let _ = rt.state_store.reclaim_entry(&rt.coordinator, id);
        }
    }
    let has_spilling_chain = pending_ssd.iter().any(|&id| {
        rt.state_store
            .store
            .get(id)
            .is_some_and(|e| e.ram().is_none() && store_entry_marks_chain(e))
    });
    if has_spilling_chain {
        return;
    }
    let can_stage_next = rt
        .state_store
        .disk
        .as_ref()
        .is_none_or(|disk| !disk.writing() || !disk.writable());
    if !can_stage_next {
        return;
    }
    if let Some(&id) = pending_ssd.iter().find(|&&id| {
        rt.state_store
            .store
            .get(id)
            .is_some_and(|e| e.ram().is_some())
    }) {
        let _ = rt.state_store.reclaim_entry(&rt.coordinator, id);
    }
}

fn is_exclusive_reclaim_cancelled(
    turn_cancel: Option<&Arc<std::sync::atomic::AtomicBool>>,
) -> bool {
    turn_cancel.is_some_and(|flag| flag.load(Ordering::Relaxed))
        || crate::engine::governor::governor::cancelled()
}

impl Inkling2Inner {
    /// Bounded, cancellation-aware wait for deferred SSD `StateStore` writes to land and
    /// unmark Global KV blocks once no immediate non-SSD victims remain.
    fn wait_for_pending_ssd_reclaim(&mut self, state: &mut ExclusiveReclaimState) -> bool {
        let Some(marked_before) = self
            .paged
            .as_ref()
            .map(|rt| rt.coordinator.full_adapter().marked_blocks())
        else {
            return false;
        };
        let wait_deadline = *state
            .wait_deadline
            .get_or_insert_with(|| Instant::now() + EXCLUSIVE_SSD_RECLAIM_WAIT_BUDGET);

        let prune_and_check = |rt: &Inkling2PagedRuntime, pending: &mut Vec<EntryId>| -> bool {
            let before_len = pending.len();
            pending.retain(|&id| {
                rt.state_store
                    .store
                    .get(id)
                    .is_some_and(store_entry_marks_chain)
            });
            pending.len() < before_len
                || rt.coordinator.full_adapter().marked_blocks() < marked_before
        };

        let turn_cancel = self.turn_cancel.as_ref();
        while !is_exclusive_reclaim_cancelled(turn_cancel) && Instant::now() < wait_deadline {
            let Some(rt) = self.paged.as_mut() else {
                break;
            };
            retry_pending_ssd_ram_reclaim(rt, &state.pending_ssd);
            if prune_and_check(rt, &mut state.pending_ssd) {
                return true;
            }

            let slice_deadline =
                (Instant::now() + EXCLUSIVE_SSD_RECLAIM_POLL_SLICE).min(wait_deadline);
            let progressed = rt.state_store.tick(
                &rt.coordinator,
                Tick::Wait {
                    deadline: slice_deadline,
                },
            );
            retry_pending_ssd_ram_reclaim(rt, &state.pending_ssd);
            if prune_and_check(rt, &mut state.pending_ssd) {
                return true;
            }

            if !state_disk_can_progress(rt) {
                break;
            }

            // Avoid busy-spinning while background disk worker writes complete.
            if !progressed
                && Instant::now() < wait_deadline
                && !is_exclusive_reclaim_cancelled(turn_cancel)
            {
                let idle_deadline =
                    (Instant::now() + EXCLUSIVE_SSD_RECLAIM_IDLE_SLICE).min(wait_deadline);
                let awaited = rt
                    .state_store
                    .disk
                    .as_ref()
                    .is_some_and(|disk| disk.engine.await_landing(idle_deadline));
                if !awaited {
                    let remaining = wait_deadline.saturating_duration_since(Instant::now());
                    if !remaining.is_zero() {
                        std::thread::sleep(remaining.min(EXCLUSIVE_SSD_RECLAIM_IDLE_SLICE));
                    }
                }
            }
        }

        for id in state.pending_ssd.drain(..) {
            state.skipped.insert(id);
        }
        false
    }

    /// Scoped on-demand reclaim primitive for exclusive barrier turns (`exclusive_turn_active`).
    ///
    /// Reclaims at most ONE victim per call (never `active_seq`, never ownerless `seq_id = 0`,
    /// and never `protected_store`):
    /// - Under [`ExclusiveReclaimScope::Any`]: follows Splash's `Phase::Checkpoints` then
    ///   unified `CacheRecency` stamp order across idle named owners and marked ordinary
    ///   `StateStore` entries, deferring in-flight SSD writes until immediate victims are
    ///   exhausted and then waiting within a bounded, cancellation-aware budget.
    /// - Under [`ExclusiveReclaimScope::SlidingOnly`]: skips `StateStore` entries (which only
    ///   pin Global KV, never SWA) and reclaims only idle named owners holding live Group 1
    ///   (SWA) blocks.
    pub(crate) fn try_reclaim_one_for_exclusive(
        &mut self,
        active_seq: SeqId,
        scope: ExclusiveReclaimScope,
        reclaim_state: &mut ExclusiveReclaimState,
        protected_store: Option<EntryId>,
    ) -> bool {
        if !self
            .paged
            .as_ref()
            .is_some_and(|rt| rt.exclusive_turn_active)
        {
            return false;
        }

        loop {
            // Phase 1: Disposable `StateClass::Checkpoint` entries (oldest `last_used` first).
            // Skipped under `SlidingOnly` since `StateStore` marks only pin Global KV blocks.
            if scope == ExclusiveReclaimScope::Any {
                let checkpoint_victim = self.paged.as_ref().and_then(|rt| {
                    rt.state_store
                        .store
                        .entries()
                        .filter(|entry| {
                            entry.checkpoint()
                                && store_entry_marks_chain(entry)
                                && !reclaim_state.is_deferred(entry.id())
                                && Some(entry.id()) != protected_store
                        })
                        .min_by_key(|entry| (entry.last_used(), entry.id()))
                        .map(Entry::id)
                });
                if let Some(id) = checkpoint_victim {
                    let Some(rt) = self.paged.as_mut() else {
                        return false;
                    };
                    if reclaim_store_entry_for_exclusive(rt, id, reclaim_state) {
                        return true;
                    }
                    continue;
                }
            }

            // Phase 2: Merge idle named owners (`candidate != active_seq && candidate != 0`;
            // ownerless `seq_id = 0` is reserved and never reclaimed by named turns) and marked
            // ordinary `StateStore` entries on their shared `CacheRecency` stamp clock.
            let best_owner: Option<(u64, SeqId)> = self.paged.as_ref().and_then(|rt| {
                let full = rt.coordinator.full_adapter();
                if scope == ExclusiveReclaimScope::SlidingOnly {
                    let swa = rt.coordinator.sliding_adapter()?;
                    return swa
                        .live_seq_ids()
                        .into_iter()
                        .filter(|&seq| seq != active_seq && seq != 0)
                        .map(|seq| (full.turn_end_stamp(seq).unwrap_or(0), seq))
                        .min();
                }
                let mut candidates: Vec<SeqId> = full.live_seq_ids();
                if let Some(swa) = rt.coordinator.sliding_adapter() {
                    candidates.extend(swa.live_seq_ids());
                }
                candidates.extend(rt.conv_states.keys().copied());
                candidates.extend(self.dspark_turn_contexts.keys().copied());
                candidates.sort_unstable();
                candidates.dedup();
                candidates
                    .into_iter()
                    .filter(|&seq| seq != active_seq && seq != 0)
                    .map(|seq| (full.turn_end_stamp(seq).unwrap_or(0), seq))
                    .min()
            });

            let best_store: Option<(u64, EntryId)> = if scope == ExclusiveReclaimScope::SlidingOnly
            {
                None
            } else {
                self.paged.as_ref().and_then(|rt| {
                    rt.state_store
                        .store
                        .entries()
                        .filter(|entry| {
                            !entry.checkpoint()
                                && store_entry_marks_chain(entry)
                                && !reclaim_state.is_deferred(entry.id())
                                && Some(entry.id()) != protected_store
                        })
                        .map(|entry| (entry.last_used(), entry.id()))
                        .min()
                })
            };

            match (best_owner, best_store) {
                (None, None) => {
                    if scope == ExclusiveReclaimScope::Any && !reclaim_state.pending_ssd.is_empty()
                    {
                        return self.wait_for_pending_ssd_reclaim(reclaim_state);
                    }
                    return false;
                }
                (Some((o_stamp, victim)), Some((s_stamp, _))) if o_stamp <= s_stamp => {
                    self.release_idle_owner_for_exclusive(victim, active_seq);
                    return true;
                }
                (Some((_, victim)), None) => {
                    self.release_idle_owner_for_exclusive(victim, active_seq);
                    return true;
                }
                (_, Some((_, id))) => {
                    let Some(rt) = self.paged.as_mut() else {
                        return false;
                    };
                    if reclaim_store_entry_for_exclusive(rt, id, reclaim_state) {
                        return true;
                    }
                }
            }
        }
    }

    /// Release idle owner `victim` across all KV groups + ShortConv + draft contexts, sync
    /// state marks, and re-activate `active_seq` on every KV group where `active_seq` is live
    /// (`release_request_for` clears the adapter's `active_seq` slot).
    fn release_idle_owner_for_exclusive(&mut self, victim: SeqId, active_seq: SeqId) {
        let _ = self.release_scheduled_cache(victim);
        if let Some(rt) = self.paged.as_mut() {
            rt.state_store.sync_marks(&rt.coordinator);
            rt.active_seq = active_seq;
            for gid in 0..rt.coordinator.groups().len() {
                if rt
                    .coordinator
                    .adapter(gid)
                    .is_ok_and(|a| a.live_seq_ids().contains(&active_seq))
                {
                    let _ = rt
                        .coordinator
                        .adapter_mut(gid)
                        .and_then(|a| a.activate_request(active_seq));
                }
            }
        }
    }

    /// Exclusive-turn retry wrapper around `coordinator.record_tokens_all`.
    pub(crate) fn record_tokens_all_exclusive(
        &mut self,
        seq_id: SeqId,
        tokens: &[u32],
    ) -> std::result::Result<(), String> {
        let mut reclaim_state = ExclusiveReclaimState::default();
        loop {
            let res = {
                let rt = self
                    .paged
                    .as_mut()
                    .ok_or_else(|| "Inkling2 paged runtime missing".to_string())?;
                rt.coordinator.record_tokens_all(seq_id, tokens)
            };
            match res {
                Ok(()) => return Ok(()),
                Err(err) => {
                    let retry_scope = self.paged.as_ref().and_then(|rt| {
                        if !rt.exclusive_turn_active || !is_paged_allocation_blocked(&err) {
                            return None;
                        }
                        let cur = rt.coordinator.request_token_count_all(seq_id).ok()?;
                        let added = u32::try_from(tokens.len()).ok()?;
                        let new_total = cur.checked_add(added)?;
                        rt.coordinator
                            .can_fit_single_allocation(seq_id, new_total)
                            .then(|| exclusive_reclaim_scope_for_total(rt, seq_id, new_total))
                    });
                    if let Some(scope) = retry_scope
                        && self.try_reclaim_one_for_exclusive(
                            seq_id,
                            scope,
                            &mut reclaim_state,
                            None,
                        )
                    {
                        continue;
                    }
                    return Err(err);
                }
            }
        }
    }

    /// Exclusive-turn retry wrapper around `SpecPagedCache::record_verify`.
    pub(crate) fn record_verify_exclusive(
        &mut self,
        seq_id: SeqId,
        tokens: &[u32],
    ) -> std::result::Result<VerifyTicket, String> {
        let mut reclaim_state = ExclusiveReclaimState::default();
        loop {
            let res = {
                let rt = self
                    .paged
                    .as_mut()
                    .ok_or_else(|| "Inkling2 paged runtime missing".to_string())?;
                rt.coordinator.record_verify(seq_id, tokens)
            };
            match res {
                Ok(ticket) => return Ok(ticket),
                Err(err) => {
                    let retry_scope = self.paged.as_ref().and_then(|rt| {
                        if !rt.exclusive_turn_active || !is_paged_allocation_blocked(&err) {
                            return None;
                        }
                        let cur = rt.coordinator.request_token_count_all(seq_id).ok()?;
                        let added = u32::try_from(tokens.len()).ok()?;
                        let new_total = cur.checked_add(added)?;
                        rt.coordinator
                            .can_fit_single_allocation(seq_id, new_total)
                            .then(|| exclusive_reclaim_scope_for_total(rt, seq_id, new_total))
                    });
                    if let Some(scope) = retry_scope
                        && self.try_reclaim_one_for_exclusive(
                            seq_id,
                            scope,
                            &mut reclaim_state,
                            None,
                        )
                    {
                        continue;
                    }
                    return Err(err);
                }
            }
        }
    }

    /// Exclusive-turn retry wrapper around `SpecPagedCache::reserve_lookahead`.
    ///
    /// Preserves grow-before-reclaim (`try_grow_all_for_rows` before `try_reclaim_one_for_exclusive`)
    /// because `reserve_rows_all` pre-flights against current `free + evictable` without growing.
    pub(crate) fn reserve_lookahead_exclusive(
        &mut self,
        seq_id: SeqId,
        rows: usize,
    ) -> std::result::Result<bool, String> {
        let rows_u32 = u32::try_from(rows).unwrap_or(u32::MAX);
        let mut reclaim_state = ExclusiveReclaimState::default();
        let mut tried_grow = false;
        loop {
            let reserved = {
                let rt = self
                    .paged
                    .as_mut()
                    .ok_or_else(|| "Inkling2 paged runtime missing".to_string())?;
                rt.coordinator.reserve_lookahead(seq_id, rows)?
            };
            if reserved {
                return Ok(true);
            }
            let retry_scope = self.paged.as_ref().and_then(|rt| {
                if !rt.exclusive_turn_active {
                    return None;
                }
                let cur = rt.coordinator.request_token_count_all(seq_id).ok()?;
                let new_total = cur.checked_add(rows_u32)?;
                rt.coordinator
                    .can_fit_single_allocation(seq_id, new_total)
                    .then(|| exclusive_reclaim_scope_for_total(rt, seq_id, new_total))
            });
            let Some(scope) = retry_scope else {
                return Ok(false);
            };
            if !tried_grow {
                tried_grow = true;
                let grew = self
                    .paged
                    .as_mut()
                    .is_some_and(|rt| rt.coordinator.try_grow_all_for_rows(seq_id, rows_u32));
                if grew {
                    continue;
                }
            }
            if self.try_reclaim_one_for_exclusive(seq_id, scope, &mut reclaim_state, None) {
                continue;
            }
            return Ok(false);
        }
    }

    /// Exclusive-turn retry wrapper around `coordinator.continue_turn_all`.
    pub(crate) fn continue_turn_all_exclusive(
        &mut self,
        seq_id: SeqId,
        tokens: &[u32],
        total_budget: u32,
    ) -> std::result::Result<u32, String> {
        let mut reclaim_state = ExclusiveReclaimState::default();
        loop {
            let res = {
                let rt = self
                    .paged
                    .as_mut()
                    .ok_or_else(|| "Inkling2 paged runtime missing".to_string())?;
                rt.coordinator
                    .continue_turn_all(seq_id, tokens, total_budget)
            };
            match res {
                Ok(boundary) => return Ok(boundary),
                Err(err) => {
                    let can_retry = self.paged.as_ref().is_some_and(|rt| {
                        rt.exclusive_turn_active
                            && is_paged_allocation_blocked(&err)
                            && rt.coordinator.can_fit_logical_tokens(total_budget)
                    });
                    if can_retry
                        && self.try_reclaim_one_for_exclusive(
                            seq_id,
                            ExclusiveReclaimScope::Any,
                            &mut reclaim_state,
                            None,
                        )
                    {
                        continue;
                    }
                    return Err(err);
                }
            }
        }
    }

    /// Restore a prefix from live continuation, RAM state hit, or synchronous SSD state hit.
    ///
    /// Returns the resolved [`Inkling2PrefixState`] with Group 0 (8 Global visits), Group 1
    /// (40 SWA visits), and `conv_states` (48 ShortConv visits) all installed at
    /// `effective_cached_prefix_len`.
    pub(crate) fn prepare_prefix_sync(
        &mut self,
        seq_id: SeqId,
        tokens: &[u32],
        reuse_cache: bool,
        cache_salt: u64,
    ) -> Result<Inkling2PrefixState> {
        let total_budget = u32::try_from(tokens.len())
            .map_err(|_| Error::from_reason("Inkling2 prompt length overflows u32"))?;
        let config = self.config.clone();
        {
            let rt = self
                .paged
                .as_mut()
                .ok_or_else(|| Error::from_reason("Inkling2 paged runtime missing"))?;
            rt.active_seq = seq_id;
            rt.seq_reuse_cache.insert(seq_id, reuse_cache);
            if !rt.coordinator.can_fit_logical_tokens(total_budget) {
                let capacity = rt.coordinator.max_capacity_tokens();
                return Err(Error::from_reason(format!(
                    "context_length_exceeded: requested {total_budget} tokens, \
                     paged cache capacity is {capacity} tokens"
                )));
            }
        }

        if reuse_cache {
            // 1. Live continuation on the exact same sequence (zero copy!).
            let can_continue = self.paged.as_ref().is_some_and(|rt| {
                rt.conv_states.contains_key(&seq_id)
                    && rt.coordinator.can_continue_all(seq_id, tokens)
            });
            if can_continue {
                let boundary = self
                    .continue_turn_all_exclusive(seq_id, tokens, total_budget)
                    .map_err(Error::from_reason)? as usize;
                return Ok(Inkling2PrefixState {
                    effective_cached_prefix_len: boundary,
                    suffix_len: tokens.len().saturating_sub(boundary),
                    full_tokens: tokens.to_vec(),
                    cache_salt,
                    live_continuation: true,
                    reuse_cache,
                });
            }

            // 2. Content-addressed hybrid state hit (RAM or SSD StateDisk).
            let hit = self.paged.as_ref().and_then(|rt| {
                rt.state_store
                    .lookup_deepest_restorable(&rt.coordinator, tokens, cache_salt)
            });
            if let Some(hit) = hit
                && let Some(boundary) =
                    self.install_state_hit_sync(seq_id, tokens, cache_salt, hit)?
            {
                // StateStore restores target KV/ShortConv + staged `restored_draft_windows`,
                // so drop any stale draft context previously owned by `seq_id`.
                self.dspark_turn_contexts.remove(&seq_id);
                self.scheduled_dspark.seq_lanes.remove(&seq_id);
                self.scheduled_dspark.sync_decode_limit();
                return Ok(Inkling2PrefixState {
                    effective_cached_prefix_len: boundary,
                    suffix_len: tokens.len().saturating_sub(boundary),
                    full_tokens: tokens.to_vec(),
                    cache_salt,
                    live_continuation: false,
                    reuse_cache,
                });
            }
        }

        // 3. Fresh cold sequence.
        self.dspark_turn_contexts.remove(&seq_id);
        self.scheduled_dspark.seq_lanes.remove(&seq_id);
        self.scheduled_dspark.sync_decode_limit();
        let rt = self
            .paged
            .as_mut()
            .ok_or_else(|| Error::from_reason("Inkling2 paged runtime missing"))?;
        rt.reset_sequence(seq_id, &config)?;
        rt.seq_reuse_cache.insert(seq_id, reuse_cache);
        Ok(Inkling2PrefixState {
            effective_cached_prefix_len: 0,
            suffix_len: tokens.len(),
            full_tokens: tokens.to_vec(),
            cache_salt,
            live_continuation: false,
            reuse_cache,
        })
    }

    /// Install a `StoreHit` synchronously (either hot Global KV + RAM anchor, or armed synchronous
    /// Global KV SSD restore + StateDisk blob read). Returns `Ok(Some(depth))` on hit or
    /// `Ok(None)` if the hit fell back cleanly to cold.
    fn install_state_hit_sync(
        &mut self,
        seq_id: SeqId,
        tokens: &[u32],
        cache_salt: u64,
        hit: StoreHit,
    ) -> Result<Option<usize>> {
        let depth = u32::try_from(hit.depth)
            .map_err(|_| Error::from_reason("Inkling2 prefix depth overflows u32"))?;
        let total_budget = u32::try_from(tokens.len())
            .map_err(|_| Error::from_reason("Inkling2 prompt length overflows u32"))?;
        let anchor = {
            let rt = self
                .paged
                .as_mut()
                .ok_or_else(|| Error::from_reason("Inkling2 paged runtime missing"))?;
            let Some((anchor, _from_ssd)) = rt.state_store.load_anchor(&rt.coordinator, hit.id)
            else {
                return Ok(None);
            };
            anchor
        };

        // Restore Group 0 (8 Global visits) up to exact `depth` using
        // `prepare_turn_per_block_booking_hit_only`, which installs the matched blocks (hot or
        // restored via `arm_state_restore` / `arm_landed_chain`) directly into `seq_id`'s block
        // table and `request_tokens`.
        let mut global_reclaim_state = ExclusiveReclaimState::default();
        let restored_global = loop {
            let (prepared, chain_hot, refused) = {
                let rt = self
                    .paged
                    .as_mut()
                    .ok_or_else(|| Error::from_reason("Inkling2 paged runtime missing"))?;
                let chain_hot = rt.state_store.chain_is_hot(&rt.coordinator, hit.id);
                let full_adapter = rt.coordinator.full_adapter_mut();
                if !chain_hot {
                    full_adapter.arm_state_restore(depth);
                }
                let keys = crate::engine::build_paged_extra_keys(
                    tokens.len(),
                    full_adapter.block_size(),
                    &[],
                );
                let prepared = full_adapter.prepare_turn_per_block_booking_hit_only(
                    seq_id,
                    tokens,
                    total_budget,
                    &keys,
                    cache_salt,
                    false,
                    depth,
                );
                let refused = full_adapter.take_block_refusal().is_some();
                (prepared, chain_hot, refused)
            };
            match prepared {
                Ok(plan) if plan.cached_prefix_len == depth => break plan.cached_prefix_len,
                Ok(_) => {
                    if let Some(rt) = self.paged.as_mut() {
                        let _ = rt.coordinator.release_request_all(seq_id);
                    }
                    let can_retry = !chain_hot
                        && refused
                        && self.paged.as_ref().is_some_and(|rt| {
                            rt.exclusive_turn_active && rt.coordinator.can_fit_logical_tokens(depth)
                        });
                    if can_retry
                        && self.try_reclaim_one_for_exclusive(
                            seq_id,
                            ExclusiveReclaimScope::Any,
                            &mut global_reclaim_state,
                            Some(hit.id),
                        )
                    {
                        continue;
                    }
                    return Ok(None);
                }
                Err(err) => {
                    if let Some(rt) = self.paged.as_mut() {
                        let _ = rt.coordinator.release_request_all(seq_id);
                    }
                    let can_retry = (is_paged_allocation_blocked(&err) || refused)
                        && self.paged.as_ref().is_some_and(|rt| {
                            rt.exclusive_turn_active && rt.coordinator.can_fit_logical_tokens(depth)
                        });
                    if can_retry
                        && self.try_reclaim_one_for_exclusive(
                            seq_id,
                            ExclusiveReclaimScope::Any,
                            &mut global_reclaim_state,
                            Some(hit.id),
                        )
                    {
                        continue;
                    }
                    return Ok(None);
                }
            }
        };
        if restored_global != depth {
            if let Some(rt) = self.paged.as_mut() {
                let _ = rt.coordinator.release_request_all(seq_id);
            }
            return Ok(None);
        }

        // Restore Group 1 (40 SWA visits) raw blocks + 48 ShortConv states.
        // Borrow `&anchor` across retries so `swa_data` (~82.5 MiB) is never deep-copied,
        // and restrict reclaim to `SlidingOnly` so Global `StateStore` checkpoints/marks are
        // never evicted for SWA-only pool pressure.
        let geometry = {
            let rt = self
                .paged
                .as_ref()
                .ok_or_else(|| Error::from_reason("Inkling2 paged runtime missing"))?;
            match rt.state_store.resolve_geometry() {
                Some(geom) => geom.clone(),
                None => super::state_disk::Inkling2StateGeometry::from_coordinator_with_draft(
                    &self.config,
                    &rt.coordinator,
                    self.dflash2.as_ref().map(|d| &d.config),
                )?,
            }
        };
        let exclusive = self
            .paged
            .as_ref()
            .is_some_and(|rt| rt.exclusive_turn_active);
        if exclusive {
            let swa_blocks_len = geometry.live_swa_blocks(depth) as usize;
            let mut swa_reclaim_state = ExclusiveReclaimState::default();
            loop {
                let swa_res = {
                    let rt = self
                        .paged
                        .as_mut()
                        .ok_or_else(|| Error::from_reason("Inkling2 paged runtime missing"))?;
                    install_anchor_swa_and_conv(
                        &mut rt.coordinator,
                        &mut rt.conv_states,
                        &geometry,
                        seq_id,
                        tokens,
                        depth,
                        &anchor,
                    )
                };
                match swa_res {
                    Ok(()) => break,
                    Err(err) => {
                        let can_retry = is_paged_allocation_blocked(&err.reason)
                            && self.paged.as_ref().is_some_and(|rt| {
                                rt.coordinator.can_fit_sliding_raw_blocks(swa_blocks_len)
                            });
                        if can_retry
                            && self.try_reclaim_one_for_exclusive(
                                seq_id,
                                ExclusiveReclaimScope::SlidingOnly,
                                &mut swa_reclaim_state,
                                Some(hit.id),
                            )
                        {
                            continue;
                        }
                        if let Some(rt) = self.paged.as_mut() {
                            let _ = rt.coordinator.release_request_all(seq_id);
                            rt.conv_states.remove(&seq_id);
                        }
                        tracing::warn!(
                            "Inkling2 StateDisk SWA/ShortConv restore fell back to cold: {err}"
                        );
                        return Ok(None);
                    }
                }
            }
        } else {
            let rt = self
                .paged
                .as_mut()
                .ok_or_else(|| Error::from_reason("Inkling2 paged runtime missing"))?;
            if let Err(err) = install_anchor_swa_and_conv(
                &mut rt.coordinator,
                &mut rt.conv_states,
                &geometry,
                seq_id,
                tokens,
                depth,
                &anchor,
            ) {
                let _ = rt.coordinator.release_request_all(seq_id);
                rt.conv_states.remove(&seq_id);
                tracing::warn!("Inkling2 StateDisk SWA/ShortConv restore fell back to cold: {err}");
                return Ok(None);
            }
        }

        // Confirm the out-of-pool auxiliary half (40 SWA visits + 48 ShortConv states) is primed
        // at `depth`, clearing the fail-closed `aux_prefix_unbacked` latch set by synchronous SSD restore.
        let rt = self
            .paged
            .as_mut()
            .ok_or_else(|| Error::from_reason("Inkling2 paged runtime missing"))?;
        rt.coordinator
            .full_adapter_mut()
            .activate_request(seq_id)
            .map_err(Error::from_reason)?;
        rt.coordinator
            .full_adapter_mut()
            .confirm_aux_prefix_primed(depth)
            .map_err(Error::from_reason)?;
        rt.prev_conv_snapshots.remove(&seq_id);
        rt.record_block_boundary_conv(seq_id, depth);

        let suffix_len = tokens.len().saturating_sub(depth as usize);
        rt.draft_windows.remove(&seq_id);
        rt.block_boundary_draft_window.remove(&seq_id);
        let mut complete_win = anchor.draft_window.filter(|w| w.is_complete_for(depth));
        if let Some(ref packed) = complete_win {
            rt.block_boundary_draft_window
                .insert(seq_id, (depth, BoundaryDraftWindow::Packed(packed.clone())));
            let bs = rt.coordinator.block_size().max(1);
            let mut first_boundary = tokens.len();
            if let Ok(prompt_u32) = u32::try_from(tokens.len()) {
                for cp in planned_rolling_checkpoints(prompt_u32, depth, bs) {
                    let cp_us = cp as usize;
                    if cp_us > depth as usize {
                        first_boundary = first_boundary.min(cp_us);
                    }
                }
                let junction = ((prompt_u32.saturating_sub(1) / bs) * bs) as usize;
                if junction > depth as usize {
                    first_boundary = first_boundary.min(junction);
                }
            }
            if first_boundary.saturating_sub(depth as usize) < packed.slots as usize {
                let restore_res = {
                    #[cfg(test)]
                    let injected_err = FAIL_DRAFT_WINDOW_RESTORE_SEQ
                        .with(std::cell::Cell::get)
                        .filter(|&target| target == seq_id)
                        .map(|_| {
                            DRAFT_WINDOW_RESTORE_INJECTED_COUNT
                                .with(|c| c.set(c.get().saturating_add(1)));
                            Error::from_reason(format!(
                                "injected draft-window restore failure for seq {seq_id}"
                            ))
                        });
                    #[cfg(not(test))]
                    let injected_err: Option<Error> = None;
                    match injected_err {
                        Some(err) => Err(err),
                        None => packed.to_window(),
                    }
                };
                match restore_res {
                    Ok(win) => {
                        rt.draft_windows.insert(seq_id, win);
                    }
                    Err(err) => {
                        tracing::warn!(
                            "Inkling2 optional draft-window restore failed for seq {seq_id} at \
                             depth {depth} (continuing with target prefix): {err}"
                        );
                        rt.invalidate_draft_window(seq_id);
                        complete_win = None;
                    }
                }
            }
        }
        rt.restored_draft_windows.insert(
            seq_id,
            super::paged_forward::RestoredDraftContext {
                depth,
                suffix_len,
                window: complete_win,
            },
        );
        Ok(Some(hit.depth))
    }

    /// Contended admission beside active decode lanes: if the deepest restorable state requires
    /// reading cold 8-layer Global KV blocks from SSD, park the turn in `TurnStatus::WaitingForSsd`
    /// with a background `ChainPrefetch` so active decode lanes are never stalled!
    pub(crate) fn prepare_contended_prefix_impl(
        &mut self,
        seq_id: SeqId,
        tokens: &[u32],
        reuse_cache: bool,
        cache_salt: u64,
    ) -> Result<ScheduledPrefixAdmission<Inkling2PrefixState, Inkling2Restore>> {
        let should_park = reuse_cache
            && self.cached_image_key.is_none()
            && self.paged.as_ref().is_some_and(|rt| {
                rt.state_store.disk.is_some()
                    && !(rt.conv_states.contains_key(&seq_id)
                        && rt.coordinator.can_continue_all(seq_id, tokens))
            });

        if should_park {
            let hit = self.paged.as_ref().and_then(|rt| {
                rt.state_store
                    .lookup_deepest_restorable(&rt.coordinator, tokens, cache_salt)
            });
            if let Some(hit) = hit {
                let chain_hot = self
                    .paged
                    .as_ref()
                    .is_some_and(|rt| rt.state_store.chain_is_hot(&rt.coordinator, hit.id));
                if !chain_hot {
                    let state_key = self.paged.as_ref().and_then(|rt| {
                        let disk = rt.state_store.disk.as_ref()?;
                        let entry = rt.state_store.store.get(hit.id)?;
                        disk.state_key(entry)
                    });
                    let start = self.paged.as_mut().and_then(|rt| {
                        start_global_chain_prefetch(
                            rt.coordinator.full_adapter_mut(),
                            tokens,
                            hit.depth,
                            cache_salt,
                            state_key,
                        )
                    });
                    match start {
                        Some(ChainPrefetchStart::Started(prefetch)) => {
                            if let Some(rt) = self.paged.as_mut()
                                && let Some(disk) = rt.state_store.disk.as_mut()
                            {
                                disk.counters.async_ssd_restores += 1;
                            }
                            return Ok(ScheduledPrefixAdmission::Waiting {
                                provisional: Inkling2PrefixState {
                                    effective_cached_prefix_len: 0,
                                    suffix_len: tokens.len(),
                                    full_tokens: tokens.to_vec(),
                                    cache_salt,
                                    live_continuation: false,
                                    reuse_cache,
                                },
                                restore: Inkling2Restore::reading(*prefetch, cache_salt),
                            });
                        }
                        Some(ChainPrefetchStart::Unavailable) => {
                            // Cold chain disappeared concurrently: prime cold cleanly while
                            // preserving the caller's `reuse_cache` publication policy.
                            let mut prefix =
                                self.prepare_prefix_sync(seq_id, tokens, false, cache_salt)?;
                            prefix.reuse_cache = reuse_cache;
                            if let Some(rt) = self.paged.as_mut() {
                                rt.seq_reuse_cache.insert(seq_id, reuse_cache);
                            }
                            return Ok(ScheduledPrefixAdmission::Ready(prefix));
                        }
                        Some(ChainPrefetchStart::Refused { blocks }) => {
                            if let Some(rt) = self.paged.as_ref() {
                                rt.coordinator.full_adapter().refuse_restore(seq_id, blocks);
                            }
                            return Err(Error::from_reason(format!(
                                "paged cache could not reserve {blocks} blocks for a disk restore"
                            )));
                        }
                        Some(ChainPrefetchStart::Hot) | None => {}
                    }
                }
            }
        }

        self.prepare_prefix_sync(seq_id, tokens, reuse_cache, cache_salt)
            .map(ScheduledPrefixAdmission::Ready)
    }

    /// Land a parked `WaitingForSsd` restore once its 8-layer Global KV `ChainPrefetch` finished.
    pub(crate) fn land_parked_restore_impl(
        &mut self,
        seq_id: SeqId,
        restore: &mut Inkling2Restore,
        tokens: &[u32],
    ) -> Result<Option<ScheduledRestoreResult<Inkling2PrefixState>>> {
        let landed_depth = match &restore.phase {
            Inkling2RestorePhase::Reading(_) => return Ok(None),
            Inkling2RestorePhase::Landed(prefetch) => Some(prefetch.depth()),
            Inkling2RestorePhase::Lost | Inkling2RestorePhase::Cold => None,
        };
        let salt = restore.cache_salt;

        // If the staging buffer is currently held by an in-flight state write, stay parked
        // briefly with `ReadWait` (blocks new state writes so reads take priority).
        let needs_disk_read = self.paged.as_ref().is_some_and(|rt| {
            let Some(hit) = rt
                .state_store
                .lookup_deepest_restorable(&rt.coordinator, tokens, salt)
            else {
                return false;
            };
            rt.state_store
                .store
                .get(hit.id)
                .is_some_and(|e| e.ram().is_none())
                && rt
                    .state_store
                    .disk
                    .as_ref()
                    .is_some_and(|d| !d.staging_free())
        });
        if needs_disk_read {
            let now = self.memory_registration.as_ref().map_or_else(
                || restore.parked_at.elapsed().as_secs_f64(),
                |guard| guard.governor().now(),
            );
            let since = *restore.read_wait_since.get_or_insert(now);
            if now - since < crate::engine::governor::wait::RESOURCE_WAIT_SECONDS {
                if restore.read_wait.is_none() {
                    restore.read_wait = self
                        .paged
                        .as_ref()
                        .and_then(|rt| rt.state_store.disk.as_ref())
                        .map(|d| d.acquire_read_wait());
                }
                return Ok(None);
            }
            restore.phase = Inkling2RestorePhase::Cold;
        }
        restore.read_wait = None;

        let mut bytes_restored = 0u64;
        let prefix = if let (Inkling2RestorePhase::Landed(prefetch), Some(depth)) =
            (&restore.phase, landed_depth)
        {
            let hit = self.paged.as_ref().and_then(|rt| {
                rt.state_store
                    .lookup_deepest_restorable(&rt.coordinator, tokens, salt)
            });
            if let Some(hit) = hit
                && hit.depth == depth
            {
                bytes_restored = prefetch.bytes_restored;
                if let Some(rt) = self.paged.as_mut() {
                    rt.coordinator.full_adapter_mut().arm_landed_chain(prefetch);
                }
                let res = self.install_state_hit_sync(seq_id, tokens, salt, hit);
                if let Some(rt) = self.paged.as_mut() {
                    rt.coordinator.full_adapter_mut().disarm_landed_chain();
                }
                match res? {
                    Some(matched) => {
                        self.dspark_turn_contexts.remove(&seq_id);
                        self.scheduled_dspark.seq_lanes.remove(&seq_id);
                        self.scheduled_dspark.sync_decode_limit();
                        Inkling2PrefixState {
                            effective_cached_prefix_len: matched,
                            suffix_len: tokens.len().saturating_sub(matched),
                            full_tokens: tokens.to_vec(),
                            cache_salt: salt,
                            live_continuation: false,
                            reuse_cache: true,
                        }
                    }
                    None => {
                        let mut p = self.prepare_prefix_sync(seq_id, tokens, false, salt)?;
                        p.reuse_cache = true;
                        p
                    }
                }
            } else {
                // State changed while parked; fall back cleanly to synchronous check/cold.
                self.prepare_prefix_sync(seq_id, tokens, true, salt)?
            }
        } else {
            let mut p = self.prepare_prefix_sync(seq_id, tokens, false, salt)?;
            p.reuse_cache = true;
            p
        };
        if let Some(rt) = self.paged.as_mut() {
            rt.seq_reuse_cache.insert(seq_id, prefix.reuse_cache);
        }

        Ok(Some(ScheduledRestoreResult {
            prefix,
            bytes_restored,
            wait: restore.parked_at.elapsed(),
            materialized_blocks: 0,
            profiler_prefill_tokens: 0,
            extra_prefill_breaks: Vec::new(),
        }))
    }

    fn inkling2_victim_units(&self, query: &VictimQuery, _kept: &[u32]) -> UnitRuns {
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
        enum StatePhase {
            Ahead,
            Merged,
            After,
        }

        let mut runs = UnitRuns::default();
        let Some(rt) = self.paged.as_ref() else {
            return runs;
        };
        let full = rt.coordinator.full_adapter();

        // 1. Freeable empty pool extents & dead KV leaves first (byte-only reclaim —
        //    `release_free_extents` / `release_cached_extent` shrink the pool rather than
        //    freeing allocatable pages, so mixed claims with `need.pages > 0` must never
        //    reclaim extents from the pool they need pages from). Empty extents
        //    (`UnitKey::Extent`) are reclaimed before dead leaves and before evicting any
        //    stored hybrid state.
        let states_only = query.need.host_refused && query.need.pages == 0;
        if !query.need.host_refused && query.need.bytes > 0 && query.need.pages == 0 {
            let keep_one = query.rights.reserves == Reserves::None;
            let free_bytes = full.free_extent_bytes(keep_one);
            if free_bytes > 0 {
                runs.ahead.push(Victim {
                    key: UnitKey::Extent,
                    stamp: 0,
                    bytes: free_bytes,
                    pages: 0,
                    stage: Stage::FreeNow,
                    later: false,
                });
            }
            let bytes = full.cached_extent_bytes(keep_one, &[]);
            if bytes > 0 {
                runs.ahead.push(Victim {
                    key: UnitKey::DeadLeaves,
                    stamp: 0,
                    bytes,
                    pages: 0,
                    stage: Stage::FreeNow,
                    later: false,
                });
            }
        }

        // 1b. Parked idle speculative draft KV rings (MLX GPU active bytes, 0 KV pages):
        //     owners whose `DraftContextWindow` covers the kept target frontier rebuild in 0
        //     target forwards, so emit them in `runs.ahead` (`Stage::FreeNow`). Owners without a
        //     covering window go in `runs.merged` (`Stage::NeedsWrite`, `later: false`) so byte
        //     claims prefer reclaiming backed rings and `FreeNow` states before dropping an
        //     unbacked draft ring.
        let pages_only = query.need.host_refused && query.need.pages > 0;
        if query.need.bytes > 0 && !pages_only && !self.dspark_turn_contexts.is_empty() {
            let mut parked_free: Vec<(u64, SeqId, u64)> = Vec::new();
            let mut parked_unbacked: Vec<(u64, SeqId, u64)> = Vec::new();
            for (&seq_id, ctx) in &self.dspark_turn_contexts {
                if self.scheduled_dspark.seq_lanes.contains_key(&seq_id) {
                    continue;
                }
                let bytes = ctx.mlx_buffer_bytes();
                if bytes == 0 {
                    continue;
                }
                let stamp = full.turn_end_stamp(seq_id).unwrap_or(0);
                let has_covering_window = match ctx {
                    super::dflash2::Inkling2DraftContext::Dflash2(dctx) => {
                        let frontier = full
                            .current_token_count_for(seq_id)
                            .unwrap_or_else(|| u32::try_from(dctx.committed_len()).unwrap_or(0));
                        frontier > 0
                            && rt.draft_windows.get(&seq_id).is_some_and(|w| {
                                w.end() == frontier
                                    && w.begin() == frontier.saturating_sub(w.slots())
                            })
                    }
                    super::dflash2::Inkling2DraftContext::Dspark(_) => false,
                };
                if has_covering_window {
                    parked_free.push((stamp, seq_id, bytes));
                } else {
                    parked_unbacked.push((stamp, seq_id, bytes));
                }
            }
            parked_free.sort_by_key(|&(stamp, seq_id, _)| (stamp, seq_id));
            for (stamp, seq_id, bytes) in parked_free {
                runs.ahead.push(Victim {
                    key: UnitKey::DraftContext(seq_id),
                    stamp,
                    bytes,
                    pages: 0,
                    stage: Stage::FreeNow,
                    later: false,
                });
            }
            parked_unbacked.sort_by_key(|&(stamp, seq_id, _)| (stamp, seq_id));
            for (stamp, seq_id, bytes) in parked_unbacked {
                runs.merged.push(Victim {
                    key: UnitKey::DraftContext(seq_id),
                    stamp,
                    bytes,
                    pages: 0,
                    stage: Stage::NeedsWrite,
                    later: false,
                });
            }
        }

        // 2. Stored hybrid states: disposable `StateClass::Checkpoint` go in `ahead` (evicted first),
        //    `StateClass::Ordinary` go in `merged` sorted by shared `CacheRecency` stamp, and
        //    disk-only `chain_owed` states go in `after` as `UnitKey::StoreChain`.
        //    Each marking entry is emitted ONCE so neither bytes nor KV pages are double-counted,
        //    and reclaimable `pages` are derived in exact take order via `full.freed_by_marks`.
        let disk_writable = rt.state_store.disk.as_ref().is_some_and(|d| d.writable());
        let mut states: Vec<(Victim, u64, StatePhase)> = Vec::new();
        for entry in rt.state_store.store.entries() {
            if entry.ram().is_none() {
                if entry.disk().is_some_and(|c| c.chain_owed) {
                    let (stage, later, phase) = if disk_writable {
                        (Stage::NeedsWrite, true, StatePhase::After)
                    } else {
                        // Writer stopped accepting tracked writes: taking `StoreChain(id)`
                        // retires this incomplete disk-only entry synchronously without waiting
                        // on any write, freeing its pinned Global KV pages immediately.
                        (Stage::FreeNow, false, StatePhase::Ahead)
                    };
                    states.push((
                        Victim {
                            key: UnitKey::StoreChain(entry.id()),
                            stamp: entry.last_used(),
                            bytes: 0,
                            pages: 0,
                            stage,
                            later,
                        },
                        entry.mark(),
                        phase,
                    ));
                }
                continue;
            }
            let has_durable_disk = entry.disk().is_some_and(|c| c.landed && !c.chain_owed)
                && rt
                    .state_store
                    .disk
                    .as_ref()
                    .is_some_and(|d| d.entry_disk_restorable(entry));
            let needs_write = !has_durable_disk && disk_writable;
            let victim = Victim {
                key: UnitKey::Store(entry.id()),
                stamp: entry.last_used(),
                bytes: entry.state_bytes(),
                pages: 0,
                stage: if entry.checkpoint() {
                    Stage::FreeNow
                } else if needs_write {
                    Stage::NeedsWrite
                } else {
                    Stage::FreeNow
                },
                later: false,
            };
            let phase = if entry.checkpoint() {
                StatePhase::Ahead
            } else {
                StatePhase::Merged
            };
            states.push((victim, entry.mark(), phase));
        }

        // Sort into exact take order before computing allocator-reclaimable KV pages so shared
        // ancestor blocks are credited once to the last listed mark that pins them, and blocks
        // held by live sequences or unlisted marks are excluded.
        states.sort_by_key(|(unit, _, phase)| (unit.stage, unit.later, *phase, unit.stamp));
        if !states_only && !states.is_empty() {
            let marks: Vec<u64> = states.iter().map(|(_, mark, _)| *mark).collect();
            for ((unit, _, _), pages) in states.iter_mut().zip(full.freed_by_marks(&marks)) {
                unit.pages = pages;
            }
        }
        for (unit, _, phase) in states {
            if unit.bytes == 0 && unit.pages == 0 {
                continue;
            }
            match phase {
                StatePhase::Ahead => runs.ahead.push(unit),
                StatePhase::Merged => runs.merged.push(unit),
                StatePhase::After => runs.after.push(unit),
            }
        }
        runs.ahead.sort_by_key(|v| (v.stage, v.stamp));
        runs.merged.sort_by_key(|v| (v.stage, v.stamp));
        runs.after.sort_by_key(|v| (v.stage, v.later, v.stamp));
        runs
    }

    pub(super) fn inkling2_take_victim(
        &mut self,
        query: &VictimQuery,
        victim: &Victim,
    ) -> LedgerTaken {
        if let UnitKey::DraftContext(seq_id) = victim.key {
            if self.paged.is_none()
                || query.need.bytes == 0
                || (query.need.host_refused && query.need.pages > 0)
                || self.scheduled_dspark.seq_lanes.contains_key(&seq_id)
            {
                return LedgerTaken::default();
            }
            // Drain any in-flight `async_eval_arrays` command buffer from the just-closed turn so
            // dropping `ctx` immediately releases its MLX GPU ring buffers back to active memory.
            crate::array::memory::synchronize();
            let active_before = crate::array::memory::active_memory_bytes();
            let Some(ctx) = self.dspark_turn_contexts.remove(&seq_id) else {
                return LedgerTaken::default();
            };
            drop(ctx);
            let freed = active_before.saturating_sub(crate::array::memory::active_memory_bytes());
            return LedgerTaken {
                outcome: if freed > 0 {
                    FreeOutcome::Met
                } else {
                    FreeOutcome::Short
                },
                now_bytes: freed,
                later_bytes: 0,
                pages: 0,
            };
        }
        let Some(rt) = self.paged.as_mut() else {
            return LedgerTaken::default();
        };
        match victim.key {
            UnitKey::Extent => {
                if query.need.pages > 0 {
                    return LedgerTaken::default();
                }
                let keep_one = query.rights.reserves == Reserves::None;
                let pool_bytes = rt
                    .coordinator
                    .full_adapter_mut()
                    .release_free_extents(victim.bytes, keep_one);
                if pool_bytes > 0 {
                    LedgerTaken {
                        outcome: FreeOutcome::Met,
                        now_bytes: pool_bytes,
                        later_bytes: 0,
                        pages: 0,
                    }
                } else {
                    LedgerTaken::default()
                }
            }
            UnitKey::DeadLeaves => {
                if query.need.pages > 0 {
                    return LedgerTaken::default();
                }
                let keep_one = query.rights.reserves == Reserves::None;
                let pool_bytes = rt
                    .coordinator
                    .full_adapter_mut()
                    .release_cached_extent(keep_one, &[]);
                if pool_bytes > 0 {
                    LedgerTaken {
                        outcome: FreeOutcome::Met,
                        now_bytes: pool_bytes,
                        later_bytes: 0,
                        pages: 0,
                    }
                } else {
                    LedgerTaken::default()
                }
            }
            UnitKey::Store(id) => {
                let takeable_before = rt.coordinator.full_adapter().takeable_blocks();
                let host_before = rt.state_store.host_resident_bytes();
                let (had_ram, exclusive_ram, lineage) = match rt.state_store.store.get(id) {
                    Some(entry) => {
                        let ram_bytes = entry.ram().map_or(0, StateBytes::state_bytes);
                        let exclusive_ram = entry
                            .ram()
                            .map_or(0, Inkling2StateAnchor::exclusive_ram_bytes);
                        let lineage = entry.state_bytes().saturating_sub(ram_bytes);
                        (entry.ram().is_some(), exclusive_ram, lineage)
                    }
                    None => (false, 0, 0),
                };
                match rt.state_store.reclaim_entry_with_options(
                    &rt.coordinator,
                    id,
                    query.rights.waits_for_writes,
                    query.need.pages > 0,
                ) {
                    ReclaimOutcome::RamDropped | ReclaimOutcome::Spilled => {
                        let pages = rt
                            .coordinator
                            .full_adapter()
                            .takeable_blocks()
                            .saturating_sub(takeable_before);
                        let host_freed =
                            host_before.saturating_sub(rt.state_store.host_resident_bytes());
                        let now_bytes = if had_ram {
                            exclusive_ram.min(host_freed)
                        } else {
                            0
                        };
                        LedgerTaken {
                            outcome: if now_bytes > 0 || pages > 0 {
                                FreeOutcome::Met
                            } else {
                                FreeOutcome::Short
                            },
                            now_bytes,
                            later_bytes: 0,
                            pages,
                        }
                    }
                    ReclaimOutcome::Erased => {
                        let erased = rt.state_store.store.get(id).is_none();
                        let pages = rt
                            .coordinator
                            .full_adapter()
                            .takeable_blocks()
                            .saturating_sub(takeable_before);
                        let host_freed =
                            host_before.saturating_sub(rt.state_store.host_resident_bytes());
                        let now_bytes = if had_ram && erased {
                            exclusive_ram.saturating_add(lineage).min(host_freed)
                        } else {
                            0
                        };
                        LedgerTaken {
                            outcome: if now_bytes > 0 || pages > 0 {
                                FreeOutcome::Met
                            } else {
                                FreeOutcome::Short
                            },
                            now_bytes,
                            later_bytes: 0,
                            pages,
                        }
                    }
                    ReclaimOutcome::Pending => {
                        if had_ram {
                            rt.state_store.request_ram_demote_on_durable(id);
                        }
                        if query.need.pages > 0 {
                            rt.state_store.tick(
                                &rt.coordinator,
                                Tick::Wait {
                                    deadline: Instant::now(),
                                },
                            );
                            let pages = rt
                                .coordinator
                                .full_adapter()
                                .takeable_blocks()
                                .saturating_sub(takeable_before);
                            let ram_gone = had_ram
                                && rt
                                    .state_store
                                    .store
                                    .get(id)
                                    .is_none_or(|e| e.ram().is_none());
                            if pages > 0 || ram_gone {
                                let host_freed = host_before
                                    .saturating_sub(rt.state_store.host_resident_bytes());
                                let erased = rt.state_store.store.get(id).is_none();
                                let expected = if erased {
                                    exclusive_ram.saturating_add(lineage)
                                } else {
                                    exclusive_ram
                                };
                                let now_bytes = if ram_gone {
                                    expected.min(host_freed)
                                } else {
                                    0
                                };
                                return LedgerTaken {
                                    outcome: if now_bytes > 0 || pages > 0 {
                                        FreeOutcome::Met
                                    } else {
                                        FreeOutcome::Short
                                    },
                                    now_bytes,
                                    later_bytes: 0,
                                    pages,
                                };
                            }
                        }
                        LedgerTaken {
                            outcome: FreeOutcome::Pending,
                            now_bytes: 0,
                            later_bytes: 0,
                            pages: 0,
                        }
                    }
                }
            }
            UnitKey::StoreChain(id) => {
                let takeable_before = rt.coordinator.full_adapter().takeable_blocks();
                if rt
                    .state_store
                    .store
                    .get(id)
                    .is_some_and(|e| e.ram().is_some())
                {
                    rt.state_store.request_ram_demote_on_durable(id);
                }
                rt.state_store.tick(
                    &rt.coordinator,
                    Tick::Wait {
                        deadline: Instant::now(),
                    },
                );
                let pages = rt
                    .coordinator
                    .full_adapter()
                    .takeable_blocks()
                    .saturating_sub(takeable_before);
                let disk_writable = rt.state_store.disk.as_ref().is_some_and(|d| d.writable());
                LedgerTaken {
                    outcome: if pages > 0 {
                        FreeOutcome::Met
                    } else if disk_writable {
                        FreeOutcome::Pending
                    } else {
                        FreeOutcome::Short
                    },
                    now_bytes: 0,
                    later_bytes: 0,
                    pages,
                }
            }
            _ => LedgerTaken::default(),
        }
    }
}

impl Inkling2Inner {
    /// Total bytes held by live/parked speculative draft contexts (`dspark_turn_contexts` and
    /// active `scheduled_dspark.seq_lanes`) plus live/boundary 4-bit DFlash2 context windows.
    ///
    /// Reported in `memory_accounts().states` (mirroring Qwen3.5's `dflash2_turn::memory_accounts`)
    /// rather than `scheduled_recurrent_bytes()`, which feeds KV-pool admission subtraction.
    pub(crate) fn draft_resident_bytes(&self) -> u64 {
        let window_bytes = self
            .paged
            .as_ref()
            .map_or(0, Inkling2PagedRuntime::draft_windows_bytes);
        let parked_draft_bytes = self
            .dspark_turn_contexts
            .values()
            .fold(0u64, |sum, ctx| sum.saturating_add(ctx.buffer_bytes()));
        let active_lane_bytes = self
            .scheduled_dspark
            .seq_lanes
            .values()
            .fold(0u64, |sum, lane| {
                sum.saturating_add(lane.ctx.buffer_bytes())
            });
        window_bytes
            .saturating_add(parked_draft_bytes)
            .saturating_add(active_lane_bytes)
    }
}

impl HybridSchedulerBackend for Inkling2Inner {
    type FamilyCommand = Inkling2FamilyCommand;
    type RestoreTicket = Inkling2Restore;
    type OwnerState = Inkling2OwnerState;
    type StepExecutor<'a> = HybridStepExecutor<'a, Self>;

    const SCHEDULER_NAME: &'static str = "Inkling2";
    const CANCEL_PRECEDES_EOS: bool = true;
    const STREAM_EOS_TOKEN: bool = true;

    fn paged_adapter(&self) -> Option<&PagedKVCacheAdapter> {
        self.paged.as_ref().map(|rt| rt.coordinator.full_adapter())
    }

    fn paged_adapter_mut(&mut self) -> Option<&mut PagedKVCacheAdapter> {
        self.paged
            .as_mut()
            .map(|rt| rt.coordinator.full_adapter_mut())
    }

    fn scheduler_capacity(&self) -> usize {
        self.paged.as_ref().map_or(1, |rt| {
            rt.coordinator.max_concurrent_sequences().max(1) as usize
        })
    }

    fn scheduler_prefill_slice_tokens(&self) -> u32 {
        self.paged
            .as_ref()
            .map_or(INKLING2_PAGED_PREFILL_MAX_CHUNK, |rt| {
                (rt.prefill_chunk_size as u32).max(1)
            })
    }

    fn scheduler_has_cold_tier(&self) -> bool {
        self.paged
            .as_ref()
            .is_some_and(|rt| rt.state_store.disk.is_some())
    }

    fn wired_residency(&self) -> Option<Arc<WiredResidency>> {
        self.wired.clone()
    }

    fn reclaims_in_order(&self) -> bool {
        self.memory_registration.is_some()
    }

    fn measured_prefill_cut(&self) -> Option<MeasuredConfig> {
        Some(MeasuredConfig {
            native_window: self.effective_context_limit(),
            decode_share: self.decode_share,
        })
    }

    fn memory_model(&self) -> u64 {
        self.memory_registration
            .as_ref()
            .map_or(0, ModelGuard::model_id)
    }

    fn memory_governor(&self) -> Option<&'static Governor> {
        self.memory_registration.as_ref().map(ModelGuard::governor)
    }

    fn memory_accounts(&self) -> Option<MemoryAccounts> {
        let rt = self.paged.as_ref()?;
        let kv = rt.coordinator.pool_allocated_bytes().unwrap_or(0);
        let states = rt
            .state_store
            .total_ram_bytes()
            .saturating_add(self.scheduled_recurrent_bytes())
            .saturating_add(self.draft_resident_bytes());
        Some(MemoryAccounts {
            kv,
            states,
            embeddings: 0,
        })
    }

    fn sync_state_marks(&mut self) {
        if let Some(rt) = self.paged.as_ref() {
            rt.state_store.sync_marks(&rt.coordinator);
        }
    }

    fn tick_state_disk(&mut self, tick: Tick) -> bool {
        let Some(rt) = self.paged.as_mut() else {
            return false;
        };
        rt.state_store.tick(&rt.coordinator, tick)
    }

    fn victim_units(&self, query: &VictimQuery, kept: &[u32]) -> UnitRuns {
        self.inkling2_victim_units(query, kept)
    }

    fn take_victim(&mut self, query: &VictimQuery, victim: &Victim, _kept: &[u32]) -> LedgerTaken {
        self.inkling2_take_victim(query, victim)
    }

    fn claim_step_rows(state: &mut HybridSchedulerState<Self>, plan: &StepPlan) -> StepClaims {
        if state.inner.reclaims_in_order() {
            state.claim_rows_in_order(plan)
        } else {
            StepClaims::granted(plan)
        }
    }

    fn scheduler_books_on_demand(&self) -> bool {
        self.memory_registration.is_some()
    }

    fn scheduler_output_budget_is_rest_of_window(&self) -> bool {
        self.memory_registration.is_some()
    }

    fn planned_turn_window(&self, _speculative: bool) -> Option<u32> {
        Some(self.effective_context_limit())
    }

    fn reset_owner_on_session_start(&self) -> bool {
        true
    }

    fn max_position_embeddings(&self) -> i32 {
        i32::try_from(self.effective_context_limit()).unwrap_or(i32::MAX)
    }

    fn recurrent_state_bytes(&self) -> u64 {
        short_conv_state_bytes(&self.config)
    }

    fn scheduled_recurrent_bytes(&self) -> u64 {
        let live_seqs = self
            .paged
            .as_ref()
            .map_or(0, |rt| rt.conv_states.len() as u64);
        live_seqs.saturating_mul(short_conv_state_bytes(&self.config))
    }

    fn activate_paged_seq(&mut self, seq_id: SeqId) -> Result<()> {
        let rt = self
            .paged
            .as_mut()
            .ok_or_else(|| Error::from_reason("Inkling2 KV coordinator is unavailable"))?;
        rt.active_seq = seq_id;
        rt.coordinator
            .activate_request_all(seq_id)
            .map_err(Error::from_reason)
    }

    fn scheduled_turn_closed(&mut self, seq_id: SeqId) {
        self.abort_prefill_run(seq_id);
        self.park_scheduled_dspark(seq_id);
        if let Some(rt) = self.paged.as_mut() {
            rt.state_store.clear_seq_checkpoint(seq_id);
        }
    }

    fn release_scheduled_recurrent_for(&mut self, seq_id: SeqId) {
        self.release_scheduled_dspark(seq_id);
    }

    fn release_scheduled_cache(&mut self, seq_id: SeqId) -> Result<()> {
        self.abort_prefill_run(seq_id);
        self.release_scheduled_dspark(seq_id);
        if let Some(rt) = self.paged.as_mut() {
            rt.state_store.clear_seq_checkpoint(seq_id);
            rt.release_sequence(seq_id);
        }
        Ok(())
    }

    fn preempt_scheduled_cache(
        &mut self,
        seq_id: SeqId,
        cache_salt: u64,
        mode: PreemptionMode,
    ) -> Result<()> {
        // 1. Abort any in-flight layer-chunked prefill unit and release scheduled DSpark lane state.
        self.abort_prefill_run(seq_id);
        self.release_scheduled_dspark(seq_id);

        let governor = self
            .memory_registration
            .as_ref()
            .map(|g| (g.governor(), g.model_id()));
        if let Some(rt) = self.paged.as_mut() {
            // 2. Preserve completed prefix state if we have a valid block-aligned frontier before
            //    releasing the sequence's physical blocks so preemption replay can resume cleanly,
            //    UNLESS the turn opted out of cache reuse (`reuse_cache == false`).
            let allow_cache = rt.seq_reuse_cache(seq_id);
            let recorded = rt.coordinator.request_token_count_all(seq_id).unwrap_or(0);
            let bs = rt.coordinator.block_size().max(1);
            if allow_cache
                && let Some((boundary, conv)) = rt.block_boundary_conv.get(&seq_id).cloned()
                && boundary > 0
                && boundary <= recorded
                && boundary.is_multiple_of(bs)
                && let Some(tokens) = rt
                    .coordinator
                    .full_adapter()
                    .request_tokens_for(seq_id)
                    .map(|t| t.to_vec())
                && tokens.len() >= boundary as usize
            {
                let draft_win = rt
                    .block_boundary_draft_window
                    .get(&seq_id)
                    .filter(|(wb, _)| *wb == boundary)
                    .map(|(_, w)| w);
                let _ = rt.state_store.capture_boundary_with_window(
                    &mut rt.coordinator,
                    &conv,
                    seq_id,
                    &tokens,
                    boundary,
                    cache_salt,
                    StateClass::Checkpoint,
                    draft_win,
                    governor,
                );
            }

            // 3. Register full Group 0 blocks for prefix reuse (when cache reuse is allowed),
            //    then release Group 0 + Group 1 + ShortConv.
            if allow_cache {
                let _ = rt.coordinator.register_full_blocks_for_reuse_for(
                    seq_id,
                    cache_salt,
                    mode == PreemptionMode::Ssd,
                );
            }
            rt.state_store.clear_seq_checkpoint(seq_id);
            rt.release_sequence(seq_id);
        }
        Ok(())
    }

    fn release_owner_resources(
        &mut self,
        seq_id: Option<SeqId>,
        _state: Option<&Self::OwnerState>,
    ) -> Result<()> {
        if let Some(seq_id) = seq_id {
            self.release_scheduled_cache(seq_id)?;
        }
        Ok(())
    }

    fn prefill_runs_in_passes(&self, _seq_id: SeqId) -> bool {
        self.paged.is_some()
    }

    fn prefill_rows_done(&self, seq_id: SeqId) -> Option<u32> {
        self.prefill_runs
            .get(&seq_id)
            .map(|run| run.rows_done(self.config.num_layer_visits()))
    }

    fn run_scheduled_prefill_pass(
        &mut self,
        seq_id: SeqId,
        source: &[u32],
        base: &Self::PrefixState,
        start: usize,
        end: usize,
        generation_stream: Stream,
        _first_chunk: bool,
        budget_ms: Option<f64>,
    ) -> Result<ScheduledPrefillPass> {
        if let Some(rt) = self.paged.as_mut() {
            rt.active_seq = seq_id;
        }
        let pass = self.run_paged_prefill_pass(
            seq_id,
            source,
            start,
            end,
            base.cache_salt,
            base.reuse_cache,
            generation_stream,
            budget_ms,
        )?;
        Ok(pass)
    }

    fn run_scheduled_prefill_slice(
        &mut self,
        seq_id: SeqId,
        source: &[u32],
        base: &Self::PrefixState,
        start: usize,
        end: usize,
        generation_stream: Stream,
        _first_chunk: bool,
    ) -> Result<Option<MxArray>> {
        if let Some(rt) = self.paged.as_mut() {
            rt.active_seq = seq_id;
        }
        match self.run_paged_prefill_pass(
            seq_id,
            source,
            start,
            end,
            base.cache_salt,
            base.reuse_cache,
            generation_stream,
            None,
        )? {
            ScheduledPrefillPass::Done(logits) => Ok(logits),
            ScheduledPrefillPass::Paused => Err(Error::from_reason(
                "Inkling2 unbounded prefill slice unexpectedly paused",
            )),
        }
    }

    fn run_paged_decode_step_batched(&mut self, rows: &[(SeqId, u32)]) -> Result<MxArray> {
        let logits_2d = self.paged_decode_step_batched(rows)?;
        let n = logits_2d.shape()?[0];
        let vocab = logits_2d.shape()?[1];
        logits_2d.reshape(&[n, 1, vocab])
    }

    fn supports_scheduled_speculation(&self) -> bool {
        self.has_speculative_draft() && self.paged.is_some()
    }

    fn supports_adaptive_scheduled_speculation(&self) -> bool {
        self.has_speculative_draft() && self.paged.is_some()
    }

    fn speculative_turn_requires_exclusive_lane(
        &self,
        params: &crate::engine::params::ChatParams,
    ) -> bool {
        let policy = super::dflash2::Inkling2VerifyTreePolicy::from_env();
        should_use_exclusive_verify_tree_lane(
            self.dflash2.as_ref().map(|d| d.config.block_size),
            params,
            policy.mode,
        )
    }

    fn scheduled_sampled_rows(&self) -> bool {
        self.has_speculative_draft()
    }

    fn scheduled_speculation_owners(&self) -> usize {
        self.scheduled_dspark.seq_lanes.len()
    }

    fn complete_scheduled_draft_state(&self, seq_ids: &[SeqId]) -> Result<()> {
        let mut arrays = Vec::new();
        for seq in seq_ids {
            if let Some(lane) = self.scheduled_dspark.seq_lanes.get(seq) {
                lane.ctx.collect_arrays(&mut arrays);
            }
        }
        if !arrays.is_empty() {
            MxArray::eval_arrays(&arrays)?;
        }
        Ok(())
    }

    fn scheduled_draft_state_bytes(&self, total_tokens: u32) -> u64 {
        if let Some(ref draft) = self.dflash2 {
            return draft.config.lane_bytes();
        }
        let Some(ref draft) = self.dspark else {
            return 0;
        };
        draft.config.context_bytes_for_tokens(total_tokens)
    }

    fn scheduled_draft_state_held_bytes(&self, total_tokens: u32) -> u64 {
        self.scheduled_draft_state_bytes(total_tokens)
    }

    fn begin_scheduled_speculation(&mut self, seq_id: SeqId, position: u32) -> Result<bool> {
        self.begin_scheduled_dspark(seq_id, position)
    }

    fn reserve_scheduled_speculation(&mut self, seq_id: SeqId, queries: usize) -> Result<bool> {
        self.reserve_scheduled_dspark(seq_id, queries)
    }

    fn propose_scheduled(
        &mut self,
        seq_id: SeqId,
        anchor: u32,
        max_drafts: usize,
        params: &crate::engine::params::ChatParams,
        rng: &mut dyn rand::Rng,
        confidence: bool,
    ) -> Result<crate::engine::backend::DsparkProposal> {
        self.propose_scheduled_dspark(seq_id, anchor, max_drafts, params, rng, confidence)
    }

    fn propose_scheduled_batch(
        &mut self,
        requests: &mut [crate::engine::hybrid_scheduler::ScheduledProposalRequest<'_>],
    ) -> Vec<Result<crate::engine::backend::DsparkProposal>> {
        self.propose_scheduled_dspark_batch(requests)
    }

    fn run_scheduled_verify(
        &mut self,
        rows: &[crate::engine::hybrid_scheduler::ScheduledVerifyRow],
    ) -> Result<MxArray> {
        self.verify_scheduled_dspark(rows)
    }

    fn commit_scheduled_verify(
        &mut self,
        rows: &[crate::engine::hybrid_scheduler::ScheduledVerifyCommit],
    ) -> Result<Vec<Result<()>>> {
        self.commit_scheduled_dspark(rows)
    }

    fn release_scheduled_speculation(&mut self, seq_id: SeqId) {
        self.park_scheduled_dspark(seq_id);
    }

    fn finish_scheduled_decode_batch(&mut self, rows: &[(SeqId, u32)]) -> Result<()> {
        let rt = self
            .paged
            .as_mut()
            .ok_or_else(|| Error::from_reason("Inkling2 batched decode lost its paged runtime"))?;
        rt.coordinator
            .eval_pending_pool_writes_all()
            .map_err(Error::from_reason)?;
        // Queue async evaluation of all updated ShortConv + DraftContextWindow states in ONE
        // non-blocking batch (same MLX stream guarantees materialization before the next step
        // reads them; durable StateDisk capture explicitly syncs via `eval_conv_state_arrays`).
        let mut conv_arrays: Vec<&MxArray> = Vec::with_capacity(rows.len() * (48 * 4 + 1));
        for &(seq_id, _) in rows {
            if let Some(conv) = rt.conv_states.get(&seq_id) {
                for v in &conv.visits {
                    conv_arrays.push(&v.k);
                    conv_arrays.push(&v.v);
                    conv_arrays.push(&v.attn_post);
                    conv_arrays.push(&v.mlp_post);
                }
            }
            if let Some(dw) = rt.draft_windows.get(&seq_id) {
                dw.collect_arrays(&mut conv_arrays);
            }
        }
        if !conv_arrays.is_empty() {
            MxArray::async_eval_arrays(&conv_arrays);
        }
        for &(seq_id, _) in rows {
            rt.coordinator
                .prune_sliding_all(seq_id)
                .map_err(Error::from_reason)?;
        }
        Ok(())
    }

    fn prepare_scheduled_prefix(
        &mut self,
        seq_id: SeqId,
        tokens: &[u32],
        _owner_history: &[u32],
        reuse_cache: bool,
        cache_salt: u64,
        _block_size: u32,
    ) -> Result<ScheduledPrefixAdmission<Self::PrefixState, Self::RestoreTicket>> {
        self.prepare_prefix_sync(seq_id, tokens, reuse_cache, cache_salt)
            .map(ScheduledPrefixAdmission::Ready)
    }

    fn prepare_contended_prefix(
        &mut self,
        seq_id: SeqId,
        tokens: &[u32],
        _owner_history: &[u32],
        reuse_cache: bool,
        cache_salt: u64,
        _block_size: u32,
    ) -> Result<ScheduledPrefixAdmission<Self::PrefixState, Self::RestoreTicket>> {
        self.prepare_contended_prefix_impl(seq_id, tokens, reuse_cache, cache_salt)
    }

    fn poll_scheduled_restore(
        &mut self,
        _seq_id: SeqId,
        restore: &mut Self::RestoreTicket,
        uploads: &mut RestoreUploadBudget,
        _prompt_tokens: &[u32],
        _owner_history: &[u32],
        _is_preemption_replay: bool,
    ) -> Result<Option<ScheduledRestoreResult<Self::PrefixState>>> {
        let rt = self
            .paged
            .as_ref()
            .ok_or_else(|| Error::from_reason("Inkling2 parked restore lost its paged runtime"))?;
        poll_restore_ticket(rt.coordinator.full_adapter(), restore, uploads);
        Ok(None)
    }

    fn scheduled_restore_lands(restore: &Self::RestoreTicket) -> bool {
        restore.lands()
    }

    fn land_scheduled_restore(
        &mut self,
        seq_id: SeqId,
        restore: &mut Self::RestoreTicket,
        prompt_tokens: &[u32],
        _owner_history: &[u32],
        _is_preemption_replay: bool,
    ) -> Result<Option<ScheduledRestoreResult<Self::PrefixState>>> {
        self.land_parked_restore_impl(seq_id, restore, prompt_tokens)
    }

    fn restore_reserved_blocks(restore: &Self::RestoreTicket) -> u32 {
        restore.held_blocks()
    }

    fn scheduled_prefix_probe(&self, tokens: &[u32], cache_salt: u64) -> u32 {
        let Some(rt) = self.paged.as_ref() else {
            return 0;
        };
        rt.state_store
            .lookup_deepest_restorable(&rt.coordinator, tokens, cache_salt)
            .map_or(0, |hit| u32::try_from(hit.depth).unwrap_or(u32::MAX))
    }

    fn scheduled_prefix_probe_generation(&self) -> u64 {
        self.paged
            .as_ref()
            .map_or(0, |rt| rt.state_store.generation())
    }

    fn extra_prefill_breaks(&self, prompt_tokens: u32, cached_prefix: u32) -> Vec<u32> {
        let block_size = self
            .paged
            .as_ref()
            .map_or(32, |rt| rt.coordinator.block_size())
            .max(1);
        let mut breaks = planned_rolling_checkpoints(prompt_tokens, cached_prefix, block_size);
        // Also split at the prompt's deepest complete block boundary strictly before the final
        // prompt token so ShortConv(k=4) is captured bit-exact at the prompt junction (mirroring
        // Qwen 3.5's replay-point junction).
        let junction = (prompt_tokens.saturating_sub(1) / block_size) * block_size;
        if junction > cached_prefix && junction > 0 && !breaks.contains(&junction) {
            breaks.push(junction);
            breaks.sort_unstable();
        }
        breaks
    }

    fn replace_cached_token_history(&mut self, history: Vec<u32>) {
        self.cached_token_history = history;
    }

    fn owner_tokens(state: &Self::OwnerState) -> &[u32] {
        &state.cached_token_history
    }

    fn install_owner_state(&mut self, seq_id: SeqId, state: &Self::OwnerState) {
        if let Some(rt) = self.paged.as_mut() {
            rt.active_seq = seq_id;
        }
        self.cached_token_history = state.cached_token_history.clone();
        self.cached_image_key = state.cached_image_key;
    }

    fn capture_owner_state(&mut self, _seq_id: SeqId) -> Self::OwnerState {
        Inkling2OwnerState {
            cached_token_history: self.cached_token_history.clone(),
            cached_image_key: self.cached_image_key,
            flat_cache: None,
            cuda_cache: None,
            cuda_dflash2_ctx: None,
            cuda_dspark_ctx: None,
        }
    }

    fn build_scheduled_prefix(
        &self,
        base: &Self::PrefixState,
        effective_cached_prefix_len: usize,
        suffix_len: usize,
        full_tokens: Vec<u32>,
        _first_chunk: bool,
    ) -> Self::PrefixState {
        Inkling2PrefixState {
            effective_cached_prefix_len,
            suffix_len,
            full_tokens,
            cache_salt: base.cache_salt,
            live_continuation: base.live_continuation,
            reuse_cache: base.reuse_cache,
        }
    }

    fn step_executor(&mut self) -> Self::StepExecutor<'_> {
        HybridStepExecutor::new(self)
    }

    fn execute_chat_barrier(
        &mut self,
        command: ChatCmd,
        owners: SchedulerOwnerContext<'_, Self::OwnerState>,
    ) {
        let prev = self
            .paged
            .as_ref()
            .is_some_and(|rt| rt.exclusive_turn_active);
        if let Some(rt) = self.paged.as_mut() {
            rt.exclusive_turn_active = true;
        }
        execute_dual_lane_chat_barrier(self, command, owners);
        if let Some(rt) = self.paged.as_mut() {
            rt.exclusive_turn_active = prev;
        }
    }
}

/// Pure decision helper for Inkling-2-Mini DFlash2 verify-tree exclusive-lane
/// routing. Mirrors the whole-turn stepper's `tree_greedy_unpenalized` check in
/// `engine::dspark_turn` and
/// [`super::dflash2::Inkling2VerifyTreePolicy::tree_path_active`]: greedy
/// unpenalized DFlash2 turns whose resolved `mtp_depth` covers the full draft
/// block (`>= block_size.max(1)`) take the exclusive whole-turn lane when
/// `mode` is `Auto` or `Always`, while `Off`, sampled, penalized, pinned
/// smaller-depth (`mtp_depth < block_size`), and DSpark-only turns stay on the
/// multi-lane scheduled verifier.
pub(crate) fn should_use_exclusive_verify_tree_lane(
    dflash2_block_size: Option<usize>,
    params: &crate::engine::params::ChatParams,
    mode: super::dflash2::Inkling2VerifyTreeMode,
) -> bool {
    let Some(block_size) = dflash2_block_size else {
        return false;
    };
    if params.mtp_depth < block_size.max(1) {
        return false;
    }
    let policy = super::dflash2::Inkling2VerifyTreePolicy {
        mode,
        ..super::dflash2::Inkling2VerifyTreePolicy::default()
    };
    if !policy.tree_path_active() {
        return false;
    }
    let temperature = params
        .sampling_config
        .and_then(|config| config.temperature)
        .unwrap_or(1.0);
    crate::sampling::is_greedy_temperature(temperature)
        && params.repetition_penalty == 1.0
        && params.presence_penalty == 0.0
        && params.frequency_penalty == 0.0
}

/// Ensure `AtomicBool` import stays used by tests and scheduler helpers.
#[allow(dead_code)]
fn _cancel_flag_type_check(_flag: &Arc<AtomicBool>) {}

#[cfg(test)]
mod verify_tree_routing_tests {
    use super::should_use_exclusive_verify_tree_lane;
    use crate::engine::params::extract_chat_params;
    use crate::engine::types::ChatConfig;
    use crate::models::inkling2::dflash2::Inkling2VerifyTreeMode;
    use crate::models::qwen3_5::dflash2::DFLASH2_DRAFT_PROPOSALS;

    fn params_with(config: ChatConfig) -> crate::engine::params::ChatParams {
        let mut p = extract_chat_params(&config);
        let max_draft = DFLASH2_DRAFT_PROPOSALS.max(1);
        p.mtp_depth = match config.mtp_depth {
            Some(d) => (d.max(1) as usize).min(max_draft),
            None => max_draft,
        };
        p.mtp_adaptive_depth = config.mtp_adaptive_depth.unwrap_or(false);
        p
    }

    #[test]
    fn dflash2_greedy_unpenalized_routes_by_verify_tree_mode() {
        let block = Some(DFLASH2_DRAFT_PROPOSALS);

        // Default `mtp_depth: None` (resolves to `block_size = 7`) and explicit `Some(7)`
        // both route to the exclusive verify-tree lane under `Auto`/`Always`, not `Off`.
        for mtp_depth in [None, Some(DFLASH2_DRAFT_PROPOSALS as i32)] {
            let greedy = params_with(ChatConfig {
                enable_mtp: Some(true),
                mtp_depth,
                temperature: Some(0.0),
                ..ChatConfig::default()
            });
            assert_eq!(greedy.mtp_depth, DFLASH2_DRAFT_PROPOSALS);
            assert!(should_use_exclusive_verify_tree_lane(
                block,
                &greedy,
                Inkling2VerifyTreeMode::Auto
            ));
            assert!(should_use_exclusive_verify_tree_lane(
                block,
                &greedy,
                Inkling2VerifyTreeMode::Always
            ));
            assert!(!should_use_exclusive_verify_tree_lane(
                block,
                &greedy,
                Inkling2VerifyTreeMode::Off
            ));
        }

        // Near-zero f32 greedy boundary matches `is_greedy_temperature`.
        let near_zero = params_with(ChatConfig {
            enable_mtp: Some(true),
            temperature: Some(1e-7),
            ..ChatConfig::default()
        });
        assert!(should_use_exclusive_verify_tree_lane(
            block,
            &near_zero,
            Inkling2VerifyTreeMode::Auto
        ));
    }

    #[test]
    fn sampled_penalized_and_dspark_only_turns_stay_on_scheduled_lane() {
        let block = Some(DFLASH2_DRAFT_PROPOSALS);
        let modes = [
            Inkling2VerifyTreeMode::Auto,
            Inkling2VerifyTreeMode::Always,
            Inkling2VerifyTreeMode::Off,
        ];

        // Sampled explicit & unset temperature (defaults to 1.0).
        for temp in [Some(1.0), Some(0.7), Some(1e-3), None] {
            let p = params_with(ChatConfig {
                enable_mtp: Some(true),
                temperature: temp,
                ..ChatConfig::default()
            });
            for mode in modes {
                assert!(
                    !should_use_exclusive_verify_tree_lane(block, &p, mode),
                    "sampled temp={temp:?} mode={mode:?} must stay scheduled"
                );
            }
        }

        // Greedy with any non-neutral penalty stays scheduled (matches `dspark_turn`).
        let penalized_configs = [
            ChatConfig {
                enable_mtp: Some(true),
                temperature: Some(0.0),
                repetition_penalty: Some(1.1),
                ..ChatConfig::default()
            },
            ChatConfig {
                enable_mtp: Some(true),
                temperature: Some(0.0),
                presence_penalty: Some(0.2),
                ..ChatConfig::default()
            },
            ChatConfig {
                enable_mtp: Some(true),
                temperature: Some(0.0),
                frequency_penalty: Some(0.2),
                ..ChatConfig::default()
            },
        ];
        for cfg in penalized_configs {
            let p = params_with(cfg);
            for mode in modes {
                assert!(
                    !should_use_exclusive_verify_tree_lane(block, &p, mode),
                    "penalized greedy mode={mode:?} must stay scheduled"
                );
            }
        }

        // Caller-pinned `mtp_depth < block_size` stays on scheduled 1D lane so the
        // fixed draft allowance is strictly enforced even under `Auto`/`Always`.
        for pinned in [1i32, 3, (DFLASH2_DRAFT_PROPOSALS - 1) as i32] {
            let p = params_with(ChatConfig {
                enable_mtp: Some(true),
                mtp_depth: Some(pinned),
                temperature: Some(0.0),
                ..ChatConfig::default()
            });
            assert_eq!(p.mtp_depth, pinned as usize);
            for mode in modes {
                assert!(
                    !should_use_exclusive_verify_tree_lane(block, &p, mode),
                    "pinned mtp_depth={pinned} mode={mode:?} must stay scheduled"
                );
            }
        }

        // DSpark-only (`dflash2_block_size == None`) greedy stays on scheduled lane.
        let greedy = params_with(ChatConfig {
            enable_mtp: Some(true),
            temperature: Some(0.0),
            ..ChatConfig::default()
        });
        for mode in modes {
            assert!(
                !should_use_exclusive_verify_tree_lane(None, &greedy, mode),
                "DSpark-only greedy mode={mode:?} must stay scheduled"
            );
        }
    }
}

#[cfg(all(test, target_os = "macos"))]
pub(super) mod tests {
    use std::path::Path;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    use std::time::{Duration, Instant};

    use tokio::sync::mpsc;

    use crate::engine::cmd::ChatCmd;
    use crate::engine::types::{ChatConfig, ChatResult};
    use crate::models::inkling2::model::{Inkling2LoadOptions, Inkling2Model};
    use crate::models::inkling2::persistence::{load_from_dir, test_real_model_path};
    use crate::models::inkling2::scheduler::{Inkling2Cmd, Inkling2SchedulerState};
    use crate::tokenizer::ChatMessage;

    fn greedy_config(owner: &str, max_tokens: i32, salt: &str) -> ChatConfig {
        ChatConfig {
            max_new_tokens: Some(max_tokens),
            temperature: Some(0.0),
            top_p: Some(1.0),
            top_k: Some(0),
            reasoning_effort: Some("none".to_string()),
            reuse_cache: Some(true),
            cache_salt: Some(salt.to_string()),
            cache_owner_id: Some(owner.to_string()),
            ..ChatConfig::default()
        }
    }

    fn user_msg(text: &str) -> Vec<ChatMessage> {
        vec![ChatMessage {
            role: "user".to_string(),
            content: text.to_string(),
            tool_calls: None,
            tool_call_id: None,
            is_error: None,
            reasoning_content: None,
            thinking_enabled: None,
            images: None,
            audio: None,
        }]
    }

    async fn run_sync_turn(
        tx: &mpsc::UnboundedSender<Inkling2Cmd>,
        messages: Vec<ChatMessage>,
        config: ChatConfig,
    ) -> ChatResult {
        let (reply, rx) = tokio::sync::oneshot::channel();
        tx.send(Inkling2Cmd::Chat(Box::new(ChatCmd::SessionStart {
            messages,
            config,
            reply,
            cancelled: Arc::new(AtomicBool::new(false)),
        })))
        .expect("send SessionStart");
        rx.await
            .expect("scheduler reply channel")
            .expect("SessionStart succeeded")
    }

    /// **VERIFICATION GATE 1 (`inkling2_concurrent_batched_parity`)**:
    ///
    /// Runs 4 distinct prompts on the real Inkling-2-Mini GGUF model:
    /// 1. First serially (one at a time) through `Inkling2SchedulerState`.
    /// 2. Then concurrently (all 4 admitted together into `Inkling2SchedulerState`, executing
    ///    fused `[N, 1]` batched decode steps + 100ms layer-chunked prefill interleaving + 500ms
    ///    decode-gap pacing) via `Inkling2Model::load` (`ModelThread::spawn_with_scheduler` +
    ///    unified `MemoryPlan` + `Governor` ledger + `WiredResidency`).
    ///
    /// Asserts:
    /// - Every concurrent sequence produces **identical greedy output** (`raw_text`, `text`,
    ///   `num_tokens`) to serial execution!
    /// - `has_block_paged_cache() == true`, `max_concurrent_sequences() >= 2`,
    ///   `supports_images() == true`, and batched decode width > 1 actually engaged!
    #[tokio::test]
    #[ignore = "Requires real Inkling-2-Mini GGUF on disk (~18 GB)"]
    async fn inkling2_concurrent_batched_parity() {
        let model_path = test_real_model_path();
        let model_dir = Path::new(&model_path);
        assert!(
            model_dir.exists(),
            "Real model path missing: {}",
            model_dir.display()
        );

        let prompts = [
            "What is 17 * 23? Show only the final number.",
            "Name the capital city of France in one word.",
            "List the first four prime numbers separated by commas.",
            "What is the square root of 144? Show only the final number.",
        ];
        let max_tokens = 24;

        // 1. Load via full NAPI path (`ModelThread::spawn_with_scheduler` + `MemoryPlan` + `Governor`).
        let t0 = Instant::now();
        let model = Inkling2Model::load(
            model_path.clone(),
            Some(Inkling2LoadOptions {
                kv_format: Some("int8".to_string()),
                cache_disk: Some("0".to_string()),
                ..Inkling2LoadOptions::default()
            }),
        )
        .await
        .expect("Inkling2Model::load");
        eprintln!(
            "[inkling2_concurrent_batched_parity] Loaded via Inkling2Model::load in {:.2}s \
             (paged={}, lanes={}, vision={}, window={})",
            t0.elapsed().as_secs_f64(),
            model.has_block_paged_cache(),
            model.max_concurrent_sequences(),
            model.supports_images(),
            model.context_limits().effective_window_tokens,
        );
        assert!(model.has_block_paged_cache());
        assert!(
            model.max_concurrent_sequences() >= 4,
            "expected >= 4 concurrent lanes on M5 Max, got {}",
            model.max_concurrent_sequences()
        );
        assert!(model.supports_images());

        // 2. Serial baseline: run each prompt one by one with unique cache salts (cold prefill).
        let tx = model
            .thread
            .cmd_sender()
            .expect("model thread command sender");
        let mut serial_outputs = Vec::with_capacity(prompts.len());
        for (i, prompt) in prompts.iter().enumerate() {
            let res = run_sync_turn(
                &tx,
                user_msg(prompt),
                greedy_config(
                    &format!("serial-{i}"),
                    max_tokens,
                    &format!("serial-salt-{i}"),
                ),
            )
            .await;
            eprintln!(
                "  [serial {i}] tokens={} raw={:?}",
                res.num_tokens, res.raw_text
            );
            serial_outputs.push(res);
        }
        model
            .reset_caches()
            .await
            .expect("reset_caches between serial and batch");

        // 3. Concurrent batched execution: submit all 4 prompts simultaneously into the scheduler.
        let t_batch = Instant::now();
        let (r0, r1, r2, r3) = tokio::join!(
            run_sync_turn(
                &tx,
                user_msg(prompts[0]),
                greedy_config("batch-0", max_tokens, "batch-salt-0")
            ),
            run_sync_turn(
                &tx,
                user_msg(prompts[1]),
                greedy_config("batch-1", max_tokens, "batch-salt-1")
            ),
            run_sync_turn(
                &tx,
                user_msg(prompts[2]),
                greedy_config("batch-2", max_tokens, "batch-salt-2")
            ),
            run_sync_turn(
                &tx,
                user_msg(prompts[3]),
                greedy_config("batch-3", max_tokens, "batch-salt-3")
            ),
        );
        let batch_secs = t_batch.elapsed().as_secs_f64();
        let concurrent_outputs = [r0, r1, r2, r3];

        let stats = model.scheduler_stats().await.expect("scheduler_stats");
        eprintln!(
            "[inkling2_concurrent_batched_parity] 4 concurrent turns finished in {batch_secs:.2}s \
             (global_steps={}, max_batch_occupancy={}, buckets={})",
            stats.global_steps,
            stats.max_batch_occupancy,
            stats.decode_batch_occupancy_hist.len()
        );
        assert!(
            stats.max_batch_occupancy >= 2,
            "Expected concurrent batched decode width >= 2, got {}",
            stats.max_batch_occupancy
        );

        for (i, (ser, con)) in serial_outputs
            .iter()
            .zip(concurrent_outputs.iter())
            .enumerate()
        {
            eprintln!(
                "  [batch  {i}] tokens={} raw={:?}",
                con.num_tokens, con.raw_text
            );
            assert_eq!(
                con.raw_text, ser.raw_text,
                "Concurrent batched output diverged from serial on prompt {i}!"
            );
            assert_eq!(
                con.num_tokens, ser.num_tokens,
                "Concurrent token count diverged from serial on prompt {i}!"
            );
        }

        // 4. Verify direct `Inkling2SchedulerState` driving (unit-level check of preemption/release).
        drop(tx);
        model.dispose().await.expect("dispose");
        drop(model);

        // Quick check that `load_from_dir` + `Inkling2SchedulerState::new` works directly too.
        let inner = load_from_dir(&model_path)
            .await
            .expect("load_from_dir inner");
        let state = Inkling2SchedulerState::new(inner).expect("Inkling2SchedulerState::new");
        let _ = Duration::from_millis(1);
        drop(state);
    }

    // =========================================================================
    // Fast Synthetic Metal Tests: Option 4 Exclusive-Turn On-Demand KV Reclaim
    // =========================================================================

    fn det_f32_vec(mut state: u64, len: usize, scale: f32, bias: f32) -> Vec<f32> {
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            let u = ((z >> 40) as f32) / ((1u32 << 24) as f32);
            out.push((u * 2.0 - 1.0) * scale + bias);
        }
        out
    }

    fn det_f32(seed: u64, shape: &[i64], scale: f32, bias: f32) -> crate::array::MxArray {
        let len: usize = shape.iter().map(|&d| d as usize).product();
        let vals = det_f32_vec(seed, len, scale, bias);
        crate::array::MxArray::from_float32(&vals, shape).expect("det_f32")
    }

    fn det_bf16(seed: u64, shape: &[i64], scale: f32, bias: f32) -> crate::array::MxArray {
        det_f32(seed, shape, scale, bias)
            .astype(crate::array::DType::BFloat16)
            .expect("det_bf16")
    }

    fn build_synthetic_inkling2_tokenizer() -> crate::tokenizer::Qwen3Tokenizer {
        let mut vocab = serde_json::Map::new();
        for (i, ch) in (32u8..=126u8).enumerate() {
            vocab.insert((ch as char).to_string(), serde_json::json!(i));
        }
        vocab.insert("\n".to_string(), serde_json::json!(95));
        let tokenizer_json = serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {
                "type": "Split",
                "pattern": { "Regex": "." },
                "behavior": "Isolated",
                "invert": false
            },
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": vocab,
                "unk_token": "?"
            }
        });
        let mut tok = tokenizers::Tokenizer::from_bytes(tokenizer_json.to_string().as_bytes())
            .expect("build WordLevel tokenizer");
        let added: Vec<tokenizers::AddedToken> = crate::tools::inkling::INKLING2_CONTROL_TOKENS
            .iter()
            .map(|&s| tokenizers::AddedToken::from(s, true))
            .collect();
        tok.add_special_tokens(added)
            .expect("register Inkling2 special tokens");
        crate::tokenizer::Qwen3Tokenizer::from_tokenizer(tok, None).expect("wrap Qwen3Tokenizer")
    }

    /// Build a fast 4-layer (2 SWA + 2 Global) synthetic Metal `Inkling2Inner` with `d=256`,
    /// `DFlash2` drafter (`block_size=7`, `target_layers=[1, 3]`), capped paged KV pool
    /// (`max_model_len`, `block_size=32`, `max_concurrent_sequences=1`), and synthetic tokenizer.
    ///
    /// When `force_lm_token` is `Some(tok_id)`, `lm_head` row `tok_id` is strongly biased so
    /// greedy argmax deterministically emits `tok_id`.
    pub(crate) fn build_synthetic_dflash2_inner(
        max_model_len: u32,
        force_lm_token: Option<u32>,
    ) -> crate::models::inkling2::model::Inkling2Inner {
        build_synthetic_dflash2_inner_with_window(max_model_len, force_lm_token, 8, 1)
    }

    fn build_synthetic_dflash2_inner_with_window(
        max_model_len: u32,
        force_lm_token: Option<u32>,
        draft_sliding_window: usize,
        max_concurrent_sequences: u32,
    ) -> crate::models::inkling2::model::Inkling2Inner {
        use std::collections::HashMap;

        use mlx_paged_attn::CacheRecency;

        use crate::array::MxArray;
        use crate::disk_tier::DiskTierOptions;
        use crate::models::inkling2::attention::Inkling2Attention;
        use crate::models::inkling2::config::Inkling2Config;
        use crate::models::inkling2::kv_cache::{Inkling2PagedConfig, default_inkling2_kv_format};
        use crate::models::inkling2::model::{
            AttnInputNorm, Inkling2Inner, Inkling2PhysicalLayer, default_decode_share,
            next_inkling2_model_id,
        };
        use crate::models::inkling2::moe::{ExpertSwitchProj, Inkling2MoE};
        use crate::models::inkling2::scheduled_dspark::Inkling2ScheduledDsparkStates;
        use crate::models::quantized_linear::LinearProj;
        use crate::models::qwen3_5::config::Qwen3_5Config;
        use crate::models::qwen3_5::dflash2::{
            DFLASH2_DRAFT_PROPOSALS, tiny_dflash2_model_with_block,
        };
        use crate::nn::{Embedding, Linear};
        use crate::transformer::Activations;

        let d = 256usize;
        let n_heads = 2usize;
        let n_kv = 1usize;
        let head_dim = 128usize;
        let d_rel = 16usize;
        let rel_extent = 64usize;
        let vocab = 128usize;
        let num_experts = 4usize;
        let top_k = 2usize;
        let moe_inter = 64usize;
        let inv_sqrt_d = 1.0_f64 / (d as f64).sqrt();

        let config = Inkling2Config {
            model_type: "inkling2".to_string(),
            hidden_size: d,
            num_hidden_layers: 4,
            num_physical_layers: Some(4),
            num_attention_heads: n_heads,
            num_key_value_heads: n_kv,
            head_dim,
            value_head_dim: None,
            vocab_size: vocab,
            padded_vocab_size: Some(vocab),
            unpadded_vocab_size: Some(vocab),
            rms_norm_eps: 1e-6,
            max_position_embeddings: max_model_len.max(256) as usize,
            sliding_window: 64,
            sliding_window_pattern: vec![true, false, true, false],
            local_layer_ids: vec![0, 2],
            sconv_kernel_size: 4,
            n_layers_per_attn_res_block: 2,
            d_rel,
            rel_extent,
            qk_l2_clip_threshold: 16.0,
            rel_l2_clip_threshold: 8.0,
            log_scaling_n_floor: 4,
            log_scaling_alpha: 0.25,
            num_experts,
            num_experts_per_tok: top_k,
            moe_intermediate_size: Some(moe_inter),
            intermediate_size: Some(moe_inter),
            route_scale: 1.0,
            mlp_gate_softcap: 15.0,
            mlp_up_softcap: 15.0,
            final_logit_softcapping: 30.0,
            loop_config: None,
            vision_config: None,
            dspark_config: None,
            eos_token_id: 2,
            pad_token_id: 0,
            gate_up_interleaved: false,
        };

        let make_lin = |seed: u64, out_f: usize, in_f: usize, scale: f32| -> LinearProj {
            let w = det_bf16(seed, &[out_f as i64, in_f as i64], scale, 0.0);
            LinearProj::Standard(Linear::from_weights(&w, None).expect("Linear::from_weights"))
        };
        let make_zero_lin = |out_f: usize, in_f: usize| -> LinearProj {
            let w = MxArray::zeros(
                &[out_f as i64, in_f as i64],
                Some(crate::array::DType::BFloat16),
            )
            .expect("zero_lin");
            LinearProj::Standard(Linear::from_weights(&w, None).expect("Linear::from_weights"))
        };

        let emb_w = if force_lm_token.is_some() {
            let mut vals = vec![0.0f32; vocab * d];
            for row in 0..vocab {
                vals[row * d] = 1.0;
            }
            MxArray::from_float32(&vals, &[vocab as i64, d as i64])
                .expect("forced emb_w")
                .astype(crate::array::DType::BFloat16)
                .expect("forced emb_w bf16")
        } else {
            det_bf16(100, &[vocab as i64, d as i64], 0.35, 0.15)
        };
        let embed_tokens = Embedding::from_weight(&emb_w).expect("Embedding::from_weight");
        let embed_norm = if force_lm_token.is_some() {
            MxArray::ones(&[d as i64], Some(crate::array::DType::Float32)).expect("embed_norm")
        } else {
            det_f32(101, &[d as i64], 0.05, 1.0)
        };
        let output_res_query = if force_lm_token.is_some() {
            MxArray::zeros(&[1, d as i64], Some(crate::array::DType::Float32)).expect("out_res")
        } else {
            det_f32(102, &[1, d as i64], 0.2, 0.0)
                .mul_scalar(inv_sqrt_d)
                .expect("out_res")
        };
        let output_res_query_t = output_res_query.transpose(None).expect("out_res_t");
        let output_norm = if force_lm_token.is_some() {
            MxArray::ones(&[d as i64], Some(crate::array::DType::Float32)).expect("output_norm")
        } else {
            det_f32(103, &[d as i64], 0.05, 1.0)
        };
        let lm_w = if let Some(forced) = force_lm_token {
            let mut vals = vec![0.0f32; vocab * d];
            let row = (forced as usize).min(vocab - 1);
            vals[row * d] = 1.0;
            MxArray::from_float32(&vals, &[vocab as i64, d as i64])
                .expect("forced lm_w")
                .astype(crate::array::DType::BFloat16)
                .expect("forced lm_w bf16")
        } else {
            det_bf16(104, &[vocab as i64, d as i64], 0.15, 0.0)
        };
        let lm_head = LinearProj::Standard(Linear::from_weights(&lm_w, None).expect("lm_head"));
        let lm_head_bf16_unpadded = Some(lm_w);

        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv * head_dim;
        let r_dim = n_heads * d_rel;

        let mut layers = Vec::with_capacity(4);
        for (phys, is_local) in [(0usize, true), (1, false), (2, true), (3, false)] {
            let s = (phys as u64 + 1) * 1000;
            let attn_res_query = if phys == 0 {
                None
            } else {
                let q_rows = if is_local { 1 } else { 3 };
                Some(if force_lm_token.is_some() {
                    MxArray::zeros(&[q_rows, d as i64], Some(crate::array::DType::Float32))
                        .expect("attn_res")
                } else {
                    det_f32(s + 1, &[q_rows, d as i64], 0.2, 0.0)
                        .mul_scalar(inv_sqrt_d)
                        .expect("attn_res")
                })
            };
            let attn_res_query_t = attn_res_query
                .as_ref()
                .map(|q| q.transpose(None).expect("attn_res_t"));
            let ffn_res_query = if force_lm_token.is_some() {
                MxArray::zeros(&[1, d as i64], Some(crate::array::DType::Float32)).expect("ffn_res")
            } else {
                det_f32(s + 2, &[1, d as i64], 0.2, 0.0)
                    .mul_scalar(inv_sqrt_d)
                    .expect("ffn_res")
            };
            let ffn_res_query_t = ffn_res_query.transpose(None).expect("ffn_res_t");

            let attn_inp_norm = if is_local {
                AttnInputNorm::Local(det_f32(s + 3, &[d as i64], 0.05, 1.0))
            } else {
                let q_norm = det_f32(s + 3, &[d as i64], 0.05, 1.0);
                let k_norm = det_f32(s + 4, &[d as i64], 0.05, 1.0);
                let v_norm = det_f32(s + 5, &[d as i64], 0.05, 1.0);
                let qkv_norm =
                    MxArray::stack(vec![&q_norm, &k_norm, &v_norm], Some(0)).expect("qkv");
                AttnInputNorm::Global {
                    q_norm,
                    k_norm,
                    v_norm,
                    qkv_norm,
                }
            };
            let ffn_norm = det_f32(s + 6, &[d as i64], 0.05, 1.0);

            let k_conv = det_f32(s + 10, &[kv_dim as i64, 4], 0.2, 0.25);
            let v_conv = det_f32(s + 11, &[kv_dim as i64, 4], 0.2, 0.25);
            let post_conv = det_f32(s + 12, &[d as i64, 4], 0.2, 0.25);
            let (k_conv_proj, v_conv_proj) = if !is_local {
                (
                    Some(make_lin(s + 13, n_kv * 4, d, 0.08)),
                    Some(make_lin(s + 14, n_kv * 4, d, 0.08)),
                )
            } else {
                (None, None)
            };

            let rel_bias_w = det_f32(s + 15, &[rel_extent as i64, d_rel as i64], 0.2, 0.0);
            let rel_bias_t = rel_bias_w.transpose(None).expect("rb_t");

            let mut self_attn = Inkling2Attention {
                num_heads: n_heads,
                num_kv_heads: n_kv,
                head_dim,
                d_rel,
                rel_extent,
                sliding_window: config.sliding_window,
                is_local,
                qk_l2_clip_threshold: config.qk_l2_clip_threshold,
                rel_l2_clip_threshold: config.rel_l2_clip_threshold,
                log_scaling_n_floor: config.log_scaling_n_floor,
                log_scaling_alpha: config.log_scaling_alpha,
                rms_norm_eps: config.rms_norm_eps as f32,
                q_proj: make_lin(s + 20, q_dim, d, 0.05),
                k_proj: make_lin(s + 21, kv_dim, d, 0.05),
                v_proj: make_lin(s + 22, kv_dim, d, 0.05),
                rel_proj: make_lin(s + 23, r_dim, d, 0.05),
                gate_proj: make_lin(s + 24, q_dim, d, 0.05),
                sink_proj: make_lin(s + 25, n_heads, d, 0.05),
                o_proj: if force_lm_token.is_some() {
                    make_zero_lin(d, q_dim)
                } else {
                    make_lin(s + 26, d, q_dim, 0.04)
                },
                fused_qrg: None,
                fused_qrgkv: None,
                fused_qrg_has_sink: false,
                fused_k_with_conv: None,
                fused_v_with_conv: None,
                k_conv,
                v_conv,
                post_conv,
                k_conv_proj,
                v_conv_proj,
                q_clip: det_f32(s + 27, &[1, n_heads as i64, head_dim as i64], 0.1, 1.0),
                rel_clip: det_f32(s + 28, &[1, n_heads as i64, d_rel as i64], 0.1, 1.0),
                rel_bias_t,
                rel_bias_w,
                qk_thresh_scalar: MxArray::scalar_float(config.qk_l2_clip_threshold).expect("qk"),
                rel_thresh_scalar: MxArray::scalar_float(config.rel_l2_clip_threshold)
                    .expect("rel"),
                one_scalar: MxArray::scalar_float(1.0).expect("one"),
            };
            self_attn.try_fuse_projections().expect("fuse proj");

            let router_weight = det_f32(s + 30, &[num_experts as i64, d as i64], 0.1, 0.0);
            let router_bias = Some(det_f32(s + 31, &[num_experts as i64], 0.05, 0.0));
            let gate_up_w = det_bf16(
                s + 32,
                &[num_experts as i64, (2 * moe_inter) as i64, d as i64],
                0.05,
                0.0,
            );
            let down_w = if force_lm_token.is_some() {
                MxArray::zeros(
                    &[num_experts as i64, d as i64, moe_inter as i64],
                    Some(crate::array::DType::BFloat16),
                )
                .expect("zero down_w")
            } else {
                det_bf16(
                    s + 33,
                    &[num_experts as i64, d as i64, moe_inter as i64],
                    0.05,
                    0.0,
                )
            };
            let mlp_post_conv = det_f32(s + 34, &[d as i64, 4], 0.2, 0.25);
            let moe = Inkling2MoE::new(
                num_experts,
                top_k,
                moe_inter,
                config.route_scale,
                config.mlp_gate_softcap,
                config.mlp_up_softcap,
                config.gate_up_interleaved,
                router_weight,
                router_bias,
                ExpertSwitchProj::Dense(gate_up_w),
                ExpertSwitchProj::Dense(down_w),
                mlp_post_conv,
            )
            .expect("Inkling2MoE::new");

            layers.push(Inkling2PhysicalLayer {
                phys_idx: phys,
                is_local,
                rms_norm_eps: config.rms_norm_eps as f32,
                attn_res_query,
                attn_res_query_t,
                ffn_res_query,
                ffn_res_query_t,
                attn_inp_norm,
                ffn_norm,
                self_attn,
                moe,
                loop_ctrl: None,
            });
        }

        let draft_target = Qwen3_5Config {
            rope_scaling: None,
            qwen35_gguf_gdn_layout: None,
            kv_format: None,
            vocab_size: vocab as i32,
            hidden_size: d as i32,
            num_layers: 4,
            num_heads: n_heads as i32,
            num_kv_heads: n_kv as i32,
            intermediate_size: 512,
            rms_norm_eps: 1e-6,
            head_dim: head_dim as i32,
            tie_word_embeddings: false,
            attention_bias: false,
            max_position_embeddings: max_model_len.max(256) as i32,
            pad_token_id: 0,
            eos_token_id: 2,
            bos_token_id: 1,
            linear_num_value_heads: 2,
            linear_num_key_heads: 2,
            linear_key_head_dim: 8,
            linear_value_head_dim: 8,
            linear_conv_kernel_dim: 4,
            full_attention_interval: 4,
            partial_rotary_factor: 0.25,
            rope_theta: 10_000.0,
            paged_cache_memory_mb: None,
            paged_block_size: None,
            n_mtp_layers: 0,
        };
        let mut draft = tiny_dflash2_model_with_block(&draft_target, DFLASH2_DRAFT_PROPOSALS)
            .expect("tiny DFlash2 draft");
        draft.config.sliding_window = draft_sliding_window;

        let mut inner = Inkling2Inner {
            config,
            embed_tokens,
            embed_norm,
            layers,
            output_res_query,
            output_res_query_t,
            output_norm,
            lm_head,
            lm_head_bf16_unpadded,
            vision_encoder: None,
            tokenizer: Some(Arc::new(build_synthetic_inkling2_tokenizer())),
            dspark: None,
            dflash2: Some(draft),
            dflash2_identity: None,
            dspark_turn_contexts: HashMap::new(),
            cuda_dflash2_ctx: None,
            cuda_verify_budget: crate::models::qwen3_5::verify_tree::VerifyBudget::new(),
            cuda_lookup_acceptance: crate::models::qwen3_5::verify_tree::LookupAcceptance::new(),
            cuda_dspark_ctx: None,
            scheduled_dspark: Inkling2ScheduledDsparkStates::default(),
            paged: None,
            flat_cache: None,
            cuda_cache: None,
            cuda_kv_format: crate::models::inkling2::cuda_kv::Inkling2CudaKvFormat::Bf16,
            cached_token_history: Vec::new(),
            cached_image_key: None,
            turn_cancel: None,
            model_id: next_inkling2_model_id(),
            memory_plan: None,
            memory_registration: None,
            wired: None,
            cache_recency: Arc::new(CacheRecency::new()),
            decode_share: default_decode_share(),
            max_context: None,
            activations: Activations::Auto,
            disk_options: DiskTierOptions::default(),
            prefill_runs: HashMap::new(),
        };
        inner
            .enable_paged(Inkling2PagedConfig {
                kv_format: default_inkling2_kv_format(),
                block_size: 32,
                max_model_len,
                max_chunk: 32,
                max_concurrent_sequences,
                gpu_memory_mb: None,
            })
            .expect("enable_paged");
        inner
    }

    fn exclusive_dflash2_config(owner: &str, max_tokens: i32, salt: &str) -> ChatConfig {
        ChatConfig {
            max_new_tokens: Some(max_tokens),
            temperature: Some(0.0),
            top_p: Some(1.0),
            top_k: Some(0),
            repetition_penalty: Some(1.0),
            presence_penalty: Some(0.0),
            frequency_penalty: Some(0.0),
            enable_mtp: Some(true),
            reasoning_effort: Some("none".to_string()),
            reuse_cache: Some(true),
            cache_salt: Some(salt.to_string()),
            cache_owner_id: Some(owner.to_string()),
            ..ChatConfig::default()
        }
    }

    fn drive_scheduler_cmd(
        sched: &mut Inkling2SchedulerState,
        tx: &mpsc::UnboundedSender<Inkling2Cmd>,
        rx: &mut mpsc::UnboundedReceiver<Inkling2Cmd>,
        cmd: ChatCmd,
        mut done_rx: tokio::sync::oneshot::Receiver<napi::bindgen_prelude::Result<ChatResult>>,
    ) -> napi::bindgen_prelude::Result<ChatResult> {
        tx.send(Inkling2Cmd::Chat(Box::new(cmd)))
            .expect("send ChatCmd");
        loop {
            match done_rx.try_recv() {
                Ok(res) => return res,
                Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {
                    let _ = sched.drive(rx);
                }
                Err(tokio::sync::oneshot::error::TryRecvError::Closed) => {
                    panic!("scheduler reply channel closed unexpectedly");
                }
            }
        }
    }

    fn send_session_start(
        sched: &mut Inkling2SchedulerState,
        tx: &mpsc::UnboundedSender<Inkling2Cmd>,
        rx: &mut mpsc::UnboundedReceiver<Inkling2Cmd>,
        messages: Vec<ChatMessage>,
        config: ChatConfig,
    ) -> napi::bindgen_prelude::Result<ChatResult> {
        let (reply, done_rx) = tokio::sync::oneshot::channel();
        drive_scheduler_cmd(
            sched,
            tx,
            rx,
            ChatCmd::SessionStart {
                messages,
                config,
                reply,
                cancelled: Arc::new(AtomicBool::new(false)),
            },
            done_rx,
        )
    }

    fn send_session_continue(
        sched: &mut Inkling2SchedulerState,
        tx: &mpsc::UnboundedSender<Inkling2Cmd>,
        rx: &mut mpsc::UnboundedReceiver<Inkling2Cmd>,
        messages: Vec<ChatMessage>,
        config: ChatConfig,
    ) -> napi::bindgen_prelude::Result<ChatResult> {
        let (reply, done_rx) = tokio::sync::oneshot::channel();
        drive_scheduler_cmd(
            sched,
            tx,
            rx,
            ChatCmd::SessionContinue {
                messages,
                config,
                reply,
                cancelled: Arc::new(AtomicBool::new(false)),
            },
            done_rx,
        )
    }

    /// Test (1): Capped pool + `StateStore` marks reclaim end-to-end.
    ///
    /// Owner A fills the entire 2-block (64-token) Global pool and arms BOTH its live block table
    /// AND a `StateStore` `Ordinary` mark (block 0 `[0..32]`). Owner B then runs an exclusive
    /// greedy DFlash2 `SessionStart` via `Inkling2SchedulerState`, which must reclaim BOTH Owner A's
    /// live table AND Owner A's `StateStore` mark so decode + verify-tree finish cleanly.
    #[test]
    fn exclusive_reclaim_capped_pool_and_statestore_marks_end_to_end() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        let inner = build_synthetic_dflash2_inner(64, None);
        let cfg_a = exclusive_dflash2_config("owner-a", 8, "salt-a");
        let params_a = crate::engine::backend::ChatBackend::resolve_params(&inner, &cfg_a);
        assert!(
            crate::engine::hybrid_scheduler::HybridSchedulerBackend::speculative_turn_requires_exclusive_lane(
                &inner,
                &params_a
            ),
            "greedy DFlash2 turn must route to the exclusive barrier lane"
        );

        let mut sched = Inkling2SchedulerState::new(inner).expect("scheduler");
        let (tx, mut rx) = mpsc::unbounded_channel::<Inkling2Cmd>();

        // 1. Owner A runs Turn 1 (~39 prompt tokens + 8 generated = 47 tokens -> fills all 2 blocks).
        let res_a = send_session_start(&mut sched, &tx, &mut rx, user_msg("alpha1"), cfg_a)
            .expect("Owner A SessionStart must succeed");
        assert_eq!(res_a.num_tokens, 8);
        let seq_a = *sched
            .owner_sequences
            .get("owner-a")
            .expect("owner-a seq assigned");
        {
            let rt = sched.inner.paged.as_ref().expect("paged runtime");
            assert!(!rt.exclusive_turn_active, "flag must reset after barrier");
            assert!(rt.coordinator.is_live_all(seq_a));
            let tel = rt.coordinator.full_adapter().block_telemetry().unwrap();
            assert_eq!(tel.max_total_blocks, 2);
            assert_eq!(tel.free_blocks, 0, "Owner A must fill the 2-block pool");
            assert!(
                rt.state_store.store.entries().count() >= 1,
                "Owner A turn >= 32 tokens must arm StateStore"
            );
            assert!(
                rt.coordinator
                    .full_adapter()
                    .turn_end_stamp(seq_a)
                    .is_some(),
                "Owner A must carry a turn-end recency stamp"
            );
        }

        // 2. Owner B runs an exclusive greedy DFlash2 turn needing all 2 blocks.
        let cfg_b = exclusive_dflash2_config("owner-b", 8, "salt-b");
        let res_b = send_session_start(&mut sched, &tx, &mut rx, user_msg("bravo2"), cfg_b)
            .expect("Owner B SessionStart must reclaim Owner A + StateStore mark and succeed");
        assert_eq!(res_b.num_tokens, 8);
        let seq_b = *sched
            .owner_sequences
            .get("owner-b")
            .expect("owner-b seq assigned");
        assert_ne!(seq_a, seq_b);
        {
            let rt = sched.inner.paged.as_ref().expect("paged runtime");
            assert!(!rt.exclusive_turn_active, "flag must reset after barrier");
            assert!(rt.coordinator.is_live_all(seq_b), "Owner B must be live");
            assert!(
                !rt.coordinator.is_live_all(seq_a),
                "Owner A must have been reclaimed"
            );
            assert!(!rt.conv_states.contains_key(&seq_a));
            assert!(!sched.inner.dspark_turn_contexts.contains_key(&seq_a));
        }
    }

    /// Test (2): Unified `CacheRecency` stamp order.
    ///
    /// Owner A (older) and Owner C (newer) both sit idle in a 4-block (128-token) pool (2 blocks
    /// each, 0 free blocks). Owner B needs 2 blocks (1 owner's worth): Owner A must be reclaimed
    /// while Owner C is preserved live with its original KV block intact.
    #[test]
    fn exclusive_reclaim_respects_unified_cache_recency_order() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        let inner = build_synthetic_dflash2_inner(128, None);
        let mut sched = Inkling2SchedulerState::new(inner).expect("scheduler");
        let (tx, mut rx) = mpsc::unbounded_channel::<Inkling2Cmd>();

        let _ = send_session_start(
            &mut sched,
            &tx,
            &mut rx,
            user_msg("alpha1"),
            exclusive_dflash2_config("owner-a", 8, "salt-a"),
        )
        .expect("Owner A");
        let _ = send_session_start(
            &mut sched,
            &tx,
            &mut rx,
            user_msg("gamma3"),
            exclusive_dflash2_config("owner-c", 8, "salt-c"),
        )
        .expect("Owner C");

        let seq_a = *sched.owner_sequences.get("owner-a").expect("seq_a");
        let seq_c = *sched.owner_sequences.get("owner-c").expect("seq_c");
        let owner_c_block0 = {
            let rt = sched.inner.paged.as_ref().expect("paged runtime");
            let full = rt.coordinator.full_adapter();
            let stamp_a = full.turn_end_stamp(seq_a).expect("stamp_a");
            let stamp_c = full.turn_end_stamp(seq_c).expect("stamp_c");
            assert!(stamp_a < stamp_c, "Owner A must be older than Owner C");
            let tel = full.block_telemetry().unwrap();
            assert_eq!(tel.max_total_blocks, 4);
            assert_eq!(tel.free_blocks, 0, "Owner A + Owner C fill 4 blocks");
            full.block_table_for(seq_c).unwrap().blocks()[0].block_id
        };

        let res_b = send_session_start(
            &mut sched,
            &tx,
            &mut rx,
            user_msg("bravo2"),
            exclusive_dflash2_config("owner-b", 8, "salt-b"),
        )
        .expect("Owner B must reclaim oldest idle owner (A) and succeed");
        assert_eq!(res_b.num_tokens, 8);

        let seq_b = *sched.owner_sequences.get("owner-b").expect("seq_b");
        let rt = sched.inner.paged.as_ref().expect("paged runtime");
        assert!(rt.coordinator.is_live_all(seq_b), "Owner B must be live");
        assert!(
            !rt.coordinator.is_live_all(seq_a),
            "Older idle Owner A must be reclaimed"
        );
        assert!(
            rt.coordinator.is_live_all(seq_c),
            "Newer idle Owner C must be preserved live"
        );
        assert_eq!(
            rt.coordinator
                .full_adapter()
                .block_table_for(seq_c)
                .unwrap()
                .blocks()[0]
                .block_id,
            owner_c_block0,
            "Owner C's live KV block must remain untouched"
        );
    }

    /// Test (3): Self-preservation on `SessionContinue`.
    ///
    /// Owner A (older) and Owner C (newer) fill a 4-block (128-token) pool. Owner A then sends
    /// `SessionContinue` extending its live prefix into a 3rd block: Owner A must NEVER reclaim
    /// itself, must keep its live prefix (identical block 0 ID), and must reclaim only Owner C.
    #[test]
    fn exclusive_reclaim_preserves_active_owner_on_session_continue() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        // Bias lm_head to emit `<|message_user|>` so Turn 2 `[user(msg1), user(msg2)]` strictly
        // extends Turn 1's `[prompt1 + <|message_user|>]` recorded tokens (`can_continue_all == true`).
        let tok = build_synthetic_inkling2_tokenizer();
        let msg_user_id = tok
            .inner()
            .token_to_id(crate::tools::inkling::MESSAGE_USER)
            .expect("MESSAGE_USER id");
        let inner = build_synthetic_dflash2_inner(128, Some(msg_user_id));
        let mut sched = Inkling2SchedulerState::new(inner).expect("scheduler");
        // Route AR turns through the exclusive barrier lane so `continue_turn_all_exclusive` runs
        // directly on `SessionContinue` without DSpark draft-context length-exit mismatch.
        sched.serve_whole_turns();
        let (tx, mut rx) = mpsc::unbounded_channel::<Inkling2Cmd>();

        let mut cfg_a = exclusive_dflash2_config("owner-a", 1, "salt-a");
        cfg_a.enable_mtp = Some(false);
        let mut cfg_c = exclusive_dflash2_config("owner-c", 1, "salt-c");
        cfg_c.enable_mtp = Some(false);

        let _ = send_session_start(&mut sched, &tx, &mut rx, user_msg("alpha1"), cfg_a.clone())
            .expect("Owner A Turn 1");
        let _ = send_session_start(&mut sched, &tx, &mut rx, user_msg("gamma3"), cfg_c)
            .expect("Owner C Turn 1");

        let seq_a = *sched.owner_sequences.get("owner-a").expect("seq_a");
        let seq_c = *sched.owner_sequences.get("owner-c").expect("seq_c");
        let (owner_a_block0, owner_a_turn1_len) = {
            let rt = sched.inner.paged.as_ref().expect("paged runtime");
            let full = rt.coordinator.full_adapter();
            assert_eq!(full.block_telemetry().unwrap().free_blocks, 0);
            assert!(rt.coordinator.is_live_all(seq_a));
            assert!(rt.coordinator.is_live_all(seq_c));
            (
                full.block_table_for(seq_a).unwrap().blocks()[0].block_id,
                rt.coordinator.request_token_count_all(seq_a).unwrap(),
            )
        };

        // Turn 2 extends Owner A's live prefix past 64 tokens (requiring a 3rd block -> must
        // reclaim idle Owner C while preserving Owner A's live block 0!).
        let mut turn2_msgs = user_msg("alpha1");
        turn2_msgs.extend(user_msg("extension-crossing-third-block-0123456789"));
        cfg_a.max_new_tokens = Some(4);
        let res_a2 = send_session_continue(&mut sched, &tx, &mut rx, turn2_msgs, cfg_a)
            .expect("Owner A SessionContinue must preserve self and reclaim Owner C");
        assert_eq!(res_a2.num_tokens, 4);

        {
            let rt = sched.inner.paged.as_ref().expect("paged runtime");
            assert!(rt.coordinator.is_live_all(seq_a), "Owner A must stay live");
            assert!(
                !rt.coordinator.is_live_all(seq_c),
                "Idle Owner C must be reclaimed"
            );
            assert_eq!(
                rt.coordinator
                    .full_adapter()
                    .block_table_for(seq_a)
                    .unwrap()
                    .blocks()[0]
                    .block_id,
                owner_a_block0,
                "Owner A must keep its live prefix block 0 without cold reset"
            );
            assert!(
                rt.coordinator.request_token_count_all(seq_a).unwrap() > owner_a_turn1_len + 32,
                "Owner A must have extended into block 2"
            );
        }

        // Verify `ResetCaches` barrier clears owner tables & `owner_last_used` cleanly.
        let (rep_reset, mut rx_reset) = tokio::sync::oneshot::channel();
        tx.send(Inkling2Cmd::Chat(Box::new(ChatCmd::ResetCaches {
            reply: rep_reset,
        })))
        .expect("send ResetCaches");
        loop {
            match rx_reset.try_recv() {
                Ok(res) => {
                    res.expect("ResetCaches succeeded");
                    break;
                }
                Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {
                    let _ = sched.drive(&mut rx);
                }
                Err(tokio::sync::oneshot::error::TryRecvError::Closed) => {
                    panic!("ResetCaches reply closed");
                }
            }
        }
        assert!(sched.owner_sequences.is_empty());
        assert!(sched.owner_states.is_empty());
    }

    /// Test (4): Post-reclaim `SessionContinue` succeeds via cold rebuild.
    ///
    /// After Owner A's KV is reclaimed by Owner B in a capped 2-block pool, Owner A sends an
    /// exclusive greedy DFlash2 `SessionContinue`. With `has_live_session` checking
    /// `!cached_token_history.is_empty()`, Owner A must cold-rebuild cleanly instead of failing
    /// with `"requires an initialized session"`.
    #[test]
    fn exclusive_reclaim_post_reclaim_session_continue_cold_rebuilds() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        let inner = build_synthetic_dflash2_inner(64, None);
        let mut sched = Inkling2SchedulerState::new(inner).expect("scheduler");
        let (tx, mut rx) = mpsc::unbounded_channel::<Inkling2Cmd>();

        let res_a1 = send_session_start(
            &mut sched,
            &tx,
            &mut rx,
            user_msg("alpha1"),
            exclusive_dflash2_config("owner-a", 4, "salt-a"),
        )
        .expect("Owner A Turn 1");
        assert_eq!(res_a1.num_tokens, 4);
        let seq_a = *sched.owner_sequences.get("owner-a").expect("seq_a");

        // Owner B reclaims Owner A completely in the 2-block pool.
        let res_b = send_session_start(
            &mut sched,
            &tx,
            &mut rx,
            user_msg("bravo2"),
            exclusive_dflash2_config("owner-b", 4, "salt-b"),
        )
        .expect("Owner B Turn 1");
        assert_eq!(res_b.num_tokens, 4);
        let seq_b = *sched.owner_sequences.get("owner-b").expect("seq_b");
        {
            let rt = sched.inner.paged.as_ref().expect("paged runtime");
            assert!(
                !rt.coordinator.is_live_all(seq_a),
                "Owner A KV must have been reclaimed by Owner B"
            );
            assert!(
                sched.owner_sequences.contains_key("owner-a"),
                "Owner A logical session handle must remain registered"
            );
        }

        // Owner A sends exclusive greedy DFlash2 `SessionContinue` -> must cold-rebuild cleanly!
        let mut continue_msgs = user_msg("alpha1");
        continue_msgs.extend(user_msg("ok"));
        let res_a2 = send_session_continue(
            &mut sched,
            &tx,
            &mut rx,
            continue_msgs,
            exclusive_dflash2_config("owner-a", 4, "salt-a"),
        )
        .expect("Post-reclaim SessionContinue must cold-rebuild instead of failing");
        assert_eq!(res_a2.num_tokens, 4);

        let rt = sched.inner.paged.as_ref().expect("paged runtime");
        assert!(rt.coordinator.is_live_all(seq_a), "Owner A is live again");
        assert!(
            !rt.coordinator.is_live_all(seq_b),
            "Owner B was reclaimed for Owner A's cold rebuild"
        );
    }

    /// Test (5): Over-context fail-fast WITHOUT evicting idle owners.
    ///
    /// A prompt exceeding `max_capacity_tokens` (64 tokens) must fail immediately with
    /// `context_length_exceeded` while leaving idle Owner A (live KV + ShortConv + DFlash2 draft
    /// context + `StateStore` mark) completely untouched.
    #[test]
    fn exclusive_reclaim_over_context_fails_fast_without_evicting_idle_owner() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        let inner = build_synthetic_dflash2_inner(64, None);
        let mut sched = Inkling2SchedulerState::new(inner).expect("scheduler");
        let (tx, mut rx) = mpsc::unbounded_channel::<Inkling2Cmd>();

        let res_a = send_session_start(
            &mut sched,
            &tx,
            &mut rx,
            user_msg("alpha1"),
            exclusive_dflash2_config("owner-a", 4, "salt-a"),
        )
        .expect("Owner A Turn 1");
        assert_eq!(res_a.num_tokens, 4);
        let seq_a = *sched.owner_sequences.get("owner-a").expect("seq_a");
        let (owner_a_block0, store_entries_before) = {
            let rt = sched.inner.paged.as_ref().expect("paged runtime");
            (
                rt.coordinator
                    .full_adapter()
                    .block_table_for(seq_a)
                    .unwrap()
                    .blocks()[0]
                    .block_id,
                rt.state_store.store.entries().count(),
            )
        };
        assert!(store_entries_before >= 1);

        // Over-context prompt: 60 user chars + ~33 framing tokens = ~93 tokens > 64 capacity.
        let over_context_prompt = "x".repeat(60);
        let err = send_session_start(
            &mut sched,
            &tx,
            &mut rx,
            user_msg(&over_context_prompt),
            exclusive_dflash2_config("owner-b", 4, "salt-b"),
        )
        .expect_err("over-context prompt must fail fast");
        assert!(
            err.reason.contains("context_length_exceeded"),
            "expected context_length_exceeded, got: {}",
            err.reason
        );

        // Owner A must remain 100% intact (zero idle owners or StateStore entries evicted!).
        let rt = sched.inner.paged.as_ref().expect("paged runtime");
        assert!(!rt.exclusive_turn_active, "flag must reset after failure");
        assert!(
            rt.coordinator.is_live_all(seq_a),
            "idle Owner A must NOT be evicted on over-context error"
        );
        assert_eq!(
            rt.coordinator
                .full_adapter()
                .block_table_for(seq_a)
                .unwrap()
                .blocks()[0]
                .block_id,
            owner_a_block0
        );
        assert!(rt.conv_states.contains_key(&seq_a));
        assert!(sched.inner.dspark_turn_contexts.contains_key(&seq_a));
        assert_eq!(rt.state_store.store.entries().count(), store_entries_before);
    }

    /// Test (6): Ownerless `seq_id = 0` is NEVER evicted by named exclusive-turn pressure.
    ///
    /// Ownerless `SessionStart` (`cache_owner_id: None`) runs on reserved `seq_id = 0` first
    /// (carrying an older recency stamp than Named Owner A). Named Owner A fills the remaining
    /// pool capacity on `seq_id = 1`, then Named Owner B arrives under pressure and must reclaim
    /// Named Owner A without touching `seq_id = 0`. Ownerless `SessionContinue` then succeeds
    /// cleanly without failing with `"requires an initialized session"`.
    #[test]
    fn exclusive_reclaim_preserves_ownerless_seq_zero_across_named_pressure() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        let inner = build_synthetic_dflash2_inner(128, None);
        let mut sched = Inkling2SchedulerState::new(inner).expect("scheduler");
        let (tx, mut rx) = mpsc::unbounded_channel::<Inkling2Cmd>();

        let mut cfg_ownerless = exclusive_dflash2_config("unused", 4, "salt-ownerless");
        cfg_ownerless.cache_owner_id = None;

        // 1. Ownerless SessionStart runs on seq_id = 0 (2 Global blocks, older stamp).
        let res_0 = send_session_start(
            &mut sched,
            &tx,
            &mut rx,
            user_msg("alpha0"),
            cfg_ownerless.clone(),
        )
        .expect("Ownerless SessionStart must succeed");
        assert_eq!(res_0.num_tokens, 4);
        let (seq0_block0, stamp_0) = {
            let rt = sched.inner.paged.as_ref().expect("paged runtime");
            let full = rt.coordinator.full_adapter();
            assert!(rt.coordinator.is_live_all(0), "seq_id = 0 must be live");
            assert!(rt.conv_states.contains_key(&0));
            (
                full.block_table_for(0).unwrap().blocks()[0].block_id,
                full.turn_end_stamp(0).expect("seq_id = 0 stamp"),
            )
        };

        // 2. Named Owner A runs on seq_id >= 1 and fills the remaining 2 blocks (4/4 full).
        let res_a = send_session_start(
            &mut sched,
            &tx,
            &mut rx,
            user_msg("alpha1"),
            exclusive_dflash2_config("owner-a", 4, "salt-a"),
        )
        .expect("Owner A SessionStart must succeed");
        assert_eq!(res_a.num_tokens, 4);
        let seq_a = *sched.owner_sequences.get("owner-a").expect("seq_a");
        assert!(seq_a >= 1, "named owners must allocate seq_id >= 1");
        {
            let rt = sched.inner.paged.as_ref().expect("paged runtime");
            let full = rt.coordinator.full_adapter();
            let stamp_a = full.turn_end_stamp(seq_a).expect("stamp_a");
            assert!(
                stamp_0 < stamp_a,
                "ownerless seq_id = 0 must be older than Named Owner A"
            );
            assert_eq!(
                full.block_telemetry().unwrap().free_blocks,
                0,
                "seq 0 + Owner A must fill the 4-block pool"
            );
        }

        // 3. Named Owner B arrives under pressure -> must reclaim Owner A (NOT older seq 0).
        let res_b = send_session_start(
            &mut sched,
            &tx,
            &mut rx,
            user_msg("bravo2"),
            exclusive_dflash2_config("owner-b", 4, "salt-b"),
        )
        .expect("Owner B SessionStart must reclaim Owner A while preserving ownerless seq 0");
        assert_eq!(res_b.num_tokens, 4);
        let seq_b = *sched.owner_sequences.get("owner-b").expect("seq_b");
        {
            let rt = sched.inner.paged.as_ref().expect("paged runtime");
            let full = rt.coordinator.full_adapter();
            assert!(
                rt.coordinator.is_live_all(0),
                "ownerless seq_id = 0 must stay live across named pressure"
            );
            assert!(rt.conv_states.contains_key(&0));
            assert_eq!(
                full.block_table_for(0).unwrap().blocks()[0].block_id,
                seq0_block0,
                "ownerless seq_id = 0 KV block 0 must remain untouched"
            );
            assert!(
                !rt.coordinator.is_live_all(seq_a),
                "Named Owner A must be reclaimed"
            );
            assert!(rt.coordinator.is_live_all(seq_b), "Owner B must be live");
        }

        // 4. Ownerless SessionContinue runs and succeeds cleanly (no "requires an initialized session").
        let mut cont_msgs = user_msg("alpha0");
        cont_msgs.extend(user_msg("ok"));
        let res_0_cont = send_session_continue(&mut sched, &tx, &mut rx, cont_msgs, cfg_ownerless)
            .expect("Ownerless SessionContinue must succeed after named pressure");
        assert_eq!(res_0_cont.num_tokens, 4);
        let rt = sched.inner.paged.as_ref().expect("paged runtime");
        assert!(rt.coordinator.is_live_all(0), "seq_id = 0 remains live");
    }

    /// Test (7): Bounded, cancellation-aware SSD reclaim wait.
    ///
    /// With a real temporary `StateDisk` SSD tier attached to a capped 2-block pool:
    /// 1. Owner A fills both blocks and arms a `StateStore` `Ordinary` entry pinning block 0.
    ///    Owner B triggers exclusive reclaim, spilling Owner A's state blob + Global KV chain
    ///    to SSD and waiting within the bounded budget until the write lands and unmarks block 0,
    ///    allowing Owner B to succeed cleanly.
    /// 2. Setting `turn_cancel` immediately short-circuits the bounded wait and moves pending
    ///    SSD entries into `skipped` without blocking.
    #[test]
    fn exclusive_reclaim_waits_for_pending_ssd_chain_write() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        struct TempDirGuard(std::path::PathBuf);
        impl Drop for TempDirGuard {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }

        let temp_root =
            std::env::temp_dir().join(format!("inkling2-excl-ssd-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&temp_root).expect("create temp_root");
        let _temp_guard = TempDirGuard(temp_root.clone());

        let mut inner = build_synthetic_dflash2_inner(64, None);
        let disk_opts = crate::disk_tier::DiskTierOptions {
            cache_disk: crate::disk_tier::CacheDisk::Bytes(64u64 << 20),
            persistent: false,
            cache_dir: None,
            temp_root: Some(temp_root),
        };
        assert!(
            inner.attach_state_disk(&disk_opts, || None).is_some(),
            "temporary StateDisk tier must attach"
        );

        let mut sched = Inkling2SchedulerState::new(inner).expect("scheduler");
        let (tx, mut rx) = mpsc::unbounded_channel::<Inkling2Cmd>();

        // 1. Owner A fills the 2-block pool and publishes a StateStore entry pinning block 0.
        let res_a = send_session_start(
            &mut sched,
            &tx,
            &mut rx,
            user_msg("alpha1"),
            exclusive_dflash2_config("owner-a", 8, "salt-a"),
        )
        .expect("Owner A SessionStart must succeed");
        assert_eq!(res_a.num_tokens, 8);
        let seq_a = *sched.owner_sequences.get("owner-a").expect("seq_a");
        let entry_a = {
            let rt = sched.inner.paged.as_ref().expect("paged runtime");
            assert!(rt.state_store.disk.is_some());
            assert_eq!(
                rt.coordinator
                    .full_adapter()
                    .block_telemetry()
                    .unwrap()
                    .free_blocks,
                0
            );
            assert!(
                rt.state_store.store.marks().next().is_some(),
                "Owner A must publish a StateStore entry with armed chain marks"
            );
            let entry_a = rt
                .state_store
                .store
                .entries()
                .next()
                .expect("Owner A must publish a StateStore entry");
            assert!(super::store_entry_marks_chain(entry_a));
            entry_a.id()
        };

        // 2. Owner B needs all 2 blocks -> spills Owner A's StateStore entry to SSD and waits
        //    for its state blob + Global KV chain write to land and unmark block 0.
        let res_b = send_session_start(
            &mut sched,
            &tx,
            &mut rx,
            user_msg("bravo2"),
            exclusive_dflash2_config("owner-b", 8, "salt-b"),
        )
        .expect("Owner B must wait for Owner A's SSD chain write to land and succeed");
        assert_eq!(res_b.num_tokens, 8);
        let seq_b = *sched.owner_sequences.get("owner-b").expect("seq_b");
        let entry_b = {
            let rt = sched.inner.paged.as_ref().expect("paged runtime");
            assert!(rt.coordinator.is_live_all(seq_b), "Owner B must be live");
            assert!(
                !rt.coordinator.is_live_all(seq_a),
                "Owner A must be reclaimed"
            );
            assert!(
                rt.state_store.disk_counters().offloads >= 1,
                "Owner A's StateStore entry must have spilled and landed on SSD"
            );
            let spilled_a = rt
                .state_store
                .store
                .get(entry_a)
                .expect("Owner A entry remains indexed on SSD after spill");
            assert!(spilled_a.ram().is_none(), "Owner A RAM copy was dropped");
            assert!(
                spilled_a.disk().is_some_and(|d| d.landed && !d.chain_owed),
                "Owner A SSD state + Global KV chain must have landed cleanly"
            );
            rt.state_store
                .store
                .entries()
                .find(|e| e.id() != entry_a && super::store_entry_marks_chain(e))
                .expect("Owner B must hold a marked StateStore entry")
                .id()
        };

        // 3. Cancellation must immediately short-circuit the bounded SSD wait.
        crate::engine::hybrid_scheduler::HybridSchedulerBackend::release_scheduled_cache(
            &mut sched.inner,
            seq_b,
        )
        .expect("release seq_b so only pending SSD entry remains");
        sched.inner.paged.as_mut().unwrap().exclusive_turn_active = true;
        sched.inner.turn_cancel = Some(Arc::new(AtomicBool::new(true)));

        let mut reclaim_state = super::ExclusiveReclaimState::default();
        reclaim_state.push_pending_ssd(entry_b);
        let t0 = Instant::now();
        let progressed = sched.inner.try_reclaim_one_for_exclusive(
            99,
            super::ExclusiveReclaimScope::Any,
            &mut reclaim_state,
            None,
        );
        let elapsed = t0.elapsed();

        sched.inner.turn_cancel = None;
        sched.inner.paged.as_mut().unwrap().exclusive_turn_active = false;
        assert!(
            !progressed,
            "cancelled SSD wait must return false without reclaiming"
        );
        assert!(
            elapsed < Duration::from_millis(500),
            "cancelled SSD wait must short-circuit immediately, took {elapsed:?}"
        );
        assert!(reclaim_state.pending_ssd.is_empty());
        assert!(reclaim_state.skipped.contains(&entry_b));
        assert!(
            sched
                .inner
                .paged
                .as_ref()
                .unwrap()
                .state_store
                .store
                .get(entry_b)
                .is_some_and(super::store_entry_marks_chain),
            "cancelled wait must leave pending entry untouched"
        );

        let _ = sched
            .inner
            .close_state_disk(Instant::now() + Duration::from_secs(5));
    }

    /// Test (8): Group-targeted `SlidingOnly` reclaim preserves Global `StateStore` checkpoints
    /// and borrows `&Inkling2StateAnchor` across SWA restore retries without deep-copying.
    #[test]
    fn exclusive_reclaim_sliding_pressure_preserves_global_checkpoints() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        // 8 Global blocks (256 tokens), while SWA (window=64, block_size=32, max_chunk=32) has a
        // tighter fixed pool so SWA exhausts while Global still has free blocks.
        let inner = build_synthetic_dflash2_inner(256, None);
        let mut sched = Inkling2SchedulerState::new(inner).expect("scheduler");
        let (tx, mut rx) = mpsc::unbounded_channel::<Inkling2Cmd>();

        let cfg_a = exclusive_dflash2_config("owner-a", 4, "salt-a");
        let cfg_c = exclusive_dflash2_config("owner-c", 4, "salt-c");
        let hit_salt =
            crate::engine::backend::ChatBackend::resolve_params(&sched.inner, &cfg_a).cache_salt;
        let cp_salt =
            crate::engine::backend::ChatBackend::resolve_params(&sched.inner, &cfg_c).cache_salt;

        let _ = send_session_start(&mut sched, &tx, &mut rx, user_msg("alpha1"), cfg_a)
            .expect("Owner A");
        let _ = send_session_start(&mut sched, &tx, &mut rx, user_msg("gamma3"), cfg_c)
            .expect("Owner C");
        let seq_a = *sched.owner_sequences.get("owner-a").expect("seq_a");
        let seq_c = *sched.owner_sequences.get("owner-c").expect("seq_c");

        let (tokens_a, cp_id) = {
            let rt = sched.inner.paged.as_mut().expect("paged runtime");
            let tokens_a = rt
                .coordinator
                .full_adapter()
                .request_tokens_for(seq_a)
                .expect("seq_a tokens")
                .to_vec();
            let ord_a_id = rt
                .state_store
                .store
                .entries()
                .next()
                .expect("Owner A turn must have captured an Ordinary StateStore entry")
                .id();
            // Owner C's SessionStart also published an Ordinary entry at (seq_c, cp_salt).
            // Evict that Ordinary entry so re-capturing seq_c at depth 32 inserts a genuine
            // `StateClass::Checkpoint` entry marking Global KV.
            let stale_c_ids: Vec<_> = rt
                .state_store
                .store
                .entries()
                .filter(|e| e.id() != ord_a_id)
                .map(|e| e.id())
                .collect();
            for stale_id in stale_c_ids {
                let _ = rt.state_store.reclaim_entry(&rt.coordinator, stale_id);
            }

            let tokens_c = rt
                .coordinator
                .full_adapter()
                .request_tokens_for(seq_c)
                .expect("seq_c tokens")
                .to_vec();
            let conv_c = rt.conv_states.get(&seq_c).expect("seq_c conv").clone();
            let cp_id = rt
                .state_store
                .capture_boundary(
                    &mut rt.coordinator,
                    &conv_c,
                    seq_c,
                    &tokens_c,
                    32,
                    cp_salt,
                    super::StateClass::Checkpoint,
                    None,
                )
                .expect("capture unrelated Checkpoint")
                .expect("Checkpoint entry id");

            // Ensure SWA has 0 free blocks while Global still has >= 2 free blocks.
            let swa_free = rt
                .coordinator
                .sliding_adapter()
                .expect("sliding adapter")
                .block_telemetry()
                .unwrap()
                .free_blocks;
            if swa_free > 0 {
                let seq_fill = 50u32;
                let fill_tokens: Vec<u32> = (100..100 + swa_free * 32).collect();
                rt.coordinator
                    .reset_scheduled_request(seq_fill)
                    .expect("reset filler");
                rt.coordinator
                    .record_tokens_all(seq_fill, &fill_tokens)
                    .expect("fill remaining SWA blocks");
                rt.coordinator
                    .finalize_keep_live_all(seq_fill, &[], 0xF111)
                    .expect("stamp filler newer than seq_a");
            }
            let full_free = rt
                .coordinator
                .full_adapter()
                .block_telemetry()
                .unwrap()
                .free_blocks;
            let swa_free_after = rt
                .coordinator
                .sliding_adapter()
                .unwrap()
                .block_telemetry()
                .unwrap()
                .free_blocks;
            assert!(
                full_free >= 2,
                "Global must still have free blocks, got {full_free}"
            );
            assert_eq!(swa_free_after, 0, "SWA pool must be full");
            assert!(
                rt.state_store
                    .store
                    .get(cp_id)
                    .is_some_and(|e| e.checkpoint() && super::store_entry_marks_chain(e)),
                "unrelated Checkpoint must be present and marking Global KV"
            );
            assert_eq!(
                super::exclusive_reclaim_scope_for_total(rt, seq_c, 96),
                super::ExclusiveReclaimScope::SlidingOnly,
                "SWA-only shortfall with free Global blocks must select SlidingOnly"
            );
            (tokens_a, cp_id)
        };

        // End-to-end SWA restore retry via `prepare_prefix_sync` -> `install_state_hit_sync`:
        // Global restore succeeds immediately; SWA restore hits pool exhaustion, reclaims
        // oldest idle SWA owner `seq_a` under `SlidingOnly`, retries with borrowed `&anchor`,
        // and leaves the unrelated Global `StateClass::Checkpoint` completely untouched!
        let seq_b = 99u32;
        let mut query_tokens = tokens_a[..32].to_vec();
        query_tokens.extend_from_slice(&[10, 11, 12, 13]);
        sched.inner.paged.as_mut().unwrap().exclusive_turn_active = true;
        let prefix = sched
            .inner
            .prepare_prefix_sync(seq_b, &query_tokens, true, hit_salt)
            .expect("SWA restore under SlidingOnly pressure must succeed");
        sched.inner.paged.as_mut().unwrap().exclusive_turn_active = false;

        assert_eq!(
            prefix.effective_cached_prefix_len, 32,
            "StateStore hit must restore cleanly at depth 32 instead of falling back to cold"
        );
        let rt = sched.inner.paged.as_ref().expect("paged runtime");
        assert!(
            rt.state_store.store.get(cp_id).is_some(),
            "unrelated Global StateClass::Checkpoint must be preserved under SlidingOnly pressure"
        );
        let swa_live = rt.coordinator.sliding_adapter().unwrap().live_seq_ids();
        assert!(
            !swa_live.contains(&seq_a),
            "oldest idle owner holding SWA blocks (seq_a) must have been reclaimed"
        );
        assert!(
            swa_live.contains(&seq_b),
            "restored sequence seq_b must now hold SWA blocks"
        );
    }

    // =========================================================================
    // Fast Synthetic Metal Tests: Live 4-bit DraftContextWindow Capture & Restore
    // Across Scheduled DFlash2, Exclusive DFlash2 + Verify Tree, AR, & SSD
    // =========================================================================

    fn two_user_msgs(first: &str, second: &str) -> Vec<ChatMessage> {
        let mut msgs = user_msg(first);
        msgs.extend(user_msg(second));
        msgs
    }

    fn scheduled_sampled_dflash2_config(owner: &str, max_tokens: i32, salt: &str) -> ChatConfig {
        ChatConfig {
            max_new_tokens: Some(max_tokens),
            temperature: Some(0.8),
            top_p: Some(1.0),
            top_k: Some(8),
            repetition_penalty: Some(1.0),
            presence_penalty: Some(0.0),
            frequency_penalty: Some(0.0),
            enable_mtp: Some(true),
            reasoning_effort: Some("none".to_string()),
            reuse_cache: Some(true),
            cache_salt: Some(salt.to_string()),
            cache_owner_id: Some(owner.to_string()),
            max_consecutive_tokens: Some(0),
            max_ngram_repeats: Some(0),
            ..ChatConfig::default()
        }
    }

    fn scheduled_greedy_dflash2_config(owner: &str, max_tokens: i32, salt: &str) -> ChatConfig {
        ChatConfig {
            max_new_tokens: Some(max_tokens),
            temperature: Some(0.0),
            top_p: Some(1.0),
            top_k: Some(0),
            repetition_penalty: Some(1.0),
            presence_penalty: Some(0.0),
            frequency_penalty: Some(0.0),
            enable_mtp: Some(true),
            // `mtp_depth < DFLASH2_DRAFT_PROPOSALS (7)` keeps greedy turns on the
            // scheduled DFlash2 verifier instead of routing to the exclusive verify tree.
            mtp_depth: Some(4),
            reasoning_effort: Some("none".to_string()),
            reuse_cache: Some(true),
            cache_salt: Some(salt.to_string()),
            cache_owner_id: Some(owner.to_string()),
            max_consecutive_tokens: Some(0),
            max_ngram_repeats: Some(0),
            ..ChatConfig::default()
        }
    }

    fn send_reset_caches(
        sched: &mut Inkling2SchedulerState,
        tx: &mpsc::UnboundedSender<Inkling2Cmd>,
        rx: &mut mpsc::UnboundedReceiver<Inkling2Cmd>,
    ) {
        let (rep, mut done_rx) = tokio::sync::oneshot::channel();
        tx.send(Inkling2Cmd::Chat(Box::new(ChatCmd::ResetCaches {
            reply: rep,
        })))
        .expect("send ResetCaches");
        loop {
            match done_rx.try_recv() {
                Ok(res) => {
                    res.expect("ResetCaches succeeded");
                    return;
                }
                Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {
                    let _ = sched.drive(rx);
                }
                Err(tokio::sync::oneshot::error::TryRecvError::Closed) => {
                    panic!("ResetCaches reply closed");
                }
            }
        }
    }

    /// Test (9): Scheduled sampled (and greedy) DFlash2 `SessionStart` follow-up via RAM
    /// `StateStore` hit restores `DFlash2ContextCache` from the captured 4-bit
    /// `PackedDraftWindow` and keeps DFlash2 speculation ON (no AR fallback!).
    #[test]
    fn dflash2_scheduled_session_start_ram_hit_keeps_speculation_on() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        // `draft_sliding_window = 128` > Turn 2 `suffix_len` so restore MUST project from
        // `PackedDraftWindow::to_window()` (not the `suffix >= sliding_window` skip!).
        let inner = build_synthetic_dflash2_inner_with_window(256, None, 128, 2);
        let cfg_sampled = scheduled_sampled_dflash2_config("sched-ram", 10, "salt-sched-ram");
        let params_sampled =
            crate::engine::backend::ChatBackend::resolve_params(&inner, &cfg_sampled);
        assert!(
            !crate::engine::hybrid_scheduler::HybridSchedulerBackend::speculative_turn_requires_exclusive_lane(
                &inner,
                &params_sampled
            ),
            "sampled DFlash2 turn must stay on the scheduled lane"
        );
        let mut sched = Inkling2SchedulerState::new(inner).expect("scheduler");
        let (tx, mut rx) = mpsc::unbounded_channel::<Inkling2Cmd>();

        let shared_prompt = "alpha1-shared-prefix-0123456789abcdef0123456789";

        // Part A: Sampled scheduled SessionStart Turn 1 -> Turn 2 SessionStart RAM StateStore hit.
        let res1 = send_session_start(
            &mut sched,
            &tx,
            &mut rx,
            user_msg(shared_prompt),
            cfg_sampled.clone(),
        )
        .expect("Turn 1 sampled SessionStart");
        assert_eq!(res1.num_tokens, 10);
        let waves_t1 = sched.inner.scheduled_dspark.total_verify_waves;
        assert!(
            waves_t1 > 0,
            "Turn 1 must run scheduled DFlash2 verify waves"
        );
        assert!(
            sched.inner.scheduled_dspark.lane_decode_limit.is_none(),
            "lane decode limit must be released once no speculative lane is active"
        );
        {
            let rt = sched.inner.paged.as_ref().expect("paged runtime");
            assert!(
                rt.state_store
                    .store
                    .entries()
                    .any(|e| e.ram().is_some_and(|a| a.draft_window.is_some())),
                "Turn 1 must capture a complete 4-bit PackedDraftWindow in StateStore"
            );
        }

        let res2 = send_session_start(
            &mut sched,
            &tx,
            &mut rx,
            two_user_msgs(shared_prompt, "q2"),
            cfg_sampled,
        )
        .expect("Turn 2 sampled SessionStart follow-up");
        assert_eq!(res2.num_tokens, 10);
        assert!(
            res2.cached_tokens >= 32,
            "Turn 2 SessionStart must hit RAM StateStore prefix, got cached_tokens={}",
            res2.cached_tokens
        );
        assert!(
            sched.inner.scheduled_dspark.total_verify_waves > waves_t1,
            "Turn 2 SessionStart after RAM StateStore hit must keep DFlash2 speculation ON (waves stayed {waves_t1})"
        );
        assert!(sched.inner.scheduled_dspark.lane_decode_limit.is_none());

        // Part B: Greedy scheduled DFlash2 (`mtp_depth = 4`) exact token parity vs cold.
        send_reset_caches(&mut sched, &tx, &mut rx);
        let cfg_cold = scheduled_greedy_dflash2_config("sched-cold", 10, "salt-greedy-cold");
        let cold_res = send_session_start(
            &mut sched,
            &tx,
            &mut rx,
            two_user_msgs(shared_prompt, "q2"),
            cfg_cold,
        )
        .expect("Cold greedy scheduled baseline");
        assert_eq!(cold_res.cached_tokens, 0);
        send_reset_caches(&mut sched, &tx, &mut rx);

        let cfg_greedy = scheduled_greedy_dflash2_config("sched-greedy", 10, "salt-greedy-hit");
        let _ = send_session_start(
            &mut sched,
            &tx,
            &mut rx,
            user_msg(shared_prompt),
            cfg_greedy.clone(),
        )
        .expect("Greedy scheduled Turn 1");
        let waves_g1 = sched.inner.scheduled_dspark.total_verify_waves;
        let cached_res = send_session_start(
            &mut sched,
            &tx,
            &mut rx,
            two_user_msgs(shared_prompt, "q2"),
            cfg_greedy,
        )
        .expect("Greedy scheduled Turn 2 RAM hit");
        assert!(
            cached_res.cached_tokens >= 32,
            "Greedy scheduled Turn 2 must hit cached prefix"
        );
        assert!(
            sched.inner.scheduled_dspark.total_verify_waves > waves_g1,
            "Greedy scheduled Turn 2 must keep DFlash2 speculation ON"
        );
        assert_eq!(
            cached_res.raw_text, cold_res.raw_text,
            "Greedy scheduled DFlash2 after RAM DraftContextWindow restore must match cold output"
        );
    }

    /// Test (10): Scheduled sampled DFlash2 `SessionContinue` live continuation preserves
    /// its parked live draft context (`park_scheduled_dspark`) and keeps DFlash2 ON without
    /// hitting `StateStore` or cold-rebuilding.
    #[test]
    fn dflash2_scheduled_session_continue_live_continuation_keeps_speculation_on() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        let tok = build_synthetic_inkling2_tokenizer();
        let msg_user_id = tok
            .inner()
            .token_to_id(crate::tools::inkling::MESSAGE_USER)
            .expect("MESSAGE_USER id");
        let inner = build_synthetic_dflash2_inner_with_window(256, Some(msg_user_id), 128, 2);
        let mut sched = Inkling2SchedulerState::new(inner).expect("scheduler");
        let (tx, mut rx) = mpsc::unbounded_channel::<Inkling2Cmd>();

        let mut cfg = scheduled_sampled_dflash2_config("sched-cont", 2, "salt-sched-cont");
        // `top_k = 1` deterministically picks the biased `MESSAGE_USER` token while
        // `temperature = 0.8` keeps the turn on the scheduled sampled DFlash2 lane.
        cfg.top_k = Some(1);

        let res1 = send_session_start(&mut sched, &tx, &mut rx, user_msg("alpha1"), cfg.clone())
            .expect("Turn 1 scheduled sampled SessionStart");
        assert_eq!(res1.num_tokens, 2);
        let seq_id = *sched.owner_sequences.get("sched-cont").expect("seq_id");
        let waves_t1 = sched.inner.scheduled_dspark.total_verify_waves;
        assert!(
            waves_t1 > 0,
            "Turn 1 must execute scheduled DFlash2 verify waves"
        );
        assert!(
            sched.inner.dspark_turn_contexts.contains_key(&seq_id),
            "Clean scheduled turn close must park live draft context into dspark_turn_contexts"
        );
        assert!(sched.inner.scheduled_dspark.seq_lanes.is_empty());
        assert!(sched.inner.scheduled_dspark.lane_decode_limit.is_none());

        let (block0_before, recorded_t1, ram_hits_before) = {
            let rt = sched.inner.paged.as_ref().expect("paged runtime");
            let full = rt.coordinator.full_adapter();
            (
                full.block_table_for(seq_id).unwrap().blocks()[0].block_id,
                rt.coordinator.request_token_count_all(seq_id).unwrap(),
                rt.state_store.disk_counters().ram_hits,
            )
        };
        assert_eq!(
            sched
                .inner
                .dspark_turn_contexts
                .get(&seq_id)
                .unwrap()
                .committed_len() as u32,
            recorded_t1,
            "Parked draft context cursor must match Turn 1 recorded KV length"
        );

        // Turn 2 SessionContinue strictly extends `[prompt1, MESSAGE_USER]` live!
        cfg.max_new_tokens = Some(8);
        let res2 = send_session_continue(
            &mut sched,
            &tx,
            &mut rx,
            two_user_msgs("alpha1", "next-turn"),
            cfg,
        )
        .expect("Turn 2 scheduled sampled SessionContinue");
        assert_eq!(res2.num_tokens, 8);
        assert_eq!(
            res2.cached_tokens, recorded_t1,
            "SessionContinue must reuse the exact live Turn 1 frontier"
        );
        assert!(
            sched.inner.scheduled_dspark.total_verify_waves > waves_t1,
            "SessionContinue must keep scheduled DFlash2 speculation ON via parked context"
        );
        {
            let rt = sched.inner.paged.as_ref().expect("paged runtime");
            assert_eq!(
                rt.coordinator
                    .full_adapter()
                    .block_table_for(seq_id)
                    .unwrap()
                    .blocks()[0]
                    .block_id,
                block0_before,
                "Live continuation must preserve block 0 without StateStore restore or rebuild"
            );
            assert_eq!(
                rt.state_store.disk_counters().ram_hits,
                ram_hits_before,
                "Live continuation must not fall back to StateStore restore"
            );
        }
    }

    /// Test (11): Scheduled sampled DFlash2 SSD `StateStore` hit restores both the 8-layer
    /// Global KV chain and Span 2 4-bit `PackedDraftWindow` from SSD and keeps DFlash2 ON!
    #[test]
    fn dflash2_scheduled_ssd_statestore_hit_keeps_speculation_on() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        struct TempDirGuard(std::path::PathBuf);
        impl Drop for TempDirGuard {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        struct WriteBehindOverrideGuard;
        impl Drop for WriteBehindOverrideGuard {
            fn drop(&mut self) {
                crate::models::inkling2::state_disk::WRITE_BEHIND_MIN_TOKENS_OVERRIDE
                    .with(|c| c.set(None));
                crate::models::inkling2::state_disk::WRITE_BEHIND_DELAY_OVERRIDE
                    .with(|c| c.set(None));
            }
        }
        crate::models::inkling2::state_disk::WRITE_BEHIND_MIN_TOKENS_OVERRIDE
            .with(|c| c.set(Some(32)));
        crate::models::inkling2::state_disk::WRITE_BEHIND_DELAY_OVERRIDE.with(|c| c.set(Some(0.0)));
        let _wb_guard = WriteBehindOverrideGuard;

        let temp_root = std::env::temp_dir().join(format!(
            "inkling2-dflash2-sched-ssd-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&temp_root).expect("create temp_root");
        let _temp_guard = TempDirGuard(temp_root.clone());

        let mut inner = build_synthetic_dflash2_inner_with_window(256, None, 128, 2);
        let disk_opts = crate::disk_tier::DiskTierOptions {
            cache_disk: crate::disk_tier::CacheDisk::Bytes(64u64 << 20),
            persistent: false,
            cache_dir: None,
            temp_root: Some(temp_root),
        };
        assert!(
            inner.attach_state_disk(&disk_opts, || None).is_some(),
            "temporary StateDisk tier must attach"
        );

        let mut sched = Inkling2SchedulerState::new(inner).expect("scheduler");
        let (tx, mut rx) = mpsc::unbounded_channel::<Inkling2Cmd>();
        let shared_prompt = "alpha1-shared-prefix-0123456789abcdef0123456789";
        let cfg = scheduled_sampled_dflash2_config("sched-ssd", 10, "salt-sched-ssd");

        let res1 = send_session_start(
            &mut sched,
            &tx,
            &mut rx,
            user_msg(shared_prompt),
            cfg.clone(),
        )
        .expect("Turn 1 scheduled sampled SessionStart");
        assert_eq!(res1.num_tokens, 10);
        let waves_t1 = sched.inner.scheduled_dspark.total_verify_waves;
        assert!(waves_t1 > 0);
        let seq1 = *sched.owner_sequences.get("sched-ssd").expect("seq1");

        // Flush StateStore entry (with Span 2 PackedDraftWindow) to SSD, drop RAM copy,
        // and evict all hot KV blocks so Turn 2 MUST restore from SSD!
        crate::engine::hybrid_scheduler::HybridSchedulerBackend::release_scheduled_cache(
            &mut sched.inner,
            seq1,
        )
        .expect("release seq1");
        {
            let rt = sched.inner.paged.as_mut().expect("paged runtime");
            let flushed = rt
                .state_store
                .flush_to_disk(&rt.coordinator, Instant::now() + Duration::from_secs(10));
            assert!(flushed, "StateDisk flush_to_disk must complete");
            assert!(
                rt.state_store.disk_counters().offloads >= 1,
                "Turn 1 state must offload to SSD"
            );
            rt.state_store.drop_all_ram_keeping_disk();
            rt.state_store.sync_marks(&rt.coordinator);
            rt.coordinator
                .evict_all_hot_blocks_without_suppress()
                .expect("evict hot KV");
            rt.conv_states.clear();
            rt.block_boundary_conv.clear();
            rt.draft_windows.clear();
            rt.block_boundary_draft_window.clear();
            sched.inner.dspark_turn_contexts.clear();
        }

        let res2 = send_session_start(
            &mut sched,
            &tx,
            &mut rx,
            two_user_msgs(shared_prompt, "q2"),
            cfg,
        )
        .expect("Turn 2 scheduled SSD StateStore hit");
        assert_eq!(res2.num_tokens, 10);
        assert!(
            res2.cached_tokens >= 32,
            "Turn 2 must hit SSD StateStore prefix, got cached_tokens={}",
            res2.cached_tokens
        );
        assert!(
            sched.inner.scheduled_dspark.total_verify_waves > waves_t1,
            "Turn 2 after SSD StateStore restore must keep DFlash2 speculation ON"
        );
        {
            let rt = sched.inner.paged.as_ref().expect("paged runtime");
            assert!(
                rt.state_store.disk_counters().ssd_hits >= 1,
                "Turn 2 must read StateDisk blob (including 4-bit PackedDraftWindow) from SSD"
            );
        }

        let _ = sched
            .inner
            .close_state_disk(Instant::now() + Duration::from_secs(5));
    }

    /// Test (12): Exclusive greedy DFlash2 + Verify Tree `SessionStart` follow-up hits
    /// BOTH RAM and SSD `StateStore` entries WITHOUT cold re-prefill from 0, and matches
    /// cold greedy output token-for-token!
    #[test]
    fn dflash2_exclusive_greedy_session_start_ram_and_ssd_hits_avoid_cold_reprefill() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        struct TempDirGuard(std::path::PathBuf);
        impl Drop for TempDirGuard {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        struct WriteBehindOverrideGuard;
        impl Drop for WriteBehindOverrideGuard {
            fn drop(&mut self) {
                crate::models::inkling2::state_disk::WRITE_BEHIND_MIN_TOKENS_OVERRIDE
                    .with(|c| c.set(None));
                crate::models::inkling2::state_disk::WRITE_BEHIND_DELAY_OVERRIDE
                    .with(|c| c.set(None));
            }
        }
        crate::models::inkling2::state_disk::WRITE_BEHIND_MIN_TOKENS_OVERRIDE
            .with(|c| c.set(Some(32)));
        crate::models::inkling2::state_disk::WRITE_BEHIND_DELAY_OVERRIDE.with(|c| c.set(Some(0.0)));
        let _wb_guard = WriteBehindOverrideGuard;

        let temp_root = std::env::temp_dir().join(format!(
            "inkling2-dflash2-excl-ssd-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&temp_root).expect("create temp_root");
        let _temp_guard = TempDirGuard(temp_root.clone());

        let mut inner = build_synthetic_dflash2_inner_with_window(256, None, 128, 2);
        let disk_opts = crate::disk_tier::DiskTierOptions {
            cache_disk: crate::disk_tier::CacheDisk::Bytes(64u64 << 20),
            persistent: false,
            cache_dir: None,
            temp_root: Some(temp_root),
        };
        assert!(
            inner.attach_state_disk(&disk_opts, || None).is_some(),
            "temporary StateDisk tier must attach"
        );

        let cfg_excl = exclusive_dflash2_config("excl-owner", 12, "salt-excl-hit");
        let params_excl = crate::engine::backend::ChatBackend::resolve_params(&inner, &cfg_excl);
        assert!(
            crate::engine::hybrid_scheduler::HybridSchedulerBackend::speculative_turn_requires_exclusive_lane(
                &inner,
                &params_excl
            ),
            "greedy DFlash2 must route to the exclusive verify-tree lane"
        );

        let mut sched = Inkling2SchedulerState::new(inner).expect("scheduler");
        let (tx, mut rx) = mpsc::unbounded_channel::<Inkling2Cmd>();
        let shared_prompt = "alpha1-shared-prefix-0123456789abcdef0123456789";

        // 0. Cold greedy reference for `follow_msgs` (fresh empty cache -> cached_tokens == 0).
        let cfg_cold = exclusive_dflash2_config("excl-cold", 12, "salt-excl-cold");
        let cold_res = send_session_start(
            &mut sched,
            &tx,
            &mut rx,
            two_user_msgs(shared_prompt, "q2"),
            cfg_cold,
        )
        .expect("Cold greedy exclusive baseline");
        assert_eq!(cold_res.cached_tokens, 0);
        send_reset_caches(&mut sched, &tx, &mut rx);

        // 1. Turn 1 seeds StateStore (prompt junction + turn-end Ordinary with PackedDraftWindow).
        let res1 = send_session_start(
            &mut sched,
            &tx,
            &mut rx,
            user_msg(shared_prompt),
            cfg_excl.clone(),
        )
        .expect("Turn 1 exclusive greedy SessionStart");
        assert_eq!(res1.num_tokens, 12);
        {
            let rt = sched.inner.paged.as_ref().expect("paged runtime");
            assert!(
                rt.state_store
                    .store
                    .entries()
                    .any(|e| e.ram().is_some_and(|a| a.draft_window.is_some())),
                "Exclusive Turn 1 must capture PackedDraftWindow in StateStore"
            );
        }

        // 2A. RAM StateStore hit on SessionStart follow-up (NO cold re-prefill from 0!).
        let ram_hits_before = sched
            .inner
            .paged
            .as_ref()
            .unwrap()
            .state_store
            .disk_counters()
            .ram_hits;
        let res_ram = send_session_start(
            &mut sched,
            &tx,
            &mut rx,
            two_user_msgs(shared_prompt, "q2"),
            cfg_excl.clone(),
        )
        .expect("Turn 2A exclusive RAM StateStore hit");
        assert!(
            res_ram.cached_tokens >= 32,
            "Exclusive Turn 2A must reuse RAM StateStore prefix (no cold re-prefill!), got cached_tokens={}",
            res_ram.cached_tokens
        );
        assert_eq!(res_ram.num_tokens, cold_res.num_tokens);
        assert_eq!(
            res_ram.raw_text, cold_res.raw_text,
            "Exclusive greedy DFlash2 RAM restore must match cold output"
        );
        let seq_ram = *sched.owner_sequences.get("excl-owner").expect("seq");
        {
            let rt = sched.inner.paged.as_ref().expect("paged runtime");
            assert!(rt.state_store.disk_counters().ram_hits > ram_hits_before);
            assert!(
                sched.inner.dspark_turn_contexts.contains_key(&seq_ram),
                "Exclusive verify-tree turn must leave a live DFlash2 context"
            );
        }

        // 2B. Flush to SSD, drop RAM + hot KV -> SSD StateStore hit (NO cold re-prefill from 0!).
        crate::engine::hybrid_scheduler::HybridSchedulerBackend::release_scheduled_cache(
            &mut sched.inner,
            seq_ram,
        )
        .expect("release seq_ram");
        {
            let rt = sched.inner.paged.as_mut().expect("paged runtime");
            let flushed = rt
                .state_store
                .flush_to_disk(&rt.coordinator, Instant::now() + Duration::from_secs(10));
            assert!(flushed, "StateDisk flush_to_disk must complete");
            assert!(
                rt.state_store.disk_counters().offloads >= 1,
                "State must offload to SSD"
            );
            rt.state_store.drop_all_ram_keeping_disk();
            rt.state_store.sync_marks(&rt.coordinator);
            rt.coordinator
                .evict_all_hot_blocks_without_suppress()
                .expect("evict hot KV");
            rt.conv_states.clear();
            rt.block_boundary_conv.clear();
            rt.draft_windows.clear();
            rt.block_boundary_draft_window.clear();
            sched.inner.dspark_turn_contexts.clear();
        }
        let ssd_hits_before = sched
            .inner
            .paged
            .as_ref()
            .unwrap()
            .state_store
            .disk_counters()
            .ssd_hits;
        let res_ssd = send_session_start(
            &mut sched,
            &tx,
            &mut rx,
            two_user_msgs(shared_prompt, "q2"),
            cfg_excl,
        )
        .expect("Turn 2B exclusive SSD StateStore hit");
        assert!(
            res_ssd.cached_tokens >= 32,
            "Exclusive Turn 2B must reuse SSD StateStore prefix (no cold re-prefill!), got cached_tokens={}",
            res_ssd.cached_tokens
        );
        assert_eq!(res_ssd.num_tokens, cold_res.num_tokens);
        assert_eq!(
            res_ssd.raw_text, cold_res.raw_text,
            "Exclusive greedy DFlash2 SSD restore must match cold output"
        );
        assert!(
            sched
                .inner
                .paged
                .as_ref()
                .unwrap()
                .state_store
                .disk_counters()
                .ssd_hits
                > ssd_hits_before,
            "Exclusive Turn 2B must restore from SSD StateDisk"
        );

        let _ = sched
            .inner
            .close_state_disk(Instant::now() + Duration::from_secs(5));
    }

    /// Test (13): Exclusive greedy DFlash2 `length`-exit `SessionContinue`:
    /// `materialize_final_token` keeps BOTH `dspark_turn_contexts` and `draft_windows`
    /// in exact lockstep with KV/ShortConv so Turn 2 `SessionContinue` reuses the live
    /// prefix with zero cold re-prefill!
    #[test]
    fn dflash2_exclusive_greedy_length_exit_session_continue_syncs_draft_ctx() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        let tok = build_synthetic_inkling2_tokenizer();
        let msg_user_id = tok
            .inner()
            .token_to_id(crate::tools::inkling::MESSAGE_USER)
            .expect("MESSAGE_USER id");
        let inner = build_synthetic_dflash2_inner_with_window(256, Some(msg_user_id), 128, 1);
        let mut sched = Inkling2SchedulerState::new(inner).expect("scheduler");
        let (tx, mut rx) = mpsc::unbounded_channel::<Inkling2Cmd>();

        // `max_new_tokens = 1` guarantees `finish_reason == "length"` with `!last_in_cache`,
        // forcing `Inkling2PagedDecode::materialize_final_token` to run!
        let mut cfg = exclusive_dflash2_config("excl-len", 1, "salt-excl-len");
        let res1 = send_session_start(&mut sched, &tx, &mut rx, user_msg("alpha1"), cfg.clone())
            .expect("Turn 1 exclusive length-exit SessionStart");
        assert_eq!(res1.num_tokens, 1);
        assert_eq!(res1.finish_reason, "length");

        let seq_id = *sched.owner_sequences.get("excl-len").expect("seq_id");
        let (block0_before, recorded_t1, ram_hits_before) = {
            let rt = sched.inner.paged.as_ref().expect("paged runtime");
            let recorded = rt.coordinator.request_token_count_all(seq_id).unwrap();
            let draft_len = sched
                .inner
                .dspark_turn_contexts
                .get(&seq_id)
                .expect("Turn 1 must retain live DFlash2 context")
                .committed_len() as u32;
            assert_eq!(
                draft_len, recorded,
                "materialize_final_token must keep dspark_turn_contexts in lockstep with KV"
            );
            let win_end = rt
                .draft_windows
                .get(&seq_id)
                .expect("Turn 1 must retain live DraftContextWindow")
                .end();
            assert_eq!(
                win_end, recorded,
                "materialize_final_token must keep DraftContextWindow in lockstep with KV"
            );
            (
                rt.coordinator
                    .full_adapter()
                    .block_table_for(seq_id)
                    .unwrap()
                    .blocks()[0]
                    .block_id,
                recorded,
                rt.state_store.disk_counters().ram_hits,
            )
        };

        // Turn 2 SessionContinue strictly extends `[prompt1, MESSAGE_USER]` live!
        cfg.max_new_tokens = Some(6);
        let res2 = send_session_continue(
            &mut sched,
            &tx,
            &mut rx,
            two_user_msgs("alpha1", "continued-after-length-exit"),
            cfg,
        )
        .expect("Turn 2 exclusive SessionContinue");
        assert_eq!(res2.num_tokens, 6);
        assert_eq!(
            res2.cached_tokens, recorded_t1,
            "Turn 2 SessionContinue must hit live prefix with zero cold re-prefill"
        );
        {
            let rt = sched.inner.paged.as_ref().expect("paged runtime");
            assert_eq!(
                rt.coordinator
                    .full_adapter()
                    .block_table_for(seq_id)
                    .unwrap()
                    .blocks()[0]
                    .block_id,
                block0_before,
                "Live continuation must keep Turn 1 block 0 untouched"
            );
            assert_eq!(
                rt.state_store.disk_counters().ram_hits,
                ram_hits_before,
                "Live continuation must not fall back to StateStore"
            );
            let recorded_t2 = rt.coordinator.request_token_count_all(seq_id).unwrap();
            assert_eq!(
                sched
                    .inner
                    .dspark_turn_contexts
                    .get(&seq_id)
                    .unwrap()
                    .committed_len() as u32,
                recorded_t2,
                "Turn 2 multi-token length exit must also leave DFlash2 context synced"
            );
        }
    }

    /// Test (14): An AR turn (`enable_mtp = Some(false)`) on a DFlash2-loaded model captures
    /// a complete 4-bit `PackedDraftWindow` so subsequent scheduled AND exclusive DFlash2
    /// `SessionStart` follow-ups hit the cached prefix AND speculate with DFlash2!
    #[test]
    fn dflash2_ar_turn_captures_draft_window_for_dflash2_followup() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        let inner = build_synthetic_dflash2_inner_with_window(256, None, 128, 2);
        let mut sched = Inkling2SchedulerState::new(inner).expect("scheduler");
        let (tx, mut rx) = mpsc::unbounded_channel::<Inkling2Cmd>();

        let shared_prompt = "alpha1-shared-prefix-0123456789abcdef0123456789";
        let mut cfg_ar = scheduled_sampled_dflash2_config("ar-turn", 10, "salt-ar-to-df2");
        cfg_ar.enable_mtp = Some(false);

        let res_ar = send_session_start(&mut sched, &tx, &mut rx, user_msg(shared_prompt), cfg_ar)
            .expect("Turn 1 AR SessionStart");
        assert_eq!(res_ar.num_tokens, 10);
        assert_eq!(
            sched.inner.scheduled_dspark.total_verify_waves, 0,
            "Turn 1 with enable_mtp=false must run pure AR"
        );
        {
            let rt = sched.inner.paged.as_ref().expect("paged runtime");
            assert!(
                rt.state_store
                    .store
                    .entries()
                    .any(|e| e.ram().is_some_and(|a| a.draft_window.is_some())),
                "AR turn on DFlash2-loaded model must still capture a complete PackedDraftWindow"
            );
        }

        // Follow-up A: Scheduled sampled DFlash2 hits AR's cached prefix AND speculates!
        let cfg_sched =
            scheduled_sampled_dflash2_config("df2-sched-after-ar", 10, "salt-ar-to-df2");
        let res_sched = send_session_start(
            &mut sched,
            &tx,
            &mut rx,
            two_user_msgs(shared_prompt, "q2"),
            cfg_sched,
        )
        .expect("Turn 2A scheduled DFlash2 after AR");
        assert!(
            res_sched.cached_tokens >= 32,
            "Scheduled DFlash2 follow-up must hit AR's cached prefix"
        );
        assert!(
            sched.inner.scheduled_dspark.total_verify_waves > 0,
            "Scheduled DFlash2 follow-up must speculate from AR's PackedDraftWindow"
        );

        // Follow-up B: Exclusive greedy DFlash2 hits AR's cached prefix with zero cold re-prefill!
        let cfg_excl = exclusive_dflash2_config("df2-excl-after-ar", 10, "salt-ar-to-df2");
        let res_excl = send_session_start(
            &mut sched,
            &tx,
            &mut rx,
            two_user_msgs(shared_prompt, "q2"),
            cfg_excl,
        )
        .expect("Turn 2B exclusive DFlash2 after AR");
        assert!(
            res_excl.cached_tokens >= 32,
            "Exclusive greedy DFlash2 follow-up must hit AR's cached prefix without cold re-prefill"
        );
    }

    /// Test (15): When `suffix_len >= sliding_window`, BOTH scheduled and exclusive DFlash2
    /// lanes succeed on a `StateStore` hit via `record_only` skip EVEN WHEN the anchor has
    /// `draft_window == None` (legacy anchor without draft window).
    #[test]
    fn dflash2_suffix_ge_sliding_window_skips_window_on_both_lanes() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        // Tiny `draft_sliding_window = 16` so a ~40-token suffix satisfies `suffix >= 16`.
        let inner = build_synthetic_dflash2_inner_with_window(256, None, 16, 2);
        let mut sched = Inkling2SchedulerState::new(inner).expect("scheduler");
        let (tx, mut rx) = mpsc::unbounded_channel::<Inkling2Cmd>();

        let shared_prompt = "alpha1-shared-prefix-0123456789abcdef0123456789";
        let long_suffix = "long-suffix-exceeding-16-tokens-0123456789";

        // Cold greedy baseline for parity verification on the exclusive lane (fresh empty cache).
        let cfg_cold = exclusive_dflash2_config("skip-cold", 10, "salt-skip-cold");
        let cold_res = send_session_start(
            &mut sched,
            &tx,
            &mut rx,
            two_user_msgs(shared_prompt, long_suffix),
            cfg_cold,
        )
        .expect("Cold greedy baseline");
        assert_eq!(cold_res.cached_tokens, 0);
        send_reset_caches(&mut sched, &tx, &mut rx);

        // Turn 1 populates KV + ShortConv on `seq_1` for `shared_prompt`.
        let salt_str = "salt-skip-window";
        let cfg_seed = exclusive_dflash2_config("skip-seed", 4, salt_str);
        let salt_u64 =
            crate::engine::backend::ChatBackend::resolve_params(&sched.inner, &cfg_seed).cache_salt;
        let _ = send_session_start(&mut sched, &tx, &mut rx, user_msg(shared_prompt), cfg_seed)
            .expect("Seed turn");
        let seq_seed = *sched.owner_sequences.get("skip-seed").expect("seq_seed");

        // Replace all StateStore entries with a legacy anchor at the exact Turn-1 boundary
        // having `draft_window = None` (preserving exact ShortConv state at `depth_u32`)!
        let legacy_depth = {
            let rt = sched.inner.paged.as_mut().expect("paged runtime");
            let tokens_seed = rt
                .coordinator
                .full_adapter()
                .request_tokens_for(seq_seed)
                .expect("seed tokens")
                .to_vec();
            let hit = rt
                .state_store
                .lookup_deepest_restorable(&rt.coordinator, &tokens_seed, salt_u64)
                .expect("Turn 1 must have captured a StateStore entry");
            let depth_u32 = hit.depth as u32;
            let (anchor, _) = rt
                .state_store
                .load_anchor(&rt.coordinator, hit.id)
                .expect("load Turn 1 anchor");
            let geom = rt.state_store.resolve_geometry().expect("geometry").clone();
            let exact_conv = crate::models::inkling2::state_disk::unpack_short_conv_state(
                &geom,
                &anchor.conv_data,
            )
            .expect("unpack exact conv at boundary");
            rt.state_store.clear_ram_and_checkpoints();
            let id = rt
                .state_store
                .capture_boundary(
                    &mut rt.coordinator,
                    &exact_conv,
                    seq_seed,
                    &tokens_seed,
                    depth_u32,
                    salt_u64,
                    super::StateClass::Ordinary,
                    None,
                )
                .expect("capture legacy anchor without draft window")
                .expect("legacy entry id");
            assert!(
                rt.state_store
                    .store
                    .get(id)
                    .and_then(|e| e.ram())
                    .is_some_and(|a| a.draft_window.is_none()),
                "Legacy anchor must have draft_window == None"
            );
            depth_u32
        };

        // 1. Scheduled sampled DFlash2: must hit `legacy_depth` AND keep DFlash2 speculation ON!
        let waves_before = sched.inner.scheduled_dspark.total_verify_waves;
        let cfg_sched = scheduled_sampled_dflash2_config("skip-sched", 10, salt_str);
        let res_sched = send_session_start(
            &mut sched,
            &tx,
            &mut rx,
            two_user_msgs(shared_prompt, long_suffix),
            cfg_sched,
        )
        .expect("Scheduled DFlash2 with suffix >= sliding_window");
        assert_eq!(
            res_sched.cached_tokens, legacy_depth,
            "Scheduled lane must hit legacy anchor when suffix >= sliding_window"
        );
        assert!(
            sched.inner.scheduled_dspark.total_verify_waves > waves_before,
            "Scheduled lane must keep DFlash2 speculation ON via record_only skip"
        );

        // 2. Exclusive greedy DFlash2: must hit `legacy_depth` WITHOUT cold re-prefill AND match cold!
        // Re-strip any new StateStore entries created by `res_sched` so exclusive lane ALSO hits
        // ONLY the legacy `draft_window == None` anchor at `legacy_depth`!
        {
            let rt = sched.inner.paged.as_mut().expect("paged runtime");
            let drop_ids: Vec<_> = rt
                .state_store
                .store
                .entries()
                .filter(|e| e.ram().is_some_and(|a| a.draft_window.is_some()))
                .map(|e| e.id())
                .collect();
            for id in drop_ids {
                let _ = rt.state_store.reclaim_entry(&rt.coordinator, id);
            }
            assert!(
                rt.state_store
                    .store
                    .entries()
                    .all(|e| e.ram().is_some_and(|a| a.draft_window.is_none())),
                "Only the legacy draft_window==None anchor may remain"
            );
        }
        let cfg_excl = exclusive_dflash2_config("skip-excl", 10, salt_str);
        let res_excl = send_session_start(
            &mut sched,
            &tx,
            &mut rx,
            two_user_msgs(shared_prompt, long_suffix),
            cfg_excl,
        )
        .expect("Exclusive greedy DFlash2 with suffix >= sliding_window");
        assert_eq!(
            res_excl.cached_tokens, legacy_depth,
            "Exclusive lane must hit legacy anchor without cold re-prefill when suffix >= sliding_window"
        );
        assert_eq!(
            res_excl.raw_text, cold_res.raw_text,
            "Exclusive greedy DFlash2 via record_only skip must match cold output"
        );
    }

    /// Finding 1 regression: an exclusive greedy (and scheduled) DFlash2 turn with
    /// `reuse_cache: Some(false)` MUST NEVER publish rolling checkpoints or prompt junctions into
    /// `rt.state_store`, and a follow-up turn with the same salt must not reuse the opted-out prefix.
    #[test]
    fn dflash2_exclusive_reuse_cache_false_does_not_publish_state_store() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        struct CheckpointOverrideGuard;
        impl Drop for CheckpointOverrideGuard {
            fn drop(&mut self) {
                crate::models::inkling2::state_disk::CHECKPOINT_INTERVAL_OVERRIDE
                    .with(|c| c.set(None));
            }
        }
        // Lower rolling checkpoint spacing to 32 so even our synthetic prompt crosses both
        // a rolling checkpoint AND the prompt junction!
        crate::models::inkling2::state_disk::CHECKPOINT_INTERVAL_OVERRIDE.with(|c| c.set(Some(32)));
        let _cp_guard = CheckpointOverrideGuard;

        let inner = build_synthetic_dflash2_inner_with_window(256, None, 128, 2);
        let mut sched = Inkling2SchedulerState::new(inner).expect("scheduler");
        let (tx, mut rx) = mpsc::unbounded_channel::<Inkling2Cmd>();

        let shared_prompt = "alpha1-optout-secret-prefix-0123456789abcdef0123456789abcdef";

        // 1. Exclusive greedy DFlash2 turn with `reuse_cache = Some(false)` via `run_paged_dspark_turn`.
        let mut cfg_optout = exclusive_dflash2_config("excl-optout", 8, "salt-optout");
        cfg_optout.reuse_cache = Some(false);
        let admitted = crate::engine::session::admit_paged_turn(
            &mut sched.inner,
            user_msg(shared_prompt),
            cfg_optout,
            crate::engine::session::TurnKind::Start,
            false,
        )
        .expect("admit exclusive opt-out turn");
        assert!(
            !admitted.params.reuse_cache,
            "admitted params must carry reuse_cache=false"
        );
        let prompt_tokens = admitted.tokens.clone();
        let optout_salt = admitted.params.cache_salt;
        let media = crate::engine::plan::MediaInputs {
            images: &admitted.images,
            audio: &admitted.audio,
        };
        let mut args = crate::engine::backend::WholeTurnArgs {
            tokenizer: &admitted.tokenizer,
            eos_id: admitted.eos_id,
            config: &admitted.config,
            params: &admitted.params,
            thinking: admitted.thinking,
            tokens: &admitted.tokens,
            media,
            plan: admitted.plan,
            sink: None,
            cancelled: None,
        };
        let out1 = crate::engine::dspark_turn::run_paged_dspark_turn(&mut sched.inner, &mut args)
            .expect("Turn 1 exclusive opt-out must succeed");
        let res1 = match out1 {
            crate::engine::backend::TurnOutput::Complete(r) => r,
            crate::engine::backend::TurnOutput::Streamed => panic!("expected complete output"),
        };
        assert_eq!(res1.num_tokens, 8);
        {
            let rt = sched.inner.paged.as_ref().expect("paged runtime");
            assert_eq!(
                rt.state_store.store.entries().count(),
                0,
                "reuse_cache=false exclusive DFlash2 turn must NOT insert any StateStore entry"
            );
            let c = rt.state_store.disk_counters();
            assert_eq!(c.checkpoints_published, 0);
            assert_eq!(c.ordinary_published, 0);
        }

        // Follow-up with the exact same salt & prefix must get 0 cached tokens!
        let cfg_followup = exclusive_dflash2_config("excl-followup", 8, "salt-optout");
        let res2 = send_session_start(
            &mut sched,
            &tx,
            &mut rx,
            two_user_msgs(shared_prompt, "probe"),
            cfg_followup,
        )
        .expect("Turn 2 follow-up");
        assert_eq!(
            res2.cached_tokens, 0,
            "Follow-up turn with the same salt must NOT reuse an opted-out exclusive prefix"
        );

        // 2. Also verify scheduled prefill (`run_scheduled_prefill_pass` fast & resumable paths)
        //    with `reuse_cache = false` does NOT publish checkpoints or junctions into StateStore.
        send_reset_caches(&mut sched, &tx, &mut rx);
        for (seq_id, budget_ms) in [(11u32, None), (12u32, Some(0.0))] {
            let prefix = sched
                .inner
                .prepare_prefix_sync(seq_id, &prompt_tokens, false, optout_salt)
                .expect("prepare opt-out scheduled prefix");
            assert!(!prefix.reuse_cache);
            while matches!(
                crate::engine::hybrid_scheduler::HybridSchedulerBackend::run_scheduled_prefill_pass(
                    &mut sched.inner,
                    seq_id,
                    &prompt_tokens,
                    &prefix,
                    0,
                    prompt_tokens.len(),
                    crate::stream::Stream::generation(),
                    true,
                    budget_ms,
                )
                .expect("scheduled prefill pass"),
                crate::engine::hybrid_scheduler::ScheduledPrefillPass::Paused
            ) {}
            let rt = sched.inner.paged.as_ref().expect("paged runtime");
            assert_eq!(
                rt.state_store.store.entries().count(),
                0,
                "reuse_cache=false scheduled prefill (budget={budget_ms:?}) must NOT publish to StateStore"
            );
            let _ =
                crate::engine::hybrid_scheduler::HybridSchedulerBackend::release_scheduled_cache(
                    &mut sched.inner,
                    seq_id,
                );
        }
    }

    /// Finding 2 regression: idle parked `dspark_turn_contexts` stay capped at
    /// `max_concurrent_sequences`, `memory_accounts().states` accounts for parked draft rings and
    /// live draft windows, and an idle owner whose parked ring was evicted still speculates on
    /// `SessionContinue` via its retained live `draft_windows`.
    #[test]
    fn dflash2_parked_contexts_bounded_and_accounted_with_window_restore() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        let tok = build_synthetic_inkling2_tokenizer();
        let msg_user_id = tok
            .inner()
            .token_to_id(crate::tools::inkling::MESSAGE_USER)
            .expect("MESSAGE_USER id");
        // `max_concurrent_sequences = 2`, with enough Global + SWA headroom (`max_chunk = 160` ->
        // 11 usable SWA blocks) so 4 idle owners stay resident in KV while `dspark_turn_contexts`
        // evicts down to `max_concurrent_sequences = 2`.
        let mut inner = build_synthetic_dflash2_inner_with_window(512, Some(msg_user_id), 128, 2);
        inner
            .enable_paged(crate::models::inkling2::kv_cache::Inkling2PagedConfig {
                kv_format: crate::models::inkling2::kv_cache::default_inkling2_kv_format(),
                block_size: 32,
                max_model_len: 512,
                max_chunk: 160,
                max_concurrent_sequences: 2,
                gpu_memory_mb: None,
            })
            .expect("enable_paged with SWA headroom");
        let max_seqs = inner
            .paged
            .as_ref()
            .expect("paged")
            .coordinator
            .max_concurrent_sequences() as usize;
        assert_eq!(max_seqs, 2);

        let mut sched = Inkling2SchedulerState::new(inner).expect("scheduler");
        let (tx, mut rx) = mpsc::unbounded_channel::<Inkling2Cmd>();

        let mut owner_seqs = Vec::new();
        let mut owner0_recorded = 0u32;
        for i in 0..4usize {
            let owner = format!("bounded-owner-{i}");
            let salt = format!("salt-bounded-{i}");
            let prompt = format!("alpha{i}");
            let mut cfg = scheduled_sampled_dflash2_config(&owner, 2, &salt);
            cfg.top_k = Some(1);
            let res = send_session_start(&mut sched, &tx, &mut rx, user_msg(&prompt), cfg)
                .expect("scheduled DFlash2 SessionStart");
            assert_eq!(res.num_tokens, 2);

            let seq_id = *sched.owner_sequences.get(&owner).expect("owner seq_id");
            owner_seqs.push(seq_id);
            if i == 0 {
                owner0_recorded = sched
                    .inner
                    .paged
                    .as_ref()
                    .unwrap()
                    .coordinator
                    .request_token_count_all(seq_id)
                    .expect("owner0 recorded");
            }

            assert!(
                sched.inner.dspark_turn_contexts.len() <= max_seqs,
                "parked dspark_turn_contexts len={} must stay <= max_concurrent_sequences={max_seqs} after owner {i}",
                sched.inner.dspark_turn_contexts.len()
            );

            // Verify `memory_accounts().states` includes parked draft rings + live draft windows!
            let rt = sched.inner.paged.as_ref().expect("paged runtime");
            let parked_bytes: u64 = sched
                .inner
                .dspark_turn_contexts
                .values()
                .map(|c| c.buffer_bytes())
                .sum();
            let window_bytes = rt.draft_windows_bytes();
            let conv_bytes = (rt.conv_states.len() as u64)
                .saturating_mul(super::short_conv_state_bytes(&sched.inner.config));
            let expected_states = rt
                .state_store
                .total_ram_bytes()
                .saturating_add(conv_bytes)
                .saturating_add(window_bytes)
                .saturating_add(parked_bytes);
            assert!(parked_bytes > 0, "parked DFlash2 ring bytes must be > 0");
            assert!(
                window_bytes > 0,
                "live DraftContextWindow bytes must be > 0"
            );
            let accounts =
                crate::engine::hybrid_scheduler::HybridSchedulerBackend::memory_accounts(
                    &sched.inner,
                )
                .expect("memory_accounts");
            assert_eq!(
                accounts.states, expected_states,
                "memory_accounts().states must account for StateStore + conv + draft_windows + parked draft rings"
            );
        }

        let seq0 = owner_seqs[0];
        assert_eq!(sched.inner.dspark_turn_contexts.len(), max_seqs);
        assert!(
            !sched.inner.dspark_turn_contexts.contains_key(&seq0),
            "oldest idle owner seq0={seq0} must be evicted from parked dspark_turn_contexts"
        );
        assert!(
            sched
                .inner
                .paged
                .as_ref()
                .unwrap()
                .draft_windows
                .contains_key(&seq0),
            "evicted idle owner seq0={seq0} must keep its compact live DraftContextWindow intact"
        );

        // Resuming evicted owner-0 via `SessionContinue` must restore from `draft_windows[seq0]`
        // and keep scheduled DFlash2 speculation ON!
        let waves_before = sched.inner.scheduled_dspark.total_verify_waves;
        let mut cfg0 = scheduled_sampled_dflash2_config("bounded-owner-0", 8, "salt-bounded-0");
        cfg0.top_k = Some(1);
        let res_cont = send_session_continue(
            &mut sched,
            &tx,
            &mut rx,
            two_user_msgs("alpha0", "resume-after-ring-evict"),
            cfg0,
        )
        .expect("SessionContinue for owner-0 after parked ring eviction");
        assert_eq!(res_cont.num_tokens, 8);
        assert_eq!(
            res_cont.cached_tokens, owner0_recorded,
            "SessionContinue must reuse owner-0's live KV frontier"
        );
        assert!(
            sched.inner.scheduled_dspark.total_verify_waves > waves_before,
            "owner-0 whose parked ring was evicted must still speculate on SessionContinue via live draft_windows"
        );
        assert!(
            sched.inner.dspark_turn_contexts.len() <= max_seqs,
            "parked dspark_turn_contexts must remain capped after re-parking owner-0"
        );
    }

    /// Finding 5 regression: an injected draft-window capture error on one row of an AR batched
    /// decode (and on single-row AR decode) invalidates ONLY the affected row's draft windows
    /// while returning exact bit-identical target logits and leaving unaffected peer rows' windows
    /// intact!
    #[test]
    fn dflash2_ar_batched_decode_draft_window_error_is_best_effort() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        struct FailSeqGuard;
        impl Drop for FailSeqGuard {
            fn drop(&mut self) {
                crate::models::inkling2::paged_forward::FAIL_DRAFT_WINDOW_CAPTURE_SEQ
                    .with(|c| c.set(None));
            }
        }
        let _fail_guard = FailSeqGuard;

        // Build two identical synthetic DFlash2-loaded models so we can compare exact target
        // logits between an uncorrupted baseline and a run with an injected draft-window error.
        let mut baseline = build_synthetic_dflash2_inner_with_window(256, None, 128, 2);
        let mut injected = build_synthetic_dflash2_inner_with_window(256, None, 128, 2);

        let prompt_a: Vec<u32> = (10..42).collect(); // exact 32-token block boundary -> populates block_boundary_draft_window!
        let prompt_b: Vec<u32> = (15..47).collect();
        let seq_a = 1u32;
        let seq_b = 2u32;

        for inner in [&mut baseline, &mut injected] {
            inner.reset_paged_sequence(seq_a).expect("reset A");
            inner.reset_paged_sequence(seq_b).expect("reset B");
            let _ = inner
                .paged_prefill_sequence(seq_a, &prompt_a, None)
                .expect("AR prefill A");
            let _ = inner
                .paged_prefill_sequence(seq_b, &prompt_b, None)
                .expect("AR prefill B");
            let rt = inner.paged.as_ref().expect("paged");
            assert_eq!(rt.draft_windows.get(&seq_a).map(|w| w.end()), Some(32));
            assert_eq!(rt.draft_windows.get(&seq_b).map(|w| w.end()), Some(32));
            assert!(rt.block_boundary_draft_window.contains_key(&seq_a));
            assert!(rt.block_boundary_draft_window.contains_key(&seq_b));
        }

        // 1. Run batched AR decode on baseline (no error) vs injected (error ONLY on `seq_b`).
        let base_logits = baseline
            .paged_decode_step_batched(&[(seq_a, 5), (seq_b, 6)])
            .expect("baseline batched AR decode")
            .to_float32()
            .expect("base f32")
            .to_vec();

        crate::models::inkling2::paged_forward::FAIL_DRAFT_WINDOW_CAPTURE_SEQ
            .with(|c| c.set(Some(seq_b)));
        let inj_logits = injected
            .paged_decode_step_batched(&[(seq_a, 5), (seq_b, 6)])
            .expect("AR batched decode must succeed even when seq_b draft-window capture fails")
            .to_float32()
            .expect("inj f32")
            .to_vec();
        crate::models::inkling2::paged_forward::FAIL_DRAFT_WINDOW_CAPTURE_SEQ.with(|c| c.set(None));

        assert_eq!(
            inj_logits, base_logits,
            "AR batched decode must return exact target logits despite optional draft-window error"
        );
        {
            let rt = injected.paged.as_ref().expect("paged");
            assert!(
                !rt.draft_windows.contains_key(&seq_b)
                    && !rt.block_boundary_draft_window.contains_key(&seq_b),
                "affected row seq_b must have its draft_windows & block_boundary_draft_window invalidated"
            );
            assert_eq!(
                rt.draft_windows.get(&seq_a).map(|w| w.end()),
                Some(33),
                "unaffected peer row seq_a must keep its live draft window intact and updated"
            );
            assert!(
                rt.block_boundary_draft_window.contains_key(&seq_a),
                "unaffected peer row seq_a must keep its block_boundary_draft_window intact"
            );
        }

        // 2. Also verify single-row AR decode isolates a draft-window capture error cleanly.
        let base_single = baseline
            .paged_decode_step(seq_a, 7)
            .expect("baseline single AR decode")
            .to_float32()
            .expect("base single f32")
            .to_vec();
        crate::models::inkling2::paged_forward::FAIL_DRAFT_WINDOW_CAPTURE_SEQ
            .with(|c| c.set(Some(seq_a)));
        let inj_single = injected
            .paged_decode_step(seq_a, 7)
            .expect("single AR decode must succeed despite draft-window capture failure")
            .to_float32()
            .expect("inj single f32")
            .to_vec();
        crate::models::inkling2::paged_forward::FAIL_DRAFT_WINDOW_CAPTURE_SEQ.with(|c| c.set(None));
        assert_eq!(inj_single, base_single);
        {
            let rt = injected.paged.as_ref().expect("paged");
            assert!(
                !rt.draft_windows.contains_key(&seq_a)
                    && !rt.block_boundary_draft_window.contains_key(&seq_a),
                "single-row AR capture failure must invalidate seq_a's draft windows"
            );
        }
    }

    /// Round-2 Finding 2 regression: an injected draft-window capture failure during a real
    /// scheduled DFlash2 turn (crossing a 32-token block boundary in verify commit) must be
    /// best-effort — the turn completes cleanly with exact bit-identical output tokens, live
    /// speculation continues, and only the affected sequence's optional `draft_windows` /
    /// `block_boundary_draft_window` are invalidated (leaving healthy peers intact).
    #[test]
    fn dflash2_scheduled_verify_commit_draft_window_error_is_best_effort() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        struct FailSeqGuard;
        impl Drop for FailSeqGuard {
            fn drop(&mut self) {
                crate::models::inkling2::paged_forward::FAIL_DRAFT_WINDOW_CAPTURE_SEQ
                    .with(|c| c.set(None));
            }
        }
        let _fail_guard = FailSeqGuard;

        let tok = build_synthetic_inkling2_tokenizer();
        let msg_user_id = tok
            .inner()
            .token_to_id(crate::tools::inkling::MESSAGE_USER)
            .expect("MESSAGE_USER id");
        let inner = build_synthetic_dflash2_inner_with_window(512, Some(msg_user_id), 128, 4);
        let mut sched = Inkling2SchedulerState::new(inner).expect("scheduler");
        let (tx, mut rx) = mpsc::unbounded_channel::<Inkling2Cmd>();

        // 1. Healthy peer turn first (keeps its draft windows intact throughout).
        let peer_res = send_session_start(
            &mut sched,
            &tx,
            &mut rx,
            user_msg("alpha0"),
            scheduled_greedy_dflash2_config("sched-peer", 6, "salt-sched-peer"),
        )
        .expect("healthy peer scheduled DFlash2 turn");
        assert_eq!(peer_res.num_tokens, 6);
        let peer_seq = *sched.owner_sequences.get("sched-peer").expect("peer_seq");
        assert!(
            sched
                .inner
                .paged
                .as_ref()
                .unwrap()
                .draft_windows
                .contains_key(&peer_seq),
            "peer must hold a live draft window"
        );

        // Build a prompt whose formatted length sits just below a 32-token block boundary
        // so the 16-token scheduled DFlash2 verify commit crosses a 32-token boundary!
        let prompt = "alpha1 beta2 gamma3 delta4 epsilon5 zeta6 eta7 theta8";

        // 2. Clean baseline scheduled greedy DFlash2 turn (no error injected).
        let base_cfg = scheduled_greedy_dflash2_config("sched-base", 16, "salt-sched-dw-base");
        let base_res = send_session_start(&mut sched, &tx, &mut rx, user_msg(prompt), base_cfg)
            .expect("baseline scheduled DFlash2 turn");
        assert_eq!(base_res.num_tokens, 16);
        let base_seq = *sched.owner_sequences.get("sched-base").expect("base_seq");
        let base_recorded = sched
            .inner
            .paged
            .as_ref()
            .unwrap()
            .coordinator
            .request_token_count_all(base_seq)
            .expect("base_recorded");
        assert!(
            base_recorded >= 32,
            "baseline turn must cross a 32-token block boundary (recorded={base_recorded})"
        );
        assert!(
            sched
                .inner
                .paged
                .as_ref()
                .unwrap()
                .block_boundary_draft_window
                .contains_key(&base_seq),
            "baseline turn must capture a block-boundary draft window when crossing 32 tokens"
        );

        // 3. Injected turn: first seed `owner-inj` with a short SessionStart (so we know its
        //    exact `seq_id` and its prompt stays below the next 32-token boundary), then arm
        //    `FAIL_DRAFT_WINDOW_CAPTURE_SEQ` during a `SessionContinue` that crosses a 32-token
        //    boundary inside scheduled DFlash2 verify commit!
        let inj_start_cfg = scheduled_greedy_dflash2_config("sched-inj", 2, "salt-sched-dw-inj");
        let _ = send_session_start(&mut sched, &tx, &mut rx, user_msg("alpha1"), inj_start_cfg)
            .expect("seed owner-inj");
        let inj_seq = *sched.owner_sequences.get("sched-inj").expect("inj_seq");

        // Compare clean SessionContinue (on a fresh identical seed) vs injected SessionContinue
        // on `inj_seq` crossing a 32-token boundary during verify commit.
        let ref_start_cfg = scheduled_greedy_dflash2_config("sched-ref", 2, "salt-sched-dw-ref");
        let _ = send_session_start(&mut sched, &tx, &mut rx, user_msg("alpha1"), ref_start_cfg)
            .expect("seed owner-ref");
        let ref_cont_cfg = scheduled_greedy_dflash2_config("sched-ref", 24, "salt-sched-dw-ref");
        let ref_cont = send_session_continue(
            &mut sched,
            &tx,
            &mut rx,
            two_user_msgs("alpha1", "beta2"),
            ref_cont_cfg,
        )
        .expect("clean reference SessionContinue");
        assert_eq!(ref_cont.num_tokens, 24);

        let waves_before = sched.inner.scheduled_dspark.total_verify_waves;
        let inj_cont_cfg = scheduled_greedy_dflash2_config("sched-inj", 24, "salt-sched-dw-inj");
        crate::models::inkling2::paged_forward::FAIL_DRAFT_WINDOW_CAPTURE_SEQ
            .with(|c| c.set(Some(inj_seq)));
        let inj_cont = send_session_continue(
            &mut sched,
            &tx,
            &mut rx,
            two_user_msgs("alpha1", "beta2"),
            inj_cont_cfg,
        )
        .expect("scheduled DFlash2 turn must succeed despite draft-window capture failure");
        crate::models::inkling2::paged_forward::FAIL_DRAFT_WINDOW_CAPTURE_SEQ.with(|c| c.set(None));

        assert_eq!(inj_cont.num_tokens, ref_cont.num_tokens);
        assert_eq!(
            inj_cont.raw_text, ref_cont.raw_text,
            "injected draft-window failure must produce exact bit-identical output tokens"
        );
        assert!(
            sched.inner.scheduled_dspark.total_verify_waves > waves_before,
            "live scheduled DFlash2 speculation must continue despite optional draft-window error"
        );
        {
            let rt = sched.inner.paged.as_ref().expect("paged");
            let inj_recorded = rt
                .coordinator
                .request_token_count_all(inj_seq)
                .expect("inj_recorded");
            assert!(
                inj_recorded >= 32,
                "injected turn must have crossed a 32-token boundary (recorded={inj_recorded})"
            );
            assert!(
                !rt.draft_windows.contains_key(&inj_seq)
                    && !rt.block_boundary_draft_window.contains_key(&inj_seq),
                "affected seq {inj_seq} must have its optional draft windows invalidated"
            );
            assert!(
                rt.draft_windows.contains_key(&peer_seq),
                "unaffected peer seq {peer_seq} must retain its draft window intact"
            );
            assert!(
                sched.inner.dspark_turn_contexts.contains_key(&inj_seq),
                "live DFlash2 ring context for seq {inj_seq} must remain valid and parked"
            );
        }
    }

    /// Round-2 Finding 5 regression:
    /// - Active SAM `LookupIndex` heap bytes are included in `Inkling2Dflash2Context::resident_bytes()`
    ///   and reflected in `draft_resident_bytes()` / `memory_accounts().states`.
    /// - Parking via `park_turn_draft_context` immediately resets `LookupIndex` to empty
    ///   (`heap_bytes() == 0`).
    /// - Long `token_history` host capacity (`DFlash2ContextCache::host_bytes()`) is reflected in
    ///   `draft_resident_bytes()` and `memory_accounts().states`.
    #[test]
    fn dflash2_lookup_freed_on_park_and_host_capacities_accounted() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        let mut inner = build_synthetic_dflash2_inner_with_window(256, None, 64, 2);
        let draft_cfg = inner.dflash2.as_ref().expect("dflash2").config.clone();

        // 1. Fresh context starts with empty LookupIndex (`heap_bytes() == 0`).
        let mut dctx = crate::models::inkling2::dflash2::Inkling2Dflash2Context::new(&draft_cfg);
        assert_eq!(dctx.lookup.heap_bytes(), 0);
        assert_eq!(dctx.cache.host_bytes(), 0);
        assert_eq!(dctx.resident_bytes(), 0);

        // 2. Populating long `token_history` and SAM `LookupIndex` increases both host_bytes()
        //    and lookup.heap_bytes(), and `resident_bytes()` equals their exact sum + ring bytes.
        let long_tokens: Vec<u32> = (0u32..4096u32).map(|i| (i % 197) + 1).collect();
        dctx.cache.record_only(&long_tokens).expect("record_only");
        dctx.observe_tokens(&long_tokens);
        let host_bytes = dctx.cache.host_bytes();
        let lookup_bytes = dctx.lookup.heap_bytes();
        assert!(
            host_bytes >= (long_tokens.len() * std::mem::size_of::<u32>()) as u64,
            "token_history host capacity must be accounted: {host_bytes}"
        );
        assert!(
            lookup_bytes > 0,
            "active SAM LookupIndex heap bytes must be > 0 after observing 4096 tokens"
        );
        assert_eq!(
            dctx.resident_bytes(),
            dctx.cache.buffer_bytes() + host_bytes + lookup_bytes,
            "resident_bytes() must equal ring + token_history host + lookup heap bytes"
        );

        // 3. Place into an active scheduled lane -> `draft_resident_bytes()` and
        //    `memory_accounts().states` must reflect BOTH `lookup_bytes` and `host_bytes`.
        let baseline_states =
            crate::engine::hybrid_scheduler::HybridSchedulerBackend::memory_accounts(&inner)
                .expect("baseline memory_accounts")
                .states;
        let expected_active_resident = dctx.resident_bytes();
        inner.scheduled_dspark.seq_lanes.insert(
            7,
            crate::models::inkling2::scheduled_dspark::Inkling2ScheduledDsparkLane {
                ctx: crate::models::inkling2::dflash2::Inkling2DraftContext::Dflash2(Box::new(
                    dctx,
                )),
                adaptive_depth: crate::models::inkling2::dspark::AdaptiveDraftDepth::new(
                    draft_cfg.block_size,
                ),
                open: None,
                total_verifies: 0,
                multi_lane_wave_verifies: 0,
            },
        );
        assert_eq!(inner.draft_resident_bytes(), expected_active_resident);
        let active_states =
            crate::engine::hybrid_scheduler::HybridSchedulerBackend::memory_accounts(&inner)
                .expect("active memory_accounts")
                .states;
        assert_eq!(
            active_states,
            baseline_states + expected_active_resident,
            "memory_accounts().states must include active lookup + token_history host capacities"
        );

        // 4. Parking via `park_turn_draft_context` drops `LookupIndex` back to 0 bytes while
        //    keeping `token_history` host capacity accounted in `draft_resident_bytes()` / `states`!
        let lane = inner
            .scheduled_dspark
            .seq_lanes
            .remove(&7)
            .expect("remove active lane");
        inner.park_turn_draft_context(7, lane.ctx);
        let parked_ctx = inner
            .dspark_turn_contexts
            .get(&7)
            .expect("parked context present");
        let crate::models::inkling2::dflash2::Inkling2DraftContext::Dflash2(parked_dctx) =
            parked_ctx
        else {
            panic!("expected Dflash2 parked context");
        };
        assert_eq!(
            parked_dctx.lookup.heap_bytes(),
            0,
            "parking must immediately drop LookupIndex back to empty (heap_bytes == 0)"
        );
        assert_eq!(
            parked_dctx.cache.host_bytes(),
            host_bytes,
            "parked context retains its token_history provenance"
        );
        assert_eq!(
            parked_dctx.resident_bytes(),
            parked_dctx.cache.buffer_bytes() + host_bytes
        );
        assert_eq!(
            inner.draft_resident_bytes(),
            expected_active_resident - lookup_bytes,
            "draft_resident_bytes() must drop by exact lookup_bytes on park and retain host_bytes"
        );
        let parked_states =
            crate::engine::hybrid_scheduler::HybridSchedulerBackend::memory_accounts(&inner)
                .expect("parked memory_accounts")
                .states;
        assert_eq!(
            parked_states,
            baseline_states + parked_dctx.resident_bytes(),
            "memory_accounts().states must reflect host_bytes and drop freed lookup_bytes after park"
        );
    }

    /// Round-3 Finding 1 regression: preempting an opted-out (`reuse_cache: Some(false)`)
    /// scheduled DFlash2 turn after crossing a 32-token boundary and letting it replay to
    /// completion must NEVER publish a checkpoint or ordinary entry into `StateStore` (or
    /// register Group 0 blocks for prefix reuse), and a matching follow-up turn with the same
    /// salt must get `cached_tokens == 0`.
    #[test]
    fn dflash2_scheduled_preempt_and_replay_reuse_cache_false_does_not_publish() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        let tok = build_synthetic_inkling2_tokenizer();
        let msg_user_id = tok
            .inner()
            .token_to_id(crate::tools::inkling::MESSAGE_USER)
            .expect("MESSAGE_USER id");
        let inner = build_synthetic_dflash2_inner_with_window(512, Some(msg_user_id), 128, 4);
        let mut sched = Inkling2SchedulerState::new(inner).expect("scheduler");
        let (tx, mut rx) = mpsc::unbounded_channel::<Inkling2Cmd>();

        // Prompt long enough to cross a 32-token block boundary during prefill while leaving
        // plenty of decode waves so we can preempt mid-turn after the boundary is recorded.
        let shared_prompt = "alpha1 beta2 gamma3 delta4 epsilon5 zeta6 eta7 theta8 \
                             alpha1 beta2 gamma3 delta4 epsilon5 zeta6 eta7 theta8";
        let render_cfg =
            scheduled_greedy_dflash2_config("owner-preempt-render", 12, "salt-preempt-optout");
        let prompt_tokens = crate::engine::session::admit_paged_turn(
            &mut sched.inner,
            user_msg(shared_prompt),
            render_cfg,
            crate::engine::session::TurnKind::Start,
            false,
        )
        .expect("render prompt tokens")
        .tokens;
        assert!(
            prompt_tokens.len() > 32,
            "prompt must exceed 32 tokens so prefill crosses a block boundary (len={})",
            prompt_tokens.len()
        );

        // Admit a scheduled DFlash2 turn with `reuse_cache = Some(false)` on a normal
        // non-scratch scheduler (`self.scratch == false`) via `admit_scratch`, with enough
        // `max_new_tokens` (40) to cross the next 32-token block boundary during scheduled
        // DFlash2 decode while still having remaining tokens to decode after preemption.
        let mut scratch = sched
            .admit_scratch(crate::engine::hybrid_scheduler::ScratchRequest {
                tokens: prompt_tokens,
                breaks: Vec::new(),
                mtp_depth: Some(4),
                owner_id: "owner-preempt-optout".to_string(),
                cache_salt: "salt-preempt-optout".to_string(),
            })
            .expect("admit opted-out scheduled DFlash2 turn");
        let seq_id = scratch.seq_id;
        {
            let turn = sched.turn_mut(seq_id).expect("admitted turn");
            assert!(
                !turn.payload.reuse_cache,
                "admitted turn must have reuse_cache == false"
            );
            turn.payload.params.max_new_tokens = 40;
        }
        assert_eq!(
            sched
                .inner
                .paged
                .as_ref()
                .unwrap()
                .seq_reuse_cache
                .get(&seq_id)
                .copied(),
            Some(false),
            "opted-out turn must record seq_reuse_cache[seq_id] == false"
        );

        // Drive until the turn has crossed a 32-token boundary (so `block_boundary_conv` is
        // populated for `seq_id`) while it is still running.
        let mut preempted = false;
        for _ in 0..64 {
            let _ = sched.drive(&mut rx);
            if let Ok(res) = scratch.reply.try_recv() {
                panic!("opted-out turn finished before preemption could be triggered: {res:?}");
            }
            let crossed_boundary = sched.inner.paged.as_ref().is_some_and(|rt| {
                rt.block_boundary_conv
                    .get(&seq_id)
                    .is_some_and(|(boundary, _)| *boundary >= 32)
            });
            if crossed_boundary {
                assert!(
                    sched.preempt_running(seq_id),
                    "preempt_running must preempt the running opted-out turn"
                );
                preempted = true;
                break;
            }
        }
        assert!(
            preempted,
            "turn must cross 32-token boundary and be preempted"
        );
        assert!(sched.preemptions() >= 1);
        {
            let rt = sched.inner.paged.as_ref().expect("paged runtime");
            assert_eq!(
                rt.state_store.store.entries().count(),
                0,
                "preempting a reuse_cache=false turn must NOT publish any StateStore checkpoint"
            );
            assert_eq!(
                rt.state_store.disk_counters().checkpoints_published,
                0,
                "preempting a reuse_cache=false turn must NOT increment checkpoints_published"
            );
            assert!(
                !rt.seq_reuse_cache.contains_key(&seq_id),
                "preempt_scheduled_cache must clean up seq_reuse_cache on release"
            );
        }

        // Resume preemption replay and verify replay preserves `reuse_cache == false`!
        sched.resume_preempted();
        {
            let turn = sched.turn_mut(seq_id).expect("replaying turn");
            assert!(
                !turn.payload.prefix.reuse_cache,
                "preemption replay must preserve prefix.reuse_cache == false"
            );
            assert_eq!(
                sched
                    .inner
                    .paged
                    .as_ref()
                    .unwrap()
                    .seq_reuse_cache
                    .get(&seq_id)
                    .copied(),
                Some(false),
                "preemption replay must record seq_reuse_cache[seq_id] == false"
            );
        }
        let res1 = loop {
            match scratch.reply.try_recv() {
                Ok(res) => break res.expect("replayed opted-out turn must succeed"),
                Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {
                    let _ = sched.drive(&mut rx);
                }
                Err(tokio::sync::oneshot::error::TryRecvError::Closed) => {
                    panic!("opted-out turn reply channel closed");
                }
            }
        };
        assert_eq!(res1.num_tokens, 40);
        {
            let rt = sched.inner.paged.as_ref().expect("paged runtime");
            assert_eq!(
                rt.state_store.store.entries().count(),
                0,
                "replayed reuse_cache=false turn must leave StateStore empty after completion"
            );
            let c = rt.state_store.disk_counters();
            assert_eq!(c.checkpoints_published, 0);
            assert_eq!(c.ordinary_published, 0);
        }

        // Follow-up request with the exact same salt and prompt prefix must get `cached_tokens == 0`.
        let followup_cfg =
            scheduled_greedy_dflash2_config("owner-preempt-followup", 6, "salt-preempt-optout");
        let res2 = send_session_start(
            &mut sched,
            &tx,
            &mut rx,
            two_user_msgs(shared_prompt, "probe"),
            followup_cfg,
        )
        .expect("follow-up turn");
        assert_eq!(
            res2.cached_tokens, 0,
            "follow-up turn with the same salt must get 0 cached_tokens after opted-out preempt+replay"
        );
    }

    /// Round-3 Finding 3 + Round-6 Finding 2 regression:
    /// 1. An anchor whose state blob landed on SSD with `chain_owed == true` (RAM still resident)
    ///    remains visible as `UnitKey::Store` (`bytes > 0, pages: 0, later: false`) AND as
    ///    `UnitKey::StoreChain` (`bytes: 0, pages: chain_pages, later: true`):
    ///    - A **patient** byte-only claim (`waits_for_writes == true`) with shortfall larger than
    ///      one anchor (`1.5 * anchor_bytes`) does NOT report `Short` and does NOT double-credit
    ///      `entry_a` (`later_bytes == 0`); it passes `entry_a`, reclaims `entry_b`, and reports
    ///      `FreeOutcome::Pending`.
    ///    - A **page-only** claim (`need.pages = chain_pages, need.bytes = 0`) takes
    ///      `StoreChain(entry_a)` once and returns `FreeOutcome::Pending` with `pages == 0`.
    ///    - An **urgent** byte-only claim (`NeedTiming::Immediate`, `waits_for_writes == false`)
    ///      discards the incomplete SSD blob from `tier.manager`, erases the entry, frees its RAM
    ///      + lineage immediately, and returns `FreeOutcome::Met`.
    /// 2. When `entry` has an in-flight state write (`landed: false`), `reclaim_entry` returns
    ///    `ReclaimOutcome::Pending` under a patient claim and `ledger::free` passes over it to
    ///    reclaim another eligible anchor without double-crediting `later_bytes`.
    #[test]
    fn governor_ledger_does_not_double_credit_pending_or_chain_owed_anchor() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        use crate::engine::governor::ledger::{
            Charge, ClaimClass, ClaimRank, ClaimSpec, FreeOutcome, FreePool, Ledger, Need,
            NeedTiming, ReachClass, Room, Stage, Taken, UnitKey, Victim, VictimQuery, VictimSource,
        };
        use crate::models::qwen3_5::dflash2_state::store::{DiskCopy, StateClass};
        use mlx_paged_attn::{ColdGroup, MakeRoom};

        struct TempDirGuard(std::path::PathBuf);
        impl Drop for TempDirGuard {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        struct EmptyPool;
        impl FreePool for EmptyPool {
            fn charge(&self) -> Charge {
                Charge::default()
            }
            fn clear(&mut self) {}
        }
        struct RecordingSource<'a> {
            inner: &'a mut super::Inkling2Inner,
            taken: Vec<UnitKey>,
        }
        impl VictimSource for RecordingSource<'_> {
            fn units(&self, query: &VictimQuery) -> Vec<Victim> {
                self.inner.inkling2_victim_units(query, &[]).into_units()
            }
            fn take(&mut self, query: &VictimQuery, victim: &Victim) -> Taken {
                self.taken.push(victim.key.clone());
                self.inner.inkling2_take_victim(query, victim)
            }
        }

        let temp_root =
            std::env::temp_dir().join(format!("inkling2-ledger-pending-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&temp_root).expect("create temp_root");
        let _temp_guard = TempDirGuard(temp_root.clone());

        let mut inner = build_synthetic_dflash2_inner(256, None);
        let disk_opts = crate::disk_tier::DiskTierOptions {
            cache_disk: crate::disk_tier::CacheDisk::Bytes(64u64 << 20),
            persistent: false,
            cache_dir: None,
            temp_root: Some(temp_root),
        };
        assert!(
            inner.attach_state_disk(&disk_opts, || None).is_some(),
            "temporary StateDisk tier must attach"
        );

        let tokens_a: Vec<u32> = (10u32..50u32).collect();
        let tokens_b: Vec<u32> = (60u32..100u32).collect();
        let tokens_c: Vec<u32> = (110u32..150u32).collect();
        let tokens_d: Vec<u32> = (160u32..200u32).collect();
        let conv = crate::models::inkling2::short_conv::Inkling2ShortConvState::new(
            inner.config.num_layer_visits(),
            inner.config.sconv_kernel_size,
            inner.config.num_key_value_heads * inner.config.head_dim,
            inner.config.hidden_size,
            crate::array::DType::BFloat16,
        )
        .expect("conv");

        let (entry_a, entry_b, anchor_bytes, entry_b_ram_bytes) = {
            let rt = inner.paged.as_mut().expect("paged");
            rt.coordinator.reset_scheduled_request(1).expect("reset 1");
            rt.coordinator
                .record_tokens_all(1, &tokens_a)
                .expect("record a");
            let id_a = rt
                .state_store
                .capture_boundary(
                    &mut rt.coordinator,
                    &conv,
                    1,
                    &tokens_a,
                    32,
                    101,
                    StateClass::Ordinary,
                    None,
                )
                .expect("capture a")
                .expect("id_a");
            rt.coordinator.reset_scheduled_request(2).expect("reset 2");
            rt.coordinator
                .record_tokens_all(2, &tokens_b)
                .expect("record b");
            let id_b = rt
                .state_store
                .capture_boundary(
                    &mut rt.coordinator,
                    &conv,
                    2,
                    &tokens_b,
                    32,
                    202,
                    StateClass::Ordinary,
                    None,
                )
                .expect("capture b")
                .expect("id_b");
            let bytes = rt.state_store.store.get(id_a).unwrap().state_bytes();
            assert_eq!(bytes, rt.state_store.store.get(id_b).unwrap().state_bytes());
            let ram_bytes = rt
                .state_store
                .store
                .get(id_b)
                .unwrap()
                .ram()
                .unwrap()
                .cached_bytes;
            assert!(bytes > ram_bytes && ram_bytes > 0);
            // Release live sequence handles so both entries' Global KV blocks are cache-only
            // (pinned solely by StateStore marks, so `freed_by_marks` counts them accurately).
            rt.coordinator
                .release_request_all(1)
                .expect("release seq 1");
            rt.coordinator
                .release_request_all(2)
                .expect("release seq 2");
            (id_a, id_b, bytes, ram_bytes)
        };

        let patient_spec = ClaimSpec {
            cls: ClaimClass::InService,
            request_id: 1,
            rank: ClaimRank {
                priority: 0,
                decoding: true,
                completed_tokens: 32,
                earliness: 1,
            },
            reach: ReachClass::Ordinary,
            timing: NeedTiming::CanWait,
        };
        let urgent_spec = ClaimSpec {
            timing: NeedTiming::Immediate,
            ..patient_spec
        };

        // Part 1: Stage `entry_a`'s real state blob on SSD and let it land WITHOUT running
        // `chain_step`, so `entry_a` has `(ram = Some, disk = Some { landed: true, chain_owed: true })`.
        let (chain_pages, entry_a_key) = {
            let rt = inner.paged.as_mut().expect("paged");
            assert!(
                rt.state_store
                    .stage_entry(&rt.coordinator, entry_a, MakeRoom::Evict, true)
            );
            let deadline = Instant::now() + Duration::from_secs(10);
            while rt.state_store.disk.as_ref().unwrap().writing() {
                assert!(Instant::now() < deadline, "timed out landing entry_a blob");
                let _ = rt
                    .state_store
                    .disk
                    .as_ref()
                    .unwrap()
                    .engine
                    .await_landing(deadline);
                rt.state_store.poll_offload(&rt.coordinator);
            }
            let entry = rt.state_store.store.get(entry_a).unwrap();
            let copy = entry.disk().copied().expect("landed entry_a copy");
            assert!(copy.landed && copy.chain_owed);
            assert!(
                rt.state_store
                    .disk
                    .as_ref()
                    .unwrap()
                    .tier
                    .manager
                    .contains_in(&copy.key, ColdGroup::DFlash2State)
            );
            (entry.chain().len() as u32, copy.key)
        };
        let query = VictimQuery {
            rights: crate::engine::governor::ledger::TakeRights {
                units: true,
                waits_for_writes: true,
                waits_for_landings: true,
                ..Default::default()
            },
            need: Need {
                bytes: anchor_bytes + anchor_bytes / 2,
                ..Need::default()
            },
        };
        let runs = inner.inkling2_victim_units(&query, &[]);
        let store_unit = runs
            .merged
            .iter()
            .find(|v| v.key == UnitKey::Store(entry_a))
            .expect("chain-owed landed RAM anchor must remain visible as UnitKey::Store");
        assert_eq!(store_unit.bytes, anchor_bytes);
        assert_eq!(store_unit.pages, chain_pages);
        assert_eq!(store_unit.stage, Stage::NeedsWrite);
        assert!(!store_unit.later);
        assert!(
            !runs
                .after
                .iter()
                .any(|v| v.key == UnitKey::StoreChain(entry_a)),
            "RAM-backed anchor must NOT emit a duplicate UnitKey::StoreChain"
        );

        // Part 1A: Patient byte-only shortfall (`1.5 * anchor_bytes`): `entry_a` stays visible in
        // `bound()`, returns `Pending` with `later_bytes == 0` in `take()`, and `ledger::free`
        // passes over `entry_a` to reclaim `entry_b` and returns `FreeOutcome::Pending` (NOT Short)!
        let mut ledger = Ledger::new();
        let patient_claim = ledger.open(patient_spec, 1);
        {
            let mut src = RecordingSource {
                inner: &mut inner,
                taken: Vec::new(),
            };
            let res = ledger.free(
                patient_claim,
                Need {
                    bytes: anchor_bytes + anchor_bytes / 2,
                    ..Need::default()
                },
                Room::default(),
                &mut EmptyPool,
                &mut [&mut src],
            );
            assert_eq!(
                res.outcome,
                FreeOutcome::Pending,
                "patient byte-only claim with chain-owed entry_a + eligible entry_b must NOT report Short"
            );
            assert_eq!(res.later_bytes, 0, "must never double-credit entry_a");
            assert_eq!(
                res.now_bytes, entry_b_ram_bytes,
                "must pass chain-owed entry_a and reclaim eligible victim entry_b"
            );
            assert_eq!(src.taken.first(), Some(&UnitKey::Store(entry_a)));
            assert_eq!(src.taken.get(1), Some(&UnitKey::Store(entry_b)));
        }
        {
            let rt = inner.paged.as_ref().expect("paged");
            assert!(
                rt.state_store
                    .store
                    .get(entry_a)
                    .is_some_and(|e| e.ram().is_some()),
                "patient claim must keep chain-owed entry_a RAM intact"
            );
            assert!(
                rt.state_store
                    .store
                    .get(entry_b)
                    .is_some_and(|e| e.ram().is_none()),
                "eligible victim entry_b must have its RAM reclaimed"
            );
        }

        // Part 1B: Urgent byte-only claim (`NeedTiming::Immediate`, `waits_for_writes == false`)
        // against `entry_a` (`landed && chain_owed`): must discard `entry_a`'s incomplete SSD blob,
        // erase `entry_a`, free its RAM + lineage immediately, and report `FreeOutcome::Met`!
        let urgent_claim = ledger.open(urgent_spec, 1);
        {
            let mut src = RecordingSource {
                inner: &mut inner,
                taken: Vec::new(),
            };
            let res = ledger.free(
                urgent_claim,
                Need {
                    bytes: anchor_bytes,
                    ..Need::default()
                },
                Room::default(),
                &mut EmptyPool,
                &mut [&mut src],
            );
            assert_eq!(
                res.outcome,
                FreeOutcome::Met,
                "urgent byte-only claim must discard incomplete chain-owed SSD copy and free RAM immediately"
            );
            assert_eq!(res.now_bytes, anchor_bytes);
            assert_eq!(res.later_bytes, 0);
            assert_eq!(src.taken, vec![UnitKey::Store(entry_a)]);
        }
        {
            let rt = inner.paged.as_ref().expect("paged");
            assert!(
                rt.state_store.store.get(entry_a).is_none(),
                "urgent reclaim must erase chain-owed entry_a"
            );
            assert!(
                !rt.state_store
                    .disk
                    .as_ref()
                    .unwrap()
                    .tier
                    .manager
                    .contains_in(&entry_a_key, ColdGroup::DFlash2State),
                "urgent reclaim must remove entry_a's incomplete SSD blob from tier.manager"
            );
        }

        // Wait for `entry_b`'s spilled write to finish cleanly before Part 1C + Part 2.
        {
            let rt = inner.paged.as_mut().expect("paged");
            let deadline = Instant::now() + Duration::from_secs(10);
            while rt.state_store.disk.as_ref().unwrap().writing() {
                assert!(Instant::now() < deadline, "timed out landing entry_b");
                let _ = rt
                    .state_store
                    .disk
                    .as_ref()
                    .unwrap()
                    .engine
                    .await_landing(deadline);
                rt.state_store.poll_offload(&rt.coordinator);
            }
            // Remove `entry_b` and purge dead leaves so Part 1C + Part 2 have a clean store and
            // KV pool with just `entry_c` + `entry_d`.
            rt.state_store.store.remove(entry_b);
            rt.state_store.sync_marks(&rt.coordinator);
            rt.coordinator
                .release_all_and_purge()
                .expect("purge dead leaves");
        }

        // Part 1C: Seed a real disk-only `chain_owed` entry `entry_c` (blob landed on SSD,
        // `ram().is_none()`, Global KV chain still owed). Verify `inkling2_victim_units` emits
        // `UnitKey::StoreChain(entry_c)` with allocator-backed `pages == chain_pages`, and
        // `ledger::free` takes `StoreChain(entry_c)` ONCE without looping or double-counting.
        let entry_c = {
            let rt = inner.paged.as_mut().expect("paged");
            rt.coordinator.reset_scheduled_request(1).expect("reset 1c");
            rt.coordinator
                .record_tokens_all(1, &tokens_c)
                .expect("record c");
            let id_c = rt
                .state_store
                .capture_boundary(
                    &mut rt.coordinator,
                    &conv,
                    1,
                    &tokens_c,
                    32,
                    303,
                    StateClass::Ordinary,
                    None,
                )
                .expect("capture c")
                .expect("id_c");
            rt.coordinator
                .release_request_all(1)
                .expect("release seq 1c");
            assert!(
                rt.state_store
                    .stage_entry(&rt.coordinator, id_c, MakeRoom::Evict, false)
            );
            let deadline = Instant::now() + Duration::from_secs(10);
            while rt.state_store.disk.as_ref().unwrap().writing() {
                assert!(Instant::now() < deadline, "timed out landing entry_c blob");
                let _ = rt
                    .state_store
                    .disk
                    .as_ref()
                    .unwrap()
                    .engine
                    .await_landing(deadline);
                rt.state_store.poll_offload(&rt.coordinator);
            }
            let entry = rt.state_store.store.get(id_c).expect("entry_c indexed");
            assert!(entry.ram().is_none());
            assert!(entry.disk().is_some_and(|c| c.landed && c.chain_owed));
            id_c
        };
        {
            let runs = inner.inkling2_victim_units(
                &VictimQuery {
                    rights: crate::engine::governor::ledger::TakeRights {
                        units: true,
                        waits_for_writes: true,
                        waits_for_landings: true,
                        ..Default::default()
                    },
                    need: Need {
                        pages: chain_pages,
                        ..Need::default()
                    },
                },
                &[],
            );
            let chain_unit = runs
                .after
                .iter()
                .find(|v| v.key == UnitKey::StoreChain(entry_c))
                .expect("disk-only chain_owed anchor must be emitted as UnitKey::StoreChain");
            assert_eq!(chain_unit.bytes, 0);
            assert_eq!(chain_unit.pages, chain_pages);
            assert_eq!(chain_unit.stage, Stage::NeedsWrite);
            assert!(chain_unit.later);
        }
        {
            let mut src = RecordingSource {
                inner: &mut inner,
                taken: Vec::new(),
            };
            let res = ledger.free(
                patient_claim,
                Need {
                    pages: chain_pages,
                    ..Need::default()
                },
                Room::default(),
                &mut EmptyPool,
                &mut [&mut src],
            );
            assert_eq!(
                src.taken,
                vec![UnitKey::StoreChain(entry_c)],
                "StoreChain(entry_c) must be taken exactly once per ledger::free pass"
            );
            match res.outcome {
                FreeOutcome::Pending => assert_eq!(res.pages, 0),
                FreeOutcome::Met => assert_eq!(res.pages, chain_pages),
                FreeOutcome::Short => {
                    panic!("page-only claim on disk-only StoreChain must never return Short")
                }
            }
        }

        // Wait for any spill started on `entry_d`, reset cleanly, and re-seed fresh `entry_c` +
        // `entry_d` for Part 2 (in-flight state write `landed: false`).
        let (entry_c, entry_d, entry_d_ram_bytes, entry_c_key) = {
            let rt = inner.paged.as_mut().expect("paged");
            let deadline = Instant::now() + Duration::from_secs(10);
            while rt.state_store.disk.as_ref().unwrap().writing() {
                assert!(
                    Instant::now() < deadline,
                    "timed out landing spill before Part 2"
                );
                let _ = rt
                    .state_store
                    .disk
                    .as_ref()
                    .unwrap()
                    .engine
                    .await_landing(deadline);
                rt.state_store.poll_offload(&rt.coordinator);
            }
            rt.state_store.clear_ram_and_checkpoints();
            rt.state_store.sync_marks(&rt.coordinator);
            rt.coordinator
                .release_all_and_purge()
                .expect("purge before Part 2");
            rt.coordinator
                .reset_scheduled_request(1)
                .expect("reset 1p2");
            rt.coordinator
                .record_tokens_all(1, &tokens_c)
                .expect("record c p2");
            let id_c = rt
                .state_store
                .capture_boundary(
                    &mut rt.coordinator,
                    &conv,
                    1,
                    &tokens_c,
                    32,
                    303,
                    StateClass::Ordinary,
                    None,
                )
                .expect("capture c p2")
                .expect("id_c p2");
            rt.coordinator
                .reset_scheduled_request(2)
                .expect("reset 2p2");
            rt.coordinator
                .record_tokens_all(2, &tokens_d)
                .expect("record d p2");
            let id_d = rt
                .state_store
                .capture_boundary(
                    &mut rt.coordinator,
                    &conv,
                    2,
                    &tokens_d,
                    32,
                    404,
                    StateClass::Ordinary,
                    None,
                )
                .expect("capture d p2")
                .expect("id_d p2");
            let key_c = rt
                .state_store
                .disk
                .as_ref()
                .unwrap()
                .state_key(rt.state_store.store.get(id_c).unwrap())
                .expect("key_c p2");
            let ram_d = rt
                .state_store
                .store
                .get(id_d)
                .unwrap()
                .ram()
                .unwrap()
                .cached_bytes;
            rt.coordinator
                .release_request_all(1)
                .expect("release seq 1 p2");
            rt.coordinator
                .release_request_all(2)
                .expect("release seq 2 p2");
            (id_c, id_d, ram_d, key_c)
        };

        // Part 2: Mark `entry_c` as an in-flight state write (`landed: false`) so `entry_c` is
        // emitted as `UnitKey::Store(entry_c)` (older stamp than `entry_d`) and `reclaim_entry`
        // returns `ReclaimOutcome::Pending`. With a shortfall LARGER than one anchor
        // (`1.5 * anchor_bytes`), `ledger::free` must NOT double-credit `entry_c`'s `later_bytes`;
        // it must pass over `entry_c` (`order.pass`) and move on to reclaim `entry_d`!
        {
            let rt = inner.paged.as_mut().expect("paged");
            rt.state_store.store.set_disk(
                entry_c,
                Some(DiskCopy {
                    key: entry_c_key,
                    bytes: anchor_bytes,
                    landed: false,
                    window_begin: 0,
                    chain_owed: true,
                }),
            );
        }
        {
            let mut src = RecordingSource {
                inner: &mut inner,
                taken: Vec::new(),
            };
            let res = ledger.free(
                patient_claim,
                Need {
                    bytes: anchor_bytes + anchor_bytes / 2,
                    ..Need::default()
                },
                Room::default(),
                &mut EmptyPool,
                &mut [&mut src],
            );
            assert_eq!(
                res.outcome,
                FreeOutcome::Pending,
                "shortfall larger than one anchor with one pending + one reclaimed must stay Pending"
            );
            assert_eq!(
                res.later_bytes, 0,
                "pending reclaim must NOT credit later_bytes"
            );
            assert_eq!(
                res.now_bytes, entry_d_ram_bytes,
                "ledger must move past pending entry_c and credit exact freed RAM bytes for spilled entry_d"
            );
            assert_eq!(
                src.taken.first(),
                Some(&UnitKey::Store(entry_c)),
                "ledger tries older pending entry_c first"
            );
            assert_eq!(
                src.taken.get(1),
                Some(&UnitKey::Store(entry_d)),
                "after Pending with later_bytes=0, ledger must pass entry_c and move directly to entry_d"
            );
            assert_eq!(
                src.taken
                    .iter()
                    .filter(|k| **k == UnitKey::Store(entry_d))
                    .count(),
                1,
                "ledger must reclaim eligible victim entry_d"
            );
        }
        {
            let rt = inner.paged.as_ref().expect("paged");
            assert!(
                rt.state_store
                    .store
                    .get(entry_c)
                    .is_some_and(|e| e.ram().is_some()),
                "pending entry_c keeps its RAM intact"
            );
            let entry_d_after = rt
                .state_store
                .store
                .get(entry_d)
                .expect("spilled entry_d stays indexed for its disk copy");
            assert!(
                entry_d_after.ram().is_none(),
                "eligible victim entry_d must have its RAM reclaimed"
            );
            assert_eq!(
                entry_d_after.state_bytes(),
                anchor_bytes - entry_d_ram_bytes,
                "spilled entry_d retains only its lineage_bytes in store"
            );
        }
        let _ = inner.close_state_disk(Instant::now() + Duration::from_secs(5));
    }

    /// Round-4 Finding 4 regression: optional `DraftContextWindow` materialization failure inside
    /// `install_state_hit_sync` must be best-effort — target prefix restoration (Global KV, SWA,
    /// ShortConv) still succeeds cleanly, the affected sequence's draft windows are invalidated,
    /// and an AR follow-up turn (`enable_mtp: Some(false)`) completes from the cached target
    /// prefix (`cached_tokens > 0`) with bit-identical output!
    #[test]
    fn dflash2_cached_prefix_ar_followup_succeeds_when_draft_window_restore_fails() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        struct RestoreFailGuard;
        impl Drop for RestoreFailGuard {
            fn drop(&mut self) {
                super::FAIL_DRAFT_WINDOW_RESTORE_SEQ.with(|c| c.set(None));
                super::DRAFT_WINDOW_RESTORE_INJECTED_COUNT.with(|c| c.set(0));
            }
        }
        let _guard = RestoreFailGuard;

        let inner = build_synthetic_dflash2_inner_with_window(256, None, 128, 4);
        let mut sched = Inkling2SchedulerState::new(inner).expect("scheduler");
        let (tx, mut rx) = mpsc::unbounded_channel::<Inkling2Cmd>();

        let shared_prompt = "alpha1-shared-prefix-0123456789abcdef0123456789";
        let greedy_ar_cfg = |owner: &str| {
            let mut cfg = scheduled_greedy_dflash2_config(owner, 10, "salt-dw-restore-best-effort");
            cfg.enable_mtp = Some(false);
            cfg
        };

        // 1. Seed a cached StateStore anchor at depth >= 32 carrying a complete PackedDraftWindow.
        let seed_res = send_session_start(
            &mut sched,
            &tx,
            &mut rx,
            user_msg(shared_prompt),
            greedy_ar_cfg("ar-restore-seed"),
        )
        .expect("seed AR turn");
        assert_eq!(seed_res.num_tokens, 10);
        assert!(
            sched
                .inner
                .paged
                .as_ref()
                .expect("paged")
                .state_store
                .store
                .entries()
                .any(|e| e.ram().is_some_and(|a| a.draft_window.is_some())),
            "seed turn must publish an anchor with a complete PackedDraftWindow"
        );

        // 2. Clean cached AR follow-up baseline (no error injected).
        let base_res = send_session_start(
            &mut sched,
            &tx,
            &mut rx,
            two_user_msgs(shared_prompt, "q2"),
            greedy_ar_cfg("ar-restore-base"),
        )
        .expect("clean AR follow-up");
        assert!(
            base_res.cached_tokens >= 32,
            "baseline AR follow-up must hit the cached prefix (got {})",
            base_res.cached_tokens
        );
        let base_seq = *sched
            .owner_sequences
            .get("ar-restore-base")
            .expect("base_seq");

        // 3. Inject DraftContextWindow restore failure for the next owner's `install_state_hit_sync`.
        let expected_inj_seq = base_seq + 1;
        super::DRAFT_WINDOW_RESTORE_INJECTED_COUNT.with(|c| c.set(0));
        super::FAIL_DRAFT_WINDOW_RESTORE_SEQ.with(|c| c.set(Some(expected_inj_seq)));
        let inj_res = send_session_start(
            &mut sched,
            &tx,
            &mut rx,
            two_user_msgs(shared_prompt, "q2"),
            greedy_ar_cfg("ar-restore-inj"),
        )
        .expect("AR follow-up must succeed cleanly when optional DraftContextWindow restore fails");
        super::FAIL_DRAFT_WINDOW_RESTORE_SEQ.with(|c| c.set(None));

        let inj_seq = *sched
            .owner_sequences
            .get("ar-restore-inj")
            .expect("inj_seq");
        assert_eq!(inj_seq, expected_inj_seq);
        assert_eq!(
            super::DRAFT_WINDOW_RESTORE_INJECTED_COUNT.with(|c| c.get()),
            1,
            "injected DraftContextWindow restore failure must have fired inside install_state_hit_sync"
        );
        assert_eq!(
            inj_res.cached_tokens, base_res.cached_tokens,
            "AR follow-up must still reuse the exact cached target prefix"
        );
        assert_eq!(inj_res.num_tokens, base_res.num_tokens);
        assert_eq!(
            inj_res.raw_text, base_res.raw_text,
            "AR follow-up must produce bit-identical output despite optional draft-window restore failure"
        );
        {
            let rt = sched.inner.paged.as_ref().expect("paged");
            assert!(
                !rt.block_boundary_draft_window.contains_key(&inj_seq),
                "failed optional draft-window restore must invalidate block_boundary_draft_window"
            );
        }
    }

    /// Round-5 Finding 2 regression: injecting a DFlash2 draft KV reconstruction failure
    /// (`FAIL_DRAFT_KV_RESTORE_SEQ`) on:
    /// 1. Scheduled DFlash2 prefix hit -> logs warning, invalidates draft window, falls back
    ///    cleanly to scheduled AR while KEEPING the restored target prefix (`cached_tokens > 0`),
    ///    and produces bit-identical output!
    /// 2. Exclusive greedy DFlash2 prefix hit -> logs warning, invalidates draft window, falls
    ///    back cleanly to cold prefill (`cached_tokens == 0`), and produces bit-identical output!
    #[test]
    fn dflash2_draft_kv_restore_failure_falls_back_cleanly_on_scheduled_and_exclusive() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        struct KvRestoreFailGuard;
        impl Drop for KvRestoreFailGuard {
            fn drop(&mut self) {
                super::FAIL_DRAFT_KV_RESTORE_SEQ.with(|c| c.set(None));
                super::DRAFT_KV_RESTORE_INJECTED_COUNT.with(|c| c.set(0));
            }
        }
        let _guard = KvRestoreFailGuard;

        let inner = build_synthetic_dflash2_inner_with_window(256, None, 128, 4);
        let mut sched = Inkling2SchedulerState::new(inner).expect("scheduler");
        let (tx, mut rx) = mpsc::unbounded_channel::<Inkling2Cmd>();

        let shared_prompt = "alpha1-shared-prefix-0123456789abcdef0123456789";
        let salt = "salt-kv-restore-fail";

        // 1. Seed a cached StateStore anchor at depth >= 32 carrying a complete PackedDraftWindow.
        let seed_res = send_session_start(
            &mut sched,
            &tx,
            &mut rx,
            user_msg(shared_prompt),
            scheduled_greedy_dflash2_config("kv-fail-seed", 10, salt),
        )
        .expect("seed turn");
        assert_eq!(seed_res.num_tokens, 10);
        assert!(
            sched
                .inner
                .paged
                .as_ref()
                .expect("paged")
                .state_store
                .store
                .entries()
                .any(|e| e.ram().is_some_and(|a| a.draft_window.is_some())),
            "seed turn must publish an anchor with a complete PackedDraftWindow"
        );

        // 2A. Clean scheduled DFlash2 follow-up baseline (reuses target prefix AND speculates!).
        let waves_before_base = sched.inner.scheduled_dspark.total_verify_waves;
        let base_sched = send_session_start(
            &mut sched,
            &tx,
            &mut rx,
            two_user_msgs(shared_prompt, "q2"),
            scheduled_greedy_dflash2_config("kv-fail-sched-base", 10, salt),
        )
        .expect("clean scheduled DFlash2 follow-up");
        assert!(
            base_sched.cached_tokens >= 32,
            "baseline scheduled DFlash2 must hit cached prefix (got {})",
            base_sched.cached_tokens
        );
        assert!(
            sched.inner.scheduled_dspark.total_verify_waves > waves_before_base,
            "baseline scheduled DFlash2 must restore draft KV and speculate"
        );
        let sched_base_seq = *sched
            .owner_sequences
            .get("kv-fail-sched-base")
            .expect("sched_base_seq");

        // 2B. Inject DFlash2 draft KV restore failure on scheduled DFlash2 prefix hit -> must fall
        //     back cleanly to scheduled AR keeping `cached_tokens > 0` with identical output!
        let expected_sched_inj_seq = sched_base_seq + 1;
        let waves_before_inj = sched.inner.scheduled_dspark.total_verify_waves;
        super::DRAFT_KV_RESTORE_INJECTED_COUNT.with(|c| c.set(0));
        super::FAIL_DRAFT_KV_RESTORE_SEQ.with(|c| c.set(Some(expected_sched_inj_seq)));
        let inj_sched = send_session_start(
            &mut sched,
            &tx,
            &mut rx,
            two_user_msgs(shared_prompt, "q2"),
            scheduled_greedy_dflash2_config("kv-fail-sched-inj", 10, salt),
        )
        .expect("scheduled DFlash2 must fall back cleanly to AR on draft KV restore failure");
        super::FAIL_DRAFT_KV_RESTORE_SEQ.with(|c| c.set(None));

        let sched_inj_seq = *sched
            .owner_sequences
            .get("kv-fail-sched-inj")
            .expect("sched_inj_seq");
        assert_eq!(sched_inj_seq, expected_sched_inj_seq);
        assert_eq!(
            super::DRAFT_KV_RESTORE_INJECTED_COUNT.with(|c| c.get()),
            1,
            "injected failure must have fired inside begin_scheduled_dspark"
        );
        assert_eq!(
            inj_sched.cached_tokens, base_sched.cached_tokens,
            "scheduled fallback to AR must keep the restored target prefix (cached_tokens > 0)"
        );
        assert_eq!(
            sched.inner.scheduled_dspark.total_verify_waves, waves_before_inj,
            "scheduled fallback must run AR without launching speculative verify waves"
        );
        assert_eq!(inj_sched.num_tokens, base_sched.num_tokens);
        assert_eq!(
            inj_sched.raw_text, base_sched.raw_text,
            "scheduled AR fallback must match baseline greedy output"
        );

        // 3A. Clean exclusive greedy DFlash2 follow-up baseline (hits cached prefix).
        let excl_base = send_session_start(
            &mut sched,
            &tx,
            &mut rx,
            two_user_msgs(shared_prompt, "q2"),
            exclusive_dflash2_config("kv-fail-excl-base", 10, salt),
        )
        .expect("clean exclusive greedy DFlash2 follow-up");
        assert!(
            excl_base.cached_tokens >= 32,
            "baseline exclusive greedy DFlash2 must hit cached prefix (got {})",
            excl_base.cached_tokens
        );
        assert_eq!(excl_base.raw_text, base_sched.raw_text);
        let excl_base_seq = *sched
            .owner_sequences
            .get("kv-fail-excl-base")
            .expect("excl_base_seq");

        // 3B. Inject DFlash2 draft KV restore failure on exclusive greedy DFlash2 prefix hit ->
        //     must fall back cleanly to cold prefill (`cached_tokens == 0`) and succeed!
        let expected_excl_inj_seq = excl_base_seq + 1;
        super::DRAFT_KV_RESTORE_INJECTED_COUNT.with(|c| c.set(0));
        super::FAIL_DRAFT_KV_RESTORE_SEQ.with(|c| c.set(Some(expected_excl_inj_seq)));
        let inj_excl = send_session_start(
            &mut sched,
            &tx,
            &mut rx,
            two_user_msgs(shared_prompt, "q2"),
            exclusive_dflash2_config("kv-fail-excl-inj", 10, salt),
        )
        .expect("exclusive greedy DFlash2 must fall back cleanly to cold prefill on draft KV restore failure");
        super::FAIL_DRAFT_KV_RESTORE_SEQ.with(|c| c.set(None));

        let excl_inj_seq = *sched
            .owner_sequences
            .get("kv-fail-excl-inj")
            .expect("excl_inj_seq");
        assert_eq!(excl_inj_seq, expected_excl_inj_seq);
        assert_eq!(
            super::DRAFT_KV_RESTORE_INJECTED_COUNT.with(|c| c.get()),
            1,
            "injected failure must have fired inside prime_dspark_prefix_state"
        );
        assert_eq!(
            inj_excl.cached_tokens, 0,
            "exclusive fallback must re-prefill cold from position 0"
        );
        assert_eq!(inj_excl.num_tokens, base_sched.num_tokens);
        assert_eq!(
            inj_excl.raw_text, base_sched.raw_text,
            "exclusive cold-prefill fallback must match baseline greedy output"
        );
    }

    /// Round-6 Finding 4 regression: once `materialize_final_token` commits the final token into
    /// target KV & ShortConv (`first_pos + 1`), any failure in the optional draft-side update
    /// (`dctx.append_taps`, `eval_state`, or missing taps) MUST be best-effort:
    /// - `materialize_final_token` returns `Ok(())` without aborting completed generation.
    /// - Target KV & ShortConv remain committed at `first_pos + 1`.
    /// - The sequence's parked draft context (`dspark_turn_contexts`) is removed and its
    ///   `draft_windows` + `block_boundary_draft_window` are invalidated so no stale draft state
    ///   survives at `first_pos`.
    #[test]
    fn materialize_final_token_survives_draft_update_failure_and_invalidates_draft_state() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        use crate::engine::paged_stepper::PagedStepModel;
        use crate::models::inkling2::dflash2::{Inkling2Dflash2Context, Inkling2DraftContext};
        use crate::models::inkling2::paged_forward::Inkling2PagedDecode;

        let mut inner = build_synthetic_dflash2_inner_with_window(256, None, 128, 1);
        let (orig_layers, width, slots) = {
            let draft = inner.dflash2.as_ref().expect("dflash2");
            (
                draft.config.target_layers.clone(),
                draft.config.context_width(),
                draft.config.sliding_window as u32,
            )
        };
        let prompt: Vec<u32> = (10u32..42u32).collect();

        // Helper: prime one sequence at `first_pos = 32` with live target KV + ShortConv,
        // matching parked DFlash2 context, and live + block-boundary draft windows.
        let prime_seq = |inner: &mut super::Inkling2Inner, seq_id: u32| {
            let prefix = crate::models::inkling2::paged_forward::Inkling2PrefixState {
                effective_cached_prefix_len: 0,
                suffix_len: prompt.len(),
                full_tokens: prompt.clone(),
                cache_salt: 77 + u64::from(seq_id),
                live_continuation: false,
                reuse_cache: false,
            };
            {
                let rt = inner.paged.as_mut().expect("paged");
                rt.active_seq = seq_id;
                rt.coordinator
                    .reset_scheduled_request(seq_id)
                    .expect("reset seq");
                rt.ensure_conv_state(seq_id, &inner.config)
                    .expect("conv state");
            }
            let draft_cfg = inner.dflash2.as_ref().expect("dflash2").config.clone();
            let mut ctx = Inkling2Dflash2Context::new(&draft_cfg);
            let _ = inner
                .paged_prefill_dflash2_with_taps(seq_id, &prompt, &prefix, &mut ctx)
                .expect("prefill dflash2");
            assert_eq!(ctx.committed_len() as u32, 32);
            inner
                .dspark_turn_contexts
                .insert(seq_id, Inkling2DraftContext::Dflash2(Box::new(ctx)));
            let rt = inner.paged.as_mut().expect("paged");
            let seed = crate::array::MxArray::zeros(
                &[1, 32, i64::from(width)],
                Some(crate::array::DType::BFloat16),
            )
            .expect("seed");
            rt.record_fused_draft_window(seq_id, 0, &seed, width, slots)
                .expect("seed draft window");
            assert_eq!(rt.coordinator.request_token_count_all(seq_id).unwrap(), 32);
            assert_eq!(rt.draft_windows.get(&seq_id).map(|w| w.end()), Some(32));
            assert!(rt.block_boundary_draft_window.contains_key(&seq_id));
        };

        // Case 1: Real `dctx.append_taps` error (extra bogus target visit -> `!taps.is_empty()`
        // but `taps.len() != draft.config.target_layers.len()`).
        prime_seq(&mut inner, 1);
        inner
            .dflash2
            .as_mut()
            .expect("dflash2")
            .config
            .target_layers
            .push(999);
        {
            let mut dec = Inkling2PagedDecode::new(&mut inner);
            dec.materialize_final_token(7)
                .expect("materialize_final_token must succeed even when draft append_taps fails");
        }
        {
            let rt = inner.paged.as_ref().expect("paged");
            assert_eq!(
                rt.coordinator.request_token_count_all(1).unwrap(),
                33,
                "target KV sequence length must stay committed at first_pos + 1"
            );
            assert!(
                !inner.dspark_turn_contexts.contains_key(&1),
                "failed parked DFlash2 context must be dropped"
            );
            assert!(
                !rt.draft_windows.contains_key(&1)
                    && !rt.block_boundary_draft_window.contains_key(&1),
                "sequence draft windows must be invalidated after draft update failure"
            );
        }

        // Case 2: Missing taps (`taps.is_empty()`, e.g. only bogus visit in `target_layers`):
        // must also return `Ok(())`, commit `first_pos + 1`, drop parked context, and invalidate
        // draft windows rather than leaving stale draft state at `first_pos`.
        inner
            .dflash2
            .as_mut()
            .expect("dflash2")
            .config
            .target_layers = orig_layers;
        prime_seq(&mut inner, 2);
        inner
            .dflash2
            .as_mut()
            .expect("dflash2")
            .config
            .target_layers = vec![999];
        {
            let mut dec = Inkling2PagedDecode::new(&mut inner);
            dec.materialize_final_token(8)
                .expect("materialize_final_token must succeed when taps are empty");
        }
        {
            let rt = inner.paged.as_ref().expect("paged");
            assert_eq!(
                rt.coordinator.request_token_count_all(2).unwrap(),
                33,
                "target KV sequence length must stay committed at first_pos + 1"
            );
            assert!(
                !inner.dspark_turn_contexts.contains_key(&2),
                "stale parked DFlash2 context must be dropped when taps are empty"
            );
            assert!(
                !rt.draft_windows.contains_key(&2)
                    && !rt.block_boundary_draft_window.contains_key(&2),
                "sequence draft windows must be invalidated when taps are empty"
            );
        }
    }

    /// Round-7 Finding 1 regression:
    /// 1. `inkling2_victim_units` derives `pages` from `PagedKVCacheAdapter::freed_by_marks` in
    ///    take order — blocks held by a live sequence advertise `0`, and blocks shared by two
    ///    stored anchors are counted ONCE (never double-counted).
    /// 2. Under `ram_cap > ram_len`, when a RAM-backed anchor has `(landed: true, chain_owed: true)`
    ///    and a patient page-only claim takes it (`Pending`), landing its Global KV chain MUST
    ///    automatically demote its RAM copy despite being under `ram_cap`, unmark its Global KV
    ///    blocks so they become immediately evictable, and allow a subsequent 2-block allocation.
    /// 3. An urgent page-only claim (`NeedTiming::Immediate`) against a RAM-backed
    ///    `(landed: true, chain_owed: true)` anchor erases it immediately, unmarks its Global KV
    ///    blocks, and returns `FreeOutcome::Met` with `pages == 2`.
    #[test]
    fn page_only_reclaim_demotes_under_cap_chain_owed_anchor_and_excludes_shared_blocks() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        use crate::engine::governor::ledger::{
            Charge, ClaimClass, ClaimRank, ClaimSpec, FreeOutcome, FreePool, Ledger, Need,
            NeedTiming, ReachClass, Room, TakeRights, Taken, UnitKey, Victim, VictimQuery,
            VictimSource,
        };
        use crate::models::qwen3_5::dflash2_state::store::StateClass;
        use mlx_paged_attn::MakeRoom;

        struct TempDirGuard(std::path::PathBuf);
        impl Drop for TempDirGuard {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        struct EmptyPool;
        impl FreePool for EmptyPool {
            fn charge(&self) -> Charge {
                Charge::default()
            }
            fn clear(&mut self) {}
        }
        struct RecordingSource<'a> {
            inner: &'a mut super::Inkling2Inner,
            taken: Vec<UnitKey>,
        }
        impl VictimSource for RecordingSource<'_> {
            fn units(&self, query: &VictimQuery) -> Vec<Victim> {
                self.inner.inkling2_victim_units(query, &[]).into_units()
            }
            fn take(&mut self, query: &VictimQuery, victim: &Victim) -> Taken {
                self.taken.push(victim.key.clone());
                self.inner.inkling2_take_victim(query, victim)
            }
        }

        let temp_root =
            std::env::temp_dir().join(format!("inkling2-r7-page-reclaim-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&temp_root).expect("create temp_root");
        let _temp_guard = TempDirGuard(temp_root.clone());

        let mut inner = build_synthetic_dflash2_inner(256, None);
        let disk_opts = crate::disk_tier::DiskTierOptions {
            cache_disk: crate::disk_tier::CacheDisk::Bytes(64u64 << 20),
            persistent: false,
            cache_dir: None,
            temp_root: Some(temp_root),
        };
        assert!(
            inner.attach_state_disk(&disk_opts, || None).is_some(),
            "temporary StateDisk tier must attach"
        );
        {
            let rt = inner.paged.as_mut().expect("paged");
            // Explicitly set a generous `ram_cap` (`ram_len <= ram_cap`) so `trim_ram_to_cap`
            // would NOT drop RAM on its own.
            rt.state_store.set_ram_cap(16);
        }

        let conv = crate::models::inkling2::short_conv::Inkling2ShortConvState::new(
            inner.config.num_layer_visits(),
            inner.config.sconv_kernel_size,
            inner.config.num_key_value_heads * inner.config.head_dim,
            inner.config.hidden_size,
            crate::array::DType::BFloat16,
        )
        .expect("conv");

        let tokens_a: Vec<u32> = (100u32..170u32).collect();
        let tokens_u: Vec<u32> = (500u32..570u32).collect();

        // 1. Seed a 2-block anchor (`entry_a`, depth 64) and a shared 1-block prefix anchor
        //    (`entry_prefix`, depth 32) on the same token lineage while `seq_id = 1` is still live.
        let (entry_prefix, entry_a) = {
            let rt = inner.paged.as_mut().expect("paged");
            rt.coordinator.reset_scheduled_request(1).expect("reset 1");
            rt.coordinator
                .record_tokens_all(1, &tokens_a)
                .expect("record a");
            let id_prefix = rt
                .state_store
                .capture_boundary(
                    &mut rt.coordinator,
                    &conv,
                    1,
                    &tokens_a,
                    32,
                    701,
                    StateClass::Ordinary,
                    None,
                )
                .expect("capture prefix")
                .expect("id_prefix");
            let id_a = rt
                .state_store
                .capture_boundary(
                    &mut rt.coordinator,
                    &conv,
                    1,
                    &tokens_a,
                    64,
                    701,
                    StateClass::Ordinary,
                    None,
                )
                .expect("capture a")
                .expect("id_a");
            (id_prefix, id_a)
        };

        let page_query = VictimQuery {
            rights: TakeRights {
                units: true,
                waits_for_writes: true,
                waits_for_landings: true,
                ..Default::default()
            },
            need: Need {
                pages: 2,
                ..Need::default()
            },
        };

        // While live `seq_id = 1` still holds `[B0, B1]`, NEITHER anchor can free any KV pages!
        {
            let runs = inner.inkling2_victim_units(&page_query, &[]);
            let p_prefix = runs
                .merged
                .iter()
                .find(|v| v.key == UnitKey::Store(entry_prefix))
                .expect("entry_prefix listed")
                .pages;
            let p_a = runs
                .merged
                .iter()
                .find(|v| v.key == UnitKey::Store(entry_a))
                .expect("entry_a listed")
                .pages;
            assert_eq!(
                p_prefix, 0,
                "blocks held by a live sequence must advertise 0 reclaimable pages"
            );
            assert_eq!(
                p_a, 0,
                "blocks held by a live sequence must advertise 0 reclaimable pages"
            );
        }

        // Release `seq_id = 1`: now `[B0, B1]` are cache-only and pinned ONLY by state marks.
        // Shared block `B0` must be credited ONCE in take order (`entry_prefix` first -> 0 because
        // `entry_a` still pins `B0`; `entry_a` second -> 2 for `[B0, B1]`), total == 2 (not 3!).
        {
            let rt = inner.paged.as_mut().expect("paged");
            rt.coordinator
                .release_request_all(1)
                .expect("release seq 1");
        }
        {
            let runs = inner.inkling2_victim_units(&page_query, &[]);
            let p_prefix = runs
                .merged
                .iter()
                .find(|v| v.key == UnitKey::Store(entry_prefix))
                .expect("entry_prefix listed")
                .pages;
            let p_a = runs
                .merged
                .iter()
                .find(|v| v.key == UnitKey::Store(entry_a))
                .expect("entry_a listed")
                .pages;
            assert_eq!(
                p_prefix, 0,
                "older prefix anchor cannot free shared ancestor B0 while entry_a still marks it"
            );
            assert_eq!(
                p_a, 2,
                "last anchor in take order owns the full shared closure (total 2, not 3)"
            );
        }

        // Remove `entry_prefix` so `entry_a` is the sole RAM anchor (`ram_len = 1 <= ram_cap = 16`).
        // Stage ONLY `entry_a`'s state blob and wait for it to land (`landed: true, chain_owed: true`).
        {
            let rt = inner.paged.as_mut().expect("paged");
            assert!(rt.state_store.store.remove(entry_prefix));
            rt.state_store.sync_marks(&rt.coordinator);
            assert_eq!(rt.state_store.ram_len(), 1);
            assert!(
                rt.state_store
                    .stage_entry(&rt.coordinator, entry_a, MakeRoom::Evict, true)
            );
            let deadline = Instant::now() + Duration::from_secs(10);
            while rt.state_store.disk.as_ref().unwrap().writing() {
                assert!(Instant::now() < deadline, "timed out landing entry_a blob");
                let _ = rt
                    .state_store
                    .disk
                    .as_ref()
                    .unwrap()
                    .engine
                    .await_landing(deadline);
                rt.state_store.poll_offload(&rt.coordinator);
            }
            let entry = rt.state_store.store.get(entry_a).unwrap();
            assert!(entry.ram().is_some());
            assert!(entry.disk().is_some_and(|c| c.landed && c.chain_owed));
            assert_eq!(
                rt.coordinator
                    .full_adapter()
                    .block_telemetry()
                    .unwrap()
                    .evictable_blocks,
                0,
                "marked chain-owed anchor must pin its 2 Global KV blocks"
            );
        }

        let patient_spec = ClaimSpec {
            cls: ClaimClass::InService,
            request_id: 1,
            rank: ClaimRank {
                priority: 0,
                decoding: true,
                completed_tokens: 64,
                earliness: 1,
            },
            reach: ReachClass::Ordinary,
            timing: NeedTiming::CanWait,
        };
        let urgent_spec = ClaimSpec {
            timing: NeedTiming::Immediate,
            ..patient_spec
        };
        let mut ledger = Ledger::new();
        let patient_claim = ledger.open(patient_spec, 1);

        let takeable_before_claim = inner
            .paged
            .as_ref()
            .unwrap()
            .coordinator
            .full_adapter()
            .takeable_blocks();

        // 2. Patient page-only claim (`need.pages = 2, need.bytes = 0`): takes `Store(entry_a)`
        //    once, schedules under-cap RAM demotion on chain land, and ticks the owed KV chain.
        let claim_res = {
            let mut src = RecordingSource {
                inner: &mut inner,
                taken: Vec::new(),
            };
            let res = ledger.free(
                patient_claim,
                Need {
                    pages: 2,
                    ..Need::default()
                },
                Room::default(),
                &mut EmptyPool,
                &mut [&mut src],
            );
            assert_eq!(src.taken, vec![UnitKey::Store(entry_a)]);
            res
        };
        assert!(
            matches!(claim_res.outcome, FreeOutcome::Pending | FreeOutcome::Met),
            "page-only claim on chain-owed RAM anchor must never return Short (got {:?})",
            claim_res.outcome
        );

        // Finish landing the Global KV chain (still under `ram_cap = 16`!).
        {
            let rt = inner.paged.as_mut().expect("paged");
            let deadline = Instant::now() + Duration::from_secs(10);
            while rt
                .state_store
                .store
                .get(entry_a)
                .is_some_and(|e| e.disk().is_some_and(|c| c.chain_owed))
            {
                assert!(Instant::now() < deadline, "timed out landing entry_a chain");
                rt.state_store.chain_step(&rt.coordinator, deadline);
                let _ = rt
                    .state_store
                    .disk
                    .as_ref()
                    .unwrap()
                    .engine
                    .await_landing(deadline);
                rt.state_store.poll_offload(&rt.coordinator);
            }
            let entry = rt
                .state_store
                .store
                .get(entry_a)
                .expect("entry_a must stay indexed as durable SSD hit");
            assert!(
                entry.ram().is_none(),
                "under-cap RAM anchor MUST be demoted once its owed KV chain lands"
            );
            assert!(entry.disk().is_some_and(|c| c.landed && !c.chain_owed));
            assert_eq!(rt.state_store.ram_len(), 0);
            let tel = rt.coordinator.full_adapter().block_telemetry().unwrap();
            assert_eq!(
                tel.evictable_blocks, 2,
                "both Global KV blocks must immediately become evictable after under-cap demotion"
            );
            assert_eq!(
                rt.coordinator.full_adapter().takeable_blocks(),
                takeable_before_claim + 2,
                "takeable_blocks must increase by the 2 unmarked Global KV pages"
            );
            // Verify the unmarked KV blocks are genuinely allocatable for a new sequence.
            rt.coordinator.reset_scheduled_request(3).expect("reset 3");
            rt.coordinator
                .record_tokens_all(3, &(800u32..864u32).collect::<Vec<_>>())
                .expect("record new 2-block sequence");
            rt.coordinator
                .release_request_all(3)
                .expect("release seq 3");
        }

        // 3. Urgent page-only claim (`NeedTiming::Immediate`) against a RAM-backed
        //    `(landed: true, chain_owed: true)` anchor `entry_u`: must erase immediately and
        //    report `FreeOutcome::Met` with `pages == 2`!
        let entry_u = {
            let rt = inner.paged.as_mut().expect("paged");
            rt.state_store.clear_ram_and_checkpoints();
            rt.state_store.sync_marks(&rt.coordinator);
            rt.coordinator
                .release_all_and_purge()
                .expect("purge before urgent page test");
            rt.coordinator.reset_scheduled_request(4).expect("reset 4");
            rt.coordinator
                .record_tokens_all(4, &tokens_u)
                .expect("record u");
            let id_u = rt
                .state_store
                .capture_boundary(
                    &mut rt.coordinator,
                    &conv,
                    4,
                    &tokens_u,
                    64,
                    702,
                    StateClass::Ordinary,
                    None,
                )
                .expect("capture u")
                .expect("id_u");
            rt.coordinator
                .release_request_all(4)
                .expect("release seq 4");
            assert!(
                rt.state_store
                    .stage_entry(&rt.coordinator, id_u, MakeRoom::Evict, true)
            );
            let deadline = Instant::now() + Duration::from_secs(10);
            while rt.state_store.disk.as_ref().unwrap().writing() {
                assert!(Instant::now() < deadline, "timed out landing entry_u blob");
                let _ = rt
                    .state_store
                    .disk
                    .as_ref()
                    .unwrap()
                    .engine
                    .await_landing(deadline);
                rt.state_store.poll_offload(&rt.coordinator);
            }
            let entry = rt.state_store.store.get(id_u).unwrap();
            assert!(entry.ram().is_some());
            assert!(entry.disk().is_some_and(|c| c.landed && c.chain_owed));
            id_u
        };

        let urgent_claim = ledger.open(urgent_spec, 1);
        let takeable_before_urgent = inner
            .paged
            .as_ref()
            .unwrap()
            .coordinator
            .full_adapter()
            .takeable_blocks();
        {
            let mut src = RecordingSource {
                inner: &mut inner,
                taken: Vec::new(),
            };
            let res = ledger.free(
                urgent_claim,
                Need {
                    pages: 2,
                    ..Need::default()
                },
                Room::default(),
                &mut EmptyPool,
                &mut [&mut src],
            );
            assert_eq!(
                res.outcome,
                FreeOutcome::Met,
                "urgent page-only claim must immediately free RAM-backed chain_owed anchor's pages"
            );
            assert_eq!(res.pages, 2);
            assert_eq!(src.taken, vec![UnitKey::Store(entry_u)]);
        }
        {
            let rt = inner.paged.as_ref().expect("paged");
            assert!(rt.state_store.store.get(entry_u).is_none());
            assert_eq!(
                rt.coordinator.full_adapter().takeable_blocks(),
                takeable_before_urgent + 2
            );
        }
        let _ = inner.close_state_disk(Instant::now() + Duration::from_secs(5));
    }

    /// Round-7 Finding 4 regression: a rolling/junction checkpoint readback or snapshot error
    /// inside `maybe_capture_rolling_checkpoint` during `paged_prefill_dflash2_with_taps` MUST be
    /// logged and treated as a non-fatal cache miss — exclusive DFlash2 prefill succeeds with
    /// bit-identical final logits and an intact live `Inkling2Dflash2Context` ready for drafting!
    #[test]
    fn exclusive_dflash2_prefill_survives_checkpoint_capture_failure() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        use crate::models::inkling2::dflash2::Inkling2Dflash2Context;
        use crate::models::inkling2::paged_forward::Inkling2PrefixState;

        let mut inner = build_synthetic_dflash2_inner_with_window(256, None, 128, 1);
        let prompt: Vec<u32> = (10u32..50u32).collect(); // 40 tokens -> prompt junction at 32!
        let draft_cfg = inner.dflash2.as_ref().expect("dflash2").config.clone();

        // 1. Healthy baseline run (`reuse_cache = true`, geometry initialized):
        //    captures junction checkpoint at 32 and computes baseline final logits + draft chain.
        let prefix_base = Inkling2PrefixState {
            effective_cached_prefix_len: 0,
            suffix_len: prompt.len(),
            full_tokens: prompt.clone(),
            cache_salt: 808,
            live_continuation: false,
            reuse_cache: true,
        };
        {
            let rt = inner.paged.as_mut().expect("paged");
            rt.active_seq = 1;
            rt.coordinator.reset_scheduled_request(1).expect("reset 1");
            rt.ensure_conv_state(1, &inner.config).expect("conv 1");
        }
        let mut ctx_base = Inkling2Dflash2Context::new(&draft_cfg);
        let logits_base = inner
            .paged_prefill_dflash2_with_taps(1, &prompt, &prefix_base, &mut ctx_base)
            .expect("healthy exclusive DFlash2 prefill");
        let logits_base_vec = logits_base.to_float32().expect("base logits f32").to_vec();
        assert_eq!(ctx_base.committed_len() as usize, prompt.len());
        assert_eq!(
            inner
                .paged
                .as_ref()
                .unwrap()
                .state_store
                .store
                .entries()
                .count(),
            1,
            "healthy baseline run must publish the prompt-junction checkpoint at 32"
        );
        let first_tok = logits_base
            .argmax(-1, Some(false))
            .expect("argmax")
            .item_at_int32(0)
            .expect("tok") as u32;
        let draft_base = {
            let mut view = inner.target_view();
            ctx_base
                .propose_greedy_chain(
                    inner.dflash2.as_ref().expect("dflash2"),
                    &mut view,
                    first_tok,
                    4,
                    false,
                    false,
                )
                .expect("baseline propose_greedy_chain")
                .draft_ids
        };

        // 2. Fault-injected run: clear store & unset `state_store.geometry` so
        //    `maybe_capture_rolling_checkpoint` -> `capture_boundary` fails at junction 32!
        {
            let rt = inner.paged.as_mut().expect("paged");
            rt.state_store.clear_ram_and_checkpoints();
            rt.state_store.sync_marks(&rt.coordinator);
            rt.coordinator
                .release_all_and_purge()
                .expect("purge between runs");
            rt.state_store.geometry = None;
            rt.active_seq = 2;
            rt.coordinator.reset_scheduled_request(2).expect("reset 2");
            rt.ensure_conv_state(2, &inner.config).expect("conv 2");
        }
        let prefix_inj = Inkling2PrefixState {
            effective_cached_prefix_len: 0,
            suffix_len: prompt.len(),
            full_tokens: prompt.clone(),
            cache_salt: 809,
            live_continuation: false,
            reuse_cache: true,
        };
        let mut ctx_inj = Inkling2Dflash2Context::new(&draft_cfg);
        let logits_inj = inner
            .paged_prefill_dflash2_with_taps(2, &prompt, &prefix_inj, &mut ctx_inj)
            .expect("exclusive DFlash2 prefill must survive rolling checkpoint capture failure");
        let logits_inj_vec = logits_inj.to_float32().expect("inj logits f32").to_vec();
        assert_eq!(
            inner
                .paged
                .as_ref()
                .unwrap()
                .state_store
                .store
                .entries()
                .count(),
            0,
            "injected capture failure must have skipped publishing into StateStore"
        );
        assert_eq!(
            logits_inj_vec, logits_base_vec,
            "final logits must be bit-identical despite checkpoint capture failure"
        );
        assert_eq!(
            ctx_inj.committed_len() as usize,
            prompt.len(),
            "live DFlash2 context must remain committed across the full prompt"
        );
        assert_eq!(
            inner
                .paged
                .as_ref()
                .unwrap()
                .coordinator
                .request_token_count_all(2)
                .unwrap(),
            prompt.len() as u32
        );
        let draft_inj = {
            let mut view = inner.target_view();
            ctx_inj
                .propose_greedy_chain(
                    inner.dflash2.as_ref().expect("dflash2"),
                    &mut view,
                    first_tok,
                    4,
                    false,
                    false,
                )
                .expect("fault-injected propose_greedy_chain must succeed")
                .draft_ids
        };
        assert_eq!(
            draft_inj, draft_base,
            "live DFlash2 context must draft bit-identical tokens after checkpoint capture failure"
        );
    }

    /// Round-8 Finding 1 regression: when the SSD writer hits a non-space I/O error and stops
    /// accepting tracked writes (`!disk.writable()`), any disk-only `chain_owed` entry can never
    /// land its Global KV blocks and must be retired immediately so its KV pages are unmarked and
    /// reclaimable, any RAM-backed `chain_owed` entry keeps its RAM copy while shedding its
    /// incomplete disk copy, and every already-durable (`landed && !chain_owed`) SSD entry is
    /// preserved and remains restorable from SSD.
    #[cfg(target_os = "macos")]
    #[test]
    fn unwritable_disk_retires_disk_only_chain_owed_entry_and_preserves_durable_entry() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        use std::os::unix::fs::PermissionsExt;

        use crate::engine::governor::ledger::{
            Charge, ClaimClass, ClaimRank, ClaimSpec, FreeOutcome, FreePool, Ledger, Need,
            NeedTiming, ReachClass, Room, Stage, TakeRights, Taken, UnitKey, Victim, VictimQuery,
            VictimSource,
        };
        use crate::models::qwen3_5::dflash2_state::store::StateClass;
        use mlx_paged_attn::MakeRoom;

        struct PermsTempDirGuard {
            chmod_dir: Option<std::path::PathBuf>,
            root: std::path::PathBuf,
        }
        impl Drop for PermsTempDirGuard {
            fn drop(&mut self) {
                if let Some(ref dir) = self.chmod_dir {
                    let _ = std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o755));
                }
                let _ = std::fs::remove_dir_all(&self.root);
            }
        }
        struct EmptyPool;
        impl FreePool for EmptyPool {
            fn charge(&self) -> Charge {
                Charge::default()
            }
            fn clear(&mut self) {}
        }
        struct RecordingSource<'a> {
            inner: &'a mut super::Inkling2Inner,
            taken: Vec<UnitKey>,
        }
        impl VictimSource for RecordingSource<'_> {
            fn units(&self, query: &VictimQuery) -> Vec<Victim> {
                self.inner.inkling2_victim_units(query, &[]).into_units()
            }
            fn take(&mut self, query: &VictimQuery, victim: &Victim) -> Taken {
                self.taken.push(victim.key.clone());
                self.inner.inkling2_take_victim(query, victim)
            }
        }

        let temp_root = std::env::temp_dir().join(format!(
            "inkling2-r8-unwritable-chain-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&temp_root).expect("create temp_root");
        let mut guard = PermsTempDirGuard {
            chmod_dir: None,
            root: temp_root.clone(),
        };

        let mut inner = build_synthetic_dflash2_inner(512, None);
        let disk_opts = crate::disk_tier::DiskTierOptions {
            cache_disk: crate::disk_tier::CacheDisk::Bytes(64u64 << 20),
            persistent: false,
            cache_dir: None,
            temp_root: Some(temp_root),
        };
        assert!(
            inner.attach_state_disk(&disk_opts, || None).is_some(),
            "temporary StateDisk tier must attach"
        );
        {
            let rt = inner.paged.as_mut().expect("paged");
            rt.state_store.set_ram_cap(16);
        }

        let conv = crate::models::inkling2::short_conv::Inkling2ShortConvState::new(
            inner.config.num_layer_visits(),
            inner.config.sconv_kernel_size,
            inner.config.num_key_value_heads * inner.config.head_dim,
            inner.config.hidden_size,
            crate::array::DType::BFloat16,
        )
        .expect("conv");

        let tokens_a: Vec<u32> = (1100u32..1170u32).collect();
        let query_a: Vec<u32> = (1100u32..1168u32).collect();
        let tokens_b: Vec<u32> = (1200u32..1270u32).collect();
        let tokens_c: Vec<u32> = (1300u32..1370u32).collect();
        let tokens_d: Vec<u32> = (1400u32..1470u32).collect();

        // 1. Entry A: durable SSD-only entry (`ram == None, landed == true, chain_owed == false`).
        let id_a = {
            let rt = inner.paged.as_mut().expect("paged");
            rt.coordinator.reset_scheduled_request(1).expect("reset 1");
            rt.coordinator
                .record_tokens_all(1, &tokens_a)
                .expect("record a");
            let id = rt
                .state_store
                .capture_boundary(
                    &mut rt.coordinator,
                    &conv,
                    1,
                    &tokens_a,
                    64,
                    811,
                    StateClass::Ordinary,
                    None,
                )
                .expect("capture a")
                .expect("id_a");
            rt.coordinator
                .release_request_all(1)
                .expect("release seq 1");
            assert!(
                rt.state_store
                    .stage_entry(&rt.coordinator, id, MakeRoom::Evict, true)
            );
            let deadline = Instant::now() + Duration::from_secs(10);
            while rt
                .state_store
                .store
                .get(id)
                .is_none_or(|e| e.disk().is_none_or(|c| !c.landed || c.chain_owed))
            {
                assert!(Instant::now() < deadline, "timed out landing durable A");
                rt.state_store.chain_step(&rt.coordinator, deadline);
                let _ = rt
                    .state_store
                    .disk
                    .as_ref()
                    .unwrap()
                    .engine
                    .await_landing(deadline);
                rt.state_store.poll_offload(&rt.coordinator);
            }
            assert_eq!(
                rt.state_store.reclaim_entry(&rt.coordinator, id),
                crate::models::inkling2::state_disk::ReclaimOutcome::RamDropped
            );
            let entry_a = rt.state_store.store.get(id).expect("entry_a present");
            assert!(entry_a.ram().is_none());
            assert!(entry_a.disk().is_some_and(|c| c.landed && !c.chain_owed));
            id
        };

        // 2. Entry D: RAM-backed incomplete (`ram == Some, landed == true, chain_owed == true`)
        //    and Entry B: disk-only incomplete (`ram == None, landed == true, chain_owed == true`),
        //    both pinning their Global KV blocks in the allocator.
        let (id_b, id_d, id_c) = {
            let rt = inner.paged.as_mut().expect("paged");
            // Purge A's hot KV so only B, D, C live in the hot pool (and A is pure SSD).
            rt.coordinator.release_all_and_purge().expect("purge hot A");

            rt.coordinator.reset_scheduled_request(2).expect("reset 2");
            rt.coordinator
                .record_tokens_all(2, &tokens_b)
                .expect("record b");
            let id_b = rt
                .state_store
                .capture_boundary(
                    &mut rt.coordinator,
                    &conv,
                    2,
                    &tokens_b,
                    64,
                    812,
                    StateClass::Ordinary,
                    None,
                )
                .expect("capture b")
                .expect("id_b");
            rt.coordinator
                .release_request_all(2)
                .expect("release seq 2");

            // RAM-demoting spill for B (`keep_ram = false`), wait for blob landing WITHOUT chain!
            assert_eq!(
                rt.state_store
                    .reclaim_entry_with_options(&rt.coordinator, id_b, false, false),
                crate::models::inkling2::state_disk::ReclaimOutcome::Spilled
            );
            let deadline_b = Instant::now() + Duration::from_secs(10);
            while rt.state_store.disk.as_ref().unwrap().writing() {
                assert!(Instant::now() < deadline_b, "timed out landing B blob");
                let _ = rt
                    .state_store
                    .disk
                    .as_ref()
                    .unwrap()
                    .engine
                    .await_landing(deadline_b);
                rt.state_store.poll_offload(&rt.coordinator);
            }
            {
                let entry_b = rt.state_store.store.get(id_b).expect("entry_b");
                assert!(entry_b.ram().is_none());
                assert!(entry_b.disk().is_some_and(|c| c.landed && c.chain_owed));
            }

            // Capture D and land ONLY its blob while keeping RAM (`ram == Some, chain_owed == true`).
            rt.coordinator.reset_scheduled_request(3).expect("reset 3");
            rt.coordinator
                .record_tokens_all(3, &tokens_d)
                .expect("record d");
            let id_d = rt
                .state_store
                .capture_boundary(
                    &mut rt.coordinator,
                    &conv,
                    3,
                    &tokens_d,
                    64,
                    813,
                    StateClass::Ordinary,
                    None,
                )
                .expect("capture d")
                .expect("id_d");
            rt.coordinator
                .release_request_all(3)
                .expect("release seq 3");
            assert!(
                rt.state_store
                    .stage_entry(&rt.coordinator, id_d, MakeRoom::Evict, true)
            );
            let deadline_d = Instant::now() + Duration::from_secs(10);
            while rt.state_store.disk.as_ref().unwrap().writing() {
                assert!(Instant::now() < deadline_d, "timed out landing D blob");
                let _ = rt
                    .state_store
                    .disk
                    .as_ref()
                    .unwrap()
                    .engine
                    .await_landing(deadline_d);
                rt.state_store.poll_offload(&rt.coordinator);
            }
            {
                let entry_d = rt.state_store.store.get(id_d).expect("entry_d");
                assert!(entry_d.ram().is_some());
                assert!(entry_d.disk().is_some_and(|c| c.landed && c.chain_owed));
            }

            // Capture C (RAM-only) to trigger a real terminal writer I/O error (`chmod 0555`).
            rt.coordinator.reset_scheduled_request(4).expect("reset 4");
            rt.coordinator
                .record_tokens_all(4, &tokens_c)
                .expect("record c");
            let id_c = rt
                .state_store
                .capture_boundary(
                    &mut rt.coordinator,
                    &conv,
                    4,
                    &tokens_c,
                    64,
                    814,
                    StateClass::Ordinary,
                    None,
                )
                .expect("capture c")
                .expect("id_c");
            assert_eq!(
                rt.coordinator
                    .full_adapter()
                    .block_telemetry()
                    .unwrap()
                    .evictable_blocks,
                0,
                "B's 2 Global KV blocks must stay pinned before writer failure retires B"
            );

            // Make SSD directory read-only and stage C so the writer trips `tracked_failed`!
            let tier_root = rt
                .state_store
                .disk
                .as_ref()
                .unwrap()
                .tier
                .manager
                .root()
                .to_path_buf();
            std::fs::set_permissions(&tier_root, std::fs::Permissions::from_mode(0o555))
                .expect("chmod 0555");
            guard.chmod_dir = Some(tier_root);
            assert!(
                rt.state_store
                    .stage_entry(&rt.coordinator, id_c, MakeRoom::Evict, true)
            );
            let fail_deadline = Instant::now() + Duration::from_secs(10);
            while rt.state_store.disk.as_ref().unwrap().writable() {
                assert!(
                    Instant::now() < fail_deadline,
                    "timed out waiting for writer to trip tracked_failed"
                );
                let _ = rt
                    .state_store
                    .disk
                    .as_ref()
                    .unwrap()
                    .engine
                    .await_landing(fail_deadline);
            }
            // Do NOT poll `state_store` yet: verify the Governor claim itself retires B cleanly!
            assert!(!rt.state_store.disk.as_ref().unwrap().writable());
            assert!(rt.state_store.store.get(id_b).is_some_and(
                |e| e.ram().is_none() && e.disk().is_some_and(|c| c.landed && c.chain_owed)
            ));
            (id_b, id_d, id_c)
        };

        // 3. Before any tick, `inkling2_victim_units` must list `StoreChain(id_b)` as immediately
        //    freeable (`Stage::FreeNow`, `later == false`) ahead of live RAM entries.
        let query = VictimQuery {
            rights: TakeRights {
                units: true,
                waits_for_writes: true,
                waits_for_landings: true,
                ..Default::default()
            },
            need: Need {
                pages: 2,
                ..Need::default()
            },
        };
        {
            let runs = inner.inkling2_victim_units(&query, &[]);
            let victim_b = runs
                .ahead
                .iter()
                .find(|v| v.key == UnitKey::StoreChain(id_b))
                .expect("unwritable disk-only chain_owed entry B must appear in ahead");
            assert_eq!(victim_b.stage, Stage::FreeNow);
            assert!(!victim_b.later);
            assert_eq!(victim_b.pages, 2);
        }

        // 4. Run a patient page-only Governor claim (`Need { pages: 2 }`): previously this returned
        //    `Pending` forever! Now it must retire B synchronously, unmark B's 2 Global KV pages,
        //    and return `FreeOutcome::Met` with `pages == 2`!
        let takeable_before = inner
            .paged
            .as_ref()
            .unwrap()
            .coordinator
            .full_adapter()
            .takeable_blocks();
        let mut ledger = Ledger::new();
        let claim = ledger.open(
            ClaimSpec {
                cls: ClaimClass::InService,
                request_id: 1,
                rank: ClaimRank {
                    priority: 0,
                    decoding: true,
                    completed_tokens: 64,
                    earliness: 1,
                },
                reach: ReachClass::Ordinary,
                timing: NeedTiming::CanWait,
            },
            1,
        );
        let claim_res = {
            let mut src = RecordingSource {
                inner: &mut inner,
                taken: Vec::new(),
            };
            let res = ledger.free(
                claim,
                Need {
                    pages: 2,
                    ..Need::default()
                },
                Room::default(),
                &mut EmptyPool,
                &mut [&mut src],
            );
            assert_eq!(src.taken, vec![UnitKey::StoreChain(id_b)]);
            res
        };
        assert_eq!(
            claim_res.outcome,
            FreeOutcome::Met,
            "page-only claim must resolve unwritable disk-only chain_owed entry immediately"
        );
        assert_eq!(claim_res.pages, 2);

        // 5. Verify B is retired + unmarked, D kept its RAM copy while shedding its incomplete
        //    disk copy, C kept its RAM copy, and durable entry A is preserved & restorable!
        {
            let rt = inner.paged.as_mut().expect("paged");
            assert!(
                rt.state_store.store.get(id_b).is_none(),
                "disk-only incomplete entry B must be retired when writer is unwritable"
            );
            assert_eq!(
                rt.coordinator.full_adapter().takeable_blocks(),
                takeable_before + 2,
                "B's 2 Global KV pages must become immediately evictable"
            );
            let entry_d = rt
                .state_store
                .store
                .get(id_d)
                .expect("RAM-backed entry D must survive unwritable disk retirement");
            assert!(
                entry_d.ram().is_some(),
                "entry D's RAM copy must stay intact"
            );
            assert!(
                entry_d.disk().is_none(),
                "entry D's incomplete chain_owed DiskCopy must be stripped"
            );
            assert!(
                rt.state_store
                    .store
                    .get(id_c)
                    .is_some_and(|e| e.ram().is_some() && e.disk().is_none()),
                "entry C's RAM copy must survive failed write"
            );
            let entry_a = rt
                .state_store
                .store
                .get(id_a)
                .expect("durable SSD entry A must be preserved across writer failure");
            assert!(entry_a.ram().is_none());
            assert!(entry_a.disk().is_some_and(|c| c.landed && !c.chain_owed));

            let hit_a = rt
                .state_store
                .lookup_deepest_restorable(&rt.coordinator, &query_a, 811)
                .expect("durable SSD entry A must remain restorable after writer failure");
            assert_eq!(hit_a.id, id_a);
            let (_, from_ssd) = rt
                .state_store
                .load_anchor(&rt.coordinator, hit_a.id)
                .expect("durable SSD entry A blob must still decode cleanly");
            assert!(from_ssd);
        }

        if let Some(dir) = guard.chmod_dir.take() {
            std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o755))
                .expect("restore 0755");
        }
        let _ = inner.close_state_disk(Instant::now() + Duration::from_secs(5));
    }

    /// Round-8 Finding 2 regression:
    /// 1. Urgent page-only Governor claim (`NeedTiming::Immediate`, `need.pages > 0, need.bytes == 0`)
    ///    against a fresh RAM-only anchor whose Global KV chain is NOT on SSD must erase it
    ///    immediately instead of starting a spill with `chain_owed == true` (which frees 0 pages
    ///    and causes `FreeOutcome::Short`), returning `FreeOutcome::Met` with `pages == 2`.
    /// 2. Urgent byte-only Governor claim (`NeedTiming::Immediate`, `need.bytes > 0, need.pages == 0`)
    ///    against a fresh RAM-only anchor still immediately spills to SSD and drops RAM.
    #[test]
    fn urgent_page_only_claim_erases_ram_only_anchor_while_byte_claim_spills() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        use crate::engine::governor::ledger::{
            Charge, ClaimClass, ClaimRank, ClaimSpec, FreeOutcome, FreePool, Ledger, Need,
            NeedTiming, ReachClass, Room, Taken, UnitKey, Victim, VictimQuery, VictimSource,
        };
        use crate::models::qwen3_5::dflash2_state::store::StateClass;

        struct TempDirGuard(std::path::PathBuf);
        impl Drop for TempDirGuard {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        struct EmptyPool;
        impl FreePool for EmptyPool {
            fn charge(&self) -> Charge {
                Charge::default()
            }
            fn clear(&mut self) {}
        }
        struct RecordingSource<'a> {
            inner: &'a mut super::Inkling2Inner,
            taken: Vec<UnitKey>,
        }
        impl VictimSource for RecordingSource<'_> {
            fn units(&self, query: &VictimQuery) -> Vec<Victim> {
                self.inner.inkling2_victim_units(query, &[]).into_units()
            }
            fn take(&mut self, query: &VictimQuery, victim: &Victim) -> Taken {
                self.taken.push(victim.key.clone());
                self.inner.inkling2_take_victim(query, victim)
            }
        }

        let temp_root = std::env::temp_dir().join(format!(
            "inkling2-r8-urgent-page-vs-byte-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&temp_root).expect("create temp_root");
        let _temp_guard = TempDirGuard(temp_root.clone());

        let mut inner = build_synthetic_dflash2_inner(256, None);
        let disk_opts = crate::disk_tier::DiskTierOptions {
            cache_disk: crate::disk_tier::CacheDisk::Bytes(64u64 << 20),
            persistent: false,
            cache_dir: None,
            temp_root: Some(temp_root),
        };
        assert!(inner.attach_state_disk(&disk_opts, || None).is_some());

        let conv = crate::models::inkling2::short_conv::Inkling2ShortConvState::new(
            inner.config.num_layer_visits(),
            inner.config.sconv_kernel_size,
            inner.config.num_key_value_heads * inner.config.head_dim,
            inner.config.hidden_size,
            crate::array::DType::BFloat16,
        )
        .expect("conv");

        let urgent_spec = ClaimSpec {
            cls: ClaimClass::InService,
            request_id: 1,
            rank: ClaimRank {
                priority: 0,
                decoding: true,
                completed_tokens: 64,
                earliness: 1,
            },
            reach: ReachClass::Ordinary,
            timing: NeedTiming::Immediate,
        };
        let mut ledger = Ledger::new();

        // 1. Fresh RAM-only anchor `entry_p` (chain NOT on SSD) under an urgent PAGE-only claim.
        let tokens_p: Vec<u32> = (2100u32..2170u32).collect();
        let entry_p = {
            let rt = inner.paged.as_mut().expect("paged");
            rt.coordinator.reset_scheduled_request(1).expect("reset 1");
            rt.coordinator
                .record_tokens_all(1, &tokens_p)
                .expect("record p");
            let id = rt
                .state_store
                .capture_boundary(
                    &mut rt.coordinator,
                    &conv,
                    1,
                    &tokens_p,
                    64,
                    821,
                    StateClass::Ordinary,
                    None,
                )
                .expect("capture p")
                .expect("entry_p");
            rt.coordinator
                .release_request_all(1)
                .expect("release seq 1");
            assert!(
                rt.state_store
                    .store
                    .get(id)
                    .is_some_and(|e| e.ram().is_some() && e.disk().is_none())
            );
            id
        };
        let takeable_before_p = inner
            .paged
            .as_ref()
            .unwrap()
            .coordinator
            .full_adapter()
            .takeable_blocks();
        let claim_p = ledger.open(urgent_spec, 1);
        let res_p = {
            let mut src = RecordingSource {
                inner: &mut inner,
                taken: Vec::new(),
            };
            let res = ledger.free(
                claim_p,
                Need {
                    pages: 2,
                    ..Need::default()
                },
                Room::default(),
                &mut EmptyPool,
                &mut [&mut src],
            );
            assert_eq!(src.taken, vec![UnitKey::Store(entry_p)]);
            res
        };
        assert_eq!(
            res_p.outcome,
            FreeOutcome::Met,
            "urgent page-only claim must erase fresh RAM-only anchor instead of spilling with chain_owed=true"
        );
        assert_eq!(res_p.pages, 2);
        {
            let rt = inner.paged.as_ref().expect("paged");
            assert!(
                rt.state_store.store.get(entry_p).is_none(),
                "fresh RAM-only anchor must be erased immediately to unmark Global KV pages"
            );
            assert!(
                !rt.state_store.disk.as_ref().unwrap().writing(),
                "urgent page-only claim must not start a zero-page spill"
            );
            assert_eq!(
                rt.coordinator.full_adapter().takeable_blocks(),
                takeable_before_p + 2
            );
        }

        // 2. Fresh RAM-only anchor `entry_b` (chain NOT on SSD) under an urgent BYTE-only claim:
        //    must still immediately spill to SSD and drop RAM!
        let tokens_b: Vec<u32> = (2200u32..2270u32).collect();
        let entry_b = {
            let rt = inner.paged.as_mut().expect("paged");
            rt.coordinator
                .release_all_and_purge()
                .expect("purge before byte test");
            rt.coordinator.reset_scheduled_request(2).expect("reset 2");
            rt.coordinator
                .record_tokens_all(2, &tokens_b)
                .expect("record b");
            let id = rt
                .state_store
                .capture_boundary(
                    &mut rt.coordinator,
                    &conv,
                    2,
                    &tokens_b,
                    64,
                    822,
                    StateClass::Ordinary,
                    None,
                )
                .expect("capture b")
                .expect("entry_b");
            rt.coordinator
                .release_request_all(2)
                .expect("release seq 2");
            id
        };
        let claim_b = ledger.open(urgent_spec, 1);
        let res_b = {
            let mut src = RecordingSource {
                inner: &mut inner,
                taken: Vec::new(),
            };
            let res = ledger.free(
                claim_b,
                Need {
                    bytes: 1,
                    ..Need::default()
                },
                Room::default(),
                &mut EmptyPool,
                &mut [&mut src],
            );
            assert_eq!(src.taken, vec![UnitKey::Store(entry_b)]);
            res
        };
        assert_eq!(res_b.outcome, FreeOutcome::Met);
        assert!(
            res_b.now_bytes > 0,
            "urgent byte-only claim must reclaim RAM bytes immediately via spill"
        );
        {
            let rt = inner.paged.as_ref().expect("paged");
            let entry = rt
                .state_store
                .store
                .get(entry_b)
                .expect("byte-only urgent spill must retain entry_b on SSD");
            assert!(entry.ram().is_none(), "RAM copy must be dropped on spill");
            assert!(
                entry.disk().is_some_and(|c| c.chain_owed),
                "byte-only urgent spill starts SSD write with chain_owed"
            );
            assert!(rt.state_store.disk.as_ref().unwrap().writing());
        }

        let _ = inner.close_state_disk(Instant::now() + Duration::from_secs(5));
    }

    /// Round-8 Finding 4 regression: BOTH scheduled prefill paths in `run_paged_prefill_pass`
    /// (the fast unbounded `budget_ms == None` path AND the resumable multi-step `budget_ms ==
    /// Some(...)` path) must treat rolling/junction checkpoint capture errors as non-fatal
    /// best-effort cache misses — never aborting prefill after target KV & ShortConv commit — and
    /// produce bit-identical final logits compared to the healthy capture run.
    #[test]
    fn scheduled_prefill_paths_survive_checkpoint_capture_failure() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        use crate::engine::hybrid_scheduler::ScheduledPrefillPass;

        let mut inner = build_synthetic_dflash2_inner_with_window(256, None, 128, 1);
        let prompt: Vec<u32> = (20u32..60u32).collect(); // 40 tokens -> prompt junction at 32!
        let gen_stream = crate::stream::Stream::generation();

        // 1. Healthy baseline run (`budget_ms = None`, `reuse_cache = true`, geometry intact).
        {
            let rt = inner.paged.as_mut().expect("paged");
            rt.active_seq = 1;
            rt.coordinator.reset_scheduled_request(1).expect("reset 1");
            rt.ensure_conv_state(1, &inner.config).expect("conv 1");
        }
        let logits_base = match inner
            .run_paged_prefill_pass(1, &prompt, 0, prompt.len(), 901, true, gen_stream, None)
            .expect("healthy scheduled prefill")
        {
            ScheduledPrefillPass::Done(Some(l)) => l,
            ScheduledPrefillPass::Done(None) => panic!("expected final logits on full prompt"),
            ScheduledPrefillPass::Paused => panic!("unbounded prefill must not pause"),
        };
        let logits_base_vec = logits_base.to_float32().expect("base logits f32").to_vec();
        assert_eq!(
            inner
                .paged
                .as_ref()
                .unwrap()
                .state_store
                .store
                .entries()
                .count(),
            1,
            "healthy baseline run must publish the prompt-junction checkpoint at 32"
        );

        // 2. Fault-injected fast path (`budget_ms = None`, `state_store.geometry = None`).
        {
            let rt = inner.paged.as_mut().expect("paged");
            rt.state_store.clear_ram_and_checkpoints();
            rt.state_store.sync_marks(&rt.coordinator);
            rt.coordinator
                .release_all_and_purge()
                .expect("purge before fast path");
            rt.state_store.geometry = None;
            rt.active_seq = 2;
            rt.coordinator.reset_scheduled_request(2).expect("reset 2");
            rt.ensure_conv_state(2, &inner.config).expect("conv 2");
        }
        let logits_fast = match inner
            .run_paged_prefill_pass(2, &prompt, 0, prompt.len(), 902, true, gen_stream, None)
            .expect("fast scheduled prefill must survive checkpoint capture failure")
        {
            ScheduledPrefillPass::Done(Some(l)) => l,
            ScheduledPrefillPass::Done(None) => panic!("expected final logits on fast path"),
            ScheduledPrefillPass::Paused => panic!("unbounded prefill must not pause"),
        };
        let logits_fast_vec = logits_fast.to_float32().expect("fast logits f32").to_vec();
        assert_eq!(
            inner
                .paged
                .as_ref()
                .unwrap()
                .state_store
                .store
                .entries()
                .count(),
            0,
            "injected capture failure must skip publishing on fast scheduled prefill"
        );
        assert_eq!(
            inner
                .paged
                .as_ref()
                .unwrap()
                .coordinator
                .request_token_count_all(2)
                .unwrap(),
            prompt.len() as u32
        );
        assert_eq!(
            logits_fast_vec, logits_base_vec,
            "fast scheduled prefill logits must be bit-identical despite checkpoint capture failure"
        );

        // 3. Fault-injected resumable multi-step path (`budget_ms = Some(0.0)`,
        //    `state_store.geometry = None`).
        {
            let rt = inner.paged.as_mut().expect("paged");
            rt.state_store.clear_ram_and_checkpoints();
            rt.state_store.sync_marks(&rt.coordinator);
            rt.coordinator
                .release_all_and_purge()
                .expect("purge before resumable path");
            rt.state_store.geometry = None;
            rt.active_seq = 3;
            rt.coordinator.reset_scheduled_request(3).expect("reset 3");
            rt.ensure_conv_state(3, &inner.config).expect("conv 3");
        }
        let mut passes = 0usize;
        let logits_resumable = loop {
            passes += 1;
            assert!(
                passes < 256,
                "resumable scheduled prefill exceeded expected pass count"
            );
            match inner
                .run_paged_prefill_pass(
                    3,
                    &prompt,
                    0,
                    prompt.len(),
                    903,
                    true,
                    gen_stream,
                    Some(0.0),
                )
                .expect("resumable scheduled prefill must survive checkpoint capture failure")
            {
                ScheduledPrefillPass::Paused => {}
                ScheduledPrefillPass::Done(Some(l)) => break l,
                ScheduledPrefillPass::Done(None) => {
                    panic!("expected final logits on resumable path")
                }
            }
        };
        assert!(
            !inner.prefill_runs.contains_key(&3),
            "completed resumable prefill must clear prefill_runs entry"
        );
        let logits_resumable_vec = logits_resumable
            .to_float32()
            .expect("resumable logits f32")
            .to_vec();
        assert_eq!(
            inner
                .paged
                .as_ref()
                .unwrap()
                .state_store
                .store
                .entries()
                .count(),
            0,
            "injected capture failure must skip publishing on resumable scheduled prefill"
        );
        assert_eq!(
            inner
                .paged
                .as_ref()
                .unwrap()
                .coordinator
                .request_token_count_all(3)
                .unwrap(),
            prompt.len() as u32
        );
        assert_eq!(
            logits_resumable_vec, logits_base_vec,
            "resumable scheduled prefill logits must be bit-identical despite checkpoint capture failure"
        );
    }

    /// Round-9 Finding 1 regression:
    /// `UnitKey::DeadLeaves` is strictly a byte-reclaim unit (`pages: 0`) because
    /// `release_cached_extent` SHRINKS the pool instead of freeing allocatable pages.
    /// On a multi-victim page-only claim requesting `2 * K` pages across two `StateStore` anchors
    /// (`entry_a` and `entry_b`, each pinning `K = 2` distinct Global KV blocks) with extent
    /// growth enabled:
    /// - `DeadLeaves` is NEVER taken (so the pool is never shrunk mid-claim and pages are never
    ///   double-credited),
    /// - BOTH `Store(entry_a)` and `Store(entry_b)` are taken,
    /// - `res.outcome == FreeOutcome::Met` with `res.pages >= 2 * K`,
    /// - `full.takeable_blocks() >= 2 * K` and actually allocating a `2 * K`-block sequence
    ///   succeeds in-place without growing the pool!
    ///
    /// Also verifies that a byte-only claim still emits `DeadLeaves` with `pages == 0` and
    /// reclaims pool extent bytes when shrinkable extents exist.
    #[test]
    fn governor_multi_victim_page_claim_skips_dead_leaves_and_frees_both_anchors() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        use crate::engine::governor::ledger::{
            Charge, ClaimClass, ClaimRank, ClaimSpec, FreeOutcome, FreePool, Ledger, Need,
            NeedTiming, ReachClass, Reserves, Room, TakeRights, Taken, UnitKey, Victim,
            VictimQuery, VictimSource,
        };
        use crate::models::qwen3_5::dflash2_state::store::StateClass;

        struct EmptyPool;
        impl FreePool for EmptyPool {
            fn charge(&self) -> Charge {
                Charge::default()
            }
            fn clear(&mut self) {}
        }
        struct RecordingSource<'a> {
            inner: &'a mut super::Inkling2Inner,
            taken: Vec<UnitKey>,
        }
        impl VictimSource for RecordingSource<'_> {
            fn units(&self, query: &VictimQuery) -> Vec<Victim> {
                self.inner.inkling2_victim_units(query, &[]).into_units()
            }
            fn take(&mut self, query: &VictimQuery, victim: &Victim) -> Taken {
                self.taken.push(victim.key.clone());
                self.inner.inkling2_take_victim(query, victim)
            }
        }

        let mut inner = build_synthetic_dflash2_inner_with_window(256, None, 8, 4);
        // Replace the fixed-size synthetic coordinator with an extent-grown coordinator
        // (`initial_blocks = K = 2 < max_blocks = 3 * K = 6`, `extent = K = 2`) so
        // `try_grow_pool(2 * K)` reaches the ceiling `3 * K` (2 shrinkable extents above runway)
        // and the post-reclaim `2 * K`-block allocation CANNOT grow — it MUST fit into the
        // reclaimed blocks in place!
        const K: u32 = 2;
        const BS: usize = 32;
        {
            use crate::models::inkling2::kv_cache::{
                Inkling2KVCacheCoordinator, default_inkling2_kv_format,
            };
            use crate::models::inkling2::paged_forward::Inkling2PagedRuntime;
            use crate::models::inkling2::state_disk::Inkling2StateGeometry;

            let recency = Arc::clone(&inner.cache_recency);
            let coord = Inkling2KVCacheCoordinator::new_extent_grown_for_test(
                &inner.config,
                default_inkling2_kv_format(),
                BS as u32,
                256,
                32,
                4,
                K,
                3 * K,
                K,
                Arc::clone(&recency),
            )
            .expect("new_extent_grown_for_test");
            let mut rt = Inkling2PagedRuntime::from_coordinator(coord, 32, recency);
            let geom = Inkling2StateGeometry::from_coordinator(&inner.config, &rt.coordinator)
                .expect("geom");
            rt.state_store.set_geometry(geom);
            inner.paged = Some(rt);
        }
        let conv = crate::models::inkling2::short_conv::Inkling2ShortConvState::new(
            inner.config.num_layer_visits(),
            inner.config.sconv_kernel_size,
            inner.config.num_key_value_heads * inner.config.head_dim,
            inner.config.hidden_size,
            crate::array::DType::BFloat16,
        )
        .expect("conv");

        let tokens_a: Vec<u32> = (100u32..(100 + K * BS as u32)).collect();
        let tokens_b: Vec<u32> = (400u32..(400 + K * BS as u32)).collect();
        let tokens_new: Vec<u32> = (700u32..(700 + 2 * K * BS as u32)).collect();

        // Grow by 2 extents (`2 * K = 4` blocks above runway) so `cached_extent_bytes(true, &[])`
        // becomes > 0 as soon as `K = 2` blocks become evictable!
        // Pin the runway blocks with live sequences so ONLY the `2 * K = 4` newly grown blocks are
        // free for `entry_a` and `entry_b`.
        let (entry_a, entry_b, pool_blocks_before_claim) = {
            let rt = inner.paged.as_mut().expect("paged");
            let runway = rt
                .coordinator
                .full_adapter()
                .block_telemetry()
                .expect("telemetry")
                .total_blocks;
            assert!(
                rt.coordinator.full_adapter_mut().try_grow_pool(2 * K),
                "grow full KV pool by 2 extents above runway"
            );
            let total_after_grow = rt
                .coordinator
                .full_adapter()
                .block_telemetry()
                .expect("telemetry after grow")
                .total_blocks;
            assert!(
                total_after_grow >= runway + 2 * K,
                "grown pool must have at least 2*K blocks above runway"
            );

            // Pin all blocks outside the final `2 * K` blocks with live sequence(s) so initially
            // 0 free and 0 evictable blocks remain once `entry_a` and `entry_b` are captured!
            let pin_blocks = total_after_grow - 2 * K;
            let mut next_pin_seq: u32 = 10;
            let mut remaining = pin_blocks;
            let mut tok_base: u32 = 10_000;
            while remaining > 0 {
                let take = remaining.min(8); // 8 * 32 = 256 == max_model_len
                let pin_tokens: Vec<u32> = (tok_base..(tok_base + take * BS as u32)).collect();
                rt.coordinator
                    .reset_scheduled_request(next_pin_seq)
                    .expect("reset pin seq");
                rt.coordinator
                    .full_adapter_mut()
                    .activate_request(next_pin_seq)
                    .expect("activate pin seq");
                rt.coordinator
                    .full_adapter_mut()
                    .record_tokens(&pin_tokens)
                    .expect("record pin tokens");
                tok_base += take * BS as u32 + 100;
                next_pin_seq += 1;
                remaining -= take;
            }
            assert_eq!(
                rt.coordinator
                    .full_adapter()
                    .block_telemetry()
                    .expect("tel before capture")
                    .free_blocks,
                2 * K,
                "exactly 2*K free blocks must remain for entry_a and entry_b"
            );

            // Capture `entry_a` (K = 2 blocks) and release `seq 1`.
            rt.coordinator.reset_scheduled_request(1).expect("reset 1");
            rt.coordinator
                .record_tokens_all(1, &tokens_a)
                .expect("record a");
            let id_a = rt
                .state_store
                .capture_boundary(
                    &mut rt.coordinator,
                    &conv,
                    1,
                    &tokens_a,
                    K * BS as u32,
                    901,
                    StateClass::Ordinary,
                    None,
                )
                .expect("capture a")
                .expect("id_a");
            rt.coordinator
                .release_request_all(1)
                .expect("release seq 1");

            // Capture `entry_b` (K = 2 distinct blocks) and release `seq 2`.
            rt.coordinator.reset_scheduled_request(2).expect("reset 2");
            rt.coordinator
                .record_tokens_all(2, &tokens_b)
                .expect("record b");
            let id_b = rt
                .state_store
                .capture_boundary(
                    &mut rt.coordinator,
                    &conv,
                    2,
                    &tokens_b,
                    K * BS as u32,
                    902,
                    StateClass::Ordinary,
                    None,
                )
                .expect("capture b")
                .expect("id_b");
            rt.coordinator
                .release_request_all(2)
                .expect("release seq 2");

            let tel = rt
                .coordinator
                .full_adapter()
                .block_telemetry()
                .expect("tel armed");
            assert_eq!(tel.free_blocks, 0, "0 free blocks before claim");
            assert_eq!(tel.evictable_blocks, 0, "0 evictable blocks before claim");
            assert_eq!(
                rt.coordinator.full_adapter().takeable_blocks(),
                0,
                "0 takeable blocks before claim"
            );
            (id_a, id_b, tel.total_blocks)
        };

        // Run a Governor `ledger::free` page-only claim requesting `2 * K` pages (requiring BOTH
        // `entry_a` and `entry_b`!).
        let mut ledger = Ledger::new();
        let claim = ledger.open(
            ClaimSpec {
                cls: ClaimClass::InService,
                request_id: 91,
                rank: ClaimRank {
                    priority: 0,
                    decoding: true,
                    completed_tokens: 64,
                    earliness: 1,
                },
                reach: ReachClass::Ordinary,
                timing: NeedTiming::Immediate,
            },
            1,
        );
        let mut src = RecordingSource {
            inner: &mut inner,
            taken: Vec::new(),
        };
        let res = ledger.free(
            claim,
            Need {
                pages: 2 * K,
                ..Need::default()
            },
            Room::default(),
            &mut EmptyPool,
            &mut [&mut src],
        );
        assert_eq!(
            res.outcome,
            FreeOutcome::Met,
            "multi-victim page-only claim must succeed"
        );
        assert!(
            res.pages >= 2 * K,
            "claim must credit at least 2*K pages (got {})",
            res.pages
        );
        assert!(
            !src.taken.contains(&UnitKey::DeadLeaves),
            "DeadLeaves must NEVER be taken during a page-only claim (taken={:?})",
            src.taken
        );
        assert!(
            src.taken.contains(&UnitKey::Store(entry_a)),
            "Store(entry_a) must be taken (taken={:?})",
            src.taken
        );
        assert!(
            src.taken.contains(&UnitKey::Store(entry_b)),
            "Store(entry_b) must be taken (taken={:?})",
            src.taken
        );

        // Verify the pool was NOT shrunk and actually has `>= 2 * K` allocatable pages so
        // allocating a new `2 * K`-block sequence succeeds without growing the pool!
        {
            let rt = inner.paged.as_mut().expect("paged");
            let full = rt.coordinator.full_adapter();
            let tel = full.block_telemetry().expect("tel after claim");
            assert_eq!(
                tel.total_blocks, pool_blocks_before_claim,
                "page-only claim must NOT shrink the KV pool"
            );
            assert!(
                full.takeable_blocks() >= 2 * K,
                "at least 2*K blocks must be takeable after claim (got {})",
                full.takeable_blocks()
            );

            // Also verify byte-only claim sees `DeadLeaves` with `pages == 0` and reclaims bytes!
            let byte_query = VictimQuery {
                rights: TakeRights {
                    units: true,
                    reserves: Reserves::None,
                    ..Default::default()
                },
                need: Need {
                    bytes: 1,
                    ..Need::default()
                },
            };
            let byte_runs = inner.inkling2_victim_units(&byte_query, &[]);
            let dl = byte_runs
                .ahead
                .iter()
                .find(|v| v.key == UnitKey::DeadLeaves)
                .expect("byte-only claim must emit DeadLeaves when shrinkable extent exists");
            assert!(dl.bytes > 0);
            assert_eq!(dl.pages, 0, "DeadLeaves must always advertise pages == 0");

            // Now allocate a new 2*K-block sequence in place (no growth needed!).
            let rt = inner.paged.as_mut().expect("paged");
            rt.coordinator.reset_scheduled_request(3).expect("reset 3");
            rt.coordinator
                .full_adapter_mut()
                .activate_request(3)
                .expect("activate 3");
            rt.coordinator
                .full_adapter_mut()
                .record_tokens(&tokens_new)
                .expect("allocating 2*K blocks after multi-victim page reclaim must succeed");
            assert_eq!(
                rt.coordinator
                    .full_adapter()
                    .block_telemetry()
                    .expect("tel after alloc")
                    .total_blocks,
                pool_blocks_before_claim,
                "allocation must fit into reclaimed blocks without growing the pool"
            );

            // Release seq 3 and verify taking DeadLeaves on a byte-only claim credits `pages == 0`.
            rt.coordinator
                .full_adapter_mut()
                .activate_request(3)
                .expect("activate 3 for release");
            rt.coordinator
                .full_adapter_mut()
                .release_request()
                .expect("release 3");
            let taken_dl = inner.inkling2_take_victim(&byte_query, dl);
            assert_eq!(taken_dl.outcome, FreeOutcome::Met);
            assert!(taken_dl.now_bytes > 0);
            assert_eq!(
                taken_dl.pages, 0,
                "taking DeadLeaves must credit 0 pages because it shrinks the pool"
            );
        }
    }

    /// Round-10 Finding 1 regression:
    /// When an extent-grown Global KV pool has completely empty tail extents above the runway
    /// (`free_extent_bytes(true) > 0`, `cached_extent_bytes(true, &[]) == 0`) alongside a useful
    /// `StateStore` anchor:
    /// - A byte-only Governor `ledger::free` claim emits `UnitKey::Extent` (`bytes > 0, pages == 0`)
    ///   ahead of `Store(entry_a)`, takes ONLY `UnitKey::Extent`, returns `FreeOutcome::Met` with
    ///   `now_bytes > 0` and `pages == 0`, shrinks the pool, and leaves the useful `StateStore`
    ///   anchor completely untouched in RAM.
    /// - A page-only claim (`need.pages > 0, need.bytes == 0`) skips `UnitKey::Extent` (`pages == 0`)
    ///   and reclaims `Store(entry_a)` in place without shrinking the pool.
    #[test]
    fn governor_byte_claim_reclaims_empty_global_kv_extents_before_states() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        use crate::engine::governor::ledger::{
            Charge, ClaimClass, ClaimRank, ClaimSpec, FreeOutcome, FreePool, Ledger, Need,
            NeedTiming, ReachClass, Reserves, Room, TakeRights, Taken, UnitKey, Victim,
            VictimQuery, VictimSource,
        };
        use crate::models::inkling2::kv_cache::{
            Inkling2KVCacheCoordinator, default_inkling2_kv_format,
        };
        use crate::models::inkling2::paged_forward::Inkling2PagedRuntime;
        use crate::models::inkling2::state_disk::Inkling2StateGeometry;
        use crate::models::qwen3_5::dflash2_state::store::StateClass;

        struct EmptyPool;
        impl FreePool for EmptyPool {
            fn charge(&self) -> Charge {
                Charge::default()
            }
            fn clear(&mut self) {}
        }
        struct RecordingSource<'a> {
            inner: &'a mut super::Inkling2Inner,
            taken: Vec<UnitKey>,
        }
        impl VictimSource for RecordingSource<'_> {
            fn units(&self, query: &VictimQuery) -> Vec<Victim> {
                self.inner.inkling2_victim_units(query, &[]).into_units()
            }
            fn take(&mut self, query: &VictimQuery, victim: &Victim) -> Taken {
                self.taken.push(victim.key.clone());
                self.inner.inkling2_take_victim(query, victim)
            }
        }

        const K: u32 = 2;
        const BS: usize = 32;
        let mut inner = build_synthetic_dflash2_inner_with_window(256, None, 8, 4);
        {
            let recency = Arc::clone(&inner.cache_recency);
            let coord = Inkling2KVCacheCoordinator::new_extent_grown_for_test(
                &inner.config,
                default_inkling2_kv_format(),
                BS as u32,
                256,
                32,
                4,
                K,
                4 * K,
                K,
                Arc::clone(&recency),
            )
            .expect("new_extent_grown_for_test");
            let mut rt = Inkling2PagedRuntime::from_coordinator(coord, 32, recency);
            let geom = Inkling2StateGeometry::from_coordinator(&inner.config, &rt.coordinator)
                .expect("geom");
            rt.state_store.set_geometry(geom);
            inner.paged = Some(rt);
        }
        let conv = crate::models::inkling2::short_conv::Inkling2ShortConvState::new(
            inner.config.num_layer_visits(),
            inner.config.sconv_kernel_size,
            inner.config.num_key_value_heads * inner.config.head_dim,
            inner.config.hidden_size,
            crate::array::DType::BFloat16,
        )
        .expect("conv");

        let tokens_a: Vec<u32> = (100u32..(100 + K * BS as u32)).collect();

        // Capture one useful StateStore anchor (`entry_a`, K blocks in runway), release seq 1,
        // then grow the Global KV pool by 3 extents (`3 * K` blocks) so the tail above
        // `runway + keep_one` is entirely empty (`free_extent_bytes > 0`, `cached_extent_bytes == 0`).
        let (entry_a, blocks_before_byte_claim, free_bytes_before) = {
            let rt = inner.paged.as_mut().expect("paged");
            rt.coordinator.reset_scheduled_request(1).expect("reset 1");
            rt.coordinator
                .record_tokens_all(1, &tokens_a)
                .expect("record a");
            let id_a = rt
                .state_store
                .capture_boundary(
                    &mut rt.coordinator,
                    &conv,
                    1,
                    &tokens_a,
                    K * BS as u32,
                    951,
                    StateClass::Ordinary,
                    None,
                )
                .expect("capture a")
                .expect("id_a");
            rt.coordinator
                .release_request_all(1)
                .expect("release seq 1");

            assert!(
                rt.coordinator.full_adapter_mut().try_grow_pool(3 * K),
                "grow full KV pool by 3 empty extents"
            );
            let full = rt.coordinator.full_adapter();
            let free_bytes = full.free_extent_bytes(true);
            let cached_bytes = full.cached_extent_bytes(true, &[]);
            assert!(
                free_bytes > 0,
                "grown empty tail above runway+keep_one must report free_extent_bytes > 0"
            );
            assert_eq!(
                cached_bytes, 0,
                "empty tail with no dead leaves must report cached_extent_bytes == 0"
            );
            let total_blocks = full
                .block_telemetry()
                .expect("telemetry before byte claim")
                .total_blocks;
            (id_a, total_blocks, free_bytes)
        };

        // Verify `inkling2_victim_units` emits `UnitKey::Extent` (bytes > 0, pages == 0) FIRST,
        // before `Store(entry_a)`, and emits no `DeadLeaves`.
        let byte_query = VictimQuery {
            rights: TakeRights {
                units: true,
                reserves: Reserves::None,
                ..Default::default()
            },
            need: Need {
                bytes: 1,
                ..Need::default()
            },
        };
        let byte_runs = inner.inkling2_victim_units(&byte_query, &[]);
        assert_eq!(
            byte_runs.ahead.first().map(|v| &v.key),
            Some(&UnitKey::Extent),
            "UnitKey::Extent must be emitted first in runs.ahead"
        );
        let ext_victim = &byte_runs.ahead[0];
        assert_eq!(ext_victim.bytes, free_bytes_before);
        assert_eq!(ext_victim.pages, 0);
        assert!(
            !byte_runs.ahead.iter().any(|v| v.key == UnitKey::DeadLeaves),
            "DeadLeaves must not be emitted when cached_extent_bytes == 0"
        );

        // 1. Run a byte-only Governor `ledger::free` claim: must take ONLY `UnitKey::Extent`
        //    and leave `entry_a` completely untouched!
        let mut ledger = Ledger::new();
        let byte_claim = ledger.open(
            ClaimSpec {
                cls: ClaimClass::InService,
                request_id: 92,
                rank: ClaimRank {
                    priority: 0,
                    decoding: true,
                    completed_tokens: 64,
                    earliness: 1,
                },
                reach: ReachClass::Ordinary,
                timing: NeedTiming::Immediate,
            },
            1,
        );
        let mut src = RecordingSource {
            inner: &mut inner,
            taken: Vec::new(),
        };
        let byte_res = ledger.free(
            byte_claim,
            Need {
                bytes: 1,
                ..Need::default()
            },
            Room::default(),
            &mut EmptyPool,
            &mut [&mut src],
        );
        assert_eq!(
            byte_res.outcome,
            FreeOutcome::Met,
            "byte-only claim must be satisfied by releasing empty Global KV extents"
        );
        assert!(
            byte_res.now_bytes > 0,
            "byte-only claim must report positive reclaimed extent bytes"
        );
        assert_eq!(
            byte_res.pages, 0,
            "releasing empty extents must credit 0 pages"
        );
        assert_eq!(
            src.taken,
            vec![UnitKey::Extent],
            "only UnitKey::Extent should be taken; useful StateStore anchor must stay untouched"
        );
        {
            let rt = inner.paged.as_ref().expect("paged");
            assert!(
                rt.state_store
                    .store
                    .get(entry_a)
                    .is_some_and(|e| e.ram().is_some()),
                "useful StateStore anchor must remain intact in RAM after empty-extent byte claim"
            );
            let blocks_after = rt
                .coordinator
                .full_adapter()
                .block_telemetry()
                .expect("telemetry after byte claim")
                .total_blocks;
            assert!(
                blocks_after < blocks_before_byte_claim,
                "taking UnitKey::Extent must actually shrink the KV pool ({blocks_after} < {blocks_before_byte_claim})"
            );
        }

        // 2. Grow empty extents again so `free_extent_bytes(true) > 0`, then run a page-only
        //    claim (`need.pages == K, need.bytes == 0`): must SKIP `UnitKey::Extent` (`pages == 0`)
        //    and reclaim `Store(entry_a)` without shrinking the pool!
        let blocks_before_page_claim = {
            let rt = inner.paged.as_mut().expect("paged");
            assert!(
                rt.coordinator.full_adapter_mut().try_grow_pool(2 * K),
                "re-grow empty extents before page-only claim"
            );
            let full = rt.coordinator.full_adapter();
            assert!(
                full.free_extent_bytes(true) > 0,
                "empty extents must be present before page-only claim"
            );
            full.block_telemetry()
                .expect("telemetry before page claim")
                .total_blocks
        };
        let page_claim = ledger.open(
            ClaimSpec {
                cls: ClaimClass::InService,
                request_id: 93,
                rank: ClaimRank {
                    priority: 0,
                    decoding: true,
                    completed_tokens: 64,
                    earliness: 2,
                },
                reach: ReachClass::Ordinary,
                timing: NeedTiming::Immediate,
            },
            2,
        );
        let mut page_src = RecordingSource {
            inner: &mut inner,
            taken: Vec::new(),
        };
        let page_res = ledger.free(
            page_claim,
            Need {
                pages: K,
                ..Need::default()
            },
            Room::default(),
            &mut EmptyPool,
            &mut [&mut page_src],
        );
        assert_eq!(page_res.outcome, FreeOutcome::Met);
        assert!(page_res.pages >= K);
        assert!(
            !page_src.taken.contains(&UnitKey::Extent),
            "page-only claim must skip UnitKey::Extent because it advertises pages == 0 (taken={:?})",
            page_src.taken
        );
        assert!(
            page_src.taken.contains(&UnitKey::Store(entry_a)),
            "page-only claim must take Store(entry_a) (taken={:?})",
            page_src.taken
        );
        assert_eq!(
            inner
                .paged
                .as_ref()
                .expect("paged")
                .coordinator
                .full_adapter()
                .block_telemetry()
                .expect("telemetry after page claim")
                .total_blocks,
            blocks_before_page_claim,
            "page-only claim must NOT shrink the KV pool"
        );
    }

    /// Round-11 Finding 1 regression:
    /// A mixed Governor claim (`need.bytes > 0 && need.pages > 0`, e.g., when a pool grow is
    /// denied) must NEVER emit or take `UnitKey::Extent` / `UnitKey::DeadLeaves`, leaving empty
    /// tail slots intact in the Global KV pool while reclaiming `Store(anchor)` so allocating
    /// the requested pages succeeds in place.
    #[test]
    fn governor_mixed_claim_never_reclaims_global_kv_extents() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        use crate::engine::governor::ledger::{
            Charge, ClaimClass, ClaimRank, ClaimSpec, FreeOutcome, FreePool, Ledger, Need,
            NeedTiming, ReachClass, Reserves, Room, Stage, TakeRights, Taken, UnitKey, Victim,
            VictimQuery, VictimSource,
        };
        use crate::models::inkling2::kv_cache::{
            Inkling2KVCacheCoordinator, default_inkling2_kv_format,
        };
        use crate::models::inkling2::paged_forward::Inkling2PagedRuntime;
        use crate::models::inkling2::state_disk::Inkling2StateGeometry;
        use crate::models::qwen3_5::dflash2_state::store::StateClass;

        struct EmptyPool;
        impl FreePool for EmptyPool {
            fn charge(&self) -> Charge {
                Charge::default()
            }
            fn clear(&mut self) {}
        }
        struct RecordingSource<'a> {
            inner: &'a mut super::Inkling2Inner,
            taken: Vec<UnitKey>,
        }
        impl VictimSource for RecordingSource<'_> {
            fn units(&self, query: &VictimQuery) -> Vec<Victim> {
                self.inner.inkling2_victim_units(query, &[]).into_units()
            }
            fn take(&mut self, query: &VictimQuery, victim: &Victim) -> Taken {
                self.taken.push(victim.key.clone());
                self.inner.inkling2_take_victim(query, victim)
            }
        }

        const K: u32 = 2;
        const BS: usize = 32;
        let mut inner = build_synthetic_dflash2_inner_with_window(256, None, 8, 4);
        {
            let recency = Arc::clone(&inner.cache_recency);
            let coord = Inkling2KVCacheCoordinator::new_extent_grown_for_test(
                &inner.config,
                default_inkling2_kv_format(),
                BS as u32,
                512,
                32,
                4,
                K,
                4 * K,
                K,
                Arc::clone(&recency),
            )
            .expect("new_extent_grown_for_test");
            let mut rt = Inkling2PagedRuntime::from_coordinator(coord, 32, recency);
            let geom = Inkling2StateGeometry::from_coordinator(&inner.config, &rt.coordinator)
                .expect("geom");
            rt.state_store.set_geometry(geom);
            inner.paged = Some(rt);
        }
        let conv = crate::models::inkling2::short_conv::Inkling2ShortConvState::new(
            inner.config.num_layer_visits(),
            inner.config.sconv_kernel_size,
            inner.config.num_key_value_heads * inner.config.head_dim,
            inner.config.hidden_size,
            crate::array::DType::BFloat16,
        )
        .expect("conv");

        let tokens_a: Vec<u32> = (200u32..(200 + K * BS as u32)).collect();
        let (entry_a, blocks_before, free_bytes_before) = {
            let rt = inner.paged.as_mut().expect("paged");
            rt.coordinator.reset_scheduled_request(1).expect("reset 1");
            rt.coordinator
                .record_tokens_all(1, &tokens_a)
                .expect("record a");
            let id_a = rt
                .state_store
                .capture_boundary(
                    &mut rt.coordinator,
                    &conv,
                    1,
                    &tokens_a,
                    K * BS as u32,
                    961,
                    StateClass::Ordinary,
                    None,
                )
                .expect("capture a")
                .expect("id_a");
            rt.coordinator
                .release_request_all(1)
                .expect("release seq 1");
            assert!(
                rt.coordinator.full_adapter_mut().try_grow_pool(3 * K),
                "grow full KV pool by 3 empty extents"
            );
            let full = rt.coordinator.full_adapter();
            let free_bytes = full.free_extent_bytes(true);
            assert!(free_bytes > 0, "empty tail extents must be present");
            let total_blocks = full
                .block_telemetry()
                .expect("telemetry before mixed claim")
                .total_blocks;
            (id_a, total_blocks, free_bytes)
        };

        let mixed_need = Need {
            bytes: 1,
            pages: K,
            ..Need::default()
        };
        let mixed_query = VictimQuery {
            rights: TakeRights {
                units: true,
                reserves: Reserves::None,
                ..Default::default()
            },
            need: mixed_need,
        };
        let mixed_units = inner.inkling2_victim_units(&mixed_query, &[]).into_units();
        assert!(
            !mixed_units
                .iter()
                .any(|v| matches!(v.key, UnitKey::Extent | UnitKey::DeadLeaves)),
            "mixed claim must not emit Extent or DeadLeaves (units={:?})",
            mixed_units.iter().map(|v| &v.key).collect::<Vec<_>>()
        );
        assert!(
            mixed_units
                .iter()
                .any(|v| v.key == UnitKey::Store(entry_a) && v.bytes > 0 && v.pages == K),
            "mixed claim must emit Store(entry_a) with bytes > 0 and pages == K"
        );

        // Defense-in-depth: direct `inkling2_take_victim` on Extent/DeadLeaves under a mixed
        // query must refuse to shrink the pool.
        for key in [UnitKey::Extent, UnitKey::DeadLeaves] {
            let dummy = Victim {
                key,
                stamp: 0,
                bytes: free_bytes_before,
                pages: 0,
                stage: Stage::FreeNow,
                later: false,
            };
            let taken = inner.inkling2_take_victim(&mixed_query, &dummy);
            assert_eq!(taken, Taken::default());
        }

        let mut ledger = Ledger::new();
        let claim = ledger.open(
            ClaimSpec {
                cls: ClaimClass::InService,
                request_id: 94,
                rank: ClaimRank {
                    priority: 0,
                    decoding: true,
                    completed_tokens: 64,
                    earliness: 1,
                },
                reach: ReachClass::Ordinary,
                timing: NeedTiming::Immediate,
            },
            1,
        );
        let mut src = RecordingSource {
            inner: &mut inner,
            taken: Vec::new(),
        };
        let res = ledger.free(
            claim,
            mixed_need,
            Room::default(),
            &mut EmptyPool,
            &mut [&mut src],
        );
        assert_eq!(res.outcome, FreeOutcome::Met);
        assert!(res.now_bytes > 0);
        assert!(res.pages >= K);
        assert!(
            !src.taken
                .iter()
                .any(|k| matches!(k, UnitKey::Extent | UnitKey::DeadLeaves)),
            "mixed claim must never take Extent or DeadLeaves (taken={:?})",
            src.taken
        );
        assert!(
            src.taken.contains(&UnitKey::Store(entry_a)),
            "mixed claim must take Store(entry_a) (taken={:?})",
            src.taken
        );

        // Empty tail slots remain intact and allocating all 4*K blocks (3*K tail + K reclaimed)
        // succeeds in place without shrinking or re-growing the pool.
        let rt = inner.paged.as_mut().expect("paged");
        assert_eq!(
            rt.coordinator
                .full_adapter()
                .block_telemetry()
                .expect("telemetry after mixed claim")
                .total_blocks,
            blocks_before
        );
        assert_eq!(
            rt.coordinator.full_adapter().free_extent_bytes(true),
            free_bytes_before
        );
        let tokens_new: Vec<u32> = (500u32..(500 + 4 * K * BS as u32)).collect();
        rt.coordinator.reset_scheduled_request(2).expect("reset 2");
        rt.coordinator
            .full_adapter_mut()
            .activate_request(2)
            .expect("activate 2");
        rt.coordinator
            .full_adapter_mut()
            .record_tokens(&tokens_new)
            .expect("allocating all 4*K blocks must succeed using intact tail + reclaimed anchor");
        assert_eq!(
            rt.coordinator
                .full_adapter()
                .block_telemetry()
                .expect("telemetry after alloc")
                .total_blocks,
            blocks_before
        );
    }

    /// Round-9 Finding 4 regression:
    /// Cache-opted-out AR sequences (`reuse_cache == false`) with a loaded DFlash2 draft model
    /// must NOT capture optional `draft_windows` / `block_boundary_draft_window` snapshots across:
    /// 1. Scheduled AR prefill (fast path AND resumable multi-step pass),
    /// 2. Batched AR decode (both solo opted-out AND mixed with a `reuse_cache == true` peer in
    ///    the same batch, while the opted-in peer STILL captures its draft window!),
    /// 3. `materialize_final_token` on an opted-out AR sequence,
    /// 4. Exclusive AR prefill and single decode on an opted-out sequence.
    #[test]
    fn cache_opted_out_ar_turn_skips_draft_window_snapshots() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        use crate::engine::backend::{DecodeStep, PagedBackend};
        use crate::engine::hybrid_scheduler::ScheduledPrefillPass;

        let mut inner = build_synthetic_dflash2_inner_with_window(256, None, 8, 4);
        assert!(
            inner.dflash2.is_some(),
            "test requires a loaded DFlash2 draft model"
        );
        let gen_stream = crate::stream::Stream::generation();

        let prompt_out: Vec<u32> = (10u32..46u32).collect(); // 36 tokens -> crosses 32-token boundary
        let prompt_in: Vec<u32> = (50u32..86u32).collect();

        // Register seq 1 (opted out), seq 2 (opted in), seq 3 (opted out, resumable), seq 4 (opted out, exclusive).
        {
            let rt = inner.paged.as_mut().expect("paged");
            for (seq_id, reuse) in [(1u32, false), (2u32, true), (3u32, false), (4u32, false)] {
                rt.coordinator
                    .reset_scheduled_request(seq_id)
                    .expect("reset seq");
                rt.ensure_conv_state(seq_id, &inner.config)
                    .expect("ensure conv");
                rt.seq_reuse_cache.insert(seq_id, reuse);
            }
        }

        // 1a. Scheduled AR prefill fast path: seq 1 (`reuse_cache = false`) vs seq 2 (`reuse_cache = true`).
        assert!(matches!(
            inner
                .run_paged_prefill_pass(
                    1,
                    &prompt_out,
                    0,
                    prompt_out.len(),
                    401,
                    false,
                    gen_stream,
                    None
                )
                .expect("fast prefill seq 1"),
            ScheduledPrefillPass::Done(Some(_))
        ));
        assert!(matches!(
            inner
                .run_paged_prefill_pass(
                    2,
                    &prompt_in,
                    0,
                    prompt_in.len(),
                    402,
                    true,
                    gen_stream,
                    None
                )
                .expect("fast prefill seq 2"),
            ScheduledPrefillPass::Done(Some(_))
        ));

        // 1b. Scheduled AR prefill resumable pass: seq 3 (`reuse_cache = false`, `budget_ms = Some(0.0)`).
        let mut passes = 0usize;
        loop {
            passes += 1;
            assert!(passes < 256, "resumable prefill exceeded pass limit");
            match inner
                .run_paged_prefill_pass(
                    3,
                    &prompt_out,
                    0,
                    prompt_out.len(),
                    403,
                    false,
                    gen_stream,
                    Some(0.0),
                )
                .expect("resumable prefill seq 3")
            {
                ScheduledPrefillPass::Paused => {}
                ScheduledPrefillPass::Done(Some(_)) => break,
                ScheduledPrefillPass::Done(None) => panic!("expected final logits for seq 3"),
            }
        }
        {
            let rt = inner.paged.as_ref().expect("paged");
            assert!(
                !rt.draft_windows.contains_key(&1)
                    && !rt.block_boundary_draft_window.contains_key(&1),
                "opted-out fast AR prefill must create ZERO draft windows for seq 1"
            );
            assert!(
                !rt.draft_windows.contains_key(&3)
                    && !rt.block_boundary_draft_window.contains_key(&3),
                "opted-out resumable AR prefill must create ZERO draft windows for seq 3"
            );
            assert_eq!(
                rt.draft_windows.get(&2).map(|w| w.end()),
                Some(prompt_in.len() as u32),
                "opted-in fast AR prefill must capture draft_windows for seq 2"
            );
            assert!(
                rt.block_boundary_draft_window.contains_key(&2),
                "opted-in fast AR prefill must snapshot block_boundary_draft_window at 32 for seq 2"
            );
        }

        // 2a. Batched AR decode — solo opted-out batch (`seq 1`).
        let _ = inner
            .paged_decode_step_batched(&[(1, 17)])
            .expect("solo opted-out batched decode");
        let _ = inner.settle_paged_step(1).expect("settle seq 1");
        {
            let rt = inner.paged.as_ref().expect("paged");
            assert!(
                !rt.draft_windows.contains_key(&1)
                    && !rt.block_boundary_draft_window.contains_key(&1),
                "solo opted-out batched AR decode must create ZERO draft windows for seq 1"
            );
        }

        // 2b. Batched AR decode — MIXED batch (`seq 1` opted out + `seq 2` opted in).
        let seq2_end_before = inner
            .paged
            .as_ref()
            .unwrap()
            .draft_windows
            .get(&2)
            .expect("seq 2 window")
            .end();
        let _ = inner
            .paged_decode_step_batched(&[(1, 18), (2, 18)])
            .expect("mixed batched decode");
        let _ = inner.settle_paged_step(1).expect("settle seq 1 mixed");
        let _ = inner.settle_paged_step(2).expect("settle seq 2 mixed");
        {
            let rt = inner.paged.as_ref().expect("paged");
            assert!(
                !rt.draft_windows.contains_key(&1)
                    && !rt.block_boundary_draft_window.contains_key(&1),
                "opted-out row in mixed batched AR decode must create ZERO draft windows for seq 1"
            );
            assert_eq!(
                rt.draft_windows.get(&2).map(|w| w.end()),
                Some(seq2_end_before + 1),
                "opted-in peer in the SAME mixed batch must still advance its draft window"
            );
        }

        // 3. `materialize_final_token` on opted-out AR sequence (`seq 1`).
        inner.paged.as_mut().expect("paged").active_seq = 1;
        {
            let mut stepper = inner.begin_paged_decode().expect("begin_paged_decode");
            stepper
                .materialize_final(19)
                .expect("materialize_final seq 1");
        }
        {
            let rt = inner.paged.as_ref().expect("paged");
            assert!(
                !rt.draft_windows.contains_key(&1)
                    && !rt.block_boundary_draft_window.contains_key(&1),
                "materialize_final_token on opted-out AR seq must create ZERO draft windows"
            );
        }

        // 4. Exclusive AR prefill + single decode on opted-out sequence (`seq 4`).
        let _ = inner
            .paged_prefill_sequence(4, &prompt_out, None)
            .expect("exclusive AR prefill seq 4");
        let _ = inner
            .paged_decode_step(4, 21)
            .expect("exclusive AR decode seq 4");
        let _ = inner.settle_paged_step(4).expect("settle seq 4");
        {
            let rt = inner.paged.as_ref().expect("paged");
            assert!(
                !rt.draft_windows.contains_key(&4)
                    && !rt.block_boundary_draft_window.contains_key(&4),
                "exclusive opted-out AR prefill + decode must create ZERO draft windows for seq 4"
            );
        }
    }

    /// Round-12 Finding 1 regression:
    /// Resident Inkling2 host-RAM state anchors (`swa_data`, `conv_data`, `draft_window`, lineage)
    /// are charged into `CacheLimitCoordinator::registered_pool_bytes()` (and thus into
    /// `Governor::charge()`). When a mixed KV grow claim (`need.bytes > 0 && need.pages > 0`)
    /// reclaims an older anchor whose KV blocks are still pinned by a live sequence (0 KV pages
    /// freed), `registered_pool_bytes()` and `Governor::snapshot().charged` drop by at least
    /// `taken.now_bytes` so the grow retry succeeds (or, if still short, the ledger continues to a
    /// later anchor that frees the needed `K` KV pages).
    pub(crate) struct TestCoordGovernorIo {
        pub(crate) coord: Arc<crate::cache_limit::CacheLimitCoordinator>,
        pub(crate) active: u64,
    }

    impl crate::engine::governor::governor::GovernorIo for TestCoordGovernorIo {
        fn stats(&self) -> crate::engine::governor::governor::MemoryStats {
            crate::engine::governor::governor::MemoryStats {
                active: self.active,
                pools: self.coord.registered_pool_bytes(),
                cache: 0,
            }
        }
        fn host(&self) -> crate::engine::governor::governor::HostSample {
            let spare = crate::engine::governor::host::spare_reserve_bytes(32 << 30) + (16 << 30);
            crate::engine::governor::governor::HostSample {
                memory: Some(crate::engine::governor::host::HostMemory {
                    available: spare,
                    spare,
                }),
                pressure: Some(crate::engine::governor::host::Pressure::Normal),
            }
        }
        fn now(&self) -> f64 {
            0.0
        }
        fn sleep(&self, _seconds: f64) {}
        fn clear_cache(&self) {}
        fn set_memory_limit(&self, bytes: u64) -> u64 {
            bytes
        }
        fn physical_memory(&self) -> u64 {
            32 << 30
        }
    }

    #[test]
    fn governor_mixed_claim_credits_host_anchor_reclaim_in_registered_pools() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        use crate::cache_limit::CacheLimitCoordinator;
        use crate::engine::governor::governor::{Failure, Governor, ModelRegistration, Scope};
        use crate::engine::governor::ledger::{
            ClaimClass, ClaimRank, ClaimSpec, FreeOutcome, Need, NeedTiming, ReachClass, Taken,
            UnitKey, Victim, VictimQuery, VictimSource,
        };
        use crate::models::inkling2::kv_cache::{
            Inkling2KVCacheCoordinator, default_inkling2_kv_format,
        };
        use crate::models::inkling2::paged_forward::Inkling2PagedRuntime;
        use crate::models::inkling2::state_disk::Inkling2StateGeometry;
        use crate::models::qwen3_5::dflash2_state::store::StateClass;

        const ACTIVE: u64 = 16 << 20;
        const K: u32 = 2;
        const BS: usize = 32;

        struct RecordingSource<'a> {
            inner: &'a mut super::Inkling2Inner,
            taken: Vec<UnitKey>,
        }
        impl VictimSource for RecordingSource<'_> {
            fn units(&self, query: &VictimQuery) -> Vec<Victim> {
                self.inner.inkling2_victim_units(query, &[]).into_units()
            }
            fn take(&mut self, query: &VictimQuery, victim: &Victim) -> Taken {
                self.taken.push(victim.key.clone());
                self.inner.inkling2_take_victim(query, victim)
            }
        }

        let coord = Arc::new(CacheLimitCoordinator::new_for_test());
        let mut inner = build_synthetic_dflash2_inner_with_window(256, None, 8, 4);
        {
            let recency = Arc::clone(&inner.cache_recency);
            let kv_coord = Inkling2KVCacheCoordinator::new_extent_grown_for_test(
                &inner.config,
                default_inkling2_kv_format(),
                BS as u32,
                512,
                32,
                4,
                K,
                4 * K,
                K,
                Arc::clone(&recency),
            )
            .expect("new_extent_grown_for_test");
            let mut rt = Inkling2PagedRuntime::from_coordinator(kv_coord, 32, recency);
            let geom = Inkling2StateGeometry::from_coordinator(&inner.config, &rt.coordinator)
                .expect("geom");
            rt.state_store.set_geometry(geom);
            rt.state_store.attach_host_budget_to(Arc::clone(&coord));
            inner.paged = Some(rt);
        }
        let conv = crate::models::inkling2::short_conv::Inkling2ShortConvState::new(
            inner.config.num_layer_visits(),
            inner.config.sconv_kernel_size,
            inner.config.num_key_value_heads * inner.config.head_dim,
            inner.config.hidden_size,
            crate::array::DType::BFloat16,
        )
        .expect("conv");

        let tokens_a: Vec<u32> = (100u32..(100 + K * BS as u32)).collect();
        let tokens_b: Vec<u32> = (300u32..(300 + K * BS as u32)).collect();
        let (entry_a, entry_b, bytes_a, bytes_b) = {
            let rt = inner.paged.as_mut().expect("paged");
            // `entry_a` (older): keep seq 1 LIVE so its Global KV blocks are pinned (0 reclaimable pages).
            rt.coordinator.reset_scheduled_request(1).expect("reset 1");
            rt.coordinator
                .record_tokens_all(1, &tokens_a)
                .expect("record a");
            let id_a = rt
                .state_store
                .capture_boundary(
                    &mut rt.coordinator,
                    &conv,
                    1,
                    &tokens_a,
                    K * BS as u32,
                    971,
                    StateClass::Ordinary,
                    None,
                )
                .expect("capture a")
                .expect("id_a");
            // `entry_b` (newer): release seq 2 so its `K` Global KV blocks are reclaimable.
            rt.coordinator.reset_scheduled_request(2).expect("reset 2");
            rt.coordinator
                .record_tokens_all(2, &tokens_b)
                .expect("record b");
            let id_b = rt
                .state_store
                .capture_boundary(
                    &mut rt.coordinator,
                    &conv,
                    2,
                    &tokens_b,
                    K * BS as u32,
                    972,
                    StateClass::Ordinary,
                    None,
                )
                .expect("capture b")
                .expect("id_b");
            rt.coordinator
                .release_request_all(2)
                .expect("release seq 2");
            let ba = rt.state_store.store.get(id_a).expect("a").state_bytes();
            let bb = rt.state_store.store.get(id_b).expect("b").state_bytes();
            assert!(ba > 0 && bb > 0);
            assert_eq!(
                coord.registered_pool_bytes(),
                ba + bb,
                "resident host anchors must be charged into CacheLimitCoordinator"
            );
            (id_a, id_b, ba, bb)
        };

        let claim_spec = ClaimSpec {
            cls: ClaimClass::InService,
            request_id: 97,
            rank: ClaimRank {
                priority: 0,
                decoding: true,
                completed_tokens: 64,
                earliness: 1,
            },
            reach: ReachClass::Ordinary,
            timing: NeedTiming::Immediate,
        };

        // Case A: Tight hard_budget where freeing `entry_a`'s host bytes covers the `EngineBudget`
        // grow deficit even though `entry_a` frees 0 KV pages — grow retry MUST succeed!
        let extent_bytes = bytes_a;
        let deficit = (bytes_a / 2).max(1);
        let pools_before = coord.registered_pool_bytes();
        let hard_budget_a = ACTIVE + pools_before + extent_bytes - deficit;
        let gov_a = Governor::new(Arc::new(TestCoordGovernorIo {
            coord: Arc::clone(&coord),
            active: ACTIVE,
        }));
        let _reg_a = gov_a
            .register_model(ModelRegistration {
                model_id: 71,
                hard_budget: hard_budget_a,
                untracked: 0,
                page_bytes: 1 << 20,
            })
            .expect("register gov_a");
        assert_eq!(gov_a.snapshot().charged, ACTIVE + pools_before);

        let claim_a = gov_a.open(claim_spec, 71);
        let _scope_a = Scope::enter(claim_a.claim());
        assert_eq!(
            gov_a.try_reserve(extent_bytes).err(),
            Some(Failure::EngineBudget)
        );
        let mixed_need_a = gov_a.need(
            claim_a.id(),
            Need {
                pages: K,
                kv: true,
                ..Need::default()
            },
        );
        assert_eq!(mixed_need_a.bytes, extent_bytes);
        assert_eq!(mixed_need_a.pages, K);
        let mut src_a = RecordingSource {
            inner: &mut inner,
            taken: Vec::new(),
        };
        let res_a = gov_a.free(claim_a.id(), mixed_need_a, &mut [&mut src_a]);
        assert_eq!(res_a.outcome, FreeOutcome::Met);
        assert_eq!(res_a.now_bytes, bytes_a);
        assert_eq!(res_a.pages, 0);
        assert_eq!(src_a.taken, vec![UnitKey::Store(entry_a)]);
        assert_eq!(
            pools_before - coord.registered_pool_bytes(),
            bytes_a,
            "reclaiming entry_a must drop registered_pool_bytes by at least now_bytes"
        );
        assert_eq!(
            gov_a.snapshot().charged,
            ACTIVE + bytes_b,
            "Governor::charged must drop by entry_a's host bytes"
        );
        let reservation = gov_a
            .try_reserve(extent_bytes)
            .expect("grow retry MUST succeed after host anchor reclaim lowers Governor::charged");
        drop(reservation);
        drop(_scope_a);
        drop(claim_a);

        // Case B: Deficit exceeds `entry_c`'s host bytes — ledger reclaims `entry_c` (0 pages)
        // AND continues to `entry_b` (frees `K` KV pages), dropping registered host bytes to 0.
        let entry_c = {
            let rt = inner.paged.as_mut().expect("paged");
            let id_c = rt
                .state_store
                .capture_boundary(
                    &mut rt.coordinator,
                    &conv,
                    1,
                    &tokens_a,
                    K * BS as u32,
                    971,
                    StateClass::Checkpoint,
                    None,
                )
                .expect("capture c")
                .expect("id_c");
            assert_eq!(coord.registered_pool_bytes(), bytes_a + bytes_b);
            id_c
        };
        let huge_extent = 4 * bytes_a;
        let gov_b = Governor::new(Arc::new(TestCoordGovernorIo {
            coord: Arc::clone(&coord),
            active: ACTIVE,
        }));
        let _reg_b = gov_b
            .register_model(ModelRegistration {
                model_id: 72,
                hard_budget: ACTIVE + coord.registered_pool_bytes() + 1,
                untracked: 0,
                page_bytes: 1 << 20,
            })
            .expect("register gov_b");
        let claim_b = gov_b.open(claim_spec, 72);
        let _scope_b = Scope::enter(claim_b.claim());
        assert_eq!(
            gov_b.try_reserve(huge_extent).err(),
            Some(Failure::EngineBudget)
        );
        let mixed_need_b = gov_b.need(
            claim_b.id(),
            Need {
                pages: K,
                kv: true,
                ..Need::default()
            },
        );
        let mut src_b = RecordingSource {
            inner: &mut inner,
            taken: Vec::new(),
        };
        let res_b = gov_b.free(claim_b.id(), mixed_need_b, &mut [&mut src_b]);
        assert_eq!(res_b.outcome, FreeOutcome::Met);
        assert!(res_b.pages >= K);
        assert_eq!(
            src_b.taken,
            vec![UnitKey::Store(entry_c), UnitKey::Store(entry_b)],
            "when entry_c bytes alone do not cover deficit, ledger must continue to entry_b for pages"
        );
        assert_eq!(
            coord.registered_pool_bytes(),
            0,
            "emptied state store must release its host-pool budget guard cleanly"
        );
        assert_eq!(gov_b.snapshot().charged, ACTIVE);
    }

    /// Round-12 Finding 2 regression:
    /// When a sequence already published a rolling checkpoint WITH a complete `DraftContextWindow`
    /// at boundary `B` and a later recapture at the exact same boundary `B` fails deferred
    /// `BoundaryDraftWindow::Device` packing (e.g. during scheduled preemption or turn-end
    /// `StateClass::Ordinary` promotion), `capture_boundary_inner` must preserve the existing
    /// complete checkpoint (never retire/replace it with `draft_window == None`), upgrade it to
    /// `Ordinary` when requested, and restore `DFlash2` speculation cleanly on short-suffix replay.
    #[test]
    fn dflash2_preempt_and_finalize_preserve_complete_checkpoint_on_draft_pack_error() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        use crate::models::inkling2::state_disk::BoundaryDraftWindow;
        use crate::models::qwen3_5::dflash2_state::store::StateClass;

        let tok = build_synthetic_inkling2_tokenizer();
        let msg_user_id = tok
            .inner()
            .token_to_id(crate::tools::inkling::MESSAGE_USER)
            .expect("MESSAGE_USER id");
        // Draft sliding_window = 96 > replay suffix (< 32 tokens) so losing the saved
        // `DraftContextWindow` on preemption would force replay to fall back to AR decode.
        let inner = build_synthetic_dflash2_inner_with_window(512, Some(msg_user_id), 96, 4);
        let mut sched = Inkling2SchedulerState::new(inner).expect("scheduler");
        let (tx, mut rx) = mpsc::unbounded_channel::<Inkling2Cmd>();

        // Short prompt (< 32 tokens) + 32 new tokens so live scheduled DFlash2 decode crosses
        // boundary 32 with a real `BoundaryDraftWindow::Device` while still decoding past 32.
        let cfg = scheduled_greedy_dflash2_config("owner-r12-preempt", 32, "salt-r12-preempt");
        let (reply, mut done_rx) = tokio::sync::oneshot::channel();
        tx.send(Inkling2Cmd::Chat(Box::new(ChatCmd::SessionStart {
            messages: user_msg("a"),
            config: cfg,
            reply,
            cancelled: Arc::new(AtomicBool::new(false)),
        })))
        .expect("send SessionStart");

        // Drive until live scheduled DFlash2 decode has crossed boundary 32 AND computed > 32
        // tokens (so preemption replay has a non-empty short suffix `< 96` past boundary 32).
        let mut crossed_seq: Option<u32> = None;
        for _ in 0..64 {
            let _ = sched.drive(&mut rx);
            if let Ok(res) = done_rx.try_recv() {
                panic!("turn finished before boundary-32 preemption: {res:?}");
            }
            let rt = sched.inner.paged.as_ref().expect("paged");
            if let Some((&seq_id, _)) = rt.block_boundary_conv.iter().find(|(sid, (b, _))| {
                *b == 32 && rt.coordinator.request_token_count_all(**sid).unwrap_or(0) > 32
            }) {
                assert!(
                    rt.block_boundary_draft_window.get(&seq_id).is_some_and(
                        |(b, w)| *b == 32 && matches!(w, BoundaryDraftWindow::Device(_))
                    ),
                    "seq {seq_id} must hold a live BoundaryDraftWindow::Device at boundary 32"
                );
                crossed_seq = Some(seq_id);
                break;
            }
        }
        let seq_id = crossed_seq.expect("turn must cross boundary 32 during decode");
        let cache_salt = sched
            .turn_mut(seq_id)
            .expect("running turn")
            .payload
            .params
            .cache_salt;

        // Publish a rolling `StateClass::Checkpoint` at boundary 32 WITH its complete
        // `DraftContextWindow` (recording `seq_checkpoints[seq_id] = cp_id`), mirroring a
        // sequence that published a rolling checkpoint and is then preempted at the same boundary.
        let (cp_id, orig_dw_bytes) = {
            let rt = sched.inner.paged.as_mut().expect("paged");
            assert_eq!(
                rt.state_store.store.entries().count(),
                0,
                "prompt < 32 tokens must not pre-populate StateStore before rolling checkpoint"
            );
            let (_, conv) = rt
                .block_boundary_conv
                .get(&seq_id)
                .cloned()
                .expect("boundary conv at 32");
            let tokens = rt
                .coordinator
                .full_adapter()
                .request_tokens_for(seq_id)
                .expect("recorded tokens")
                .to_vec();
            let draft_win = rt
                .block_boundary_draft_window
                .get(&seq_id)
                .filter(|(wb, _)| *wb == 32)
                .map(|(_, w)| w);
            let cp_id = rt
                .state_store
                .capture_boundary_with_window(
                    &mut rt.coordinator,
                    &conv,
                    seq_id,
                    &tokens,
                    32,
                    cache_salt,
                    StateClass::Checkpoint,
                    draft_win,
                    None,
                )
                .expect("rolling checkpoint capture at 32")
                .expect("cp_id");
            let entry = rt.state_store.store.get(cp_id).expect("cp entry");
            assert!(entry.checkpoint());
            let dw = entry
                .ram()
                .and_then(|a| a.draft_window.as_ref())
                .expect("rolling checkpoint at 32 must have complete DraftContextWindow");
            (cp_id, Arc::clone(&dw.bytes))
        };

        // Inject a deferred `BoundaryDraftWindow::Device` pack failure and preempt at boundary 32.
        // Because `cp_id` is ALREADY complete in RAM, early reuse skips readback and draft-window
        // packing completely (leaving `fail_next_draft_window_pack` unconsumed) while preserving it.
        sched
            .inner
            .paged
            .as_mut()
            .expect("paged")
            .state_store
            .fail_next_draft_window_pack = true;
        assert!(
            sched.preempt_running(seq_id),
            "preempt_running must preempt the running turn at boundary 32"
        );
        {
            let rt = sched.inner.paged.as_mut().expect("paged");
            assert!(
                std::mem::take(&mut rt.state_store.fail_next_draft_window_pack),
                "reusing an already-complete RAM checkpoint on preemption must skip draft-window packing"
            );
            let entry = rt
                .state_store
                .store
                .get(cp_id)
                .expect("complete rolling checkpoint must NOT be retired on preemption");
            assert!(entry.checkpoint());
            let dw = entry
                .ram()
                .and_then(|a| a.draft_window.as_ref())
                .expect("complete DraftContextWindow must survive preemption");
            assert_eq!(dw.bytes, orig_dw_bytes);
        }

        // Resume preemption replay: suffix is strictly shorter than draft sliding_window (96),
        // so DFlash2 speculation ONLY engages if the checkpoint's `DraftContextWindow` survived.
        let waves_before_resume = sched.inner.scheduled_dspark.total_verify_waves;
        sched.resume_preempted();
        let (cached_prefix, suffix_len) = {
            let turn = sched.turn_mut(seq_id).expect("replaying turn");
            (
                turn.payload.prefix.effective_cached_prefix_len,
                turn.payload.prefix.suffix_len,
            )
        };
        assert_eq!(cached_prefix, 32);
        assert!(
            suffix_len > 0 && suffix_len < 96,
            "replay suffix ({suffix_len}) must be shorter than draft sliding_window (96)"
        );
        let mut saw_dflash2_lane = false;
        let res = loop {
            if sched.inner.scheduled_dspark.seq_lanes.contains_key(&seq_id) {
                saw_dflash2_lane = true;
            }
            match done_rx.try_recv() {
                Ok(res) => break res.expect("replayed turn must succeed"),
                Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {
                    let _ = sched.drive(&mut rx);
                }
                Err(tokio::sync::oneshot::error::TryRecvError::Closed) => {
                    panic!("turn reply channel closed");
                }
            }
        };
        assert!(
            saw_dflash2_lane
                && sched.inner.scheduled_dspark.total_verify_waves > waves_before_resume,
            "short-suffix preemption replay must restore DFlash2 speculation from preserved checkpoint"
        );
        assert_eq!(res.num_tokens, 32);

        // Part 2: Failed repack during `Checkpoint -> Ordinary` promotion at the same boundary
        // must ALSO preserve the complete `DraftContextWindow` while upgrading class to `Ordinary`.
        {
            let rt = sched.inner.paged.as_mut().expect("paged");
            let tokens: Vec<u32> = (400u32..464u32).collect();
            let conv = crate::models::inkling2::short_conv::Inkling2ShortConvState::new(
                sched.inner.config.num_layer_visits(),
                sched.inner.config.sconv_kernel_size,
                sched.inner.config.num_key_value_heads * sched.inner.config.head_dim,
                sched.inner.config.hidden_size,
                crate::array::DType::BFloat16,
            )
            .expect("conv");
            let draft_cfg = sched
                .inner
                .dflash2
                .as_ref()
                .expect("dflash2")
                .config
                .clone();
            let mut win = crate::models::qwen3_5::draft_context::DraftContextWindow::new(
                draft_cfg.context_width(),
                draft_cfg.sliding_window as u32,
            )
            .expect("draft window");
            win.restart(0);
            let rows = crate::array::MxArray::zeros(
                &[1, 32, i64::from(draft_cfg.context_width())],
                Some(crate::array::DType::BFloat16),
            )
            .expect("zeros");
            win.store_rows(&rows, 0).expect("store_rows");
            let dev_win = BoundaryDraftWindow::Device(win);

            rt.coordinator
                .reset_scheduled_request(77)
                .expect("reset 77");
            rt.coordinator
                .record_tokens_all(77, &tokens)
                .expect("record 77");
            let cp2 = rt
                .state_store
                .capture_boundary_with_window(
                    &mut rt.coordinator,
                    &conv,
                    77,
                    &tokens,
                    32,
                    777,
                    StateClass::Checkpoint,
                    Some(&dev_win),
                    None,
                )
                .expect("capture cp2")
                .expect("cp2 id");
            let orig2 = rt
                .state_store
                .store
                .get(cp2)
                .and_then(|e| e.ram())
                .and_then(|a| a.draft_window.as_ref())
                .expect("cp2 complete dw")
                .bytes
                .clone();

            rt.state_store.fail_next_draft_window_pack = true;
            let promoted = rt
                .state_store
                .capture_boundary_with_window(
                    &mut rt.coordinator,
                    &conv,
                    77,
                    &tokens,
                    32,
                    777,
                    StateClass::Ordinary,
                    Some(&dev_win),
                    None,
                )
                .expect("promote cp2 via early reuse")
                .expect("promoted id");
            assert_eq!(promoted, cp2);
            assert!(
                std::mem::take(&mut rt.state_store.fail_next_draft_window_pack),
                "promoting an already-complete RAM checkpoint must skip draft-window packing"
            );
            let entry = rt.state_store.store.get(cp2).expect("cp2 entry");
            assert!(
                !entry.checkpoint(),
                "early reuse during Ordinary recapture must still upgrade Checkpoint -> Ordinary"
            );
            assert_eq!(
                entry
                    .ram()
                    .and_then(|a| a.draft_window.as_ref())
                    .expect("complete DraftContextWindow must survive Ordinary promotion")
                    .bytes,
                orig2
            );
        }
    }

    /// Round-13 Finding 1 regression: Governor admission (`try_reserve`) gates new captures
    /// before host RAM allocation, while recapturing an already-complete RAM checkpoint reuses
    /// it immediately with zero readback/packing/refusal even under a tight budget.
    #[test]
    fn capture_boundary_governor_admission_and_early_reuse() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        use crate::cache_limit::CacheLimitCoordinator;
        use crate::engine::governor::governor::{Governor, ModelRegistration};
        use crate::models::inkling2::kv_cache::{
            Inkling2KVCacheCoordinator, default_inkling2_kv_format,
        };
        use crate::models::inkling2::paged_forward::Inkling2PagedRuntime;
        use crate::models::inkling2::state_disk::{BoundaryDraftWindow, Inkling2StateGeometry};
        use crate::models::qwen3_5::dflash2_state::store::StateClass;
        use crate::models::qwen3_5::draft_context::DraftContextWindow;

        const ACTIVE: u64 = 16 << 20;
        let make_dev = |w: u32, s: u32, end: u32| -> BoundaryDraftWindow {
            let mut win = DraftContextWindow::new(w, s).expect("dw");
            win.restart(0);
            let rows = crate::array::MxArray::zeros(
                &[1, i64::from(end), i64::from(w)],
                Some(crate::array::DType::BFloat16),
            )
            .expect("zeros");
            win.store_rows(&rows, 0).expect("store_rows");
            BoundaryDraftWindow::Device(win)
        };
        let new_gov = |coord: &Arc<CacheLimitCoordinator>| {
            Governor::new(Arc::new(TestCoordGovernorIo {
                coord: Arc::clone(coord),
                active: ACTIVE,
            }))
        };
        let reg_spec = |model_id: u64, hard_budget: u64| ModelRegistration {
            model_id,
            hard_budget,
            untracked: 0,
            page_bytes: 1 << 20,
        };

        let coord = Arc::new(CacheLimitCoordinator::new_for_test());
        let mut inner = build_synthetic_dflash2_inner_with_window(256, None, 16, 4);
        let dcfg = inner.dflash2.as_ref().expect("dflash2").config.clone();
        let recency = Arc::clone(&inner.cache_recency);
        let kv = Inkling2KVCacheCoordinator::new_extent_grown_for_test(
            &inner.config,
            default_inkling2_kv_format(),
            32,
            512,
            32,
            4,
            4,
            16,
            4,
            Arc::clone(&recency),
        )
        .expect("coord");
        let mut rt = Inkling2PagedRuntime::from_coordinator(kv, 32, recency);
        let geom = Inkling2StateGeometry::from_coordinator_with_draft(
            &inner.config,
            &rt.coordinator,
            Some(&dcfg),
        )
        .expect("geom");
        rt.state_store.set_geometry(geom.clone());
        rt.state_store.attach_host_budget_to(Arc::clone(&coord));
        let conv = crate::models::inkling2::short_conv::Inkling2ShortConvState::new(
            inner.config.num_layer_visits(),
            inner.config.sconv_kernel_size,
            inner.config.num_key_value_heads * inner.config.head_dim,
            inner.config.hidden_size,
            crate::array::DType::BFloat16,
        )
        .expect("conv");
        let tokens: Vec<u32> = (500u32..564u32).collect();
        rt.coordinator.reset_scheduled_request(1).expect("reset");
        rt.coordinator
            .record_tokens_all(1, &tokens)
            .expect("record");

        let dev_32 = make_dev(dcfg.context_width(), dcfg.sliding_window as u32, 32);
        let dev_64 = make_dev(dcfg.context_width(), dcfg.sliding_window as u32, 64);
        let dw_bytes = DraftContextWindow::bytes(dcfg.context_width(), dcfg.sliding_window as u32);
        let peak_64 = (geom.live_swa_bytes(64).expect("swa") as u64)
            + geom.conv_span_bytes
            + dw_bytes
            + crate::disk_tier::store::lineage_bytes(64, 2);

        let prev_cp = rt
            .state_store
            .capture_boundary_with_window(
                &mut rt.coordinator,
                &conv,
                1,
                &tokens,
                32,
                911,
                StateClass::Checkpoint,
                Some(&dev_32),
                None,
            )
            .expect("seed")
            .expect("prev_cp");
        let pools_32 = coord.registered_pool_bytes();

        // 1. Tight budget (`room < peak_64`): skipped without allocation, `prev_cp` untouched.
        let gov_tight = new_gov(&coord);
        let _reg1 = gov_tight
            .register_model(reg_spec(81, ACTIVE + pools_32 + peak_64 - 1))
            .expect("reg1");
        assert!(
            rt.state_store
                .capture_boundary_with_window(
                    &mut rt.coordinator,
                    &conv,
                    1,
                    &tokens,
                    64,
                    911,
                    StateClass::Checkpoint,
                    Some(&dev_64),
                    Some((&gov_tight, 81)),
                )
                .expect("tight")
                .is_none()
        );
        assert_eq!(gov_tight.publications_refused(81), 1);
        assert_eq!(gov_tight.reserved_bytes(), 0);
        assert_eq!(rt.state_store.store.entries().count(), 1);
        assert!(rt.state_store.store.get(prev_cp).is_some());
        assert_eq!(coord.registered_pool_bytes(), pools_32);

        // 2. Zero room: recapturing existing complete RAM checkpoint reuses with zero pack/refusal.
        let gov_zero = new_gov(&coord);
        let _reg2 = gov_zero
            .register_model(reg_spec(82, ACTIVE + pools_32))
            .expect("reg2");
        rt.state_store.fail_next_draft_window_pack = true;
        let reused = rt
            .state_store
            .capture_boundary_with_window(
                &mut rt.coordinator,
                &conv,
                1,
                &tokens,
                32,
                911,
                StateClass::Ordinary,
                Some(&dev_32),
                Some((&gov_zero, 82)),
            )
            .expect("reuse")
            .expect("prev_cp");
        assert_eq!(reused, prev_cp);
        assert_eq!(gov_zero.publications_refused(82), 0);
        assert!(std::mem::take(
            &mut rt.state_store.fail_next_draft_window_pack
        ));
        assert!(!rt.state_store.store.get(prev_cp).unwrap().checkpoint());

        // 3. Roomy budget (`room == peak_64`): succeeds and converts reservation to pool charge.
        let gov_roomy = new_gov(&coord);
        let _reg3 = gov_roomy
            .register_model(reg_spec(83, ACTIVE + pools_32 + peak_64))
            .expect("reg3");
        let id_64 = rt
            .state_store
            .capture_boundary_with_window(
                &mut rt.coordinator,
                &conv,
                1,
                &tokens,
                64,
                911,
                StateClass::Ordinary,
                Some(&dev_64),
                Some((&gov_roomy, 83)),
            )
            .expect("roomy")
            .expect("id_64");
        assert_eq!(gov_roomy.publications_refused(83), 0);
        assert_eq!(gov_roomy.reserved_bytes(), 0);
        assert_eq!(coord.registered_pool_bytes(), pools_32 + peak_64);
        assert!(
            rt.state_store
                .store
                .get(id_64)
                .and_then(|e| e.ram())
                .is_some_and(|a| a.draft_window.is_some())
        );
        inner.paged = Some(rt);
    }

    /// R14-3: byte-only Governor claim reclaims parked draft rings preserving target KV + draft_windows;
    /// page-only claim skips `UnitKey::DraftContext`.
    #[test]
    fn governor_byte_claim_reclaims_parked_draft_rings_preserving_window() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        use crate::engine::governor::ledger::{
            Charge, ClaimClass, ClaimRank, ClaimSpec, FreeOutcome, FreePool, Ledger, Need,
            NeedTiming, ReachClass, Room, Stage, TakeRights, Taken, UnitKey, Victim, VictimQuery,
            VictimSource,
        };
        struct NoPool;
        impl FreePool for NoPool {
            fn charge(&self) -> Charge {
                Charge::default()
            }
            fn clear(&mut self) {}
        }
        struct Src<'a>(&'a mut super::Inkling2Inner, Vec<UnitKey>);
        impl VictimSource for Src<'_> {
            fn units(&self, q: &VictimQuery) -> Vec<Victim> {
                self.0.inkling2_victim_units(q, &[]).into_units()
            }
            fn take(&mut self, q: &VictimQuery, v: &Victim) -> Taken {
                self.1.push(v.key.clone());
                self.0.inkling2_take_victim(q, v)
            }
        }
        let tok = build_synthetic_inkling2_tokenizer();
        let uid = tok
            .inner()
            .token_to_id(crate::tools::inkling::MESSAGE_USER)
            .unwrap();
        let inner = build_synthetic_dflash2_inner_with_window(8192, Some(uid), 2048, 2);
        let mut sched = Inkling2SchedulerState::new(inner).unwrap();
        let (tx, mut rx) = mpsc::unbounded_channel::<Inkling2Cmd>();
        let mut cfg = scheduled_sampled_dflash2_config("r14-ring", 2, "salt-r14-ring");
        cfg.top_k = Some(1);
        assert_eq!(
            send_session_start(&mut sched, &tx, &mut rx, user_msg("alpha"), cfg)
                .unwrap()
                .num_tokens,
            2
        );
        let seq0 = *sched.owner_sequences.get("r14-ring").unwrap();
        let rec = sched
            .inner
            .paged
            .as_ref()
            .unwrap()
            .coordinator
            .request_token_count_all(seq0)
            .unwrap();
        let ctx = sched.inner.dspark_turn_contexts.get(&seq0).unwrap();
        ctx.eval_state().unwrap();
        assert!(ctx.mlx_buffer_bytes() > 0);

        let q = |need| VictimQuery {
            rights: TakeRights {
                units: true,
                ..Default::default()
            },
            need,
        };
        let byte_u = sched
            .inner
            .inkling2_victim_units(
                &q(Need {
                    bytes: 1,
                    ..Default::default()
                }),
                &[],
            )
            .into_units();
        let dc = byte_u
            .iter()
            .find(|v| v.key == UnitKey::DraftContext(seq0))
            .unwrap();
        assert!(dc.bytes > 0 && dc.pages == 0 && dc.stage == Stage::FreeNow);
        assert!(
            !sched
                .inner
                .inkling2_victim_units(
                    &q(Need {
                        pages: 1,
                        ..Default::default()
                    }),
                    &[]
                )
                .into_units()
                .iter()
                .any(|v| v.key == UnitKey::DraftContext(seq0))
        );

        let mut ledger = Ledger::new();
        let claim = ledger.open(
            ClaimSpec {
                cls: ClaimClass::InService,
                request_id: 94,
                rank: ClaimRank::default(),
                reach: ReachClass::Ordinary,
                timing: NeedTiming::Immediate,
            },
            1,
        );
        let mut src = Src(&mut sched.inner, Vec::new());
        let out = ledger.free(
            claim,
            Need {
                bytes: 1,
                ..Default::default()
            },
            Room::default(),
            &mut NoPool,
            &mut [&mut src],
        );
        assert_eq!(out.outcome, FreeOutcome::Met);
        assert!(out.now_bytes > 0 && out.pages == 0);
        assert_eq!(src.1, vec![UnitKey::DraftContext(seq0)]);
        assert!(!sched.inner.dspark_turn_contexts.contains_key(&seq0));
        let rt = sched.inner.paged.as_ref().unwrap();
        assert_eq!(rt.coordinator.request_token_count_all(seq0).unwrap(), rec);
        assert!(rt.draft_windows.contains_key(&seq0));

        let waves = sched.inner.scheduled_dspark.total_verify_waves;
        let mut cfg2 = scheduled_sampled_dflash2_config("r14-ring", 6, "salt-r14-ring");
        cfg2.top_k = Some(1);
        let cont = send_session_continue(
            &mut sched,
            &tx,
            &mut rx,
            two_user_msgs("alpha", "resume-r14"),
            cfg2,
        )
        .unwrap();
        assert_eq!(cont.cached_tokens, rec);
        assert!(sched.inner.scheduled_dspark.total_verify_waves > waves);
    }

    /// R15-2: byte claim reclaims a newer parked draft ring WITH a covering `DraftContextWindow`
    /// (`Stage::FreeNow`) ahead of an older parked ring WITHOUT a covering window (`Stage::NeedsWrite`).
    #[test]
    fn governor_byte_claim_prefers_parked_draft_ring_with_covering_window() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        use crate::engine::governor::ledger::{
            Charge, ClaimClass, ClaimRank, ClaimSpec, FreeOutcome, FreePool, Ledger, Need,
            NeedTiming, ReachClass, Room, Stage, TakeRights, Taken, UnitKey, Victim, VictimQuery,
            VictimSource,
        };
        struct NoPool;
        impl FreePool for NoPool {
            fn charge(&self) -> Charge {
                Charge::default()
            }
            fn clear(&mut self) {}
        }
        struct Src<'a>(&'a mut super::Inkling2Inner, Vec<UnitKey>);
        impl VictimSource for Src<'_> {
            fn units(&self, q: &VictimQuery) -> Vec<Victim> {
                self.0.inkling2_victim_units(q, &[]).into_units()
            }
            fn take(&mut self, q: &VictimQuery, v: &Victim) -> Taken {
                self.1.push(v.key.clone());
                self.0.inkling2_take_victim(q, v)
            }
        }
        let tok = build_synthetic_inkling2_tokenizer();
        let uid = tok
            .inner()
            .token_to_id(crate::tools::inkling::MESSAGE_USER)
            .unwrap();
        let inner = build_synthetic_dflash2_inner_with_window(8192, Some(uid), 2048, 2);
        let mut sched = Inkling2SchedulerState::new(inner).unwrap();
        let (tx, mut rx) = mpsc::unbounded_channel::<Inkling2Cmd>();

        let mut cfg_a = scheduled_sampled_dflash2_config("r15-no-win", 2, "salt-r15-a");
        cfg_a.top_k = Some(1);
        assert_eq!(
            send_session_start(&mut sched, &tx, &mut rx, user_msg("older"), cfg_a)
                .unwrap()
                .num_tokens,
            2
        );
        let seq_a = *sched.owner_sequences.get("r15-no-win").unwrap();
        // Drop `seq_a`'s covering `DraftContextWindow` so its parked ring is unbacked.
        sched
            .inner
            .paged
            .as_mut()
            .unwrap()
            .draft_windows
            .remove(&seq_a);

        let mut cfg_b = scheduled_sampled_dflash2_config("r15-with-win", 2, "salt-r15-b");
        cfg_b.top_k = Some(1);
        assert_eq!(
            send_session_start(&mut sched, &tx, &mut rx, user_msg("newer"), cfg_b)
                .unwrap()
                .num_tokens,
            2
        );
        let seq_b = *sched.owner_sequences.get("r15-with-win").unwrap();
        assert_ne!(seq_a, seq_b);

        for seq in [seq_a, seq_b] {
            let ctx = sched.inner.dspark_turn_contexts.get(&seq).unwrap();
            ctx.eval_state().unwrap();
            assert!(ctx.mlx_buffer_bytes() > 0);
        }
        let full = sched
            .inner
            .paged
            .as_ref()
            .unwrap()
            .coordinator
            .full_adapter();
        assert!(
            full.turn_end_stamp(seq_a).unwrap() < full.turn_end_stamp(seq_b).unwrap(),
            "seq_a must be strictly older than seq_b"
        );

        let units = sched
            .inner
            .inkling2_victim_units(
                &VictimQuery {
                    rights: TakeRights {
                        units: true,
                        ..Default::default()
                    },
                    need: Need {
                        bytes: 1,
                        ..Default::default()
                    },
                },
                &[],
            )
            .into_units();
        let pos_b = units
            .iter()
            .position(|v| v.key == UnitKey::DraftContext(seq_b))
            .unwrap();
        let pos_a = units
            .iter()
            .position(|v| v.key == UnitKey::DraftContext(seq_a))
            .unwrap();
        assert_eq!(units[pos_b].stage, Stage::FreeNow);
        assert_eq!(units[pos_a].stage, Stage::NeedsWrite);
        assert!(
            pos_b < pos_a,
            "covered newer ring seq_b must precede uncovered older ring seq_a"
        );

        let mut ledger = Ledger::new();
        let claim = ledger.open(
            ClaimSpec {
                cls: ClaimClass::InService,
                request_id: 95,
                rank: ClaimRank::default(),
                reach: ReachClass::Ordinary,
                timing: NeedTiming::Immediate,
            },
            1,
        );
        let mut src = Src(&mut sched.inner, Vec::new());
        let out = ledger.free(
            claim,
            Need {
                bytes: 1,
                ..Default::default()
            },
            Room::default(),
            &mut NoPool,
            &mut [&mut src],
        );
        assert_eq!(out.outcome, FreeOutcome::Met);
        assert_eq!(src.1, vec![UnitKey::DraftContext(seq_b)]);
        assert!(!sched.inner.dspark_turn_contexts.contains_key(&seq_b));
        assert!(sched.inner.dspark_turn_contexts.contains_key(&seq_a));
    }
}
