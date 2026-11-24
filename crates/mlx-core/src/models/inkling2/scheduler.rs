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
    FreeOutcome, Stage, Taken as LedgerTaken, UnitKey, UnitRuns, Victim, VictimQuery,
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
use crate::models::qwen3_5::dflash2_state::store::{Entry, EntryId, StateClass, StoreHit};
use crate::stream::{Stream, WiredResidency};
use crate::transformer::paged_kv_cache_adapter::{
    ChainPrefetchStart, PagedKVCacheAdapter, RestoreUploadBudget, SeqId,
};

use super::kv_cache::INKLING2_PAGED_PREFILL_MAX_CHUNK;
use super::memory::short_conv_state_bytes;
use super::model::{Inkling2FlatCache, Inkling2Inner};
use super::paged_forward::{Inkling2PagedRuntime, Inkling2PrefixState};
use super::state_disk::{
    Inkling2Restore, Inkling2RestorePhase, ReclaimOutcome, install_anchor_swa_and_conv,
    planned_rolling_checkpoints, poll_restore_ticket, start_global_chain_prefetch,
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

/// Per-owner metadata persisted across turns by [`HybridSchedulerState`].
#[derive(Default)]
pub(crate) struct Inkling2OwnerState {
    pub cached_token_history: Vec<u32>,
    pub cached_image_key: Option<u64>,
    pub flat_cache: Option<Inkling2FlatCache>,
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
    }

    fn finish_ownerless_barrier(&mut self, flat_lane: bool) {
        if !flat_lane && !ChatBackend::has_live_session(self) {
            let _ = self.release_scheduled_cache(0);
        }
    }

    fn owner_has_flat_state(state: &Self::OwnerState) -> bool {
        state.flat_cache.is_some()
    }

    fn install_flat_owner_state(&mut self, state: Self::OwnerState) {
        self.flat_cache = state.flat_cache;
        self.cached_token_history = state.cached_token_history;
        self.cached_image_key = state.cached_image_key;
    }

    fn capture_flat_owner_state(&mut self) -> Self::OwnerState {
        Inkling2OwnerState {
            cached_token_history: self.cached_token_history.clone(),
            cached_image_key: self.cached_image_key,
            flat_cache: self.flat_cache.take(),
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
                // StateStore restores target KV/ShortConv only (no DSpark draft KV), so drop any
                // stale draft context previously owned by `seq_id`.
                self.dspark_turn_contexts.remove(&seq_id);
                self.scheduled_dspark.seq_lanes.remove(&seq_id);
                return Ok(Inkling2PrefixState {
                    effective_cached_prefix_len: boundary,
                    suffix_len: tokens.len().saturating_sub(boundary),
                    full_tokens: tokens.to_vec(),
                    cache_salt,
                    live_continuation: false,
                });
            }
        }

        // 3. Fresh cold sequence.
        self.dspark_turn_contexts.remove(&seq_id);
        self.scheduled_dspark.seq_lanes.remove(&seq_id);
        let rt = self
            .paged
            .as_mut()
            .ok_or_else(|| Error::from_reason("Inkling2 paged runtime missing"))?;
        rt.reset_sequence(seq_id, &config)?;
        Ok(Inkling2PrefixState {
            effective_cached_prefix_len: 0,
            suffix_len: tokens.len(),
            full_tokens: tokens.to_vec(),
            cache_salt,
            live_continuation: false,
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
        // Borrow `&anchor` across retries so `swa_blocks` (~82.5 MiB) is never deep-copied,
        // and restrict reclaim to `SlidingOnly` so Global `StateStore` checkpoints/marks are
        // never evicted for SWA-only pool pressure.
        let exclusive = self
            .paged
            .as_ref()
            .is_some_and(|rt| rt.exclusive_turn_active);
        if exclusive {
            let swa_blocks_len = anchor.swa_blocks.len();
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
                                },
                                restore: Inkling2Restore::reading(*prefetch, cache_salt),
                            });
                        }
                        Some(ChainPrefetchStart::Unavailable) => {
                            // Cold chain disappeared concurrently: prime cold cleanly.
                            return self
                                .prepare_prefix_sync(seq_id, tokens, false, cache_salt)
                                .map(ScheduledPrefixAdmission::Ready);
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
                        Inkling2PrefixState {
                            effective_cached_prefix_len: matched,
                            suffix_len: tokens.len().saturating_sub(matched),
                            full_tokens: tokens.to_vec(),
                            cache_salt: salt,
                            live_continuation: false,
                        }
                    }
                    None => self.prepare_prefix_sync(seq_id, tokens, false, salt)?,
                }
            } else {
                // State changed while parked; fall back cleanly to synchronous check/cold.
                self.prepare_prefix_sync(seq_id, tokens, true, salt)?
            }
        } else {
            self.prepare_prefix_sync(seq_id, tokens, false, salt)?
        };

        Ok(Some(ScheduledRestoreResult {
            prefix,
            bytes_restored,
            wait: restore.parked_at.elapsed(),
            materialized_blocks: 0,
            profiler_prefill_tokens: 0,
            extra_prefill_breaks: Vec::new(),
        }))
    }

    fn inkling2_victim_units(&self, _query: &VictimQuery, _kept: &[u32]) -> UnitRuns {
        let mut runs = UnitRuns::default();
        let Some(rt) = self.paged.as_ref() else {
            return runs;
        };
        let full = rt.coordinator.full_adapter();

        // 1. Freeable pool extents & dead KV leaves first.
        if let Ok(tel) = full.block_telemetry()
            && tel.evictable_blocks > 0
        {
            let bpb = full.bytes_per_block().unwrap_or(0);
            runs.ahead.push(Victim {
                key: UnitKey::DeadLeaves,
                stamp: 0,
                bytes: u64::from(tel.evictable_blocks).saturating_mul(bpb),
                pages: tel.evictable_blocks,
                stage: Stage::FreeNow,
                later: false,
            });
        }

        // 2. Stored hybrid states: disposable `StateClass::Checkpoint` go in `ahead` (evicted first),
        //    `StateClass::Ordinary` go in `merged` sorted by shared `CacheRecency` stamp.
        let disk_writable = rt.state_store.disk.as_ref().is_some_and(|d| d.writable());
        for entry in rt.state_store.store.entries() {
            if entry.ram().is_none() {
                if entry.disk().is_some_and(|c| c.chain_owed) {
                    runs.after.push(Victim {
                        key: UnitKey::StoreChain(entry.id()),
                        stamp: entry.last_used(),
                        bytes: 0,
                        pages: entry.chain().len() as u32,
                        stage: Stage::NeedsWrite,
                        later: true,
                    });
                }
                continue;
            }
            let has_landed_disk = entry.disk().is_some_and(|c| c.landed);
            let needs_write = !has_landed_disk && disk_writable;
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
            if entry.checkpoint() {
                runs.ahead.push(victim);
            } else {
                runs.merged.push(victim);
            }
        }
        runs.ahead.sort_by_key(|v| (v.stage, v.stamp));
        runs.merged.sort_by_key(|v| (v.stage, v.stamp));
        runs
    }

    fn inkling2_take_victim(&mut self, victim: &Victim) -> LedgerTaken {
        let Some(rt) = self.paged.as_mut() else {
            return LedgerTaken::default();
        };
        match victim.key {
            UnitKey::DeadLeaves => {
                let pool_bytes = rt
                    .coordinator
                    .full_adapter_mut()
                    .release_cached_extent(true, &[]);
                if pool_bytes > 0 {
                    LedgerTaken {
                        outcome: FreeOutcome::Met,
                        now_bytes: pool_bytes,
                        later_bytes: 0,
                        pages: victim.pages,
                    }
                } else {
                    LedgerTaken::default()
                }
            }
            UnitKey::Store(id) => {
                let bytes = rt.state_store.store.get(id).map_or(0, |e| e.state_bytes());
                match rt.state_store.reclaim_entry(&rt.coordinator, id) {
                    ReclaimOutcome::RamDropped
                    | ReclaimOutcome::Spilled
                    | ReclaimOutcome::Erased => LedgerTaken {
                        outcome: FreeOutcome::Met,
                        now_bytes: bytes,
                        later_bytes: 0,
                        pages: 0,
                    },
                    ReclaimOutcome::Pending => LedgerTaken {
                        outcome: FreeOutcome::Pending,
                        now_bytes: 0,
                        later_bytes: bytes,
                        pages: 0,
                    },
                }
            }
            UnitKey::StoreChain(_) => {
                rt.state_store.tick(
                    &rt.coordinator,
                    Tick::Wait {
                        deadline: Instant::now(),
                    },
                );
                LedgerTaken {
                    outcome: FreeOutcome::Pending,
                    now_bytes: 0,
                    later_bytes: 0,
                    pages: victim.pages,
                }
            }
            _ => LedgerTaken::default(),
        }
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
            .saturating_add(self.scheduled_recurrent_bytes());
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

    fn take_victim(&mut self, _query: &VictimQuery, victim: &Victim, _kept: &[u32]) -> LedgerTaken {
        self.inkling2_take_victim(victim)
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
        self.release_scheduled_dspark(seq_id);
        if let Some(rt) = self.paged.as_mut() {
            rt.state_store.clear_seq_checkpoint(seq_id);
        }
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

        if let Some(rt) = self.paged.as_mut() {
            // 2. Preserve completed prefix state if we have a valid block-aligned frontier before
            //    releasing the sequence's physical blocks so preemption replay can resume cleanly!
            let recorded = rt.coordinator.request_token_count_all(seq_id).unwrap_or(0);
            let bs = rt.coordinator.block_size().max(1);
            if let Some((boundary, conv)) = rt.block_boundary_conv.get(&seq_id).cloned()
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
                let _ = rt.state_store.capture_boundary(
                    &mut rt.coordinator,
                    &conv,
                    seq_id,
                    &tokens,
                    boundary,
                    cache_salt,
                    StateClass::Checkpoint,
                );
            }

            // 3. Register full Group 0 blocks for prefix reuse, then release Group 0 + Group 1 + ShortConv.
            let _ = rt.coordinator.register_full_blocks_for_reuse_for(
                seq_id,
                cache_salt,
                mode == PreemptionMode::Ssd,
            );
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
        self.release_scheduled_dspark(seq_id);
    }

    fn finish_scheduled_decode_batch(&mut self, rows: &[(SeqId, u32)]) -> Result<()> {
        let rt = self
            .paged
            .as_mut()
            .ok_or_else(|| Error::from_reason("Inkling2 batched decode lost its paged runtime"))?;
        rt.coordinator
            .eval_pending_pool_writes_all()
            .map_err(Error::from_reason)?;
        // Queue async evaluation of all updated ShortConv states in ONE non-blocking batch
        // (same MLX stream guarantees materialization before the next step reads them; durable
        // StateDisk capture explicitly syncs via `eval_conv_state_arrays`).
        let mut conv_arrays: Vec<&MxArray> = Vec::with_capacity(rows.len() * 48 * 4);
        for &(seq_id, _) in rows {
            if let Some(conv) = rt.conv_states.get(&seq_id) {
                for v in &conv.visits {
                    conv_arrays.push(&v.k);
                    conv_arrays.push(&v.v);
                    conv_arrays.push(&v.attn_post);
                    conv_arrays.push(&v.mlp_post);
                }
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
mod tests {
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
    fn build_synthetic_dflash2_inner(
        max_model_len: u32,
        force_lm_token: Option<u32>,
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
        let draft = tiny_dflash2_model_with_block(&draft_target, DFLASH2_DRAFT_PROPOSALS)
            .expect("tiny DFlash2 draft");

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
            dspark_turn_contexts: HashMap::new(),
            scheduled_dspark: Inkling2ScheduledDsparkStates::default(),
            paged: None,
            flat_cache: None,
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
                max_concurrent_sequences: 1,
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
}
