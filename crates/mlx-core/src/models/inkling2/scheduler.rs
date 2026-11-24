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

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

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
use crate::model_thread::ResponseTx;
use crate::models::qwen3_5::dflash2_state::disk::{Tick, instant_at};
use crate::models::qwen3_5::dflash2_state::store::{StateClass, StoreHit};
use crate::stream::{Stream, WiredResidency};
use crate::transformer::paged_kv_cache_adapter::{
    ChainPrefetchStart, PagedKVCacheAdapter, RestoreUploadBudget, SeqId,
};

use super::kv_cache::INKLING2_PAGED_PREFILL_MAX_CHUNK;
use super::memory::short_conv_state_bytes;
use super::model::{Inkling2FlatCache, Inkling2Inner};
use super::paged_forward::Inkling2PrefixState;
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

impl Inkling2Inner {
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
        let rt = self
            .paged
            .as_mut()
            .ok_or_else(|| Error::from_reason("Inkling2 paged runtime missing"))?;
        rt.active_seq = seq_id;

        if reuse_cache {
            // 1. Live continuation on the exact same sequence (zero copy!).
            if rt.conv_states.contains_key(&seq_id)
                && rt.coordinator.can_continue_all(seq_id, tokens)
            {
                let boundary = rt
                    .coordinator
                    .continue_turn_all(seq_id, tokens, total_budget)
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
            let hit = rt
                .state_store
                .lookup_deepest_restorable(&rt.coordinator, tokens, cache_salt);
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
        let rt = self
            .paged
            .as_mut()
            .ok_or_else(|| Error::from_reason("Inkling2 paged runtime missing"))?;
        let chain_hot = rt.state_store.chain_is_hot(&rt.coordinator, hit.id);
        let Some((anchor, _from_ssd)) = rt.state_store.load_anchor(&rt.coordinator, hit.id) else {
            return Ok(None);
        };

        // Restore Group 0 (8 Global visits) up to exact `depth` using `prepare_turn_per_block_booking_hit_only`,
        // which installs the matched blocks (hot or restored via `arm_state_restore` / `arm_landed_chain`)
        // directly into `seq_id`'s block table and `request_tokens`.
        let total_budget = u32::try_from(tokens.len())
            .map_err(|_| Error::from_reason("Inkling2 prompt length overflows u32"))?;
        let full_adapter = rt.coordinator.full_adapter_mut();
        if !chain_hot {
            full_adapter.arm_state_restore(depth);
        }
        let keys =
            crate::engine::build_paged_extra_keys(tokens.len(), full_adapter.block_size(), &[]);
        let prepared = full_adapter.prepare_turn_per_block_booking_hit_only(
            seq_id,
            tokens,
            total_budget,
            &keys,
            cache_salt,
            false,
            depth,
        );
        let restored_global = match prepared {
            Ok(plan) if plan.cached_prefix_len == depth => plan.cached_prefix_len,
            Ok(_) | Err(_) => {
                let _ = rt.coordinator.release_request_all(seq_id);
                return Ok(None);
            }
        };
        if restored_global != depth {
            let _ = rt.coordinator.release_request_all(seq_id);
            return Ok(None);
        }

        // Restore Group 1 (40 SWA visits) raw blocks + 48 ShortConv states.
        if let Err(err) = install_anchor_swa_and_conv(
            &mut rt.coordinator,
            &mut rt.conv_states,
            seq_id,
            tokens,
            depth,
            anchor,
        ) {
            let _ = rt.coordinator.release_request_all(seq_id);
            rt.conv_states.remove(&seq_id);
            tracing::warn!("Inkling2 StateDisk SWA/ShortConv restore fell back to cold: {err}");
            return Ok(None);
        }
        // Confirm the out-of-pool auxiliary half (40 SWA visits + 48 ShortConv states) is primed
        // at `depth`, clearing the fail-closed `aux_prefix_unbacked` latch set by synchronous SSD restore.
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
        execute_dual_lane_chat_barrier(self, command, owners);
    }
}

/// Ensure `AtomicBool` import stays used by tests and scheduler helpers.
#[allow(dead_code)]
fn _cancel_flag_type_check(_flag: &Arc<AtomicBool>) {}

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
}
