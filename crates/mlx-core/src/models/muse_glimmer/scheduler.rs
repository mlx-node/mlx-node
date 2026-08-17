use napi::bindgen_prelude::{Error, Result};

use super::{MuseGlimmerCmd, MuseGlimmerInner, MusePrefixState};
use crate::array::MxArray;
use crate::engine::backend::ChatBackend;
use crate::engine::cmd::{ChatCmd, handle_chat_cmd};
use crate::engine::hybrid_scheduler::{
    HybridSchedulerBackend, HybridSchedulerCommand, HybridSchedulerState, HybridStepExecutor,
    NoRestoreTicket, ScheduledPrefixAdmission, ScheduledReply, SchedulerOwnerContext,
};
use crate::engine::scheduler::PreemptionMode;
use crate::engine::types::ChatConfig;
use crate::model_thread::ResponseTx;
use crate::models::gemma4::layer_cache::Gemma4LayerCache;
use crate::models::muse_glimmer::kv_cache::PagedWindowSlot;
use crate::stream::Stream;
use crate::transformer::paged_kv_cache_adapter::{PagedKVCacheAdapter, SeqId};

impl HybridSchedulerCommand for MuseGlimmerCmd {
    fn as_chat(&self) -> Option<&ChatCmd> {
        match self {
            Self::Chat(chat) => Some(chat),
            Self::SchedulerStats { .. } => None,
        }
    }

    fn into_chat(self) -> std::result::Result<ChatCmd, Self> {
        match self {
            Self::Chat(chat) => Ok(*chat),
            other => Err(other),
        }
    }

    fn into_scheduler_stats(
        self,
    ) -> std::result::Result<ResponseTx<crate::engine::SchedulerStatsJs>, Self> {
        match self {
            Self::SchedulerStats { reply } => Ok(reply),
            other => Err(other),
        }
    }
}

#[derive(Default)]
pub(crate) struct MuseOwnerState {
    history: Vec<u32>,
    flat_caches: Option<Vec<Gemma4LayerCache>>,
}

pub(crate) type MuseSchedulerState = HybridSchedulerState<MuseGlimmerInner>;

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

fn owner_id(command: &ChatCmd) -> Option<&str> {
    chat_config(command)
        .and_then(|config| config.cache_owner_id.as_deref())
        .filter(|owner| !owner.is_empty())
}

fn is_start(command: &ChatCmd) -> bool {
    matches!(
        command,
        ChatCmd::SessionStart { .. } | ChatCmd::StreamSessionStart { .. }
    )
}

fn send_error(command: ChatCmd, error: Error) {
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

impl HybridSchedulerBackend for MuseGlimmerInner {
    type Command = MuseGlimmerCmd;
    type RestoreTicket = NoRestoreTicket;
    type OwnerState = MuseOwnerState;
    type StepExecutor<'a> = HybridStepExecutor<'a, Self>;

    const SCHEDULER_NAME: &'static str = "Muse-Glimmer";

    fn paged_adapter(&self) -> Option<&PagedKVCacheAdapter> {
        self.paged.as_ref().map(|paged| {
            let _admitted_decode_windows: &[PagedWindowSlot] = &paged.decode_windows;
            paged.coordinator.full_adapter()
        })
    }

    fn paged_adapter_mut(&mut self) -> Option<&mut PagedKVCacheAdapter> {
        self.paged
            .as_mut()
            .map(|paged| paged.coordinator.full_adapter_mut())
    }

    fn scheduler_capacity(&self) -> usize {
        self.paged.as_ref().map_or(1, |paged| {
            paged.coordinator.max_concurrent_sequences() as usize
        })
    }

    fn scheduler_prefill_slice_tokens(&self) -> u32 {
        super::PREFILL_STEP_SIZE as u32
    }

    fn scheduler_has_cold_tier(&self) -> bool {
        // Like Gemma4, ordinary prefix admission may restore the atomic
        // full/sliding record. In-flight SSD preemption remains disabled: a
        // partial turn has not published a matching sliding sidecar yet.
        false
    }

    fn reset_owner_on_session_start(&self) -> bool {
        true
    }

    fn release_scheduled_cache(&mut self, seq_id: SeqId) -> Result<()> {
        self.release_paged_request(seq_id)
            .map(|_| ())
            .map_err(Error::from_reason)
    }

    fn preempt_scheduled_cache(
        &mut self,
        seq_id: SeqId,
        _cache_salt: u64,
        _mode: PreemptionMode,
    ) -> Result<()> {
        self.release_scheduled_cache(seq_id)
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

    fn max_position_embeddings(&self) -> i32 {
        i32::try_from(self.config.text_config.max_position_embeddings).unwrap_or(i32::MAX)
    }

    fn activate_paged_seq(&mut self, seq_id: SeqId) -> Result<()> {
        self.set_active_paged_owner(seq_id);
        self.paged
            .as_mut()
            .ok_or_else(|| Error::from_reason("Muse-Glimmer paged runtime is unavailable"))?
            .coordinator
            .activate_request_all(seq_id)
            .map_err(Error::from_reason)
    }

    fn run_paged_decode_step_batched(&mut self, rows: &[(SeqId, u32)]) -> Result<MxArray> {
        MuseGlimmerInner::run_paged_decode_step_batched(self, rows)
    }

    fn replace_cached_token_history(&mut self, history: Vec<u32>) {
        self.cached_token_history = history;
    }

    fn owner_tokens(state: &Self::OwnerState) -> &[u32] {
        &state.history
    }

    fn install_owner_state(&mut self, seq_id: SeqId, state: &Self::OwnerState) {
        self.set_active_paged_owner(seq_id);
        self.cached_token_history = state.history.clone();
    }

    fn capture_owner_state(&mut self, _seq_id: SeqId) -> Self::OwnerState {
        MuseOwnerState {
            history: self.cached_token_history.clone(),
            flat_caches: None,
        }
    }

    fn build_scheduled_prefix(
        &self,
        _base: &Self::PrefixState,
        effective_cached_prefix_len: usize,
        suffix_len: usize,
        _full_tokens: Vec<u32>,
        _first_chunk: bool,
    ) -> Self::PrefixState {
        MusePrefixState {
            effective_cached_prefix_len,
            suffix_len,
        }
    }

    fn run_scheduled_prefill_slice(
        &mut self,
        seq_id: SeqId,
        source: &[u32],
        _base: &Self::PrefixState,
        start: usize,
        end: usize,
        _generation_stream: Stream,
        _first_chunk: bool,
    ) -> Result<Option<MxArray>> {
        self.set_active_paged_owner(seq_id);
        self.run_paged_prefill_slice(&source[start..end], start as u32)
            .map(Some)
    }

    fn finish_scheduled_decode_batch(&mut self, rows: &[(SeqId, u32)]) -> Result<()> {
        let paged = self
            .paged
            .as_mut()
            .ok_or_else(|| Error::from_reason("Muse-Glimmer paged runtime is unavailable"))?;
        paged
            .coordinator
            .eval_pending_pool_writes_all()
            .map_err(Error::from_reason)?;
        for &(seq_id, _) in rows {
            self.remember_sliding_cold_checkpoints(seq_id)?;
        }
        let paged = self
            .paged
            .as_mut()
            .ok_or_else(|| Error::from_reason("Muse-Glimmer paged runtime is unavailable"))?;
        for &(seq_id, _) in rows {
            paged
                .coordinator
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
        let mut cached =
            self.prepare_paged_text_request(seq_id, tokens, cache_salt, reuse_cache)?;
        if cached >= tokens.len() as u32 {
            self.paged
                .as_mut()
                .ok_or_else(|| Error::from_reason("Muse-Glimmer paged runtime is unavailable"))?
                .coordinator
                .reset_scheduled_request(seq_id)
                .map_err(Error::from_reason)?;
            cached = 0;
        }
        Ok(ScheduledPrefixAdmission::Ready(MusePrefixState {
            effective_cached_prefix_len: cached as usize,
            suffix_len: tokens.len().saturating_sub(cached as usize),
        }))
    }

    fn extra_prefill_breaks(&self, _prompt_tokens: u32, _cached_prefix: u32) -> Vec<u32> {
        self.cold_anchor_rungs()
    }

    fn step_executor(&mut self) -> Self::StepExecutor<'_> {
        HybridStepExecutor::new(self)
    }

    fn execute_barrier(
        &mut self,
        command: Self::Command,
        owners: SchedulerOwnerContext<'_, Self::OwnerState>,
    ) {
        let MuseGlimmerCmd::Chat(command) = command else {
            return;
        };
        if matches!(command.as_ref(), ChatCmd::ResetCaches { .. }) {
            self.select_ownerless_lane(self.paged.is_none());
            handle_chat_cmd(self, *command);
            return;
        }
        let command = *command;
        let flat_lane = chat_config(&command).is_some_and(|config| config.enable_mtp == Some(true))
            || self.paged.is_none();
        let Some(owner) = owner_id(&command).map(str::to_owned) else {
            self.select_ownerless_lane(flat_lane);
            handle_chat_cmd(self, command);
            return;
        };
        if is_start(&command) {
            if let Some(seq_id) = owners.owner_sequences.remove(&owner)
                && let Err(error) = self.release_scheduled_cache(seq_id)
            {
                send_error(command, error);
                return;
            }
            owners.owner_states.remove(&owner);
        } else {
            let state = owners.owner_states.get(&owner);
            let matching = if flat_lane {
                state.is_some_and(|state| state.flat_caches.is_some())
            } else {
                owners.owner_sequences.contains_key(&owner)
                    && state.is_some_and(|state| state.flat_caches.is_none())
            };
            if !matching {
                let conflicting = state.is_some() || owners.owner_sequences.contains_key(&owner);
                let message = if conflicting {
                    "Muse-Glimmer cannot switch a live cache owner between paged AR and flat DFlash layouts; start a new session"
                } else {
                    "chat session continuation requires an initialized cache owner (call chatSessionStart first)"
                };
                send_error(command, Error::from_reason(message));
                return;
            }
        }
        let state = owners.owner_states.remove(&owner).unwrap_or_default();
        if flat_lane {
            self.install_flat_owner_caches(state.flat_caches);
            self.cached_token_history = state.history;
            handle_chat_cmd(self, command);
            if ChatBackend::has_live_session(self) {
                owners.owner_states.insert(
                    owner,
                    MuseOwnerState {
                        history: self.cached_token_history.clone(),
                        flat_caches: Some(self.take_flat_owner_caches()),
                    },
                );
            } else {
                owners.owner_sequences.remove(&owner);
            }
        } else {
            let seq_id = owners
                .owner_sequences
                .get(&owner)
                .copied()
                .unwrap_or_else(|| {
                    let seq_id = *owners.next_seq_id;
                    *owners.next_seq_id = owners.next_seq_id.saturating_add(1);
                    owners.owner_sequences.insert(owner.clone(), seq_id);
                    seq_id
                });
            self.set_active_paged_owner(seq_id);
            self.cached_token_history = state.history;
            handle_chat_cmd(self, command);
            if ChatBackend::has_live_session(self) {
                owners.owner_states.insert(
                    owner,
                    MuseOwnerState {
                        history: self.cached_token_history.clone(),
                        flat_caches: None,
                    },
                );
            } else {
                owners.owner_sequences.remove(&owner);
            }
        }
    }
}
