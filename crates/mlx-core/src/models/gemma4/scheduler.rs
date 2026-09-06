use napi::bindgen_prelude::{Error, Result};

use super::*;
use crate::array::MxArray;
use crate::engine::backend::ChatBackend;
use crate::engine::cmd::{ChatCmd, handle_chat_cmd};
use crate::engine::hybrid_scheduler::{
    HybridSchedulerBackend, HybridSchedulerState, HybridStepExecutor, NoRestoreTicket,
    ScheduledPrefixAdmission, ScheduledReply, SchedulerOwnerContext,
};
use crate::engine::scheduler::PreemptionMode;
use crate::stream::Stream;
use crate::transformer::paged_kv_cache_adapter::{PagedKVCacheAdapter, SeqId};

fn scheduler_prefill_slice_tokens() -> u32 {
    super::gemma4_paged_prefill_group_max_chunk().max(1)
}

/// Commands owned by the Gemma 4 scheduler thread. Optional draft and media
/// capabilities remain ordinary chat commands; request classification decides
/// whether they enter the scheduled or exclusive lane.
pub(crate) type Gemma4Cmd = crate::engine::model_command::ModelCommand;

pub(crate) type Gemma4SchedulerState = HybridSchedulerState<Gemma4Inner>;

fn gemma4_chat_parts(command: &ChatCmd) -> Option<(&[ChatMessage], &ChatConfig)> {
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

fn gemma4_owner_id(command: &ChatCmd) -> Option<&str> {
    gemma4_chat_parts(command)
        .and_then(|(_, config)| config.cache_owner_id.as_deref())
        .filter(|owner| !owner.is_empty())
}

fn gemma4_is_session_start(command: &ChatCmd) -> bool {
    matches!(
        command,
        ChatCmd::SessionStart { .. } | ChatCmd::StreamSessionStart { .. }
    )
}

fn gemma4_send_error(command: ChatCmd, error: Error) {
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

impl HybridSchedulerBackend for Gemma4Inner {
    type FamilyCommand = std::convert::Infallible;
    type RestoreTicket = NoRestoreTicket;
    type OwnerState = Gemma4SchedulerOwnerState;
    type StepExecutor<'a> = HybridStepExecutor<'a, Self>;

    const SCHEDULER_NAME: &'static str = "Gemma4";
    const CANCEL_PRECEDES_EOS: bool = true;
    const STREAM_EOS_TOKEN: bool = true;

    fn paged_adapter(&self) -> Option<&PagedKVCacheAdapter> {
        self.kv_cache_coordinator
            .as_ref()
            .map(Gemma4KVCacheCoordinator::full_adapter)
    }

    fn paged_adapter_mut(&mut self) -> Option<&mut PagedKVCacheAdapter> {
        self.kv_cache_coordinator
            .as_mut()
            .map(Gemma4KVCacheCoordinator::full_adapter_mut)
    }

    fn scheduler_capacity(&self) -> usize {
        self.kv_cache_coordinator.as_ref().map_or(1, |coordinator| {
            coordinator.max_concurrent_sequences() as usize
        })
    }

    fn scheduler_prefill_slice_tokens(&self) -> u32 {
        scheduler_prefill_slice_tokens()
    }

    fn scheduler_has_cold_tier(&self) -> bool {
        // Gemma4's full/sliding sidecar is a joint commit record. Ordinary
        // prefix admission may restore it, but scheduler preemption must drop
        // every group and recompute rather than offloading only the full group.
        false
    }

    fn reset_owner_on_session_start(&self) -> bool {
        true
    }

    fn release_scheduled_cache(&mut self, seq_id: SeqId) -> Result<()> {
        self.release_scheduled_speculation(seq_id);
        if let Some(coordinator) = self.kv_cache_coordinator.as_mut() {
            coordinator
                .release_request_all(seq_id)
                .map(|_| ())
                .map_err(Error::from_reason)?;
        }
        Ok(())
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
        self.config.max_position_embeddings
    }

    fn activate_paged_seq(&mut self, seq_id: SeqId) -> Result<()> {
        self.set_active_paged_owner(seq_id);
        self.kv_cache_coordinator
            .as_mut()
            .ok_or_else(|| Error::from_reason("Gemma4 KV coordinator is unavailable"))?
            .activate_request_all(seq_id)
            .map_err(Error::from_reason)
    }

    fn supports_scheduled_speculation(&self) -> bool {
        self.dspark_draft().is_some()
    }

    fn scheduled_draft_state_bytes(&self, total_tokens: u32) -> u64 {
        let Some(draft) = self.dspark_draft() else {
            return 0;
        };
        // K and V for every draft layer. Charge f32 plus one cache-growth
        // quantum even when loaded weights produce smaller bf16 arrays.
        (draft.config.num_hidden_layers as u64)
            .saturating_mul(draft.config.num_global_key_value_heads.max(0) as u64)
            .saturating_mul(draft.config.global_head_dim.max(0) as u64)
            .saturating_mul(8)
            .saturating_mul(u64::from(total_tokens).saturating_add(256))
    }

    fn begin_scheduled_speculation(&mut self, seq_id: SeqId, position: u32) -> Result<()> {
        self.begin_scheduled_dspark(seq_id, position)
    }

    fn reserve_scheduled_speculation(&mut self, seq_id: SeqId, queries: usize) -> Result<bool> {
        use crate::engine::spec_paged::SpecPagedCache;
        Gemma4SpecPagedCache::new(self)
            .reserve_lookahead(seq_id, queries)
            .map_err(Error::from_reason)
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
        self.scheduled_dspark_states.remove(&seq_id);
    }

    fn run_paged_decode_step_batched(&mut self, rows: &[(SeqId, u32)]) -> Result<MxArray> {
        self.run_paged_decode_step_batched(rows)
    }

    fn replace_cached_token_history(&mut self, history: Vec<u32>) {
        self.cached_token_history = history;
    }

    fn owner_tokens(state: &Self::OwnerState) -> &[u32] {
        &state.metadata.cached_token_history
    }

    fn install_owner_state(&mut self, seq_id: SeqId, state: &Self::OwnerState) {
        self.set_active_paged_owner(seq_id);
        self.install_owner_metadata(state.metadata.clone());
    }

    fn capture_owner_state(&mut self, _seq_id: SeqId) -> Self::OwnerState {
        Gemma4SchedulerOwnerState {
            metadata: self.owner_metadata(),
            flat_caches: None,
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
        Gemma4PrefixState {
            effective_cached_prefix_len,
            suffix_len,
            sliding_primed_prefix_len: base.sliding_primed_prefix_len,
            cache_salt: base.cache_salt,
            full_tokens,
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
        self.run_scheduled_paged_prefill_slice(
            seq_id,
            &source[start..end],
            start as u32,
            end == source.len(),
        )
    }

    fn finish_scheduled_decode_batch(&mut self, rows: &[(SeqId, u32)]) -> Result<()> {
        self.kv_cache_coordinator
            .as_mut()
            .ok_or_else(|| Error::from_reason("Gemma4 batched decode lost its KV coordinator"))?
            .eval_pending_pool_writes_all()
            .map_err(Error::from_reason)?;
        for &(seq_id, _) in rows {
            self.remember_grouped_sliding_cold_checkpoint(seq_id)?;
        }
        let coordinator = self
            .kv_cache_coordinator
            .as_mut()
            .ok_or_else(|| Error::from_reason("Gemma4 batched decode lost its KV coordinator"))?;
        for &(seq_id, _) in rows {
            coordinator
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
        self.set_active_paged_owner(seq_id);
        let mut cached =
            self.prepare_scheduled_text_request(seq_id, tokens, cache_salt, reuse_cache)?;
        if cached >= tokens.len() as u32 {
            self.kv_cache_coordinator
                .as_mut()
                .ok_or_else(|| Error::from_reason("Gemma4 KV coordinator is unavailable"))?
                .reset_scheduled_request(seq_id)
                .map_err(Error::from_reason)?;
            cached = 0;
        }
        Ok(ScheduledPrefixAdmission::Ready(Gemma4PrefixState {
            effective_cached_prefix_len: cached as usize,
            suffix_len: tokens.len().saturating_sub(cached as usize),
            sliding_primed_prefix_len: cached,
            cache_salt,
            full_tokens: tokens.to_vec(),
        }))
    }

    fn extra_prefill_breaks(&self, _prompt_tokens: u32, _cached_prefix: u32) -> Vec<u32> {
        self.scheduled_cold_anchor_rungs()
    }

    fn step_executor(&mut self) -> Self::StepExecutor<'_> {
        HybridStepExecutor::new(self)
    }

    fn execute_chat_barrier(
        &mut self,
        command: ChatCmd,
        owners: SchedulerOwnerContext<'_, Self::OwnerState>,
    ) {
        if matches!(&command, ChatCmd::ResetCaches { .. }) {
            self.scheduled_dspark_states.clear();
            self.set_active_paged_owner(0);
            handle_chat_cmd(self, command);
            return;
        }
        // Only the ASSISTANT drafter needs the flat target-cache lane (its
        // Q-only attention reads `Gemma4LayerCache` K/V arrays directly).
        // DSpark verifies against the paged pools, so a DSpark command stays
        // on the paged lane — installing flat caches there would hide the
        // pools from the planner and silently downgrade the turn to AR.
        let flat_lane = gemma4_chat_parts(&command).is_some_and(|(_, config)| {
            (config.enable_mtp == Some(true) && self.assistant_draft().is_some())
                || self.kv_cache_coordinator.is_none()
        });
        let Some(owner_id) = gemma4_owner_id(&command).map(str::to_owned) else {
            self.select_ownerless_lane(flat_lane);
            handle_chat_cmd(self, command);
            return;
        };
        if gemma4_is_session_start(&command) {
            let seq_id = owners.owner_sequences.get(&owner_id).copied();
            if let Err(error) =
                self.release_owner_resources(seq_id, owners.owner_states.get(&owner_id))
            {
                gemma4_send_error(command, error);
                return;
            }
            owners.owner_sequences.remove(&owner_id);
            owners.owner_states.remove(&owner_id);
        } else {
            let state = owners.owner_states.get(&owner_id);
            let matching_lane = if flat_lane {
                state.is_some_and(|state| state.flat_caches.is_some())
            } else {
                owners.owner_sequences.contains_key(&owner_id)
            };
            if !matching_lane {
                let conflicting = if flat_lane {
                    owners.owner_sequences.contains_key(&owner_id)
                } else {
                    state.is_some_and(|state| state.flat_caches.is_some())
                };
                let message = if conflicting {
                    "Gemma4 cannot switch a live cache owner between autoregressive/media and assistant-draft cache layouts; start a new session"
                } else {
                    "chat session continuation requires an initialized cache owner (call chatSessionStart first)"
                };
                gemma4_send_error(command, Error::from_reason(message));
                return;
            }
        }
        let state = owners.owner_states.remove(&owner_id).unwrap_or_default();
        if flat_lane {
            self.install_flat_owner_caches(state.flat_caches);
            self.install_owner_metadata(state.metadata);
            handle_chat_cmd(self, command);
            let live = ChatBackend::has_live_session(self);
            let state = Gemma4SchedulerOwnerState {
                metadata: self.owner_metadata(),
                flat_caches: self.take_flat_owner_caches(),
            };
            if live {
                owners.owner_states.insert(owner_id, state);
            } else {
                owners.owner_sequences.remove(&owner_id);
            }
        } else {
            let seq_id = owners
                .owner_sequences
                .get(&owner_id)
                .copied()
                .unwrap_or_else(|| {
                    let seq_id = *owners.next_seq_id;
                    *owners.next_seq_id = owners.next_seq_id.saturating_add(1);
                    owners.owner_sequences.insert(owner_id.clone(), seq_id);
                    seq_id
                });
            self.set_active_paged_owner(seq_id);
            self.install_owner_metadata(state.metadata);
            handle_chat_cmd(self, command);
            if ChatBackend::has_live_session(self) {
                owners.owner_states.insert(
                    owner_id,
                    Gemma4SchedulerOwnerState {
                        metadata: self.owner_metadata(),
                        flat_caches: None,
                    },
                );
            } else {
                owners.owner_sequences.remove(&owner_id);
            }
        }
    }
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
        let mut inner = tiny_paged_inner();
        // The flat lane is the ASSISTANT drafter's; without one loaded an
        // `enable_mtp` command has no flat layout to switch INTO.
        inner.draft =
            crate::models::gemma4::dspark_decode::tests::tiny_inner_with_assistant_draft()
                .draft
                .take();
        assert!(
            crate::engine::hybrid_scheduler::scheduler_max_num_seqs_for(
                inner
                    .kv_cache_coordinator
                    .as_ref()
                    .map_or(1, |coordinator| coordinator.max_concurrent_sequences()
                        as usize),
            ) > 1,
            "shared full-attention blocks must not statically partition one max context per slot"
        );
        let inner = inner;
        let mut state =
            Gemma4SchedulerState::new(inner).expect("construct generic Gemma4 scheduler");
        state.owner_sequences.insert("session-a".to_string(), 7);
        state.owner_states.insert(
            "session-a".to_string(),
            Gemma4SchedulerOwnerState::default(),
        );
        let config = ChatConfig {
            cache_owner_id: Some("session-a".to_string()),
            enable_mtp: Some(true),
            ..ChatConfig::default()
        };
        let (reply, result) = tokio::sync::oneshot::channel();
        <Gemma4Inner as HybridSchedulerBackend>::execute_barrier(
            &mut state.inner,
            Gemma4Cmd::Chat(Box::new(ChatCmd::SessionContinue {
                messages: Vec::new(),
                config,
                reply,
                cancelled: Arc::new(AtomicBool::new(false)),
            })),
            SchedulerOwnerContext {
                owner_sequences: &mut state.owner_sequences,
                owner_states: &mut state.owner_states,
                next_seq_id: &mut 8,
            },
        );
        let error = result
            .blocking_recv()
            .expect("mode-switch reply")
            .expect_err("live paged-to-draft continuation must fail closed");
        assert!(error.reason.contains("cannot switch a live cache owner"));
        assert_eq!(state.owner_sequences.get("session-a"), Some(&7));
        assert!(
            state
                .owner_states
                .get("session-a")
                .is_some_and(|owner| owner.flat_caches.is_none())
        );
    }

    /// A DSpark command keeps the PAGED lane. `enable_mtp` alone must not
    /// claim the flat cache layout any more — DSpark verifies against the
    /// paged pools, and installing flat caches for its command would hide
    /// those pools from the planner and silently downgrade the turn to AR.
    ///
    /// Mutation this catches: restoring the old
    /// `config.enable_mtp == Some(true)` predicate, which sends a live paged
    /// owner into the flat-layout conflict below.
    #[test]
    fn a_dspark_command_stays_on_the_live_paged_owner() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        let mut inner = tiny_paged_inner();
        inner.draft = crate::models::gemma4::dspark_decode::tests::tiny_inner_with_draft()
            .draft
            .take();
        let mut state =
            Gemma4SchedulerState::new(inner).expect("construct generic Gemma4 scheduler");
        state.owner_sequences.insert("session-a".to_string(), 7);
        state.owner_states.insert(
            "session-a".to_string(),
            Gemma4SchedulerOwnerState::default(),
        );
        let config = ChatConfig {
            cache_owner_id: Some("session-a".to_string()),
            enable_mtp: Some(true),
            ..ChatConfig::default()
        };
        let (reply, result) = tokio::sync::oneshot::channel();
        <Gemma4Inner as HybridSchedulerBackend>::execute_barrier(
            &mut state.inner,
            Gemma4Cmd::Chat(Box::new(ChatCmd::SessionContinue {
                messages: Vec::new(),
                config,
                reply,
                cancelled: Arc::new(AtomicBool::new(false)),
            })),
            SchedulerOwnerContext {
                owner_sequences: &mut state.owner_sequences,
                owner_states: &mut state.owner_states,
                next_seq_id: &mut 8,
            },
        );
        // The empty-message continuation still fails downstream; what matters
        // is that it was ADMITTED to the paged lane instead of refused as a
        // cache-layout switch.
        if let Err(error) = result.blocking_recv().expect("lane reply") {
            assert!(
                !error.reason.contains("cannot switch a live cache owner"),
                "a DSpark command must not be treated as a flat-layout owner: {}",
                error.reason
            );
        }
        assert!(
            state
                .owner_states
                .get("session-a")
                .is_none_or(|owner| owner.flat_caches.is_none()),
            "no flat cache layout may be installed for a DSpark command"
        );
    }

    #[test]
    fn releasing_an_owner_drops_every_generic_owner_registry() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        let mut state = Gemma4SchedulerState::new(tiny_paged_inner())
            .expect("construct generic Gemma4 scheduler");
        state.owner_sequences.insert("session-a".to_string(), 7);
        state.inner.init_caches_sync().expect("flat caches");
        state.owner_states.insert(
            "session-a".to_string(),
            Gemma4SchedulerOwnerState {
                metadata: Gemma4OwnerMetadata {
                    cached_token_history: vec![1, 2, 3],
                    ..Gemma4OwnerMetadata::default()
                },
                flat_caches: state.inner.take_flat_owner_caches(),
            },
        );

        state
            .release_cache_owner_now("session-a")
            .expect("release owner");
        assert!(!state.owner_sequences.contains_key("session-a"));
        assert!(!state.owner_states.contains_key("session-a"));
    }
}
