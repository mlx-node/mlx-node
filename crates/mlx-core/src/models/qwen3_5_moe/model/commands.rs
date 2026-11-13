//! The model-thread command enum for Qwen3.5 MoE, its dispatch handler, and
//! the training / hybrid-scheduler backends that forward into the seams.

use super::*;

/// Commands dispatched from NAPI methods to the dedicated model thread.
pub(crate) type Qwen35MoeCmd = crate::engine::model_command::ModelCommand<Qwen35MoeFamilyCommand>;

pub(crate) enum Qwen35MoeFamilyCommand {
    /// Static FP8 activation-amax calibration prefill — see
    /// [`Qwen35MoeInner::calibrate_prefill_raw_sync`] for the contract.
    CalibratePrefillRaw {
        texts: Vec<String>,
        calib_seq: u32,
        reply: ResponseTx<u32>,
    },
    /// Teacher-forced output-quality eval (`mlx eval`). MoE counterpart of the
    /// dense [`crate::models::qwen3_5::model::crate::models::qwen3_5::model::Qwen35FamilyCommand::EvalTeacherForced`],
    /// carrying the same request onto the model thread where the tokenizer
    /// lives; caches are re-initialized per row so each sequence is an
    /// independent turn-0 prefill.
    EvalTeacherForced {
        request: crate::quality::EvalRequest,
        reply: ResponseTx<crate::quality::EvalOutcome>,
    },
    SaveModel {
        save_path: String,
        reply: ResponseTx<()>,
    },
    /// Closes the SSD cache at a clean stop by `deadline_ms` (milliseconds
    /// on `Date.now()`'s clock); replies whether every restore point and
    /// queued write reached the disk.
    CloseCache {
        deadline_ms: f64,
        reply: ResponseTx<bool>,
    },
    /// Training-session commands shared with the model-neutral engine. The
    /// thread loop routes these to
    /// [`crate::engine::cmd::handle_train_cmd`], which drives the
    /// [`TrainBackend`] impl on [`Qwen35MoeInner`].
    Train(TrainCmd),
    /// Test-only: snapshot the paged-MTP GDN bookkeeping between turns — see
    /// [`MoeMtpPagedGdnStateForTest`].
    #[doc(hidden)]
    MtpPagedGdnStateForTest {
        reply: ResponseTx<MoeMtpPagedGdnStateForTest>,
    },
    /// Test-only state oracle: recompute GDN over the persisted history
    /// checkpoint's own token key from FRESH caches and bit-compare the
    /// conv/recurrent arrays against the checkpoint. `Ok(true)` iff every
    /// linear layer matches exactly — i.e. the persisted state equals what its
    /// key claims it is.
    #[doc(hidden)]
    GdnHistoryCheckpointOracleForTest { reply: ResponseTx<bool> },
}

impl FromTrainCmd for Qwen35MoeFamilyCommand {
    #[inline]
    fn from_train(cmd: TrainCmd) -> Self {
        Qwen35MoeFamilyCommand::Train(cmd)
    }
}

/// Training backend the model-neutral [`handle_train_cmd`] drives. Each
/// method forwards to the inherent `*_sync_impl` body on
/// [`Qwen35MoeInner`].
impl TrainBackend for Qwen35MoeInner {
    fn training_state_mut(
        &mut self,
    ) -> &mut Option<crate::training_state::ModelThreadTrainingState> {
        &mut self.training_state
    }

    fn init_training_sync(
        &mut self,
        config: Box<crate::grpo::engine::GRPOEngineConfig>,
        model_type: crate::training_model::ModelType,
    ) -> Result<()> {
        self.init_training_sync_impl(*config, model_type)
    }

    fn generate_for_training_thread_sync(
        &mut self,
        prompts: Vec<Vec<ChatMessage>>,
        group_size: usize,
        gen_config: crate::models::qwen3::GenerationConfig,
        enable_thinking: Option<bool>,
        tools: Option<Vec<ToolDefinition>>,
    ) -> Result<crate::training_model::GenerationPlainData> {
        self.generate_for_training_thread_sync_impl(
            prompts,
            group_size,
            gen_config,
            enable_thinking,
            tools,
        )
    }

    fn train_step_grpo_sync(
        &mut self,
        rewards: Vec<f64>,
        group_size: i32,
        loss_config: crate::grpo::loss::GRPOLossConfig,
        valid_indices: Option<Vec<usize>>,
    ) -> Result<crate::training_model::TrainStepPlainMetrics> {
        self.train_step_grpo_sync_impl(rewards, group_size, loss_config, valid_indices)
    }

    fn train_step_sft_sync(
        &mut self,
        input_ids: Vec<i32>,
        input_shape: Vec<i64>,
        labels: Vec<i32>,
        labels_shape: Vec<i64>,
        config: crate::sft::engine::SftEngineConfig,
    ) -> Result<crate::training_model::TrainStepPlainMetrics> {
        self.train_step_sft_sync_impl(input_ids, input_shape, labels, labels_shape, config)
    }

    fn save_optimizer_state_sync(&self, path: String) -> Result<()> {
        self.save_optimizer_state_sync_impl(path)
    }

    fn load_optimizer_state_sync(&mut self, path: String) -> Result<()> {
        self.load_optimizer_state_sync_impl(path)
    }
}

/// Command handler for the dedicated model thread.
impl crate::engine::model_command::FamilyCommand<Qwen35MoeInner> for Qwen35MoeFamilyCommand {
    fn execute(self, inner: &mut Qwen35MoeInner) {
        match self {
            Qwen35MoeFamilyCommand::CalibratePrefillRaw {
                texts,
                calib_seq,
                reply,
            } => {
                let _ = reply.send(inner.calibrate_prefill_raw_sync(texts, calib_seq));
            }
            Qwen35MoeFamilyCommand::EvalTeacherForced { request, reply } => {
                let _ = reply.send(crate::quality::runner::run(inner, request));
            }
            Qwen35MoeFamilyCommand::SaveModel { save_path, reply } => {
                let _ = reply.send(inner.save_model_sync(&save_path));
            }
            Qwen35MoeFamilyCommand::CloseCache { deadline_ms, reply } => {
                use crate::models::qwen3_5::dflash2_state::disk::{close_cache, instant_at};
                let _ = reply.send(Ok(close_cache(inner, instant_at(deadline_ms))));
            }
            Qwen35MoeFamilyCommand::Train(train_cmd) => {
                handle_train_cmd(inner, train_cmd);
            }
            Qwen35MoeFamilyCommand::MtpPagedGdnStateForTest { reply } => {
                let _ = reply.send(Ok(MoeMtpPagedGdnStateForTest {
                    paged_active: inner.paged_adapter.is_some(),
                    history_len: inner.cached_token_history.len(),
                    last_rollback_unemitted: inner.paged_mtp_last_rollback_unemitted,
                    gdn_rewinds: inner.paged_mtp_gdn_rewinds,
                    gdn_invalidations: inner.paged_mtp_gdn_invalidations,
                    state_dirty: inner.paged_gdn_state_dirty,
                    has_history_checkpoint: inner.gdn_last_history_checkpoint.is_some(),
                }));
            }
            Qwen35MoeFamilyCommand::GdnHistoryCheckpointOracleForTest { reply } => {
                let _ = reply.send(inner.moe_gdn_history_checkpoint_recompute_matches_for_test());
            }
        }
    }
}

impl HybridSchedulerBackend for Qwen35MoeInner {
    type FamilyCommand = Qwen35MoeFamilyCommand;
    type RestoreTicket = crate::engine::hybrid_scheduler::NoRestoreTicket;
    type OwnerState = Vec<u32>;
    type StepExecutor<'a> = crate::engine::hybrid_scheduler::HybridStepExecutor<'a, Self>;

    const SCHEDULER_NAME: &'static str = "Qwen3.5 MoE";
    const ENABLED_BY_DEFAULT: bool = false;

    fn wired_residency(&self) -> Option<std::sync::Arc<crate::stream::WiredResidency>> {
        self.wired.clone()
    }

    fn paged_adapter(&self) -> Option<&PagedKVCacheAdapter> {
        self.paged_adapter.as_ref()
    }

    fn paged_adapter_mut(&mut self) -> Option<&mut PagedKVCacheAdapter> {
        self.paged_adapter.as_mut()
    }

    /// The turn's replay point leaves InUse when it closes.
    fn scheduled_turn_closed(&mut self, seq_id: SeqId) {
        self.end_gdn_use(seq_id);
        self.dflash2_states.end_turn(seq_id);
    }

    fn reclaims_in_order(&self) -> bool {
        true
    }

    fn measured_prefill_cut(&self) -> Option<crate::engine::scheduler::MeasuredConfig> {
        Some(crate::engine::scheduler::MeasuredConfig {
            native_window: self.config.max_position_embeddings.max(0) as u32,
            decode_share: self.decode_share,
        })
    }

    fn dflash2_state_status(
        &self,
    ) -> Option<crate::engine::governor::governor::DFlash2StateStatus> {
        Some(crate::models::qwen3_5::dflash2_turn::state_status(self))
    }

    fn memory_accounts(&self) -> Option<crate::engine::governor::governor::MemoryAccounts> {
        Some(crate::models::qwen3_5::dflash2_turn::memory_accounts(
            self,
            crate::engine::hybrid_scheduler::HybridSchedulerBackend::scheduled_recurrent_bytes(
                self,
            ),
        ))
    }

    fn reset_after_gpu_oom(&mut self) -> Result<()> {
        crate::engine::backend::ChatBackend::reset_caches(
            self,
            crate::engine::backend::ResetScope::Command,
        )?;
        crate::models::qwen3_5::dflash2_turn::drop_unheld_vision(self);
        Ok(())
    }

    fn memory_model(&self) -> u64 {
        self.memory_registration
            .as_ref()
            .map_or(0, crate::engine::governor::governor::ModelGuard::model_id)
    }

    fn sync_state_marks(&mut self) {
        self.sync_gdn_state_marks();
    }

    fn tick_state_disk(&mut self, tick: crate::models::qwen3_5::dflash2_state::disk::Tick) -> bool {
        crate::models::qwen3_5::dflash2_state::disk::tick(self, tick)
    }

    fn memory_governor(&self) -> Option<&'static crate::engine::governor::governor::Governor> {
        self.memory_registration
            .as_ref()
            .map(crate::engine::governor::governor::ModelGuard::governor)
    }

    fn victim_units(
        &self,
        query: &crate::engine::governor::ledger::VictimQuery,
        kept: &[u32],
    ) -> crate::engine::governor::ledger::UnitRuns {
        crate::models::qwen3_5::dflash2_turn::model_units(
            self,
            crate::models::qwen3_5::dflash2_turn::Keep {
                tokens: kept,
                keys: None,
            },
            query,
        )
    }

    fn take_victim(
        &mut self,
        query: &crate::engine::governor::ledger::VictimQuery,
        victim: &crate::engine::governor::ledger::Victim,
        kept: &[u32],
    ) -> crate::engine::governor::ledger::Taken {
        crate::models::qwen3_5::dflash2_turn::take_model_unit(
            self,
            crate::models::qwen3_5::dflash2_turn::Keep {
                tokens: kept,
                keys: None,
            },
            query,
            victim,
        )
    }

    fn claim_step_rows(
        state: &mut crate::engine::hybrid_scheduler::HybridSchedulerState<Self>,
        plan: &crate::engine::scheduler::StepPlan,
    ) -> crate::engine::hybrid_scheduler::StepClaims {
        state.claim_rows_in_order(plan)
    }

    fn scheduler_books_on_demand(&self) -> bool {
        true
    }

    fn scheduler_output_budget_is_rest_of_window(&self) -> bool {
        true
    }

    fn supports_scheduled_speculation(&self) -> bool {
        self.has_mtp_weights()
            && self.dflash2.is_none()
            && self.paged_adapter.is_some()
            && self.config.recurrent_state_bytes() > 0
            && self.cached_rope_deltas.unwrap_or(0) == 0
    }
    fn scheduled_draft_state_bytes(&self, total_tokens: u32) -> u64 {
        crate::models::qwen3_5::scheduled_mtp::ScheduledMtpState::reservation_bytes(
            &self.config.to_dense_config(),
            total_tokens,
        )
    }
    fn planned_turn_window(&self, _speculative: bool) -> Option<u32> {
        let Some(pool) = self
            .memory_plan
            .as_ref()
            .and_then(|plan| plan.pool.as_ref())
        else {
            return Some(self.pool_context_limit());
        };
        Some(pool.lane.window.maximum())
    }
    fn scheduled_draft_state_held_bytes(&self, total_tokens: u32) -> u64 {
        crate::models::qwen3_5::scheduled_mtp::ScheduledMtpState::held_bytes(
            &self.config.to_dense_config(),
            total_tokens,
        )
    }
    fn begin_scheduled_speculation(&mut self, seq_id: SeqId, position: u32) -> Result<bool> {
        crate::models::qwen3_5::scheduled_mtp::ScheduledMtpTarget::begin_scheduled_mtp(
            self, seq_id, position,
        )
    }
    fn reserve_scheduled_speculation(&mut self, seq_id: SeqId, queries: usize) -> Result<bool> {
        if queries > crate::models::qwen3_5::scheduled_mtp::MAX_DRAFTS + 1 {
            return Err(Error::from_reason(
                "MTP reservation exceeds its maximum verifier width",
            ));
        }
        let adapter = self
            .paged_adapter
            .as_mut()
            .ok_or_else(|| Error::from_reason("MTP reservation has no paged adapter"))?;
        match adapter.reserve_rows_for(seq_id, queries as u32) {
            Ok(_) => Ok(true),
            Err(error) if error.starts_with("context_length_exceeded:") => Ok(false),
            Err(error) => Err(Error::from_reason(error)),
        }
    }
    fn propose_scheduled(
        &mut self,
        seq_id: SeqId,
        anchor: u32,
        cap: usize,
        params: &crate::engine::params::ChatParams,
        rng: &mut dyn rand::Rng,
        _confidence: bool,
    ) -> Result<crate::engine::backend::DsparkProposal> {
        crate::models::qwen3_5::scheduled_mtp::ScheduledMtpTarget::propose_scheduled_mtp(
            self, seq_id, anchor, cap, params, rng,
        )
    }
    fn run_scheduled_verify(
        &mut self,
        rows: &[crate::engine::hybrid_scheduler::ScheduledVerifyRow],
    ) -> Result<MxArray> {
        crate::models::qwen3_5::scheduled_mtp::ScheduledMtpTarget::verify_scheduled_mtp(self, rows)
    }
    fn commit_scheduled_verify(
        &mut self,
        rows: &[crate::engine::hybrid_scheduler::ScheduledVerifyCommit],
    ) -> Result<Vec<Result<()>>> {
        crate::models::qwen3_5::scheduled_mtp::ScheduledMtpTarget::commit_scheduled_mtp(self, rows)
    }
    fn release_scheduled_speculation(&mut self, seq_id: SeqId) {
        self.scheduled_mtp.release(seq_id);
    }

    fn max_position_embeddings(&self) -> i32 {
        self.context_cap() as i32
    }

    fn recurrent_state_bytes(&self) -> u64 {
        self.config.recurrent_state_bytes()
    }

    fn scheduled_recurrent_bytes(&self) -> u64 {
        self.scheduled_recurrent_bytes()
    }

    fn has_scheduled_recurrent(&self, seq_id: SeqId) -> bool {
        self.has_scheduled_recurrent(seq_id)
    }

    fn can_activate_scheduled_recurrent(&self, seq_id: SeqId) -> bool {
        self.can_activate_scheduled_recurrent(seq_id)
    }

    fn activate_scheduled_recurrent(&mut self, seq_id: SeqId) -> Result<()> {
        self.activate_scheduled_recurrent(seq_id)
    }

    fn activate_paged_seq(&mut self, seq_id: SeqId) -> Result<()> {
        self.activate_paged_seq(seq_id)
    }

    fn park_active_scheduled_recurrent(&mut self) -> Result<()> {
        self.park_active_scheduled_recurrent()
    }

    fn release_scheduled_recurrent_for(&mut self, seq_id: SeqId) {
        self.release_scheduled_recurrent_for(seq_id);
    }

    fn run_paged_decode_step_batched(&mut self, rows: &[(SeqId, u32)]) -> Result<MxArray> {
        self.run_paged_decode_step_batched(rows)
    }

    fn replace_cached_token_history(&mut self, history: Vec<u32>) {
        self.cached_token_history = history;
    }

    /// The scheduler installs the owner's history before it activates the
    /// sequence on some paths: the live draft window no longer matches it.
    fn install_owner_state(&mut self, _seq_id: SeqId, state: &Self::OwnerState) {
        if self.active_scheduled_seq.is_none() {
            self.invalidate_dflash2_context("install_owner_state");
        }
        self.cached_token_history = state.clone();
    }

    fn owner_tokens(state: &Self::OwnerState) -> &[u32] {
        state
    }

    fn capture_owner_state(&mut self, _seq_id: SeqId) -> Self::OwnerState {
        self.cached_token_history.clone()
    }

    fn build_scheduled_prefix(
        &self,
        base: &Self::PrefixState,
        effective_cached_prefix_len: usize,
        suffix_len: usize,
        full_tokens: Vec<u32>,
        first_chunk: bool,
    ) -> Self::PrefixState {
        Qwen35MoePrefixState {
            effective_cached_prefix_len,
            suffix_len,
            full_tokens,
            gdn_prefix_already_primed: !first_chunk || base.gdn_prefix_already_primed,
            checkpoint_extra_keys: base.checkpoint_extra_keys.clone(),
            checkpoint_cache_salt: base.checkpoint_cache_salt,
        }
    }

    fn step_executor(&mut self) -> Self::StepExecutor<'_> {
        crate::engine::hybrid_scheduler::HybridStepExecutor::new(self)
    }
}
