//! The model-thread command enum for Qwen3.5 MoE, its dispatch handler, and
//! the training / hybrid-scheduler backends that forward into the seams.

use super::*;

/// Commands dispatched from NAPI methods to the dedicated model thread.
pub(crate) enum Qwen35MoeCmd {
    /// All chat-session traffic (sync + streaming starts/continues/tool
    /// turns + cache reset), routed through the model-neutral engine
    /// dispatcher ([`crate::engine::cmd::handle_chat_cmd`]) against the
    /// [`ChatBackend`] impl on [`Qwen35MoeInner`]. The per-variant
    /// behavioural contracts live on [`crate::engine::cmd::ChatCmd`].
    Chat(ChatCmd),
    Generate {
        prompt_tokens: MxArray,
        config: Qwen3_5MoeGenerationConfig,
        reply: ResponseTx<Qwen3_5MoeGenerationResult>,
    },
    /// Static FP8 activation-amax calibration prefill — see
    /// [`Qwen35MoeInner::calibrate_prefill_raw_sync`] for the contract.
    CalibratePrefillRaw {
        texts: Vec<String>,
        calib_seq: u32,
        reply: ResponseTx<u32>,
    },
    /// Teacher-forced output-quality eval (`mlx eval`). MoE counterpart of the
    /// dense [`crate::models::qwen3_5::model::Qwen35Cmd::EvalTeacherForced`],
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
    /// Training-session commands shared with the model-neutral engine. The
    /// thread loop routes these to
    /// [`crate::engine::cmd::handle_train_cmd`], which drives the
    /// [`TrainBackend`] impl on [`Qwen35MoeInner`].
    Train(TrainCmd),
    SchedulerStats {
        reply: ResponseTx<engine::SchedulerStatsJs>,
    },
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

impl FromTrainCmd for Qwen35MoeCmd {
    #[inline]
    fn from_train(cmd: TrainCmd) -> Self {
        Qwen35MoeCmd::Train(cmd)
    }
}

crate::engine::command_adapter::impl_scheduler_command!(Qwen35MoeCmd, direct);

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
pub(crate) fn handle_qwen35_moe_cmd(inner: &mut Qwen35MoeInner, cmd: Qwen35MoeCmd) {
    match cmd {
        // No per-request cache drain here — the TS idle sweeper in
        // `@mlx-node/server` owns between-turn drains.
        Qwen35MoeCmd::Chat(chat_cmd) => {
            handle_chat_cmd(inner, chat_cmd);
        }
        Qwen35MoeCmd::Generate {
            prompt_tokens,
            config,
            reply,
        } => {
            let _ = reply.send(inner.generate_sync(prompt_tokens, config));
        }
        Qwen35MoeCmd::CalibratePrefillRaw {
            texts,
            calib_seq,
            reply,
        } => {
            let _ = reply.send(inner.calibrate_prefill_raw_sync(texts, calib_seq));
        }
        Qwen35MoeCmd::EvalTeacherForced { request, reply } => {
            let _ = reply.send(crate::quality::runner::run(inner, request));
        }
        Qwen35MoeCmd::SaveModel { save_path, reply } => {
            let _ = reply.send(inner.save_model_sync(&save_path));
        }
        Qwen35MoeCmd::Train(train_cmd) => {
            handle_train_cmd(inner, train_cmd);
        }
        Qwen35MoeCmd::SchedulerStats { reply } => {
            let _ = reply.send(Ok(engine::scheduler::SchedulerStats::default().to_js()));
        }
        Qwen35MoeCmd::MtpPagedGdnStateForTest { reply } => {
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
        Qwen35MoeCmd::GdnHistoryCheckpointOracleForTest { reply } => {
            let _ = reply.send(inner.moe_gdn_history_checkpoint_recompute_matches_for_test());
        }
    }
}

impl HybridSchedulerBackend for Qwen35MoeInner {
    type Command = Qwen35MoeCmd;
    type RestoreTicket = crate::engine::hybrid_scheduler::NoRestoreTicket;
    type OwnerState = Vec<u32>;
    type StepExecutor<'a> = crate::engine::hybrid_scheduler::HybridStepExecutor<'a, Self>;

    const SCHEDULER_NAME: &'static str = "Qwen3.5 MoE";
    const ENABLED_BY_DEFAULT: bool = false;

    fn paged_adapter(&self) -> Option<&PagedKVCacheAdapter> {
        self.paged_adapter.as_ref()
    }

    fn paged_adapter_mut(&mut self) -> Option<&mut PagedKVCacheAdapter> {
        self.paged_adapter.as_mut()
    }

    fn max_position_embeddings(&self) -> i32 {
        self.config.max_position_embeddings
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

    fn execute_barrier(
        &mut self,
        command: Self::Command,
        _owners: crate::engine::hybrid_scheduler::SchedulerOwnerContext<'_, Self::OwnerState>,
    ) {
        handle_qwen35_moe_cmd(self, command);
    }
}
