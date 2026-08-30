//! `Qwen35Cmd` and the command-routing trait impls (train, chat, hybrid scheduler).

use super::*;

/// Commands dispatched from NAPI methods to the dedicated model thread.
pub(crate) enum Qwen35Cmd {
    /// All chat-session traffic (sync + streaming starts/continues/tool
    /// turns + cache reset), routed through the model-neutral engine
    /// dispatcher ([`crate::engine::cmd::handle_chat_cmd`]) against the
    /// [`ChatBackend`] impl on [`Qwen35Inner`]. The per-variant
    /// behavioural contracts live on [`crate::engine::cmd::ChatCmd`].
    Chat(ChatCmd),
    Generate {
        prompt_tokens: MxArray,
        config: Qwen3_5GenerationConfig,
        reply: ResponseTx<Qwen3_5GenerationResult>,
    },
    /// Static FP8 activation-amax calibration prefill (NVIDIA modelopt
    /// `MaxCalibrator`). For each raw text: tokenize WITHOUT the chat template,
    /// truncate to `calib_seq` tokens, then run PREFILL ONLY (no generation, no
    /// generated token) so every mxfp8 attn/GDN projection's activation tap
    /// fires once, resetting caches between rows. Runs on the model thread where
    /// the tokenizer lives. The command body SELF-ARMS this model thread's
    /// thread-local `ActivationAmaxCollector` flag (via `CalibrationArmGuard`)
    /// for the prefill's duration; the NAPI caller drains+persists the collected
    /// amax afterwards — this command never touches `config.json`. Replies with
    /// the number of rows actually prefilled (rows that were empty after
    /// tokenize+truncate are skipped).
    CalibratePrefillRaw {
        texts: Vec<String>,
        calib_seq: u32,
        reply: ResponseTx<u32>,
    },
    /// Teacher-forced output-quality eval (`mlx eval`). Runs the reference AR
    /// prefill over each sequence and folds the head over positions in chunks,
    /// either CAPTURING the bf16 teacher's top-`K` next-token distribution into
    /// a cache or SCORING this checkpoint against one. Runs on the model thread
    /// where the tokenizer lives; caches are re-initialized per row so each
    /// sequence is an independent turn-0 prefill.
    EvalTeacherForced {
        request: crate::quality::EvalRequest,
        reply: ResponseTx<crate::quality::EvalOutcome>,
    },
    SaveModel {
        save_path: String,
        reply: ResponseTx<()>,
    },
    /// Test-only: snapshot the flat-MTP cache state between turns —
    /// `(cached_token_history.len(), flat_mtp_caches_desynced,
    /// flat_full_reprefill_count, flat_mtp_last_rollback_unemitted)`. The length
    /// is the committed prompt+generation history (how many tokens a turn
    /// actually committed, independent of the warm/heal path a later turn
    /// takes); the flag is whether a mid-cycle stop stranded tokens and armed
    /// the heal; the count is the monotonic number of full-history re-prefill
    /// heals taken so far; and the final value is the independently
    /// engine-computed tail passed to the family rollback hook.
    #[doc(hidden)]
    MtpFlatStateForTest {
        reply: ResponseTx<(usize, bool, u64, usize)>,
    },
    /// Test-only: arm the flat-MTP desync heal (`flat_mtp_caches_desynced =
    /// true`) so the NEXT delta turn takes the discard+re-prefill path
    /// deterministically. The heal re-prefills from `cached_token_history` and
    /// ignores the (discarded) cache contents, so arming the flag on a clean
    /// session faithfully exercises the heal without a host-timing-dependent
    /// mid-cycle cancel.
    #[doc(hidden)]
    ForceFlatMtpDesyncForTest { reply: ResponseTx<()> },
    /// Test-only: snapshot the paged-MTP GDN bookkeeping between turns — see
    /// [`MtpPagedGdnStateForTest`].
    #[doc(hidden)]
    MtpPagedGdnStateForTest {
        reply: ResponseTx<MtpPagedGdnStateForTest>,
    },
    /// Test-only: arm the one-shot forced frontier mismatch so the NEXT paged
    /// epilogue takes the refuse-to-persist branch deterministically (the
    /// natural trigger is a swallowed adapter-truncate failure, which no test
    /// can provoke on demand). Pattern of `ForceFlatMtpDesyncForTest`.
    #[doc(hidden)]
    ForcePagedGdnMismatchForTest { reply: ResponseTx<()> },
    /// Test-only state oracle: recompute GDN over the persisted history
    /// checkpoint's own token key from FRESH caches and bit-compare the
    /// conv/recurrent arrays against the checkpoint. `Ok(true)` iff every
    /// linear layer matches exactly — i.e. the persisted state equals what
    /// its key claims it is.
    #[doc(hidden)]
    GdnHistoryCheckpointOracleForTest { reply: ResponseTx<bool> },
    /// Training-session commands shared with the model-neutral engine. The
    /// thread loop routes these to
    /// [`crate::engine::cmd::handle_train_cmd`], which drives the
    /// [`TrainBackend`] impl on [`Qwen35Inner`].
    Train(TrainCmd),
    SchedulerStats {
        reply: ResponseTx<engine::SchedulerStatsJs>,
    },
}

impl FromChatCmd for Qwen35Cmd {
    #[inline]
    fn from_chat(cmd: ChatCmd) -> Self {
        Qwen35Cmd::Chat(cmd)
    }
}

impl FromTrainCmd for Qwen35Cmd {
    #[inline]
    fn from_train(cmd: TrainCmd) -> Self {
        Qwen35Cmd::Train(cmd)
    }
}

impl HybridSchedulerCommand for Qwen35Cmd {
    fn as_chat(&self) -> Option<&ChatCmd> {
        match self {
            Self::Chat(chat) => Some(chat),
            _ => None,
        }
    }

    fn into_chat(self) -> std::result::Result<ChatCmd, Self> {
        match self {
            Self::Chat(chat) => Ok(chat),
            other => Err(other),
        }
    }

    fn into_scheduler_stats(
        self,
    ) -> std::result::Result<ResponseTx<engine::SchedulerStatsJs>, Self> {
        match self {
            Self::SchedulerStats { reply } => Ok(reply),
            other => Err(other),
        }
    }
}

/// Training backend the model-neutral [`handle_train_cmd`] drives. Each
/// method forwards to the inherent `*_sync_impl` body on [`Qwen35Inner`].
impl TrainBackend for Qwen35Inner {
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
pub(crate) fn handle_qwen35_cmd(inner: &mut Qwen35Inner, cmd: Qwen35Cmd) {
    match cmd {
        // All chat-session traffic routes through the model-neutral
        // engine dispatcher against `Qwen35Inner`'s `ChatBackend` impl.
        // (The engine dispatcher carries the historical NOTE forward: no
        // per-request cache drain here — the TS idle sweeper in
        // `@mlx-node/server` handles between-turn drains.)
        Qwen35Cmd::Chat(chat_cmd) => {
            handle_chat_cmd(inner, chat_cmd);
        }
        Qwen35Cmd::Generate {
            prompt_tokens,
            config,
            reply,
        } => {
            let _ = reply.send(inner.generate_sync(prompt_tokens, config));
        }
        Qwen35Cmd::CalibratePrefillRaw {
            texts,
            calib_seq,
            reply,
        } => {
            let _ = reply.send(inner.calibrate_prefill_raw_sync(texts, calib_seq));
        }
        Qwen35Cmd::EvalTeacherForced { request, reply } => {
            let _ = reply.send(crate::quality::runner::run(inner, request));
        }
        Qwen35Cmd::SaveModel { save_path, reply } => {
            let _ = reply.send(inner.save_model_sync(&save_path));
        }
        Qwen35Cmd::MtpFlatStateForTest { reply } => {
            let _ = reply.send(Ok((
                inner.cached_token_history.len(),
                inner.flat_mtp_caches_desynced,
                inner.flat_full_reprefill_count,
                inner.flat_mtp_last_rollback_unemitted,
            )));
        }
        Qwen35Cmd::ForceFlatMtpDesyncForTest { reply } => {
            inner.flat_mtp_caches_desynced = true;
            let _ = reply.send(Ok(()));
        }
        Qwen35Cmd::MtpPagedGdnStateForTest { reply } => {
            let _ = reply.send(Ok(MtpPagedGdnStateForTest {
                paged_active: inner.paged_adapter.is_some(),
                history_len: inner.cached_token_history.len(),
                last_rollback_unemitted: inner.paged_mtp_last_rollback_unemitted,
                gdn_rewinds: inner.paged_mtp_gdn_rewinds,
                gdn_invalidations: inner.paged_mtp_gdn_invalidations,
                state_dirty: inner.paged_gdn_state_dirty,
                has_history_checkpoint: inner.gdn_last_history_checkpoint.is_some(),
                last_prefix_prepare_state: inner.last_gdn_prefix_prepare_state,
            }));
        }
        Qwen35Cmd::ForcePagedGdnMismatchForTest { reply } => {
            inner.paged_gdn_force_mismatch_for_test = true;
            let _ = reply.send(Ok(()));
        }
        Qwen35Cmd::GdnHistoryCheckpointOracleForTest { reply } => {
            let _ = reply.send(inner.gdn_history_checkpoint_recompute_matches_for_test());
        }
        // --- Training commands ---
        Qwen35Cmd::Train(train_cmd) => {
            handle_train_cmd(inner, train_cmd);
        }
        Qwen35Cmd::SchedulerStats { reply } => {
            let _ = reply.send(Ok(engine::scheduler::SchedulerStats::default().to_js()));
        }
    }
}

impl HybridSchedulerBackend for Qwen35Inner {
    type Command = Qwen35Cmd;
    type RestoreTicket = crate::engine::hybrid_scheduler::NoRestoreTicket;
    type OwnerState = Vec<u32>;
    type StepExecutor<'a> = crate::engine::hybrid_scheduler::HybridStepExecutor<'a, Self>;

    const SCHEDULER_NAME: &'static str = "Qwen3.5 dense";
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
        Qwen35PrefixState {
            effective_cached_prefix_len,
            suffix_len,
            full_tokens,
            cache_salt: base.cache_salt,
            gdn_prefix_already_primed: !first_chunk || base.gdn_prefix_already_primed,
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
        handle_qwen35_cmd(self, command);
    }
}
