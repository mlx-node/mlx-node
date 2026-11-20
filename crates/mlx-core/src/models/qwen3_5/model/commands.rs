//! `Qwen35Cmd` and the command-routing trait impls (train, chat, hybrid scheduler).

use super::*;

/// Commands dispatched from NAPI methods to the dedicated model thread.
pub(crate) type Qwen35Cmd = crate::engine::model_command::ModelCommand<Qwen35FamilyCommand>;

pub(crate) enum Qwen35FamilyCommand {
    DecideClef {
        request: crate::models::clef::encoding::Request,
        cancelled: Arc<AtomicBool>,
        reply: ResponseTx<String>,
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
    /// Closes the SSD cache at a clean stop by `deadline_ms` (milliseconds
    /// on `Date.now()`'s clock); replies whether every restore point and
    /// queued write reached the disk.
    CloseCache {
        deadline_ms: f64,
        reply: ResponseTx<bool>,
    },
    /// Test-only: snapshot the paged-MTP GDN bookkeeping between turns — see
    /// [`MtpPagedGdnStateForTest`].
    #[doc(hidden)]
    MtpPagedGdnStateForTest {
        reply: ResponseTx<MtpPagedGdnStateForTest>,
    },
    /// Test-only: arm the one-shot forced frontier mismatch so the NEXT paged
    /// epilogue takes the refuse-to-persist branch deterministically (the
    /// natural trigger is a swallowed adapter-truncate failure, which no test
    /// can provoke on demand).
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
}

impl FromTrainCmd for Qwen35FamilyCommand {
    #[inline]
    fn from_train(cmd: TrainCmd) -> Self {
        Qwen35FamilyCommand::Train(cmd)
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
impl crate::engine::model_command::FamilyCommand<Qwen35Inner> for Qwen35FamilyCommand {
    fn execute(self, inner: &mut Qwen35Inner) {
        match self {
            Qwen35FamilyCommand::DecideClef {
                request,
                cancelled,
                reply,
            } => {
                let _ = reply.send(inner.decide_clef(&request, &cancelled));
            }
            Qwen35FamilyCommand::CalibratePrefillRaw {
                texts,
                calib_seq,
                reply,
            } => {
                let _ = reply.send(inner.calibrate_prefill_raw_sync(texts, calib_seq));
            }
            Qwen35FamilyCommand::EvalTeacherForced { request, reply } => {
                let _ = reply.send(crate::quality::runner::run(inner, request));
            }
            Qwen35FamilyCommand::SaveModel { save_path, reply } => {
                let _ = reply.send(inner.save_model_sync(&save_path));
            }
            Qwen35FamilyCommand::CloseCache { deadline_ms, reply } => {
                use crate::models::qwen3_5::dflash2_state::disk::{close_cache, instant_at};
                let _ = reply.send(Ok(close_cache(inner, instant_at(deadline_ms))));
            }
            Qwen35FamilyCommand::MtpPagedGdnStateForTest { reply } => {
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
            Qwen35FamilyCommand::ForcePagedGdnMismatchForTest { reply } => {
                inner.paged_gdn_force_mismatch_for_test = true;
                let _ = reply.send(Ok(()));
            }
            Qwen35FamilyCommand::GdnHistoryCheckpointOracleForTest { reply } => {
                let _ = reply.send(inner.gdn_history_checkpoint_recompute_matches_for_test());
            }
            Qwen35FamilyCommand::Train(train_cmd) => {
                handle_train_cmd(inner, train_cmd);
            }
        }
    }
}

impl HybridSchedulerBackend for Qwen35Inner {
    type FamilyCommand = Qwen35FamilyCommand;
    type RestoreTicket = super::restore::Qwen35Restore;
    type OwnerState = Vec<u32>;
    type StepExecutor<'a> = crate::engine::hybrid_scheduler::HybridStepExecutor<'a, Self>;

    const SCHEDULER_NAME: &'static str = "Qwen3.5 dense";

    /// The decode lanes the load's memory plan granted.
    fn scheduler_capacity(&self) -> usize {
        self.scheduled_lanes() as usize
    }

    fn wired_residency(&self) -> Option<std::sync::Arc<crate::stream::WiredResidency>> {
        self.wired.clone()
    }

    fn paged_adapter(&self) -> Option<&PagedKVCacheAdapter> {
        self.paged_adapter.as_ref()
    }

    fn paged_adapter_mut(&mut self) -> Option<&mut PagedKVCacheAdapter> {
        self.paged_adapter.as_mut()
    }

    /// The turn's replay point leaves InUse when it closes. Its DFlash2 lane
    /// ends here, the first hook of a turn's end: the history checkpoint and
    /// the parked owner state read the sequence's cells after this.
    fn scheduled_turn_closed(&mut self, seq_id: SeqId) {
        self.drop_prefill_run(seq_id);
        crate::models::qwen3_5::scheduled_dflash2::end_lane(self, seq_id);
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

    fn hold_state_tier(&mut self) -> Option<crate::models::qwen3_5::dflash2_state::disk::HeldTier> {
        crate::models::qwen3_5::dflash2_state::disk::hold_tier(self)
    }

    fn release_state_tier(&mut self, held: crate::models::qwen3_5::dflash2_state::disk::HeldTier) {
        crate::models::qwen3_5::dflash2_state::disk::release_tier(self, held);
    }

    fn warm_state_restore(&mut self, seq_id: SeqId) -> Result<bool> {
        let cell = if self.active_scheduled_seq == Some(seq_id) {
            self.caches.as_deref()
        } else {
            self.scheduled_recurrent.live(seq_id).map(Vec::as_slice)
        };
        crate::models::qwen3_5::dflash2_state::paged::warm_install_linear(
            cell,
            self.fresh_dflash2_layer_caches(),
        )
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
        self.has_speculative_weights()
            && self.paged_adapter.is_some()
            && self.config.recurrent_state_bytes() > 0
            && self.cached_rope_deltas.unwrap_or(0) == 0
    }
    fn scheduled_draft_state_bytes(&self, total_tokens: u32) -> u64 {
        if let Some(bytes) = crate::models::qwen3_5::scheduled_dflash2::sequence_lane_bytes(self) {
            return bytes;
        }
        crate::models::qwen3_5::scheduled_mtp::ScheduledMtpState::reservation_bytes(
            &self.config,
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
        if let Some(bytes) = crate::models::qwen3_5::scheduled_dflash2::sequence_lane_bytes(self) {
            return bytes;
        }
        crate::models::qwen3_5::scheduled_mtp::ScheduledMtpState::held_bytes(
            &self.config,
            total_tokens,
        )
    }
    fn scheduled_lane_state_bytes(&self) -> u64 {
        crate::models::qwen3_5::scheduled_dflash2::sequence_lane_bytes(self).unwrap_or(0)
    }
    fn prepare_contended_prefix(
        &mut self,
        seq_id: SeqId,
        tokens: &[u32],
        _owner_history: &[u32],
        reuse_cache: bool,
        cache_salt: u64,
        block_size: u32,
    ) -> Result<
        crate::engine::hybrid_scheduler::ScheduledPrefixAdmission<
            Self::PrefixState,
            Self::RestoreTicket,
        >,
    > {
        self.prepare_parked_prefix(seq_id, tokens, reuse_cache, cache_salt, block_size)
    }
    fn poll_scheduled_restore(
        &mut self,
        _seq_id: SeqId,
        restore: &mut Self::RestoreTicket,
        uploads: &mut crate::transformer::paged_kv_cache_adapter::RestoreUploadBudget,
        _prompt_tokens: &[u32],
        _owner_history: &[u32],
        _is_preemption_replay: bool,
    ) -> Result<Option<crate::engine::hybrid_scheduler::ScheduledRestoreResult<Self::PrefixState>>>
    {
        self.poll_parked_restore(restore, uploads)
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
    ) -> Result<Option<crate::engine::hybrid_scheduler::ScheduledRestoreResult<Self::PrefixState>>>
    {
        self.land_parked_restore(seq_id, restore, prompt_tokens)
    }
    fn restore_reserved_blocks(restore: &Self::RestoreTicket) -> u32 {
        restore.held_blocks()
    }
    fn scheduled_prefix_probe(&self, tokens: &[u32], cache_salt: u64) -> u32 {
        crate::models::qwen3_5::dflash2_turn::stored_resume_depth(self, tokens, cache_salt)
    }
    fn scheduled_prefix_probe_generation(&self) -> u64 {
        self.dflash2_states.store.generation()
    }
    /// On a draft load the sequence's DFlash2 lane, at any position: its
    /// first prefill slice places the ring at the slice's start.
    fn begin_scheduled_speculation(&mut self, seq_id: SeqId, position: u32) -> Result<bool> {
        if let Some(draft) = self.dflash2.as_ref() {
            self.dflash2_states.begin_lane(seq_id, &draft.config);
            return Ok(true);
        }
        crate::models::qwen3_5::scheduled_mtp::ScheduledMtpTarget::begin_scheduled_mtp(
            self, seq_id, position,
        )
    }
    fn reserve_scheduled_speculation(&mut self, seq_id: SeqId, queries: usize) -> Result<bool> {
        if self.dflash2.is_some() && self.dflash2_states.seq_lanes.contains_key(&seq_id) {
            return crate::models::qwen3_5::scheduled_dflash2::reserve_lane(self, seq_id);
        }
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
    fn scheduled_sampled_rows(&self) -> bool {
        self.dflash2.is_some()
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
        if self.dflash2.is_some() {
            return crate::models::qwen3_5::scheduled_dflash2::propose(
                self, seq_id, anchor, cap, params, rng,
            );
        }
        crate::models::qwen3_5::scheduled_mtp::ScheduledMtpTarget::propose_scheduled_mtp(
            self, seq_id, anchor, cap, params, rng,
        )
    }
    fn run_scheduled_verify(
        &mut self,
        rows: &[crate::engine::hybrid_scheduler::ScheduledVerifyRow],
    ) -> Result<MxArray> {
        if self.dflash2.is_some() {
            return crate::models::qwen3_5::scheduled_dflash2::verify(self, rows);
        }
        crate::models::qwen3_5::scheduled_mtp::ScheduledMtpTarget::verify_scheduled_mtp(self, rows)
    }
    fn commit_scheduled_verify(
        &mut self,
        rows: &[crate::engine::hybrid_scheduler::ScheduledVerifyCommit],
    ) -> Result<Vec<Result<()>>> {
        if self.dflash2.is_some() {
            return crate::models::qwen3_5::scheduled_dflash2::commit(self, rows);
        }
        crate::models::qwen3_5::scheduled_mtp::ScheduledMtpTarget::commit_scheduled_mtp(self, rows)
    }
    fn release_scheduled_speculation(&mut self, seq_id: SeqId) {
        self.drop_prefill_run(seq_id);
        self.scheduled_mtp.release(seq_id);
        crate::models::qwen3_5::scheduled_dflash2::end_lane(self, seq_id);
    }

    fn token_vocab_size(&self) -> u32 {
        let rows = self.config.vocab_size.max(1) as u32;
        self.tokenizer
            .as_ref()
            .map_or(rows, |tokenizer| tokenizer.vocab_size().clamp(1, rows))
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

    fn per_sequence_entries(&self) -> usize {
        self.paged_adapter
            .as_ref()
            .map_or(0, PagedKVCacheAdapter::per_sequence_entries)
            + self.gdn_in_use.len()
    }

    fn retained_state_entries(&self) -> usize {
        self.gdn_prefix_checkpoints.len() + usize::from(self.gdn_last_history_checkpoint.is_some())
    }

    fn scheduled_speculation_owners(&self) -> usize {
        self.scheduled_mtp.owner_count()
            + self.dflash2_states.seq_lanes.len()
            + self
                .prefill_runs
                .values()
                .filter(|run| run.holds_decode_lane())
                .count()
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
        self.drop_prefill_run(seq_id);
        self.release_scheduled_recurrent_for(seq_id);
    }

    fn release_scheduled_cache(&mut self, seq_id: SeqId) -> Result<()> {
        self.drop_prefill_run(seq_id);
        let Some(adapter) = self.paged_adapter.as_mut() else {
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

    /// A unit in flight goes back to its start first: its booked rows are
    /// not written in the layers it did not run, so none of them registers.
    fn preempt_scheduled_cache(
        &mut self,
        seq_id: SeqId,
        cache_salt: u64,
        mode: crate::engine::scheduler::PreemptionMode,
    ) -> Result<()> {
        self.end_prefill_run(seq_id)?;
        let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
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
                mode == crate::engine::scheduler::PreemptionMode::Ssd,
            )
            .and_then(|_| adapter.release_request_for(seq_id).map(|_| ()))
            .map_err(Error::from_reason)
    }

    /// A draft load's text prefills take the draft-lane path; a pass checks
    /// the sequence's own path again.
    fn prefill_runs_in_passes(&self, seq_id: SeqId) -> bool {
        self.dflash2.is_some()
            && !self.scheduled_mtp.owners.contains_key(&seq_id)
            && self.cached_rope_deltas.unwrap_or(0) == 0
    }

    fn prefill_rows_done(&self, seq_id: SeqId) -> Option<u32> {
        self.prefill_runs
            .get(&seq_id)
            .map(|run| run.driver.rows_done(self.config.num_layers as usize))
    }

    fn run_scheduled_prefill_pass(
        &mut self,
        seq_id: SeqId,
        source: &[u32],
        base: &Self::PrefixState,
        start: usize,
        end: usize,
        generation_stream: Stream,
        first_chunk: bool,
        budget_ms: Option<f64>,
    ) -> Result<crate::engine::hybrid_scheduler::ScheduledPrefillPass> {
        use crate::engine::hybrid_scheduler::ScheduledPrefillPass;
        #[cfg(test)]
        let budget_ms = budget_ms.map(|ms| {
            crate::models::qwen3_5::paged_forward::PREFILL_PASS_BUDGET_OVERRIDE
                .with(std::cell::Cell::get)
                .unwrap_or(ms)
        });
        let held = self.prefill_runs.contains_key(&seq_id);
        if (budget_ms.is_none() && !held)
            || !self.prefill_takes_passes(seq_id)
            || self.draft_lane_turn()
        {
            self.end_prefill_run(seq_id)?;
            return self
                .run_scheduled_prefill_slice(
                    seq_id,
                    source,
                    base,
                    start,
                    end,
                    generation_stream,
                    first_chunk,
                )
                .map(ScheduledPrefillPass::Done);
        }
        let prefix = self.build_scheduled_prefix(
            base,
            start,
            end.saturating_sub(start),
            source.to_vec(),
            first_chunk,
        );
        self.dense_paged_prefill_pass(&prefix, start, end, budget_ms)
    }

    /// A draft lane's unit short of its source's end gives no logits (the
    /// scheduler drops them): its last layer only stores its K/V.
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
        if self.draft_lane_turn() {
            return self
                .paged_prefill(&source[start..end], &prefix, generation_stream)
                .map(Some);
        }
        let end_output = crate::models::qwen3_5::paged_forward::unit_end_output(end, source.len());
        self.dense_paged_prefill_unit(&source[start..end], &prefix, None, end_output)
            .map(|(logits, _)| logits)
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
            self.drop_live_dflash2_window("install_owner_state");
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
}
