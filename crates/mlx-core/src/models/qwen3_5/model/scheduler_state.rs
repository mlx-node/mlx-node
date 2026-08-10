//! Qwen3.5 dense continuous-batching state.
//!
//! Plain text autoregressive turns share one scheduler step. Native MTP,
//! multimodal turns, raw generation, calibration, persistence, and training
//! remain ordered barriers and execute through the existing command handler.

use super::*;

enum ScheduledReply {
    Sync(ResponseTx<ChatResult>),
    Stream(StreamTx<ChatStreamChunk>),
}

impl ScheduledReply {
    fn sink(&self) -> Option<&dyn ChunkSink> {
        match self {
            Self::Sync(_) => None,
            Self::Stream(stream) => Some(stream),
        }
    }

    fn send_error(self, error: Error, cancelled: &AtomicBool) {
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

struct ScheduledTurn {
    owner_id: String,
    tokenizer: Arc<Qwen3Tokenizer>,
    eos_id: u32,
    config: ChatConfig,
    params: engine::params::ChatParams,
    thinking: ThinkingSetup,
    prompt_tokens: Vec<u32>,
    prefix: Qwen35PrefixState,
    is_delta: bool,
    reuse_cache: bool,
    response: ScheduledReply,
    generated_tokens: Vec<u32>,
    finish_reason: String,
    reasoning_tracker: engine::penalties::ReasoningTracker,
    extra_eos_ids: Vec<u32>,
    generation_start: Option<Instant>,
    first_token_instant: Option<Instant>,
    generation_stream: Stream,
    profiler: crate::decode_profiler::DecodeProfiler,
    emitter: Option<Box<dyn StreamEmitter>>,
    stream_skip_special: bool,
    decode_ids: Vec<u32>,
    decode_prefix: String,
    decode_prefix_index: usize,
    streamed_text_len: usize,
    last_is_reasoning: bool,
    failure: Option<Error>,
    allocation_failed: bool,
    preemption_replay: Option<PreemptionReplay>,
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
    batch_index: Option<usize>,
}

struct Qwen35StepExecutor<'a> {
    inner: &'a mut Qwen35Inner,
}

impl Qwen35StepExecutor<'_> {
    fn blocked(row: &engine::scheduler::StepRow) -> RowStepResult {
        RowStepResult {
            seq_id: row.seq_id,
            num_computed_tokens: 0,
            generated_token: None,
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
        turn: &mut TurnState<ScheduledTurn>,
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
            finished: true,
            allocation_blocked: false,
            prefill_micros: 0,
        }
    }

    fn stream_token(turn: &mut TurnState<ScheduledTurn>, token_id: u32, is_reasoning: bool) {
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
            emitter.on_token_text(&text, is_reasoning, payload.params.include_reasoning, sink);
        }
    }

    fn execute_prefill(
        &mut self,
        row: &engine::scheduler::StepRow,
        turn: &mut TurnState<ScheduledTurn>,
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
                Error::from_reason("Qwen3.5 scheduler prefill slice exceeds prompt"),
            ));
        }
        if let Err(error) = self.inner.activate_paged_seq(row.seq_id) {
            return Ok(Self::fail(turn, row, error));
        }
        self.inner
            .set_turn_cancel_flag(turn.cancelled.as_ref().map(Arc::clone));
        let started = Instant::now();
        turn.payload.profiler.begin_prefill();
        let full_tokens = turn.payload.preemption_replay.as_ref().map_or_else(
            || turn.payload.prompt_tokens.clone(),
            |replay| replay.tokens.clone(),
        );
        let first_chunk = start == turn.payload.prefix.effective_cached_prefix_len;
        let prefix = Qwen35PrefixState {
            effective_cached_prefix_len: start,
            suffix_len: row.num_tokens as usize,
            full_tokens,
            gdn_prefix_already_primed: !first_chunk
                || turn.payload.prefix.gdn_prefix_already_primed,
        };
        let suffix = turn
            .payload
            .preemption_replay
            .as_ref()
            .map_or(&turn.payload.prompt_tokens[start..end], |replay| {
                &replay.tokens[start..end]
            });
        let logits = match self
            .inner
            .paged_prefill(suffix, &prefix, turn.payload.generation_stream)
        {
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
            return Ok(RowStepResult {
                seq_id: row.seq_id,
                num_computed_tokens: row.num_tokens,
                generated_token: None,
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
                finished: false,
                allocation_blocked: false,
                prefill_micros: Self::elapsed_micros(started),
            });
        }
        turn.payload.preemption_replay = None;
        let logits = match engine::penalties::apply_all_penalties(
            logits,
            &turn.token_history,
            &turn.payload.params,
        ) {
            Ok(logits) => logits,
            Err(error) => return Ok(Self::fail(turn, row, error)),
        };
        let sampled = match sample(&logits, turn.payload.params.sampling_config) {
            Ok(sampled) => sampled,
            Err(error) => return Ok(Self::fail(turn, row, error)),
        };
        sampled.eval();
        if turn.payload.params.report_performance {
            turn.payload.first_token_instant = Some(Instant::now());
        }
        crate::array::synchronize_and_clear_cache();
        if turn.payload.params.max_new_tokens == 0 {
            turn.payload.profiler.snapshot_memory_after();
            turn.payload.profiler.report();
            return Ok(RowStepResult {
                seq_id: row.seq_id,
                num_computed_tokens: row.num_tokens,
                generated_token: None,
                finished: true,
                allocation_blocked: false,
                prefill_micros: Self::elapsed_micros(started),
            });
        }
        let token = match sampled.item_at_int32(0) {
            Ok(token) => token as u32,
            Err(error) => return Ok(Self::fail(turn, row, error)),
        };
        Ok(RowStepResult {
            seq_id: row.seq_id,
            num_computed_tokens: row.num_tokens,
            generated_token: Some(token),
            finished: false,
            allocation_blocked: false,
            prefill_micros: Self::elapsed_micros(started),
        })
    }

    fn finish_decode_row(turn: &mut TurnState<ScheduledTurn>, row: &PreparedDecodeRow) {
        turn.payload.profiler.mark_first_token();
        let is_reasoning = turn.payload.reasoning_tracker.observe_token(row.token_id);
        turn.payload.last_is_reasoning = is_reasoning;
        if row.stops_at_eos {
            turn.payload.finish_reason = String::from("stop");
        } else if row.cancelled {
            turn.payload.finish_reason = String::from("cancelled");
        } else {
            if turn.payload.response.sink().is_some() {
                Self::stream_token(turn, row.token_id, is_reasoning);
            }
            if let Some(reason) = row.repetition {
                turn.payload.finish_reason = reason.to_string();
            }
        }
    }

    fn sample_next(turn: &mut TurnState<ScheduledTurn>, mut logits: MxArray) -> Result<u32> {
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
        turn.payload.profiler.begin("schedule_eval");
        MxArray::async_eval_arrays(&[&sampled, &logits]);
        turn.payload.profiler.end();
        sampled.eval();
        Ok(sampled.item_at_int32(0)? as u32)
    }

    fn prepare_decode_rows(
        plan: &StepPlan,
        running: &mut [TurnState<ScheduledTurn>],
    ) -> (
        Vec<PreparedDecodeRow>,
        Vec<(usize, RowStepResult)>,
        Vec<(SeqId, u32)>,
    ) {
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
                .expect("scheduler validated decode row");
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
            let terminal = stops_at_eos || planned.cancel_snapshot || repetition.is_some();
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
                batch_index,
            });
        }
        (work, early, batch)
    }

    fn execute_decode_rows(
        &mut self,
        plan: &StepPlan,
        running: &mut [TurnState<ScheduledTurn>],
    ) -> (Vec<(usize, RowStepResult)>, usize) {
        let (work, mut results, batch_rows) = Self::prepare_decode_rows(plan, running);
        let executed_decode_batch = batch_rows.len();
        crate::array::maybe_clear_cache_for_paged_step(plan.global_step as i32);
        for row in &work {
            if row.batch_index.is_some() {
                running
                    .iter_mut()
                    .find(|turn| turn.seq_id == row.seq_id)
                    .expect("prepared row")
                    .payload
                    .profiler
                    .begin("forward");
            }
        }
        let logits = if batch_rows.is_empty() {
            Ok(None)
        } else {
            let _stream_context = StreamContext::new(Stream::default(DeviceType::Gpu));
            self.inner
                .run_paged_decode_step_batched(&batch_rows)
                .map(Some)
        };
        for row in &work {
            if row.batch_index.is_some() {
                running
                    .iter_mut()
                    .find(|turn| turn.seq_id == row.seq_id)
                    .expect("prepared row")
                    .payload
                    .profiler
                    .end();
            }
        }
        for row in work {
            let planned = &plan.rows[row.plan_index];
            let turn = running
                .iter_mut()
                .find(|turn| turn.seq_id == row.seq_id)
                .expect("prepared row remains running");
            let row_logits = match (&logits, row.batch_index) {
                (Err(error), Some(_)) if is_paged_allocation_blocked(&error.reason) => {
                    let rolled_back = turn.payload.generated_tokens.pop();
                    debug_assert_eq!(rolled_back, Some(row.token_id));
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
                (Ok(Some(logits)), Some(index)) => match logits
                    .slice_axis(0, index as i64, index as i64 + 1)
                    .and_then(|row| row.squeeze(Some(&[1])))
                {
                    Ok(logits) => Some(logits),
                    Err(error) => {
                        results.push((row.plan_index, Self::fail(turn, planned, error)));
                        continue;
                    }
                },
                _ => None,
            };
            let next = match row_logits {
                Some(logits) => match Self::sample_next(turn, logits) {
                    Ok(token) => Some(token),
                    Err(error) => {
                        results.push((row.plan_index, Self::fail(turn, planned, error)));
                        continue;
                    }
                },
                None => None,
            };
            turn.payload.profiler.step();
            Self::finish_decode_row(turn, &row);
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
                    finished,
                    allocation_blocked: false,
                    prefill_micros: 0,
                },
            ));
        }
        (results, executed_decode_batch)
    }
}

impl StepExecutor<ScheduledTurn> for Qwen35StepExecutor<'_> {
    type Error = std::convert::Infallible;

    fn execute(
        &mut self,
        plan: &StepPlan,
        running: &mut [TurnState<ScheduledTurn>],
    ) -> std::result::Result<StepResult, Self::Error> {
        let mut rows = Vec::with_capacity(plan.rows.len());
        rows.resize_with(plan.rows.len(), || None);
        let (decode_results, executed_decode_batch) = self.execute_decode_rows(plan, running);
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
                .expect("scheduler validated running row");
            rows[index] = Some(
                self.execute_prefill(planned, turn)
                    .unwrap_or_else(|error| Self::fail(turn, planned, error)),
            );
        }
        Ok(StepResult {
            rows: rows
                .into_iter()
                .map(|row| row.expect("every planned row executed"))
                .collect(),
            executed_decode_batch,
            rows_alloc_evicted: running
                .iter()
                .filter(|turn| turn.payload.allocation_failed)
                .count() as u32,
        })
    }
}

pub(crate) struct Qwen35SchedulerState {
    inner: Qwen35Inner,
    enabled: bool,
    scheduler: Scheduler<ScheduledTurn, Qwen35Cmd, Qwen35Cmd>,
    pending: VecDeque<Qwen35Cmd>,
    prepared_waiting: Option<Box<PreparedTurn>>,
    owner_sequences: HashMap<String, SeqId>,
    owner_histories: HashMap<String, Vec<u32>>,
    next_seq_id: SeqId,
}

impl Qwen35SchedulerState {
    pub(crate) fn continuous_batching_enabled() -> bool {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var("MLX_QWEN35_CONTINUOUS_BATCHING").is_ok_and(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
        })
    }

    pub(crate) fn new(inner: Qwen35Inner) -> Self {
        let enabled = Self::continuous_batching_enabled() && inner.paged_adapter.is_some();
        Self {
            inner,
            enabled,
            scheduler: Scheduler::new(scheduler_max_num_seqs(), scheduler_max_batched_tokens())
                .expect("validated Qwen3.5 scheduler limits"),
            pending: VecDeque::new(),
            prepared_waiting: None,
            owner_sequences: HashMap::new(),
            owner_histories: HashMap::new(),
            next_seq_id: 1,
        }
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

    fn chat_requires_barrier(&self, command: &ChatCmd) -> bool {
        if Self::chat_config(command).is_some_and(|config| config.enable_mtp == Some(true)) {
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

    fn release_cache_owner_now(&mut self, owner_id: &str) -> Result<()> {
        let Some(&seq_id) = self.owner_sequences.get(owner_id) else {
            self.owner_histories.remove(owner_id);
            return Ok(());
        };
        let release_error = self.inner.paged_adapter.as_mut().and_then(|adapter| {
            if adapter.block_table_for(seq_id).is_some() {
                adapter.release_request_for(seq_id).err()
            } else {
                None
            }
        });
        self.inner.release_scheduled_recurrent_for(seq_id);
        if let Some(error) = release_error {
            return Err(Error::from_reason(error));
        }
        self.owner_sequences.remove(owner_id);
        self.owner_histories.remove(owner_id);
        Ok(())
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
                    .enqueue_barrier(Qwen35Cmd::Chat(ChatCmd::ResetCaches { reply }));
                return None;
            }
            ChatCmd::ReleaseCacheOwner { .. } => {
                unreachable!("cache-owner release is handled before turn preparation")
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
        if kind == engine::session::TurnKind::Continue
            && !self.owner_histories.contains_key(&owner_id)
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
        self.inner.cached_token_history = self
            .owner_histories
            .get(&owner_id)
            .cloned()
            .unwrap_or_default();
        let mut admitted =
            match engine::session::admit_paged_turn(&mut self.inner, messages, config, kind) {
                Ok(admitted) => admitted,
                Err(error) => {
                    if newly_assigned {
                        self.owner_sequences.remove(&owner_id);
                    }
                    response.send_error(error, cancelled.as_ref());
                    return None;
                }
            };
        if admitted.plan.path() != engine::plan::TurnPath::Paged
            || !admitted.images.is_empty()
            || !admitted.audio.is_empty()
            || admitted.params.enable_mtp
        {
            if newly_assigned {
                self.owner_sequences.remove(&owner_id);
            }
            response.send_error(
                Error::from_reason(
                    "Qwen3.5 scheduler only admits plain text paged autoregressive turns",
                ),
                cancelled.as_ref(),
            );
            return None;
        }
        let prompt_tokens = admitted.tokens.len() as u32;
        let requested_max_new_tokens = admitted.params.max_new_tokens.max(0) as u32;
        let trained_context = u32::try_from(self.inner.config.max_position_embeddings)
            .unwrap_or(1)
            .max(1);
        let context = trained_context.min(scheduler_per_seq_context());
        let max_new_tokens = match engine::scheduler::clamp_scheduled_output_tokens(
            prompt_tokens,
            requested_max_new_tokens,
            context,
        ) {
            Ok(max_new_tokens) => max_new_tokens,
            Err(error) => {
                if newly_assigned {
                    self.owner_sequences.remove(&owner_id);
                }
                response.send_error(Error::from_reason(error), cancelled.as_ref());
                return None;
            }
        };
        admitted.params.max_new_tokens = max_new_tokens as i32;
        let requested_tokens = prompt_tokens.saturating_add(max_new_tokens);
        let block_size = self
            .inner
            .paged_adapter
            .as_ref()
            .expect("paged route checked")
            .block_size();
        let full_blocks = requested_tokens.div_ceil(block_size);
        let reservation_blocks = if scheduler_reserve_full_isl() {
            full_blocks
        } else {
            prompt_tokens
                .div_ceil(block_size)
                .saturating_add(u32::from(max_new_tokens != 0))
                .min(full_blocks)
        };
        Some(Box::new(PreparedTurn {
            admitted,
            response,
            cancelled,
            owner_id,
            seq_id,
            newly_assigned,
            reservation_blocks,
            block_size,
        }))
    }

    fn cleanup_rejected_prepared(&mut self, prepared: &PreparedTurn) {
        if prepared.newly_assigned {
            self.owner_histories.remove(&prepared.owner_id);
            self.owner_sequences.remove(&prepared.owner_id);
            self.inner.release_scheduled_recurrent_for(prepared.seq_id);
        }
    }

    /// Keep the two-unit residency cap as a queueing boundary. An idle warm
    /// row may be discarded because its owner history can rebuild the exact
    /// GDN state through the ordinary prefix-prime path; a scheduler-owned row
    /// is never evicted here.
    fn ensure_recurrent_slot(&mut self, seq_id: SeqId) -> bool {
        if self.inner.can_activate_scheduled_recurrent(seq_id) {
            return true;
        }
        let idle_victim = self
            .owner_sequences
            .values()
            .copied()
            .filter(|&candidate| {
                candidate != seq_id
                    && self.inner.has_scheduled_recurrent(candidate)
                    && !self.scheduler.contains_seq(candidate)
            })
            .min();
        let Some(victim) = idle_victim else {
            return false;
        };
        self.inner.release_scheduled_recurrent_for(victim);
        self.inner.can_activate_scheduled_recurrent(seq_id)
    }

    fn admit_prepared(
        &mut self,
        prepared: Box<PreparedTurn>,
    ) -> std::result::Result<Option<TurnState<ScheduledTurn>>, Box<PreparedTurn>> {
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
        let (telemetry, bytes_per_block) = match self
            .inner
            .paged_adapter
            .as_ref()
            .expect("paged route checked")
            .block_telemetry()
            .and_then(|telemetry| {
                self.inner
                    .paged_adapter
                    .as_ref()
                    .expect("paged route checked")
                    .bytes_per_block()
                    .map(|bytes| (telemetry, bytes))
            }) {
            Ok(values) => values,
            Err(error) => {
                self.cleanup_rejected_prepared(&prepared);
                prepared
                    .response
                    .send_error(Error::from_reason(error), prepared.cancelled.as_ref());
                return Ok(None);
            }
        };
        if prepared.reservation_blocks > telemetry.total_blocks {
            self.cleanup_rejected_prepared(&prepared);
            prepared.response.send_error(
                Error::from_reason(format!(
                    "context_length_exceeded: request requires {} paged blocks but the pool has {}",
                    prepared.reservation_blocks, telemetry.total_blocks
                )),
                prepared.cancelled.as_ref(),
            );
            return Ok(None);
        }
        if !self.ensure_recurrent_slot(prepared.seq_id) {
            return Err(prepared);
        }
        let candidate_state_bytes = if self.inner.has_scheduled_recurrent(prepared.seq_id) {
            0
        } else {
            self.inner.config.recurrent_state_bytes()
        };
        let decision = self.scheduler.try_reserve_memory(
            engine::scheduler::MemoryTelemetry {
                capacity_bytes: u64::from(telemetry.total_blocks).saturating_mul(bytes_per_block),
                free_bytes: u64::from(telemetry.free_blocks).saturating_mul(bytes_per_block),
                reclaimable_bytes: u64::from(telemetry.reclaimable_blocks)
                    .saturating_mul(bytes_per_block),
            },
            prepared.reservation_blocks,
            bytes_per_block,
            self.inner.scheduled_recurrent_bytes(),
            candidate_state_bytes,
            scheduler_watermark_fraction(),
        );
        if !decision.admitted {
            return Err(prepared);
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
        } = *prepared;
        if cancelled.load(Ordering::Relaxed) {
            if newly_assigned {
                self.owner_histories.remove(&owner_id);
                self.owner_sequences.remove(&owner_id);
                self.inner.release_scheduled_recurrent_for(seq_id);
            }
            response.send_error(
                Error::from_reason(engine::session::CHAT_SESSION_CANCELLED),
                cancelled.as_ref(),
            );
            return Ok(None);
        }
        if let Err(error) = self.inner.activate_scheduled_recurrent(seq_id) {
            if newly_assigned {
                self.owner_histories.remove(&owner_id);
                self.owner_sequences.remove(&owner_id);
                self.inner.release_scheduled_recurrent_for(seq_id);
            }
            response.send_error(error, cancelled.as_ref());
            return Ok(None);
        }
        self.inner.cached_token_history = self
            .owner_histories
            .get(&owner_id)
            .cloned()
            .unwrap_or_default();
        self.inner.set_cache_owner_id(
            &admitted.params.cache_owner_id,
            admitted.params.cache_root_owner_id.as_deref(),
        );
        self.inner
            .set_turn_cancel_flag(Some(Arc::clone(&cancelled)));
        let prefix =
            match self
                .inner
                .prime_prefix_state(&admitted.tokens, true, block_size as usize, &[], 0)
            {
                Ok(prefix) => prefix,
                Err(error) => {
                    self.inner.abort_paged_turn();
                    self.inner.release_scheduled_recurrent_for(seq_id);
                    self.inner.set_turn_cancel_flag(None);
                    self.owner_histories.remove(&owner_id);
                    self.owner_sequences.remove(&owner_id);
                    response.send_error(error, cancelled.as_ref());
                    return Ok(None);
                }
            };
        if prefix.suffix_len == 0 {
            self.inner.abort_paged_turn();
            self.inner.release_scheduled_recurrent_for(seq_id);
            self.inner.set_turn_cancel_flag(None);
            self.owner_histories.remove(&owner_id);
            self.owner_sequences.remove(&owner_id);
            response.send_error(
                Error::from_reason("Qwen3.5 scheduler produced an empty prefill suffix"),
                cancelled.as_ref(),
            );
            return Ok(None);
        }
        let materialized_blocks = self
            .inner
            .paged_adapter
            .as_ref()
            .and_then(|adapter| adapter.block_table_for(seq_id))
            .map(|table| table.num_blocks() as u32)
            .unwrap_or(0);
        let is_streaming = response.sink().is_some();
        let generation_stream = Stream::new(DeviceType::Gpu);
        let mut profiler = crate::decode_profiler::DecodeProfiler::new(
            self.inner
                .profiler_label(admitted.plan.is_delta, is_streaming),
            self.inner.family_name(),
        );
        profiler.set_prompt_tokens(prefix.suffix_len as u32);
        profiler.snapshot_memory_before();
        let generation_start = admitted.params.report_performance.then(Instant::now);
        let reuse_cache = admitted.plan.is_delta || admitted.params.reuse_cache;
        let payload = ScheduledTurn {
            owner_id,
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
            reasoning_tracker: engine::penalties::ReasoningTracker::from_setup(
                &admitted.thinking,
                admitted.think_end_id,
            ),
            extra_eos_ids: self.inner.extra_eos_ids(),
            generation_start,
            first_token_instant: None,
            generation_stream,
            profiler,
            emitter: is_streaming.then(|| self.inner.stream_emitter()),
            stream_skip_special: self.inner.stream_skip_special_tokens(),
            decode_ids: Vec::new(),
            decode_prefix: String::new(),
            decode_prefix_index: 0,
            streamed_text_len: 0,
            last_is_reasoning: admitted.thinking.enabled,
            failure: None,
            allocation_failed: false,
            preemption_replay: None,
        };
        let prompt_len = admitted.tokens.len() as u32;
        let mut breaks = Vec::new();
        let mut boundary = payload.prefix.effective_cached_prefix_len as u32;
        while boundary < prompt_len {
            boundary = boundary
                .saturating_add(scheduler_long_prefill_tokens())
                .min(prompt_len);
            breaks.push(boundary);
        }
        let turn = TurnState::new(
            seq_id,
            admitted.tokens,
            payload.prefix.effective_cached_prefix_len as u32,
            breaks,
            Some(cancelled),
            payload,
        )
        .expect("prime-derived Qwen3.5 turn must satisfy scheduler invariants")
        .with_block_reservation(reservation_blocks, materialized_blocks, block_size)
        .with_recurrent_state_reservation(self.inner.config.recurrent_state_bytes());
        Ok(Some(turn))
    }

    fn fail_preempted(&mut self, mut turn: TurnState<ScheduledTurn>, error: String) {
        turn.payload.failure = Some(Error::from_reason(error));
        self.finish_completed(turn);
    }

    fn enqueue_turn_or_reject(&mut self, mut turn: TurnState<ScheduledTurn>) {
        if self.scheduler.contains_seq(turn.seq_id) {
            let error = format!("duplicate scheduler sequence {}", turn.seq_id);
            tracing::error!("Qwen3.5 scheduler admission failed: {error}");
            turn.payload.failure = Some(Error::from_reason(error));
            self.finish_completed(turn);
            return;
        }
        self.scheduler
            .enqueue_turn(turn)
            .expect("duplicate prechecked above");
    }

    fn reap_cancelled_waiters(&mut self) {
        for mut turn in self.scheduler.take_cancelled_waiters() {
            turn.payload.failure =
                Some(Error::from_reason(engine::session::CHAT_SESSION_CANCELLED));
            self.finish_completed(turn);
        }
    }

    /// Release both halves of a hybrid victim. Full-attention blocks remain
    /// reusable through their verified hashes; request-local GDN arrays are
    /// deliberately dropped because they cannot be rewound. Resume enters
    /// through `prime_prefix_state`, whose right-to-left checkpoint/sidecar
    /// lookup reconstructs the deepest GDN boundary that agrees with K/V.
    fn handle_preempted(&mut self, mut turn: TurnState<ScheduledTurn>) {
        let prefix_tokens = turn.num_computed_tokens;
        let prompt_tokens = turn.payload.prompt_tokens.clone();
        let replay = match install_preemption_replay(
            &mut turn,
            &prompt_tokens,
            scheduler_long_prefill_tokens(),
        ) {
            Ok(replay) => replay,
            Err(error) => {
                self.fail_preempted(turn, error);
                return;
            }
        };
        let (bytes_per_block, has_cold_tier) = match self.inner.paged_adapter.as_ref() {
            Some(adapter) => match adapter.bytes_per_block() {
                Ok(bytes) => (bytes, adapter.cold_tier().is_some()),
                Err(error) => {
                    self.fail_preempted(turn, error);
                    return;
                }
            },
            None => {
                self.fail_preempted(turn, "Qwen3.5 paged adapter is unavailable".to_string());
                return;
            }
        };
        let mut mode =
            self.scheduler
                .preemption_mode(prefix_tokens, turn.block_size, bytes_per_block);
        if mode == PreemptionMode::Ssd && !has_cold_tier {
            mode = PreemptionMode::Recompute;
        }
        let lifecycle = self
            .inner
            .paged_adapter
            .as_mut()
            .expect("Qwen3.5 paged adapter checked above")
            .register_full_blocks_for_reuse_for(turn.seq_id, &[], 0, mode == PreemptionMode::Ssd)
            .and_then(|_| {
                self.inner
                    .paged_adapter
                    .as_mut()
                    .expect("Qwen3.5 paged adapter checked above")
                    .release_request_for(turn.seq_id)
                    .map(|_| ())
            });
        self.inner.release_scheduled_recurrent_for(turn.seq_id);
        if let Err(error) = lifecycle {
            let _ = self
                .inner
                .paged_adapter
                .as_mut()
                .expect("Qwen3.5 paged adapter checked above")
                .release_request_for(turn.seq_id);
            self.fail_preempted(turn, error);
            return;
        }
        turn.payload.preemption_replay = Some(replay);
        self.scheduler.record_preemption_mode(mode);
        self.scheduler.prepend_preempted(turn);
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
        let (telemetry, bytes_per_block) = match self
            .inner
            .paged_adapter
            .as_ref()
            .expect("preempted Qwen3.5 turn requires paged adapter")
            .block_telemetry()
            .and_then(|telemetry| {
                self.inner
                    .paged_adapter
                    .as_ref()
                    .expect("adapter exists")
                    .bytes_per_block()
                    .map(|bytes| (telemetry, bytes))
            }) {
            Ok(values) => values,
            Err(error) => {
                self.fail_preempted(turn, error);
                return;
            }
        };
        if !self.ensure_recurrent_slot(turn.seq_id) {
            self.scheduler.prepend_preempted(turn);
            return;
        }
        let decision = self.scheduler.try_reserve_memory(
            engine::scheduler::MemoryTelemetry {
                capacity_bytes: u64::from(telemetry.total_blocks).saturating_mul(bytes_per_block),
                free_bytes: u64::from(telemetry.free_blocks).saturating_mul(bytes_per_block),
                reclaimable_bytes: u64::from(telemetry.reclaimable_blocks)
                    .saturating_mul(bytes_per_block),
            },
            turn.block_reservation_total,
            bytes_per_block,
            self.inner.scheduled_recurrent_bytes(),
            turn.recurrent_state_bytes,
            scheduler_watermark_fraction(),
        );
        if !decision.admitted {
            self.scheduler.prepend_preempted(turn);
            return;
        }
        let Some(replay) = turn.payload.preemption_replay.as_ref() else {
            self.fail_preempted(
                turn,
                "Qwen3.5 preempted turn is missing replay state".to_string(),
            );
            return;
        };
        let replay_tokens = replay.tokens.clone();
        let target = replay_tokens.len() as u32;
        self.inner.cached_token_history = self
            .owner_histories
            .get(&turn.payload.owner_id)
            .cloned()
            .unwrap_or_default();
        self.inner.set_cache_owner_id(
            &turn.payload.params.cache_owner_id,
            turn.payload.params.cache_root_owner_id.as_deref(),
        );
        if let Err(error) = self.inner.activate_scheduled_recurrent(turn.seq_id) {
            self.fail_preempted(turn, error.reason);
            return;
        }
        let prefix = match self.inner.prime_prefix_state(
            &replay_tokens,
            true,
            turn.block_size as usize,
            &[],
            0,
        ) {
            Ok(prefix) => prefix,
            Err(error) if is_paged_allocation_blocked(&error.reason) => {
                let _ = self
                    .inner
                    .paged_adapter
                    .as_mut()
                    .expect("preempted Qwen3.5 turn requires paged adapter")
                    .release_request_for(turn.seq_id);
                self.inner.release_scheduled_recurrent_for(turn.seq_id);
                self.scheduler.prepend_preempted(turn);
                return;
            }
            Err(error) => {
                let _ = self
                    .inner
                    .paged_adapter
                    .as_mut()
                    .expect("preempted Qwen3.5 turn requires paged adapter")
                    .release_request_for(turn.seq_id);
                self.inner.release_scheduled_recurrent_for(turn.seq_id);
                self.fail_preempted(turn, error.reason);
                return;
            }
        };
        let materialized_blocks = self
            .inner
            .paged_adapter
            .as_ref()
            .and_then(|adapter| adapter.block_table_for(turn.seq_id))
            .map(|table| table.num_blocks() as u32)
            .unwrap_or(0);
        turn.num_computed_tokens = prefix.effective_cached_prefix_len as u32;
        turn.block_materialized_blocks = materialized_blocks;
        turn.pinned_prefill_breaks.clear();
        let mut boundary = turn.num_computed_tokens;
        while boundary < target {
            boundary = boundary
                .saturating_add(scheduler_long_prefill_tokens())
                .min(target);
            turn.pinned_prefill_breaks.push(boundary);
        }
        turn.payload
            .preemption_replay
            .as_mut()
            .expect("replay checked above")
            .cached_prefix = turn.num_computed_tokens;
        turn.payload.prefix = prefix;
        self.scheduler.ready_preempted(turn, false);
    }

    fn finish_completed(&mut self, mut turn: TurnState<ScheduledTurn>) {
        let cancelled = turn.cancelled.take().expect("scheduled turn cancel flag");
        if let Err(error) = self.inner.activate_paged_seq(turn.seq_id) {
            self.inner.release_scheduled_recurrent_for(turn.seq_id);
            self.owner_histories.remove(&turn.payload.owner_id);
            self.owner_sequences.remove(&turn.payload.owner_id);
            turn.payload.response.send_error(error, cancelled.as_ref());
            return;
        }
        if let Some(error) = turn.payload.failure.take() {
            self.inner.abort_paged_turn();
            self.inner.release_scheduled_recurrent_for(turn.seq_id);
            self.inner.set_turn_cancel_flag(None);
            self.owner_histories.remove(&turn.payload.owner_id);
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
                effective_cached_prefix_len: turn.payload.prefix.effective_cached_prefix_len,
                suffix_len: turn.payload.prefix.suffix_len,
                generated_tokens: &turn.payload.generated_tokens,
                finish_reason: std::mem::take(&mut turn.payload.finish_reason),
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
        let parked = if outcome.is_ok() && turn.payload.reuse_cache {
            self.owner_histories.insert(
                turn.payload.owner_id.clone(),
                self.inner.cached_token_history.clone(),
            );
            self.inner.park_active_scheduled_recurrent()
        } else {
            self.inner.release_scheduled_recurrent_for(turn.seq_id);
            Ok(())
        };
        self.inner.set_turn_cancel_flag(None);
        if outcome.is_err() || parked.is_err() {
            self.owner_histories.remove(&turn.payload.owner_id);
            self.owner_sequences.remove(&turn.payload.owner_id);
        }
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
                        Ok(TurnOutput::Streamed) => Err(Error::from_reason(
                            "Qwen3.5 scheduler returned streamed output for a sync turn",
                        )),
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
        receiver: &mut tokio::sync::mpsc::UnboundedReceiver<Qwen35Cmd>,
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
        while !deferred
            && !self.scheduler.has_pending_control()
            && let Some(command) = self.pending.front()
        {
            let must_wait_for_legacy_owner = matches!(command, Qwen35Cmd::Chat(chat)
                if !Self::chat_has_explicit_owner(chat) && self.scheduler.has_work());
            if must_wait_for_legacy_owner {
                break;
            }
            if matches!(command, Qwen35Cmd::Chat(chat) if !matches!(chat, ChatCmd::ReleaseCacheOwner { .. }))
                && self.scheduler.waiting_len() + self.scheduler.running_len()
                    >= self.scheduler.max_num_seqs()
            {
                break;
            }
            let command = self.pending.pop_front().expect("front checked");
            if (Self::force_serial() || !self.enabled)
                && !matches!(command, Qwen35Cmd::Chat(ChatCmd::ReleaseCacheOwner { .. }))
            {
                self.scheduler.enqueue_exclusive(command);
                break;
            }
            match command {
                Qwen35Cmd::Chat(ChatCmd::ReleaseCacheOwner { owner_id, reply }) => {
                    if self.cache_owner_release_blocked(&owner_id) {
                        self.pending
                            .push_front(Qwen35Cmd::Chat(ChatCmd::ReleaseCacheOwner {
                                owner_id,
                                reply,
                            }));
                        break;
                    }
                    let _ = reply.send(self.release_cache_owner_now(&owner_id));
                }
                Qwen35Cmd::Chat(chat) => {
                    let is_reset = matches!(chat, ChatCmd::ResetCaches { .. });
                    let mut exclusive = false;
                    if self.inner.paged_adapter.is_none()
                        || !Self::chat_has_explicit_owner(&chat)
                        || self.chat_requires_barrier(&chat)
                    {
                        self.scheduler.enqueue_exclusive(Qwen35Cmd::Chat(chat));
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
                barrier => {
                    self.scheduler.enqueue_barrier(barrier);
                    break;
                }
            }
        }

        let action = {
            let mut executor = Qwen35StepExecutor {
                inner: &mut self.inner,
            };
            self.scheduler.drive_once(&mut executor)
        };
        let scheduler_idle = matches!(&action, Ok(SchedulerAction::Idle));
        let mut may_resume_preempted = true;
        match action {
            Ok(SchedulerAction::Idle) => {}
            Ok(SchedulerAction::Exclusive(command) | SchedulerAction::Barrier(command)) => {
                if let Qwen35Cmd::SchedulerStats { reply } = command {
                    let _ = reply.send(Ok(self.scheduler.stats().to_js()));
                    return LoopControl::Continue;
                }
                let reset = matches!(command, Qwen35Cmd::Chat(ChatCmd::ResetCaches { .. }));
                handle_qwen35_cmd(&mut self.inner, command);
                if reset {
                    self.owner_sequences.clear();
                    self.owner_histories.clear();
                }
            }
            Ok(SchedulerAction::Stepped {
                completed,
                preempted,
                ..
            }) => {
                for turn in completed {
                    self.finish_completed(turn);
                }
                if let Some(turn) = preempted {
                    self.handle_preempted(turn);
                    may_resume_preempted = false;
                }
            }
            Err(error) => panic!("Qwen3.5 scheduler invariant failure: {error:?}"),
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
mod tests {
    use super::*;
    use crate::models::qwen3_5::config::Qwen3_5Config;

    fn tiny_config() -> Qwen3_5Config {
        Qwen3_5Config {
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
            paged_block_size: Some(16),
            use_block_paged_cache: None,
            persist_paged_cache: None,
            n_mtp_layers: 0,
        }
    }

    #[test]
    fn recurrent_slot_eviction_skips_history_only_owners() {
        let inner = Qwen35Inner::new(tiny_config()).expect("construct tiny dense model");
        let mut state = Qwen35SchedulerState::new(inner);
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
    fn cache_owner_release_drops_history_sequence_and_recurrent_state() {
        let inner = Qwen35Inner::new(tiny_config()).expect("construct tiny dense model");
        let mut state = Qwen35SchedulerState::new(inner);
        state.owner_sequences.insert("stateless-owner".into(), 9);
        state
            .owner_histories
            .insert("stateless-owner".into(), vec![7, 8]);
        state
            .inner
            .activate_scheduled_recurrent(9)
            .expect("activate owner recurrent state");

        state
            .release_cache_owner_now("stateless-owner")
            .expect("release owner");

        assert!(!state.owner_sequences.contains_key("stateless-owner"));
        assert!(!state.owner_histories.contains_key("stateless-owner"));
        assert!(!state.inner.has_scheduled_recurrent(9));
    }

    #[test]
    fn decode_residency_accepts_extra_warm_rows() {
        let mut inner = Qwen35Inner::new(tiny_config()).expect("construct tiny dense model");
        inner
            .activate_scheduled_recurrent(11)
            .expect("activate warm row");
        inner
            .activate_scheduled_recurrent(22)
            .expect("park warm and activate selected row");
        inner
            .park_active_scheduled_recurrent()
            .expect("park selected row");

        assert_eq!(inner.scheduled_recurrent.live_len(), 2);
        inner
            .validate_scheduled_decode_residency(&[(22, 7)])
            .expect("a decode subset may coexist with an extra warm row");
        let error = inner
            .validate_scheduled_decode_residency(&[(33, 7)])
            .expect_err("a genuinely missing selected row must still fail closed");
        assert!(error.reason.contains("33"));
    }
}
