//! One packed verifier wave, followed by independent owner commits and emission.
use super::*;
use engine::backend::DsparkProposal;
use engine::dspark_turn::{CycleStop, accept_dspark_proposal, clamp_dspark_cycle};
use engine::scheduler::SpeculativeRowResult;

struct VerifyWork {
    plan_index: usize,
    turn_index: usize,
    proposal: DsparkProposal,
    forced: Option<u32>,
}

fn empty_proposal() -> DsparkProposal {
    DsparkProposal {
        draft_ids: Vec::new(),
        draft_dists: Vec::new(),
        draft_sparse_dists: Vec::new(),
        keep_probabilities: None,
    }
}

impl<B: HybridSchedulerBackend> HybridStepExecutor<'_, B> {
    fn speculative_failure(
        turn: &mut TurnState<ScheduledTurn<B::PrefixState>>,
        row: &engine::scheduler::StepRow,
        error: Error,
    ) -> RowStepResult {
        let mut result = Self::fail(turn, row, error);
        result.num_computed_tokens = 0;
        result.speculative = Some(SpeculativeRowResult {
            verified_tokens: 0,
            generated_tokens: Vec::new(),
        });
        result
    }

    pub(super) fn execute_speculative_rows(
        &mut self,
        plan: &StepPlan,
        running: &mut [TurnState<ScheduledTurn<B::PrefixState>>],
    ) -> Result<(Vec<(usize, RowStepResult)>, usize, usize)> {
        let mut results = Vec::new();
        let mut work = Vec::new();
        let mut verify_rows = Vec::new();
        for (plan_index, row) in plan
            .rows
            .iter()
            .enumerate()
            .filter(|(_, row)| row.kind == StepKind::Decode)
        {
            let turn_index = running
                .iter()
                .position(|turn| turn.seq_id == row.seq_id)
                .ok_or_else(|| Error::from_reason("scheduled verifier lost its owner"))?;
            let turn = &mut running[turn_index];
            let Some(&anchor) = turn.token_history.last() else {
                results.push((
                    plan_index,
                    Self::speculative_failure(
                        turn,
                        row,
                        Error::from_reason("missing speculative anchor"),
                    ),
                ));
                continue;
            };
            // Prefill creates an un-emitted seed. Every later cycle emits its
            // boundary immediately, leaving that same token pending for KV.
            if !turn.payload.pending_token_emitted {
                turn.payload.generated_tokens.push(anchor);
                turn.payload.profiler.step();
                let eos =
                    anchor == turn.payload.eos_id || turn.payload.extra_eos_ids.contains(&anchor);
                let repetition = check_repetition_cutoff(
                    &turn.payload.generated_tokens,
                    turn.payload.params.max_consecutive_tokens,
                    turn.payload.params.max_ngram_repeats,
                    turn.payload.params.ngram_size,
                );
                let observed = !row.cancel_snapshot
                    && !(eos && !B::STREAM_EOS_TOKEN)
                    && turn
                        .payload
                        .turn_token_observer
                        .as_mut()
                        .is_some_and(|observer| observer.observe_token_id(anchor));
                let seed = PreparedDecodeRow {
                    plan_index,
                    seq_id: row.seq_id,
                    token_id: anchor,
                    terminal: eos || row.cancel_snapshot || repetition.is_some() || observed,
                    at_length: turn.payload.generated_tokens.len()
                        >= turn.payload.params.max_new_tokens.max(0) as usize,
                    stops_at_eos: eos,
                    cancelled: row.cancel_snapshot,
                    repetition,
                    observer_stopped: observed,
                    batch_index: None,
                };
                Self::finish_decode_row(turn, &seed);
                turn.payload.pending_token_emitted = true;
                if seed.terminal || seed.at_length {
                    turn.payload.profiler.snapshot_memory_after();
                    turn.payload.profiler.report();
                    results.push((
                        plan_index,
                        RowStepResult {
                            seq_id: row.seq_id,
                            num_computed_tokens: 0,
                            generated_token: None,
                            speculative: Some(SpeculativeRowResult {
                                verified_tokens: 0,
                                generated_tokens: Vec::new(),
                            }),
                            finished: true,
                            allocation_blocked: false,
                            prefill_micros: 0,
                        },
                    ));
                    continue;
                }
            }
            if row.cancel_snapshot {
                turn.payload.finish_reason = "cancelled".into();
                turn.payload.profiler.snapshot_memory_after();
                turn.payload.profiler.report();
                results.push((
                    plan_index,
                    RowStepResult {
                        seq_id: row.seq_id,
                        num_computed_tokens: 0,
                        generated_token: None,
                        speculative: Some(SpeculativeRowResult {
                            verified_tokens: 0,
                            generated_tokens: Vec::new(),
                        }),
                        finished: true,
                        allocation_blocked: false,
                        prefill_micros: 0,
                    },
                ));
                continue;
            }
            let max_new = turn.payload.params.max_new_tokens.max(0) as usize;
            let remaining = max_new.saturating_sub(turn.payload.generated_tokens.len());
            let mut cap = if turn.payload.scheduled_speculation.is_some() {
                (row.num_tokens as usize).saturating_sub(1)
            } else {
                0
            };
            cap = cap.min(remaining.saturating_sub(1));
            if let Some(budget) = turn.payload.reasoning_tracker.unforced_token_budget() {
                cap = cap.min(budget.saturating_sub(1));
            }
            let forced = if turn.payload.reasoning_tracker.force_think_end_pending() {
                cap = 0;
                Some(turn.payload.reasoning_tracker.forced_token_id()?)
            } else {
                None
            };
            // Reserve before proposal construction. On pressure retry only the
            // known anchor; the scheduler can preempt if even that cannot fit.
            let reserve = self
                .inner
                .reserve_scheduled_speculation(row.seq_id, cap + 1)
                .and_then(|reserved| {
                    if reserved {
                        Ok(true)
                    } else {
                        cap = 0;
                        self.inner.reserve_scheduled_speculation(row.seq_id, 1)
                    }
                });
            match reserve {
                Ok(true) => {}
                Ok(false) => {
                    results.push((plan_index, Self::blocked(row)));
                    continue;
                }
                Err(error) => {
                    results.push((plan_index, Self::speculative_failure(turn, row, error)));
                    continue;
                }
            }
            let proposal = if cap > 0 {
                let rng = turn.payload.scheduled_speculation.as_mut().unwrap();
                self.inner.propose_scheduled(
                    row.seq_id,
                    anchor,
                    cap,
                    &turn.payload.params,
                    rng,
                    false,
                )
            } else {
                Ok(empty_proposal())
            };
            let proposal = match proposal {
                Ok(proposal)
                    if proposal.draft_ids.len() <= cap
                        && proposal.draft_ids.iter().all(|&token| token >= 0) =>
                {
                    proposal
                }
                Ok(_) => {
                    results.push((
                        plan_index,
                        Self::speculative_failure(
                            turn,
                            row,
                            Error::from_reason("invalid scheduled draft proposal"),
                        ),
                    ));
                    continue;
                }
                Err(error) => {
                    results.push((plan_index, Self::speculative_failure(turn, row, error)));
                    continue;
                }
            };
            let mut tokens = Vec::with_capacity(proposal.draft_ids.len() + 1);
            tokens.push(anchor);
            tokens.extend(proposal.draft_ids.iter().map(|&token| token as u32));
            verify_rows.push(ScheduledVerifyRow {
                seq_id: row.seq_id,
                first_position: row.token_start,
                tokens,
                speculative: turn.payload.scheduled_speculation.is_some(),
            });
            work.push(VerifyWork {
                plan_index,
                turn_index,
                proposal,
                forced,
            });
        }
        if work.is_empty() {
            return Ok((results, 0, 0));
        }
        crate::array::maybe_clear_cache_for_paged_step(plan.global_step as i32);
        let logits = {
            let _stream_context = StreamContext::new(Stream::default(DeviceType::Gpu));
            self.inner.run_scheduled_verify(&verify_rows)
        };
        let logits = match logits {
            Ok(logits) => logits,
            Err(error) => {
                for item in work {
                    let turn = &mut running[item.turn_index];
                    results.push((
                        item.plan_index,
                        Self::speculative_failure(
                            turn,
                            &plan.rows[item.plan_index],
                            Error::from_reason(error.reason.clone()),
                        ),
                    ));
                }
                return Ok((results, 0, 0));
            }
        };
        let greedy = work.iter().all(|item| {
            let params = &running[item.turn_index].payload.params;
            item.forced.is_none()
                && crate::sampling::is_greedy_temperature(
                    params
                        .sampling_config
                        .and_then(|config| config.temperature)
                        .unwrap_or(1.0),
                )
                && params.repetition_penalty == 1.0
                && params.presence_penalty == 0.0
                && params.frequency_penalty == 0.0
        });
        let greedy_tokens = if greedy {
            logits
                .argmax(-1, None)
                .and_then(|tokens| {
                    tokens.eval();
                    (0..verify_rows.iter().map(|row| row.tokens.len()).sum())
                        .map(|i| tokens.item_at_int32(i))
                        .collect::<Result<Vec<_>>>()
                })
                .ok()
        } else {
            None
        };
        let greedy_occupancy = if greedy_tokens.is_some() {
            work.len()
        } else {
            0
        };
        let mut decisions = Vec::with_capacity(work.len());
        let mut commits = Vec::with_capacity(work.len());
        let mut offset = 0i64;
        for (item, verify) in work.iter().zip(&verify_rows) {
            let turn = &mut running[item.turn_index];
            let end = offset + verify.tokens.len() as i64;
            let accepted = logits
                .slice_axis(0, offset, end)
                .and_then(|row| row.transpose(Some(&[1, 0, 2])))
                .and_then(|row| {
                    if let Some(tokens) = greedy_tokens.as_ref() {
                        let target = &tokens[offset as usize..end as usize];
                        let mut count = 0;
                        while count < item.proposal.draft_ids.len()
                            && target[count] == item.proposal.draft_ids[count]
                        {
                            count += 1;
                        }
                        Ok((count, target[count] as u32))
                    } else if let Some(forced) = item.forced {
                        row.eval(); // forcing a host id must still complete target/cache writes
                        Ok((0, forced))
                    } else if let Some(rng) = turn.payload.scheduled_speculation.as_mut() {
                        accept_dspark_proposal(
                            &row,
                            &item.proposal,
                            &turn.token_history,
                            &turn.payload.params,
                            rng,
                        )
                    } else {
                        accept_dspark_proposal(
                            &row,
                            &item.proposal,
                            &turn.token_history,
                            &turn.payload.params,
                            &mut rand::rng(),
                        )
                    }
                });
            offset = end;
            let decision = accepted.map(|(count, boundary)| {
                let mut tokens = item.proposal.draft_ids[..count]
                    .iter()
                    .map(|&token| token as u32)
                    .collect::<Vec<_>>();
                tokens.push(boundary);
                let cancelled = turn.cancelled.as_ref();
                let observer = &mut turn.payload.turn_token_observer;
                let clamp = clamp_dspark_cycle(
                    &turn.payload.generated_tokens,
                    &tokens,
                    count,
                    &turn.payload.params,
                    turn.payload.params.max_new_tokens.max(0) as usize,
                    turn.payload.eos_id,
                    || cancelled.is_some_and(|flag| flag.load(Ordering::Relaxed)),
                    |token| {
                        observer
                            .as_mut()
                            .is_some_and(|observer| observer.observe_token_id(token))
                    },
                );
                tokens.truncate(clamp.emit_count);
                (tokens, clamp, count)
            });
            commits.push(ScheduledVerifyCommit {
                seq_id: verify.seq_id,
                keep: decision.as_ref().map_or(0, |(_, clamp, _)| clamp.keep),
            });
            decisions.push(decision);
        }
        // This closes every ticket, including failed acceptance owners with
        // keep=0. Publication and owner reuse happen only after all commits.
        let committed = self.inner.commit_scheduled_verify(&commits);
        let committed = match committed {
            Ok(results) if results.len() == work.len() => results,
            outcome => {
                let reason = outcome.err().map_or_else(
                    || "scheduled commit result width mismatch".into(),
                    |error| error.reason,
                );
                work.iter()
                    .map(|_| Err(Error::from_reason(reason.clone())))
                    .collect()
            }
        };
        let occupancy = work.len();
        for (((item, verify), decision), commit) in work
            .into_iter()
            .zip(verify_rows)
            .zip(decisions)
            .zip(committed)
        {
            let turn = &mut running[item.turn_index];
            let planned = &plan.rows[item.plan_index];
            let (tokens, clamp, accepted_count) =
                match decision.and_then(|decision| commit.map(|()| decision)) {
                    Ok(decision) => decision,
                    Err(error) => {
                        self.inner.release_scheduled_speculation(turn.seq_id);
                        results.push((
                            item.plan_index,
                            Self::speculative_failure(turn, planned, error),
                        ));
                        continue;
                    }
                };
            if item.forced.is_some() {
                turn.payload.reasoning_tracker.should_force_think_end();
            }
            if !item.proposal.draft_ids.is_empty() {
                turn.payload
                    .profiler
                    .record_mtp_cycle(item.proposal.draft_ids.len(), accepted_count);
            }
            for (index, &token) in tokens.iter().enumerate() {
                turn.payload.generated_tokens.push(token);
                turn.payload.profiler.step();
                let reasoning = turn.payload.reasoning_tracker.observe_token(token);
                turn.payload.last_is_reasoning = reasoning;
                turn.payload.profiler.mark_first_token();
                let cancel_token =
                    clamp.stop == Some(CycleStop::Cancelled) && index + 1 == tokens.len();
                let eos =
                    token == turn.payload.eos_id || turn.payload.extra_eos_ids.contains(&token);
                if !cancel_token
                    && !(eos && !B::STREAM_EOS_TOKEN)
                    && turn.payload.response.sink().is_some()
                {
                    Self::stream_token(turn, token, reasoning);
                }
            }
            let finished = clamp.stop.is_some()
                || turn.payload.generated_tokens.len()
                    >= turn.payload.params.max_new_tokens.max(0) as usize;
            if let Some(stop) = clamp.stop {
                turn.payload.finish_reason = match stop {
                    CycleStop::Length => "length",
                    CycleStop::Stop => "stop",
                    CycleStop::Cancelled => "cancelled",
                    CycleStop::Repetition(reason) => reason,
                }
                .into();
            }
            if finished {
                turn.payload.profiler.snapshot_memory_after();
                turn.payload.profiler.report();
            }
            results.push((
                item.plan_index,
                RowStepResult {
                    seq_id: turn.seq_id,
                    num_computed_tokens: clamp.keep as u32,
                    generated_token: None,
                    speculative: Some(SpeculativeRowResult {
                        verified_tokens: verify.tokens.len() as u32,
                        generated_tokens: tokens,
                    }),
                    finished,
                    allocation_blocked: false,
                    prefill_micros: 0,
                },
            ));
        }
        Ok((results, occupancy, greedy_occupancy))
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use crate::models::gemma4::dspark_decode::tests::{
        seeded_tiny_paged_inner_with_draft, tiny_qwen_tokenizer, tiny_turn_config,
    };
    use rand::SeedableRng;

    fn request<B: HybridSchedulerBackend>(
        inner: &mut B,
        seq: u32,
        prompt: Vec<u32>,
        speculative: bool,
        cancel: Arc<AtomicBool>,
    ) -> TurnState<ScheduledTurn<B::PrefixState>> {
        let config = tiny_turn_config(Some(3), 20);
        let params = inner.resolve_params(&config);
        let thinking = inner.thinking_setup(&config);
        let prefix = match inner
            .prepare_scheduled_prefix(seq, &prompt, &[], false, 0, 8)
            .unwrap()
        {
            ScheduledPrefixAdmission::Ready(prefix) => prefix,
            _ => unreachable!(),
        };
        let (reply, _) = tokio::sync::oneshot::channel();
        let payload = ScheduledTurn {
            owner_id: format!("owner-{seq}"),
            scheduled_speculation: speculative
                .then(|| rand::rngs::StdRng::seed_from_u64(u64::from(seq))),
            pending_token_emitted: false,
            tokenizer: tiny_qwen_tokenizer(),
            eos_id: 999,
            config,
            params,
            thinking,
            prompt_tokens: prompt.clone(),
            prefix,
            is_delta: false,
            reuse_cache: false,
            response: ScheduledReply::Sync(reply),
            generated_tokens: Vec::new(),
            finish_reason: "length".into(),
            reasoning_tracker: engine::penalties::ReasoningTracker::from_setup(&thinking, None),
            extra_eos_ids: Vec::new(),
            generation_start: None,
            first_token_instant: None,
            generation_stream: Stream::default(DeviceType::Gpu),
            profiler: DecodeProfiler::new("scheduled-test", "Gemma4"),
            emitter: None,
            turn_token_observer: None,
            stream_skip_special: false,
            decode_ids: Vec::new(),
            decode_prefix: String::new(),
            decode_prefix_index: 0,
            streamed_text_len: 0,
            last_is_reasoning: false,
            failure: None,
            allocation_failed: false,
            preemption_replay: None,
        };
        let mut turn = TurnState::new(seq, prompt, 0, Vec::new(), Some(cancel), payload).unwrap();
        turn.decode_draft_allowance = if speculative { 3 } else { 0 };
        turn
    }

    fn run(speculate: bool, cancel_first: bool) -> (Vec<(u32, Vec<u32>)>, usize) {
        run_with(
            seeded_tiny_paged_inner_with_draft(1729).unwrap(),
            speculate,
            cancel_first,
        )
    }

    fn run_with<B: HybridSchedulerBackend>(
        inner: B,
        speculate: bool,
        cancel_first: bool,
    ) -> (Vec<(u32, Vec<u32>)>, usize) {
        run_with_sampling(inner, speculate, cancel_first, false)
    }

    fn run_with_sampling<B: HybridSchedulerBackend>(
        mut inner: B,
        speculate: bool,
        cancel_first: bool,
        sampled: bool,
    ) -> (Vec<(u32, Vec<u32>)>, usize) {
        let mut scheduler = Scheduler::<_, (), ()>::new(2, 12).unwrap();
        let cancel = Arc::new(AtomicBool::new(false));
        let mut first = request(
            &mut inner,
            41,
            vec![0, 1, 2, 3, 4, 5, 6],
            speculate,
            Arc::clone(&cancel),
        );
        let mut second = request(
            &mut inner,
            42,
            vec![4, 5, 6, 7, 8, 9, 10, 11, 12],
            speculate,
            Arc::new(AtomicBool::new(false)),
        );
        if sampled {
            for turn in [&mut first, &mut second] {
                turn.payload
                    .params
                    .sampling_config
                    .as_mut()
                    .unwrap()
                    .temperature = Some(0.8);
                turn.payload.params.repetition_penalty = 1.1;
                turn.payload.params.presence_penalty = 0.2;
                turn.payload.params.frequency_penalty = 0.1;
            }
        }
        scheduler.enqueue_turn(first).unwrap();
        scheduler.enqueue_turn(second).unwrap();
        let mut completed = Vec::new();
        for step in 0..100 {
            if cancel_first && step == 4 {
                cancel.store(true, Ordering::Relaxed);
            }
            let action = scheduler
                .drive_once(&mut HybridStepExecutor::new(&mut inner))
                .unwrap();
            if let SchedulerAction::Stepped {
                completed: turns, ..
            } = action
            {
                for turn in turns {
                    assert!(
                        turn.payload.failure.is_none(),
                        "owner {} failed: {:?}",
                        turn.seq_id,
                        turn.payload.failure
                    );
                    let generated = turn.payload.generated_tokens;
                    let mut expected = turn.payload.prompt_tokens;
                    expected.extend_from_slice(&generated[..generated.len().saturating_sub(1)]);
                    assert_eq!(
                        inner
                            .paged_adapter()
                            .unwrap()
                            .request_tokens_for(turn.seq_id)
                            .unwrap(),
                        expected,
                        "owner {} committed a rejected/final token",
                        turn.seq_id
                    );
                    inner.release_scheduled_speculation(turn.seq_id);
                    completed.push((turn.seq_id, generated));
                }
            }
            if completed.len() == 2 {
                break;
            }
        }
        assert_eq!(completed.len(), 2);
        completed.sort_by_key(|(seq, _)| *seq);
        (completed, scheduler.stats().max_batch_occupancy)
    }

    #[test]
    fn concurrent_dflash_matches_ar_tokens_and_isolates_cancelled_owner() {
        use crate::models::muse_glimmer::model::scheduled_dflash::tests::seeded_inner;
        let (ar, _) = run_with(seeded_inner(1908).unwrap(), false, false);
        let (spec, occupancy) = run_with(seeded_inner(1908).unwrap(), true, false);
        assert_eq!(ar, spec);
        assert_eq!(occupancy, 2);
        let (cancelled, _) = run_with(seeded_inner(1908).unwrap(), true, true);
        assert_eq!(cancelled[1], spec[1]);
        assert!(cancelled[0].1.len() < spec[0].1.len());
    }

    #[test]
    fn concurrent_dspark_matches_ar_tokens_and_isolates_cancelled_owner() {
        let (ar, _) = run(false, false);
        let (spec, occupancy) = run(true, false);
        assert_eq!(ar, spec, "packed speculative output must match greedy AR");
        assert_eq!(
            occupancy, 2,
            "requests must share real target forward waves"
        );
        let (cancelled, _) = run(true, true);
        assert_eq!(
            cancelled[1], spec[1],
            "cancellation changed the other owner"
        );
        assert!(cancelled[0].1.len() < spec[0].1.len());
    }

    #[test]
    fn invalid_speculative_prefill_mass_fails_before_recording_a_sentinel_token() {
        let mut inner = seeded_tiny_paged_inner_with_draft(1729).unwrap();
        let poisoned = inner
            .embed_tokens
            .get_weight()
            .mul_scalar(f64::NAN)
            .unwrap();
        inner.lm_head = Some(
            crate::models::gemma4::quantized_linear::LinearProj::Standard(
                crate::nn::Linear::from_weights(&poisoned, None).unwrap(),
            ),
        );
        let mut turn = request(
            &mut inner,
            41,
            vec![0, 1, 2, 3, 4, 5, 6],
            true,
            Arc::new(AtomicBool::new(false)),
        );
        turn.payload
            .params
            .sampling_config
            .as_mut()
            .unwrap()
            .temperature = Some(0.8);
        let mut scheduler = Scheduler::<_, (), ()>::new(2, 12).unwrap();
        scheduler.enqueue_turn(turn).unwrap();
        let SchedulerAction::Stepped { completed, .. } = scheduler
            .drive_once(&mut HybridStepExecutor::new(&mut inner))
            .unwrap()
        else {
            panic!("expected failed prefill step");
        };
        assert_eq!(completed.len(), 1);
        let turn = &completed[0];
        assert!(
            turn.payload
                .failure
                .as_ref()
                .unwrap()
                .reason
                .contains("valid probability mass")
        );
        assert!(turn.payload.generated_tokens.is_empty());
        assert!(!turn.token_history.contains(&u32::MAX));
        assert!(
            !inner
                .paged_adapter()
                .unwrap()
                .request_tokens_for(41)
                .unwrap()
                .contains(&u32::MAX)
        );
    }

    #[test]
    fn sampled_dspark_and_dflash_keep_per_owner_rng_and_committed_history() {
        fn check<B: HybridSchedulerBackend>(build: impl Fn() -> B) {
            let (complete, occupancy) = run_with_sampling(build(), true, false, true);
            assert_eq!(occupancy, 2);
            let (cancelled, _) = run_with_sampling(build(), true, true, true);
            assert_eq!(
                cancelled[1], complete[1],
                "cancellation changed the peer's sampled trajectory"
            );
            assert!(cancelled[0].1.len() < complete[0].1.len());
        }
        check(|| seeded_tiny_paged_inner_with_draft(1729).unwrap());
        check(|| {
            crate::models::muse_glimmer::model::scheduled_dflash::tests::seeded_inner(1908).unwrap()
        });
    }
}
