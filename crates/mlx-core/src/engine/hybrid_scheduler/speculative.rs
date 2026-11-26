//! One packed verifier wave, followed by independent owner commits and emission.
use super::*;
#[cfg(test)]
use engine::backend::ThinkingSetup;
use engine::backend::{DsparkProposal, ThinkEndResolution};
use engine::dspark_timeline::{self as timeline, Mark};
use engine::dspark_turn::{
    CycleStop, SampledRows, accept_dspark_proposal, clamp_dspark_cycle, submit_sampled_rows,
    thinking_budget_wave_limit,
};
use engine::penalties::ThinkingCut;
#[cfg(test)]
use engine::penalties::{ReasoningClose, ReasoningTracker};
use engine::scheduler::SpeculativeRowResult;

struct VerifyWork {
    plan_index: usize,
    turn_index: usize,
    anchor: u32,
    cap: usize,
    proposal: DsparkProposal,
    cut: Option<ThinkingCut>,
    think_close: ThinkEndResolution,
}

#[cfg(test)]
thread_local! {
    static TEST_THINK_END_OVERRIDE: std::cell::RefCell<Option<ThinkEndResolution>> =
        const { std::cell::RefCell::new(None) };
    static TEST_DRAFT_OVERRIDE: std::cell::RefCell<
        Option<Box<dyn FnMut(u32, u32, usize) -> Option<Vec<i32>>>>,
    > = const { std::cell::RefCell::new(None) };
    static TEST_POST_BATCH_PROPOSAL_HOOK: std::cell::RefCell<
        Option<Box<dyn FnMut(&[u32], &mut [Result<DsparkProposal>])>>,
    > = const { std::cell::RefCell::new(None) };
    static TEST_SOLO_PROPOSAL_HOOK: std::cell::RefCell<
        Option<Box<dyn FnMut(u32, Result<DsparkProposal>) -> Result<DsparkProposal>>>,
    > = const { std::cell::RefCell::new(None) };
    static TEST_DRAFT_EVAL_OVERRIDE: std::cell::RefCell<
        Option<Box<dyn FnMut(&[&MxArray], &'static str) -> Option<Result<()>>>>,
    > = const { std::cell::RefCell::new(None) };
}

fn collect_proposal_device_arrays<'a>(prop: &'a DsparkProposal, out: &mut Vec<&'a MxArray>) {
    if let Some(ref ids) = prop.device_draft_ids {
        out.push(ids);
    }
    if let Some(ref sparse) = prop.device_draft_sparse {
        out.push(&sparse.ids);
        out.push(&sparse.probs);
    }
}

fn eval_scheduled_draft_arrays(arrays: &[&MxArray], context: &'static str) -> Result<()> {
    #[cfg(test)]
    if let Some(override_res) = TEST_DRAFT_EVAL_OVERRIDE
        .with(|slot| slot.borrow_mut().as_mut().and_then(|f| f(arrays, context)))
    {
        return override_res;
    }
    MxArray::eval_arrays_with_context(arrays, context)
}

fn empty_proposal() -> DsparkProposal {
    DsparkProposal {
        draft_ids: Vec::new(),
        device_draft_ids: None,
        device_draft_sparse: None,
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

    /// Target rows for `queries` and the draft state the step grows
    /// outside the pool. A refusal blocks the row, so the scheduler
    /// reclaims or preempts instead of the draft caches outgrowing memory.
    /// A family on the ledger had the growth admitted by the row's step
    /// claim.
    fn reserve_speculative_step(
        &mut self,
        running: &[Turn<B>],
        turn_index: usize,
        queries: usize,
    ) -> Result<bool> {
        let seq_id = running[turn_index].seq_id;
        if !self.inner.reserve_scheduled_speculation(seq_id, queries)? {
            return Ok(false);
        }
        if self.inner.reclaims_in_order() {
            return Ok(true);
        }
        let own = &running[turn_index];
        let others: Vec<&Turn<B>> = running
            .iter()
            .filter(|turn| turn.seq_id != own.seq_id)
            .collect();
        let queries = u32::try_from(queries).unwrap_or(u32::MAX);
        Ok(draft_shortfall(&*self.inner, own, &others, queries)? == 0)
    }

    pub(super) fn execute_speculative_rows(
        &mut self,
        plan: &StepPlan,
        running: &mut [TurnState<ScheduledTurn<B::PrefixState>>],
    ) -> Result<(Vec<(usize, RowStepResult)>, usize, usize)> {
        let mut results = Vec::new();
        let mut work = Vec::new();
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
            timeline::lane_cycle_start(row.seq_id, turn.token_history.len());
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
                let is_reasoning = turn.payload.reasoning_tracker.observe_token(anchor);
                Self::finish_decode_row(turn, &seed, is_reasoning);
                turn.payload.pending_token_emitted = true;
                if seed.terminal || seed.at_length {
                    turn.payload.profiler.snapshot_memory_after();
                    turn.payload.profiler.report();
                    timeline::end_lane(row.seq_id);
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
                            prefill_in_flight: false,
                        },
                    ));
                    continue;
                }
            }
            if row.cancel_snapshot {
                turn.payload.finish_reason = "cancelled".into();
                turn.payload.profiler.snapshot_memory_after();
                turn.payload.profiler.report();
                timeline::end_lane(row.seq_id);
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
                        prefill_in_flight: false,
                    },
                ));
                continue;
            }
            timeline::lane_mark(row.seq_id, Mark::StopChecks);
            let max_new = turn.payload.params.max_new_tokens.max(0) as usize;
            let remaining = max_new.saturating_sub(turn.payload.generated_tokens.len());
            let mut cap = if turn.payload.scheduled_speculation.is_some() {
                (row.num_tokens as usize).saturating_sub(1)
            } else {
                0
            };
            cap = cap.min(remaining.saturating_sub(1));
            // A thinking cut is a row of this verify: the row keeps its cap.
            let cut = turn.payload.reasoning_tracker.thinking_cut();
            let think_close = {
                #[cfg(test)]
                {
                    TEST_THINK_END_OVERRIDE
                        .with(|c| c.borrow().clone())
                        .unwrap_or_else(|| {
                            self.inner
                                .think_end_for_turn(&turn.payload.config, &turn.payload.tokenizer)
                        })
                }
                #[cfg(not(test))]
                {
                    self.inner
                        .think_end_for_turn(&turn.payload.config, &turn.payload.tokenizer)
                }
            };
            // Reserve before proposal construction. On pressure retry only the
            // known anchor; the scheduler can preempt if even that cannot fit.
            let reserve = self
                .reserve_speculative_step(running, turn_index, cap + 1)
                .and_then(|reserved| {
                    if reserved {
                        Ok(true)
                    } else {
                        cap = 0;
                        self.reserve_speculative_step(running, turn_index, 1)
                    }
                });
            match reserve {
                Ok(true) => timeline::lane_mark(row.seq_id, Mark::Plan),
                Ok(false) => {
                    results.push((plan_index, Self::blocked(row)));
                    continue;
                }
                Err(error) => {
                    results.push((
                        plan_index,
                        Self::speculative_failure(&mut running[turn_index], row, error),
                    ));
                    continue;
                }
            }
            work.push(VerifyWork {
                plan_index,
                turn_index,
                anchor,
                cap,
                proposal: empty_proposal(),
                cut,
                think_close,
            });
        }
        if work.is_empty() {
            return Ok((results, 0, 0));
        }
        // A mixed fixed/adaptive speculative wave keeps the fixed policy.
        // A speculative owner can temporarily have cap zero under pressure or
        // forcing; that does not turn it into an ordinary AR peer.
        let eligible = work.iter().any(|item| item.cap > 0)
            && work.iter().all(|item| {
                let params = &running[item.turn_index].payload.params;
                running[item.turn_index]
                    .payload
                    .scheduled_speculation
                    .is_none()
                    || (params.mtp_adaptive_depth
                        && crate::sampling::is_greedy_temperature(
                            params
                                .sampling_config
                                .and_then(|c| c.temperature)
                                .unwrap_or(1.0),
                        ))
            });
        let profile_rows = work
            .iter()
            .map(|item| {
                let row = &plan.rows[item.plan_index];
                engine::verification_budget::ScheduledBudgetRow {
                    seq_id: row.seq_id,
                    context_band: row.token_start / 2048,
                    cap: item.cap,
                }
            })
            .collect::<Vec<_>>();
        let mut profiling = false;
        let probe = self
            .inner
            .scheduled_verification_budget()
            .and_then(|policy| {
                if eligible {
                    policy.configure(&profile_rows);
                    profiling = true;
                    policy.next_probe()
                } else {
                    policy.clear();
                    None
                }
            });
        let seq_ids = profile_rows
            .iter()
            .map(|row| row.seq_id)
            .collect::<Vec<_>>();
        if profiling {
            self.inner.complete_scheduled_draft_state(&seq_ids)?;
        }
        let wave_started = profiling.then(Instant::now);
        let zero_probe = probe
            .as_ref()
            .is_some_and(|shape| shape.iter().all(|&n| n == 0));
        let mut proposals: Vec<Result<DsparkProposal>> =
            (0..work.len()).map(|_| Ok(empty_proposal())).collect();
        let mut eligible_work_indices = Vec::new();
        for (wi, item) in work.iter().enumerate() {
            if item.cap > 0 && !zero_probe {
                if running[item.turn_index]
                    .payload
                    .scheduled_speculation
                    .is_some()
                {
                    eligible_work_indices.push(wi);
                } else {
                    proposals[wi] = Err(Error::from_reason(
                        "scheduled speculation state missing for capped row",
                    ));
                    timeline::lane_mark(plan.rows[item.plan_index].seq_id, Mark::DraftSubmit);
                }
            }
        }
        if !eligible_work_indices.is_empty() {
            let mut by_turn: HashMap<usize, &mut TurnState<ScheduledTurn<B::PrefixState>>> =
                running.iter_mut().enumerate().collect();
            let mut batch_reqs = Vec::with_capacity(eligible_work_indices.len());
            for &wi in &eligible_work_indices {
                let item = &work[wi];
                let row = &plan.rows[item.plan_index];
                let turn = by_turn
                    .remove(&item.turn_index)
                    .ok_or_else(|| Error::from_reason("scheduled verifier duplicate turn"))?;
                let payload = &mut turn.payload;
                let rng = payload.scheduled_speculation.as_mut().ok_or_else(|| {
                    Error::from_reason("scheduled speculation state missing for capped row")
                })?;
                batch_reqs.push(ScheduledProposalRequest {
                    seq_id: row.seq_id,
                    anchor: item.anchor,
                    max_drafts: item.cap,
                    params: &payload.params,
                    rng,
                    confidence: profiling,
                });
            }
            let batch_out = if batch_reqs.len() == 1 {
                let _focus = timeline::focus_lane(batch_reqs[0].seq_id);
                self.inner.propose_scheduled_batch(&mut batch_reqs)
            } else {
                self.inner.propose_scheduled_batch(&mut batch_reqs)
            };
            let mut batch_iter = batch_out.into_iter();
            for &wi in &eligible_work_indices {
                timeline::lane_mark(plan.rows[work[wi].plan_index].seq_id, Mark::DraftSubmit);
                proposals[wi] = batch_iter.next().unwrap_or_else(|| {
                    Err(Error::from_reason("scheduled proposal width mismatch"))
                });
            }
        }
        #[cfg(test)]
        {
            let wave_seqs: Vec<u32> = work
                .iter()
                .map(|item| plan.rows[item.plan_index].seq_id)
                .collect();
            TEST_POST_BATCH_PROPOSAL_HOOK.with(|slot| {
                if let Some(f) = slot.borrow_mut().as_mut() {
                    f(&wave_seqs, &mut proposals);
                }
            });
        }
        let mut proposed = Vec::with_capacity(work.len());
        // Materialize ALL device-resident draft proposals in ONE single GPU sync instead of N.
        // If that batched eval fails, NEVER re-evaluate outputs from the failed Metal command
        // buffer (MLX may already have marked those arrays evaluated): preserve the original
        // error, discard all affected batch handles, restore each eligible lane's committed
        // draft-ring handles, and rebuild fresh solo proposals so healthy peers survive cleanly.
        let batch_eval_err = {
            let mut eval_refs: Vec<&MxArray> = Vec::new();
            for prop in proposals.iter().flatten() {
                collect_proposal_device_arrays(prop, &mut eval_refs);
            }
            if eval_refs.is_empty() {
                None
            } else {
                eval_scheduled_draft_arrays(&eval_refs, "scheduled_draft_proposals").err()
            }
        };
        if let Some(batch_err) = batch_eval_err {
            let mut solo_retry_indices = Vec::new();
            for (wi, res) in proposals.iter_mut().enumerate() {
                let has_device_arrays = res.as_ref().is_ok_and(|prop| {
                    let mut lane_refs: Vec<&MxArray> = Vec::new();
                    collect_proposal_device_arrays(prop, &mut lane_refs);
                    !lane_refs.is_empty()
                });
                if !has_device_arrays {
                    continue;
                }
                *res = Err(Error::from_reason(batch_err.reason.clone()));
                if eligible_work_indices.len() > 1 && eligible_work_indices.contains(&wi) {
                    solo_retry_indices.push(wi);
                }
            }
            for wi in solo_retry_indices {
                let item = &work[wi];
                let row = &plan.rows[item.plan_index];
                let payload = &mut running[item.turn_index].payload;
                let Some(rng) = payload.scheduled_speculation.as_mut() else {
                    continue;
                };
                self.inner.restore_scheduled_draft_handles(row.seq_id);
                let solo = self.inner.propose_scheduled(
                    row.seq_id,
                    item.anchor,
                    item.cap,
                    &payload.params,
                    rng,
                    profiling,
                );
                #[cfg(test)]
                let solo = TEST_SOLO_PROPOSAL_HOOK.with(|slot| {
                    if let Some(f) = slot.borrow_mut().as_mut() {
                        f(row.seq_id, solo)
                    } else {
                        solo
                    }
                });
                let solo = solo.and_then(|prop| {
                    let mut lane_refs: Vec<&MxArray> = Vec::new();
                    collect_proposal_device_arrays(&prop, &mut lane_refs);
                    if !lane_refs.is_empty() {
                        eval_scheduled_draft_arrays(
                            &lane_refs,
                            "scheduled_draft_proposal_solo_retry",
                        )?;
                    }
                    Ok(prop)
                });
                proposals[wi] = solo;
            }
        }
        for (mut item, proposal) in work.into_iter().zip(proposals) {
            let turn = &mut running[item.turn_index];
            let row = &plan.rows[item.plan_index];
            // The scheduled verify wave is host-batched (`ScheduledVerifyRow`
            // tokens), so a device-resident proposal is normalized to host
            // ids here at the boundary rather than carried deeper.
            match proposal.and_then(|mut proposal| {
                proposal.materialize_draft_ids()?;
                Ok(proposal)
            }) {
                Ok(proposal)
                    if proposal
                        .draft_len()
                        .map(|len| len <= item.cap)
                        .unwrap_or(false)
                        && proposal.draft_ids.iter().all(|&token| token >= 0) =>
                {
                    item.proposal = proposal;
                    #[cfg(test)]
                    if let Some(override_ids) = TEST_DRAFT_OVERRIDE.with(|slot| {
                        slot.borrow_mut()
                            .as_mut()
                            .and_then(|f| f(row.seq_id, item.anchor, item.cap))
                    }) {
                        let len = override_ids.len().min(item.cap);
                        item.proposal.draft_ids = override_ids[..len].to_vec();
                        item.proposal.draft_dists.clear();
                        item.proposal.draft_sparse_dists.clear();
                        item.proposal.keep_probabilities = None;
                    }
                    // At row 0 the end tokens are the first proposals; their
                    // keep probability is 1 before the wave's budget reads it.
                    let replaced = match &item.cut {
                        Some(cut) => {
                            let len = item.proposal.draft_ids.len();
                            item.proposal.replace_prefix(cut.proposals(len))
                        }
                        None => Ok(()),
                    };
                    match replaced {
                        Ok(()) => proposed.push(item),
                        Err(error) => {
                            self.inner.release_scheduled_speculation(turn.seq_id);
                            results.push((
                                item.plan_index,
                                Self::speculative_failure(turn, row, error),
                            ));
                        }
                    }
                }
                outcome => {
                    let error = outcome
                        .err()
                        .unwrap_or_else(|| Error::from_reason("invalid scheduled draft proposal"));
                    self.inner.release_scheduled_speculation(turn.seq_id);
                    results.push((item.plan_index, Self::speculative_failure(turn, row, error)));
                }
            }
        }
        let mut work = proposed;
        if profiling
            && (work.len() != profile_rows.len()
                || (!zero_probe
                    && work.iter().any(|item| {
                        item.proposal
                            .draft_len()
                            .map(|len| len != item.cap)
                            .unwrap_or(true)
                    })))
        {
            profiling = false;
            if let Some(budget) = self.inner.scheduled_verification_budget() {
                budget.clear();
            }
        }
        if work.is_empty() {
            return Ok((results, 0, 0));
        }
        let choice = if profiling {
            probe.map(|shape| (shape, true)).or_else(|| {
                let confidence = work
                    .iter()
                    .map(|item| item.proposal.keep_probabilities.as_deref().unwrap_or(&[]))
                    .collect::<Vec<_>>();
                self.inner
                    .scheduled_verification_budget()
                    .and_then(|budget| budget.choose(&confidence))
            })
        } else {
            None
        };
        let fallback = choice
            .as_ref()
            .is_some_and(|(shape, calibration)| !calibration && shape.iter().all(|&n| n == 0));
        if let Some((shape, _)) = choice {
            for (item, keep) in work.iter_mut().zip(shape) {
                item.proposal.truncate(keep);
                if fallback {
                    let turn = &mut running[item.turn_index];
                    self.inner.release_scheduled_speculation(turn.seq_id);
                    turn.payload.scheduled_speculation = None;
                    turn.decode_draft_allowance = 0;
                }
            }
        }
        let mut seen_verify_seqs = std::collections::HashSet::with_capacity(work.len());
        let mut valid_work = Vec::with_capacity(work.len());
        let mut valid_verify_rows = Vec::with_capacity(work.len());
        for item in work {
            let row = &plan.rows[item.plan_index];
            timeline::lane_mark(row.seq_id, Mark::DraftPost);
            let mut tokens = Vec::with_capacity(item.proposal.draft_ids.len() + 1);
            tokens.push(item.anchor);
            tokens.extend(item.proposal.draft_ids.iter().map(|&token| token as u32));
            let verify_row = ScheduledVerifyRow {
                seq_id: row.seq_id,
                first_position: row.token_start,
                tokens,
                speculative: running[item.turn_index]
                    .payload
                    .scheduled_speculation
                    .is_some(),
            };
            let valid = if !seen_verify_seqs.insert(verify_row.seq_id) {
                Err(Error::from_reason(
                    "duplicate or empty scheduled verifier row",
                ))
            } else {
                self.inner.validate_scheduled_verify_row(&verify_row)
            };
            match valid {
                Ok(()) => {
                    valid_work.push(item);
                    valid_verify_rows.push(verify_row);
                }
                Err(error) => {
                    let turn = &mut running[item.turn_index];
                    results.push((item.plan_index, Self::speculative_failure(turn, row, error)));
                }
            }
        }
        let mut work = valid_work;
        let verify_rows = valid_verify_rows;
        if profiling && work.len() != profile_rows.len() {
            profiling = false;
            if let Some(budget) = self.inner.scheduled_verification_budget() {
                budget.clear();
            }
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
            crate::sampling::is_greedy_temperature(
                params
                    .sampling_config
                    .and_then(|config| config.temperature)
                    .unwrap_or(1.0),
            ) && params.repetition_penalty == 1.0
                && params.presence_penalty == 0.0
                && params.frequency_penalty == 0.0
        });
        // A failed eval or read fails every row: MLX throws a failed GPU
        // command's error once, so a per-row fallback would read what it wrote.
        let greedy_tokens = if greedy {
            logits.argmax(-1, None).ok().map(|tokens| {
                MxArray::eval_arrays_with_context(&[&tokens], "scheduled_verify_greedy")?;
                for row in &verify_rows {
                    timeline::lane_mark(row.seq_id, Mark::VerifyWait);
                }
                (0..verify_rows.iter().map(|row| row.tokens.len()).sum())
                    .map(|i| tokens.item_at_int32(i))
                    .collect::<Result<Vec<_>>>()
            })
        } else {
            None
        };
        let greedy_occupancy = if matches!(greedy_tokens, Some(Ok(_))) {
            work.len()
        } else {
            0
        };
        // The wave's sampled lanes queue their target rows, each with its
        // own rng, and one eval forces them all.
        let mut sampled: Vec<Option<Result<SampledRows>>> = work.iter().map(|_| None).collect();
        let rows_route = self.inner.scheduled_sampled_rows();
        if greedy_tokens.is_none() && rows_route {
            let mut offset = 0i64;
            for ((item, verify), slot) in work.iter().zip(&verify_rows).zip(&mut sampled) {
                let end = offset + verify.tokens.len() as i64;
                let payload = &mut running[item.turn_index].payload;
                let rng = payload.scheduled_speculation.as_mut();
                if let Some(rng) = rng {
                    *slot = logits
                        .slice_axis(0, offset, end)
                        .and_then(|rows| {
                            submit_sampled_rows(&rows, &item.proposal, &payload.params, rng)
                        })
                        .transpose();
                }
                offset = end;
            }
            let queued = sampled
                .iter()
                .filter_map(|slot| slot.as_ref()?.as_ref().ok())
                .map(SampledRows::records)
                .collect::<Vec<_>>();
            // A failed eval fails every lane of the step, as above.
            if !queued.is_empty()
                && let Err(error) =
                    MxArray::eval_arrays_with_context(&queued, "scheduled_verify_sampled_rows")
            {
                for slot in &mut sampled {
                    *slot = Some(Err(Error::from_reason(error.reason.clone())));
                }
            }
            for (verify, slot) in verify_rows.iter().zip(&sampled) {
                if slot.is_some() {
                    timeline::lane_mark(verify.seq_id, Mark::VerifyWait);
                }
            }
        }
        let mut decisions = Vec::with_capacity(work.len());
        let mut commits = Vec::with_capacity(work.len());
        let mut offset = 0i64;
        for ((item, verify), rows) in work.iter_mut().zip(&verify_rows).zip(sampled) {
            let turn = &mut running[item.turn_index];
            let end = offset + verify.tokens.len() as i64;
            let focus = timeline::focus_lane(verify.seq_id);
            let accepted = if let Some(rows) = rows {
                rows.and_then(|rows| rows.accept(&mut item.proposal, item.cut.as_ref()))
            } else {
                logits
                    .slice_axis(0, offset, end)
                    .and_then(|row| row.transpose(Some(&[1, 0, 2])))
                    .and_then(|row| {
                        if let Some(tokens) = greedy_tokens.as_ref() {
                            let tokens = tokens
                                .as_ref()
                                .map_err(|error| Error::from_reason(error.reason.clone()))?;
                            let mut target = tokens[offset as usize..end as usize].to_vec();
                            if let Some(forced) = item
                                .cut
                                .as_ref()
                                .map(|c| c.targets(&item.proposal.draft_ids))
                            {
                                for (row, token) in target.iter_mut().enumerate() {
                                    if let Some(end) = forced.at(row) {
                                        *token = end as i32;
                                    }
                                }
                            }
                            let mut count = 0;
                            while count < item.proposal.draft_ids.len()
                                && target[count] == item.proposal.draft_ids[count]
                            {
                                count += 1;
                            }
                            Ok((count, target[count] as u32))
                        } else if let Some(rng) = turn.payload.scheduled_speculation.as_mut() {
                            accept_dspark_proposal(
                                &row,
                                &mut item.proposal,
                                &turn.token_history,
                                &turn.payload.params,
                                rng,
                                item.cut.as_ref(),
                            )
                        } else {
                            accept_dspark_proposal(
                                &row,
                                &mut item.proposal,
                                &turn.token_history,
                                &turn.payload.params,
                                &mut rand::rng(),
                                item.cut.as_ref(),
                            )
                        }
                    })
            };
            drop(focus);
            timeline::lane_mark(verify.seq_id, Mark::Accept);
            offset = end;
            let decision = accepted.map(|(count, boundary)| {
                let mut tokens = item.proposal.draft_ids[..count]
                    .iter()
                    .map(|&token| token as u32)
                    .collect::<Vec<_>>();
                tokens.push(boundary);
                let budget_limit = thinking_budget_wave_limit(
                    &turn.payload.reasoning_tracker,
                    turn.payload.thinking,
                    &item.think_close,
                    &tokens,
                );
                let effective_count = if budget_limit < tokens.len() {
                    tokens.truncate(budget_limit);
                    budget_limit.saturating_sub(1)
                } else {
                    count
                };
                let cancelled = turn.cancelled.as_ref();
                let observer = &mut turn.payload.turn_token_observer;
                let clamp = clamp_dspark_cycle(
                    &turn.payload.generated_tokens,
                    &tokens,
                    effective_count,
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
                timeline::lane_counts(
                    verify.seq_id,
                    item.proposal.draft_ids.len(),
                    effective_count,
                    clamp.emit_count,
                );
                timeline::lane_mark(verify.seq_id, Mark::StopClamp);
                (tokens, clamp, effective_count)
            });
            commits.push(ScheduledVerifyCommit {
                seq_id: verify.seq_id,
                keep: decision.as_ref().map_or(0, |(_, clamp, _)| clamp.keep),
            });
            decisions.push(decision);
        }
        // This closes every ticket, including failed acceptance owners with
        // keep=0. Publication and owner reuse happen only after all commits.
        // The zero probe must retain its draft context for later calibration.
        // Permanent AR drops that maintenance. Use the completed target and
        // acceptance cost as a conservative AR lower bound, excluding commit
        // as well as calibration-only draft append/materialization. This can
        // prefer AR too early, but cannot credit speculation for an inflated
        // AR baseline.
        let zero_probe_ns = if profiling && zero_probe {
            wave_started.map_or(0, |t| {
                t.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64
            })
        } else {
            0
        };
        let committed = self.inner.commit_scheduled_verify(&commits);
        let mut committed = match committed {
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
        if profiling
            && !fallback
            && committed.iter().all(Result::is_ok)
            && decisions.iter().all(Result::is_ok)
        {
            match self.inner.complete_scheduled_draft_state(&seq_ids) {
                Ok(()) => {
                    let shape = verify_rows
                        .iter()
                        .map(|row| row.tokens.len() - 1)
                        .collect::<Vec<_>>();
                    if let Some(budget) = self.inner.scheduled_verification_budget() {
                        budget.record(
                            &shape,
                            if zero_probe {
                                zero_probe_ns
                            } else {
                                wave_started.map_or(0, |t| {
                                    t.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64
                                })
                            },
                        );
                    }
                }
                Err(error) => {
                    for result in &mut committed {
                        *result = Err(Error::from_reason(error.reason.clone()));
                    }
                    if let Some(budget) = self.inner.scheduled_verification_budget() {
                        budget.clear();
                    }
                }
            }
        }
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
            let drafted = item.proposal.draft_ids.len();
            let replaced = item.cut.as_ref().map_or(0, |c| c.replaced(drafted));
            if drafted > replaced {
                turn.payload
                    .profiler
                    .record_mtp_cycle(drafted - replaced, accepted_count.saturating_sub(replaced));
            }
            timeline::lane_mark(turn.seq_id, Mark::PostCommit);
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
            if !tokens.is_empty() {
                turn.payload.clock.token_event();
            }
            timeline::lane_mark(turn.seq_id, Mark::Emit);
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
                timeline::end_lane(turn.seq_id);
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
                    prefill_in_flight: false,
                },
            ));
        }
        Ok((results, occupancy, greedy_occupancy))
    }
}

/// Bytes the off-pool state of `own` (at its `queries` rows) and of
/// `others`, the idle owners' parked rows, and the bookings not yet in the
/// pool lack of what the pool may still hand out after the step's target
/// rows: free, reclaimable, marked and not yet grown up to its max; 0 when
/// it fits. State lives outside the pool, so target growth never takes it
/// out of those bytes and the pool never grows for it (Splash grows only
/// for KV pages, `KvPool::acquirePages`).
fn draft_shortfall<B: HybridSchedulerBackend>(
    inner: &B,
    own: &Turn<B>,
    others: &[&Turn<B>],
    queries: u32,
) -> Result<u64> {
    if !inner.scheduler_books_on_demand()
        || !own.recurrent_state_reservation_active
        || own.payload.scheduled_speculation.is_none()
    {
        return Ok(0);
    }
    let Some(snapshot) = inner.scheduler_cache_snapshot()? else {
        return Ok(0);
    };
    let need = std::iter::once((own, true))
        .chain(others.iter().map(|turn| (*turn, false)))
        .fold(0u64, |sum, (turn, is_own)| {
            let tokens = if is_own {
                turn.num_tokens.saturating_add(queries)
            } else {
                turn.num_tokens.saturating_add(turn.decode_draft_allowance)
            };
            sum.saturating_add(step_state_bytes(
                inner,
                turn,
                tokens,
                snapshot.bytes_per_block,
            ))
        });
    let owned = std::iter::once(own)
        .chain(others.iter().copied())
        .filter(|turn| {
            turn.recurrent_state_reservation_active && inner.has_scheduled_recurrent(turn.seq_id)
        })
        .count() as u64;
    let need = need.saturating_add(
        inner
            .scheduled_recurrent_bytes()
            .saturating_sub(inner.recurrent_state_bytes().saturating_mul(owned)),
    );
    let ungrown = snapshot
        .max_total_blocks
        .saturating_sub(snapshot.blocks.total_blocks);
    // The snapshot leaves marked chains out of what `allocate` may take;
    // the draft fit is a byte estimate, not a reservation, so it counts
    // them as before.
    let marked = inner
        .paged_adapter()
        .map_or(0, PagedKVCacheAdapter::marked_blocks);
    let available = u64::from(snapshot.blocks.free_blocks)
        .saturating_add(u64::from(snapshot.blocks.reclaimable_blocks))
        .saturating_add(u64::from(marked))
        .saturating_add(u64::from(ungrown))
        .saturating_mul(snapshot.bytes_per_block);
    Ok(need.saturating_sub(available))
}

#[cfg(all(test, target_os = "macos"))]
pub(super) mod tests {
    use super::*;
    use crate::models::gemma4::dspark_decode::tests::{
        seeded_tiny_paged_inner_with_draft, tiny_qwen_tokenizer, tiny_turn_config,
    };
    use rand::SeedableRng;

    pub(in crate::engine::hybrid_scheduler) fn request<B: HybridSchedulerBackend>(
        inner: &mut B,
        seq: u32,
        prompt: Vec<u32>,
        speculative: bool,
        cancel: Arc<AtomicBool>,
    ) -> TurnState<ScheduledTurn<B::PrefixState>> {
        request_reusing(inner, seq, prompt, speculative, cancel, false)
    }

    /// [`request`] whose prompt reuses the cache when `reuse`.
    pub(in crate::engine::hybrid_scheduler) fn request_reusing<B: HybridSchedulerBackend>(
        inner: &mut B,
        seq: u32,
        prompt: Vec<u32>,
        speculative: bool,
        cancel: Arc<AtomicBool>,
        reuse: bool,
    ) -> TurnState<ScheduledTurn<B::PrefixState>> {
        let config = tiny_turn_config(Some(3), 20);
        let params = inner.resolve_params(&config);
        let thinking = inner.thinking_setup(&config);
        let prefix = match inner
            .prepare_scheduled_prefix(seq, &prompt, &[], reuse, 0, 8)
            .unwrap()
        {
            ScheduledPrefixAdmission::Ready(prefix) => prefix,
            _ => unreachable!(),
        };
        // The prompt's own blocks, as its prefill slices' claims book them.
        if let Some(adapter) = inner.paged_adapter_mut() {
            let booked = adapter.current_token_count_for(seq).unwrap_or(0);
            adapter
                .reserve_rows_for(seq, (prompt.len() as u32).saturating_sub(booked))
                .unwrap();
        }
        let (reply, _) = tokio::sync::oneshot::channel();
        let requested_tokens = engine::scheduler::scheduled_materialized_tokens(
            prompt.len() as u32,
            params.max_new_tokens.max(0) as u32,
        );
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
            prefix_breakpoints: Vec::new(),
            prefix,
            is_delta: false,
            reuse_cache: reuse,
            response: ScheduledReply::Sync(reply),
            generated_tokens: Vec::new(),
            finish_reason: "length".into(),
            reasoning_tracker: engine::penalties::ReasoningTracker::new(
                thinking.enabled,
                thinking.budget,
                &[],
            ),
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
            vacated_stop: false,
            allocation_failed: false,
            preemption_replay: None,
            requested_tokens,
            claim: None,
            cell: None,
            draft_lane: None,
            clock: Default::default(),
        };
        let cached = if reuse {
            payload.prefix.effective_cached_prefix_len() as u32
        } else {
            0
        };
        let mut turn =
            TurnState::new(seq, prompt, cached, Vec::new(), Some(cancel), payload).unwrap();
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
        inner: B,
        speculate: bool,
        cancel_first: bool,
        sampled: bool,
    ) -> (Vec<(u32, Vec<u32>)>, usize) {
        run_modes(inner, speculate, cancel_first, sampled, false, false)
    }

    fn run_modes<B: HybridSchedulerBackend>(
        mut inner: B,
        speculate: bool,
        cancel_first: bool,
        sampled: bool,
        adaptive: bool,
        fixed_zero_peer: bool,
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
        for turn in [&mut first, &mut second] {
            turn.payload.params.mtp_adaptive_depth = adaptive;
        }
        if fixed_zero_peer {
            second.payload.params.mtp_adaptive_depth = false;
            second.decode_draft_allowance = 0;
        }
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
        let mut measured_adaptive_wave = false;
        let mut fixed_peer_waves = 0;
        for step in 0..100 {
            if cancel_first && step == 4 {
                cancel.store(true, Ordering::Relaxed);
            }
            let action = scheduler
                .drive_once(&mut HybridStepExecutor::new(&mut inner))
                .unwrap();
            measured_adaptive_wave |= inner
                .scheduled_verification_budget()
                .is_some_and(|policy| policy.has_measurements());
            if let SchedulerAction::Stepped {
                plan,
                completed: turns,
                ..
            } = action
            {
                if fixed_zero_peer
                    && plan
                        .rows
                        .iter()
                        .filter(|row| row.kind == StepKind::Decode)
                        .count()
                        == 2
                    && !turns.iter().any(|turn| turn.seq_id == 42)
                {
                    fixed_peer_waves += 1;
                    assert!(
                        !inner
                            .scheduled_verification_budget()
                            .unwrap()
                            .has_measurements(),
                        "a live zero-cap fixed speculator entered adaptive calibration"
                    );
                }
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
        if adaptive && !fixed_zero_peer {
            assert!(measured_adaptive_wave);
        }
        if fixed_zero_peer {
            assert!(fixed_peer_waves > 0);
        }
        completed.sort_by_key(|(seq, _)| *seq);
        (completed, scheduler.stats().max_batch_occupancy)
    }

    #[cfg(target_os = "macos")]
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

    /// Two owners cut at a thinking budget of 2 with the end `[13, 14]`; owner
    /// 42 speculates when `speculate_second`.
    fn run_ending(speculate_second: bool) -> Vec<(u32, Vec<u32>)> {
        run_ending_with(speculate_second, &[13, 14]).0
    }

    /// [`run_ending`] with `end`; also owner 42's generated length after each
    /// step.
    fn run_ending_with(speculate_second: bool, end: &[u32]) -> (Vec<(u32, Vec<u32>)>, Vec<usize>) {
        run_ending_stopped(speculate_second, end, None, None, &[])
    }

    /// [`run_ending_with`] with a `max_new_tokens`, an EOS id and extra EOS
    /// ids of its own.
    fn run_ending_stopped(
        speculate_second: bool,
        end: &[u32],
        max_new_tokens: Option<i32>,
        eos: Option<u32>,
        extra: &[u32],
    ) -> (Vec<(u32, Vec<u32>)>, Vec<usize>) {
        let mut inner = seeded_tiny_paged_inner_with_draft(1729).unwrap();
        let mut scheduler = Scheduler::<_, (), ()>::new(2, 12).unwrap();
        for (seq, prompt, speculate) in [
            (41, vec![0, 1, 2, 3, 4, 5, 6], false),
            (42, vec![4, 5, 6, 7, 8, 9, 10, 11, 12], speculate_second),
        ] {
            let mut turn = request(
                &mut inner,
                seq,
                prompt,
                speculate,
                Arc::new(AtomicBool::new(false)),
            );
            turn.payload.reasoning_tracker =
                engine::penalties::ReasoningTracker::new(true, Some(2), end);
            if let Some(max) = max_new_tokens {
                turn.payload.params.max_new_tokens = max;
            }
            if let Some(eos) = eos {
                turn.payload.eos_id = eos;
            }
            turn.payload.params.extra_eos_ids = extra.to_vec();
            turn.payload.extra_eos_ids = extra.to_vec();
            scheduler.enqueue_turn(turn).unwrap();
        }
        let mut completed = Vec::new();
        let mut lengths = Vec::new();
        for _ in 0..100 {
            if let SchedulerAction::Stepped {
                completed: turns, ..
            } = scheduler
                .drive_once(&mut HybridStepExecutor::new(&mut inner))
                .unwrap()
            {
                for turn in turns {
                    assert!(turn.payload.failure.is_none(), "{:?}", turn.payload.failure);
                    inner.release_scheduled_speculation(turn.seq_id);
                    if turn.seq_id == 42 {
                        lengths.push(turn.payload.generated_tokens.len());
                    }
                    completed.push((turn.seq_id, turn.payload.generated_tokens));
                }
            }
            if let Some(turn) = scheduler.running().iter().find(|turn| turn.seq_id == 42) {
                lengths.push(turn.payload.generated_tokens.len());
            }
            if completed.len() == 2 {
                break;
            }
        }
        completed.sort_by_key(|(seq, _)| *seq);
        (completed, lengths)
    }

    #[test]
    fn scheduled_rows_end_thinking_with_the_whole_close() {
        let ar = run_ending(false);
        let mixed = run_ending(true);
        for (seq, generated) in &ar {
            assert_eq!(&generated[2..4], &[13, 14], "owner {seq}: {generated:?}");
        }
        assert_eq!(ar, mixed, "a speculative row commits the end exactly as AR");
    }

    #[test]
    fn a_scheduled_end_is_drafted() {
        let mut end = vec![13; 8];
        end.push(14);
        let (ar, _) = run_ending_with(false, &end);
        let (mixed, lengths) = run_ending_with(true, &end);
        for (seq, generated) in &ar {
            assert_eq!(
                &generated[2..11],
                end.as_slice(),
                "owner {seq}: {generated:?}"
            );
        }
        assert_eq!(ar, mixed, "a speculative row commits the end exactly as AR");
        let stretch = 2..2 + end.len();
        let mut before = 0;
        let mut steps = 0;
        for &now in &lengths {
            if (before..now).any(|n| stretch.contains(&n)) {
                steps += 1;
            }
            before = now;
        }
        assert!(
            steps <= 1 + (end.len() - 1).div_ceil(4),
            "{steps} steps grew the end: {lengths:?}"
        );
    }

    /// `max_new_tokens` reached inside the end text ends a speculative row
    /// where it ends an AR row.
    #[test]
    fn a_scheduled_end_cut_by_max_new_tokens_matches_ar() {
        let mut end = vec![13; 8];
        end.push(14);
        for inside in [1, 3, 5, 8] {
            let max = Some(2 + inside as i32);
            let (ar, _) = run_ending_stopped(false, &end, max, None, &[]);
            let (mixed, _) = run_ending_stopped(true, &end, max, None, &[]);
            for (seq, generated) in &ar {
                assert_eq!(generated.len(), 2 + inside, "owner {seq}: {generated:?}");
                assert_eq!(&generated[2..], &end[..inside], "owner {seq}");
            }
            assert_eq!(ar, mixed, "inside={inside}");
        }
    }

    /// An EOS, or an extra EOS id, inside the end text stops a speculative
    /// row on it, as it stops an AR row.
    #[test]
    fn a_scheduled_eos_inside_the_end_stops_the_row() {
        let end = [13, 14, 13, 13, 13, 13, 9];
        for (label, eos, extra) in [("eos", Some(14), vec![]), ("extra eos", None, vec![14])] {
            let (ar, _) = run_ending_stopped(false, &end, None, eos, &extra);
            let (mixed, _) = run_ending_stopped(true, &end, None, eos, &extra);
            for (seq, generated) in &ar {
                assert_eq!(
                    &generated[2..],
                    &[13, 14],
                    "{label}: owner {seq}: {generated:?}"
                );
            }
            assert_eq!(ar, mixed, "{label}");
        }
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
    fn adaptive_dspark_calibration_and_fallback_preserve_owner_outputs() {
        let (ar, _) = run(false, false);
        let (adaptive, occupancy) = run_modes(
            seeded_tiny_paged_inner_with_draft(1729).unwrap(),
            true,
            false,
            false,
            true,
            false,
        );
        assert_eq!(occupancy, 2);
        assert_eq!(ar, adaptive);
        let (cancelled, _) = run_modes(
            seeded_tiny_paged_inner_with_draft(1729).unwrap(),
            true,
            true,
            false,
            true,
            false,
        );
        assert_eq!(cancelled[1], adaptive[1]);
        assert!(cancelled[0].1.len() < adaptive[0].1.len());
    }

    #[test]
    fn a_zero_draft_fixed_peer_cannot_enter_adaptive_fallback() {
        let (ar, _) = run(false, false);
        let (mixed, occupancy) = run_modes(
            seeded_tiny_paged_inner_with_draft(1729).unwrap(),
            true,
            false,
            false,
            true,
            true,
        );
        assert_eq!(occupancy, 2);
        assert_eq!(ar, mixed);
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

    #[cfg(target_os = "macos")]
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
    #[test]
    fn native_mtp_scheduler_commits_owner_history_and_isolates_cancellation() {
        if crate::test_support::paged_prefill_unsupported(
            "native_mtp_scheduler_commits_owner_history_and_isolates_cancellation",
        ) {
            return;
        }
        fn check<B: HybridSchedulerBackend>(build: impl Fn() -> B) {
            let (ar, _) = run_with(build(), false, false);
            let (spec, occupancy) = run_with(build(), true, false);
            assert_eq!(occupancy, 2);
            assert_eq!(
                spec, ar,
                "greedy native MTP differs from AR on tiny fixture"
            );
            for sampled in [false, true] {
                let (complete, _) = run_with_sampling(build(), true, false, sampled);
                let (cancelled, _) = run_with_sampling(build(), true, true, sampled);
                assert_eq!(
                    cancelled[1], complete[1],
                    "cancellation changed native MTP peer"
                );
                assert!(cancelled[0].1.len() < complete[0].1.len());
            }
        }
        check(|| crate::models::qwen3_5::model::scheduled_mtp::seeded_inner(8107));
        check(|| crate::models::qwen3_5::model::scheduled_mtp::seeded_moe_inner(8107));
    }

    /// A native error at the greedy verify eval fails the speculative row with
    /// MLX's text instead of the per-row acceptance reading what the failed
    /// command wrote. Mutation F1_VERIFY_GREEDY (the eval's error is ignored)
    /// → red.
    #[test]
    fn a_native_error_at_the_greedy_verify_fails_the_row() {
        if crate::test_support::paged_prefill_unsupported(
            "a_native_error_at_the_greedy_verify_fails_the_row",
        ) {
            return;
        }
        const OOM: &str = "[malloc] Unable to allocate 1073741824 bytes.";
        let mut inner = crate::models::qwen3_5::model::scheduled_mtp::seeded_inner(8107);
        let mut scheduler = Scheduler::<_, (), ()>::new(2, 12).unwrap();
        let turn = request(
            &mut inner,
            41,
            vec![0, 1, 2, 3, 4, 5, 6],
            true,
            Arc::new(AtomicBool::new(false)),
        );
        scheduler.enqueue_turn(turn).unwrap();
        let _armed = crate::array::fail_native_call_at("scheduled_verify_greedy", OOM);
        let mut failure = None;
        for _ in 0..100 {
            let action = scheduler
                .drive_once(&mut HybridStepExecutor::new(&mut inner))
                .unwrap();
            if let SchedulerAction::Stepped { completed, .. } = action
                && let Some(turn) = completed.into_iter().next()
            {
                failure = Some(turn.payload.failure);
                break;
            }
        }
        let Some(Some(error)) = failure else {
            panic!("the native error fails the row: {failure:?}");
        };
        assert_eq!(
            error.reason,
            format!(
                "MLX eval failed in scheduled_verify_greedy while materializing 1 arrays: {OOM}"
            )
        );
    }

    #[test]
    fn cached_speculative_starts_and_preemption_replay_keep_scheduled_ar() {
        if crate::test_support::paged_prefill_unsupported(
            "cached_speculative_starts_and_preemption_replay_keep_scheduled_ar",
        ) {
            return;
        }
        use crate::models::qwen3_5::scheduled_mtp::ScheduledMtpTarget;

        fn check<B: HybridSchedulerBackend>(
            mut inner: B,
            replay: bool,
            has_draft_owner: impl Fn(&B, u32) -> bool,
        ) {
            let source = (0..12).chain([15; 4]).collect::<Vec<u32>>();
            let mut warm = request(
                &mut inner,
                41,
                source.clone(),
                true,
                Arc::new(AtomicBool::new(false)),
            );
            // Replay must also discard a draft owner from the earlier wave.
            if replay {
                assert!(inner.begin_scheduled_speculation(41, 0).unwrap());
                assert!(has_draft_owner(&inner, 41));
            }
            // Materialize a target prefix as after a warm/SSD hit or replay.
            // Keep the remaining suffix split across chunks.
            if let Some(logits) = inner
                .run_scheduled_prefill_slice(
                    41,
                    &source,
                    &warm.payload.prefix,
                    0,
                    8,
                    warm.payload.generation_stream,
                    true,
                )
                .unwrap()
            {
                logits.eval();
            }
            warm.payload.prefix = inner.build_scheduled_prefix(
                &warm.payload.prefix,
                8,
                source.len() - 8,
                source.clone(),
                false,
            );
            warm.num_computed_tokens = 8;
            warm.pinned_prefill_breaks = vec![12, 16];
            warm.recurrent_state_bytes =
                inner.recurrent_state_bytes() + inner.scheduled_draft_state_bytes(36);
            if replay {
                // The final generated token was already emitted before
                // preemption but has not yet been materialized into KV.
                warm.payload.prompt_tokens.truncate(12);
                warm.payload.generated_tokens = vec![15; 5];
                warm.payload.pending_token_emitted = true;
                warm.token_history.push(15);
                warm.num_tokens += 1;
                warm.payload.preemption_replay = Some(PreemptionReplay {
                    tokens: source,
                    cached_prefix: 8,
                    suppress_sample: true,
                });
            }
            let peer = request(
                &mut inner,
                42,
                vec![4, 5, 6, 7, 8, 9, 10, 11, 12],
                true,
                Arc::new(AtomicBool::new(false)),
            );
            let mut scheduler = Scheduler::<_, (), ()>::new(2, 12).unwrap();
            scheduler.enqueue_turn(warm).unwrap();
            scheduler.enqueue_turn(peer).unwrap();
            let mut completed = 0;
            for _ in 0..100 {
                let action = scheduler
                    .drive_once(&mut HybridStepExecutor::new(&mut inner))
                    .unwrap();
                assert!(
                    !has_draft_owner(&inner, 41),
                    "{} cached target prefix must not create an unseeded draft owner",
                    B::SCHEDULER_NAME
                );
                if let SchedulerAction::Stepped {
                    completed: turns, ..
                } = action
                {
                    for turn in turns {
                        assert!(turn.payload.failure.is_none(), "{:?}", turn.payload.failure);
                        let mut expected = turn.payload.prompt_tokens.clone();
                        let generated = &turn.payload.generated_tokens;
                        assert_eq!(generated.len(), 20);
                        expected.extend_from_slice(&generated[..generated.len() - 1]);
                        assert_eq!(
                            inner
                                .paged_adapter()
                                .unwrap()
                                .request_tokens_for(turn.seq_id)
                                .unwrap(),
                            expected,
                            "fallback duplicated a pending token or lost committed history"
                        );
                        if turn.seq_id == 41 {
                            assert!(turn.payload.scheduled_speculation.is_none());
                            assert_eq!(turn.decode_draft_allowance, 0);
                            assert_eq!(turn.recurrent_state_bytes, inner.recurrent_state_bytes());
                        } else {
                            assert!(turn.payload.scheduled_speculation.is_some());
                        }
                        inner.release_scheduled_speculation(turn.seq_id);
                        completed += 1;
                    }
                }
                if completed == 2 {
                    break;
                }
            }
            assert_eq!(completed, 2);
            assert_eq!(scheduler.stats().max_batch_occupancy, 2);
        }
        for replay in [false, true] {
            check(
                crate::models::qwen3_5::model::scheduled_mtp::seeded_inner(8107),
                replay,
                |inner, seq| inner.mtp_state().owners.contains_key(&seq),
            );
            check(
                crate::models::qwen3_5::model::scheduled_mtp::seeded_moe_inner(8107),
                replay,
                |inner, seq| inner.mtp_state().owners.contains_key(&seq),
            );
            check(
                seeded_tiny_paged_inner_with_draft(1729).unwrap(),
                replay,
                |inner, seq| inner.scheduled_dspark_states.contains_key(&seq),
            );
            check(
                crate::models::muse_glimmer::model::scheduled_dflash::tests::seeded_inner(1908)
                    .unwrap(),
                replay,
                |inner, seq| inner.scheduled_dflash_states.contains_key(&seq),
            );
        }
    }

    struct EmitClock {
        inner: Box<dyn StreamEmitter>,
        first_emit_done: Arc<std::sync::Mutex<Option<Instant>>>,
    }

    impl StreamEmitter for EmitClock {
        fn on_token_id(
            &mut self,
            token_id: u32,
            token_text: &str,
            is_reasoning: bool,
            include_reasoning: bool,
            sink: &dyn ChunkSink,
        ) {
            self.inner
                .on_token_id(token_id, token_text, is_reasoning, include_reasoning, sink);
            self.first_emit_done
                .lock()
                .unwrap()
                .get_or_insert_with(Instant::now);
        }

        fn on_token_text(
            &mut self,
            token_text: &str,
            is_reasoning: bool,
            include_reasoning: bool,
            sink: &dyn ChunkSink,
        ) {
            self.inner
                .on_token_text(token_text, is_reasoning, include_reasoning, sink);
        }

        fn on_residual(
            &mut self,
            residual: &str,
            is_reasoning: bool,
            include_reasoning: bool,
            sink: &dyn ChunkSink,
        ) {
            self.inner
                .on_residual(residual, is_reasoning, include_reasoning, sink);
        }

        fn finish(&mut self, result: &ChatResult, sink: &dyn ChunkSink) {
            self.inner.finish(result, sink);
        }
    }

    /// The reported TTFT must cover the first token's delivery to the
    /// stream, not only its sample: the prefill-sampled token is emitted one
    /// decode forward later, on both the AR and the speculative seed paths.
    ///
    /// MUTATION: stamp `first_token_instant` right after `sampled.eval()` in
    /// `execute_prefill` (the pre-fix site) and drop the stamp in
    /// `finish_decode_row` — both arms fail: the stamp precedes the emit.
    #[test]
    fn reported_ttft_is_not_earlier_than_the_first_streamed_token() {
        for speculate in [false, true] {
            let mut inner = seeded_tiny_paged_inner_with_draft(1729).unwrap();
            let mut turn = request(
                &mut inner,
                41,
                vec![0, 1, 2, 3, 4, 5, 6],
                speculate,
                Arc::new(AtomicBool::new(false)),
            );
            let (stream_tx, mut stream_rx) = crate::model_thread::stream_channel(64);
            let first_emit_done = Arc::new(std::sync::Mutex::new(None));
            turn.payload.response = ScheduledReply::Stream(stream_tx);
            turn.payload.emitter = Some(Box::new(EmitClock {
                inner: Box::new(engine::backend::DefaultStreamEmitter),
                first_emit_done: Arc::clone(&first_emit_done),
            }));
            turn.payload.params.include_reasoning = true;
            assert!(turn.payload.params.report_performance);
            let generation_start = Instant::now();
            turn.payload.generation_start = Some(generation_start);
            let mut scheduler = Scheduler::<_, (), ()>::new(2, 12).unwrap();
            scheduler.enqueue_turn(turn).unwrap();
            let mut finished = None;
            for _ in 0..100 {
                if let SchedulerAction::Stepped { mut completed, .. } = scheduler
                    .drive_once(&mut HybridStepExecutor::new(&mut inner))
                    .unwrap()
                    && let Some(turn) = completed.pop()
                {
                    finished = Some(turn);
                    break;
                }
            }
            let turn = finished.expect("the turn completes");
            assert!(turn.payload.failure.is_none(), "{:?}", turn.payload.failure);
            assert!(
                turn.payload.generated_tokens.len() > 1,
                "speculate={speculate}: the turn must decode past its first token"
            );
            assert!(
                stream_rx.try_recv().is_ok_and(|chunk| chunk.is_ok()),
                "speculate={speculate}: the first token reached the stream"
            );
            let emitted = first_emit_done
                .lock()
                .unwrap()
                .expect("the first token was emitted");
            let stamped = turn
                .payload
                .first_token_instant
                .expect("report_performance stamps the first token");
            assert!(
                stamped >= emitted,
                "speculate={speculate}: TTFT stamped {:?} before the first emit",
                emitted.duration_since(stamped)
            );
            let metrics = engine::finalize::compute_performance_metrics(
                turn.payload.generation_start,
                turn.payload.first_token_instant,
                turn.payload.prompt_tokens.len(),
                turn.payload.generated_tokens.len(),
            )
            .unwrap();
            let emitted_ms = emitted.duration_since(generation_start).as_secs_f64() * 1000.0;
            assert!(
                metrics.ttft_ms >= emitted_ms,
                "speculate={speculate}: reported ttft {} ms < first emit {} ms",
                metrics.ttft_ms,
                emitted_ms
            );
        }
    }

    /// A speculative decode row's draft-state growth is a bytes claim under
    /// its request's claim (Splash: state is an allocation of bytes, never
    /// of pool pages): with room one byte short of the growth it is denied
    /// with the growth's bytes and no blocks, though the pool has free
    /// blocks; for a bytes need the idle owner's parked row is a unit and
    /// its blocks are not, for a blocks need its blocks are.
    /// Mutation D1: the draft need counts the pool's free blocks → red (the
    /// row is granted).
    #[test]
    fn draft_state_is_a_bytes_claim() {
        use crate::engine::governor::governor::tests::{FakeIo, G, host, model, stats};
        use crate::engine::governor::ledger::TakeRights;
        use crate::engine::hybrid_scheduler::tests::mtp_step_state_with;
        let Some((mut state, _)) = mtp_step_state_with(64, 64) else {
            return;
        };
        let cancel = Arc::new(AtomicBool::new(false));
        drop(request(
            &mut state.inner,
            2,
            (0..40).map(|token| 500 + token).collect(),
            false,
            Arc::clone(&cancel),
        ));
        state.owner_sequences.insert("idle".into(), 2);
        let mut turn = request(
            &mut state.inner,
            1,
            (0..40).map(|token| token % 16).collect(),
            true,
            cancel,
        );
        turn.num_computed_tokens = 40;
        turn.token_history.push(3);
        turn.num_tokens = 41;
        let growth = state.inner.scheduled_draft_state_held_bytes(44)
            - state.inner.scheduled_draft_state_held_bytes(40);
        assert!(growth > 0);
        let io = FakeIo::new(
            stats(0),
            host(32 * G, crate::engine::governor::host::Pressure::Normal),
        );
        let governor: &'static Governor = Box::leak(Box::new(Governor::new(io)));
        state.inner.memory_registration = Some(
            governor
                .register_model(model(growth - 1, 0))
                .expect("register"),
        );
        {
            let units = SchedUnits {
                state: &mut state,
                claimant: 1,
                kept: &[],
            };
            let query = |need: Need| VictimQuery {
                rights: TakeRights {
                    units: true,
                    certain_reuse: crate::engine::governor::ledger::CertainReuse::All,
                    ..TakeRights::default()
                },
                need,
            };
            let owner = |units: Vec<Victim>| {
                units
                    .into_iter()
                    .find(|unit| unit.key == LedgerKey::Owner(2))
                    .expect("the idle owner is a unit")
            };
            let bytes = owner(units.units(&query(Need {
                bytes: growth,
                ..Need::default()
            })));
            assert!(bytes.bytes > 0);
            assert_eq!(bytes.pages, 0);
            let pages = owner(units.units(&query(Need {
                pages: 1,
                kv: true,
                ..Need::default()
            })));
            assert!(pages.pages > 0);
        }
        state.scheduler.reinstate_running(turn);
        let plan = StepPlan {
            global_step: 0,
            rows: vec![engine::scheduler::StepRow {
                seq_id: 1,
                kind: StepKind::Decode,
                token_start: 40,
                num_tokens: 4,
                cancel_snapshot: false,
            }],
            contended: false,
            leader_cut: None,
            prefill_pass_ms: None,
        };
        let claims = state.claim_rows_in_order(&plan);
        let denial = claims.grants[0].expect("the draft growth is refused");
        assert_eq!(denial.failure, Some(Failure::EngineBudget));
        assert!(denial.blocks.is_none());
        let claim = state
            .scheduler
            .running_turn_mut(1)
            .and_then(|turn| turn.payload.claim.as_ref().map(ClaimHandle::id))
            .expect("the row's claim");
        let refusal = governor.ledger(|ledger| ledger.claim(claim).map(|state| state.refusal));
        assert_eq!(refusal.map(|need| need.bytes), Some(growth));
    }

    /// An MTP prefill slice grows the draft K/V of its rows (Splash
    /// `Engine::growLane` claims a lane's state to its work end for every
    /// row): with room one byte short of that growth the slice is denied
    /// with its bytes and no blocks, though its KV rows are booked.
    /// Mutation MP1: only decode rows claim draft state → red (the slice
    /// is granted).
    #[test]
    fn an_mtp_prefill_slice_claims_its_draft_growth() {
        use crate::engine::governor::governor::tests::{FakeIo, G, host, model, stats};
        use crate::engine::hybrid_scheduler::tests::mtp_step_state_with;
        let Some((mut state, _)) = mtp_step_state_with(64, 64) else {
            return;
        };
        let turn = request(
            &mut state.inner,
            1,
            (0..40).map(|token| token % 16).collect(),
            true,
            Arc::new(AtomicBool::new(false)),
        );
        let growth = state.inner.scheduled_draft_state_held_bytes(16)
            - state.inner.scheduled_draft_state_held_bytes(0);
        assert!(growth > 0);
        let io = FakeIo::new(
            stats(0),
            host(32 * G, crate::engine::governor::host::Pressure::Normal),
        );
        let governor: &'static Governor = Box::leak(Box::new(Governor::new(io)));
        state.inner.memory_registration = Some(
            governor
                .register_model(model(growth - 1, 0))
                .expect("register"),
        );
        state.scheduler.reinstate_running(turn);
        let plan = StepPlan {
            global_step: 0,
            rows: vec![engine::scheduler::StepRow {
                seq_id: 1,
                kind: StepKind::Prefill,
                token_start: 0,
                num_tokens: 16,
                cancel_snapshot: false,
            }],
            contended: false,
            leader_cut: None,
            prefill_pass_ms: None,
        };
        let claims = state.claim_rows_in_order(&plan);
        let denial = claims.grants[0].expect("the draft growth is refused");
        assert_eq!(denial.failure, Some(Failure::EngineBudget));
        assert!(denial.blocks.is_none());
        let claim = state
            .scheduler
            .running_turn_mut(1)
            .and_then(|turn| turn.payload.claim.as_ref().map(ClaimHandle::id))
            .expect("the row's claim");
        let refusal = governor.ledger(|ledger| ledger.claim(claim).map(|state| state.refusal));
        assert_eq!(refusal.map(|need| need.bytes), Some(growth));
    }

    /// Pre-verify per-row validation isolates a single lane whose target
    /// frontier is mismatched so ONLY that lane fails while its healthy peer in
    /// the same 2-lane speculative wave completes normally with exact output.
    #[test]
    fn mismatched_frontier_row_fails_in_isolation_without_killing_healthy_peer() {
        let (baseline, _) = run(true, false);
        let mut inner = seeded_tiny_paged_inner_with_draft(1729).unwrap();
        let mut scheduler = Scheduler::<_, (), ()>::new(2, 12).unwrap();
        let first = request(
            &mut inner,
            41,
            vec![0, 1, 2, 3, 4, 5, 6],
            true,
            Arc::new(AtomicBool::new(false)),
        );
        let second = request(
            &mut inner,
            42,
            vec![4, 5, 6, 7, 8, 9, 10, 11, 12],
            true,
            Arc::new(AtomicBool::new(false)),
        );
        scheduler.enqueue_turn(first).unwrap();
        scheduler.enqueue_turn(second).unwrap();

        let _guard = crate::engine::hybrid_scheduler::set_test_validate_scheduled_verify_row(
            Arc::new(|adapter, row| {
                let frontier = adapter
                    .and_then(|a| a.current_token_count_for(row.seq_id))
                    .ok_or_else(|| {
                        Error::from_reason("scheduled verifier has no target frontier")
                    })?;
                if u64::from(frontier) != u64::from(row.first_position) {
                    return Err(Error::from_reason(
                        "scheduled verifier target frontier mismatch",
                    ));
                }
                Ok(())
            }),
        );

        let mut injected = false;
        let mut lane_41_failure = None;
        let mut lane_42_tokens = None;
        for _ in 0..100 {
            // Once both lanes have finished prefill and are ready to enter a
            // 2-lane decode verify wave, desync ONLY lane 41's target frontier.
            if !injected
                && scheduler.running().len() == 2
                && scheduler
                    .running()
                    .iter()
                    .all(|t| t.num_computed_tokens >= t.payload.prompt_tokens.len() as u32)
            {
                inner
                    .paged_adapter_mut()
                    .unwrap()
                    .rollback_last_tokens_for(41, 1)
                    .unwrap();
                injected = true;
            }
            let action = scheduler
                .drive_once(&mut HybridStepExecutor::new(&mut inner))
                .unwrap();
            if let SchedulerAction::Stepped { completed, .. } = action {
                for turn in completed {
                    inner.release_scheduled_speculation(turn.seq_id);
                    if turn.seq_id == 41 {
                        lane_41_failure = turn.payload.failure;
                    } else if turn.seq_id == 42 {
                        assert!(
                            turn.payload.failure.is_none(),
                            "healthy peer lane 42 must not fail: {:?}",
                            turn.payload.failure
                        );
                        lane_42_tokens = Some(turn.payload.generated_tokens);
                    }
                }
            }
            if lane_41_failure.is_some() && lane_42_tokens.is_some() {
                break;
            }
        }
        assert!(injected, "both lanes must reach decode wave");
        let err = lane_41_failure.expect("mismatched lane 41 must fail");
        assert!(
            err.reason
                .contains("scheduled verifier target frontier mismatch"),
            "unexpected lane 41 error: {}",
            err.reason
        );
        assert_eq!(
            lane_42_tokens.expect("healthy peer lane 42 must finish"),
            baseline[1].1,
            "healthy peer lane 42 output must match baseline exactly"
        );
    }

    struct DynamicThinkTestGuards;
    impl Drop for DynamicThinkTestGuards {
        fn drop(&mut self) {
            TEST_THINK_END_OVERRIDE.with(|slot| *slot.borrow_mut() = None);
            TEST_DRAFT_OVERRIDE.with(|slot| *slot.borrow_mut() = None);
        }
    }

    /// Drive a single request (greedy) with optional dynamic open-token thinking
    /// setup and optional wave-1 draft override, returning `(generated, per_wave_lens, outcome)`.
    fn run_single_dynamic_thinking(
        prompt: Vec<u32>,
        speculate: bool,
        budget: Option<i32>,
        close: Option<ThinkEndResolution>,
        draft_allowance: usize,
        max_new_tokens: i32,
        wave1_drafts: Option<Vec<i32>>,
    ) -> (Vec<u32>, Vec<usize>, engine::penalties::ReasoningOutcome) {
        let _guards = DynamicThinkTestGuards;
        TEST_THINK_END_OVERRIDE.with(|slot| *slot.borrow_mut() = close.clone());
        if let Some(drafts) = wave1_drafts {
            let mut fired = false;
            TEST_DRAFT_OVERRIDE.with(|slot| {
                *slot.borrow_mut() = Some(Box::new(move |_seq, _anchor, cap| {
                    if fired {
                        return None;
                    }
                    fired = true;
                    Some(drafts[..drafts.len().min(cap)].to_vec())
                }));
            });
        }

        let mut inner = seeded_tiny_paged_inner_with_draft(1729).unwrap();
        let mut scheduler = Scheduler::<_, (), ()>::new(2, 16).unwrap();
        let mut turn = request(
            &mut inner,
            41,
            prompt,
            speculate,
            Arc::new(AtomicBool::new(false)),
        );
        turn.payload.params.max_new_tokens = max_new_tokens;
        turn.decode_draft_allowance = if speculate { draft_allowance as u32 } else { 0 };
        if let Some(ref close) = close {
            turn.payload.thinking = ThinkingSetup {
                enabled: true,
                budget,
                cycle_guard: false,
            };
            turn.payload.reasoning_tracker =
                ReasoningTracker::from_setup(&turn.payload.thinking, close);
            turn.payload.last_is_reasoning = false;
        }
        scheduler.enqueue_turn(turn).unwrap();

        let mut wave_lens = Vec::new();
        let mut prev_len = 0usize;
        let mut finished: Option<(Vec<u32>, engine::penalties::ReasoningOutcome)> = None;
        for _ in 0..100 {
            let action = scheduler
                .drive_once(&mut HybridStepExecutor::new(&mut inner))
                .unwrap();
            if let SchedulerAction::Stepped { completed, .. } = action {
                for turn in completed {
                    assert!(
                        turn.payload.failure.is_none(),
                        "owner {} failed: {:?}",
                        turn.seq_id,
                        turn.payload.failure
                    );
                    let generated = turn.payload.generated_tokens;
                    if generated.len() > prev_len {
                        wave_lens.push(generated.len() - prev_len);
                        prev_len = generated.len();
                    }
                    let mut expected = turn.payload.prompt_tokens;
                    expected.extend_from_slice(&generated[..generated.len().saturating_sub(1)]);
                    assert_eq!(
                        inner
                            .paged_adapter()
                            .unwrap()
                            .request_tokens_for(turn.seq_id)
                            .unwrap(),
                        expected,
                        "KV/request history must match committed tokens after wave rollback"
                    );
                    let outcome = turn.payload.reasoning_tracker.outcome();
                    inner.release_scheduled_speculation(turn.seq_id);
                    finished = Some((generated, outcome));
                }
            }
            if let Some(running_turn) = scheduler.running().iter().find(|t| t.seq_id == 41) {
                let len = running_turn.payload.generated_tokens.len();
                if len > prev_len {
                    wave_lens.push(len - prev_len);
                    prev_len = len;
                }
            }
            if finished.is_some() {
                break;
            }
        }
        let (generated, outcome) = finished.expect("single turn must finish");
        (generated, wave_lens, outcome)
    }

    /// Find a deterministic prompt whose greedy baseline produces a fresh token
    /// at index `open_idx` (`1 <= open_idx <= 2`) that did NOT appear in
    /// `baseline[..open_idx]`, so the seed emitted from prefill (`baseline[0]`)
    /// leaves the turn outside thinking and Wave 1 dynamically opens thinking.
    fn pick_dynamic_open_baseline(max_new: i32) -> (Vec<u32>, Vec<u32>, usize) {
        let candidates = [
            vec![0, 1, 2, 3, 4, 5, 6],
            vec![4, 5, 6, 7, 8, 9, 10, 11, 12],
            vec![1, 3, 5, 7, 2, 4, 6],
            vec![6, 5, 4, 3, 2, 1, 0],
            vec![2, 0, 4, 1, 6, 3, 5],
        ];
        for prompt in candidates {
            let (baseline, _, _) =
                run_single_dynamic_thinking(prompt.clone(), false, None, None, 0, max_new, None);
            if let Some(open_idx) = (1..=2)
                .find(|&idx| idx < baseline.len() && !baseline[..idx].contains(&baseline[idx]))
            {
                return (prompt, baseline, open_idx);
            }
        }
        panic!("no candidate prompt produced a fresh dynamic open token in Wave 1");
    }

    /// Finding 1 regression (zero budget): a dynamic open-token speculative wave
    /// starting outside thinking (`anchor = baseline[0]`, `thinking_cut() == None`)
    /// that proposes `[..., open_marker, thought_1, thought_2, ...]` (all matching
    /// target argmax) under `budget == 0` must truncate Wave 1 right after
    /// `open_marker`, roll back KV/draft cleanly, force the close sequence
    /// immediately in Wave 2, emit ZERO thought tokens past budget, and match AR.
    #[test]
    fn dynamic_open_zero_budget_speculative_wave_never_commits_thought_tokens() {
        let (prompt, baseline, open_idx) = pick_dynamic_open_baseline(10);
        assert!(baseline.len() >= open_idx + 5);
        let open_id = baseline[open_idx];
        let close_ids: Vec<u32> = (0..16u32)
            .filter(|t| !baseline[..open_idx + 5].contains(t))
            .take(2)
            .collect();
        assert_eq!(
            close_ids.len(),
            2,
            "tiny vocab must have 2 unused close ids"
        );
        let close = ThinkEndResolution::with_open_token(open_id, Arc::from(close_ids.as_slice()));

        let (ar_tokens, _, ar_outcome) = run_single_dynamic_thinking(
            prompt.clone(),
            false,
            Some(0),
            Some(close.clone()),
            0,
            10,
            None,
        );
        // Prefill produces seed `baseline[0]` (observed outside thinking); Wave 1
        // verifies `[baseline[0], baseline[1], ..., open_id, t1, t2, t3]` so greedy
        // verification accepts open_id + 3 thought tokens before budget truncation.
        let draft_len = open_idx + 3;
        let wave1_drafts: Vec<i32> = baseline[1..1 + draft_len]
            .iter()
            .map(|&t| t as i32)
            .collect();
        let (spec_tokens, wave_lens, spec_outcome) = run_single_dynamic_thinking(
            prompt,
            true,
            Some(0),
            Some(close),
            draft_len,
            10,
            Some(wave1_drafts),
        );

        assert_eq!(
            wave_lens[0],
            open_idx + 1,
            "Step 1 (seed + Wave 1) under budget=0 must stop right after open_id (lens={wave_lens:?})"
        );
        assert_eq!(
            spec_tokens, ar_tokens,
            "speculative output must match greedy AR with zero thought tokens past budget"
        );
        assert_eq!(spec_tokens[open_idx], open_id);
        assert_eq!(
            &spec_tokens[open_idx + 1..open_idx + 3],
            close_ids.as_slice(),
            "close sequence must follow open_id immediately with 0 thought tokens"
        );
        assert_eq!(spec_outcome, ar_outcome);
        assert_eq!(spec_outcome.close, Some(ReasoningClose::Budget));
        assert_eq!(spec_outcome.cut_at, Some(0));
    }

    /// Finding 1 regression (tight non-zero budget): a dynamic open-token
    /// speculative wave starting outside thinking (`thinking_cut() == None`)
    /// that proposes `[..., open_marker, t1, t2, t3, t4]` (all matching target
    /// argmax) under `budget == 2` must truncate right after `t2`, drop `t3`/`t4`,
    /// roll back KV/draft cleanly, force the close sequence on the next wave,
    /// and match greedy AR bit-for-bit.
    #[test]
    fn dynamic_open_tight_budget_crossed_mid_wave_truncates_and_matches_ar() {
        let (prompt, baseline, open_idx) = pick_dynamic_open_baseline(12);
        assert!(baseline.len() >= open_idx + 6);
        let open_id = baseline[open_idx];
        let close_ids: Vec<u32> = (0..16u32)
            .filter(|t| !baseline[..open_idx + 6].contains(t))
            .take(2)
            .collect();
        assert_eq!(
            close_ids.len(),
            2,
            "tiny vocab must have 2 unused close ids"
        );
        let close = ThinkEndResolution::with_open_token(open_id, Arc::from(close_ids.as_slice()));

        let (ar_tokens, _, ar_outcome) = run_single_dynamic_thinking(
            prompt.clone(),
            false,
            Some(2),
            Some(close.clone()),
            0,
            12,
            None,
        );
        // Prefill produces seed `baseline[0]` (outside thinking); Wave 1 verifies
        // `[baseline[0], ..., open_id, t1, t2, t3, t4]` so greedy verification
        // accepts 2 extra thought tokens past budget=2 before budget truncation.
        let draft_len = open_idx + 4;
        let wave1_drafts: Vec<i32> = baseline[1..1 + draft_len]
            .iter()
            .map(|&t| t as i32)
            .collect();
        let (spec_tokens, wave_lens, spec_outcome) = run_single_dynamic_thinking(
            prompt,
            true,
            Some(2),
            Some(close),
            draft_len,
            12,
            Some(wave1_drafts),
        );

        assert_eq!(
            wave_lens[0],
            open_idx + 3,
            "Step 1 (seed + Wave 1) under budget=2 must stop right after [open_id, t1, t2] (lens={wave_lens:?})"
        );
        assert_eq!(
            spec_tokens, ar_tokens,
            "speculative output must match greedy AR when budget=2 is crossed mid-wave"
        );
        assert_eq!(spec_tokens[open_idx], open_id);
        assert_eq!(
            &spec_tokens[open_idx + 1..open_idx + 3],
            &baseline[open_idx + 1..open_idx + 3]
        );
        assert_eq!(
            &spec_tokens[open_idx + 3..open_idx + 5],
            close_ids.as_slice(),
            "close sequence must immediately follow the 2 allowed thought tokens"
        );
        assert_eq!(spec_outcome, ar_outcome);
        assert_eq!(spec_outcome.close, Some(ReasoningClose::Budget));
        assert_eq!(spec_outcome.cut_at, Some(2));
    }

    /// Unit tests for `thinking_budget_wave_limit` edge cases: reopening a
    /// second thought block within the same wave after crossing budget,
    /// immediate model self-close under budget=0, and partial multi-token end.
    #[test]
    fn thinking_budget_wave_limit_handles_reopen_self_close_and_multitoken_end() {
        const OPEN: u32 = 50;
        const END0: u32 = 51;
        const CLOSE: u32 = 52;
        let close = ThinkEndResolution::with_open_token(OPEN, Arc::from([END0, CLOSE]));

        // 1. Re-opening thinking in the same wave hits budget=2 on the second block.
        let setup_2 = ThinkingSetup {
            enabled: true,
            budget: Some(2),
            cycle_guard: false,
        };
        let tracker_2 = ReasoningTracker::from_setup(&setup_2, &close);
        assert_eq!(
            thinking_budget_wave_limit(&tracker_2, setup_2, &close, &[OPEN, 1, CLOSE, OPEN, 2, 3]),
            5,
            "must drop token `3` after 2 cumulative thought tokens across two blocks"
        );

        // 2. Model immediately emits the full end sequence under budget=0 -> no truncation.
        let setup_0 = ThinkingSetup {
            enabled: true,
            budget: Some(0),
            cycle_guard: false,
        };
        let tracker_0 = ReasoningTracker::from_setup(&setup_0, &close);
        assert_eq!(
            thinking_budget_wave_limit(&tracker_0, setup_0, &close, &[OPEN, END0, CLOSE, 99]),
            4,
            "matching forced end tokens must not be truncated"
        );

        // 3. Partial match of multi-token end followed by a stray thought token truncates before stray.
        assert_eq!(
            thinking_budget_wave_limit(&tracker_0, setup_0, &close, &[OPEN, END0, 77]),
            2,
            "stray thought token after partial multi-token end must be truncated"
        );
    }

    struct DeviceEvalTestGuards;
    impl Drop for DeviceEvalTestGuards {
        fn drop(&mut self) {
            TEST_POST_BATCH_PROPOSAL_HOOK.with(|slot| *slot.borrow_mut() = None);
            TEST_SOLO_PROPOSAL_HOOK.with(|slot| *slot.borrow_mut() = None);
            TEST_DRAFT_EVAL_OVERRIDE.with(|slot| *slot.borrow_mut() = None);
        }
    }

    /// Finding 2 regression: when batched evaluation of deferred device-backed
    /// draft proposals fails in a multi-lane wave, the executor must NOT fail
    /// every device-backed lane.
    ///
    /// Covers both:
    /// 1. Independent per-lane failure: lane 41's device array fails eval while
    ///    lane 42's device array evaluates cleanly in per-lane isolation (no
    ///    unnecessary solo retry on lane 42); lane 41 fails in isolation and
    ///    lane 42 finishes matching baseline bit-for-bit.
    /// 2. Shared batched graph failure: both batched device arrays fail isolated
    ///    eval (simulating a shared batched forward graph poisoned by lane 41);
    ///    solo retry recovers lane 42 with a fresh device proposal while lane 41
    ///    fails in isolation and lane 42 finishes matching baseline bit-for-bit.
    #[test]
    fn deferred_device_draft_eval_failure_isolates_failing_lane_and_preserves_healthy_peer() {
        let (baseline, _) = run(true, false);

        for shared_graph_poison in [false, true] {
            let _guards = DeviceEvalTestGuards;
            let bad_ptrs = Arc::new(std::sync::Mutex::new(
                std::collections::HashSet::<usize>::new(),
            ));
            let solo_retried = Arc::new(std::sync::Mutex::new(Vec::<u32>::new()));
            let batch_evals_failed = Arc::new(std::sync::atomic::AtomicUsize::new(0));

            let injected = Arc::new(AtomicBool::new(false));
            {
                let bad_ptrs = Arc::clone(&bad_ptrs);
                let injected = Arc::clone(&injected);
                TEST_POST_BATCH_PROPOSAL_HOOK.with(|slot| {
                    *slot.borrow_mut() = Some(Box::new(move |wave_seqs, proposals| {
                        if injected.load(Ordering::Relaxed)
                            || wave_seqs.len() != 2
                            || !wave_seqs.contains(&41)
                            || !wave_seqs.contains(&42)
                        {
                            return;
                        }
                        injected.store(true, Ordering::Relaxed);
                        for (&seq, res) in wave_seqs.iter().zip(proposals.iter_mut()) {
                            let Ok(prop) = res.as_mut() else {
                                continue;
                            };
                            let host_ids = if prop.draft_ids.is_empty() {
                                vec![0i32]
                            } else {
                                std::mem::take(&mut prop.draft_ids)
                            };
                            let dev = MxArray::from_int32(&host_ids, &[host_ids.len() as i64])
                                .expect("device draft ids");
                            if seq == 41 || shared_graph_poison {
                                bad_ptrs.lock().unwrap().insert(dev.as_raw_ptr() as usize);
                            }
                            prop.device_draft_ids = Some(dev);
                        }
                    }));
                });
            }
            {
                let bad_ptrs = Arc::clone(&bad_ptrs);
                let solo_retried = Arc::clone(&solo_retried);
                TEST_SOLO_PROPOSAL_HOOK.with(|slot| {
                    *slot.borrow_mut() = Some(Box::new(move |seq, solo| {
                        solo_retried.lock().unwrap().push(seq);
                        let mut prop = solo?;
                        let host_ids = if prop.draft_ids.is_empty() {
                            vec![0i32]
                        } else {
                            std::mem::take(&mut prop.draft_ids)
                        };
                        let dev = MxArray::from_int32(&host_ids, &[host_ids.len() as i64])
                            .expect("solo device draft ids");
                        let ptr = dev.as_raw_ptr() as usize;
                        if seq == 41 {
                            bad_ptrs.lock().unwrap().insert(ptr);
                        } else {
                            // Guard against allocator pointer reuse after the dropped batched array.
                            bad_ptrs.lock().unwrap().remove(&ptr);
                        }
                        prop.device_draft_ids = Some(dev);
                        Ok(prop)
                    }));
                });
            }
            {
                let bad_ptrs = Arc::clone(&bad_ptrs);
                let batch_evals_failed = Arc::clone(&batch_evals_failed);
                TEST_DRAFT_EVAL_OVERRIDE.with(|slot| {
                    *slot.borrow_mut() = Some(Box::new(move |arrays, context| {
                        let guard = bad_ptrs.lock().unwrap();
                        if arrays
                            .iter()
                            .any(|arr| guard.contains(&(arr.as_raw_ptr() as usize)))
                        {
                            if context == "scheduled_draft_proposals" {
                                batch_evals_failed.fetch_add(1, Ordering::Relaxed);
                            }
                            return Some(Err(Error::from_reason(
                                "injected deferred device draft eval failure",
                            )));
                        }
                        None
                    }));
                });
            }

            let mut inner = seeded_tiny_paged_inner_with_draft(1729).unwrap();
            let mut scheduler = Scheduler::<_, (), ()>::new(2, 12).unwrap();
            let first = request(
                &mut inner,
                41,
                vec![0, 1, 2, 3, 4, 5, 6],
                true,
                Arc::new(AtomicBool::new(false)),
            );
            let second = request(
                &mut inner,
                42,
                vec![4, 5, 6, 7, 8, 9, 10, 11, 12],
                true,
                Arc::new(AtomicBool::new(false)),
            );
            scheduler.enqueue_turn(first).unwrap();
            scheduler.enqueue_turn(second).unwrap();

            let mut lane_41_failure = None;
            let mut lane_42_tokens = None;
            for _ in 0..100 {
                let action = scheduler
                    .drive_once(&mut HybridStepExecutor::new(&mut inner))
                    .unwrap();
                if let SchedulerAction::Stepped { completed, .. } = action {
                    for turn in completed {
                        inner.release_scheduled_speculation(turn.seq_id);
                        if turn.seq_id == 41 {
                            lane_41_failure = turn.payload.failure;
                        } else if turn.seq_id == 42 {
                            assert!(
                                turn.payload.failure.is_none(),
                                "shared_graph_poison={shared_graph_poison}: healthy peer lane 42 must not fail: {:?}",
                                turn.payload.failure
                            );
                            lane_42_tokens = Some(turn.payload.generated_tokens);
                        }
                    }
                }
                if lane_41_failure.is_some() && lane_42_tokens.is_some() {
                    break;
                }
            }

            assert!(
                injected.load(Ordering::Relaxed),
                "shared_graph_poison={shared_graph_poison}: 2-lane wave must trigger device proposal injection"
            );
            assert!(
                batch_evals_failed.load(Ordering::Relaxed) >= 1,
                "shared_graph_poison={shared_graph_poison}: batched device eval must fail and enter isolation"
            );
            let err = lane_41_failure.expect("failing lane 41 must report error");
            assert!(
                err.reason
                    .contains("injected deferred device draft eval failure"),
                "unexpected lane 41 error: {}",
                err.reason
            );
            assert_eq!(
                lane_42_tokens.expect("healthy peer lane 42 must complete"),
                baseline[1].1,
                "shared_graph_poison={shared_graph_poison}: healthy peer lane 42 must match baseline exactly"
            );
            let retried = solo_retried.lock().unwrap();
            assert!(
                retried.contains(&41),
                "failing lane 41 must attempt solo retry before failing"
            );
            assert!(
                retried.contains(&42),
                "shared_graph_poison={shared_graph_poison}: healthy lane 42 must rebuild solo rather than re-evaluating handles from a failed batch command"
            );
        }
    }
}
