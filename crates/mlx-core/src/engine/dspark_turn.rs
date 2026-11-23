//! Engine-owned DSpark (draft-model speculative decode) whole-turn loop —
//! the DSpark analog of [`crate::engine::mtp_turn`]. Families opt in via
//! [`crate::engine::backend::DsparkBackend`]; their `ChatBackend::run_speculative_turn`
//! override calls [`run_dspark_turn`].
//!
//! Structure mirrors [`crate::engine::mtp_turn::run_mtp_turn`] (the
//! initial-`y` emit guarded by budget, the pre-cycle stop-check order, the
//! finish-reason strings, the emit-block bookkeeping, the
//! every-256-emitted-token cache clear). The cycle body is DSpark's own
//! propose → verify → accept → STOP-CLAMP → commit → emit sequence: unlike
//! MTP there is NO post-commit rollback/heal path, so every stop condition
//! is resolved BEFORE [`crate::engine::backend::DsparkStepper::commit`] and
//! the stepper commits exactly once per cycle.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use napi::bindgen_prelude::*;

use crate::array::{DType, MxArray};
use crate::decode_profiler::DecodeProfiler;
use crate::engine::backend::{
    DecodeStep, DsparkBackend, DsparkProposal, DsparkStepper, FinalizeArgs, PagedBackend,
    PagedPrefix, TurnOutput, TurnTokenObserver, WholeTurnArgs,
};
use crate::engine::decode::{StreamingCtx, TurnStreaming};
use crate::engine::dspark_timeline::{self as timeline, Mark};
use crate::engine::finalize::compute_performance_metrics;
use crate::engine::paged_turn::FinishPagedTurnArgs;
use crate::engine::params::{ChatParams, generated_capacity_hint, resolve_enable_thinking};
use crate::engine::penalties::{CutTargets, ReasoningTracker, ThinkingCut, apply_all_penalties};
use crate::engine::plan::SpeculativeDraftWidth;
use crate::engine::spec_paged::SpecTurnEpilogue;
use crate::sampling;
use crate::stream::Stream;

/// Before the first rejection, row i's history is exactly `history + drafts[..i]`.
/// Prepare every deterministic decision from that prefix, then cross the device
/// boundary once. Decisions after the first mismatch are discarded. This must
/// not be used for stochastic acceptance, whose residual draws are sequential.
/// A row `forced` decides takes its token from there.
fn penalized_greedy_accept(
    logits: &MxArray,
    drafts: &[i32],
    history: &[u32],
    params: &ChatParams,
    forced: Option<&CutTargets>,
) -> Result<(usize, u32)> {
    let vocab = logits.shape_at(2)?;
    let mut prefix = history.to_vec();
    let mut decisions = Vec::with_capacity(drafts.len() + 1);
    for i in 0..=drafts.len() {
        let row = logits
            .slice(&[0, i as i64, 0], &[1, i as i64 + 1, vocab])?
            .squeeze(Some(&[0, 1]))?;
        decisions.push(apply_all_penalties(row, &prefix, params)?.argmax(0, None)?);
        if let Some(&draft) = drafts.get(i) {
            prefix.push(draft as u32);
        }
    }
    MxArray::eval_arrays_with_context(
        &decisions.iter().collect::<Vec<_>>(),
        "speculative_greedy_accept",
    )?;
    timeline::mark(Mark::VerifyWait);
    for (i, decision) in decisions.iter().enumerate() {
        let target = match forced.and_then(|f| f.at(i)) {
            Some(token) => token as i32,
            None => decision.item_at_int32(0)?,
        };
        if drafts.get(i) != Some(&target) {
            return Ok((i, target as u32));
        }
    }
    unreachable!("the bonus row has no corresponding draft")
}

/// Budgeted greedy acceptance never reads target rows outside the valid prefix.
/// The final valid row can label one further draft, but its prediction must be
/// emitted as the boundary because its own cache slot is not part of the prefix.
fn accept_device_budgeted_greedy(
    logits: &MxArray,
    proposal: &mut DsparkProposal,
    budget: &MxArray,
) -> Result<(usize, u32, usize, usize)> {
    let draft_len = proposal.draft_len()?;
    if logits.ndim()? != 3
        || logits.shape_at(0)? != 1
        || logits.shape_at(1)? != (draft_len + 1) as i64
        || budget.dtype()? != DType::Int32
        || budget.size()? != 1
    {
        return Err(Error::from_reason(
            "invalid device verification budget contract",
        ));
    }
    let argmax = logits.argmax(-1, None)?;
    MxArray::eval_arrays_with_context(&[&argmax, budget], "speculative_budgeted_greedy_accept")?;
    timeline::mark(Mark::VerifyWait);
    let rows = usize::try_from(budget.item_at_int32(0)?)
        .map_err(|_| Error::from_reason("negative device verification row budget"))?;
    if rows == 0 || rows > draft_len + 1 {
        return Err(Error::from_reason(
            "device verification budget exceeds physical rows",
        ));
    }
    proposal.materialize_draft_ids()?;
    let mut raw_matches = 0;
    while raw_matches < rows.min(draft_len)
        && argmax.item_at_int32(raw_matches)? == proposal.draft_ids[raw_matches]
    {
        raw_matches += 1;
    }
    let accepted = raw_matches.min(rows - 1);
    Ok((
        accepted,
        argmax.item_at_int32(accepted)? as u32,
        rows,
        raw_matches,
    ))
}

#[cfg(test)]
thread_local! {
    /// Sampled acceptances on this thread: (device graph, host walk).
    pub(crate) static TEST_SAMPLED_ACCEPTS: std::cell::Cell<(usize, usize)> =
        const { std::cell::Cell::new((0, 0)) };
}

/// Shared whole-turn and scheduler acceptance. Probability/penalty work stays
/// on device; only proposal and boundary token ids cross to the engine.
///
/// `proposal` is `&mut` for [`DsparkProposal::materialize_draft_ids`]: a
/// device-resident proposal is back-filled to host ids INSIDE this function,
/// after the acceptance eval has forced the shared verify graph — never
/// earlier (an early read would reintroduce the proposal-boundary sync the
/// device path exists to remove).
///
/// `cut` decides the rows at and after a thinking cut: each selects its end
/// token, with no draw.
fn accept_dspark_greedy_argmax(
    argmax_arr: &MxArray,
    proposal: &mut DsparkProposal,
    cut: Option<&ThinkingCut>,
) -> Result<(usize, u32)> {
    let draft_len = proposal.draft_len()?;
    MxArray::eval_arrays_with_context(&[argmax_arr], "speculative_greedy_accept")?;
    timeline::mark(Mark::VerifyWait);
    // The verify graph (incl. a device-resident proposal path) is
    // now materialized: the host back-fill below is a plain copy.
    proposal.materialize_draft_ids()?;
    let forced = cut_targets(cut, &proposal.draft_ids);
    let mut target_argmax: Vec<i32> = Vec::with_capacity(draft_len + 1);
    for i in 0..=draft_len {
        target_argmax.push(match forced.as_ref().and_then(|f| f.at(i)) {
            Some(token) => token as i32,
            None => argmax_arr.item_at_int32(i)?,
        });
    }
    let mut k = 0usize;
    while k < draft_len && target_argmax[k] == proposal.draft_ids[k] {
        k += 1;
    }
    Ok((k, target_argmax[k] as u32))
}

pub(crate) fn accept_dspark_proposal(
    logits: &MxArray,
    proposal: &mut DsparkProposal,
    hist: &[u32],
    p: &ChatParams,
    rng: &mut dyn rand::Rng,
    cut: Option<&ThinkingCut>,
) -> Result<(usize, u32)> {
    let draft_len = proposal.draft_len()?;
    if logits.ndim()? != 3
        || logits.shape_at(0)? != 1
        || logits.shape_at(1)? != (draft_len + 1) as i64
    {
        return Err(Error::from_reason(
            "speculative verifier logits do not match proposal width",
        ));
    }
    // Turn-constant accept-policy gates (params are fixed for the turn).
    // At GREEDY temperature acceptance is ALWAYS argmax-based and never
    // reads proposal dists or consumes RNG (the DsparkProposal contract
    // ships empty `draft_dists` there); `penalties_no_op` only selects
    // between the batched-argmax fast path and the per-row penalized-argmax
    // path. This matches `run_mtp_cycle`'s T=0-with-penalties behavior: its
    // legacy dense branch feeds the PENALIZED row into
    // `accept_with_residual`, whose T=0 shortcut reduces to exactly
    // "penalized argmax == draft id" with an argmax boundary.
    let temperature = p.sampling_config.and_then(|c| c.temperature).unwrap_or(1.0);
    let sampling_cfg = p.sampling_config.unwrap_or_default();
    let penalties_no_op =
        p.repetition_penalty == 1.0 && p.presence_penalty == 0.0 && p.frequency_penalty == 0.0;
    let greedy_temp = sampling::is_greedy_temperature(temperature);
    let greedy_fast = greedy_temp && penalties_no_op;

    if greedy_fast {
        // Greedy fast path (penalties no-op): ONE batched argmax over
        // all 1+L rows, a single eval, then CPU reads. No RNG consumed
        // (mirrors `accept_with_residual`'s T=0 shortcut).
        let argmax_arr = logits.argmax(-1, None)?;
        accept_dspark_greedy_argmax(&argmax_arr, proposal, cut)
    } else if greedy_temp {
        // Host ids feed the penalized history prefix before the eval, so a
        // device path (not produced under active penalties) must land first.
        proposal.materialize_draft_ids()?;
        let forced = cut_targets(cut, &proposal.draft_ids);
        penalized_greedy_accept(logits, &proposal.draft_ids, hist, p, forced.as_ref())
    } else {
        // Sampled path: per-position Leviathan accept + residual
        // resample against the stepper's proposal densities.
        #[cfg(test)]
        TEST_SAMPLED_ACCEPTS.with(|c| c.set((c.get().0, c.get().1 + 1)));
        (|| {
            // Sampled accept consumes host proposal ids and rows at graph-build
            // time; a device-drawn proposal lands here first.
            proposal.materialize_draft_ids()?;
            let forced = cut_targets(cut, &proposal.draft_ids);
            if forced.is_some() {
                // A cycle whose rows are all decided reads no target; the
                // verify graph is still forced before its anchor's cache state
                // is kept.
                MxArray::eval_arrays_with_context(&[logits], "speculative_cut_accept")?;
                timeline::mark(Mark::VerifyWait);
            }
            let forced_at = |row: usize| forced.as_ref().and_then(|f| f.at(row));
            // A zero-draft cycle is the intentional AR-through-verify
            // fallback: it owns no proposal rows and samples only the
            // verifier boundary below. Enforce exclusive dense/sparse
            // proposal ownership only when there is a drafted token.
            let has_dense = draft_len > 0 && proposal.draft_dists.len() == draft_len;
            let has_sparse = draft_len > 0 && proposal.draft_sparse_dists.len() == draft_len;
            if draft_len > 0 && has_dense == has_sparse {
                return Err(Error::from_reason(format!(
                    "DSpark sampled accept requires exactly one dense or sparse proposal \
                         distribution per drafted token (got {} dense, {} sparse for {} drafts)",
                    proposal.draft_dists.len(),
                    proposal.draft_sparse_dists.len(),
                    draft_len
                )));
            }
            let vocab = logits.shape_at(2)?;
            let mut hist_extended: Vec<u32> = hist.to_vec();
            // Up to the first rejection, the history is the known draft
            // prefix. Construct deterministic target distributions together
            // without drawing acceptance coins or residual samples early.
            let mut targets = Vec::with_capacity(draft_len);
            for (i, &draft_id) in proposal.draft_ids.iter().enumerate().take(draft_len) {
                let v_slice = logits.slice(&[0, i as i64, 0], &[1, (i + 1) as i64, vocab])?;
                let v_1d = v_slice.squeeze(Some(&[0, 1]))?;
                let penalized = apply_all_penalties(v_1d, &hist_extended, p)?;
                targets.push(
                    sampling::sampling_distribution(&penalized, p.sampling_config)?
                        .astype(DType::Float32)?,
                );
                hist_extended.push(draft_id as u32);
            }
            let roots = targets
                .iter()
                .chain(proposal.draft_dists.iter())
                .collect::<Vec<_>>();
            MxArray::eval_arrays_with_context(&roots, "speculative_sampled_accept")?;
            timeline::mark(Mark::VerifyWait);
            let mut k = 0usize;
            let mut boundary: Option<u32> = None;
            for (i, p_target) in targets.iter().enumerate() {
                let (accept, out_tok) = if let Some(token) = forced_at(i) {
                    (proposal.draft_ids[i] == token as i32, token as i32)
                } else if has_sparse {
                    sampling::accept_with_residual_dense_target_sparse_draft(
                        p_target,
                        proposal.draft_sparse_dists[i].as_row(),
                        proposal.draft_ids[i],
                        rng,
                    )?
                } else {
                    sampling::accept_with_residual(
                        p_target,
                        &proposal.draft_dists[i],
                        proposal.draft_ids[i],
                        &sampling_cfg,
                        rng,
                    )?
                };
                if accept {
                    k += 1;
                } else {
                    boundary = Some(out_tok as u32);
                    break;
                }
            }
            let boundary_id = match boundary.or_else(|| forced_at(draft_len)) {
                Some(b) => b,
                None => {
                    // All L drafts accepted: bonus sample from row L.
                    let i = draft_len;
                    let v_slice = logits.slice(&[0, i as i64, 0], &[1, (i + 1) as i64, vocab])?;
                    let v_1d = v_slice.squeeze(Some(&[0, 1]))?;
                    let penalized = apply_all_penalties(v_1d, &hist_extended, p)?;
                    let distribution =
                        sampling::sampling_distribution(&penalized, p.sampling_config)?
                            .astype(DType::Float32)?;
                    sampling::sample_dense_distribution(&distribution, rng)? as u32
                }
            };
            Ok((k, boundary_id))
        })()
    }
}

/// Sampled acceptance of a device-drawn proposal in one device graph (Splash
/// `accept_sampled_lane`, runtime/metal/kernels/decode/sampling.metal,
/// Apache-2.0): draft `d_i` stays while `u_i * q_i(d_i) < p_i(d_i)`; the first
/// refused row emits its draw from `(p_i - q_i)+` (its argmax when that has
/// no mass), and a fully accepted block emits the bonus row's draw from
/// `p_L`. Every coin and draw key is taken from `rng` before the graph, the
/// same count each cycle; unused ones are discarded.
pub(crate) fn accept_device_sampled(
    logits: &MxArray,
    proposal: &mut DsparkProposal,
    p: &ChatParams,
    rng: &mut dyn rand::Rng,
) -> Result<(usize, u32)> {
    use rand::RngExt;
    let draft_len = proposal.draft_len()?;
    let temperature = p.sampling_config.and_then(|c| c.temperature).unwrap_or(1.0);
    if sampling::is_greedy_temperature(temperature)
        || p.repetition_penalty != 1.0
        || p.presence_penalty != 0.0
        || p.frequency_penalty != 0.0
    {
        return Err(Error::from_reason(
            "device sampled acceptance requires an unpenalized sampled turn",
        ));
    }
    let (Some(ids), Some(rows)) = (
        proposal.device_draft_ids.as_ref(),
        proposal.device_draft_sparse.as_ref(),
    ) else {
        return Err(Error::from_reason(
            "device sampled acceptance needs a device-drawn proposal",
        ));
    };
    if draft_len == 0
        || logits.ndim()? != 3
        || logits.shape_at(0)? != 1
        || logits.shape_at(1)? != (draft_len + 1) as i64
        || rows.ids.shape_at(0)? != draft_len as i64
    {
        return Err(Error::from_reason(
            "speculative verifier logits do not match proposal width",
        ));
    }
    #[cfg(test)]
    TEST_SAMPLED_ACCEPTS.with(|c| c.set((c.get().0 + 1, c.get().1)));
    let vocab = logits.shape_at(2)?;
    let len = draft_len as i64;
    let mut targets = Vec::with_capacity(draft_len + 1);
    for i in 0..=len {
        let row = logits
            .slice(&[0, i, 0], &[1, i + 1, vocab])?
            .squeeze(Some(&[0, 1]))?;
        targets.push(
            sampling::sampling_distribution(&row, p.sampling_config)?.astype(DType::Float32)?,
        );
    }
    let coins: Vec<f32> = (0..draft_len).map(|_| rng.random::<f32>()).collect();
    let coins = MxArray::from_float32(&coins, &[len])?;
    let drafted = MxArray::stack(targets[..draft_len].iter().collect(), Some(0))?;
    let column = ids.reshape(&[len, 1])?;
    let p_draft = drafted.take_along_axis(&column, 1)?.reshape(&[len])?;
    let q_draft = rows
        .probs
        .mul(&rows.ids.equal(&column)?.astype(DType::Float32)?)?
        .sum(Some(&[1]), None)?;
    let kept = coins.mul(&q_draft)?.less(&p_draft)?;
    let refused = MxArray::concatenate(
        &kept.logical_not()?.astype(DType::Int32)?,
        &MxArray::from_int32(&[1], &[1])?,
        0,
    )?;
    let accepted = refused.argmax(0, None)?;
    let residual = drafted
        .sub(
            &MxArray::zeros(&[len, vocab], Some(DType::Float32))?.put_along_axis(
                &rows.ids,
                &rows.probs,
                1,
            )?,
        )?
        .clip(Some(0.0), None)?;
    let mut draws = Vec::with_capacity(draft_len + 1);
    for i in 0..len {
        let row = residual.slice_axis(0, i, i + 1)?.reshape(&[vocab])?;
        draws.push(sampling::residual_draw_array(&targets[i as usize], &row, rng)?.reshape(&[1])?);
    }
    draws.push(sampling::sample_dense_distribution_array(&targets[draft_len], rng)?.reshape(&[1])?);
    let boundary =
        MxArray::concatenate_many(draws.iter().collect(), Some(0))?.take(&accepted, 0)?;
    MxArray::eval_arrays_with_context(
        &[&accepted, &boundary],
        "speculative_device_sampled_accept",
    )?;
    timeline::mark(Mark::VerifyWait);
    let k = usize::try_from(accepted.item_at_int32(0)?)
        .map_err(|_| Error::from_reason("device sampled acceptance: negative count"))?;
    let boundary = boundary.item_at_int32(0)?;
    if boundary < 0 {
        return Err(Error::from_reason(
            "speculative target distribution has no positive probability mass",
        ));
    }
    proposal.device_draft_sparse = None;
    proposal.materialize_draft_ids()?;
    Ok((k, boundary as u32))
}

/// The largest `top_k` the device rows take (`DFLASH2_ROWS_MAX_TOP_K`).
const SAMPLED_ROWS_MAX_TOP_K: i32 = 256;
/// The widest draft candidate row (`DFLASH2_ROWS_MAX_CANDIDATES`).
const SAMPLED_ROWS_MAX_CANDIDATES: usize = 64;
/// A record holds token ids as f32.
const SAMPLED_ROWS_MAX_VOCAB: i64 = 1 << 24;
const SAMPLED_ROWS_RECORD: usize = 4;

#[cfg(test)]
thread_local! {
    /// Cycles on this thread whose target rows the device kernels decided.
    pub(crate) static TEST_SAMPLED_ROWS: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
}

/// Whether the device rows take a turn with these parameters: sampled, no
/// penalty, `top_k` within the kernels' list, mlx-node's filter order, a GPU
/// that launches the kernels.
pub(crate) fn sampled_rows_take(p: &ChatParams) -> bool {
    let config = p.sampling_config.unwrap_or_default();
    !sampling::is_greedy_temperature(config.temperature.unwrap_or(1.0))
        && p.repetition_penalty == 1.0
        && p.presence_penalty == 0.0
        && p.frequency_penalty == 0.0
        && (1..=SAMPLED_ROWS_MAX_TOP_K).contains(&config.top_k.unwrap_or(0))
        && sampling::default_filter_order()
        && unsafe { mlx_sys::mlx_dflash2_sampled_rows_supported() }
}

/// A cycle's target rows, queued on the device: one record per verify row
/// and the cycle's acceptance coins.
pub(crate) struct SampledRows {
    records: MxArray,
    coins: Vec<f64>,
}

/// Queues the target rows of a sampled DFlash2 cycle (Splash
/// `Sampling::addSelection`, runtime/ops/Sampling.cpp, Apache-2.0): every
/// row's distribution under the turn's temperature, top_k, top_p and min_p,
/// each drafted row's draft token probabilities and its draw from the
/// residual, and the last row's draw from the target. `logits` holds the
/// `1 + L` rows of the cycle. One draw uniform per row and one coin per
/// drafted row come from `rng` before the dispatch, the same count each
/// cycle. `None`: the kernels do not take the cycle and `rng` is untouched.
pub(crate) fn submit_sampled_rows(
    logits: &MxArray,
    proposal: &DsparkProposal,
    p: &ChatParams,
    rng: &mut dyn rand::Rng,
) -> Result<Option<SampledRows>> {
    use rand::RngExt;
    if !sampled_rows_take(p) {
        return Ok(None);
    }
    let draft_len = proposal.draft_len()?;
    let rows = draft_len + 1;
    let ndim = logits.ndim()?;
    let vocab = if ndim == 0 {
        0
    } else {
        logits.shape_at(ndim - 1)?
    };
    if vocab == 0 || logits.size()? != rows as u64 * vocab as u64 {
        return Err(Error::from_reason(
            "speculative verifier logits do not match proposal width",
        ));
    }
    if vocab > SAMPLED_ROWS_MAX_VOCAB || rows > 64 {
        return Ok(None);
    }
    let drafts = if draft_len == 0 {
        None
    } else if let (Some(ids), Some(sparse)) = (
        proposal.device_draft_ids.as_ref(),
        proposal.device_draft_sparse.as_ref(),
    ) {
        if sparse.ids.ndim()? != 2
            || sparse.ids.shape_at(0)? != draft_len as i64
            || sparse.ids.shape_at(1)? > SAMPLED_ROWS_MAX_CANDIDATES as i64
        {
            return Ok(None);
        }
        Some((
            ids.astype(DType::Int32)?,
            sparse.ids.astype(DType::Int32)?,
            sparse.probs.astype(DType::Float32)?,
        ))
    } else if proposal.device_draft_ids.is_none() && proposal.draft_sparse_dists.len() == draft_len
    {
        let width = proposal
            .draft_sparse_dists
            .iter()
            .map(|row| row.candidates().0.len())
            .max()
            .unwrap_or(0);
        if width == 0 || width > SAMPLED_ROWS_MAX_CANDIDATES {
            return Ok(None);
        }
        let mut candidate_ids = vec![-1i32; draft_len * width];
        let mut candidate_probs = vec![0f32; draft_len * width];
        for (row, sparse) in proposal.draft_sparse_dists.iter().enumerate() {
            let (ids, probs) = sparse.candidates();
            candidate_ids[row * width..row * width + ids.len()].copy_from_slice(ids);
            for (slot, &prob) in probs.iter().enumerate() {
                candidate_probs[row * width + slot] = prob as f32;
            }
        }
        let shape = [draft_len as i64, width as i64];
        Some((
            MxArray::from_int32(&proposal.draft_ids, &[draft_len as i64])?,
            MxArray::from_int32(&candidate_ids, &shape)?,
            MxArray::from_float32(&candidate_probs, &shape)?,
        ))
    } else {
        return Ok(None);
    };
    let config = p.sampling_config.unwrap_or_default();
    let draws: Vec<f32> = (0..rows).map(|_| rng.random::<f32>()).collect();
    let coins: Vec<f64> = (0..draft_len).map(|_| rng.random::<f64>()).collect();
    let logits = logits
        .reshape(&[rows as i64, vocab])?
        .astype(DType::Float32)?;
    let uniforms = MxArray::from_float32(&draws, &[rows as i64])?;
    let null = std::ptr::null_mut();
    let (ids, candidates, probabilities) = match &drafts {
        Some((ids, candidates, probabilities)) => {
            (ids.handle.0, candidates.handle.0, probabilities.handle.0)
        }
        None => (null, null, null),
    };
    let mut output = std::ptr::null_mut();
    let taken = unsafe {
        mlx_sys::mlx_dflash2_sampled_rows(
            logits.handle.0,
            uniforms.handle.0,
            ids,
            candidates,
            probabilities,
            config.temperature.unwrap_or(1.0) as f32,
            config.top_k.unwrap_or(0),
            config.top_p.unwrap_or(1.0) as f32,
            config.min_p.unwrap_or(0.0) as f32,
            &mut output,
        )
    };
    if taken != 1 {
        return Err(Error::from_reason(
            "sampled DFlash2 rows: the device kernels refused a checked cycle",
        ));
    }
    let records = MxArray::from_handle(output, "dflash2_sampled_rows")?;
    #[cfg(test)]
    TEST_SAMPLED_ROWS.with(|c| c.set(c.get() + 1));
    Ok(Some(SampledRows { records, coins }))
}

impl SampledRows {
    /// The lazy records, for a caller that forces several cycles at once.
    pub(crate) fn records(&self) -> &MxArray {
        &self.records
    }

    /// The cycle's accepted draft count and boundary token, by the
    /// acceptance law of [`accept_dspark_proposal`]: draft `d_i` stays while
    /// its coin is below `min(1, p_i(d_i) / q_i(d_i))`; the first refused row
    /// emits its residual draw, a fully accepted block the last row's draw.
    /// `cut` decides its rows with no coin. Reads the records, which forces
    /// them and the verify graph when the caller has not.
    pub(crate) fn accept(
        self,
        proposal: &mut DsparkProposal,
        cut: Option<&ThinkingCut>,
    ) -> Result<(usize, u32)> {
        let records = self.records.to_float32()?;
        proposal.device_draft_sparse = None;
        proposal.materialize_draft_ids()?;
        let draft_len = proposal.draft_ids.len();
        if self.coins.len() != draft_len || records.len() != (draft_len + 1) * SAMPLED_ROWS_RECORD {
            return Err(Error::from_reason(
                "sampled DFlash2 rows do not match the proposal",
            ));
        }
        let forced = cut_targets(cut, &proposal.draft_ids);
        let forced_at = |row: usize| forced.as_ref().and_then(|f| f.at(row));
        let drawn = |row: usize| {
            let token = records[row * SAMPLED_ROWS_RECORD + 2];
            if token >= 0.0 {
                Ok(token as u32)
            } else {
                Err(Error::from_reason(
                    "speculative target distribution has no positive probability mass",
                ))
            }
        };
        for row in 0..draft_len {
            if let Some(token) = forced_at(row) {
                if proposal.draft_ids[row] == token as i32 {
                    continue;
                }
                return Ok((row, token));
            }
            let target = f64::from(records[row * SAMPLED_ROWS_RECORD]);
            let draft = f64::from(records[row * SAMPLED_ROWS_RECORD + 1]);
            if self.coins[row] < sampling::acceptance_probability_from_probs(target, draft) {
                continue;
            }
            return Ok((row, drawn(row)?));
        }
        match forced_at(draft_len) {
            Some(token) => Ok((draft_len, token)),
            None => Ok((draft_len, drawn(draft_len)?)),
        }
    }
}

/// The rows `cut` decides for `drafts`.
fn cut_targets(cut: Option<&ThinkingCut>, drafts: &[i32]) -> Option<CutTargets> {
    cut.map(|c| c.targets(drafts))
}

/// Required arguments of [`run_dspark_turn`] — the DSpark analog of
/// [`crate::engine::mtp_turn::MtpTurnArgs`], minus the MTP-only
/// prompt-hidden seed fields (the DSpark stepper owns its draft-context
/// seeding) and plus the draft `block_size`. Constructed in production by
/// [`run_paged_dspark_turn`] and by gemma4's `flat_draft_chat_turn`.
pub(crate) struct DsparkTurnArgs<'a> {
    /// First generated token (sampled from the prefill logits BEFORE the
    /// turn). The loop takes ownership and emits it first (guarded by the
    /// budget), exactly as `run_mtp_turn` emits its initial `y`.
    pub y: MxArray,
    /// Draft block size — the hard per-cycle cap on drafted tokens. The
    /// per-cycle cap is `min(block_size, params.mtp_depth, remaining - 1)`.
    pub block_size: usize,
    pub params: &'a ChatParams,
    pub reasoning_tracker: &'a mut ReasoningTracker,
    pub profiler: &'a mut DecodeProfiler,
    pub max_new_tokens: i32,
    pub eos_id: u32,
    pub generated_tokens: &'a mut Vec<u32>,
    pub token_history: &'a mut Vec<u32>,
    pub finish_reason: &'a mut String,
    pub first_token_instant: &'a mut Option<Instant>,
    pub report_perf: bool,
    pub generation_stream: Stream,
    /// Per-turn cooperative cancel flag (H2) — the family core's clone of
    /// the backend-installed `turn_cancel`. Polled at the SAME per-cycle
    /// snapshot points the STREAMING path already polls (the pre-cycle
    /// check and the stop-clamp simulation), so a disconnected
    /// non-streaming client stops a DSpark decode instead of holding the
    /// model mutex to budget exhaustion. `None` ⇒ legacy behavior.
    /// Streaming turns carry the SAME flag in `StreamingCtx.cancelled`;
    /// the polls are idempotent, so double-polling is harmless.
    pub cancel_flag: Option<&'a AtomicBool>,
    /// Family protocol guard, present independently of a streaming sink.
    pub turn_token_observer: Option<Box<dyn TurnTokenObserver>>,
}

/// Terminal outs of [`run_dspark_turn`] the caller threads into its save /
/// next-turn bookkeeping. Minimal mirror of
/// [`crate::engine::mtp_turn::MtpTurnOutcome`]: DSpark has no flat-desync
/// side channel (the stop-clamp runs BEFORE commit, so the target and draft
/// caches can never desync), so only `last_in_cache` is surfaced.
pub(crate) struct DsparkTurnOutcome {
    /// Whether the LAST emitted token's K/V is already in the target cache.
    /// `true` iff the last emitted token is a KEPT accepted draft (verify
    /// wrote its slot and commit kept it); `false` when it is a cycle's
    /// boundary token (bonus/residual — never written this cycle; it only
    /// gains K/V as the NEXT cycle's verify anchor) AND on every in-cycle
    /// stop cut — the stop-clamp's AR-parity exclusion never keeps the
    /// tripping token's slot, so every stop exit reports `false`, exactly
    /// like the AR reference (which never forwards its final token on a
    /// non-length stop). The save uses `drop_last_always = !last_in_cache`.
    pub last_in_cache: bool,
}

/// In-cycle stop resolved by the pre-commit stop-clamp. Reason strings
/// byte-match `run_mtp_turn`'s.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CycleStop {
    Length,
    Stop,
    Cancelled,
    Repetition(&'static str),
}

pub(crate) struct DsparkCycleDecision {
    pub emit_count: usize,
    pub keep: usize,
    pub stop: Option<CycleStop>,
}

/// Resolve every stop before committing any verifier row. The engine invokes
/// observers here exactly once, then emits the already-checked prefix.
pub(crate) fn clamp_dspark_cycle(
    generated: &[u32],
    accepted: &[u32],
    accepted_drafts_k: usize,
    p: &ChatParams,
    max_as_usize: usize,
    eos_id: u32,
    mut cancelled: impl FnMut() -> bool,
    mut observe: impl FnMut(u32) -> bool,
) -> DsparkCycleDecision {
    let mut emit_count: usize = 0;
    let mut stop: Option<CycleStop> = None;
    {
        let mut sim: Vec<u32> = Vec::with_capacity(generated.len() + accepted.len());
        sim.extend_from_slice(generated);
        for (idx, &tok) in accepted.iter().enumerate() {
            // Defensive: with the over-return check above, a cycle
            // emits at most `l_cap + 1 <= remaining` tokens, so this
            // budget arm is unreachable for compliant flows. Kept for
            // byte-parity with mtp's emit-loop guard and as
            // defense-in-depth against future L_cap-math regressions.
            if sim.len() >= max_as_usize {
                stop = Some(CycleStop::Length);
                break;
            }
            sim.push(tok);
            emit_count = idx + 1;
            // H2: `turn_cancelled()` extends the SAME stop-clamp poll
            // to sync turns; the streaming read is kept so streaming
            // behavior never depends on `cancel_flag` also being wired.
            if cancelled() {
                stop = Some(CycleStop::Cancelled);
                break;
            }
            if observe(tok) {
                stop = Some(CycleStop::Stop);
                break;
            }
            if tok == eos_id || p.extra_eos_ids.contains(&tok) {
                stop = Some(CycleStop::Stop);
                break;
            }
            if let Some(reason_str) = crate::sampling::check_repetition_cutoff(
                &sim,
                p.max_consecutive_tokens,
                p.max_ngram_repeats,
                p.ngram_size,
            ) {
                stop = Some(CycleStop::Repetition(reason_str));
                break;
            }
        }
    }

    // Anchor's K/V was written at verify position 0; the boundary has
    // NO K/V (it becomes the next cycle's anchor), so it never counts
    // toward `keep` — hence `min(emit_count, k)` counts only kept
    // accepted-draft slots.
    //
    // AR-PARITY on in-cycle stops: an in-cycle stop token (EOS /
    // cancel-observed / repetition-tripping — always the LAST emitted
    // token) is NEVER committed to the target cache. The AR reference
    // never forwards its final token on a non-length stop and the
    // family saves drop it from the persisted history, so keeping its
    // slot here would leave the speculative session one K/V (and one
    // history token, via `last_in_cache`) AHEAD of the AR session for
    // the same transcript. When the clamp cut at an ACCEPTED DRAFT
    // (`emit_count <= k` — Stop/Cancelled/Repetition always follow a
    // push, so `emit_count >= 1` there), exclude the tripping token's
    // slot; a boundary-stop cut changes nothing (no slot to exclude).
    let kept_drafts = match stop {
        Some(CycleStop::Stop) | Some(CycleStop::Cancelled) | Some(CycleStop::Repetition(_))
            if emit_count <= accepted_drafts_k =>
        {
            emit_count.saturating_sub(1)
        }
        _ => emit_count.min(accepted_drafts_k),
    };
    let keep = 1 + kept_drafts;
    DsparkCycleDecision {
        emit_count,
        keep,
        stop,
    }
}

/// Number of real speculative cycles used to compare DSpark against one
/// target-only probe on this exact model, context, and host. Two samples
/// damp the first-cycle lazy-evaluation/warmup skew without making a losing
/// draft path dominate a short agent response.
const DSPARK_BREAK_EVEN_SPEC_CYCLES: u8 = 2;

/// Hysteresis above measured target-only throughput. A tie is not enough to
/// retain the draft: its acceptance and latency can drift with prompt domain
/// and context length, so speculation must clear the probe by 5%.
const DSPARK_BREAK_EVEN_MIN_SPEEDUP: f64 = 1.05;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DsparkMeasurementCycle {
    None,
    ArProbe,
    SpecProbe,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DsparkBreakEvenChoice {
    KeepSpeculation,
    FallBackToAr,
}

#[derive(Clone, Copy, Debug)]
struct DsparkBreakEvenDecision {
    choice: DsparkBreakEvenChoice,
    ar_tokens_per_second: f64,
    spec_tokens_per_second: f64,
    spec_tokens: u64,
    spec_cycles: u8,
}

#[derive(Clone, Copy, Debug)]
enum DsparkBreakEvenState {
    Disabled,
    ArProbe,
    SpecProbe {
        ar_wall_ns: u64,
        spec_wall_ns: u128,
        spec_tokens: u64,
        spec_cycles: u8,
    },
    KeepSpeculation,
    ArFallback,
}

/// Per-turn, hardware-measured break-even guard for DSpark.
///
/// The DeepSpec latency objective is accepted tokens divided by draft plus
/// target-verify time. That break-even point is host- and context-dependent,
/// so the guard first measures one exact target-only cycle, then two ordinary
/// DSpark cycles. A losing draft permanently yields to the target's normal AR
/// forward for the remainder of the turn. The pure state machine is kept
/// separate from the model stepper so its decision math is deterministic in
/// unit tests.
#[derive(Clone, Copy, Debug)]
struct DsparkBreakEvenPolicy {
    state: DsparkBreakEvenState,
}

impl DsparkBreakEvenPolicy {
    fn new(enabled: bool) -> Self {
        Self {
            state: if enabled {
                DsparkBreakEvenState::ArProbe
            } else {
                DsparkBreakEvenState::Disabled
            },
        }
    }

    /// Select the actual draft cap and label a calibration cycle. A natural
    /// tail cycle (`normal_cap == 0`) is never mistaken for the AR probe.
    fn plan_cycle(&self, normal_cap: usize) -> (usize, DsparkMeasurementCycle) {
        if normal_cap == 0 {
            return (0, DsparkMeasurementCycle::None);
        }
        match self.state {
            DsparkBreakEvenState::Disabled | DsparkBreakEvenState::KeepSpeculation => {
                (normal_cap, DsparkMeasurementCycle::None)
            }
            DsparkBreakEvenState::ArProbe => (0, DsparkMeasurementCycle::ArProbe),
            DsparkBreakEvenState::SpecProbe { .. } => {
                (normal_cap, DsparkMeasurementCycle::SpecProbe)
            }
            DsparkBreakEvenState::ArFallback => (0, DsparkMeasurementCycle::None),
        }
    }

    fn record_cycle(
        &mut self,
        kind: DsparkMeasurementCycle,
        wall_ns: u64,
        committed_tokens: u32,
    ) -> Option<DsparkBreakEvenDecision> {
        let wall_ns = wall_ns.max(1);
        match (self.state, kind) {
            (DsparkBreakEvenState::ArProbe, DsparkMeasurementCycle::ArProbe) => {
                self.state = DsparkBreakEvenState::SpecProbe {
                    ar_wall_ns: wall_ns,
                    spec_wall_ns: 0,
                    spec_tokens: 0,
                    spec_cycles: 0,
                };
                None
            }
            (
                DsparkBreakEvenState::SpecProbe {
                    ar_wall_ns,
                    spec_wall_ns,
                    spec_tokens,
                    spec_cycles,
                },
                DsparkMeasurementCycle::SpecProbe,
            ) => {
                let spec_wall_ns = spec_wall_ns + u128::from(wall_ns);
                let spec_tokens = spec_tokens + u64::from(committed_tokens);
                let spec_cycles = spec_cycles + 1;
                if spec_cycles < DSPARK_BREAK_EVEN_SPEC_CYCLES {
                    self.state = DsparkBreakEvenState::SpecProbe {
                        ar_wall_ns,
                        spec_wall_ns,
                        spec_tokens,
                        spec_cycles,
                    };
                    return None;
                }

                let ar_tokens_per_second = 1.0e9 / ar_wall_ns as f64;
                let spec_tokens_per_second = spec_tokens as f64 * 1.0e9 / spec_wall_ns as f64;
                let choice = if spec_tokens_per_second
                    >= ar_tokens_per_second * DSPARK_BREAK_EVEN_MIN_SPEEDUP
                {
                    DsparkBreakEvenChoice::KeepSpeculation
                } else {
                    DsparkBreakEvenChoice::FallBackToAr
                };
                self.state = match choice {
                    DsparkBreakEvenChoice::KeepSpeculation => DsparkBreakEvenState::KeepSpeculation,
                    DsparkBreakEvenChoice::FallBackToAr => DsparkBreakEvenState::ArFallback,
                };
                Some(DsparkBreakEvenDecision {
                    choice,
                    ar_tokens_per_second,
                    spec_tokens_per_second,
                    spec_tokens,
                    spec_cycles,
                })
            }
            // Unmeasured/tail cycles and calls after a terminal decision do
            // not perturb the calibration state.
            _ => None,
        }
    }
}

/// Engine-owned DSpark propose/verify whole-turn loop.
///
/// Per cycle: pre-checks (mtp_turn's exact order) → propose (skipped on the
/// degenerate `L_cap == 0` cycle; hard error on an over-long return) → ONE
/// batched verify over `[anchor, drafts..]` → acceptance (greedy temperature:
/// argmax-based — batched fast path when penalties are no-op, per-row
/// penalized argmax otherwise, never dists/RNG; sampled temperature:
/// per-position `accept_with_residual`) → STOP-CLAMP over the accepted list
/// BEFORE commit → `commit(keep = 1 + kept_drafts, 1 + L)` where
/// `kept_drafts = min(emit_count, k)` EXCEPT that an in-cycle stop token's
/// slot is never kept (AR-parity: the tripping token is excluded when the
/// clamp cut at an accepted draft, so a stop exit leaves the cache exactly
/// where the AR reference would) → emit → cache-clear cadence → stop or
/// `anchor = boundary; eval_boundary`.
///
/// Driven in production by [`run_paged_dspark_turn`] on the paged lane and
/// by gemma4's `flat_draft_chat_turn` on the flat one; the module's mock
/// tests pin the loop contract.
pub(crate) fn run_dspark_turn<B: DsparkBackend, R: rand::Rng>(
    backend: &mut B,
    rng: &mut R,
    args: DsparkTurnArgs<'_>,
    mut streaming: Option<StreamingCtx<'_, '_>>,
) -> Result<DsparkTurnOutcome> {
    let DsparkTurnArgs {
        y,
        block_size,
        params: p,
        reasoning_tracker: tracker,
        profiler,
        max_new_tokens: max,
        eos_id,
        generated_tokens: generated,
        token_history: hist,
        finish_reason: reason,
        first_token_instant: first_tok,
        report_perf: report,
        generation_stream,
        cancel_flag,
        mut turn_token_observer,
    } = args;
    // One closure for every cancel snapshot point below — sync turns poll
    // `cancel_flag` directly; streaming turns pass the same flag, so a
    // single ungated read covers both paths byte-identically.
    let turn_cancelled = || cancel_flag.is_some_and(|flag| flag.load(Ordering::Relaxed));

    // I1: the speculative slot margin one cycle needs is the backend's
    // `SpeculativePlan` property, read once at turn entry. `0` ⇒ the backend
    // declares no speculative plan, so no margin is defined and the per-cycle
    // reservation below is inert (the flat steppers' shape).
    let lookahead_rows = backend.execution_plan().speculative.map_or(0, |plan| {
        plan.lookahead_rows_for(SpeculativeDraftWidth::DraftBlockSize(block_size))
    });

    let mut tf = super::tf_research::TfMode::begin(hist)?;
    let mut step = backend.begin_dspark_decode(block_size)?;
    if step.supports_tree_verify() {
        step.observe_emitted_tokens(hist);
    }

    // Materialize the prefill-sampled seed once; it is the first anchor.
    y.eval();
    let mut anchor: u32 = match tracker.take_forced_token() {
        Some(forced) => forced,
        None => y.item_at_int32(0)? as u32,
    };

    // `last_in_cache` default `true`: a zero-budget exit emits nothing, so
    // the last cached token is the prompt's last token, which IS in cache.
    let mut last_in_cache = true;
    let mut last_clear_at: usize = generated.len();
    let mut observer_stopped = false;

    // Same nonpositive-budget clamp as `run_mtp_turn` (see its PARITY-FIX
    // comment): a negative `max` must behave as 0, not wrap to a huge usize.
    // Teacher forcing also caps the turn at the reference length, so the
    // per-cycle `remaining` math and the length stop follow the reference.
    let max_as_usize: usize = tf.cap_max_new((max).max(0) as usize);

    // Initial-`y` emit, guarded by the budget — mirrors `run_mtp_turn`'s
    // initial push verbatim (eval / observe_token / streaming skip-on-cancel
    // with NO break / profiler bookkeeping).
    if generated.len() < max_as_usize {
        let _stream_ctx = crate::stream::StreamContext::new(generation_stream);
        profiler.begin("extract");
        let initial_token_id = anchor;
        profiler.end();
        profiler.mark_first_token();
        if report && first_tok.is_none() {
            *first_tok = Some(std::time::Instant::now());
        }
        generated.push(initial_token_id);
        hist.push(initial_token_id);
        if step.supports_tree_verify() {
            step.observe_emitted_tokens(&[initial_token_id]);
        }
        profiler.step();
        let _is_reasoning = tracker.observe_token(initial_token_id);
        if !turn_cancelled() {
            observer_stopped = turn_token_observer
                .as_deref_mut()
                .is_some_and(|observer| observer.observe_token_id(initial_token_id));
        }
        if let Some(s) = streaming.as_mut() {
            *s.last_is_reasoning = _is_reasoning;
            if !s.cancelled.load(Ordering::Relaxed) {
                let token_text = crate::tokenizer::Qwen3Tokenizer::step_decode_stream(
                    s.decode_stream,
                    s.tokenizer,
                    initial_token_id,
                    generated,
                    *s.streamed_text_len,
                );
                *s.streamed_text_len += token_text.len();
                s.emitter.on_token_id(
                    initial_token_id,
                    &token_text,
                    _is_reasoning,
                    p.include_reasoning,
                    s.callback,
                );
            }
        }
        // The just-emitted anchor has no K/V yet — the first cycle's verify
        // writes it at position 0.
        last_in_cache = false;
    }

    // Preserve the old fixed-block behavior on short generations. The
    // calibration is useful only when the remaining budget can hold one AR
    // probe plus two FULL speculative cycles even if both accept every draft;
    // otherwise replacing the first block with a probe could add target
    // forwards without ever reaching a decision.
    let calibration_draft_cap = block_size.min(p.mtp_depth);
    let calibration_required_remaining = 1usize.saturating_add(
        (calibration_draft_cap.saturating_add(1))
            .saturating_mul(usize::from(DSPARK_BREAK_EVEN_SPEC_CYCLES)),
    );
    let calibration_available_remaining = max_as_usize.saturating_sub(generated.len());
    let adaptive_ar_fallback = p.mtp_adaptive_depth
        && step.supports_adaptive_ar_fallback()
        && calibration_available_remaining >= calibration_required_remaining;
    // Later Markov confidence rows depend on earlier sampled proposals. A
    // global optimum selected from those rows would condition q on its own
    // sample. Keep this optimizer greedy-only; stochastic turns retain the
    // history-based break-even guard and exact rejection probabilities.
    let mut verification_budget = (adaptive_ar_fallback
        && step.supports_adaptive_verification()
        && sampling::is_greedy_temperature(
            p.sampling_config.and_then(|c| c.temperature).unwrap_or(1.0),
        ))
    .then(|| super::verification_budget::VerificationBudget::new(calibration_draft_cap));
    step.set_adaptive_verification(verification_budget.is_some());
    let mut break_even =
        DsparkBreakEvenPolicy::new(adaptive_ar_fallback && verification_budget.is_none());
    if adaptive_ar_fallback {
        tracing::info!(
            target: "mlx_core::inference",
            event = "dspark_break_even_calibration_started",
            speculative_probe_cycles = if verification_budget.is_some() { 3 } else { DSPARK_BREAK_EVEN_SPEC_CYCLES },
            draft_cap = calibration_draft_cap,
            required_remaining_tokens = calibration_required_remaining,
            available_remaining_tokens = calibration_available_remaining,
            minimum_speedup = DSPARK_BREAK_EVEN_MIN_SPEEDUP,
            "DSpark break-even calibration started"
        );
    }

    let tree_temperature = p.sampling_config.and_then(|c| c.temperature).unwrap_or(1.0);
    let tree_greedy_unpenalized = sampling::is_greedy_temperature(tree_temperature)
        && p.repetition_penalty == 1.0
        && p.presence_penalty == 0.0
        && p.frequency_penalty == 0.0;

    // DELIBERATE DIVERGENCE from `run_mtp_turn`: cancellation is checked at
    // the cycle top and inside the pre-commit stop-clamp, NOT per emitted
    // token. This bounds cancel latency to <= block_size + 1 tokens while
    // preserving the commit-exactly-once, caches-never-desync invariant —
    // gemma4 has no post-commit heal path, so a mid-emit cancel must never
    // be observed AFTER the cycle's commit already kept its slots.
    loop {
        timeline::cycle_start(hist.len());
        if observer_stopped {
            *reason = String::from("stop");
            break;
        }

        // Pre-cycle stop checks — `run_mtp_turn`'s exact order and reason
        // strings: zero budget → EOS/extra-EOS → cancel → repetition → max.
        if max_as_usize == 0 {
            if reason.is_empty() {
                *reason = String::from("length");
            }
            break;
        }
        if let Some(&last) = generated.last()
            && (last == eos_id || p.extra_eos_ids.contains(&last))
        {
            *reason = String::from("stop");
            // The stop token is the prior cycle's boundary (or the initial
            // seed) — never verified, so not in the physical cache.
            last_in_cache = false;
            break;
        }
        // H2: `turn_cancelled()` extends the SAME pre-cycle poll to sync
        // turns (`args.cancel_flag`); the streaming read is kept so
        // streaming behavior never depends on the caller also wiring
        // `cancel_flag`.
        if turn_cancelled()
            || streaming
                .as_ref()
                .is_some_and(|s| s.cancelled.load(Ordering::Relaxed))
        {
            *reason = String::from("cancelled");
            last_in_cache = false;
            break;
        }
        if let Some(reason_str) = crate::sampling::check_repetition_cutoff(
            generated,
            p.max_consecutive_tokens,
            p.max_ngram_repeats,
            p.ngram_size,
        ) {
            *reason = reason_str.to_string();
            last_in_cache = false;
            break;
        }
        if generated.len() >= max_as_usize {
            if reason.is_empty() {
                *reason = String::from("length");
            }
            break;
        }

        timeline::mark(Mark::StopChecks);
        let _stream_ctx = crate::stream::StreamContext::new(generation_stream);
        profiler.begin_cycle("dspark_cycle");
        let cycle_started_at = Instant::now();

        // Per-cycle draft cap. `remaining >= 1` here (the length check above
        // broke otherwise). `remaining - 1` reserves the boundary token's
        // budget slot: a cycle emits at most `L + 1` tokens.
        let remaining: usize = max_as_usize.saturating_sub(generated.len());
        let normal_l_cap: usize = block_size.min(p.mtp_depth).min(remaining.saturating_sub(1));
        if let Some(policy) = verification_budget.as_mut() {
            policy.observe_context(hist.len());
        }
        let (mut l_cap, mut measurement_cycle) = if let Some(policy) = &verification_budget {
            if policy.is_fallback() {
                (0, DsparkMeasurementCycle::None)
            } else if let Some(cap) = policy.calibration_cap(normal_l_cap) {
                (
                    cap,
                    if cap == 0 {
                        DsparkMeasurementCycle::ArProbe
                    } else {
                        DsparkMeasurementCycle::SpecProbe
                    },
                )
            } else {
                (normal_l_cap, DsparkMeasurementCycle::None)
            }
        } else {
            break_even.plan_cycle(normal_l_cap)
        };

        // Per-cycle speculative admission: every committed token moves the
        // frontier, so the lookahead region is re-reserved for THIS cycle (a
        // covered no-op while the previous reservation still holds).
        // Exhaustion drops the draft block rather than erroring the turn —
        // the cycle degenerates to one AR step through verify, whose single
        // row the ordinary decode allocation path still covers. A
        // calibration cycle already planned at `l_cap == 0` writes one row
        // too and is left alone.
        if l_cap >= 1 && lookahead_rows > 0 && !step.reserve_cycle_lookahead(lookahead_rows)? {
            tracing::debug!(
                target: "mlx_core::mtp",
                lookahead_rows,
                "DSpark cycle drafted nothing: speculative lookahead reservation \
                 exhausted; AR-decoding instead"
            );
            l_cap = 0;
            measurement_cycle = DsparkMeasurementCycle::None;
        }

        // A thinking cut is a row of this cycle's verify: the cycle keeps its
        // depth, and a probe never times a cycle the cut decides.
        let cut = if tf.is_forcing() {
            None
        } else {
            tracker.thinking_cut()
        };
        if l_cap >= 1 && cut.as_ref().is_some_and(|c| c.touches(l_cap)) {
            measurement_cycle = DsparkMeasurementCycle::None;
        }
        timeline::mark(Mark::Plan);

        let tree_max_nodes: usize = 16;
        let tree_max_depth: usize = remaining.saturating_sub(1).min(15);
        let use_tree = step.supports_tree_verify()
            && remaining > 1
            && l_cap >= 1
            && measurement_cycle == DsparkMeasurementCycle::None
            && tree_greedy_unpenalized
            && !tf.is_forcing()
            && !cut.as_ref().is_some_and(|c| c.touches(tree_max_depth));

        let (accepted, emit_count, kept_drafts, stop, boundary_id, device_budget_observation) =
            if use_tree {
                profiler.begin("dspark_verify");
                let tree_res = step.verify_tree_greedy(anchor, tree_max_nodes, tree_max_depth);
                profiler.end();
                let tree_out = tree_res?;
                if tree_out.emitted_tokens.is_empty()
                    || tree_out.accepted_nodes.len() != tree_out.emitted_tokens.len()
                    || tree_out.accepted_nodes[0] != 0
                    || tree_out.emitted_tokens.len() > remaining
                {
                    return Err(Error::from_reason(format!(
                        "DSpark tree verify contract violation: emitted={}, accepted_nodes={:?}, remaining={remaining}",
                        tree_out.emitted_tokens.len(),
                        tree_out.accepted_nodes,
                    )));
                }
                let accepted_drafts_k = tree_out.accepted_nodes.len() - 1;
                let Some(&boundary_id) = tree_out.emitted_tokens.last() else {
                    return Err(Error::from_reason(
                        "DSpark tree verify emitted_tokens unexpectedly empty",
                    ));
                };
                let accepted = tree_out.emitted_tokens;
                let DsparkCycleDecision {
                    emit_count,
                    keep,
                    stop,
                } = clamp_dspark_cycle(
                    generated,
                    &accepted,
                    accepted_drafts_k,
                    p,
                    max_as_usize,
                    eos_id,
                    || {
                        turn_cancelled()
                            || streaming
                                .as_ref()
                                .is_some_and(|s| s.cancelled.load(Ordering::Relaxed))
                    },
                    |token| {
                        turn_token_observer
                            .as_deref_mut()
                            .is_some_and(|observer| observer.observe_token_id(token))
                    },
                );
                let kept_drafts = keep - 1;
                timeline::cycle_counts(tree_out.draft_count, accepted_drafts_k, emit_count);
                timeline::mark(Mark::StopClamp);
                let keep_nodes = &tree_out.accepted_nodes[..keep];
                let keep_contiguous = if keep == tree_out.accepted_nodes.len() {
                    tree_out.is_contiguous_prefix
                } else {
                    keep_nodes.iter().enumerate().all(|(i, &n)| n == i)
                };
                profiler.begin("dspark_commit");
                let commit_res = step.commit_tree(keep_nodes, keep_contiguous);
                profiler.end();
                commit_res?;
                if tree_out.draft_count >= 1 {
                    profiler.record_mtp_cycle(tree_out.draft_count, tree_out.accepted_draft_count);
                }
                (accepted, emit_count, kept_drafts, stop, boundary_id, None)
            } else {
                // `L_cap == 0` (remaining == 1): skip propose entirely — the cycle
                // degenerates to a single AR step THROUGH verify (verify_ids =
                // [anchor] only), keeping the anchor's K/V write on the one path.
                let mut proposal = if l_cap >= 1 {
                    profiler.begin("dspark_propose");
                    let res = step.propose(anchor, l_cap, p, rng);
                    profiler.end();
                    res?
                } else {
                    DsparkProposal {
                        draft_ids: Vec::new(),
                        device_draft_ids: None,
                        device_draft_sparse: None,
                        draft_dists: Vec::new(),
                        draft_sparse_dists: Vec::new(),
                        keep_probabilities: None,
                    }
                };
                // Contract enforcement at the proposal boundary: `propose` may
                // return FEWER than `l_cap` (confidence truncation), NEVER more. An
                // over-long block would inflate the verify write past the
                // `remaining - 1` budget cap's target-cache slot expectations, and
                // the stop-clamp runs AFTER verify so it cannot un-write those
                // slots — a hard error surfaces the stepper bug instead of masking
                // it.
                let proposed_len = proposal.draft_len()?;
                if proposed_len > l_cap {
                    return Err(Error::from_reason(format!(
                        "DSpark propose over-returned: {proposed_len} draft tokens for a cap of {l_cap} \
                 (the DsparkStepper::propose contract allows fewer, never more)"
                    )));
                }
                // At row 0 the end tokens are the first proposals; the cycle then
                // verifies host ids.
                let touched = cut.as_ref().is_some_and(|c| c.touches(proposed_len));
                if touched && let Some(c) = cut.as_ref() {
                    proposal.materialize_draft_ids()?;
                    proposal.replace_prefix(c.proposals(proposed_len))?;
                }

                let proposal_ns = cycle_started_at
                    .elapsed()
                    .as_nanos()
                    .min(u128::from(u64::MAX)) as u64;
                if measurement_cycle == DsparkMeasurementCycle::None
                    && proposed_len > 0
                    && let (Some(policy), Some(confidence)) =
                        (&mut verification_budget, &proposal.keep_probabilities)
                    && let Some(keep) = policy.choose(confidence, proposal_ns)
                {
                    if keep == 0 {
                        step.enter_ar_fallback()?;
                    }
                    proposal.truncate(keep);
                }
                let draft_len = proposal.draft_len()?;
                let replaced = cut.as_ref().map_or(0, |c| c.replaced(draft_len));
                timeline::mark(Mark::DraftPost);

                profiler.begin("dspark_verify");
                // The one-token calibration path deliberately omits speculative
                // rollback snapshotting, but still builds the ordinary tapped target
                // graph needed to seed the following speculative cycle. Start the AR
                // timer immediately before that method: target graph construction and
                // dispatch, GPU force, and sampling are measured; only draft-context
                // fusion in `commit_ar_probe` remains outside it.
                let verify_started_at = Instant::now();
                let ar_probe_started_at = (measurement_cycle == DsparkMeasurementCycle::ArProbe)
                    .then_some(verify_started_at);
                // Teacher forcing: the reference continuation replaces the drafts as
                // the verify input, through the same host verify path.
                let tf_ids = tf.verify_ids(generated, anchor, draft_len)?;
                let temperature = p.sampling_config.and_then(|c| c.temperature).unwrap_or(1.0);
                let penalties_no_op = p.repetition_penalty == 1.0
                    && p.presence_penalty == 0.0
                    && p.frequency_penalty == 0.0;
                let use_fused_greedy = step.supports_fused_greedy_verify()
                    && measurement_cycle != DsparkMeasurementCycle::ArProbe
                    && tf_ids.is_none()
                    && !(!touched && step.device_verification_budget().is_some())
                    && sampling::is_greedy_temperature(temperature)
                    && penalties_no_op;

                let mut device_budget_observation = None;
                let accept_res = if use_fused_greedy {
                    let argmax_res = if let Some(device_ids) = proposal.device_draft_ids.as_ref() {
                        let anchor_arr = MxArray::from_int32(&[anchor as i32], &[1])?;
                        let ids = MxArray::concatenate(&anchor_arr, device_ids, 0)?;
                        step.verify_device_greedy(&ids)
                    } else {
                        let mut verify_ids: Vec<u32> = Vec::with_capacity(1 + draft_len);
                        verify_ids.push(anchor);
                        verify_ids.extend(proposal.draft_ids.iter().map(|&id| id as u32));
                        step.verify_greedy(&verify_ids)
                    };
                    profiler.end();
                    let argmax_arr = argmax_res?;
                    profiler.begin("dspark_accept");
                    let res = accept_dspark_greedy_argmax(&argmax_arr, &mut proposal, cut.as_ref());
                    profiler.end();
                    res
                } else {
                    let verify_res = if measurement_cycle == DsparkMeasurementCycle::ArProbe {
                        step.verify_ar_probe(anchor)
                    } else if let Some(ids) = tf_ids.as_ref() {
                        step.verify(ids)
                    } else if let Some(device_ids) = proposal.device_draft_ids.as_ref() {
                        // Device-resident proposal: anchor + draft path enter the target
                        // graph without a host round-trip (the acceptance eval below
                        // remains the cycle's single materialization point).
                        let anchor_arr = MxArray::from_int32(&[anchor as i32], &[1])?;
                        let ids = MxArray::concatenate(&anchor_arr, device_ids, 0)?;
                        step.verify_device(&ids)
                    } else {
                        let mut verify_ids: Vec<u32> = Vec::with_capacity(1 + draft_len);
                        verify_ids.push(anchor);
                        verify_ids.extend(proposal.draft_ids.iter().map(|&id| id as u32));
                        step.verify(&verify_ids)
                    };
                    profiler.end();
                    let verify_out = verify_res?;
                    let logits = verify_out.logits; // [1, 1+L, vocab]

                    // Acceptance: `k` accepted drafts (prefix of `draft_ids`) + ONE
                    // boundary token (bonus on full accept, residual on rejection).
                    profiler.begin("dspark_accept");
                    let res = if tf_ids.is_some() {
                        tf.accept(&logits, &mut proposal, generated)
                    } else if !touched && let Some(budget) = step.device_verification_budget() {
                        if !sampling::is_greedy_temperature(temperature) || !penalties_no_op {
                            return Err(Error::from_reason(
                                "device verification budgeting requires unpenalized greedy acceptance",
                            ));
                        }
                        accept_device_budgeted_greedy(&logits, &mut proposal, budget).map(
                            |(k, token, rows, raw)| {
                                device_budget_observation = Some((rows, raw));
                                (k, token)
                            },
                        )
                    } else {
                        let rows_route = step.device_sampled_rows();
                        let submitted = if rows_route {
                            submit_sampled_rows(&logits, &proposal, p, rng)
                        } else {
                            Ok(None)
                        };
                        match submitted {
                            Err(error) => Err(error),
                            Ok(Some(rows)) => MxArray::eval_arrays_with_context(
                                &[rows.records()],
                                "speculative_sampled_rows",
                            )
                            .and_then(|()| {
                                timeline::mark(Mark::VerifyWait);
                                rows.accept(&mut proposal, cut.as_ref())
                            }),
                            Ok(None) if !touched && proposal.device_draft_sparse.is_some() => {
                                accept_device_sampled(&logits, &mut proposal, p, rng)
                            }
                            Ok(None) => accept_dspark_proposal(
                                &logits,
                                &mut proposal,
                                hist,
                                p,
                                rng,
                                cut.as_ref(),
                            ),
                        }
                    };
                    profiler.end();
                    res
                };
                let (accepted_drafts_k, boundary_id) = accept_res?;
                timeline::mark(Mark::Accept);
                // Acceptance has forced the verify graph. Back-fill a device proposal
                // now so commit can reuse the engine's exact verify provenance instead
                // of reading the concatenated verify input from the device again. The
                // ordinary accept paths already did this.
                proposal.materialize_draft_ids()?;
                let verify_ns = verify_started_at
                    .elapsed()
                    .as_nanos()
                    .min(u128::from(u64::MAX)) as u64;
                // Acceptance's token read has now fully forced the target graph.
                // AR measures from immediately before its dedicated target forward;
                // speculative probes retain the whole-cycle propose+verify timer.
                // Both stop before their respective commit/materialization handling.
                let measured_before_commit_started_at =
                    ar_probe_started_at.unwrap_or(cycle_started_at);
                let precommit_wall_ns = measured_before_commit_started_at
                    .elapsed()
                    .as_nanos()
                    .min(u128::from(u64::MAX)) as u64;

                let mut accepted: Vec<u32> = Vec::with_capacity(accepted_drafts_k + 1);
                accepted.extend(
                    proposal.draft_ids[..accepted_drafts_k]
                        .iter()
                        .map(|&id| id as u32),
                );
                accepted.push(boundary_id);

                // STOP-CLAMP BEFORE COMMIT: walk the accepted list simulating the
                // history, replaying `run_mtp_turn`'s emit-loop per-token check
                // order (budget-before-push → push → cancel → EOS/extra-EOS →
                // repetition) so the stop decision is byte-equivalent — but taken
                // BEFORE `commit`, so the stepper keeps exactly the emitted
                // prefix's K/V and never needs a post-commit rollback. The
                // EOS/repetition/cancel cuts are INCLUSIVE of the tripping token;
                // the budget cut is EXCLUSIVE (mtp checks the budget before the
                // push).
                let DsparkCycleDecision {
                    emit_count,
                    keep,
                    stop,
                } = clamp_dspark_cycle(
                    generated,
                    &accepted,
                    accepted_drafts_k,
                    p,
                    max_as_usize,
                    eos_id,
                    || {
                        turn_cancelled()
                            || streaming
                                .as_ref()
                                .is_some_and(|s| s.cancelled.load(Ordering::Relaxed))
                    },
                    |token| {
                        turn_token_observer
                            .as_deref_mut()
                            .is_some_and(|observer| observer.observe_token_id(token))
                    },
                );
                let kept_drafts = keep - 1;
                timeline::cycle_counts(draft_len, accepted_drafts_k, emit_count);
                timeline::mark(Mark::StopClamp);
                let verified_ids = tf_ids.unwrap_or_else(|| {
                    let mut verified_ids = Vec::with_capacity(1 + draft_len);
                    verified_ids.push(anchor);
                    verified_ids.extend(proposal.draft_ids.iter().map(|&id| id as u32));
                    verified_ids
                });
                profiler.begin("dspark_commit");
                let commit_res = if measurement_cycle == DsparkMeasurementCycle::ArProbe {
                    if keep != 1 || draft_len != 0 {
                        Err(Error::from_reason(format!(
                            "DSpark AR probe contract violation: keep={keep}, draft_len={draft_len}"
                        )))
                    } else {
                        step.commit_ar_probe()
                    }
                } else {
                    step.commit_with_provenance(keep, 1 + draft_len, &verified_ids)
                };
                profiler.end();
                commit_res?;
                let device_budget_observation = match device_budget_observation {
                    Some((rows, raw_matches)) => {
                        // Charge this cycle's committed GPU work before host-only emission.
                        // Boundary submission includes the rollback and draft-context roots.
                        let boundary = MxArray::from_int32(&[boundary_id as i32], &[1])?;
                        profiler.begin("dspark_eval_boundary");
                        step.eval_boundary(&boundary);
                        profiler.end();
                        crate::array::synchronize();
                        let milliseconds = cycle_started_at.elapsed().as_secs_f64() * 1000.0;
                        Some((rows, raw_matches, milliseconds))
                    }
                    None => None,
                };
                if tf.is_forcing() {
                    tf.end_cycle(cycle_started_at.elapsed());
                }
                let break_even_decision = if stop.is_some() {
                    None
                } else {
                    match measurement_cycle {
                        DsparkMeasurementCycle::None => None,
                        DsparkMeasurementCycle::ArProbe => {
                            // Build/materialize the context needed by the following
                            // speculative probes, but deliberately exclude that
                            // draft-only calibration cost from target-AR throughput.
                            step.materialize_adaptive_state()?;
                            if let Some(policy) = verification_budget.as_mut() {
                                policy.record(0, verify_ns);
                            }
                            break_even.record_cycle(
                                measurement_cycle,
                                precommit_wall_ns,
                                (accepted_drafts_k + 1) as u32,
                            )
                        }
                        DsparkMeasurementCycle::SpecProbe => {
                            // Include this cycle's own context append. Without the
                            // eval, MLX would charge it to the next proposal and make
                            // the two probe timings order-dependent.
                            step.materialize_adaptive_state()?;
                            if let Some(policy) = verification_budget.as_mut() {
                                // Include rollback, durable settlement and this cycle's
                                // context append. Only calibration forces this boundary;
                                // steady-state proposals retain their asynchronous graph.
                                policy.record(
                                    draft_len,
                                    verify_started_at
                                        .elapsed()
                                        .as_nanos()
                                        .min(u128::from(u64::MAX))
                                        as u64,
                                );
                            }
                            let wall_ns = cycle_started_at
                                .elapsed()
                                .as_nanos()
                                .min(u128::from(u64::MAX))
                                as u64;
                            break_even.record_cycle(
                                measurement_cycle,
                                wall_ns,
                                (accepted_drafts_k + 1) as u32,
                            )
                        }
                    }
                };

                if let Some(decision) = break_even_decision {
                    if decision.choice == DsparkBreakEvenChoice::FallBackToAr {
                        step.enter_ar_fallback()?;
                    }
                    tracing::info!(
                        target: "mlx_core::inference",
                        event = "dspark_break_even_decision",
                        decision = match decision.choice {
                            DsparkBreakEvenChoice::KeepSpeculation => "keep_speculation",
                            DsparkBreakEvenChoice::FallBackToAr => "target_only_ar",
                        },
                        ar_tokens_per_second = decision.ar_tokens_per_second,
                        spec_tokens_per_second = decision.spec_tokens_per_second,
                        spec_tokens = decision.spec_tokens,
                        spec_cycles = decision.spec_cycles,
                        minimum_speedup = DSPARK_BREAK_EVEN_MIN_SPEEDUP,
                        "DSpark break-even decision"
                    );
                }

                // Per-cycle acceptance stats: DRAFTED depth `L` and accepted draft
                // count `k` — the same argument semantics as `run_mtp_cycle`'s
                // `record_mtp_cycle(effective_depth, accepted_drafts)` (boundary
                // excluded from both). Degenerate `L == 0` cycles are plain AR
                // steps and are NOT recorded, nor are end tokens that replaced
                // proposals.
                if draft_len - replaced >= 1 {
                    profiler.record_mtp_cycle(
                        draft_len - replaced,
                        accepted_drafts_k.saturating_sub(replaced),
                    );
                }
                (
                    accepted,
                    emit_count,
                    kept_drafts,
                    stop,
                    boundary_id,
                    device_budget_observation,
                )
            };

        // Emit the clamped prefix — `run_mtp_turn`'s emit-block bookkeeping
        // (push / observe / detok / emitter), with the stop checks already
        // resolved by the clamp above and NO per-token cancel check (see
        // the divergence note at the loop top). Cancel parity: the token on
        // which the clamp observed the cancel is committed but NOT streamed,
        // matching the mtp emit arm's break-before-detok.
        timeline::mark(Mark::PostCommit);
        profiler.begin("dspark_emit_loop");
        for (idx, &tok_id) in accepted[..emit_count].iter().enumerate() {
            generated.push(tok_id);
            hist.push(tok_id);
            profiler.step();
            let _is_reasoning = tracker.observe_token(tok_id);
            if let Some(s) = streaming.as_mut() {
                *s.last_is_reasoning = _is_reasoning;
                if matches!(stop, Some(CycleStop::Cancelled)) && idx + 1 == emit_count {
                    continue;
                }
                let token_text = crate::tokenizer::Qwen3Tokenizer::step_decode_stream(
                    s.decode_stream,
                    s.tokenizer,
                    tok_id,
                    generated,
                    *s.streamed_text_len,
                );
                *s.streamed_text_len += token_text.len();
                s.emitter.on_token_id(
                    tok_id,
                    &token_text,
                    _is_reasoning,
                    p.include_reasoning,
                    s.callback,
                );
            }
        }
        profiler.end();
        if step.supports_tree_verify() && emit_count > 0 {
            step.observe_emitted_tokens(&accepted[..emit_count]);
        }
        timeline::mark(Mark::Emit);

        // Bound allocator cache growth at the established token cadence.
        // On CUDA/WDDM, unconditional `clear_cache()` runs `cudaMemPoolTrimTo(0)`
        // (~670 ms stall per trim) even when the cache is only a few MB; only
        // trim on CUDA when cache memory actually exceeds 1 GiB.
        if generated.len() >= last_clear_at + 256 {
            let should_clear = if crate::accelerator::cuda_ops_available() {
                crate::array::get_cache_memory()
                    > crate::accelerator::cuda_cache_clear_threshold_bytes()
            } else {
                true
            };
            if should_clear {
                profiler.begin("dspark_cache_clear");
                crate::array::synchronize_and_clear_cache();
                profiler.end();
                timeline::mark(Mark::CacheClear);
            }
            last_clear_at = generated.len();
        }

        // `last_in_cache` invariant: true iff the LAST emitted token's K/V
        // is in the target cache — i.e. it is one of the `kept_drafts`
        // slots. `emit_count == k + 1` (full emit incl. boundary) on every
        // non-stop cycle → false; in-cycle stop tokens are never kept (the
        // AR-parity exclusion above) → false on every stop cut, whether the
        // stop token was an accepted draft or the boundary.
        last_in_cache = emit_count <= kept_drafts;

        if let Some(kind) = stop {
            if let Some((rows, raw_matches, milliseconds)) = device_budget_observation {
                step.observe_device_verification(rows, raw_matches, milliseconds)?;
            }
            match kind {
                CycleStop::Length => {
                    if reason.is_empty() {
                        *reason = String::from("length");
                    }
                }
                CycleStop::Stop => *reason = String::from("stop"),
                CycleStop::Cancelled => *reason = String::from("cancelled"),
                CycleStop::Repetition(r) => *reason = r.to_string(),
            }
            profiler.end();
            break;
        }

        // Continue: the boundary becomes the next cycle's anchor.
        anchor = boundary_id;
        if device_budget_observation.is_none() {
            let y_arr = MxArray::from_int32(&[boundary_id as i32], &[1])?;
            profiler.begin("dspark_eval_boundary");
            step.eval_boundary(&y_arr);
            profiler.end();
        }
        if let Some((rows, raw_matches, milliseconds)) = device_budget_observation {
            step.observe_device_verification(rows, raw_matches, milliseconds)?;
        }
        profiler.end();
        timeline::mark(Mark::BoundarySubmit);
    }

    step.finish()?;
    tf.finish(generated, max)?;
    profiler.snapshot_memory_after();
    profiler.report();

    Ok(DsparkTurnOutcome { last_in_cache })
}

/// Sub-trait for a family whose DSpark speculative turn runs against the
/// target's PAGED KV state: [`PagedBackend`]'s turn lifecycle (prime →
/// prefill → reconcile → finalize → save) with [`DsparkBackend`]'s
/// propose/verify stepper in place of the autoregressive decode loop.
///
/// Production implementation: gemma4 (`models::gemma4::dspark_decode`).
pub(crate) trait PagedDsparkBackend: PagedBackend + DsparkBackend {
    /// [`PagedBackend::paged_prefill`] with the draft's per-turn state built
    /// alongside it, stashed for [`DsparkBackend::begin_dspark_decode`] to
    /// take.
    ///
    /// Target-side compute must be byte-identical to the plain paged
    /// prefill — the T=0 oracle compares the speculative and autoregressive
    /// lanes on the same prompt — so the draft side rides the SAME chunk
    /// walk rather than a second one.
    fn paged_prefill_with_draft_state(
        &mut self,
        suffix_tokens: &[u32],
        prefix: &Self::PrefixState,
        stream: Stream,
    ) -> Result<MxArray>;

    /// The per-cycle drafted-block cap (the drafter's own block width), and
    /// the quantity the turn's lookahead reservation is sized from.
    fn dspark_block_size(&self) -> Result<usize>;

    /// The paged sequence this turn owns. Diagnostic only: it names the
    /// sequence in the abandoned-epilogue report.
    fn paged_spec_seq_id(&self) -> u32;

    /// The turn's prime: [`PagedBackend::prime_prefix_state`] unless the
    /// family reuses a prefix only together with its draft state.
    fn prime_dspark_prefix_state(
        &mut self,
        tokens: &[u32],
        reuse_cache: bool,
        cache_salt: u64,
    ) -> Result<Self::PrefixState> {
        self.prime_prefix_state(tokens, reuse_cache, 0, &[], cache_salt)
    }
}

/// Drive one PAGED DSpark speculative whole turn.
///
/// MIRRORS [`crate::engine::paged_turn::run_paged_turn`] hook for hook —
/// prime → prefill → first-token sample → decode → materialize-final →
/// end-decode → epilogue — substituting the tapped prefill and
/// [`run_dspark_turn`] for the plain prefill and `run_decode_loop`. The
/// turn tail is literally the autoregressive one: the same
/// [`DecodeStep::materialize_final`] and [`DecodeStep::end_decode`] calls
/// on the same [`PagedBackend::begin_paged_decode`] stepper, so a
/// speculative turn cannot end on a frontier the autoregressive turn would
/// have settled. The epilogue is not called directly: [`SpecTurnEpilogue`]
/// is minted here and is the only way to reach it (L-EPILOGUE).
pub(crate) fn run_paged_dspark_turn<B: PagedDsparkBackend>(
    backend: &mut B,
    args: &mut WholeTurnArgs<'_>,
) -> Result<TurnOutput> {
    let tokenizer = args.tokenizer.clone();
    let eos_id = args.eos_id;
    let thinking = args.thinking;
    let is_delta = args.plan.is_delta;
    let is_streaming = args.sink.is_some();
    let close = backend.think_end_for_turn(args.config, &tokenizer);

    // The session core's params (a family's context preflight may have
    // clamped them), plus the stop-set `run_dspark_turn` reads from
    // `ChatParams::extra_eos_ids` (the autoregressive loops thread it as a
    // separate argument instead).
    let mut p = args.params.clone();
    p.extra_eos_ids = backend.extra_eos_ids();
    let report_perf = p.report_performance;
    let max_new_tokens = p.max_new_tokens;
    let reuse_cache = if is_delta { true } else { p.reuse_cache };
    let block_size = backend.dspark_block_size()?;

    let generation_start = report_perf.then(Instant::now);
    let mut first_token_instant: Option<Instant> = None;

    // L-EPILOGUE: from here the turn owes exactly one epilogue, and this
    // token is the only handle to it.
    let epilogue = SpecTurnEpilogue::begin(backend.paged_spec_seq_id());

    backend.set_generation_suffix(tokenizer.generation_suffix(args.config).ok().flatten());
    let prefix = match backend.prime_dspark_prefix_state(args.tokens, reuse_cache, p.cache_salt) {
        Ok(prefix) => prefix,
        Err(error) => {
            epilogue.abort(backend);
            return Err(error);
        }
    };
    let effective_cached_prefix_len = prefix.effective_cached_prefix_len();
    let suffix_len = prefix.suffix_len();
    if suffix_len == 0 {
        epilogue.abort(backend);
        return Err(Error::from_reason(
            "run_paged_dspark_turn: empty prefill suffix (prime_prefix_state must cap cached \
             prefix at len-1)",
        ));
    }
    let suffix = &args.tokens[effective_cached_prefix_len..];

    let mut token_history: Vec<u32> = args.tokens.to_vec();
    let mut generated_tokens: Vec<u32> =
        Vec::with_capacity(generated_capacity_hint(max_new_tokens));
    let mut finish_reason = String::from("length");

    let generation_stream = Stream::generation();
    let _wired_ctx = backend
        .wired_limit_bytes()
        .map(|bytes| crate::stream::WiredLimitContext::new(bytes, vec![generation_stream]));

    let mut profiler = DecodeProfiler::new(
        backend.profiler_label(is_delta, is_streaming),
        backend.family_name(),
    );
    profiler.set_prompt_tokens(suffix_len as u32);
    profiler.snapshot_memory_before();

    let mut reasoning_tracker = ReasoningTracker::from_setup(&thinking, &close);
    let stream_skip_special = backend.stream_skip_special_tokens();
    // One streaming bundle — detokenizer + cursors + emitter — built only
    // when a sink exists so the sync path never runs the emitter hook.
    let mut turn_streaming = args.sink.map(|_| {
        TurnStreaming::new(
            backend,
            tokenizer.inner(),
            thinking.enabled,
            stream_skip_special,
        )
    });
    let turn_token_observer = backend.turn_token_observer();

    profiler.begin_prefill();
    let last_logits =
        match backend.paged_prefill_with_draft_state(suffix, &prefix, generation_stream) {
            Ok(logits) => logits,
            Err(error) => {
                epilogue.abort(backend);
                return Err(error);
            }
        };
    profiler.end_prefill();

    let last_logits = match apply_all_penalties(last_logits, &token_history, &p) {
        Ok(logits) => logits,
        Err(error) => {
            epilogue.abort(backend);
            return Err(error);
        }
    };
    let y = match sampling::sample(&last_logits, p.sampling_config) {
        Ok(y) => y,
        Err(error) => {
            epilogue.abort(backend);
            return Err(error);
        }
    };
    y.eval();
    if report_perf {
        first_token_instant = Some(Instant::now());
    }
    let should_clear_post_prefill = if crate::accelerator::cuda_ops_available() {
        crate::array::get_cache_memory() > crate::accelerator::cuda_cache_clear_threshold_bytes()
    } else {
        true
    };
    if should_clear_post_prefill {
        crate::array::synchronize_and_clear_cache();
    }

    let mut rng = rand::rng();
    let outcome = {
        let streaming_ctx = turn_streaming
            .as_mut()
            .and_then(|ts| ts.ctx(args.sink, args.cancelled));
        run_dspark_turn(
            backend,
            &mut rng,
            DsparkTurnArgs {
                y,
                block_size,
                params: &p,
                reasoning_tracker: &mut reasoning_tracker,
                profiler: &mut profiler,
                max_new_tokens,
                eos_id,
                generated_tokens: &mut generated_tokens,
                token_history: &mut token_history,
                finish_reason: &mut finish_reason,
                first_token_instant: &mut first_token_instant,
                report_perf,
                generation_stream,
                cancel_flag: args.cancelled,
                turn_token_observer,
            },
            streaming_ctx,
        )
    };
    let outcome = match outcome {
        Ok(outcome) => Some(outcome),
        Err(error) => {
            if let Err(error) =
                crate::engine::paged_turn::vacated_stop(error, &generated_tokens, args.sink)
            {
                epilogue.abort(backend);
                return Err(error);
            }
            finish_reason = "cancelled".to_string();
            None
        }
    };

    // The autoregressive lane's turn tail, unchanged. LENGTH exits keep
    // every token, so the last one's K/V must exist for the adapter cursor
    // to equal the saved history — a kept accepted draft already has it, a
    // cycle boundary never does. `end_decode` then settles the frontier that
    // row moved: the per-cycle settles inside `run_dspark_turn` stop at the
    // last COMMIT, so without this the sliding prune and the cold-rung walk
    // would never see the final token.
    let tail: Result<()> = match &outcome {
        Some(outcome) => (|| {
            let mut step = backend.begin_paged_decode()?;
            if finish_reason == "length"
                && !outcome.last_in_cache
                && let Some(&final_token) = generated_tokens.last()
            {
                step.materialize_final(final_token)?;
            }
            step.end_decode()
        })(),
        None => Ok(()),
    };
    if let Err(error) = tail {
        epilogue.abort(backend);
        return Err(error);
    }

    let finish = if outcome.is_some() {
        SpecTurnEpilogue::finish::<B>
    } else {
        SpecTurnEpilogue::vacated::<B>
    };
    finish(
        epilogue,
        backend,
        FinishPagedTurnArgs {
            tokenizer: &tokenizer,
            params: &p,
            config: args.config,
            thinking,
            is_delta,
            reuse_cache,
            prompt_tokens: args.tokens,
            effective_cached_prefix_len,
            suffix_len,
            generated_tokens: &generated_tokens,
            finish_reason,
            retain_final_length_token: true,
            generation_start,
            first_token_instant,
            reasoning: reasoning_tracker.outcome(),
            profiler: &profiler,
            stream_skip_special,
            // The epilogue takes scalar copies — the same defaults the
            // locals had when the (sink-less) sync path never created a
            // streaming bundle.
            streamed_text_len: turn_streaming.as_ref().map_or(0, |ts| ts.streamed_text_len),
            last_is_reasoning: turn_streaming
                .as_ref()
                .map_or(thinking.enabled, |ts| ts.last_is_reasoning),
            sink: args.sink,
            emitter: turn_streaming.map(|ts| ts.emitter),
        },
    )
}

/// Build parallel-infill query token ids: `[anchor, mask, ..., mask]` with
/// `1 + draft_len` entries for DFlash / DFlash2 drafters.
pub(crate) fn parallel_query_ids(anchor: u32, mask: usize, draft_len: usize) -> Vec<i32> {
    let mut ids = Vec::with_capacity(draft_len.saturating_add(1));
    ids.push(anchor as i32);
    ids.resize(draft_len.saturating_add(1), mask as i32);
    ids
}

/// Non-causal sliding-window attention mask `[1, 1, query_len, key_len]` for
/// parallel-infill drafters whose keys span `key_base..key_base+key_len`.
/// Returns `Ok(None)` when every key already falls inside `window`.
pub(crate) fn non_causal_sliding_mask(
    query_base: i32,
    query_len: i64,
    key_base: i32,
    key_len: i64,
    window: i64,
) -> Result<Option<MxArray>> {
    if key_len <= window {
        return Ok(None);
    }
    let queries = MxArray::arange(
        f64::from(query_base),
        f64::from(query_base) + query_len as f64,
        None,
        None,
    )?
    .reshape(&[query_len, 1])?;
    let keys = MxArray::arange(
        f64::from(key_base),
        f64::from(key_base) + key_len as f64,
        None,
        None,
    )?
    .reshape(&[1, key_len])?;
    let distance = queries.sub(&keys)?;
    let window = MxArray::scalar_int(window as i32)?;
    Ok(Some(
        distance
            .less(&window)?
            .reshape(&[1, 1, query_len, key_len])?,
    ))
}

/// Per-turn prefix plan returned by [`FlatDsparkBackend::prepare_flat_dspark_turn`].
pub(crate) struct FlatDsparkPrefix {
    /// Tokens to prefill on this turn (full prompt on cold/miss, suffix on hit/delta).
    pub prefill_tokens: Vec<u32>,
    /// Number of prompt tokens served from cache (reported in `ChatResult.cached_tokens`).
    pub cached_prefix_len: usize,
    /// Prior live history length on delta turns (used for streaming delta prompt accounting).
    pub prior_cached_len: usize,
    /// Hard per-cycle draft block cap passed to [`run_dspark_turn`].
    pub block_size: usize,
}

/// Sub-trait for a family whose flat-KV draft speculative turn runs through
/// [`run_flat_dspark_turn`] instead of duplicating the prefill/sample/loop/save/finalize
/// boilerplate (Gemma4 Assistant, Muse-Glimmer DFlash, Qwen4Exp flat DFlash2).
pub(crate) trait FlatDsparkBackend: DsparkBackend {
    /// Whether [`run_flat_dspark_turn`] overwrites `result.thinking_enabled` with
    /// `resolve_enable_thinking(args.config).unwrap_or(true)` after `finalize_turn`.
    /// Families under `ThinkingPolicy::AlwaysOnBudgetFromEffort` (Muse-Glimmer) set
    /// this to `false` to preserve `args.thinking.enabled`.
    const OVERWRITE_THINKING_ENABLED: bool = true;

    /// Validate prompt/draft/cancel state and resolve the prefix vs. prefill suffix.
    ///
    /// Errors returned here do NOT invoke [`Self::fail_closed_flat_dspark`] so a
    /// pre-prefill cancel preserves the live session; backends that mutate state
    /// before failing here (e.g. `draft_taps_on`) must fail-close themselves.
    fn prepare_flat_dspark_turn(
        &mut self,
        tokens: &[u32],
        params: &ChatParams,
        is_delta: bool,
    ) -> Result<FlatDsparkPrefix>;

    /// Run the flat target prefill alongside draft-state capture and stash the
    /// per-turn draft state for [`DsparkBackend::begin_dspark_decode`].
    fn flat_prefill_with_draft_state(
        &mut self,
        prefix: &FlatDsparkPrefix,
        stream: Stream,
    ) -> Result<MxArray>;

    /// Post-prefill cache evaluation hook (default: `self.eval_caches()`).
    fn eval_after_flat_prefill(&self) -> Result<()> {
        self.eval_caches()
    }

    /// Fail closed after any error between prefill start and save completion.
    fn fail_closed_flat_dspark(&mut self, error: Error) -> Error;

    /// Persist post-turn cache/history state (including optional length-exit
    /// final-token materialization and draft-tap teardown).
    #[allow(clippy::too_many_arguments)]
    fn finish_flat_dspark_turn(
        &mut self,
        tokens: &[u32],
        generated_tokens: &[u32],
        finish_reason: &str,
        last_in_cache: bool,
        is_delta: bool,
        reuse_cache: bool,
        stream: Stream,
    ) -> Result<()>;
}

/// Drive one FLAT-KV DSpark / DFlash / Assistant speculative whole turn,
/// supporting both sync and streaming callers through `args.sink`.
pub(crate) fn run_flat_dspark_turn<B: FlatDsparkBackend>(
    backend: &mut B,
    args: &mut WholeTurnArgs<'_>,
) -> Result<TurnOutput> {
    let tokenizer = args.tokenizer.clone();
    let eos_id = args.eos_id;
    let thinking = args.thinking;
    let is_delta = args.plan.is_delta;
    let tokens: Vec<u32> = args.tokens.to_vec();
    let is_streaming = args.sink.is_some();

    let mut params = backend.resolve_params(args.config);
    params.extra_eos_ids = backend.extra_eos_ids();
    let reuse_cache = params.reuse_cache;
    let report_perf = params.report_performance;
    let max_new_tokens = params.max_new_tokens;

    let prefix = backend.prepare_flat_dspark_turn(&tokens, &params, is_delta)?;

    let generation_start = report_perf.then(Instant::now);
    let mut first_token_instant: Option<Instant> = None;

    let prompt_token_count = tokens.len();
    let mut token_history: Vec<u32> = tokens.clone();
    let mut generated_tokens: Vec<u32> =
        Vec::with_capacity(generated_capacity_hint(max_new_tokens));
    let mut finish_reason = String::from("length");

    let generation_stream = Stream::generation();

    let mut profiler = DecodeProfiler::new(
        backend.profiler_label(is_delta, is_streaming),
        backend.family_name(),
    );
    profiler.set_prompt_tokens(prefix.prefill_tokens.len() as u32);
    profiler.snapshot_memory_before();

    let close = backend.think_end_for_turn(args.config, &tokenizer);
    let mut reasoning_tracker = ReasoningTracker::from_setup(&thinking, &close);

    let stream_skip_special = backend.stream_skip_special_tokens();
    let mut turn_streaming = args.sink.map(|_| {
        TurnStreaming::new(
            backend,
            tokenizer.inner(),
            thinking.enabled,
            stream_skip_special,
        )
    });
    let turn_token_observer = backend.turn_token_observer();

    let progress = args.sink.and_then(|sink| sink.progress());
    if let Some(progress) = progress {
        progress.begin_prefill(
            prefix.prefill_tokens.len() as u32,
            prefix.cached_prefix_len as u32,
        );
    }
    profiler.begin_prefill();
    let last_logits = match backend.flat_prefill_with_draft_state(&prefix, generation_stream) {
        Ok(logits) => logits,
        Err(error) => return Err(backend.fail_closed_flat_dspark(error)),
    };
    profiler.end_prefill();

    let y = match apply_all_penalties(last_logits, &token_history, &params)
        .and_then(|logits| sampling::sample(&logits, params.sampling_config))
    {
        Ok(token) => token,
        Err(error) => return Err(backend.fail_closed_flat_dspark(error)),
    };
    y.eval();

    if let Err(error) = backend.eval_after_flat_prefill() {
        return Err(backend.fail_closed_flat_dspark(error));
    }
    if let Some(progress) = progress {
        progress.first_token();
    }
    if report_perf {
        first_token_instant = Some(Instant::now());
    }

    let mut rng = rand::rng();
    let outcome = {
        let streaming_ctx = turn_streaming
            .as_mut()
            .and_then(|ts| ts.ctx(args.sink, args.cancelled));
        run_dspark_turn(
            backend,
            &mut rng,
            DsparkTurnArgs {
                y,
                block_size: prefix.block_size,
                params: &params,
                reasoning_tracker: &mut reasoning_tracker,
                profiler: &mut profiler,
                max_new_tokens,
                eos_id,
                generated_tokens: &mut generated_tokens,
                token_history: &mut token_history,
                finish_reason: &mut finish_reason,
                first_token_instant: &mut first_token_instant,
                report_perf,
                generation_stream,
                cancel_flag: args.cancelled,
                turn_token_observer,
            },
            streaming_ctx,
        )
    };
    let last_in_cache = match outcome {
        Ok(outcome) => outcome.last_in_cache,
        Err(error) => return Err(backend.fail_closed_flat_dspark(error)),
    };

    if let Err(error) = backend.finish_flat_dspark_turn(
        &tokens,
        &generated_tokens,
        &finish_reason,
        last_in_cache,
        is_delta,
        reuse_cache,
        generation_stream,
    ) {
        return Err(backend.fail_closed_flat_dspark(error));
    }

    let performance = if report_perf {
        compute_performance_metrics(
            generation_start,
            first_token_instant,
            prefix.prefill_tokens.len(),
            generated_tokens.len(),
        )
        .map(|mut metrics| {
            backend.augment_performance(&profiler, &mut metrics);
            metrics
        })
    } else {
        None
    };
    let reasoning = reasoning_tracker.outcome();

    if let (Some(sink), Some(ts)) = (args.sink, turn_streaming.as_mut()) {
        let full_text = tokenizer
            .decode_sync(&generated_tokens, stream_skip_special)
            .unwrap_or_else(|e| {
                tracing::warn!("Failed to decode generated tokens: {}", e);
                String::new()
            });
        ts.flush_residual(&full_text, params.include_reasoning, sink);
    }

    let reported_prompt_tokens: u32 = if is_delta && is_streaming {
        let delta_len = prompt_token_count.saturating_sub(prefix.prior_cached_len);
        backend.stream_delta_prompt_tokens(prompt_token_count, delta_len)
    } else {
        prompt_token_count as u32
    };

    let mut result = backend.finalize_turn(FinalizeArgs {
        tokenizer: &tokenizer,
        generated_tokens: &generated_tokens,
        finish_reason,
        think_end_id: close.think_end_id,
        think_end_str: close.think_end_str.as_deref(),
        think_end_extra_ids: &close.think_end_extra_ids,
        performance,
        include_reasoning: params.include_reasoning,
        thinking_enabled: thinking.enabled,
        prompt_tokens: reported_prompt_tokens,
        reasoning,
    })?;
    if B::OVERWRITE_THINKING_ENABLED {
        result.thinking_enabled = resolve_enable_thinking(args.config).unwrap_or(true);
    }
    result.cached_tokens = prefix.cached_prefix_len as u32;

    if let (Some(sink), Some(ts)) = (args.sink, turn_streaming.as_mut()) {
        ts.emitter.finish(&result, sink);
        return Ok(TurnOutput::Streamed);
    }
    Ok(TurnOutput::Complete(Box::new(result)))
}

#[cfg(test)]
mod tests {
    //! `run_dspark_turn` tests over a scripted [`MockDsparkStepper`] — NO
    //! model, NO Metal-heavy ops (tiny vocab-16 arrays only). Mirrors the
    //! `mtp_turn` mock-test approach: a per-cycle script drives
    //! propose/verify deterministically and an ordered call ledger records
    //! every stepper call so the tests assert the REAL sequencing
    //! (propose → verify → commit(keep, total) → eval_boundary), the
    //! emitted token stream, the finish-reason strings, `last_in_cache`,
    //! and the profiler's acceptance stats.

    use std::cell::{Cell, RefCell};
    use std::rc::Rc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    use napi::bindgen_prelude::*;

    use crate::array::MxArray;
    use crate::decode_profiler::DecodeProfiler;
    use crate::engine::backend::{
        ChatBackend, ChunkSink, DefaultStreamEmitter, DsparkBackend, DsparkProposal, DsparkStepper,
        DsparkVerifyOutput, FinalizeArgs, ResetScope, SaveStateArgs, TurnSetup, TurnTokenObserver,
    };
    use crate::engine::decode::StreamingCtx;
    use crate::engine::params::ChatParams;
    use crate::engine::penalties::{ReasoningClose, ReasoningOutcome, ReasoningTracker};
    use crate::engine::types::{ChatResult, ChatStreamChunk};
    use crate::sampling::SamplingConfig;
    use crate::stream::Stream;
    use crate::tokenizer::Qwen3Tokenizer;

    use super::{
        DSPARK_BREAK_EVEN_SPEC_CYCLES, DsparkBreakEvenChoice, DsparkBreakEvenPolicy,
        DsparkMeasurementCycle, DsparkTurnArgs, accept_dspark_proposal, apply_all_penalties,
        penalized_greedy_accept, run_dspark_turn,
    };

    #[test]
    fn break_even_policy_falls_back_when_speculation_loses() {
        let mut policy = DsparkBreakEvenPolicy::new(true);
        assert_eq!(policy.plan_cycle(7), (0, DsparkMeasurementCycle::ArProbe));
        assert!(
            policy
                .record_cycle(DsparkMeasurementCycle::ArProbe, 10_000_000, 1)
                .is_none()
        );

        for cycle in 0..DSPARK_BREAK_EVEN_SPEC_CYCLES {
            assert_eq!(policy.plan_cycle(7), (7, DsparkMeasurementCycle::SpecProbe));
            let decision = policy.record_cycle(DsparkMeasurementCycle::SpecProbe, 100_000_000, 2);
            if cycle + 1 < DSPARK_BREAK_EVEN_SPEC_CYCLES {
                assert!(decision.is_none());
            } else {
                let decision = decision.expect("final calibration cycle must decide");
                assert_eq!(decision.choice, DsparkBreakEvenChoice::FallBackToAr);
                assert!(decision.spec_tokens_per_second < decision.ar_tokens_per_second);
            }
        }
        assert_eq!(
            policy.plan_cycle(7),
            (0, DsparkMeasurementCycle::None),
            "a losing draft must stay in target-only mode"
        );
    }

    #[test]
    fn break_even_policy_keeps_profitable_speculation() {
        let mut policy = DsparkBreakEvenPolicy::new(true);
        assert!(
            policy
                .record_cycle(DsparkMeasurementCycle::ArProbe, 10_000_000, 1)
                .is_none()
        );
        let mut decision = None;
        for _ in 0..DSPARK_BREAK_EVEN_SPEC_CYCLES {
            decision = policy.record_cycle(DsparkMeasurementCycle::SpecProbe, 10_000_000, 2);
        }
        let decision = decision.expect("calibration must decide after two speculative cycles");
        assert_eq!(decision.choice, DsparkBreakEvenChoice::KeepSpeculation);
        assert!(decision.spec_tokens_per_second > decision.ar_tokens_per_second);
        assert_eq!(
            policy.plan_cycle(7),
            (7, DsparkMeasurementCycle::None),
            "a winning draft must retain the caller's normal cap"
        );
    }

    #[test]
    fn break_even_policy_disabled_and_tail_cycles_are_inert() {
        let mut disabled = DsparkBreakEvenPolicy::new(false);
        assert_eq!(disabled.plan_cycle(7), (7, DsparkMeasurementCycle::None));
        assert!(
            disabled
                .record_cycle(DsparkMeasurementCycle::None, 1, 1)
                .is_none()
        );

        let enabled = DsparkBreakEvenPolicy::new(true);
        assert_eq!(
            enabled.plan_cycle(0),
            (0, DsparkMeasurementCycle::None),
            "a natural one-token tail must not consume the AR calibration probe"
        );
    }

    /// One recorded `DsparkStepper` call, tagged with the payload the loop
    /// handed it so tests assert the exact per-cycle sequencing.
    #[derive(Clone, Debug, PartialEq, Eq)]
    enum Call {
        Reserve {
            rows: usize,
            granted: bool,
        },
        Propose {
            anchor: u32,
            max_len: usize,
        },
        VerifyArProbe {
            anchor: u32,
        },
        Verify {
            ids: Vec<u32>,
        },
        /// Device-resident verify — the engine hands the stepper the
        /// `[1+L]` concat itself, so the ledger keeps only the width.
        VerifyDevice {
            len: usize,
        },
        CommitArProbe,
        Commit {
            keep: usize,
            total: usize,
        },
        EnterArFallback,
        EvalBoundary {
            token: i32,
        },
        ObserveDeviceVerification {
            rows: usize,
            raw_matches: usize,
            milliseconds_bits: u64,
            synchronize_calls: u32,
            committed_state_available: bool,
        },
    }

    /// Canned per-cycle script.
    ///
    /// `draft_ids` is what `propose` returns (a script MAY exceed the
    /// requested `max_len` to exercise the engine's over-return error).
    /// `draft_dists[i]` is the full `[vocab]` f32 proposal row `q_i`
    /// (empty on greedy scripts). Verify logits come from `verify_rows`
    /// when set (explicit full rows, for penalty-sensitive tests);
    /// otherwise each row is a one-hot spike at `verify_argmax[j]`
    /// (`j` in `0..=L`; missing entries fall back to 0).
    #[derive(Clone)]
    struct CycleScript {
        draft_ids: Vec<i32>,
        draft_dists: Vec<Vec<f32>>,
        verify_argmax: Vec<i32>,
        verify_rows: Option<Vec<Vec<f32>>>,
    }

    impl CycleScript {
        fn greedy(draft_ids: Vec<i32>, verify_argmax: Vec<i32>) -> Self {
            Self {
                draft_ids,
                draft_dists: Vec::new(),
                verify_argmax,
                verify_rows: None,
            }
        }

        fn with_rows(draft_ids: Vec<i32>, verify_rows: Vec<Vec<f32>>) -> Self {
            Self {
                draft_ids,
                draft_dists: Vec::new(),
                verify_argmax: Vec::new(),
                verify_rows: Some(verify_rows),
            }
        }
    }

    /// Full-vocab one-hot f32 row (mass 1.0 at `id`).
    fn one_hot(vocab: i64, id: i32) -> Vec<f32> {
        let mut row = vec![0.0f32; vocab as usize];
        row[id as usize] = 1.0;
        row
    }

    /// `[vocab]` logits row whose argmax is `argmax_id`: spike 10.0 there,
    /// `fill` elsewhere. `fill == 0.0` for greedy scripts (only the argmax
    /// matters); `-1e30` for sampled scripts so `softmax` is EXACTLY one-hot
    /// (f32 `exp(-1e30) == 0.0` underflow) and every accept/residual/bonus
    /// draw is deterministic BY CONSTRUCTION (the mtp_turn dense-test trick).
    fn logits_row(vocab: i64, argmax_id: i32, fill: f32) -> Vec<f32> {
        let mut row = vec![fill; vocab as usize];
        if (0..vocab as i32).contains(&argmax_id) {
            row[argmax_id as usize] = 10.0;
        }
        row
    }

    /// Scripted [`DsparkStepper`] double. Records every call into the
    /// shared ordered ledger and fabricates verify logits from the current
    /// cycle's `verify_argmax`. The cycle cursor advances on `commit` (the
    /// one call every cycle makes exactly once).
    struct MockDsparkStepper {
        vocab: i64,
        cycles: Vec<CycleScript>,
        neg_fill: f32,
        cursor: Cell<usize>,
        verify_count: Cell<usize>,
        /// Flip this flag DURING the Nth (0-based) verify — models a cancel
        /// arriving mid-cycle, between the loop-top check and the clamp.
        cancel_on_verify: Option<(usize, Arc<AtomicBool>)>,
        adaptive_ar_fallback: bool,
        ar_fallback: bool,
        propose_delay: Option<Duration>,
        /// Synthetic target graph construction/dispatch inside the dedicated
        /// probe. Calibration must include this delay.
        ar_probe_target_build_delay: Option<Duration>,
        /// 0-based cycle from which `reserve_cycle_lookahead` reports pool
        /// exhaustion — the paged steppers' `Ok(false)`.
        reserve_exhausted_from: Option<usize>,
        reserve_calls: Cell<usize>,
        /// Emit `device_draft_ids` instead of host ids — exercises the
        /// engine's `verify_device` dispatch and post-eval materialization.
        /// A sampled position script also gets the device-drawn rows.
        emit_device_proposals: bool,
        sampled_rows: bool,
        device_budget_enabled: bool,
        verify_row_budget: Option<MxArray>,
        pending_committed_state: RefCell<Option<MxArray>>,
        ledger: Rc<RefCell<Vec<Call>>>,
        /// When set, targets and drafts are read from the output index each
        /// row predicts; `committed` is the anchor's index.
        positions: Option<Positions>,
        committed: Cell<usize>,
        last_verify: RefCell<Vec<u32>>,
        spans: Rc<RefCell<Vec<Span>>>,
    }

    /// The draft at a listed output index.
    const MISS: i32 = 30;

    /// The target at output index `n` is `targets[n]`; the draft for `n` is
    /// the same except at `misses`.
    #[derive(Clone)]
    struct Positions {
        targets: Vec<i32>,
        misses: Vec<usize>,
    }

    impl Positions {
        fn target(&self, n: usize) -> i32 {
            self.targets.get(n).copied().unwrap_or(0)
        }

        fn draft(&self, n: usize) -> i32 {
            if self.misses.contains(&n) {
                MISS
            } else {
                self.target(n)
            }
        }
    }

    /// One verify: the output index its row 0 predicts, its drafts, and the
    /// tokens its commit kept.
    #[derive(Clone, Copy, Debug)]
    struct Span {
        start: usize,
        drafts: usize,
        kept: usize,
    }

    impl MockDsparkStepper {
        fn script(&self) -> CycleScript {
            self.cycles
                .get(self.cursor.get())
                .cloned()
                .unwrap_or_else(|| CycleScript::greedy(Vec::new(), Vec::new()))
        }

        fn scripted_verify_output(&self, rows: usize) -> Result<DsparkVerifyOutput> {
            let script = self.script();
            let mut flat: Vec<f32> = Vec::with_capacity(rows * self.vocab as usize);
            if let Some(positions) = &self.positions {
                let start = self.committed.get() + 1;
                self.spans.borrow_mut().push(Span {
                    start,
                    drafts: rows - 1,
                    kept: 0,
                });
                for j in 0..rows {
                    flat.extend(logits_row(
                        self.vocab,
                        positions.target(start + j),
                        self.neg_fill,
                    ));
                }
                let logits = MxArray::from_float32(&flat, &[1, rows as i64, self.vocab])?;
                return Ok(DsparkVerifyOutput { logits });
            }
            for j in 0..rows {
                match script.verify_rows.as_ref().and_then(|r| r.get(j)) {
                    Some(row) => flat.extend_from_slice(row),
                    None => {
                        let argmax_id = script.verify_argmax.get(j).copied().unwrap_or(0);
                        flat.extend(logits_row(self.vocab, argmax_id, self.neg_fill));
                    }
                }
            }
            let logits = MxArray::from_float32(&flat, &[1, rows as i64, self.vocab])?;
            let n = self.verify_count.get();
            self.verify_count.set(n + 1);
            if let Some((at, flag)) = self.cancel_on_verify.as_ref()
                && *at == n
            {
                flag.store(true, Ordering::Relaxed);
            }
            Ok(DsparkVerifyOutput { logits })
        }
    }

    impl DsparkStepper for MockDsparkStepper {
        fn supports_adaptive_ar_fallback(&self) -> bool {
            self.adaptive_ar_fallback
        }

        fn reserve_cycle_lookahead(&mut self, rows: usize) -> Result<bool> {
            let call = self.reserve_calls.get();
            self.reserve_calls.set(call + 1);
            let granted = self
                .reserve_exhausted_from
                .is_none_or(|exhausted_from| call < exhausted_from);
            self.ledger
                .borrow_mut()
                .push(Call::Reserve { rows, granted });
            Ok(granted)
        }

        fn enter_ar_fallback(&mut self) -> Result<()> {
            if !self.adaptive_ar_fallback {
                return Err(Error::from_reason("mock fallback is disabled"));
            }
            self.ar_fallback = true;
            self.ledger.borrow_mut().push(Call::EnterArFallback);
            Ok(())
        }

        fn verify_ar_probe(&mut self, anchor_id: u32) -> Result<DsparkVerifyOutput> {
            if let Some(delay) = self.ar_probe_target_build_delay {
                std::thread::sleep(delay);
            }
            self.ledger
                .borrow_mut()
                .push(Call::VerifyArProbe { anchor: anchor_id });
            self.scripted_verify_output(1)
        }

        fn commit_ar_probe(&mut self) -> Result<()> {
            self.ledger.borrow_mut().push(Call::CommitArProbe);
            self.committed.set(self.committed.get() + 1);
            self.cursor.set(self.cursor.get() + 1);
            Ok(())
        }

        fn propose(
            &mut self,
            anchor_id: u32,
            max_len: usize,
            params: &ChatParams,
            _rng: &mut dyn rand::Rng,
        ) -> Result<DsparkProposal> {
            if self.ar_fallback {
                return Err(Error::from_reason("mock propose after AR fallback"));
            }
            if let Some(delay) = self.propose_delay {
                std::thread::sleep(delay);
            }
            self.ledger.borrow_mut().push(Call::Propose {
                anchor: anchor_id,
                max_len,
            });
            if let Some(positions) = &self.positions {
                let draft_ids = (0..max_len)
                    .map(|i| positions.draft(self.committed.get() + 1 + i))
                    .collect::<Vec<_>>();
                let greedy = crate::sampling::is_greedy_temperature(
                    params
                        .sampling_config
                        .and_then(|c| c.temperature)
                        .unwrap_or(1.0),
                );
                if self.emit_device_proposals {
                    let len = draft_ids.len() as i64;
                    let device_draft_sparse = if greedy {
                        None
                    } else {
                        let ids = draft_ids
                            .iter()
                            .flat_map(|&id| [id, (id + 1) % self.vocab as i32])
                            .collect::<Vec<_>>();
                        let probs = draft_ids
                            .iter()
                            .flat_map(|_| [1.0f32, 0.0])
                            .collect::<Vec<_>>();
                        Some(crate::engine::backend::DeviceSparseRows {
                            ids: MxArray::from_int32(&ids, &[len, 2])?,
                            probs: MxArray::from_float32(&probs, &[len, 2])?,
                            vocab_size: self.vocab as usize,
                        })
                    };
                    return Ok(DsparkProposal {
                        draft_ids: Vec::new(),
                        device_draft_ids: Some(MxArray::from_int32(&draft_ids, &[len])?),
                        device_draft_sparse,
                        draft_dists: Vec::new(),
                        draft_sparse_dists: Vec::new(),
                        keep_probabilities: None,
                    });
                }
                let draft_dists = if greedy {
                    Vec::new()
                } else {
                    draft_ids
                        .iter()
                        .map(|&id| MxArray::from_float32(&one_hot(self.vocab, id), &[self.vocab]))
                        .collect::<Result<Vec<_>>>()?
                };
                return Ok(DsparkProposal {
                    draft_ids,
                    device_draft_ids: None,
                    device_draft_sparse: None,
                    draft_dists,
                    draft_sparse_dists: Vec::new(),
                    keep_probabilities: None,
                });
            }
            let script = self.script();
            self.verify_row_budget = (self.device_budget_enabled && params.mtp_adaptive_depth)
                .then(|| MxArray::from_int32(&[script.draft_ids.len() as i32 + 1], &[1]))
                .transpose()?;
            let draft_dists = script
                .draft_dists
                .iter()
                .map(|row| MxArray::from_float32(row, &[self.vocab]))
                .collect::<Result<Vec<_>>>()?;
            let (draft_ids, device_draft_ids) = if self.emit_device_proposals {
                (
                    Vec::new(),
                    Some(MxArray::from_int32(
                        &script.draft_ids,
                        &[script.draft_ids.len() as i64],
                    )?),
                )
            } else {
                (script.draft_ids.clone(), None)
            };
            Ok(DsparkProposal {
                draft_ids,
                device_draft_ids,
                device_draft_sparse: None,
                draft_dists,
                draft_sparse_dists: Vec::new(),
                keep_probabilities: None,
            })
        }

        fn verify(&mut self, verify_ids: &[u32]) -> Result<DsparkVerifyOutput> {
            if self.ar_fallback && verify_ids.len() != 1 {
                return Err(Error::from_reason(
                    "mock AR fallback requires a one-token verify",
                ));
            }
            self.ledger.borrow_mut().push(Call::Verify {
                ids: verify_ids.to_vec(),
            });
            *self.last_verify.borrow_mut() = verify_ids.to_vec();
            self.scripted_verify_output(verify_ids.len())
        }

        fn verify_device(&mut self, verify_ids: &MxArray) -> Result<DsparkVerifyOutput> {
            let len = usize::try_from(verify_ids.shape_at(0)?)
                .map_err(|_| Error::from_reason("mock verify_device bad len"))?;
            self.ledger.borrow_mut().push(Call::VerifyDevice { len });
            if self.positions.is_some() {
                *self.last_verify.borrow_mut() =
                    verify_ids.to_int32()?.iter().map(|&id| id as u32).collect();
            }
            self.scripted_verify_output(len)
        }

        fn device_verification_budget(&self) -> Option<&MxArray> {
            self.verify_row_budget.as_ref()
        }

        fn device_sampled_rows(&self) -> bool {
            self.sampled_rows
        }

        fn observe_device_verification(
            &mut self,
            rows: usize,
            raw_matches: usize,
            milliseconds: f64,
        ) -> Result<()> {
            let pending = self
                .pending_committed_state
                .take()
                .ok_or_else(|| Error::from_reason("mock observation has no committed state"))?;
            let committed_state_available =
                unsafe { mlx_sys::mlx_array_is_available(pending.as_raw_ptr()) };
            let synchronize_calls =
                crate::array::memory::TEST_SYNCHRONIZE_CALLS.with(|calls| calls.get());
            self.ledger
                .borrow_mut()
                .push(Call::ObserveDeviceVerification {
                    rows,
                    raw_matches,
                    milliseconds_bits: milliseconds.to_bits(),
                    synchronize_calls,
                    committed_state_available,
                });
            self.verify_row_budget = None;
            Ok(())
        }

        fn commit(&mut self, keep: usize, total_written: usize) -> Result<()> {
            self.ledger.borrow_mut().push(Call::Commit {
                keep,
                total: total_written,
            });
            self.committed.set(self.committed.get() + keep);
            if self.positions.is_some()
                && let Some(span) = self.spans.borrow_mut().last_mut()
            {
                span.kept = keep;
            }
            if self.verify_row_budget.is_some() {
                let pending = MxArray::from_int32(&[keep as i32], &[1])?.add_scalar(1.0)?;
                *self.pending_committed_state.borrow_mut() = Some(pending);
            }
            self.cursor.set(self.cursor.get() + 1);
            Ok(())
        }

        fn commit_with_provenance(
            &mut self,
            keep: usize,
            total_written: usize,
            verified_ids: &[u32],
        ) -> Result<()> {
            if self.positions.is_some() {
                if verified_ids != self.last_verify.borrow().as_slice()
                    || verified_ids.len() != total_written
                {
                    return Err(Error::from_reason(format!(
                        "mock commit provenance mismatch: ids={verified_ids:?}, verified={:?}",
                        self.last_verify.borrow()
                    )));
                }
                return self.commit(keep, total_written);
            }
            let expected_drafts = self
                .script()
                .draft_ids
                .iter()
                .take(total_written.saturating_sub(1))
                .map(|&id| id as u32)
                .collect::<Vec<_>>();
            if verified_ids.len() != total_written
                || verified_ids.get(1..) != Some(expected_drafts.as_slice())
            {
                return Err(Error::from_reason(format!(
                    "mock commit provenance mismatch: ids={verified_ids:?}, total={total_written}, drafts={expected_drafts:?}"
                )));
            }
            self.commit(keep, total_written)
        }

        fn eval_boundary(&self, token: &MxArray) {
            let pending = self.pending_committed_state.borrow();
            if let Some(pending) = pending.as_ref() {
                MxArray::async_eval_arrays(&[token, pending]);
            } else {
                token.eval();
            }
            let id = token.item_at_int32(0).unwrap_or(-1);
            self.ledger
                .borrow_mut()
                .push(Call::EvalBoundary { token: id });
        }
    }

    /// A never-constructed `DecodeStep` so `MockDsparkBackend` satisfies the
    /// `ChatBackend::Decode<'a>` bound — `begin_decode` is never called on
    /// the DSpark path.
    struct NeverDecode;
    impl crate::engine::backend::DecodeStep for NeverDecode {
        fn forward(&mut self, _input_ids: &MxArray) -> Result<(MxArray, bool)> {
            Err(Error::from_reason("NeverDecode::forward must not run"))
        }
        fn eval_step(&mut self, _next_token: &MxArray, _logits: &MxArray, _budget_forced: bool) {}
    }

    /// Scripted [`DsparkBackend`]. `begin_dspark_decode` builds a fresh
    /// stepper wired to the shared ledger so tests read the call sequence
    /// after the turn.
    struct MockDsparkBackend {
        vocab: i64,
        cycles: Vec<CycleScript>,
        neg_fill: f32,
        cancel_on_verify: Option<(usize, Arc<AtomicBool>)>,
        adaptive_ar_fallback: bool,
        propose_delay: Option<Duration>,
        ar_probe_target_build_delay: Option<Duration>,
        reserve_exhausted_from: Option<usize>,
        /// Drafted block width the plan sizes the lookahead reservation
        /// from; `None` advertises no speculative plan at all.
        draft_block_size: Option<usize>,
        /// Forwarded to the stepper — proposals carry `device_draft_ids`.
        emit_device_proposals: bool,
        sampled_rows: bool,
        device_budget_enabled: bool,
        ledger: Rc<RefCell<Vec<Call>>>,
        begin_calls: Cell<usize>,
        /// `block_size` observed by `begin_dspark_decode` (setup plumbing).
        seen_block_size: Cell<usize>,
        positions: Option<Positions>,
        spans: Rc<RefCell<Vec<Span>>>,
    }

    impl MockDsparkBackend {
        fn greedy(vocab: i64, cycles: Vec<CycleScript>) -> Self {
            Self {
                vocab,
                cycles,
                neg_fill: 0.0,
                cancel_on_verify: None,
                adaptive_ar_fallback: false,
                propose_delay: None,
                ar_probe_target_build_delay: None,
                reserve_exhausted_from: None,
                draft_block_size: None,
                emit_device_proposals: false,
                sampled_rows: false,
                device_budget_enabled: false,
                ledger: Rc::new(RefCell::new(Vec::new())),
                begin_calls: Cell::new(0),
                seen_block_size: Cell::new(usize::MAX),
                positions: None,
                spans: Rc::new(RefCell::new(Vec::new())),
            }
        }

        /// A turn over a position script (vocab 32); `sampled` gives one-hot
        /// verify rows for the sampled path.
        fn positions(targets: Vec<i32>, misses: Vec<usize>, sampled: bool) -> Self {
            let mut backend = if sampled {
                Self::sampled(32, Vec::new())
            } else {
                Self::greedy(32, Vec::new())
            };
            backend.positions = Some(Positions { targets, misses });
            backend
        }

        fn spans(&self) -> Vec<Span> {
            self.spans.borrow().clone()
        }

        /// Sampled-path backend: `-1e30` fill makes verify softmax rows
        /// exactly one-hot (see [`logits_row`]).
        fn sampled(vocab: i64, cycles: Vec<CycleScript>) -> Self {
            let mut b = Self::greedy(vocab, cycles);
            b.neg_fill = -1.0e30;
            b
        }

        /// Advertise a draft-model speculative plan of `block_size` and
        /// exhaust the pool from the `exhausted_from`-th cycle on.
        fn with_lookahead_exhausted_from(
            mut self,
            block_size: usize,
            exhausted_from: usize,
        ) -> Self {
            self.draft_block_size = Some(block_size);
            self.reserve_exhausted_from = Some(exhausted_from);
            self
        }

        /// Stepper emits `device_draft_ids` proposals — the engine must
        /// dispatch `verify_device` and materialize ids post-eval.
        fn with_device_proposals(mut self) -> Self {
            self.emit_device_proposals = true;
            self
        }

        /// Stepper asks for the device rows of its sampled cycles.
        fn with_sampled_rows(mut self) -> Self {
            self.sampled_rows = true;
            self
        }

        fn with_device_budget(mut self) -> Self {
            self.device_budget_enabled = true;
            self
        }

        fn with_slow_adaptive_fallback(mut self) -> Self {
            self.adaptive_ar_fallback = true;
            self.propose_delay = Some(Duration::from_millis(100));
            self
        }

        fn with_slow_ar_probe_target_build(mut self) -> Self {
            self.adaptive_ar_fallback = true;
            self.propose_delay = Some(Duration::from_millis(20));
            self.ar_probe_target_build_delay = Some(Duration::from_millis(100));
            self
        }

        fn ledger_snapshot(&self) -> Vec<Call> {
            self.ledger.borrow().clone()
        }
    }

    // Minimal `ChatBackend` surface — `run_dspark_turn` calls NONE of these.
    impl ChatBackend for MockDsparkBackend {
        fn tokenizer(&self) -> Result<Arc<Qwen3Tokenizer>> {
            Err(Error::from_reason(
                "MockDsparkBackend::tokenizer must not run",
            ))
        }
        fn family_name(&self) -> &'static str {
            "mock_dspark"
        }
        fn session_eos_id(&self, _tok: &Qwen3Tokenizer) -> Result<u32> {
            Ok(u32::MAX)
        }
        fn cached_token_history(&self) -> &[u32] {
            &[]
        }
        fn reset_caches(&mut self, _scope: ResetScope) -> Result<()> {
            Ok(())
        }
        fn verify_cache_prefix(&self, _tokens: &[u32], _reuse_cache: bool) -> usize {
            0
        }
        fn save_cache_state(&mut self, _args: SaveStateArgs<'_>) {}
        fn eval_caches(&self) -> Result<()> {
            Ok(())
        }
        fn prefill(&mut self, _prompt_tokens: &[u32], _stream: Stream) -> Result<MxArray> {
            Err(Error::from_reason(
                "MockDsparkBackend::prefill must not run",
            ))
        }

        type Decode<'a>
            = NeverDecode
        where
            Self: 'a;

        fn begin_decode(&mut self, _turn: &TurnSetup<'_>) -> Result<Self::Decode<'_>> {
            Err(Error::from_reason(
                "MockDsparkBackend::begin_decode must not run on the DSpark path",
            ))
        }

        fn finalize_turn(&self, _args: FinalizeArgs<'_>) -> Result<ChatResult> {
            Err(Error::from_reason(
                "MockDsparkBackend::finalize_turn must not run",
            ))
        }

        fn execution_plan(&self) -> crate::engine::plan::ExecutionPlan {
            crate::engine::plan::ExecutionPlan {
                speculative: self
                    .draft_block_size
                    .map(|_| crate::engine::plan::SpeculativePlan {
                        kind: crate::engine::plan::SpeculativeKind::DraftModel,
                        supported_input_media: crate::engine::plan::MediaCapabilities::NONE,
                        supported_context_media: crate::engine::plan::MediaCapabilities::NONE,
                        supports_paged_attention: true,
                        supports_streaming: true,
                    }),
                ..crate::engine::plan::ExecutionPlan::TEXT_ONLY
            }
        }
    }

    impl DsparkBackend for MockDsparkBackend {
        type DsparkDecode<'a>
            = MockDsparkStepper
        where
            Self: 'a;

        fn begin_dspark_decode(&mut self, block_size: usize) -> Result<Self::DsparkDecode<'_>> {
            self.begin_calls.set(self.begin_calls.get() + 1);
            self.seen_block_size.set(block_size);
            Ok(MockDsparkStepper {
                vocab: self.vocab,
                cycles: self.cycles.clone(),
                neg_fill: self.neg_fill,
                cursor: Cell::new(0),
                verify_count: Cell::new(0),
                cancel_on_verify: self.cancel_on_verify.clone(),
                adaptive_ar_fallback: self.adaptive_ar_fallback,
                ar_fallback: false,
                propose_delay: self.propose_delay,
                ar_probe_target_build_delay: self.ar_probe_target_build_delay,
                reserve_exhausted_from: self.reserve_exhausted_from,
                reserve_calls: Cell::new(0),
                emit_device_proposals: self.emit_device_proposals,
                sampled_rows: self.sampled_rows,
                device_budget_enabled: self.device_budget_enabled,
                verify_row_budget: None,
                pending_committed_state: RefCell::new(None),
                ledger: Rc::clone(&self.ledger),
                positions: self.positions.clone(),
                committed: Cell::new(0),
                last_verify: RefCell::new(Vec::new()),
                spans: Rc::clone(&self.spans),
            })
        }
    }

    /// T=0 greedy `ChatParams` with all penalties at their no-op defaults —
    /// drives the loop's batched-argmax fast path.
    fn greedy_params() -> ChatParams {
        ChatParams {
            cache_salt: 0,
            cache_owner_id: String::new(),
            cache_root_owner_id: None,
            max_new_tokens: 64,
            repetition_penalty: 1.0,
            repetition_context_size: 0,
            presence_penalty: 0.0,
            presence_context_size: 0,
            frequency_penalty: 0.0,
            frequency_context_size: 0,
            max_consecutive_tokens: 0,
            max_ngram_repeats: 0,
            ngram_size: 0,
            sampling_config: Some(SamplingConfig {
                temperature: Some(0.0),
                top_k: Some(0),
                top_p: Some(1.0),
                min_p: Some(0.0),
            }),
            report_performance: false,
            reuse_cache: true,
            include_reasoning: true,
            extra_eos_ids: Vec::new(),
            enable_mtp: true,
            mtp_depth: 2,
            mtp_adaptive_depth: false,
        }
    }

    /// T=1.0 `ChatParams` — drives the sampled `accept_with_residual` path
    /// (deterministic via the one-hot verify rows / proposal dists).
    fn dense_params() -> ChatParams {
        ChatParams {
            sampling_config: Some(SamplingConfig {
                temperature: Some(1.0),
                top_k: Some(0),
                top_p: Some(1.0),
                min_p: Some(0.0),
            }),
            ..greedy_params()
        }
    }

    struct TurnOut {
        generated: Vec<u32>,
        profiled_generated_tokens: u64,
        profiled_decode_tokens: u64,
        finish_reason: String,
        last_in_cache: bool,
        ledger: Vec<Call>,
        /// `DecodeProfiler::mtp_acceptance_summary()` — the
        /// `record_mtp_cycle` seam: `(mean accepted drafts per cycle,
        /// per-position accept rate, cycles)`, `None` when no cycle
        /// recorded.
        acceptance: Option<(f64, Vec<f64>, u32)>,
    }

    /// Raw outcome of a SYNC turn drive — keeps the `Result` so contract-
    /// error tests can assert the failure alongside the ledger state.
    struct RawTurnOut {
        outcome: ReasoningOutcome,
        result: Result<super::DsparkTurnOutcome>,
        generated: Vec<u32>,
        profiled_generated_tokens: u64,
        profiled_decode_tokens: u64,
        finish_reason: String,
        acceptance: Option<(f64, Vec<f64>, u32)>,
    }

    /// Drive the SYNC (non-streaming) turn over the scripted backend with a
    /// caller-supplied RNG, without unwrapping the loop result.
    fn drive_turn_raw<R: rand::Rng>(
        backend: &mut MockDsparkBackend,
        params: ChatParams,
        first_token: u32,
        eos_id: u32,
        block_size: usize,
        rng: &mut R,
    ) -> RawTurnOut {
        drive_turn_raw_with_cancel(
            backend,
            params,
            first_token,
            eos_id,
            block_size,
            rng,
            None,
            None,
        )
    }

    /// [`drive_turn_raw`] with an H2 sync cancel flag threaded into
    /// `DsparkTurnArgs.cancel_flag` — the streaming ctx stays `None`, so any
    /// "cancelled" outcome can ONLY come from the ungated sync polls.
    fn drive_turn_raw_with_cancel<R: rand::Rng>(
        backend: &mut MockDsparkBackend,
        params: ChatParams,
        first_token: u32,
        eos_id: u32,
        block_size: usize,
        rng: &mut R,
        cancel_flag: Option<&AtomicBool>,
        turn_token_observer: Option<Box<dyn TurnTokenObserver>>,
    ) -> RawTurnOut {
        drive_turn_with_tracker(
            backend,
            params,
            first_token,
            eos_id,
            block_size,
            rng,
            cancel_flag,
            turn_token_observer,
            ReasoningTracker::new(false, None, &[]),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn drive_turn_with_tracker<R: rand::Rng>(
        backend: &mut MockDsparkBackend,
        params: ChatParams,
        first_token: u32,
        eos_id: u32,
        block_size: usize,
        rng: &mut R,
        cancel_flag: Option<&AtomicBool>,
        turn_token_observer: Option<Box<dyn TurnTokenObserver>>,
        mut tracker: ReasoningTracker,
    ) -> RawTurnOut {
        let mut profiler = DecodeProfiler::new("dspark_turn_test", "test");
        let mut generated: Vec<u32> = Vec::new();
        let mut token_history: Vec<u32> = Vec::new();
        let mut finish_reason = String::from("length");
        let mut first_token_instant: Option<Instant> = None;
        let y = MxArray::from_int32(&[first_token as i32], &[1])
            .unwrap_or_else(|e| panic!("y construction: {}", e.reason));
        let generation_stream = Stream::generation();
        let max_new_tokens = params.max_new_tokens;

        let result = run_dspark_turn(
            backend,
            rng,
            DsparkTurnArgs {
                y,
                block_size,
                params: &params,
                reasoning_tracker: &mut tracker,
                profiler: &mut profiler,
                max_new_tokens,
                eos_id,
                generated_tokens: &mut generated,
                token_history: &mut token_history,
                finish_reason: &mut finish_reason,
                first_token_instant: &mut first_token_instant,
                report_perf: false,
                generation_stream,
                cancel_flag,
                turn_token_observer,
            },
            None,
        );

        // The commit-exactly-once design keeps the histories in lockstep on
        // BOTH the success and the error path (emission always pushes both).
        assert_eq!(token_history, generated, "history must mirror generated");
        let (profiled_generated_tokens, profiled_decode_tokens) = profiler.token_counts_for_test();

        RawTurnOut {
            outcome: tracker.outcome(),
            result,
            generated,
            profiled_generated_tokens,
            profiled_decode_tokens,
            finish_reason,
            acceptance: profiler.mtp_acceptance_summary(),
        }
    }

    /// Drive the SYNC (non-streaming) turn over the scripted backend.
    fn drive_turn(
        backend: &mut MockDsparkBackend,
        params: ChatParams,
        first_token: u32,
        eos_id: u32,
        block_size: usize,
    ) -> TurnOut {
        let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(0xD5_9A2B_C0DE);
        let raw = drive_turn_raw(backend, params, first_token, eos_id, block_size, &mut rng);
        let outcome = raw
            .result
            .unwrap_or_else(|e| panic!("run_dspark_turn failed: {}", e.reason));

        TurnOut {
            generated: raw.generated,
            profiled_generated_tokens: raw.profiled_generated_tokens,
            profiled_decode_tokens: raw.profiled_decode_tokens,
            finish_reason: raw.finish_reason,
            last_in_cache: outcome.last_in_cache,
            ledger: backend.ledger_snapshot(),
            acceptance: raw.acceptance,
        }
    }

    /// Counting RNG wrapper — proves the greedy accept paths consume ZERO
    /// random draws. rand 0.10: implementing `TryRng<Error = Infallible>`
    /// yields the infallible `Rng` via the blanket impl.
    struct CountingRng {
        inner: rand::rngs::StdRng,
        draws: Rc<Cell<usize>>,
    }

    impl rand::TryRng for CountingRng {
        type Error = std::convert::Infallible;

        fn try_next_u32(&mut self) -> std::result::Result<u32, Self::Error> {
            self.draws.set(self.draws.get() + 1);
            rand::TryRng::try_next_u32(&mut self.inner)
        }

        fn try_next_u64(&mut self) -> std::result::Result<u64, Self::Error> {
            self.draws.set(self.draws.get() + 1);
            rand::TryRng::try_next_u64(&mut self.inner)
        }

        fn try_fill_bytes(&mut self, dst: &mut [u8]) -> std::result::Result<(), Self::Error> {
            self.draws.set(self.draws.get() + 1);
            rand::TryRng::try_fill_bytes(&mut self.inner, dst)
        }
    }

    fn count(ledger: &[Call], pred: impl Fn(&Call) -> bool) -> usize {
        ledger.iter().filter(|c| pred(c)).count()
    }

    fn commits(ledger: &[Call]) -> Vec<(usize, usize)> {
        ledger
            .iter()
            .filter_map(|c| match c {
                Call::CommitArProbe => Some((1, 1)),
                Call::Commit { keep, total } => Some((*keep, *total)),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn adaptive_fallback_excludes_ar_cycles_from_mtp_metrics() {
        // The real target is already warm from prefill. Warm the tiny mock's
        // first Metal argmax too, so shader compilation is not mistaken for
        // target AR latency by this sequencing/metrics test.
        let warm_logits = MxArray::from_float32(&[0.0; 16], &[1, 1, 16]).expect("warm logits");
        let warm_argmax = warm_logits.argmax(-1, None).expect("warm argmax");
        warm_argmax.eval();
        let _ = warm_argmax.item_at_int32(0).expect("warm argmax value");

        let mut backend = MockDsparkBackend::greedy(
            16,
            vec![
                // Target-only AR baseline probe.
                CycleScript::greedy(Vec::new(), vec![1]),
                // Two deliberately slow, full-accept speculative probes.
                CycleScript::greedy(vec![2, 3], vec![2, 3, 4]),
                CycleScript::greedy(vec![5, 6], vec![5, 6, 7]),
                // Target-only fallback tail.
                CycleScript::greedy(Vec::new(), vec![8]),
            ],
        )
        .with_slow_adaptive_fallback();
        let mut p = greedy_params();
        p.max_new_tokens = 9;
        p.mtp_adaptive_depth = true;
        let out = drive_turn(&mut backend, p, 0, 15, 2);

        assert_eq!(out.generated, (0..=8).collect::<Vec<_>>());
        assert!(
            !out.last_in_cache,
            "the final AR boundary remains unmaterialized for family save parity"
        );
        assert_eq!(
            commits(&out.ledger),
            vec![(1, 1), (3, 3), (3, 3), (1, 1)],
            "AR probe/fallback keep only the verified anchor; speculative probes keep their accepted drafts"
        );
        assert_eq!(
            count(&out.ledger, |c| matches!(c, Call::Propose { .. })),
            2,
            "only the two speculative calibration cycles may propose"
        );
        assert_eq!(
            count(&out.ledger, |c| matches!(c, Call::EnterArFallback)),
            1,
            "the losing draft must switch exactly once"
        );
        let fallback_at = out
            .ledger
            .iter()
            .position(|c| matches!(c, Call::EnterArFallback))
            .expect("fallback call recorded");
        assert!(
            out.ledger[fallback_at + 1..]
                .iter()
                .all(|c| !matches!(c, Call::Propose { .. })),
            "target-only fallback must never propose"
        );

        let (mean_accepted, _, cycles) = out
            .acceptance
            .expect("the two speculative probes must be profiled");
        assert_eq!(cycles, 2, "AR probe/fallback cycles are not MTP cycles");
        assert!((mean_accepted - 2.0).abs() < 1e-9);
    }

    #[test]
    fn adaptive_ar_probe_includes_target_graph_build_in_timer() {
        // Warm the exact target-force primitive used by the mock. The test
        // then puts 100 ms of synthetic TARGET graph construction/dispatch
        // inside the dedicated AR probe and 20 ms in each speculative
        // proposal. Apples-to-apples timing must include the 100 ms, making
        // speculation win. The old biased timer started after verify and
        // incorrectly chose target-only fallback for this exact script.
        let warm_logits = MxArray::from_float32(&[0.0; 16], &[1, 1, 16]).expect("warm logits");
        let warm_argmax = warm_logits.argmax(-1, None).expect("warm argmax");
        warm_argmax.eval();
        let _ = warm_argmax.item_at_int32(0).expect("warm argmax value");

        let mut backend = MockDsparkBackend::greedy(
            16,
            vec![
                CycleScript::greedy(Vec::new(), vec![1]),
                CycleScript::greedy(vec![2, 3], vec![2, 3, 4]),
                CycleScript::greedy(vec![5, 6], vec![5, 6, 7]),
                CycleScript::greedy(Vec::new(), vec![8]),
            ],
        )
        .with_slow_ar_probe_target_build();
        let mut p = greedy_params();
        p.max_new_tokens = 9;
        p.mtp_adaptive_depth = true;
        let out = drive_turn(&mut backend, p, 0, 15, 2);

        assert_eq!(out.generated, (0..=8).collect::<Vec<_>>());
        assert_eq!(
            count(&out.ledger, |c| matches!(c, Call::EnterArFallback)),
            0,
            "target graph construction/dispatch must be charged to the AR probe"
        );
        assert!(
            matches!(out.ledger.first(), Some(Call::VerifyArProbe { anchor: 0 })),
            "calibration must use the dedicated no-rollback probe path"
        );
        assert_eq!(
            count(&out.ledger, |c| matches!(c, Call::CommitArProbe)),
            1,
            "the dedicated probe must be committed exactly once"
        );
    }

    #[test]
    fn adaptive_guard_preserves_fixed_dspark_on_short_turns() {
        let mut backend =
            MockDsparkBackend::greedy(16, vec![CycleScript::greedy(vec![1, 2], vec![1, 2, 3])])
                .with_slow_adaptive_fallback();
        let mut p = greedy_params();
        p.max_new_tokens = 4;
        p.mtp_adaptive_depth = true;
        let out = drive_turn(&mut backend, p, 0, 15, 2);

        assert_eq!(out.generated, vec![0, 1, 2, 3]);
        assert!(matches!(out.ledger.first(), Some(Call::Propose { .. })));
        assert_eq!(
            count(&out.ledger, |c| matches!(c, Call::EnterArFallback)),
            0,
            "a budget too short to finish calibration keeps the old schedule"
        );
        let (_, _, cycles) = out.acceptance.expect("fixed DSpark cycle profiled");
        assert_eq!(cycles, 1);
    }

    #[test]
    fn device_budget_observation_settles_commit_without_changing_fixed_schedule() {
        fn observations(ledger: &[Call]) -> Vec<(usize, usize, f64, u32, bool)> {
            ledger
                .iter()
                .filter_map(|call| match call {
                    Call::ObserveDeviceVerification {
                        rows,
                        raw_matches,
                        milliseconds_bits,
                        synchronize_calls,
                        committed_state_available,
                    } => Some((
                        *rows,
                        *raw_matches,
                        f64::from_bits(*milliseconds_bits),
                        *synchronize_calls,
                        *committed_state_available,
                    )),
                    _ => None,
                })
                .collect()
        }

        let scripts = vec![
            CycleScript::greedy(vec![4, 5], vec![4, 5, 6]),
            CycleScript::greedy(vec![7, 8], vec![7, 8, 9]),
        ];
        let mut backend = MockDsparkBackend::greedy(16, scripts.clone()).with_device_budget();
        let mut params = greedy_params();
        params.max_new_tokens = 7;
        params.mtp_depth = 2;
        params.mtp_adaptive_depth = true;
        crate::array::memory::TEST_SYNCHRONIZE_CALLS.with(|calls| calls.set(0));
        let out = drive_turn(&mut backend, params.clone(), 3, 15, 2);

        assert_eq!(out.generated, vec![3, 4, 5, 6, 7, 8, 9]);
        let observed = observations(&out.ledger);
        assert_eq!(observed.len(), 2);
        for (index, &(rows, raw_matches, milliseconds, synchronizes, available)) in
            observed.iter().enumerate()
        {
            assert_eq!((rows, raw_matches), (3, 2));
            assert!(milliseconds.is_finite() && milliseconds > 0.0);
            assert_eq!(synchronizes, index as u32 + 1);
            assert!(
                available,
                "cycle {index} observed an unfinished committed root"
            );
        }
        let first_observation = out
            .ledger
            .iter()
            .position(|call| matches!(call, Call::ObserveDeviceVerification { .. }))
            .expect("first observation");
        let second_proposal = out
            .ledger
            .iter()
            .enumerate()
            .filter(|(_, call)| matches!(call, Call::Propose { .. }))
            .nth(1)
            .map(|(index, _)| index)
            .expect("second proposal");
        assert!(first_observation < second_proposal);
        crate::array::memory::TEST_SYNCHRONIZE_CALLS.with(|calls| assert_eq!(calls.get(), 2));

        // The terminal path uses the same committed-state boundary and must
        // not leave its observation for finish or a later turn.
        let mut stopping =
            MockDsparkBackend::greedy(16, vec![scripts[0].clone()]).with_device_budget();
        crate::array::memory::TEST_SYNCHRONIZE_CALLS.with(|calls| calls.set(0));
        let stopped = drive_turn(&mut stopping, params, 3, 6, 2);
        assert_eq!(stopped.generated, vec![3, 4, 5, 6]);
        assert_eq!(stopped.finish_reason, "stop");
        let observed = observations(&stopped.ledger);
        assert_eq!(observed.len(), 1);
        assert_eq!(observed[0].3, 1);
        assert!(observed[0].4);

        // A fixed-depth request on the same stepper adds neither an adaptive
        // synchronization nor an observation.
        let mut fixed =
            MockDsparkBackend::greedy(16, vec![scripts[0].clone()]).with_device_budget();
        let mut fixed_params = greedy_params();
        fixed_params.max_new_tokens = 4;
        fixed_params.mtp_depth = 2;
        crate::array::memory::TEST_SYNCHRONIZE_CALLS.with(|calls| calls.set(0));
        let fixed = drive_turn(&mut fixed, fixed_params, 3, 15, 2);
        assert_eq!(fixed.generated, vec![3, 4, 5, 6]);
        assert!(observations(&fixed.ledger).is_empty());
        crate::array::memory::TEST_SYNCHRONIZE_CALLS.with(|calls| assert_eq!(calls.get(), 0));
    }

    // ---- 1. emission order + per-cycle commit ledger ---------------------

    /// Pool exhaustion mid-turn drops the DRAFT BLOCK, not the turn: the
    /// exhausted cycle skips `propose` entirely and takes ONE autoregressive
    /// step through `verify` (a 1-token block, `commit(1, 1)`), so the turn
    /// still finishes on budget.
    ///
    /// The reservation is sized from the backend's `SpeculativePlan`
    /// (`lookahead_rows_for(DraftBlockSize(block_size))` = `block_size + 1`),
    /// never a local `depth + 1` (I1).
    ///
    /// Mutation this catches: ignoring `reserve_cycle_lookahead`'s `false`
    /// (the cycle then proposes and verifies a full block against an
    /// unreserved region), or erroring the turn instead of degrading.
    #[test]
    fn dspark_turn_lookahead_exhaustion_degrades_to_one_ar_step() {
        // Cycle 0 drafts 2 and both are accepted. Later cycles carry a
        // 1-token script the engine may never reach: the pool is exhausted
        // from cycle 1 on, so those cycles must skip `propose` outright.
        let backend = MockDsparkBackend::greedy(
            16,
            vec![
                CycleScript::greedy(vec![5, 6], vec![5, 6, 7]),
                CycleScript::greedy(vec![8], vec![8, 9]),
                CycleScript::greedy(vec![11], vec![11, 12]),
            ],
        )
        .with_lookahead_exhausted_from(2, 1);
        let mut backend = backend;
        let mut params = greedy_params();
        params.max_new_tokens = 6;
        params.mtp_depth = 2;
        let out = drive_turn(&mut backend, params, 6, 999, 2);

        let reserves: Vec<(usize, bool)> = out
            .ledger
            .iter()
            .filter_map(|call| match call {
                Call::Reserve { rows, granted } => Some((*rows, *granted)),
                _ => None,
            })
            .collect();
        assert!(
            reserves.iter().all(|&(rows, _)| rows == 3),
            "every cycle must reserve anchor + block_size rows from the plan, got {reserves:?}"
        );
        assert_eq!(
            reserves.first().copied(),
            Some((3, true)),
            "the first cycle must be admitted"
        );
        assert!(
            reserves.len() >= 2 && reserves[1..].iter().all(|&(_, granted)| !granted),
            "cycles from the second on must see the pool exhausted, got {reserves:?}"
        );

        let proposes = count(&out.ledger, |c| matches!(c, Call::Propose { .. }));
        assert_eq!(
            proposes, 1,
            "only the admitted cycle may draft a block; an exhausted cycle skips propose"
        );
        let verify_widths: Vec<usize> = out
            .ledger
            .iter()
            .filter_map(|call| match call {
                Call::Verify { ids } => Some(ids.len()),
                _ => None,
            })
            .collect();
        assert_eq!(
            verify_widths.first().copied(),
            Some(3),
            "the admitted cycle verifies anchor + 2 drafts"
        );
        assert!(
            verify_widths[1..].iter().all(|&width| width == 1),
            "every exhausted cycle must verify a single anchor row, got {verify_widths:?}"
        );
        assert!(
            commits(&out.ledger)[1..]
                .iter()
                .all(|&(keep, total)| keep == 1 && total == 1),
            "an exhausted cycle commits exactly its one AR row, got {:?}",
            commits(&out.ledger)
        );
        assert_eq!(
            out.finish_reason, "length",
            "exhaustion must degrade the cycle, never fail the turn"
        );
        assert_eq!(out.generated.len(), 6, "the turn still fills its budget");
    }

    /// Plain decoding over a position script: the end token when one is due,
    /// else the target, until `eos` or `max` tokens.
    fn plain(
        targets: &[i32],
        mut tracker: ReasoningTracker,
        eos: u32,
        max: usize,
    ) -> (Vec<u32>, ReasoningOutcome) {
        let mut out = Vec::new();
        while out.len() < max {
            let token = tracker
                .take_forced_token()
                .unwrap_or(targets[out.len()] as u32);
            tracker.observe_token(token);
            out.push(token);
            if token == eos {
                break;
            }
        }
        (out, tracker.outcome())
    }

    fn content(n: usize) -> i32 {
        1 + (n % 7) as i32
    }

    fn verify_widths(ledger: &[Call]) -> Vec<usize> {
        ledger
            .iter()
            .filter_map(|call| match call {
                Call::Verify { ids } => Some(ids.len()),
                Call::VerifyDevice { len } => Some(*len),
                _ => None,
            })
            .collect()
    }

    /// 510 thinking tokens of a loop, then the loop goes on: the guard hits at
    /// 512 inside the first block and the cut lands 64 tokens on, a row of a
    /// full block.
    #[test]
    fn a_cycle_hit_inside_a_dspark_block_cuts_as_a_row() {
        let mut targets = vec![5; 200];
        targets[75] = 15;
        let mut backend = MockDsparkBackend::positions(targets, Vec::new(), false);
        let mut params = greedy_params();
        params.max_new_tokens = 200;
        let mut tracker = ReasoningTracker::guarded(true, None, &[8, 9], true);
        for _ in 0..510 {
            tracker.observe_token(5);
        }
        let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(42);
        let out = drive_turn_with_tracker(
            &mut backend,
            params,
            5,
            15,
            2,
            &mut rng,
            None,
            None,
            tracker,
        );
        assert!(out.result.is_ok(), "{:?}", out.result.err());
        let mut expected = vec![5; 66];
        expected.extend([8, 9, 5, 5, 5, 5, 5, 5, 5, 15]);
        assert_eq!(out.generated, expected);
        assert_eq!(out.outcome.close, Some(ReasoningClose::Cycle));
        let caps: Vec<usize> = backend
            .ledger_snapshot()
            .iter()
            .filter_map(|call| match call {
                Call::Propose { max_len, .. } => Some(*max_len),
                _ => None,
            })
            .collect();
        assert!(caps.iter().all(|&cap| cap == 2), "{caps:?}");
    }

    #[test]
    fn thinking_budget_is_exact_across_dspark_cycles() {
        let end = [8, 10, 9];
        for budget in [0, 1, 2, 3, 5, 12, 16] {
            for block in [1, 3, 7] {
                for (sampled, device) in [(false, false), (true, false), (true, true)] {
                    let stop = budget as usize + end.len() + 10;
                    let mut targets: Vec<i32> = (0..200).map(content).collect();
                    targets[stop] = 15;
                    let misses = vec![2, 7, budget as usize + 1, budget as usize + 5];
                    let mut backend =
                        MockDsparkBackend::positions(targets.clone(), misses, sampled);
                    if device {
                        backend = backend.with_device_proposals();
                    }
                    let mut params = if sampled {
                        dense_params()
                    } else {
                        greedy_params()
                    };
                    params.max_new_tokens = 200;
                    params.mtp_depth = block;
                    let tracker = || ReasoningTracker::new(true, Some(budget), &end);
                    let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(42);
                    let out = drive_turn_with_tracker(
                        &mut backend,
                        params,
                        targets[0] as u32,
                        15,
                        block,
                        &mut rng,
                        None,
                        None,
                        tracker(),
                    );
                    let label = format!(
                        "budget={budget}, block={block}, sampled={sampled}, device={device}"
                    );
                    assert!(out.result.is_ok(), "{label}: {:?}", out.result.err());
                    let (expected, outcome) = plain(&targets, tracker(), 15, 200);
                    assert_eq!(out.generated, expected, "{label}");
                    assert_eq!(out.outcome, outcome, "{label}");
                    assert_eq!(out.outcome.close, Some(ReasoningClose::Budget), "{label}");
                    assert_eq!(out.outcome.cut_at, Some(budget as u32), "{label}");
                    assert_eq!(out.finish_reason, "stop", "{label}");
                    let ledger = backend.ledger_snapshot();
                    assert!(
                        verify_widths(&ledger).iter().all(|&w| w == block + 1),
                        "{label}: every block keeps its depth, {:?}",
                        verify_widths(&ledger)
                    );
                    assert_eq!(
                        commits(&ledger).iter().map(|(keep, _)| keep).sum::<usize>(),
                        out.generated.len() - 1,
                        "{label}: every token but the last is cached once"
                    );
                }
            }
        }
    }

    /// A device-drawn sampled cycle accepts on the device unless a thinking
    /// cut decides one of its rows; a budget the cycle does not reach (the
    /// server's default 12,000) leaves every cycle on the device.
    #[test]
    fn a_sampled_cycle_accepts_on_the_device_unless_a_cut_decides_a_row() {
        let end = [8u32, 10, 9];
        let turn = |tracker: ReasoningTracker| {
            let mut targets: Vec<i32> = (0..400).map(content).collect();
            targets[390] = 15;
            let misses = vec![2, 7, 19, 40, 41, 77];
            let mut backend =
                MockDsparkBackend::positions(targets.clone(), misses, true).with_device_proposals();
            let mut params = dense_params();
            params.max_new_tokens = 120;
            params.mtp_depth = 7;
            let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(42);
            super::TEST_SAMPLED_ACCEPTS.with(|c| c.set((0, 0)));
            let out = drive_turn_with_tracker(
                &mut backend,
                params,
                targets[0] as u32,
                15,
                7,
                &mut rng,
                None,
                None,
                tracker,
            );
            if let Err(error) = &out.result {
                panic!("{}", error.reason);
            }
            let drafted = backend
                .ledger_snapshot()
                .iter()
                .filter(|call| matches!(call, Call::Propose { .. }))
                .count();
            (super::TEST_SAMPLED_ACCEPTS.with(|c| c.get()), drafted, out)
        };
        for (label, budget, thinking) in [
            ("thinking, budget 12000", Some(12_000), true),
            ("thinking, no budget", None, true),
            ("not thinking", None, false),
        ] {
            let ends: &[u32] = if thinking { &end } else { &[] };
            let ((device, host), drafted, _) = turn(ReasoningTracker::new(thinking, budget, ends));
            assert!(drafted > 0, "{label}: no cycle drafted");
            assert_eq!(
                (device, host),
                (drafted, 0),
                "{label}: (device, host) accepts"
            );
        }
        let ((device, host), drafted, out) = turn(ReasoningTracker::new(true, Some(40), &end));
        assert_eq!(out.outcome.cut_at, Some(40));
        assert!(
            (1..=2).contains(&host),
            "the cut's cycles accept on the host: {host} of {drafted}"
        );
        assert_eq!(
            device + host,
            drafted,
            "every other cycle accepts on the device"
        );
    }

    /// A sampled, unpenalized cycle takes the device rows only when its
    /// stepper asks for them and its parameters are on the kernels' list;
    /// every other cycle keeps the filter chain.
    #[test]
    fn sampled_cycles_take_the_device_rows_only_for_a_stepper_that_asks() {
        if !unsafe { mlx_sys::mlx_metal_is_available() }
            || crate::test_support::apple_gpu_family_absent(
                "sampled_cycles_take_the_device_rows_only_for_a_stepper_that_asks",
            )
        {
            return;
        }
        let turn = |asks: bool, device: bool, params: ChatParams| {
            let mut targets: Vec<i32> = (0..400).map(content).collect();
            targets[390] = 15;
            let misses = vec![2, 7, 19, 40, 41, 77];
            let mut backend = MockDsparkBackend::positions(targets.clone(), misses, true);
            if device {
                backend = backend.with_device_proposals();
            }
            if asks {
                backend = backend.with_sampled_rows();
            }
            let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(42);
            super::TEST_SAMPLED_ACCEPTS.with(|c| c.set((0, 0)));
            super::TEST_SAMPLED_ROWS.with(|c| c.set(0));
            let out = drive_turn_with_tracker(
                &mut backend,
                params,
                targets[0] as u32,
                15,
                7,
                &mut rng,
                None,
                None,
                ReasoningTracker::new(false, None, &[]),
            );
            if let Err(error) = &out.result {
                panic!("{}", error.reason);
            }
            let drafted = backend
                .ledger_snapshot()
                .iter()
                .filter(|call| matches!(call, Call::Propose { .. }))
                .count();
            let (device, host) = super::TEST_SAMPLED_ACCEPTS.with(|c| c.get());
            (
                super::TEST_SAMPLED_ROWS.with(|c| c.get()),
                device + host,
                drafted,
            )
        };
        let app = |top_k: i32| {
            let mut params = dense_params();
            params.sampling_config = Some(SamplingConfig {
                temperature: Some(0.6),
                top_k: Some(top_k),
                top_p: Some(0.95),
                min_p: Some(0.0),
            });
            params.max_new_tokens = 120;
            params.mtp_depth = 7;
            params
        };
        let (rows, chain, drafted) = turn(true, true, app(20));
        assert!(drafted > 0);
        assert!(
            rows >= drafted && chain == 0,
            "a stepper that asks: {rows} device rows cycles, {chain} chain cycles, {drafted} drafted"
        );
        let (rows, chain, drafted) = turn(false, true, app(20));
        assert_eq!(
            (rows, chain >= drafted),
            (0, true),
            "a stepper that does not ask keeps the filter chain ({chain} of {drafted})"
        );
        let (rows, chain, drafted) = turn(true, true, app(0));
        assert_eq!(
            (rows, chain >= drafted),
            (0, true),
            "top_k 0 keeps the filter chain"
        );
        let mut penalized = app(20);
        penalized.repetition_penalty = 1.1;
        penalized.repetition_context_size = 64;
        let (rows, chain, drafted) = turn(true, false, penalized);
        assert_eq!(
            (rows, chain >= drafted),
            (0, true),
            "a penalized turn keeps the filter chain"
        );
        let mut greedy = app(20);
        greedy.sampling_config = greedy_params().sampling_config;
        let (rows, chain, _) = turn(true, true, greedy);
        assert_eq!((rows, chain), (0, 0), "a greedy turn draws nothing");
    }

    /// A thinking cut on cycles the device rows decide lands where the
    /// filter chain puts it: on the app's sampling (T 0.6, top_k 20, top_p
    /// 0.95), for every budget and block, the device rows turn gives the
    /// tokens, close and cut row of the same turn on the filter chain and of
    /// plain decoding, and every cycle of it, the cut's cycles too, took the
    /// device rows. The verify rows are one-hot, so the two routes see the
    /// same draws.
    #[test]
    fn a_thinking_cut_on_the_device_rows_lands_where_the_filter_chain_puts_it() {
        if !unsafe { mlx_sys::mlx_metal_is_available() }
            || crate::test_support::apple_gpu_family_absent(
                "a_thinking_cut_on_the_device_rows_lands_where_the_filter_chain_puts_it",
            )
        {
            return;
        }
        let end = [8u32, 10, 9];
        for budget in [0, 1, 2, 3, 5, 12] {
            for block in [1, 3, 7] {
                let stop = budget as usize + end.len() + 10;
                let mut targets: Vec<i32> = (0..200).map(content).collect();
                targets[stop] = 15;
                let misses = vec![2, 7, budget as usize + 1, budget as usize + 5];
                let tracker = || ReasoningTracker::new(true, Some(budget), &end);
                let turn = |rows: bool| {
                    let mut backend =
                        MockDsparkBackend::positions(targets.clone(), misses.clone(), true)
                            .with_device_proposals();
                    if rows {
                        backend = backend.with_sampled_rows();
                    }
                    let mut params = dense_params();
                    params.sampling_config = Some(SamplingConfig {
                        temperature: Some(0.6),
                        top_k: Some(20),
                        top_p: Some(0.95),
                        min_p: Some(0.0),
                    });
                    params.max_new_tokens = 200;
                    params.mtp_depth = block;
                    let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(42);
                    super::TEST_SAMPLED_ACCEPTS.with(|c| c.set((0, 0)));
                    super::TEST_SAMPLED_ROWS.with(|c| c.set(0));
                    let out = drive_turn_with_tracker(
                        &mut backend,
                        params,
                        targets[0] as u32,
                        15,
                        block,
                        &mut rng,
                        None,
                        None,
                        tracker(),
                    );
                    let verifies = verify_widths(&backend.ledger_snapshot()).len();
                    (
                        out,
                        verifies,
                        super::TEST_SAMPLED_ROWS.with(|c| c.get()),
                        super::TEST_SAMPLED_ACCEPTS.with(|c| c.get()),
                    )
                };
                let label = format!("budget={budget}, block={block}");
                let (device, verifies, rows, accepts) = turn(true);
                let (chain, _, chain_rows, _) = turn(false);
                for (route, out) in [("device rows", &device), ("filter chain", &chain)] {
                    if let Err(error) = &out.result {
                        panic!("{label}, {route}: {}", error.reason);
                    }
                    assert_eq!(
                        out.outcome.close,
                        Some(ReasoningClose::Budget),
                        "{label}, {route}"
                    );
                    assert_eq!(out.outcome.cut_at, Some(budget as u32), "{label}, {route}");
                }
                let (expected, outcome) = plain(&targets, tracker(), 15, 200);
                assert_eq!(device.generated, expected, "{label}: plain decoding");
                assert_eq!(device.outcome, outcome, "{label}: plain decoding");
                assert_eq!(
                    device.generated, chain.generated,
                    "{label}: the filter chain"
                );
                assert_eq!(device.outcome, chain.outcome, "{label}: the filter chain");
                assert!(
                    verifies > 0 && rows >= verifies && accepts == (0, 0),
                    "{label}: every verify takes the device rows: {rows} rows cycles, \
                     {verifies} verifies, (device, host) accepts {accepts:?}"
                );
                assert_eq!(
                    chain_rows, 0,
                    "{label}: the chain turn takes no device rows"
                );
            }
        }
    }

    /// Plain decoding that stops on any of `stops`.
    fn plain_stops(
        targets: &[i32],
        mut tracker: ReasoningTracker,
        stops: &[u32],
        max: usize,
    ) -> (Vec<u32>, ReasoningOutcome) {
        let mut out = Vec::new();
        while out.len() < max {
            let token = tracker
                .take_forced_token()
                .unwrap_or(targets[out.len()] as u32);
            tracker.observe_token(token);
            out.push(token);
            if stops.contains(&token) {
                break;
            }
        }
        (out, tracker.outcome())
    }

    /// One turn over the position script with `end`, `max` tokens and the
    /// extra EOS ids; its transcript, outcome and finish reason.
    fn ending_turn(
        targets: &[i32],
        end: &[u32],
        budget: i32,
        max: i32,
        extra: &[u32],
        block: usize,
        sampled: bool,
    ) -> (Vec<u32>, ReasoningOutcome, String) {
        let misses = vec![2, budget as usize + 1];
        let mut backend = MockDsparkBackend::positions(targets.to_vec(), misses, sampled);
        let mut params = if sampled {
            dense_params()
        } else {
            greedy_params()
        };
        params.max_new_tokens = max;
        params.mtp_depth = block;
        params.extra_eos_ids = extra.to_vec();
        let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(42);
        let out = drive_turn_with_tracker(
            &mut backend,
            params,
            targets[0] as u32,
            15,
            block,
            &mut rng,
            None,
            None,
            ReasoningTracker::new(true, Some(budget), end),
        );
        if let Err(error) = &out.result {
            panic!("{}", error.reason);
        }
        (out.generated, out.outcome, out.finish_reason)
    }

    /// `max_new_tokens` reached inside the end text: the turn ends there,
    /// as plain decoding does.
    #[test]
    fn max_new_tokens_inside_the_end_matches_plain() {
        let end = [8, 10, 11, 12, 9];
        let targets: Vec<i32> = (0..200).map(content).collect();
        for budget in [0, 1, 3, 5] {
            for inside in 1..end.len() {
                for block in [1, 3, 7] {
                    for sampled in [false, true] {
                        let max = budget as usize + inside;
                        let label = format!(
                            "budget={budget}, inside={inside}, block={block}, sampled={sampled}"
                        );
                        let (generated, outcome, finish) =
                            ending_turn(&targets, &end, budget, max as i32, &[], block, sampled);
                        let tracker = ReasoningTracker::new(true, Some(budget), &end);
                        let (expected, plain_outcome) = plain(&targets, tracker, 15, max);
                        assert_eq!(generated, expected, "{label}");
                        assert_eq!(outcome, plain_outcome, "{label}");
                        assert_eq!(finish, "length", "{label}");
                    }
                }
            }
        }
    }

    /// An EOS, or an extra EOS id, inside the end text stops the turn on it,
    /// as plain decoding does.
    #[test]
    fn an_eos_inside_the_end_stops_the_turn() {
        let targets: Vec<i32> = (0..200).map(content).collect();
        for (label, end, extra) in [
            ("eos", [8, 15, 9], vec![]),
            ("extra eos", [8, 14, 9], vec![14]),
        ] {
            for budget in [0, 1, 3, 5] {
                for block in [1, 3, 7] {
                    for sampled in [false, true] {
                        let label =
                            format!("{label}: budget={budget}, block={block}, sampled={sampled}");
                        let (generated, outcome, finish) =
                            ending_turn(&targets, &end, budget, 200, &extra, block, sampled);
                        let mut stops = vec![15];
                        stops.extend(&extra);
                        let tracker = ReasoningTracker::new(true, Some(budget), &end);
                        let (expected, plain_outcome) = plain_stops(&targets, tracker, &stops, 200);
                        assert_eq!(expected.len(), budget as usize + 2, "{label}");
                        assert_eq!(generated, expected, "{label}");
                        assert_eq!(outcome, plain_outcome, "{label}");
                        assert_eq!(finish, "stop", "{label}");
                    }
                }
            }
        }
    }

    #[test]
    fn a_64_token_end_takes_at_most_nine_cycles_at_block_7() {
        let mut end = vec![10u32; 63];
        end.push(9);
        let budget = 3;
        let mut targets: Vec<i32> = (0..200).map(content).collect();
        targets[budget + end.len() + 4] = 15;
        let mut backend = MockDsparkBackend::positions(targets.clone(), Vec::new(), false);
        let mut params = greedy_params();
        params.max_new_tokens = 200;
        params.mtp_depth = 7;
        let tracker = || ReasoningTracker::new(true, Some(budget as i32), &end);
        let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(42);
        let out = drive_turn_with_tracker(
            &mut backend,
            params,
            targets[0] as u32,
            15,
            7,
            &mut rng,
            None,
            None,
            tracker(),
        );
        assert!(out.result.is_ok(), "{:?}", out.result.err());
        let (expected, _) = plain(&targets, tracker(), 15, 200);
        assert_eq!(out.generated, expected);
        let ends = budget..budget + end.len();
        let spans = backend.spans();
        assert!(spans.iter().all(|span| span.drafts == 7), "{spans:?}");
        let cycles = spans
            .iter()
            .filter(|span| (span.start..span.start + span.kept).any(|n| ends.contains(&n)))
            .count();
        assert!(cycles <= 1 + (end.len() - 1).div_ceil(8), "{cycles} cycles");
    }

    #[test]
    fn a_cut_row_decides_the_accept_in_every_path() {
        let vocab = 16;
        let cut = ReasoningTracker::new(true, Some(1), &[8, 9])
            .thinking_cut()
            .expect("row 1");
        assert_eq!(cut.row, 1);
        let rows = |fill: f32| {
            [5, 6, 7, 7]
                .iter()
                .flat_map(|&id| logits_row(vocab, id, fill))
                .collect::<Vec<_>>()
        };
        let drafts = vec![5, 8, 7];
        let mut penalized = greedy_params();
        penalized.repetition_penalty = 1.1;
        penalized.repetition_context_size = 8;
        let mut sparse = dense_params();
        sparse.sampling_config.as_mut().unwrap().top_k = Some(4);
        for (label, params, sampled, sparse_q) in [
            ("greedy_fast", greedy_params(), false, false),
            ("penalized", penalized, false, false),
            ("sampled dense", dense_params(), true, false),
            ("sampled sparse", sparse, true, true),
        ] {
            let fill = if sampled { -1.0e30 } else { 0.0 };
            let logits = MxArray::from_float32(&rows(fill), &[1, 4, vocab]).unwrap();
            let mut proposal = DsparkProposal {
                draft_ids: drafts.clone(),
                device_draft_ids: None,
                device_draft_sparse: None,
                draft_dists: Vec::new(),
                draft_sparse_dists: Vec::new(),
                keep_probabilities: None,
            };
            if sampled && sparse_q {
                let q = drafts
                    .iter()
                    .flat_map(|&id| logits_row(vocab, id, -1.0e30))
                    .collect::<Vec<_>>();
                let q = MxArray::from_float32(&q, &[3, vocab]).unwrap();
                let config = params.sampling_config.unwrap();
                let q = crate::sampling::sparse_distributions_from_logits(&q, &config)
                    .unwrap()
                    .expect("sparse rows");
                proposal.draft_sparse_dists = (0..3).map(|i| q.row_owned(i).unwrap()).collect();
            } else if sampled {
                proposal.draft_dists = drafts
                    .iter()
                    .map(|&id| MxArray::from_float32(&one_hot(vocab, id), &[vocab]).unwrap())
                    .collect();
            }
            let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(9);
            let accepted = accept_dspark_proposal(
                &logits,
                &mut proposal,
                &[1, 2],
                &params,
                &mut rng,
                Some(&cut),
            )
            .unwrap_or_else(|e| panic!("{label}: {}", e.reason));
            assert_eq!(accepted, (2, 9), "{label}");
        }
    }

    #[test]
    fn dspark_turn_multi_cycle_greedy_emission_and_commit_ledger() {
        // vocab 16, eos 15 (never produced). Three cycles:
        //   cycle0 depth 2: drafts [4,5], verify [4,5,6] → full accept k=2,
        //          boundary 6, emits [4,5,6], commit(keep 3, total 3).
        //   cycle1 depth 2: drafts [7,8], verify [7,9,0] → k=1 (9 != 8),
        //          residual boundary 9, emits [7,9], commit(keep 2, total 3).
        //   cycle2 near-tail: remaining 2 → l_cap 1 → propose(max_len 1),
        //          drafts [10], verify [10,11] → k=1, boundary 11, emits
        //          [10,11], commit(keep 2, total 2).
        // Budget 8: seed [3] + 3+2+2 cycle tokens = 8 → clean loop-top
        // length exit.
        let mut backend = MockDsparkBackend::greedy(
            16,
            vec![
                CycleScript::greedy(vec![4, 5], vec![4, 5, 6]),
                CycleScript::greedy(vec![7, 8], vec![7, 9, 0]),
                CycleScript::greedy(vec![10], vec![10, 11]),
            ],
        );
        let mut p = greedy_params();
        p.max_new_tokens = 8;
        p.mtp_depth = 2;
        let out = drive_turn(&mut backend, p, 3, 15, 2);

        assert_eq!(
            out.generated,
            vec![3, 4, 5, 6, 7, 9, 10, 11],
            "seed + per-cycle accepted prefix + boundary, in cycle order"
        );
        assert_eq!(out.profiled_generated_tokens, out.generated.len() as u64);
        assert_eq!(
            out.profiled_decode_tokens,
            out.generated.len().saturating_sub(1) as u64,
            "variable DSpark cycle commits must obey the shared profiler invariant"
        );
        assert_eq!(out.finish_reason, "length");
        assert!(
            !out.last_in_cache,
            "the last emitted token is cycle2's boundary — no K/V slot"
        );
        assert_eq!(
            backend.begin_calls.get(),
            1,
            "exactly one begin_dspark_decode"
        );
        assert_eq!(backend.seen_block_size.get(), 2, "setup carries block_size");

        assert_eq!(
            out.ledger,
            vec![
                Call::Propose {
                    anchor: 3,
                    max_len: 2
                },
                Call::Verify { ids: vec![3, 4, 5] },
                Call::Commit { keep: 3, total: 3 },
                Call::EvalBoundary { token: 6 },
                Call::Propose {
                    anchor: 6,
                    max_len: 2
                },
                Call::Verify { ids: vec![6, 7, 8] },
                Call::Commit { keep: 2, total: 3 },
                Call::EvalBoundary { token: 9 },
                Call::Propose {
                    anchor: 9,
                    max_len: 1
                },
                Call::Verify { ids: vec![9, 10] },
                Call::Commit { keep: 2, total: 2 },
                Call::EvalBoundary { token: 11 },
            ],
            "per cycle: propose → verify([anchor, drafts..]) → \
             commit(keep = 1 + k, total = 1 + L) → eval_boundary(boundary)"
        );

        // Stats seam: cycles (L, k) = (2,2), (2,1), (1,1) → mean 4/3,
        // per-position rates [3/3, 1/2], 3 cycles.
        let (mean, per_pos, cycles) = out.acceptance.expect("3 cycles recorded");
        assert!((mean - 4.0 / 3.0).abs() < 1e-9);
        assert_eq!(per_pos, vec![1.0, 0.5]);
        assert_eq!(cycles, 3);
    }

    /// The same script driven over `device_draft_ids` proposals: the engine
    /// must concat anchor + device path into `verify_device` (never `verify`),
    /// materialize host ids only inside acceptance, and emit the exact same
    /// token stream as the host-proposal run.
    #[test]
    fn dspark_turn_device_proposals_dispatch_verify_device() {
        let script = vec![
            CycleScript::greedy(vec![4, 5], vec![4, 5, 6]),
            CycleScript::greedy(vec![7, 8], vec![7, 9, 0]),
            CycleScript::greedy(vec![10], vec![10, 11]),
        ];
        let mut device_backend =
            MockDsparkBackend::greedy(16, script.clone()).with_device_proposals();
        let mut host_backend = MockDsparkBackend::greedy(16, script);
        let mut p = greedy_params();
        p.max_new_tokens = 8;
        p.mtp_depth = 2;
        let device_out = drive_turn(&mut device_backend, p.clone(), 3, 15, 2);
        let host_out = drive_turn(&mut host_backend, p, 3, 15, 2);

        assert_eq!(device_out.generated, host_out.generated);
        assert_eq!(
            device_out.ledger,
            vec![
                Call::Propose {
                    anchor: 3,
                    max_len: 2
                },
                Call::VerifyDevice { len: 3 },
                Call::Commit { keep: 3, total: 3 },
                Call::EvalBoundary { token: 6 },
                Call::Propose {
                    anchor: 6,
                    max_len: 2
                },
                Call::VerifyDevice { len: 3 },
                Call::Commit { keep: 2, total: 3 },
                Call::EvalBoundary { token: 9 },
                Call::Propose {
                    anchor: 9,
                    max_len: 1
                },
                Call::VerifyDevice { len: 2 },
                Call::Commit { keep: 2, total: 2 },
                Call::EvalBoundary { token: 11 },
            ],
            "device proposals: propose → verify_device(1+L) → commit → boundary"
        );
        assert_eq!(
            count(&device_out.ledger, |c| matches!(c, Call::Verify { .. })),
            0,
            "device proposals must never reach the host verify path"
        );
    }

    // ---- 2. keep math (stop-clamp BEFORE commit) --------------------------

    #[test]
    fn dspark_turn_eos_at_accepted_draft_keep_math() {
        // depth 3: drafts [4,15,6] all accepted (verify [4,15,6,7]) + bonus
        // 7 → accepted = [4,15,6,7]. EOS (15) is accepted draft j=1: the
        // clamp cuts INCLUSIVE at it → emit j+1 = 2 tokens. AR-PARITY: the
        // EOS itself is NEVER committed (AR never forwards its final token
        // on a stop and the save drops it from history) → keep = 1 + j = 2
        // (anchor + draft 4 only), total = 4, reason "stop", and
        // last_in_cache FALSE so the save drops the EOS from history too.
        let mut backend = MockDsparkBackend::greedy(
            16,
            vec![CycleScript::greedy(vec![4, 15, 6], vec![4, 15, 6, 7])],
        );
        let mut p = greedy_params();
        p.mtp_depth = 3;
        let out = drive_turn(&mut backend, p, 3, 15, 3);

        assert_eq!(out.generated, vec![3, 4, 15], "emit stops at the EOS draft");
        assert_eq!(out.finish_reason, "stop");
        assert!(
            !out.last_in_cache,
            "EOS at accepted draft j: its slot is excluded from keep → \
             last_in_cache = false (save drops it, matching AR)"
        );
        assert_eq!(
            commits(&out.ledger),
            vec![(2, 4)],
            "keep = 1 + j = 2 (EOS slot excluded), total = 1+L = 4"
        );
        // The clamp resolves the stop BEFORE commit — no second cycle runs.
        assert_eq!(count(&out.ledger, |c| matches!(c, Call::Verify { .. })), 1);
        assert_eq!(
            count(&out.ledger, |c| matches!(c, Call::EvalBoundary { .. })),
            0,
            "stop cycles do not schedule a next-anchor eval"
        );
        // Acceptance is recorded from the ACCEPT result (k=3), not the
        // clamped emission.
        let (mean, _, cycles) = out.acceptance.expect("one cycle recorded");
        assert!((mean - 3.0).abs() < 1e-9);
        assert_eq!(cycles, 1);
    }

    #[test]
    fn dspark_turn_eos_at_first_accepted_draft_keeps_anchor_only() {
        // AR-parity edge: EOS is the FIRST accepted draft (j=0). emit = 1,
        // its slot is excluded → keep = 1 (anchor only, kept_drafts = 0),
        // total = 1+L = 4, last_in_cache FALSE. Post-turn state is
        // identical to an AR turn that generated [anchor, EOS] and dropped
        // the EOS: only the anchor's K/V (and history entry) survive.
        let mut backend = MockDsparkBackend::greedy(
            16,
            vec![CycleScript::greedy(vec![15, 5, 6], vec![15, 5, 6, 7])],
        );
        let mut p = greedy_params();
        p.mtp_depth = 3;
        let out = drive_turn(&mut backend, p, 3, 15, 3);

        assert_eq!(out.generated, vec![3, 15], "emit stops at the EOS draft");
        assert_eq!(out.finish_reason, "stop");
        assert!(!out.last_in_cache);
        assert_eq!(
            commits(&out.ledger),
            vec![(1, 4)],
            "keep = 1 (anchor only): the EOS slot and all later verify rows roll back"
        );
        assert_eq!(
            count(&out.ledger, |c| matches!(c, Call::EvalBoundary { .. })),
            0
        );
    }

    #[test]
    fn dspark_turn_eos_at_boundary_keep_math() {
        // depth 2: drafts [4,5] accepted, boundary (bonus) IS the EOS 15 →
        // accepted = [4,5,15], emit all 3, keep = 1+k = 3, total = 3,
        // reason "stop". The boundary has NO K/V slot → last_in_cache FALSE.
        let mut backend =
            MockDsparkBackend::greedy(16, vec![CycleScript::greedy(vec![4, 5], vec![4, 5, 15])]);
        let p = greedy_params();
        let out = drive_turn(&mut backend, p, 3, 15, 2);

        assert_eq!(out.generated, vec![3, 4, 5, 15]);
        assert_eq!(out.finish_reason, "stop");
        assert!(
            !out.last_in_cache,
            "EOS at the boundary: no K/V slot → last_in_cache = false"
        );
        assert_eq!(commits(&out.ledger), vec![(3, 3)], "keep = 1 + k = 3");
    }

    #[test]
    fn dspark_turn_budget_exact_fit_no_mid_block_cut() {
        // With the over-return contract enforced, a compliant cycle emits at
        // most `l_cap + 1 <= remaining` tokens, so a mid-block BUDGET cut is
        // arithmetically unreachable (the clamp's budget arm is defensive).
        // The tightest compliant case lands EXACTLY on the budget: the cycle
        // completes uncut (full keep, eval_boundary fired) and the run exits
        // "length" at the next loop top.
        let mut backend =
            MockDsparkBackend::greedy(16, vec![CycleScript::greedy(vec![4, 5], vec![4, 5, 6])]);
        let mut p = greedy_params();
        p.max_new_tokens = 4; // seed + l_cap(2) drafts + boundary = exactly 4
        let out = drive_turn(&mut backend, p, 3, 15, 2);

        assert_eq!(
            out.generated,
            vec![3, 4, 5, 6],
            "exact fit: every cycle token emitted, no clamp cut"
        );
        assert_eq!(out.finish_reason, "length");
        assert!(
            !out.last_in_cache,
            "clean length exit lands on the unverified boundary token"
        );
        assert_eq!(
            commits(&out.ledger),
            vec![(3, 3)],
            "full keep — no clamp cut"
        );
        assert_eq!(
            count(&out.ledger, |c| matches!(c, Call::EvalBoundary { .. })),
            1,
            "the cycle completed (no mid-block stop) before the loop-top length exit"
        );
    }

    #[test]
    fn dspark_turn_overlong_proposal_is_rejected() {
        // propose returns 3 drafts for a cap of 2 — the engine must hard-
        // error at the proposal boundary, BEFORE verify writes any
        // target-cache slot (the stop-clamp runs after verify and could not
        // protect the near-tail slot budget).
        let mut backend = MockDsparkBackend::greedy(
            16,
            vec![CycleScript::greedy(vec![4, 5, 6], vec![4, 5, 6, 7])],
        );
        let mut p = greedy_params();
        p.max_new_tokens = 8;
        let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(0xD5_9A2B_C0DE);
        let raw = drive_turn_raw(&mut backend, p, 3, 15, 2, &mut rng);

        let err = match raw.result {
            Ok(_) => panic!("over-long proposal must be rejected"),
            Err(e) => e,
        };
        assert!(
            err.reason.contains("over-returned"),
            "error names the contract violation, got: {}",
            err.reason
        );
        assert_eq!(
            backend.ledger_snapshot(),
            vec![Call::Propose {
                anchor: 3,
                max_len: 2
            }],
            "hard error fires before verify — no target-cache slots written"
        );
        assert_eq!(raw.generated, vec![3], "only the seed was committed");
    }

    fn serial_penalized_accept(
        logits: &MxArray,
        drafts: &[i32],
        history: &[u32],
        p: &ChatParams,
    ) -> Result<(usize, u32)> {
        let mut history = history.to_vec();
        let vocab = logits.shape_at(2)?;
        for i in 0..=drafts.len() {
            let row = logits
                .slice(&[0, i as i64, 0], &[1, i as i64 + 1, vocab])?
                .squeeze(Some(&[0, 1]))?;
            let decision = apply_all_penalties(row, &history, p)?.argmax(0, None)?;
            decision.eval();
            let token = decision.item_at_int32(0)?;
            if drafts.get(i) != Some(&token) {
                return Ok((i, token as u32));
            }
            history.push(token as u32);
        }
        unreachable!()
    }

    #[test]
    fn grouped_penalized_accept_matches_sequential_at_every_rejection_frontier() {
        let history = [1, 2, 1, 3, 4, 1];
        for context in [1, 2, 3, 20] {
            let mut p = greedy_params();
            p.repetition_penalty = 1.4;
            p.presence_penalty = 0.3;
            p.frequency_penalty = 0.2;
            p.repetition_context_size = context;
            p.presence_context_size = context;
            p.frequency_context_size = context;
            for len in 0..=5 {
                let logits = MxArray::from_float32(
                    &(0..(len + 1) * 16)
                        .map(|i| ((i * 13) % 23) as f32 / 7.0 - 1.0)
                        .collect::<Vec<_>>(),
                    &[1, len as i64 + 1, 16],
                )
                .unwrap();
                let mut drafts = Vec::new();
                for _ in 0..len {
                    let (_, next) =
                        serial_penalized_accept(&logits, &drafts, &history, &p).unwrap();
                    drafts.push(next as i32);
                }
                for rejection in 0..=len {
                    let mut proposal = drafts.clone();
                    if rejection < len {
                        proposal[rejection] = (proposal[rejection] + 1) % 16;
                    }
                    let expected =
                        serial_penalized_accept(&logits, &proposal, &history, &p).unwrap();
                    assert_eq!(expected.0, rejection);
                    assert_eq!(
                        penalized_greedy_accept(&logits, &proposal, &history, &p, None).unwrap(),
                        expected,
                        "context={context}, len={len}, rejection={rejection}"
                    );
                }
            }
        }
    }

    #[test]
    #[ignore = "manual control-flow microbenchmark; run in release with --nocapture"]
    fn benchmark_penalized_accept_completion() {
        use std::time::Instant;
        const VOCAB: i64 = 32_000;
        let mut p = greedy_params();
        p.repetition_penalty = 1.2;
        p.repetition_context_size = 256;
        let history: Vec<u32> = (0..256).collect();
        for len in [1, 3, 7] {
            let logits = MxArray::from_float32(
                &(0..(len + 1) * VOCAB as usize)
                    .map(|i| ((i * 13) % 193) as f32 / 71.0)
                    .collect::<Vec<_>>(),
                &[1, len as i64 + 1, VOCAB],
            )
            .unwrap();
            logits.eval();
            let mut drafts = Vec::new();
            for _ in 0..len {
                drafts.push(
                    serial_penalized_accept(&logits, &drafts, &history, &p)
                        .unwrap()
                        .1 as i32,
                );
            }
            for reject_first in [false, true] {
                let mut proposal = drafts.clone();
                if reject_first {
                    proposal[0] = (proposal[0] + 1) % VOCAB as i32;
                }
                let mut times = [Vec::new(), Vec::new()];
                for round in 0..8 {
                    for mode in [round % 2, 1 - round % 2] {
                        let start = Instant::now();
                        for _ in 0..100 {
                            let out = if mode == 0 {
                                serial_penalized_accept(&logits, &proposal, &history, &p)
                            } else {
                                penalized_greedy_accept(&logits, &proposal, &history, &p, None)
                            };
                            std::hint::black_box(out.unwrap());
                        }
                        if round > 0 {
                            times[mode].push(start.elapsed().as_secs_f64() * 10_000.0);
                        }
                    }
                }
                for t in &mut times {
                    t.sort_by(f64::total_cmp);
                }
                println!(
                    "ACCEPT_BENCH drafts={len} reject_first={reject_first} serial_us={:.2} grouped_us={:.2} speedup={:.3}",
                    times[0][3],
                    times[1][3],
                    times[0][3] / times[1][3]
                );
            }
        }
    }

    #[test]
    fn dspark_turn_greedy_with_penalties_penalized_argmax_no_dists_no_rng() {
        // T=0 + repetition_penalty 2.0: valid params must run the
        // penalized-greedy arm — EMPTY draft_dists are accepted (the trait
        // ships no dists at greedy temperature), the accept decision follows
        // the PENALIZED argmax, and zero RNG draws are consumed.
        //
        // Row 0 (anchor 3, history [3]): raw argmax is 3 (10.0), but 3 is in
        // history → 10/2 = 5 < 9.0 at token 4 → penalized argmax 4 == draft
        // 4 → ACCEPT (a raw-argmax accept would reject here — this pins the
        // penalty actually driving the decision).
        // Row 1 (boundary, extended history [3, 4]): raw argmax 4 (10.0)
        // penalized to 5 < 8.0 at token 6 → boundary 6.
        let mut row0 = vec![0.0f32; 16];
        row0[3] = 10.0;
        row0[4] = 9.0;
        let mut row1 = vec![0.0f32; 16];
        row1[4] = 10.0;
        row1[6] = 8.0;
        let mut backend =
            MockDsparkBackend::greedy(16, vec![CycleScript::with_rows(vec![4], vec![row0, row1])]);
        let mut p = greedy_params();
        p.max_new_tokens = 3; // seed + 1 draft + boundary; l_cap = 1
        p.repetition_penalty = 2.0;
        p.repetition_context_size = 20;

        let draws = Rc::new(Cell::new(0usize));
        let mut rng = CountingRng {
            inner: <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(0xD5_9A2B_C0DE),
            draws: Rc::clone(&draws),
        };
        let raw = drive_turn_raw(&mut backend, p, 3, 15, 2, &mut rng);
        let outcome = raw
            .result
            .unwrap_or_else(|e| panic!("penalized-greedy turn failed: {}", e.reason));

        assert_eq!(
            raw.generated,
            vec![3, 4, 6],
            "acceptance and boundary follow the PENALIZED argmax"
        );
        assert_eq!(raw.finish_reason, "length");
        assert!(!outcome.last_in_cache, "run ends on the boundary token");
        let ledger = backend.ledger_snapshot();
        assert_eq!(
            ledger.first(),
            Some(&Call::Propose {
                anchor: 3,
                max_len: 1
            }),
            "near-tail cap still applies on the penalized-greedy arm"
        );
        assert_eq!(commits(&ledger), vec![(2, 2)], "keep = 1 + k = 2");
        assert_eq!(
            raw.acceptance.map(|(mean, _, cycles)| (mean, cycles)),
            Some((1.0, 1)),
            "record_mtp_cycle(1, 1) for the penalized-greedy cycle"
        );
        assert_eq!(
            draws.get(),
            0,
            "greedy temperature must not consume RNG, even with active penalties"
        );
    }

    #[test]
    fn dspark_turn_repetition_clamp_mid_block() {
        // max_consecutive_tokens = 3: the third consecutive 9 trips the
        // cutoff mid-block. accepted = [9,9,9,9] (drafts [9,9,9] all
        // "accepted", bonus 9); the clamp cuts INCLUSIVE at the third 9 →
        // emit 3, reason "repetition". AR-parity: the tripping token is an
        // in-cycle stop token — its slot is excluded → keep = 1 + 2 = 3,
        // total 4, last_in_cache FALSE.
        let mut backend = MockDsparkBackend::greedy(
            16,
            vec![CycleScript::greedy(vec![9, 9, 9], vec![9, 9, 9, 9])],
        );
        let mut p = greedy_params();
        p.mtp_depth = 3;
        p.max_consecutive_tokens = 3;
        let out = drive_turn(&mut backend, p, 3, 15, 3);

        assert_eq!(out.generated, vec![3, 9, 9, 9]);
        assert_eq!(out.finish_reason, "repetition");
        assert!(
            !out.last_in_cache,
            "repetition tripped on an accepted draft — the tripping token's \
             slot is excluded (AR-parity) so the save drops it"
        );
        assert_eq!(commits(&out.ledger), vec![(3, 4)]);
    }

    // ---- 3. L_cap math -----------------------------------------------------

    #[test]
    fn dspark_turn_remaining_one_degenerate_ar_cycle() {
        // Budget 2: after the seed, remaining == 1 → l_cap == 0 → NO
        // propose; verify runs with the 1-token [anchor] slice, boundary =
        // argmax(row 0), commit(keep 1, total 1), and NO record_mtp_cycle.
        let mut backend = MockDsparkBackend::greedy(16, vec![CycleScript::greedy(vec![], vec![8])]);
        let mut p = greedy_params();
        p.max_new_tokens = 2;
        let out = drive_turn(&mut backend, p, 3, 15, 2);

        assert_eq!(
            out.generated,
            vec![3, 8],
            "seed + single AR-through-verify token"
        );
        assert_eq!(out.finish_reason, "length");
        assert_eq!(
            out.ledger,
            vec![
                Call::Verify { ids: vec![3] },
                Call::Commit { keep: 1, total: 1 },
                Call::EvalBoundary { token: 8 },
            ],
            "degenerate cycle: no propose, verify([anchor]) only"
        );
        assert!(
            out.acceptance.is_none(),
            "L == 0 cycles are plain AR steps — record_mtp_cycle must NOT run"
        );
        assert!(
            !out.last_in_cache,
            "the degenerate cycle's emitted token is its boundary — no K/V"
        );
    }

    #[test]
    fn dspark_turn_sampled_remaining_one_degenerate_ar_cycle() {
        // Sampled decoding uses the same zero-draft AR-through-verify cycle.
        // With no drafted token both proposal distribution vectors are empty;
        // that is valid and must sample row 0 rather than failing the
        // dense-vs-sparse exclusivity check.
        let mut backend =
            MockDsparkBackend::sampled(16, vec![CycleScript::greedy(Vec::new(), vec![8])]);
        let mut p = dense_params();
        p.max_new_tokens = 2;
        let out = drive_turn(&mut backend, p, 3, 15, 2);

        assert_eq!(out.generated, vec![3, 8]);
        assert_eq!(out.finish_reason, "length");
        assert_eq!(
            out.ledger,
            vec![
                Call::Verify { ids: vec![3] },
                Call::Commit { keep: 1, total: 1 },
                Call::EvalBoundary { token: 8 },
            ]
        );
        assert!(out.acceptance.is_none());
        assert!(!out.last_in_cache);
    }

    #[test]
    fn dspark_turn_depth_and_block_size_cap_propose_max_len() {
        // mtp_depth 1 < block_size 4 → propose asked for max_len 1.
        let mut backend =
            MockDsparkBackend::greedy(16, vec![CycleScript::greedy(vec![4], vec![4, 5])]);
        let mut p = greedy_params();
        p.max_new_tokens = 3;
        p.mtp_depth = 1;
        let out = drive_turn(&mut backend, p, 3, 15, 4);
        assert_eq!(
            out.ledger.first(),
            Some(&Call::Propose {
                anchor: 3,
                max_len: 1
            }),
            "mtp_depth < block_size caps the proposal request"
        );

        // block_size 1 < mtp_depth 4 → also max_len 1.
        let mut backend =
            MockDsparkBackend::greedy(16, vec![CycleScript::greedy(vec![4], vec![4, 5])]);
        let mut p = greedy_params();
        p.max_new_tokens = 3;
        p.mtp_depth = 4;
        let out = drive_turn(&mut backend, p, 3, 15, 1);
        assert_eq!(
            out.ledger.first(),
            Some(&Call::Propose {
                anchor: 3,
                max_len: 1
            }),
            "block_size < mtp_depth caps the proposal request"
        );
    }

    // ---- 6. streaming -------------------------------------------------------

    /// Recording sink — collects every chunk the loop emits.
    struct RecSink {
        chunks: Mutex<Vec<ChatStreamChunk>>,
    }

    impl ChunkSink for RecSink {
        fn send(&self, chunk: Result<ChatStreamChunk>) {
            if let (Ok(c), Ok(mut v)) = (chunk, self.chunks.lock()) {
                v.push(c);
            }
        }
    }

    /// Word-level tokenizer over a tiny fixed vocab (ids 0..=6) so the
    /// DecodeStream produces deterministic per-token text — same fixture as
    /// the `run_decode_loop` streaming tests.
    fn tiny_tokenizer() -> tokenizers::Tokenizer {
        let json = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": null,
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": {
                    "t0": 0,
                    "t1": 1,
                    "end": 2,
                    "c3": 3,
                    "c4": 4,
                    "eos": 5,
                    "<unk>": 6
                },
                "unk_token": "<unk>"
            }
        }"#;
        tokenizers::Tokenizer::from_bytes(json.as_bytes())
            .unwrap_or_else(|e| panic!("tiny tokenizer build failed: {e}"))
    }

    struct StreamOut {
        generated: Vec<u32>,
        finish_reason: String,
        last_in_cache: bool,
        chunks: Vec<String>,
        ledger: Vec<Call>,
        /// Bytes of `decode(generated)` covered by the step-streamed chunks —
        /// the whole-turn residual flush emits exactly
        /// `decode(generated)[streamed_text_len..]` after the loop returns.
        streamed_text_len: usize,
    }

    /// Drive the STREAMING turn over the scripted backend. `pre_cancelled`
    /// sets the cancel flag before the turn starts; the backend's
    /// `cancel_on_verify` can flip it mid-cycle instead.
    fn drive_streaming_turn_with_emitter(
        backend: &mut MockDsparkBackend,
        params: ChatParams,
        first_token: u32,
        eos_id: u32,
        block_size: usize,
        cancelled: Arc<AtomicBool>,
        emitter: &mut dyn crate::engine::backend::StreamEmitter,
        turn_token_observer: Option<Box<dyn TurnTokenObserver>>,
    ) -> StreamOut {
        let mut tracker = ReasoningTracker::new(false, None, &[]);
        let mut profiler = DecodeProfiler::new("dspark_stream_test", "test");
        let mut generated: Vec<u32> = Vec::new();
        let mut token_history: Vec<u32> = Vec::new();
        let mut finish_reason = String::from("length");
        let mut first_token_instant: Option<Instant> = None;
        let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(0xD5_9A2B_C0DE);
        let y = MxArray::from_int32(&[first_token as i32], &[1])
            .unwrap_or_else(|e| panic!("y construction: {}", e.reason));
        let generation_stream = Stream::generation();
        let max_new_tokens = params.max_new_tokens;

        let tokenizer = tiny_tokenizer();
        let mut decode_stream = tokenizer.decode_stream(true);
        let sink = RecSink {
            chunks: Mutex::new(Vec::new()),
        };
        let mut streamed_text_len = 0usize;
        let mut last_is_reasoning = false;
        let outcome = run_dspark_turn(
            backend,
            &mut rng,
            DsparkTurnArgs {
                y,
                block_size,
                params: &params,
                reasoning_tracker: &mut tracker,
                profiler: &mut profiler,
                max_new_tokens,
                eos_id,
                generated_tokens: &mut generated,
                token_history: &mut token_history,
                finish_reason: &mut finish_reason,
                first_token_instant: &mut first_token_instant,
                report_perf: false,
                generation_stream,
                // The streaming drive exercises the StreamingCtx polls in
                // isolation — production (`run_paged_dspark_turn`, gemma4
                // `flat_draft_chat_turn`) also wires the same flag here.
                cancel_flag: None,
                turn_token_observer,
            },
            Some(StreamingCtx {
                callback: &sink,
                cancelled: &cancelled,
                decode_stream: &mut decode_stream,
                tokenizer: &tokenizer,
                streamed_text_len: &mut streamed_text_len,
                last_is_reasoning: &mut last_is_reasoning,
                emitter,
            }),
        )
        .unwrap_or_else(|e| panic!("run_dspark_turn (streaming) failed: {}", e.reason));

        let chunks = sink
            .chunks
            .lock()
            .unwrap_or_else(|e| panic!("sink poisoned: {e}"))
            .iter()
            .map(|c| c.text.clone())
            .collect();

        StreamOut {
            generated,
            finish_reason,
            last_in_cache: outcome.last_in_cache,
            chunks,
            ledger: backend.ledger_snapshot(),
            streamed_text_len,
        }
    }

    fn drive_streaming_turn(
        backend: &mut MockDsparkBackend,
        params: ChatParams,
        first_token: u32,
        eos_id: u32,
        block_size: usize,
        cancelled: Arc<AtomicBool>,
    ) -> StreamOut {
        let mut emitter = DefaultStreamEmitter;
        drive_streaming_turn_with_emitter(
            backend,
            params,
            first_token,
            eos_id,
            block_size,
            cancelled,
            &mut emitter,
            None,
        )
    }

    /// The residual the whole-turn core's post-loop flush would emit:
    /// `decode(generated)[streamed_text_len..]` — the platform-wide
    /// streaming contract (engine/decode.rs cancel-snapshot comment) is
    /// that step chunks + this residual reconstruct EXACTLY
    /// `decode(generated_tokens)`, on cancelled turns included.
    fn residual_of(out: &StreamOut) -> String {
        let full = tiny_tokenizer()
            .decode(&out.generated, true)
            .unwrap_or_else(|e| panic!("decode failed: {e}"));
        assert!(
            out.streamed_text_len <= full.len(),
            "streamed_text_len {} ran past decode(generated) ({} bytes)",
            out.streamed_text_len,
            full.len()
        );
        full[out.streamed_text_len..].to_string()
    }

    #[test]
    fn dspark_turn_streaming_chunks_match_sync_emission() {
        // Same script sync vs streaming: identical committed tokens, and the
        // chunk concatenation equals the full detokenization of the sync run.
        let script = vec![CycleScript::greedy(vec![1, 3], vec![1, 3, 4])];
        let params = || {
            let mut p = greedy_params();
            p.max_new_tokens = 4;
            p
        };

        let mut sync_backend = MockDsparkBackend::greedy(7, script.clone());
        let sync = drive_turn(&mut sync_backend, params(), 0, 5, 2);

        let mut stream_backend = MockDsparkBackend::greedy(7, script);
        let streamed = drive_streaming_turn(
            &mut stream_backend,
            params(),
            0,
            5,
            2,
            Arc::new(AtomicBool::new(false)),
        );

        assert_eq!(streamed.generated, sync.generated);
        assert_eq!(streamed.finish_reason, sync.finish_reason);
        assert_eq!(streamed.generated, vec![0, 1, 3, 4]);

        let concat: String = streamed.chunks.concat();
        let full_text = tiny_tokenizer()
            .decode(&sync.generated, true)
            .unwrap_or_else(|e| panic!("decode failed: {e}"));
        assert_eq!(
            concat, full_text,
            "streamed chunk concatenation must equal the sync emission's text"
        );
        assert_eq!(
            residual_of(&streamed),
            "",
            "an uncancelled turn leaves nothing for the residual flush"
        );
    }

    struct StoppingObserver {
        stop_id: u32,
    }

    impl TurnTokenObserver for StoppingObserver {
        fn observe_token_id(&mut self, token_id: u32) -> bool {
            token_id == self.stop_id
        }
    }

    #[test]
    fn dspark_streaming_observer_stop_clamps_before_commit() {
        let mut backend =
            MockDsparkBackend::greedy(7, vec![CycleScript::greedy(vec![1, 3], vec![1, 3, 4])]);
        let mut emitter = DefaultStreamEmitter;
        let out = drive_streaming_turn_with_emitter(
            &mut backend,
            greedy_params(),
            0,
            5,
            2,
            Arc::new(AtomicBool::new(false)),
            &mut emitter,
            Some(Box::new(StoppingObserver { stop_id: 3 })),
        );

        assert_eq!(out.generated, vec![0, 1, 3]);
        assert_eq!(out.finish_reason, "stop");
        assert!(!out.last_in_cache, "the stopping token is not persisted");
        assert_eq!(out.chunks, vec!["t0", " t1", " c3"]);
        assert_eq!(commits(&out.ledger), vec![(2, 3)]);
        assert_eq!(
            count(&out.ledger, |call| matches!(
                call,
                Call::EvalBoundary { .. }
            )),
            0,
            "no next speculative cycle is scheduled"
        );
    }

    #[test]
    fn dspark_sync_observer_stop_clamps_before_commit() {
        let mut backend =
            MockDsparkBackend::greedy(7, vec![CycleScript::greedy(vec![1, 3], vec![1, 3, 4])]);
        let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(0xD5_9A2B_C0DE);
        let raw = drive_turn_raw_with_cancel(
            &mut backend,
            greedy_params(),
            0,
            5,
            2,
            &mut rng,
            None,
            Some(Box::new(StoppingObserver { stop_id: 3 })),
        );
        let outcome = raw
            .result
            .unwrap_or_else(|e| panic!("run_dspark_turn failed: {}", e.reason));

        assert_eq!(raw.generated, vec![0, 1, 3]);
        assert_eq!(raw.finish_reason, "stop");
        assert!(
            !outcome.last_in_cache,
            "the stopping token is not persisted"
        );
        assert_eq!(commits(&backend.ledger_snapshot()), vec![(2, 3)]);
    }

    #[test]
    fn dspark_turn_streaming_cancel_at_cycle_boundary() {
        // Cancel set BEFORE the turn: the initial seed is committed but not
        // streamed (mtp_turn parity: skip, no break), then the loop-top
        // cancel check exits "cancelled" before any propose/verify.
        let mut backend =
            MockDsparkBackend::greedy(7, vec![CycleScript::greedy(vec![1, 3], vec![1, 3, 4])]);
        let out = drive_streaming_turn(
            &mut backend,
            greedy_params(),
            0,
            5,
            2,
            Arc::new(AtomicBool::new(true)),
        );

        assert_eq!(
            out.generated,
            vec![0],
            "only the pre-sampled seed is committed"
        );
        assert_eq!(out.finish_reason, "cancelled");
        assert!(!out.last_in_cache, "the unverified seed has no K/V");
        assert!(
            out.chunks.is_empty(),
            "cancelled before any chunk was streamed"
        );
        assert_eq!(
            residual_of(&out),
            "t0",
            "the whole-turn residual flush delivers the committed-but-unstreamed \
             seed's text (mtp initial-arm parity: total streamed == decode(generated))"
        );
        assert!(
            out.ledger.is_empty(),
            "cancel at the cycle boundary: no propose/verify/commit ran"
        );
    }

    #[test]
    fn dspark_turn_streaming_cancel_in_clamp_commits_exactly_once() {
        // Cancel flips DURING the first verify (after the loop-top check).
        // The clamp observes it after the first accepted token: emit 1,
        // reason "cancelled". AR-parity: the cancel-observed token is an
        // in-cycle stop token — its slot is excluded → commit(keep = 1,
        // total 3) — commit still happens EXACTLY once and the cancel token
        // is emitted (in `generated`) but NOT streamed and NOT persisted
        // (mtp emit-arm parity + AR drop-last).
        let cancelled = Arc::new(AtomicBool::new(false));
        let mut backend =
            MockDsparkBackend::greedy(7, vec![CycleScript::greedy(vec![1, 3], vec![1, 3, 4])]);
        backend.cancel_on_verify = Some((0, Arc::clone(&cancelled)));
        let out = drive_streaming_turn(&mut backend, greedy_params(), 0, 5, 2, cancelled);

        assert_eq!(
            out.generated,
            vec![0, 1],
            "seed + the one token emitted before the clamp saw the cancel"
        );
        assert_eq!(out.finish_reason, "cancelled");
        assert!(
            !out.last_in_cache,
            "the cancel token's slot is excluded from keep (AR-parity)"
        );
        assert_eq!(
            out.chunks,
            vec![String::from("t0")],
            "the cancel-observed token is committed but not streamed"
        );
        // Residual contract (AR/MTP parity): the whole-turn flush emits
        // decode(generated)[streamed_text_len..]; on a cancel cut that
        // suffix is EXACTLY the single cancel-observed token's text —
        // never more (no other suppressed tokens can hide behind it), so
        // step chunks + residual reconstruct decode(generated), the same
        // total the AR loop's documented origin/main cancel semantics
        // produce (engine/decode.rs cancel-snapshot comment).
        assert_eq!(
            residual_of(&out),
            " t1",
            "the unstreamed suffix must be exactly the cancel-observed token's text"
        );
        assert_eq!(
            commits(&out.ledger),
            vec![(1, 3)],
            "commit-exactly-once with the clamped, stop-excluded keep"
        );
        assert_eq!(
            count(&out.ledger, |c| matches!(c, Call::EvalBoundary { .. })),
            0,
            "cancel stop breaks before scheduling a next anchor"
        );
    }

    #[test]
    fn dspark_turn_sync_preset_cancel_stops_before_any_cycle() {
        // H2: a NON-streaming turn (streaming ctx None) whose cancel flag is
        // already set at loop entry — the ungated pre-cycle poll must exit
        // "cancelled" after the seed, before ANY propose/verify/commit.
        // Catches the mutation that re-gates the pre-cycle poll on
        // `streaming.is_some()` (the pre-fix behavior): under it this turn
        // runs speculative cycles to the length budget.
        let cancel = Arc::new(AtomicBool::new(true));
        let mut backend =
            MockDsparkBackend::greedy(7, vec![CycleScript::greedy(vec![1, 3], vec![1, 3, 4])]);
        let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(0xD5_9A2B_C0DE);
        let raw = drive_turn_raw_with_cancel(
            &mut backend,
            greedy_params(),
            0,
            5,
            2,
            &mut rng,
            Some(cancel.as_ref()),
            None,
        );
        let outcome = raw
            .result
            .unwrap_or_else(|e| panic!("run_dspark_turn failed: {}", e.reason));

        assert_eq!(raw.finish_reason, "cancelled");
        assert_eq!(raw.generated, vec![0], "only the prefill seed commits");
        assert!(!outcome.last_in_cache, "the unverified seed has no K/V");
        assert!(
            backend.ledger_snapshot().is_empty(),
            "cancel at the cycle boundary: no propose/verify/commit ran"
        );
    }

    #[test]
    fn dspark_turn_sync_cancel_in_clamp_commits_exactly_once() {
        // H2 sync twin of `dspark_turn_streaming_cancel_in_clamp_commits_
        // exactly_once`: the flag flips DURING the first verify (after the
        // pre-cycle check), the streaming ctx is None, and the flag rides
        // ONLY `DsparkTurnArgs.cancel_flag`. The ungated stop-clamp poll —
        // the SAME snapshot point the streaming twin polls — must observe it
        // after the first accepted token: emit 1, reason "cancelled",
        // AR-parity slot exclusion → commit(keep = 1, total 3), exactly
        // once. Catches the mutation that re-gates the clamp poll on the
        // streaming ctx: under it the whole cycle emits and the turn ends
        // "length" with a (3, 3) commit.
        let cancel = Arc::new(AtomicBool::new(false));
        let mut backend =
            MockDsparkBackend::greedy(7, vec![CycleScript::greedy(vec![1, 3], vec![1, 3, 4])]);
        backend.cancel_on_verify = Some((0, Arc::clone(&cancel)));
        let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(0xD5_9A2B_C0DE);
        let raw = drive_turn_raw_with_cancel(
            &mut backend,
            greedy_params(),
            0,
            5,
            2,
            &mut rng,
            Some(cancel.as_ref()),
            None,
        );
        let outcome = raw
            .result
            .unwrap_or_else(|e| panic!("run_dspark_turn failed: {}", e.reason));
        let ledger = backend.ledger_snapshot();

        assert_eq!(raw.finish_reason, "cancelled");
        assert_eq!(
            raw.generated,
            vec![0, 1],
            "seed + the one token emitted before the clamp saw the cancel"
        );
        assert!(
            !outcome.last_in_cache,
            "the cancel token's slot is excluded from keep (AR-parity)"
        );
        assert_eq!(
            commits(&ledger),
            vec![(1, 3)],
            "commit-exactly-once with the clamped, stop-excluded keep"
        );
        assert_eq!(
            count(&ledger, |c| matches!(c, Call::EvalBoundary { .. })),
            0,
            "cancel stop breaks before scheduling a next anchor"
        );
    }

    // ---- 7. sampled path ----------------------------------------------------

    #[test]
    fn dspark_turn_sampled_all_accept_and_bonus() {
        // T=1.0 with exact one-hot rows: drafts [4,5] with proposal dists
        // one-hot at 4/5 and verify rows one-hot at [4,5,6] → both accept
        // with ratio exactly 1.0, bonus sample from row 2 is 6
        // (deterministic by construction). keep = 1 + min(3, 2) = 3.
        let mut backend = MockDsparkBackend::sampled(
            16,
            vec![CycleScript {
                draft_ids: vec![4, 5],
                draft_dists: vec![one_hot(16, 4), one_hot(16, 5)],
                verify_argmax: vec![4, 5, 6],
                verify_rows: None,
            }],
        );
        let mut p = dense_params();
        p.max_new_tokens = 4;
        let out = drive_turn(&mut backend, p, 3, 15, 2);

        assert_eq!(
            out.generated,
            vec![3, 4, 5, 6],
            "all-accept emits both drafts + the bonus boundary"
        );
        assert_eq!(out.finish_reason, "length");
        assert_eq!(commits(&out.ledger), vec![(3, 3)]);
        let (mean, _, cycles) = out.acceptance.expect("one sampled cycle recorded");
        assert!((mean - 2.0).abs() < 1e-9, "k == 2 accepted drafts");
        assert_eq!(cycles, 1);
    }

    #[test]
    fn dspark_turn_sampled_first_position_reject_residual_boundary() {
        // T=1.0: draft 4 proposed from a one-hot-at-4 dist, but the verify
        // row 0 puts ALL target mass on 9 → p_t[4] == 0 → deterministic
        // reject; the residual (p - q)+ is one-hot at 9, so the boundary is
        // 9. k = 0, keep = 1 + min(1, 0) = 1, total = 1 + L = 3. The
        // residual doubles as the EOS so the turn ends on this cycle.
        let mut backend = MockDsparkBackend::sampled(
            16,
            vec![CycleScript {
                draft_ids: vec![4, 5],
                draft_dists: vec![one_hot(16, 4), one_hot(16, 5)],
                verify_argmax: vec![9, 0, 0],
                verify_rows: None,
            }],
        );
        let p = dense_params();
        let out = drive_turn(&mut backend, p, 3, 9, 2);

        assert_eq!(
            out.generated,
            vec![3, 9],
            "first-position reject emits only the residual boundary"
        );
        assert_eq!(out.finish_reason, "stop");
        assert!(!out.last_in_cache, "the residual boundary has no K/V");
        assert_eq!(commits(&out.ledger), vec![(1, 3)]);
        let (mean, _, cycles) = out.acceptance.expect("cycle recorded with k = 0");
        assert!(mean.abs() < 1e-9, "zero accepted drafts");
        assert_eq!(cycles, 1);
    }
}

#[cfg(test)]
mod device_budget_tests {
    use super::*;

    #[test]
    fn budget_caps_retention_but_labels_last_valid_row() -> Result<()> {
        // Every proposed id matches, including masked rows whose values must
        // never affect the retention frontier or training labels.
        let mut data = vec![-10.0f32; 8 * 16];
        for row in 0..8 {
            data[row * 16 + row + 1] = 10.0;
        }
        let logits = MxArray::from_float32(&data, &[1, 8, 16])?;
        for rows in 1..=8 {
            let mut proposal = DsparkProposal {
                draft_ids: Vec::new(),
                device_draft_ids: Some(MxArray::from_int32(&[1, 2, 3, 4, 5, 6, 7], &[7])?),
                device_draft_sparse: None,
                draft_dists: Vec::new(),
                draft_sparse_dists: Vec::new(),
                keep_probabilities: None,
            };
            let budget = MxArray::from_int32(&[rows], &[1])?;
            let (accepted, boundary, observed_rows, raw) =
                accept_device_budgeted_greedy(&logits, &mut proposal, &budget)?;
            assert_eq!(accepted, rows as usize - 1);
            assert_eq!(boundary, rows as u32);
            assert_eq!(observed_rows, rows as usize);
            assert_eq!(raw, (rows as usize).min(7));
            assert_eq!(
                proposal.draft_len()?,
                7,
                "physical rollback width must remain eight"
            );
        }
        Ok(())
    }

    #[test]
    fn budgeted_rejection_and_invalid_budgets_fail_before_commit() -> Result<()> {
        let mut data = vec![-10.0f32; 8 * 16];
        for row in 0..8 {
            data[row * 16 + if row == 2 { 11 } else { row + 1 }] = 10.0;
        }
        let logits = MxArray::from_float32(&data, &[1, 8, 16])?;
        let mut proposal = DsparkProposal {
            draft_ids: vec![1, 2, 3, 4, 5, 6, 7],
            device_draft_ids: None,
            device_draft_sparse: None,
            draft_dists: Vec::new(),
            draft_sparse_dists: Vec::new(),
            keep_probabilities: None,
        };
        let budget = MxArray::from_int32(&[4], &[1])?;
        assert_eq!(
            accept_device_budgeted_greedy(&logits, &mut proposal, &budget)?,
            (2, 11, 4, 2)
        );
        for invalid in [-1, 0, 9] {
            assert!(
                accept_device_budgeted_greedy(
                    &logits,
                    &mut proposal,
                    &MxArray::from_int32(&[invalid], &[1])?
                )
                .is_err()
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tree_verify_tests {
    use std::cell::RefCell;
    use std::collections::VecDeque;
    use std::rc::Rc;
    use std::sync::Arc;

    use super::*;
    use crate::engine::backend::{
        ChatBackend, DsparkTreeGreedyResult, DsparkVerifyOutput, FinalizeArgs, ResetScope,
        SaveStateArgs, TurnSetup,
    };
    use crate::engine::penalties::ReasoningTracker;
    use crate::engine::types::ChatResult;
    use crate::sampling::SamplingConfig;
    use crate::tokenizer::Qwen3Tokenizer;

    #[derive(Clone, Debug, PartialEq, Eq)]
    enum TreeCall {
        Observe(Vec<u32>),
        VerifyTree {
            anchor: u32,
            max_nodes: usize,
            max_depth: usize,
        },
        CommitTree {
            nodes: Vec<usize>,
            contiguous: bool,
        },
        LinearPropose {
            anchor: u32,
            max_len: usize,
        },
        LinearVerify(Vec<u32>),
        LinearCommit {
            keep: usize,
            total: usize,
        },
    }

    struct TreeScriptStep {
        emitted_tokens: Vec<u32>,
        accepted_nodes: Vec<usize>,
        draft_count: usize,
        tree_nodes: usize,
    }

    struct TreeMockBackend {
        tree_steps: VecDeque<TreeScriptStep>,
        linear_next_token: u32,
        calls: Rc<RefCell<Vec<TreeCall>>>,
    }

    struct TreeMockStepper<'a> {
        backend: &'a mut TreeMockBackend,
    }

    impl DsparkStepper for TreeMockStepper<'_> {
        fn supports_tree_verify(&self) -> bool {
            true
        }

        fn observe_emitted_tokens(&mut self, tokens: &[u32]) {
            self.backend
                .calls
                .borrow_mut()
                .push(TreeCall::Observe(tokens.to_vec()));
        }

        fn verify_tree_greedy(
            &mut self,
            anchor: u32,
            max_nodes: usize,
            max_depth: usize,
        ) -> Result<DsparkTreeGreedyResult> {
            self.backend.calls.borrow_mut().push(TreeCall::VerifyTree {
                anchor,
                max_nodes,
                max_depth,
            });
            let step = self
                .backend
                .tree_steps
                .pop_front()
                .ok_or_else(|| Error::from_reason("unexpected verify_tree_greedy call"))?;
            let is_contiguous_prefix = step.accepted_nodes.iter().enumerate().all(|(i, &n)| n == i);
            let accepted_draft_count = step.accepted_nodes.len().saturating_sub(1);
            Ok(DsparkTreeGreedyResult {
                emitted_tokens: step.emitted_tokens,
                accepted_nodes: step.accepted_nodes,
                is_contiguous_prefix,
                draft_count: step.draft_count,
                tree_nodes: step.tree_nodes,
                accepted_draft_count,
            })
        }

        fn commit_tree(
            &mut self,
            accepted_nodes: &[usize],
            is_contiguous_prefix: bool,
        ) -> Result<()> {
            self.backend.calls.borrow_mut().push(TreeCall::CommitTree {
                nodes: accepted_nodes.to_vec(),
                contiguous: is_contiguous_prefix,
            });
            Ok(())
        }

        fn propose(
            &mut self,
            anchor_id: u32,
            max_len: usize,
            _params: &ChatParams,
            _rng: &mut dyn rand::Rng,
        ) -> Result<DsparkProposal> {
            self.backend
                .calls
                .borrow_mut()
                .push(TreeCall::LinearPropose {
                    anchor: anchor_id,
                    max_len,
                });
            Ok(DsparkProposal {
                draft_ids: Vec::new(),
                device_draft_ids: None,
                device_draft_sparse: None,
                draft_dists: Vec::new(),
                draft_sparse_dists: Vec::new(),
                keep_probabilities: None,
            })
        }

        fn verify(&mut self, verify_ids: &[u32]) -> Result<DsparkVerifyOutput> {
            self.backend
                .calls
                .borrow_mut()
                .push(TreeCall::LinearVerify(verify_ids.to_vec()));
            let mut row = vec![-10.0f32; 32];
            row[self.backend.linear_next_token as usize] = 10.0;
            let logits = MxArray::from_float32(&row, &[1, 1, 32])?;
            Ok(DsparkVerifyOutput { logits })
        }

        fn commit(&mut self, keep: usize, total_written: usize) -> Result<()> {
            self.backend
                .calls
                .borrow_mut()
                .push(TreeCall::LinearCommit {
                    keep,
                    total: total_written,
                });
            Ok(())
        }

        fn eval_boundary(&self, _token: &MxArray) {}
    }

    struct DummyDecode;
    impl DecodeStep for DummyDecode {
        fn forward(&mut self, _input_ids: &MxArray) -> Result<(MxArray, bool)> {
            unreachable!()
        }
    }

    impl ChatBackend for TreeMockBackend {
        type Decode<'a>
            = DummyDecode
        where
            Self: 'a;

        fn tokenizer(&self) -> Result<Arc<Qwen3Tokenizer>> {
            unreachable!()
        }
        fn family_name(&self) -> &'static str {
            "tree_mock"
        }
        fn session_eos_id(&self, _tok: &Qwen3Tokenizer) -> Result<u32> {
            Ok(31)
        }
        fn cached_token_history(&self) -> &[u32] {
            &[]
        }
        fn reset_caches(&mut self, _scope: ResetScope) -> Result<()> {
            Ok(())
        }
        fn verify_cache_prefix(&self, _tokens: &[u32], _reuse_cache: bool) -> usize {
            0
        }
        fn save_cache_state(&mut self, _args: SaveStateArgs<'_>) {}
        fn eval_caches(&self) -> Result<()> {
            Ok(())
        }
        fn prefill(&mut self, _tokens: &[u32], _stream: Stream) -> Result<MxArray> {
            unreachable!()
        }
        fn begin_decode(&mut self, _setup: &TurnSetup<'_>) -> Result<Self::Decode<'_>> {
            Ok(DummyDecode)
        }
        fn finalize_turn(&self, _args: FinalizeArgs<'_>) -> Result<ChatResult> {
            unreachable!()
        }
    }

    impl DsparkBackend for TreeMockBackend {
        type DsparkDecode<'a>
            = TreeMockStepper<'a>
        where
            Self: 'a;

        fn begin_dspark_decode(&mut self, _block_size: usize) -> Result<Self::DsparkDecode<'_>> {
            Ok(TreeMockStepper { backend: self })
        }
    }

    fn greedy_tree_params(max_new_tokens: i32) -> ChatParams {
        ChatParams {
            max_new_tokens,
            sampling_config: Some(SamplingConfig {
                temperature: Some(0.0),
                top_k: None,
                top_p: None,
                min_p: None,
            }),
            repetition_penalty: 1.0,
            repetition_context_size: 20,
            presence_penalty: 0.0,
            presence_context_size: 20,
            frequency_penalty: 0.0,
            frequency_context_size: 20,
            max_consecutive_tokens: 0,
            max_ngram_repeats: 0,
            ngram_size: 0,
            include_reasoning: true,
            report_performance: false,
            reuse_cache: false,
            enable_mtp: true,
            mtp_depth: 7,
            mtp_adaptive_depth: false,
            extra_eos_ids: Vec::new(),
            cache_salt: 0,
            cache_owner_id: String::new(),
            cache_root_owner_id: None,
        }
    }

    #[test]
    fn tree_verify_non_contiguous_branch_and_linear_tail() {
        let calls = Rc::new(RefCell::new(Vec::new()));
        let mut backend = TreeMockBackend {
            tree_steps: VecDeque::from([TreeScriptStep {
                emitted_tokens: vec![5, 6, 7],
                accepted_nodes: vec![0, 3, 5],
                draft_count: 7,
                tree_nodes: 16,
            }]),
            linear_next_token: 8,
            calls: Rc::clone(&calls),
        };
        let params = greedy_tree_params(5);
        let y = MxArray::from_int32(&[4], &[1]).unwrap();
        let mut tracker = ReasoningTracker::new(false, None, &[]);
        let mut profiler = DecodeProfiler::new("test", "tree_mock");
        let mut generated = Vec::new();
        let mut history = vec![1, 2, 3];
        let mut finish_reason = String::new();
        let mut first_tok = None;
        let mut rng = rand::rng();

        let outcome = run_dspark_turn(
            &mut backend,
            &mut rng,
            DsparkTurnArgs {
                y,
                block_size: 7,
                params: &params,
                reasoning_tracker: &mut tracker,
                profiler: &mut profiler,
                max_new_tokens: 5,
                eos_id: 31,
                generated_tokens: &mut generated,
                token_history: &mut history,
                finish_reason: &mut finish_reason,
                first_token_instant: &mut first_tok,
                report_perf: false,
                generation_stream: Stream::generation(),
                cancel_flag: None,
                turn_token_observer: None,
            },
            None,
        )
        .unwrap();

        assert_eq!(generated, vec![4, 5, 6, 7, 8]);
        assert_eq!(history, vec![1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(finish_reason, "length");
        assert!(!outcome.last_in_cache);
        assert_eq!(
            *calls.borrow(),
            vec![
                TreeCall::Observe(vec![1, 2, 3]),
                TreeCall::Observe(vec![4]),
                TreeCall::VerifyTree {
                    anchor: 4,
                    max_nodes: 16,
                    max_depth: 3,
                },
                TreeCall::CommitTree {
                    nodes: vec![0, 3, 5],
                    contiguous: false,
                },
                TreeCall::Observe(vec![5, 6, 7]),
                // remaining == 1 falls back to single-token linear verify.
                TreeCall::LinearVerify(vec![7]),
                TreeCall::LinearCommit { keep: 1, total: 1 },
                TreeCall::Observe(vec![8]),
            ]
        );
        let (mean, _, cycles) = profiler
            .mtp_acceptance_summary()
            .expect("tree cycle must record MTP stats");
        assert_eq!(cycles, 1);
        assert!((mean - 2.0).abs() < 1e-9);
    }

    #[test]
    fn tree_verify_clamps_eos_and_recomputes_contiguity() {
        // Case A: EOS at index 1 of non-contiguous [0, 2, 4] -> keep = 2 -> [0, 2] (non-contiguous).
        let calls_a = Rc::new(RefCell::new(Vec::new()));
        let mut backend_a = TreeMockBackend {
            tree_steps: VecDeque::from([TreeScriptStep {
                emitted_tokens: vec![10, 31, 12],
                accepted_nodes: vec![0, 2, 4],
                draft_count: 7,
                tree_nodes: 12,
            }]),
            linear_next_token: 0,
            calls: Rc::clone(&calls_a),
        };
        let params = greedy_tree_params(10);
        let mut tracker = ReasoningTracker::new(false, None, &[]);
        let mut profiler = DecodeProfiler::new("test", "tree_mock");
        let mut generated = Vec::new();
        let mut history = vec![1, 2];
        let mut finish_reason = String::new();
        let mut first_tok = None;
        let mut rng = rand::rng();

        let out_a = run_dspark_turn(
            &mut backend_a,
            &mut rng,
            DsparkTurnArgs {
                y: MxArray::from_int32(&[4], &[1]).unwrap(),
                block_size: 7,
                params: &params,
                reasoning_tracker: &mut tracker,
                profiler: &mut profiler,
                max_new_tokens: 10,
                eos_id: 31,
                generated_tokens: &mut generated,
                token_history: &mut history,
                finish_reason: &mut finish_reason,
                first_token_instant: &mut first_tok,
                report_perf: false,
                generation_stream: Stream::generation(),
                cancel_flag: None,
                turn_token_observer: None,
            },
            None,
        )
        .unwrap();

        assert_eq!(generated, vec![4, 10, 31]);
        assert_eq!(finish_reason, "stop");
        assert!(!out_a.last_in_cache);
        assert!(calls_a.borrow().contains(&TreeCall::CommitTree {
            nodes: vec![0, 2],
            contiguous: false,
        }));

        // Case B: EOS at index 1 of [0, 1, 5] -> keep = 2 -> [0, 1] (becomes contiguous!).
        let calls_b = Rc::new(RefCell::new(Vec::new()));
        let mut backend_b = TreeMockBackend {
            tree_steps: VecDeque::from([TreeScriptStep {
                emitted_tokens: vec![10, 31, 12],
                accepted_nodes: vec![0, 1, 5],
                draft_count: 7,
                tree_nodes: 12,
            }]),
            linear_next_token: 0,
            calls: Rc::clone(&calls_b),
        };
        let mut tracker = ReasoningTracker::new(false, None, &[]);
        let mut profiler = DecodeProfiler::new("test", "tree_mock");
        let mut generated = Vec::new();
        let mut history = vec![1, 2];
        let mut finish_reason = String::new();
        let mut first_tok = None;

        let out_b = run_dspark_turn(
            &mut backend_b,
            &mut rng,
            DsparkTurnArgs {
                y: MxArray::from_int32(&[4], &[1]).unwrap(),
                block_size: 7,
                params: &params,
                reasoning_tracker: &mut tracker,
                profiler: &mut profiler,
                max_new_tokens: 10,
                eos_id: 31,
                generated_tokens: &mut generated,
                token_history: &mut history,
                finish_reason: &mut finish_reason,
                first_token_instant: &mut first_tok,
                report_perf: false,
                generation_stream: Stream::generation(),
                cancel_flag: None,
                turn_token_observer: None,
            },
            None,
        )
        .unwrap();

        assert_eq!(generated, vec![4, 10, 31]);
        assert_eq!(finish_reason, "stop");
        assert!(!out_b.last_in_cache);
        assert!(calls_b.borrow().contains(&TreeCall::CommitTree {
            nodes: vec![0, 1],
            contiguous: true,
        }));
    }

    #[test]
    fn tree_verify_falls_back_to_linear_when_penalties_active() {
        let calls = Rc::new(RefCell::new(Vec::new()));
        let mut backend = TreeMockBackend {
            tree_steps: VecDeque::new(),
            linear_next_token: 31,
            calls: Rc::clone(&calls),
        };
        let mut params = greedy_tree_params(5);
        params.repetition_penalty = 1.1;
        let mut tracker = ReasoningTracker::new(false, None, &[]);
        let mut profiler = DecodeProfiler::new("test", "tree_mock");
        let mut generated = Vec::new();
        let mut history = vec![1, 2, 3];
        let mut finish_reason = String::new();
        let mut first_tok = None;
        let mut rng = rand::rng();

        let _ = run_dspark_turn(
            &mut backend,
            &mut rng,
            DsparkTurnArgs {
                y: MxArray::from_int32(&[4], &[1]).unwrap(),
                block_size: 7,
                params: &params,
                reasoning_tracker: &mut tracker,
                profiler: &mut profiler,
                max_new_tokens: 5,
                eos_id: 31,
                generated_tokens: &mut generated,
                token_history: &mut history,
                finish_reason: &mut finish_reason,
                first_token_instant: &mut first_tok,
                report_perf: false,
                generation_stream: Stream::generation(),
                cancel_flag: None,
                turn_token_observer: None,
            },
            None,
        )
        .unwrap();

        assert_eq!(generated, vec![4, 31]);
        assert_eq!(finish_reason, "stop");
        assert!(
            calls
                .borrow()
                .iter()
                .all(|c| !matches!(c, TreeCall::VerifyTree { .. })),
            "penalized turns must not invoke verify_tree_greedy"
        );
        assert!(
            calls
                .borrow()
                .iter()
                .any(|c| matches!(c, TreeCall::LinearPropose { .. })),
            "penalized turns must route through linear propose"
        );
    }
}
