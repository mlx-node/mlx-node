//! Engine-owned DSpark (draft-model speculative decode) whole-turn loop —
//! the DSpark analog of [`crate::engine::mtp_turn`]. Families opt in via
//! [`crate::engine::backend::DsparkBackend`]; their `ChatBackend::mtp_turn`
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

use std::sync::atomic::Ordering;
use std::time::Instant;

use napi::bindgen_prelude::*;

use crate::array::{DType, MxArray};
use crate::decode_profiler::DecodeProfiler;
use crate::engine::backend::{DsparkBackend, DsparkProposal, DsparkStepper, DsparkTurnSetup};
use crate::engine::decode::StreamingCtx;
use crate::engine::params::ChatParams;
use crate::engine::penalties::{ReasoningTracker, apply_all_penalties};
use crate::sampling;
use crate::stream::Stream;

/// Required arguments of [`run_dspark_turn`] — the DSpark analog of
/// [`crate::engine::mtp_turn::MtpTurnArgs`], minus the MTP-only
/// prompt-hidden seed fields (the DSpark stepper owns its draft-context
/// seeding) and plus the draft `block_size`. Constructed in production by
/// gemma4's `dspark_chat_turn`.
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
enum CycleStop {
    Length,
    Stop,
    Cancelled,
    Repetition(&'static str),
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
/// Driven in production by gemma4's `mtp_turn` override
/// (`dspark_chat_turn`); the module's mock tests pin the loop contract.
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
    } = args;

    let setup = DsparkTurnSetup {
        params: p,
        block_size,
    };
    let mut step = backend.begin_dspark_decode(&setup)?;

    // Materialize the prefill-sampled seed once; it is the first anchor.
    y.eval();
    let mut anchor: u32 = y.item_at_int32(0)? as u32;

    // `last_in_cache` default `true`: a zero-budget exit emits nothing, so
    // the last cached token is the prompt's last token, which IS in cache.
    let mut last_in_cache = true;
    let mut last_clear_at: usize = generated.len();

    // Same nonpositive-budget clamp as `run_mtp_turn` (see its PARITY-FIX
    // comment): a negative `max` must behave as 0, not wrap to a huge usize.
    let max_as_usize: usize = (max).max(0) as usize;

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
        let _is_reasoning = tracker.observe_token(initial_token_id);
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
                s.emitter.on_token_text(
                    &token_text,
                    _is_reasoning,
                    p.include_reasoning,
                    s.callback,
                );
            }
        }
        profiler.step();
        // The just-emitted anchor has no K/V yet — the first cycle's verify
        // writes it at position 0.
        last_in_cache = false;
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

    // DELIBERATE DIVERGENCE from `run_mtp_turn`: cancellation is checked at
    // the cycle top and inside the pre-commit stop-clamp, NOT per emitted
    // token. This bounds cancel latency to <= block_size + 1 tokens while
    // preserving the commit-exactly-once, caches-never-desync invariant —
    // gemma4 has no post-commit heal path, so a mid-emit cancel must never
    // be observed AFTER the cycle's commit already kept its slots.
    loop {
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
        if let Some(s) = streaming.as_ref()
            && s.cancelled.load(Ordering::Relaxed)
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

        let _stream_ctx = crate::stream::StreamContext::new(generation_stream);

        // Per-cycle draft cap. `remaining >= 1` here (the length check above
        // broke otherwise). `remaining - 1` reserves the boundary token's
        // budget slot: a cycle emits at most `L + 1` tokens.
        let remaining: usize = max_as_usize.saturating_sub(generated.len());
        let l_cap: usize = block_size.min(p.mtp_depth).min(remaining.saturating_sub(1));

        // `L_cap == 0` (remaining == 1): skip propose entirely — the cycle
        // degenerates to a single AR step THROUGH verify (verify_ids =
        // [anchor] only), keeping the anchor's K/V write on the one path.
        let proposal = if l_cap >= 1 {
            profiler.begin("dspark_propose");
            let res = step.propose(anchor, l_cap, p, rng);
            profiler.end();
            res?
        } else {
            DsparkProposal {
                draft_ids: Vec::new(),
                draft_dists: Vec::new(),
            }
        };
        // Contract enforcement at the proposal boundary: `propose` may
        // return FEWER than `l_cap` (confidence truncation), NEVER more. An
        // over-long block would inflate the verify write past the
        // `remaining - 1` budget cap's target-cache slot expectations, and
        // the stop-clamp runs AFTER verify so it cannot un-write those
        // slots — a hard error surfaces the stepper bug instead of masking
        // it.
        let draft_len = proposal.draft_ids.len();
        if draft_len > l_cap {
            return Err(Error::from_reason(format!(
                "DSpark propose over-returned: {draft_len} draft tokens for a cap of {l_cap} \
                 (the DsparkStepper::propose contract allows fewer, never more)"
            )));
        }

        let mut verify_ids: Vec<u32> = Vec::with_capacity(1 + draft_len);
        verify_ids.push(anchor);
        verify_ids.extend(proposal.draft_ids.iter().map(|&id| id as u32));

        profiler.begin("dspark_verify");
        let verify_res = step.verify(&verify_ids);
        profiler.end();
        let verify_out = verify_res?;
        let logits = verify_out.logits; // [1, 1+L, vocab]

        // Acceptance: `k` accepted drafts (prefix of `draft_ids`) + ONE
        // boundary token (bonus on full accept, residual on rejection).
        profiler.begin("dspark_accept");
        let accept_res: Result<(usize, u32)> = if greedy_fast {
            // Greedy fast path (penalties no-op): ONE batched argmax over
            // all 1+L rows, a single eval, then CPU reads. No RNG consumed
            // (mirrors `accept_with_residual`'s T=0 shortcut).
            (|| {
                let argmax_arr = logits.argmax(-1, None)?;
                argmax_arr.eval();
                let mut target_argmax: Vec<i32> = Vec::with_capacity(draft_len + 1);
                for i in 0..=draft_len {
                    target_argmax.push(argmax_arr.item_at_int32(i)?);
                }
                let mut k = 0usize;
                while k < draft_len && target_argmax[k] == proposal.draft_ids[k] {
                    k += 1;
                }
                Ok((k, target_argmax[k] as u32))
            })()
        } else if greedy_temp {
            // Penalized-greedy path (T=0 with active penalties): per-row
            // `apply_all_penalties` over the sequentially extended history,
            // then a plain argmax decides accept/boundary. No proposal
            // dists, no RNG — behaviorally identical to `run_mtp_cycle`'s
            // legacy dense branch at T=0, where `accept_with_residual`'s
            // argmax shortcut reads the penalized row and the bonus
            // `sample(penalized)` degenerates to its argmax.
            (|| {
                let vocab = logits.shape_at(2)?;
                let mut hist_extended: Vec<u32> = hist.clone();
                let mut k = 0usize;
                let mut boundary: Option<u32> = None;
                for i in 0..draft_len {
                    let v_slice = logits.slice(&[0, i as i64, 0], &[1, (i + 1) as i64, vocab])?;
                    let v_1d = v_slice.squeeze(Some(&[0, 1]))?;
                    let penalized = apply_all_penalties(v_1d, &hist_extended, p)?;
                    let argmax_arr = penalized.argmax(0, None)?;
                    argmax_arr.eval();
                    let target_id = argmax_arr.item_at_int32(0)?;
                    if target_id == proposal.draft_ids[i] {
                        k += 1;
                        hist_extended.push(target_id as u32);
                    } else {
                        boundary = Some(target_id as u32);
                        break;
                    }
                }
                let boundary_id = match boundary {
                    Some(b) => b,
                    None => {
                        // All L drafts accepted: boundary = penalized
                        // argmax of row L.
                        let i = draft_len;
                        let v_slice =
                            logits.slice(&[0, i as i64, 0], &[1, (i + 1) as i64, vocab])?;
                        let v_1d = v_slice.squeeze(Some(&[0, 1]))?;
                        let penalized = apply_all_penalties(v_1d, &hist_extended, p)?;
                        let argmax_arr = penalized.argmax(0, None)?;
                        argmax_arr.eval();
                        argmax_arr.item_at_int32(0)? as u32
                    }
                };
                Ok((k, boundary_id))
            })()
        } else {
            // Sampled path: per-position Leviathan accept + residual
            // resample against the stepper's proposal densities.
            (|| {
                if proposal.draft_dists.len() != draft_len {
                    return Err(Error::from_reason(format!(
                        "DSpark sampled accept requires one proposal distribution per \
                         drafted token (got {} dists for {} drafts)",
                        proposal.draft_dists.len(),
                        draft_len
                    )));
                }
                let vocab = logits.shape_at(2)?;
                let mut hist_extended: Vec<u32> = hist.clone();
                let mut k = 0usize;
                let mut boundary: Option<u32> = None;
                for i in 0..draft_len {
                    let v_slice = logits.slice(&[0, i as i64, 0], &[1, (i + 1) as i64, vocab])?;
                    let v_1d = v_slice.squeeze(Some(&[0, 1]))?;
                    let penalized = apply_all_penalties(v_1d, &hist_extended, p)?;
                    let p_target = sampling::sampling_distribution(&penalized, p.sampling_config)?
                        .astype(DType::Float32)?;
                    p_target.eval();
                    let (accept, out_tok) = sampling::accept_with_residual(
                        &p_target,
                        &proposal.draft_dists[i],
                        proposal.draft_ids[i],
                        &sampling_cfg,
                        rng,
                    )?;
                    if accept {
                        k += 1;
                        hist_extended.push(out_tok as u32);
                    } else {
                        boundary = Some(out_tok as u32);
                        break;
                    }
                }
                let boundary_id = match boundary {
                    Some(b) => b,
                    None => {
                        // All L drafts accepted: bonus sample from row L.
                        let i = draft_len;
                        let v_slice =
                            logits.slice(&[0, i as i64, 0], &[1, (i + 1) as i64, vocab])?;
                        let v_1d = v_slice.squeeze(Some(&[0, 1]))?;
                        let penalized = apply_all_penalties(v_1d, &hist_extended, p)?;
                        let bonus = sampling::sample(&penalized, p.sampling_config)?;
                        bonus.eval();
                        bonus.item_at_int32(0)? as u32
                    }
                };
                Ok((k, boundary_id))
            })()
        };
        profiler.end();
        let (accepted_drafts_k, boundary_id) = accept_res?;

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
                if let Some(s) = streaming.as_ref()
                    && s.cancelled.load(Ordering::Relaxed)
                {
                    stop = Some(CycleStop::Cancelled);
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
        profiler.begin("dspark_commit");
        let commit_res = step.commit(keep, 1 + draft_len);
        profiler.end();
        commit_res?;

        // Per-cycle acceptance stats: DRAFTED depth `L` and accepted draft
        // count `k` — the same argument semantics as `run_mtp_cycle`'s
        // `record_mtp_cycle(effective_depth, accepted_drafts)` (boundary
        // excluded from both). Degenerate `L == 0` cycles are plain AR
        // steps and are NOT recorded.
        if draft_len >= 1 {
            profiler.record_mtp_cycle(draft_len, accepted_drafts_k);
        }

        // Emit the clamped prefix — `run_mtp_turn`'s emit-block bookkeeping
        // (push / observe / detok / emitter), with the stop checks already
        // resolved by the clamp above and NO per-token cancel check (see
        // the divergence note at the loop top). Cancel parity: the token on
        // which the clamp observed the cancel is committed but NOT streamed,
        // matching the mtp emit arm's break-before-detok.
        profiler.begin("dspark_emit_loop");
        for (idx, &tok_id) in accepted[..emit_count].iter().enumerate() {
            generated.push(tok_id);
            hist.push(tok_id);
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
                s.emitter.on_token_text(
                    &token_text,
                    _is_reasoning,
                    p.include_reasoning,
                    s.callback,
                );
            }
        }
        profiler.end();

        // Every-256-emitted-token cache clear (mtp_turn's cadence).
        if generated.len() >= last_clear_at + 256 {
            crate::array::synchronize_and_clear_cache();
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
            break;
        }

        // Continue: the boundary becomes the next cycle's anchor.
        anchor = boundary_id;
        let y_arr = MxArray::from_int32(&[boundary_id as i32], &[1])?;
        step.eval_boundary(&y_arr);
        profiler.step();
    }

    profiler.snapshot_memory_after();
    profiler.report();

    Ok(DsparkTurnOutcome { last_in_cache })
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
    use std::time::Instant;

    use napi::bindgen_prelude::*;

    use crate::array::MxArray;
    use crate::decode_profiler::DecodeProfiler;
    use crate::engine::backend::{
        ChatBackend, ChunkSink, DefaultStreamEmitter, DsparkBackend, DsparkProposal, DsparkStepper,
        DsparkTurnSetup, DsparkVerifyOutput, FinalizeArgs, ResetScope, SaveStateArgs, TurnSetup,
    };
    use crate::engine::decode::StreamingCtx;
    use crate::engine::params::ChatParams;
    use crate::engine::penalties::ReasoningTracker;
    use crate::engine::types::{ChatResult, ChatStreamChunk};
    use crate::sampling::SamplingConfig;
    use crate::stream::{DeviceType, Stream};
    use crate::tokenizer::Qwen3Tokenizer;

    use super::{DsparkTurnArgs, run_dspark_turn};

    /// One recorded `DsparkStepper` call, tagged with the payload the loop
    /// handed it so tests assert the exact per-cycle sequencing.
    #[derive(Clone, Debug, PartialEq, Eq)]
    enum Call {
        Propose { anchor: u32, max_len: usize },
        Verify { ids: Vec<u32> },
        Commit { keep: usize, total: usize },
        EvalBoundary { token: i32 },
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
        ledger: Rc<RefCell<Vec<Call>>>,
    }

    impl MockDsparkStepper {
        fn script(&self) -> CycleScript {
            self.cycles
                .get(self.cursor.get())
                .cloned()
                .unwrap_or_else(|| CycleScript::greedy(Vec::new(), Vec::new()))
        }
    }

    impl DsparkStepper for MockDsparkStepper {
        fn propose(
            &mut self,
            anchor_id: u32,
            max_len: usize,
            _params: &ChatParams,
            _rng: &mut dyn rand::Rng,
        ) -> Result<DsparkProposal> {
            self.ledger.borrow_mut().push(Call::Propose {
                anchor: anchor_id,
                max_len,
            });
            let script = self.script();
            let draft_dists = script
                .draft_dists
                .iter()
                .map(|row| MxArray::from_float32(row, &[self.vocab]))
                .collect::<Result<Vec<_>>>()?;
            Ok(DsparkProposal {
                draft_ids: script.draft_ids.clone(),
                draft_dists,
            })
        }

        fn verify(&mut self, verify_ids: &[u32]) -> Result<DsparkVerifyOutput> {
            self.ledger.borrow_mut().push(Call::Verify {
                ids: verify_ids.to_vec(),
            });
            let script = self.script();
            let rows = verify_ids.len();
            let mut flat: Vec<f32> = Vec::with_capacity(rows * self.vocab as usize);
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

        fn commit(&mut self, keep: usize, total_written: usize) -> Result<()> {
            self.ledger.borrow_mut().push(Call::Commit {
                keep,
                total: total_written,
            });
            self.cursor.set(self.cursor.get() + 1);
            Ok(())
        }

        fn eval_boundary(&self, token: &MxArray) {
            token.eval();
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
        ledger: Rc<RefCell<Vec<Call>>>,
        begin_calls: Cell<usize>,
        /// `block_size` observed by `begin_dspark_decode` (setup plumbing).
        seen_block_size: Cell<usize>,
    }

    impl MockDsparkBackend {
        fn greedy(vocab: i64, cycles: Vec<CycleScript>) -> Self {
            Self {
                vocab,
                cycles,
                neg_fill: 0.0,
                cancel_on_verify: None,
                ledger: Rc::new(RefCell::new(Vec::new())),
                begin_calls: Cell::new(0),
                seen_block_size: Cell::new(usize::MAX),
            }
        }

        /// Sampled-path backend: `-1e30` fill makes verify softmax rows
        /// exactly one-hot (see [`logits_row`]).
        fn sampled(vocab: i64, cycles: Vec<CycleScript>) -> Self {
            let mut b = Self::greedy(vocab, cycles);
            b.neg_fill = -1.0e30;
            b
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
    }

    impl DsparkBackend for MockDsparkBackend {
        type DsparkDecode<'a>
            = MockDsparkStepper
        where
            Self: 'a;

        fn begin_dspark_decode(
            &mut self,
            setup: &DsparkTurnSetup<'_>,
        ) -> Result<Self::DsparkDecode<'_>> {
            self.begin_calls.set(self.begin_calls.get() + 1);
            self.seen_block_size.set(setup.block_size);
            Ok(MockDsparkStepper {
                vocab: self.vocab,
                cycles: self.cycles.clone(),
                neg_fill: self.neg_fill,
                cursor: Cell::new(0),
                verify_count: Cell::new(0),
                cancel_on_verify: self.cancel_on_verify.clone(),
                ledger: Rc::clone(&self.ledger),
            })
        }
    }

    /// T=0 greedy `ChatParams` with all penalties at their no-op defaults —
    /// drives the loop's batched-argmax fast path.
    fn greedy_params() -> ChatParams {
        ChatParams {
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
        result: Result<super::DsparkTurnOutcome>,
        generated: Vec<u32>,
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
        let mut tracker = ReasoningTracker::new(false, None, None);
        let mut profiler = DecodeProfiler::new("dspark_turn_test", "test");
        let mut generated: Vec<u32> = Vec::new();
        let mut token_history: Vec<u32> = Vec::new();
        let mut finish_reason = String::from("length");
        let mut first_token_instant: Option<Instant> = None;
        let y = MxArray::from_int32(&[first_token as i32], &[1])
            .unwrap_or_else(|e| panic!("y construction: {}", e.reason));
        let generation_stream = Stream::new(DeviceType::Gpu);
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
            },
            None,
        );

        // The commit-exactly-once design keeps the histories in lockstep on
        // BOTH the success and the error path (emission always pushes both).
        assert_eq!(token_history, generated, "history must mirror generated");

        RawTurnOut {
            result,
            generated,
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
                Call::Commit { keep, total } => Some((*keep, *total)),
                _ => None,
            })
            .collect()
    }

    // ---- 1. emission order + per-cycle commit ledger ---------------------

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
    fn drive_streaming_turn(
        backend: &mut MockDsparkBackend,
        params: ChatParams,
        first_token: u32,
        eos_id: u32,
        block_size: usize,
        cancelled: Arc<AtomicBool>,
    ) -> StreamOut {
        let mut tracker = ReasoningTracker::new(false, None, None);
        let mut profiler = DecodeProfiler::new("dspark_stream_test", "test");
        let mut generated: Vec<u32> = Vec::new();
        let mut token_history: Vec<u32> = Vec::new();
        let mut finish_reason = String::from("length");
        let mut first_token_instant: Option<Instant> = None;
        let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(0xD5_9A2B_C0DE);
        let y = MxArray::from_int32(&[first_token as i32], &[1])
            .unwrap_or_else(|e| panic!("y construction: {}", e.reason));
        let generation_stream = Stream::new(DeviceType::Gpu);
        let max_new_tokens = params.max_new_tokens;

        let tokenizer = tiny_tokenizer();
        let mut decode_stream = tokenizer.decode_stream(true);
        let sink = RecSink {
            chunks: Mutex::new(Vec::new()),
        };
        let mut streamed_text_len = 0usize;
        let mut last_is_reasoning = false;
        let mut emitter = DefaultStreamEmitter;

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
            },
            Some(StreamingCtx {
                callback: &sink,
                cancelled: &cancelled,
                decode_stream: &mut decode_stream,
                tokenizer: &tokenizer,
                streamed_text_len: &mut streamed_text_len,
                last_is_reasoning: &mut last_is_reasoning,
                emitter: &mut emitter,
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
