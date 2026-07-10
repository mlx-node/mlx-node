//! Engine-owned MTP (Multi-Token Prediction) propose/verify whole-turn
//! path — the MTP analog of [`crate::engine::paged_turn`]. Families opt in
//! via [`crate::engine::backend::MtpBackend`]; their
//! `ChatBackend::run_speculative_turn` delegates to `run_mtp_turn`.
//!
//! SCAFFOLD STEP: the relocated `decode_loop_mtp!` outer body
//! (`run_mtp_turn`) and the relocated `run_mtp_cycle_inner` (`run_mtp_cycle`)
//! land in later steps. Today this module carries ONLY the
//! [`MtpStepper`](crate::engine::backend::MtpStepper) contract's test
//! harness — a scripted [`MockMtpStepper`] double + call-ledger unit tests
//! that PROVE the trait + GAT lifetimes + the strictly-sequential
//! `&mut self` borrow model compile and are usable. Nothing in production
//! calls this module yet, so the families' MTP behavior is byte-identical.

use std::sync::atomic::Ordering;
use std::time::Instant;

use napi::bindgen_prelude::*;

use crate::array::MxArray;
use crate::decode_profiler::DecodeProfiler;
use crate::engine::backend::{MtpBackend, MtpStepper, MtpTurnSetup};
use crate::engine::params::ChatParams;
use crate::engine::penalties::{ReasoningTracker, apply_all_penalties};
use crate::models::qwen3_5::mtp_decode::{
    MtpCommitAnchor, MtpCycleOutcome, MtpVerifyOutput, mtp_batch_target_arrays_enabled,
    mtp_defer_verify_hidden_eval, mtp_draft_sampling_config, mtp_greedy_argmax_only_verify_enabled,
    mtp_native_sparse_verify_enabled, mtp_target_distribution_first_enabled, mtp_trace_acceptance,
    mtp_verify_async_eval, mtp_verify_top1_check_enabled, sparse_accept_gate,
    trace_acceptance_dense, trace_acceptance_emit, trace_acceptance_greedy,
    trace_acceptance_sparse,
};
use crate::sampling;
use crate::stream::Stream;

use crate::engine::decode::{StreamingCtx, mtp_trace_logits, trace_top2};

/// One MTP draft+verify cycle, generic over [`MtpStepper`] — a VERBATIM,
/// mechanical relocation of
/// [`crate::models::qwen3_5::mtp_decode::run_mtp_cycle_inner`], calling
/// `step.*` where the original calls `ops.*`. Every surrounding line
/// (sampling, accept-branch selection, async_eval scheduling, K arithmetic,
/// commit-anchor handling, profiler begin/end, the `verify_hiddens[:, K, :]`
/// return slice, adaptive/EV-depth orchestration) is byte-for-byte identical
/// in logic and ORDER.
///
/// DEAD CODE in this step: nothing in production drives it yet (the family
/// steppers + the engine-owned `run_mtp_turn` loop that calls it land in a
/// later step), so the relocated `run_mtp_cycle_inner` remains the sole
/// production cycle and the families stay byte-identical. Exercised only by
/// the module's mock tests.
///
/// Translated `ops.*` → `step.*` swap sites (the ONLY substantive change
/// vs the original body):
///   * `(ops.draft_step)(a, b)` → `step.draft_step(a, b)`
///   * `(ops.snapshot_main_linear)()` → `step.snapshot_main_linear()`
///   * the verify dispatch: `ops.verify_step_argmax_only` /
///     `ops.verify_step_sparse` boxed-`Option` fields →
///     `step.verify_step_argmax_only(..)` / `step.verify_step_sparse(..)`
///     (each returns `Option<Result<..>>`: `Some` = use it, `None` = fall
///     back to `step.verify_step(..)`); `(ops.verify_step)(..)` →
///     `step.verify_step(..)`
///   * `(ops.commit_mtp)(..)` → `step.commit_mtp(..)`
///   * `(ops.rollback)(k, d)` → `step.rollback(k, d)`
///   * `(ops.restore_and_replay_main)(ids, emb)` →
///     `step.restore_and_replay_main(ids, emb)`
#[allow(dead_code)]
pub(crate) fn run_mtp_cycle<S: MtpStepper>(
    step: &mut S,
    prev_hidden_in: MxArray,
    prev_emb_in: MxArray,
    last_committed_id: u32,
    embedding_weight: &MxArray,
    token_history: &[u32],
    params: &ChatParams,
    rng: &mut impl rand::Rng,
    profiler: &mut crate::decode_profiler::DecodeProfiler,
    depth: usize,
    mut ev_depth_policy: Option<
        &mut crate::models::qwen3_5::adaptive_depth::ExpectedValueDepthPolicy,
    >,
    commit_anchor: MtpCommitAnchor,
) -> Result<(MtpCycleOutcome, MxArray)> {
    use crate::array::{DType, MxArray as A};

    debug_assert!(depth >= 1, "run_mtp_cycle: depth must be >= 1");

    // Keep the ORIGINAL cycle-seed hidden alive for the committed-history
    // commit. `prev_hidden_in` is h(token before `last_committed_id`) —
    // the correct hidden to pair with the
    // embedding of `last_committed_id` for that token's MTP slot. The
    // draft loop below moves `prev_hidden_in` into the mutable
    // `prev_hidden` local and overwrites it step by step, so clone the
    // (cheap, refcounted) handle now before that happens.
    let commit_seed_hidden = prev_hidden_in.clone();

    // Step 1: D draft steps via the per-step `draft_step` loop.
    profiler.begin("mtp_draft_total");
    let temperature = params
        .sampling_config
        .and_then(|c| c.temperature)
        .unwrap_or(1.0);
    let sampling_cfg = params.sampling_config.unwrap_or_default();
    let draft_sampling_cfg = mtp_draft_sampling_config(sampling_cfg);
    // Fast-path eligibility: at T=0 with all penalties at defaults, the
    // per-position accept decision collapses to
    // `argmax(verify_logits[i]) == draft_id[i]` (the argmax shortcut in
    // `accept_with_residual`). Compute this before draft construction so
    // the deterministic path can avoid building unused draft probability
    // tensors.
    let penalties_no_op = params.repetition_penalty == 1.0
        && params.presence_penalty == 0.0
        && params.frequency_penalty == 0.0;
    let use_sparse_accept =
        sparse_accept_gate() && sampling::is_greedy_temperature(temperature) && penalties_no_op;
    let use_sparse_stochastic_accept = mtp_batch_target_arrays_enabled()
        && !sampling::is_greedy_temperature(temperature)
        && penalties_no_op
        && sampling::sparse_distribution_supported(&sampling_cfg)
        && sampling::sparse_distribution_supported(&draft_sampling_cfg);
    let mut prev_hidden = prev_hidden_in;
    let mut prev_emb = prev_emb_in;
    let mut draft_ids: Vec<i32> = Vec::with_capacity(depth);
    let mut draft_probs: Vec<MxArray> = if use_sparse_accept || use_sparse_stochastic_accept {
        Vec::new()
    } else {
        Vec::with_capacity(depth)
    };
    let mut draft_sparse_probs: Vec<sampling::SparseDistribution> = if use_sparse_stochastic_accept
    {
        Vec::with_capacity(depth)
    } else {
        Vec::new()
    };
    // `step_input_id` is the token whose hidden/embedding seed this
    // draft step: `last_committed_id` for step 0, then each prior
    // drafted id. Logged per step so a debug run can reconstruct
    // the full draft chain.
    let mut step_input_id = last_committed_id as i32;
    for step_idx in 0..depth {
        let (h_next, draft_logits) = step.draft_step(&prev_hidden, &prev_emb)?;
        let logits_1d = if use_sparse_accept {
            None
        } else {
            // draft_logits is [1, vocab]; squeeze to [vocab] for the
            // probability distribution consumed by accept/reject.
            Some(draft_logits.squeeze(Some(&[0]))?)
        };
        let probs = if use_sparse_accept || use_sparse_stochastic_accept {
            None
        } else {
            // The stochastic accept path consumes this `probs` as
            // the proposal density `q` inside `accept_with_residual`
            // (`min(1, p/q)` + `(p - q)+` residual). For Leviathan-Chen
            // exactness `q` MUST be the distribution the
            // draft token was actually drawn from. The draft id below (T>0
            // branch) is drawn via `sampling::sample(&draft_logits, ..)`
            // → `mlx_compiled_sample_full`, which converts logits→logprobs,
            // applies the top_k/top_p/min_p filters ON THE LOGPROBS, then
            // applies temperature ONLY at the final categorical draw.
            //
            // A `softmax(apply_sampling(logits))` rebuild did NOT match that
            // draw: `apply_sampling` scales by temperature FIRST and then
            // filters (and it ERRORS at T=0 because `apply_temperature`
            // rejects `temperature <= 0`). Build `q` from the SAME compiled
            // filter chain instead, via `sampling::sampling_distribution`,
            // which returns `softmax(filtered_logits / temperature)` under the
            // active `sampler_parity_mode()` — matching the draw by
            // construction for ALL configs (incl. the common `top_k==0` plain
            // temperature/top_p case) and both parity modes.
            //
            // NOTE: at T=0 the non-sparse `else` accept branch is only reached
            // when `MLX_MTP_SPARSE_ACCEPT` is disabled; in that case
            // `accept_with_residual` takes its argmax-only shortcut and never
            // reads `q`. `sampling_distribution` at T=0 returns the (valid,
            // 1D `[vocab]`) one-hot argmax distribution — it does NOT error,
            // and is ignored by the accept shortcut — so every T=0 commit
            // decision stays byte-identical. Only the T>0 probability-ratio
            // path is corrected.
            let raw_1d = logits_1d.as_ref().ok_or_else(|| {
                Error::from_reason(
                    "MTP draft logits_1d unexpectedly None (sparse-accept gating mismatch)",
                )
            })?;
            // `sample()` at the draw site uses `params.sampling_config` (the
            // target config), so build `q` from the SAME config — not
            // `draft_sampling_cfg`, which only feeds the sparse path's draw.
            Some(
                sampling::sampling_distribution(raw_1d, params.sampling_config)?
                    .astype(DType::Float32)?,
            )
        };
        let mut sparse_draft = None;
        let tok_id = if use_sparse_stochastic_accept {
            let sparse_rows = sampling::sparse_distributions_from_logits(
                logits_1d.as_ref().ok_or_else(|| {
                    Error::from_reason(
                        "MTP draft logits_1d unexpectedly None (sparse-accept gating mismatch)",
                    )
                })?,
                &draft_sampling_cfg,
            )?
            .ok_or_else(|| {
                Error::from_reason(
                    "MTP sparse stochastic draft path became ineligible after gating",
                )
            })?;
            let draft_dist = sparse_rows.row_owned(0)?;
            let sampled = draft_dist.as_row().sample(rng)?;
            sparse_draft = Some(draft_dist);
            sampled
        } else {
            // Sample the drafted token using the same sampling pipeline
            // the main path uses — drafter and verifier must agree on
            // their proposal distribution for Leviathan-Chen.
            let tok = sampling::sample(&draft_logits, params.sampling_config)?;
            tok.eval();
            tok.item_at_int32(0)?
        };
        let draft_metrics = crate::models::qwen3_5::adaptive_depth::DraftMetrics {
            top1_prob_topk: sparse_draft
                .as_ref()
                .and_then(|dist| dist.as_row().top_entry().map(|(_, prob)| prob)),
        };
        tracing::trace!(
            target: "mlx_core::mtp::draft",
            step = step_idx,
            input_id = step_input_id,
            drafted_id = tok_id,
            "MTP per-step draft"
        );
        draft_ids.push(tok_id);
        if let Some(sparse_draft) = sparse_draft {
            draft_sparse_probs.push(sparse_draft);
        }
        if let Some(probs) = probs {
            draft_probs.push(probs);
        }
        // Keep the draft step's hidden/embedding handles alive even if the
        // EV gate stops here. The fixed-depth path always retains these
        // handles through the cycle tail; matching that lifetime matters
        // for MLX's lazy cache writes.
        prev_hidden = h_next;
        let id_arr = A::from_int32(&[tok_id], &[1])?;
        let emb_2d = embedding_weight.take(&id_arr, 0)?; // [1, hidden]
        let hidden = emb_2d.shape_at(1)?;
        prev_emb = emb_2d.reshape(&[1, 1, hidden])?;
        step_input_id = tok_id;
        if let Some(policy) = ev_depth_policy.as_mut()
            && draft_ids.len() < depth
        {
            profiler.begin("mtp_draft_gate");
            let decision =
                policy.should_continue_after_draft(draft_ids.len(), depth, draft_metrics);
            profiler.end();
            tracing::trace!(
                target: "mlx_core::mtp::adaptive",
                drafted_depth = draft_ids.len(),
                next_depth = decision.next_depth,
                expected_extra_accept = decision.expected_extra_accept,
                required_extra_accept = decision.required_extra_accept,
                continue_drafting = decision.continue_drafting,
                "MTP EV depth gate"
            );
            if !decision.continue_drafting {
                break;
            }
        }
    }
    profiler.end();
    let effective_depth = draft_ids.len();
    debug_assert!(
        effective_depth >= 1,
        "MTP EV depth gate must leave at least one draft token"
    );
    // `trace!` not `debug!` — the full `draft_ids` vector is per-token
    // detail; one record per cycle would flood a long decode at debug.
    tracing::trace!(
        target: "mlx_core::mtp",
        depth,
        effective_depth,
        draft_ids = ?draft_ids,
        "MTP draft phase complete"
    );

    // Step 2: build verify input [last_committed_id, d_0, ..., d_{D-1}].
    let mut verify_ids: Vec<i32> = Vec::with_capacity(effective_depth + 1);
    verify_ids.push(last_committed_id as i32);
    verify_ids.extend(draft_ids.iter().copied());
    let verify_in = A::from_int32(&verify_ids, &[1, (effective_depth + 1) as i64])?;
    // `trace!` not `debug!` — the full `verify_ids` vector is per-token
    // detail; keep debug to compact once-per-cycle summaries.
    tracing::trace!(
        target: "mlx_core::mtp",
        depth,
        effective_depth,
        last_committed_id,
        verify_ids = ?verify_ids,
        "MTP verify input built"
    );
    // Snapshot the main path's GDN linear caches + offset BEFORE verify
    // runs its D+1 sequential forwards. Verify mutates the main-path
    // caches in place; on rejection we restore from this snapshot and replay only
    // the K accepted drafts so the linear recurrent state matches the
    // committed token stream. On full accept the snapshot is discarded —
    // verify already left the linear state correctly advanced.
    profiler.begin("mtp_tape_snapshot");
    step.snapshot_main_linear();
    profiler.end();
    tracing::trace!(
        target: "mlx_core::mtp",
        depth,
        "MTP main-linear caches + offset snapshot taken (pre-verify)"
    );
    // Verify returns BOTH logits and per-position hiddens.
    // Logits: `[1, depth+1, vocab]`; hiddens: `[1, depth+1, hidden]`.
    // We hold off on slicing the hidden until after the accept loop
    // computes K (= number of accepted drafts) so we can pick
    // `verify_hiddens[:, K, :]` — the correct prediction context for
    // the next cycle's first MTP draft.
    // The gap between `mtp_cycle` and this floor is the headroom
    // available to algorithmic work.
    let verify_only_t0 = std::time::Instant::now();
    profiler.begin("mtp_verify_dispatch");
    let trace_logits = mtp_trace_logits();
    let trace_acceptance = mtp_trace_acceptance();
    let use_native_sparse_verify = use_sparse_stochastic_accept
        && mtp_native_sparse_verify_enabled()
        && sampling::sampler_parity_is_mtplx()
        && !trace_logits;
    let use_greedy_argmax_only_verify = use_sparse_accept
        && mtp_greedy_argmax_only_verify_enabled()
        && !trace_logits
        && !trace_acceptance
        && !mtp_verify_top1_check_enabled();
    let verify_step_res = if let Some(res) = use_greedy_argmax_only_verify
        .then(|| {
            profiler.begin("mtp_verify_dispatch_argmax_only");
            let res = step.verify_step_argmax_only(&verify_in, embedding_weight, effective_depth);
            profiler.end();
            res
        })
        .flatten()
    {
        res
    } else if let Some(res) = use_native_sparse_verify
        .then(|| {
            step.verify_step_sparse(&verify_in, embedding_weight, effective_depth, &sampling_cfg)
        })
        .flatten()
    {
        res
    } else {
        step.verify_step(&verify_in, embedding_weight, effective_depth)
    };
    profiler.end();
    let MtpVerifyOutput {
        logits: verify_logits,
        hiddens: verify_hiddens,
        target_argmax: verify_target_argmax,
        target_sparse: verify_target_sparse,
    } = verify_step_res?;
    tracing::debug!(
        target: "mlx_core::mtp",
        depth = effective_depth,
        requested_depth = depth,
        verify_tokens = effective_depth + 1,
        "MTP verify dispatched (batched target forward over depth+1 tokens)"
    );
    // Async-eval over verify outputs. By default we dispatch verify
    // (logits + hiddens) via `async_eval` instead of the synchronous
    // `eval()` below. The kernel launch returns immediately, letting the
    // CPU construct the accept loop's penalty / softmax / slice graph
    // while the verify command buffer is still executing on the GPU. The
    // first downstream `eval()` (the accept loop's `p_target.eval()` at
    // the per-position softmax) syncs on completion. Semantic equivalent
    // of MTPLX's `LAZY_VERIFY_LOGITS` (`MTPLX/mtplx/generation.py:49,
    // 3894`).
    //
    // We batch `verify_hiddens` into the same async_eval call so MLX's
    // scheduler can fuse it with the verify logits graph (they share
    // the per-position `final_norm` outputs). Only the post-accept
    // `verify_hiddens[:, K, :]` slice is actually realised on-device
    // by the chained-cycle path; for the default Step-A path the
    // batch eval is still cheap (one extra command-buffer entry).
    //
    // `MLX_MTP_VERIFY_ASYNC_EVAL=0` reverts to the synchronous
    // `verify_logits.eval()` barrier — byte-identical for
    // parity-debugging or hardware where the overlap budget is negligible.
    // Fast-path acceptance. When eligible, collapse the D+1 per-position
    // softmax materializations into ONE batched
    // `argmax(verify_logits, axis=-1)` op + one `.eval()` reading
    // D+1 int32 values.
    //
    // Why this is safe:
    //   * T=0 → `accept_with_residual` only reads `argmax(p_target)`
    //     vs `draft_id`. `softmax` is monotone so `argmax(softmax(x))
    //     == argmax(x)`. No probabilities are ever consumed.
    //   * Penalties default → `apply_all_penalties` is the identity,
    //     so `hist_extended` does NOT affect the per-position logits.
    //     We can compute all D+1 argmaxes BEFORE the accept loop.
    //   * Bonus token on full-accept = argmax at position D, also a
    //     trivial readout from the same batched array.
    //
    // When ineligible (T>0, or any penalty non-default), fall through to
    // the per-position path below.

    let sparse_verify_argmax = if use_sparse_accept {
        verify_target_argmax.as_ref()
    } else {
        None
    };
    let verify_logits_ref = verify_logits.as_ref();

    profiler.begin("mtp_verify_eval");
    let defer_hidden = mtp_defer_verify_hidden_eval();
    let target_distribution_first = use_sparse_stochastic_accept
        && defer_hidden
        && mtp_target_distribution_first_enabled()
        && verify_logits_ref.is_some()
        && !trace_logits;
    if target_distribution_first {
        tracing::debug!(
            target: "mlx_core::mtp::verify_async_eval",
            depth = effective_depth,
            requested_depth = depth,
            "W6.23 target-distribution-first verify scheduling"
        );
    } else if mtp_verify_async_eval() {
        tracing::debug!(
            target: "mlx_core::mtp::verify_async_eval",
            depth = effective_depth,
            requested_depth = depth,
            defer_hidden,
            "W6.9 async_eval verify outputs"
        );
        if let Some(argmax_arr) = sparse_verify_argmax {
            let mut eval_arrays: Vec<&MxArray> =
                Vec::with_capacity(1 + usize::from(trace_logits) + usize::from(!defer_hidden));
            eval_arrays.push(argmax_arr);
            if trace_logits && let Some(verify_logits) = verify_logits_ref {
                eval_arrays.push(verify_logits);
            }
            if !defer_hidden {
                eval_arrays.push(&verify_hiddens);
            }
            MxArray::async_eval_arrays(&eval_arrays);
        } else if let Some(verify_logits) = verify_logits_ref {
            if defer_hidden {
                MxArray::async_eval_arrays(&[verify_logits]);
            } else {
                MxArray::async_eval_arrays(&[verify_logits, &verify_hiddens]);
            }
        } else if !defer_hidden {
            MxArray::async_eval_arrays(&[&verify_hiddens]);
        }
    } else {
        // We materialize logits now so per-position slicing reads
        // from a CPU-resident buffer for penalty application. The
        // hiddens ride on the same lazy graph; we only eval the
        // K-th slice below.
        //
        // Note: the sparse-accept path also benefits from this eager
        // eval — folding verify materialization into the accept-loop
        // argmax op (one combined sync) measured ~10% slower than two
        // separate syncs. The eager eval here lets MLX's scheduler
        // pipeline the verify command buffer with the subsequent argmax
        // dispatch build, which the combined-eval variant defeats. Kept
        // unconditional.
        if let Some(argmax_arr) = sparse_verify_argmax {
            argmax_arr.eval();
            if trace_logits && let Some(verify_logits) = verify_logits_ref {
                verify_logits.eval();
            }
        } else if let Some(verify_logits) = verify_logits_ref {
            verify_logits.eval();
        } else if !defer_hidden {
            verify_hiddens.eval();
        }
        tracing::debug!(
            target: "mlx_core::mtp::verify_async_eval",
            depth = effective_depth,
            requested_depth = depth,
            sparse_argmax = sparse_verify_argmax.is_some(),
            "verify eval (synchronous; async-eval disabled)"
        );
    }
    profiler.end();
    profiler.record_duration("mtp_verify_floor", verify_only_t0.elapsed());
    let vocab = if let Some(verify_logits) = verify_logits_ref {
        verify_logits.shape_at(2)?
    } else if let Some(target_sparse) = verify_target_sparse.as_ref() {
        target_sparse.vocab_size() as i64
    } else {
        embedding_weight.shape_at(0)?
    };

    // Step 3: per-position accept/reject. Build extended history as
    // we accept; rejecting at position i halts the loop.
    let mut accepted_tokens: Vec<u32> = Vec::with_capacity(effective_depth + 1);
    let mut all_accepted = true;
    let mut rejection_residual: Option<i32> = None;

    if use_sparse_accept {
        // ONE batched argmax over all D+1 verify positions. Shape
        // `[1, D+1, vocab]` → `[1, D+1]` int32. At T=0 we care only
        // about per-position argmax — no full-vocab softmax
        // materialization needed.
        //
        // `verify_logits` may still be lazy from the verify dispatch
        // (especially under `MLX_MTP_VERIFY_ASYNC_EVAL=1`). The
        // `.eval()` below is the SINGLE sync point for the accept
        // loop — vs the D × per-position `p_target.eval()`
        // path that forces D full-vocab softmaxes through Metal.
        profiler.begin("mtp_accept_argmax");
        let fallback_argmax;
        let argmax_arr = if let Some(argmax_arr) = sparse_verify_argmax {
            argmax_arr
        } else {
            let verify_logits = verify_logits_ref.ok_or_else(|| {
                Error::from_reason(
                    "MTP greedy sparse accept requires verifier logits or precomputed target argmax",
                )
            })?;
            fallback_argmax = verify_logits.argmax(-1, None)?;
            &fallback_argmax
        };
        argmax_arr.eval();

        // Extract D+1 int32s into a CPU buffer. `verify_logits` was
        // `[1, D+1, vocab]`; the argmax over the last axis yields
        // `[1, D+1]`. We read flat positions 0..=depth.
        let mut target_argmax: Vec<i32> = Vec::with_capacity(effective_depth + 1);
        for i in 0..=effective_depth {
            target_argmax.push(argmax_arr.item_at_int32(i)?);
        }
        if sparse_verify_argmax.is_some() && mtp_verify_top1_check_enabled() {
            let verify_logits = verify_logits_ref.ok_or_else(|| {
                Error::from_reason("MTP verifier top1 check requires verifier logits")
            })?;
            let fallback_argmax = verify_logits.argmax(-1, None)?;
            fallback_argmax.eval();
            for (i, &compiled_id) in target_argmax.iter().enumerate() {
                let fallback_id = fallback_argmax.item_at_int32(i)?;
                if compiled_id != fallback_id {
                    return Err(Error::from_reason(format!(
                        "MTP verifier top1 mismatch at slot {i}: compiled={compiled_id}, fallback={fallback_id}"
                    )));
                }
            }
        }
        profiler.end();

        // Accept loop runs entirely on CPU buffers — no further GPU
        // syncs. The Leviathan-Chen accept-reject coin is unused at
        // T=0 (deterministic argmax decision); `rng` is intentionally
        // not advanced, matching `accept_with_residual`'s T=0
        // shortcut (zero RNG consumed).
        profiler.begin("mtp_accept_loop");
        for i in 0..effective_depth {
            let target_id = target_argmax[i];
            let accept = target_id == draft_ids[i];
            if trace_acceptance {
                let top2 = verify_logits_ref.and_then(|verify_logits| {
                    verify_logits
                        .slice(&[0, i as i64, 0], &[1, (i + 1) as i64, vocab])
                        .and_then(|s| s.squeeze(Some(&[0, 1])))
                        .and_then(|v1d| trace_top2(&v1d, vocab))
                        .ok()
                });
                trace_acceptance_greedy(
                    effective_depth,
                    i,
                    token_history.len(),
                    last_committed_id,
                    draft_ids[i],
                    target_id,
                    accept,
                    top2.as_ref(),
                );
            }
            tracing::trace!(
                target: "mlx_core::mtp::accept",
                pos = i,
                draft_id = draft_ids[i],
                target_id,
                accepted = accept,
                "MTP sparse accept position"
            );
            if accept {
                let id_u = target_id as u32;
                accepted_tokens.push(id_u);
            } else {
                all_accepted = false;
                rejection_residual = Some(target_id);
                accepted_tokens.push(target_id as u32);
                break;
            }
        }
        if all_accepted {
            // Bonus token = argmax at position D. Same batched
            // array, no extra ops, no extra eval.
            let bonus_id = target_argmax[effective_depth] as u32;
            tracing::trace!(
                target: "mlx_core::mtp::accept",
                bonus_id,
                "MTP bonus token (full accept, sparse path)"
            );
            accepted_tokens.push(bonus_id);
        }
        profiler.end();
    } else if use_sparse_stochastic_accept {
        profiler.begin("mtp_accept_sparse_probs");
        let target_sparse_from_logits;
        let target_sparse = if let Some(rows) = verify_target_sparse.as_ref() {
            rows.validate_for_accept(effective_depth + 1, vocab as usize, &sampling_cfg)?;
            rows
        } else {
            let verify_logits = verify_logits_ref.ok_or_else(|| {
                Error::from_reason(
                    "MTP sparse stochastic target path requires verifier logits or precomputed sparse rows",
                )
            })?;
            target_sparse_from_logits =
                sampling::sparse_distributions_from_logits(verify_logits, &sampling_cfg)?
                    .ok_or_else(|| {
                        Error::from_reason(
                            "MTP sparse stochastic target path became ineligible after gating",
                        )
                    })?;
            &target_sparse_from_logits
        };
        profiler.end();

        // Exact stochastic accept loop over tiny CPU-side top-k distributions.
        // No per-position full-vocab softmax/eval; rejection residuals and the
        // full-accept bonus sample from the same precomputed target rows.
        profiler.begin("mtp_accept_loop");
        // `i` indexes several parallel collections (`target_sparse`,
        // `draft_sparse_probs`, `draft_ids`) and doubles as the trace `pos`,
        // so a single `enumerate()` over one of them would not be clearer.
        #[allow(clippy::needless_range_loop)]
        for i in 0..effective_depth {
            let target_p = target_sparse.row(i)?;
            let draft_q = draft_sparse_probs
                .get(i)
                .ok_or_else(|| {
                    Error::from_reason(format!(
                        "MTP sparse stochastic draft distribution missing at position {}",
                        i
                    ))
                })?
                .as_row();
            let (accept, out_tok) =
                sampling::accept_with_residual_sparse(target_p, draft_q, draft_ids[i], rng)?;
            if trace_acceptance {
                trace_acceptance_sparse(
                    "sparse_stochastic",
                    effective_depth,
                    i,
                    token_history.len(),
                    last_committed_id,
                    draft_ids[i],
                    target_p,
                    draft_q,
                    accept,
                    out_tok,
                );
            }
            tracing::trace!(
                target: "mlx_core::mtp::accept",
                pos = i,
                draft_id = draft_ids[i],
                out_tok,
                accepted = accept,
                "MTP sparse stochastic accept position"
            );
            if accept {
                let id_u = out_tok as u32;
                accepted_tokens.push(id_u);
            } else {
                all_accepted = false;
                rejection_residual = Some(out_tok);
                accepted_tokens.push(out_tok as u32);
                break;
            }
        }

        if all_accepted {
            let bonus_id = target_sparse.row(effective_depth)?.sample(rng)? as u32;
            tracing::trace!(
                target: "mlx_core::mtp::accept",
                bonus_id,
                "MTP bonus token (full accept, sparse stochastic path)"
            );
            accepted_tokens.push(bonus_id);
        }
        profiler.end();
    } else {
        let verify_logits = verify_logits_ref
            .ok_or_else(|| Error::from_reason("MTP legacy accept requires verifier logits"))?;
        let mut hist_extended: Vec<u32> = token_history.to_vec();
        // Per-position path. Used for T>0 (where residual
        // sampling needs the full target distribution) and for
        // penalty-active configurations (where `hist_extended`
        // mutates the per-position logits inside the loop).
        // Note: this wrap includes the full-accept bonus-token sample
        // (sample + eval), whereas the sparse-accept branch's bonus is
        // a CPU buffer read inside the same phase name.
        profiler.begin("mtp_accept_loop");
        for i in 0..effective_depth {
            // verify_logits[0, i, :] → [vocab]
            let v_slice = verify_logits.slice(&[0, i as i64, 0], &[1, (i + 1) as i64, vocab])?;
            let v_logits_1d = v_slice.squeeze(Some(&[0, 1]))?;
            let penalized = apply_all_penalties(v_logits_1d, &hist_extended, params)?;
            // The target density `p` consumed by `accept_with_residual`
            // (`min(1, p/q)` + `(p - q)+` residual) MUST match the
            // distribution the verify/bonus token is drawn from. The
            // bonus on full-accept (and the residual draw on rejection) is
            // sampled via `sampling::sample(&penalized, ..)` →
            // `mlx_compiled_sample_full`, which filters logprobs then applies
            // temperature at the categorical draw. A raw `softmax(penalized)`
            // (no temperature, no top_k/top_p/min_p) did NOT match that draw,
            // biasing accept/reject and the residual resample whenever
            // temperature != 1 and/or filters are active. Build `p` from the
            // SAME compiled filter chain via `sampling::sampling_distribution`.
            //
            // At T=0 `accept_with_residual` only reads `argmax(p_target)`;
            // `sampling_distribution` returns the one-hot argmax there, so the
            // argmax (and thus the T=0 commit decision) matches a plain
            // `softmax` of the same logits while never erroring at T=0.
            let p_target = sampling::sampling_distribution(&penalized, params.sampling_config)?
                .astype(DType::Float32)?;
            p_target.eval();

            let sampling_cfg = params.sampling_config.unwrap_or_default();
            let (accept, out_tok) = sampling::accept_with_residual(
                &p_target,
                &draft_probs[i],
                draft_ids[i],
                &sampling_cfg,
                rng,
            )?;
            if trace_acceptance
                && let Err(e) = trace_acceptance_dense(
                    effective_depth,
                    i,
                    token_history.len(),
                    last_committed_id,
                    draft_ids[i],
                    &p_target,
                    &draft_probs[i],
                    &sampling_cfg,
                    accept,
                    out_tok,
                )
            {
                trace_acceptance_emit(serde_json::json!({
                    "schema_version": 1,
                    "path": "legacy_dense",
                    "depth": effective_depth,
                    "requested_depth": depth,
                    "slot": i,
                    "position": token_history.len() + i,
                    "last_committed_id": last_committed_id,
                    "draft_id": draft_ids[i],
                    "accepted": accept,
                    "out_token": out_tok,
                    "error": e.reason,
                }));
            }
            tracing::trace!(
                target: "mlx_core::mtp::accept",
                pos = i,
                draft_id = draft_ids[i],
                out_tok,
                accepted = accept,
                "MTP legacy accept position"
            );
            if accept {
                let id_u = out_tok as u32;
                accepted_tokens.push(id_u);
                hist_extended.push(id_u);
            } else {
                all_accepted = false;
                rejection_residual = Some(out_tok);
                accepted_tokens.push(out_tok as u32);
                break;
            }
        }

        if all_accepted {
            // Step 4 (bonus): sample from verify position D (after all
            // drafts accepted). Apply penalties consistent with the
            // extended history.
            let i = effective_depth;
            let v_slice = verify_logits.slice(&[0, i as i64, 0], &[1, (i + 1) as i64, vocab])?;
            let v_logits_1d = v_slice.squeeze(Some(&[0, 1]))?;
            let penalized = apply_all_penalties(v_logits_1d, &hist_extended, params)?;
            let bonus = sampling::sample(&penalized, params.sampling_config)?;
            bonus.eval();
            let bonus_id = bonus.item_at_int32(0)? as u32;
            tracing::trace!(
                target: "mlx_core::mtp::accept",
                bonus_id,
                "MTP bonus token (full accept, legacy path)"
            );
            accepted_tokens.push(bonus_id);
        }
        profiler.end();
    }

    // Diagnostic — `MLX_MTP_TRACE_LOGITS=1` per-committed-token verify
    // top-2 logit trace. Runs AFTER the accept loop so it is read-only
    // and does not perturb the sparse/per-position accept hot path. Each
    // `accepted_tokens[j]` was committed from verify slot `j` of the
    // batched verify forward; `verify_logits` is `[1, depth+1, vocab]`.
    // The first `K` slots are accepted drafts; the final slot is the
    // boundary token (bonus on full accept, residual on rejection).
    // Position label `token_history.len() + j` aligns with the AR
    // loop's `$hist.len() + 1` numbering (same prompt base).
    if mtp_trace_logits() {
        let verify_logits = verify_logits_ref
            .ok_or_else(|| Error::from_reason("MTP_TRACE_LOGITS requires verifier logits"))?;
        for (j, &committed_id) in accepted_tokens.iter().enumerate() {
            let slot = j as i64;
            let source = if all_accepted && j + 1 == accepted_tokens.len() {
                "verify-bonus"
            } else if !all_accepted && j + 1 == accepted_tokens.len() {
                "verify-residual"
            } else {
                "verify-draft"
            };
            let v_slice_res = verify_logits
                .slice(&[0, slot, 0], &[1, slot + 1, vocab])
                .and_then(|s| s.squeeze(Some(&[0, 1])));
            match v_slice_res.and_then(|v1d| trace_top2(&v1d, vocab)) {
                Ok(t2) => {
                    eprintln!(
                        "MTP_TRACE_LOGITS source={} verify_slot={} pos={} \
                         token_id={} top1_id={} top1_logit={:.6} top2_id={} \
                         top2_logit={:.6} gap={:.6}",
                        source,
                        j,
                        token_history.len() + j,
                        committed_id,
                        t2.top1_id,
                        t2.top1_logit,
                        t2.top2_id,
                        t2.top2_logit,
                        t2.top1_logit - t2.top2_logit,
                    );
                }
                Err(e) => {
                    eprintln!(
                        "MTP_TRACE_LOGITS source={} verify_slot={} pos={} ERROR {}",
                        source,
                        j,
                        token_history.len() + j,
                        e.reason,
                    );
                }
            }
        }
    }

    // Step 5: rollback. `accepted_drafts` is the number of draft
    // tokens (out of `effective_depth`) whose K/V we are KEEPING in BOTH the
    // main and the MTP draft caches. The rest must be discarded.
    //
    // Layout BEFORE this cycle (right after the macro's Step A):
    //   - Main offset advanced by 1 (Step A wrote K/V for `y`, the
    //     prior cycle's last accepted token, at the next free slot).
    //   - MTP draft offset unchanged since the prior cycle's
    //     rollback (the MTP path mirrors a snapshot of the main
    //     offset and only moves on draft / rollback).
    //
    // Verify wrote K/V for ALL `effective_depth + 1` inputs of
    // `[last_committed_id, d_0, .., d_{effective_depth-1}]` into the
    // MAIN cache (advancing main offset by `effective_depth + 1`). Draft
    // steps wrote K/V for the `effective_depth` drafted tokens into the
    // MTP cache (advancing MTP offset by `effective_depth`).
    //
    //   - On full accept: ALL `effective_depth + 1` verify positions are kept
    //     in main (last_committed + `effective_depth` drafts) and ALL `effective_depth`
    //     draft positions are kept in MTP. The bonus token has no
    //     K/V written this cycle — its K/V will be laid down by the
    //     NEXT cycle's Step A.
    //   - On rejection after `K` accepted drafts: we keep the
    //     last_committed slot + the first `K` draft slots in main
    //     (= `K + 1` main verify slots) and the first `K` slots in
    //     MTP. The REJECTED draft's K/V is discarded by offset
    //     rewind in BOTH caches. The verifier's residual sample is
    //     emitted as a token but has no K/V written this cycle —
    //     its K/V will be laid down by the NEXT cycle's Step A.
    //
    // Both deltas reduce to `accepted_drafts - effective_depth`:
    //   - main_delta = (K + 1) - (effective_depth + 1) = K - effective_depth
    //   - mtp_delta  = K       - effective_depth
    let accepted_drafts = if all_accepted {
        effective_depth
    } else {
        // accepted_tokens contains `K` accepted drafts + 1 residual.
        accepted_tokens.len() - 1
    };
    if let Some(policy) = ev_depth_policy.as_mut() {
        policy.observe(effective_depth, accepted_drafts);
    }
    // Per-cycle acceptance: feeds the profiler's acceptance summary
    // (surfaced on `PerformanceMetrics` + the stderr report).
    profiler.record_mtp_cycle(effective_depth, accepted_drafts);
    tracing::debug!(
        target: "mlx_core::mtp",
        depth = effective_depth,
        requested_depth = depth,
        accepted_drafts,
        all_accepted,
        committed = accepted_tokens.len(),
        "MTP cycle accept result"
    );

    // Committed-history commit.
    //
    // Step-A cycles commit the full newly emitted sequence
    // `[last_committed_id] ++ accepted_tokens`: Step A sampled
    // `last_committed_id`, so it is not in the persistent MTP cache yet.
    //
    // Chained cycles skip Step A. Their `last_committed_id` is the prior
    // cycle's boundary token, already committed by that prior cycle. The
    // commit must therefore skip the anchor and append only
    // `accepted_tokens`, advancing the stepper's committed length by the
    // number of
    // newly emitted tokens. Re-committing the anchor would drift the MTP
    // RoPE base by one slot per chained cycle.
    let committed_ids: Vec<u32> = match commit_anchor {
        MtpCommitAnchor::IncludeAnchor => {
            let mut ids = Vec::with_capacity(accepted_tokens.len() + 1);
            ids.push(last_committed_id);
            ids.extend(accepted_tokens.iter().copied());
            ids
        }
        MtpCommitAnchor::SkipAlreadyCommittedAnchor => accepted_tokens.clone(),
    };
    profiler.begin("mtp_commit");
    let commit_res = step.commit_mtp(
        commit_anchor,
        &commit_seed_hidden,
        &verify_hiddens,
        &committed_ids,
        accepted_drafts,
        embedding_weight,
    );
    profiler.end();
    commit_res?;

    profiler.begin("mtp_rollback");
    step.rollback(accepted_drafts, effective_depth);
    profiler.end();
    tracing::debug!(
        target: "mlx_core::mtp",
        accepted_drafts,
        depth = effective_depth,
        requested_depth = depth,
        offset_delta = accepted_drafts as i64 - effective_depth as i64,
        "MTP rollback applied"
    );

    // On rejection, restore the main path's GDN linear caches (back to
    // "after Step A": Step A processed `y_N` and the snapshot was taken
    // right after) and replay the K + 1 committed tokens that verify
    // processed but the restore discarded:
    //   * `last_committed_id` (= y_{N+1}, the token Step A sampled
    //     and the cycle treated as the verify-position-0 anchor),
    //   * `d_0..d_{K-1}` (the K accepted drafts).
    // The residual sample R is NOT replayed — its K/V will be laid
    // down by the NEXT outer iteration's Step A (it becomes `y` at
    // the loop boundary).
    //
    // Post-replay main offset = snapshot_offset + K + 1, matching
    // what the previous direct `adjust_offset(K - depth)` rollback
    // produced. Post-replay linear state = AR equivalent for the
    // `[y_N, y_{N+1}, d_0..d_{K-1}]` token prefix.
    //
    // On full accept the rollback hook receives `(accepted_drafts=depth,
    // depth)` and may still normalize the main linear state from the
    // recorded tape. The verifier's full window is logically kept, but
    // the dense GDN recurrent cache must remain byte-compatible with
    // serial AR across the next Step A.
    if !all_accepted {
        let mut replay_ids: Vec<u32> = Vec::with_capacity(accepted_drafts + 1);
        replay_ids.push(last_committed_id);
        // accepted_tokens = [d_0, .., d_{K-1}, residual]; we replay
        // only the K accepted drafts (NOT the residual).
        replay_ids.extend_from_slice(&accepted_tokens[..accepted_drafts]);
        tracing::debug!(
            target: "mlx_core::mtp",
            replay_token_count = replay_ids.len(),
            last_committed_id,
            "MTP tape replay (restore main caches + replay accepted prefix)"
        );
        profiler.begin("mtp_tape_replay");
        let replay_res = step.restore_and_replay_main(&replay_ids, embedding_weight);
        profiler.end();
        replay_res?;
    }

    let _ = rejection_residual; // documented above; only used for clarity
    // `prev_hidden` / `prev_emb` are no longer needed (they were the
    // INPUTS to the cycle's drafts; the verify pass downstream of
    // them is already evaluated). They drop at end-of-function with
    // the rest of the locals; the underlying lazy MLX arrays stay
    // alive as long as any other handle still holds them.

    // Pick the position-K slice of `verify_hiddens` and return it so the
    // caller (the `decode_loop_mtp!` macro) can chain cycles: the NEXT
    // cycle's first MTP draft uses this hidden as `prev_hidden`,
    // eliminating the per-cycle main-model "Step A" forward.
    //
    // Semantics: `verify_hiddens[K]` is the post-final-norm hidden at
    // verify position K — the prediction context for the committed
    // token at position K+1 of `[last_committed, d_0, ..., d_{D-1}]`,
    // i.e. the BONUS token on full-accept (K=D, position K+1 = bonus's
    // would-be slot) or the RESIDUAL token on rejection (K<D, position
    // K+1 = rejected draft's slot, replaced by residual). Either way,
    // the next cycle's MTP draft gets `(prev_hidden=verify_hiddens[K],
    // prev_emb=embed(committed_K+1))` which matches the training
    // contract of the MTP head: `MTP(h_t, embed(t+1)) -> logits at
    // t+2`.
    //
    // Why K (not D, not D+1): position D only matches when ALL drafts
    // are accepted (K==D). Chaining a partial-accept cycle from
    // position D's hidden — the prediction context for the rejected
    // draft — diverges the MTP head's drafts from main, dropping mean
    // acceptance from ~1.5 to ~0.8 tokens/cycle.
    let hidden_dim = verify_hiddens.shape_at(2)?;
    let verify_hidden_k = verify_hiddens.slice(
        &[0, accepted_drafts as i64, 0],
        &[1, (accepted_drafts + 1) as i64, hidden_dim],
    )?;
    Ok((
        MtpCycleOutcome {
            tokens: accepted_tokens,
            requested_depth: depth,
            effective_depth,
        },
        verify_hidden_k,
    ))
}

/// Required arguments of [`run_mtp_turn`] — the MTP analog of
/// [`crate::engine::decode::DecodeLoopArgs`]. Every field is the macro
/// parameter of the SAME name in `decode_loop_mtp!`; the turn-constant
/// `embedding_weight` + the per-cycle scratch live on the
/// [`MtpStepper`] (captured at `begin_mtp_decode`), so they are NOT here.
///
/// `#[allow(dead_code)]`: SCAFFOLD — nothing in production calls
/// [`run_mtp_turn`] yet (the family steppers + the rewire land in a later
/// step), so the relocated loop is byte-identical dead code.
#[allow(dead_code)]
pub(crate) struct MtpTurnArgs<'a> {
    /// First generated token (sampled from the prefill logits BEFORE the
    /// turn). The loop takes ownership; its final reassignment is not
    /// observed by callers (== the macro's `y`).
    pub y: MxArray,
    /// Requested draft depth for this turn (`params.mtp_depth`) — the
    /// macro's `mtp_depth`. The stepper still applies its intra-cycle
    /// adaptive/EV gates on top.
    pub depth: usize,
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
    /// Prompt-prefix MTP seed inputs, forwarded verbatim into
    /// [`MtpTurnSetup`] so [`MtpBackend::begin_mtp_decode`] can commit the
    /// prompt prefix into the drafter's committed-history cache before the
    /// loop. == the `prompt_hidden` / `prompt_hidden_ids` /
    /// `prompt_hidden_position_base` fields of `ChatDecodeInputs` the eager
    /// block read. `None` / `0` ⇒ no prompt seed (cycle-history v1).
    pub prompt_hidden: Option<MxArray>,
    pub prompt_hidden_ids: Option<Vec<u32>>,
    pub prompt_hidden_position_base: usize,
}

/// Terminal outs of [`run_mtp_turn`] the caller threads into its save /
/// next-turn bookkeeping — the engine-owned surfaces of the two
/// side-channels the family code reads AFTER the `decode_loop_mtp!` macro
/// today (`last_in_cache` and the `mtp_desynced` cell).
///
/// `#[allow(dead_code)]`: SCAFFOLD — produced only by the dead
/// [`run_mtp_turn`] / the module's mock tests until the family rewire.
#[allow(dead_code)]
pub(crate) struct MtpTurnOutcome {
    /// Whether the LAST emitted token's K/V is already in the physical
    /// cache. The save uses `drop_last_always = !last_in_cache` (the
    /// 90128bfd unforwarded-boundary rule). == the macro's
    /// `last_in_cache` ident at loop exit.
    pub last_in_cache: bool,
    /// Whether a mid-cycle stop left the FLAT caches desynced
    /// (`rollback_unemitted` with `unemitted > 0`). Flat / MoE set it;
    /// the paged stepper's [`MtpStepper::into_desynced`] MUST return
    /// `false`. The caller propagates it into
    /// `self.flat_mtp_caches_desynced` exactly as the post-macro code does.
    pub desynced: bool,
}

/// Engine-owned MTP propose/verify whole-turn loop — the relocated SYNC
/// (non-streaming) body of `decode_loop_mtp!`, generic over
/// [`MtpBackend`]. The MTP analog of
/// [`crate::engine::decode::run_decode_loop`].
///
/// Calls [`MtpBackend::begin_mtp_decode`] to build the per-turn stepper,
/// then drives the relocated outer loop: the initial-`y` emit, Step A vs.
/// chained-hidden routing, [`run_mtp_cycle`] per cycle, the per-token emit
/// loop, and the mid-cycle-stop `rollback_unemitted`. Returns the
/// [`MtpTurnOutcome`] (`last_in_cache` + `desynced`) AND surfaces any
/// full-accept GDN tape-replay error stashed by the infallible
/// [`MtpStepper::rollback`] via [`MtpStepper::take_replay_error`] after the
/// loop — the two side-channels the family code reads after the macro
/// today.
///
/// VERBATIM, mechanical relocation of the macro's SYNC arm: every
/// surrounding line (the initial-`y` emit, the `max_as_usize` negative
/// clamp, all stop checks, the `do_step_a` routing, the chained_hidden
/// stash/drain, the adaptive/EV depth pick + near-tail cap, the per-token
/// emit loop with `observe`/force-end + every-256-token cache clear, the
/// `last_in_cache` bookkeeping at the 3 break sites, and the
/// `eval_step_with_chained_hidden` fused chained eval at the iteration
/// boundary) is byte-for-byte identical in logic and ORDER, swapping the
/// macro's `$mtp.<closure>` for `step.<method>` and
/// `run_mtp_cycle_inner(&mut $mtp, ..)` for `run_mtp_cycle(step, ..)`.
///
/// The macro→method swaps (the ONLY substantive change vs the original
/// body):
///   * `($mtp.forward_with_hidden)(ids, emb)` → `step.forward_with_hidden(ids, emb)`
///   * `($mtp.eval_step)(t, l, b)` → `step.eval_step(t, l, b)`
///   * `($mtp.begin_cycle)(c)` → `step.begin_cycle(c)`
///   * `$mtp.committed_history_active` (field) → `step.committed_history_active()`
///   * `($mtp.eval_step_with_chained_hidden)(y, h)` → `step.eval_step_with_chained_hidden(y, h)`
///   * `($mtp.rollback_unemitted)(u)` → `step.rollback_unemitted(u)`
///   * `run_mtp_cycle_inner(&mut $mtp, ..)` → `run_mtp_cycle(step, ..)`
///
/// The post-loop `replay_err_cell` / `mtp_desynced` reads become the
/// engine-owned `step.take_replay_error()?` / `step.into_desynced()` outs.
///
/// The optional `streaming` arm is the engine-owned analog of
/// `decode_loop_mtp!`'s `$(, streaming: {...})?`: the relocated
/// per-token streaming emit points at the SAME three sites the sync path
/// pushes (the initial-`y` push, Step A's sampled token, and the cycle
/// emit loop) plus the pre-loop cancellation break, routed through the
/// shared [`StreamingCtx`] / [`crate::engine::backend::StreamEmitter`] /
/// [`crate::engine::backend::ChunkSink`] abstraction — the SAME emitter
/// type / incremental detokenization (`step_decode_stream`) /
/// reasoning-suppression gate `run_decode_loop` uses. `None` ⇒ the SYNC
/// (non-streaming) path; both share ONE loop with a sink switch, so the
/// sync path is byte-identical with or without the arm wired.
#[allow(dead_code)]
pub(crate) fn run_mtp_turn<B: MtpBackend, R: rand::Rng>(
    backend: &mut B,
    rng: &mut R,
    args: MtpTurnArgs<'_>,
    mut streaming: Option<StreamingCtx<'_, '_>>,
) -> Result<MtpTurnOutcome> {
    let MtpTurnArgs {
        mut y,
        depth,
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
        prompt_hidden,
        prompt_hidden_ids,
        prompt_hidden_position_base,
    } = args;

    // Materialize the first sampled token's id before building the setup.
    // The eager block's prompt seed read `y.item_at_int32(0)` after a
    // `y.eval()` to append `y` to the committed run `[prompt_ids[1..], y]`;
    // `begin_mtp_decode` now owns that seed, so the engine evals `y` once
    // here and hands the id through the setup. `y.eval()` is idempotent —
    // the loop's initial-emit re-evals the same materialized value, so the
    // sampled token (and every downstream commit) is byte-identical.
    y.eval();
    let first_sampled_token = y.item_at_int32(0)? as u32;

    // The turn-constant embedding weight + the requested depth + the
    // per-cycle scratch (and the prompt-prefix seed) are captured into the
    // stepper at `begin_mtp_decode` (the analog of `begin_paged_decode`). The
    // macro threaded `embedding_weight` as `$emb`; the stepper now owns it and
    // exposes it via `embedding_weight()`. Read once at turn entry.
    let setup = MtpTurnSetup {
        params: p,
        is_delta: false,
        depth,
        prompt_hidden: prompt_hidden.as_ref(),
        prompt_hidden_ids: prompt_hidden_ids.as_deref(),
        prompt_hidden_position_base,
        first_sampled_token,
    };
    let mut step = backend.begin_mtp_decode(&setup)?;
    // Turn-entry reads the engine takes over from the macro/family wiring:
    // the profiler relabel (mirrors `DecodeStep::profiler_relabel`) and the
    // owned embedding-weight handle the macro passed as `$emb`.
    if let Some(label) = step.profiler_relabel() {
        profiler.set_label(label);
    }
    let emb = step.embedding_weight().clone();

    // `last_in_cache` is owned by the loop here (the macro mutated the
    // caller's `$last_in_cache` ident in place) and returned in the
    // outcome. Default `true`: a clean length/EOS exit on a forwarded
    // token keeps the boundary token in cache.
    let mut last_in_cache = true;

    // Emit the FIRST token via a normal main-path forward+hidden.
    // The MTP loop needs an established last-committed token AND
    // its post-final-norm hidden state to seed the first draft.
    // After this initial forward, `prev_hidden` / `prev_emb`
    // carry the seed for the next cycle.
    let mut prev_hidden_opt: Option<MxArray>;
    let mut prev_emb_opt: Option<MxArray>;
    let mut last_committed_id_opt: Option<u32>;

    // Chained-cycle state. `run_mtp_cycle` slices
    // `verify_hiddens[:, K, :]` and returns it; we stash that
    // `[1, 1, hidden]` here so the NEXT outer iteration can skip
    // Step A's ~150 ms main-model forward and feed the chained
    // hidden directly into the cycle's first MTP draft.
    //
    // K = number of accepted drafts this cycle. Semantics:
    // `verify_hiddens[K]` is the prediction context for the
    // committed token at position K+1 (bonus on full-accept,
    // residual on rejection) — i.e. for the LAST emitted token of
    // this cycle. The next cycle's MTP draft is therefore
    // `MTP(prev_hidden=verify_hiddens[K], prev_emb=embed(y)) ->
    // next-next logits`, matching the head's training contract.
    //
    // Chaining is GPU-gen-gated (default ON M5+, OFF M1–M4); override
    // with `MLX_MTP_CHAINED_CYCLES=0/1`. The position-K slice makes it
    // SEMANTICALLY correct — byte-exact T=0 parity holds in both modes.
    //
    // Invariants:
    //   - `None` on the FIRST iteration (no prior verify) — Step A
    //     runs unconditionally and re-seeds the hidden from a real
    //     main forward.
    //   - `None` when forced-think-end fires — that path needs Step
    //     A's forward to write `y`'s K/V before injecting the
    //     forced token. (See the force-end branch below.)
    //   - `Some(hidden)` after every successful cycle, to be drained
    //     by the NEXT iteration before its cycle runs.
    //
    // The hidden is a lazy MLX array referencing the verify's
    // position-K `final_norm` graph node; the eager verify step
    // returns it alongside the verify logits, and it stays alive
    // for the rest of the decode loop as long as the cycle holds it.
    let chained_cycles_enabled: bool =
        crate::models::qwen3_5::mtp_decode::mtp_chained_cycles_enabled();
    let mut chained_hidden_opt: Option<MxArray> = None;

    // Adaptive MTP depth policy. When `mtp_adaptive_depth` is true
    // (explicit opt-in from ChatConfig), the policy picks the
    // per-cycle draft depth from a per-depth EMA of
    // `accepted_tokens / cycle_wall_ns` plus a DFlash-style 3-state
    // machine (`full | reduced | probe`). When false, the policy is
    // constructed but `pick_depth()` returns `p.mtp_depth` on every
    // call (no transitions ever fire because `record_cycle` is gated
    // below).
    //
    // The eager MTP verify forward builds its graph per cycle from
    // the live depth, so swinging the depth freely between cycles
    // carries no extra setup cost.
    let mut mtp_depth_policy = crate::models::qwen3_5::adaptive_depth::AdaptiveDepthPolicy::new(
        depth.min(u8::MAX as usize) as u8,
    );
    let mtp_adaptive_depth_mode =
        crate::models::qwen3_5::adaptive_depth::adaptive_depth_mode_from_env();
    let mut mtp_ev_depth_policy =
        crate::models::qwen3_5::adaptive_depth::ExpectedValueDepthPolicy::new(
            depth.min(u8::MAX as usize) as u8,
        );

    // Track cycles for the every-256-emitted-token cache clear.
    // We use the running `generated.len()` rather than a separate step
    // counter so MTP and non-MTP loops stay byte-equivalent on
    // the cache-clear cadence.
    let mut last_clear_at: usize = generated.len();

    // PARITY-FIX (budget): `max` is the raw `max_new_tokens: i32`
    // with no upstream clamp on the chat/MTP path, so it can be `0`
    // or NEGATIVE (reachable via `ChatConfig.maxNewTokens` and
    // `/v1/responses` `max_output_tokens`). AR's `decode_loop!` uses
    // `for step in 0..max` — an empty range for `max <= 0` — and
    // therefore emits 0 tokens. The MTP loop below compares
    // `generated.len()` against the budget via `as usize`; a NEGATIVE
    // `max` would wrap to a huge `usize` and never trip the length
    // cap (effectively unbounded). Clamp negatives to 0 ONCE here
    // and use this value for every budget comparison so MTP matches
    // AR's "0 new tokens for a nonpositive budget" semantics. For
    // `max >= 1` this is numerically identical to `(max as usize)`
    // ⇒ byte-for-byte identical behavior for valid budgets.
    let max_as_usize: usize = (max).max(0) as usize;

    // PARITY-FIX: emit the initial `y` (sampled from the prefill's
    // last logits BEFORE this loop was entered) before Step A's
    // first iteration. AR's `decode_loop!` macro emits its input
    // `y` at the top of each iteration; MTP's Step A only emits
    // the SAMPLED next token, which means the very first token of
    // the generation (the prefill's seed sample) never reached
    // `gen`. Without this push MTP's output is the AR output
    // shifted left by one token. We mirror the per-token bookkeeping
    // Step A does (eval, tracker.observe_token, profiler) so the
    // initial token participates identically. The stop checks (EOS,
    // length, cancel, repetition) run at the top of the loop body
    // below — they read `gen` so the initial push is visible.
    //
    // Guarded on the budget: at entry `gen` holds only
    // generated tokens (0 here), so `generated.len() < max_as_usize` is
    // `0 < 0 == false` when `max <= 0` ⇒ NO initial push, matching
    // AR. For `max >= 1` the guard is `0 < max` (true) ⇒ the push
    // runs exactly as before.
    if generated.len() < max_as_usize {
        let _stream_ctx = crate::stream::StreamContext::new(generation_stream);
        profiler.begin("extract");
        y.eval();
        let initial_token_id = y.item_at_int32(0)? as u32;
        profiler.end();
        profiler.mark_first_token();
        if report && first_tok.is_none() {
            *first_tok = Some(std::time::Instant::now());
        }
        generated.push(initial_token_id);
        hist.push(initial_token_id);
        let _is_reasoning = tracker.observe_token(initial_token_id);
        // Streaming-only — relocated VERBATIM from `decode_loop_mtp!`'s
        // initial-`$y` arm. The initial seed's cancel-check just SKIPS the
        // detok+emit (NO break, unlike Step A / the emit loop): a cancel
        // before the first forward leaves the seed committed but unstreamed.
        // Detokenize + length-advance stay OUTSIDE the emitter's gate so
        // DecodeStream sees every token (matching `run_decode_loop`).
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
    }

    loop {
        // Zero budget (nonpositive clamped to 0): AR's `for step in 0..max`
        // never iterates and never observes cancel/EOS/repetition, so its
        // finish_reason stays "length". `max_as_usize` is loop-invariant, so
        // for any budget >= 1 this is a dead branch (no behavior change);
        // only a 0 budget short-circuits here, before the cancelled check.
        if max_as_usize == 0 {
            if reason.is_empty() {
                *reason = String::from("length");
            }
            break;
        }
        // PARITY-FIX: re-check the same stop conditions Step A
        // uses, BEFORE the forward, so the initial push (above)
        // and any prior-iteration push that landed us on a stop
        // condition exit cleanly without one more forward.
        if let Some(&last) = generated.last()
            && (last == eos_id || p.extra_eos_ids.contains(&last))
        {
            *reason = String::from("stop");
            // This pre-forward re-check fires on the initial seed (or a
            // prior-iteration token) BEFORE any forward consumed it, so
            // the stop token is not yet in the physical cache.
            last_in_cache = false;
            break;
        }
        // Streaming-only pre-loop cancel check — relocated VERBATIM from
        // `decode_loop_mtp!` (it sits between the EOS pre-check and the
        // repetition pre-check). A cancel observed at the iteration top
        // exits "cancelled" before any forward; the last emitted token is
        // the unforwarded seed/boundary, so it is not yet in the cache.
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

        // ---- Step A vs. chained-hidden decision. ------------------
        // When chained cycles are enabled (`chained_cycles_enabled`,
        // GPU-generation-gated: default ON on M5+/gen>=17, OFF on
        // M1–M4; override via `MLX_MTP_CHAINED_CYCLES=0/1`): skip
        // Step A's full main-model forward when a chained verify
        // hidden is available from the prior cycle, unless the
        // tracker is about to force a think-end token (the
        // forced-token path needs Step A to forward `y` so its K/V
        // is committed before we inject the forced token). The gate
        // is a non-consuming `force_think_end_pending()` peek so the
        // pending flag survives into Step A's single consume below.
        //
        // When chained cycles are disabled (M1–M4 default, or
        // `MLX_MTP_CHAINED_CYCLES=0`): always Step A, byte-exact with
        // the non-chained path. On M1–M4 the chained path still
        // regresses depth-3 acceptance (a lazy-slice eval-scheduling
        // stall), see the comment block at the top of
        // `decode_loop_mtp!` for details.
        //
        // On the chained path the prior cycle's verify already
        // committed all accepted tokens' K/V, and the next cycle's
        // verify will write `y`'s K/V at its position-0 input.
        // The MTP draft seeds from `chained_hidden_opt`
        // (`verify_hiddens[K]` — the prediction context for the
        // committed token at position K+1, i.e. y itself). T=0
        // parity is preserved because verify (= main model) is the
        // ground truth and at T=0 the residual-sampler picks the
        // same token regardless of draft accuracy.
        let do_step_a = !chained_cycles_enabled
            || chained_hidden_opt.is_none()
            || tracker.force_think_end_pending();
        let cycle_seed_was_chained = !do_step_a;

        let _stream_ctx = crate::stream::StreamContext::new(generation_stream);

        if do_step_a {
            profiler.begin("forward");
            let next_ids = y.reshape(&[1, 1])?;
            let (mut logits, hidden, needs_squeeze) = step.forward_with_hidden(&next_ids, &emb)?;
            if needs_squeeze {
                logits = logits.squeeze(Some(&[1]))?;
            }
            profiler.end();

            let (next_token, budget_forced) = if tracker.should_force_think_end() {
                let forced_id = tracker.forced_token_id()? as i32;
                tracing::debug!(
                    target: "mlx_core::mtp",
                    forced_id,
                    "MTP Step A: forcing think-end token (reasoning budget tripped)"
                );
                (MxArray::from_int32(&[forced_id], &[1])?, true)
            } else {
                profiler.begin("rep_penalty");
                logits = apply_all_penalties(logits, hist, p)?;
                profiler.end();

                profiler.begin("sample");
                let t = crate::sampling::sample(&logits, p.sampling_config)?;
                profiler.end();
                (t, false)
            };

            profiler.begin("eval_caches");
            step.eval_step(&next_token, &logits, budget_forced);
            profiler.end();

            profiler.begin("eval_token");
            next_token.eval();
            profiler.end();

            profiler.begin("extract");
            let token_id = next_token.item_at_int32(0)? as u32;
            profiler.end();
            profiler.mark_first_token();
            if report && first_tok.is_none() {
                *first_tok = Some(std::time::Instant::now());
            }

            generated.push(token_id);
            hist.push(token_id);
            let _is_reasoning = tracker.observe_token(token_id);

            // Streaming-only — relocated VERBATIM from `decode_loop_mtp!`'s
            // Step A arm. It runs AFTER `observe_token` and BEFORE the EOS
            // check (the macro's order): a cancel observed here breaks
            // "cancelled" (the just-committed token is unforwarded, so
            // `last_in_cache = false`); otherwise detokenize + length-advance
            // (outside the emitter's gate) then emit through the emitter.
            if let Some(s) = streaming.as_mut() {
                *s.last_is_reasoning = _is_reasoning;
                if s.cancelled.load(Ordering::Relaxed) {
                    *reason = String::from("cancelled");
                    last_in_cache = false;
                    break;
                }
                let token_text = crate::tokenizer::Qwen3Tokenizer::step_decode_stream(
                    s.decode_stream,
                    s.tokenizer,
                    token_id,
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

            if token_id == eos_id || p.extra_eos_ids.contains(&token_id) {
                *reason = String::from("stop");
                // Step A's sampled token only becomes the next `y` and is
                // forwarded on the NEXT iteration, so it is not yet in the
                // physical cache when we stop here.
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

            // Seed for MTP cycles using the hidden returned from this
            // forward. `hidden` is `[1, hidden_size]`; reshape to
            // `[1, 1, hidden]` for the draft FFI's `[B, T, hidden]`
            // contract.
            let hidden_dim = hidden.shape_at(1)?;
            prev_hidden_opt = Some(hidden.reshape(&[1, 1, hidden_dim])?);
            // prev_emb is the embedding of the JUST-emitted token.
            let id_arr = MxArray::from_int32(&[token_id as i32], &[1])?;
            let emb_2d = emb.take(&id_arr, 0)?;
            let h = emb_2d.shape_at(1)?;
            prev_emb_opt = Some(emb_2d.reshape(&[1, 1, h])?);
            last_committed_id_opt = Some(token_id);
            y = next_token;
        } else {
            // ---- Chained path: skip Step A entirely. --------------
            // `y` already holds the prior cycle's last accepted
            // token (set by that cycle's tail update). That token
            // has already been pushed to `gen` / `hist` /
            // `tracker` AND streamed to the callback. Its K/V will
            // be written by THIS cycle's verify at position 0.
            //
            // We just need to seed the cycle's MTP draft inputs
            // from the chained hidden and the embedding of `y`.
            let chained_h = chained_hidden_opt.take().ok_or_else(|| {
                napi::Error::from_reason(
                    "chained_hidden_opt is Some on the chained path \
                     (guarded by do_step_a)",
                )
            })?;
            // `run_mtp_cycle` already sliced the K-th
            // position out of the verify hiddens, so `chained_h`
            // arrives shaped `[1, 1, hidden]` — the same shape the
            // draft FFI's `[B, T, hidden]` contract expects, no
            // reshape needed.
            prev_hidden_opt = Some(chained_h);

            // Read `y`'s id without re-evaluating it; the prior
            // cycle tail already ran `MxArray::from_int32(...)` to
            // produce a fully materialised `[1]` int32 array, so
            // `item_at_int32(0)` here is a CPU-only read.
            y.eval();
            let token_id = y.item_at_int32(0)? as u32;

            let id_arr = MxArray::from_int32(&[token_id as i32], &[1])?;
            let emb_2d = emb.take(&id_arr, 0)?;
            let h = emb_2d.shape_at(1)?;
            prev_emb_opt = Some(emb_2d.reshape(&[1, 1, h])?);
            last_committed_id_opt = Some(token_id);
            // Note: no `y =` assignment — `y` is already correct.
            // No tracker.observe_token / no generated.push / no callback —
            // the prior cycle's emit loop already handled all of
            // that for the same `token_id`.
        }
        profiler.step();

        // ---- Step B: ONE MTP draft+verify cycle. -------------------
        // On the chained path the prior verify already committed
        // bonus/residual; this cycle's verify writes the chained
        // `y`'s K/V at position 0 and extends the prefix by D more
        // drafts. On full accept per cycle we emit D+1 tokens for
        // D draft steps + 1 verify (one fewer main forward when
        // chaining).
        if generated.len() >= max_as_usize {
            if reason.is_empty() {
                *reason = String::from("length");
            }
            break;
        }
        if tracker.force_think_end_pending() {
            // Budget tripped during Step A's observe (after Step A's
            // consume) — defer the forced token to the NEXT cycle's
            // Step A. This is a NON-consuming peek: the flag stays set,
            // so next cycle's routing peek (`do_step_a`) forces Step A
            // and the single consuming call there emits `</think>`.
            tracing::debug!(
                target: "mlx_core::mtp",
                "MTP cycle skipped: think-end queued, deferring to next Step A"
            );
            continue;
        }

        let prev_h = prev_hidden_opt.take().ok_or_else(|| {
            napi::Error::from_reason("prev_hidden seeded by Step A or chained path")
        })?;
        let prev_e = prev_emb_opt
            .take()
            .ok_or_else(|| napi::Error::from_reason("prev_emb seeded by Step A or chained path"))?;
        let last_id = last_committed_id_opt.ok_or_else(|| {
            napi::Error::from_reason("last_committed seeded by Step A or chained path")
        })?;
        // Re-anchor the MTP cache to the main path's CURRENT offset
        // before launching this cycle's drafts. On the Step-A path
        // the main offset has
        // advanced by 1 (Step A's forward) + the prior cycle's
        // verify advancement. On the chained path the main offset
        // has only advanced by the prior cycle's verify (Step A
        // was skipped). EITHER way, this resets the MTP K/V and
        // sets the MTP offset = current main offset, which is
        // exactly the contract `begin_cycle` is documented to
        // honour. Without it the MTP draft RoPE positions diverge
        // and drafts produce gibberish.
        // The `begin_cycle` hook emits its own
        // `mlx_core::mtp` trace (old/new MTP offset) — it is the
        // only site that knows the dense-vs-MoE offset getters.
        step.begin_cycle(cycle_seed_was_chained && step.committed_history_active());
        // Per-cycle depth selection. When adaptive is OFF,
        // `pick_depth()` returns the seed depth unchanged
        // (`record_cycle` is gated below). When adaptive is ON, the
        // policy hill-climbs across depth-EMA + manages the
        // `full | reduced | probe` state machine.
        let cycle_depth: usize = if p.mtp_adaptive_depth {
            match mtp_adaptive_depth_mode {
                crate::models::qwen3_5::adaptive_depth::AdaptiveDepthMode::Throughput => {
                    mtp_depth_policy.pick_depth() as usize
                }
                crate::models::qwen3_5::adaptive_depth::AdaptiveDepthMode::ExpectedValue => {
                    mtp_ev_depth_policy.max_depth() as usize
                }
            }
        } else {
            depth
        };
        // Near-tail budget cap. The verify writes `depth+1`
        // target-cache slots BEFORE the post-verify truncation to
        // `max_new_tokens`; when fewer than `depth+1` main-cache
        // slots remain near the tail the write can overrun the
        // rounded `max_kv_len` allocation. Cap the effective cycle
        // depth so the verify never needs more main-cache slots than
        // the remaining generation budget can absorb. `remaining` is
        // `>= 1` here (the `generated.len() >= max` check above already
        // broke the loop otherwise). With `effective_depth =
        // remaining - 1` the verify writes exactly `remaining`
        // slots and the cycle emits at most `remaining` tokens.
        let remaining: usize = max_as_usize.saturating_sub(generated.len());
        let cycle_depth: usize = cycle_depth.min(remaining.saturating_sub(1));
        if cycle_depth < 1 {
            // Only 1 token of budget left — an MTP cycle would
            // draft+verify more tokens than can be emitted. Fall
            // back to single-token AR decode: skip this cycle and
            // let the next iteration's Step A emit the final token
            // (its post-emit `generated.len() >= max` check then breaks
            // the loop with reason "length"). `chained_hidden_opt`
            // is still `None` here, so Step A runs unconditionally.
            tracing::debug!(
                target: "mlx_core::mtp",
                remaining,
                "MTP cycle skipped near tail: AR-decoding the final token(s)"
            );
            continue;
        }
        profiler.begin("mtp_cycle");
        let cycle_started_at = std::time::Instant::now();
        let commit_anchor = if cycle_seed_was_chained && step.committed_history_active() {
            MtpCommitAnchor::SkipAlreadyCommittedAnchor
        } else {
            MtpCommitAnchor::IncludeAnchor
        };
        let ev_depth_policy = if p.mtp_adaptive_depth
            && matches!(
                mtp_adaptive_depth_mode,
                crate::models::qwen3_5::adaptive_depth::AdaptiveDepthMode::ExpectedValue
            ) {
            Some(&mut mtp_ev_depth_policy)
        } else {
            None
        };
        let cycle_res = run_mtp_cycle(
            &mut step,
            prev_h,
            prev_e,
            last_id,
            &emb,
            hist,
            p,
            rng,
            profiler,
            cycle_depth,
            ev_depth_policy,
            commit_anchor,
        );
        profiler.end();
        // `run_mtp_cycle` returns the verify-final hidden so
        // the NEXT outer iteration can skip Step A's ~150 ms
        // main-model forward. We stash it into `chained_hidden_opt`;
        // the iteration boundary's `do_step_a` check will drain it.
        let (outcome, verify_last_hidden) = cycle_res?;
        chained_hidden_opt = Some(verify_last_hidden);

        // Throttled per-cycle MTP trace. Mirrors the AR loop's
        // every-32-steps cadence in token-count units so MTP and
        // AR runs leave comparable breadcrumb density.
        if (generated.len() / 32) != ((generated.len() + outcome.tokens.len()) / 32) {
            let first_tok_id = outcome.tokens.first().copied().unwrap_or(0);
            tracing::info!(
                "Qwen3.5 decode MTP cycle gen_len={} depth={} committed={} \
                 first_tok={}",
                generated.len(),
                outcome.effective_depth,
                outcome.tokens.len(),
                first_tok_id,
            );
        }
        // Feed observation to the policy AFTER the cycle's tokens
        // have been counted but BEFORE the emit loop's stop checks
        // (so the record always runs even on partial-emit due to
        // EOS / length / cancel). `committed` is the number
        // of tokens the cycle actually produced (drafts accepted +
        // residual/bonus); range `[1, depth+1]`.
        let cycle_wall_ns: u64 = cycle_started_at
            .elapsed()
            .as_nanos()
            .min(u128::from(u64::MAX)) as u64;
        let cycle_committed: u32 = outcome.tokens.len() as u32;
        if p.mtp_adaptive_depth
            && matches!(
                mtp_adaptive_depth_mode,
                crate::models::qwen3_5::adaptive_depth::AdaptiveDepthMode::Throughput
            )
        {
            mtp_depth_policy.record_cycle(crate::models::qwen3_5::adaptive_depth::CycleStats {
                depth: outcome.effective_depth as u8,
                committed: cycle_committed,
                wall_ns: cycle_wall_ns,
            });
            tracing::debug!(
                target: "mlx_core::mtp::adaptive",
                state = mtp_depth_policy.state_label(),
                depth = outcome.effective_depth,
                requested_depth = outcome.requested_depth,
                committed = cycle_committed,
                wall_ms = (cycle_wall_ns as f64) / 1_000_000.0,
                next_depth = mtp_depth_policy.pick_depth(),
                "W6.8 cycle"
            );
        }

        // Emit each accepted token through the same stop /
        // streaming pipeline as the single-token loop.
        let mut hit_stop = false;
        let mut cycle_emitted: usize = 0;
        profiler.begin("mtp_emit_loop");
        for tok_id in outcome.tokens.iter().copied() {
            if generated.len() >= max_as_usize {
                if reason.is_empty() {
                    *reason = String::from("length");
                }
                hit_stop = true;
                break;
            }
            generated.push(tok_id);
            hist.push(tok_id);
            cycle_emitted += 1;
            let _is_reasoning = tracker.observe_token(tok_id);
            // Streaming-only — relocated VERBATIM from `decode_loop_mtp!`'s
            // emit-loop arm. It runs AFTER `observe_token` and BEFORE the EOS
            // check (the macro's order). A cancel here breaks "cancelled":
            // the last outcome token is the unforwarded boundary
            // (bonus/residual), so keep an earlier emitted token (verify
            // wrote its K/V) but drop the boundary —
            // `last_in_cache = cycle_emitted < outcome.tokens.len()`.
            // Detokenize + length-advance stay outside the emitter's gate so
            // DecodeStream sees every token.
            if let Some(s) = streaming.as_mut() {
                *s.last_is_reasoning = _is_reasoning;
                if s.cancelled.load(Ordering::Relaxed) {
                    *reason = String::from("cancelled");
                    hit_stop = true;
                    last_in_cache = cycle_emitted < outcome.tokens.len();
                    break;
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
            if tok_id == eos_id || p.extra_eos_ids.contains(&tok_id) {
                *reason = String::from("stop");
                hit_stop = true;
                // The boundary (last) outcome token is forwarded only by the
                // next cycle's Step A, so it is not yet in the physical cache.
                last_in_cache = cycle_emitted < outcome.tokens.len();
                break;
            }
            if let Some(reason_str) = crate::sampling::check_repetition_cutoff(
                generated,
                p.max_consecutive_tokens,
                p.max_ngram_repeats,
                p.ngram_size,
            ) {
                *reason = reason_str.to_string();
                hit_stop = true;
                // The boundary (last) outcome token is forwarded only by the
                // next cycle's Step A, so it is not yet in the physical cache.
                last_in_cache = cycle_emitted < outcome.tokens.len();
                break;
            }
        }
        profiler.end();
        tracing::debug!(
            target: "mlx_core::mtp",
            cycle_committed,
            gen_len = generated.len(),
            hit_stop,
            cycle_emitted,
            "MTP cycle emit loop done"
        );

        // Every-256-emitted-token cache clear (matches the
        // single-token loop's cadence in token-count units).
        if generated.len() >= last_clear_at + 256 {
            crate::array::synchronize_and_clear_cache();
            last_clear_at = generated.len();
        }

        if hit_stop {
            let unemitted = outcome.tokens.len().saturating_sub(cycle_emitted);
            if unemitted > 0 {
                step.rollback_unemitted(unemitted);
            }
            break;
        }
        // Set `y` to the last accepted token so the next Step A
        // feeds the right token through main-path forward.
        // (Step A unconditionally re-seeds `prev_hidden_opt` /
        // `prev_emb_opt` / `last_committed_id_opt`, so no explicit
        // drain here.)
        let last = *outcome
            .tokens
            .last()
            .ok_or_else(|| napi::Error::from_reason("at least one accepted"))?
            as i32;
        y = MxArray::from_int32(&[last], &[1])?;

        // When chaining IS enabled, flush the main path's KV-cache
        // lazy graph BEFORE the next cycle starts AND fuse the
        // chained `verify_hidden[K]` slice into the SAME `async_eval`
        // batch as `(token, main layer caches)`.
        //
        // A plain `eval_step(y, h, false)` here would leave the
        // chained hidden LAZY across the iteration boundary (that
        // helper ignores `h` unless `budget_forced`), so when the
        // next cycle's `draft_step(...)` built its graph against
        // `prev_hidden = chained_hidden`, materializing the slice
        // forced a mid-cycle Metal command-buffer roundtrip that the
        // Step-A bypass doesn't pay (Step A's `forward_with_hidden`
        // returns `(logits, hidden)` as siblings of the same forward
        // graph and `eval_step` co-schedules them via
        // `next_token → logits → hidden`).
        //
        // `eval_step_with_chained_hidden` extends the same dispatch
        // to include the slice — so it becomes a sibling of
        // `(token, caches)`, the kernel scheduler can overlap its
        // materialization with the next cycle's draft graph
        // construction, and the chained path stops paying the
        // per-cycle roundtrip.
        //
        // On the Step-A path this whole branch is dead anyway:
        // Step A's `eval_step` call at the top of the NEXT
        // iteration handles the cache flush, and there is no
        // chained hidden to fold. The branch is only entered when
        // `chained_cycles_enabled=true` AND `chained_hidden_opt`
        // is `Some(...)`.
        if chained_cycles_enabled && let Some(ref h) = chained_hidden_opt {
            step.eval_step_with_chained_hidden(&y, h);
        }
        profiler.step();
    }

    profiler.snapshot_memory_after();
    profiler.report();

    // Surface any GDN tape-replay error stashed by the infallible
    // `rollback` on a FULL-accept cycle (the partial-accept path
    // already surfaces it via `restore_and_replay_main`'s `?` inside
    // `run_mtp_cycle`). Without this, a full-accept replay failure would
    // be silently swallowed. == the macro callers' post-loop
    // `replay_err_cell.borrow_mut().take()` check.
    if let Some(e) = step.take_replay_error() {
        return Err(e);
    }

    // `into_desynced` consumes the stepper; the caller threads the
    // result into `self.flat_mtp_caches_desynced` exactly as the
    // post-macro `mtp_desynced.get()` read does. Paged MUST return false.
    let desynced = step.into_desynced();

    Ok(MtpTurnOutcome {
        last_in_cache,
        desynced,
    })
}

#[cfg(test)]
mod tests {
    //! `MtpStepper`-contract tests over a scripted mock — NO model, NO
    //! Metal. The UNIQUE value here is proving the trait's GAT lifetimes
    //! and the strictly-sequential `&mut self` borrow model: the harness
    //! drives a short scripted propose/verify/commit/rollback sequence and
    //! asserts the recorded call ledger, exactly as the
    //! `run_paged_turn` mock asserts the paged lifecycle sequence.

    use std::cell::RefCell;

    use napi::bindgen_prelude::*;

    use crate::array::MxArray;
    use crate::engine::backend::MtpStepper;
    use crate::engine::params::ChatParams;
    use crate::models::qwen3_5::adaptive_depth::ExpectedValueDepthPolicy;
    use crate::models::qwen3_5::mtp_decode::{
        ForceSparseAcceptGuard, MtpCommitAnchor, MtpCycleOutcome, MtpVerifyOutput,
    };
    use crate::sampling::SamplingConfig;

    use super::{MtpTurnArgs, run_mtp_cycle, run_mtp_turn};

    /// One recorded `MtpStepper` call, tagged so a test can assert the
    /// exact propose/verify/commit/rollback ORDER (the analog of the paged
    /// harness's `Vec<&'static str>` ledger, enum-typed so per-call payload
    /// — depths, accept counts — rides along).
    #[derive(Clone, Debug, PartialEq, Eq)]
    enum Call {
        EmbeddingWeight,
        CommittedHistoryActive,
        ProfilerRelabel,
        ForwardWithHidden,
        DraftStep,
        VerifyStep { depth: usize },
        VerifyStepArgmaxOnly { depth: usize },
        VerifyStepSparse { depth: usize },
        SnapshotMainLinear,
        Rollback { accepted: usize, depth: usize },
        RestoreAndReplayMain { accepted: usize },
        CommitMtp { anchor: MtpCommitAnchor, k: usize },
        BeginCycle { chained: bool },
        EvalStep { budget_forced: bool },
        EvalStepWithChainedHidden,
        RollbackUnemitted { unemitted: usize },
        TakeReplayError,
        IntoDesynced,
    }

    /// Tiny lazy `[1, 1]` array — fabricated WITHOUT Metal (mlx arrays are
    /// lazy, so construction never touches the GPU). The mock hands these
    /// back wherever the contract returns an [`MxArray`]; the engine never
    /// evals them in S0 (no loop yet), and the borrow-model proof needs
    /// only that the handles thread cleanly between calls.
    fn lazy_scalar(v: f32) -> MxArray {
        MxArray::from_float32(&[v], &[1, 1]).expect("lazy [1,1] array construction is infallible")
    }

    /// Scripted [`MtpStepper`] double. Records every call into an ordered
    /// ledger (interior-mutable so the `&self` `eval_step*` /
    /// `profiler_relabel` / `embedding_weight` methods can record too) and
    /// returns canned lazy arrays / values.
    ///
    /// `committed_history` toggles [`MtpStepper::committed_history_active`]
    /// (dense=true / MoE=false); `relabel` is the canned
    /// [`MtpStepper::profiler_relabel`]; `desynced` is the canned
    /// [`MtpStepper::into_desynced`] terminal value (paged MUST be false);
    /// `replay_error` lets a test script a stashed rollback error the
    /// engine would surface via [`MtpStepper::take_replay_error`].
    ///
    /// `has_argmax_only` / `has_sparse` gate the optional verify fast paths
    /// (default-`None` on every eager family today) so a test can prove
    /// both the "fast path present" and "fall back to verify_step" arms
    /// compile and dispatch.
    struct MockMtpStepper {
        ledger: RefCell<Vec<Call>>,
        emb: MxArray,
        committed_history: bool,
        relabel: Option<&'static str>,
        desynced: bool,
        replay_error: RefCell<Option<Error>>,
        has_argmax_only: bool,
        has_sparse: bool,
        // ---- canned-array driving (the `run_mtp_cycle` integration path) ----
        // When `Some`, `draft_step` / `verify_step` return REAL shaped MLX
        // arrays so `run_mtp_cycle` executes its T=0 sparse-accept branch with
        // the real argmax/eval/slice math (no Metal model). `None` keeps the
        // tiny scalar returns the call-ledger unit tests use.
        cycle: Option<CycleScript>,
        // ---- whole-turn driving (the `run_mtp_turn` integration path) ----
        // When `Some`, `forward_with_hidden` (Step A) returns `[1, vocab]`
        // logits with a scripted argmax AND each cycle's `draft_step` /
        // `verify_step` read per-cycle argmaxes, so `run_mtp_turn` walks a
        // fully deterministic multi-cycle propose/verify sequence with no
        // Metal model. Mutually exclusive with `cycle` (turn wins if both).
        turn: Option<TurnScript>,
        // Optional ledger shared with the owning `MockMtpBackend` so the test
        // can read the call sequence AFTER `run_mtp_turn` consumes the stepper
        // via `into_desynced(self)`. `record` mirrors every call here too.
        shared_ledger: Option<std::rc::Rc<RefCell<Vec<Call>>>>,
        // `(committed_ids, k_accepted)` the cycle handed to the most recent
        // `commit_mtp` — the only place it surfaces the exact committed-token
        // sequence (anchor policy + accepted prefix + boundary). `None` until
        // the first commit.
        commit_payload: RefCell<Option<(Vec<u32>, usize)>>,
    }

    /// Canned per-cycle script for the `run_mtp_cycle` integration tests.
    ///
    /// `vocab` / `hidden` size the logits/hidden arrays. `draft_argmax[i]` is
    /// the token the i-th `draft_step` will produce (argmax of its `[1,vocab]`
    /// logits at T=0). `verify_argmax[j]` is `argmax(verify_logits[0, j, :])` —
    /// length `depth + 1` — so the accept loop decides
    /// `verify_argmax[i] == draft_argmax[i]` per position and reads
    /// `verify_argmax[depth]` as the full-accept bonus. The mock builds an
    /// `embedding_weight` of `[vocab, hidden]` so the cycle's per-draft
    /// `embedding_weight.take(id)` succeeds.
    struct CycleScript {
        vocab: i64,
        hidden: i64,
        draft_argmax: Vec<i32>,
        verify_argmax: Vec<i32>,
        next_draft: std::cell::Cell<usize>,
        /// Value placed at every NON-argmax logit slot. `0.0` for the T=0
        /// sparse-accept tests (only the argmax matters there, so the softmax
        /// shape is irrelevant). The dense-accept tests set this to an
        /// under-flowing negative (`-1e30`, `exp == 0.0` in f32) so `softmax`
        /// is EXACTLY one-hot at the argmax: the draft / bonus / residual draws
        /// are degenerate and the stochastic `accept_with_residual` outcome is
        /// deterministic BY CONSTRUCTION, independent of both the MLX sampler
        /// RNG and the Rust accept RNG.
        neg_fill: f32,
    }

    /// Per-cycle argmax script for the whole-turn [`TurnScript`].
    /// `draft_argmax[i]` is the i-th `draft_step`'s argmax; `verify_argmax[j]`
    /// is `argmax(verify_logits[0, j, :])` (length `depth + 1`). The accept
    /// loop then decides `verify_argmax[i] == draft_argmax[i]` per position
    /// and reads `verify_argmax[depth]` as the full-accept bonus — exactly the
    /// `CycleScript` contract, replayed once per cycle.
    #[derive(Clone)]
    struct CycleArgmax {
        draft_argmax: Vec<i32>,
        verify_argmax: Vec<i32>,
    }

    /// Whole-turn argmax script driving `run_mtp_turn` over the mock with no
    /// Metal model.
    ///
    /// `step_a_tokens[n]` is the argmax of the n-th `forward_with_hidden`
    /// (Step A) `[1, vocab]` logits — the token Step A samples and emits at
    /// the top of outer iteration n. `cycles[c]` scripts cycle c's per-draft
    /// and per-verify argmaxes; the cursor advances once per `verify_step`
    /// (which marks the end of a cycle) and resets the in-cycle draft cursor.
    /// Cursors that run past the end of their script repeat the last value for
    /// Step A and yield argmax 0 for drafts so an over-long run stays defined.
    struct TurnScript {
        vocab: i64,
        hidden: i64,
        step_a_tokens: Vec<i32>,
        cycles: Vec<CycleArgmax>,
        fwd_cursor: std::cell::Cell<usize>,
        cycle_cursor: std::cell::Cell<usize>,
        draft_in_cycle: std::cell::Cell<usize>,
    }

    impl MockMtpStepper {
        fn new() -> Self {
            Self {
                ledger: RefCell::new(Vec::new()),
                emb: lazy_scalar(1.0),
                committed_history: true,
                relabel: None,
                desynced: false,
                replay_error: RefCell::new(None),
                has_argmax_only: false,
                has_sparse: false,
                cycle: None,
                turn: None,
                shared_ledger: None,
                commit_payload: RefCell::new(None),
            }
        }

        /// Build a canned-array mock for the `run_mtp_cycle` integration path.
        /// `embedding_weight` becomes a `[vocab, hidden]` array of zeros (only
        /// its shape + `take` indexing matter to the cycle).
        fn with_cycle(
            vocab: i64,
            hidden: i64,
            draft_argmax: Vec<i32>,
            verify_argmax: Vec<i32>,
        ) -> Self {
            let mut s = Self::new();
            s.emb =
                MxArray::from_float32(&vec![0.0f32; (vocab * hidden) as usize], &[vocab, hidden])
                    .expect("embedding_weight [vocab,hidden] construction is infallible");
            s.cycle = Some(CycleScript {
                vocab,
                hidden,
                draft_argmax,
                verify_argmax,
                next_draft: std::cell::Cell::new(0),
                neg_fill: 0.0,
            });
            s
        }

        /// Like [`with_cycle`], but gives the logits EXACT zero-support at every
        /// non-argmax slot so the cycle can run its DENSE stochastic
        /// `accept_with_residual` branch (T>0) deterministically BY
        /// CONSTRUCTION. In f32 `exp(-1e30) == 0.0` (underflow), so `softmax`
        /// over these logits is exactly one-hot at the argmax: the per-position
        /// draft / bonus / residual categorical draws are degenerate (all mass
        /// on one token), so the emitted tokens do NOT depend on the MLX sampler
        /// RNG at all. A full-accept position then has draft logits == verify
        /// logits (accept ratio exactly 1.0, accept for any draw) and a
        /// disagreement puts ALL residual mass on the verifier argmax.
        ///
        /// A literal `-inf` would also be exact-one-hot but risks `inf * 0 ==
        /// NaN` in downstream ops; a finite under-flowing fill is strictly safer.
        fn with_cycle_dense(
            vocab: i64,
            hidden: i64,
            draft_argmax: Vec<i32>,
            verify_argmax: Vec<i32>,
        ) -> Self {
            let mut s = Self::with_cycle(vocab, hidden, draft_argmax, verify_argmax);
            if let Some(c) = s.cycle.as_mut() {
                c.neg_fill = -1.0e30;
            }
            s
        }

        /// Build a whole-turn mock for the `run_mtp_turn` integration path.
        /// `embedding_weight` is a `[vocab, hidden]` zero table; the script
        /// drives Step A + every cycle deterministically.
        fn with_turn(
            vocab: i64,
            hidden: i64,
            step_a_tokens: Vec<i32>,
            cycles: Vec<CycleArgmax>,
        ) -> Self {
            let mut s = Self::new();
            s.emb =
                MxArray::from_float32(&vec![0.0f32; (vocab * hidden) as usize], &[vocab, hidden])
                    .expect("embedding_weight [vocab,hidden] construction is infallible");
            s.turn = Some(TurnScript {
                vocab,
                hidden,
                step_a_tokens,
                cycles,
                fwd_cursor: std::cell::Cell::new(0),
                cycle_cursor: std::cell::Cell::new(0),
                draft_in_cycle: std::cell::Cell::new(0),
            });
            s
        }

        fn record(&self, c: Call) {
            if let Some(shared) = self.shared_ledger.as_ref() {
                shared.borrow_mut().push(c.clone());
            }
            self.ledger.borrow_mut().push(c);
        }

        fn snapshot(&self) -> Vec<Call> {
            self.ledger.borrow().clone()
        }
    }

    /// Build a `[1, vocab]` (or generally `[..., vocab]`) f32 logits row whose
    /// argmax over the final axis is `argmax_id`: a one-hot-ish vector with a
    /// large positive spike at `argmax_id` and zeros elsewhere.
    fn logits_row(vocab: i64, argmax_id: i32) -> Vec<f32> {
        logits_row_filled(vocab, argmax_id, 0.0)
    }

    /// `logits_row` with an explicit non-argmax fill value. A large negative
    /// fill (e.g. `-30.0`) makes `softmax` ~one-hot at `argmax_id`, which the
    /// dense `accept_with_residual` tests rely on for deterministic accept /
    /// reject (the argmax stays `argmax_id` for any fill `< 10.0`, so the T=0
    /// argmax-only callers are unaffected by the choice of fill).
    fn logits_row_filled(vocab: i64, argmax_id: i32, fill: f32) -> Vec<f32> {
        let mut row = vec![fill; vocab as usize];
        if (0..vocab as i32).contains(&argmax_id) {
            row[argmax_id as usize] = 10.0;
        }
        row
    }

    impl MtpStepper for MockMtpStepper {
        fn embedding_weight(&self) -> &MxArray {
            self.record(Call::EmbeddingWeight);
            &self.emb
        }

        fn committed_history_active(&self) -> bool {
            self.record(Call::CommittedHistoryActive);
            self.committed_history
        }

        fn profiler_relabel(&self) -> Option<&'static str> {
            self.record(Call::ProfilerRelabel);
            self.relabel
        }

        fn forward_with_hidden(
            &mut self,
            _ids: &MxArray,
            _emb: &MxArray,
        ) -> Result<(MxArray, MxArray, bool)> {
            self.record(Call::ForwardWithHidden);
            match self.turn.as_ref() {
                Some(t) => {
                    // Step A logits `[1, vocab]` whose argmax (the T=0 draw,
                    // after the engine's `squeeze(axis=1)` on `needs_squeeze`)
                    // is the scripted `step_a_tokens[n]`; the last entry
                    // repeats once the script is exhausted. Hidden is
                    // `[1, hidden]` (eager Step-A shape; the engine reshapes
                    // it to `[1, 1, hidden]`).
                    let n = t.fwd_cursor.get();
                    t.fwd_cursor.set(n + 1);
                    let idx = n.min(t.step_a_tokens.len().saturating_sub(1));
                    let argmax_id = t.step_a_tokens.get(idx).copied().unwrap_or(0);
                    let logits =
                        MxArray::from_float32(&logits_row(t.vocab, argmax_id), &[1, 1, t.vocab])?;
                    let hidden =
                        MxArray::from_float32(&vec![0.0f32; t.hidden as usize], &[1, t.hidden])?;
                    Ok((logits, hidden, true))
                }
                // (logits [1,1], hidden [1,1], needs_squeeze) — eager shape.
                None => Ok((lazy_scalar(0.0), lazy_scalar(0.0), true)),
            }
        }

        fn draft_step(
            &mut self,
            _prev_h: &MxArray,
            _prev_emb: &MxArray,
        ) -> Result<(MxArray, MxArray)> {
            self.record(Call::DraftStep);
            if let Some(t) = self.turn.as_ref() {
                // h_next [1,1,hidden]; draft_logits [1,vocab] whose argmax (the
                // T=0 draw) is the current cycle's scripted `draft_argmax[i]`.
                let c_idx = t.cycle_cursor.get();
                let i = t.draft_in_cycle.get();
                t.draft_in_cycle.set(i + 1);
                let argmax_id = t
                    .cycles
                    .get(c_idx)
                    .and_then(|c| c.draft_argmax.get(i).copied())
                    .unwrap_or(0);
                let h_next =
                    MxArray::from_float32(&vec![0.0f32; t.hidden as usize], &[1, 1, t.hidden])?;
                let draft_logits =
                    MxArray::from_float32(&logits_row(t.vocab, argmax_id), &[1, t.vocab])?;
                return Ok((h_next, draft_logits));
            }
            match self.cycle.as_ref() {
                Some(c) => {
                    // h_next [1,1,hidden]; draft_logits [1,vocab] whose argmax
                    // (the T=0 draw) is the scripted `draft_argmax[step]`.
                    let i = c.next_draft.get();
                    c.next_draft.set(i + 1);
                    let argmax_id = c.draft_argmax.get(i).copied().unwrap_or(0);
                    let h_next =
                        MxArray::from_float32(&vec![0.0f32; c.hidden as usize], &[1, 1, c.hidden])?;
                    let draft_logits = MxArray::from_float32(
                        &logits_row_filled(c.vocab, argmax_id, c.neg_fill),
                        &[1, c.vocab],
                    )?;
                    Ok((h_next, draft_logits))
                }
                None => Ok((lazy_scalar(0.0), lazy_scalar(0.0))),
            }
        }

        fn verify_step(
            &mut self,
            _ids: &MxArray,
            _emb: &MxArray,
            depth: usize,
        ) -> Result<MtpVerifyOutput> {
            self.record(Call::VerifyStep { depth });
            if let Some(t) = self.turn.as_ref() {
                // logits [1, depth+1, vocab] with per-position argmax driven by
                // the current cycle's `verify_argmax`; hiddens
                // [1, depth+1, hidden]. `verify_step` marks the END of a cycle,
                // so AFTER building the outputs advance the cycle cursor and
                // reset the in-cycle draft cursor.
                let c_idx = t.cycle_cursor.get();
                let rows = depth + 1;
                let mut flat: Vec<f32> = Vec::with_capacity(rows * t.vocab as usize);
                for j in 0..rows {
                    let argmax_id = t
                        .cycles
                        .get(c_idx)
                        .and_then(|c| c.verify_argmax.get(j).copied())
                        .unwrap_or(0);
                    flat.extend(logits_row(t.vocab, argmax_id));
                }
                let logits = MxArray::from_float32(&flat, &[1, rows as i64, t.vocab])?;
                let hiddens = MxArray::from_float32(
                    &vec![0.0f32; rows * t.hidden as usize],
                    &[1, rows as i64, t.hidden],
                )?;
                t.cycle_cursor.set(c_idx + 1);
                t.draft_in_cycle.set(0);
                return Ok(MtpVerifyOutput::logits_only(logits, hiddens));
            }
            match self.cycle.as_ref() {
                Some(c) => {
                    // logits [1, depth+1, vocab] with per-position argmax driven
                    // by `verify_argmax`; hiddens [1, depth+1, hidden].
                    let rows = depth + 1;
                    let mut flat: Vec<f32> = Vec::with_capacity(rows * c.vocab as usize);
                    for j in 0..rows {
                        let argmax_id = c.verify_argmax.get(j).copied().unwrap_or(0);
                        flat.extend(logits_row_filled(c.vocab, argmax_id, c.neg_fill));
                    }
                    let logits = MxArray::from_float32(&flat, &[1, rows as i64, c.vocab])?;
                    let hiddens = MxArray::from_float32(
                        &vec![0.0f32; rows * c.hidden as usize],
                        &[1, rows as i64, c.hidden],
                    )?;
                    Ok(MtpVerifyOutput::logits_only(logits, hiddens))
                }
                None => Ok(MtpVerifyOutput::logits_only(
                    lazy_scalar(0.0),
                    lazy_scalar(0.0),
                )),
            }
        }

        fn verify_step_argmax_only(
            &mut self,
            _ids: &MxArray,
            _emb: &MxArray,
            depth: usize,
        ) -> Option<Result<MtpVerifyOutput>> {
            self.record(Call::VerifyStepArgmaxOnly { depth });
            if self.has_argmax_only {
                Some(Ok(MtpVerifyOutput::logits_only(
                    lazy_scalar(0.0),
                    lazy_scalar(0.0),
                )))
            } else {
                None
            }
        }

        fn verify_step_sparse(
            &mut self,
            _ids: &MxArray,
            _emb: &MxArray,
            depth: usize,
            _cfg: &SamplingConfig,
        ) -> Option<Result<MtpVerifyOutput>> {
            self.record(Call::VerifyStepSparse { depth });
            if self.has_sparse {
                Some(Ok(MtpVerifyOutput::logits_only(
                    lazy_scalar(0.0),
                    lazy_scalar(0.0),
                )))
            } else {
                None
            }
        }

        fn snapshot_main_linear(&mut self) {
            self.record(Call::SnapshotMainLinear);
        }

        fn rollback(&mut self, accepted_drafts: usize, depth: usize) {
            self.record(Call::Rollback {
                accepted: accepted_drafts,
                depth,
            });
        }

        fn restore_and_replay_main(&mut self, accepted: &[u32], _emb: &MxArray) -> Result<()> {
            self.record(Call::RestoreAndReplayMain {
                accepted: accepted.len(),
            });
            Ok(())
        }

        fn commit_mtp(
            &mut self,
            anchor: MtpCommitAnchor,
            _seed_h: &MxArray,
            _verify_hiddens: &MxArray,
            committed_ids: &[u32],
            k_accepted: usize,
            _emb: &MxArray,
        ) -> Result<()> {
            self.record(Call::CommitMtp {
                anchor,
                k: k_accepted,
            });
            // Committed-sequence shape: `IncludeAnchor` prepends the anchor
            // (`[last_committed, d_0..d_{K-1}, boundary]` = K+2);
            // `SkipAlreadyCommittedAnchor` (chained cycles) omits it
            // (`[d_0..d_{K-1}, boundary]` = K+1).
            let expected_len = match anchor {
                MtpCommitAnchor::IncludeAnchor => k_accepted + 2,
                MtpCommitAnchor::SkipAlreadyCommittedAnchor => k_accepted + 1,
            };
            assert_eq!(
                committed_ids.len(),
                expected_len,
                "committed_ids length must match the anchor policy"
            );
            *self.commit_payload.borrow_mut() = Some((committed_ids.to_vec(), k_accepted));
            Ok(())
        }

        fn begin_cycle(&mut self, chained_anchor: bool) {
            self.record(Call::BeginCycle {
                chained: chained_anchor,
            });
        }

        fn eval_step(&self, _token: &MxArray, _logits: &MxArray, budget_forced: bool) {
            self.record(Call::EvalStep { budget_forced });
        }

        fn eval_step_with_chained_hidden(&self, _token: &MxArray, _chained_h: &MxArray) {
            self.record(Call::EvalStepWithChainedHidden);
        }

        fn rollback_unemitted(&mut self, unemitted: usize) {
            self.record(Call::RollbackUnemitted { unemitted });
        }

        fn take_replay_error(&mut self) -> Option<Error> {
            self.record(Call::TakeReplayError);
            self.replay_error.borrow_mut().take()
        }

        fn into_desynced(self) -> bool {
            self.record(Call::IntoDesynced);
            self.desynced
        }
    }

    /// Drive a short scripted propose/verify/commit/rollback sequence
    /// through the trait, EXACTLY in the order `run_mtp_cycle` calls the
    /// stepper methods (Step A forward → begin_cycle → D draft steps →
    /// snapshot → verify → commit → rollback → restore/replay on reject →
    /// eval), then the iteration-boundary fused chained eval. Proves the
    /// strictly-sequential `&mut self` borrow model + GAT-free dyn-less
    /// dispatch compile and run — no Metal, no model.
    fn drive_one_reject_cycle(step: &mut MockMtpStepper, depth: usize, accepted: usize) {
        // Turn-entry reads (the engine pulls these once before the loop).
        let _relabel = step.profiler_relabel();
        let _committed = step.committed_history_active();

        // Step A: main-path forward → seed hidden/emb. `emb` is read
        // through `&self` then re-borrowed into the `&mut self` forward —
        // the clone breaks the borrow overlap the real loop also avoids.
        let emb = step.embedding_weight().clone();
        let (_logits, _hidden, _sq) = step
            .forward_with_hidden(&lazy_scalar(0.0), &emb)
            .expect("mock forward never fails");

        // Re-anchor, then D draft steps threading (h_next, emb) forward.
        step.begin_cycle(false);
        let mut prev_h = lazy_scalar(0.0);
        let mut prev_emb = emb.clone();
        for _ in 0..depth {
            let (h_next, _draft_logits) = step
                .draft_step(&prev_h, &prev_emb)
                .expect("mock draft never fails");
            prev_h = h_next;
            prev_emb = lazy_scalar(0.0);
        }

        // Snapshot → verify (fast-path probe falls through to verify_step).
        step.snapshot_main_linear();
        let _argmax = step.verify_step_argmax_only(&lazy_scalar(0.0), &emb, depth);
        let _verify = step
            .verify_step(&lazy_scalar(0.0), &emb, depth)
            .expect("mock verify never fails");

        // Commit the K+2 committed sequence, then rollback + replay on the
        // partial-accept (reject) arm.
        let committed_ids: Vec<u32> = std::iter::repeat_n(7u32, accepted + 2).collect();
        step.commit_mtp(
            MtpCommitAnchor::IncludeAnchor,
            &lazy_scalar(0.0),
            &lazy_scalar(0.0),
            &committed_ids,
            accepted,
            &emb,
        )
        .expect("mock commit never fails");
        step.rollback(accepted, depth);
        if accepted < depth {
            let accepted_ids: Vec<u32> = std::iter::repeat_n(7u32, accepted).collect();
            step.restore_and_replay_main(&accepted_ids, &emb)
                .expect("mock replay never fails");
        }
        // The engine surfaces any stashed replay error after the
        // (infallible) rollback.
        let _stashed = step.take_replay_error();

        // Per-token eval + the iteration-boundary fused chained eval.
        step.eval_step(&lazy_scalar(0.0), &lazy_scalar(0.0), false);
        step.eval_step_with_chained_hidden(&lazy_scalar(0.0), &lazy_scalar(0.0));
    }

    #[test]
    fn mtp_stepper_reject_cycle_call_sequence() {
        let mut step = MockMtpStepper::new();
        drive_one_reject_cycle(&mut step, 3, 1);
        let desynced = step.snapshot();
        // `into_desynced` consumes `self`; capture the terminal value
        // separately after snapshotting the ledger.
        let mut step2 = MockMtpStepper::new();
        drive_one_reject_cycle(&mut step2, 3, 1);
        let terminal_desynced = step2.into_desynced();

        assert_eq!(
            desynced,
            vec![
                Call::ProfilerRelabel,
                Call::CommittedHistoryActive,
                Call::EmbeddingWeight,
                Call::ForwardWithHidden,
                Call::BeginCycle { chained: false },
                Call::DraftStep,
                Call::DraftStep,
                Call::DraftStep,
                Call::SnapshotMainLinear,
                Call::VerifyStepArgmaxOnly { depth: 3 },
                Call::VerifyStep { depth: 3 },
                Call::CommitMtp {
                    anchor: MtpCommitAnchor::IncludeAnchor,
                    k: 1,
                },
                Call::Rollback {
                    accepted: 1,
                    depth: 3,
                },
                Call::RestoreAndReplayMain { accepted: 1 },
                Call::TakeReplayError,
                Call::EvalStep {
                    budget_forced: false,
                },
                Call::EvalStepWithChainedHidden,
            ],
            "the engine must drive the MTP propose/verify/commit/rollback \
             sequence in the order run_mtp_cycle calls the stepper methods"
        );
        // Paged MUST report not-desynced; the mock default mirrors that.
        assert!(
            !terminal_desynced,
            "into_desynced default (paged contract) is false"
        );
    }

    #[test]
    fn mtp_stepper_full_accept_skips_restore_and_replay() {
        // K == depth (full accept) → the engine SKIPS restore_and_replay_main
        // (verify already advanced the linear state through all D drafts).
        let mut step = MockMtpStepper::new();
        drive_one_reject_cycle(&mut step, 2, 2);
        let seq = step.snapshot();
        assert!(
            !seq.contains(&Call::RestoreAndReplayMain { accepted: 2 }),
            "full accept must NOT replay"
        );
        assert!(
            seq.contains(&Call::Rollback {
                accepted: 2,
                depth: 2,
            }),
            "full accept still calls rollback(accepted=depth, depth) for GDN normalization"
        );
    }

    #[test]
    fn mtp_stepper_verify_fast_paths_dispatch() {
        // argmax-only fast path present → returns Some, engine uses it.
        let mut argmax = MockMtpStepper::new();
        argmax.has_argmax_only = true;
        let r = argmax.verify_step_argmax_only(&lazy_scalar(0.0), &lazy_scalar(0.0), 4);
        assert!(r.is_some(), "argmax-only present must return Some");
        assert!(r.expect("present").is_ok());

        // sparse fast path present → Some; absent default → None (fall back
        // to verify_step, the eager-family shape).
        let mut sparse = MockMtpStepper::new();
        sparse.has_sparse = true;
        let cfg = SamplingConfig::default();
        let s = sparse.verify_step_sparse(&lazy_scalar(0.0), &lazy_scalar(0.0), 4, &cfg);
        assert!(s.is_some(), "sparse present must return Some");

        let mut none = MockMtpStepper::new();
        assert!(
            none.verify_step_argmax_only(&lazy_scalar(0.0), &lazy_scalar(0.0), 4)
                .is_none(),
            "absent argmax-only default is None"
        );
        assert!(
            none.verify_step_sparse(&lazy_scalar(0.0), &lazy_scalar(0.0), 4, &cfg)
                .is_none(),
            "absent sparse default is None"
        );
    }

    #[test]
    fn mtp_stepper_surfaces_stashed_replay_error() {
        // A stashed rollback-replay error is surfaced by take_replay_error
        // (the engine then `?`-propagates it AFTER the infallible rollback).
        let mut step = MockMtpStepper::new();
        *step.replay_error.borrow_mut() = Some(Error::from_reason("scripted replay failure"));
        step.rollback(1, 3);
        let err = step.take_replay_error();
        assert!(err.is_some(), "stashed replay error must surface");
        assert_eq!(err.expect("present").reason, "scripted replay failure");
        // Drained: a second take yields None.
        assert!(
            step.take_replay_error().is_none(),
            "take_replay_error drains the stash"
        );
    }

    #[test]
    fn mtp_stepper_into_desynced_paged_vs_flat() {
        // Flat/MoE may set the desync flag on a mid-cycle stop; paged MUST
        // return false. Both arms compile through the consuming `self`
        // signature.
        let flat = {
            let mut s = MockMtpStepper::new();
            s.desynced = true;
            s.rollback_unemitted(2); // mid-cycle stop left 2 unemitted
            s
        };
        assert!(flat.into_desynced(), "flat mid-cycle stop reports desynced");

        let paged = {
            let mut s = MockMtpStepper::new();
            s.desynced = false; // paged truncates the adapter, never flat-desyncs
            s.rollback_unemitted(2);
            s
        };
        assert!(
            !paged.into_desynced(),
            "paged into_desynced contract is false"
        );
    }

    // -----------------------------------------------------------------------
    // `run_mtp_cycle` integration tests — DRIVE the relocated cycle over the
    // `MockMtpStepper` with REAL shaped canned arrays so the T=0 sparse-accept
    // branch's argmax/eval/slice math actually runs (no Metal model). Mock
    // numerics are fake, so the assertions are STRUCTURAL: emitted token
    // count, accepted-draft K (read off the `CommitMtp` ledger entry), and the
    // call ORDER `run_mtp_cycle` itself drives.
    // -----------------------------------------------------------------------

    /// T=0 greedy `ChatParams` — drives `run_mtp_cycle` down the
    /// sparse-accept (deterministic argmax) branch with all penalties at
    /// their no-op defaults. Only the fields the cycle reads are set
    /// meaningfully; the rest are inert.
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
            mtp_depth: 3,
            mtp_adaptive_depth: false,
        }
    }

    /// T=1.0 `ChatParams` — drives `run_mtp_cycle` down the DENSE stochastic
    /// `accept_with_residual` branch: `temperature == 1.0` is not greedy
    /// (`use_sparse_accept == false`) and the batched-target-array env is
    /// off by default (`use_sparse_stochastic_accept == false`), so neither
    /// sparse fast path applies. All penalties stay at their no-op defaults.
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

    /// Outcome of one scripted `run_mtp_cycle` over the canned mock.
    struct ScriptedCycle {
        outcome: MtpCycleOutcome,
        ledger: Vec<Call>,
        /// `(committed_ids, k_accepted)` the cycle handed to `commit_mtp` —
        /// the exact committed-token sequence under the chosen anchor policy.
        commit_payload: Option<(Vec<u32>, usize)>,
        /// Whether the T=0 sparse-accept commit path (`mtp_accept_argmax`)
        /// actually ran. Fail-closed coverage: a test asserting the
        /// deterministic-commit contract must confirm it took that branch.
        ran_sparse: bool,
    }

    /// Run one scripted `run_mtp_cycle` over the canned mock with an explicit
    /// EV depth policy and commit anchor. Forces the sparse-accept gate ON so
    /// the deterministic T=0 branch runs regardless of `MLX_MTP_SPARSE_ACCEPT`,
    /// and enables the profiler so `ran_sparse` can fail-closed.
    fn run_scripted_cycle_full(
        vocab: i64,
        hidden: i64,
        draft_argmax: Vec<i32>,
        verify_argmax: Vec<i32>,
        depth: usize,
        last_committed_id: u32,
        ev_policy: Option<ExpectedValueDepthPolicy>,
        commit_anchor: MtpCommitAnchor,
    ) -> ScriptedCycle {
        let _force = ForceSparseAcceptGuard::force(true);
        let mut step = MockMtpStepper::with_cycle(vocab, hidden, draft_argmax, verify_argmax);
        let params = greedy_params();
        let mut rng = rand::rng();
        let mut profiler = crate::decode_profiler::DecodeProfiler::new("mtp_test", "test");
        profiler.enable_for_test();
        // Embedding weight is the mock's own `[vocab, hidden]` table.
        let emb = step.emb.clone();
        let prev_hidden =
            MxArray::from_float32(&vec![0.0f32; hidden as usize], &[1, 1, hidden]).unwrap();
        let prev_emb =
            MxArray::from_float32(&vec![0.0f32; hidden as usize], &[1, 1, hidden]).unwrap();
        let token_history: Vec<u32> = vec![1, 2, 3];
        let mut ev = ev_policy;
        let (outcome, _vh) = run_mtp_cycle(
            &mut step,
            prev_hidden,
            prev_emb,
            last_committed_id,
            &emb,
            &token_history,
            &params,
            &mut rng,
            &mut profiler,
            depth,
            ev.as_mut(),
            commit_anchor,
        )
        .expect("scripted run_mtp_cycle must succeed");
        let ledger = step.snapshot();
        let commit_payload = step.commit_payload.borrow().clone();
        let ran_sparse = profiler.ran_phase("mtp_accept_argmax");
        ScriptedCycle {
            outcome,
            ledger,
            commit_payload,
            ran_sparse,
        }
    }

    /// Thin wrapper for the common case: no EV gate, `IncludeAnchor` (Step-A)
    /// commit policy. Returns just the outcome + call ledger.
    fn run_scripted_cycle(
        vocab: i64,
        hidden: i64,
        draft_argmax: Vec<i32>,
        verify_argmax: Vec<i32>,
        depth: usize,
        last_committed_id: u32,
    ) -> (MtpCycleOutcome, Vec<Call>) {
        let r = run_scripted_cycle_full(
            vocab,
            hidden,
            draft_argmax,
            verify_argmax,
            depth,
            last_committed_id,
            None,
            MtpCommitAnchor::IncludeAnchor,
        );
        (r.outcome, r.ledger)
    }

    /// Outcome of one scripted `run_mtp_cycle` that took the DENSE stochastic
    /// `accept_with_residual` branch. The three `ran_*` flags fail-closed:
    /// the dense branch fires `mtp_accept_loop` and NEITHER sparse phase, so a
    /// test asserts `ran_accept_loop && !ran_argmax_sparse &&
    /// !ran_stochastic_sparse` to prove it really exercised the dense path.
    struct DenseScriptedCycle {
        outcome: MtpCycleOutcome,
        ledger: Vec<Call>,
        commit_payload: Option<(Vec<u32>, usize)>,
        ran_accept_loop: bool,
        ran_argmax_sparse: bool,
        ran_stochastic_sparse: bool,
    }

    /// Run one scripted `run_mtp_cycle` down the DENSE stochastic
    /// `accept_with_residual` branch (T=1.0). Forces the sparse-accept gate
    /// OFF (defensive — T≠0 already disqualifies it). `with_cycle_dense`'s
    /// exact-one-hot logits make every draw degenerate, so the result is
    /// deterministic by construction; the FIXED RNG seed is belt-and-suspenders
    /// (with one-hot support the accept/residual draw never has a choice to
    /// make). Captures the profiler phases so the caller can prove the dense
    /// branch ran.
    ///
    /// This restores the engine-level dense-accept coverage the deleted
    /// `mod mtp_cycle_tests` provided at T=1.0 (the migrated `run_scripted_cycle`
    /// tests only exercise the T=0 sparse-accept branch).
    fn run_scripted_cycle_dense(
        vocab: i64,
        hidden: i64,
        draft_argmax: Vec<i32>,
        verify_argmax: Vec<i32>,
        depth: usize,
        last_committed_id: u32,
        commit_anchor: MtpCommitAnchor,
    ) -> DenseScriptedCycle {
        let _force = ForceSparseAcceptGuard::force(false);
        let mut step = MockMtpStepper::with_cycle_dense(vocab, hidden, draft_argmax, verify_argmax);
        let params = dense_params();
        let mut rng =
            <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(0xD0E5_DEAD_BEEF_F00D);
        let mut profiler = crate::decode_profiler::DecodeProfiler::new("mtp_dense_test", "test");
        profiler.enable_for_test();
        let emb = step.emb.clone();
        let prev_hidden =
            MxArray::from_float32(&vec![0.0f32; hidden as usize], &[1, 1, hidden]).unwrap();
        let prev_emb =
            MxArray::from_float32(&vec![0.0f32; hidden as usize], &[1, 1, hidden]).unwrap();
        let token_history: Vec<u32> = vec![1, 2, 3];
        let (outcome, _vh) = run_mtp_cycle(
            &mut step,
            prev_hidden,
            prev_emb,
            last_committed_id,
            &emb,
            &token_history,
            &params,
            &mut rng,
            &mut profiler,
            depth,
            None,
            commit_anchor,
        )
        .expect("scripted dense run_mtp_cycle must succeed");
        DenseScriptedCycle {
            outcome,
            ledger: step.snapshot(),
            commit_payload: step.commit_payload.borrow().clone(),
            ran_accept_loop: profiler.ran_phase("mtp_accept_loop"),
            ran_argmax_sparse: profiler.ran_phase("mtp_accept_argmax"),
            ran_stochastic_sparse: profiler.ran_phase("mtp_accept_sparse_probs"),
        }
    }

    /// Assert the cycle really took the DENSE `accept_with_residual` branch:
    /// the accept loop ran and NEITHER sparse fast path did.
    fn assert_dense_branch(r: &DenseScriptedCycle, label: &str) {
        assert!(
            r.ran_accept_loop,
            "{label}: the accept loop must have run (dense branch fires `mtp_accept_loop`)"
        );
        assert!(
            !r.ran_argmax_sparse,
            "{label}: must NOT take the T=0 sparse-accept fast path (`mtp_accept_argmax`)"
        );
        assert!(
            !r.ran_stochastic_sparse,
            "{label}: must NOT take the batched sparse-stochastic fast path \
             (`mtp_accept_sparse_probs`)"
        );
    }

    /// Extract the `k_accepted` the cycle reported through its single
    /// `CommitMtp` call (the only K surface in the ledger).
    fn commit_k(ledger: &[Call]) -> usize {
        ledger
            .iter()
            .find_map(|c| match c {
                Call::CommitMtp { k, .. } => Some(*k),
                _ => None,
            })
            .expect("run_mtp_cycle must emit exactly one CommitMtp")
    }

    #[test]
    fn run_mtp_cycle_full_accept_depth3() {
        // depth 3, every draft accepted: verify argmax == draft id at 0,1,2;
        // position 3 is the full-accept bonus (id 6). The cycle commits all 3
        // drafts + the bonus (4 tokens), K == depth, and SKIPS
        // restore_and_replay_main (verify already advanced the linear state).
        let ScriptedCycle {
            outcome,
            ledger,
            commit_payload,
            ..
        } = run_scripted_cycle_full(
            8,
            4,
            vec![3, 4, 5],
            vec![3, 4, 5, 6],
            3,
            3,
            None,
            MtpCommitAnchor::IncludeAnchor,
        );

        assert_eq!(
            outcome.tokens,
            vec![3, 4, 5, 6],
            "full accept emits 3 accepted drafts + bonus"
        );
        assert_eq!(outcome.requested_depth, 3);
        assert_eq!(outcome.effective_depth, 3);
        assert_eq!(commit_k(&ledger), 3, "K == effective_depth on full accept");
        assert_eq!(
            commit_payload,
            Some((vec![3u32, 3, 4, 5, 6], 3)),
            "full-accept IncludeAnchor commits [last_committed, all drafts, bonus]"
        );

        // Call ORDER the cycle itself drives (turn-entry reads like
        // profiler_relabel / committed_history_active / embedding_weight and
        // the Step-A forward / begin_cycle live in the macro/engine, NOT in
        // run_mtp_cycle — so they MUST NOT appear here).
        assert_eq!(
            ledger,
            vec![
                Call::DraftStep,
                Call::DraftStep,
                Call::DraftStep,
                Call::SnapshotMainLinear,
                Call::VerifyStep { depth: 3 },
                Call::CommitMtp {
                    anchor: MtpCommitAnchor::IncludeAnchor,
                    k: 3,
                },
                Call::Rollback {
                    accepted: 3,
                    depth: 3,
                },
            ],
            "full-accept cycle: 3 drafts → snapshot → verify → commit → rollback (no replay)"
        );
        assert!(
            !ledger
                .iter()
                .any(|c| matches!(c, Call::RestoreAndReplayMain { .. })),
            "full accept must NOT restore_and_replay_main"
        );
    }

    #[test]
    fn run_mtp_cycle_full_accept_depth2() {
        // depth 2 full accept: tokens = [d0, d1, bonus] (3), K == 2, no replay.
        let (outcome, ledger) = run_scripted_cycle(8, 4, vec![2, 5], vec![2, 5, 7], 2, 4);
        assert_eq!(outcome.tokens, vec![2, 5, 7]);
        assert_eq!(outcome.effective_depth, 2);
        assert_eq!(commit_k(&ledger), 2);
        assert!(
            !ledger
                .iter()
                .any(|c| matches!(c, Call::RestoreAndReplayMain { .. })),
            "depth-2 full accept must NOT replay"
        );
    }

    #[test]
    fn run_mtp_cycle_partial_accept_rejects_at_pos1() {
        // depth 3, reject at position 1: draft ids [3,4,5]; verify argmax
        // [3, 9, *, *] → pos 0 accepts (3==3), pos 1 rejects (9 != 4) and the
        // residual 9 is emitted. Emitted tokens = [3, 9] (1 accepted draft + 1
        // residual), K == 1, and restore_and_replay_main IS called with the 1
        // accepted draft.
        let (outcome, ledger) = run_scripted_cycle(16, 4, vec![3, 4, 5], vec![3, 9, 0, 0], 3, 3);

        assert_eq!(
            outcome.tokens,
            vec![3, 9],
            "reject at pos1 emits 1 accepted draft + residual"
        );
        assert_eq!(outcome.effective_depth, 3, "all 3 drafts were still built");
        assert_eq!(commit_k(&ledger), 1, "K == accepted-draft prefix length");

        assert_eq!(
            ledger,
            vec![
                Call::DraftStep,
                Call::DraftStep,
                Call::DraftStep,
                Call::SnapshotMainLinear,
                Call::VerifyStep { depth: 3 },
                Call::CommitMtp {
                    anchor: MtpCommitAnchor::IncludeAnchor,
                    k: 1,
                },
                Call::Rollback {
                    accepted: 1,
                    depth: 3,
                },
                Call::RestoreAndReplayMain { accepted: 2 },
            ],
            "reject cycle: drafts → snapshot → verify → commit → rollback → replay"
        );
    }

    #[test]
    fn run_mtp_cycle_all_reject_emits_residual() {
        // depth 3, reject at position 0: draft [1,2,3]; verify argmax
        // [6,7,0,0] → pos 0 rejects (6 != 1) and the residual (6) is emitted.
        // Emits exactly 1 residual, K == 0, rollback (0,3) so the dispatch
        // delta `0 - 3 = -3` rewinds the full window, and replay re-runs only
        // the anchor (`[last_committed]`, len 1).
        let ScriptedCycle {
            outcome,
            ledger,
            commit_payload,
            ..
        } = run_scripted_cycle_full(
            8,
            4,
            vec![1, 2, 3],
            vec![6, 7, 0, 0],
            3,
            0,
            None,
            MtpCommitAnchor::IncludeAnchor,
        );
        assert_eq!(
            outcome.tokens,
            vec![6],
            "all-reject emits exactly 1 residual"
        );
        assert_eq!(outcome.effective_depth, 3, "all 3 drafts were still built");
        assert_eq!(commit_k(&ledger), 0, "K == 0 on first-position reject");
        assert_eq!(
            commit_payload,
            Some((vec![0u32, 6], 0)),
            "all-reject IncludeAnchor commits [last_committed, residual]"
        );
        assert_eq!(
            ledger,
            vec![
                Call::DraftStep,
                Call::DraftStep,
                Call::DraftStep,
                Call::SnapshotMainLinear,
                Call::VerifyStep { depth: 3 },
                Call::CommitMtp {
                    anchor: MtpCommitAnchor::IncludeAnchor,
                    k: 0,
                },
                Call::Rollback {
                    accepted: 0,
                    depth: 3,
                },
                Call::RestoreAndReplayMain { accepted: 1 },
            ],
            "all-reject: drafts → snapshot → verify → commit → rollback → replay(anchor only)"
        );
    }

    #[test]
    fn run_mtp_cycle_depth_one_degenerates() {
        // depth 1: 1 draft + 1 verify slot. Full accept → 2 tokens
        // [draft, bonus]; K == 1; verify already advanced the linear state so
        // there is NO replay.
        let ScriptedCycle {
            outcome,
            ledger,
            commit_payload,
            ..
        } = run_scripted_cycle_full(
            8,
            4,
            vec![5],
            vec![5, 7],
            1,
            0,
            None,
            MtpCommitAnchor::IncludeAnchor,
        );
        assert_eq!(
            outcome.tokens,
            vec![5, 7],
            "depth-1 full accept = [draft, bonus]"
        );
        assert_eq!(outcome.effective_depth, 1);
        assert_eq!(commit_k(&ledger), 1);
        assert_eq!(
            commit_payload,
            Some((vec![0u32, 5, 7], 1)),
            "depth-1 full-accept commits [last_committed, draft, bonus]"
        );
        assert!(
            !ledger
                .iter()
                .any(|c| matches!(c, Call::RestoreAndReplayMain { .. })),
            "depth-1 full accept must NOT replay"
        );
    }

    #[test]
    fn run_mtp_cycle_partial_reject_at_pos2_reports_k2() {
        // depth 3, reject at position 2: draft [1,2,3]; verify argmax
        // [1,2,6,0] → pos 0,1 accept, pos 2 rejects (6 != 3), residual 6.
        // Emits [1,2,6] (2 accepted drafts + residual). K == 2 (NOT 3 — the
        // residual has no draft K/V slot and is excluded), rollback (2,3),
        // replay = [anchor, d_0, d_1] (len 3). Locks the off-by-one regression.
        let ScriptedCycle {
            outcome,
            ledger,
            commit_payload,
            ..
        } = run_scripted_cycle_full(
            16,
            4,
            vec![1, 2, 3],
            vec![1, 2, 6, 0],
            3,
            0,
            None,
            MtpCommitAnchor::IncludeAnchor,
        );
        assert_eq!(
            outcome.tokens,
            vec![1, 2, 6],
            "2 accepted drafts + residual"
        );
        assert_eq!(
            commit_k(&ledger),
            2,
            "K == accepted-draft prefix length (residual excluded)"
        );
        assert_eq!(
            commit_payload,
            Some((vec![0u32, 1, 2, 6], 2)),
            "partial-reject IncludeAnchor commits [last_committed, accepted drafts, residual]"
        );
        assert_eq!(
            ledger,
            vec![
                Call::DraftStep,
                Call::DraftStep,
                Call::DraftStep,
                Call::SnapshotMainLinear,
                Call::VerifyStep { depth: 3 },
                Call::CommitMtp {
                    anchor: MtpCommitAnchor::IncludeAnchor,
                    k: 2,
                },
                Call::Rollback {
                    accepted: 2,
                    depth: 3,
                },
                Call::RestoreAndReplayMain { accepted: 3 },
            ],
            "partial-reject K=2: replay [anchor, d_0, d_1] (len 3)"
        );
    }

    // ---- DENSE stochastic `accept_with_residual` branch (T=1.0) ----
    // The tests above force the T=0 sparse-accept fast path. These three drive
    // the same accept / partial-reject / all-reject contracts through the DENSE
    // `accept_with_residual` branch instead (T=1.0, exact-one-hot logits →
    // deterministic by construction), restoring the engine-level coverage the
    // deleted `mod mtp_cycle_tests` provided at T=1.0. Each asserts (via
    // `assert_dense_branch`) that it really took the dense path — the accept
    // loop ran and NEITHER sparse fast path did.

    #[test]
    fn run_mtp_cycle_dense_full_accept_depth3() {
        // depth 3, every draft accepted through the DENSE accept ratio
        // (draft logits == verify logits at each accept position → ratio
        // exactly 1.0). Commits 3 drafts + bonus, K == depth, no replay.
        let r = run_scripted_cycle_dense(
            8,
            4,
            vec![3, 4, 5],
            vec![3, 4, 5, 6],
            3,
            3,
            MtpCommitAnchor::IncludeAnchor,
        );
        assert_dense_branch(&r, "dense_full_accept");
        assert_eq!(
            r.outcome.tokens,
            vec![3, 4, 5, 6],
            "dense full accept emits 3 accepted drafts + bonus"
        );
        assert_eq!(r.outcome.effective_depth, 3);
        assert_eq!(
            r.commit_payload,
            Some((vec![3u32, 3, 4, 5, 6], 3)),
            "dense full-accept IncludeAnchor commits [last_committed, all drafts, bonus]"
        );
        assert_eq!(
            r.ledger,
            vec![
                Call::DraftStep,
                Call::DraftStep,
                Call::DraftStep,
                Call::SnapshotMainLinear,
                Call::VerifyStep { depth: 3 },
                Call::CommitMtp {
                    anchor: MtpCommitAnchor::IncludeAnchor,
                    k: 3,
                },
                Call::Rollback {
                    accepted: 3,
                    depth: 3,
                },
            ],
            "dense full-accept: 3 drafts → snapshot → verify → commit → rollback (no replay)"
        );
    }

    #[test]
    fn run_mtp_cycle_dense_partial_reject_at_pos2() {
        // depth 3, dense accept at pos 0,1 (ratio 1.0), reject at pos 2 (target
        // mass sits on verifier argmax 6, so the dense reject + residual draw
        // lands on 6). Emits [1,2,6], K == 2, rollback (2,3), replay len 3.
        let r = run_scripted_cycle_dense(
            16,
            4,
            vec![1, 2, 3],
            vec![1, 2, 6, 0],
            3,
            0,
            MtpCommitAnchor::IncludeAnchor,
        );
        assert_dense_branch(&r, "dense_partial_reject");
        assert_eq!(
            r.outcome.tokens,
            vec![1, 2, 6],
            "dense partial-reject: 2 accepted drafts + residual"
        );
        assert_eq!(
            r.commit_payload,
            Some((vec![0u32, 1, 2, 6], 2)),
            "dense partial-reject IncludeAnchor commits [last_committed, accepted drafts, residual]"
        );
        assert_eq!(
            r.ledger,
            vec![
                Call::DraftStep,
                Call::DraftStep,
                Call::DraftStep,
                Call::SnapshotMainLinear,
                Call::VerifyStep { depth: 3 },
                Call::CommitMtp {
                    anchor: MtpCommitAnchor::IncludeAnchor,
                    k: 2,
                },
                Call::Rollback {
                    accepted: 2,
                    depth: 3,
                },
                Call::RestoreAndReplayMain { accepted: 3 },
            ],
            "dense partial-reject K=2: replay [anchor, d_0, d_1] (len 3)"
        );
    }

    #[test]
    fn run_mtp_cycle_dense_all_reject_emits_residual() {
        // depth 3, dense reject at position 0 (target mass on verifier argmax
        // 6 → residual draw is 6). Emits exactly 1 residual, K == 0, rollback
        // (0,3), replay re-runs only the anchor (len 1).
        let r = run_scripted_cycle_dense(
            8,
            4,
            vec![1, 2, 3],
            vec![6, 7, 0, 0],
            3,
            0,
            MtpCommitAnchor::IncludeAnchor,
        );
        assert_dense_branch(&r, "dense_all_reject");
        assert_eq!(
            r.outcome.tokens,
            vec![6],
            "dense all-reject emits exactly 1 residual"
        );
        assert_eq!(
            r.commit_payload,
            Some((vec![0u32, 6], 0)),
            "dense all-reject IncludeAnchor commits [last_committed, residual]"
        );
        assert_eq!(
            r.ledger,
            vec![
                Call::DraftStep,
                Call::DraftStep,
                Call::DraftStep,
                Call::SnapshotMainLinear,
                Call::VerifyStep { depth: 3 },
                Call::CommitMtp {
                    anchor: MtpCommitAnchor::IncludeAnchor,
                    k: 0,
                },
                Call::Rollback {
                    accepted: 0,
                    depth: 3,
                },
                Call::RestoreAndReplayMain { accepted: 1 },
            ],
            "dense all-reject: drafts → snapshot → verify → commit → rollback → replay(anchor only)"
        );
    }

    #[test]
    fn run_mtp_cycle_ev_depth_gate_shortens_effective_depth() {
        // Caller requests depth 3, but the EV cost model (accept ewma
        // [0.70, 0.10, ..], `min_extra_accept` 0.30) votes against deepening
        // past `base_depth = 1`, so verify/rollback/commit must all use the
        // shortened `effective_depth = 1`. `for_test` leaves `allow_deepen`
        // true, so the cost model is the sole gate (matches the original
        // closure-driven contract test).
        let ev = ExpectedValueDepthPolicy::for_test(3, 1, [0.70, 0.10, 0.05, 0.05, 0.05], 0.30);
        let ScriptedCycle {
            outcome,
            ledger,
            commit_payload,
            ..
        } = run_scripted_cycle_full(
            8,
            4,
            vec![1, 2, 3],
            vec![1, 4],
            3,
            0,
            Some(ev),
            MtpCommitAnchor::IncludeAnchor,
        );
        assert_eq!(outcome.requested_depth, 3);
        assert_eq!(
            outcome.effective_depth, 1,
            "EV gate shortened the cycle to one draft"
        );
        assert_eq!(
            outcome.tokens,
            vec![1, 4],
            "shortened full accept emits the accepted draft plus bonus"
        );
        assert_eq!(commit_k(&ledger), 1);
        assert_eq!(
            commit_payload,
            Some((vec![0u32, 1, 4], 1)),
            "commit payload matches the shortened verify window"
        );
        assert_eq!(
            ledger,
            vec![
                Call::DraftStep,
                Call::SnapshotMainLinear,
                Call::VerifyStep { depth: 1 },
                Call::CommitMtp {
                    anchor: MtpCommitAnchor::IncludeAnchor,
                    k: 1,
                },
                Call::Rollback {
                    accepted: 1,
                    depth: 1,
                },
            ],
            "EV-gated cycle: 1 draft → snapshot → verify(depth 1) → commit → rollback (no replay)"
        );
    }

    #[test]
    fn run_mtp_cycle_chained_full_accept_skips_anchor() {
        // Chained cycle (`SkipAlreadyCommittedAnchor`): the anchor was already
        // committed by the prior cycle, so a full-accept commit carries only
        // the newly emitted tokens `[d_0, d_1, d_2, bonus]` (K+1, no anchor).
        let ScriptedCycle {
            outcome,
            commit_payload,
            ..
        } = run_scripted_cycle_full(
            8,
            4,
            vec![1, 2, 3],
            vec![1, 2, 3, 4],
            3,
            0,
            None,
            MtpCommitAnchor::SkipAlreadyCommittedAnchor,
        );
        assert_eq!(outcome.tokens, vec![1, 2, 3, 4]);
        assert_eq!(
            commit_payload,
            Some((vec![1u32, 2, 3, 4], 3)),
            "chained full-accept commits only newly emitted tokens (no anchor)"
        );
    }

    #[test]
    fn run_mtp_cycle_chained_partial_reject_skips_anchor() {
        // Chained partial reject (K=2): commit carries only `[d_0, d_1,
        // residual]` — the anchor is not re-committed.
        let ScriptedCycle {
            outcome,
            commit_payload,
            ..
        } = run_scripted_cycle_full(
            8,
            4,
            vec![1, 2, 3],
            vec![1, 2, 6, 0],
            3,
            0,
            None,
            MtpCommitAnchor::SkipAlreadyCommittedAnchor,
        );
        assert_eq!(outcome.tokens, vec![1, 2, 6]);
        assert_eq!(
            commit_payload,
            Some((vec![1u32, 2, 6], 2)),
            "chained partial-reject must not re-commit the anchor token"
        );
    }

    #[test]
    fn run_mtp_cycle_chained_all_reject_single_residual() {
        // Chained all-reject (K=0): commit carries only the single residual
        // (`[residual]`, K+1 = 1) under the chained one-token commit path.
        let ScriptedCycle {
            outcome,
            commit_payload,
            ..
        } = run_scripted_cycle_full(
            8,
            4,
            vec![1, 2, 3],
            vec![6, 7, 0, 0],
            3,
            0,
            None,
            MtpCommitAnchor::SkipAlreadyCommittedAnchor,
        );
        assert_eq!(outcome.tokens, vec![6]);
        assert_eq!(
            commit_payload,
            Some((vec![6u32], 0)),
            "chained all-reject uses the one-token residual commit path"
        );
    }

    /// The T=0 safety gate that graduates `MLX_MTP_EV_ALLOW_DEEPEN`.
    ///
    /// Invariant: intra-cycle deepening can only EXTEND the committed sequence
    /// — it must NEVER mutate an already-committed prefix. The SAME
    /// deterministic T=0 cycle is run twice, differing only in `allow_deepen`
    /// (shallow `false` stops the gate at `base_depth = 1`; deep `true` extends
    /// it to `max_depth = 3`). `accept_ewma` pinned high + costs zeroed
    /// (`for_test`) so the cost model always votes to deepen when allowed —
    /// `allow_deepen` is the SOLE difference. With drafter/verifier argmaxes
    /// equal on every overlapping position both runs full-accept, and the
    /// shallow token/commit window MUST be a byte-identical prefix of the deep
    /// one.
    ///
    /// GREEN => the Rust accept/commit layer the EV gate controls is
    /// depth-invariant at T=0 => the deepen flag is safe. RED => a REAL safety
    /// violation; keep the flag OFF and root-cause. Do NOT weaken this test.
    #[test]
    fn run_mtp_cycle_ev_deepen_t0_committed_tokens_byte_identical() {
        // draft d_0=1, d_1=2, d_2=3; verifier argmax matches each plus a bonus
        // argmax (4) at the final slot — depth-independent (causal) per slot.
        let run = |allow_deepen: bool| -> ScriptedCycle {
            let mut ev =
                ExpectedValueDepthPolicy::for_test(3, 1, [0.99, 0.99, 0.99, 0.99, 0.99], 0.0);
            ev.set_allow_deepen(allow_deepen);
            run_scripted_cycle_full(
                8,
                4,
                vec![1, 2, 3],
                vec![1, 2, 3, 4],
                3,
                0,
                Some(ev),
                MtpCommitAnchor::IncludeAnchor,
            )
        };

        let shallow = run(false);
        let deep = run(true);

        // Fail closed: the byte-equivalence claim is only meaningful if both
        // runs drove the production T=0 sparse-accept commit path.
        assert!(
            shallow.ran_sparse && deep.ran_sparse,
            "both runs must exercise the T=0 sparse-accept commit path"
        );

        // Sanity: the flag actually changed the drafted depth (else vacuous).
        assert_eq!(
            shallow.outcome.effective_depth, 1,
            "allow_deepen=false stops the gate at base_depth=1"
        );
        assert_eq!(
            deep.outcome.effective_depth, 3,
            "allow_deepen=true extends the gate to max_depth on a full-accept chain"
        );

        assert_eq!(shallow.outcome.tokens, vec![1u32, 2]);
        assert_eq!(deep.outcome.tokens, vec![1u32, 2, 3, 4]);

        // THE SAFETY INVARIANT: the shallow emitted-token window is a
        // byte-identical PREFIX of the deep window — deepening only appends.
        assert!(
            deep.outcome
                .tokens
                .starts_with(&shallow.outcome.tokens[..1]),
            "deepen must not mutate the first committed token: shallow={:?} deep={:?}",
            shallow.outcome.tokens,
            deep.outcome.tokens
        );
        assert_eq!(
            shallow.outcome.tokens[1], deep.outcome.tokens[1],
            "shallow full-accept bonus@K_s equals the token deeper drafting commits at that slot"
        );

        // Commit payloads carry the same invariant (IncludeAnchor →
        // [last_committed, emitted...]).
        let (shallow_ids, shallow_k) = shallow.commit_payload.expect("shallow commit");
        let (deep_ids, deep_k) = deep.commit_payload.expect("deep commit");
        assert_eq!(shallow_ids, vec![0u32, 1, 2]);
        assert_eq!(deep_ids, vec![0u32, 1, 2, 3, 4]);
        assert_eq!(shallow_k, 1);
        assert_eq!(deep_k, 3);
        assert_eq!(
            shallow_ids[..2],
            deep_ids[..2],
            "committed anchor + first accepted draft are byte-identical across depths"
        );
    }

    // -----------------------------------------------------------------------
    // `run_mtp_turn` integration tests — DRIVE the relocated OUTER MTP loop
    // end-to-end over a scripted `MockMtpBackend` (whose `begin_mtp_decode`
    // hands back a turn-scripted `MockMtpStepper`). NO model, NO Metal. The
    // mock argmaxes are fake, so the assertions are STRUCTURAL: the emitted
    // token sequence, the finish_reason, the `last_in_cache` / `desynced`
    // outs, and the `MtpStepper` call ledger (forward / begin_cycle / draft /
    // snapshot / verify / commit / rollback / rollback_unemitted /
    // take_replay_error / into_desynced) the engine drives.
    //
    // Chained cycles are forced OFF (`MLX_MTP_CHAINED_CYCLES=0`) so the loop
    // runs its deterministic always-Step-A mode — the chained path is a
    // GPU-gen-gated cross-cycle optimization whose ledger shape depends on
    // host hardware, which would make these structural assertions flaky.
    // -----------------------------------------------------------------------

    use std::rc::Rc;
    use std::sync::Arc;
    use std::time::Instant;

    use crate::engine::backend::{
        ChatBackend, FinalizeArgs, MtpBackend, MtpTurnSetup, ResetScope, SaveStateArgs, TurnSetup,
    };
    use crate::engine::penalties::ReasoningTracker;
    use crate::engine::types::ChatResult;
    use crate::stream::{DeviceType, Stream};
    use crate::tokenizer::Qwen3Tokenizer;

    /// Serializes the `MLX_MTP_CHAINED_CYCLES` set + the `mtp_chained_cycles_enabled`
    /// OnceLock read across the turn tests so the forced-OFF value is the one
    /// that caches. The OnceLock has no in-process caller other than
    /// `run_mtp_turn`, so a single deterministic "0" write before the first
    /// `run_mtp_turn` of the binary pins it OFF for all three tests.
    static CHAINED_OFF_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Force chained cycles OFF for a turn test, returning the lock guard that
    /// keeps the env write + the loop's `mtp_chained_cycles_enabled` read
    /// serialized.
    fn force_chained_off() -> std::sync::MutexGuard<'static, ()> {
        let guard = CHAINED_OFF_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        // SAFETY: the lock serializes this write with every other turn test's
        // write + read; all write the SAME value, and no production code in
        // the lib-test binary writes this var.
        unsafe {
            std::env::set_var("MLX_MTP_CHAINED_CYCLES", "0");
        }
        guard
    }

    /// A never-constructed [`DecodeStep`] so `MockMtpBackend` can satisfy the
    /// `ChatBackend::Decode<'a>: DecodeStep` bound. `begin_decode` is never
    /// called on the MTP path, so the methods are unreachable in practice —
    /// they propagate an error instead of panicking (no `unreachable!`).
    struct NeverDecode;
    impl crate::engine::backend::DecodeStep for NeverDecode {
        fn forward(&mut self, _input_ids: &MxArray) -> Result<(MxArray, bool)> {
            Err(Error::from_reason("NeverDecode::forward must not run"))
        }
        fn eval_step(&mut self, _next_token: &MxArray, _logits: &MxArray, _budget_forced: bool) {}
    }

    /// Scripted [`MtpBackend`] for the `run_mtp_turn` integration tests. Holds
    /// the turn script + the canned terminal outs; `begin_mtp_decode` builds a
    /// fresh `MockMtpStepper` wired to the shared ledger so the test can read
    /// the call sequence AFTER `run_mtp_turn` consumes the stepper.
    struct MockMtpBackend {
        vocab: i64,
        hidden: i64,
        step_a_tokens: Vec<i32>,
        cycles: Vec<CycleArgmax>,
        /// Canned [`MtpStepper::into_desynced`] terminal (paged MUST be false;
        /// flat/MoE may set true on a mid-cycle stop).
        desynced: bool,
        /// Shared ledger the constructed stepper mirrors every call into.
        ledger: Rc<RefCell<Vec<Call>>>,
        /// Records that `begin_mtp_decode` ran exactly once.
        begin_calls: std::cell::Cell<usize>,
    }

    impl MockMtpBackend {
        fn new(
            vocab: i64,
            hidden: i64,
            step_a_tokens: Vec<i32>,
            cycles: Vec<CycleArgmax>,
            desynced: bool,
        ) -> Self {
            Self {
                vocab,
                hidden,
                step_a_tokens,
                cycles,
                desynced,
                ledger: Rc::new(RefCell::new(Vec::new())),
                begin_calls: std::cell::Cell::new(0),
            }
        }

        fn ledger_snapshot(&self) -> Vec<Call> {
            self.ledger.borrow().clone()
        }
    }

    // Minimal `ChatBackend` surface — `run_mtp_turn` calls NONE of these (it
    // only drives `begin_mtp_decode` + the `MtpStepper`), so the never-reached
    // methods propagate an error instead of panicking.
    impl ChatBackend for MockMtpBackend {
        fn tokenizer(&self) -> Result<Arc<Qwen3Tokenizer>> {
            Err(Error::from_reason(
                "MockMtpBackend::tokenizer must not run (run_mtp_turn never reads it)",
            ))
        }
        fn family_name(&self) -> &'static str {
            "mock_mtp"
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
            Err(Error::from_reason("MockMtpBackend::prefill must not run"))
        }

        type Decode<'a>
            = NeverDecode
        where
            Self: 'a;

        fn begin_decode(&mut self, _turn: &TurnSetup<'_>) -> Result<Self::Decode<'_>> {
            Err(Error::from_reason(
                "MockMtpBackend::begin_decode must not run on the MTP path",
            ))
        }

        fn finalize_turn(&self, _args: FinalizeArgs<'_>) -> Result<ChatResult> {
            Err(Error::from_reason(
                "MockMtpBackend::finalize_turn must not run (run_mtp_turn never finalizes)",
            ))
        }
    }

    impl MtpBackend for MockMtpBackend {
        type MtpDecode<'a>
            = MockMtpStepper
        where
            Self: 'a;

        fn begin_mtp_decode(&mut self, _setup: &MtpTurnSetup<'_>) -> Result<Self::MtpDecode<'_>> {
            self.begin_calls.set(self.begin_calls.get() + 1);
            let mut step = MockMtpStepper::with_turn(
                self.vocab,
                self.hidden,
                self.step_a_tokens.clone(),
                self.cycles.clone(),
            );
            step.desynced = self.desynced;
            step.shared_ledger = Some(Rc::clone(&self.ledger));
            Ok(step)
        }
    }

    struct TurnOut {
        generated: Vec<u32>,
        finish_reason: String,
        last_in_cache: bool,
        desynced: bool,
        ledger: Vec<Call>,
    }

    /// Drive `run_mtp_turn` over the scripted backend with greedy T=0 params
    /// (sparse-accept forced ON so the deterministic argmax branch runs). The
    /// prefill seed `y` is `first_token`; `max_new_tokens` bounds the budget.
    fn drive_turn(
        backend: &mut MockMtpBackend,
        first_token: u32,
        max_new_tokens: i32,
        eos_id: u32,
        depth: usize,
    ) -> TurnOut {
        let _force_sparse = ForceSparseAcceptGuard::force(true);
        let params = {
            let mut p = greedy_params();
            p.max_new_tokens = max_new_tokens;
            p.mtp_depth = depth;
            p
        };
        let mut tracker = ReasoningTracker::new(false, None, None);
        let mut profiler = crate::decode_profiler::DecodeProfiler::new("mtp_turn_test", "test");
        let mut generated: Vec<u32> = Vec::new();
        let mut token_history: Vec<u32> = Vec::new();
        let mut finish_reason = String::from("length");
        let mut first_token_instant: Option<Instant> = None;
        let mut rng = rand::rng();
        let y = MxArray::from_int32(&[first_token as i32], &[1])
            .unwrap_or_else(|e| panic!("y construction: {}", e.reason));
        let generation_stream = Stream::new(DeviceType::Gpu);

        let outcome = run_mtp_turn(
            backend,
            &mut rng,
            MtpTurnArgs {
                y,
                depth,
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
                prompt_hidden: None,
                prompt_hidden_ids: None,
                prompt_hidden_position_base: 0,
            },
            // The whole-turn mock tests drive the SYNC path.
            None,
        )
        .unwrap_or_else(|e| panic!("run_mtp_turn failed: {}", e.reason));

        // The loop keeps token_history in lockstep with generated_tokens.
        assert_eq!(token_history, generated, "history must mirror generated");

        TurnOut {
            generated,
            finish_reason,
            last_in_cache: outcome.last_in_cache,
            desynced: outcome.desynced,
            ledger: backend.ledger_snapshot(),
        }
    }

    /// Count ledger entries matching a predicate.
    fn count(ledger: &[Call], pred: impl Fn(&Call) -> bool) -> usize {
        ledger.iter().filter(|c| pred(c)).count()
    }

    #[test]
    fn run_mtp_turn_normal_full_length_run() {
        let _chained_off = force_chained_off();
        // depth 2, every cycle FULL-ACCEPT, no EOS — a genuine MULTI-cycle run
        // that walks to the LENGTH budget. Step A emits the prefill seed (3)
        // first; then each outer iteration's Step A forwards the prior accepted
        // token and emits a NEW non-EOS token, and the cycle emits its
        // depth+1 accepted-drafts+bonus. vocab 16; no token is the EOS id (15).
        //
        // Per-cycle full accept: verify_argmax == draft_argmax at positions
        // 0,1 and a bonus at position 2.
        let cycle = CycleArgmax {
            draft_argmax: vec![4, 5],
            verify_argmax: vec![4, 5, 6],
        };
        // Step A tokens for iterations: 7, 8, 9, ... (non-EOS). Plenty of
        // cycles scripted; the length cap stops the run.
        let mut backend = MockMtpBackend::new(
            16,
            4,
            vec![7, 8, 9, 10, 11, 12],
            vec![cycle; 8],
            /* desynced */ false,
        );

        // Budget 8 drives TWO full outer iterations (the near-tail cap shrinks
        // the 2nd cycle's depth so the run lands EXACTLY on the budget — a
        // clean top-of-loop length exit, not a mid-cycle truncation):
        //   gen: [3]                  (initial seed emit)
        //   iter0 Step A -> 7         gen=[3,7]
        //   iter0 cycle(depth 2) full accept -> 4,5,bonus 6   gen=[3,7,4,5,6]
        //   iter1 Step A -> 8         gen=[3,7,4,5,6,8]
        //   iter1 cycle: remaining=2 -> near-tail cap depth 1 -> draft 4,
        //                verify [4,5] full accept -> 4,bonus 5  gen=[...,4,5]
        //   iter2 top: len 8 >= 8 -> length stop (clean, last_in_cache true).
        let out = drive_turn(&mut backend, 3, 8, 15, 2);

        assert_eq!(
            out.generated,
            vec![3, 7, 4, 5, 6, 8, 4, 5],
            "seed + 2 full outer iterations (Step A + full-accept cycle each)"
        );
        assert_eq!(out.finish_reason, "length");
        // Clean length exit at the top of the loop (no mid-cycle truncation):
        // the boundary token was emitted by a completed cycle, no
        // rollback_unemitted, no desync.
        assert!(out.last_in_cache, "clean length exit keeps last_in_cache");
        assert!(!out.desynced, "no mid-cycle stop -> not desynced");
        assert_eq!(backend.begin_calls.get(), 1, "exactly one begin_mtp_decode");

        // Ledger: TWO Step-A forwards + TWO cycles (begin_cycle / snapshot /
        // verify / commit / rollback each). The 2nd cycle is depth-1 (near-tail
        // cap), so DraftStep count is 2 (cycle0) + 1 (cycle1) = 3. The terminal
        // take_replay_error + into_desynced fire exactly once each.
        assert_eq!(
            count(&out.ledger, |c| matches!(c, Call::ForwardWithHidden)),
            2,
            "two Step-A forwards (chained OFF -> Step A every outer iteration)"
        );
        assert_eq!(
            count(&out.ledger, |c| matches!(c, Call::BeginCycle { .. })),
            2
        );
        assert_eq!(count(&out.ledger, |c| matches!(c, Call::DraftStep)), 3);
        assert_eq!(
            count(&out.ledger, |c| matches!(c, Call::VerifyStep { .. })),
            2
        );
        assert_eq!(
            count(&out.ledger, |c| matches!(c, Call::CommitMtp { .. })),
            2
        );
        assert_eq!(
            count(&out.ledger, |c| matches!(c, Call::Rollback { .. })),
            2
        );
        assert_eq!(
            count(&out.ledger, |c| matches!(c, Call::RollbackUnemitted { .. })),
            0,
            "clean length exit -> no rollback_unemitted"
        );
        assert_eq!(
            count(&out.ledger, |c| matches!(c, Call::TakeReplayError)),
            1,
            "engine surfaces full-accept replay error once post-loop"
        );
        assert_eq!(
            count(&out.ledger, |c| matches!(c, Call::IntoDesynced)),
            1,
            "engine consumes the stepper's desync out once"
        );
    }

    #[test]
    fn run_mtp_turn_early_eos_stop() {
        let _chained_off = force_chained_off();
        // depth 2. The FIRST cycle's verify emits the EOS id (15) at accept
        // position 1, so the emit loop commits the accepted draft (4) then the
        // EOS (15) and STOPS with finish_reason "stop". last_in_cache is false:
        // the EOS is the cycle's boundary token (its K/V is only laid down by
        // the next cycle's Step A, which never runs).
        let cycle = CycleArgmax {
            // draft 0 -> 4 (accepted: verify_argmax[0]==4); draft 1 -> 5
            // (REJECTED: verify_argmax[1]==15 != 5) so the residual 15 (EOS)
            // is emitted and the accept loop stops. Emitted cycle tokens:
            // [4, 15].
            draft_argmax: vec![4, 5],
            verify_argmax: vec![4, 15, 6],
        };
        let mut backend = MockMtpBackend::new(
            16,
            4,
            vec![7, 8, 9],
            vec![cycle; 4],
            /* desynced */ false,
        );

        // gen: [3] (seed), iter0 Step A -> 7 (=[3,7]), cycle emits 4 then 15:
        //   push 4 (=[3,7,4]); push 15 -> EOS -> stop mid-cycle? No: the cycle
        //   emitted ALL its tokens [4,15] (cycle_emitted == 2 == len), so this
        //   is a stop on the LAST cycle token -> last_in_cache = (2 < 2) = false,
        //   no rollback_unemitted (unemitted == 0).
        let out = drive_turn(&mut backend, 3, 64, 15, 2);

        assert_eq!(
            out.generated,
            vec![3, 7, 4, 15],
            "seed + Step-A(7) + cycle(accept 4, residual EOS 15)"
        );
        assert_eq!(out.finish_reason, "stop");
        assert!(
            !out.last_in_cache,
            "EOS is the cycle's unforwarded boundary token -> not in cache"
        );
        assert!(!out.desynced, "stop on the LAST cycle token -> no desync");
        // The EOS landed as the cycle's final emitted token, so the emit loop
        // ran to completion (no unemitted remainder).
        assert_eq!(
            count(&out.ledger, |c| matches!(c, Call::RollbackUnemitted { .. })),
            0,
            "EOS on the last cycle token -> emit loop completed, no rollback_unemitted"
        );
        assert_eq!(
            count(&out.ledger, |c| matches!(c, Call::VerifyStep { .. })),
            1,
            "stopped during the first cycle"
        );
        assert_eq!(count(&out.ledger, |c| matches!(c, Call::IntoDesynced)), 1);
    }

    #[test]
    fn run_mtp_turn_mid_cycle_stop_rolls_back_unemitted() {
        let _chained_off = force_chained_off();
        // A TRUE mid-cycle stop: the cycle FULLY accepts depth-3 drafts but an
        // EOS lands as the 2nd ACCEPTED draft (not the last token), so the emit
        // loop breaks BEFORE the cycle's remaining tokens — leaving an
        // unemitted remainder the engine rolls back via rollback_unemitted, and
        // the flat stepper reports desynced.
        //
        // (A length-budget mid-cycle stop is impossible by construction: the
        // near-tail depth cap shrinks the cycle so it emits exactly the
        // remaining budget, never overshooting. EOS / repetition / cancel are
        // the only mid-cycle stop sources — here EOS as an accepted draft.)
        //
        // drafts [4, 15(EOS), 6]; verify_argmax [4, 15, 6, 7] → all 3 accepted
        // (4==4, 15==15, 6==6) + bonus 7, so outcome.tokens = [4, 15, 6, 7].
        let cycle = CycleArgmax {
            draft_argmax: vec![4, 15, 6],
            verify_argmax: vec![4, 15, 6, 7],
        };
        // desynced = true: the flat/MoE stepper sets its desync flag when
        // rollback_unemitted fires with unemitted > 0.
        let mut backend = MockMtpBackend::new(
            16,
            4,
            vec![9, 10, 11],
            vec![cycle; 4],
            /* desynced */ true,
        );

        // gen: [3] (seed), iter0 Step A -> 9 (=[3,9]). A generous budget (64)
        // keeps the near-tail cap from shrinking depth, so the full depth-3
        // cycle runs: outcome.tokens = [4, 15, 6, 7]. Emit loop:
        //   push 4  -> gen=[3,9,4]
        //   push 15 -> EOS -> "stop", cycle_emitted = 2 < tokens.len 4 ->
        //              last_in_cache = true; hit_stop; break.
        //   unemitted = 4 - 2 = 2 -> rollback_unemitted(2).
        let out = drive_turn(&mut backend, 3, 64, 15, 3);

        assert_eq!(
            out.generated,
            vec![3, 9, 4, 15],
            "seed + Step-A(9) + 2 of the cycle's 4 tokens before the mid-cycle EOS"
        );
        assert_eq!(out.finish_reason, "stop");
        // The EOS (15) is an emitted-but-not-last cycle token whose K/V verify
        // wrote, and the boundary (bonus 7) was never emitted — so the last
        // EMITTED token IS in cache: cycle_emitted (2) < tokens.len (4).
        assert!(
            out.last_in_cache,
            "mid-cycle stop before the boundary keeps the last emitted token in cache"
        );
        assert!(
            out.desynced,
            "mid-cycle stop with unemitted>0 leaves the flat caches desynced"
        );
        // rollback_unemitted fired exactly once with the 2-token remainder
        // ([6, 7], the accepted-but-unemitted cycle tail).
        assert_eq!(
            out.ledger
                .iter()
                .filter_map(|c| match c {
                    Call::RollbackUnemitted { unemitted } => Some(*unemitted),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            vec![2],
            "rollback_unemitted(2) — the 2 accepted-but-unemitted cycle tokens"
        );
        assert_eq!(count(&out.ledger, |c| matches!(c, Call::IntoDesynced)), 1);
    }
}
