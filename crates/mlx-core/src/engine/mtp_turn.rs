//! Engine-owned MTP (Multi-Token Prediction) propose/verify whole-turn
//! path — the MTP analog of [`crate::engine::paged_turn`]. Families opt in
//! via [`crate::engine::backend::MtpBackend`]; their
//! `ChatBackend::mtp_turn` body becomes `Some(run_mtp_turn(self, args))`.
//!
//! SCAFFOLD STEP: the relocated `decode_loop_mtp!` outer body
//! (`run_mtp_turn`) and the relocated `run_mtp_cycle_inner` (`run_mtp_cycle`)
//! land in later steps. Today this module carries ONLY the
//! [`MtpStepper`](crate::engine::backend::MtpStepper) contract's test
//! harness — a scripted [`MockMtpStepper`] double + call-ledger unit tests
//! that PROVE the trait + GAT lifetimes + the strictly-sequential
//! `&mut self` borrow model compile and are usable. Nothing in production
//! calls this module yet, so the families' MTP behavior is byte-identical.

use napi::bindgen_prelude::*;

use crate::array::MxArray;
use crate::engine::backend::MtpStepper;
use crate::engine::params::ChatParams;
use crate::engine::penalties::apply_all_penalties;
use crate::models::qwen3_5::mtp_decode::{
    MtpCommitAnchor, MtpCycleOutcome, MtpVerifyOutput, mtp_batch_target_arrays_enabled,
    mtp_defer_verify_hidden_eval, mtp_draft_sampling_config, mtp_greedy_argmax_only_verify_enabled,
    mtp_native_sparse_verify_enabled, mtp_target_distribution_first_enabled, mtp_trace_acceptance,
    mtp_verify_async_eval, mtp_verify_top1_check_enabled, sparse_accept_gate,
    trace_acceptance_dense, trace_acceptance_emit, trace_acceptance_greedy,
    trace_acceptance_sparse,
};
use crate::sampling;

use crate::engine::decode::{mtp_trace_logits, trace_top2};

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
        // for MLX's lazy compiled cache writes.
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
    // runs its D+1 sequential forwards. Verify mutates `g_compiled_caches`
    // in place; on rejection we restore from this snapshot and replay only
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
        // hiddens ride on the same compiled graph; we only eval the
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
    // `accepted_tokens`, advancing `g_mtp_committed_len` by the number of
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
    use crate::models::qwen3_5::mtp_decode::{
        ForceSparseAcceptGuard, MtpCommitAnchor, MtpVerifyOutput,
    };
    use crate::sampling::SamplingConfig;

    use super::run_mtp_cycle;

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
            });
            s
        }

        fn record(&self, c: Call) {
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
        let mut row = vec![0.0f32; vocab as usize];
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
            // (logits [1,1], hidden [1,1], needs_squeeze) — eager shape.
            Ok((lazy_scalar(0.0), lazy_scalar(0.0), true))
        }

        fn draft_step(
            &mut self,
            _prev_h: &MxArray,
            _prev_emb: &MxArray,
        ) -> Result<(MxArray, MxArray)> {
            self.record(Call::DraftStep);
            match self.cycle.as_ref() {
                Some(c) => {
                    // h_next [1,1,hidden]; draft_logits [1,vocab] whose argmax
                    // (the T=0 draw) is the scripted `draft_argmax[step]`.
                    let i = c.next_draft.get();
                    c.next_draft.set(i + 1);
                    let argmax_id = c.draft_argmax.get(i).copied().unwrap_or(0);
                    let h_next =
                        MxArray::from_float32(&vec![0.0f32; c.hidden as usize], &[1, 1, c.hidden])?;
                    let draft_logits =
                        MxArray::from_float32(&logits_row(c.vocab, argmax_id), &[1, c.vocab])?;
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
            match self.cycle.as_ref() {
                Some(c) => {
                    // logits [1, depth+1, vocab] with per-position argmax driven
                    // by `verify_argmax`; hiddens [1, depth+1, hidden].
                    let rows = depth + 1;
                    let mut flat: Vec<f32> = Vec::with_capacity(rows * c.vocab as usize);
                    for j in 0..rows {
                        let argmax_id = c.verify_argmax.get(j).copied().unwrap_or(0);
                        flat.extend(logits_row(c.vocab, argmax_id));
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
            // The K+2 committed-sequence shape the real commit consumes.
            assert_eq!(
                committed_ids.len(),
                k_accepted + 2,
                "committed_ids is [last_committed, d_0..d_{{K-1}}, boundary] (K+2)"
            );
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
    /// through the trait, EXACTLY in the order `run_mtp_cycle_inner` calls
    /// `ops.*` today (Step A forward → begin_cycle → D draft steps →
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
             sequence in the order run_mtp_cycle_inner calls ops.* today"
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

    /// Run one scripted `run_mtp_cycle` over the canned mock. Returns the
    /// cycle outcome, the recorded call ledger, and the `CommitMtp` ledger
    /// entry's accepted-draft count `k` (the only place the cycle surfaces K).
    /// Forces the sparse-accept gate ON so the deterministic T=0 branch runs
    /// regardless of `MLX_MTP_SPARSE_ACCEPT`.
    fn run_scripted_cycle(
        vocab: i64,
        hidden: i64,
        draft_argmax: Vec<i32>,
        verify_argmax: Vec<i32>,
        depth: usize,
        last_committed_id: u32,
    ) -> (
        crate::models::qwen3_5::mtp_decode::MtpCycleOutcome,
        Vec<Call>,
    ) {
        let _force = ForceSparseAcceptGuard::force(true);
        let mut step = MockMtpStepper::with_cycle(vocab, hidden, draft_argmax, verify_argmax);
        let params = greedy_params();
        let mut rng = rand::rng();
        let mut profiler = crate::decode_profiler::DecodeProfiler::new("mtp_test", "test");
        // Embedding weight is the mock's own `[vocab, hidden]` table.
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
            MtpCommitAnchor::IncludeAnchor,
        )
        .expect("scripted run_mtp_cycle must succeed");
        let ledger = step.snapshot();
        (outcome, ledger)
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
        let (outcome, ledger) = run_scripted_cycle(8, 4, vec![3, 4, 5], vec![3, 4, 5, 6], 3, 3);

        assert_eq!(
            outcome.tokens,
            vec![3, 4, 5, 6],
            "full accept emits 3 accepted drafts + bonus"
        );
        assert_eq!(outcome.requested_depth, 3);
        assert_eq!(outcome.effective_depth, 3);
        assert_eq!(commit_k(&ledger), 3, "K == effective_depth on full accept");

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
}
