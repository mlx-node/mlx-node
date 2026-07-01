//! Shared autoregressive decode-loop machinery.
//!
//! [`run_decode_loop`] is the engine's generic decode loop over the
//! [`DecodeStep`] backend trait — every family's standard chat flow
//! drives it via the session cores. The `decode_loop!` macro and its
//! `DecodeOps` closure bundle live in `models::qwen3_5::mtp_decode`;
//! their only consumers are the qwen3_5 dense/MoE MTP/vision whole-turn
//! cores.
//!
//! NOTE: `mtp_trace_logits` / `Top2` / `trace_top2` live HERE (not in
//! `models::qwen3_5::mtp_decode`) because they serve both this loop's
//! AR per-token trace diagnostics and the MTP path, which imports them
//! from this module.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use napi::bindgen_prelude::*;

use crate::array::MxArray;
use crate::decode_profiler::DecodeProfiler;
use crate::engine::backend::{ChunkSink, DecodeStep, StreamEmitter};
use crate::engine::params::ChatParams;
use crate::engine::penalties::{ReasoningTracker, apply_all_penalties};
use crate::stream::{Stream, StreamContext};
use crate::tokenizer::Qwen3Tokenizer;

// Diagnostic — per-committed-token top-2 logit trace.
//
// `MLX_MTP_TRACE_LOGITS=1` (or `true` / `on`) enables an env-gated
// per-token logit trace emitted to stderr. For each committed decode
// token it logs the position index, the committed token id, and the
// top-2 (token id + logit value) of the forward that produced it:
//   * the AR `decode_loop!` logs the single-token decode forward;
//   * the engine's `run_mtp_cycle` logs the batched verify forward, per
//     verify slot.
//
// The trace exists to resolve whether an AR-vs-MTP argmax flip is a
// benign batched-vs-single kernel near-tie (both forwards have the
// SAME top-2 set with logits agreeing within bf16 epsilon) or a real
// verify-path bug (the verify forward computes a substantially
// different logit vector). Default OFF; read once per process and
// cached. Lines are prefixed `MTP_TRACE_LOGITS` for easy grep.
pub(crate) fn mtp_trace_logits() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| match std::env::var("MLX_MTP_TRACE_LOGITS") {
        Ok(v) => {
            let v = v.trim();
            v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("on")
        }
        Err(_) => false, // default OFF — diagnostic instrumentation
    })
}

/// Top-2 entries `(id, logit)` of a logits vector — used by the
/// `MLX_MTP_TRACE_LOGITS` diagnostic.
pub(crate) struct Top2 {
    pub top1_id: i32,
    pub top1_logit: f32,
    pub top2_id: i32,
    pub top2_logit: f32,
}

/// Compute the top-2 `(id, logit)` of a 1-D logits array.
///
/// `logits_1d` must be a `[vocab]` array (any float dtype — values are
/// read back as f32). Uses a descending sort of the indices via
/// `argsort` then a single `.eval()`; the two winning logit values are
/// read by flat index from an f32 copy of the logits. No `.unwrap()` /
/// `.expect()` — every fallible step propagates with `?`, so this is
/// safe to call from the decode path.
pub(crate) fn trace_top2(logits_1d: &MxArray, vocab: i64) -> Result<Top2> {
    use crate::array::DType;

    // argsort is ascending; the last two entries are the top-2.
    let order = logits_1d.argsort(Some(-1))?;
    let logits_f32 = logits_1d.astype(DType::Float32)?;
    order.eval();
    logits_f32.eval();

    let last = (vocab - 1).max(0) as usize;
    let second = (vocab - 2).max(0) as usize;
    let top1_id = order.item_at_int32(last)?;
    let top2_id = order.item_at_int32(second)?;
    let top1_logit = logits_f32.item_at_float32(top1_id as usize)?;
    let top2_logit = logits_f32.item_at_float32(top2_id as usize)?;
    Ok(Top2 {
        top1_id,
        top1_logit,
        top2_id,
        top2_logit,
    })
}

/// The incremental-detokenization stream type from the `tokenizers`
/// crate, instantiated with the wrapper types `tokenizers::Tokenizer`
/// uses (same concrete type `Qwen3Tokenizer::step_decode_stream`
/// accepts). `'t` is the borrow of the underlying tokenizer.
pub(crate) type TokDecodeStream<'t> = tokenizers::DecodeStream<
    't,
    tokenizers::ModelWrapper,
    tokenizers::NormalizerWrapper,
    tokenizers::PreTokenizerWrapper,
    tokenizers::PostProcessorWrapper,
    tokenizers::DecoderWrapper,
>;

/// Required arguments of [`run_decode_loop`]. The turn-constant
/// `embedding_weight` is captured by the [`DecodeStep`] impl at
/// `begin_decode` time, and the per-step ops are the `step` trait
/// object/impl.
pub(crate) struct DecodeLoopArgs<'a> {
    /// First generated token (sampled from the prefill logits). The loop
    /// takes ownership; its final reassignment is not observed by callers
    /// (see the `_final_sampled_token` note at the dense call site).
    pub y: MxArray,
    pub params: &'a ChatParams,
    pub reasoning_tracker: &'a mut ReasoningTracker,
    pub profiler: &'a mut DecodeProfiler,
    pub max_new_tokens: i32,
    pub eos_id: u32,
    /// Additional stop-token ids honored alongside `eos_id`. The session
    /// core computes the set ONCE per turn from
    /// [`crate::engine::backend::ChatBackend::extra_eos_ids`] — not per
    /// step. ChatML families pass their `generation_config.json` eos ids
    /// (empty when the checkpoint ships no such file); Gemma4 passes its
    /// model-config `eos_token_ids`. The stop check below combines this set
    /// with `eos_id` as a UNION (`token_id == eos_id ||
    /// extra_eos_ids.contains(&token_id)`) — see
    /// [`crate::engine::params::ModelGenerationDefaults`] for the full
    /// override order.
    pub extra_eos_ids: &'a [u32],
    /// Streaming-only ordering knob: check the stop set BEFORE
    /// cancellation/emission. From
    /// [`crate::engine::backend::ChatBackend::eos_before_emit`]; `false`
    /// is the ChatML emit-then-check order, `true` is lfm2's
    /// check-then-emit order. Ignored on non-streaming runs (no
    /// emission to order against).
    pub eos_before_emit: bool,
    pub generated_tokens: &'a mut Vec<u32>,
    pub token_history: &'a mut Vec<u32>,
    pub finish_reason: &'a mut String,
    pub first_token_instant: &'a mut Option<Instant>,
    pub report_perf: bool,
    pub generation_stream: Stream,
}

/// Streaming sub-block arguments of [`run_decode_loop`]. `'t` is the
/// tokenizer borrow backing the [`TokDecodeStream`].
///
/// `tokenizer` is the raw `tokenizers::Tokenizer`; call sites pass
/// `qwen3_tokenizer.inner()`.
pub(crate) struct StreamingCtx<'s, 't> {
    pub callback: &'s dyn ChunkSink,
    pub cancelled: &'s AtomicBool,
    pub decode_stream: &'s mut TokDecodeStream<'t>,
    pub tokenizer: &'t tokenizers::Tokenizer,
    pub streamed_text_len: &'s mut usize,
    pub last_is_reasoning: &'s mut bool,
    /// Per-family chunk emitter. EVERY committed token's incremental text
    /// is routed through [`StreamEmitter::on_token_text`] — the
    /// `include_reasoning` suppression gate lives in the emitter, so
    /// family emitters observe suppressed (and empty) texts too. From
    /// [`crate::engine::backend::ChatBackend::stream_emitter`], created
    /// once per turn by the session core.
    pub emitter: &'s mut dyn StreamEmitter,
}

/// Generic decode loop over a [`DecodeStep`].
///
/// Behavior matches the `decode_loop!` macro (still used by the
/// MTP/vision whole-turn cores in `models::qwen3_5::mtp_decode`) at the
/// engine defaults; the two must stay in lockstep. These seams each
/// default to that behavior:
///   * (a) the throttled every-32-step trace reads
///     `step.trace_offset()`; `None` skips the `tracing::info!` line
///     entirely. All steppers inherit the `None` default (the eager
///     forward mutates its Rust cache in place), so the line is dormant.
///     The line's family prefix comes from `step.trace_name()` (default
///     `"model"`).
///   * (b) the stop check matches `eos_id` OR any id in
///     `args.extra_eos_ids` (empty == a single-id check; Gemma4's config
///     eos set) and — gated on `args.eos_before_emit` — may run BEFORE
///     the streaming cancellation/emission (lfm2's order; default
///     `false` is the ChatML order). Both checks cover every committed
///     token including the first prefill-sampled one (the step-0 commit).
///   * (c) streaming emission is routed through
///     [`StreamingCtx::emitter`] — [`StreamEmitter::on_token_text`]
///     owns the `include_reasoning` suppression gate; the default
///     emitter is the raw inline chunk emission.
///
/// The rest: pipelined next-graph build, budget forcing via
/// [`ReasoningTracker`], [`apply_all_penalties`], `sampling::sample`,
/// EOS + `check_repetition_cutoff` stops, profiler begin/end/mark/step
/// calls, the `MLX_MTP_TRACE_LOGITS` diagnostic block, and the streaming
/// sub-block (cancellation, `step_decode_stream` incremental
/// detokenization with error recovery, `is_reasoning` tagging).
///
///   * (d) the per-step cache-maintenance cadence is routed through
///     [`DecodeStep::maintain_cache`] — the default is the every-256-step
///     `clear_cache` (no `synchronize()`, matching mlx-lm's reference
///     cadence — see the trait doc); paged steppers override to their
///     own cadence (`maybe_clear_cache_for_paged_step`).
pub(crate) fn run_decode_loop<S: DecodeStep>(
    step: &mut S,
    args: DecodeLoopArgs<'_>,
    mut streaming: Option<StreamingCtx<'_, '_>>,
) -> Result<()> {
    let DecodeLoopArgs {
        mut y,
        params: p,
        reasoning_tracker,
        profiler,
        max_new_tokens,
        eos_id,
        extra_eos_ids,
        eos_before_emit,
        generated_tokens,
        token_history,
        finish_reason,
        first_token_instant,
        report_perf,
        generation_stream,
    } = args;

    for step_idx in 0..max_new_tokens {
        // vLLM-aligned penalty context. Materialize and extract the
        // CURRENT token, then push it to `token_history` HERE — at the
        // loop TOP, BEFORE the next_y block samples the next token. vLLM
        // appends each sampled token to the live output list AFTER that
        // step's sample but BEFORE the next step's penalty
        // (gpu_model_runner.py:3691 -> sampler.py:408 ->
        // model_executor/layers/utils.py apply_penalties), so a token is
        // never in its OWN penalty but always in the NEXT one. Pushing at
        // the loop BOTTOM instead would leave the just-emitted token OUT
        // of the penalty context for the next sample (a 1-token lag);
        // pushing here closes that lag (identity at default penalties:
        // repetition=1.0/presence=0.0/frequency=0.0).
        profiler.begin("eval_token");
        y.eval();
        profiler.end();

        profiler.begin("extract");
        let token_id = y.item_at_int32(0)? as u32;
        profiler.end();
        token_history.push(token_id);
        // `generated_tokens` (the OUTPUT stream) is pushed HERE too — at
        // the loop TOP, before the terminal checks — so the
        // repetition-cutoff check below sees the CURRENT token BEFORE the
        // forward decides whether to run. The forward block does NOT read
        // `generated_tokens`, so pushing it here is safe; both histories
        // advance together at the top.
        generated_tokens.push(token_id);

        // Cache-maintenance cadence runs EVERY committed step, here at the
        // loop TOP — including terminal/length-exit steps that break
        // before the forward. The default `maintain_cache` is the FLAT
        // every-256-step `clear_cache` (a cache clear changes timing, not
        // values, and the 256-cadence keys off `step_idx`); paged
        // steppers override to their own cadence
        // (`maybe_clear_cache_for_paged_step`).
        step.maintain_cache(step_idx);

        // Compute the terminal flags BEFORE the forward
        // (terminal-before-forward ordering). A terminal current token
        // must NOT trigger the next (fallible) paged decode forward: under
        // pool pressure that forward can return Err and abort a turn that
        // should cleanly stop, and it over-records the stop token into the
        // paged adapter.
        //
        // Stop-set membership: session EOS or any extra family stop id
        // (the set is computed once per turn by the caller, not per step).
        let stops_at_eos = token_id == eos_id || extra_eos_ids.contains(&token_id);
        // Cancellation snapshot read ONCE at the iteration top and used
        // BOTH to gate the forward (feeds `is_terminal` below) AND for the
        // streaming emit-block cancel-break further down. origin/main's
        // paged loop takes a single cancel check at the iteration top —
        // before its emit and before its bottom-of-loop forward. Reusing
        // the same snapshot for the emit break (rather than re-reading
        // fresh after this loop's forward) keeps the break on the SAME
        // iteration as origin/main; see the emit block for the
        // one-token-divergence proof.
        let cancelled = streaming
            .as_ref()
            .map(|s| s.cancelled.load(Ordering::Relaxed))
            .unwrap_or(false);
        // Repetition-cutoff result computed once; the returned reason is
        // reused by the break below.
        let repetition = crate::sampling::check_repetition_cutoff(
            generated_tokens,
            p.max_consecutive_tokens,
            p.max_ngram_repeats,
            p.ngram_size,
        );
        let is_terminal = stops_at_eos || cancelled || repetition.is_some();

        // This loop builds the next forward HERE — before the streaming
        // emit block further down — whereas origin/main's paged loop
        // emitted the current token THEN ran its forward at the loop
        // bottom. The divergence is sound on two grounds. (1) Correctness:
        // on a forward `Err` the current token's chunk is dropped instead
        // of emitted — but a paged-decode forward only fails OOM-class, on
        // a turn that is already aborting (the error propagates and
        // `run_paged_turn` releases the request), so the dropped chunk
        // belongs to a turn no consumer completes. (2) No added latency:
        // the forward is
        // async-scheduled (non-blocking — `eval_step` schedules, the next
        // iteration's loop-top `y.eval()` forces it), and the emit only
        // detokenizes the ALREADY-known current token, so emitting after
        // the scheduling call adds no wall time. This ordering is what
        // encodes the vLLM-aligned penalty context (current token pushed
        // before the next sample) and the cancel-snapshot parity (the
        // pre-forward `cancelled` read reused by the emit break) — see the
        // emit block's one-token-divergence proof.
        let next_y = if step_idx + 1 < max_new_tokens && !is_terminal {
            let _stream_ctx = StreamContext::new(generation_stream);

            profiler.begin("forward");
            let next_ids = y.reshape(&[1, 1])?;
            // `token_id` was already extracted from `y` at the loop top
            // (`y.item_at_int32`); hand it down so a paged stepper need NOT
            // re-`item_at` the fresh `next_ids` reshape (an extra per-step
            // sync). Flat steppers ignore it via the default that delegates
            // to `forward`. Byte-identical: same scalar, different source.
            let (mut logits, needs_squeeze) = step.forward_with_token(&next_ids, token_id)?;
            if needs_squeeze {
                logits = logits.squeeze(Some(&[1]))?;
            }
            profiler.end();

            let (next_token, budget_forced) = if reasoning_tracker.should_force_think_end() {
                let forced_id = reasoning_tracker.forced_token_id()? as i32;
                (MxArray::from_int32(&[forced_id], &[1])?, true)
            } else {
                profiler.begin("rep_penalty");
                logits = apply_all_penalties(logits, token_history, p)?;
                profiler.end();

                profiler.begin("sample");
                let t = crate::sampling::sample(&logits, p.sampling_config)?;
                profiler.end();
                (t, false)
            };

            profiler.begin("eval_caches");
            step.eval_step(&next_token, &logits, budget_forced);
            profiler.end();

            // Diagnostic — `MLX_MTP_TRACE_LOGITS=1` per-token AR top-2
            // logit trace. `logits` is the post-penalty single-token
            // decode forward that PREDICTS the token at position
            // `token_history.len()` (the current `y` was pushed at the
            // loop top for the vLLM-aligned penalty context, so it now
            // sits at `token_history.len() - 1`). `budget_forced` skips
            // the real logits, so only trace the sampled path.
            if !budget_forced && mtp_trace_logits() {
                let logits_1d = if logits.ndim()? == 2 {
                    logits.squeeze(Some(&[0]))?
                } else {
                    logits.clone()
                };
                let vocab = logits_1d.shape_at(0)?;
                match trace_top2(&logits_1d, vocab) {
                    Ok(t2) => {
                        next_token.eval();
                        let predicted = next_token.item_at_int32(0)?;
                        eprintln!(
                            "MTP_TRACE_LOGITS source=AR pos={} token_id={} \
                             top1_id={} top1_logit={:.6} top2_id={} \
                             top2_logit={:.6} gap={:.6}",
                            token_history.len(),
                            predicted,
                            t2.top1_id,
                            t2.top1_logit,
                            t2.top2_id,
                            t2.top2_logit,
                            t2.top1_logit - t2.top2_logit,
                        );
                    }
                    Err(e) => {
                        eprintln!(
                            "MTP_TRACE_LOGITS source=AR pos={} ERROR {}",
                            token_history.len(),
                            e.reason,
                        );
                    }
                }
            }

            Some(next_token)
        } else {
            None
        };

        // TTFT marker stays at its ORIGINAL position (it does not use
        // `token_id`): the current token's `y.eval()` + extract +
        // `token_history.push` moved to the loop TOP (vLLM penalty
        // alignment, see the comment there), but the first-token timing
        // must remain byte-identical, so it is NOT moved.
        profiler.mark_first_token();
        if report_perf && first_token_instant.is_none() {
            *first_token_instant = Some(Instant::now());
        }

        // `token_history.push` / `generated_tokens.push` already happened
        // at the loop TOP. The reasoning observation stays HERE — it must
        // run AFTER the next_y block's `should_force_think_end()` check so
        // the budget-forcing timing is correct; `should_force_think_end`
        // is a non-consuming peek, so skipping it on a terminal step (no
        // forward) is safe.
        let is_reasoning = reasoning_tracker.observe_token(token_id);

        // Throttled per-step decode trace (AR / single-token loop).
        // Logs every 32 steps so long decode runs leave a sparse
        // breadcrumb trail (step idx, sampled token, cache offset from
        // the stepper — `None` skips the line; see intended difference
        // (a) in the fn docs).
        if step_idx % 32 == 0
            && let Some(cache_offset) = step.trace_offset()
        {
            tracing::info!(
                "{} decode AR step={} sampled_token_id={} cache_offset={} gen_len={}",
                step.trace_name(),
                step_idx,
                token_id,
                cache_offset,
                generated_tokens.len(),
            );
        }

        // Streaming-only block. Uses the PRECOMPUTED terminal flags so
        // the emit ordering and finish_reason strings stay consistent
        // with the non-streaming path.
        if let Some(s) = streaming.as_mut() {
            *s.last_is_reasoning = is_reasoning;

            // lfm2's order ("Check stop condition before streaming to
            // avoid leaking EOS text"): stop set BEFORE cancellation
            // and BEFORE emission — also resolves the EOS+cancel race
            // as "stop". Default false == ChatML emit-then-check.
            if eos_before_emit && stops_at_eos {
                *finish_reason = String::from("stop");
                break;
            }

            // Reuse the pre-forward `cancelled` snapshot here — do NOT
            // re-read fresh. origin/main's paged streaming pushes the
            // token to `generated_tokens`, checks cancellation at the
            // iteration TOP, emits, and runs its decode forward LAST; the
            // post-loop residual flush then re-streams
            // `decode(generated_tokens)[streamed_text_len..]`, so the TOTAL
            // streamed text equals `decode(generated_tokens)`. A fresh
            // re-read here (after this loop's forward, which is gated to run
            // BEFORE the emit) would break one iteration EARLIER than
            // origin/main, dropping the post-forward token from
            // `generated_tokens` — diverging by exactly one token in BOTH the
            // streamed text and (on a `reuse_cache` cancelled turn) the saved
            // history. Reusing the snapshot keeps the break on the same
            // iteration as origin/main.
            if cancelled {
                *finish_reason = String::from("cancelled");
                break;
            }

            let token_text = Qwen3Tokenizer::step_decode_stream(
                s.decode_stream,
                s.tokenizer,
                token_id,
                generated_tokens,
                *s.streamed_text_len,
            );
            *s.streamed_text_len += token_text.len();
            // Emission is delegated to the per-family emitter; the
            // include_reasoning suppression gate lives THERE (the default
            // emitter applies the gate then emits the chunk). Detokenize +
            // length-advance above stay OUTSIDE the emitter so
            // DecodeStream sees every token.
            s.emitter
                .on_token_text(&token_text, is_reasoning, p.include_reasoning, s.callback);
        }

        if stops_at_eos {
            *finish_reason = String::from("stop");
            break;
        }

        if let Some(reason) = repetition {
            *finish_reason = reason.to_string();
            break;
        }

        match next_y {
            Some(next) => y = next,
            None => break,
        }

        profiler.step();
    }

    profiler.snapshot_memory_after();
    profiler.report();
    Ok(())
}

#[cfg(test)]
mod run_decode_loop_tests {
    //! Mock-driven tests for [`run_decode_loop`] — a scripted
    //! [`DecodeStep`] steers the T=0 argmax through small-vocab logits
    //! so every loop behavior (EOS stop, repetition cutoff, budget
    //! forcing, streaming suppression, the 256-step cache-clear cadence)
    //! can be pinned without loading a model.

    use std::sync::Mutex;
    use std::sync::atomic::AtomicBool;

    use napi::bindgen_prelude::*;

    use super::{DecodeLoopArgs, StreamingCtx, run_decode_loop};
    use crate::array::MxArray;
    use crate::decode_profiler::DecodeProfiler;
    use crate::engine::backend::{ChunkSink, DecodeStep, DefaultStreamEmitter, StreamEmitter};
    use crate::engine::params::{ChatParams, extract_chat_params};
    use crate::engine::penalties::ReasoningTracker;
    use crate::engine::types::{ChatConfig, ChatResult, ChatStreamChunk};
    use crate::stream::{DeviceType, Stream};

    /// Scripted stepper: forward call N returns `[1, vocab]` logits
    /// whose argmax is `script[N]` (the last entry repeats once the
    /// script is exhausted). At T=0 the sampler then deterministically
    /// selects that token.
    struct MockStep {
        script: Vec<u32>,
        vocab: i64,
        forward_calls: usize,
        eval_calls: usize,
    }

    impl MockStep {
        fn new(script: Vec<u32>, vocab: i64) -> Self {
            Self {
                script,
                vocab,
                forward_calls: 0,
                eval_calls: 0,
            }
        }
    }

    impl DecodeStep for MockStep {
        fn forward(&mut self, _input_ids: &MxArray) -> Result<(MxArray, bool)> {
            let idx = self.forward_calls.min(self.script.len().saturating_sub(1));
            self.forward_calls += 1;
            let target = self.script[idx] as usize;
            let mut v = vec![0.0f32; self.vocab as usize];
            v[target] = 10.0;
            // Compiled-path shape: [1, vocab], no squeeze needed.
            Ok((MxArray::from_float32(&v, &[1, self.vocab])?, false))
        }

        fn eval_step(&mut self, next_token: &MxArray, logits: &MxArray, budget_forced: bool) {
            // Mirrors the eager closure: schedule async eval; on the
            // budget-forced path also force the logits so the lazy
            // graph stays bounded.
            MxArray::async_eval_arrays(&[next_token]);
            if budget_forced {
                logits.eval();
            }
            self.eval_calls += 1;
        }
    }

    /// Tie stepper for the vLLM-penalty-alignment regression. Every
    /// `forward` returns the SAME `[1, vocab]` logits with two near-tied
    /// entries: `high_id` slightly above `low_id` (everything else 0).
    /// Pre-penalty the argmax is `high_id`; once `high_id` is in the
    /// repetition-penalty context its positive logit is divided down
    /// below `low_id`, flipping the argmax to `low_id`. Whether that flip
    /// happens at the FIRST decode sample is exactly the loop-order
    /// question the test pins.
    struct TieStep {
        vocab: i64,
        high_id: u32,
        high_logit: f32,
        low_id: u32,
        low_logit: f32,
    }

    impl DecodeStep for TieStep {
        fn forward(&mut self, _input_ids: &MxArray) -> Result<(MxArray, bool)> {
            let mut v = vec![0.0f32; self.vocab as usize];
            v[self.high_id as usize] = self.high_logit;
            v[self.low_id as usize] = self.low_logit;
            Ok((MxArray::from_float32(&v, &[1, self.vocab])?, false))
        }

        fn eval_step(&mut self, next_token: &MxArray, logits: &MxArray, budget_forced: bool) {
            MxArray::async_eval_arrays(&[next_token]);
            if budget_forced {
                logits.eval();
            }
        }
    }

    /// Greedy (T=0) params from a default `ChatConfig` plus overrides.
    fn greedy_params(mutate: impl FnOnce(&mut ChatConfig)) -> ChatParams {
        let mut cfg = ChatConfig {
            temperature: Some(0.0),
            ..Default::default()
        };
        mutate(&mut cfg);
        extract_chat_params(&cfg)
    }

    struct LoopOutcome {
        generated: Vec<u32>,
        finish_reason: String,
    }

    /// Drive `run_decode_loop` non-streaming with a fresh profiler /
    /// stream and return the committed tokens + finish reason.
    fn drive(
        step: &mut MockStep,
        first_token: u32,
        params: &ChatParams,
        tracker: &mut ReasoningTracker,
        max_new_tokens: i32,
        eos_id: u32,
        extra_eos_ids: &[u32],
    ) -> Result<LoopOutcome> {
        let y = MxArray::from_int32(&[first_token as i32], &[1])?;
        let mut profiler = DecodeProfiler::new("test", "mock");
        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut token_history: Vec<u32> = Vec::new();
        let mut finish_reason = String::from("length");
        let mut first_token_instant: Option<std::time::Instant> = None;
        let generation_stream = Stream::new(DeviceType::Gpu);

        run_decode_loop(
            step,
            DecodeLoopArgs {
                y,
                params,
                reasoning_tracker: tracker,
                profiler: &mut profiler,
                max_new_tokens,
                eos_id,
                extra_eos_ids,
                eos_before_emit: false,
                generated_tokens: &mut generated_tokens,
                token_history: &mut token_history,
                finish_reason: &mut finish_reason,
                first_token_instant: &mut first_token_instant,
                report_perf: false,
                generation_stream,
            },
            None,
        )?;

        // The loop must keep token_history in lockstep with
        // generated_tokens (same pushes, same order).
        assert_eq!(token_history, generated_tokens);

        Ok(LoopOutcome {
            generated: generated_tokens,
            finish_reason,
        })
    }

    #[test]
    fn stops_at_eos_with_finish_reason_stop() {
        let params = greedy_params(|_| {});
        let mut tracker = ReasoningTracker::new(false, None, None);
        let mut step = MockStep::new(vec![7], 16);

        let out = drive(&mut step, 3, &params, &mut tracker, 10, 7, &[])
            .unwrap_or_else(|e| panic!("loop failed: {}", e.reason));

        // Step 0 commits the prefill token (3); the scripted forward
        // produced EOS (7) which step 1 commits, then stops.
        assert_eq!(out.generated, vec![3, 7]);
        assert_eq!(out.finish_reason, "stop");
    }

    #[test]
    fn repetition_cutoff_triggers() {
        // max_consecutive_tokens = 3: three identical commits trip the
        // consecutive-token detector.
        let params = greedy_params(|cfg| {
            cfg.max_consecutive_tokens = Some(3);
        });
        let mut tracker = ReasoningTracker::new(false, None, None);
        let mut step = MockStep::new(vec![5], 16);

        let out = drive(&mut step, 5, &params, &mut tracker, 20, 7, &[])
            .unwrap_or_else(|e| panic!("loop failed: {}", e.reason));

        assert_eq!(out.generated, vec![5, 5, 5]);
        assert_eq!(out.finish_reason, "repetition");
    }

    #[test]
    fn budget_forcing_injects_think_end_token() {
        const THINK_END: u32 = 9;
        let params = greedy_params(|_| {});
        // Budget 2: after two observed thinking tokens the tracker
        // forces `</think>` as the NEXT pipelined token.
        let mut tracker = ReasoningTracker::new(true, Some(2), Some(THINK_END));
        let mut step = MockStep::new(vec![5], 16);

        let out = drive(&mut step, 4, &params, &mut tracker, 6, 7, &[])
            .unwrap_or_else(|e| panic!("loop failed: {}", e.reason));

        // Pipeline timeline: commits [4, 5] trip the budget; the step
        // building the 3rd pipelined token consumes the force flag, so
        // one over-budget token (5) is already in flight and the forced
        // `</think>` lands at index 3.
        assert_eq!(out.generated, vec![4, 5, 5, THINK_END, 5, 5]);
        assert_eq!(out.finish_reason, "length");
        // 3 reasoning tokens observed (incl. the in-flight over-budget
        // one); the forced `</think>` exits thinking, trailing 5s are
        // content.
        assert_eq!(tracker.reasoning_token_count(), 3);
    }

    #[test]
    fn long_run_completes_through_cache_clear_cadence() {
        // >300 steps so the every-256-step `clear_cache` branch (the
        // FLAT `DecodeStep::maintain_cache` default) executes at least
        // once. Cutoffs disabled so the constant script can't trip them.
        let params = greedy_params(|cfg| {
            cfg.max_consecutive_tokens = Some(0);
            cfg.max_ngram_repeats = Some(0);
        });
        let mut tracker = ReasoningTracker::new(false, None, None);
        let mut step = MockStep::new(vec![1], 16);

        let out = drive(&mut step, 1, &params, &mut tracker, 400, 15, &[])
            .unwrap_or_else(|e| panic!("loop failed: {}", e.reason));

        assert_eq!(out.generated.len(), 400);
        assert!(out.generated.iter().all(|&t| t == 1));
        assert_eq!(out.finish_reason, "length");
        // Last iteration skips the pipelined forward (step+1 == max).
        assert_eq!(step.forward_calls, 399);
    }

    #[test]
    fn maintain_cache_default_clears_without_synchronize() {
        // Proves the FLAT `DecodeStep::maintain_cache` default cadence
        // (crates/mlx-core/src/engine/backend.rs) fires `clear_cache`
        // and NEVER `synchronize` — the redundant-stall this fix
        // removes. Thread-local counters (`array::memory`) so this is
        // race-free against other tests running concurrently.
        use crate::array::memory::{TEST_CLEAR_CACHE_CALLS, TEST_SYNCHRONIZE_CALLS};

        TEST_SYNCHRONIZE_CALLS.with(|c| c.set(0));
        TEST_CLEAR_CACHE_CALLS.with(|c| c.set(0));

        let params = greedy_params(|cfg| {
            cfg.max_consecutive_tokens = Some(0);
            cfg.max_ngram_repeats = Some(0);
        });
        let mut tracker = ReasoningTracker::new(false, None, None);
        let mut step = MockStep::new(vec![1], 16);

        // >256 steps so the cadence branch fires at least once.
        drive(&mut step, 1, &params, &mut tracker, 300, 15, &[])
            .unwrap_or_else(|e| panic!("loop failed: {}", e.reason));

        assert_eq!(
            TEST_SYNCHRONIZE_CALLS.with(|c| c.get()),
            0,
            "FLAT maintain_cache default must not call the stream-less \
             mlx_synchronize — see crate::stream::StreamContext (the \
             generation stream is not the default stream at this point)"
        );
        assert!(
            TEST_CLEAR_CACHE_CALLS.with(|c| c.get()) >= 1,
            "FLAT maintain_cache default should still drain the allocator \
             cache on the 256-step cadence"
        );
    }

    // ---- streaming ----

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

    /// Word-level tokenizer over a tiny fixed vocab so the
    /// DecodeStream produces deterministic per-token text
    /// (space-joined words; ids 0..=5). Built via the standard
    /// tokenizer.json deserialization path (the builder API requires
    /// tokenizers' internal AHashMap, which is not re-exported).
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

    #[test]
    fn streaming_suppresses_reasoning_chunks_but_detokenization_advances() {
        const THINK_END: u32 = 2;
        const EOS: u32 = 5;
        // include_reasoning = false → reasoning deltas (incl. the
        // `</think>` closer) must be suppressed while the DecodeStream
        // still sees every token.
        let params = greedy_params(|cfg| {
            cfg.include_reasoning = Some(false);
        });
        let mut tracker = ReasoningTracker::new(true, None, Some(THINK_END));
        let mut step = MockStep::new(vec![1, THINK_END, 3, 4, EOS], 7);

        let tokenizer = tiny_tokenizer();
        let mut decode_stream = tokenizer.decode_stream(true);
        let sink = RecSink {
            chunks: Mutex::new(Vec::new()),
        };
        let cancelled = AtomicBool::new(false);
        let mut streamed_text_len = 0usize;
        let mut last_is_reasoning = false;
        let mut emitter = DefaultStreamEmitter;

        let y = MxArray::from_int32(&[0], &[1]).unwrap_or_else(|e| panic!("{}", e.reason));
        let mut profiler = DecodeProfiler::new("test", "mock");
        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut token_history: Vec<u32> = Vec::new();
        let mut finish_reason = String::from("length");
        let mut first_token_instant: Option<std::time::Instant> = None;
        let generation_stream = Stream::new(DeviceType::Gpu);

        run_decode_loop(
            &mut step,
            DecodeLoopArgs {
                y,
                params: &params,
                reasoning_tracker: &mut tracker,
                profiler: &mut profiler,
                max_new_tokens: 10,
                eos_id: EOS,
                extra_eos_ids: &[],
                eos_before_emit: false,
                generated_tokens: &mut generated_tokens,
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
        .unwrap_or_else(|e| panic!("loop failed: {}", e.reason));

        // Committed: prefill token 0 (reasoning), 1 (reasoning),
        // 2 (</think>, still reasoning), 3 + 4 (content), 5 (eos).
        assert_eq!(generated_tokens, vec![0, 1, THINK_END, 3, 4, EOS]);
        assert_eq!(finish_reason, "stop");
        assert!(!last_is_reasoning);

        // Only the 3 content tokens were emitted; all tagged
        // is_reasoning == Some(false).
        let chunks = sink
            .chunks
            .lock()
            .unwrap_or_else(|e| panic!("sink poisoned: {e}"));
        let sent: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(sent, vec![" c3", " c4", " eos"]);
        assert!(chunks.iter().all(|c| c.is_reasoning == Some(false)));

        // Detokenization advanced through the SUPPRESSED tokens too:
        // streamed_text_len covers the full decoded text, not just the
        // emitted chunks.
        let full_text = "t0 t1 end c3 c4 eos";
        assert_eq!(streamed_text_len, full_text.len());
        let sent_len: usize = chunks.iter().map(|c| c.text.len()).sum();
        assert!(
            sent_len < streamed_text_len,
            "suppressed reasoning text must still advance the detok cursor \
             (sent {sent_len} vs advanced {streamed_text_len})"
        );
    }

    #[test]
    fn streaming_emits_reasoning_chunks_when_included() {
        const THINK_END: u32 = 2;
        const EOS: u32 = 5;
        // include_reasoning defaults to true → every delta is emitted,
        // reasoning ones tagged is_reasoning == Some(true).
        let params = greedy_params(|_| {});
        let mut tracker = ReasoningTracker::new(true, None, Some(THINK_END));
        let mut step = MockStep::new(vec![1, THINK_END, 3, EOS], 7);

        let tokenizer = tiny_tokenizer();
        let mut decode_stream = tokenizer.decode_stream(true);
        let sink = RecSink {
            chunks: Mutex::new(Vec::new()),
        };
        let cancelled = AtomicBool::new(false);
        let mut streamed_text_len = 0usize;
        let mut last_is_reasoning = false;
        let mut emitter = DefaultStreamEmitter;

        let y = MxArray::from_int32(&[0], &[1]).unwrap_or_else(|e| panic!("{}", e.reason));
        let mut profiler = DecodeProfiler::new("test", "mock");
        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut token_history: Vec<u32> = Vec::new();
        let mut finish_reason = String::from("length");
        let mut first_token_instant: Option<std::time::Instant> = None;
        let generation_stream = Stream::new(DeviceType::Gpu);

        run_decode_loop(
            &mut step,
            DecodeLoopArgs {
                y,
                params: &params,
                reasoning_tracker: &mut tracker,
                profiler: &mut profiler,
                max_new_tokens: 10,
                eos_id: EOS,
                extra_eos_ids: &[],
                eos_before_emit: false,
                generated_tokens: &mut generated_tokens,
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
        .unwrap_or_else(|e| panic!("loop failed: {}", e.reason));

        assert_eq!(generated_tokens, vec![0, 1, THINK_END, 3, EOS]);
        let chunks = sink
            .chunks
            .lock()
            .unwrap_or_else(|e| panic!("sink poisoned: {e}"));
        // One chunk per committed token; reasoning tagging flips after
        // the `</think>` closer (which itself is reasoning).
        let tags: Vec<Option<bool>> = chunks.iter().map(|c| c.is_reasoning).collect();
        assert_eq!(
            tags,
            vec![Some(true), Some(true), Some(true), Some(false), Some(false)]
        );
        let sent_len: usize = chunks.iter().map(|c| c.text.len()).sum();
        assert_eq!(sent_len, streamed_text_len);
    }

    /// Stepper that flips a shared cancellation flag DURING a forward —
    /// i.e. AFTER the loop's pre-forward cancellation snapshot is taken
    /// but BEFORE the streaming emit-check runs. Used to pin origin/main
    /// streaming-cancel parity: the emit block must REUSE the pre-forward
    /// snapshot (NOT re-read fresh), so a cancel that lands during this
    /// iteration's forward is acted on at the NEXT iteration's top —
    /// exactly like origin/main's paged loop, whose single cancel check
    /// runs at the iteration top with its forward LAST.
    ///
    /// `forward` argmax-scripts like `MockStep`; on the
    /// `flip_on_forward`-th forward call (1-indexed) it stores `true`
    /// into `cancel` BEFORE returning the logits, so the cancel becomes
    /// visible only after that step's forward — exactly the
    /// "cancel arrives during the forward" race.
    struct CancelDuringForwardStep<'a> {
        script: Vec<u32>,
        vocab: i64,
        forward_calls: usize,
        cancel: &'a AtomicBool,
        flip_on_forward: usize,
    }

    impl DecodeStep for CancelDuringForwardStep<'_> {
        fn forward(&mut self, _input_ids: &MxArray) -> Result<(MxArray, bool)> {
            let idx = self.forward_calls.min(self.script.len().saturating_sub(1));
            self.forward_calls += 1;
            // Flip the shared cancel flag mid-forward (after the loop's
            // pre-forward snapshot, before the emit-check). The
            // 1-indexed Nth forward corresponds to step_idx N-1.
            if self.forward_calls == self.flip_on_forward {
                self.cancel
                    .store(true, std::sync::atomic::Ordering::Relaxed);
            }
            let target = self.script[idx] as usize;
            let mut v = vec![0.0f32; self.vocab as usize];
            v[target] = 10.0;
            Ok((MxArray::from_float32(&v, &[1, self.vocab])?, false))
        }

        fn eval_step(&mut self, next_token: &MxArray, logits: &MxArray, budget_forced: bool) {
            MxArray::async_eval_arrays(&[next_token]);
            if budget_forced {
                logits.eval();
            }
        }
    }

    /// Streaming-cancellation origin/main-parity lock — PASSES with the
    /// snapshot reuse, FAILS on a fresh re-read. The emit block reuses the
    /// pre-forward `cancelled` snapshot, so a cancel arriving DURING a
    /// forward suppresses the NEXT step's emit (this step's token was
    /// already committed and is still streamed), matching origin/main's
    /// paged loop, whose single cancel check runs at the iteration top
    /// with its forward LAST and whose post-loop residual re-streams
    /// `decode(generated_tokens)`.
    ///
    /// Timeline (prefill seed id 1; script 3,4; eos id 5 never hit;
    /// `flip_on_forward = 2` → cancel flips during step_idx 1's forward).
    /// step_idx 0 commits 1, snapshot=false → forward #1 (→ 3), emits
    /// "t1". step_idx 1 commits 3, snapshot=false (cancel not yet set) →
    /// forward #2 flips cancel=true (→ 4); the emit-check reuses that
    /// false snapshot → emits " c3" and continues. step_idx 2 commits 4,
    /// its snapshot now reads true → `is_terminal` skips the forward and
    /// the emit-check breaks "cancelled" WITHOUT emitting token 4
    /// per-token. Emitted == ["t1", " c3"]; `generated_tokens` == [1,3,4]
    /// (token 4 IS committed — origin/main's residual would re-stream it).
    ///
    /// A FRESH re-read would instead read true inside step_idx 1's
    /// emit-check, breaking there with emitted == ["t1"] and
    /// `generated_tokens` == [1,3] — one token short of origin/main in
    /// BOTH the stream and the committed set. Both variants finish
    /// "cancelled". Deterministic (T=0 argmax, no early EOS/repetition,
    /// thinking disabled so no suppression).
    #[test]
    fn streaming_cancel_during_forward_emits_current_token_then_breaks_next_step_matching_main() {
        const EOS: u32 = 5; // not in the committed sequence
        let params = greedy_params(|_| {});
        // Thinking disabled: every emitted token flows through (no
        // reasoning suppression confounds the emitted set).
        let mut tracker = ReasoningTracker::new(false, None, None);

        // ONE atomic, shared by both the stepper (which flips it) and the
        // StreamingCtx (which the loop reads). Two shared &AtomicBool
        // borrows of the same atomic are fine; store-through-& is allowed.
        let cancelled = AtomicBool::new(false);
        let mut step = CancelDuringForwardStep {
            script: vec![3, 4],
            vocab: 7,
            forward_calls: 0,
            cancel: &cancelled,
            // Flip during the 2nd forward == step_idx 1's forward.
            flip_on_forward: 2,
        };

        let tokenizer = tiny_tokenizer();
        let mut decode_stream = tokenizer.decode_stream(true);
        let sink = RecSink {
            chunks: Mutex::new(Vec::new()),
        };
        let mut streamed_text_len = 0usize;
        let mut last_is_reasoning = false;
        let mut emitter = DefaultStreamEmitter;

        // Prefill seed id 1 ("t1").
        let y = MxArray::from_int32(&[1], &[1]).unwrap_or_else(|e| panic!("{}", e.reason));
        let mut profiler = DecodeProfiler::new("test", "mock");
        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut token_history: Vec<u32> = Vec::new();
        let mut finish_reason = String::from("length");
        let mut first_token_instant: Option<std::time::Instant> = None;
        let generation_stream = Stream::new(DeviceType::Gpu);

        run_decode_loop(
            &mut step,
            DecodeLoopArgs {
                y,
                params: &params,
                reasoning_tracker: &mut tracker,
                profiler: &mut profiler,
                max_new_tokens: 10,
                eos_id: EOS,
                extra_eos_ids: &[],
                eos_before_emit: false,
                generated_tokens: &mut generated_tokens,
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
        .unwrap_or_else(|e| panic!("loop failed: {}", e.reason));

        assert_eq!(finish_reason, "cancelled");

        let chunks = sink
            .chunks
            .lock()
            .unwrap_or_else(|e| panic!("sink poisoned: {e}"));
        let sent: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
        // Snapshot reuse: step_idx 1's token (" c3") is committed BEFORE
        // its forward and emitted because the pre-forward snapshot was
        // still false; the loop then breaks at step_idx 2 (whose snapshot
        // reads true) so token 4 is NOT emitted per-token. A fresh re-read
        // would break inside step_idx 1 with emitted == ["t1"] — one token
        // short of origin/main.
        assert_eq!(
            sent,
            vec!["t1", " c3"],
            "snapshot reuse must keep the cancel break on the SAME iteration as \
             origin/main: the step_idx 1 token (committed pre-forward) is still \
             streamed; a fresh re-read would drop it"
        );
        drop(chunks);
        // origin/main token count: token 4 (the post-forward commit the
        // fresh re-read would have dropped) IS in `generated_tokens`, so the
        // post-loop residual flush would re-stream it — the observable
        // total-streamed-text contract.
        assert_eq!(
            generated_tokens,
            vec![1, 3, 4],
            "the post-forward token (4) must be committed (origin/main count); the \
             fresh re-read would have produced [1, 3]"
        );
    }

    // ---- optional hook seams ----

    /// Drive `run_decode_loop` in streaming mode with a caller-supplied
    /// emitter / ordering knob / pre-set cancellation, returning the
    /// committed tokens, finish reason, and emitted chunks.
    #[allow(clippy::too_many_arguments)]
    fn drive_streaming(
        step: &mut MockStep,
        first_token: u32,
        params: &ChatParams,
        tracker: &mut ReasoningTracker,
        eos_id: u32,
        extra_eos_ids: &[u32],
        eos_before_emit: bool,
        emitter: &mut dyn StreamEmitter,
        cancelled_pre_set: bool,
    ) -> (Vec<u32>, String, Vec<ChatStreamChunk>) {
        let tokenizer = tiny_tokenizer();
        let mut decode_stream = tokenizer.decode_stream(true);
        let sink = RecSink {
            chunks: Mutex::new(Vec::new()),
        };
        let cancelled = AtomicBool::new(cancelled_pre_set);
        let mut streamed_text_len = 0usize;
        let mut last_is_reasoning = false;

        let y = MxArray::from_int32(&[first_token as i32], &[1])
            .unwrap_or_else(|e| panic!("{}", e.reason));
        let mut profiler = DecodeProfiler::new("test", "mock");
        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut token_history: Vec<u32> = Vec::new();
        let mut finish_reason = String::from("length");
        let mut first_token_instant: Option<std::time::Instant> = None;
        let generation_stream = Stream::new(DeviceType::Gpu);

        run_decode_loop(
            step,
            DecodeLoopArgs {
                y,
                params,
                reasoning_tracker: tracker,
                profiler: &mut profiler,
                max_new_tokens: 10,
                eos_id,
                extra_eos_ids,
                eos_before_emit,
                generated_tokens: &mut generated_tokens,
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
                emitter,
            }),
        )
        .unwrap_or_else(|e| panic!("loop failed: {}", e.reason));

        let chunks = sink
            .chunks
            .into_inner()
            .unwrap_or_else(|e| panic!("sink poisoned: {e}"));
        (generated_tokens, finish_reason, chunks)
    }

    /// D3 — extra stop ids are honored alongside the session EOS with
    /// finish_reason "stop" (Gemma4's config eos set). The session EOS
    /// id itself is NOT in the script, so only the extra set can stop
    /// the run.
    #[test]
    fn stops_on_extra_eos_id_with_finish_reason_stop() {
        let params = greedy_params(|_| {});
        let mut tracker = ReasoningTracker::new(false, None, None);
        let mut step = MockStep::new(vec![9], 16);

        let out = drive(&mut step, 3, &params, &mut tracker, 10, 7, &[9, 11])
            .unwrap_or_else(|e| panic!("loop failed: {}", e.reason));

        assert_eq!(out.generated, vec![3, 9]);
        assert_eq!(out.finish_reason, "stop");
    }

    /// D3 — the extra stop set covers the FIRST prefill-sampled token
    /// too (the step-0 commit goes through the same in-loop check).
    #[test]
    fn extra_eos_stops_on_first_prefill_sampled_token() {
        let params = greedy_params(|_| {});
        let mut tracker = ReasoningTracker::new(false, None, None);
        let mut step = MockStep::new(vec![1], 16);

        let out = drive(&mut step, 9, &params, &mut tracker, 10, 7, &[9])
            .unwrap_or_else(|e| panic!("loop failed: {}", e.reason));

        assert_eq!(out.generated, vec![9]);
        assert_eq!(out.finish_reason, "stop");
    }

    /// D10 — default (ChatML) order emits the EOS token's chunk before
    /// stopping; lfm2's eos_before_emit order stops first so the EOS
    /// text never reaches the stream. Token bytes identical either way.
    #[test]
    fn streaming_eos_before_emit_suppresses_eos_chunk() {
        const EOS: u32 = 5;
        for (eos_before_emit, expected_texts) in [
            (false, vec!["t1", " c3", " eos"]),
            (true, vec!["t1", " c3"]),
        ] {
            let params = greedy_params(|_| {});
            let mut tracker = ReasoningTracker::new(false, None, None);
            let mut step = MockStep::new(vec![3, EOS], 7);
            let mut emitter = DefaultStreamEmitter;

            let (generated, finish, chunks) = drive_streaming(
                &mut step,
                1,
                &params,
                &mut tracker,
                EOS,
                &[],
                eos_before_emit,
                &mut emitter,
                false,
            );

            assert_eq!(
                generated,
                vec![1, 3, EOS],
                "eos_before_emit={eos_before_emit}"
            );
            assert_eq!(finish, "stop", "eos_before_emit={eos_before_emit}");
            let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
            assert_eq!(texts, expected_texts, "eos_before_emit={eos_before_emit}");
        }
    }

    /// The EOS+cancel race: lfm2's order checks the stop set BEFORE the
    /// cancellation flag, so a turn whose token is EOS while cancellation
    /// is pending finishes "stop"; the default order finishes
    /// "cancelled".
    #[test]
    fn streaming_eos_before_emit_wins_eos_cancel_race() {
        const EOS: u32 = 5;
        for (eos_before_emit, expected_finish) in [(false, "cancelled"), (true, "stop")] {
            let params = greedy_params(|_| {});
            let mut tracker = ReasoningTracker::new(false, None, None);
            let mut step = MockStep::new(vec![1], 7);
            let mut emitter = DefaultStreamEmitter;

            let (generated, finish, chunks) = drive_streaming(
                &mut step,
                EOS, // first committed token IS the session EOS
                &params,
                &mut tracker,
                EOS,
                &[],
                eos_before_emit,
                &mut emitter,
                true, // cancellation already pending
            );

            assert_eq!(generated, vec![EOS]);
            assert_eq!(finish, expected_finish, "eos_before_emit={eos_before_emit}");
            assert!(chunks.is_empty(), "no chunk on either race outcome");
        }
    }

    /// Recording emitter: the loop must route EVERY committed token's
    /// text through the emitter (suppressed/reasoning ones included — the
    /// suppression gate lives in the EMITTER, not the loop), and the sink
    /// only sees what the emitter chooses to send (here: nothing).
    struct RecordingEmitter {
        seen: Vec<(String, bool, bool)>,
        residuals: Vec<String>,
        finished: usize,
    }

    impl StreamEmitter for RecordingEmitter {
        fn on_token_text(
            &mut self,
            token_text: &str,
            is_reasoning: bool,
            include_reasoning: bool,
            _sink: &dyn ChunkSink,
        ) {
            self.seen
                .push((token_text.to_string(), is_reasoning, include_reasoning));
        }

        fn on_residual(
            &mut self,
            residual: &str,
            _is_reasoning: bool,
            _include_reasoning: bool,
            _sink: &dyn ChunkSink,
        ) {
            self.residuals.push(residual.to_string());
        }

        fn finish(&mut self, _result: &ChatResult, _sink: &dyn ChunkSink) {
            self.finished += 1;
        }
    }

    #[test]
    fn custom_emitter_observes_suppressed_tokens_and_owns_the_sink() {
        const THINK_END: u32 = 2;
        const EOS: u32 = 5;
        // include_reasoning = false: the DEFAULT emitter would suppress
        // the reasoning deltas; a custom emitter must still observe
        // them (Gemma4's parser needs every byte).
        let params = greedy_params(|cfg| {
            cfg.include_reasoning = Some(false);
        });
        let mut tracker = ReasoningTracker::new(true, None, Some(THINK_END));
        let mut step = MockStep::new(vec![1, THINK_END, 3, EOS], 7);
        let mut emitter = RecordingEmitter {
            seen: Vec::new(),
            residuals: Vec::new(),
            finished: 0,
        };

        let (generated, finish, chunks) = drive_streaming(
            &mut step,
            0,
            &params,
            &mut tracker,
            EOS,
            &[],
            false,
            &mut emitter,
            false,
        );

        assert_eq!(generated, vec![0, 1, THINK_END, 3, EOS]);
        assert_eq!(finish, "stop");
        // One on_token_text per committed token, reasoning tags intact,
        // the turn's include_reasoning threaded through.
        let texts: Vec<&str> = emitter.seen.iter().map(|(t, _, _)| t.as_str()).collect();
        assert_eq!(texts, vec!["t0", " t1", " end", " c3", " eos"]);
        let tags: Vec<bool> = emitter.seen.iter().map(|(_, r, _)| *r).collect();
        assert_eq!(tags, vec![true, true, true, false, false]);
        assert!(emitter.seen.iter().all(|(_, _, inc)| !inc));
        // The emitter sent nothing — the loop must not bypass it.
        assert!(chunks.is_empty(), "loop bypassed the emitter: {chunks:?}");
        // The loop never calls on_residual/finish (those are the
        // session core's post-loop responsibilities).
        assert!(emitter.residuals.is_empty());
        assert_eq!(emitter.finished, 0);
    }

    /// vLLM penalty-alignment regression. The just-emitted token must be
    /// in the penalty context when sampling the NEXT token (vLLM appends
    /// each sampled token to the live output list BEFORE the next step's
    /// penalty: gpu_model_runner.py:3691 -> sampler.py:408 ->
    /// apply_penalties).
    ///
    /// Setup: the prefill seed `y == HIGH` (HIGH is NOT in the prompt —
    /// the `drive` helper starts with an EMPTY token_history). Every
    /// decode `forward` returns a near-tie `HIGH=10.0`, `LOW=8.0`. With
    /// `repetition_penalty = 2.0`, the loop pushes HIGH into
    /// token_history at the loop TOP, so the step-0 `apply_all_penalties`
    /// divides HIGH's logit (10.0 -> 5.0) and LOW (8.0, un-penalized)
    /// wins -> the 2nd committed token is LOW.
    ///
    /// Pushing at the loop BOTTOM instead would leave token_history EMPTY
    /// at the step-0 sample -> no penalty applied -> HIGH (10.0 > 8.0)
    /// wins -> the 2nd token would be HIGH. The single assertion
    /// (`generated[1] == LOW`) therefore distinguishes the two loop
    /// orderings. Deterministic (T=0 argmax).
    #[test]
    fn just_emitted_token_is_in_next_penalty_context_vllm_aligned() {
        const HIGH: u32 = 4; // prefill seed + pre-penalty argmax
        const LOW: u32 = 9; // wins once HIGH is penalized
        const EOS: u32 = 1; // distinct from HIGH/LOW; never sampled

        // repetition_penalty 2.0 (the only non-default penalty); cutoffs
        // OFF so the constant near-tie can't trip them. budget 2 so the
        // loop commits exactly [HIGH, LOW] then stops on the pipelined
        // None.
        let params = greedy_params(|cfg| {
            cfg.repetition_penalty = Some(2.0);
            cfg.max_consecutive_tokens = Some(0);
            cfg.max_ngram_repeats = Some(0);
        });
        let mut tracker = ReasoningTracker::new(false, None, None);
        let mut step = TieStep {
            vocab: 16,
            high_id: HIGH,
            high_logit: 10.0,
            low_id: LOW,
            low_logit: 8.0,
        };

        let y =
            MxArray::from_int32(&[HIGH as i32], &[1]).unwrap_or_else(|e| panic!("{}", e.reason));
        let mut profiler = DecodeProfiler::new("test", "mock");
        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut token_history: Vec<u32> = Vec::new();
        let mut finish_reason = String::from("length");
        let mut first_token_instant: Option<std::time::Instant> = None;
        let generation_stream = Stream::new(DeviceType::Gpu);

        run_decode_loop(
            &mut step,
            DecodeLoopArgs {
                y,
                params: &params,
                reasoning_tracker: &mut tracker,
                profiler: &mut profiler,
                max_new_tokens: 2,
                eos_id: EOS,
                extra_eos_ids: &[],
                eos_before_emit: false,
                generated_tokens: &mut generated_tokens,
                token_history: &mut token_history,
                finish_reason: &mut finish_reason,
                first_token_instant: &mut first_token_instant,
                report_perf: false,
                generation_stream,
            },
            None,
        )
        .unwrap_or_else(|e| panic!("loop failed: {}", e.reason));

        // Step 0 commits the prefill seed HIGH; step 1 commits LOW (HIGH
        // penalized out of the running). A bottom-push loop would produce
        // [HIGH, HIGH].
        assert_eq!(
            generated_tokens,
            vec![HIGH, LOW],
            "the just-emitted HIGH must be in the next sample's penalty context \
             (vLLM-aligned): with repetition_penalty it is divided down and LOW wins; \
             a bottom-push loop would leave HIGH out of context, so HIGH would repeat"
        );
        // token_history stays in lockstep with the output stream.
        assert_eq!(token_history, generated_tokens);
    }
}
