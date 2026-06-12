//! Shared autoregressive decode-loop machinery.
//!
//! `DecodeOps` + the `decode_loop!` macro are transitional — S5 replaces
//! them with the generic `run_decode_loop` over the backend traits.
//!
//! NOTE: `mtp_trace_logits` / `Top2` / `trace_top2` live HERE (not in
//! `models::qwen3_5::mtp_decode`) because they also serve the AR
//! `decode_loop!` per-token trace diagnostics; the MTP path imports
//! them from this module.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use napi::bindgen_prelude::*;

use crate::array::MxArray;
use crate::decode_profiler::DecodeProfiler;
use crate::engine::backend::{ChunkSink, DecodeStep};
use crate::engine::params::ChatParams;
use crate::engine::penalties::{ReasoningTracker, apply_all_penalties};
use crate::engine::types::ChatStreamChunk;
use crate::stream::{Stream, StreamContext};
use crate::tokenizer::Qwen3Tokenizer;

// Diagnostic — per-committed-token top-2 logit trace.
//
// `MLX_MTP_TRACE_LOGITS=1` (or `true` / `on`) enables an env-gated
// per-token logit trace emitted to stderr. For each committed decode
// token it logs the position index, the committed token id, and the
// top-2 (token id + logit value) of the forward that produced it:
//   * the AR `decode_loop!` logs the single-token decode forward;
//   * `run_mtp_cycle_inner` logs the batched verify forward, per
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

/// Closures for model-specific operations in the decode loop.
///
/// `F`: forward pass — takes (input_ids [1,1], embedding_weight) → Result<(logits, needs_squeeze)>.
/// `E`: eval step — takes (next_token, logits, budget_forced) → schedules async eval.
pub(crate) struct DecodeOps<F, E>
where
    F: FnMut(&MxArray, &MxArray) -> Result<(MxArray, bool)>,
    E: Fn(&MxArray, &MxArray, bool),
{
    pub forward: F,
    pub eval_step: E,
}

/// Pipelined decode loop shared across all Qwen3.5 model variants.
///
/// Generates the token-by-token decode loop with:
/// - Pipelining: builds step N+1's graph before blocking on step N
/// - Budget enforcement via ReasoningTracker
/// - Penalty application via apply_all_penalties
/// - Stop conditions: EOS, repetition cutoff
/// - Every-256-step synchronize_and_clear_cache
/// - Profiler instrumentation
///
/// The optional `streaming:` block adds callback emission, cancellation,
/// incremental detokenization, and is_reasoning tagging.
macro_rules! decode_loop {
    (
        ops: $ops:expr,
        y: $y:expr,
        embedding_weight: $emb:expr,
        params: $p:expr,
        reasoning_tracker: $tracker:expr,
        profiler: $profiler:expr,
        max_new_tokens: $max:expr,
        eos_id: $eos:expr,
        generated_tokens: $gen:expr,
        token_history: $hist:expr,
        finish_reason: $reason:expr,
        first_token_instant: $first_tok:expr,
        report_perf: $report:expr,
        generation_stream: $stream:expr
        $(, streaming: {
            callback: $cb:expr,
            cancelled: $cancelled:expr,
            decode_stream: $ds:expr,
            tokenizer: $tok:expr,
            streamed_text_len: $slen:expr,
            last_is_reasoning: $last_r:expr
        })?
    ) => {{
        for step in 0..$max {
            let next_y = if step + 1 < $max {
                let _stream_ctx = $crate::stream::StreamContext::new($stream);

                $profiler.begin("forward");
                let next_ids = $y.reshape(&[1, 1])?;
                let (mut logits, needs_squeeze) = ($ops.forward)(&next_ids, &$emb)?;
                if needs_squeeze {
                    logits = logits.squeeze(Some(&[1]))?;
                }
                $profiler.end();

                let (next_token, budget_forced) =
                    if $tracker.should_force_think_end() {
                        let forced_id = $tracker.forced_token_id()? as i32;
                        ($crate::array::MxArray::from_int32(&[forced_id], &[1])?, true)
                    } else {
                        $profiler.begin("rep_penalty");
                        logits = $crate::engine::penalties::apply_all_penalties(
                            logits, &$hist, &$p,
                        )?;
                        $profiler.end();

                        $profiler.begin("sample");
                        let t = $crate::sampling::sample(&logits, $p.sampling_config)?;
                        $profiler.end();
                        (t, false)
                    };

                $profiler.begin("eval_caches");
                ($ops.eval_step)(&next_token, &logits, budget_forced);
                $profiler.end();

                // Diagnostic — `MLX_MTP_TRACE_LOGITS=1` per-token AR
                // top-2 logit trace. `logits` is the post-penalty
                // single-token decode forward that PREDICTS the token
                // at position `$hist.len() + 1` (the current `$y` sits
                // at `$hist.len()`). `budget_forced` skips the real
                // logits, so only trace the sampled path.
                if !budget_forced
                    && $crate::engine::decode::mtp_trace_logits()
                {
                    let logits_1d = if logits.ndim()? == 2 {
                        logits.squeeze(Some(&[0]))?
                    } else {
                        logits.clone()
                    };
                    let vocab = logits_1d.shape_at(0)?;
                    match $crate::engine::decode::trace_top2(
                        &logits_1d, vocab,
                    ) {
                        Ok(t2) => {
                            next_token.eval();
                            let predicted = next_token.item_at_int32(0)?;
                            eprintln!(
                                "MTP_TRACE_LOGITS source=AR pos={} token_id={} \
                                 top1_id={} top1_logit={:.6} top2_id={} \
                                 top2_logit={:.6} gap={:.6}",
                                $hist.len() + 1,
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
                                $hist.len() + 1,
                                e.reason,
                            );
                        }
                    }
                }

                Some(next_token)
            } else {
                None
            };

            $profiler.begin("eval_token");
            $y.eval();
            $profiler.end();

            $profiler.begin("extract");
            let token_id = $y.item_at_int32(0)? as u32;
            $profiler.end();
            $profiler.mark_first_token();
            if $report && $first_tok.is_none() {
                $first_tok = Some(std::time::Instant::now());
            }

            $gen.push(token_id);
            $hist.push(token_id);
            let _is_reasoning = $tracker.observe_token(token_id);

            // Throttled per-step decode trace (AR / single-token loop).
            // Logs every 32 steps so long decode runs leave a sparse
            // breadcrumb trail (step idx, sampled token, cache offset
            // from the dense compiled global — MoE callers can ignore
            // the offset field).
            if step % 32 == 0 {
                let cache_offset = unsafe { mlx_sys::mlx_qwen35_get_cache_offset() };
                tracing::info!(
                    "Qwen3.5 decode AR step={} sampled_token_id={} cache_offset={} gen_len={}",
                    step,
                    token_id,
                    cache_offset,
                    $gen.len(),
                );
            }

            // Streaming-only block (conditionally compiled via macro repetition)
            $(
                $last_r = _is_reasoning;

                if $cancelled.load(std::sync::atomic::Ordering::Relaxed) {
                    $reason = String::from("cancelled");
                    break;
                }

                let token_text = $crate::tokenizer::Qwen3Tokenizer::step_decode_stream(
                    &mut $ds,
                    $tok.inner(),
                    token_id,
                    &$gen,
                    $slen,
                );
                $slen += token_text.len();
                // Suppress reasoning (<think>…</think>) deltas from the stream
                // when include_reasoning == false. Detokenize + length-advance
                // above stay OUTSIDE this gate so DecodeStream sees every token.
                if $p.include_reasoning || !_is_reasoning {
                    $cb.call(
                        Ok($crate::engine::types::ChatStreamChunk {
                            text: token_text,
                            done: false,
                            finish_reason: None,
                            tool_calls: None,
                            thinking: None,
                            num_tokens: None,
                            prompt_tokens: None,
                            reasoning_tokens: None,
                            raw_text: None,
                            cached_tokens: None,
                            performance: None,
                            is_reasoning: Some(_is_reasoning),
                        }),
                        napi::threadsafe_function::ThreadsafeFunctionCallMode::NonBlocking,
                    );
                }
            )?

            if token_id == $eos {
                $reason = String::from("stop");
                break;
            }

            if let Some(reason) = $crate::sampling::check_repetition_cutoff(
                &$gen,
                $p.max_consecutive_tokens,
                $p.max_ngram_repeats,
                $p.ngram_size,
            ) {
                $reason = reason.to_string();
                break;
            }

            match next_y {
                Some(next) => $y = next,
                None => break,
            }

            $profiler.step();

            if (step + 1) % 256 == 0 {
                $crate::array::synchronize_and_clear_cache();
            }
        }

        $profiler.snapshot_memory_after();
        $profiler.report();
    }};
}

pub(crate) use decode_loop;

/// The incremental-detokenization stream type from the `tokenizers`
/// crate, instantiated with the wrapper types `tokenizers::Tokenizer`
/// uses (same concrete type `Qwen3Tokenizer::step_decode_stream`
/// accepts). `'t` is the borrow of the underlying tokenizer.
// consumed from S7 family migrations; remove in S12
#[allow(dead_code)]
pub(crate) type TokDecodeStream<'t> = tokenizers::DecodeStream<
    't,
    tokenizers::ModelWrapper,
    tokenizers::NormalizerWrapper,
    tokenizers::PreTokenizerWrapper,
    tokenizers::PostProcessorWrapper,
    tokenizers::DecoderWrapper,
>;

/// Required arguments of [`run_decode_loop`] — mirrors the
/// `decode_loop!` macro's parameter list one-to-one, minus
/// `embedding_weight` (turn-constant; captured by the
/// [`DecodeStep`] impl at `begin_decode` time) and `ops` (now the
/// `step` trait object/impl).
// consumed from S7 family migrations; remove in S12
#[allow(dead_code)]
pub(crate) struct DecodeLoopArgs<'a> {
    /// First generated token (sampled from the prefill logits). The loop
    /// takes ownership — the macro's final reassignment was never
    /// observed by callers (see the `_final_sampled_token` note at the
    /// dense call site).
    pub y: MxArray,
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

/// Streaming sub-block arguments of [`run_decode_loop`] — mirrors the
/// macro's optional `streaming: { .. }` group. `'t` is the tokenizer
/// borrow backing the [`TokDecodeStream`].
///
/// `tokenizer` is the raw `tokenizers::Tokenizer` (what the macro
/// reached via `$tok.inner()`); call sites pass
/// `qwen3_tokenizer.inner()`.
// consumed from S7 family migrations; remove in S12
#[allow(dead_code)]
pub(crate) struct StreamingCtx<'s, 't> {
    pub callback: &'s dyn ChunkSink,
    pub cancelled: &'s AtomicBool,
    pub decode_stream: &'s mut TokDecodeStream<'t>,
    pub tokenizer: &'t tokenizers::Tokenizer,
    pub streamed_text_len: &'s mut usize,
    pub last_is_reasoning: &'s mut bool,
}

/// Generic decode loop over a [`DecodeStep`] — the faithful port of the
/// `decode_loop!` macro body (which stays until S12; families adopt this
/// fn from S7).
///
/// Behavior is byte-identical to the macro with exactly two intended
/// differences:
///   * (a) the throttled every-32-step trace reads
///     `step.trace_offset()` instead of hardcoding
///     `mlx_sys::mlx_qwen35_get_cache_offset()`; `None` skips the
///     `tracing::info!` line entirely (non-qwen3.5-compiled steppers).
///   * (b) `ChatStreamChunk` is referenced directly from
///     `engine::types` (the macro spelled the same type via
///     `$crate::engine::types::ChatStreamChunk`).
///
/// Everything else is preserved: pipelined next-graph build, budget
/// forcing via [`ReasoningTracker`], [`apply_all_penalties`],
/// `sampling::sample`, EOS + `check_repetition_cutoff` stops,
/// every-256-step `synchronize_and_clear_cache`, profiler
/// begin/end/mark/step calls, the `MLX_MTP_TRACE_LOGITS` diagnostic
/// block, and the streaming sub-block (cancellation,
/// `step_decode_stream` incremental detokenization with error recovery,
/// `include_reasoning` suppression, `is_reasoning` tagging).
// consumed from S7 family migrations; remove in S12
#[allow(dead_code)]
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
        generated_tokens,
        token_history,
        finish_reason,
        first_token_instant,
        report_perf,
        generation_stream,
    } = args;

    for step_idx in 0..max_new_tokens {
        let next_y = if step_idx + 1 < max_new_tokens {
            let _stream_ctx = StreamContext::new(generation_stream);

            profiler.begin("forward");
            let next_ids = y.reshape(&[1, 1])?;
            let (mut logits, needs_squeeze) = step.forward(&next_ids)?;
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
            // `token_history.len() + 1` (the current `y` sits at
            // `token_history.len()`). `budget_forced` skips the real
            // logits, so only trace the sampled path.
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
                            token_history.len() + 1,
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
                            token_history.len() + 1,
                            e.reason,
                        );
                    }
                }
            }

            Some(next_token)
        } else {
            None
        };

        profiler.begin("eval_token");
        y.eval();
        profiler.end();

        profiler.begin("extract");
        let token_id = y.item_at_int32(0)? as u32;
        profiler.end();
        profiler.mark_first_token();
        if report_perf && first_token_instant.is_none() {
            *first_token_instant = Some(Instant::now());
        }

        generated_tokens.push(token_id);
        token_history.push(token_id);
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
                "Qwen3.5 decode AR step={} sampled_token_id={} cache_offset={} gen_len={}",
                step_idx,
                token_id,
                cache_offset,
                generated_tokens.len(),
            );
        }

        // Streaming-only block (the macro's optional repetition group).
        if let Some(s) = streaming.as_mut() {
            *s.last_is_reasoning = is_reasoning;

            if s.cancelled.load(Ordering::Relaxed) {
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
            // Suppress reasoning (<think>…</think>) deltas from the
            // stream when include_reasoning == false. Detokenize +
            // length-advance above stay OUTSIDE this gate so
            // DecodeStream sees every token.
            if p.include_reasoning || !is_reasoning {
                s.callback.send(Ok(ChatStreamChunk {
                    text: token_text,
                    done: false,
                    finish_reason: None,
                    tool_calls: None,
                    thinking: None,
                    num_tokens: None,
                    prompt_tokens: None,
                    reasoning_tokens: None,
                    raw_text: None,
                    cached_tokens: None,
                    performance: None,
                    is_reasoning: Some(is_reasoning),
                }));
            }
        }

        if token_id == eos_id {
            *finish_reason = String::from("stop");
            break;
        }

        if let Some(reason) = crate::sampling::check_repetition_cutoff(
            generated_tokens,
            p.max_consecutive_tokens,
            p.max_ngram_repeats,
            p.ngram_size,
        ) {
            *finish_reason = reason.to_string();
            break;
        }

        match next_y {
            Some(next) => y = next,
            None => break,
        }

        profiler.step();

        if (step_idx + 1) % 256 == 0 {
            crate::array::synchronize_and_clear_cache();
        }
    }

    profiler.snapshot_memory_after();
    profiler.report();
    Ok(())
}

/// Policy decision for the C++ compiled paged forward fallback.
///
/// Inputs:
/// * `compiled_step_completed` — whether ANY compiled C++ paged step
///   has succeeded earlier in this turn.
///
/// Output:
/// * `true` — propagate the forward error as fatal. Returned when a
///   compiled step has previously succeeded; the C++ side has advanced
///   its per-layer GDN linear-cache globals (conv_state /
///   recurrent_state) but those updates are never imported back into
///   `self.caches`. Falling back to the pure-Rust paged decode after
///   that point would read stale pre-step state and silently corrupt
///   the response.
/// * `false` — safe to fall back to the pure-Rust paged decode.
///   Returned when no compiled step has succeeded yet; the only failure
///   mode at that point is an init/configuration mismatch caught at
///   first dispatch, which leaves `self.caches` consistent with
///   `paged_adapter` after a `rollback_last_tokens(1)`.
///
/// This mirrors the policy applied identically in the dense and MoE
/// sync + streaming decode loops; extracting it as a stand-alone helper
/// keeps the tests in lockstep.
#[inline]
pub(crate) fn should_propagate_compiled_paged_error(compiled_step_completed: bool) -> bool {
    compiled_step_completed
}

#[cfg(test)]
mod run_decode_loop_tests {
    //! Mock-driven tests for [`run_decode_loop`] — a scripted
    //! [`DecodeStep`] steers the T=0 argmax through small-vocab logits
    //! so every macro-ported behavior (EOS stop, repetition cutoff,
    //! budget forcing, streaming suppression, the 256-step cache-clear
    //! cadence) can be pinned without loading a model.

    use std::sync::Mutex;
    use std::sync::atomic::AtomicBool;

    use napi::bindgen_prelude::*;

    use super::{DecodeLoopArgs, StreamingCtx, run_decode_loop};
    use crate::array::MxArray;
    use crate::decode_profiler::DecodeProfiler;
    use crate::engine::backend::{ChunkSink, DecodeStep};
    use crate::engine::params::{ChatParams, extract_chat_params};
    use crate::engine::penalties::ReasoningTracker;
    use crate::engine::types::{ChatConfig, ChatStreamChunk};
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

        let out = drive(&mut step, 3, &params, &mut tracker, 10, 7)
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

        let out = drive(&mut step, 5, &params, &mut tracker, 20, 7)
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

        let out = drive(&mut step, 4, &params, &mut tracker, 6, 7)
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
        // >300 steps so the every-256-step synchronize_and_clear_cache
        // branch executes at least once. Cutoffs disabled so the
        // constant script can't trip them.
        let params = greedy_params(|cfg| {
            cfg.max_consecutive_tokens = Some(0);
            cfg.max_ngram_repeats = Some(0);
        });
        let mut tracker = ReasoningTracker::new(false, None, None);
        let mut step = MockStep::new(vec![1], 16);

        let out = drive(&mut step, 1, &params, &mut tracker, 400, 15)
            .unwrap_or_else(|e| panic!("loop failed: {}", e.reason));

        assert_eq!(out.generated.len(), 400);
        assert!(out.generated.iter().all(|&t| t == 1));
        assert_eq!(out.finish_reason, "length");
        // Last iteration skips the pipelined forward (step+1 == max).
        assert_eq!(step.forward_calls, 399);
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
}

#[cfg(test)]
mod compiled_paged_fallback_policy_tests {
    use super::should_propagate_compiled_paged_error;

    /// Regression test: mid-turn fallback after a successful compiled
    /// step would corrupt the GDN linear cache state. The policy must
    /// propagate the error as fatal once any compiled step has completed;
    /// only the first-step failure is safe to fall back to pure-Rust decode.
    #[test]
    fn no_compiled_step_yet_allows_fallback() {
        assert!(
            !should_propagate_compiled_paged_error(false),
            "first-step compiled forward failure must allow fallback to pure-Rust paged decode \
             (self.caches is still consistent with paged_adapter pre-rollback)"
        );
    }

    #[test]
    fn after_successful_compiled_step_propagates_as_fatal() {
        assert!(
            should_propagate_compiled_paged_error(true),
            "compiled forward failure AFTER a successful compiled step must propagate as fatal: \
             the C++ GDN linear-cache globals advanced but self.caches is stale, so a pure-Rust \
             fallback would silently corrupt the response"
        );
    }
}
