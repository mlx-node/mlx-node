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

use napi::bindgen_prelude::*;

use crate::array::MxArray;

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
