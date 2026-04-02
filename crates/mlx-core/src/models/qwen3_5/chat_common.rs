//! Shared chat/decode infrastructure for Qwen3.5 Dense and MoE models.
//!
//! Extracts identical boilerplate from `chat()` and `chat_stream()` methods
//! across both model variants: config extraction, penalty application,
//! performance metrics, result finalization, and cache management.

use std::sync::{Arc, RwLock};

use napi::bindgen_prelude::*;

use crate::array::MxArray;
use crate::sampling::{
    SamplingConfig, apply_frequency_penalty, apply_presence_penalty, apply_repetition_penalty,
};
use crate::tokenizer::Qwen3Tokenizer;
use crate::tools;

use super::layer_cache::Qwen3_5LayerCache;
use super::model::{ChatConfig, ChatResult};

/// Extracted chat parameters with defaults applied.
pub(crate) struct ChatParams {
    pub max_new_tokens: i32,
    pub repetition_penalty: f64,
    pub repetition_context_size: i32,
    pub presence_penalty: f64,
    pub presence_context_size: i32,
    pub frequency_penalty: f64,
    pub frequency_context_size: i32,
    pub max_consecutive_tokens: i32,
    pub max_ngram_repeats: i32,
    pub ngram_size: i32,
    pub sampling_config: Option<SamplingConfig>,
    pub report_performance: bool,
    pub reuse_cache: bool,
    pub thinking_token_budget: Option<i32>,
    pub include_reasoning: bool,
}

/// Resolve the effective `enable_thinking` value from `reasoning_effort`.
///
/// In vLLM, `enable_thinking` is a low-level template kwarg nested inside
/// `chat_template_kwargs`. `reasoning_effort` is the user-facing control that
/// drives it. This function maps the user-facing API to the template parameter.
pub(crate) fn resolve_enable_thinking(config: &ChatConfig) -> Option<bool> {
    match config.reasoning_effort.as_deref() {
        Some("none") | Some("low") => Some(false),
        Some("medium") | Some("high") => Some(true),
        _ => None, // not set → default (template decides, typically true)
    }
}

/// Resolve `include_reasoning` from config, with `reasoning_effort: "none"` default.
pub(crate) fn resolve_include_reasoning(config: &ChatConfig) -> bool {
    config
        .include_reasoning
        .unwrap_or(!matches!(config.reasoning_effort.as_deref(), Some("none")))
}

/// Extract ChatConfig fields into flat variables with defaults.
pub(crate) fn extract_chat_params(config: &ChatConfig) -> ChatParams {
    ChatParams {
        max_new_tokens: config.max_new_tokens.unwrap_or(2048),
        repetition_penalty: config.repetition_penalty.unwrap_or(1.0),
        repetition_context_size: config.repetition_context_size.unwrap_or(256),
        presence_penalty: config.presence_penalty.unwrap_or(0.0),
        presence_context_size: config.presence_context_size.unwrap_or(20),
        frequency_penalty: config.frequency_penalty.unwrap_or(0.0),
        frequency_context_size: config.frequency_context_size.unwrap_or(20),
        max_consecutive_tokens: config.max_consecutive_tokens.unwrap_or(16),
        max_ngram_repeats: config.max_ngram_repeats.unwrap_or(3),
        ngram_size: config.ngram_size.unwrap_or(64),
        sampling_config: Some(SamplingConfig {
            temperature: config.temperature,
            top_k: config.top_k,
            top_p: config.top_p,
            min_p: config.min_p,
        }),
        report_performance: config.report_performance.unwrap_or(false),
        reuse_cache: config.reuse_cache.unwrap_or(true),
        thinking_token_budget: config.thinking_token_budget,
        include_reasoning: resolve_include_reasoning(config),
    }
}

/// Apply repetition + presence + frequency penalties to logits.
pub(crate) fn apply_all_penalties(
    mut logits: MxArray,
    token_history: &[u32],
    params: &ChatParams,
) -> Result<MxArray> {
    if params.repetition_penalty != 1.0 && !token_history.is_empty() {
        logits = apply_repetition_penalty(
            &logits,
            token_history,
            params.repetition_penalty,
            Some(params.repetition_context_size),
        )?;
    }
    if params.presence_penalty != 0.0 {
        logits = apply_presence_penalty(
            &logits,
            token_history,
            params.presence_penalty,
            Some(params.presence_context_size),
        )?;
    }
    if params.frequency_penalty != 0.0 {
        logits = apply_frequency_penalty(
            &logits,
            token_history,
            params.frequency_penalty,
            Some(params.frequency_context_size),
        )?;
    }
    Ok(logits)
}

/// Tracks reasoning vs content state during token-by-token generation.
///
/// For Qwen3.5: the template injects `<think>\n` when thinking is enabled.
/// The model generates thinking tokens, then emits `</think>` (think_end_id),
/// then generates content. This tracker detects the transition at the TOKEN
/// level — no text parsing needed during decoding.
pub(crate) struct ReasoningTracker {
    in_thinking: bool,
    thinking_token_count: i32,
    budget: Option<i32>,
    think_end_id: Option<u32>,
    force_think_end: bool,
    /// Set after `should_force_think_end` is consumed, prevents re-triggering
    /// from subsequent `observe_token` calls before the forced token is extracted.
    end_scheduled: bool,
}

impl ReasoningTracker {
    /// Create a new tracker.
    ///
    /// `starts_in_thinking`: true when the template injected `<think>\n` (thinking enabled).
    /// `budget`: maximum thinking tokens before forcing `</think>`. None = unlimited.
    /// `think_end_id`: token ID for `</think>` from the tokenizer vocabulary.
    pub fn new(starts_in_thinking: bool, budget: Option<i32>, think_end_id: Option<u32>) -> Self {
        // Budget=0 means "no thinking tokens at all" — force </think> immediately
        // on the first decode step, before any thinking token is generated.
        let force_immediately = starts_in_thinking && budget == Some(0) && think_end_id.is_some();
        Self {
            in_thinking: starts_in_thinking,
            thinking_token_count: 0,
            budget,
            think_end_id,
            force_think_end: force_immediately,
            end_scheduled: false,
        }
    }

    /// Process a generated token. Returns whether this token is reasoning content.
    ///
    /// Call AFTER extracting the token ID from the GPU each decode step.
    pub fn observe_token(&mut self, token_id: u32) -> bool {
        if !self.in_thinking {
            return false;
        }

        if self.think_end_id == Some(token_id) {
            self.in_thinking = false;
            self.force_think_end = false;
            self.end_scheduled = false;
            return true; // </think> itself is part of reasoning
        }

        self.thinking_token_count += 1;
        if let Some(budget) = self.budget
            && self.thinking_token_count >= budget
            && !self.end_scheduled
        {
            self.force_think_end = true;
        }
        true
    }

    /// Whether the next token should be forced to think_end_id.
    /// Consumes the flag — returns true at most once per budget trigger.
    ///
    /// Check this BEFORE building the next decode step's graph.
    pub fn should_force_think_end(&mut self) -> bool {
        if self.force_think_end && self.think_end_id.is_some() {
            self.force_think_end = false;
            self.end_scheduled = true;
            true
        } else {
            false
        }
    }

    /// The think_end token ID to force. Only valid when `should_force_think_end()` returned true.
    pub fn forced_token_id(&self) -> u32 {
        self.think_end_id
            .expect("should_force_think_end was true but think_end_id is None")
    }
}

/// Compute TTFT / prefill tok/s / decode tok/s performance metrics.
pub(crate) fn compute_performance_metrics(
    generation_start: Option<std::time::Instant>,
    first_token_instant: Option<std::time::Instant>,
    prefill_tokens_len: usize,
    generated_tokens_len: usize,
) -> Option<crate::profiling::PerformanceMetrics> {
    let (gen_start, first_tok) = match (generation_start, first_token_instant) {
        (Some(gs), Some(ft)) => (gs, ft),
        _ => return None,
    };
    let generation_end = std::time::Instant::now();
    let actual_prefill_toks = prefill_tokens_len as f64;
    let gen_toks = generated_tokens_len as f64;
    let ttft_ms = first_tok.duration_since(gen_start).as_secs_f64() * 1000.0;
    let decode_ms = generation_end.duration_since(first_tok).as_secs_f64() * 1000.0;
    Some(crate::profiling::PerformanceMetrics {
        ttft_ms,
        prefill_tokens_per_second: if ttft_ms > 0.0 {
            actual_prefill_toks / (ttft_ms / 1000.0)
        } else {
            0.0
        },
        decode_tokens_per_second: if decode_ms > 0.0 && gen_toks > 1.0 {
            (gen_toks - 1.0) / (decode_ms / 1000.0)
        } else {
            0.0
        },
    })
}

/// Shared finalization: parse thinking + tool calls from decoded text.
///
/// Four-way branching based on the request's reasoning state:
/// 1. `!thinking_enabled`: no-thinking mode — all text is content, no reasoning parsing.
/// 2. `thinking_enabled` + `</think>` token confirmed: split at token-confirmed boundary.
/// 3. `thinking_enabled` + no `</think>` token + `think_end_id` exists: truncated generation.
/// 4. `thinking_enabled` + no `think_end_id` in vocab: text-level fallback via `split_at_think_end`.
///
/// `include_reasoning`: when false, thinking field is suppressed (set to None).
pub(crate) fn parse_thinking_and_tools(
    text: &str,
    generated_tokens: &[u32],
    thinking_enabled: bool,
    think_end_id: Option<u32>,
    think_end_str: Option<&str>,
    include_reasoning: bool,
) -> (String, Vec<tools::ToolCallResult>, Option<String>) {
    let (clean_text, tool_calls, thinking) = if !thinking_enabled {
        // No-thinking mode: all text is content, passed through verbatim.
        // Any literal <think> tags are normal model output, not markup.
        let (clean, calls) = tools::parse_tool_calls(text);
        (clean, calls, None)
    } else if tools::has_think_end_token(generated_tokens, think_end_id) {
        // Thinking mode with confirmed </think>: split at token boundary.
        tools::split_at_think_end(text, think_end_str)
    } else if think_end_id.is_some() {
        // Thinking mode, truncated (no </think> before EOS/max_tokens):
        // entire output is reasoning, no content.
        let thinking_text = text.trim();
        // Strip leading <think>/<longcat_think> from old-style templates
        // that emit it in the generated text.
        let thinking_text = thinking_text
            .strip_prefix("<think>")
            .or_else(|| thinking_text.strip_prefix("<longcat_think>"))
            .unwrap_or(thinking_text)
            .trim();
        let thinking = if thinking_text.is_empty() {
            None
        } else {
            Some(thinking_text.to_string())
        };
        (String::new(), vec![], thinking)
    } else {
        // No think_end_id in vocab — cannot do token-level detection.
        // Fall back to text-level parsing via split_at_think_end(None).
        tools::split_at_think_end(text, None)
    };

    // Suppress reasoning if not requested
    let thinking = if include_reasoning { thinking } else { None };

    (clean_text, tool_calls, thinking)
}

/// Decode tokens, parse thinking/tool_calls, build ChatResult.
pub(crate) fn finalize_chat_result(
    tokenizer: &Qwen3Tokenizer,
    generated_tokens: &[u32],
    finish_reason: String,
    think_end_id: Option<u32>,
    think_end_str: Option<&str>,
    performance: Option<crate::profiling::PerformanceMetrics>,
    include_reasoning: bool,
    thinking_enabled: bool,
) -> Result<ChatResult> {
    let text = tokenizer
        .decode_sync(generated_tokens, true)
        .unwrap_or_else(|e| {
            tracing::warn!("Failed to decode generated tokens: {}", e);
            String::new()
        });

    let num_tokens = generated_tokens.len() as u32;

    let (clean_text, tool_calls, thinking) = parse_thinking_and_tools(
        &text,
        generated_tokens,
        thinking_enabled,
        think_end_id,
        think_end_str,
        include_reasoning,
    );

    // If we have valid tool calls, override finish reason
    let finish_reason = if tool_calls.iter().any(|tc| tc.status == "ok") {
        "tool_calls".to_string()
    } else {
        finish_reason
    };

    Ok(ChatResult {
        text: clean_text,
        tool_calls,
        thinking,
        num_tokens,
        finish_reason,
        raw_text: text,
        performance,
    })
}

/// Save or clear cache state after generation.
pub(crate) fn save_cache_state(
    reuse_cache: bool,
    has_images: bool,
    generated_tokens: &[u32],
    finish_reason: &str,
    tokens: &[u32],
    expanded_tokens: Option<&[u32]>,
    image_cache_key: u64,
    cached_token_history_arc: &Arc<RwLock<Vec<u32>>>,
    cached_image_key_arc: &Arc<RwLock<Option<u64>>>,
    cached_rope_deltas_arc: &Arc<RwLock<Option<i32>>>,
    caches_arc: &Arc<RwLock<Option<Vec<Qwen3_5LayerCache>>>>,
) -> Result<()> {
    if reuse_cache {
        let mut full_history = if has_images {
            expanded_tokens.unwrap_or(tokens).to_vec()
        } else {
            tokens.to_vec()
        };
        // Only include tokens that were actually forwarded through the model.
        // When stopped at max_tokens ("length"), the last token was never forwarded
        // (the pipelined loop skips forward on the final step).
        let history_tokens = if finish_reason == "length" && !generated_tokens.is_empty() {
            &generated_tokens[..generated_tokens.len() - 1]
        } else {
            generated_tokens
        };
        full_history.extend_from_slice(history_tokens);
        if let Ok(mut th) = cached_token_history_arc.write() {
            *th = full_history;
        }
        if let Ok(mut ik) = cached_image_key_arc.write() {
            *ik = if has_images {
                Some(image_cache_key)
            } else {
                None
            };
        }
    } else {
        // reuseCache: false — clear all cache state to free GPU memory
        if let Ok(mut cg) = caches_arc.write() {
            *cg = None;
        }
        if let Ok(mut th) = cached_token_history_arc.write() {
            th.clear();
        }
        if let Ok(mut ik) = cached_image_key_arc.write() {
            *ik = None;
        }
        if let Ok(mut rd) = cached_rope_deltas_arc.write() {
            *rd = None;
        }
    }
    Ok(())
}

/// Verify cache prefix match for reuse. Returns the number of cached tokens
/// that match the current input (0 if no match).
pub(crate) fn verify_cache_prefix(
    reuse_cache: bool,
    has_images: bool,
    tokens: &[u32],
    tokens_for_matching: &[u32],
    image_cache_key: u64,
    cached_token_history: &[u32],
    cached_image_key: &Arc<RwLock<Option<u64>>>,
    has_caches: bool,
) -> Result<usize> {
    if !reuse_cache {
        return Ok(0);
    }
    let cached = cached_token_history;
    if has_images {
        let cached_img_key_guard = cached_image_key
            .read()
            .map_err(|_| Error::from_reason("Failed to read cached image key"))?;
        if let Some(cached_key) = *cached_img_key_guard
            && cached_key == image_cache_key
            && !cached.is_empty()
            && tokens_for_matching.len() >= cached.len()
            && tokens_for_matching[..cached.len()] == cached[..]
            && has_caches
        {
            return Ok(cached.len());
        }
        Ok(0)
    } else if !cached.is_empty()
        && tokens.len() >= cached.len()
        && tokens[..cached.len()] == cached[..]
        && has_caches
    {
        Ok(cached.len())
    } else {
        Ok(0)
    }
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
                        let forced_id = $tracker.forced_token_id() as i32;
                        ($crate::array::MxArray::from_int32(&[forced_id], &[1])?, true)
                    } else {
                        $profiler.begin("rep_penalty");
                        logits = $crate::models::qwen3_5::chat_common::apply_all_penalties(
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
                $cb.call(
                    Ok($crate::models::qwen3_5::model::ChatStreamChunk {
                        text: token_text,
                        done: false,
                        finish_reason: None,
                        tool_calls: None,
                        thinking: None,
                        num_tokens: None,
                        raw_text: None,
                        performance: None,
                        is_reasoning: Some(_is_reasoning),
                    }),
                    napi::threadsafe_function::ThreadsafeFunctionCallMode::NonBlocking,
                );
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

#[cfg(test)]
mod tests {
    use super::*;

    const THINK_END_ID: u32 = 151668; // example </think> token ID

    #[test]
    fn test_tracker_starts_in_thinking() {
        let mut tracker = ReasoningTracker::new(true, None, Some(THINK_END_ID));
        assert!(tracker.observe_token(100)); // reasoning
        assert!(tracker.observe_token(200)); // reasoning
        assert!(!tracker.should_force_think_end());
    }

    #[test]
    fn test_tracker_transitions_on_think_end() {
        let mut tracker = ReasoningTracker::new(true, None, Some(THINK_END_ID));
        assert!(tracker.observe_token(100)); // reasoning
        assert!(tracker.observe_token(THINK_END_ID)); // </think> is still reasoning
        assert!(!tracker.observe_token(300)); // now content
        assert!(!tracker.observe_token(400)); // still content
    }

    #[test]
    fn test_tracker_starts_in_content() {
        let mut tracker = ReasoningTracker::new(false, None, Some(THINK_END_ID));
        assert!(!tracker.observe_token(100));
        assert!(!tracker.observe_token(200));
        assert!(!tracker.should_force_think_end());
    }

    #[test]
    fn test_tracker_budget_enforcement() {
        // Budget=3: allows exactly 3 thinking tokens, then forces on the 3rd.
        let mut tracker = ReasoningTracker::new(true, Some(3), Some(THINK_END_ID));
        assert!(tracker.observe_token(100)); // count→1
        assert!(!tracker.should_force_think_end());
        assert!(tracker.observe_token(200)); // count→2
        assert!(!tracker.should_force_think_end());
        assert!(tracker.observe_token(300)); // count→3, 3>=3 → force!
        assert!(tracker.should_force_think_end());
        assert_eq!(tracker.forced_token_id(), THINK_END_ID);
    }

    #[test]
    fn test_tracker_budget_zero() {
        // Budget=0: force is set in new() — triggers BEFORE any thinking token.
        let mut tracker = ReasoningTracker::new(true, Some(0), Some(THINK_END_ID));
        assert!(tracker.should_force_think_end()); // immediate, no observe needed
    }

    #[test]
    fn test_tracker_budget_zero_vs_one() {
        // Budget=0: force immediately (0 thinking tokens allowed).
        let mut t0 = ReasoningTracker::new(true, Some(0), Some(THINK_END_ID));
        assert!(t0.should_force_think_end()); // before any observe

        // Budget=1: allows exactly 1 thinking token before forcing.
        let mut t1 = ReasoningTracker::new(true, Some(1), Some(THINK_END_ID));
        assert!(!t1.should_force_think_end()); // not yet
        assert!(t1.observe_token(100)); // count→1, 1>=1 → force!
        assert!(t1.should_force_think_end()); // triggers after 1st token
    }

    #[test]
    fn test_tracker_budget_clears_on_think_end() {
        let mut tracker = ReasoningTracker::new(true, Some(2), Some(THINK_END_ID));
        assert!(tracker.observe_token(100)); // count→1
        assert!(!tracker.should_force_think_end());
        assert!(tracker.observe_token(200)); // count→2, 2>=2 → force!
        assert!(tracker.should_force_think_end());
        // When the forced think_end token is generated:
        assert!(tracker.observe_token(THINK_END_ID)); // transitions to content
        assert!(!tracker.should_force_think_end()); // force cleared
        assert!(!tracker.observe_token(300)); // now content
    }

    #[test]
    fn test_tracker_no_double_force_with_pipeline_lag() {
        // Simulates pipelined decode: after should_force_think_end() is consumed,
        // the pipeline extracts an over-budget token before the forced </think>
        // arrives. The tracker must NOT re-trigger forcing.
        let mut tracker = ReasoningTracker::new(true, Some(3), Some(THINK_END_ID));
        tracker.observe_token(100); // count→1
        tracker.observe_token(200); // count→2
        tracker.observe_token(300); // count→3, 3>=3 → force=true

        // Phase A of step N+1: consume the force flag
        assert!(tracker.should_force_think_end()); // returns true, sets end_scheduled
        assert!(!tracker.should_force_think_end()); // already consumed — must be false

        // Phase B of step N+1: the pipeline extracts the over-budget token (not </think>)
        assert!(tracker.observe_token(400)); // still reasoning, count→4
        // Must NOT re-trigger forcing despite count(4) >= budget(3)
        assert!(!tracker.should_force_think_end());

        // Phase B of step N+2: the forced </think> token is finally extracted
        assert!(tracker.observe_token(THINK_END_ID)); // transitions to content
        assert!(!tracker.should_force_think_end());

        // Phase B of step N+3: normal content token
        assert!(!tracker.observe_token(500)); // content
    }

    #[test]
    fn test_tracker_no_budget() {
        let mut tracker = ReasoningTracker::new(true, None, Some(THINK_END_ID));
        for i in 0..1000 {
            assert!(tracker.observe_token(i));
            assert!(!tracker.should_force_think_end());
        }
    }

    #[test]
    fn test_tracker_no_think_end_id() {
        let mut tracker = ReasoningTracker::new(true, Some(5), None);
        // Without think_end_id, should_force_think_end is always false
        for i in 0..100 {
            tracker.observe_token(i);
            assert!(!tracker.should_force_think_end());
        }
    }

    #[test]
    fn test_tracker_no_think_end_id_labels_as_reasoning() {
        // When thinking is enabled but think_end_id is missing (tokenizer
        // renders </think> as multiple tokens), observe_token should still
        // return true (reasoning) for every token — consistent with the
        // text-level finalization that will find reasoning via parsing.
        let mut tracker = ReasoningTracker::new(true, None, None);
        assert!(tracker.observe_token(100)); // reasoning
        assert!(tracker.observe_token(200)); // reasoning
        assert!(tracker.observe_token(300)); // reasoning
        // Never transitions — no think_end_id to match
        assert!(!tracker.should_force_think_end()); // budget disabled
    }
}
