//! Chat-result finalization: performance metrics, thinking / tool-call
//! parsing, `ChatResult` assembly, and stream-error reporting.

use napi::bindgen_prelude::*;

use crate::engine::types::{ChatResult, ChatStreamChunk};
use crate::model_thread::StreamTx;
use crate::tokenizer::Qwen3Tokenizer;
use crate::tools;

/// Report a guard-violation error through the stream channel.
///
/// Used by the streaming session entry points (`chat_stream_session_*`
/// and `chat_stream_tokens_delta_sync`) to surface pre-decode guard
/// failures — text-only violations, missing tokenizer special tokens,
/// reuse_cache=false, empty delta, etc.
///
/// Sends an `Err(napi::Error::from_reason(message))` item into the
/// mpsc so the NAPI forwarding task invokes the TS callback with
/// `(err, null)`. On the TS side, `_runChatStream` pushes the error
/// onto its queue and throws it from the async generator, which
/// `ChatSession.sendStream` catches in its `try { ... } finally`
/// block. The finally clears `inFlight`, `sawFinal` stays false, and
/// `turnCount` is NOT incremented — so the next `sendStream()` call
/// re-routes through `chatStreamSessionStart` instead of trying to
/// continue a session that never initialized. The exception also
/// re-throws to the caller so the failure is observable.
///
/// Important: do NOT emit a fake `done: true` `ChatStreamChunk` with
/// `finish_reason: "error"` here. The TS side would treat it as a
/// successful final chunk and advance the session to a bricked turn 1.
/// Guard failures MUST come through as `Err` so the error path is
/// exercised.
pub(crate) fn send_stream_error(stream_tx: &StreamTx<ChatStreamChunk>, message: &str) {
    let _ = stream_tx.send(Err(napi::Error::from_reason(message.to_string())));
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
        // MTP acceptance is filled in post-hoc by the MTP decode paths
        // via `DecodeProfiler::fill_mtp_acceptance` — the profiler is in
        // scope there but not here. Stays `None` on autoregressive runs.
        mtp_mean_accepted_tokens: None,
        mtp_mean_accepted_tokens_total: None,
        mtp_acceptance_by_position: None,
        mtp_cycles: None,
        mtp_mean_depth: None,
        profile_phases: None,
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
        // No think_end_id in vocab — cannot do token-level detection. Isolate reasoning
        // from content at the text level BEFORE extracting tool calls, so a `<tool_call>`
        // nested inside a reasoning block is NOT surfaced as an executable tool call.
        // This matches the token-confirmed path (which parses tool calls only from the
        // post-`</think>` content) and the `raw_text` scrub, so on this fallback the
        // `text`, `tool_calls`, and `raw_text` fields stay consistent. A standalone
        // `<tool_call>` outside reasoning is preserved and still extracted.
        let content = tools::strip_reasoning_preserving_tools(text);
        let (clean, calls) = tools::parse_tool_calls(&content);
        // Thinking field keeps the prior fallback derivation (reasoning parsed from the
        // tool-stripped text, so an in-argument think tag of a standalone tool call does
        // not masquerade as reasoning).
        let (text_without_tools, _) = tools::parse_tool_calls(text);
        let thinking = tools::parse_thinking(&text_without_tools).1;
        (clean.trim().to_string(), calls, thinking)
    };

    // Suppress reasoning if not requested
    let thinking = if include_reasoning { thinking } else { None };

    (clean_text, tool_calls, thinking)
}

/// Build the `raw_text` field with the reasoning span removed when reasoning is
/// not requested.
///
/// `raw_text` is normally the verbatim decoded generation (including
/// `<think>…</think>`). When `include_reasoning` is false we additionally strip
/// the reasoning span so a direct `raw_text` consumer cannot recover the model's
/// chain-of-thought — matching the suppression already applied to the parsed
/// `thinking` field and the streamed reasoning deltas.
///
/// The post-`</think>` content is kept VERBATIM (tool-call markup, whitespace,
/// the model's exact bytes) so `raw_text`'s downstream uses (e.g. tool-call
/// markup recovery) keep working. The branch structure mirrors
/// `parse_thinking_and_tools` so the boundary is identical to the one used for
/// the parsed `thinking`/`text` fields.
pub(crate) fn raw_text_with_reasoning_suppressed(
    text: &str,
    generated_tokens: &[u32],
    thinking_enabled: bool,
    think_end_id: Option<u32>,
    think_end_str: Option<&str>,
    include_reasoning: bool,
) -> String {
    // Reasoning requested, or no-thinking mode (all output is content): verbatim.
    if include_reasoning || !thinking_enabled {
        return text.to_string();
    }
    if tools::has_think_end_token(generated_tokens, think_end_id) {
        // Confirmed </think>: keep everything after the FIRST occurrence verbatim.
        if let Some(tag) = think_end_str
            && let Some(close_pos) = text.find(tag)
        {
            return text[close_pos + tag.len()..].to_string();
        }
        // Token confirmed but tag string unavailable/unlocatable: fall through to
        // the text-level strip below.
    } else if think_end_id.is_some() {
        // Truncated generation (no </think> before EOS/max): all reasoning.
        return String::new();
    }
    // No think_end_id in vocab (or tag unlocatable): text-level scrub. Strips EVERY
    // reasoning block of BOTH `<think>`/`<longcat_think>` families (parse_thinking
    // alone only handles the first family) while preserving `<tool_call>…</tool_call>`
    // spans verbatim — so reasoning-looking tags inside tool arguments can't corrupt
    // the tool markup that `raw_text` consumers (e.g. server tool-call recovery) rely on.
    tools::strip_reasoning_preserving_tools(text)
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
    prompt_tokens: u32,
    reasoning_tokens: u32,
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

    let raw_text = raw_text_with_reasoning_suppressed(
        &text,
        generated_tokens,
        thinking_enabled,
        think_end_id,
        think_end_str,
        include_reasoning,
    );

    Ok(ChatResult {
        text: clean_text,
        tool_calls,
        thinking,
        num_tokens,
        prompt_tokens,
        reasoning_tokens,
        finish_reason,
        raw_text,
        // Callers that reused a cached prefix overwrite this via their own
        // `cached_prefix_len as u32` after this function returns. Defaulting
        // to zero keeps the behavior of callers that do not (yet) thread
        // the value through intact.
        cached_tokens: 0,
        performance,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const THINK_END_ID: u32 = 151668; // example </think> token ID

    #[test]
    fn test_raw_text_with_reasoning_suppressed() {
        // Token sequences: a sequence CONTAINING THINK_END_ID confirms </think>
        // (has_think_end_token == true); a sequence WITHOUT it but with a
        // think_end_id provided is a truncated generation.
        let confirmed_tokens = [101u32, 102, THINK_END_ID, 301, 302];
        let truncated_tokens = [101u32, 102, 103, 104]; // no THINK_END_ID

        // 1. include_reasoning == true → verbatim (reasoning span intact).
        let text = "<think>secret reasoning</think>\nVisible answer";
        let out = raw_text_with_reasoning_suppressed(
            text,
            &confirmed_tokens,
            true, // thinking_enabled
            Some(THINK_END_ID),
            Some("</think>"),
            true, // include_reasoning
        );
        assert_eq!(out, text, "include_reasoning=true must keep raw verbatim");

        // 2. include_reasoning == false + confirmed </think>: keep everything
        //    after the FIRST </think> VERBATIM, including a <tool_call> that
        //    lives in the content portion.
        let text = "<think>secret reasoning</think>\n<tool_call>{\"name\":\"f\"}</tool_call>";
        let out = raw_text_with_reasoning_suppressed(
            text,
            &confirmed_tokens,
            true,
            Some(THINK_END_ID),
            Some("</think>"),
            false,
        );
        assert_eq!(
            out, "\n<tool_call>{\"name\":\"f\"}</tool_call>",
            "must keep post-</think> content (incl. tool markup) verbatim"
        );
        assert!(
            !out.contains("<think>"),
            "no opening think tag should remain"
        );
        assert!(
            !out.contains("</think>"),
            "no closing think tag should remain"
        );
        assert!(
            out.contains("<tool_call>"),
            "tool-call markup must be preserved"
        );

        // 3. include_reasoning == false + truncated generation (think_end_id in
        //    vocab but NOT present in generated tokens): all output is reasoning.
        let text = "<think>unterminated reasoning that hit EOS";
        let out = raw_text_with_reasoning_suppressed(
            text,
            &truncated_tokens,
            true,
            Some(THINK_END_ID),
            Some("</think>"),
            false,
        );
        assert_eq!(out, "", "truncated generation scrubs to empty string");

        // 4. include_reasoning == false + thinking disabled: all output is
        //    content, so raw_text stays verbatim (even literal think tags).
        let text = "<think> is just literal text here";
        let out = raw_text_with_reasoning_suppressed(
            text,
            &confirmed_tokens,
            false, // thinking_enabled == false
            Some(THINK_END_ID),
            Some("</think>"),
            false,
        );
        assert_eq!(out, text, "no-thinking mode keeps raw verbatim");

        // 5. Text-level fallback: no think_end_id in vocab and no tag string.
        //    Paired <think>…</think> is stripped. The fallback delegates to
        //    `tools::parse_thinking` (the SAME function parse_thinking_and_tools'
        //    fallback uses) so the boundary is identical to the parsed `text`
        //    field — which means the remainder is trimmed (parse_thinking's
        //    strip_tag_blocks trims), e.g. "\nABC" -> "ABC". Boundary fidelity is
        //    the priority here over byte-verbatimness; no reasoning leaks.
        let text = "<think>r</think>\nABC";
        let out = raw_text_with_reasoning_suppressed(
            text,
            &confirmed_tokens,
            true,
            None, // think_end_id
            None, // think_end_str
            false,
        );
        assert_eq!(out, "ABC", "text-level fallback strips reasoning span");
        assert!(!out.contains("<think>"));
        assert!(!out.contains("</think>"));
        assert!(!out.contains('r'), "reasoning content must be gone");
    }

    #[test]
    fn test_strip_reasoning_span_multi_block() {
        // Text-level fallback (think_end_id == None && think_end_str == None)
        // must strip ALL paired <think>…</think> blocks, not just the first.
        // A single leftover block would leak chain-of-thought into raw_text even
        // though the parsed `thinking` field is fully suppressed.
        let text = "<think>a</think>\nmid\n<think>b</think>\nanswer";
        let out = raw_text_with_reasoning_suppressed(
            text,
            &[101u32, 102, 103], // no think_end_id present (forces fallback anyway)
            true,                // thinking_enabled
            None,                // think_end_id → text-level fallback
            None,                // think_end_str
            false,               // include_reasoning
        );
        assert!(
            !out.contains("<think>"),
            "no opening think tag may remain: {out:?}"
        );
        assert!(
            !out.contains("</think>"),
            "no closing think tag may remain: {out:?}"
        );
        assert!(
            out.contains("mid"),
            "non-reasoning content between blocks must survive: {out:?}"
        );
        assert!(
            out.contains("answer"),
            "trailing non-reasoning content must survive: {out:?}"
        );
    }

    #[test]
    fn test_raw_text_fallback_mixed_unmatched_tag() {
        // Regression: an UNMATCHED `<think>` opener appearing BEFORE a valid
        // `<longcat_think>…</longcat_think>` block must NOT prevent the longcat
        // reasoning from being stripped. `tools::parse_thinking` checks each tag
        // family separately, so it strips the longcat block while leaving the
        // stray `<think>` literal — exactly matching the parsed `thinking`
        // boundary, so no reasoning CONTENT leaks.
        let text = "prefix <think> literal <longcat_think>secret</longcat_think> suffix";
        let out = raw_text_with_reasoning_suppressed(
            text,
            &[101u32, 102, 103], // no think_end_id present → text-level fallback
            true,                // thinking_enabled
            None,                // think_end_id
            None,                // think_end_str
            false,               // include_reasoning
        );
        assert!(
            !out.contains("secret"),
            "longcat reasoning content must not leak: {out:?}"
        );
        assert!(
            !out.contains("longcat_think"),
            "no longcat reasoning tag may remain: {out:?}"
        );
        assert!(
            out.contains("prefix") && out.contains("suffix"),
            "non-reasoning content must survive: {out:?}"
        );
    }

    #[test]
    fn test_fallback_excludes_reasoning_nested_tool_call() {
        // No think_end_id in vocab (text-level fallback): a `<tool_call>` nested inside
        // a reasoning block must NOT be surfaced as an executable tool call — it is the
        // model THINKING about a call, not emitting one. This matches the token-confirmed
        // path (tool calls parsed only from post-`</think>` content) and the raw_text
        // scrub, keeping `tool_calls` consistent with `raw_text` on the fallback.
        let text =
            "<think>maybe I should <tool_call>{\"name\":\"f\"}</tool_call></think>\nfinal answer";
        let (clean, calls, thinking) = parse_thinking_and_tools(
            text,
            &[101u32, 102, 103], // no think_end_id present → fallback
            true,                // thinking_enabled
            None,                // think_end_id
            None,                // think_end_str
            true,                // include_reasoning (thinking field populated)
        );
        assert!(
            calls.is_empty(),
            "reasoning-nested tool call must not leak into tool_calls: {calls:?}"
        );
        assert!(
            clean.contains("final answer"),
            "post-reasoning content survives as text: {clean:?}"
        );
        assert!(
            thinking.is_some_and(|t| t.contains("maybe I should")),
            "reasoning is captured in the thinking field"
        );
    }

    #[test]
    fn test_fallback_extracts_standalone_tool_call_after_reasoning() {
        // Companion to the above: a `<tool_call>` OUTSIDE reasoning on the fallback path
        // is still extracted (isolating reasoning does not drop real calls).
        let text = "<think>let me call it</think>\n<tool_call>{\"name\":\"f\"}</tool_call>";
        let (_clean, calls, _thinking) =
            parse_thinking_and_tools(text, &[101u32, 102, 103], true, None, None, false);
        assert_eq!(
            calls.len(),
            1,
            "standalone tool call is extracted: {calls:?}"
        );
        assert_eq!(calls[0].name, "f");
    }

    #[test]
    fn test_fallback_straddling_tool_call_does_not_leak() {
        // A `<tool_call>` opens inside `<think>` but its `</tool_call>`
        // lands after `</think>`, so the span straddles the reasoning
        // boundary. On the no-think_end_id fallback, neither `tool_calls` nor the `text`
        // field may surface a call that began in reasoning, and the reasoning prefix must
        // not leak into `text`.
        let text = "<think>secret <tool_call><function=leak></think>\n<parameter=q>1</parameter></function></tool_call>";
        let (clean, calls, _thinking) =
            parse_thinking_and_tools(text, &[101u32, 102, 103], true, None, None, false);
        assert!(
            calls.is_empty(),
            "straddling reasoning-started tool call must not be extracted: {calls:?}"
        );
        assert!(
            !clean.contains("secret") && !clean.contains("<think>"),
            "reasoning prefix must not leak into text: {clean:?}"
        );
    }

    #[test]
    fn test_raw_text_straddling_tool_call_does_not_leak() {
        // Same straddling shape through the raw_text scrubber: with include_reasoning=false
        // the reasoning prefix and the reasoning-started tool markup must both be gone.
        let text = "<think>secret <tool_call><function=leak></think>\n<parameter=q>1</parameter></function></tool_call>";
        let out = raw_text_with_reasoning_suppressed(
            text,
            &[101u32, 102, 103], // no think_end_id present → text-level fallback
            true,                // thinking_enabled
            None,                // think_end_id
            None,                // think_end_str
            false,               // include_reasoning
        );
        assert!(
            !out.contains("secret") && !out.contains("<think>"),
            "reasoning prefix must not leak into raw_text: {out:?}"
        );
    }
}

#[cfg(test)]
mod compute_performance_metrics_tests {
    //! `compute_performance_metrics` is a plain numerator / ttft divider:
    //! `prefill_tokens_per_second = prefill_tokens_len / (ttft_ms/1000)`. The
    //! numerator choice (e.g. LFM2-paged's full-prompt count) MUST be made at
    //! the call site — the divider here applies whatever numerator it is given,
    //! verbatim. Cheap, deterministic, no GPU.
    use super::compute_performance_metrics;
    use std::time::{Duration, Instant};

    #[test]
    fn compute_performance_metrics_uses_given_numerator_directly() {
        let t0 = Instant::now();
        let ttft = Duration::from_millis(150);
        let first_tok = t0 + ttft;

        // Full-prompt numerator (1000) -> ~6667 tok/s; ttft ~= 150ms.
        let m = compute_performance_metrics(Some(t0), Some(first_tok), 1000, 8)
            .expect("metrics present when both instants are Some");
        assert!(
            (m.ttft_ms - 150.0).abs() < 5.0,
            "ttft_ms should reflect first_tok - gen_start (~150ms), got {}",
            m.ttft_ms
        );
        let expected_full = 1000.0 / 0.150;
        assert!(
            (m.prefill_tokens_per_second - expected_full).abs() / expected_full < 0.05,
            "full-prompt numerator must divide directly: expected ~{expected_full:.0}, got {}",
            m.prefill_tokens_per_second
        );

        // Suffix-scale numerator (6) -> ~40 tok/s (6 / 0.150). Proves the
        // function is a plain divider and the numerator is the load-bearing
        // choice.
        let m_suffix =
            compute_performance_metrics(Some(t0), Some(first_tok), 6, 8).expect("metrics present");
        let expected_suffix = 6.0 / 0.150;
        assert!(
            (m_suffix.prefill_tokens_per_second - expected_suffix).abs() / expected_suffix < 0.05,
            "suffix numerator divides directly: expected ~{expected_suffix:.0}, got {}",
            m_suffix.prefill_tokens_per_second
        );
        // And the full-prompt value is >5x the suffix value: a suffix
        // numerator under-reports by exactly the cached-prefix ratio.
        assert!(
            m.prefill_tokens_per_second > 5.0 * m_suffix.prefill_tokens_per_second,
            "full-prompt tok/s ({}) must be >5x suffix tok/s ({})",
            m.prefill_tokens_per_second,
            m_suffix.prefill_tokens_per_second
        );
    }
}
