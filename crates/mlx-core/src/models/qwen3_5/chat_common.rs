//! Shared chat/decode infrastructure for Qwen3.5 Dense and MoE models.
//!
//! Extracts identical boilerplate from the session entry points
//! (`chat_session_start_sync` / `chat_session_continue_sync` /
//! `chat_session_continue_tool_sync` and their `chat_stream_*` streaming
//! counterparts) across both model variants: config extraction, penalty
//! application, performance metrics, result finalization, and cache
//! management.

use std::hash::{DefaultHasher, Hash, Hasher};

use napi::bindgen_prelude::*;

use crate::array::MxArray;
use crate::model_thread::StreamTx;
use crate::sampling::{
    SamplingConfig, apply_frequency_penalty, apply_presence_penalty, apply_repetition_penalty,
};
use crate::tokenizer::{ChatMessage, Qwen3Tokenizer};
use crate::tools;

use super::layer_cache::Qwen3_5LayerCache;
use super::model::{ChatConfig, ChatResult, ChatStreamChunk};

/// Load-bearing typed error prefix used when `chat_session_continue_sync`
/// rejects an image parameter because images are changing mid-session.
///
/// Wire contract: when the Rust session-continue path detects that the
/// caller is trying to switch the active image set after a session has
/// already been initialized with different images, it returns a
/// `napi::Error` whose message begins with this prefix. The TypeScript
/// session layer pattern-matches the prefix to recognize the condition
/// and trigger an image-change restart (tearing down the old session
/// state and re-entering the `chat_session_start` path).
///
/// Because TS matches on the literal prefix, this constant MUST NOT
/// change without a coordinated update on both sides of the NAPI
/// boundary.
///
/// Introduced as part of the chat_common helper promotion.
pub(crate) const IMAGE_CHANGE_RESTART_PREFIX: &str = "IMAGE_CHANGE_REQUIRES_SESSION_RESTART:";

/// Hash raw image bytes to a u64 key for cache lookup.
fn hash_image_bytes(bytes: &[u8]) -> u64 {
    let mut hasher = DefaultHasher::new();
    bytes.hash(&mut hasher);
    hasher.finish()
}

/// Combine individual image hashes into a single cache key.
/// Order matters: different orderings of the same images produce different keys.
fn combine_image_hashes(hashes: &[u64]) -> u64 {
    let mut hasher = DefaultHasher::new();
    for h in hashes {
        h.hash(&mut hasher);
    }
    hasher.finish()
}

/// Compute a combined cache key from raw image bytes.
pub(crate) fn compute_image_cache_key(all_images: &[Vec<u8>]) -> u64 {
    let individual_hashes: Vec<u64> = all_images.iter().map(|img| hash_image_bytes(img)).collect();
    combine_image_hashes(&individual_hashes)
}

/// Build per-block extra_keys for the paged adapter's prefix-cache walk.
///
/// Phase 6 multimodal cache isolation: when the prompt contains image
/// tokens, the per-block extra_keys ensure that "same prompt + different
/// image" produces a cache miss (preventing stale-image KV reuse). For
/// text-only prompts (`token_image_positions` is empty), every block gets
/// an empty extra_keys vec — bit-equal to passing `&[]` uniformly to the
/// uniform `find_cached_prefix` / `finalize_turn_keep_live` API.
///
/// `total_tokens` is the FULL prompt length (cached prefix + new suffix
/// the request will write). The number of full blocks covered is
/// `total_tokens / block_size`; the trailing partial block (if any) is
/// not registered until full and so gets no entry here.
///
/// `token_image_positions` should be sorted by `token_pos` for stable
/// hashes (the helper preserves input order; reordered inputs would
/// produce different hashes). Today's Qwen3.5 paged dispatch is text-only
/// (image-bearing turns are routed to the flat path), so the production
/// call always passes `&[]` here. The hook stays in place so that when
/// VLM-paged forward integration lands, the call site only needs to swap
/// in the real image positions.
pub(crate) fn build_paged_extra_keys(
    total_tokens: usize,
    block_size: u32,
    token_image_positions: &[(u32, u64)],
) -> Vec<Vec<u64>> {
    let block_size_us = block_size as usize;
    if block_size_us == 0 {
        return Vec::new();
    }
    // Cover every block the request might register (full blocks only).
    // The adapter's per-block API tolerates an over-long vec by indexing
    // only what it needs, so erring high is safe.
    let num_blocks = total_tokens.div_ceil(block_size_us);
    crate::transformer::paged_kv_cache_adapter::compute_per_block_image_extra_keys(
        token_image_positions,
        num_blocks,
        block_size,
    )
}

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
/// Important: historically this helper emitted a fake `done: true`
/// `ChatStreamChunk` with `finish_reason: "error"`, which the TS side
/// treated as a successful final chunk and caused the session to
/// advance to a bricked turn 1. Do NOT reintroduce that pattern —
/// guard failures MUST come through as `Err` so the error path is
/// exercised.
pub(crate) fn send_stream_error(stream_tx: &StreamTx<ChatStreamChunk>, message: &str) {
    let _ = stream_tx.send(Err(napi::Error::from_reason(message.to_string())));
}

/// Build a synthetic `ChatMessage` wrapping a user-role text-only message.
///
/// Used by the session-continue paths to feed a single user turn through
/// `Qwen3Tokenizer::sanitize_messages_public` without leaking any of the
/// extended optional fields (tool calls, images, etc.) that a real client
/// request might carry. Those fields are deliberately set to `None` so
/// the sanitization pass only has to police the textual `content` field.
pub(crate) fn build_synthetic_user_message(user: &str) -> ChatMessage {
    ChatMessage {
        role: "user".to_string(),
        content: user.to_string(),
        tool_calls: None,
        tool_call_id: None,
        is_error: None,
        reasoning_content: None,
        images: None,
    }
}

/// Build the ChatML wire-format delta text for a session-continue turn.
///
/// The cached history ends on `<|im_end|>` (because `chat_session_start_sync`
/// uses `im_end_id` as eos). The leading `\n` closes that turn's line; then
/// we open a new user turn and prime an assistant turn.
///
/// When thinking mode is explicitly enabled (`reasoning_effort ∈ {"medium",
/// "high"}`) or left as default, the Qwen3.5 jinja template inserts
/// `<think>\n` after the assistant prelude — mirror that here so the delta
/// stays template-equivalent. When thinking is explicitly disabled
/// (`Some(false)`), omit the prefix so the first generated token is a
/// plain content token.
///
/// `sanitized_user` MUST already be passed through
/// `Qwen3Tokenizer::sanitize_messages_public` by the caller — this helper
/// does not re-sanitize.
pub(crate) fn build_chatml_continue_delta_text(
    sanitized_user: &str,
    enable_thinking: Option<bool>,
) -> String {
    let thinking_prefix = match enable_thinking {
        Some(false) => "",
        // None = template default (Qwen3.5: thinking on) and
        // Some(true) both take the thinking path.
        _ => "<think>\n",
    };
    format!(
        "\n<|im_start|>user\n{sanitized_user}<|im_end|>\n<|im_start|>assistant\n{thinking_prefix}",
    )
}

/// Build the ChatML wire-format delta text for a tool-result turn.
///
/// Qwen3.5's chat template renders tool-role messages as a `user` turn
/// wrapping the tool result in `<tool_response>` tags:
///
/// ```text
/// <|im_start|>user
/// <tool_response>
/// {content}
/// </tool_response><|im_end|>
/// ```
///
/// The `tool_call_id` is NOT rendered anywhere by the template — Qwen
/// identifies tool responses purely by position and wrapper tags, so we
/// intentionally drop it here. Callers may still log it for their own
/// bookkeeping, but it does not enter the wire format.
///
/// Like `build_chatml_continue_delta_text`, this helper assumes the cached
/// history ends on `<|im_end|>` and emits a leading `\n` to close that
/// turn's line. After the tool response we open an assistant turn ready
/// for the next generation step.
///
/// Thinking-prefix handling mirrors `build_chatml_continue_delta_text`:
/// when thinking mode is explicitly disabled (`Some(false)`), omit the
/// `<think>\n` prefix so the first generated token is a plain content
/// token. Otherwise (`None` / `Some(true)`) emit the `<think>\n` prefix,
/// matching what the Qwen3.5 jinja template does after the assistant
/// opener. Callers resolve `enable_thinking` from the current
/// `ChatConfig` via `resolve_enable_thinking` before calling this helper.
///
/// `is_error` is the model-facing failure cue: when `Some(true)`, the
/// shared [`crate::tokenizer::TOOL_ERROR_MARKER`] is prepended to
/// `content` inside the `<tool_response>` wrapper. The structured
/// `ChatMessage::is_error` field on the originating message is the
/// authoritative signal; the marker injection here only affects the
/// wire bytes the model decodes. `None` / `Some(false)` produce the
/// unmarked wire format and stay byte-equal to the pre-feature output.
pub(crate) fn build_chatml_tool_delta_text(
    _tool_call_id: &str,
    content: &str,
    enable_thinking: Option<bool>,
    is_error: Option<bool>,
) -> String {
    let thinking_prefix = match enable_thinking {
        Some(false) => "",
        // None = template default (Qwen3.5: thinking on) and
        // Some(true) both take the thinking path.
        _ => "<think>\n",
    };
    let rendered_content = crate::tokenizer::apply_tool_error_marker(content, is_error);
    format!(
        "\n<|im_start|>user\n<tool_response>\n{rendered_content}\n</tool_response><|im_end|>\n<|im_start|>assistant\n{thinking_prefix}",
    )
}

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

/// Default thinking-token budget for models whose chat template CANNOT suppress thinking
/// (e.g. LFM2). None = unlimited. Qwen3.5 must NOT call this (its template honors enable_thinking).
pub(crate) fn default_thinking_budget_for_effort(reasoning_effort: Option<&str>) -> Option<i32> {
    match reasoning_effort {
        Some("none") => Some(0),  // force </think> ASAP → minimal thinking
        Some("low") => Some(256), // small cap; short reasoning still leaves room to answer
        _ => None,                // medium/high/unset → unlimited (preserves current default)
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

    /// Number of tokens generated during reasoning (inside <think>...</think>).
    pub fn reasoning_token_count(&self) -> u32 {
        self.thinking_token_count.max(0) as u32
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

/// Whether the compiled init should re-apply the saved M-RoPE offset
/// (`cached_rope_deltas`) after building the decode graph.
///
/// The offset is saved only when a VLM prefill ran, so `has_saved_delta`
/// is effectively "the live KV cache encodes image attention". Two
/// callers need to re-apply it:
///   - **Fresh VLM prefill reusing a cached prefix** (`has_images &&
///     cached_prefix_len > 0`): the new turn shares its image grid with
///     the cached one, and the saved offset carries the image-adjusted
///     M-RoPE position forward into the rebuilt compiled graph.
///   - **Session delta continuation** (`is_delta`): the delta prefill
///     just ran on top of the live KV caches, which still encode the
///     prior VLM prefill's image attention. Without re-applying the
///     offset, the newly-built compiled graph would decode at a
///     sequential M-RoPE position and misposition all generated tokens
///     relative to the cached image patches.
///
/// Pure function — extracted so the decision can be unit-tested
/// without instantiating the compiled decoder.
pub(crate) fn should_reapply_rope_delta(
    has_saved_delta: bool,
    is_delta: bool,
    has_images: bool,
    cached_prefix_len: usize,
) -> bool {
    has_saved_delta && (is_delta || (has_images && cached_prefix_len > 0))
}

/// Whether the compiled init should clear `cached_rope_deltas` after
/// building the decode graph.
///
/// Only fresh text-only prefills clear the offset: they signal that the
/// non-delta cache-prefix verify dropped any prior image-bearing cache,
/// so the stored offset is stale. Delta continuations preserve the
/// offset so chained text-only turns on an image session keep the
/// image-adjusted M-RoPE position.
///
/// Pure function — extracted so the decision can be unit-tested.
pub(crate) fn should_clear_rope_delta(is_delta: bool, has_images: bool) -> bool {
    !has_images && !is_delta
}

/// Direct-ownership version of `save_cache_state` for dedicated-thread models.
///
/// Takes `&mut` refs instead of `Arc<RwLock<>>`. Used by Qwen3.5 Dense on
/// its dedicated model thread.
pub(crate) fn save_cache_state_direct(
    reuse_cache: bool,
    has_images: bool,
    generated_tokens: &[u32],
    finish_reason: &str,
    tokens: &[u32],
    expanded_tokens: Option<&[u32]>,
    image_cache_key: u64,
    cached_token_history: &mut Vec<u32>,
    cached_image_key: &mut Option<u64>,
    cached_rope_deltas: &mut Option<i32>,
    caches: &mut Option<Vec<Qwen3_5LayerCache>>,
) {
    if reuse_cache {
        let mut full_history = if has_images {
            expanded_tokens.unwrap_or(tokens).to_vec()
        } else {
            tokens.to_vec()
        };
        let history_tokens = if finish_reason == "length" && !generated_tokens.is_empty() {
            &generated_tokens[..generated_tokens.len() - 1]
        } else {
            generated_tokens
        };
        full_history.extend_from_slice(history_tokens);
        *cached_token_history = full_history;
        *cached_image_key = if has_images {
            Some(image_cache_key)
        } else {
            None
        };
    } else {
        *caches = None;
        cached_token_history.clear();
        *cached_image_key = None;
        *cached_rope_deltas = None;
    }
}

/// Commit session state after a text-only delta continuation.
///
/// The delta path (`chat_tokens_delta_sync` / `chat_stream_tokens_delta_sync`)
/// appends a text delta on top of the live KV caches without touching the
/// image attention state baked in by the preceding prefill. The "current
/// turn is text-only" signal (`has_images == false`) MUST NOT be conflated
/// with "the session has no image context" — the KV caches still encode
/// every image patch from the earlier `chat_session_start` / VLM prefill,
/// and clearing `cached_image_key` here would make the next cache-prefix
/// verify think the session is pure text and accept a future image-carrying
/// turn via the delta path (which produces garbage because the mrope
/// offset `cached_rope_deltas` is stale for the new image grid).
///
/// This helper is identical to [`save_cache_state_direct`] except that it
/// leaves `cached_image_key` untouched on the `reuse_cache=true` branch.
/// The full-reset `reuse_cache=false` branch still clears everything —
/// same invariant as the prefill helper.
#[allow(clippy::too_many_arguments)]
pub(crate) fn save_cache_state_after_delta(
    reuse_cache: bool,
    generated_tokens: &[u32],
    finish_reason: &str,
    save_tokens: &[u32],
    cached_token_history: &mut Vec<u32>,
    cached_image_key: &mut Option<u64>,
    cached_rope_deltas: &mut Option<i32>,
    caches: &mut Option<Vec<Qwen3_5LayerCache>>,
) {
    if reuse_cache {
        let mut full_history = save_tokens.to_vec();
        let history_tokens = if finish_reason == "length" && !generated_tokens.is_empty() {
            &generated_tokens[..generated_tokens.len() - 1]
        } else {
            generated_tokens
        };
        full_history.extend_from_slice(history_tokens);
        *cached_token_history = full_history;
        // `cached_image_key` intentionally preserved — see doc comment.
    } else {
        *caches = None;
        cached_token_history.clear();
        *cached_image_key = None;
        *cached_rope_deltas = None;
    }
}

/// Direct-ownership version of `verify_cache_prefix` for dedicated-thread models.
///
/// Takes direct refs instead of `Arc<RwLock<>>`. Used by Qwen3.5 Dense on
/// its dedicated model thread.
///
/// # Return-value invariant (load-bearing)
///
/// This helper returns **either `0` (cache miss — caller MUST reset caches
/// before prefill) or `cached.len()` (exact-append hit — the new prompt
/// strictly extends the cached history)**. It **never** returns an
/// intermediate value such as "the first K tokens match, rewind to K".
///
/// That all-or-nothing contract is what makes it safe to drive Qwen3.5's
/// **hybrid linear + attention stack**. The Gated Delta Net (GDN) layers
/// carry a *recurrent* state (`conv_state`, `recurrent_state` in
/// [`super::layer_cache::Qwen3_5LayerCache::Linear`]) that folds every
/// absorbed token irreversibly into its hidden state — unlike a standard
/// KV cache, a GDN cache **cannot be trimmed or rewound mid-sequence**
/// without corrupting the representation. A non-zero return from this
/// function therefore always means "the incoming tokens are a *pure append*
/// on top of the cached state; continue decoding from the current live
/// caches". No mid-sequence rewind ever happens.
///
/// Any future modification that would relax this contract (e.g. returning
/// a prefix count less than `cached.len()`) MUST simultaneously ensure the
/// caller either (a) restricts the relaxation to pure-KVCache models or
/// (b) introduces GDN-state checkpointing to enable mid-sequence rewinds.
/// Neither has been done — the invariant here is the sole reason the
/// refactor that moves `reset_caches_sync()` from the outer session-start
/// path into the `cached_prefix_len == 0` branch of `chat_sync_core` is
/// safe for Qwen3.5 Dense and MoE.
pub(crate) fn verify_cache_prefix_direct(
    reuse_cache: bool,
    has_images: bool,
    tokens: &[u32],
    tokens_for_matching: &[u32],
    image_cache_key: u64,
    cached_token_history: &[u32],
    cached_image_key: &Option<u64>,
    has_caches: bool,
) -> usize {
    if !reuse_cache {
        return 0;
    }
    let cached = cached_token_history;
    if has_images {
        if let Some(cached_key) = *cached_image_key
            && cached_key == image_cache_key
            && !cached.is_empty()
            && tokens_for_matching.len() >= cached.len()
            && tokens_for_matching[..cached.len()] == cached[..]
            && has_caches
        {
            return cached.len();
        }
        0
    } else if !cached.is_empty()
        && tokens.len() >= cached.len()
        && tokens[..cached.len()] == cached[..]
        && has_caches
    {
        cached.len()
    } else {
        0
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
                // Suppress reasoning (<think>…</think>) deltas from the stream
                // when include_reasoning == false. Detokenize + length-advance
                // above stay OUTSIDE this gate so DecodeStream sees every token.
                if $p.include_reasoning || !_is_reasoning {
                    $cb.call(
                        Ok($crate::models::qwen3_5::model::ChatStreamChunk {
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

    /// Regression test for review Finding 1 (HIGH): mid-turn fallback
    /// after a successful compiled step would corrupt the GDN linear
    /// cache state. The policy must propagate the error as fatal once
    /// any compiled step has completed; only the first-step failure is
    /// safe to fall back to pure-Rust decode.
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

#[cfg(test)]
mod tool_delta_marker_tests {
    //! Guard the structured `is_error` channel on
    //! `build_chatml_tool_delta_text`. The renderer injects the
    //! `TOOL_ERROR_MARKER` cue into the `<tool_response>` wire content
    //! only when the caller passes `Some(true)`. `None` and
    //! `Some(false)` keep the output byte-equal to the pre-feature
    //! behavior — guarding both the hot (successful) path and the
    //! explicit-false path against accidental drift.

    use super::build_chatml_tool_delta_text;
    use crate::tokenizer::TOOL_ERROR_MARKER;

    #[test]
    fn tool_delta_injects_marker_when_is_error_true() {
        // `Some(true)` must produce the marker prefix inside the
        // `<tool_response>` wrapper. The marker is the single shared
        // constant — using it directly here keeps the test in sync
        // with any future rename.
        let payload = "boom: connection refused";
        let rendered = build_chatml_tool_delta_text("call_fail", payload, None, Some(true));
        let expected_inner = format!("{TOOL_ERROR_MARKER}{payload}");
        assert!(
            rendered.contains(&expected_inner),
            "expected error marker inside <tool_response> wrapper; got:\n{rendered}",
        );
        // The wrapper itself must stay correct (we don't want to ship
        // a malformed delta that only the unflagged path renders right).
        assert!(
            rendered.contains("<tool_response>\n"),
            "wrapper open missing"
        );
        assert!(
            rendered.contains("</tool_response>"),
            "wrapper close missing"
        );
    }

    #[test]
    fn tool_delta_skips_marker_when_is_error_none() {
        // None = default; pre-feature output. The marker MUST NOT
        // appear anywhere in the wire text.
        let payload = "{\"temperature\": 72}";
        let rendered = build_chatml_tool_delta_text("call_ok", payload, None, None);
        assert!(
            !rendered.contains(TOOL_ERROR_MARKER),
            "marker leaked into unflagged delta:\n{rendered}",
        );
        assert!(
            rendered.contains(payload),
            "original content missing from delta:\n{rendered}",
        );
    }

    #[test]
    fn tool_delta_skips_marker_when_is_error_some_false() {
        // Explicit `Some(false)` is the same as `None` — only
        // `Some(true)` flips the marker on.
        let payload = "ok";
        let rendered = build_chatml_tool_delta_text("call_ok", payload, None, Some(false));
        assert!(
            !rendered.contains(TOOL_ERROR_MARKER),
            "marker leaked into Some(false) delta:\n{rendered}",
        );
    }

    #[test]
    fn tool_delta_does_not_remark_content_that_resembles_marker() {
        // The structured channel removes the collision concern: a
        // successful tool result whose literal content begins with the
        // marker text must NOT double-prefix the marker on its way
        // through the renderer.
        let suspicious = format!("{TOOL_ERROR_MARKER}this is a successful payload");
        let rendered = build_chatml_tool_delta_text("call_ok", &suspicious, None, None);
        // Exactly one occurrence — the original payload — no extra
        // prefix.
        let occurrences = rendered.matches(TOOL_ERROR_MARKER).count();
        assert_eq!(
            occurrences, 1,
            "marker count should be 1 (the original literal); got {occurrences} in:\n{rendered}",
        );
    }

    #[test]
    fn tool_delta_marker_interacts_correctly_with_thinking_prefix() {
        // The marker and the `<think>\n` prefix occupy different slots
        // in the delta. Both must render together when both are active:
        // marker inside `<tool_response>`, `<think>\n` after the
        // assistant opener.
        let rendered = build_chatml_tool_delta_text("call_fail", "boom", Some(true), Some(true));
        assert!(
            rendered.contains(&format!("{TOOL_ERROR_MARKER}boom")),
            "marker missing from thinking-enabled delta:\n{rendered}",
        );
        assert!(
            rendered.contains("<|im_start|>assistant\n<think>\n"),
            "thinking prefix missing from thinking-enabled delta:\n{rendered}",
        );
    }
}

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
    fn test_stream_reasoning_gate_predicate() {
        // Drives the boundary semantics the streaming send-gate relies on:
        // `observe_token` returns true for reasoning tokens INCLUDING the
        // `</think>` closer, and false for the first content token after.
        // The send-gate is `include_reasoning || !is_reasoning`.
        //
        // Token ids are chosen distinct from THINK_END_ID for the
        // reasoning/content tokens.
        let seq = [101u32, 102, THINK_END_ID, 301, 302];

        // include_reasoning == false: suppress the 3 reasoning tokens
        // (including the </think> closer), emit the 2 content tokens.
        {
            let mut tracker = ReasoningTracker::new(true, None, Some(THINK_END_ID));
            let include_reasoning = false;
            let gate: Vec<bool> = seq
                .iter()
                .map(|&tok| {
                    let is_reasoning = tracker.observe_token(tok);
                    include_reasoning || !is_reasoning
                })
                .collect();
            assert_eq!(gate, vec![false, false, false, true, true]);
        }

        // include_reasoning == true: emit everything.
        {
            let mut tracker = ReasoningTracker::new(true, None, Some(THINK_END_ID));
            let include_reasoning = true;
            let gate: Vec<bool> = seq
                .iter()
                .map(|&tok| {
                    let is_reasoning = tracker.observe_token(tok);
                    include_reasoning || !is_reasoning
                })
                .collect();
            assert_eq!(gate, vec![true, true, true, true, true]);
        }
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
    fn test_default_thinking_budget_for_effort() {
        // none → Some(0): force </think> ASAP (minimal thinking).
        assert_eq!(default_thinking_budget_for_effort(Some("none")), Some(0));
        // low → Some(256): small cap.
        assert_eq!(default_thinking_budget_for_effort(Some("low")), Some(256));
        // medium / high / unset / unknown → None (unlimited; preserves default).
        assert_eq!(default_thinking_budget_for_effort(Some("medium")), None);
        assert_eq!(default_thinking_budget_for_effort(Some("high")), None);
        assert_eq!(default_thinking_budget_for_effort(None), None);
        assert_eq!(default_thinking_budget_for_effort(Some("bogus")), None);
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
        // reasoning from being stripped. The earlier hand-rolled scanner picked
        // the earliest opener of either family and bailed when it had no close,
        // leaking the later block. Delegating to `tools::parse_thinking` (which
        // checks each tag family separately) strips the longcat block while
        // leaving the stray `<think>` literal — exactly matching the parsed
        // `thinking` boundary, so no reasoning CONTENT leaks.
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
        // is still extracted (the fix isolates reasoning, it does not drop real calls).
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
        // Adversarial (Codex No-ship): a `<tool_call>` opens inside `<think>` but its
        // `</tool_call>` lands after `</think>`, so the span straddles the reasoning
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
mod save_cache_state_after_delta_tests {
    //! Guards the sticky-`cached_image_key` invariant on the text-only
    //! delta path. Before the fix, `save_cache_state_direct(has_images:
    //! false, ...)` was called after every delta continuation, which
    //! cleared `cached_image_key` even though the live KV cache still
    //! encoded the prior prefill's image attention state. That
    //! contradicted the TS `ChatSession` routing contract (warm cache
    //! across text-only follow-ups) and caused the delta path to fail
    //! with a cryptic "chat_tokens_delta_sync is text-only; session
    //! currently holds image state" on the very next turn.
    use super::save_cache_state_after_delta;

    #[test]
    fn delta_preserves_cached_image_key_on_reuse_cache_true() {
        let mut cached_history: Vec<u32> = vec![1, 2, 3];
        let mut cached_image_key: Option<u64> = Some(0xdeadbeef);
        let mut cached_rope_deltas: Option<i32> = Some(5);
        let mut caches: Option<Vec<super::Qwen3_5LayerCache>> =
            Some(vec![super::Qwen3_5LayerCache::new_full_attention()]);

        save_cache_state_after_delta(
            /* reuse_cache */ true,
            /* generated_tokens */ &[10, 11],
            /* finish_reason */ "stop",
            /* save_tokens */ &[1, 2, 3, 4],
            &mut cached_history,
            &mut cached_image_key,
            &mut cached_rope_deltas,
            &mut caches,
        );

        // Token history extended: pre-decode snapshot + generated tokens
        assert_eq!(cached_history, vec![1, 2, 3, 4, 10, 11]);
        // Image key preserved — THE invariant under test
        assert_eq!(cached_image_key, Some(0xdeadbeef));
        // Other cache state untouched
        assert_eq!(cached_rope_deltas, Some(5));
        assert!(caches.is_some());
    }

    #[test]
    fn delta_drops_trailing_generated_token_on_length_stop() {
        // Matches `save_cache_state_direct` truncation semantics: if the
        // decode terminated at max_new_tokens, the last generated token
        // was cut off mid-stream and must not be persisted.
        let mut cached_history: Vec<u32> = vec![];
        let mut cached_image_key: Option<u64> = Some(42);
        let mut cached_rope_deltas: Option<i32> = None;
        let mut caches: Option<Vec<super::Qwen3_5LayerCache>> = None;

        save_cache_state_after_delta(
            true,
            &[10, 11, 12],
            "length",
            &[1, 2],
            &mut cached_history,
            &mut cached_image_key,
            &mut cached_rope_deltas,
            &mut caches,
        );

        assert_eq!(cached_history, vec![1, 2, 10, 11]);
        assert_eq!(cached_image_key, Some(42));
    }

    #[test]
    fn delta_full_reset_clears_everything_when_reuse_cache_false() {
        // `reuse_cache=false` is the cold-path invariant from the prefill
        // helper — when the caller opts out of cache reuse, every piece
        // of session state must be cleared regardless of whether the
        // image key was previously populated.
        let mut cached_history: Vec<u32> = vec![1, 2, 3];
        let mut cached_image_key: Option<u64> = Some(0xabc);
        let mut cached_rope_deltas: Option<i32> = Some(7);
        let mut caches: Option<Vec<super::Qwen3_5LayerCache>> =
            Some(vec![super::Qwen3_5LayerCache::new_linear()]);

        save_cache_state_after_delta(
            false,
            &[10],
            "stop",
            &[1],
            &mut cached_history,
            &mut cached_image_key,
            &mut cached_rope_deltas,
            &mut caches,
        );

        assert!(cached_history.is_empty());
        assert!(cached_image_key.is_none());
        assert!(cached_rope_deltas.is_none());
        assert!(caches.is_none());
    }

    #[test]
    fn delta_with_text_only_session_keeps_key_none() {
        // Sanity: if the session never had images, the delta must not
        // fabricate a key either.
        let mut cached_history: Vec<u32> = vec![];
        let mut cached_image_key: Option<u64> = None;
        let mut cached_rope_deltas: Option<i32> = None;
        let mut caches: Option<Vec<super::Qwen3_5LayerCache>> = None;

        save_cache_state_after_delta(
            true,
            &[42],
            "stop",
            &[1, 2],
            &mut cached_history,
            &mut cached_image_key,
            &mut cached_rope_deltas,
            &mut caches,
        );

        assert_eq!(cached_image_key, None);
        assert_eq!(cached_history, vec![1, 2, 42]);
    }
}

#[cfg(test)]
mod rope_delta_gate_tests {
    //! Guards the M-RoPE offset lifecycle across the compiled decode
    //! init branch. The prior bug hard-coded `has_images: false` on the
    //! delta path and unconditionally cleared `cached_rope_deltas`,
    //! which caused the compiled graph to decode text-only deltas at a
    //! sequential position instead of the image-adjusted position —
    //! mispositioning every generated token relative to the cached
    //! image patches baked in by the earlier VLM prefill.
    use super::{should_clear_rope_delta, should_reapply_rope_delta};

    // ---- should_reapply_rope_delta ----

    #[test]
    fn reapply_skipped_when_no_saved_delta() {
        // Text-only session, nothing to re-apply.
        assert!(!should_reapply_rope_delta(false, false, false, 0));
        // Image session with delta, but saved offset missing (fresh VLM
        // prefill clears it before setting — we never enter the gated
        // branch without a saved offset).
        assert!(!should_reapply_rope_delta(false, true, false, 0));
        assert!(!should_reapply_rope_delta(false, false, true, 100));
    }

    #[test]
    fn reapply_fires_on_fresh_vlm_cache_prefix_reuse() {
        // Fresh VLM prefill reusing a cached prefix: both `has_images`
        // AND a non-zero `cached_prefix_len` must be present. The saved
        // offset was written on the prior turn's VLM prefill, so a
        // matching key + prefix means we rebuild the compiled graph at
        // the same image-adjusted position.
        assert!(should_reapply_rope_delta(true, false, true, 100));
    }

    #[test]
    fn reapply_skipped_on_fresh_vlm_without_prefix_match() {
        // VLM prefill without prefix reuse (cached_prefix_len == 0):
        // the compiled init already ran the fresh prefill path, which
        // computed the offset from scratch via M-RoPE. No re-apply.
        assert!(!should_reapply_rope_delta(true, false, true, 0));
    }

    #[test]
    fn reapply_skipped_on_fresh_text_prefill() {
        // Fresh text prefill with no image state: the cache-prefix
        // verify already dropped any prior image-bearing cache, so the
        // saved offset is stale. `should_clear_rope_delta` handles that
        // case by nulling it; re-apply stays off.
        assert!(!should_reapply_rope_delta(true, false, false, 50));
        assert!(!should_reapply_rope_delta(true, false, false, 0));
    }

    #[test]
    fn reapply_fires_on_delta_continuation_with_saved_offset() {
        // THE invariant this fix introduces: delta continuations on an
        // image-bearing session re-apply the saved offset regardless of
        // `has_images` (which is always false on the delta path by
        // construction — delta prefills are text-only) and regardless
        // of `cached_prefix_len` (which is always 0 on the delta path
        // because the live KV cache already contains the full prior
        // history and the delta bypasses the prefix-match flow).
        assert!(should_reapply_rope_delta(true, true, false, 0));
    }

    #[test]
    fn reapply_fires_on_chained_delta_turns() {
        // Chained text-only deltas on the same image session: each
        // turn's compiled init must re-apply the offset so the session
        // stays positioned correctly. The save helper preserves
        // `cached_rope_deltas` on the reuse_cache branch, so the next
        // turn sees `has_saved_delta=true`.
        assert!(should_reapply_rope_delta(true, true, false, 0));
    }

    // ---- should_clear_rope_delta ----

    #[test]
    fn clear_fires_only_on_fresh_text_prefill() {
        // The ONE case where the saved offset is stale: a non-delta
        // text prefill. The cache-prefix verify already dropped any
        // prior image cache, so the offset has nothing valid to apply
        // to on the next turn.
        assert!(should_clear_rope_delta(false, false));
    }

    #[test]
    fn clear_skipped_on_delta_path() {
        // Delta continuations (text-only by construction) preserve the
        // offset — regression gate for the bug this fix addresses. The
        // live KV cache still encodes the prior VLM prefill's image
        // attention, so the next delta turn (and the one after that)
        // must re-apply the same saved offset.
        assert!(!should_clear_rope_delta(true, false));
    }

    #[test]
    fn clear_skipped_on_vlm_prefill() {
        // VLM prefill sets a fresh offset and must not nuke it after
        // init. The `is_delta` axis is false on the non-delta prefill
        // path; the `has_images` axis guards the clear.
        assert!(!should_clear_rope_delta(false, true));
    }

    #[test]
    fn clear_skipped_on_vlm_delta_combination() {
        // Belt-and-suspenders: even if a future caller ever set
        // `is_delta=true, has_images=true`, the clear stays off. No
        // current caller does this — the delta path rejects images at
        // entry — but the gate is written defensively.
        assert!(!should_clear_rope_delta(true, true));
    }
}

#[cfg(test)]
mod verify_cache_prefix_invariant_tests {
    //! Guards the all-or-nothing return-value invariant of
    //! `verify_cache_prefix_direct` documented on its rustdoc. The Qwen3.5
    //! chat_session_start refactor — which moves the unconditional
    //! `reset_caches_sync()` out of the outer session-start path and
    //! relies on verify returning either `0` or the full cached length
    //! to drive the in-core reset-on-miss branch — is **only** safe as
    //! long as this function never returns a mid-sequence prefix length.
    //! A regression here would silently let the caller resume decoding on
    //! a GDN recurrent state that no longer corresponds to the token
    //! prefix in the KV cache, corrupting every generated token.
    use super::verify_cache_prefix_direct;

    #[test]
    fn returns_zero_when_reuse_cache_disabled() {
        // `reuse_cache = false` short-circuits; everything else is
        // irrelevant. This is the "caller explicitly opted out" path.
        assert_eq!(
            verify_cache_prefix_direct(
                false,
                false,
                &[1, 2, 3, 4],
                &[1, 2, 3, 4],
                0,
                &[1, 2, 3],
                &None,
                true,
            ),
            0,
        );
    }

    #[test]
    fn returns_zero_when_no_caches() {
        // `has_caches = false` means the model has no live KV caches to
        // resume from — a full prefill is required even if the history
        // matches.
        assert_eq!(
            verify_cache_prefix_direct(
                true,
                false,
                &[1, 2, 3, 4],
                &[1, 2, 3, 4],
                0,
                &[1, 2, 3],
                &None,
                false,
            ),
            0,
        );
    }

    #[test]
    fn returns_zero_on_empty_history() {
        // First session-start turn: nothing cached yet, so we must
        // prefill the whole prompt.
        assert_eq!(
            verify_cache_prefix_direct(
                true,
                false,
                &[1, 2, 3, 4],
                &[1, 2, 3, 4],
                0,
                &[],
                &None,
                true,
            ),
            0,
        );
    }

    #[test]
    fn returns_zero_on_first_token_mismatch() {
        // Histories diverge at index 0 — no reusable prefix.
        assert_eq!(
            verify_cache_prefix_direct(
                true,
                false,
                &[9, 2, 3, 4],
                &[9, 2, 3, 4],
                0,
                &[1, 2, 3],
                &None,
                true,
            ),
            0,
        );
    }

    #[test]
    fn returns_zero_on_midsequence_mismatch() {
        // CRITICAL: histories match for 2 tokens then diverge. The
        // function MUST return 0 (full miss), NOT 2 (partial hit).
        // A partial hit would signal the caller to reuse only the first
        // 2 positions of the KV cache — which for the GDN linear layers
        // would require rewinding the recurrent state, which is
        // impossible. The all-or-nothing contract is what keeps this
        // safe.
        assert_eq!(
            verify_cache_prefix_direct(
                true,
                false,
                &[1, 2, 7, 4],
                &[1, 2, 7, 4],
                0,
                &[1, 2, 3],
                &None,
                true,
            ),
            0,
        );
    }

    #[test]
    fn returns_zero_on_shorter_new_prompt() {
        // New prompt is shorter than the cached history — can't be a
        // forward extension. Rewinding is infeasible (see above), so
        // return 0 and force a fresh prefill.
        assert_eq!(
            verify_cache_prefix_direct(
                true,
                false,
                &[1, 2],
                &[1, 2],
                0,
                &[1, 2, 3, 4, 5],
                &None,
                true,
            ),
            0,
        );
    }

    #[test]
    fn returns_full_length_on_exact_append_hit() {
        // Happy path: the new prompt is `cached + [extra]`. The function
        // returns `cached.len()` so the caller prefills only the delta
        // tail. This is the whole point of the cache-reuse machinery.
        let cached = vec![1u32, 2, 3, 4];
        let new_prompt = vec![1u32, 2, 3, 4, 5, 6];
        assert_eq!(
            verify_cache_prefix_direct(
                true,
                false,
                &new_prompt,
                &new_prompt,
                0,
                &cached,
                &None,
                true,
            ),
            cached.len(),
        );
    }

    #[test]
    fn returns_full_length_on_exact_match() {
        // Edge case: new prompt is byte-identical to cached. Returns
        // `cached.len()` — the caller's zero-delta guard then takes
        // over (see the matching comment in `qwen3_5/model.rs` and
        // `qwen3_5_moe/model.rs`).
        let cached = vec![1u32, 2, 3, 4];
        assert_eq!(
            verify_cache_prefix_direct(true, false, &cached, &cached, 0, &cached, &None, true,),
            cached.len(),
        );
    }

    #[test]
    fn returns_zero_on_image_key_mismatch() {
        // VLM path: cached image key differs from the current turn's
        // key — the images changed, so the cached KV state no longer
        // represents the new prompt's image attention. Full reset.
        let cached = vec![1u32, 2, 3];
        let new_prompt = vec![1u32, 2, 3, 4];
        assert_eq!(
            verify_cache_prefix_direct(
                true,
                true,
                &new_prompt,
                &new_prompt,
                /* new image key */ 999,
                &cached,
                &Some(42),
                true,
            ),
            0,
        );
    }

    #[test]
    fn returns_full_length_on_vlm_image_key_match() {
        // VLM happy path: same images, new text tail. Returns the
        // cached prefix length so the caller prefills only the delta.
        let cached = vec![1u32, 2, 3];
        let new_prompt = vec![1u32, 2, 3, 4, 5];
        assert_eq!(
            verify_cache_prefix_direct(
                true,
                true,
                &new_prompt,
                &new_prompt,
                42,
                &cached,
                &Some(42),
                true,
            ),
            cached.len(),
        );
    }

    #[test]
    fn returns_zero_on_vlm_missing_image_key() {
        // VLM turn but cached state carries no image key — the cache
        // came from a prior text-only exchange, not a VLM prefill.
        // Safety requires a fresh VLM prefill, not a reuse.
        let cached = vec![1u32, 2, 3];
        let new_prompt = vec![1u32, 2, 3, 4];
        assert_eq!(
            verify_cache_prefix_direct(
                true,
                true,
                &new_prompt,
                &new_prompt,
                42,
                &cached,
                &None,
                true,
            ),
            0,
        );
    }

    /// The contract-level invariant: across a broad sweep of inputs the
    /// return value is ALWAYS either `0` or `cached.len()`. Any
    /// intermediate value would corrupt GDN recurrent state on reuse.
    ///
    /// This property-style sweep is belt-and-suspenders on top of the
    /// targeted unit tests above: even if a future refactor changes
    /// branch structure, the invariant holds by construction.
    #[test]
    fn invariant_return_value_is_always_zero_or_cached_len() {
        let cached = vec![10u32, 20, 30, 40, 50];
        // Every prefix-plus-suffix combination and a selection of
        // divergent inputs.
        let candidates: Vec<Vec<u32>> = vec![
            vec![],
            vec![10],
            vec![10, 20],
            vec![10, 20, 30],
            vec![10, 20, 30, 40],
            cached.clone(),
            [cached.clone(), vec![60]].concat(),
            [cached.clone(), vec![60, 70, 80]].concat(),
            vec![99, 20, 30, 40, 50, 60],
            vec![10, 20, 99, 40, 50, 60],
            vec![10, 20, 30, 40, 99, 60],
        ];

        for candidate in &candidates {
            let result = verify_cache_prefix_direct(
                true, false, candidate, candidate, 0, &cached, &None, true,
            );
            assert!(
                result == 0 || result == cached.len(),
                "invariant violated: result={} for candidate={:?} (expected 0 or {})",
                result,
                candidate,
                cached.len(),
            );
        }
    }
}
