//! Chat parameter extraction and request-budget helpers.
//!
//! `ChatConfig` → `ChatParams` extraction with defaults, the
//! reasoning-effort → thinking resolution helpers, the KV-capacity /
//! generated-capacity budget helpers, and the ChatML delta-text
//! builders used by the session-continue and tool-result paths.

use napi::bindgen_prelude::*;

use crate::engine::types::ChatConfig;
use crate::sampling::SamplingConfig;
use crate::tokenizer::ChatMessage;

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

/// Sampling + stop-token defaults read from a model's
/// `generation_config.json`, applied when a request leaves the matching
/// field unspecified.
///
/// Precedence is `request value > these defaults > the sampler's builtin
/// fallback`. A `None` sampling field here means the model ships no
/// default for it, so the request and then the builtin fallback decide.
/// `eos_token_ids` carries every stop id from the file's `eos_token_id`
/// (scalar or array); the engine merges them into the per-turn stop set
/// alongside the tokenizer's primary EOS.
#[derive(Debug, Clone, Default)]
pub struct ModelGenerationDefaults {
    pub temperature: Option<f64>,
    pub top_k: Option<i32>,
    pub top_p: Option<f64>,
    pub min_p: Option<f64>,
    pub repetition_penalty: Option<f64>,
    pub eos_token_ids: Vec<u32>,
}

/// Pre-fill any unspecified sampling field of `cfg` from `d`.
///
/// Each field is filled only when the request left it `None`, so an
/// explicit request value always wins. A `None` default field is a
/// no-op. Stop tokens (`eos_token_ids`) are NOT applied here — the engine
/// folds them in via [`crate::engine::backend::ChatBackend::extra_eos_ids`].
pub(crate) fn apply_generation_defaults(cfg: &mut ChatConfig, d: &ModelGenerationDefaults) {
    if cfg.temperature.is_none() {
        cfg.temperature = d.temperature;
    }
    if cfg.top_k.is_none() {
        cfg.top_k = d.top_k;
    }
    if cfg.top_p.is_none() {
        cfg.top_p = d.top_p;
    }
    if cfg.min_p.is_none() {
        cfg.min_p = d.min_p;
    }
    if cfg.repetition_penalty.is_none() {
        cfg.repetition_penalty = d.repetition_penalty;
    }
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
    pub include_reasoning: bool,
    /// MTP: opt-in flag enabling the Multi-Token Prediction speculative
    /// decode loop. Effective only on the dense compiled path AND when
    /// the model checkpoint carries an MTP head
    /// (`Qwen35Inner::has_mtp_weights`). The eager Rust forward, the
    /// paged path, MoE, and VLM decode loops all continue to use the
    /// single-token `decode_loop!` macro regardless. Default: `false`.
    pub enable_mtp: bool,
    /// MTP: number of draft tokens per speculative cycle, fed to the
    /// `forward_mtp_draft_compiled` / `forward_mtp_verify_compiled` FFI.
    /// Must be in `[1, 5]` to satisfy the verify-FFI contract. Default:
    /// 1 on the current bf16 MTP-head lane. When
    /// `mtp_adaptive_depth = true`, this value is only used as the
    /// initial depth — the `AdaptiveDepthPolicy` picks per-cycle.
    pub mtp_depth: usize,
    /// MTP: when true, the decode loop runs an `AdaptiveDepthPolicy`
    /// (`adaptive_depth.rs`) that picks the draft depth per cycle by
    /// maximising per-depth EMA of `accepted_tokens / cycle_wall_ns`.
    /// When false, the loop pins `mtp_depth` for every cycle.
    ///
    /// Default resolution (`extract_chat_params`):
    ///   * User set `mtpAdaptiveDepth` explicitly → use that.
    ///   * User set `mtpDepth` (a fixed numeric depth) but NOT
    ///     `mtpAdaptiveDepth` → `false` (pin to user's depth).
    ///   * Neither field set → `false` (pin to default depth 1). Set
    ///     `mtpAdaptiveDepth=true` explicitly to enable the adaptive
    ///     policy.
    pub mtp_adaptive_depth: bool,
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

/// Declarative thinking-mode policy for one family (P1 3.1 "easy thinking
/// budget"). [`resolve`] turns this + the request `ChatConfig` into the
/// concrete [`crate::engine::backend::ThinkingSetup`] the engine feeds
/// `ReasoningTracker::new`. Each variant is a 1:1 transcription of a
/// pre-P1 per-family `thinking_setup()` body.
pub(crate) enum ThinkingPolicy {
    /// No think-budget machinery: tracker permanently outside a think
    /// block (`enabled:false, budget:None`). == legacy gemma4.
    None,
    /// Honor the chat template's `enable_thinking` (default-on when
    /// unset); budget = the explicit `thinking_token_budget` only. ==
    /// legacy qwen3 / qwen3_5 / qwen3_5_moe. This is the DEFAULT policy.
    TemplateHonoring,
    /// Always inside a think block; explicit `thinking_token_budget`
    /// wins, else derive from `reasoning_effort` via
    /// [`default_thinking_budget_for_effort`]. == legacy lfm2 (whose
    /// template ignores `enable_thinking`). Footgun preserved verbatim:
    /// `reasoning_effort:"low"` caps the budget to 256 but does NOT
    /// disable thinking (`enabled` stays `true`).
    AlwaysOnBudgetFromEffort,
}

/// Resolve a [`ThinkingPolicy`] + request config into the concrete
/// per-turn [`crate::engine::backend::ThinkingSetup`]. The bodies are
/// byte-for-byte the pre-P1 per-family `thinking_setup()` impls.
pub(crate) fn resolve(
    policy: ThinkingPolicy,
    config: &ChatConfig,
) -> crate::engine::backend::ThinkingSetup {
    use crate::engine::backend::ThinkingSetup;
    match policy {
        ThinkingPolicy::None => ThinkingSetup {
            enabled: false,
            budget: None,
        },
        ThinkingPolicy::TemplateHonoring => ThinkingSetup {
            enabled: resolve_enable_thinking(config).unwrap_or(true),
            budget: config.thinking_token_budget,
        },
        ThinkingPolicy::AlwaysOnBudgetFromEffort => ThinkingSetup {
            enabled: true,
            budget: config
                .thinking_token_budget
                .or_else(|| default_thinking_budget_for_effort(config.reasoning_effort.as_deref())),
        },
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
        // Nonpositive budgets clamp to 0 (AR-equivalent empty completion)
        // so downstream max_kv_len / cache sizing / the decode macro never
        // see a negative budget. This single point feeds qwen3_5
        // dense/paged + qwen3_5_moe (incl. MTP); it is the core backstop
        // behind the server-side reject in `/v1/responses`.
        max_new_tokens: config.max_new_tokens.unwrap_or(2048).max(0),
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
        include_reasoning: resolve_include_reasoning(config),
        // MTP defaults OFF. When MTP is enabled and the caller does not
        // choose a depth, pin depth 1: current M5 Max measurements show
        // deeper bf16 MTP-head cycles lose more verify/draft time than
        // they recover from acceptance.
        enable_mtp: config.enable_mtp.unwrap_or(false),
        // Clamp the SIGNED depth before casting to usize: a negative
        // `mtpDepth` (reachable via the public `ChatConfig` surface) would
        // otherwise wrap (`-1 as usize` == usize::MAX) and clamp UP to 5,
        // forcing the slowest/deepest MTP path. Clamping first preserves the
        // documented `<1 → 1` behavior.
        mtp_depth: config
            .mtp_depth
            .map(|d| d.clamp(1, 5) as usize)
            .unwrap_or(1),
        // Adaptive depth policy is opt-in by default. An explicit
        // `mtpAdaptiveDepth` always wins. See
        // `ChatParams::mtp_adaptive_depth` docs.
        mtp_adaptive_depth: config.mtp_adaptive_depth.unwrap_or(false),
    }
}

/// Round a `(prefill_len + max_new_tokens)` token budget up to the next multiple of 256
/// for KV-cache capacity sizing. Computed in i64 so a hostile or absurd `max_new_tokens`
/// near `i32::MAX` cannot overflow the i32 sum (which would panic in debug / silently wrap
/// in release) before cache initialization. Inputs are floored at 0 (callers already clamp
/// budgets to >= 0 via `extract_chat_params`; this is defense in depth — for any
/// non-negative input the result is byte-identical to the legacy
/// `((prefill_len + max_new_tokens + 255) / 256) * 256`).
///
/// Returns `Err` when the rounded capacity would exceed `i32::MAX`, since the native
/// cache/FFI APIs are i32-typed; the caller surfaces this as a normal request error
/// instead of overflowing.
pub fn kv_capacity_round_up(prefill_len: i32, max_new_tokens: i32) -> Result<i32> {
    let total = (prefill_len.max(0) as i64) + (max_new_tokens.max(0) as i64) + 255;
    let rounded = (total / 256) * 256;
    if rounded > i32::MAX as i64 {
        return Err(Error::from_reason(format!(
            "requested KV-cache capacity {rounded} (prefill_len={prefill_len} + \
             max_new_tokens={max_new_tokens}, rounded up to a multiple of 256) exceeds the \
             maximum supported size of {}",
            i32::MAX
        )));
    }
    Ok(rounded as i32)
}

/// Saturating variant for DISPLAY/TRACE only — never errors. Clamps to the largest
/// multiple of 256 representable in i32. MUST NOT be used to size a real allocation.
pub fn kv_capacity_round_up_saturating(prefill_len: i32, max_new_tokens: i32) -> i32 {
    kv_capacity_round_up(prefill_len, max_new_tokens).unwrap_or((i32::MAX / 256) * 256)
}

/// Eager-allocation cap for generated-output `Vec::with_capacity` hints. A
/// `Vec::with_capacity` reserves memory immediately, so a hostile-but-accepted
/// token budget near `i32::MAX` would otherwise reserve gigabytes up front
/// (~8 GiB for a `Vec<u32>`). Real budgets up to this cap pre-allocate exactly;
/// larger budgets pre-allocate this much then grow via amortized doubling (a few
/// reallocs of a few KiB — negligible next to multi-second decode). Bounding the
/// HINT changes no observable behavior because the Vec still grows to hold every
/// generated token.
pub const GENERATED_CAPACITY_HINT_CAP: usize = 8192;

/// Bounded `Vec::with_capacity` hint for a generated-output buffer (tokens or
/// logprobs) sized from an untrusted `max_new_tokens` budget. Floors negatives at
/// 0 (so a negative budget can never produce a `usize::MAX` capacity that aborts)
/// and caps the eager reservation at [`GENERATED_CAPACITY_HINT_CAP`]; the buffer
/// still grows as needed during decode.
pub fn generated_capacity_hint(max_new_tokens: i32) -> usize {
    (max_new_tokens.max(0) as usize).min(GENERATED_CAPACITY_HINT_CAP)
}

#[cfg(test)]
mod mtp_params_tests {
    //! MTP defaults + override plumbing for `ChatParams`. No Metal
    //! required; purely tests the `ChatConfig → ChatParams` extraction.

    use super::extract_chat_params;
    use crate::engine::types::ChatConfig;

    fn base_config() -> ChatConfig {
        ChatConfig {
            max_new_tokens: None,
            temperature: None,
            top_k: None,
            top_p: None,
            min_p: None,
            repetition_penalty: None,
            repetition_context_size: None,
            presence_penalty: None,
            presence_context_size: None,
            frequency_penalty: None,
            frequency_context_size: None,
            max_consecutive_tokens: None,
            max_ngram_repeats: None,
            ngram_size: None,
            tools: None,
            reasoning_effort: None,
            thinking_token_budget: None,
            include_reasoning: None,
            report_performance: None,
            reuse_cache: None,
            enable_mtp: None,
            mtp_depth: None,
            mtp_adaptive_depth: None,
        }
    }

    /// Defaults: MTP off, depth 1.
    #[test]
    fn defaults_disable_mtp() {
        let cfg = base_config();
        let p = extract_chat_params(&cfg);
        assert!(!p.enable_mtp, "enable_mtp must default to false");
        assert_eq!(p.mtp_depth, 1, "mtp_depth must default to 1");
    }

    /// User override: `enable_mtp=true`, `mtp_depth=2` flows through.
    #[test]
    fn user_overrides_pass_through() {
        let mut cfg = base_config();
        cfg.enable_mtp = Some(true);
        cfg.mtp_depth = Some(2);
        let p = extract_chat_params(&cfg);
        assert!(p.enable_mtp);
        assert_eq!(p.mtp_depth, 2);
    }

    /// Depth clamping: <1 clamps to 1, >5 clamps to 5.
    #[test]
    fn depth_clamps_to_verify_ffi_range() {
        let mut cfg = base_config();
        cfg.mtp_depth = Some(0);
        let p = extract_chat_params(&cfg);
        assert_eq!(p.mtp_depth, 1, "mtp_depth=0 must clamp to 1");

        cfg.mtp_depth = Some(99);
        let p = extract_chat_params(&cfg);
        assert_eq!(
            p.mtp_depth, 5,
            "mtp_depth=99 must clamp to verify-FFI max 5"
        );

        // Negative depths are "<1" and must clamp to 1 — NOT wrap to
        // usize::MAX and clamp up to the slowest depth 5.
        cfg.mtp_depth = Some(-1);
        let p = extract_chat_params(&cfg);
        assert_eq!(p.mtp_depth, 1, "mtp_depth=-1 must clamp to 1, not 5");

        cfg.mtp_depth = Some(i32::MIN);
        let p = extract_chat_params(&cfg);
        assert_eq!(p.mtp_depth, 1, "mtp_depth=i32::MIN must clamp to 1");
    }

    /// `mtp_adaptive_depth` default resolution.
    ///
    ///   * Neither `mtpAdaptiveDepth` nor `mtpDepth` set → adaptive OFF.
    ///   * `mtpDepth` set, `mtpAdaptiveDepth` unset → adaptive OFF
    ///     (caller pinned a depth).
    ///   * `mtpAdaptiveDepth = Some(true)`, `mtpDepth` set → adaptive
    ///     ON (explicit field wins; depth becomes initial seed).
    ///   * `mtpAdaptiveDepth = Some(false)`, `mtpDepth` unset → OFF.
    #[test]
    fn adaptive_depth_default_resolution() {
        // Default: no fields set → adaptive OFF.
        let cfg = base_config();
        let p = extract_chat_params(&cfg);
        assert!(
            !p.mtp_adaptive_depth,
            "mtp_adaptive_depth must default to false when neither field is set"
        );

        // User pinned depth → adaptive OFF.
        let mut cfg = base_config();
        cfg.mtp_depth = Some(4);
        let p = extract_chat_params(&cfg);
        assert!(
            !p.mtp_adaptive_depth,
            "setting mtpDepth alone must pin (adaptive OFF)"
        );
        assert_eq!(p.mtp_depth, 4);

        // Explicit adaptive=true with pinned depth → adaptive ON.
        let mut cfg = base_config();
        cfg.mtp_depth = Some(2);
        cfg.mtp_adaptive_depth = Some(true);
        let p = extract_chat_params(&cfg);
        assert!(p.mtp_adaptive_depth);
        assert_eq!(p.mtp_depth, 2, "depth becomes the initial seed");

        // Explicit adaptive=false with no depth → OFF (uses default 1).
        let mut cfg = base_config();
        cfg.mtp_adaptive_depth = Some(false);
        let p = extract_chat_params(&cfg);
        assert!(!p.mtp_adaptive_depth);
        assert_eq!(p.mtp_depth, 1);
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

    /// Legacy formula the helper must reproduce byte-for-byte on the valid
    /// (non-overflowing, non-negative) range.
    fn legacy_round_up(prefill_len: i32, max_new_tokens: i32) -> i32 {
        ((prefill_len + max_new_tokens + 255) / 256) * 256
    }

    #[test]
    fn kv_capacity_round_up_matches_legacy_formula() {
        // Spread of normal (prefill_len, max_new_tokens) pairs that cannot
        // overflow i32. For every non-negative non-overflowing input the
        // helper is byte-identical to the legacy idiom.
        let cases = [
            (0, 0),
            (1, 1),
            (255, 1),
            (256, 1),
            (1000, 2048),
            (10, 2038),
            (4096, 0),
            (0, 4096),
            (8192, 8192),
            (123_456, 654_321),
        ];
        for (p, m) in cases {
            let expected = legacy_round_up(p, m);
            assert_eq!(
                kv_capacity_round_up(p, m).unwrap(),
                expected,
                "kv_capacity_round_up({p}, {m}) must match legacy formula"
            );
        }
        // Spot-check a couple of the trickier ones by hand.
        assert_eq!(kv_capacity_round_up(0, 0).unwrap(), 0);
        assert_eq!(kv_capacity_round_up(1, 1).unwrap(), 256);
        assert_eq!(kv_capacity_round_up(255, 1).unwrap(), 256);
        assert_eq!(kv_capacity_round_up(256, 1).unwrap(), 512);
        // 10 + 2038 + 255 = 2303; 2303 / 256 = 8; 8 * 256 = 2048.
        assert_eq!(kv_capacity_round_up(10, 2038).unwrap(), 2048);
    }

    #[test]
    fn kv_capacity_round_up_boundary_exact() {
        // Largest representable multiple of 256 in i32.
        assert_eq!((i32::MAX / 256) * 256, 2_147_483_392);

        // Already a multiple of 256 (after +255 it lands exactly on i32::MAX,
        // which floors back to 2_147_483_392) — no overflow.
        assert_eq!(
            kv_capacity_round_up(0, 2_147_483_392).unwrap(),
            2_147_483_392
        );

        // One more token rounds up to 2_147_483_648 (> i32::MAX) -> Err.
        assert!(kv_capacity_round_up(0, 2_147_483_393).is_err());

        // The exact review finding: any non-empty prompt + an i32::MAX budget
        // would overflow the i32 sum in the legacy formula. Now a clean Err.
        assert!(kv_capacity_round_up(1, i32::MAX).is_err());

        // i32::MAX budget alone already rounds up past i32::MAX -> Err.
        assert!(kv_capacity_round_up(0, i32::MAX).is_err());
    }

    #[test]
    fn kv_capacity_round_up_saturating_never_panics() {
        // Saturating variant clamps to the largest in-range multiple of 256
        // instead of erroring — for display/trace only.
        assert_eq!(kv_capacity_round_up_saturating(1, i32::MAX), 2_147_483_392);
        // Valid inputs pass through unchanged.
        assert_eq!(kv_capacity_round_up_saturating(1, 1), 256);
    }

    #[test]
    fn kv_capacity_round_up_floors_negative_inputs() {
        // Defense in depth: negative inputs are floored at 0 (callers already
        // clamp, but the helper must never produce a negative or wrapped size).
        assert_eq!(kv_capacity_round_up(-5, -5).unwrap(), 0);
        assert_eq!(kv_capacity_round_up(-1, 1).unwrap(), 256);
        // (256, -1000) floors to (256, 0): 256 + 0 + 255 = 511; 511/256 = 1; *256 = 256.
        assert_eq!(kv_capacity_round_up(256, -1000).unwrap(), 256);
        // Negative input must produce the SAME result as the floored positive input.
        assert_eq!(
            kv_capacity_round_up(256, -1000).unwrap(),
            kv_capacity_round_up(256, 0).unwrap()
        );
    }

    #[test]
    fn generated_capacity_hint_caps_and_floors() {
        // [high] scenario: a hostile-but-accepted budget near i32::MAX must NOT
        // trigger a multi-GiB eager reservation — the hint is capped.
        assert_eq!(generated_capacity_hint(i32::MAX), 8192);
        assert_eq!(
            generated_capacity_hint(i32::MAX),
            GENERATED_CAPACITY_HINT_CAP
        );
        // Negative budgets floor at 0 (never wrap to usize::MAX → abort).
        assert_eq!(generated_capacity_hint(-5), 0);
        assert_eq!(generated_capacity_hint(i32::MIN), 0);
        // Below-cap budgets pass through exactly (behavior-neutral pre-alloc).
        assert_eq!(generated_capacity_hint(0), 0);
        assert_eq!(generated_capacity_hint(100), 100);
        assert_eq!(generated_capacity_hint(2048), 2048);
        // Exact cap boundary.
        assert_eq!(generated_capacity_hint(8192), 8192);
        assert_eq!(generated_capacity_hint(8193), 8192);
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
}
