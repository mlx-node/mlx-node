//! Shared chat/decode infrastructure for Qwen3.5 Dense and MoE models.
//!
//! Extracts identical boilerplate from the session entry points
//! (`chat_session_start_sync` / `chat_session_continue_sync` /
//! `chat_session_continue_tool_sync` and their `chat_stream_*` streaming
//! counterparts) across both model variants: config extraction, penalty
//! application, performance metrics, result finalization, and cache
//! management.

use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::OnceLock;

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

// ---------------------------------------------------------------------------
// MTP runtime flag inventory
// ---------------------------------------------------------------------------
//
// The W6.5–W6.19 perf chain added six runtime knobs gating individual
// optimizations. All are read at most once per process and cached. The
// truthy vocabulary is uniform across the five env-var flags: trim()
// + `1` / `true` / `on` (case-insensitive). The adaptive depth knob is
// surfaced through the TypeScript `ChatConfig.mtpAdaptiveDepth` field
// rather than an env var because it interacts with the user-set
// `mtpDepth` and needs per-session resolution.
//
// | Knob                          | Default | Workstream | Opt direction |
// |-------------------------------|---------|------------|---------------|
// | `MLX_MTP_USE_TAPE_REPLAY`     | ON      | W6.6       | opt-OUT       |
// | (eager verify prewarm)        | ON      | W6.7       | n/a (always)  |
// | `mtpAdaptiveDepth` (TS field) | ON*     | W6.8       | per-session   |
// | `MLX_MTP_CHAINED_CYCLES`      | OFF     | W6.5       | opt-IN        |
// | `MLX_MTP_VERIFY_ASYNC_EVAL`   | OFF     | W6.9       | opt-IN        |
// | `MLX_MTP_FUSED_DRAFT`         | OFF     | W6.18      | opt-IN        |
// | `MLX_MTP_SPARSE_ACCEPT`       | OFF     | W6.19      | opt-IN        |
//
// * adaptive defaults ON when `enableMtp=true` and `mtpDepth` is unset;
//   defaults OFF (pinned) when `mtpDepth` is set explicitly.
//
// Naming disambiguation:
//   - `MLX_MTP_CHAINED_CYCLES` (W6.5) — CROSS-CYCLE hidden-state export.
//     Each cycle's `verify_hidden[K]` slice seeds the next cycle's first
//     MTP draft; see `eval_step_with_chained_hidden` below.
//   - `MLX_MTP_FUSED_DRAFT` (W6.18) — WITHIN-CYCLE draft-step fusion.
//     All D draft steps within a single cycle compile into one
//     `mlx::core::compile`d graph; see `MtpOps::fused_draft` and
//     `mlx_qwen35_mtp_draft_fused_compiled` in the C++ FFI.
//
// Interaction notes:
//   - `MLX_MTP_USE_TAPE_REPLAY=0` falls back to the W6 Bug #4 K+1 replay
//     path; safe to combine with all other flags.
//   - `MLX_MTP_CHAINED_CYCLES=1` is currently slower than the default
//     Step-A path at depth ≥ 2 even after the W6.5-resume fix (the
//     `verify_hidden[K]` slice is now batched into the next-cycle
//     `async_eval` — see `eval_step_with_chained_hidden` below). The
//     residual ~18% gap on bf16/M3 Max smoke traces to cross-cycle CPU
//     bookkeeping, not the slice DMA the original W6.5 plan blamed.
//     Further work tracked in the literal-prefetch follow-up.
//   - `MLX_MTP_VERIFY_ASYNC_EVAL=1` overlaps verify dispatch with the
//     accept loop's CPU-side graph construction; composes cleanly with
//     all other flags.
//   - `MLX_MTP_FUSED_DRAFT=1` routes all D draft steps through one
//     compiled graph. Default OFF because the current smoke target
//     (qwen3.6-27b-nvfp4-mtp, depth=3, M3 Max) shows no measured win
//     — the per-step `tok.eval()` barriers do not serialize into 3 ×
//     30–50 ms stalls on this hardware. Infrastructure is kept opt-in
//     pending the Step-A-bypass follow-up where draft syncs are
//     expected to dominate the cycle wall.

// W6.6 — Tape-replay rollback for GDN linear-attention.
//
// Replaces the K+1 main-model forwards the W6 Bug #4 fix ran on every
// partial-accept verify cycle. When ON (default), the cycle arms tape
// recording on the dense compiled path BEFORE verify; the per-step
// `(tape, k, g, qkv)` tensors are accumulated during the D+1 verify
// forwards; on rejection a single Metal kernel replays only the
// accepted prefix into the pre-verify snapshot state and the conv state
// is rebuilt by slicing the recorded qkv. Saves ~30% of cycle wall-time
// (the W6 plan's rollback budget).
//
// Opt-out: `MLX_MTP_USE_TAPE_REPLAY=0` (or `false`) falls back to the
// K+1 replay path for a release while we shake out kernel bugs. The
// env var is read once per process and cached.
pub(crate) fn mtp_use_tape_replay() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| match std::env::var("MLX_MTP_USE_TAPE_REPLAY") {
        Ok(v) => {
            let v = v.trim();
            !(v == "0" || v.eq_ignore_ascii_case("false") || v.eq_ignore_ascii_case("off"))
        }
        Err(_) => true, // default ON
    })
}

// W6.9 — Async-prefetch pipeline.
//
// Replaces the synchronous `verify_logits.eval()` at the end of
// `run_mtp_cycle_inner`'s verify step with a single batched
// `mlx::core::async_eval` over `(verify_logits, verify_hiddens)`. The
// dispatch is non-blocking, so CPU control flow continues into the
// accept loop's penalty / softmax / slice graph construction while the
// GPU is still running the verify command buffer. The first downstream
// `eval()` (the accept loop's `p_target.eval()`) then implicitly
// synchronizes — the same overlap pattern DFlash uses with
// `mx.async_eval(posterior)` at `spec_epoch.py:2069`.
//
// This is the scheduler-level realisation of the W6.9 plan's "stash
// next-cycle draft handle, drain at cycle start" idea. The literal
// stash-and-drain refactor would require plumbing a pre-built draft
// graph handle through `run_mtp_cycle_inner` AND mutating
// `g_mtp_compiled_caches` BEFORE the accept loop knows K — both
// architecturally invasive AND only safe in the opt-in
// `MLX_MTP_CHAINED_CYCLES` path (default off). The async_eval pipeline
// captures the same overlap-with-CPU-graph-construction win without
// touching the FFI surface, the `MtpOps` closures, or the cycle
// state machine.
//
// Opt-in: `MLX_MTP_VERIFY_ASYNC_EVAL=1` (or `true`). Default OFF until
// the parity gates have shipped a couple of releases. The env var is
// read once per process and cached.
//
// Naming note: an earlier draft of this flag was called
// `MLX_MTP_PREFETCH`, but the change is overlap-with-CPU-graph-
// construction (intra-cycle) — NOT cross-cycle draft staging. The
// literal "stash next-cycle draft handle, drain at cycle start"
// prefetch from the W6.9 plan lives in a follow-up task scoped to
// `MLX_MTP_CHAINED_CYCLES=1`. The current flag's name reflects the
// actual mechanism so users don't expect cross-cycle behaviour.
pub(crate) fn mtp_verify_async_eval() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| match std::env::var("MLX_MTP_VERIFY_ASYNC_EVAL") {
        Ok(v) => {
            let v = v.trim();
            v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("on")
        }
        Err(_) => false, // default OFF — flip in a follow-up after parity proven
    })
}

// W6.5 — Chained cycles via verify-hidden export.
//
// Opt-in: `MLX_MTP_CHAINED_CYCLES=1` (or `true` / `on`). The W6.5-resume
// follow-up fused the chained `verify_hidden[K]` slice into the same
// `async_eval` batch as `(token, g_compiled_caches)` at end-of-iteration
// (see `eval_step_with_chained_hidden` in `MtpOps`) so the slice becomes
// a sibling of the next-cycle draft's first inputs rather than a late
// dependency materialized inside the draft graph build. Whether the
// default flips back to ON depends on the smoke benchmark — see the
// commit log for the per-run measurement and the `mtp_speculative_decoding`
// memory page for the running absolute-ratio history.
//
// The env var is read once per process and cached.
pub(crate) fn mtp_chained_cycles_enabled() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| match std::env::var("MLX_MTP_CHAINED_CYCLES") {
        Ok(v) => {
            let v = v.trim();
            v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("on")
        }
        Err(_) => false, // default OFF — will flip post-resume benchmark
    })
}

// W6.18 — Fused within-cycle draft graph.
//
// Opt-in: `MLX_MTP_FUSED_DRAFT=1` (or `true` / `on`). When ON AND the
// caller installed an `MtpOps::fused_draft` closure AND temperature is
// effectively zero, `run_mtp_cycle_inner` routes all D draft steps
// through ONE `mlx::core::compile`d graph (the
// `mlx_qwen35_mtp_draft_fused_compiled` FFI) instead of looping the
// per-step `draft_step` closure D times.
//
// Default OFF: the current smoke target (qwen3.6-27b-nvfp4-mtp,
// depth=3, M3 Max) measured 0.61× → 0.60× AR ratio — no observable
// win. Per-cycle wall-time instrumentation shows the fused FFI runs
// all D=3 draft steps in ~33 ms total, vs the per-step path's
// ~30–42 ms total — equivalent. The per-step `tok.eval()` barriers do
// not serialize into 3 × 30–50 ms stalls (which the task brief
// assumed, extrapolating from a 4B-bf16 measurement); MLX already
// overlaps CPU work between sync points. The remaining MTP overhead
// vs AR lives in Step A + batched verify + accept loop, not in
// drafting. The infrastructure is preserved opt-in for a future
// Step-A-bypass follow-up where draft syncs are expected to pay off.
//
// Naming: independent of the W6.5 `MLX_MTP_CHAINED_CYCLES` knob; see
// the "Naming disambiguation" block in the inventory above.
//
// The env var is read once per process and cached.
pub(crate) fn mtp_fused_draft_enabled() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| match std::env::var("MLX_MTP_FUSED_DRAFT") {
        Ok(v) => {
            let v = v.trim();
            v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("on")
        }
        Err(_) => false, // default OFF — opt-in until a target shows perf win
    })
}

// W6.19 — Accept-loop sync collapse via on-device sparse top-K /
// batched argmax (MTPLX-style).
//
// Replaces the legacy per-position accept loop's D forced GPU syncs
// (each materializing a full-vocab softmax of ~151k floats) with ONE
// batched on-device op over all `D+1` verify positions. On the T=0
// (greedy) path this is `argmax(verify_logits, axis=-1)` → `[1, D+1]`
// int32, evaluated once. On T>0 we keep the legacy per-position
// path (residual sampling still needs the full target distribution
// to draw from `(p_target - p_draft)+`).
//
// Eligibility (T=0 fast path):
//   - `temperature <= 1e-6` (matches `accept_with_residual`'s argmax
//     shortcut — Bug #3 invariant).
//   - All penalties at defaults (repetition=1.0, presence=0.0,
//     frequency=0.0). When any penalty is active, the per-position
//     `apply_all_penalties` call depends on `hist_extended` which
//     mutates inside the accept loop — we cannot precompute the
//     argmax in one shot without re-applying the penalty per
//     position.
//
// Opt-IN: `MLX_MTP_SPARSE_ACCEPT=1` (or `true` / `on`). Default OFF
// because the current smoke target (qwen3.6-27b-nvfp4-mtp, depth=3,
// M3 Max) shows no measured perf win — the W6.19 bench was 0.547×
// vs 0.56× baseline AR ratio, well within run-to-run noise. The
// optimization composes cleanly with all other MTP knobs and is
// kept opt-in pending hardware/model targets where the MLX
// scheduler exposes the D × full-vocab softmax materialization
// cost. Matches the W6.18 polish precedent: perf-neutral knobs are
// opt-IN so they don't bit-rot the default code path. The env var
// is read once per process and cached.
pub(crate) fn mtp_sparse_accept_enabled() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| match std::env::var("MLX_MTP_SPARSE_ACCEPT") {
        Ok(v) => {
            let v = v.trim();
            v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("on")
        }
        Err(_) => false, // default OFF — opt-in until a target shows perf win
    })
}

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
    /// W6 (MTP): opt-in flag enabling the Multi-Token Prediction
    /// speculative decode loop. Effective only on the dense compiled
    /// path AND when the model checkpoint carries an MTP head
    /// (`Qwen35Inner::has_mtp_weights`). The eager Rust forward, the
    /// paged path, MoE, and VLM decode loops all continue to use the
    /// single-token `decode_loop!` macro regardless. Default: `false`.
    pub enable_mtp: bool,
    /// W6 (MTP): number of draft tokens per speculative cycle, fed to
    /// the W5 `forward_mtp_draft_compiled` / `forward_mtp_verify_compiled`
    /// FFI. Must be in `[1, 5]` to satisfy the verify-FFI contract.
    /// Default: 3 (MTPLX paper's reported sweet spot). When
    /// `mtp_adaptive_depth = true`, this value is only used as the
    /// initial depth — the W6.8 `AdaptiveDepthPolicy` picks per-cycle.
    pub mtp_depth: usize,
    /// W6.8 (MTP): when true, the decode loop runs an
    /// `AdaptiveDepthPolicy` (`adaptive_depth.rs`) that picks the draft
    /// depth per cycle by maximising per-depth EMA of
    /// `accepted_tokens / cycle_wall_ns`. When false, the loop pins
    /// `mtp_depth` for every cycle (legacy pre-W6.8 behaviour).
    ///
    /// Default resolution (`extract_chat_params`):
    ///   * User set `mtpAdaptiveDepth` explicitly → use that.
    ///   * User set `mtpDepth` (a fixed numeric depth) but NOT
    ///     `mtpAdaptiveDepth` → `false` (pin to user's depth).
    ///   * Neither field set → `true` (adaptive ON by default; the
    ///     compiled verify graphs for all depths in `{1..=5}` are
    ///     pre-warmed at model load so switching is zero-cost).
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
        // W6 (MTP) — defaults keep MTP OFF until the per-request opt-in
        // lands in the TS surface (W7). `mtp_depth = 3` matches MTPLX's
        // reported sweet spot; clamped to `[1, 5]` by the W5 verify FFI.
        enable_mtp: config.enable_mtp.unwrap_or(false),
        mtp_depth: config
            .mtp_depth
            .map(|d| (d as usize).clamp(1, 5))
            .unwrap_or(3),
        // W6.8 — adaptive depth policy ON by default when neither
        // `mtpAdaptiveDepth` nor `mtpDepth` was set; OFF when the user
        // pinned a specific `mtpDepth`. An explicit `mtpAdaptiveDepth`
        // always wins. See `ChatParams::mtp_adaptive_depth` docs.
        mtp_adaptive_depth: config
            .mtp_adaptive_depth
            .unwrap_or(config.mtp_depth.is_none()),
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

// =============================================================================
// W6 — MTP speculative decode loop (dense compiled path only).
//
// Sister to `decode_loop!` above; preserves every behavior of the
// single-token loop (penalties, ReasoningTracker / budget, EOS,
// repetition cutoff, every-256-step cache clear, streaming +
// cancellation) while emitting up to `mtp_depth + 1` tokens per
// outer iteration via the W5 draft + verify FFI plus the W4
// `accept_with_residual` sampler.
//
// Cache-rollback strategy (compiled path):
//   - The verify FFI advances the MAIN compiled K/V offset by
//     `depth + 1`. On any rejection we rewind the main offset by
//     `accepted_count - (depth + 1)` (a negative delta) via
//     `mlx_qwen35_compiled_adjust_offset`. We do NOT zero the K/V
//     buffer entries at positions `[accepted .. depth+1]` — the
//     next forward simply overwrites them.
//   - The MTP draft FFI advances its OWN offset by 1 per draft
//     step. On any rejection we rewind by `accepted_count - depth`
//     via `mlx_qwen35_mtp_compiled_adjust_offset`. The MTP path is
//     by design 1-token behind the main path's accepted prefix.
//   - W3's `Qwen3_5LayerCache::snapshot_all` / `restore_all` is the
//     EAGER-PATH rollback primitive. On the compiled path the live
//     K/V lives in `g_compiled_caches` (C++), not `self.caches`, so
//     the snapshot/restore is intentionally NOT used here.
//
// Tracker / budget invariant:
//   - `should_force_think_end()` is checked BEFORE starting each
//     draft cycle. If forced, the macro emits ONE forced token via
//     the normal main-path forward + sampler and skips the cycle.
//   - It is also checked BEFORE accepting each individual verified
//     token. If forced mid-cycle, the macro aborts the remaining
//     accepted tokens, rewinds the offsets to (already-emitted + 1)
//     committed positions on top of the forced token, and emits the
//     forced token through the normal path.
// =============================================================================

/// Closure bundle for the MTP cycle. Mirrors `DecodeOps` but adds
/// draft / verify / rollback hooks that are only meaningful on the
/// dense compiled path.
///
/// `F`  : single main-path forward step returning `(logits, hidden,
///        needs_squeeze)`. `hidden` is `[1, hidden_size]` bf16.
/// `D`  : MTP draft step returning `(h_next, draft_logits)` where
///        `h_next` is `[1, 1, hidden]` and `draft_logits` is
///        `[1, vocab]`.
/// `V`  : MTP verify step returning verify logits of shape
///        `[1, depth + 1, vocab]`.
/// `R`  : rollback hook receiving `(accepted, depth)`. On rejection
///        the implementor calls
///        `mlx_qwen35_mtp_compiled_adjust_offset(accepted as i32 -
///        depth as i32)` (MTP path only). The MAIN path's offset is
///        rewound by `restore_and_replay_main` via the snapshot taken
///        in `snapshot_main_linear` — DO NOT also call
///        `mlx_qwen35_compiled_adjust_offset` here, or the main
///        offset will double-rewind.
/// `E`  : eval-step hook (same contract as `DecodeOps::eval_step`)
///        called after every emitted token to flush the lazy graph.
/// `B`  : begin-cycle hook called once per outer iteration, BEFORE
///        the draft steps, AFTER Step A's main-path forward. The
///        implementor reads the main path's current offset (via
///        `mlx_qwen35_get_cache_offset` / `mlx_qwen35_moe_get_cache_offset`)
///        and calls the corresponding `*_begin_cycle(main_offset)` FFI
///        to zero the MTP K/V caches and re-anchor the MTP offset.
///        This fixes W6 Bug #2 (mid-stream divergence): without the
///        reset the MTP offset lags the main offset by 2 per cycle.
/// `S`  : snapshot hook for the main path's GDN linear-attention caches
///        plus the main decode offset. Called once per cycle AFTER
///        Step A and BEFORE verify. Implementor calls
///        `mlx_qwen35_compiled_snapshot_linear_caches` (or the MoE
///        equivalent). The snapshot is consumed by
///        `restore_and_replay_main` on rejection — without it the
///        GDN recurrent state stays polluted with rejected draft
///        positions and the next Step A produces wrong logits (W6 Bug
///        #4).
/// `RR` : restore + replay hook called on rejection (any
///        `accepted_drafts < depth`) AFTER `rollback`. Receives the
///        list of accepted draft token IDs (NOT including the residual
///        sample, NOT including `last_committed`) and the embedding
///        weight. Implementor:
///          1. Calls `mlx_qwen35_compiled_restore_linear_caches`
///             (rewinds linear caches AND main offset to the
///             snapshot point);
///          2. For each accepted draft, runs ONE
///             `mlx_qwen35_forward_compiled` (via
///             `forward_with_hidden` to keep the implementation
///             path-agnostic) so the main linear state catches up to
///             "after Step A + K accepted drafts" and the main offset
///             reaches `snapshot_offset + K`.
///        On full-accept the macro skips this hook (verify already
///        left the linear state advanced through all D drafts).
pub(crate) struct MtpOps<F, D, V, R, E, EX, B, S, RR>
where
    F: FnMut(&MxArray, &MxArray) -> Result<(MxArray, MxArray, bool)>,
    D: FnMut(&MxArray, &MxArray) -> Result<(MxArray, MxArray)>,
    // W6.5 — verify returns (logits, verify_hiddens). Logits shape is
    // `[1, depth+1, vocab]`; hiddens shape is `[1, depth+1, hidden]`.
    // The hiddens carry the post-final-norm output at EVERY verify
    // position. After `run_mtp_cycle_inner` computes the number of
    // accepted drafts K, it slices `verify_hiddens[:, K, :]` to seed
    // the next cycle's first MTP draft — that hidden is the prediction
    // context for the committed token at position K+1 (bonus on
    // full-accept, residual on rejection), matching the MTP head's
    // training contract.
    V: FnMut(&MxArray, &MxArray, usize) -> Result<(MxArray, MxArray)>,
    R: FnMut(usize, usize),
    E: Fn(&MxArray, &MxArray, bool),
    // W6.5-resume — same shape as `eval_step` but folds the chained
    // `verify_hidden[K]` slice into the SAME `async_eval` batch as
    // `(token, g_compiled_caches)`. Used by the chained-cycles macro
    // path at end-of-iteration so the slice is co-materialized with
    // the next cycle's first-draft input sources, eliminating the
    // mid-cycle Metal command-buffer roundtrip the lazy path forces.
    //
    // Contract: `token` is the just-set `$y` (CPU-only integer array
    // of the last accepted token); `chained_hidden` is the lazy
    // `verify_hiddens[:, K, :]` slice the prior cycle produced. The
    // closure MUST schedule `async_eval` (or equivalent) over the
    // dense / MoE compiled caches PLUS both arrays so the slice
    // becomes a sibling of the next-cycle draft graph rather than a
    // late dependency.
    EX: Fn(&MxArray, &MxArray),
    B: FnMut(),
    S: FnMut(),
    RR: FnMut(&[u32], &MxArray) -> Result<()>,
{
    pub forward_with_hidden: F,
    pub draft_step: D,
    pub verify_step: V,
    pub rollback: R,
    pub eval_step: E,
    pub eval_step_with_chained_hidden: EX,
    pub begin_cycle: B,
    pub snapshot_main_linear: S,
    pub restore_and_replay_main: RR,
    /// W6.18 — Optional fused-draft callback. When `Some(...)` AND
    /// `MLX_MTP_FUSED_DRAFT=1` AND `run_mtp_cycle_inner` decides to use
    /// the fused path (currently gated on `temperature <= 1e-6`), all D
    /// draft steps run inside ONE compiled graph via this closure.
    ///
    /// "Fused" refers to the WITHIN-CYCLE fusion of D draft steps —
    /// independent of the W6.5 CROSS-CYCLE `MLX_MTP_CHAINED_CYCLES`
    /// concept (verify-hidden export across cycles).
    ///
    /// Contract:
    ///
    /// * Inputs: `prev_hidden` / `prev_emb` (`[1, 1, hidden]` bf16),
    ///   embedding_weight (`[vocab, hidden]`), depth ∈ {1..5}.
    /// * Outputs: `(draft_ids, draft_probs)` where `draft_ids` is `[D]`
    ///   int32 (drafted token IDs in step order) and `draft_probs` is
    ///   `[D, vocab]` fp32 (per-step softmax probs). At T<=1e-6 the
    ///   per-step probs are unused numerically — `accept_with_residual`
    ///   only reads the draft IDs in the greedy path.
    ///
    /// `None` disables fusion and forces the per-step `draft_step`
    /// loop (legacy path; required for tests and MoE).
    pub fused_draft:
        Option<Box<dyn FnMut(&MxArray, &MxArray, &MxArray, usize) -> Result<(MxArray, MxArray)>>>,
}

/// Outcome of `run_mtp_cycle_inner` — the list of accepted tokens for
/// this cycle plus whether a rejection forced a rollback (used by the
/// macro to log / observe).
pub(crate) struct MtpCycleOutcome {
    /// Accepted token IDs in emission order. Always at least one
    /// element on success (residual sample on full reject, or
    /// bonus token on full accept).
    pub tokens: Vec<u32>,
}

/// One MTP draft+verify cycle. Pure helper — the caller drives the
/// per-token streaming, EOS, cancellation, and tracker bookkeeping
/// inside `decode_loop_mtp!`.
///
/// Contract:
/// * `prev_hidden_in` / `prev_emb_in` are `[1, 1, hidden]` bf16.
/// * `last_committed_id` is the token at the end of `token_history`
///   — used as the first column of the verify input.
/// * `depth` MUST be `>= 1` and `<= 5` (verify FFI clamps).
/// * `embedding_weight` is the model's embedding table (already
///   resolved to the LM head when `tie_word_embeddings=false` on the
///   caller side — same arg as `forward_with_hidden`).
///
/// On error any partial offset / K/V drift from the partial cycle
/// is the caller's problem; production callers fold the cycle inside
/// `DENSE_COMPILED_MUTEX` so a `?` early-return drops the
/// `CompiledResetGuard` and wipes the C++ state cleanly.
pub(crate) fn run_mtp_cycle_inner<F, D, V, R, E, EX, B, S, RR>(
    ops: &mut MtpOps<F, D, V, R, E, EX, B, S, RR>,
    prev_hidden_in: MxArray,
    prev_emb_in: MxArray,
    last_committed_id: u32,
    embedding_weight: &MxArray,
    token_history: &[u32],
    params: &ChatParams,
    rng: &mut impl rand::Rng,
    profiler: &mut crate::decode_profiler::DecodeProfiler,
    depth: usize,
) -> Result<(MtpCycleOutcome, MxArray)>
where
    F: FnMut(&MxArray, &MxArray) -> Result<(MxArray, MxArray, bool)>,
    D: FnMut(&MxArray, &MxArray) -> Result<(MxArray, MxArray)>,
    V: FnMut(&MxArray, &MxArray, usize) -> Result<(MxArray, MxArray)>,
    R: FnMut(usize, usize),
    E: Fn(&MxArray, &MxArray, bool),
    EX: Fn(&MxArray, &MxArray),
    B: FnMut(),
    S: FnMut(),
    RR: FnMut(&[u32], &MxArray) -> Result<()>,
{
    use crate::array::{DType, MxArray as A};
    use crate::nn::Activations;
    use crate::sampling;

    debug_assert!(depth >= 1, "run_mtp_cycle_inner: depth must be >= 1");

    // Step 1: D draft steps. Either fused into one compiled graph
    // (W6.18 fused path) or the legacy per-step loop.
    //
    // Fused path gating:
    //   1. `MLX_MTP_FUSED_DRAFT=1` — opt-in env var (default OFF; see
    //      `mtp_fused_draft_enabled` for rationale).
    //   2. `ops.fused_draft` is `Some(...)` — the caller installed
    //      the FFI dispatch (production dense path does; MoE / tests
    //      pass None).
    //   3. `temperature <= 1e-6` — at T=0 (greedy) the per-step Rust
    //      path collapses to argmax via `sampling::sample` (see
    //      `compiled_sample_full`); the fused graph also argmaxes
    //      on-device, so the drafted IDs are byte-equivalent.
    //      At T>0 the per-step path stochastically samples (consuming
    //      MLX's global RNG between steps); the fused graph uses
    //      argmax, which is a SEMANTIC change for the drafter only —
    //      verify is still the ground truth, so emitted tokens still
    //      come from the target distribution per Leviathan-Chen, but
    //      acceptance rate may differ. We keep T>0 on the per-step
    //      path to preserve the existing contract.
    //
    // On any fused-path error we fall back to the per-step loop in
    // the same cycle (the eager Rust path is the safety net).
    profiler.begin("mtp_draft_total");
    let temperature = params
        .sampling_config
        .and_then(|c| c.temperature)
        .unwrap_or(1.0);
    let try_fused = mtp_fused_draft_enabled() && ops.fused_draft.is_some() && temperature <= 1e-6;

    let mut prev_hidden = prev_hidden_in;
    let mut prev_emb = prev_emb_in;
    let mut draft_ids: Vec<i32> = Vec::with_capacity(depth);
    let mut draft_probs: Vec<MxArray> = Vec::with_capacity(depth);

    let mut used_fused = false;
    if try_fused && let Some(ref mut fused) = ops.fused_draft {
        match fused(&prev_hidden, &prev_emb, embedding_weight, depth) {
            Ok((ids_arr, probs_arr)) => {
                // ids_arr is `[depth]` int32; eval and read once.
                ids_arr.eval();
                for i in 0..depth {
                    draft_ids.push(ids_arr.item_at_int32(i)?);
                }
                // probs_arr is `[depth, vocab]` fp32. Slice into per-step
                // `[vocab]` rows; we do NOT eval the slices here — at
                // T=0 `accept_with_residual` ignores p_draft numerically
                // (only reads draft_id), so the lazy slice never gets
                // materialised by Metal. This avoids a per-position
                // eval barrier that would defeat the very sync collapse
                // this task is about. At T>0 the fused path is gated
                // off (try_fused=false), so this code is unreachable.
                let vocab = probs_arr.shape_at(1)?;
                for i in 0..depth {
                    let row = probs_arr.slice(&[i as i64, 0], &[(i + 1) as i64, vocab])?;
                    // Squeeze to `[vocab]` for the per-position
                    // `accept_with_residual` contract.
                    draft_probs.push(row.squeeze(Some(&[0]))?);
                }
                used_fused = true;
            }
            Err(e) => {
                // W6.18 — fused path failed (init not done, C++
                // exception, shape mismatch). Log + fall back to
                // the per-step loop so the cycle still completes.
                tracing::warn!(
                    target: "mlx_core::mtp::fused",
                    error = %e.reason,
                    "W6.18 fused draft FFI failed; falling back to per-step path"
                );
                draft_ids.clear();
                draft_probs.clear();
            }
        }
    }

    if !used_fused {
        for _ in 0..depth {
            let (h_next, draft_logits) = (ops.draft_step)(&prev_hidden, &prev_emb)?;
            // draft_logits is [1, vocab]; squeeze to [vocab] for softmax.
            let logits_1d = draft_logits.squeeze(Some(&[0]))?;
            let probs = Activations::softmax(&logits_1d, Some(-1))?.astype(DType::Float32)?;
            // Sample the drafted token using the same sampling pipeline
            // the main path uses — drafter and verifier must agree on
            // their proposal distribution for Leviathan-Chen.
            let tok = sampling::sample(&draft_logits, params.sampling_config)?;
            tok.eval();
            let tok_id = tok.item_at_int32(0)?;
            draft_ids.push(tok_id);
            draft_probs.push(probs);
            // Update prev_hidden for next draft step.
            prev_hidden = h_next;
            // prev_emb for the next draft is the embedding of the token
            // we just drafted.
            let id_arr = A::from_int32(&[tok_id], &[1])?;
            let emb_2d = embedding_weight.take(&id_arr, 0)?; // [1, hidden]
            let hidden = emb_2d.shape_at(1)?;
            prev_emb = emb_2d.reshape(&[1, 1, hidden])?;
        }
    }
    profiler.end();

    // Step 2: build verify input [last_committed_id, d_0, ..., d_{D-1}].
    let mut verify_ids: Vec<i32> = Vec::with_capacity(depth + 1);
    verify_ids.push(last_committed_id as i32);
    verify_ids.extend(draft_ids.iter().copied());
    let verify_in = A::from_int32(&verify_ids, &[1, (depth + 1) as i64])?;
    // W6 Bug #4 — snapshot the main path's GDN linear caches + offset
    // BEFORE verify runs its D+1 sequential forwards. Verify mutates
    // `g_compiled_caches` in place; on rejection we restore from this
    // snapshot and replay only the K accepted drafts so the linear
    // recurrent state matches the committed token stream. On full
    // accept the snapshot is discarded — verify already left the
    // linear state correctly advanced.
    profiler.begin("mtp_tape_snapshot");
    (ops.snapshot_main_linear)();
    profiler.end();
    // W6.5 — verify returns BOTH logits and per-position hiddens.
    // Logits: `[1, depth+1, vocab]`; hiddens: `[1, depth+1, hidden]`.
    // We hold off on slicing the hidden until after the accept loop
    // computes K (= number of accepted drafts) so we can pick
    // `verify_hiddens[:, K, :]` — the correct prediction context for
    // the next cycle's first MTP draft.
    // Phase 1 instrumentation: the gap between `mtp_cycle` and this
    // floor is the headroom available to algorithmic work.
    let verify_only_t0 = std::time::Instant::now();
    profiler.begin("mtp_verify_dispatch");
    let verify_step_res = (ops.verify_step)(&verify_in, embedding_weight, depth);
    profiler.end();
    let (verify_logits, verify_hiddens) = verify_step_res?;
    // W6.9 — Async-eval over verify outputs. When
    // `MLX_MTP_VERIFY_ASYNC_EVAL=1`, we dispatch verify (logits +
    // hiddens) via `async_eval` instead of the synchronous `eval()`
    // below. The kernel launch returns immediately, letting the CPU
    // construct the accept loop's penalty / softmax / slice graph
    // while the verify command buffer is still executing on the GPU.
    // The first downstream `eval()` (the accept loop's
    // `p_target.eval()` at the per-position softmax) syncs on
    // completion — the same overlap pattern DFlash applies in
    // `spec_epoch.py:2069`'s `mx.async_eval(posterior)`.
    //
    // We batch `verify_hiddens` into the same async_eval call so MLX's
    // scheduler can fuse it with the verify logits graph (they share
    // the per-position `final_norm` outputs). Only the post-accept
    // `verify_hiddens[:, K, :]` slice is actually realised on-device
    // by the chained-cycle path; for the default Step-A path the
    // batch eval is still cheap (one extra command-buffer entry).
    //
    // When the env var is OFF (default), behaviour is byte-identical
    // to pre-W6.9 — `verify_logits.eval()` blocks until verify is
    // done before the accept loop's softmax graph is built.
    // W6.19 — Fast-path eligibility: at T=0 with all penalties at
    // defaults, the per-position accept decision collapses to
    // `argmax(verify_logits[i]) == draft_id[i]` (Bug #3 invariant
    // in `accept_with_residual`). When eligible, collapse the D+1
    // per-position softmax materializations into ONE batched
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
    // When ineligible (T>0, or any penalty non-default), fall
    // through to the legacy per-position path below.
    let temp = params
        .sampling_config
        .and_then(|c| c.temperature)
        .unwrap_or(1.0);
    let penalties_no_op = params.repetition_penalty == 1.0
        && params.presence_penalty == 0.0
        && params.frequency_penalty == 0.0;
    let use_sparse_accept = mtp_sparse_accept_enabled() && temp <= 1e-6 && penalties_no_op;

    profiler.begin("mtp_verify_eval");
    if mtp_verify_async_eval() {
        tracing::debug!(
            target: "mlx_core::mtp::verify_async_eval",
            depth = depth,
            "W6.9 async_eval(verify_logits, verify_hiddens)"
        );
        MxArray::async_eval_arrays(&[&verify_logits, &verify_hiddens]);
    } else {
        // We materialize logits now so per-position slicing reads
        // from a CPU-resident buffer for penalty application. The
        // hiddens ride on the same compiled graph; we only eval the
        // K-th slice below.
        //
        // W6.19 note: the sparse-accept path also benefits from this
        // eager eval — folding verify materialization into the
        // accept-loop argmax op (one combined sync) measured ~10%
        // slower than two separate syncs on the qwen3.6-27b smoke.
        // The eager eval here lets MLX's scheduler pipeline the
        // verify command buffer with the subsequent argmax dispatch
        // build, which the combined-eval variant defeats. Kept
        // unconditional.
        verify_logits.eval();
    }
    profiler.end();
    profiler.record_duration("mtp_verify_floor", verify_only_t0.elapsed());
    let vocab = verify_logits.shape_at(2)?;

    // Step 3: per-position accept/reject. Build extended history as
    // we accept; rejecting at position i halts the loop.
    let mut accepted_tokens: Vec<u32> = Vec::with_capacity(depth + 1);
    let mut hist_extended: Vec<u32> = token_history.to_vec();
    let mut all_accepted = true;
    let mut rejection_residual: Option<i32> = None;

    if use_sparse_accept {
        // ONE batched argmax over all D+1 verify positions. Shape
        // `[1, D+1, vocab]` → `[1, D+1]` int32. Bug #3: at T=0 we
        // care only about per-position argmax — no full-vocab
        // softmax materialization needed.
        //
        // `verify_logits` may still be lazy from the verify dispatch
        // (especially under `MLX_MTP_VERIFY_ASYNC_EVAL=1`). The
        // `.eval()` below is the SINGLE sync point for the accept
        // loop — vs the legacy D × per-position `p_target.eval()`
        // path that forced D full-vocab softmaxes through Metal.
        profiler.begin("mtp_accept_argmax");
        let argmax_arr = verify_logits.argmax(-1, None)?;
        argmax_arr.eval();

        // Extract D+1 int32s into a CPU buffer. `verify_logits` was
        // `[1, D+1, vocab]`; the argmax over the last axis yields
        // `[1, D+1]`. We read flat positions 0..=depth.
        let mut target_argmax: Vec<i32> = Vec::with_capacity(depth + 1);
        for i in 0..=depth {
            target_argmax.push(argmax_arr.item_at_int32(i)?);
        }
        profiler.end();

        // Accept loop runs entirely on CPU buffers — no further GPU
        // syncs. The Leviathan-Chen accept-reject coin is unused at
        // T=0 (deterministic argmax decision); `rng` is intentionally
        // not advanced, matching `accept_with_residual`'s T=0
        // shortcut (zero RNG consumed).
        profiler.begin("mtp_accept_loop");
        for i in 0..depth {
            let target_id = target_argmax[i];
            if target_id == draft_ids[i] {
                let id_u = target_id as u32;
                accepted_tokens.push(id_u);
                hist_extended.push(id_u);
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
            let bonus_id = target_argmax[depth] as u32;
            accepted_tokens.push(bonus_id);
        }
        profiler.end();
    } else {
        // Legacy per-position path. Kept for T>0 (where residual
        // sampling needs the full target distribution) and for
        // penalty-active configurations (where `hist_extended`
        // mutates the per-position logits inside the loop).
        // Note: this wrap includes the full-accept bonus-token sample
        // (sample + eval), whereas the sparse-accept branch's bonus is
        // a CPU buffer read inside the same phase name.
        profiler.begin("mtp_accept_loop");
        for i in 0..depth {
            // verify_logits[0, i, :] → [vocab]
            let v_slice = verify_logits.slice(&[0, i as i64, 0], &[1, (i + 1) as i64, vocab])?;
            let v_logits_1d = v_slice.squeeze(Some(&[0, 1]))?;
            let penalized = apply_all_penalties(v_logits_1d, &hist_extended, params)?;
            let p_target = Activations::softmax(&penalized, Some(-1))?.astype(DType::Float32)?;
            p_target.eval();

            let sampling_cfg = params.sampling_config.unwrap_or_default();
            let (accept, out_tok) = sampling::accept_with_residual(
                &p_target,
                &draft_probs[i],
                draft_ids[i],
                &sampling_cfg,
                rng,
            )?;
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
            let i = depth;
            let v_slice = verify_logits.slice(&[0, i as i64, 0], &[1, (i + 1) as i64, vocab])?;
            let v_logits_1d = v_slice.squeeze(Some(&[0, 1]))?;
            let penalized = apply_all_penalties(v_logits_1d, &hist_extended, params)?;
            let bonus = sampling::sample(&penalized, params.sampling_config)?;
            bonus.eval();
            let bonus_id = bonus.item_at_int32(0)? as u32;
            accepted_tokens.push(bonus_id);
        }
        profiler.end();
    }

    // Step 5: rollback. `accepted_drafts` is the number of draft
    // tokens (out of `depth`) whose K/V we are KEEPING in BOTH the
    // main and the MTP draft caches. The rest must be discarded.
    //
    // Layout BEFORE this cycle (right after the macro's Step A):
    //   - Main offset advanced by 1 (Step A wrote K/V for `y`, the
    //     prior cycle's last accepted token, at the next free slot).
    //   - MTP draft offset unchanged since the prior cycle's
    //     rollback (the MTP path mirrors a snapshot of the main
    //     offset and only moves on draft / rollback).
    //
    // Verify wrote K/V for ALL `depth + 1` inputs of
    // `[last_committed_id, d_0, .., d_{depth-1}]` into the MAIN cache
    // (advancing main offset by `depth + 1`). Draft steps wrote K/V
    // for the `depth` drafted tokens into the MTP cache (advancing
    // MTP offset by `depth`).
    //
    //   - On full accept: ALL `depth + 1` verify positions are kept
    //     in main (last_committed + `depth` drafts) and ALL `depth`
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
    // Both deltas reduce to `accepted_drafts - depth`:
    //   - main_delta = (K + 1) - (depth + 1) = K - depth
    //   - mtp_delta  = K       - depth
    let accepted_drafts = if all_accepted {
        depth
    } else {
        // accepted_tokens contains `K` accepted drafts + 1 residual.
        accepted_tokens.len() - 1
    };
    profiler.begin("mtp_rollback");
    (ops.rollback)(accepted_drafts, depth);
    profiler.end();

    // W6 Bug #4 fix — on rejection, restore the main path's GDN
    // linear caches (back to "after Step A": Step A processed `y_N`
    // and the snapshot was taken right after) and replay the
    // K + 1 committed tokens that verify processed but the restore
    // discarded:
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
    // On full accept verify already left the linear state advanced
    // through `[y_N, y_{N+1}, d_0..d_{depth-1}]` (note y_{N+1} is
    // re-processed, mirroring AR), so the snapshot is simply
    // discarded on the next snapshot or reset.
    if !all_accepted {
        let mut replay_ids: Vec<u32> = Vec::with_capacity(accepted_drafts + 1);
        replay_ids.push(last_committed_id);
        // accepted_tokens = [d_0, .., d_{K-1}, residual]; we replay
        // only the K accepted drafts (NOT the residual).
        replay_ids.extend_from_slice(&accepted_tokens[..accepted_drafts]);
        profiler.begin("mtp_tape_replay");
        let replay_res = (ops.restore_and_replay_main)(&replay_ids, embedding_weight);
        profiler.end();
        replay_res?;
    }

    let _ = rejection_residual; // documented above; only used for clarity
    // `prev_hidden` / `prev_emb` are no longer needed (they were the
    // INPUTS to the cycle's drafts; the verify pass downstream of
    // them is already evaluated). They drop at end-of-function with
    // the rest of the locals; the underlying lazy MLX arrays stay
    // alive as long as any other handle still holds them.

    // W6.5 — pick the position-K slice of `verify_hiddens` and return
    // it so the caller (the `decode_loop_mtp!` macro) can chain cycles:
    // the NEXT cycle's first MTP draft uses this hidden as
    // `prev_hidden`, eliminating the per-cycle main-model "Step A"
    // forward.
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
    // Why K (not D, not D+1): the prior W6.5 scaffolding shipped
    // position D unconditionally, which only matches when ALL drafts
    // are accepted (K==D). Partial-accept cycles chained from
    // position D's hidden — the prediction context for the rejected
    // draft — and the MTP head's drafts diverged from main, dropping
    // mean acceptance from ~1.5 to ~0.8 tokens/cycle.
    let hidden_dim = verify_hiddens.shape_at(2)?;
    let verify_hidden_k = verify_hiddens.slice(
        &[0, accepted_drafts as i64, 0],
        &[1, (accepted_drafts + 1) as i64, hidden_dim],
    )?;
    Ok((
        MtpCycleOutcome {
            tokens: accepted_tokens,
        },
        verify_hidden_k,
    ))
}

/// MTP speculative decode loop. See `decode_loop!` for the
/// single-token sister macro this mirrors.
///
/// Required arguments mirror `decode_loop!`. Adds:
///   - `mtp_ops`: an [`MtpOps`] struct.
///   - `mtp_depth`: initial / fixed number of draft tokens per cycle
///     (`>= 1`, `<= 5`). When the W6.8 adaptive policy is ON (see
///     `params.mtp_adaptive_depth`), this value seeds
///     `AdaptiveDepthPolicy::new(...)` and the per-cycle depth is then
///     chosen by `pick_depth()`. When adaptive is OFF, this value is
///     used unchanged for every cycle.
///   - `mtp_rng`: an `&mut impl rand::Rng` driving the acceptance
///     coin flip. The caller picks the seed strategy
///     (`rand::rng()` — thread-local CSPRNG — is the typical
///     production choice; `StdRng::seed_from_u64(seed)` is preferred
///     in tests for determinism).
///
/// The `streaming` block is OPTIONAL — same shape as `decode_loop!`.
macro_rules! decode_loop_mtp {
    (
        mtp_ops: $mtp:expr,
        mtp_depth: $depth:expr,
        mtp_rng: $rng:expr,
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
        // Emit the FIRST token via a normal main-path forward+hidden.
        // The MTP loop needs an established last-committed token AND
        // its post-final-norm hidden state to seed the first draft.
        // After this initial forward, `prev_hidden` / `prev_emb`
        // carry the seed for the next cycle.
        let mut prev_hidden_opt: Option<$crate::array::MxArray>;
        let mut prev_emb_opt: Option<$crate::array::MxArray>;
        let mut last_committed_id_opt: Option<u32>;

        // W6.5 — chained-cycle state. `run_mtp_cycle_inner` slices
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
        // `MTP(prev_hidden=verify_hiddens[K], prev_emb=embed($y)) ->
        // next-next logits`, matching the head's training contract.
        //
        // Default policy (W6.5 follow-up; preserved post-W6.7): chaining
        // DISABLED. The position-K slice (commit 2322841) makes chaining
        // SEMANTICALLY correct — byte-exact parity at T=0 holds in both
        // modes — but at depth=3 it still measures slower than the
        // Step-A path even with W6.7's batched verify (0.60× vs 0.74×
        // on M3 Max bf16, 200-token smoke). The likely cause is the
        // per-cycle `verify_hiddens[K]` slice + eval pulling an extra
        // dependency through the compiled-graph cache that the Step-A
        // path doesn't pay. Re-measure when the chained-hidden eval is
        // folded into the next cycle's draft trace (W6.8 candidate).
        //
        // Set `MLX_MTP_CHAINED_CYCLES=1` to opt INTO chaining for
        // testing / measurement against the Step-A baseline.
        //
        // Invariants:
        //   - `None` on the FIRST iteration (no prior verify) — Step A
        //     runs unconditionally and re-seeds the hidden from a real
        //     main forward.
        //   - `None` when forced-think-end fires — that path needs Step
        //     A's forward to write `$y`'s K/V before injecting the
        //     forced token. (See the force-end branch below.)
        //   - `Some(hidden)` after every successful cycle, to be drained
        //     by the NEXT iteration before its cycle runs.
        //
        // The hidden is a lazy MLX array referencing the verify's
        // position-K `final_norm` graph node; it stays alive because
        // `g_compiled_caches` (its upstream) is alive for the rest of
        // the decode loop. See `mlx_qwen35_mtp_verify_compiled_with_hidden`
        // for the C++ lifetime contract.
        let chained_cycles_enabled: bool = $crate::models::qwen3_5::chat_common::mtp_chained_cycles_enabled();
        let mut chained_hidden_opt: Option<$crate::array::MxArray> = None;

        // W6.8 — Adaptive MTP depth policy. When `mtp_adaptive_depth`
        // is true (default unless the caller pinned `mtpDepth`), the
        // policy picks the per-cycle draft depth from a per-depth EMA
        // of `accepted_tokens / cycle_wall_ns` plus a DFlash-style
        // 3-state machine (`full | reduced | probe`). When false, the
        // policy is constructed but `pick_depth()` returns
        // `$p.mtp_depth` on every call (no transitions ever fire
        // because `record_cycle` is gated below).
        //
        // The compiled MTP verify graphs are pre-warmed at model load
        // for every depth `D ∈ {1..=5}` (W6.7), so swinging the depth
        // freely between cycles is zero-cost from the compile side.
        let mut mtp_depth_policy =
            $crate::models::qwen3_5::adaptive_depth::AdaptiveDepthPolicy::new(
                $depth.min(u8::MAX as usize) as u8,
            );

        // Track cycles for the every-256-emitted-token cache clear.
        // We use the running `gen.len()` rather than a separate step
        // counter so MTP and non-MTP loops stay byte-equivalent on
        // the cache-clear cadence.
        let mut last_clear_at: usize = $gen.len();

        // PARITY-FIX: emit the initial `$y` (sampled from the prefill's
        // last logits BEFORE this macro was entered) before Step A's
        // first iteration. AR's `decode_loop!` macro emits its input
        // `$y` at the top of each iteration; MTP's Step A only emits
        // the SAMPLED next token, which means the very first token of
        // the generation (the prefill's seed sample) never reached
        // `$gen`. Without this push MTP's output is the AR output
        // shifted left by one token — the source of the W6 parity
        // mismatch reported in `examples/qwen35-mtp-smoke.ts`. We
        // mirror the per-token bookkeeping Step A does (eval, stream
        // callback, tracker.observe_token, profiler) so the initial
        // token participates identically. The stop checks (EOS,
        // length, cancel, repetition) run at the top of the loop body
        // below — they read `$gen` so the initial push is visible.
        {
            let _stream_ctx = $crate::stream::StreamContext::new($stream);
            $profiler.begin("extract");
            $y.eval();
            let initial_token_id = $y.item_at_int32(0)? as u32;
            $profiler.end();
            $profiler.mark_first_token();
            if $report && $first_tok.is_none() {
                $first_tok = Some(std::time::Instant::now());
            }
            $gen.push(initial_token_id);
            $hist.push(initial_token_id);
            let _is_reasoning = $tracker.observe_token(initial_token_id);
            $(
                $last_r = _is_reasoning;
                if !$cancelled.load(std::sync::atomic::Ordering::Relaxed) {
                    let token_text = $crate::tokenizer::Qwen3Tokenizer::step_decode_stream(
                        &mut $ds, $tok.inner(), initial_token_id, &$gen, $slen,
                    );
                    $slen += token_text.len();
                    $cb.call(
                        Ok($crate::models::qwen3_5::model::ChatStreamChunk {
                            text: token_text, done: false, finish_reason: None,
                            tool_calls: None, thinking: None, num_tokens: None,
                            prompt_tokens: None, reasoning_tokens: None,
                            raw_text: None, cached_tokens: None, performance: None,
                            is_reasoning: Some(_is_reasoning),
                        }),
                        napi::threadsafe_function::ThreadsafeFunctionCallMode::NonBlocking,
                    );
                }
            )?
            $profiler.step();
        }

        loop {
            // PARITY-FIX: re-check the same stop conditions Step A
            // uses, BEFORE the forward, so the initial push (above)
            // and any prior-iteration push that landed us on a stop
            // condition exit cleanly without one more forward.
            if let Some(&last) = $gen.last() {
                if last == $eos {
                    $reason = String::from("stop");
                    break;
                }
            }
            $(
                if $cancelled.load(std::sync::atomic::Ordering::Relaxed) {
                    $reason = String::from("cancelled");
                    break;
                }
            )?
            if let Some(reason) = $crate::sampling::check_repetition_cutoff(
                &$gen, $p.max_consecutive_tokens, $p.max_ngram_repeats, $p.ngram_size,
            ) {
                $reason = reason.to_string();
                break;
            }
            if $gen.len() >= ($max as usize) {
                if $reason.is_empty() { $reason = String::from("length"); }
                break;
            }

            // ---- Step A vs. chained-hidden decision (W6.5). -----------
            // Default (chained_cycles_enabled=true): skip Step A's full
            // main-model forward when a chained verify hidden is
            // available from the prior cycle, unless the tracker is
            // about to force a think-end token (the forced-token path
            // needs Step A to forward `$y` so its K/V is committed
            // before we inject the forced token).
            //
            // Default (`MLX_MTP_CHAINED_CYCLES` unset): always Step A,
            // matching pre-W6.5 behaviour byte-exact. The chained path
            // stays opt-in via `MLX_MTP_CHAINED_CYCLES=1`; even after
            // W6.7's batched verify (which eliminates the per-position
            // hidden-capture loop the W6.5 follow-up flagged) the
            // chained path still regresses perf, see the comment block
            // at the top of `decode_loop_mtp!` for details.
            //
            // On the chained path the prior cycle's verify already
            // committed all accepted tokens' K/V, and the next cycle's
            // verify will write `$y`'s K/V at its position-0 input.
            // The MTP draft seeds from `chained_hidden_opt`
            // (`verify_hiddens[K]` — the prediction context for the
            // committed token at position K+1, i.e. $y itself). T=0
            // parity is preserved because verify (= main model) is the
            // ground truth and at T=0 the residual-sampler picks the
            // same token regardless of draft accuracy.
            let do_step_a = !chained_cycles_enabled
                || chained_hidden_opt.is_none()
                || $tracker.should_force_think_end();

            let _stream_ctx = $crate::stream::StreamContext::new($stream);

            if do_step_a {
                $profiler.begin("forward");
                let next_ids = $y.reshape(&[1, 1])?;
                let (mut logits, hidden, needs_squeeze) =
                    ($mtp.forward_with_hidden)(&next_ids, &$emb)?;
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
                ($mtp.eval_step)(&next_token, &logits, budget_forced);
                $profiler.end();

                $profiler.begin("eval_token");
                next_token.eval();
                $profiler.end();

                $profiler.begin("extract");
                let token_id = next_token.item_at_int32(0)? as u32;
                $profiler.end();
                $profiler.mark_first_token();
                if $report && $first_tok.is_none() {
                    $first_tok = Some(std::time::Instant::now());
                }

                $gen.push(token_id);
                $hist.push(token_id);
                let _is_reasoning = $tracker.observe_token(token_id);

                $(
                    $last_r = _is_reasoning;
                    if $cancelled.load(std::sync::atomic::Ordering::Relaxed) {
                        $reason = String::from("cancelled");
                        break;
                    }
                    let token_text = $crate::tokenizer::Qwen3Tokenizer::step_decode_stream(
                        &mut $ds, $tok.inner(), token_id, &$gen, $slen,
                    );
                    $slen += token_text.len();
                    $cb.call(
                        Ok($crate::models::qwen3_5::model::ChatStreamChunk {
                            text: token_text, done: false, finish_reason: None,
                            tool_calls: None, thinking: None, num_tokens: None,
                            prompt_tokens: None, reasoning_tokens: None,
                            raw_text: None, cached_tokens: None, performance: None,
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
                    &$gen, $p.max_consecutive_tokens, $p.max_ngram_repeats, $p.ngram_size,
                ) {
                    $reason = reason.to_string();
                    break;
                }
                if $gen.len() >= ($max as usize) {
                    if $reason.is_empty() { $reason = String::from("length"); }
                    break;
                }

                // Seed for MTP cycles using the hidden returned from this
                // forward. `hidden` is `[1, hidden_size]`; reshape to
                // `[1, 1, hidden]` for the draft FFI's `[B, T, hidden]`
                // contract.
                let hidden_dim = hidden.shape_at(1)?;
                prev_hidden_opt = Some(hidden.reshape(&[1, 1, hidden_dim])?);
                // prev_emb is the embedding of the JUST-emitted token.
                let id_arr = $crate::array::MxArray::from_int32(&[token_id as i32], &[1])?;
                let emb_2d = $emb.take(&id_arr, 0)?;
                let h = emb_2d.shape_at(1)?;
                prev_emb_opt = Some(emb_2d.reshape(&[1, 1, h])?);
                last_committed_id_opt = Some(token_id);
                $y = next_token;
            } else {
                // ---- Chained path: skip Step A entirely (W6.5). -------
                // `$y` already holds the prior cycle's last accepted
                // token (set by that cycle's tail update). That token
                // has already been pushed to `$gen` / `$hist` /
                // `tracker` AND streamed to the callback. Its K/V will
                // be written by THIS cycle's verify at position 0.
                //
                // We just need to seed the cycle's MTP draft inputs
                // from the chained hidden and the embedding of `$y`.
                let chained_h = chained_hidden_opt
                    .take()
                    .expect("chained_hidden_opt is Some on the chained path (guarded by do_step_a)");
                // `run_mtp_cycle_inner` already sliced the K-th
                // position out of the verify hiddens, so `chained_h`
                // arrives shaped `[1, 1, hidden]` — the same shape the
                // draft FFI's `[B, T, hidden]` contract expects, no
                // reshape needed.
                prev_hidden_opt = Some(chained_h);

                // Read `$y`'s id without re-evaluating it; the prior
                // cycle tail already ran `MxArray::from_int32(...)` to
                // produce a fully materialised `[1]` int32 array, so
                // `item_at_int32(0)` here is a CPU-only read.
                $y.eval();
                let token_id = $y.item_at_int32(0)? as u32;

                let id_arr = $crate::array::MxArray::from_int32(&[token_id as i32], &[1])?;
                let emb_2d = $emb.take(&id_arr, 0)?;
                let h = emb_2d.shape_at(1)?;
                prev_emb_opt = Some(emb_2d.reshape(&[1, 1, h])?);
                last_committed_id_opt = Some(token_id);
                // Note: no `$y =` assignment — `$y` is already correct.
                // No tracker.observe_token / no $gen.push / no callback —
                // the prior cycle's emit loop already handled all of
                // that for the same `token_id`.
            }
            $profiler.step();

            // ---- Step B: ONE MTP draft+verify cycle. -------------------
            // On the chained path the prior verify already committed
            // bonus/residual; this cycle's verify writes the chained
            // `$y`'s K/V at position 0 and extends the prefix by D more
            // drafts. On full accept per cycle we emit D+1 tokens for
            // D draft steps + 1 verify (one fewer main forward than
            // pre-W6.5).
            if $gen.len() >= ($max as usize) {
                if $reason.is_empty() { $reason = String::from("length"); }
                break;
            }
            if $tracker.should_force_think_end() {
                // Budget tripped during Step A's observe — defer the
                // forced token to the next Step A. On the chained path
                // this can't fire (do_step_a above forces Step A when
                // think-end is queued) but keep the guard for clarity.
                continue;
            }

            let prev_h = prev_hidden_opt
                .take()
                .expect("prev_hidden seeded by Step A or chained path");
            let prev_e = prev_emb_opt
                .take()
                .expect("prev_emb seeded by Step A or chained path");
            let last_id = last_committed_id_opt
                .expect("last_committed seeded by Step A or chained path");
            // W6 Bug #2 fix (Option Reset): re-anchor the MTP cache to
            // the main path's CURRENT offset before launching this
            // cycle's drafts. On the Step-A path the main offset has
            // advanced by 1 (Step A's forward) + the prior cycle's
            // verify advancement. On the chained path the main offset
            // has only advanced by the prior cycle's verify (Step A
            // was skipped). EITHER way, this resets the MTP K/V and
            // sets the MTP offset = current main offset, which is
            // exactly the contract `begin_cycle` is documented to
            // honour. Without it the MTP draft RoPE positions diverge
            // and drafts produce gibberish.
            ($mtp.begin_cycle)();
            // W6.8 — per-cycle depth selection. When adaptive is OFF,
            // `pick_depth()` returns the seed depth unchanged
            // (`record_cycle` is gated below). When adaptive is ON, the
            // policy hill-climbs across depth-EMA + manages the
            // `full | reduced | probe` state machine.
            let cycle_depth: usize = if $p.mtp_adaptive_depth {
                mtp_depth_policy.pick_depth() as usize
            } else {
                $depth
            };
            $profiler.begin("mtp_cycle");
            let cycle_started_at = std::time::Instant::now();
            let cycle_res =
                $crate::models::qwen3_5::chat_common::run_mtp_cycle_inner(
                    &mut $mtp,
                    prev_h,
                    prev_e,
                    last_id,
                    &$emb,
                    &$hist,
                    &$p,
                    &mut $rng,
                    &mut $profiler,
                    cycle_depth,
                );
            $profiler.end();
            // W6.5 — `run_mtp_cycle_inner` returns the verify-final
            // hidden so the NEXT outer iteration can skip Step A's
            // ~150 ms main-model forward. We stash it into
            // `chained_hidden_opt`; the iteration boundary's `do_step_a`
            // check will drain it.
            let (outcome, verify_last_hidden) = cycle_res?;
            chained_hidden_opt = Some(verify_last_hidden);

            // Throttled per-cycle MTP trace. Mirrors the AR loop's
            // every-32-steps cadence in token-count units so MTP and
            // AR runs leave comparable breadcrumb density. Reports the
            // dense compiled main-path cache offset; on the MoE path
            // this is the dense global and will read 0 — verify MoE
            // offsets via `mtp_chained_hidden_opt`-flavoured tooling.
            if ($gen.len() / 32) != (($gen.len() + outcome.tokens.len()) / 32) {
                let cache_offset = unsafe { mlx_sys::mlx_qwen35_get_cache_offset() };
                let first_tok = outcome.tokens.first().copied().unwrap_or(0);
                tracing::info!(
                    "Qwen3.5 decode MTP cycle gen_len={} depth={} committed={} \
                     first_tok={} cache_offset={}",
                    $gen.len(),
                    cycle_depth,
                    outcome.tokens.len(),
                    first_tok,
                    cache_offset,
                );
            }
            // W6.8 — feed observation to the policy AFTER the cycle's
            // tokens have been counted but BEFORE the emit loop's stop
            // checks (so the record always runs even on partial-emit
            // due to EOS / length / cancel). `committed` is the number
            // of tokens the cycle actually produced (drafts accepted +
            // residual/bonus); range `[1, depth+1]`.
            let cycle_wall_ns: u64 = cycle_started_at
                .elapsed()
                .as_nanos()
                .min(u128::from(u64::MAX)) as u64;
            let cycle_committed: u32 = outcome.tokens.len() as u32;
            if $p.mtp_adaptive_depth {
                mtp_depth_policy.record_cycle(
                    $crate::models::qwen3_5::adaptive_depth::CycleStats {
                        depth: cycle_depth as u8,
                        committed: cycle_committed,
                        wall_ns: cycle_wall_ns,
                    },
                );
                tracing::debug!(
                    target: "mlx_core::mtp::adaptive",
                    state = mtp_depth_policy.state_label(),
                    depth = cycle_depth,
                    committed = cycle_committed,
                    wall_ms = (cycle_wall_ns as f64) / 1_000_000.0,
                    next_depth = mtp_depth_policy.pick_depth(),
                    "W6.8 cycle"
                );
            }

            // Emit each accepted token through the same stop /
            // streaming pipeline as the single-token loop.
            let mut hit_stop = false;
            $profiler.begin("mtp_emit_loop");
            for tok_id in outcome.tokens.iter().copied() {
                if $gen.len() >= ($max as usize) {
                    if $reason.is_empty() { $reason = String::from("length"); }
                    hit_stop = true;
                    break;
                }
                $gen.push(tok_id);
                $hist.push(tok_id);
                let _is_reasoning = $tracker.observe_token(tok_id);
                $(
                    $last_r = _is_reasoning;
                    if $cancelled.load(std::sync::atomic::Ordering::Relaxed) {
                        $reason = String::from("cancelled");
                        hit_stop = true;
                        break;
                    }
                    let token_text = $crate::tokenizer::Qwen3Tokenizer::step_decode_stream(
                        &mut $ds, $tok.inner(), tok_id, &$gen, $slen,
                    );
                    $slen += token_text.len();
                    $cb.call(
                        Ok($crate::models::qwen3_5::model::ChatStreamChunk {
                            text: token_text, done: false, finish_reason: None,
                            tool_calls: None, thinking: None, num_tokens: None,
                            prompt_tokens: None, reasoning_tokens: None,
                            raw_text: None, cached_tokens: None, performance: None,
                            is_reasoning: Some(_is_reasoning),
                        }),
                        napi::threadsafe_function::ThreadsafeFunctionCallMode::NonBlocking,
                    );
                )?
                if tok_id == $eos {
                    $reason = String::from("stop");
                    hit_stop = true;
                    break;
                }
                if let Some(reason) = $crate::sampling::check_repetition_cutoff(
                    &$gen, $p.max_consecutive_tokens, $p.max_ngram_repeats, $p.ngram_size,
                ) {
                    $reason = reason.to_string();
                    hit_stop = true;
                    break;
                }
            }
            $profiler.end();

            // Every-256-emitted-token cache clear (matches the
            // single-token loop's cadence in token-count units).
            if $gen.len() >= last_clear_at + 256 {
                $crate::array::synchronize_and_clear_cache();
                last_clear_at = $gen.len();
            }

            if hit_stop { break; }
            // Set `$y` to the last accepted token so the next Step A
            // feeds the right token through main-path forward.
            // (Step A unconditionally re-seeds `prev_hidden_opt` /
            // `prev_emb_opt` / `last_committed_id_opt`, so no explicit
            // drain here.)
            let last = *outcome.tokens.last().expect("at least one accepted") as i32;
            $y = $crate::array::MxArray::from_int32(&[last], &[1])?;

            // W6.5-resume — when chaining IS enabled, flush the main
            // path's KV-cache lazy graph BEFORE the next cycle starts
            // AND fuse the chained `verify_hidden[K]` slice into the
            // SAME `async_eval` batch as `(token, g_compiled_caches)`.
            //
            // PRIOR (post-W6.5, pre-resume): we called the plain
            // `eval_step($y, h, false)` here. That helper does
            // `async_eval({$y} ++ g_compiled_caches)` and IGNORES `h`
            // (it only evals the logits arg when `budget_forced=true`).
            // So the chained hidden stayed LAZY across the iteration
            // boundary — when the next cycle's `(ops.draft_step)(...)`
            // built its graph against `prev_hidden = chained_hidden`,
            // materializing the slice forced a mid-cycle Metal
            // command-buffer roundtrip that the Step-A bypass doesn't
            // pay (because Step A's `forward_with_hidden` returns
            // `(logits, hidden)` as siblings of the same compiled
            // forward and the macro's `eval_step` co-schedules them
            // via `next_token → logits → hidden` dependency).
            //
            // The new helper `eval_step_with_chained_hidden` extends
            // the same dispatch to include the slice — so it becomes a
            // sibling of `(token, caches)`, the kernel scheduler can
            // overlap its materialization with the next cycle's draft
            // graph construction, and the chained path stops paying
            // the per-cycle roundtrip.
            //
            // On the Step-A path this whole branch is dead anyway:
            // Step A's `eval_step` call at the top of the NEXT
            // iteration handles the cache flush, and there is no
            // chained hidden to fold. The branch is only entered when
            // `chained_cycles_enabled=true` AND `chained_hidden_opt`
            // is `Some(...)`.
            if chained_cycles_enabled {
                if let Some(ref h) = chained_hidden_opt {
                    ($mtp.eval_step_with_chained_hidden)(&$y, h);
                }
            }
            $profiler.step();
        }

        $profiler.snapshot_memory_after();
        $profiler.report();
    }};
}

pub(crate) use decode_loop_mtp;

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
mod mtp_params_tests {
    //! W6 (MTP) — defaults + override plumbing for `ChatParams`.
    //! No Metal required; purely tests the `ChatConfig → ChatParams`
    //! extraction.

    use super::extract_chat_params;
    use crate::models::qwen3_5::model::ChatConfig;

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

    /// Defaults: MTP off, depth 3.
    #[test]
    fn defaults_disable_mtp() {
        let cfg = base_config();
        let p = extract_chat_params(&cfg);
        assert!(!p.enable_mtp, "enable_mtp must default to false");
        assert_eq!(p.mtp_depth, 3, "mtp_depth must default to 3");
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
    }

    /// W6.8 — `mtp_adaptive_depth` default resolution.
    ///
    ///   * Neither `mtpAdaptiveDepth` nor `mtpDepth` set → adaptive ON.
    ///   * `mtpDepth` set, `mtpAdaptiveDepth` unset → adaptive OFF
    ///     (caller pinned a depth).
    ///   * `mtpAdaptiveDepth = Some(true)`, `mtpDepth` set → adaptive
    ///     ON (explicit field wins; depth becomes initial seed).
    ///   * `mtpAdaptiveDepth = Some(false)`, `mtpDepth` unset → OFF.
    #[test]
    fn adaptive_depth_default_resolution() {
        // Default: no fields set → adaptive ON.
        let cfg = base_config();
        let p = extract_chat_params(&cfg);
        assert!(
            p.mtp_adaptive_depth,
            "mtp_adaptive_depth must default to true when neither field is set"
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

        // Explicit adaptive=false with no depth → OFF (uses default 3).
        let mut cfg = base_config();
        cfg.mtp_adaptive_depth = Some(false);
        let p = extract_chat_params(&cfg);
        assert!(!p.mtp_adaptive_depth);
        assert_eq!(p.mtp_depth, 3);
    }
}

#[cfg(test)]
mod mtp_cycle_tests {
    //! W6 (MTP) — `run_mtp_cycle_inner` smoke tests with mock
    //! draft / verify closures. Each test invokes the helper with a
    //! tiny synthetic vocab and validates: emitted token count,
    //! rollback callback receives the expected
    //! `(accepted_drafts, depth)` pair. Drafter and verifier closures
    //! track call counts so the tests double as wiring assertions.
    //!
    //! The rollback contract: `accepted_drafts` is the number of
    //! draft positions whose K/V we keep (range `0..=depth`). The
    //! dispatch-site callback in `model.rs` translates this into the
    //! single shared offset delta `accepted_drafts - depth` and
    //! applies it to BOTH the main and MTP compiled offsets.
    //!
    //! These are Metal-light: they DO use MLX softmax / sample /
    //! take inside the helper, so the tests skip cleanly when Metal
    //! is unavailable (mirrors the `compiled_ffi_tests` pattern in
    //! `mtp.rs`).
    //!
    //! Full token-for-token parity vs the eager Rust MTP forward is
    //! intentionally out of scope here — see task #8 of the W6 plan
    //! for the integration smoke that exercises real weights.

    use std::cell::RefCell;

    use crate::array::MxArray;
    use crate::decode_profiler::DecodeProfiler;
    use crate::models::qwen3_5::chat_common::{MtpOps, extract_chat_params, run_mtp_cycle_inner};
    use crate::models::qwen3_5::model::ChatConfig;

    use rand::SeedableRng;
    use rand::rngs::StdRng;

    const VOCAB: i64 = 8;
    const HIDDEN: i64 = 4;

    fn default_params() -> super::ChatParams {
        extract_chat_params(&ChatConfig {
            max_new_tokens: None,
            temperature: Some(1.0),
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
            enable_mtp: Some(true),
            mtp_depth: Some(3),
            mtp_adaptive_depth: None,
        })
    }

    /// Build a fake embedding table — vocab x hidden, deterministic
    /// distinct values per row so `take` returns recognizable embs.
    fn fake_embedding() -> Option<MxArray> {
        let mut data = Vec::with_capacity((VOCAB * HIDDEN) as usize);
        for v in 0..VOCAB {
            for h in 0..HIDDEN {
                data.push((v * 10 + h) as f32);
            }
        }
        MxArray::from_float32(&data, &[VOCAB, HIDDEN]).ok()
    }

    fn fake_hidden() -> Option<MxArray> {
        let data = vec![0.5f32; HIDDEN as usize];
        MxArray::from_float32(&data, &[1, 1, HIDDEN]).ok()
    }

    /// Construct a draft-step closure that always returns peaked
    /// logits at `draft_id_per_step[step]` and a constant h_next.
    /// Tracks call count in `counter`.
    fn make_draft<'a>(
        draft_id_per_step: &'a [i32],
        counter: &'a RefCell<usize>,
    ) -> impl FnMut(&MxArray, &MxArray) -> napi::Result<(MxArray, MxArray)> + 'a {
        move |_prev_h: &MxArray, _prev_e: &MxArray| {
            let step = *counter.borrow();
            *counter.borrow_mut() += 1;
            let id = draft_id_per_step[step % draft_id_per_step.len()];
            let mut logits = vec![-10.0f32; VOCAB as usize];
            logits[id as usize] = 10.0;
            let draft_logits = MxArray::from_float32(&logits, &[1, VOCAB]).expect("draft logits");
            let h_next = MxArray::from_float32(&vec![0.0f32; HIDDEN as usize], &[1, 1, HIDDEN])
                .expect("h_next");
            Ok((h_next, draft_logits))
        }
    }

    /// Construct a verify-step closure that returns logits peaked at
    /// `verify_id_per_position[i]` for each verify position, plus a
    /// zero `[1, depth+1, hidden]` stand-in for the per-position
    /// verify hiddens (W6.5 fix — the production closure exports the
    /// post-final-norm hidden at EVERY verify position so the caller
    /// can slice `verify_hiddens[:, K, :]` for chaining). For the
    /// mock tests below we don't care about contents, only shape, so
    /// a fresh zeros tensor matches the `[1, depth+1, hidden_size]`
    /// contract `run_mtp_cycle_inner` slices from.
    fn make_verify<'a>(
        verify_id_per_position: &'a [i32],
        counter: &'a RefCell<usize>,
    ) -> impl FnMut(&MxArray, &MxArray, usize) -> napi::Result<(MxArray, MxArray)> + 'a {
        move |_ids: &MxArray, _emb: &MxArray, depth: usize| {
            *counter.borrow_mut() += 1;
            let positions = depth + 1;
            assert_eq!(verify_id_per_position.len(), positions);
            let mut data = vec![-10.0f32; positions * VOCAB as usize];
            for (i, &id) in verify_id_per_position.iter().enumerate() {
                data[i * VOCAB as usize + id as usize] = 10.0;
            }
            let arr =
                MxArray::from_float32(&data, &[1, positions as i64, VOCAB]).expect("verify logits");
            // Per-position verify hiddens: [1, depth+1, hidden].
            // Mirrors the production stacked `[1, D+1, hidden_size]`
            // contract `mlx_qwen35_mtp_verify_compiled_with_hidden`
            // ships.
            let zero_hiddens = vec![0.0f32; positions * HIDDEN as usize];
            let hiddens = MxArray::from_float32(&zero_hiddens, &[1, positions as i64, HIDDEN])
                .expect("verify hiddens stub");
            Ok((arr, hiddens))
        }
    }

    fn skip_if_metal_unavailable<T, E: std::fmt::Display>(
        label: &str,
        r: Result<T, E>,
    ) -> Option<T> {
        match r {
            Ok(v) => Some(v),
            Err(e) => {
                let msg = format!("{}", e);
                if msg.contains("Metal") || msg.contains("device") {
                    eprintln!("skipping {label} (Metal unavailable): {msg}");
                    None
                } else {
                    panic!("unexpected failure in {label}: {msg}");
                }
            }
        }
    }

    /// All-accept path: drafter and verifier agree on every drafted
    /// token; cycle emits `depth + 1` tokens and the rollback
    /// callback fires with `(accepted_drafts=depth, depth=depth)` so
    /// the resulting `accepted_drafts - depth = 0` delta leaves both
    /// the main and the MTP offsets where the verify pass left them.
    #[test]
    fn all_accept_emits_depth_plus_one_tokens() {
        let depth = 3usize;
        let Some(emb) = fake_embedding() else { return };
        let Some(prev_h) = fake_hidden() else { return };
        let Some(prev_e) = fake_hidden() else { return };

        let draft_ids = vec![1i32, 2, 3];
        let verify_ids = vec![1i32, 2, 3, 4];
        let draft_ctr = RefCell::new(0usize);
        let verify_ctr = RefCell::new(0usize);
        let rollback_seen = RefCell::new(None::<(usize, usize)>);

        let mut ops = MtpOps {
            forward_with_hidden: |_ids: &MxArray,
                                  _emb: &MxArray|
             -> napi::Result<(MxArray, MxArray, bool)> {
                unreachable!("forward_with_hidden is not called inside run_mtp_cycle_inner")
            },
            draft_step: make_draft(&draft_ids, &draft_ctr),
            verify_step: make_verify(&verify_ids, &verify_ctr),
            rollback: |a: usize, d: usize| {
                *rollback_seen.borrow_mut() = Some((a, d));
            },
            eval_step: |_t: &MxArray, _l: &MxArray, _b: bool| {},
            eval_step_with_chained_hidden: |_t: &MxArray, _h: &MxArray| {},
            begin_cycle: || {},
            snapshot_main_linear: || {},
            restore_and_replay_main: |_: &[u32], _: &MxArray| Ok(()),
            // W6.18 — tests exercise the per-step `draft_step` path.
            // The fused-draft FFI is dense-only and the tests don't
            // need to validate it (cycle-level acceptance semantics
            // are path-agnostic).
            fused_draft: None,
        };
        let params = default_params();
        let mut rng = StdRng::seed_from_u64(0xC0FFEE);
        let mut profiler = DecodeProfiler::new("test", "test");

        let res = run_mtp_cycle_inner(
            &mut ops,
            prev_h,
            prev_e,
            0u32,
            &emb,
            &[],
            &params,
            &mut rng,
            &mut profiler,
            depth,
        );
        let Some((outcome, _verify_hidden)) = skip_if_metal_unavailable("all_accept", res) else {
            return;
        };
        assert_eq!(*draft_ctr.borrow(), depth, "must run depth draft steps");
        assert_eq!(*verify_ctr.borrow(), 1, "must run exactly one verify step");
        assert_eq!(
            outcome.tokens.len(),
            depth + 1,
            "all-accept must emit depth+1 tokens (depth drafts + 1 bonus)"
        );
        assert_eq!(outcome.tokens, vec![1u32, 2, 3, 4]);
        assert_eq!(
            *rollback_seen.borrow(),
            Some((depth, depth)),
            "rollback callback must receive (accepted_drafts=depth, depth=depth) on full \
             accept — produces zero offset delta in the dispatch-site formula \
             `accepted_drafts - depth`"
        );
    }

    /// Depth-1 degeneracy: 1 draft + 1 verify position still works.
    #[test]
    fn depth_one_degenerates_correctly() {
        let depth = 1usize;
        let Some(emb) = fake_embedding() else { return };
        let Some(prev_h) = fake_hidden() else { return };
        let Some(prev_e) = fake_hidden() else { return };

        let draft_ids = vec![5i32];
        let verify_ids = vec![5i32, 7];
        let draft_ctr = RefCell::new(0usize);
        let verify_ctr = RefCell::new(0usize);
        let rollback_seen = RefCell::new(None::<(usize, usize)>);

        let mut ops = MtpOps {
            forward_with_hidden: |_ids: &MxArray,
                                  _emb: &MxArray|
             -> napi::Result<(MxArray, MxArray, bool)> {
                unreachable!()
            },
            draft_step: make_draft(&draft_ids, &draft_ctr),
            verify_step: make_verify(&verify_ids, &verify_ctr),
            rollback: |a: usize, d: usize| {
                *rollback_seen.borrow_mut() = Some((a, d));
            },
            eval_step: |_t: &MxArray, _l: &MxArray, _b: bool| {},
            eval_step_with_chained_hidden: |_t: &MxArray, _h: &MxArray| {},
            begin_cycle: || {},
            snapshot_main_linear: || {},
            restore_and_replay_main: |_: &[u32], _: &MxArray| Ok(()),
            // W6.18 — tests exercise the per-step `draft_step` path.
            // The fused-draft FFI is dense-only and the tests don't
            // need to validate it (cycle-level acceptance semantics
            // are path-agnostic).
            fused_draft: None,
        };
        let params = default_params();
        let mut rng = StdRng::seed_from_u64(0xBADC0DE);
        let mut profiler = DecodeProfiler::new("test", "test");

        let res = run_mtp_cycle_inner(
            &mut ops,
            prev_h,
            prev_e,
            0u32,
            &emb,
            &[],
            &params,
            &mut rng,
            &mut profiler,
            depth,
        );
        let Some((outcome, _verify_hidden)) = skip_if_metal_unavailable("depth_one", res) else {
            return;
        };
        assert_eq!(*draft_ctr.borrow(), 1);
        assert_eq!(outcome.tokens.len(), 2, "depth=1 + full accept = 2 tokens");
        assert_eq!(outcome.tokens, vec![5u32, 7u32]);
        assert_eq!(*rollback_seen.borrow(), Some((1, 1)));
    }

    /// All-reject path: drafter and verifier argmaxes disagree on
    /// position 0 — cycle emits exactly 1 residual token and the
    /// rollback callback reports `accepted_drafts=0` so the
    /// dispatch-site delta `0 - depth = -depth` rewinds the full
    /// draft window on BOTH the main and MTP offsets.
    #[test]
    fn all_reject_emits_one_residual() {
        let depth = 3usize;
        let Some(emb) = fake_embedding() else { return };
        let Some(prev_h) = fake_hidden() else { return };
        let Some(prev_e) = fake_hidden() else { return };

        let draft_ids = vec![1i32, 2, 3];
        let verify_ids = vec![6i32, 7, 0, 0];
        let draft_ctr = RefCell::new(0usize);
        let verify_ctr = RefCell::new(0usize);
        let rollback_seen = RefCell::new(None::<(usize, usize)>);

        let mut ops = MtpOps {
            forward_with_hidden: |_ids: &MxArray,
                                  _emb: &MxArray|
             -> napi::Result<(MxArray, MxArray, bool)> {
                unreachable!()
            },
            draft_step: make_draft(&draft_ids, &draft_ctr),
            verify_step: make_verify(&verify_ids, &verify_ctr),
            rollback: |a: usize, d: usize| {
                *rollback_seen.borrow_mut() = Some((a, d));
            },
            eval_step: |_t: &MxArray, _l: &MxArray, _b: bool| {},
            eval_step_with_chained_hidden: |_t: &MxArray, _h: &MxArray| {},
            begin_cycle: || {},
            snapshot_main_linear: || {},
            restore_and_replay_main: |_: &[u32], _: &MxArray| Ok(()),
            // W6.18 — tests exercise the per-step `draft_step` path.
            // The fused-draft FFI is dense-only and the tests don't
            // need to validate it (cycle-level acceptance semantics
            // are path-agnostic).
            fused_draft: None,
        };
        let params = default_params();
        let mut rng = StdRng::seed_from_u64(0xDEAD);
        let mut profiler = DecodeProfiler::new("test", "test");

        let res = run_mtp_cycle_inner(
            &mut ops,
            prev_h,
            prev_e,
            0u32,
            &emb,
            &[],
            &params,
            &mut rng,
            &mut profiler,
            depth,
        );
        let Some((outcome, _verify_hidden)) = skip_if_metal_unavailable("all_reject", res) else {
            return;
        };
        assert_eq!(*draft_ctr.borrow(), depth);
        assert_eq!(
            outcome.tokens.len(),
            1,
            "all-reject at position 0 emits 1 residual token"
        );
        assert_eq!(
            *rollback_seen.borrow(),
            Some((0, depth)),
            "rollback must report accepted_drafts=0 on first-position reject so the \
             dispatch-site delta `0 - depth = -depth` rewinds the full draft window \
             on both caches"
        );
    }

    /// Partial-reject regression: drafter and verifier agree on the
    /// first two positions, then diverge at position 2 (out of
    /// `depth=3`). Cycle emits 3 tokens — 2 accepted drafts plus 1
    /// verifier residual — and the rollback callback reports
    /// `accepted_drafts=2`, NOT `accepted_drafts=3` (the pre-fix
    /// bug). This regression locks in the invariant that the
    /// residual sample does NOT count toward `accepted_drafts`,
    /// because the residual has no draft K/V slot; its K/V will be
    /// laid down by the NEXT cycle's Step A.
    #[test]
    fn partial_reject_reports_accepted_draft_count() {
        let depth = 3usize;
        let Some(emb) = fake_embedding() else { return };
        let Some(prev_h) = fake_hidden() else { return };
        let Some(prev_e) = fake_hidden() else { return };

        // Drafter argmaxes at 1, 2, 3; verifier agrees on the first
        // two positions (1, 2) and diverges (argmax 6) at position 2.
        // Position 3 is the bonus slot — never sampled on
        // partial-reject. The accept loop walks verify positions
        // 0..depth and compares each against `draft_ids[i]`.
        let draft_ids = vec![1i32, 2, 3];
        let verify_ids = vec![1i32, 2, 6, 0];
        let draft_ctr = RefCell::new(0usize);
        let verify_ctr = RefCell::new(0usize);
        let rollback_seen = RefCell::new(None::<(usize, usize)>);

        let mut ops = MtpOps {
            forward_with_hidden: |_ids: &MxArray,
                                  _emb: &MxArray|
             -> napi::Result<(MxArray, MxArray, bool)> {
                unreachable!()
            },
            draft_step: make_draft(&draft_ids, &draft_ctr),
            verify_step: make_verify(&verify_ids, &verify_ctr),
            rollback: |a: usize, d: usize| {
                *rollback_seen.borrow_mut() = Some((a, d));
            },
            eval_step: |_t: &MxArray, _l: &MxArray, _b: bool| {},
            eval_step_with_chained_hidden: |_t: &MxArray, _h: &MxArray| {},
            begin_cycle: || {},
            snapshot_main_linear: || {},
            restore_and_replay_main: |_: &[u32], _: &MxArray| Ok(()),
            // W6.18 — tests exercise the per-step `draft_step` path.
            // The fused-draft FFI is dense-only and the tests don't
            // need to validate it (cycle-level acceptance semantics
            // are path-agnostic).
            fused_draft: None,
        };
        let params = default_params();
        let mut rng = StdRng::seed_from_u64(0xFEED);
        let mut profiler = DecodeProfiler::new("test", "test");

        let res = run_mtp_cycle_inner(
            &mut ops,
            prev_h,
            prev_e,
            0u32,
            &emb,
            &[],
            &params,
            &mut rng,
            &mut profiler,
            depth,
        );
        let Some((outcome, _verify_hidden)) = skip_if_metal_unavailable("partial_reject", res)
        else {
            return;
        };
        assert_eq!(*draft_ctr.borrow(), depth);
        // 2 accepted drafts (positions 0,1) + 1 residual at position 2.
        assert_eq!(
            outcome.tokens.len(),
            3,
            "partial-reject K=2 must emit 2 accepted drafts + 1 residual = 3 tokens"
        );
        // On accept, `accept_with_residual` returns the original
        // drafted id (not the verifier's argmax).
        assert_eq!(
            outcome.tokens[0], 1u32,
            "first accepted draft is draft_ids[0]"
        );
        assert_eq!(
            outcome.tokens[1], 2u32,
            "second accepted draft is draft_ids[1]"
        );
        // Residual is sampled from `(p_target - p_draft)+`; with
        // sharply-peaked argmax disagreement at position 2 it MUST
        // equal the verifier argmax (6).
        assert_eq!(
            outcome.tokens[2], 6u32,
            "residual must be the verifier argmax under peaked disagreement"
        );
        assert_eq!(
            *rollback_seen.borrow(),
            Some((2, depth)),
            "rollback must report accepted_drafts=K=2 on partial reject (NOT K + 1 = 3); \
             the dispatch-site delta `2 - 3 = -1` rewinds exactly the rejected draft slot \
             on both caches. Locks in the pre-fix off-by-one regression."
        );
    }
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
