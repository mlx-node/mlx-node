//! Shared chat types used by every model family's chat entry points.
//!
//! Holds the NAPI-exported [`ChatConfig`] / [`ChatResult`] /
//! [`ChatStreamChunk`] request-response types plus the
//! [`ChatStreamHandle`] streaming-cancellation token used by every
//! family chat entry point.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use napi_derive::napi;

use crate::tokenizer::ToolDefinition;
use crate::tools::ToolCallResult;

/// Unified chat configuration shared by all model variants (Qwen3, Qwen3.5, Qwen3.5 MoE).
#[napi(object)]
#[derive(Debug, Clone, Default)]
pub struct ChatConfig {
    #[napi(ts_type = "number | undefined")]
    pub max_new_tokens: Option<i32>,
    #[napi(ts_type = "number | undefined")]
    pub temperature: Option<f64>,
    #[napi(ts_type = "number | undefined")]
    pub top_k: Option<i32>,
    #[napi(ts_type = "number | undefined")]
    pub top_p: Option<f64>,
    #[napi(ts_type = "number | undefined")]
    pub min_p: Option<f64>,
    /// Repetition penalty (1.0 = disabled). Penalizes tokens already in context.
    #[napi(ts_type = "number | undefined")]
    pub repetition_penalty: Option<f64>,
    /// Size of the context window for repetition penalty (default: 256)
    #[napi(ts_type = "number | undefined")]
    pub repetition_context_size: Option<i32>,
    /// Presence penalty (0.0 = disabled). Subtracts a flat penalty from logits of any
    /// token that appeared at least once in context. Matches OpenAI API semantics.
    #[napi(ts_type = "number | undefined")]
    pub presence_penalty: Option<f64>,
    /// Number of recent tokens to consider for presence penalty (default: 20)
    #[napi(ts_type = "number | undefined")]
    pub presence_context_size: Option<i32>,
    /// Frequency penalty (0.0 = disabled). Subtracts penalty * occurrence_count from
    /// logits of each token in context. Matches OpenAI API semantics.
    #[napi(ts_type = "number | undefined")]
    pub frequency_penalty: Option<f64>,
    /// Number of recent tokens to consider for frequency penalty (default: 20)
    #[napi(ts_type = "number | undefined")]
    pub frequency_context_size: Option<i32>,
    /// Max consecutive identical tokens before stopping (default: 0 = disabled; opt in with a positive value)
    #[napi(ts_type = "number | undefined")]
    pub max_consecutive_tokens: Option<i32>,
    /// Max n-gram repetitions before stopping (default: 0 = disabled; opt in with a positive value)
    #[napi(ts_type = "number | undefined")]
    pub max_ngram_repeats: Option<i32>,
    /// Max pattern size for n-gram repetition detection (default: 0 = disabled; opt in with a positive value)
    #[napi(ts_type = "number | undefined")]
    pub ngram_size: Option<i32>,
    #[napi(ts_type = "Array<ToolDefinition>")]
    pub tools: Option<Vec<ToolDefinition>>,
    /// Reasoning effort level. Controls whether the model thinks before answering.
    /// - "none" / "low": thinking disabled (template injects closed think block).
    ///   "none" also sets includeReasoning to false by default.
    /// - "medium" / "high": thinking enabled (default behavior).
    /// - Not set: thinking enabled (model thinks naturally).
    #[napi(ts_type = "string | undefined")]
    pub reasoning_effort: Option<String>,
    /// Maximum number of thinking tokens before forcing </think>.
    /// When the model has generated this many tokens while in thinking mode,
    /// the next token is forced to be the think_end token. None = unlimited.
    #[napi(ts_type = "number | undefined")]
    pub thinking_token_budget: Option<i32>,
    /// Whether to include reasoning/thinking content in the output.
    /// When false, the `thinking` field of ChatResult/ChatStreamChunk will always be None.
    /// Default: true (false when reasoningEffort is "none").
    #[napi(ts_type = "boolean | undefined")]
    pub include_reasoning: Option<bool>,
    /// When true, include performance metrics (TTFT, prefill tok/s, decode tok/s) in the result
    #[napi(ts_type = "boolean | undefined")]
    pub report_performance: Option<bool>,
    /// Reuse KV cache across chat-session turns for incremental prefill. Default: true.
    /// When true, the model preserves its KV cache after generation. On the next
    /// `chatSessionStart` / `chatSessionContinue` call, it prefix-matches the new
    /// token sequence against the cached tokens and only prefills the delta —
    /// avoiding redundant computation for multi-turn conversations.
    #[napi(ts_type = "boolean | undefined")]
    pub reuse_cache: Option<bool>,
    /// MTP: opt-in flag enabling the Multi-Token Prediction speculative decode
    /// loop (pure-Rust eager; qwen3.5 dense and MoE). Requires the model
    /// checkpoint to carry an MTP head (otherwise silently ignored). Default:
    /// `false`.
    #[napi(ts_type = "boolean | undefined")]
    pub enable_mtp: Option<bool>,
    /// MTP: number of draft tokens per speculative cycle.
    ///
    /// On Qwen3.5 native MTP heads it is clamped to `[1, 5]` by the verify
    /// FFI contract, and when unset native code currently pins depth 1.
    /// When `mtpAdaptiveDepth` is `true`, this value is used as the
    /// throughput-policy seed and the expected-value policy's max depth.
    /// Adaptive depth is opt-in; set `mtpAdaptiveDepth: true` explicitly to
    /// enable it.
    ///
    /// Gemma4 external drafts (`draftModelPath`) resolve the field per draft
    /// variant instead (`gemma4/model.rs` `resolve_params`, always from the
    /// RAW config value — the engine's central `[1, 5]` clamp is an MTP-head
    /// contract that does not apply to external drafts):
    /// - DSpark: an unset `mtpDepth` runs full draft blocks (the draft
    ///   checkpoint's block size — 7 tokens on `dspark_gemma4_12b_block7`),
    ///   and an explicit `mtpDepth` acts as a CAP on that block (clamped to
    ///   `[1, blockSize]`).
    /// - Assistant (Google `gemma-4-*-it-assistant`): an unset `mtpDepth`
    ///   drafts 3 tokens per cycle (`ASSISTANT_DEFAULT_DEPTH`), and an
    ///   explicit `mtpDepth` clamps to `[1, 8]` (`ASSISTANT_MAX_DEPTH`).
    ///
    /// `mtpAdaptiveDepth` is ignored for both Gemma4 external-draft variants.
    #[napi(ts_type = "number | undefined")]
    pub mtp_depth: Option<i32>,
    /// MTP: when true, the decode loop runs the adaptive
    /// depth policy. Default mode is a per-depth EMA hill-climb plus
    /// DFlash-style 3-state machine `full | reduced | probe`.
    /// `MLX_MTP_ADAPTIVE_DEPTH_MODE=expected-value` instead uses the
    /// MTPLX-style intra-cycle expected-value gate, which deepens toward
    /// `mtpDepth` by default (T=0 byte-parity verified); set
    /// `MLX_MTP_EV_ALLOW_DEEPEN=0` to pin the base depth.
    /// When false, the loop pins `mtpDepth` for every cycle.
    ///
    /// Default: false. An explicit value always wins over the default.
    #[napi(ts_type = "boolean | undefined")]
    pub mtp_adaptive_depth: Option<bool>,
}

/// Unified chat result shared by all model variants (Qwen3, Qwen3.5, Qwen3.5 MoE).
#[napi(object)]
#[derive(Debug, Clone)]
pub struct ChatResult {
    pub text: String,
    pub tool_calls: Vec<ToolCallResult>,
    pub thinking: Option<String>,
    pub num_tokens: u32,
    pub prompt_tokens: u32,
    pub reasoning_tokens: u32,
    pub finish_reason: String,
    pub raw_text: String,
    /// Number of prompt tokens served from the reused KV-cache prefix.
    ///
    /// When the native prefix-cache machinery successfully matches the new
    /// prompt against the cached conversation history (via
    /// `verify_cache_prefix_direct`), only the trailing delta is re-prefilled
    /// and this field reports the length of the reused prefix. `0` when
    /// the cache was missed or disabled and the full prompt had to be
    /// re-prefilled.
    pub cached_tokens: u32,
    /// Performance metrics (present when `reportPerformance: true` in config)
    pub performance: Option<crate::profiling::PerformanceMetrics>,
}

/// A single chunk emitted during streaming chat generation.
#[napi(object)]
#[derive(Debug, Clone)]
pub struct ChatStreamChunk {
    pub text: String,
    pub done: bool,
    pub finish_reason: Option<String>,
    pub tool_calls: Option<Vec<ToolCallResult>>,
    pub thinking: Option<String>,
    pub num_tokens: Option<u32>,
    pub prompt_tokens: Option<u32>,
    pub reasoning_tokens: Option<u32>,
    pub raw_text: Option<String>,
    /// Number of prompt tokens served from the reused KV-cache prefix on
    /// this turn. Populated on the terminal chunk (`done == true`) only;
    /// `None` on mid-stream delta chunks.
    ///
    /// Zero on a cache miss or disabled reuse; equal to the matched
    /// prefix length on a hit. Mirrors `ChatResult.cached_tokens`
    /// verbatim so session-aware streaming consumers can observe
    /// prefix-cache reuse without round-tripping to the non-streaming
    /// path. Non-terminal chunks always carry `None` — only the
    /// terminal chunk is authoritative.
    #[napi(ts_type = "number | undefined")]
    pub cached_tokens: Option<u32>,
    /// Performance metrics (only present in the final chunk when `reportPerformance: true`)
    pub performance: Option<crate::profiling::PerformanceMetrics>,
    /// Whether this delta chunk contains reasoning/thinking content.
    /// true = reasoning (inside <think>...</think>), false = content (after </think>).
    /// Only present on intermediate (non-final) chunks.
    #[napi(ts_type = "boolean | undefined")]
    pub is_reasoning: Option<bool>,
}

/// Handle returned by the streaming chat-session entry points
/// (`chat_stream_session_start`, `chat_stream_session_continue`,
/// `chat_stream_session_continue_tool`) to control an in-progress
/// streaming generation.
#[napi]
pub struct ChatStreamHandle {
    pub(crate) cancelled: Arc<AtomicBool>,
}

#[napi]
impl ChatStreamHandle {
    #[napi]
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }
}
