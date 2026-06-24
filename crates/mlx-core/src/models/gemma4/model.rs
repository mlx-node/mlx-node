use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use napi::bindgen_prelude::*;
use napi::threadsafe_function::ThreadsafeFunctionCallMode;
use napi_derive::napi;

use crate::array::mask::create_causal_mask;
use crate::array::{DType, MxArray};
use crate::decode_profiler::DecodeProfiler;
use crate::engine::backend::{
    ChatBackend, ChunkSink, DecodeStep, FinalizeArgs, PagedBackend, PagedPrefix, PagedTurnSetup,
    ResetScope, SaveStateArgs, StreamEmitter, TurnOutput, TurnSetup, WholeTurnArgs,
};
use crate::engine::cmd::ChatCmd;
use crate::engine::params::ChatParams;
use crate::inference_trace::{
    elapsed_ms, enabled as inference_trace_enabled, write as write_inference_trace,
};
use crate::models::gemma4::quantized_linear::LinearProj;
use crate::nn::{Embedding, Linear, RMSNorm};
use crate::profiling::PerformanceMetrics;
use crate::sampling::{SamplingConfig, sample};
use crate::stream::{DeviceType, Stream, StreamContext};
use crate::tokenizer::{ChatMessage, Qwen3Tokenizer};
use crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter;
use crate::transformer::rotating_kv_cache::RotatingKVCacheSnapshot;
use crate::transformer::{
    AttentionKind, KVCacheDType, KVCacheGroup, KVCachePhysicalLayout, LayerKVCacheSpec,
    derive_layer_kv_cache_routes, group_layer_kv_cache_specs,
};

use super::image_processor::{Gemma4ImageProcessor, ProcessedGemma4Image};
use super::vision::{Gemma4MultimodalEmbedder, Gemma4VisionModel};
use super::vision_embedder::Gemma4UnifiedVisionEmbedder;
use super::vision_mask::apply_bidirectional_vision_overlay;

/// Convert a JSON value to Gemma4's tool-call DSL format.
/// Strings → <|"|>str<|"|>, numbers/bools → bare, objects/arrays → recursive.
fn format_gemma4_value(val: &serde_json::Value) -> String {
    match val {
        serde_json::Value::String(s) => format!("<|\"|>{}<|\"|>", s),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Null => "null".to_string(),
        serde_json::Value::Array(arr) => {
            let items: Vec<String> = arr.iter().map(format_gemma4_value).collect();
            format!("[{}]", items.join(","))
        }
        serde_json::Value::Object(map) => {
            let mut pairs: Vec<(String, String)> = map
                .iter()
                .map(|(k, v)| (k.clone(), format_gemma4_value(v)))
                .collect();
            pairs.sort_by(|a, b| a.0.cmp(&b.0));
            let inner: Vec<String> = pairs.iter().map(|(k, v)| format!("{}:{}", k, v)).collect();
            format!("{{{}}}", inner.join(","))
        }
    }
}

/// Test-only accessor for `json_args_to_gemma4_dsl`. Used by the
/// output-parser round-trip test to verify that the parser is the exact
/// inverse of the encoder for fixture inputs.
#[cfg(test)]
pub(crate) fn json_args_to_gemma4_dsl_for_test(json_str: &str) -> String {
    json_args_to_gemma4_dsl(json_str)
}

/// Convert JSON arguments string to Gemma4 tool-call DSL.
/// Returns the inner key:value pairs (without outer braces).
fn json_args_to_gemma4_dsl(json_str: &str) -> String {
    if let Ok(serde_json::Value::Object(map)) = serde_json::from_str(json_str) {
        let mut pairs: Vec<(String, String)> = map
            .iter()
            .map(|(k, v)| (k.clone(), format_gemma4_value(v)))
            .collect();
        pairs.sort_by(|a, b| a.0.cmp(&b.0));
        pairs
            .iter()
            .map(|(k, v)| format!("{}:{}", k, v))
            .collect::<Vec<_>>()
            .join(",")
    } else {
        // If not valid JSON object, pass through as-is
        json_str.to_string()
    }
}

/// Strip Gemma4 control tokens from user-supplied content to prevent prompt injection.
///
/// Removes all Gemma4 delimiter tokens that could allow a malicious message to
/// hijack the turn structure or inject synthetic tool calls/responses.
fn escape_gemma4_content(s: &str) -> String {
    s.replace("<|turn>", "")
        .replace("<turn|>", "")
        .replace("<|tool_call>", "")
        .replace("<tool_call|>", "")
        .replace("<|tool_response>", "")
        .replace("<tool_response|>", "")
        .replace("<|tool>", "")
        .replace("<tool|>", "")
        .replace("<|channel>", "")
        .replace("<channel|>", "")
        .replace("<|think|>", "")
}

use super::config::Gemma4Config;
use super::decoder_layer::{Gemma4DecoderLayer, Gemma4LayerKind};
use super::layer_cache::Gemma4LayerCache;
use crate::engine;
use crate::engine::types::{ChatConfig, ChatResult, ChatStreamChunk};
use tracing::info;

/// PLE (Per-Layer Embeddings) model-level components.
///
/// Provides per-layer token-level information to each decoder layer.
/// Present in E2B (2.3B) and E4B (4.5B) models.
pub(crate) struct PleComponents {
    /// Embedding table: [vocab_size_per_layer_input, num_layers * ple_dim]
    pub embed_tokens_per_layer: Embedding,
    /// Projection: [hidden_size, num_layers * ple_dim]
    pub per_layer_model_projection: Linear,
    /// Norm applied per ple_dim slice: weight shape [ple_dim]
    pub per_layer_projection_norm: RMSNorm,
    /// Scale factor: 2.0^(-0.5) = 1/sqrt(2) for per_layer_input_scale
    pub per_layer_input_scale: f64,
    /// Scale factor: hidden_size^(-0.5) for per_layer_model_projection_scale
    pub per_layer_model_projection_scale: f64,
    /// Dimension of per-layer embeddings
    pub ple_dim: i32,
    /// Number of layers
    pub num_layers: i32,
    /// PLE vocab size (may be smaller than main vocab_size)
    pub vocab_size_per_layer_input: i32,
}

/// Adapter giving the paged/vision streaming cores a `cb.call(result, mode)`
/// shape over the engine's [`ChunkSink`].
///
/// The engine owns the channel and hands the probes/emitter a `&dyn
/// ChunkSink`, so the wrapper forwards `.call()` to [`ChunkSink::send`].
/// The call mode is meaningless on the mpsc path and is dropped.
struct StreamSender<'a>(&'a dyn ChunkSink);

impl StreamSender<'_> {
    fn call(&self, result: Result<ChatStreamChunk>, _mode: ThreadsafeFunctionCallMode) {
        self.0.send(result);
    }
}

fn emit_stream_delta(text: String, is_reasoning: bool, cb: &StreamSender<'_>) {
    if text.is_empty() {
        return;
    }
    cb.call(
        Ok(ChatStreamChunk {
            text,
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
        }),
        ThreadsafeFunctionCallMode::NonBlocking,
    );
}

/// Gemma4 marks both hidden reasoning and some answer-only turns with
/// `<|channel>thought\n...<channel|>`. Once a reasoning delta has been
/// streamed to Anthropic SSE we cannot re-label that content as visible
/// text, so keep leading channel bytes pending until a visible text/tool
/// segment proves the channel was real reasoning. If generation ends
/// with only that pending channel body, surface it as normal text.
#[derive(Default)]
struct Gemma4StreamDispatchState {
    pending_reasoning: String,
    visible_text_emitted: bool,
    tool_call_seen: bool,
}

impl Gemma4StreamDispatchState {
    fn dispatch_segments(
        &mut self,
        segments: Vec<super::output_parser::StreamSegment>,
        cb: &StreamSender<'_>,
    ) {
        use super::output_parser::StreamSegment;
        for seg in segments {
            match seg {
                StreamSegment::Text(text) => {
                    if text.is_empty() {
                        continue;
                    }
                    self.flush_pending_reasoning(cb);
                    self.visible_text_emitted = true;
                    emit_stream_delta(text, false, cb);
                }
                StreamSegment::Reasoning(text) => {
                    if text.is_empty() {
                        continue;
                    }
                    if self.visible_text_emitted || self.tool_call_seen {
                        emit_stream_delta(text, true, cb);
                    } else {
                        self.pending_reasoning.push_str(&text);
                    }
                }
                StreamSegment::ToolCall(_) => {
                    self.tool_call_seen = true;
                    self.flush_pending_reasoning(cb);
                    // Accumulated on `parser.tool_calls()` for the terminal chunk.
                }
            }
        }
    }

    fn finish(&mut self, cb: &StreamSender<'_>) {
        if self.pending_reasoning.is_empty() {
            return;
        }
        let text = std::mem::take(&mut self.pending_reasoning);
        if self.visible_text_emitted || self.tool_call_seen {
            emit_stream_delta(text, true, cb);
        } else {
            self.visible_text_emitted = true;
            emit_stream_delta(text, false, cb);
        }
    }

    fn flush_pending_reasoning(&mut self, cb: &StreamSender<'_>) {
        if self.pending_reasoning.is_empty() {
            return;
        }
        let text = std::mem::take(&mut self.pending_reasoning);
        emit_stream_delta(text, true, cb);
    }
}

fn promote_channel_only_output(parsed: &mut super::output_parser::Gemma4ParsedOutput) {
    if parsed.text.trim().is_empty()
        && parsed.tool_calls.is_empty()
        && parsed
            .thinking
            .as_deref()
            .is_some_and(|thinking| !thinking.trim().is_empty())
    {
        parsed.text = parsed.thinking.take().unwrap_or_default();
    }
}

/// Gemma4's [`StreamEmitter`]: routes every committed token's raw
/// (special-token-preserving — [`ChatBackend::stream_skip_special_tokens`]
/// returns `false`) text through [`Gemma4StreamParser`] +
/// [`Gemma4StreamDispatchState`]: channel/tool-call segmentation,
/// pending-reasoning buffering, channel-only promotion, empty-chunk
/// filtering. `is_reasoning` / `include_reasoning` are deliberately
/// ignored — Gemma4's reasoning labeling comes from the parser's channel
/// markers, not the engine's `<think>`-token tracker (which never
/// activates: [`ChatBackend::thinking_setup`] returns `enabled: false`).
struct Gemma4Emitter {
    parser: super::output_parser::Gemma4StreamParser,
    dispatch: Gemma4StreamDispatchState,
}

impl Gemma4Emitter {
    fn new() -> Self {
        Self {
            parser: super::output_parser::Gemma4StreamParser::new(),
            dispatch: Gemma4StreamDispatchState::default(),
        }
    }
}

impl StreamEmitter for Gemma4Emitter {
    fn on_token_text(
        &mut self,
        token_text: &str,
        _is_reasoning: bool,
        _include_reasoning: bool,
        sink: &dyn ChunkSink,
    ) {
        let cb = StreamSender(sink);
        let segments = self.parser.feed(token_text);
        self.dispatch.dispatch_segments(segments, &cb);
    }

    fn on_residual(
        &mut self,
        residual: &str,
        _is_reasoning: bool,
        _include_reasoning: bool,
        sink: &dyn ChunkSink,
    ) {
        // Residual flush: feed the leftover bytes through the same parser.
        // The trailing `flush()` lives in `finish` below (the engine calls
        // `finish` unconditionally, so the flush happens whether or not a
        // residual existed — identical segment sequence either way since
        // `dispatch_segments` is stateful-sequential).
        let cb = StreamSender(sink);
        let segments = self.parser.feed(residual);
        self.dispatch.dispatch_segments(segments, &cb);
    }

    fn finish(&mut self, result: &ChatResult, sink: &dyn ChunkSink) {
        let cb = StreamSender(sink);
        let tail = self.parser.flush();
        self.dispatch.dispatch_segments(tail, &cb);
        self.dispatch.finish(&cb);

        // Terminal chunk: text stays empty (segments already streamed);
        // tool_calls/thinking come from the stream parser
        // (`parser.tool_calls()` / `.thinking()`); everything else from the
        // finalized result. `result.finish_reason` already carries the
        // tool_calls promotion from `finalize_turn`, which parses the same
        // raw text the parser does.
        let parsed_tool_calls = self.parser.tool_calls();
        let parsed_thinking = self.parser.thinking();
        cb.call(
            Ok(ChatStreamChunk {
                text: String::new(),
                done: true,
                finish_reason: Some(result.finish_reason.clone()),
                tool_calls: Some(parsed_tool_calls),
                thinking: parsed_thinking,
                num_tokens: Some(result.num_tokens),
                prompt_tokens: Some(result.prompt_tokens),
                reasoning_tokens: Some(result.reasoning_tokens),
                raw_text: Some(result.raw_text.clone()),
                cached_tokens: Some(result.cached_tokens),
                performance: result.performance.clone(),
                is_reasoning: None,
            }),
            ThreadsafeFunctionCallMode::NonBlocking,
        );
    }
}

/// Internal model state owned exclusively by the dedicated model thread.
///
/// No `Arc<RwLock<>>` — the model thread has sole ownership.
pub(crate) struct Gemma4Inner {
    pub(crate) config: Gemma4Config,
    pub(crate) embed_tokens: Embedding,
    pub(crate) layers: Vec<Gemma4DecoderLayer>,
    pub(crate) final_norm: RMSNorm,
    pub(crate) lm_head: Option<LinearProj>,
    /// Pre-transposed embedding weight for tied lm_head: [hidden_size, vocab_size].
    /// Only populated when tie_word_embeddings=true.
    pub(crate) embed_weight_t: Option<MxArray>,
    pub(crate) ple: Option<PleComponents>,
    // Vision components (None for text-only models)
    pub(crate) vision_tower: Option<Gemma4VisionModel>,
    /// Encoder-free unified vision embedder. `Some` only for the unified
    /// multimodal checkpoint (`unified_vision_config.is_some()`); mutually
    /// exclusive with `vision_tower` (the SigLIP path).
    pub(crate) unified_vision_embedder: Option<Gemma4UnifiedVisionEmbedder>,
    pub(crate) embed_vision: Option<Gemma4MultimodalEmbedder>,
    /// Encoder-free unified AUDIO embedder. `Some` only when the checkpoint
    /// declares an `audio_config` (`config.has_audio`). Structurally identical
    /// to `embed_vision` (RMSNormNoScale + Linear), but projects raw
    /// 640-sample audio windows (`audio_embed_dim` → `hidden_size`).
    pub(crate) embed_audio: Option<Gemma4MultimodalEmbedder>,
    pub(crate) image_processor: Option<Gemma4ImageProcessor>,
    pub(crate) tokenizer: Option<Arc<Qwen3Tokenizer>>,
    /// Lazily-initialized KV caches that persist across chat turns.
    ///
    /// `None` after construction and after `reset_caches_sync`. Populated
    /// by `init_caches_sync`, triggered on the first turn of a session by
    /// the engine's miss-path `reset_caches(ResetScope::PrefixMiss)` (or
    /// defensively inside [`ChatBackend::prefill`] / the vision cores).
    /// Shared across turns by the session API.
    pub(crate) caches: Option<Vec<Gemma4LayerCache>>,
    /// Tokens (post image-expansion) whose KV state is currently live in
    /// `caches`. Maintained in parallel with `caches` for prefix-reuse
    /// verification in Step 5c. Empty when no session is active.
    pub(crate) cached_token_history: Vec<u32>,
    /// Content hash of the image set associated with the live cache. Used
    /// in Step 5c to detect mid-session image changes (which require a
    /// full session restart). `None` when no session is active or the
    /// session is text-only.
    pub(crate) cached_image_key: Option<u64>,
    /// Content hash of the audio set associated with the live cache. Audio
    /// counterpart of `cached_image_key`: set after an audio prefill so a
    /// follow-up text delta is rejected (the continue path is text-only) and
    /// a follow-up audio turn cold-restarts. `None` for text-only / image-only
    /// sessions.
    pub(crate) cached_audio_key: Option<u64>,
    /// Block-paged KV adapter (vLLM-style refcounted prefix cache).
    ///
    /// **Opt-in via `Gemma4Config::use_block_paged_cache`**. Gemma4's
    /// hybrid sliding+global attention, K=V sharing, KV-shared layers
    /// (`forward_shared`), MoE/PLE branches, and per-layer-type head
    /// dimensions are all handled by
    /// `Gemma4DecoderLayer::forward_paged_or_flat`, which routes only
    /// global attention layers through this adapter. Defaults to `None`
    /// when the config flag is unset, in which case the model falls
    /// back to the flat `Gemma4LayerCache` path.
    pub(crate) paged_adapter: Option<PagedKVCacheAdapter>,
    sliding_prefix_checkpoints: VecDeque<Gemma4SlidingPrefixCheckpoint>,
    sliding_prompt_boundary_checkpoint: Option<Gemma4SlidingPrefixCheckpoint>,
    sliding_last_history_checkpoint: Option<Gemma4SlidingHistoryCheckpoint>,
    /// True only while a media (audio / non-unified image) turn left its
    /// global paged KV live AND a sliding history checkpoint remembered at the
    /// full kept-live prefix, so a follow-up text delta can warm-continue on
    /// the live media KV causally. Set exclusively by
    /// `finalize_vision_turn_media_state` on the continuable branch; reset to
    /// `false` at every non-continuable point (`clear_reuse_state`, both vision
    /// prefill-start blocks, the non-continuable finalize). When `false`, the
    /// `text_delta_image_guard` rejects a media-session delta as today.
    media_session_continuable: bool,
    pub(crate) model_id: u64,
}

/// Gemma 4 dense language model.
///
/// Supports E2B (2.3B), E4B (4.5B), and 31B variants.
/// Features: hybrid attention (sliding + global), GeGLU MLP, logit softcapping,
/// embedding scaling, and optional per-layer embeddings.
///
/// All model state lives on a dedicated OS thread. NAPI methods dispatch
/// commands via channels and await responses.
#[napi]
pub struct Gemma4Model {
    /// Dedicated model thread owning `Gemma4Inner`. `None` when the model
    /// was constructed via `new(config)` without loading weights — in that
    /// uninitialized state every session method returns an error and
    /// only `isInitialized` is meaningful. Mirrors the same `Option<..>`
    /// gate used by the OCR models (`VLModel`, `QianfanOCRModel`).
    ///
    /// Gemma4 is chat-only (no training/generate variants), so the
    /// thread dispatches the model-neutral [`ChatCmd`] directly via
    /// `engine::cmd::handle_chat_cmd::<Gemma4Inner>` — no per-family
    /// command enum.
    pub(crate) thread: Option<crate::model_thread::ModelThread<ChatCmd>>,
    pub(crate) model_id: u64,
    /// Whether the loaded config includes `vision_config`. Mirrored here so
    /// the NAPI side can fail fast on image inputs to a text-only model
    /// without round-tripping to the model thread. The actual image
    /// processor lives on `Gemma4Inner` and runs on the model thread.
    pub(crate) has_vision: bool,
    /// Whether the loaded config declares an `audio_config` (unified Gemma 4
    /// audio support, `Gemma4Config::has_audio`). Mirrored here so the NAPI
    /// image-guard can fail fast on audio inputs to a model with no audio
    /// support without round-tripping to the model thread.
    pub(crate) has_audio: bool,
    /// Whether the model was loaded with real weights. `false` for
    /// `new Gemma4Model(config)` calls that never called `load()`.
    /// Session methods check this and refuse to dispatch when false,
    /// since the coordinator was never told about this model's delta
    /// (its guard is `None`) — running inference on that stub would
    /// under-cap the allocator.
    pub(crate) initialized: bool,
    /// Snapshot of `Gemma4Inner::paged_adapter.is_some()` captured at
    /// construction time. Default-OFF on Gemma4 (parity-blocked — see
    /// `Gemma4Config::use_block_paged_cache` and the WIP per-layer
    /// numerical-diff tracker), so this is `false` for the entire matrix
    /// of currently-shipping configs. Stubs from `new(config)` always
    /// report `false` because no inner was constructed. Surfaced through
    /// the `hasBlockPagedCache()` NAPI method so server endpoints can
    /// branch on it without round-tripping through the model thread.
    pub(crate) paged_active: bool,
    /// RAII: unregisters this model's delta from the cache-limit
    /// coordinator on drop. `None` for instances constructed via the
    /// synchronous `new(config)` path that never loaded weights.
    pub(crate) _cache_limit_guard: Option<crate::cache_limit::CacheLimitGuard>,
}

static MODEL_ID_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Classification of the prefix-cache decision made from a
/// [`Gemma4Inner::verify_cache_prefix`] return value plus the incoming
/// token count.
///
/// Test-only mirror of the reset-or-reuse branch the engine session
/// core (`engine::session::chat_turn_core`) takes from this backend's
/// `verify_cache_prefix` return — separating the decision
/// logic from the native state mutation so the "exact-match routes to
/// miss" invariant can be pinned by pure-logic unit tests that do not
/// require a loaded Gemma4 model. Production code keeps the inlined
/// form for zero-overhead dispatch; this enum exists solely to drive
/// `prefix_cache_decision_tests`'s four-case coverage (empty cache,
/// strict-extend hit, divergence miss, exact-match miss). Any change
/// to the inlined production branch MUST be mirrored here or the test
/// ceases to guard the real code.
#[cfg(test)]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub(crate) enum PrefixCacheDecision {
    /// Strict-extend hit: the new prompt begins with the cached prefix
    /// and carries additional delta tokens. Warm-reuse safe: skip the
    /// cached prefix and prefill only the tail.
    StrictExtendHit,
    /// Cache miss — covers three sub-cases that all dispatch through
    /// the same `reset_caches_sync` + `init_caches_sync` + full-prefill
    /// branch:
    /// * `cached_prefix_len == 0` (no prior cache or verifier rejected
    ///   the prefix overlap for any reason).
    /// * `cached_prefix_len == tokens_len` (exact-match) — routed to
    ///   miss because Gemma4 has no snapshot of final-step logits and
    ///   no cheap rewind primitive for its sliding-window cache.
    Miss,
}

const GEMMA4_SLIDING_PREFIX_CHECKPOINT_MIN_LIMIT: usize = 16;
const GEMMA4_SLIDING_PREFIX_CHECKPOINT_WINDOW_MULTIPLIER: usize = 2;
const GEMMA4_SLIDING_PREFIX_CHECKPOINT_MAX_DEFAULT_LIMIT: usize = 128;
const GEMMA4_PAGED_CACHE_MIN_DEFAULT_MEMORY_MB: u32 = 256;
const BYTES_PER_MIB: u64 = 1024 * 1024;

struct Gemma4SlidingPrefixCheckpoint {
    prefix_len: u32,
    block_size: u32,
    final_block_hash: u64,
    tokens: Vec<u32>,
    snapshots: Vec<Option<RotatingKVCacheSnapshot>>,
}

struct Gemma4SlidingHistoryCheckpoint {
    tokens: Vec<u32>,
    snapshots: Vec<Option<RotatingKVCacheSnapshot>>,
}

struct Gemma4SlidingPrefixCheckpointHit {
    prefix_len: u32,
    caches: Vec<Gemma4LayerCache>,
}

#[derive(Default)]
struct Gemma4SlidingCheckpointStoreTrace {
    stored: bool,
    eval_ms: f64,
    snapshot_ms: f64,
    token_clone_ms: f64,
    update_ms: f64,
    total_ms: f64,
}

impl Gemma4SlidingCheckpointStoreTrace {
    fn finish(mut self, start: Option<std::time::Instant>) -> Self {
        self.total_ms = start.map(elapsed_ms).unwrap_or(0.0);
        self
    }
}

struct Gemma4SlidingPrefixPreparation {
    state: &'static str,
    primed_prefix_len: u32,
}

struct Gemma4PagedTurnPreparation {
    cached_prefix_len: u32,
    suffix_len: u32,
    sliding_primed_prefix_len: u32,
}

fn compute_gemma4_paged_prefix_block_hash(
    tokens: &[u32],
    prefix_len: u32,
    block_size: u32,
    cache_salt: u64,
) -> Option<u64> {
    if prefix_len == 0 || block_size == 0 || !prefix_len.is_multiple_of(block_size) {
        return None;
    }

    let prefix_len = prefix_len as usize;
    let block_size = block_size as usize;
    if prefix_len > tokens.len() {
        return None;
    }

    let num_blocks = prefix_len / block_size;
    let mut parent_hash = 0;
    for block_idx in 0..num_blocks {
        let start = block_idx * block_size;
        let end = start + block_size;
        parent_hash = if block_idx == 0 && cache_salt != 0 {
            mlx_paged_attn::hash_tokens(&tokens[start..end], parent_hash, &[cache_salt])
        } else {
            mlx_paged_attn::hash_tokens(&tokens[start..end], parent_hash, &[])
        };
    }

    Some(parent_hash)
}

fn gemma4_sliding_caches_ready_at(
    config: &Gemma4Config,
    caches: Option<&[Gemma4LayerCache]>,
    offset: u32,
) -> Result<bool> {
    let Some(caches) = caches else {
        return Ok(false);
    };
    if caches.len() != config.num_hidden_layers as usize {
        return Ok(false);
    }
    for (layer_idx, cache) in caches.iter().enumerate() {
        if !config.is_sliding_layer(layer_idx) {
            continue;
        }
        if !cache.sliding_offset_matches(offset as i32)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn snapshot_gemma4_sliding_caches(
    config: &Gemma4Config,
    caches: &[Gemma4LayerCache],
    expected_offset: u32,
) -> Result<Option<Vec<Option<RotatingKVCacheSnapshot>>>> {
    if !gemma4_sliding_caches_ready_at(config, Some(caches), expected_offset)? {
        return Ok(None);
    }

    let mut snapshots = Vec::with_capacity(caches.len());
    for (layer_idx, cache) in caches.iter().enumerate() {
        if config.is_sliding_layer(layer_idx) {
            let Some(snapshot) = cache.snapshot_sliding()? else {
                return Ok(None);
            };
            snapshots.push(Some(snapshot));
        } else {
            snapshots.push(None);
        }
    }
    Ok(Some(snapshots))
}

fn materialize_gemma4_sliding_snapshots(
    snapshots: &mut [Option<RotatingKVCacheSnapshot>],
) -> Result<()> {
    for snapshot in snapshots
        .iter_mut()
        .filter_map(|snapshot| snapshot.as_mut())
    {
        snapshot.keys = snapshot.keys.copy()?;
        snapshot.values = snapshot.values.copy()?;
    }

    let mut arrays: Vec<&MxArray> = Vec::new();
    for snapshot in snapshots.iter().filter_map(|snapshot| snapshot.as_ref()) {
        arrays.push(&snapshot.keys);
        arrays.push(&snapshot.values);
    }
    MxArray::eval_arrays(&arrays)
}

fn restore_gemma4_sliding_caches(
    config: &Gemma4Config,
    snapshots: &[Option<RotatingKVCacheSnapshot>],
    expected_offset: u32,
) -> Result<Option<Vec<Gemma4LayerCache>>> {
    if snapshots.len() != config.num_hidden_layers as usize {
        return Ok(None);
    }

    let mut caches = init_caches_for_config(config);
    for (layer_idx, cache) in caches
        .iter_mut()
        .enumerate()
        .take(config.num_hidden_layers as usize)
    {
        if !config.is_sliding_layer(layer_idx) {
            continue;
        }
        let Some(snapshot) = snapshots.get(layer_idx).and_then(|s| s.as_ref()) else {
            return Ok(None);
        };
        if snapshot.offset != expected_offset as i32 {
            return Ok(None);
        }
        cache.restore_sliding_snapshot(snapshot)?;
    }

    if !gemma4_sliding_caches_ready_at(config, Some(&caches), expected_offset)? {
        return Ok(None);
    }

    Ok(Some(caches))
}

/// Test-only helper: decide what to do given the verifier's answer and
/// the incoming prompt length. Exact-match (`cached_prefix_len ==
/// tokens_len`) and zero-length prefix both route to
/// [`PrefixCacheDecision::Miss`].
///
/// Mirrors the engine session core's reset-or-reuse branch over this
/// backend's `verify_cache_prefix` return
/// (`engine::session::chat_turn_core`); lifting it out keeps the
/// invariant pinnable without loading a real Gemma4 model.
#[cfg(test)]
#[inline]
pub(crate) fn classify_prefix_cache_decision(
    cached_prefix_len: usize,
    tokens_len: usize,
) -> PrefixCacheDecision {
    if cached_prefix_len > 0 && cached_prefix_len < tokens_len {
        PrefixCacheDecision::StrictExtendHit
    } else {
        PrefixCacheDecision::Miss
    }
}

impl Gemma4Inner {
    /// Create a new Gemma4Inner with empty (uninitialized) weights.
    pub(crate) fn new(config: Gemma4Config) -> Result<Self> {
        let num_layers = config.num_hidden_layers as usize;
        let hidden_size = config.hidden_size as u32;
        let vocab_size = config.vocab_size as u32;

        let embed_tokens = Embedding::new(vocab_size, hidden_size)?;
        let final_norm = RMSNorm::new(hidden_size, Some(config.rms_norm_eps))?;

        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(LinearProj::Standard(Linear::new(
                hidden_size,
                vocab_size,
                Some(false),
            )?))
        };

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(Gemma4DecoderLayer::new(&config, i)?);
        }

        // Initialize PLE model-level components if enabled
        let ple = if config.per_layer_input_embeds {
            let ple_dim = config.ple_dim();
            let vocab_ple = config.vocab_size_per_layer_input.unwrap_or(0);
            if ple_dim > 0 && vocab_ple > 0 {
                let total_ple_dim = (num_layers as i32) * ple_dim;
                Some(PleComponents {
                    embed_tokens_per_layer: Embedding::new(vocab_ple as u32, total_ple_dim as u32)?,
                    per_layer_model_projection: Linear::new(
                        hidden_size,
                        total_ple_dim as u32,
                        Some(false),
                    )?,
                    per_layer_projection_norm: RMSNorm::new(
                        ple_dim as u32,
                        Some(config.rms_norm_eps),
                    )?,
                    per_layer_input_scale: 2.0_f64.powf(-0.5),
                    per_layer_model_projection_scale: (config.hidden_size as f64).powf(-0.5),
                    ple_dim,
                    num_layers: num_layers as i32,
                    vocab_size_per_layer_input: vocab_ple,
                })
            } else {
                None
            }
        } else {
            None
        };

        // Initialize vision components. Two disjoint paths:
        //  - SigLIP vision tower (dense gemma4 family), driven by `vision_config`.
        //  - Encoder-free unified embedder, driven by `unified_vision_config`.
        let (vision_tower, unified_vision_embedder, embed_vision, image_processor) =
            if let Some(ref vc) = config.vision_config {
                let vt = Gemma4VisionModel::new(vc)?;
                let ev = Gemma4MultimodalEmbedder::new(
                    vc.hidden_size,
                    config.hidden_size,
                    vc.rms_norm_eps,
                )?;
                let ip = Gemma4ImageProcessor::new(
                    vc.patch_size,
                    vc.default_output_length,
                    vc.pooling_kernel_size,
                );
                (Some(vt), None, Some(ev), Some(ip))
            } else if let Some(ref uvc) = config.unified_vision_config {
                let embedder = Gemma4UnifiedVisionEmbedder::new(uvc)?;
                let ev = Gemma4MultimodalEmbedder::new(
                    uvc.output_proj_dims,
                    config.hidden_size,
                    uvc.rms_norm_eps,
                )?;
                let ip = Gemma4ImageProcessor::new_unified(
                    uvc.patch_size,
                    uvc.num_soft_tokens,
                    uvc.pooling_kernel_size,
                    uvc.model_patch_size,
                );
                (None, Some(embedder), Some(ev), Some(ip))
            } else {
                (None, None, None, None)
            };

        // Encoder-free unified audio embedder. Built only when the checkpoint
        // declares an `audio_config` (`has_audio`). The raw-window projection is
        // Linear(audio_samples_per_token → hidden_size); the embedder's
        // `set_weight` later validates the [hidden, in] shape against the loaded
        // [3840, 640] tensor.
        let embed_audio = if config.has_audio {
            let in_dim = config.audio_samples_per_token.unwrap_or(640);
            Some(Gemma4MultimodalEmbedder::new(
                in_dim,
                config.hidden_size,
                config.rms_norm_eps,
            )?)
        } else {
            None
        };

        let model_id = MODEL_ID_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Block-paged KV adapter — default-on; opt out via
        // `use_block_paged_cache: false`.
        //
        // The long-term source of truth is the model-independent
        // LayerKVCacheSpec plan: Gemma4 declares full/sliding/shared KV
        // requirements, common transformer code groups those specs, and model
        // dispatch should consume opaque group metadata. Runtime still uses the
        // existing single PagedKVCacheAdapter for full-attention groups while
        // sliding-window groups stay on RotatingKVCache until true paged
        // sliding eviction is wired.
        //
        // Cache dtype: BFloat16 (Gemma4's production dtype). KV-shared layers
        // are aliases and do not consume physical pool slots; they resolve to
        // their anchor's group ordinal through `compute_layer_kinds`.
        // The block-paged KV path uses Metal-only kernels; on a non-Metal
        // backend (the CUDA/Linux build) its write/gather methods are throwing
        // stubs. Force flat eager there by leaving the adapter None, so the
        // `paged_adapter.is_some()` routing falls through to the flat path.
        // macOS is unaffected — the probe is always true, so the default wins.
        let paged_adapter = if config.use_block_paged_cache.unwrap_or(true)
            && crate::engine::persistence::compiled_forward_backend_available()
        {
            let block_size = config.paged_block_size.unwrap_or(16);
            let kv_cache_specs =
                compute_layer_kv_cache_specs(&config, block_size, KVCacheDType::BFloat16).map_err(
                    |e| {
                        Error::from_reason(format!(
                            "Gemma4 block-paged adapter: failed to build KV cache specs: {e}"
                        ))
                    },
                )?;
            let kv_cache_groups = compute_layer_kv_cache_groups(
                &config,
                block_size,
                KVCacheDType::BFloat16,
                gemma4_paged_prefill_group_max_chunk(),
            )
            .map_err(|e| {
                Error::from_reason(format!(
                    "Gemma4 block-paged adapter: failed to group KV cache specs: {e}"
                ))
            })?;
            let full_groups: Vec<&KVCacheGroup> = kv_cache_groups
                .iter()
                .filter(|group| matches!(group.attention_kind, AttentionKind::Full))
                .collect();
            if full_groups.len() > 1 {
                return Err(Error::from_reason(format!(
                    "Gemma4 block-paged adapter currently supports one full-attention KV group, \
                     but spec grouping produced {} groups. This model needs the grouped \
                     HybridKVCacheManager path.",
                    full_groups.len()
                )));
            }
            let Some(full_group) = full_groups.first().copied() else {
                return Err(napi::Error::from_reason(
                    "Gemma4 block-paged adapter: config has no full_attention KV group; \
                     paged KV cache requires at least one global attention layer",
                ));
            };
            let num_global_layers = physical_full_attention_layer_count(&kv_cache_specs) as u32;
            if num_global_layers == 0 {
                return Err(napi::Error::from_reason(
                    "Gemma4 block-paged adapter: config has no full_attention layers; \
                     paged KV cache requires at least one global attention layer",
                ));
            }

            let head_size = full_group.physical_layout.head_size;
            let num_kv_heads = full_group.physical_layout.num_kv_heads;
            let max_seq_len = u32::try_from(config.max_position_embeddings).map_err(|_| {
                napi::Error::from_reason(format!(
                    "Gemma4 block-paged adapter: invalid max_position_embeddings={}",
                    config.max_position_embeddings
                ))
            })?;
            if max_seq_len == 0 {
                return Err(napi::Error::from_reason(
                    "Gemma4 block-paged adapter: max_position_embeddings must be > 0",
                ));
            }
            let default_gpu_memory_mb = gemma4_default_paged_cache_memory_mb(
                max_seq_len,
                block_size,
                head_size,
                num_kv_heads,
                num_global_layers,
            );
            let (gpu_memory_mb, paged_cache_memory_source) =
                if let Some(configured_memory_mb) = config.paged_cache_memory_mb {
                    (configured_memory_mb, "config")
                } else {
                    (default_gpu_memory_mb, "auto_full_context")
                };

            let pa_config = mlx_paged_attn::PagedAttentionConfig {
                block_size,
                gpu_memory_mb,
                head_size,
                num_kv_heads,
                // Pool covers only physical full-attention layers. KV-shared
                // aliases reuse their anchor's slot and do not allocate.
                num_layers: num_global_layers,
                use_fp8_cache: Some(false),
                max_seq_len: Some(max_seq_len),
                max_batch_size: Some(32),
            };

            let num_blocks = pa_config.calculate_num_blocks();
            if num_blocks == 0 {
                return Err(napi::Error::from_reason(format!(
                    "Gemma4 block-paged adapter: gpu_memory_mb={gpu_memory_mb} too small \
                     (head_size={head_size}, num_kv_heads={num_kv_heads}, \
                     block_size={block_size}, num_global_layers={num_global_layers})",
                )));
            }

            let allocator = Arc::new(std::sync::Mutex::new(mlx_paged_attn::BlockAllocator::new(
                num_blocks, block_size,
            )));

            let cache_dtype = mlx_paged_attn::metal::MetalDtype::BFloat16;
            let pool = mlx_paged_attn::LayerKVPool::new(pa_config, num_blocks, cache_dtype)
                .map_err(|e| {
                    napi::Error::from_reason(format!(
                        "Failed to construct LayerKVPool for Gemma4 block-paged adapter: {e}"
                    ))
                })?;

            let adapter =
                PagedKVCacheAdapter::new(allocator, Arc::new(pool), block_size).map_err(|e| {
                    napi::Error::from_reason(format!(
                        "Failed to construct Gemma4 PagedKVCacheAdapter: {e}"
                    ))
                })?;

            tracing::info!(
                "Gemma4 block-paged adapter enabled: num_blocks={num_blocks}, \
                 block_size={block_size}, gpu_memory_mb={gpu_memory_mb}, \
                 paged_cache_memory_source={paged_cache_memory_source}, \
                 max_seq_len={max_seq_len}, max_cached_tokens={}, \
                 physical_full_layers={num_global_layers}, kv_groups={}, \
                 full_group_max_admission_blocks={}, cache_dtype=BFloat16",
                num_blocks.saturating_mul(block_size),
                kv_cache_groups.len(),
                full_group.max_admission_blocks
            );
            Some(adapter)
        } else {
            None
        };

        Ok(Self {
            config,
            embed_tokens,
            layers,
            final_norm,
            lm_head,
            embed_weight_t: None,
            ple,
            vision_tower,
            unified_vision_embedder,
            embed_vision,
            embed_audio,
            image_processor,
            tokenizer: None,
            caches: None,
            cached_token_history: Vec::new(),
            cached_image_key: None,
            cached_audio_key: None,
            paged_adapter,
            sliding_prefix_checkpoints: VecDeque::new(),
            sliding_prompt_boundary_checkpoint: None,
            sliding_last_history_checkpoint: None,
            media_session_continuable: false,
            model_id,
        })
    }

    /// Initialize the per-turn KV caches in-place.
    ///
    /// Called on the first turn of a session by the engine's miss-path
    /// `reset_caches(ResetScope::PrefixMiss)` and the vision cores (or
    /// defensively whenever `self.caches` is `None` because a previous
    /// `reset_caches_sync` wiped them). Subsequent turns reuse the
    /// already-populated cache in-place.
    ///
    /// Layer-type routing mirrors the free `init_caches_for_config` used
    /// by `warmup_forward`: global layers get `KVCache`, sliding layers get
    /// `RotatingKVCache` with `config.sliding_window`.
    pub(crate) fn init_caches_sync(&mut self) -> Result<()> {
        let caches = (0..self.config.num_hidden_layers as usize)
            .map(|i| {
                if self.config.is_global_layer(i) {
                    Gemma4LayerCache::new_global()
                } else {
                    Gemma4LayerCache::new_sliding(self.config.sliding_window)
                }
            })
            .collect();
        self.caches = Some(caches);
        self.clear_reuse_state();
        Ok(())
    }

    /// Build the per-layer routing list for the paged dispatch.
    ///
    /// See [`compute_layer_kinds`] (free helper) for full semantics.
    /// This wrapper is the on-`Gemma4Inner` entry point used by the
    /// chat-session forward dispatch.
    pub(crate) fn compute_layer_kinds(&self) -> Result<Vec<Gemma4LayerKind>> {
        compute_layer_kinds_from_kv_cache_specs(&self.config).map_err(|e| {
            Error::from_reason(format!(
                "Gemma4 compute_layer_kinds: failed to derive KV routes: {e}"
            ))
        })
    }

    /// Drop the live KV caches and clear reuse-tracking state.
    ///
    /// `Gemma4LayerCache` has no `reset()` (the inner `KVCache` /
    /// `RotatingKVCache` don't expose one here), so this simply takes the
    /// Vec and lets the next `init_caches_sync` rebuild. Cleared reuse
    /// state ensures a subsequent chat turn can't mistakenly claim a cache
    /// prefix hit against stale history.
    ///
    /// Called by the session API's reset path
    /// (`ChatBackend::reset_caches`) so that a fresh turn starts from an
    /// empty cache. The prefill/decode primitives never call it directly
    /// — they trust their caller's cache-management.
    pub(crate) fn reset_caches_sync(&mut self) -> Result<()> {
        self.caches = None;
        self.clear_reuse_state();
        Ok(())
    }

    /// Clear cached token history and image key. Called from both
    /// `init_caches_sync` and `reset_caches_sync`.
    fn clear_reuse_state(&mut self) {
        self.cached_token_history.clear();
        self.cached_image_key = None;
        self.cached_audio_key = None;
        self.sliding_prefix_checkpoints.clear();
        self.sliding_prompt_boundary_checkpoint = None;
        self.sliding_last_history_checkpoint = None;
        // Covers both reset paths (init_caches_sync + reset_caches_sync): a
        // session that just dropped its media KV can no longer warm-continue.
        self.media_session_continuable = false;
    }

    fn find_gemma4_sliding_history_checkpoint(
        &self,
        tokens: &[u32],
        prefix_len: u32,
    ) -> Result<Option<Vec<Gemma4LayerCache>>> {
        let Some(prefix_tokens) = tokens.get(..prefix_len as usize) else {
            return Ok(None);
        };
        let Some(checkpoint) = self.sliding_last_history_checkpoint.as_ref() else {
            return Ok(None);
        };
        if checkpoint.tokens.as_slice() != prefix_tokens {
            return Ok(None);
        }
        restore_gemma4_sliding_caches(&self.config, &checkpoint.snapshots, prefix_len)
    }

    fn remember_gemma4_sliding_history_checkpoint(
        &mut self,
        history_tokens: &[u32],
    ) -> Result<Gemma4SlidingCheckpointStoreTrace> {
        let trace_enabled = inference_trace_enabled();
        let total_start = trace_enabled.then(std::time::Instant::now);
        let mut trace = Gemma4SlidingCheckpointStoreTrace::default();
        if history_tokens.is_empty() {
            self.sliding_last_history_checkpoint = None;
            return Ok(trace.finish(total_start));
        }

        let expected_offset = history_tokens.len() as u32;
        if !gemma4_sliding_caches_ready_at(&self.config, self.caches.as_deref(), expected_offset)? {
            self.sliding_last_history_checkpoint = None;
            return Ok(trace.finish(total_start));
        }

        let eval_start = trace_enabled.then(std::time::Instant::now);
        eval_gemma4_caches(
            self.caches
                .as_ref()
                .ok_or_else(|| Error::from_reason("Gemma4 sliding checkpoint caches missing"))?,
        )?;
        trace.eval_ms = eval_start.map(elapsed_ms).unwrap_or(0.0);

        let snapshot_start = trace_enabled.then(std::time::Instant::now);
        let Some(snapshots) = snapshot_gemma4_sliding_caches(
            &self.config,
            self.caches
                .as_ref()
                .ok_or_else(|| Error::from_reason("Gemma4 sliding checkpoint caches missing"))?,
            expected_offset,
        )?
        else {
            self.sliding_last_history_checkpoint = None;
            trace.snapshot_ms = snapshot_start.map(elapsed_ms).unwrap_or(0.0);
            return Ok(trace.finish(total_start));
        };
        trace.snapshot_ms = snapshot_start.map(elapsed_ms).unwrap_or(0.0);

        let token_clone_start = trace_enabled.then(std::time::Instant::now);
        let tokens = history_tokens.to_vec();
        trace.token_clone_ms = token_clone_start.map(elapsed_ms).unwrap_or(0.0);

        let update_start = trace_enabled.then(std::time::Instant::now);
        self.sliding_last_history_checkpoint =
            Some(Gemma4SlidingHistoryCheckpoint { tokens, snapshots });
        trace.update_ms = update_start.map(elapsed_ms).unwrap_or(0.0);
        trace.stored = true;
        Ok(trace.finish(total_start))
    }

    fn find_gemma4_sliding_prefix_checkpoint(
        &self,
        tokens: &[u32],
        prefix_len: u32,
        block_size: u32,
        cache_salt: u64,
    ) -> Result<Option<Gemma4SlidingPrefixCheckpointHit>> {
        fn try_restore_checkpoint(
            config: &Gemma4Config,
            checkpoint: &Gemma4SlidingPrefixCheckpoint,
            tokens: &[u32],
            target_prefix_len: u32,
            block_size: u32,
            cache_salt: u64,
        ) -> Result<Option<Gemma4SlidingPrefixCheckpointHit>> {
            if checkpoint.prefix_len > target_prefix_len || checkpoint.block_size != block_size {
                return Ok(None);
            }
            let Some(prefix_tokens) = tokens.get(..checkpoint.prefix_len as usize) else {
                return Ok(None);
            };
            if checkpoint.tokens.as_slice() != prefix_tokens {
                return Ok(None);
            }
            let Some(final_block_hash) = compute_gemma4_paged_prefix_block_hash(
                tokens,
                checkpoint.prefix_len,
                block_size,
                cache_salt,
            ) else {
                return Ok(None);
            };
            if checkpoint.final_block_hash != final_block_hash {
                return Ok(None);
            }
            let Some(caches) = restore_gemma4_sliding_caches(
                config,
                &checkpoint.snapshots,
                checkpoint.prefix_len,
            )?
            else {
                return Ok(None);
            };
            Ok(Some(Gemma4SlidingPrefixCheckpointHit {
                prefix_len: checkpoint.prefix_len,
                caches,
            }))
        }

        let mut best_hit: Option<Gemma4SlidingPrefixCheckpointHit> = None;
        if let Some(checkpoint) = self.sliding_prompt_boundary_checkpoint.as_ref()
            && let Some(hit) = try_restore_checkpoint(
                &self.config,
                checkpoint,
                tokens,
                prefix_len,
                block_size,
                cache_salt,
            )?
        {
            best_hit = Some(hit);
        }

        for checkpoint in self.sliding_prefix_checkpoints.iter().rev() {
            if best_hit
                .as_ref()
                .is_some_and(|hit| hit.prefix_len >= checkpoint.prefix_len)
            {
                continue;
            }
            if let Some(hit) = try_restore_checkpoint(
                &self.config,
                checkpoint,
                tokens,
                prefix_len,
                block_size,
                cache_salt,
            )? {
                if hit.prefix_len == prefix_len {
                    return Ok(Some(hit));
                }
                best_hit = Some(hit);
            }
        }

        Ok(best_hit)
    }

    fn remember_gemma4_sliding_prefix_checkpoint(
        &mut self,
        tokens: &[u32],
        prefix_len: u32,
        block_size: u32,
        cache_salt: u64,
    ) -> Result<Gemma4SlidingCheckpointStoreTrace> {
        let trace_enabled = inference_trace_enabled();
        let total_start = trace_enabled.then(std::time::Instant::now);
        let mut trace = Gemma4SlidingCheckpointStoreTrace::default();
        let Some(final_block_hash) =
            compute_gemma4_paged_prefix_block_hash(tokens, prefix_len, block_size, cache_salt)
        else {
            return Ok(trace.finish(total_start));
        };
        let Some(prefix_tokens) = tokens.get(..prefix_len as usize) else {
            return Ok(trace.finish(total_start));
        };
        if !gemma4_sliding_caches_ready_at(&self.config, self.caches.as_deref(), prefix_len)? {
            return Ok(trace.finish(total_start));
        }

        let eval_start = trace_enabled.then(std::time::Instant::now);
        eval_gemma4_caches(
            self.caches
                .as_ref()
                .ok_or_else(|| Error::from_reason("Gemma4 sliding prefix caches missing"))?,
        )?;
        trace.eval_ms = eval_start.map(elapsed_ms).unwrap_or(0.0);

        let snapshot_start = trace_enabled.then(std::time::Instant::now);
        let Some(snapshots) = snapshot_gemma4_sliding_caches(
            &self.config,
            self.caches
                .as_ref()
                .ok_or_else(|| Error::from_reason("Gemma4 sliding prefix caches missing"))?,
            prefix_len,
        )?
        else {
            trace.snapshot_ms = snapshot_start.map(elapsed_ms).unwrap_or(0.0);
            return Ok(trace.finish(total_start));
        };
        trace.snapshot_ms = snapshot_start.map(elapsed_ms).unwrap_or(0.0);

        let token_clone_start = trace_enabled.then(std::time::Instant::now);
        let prefix_tokens = prefix_tokens.to_vec();
        trace.token_clone_ms = token_clone_start.map(elapsed_ms).unwrap_or(0.0);

        let update_start = trace_enabled.then(std::time::Instant::now);
        self.sliding_prefix_checkpoints.retain(|checkpoint| {
            !(checkpoint.prefix_len == prefix_len
                && checkpoint.block_size == block_size
                && checkpoint.final_block_hash == final_block_hash
                && checkpoint.tokens == prefix_tokens)
        });
        self.sliding_prefix_checkpoints
            .push_back(Gemma4SlidingPrefixCheckpoint {
                prefix_len,
                block_size,
                final_block_hash,
                tokens: prefix_tokens,
                snapshots,
            });
        let checkpoint_limit = gemma4_sliding_prefix_checkpoint_limit(&self.config, block_size);
        trim_gemma4_sliding_prefix_checkpoints(
            &mut self.sliding_prefix_checkpoints,
            checkpoint_limit,
            trace_enabled,
        );
        trace.update_ms = update_start.map(elapsed_ms).unwrap_or(0.0);
        trace.stored = true;
        Ok(trace.finish(total_start))
    }

    fn remember_gemma4_sliding_materialized_prefix_checkpoint(
        &mut self,
        tokens: &[u32],
        prefix_len: u32,
        block_size: u32,
        cache_salt: u64,
    ) -> Result<Gemma4SlidingCheckpointStoreTrace> {
        let trace_enabled = inference_trace_enabled();
        let total_start = trace_enabled.then(std::time::Instant::now);
        let mut trace = Gemma4SlidingCheckpointStoreTrace::default();
        let Some(final_block_hash) =
            compute_gemma4_paged_prefix_block_hash(tokens, prefix_len, block_size, cache_salt)
        else {
            return Ok(trace.finish(total_start));
        };
        let Some(prefix_tokens) = tokens.get(..prefix_len as usize) else {
            return Ok(trace.finish(total_start));
        };
        if !gemma4_sliding_caches_ready_at(&self.config, self.caches.as_deref(), prefix_len)? {
            return Ok(trace.finish(total_start));
        }

        let snapshot_start = trace_enabled.then(std::time::Instant::now);
        let Some(mut snapshots) = snapshot_gemma4_sliding_caches(
            &self.config,
            self.caches
                .as_ref()
                .ok_or_else(|| Error::from_reason("Gemma4 sliding prefix caches missing"))?,
            prefix_len,
        )?
        else {
            trace.snapshot_ms = snapshot_start.map(elapsed_ms).unwrap_or(0.0);
            return Ok(trace.finish(total_start));
        };
        trace.snapshot_ms = snapshot_start.map(elapsed_ms).unwrap_or(0.0);

        let eval_start = trace_enabled.then(std::time::Instant::now);
        materialize_gemma4_sliding_snapshots(&mut snapshots)?;
        trace.eval_ms = eval_start.map(elapsed_ms).unwrap_or(0.0);

        let token_clone_start = trace_enabled.then(std::time::Instant::now);
        let prefix_tokens = prefix_tokens.to_vec();
        trace.token_clone_ms = token_clone_start.map(elapsed_ms).unwrap_or(0.0);

        let update_start = trace_enabled.then(std::time::Instant::now);
        self.sliding_prefix_checkpoints.retain(|checkpoint| {
            !(checkpoint.prefix_len == prefix_len
                && checkpoint.block_size == block_size
                && checkpoint.final_block_hash == final_block_hash
                && checkpoint.tokens == prefix_tokens)
        });
        self.sliding_prefix_checkpoints
            .push_back(Gemma4SlidingPrefixCheckpoint {
                prefix_len,
                block_size,
                final_block_hash,
                tokens: prefix_tokens,
                snapshots,
            });
        let checkpoint_limit = gemma4_sliding_prefix_checkpoint_limit(&self.config, block_size);
        trim_gemma4_sliding_prefix_checkpoints(
            &mut self.sliding_prefix_checkpoints,
            checkpoint_limit,
            trace_enabled,
        );
        trace.update_ms = update_start.map(elapsed_ms).unwrap_or(0.0);
        trace.stored = true;
        Ok(trace.finish(total_start))
    }

    fn remember_gemma4_sliding_materialized_prompt_boundary_checkpoint(
        &mut self,
        tokens: &[u32],
        prefix_len: u32,
        block_size: u32,
        cache_salt: u64,
    ) -> Result<Gemma4SlidingCheckpointStoreTrace> {
        let trace_enabled = inference_trace_enabled();
        let total_start = trace_enabled.then(std::time::Instant::now);
        let mut trace = Gemma4SlidingCheckpointStoreTrace::default();
        let Some(final_block_hash) =
            compute_gemma4_paged_prefix_block_hash(tokens, prefix_len, block_size, cache_salt)
        else {
            self.sliding_prompt_boundary_checkpoint = None;
            return Ok(trace.finish(total_start));
        };
        let Some(prefix_tokens) = tokens.get(..prefix_len as usize) else {
            self.sliding_prompt_boundary_checkpoint = None;
            return Ok(trace.finish(total_start));
        };
        if !gemma4_sliding_caches_ready_at(&self.config, self.caches.as_deref(), prefix_len)? {
            self.sliding_prompt_boundary_checkpoint = None;
            return Ok(trace.finish(total_start));
        }

        let snapshot_start = trace_enabled.then(std::time::Instant::now);
        let Some(mut snapshots) = snapshot_gemma4_sliding_caches(
            &self.config,
            self.caches
                .as_ref()
                .ok_or_else(|| Error::from_reason("Gemma4 sliding prefix caches missing"))?,
            prefix_len,
        )?
        else {
            self.sliding_prompt_boundary_checkpoint = None;
            trace.snapshot_ms = snapshot_start.map(elapsed_ms).unwrap_or(0.0);
            return Ok(trace.finish(total_start));
        };
        trace.snapshot_ms = snapshot_start.map(elapsed_ms).unwrap_or(0.0);

        let eval_start = trace_enabled.then(std::time::Instant::now);
        materialize_gemma4_sliding_snapshots(&mut snapshots)?;
        trace.eval_ms = eval_start.map(elapsed_ms).unwrap_or(0.0);

        let token_clone_start = trace_enabled.then(std::time::Instant::now);
        let prefix_tokens = prefix_tokens.to_vec();
        trace.token_clone_ms = token_clone_start.map(elapsed_ms).unwrap_or(0.0);

        let update_start = trace_enabled.then(std::time::Instant::now);
        self.sliding_prompt_boundary_checkpoint = Some(Gemma4SlidingPrefixCheckpoint {
            prefix_len,
            block_size,
            final_block_hash,
            tokens: prefix_tokens,
            snapshots,
        });
        trace.update_ms = update_start.map(elapsed_ms).unwrap_or(0.0);
        trace.stored = true;
        Ok(trace.finish(total_start))
    }

    fn maybe_remember_gemma4_sliding_decode_boundary_checkpoint(
        &mut self,
        trace_label: &str,
        trace_enabled: bool,
    ) -> Result<()> {
        let Some(adapter) = self.paged_adapter.as_ref() else {
            return Ok(());
        };
        let block_size = adapter.block_size();
        let prefix_len = adapter.current_token_count();
        let checkpoint_interval =
            gemma4_sliding_decode_checkpoint_interval(&self.config, block_size);
        if prefix_len == 0
            || checkpoint_interval == 0
            || !prefix_len.is_multiple_of(checkpoint_interval)
        {
            return Ok(());
        }
        let request_tokens = adapter.request_tokens().to_vec();

        let store_trace = self.remember_gemma4_sliding_materialized_prefix_checkpoint(
            &request_tokens,
            prefix_len,
            block_size,
            0,
        )?;
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] gemma4 {trace_label}_sliding_block_checkpoint boundary_tokens={} block_size={} checkpoint_interval={} stored={} materialize_ms={:.1} snapshot_ms={:.1} token_clone_ms={:.1} update_ms={:.1} total_ms={:.1}",
                prefix_len,
                block_size,
                checkpoint_interval,
                store_trace.stored,
                store_trace.eval_ms,
                store_trace.snapshot_ms,
                store_trace.token_clone_ms,
                store_trace.update_ms,
                store_trace.total_ms
            ));
        }
        Ok(())
    }

    fn maybe_remember_gemma4_sliding_prompt_boundary_checkpoint(
        &mut self,
        trace_label: &str,
        tokens: &[u32],
        boundary_len: u32,
        trace_enabled: bool,
    ) -> Result<()> {
        let Some(adapter) = self.paged_adapter.as_ref() else {
            return Ok(());
        };
        let block_size = adapter.block_size();
        if boundary_len == 0 || block_size == 0 || !boundary_len.is_multiple_of(block_size) {
            return Ok(());
        }

        let store_trace = self.remember_gemma4_sliding_materialized_prompt_boundary_checkpoint(
            tokens,
            boundary_len,
            block_size,
            0,
        )?;
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] gemma4 {trace_label}_sliding_prompt_checkpoint boundary_tokens={} block_size={} stored={} materialize_ms={:.1} snapshot_ms={:.1} token_clone_ms={:.1} update_ms={:.1} total_ms={:.1}",
                boundary_len,
                block_size,
                store_trace.stored,
                store_trace.eval_ms,
                store_trace.snapshot_ms,
                store_trace.token_clone_ms,
                store_trace.update_ms,
                store_trace.total_ms
            ));
        }
        Ok(())
    }

    fn prepare_gemma4_sliding_prefix_state(
        &mut self,
        tokens: &[u32],
        cached_prefix_len: u32,
        continued_live_prefix: bool,
    ) -> Result<Gemma4SlidingPrefixPreparation> {
        let trace_enabled = inference_trace_enabled();
        let prepare_start = trace_enabled.then(std::time::Instant::now);

        if cached_prefix_len == 0 {
            self.caches = Some(init_caches_for_config(&self.config));
            if trace_enabled {
                write_inference_trace(format_args!(
                    "[MLX_TRACE] gemma4 sliding_prefix_prepare_done state=fresh cached_prefix_tokens=0 elapsed_ms={:.1}",
                    prepare_start.map(elapsed_ms).unwrap_or(0.0)
                ));
            }
            return Ok(Gemma4SlidingPrefixPreparation {
                state: "fresh",
                primed_prefix_len: 0,
            });
        }

        if continued_live_prefix
            && gemma4_sliding_caches_ready_at(
                &self.config,
                self.caches.as_deref(),
                cached_prefix_len,
            )?
        {
            if trace_enabled {
                write_inference_trace(format_args!(
                    "[MLX_TRACE] gemma4 sliding_prefix_prepare_done state=live cached_prefix_tokens={} elapsed_ms={:.1}",
                    cached_prefix_len,
                    prepare_start.map(elapsed_ms).unwrap_or(0.0)
                ));
            }
            return Ok(Gemma4SlidingPrefixPreparation {
                state: "live",
                primed_prefix_len: cached_prefix_len,
            });
        }

        let matches_live_history = self.cached_token_history.len() == cached_prefix_len as usize
            && tokens.starts_with(&self.cached_token_history);
        if matches_live_history
            && gemma4_sliding_caches_ready_at(
                &self.config,
                self.caches.as_deref(),
                cached_prefix_len,
            )?
        {
            if trace_enabled {
                write_inference_trace(format_args!(
                    "[MLX_TRACE] gemma4 sliding_prefix_prepare_done state=last_history cached_prefix_tokens={} elapsed_ms={:.1}",
                    cached_prefix_len,
                    prepare_start.map(elapsed_ms).unwrap_or(0.0)
                ));
            }
            return Ok(Gemma4SlidingPrefixPreparation {
                state: "last_history",
                primed_prefix_len: cached_prefix_len,
            });
        }

        let history_lookup_start = trace_enabled.then(std::time::Instant::now);
        if let Some(caches) =
            self.find_gemma4_sliding_history_checkpoint(tokens, cached_prefix_len)?
        {
            self.caches = Some(caches);
            if trace_enabled {
                write_inference_trace(format_args!(
                    "[MLX_TRACE] gemma4 sliding_prefix_prepare_done state=last_history_checkpoint cached_prefix_tokens={} history_lookup_ms={:.1} elapsed_ms={:.1}",
                    cached_prefix_len,
                    history_lookup_start.map(elapsed_ms).unwrap_or(0.0),
                    prepare_start.map(elapsed_ms).unwrap_or(0.0)
                ));
            }
            return Ok(Gemma4SlidingPrefixPreparation {
                state: "last_history_checkpoint",
                primed_prefix_len: cached_prefix_len,
            });
        }

        let block_size = self
            .paged_adapter
            .as_ref()
            .map(|adapter| adapter.block_size())
            .unwrap_or(0);
        let prefix_lookup_start = trace_enabled.then(std::time::Instant::now);
        if let Some(hit) =
            self.find_gemma4_sliding_prefix_checkpoint(tokens, cached_prefix_len, block_size, 0)?
        {
            let hit_prefix_len = hit.prefix_len;
            self.caches = Some(hit.caches);
            let state = if hit_prefix_len == cached_prefix_len {
                "prefix_checkpoint"
            } else {
                "partial_prefix_checkpoint"
            };
            if trace_enabled {
                write_inference_trace(format_args!(
                    "[MLX_TRACE] gemma4 sliding_prefix_prepare_done state={} cached_prefix_tokens={} primed_prefix_tokens={} replay_delta_tokens={} prefix_lookup_ms={:.1} elapsed_ms={:.1}",
                    state,
                    cached_prefix_len,
                    hit_prefix_len,
                    cached_prefix_len.saturating_sub(hit_prefix_len),
                    prefix_lookup_start.map(elapsed_ms).unwrap_or(0.0),
                    prepare_start.map(elapsed_ms).unwrap_or(0.0)
                ));
            }
            return Ok(Gemma4SlidingPrefixPreparation {
                state,
                primed_prefix_len: hit_prefix_len,
            });
        }

        self.caches = Some(init_caches_for_config(&self.config));
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] gemma4 sliding_prefix_prepare_done state=replay cached_prefix_tokens={} history_lookup_ms={:.1} prefix_lookup_ms={:.1} elapsed_ms={:.1}",
                cached_prefix_len,
                history_lookup_start.map(elapsed_ms).unwrap_or(0.0),
                prefix_lookup_start.map(elapsed_ms).unwrap_or(0.0),
                prepare_start.map(elapsed_ms).unwrap_or(0.0)
            ));
        }
        Ok(Gemma4SlidingPrefixPreparation {
            state: "replay",
            primed_prefix_len: 0,
        })
    }

    pub(crate) fn set_tokenizer(&mut self, tokenizer: Arc<Qwen3Tokenizer>) {
        self.tokenizer = Some(tokenizer);
    }

    /// Decode + resize + patch raw image bytes and expand the rendered
    /// prompt's per-image `<|image|>` placeholders.
    ///
    /// The engine session core owns message-side image extraction
    /// (`engine::session::extract_images_from_messages`) and prompt
    /// rendering; the raw bytes arrive via [`WholeTurnArgs::images`].
    /// The "no vision support" rejection surfaces from INSIDE the vision
    /// turn (after render).
    fn prepare_vision_tokens(
        &self,
        rendered_tokens: &[u32],
        raw_images: &[Vec<u8>],
    ) -> Result<(Vec<u32>, Vec<ProcessedGemma4Image>, Option<u64>)> {
        let ip = self.image_processor.as_ref().ok_or_else(|| {
            Error::from_reason(
                "Images provided but model has no vision support (no vision_config in config.json)",
            )
        })?;
        let mut processed_images = Vec::with_capacity(raw_images.len());
        for bytes in raw_images {
            processed_images.push(ip.process_bytes(bytes)?);
        }

        // Compute the image cache key BEFORE the prefill so it can be
        // recorded on `self.cached_image_key` after the decode loop.
        // Session callers inspect this field to decide whether a
        // session-continue delta is allowed (text-only) or requires
        // a fresh `chat_session_start`.
        let new_image_key = Some(engine::compute_image_cache_key(raw_images));

        // Expand image tokens. Gemma4 uses: <|image>  (BOI) +
        // <|image|> × num_soft_tokens + <image|> (EOI). The chat template
        // inserts a single <|image|> per image; we expand it here.
        let image_token_id = self.config.image_token_id.unwrap_or(258880) as u32;
        let boi_token_id = self.config.boi_token_id.unwrap_or(255999) as u32;
        let eoi_token_id = self.config.eoi_token_id.unwrap_or(258882) as u32;
        let expanded = expand_image_tokens(
            rendered_tokens,
            &processed_images,
            image_token_id,
            boi_token_id,
            eoi_token_id,
        );

        Ok((expanded, processed_images, new_image_key))
    }

    /// Decode raw (encoded) audio bytes and expand the rendered prompt's
    /// per-clip `<|audio|>` placeholders into `boa + audio×n_frames + eoa`
    /// spans. The audio counterpart of [`Self::prepare_vision_tokens`].
    ///
    /// Each clip is decoded (`decode_wav_to_pcm`) into a mono 16 kHz f32
    /// waveform and framed (`frames_from_pcm`) into `[n_frames, 640]` raw
    /// windows; the per-clip frame counts drive `expand_audio_tokens`. All
    /// clips' frames are concatenated (axis 0) into a single
    /// `[total_frames, 640]` tensor so the merge scatter feeds them in order.
    /// `tokens` is the (possibly image-expanded) token stream; the audio
    /// expansion runs on top of it, leaving image spans untouched.
    fn prepare_audio_tokens(
        &self,
        tokens: &[u32],
        raw_audio: &[Vec<u8>],
    ) -> Result<(Vec<u32>, MxArray, Option<u64>)> {
        let spt = self.config.audio_samples_per_token.unwrap_or(640) as usize;
        let audio_token_id = self.config.audio_token_id.unwrap_or(258881) as u32;
        let boa_token_id = self.config.boa_token_id.unwrap_or(256000) as u32;
        let eoa_token_id = self.config.eoa_token_id.unwrap_or(258883) as u32;

        let mut per_clip_frames: Vec<MxArray> = Vec::with_capacity(raw_audio.len());
        let mut n_frames_per_clip: Vec<usize> = Vec::with_capacity(raw_audio.len());
        for bytes in raw_audio {
            let pcm = super::audio_processor::decode_wav_to_pcm(bytes)?;
            let frames = super::audio_processor::frames_from_pcm(&pcm, spt)?;
            let n = frames.shape_at(0)? as usize;
            n_frames_per_clip.push(n);
            per_clip_frames.push(frames);
        }

        let audio_frames = if per_clip_frames.len() == 1 {
            per_clip_frames.remove(0)
        } else {
            let refs: Vec<&MxArray> = per_clip_frames.iter().collect();
            MxArray::concatenate_many(refs, Some(0))?
        };

        let expanded = super::audio_processor::expand_audio_tokens(
            tokens,
            &n_frames_per_clip,
            audio_token_id,
            boa_token_id,
            eoa_token_id,
        )?;

        // Audio uses the same byte-identity cache key as images so an
        // audio-change cold-restarts the session server-side.
        let new_audio_key = Some(engine::compute_image_cache_key(raw_audio));

        Ok((expanded, audio_frames, new_audio_key))
    }

    /// Build the merged multimodal+text input embeddings for a prefill.
    ///
    /// Scatters image features (`@image_token_id`) AND audio features
    /// (`@audio_token_id`) into the SAME `sqrt(hidden)`-scaled text stream
    /// via chained `masked_scatter`s. Image-only turns skip the audio scatter
    /// (the image scatter math matches the prior vision-only prefill exactly);
    /// audio-only turns skip the image scatter. Returns `None` only when
    /// neither modality contributes features (text-only fallback).
    fn build_gemma4_multimodal_embeds(
        &self,
        prompt: &MxArray,
        processed_images: &[ProcessedGemma4Image],
        audio_frames: Option<&MxArray>,
    ) -> Result<Option<MxArray>> {
        let has_image_features = !processed_images.is_empty() && self.embed_vision.is_some();
        let has_audio_features = audio_frames.is_some() && self.embed_audio.is_some();
        if !has_image_features && !has_audio_features {
            return Ok(None);
        }

        // Base scaled text stream (built once; both scatters write into it).
        let text_embeds = self.embed_tokens.forward(prompt)?;
        let mut merged = text_embeds.mul_scalar((self.config.hidden_size as f64).sqrt())?;
        let embed_dtype = merged.dtype()?;

        // Image scatter @ image_token_id.
        if has_image_features {
            let ev = self.embed_vision.as_ref().unwrap();
            let image_token_id = self.config.image_token_id.unwrap_or(258880);
            let mut all_features: Vec<MxArray> = Vec::new();
            for proc in processed_images {
                let features = if let Some(vt) = self.vision_tower.as_ref() {
                    vt.forward(&proc.pixel_values)?
                } else if let Some(ve) = self.unified_vision_embedder.as_ref() {
                    let positions = proc.position_ids.as_ref().ok_or_else(|| {
                        Error::from_reason(
                            "Unified vision embedder requires per-patch position ids, but none \
                             were produced by the image processor.",
                        )
                    })?;
                    ve.forward(&proc.pixel_values, positions)?.expand_dims(0)?
                } else {
                    return Err(Error::from_reason(
                        "Image features requested but no vision tower / unified embedder present",
                    ));
                };
                all_features.push(ev.forward(&features)?);
            }
            let image_features = if all_features.len() == 1 {
                all_features.remove(0)
            } else {
                let refs: Vec<&MxArray> = all_features.iter().collect();
                MxArray::concatenate_many(refs, Some(1))?
            };
            let image_features = image_features.astype(embed_dtype)?;

            let image_token = MxArray::scalar_int(image_token_id)?;
            let image_mask = prompt.equal(&image_token)?;
            let mask_count_arr = image_mask.astype(DType::Int32)?.sum(None, None)?;
            mask_count_arr.eval();
            let mask_count = mask_count_arr.item_at_int32(0)? as i64;
            let feature_count = image_features.shape_at(1)?;
            if mask_count != feature_count {
                return Err(Error::new(
                    Status::GenericFailure,
                    format!(
                        "Image token count ({mask_count}) does not match vision feature count ({feature_count}). \
                         Check that image token expansion produced the correct number of tokens."
                    ),
                ));
            }
            let image_mask_expanded = image_mask.expand_dims(-1)?.broadcast_to(&merged.shape()?)?;
            merged = masked_scatter(&merged, &image_mask_expanded, &image_features)?;
        }

        // Audio scatter @ audio_token_id (CAUSAL; audio features unscaled).
        if has_audio_features {
            let ea = self.embed_audio.as_ref().unwrap();
            let audio_token_id = self.config.audio_token_id.unwrap_or(258881);
            let audio_features = ea.forward(audio_frames.unwrap())?.astype(embed_dtype)?;

            let audio_token = MxArray::scalar_int(audio_token_id)?;
            let audio_mask = prompt.equal(&audio_token)?;
            let mask_count_arr = audio_mask.astype(DType::Int32)?.sum(None, None)?;
            mask_count_arr.eval();
            let mask_count = mask_count_arr.item_at_int32(0)? as i64;
            let feature_count = audio_features.shape_at(0)?;
            if mask_count != feature_count {
                return Err(Error::new(
                    Status::GenericFailure,
                    format!(
                        "Audio token count ({mask_count}) does not match audio frame count ({feature_count}). \
                         Check that audio token expansion produced the correct number of frames."
                    ),
                ));
            }
            // Zero-frame audio has no scatter targets; leave the stream as-is
            // (a `masked_scatter` over an empty source would divide by zero).
            if feature_count > 0 {
                let audio_mask_expanded =
                    audio_mask.expand_dims(-1)?.broadcast_to(&merged.shape()?)?;
                merged = masked_scatter(&merged, &audio_mask_expanded, &audio_features)?;
            }
        }

        Ok(Some(merged))
    }

    /// Prepare the merged multimodal prompt for a paged prefill: expand audio
    /// placeholders (when audio present) then image placeholders (when images
    /// present) on the rendered token stream, and decode/frame the audio.
    ///
    /// Audio expansion runs FIRST so that on the manual no-placeholder fallback
    /// (tokenizer without a chat template — neither `<|image|>` nor `<|audio|>`
    /// is emitted) each modality's span is inserted right after BOS, and the
    /// expansion that runs LAST lands first. Running image expansion last keeps
    /// the serializer's canonical `BOS -> image -> audio -> text` order. On the
    /// chat-template path each expansion replaces only its own placeholder id in
    /// place, so content order is preserved regardless of which runs first.
    ///
    /// Returns `(tokens, processed_images, audio_frames, new_image_key,
    /// new_audio_key)`. Image-only turns never touch the audio path and leave
    /// `audio_frames`/`new_audio_key` as `None` (byte-identical to the old
    /// vision-only flow); audio-only turns never run the image processor and
    /// leave `processed_images` empty + `new_image_key` `None`.
    #[allow(clippy::type_complexity)]
    fn prepare_multimodal_tokens(
        &self,
        rendered_tokens: &[u32],
        raw_images: &[Vec<u8>],
        raw_audio: &[Vec<u8>],
    ) -> Result<(
        Vec<u32>,
        Vec<ProcessedGemma4Image>,
        Option<MxArray>,
        Option<u64>,
        Option<u64>,
    )> {
        // Audio expansion first (only when audio present — keeps image-only
        // turns off the audio path and leaves `new_audio_key` None). On the
        // no-placeholder fallback each modality's span is inserted right after
        // BOS, so whichever expansion runs LAST lands first; running image last
        // (below) yields the canonical BOS -> image -> audio -> text order.
        let mut audio_frames: Option<MxArray> = None;
        let mut new_audio_key: Option<u64> = None;
        let tokens_after_audio = if raw_audio.is_empty() {
            rendered_tokens.to_vec()
        } else {
            let (expanded, frames, audio_key) =
                self.prepare_audio_tokens(rendered_tokens, raw_audio)?;
            audio_frames = Some(frames);
            new_audio_key = audio_key;
            expanded
        };

        // Image expansion on top of the (possibly audio-expanded) stream — runs
        // LAST so its spans precede the audio spans on the fallback path. Image
        // expansion only touches `<|image|>` ids, so the audio spans are inert
        // to it on the chat-template path.
        let (tokens, processed_images, new_image_key) = if raw_images.is_empty() {
            (tokens_after_audio, Vec::new(), None)
        } else {
            self.prepare_vision_tokens(&tokens_after_audio, raw_images)?
        };

        Ok((
            tokens,
            processed_images,
            audio_frames,
            new_image_key,
            new_audio_key,
        ))
    }

    /// Terminal media-state finalize shared by both vision cores (sync +
    /// stream), so the two stay byte-identical. Resolves the session into
    /// exactly ONE of two states, never partial:
    ///
    /// - **Continuable** (when `media_continuable` — i.e. ANY image or audio
    ///   turn, including the unified bidirectional-vision image — AND
    ///   `reuse_cache`, AND `finalize_turn_keep_live_per_block` succeeds, AND the
    ///   sliding-history checkpoint actually `stored`, AND the adapter is
    ///   `live_for_continue`): the global paged KV is kept live (full blocks
    ///   registered for content-addressed reuse) and the marker is set so
    ///   `text_delta_image_guard` lets the next text delta through. On that delta
    ///   the global prefix is reused IN-PLACE (`continue_turn` keeps the block
    ///   table, `cachedTokens > 0`, only the new suffix is forwarded — it is NOT
    ///   re-walked) and the sliding caches resolve to `state="live"`
    ///   (`continued_live_prefix && gemma4_sliding_caches_ready_at`), so
    ///   `run_sliding_only_prefill` is skipped and no media position is ever
    ///   re-embedded from a raw `<|image|>`/`<|audio|>` id. Mirrors the
    ///   qwen3_5_moe two-state finalize.
    /// - **Non-continuable** (`reuse_cache=false`, a keep-live failure, or the
    ///   sliding checkpoint did not store / the adapter is not
    ///   `live_for_continue`): `release_request` only, keep history + media keys
    ///   live so the guard is reachable and REJECTS (marker stays false) and the
    ///   follow-up text delta cold-restarts. The vision core does NOT
    ///   `reset_caches_sync` here, unlike the text/MoE path.
    ///
    /// ## Why `stored && live_for_continue` is the faithfulness gate
    /// `gemma4_sliding_caches_ready_at` requires EVERY `is_sliding_layer` flat
    /// cache populated. On KV-shared checkpoints (e2b: `SharedOnSliding` layers
    /// physically store no flat K/V — they read the anchor's), that is never
    /// satisfiable, so the checkpoint is a structural no-op (`stored == false`).
    /// A warm media→text continue is only numerically faithful when the media
    /// positions' sliding K/V can be reused IN PLACE: a text token's true
    /// embedding IS `embed_tokens.forward(id)` (replay-safe), but a media
    /// position's is a scattered SigLIP/audio feature that replay CANNOT rebuild
    /// from the raw special-token id. So the marker is armed ONLY when
    /// `stored && live_for_continue`: non-KV-shared checkpoints (12B audio AND
    /// unified image, `num_kv_shared_layers=0`) store real K/V and warm-continue
    /// via `state="live"`; KV-shared checkpoints (e2b) store nothing, leave the
    /// marker off, and cleanly cold-restart.
    ///
    /// ## R1 sliding-offset reconciliation (the length-finish materialize)
    /// The vision decode loop never forwards the final sampled token, so after
    /// the loop the live (non-shared) sliding caches AND the global paged KV sit
    /// at offset `prefill_len + G - 1`. The drop-last history rule yields
    /// `cached_token_history.len() == prefill_len + G - 1` on
    /// stop/repetition/cancelled (offsets MATCH) but `prefill_len + G` on a
    /// `"length"` finish (one short). On the continuable+`"length"` path we
    /// forward that final token once via `run_paged_decode_step` — exactly what
    /// the text path's `materialize_final` does (`paged_turn.rs` length gate →
    /// `Gemma4PagedDecode::materialize_final` → `run_paged_decode_step`) —
    /// advancing both caches to `prefill_len + G` so the kept-live global KV
    /// content-addresses against the saved history for the next delta's live
    /// restore. (Verified byte-exact by the non-unified-image warm==cold golden.)
    #[allow(clippy::too_many_arguments)]
    fn finalize_vision_turn_media_state(
        &mut self,
        expanded_tokens: &[u32],
        generated_tokens: &[u32],
        finish_reason: &str,
        new_image_key: Option<u64>,
        new_audio_key: Option<u64>,
        media_continuable: bool,
        reuse_cache: bool,
    ) -> Result<()> {
        let continuable_eligible = reuse_cache && media_continuable;
        let is_length = finish_reason == "length";

        // Drop-last history (mirrors the non-continuable save the vision cores
        // do today and the text path's `save_paged_history`): keep all tokens on
        // a `"length"` finish, otherwise drop the terminal token.
        let history_tokens: &[u32] = if !is_length && !generated_tokens.is_empty() {
            &generated_tokens[..generated_tokens.len() - 1]
        } else {
            generated_tokens
        };
        let mut full_history = Vec::with_capacity(expanded_tokens.len() + history_tokens.len());
        full_history.extend_from_slice(expanded_tokens);
        full_history.extend_from_slice(history_tokens);

        if continuable_eligible {
            // R1: align the sliding caches with the keep-all history before any
            // checkpoint. On a `"length"` finish the loop left the final token
            // unforwarded (offset == history.len() - 1); forward it now so both
            // the global paged KV and the sliding caches reach history.len().
            if is_length && let Some(&last_token) = generated_tokens.last() {
                // Forwards the token through the paged adapter + sliding caches.
                // A failure here aborts the turn before any state is published
                // (the request is still live; the caller's Err path releases it).
                let _logits = self.run_paged_decode_step(last_token)?;
            }

            let (keep_live_ok, live_for_continue) = match self.paged_adapter.as_mut() {
                Some(adapter) => {
                    let total = adapter.request_tokens().len();
                    let bs = adapter.block_size();
                    let extra = engine::build_paged_extra_keys(total, bs, &[]);
                    let ok = adapter.finalize_turn_keep_live_per_block(&extra, 0).is_ok();
                    (ok, adapter.is_live_for_continue())
                }
                None => (false, false),
            };

            if keep_live_ok {
                // Publish history FIRST: the checkpoint reads its length, and
                // the next delta's prefix restore matches against it.
                self.cached_token_history = full_history;
                self.cached_image_key = new_image_key;
                self.cached_audio_key = new_audio_key;
                let history_for_ckpt = self.cached_token_history.clone();
                let stored = self
                    .remember_gemma4_sliding_history_checkpoint(&history_for_ckpt)
                    .map(|trace| trace.stored)
                    .unwrap_or(false);
                // Warm continuation is only faithful when the sliding state is
                // restorable from a stored checkpoint (or the in-place live
                // caches it implies). A text position's true embedding IS
                // `embed_tokens.forward(id)`, so REPLAY rebuilds it exactly. A
                // MEDIA position's true embedding is a scattered SigLIP/audio
                // feature that replay cannot reconstruct from the raw
                // `<|image|>`/`<|audio|>` special-token id. On KV-shared
                // checkpoints (e2b) the shared-on-sliding layers hold no flat
                // K/V, so `stored == false` and the next delta would rebuild
                // media-position sliding K/V from raw token embeddings —
                // numerically wrong. Downgrade to a clean non-continuable state
                // so the follow-up delta cold-restarts instead.
                //
                // `live_for_continue` guards a second gap: a keep-live with zero
                // FULL blocks (a media turn shorter than `block_size`) returns
                // Ok without registering the request, so the next delta could
                // not take the live-continue path and would re-prefill the media
                // placeholders as text. Unreachable on shipped configs (media
                // turns far exceed the 16-token block), but cheap to gate.
                if stored && live_for_continue {
                    self.media_session_continuable = true;
                    return Ok(());
                }
                if let Some(adapter) = self.paged_adapter.as_mut() {
                    let _ = adapter.release_request();
                }
                self.media_session_continuable = false;
                return Ok(());
            }
            // keep-live failed: fall through to the non-continuable teardown.
        }

        // Non-continuable: release the global KV but keep history + media keys so
        // a follow-up text delta reaches `text_delta_image_guard`, which rejects
        // it (marker is false). Matches the vision core's prior behavior.
        if let Some(adapter) = self.paged_adapter.as_mut() {
            let _ = adapter.release_request();
        }
        self.cached_token_history = full_history;
        self.cached_image_key = new_image_key;
        self.cached_audio_key = new_audio_key;
        self.media_session_continuable = false;
        Ok(())
    }

    /// Whether this expanded prompt is a media turn eligible to warm-continue a
    /// follow-up text delta. Eligibility is broad: ANY image or audio turn,
    /// including the unified bidirectional-vision image. Faithfulness is NOT
    /// decided here — the `stored && live_for_continue` gate in
    /// `finalize_vision_turn_media_state` is the real safety net: a non-KV-shared
    /// checkpoint (12B, `num_kv_shared_layers=0`) physically stores sliding K/V
    /// so the delta hits `state="live"` and warm-continues; a KV-shared
    /// checkpoint (e2b) stores nothing so the delta cold-restarts. A warm text
    /// delta routes through the generic causal text path (no overlay), so it is
    /// numerically faithful regardless of how the media turn built its K/V.
    fn gemma4_media_continuable(&self, expanded_tokens: &[u32]) -> bool {
        let image_token_id = self.config.image_token_id.unwrap_or(258880) as u32;
        let audio_token_id = self.config.audio_token_id.unwrap_or(258881) as u32;
        let has_image = expanded_tokens.contains(&image_token_id);
        let has_audio = expanded_tokens.contains(&audio_token_id);
        has_audio || has_image
    }

    /// Vision (VLM) whole-turn core over the BLOCK-PAGED backend,
    /// non-streaming.
    ///
    /// Shared multimodal prep (`prepare_multimodal_tokens` to expand
    /// `<|image|>` / `<|audio|>` placeholders, `build_gemma4_multimodal_embeds`
    /// to `masked_scatter` image+audio features into the residual) writes
    /// full-attention K/V into the paged adapter pool. Sliding layers still use
    /// the flat rotating caches.
    ///
    /// Single-image-turn-only and cold-start by construction: the adapter is
    /// reset with `max_cache_hit_tokens = 0` and the sliding caches are rebuilt
    /// fresh, so `cached_prefix_len == 0` and there is no warm-continue. The
    /// request is released on BOTH success and error; `cached_tokens` is 0.
    fn vision_paged_turn_sync_core(
        &mut self,
        rendered_tokens: &[u32],
        raw_images: &[Vec<u8>],
        raw_audio: &[Vec<u8>],
        tokenizer: &Arc<Qwen3Tokenizer>,
        config: &ChatConfig,
        eos_token_id: u32,
    ) -> Result<ChatResult> {
        let max_new_tokens = config.max_new_tokens.unwrap_or(2048);
        let (tokens, processed_images, audio_frames, new_image_key, new_audio_key) =
            self.prepare_multimodal_tokens(rendered_tokens, raw_images, raw_audio)?;
        if tokens.is_empty() {
            return Err(Error::from_reason("Empty prompt"));
        }
        let sampling_config = make_sampling_config(config, &self.config);
        let repetition_cutoff = repetition_cutoff_from_config(config);
        let eos_ids = self.config.eos_token_ids.clone();

        let prefill_slice: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let prefill_len = prefill_slice.len();
        let prompt = MxArray::from_int32(&prefill_slice, &[1, prefill_len as i64])?;
        let prompt_token_count = tokens.len();

        let generation_stream = Stream::new(DeviceType::Gpu);
        let _wired_ctx = crate::stream::WiredLimitContext::new(usize::MAX, vec![generation_stream]);

        let generation_start = std::time::Instant::now();

        let vision_embeds =
            self.build_gemma4_multimodal_embeds(&prompt, &processed_images, audio_frames.as_ref())?;

        // Derive layer kinds before acquiring the paged request. This reads
        // only `self.config` (no adapter/cache dependency) and is fallible, so
        // running it here keeps the only fallible op ahead of the request
        // acquisition — an early Err can never leak a prepared request.
        let layer_kinds = self.compute_layer_kinds()?;

        // Cold-start the paged adapter on the expanded sequence.
        let seq_id: u32 = 0;
        let total_budget = tokens.len() as u32;
        // A new media set is non-continuable until its own finalize re-arms the
        // marker. Reset BEFORE the side-effecting prepare below, which releases
        // any prior kept-live request and can then fail (block exhaustion) via
        // `?` — a stale `true` would wrongly admit a later text delta.
        self.media_session_continuable = false;
        {
            let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                Error::from_reason("vision_paged_turn_sync_core: paged_adapter is None")
            })?;
            adapter
                .prepare_turn_with_max_cache_hit_tokens(
                    seq_id,
                    &tokens,
                    total_budget,
                    /* reuse_cache */ false,
                    &[],
                    /* cache_salt */ 0,
                    /* skip_lookup */ true,
                    /* max_cache_hit_tokens */ 0,
                )
                .map_err(Error::from_reason)?;
        }
        // Fresh sliding flat caches + clear all reuse/checkpoint state so the
        // cold prefill starts from an empty context.
        self.caches = Some(init_caches_for_config(&self.config));
        self.cached_token_history.clear();
        self.cached_image_key = None;
        self.cached_audio_key = None;
        self.sliding_prefix_checkpoints.clear();
        self.sliding_prompt_boundary_checkpoint = None;
        self.sliding_last_history_checkpoint = None;

        let forward_result = (|| -> Result<(Vec<u32>, String)> {
            let last_logits = {
                let _stream_ctx = StreamContext::new(generation_stream);
                crate::models::gemma4::diagnostic::set_step(-1);
                match vision_embeds {
                    Some(ref embeds) => {
                        self.run_paged_vlm_prefill(&tokens, embeds, &layer_kinds)?
                    }
                    None => {
                        // Text-only fallback (checkpoint lacks the vision
                        // tower): drive the same paged prefill seeded from
                        // token embeddings.
                        let text_embeds = self.embed_tokens.forward(&prompt)?;
                        let text_embeds =
                            text_embeds.mul_scalar((self.config.hidden_size as f64).sqrt())?;
                        self.run_paged_vlm_prefill(&tokens, &text_embeds, &layer_kinds)?
                    }
                }
            };

            crate::array::synchronize_and_clear_cache();

            let mut y = sample_next_token(&last_logits, sampling_config)?;
            y.eval();

            let mut generated_tokens: Vec<u32> = Vec::new();
            let mut finish_reason = String::from("length");

            for step in 0..max_new_tokens {
                let token_id = y.item_at_int32(0)? as u32;
                generated_tokens.push(token_id);

                if is_eos_token(token_id, &eos_ids, eos_token_id) {
                    finish_reason = String::from("stop");
                    break;
                }
                if let Some(reason) =
                    check_gemma4_repetition_cutoff(&generated_tokens, repetition_cutoff)
                {
                    finish_reason = reason.to_string();
                    break;
                }
                if step + 1 >= max_new_tokens {
                    break;
                }

                let next_logits = {
                    let _stream_ctx = StreamContext::new(generation_stream);
                    crate::models::gemma4::diagnostic::set_step(step);
                    self.run_paged_decode_step(token_id)?
                };
                let next_logits = next_logits.squeeze(Some(&[1]))?;
                y = sample_next_token(&next_logits, sampling_config)?;
                y.eval();

                crate::array::maybe_clear_cache_for_paged_step(step);
            }

            Ok((generated_tokens, finish_reason))
        })();

        // The Ok branch does NOT release the request here — the media-state
        // finalize decides between keep-live (continuable) and release
        // (non-continuable). The Err branch still releases fully.
        let (generated_tokens, finish_reason) = match forward_result {
            Ok(t) => t,
            Err(e) => {
                if let Some(adapter) = self.paged_adapter.as_mut() {
                    let _ = adapter.release_request();
                }
                return Err(e);
            }
        };

        let first_token_instant = std::time::Instant::now();

        let raw_text = tokenizer.decode_sync(&generated_tokens, false)?;

        // Two-state media finalize: keep the global paged KV live + remember a
        // sliding history checkpoint when this is a pure-causal media turn
        // (audio / non-unified image) under reuse, so a follow-up text delta
        // warm-continues; otherwise release + keep history/keys so the guard
        // rejects (single-shot, as today). A finalize Err means the live
        // request must be released before returning.
        let media_continuable = self.gemma4_media_continuable(&tokens);
        let reuse_cache = config.reuse_cache.unwrap_or(true);
        if let Err(e) = self.finalize_vision_turn_media_state(
            &tokens,
            &generated_tokens,
            &finish_reason,
            new_image_key,
            new_audio_key,
            media_continuable,
            reuse_cache,
        ) {
            if let Some(adapter) = self.paged_adapter.as_mut() {
                let _ = adapter.release_request();
            }
            return Err(e);
        }

        let generation_end = std::time::Instant::now();
        let ttft_ms = first_token_instant
            .duration_since(generation_start)
            .as_secs_f64()
            * 1000.0;
        let decode_ms = generation_end
            .duration_since(first_token_instant)
            .as_secs_f64()
            * 1000.0;
        let gen_toks = generated_tokens.len() as f64;

        let performance = Some(crate::profiling::PerformanceMetrics {
            ttft_ms,
            prefill_tokens_per_second: if ttft_ms > 0.0 {
                prefill_len as f64 / (ttft_ms / 1000.0)
            } else {
                0.0
            },
            decode_tokens_per_second: if decode_ms > 0.0 && gen_toks > 1.0 {
                (gen_toks - 1.0) / (decode_ms / 1000.0)
            } else {
                0.0
            },
            mtp_mean_accepted_tokens: None,
            mtp_mean_accepted_tokens_total: None,
            mtp_acceptance_by_position: None,
            mtp_cycles: None,
            mtp_mean_depth: None,
            profile_phases: None,
        });

        let mut parsed = super::output_parser::parse_gemma4_output(&raw_text);
        promote_channel_only_output(&mut parsed);
        let finish_reason = if parsed.tool_calls.iter().any(|tc| tc.status == "ok") {
            "tool_calls".to_string()
        } else {
            finish_reason
        };

        Ok(ChatResult {
            text: parsed.text,
            tool_calls: parsed.tool_calls,
            thinking: parsed.thinking,
            num_tokens: generated_tokens.len() as u32,
            prompt_tokens: prompt_token_count as u32,
            reasoning_tokens: 0,
            finish_reason,
            raw_text,
            cached_tokens: 0,
            performance,
        })
    }

    /// Streaming twin of [`Self::vision_paged_turn_sync_core`]. Same paged
    /// prefill + decode spine; streams parser segments and emits the terminal
    /// chunk itself.
    #[allow(clippy::too_many_arguments)]
    fn vision_paged_turn_stream_core(
        &mut self,
        rendered_tokens: &[u32],
        raw_images: &[Vec<u8>],
        raw_audio: &[Vec<u8>],
        tokenizer: &Arc<Qwen3Tokenizer>,
        config: &ChatConfig,
        eos_token_id: u32,
        sink: &dyn ChunkSink,
        cancelled: &AtomicBool,
    ) -> Result<()> {
        let cb = StreamSender(sink);
        let max_new_tokens = config.max_new_tokens.unwrap_or(2048);
        let (tokens, processed_images, audio_frames, new_image_key, new_audio_key) =
            self.prepare_multimodal_tokens(rendered_tokens, raw_images, raw_audio)?;
        if tokens.is_empty() {
            return Err(Error::from_reason("Empty prompt"));
        }
        let sampling_config = make_sampling_config(config, &self.config);
        let repetition_cutoff = repetition_cutoff_from_config(config);
        let eos_ids = self.config.eos_token_ids.clone();

        let prefill_slice: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let prefill_len = prefill_slice.len();
        let prompt = MxArray::from_int32(&prefill_slice, &[1, prefill_len as i64])?;
        let prompt_token_count = tokens.len();

        let generation_stream = Stream::new(DeviceType::Gpu);
        let _wired_ctx = crate::stream::WiredLimitContext::new(usize::MAX, vec![generation_stream]);

        let generation_start = std::time::Instant::now();

        let vision_embeds =
            self.build_gemma4_multimodal_embeds(&prompt, &processed_images, audio_frames.as_ref())?;

        // Derive layer kinds before acquiring the paged request (fallible, but
        // depends only on `self.config`). Hoisting it ahead of the request
        // acquisition keeps an early Err from leaking a prepared request.
        let layer_kinds = self.compute_layer_kinds()?;

        let seq_id: u32 = 0;
        let total_budget = tokens.len() as u32;
        // A new media set is non-continuable until its own finalize re-arms the
        // marker. Reset BEFORE the side-effecting prepare below, which releases
        // any prior kept-live request and can then fail (block exhaustion) via
        // `?` — a stale `true` would wrongly admit a later text delta.
        self.media_session_continuable = false;
        {
            let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                Error::from_reason("vision_paged_turn_stream_core: paged_adapter is None")
            })?;
            adapter
                .prepare_turn_with_max_cache_hit_tokens(
                    seq_id,
                    &tokens,
                    total_budget,
                    /* reuse_cache */ false,
                    &[],
                    /* cache_salt */ 0,
                    /* skip_lookup */ true,
                    /* max_cache_hit_tokens */ 0,
                )
                .map_err(Error::from_reason)?;
        }
        self.caches = Some(init_caches_for_config(&self.config));
        self.cached_token_history.clear();
        self.cached_image_key = None;
        self.cached_audio_key = None;
        self.sliding_prefix_checkpoints.clear();
        self.sliding_prompt_boundary_checkpoint = None;
        self.sliding_last_history_checkpoint = None;

        let mut decode_stream = tokenizer.inner().decode_stream(false);
        let mut streamed_text_len = 0;
        let mut stream_parser = super::output_parser::Gemma4StreamParser::new();
        let mut stream_dispatch = Gemma4StreamDispatchState::default();

        let forward_result = (|| -> Result<(Vec<u32>, String)> {
            let last_logits = {
                let _stream_ctx = StreamContext::new(generation_stream);
                crate::models::gemma4::diagnostic::set_step(-1);
                match vision_embeds {
                    Some(ref embeds) => {
                        self.run_paged_vlm_prefill(&tokens, embeds, &layer_kinds)?
                    }
                    None => {
                        let text_embeds = self.embed_tokens.forward(&prompt)?;
                        let text_embeds =
                            text_embeds.mul_scalar((self.config.hidden_size as f64).sqrt())?;
                        self.run_paged_vlm_prefill(&tokens, &text_embeds, &layer_kinds)?
                    }
                }
            };

            crate::array::synchronize_and_clear_cache();

            let mut y = sample_next_token(&last_logits, sampling_config)?;
            y.eval();

            let mut generated_tokens: Vec<u32> = Vec::new();
            let mut finish_reason = String::from("length");

            for step in 0..max_new_tokens {
                let token_id = y.item_at_int32(0)? as u32;
                generated_tokens.push(token_id);

                if cancelled.load(Ordering::Relaxed) {
                    finish_reason = "cancelled".to_string();
                    break;
                }

                let token_text = Qwen3Tokenizer::step_decode_stream(
                    &mut decode_stream,
                    tokenizer.inner(),
                    token_id,
                    &generated_tokens,
                    streamed_text_len,
                );
                streamed_text_len += token_text.len();
                let segments = stream_parser.feed(&token_text);
                stream_dispatch.dispatch_segments(segments, &cb);

                if is_eos_token(token_id, &eos_ids, eos_token_id) {
                    finish_reason = "stop".to_string();
                    break;
                }
                if let Some(reason) =
                    check_gemma4_repetition_cutoff(&generated_tokens, repetition_cutoff)
                {
                    finish_reason = reason.to_string();
                    break;
                }
                if step + 1 >= max_new_tokens {
                    break;
                }

                let next_logits = {
                    let _stream_ctx = StreamContext::new(generation_stream);
                    crate::models::gemma4::diagnostic::set_step(step);
                    self.run_paged_decode_step(token_id)?
                };
                let next_logits = next_logits.squeeze(Some(&[1]))?;
                y = sample_next_token(&next_logits, sampling_config)?;
                y.eval();

                crate::array::maybe_clear_cache_for_paged_step(step);
            }

            Ok((generated_tokens, finish_reason))
        })();

        // The Ok branch does NOT release the request here — the media-state
        // finalize decides between keep-live (continuable) and release
        // (non-continuable). The Err branch still releases fully.
        let (generated_tokens, finish_reason) = match forward_result {
            Ok(t) => t,
            Err(e) => {
                if let Some(adapter) = self.paged_adapter.as_mut() {
                    let _ = adapter.release_request();
                }
                return Err(e);
            }
        };

        let first_token_instant = std::time::Instant::now();

        let raw_text = tokenizer.decode_sync(&generated_tokens, false)?;

        // Flush residual bytes through the stream parser.
        if raw_text.len() > streamed_text_len {
            let residual = raw_text[streamed_text_len..].to_string();
            let mut segments = stream_parser.feed(&residual);
            segments.extend(stream_parser.flush());
            stream_dispatch.dispatch_segments(segments, &cb);
        } else {
            let tail = stream_parser.flush();
            stream_dispatch.dispatch_segments(tail, &cb);
        }
        stream_dispatch.finish(&cb);

        // Two-state media finalize (identical to the sync core via the shared
        // helper): keep-live + sliding checkpoint for a continuable pure-causal
        // media turn, else release + keep history/keys so the guard rejects.
        let media_continuable = self.gemma4_media_continuable(&tokens);
        let reuse_cache = config.reuse_cache.unwrap_or(true);
        if let Err(e) = self.finalize_vision_turn_media_state(
            &tokens,
            &generated_tokens,
            &finish_reason,
            new_image_key,
            new_audio_key,
            media_continuable,
            reuse_cache,
        ) {
            if let Some(adapter) = self.paged_adapter.as_mut() {
                let _ = adapter.release_request();
            }
            return Err(e);
        }

        let generation_end = std::time::Instant::now();
        let ttft_ms = first_token_instant
            .duration_since(generation_start)
            .as_secs_f64()
            * 1000.0;
        let decode_ms = generation_end
            .duration_since(first_token_instant)
            .as_secs_f64()
            * 1000.0;
        let gen_toks = generated_tokens.len() as f64;

        let performance = Some(crate::profiling::PerformanceMetrics {
            ttft_ms,
            prefill_tokens_per_second: if ttft_ms > 0.0 {
                prefill_len as f64 / (ttft_ms / 1000.0)
            } else {
                0.0
            },
            decode_tokens_per_second: if decode_ms > 0.0 && gen_toks > 1.0 {
                (gen_toks - 1.0) / (decode_ms / 1000.0)
            } else {
                0.0
            },
            mtp_mean_accepted_tokens: None,
            mtp_mean_accepted_tokens_total: None,
            mtp_acceptance_by_position: None,
            mtp_cycles: None,
            mtp_mean_depth: None,
            profile_phases: None,
        });

        let parsed_tool_calls = stream_parser.tool_calls();
        let parsed_thinking = stream_parser.thinking();
        let finish_reason = if parsed_tool_calls.iter().any(|tc| tc.status == "ok") {
            "tool_calls".to_string()
        } else {
            finish_reason
        };

        cb.call(
            Ok(ChatStreamChunk {
                text: String::new(),
                done: true,
                finish_reason: Some(finish_reason),
                tool_calls: Some(parsed_tool_calls),
                thinking: parsed_thinking,
                num_tokens: Some(generated_tokens.len() as u32),
                prompt_tokens: Some(prompt_token_count as u32),
                reasoning_tokens: Some(0),
                raw_text: Some(raw_text),
                cached_tokens: Some(0),
                performance,
                is_reasoning: None,
            }),
            ThreadsafeFunctionCallMode::NonBlocking,
        );

        Ok(())
    }

    // =================================================================
    // Block-paged dispatch (paged_turn_sync_core + helpers).
    //
    // Mirrors Qwen3's `paged_turn_sync_core` and LFM2's `forward_paged_or_flat`
    // pattern — sliding layers continue to use the existing flat
    // `Gemma4LayerCache::Sliding` path while global layers route through
    // `PagedKVCacheAdapter`. KV-shared layers are routed through their
    // anchor's physical KV slot/stash using routes derived from
    // `LayerKVCacheSpec`.
    //
    // Lifecycle (mirrors Qwen3 / LFM2):
    // 1. Adapter cold-start (or warm-continue when previous turn
    //    finalize_turn_keep_live'd a strict-prefix request).
    // 2. Sliding caches are restored from the live turn/checkpoint when
    //    available; otherwise the cached prefix is replayed through sliding
    //    layers before suffix prefill.
    // 3. Prefill via `run_paged_prefill_chunk` over the suffix.
    // 4. Decode loop via `run_paged_decode_step`.
    // 5. End-of-turn (success): `finalize_turn_keep_live` so the next
    //    turn's `continue_turn` can build on top of the partial trailing
    //    block's K/V (same partial-block carry trick as Qwen3 / LFM2).
    //
    // Caveats / scope:
    // * Text-only — vision turns dispatch through the flat path.
    // * Sliding layers still use flat rotating caches; true paged sliding
    //   storage is a separate kernel/storage step.
    // * Exact prefix hits are capped at `prompt_len - 1` so the final
    //   prompt token is always recomputed to produce logits.
    // =================================================================

    fn suppress_large_sliding_prefix_reuse_if_needed(
        &mut self,
        trace_label: &str,
        tokens: &[u32],
        total_budget: u32,
        seq_id: u32,
        restore_tokens: u32,
        trace_enabled: bool,
    ) -> Result<bool> {
        let block_size = self
            .paged_adapter
            .as_ref()
            .map(|adapter| adapter.block_size())
            .unwrap_or(0);
        let Some(suppression) = gemma4_large_sliding_restore_suppression_limit(
            &self.config,
            block_size,
            restore_tokens,
        ) else {
            return Ok(false);
        };

        // Sliding layers are recursive: without a close sliding checkpoint,
        // restoring a large paged-prefix hit can be slower than simply
        // recomputing the prompt. Keep prefix reuse only when the missing
        // sliding delta fits within the normal checkpoint interval; otherwise
        // fall back to a coherent cold prefill. Operators can set
        // MLX_GEMMA4_MAX_SLIDING_RESTORE_TOKENS=off for debugging.
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] gemma4 {}_prefix_reuse_suppressed reason=large_sliding_restore_limit restore_tokens={} limit={} limit_source={} block_size={}",
                trace_label, restore_tokens, suppression.limit, suppression.source, block_size
            ));
        }
        let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
            Error::from_reason(format!(
                "{}: paged_adapter is None while suppressing large sliding restore",
                trace_label
            ))
        })?;
        let _ = adapter.release_request();
        adapter
            .reset_for_new_request(seq_id)
            .map_err(Error::from_reason)?;
        let prefix = adapter
            .find_cached_prefix(tokens, &[], 0, true)
            .map_err(Error::from_reason)?;
        let allocated = adapter
            .allocate_suffix_blocks(total_budget)
            .map_err(Error::from_reason)?;
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] gemma4 {}_adapter_reset_done reason=large_sliding_restore_suppressed cached_prefix_tokens={} cached_blocks={} allocated_blocks={} request_tokens={} blocks={}",
                trace_label,
                prefix.cached_token_count,
                prefix.blocks.len(),
                allocated,
                adapter.current_token_count(),
                adapter.num_allocated_blocks()
            ));
        }
        Ok(true)
    }

    fn prepare_gemma4_paged_turn(
        &mut self,
        trace_label: &str,
        tokens: &[u32],
        reuse_cache: bool,
        total_budget: u32,
        seq_id: u32,
        trace_enabled: bool,
    ) -> Result<Gemma4PagedTurnPreparation> {
        let image_token_id = self.config.image_token_id.unwrap_or(258880) as u32;
        let audio_token_id = self.config.audio_token_id.unwrap_or(258881) as u32;
        let prompt_holds_media =
            prompt_holds_media_placeholders(tokens, image_token_id, audio_token_id);
        let plan = {
            let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                Error::from_reason(format!(
                    "{trace_label}: paged_adapter is None while preparing paged turn"
                ))
            })?;
            let adapter_live = adapter.is_live_for_continue();
            let adapter_request_tokens = adapter.request_tokens().len();
            let adapter_common_prefix = tokens
                .iter()
                .zip(adapter.request_tokens().iter())
                .take_while(|(a, b)| a == b)
                .count();
            if trace_enabled {
                write_inference_trace(format_args!(
                    "[MLX_TRACE] gemma4 {trace_label}_adapter_prepare live={} request_tokens={} common_prefix={} total_budget={} reuse_cache={}",
                    adapter_live,
                    adapter_request_tokens,
                    adapter_common_prefix,
                    total_budget,
                    reuse_cache
                ));
            }
            let max_cache_hit_tokens = total_budget.saturating_sub(1);
            // skip_lookup is on when the prompt still carries media placeholders:
            // any fallback that drops to a content-address prefix lookup (e.g. a
            // continue-turn-failure reset) then re-prefills the placeholders as
            // text instead of matching the token-only block hash of media K/V
            // registered by another session, so it can never consume that
            // session's stale media features.
            let plan = adapter
                .prepare_turn_with_max_cache_hit_tokens(
                    seq_id,
                    tokens,
                    total_budget,
                    reuse_cache,
                    &[],
                    0,
                    prompt_holds_media,
                    max_cache_hit_tokens,
                )
                .map_err(Error::from_reason)?;
            if trace_enabled {
                write_inference_trace(format_args!(
                    "[MLX_TRACE] gemma4 {trace_label}_adapter_prepare_done reason={:?} cached_prefix_tokens={} cached_blocks={} allocated_blocks={} request_tokens={} blocks={} continued_live={}",
                    plan.reason,
                    plan.cached_prefix_len,
                    plan.cached_blocks,
                    plan.allocated_blocks,
                    adapter.current_token_count(),
                    adapter.num_allocated_blocks(),
                    plan.continued_live_prefix
                ));
            }
            plan
        };

        let mut cached_prefix_len = plan.cached_prefix_len;
        let mut sliding_preparation = self.prepare_gemma4_sliding_prefix_state(
            tokens,
            cached_prefix_len,
            plan.continued_live_prefix,
        )?;
        if sliding_preparation.primed_prefix_len < cached_prefix_len {
            let suppressed = self.suppress_large_sliding_prefix_reuse_if_needed(
                trace_label,
                tokens,
                total_budget,
                seq_id,
                cached_prefix_len.saturating_sub(sliding_preparation.primed_prefix_len),
                trace_enabled,
            )?;
            if suppressed {
                let previous_cached_prefix_len = cached_prefix_len;
                cached_prefix_len = 0;
                sliding_preparation =
                    self.prepare_gemma4_sliding_prefix_state(tokens, cached_prefix_len, false)?;
                if trace_enabled {
                    write_inference_trace(format_args!(
                        "[MLX_TRACE] gemma4 {trace_label}_cached_prefix_reset previous_cached_prefix_tokens={} reason=sliding_restore_limit",
                        previous_cached_prefix_len
                    ));
                }
            }
        }

        let suffix_len = total_budget.checked_sub(cached_prefix_len).ok_or_else(|| {
            Error::from_reason(format!(
                "{trace_label}: cached_prefix_len {cached_prefix_len} exceeds total_budget \
                 {total_budget}"
            ))
        })?;
        if trace_enabled {
            let already_primed = sliding_preparation.primed_prefix_len == cached_prefix_len;
            write_inference_trace(format_args!(
                "[MLX_TRACE] gemma4 {trace_label}_sliding_prefix_state state={} cached_prefix_tokens={} sliding_primed_prefix_tokens={} replay_delta_tokens={} suffix_tokens={} already_primed={} continued_live={}",
                sliding_preparation.state,
                cached_prefix_len,
                sliding_preparation.primed_prefix_len,
                cached_prefix_len.saturating_sub(sliding_preparation.primed_prefix_len),
                suffix_len,
                already_primed,
                plan.continued_live_prefix
            ));
        }

        Ok(Gemma4PagedTurnPreparation {
            cached_prefix_len,
            suffix_len,
            sliding_primed_prefix_len: sliding_preparation.primed_prefix_len,
        })
    }

    /// Run a paged-attention prefill over the full prompt, dispatching
    /// per-layer between the adapter (global layers) and the existing
    /// flat path (sliding layers).
    ///
    /// `full_tokens` is the entire prompt (sliding layers re-prefill
    /// from token 0). `suffix_tokens` is the new portion beyond the
    /// paged prefix-cache hit (used by `record_tokens` +
    /// `update_keys_values` for global layers). `cached_prefix_len`
    /// is the paged-cache hit length.
    ///
    /// Returns the last position's logits squeezed to `[vocab]`.
    ///
    /// ## Prefill split (parity with the flat path)
    ///
    /// The flat path's `prefill_body_gemma4` processes tokens
    /// `[0..N-1]` through `forward_body`, then the caller runs a
    /// SECOND, single-token `forward_inner` for the final token. That
    /// second dispatch is load-bearing — see the doc-comment on
    /// `prefill_body_gemma4`: "SDPA computes slightly different
    /// numerical results for multi-token causal attention vs
    /// single-token attention with cached K/V. These small differences
    /// compound through layers, causing divergent logits if the last
    /// prompt token is processed in the same batch as the rest."
    ///
    /// This function mirrors that split for the paged path so the
    /// K/V-cache reduction order at the prefill→decode boundary
    /// matches between flat and paged. Without the split, BF16 SDPA
    /// drift on the last layer's hidden state at step 0 (~1%) flips
    /// argmax to a nearby zero-embedding `<unused>` token, causing the
    /// `<turn|>` stop signal to be missed and the decoder to fall into
    /// the all-zero-input cycle (`mean(V)` attention output → `id+1`
    /// counting cascade).
    fn run_paged_prefill_chunk(
        &mut self,
        full_tokens: &[u32],
        suffix_tokens: &[u32],
        cached_prefix_len: u32,
        sliding_primed_prefix_len: u32,
    ) -> Result<MxArray> {
        if suffix_tokens.is_empty() {
            return Err(Error::from_reason(
                "run_paged_prefill_chunk called with empty suffix",
            ));
        }
        if sliding_primed_prefix_len > cached_prefix_len {
            return Err(Error::from_reason(format!(
                "Gemma4 paged prefill sliding_primed_prefix_len {} exceeds cached_prefix_len {}",
                sliding_primed_prefix_len, cached_prefix_len
            )));
        }

        let suffix_len = suffix_tokens.len() as u32;
        let layer_kinds = self.compute_layer_kinds()?;
        let trace_enabled = inference_trace_enabled();
        let trace_start = trace_enabled.then(std::time::Instant::now);
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] gemma4 paged_prefill_start full_tokens={} cached_prefix_tokens={} suffix_tokens={}",
                full_tokens.len(),
                cached_prefix_len,
                suffix_tokens.len()
            ));
        }

        // For sliding layers we need state at position cached_prefix_len.
        // Sliding layers are restored each turn via reset_caches_sync, so
        // we need to reprefill any unprimed cached-prefix delta through
        // them BEFORE the suffix can attend. When a sparse checkpoint hits,
        // this is only the delta from that checkpoint to cached_prefix_len.
        if sliding_primed_prefix_len < cached_prefix_len {
            let prefix =
                &full_tokens[(sliding_primed_prefix_len as usize)..(cached_prefix_len as usize)];
            let sliding_trace_start = trace_enabled.then(std::time::Instant::now);
            self.run_sliding_only_prefill(prefix, sliding_primed_prefix_len, &layer_kinds)?;
            let block_size = self
                .paged_adapter
                .as_ref()
                .map(|adapter| adapter.block_size())
                .unwrap_or(0);
            let store_trace = self.remember_gemma4_sliding_prefix_checkpoint(
                full_tokens,
                cached_prefix_len,
                block_size,
                0,
            )?;
            if trace_enabled {
                write_inference_trace(format_args!(
                    "[MLX_TRACE] gemma4 paged_prefill_sliding_prefix_done cached_prefix_tokens={} restored_prefix_tokens={} replay_tokens={} checkpoint_stored={} store_eval_ms={:.1} store_snapshot_ms={:.1} store_token_clone_ms={:.1} store_update_ms={:.1} store_ms={:.1} elapsed_ms={:.1}",
                    cached_prefix_len,
                    sliding_primed_prefix_len,
                    prefix.len(),
                    store_trace.stored,
                    store_trace.eval_ms,
                    store_trace.snapshot_ms,
                    store_trace.token_clone_ms,
                    store_trace.update_ms,
                    store_trace.total_ms,
                    sliding_trace_start.map(elapsed_ms).unwrap_or(0.0)
                ));
            }
        } else if cached_prefix_len > 0 && trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] gemma4 paged_prefill_sliding_prefix_skipped cached_prefix_tokens={} sliding_primed_prefix_tokens={} reason=already_primed",
                cached_prefix_len, sliding_primed_prefix_len
            ));
        }

        crate::models::gemma4::diagnostic::set_path("paged");
        crate::models::gemma4::diagnostic::set_step(-1);

        // Two-pass split (mirrors flat `prefill_body_gemma4 →
        // forward_inner`):
        //   Pass 1: tokens `[..suffix_len-1]` (no-op if suffix_len == 1).
        //           Run this body in bounded chunks so long-context paged
        //           prefill does not build a single enormous lazy graph before
        //           the first cache materialization.
        //   Pass 2: the FINAL token (length 1). Now
        //           `cached_prefix_len_for_chunk = cached_prefix_len +
        //           suffix_len - 1`, which is > 0, so global layers
        //           take the cache-hit `read_kv_range` branch in
        //           `forward_paged` — the same path decode uses. This
        //           aligns the SDPA reduction order with the flat
        //           path's `forward_inner` dispatch.
        let configured_chunk_size = crate::array::paged_prefill_chunk_size();
        let mut pass2_first_position = cached_prefix_len;
        if suffix_len > 1 {
            // --- Pass 1: all-but-last suffix tokens, chunked. ---
            let pass1_tokens = &suffix_tokens[..(suffix_len as usize - 1)];
            let num_query_heads = u32::try_from(self.config.num_attention_heads).map_err(|_| {
                Error::from_reason(format!(
                    "Gemma4 paged prefill invalid num_attention_heads={}",
                    self.config.num_attention_heads
                ))
            })?;
            let global_head_size =
                u32::try_from(self.config.effective_head_dim(true)).map_err(|_| {
                    Error::from_reason(format!(
                        "Gemma4 paged prefill invalid global head_dim={}",
                        self.config.effective_head_dim(true)
                    ))
                })?;
            let paged_attention_enabled =
                gemma4_paged_prefill_paged_attention_enabled_for_chunking();
            let block_size = self
                .paged_adapter
                .as_ref()
                .map(|adapter| adapter.block_size())
                .unwrap_or(0);
            let full_tokens_len = u32::try_from(full_tokens.len())
                .map_err(|_| Error::from_reason("Gemma4 paged prefill token count exceeds u32"))?;
            let prompt_checkpoint_boundary_len = full_tokens_len
                .checked_div(block_size)
                .map(|blocks| blocks.saturating_mul(block_size))
                .unwrap_or(0);
            let checkpoint_interval =
                gemma4_sliding_decode_checkpoint_interval(&self.config, block_size);
            let mut body_chunk_plan =
                gemma4_paged_prefill_body_chunk_plan_with_checkpoint_interval(
                    configured_chunk_size,
                    pass1_tokens.len(),
                    pass2_first_position,
                    num_query_heads,
                    global_head_size,
                    paged_attention_enabled,
                    checkpoint_interval,
                )?;
            gemma4_split_body_chunk_plan_at_position(
                &mut body_chunk_plan,
                prompt_checkpoint_boundary_len,
            );
            let total_body_chunks = body_chunk_plan.len();
            let first_body_chunk_size = body_chunk_plan.first().map(|chunk| chunk.len).unwrap_or(0);
            let min_body_chunk_size = body_chunk_plan
                .iter()
                .map(|chunk| chunk.len)
                .min()
                .unwrap_or(0);
            let max_body_chunk_size = body_chunk_plan
                .iter()
                .map(|chunk| chunk.len)
                .max()
                .unwrap_or(0);
            let dynamic_v2_aux_caps = body_chunk_plan
                .iter()
                .filter(|chunk| chunk.capped_by_v2_aux_limit)
                .count();
            if trace_enabled {
                write_inference_trace(format_args!(
                    "[MLX_TRACE] gemma4 paged_prefill_body_chunking body_tokens={} chunk_size={} configured_chunk_size={} chunks={} min_chunk_size={} max_chunk_size={} dynamic_v2_aux_caps={} paged_attention_enabled={}",
                    pass1_tokens.len(),
                    first_body_chunk_size,
                    configured_chunk_size,
                    total_body_chunks,
                    min_body_chunk_size,
                    max_body_chunk_size,
                    dynamic_v2_aux_caps,
                    paged_attention_enabled
                ));
            }
            for (chunk_idx, chunk_plan) in body_chunk_plan.iter().enumerate() {
                let chunk_end = chunk_plan
                    .start
                    .checked_add(chunk_plan.len)
                    .ok_or_else(|| Error::from_reason("Gemma4 paged prefill chunk end overflow"))?;
                let chunk = pass1_tokens
                    .get(chunk_plan.start..chunk_end)
                    .ok_or_else(|| {
                        Error::from_reason("Gemma4 paged prefill chunk plan out of range")
                    })?;
                let chunk_first_position = chunk_plan.first_position;
                debug_assert_eq!(chunk_first_position, pass2_first_position);
                let chunk_trace_start = trace_enabled.then(std::time::Instant::now);
                if trace_enabled {
                    write_inference_trace(format_args!(
                        "[MLX_TRACE] gemma4 paged_prefill_body_chunk_start chunk={}/{} first_position={} tokens={} capped_by_v2_aux_limit={}",
                        chunk_idx + 1,
                        total_body_chunks,
                        chunk_first_position,
                        chunk.len(),
                        chunk_plan.capped_by_v2_aux_limit
                    ));
                }
                {
                    let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                        Error::from_reason("run_paged_prefill_chunk: paged_adapter is None")
                    })?;
                    if trace_enabled {
                        write_inference_trace(format_args!(
                            "[MLX_TRACE] gemma4 paged_prefill_record_tokens_start chunk={}/{} first_position={} tokens={} current_tokens_before={} blocks_before={}",
                            chunk_idx + 1,
                            total_body_chunks,
                            chunk_first_position,
                            chunk.len(),
                            adapter.current_token_count(),
                            adapter.num_allocated_blocks()
                        ));
                    }
                    adapter.record_tokens(chunk).map_err(Error::from_reason)?;
                    if trace_enabled {
                        write_inference_trace(format_args!(
                            "[MLX_TRACE] gemma4 paged_prefill_record_tokens_done chunk={}/{} current_tokens_after={} blocks_after={}",
                            chunk_idx + 1,
                            total_body_chunks,
                            adapter.current_token_count(),
                            adapter.num_allocated_blocks()
                        ));
                    }
                }
                let layer_loop_start = trace_enabled.then(std::time::Instant::now);
                if trace_enabled {
                    write_inference_trace(format_args!(
                        "[MLX_TRACE] gemma4 paged_prefill_layer_loop_start chunk={}/{} first_position={} cached_prefix_for_chunk={} tokens={}",
                        chunk_idx + 1,
                        total_body_chunks,
                        chunk_first_position,
                        chunk_first_position,
                        chunk.len()
                    ));
                }
                let _hidden_pass1 = self.run_paged_prefill_layer_loop(
                    chunk,
                    chunk_first_position,
                    chunk_first_position,
                    &layer_kinds,
                )?;
                if let Some(adapter) = self.paged_adapter.as_mut() {
                    adapter
                        .eval_pending_pool_writes()
                        .map_err(Error::from_reason)?;
                }
                if trace_enabled {
                    write_inference_trace(format_args!(
                        "[MLX_TRACE] gemma4 paged_prefill_layer_loop_done chunk={}/{} first_position={} tokens={} elapsed_ms={:.1}",
                        chunk_idx + 1,
                        total_body_chunks,
                        chunk_first_position,
                        chunk.len(),
                        layer_loop_start.map(elapsed_ms).unwrap_or(0.0)
                    ));
                }

                // Materialize writes from this body chunk before the next
                // chunk reads through them. Native paged writes are lazy graph
                // nodes; sliding flat caches are lazy too.
                if let Some(caches) = self.caches.as_ref() {
                    eval_gemma4_caches(caches)?;
                }
                crate::array::clear_cache();
                pass2_first_position = pass2_first_position
                    .checked_add(chunk.len() as u32)
                    .ok_or_else(|| {
                        Error::from_reason("Gemma4 paged prefill token position overflow")
                    })?;
                // Global paged-cache hits can land at any prior full-block
                // prefix, not just the previous prompt boundary. Persist
                // sliding snapshots at the normal window stride during
                // prefill too, so a later branch switch can restore from a
                // nearby sliding state instead of replaying tens of thousands
                // of sliding tokens or falling back to a full cold prefill.
                self.maybe_remember_gemma4_sliding_decode_boundary_checkpoint(
                    "paged_prefill",
                    trace_enabled,
                )?;
                if pass2_first_position == prompt_checkpoint_boundary_len {
                    self.maybe_remember_gemma4_sliding_prompt_boundary_checkpoint(
                        "paged_prefill",
                        full_tokens,
                        prompt_checkpoint_boundary_len,
                        trace_enabled,
                    )?;
                }
                if trace_enabled {
                    write_inference_trace(format_args!(
                        "[MLX_TRACE] gemma4 paged_prefill_body_chunk_done chunk={}/{} next_position={} elapsed_ms={:.1}",
                        chunk_idx + 1,
                        total_body_chunks,
                        pass2_first_position,
                        chunk_trace_start.map(elapsed_ms).unwrap_or(0.0)
                    ));
                }
            }
        }

        // --- Pass 2: the FINAL suffix token (length 1). ---
        let pass2_tokens = &suffix_tokens[(suffix_len as usize - 1)..];
        {
            let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                Error::from_reason("run_paged_prefill_chunk: paged_adapter is None")
            })?;
            if trace_enabled {
                write_inference_trace(format_args!(
                    "[MLX_TRACE] gemma4 paged_prefill_final_record_tokens_start first_position={} tokens={} current_tokens_before={} blocks_before={}",
                    pass2_first_position,
                    pass2_tokens.len(),
                    adapter.current_token_count(),
                    adapter.num_allocated_blocks()
                ));
            }
            adapter
                .record_tokens(pass2_tokens)
                .map_err(Error::from_reason)?;
            if trace_enabled {
                write_inference_trace(format_args!(
                    "[MLX_TRACE] gemma4 paged_prefill_final_record_tokens_done current_tokens_after={} blocks_after={}",
                    adapter.current_token_count(),
                    adapter.num_allocated_blocks()
                ));
            }
        }
        let pass2_cached_prefix_len = pass2_first_position;
        let pass2_layer_loop_start = trace_enabled.then(std::time::Instant::now);
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] gemma4 paged_prefill_final_layer_loop_start first_position={} cached_prefix_for_chunk={} tokens={}",
                pass2_first_position,
                pass2_cached_prefix_len,
                pass2_tokens.len()
            ));
        }
        let mut hidden_states = self.run_paged_prefill_layer_loop(
            pass2_tokens,
            pass2_first_position,
            pass2_cached_prefix_len,
            &layer_kinds,
        )?;
        if let Some(adapter) = self.paged_adapter.as_mut() {
            adapter
                .eval_pending_pool_writes()
                .map_err(Error::from_reason)?;
        }
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] gemma4 paged_prefill_final_layer_loop_done first_position={} tokens={} elapsed_ms={:.1}",
                pass2_first_position,
                pass2_tokens.len(),
                pass2_layer_loop_start.map(elapsed_ms).unwrap_or(0.0)
            ));
        }

        self.maybe_remember_gemma4_sliding_prompt_boundary_checkpoint(
            "paged_prefill",
            full_tokens,
            pass2_first_position + pass2_tokens.len() as u32,
            trace_enabled,
        )?;

        // Final norm + lm_head + softcap (only for the final token).
        hidden_states = self.final_norm.forward(&hidden_states)?;
        crate::models::gemma4::diagnostic::dump_norm(0, "post_final_norm", &hidden_states, None);
        let logits = if let Some(ref head) = self.lm_head {
            head.forward(&hidden_states)?
        } else if let Some(ref w_t) = self.embed_weight_t {
            hidden_states.matmul(w_t)?
        } else {
            let weight = self.embed_tokens.get_weight();
            let weight_t = weight.transpose(Some(&[1, 0]))?;
            hidden_states.matmul(&weight_t)?
        };
        crate::models::gemma4::diagnostic::dump_logits("pre_softcap", &logits);
        let logits = if let Some(cap) = self.config.final_logit_softcapping {
            let cap_arr = MxArray::scalar_float_like(cap, &logits)?;
            let handle = unsafe { mlx_sys::mlx_logit_softcap(logits.handle.0, cap_arr.handle.0) };
            let capped = MxArray::from_handle(handle, "logit_softcap")?;
            crate::models::gemma4::diagnostic::dump_logits("post_softcap", &capped);
            capped
        } else {
            crate::models::gemma4::diagnostic::dump_logits("post_softcap", &logits);
            logits
        };

        let last_seq_len = logits.shape_at(1)?;
        let last = logits
            .slice_axis(1, last_seq_len - 1, last_seq_len)?
            .squeeze(Some(&[0, 1]))?;
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] gemma4 paged_prefill_done suffix_tokens={} elapsed_ms={:.1}",
                suffix_tokens.len(),
                trace_start.map(elapsed_ms).unwrap_or(0.0)
            ));
        }
        Ok(last)
    }

    /// One forward pass through the embed → PLE → layer-loop pipeline
    /// for a single contiguous chunk of tokens. Returns the chunk's
    /// post-final-layer hidden state (NO final norm / lm_head / softcap
    /// — the caller decides whether to apply those).
    ///
    /// `chunk_tokens` is the slice being processed THIS call.
    /// `first_logical_position` is the absolute logical position of
    /// `chunk_tokens[0]` in the request (used as the RoPE offset and
    /// the slot-mapping anchor). `cached_prefix_len_for_chunk` is the
    /// number of K/V tokens already in the paged pool BEFORE this
    /// chunk's writes — when this is > 0 the global-layer SDPA takes
    /// the `read_kv_range` cache-hit branch, matching decode's
    /// reduction order. `layer_kinds` is the per-layer routing
    /// classification (Sliding / GlobalPaged / SharedOnGlobal /
    /// SharedOnSliding).
    ///
    /// Caller must have already called `record_tokens(chunk_tokens)`
    /// on the paged adapter so `update_keys_values`'s alignment check
    /// (`first_logical_position == current_token_count - chunk.len()`)
    /// passes.
    fn run_paged_prefill_layer_loop(
        &mut self,
        chunk_tokens: &[u32],
        first_logical_position: u32,
        cached_prefix_len_for_chunk: u32,
        layer_kinds: &[Gemma4LayerKind],
    ) -> Result<MxArray> {
        let chunk_len = chunk_tokens.len() as u32;
        if chunk_len == 0 {
            return Err(Error::from_reason(
                "run_paged_prefill_layer_loop: chunk_tokens must be non-empty",
            ));
        }
        let trace_enabled = inference_trace_enabled();
        let trace_start = trace_enabled.then(std::time::Instant::now);
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] gemma4 paged_prefill_layer_loop_enter first_position={} cached_prefix_for_chunk={} tokens={} layers={}",
                first_logical_position,
                cached_prefix_len_for_chunk,
                chunk_len,
                self.layers.len()
            ));
        }

        let input_ids = MxArray::from_uint32(chunk_tokens, &[1, chunk_len as i64])?;
        let mut hidden_states = self.embed_tokens.forward(&input_ids)?;
        // Apply Gemma4 embedding scaling (sqrt(hidden_size)).
        hidden_states = hidden_states.mul_scalar((self.config.hidden_size as f64).sqrt())?;

        // Compute PLE (per-layer embeddings) for the chunk's tokens.
        // Mirrors `forward_body`: PLE feeds an additive residual inside
        // every layer's `apply_ffn_ple_scalar` tail. For Gemma4 E2B/E4B
        // this is load-bearing — dropping it produces nonsense logits
        // because each layer is missing a critical residual
        // contribution. Sliding-only re-prefill of any cached prefix
        // doesn't propagate PLE through the global layers we'll touch
        // here (their stored K/V already accounts for it).
        let projected_ple: Option<MxArray> = if let Some(ref ple) = self.ple {
            let pre_layer_h = hidden_states.clone();
            Some(compute_ple(
                &input_ids,
                &pre_layer_h,
                ple,
                chunk_len as i64,
            )?)
        } else {
            None
        };

        // Build sliding masks against the bounded rotating-cache attention view,
        // not the absolute prompt offset. This mirrors mlx-lm's
        // RotatingKVCache.make_mask behavior and avoids huge long-context masks.
        let seq_len = chunk_len as i64;
        let sliding_offset = self
            .caches
            .as_ref()
            .and_then(|caches| {
                caches
                    .iter()
                    .enumerate()
                    .find(|(i, _)| self.config.is_sliding_layer(*i))
                    .map(|(_, c)| c.get_offset())
            })
            .unwrap_or(0);
        let sliding_window = self.config.sliding_window as i64;
        let sliding_mask_offset =
            sliding_mask_offset_for_chunk(seq_len, sliding_offset, sliding_window);
        if trace_enabled && (sliding_offset > 0 || sliding_mask_offset.is_some()) {
            write_inference_trace(format_args!(
                "[MLX_TRACE] gemma4 paged_prefill_sliding_mask seq_len={} cache_offset={} mask_offset={} window={} explicit_mask={}",
                seq_len,
                sliding_offset,
                sliding_mask_offset.unwrap_or(0),
                sliding_window,
                sliding_mask_offset.is_some()
            ));
        }
        let sliding_mask = sliding_mask_offset
            .map(|offset| create_sliding_mask(seq_len, offset, sliding_window))
            .transpose()?;

        let has_kv_sharing = self.config.num_kv_shared_layers.is_some_and(|n| n > 0);
        let num_layers = self.layers.len();
        // Stash for sliding-anchor K/V reused by SharedOnSliding layers.
        let mut sliding_shared_kv: HashMap<u32, (MxArray, MxArray)> = HashMap::new();

        #[allow(clippy::needless_range_loop)]
        for layer_idx in 0..num_layers {
            crate::models::gemma4::diagnostic::set_layer(layer_idx);
            let kind = layer_kinds[layer_idx];
            let layer_trace_start = trace_enabled.then(std::time::Instant::now);
            if trace_enabled {
                write_inference_trace(format_args!(
                    "[MLX_TRACE] gemma4 paged_prefill_layer_start layer={} kind={:?} first_position={} cached_prefix_for_chunk={} tokens={}",
                    layer_idx, kind, first_logical_position, cached_prefix_len_for_chunk, chunk_len
                ));
            }
            let layer: &Gemma4DecoderLayer = unsafe {
                let ptr = self.layers.as_ptr().add(layer_idx);
                &*ptr
            };
            let mask: Option<&MxArray> = if matches!(kind, Gemma4LayerKind::Sliding) {
                sliding_mask.as_ref()
            } else {
                None
            };

            let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                Error::from_reason(
                    "run_paged_prefill_layer_loop: paged_adapter dropped mid-forward",
                )
            })?;
            let flat_cache: Option<&mut Gemma4LayerCache> =
                if matches!(kind, Gemma4LayerKind::Sliding) {
                    let caches = unsafe {
                        let raw = self.caches.as_mut().ok_or_else(|| {
                            Error::from_reason(
                                "run_paged_prefill_layer_loop: sliding cache slot missing",
                            )
                        })? as *mut Vec<Gemma4LayerCache>;
                        &mut *raw
                    };
                    Some(&mut caches[layer_idx])
                } else {
                    None
                };

            // Build SharedKvInputs for shared layer kinds.
            let shared_inputs = match kind {
                Gemma4LayerKind::SharedOnGlobal { .. } => {
                    // Anchor's pool currently holds
                    // `cached_prefix_len_for_chunk + chunk_len` tokens
                    // for this layer (the anchor wrote its part of
                    // this chunk earlier in the same loop).
                    let total_ctx = cached_prefix_len_for_chunk + chunk_len;
                    Some(super::decoder_layer::SharedKvInputs {
                        cache_offset: first_logical_position as i32,
                        total_ctx,
                        keys: None,
                        values: None,
                    })
                }
                Gemma4LayerKind::SharedOnSliding { anchor_layer_idx } => {
                    let (k, v) = sliding_shared_kv.get(&anchor_layer_idx).ok_or_else(|| {
                        Error::from_reason(format!(
                            "run_paged_prefill_layer_loop: SharedOnSliding anchor {} stash \
                             missing",
                            anchor_layer_idx
                        ))
                    })?;
                    let cache_offset =
                        (first_logical_position as i32 + chunk_len as i32) - seq_len as i32;
                    Some(super::decoder_layer::SharedKvInputs {
                        cache_offset,
                        total_ctx: 0, // unused for SharedOnSliding
                        keys: Some(k),
                        values: Some(v),
                    })
                }
                _ => None,
            };

            // For Sliding layers that anchor a SharedOnSliding chain,
            // request the stash so the shared layer can pull K/V.
            let needs_stash = has_kv_sharing
                && matches!(kind, Gemma4LayerKind::Sliding)
                && self.config.should_store_shared_kv(layer_idx);

            // Slice the per-layer PLE input ([B, T, num_layers, ple_dim] →
            // [B, T, ple_dim]). Mirrors `forward_body`'s per-layer slice.
            let ple_input = projected_ple.as_ref().map(|p| {
                p.slice_axis(2, layer_idx as i64, layer_idx as i64 + 1)
                    .and_then(|s| s.squeeze(Some(&[2])))
            });
            let ple_input_ref = match &ple_input {
                Some(Ok(arr)) => Some(arr),
                _ => None,
            };

            let next_hidden_states = layer.forward_paged_or_flat(
                &hidden_states,
                kind,
                adapter,
                first_logical_position,
                cached_prefix_len_for_chunk,
                /* is_prefill */ true,
                mask,
                flat_cache,
                ple_input_ref,
                needs_stash,
                shared_inputs,
            )?;
            if trace_enabled {
                write_inference_trace(format_args!(
                    "[MLX_TRACE] gemma4 paged_prefill_layer_done layer={} kind={:?} elapsed_ms={:.1}",
                    layer_idx,
                    kind,
                    layer_trace_start.map(elapsed_ms).unwrap_or(0.0)
                ));
            }
            hidden_states = next_hidden_states;

            // After a Sliding anchor's forward, capture its stash so
            // downstream SharedOnSliding layers can attend over it.
            if needs_stash {
                let caches = unsafe {
                    let raw = self.caches.as_mut().ok_or_else(|| {
                        Error::from_reason(
                            "run_paged_prefill_layer_loop: sliding cache slot missing \
                             post-forward",
                        )
                    })? as *mut Vec<Gemma4LayerCache>;
                    &mut *raw
                };
                if let Some((k, v)) = caches[layer_idx].take_stashed_kv() {
                    sliding_shared_kv.insert(layer_idx as u32, (k, v));
                }
            }
            // Smooth the prefill memory peak: every K layers, materialize the
            // residual stream so MLX can release the upstream graph nodes
            // (embedding + every prior layer's attention/MLP/PLE intermediates)
            // from the cache pool. Without this the in-flight lazy graph
            // accumulates on long contexts before the post-prefill sync fires.
            // Cadence is `MLX_PAGED_PREFILL_EVAL_INTERVAL` (default 8).
            crate::array::maybe_eval_clear_for_paged_prefill_layer(layer_idx, &hidden_states)?;
        }

        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] gemma4 paged_prefill_layer_loop_exit first_position={} tokens={} elapsed_ms={:.1}",
                first_logical_position,
                chunk_len,
                trace_start.map(elapsed_ms).unwrap_or(0.0)
            ));
        }
        Ok(hidden_states)
    }

    /// Vision variant of [`Self::run_paged_prefill_layer_loop`]: drives one
    /// contiguous chunk of the merged image+text embeddings through the hybrid
    /// paged dispatch (global → adapter, sliding → flat rotating cache,
    /// KV-shared → anchor stash).
    ///
    /// Identical layer routing to the text loop, with two image-aware seams:
    ///   * the residual stream is seeded from the supplied `chunk_embeds`
    ///     (the `masked_scatter` output for this chunk, ALREADY scaled by
    ///     `sqrt(hidden_size)` by the caller) instead of
    ///     `embed_tokens.forward(token_ids)`;
    ///   * PLE per-layer embeddings zero the image-token positions in
    ///     `chunk_token_ids` before `compute_ple`, because the image positions
    ///     carry vision features in the residual, not token PLE residuals.
    ///
    /// `chunk_token_ids` is the expanded token slice for this chunk (drives the
    /// PLE image mask and the sliding-mask sequence length).
    /// `chunk_embeds` is `[1, chunk_len, hidden]`.
    #[allow(clippy::too_many_arguments)]
    fn run_paged_vlm_prefill_layer_loop(
        &mut self,
        chunk_token_ids: &[u32],
        chunk_embeds: &MxArray,
        first_logical_position: u32,
        cached_prefix_len_for_chunk: u32,
        layer_kinds: &[Gemma4LayerKind],
        overlay_type_ids: Option<&MxArray>,
    ) -> Result<MxArray> {
        let chunk_len = chunk_token_ids.len() as u32;
        if chunk_len == 0 {
            return Err(Error::from_reason(
                "run_paged_vlm_prefill_layer_loop: chunk_token_ids must be non-empty",
            ));
        }

        let input_ids = MxArray::from_uint32(chunk_token_ids, &[1, chunk_len as i64])?;
        let mut hidden_states = chunk_embeds.clone();

        // PLE over media-masked token ids: image AND audio positions hold
        // projected media features (not token embeddings), so their PLE
        // residual must be zero.
        let projected_ple: Option<MxArray> = if let Some(ref ple) = self.ple {
            let image_token_id = self.config.image_token_id.unwrap_or(258880);
            let image_token = MxArray::scalar_int(image_token_id)?;
            let mut media_mask = input_ids.equal(&image_token)?;
            if let Some(audio_token_id) = self.config.audio_token_id {
                let audio_token = MxArray::scalar_int(audio_token_id)?;
                let audio_mask = input_ids.equal(&audio_token)?;
                media_mask = media_mask.logical_or(&audio_mask)?;
            }
            let zero = MxArray::scalar_int(0)?;
            // Media positions (image and audio) are excluded from the PLE
            // residual because their embedding is the projected media feature,
            // not a learned token.
            let masked_ids = media_mask.where_(&zero, &input_ids)?;
            let pre_layer_h = hidden_states.clone();
            Some(compute_ple(
                &masked_ids,
                &pre_layer_h,
                ple,
                chunk_len as i64,
            )?)
        } else {
            None
        };

        // Sliding mask against the bounded rotating-cache attention view —
        // identical derivation to the text paged loop.
        let seq_len = chunk_len as i64;
        let sliding_offset = self
            .caches
            .as_ref()
            .and_then(|caches| {
                caches
                    .iter()
                    .enumerate()
                    .find(|(i, _)| self.config.is_sliding_layer(*i))
                    .map(|(_, c)| c.get_offset())
            })
            .unwrap_or(0);
        let sliding_window = self.config.sliding_window as i64;
        let sliding_mask_offset =
            sliding_mask_offset_for_chunk(seq_len, sliding_offset, sliding_window);
        let mut sliding_mask = sliding_mask_offset
            .map(|offset| create_sliding_mask(seq_len, offset, sliding_window))
            .transpose()?;

        // Unified-vision bidirectional overlay. Active only on the cold-start
        // single-chunk prefill (`overlay_type_ids` is Some and
        // `cached_prefix_len_for_chunk == 0`), where every mask key dimension
        // equals `seq_len`. Both layer types get an EXPLICIT materialized
        // boolean keep-mask (true=keep): the global layer's normal None/causal
        // fast path and the sliding layer's possibly-None window mask are
        // replaced by `base | same_image_block`.
        let overlay_active = overlay_type_ids.is_some() && cached_prefix_len_for_chunk == 0;
        let overlay_global_mask: Option<MxArray> = if overlay_active {
            let type_ids = overlay_type_ids.unwrap();
            let base = create_causal_mask(seq_len as i32, None, None)?;
            let base = base.reshape(&[1, 1, seq_len, seq_len])?;
            Some(apply_bidirectional_vision_overlay(&base, type_ids)?)
        } else {
            None
        };
        if overlay_active {
            let type_ids = overlay_type_ids.unwrap();
            let base = create_causal_mask(seq_len as i32, None, Some(sliding_window as i32))?;
            let base = base.reshape(&[1, 1, seq_len, seq_len])?;
            sliding_mask = Some(apply_bidirectional_vision_overlay(&base, type_ids)?);
        }

        let has_kv_sharing = self.config.num_kv_shared_layers.is_some_and(|n| n > 0);
        let num_layers = self.layers.len();
        let mut sliding_shared_kv: HashMap<u32, (MxArray, MxArray)> = HashMap::new();

        #[allow(clippy::needless_range_loop)]
        for layer_idx in 0..num_layers {
            crate::models::gemma4::diagnostic::set_layer(layer_idx);
            let kind = layer_kinds[layer_idx];
            let layer: &Gemma4DecoderLayer = unsafe {
                let ptr = self.layers.as_ptr().add(layer_idx);
                &*ptr
            };
            let mask: Option<&MxArray> = if matches!(kind, Gemma4LayerKind::Sliding) {
                sliding_mask.as_ref()
            } else {
                // Global/full layers normally pass None (internal causal). When
                // the overlay is active they receive the explicit bidirectional
                // keep-mask, which `forward_paged` applies in the fresh-prefill
                // branch.
                overlay_global_mask.as_ref()
            };

            let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                Error::from_reason(
                    "run_paged_vlm_prefill_layer_loop: paged_adapter dropped mid-forward",
                )
            })?;
            let flat_cache: Option<&mut Gemma4LayerCache> =
                if matches!(kind, Gemma4LayerKind::Sliding) {
                    let caches = unsafe {
                        let raw = self.caches.as_mut().ok_or_else(|| {
                            Error::from_reason(
                                "run_paged_vlm_prefill_layer_loop: sliding cache slot missing",
                            )
                        })? as *mut Vec<Gemma4LayerCache>;
                        &mut *raw
                    };
                    Some(&mut caches[layer_idx])
                } else {
                    None
                };

            let shared_inputs = match kind {
                Gemma4LayerKind::SharedOnGlobal { .. } => {
                    let total_ctx = cached_prefix_len_for_chunk + chunk_len;
                    Some(super::decoder_layer::SharedKvInputs {
                        cache_offset: first_logical_position as i32,
                        total_ctx,
                        keys: None,
                        values: None,
                    })
                }
                Gemma4LayerKind::SharedOnSliding { anchor_layer_idx } => {
                    let (k, v) = sliding_shared_kv.get(&anchor_layer_idx).ok_or_else(|| {
                        Error::from_reason(format!(
                            "run_paged_vlm_prefill_layer_loop: SharedOnSliding anchor {} stash \
                             missing",
                            anchor_layer_idx
                        ))
                    })?;
                    let cache_offset =
                        (first_logical_position as i32 + chunk_len as i32) - seq_len as i32;
                    Some(super::decoder_layer::SharedKvInputs {
                        cache_offset,
                        total_ctx: 0,
                        keys: Some(k),
                        values: Some(v),
                    })
                }
                _ => None,
            };

            let needs_stash = has_kv_sharing
                && matches!(kind, Gemma4LayerKind::Sliding)
                && self.config.should_store_shared_kv(layer_idx);

            let ple_input = projected_ple.as_ref().map(|p| {
                p.slice_axis(2, layer_idx as i64, layer_idx as i64 + 1)
                    .and_then(|s| s.squeeze(Some(&[2])))
            });
            let ple_input_ref = match &ple_input {
                Some(Ok(arr)) => Some(arr),
                _ => None,
            };

            let next_hidden_states = layer.forward_paged_or_flat(
                &hidden_states,
                kind,
                adapter,
                first_logical_position,
                cached_prefix_len_for_chunk,
                /* is_prefill */ true,
                mask,
                flat_cache,
                ple_input_ref,
                needs_stash,
                shared_inputs,
            )?;
            hidden_states = next_hidden_states;

            if needs_stash {
                let caches = unsafe {
                    let raw = self.caches.as_mut().ok_or_else(|| {
                        Error::from_reason(
                            "run_paged_vlm_prefill_layer_loop: sliding cache slot missing \
                             post-forward",
                        )
                    })? as *mut Vec<Gemma4LayerCache>;
                    &mut *raw
                };
                if let Some((k, v)) = caches[layer_idx].take_stashed_kv() {
                    sliding_shared_kv.insert(layer_idx as u32, (k, v));
                }
            }
            crate::array::maybe_eval_clear_for_paged_prefill_layer(layer_idx, &hidden_states)?;
        }

        Ok(hidden_states)
    }

    /// Cold-start paged prefill over the merged image+text embeddings.
    ///
    /// Single-shot only: the adapter holds zero tokens and the sliding flat
    /// caches were freshly built, so `cached_prefix_len == 0` and there is no
    /// prefix-cache restore. Splits the merged-embedding body prefill from a
    /// last-token `forward_inner`, a split that is load-bearing — see
    /// [`Self::run_paged_prefill_chunk`] for why
    /// the final prompt token must run through the cache-hit branch separately
    /// (BF16 SDPA drift otherwise flips argmax to a zero-embedding `<unused>`
    /// token and the `<turn|>` stop is missed).
    ///
    /// `expanded_tokens` is the full `BOI + N×image + EOI` expanded sequence.
    /// `inputs_embeds` is `[1, prompt_len, hidden]`, ALREADY scaled by
    /// `sqrt(hidden_size)` and with vision features scattered at the image
    /// positions. Returns the final token's logits squeezed to `[vocab]`.
    fn run_paged_vlm_prefill(
        &mut self,
        expanded_tokens: &[u32],
        inputs_embeds: &MxArray,
        layer_kinds: &[Gemma4LayerKind],
    ) -> Result<MxArray> {
        if expanded_tokens.is_empty() {
            return Err(Error::from_reason(
                "run_paged_vlm_prefill called with empty prompt",
            ));
        }
        let prompt_len = expanded_tokens.len() as u32;

        crate::models::gemma4::diagnostic::set_path("paged");
        crate::models::gemma4::diagnostic::set_step(-1);

        // Unified-vision bidirectional overlay gate: is_unified +
        // use_bidirectional_attention=="vision" + image tokens present + no audio
        // tokens + prefill (seq_len>1). Mixed image+audio prompts stay causal
        // (audio wins) — see `vision_overlay_active`. When active, the whole image
        // block must live in ONE prefill chunk so bidirectionality is not severed
        // by chunk boundaries.
        let image_token_id = self.config.image_token_id.unwrap_or(258880) as u32;
        let audio_token_id = self.config.audio_token_id.unwrap_or(258881) as u32;
        let has_image = expanded_tokens.contains(&image_token_id);
        let has_audio = expanded_tokens.contains(&audio_token_id);
        let overlay_full_type_ids: Option<MxArray> = if super::vision_mask::vision_overlay_active(
            self.config.is_unified,
            self.config.use_bidirectional_attention.as_deref() == Some("vision"),
            has_image,
            has_audio,
            prompt_len as usize,
        ) {
            Some(super::vision_mask::build_image_token_type_ids(
                expanded_tokens,
                image_token_id,
            )?)
        } else {
            None
        };
        let overlay_active = overlay_full_type_ids.is_some();
        // The overlay only reaches GlobalPaged/Sliding layers. KV-shared layers
        // (SharedOnGlobal/SharedOnSliding) run forward_paged_shared, which takes
        // no mask and would silently stay causal — a half-applied overlay across
        // the stack. The 12B unified checkpoint has num_kv_shared_layers==0, so
        // this never fires; fail loudly rather than corrupt attention if a shared
        // unified checkpoint is ever loaded.
        if overlay_active && self.config.num_kv_shared_layers.is_some_and(|n| n > 0) {
            return Err(Error::from_reason(
                "Gemma4 unified-vision bidirectional overlay is unsupported with KV-shared layers \
                 (num_kv_shared_layers > 0): forward_paged_shared does not carry the overlay mask",
            ));
        }

        // Pass 1: tokens [0..prompt_len-1] in bounded chunks. Pass 2: the
        // FINAL token, run with cached_prefix_len_for_chunk > 0 so global
        // layers take the same cache-hit reduction order decode uses.
        let mut pass1_position: u32 = 0;
        if prompt_len > 1 {
            let pass1_len = (prompt_len - 1) as usize;
            let configured_chunk_size = crate::array::paged_prefill_chunk_size();
            // Force a single chunk when the overlay is active: a split would put
            // part of the image block in a later (cache-hit) chunk that no
            // longer carries the bidirectional mask.
            let chunk_size = if overlay_active || configured_chunk_size <= 0 {
                pass1_len
            } else {
                (configured_chunk_size as usize).max(1)
            };
            let mut offset: usize = 0;
            while offset < pass1_len {
                let end = (offset + chunk_size).min(pass1_len);
                let chunk_tokens = &expanded_tokens[offset..end];
                let chunk_len = (end - offset) as i64;
                let chunk_embeds =
                    inputs_embeds.slice_axis(1, offset as i64, offset as i64 + chunk_len)?;
                // Per-chunk type-ids slice. With the forced single chunk this is
                // the whole pass-1 span; the explicit slice keeps the code
                // correct even if a future chunking path is added.
                let chunk_type_ids: Option<MxArray> = match &overlay_full_type_ids {
                    Some(ids) => Some(ids.slice_axis(1, offset as i64, end as i64)?),
                    None => None,
                };
                {
                    let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                        Error::from_reason("run_paged_vlm_prefill: paged_adapter is None")
                    })?;
                    adapter
                        .record_tokens(chunk_tokens)
                        .map_err(Error::from_reason)?;
                }
                let _hidden = self.run_paged_vlm_prefill_layer_loop(
                    chunk_tokens,
                    &chunk_embeds,
                    pass1_position,
                    pass1_position,
                    layer_kinds,
                    chunk_type_ids.as_ref(),
                )?;
                if let Some(adapter) = self.paged_adapter.as_mut() {
                    adapter
                        .eval_pending_pool_writes()
                        .map_err(Error::from_reason)?;
                }
                if let Some(caches) = self.caches.as_ref() {
                    eval_gemma4_caches(caches)?;
                }
                crate::array::clear_cache();
                pass1_position = pass1_position
                    .checked_add((end - offset) as u32)
                    .ok_or_else(|| {
                        Error::from_reason("run_paged_vlm_prefill: token position overflow")
                    })?;
                offset = end;
            }
        }

        // Pass 2: the FINAL token (length 1).
        let last_idx = (prompt_len - 1) as usize;
        let pass2_tokens = &expanded_tokens[last_idx..];
        let pass2_embeds = inputs_embeds.slice_axis(1, last_idx as i64, prompt_len as i64)?;
        {
            let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                Error::from_reason("run_paged_vlm_prefill: paged_adapter is None")
            })?;
            adapter
                .record_tokens(pass2_tokens)
                .map_err(Error::from_reason)?;
        }
        let mut hidden_states = self.run_paged_vlm_prefill_layer_loop(
            pass2_tokens,
            &pass2_embeds,
            pass1_position,
            pass1_position,
            layer_kinds,
            // Pass 2 is the single final token (seq_len==1); the overlay never
            // applies to a single-token query.
            None,
        )?;
        if let Some(adapter) = self.paged_adapter.as_mut() {
            adapter
                .eval_pending_pool_writes()
                .map_err(Error::from_reason)?;
        }

        hidden_states = self.final_norm.forward(&hidden_states)?;
        crate::models::gemma4::diagnostic::dump_norm(0, "post_final_norm", &hidden_states, None);
        let logits = if let Some(ref head) = self.lm_head {
            head.forward(&hidden_states)?
        } else if let Some(ref w_t) = self.embed_weight_t {
            hidden_states.matmul(w_t)?
        } else {
            let weight = self.embed_tokens.get_weight();
            let weight_t = weight.transpose(Some(&[1, 0]))?;
            hidden_states.matmul(&weight_t)?
        };
        crate::models::gemma4::diagnostic::dump_logits("pre_softcap", &logits);
        let logits = if let Some(cap) = self.config.final_logit_softcapping {
            let cap_arr = MxArray::scalar_float_like(cap, &logits)?;
            let handle = unsafe { mlx_sys::mlx_logit_softcap(logits.handle.0, cap_arr.handle.0) };
            let capped = MxArray::from_handle(handle, "logit_softcap")?;
            crate::models::gemma4::diagnostic::dump_logits("post_softcap", &capped);
            capped
        } else {
            crate::models::gemma4::diagnostic::dump_logits("post_softcap", &logits);
            logits
        };

        let last_seq_len = logits.shape_at(1)?;
        logits
            .slice_axis(1, last_seq_len - 1, last_seq_len)?
            .squeeze(Some(&[0, 1]))
    }

    /// Run one paged decode step: feed `[token_id]` through the model.
    fn run_paged_decode_step(&mut self, token_id: u32) -> Result<MxArray> {
        let first_logical_position = {
            let adapter = self.paged_adapter.as_ref().ok_or_else(|| {
                Error::from_reason("run_paged_decode_step: paged_adapter is None")
            })?;
            adapter.current_token_count()
        };
        {
            let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                Error::from_reason("run_paged_decode_step: paged_adapter dropped")
            })?;
            adapter
                .record_tokens(&[token_id])
                .map_err(Error::from_reason)?;
        }

        let layer_kinds = self.compute_layer_kinds()?;

        let input_ids = MxArray::from_uint32(&[token_id], &[1, 1])?;
        let mut hidden_states = self.embed_tokens.forward(&input_ids)?;
        hidden_states = hidden_states.mul_scalar((self.config.hidden_size as f64).sqrt())?;

        // Compute PLE for the single decode token. Same load-bearing
        // residual contribution as the prefill path — see the comment in
        // `run_paged_prefill_chunk` for why dropping this destroys logits
        // on Gemma4 E2B/E4B.
        let projected_ple_step: Option<MxArray> = if let Some(ref ple) = self.ple {
            let pre_layer_h = hidden_states.clone();
            Some(compute_ple(&input_ids, &pre_layer_h, ple, 1)?)
        } else {
            None
        };

        let has_kv_sharing = self.config.num_kv_shared_layers.is_some_and(|n| n > 0);
        let num_layers = self.layers.len();
        let mut sliding_shared_kv: HashMap<u32, (MxArray, MxArray)> = HashMap::new();
        crate::models::gemma4::diagnostic::set_path("paged");
        #[allow(clippy::needless_range_loop)]
        for layer_idx in 0..num_layers {
            crate::models::gemma4::diagnostic::set_layer(layer_idx);
            let kind = layer_kinds[layer_idx];
            let layer: &Gemma4DecoderLayer = unsafe {
                let ptr = self.layers.as_ptr().add(layer_idx);
                &*ptr
            };
            let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                Error::from_reason("run_paged_decode_step: paged_adapter dropped mid-forward")
            })?;
            let flat_cache: Option<&mut Gemma4LayerCache> =
                if matches!(kind, Gemma4LayerKind::Sliding) {
                    let caches = unsafe {
                        let raw = self.caches.as_mut().ok_or_else(|| {
                            Error::from_reason("run_paged_decode_step: sliding cache slot missing")
                        })? as *mut Vec<Gemma4LayerCache>;
                        &mut *raw
                    };
                    Some(&mut caches[layer_idx])
                } else {
                    None
                };

            let shared_inputs = match kind {
                Gemma4LayerKind::SharedOnGlobal { .. } => {
                    // Anchor's slot already has the new token (it ran
                    // its own forward_paged earlier in this loop, which
                    // wrote K/V via update_keys_values). Read full ctx.
                    let total_ctx = first_logical_position + 1;
                    Some(super::decoder_layer::SharedKvInputs {
                        cache_offset: first_logical_position as i32,
                        total_ctx,
                        keys: None,
                        values: None,
                    })
                }
                Gemma4LayerKind::SharedOnSliding { anchor_layer_idx } => {
                    let (k, v) = sliding_shared_kv.get(&anchor_layer_idx).ok_or_else(|| {
                        Error::from_reason(format!(
                            "run_paged_decode_step: SharedOnSliding anchor {} stash missing",
                            anchor_layer_idx
                        ))
                    })?;
                    let cache_offset = first_logical_position as i32;
                    Some(super::decoder_layer::SharedKvInputs {
                        cache_offset,
                        total_ctx: 0,
                        keys: Some(k),
                        values: Some(v),
                    })
                }
                _ => None,
            };

            let needs_stash = has_kv_sharing
                && matches!(kind, Gemma4LayerKind::Sliding)
                && self.config.should_store_shared_kv(layer_idx);

            // Slice the per-layer PLE input ([B, T, num_layers, ple_dim] →
            // [B, T, ple_dim]).
            let ple_input = projected_ple_step.as_ref().map(|p| {
                p.slice_axis(2, layer_idx as i64, layer_idx as i64 + 1)
                    .and_then(|s| s.squeeze(Some(&[2])))
            });
            let ple_input_ref = match &ple_input {
                Some(Ok(arr)) => Some(arr),
                _ => None,
            };

            hidden_states = layer.forward_paged_or_flat(
                &hidden_states,
                kind,
                adapter,
                first_logical_position,
                /* cached_prefix_len */ 0,
                /* is_prefill */ false,
                /* mask */ None,
                flat_cache,
                ple_input_ref,
                needs_stash,
                shared_inputs,
            )?;

            if needs_stash {
                let caches = unsafe {
                    let raw = self.caches.as_mut().ok_or_else(|| {
                        Error::from_reason(
                            "run_paged_decode_step: sliding cache slot missing post-forward",
                        )
                    })? as *mut Vec<Gemma4LayerCache>;
                    &mut *raw
                };
                if let Some((k, v)) = caches[layer_idx].take_stashed_kv() {
                    sliding_shared_kv.insert(layer_idx as u32, (k, v));
                }
            }
        }

        hidden_states = self.final_norm.forward(&hidden_states)?;
        crate::models::gemma4::diagnostic::dump_norm(0, "post_final_norm", &hidden_states, None);
        let logits = if let Some(ref head) = self.lm_head {
            head.forward(&hidden_states)?
        } else if let Some(ref w_t) = self.embed_weight_t {
            hidden_states.matmul(w_t)?
        } else {
            let weight = self.embed_tokens.get_weight();
            let weight_t = weight.transpose(Some(&[1, 0]))?;
            hidden_states.matmul(&weight_t)?
        };
        crate::models::gemma4::diagnostic::dump_logits("pre_softcap", &logits);
        let logits = if let Some(cap) = self.config.final_logit_softcapping {
            let cap_arr = MxArray::scalar_float_like(cap, &logits)?;
            let handle = unsafe { mlx_sys::mlx_logit_softcap(logits.handle.0, cap_arr.handle.0) };
            let capped = MxArray::from_handle(handle, "logit_softcap")?;
            crate::models::gemma4::diagnostic::dump_logits("post_softcap", &capped);
            capped
        } else {
            crate::models::gemma4::diagnostic::dump_logits("post_softcap", &logits);
            logits
        };
        Ok(logits)
    }

    /// Replay cached prefix tokens to reconstruct the flat sliding caches.
    /// Used to bring sliding-layer state up to the paged cache's
    /// `cached_prefix_len` boundary before the main `run_paged_prefill_chunk`
    /// continues with the suffix.
    ///
    /// Global layers run as read-only Q projections against their existing
    /// paged K/V. That keeps hidden states flowing into later sliding layers
    /// without rebuilding throwaway global K/V for the cached prefix.
    fn run_sliding_only_prefill(
        &mut self,
        prefix_tokens: &[u32],
        first_logical_position: u32,
        layer_kinds: &[Gemma4LayerKind],
    ) -> Result<()> {
        if prefix_tokens.is_empty() {
            return Ok(());
        }
        let configured_chunk_size = crate::array::paged_prefill_chunk_size();
        let num_query_heads = u32::try_from(self.config.num_attention_heads).map_err(|_| {
            Error::from_reason(format!(
                "Gemma4 sliding restore invalid num_attention_heads={}",
                self.config.num_attention_heads
            ))
        })?;
        let global_head_size =
            u32::try_from(self.config.effective_head_dim(true)).map_err(|_| {
                Error::from_reason(format!(
                    "Gemma4 sliding restore invalid global head_dim={}",
                    self.config.effective_head_dim(true)
                ))
            })?;
        let paged_attention_enabled = gemma4_paged_prefill_paged_attention_enabled_for_chunking();
        let mut chunk_plan = gemma4_paged_prefill_body_chunk_plan(
            configured_chunk_size,
            prefix_tokens.len(),
            first_logical_position,
            num_query_heads,
            global_head_size,
            paged_attention_enabled,
        )?;
        gemma4_coalesce_single_token_restore_chunks(&mut chunk_plan);

        let trace_enabled = inference_trace_enabled();
        let total_trace_start = trace_enabled.then(std::time::Instant::now);
        if trace_enabled {
            let first_chunk_size = chunk_plan.first().map(|chunk| chunk.len).unwrap_or(0);
            let min_chunk_size = chunk_plan.iter().map(|chunk| chunk.len).min().unwrap_or(0);
            let max_chunk_size = chunk_plan.iter().map(|chunk| chunk.len).max().unwrap_or(0);
            let aux_caps = chunk_plan
                .iter()
                .filter(|chunk| chunk.capped_by_v2_aux_limit)
                .count();
            write_inference_trace(format_args!(
                "[MLX_TRACE] gemma4 sliding_prefix_restore_start first_position={} prefix_tokens={} chunks={} chunk_size={} min_chunk_size={} max_chunk_size={} configured_chunk_size={} dynamic_v2_aux_caps={} path=paged_global_readonly",
                first_logical_position,
                prefix_tokens.len(),
                chunk_plan.len(),
                first_chunk_size,
                min_chunk_size,
                max_chunk_size,
                configured_chunk_size,
                aux_caps
            ));
        }

        let total_chunks = chunk_plan.len();
        for (chunk_idx, chunk_plan) in chunk_plan.iter().enumerate() {
            let chunk_end = chunk_plan
                .start
                .checked_add(chunk_plan.len)
                .ok_or_else(|| Error::from_reason("Gemma4 sliding restore chunk end overflow"))?;
            let chunk = prefix_tokens
                .get(chunk_plan.start..chunk_end)
                .ok_or_else(|| {
                    Error::from_reason("Gemma4 sliding restore chunk plan out of range")
                })?;
            let chunk_trace_start = trace_enabled.then(std::time::Instant::now);
            if trace_enabled {
                write_inference_trace(format_args!(
                    "[MLX_TRACE] gemma4 sliding_prefix_restore_chunk_start chunk={}/{} first_position={} tokens={} capped_by_v2_aux_limit={}",
                    chunk_idx + 1,
                    total_chunks,
                    chunk_plan.first_position,
                    chunk.len(),
                    chunk_plan.capped_by_v2_aux_limit
                ));
            }

            self.run_sliding_prefix_restore_layer_loop(
                chunk,
                chunk_plan.first_position,
                layer_kinds,
            )?;

            if let Some(caches) = self.caches.as_ref() {
                eval_gemma4_caches(caches)?;
            }
            crate::array::clear_cache();

            if trace_enabled {
                write_inference_trace(format_args!(
                    "[MLX_TRACE] gemma4 sliding_prefix_restore_chunk_done chunk={}/{} next_position={} elapsed_ms={:.1}",
                    chunk_idx + 1,
                    total_chunks,
                    chunk_plan.first_position + chunk.len() as u32,
                    chunk_trace_start.map(elapsed_ms).unwrap_or(0.0)
                ));
            }
        }

        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] gemma4 sliding_prefix_restore_done first_position={} prefix_tokens={} chunks={} elapsed_ms={:.1}",
                first_logical_position,
                prefix_tokens.len(),
                total_chunks,
                total_trace_start.map(elapsed_ms).unwrap_or(0.0)
            ));
        }
        Ok(())
    }

    fn run_sliding_prefix_restore_layer_loop(
        &mut self,
        chunk_tokens: &[u32],
        first_logical_position: u32,
        layer_kinds: &[Gemma4LayerKind],
    ) -> Result<()> {
        let chunk_len = chunk_tokens.len() as u32;
        if chunk_len == 0 {
            return Ok(());
        }

        let input_ids = MxArray::from_uint32(chunk_tokens, &[1, chunk_len as i64])?;
        let mut hidden_states = self.embed_tokens.forward(&input_ids)?;
        hidden_states = hidden_states.mul_scalar((self.config.hidden_size as f64).sqrt())?;

        let projected_ple_prefix: Option<MxArray> = if let Some(ref ple) = self.ple {
            let pre_layer_h = hidden_states.clone();
            Some(compute_ple(
                &input_ids,
                &pre_layer_h,
                ple,
                chunk_len as i64,
            )?)
        } else {
            None
        };

        let sliding_offset = self
            .caches
            .as_ref()
            .and_then(|caches| {
                caches
                    .iter()
                    .enumerate()
                    .find(|(i, _)| self.config.is_sliding_layer(*i))
                    .map(|(_, c)| c.get_offset())
            })
            .unwrap_or(0);
        if sliding_offset != first_logical_position as i32 {
            return Err(Error::from_reason(format!(
                "Gemma4 sliding restore cache offset mismatch: expected {} got {}",
                first_logical_position, sliding_offset
            )));
        }

        let seq_len = chunk_len as i64;
        let sliding_window = self.config.sliding_window as i64;
        let sliding_mask_offset =
            sliding_mask_offset_for_chunk(seq_len, sliding_offset, sliding_window);
        let sliding_mask = sliding_mask_offset
            .map(|offset| create_sliding_mask(seq_len, offset, sliding_window))
            .transpose()?;

        let has_kv_sharing = self.config.num_kv_shared_layers.is_some_and(|n| n > 0);
        let total_ctx = first_logical_position
            .checked_add(chunk_len)
            .ok_or_else(|| Error::from_reason("Gemma4 sliding restore total_ctx overflow"))?;
        let mut sliding_shared_kv: HashMap<u32, (MxArray, MxArray)> = HashMap::new();

        #[allow(clippy::needless_range_loop)]
        for layer_idx in 0..self.layers.len() {
            crate::models::gemma4::diagnostic::set_layer(layer_idx);
            let kind = layer_kinds[layer_idx];
            let layer: &Gemma4DecoderLayer = unsafe {
                let ptr = self.layers.as_ptr().add(layer_idx);
                &*ptr
            };
            let ple_input = projected_ple_prefix.as_ref().map(|p| {
                p.slice_axis(2, layer_idx as i64, layer_idx as i64 + 1)
                    .and_then(|s| s.squeeze(Some(&[2])))
            });
            let ple_input_ref = match &ple_input {
                Some(Ok(arr)) => Some(arr),
                _ => None,
            };

            let needs_stash = has_kv_sharing
                && matches!(kind, Gemma4LayerKind::Sliding)
                && self.config.should_store_shared_kv(layer_idx);

            match kind {
                Gemma4LayerKind::Sliding => {
                    let caches = unsafe {
                        let raw = self.caches.as_mut().ok_or_else(|| {
                            Error::from_reason(
                                "run_sliding_prefix_restore_layer_loop: sliding cache missing",
                            )
                        })? as *mut Vec<Gemma4LayerCache>;
                        &mut *raw
                    };
                    hidden_states = layer.forward(
                        &hidden_states,
                        sliding_mask.as_ref(),
                        Some(&mut caches[layer_idx]),
                        ple_input_ref,
                        needs_stash,
                    )?;
                    if needs_stash && let Some((k, v)) = caches[layer_idx].take_stashed_kv() {
                        sliding_shared_kv.insert(layer_idx as u32, (k, v));
                    }
                }
                Gemma4LayerKind::GlobalPaged { paged_idx } => {
                    let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                        Error::from_reason(
                            "run_sliding_prefix_restore_layer_loop: paged_adapter missing",
                        )
                    })?;
                    hidden_states = layer.forward_paged_or_flat(
                        &hidden_states,
                        Gemma4LayerKind::SharedOnGlobal {
                            anchor_paged_idx: paged_idx,
                        },
                        adapter,
                        first_logical_position,
                        first_logical_position,
                        /* is_prefill */ true,
                        /* mask */ None,
                        /* flat_cache */ None,
                        ple_input_ref,
                        /* needs_stash */ false,
                        Some(super::decoder_layer::SharedKvInputs {
                            cache_offset: first_logical_position as i32,
                            total_ctx,
                            keys: None,
                            values: None,
                        }),
                    )?;
                }
                Gemma4LayerKind::SharedOnGlobal { .. } => {
                    let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                        Error::from_reason(
                            "run_sliding_prefix_restore_layer_loop: paged_adapter missing",
                        )
                    })?;
                    hidden_states = layer.forward_paged_or_flat(
                        &hidden_states,
                        kind,
                        adapter,
                        first_logical_position,
                        first_logical_position,
                        /* is_prefill */ true,
                        /* mask */ None,
                        /* flat_cache */ None,
                        ple_input_ref,
                        /* needs_stash */ false,
                        Some(super::decoder_layer::SharedKvInputs {
                            cache_offset: first_logical_position as i32,
                            total_ctx,
                            keys: None,
                            values: None,
                        }),
                    )?;
                }
                Gemma4LayerKind::SharedOnSliding { anchor_layer_idx } => {
                    let (k, v) = sliding_shared_kv.get(&anchor_layer_idx).ok_or_else(|| {
                        Error::from_reason(format!(
                            "run_sliding_prefix_restore_layer_loop: SharedOnSliding anchor {} stash missing",
                            anchor_layer_idx
                        ))
                    })?;
                    hidden_states = layer.forward_paged_or_flat(
                        &hidden_states,
                        kind,
                        self.paged_adapter.as_mut().ok_or_else(|| {
                            Error::from_reason(
                                "run_sliding_prefix_restore_layer_loop: paged_adapter missing",
                            )
                        })?,
                        first_logical_position,
                        first_logical_position,
                        /* is_prefill */ true,
                        /* mask */ None,
                        /* flat_cache */ None,
                        ple_input_ref,
                        /* needs_stash */ false,
                        Some(super::decoder_layer::SharedKvInputs {
                            cache_offset: first_logical_position as i32,
                            total_ctx: 0,
                            keys: Some(k),
                            values: Some(v),
                        }),
                    )?;
                }
            }

            crate::array::maybe_eval_clear_for_paged_prefill_layer(layer_idx, &hidden_states)?;
        }

        Ok(())
    }

    // =================================================================
    // Session API (Step 5c of the chat-session refactor).
    //
    // Gemma4's wire format uses `<turn|>` / `<|turn>` delimiters with
    // "model" as the assistant role (not ChatML / Qwen3.5). The session
    // primitives here mirror the Qwen3 / LFM2 surface but with Gemma4's
    // wire format baked into the delta text builders.
    //
    // Image-change invariant: `chat_session_continue` / `_tool` run on
    // top of the live caches, so they MUST be text-only. If the session
    // currently carries image state (i.e. `cached_image_key.is_some()`)
    // we surface an `IMAGE_CHANGE_REQUIRES_SESSION_RESTART:`-prefixed
    // error so the TS `ChatSession` layer can route the caller back
    // through a fresh `chat_session_start`.
    // =================================================================

    /// Resolve the token id for Gemma4's `<turn|>` turn terminator.
    ///
    /// Used as the `eos_token_id` in the session-start path so the
    /// decode loop leaves the caches on a clean `<turn|>` boundary that
    /// subsequent `chat_session_continue_sync` /
    /// `chat_session_continue_tool_sync` calls can append a raw delta on
    /// top of. Computed on demand rather than cached — encoding a
    /// special token is O(1) and the cost is trivial relative to a
    /// chat turn.
    pub(crate) fn turn_end_id(&self) -> Result<u32> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))?;
        let ids = tokenizer.encode_sync("<turn|>", Some(false))?;
        if ids.is_empty() {
            return Err(Error::from_reason(
                "Tokenizer encoded <turn|> to empty id vector",
            ));
        }
        if ids.len() != 1 {
            return Err(Error::from_reason(format!(
                "Tokenizer encoded <turn|> to {} tokens; expected 1",
                ids.len()
            )));
        }
        Ok(ids[0])
    }

    /// Vision whole-turn dispatch for the engine's
    /// [`ChatBackend::vision_turn`] probe. Only fresh turns carry
    /// images (the engine's delta inputs are text-only by construction
    /// and the delta image guard rejects image-holding sessions), so
    /// the paged cores cold-start unconditionally —
    /// `verify_cache_prefix(.., has_images = true)` forces a miss.
    ///
    /// Image turns run ONLY on the block-paged KV backend. A model with
    /// no paged adapter (explicit `use_block_paged_cache: false`, a
    /// non-Metal build, or paged init failure) has no vision path and
    /// returns an error instead of silently falling back.
    fn vision_chat_turn(&mut self, args: &mut WholeTurnArgs<'_>) -> Result<TurnOutput> {
        if self.paged_adapter.is_none() {
            return Err(Error::from_reason(
                "gemma4 image turns require the block-paged KV backend; the model was loaded \
                 without a paged adapter (use_block_paged_cache=false, non-Metal build, or paged \
                 init failed)",
            ));
        }
        let tokenizer = args.tokenizer.clone();
        match (args.sink, args.cancelled) {
            (Some(sink), Some(cancelled)) => {
                self.vision_paged_turn_stream_core(
                    args.tokens,
                    args.images,
                    args.audio,
                    &tokenizer,
                    args.config,
                    args.eos_id,
                    sink,
                    cancelled,
                )?;
                Ok(TurnOutput::Streamed)
            }
            _ => {
                let result = self.vision_paged_turn_sync_core(
                    args.tokens,
                    args.images,
                    args.audio,
                    &tokenizer,
                    args.config,
                    args.eos_id,
                )?;
                Ok(TurnOutput::Complete(Box::new(result)))
            }
        }
    }
}

/// Eager flat decode stepper for one gemma4 turn
/// ([`ChatBackend::begin_decode`]). Runs the flat decode-loop step body:
/// `diagnostic::set_step(step)` before every forward (the
/// `MLX_DEBUG_GEMMA4_DUMP` per-step dump), `forward_inner` over the live
/// session caches, async-eval of the sampled token only (gemma4 never
/// async-evals the logits).
pub(crate) struct Gemma4Decode<'a> {
    inner: &'a mut Gemma4Inner,
    /// Diagnostic step counter. The engine loop has no step index in the
    /// `DecodeStep` seam, so the stepper carries its own 0-based sequence
    /// to feed `set_step`.
    step: i32,
}

impl DecodeStep for Gemma4Decode<'_> {
    fn forward(&mut self, input_ids: &MxArray) -> Result<(MxArray, bool)> {
        let inner = &mut *self.inner;
        let caches = inner
            .caches
            .as_mut()
            .ok_or_else(|| Error::from_reason("Gemma4 decode: caches missing"))?;
        crate::models::gemma4::diagnostic::set_step(self.step);
        self.step += 1;
        let logits = forward_inner(
            input_ids,
            &inner.embed_tokens,
            &inner.layers,
            caches,
            &inner.final_norm,
            &inner.lm_head,
            inner.embed_weight_t.as_ref(),
            inner.ple.as_ref(),
            &inner.config,
        )?;
        // `true` requests the engine's `squeeze(Some(&[1]))`: the eager
        // forward returns `[1, 1, vocab]`.
        Ok((logits, true))
    }

    fn eval_step(&mut self, next_token: &MxArray, _logits: &MxArray, _budget_forced: bool) {
        MxArray::async_eval_arrays(&[next_token]);
    }

    fn materialize_final(&mut self, token_id: u32) -> Result<()> {
        // LENGTH-exit only (the engine gates the call): run ONE more
        // `forward_inner` for the final committed token so its K/V lands in
        // the live session caches, then DISCARD the logits. This makes the
        // per-layer cache offsets equal the keep-all-on-length saved
        // history. No sample / push / emit. Like the paged override, this
        // deliberately does NOT fire a sliding decode-boundary checkpoint.
        let inner = &mut *self.inner;
        let caches = inner
            .caches
            .as_mut()
            .ok_or_else(|| Error::from_reason("Gemma4 materialize_final: caches missing"))?;
        let input_ids = MxArray::from_int32(&[token_id as i32], &[1, 1])?;
        crate::models::gemma4::diagnostic::set_step(self.step);
        self.step += 1;
        let _logits = forward_inner(
            &input_ids,
            &inner.embed_tokens,
            &inner.layers,
            caches,
            &inner.final_norm,
            &inner.lm_head,
            inner.embed_weight_t.as_ref(),
            inner.ple.as_ref(),
            &inner.config,
        )?;
        Ok(())
    }
}

/// Paged decode stepper for gemma4 (pure-eager — no compiled path, so no
/// lifecycle/reset guard fields). Drives
/// [`crate::engine::decode::run_decode_loop`] through
/// [`Gemma4Inner::run_paged_decode_step`], advancing the per-instance
/// sliding-window KV checkpoint machinery as a side effect of each
/// committed decode step.
pub(crate) struct Gemma4PagedDecode<'a> {
    /// Diagnostic step counter, fed to `set_step` before every paged
    /// forward. The engine loop has no step index in the `DecodeStep`
    /// seam, so the stepper carries its own.
    step: i32,
    inner: &'a mut Gemma4Inner,
}

impl DecodeStep for Gemma4PagedDecode<'_> {
    fn forward(&mut self, input_ids: &MxArray) -> Result<(MxArray, bool)> {
        // The loop hands the already-extracted token via
        // `forward_with_token`; recover it here from the `[1, 1]` input for
        // the bare `forward` contract (idempotent eval with the loop-top
        // `y.eval()`).
        let token_id = input_ids.item_at_int32(0)? as u32;
        self.forward_with_token(input_ids, token_id)
    }

    fn forward_with_token(
        &mut self,
        _input_ids: &MxArray,
        token_id: u32,
    ) -> Result<(MxArray, bool)> {
        crate::models::gemma4::diagnostic::set_step(self.step);
        self.step += 1;
        let trace_enabled = inference_trace_enabled();
        // `run_paged_decode_step` records the token in the adapter at its
        // top (BEFORE the forward), then returns `[1, 1, vocab]`.
        let logits = self.inner.run_paged_decode_step(token_id)?;
        // The sliding-window decode-boundary checkpoint runs RIGHT AFTER
        // the forward, reading the adapter's post-record cursor. It must
        // NOT move to `maintain_cache` (which runs at the loop TOP, before
        // this forward, so it would read a stale cursor) — see the engine
        // loop ordering. Fallible: a checkpoint/eval error aborts the turn.
        self.inner
            .maybe_remember_gemma4_sliding_decode_boundary_checkpoint("paged", trace_enabled)?;
        // `run_paged_decode_step` returns `[1, 1, vocab]`; `true` requests
        // the engine's squeeze of axis 1 (the eager convention).
        Ok((logits, true))
    }

    fn eval_step(&mut self, next_token: &MxArray, _logits: &MxArray, _budget_forced: bool) {
        // Async-eval the sampled token only (gemma4 never async-evals the
        // logits); the loop-top `y.eval()` forces materialization next
        // iteration.
        MxArray::async_eval_arrays(&[next_token]);
    }

    fn maintain_cache(&mut self, step: i32) {
        // Paged cadence — the per-step
        // `maybe_clear_cache_for_paged_step(step)`.
        crate::array::maybe_clear_cache_for_paged_step(step);
    }

    fn materialize_final(&mut self, token_id: u32) -> Result<()> {
        // LENGTH-exit only (the engine gates the call): run ONE more
        // `run_paged_decode_step` for the final committed token so its K/V
        // lands in the paged adapter, then DISCARD the logits. The adapter's
        // `request_tokens()` / cursor advances by exactly 1 to equal the
        // saved keep-all history.
        //
        // Deliberately does NOT fire the sliding decode-boundary checkpoint:
        // the final length-exit token is not checkpointed at the boundary.
        // The history checkpoint (in `save_paged_history`) covers the kept
        // history instead.
        let _logits = self.inner.run_paged_decode_step(token_id)?;
        Ok(())
    }
    // end_decode → default Ok(()).
}

/// gemma4 paged prefix state — the effective prefix/suffix split from
/// `prepare_gemma4_paged_turn`. `effective_cached_prefix_len` is the
/// POST-suppression length (the prepare may zero the plan's cached_len
/// when a large sliding-prefix reuse is suppressed). `full_tokens`
/// carries the entire prompt: the engine hands `paged_prefill` only the
/// suffix, but `run_paged_prefill_chunk` re-prefills the sliding layers
/// from the prompt start, and `sliding_primed_prefix_len` tells it how
/// much of the cached prefix the sliding caches already hold.
pub(crate) struct Gemma4PrefixState {
    effective_cached_prefix_len: usize,
    suffix_len: usize,
    sliding_primed_prefix_len: u32,
    full_tokens: Vec<u32>,
}

impl PagedPrefix for Gemma4PrefixState {
    fn effective_cached_prefix_len(&self) -> usize {
        self.effective_cached_prefix_len
    }
    fn suffix_len(&self) -> usize {
        self.suffix_len
    }
}

impl PagedBackend for Gemma4Inner {
    type PagedDecode<'a>
        = Gemma4PagedDecode<'a>
    where
        Self: 'a;
    type PrefixState = Gemma4PrefixState;

    fn prime_prefix_state(
        &mut self,
        plan: &[u32],
        reuse_cache: bool,
        _block_size: usize,
        _extra_keys: &[u64],
        _cache_salt: u64,
    ) -> Result<Self::PrefixState> {
        let trace_enabled = inference_trace_enabled();
        let total_budget = plan.len() as u32;
        // Per-turn seq_id: the adapter is single-request and the prepare's
        // warm-continue / cold-reset arms make the previous seq_id
        // irrelevant.
        let seq_id: u32 = 0;
        // The prepare runs the adapter's warm-continue / cold-reset arms,
        // applies the vLLM `max_cache_hit_tokens = total_budget - 1` cap,
        // and may ZERO the cached prefix mid-prepare when a large
        // sliding-prefix reuse is suppressed — so the EFFECTIVE
        // post-suppression length surfaces here (never the plan's raw
        // cached_len).
        let prep = self.prepare_gemma4_paged_turn(
            "paged",
            plan,
            reuse_cache,
            total_budget,
            seq_id,
            trace_enabled,
        )?;
        Ok(Gemma4PrefixState {
            effective_cached_prefix_len: prep.cached_prefix_len as usize,
            suffix_len: prep.suffix_len as usize,
            sliding_primed_prefix_len: prep.sliding_primed_prefix_len,
            // Sliding-layer re-prefill needs the FULL prompt, not just the
            // suffix the engine passes to `paged_prefill`.
            full_tokens: plan.to_vec(),
        })
    }

    fn paged_prefill(
        &mut self,
        suffix_tokens: &[u32],
        prefix: &Self::PrefixState,
        _stream: Stream,
    ) -> Result<MxArray> {
        // Mark the diagnostic step as -1 (prefill) before the forward
        // (diagnostic-only). The engine fires the post-prefill
        // `synchronize_and_clear_cache` AFTER this returns.
        crate::models::gemma4::diagnostic::set_step(-1);
        self.run_paged_prefill_chunk(
            &prefix.full_tokens,
            suffix_tokens,
            prefix.effective_cached_prefix_len as u32,
            prefix.sliding_primed_prefix_len,
        )
    }

    fn begin_paged_decode(&mut self, _setup: &PagedTurnSetup<'_>) -> Result<Self::PagedDecode<'_>> {
        Ok(Gemma4PagedDecode {
            step: 0,
            inner: self,
        })
    }

    fn finalize_paged_turn(&mut self, reuse_cache: bool) {
        // Terminal lifecycle for the paged turn. Success: keep the request
        // live across turns when reuse is on so the next turn builds on the
        // partial trailing block's live K/V; otherwise register full blocks
        // for reuse + release. Infallible (`let _ =` every call — a teardown
        // failure must not mask the turn result).
        if let Some(adapter) = self.paged_adapter.as_mut() {
            if reuse_cache {
                let _ = adapter.finalize_turn_keep_live(&[], 0);
            } else {
                let _ = adapter.register_full_blocks_for_reuse(&[], 0);
                let _ = adapter.release_request();
            }
        }
    }

    fn abort_paged_turn(&mut self) {
        // Error-path teardown: release the request fully — partial
        // block_table state is unsafe to keep around. Infallible (`let _ =`).
        if let Some(adapter) = self.paged_adapter.as_mut() {
            let _ = adapter.release_request();
        }
    }

    fn save_paged_history(
        &mut self,
        save_tokens: &[u32],
        generated: &[u32],
        keep_all: bool,
        reuse_cache: bool,
    ) -> Result<()> {
        // Save token history ONLY — the adapter's pool owns the K/V.
        // `keep_all` is the flat rule (engine: `finish_reason ==
        // "length"`); when it is false the terminal stop token is dropped
        // (DROP-LAST trim). The engine reconciles `request_tokens()` to this
        // same trimmed history via `reconcile_paged_request_tokens` BEFORE
        // finalize, so the adapter and the saved history stay aligned for
        // the next turn's warm-continue.
        if reuse_cache {
            let mut full_history = save_tokens.to_vec();
            let history_tokens = if keep_all || generated.is_empty() {
                generated
            } else {
                &generated[..generated.len() - 1]
            };
            full_history.extend_from_slice(history_tokens);
            self.cached_token_history = full_history;
            // A text save is text-only, so drop any stale media key — mirrors
            // the flat `save_cache_state` fresh-turn clear. On a warm reuse the
            // delta image guard already kept these `None`, so this is a no-op;
            // on a fresh start it clears a key a prior media turn left behind.
            self.cached_image_key = None;
            self.cached_audio_key = None;
            // Sliding-window warm-continue checkpoint keyed on the freshly
            // set history (post-reconcile `request_tokens()` == the trimmed
            // history). Fallible: a checkpoint/eval error aborts the turn so
            // reusable state is never published without a materialized
            // checkpoint.
            let history_for_checkpoint = self.cached_token_history.clone();
            let _store_trace =
                self.remember_gemma4_sliding_history_checkpoint(&history_for_checkpoint)?;
        } else {
            self.cached_token_history.clear();
            self.sliding_last_history_checkpoint = None;
            // Fresh paged start: a text turn holds no media, so clear any media
            // key a prior turn on this reused model left set (mirrors the flat
            // `save_cache_state` fresh-turn clear). Without the audio clear a
            // text-only start over a model whose last turn was audio would leave
            // `cached_audio_key` stale and the delta image guard would wrongly
            // force an "audio state" restart on the text-only session.
            self.cached_image_key = None;
            self.cached_audio_key = None;
        }
        Ok(())
    }

    fn reconcile_paged_request_tokens(
        &mut self,
        prompt_len: usize,
        generated: &[u32],
        keep_all: bool,
    ) -> bool {
        // Perf-parity warm-continue restore (see the trait doc). The
        // pipelined decode loop records the stop token into the adapter
        // (its forward ran at the loop top BEFORE the stop-check), but the
        // saved history DROPS it on a non-length exit. Roll the adapter back
        // to the to-be-saved history length so `request_tokens()` matches
        // the persisted history. `history_len` uses the EXACT same trim as
        // `save_paged_history`; `saturating_sub` makes it a no-op on a length
        // exit (`materialize_final` already recorded the final token) and on
        // a final-step stop (forward never ran).
        let Some(adapter) = self.paged_adapter.as_mut() else {
            return true;
        };
        let history_len = if keep_all || generated.is_empty() {
            generated.len()
        } else {
            generated.len() - 1
        };
        let target_len = prompt_len + history_len;
        let surplus = adapter.request_tokens().len().saturating_sub(target_len);
        if surplus > 0
            && let Err(e) = adapter.rollback_last_tokens(surplus as u32)
        {
            tracing::warn!(
                target: "mlx_core::gemma4::paged",
                "reconcile_paged_request_tokens: rollback_last_tokens({surplus}) failed \
                 (finalize releases the request; next turn cold-prefills): {e}",
            );
            return false;
        }
        true
    }
}

impl ChatBackend for Gemma4Inner {
    fn tokenizer(&self) -> Result<Arc<Qwen3Tokenizer>> {
        self.tokenizer
            .clone()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))
    }

    fn family_name(&self) -> &'static str {
        "gemma4"
    }

    fn session_eos_id(&self, _tok: &Qwen3Tokenizer) -> Result<u32> {
        // Gemma4 stops on its `<turn|>` turn terminator, not `<|im_end|>`.
        self.turn_end_id()
    }

    fn policy(&self) -> engine::ThinkingPolicy {
        // Legacy gemma4 had NO think-budget machinery: its decode loops
        // never tracked reasoning tokens (`reasoning_tokens: 0` on every
        // result) and never forced `</think>`. `ThinkingPolicy::None`
        // resolves to `{enabled:false, budget:None}`, keeping the
        // engine's `ReasoningTracker` permanently outside a think block —
        // the reasoning SEGMENTATION still happens downstream in
        // `parse_gemma4_output` / `Gemma4StreamParser`, which key on
        // `<|channel>` markers, not the tracker.
        engine::ThinkingPolicy::None
    }

    fn resolve_params(&self, config: &ChatConfig) -> ChatParams {
        let mut p = engine::extract_chat_params(config);
        // Fold the MODEL-config sampling defaults in; unset → T=0 greedy.
        // The engine's `sampling::sample` argmax fast path at T=0 is the
        // greedy argmax.
        p.sampling_config = make_sampling_config(config, &self.config);
        // gemma4 treats the penalty fields as no-ops. Neutralize so the
        // engine's `apply_all_penalties` skips all penalty work
        // structurally.
        p.repetition_penalty = 1.0;
        p.presence_penalty = 0.0;
        p.frequency_penalty = 0.0;
        // gemma4 ALWAYS returns Some(PerformanceMetrics), regardless of
        // `config.report_performance`.
        p.report_performance = true;
        // gemma4 never suppresses reasoning deltas at the loop level
        // (`include_reasoning` is a no-op here; the stream parser routes
        // channel segments itself). Defensive: pin `true` so the engine's
        // emitter gate can never suppress.
        p.include_reasoning = true;
        p
    }

    /// Template default path == the engine default; template-less
    /// checkpoints take gemma4's manual `<|turn>` wire-format fallback.
    /// A single no-template `enable_thinking` error string covers all
    /// entry points.
    fn render_prompt(
        &self,
        tok: &Qwen3Tokenizer,
        messages: &[ChatMessage],
        config: &ChatConfig,
    ) -> Result<Vec<u32>> {
        let enable_thinking = engine::resolve_enable_thinking(config);
        // Try the tokenizer's chat template if available (handles role
        // mapping, special tokens, and variant-specific formatting
        // automatically). Fall back to manual Gemma4 format if no
        // template was loaded.
        if tok.has_chat_template() {
            return tok.apply_chat_template_sync(
                messages,
                Some(true), // add_generation_prompt
                config.tools.as_deref(),
                enable_thinking, // None = template default
            );
        }
        // Manual fallback: thinking control requires a chat template
        if enable_thinking == Some(true) {
            return Err(Error::from_reason(
                "enable_thinking=true requires a chat template (not found in tokenizer_config.json or chat_template.jinja)",
            ));
        }
        // Manual Gemma4 format matching the canonical template.
        // Role mapping: "assistant" → "model", "developer" → "system".
        // Tool calls serialized as <|tool_call>call:name{args}<tool_call|>.
        // Tool responses wrapped in <|tool_response>...<tool_response|>.
        // BOS prepended explicitly (matching {{ bos_token }} in template).
        let mut prompt_text = String::from("<bos>");
        for msg in messages {
            let role = match msg.role.as_str() {
                "assistant" => "model",
                "developer" => "system",
                other => other,
            };

            // All roles (including "tool") use the same <|turn>role\n...<turn|>\n format.
            // This matches the canonical tokenizer behavior verified against HF.
            {
                prompt_text.push_str(&format!("<|turn>{}\n", role));

                // Emit tool calls for assistant/model messages
                if let Some(ref tool_calls) = msg.tool_calls {
                    for tc in tool_calls {
                        prompt_text.push_str(&format!(
                            "<|tool_call>call:{}{{{}}}<tool_call|>",
                            tc.name,
                            json_args_to_gemma4_dsl(&escape_gemma4_content(&tc.arguments))
                        ));
                    }
                }

                // Emit content (sanitized to prevent control-token injection)
                prompt_text.push_str(&escape_gemma4_content(&msg.content));
                prompt_text.push_str("<turn|>\n");
            }
        }
        prompt_text.push_str("<|turn>model\n");
        tok.encode_sync(&prompt_text, Some(false))
    }

    fn render_continue_delta(
        &self,
        tok: &Qwen3Tokenizer,
        user_message: &str,
        config: &ChatConfig,
    ) -> Result<Vec<u32>> {
        // Subject the session path to the same sanitization as the
        // session start path so role/content injection guards stay
        // uniform across all entry points.
        let synthetic = engine::build_synthetic_user_message(user_message);
        let sanitized = Qwen3Tokenizer::sanitize_messages_public(std::slice::from_ref(&synthetic));
        let sanitized_user = &sanitized[0].content;

        let enable_thinking = engine::resolve_enable_thinking(config);
        let delta_text = build_gemma4_continue_delta_text(sanitized_user, enable_thinking);
        tok.encode_sync(&delta_text, Some(false))
    }

    fn render_tool_delta(
        &self,
        tok: &Qwen3Tokenizer,
        tool_call_id: &str,
        content: &str,
        is_error: Option<bool>,
        config: &ChatConfig,
    ) -> Result<Vec<u32>> {
        let enable_thinking = engine::resolve_enable_thinking(config);
        let delta_text =
            build_gemma4_tool_delta_text(tool_call_id, content, enable_thinking, is_error);
        tok.encode_sync(&delta_text, Some(false))
    }

    fn cached_token_history(&self) -> &[u32] {
        &self.cached_token_history
    }

    fn reset_caches(&mut self, scope: ResetScope) -> Result<()> {
        // Legacy miss branch ran `reset_caches_sync()? +
        // init_caches_sync()?` back-to-back (the flat prefill needs live
        // caches); the explicit command reset only cleared (caches stay
        // `None` until the next turn's lazy init).
        self.reset_caches_sync()?;
        if scope == ResetScope::PrefixMiss {
            self.init_caches_sync()?;
        }
        // The EXPLICIT command reset must restore a fully cold state.
        // gemma4's flat reset path (`reset_caches_sync`) never touches the
        // paged adapter, so a prior turn's request stays live AND its full
        // blocks stay content-addressed in the per-instance BlockAllocator's
        // prefix cache. A reset-then-rerun of the same prompt would then take
        // the prefix-hit suffix-prefill path (via `find_longest_cache_hit`
        // inside `prepare_gemma4_paged_turn`) — a different bf16 reduction
        // order than the cold full prefill, enough to flip a greedy
        // near-tie.
        // `release_request_and_purge_prefix_cache` releases the live request
        // (the release gemma4's reset otherwise skips) AND purges every
        // prefix-cache entry. The turn-internal `PrefixMiss` reset keeps the
        // prefix cache (cross-request block reuse after a history miss is the
        // paged design's entire point).
        if scope == ResetScope::Command
            && let Some(adapter) = self.paged_adapter.as_mut()
        {
            adapter
                .release_request_and_purge_prefix_cache()
                .map_err(|e| {
                    Error::from_reason(format!(
                        "gemma4 reset_caches: paged prefix-cache purge failed: {e}"
                    ))
                })?;
        }
        Ok(())
    }

    /// Prefix-reuse check. The engine routes every image-bearing turn
    /// through `vision_turn` BEFORE this check, so only the session-side
    /// image gate (`cached_image_key.is_some()` → miss) is needed here;
    /// there is no `has_images` parameter.
    ///
    /// All-or-nothing: returns `0` or `cached.len()` (exact-match falls
    /// through the `hit == tokens.len()` branch in the session core to
    /// the miss/reset path — gemma4's sliding-window cache has no "rewind
    /// by one" primitive).
    fn verify_cache_prefix(&self, tokens: &[u32], reuse_cache: bool) -> usize {
        if !reuse_cache {
            return 0;
        }
        // Text-only prefix reuse: force a miss whenever the cached
        // session holds image or audio state UNLESS the media turn is
        // continuable (kept-live + sliding checkpoint at the full prefix). This
        // keeps prefix reuse strictly aligned with text-only sessions and
        // sidesteps the media-key coordination the Qwen3.5 shared helper
        // handles, while letting a continuable media session reuse an
        // exactly-cached prefix.
        if (self.cached_image_key.is_some() || self.cached_audio_key.is_some())
            && !self.media_session_continuable
        {
            return 0;
        }
        // The live KV caches must exist — `cached_token_history` can
        // carry stale content after a prior `reset_caches_sync` if any
        // caller forgot to also clear it, so both must line up.
        if self.caches.is_none() {
            return 0;
        }
        let cached = &self.cached_token_history;
        if cached.is_empty() {
            return 0;
        }
        if tokens.len() < cached.len() {
            return 0;
        }
        if tokens[..cached.len()] != cached[..] {
            return 0;
        }
        cached.len()
    }

    fn save_cache_state(&mut self, args: SaveStateArgs<'_>) {
        // Flat save (identical on the fresh and delta paths): persist
        // `prompt + generated`, dropping the terminal turn-boundary token
        // when the decode terminated on stop so the cached history ends on
        // the `<turn|>` boundary the next delta re-renders itself.
        // Unconditional — there is no `reuse_cache` branch here (only the
        // paged core has one, and paged turns never reach this hook), and
        // the engine's session_start guard rejects `reuse_cache=Some(false)`
        // anyway.
        let history_tokens: &[u32] =
            if args.finish_reason != "length" && !args.generated_tokens.is_empty() {
                &args.generated_tokens[..args.generated_tokens.len() - 1]
            } else {
                args.generated_tokens
            };
        let mut new_history = Vec::with_capacity(args.save_tokens.len() + history_tokens.len());
        new_history.extend_from_slice(args.save_tokens);
        new_history.extend_from_slice(history_tokens);
        self.cached_token_history = new_history;
        if !args.is_delta {
            // Fresh text-only turn: clear any stale image/audio key (a
            // text-only turn has no multimodal key to set). Delta turns leave
            // them untouched — text-only by the delta image guard, so they are
            // structurally `None`.
            self.cached_image_key = None;
            self.cached_audio_key = None;
        }
    }

    fn eval_caches(&self) -> Result<()> {
        // Materialize the prefill KV before entering the decode loop.
        eval_gemma4_caches(
            self.caches
                .as_ref()
                .ok_or_else(|| Error::from_reason("Gemma4 eval_caches: caches missing"))?,
        )
    }

    /// Flat prefill for the engine's generic flow. `prefill_body_gemma4`
    /// processes `tokens[0 .. N-1]` through the body (a no-op when
    /// `N == 1`), the per-layer KV evals materialize, then the last
    /// token runs the full forward for sampling-ready `[1, vocab]`
    /// logits. Serves the fresh path (full prompt or strict-extend
    /// tail) and the session-delta path identically.
    ///
    /// `diagnostic::set_step(-1)` marks the prefill forward for
    /// `MLX_DEBUG_GEMMA4_DUMP`, uniformly across entry points.
    fn prefill(&mut self, prompt_tokens: &[u32], stream: Stream) -> Result<MxArray> {
        // Defensive: caches must be live before the prefill runs. The
        // engine's miss-reset re-inits, and verify/`has_live_session`
        // check liveness — but if somebody cleared the caches
        // out-of-band between turns, re-init here.
        if self.caches.is_none() {
            self.init_caches_sync()?;
        }

        let prefill_slice: Vec<i32> = prompt_tokens.iter().map(|&t| t as i32).collect();
        let prefill_len = prefill_slice.len();
        let prompt = MxArray::from_int32(&prefill_slice, &[1, prefill_len as i64])?;

        {
            let _stream_ctx = StreamContext::new(stream);
            let caches = self
                .caches
                .as_mut()
                .ok_or_else(|| Error::from_reason("Gemma4 prefill: caches missing"))?;
            prefill_body_gemma4(
                &prompt,
                &self.embed_tokens,
                &self.layers,
                caches,
                &self.final_norm,
                self.ple.as_ref(),
                &self.config,
            )?;
        }
        eval_gemma4_caches(
            self.caches
                .as_ref()
                .ok_or_else(|| Error::from_reason("Gemma4 prefill: caches missing"))?,
        )?;

        // Last token → logits. `prefill_body_gemma4` processed
        // `[0 .. prefill_len - 1]` and left the final token for us.
        let last_token = prompt.slice_axis(1, prefill_len as i64 - 1, prefill_len as i64)?;
        let logits = {
            let _stream_ctx = StreamContext::new(stream);
            let caches = self
                .caches
                .as_mut()
                .ok_or_else(|| Error::from_reason("Gemma4 prefill: caches missing"))?;
            crate::models::gemma4::diagnostic::set_step(-1);
            forward_inner(
                &last_token,
                &self.embed_tokens,
                &self.layers,
                caches,
                &self.final_norm,
                &self.lm_head,
                self.embed_weight_t.as_ref(),
                self.ple.as_ref(),
                &self.config,
            )?
        };
        logits.squeeze(Some(&[1]))
    }

    type Decode<'a>
        = Gemma4Decode<'a>
    where
        Self: 'a;

    fn begin_decode(&mut self, _turn: &TurnSetup<'_>) -> Result<Self::Decode<'_>> {
        // No compiled path, no turn-constant captures: gemma4's eager
        // decode threads everything through the live session caches.
        Ok(Gemma4Decode {
            inner: self,
            step: 0,
        })
    }

    /// Gemma4 output finalization: raw decode (`skip_special_tokens =
    /// false` so the channel/tool-call DSL markers survive) →
    /// `parse_gemma4_output` → `promote_channel_only_output` →
    /// tool-calls finish-reason promotion. `reasoning_tokens` arrives as
    /// 0 (thinking disabled) and `prompt_tokens` / `performance` are
    /// passed through unchanged. `cached_tokens` is overwritten by the
    /// session core.
    fn finalize_turn(&self, args: FinalizeArgs<'_>) -> Result<ChatResult> {
        let raw_text = args.tokenizer.decode_sync(args.generated_tokens, false)?;
        let mut parsed = super::output_parser::parse_gemma4_output(&raw_text);
        promote_channel_only_output(&mut parsed);
        let finish_reason = if parsed.tool_calls.iter().any(|tc| tc.status == "ok") {
            "tool_calls".to_string()
        } else {
            args.finish_reason
        };
        Ok(ChatResult {
            text: parsed.text,
            tool_calls: parsed.tool_calls,
            thinking: parsed.thinking,
            num_tokens: args.generated_tokens.len() as u32,
            prompt_tokens: args.prompt_tokens,
            reasoning_tokens: args.reasoning_tokens,
            finish_reason,
            raw_text,
            cached_tokens: 0,
            performance: args.performance,
        })
    }

    fn has_paged_adapter(&self) -> bool {
        self.paged_adapter.is_some()
    }

    /// UNCONDITIONALLY `true` — even for checkpoints without a vision
    /// tower. Image-bearing messages are accepted on every entry point and
    /// surface the exact "Images provided but model has no vision support
    /// (no vision_config in config.json)" error from INSIDE the turn (after
    /// template rendering); returning `false` here would replace that with
    /// the engine's typed pre-render restart-prefix error. The error
    /// surfaces from inside `vision_turn` instead (see
    /// `prepare_vision_tokens`).
    fn supports_images(&self) -> bool {
        true
    }

    /// Audio support is gated on the unified `embed_audio` projection being
    /// built at load time (`config.has_audio`). Unlike `supports_images`
    /// (which is unconditionally `true` so the "no vision support" error
    /// surfaces from inside the turn), audio is rejected at the pre-render
    /// guard for non-audio checkpoints: there is no audio entry inside the
    /// turn to surface a clearer message, and the engine's typed restart
    /// prefix is the correct contract.
    fn supports_audio(&self) -> bool {
        self.embed_audio.is_some()
    }

    fn extra_eos_ids(&self) -> Vec<u32> {
        // The MODEL-config eos list (`<eos>` / `<end_of_turn>`) honored
        // alongside the session `<turn|>` id. A negative config id can
        // never equal a `u32`-cast token, so filter those out instead of
        // wrapping.
        self.config
            .eos_token_ids
            .iter()
            .filter(|&&id| id >= 0)
            .map(|&id| id as u32)
            .collect()
    }

    fn stream_skip_special_tokens(&self) -> bool {
        // `decode_stream(false)`: the stream parser must see the
        // `<|channel>` / `<|tool_call>` markers. The residual flush then
        // decodes with the same flag (engine guarantee), keeping
        // `streamed_text_len` accounting consistent.
        false
    }

    fn stream_emitter(&self) -> Box<dyn StreamEmitter> {
        Box::new(Gemma4Emitter::new())
    }

    /// REJECT text deltas on image-holding sessions despite
    /// `supports_images() == true`: gemma4's prefix reuse is text-only, so
    /// a delta on top of an image session would prefill on caches whose
    /// positions include expanded image tokens the history bookkeeping
    /// does not model. The message has NO space after the prefix:
    /// `"{PREFIX}{entry_fn} is text-only; session currently holds image
    /// state"`.
    fn text_delta_image_guard(&self, entry_fn: &'static str) -> Option<String> {
        // Warm-continue: a continuable media turn (audio / non-unified image)
        // kept its global paged KV live + a sliding history checkpoint at the
        // full prefix, so a text delta restores causally on the live media KV.
        // The marker ALONE is insufficient: the live paged request must STILL
        // exist (`is_live_for_continue()`), because the warm continue reads the
        // adapter's live `block_table` directly. On a shared cross-session
        // adapter another session may have run `reset_for_new_request` and
        // released the request after this session armed the marker; then the
        // text path would instead do a content-address prefix lookup over
        // `[media-prefix + delta]` — which can hit stale media-feature K/V or
        // unfaithfully re-prefill the media placeholders. Require both the
        // marker AND a live request; otherwise fall through to the restart
        // rejection so the TS floor cold-restarts (resend full history →
        // faithful vision/audio prefill, no media-placeholder content lookup).
        if self.media_session_continuable
            && self
                .paged_adapter
                .as_ref()
                .is_some_and(|adapter| adapter.is_live_for_continue())
        {
            return None;
        }
        // A continuable media session whose paged request is no longer live
        // must cold-restart, not warm-continue against a released request.
        // A warm text delta clears `cached_image_key` but leaves the marker
        // armed, so the key checks below would silently fall through to `None`;
        // gate on the marker (the true media-held signal) and emit the audio /
        // image restart message that matches whichever media key still remains.
        if self.media_session_continuable {
            let media_state = if self.cached_audio_key.is_some() {
                "audio"
            } else {
                "image"
            };
            return Some(format!(
                "{}{entry_fn} is text-only; session currently holds {media_state} state",
                engine::IMAGE_CHANGE_RESTART_PREFIX
            ));
        }
        if self.cached_image_key.is_some() {
            Some(format!(
                "{}{entry_fn} is text-only; session currently holds image state",
                engine::IMAGE_CHANGE_RESTART_PREFIX
            ))
        } else if self.cached_audio_key.is_some() {
            Some(format!(
                "{}{entry_fn} is text-only; session currently holds audio state",
                engine::IMAGE_CHANGE_RESTART_PREFIX
            ))
        } else {
            None
        }
    }

    fn augment_performance(&self, _profiler: &DecodeProfiler, _metrics: &mut PerformanceMetrics) {
        // No-op: gemma4 has no MTP heads (acceptance fields stay None) and
        // its metrics carry no `profile_phases`. The default would only add
        // profiling-gated extras; keep the payload byte-stable instead.
    }

    fn has_live_session(&self) -> bool {
        // Requires an initialized session: a non-empty
        // `cached_token_history` AND live `caches`.
        !self.cached_token_history.is_empty() && self.caches.is_some()
    }

    fn session_holds_images(&self) -> bool {
        self.cached_image_key.is_some()
    }

    fn paged_turn(&mut self, args: &mut WholeTurnArgs<'_>) -> Option<Result<TurnOutput>> {
        // Gated on `has_paged_adapter()`; with the adapter live EVERY
        // text turn (fresh + delta, sync + streaming) takes the generic
        // paged engine, which drives the adapter lifecycle via
        // [`PagedBackend`] and reuses the shared `run_decode_loop`.
        Some(crate::engine::paged_turn::run_paged_turn(self, args))
    }

    fn vision_turn(&mut self, args: &mut WholeTurnArgs<'_>) -> Option<Result<TurnOutput>> {
        Some(self.vision_chat_turn(args))
    }
}

/// Build the Gemma4 wire-format delta text for a session-continue turn.
///
/// The cached history ends on `<turn|>` (because
/// `chat_session_start_sync` uses `turn_end_id` as eos). The leading
/// `\n` closes that turn's line; then we open a new user turn and
/// prime an assistant ("model") turn.
///
/// Gemma4's chat template does NOT inject a `<think>\n` prefix after
/// the assistant opener the way Qwen3.5's does — `enable_thinking`
/// affects which template branch renders, not the raw delta. We
/// accept the parameter for API symmetry but deliberately ignore it.
///
/// `sanitized_user` MUST already be passed through
/// `Qwen3Tokenizer::sanitize_messages_public` by the caller.
fn build_gemma4_continue_delta_text(sanitized_user: &str, enable_thinking: Option<bool>) -> String {
    // `enable_thinking` intentionally unused: Gemma4's template does
    // not render a `<think>` prefix on the raw delta path.
    let _ = enable_thinking;
    format!("\n<|turn>user\n{sanitized_user}<turn|>\n<|turn>model\n")
}

/// Build the Gemma4 wire-format delta text for a tool-result turn.
///
/// Gemma4's chat template renders tool-role messages as plain
/// `<|turn>tool\n{content}<turn|>` blocks — no `<tool_response>`
/// wrapping (unlike Qwen3.5). The `tool_call_id` is NOT rendered:
/// Gemma4 identifies tool responses positionally in the turn stream,
/// not via an explicit id field.
///
/// Tool content is passed through [`escape_gemma4_content`] so
/// malicious tool output containing Gemma4 delimiter tokens can't
/// escape the tool turn and inject synthetic structure. The shared
/// [`crate::tokenizer::TOOL_ERROR_MARKER`] (when `is_error == Some(true)`)
/// is prepended BEFORE escaping so the marker text — which contains
/// no Gemma4 delimiter tokens — passes through verbatim and the
/// downstream escaping still protects any user content that follows.
fn build_gemma4_tool_delta_text(
    _tool_call_id: &str,
    content: &str,
    enable_thinking: Option<bool>,
    is_error: Option<bool>,
) -> String {
    // `enable_thinking` intentionally unused: see
    // `build_gemma4_continue_delta_text` for why the raw delta path
    // ignores reasoning mode.
    let _ = enable_thinking;
    let rendered_content = crate::tokenizer::apply_tool_error_marker(content, is_error);
    let escaped = escape_gemma4_content(&rendered_content);
    format!("\n<|turn>tool\n{escaped}<turn|>\n<|turn>model\n")
}

#[napi]
impl Gemma4Model {
    /// Create an uninitialized `Gemma4Model` stub from a config.
    ///
    /// **Prefer [`Gemma4Model::load`]** for any real usage — `new(config)`
    /// is a config-only stub that matches the OCR-model pattern
    /// (`VLModel::new(config)`, `QianfanOCRModel::new(config)`) and is
    /// intentionally NOT runnable. It was introduced in the cache-limit
    /// coordinator work so that the coordinator's per-model delta is
    /// registered exclusively on the `load()` path, eliminating a
    /// baseline-registration gap where a no-op `new(config)` would have
    /// leaked an empty guard into the coordinator.
    ///
    /// This path does NOT spawn a model thread, NOT materialize any
    /// weights, and NOT register with the cache-limit coordinator. The
    /// returned instance is only useful for config inspection — every
    /// session method (`chatSessionStart` / `chatSessionContinue` /
    /// `chatSessionContinueTool` and their streaming variants) rejects
    /// with a `napi::Error` whose message is exactly
    /// `"Model not initialized. Call Gemma4Model.load() first."` until
    /// `load()` runs and installs the underlying model thread. The
    /// synchronous `resetCaches()` call is a silent no-op on the stub
    /// to keep `ChatSession.reset()` idempotent across both runnable
    /// and stub instances.
    ///
    /// A runnable model requires `await Gemma4Model.load(path)`. The
    /// constructor signature is fixed by NAPI-RS; the stub-only behavior is
    /// covered by the regression tests in
    /// `__test__/models/model-loader-gemma4.test.ts`.
    #[napi(constructor)]
    pub fn new(config: Gemma4Config) -> Self {
        let has_vision = config.vision_config.is_some() || config.unified_vision_config.is_some();
        let has_audio = config.has_audio;
        Self {
            thread: None,
            model_id: 0,
            has_vision,
            has_audio,
            initialized: false,
            paged_active: false,
            _cache_limit_guard: None,
        }
    }

    /// Returns true if weights have been loaded via `load()`.
    #[napi(getter)]
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Whether the block-paged KV cache adapter is active on this model
    /// instance.
    ///
    /// `true` iff `Gemma4Inner::paged_adapter` was successfully
    /// constructed at load time (driven by
    /// `Gemma4Config::use_block_paged_cache`). The
    /// `gemma4_paged_vs_flat_parity` integration test pins greedy
    /// byte-equal at BF16 against real Gemma-4-E2B-IT weights. Stubs
    /// constructed via `new(config)` always return `false`. Surfaced
    /// through this NAPI method so server endpoints can branch on it
    /// without a model-thread roundtrip.
    #[napi]
    pub fn has_block_paged_cache(&self) -> bool {
        self.paged_active
    }

    #[napi]
    pub fn model_id(&self) -> u32 {
        self.model_id as u32
    }

    /// Load a Gemma4 model from a directory.
    #[napi]
    pub async fn load(model_path: String) -> Result<Gemma4Model> {
        Self::load_from_dir(&model_path).await
    }
}

crate::models::chat_napi::chat_napi_surface! {
    class: Gemma4Model,
    thread_cmd: crate::engine::cmd::ChatCmd,
    thread: { option: "Model not initialized. Call Gemma4Model.load() first." },
    image_guard: { vision: has_vision, audio: has_audio },
    ts_stream_start: "messages: ChatMessage[], config: ChatConfig | null | undefined, callback: (err: Error | null, chunk: ChatStreamChunk) => void",
    ts_stream_continue: "userMessage: string, images: Uint8Array[] | null | undefined, audio: Uint8Array[] | null | undefined, config: ChatConfig | null | undefined, callback: (err: Error | null, chunk: ChatStreamChunk) => void",
    ts_stream_continue_tool: "toolCallId: string, content: string, config: ChatConfig | null | undefined, callback: (err: Error | null, chunk: ChatStreamChunk) => void, isError?: boolean | null | undefined",
}

/// How many layers to batch per eval during warmup.
///
/// Larger GPUs can handle bigger Metal command buffers before timing out,
/// but the timeout is nondeterministic (thermal state, system load).
/// Uses `max_recommended_working_set_size` (GPU memory) as proxy:
///   ≤128 GB → 1  (base / Pro / Max)
///   ≤384 GB → 2  (Ultra variants)
///   >384 GB → 4  (future hardware)
fn warmup_layer_batch_size() -> usize {
    let gb = crate::stream::WiredLimitContext::get_max_working_set_size() / (1 << 30);
    match gb {
        0..=128 => 1,
        129..=384 => 2,
        _ => 4,
    }
}

/// Single-token forward pass to trigger Metal shader compilation at load time.
/// Layers are eval'd in batches (sized by GPU capability) to keep Metal
/// command buffers under the timeout limit on cold shader cache.
pub(crate) fn warmup_forward(inner: &Gemma4Inner) -> Result<()> {
    let config = &inner.config;
    let batch = warmup_layer_batch_size();
    let mem_before = crate::array::get_active_memory();
    info!(
        "[warmup] layer batch size: {} (GPU mem: query complete)",
        batch
    );

    {
        let mut caches = init_caches_for_config(config);
        let dummy = MxArray::from_int32(&[1i32], &[1, 1])?;

        let mut h = inner.embed_tokens.forward(&dummy)?;
        h = h.mul_scalar((config.hidden_size as f64).sqrt())?;
        h.eval();

        for (i, layer) in inner.layers.iter().enumerate() {
            h = layer.forward(&h, None, Some(&mut caches[i]), None, false)?;
            if (i + 1) % batch == 0 || i + 1 == inner.layers.len() {
                h.eval();
            }
        }

        h = inner.final_norm.forward(&h)?;
        let logits = if let Some(ref head) = inner.lm_head {
            head.forward(&h)?
        } else if let Some(ref w_t) = inner.embed_weight_t {
            h.matmul(w_t)?
        } else {
            let weight = inner.embed_tokens.get_weight();
            let weight_t = weight.transpose(Some(&[1, 0]))?;
            h.matmul(&weight_t)?
        };
        logits.eval();
    }

    crate::array::synchronize_and_clear_cache();
    let mem_after = crate::array::get_active_memory();
    info!(
        "[warmup] memory: {:.2} GB → {:.2} GB (delta: {:.2} GB)",
        mem_before / 1e9,
        mem_after / 1e9,
        (mem_after - mem_before) / 1e9
    );

    Ok(())
}

/// Build throwaway KV caches for a Gemma4 config.
///
/// Used by `warmup_forward` to run a single dummy token through the
/// full layer stack at load time (triggering Metal shader compilation)
/// without touching the persistent `self.caches` on `Gemma4Inner`. The
/// persistent path initializes its caches via `init_caches_sync` from
/// the engine's miss-path `reset_caches(ResetScope::PrefixMiss)` (or
/// defensively inside `ChatBackend::prefill` / the vision cores).
fn init_caches_for_config(config: &Gemma4Config) -> Vec<Gemma4LayerCache> {
    let num_layers = config.num_hidden_layers as usize;
    let mut caches = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        if config.is_global_layer(i) {
            caches.push(Gemma4LayerCache::new_global());
        } else {
            caches.push(Gemma4LayerCache::new_sliding(config.sliding_window));
        }
    }
    caches
}

/// Check whether `token` should terminate decoding.
///
/// The config-level `eos_token_ids` are always honored. The caller-supplied
/// `eos_token_id` is treated as an additional stop token — it does NOT
/// replace the config list. Session-start callers get their clean boundary
/// token (for Gemma4 that is `<turn|>`) while still respecting the
/// underlying model's intrinsic eos set.
#[inline]
fn is_eos_token(token: u32, eos_ids: &[i32], eos_token_id: u32) -> bool {
    if eos_ids.contains(&(token as i32)) {
        return true;
    }
    eos_token_id == token
}

#[derive(Clone, Copy)]
struct Gemma4RepetitionCutoff {
    max_consecutive_tokens: i32,
    max_ngram_repeats: i32,
    ngram_size: i32,
}

fn repetition_cutoff_from_config(config: &ChatConfig) -> Gemma4RepetitionCutoff {
    Gemma4RepetitionCutoff {
        max_consecutive_tokens: config.max_consecutive_tokens.unwrap_or(16),
        max_ngram_repeats: config.max_ngram_repeats.unwrap_or(3),
        ngram_size: config.ngram_size.unwrap_or(64),
    }
}

fn check_gemma4_repetition_cutoff(
    generated_tokens: &[u32],
    cutoff: Gemma4RepetitionCutoff,
) -> Option<&'static str> {
    crate::sampling::check_repetition_cutoff(
        generated_tokens,
        cutoff.max_consecutive_tokens,
        cutoff.max_ngram_repeats,
        cutoff.ngram_size,
    )
}

fn make_sampling_config(
    config: &ChatConfig,
    model_config: &Gemma4Config,
) -> Option<SamplingConfig> {
    let temp = config
        .temperature
        .or(model_config.default_temperature)
        .unwrap_or(0.0);
    if temp <= 0.0 {
        // Greedy: use a near-zero temperature for argmax-like behavior.
        // Cannot pass None because sample() defaults to temperature=1.0.
        return Some(SamplingConfig {
            temperature: Some(0.0),
            top_k: None,
            top_p: None,
            min_p: None,
        });
    }
    Some(SamplingConfig {
        temperature: Some(temp),
        top_k: config.top_k.or(model_config.default_top_k),
        top_p: config.top_p.or(model_config.default_top_p),
        min_p: config.min_p,
    })
}

fn sample_next_token(logits: &MxArray, config: Option<SamplingConfig>) -> Result<MxArray> {
    if is_greedy_sampling(config) {
        return logits.argmax(-1, Some(false));
    }
    sample(logits, config)
}

fn is_greedy_sampling(config: Option<SamplingConfig>) -> bool {
    config.is_some_and(|cfg| {
        cfg.temperature.unwrap_or(1.0) <= 0.0
            && cfg.top_k.is_none()
            && cfg.top_p.is_none()
            && cfg.min_p.is_none()
    })
}

/// Transformer body: embedding through decoder layers and final norm.
///
/// Matches mlx-vlm `Gemma4TextModel.__call__`. Does NOT run lm_head or softcap.
/// Used by chunked prefill for intermediate chunks and by the full forward.
///
/// When `inputs_embeds` is provided, uses it directly (skipping embedding lookup).
/// When `per_layer_inputs` is provided, uses it directly (skipping PLE computation).
fn forward_body(
    input_ids: Option<&MxArray>,
    inputs_embeds: Option<MxArray>,
    embedding: &Embedding,
    layers: &[Gemma4DecoderLayer],
    caches: &mut [Gemma4LayerCache],
    final_norm: &RMSNorm,
    ple: Option<&PleComponents>,
    per_layer_inputs: Option<&MxArray>,
    config: &Gemma4Config,
) -> Result<MxArray> {
    // Step 1: Embedding (or use pre-computed embeddings)
    let mut h = if let Some(embeds) = inputs_embeds {
        embeds
    } else {
        let ids = input_ids.ok_or_else(|| {
            Error::from_reason("forward_body: either input_ids or inputs_embeds must be provided")
        })?;
        let emb = embedding.forward(ids)?;
        emb.mul_scalar((config.hidden_size as f64).sqrt())?
    };

    let seq_len = h.shape_at(1)?;

    // Step 2: PLE (per-layer embeddings) — compute or reuse
    let owned_ple: Option<MxArray>;
    let effective_ple: Option<&MxArray> = if let Some(ple_inputs) = per_layer_inputs {
        // Pre-computed: might need to slice for chunked prefill
        if ple_inputs.shape_at(1)? != seq_len {
            // Slice to match current chunk (chunked prefill)
            let cache_offset = caches
                .iter()
                .find_map(|c| {
                    let off = c.get_offset();
                    if off > 0 { Some(off as i64) } else { None }
                })
                .unwrap_or(0);
            let max_start = ple_inputs.shape_at(1)? - seq_len;
            let start = cache_offset.min(max_start);
            owned_ple = Some(ple_inputs.slice_axis(1, start, start + seq_len)?);
            owned_ple.as_ref()
        } else {
            Some(ple_inputs)
        }
    } else if let Some(ple) = ple {
        if let Some(ids) = input_ids {
            owned_ple = Some(compute_ple(ids, &h, ple, seq_len)?);
            owned_ple.as_ref()
        } else {
            None
        }
    } else {
        None
    };

    // Step 3: Project PLE if we have per-layer inputs
    // Matches mlx-vlm project_per_layer_inputs: projects h and combines with token PLEs
    let projected_ple: Option<MxArray> = if let Some(ple_data) = effective_ple {
        if let Some(ple) = ple {
            Some(project_per_layer_inputs(&h, ple_data, ple)?)
        } else {
            None
        }
    } else {
        None
    };

    // Step 4: Build masks
    // Global layers: None during prefill → triggers fused causal SDPA kernel
    // Sliding layers: explicit windowed mask during prefill
    // Decode (seq_len == 1): None for both
    //
    // Matches mlx-vlm create_attention_mask behavior:
    //   global → "causal" string → fused kernel
    //   sliding → explicit mask with window constraint
    // Sliding mask: only needed when the previous rotating-cache view plus the
    // current chunk exceeds the window. Matches mlx-lm RotatingKVCache.make_mask.
    let sliding_window = config.sliding_window as i64;
    let sliding_mask_offset = if seq_len > 1 {
        let sliding_idx = (0..config.num_hidden_layers as usize)
            .find(|&i| config.is_sliding_layer(i))
            .unwrap_or(0);
        let offset = if sliding_idx < caches.len() {
            caches[sliding_idx].get_offset()
        } else {
            0
        };
        sliding_mask_offset_for_chunk(seq_len, offset, sliding_window)
    } else {
        None
    };
    let sliding_mask = sliding_mask_offset
        .map(|offset| create_sliding_mask(seq_len, offset, sliding_window))
        .transpose()?;

    // Step 5: Forward through layers with KV cache sharing
    let has_kv_sharing = config.num_kv_shared_layers.is_some_and(|n| n > 0);
    let mut shared_kv: HashMap<usize, (MxArray, MxArray)> = HashMap::new();

    crate::models::gemma4::diagnostic::set_path("flat");

    for (i, layer) in layers.iter().enumerate() {
        crate::models::gemma4::diagnostic::set_layer(i);
        let is_global = config.is_global_layer(i);

        // Global layers: None mask → attention module uses causal SDPA or no-mask path
        // Sliding layers: explicit windowed mask
        let mask: Option<&MxArray> = if is_global {
            None
        } else {
            sliding_mask.as_ref()
        };

        let ple_input = projected_ple.as_ref().map(|p| {
            // projected_ple shape: [B, T, num_layers, ple_dim], extract layer i
            p.slice_axis(2, i as i64, i as i64 + 1)
                .and_then(|s| s.squeeze(Some(&[2])))
        });
        let ple_input_ref = match &ple_input {
            Some(Ok(arr)) => Some(arr),
            _ => None,
        };

        if has_kv_sharing && config.is_kv_shared_layer(i) {
            let anchor_idx = config.kv_shared_anchor(i).ok_or_else(|| {
                Error::from_reason(format!(
                    "Layer {} is shared but has no anchor (missing layer type match)",
                    i
                ))
            })?;

            let (shared_keys, shared_values) = shared_kv.get(&anchor_idx).ok_or_else(|| {
                Error::from_reason(format!(
                    "Anchor layer {} K/V not found for shared layer {}",
                    anchor_idx, i
                ))
            })?;

            // Shared layer uses anchor's cache offset.
            // Subtract seq_len to get pre-update offset (queries need same positions as anchor).
            let cache_offset = caches[anchor_idx].get_offset() - seq_len as i32;

            h = layer.forward_shared(
                &h,
                mask,
                shared_keys,
                shared_values,
                cache_offset,
                ple_input_ref,
            )?;
        } else {
            let needs_stash = has_kv_sharing && config.should_store_shared_kv(i);
            h = layer.forward(&h, mask, Some(&mut caches[i]), ple_input_ref, needs_stash)?;

            if has_kv_sharing
                && config.should_store_shared_kv(i)
                && let Some((keys, values)) = caches[i].take_stashed_kv()
            {
                shared_kv.insert(i, (keys, values));
            }
        }
    }

    // Final norm
    final_norm.forward(&h)
}

/// Full forward pass: transformer body + lm_head + logit softcapping.
///
/// Used for the final prefill chunk and for each decode step.
fn forward_inner(
    input_ids: &MxArray,
    embedding: &Embedding,
    layers: &[Gemma4DecoderLayer],
    caches: &mut [Gemma4LayerCache],
    final_norm: &RMSNorm,
    lm_head: &Option<LinearProj>,
    embed_weight_t: Option<&MxArray>,
    ple: Option<&PleComponents>,
    config: &Gemma4Config,
) -> Result<MxArray> {
    let h = forward_body(
        Some(input_ids),
        None,
        embedding,
        layers,
        caches,
        final_norm,
        ple,
        None,
        config,
    )?;

    crate::models::gemma4::diagnostic::dump_norm(0, "post_final_norm", &h, None);

    // LM head or tied embeddings
    let logits = if let Some(head) = lm_head {
        head.forward(&h)?
    } else if let Some(w_t) = embed_weight_t {
        h.matmul(w_t)?
    } else {
        let weight = embedding.get_weight();
        let weight_t = weight.transpose(Some(&[1, 0]))?;
        h.matmul(&weight_t)?
    };
    crate::models::gemma4::diagnostic::dump_logits("pre_softcap", &logits);

    // Logit softcapping — compiled fused kernel (matches Python's mx.compile logit_softcap)
    if let Some(cap) = config.final_logit_softcapping {
        let cap_arr = MxArray::scalar_float_like(cap, &logits)?;
        let handle = unsafe { mlx_sys::mlx_logit_softcap(logits.handle.0, cap_arr.handle.0) };
        let capped = MxArray::from_handle(handle, "logit_softcap")?;
        crate::models::gemma4::diagnostic::dump_logits("post_softcap", &capped);
        Ok(capped)
    } else {
        crate::models::gemma4::diagnostic::dump_logits("post_softcap", &logits);
        Ok(logits)
    }
}

/// Compute PLE (per-layer embeddings) from input_ids.
/// Returns shape [B, T, num_layers, ple_dim].
fn compute_ple(
    input_ids: &MxArray,
    h: &MxArray,
    ple: &PleComponents,
    seq_len: i64,
) -> Result<MxArray> {
    let ple_dim = ple.ple_dim as i64;
    let num_layers = ple.num_layers as i64;

    // Mask OOV token IDs to 0 for PLE embedding
    let ple_vocab = MxArray::scalar_int(ple.vocab_size_per_layer_input)?;
    let zero = MxArray::scalar_int(0)?;
    let valid_mask = input_ids
        .greater_equal(&zero)?
        .logical_and(&input_ids.less(&ple_vocab)?)?;
    let masked_ids = valid_mask.where_(input_ids, &zero)?;

    // per_layer_embeds: [B, T, num_layers * ple_dim]
    let per_layer_embeds = ple.embed_tokens_per_layer.forward(&masked_ids)?;
    let per_layer_embeds = per_layer_embeds.mul_scalar((ple.ple_dim as f64).sqrt())?;
    let batch = per_layer_embeds.shape_at(0)?;
    let per_layer_embeds = per_layer_embeds.reshape(&[batch, seq_len, num_layers, ple_dim])?;

    // Project from main hidden state
    let projected = ple.per_layer_model_projection.forward(h)?;
    let projected = projected.mul_scalar(ple.per_layer_model_projection_scale)?;
    let projected = projected.reshape(&[batch, seq_len, num_layers, ple_dim])?;

    let projected = ple.per_layer_projection_norm.forward(&projected)?;

    // Combine: (normed_projection + per_layer_embeds) * 1/sqrt(2)
    let combined = projected.add(&per_layer_embeds)?;
    combined.mul_scalar(ple.per_layer_input_scale)
}

/// Project per-layer inputs: combine PLE data with hidden state projection.
/// Returns shape [B, T, num_layers, ple_dim].
fn project_per_layer_inputs(
    _h: &MxArray,
    per_layer_data: &MxArray,
    _ple: &PleComponents,
) -> Result<MxArray> {
    // PLE data is already fully computed (combined projection + token embeddings)
    Ok(per_layer_data.clone())
}

/// Build the per-layer routing list for the paged dispatch (pure
/// function over a `Gemma4Config`).
///
/// Returns `Vec<Gemma4LayerKind>` of length `config.num_hidden_layers`
/// where each entry classifies a layer as:
/// * `Sliding` — stays on the flat `Gemma4LayerCache::Sliding` path.
/// * `GlobalPaged { paged_idx }` — routes through the paged adapter
///   at the given global-layer ordinal.
/// * `SharedOnGlobal { anchor_paged_idx }` — KV-shared layer whose
///   anchor is a global layer; reads K/V via the adapter.
/// * `SharedOnSliding { anchor_layer_idx }` — KV-shared layer whose
///   anchor is a sliding layer; reads K/V from the anchor's flat
///   cache stash.
///
/// `paged_idx` counts only physical non-shared `full_attention` layers in
/// their original decoder order — matches the `LayerKVPool` slot count from
/// `Gemma4Inner::new`. KV-shared layers do NOT consume a paged slot (they
/// reuse the anchor's K/V); the shared variants carry the anchor's index so
/// the shared forward path can resolve it.
///
/// Lifted to a free helper so unit tests can drive it without owning a
/// `Gemma4Inner` (which requires loaded weights). Mirrors LFM2's
/// `compute_layer_kinds` pattern.
#[cfg(test)]
pub(crate) fn compute_layer_kinds(config: &Gemma4Config) -> Vec<Gemma4LayerKind> {
    compute_layer_kinds_from_kv_cache_specs(config)
        .expect("Gemma4 layer kinds must derive from valid KV cache specs")
}

pub(crate) fn compute_layer_kinds_from_kv_cache_specs(
    config: &Gemma4Config,
) -> std::result::Result<Vec<Gemma4LayerKind>, String> {
    let n = config.num_hidden_layers as usize;
    let block_size = config.paged_block_size.unwrap_or(16);
    let specs = compute_layer_kv_cache_specs(config, block_size, KVCacheDType::BFloat16)?;
    let max_model_len = u32::try_from(config.max_position_embeddings).map_err(|_| {
        format!(
            "Gemma4 layer kind routes: invalid max_position_embeddings {}",
            config.max_position_embeddings
        )
    })?;
    let routes = derive_layer_kv_cache_routes(
        &specs,
        max_model_len,
        gemma4_paged_prefill_group_max_chunk(),
    )
    .map_err(|e| format!("Gemma4 layer kind route derivation failed: {e}"))?;

    let mut kinds = vec![None; n];
    for route in routes {
        if route.layer_index >= n {
            return Err(format!(
                "Gemma4 layer kind route derivation produced out-of-range layer {} for {n} layers",
                route.layer_index
            ));
        }
        let physical_ordinal = u32::try_from(route.physical_layer_ordinal).map_err(|_| {
            format!(
                "Gemma4 layer kind route ordinal {} does not fit u32",
                route.physical_layer_ordinal
            )
        })?;
        let kind = match (route.shared_kv_anchor, route.attention_kind) {
            (Some(_), AttentionKind::Full) => Gemma4LayerKind::SharedOnGlobal {
                anchor_paged_idx: physical_ordinal,
            },
            (Some(anchor), AttentionKind::SlidingWindow { .. }) => {
                let anchor_layer_idx = u32::try_from(anchor).map_err(|_| {
                    format!("Gemma4 shared sliding anchor layer {anchor} does not fit u32")
                })?;
                Gemma4LayerKind::SharedOnSliding { anchor_layer_idx }
            }
            (None, AttentionKind::Full) => Gemma4LayerKind::GlobalPaged {
                paged_idx: physical_ordinal,
            },
            (None, AttentionKind::SlidingWindow { .. }) => Gemma4LayerKind::Sliding,
        };
        kinds[route.layer_index] = Some(kind);
    }

    kinds
        .into_iter()
        .enumerate()
        .map(|(layer_index, kind)| {
            kind.ok_or_else(|| {
                format!("Gemma4 layer kind route derivation missed layer {layer_index}")
            })
        })
        .collect()
}

/// Build Gemma4's model-independent KV-cache specs.
///
/// The specs are the long-term source of truth for the paged/sliding cache
/// architecture: models declare attention/cache requirements, and common
/// transformer infrastructure groups layers and owns block tables. The current
/// Gemma4 runtime still routes through `Gemma4LayerKind`, but both helpers must
/// agree on physical storage ownership: KV-shared layers are aliases and do not
/// allocate separate cache slots.
pub(crate) fn compute_layer_kv_cache_specs(
    config: &Gemma4Config,
    block_size: u32,
    cache_dtype: KVCacheDType,
) -> std::result::Result<Vec<LayerKVCacheSpec>, String> {
    if block_size == 0 {
        return Err("Gemma4 KV cache specs require block_size > 0".to_string());
    }
    if config.sliding_window <= 0 {
        return Err(format!(
            "Gemma4 KV cache specs require sliding_window > 0, got {}",
            config.sliding_window
        ));
    }

    let n = config.num_hidden_layers as usize;
    let mut specs = Vec::with_capacity(n);
    for layer_index in 0..n {
        let is_global = config.is_global_layer(layer_index);
        let head_size = u32::try_from(config.effective_head_dim(is_global)).map_err(|_| {
            format!(
                "Gemma4 KV cache specs: layer {layer_index} has invalid head_dim {}",
                config.effective_head_dim(is_global)
            )
        })?;
        let num_kv_heads = u32::try_from(config.effective_kv_heads(is_global)).map_err(|_| {
            format!(
                "Gemma4 KV cache specs: layer {layer_index} has invalid num_kv_heads {}",
                config.effective_kv_heads(is_global)
            )
        })?;
        let layout = KVCachePhysicalLayout::new(block_size, num_kv_heads, head_size, cache_dtype);
        if !layout.is_valid() {
            return Err(format!(
                "Gemma4 KV cache specs: layer {layer_index} has invalid physical layout \
                 block_size={block_size}, num_kv_heads={num_kv_heads}, head_size={head_size}"
            ));
        }

        let attention_kind = if is_global {
            AttentionKind::Full
        } else {
            AttentionKind::SlidingWindow {
                sliding_window: config.sliding_window as u32,
            }
        };
        let mut spec = LayerKVCacheSpec::new(layer_index, attention_kind, layout);
        if config.is_kv_shared_layer(layer_index) {
            let anchor = config.kv_shared_anchor(layer_index).ok_or_else(|| {
                format!(
                    "Gemma4 KV cache specs: layer {layer_index} is KV-shared but has no \
                     resolvable anchor"
                )
            })?;
            spec = spec.shared_with_anchor(anchor);
        }
        specs.push(spec);
    }

    crate::transformer::validate_layer_kv_cache_specs(&specs)
        .map_err(|e| format!("Gemma4 KV cache specs failed validation: {e}"))?;
    Ok(specs)
}

pub(crate) fn compute_layer_kv_cache_groups(
    config: &Gemma4Config,
    block_size: u32,
    cache_dtype: KVCacheDType,
    max_chunk: u32,
) -> std::result::Result<Vec<KVCacheGroup>, String> {
    let specs = compute_layer_kv_cache_specs(config, block_size, cache_dtype)?;
    let max_model_len = u32::try_from(config.max_position_embeddings).map_err(|_| {
        format!(
            "Gemma4 KV cache groups: invalid max_position_embeddings {}",
            config.max_position_embeddings
        )
    })?;
    group_layer_kv_cache_specs(&specs, max_model_len, max_chunk)
        .map_err(|e| format!("Gemma4 KV cache grouping failed: {e}"))
}

fn physical_full_attention_layer_count(specs: &[LayerKVCacheSpec]) -> usize {
    specs
        .iter()
        .filter(|spec| {
            spec.shared_kv_anchor.is_none() && matches!(spec.attention_kind, AttentionKind::Full)
        })
        .count()
}

fn gemma4_default_paged_cache_memory_mb(
    max_seq_len: u32,
    block_size: u32,
    head_size: u32,
    num_kv_heads: u32,
    num_layers: u32,
) -> u32 {
    if max_seq_len == 0 || block_size == 0 || head_size == 0 || num_kv_heads == 0 || num_layers == 0
    {
        return GEMMA4_PAGED_CACHE_MIN_DEFAULT_MEMORY_MB;
    }

    let max_blocks = u64::from(max_seq_len.div_ceil(block_size));
    let bytes_per_block = 2u64
        .saturating_mul(u64::from(num_kv_heads))
        .saturating_mul(u64::from(head_size))
        .saturating_mul(u64::from(block_size))
        .saturating_mul(2)
        .saturating_mul(u64::from(num_layers));
    let required_mb = bytes_per_block
        .saturating_mul(max_blocks)
        .div_ceil(BYTES_PER_MIB)
        .max(u64::from(GEMMA4_PAGED_CACHE_MIN_DEFAULT_MEMORY_MB));
    u32::try_from(required_mb).unwrap_or(u32::MAX)
}

/// Default prefill chunk size (tokens per chunk).
/// Note: mlx-lm uses 2048 but the first eval triggers Metal shader compilation
/// which can GPU-timeout with very large graphs. Using 512 keeps individual
/// command buffers under Metal's timeout limit.
const GEMMA4_PREFILL_STEP_SIZE: i64 = 512;
const GEMMA4_PAGED_ATTENTION_V2_PARTITION_SIZE: u64 = 512;
const GEMMA4_PAGED_ATTENTION_V2_AUX_ELEM_LIMIT: u128 = i32::MAX as u128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Gemma4SlidingRestoreLimitOverride {
    Cap(u32),
    Uncapped,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Gemma4SlidingRestoreSuppression {
    limit: u32,
    source: &'static str,
}

fn parse_gemma4_sliding_restore_limit(value: &str) -> Option<Gemma4SlidingRestoreLimitOverride> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    if matches!(
        value.to_ascii_lowercase().as_str(),
        "off" | "none" | "false" | "no" | "unlimited" | "uncapped"
    ) {
        return Some(Gemma4SlidingRestoreLimitOverride::Uncapped);
    }
    value
        .parse::<u32>()
        .ok()
        .map(Gemma4SlidingRestoreLimitOverride::Cap)
}

fn gemma4_sliding_restore_limit_override() -> Option<Gemma4SlidingRestoreLimitOverride> {
    static OVERRIDE: OnceLock<Option<Gemma4SlidingRestoreLimitOverride>> = OnceLock::new();
    *OVERRIDE.get_or_init(|| {
        std::env::var("MLX_GEMMA4_MAX_SLIDING_RESTORE_TOKENS")
            .ok()
            .and_then(|value| parse_gemma4_sliding_restore_limit(&value))
    })
}

fn gemma4_default_sliding_restore_limit(config: &Gemma4Config, block_size: u32) -> Option<u32> {
    let interval = gemma4_sliding_decode_checkpoint_interval(config, block_size);
    (interval > 0).then_some(interval)
}

fn gemma4_large_sliding_restore_suppression_limit_for_override(
    config: &Gemma4Config,
    block_size: u32,
    override_limit: Option<Gemma4SlidingRestoreLimitOverride>,
    restore_tokens: u32,
) -> Option<Gemma4SlidingRestoreSuppression> {
    let (limit, source) = match override_limit {
        Some(Gemma4SlidingRestoreLimitOverride::Uncapped) => return None,
        Some(Gemma4SlidingRestoreLimitOverride::Cap(limit)) => (limit, "env"),
        None => (
            gemma4_default_sliding_restore_limit(config, block_size)?,
            "default",
        ),
    };
    (restore_tokens > limit).then_some(Gemma4SlidingRestoreSuppression { limit, source })
}

fn gemma4_large_sliding_restore_suppression_limit(
    config: &Gemma4Config,
    block_size: u32,
    restore_tokens: u32,
) -> Option<Gemma4SlidingRestoreSuppression> {
    gemma4_large_sliding_restore_suppression_limit_for_override(
        config,
        block_size,
        gemma4_sliding_restore_limit_override(),
        restore_tokens,
    )
}

fn parse_gemma4_sliding_checkpoint_limit(value: &str) -> Option<usize> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    value.parse::<usize>().ok().filter(|limit| *limit > 0)
}

fn gemma4_sliding_checkpoint_limit_override() -> Option<usize> {
    static OVERRIDE: OnceLock<Option<usize>> = OnceLock::new();
    *OVERRIDE.get_or_init(|| {
        std::env::var("MLX_GEMMA4_SLIDING_CHECKPOINT_LIMIT")
            .ok()
            .and_then(|value| parse_gemma4_sliding_checkpoint_limit(&value))
    })
}

fn gemma4_sliding_prefix_checkpoint_limit_for_override(
    config: &Gemma4Config,
    block_size: u32,
    override_limit: Option<usize>,
) -> usize {
    if let Some(limit) = override_limit {
        return limit;
    }
    let sliding_window = config.sliding_window.max(0) as usize;
    let block_size = block_size as usize;
    if sliding_window == 0 || block_size == 0 {
        return GEMMA4_SLIDING_PREFIX_CHECKPOINT_MIN_LIMIT;
    }
    sliding_window
        .div_ceil(block_size)
        .saturating_mul(GEMMA4_SLIDING_PREFIX_CHECKPOINT_WINDOW_MULTIPLIER)
        .clamp(
            GEMMA4_SLIDING_PREFIX_CHECKPOINT_MIN_LIMIT,
            GEMMA4_SLIDING_PREFIX_CHECKPOINT_MAX_DEFAULT_LIMIT,
        )
}

fn gemma4_sliding_prefix_checkpoint_limit(config: &Gemma4Config, block_size: u32) -> usize {
    gemma4_sliding_prefix_checkpoint_limit_for_override(
        config,
        block_size,
        gemma4_sliding_checkpoint_limit_override(),
    )
}

fn gemma4_sliding_decode_checkpoint_interval(config: &Gemma4Config, block_size: u32) -> u32 {
    if block_size == 0 {
        return 0;
    }
    let sliding_window = config.sliding_window.max(0) as u32;
    let target = sliding_window.max(block_size);
    target.div_ceil(block_size).saturating_mul(block_size)
}

fn trim_gemma4_sliding_prefix_checkpoints(
    checkpoints: &mut VecDeque<Gemma4SlidingPrefixCheckpoint>,
    limit: usize,
    trace_enabled: bool,
) {
    if limit == 0 {
        return;
    }
    let mut evicted = 0usize;
    let mut first_prefix_len = None;
    let mut last_prefix_len = None;
    while checkpoints.len() > limit {
        if let Some(checkpoint) = checkpoints.pop_front() {
            first_prefix_len.get_or_insert(checkpoint.prefix_len);
            last_prefix_len = Some(checkpoint.prefix_len);
            evicted += 1;
        }
    }
    if trace_enabled && evicted > 0 {
        write_inference_trace(format_args!(
            "[MLX_TRACE] gemma4 sliding_prefix_checkpoint_evict evicted={} limit={} remaining={} first_prefix_tokens={} last_prefix_tokens={}",
            evicted,
            limit,
            checkpoints.len(),
            first_prefix_len.unwrap_or(0),
            last_prefix_len.unwrap_or(0)
        ));
    }
}

fn gemma4_paged_prefill_group_max_chunk() -> u32 {
    let configured_chunk_size = crate::array::paged_prefill_chunk_size();
    if configured_chunk_size > 0 {
        configured_chunk_size as u32
    } else {
        GEMMA4_PREFILL_STEP_SIZE as u32
    }
}

fn gemma4_paged_prefill_body_chunk_size(configured_chunk_size: i32, body_tokens: usize) -> usize {
    if configured_chunk_size > 0 {
        configured_chunk_size as usize
    } else {
        body_tokens
    }
}

fn gemma4_paged_prefill_paged_attention_enabled_for_chunking() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        crate::inference_trace::env_flag_enabled_or_default(
            "MLX_GEMMA4_PAGED_PREFILL_PAGED_ATTENTION",
            true,
        )
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Gemma4PagedPrefillBodyChunk {
    start: usize,
    len: usize,
    first_position: u32,
    capped_by_v2_aux_limit: bool,
}

fn gemma4_coalesce_single_token_restore_chunks(chunks: &mut Vec<Gemma4PagedPrefillBodyChunk>) {
    if chunks.len() < 2 || chunks.iter().all(|chunk| chunk.len > 1) {
        return;
    }

    let mut merged = Vec::with_capacity(chunks.len());
    let mut idx = 0usize;
    while idx < chunks.len() {
        let mut chunk = chunks[idx].clone();
        if chunk.len == 1 && idx + 1 < chunks.len() {
            let next = &chunks[idx + 1];
            chunk.len += next.len;
            chunk.capped_by_v2_aux_limit |= next.capped_by_v2_aux_limit;
            merged.push(chunk);
            idx += 2;
            continue;
        }
        if chunk.len == 1
            && let Some(previous) = merged.last_mut()
        {
            previous.len += 1;
            previous.capped_by_v2_aux_limit |= chunk.capped_by_v2_aux_limit;
        } else {
            merged.push(chunk);
        }
        idx += 1;
    }
    *chunks = merged;
}

fn gemma4_split_body_chunk_plan_at_position(
    chunks: &mut Vec<Gemma4PagedPrefillBodyChunk>,
    boundary_position: u32,
) {
    if boundary_position == 0 {
        return;
    }

    let Some(idx) = chunks.iter().position(|chunk| {
        let first = chunk.first_position as u64;
        let end = first + chunk.len as u64;
        boundary_position as u64 > first && (boundary_position as u64) < end
    }) else {
        return;
    };

    let chunk = &mut chunks[idx];
    let before_len = (boundary_position - chunk.first_position) as usize;
    let after_len = chunk.len - before_len;
    let after_chunk = Gemma4PagedPrefillBodyChunk {
        start: chunk.start + before_len,
        len: after_len,
        first_position: boundary_position,
        capped_by_v2_aux_limit: chunk.capped_by_v2_aux_limit,
    };
    chunk.len = before_len;
    chunks.insert(idx + 1, after_chunk);
}

fn gemma4_paged_attention_v2_aux_fits(
    num_new_tokens: usize,
    first_position: u32,
    num_query_heads: u32,
    head_size: u32,
) -> bool {
    if num_new_tokens == 0 || num_query_heads == 0 || head_size == 0 {
        return false;
    }

    let max_context_len = first_position as u64 + num_new_tokens as u64;
    if max_context_len <= GEMMA4_PAGED_ATTENTION_V2_PARTITION_SIZE {
        return true;
    }

    let max_num_partitions = max_context_len.div_ceil(GEMMA4_PAGED_ATTENTION_V2_PARTITION_SIZE);
    let exp_sums_size = (num_new_tokens as u128)
        .saturating_mul(num_query_heads as u128)
        .saturating_mul(max_num_partitions as u128);
    let tmp_out_size = exp_sums_size.saturating_mul(head_size as u128);

    exp_sums_size <= GEMMA4_PAGED_ATTENTION_V2_AUX_ELEM_LIMIT
        && tmp_out_size <= GEMMA4_PAGED_ATTENTION_V2_AUX_ELEM_LIMIT
}

fn gemma4_paged_prefill_aux_limited_chunk_size(
    configured_chunk_size: i32,
    remaining_tokens: usize,
    first_position: u32,
    num_query_heads: u32,
    head_size: u32,
    paged_attention_enabled: bool,
) -> (usize, bool) {
    let base = gemma4_paged_prefill_body_chunk_size(configured_chunk_size, remaining_tokens)
        .min(remaining_tokens)
        .max(1);

    if !paged_attention_enabled
        || gemma4_paged_attention_v2_aux_fits(base, first_position, num_query_heads, head_size)
    {
        return (base, false);
    }

    let mut lo = 1usize;
    let mut hi = base.saturating_sub(1).max(1);
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if gemma4_paged_attention_v2_aux_fits(mid, first_position, num_query_heads, head_size) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }

    (lo.max(1), true)
}

fn gemma4_paged_prefill_body_chunk_plan(
    configured_chunk_size: i32,
    body_tokens: usize,
    first_position: u32,
    num_query_heads: u32,
    head_size: u32,
    paged_attention_enabled: bool,
) -> Result<Vec<Gemma4PagedPrefillBodyChunk>> {
    gemma4_paged_prefill_body_chunk_plan_inner(
        configured_chunk_size,
        body_tokens,
        first_position,
        num_query_heads,
        head_size,
        paged_attention_enabled,
        0,
    )
}

fn gemma4_paged_prefill_body_chunk_plan_with_checkpoint_interval(
    configured_chunk_size: i32,
    body_tokens: usize,
    first_position: u32,
    num_query_heads: u32,
    head_size: u32,
    paged_attention_enabled: bool,
    checkpoint_interval: u32,
) -> Result<Vec<Gemma4PagedPrefillBodyChunk>> {
    gemma4_paged_prefill_body_chunk_plan_inner(
        configured_chunk_size,
        body_tokens,
        first_position,
        num_query_heads,
        head_size,
        paged_attention_enabled,
        checkpoint_interval,
    )
}

fn gemma4_paged_prefill_body_chunk_plan_inner(
    configured_chunk_size: i32,
    body_tokens: usize,
    first_position: u32,
    num_query_heads: u32,
    head_size: u32,
    paged_attention_enabled: bool,
    checkpoint_interval: u32,
) -> Result<Vec<Gemma4PagedPrefillBodyChunk>> {
    let mut chunks = Vec::new();
    let mut start = 0usize;
    let mut position = first_position;
    while start < body_tokens {
        let remaining = body_tokens - start;
        let (mut len, capped_by_v2_aux_limit) = gemma4_paged_prefill_aux_limited_chunk_size(
            configured_chunk_size,
            remaining,
            position,
            num_query_heads,
            head_size,
            paged_attention_enabled,
        );
        if checkpoint_interval > 0 {
            let position_u64 = position as u64;
            let interval_u64 = checkpoint_interval as u64;
            let chunk_end = position_u64
                .checked_add(len as u64)
                .ok_or_else(|| Error::from_reason("Gemma4 paged prefill chunk end overflow"))?;
            let next_checkpoint = ((position_u64 / interval_u64) + 1)
                .checked_mul(interval_u64)
                .ok_or_else(|| {
                    Error::from_reason("Gemma4 paged prefill checkpoint boundary overflow")
                })?;
            if next_checkpoint > position_u64 && next_checkpoint < chunk_end {
                len = usize::try_from(next_checkpoint - position_u64).map_err(|_| {
                    Error::from_reason("Gemma4 paged prefill checkpoint chunk length overflow")
                })?;
            }
        }
        if len == 0 {
            return Err(Error::from_reason(
                "Gemma4 paged prefill dynamic chunking produced an empty chunk",
            ));
        }
        chunks.push(Gemma4PagedPrefillBodyChunk {
            start,
            len,
            first_position: position,
            capped_by_v2_aux_limit,
        });
        start = start
            .checked_add(len)
            .ok_or_else(|| Error::from_reason("Gemma4 paged prefill chunk start overflow"))?;
        position = position
            .checked_add(len as u32)
            .ok_or_else(|| Error::from_reason("Gemma4 paged prefill token position overflow"))?;
    }
    Ok(chunks)
}

/// Evaluate all Gemma4 cache arrays to materialize them on GPU.
/// Must be called between prefill chunks to break lazy dependency chains.
fn eval_gemma4_caches(caches: &[Gemma4LayerCache]) -> Result<()> {
    let mut arrays: Vec<&MxArray> = Vec::new();
    for cache in caches {
        cache.collect_cache_arrays(&mut arrays);
    }
    if !arrays.is_empty() {
        let trace_enabled = inference_trace_enabled();
        let trace_start = trace_enabled.then(std::time::Instant::now);
        MxArray::eval_arrays(&arrays)?;
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] gemma4 eval_caches arrays={} elapsed_ms={:.1}",
                arrays.len(),
                trace_start.map(elapsed_ms).unwrap_or(0.0)
            ));
        }
    }
    Ok(())
}

/// Chunked prefill: process all tokens EXCEPT the last one.
///
/// Matches mlx-lm generate.py generate_step prefill pattern:
/// - The prefill loop processes tokens [0:N-1] (all but the last)
/// - The last token is processed by the caller via `forward_inner`, which
///   also produces the logits used to sample the first output token
///
/// This is CRITICAL for correctness: SDPA computes slightly different numerical
/// results for multi-token causal attention vs single-token attention with cached
/// K/V. These small differences compound through layers, causing divergent logits
/// if the last prompt token is processed in the same batch as the rest.
///
/// 1. Embed ALL tokens once upfront (including PLE if enabled)
/// 2. Run only the transformer body for each chunk (no lm_head)
/// 3. Stop BEFORE the last token — the caller handles it via forward_inner
fn prefill_body_gemma4(
    prompt: &MxArray,
    embedding: &Embedding,
    layers: &[Gemma4DecoderLayer],
    caches: &mut [Gemma4LayerCache],
    final_norm: &RMSNorm,
    ple: Option<&PleComponents>,
    config: &Gemma4Config,
) -> Result<()> {
    let total_len = prompt.shape_at(1)?;

    // Must have at least 2 tokens (1 for prefill, 1 for caller to process)
    if total_len <= 1 {
        return Ok(());
    }

    // Process tokens [0:N-1] — leave last token for the caller
    let prefill_len = total_len - 1;

    // Step 1: Embed tokens [0:N-1]
    let prefill_ids = prompt.slice_axis(1, 0, prefill_len)?;
    let all_embeds = {
        let emb = embedding.forward(&prefill_ids)?;
        emb.mul_scalar((config.hidden_size as f64).sqrt())?
    };

    // Step 2: Compute PLE for prefill tokens (if enabled)
    let all_ple: Option<MxArray> = if let Some(ple) = ple {
        Some(compute_ple(&prefill_ids, &all_embeds, ple, prefill_len)?)
    } else {
        None
    };

    let mut offset: i64 = 0;

    // Process in chunks
    while prefill_len - offset > GEMMA4_PREFILL_STEP_SIZE {
        let chunk_embeds = all_embeds.slice_axis(1, offset, offset + GEMMA4_PREFILL_STEP_SIZE)?;
        let chunk_ple = all_ple
            .as_ref()
            .map(|p| p.slice_axis(1, offset, offset + GEMMA4_PREFILL_STEP_SIZE))
            .transpose()?;

        let _hidden = forward_body(
            None,
            Some(chunk_embeds),
            embedding,
            layers,
            caches,
            final_norm,
            ple,
            chunk_ple.as_ref(),
            config,
        )?;
        eval_gemma4_caches(caches)?;
        crate::array::clear_cache();
        offset += GEMMA4_PREFILL_STEP_SIZE;
    }

    // Final chunk (still body only — no lm_head needed)
    if offset < prefill_len {
        let remaining_embeds = all_embeds.slice_axis(1, offset, prefill_len)?;
        let remaining_ple = all_ple
            .as_ref()
            .map(|p| p.slice_axis(1, offset, prefill_len))
            .transpose()?;

        let _hidden = forward_body(
            None,
            Some(remaining_embeds),
            embedding,
            layers,
            caches,
            final_norm,
            ple,
            remaining_ple.as_ref(),
            config,
        )?;
    }

    Ok(())
}

fn create_sliding_mask(seq_len: i64, offset: i32, window_size: i64) -> Result<MxArray> {
    let total_len = seq_len + offset as i64;
    let rows = MxArray::arange(offset as f64, (offset as i64 + seq_len) as f64, None, None)?;
    let cols = MxArray::arange(0.0, total_len as f64, None, None)?;
    let rows = rows.reshape(&[seq_len, 1])?;
    let cols = cols.reshape(&[1, total_len])?;
    let distance = rows.sub(&cols)?;

    let zero = MxArray::scalar_int(0)?;
    let window = MxArray::scalar_int(window_size as i32)?;
    let causal = distance.greater_equal(&zero)?;
    let in_window = distance.less(&window)?;
    let valid = causal.logical_and(&in_window)?;

    // MLX bool mask semantics are `true = keep`. Returning bool here keeps the
    // mask dtype independent of Gemma4's BF16 residual stream; an additive
    // float32 mask is rejected by `mx.fast.scaled_dot_product_attention` for
    // BF16 Q/K/V because it would promote the output away from BF16.
    valid.reshape(&[1, 1, seq_len, total_len])
}

fn sliding_mask_offset_for_chunk(seq_len: i64, cache_offset: i32, window_size: i64) -> Option<i32> {
    if seq_len <= 1 || window_size <= 0 {
        return None;
    }

    let prior_len = (cache_offset.max(0) as i64).min(window_size);
    if prior_len + seq_len > window_size {
        Some(prior_len as i32)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Vision helpers
// ---------------------------------------------------------------------------

/// Expand image tokens in a token sequence.
///
/// The chat template inserts a single `<|image|>` per image. This function
/// replaces each occurrence with: `boi_token + image_token × num_soft_tokens + eoi_token`.
///
/// If there are fewer `<|image|>` tokens than processed images, the extra images
/// are ignored (manual fallback may not have inserted tokens).
/// If there are no `<|image|>` tokens but images exist, we insert the expanded
/// sequence after the first token (BOS).
fn expand_image_tokens(
    tokens: &[u32],
    processed_images: &[super::image_processor::ProcessedGemma4Image],
    image_token_id: u32,
    boi_token_id: u32,
    eoi_token_id: u32,
) -> Vec<u32> {
    let image_count = tokens.iter().filter(|&&t| t == image_token_id).count();

    if image_count == 0 && !processed_images.is_empty() {
        // Manual fallback: insert expanded tokens after BOS (position 0)
        if tokens.is_empty() {
            return Vec::new();
        }
        let mut result = Vec::with_capacity(
            tokens.len()
                + processed_images
                    .iter()
                    .map(|p| p.num_soft_tokens as usize + 2)
                    .sum::<usize>(),
        );
        result.push(tokens[0]); // BOS
        for proc in processed_images {
            result.push(boi_token_id);
            for _ in 0..proc.num_soft_tokens {
                result.push(image_token_id);
            }
            result.push(eoi_token_id);
        }
        result.extend_from_slice(&tokens[1..]);
        return result;
    }

    // Replace each <|image|> with the expanded BOI + N×image_token + EOI sequence
    let mut result = Vec::with_capacity(tokens.len() * 2);
    let mut img_idx = 0;
    for &t in tokens {
        if t == image_token_id && img_idx < processed_images.len() {
            let num_soft = processed_images[img_idx].num_soft_tokens;
            result.push(boi_token_id);
            for _ in 0..num_soft {
                result.push(image_token_id);
            }
            result.push(eoi_token_id);
            img_idx += 1;
        } else {
            result.push(t);
        }
    }
    result
}

/// masked_scatter: replace positions where mask=true with values from source.
///
/// Matches Python: `mx.where(mask_flat, aligned, input_flat).reshape(input.shape)`
/// where `aligned = source.flatten()[(cumsum(mask_flat) - 1) % source.size]`
fn masked_scatter(input: &MxArray, mask: &MxArray, source: &MxArray) -> Result<MxArray> {
    let input_shape = input.shape()?;
    let mask_flat = mask.reshape(&[-1])?.astype(DType::Int32)?;
    let input_flat = input.reshape(&[-1])?;

    let source_flat = source.reshape(&[-1])?;
    let source_size = source_flat.shape_at(0)?;

    // cumsum of mask gives 1-based indices into source; subtract 1 for 0-based
    let indices = mask_flat.cumsum(0)?.sub(&MxArray::scalar_int(1)?)?;
    // Modulo source_size to handle wrap-around safely
    let source_size_arr = MxArray::scalar_int(source_size as i32)?;
    let safe_indices = indices.remainder(&source_size_arr)?;
    let aligned = source_flat.take(&safe_indices, 0)?;

    // where mask=1 use aligned (source), else keep input
    let result = mask_flat.where_(&aligned, &input_flat)?;
    result.reshape(&input_shape)
}

/// Reports whether `tokens` carry an image or audio placeholder id.
///
/// Used to decide whether a paged text turn may run a content-address prefix
/// lookup. Per-block prefix-cache hashes cover only token ids, not media
/// feature K/V, so a prompt that still holds media placeholders must skip the
/// lookup: otherwise a continue-turn-failure fallback could match the
/// token-only hash of media blocks registered by another session and reuse
/// that session's stale media K/V.
fn prompt_holds_media_placeholders(
    tokens: &[u32],
    image_token_id: u32,
    audio_token_id: u32,
) -> bool {
    tokens.contains(&image_token_id) || tokens.contains(&audio_token_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::gemma4::output_parser::{StreamSegment, parse_gemma4_output};

    #[test]
    fn prompt_holds_media_placeholders_detects_image_audio_and_text() {
        let image_token_id = 258880u32;
        let audio_token_id = 258881u32;

        let image_prompt = [1u32, 2, image_token_id, 3];
        assert!(prompt_holds_media_placeholders(
            &image_prompt,
            image_token_id,
            audio_token_id
        ));

        let audio_prompt = [4u32, audio_token_id, 5];
        assert!(prompt_holds_media_placeholders(
            &audio_prompt,
            image_token_id,
            audio_token_id
        ));

        let text_prompt = [6u32, 7, 8, 9];
        assert!(!prompt_holds_media_placeholders(
            &text_prompt,
            image_token_id,
            audio_token_id
        ));
    }

    /// Pins the composition order `prepare_multimodal_tokens` relies on: on the
    /// manual no-placeholder fallback (tokenizer without a chat template),
    /// audio expansion runs FIRST and image expansion runs LAST, so the image
    /// span lands first after BOS, yielding the canonical
    /// `BOS -> image -> audio -> text` order. If the two expansions were
    /// composed in the old order (image first, audio last) this would produce
    /// `BOS -> audio -> image -> text` and fail.
    #[test]
    fn no_placeholder_fallback_orders_image_before_audio() {
        let image_token_id = 258880u32;
        let audio_token_id = 258881u32;
        let boi = 255999u32;
        let eoi = 258882u32;
        let boa = 256000u32;
        let eoa = 258883u32;
        let bos = 2u32;
        let text = 9u32;

        // No <|image|>/<|audio|> placeholders, one image (3 soft tokens) + one
        // 2-frame audio clip. Audio expansion runs first (inserts after BOS),
        // then image expansion on the audio-expanded stream (also inserts after
        // BOS, so it precedes the audio span).
        let tokens = vec![bos, text];
        let audio_expanded = crate::models::gemma4::audio_processor::expand_audio_tokens(
            &tokens,
            &[2],
            audio_token_id,
            boa,
            eoa,
        )
        .unwrap();
        assert_eq!(
            audio_expanded,
            vec![bos, boa, audio_token_id, audio_token_id, eoa, text],
            "audio fallback inserts its span right after BOS",
        );

        let image = ProcessedGemma4Image {
            pixel_values: MxArray::zeros(&[1, 1], Some(DType::Float32)).unwrap(),
            num_soft_tokens: 3,
            position_ids: None,
        };
        let final_tokens = expand_image_tokens(
            &audio_expanded,
            std::slice::from_ref(&image),
            image_token_id,
            boi,
            eoi,
        );

        // Image span precedes the audio span: BOS, image, audio, text.
        assert_eq!(
            final_tokens,
            vec![
                bos,
                boi,
                image_token_id,
                image_token_id,
                image_token_id,
                eoi,
                boa,
                audio_token_id,
                audio_token_id,
                eoa,
                text,
            ],
            "image runs last in the fallback so its span lands first after BOS",
        );

        // Cross-check: the image markers appear before the audio markers.
        let boi_pos = final_tokens.iter().position(|&t| t == boi).unwrap();
        let boa_pos = final_tokens.iter().position(|&t| t == boa).unwrap();
        assert!(
            boi_pos < boa_pos,
            "image span must precede audio span (boi at {boi_pos}, boa at {boa_pos})",
        );
    }

    #[test]
    fn stream_dispatch_promotes_channel_only_output_to_visible_text() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let sender = StreamSender(&tx);
        let mut state = Gemma4StreamDispatchState::default();

        state.dispatch_segments(
            vec![StreamSegment::Reasoning("final answer".into())],
            &sender,
        );
        assert!(rx.try_recv().is_err());

        state.finish(&sender);
        let chunk = rx.try_recv().unwrap().unwrap();
        assert_eq!(chunk.text, "final answer");
        assert_eq!(chunk.is_reasoning, Some(false));
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn stream_dispatch_keeps_reasoning_when_visible_text_follows() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let sender = StreamSender(&tx);
        let mut state = Gemma4StreamDispatchState::default();

        state.dispatch_segments(
            vec![
                StreamSegment::Reasoning("scratch".into()),
                StreamSegment::Text("answer".into()),
            ],
            &sender,
        );
        state.finish(&sender);

        let reasoning = rx.try_recv().unwrap().unwrap();
        assert_eq!(reasoning.text, "scratch");
        assert_eq!(reasoning.is_reasoning, Some(true));

        let text = rx.try_recv().unwrap().unwrap();
        assert_eq!(text.text, "answer");
        assert_eq!(text.is_reasoning, Some(false));
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn stream_dispatch_keeps_reasoning_when_tool_call_follows() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let sender = StreamSender(&tx);
        let mut state = Gemma4StreamDispatchState::default();

        state.dispatch_segments(
            vec![
                StreamSegment::Reasoning("scratch".into()),
                StreamSegment::ToolCall(crate::tools::ToolCallResult::ok(
                    "tool".into(),
                    serde_json::json!({}),
                    String::new(),
                )),
            ],
            &sender,
        );
        state.finish(&sender);

        let reasoning = rx.try_recv().unwrap().unwrap();
        assert_eq!(reasoning.text, "scratch");
        assert_eq!(reasoning.is_reasoning, Some(true));
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn promote_channel_only_output_moves_thinking_to_text() {
        let mut parsed = parse_gemma4_output("<|channel>thought\nvisible answer<channel|>");
        promote_channel_only_output(&mut parsed);

        assert_eq!(parsed.text, "visible answer");
        assert!(parsed.thinking.is_none());
        assert!(parsed.tool_calls.is_empty());
    }

    #[test]
    fn sliding_mask_is_valid_for_bf16_gqa_attention() {
        let q = MxArray::zeros(&[1, 4, 4, 16], Some(DType::BFloat16)).unwrap();
        let k = MxArray::zeros(&[1, 1, 6, 16], Some(DType::BFloat16)).unwrap();
        let v = MxArray::zeros(&[1, 1, 6, 16], Some(DType::BFloat16)).unwrap();
        let mask = create_sliding_mask(4, 2, 3).unwrap();

        assert_eq!(mask.shape_at(0).unwrap(), 1);
        assert_eq!(mask.shape_at(1).unwrap(), 1);
        assert_eq!(mask.shape_at(2).unwrap(), 4);
        assert_eq!(mask.shape_at(3).unwrap(), 6);

        let out = crate::array::scaled_dot_product_attention(&q, &k, &v, 1.0, Some(&mask)).unwrap();
        let values = out.to_float32().unwrap();
        assert_eq!(values.len(), 4 * 4 * 16);
        assert!(values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn sliding_mask_offset_uses_rotating_window_view() {
        assert_eq!(sliding_mask_offset_for_chunk(512, 16, 1024), None);
        assert_eq!(sliding_mask_offset_for_chunk(512, 528, 1024), Some(528));
        assert_eq!(sliding_mask_offset_for_chunk(512, 43_688, 1024), Some(1024));
        assert_eq!(sliding_mask_offset_for_chunk(2048, 0, 1024), Some(0));
        assert_eq!(sliding_mask_offset_for_chunk(1, 4096, 1024), None);
    }

    #[test]
    fn test_gemma4_paged_prefill_body_chunk_size_honors_configured_size() {
        assert_eq!(
            super::gemma4_paged_prefill_body_chunk_size(4096, 27_938),
            4096
        );
        assert_eq!(
            super::gemma4_paged_prefill_body_chunk_size(512, 27_938),
            512
        );
        assert_eq!(
            super::gemma4_paged_prefill_body_chunk_size(0, 27_938),
            27_938
        );
    }

    #[test]
    fn test_gemma4_paged_prefill_body_chunk_plan_caps_v2_aux() {
        let plan =
            super::gemma4_paged_prefill_body_chunk_plan(8192, 27_938, 16, 16, 512, true).unwrap();
        assert_eq!(plan.first().unwrap().len, 8192);
        assert!(plan.iter().any(|chunk| chunk.capped_by_v2_aux_limit));

        let mut expected_start = 0usize;
        let mut expected_position = 16u32;
        for chunk in &plan {
            assert_eq!(chunk.start, expected_start);
            assert_eq!(chunk.first_position, expected_position);
            assert!(super::gemma4_paged_attention_v2_aux_fits(
                chunk.len,
                chunk.first_position,
                16,
                512
            ));
            expected_start += chunk.len;
            expected_position += chunk.len as u32;
        }
        assert_eq!(expected_start, 27_938);

        let uncapped =
            super::gemma4_paged_prefill_body_chunk_plan(8192, 27_938, 16, 16, 512, false).unwrap();
        assert_eq!(uncapped.len(), 4);
        assert!(uncapped.iter().all(|chunk| !chunk.capped_by_v2_aux_limit));
    }

    #[test]
    fn test_gemma4_sliding_restore_chunk_plan_avoids_singletons() {
        let mut plan =
            super::gemma4_paged_prefill_body_chunk_plan(4, 9, 0, 16, 512, false).unwrap();
        assert_eq!(
            plan.iter().map(|chunk| chunk.len).collect::<Vec<_>>(),
            vec![4, 4, 1]
        );

        super::gemma4_coalesce_single_token_restore_chunks(&mut plan);
        assert_eq!(
            plan.iter().map(|chunk| chunk.len).collect::<Vec<_>>(),
            vec![4, 5]
        );
        assert_eq!(plan[1].start, 4);
        assert_eq!(plan[1].first_position, 4);

        let mut one_token_chunks =
            super::gemma4_paged_prefill_body_chunk_plan(1, 5, 0, 16, 512, false).unwrap();
        super::gemma4_coalesce_single_token_restore_chunks(&mut one_token_chunks);
        assert_eq!(
            one_token_chunks
                .iter()
                .map(|chunk| chunk.len)
                .collect::<Vec<_>>(),
            vec![2, 3]
        );
        assert_eq!(one_token_chunks[1].first_position, 2);
    }

    #[test]
    fn test_gemma4_paged_prefill_chunk_plan_splits_prompt_cache_boundary() {
        let mut plan =
            super::gemma4_paged_prefill_body_chunk_plan(1024, 1432, 44_320, 16, 512, false)
                .unwrap();
        super::gemma4_split_body_chunk_plan_at_position(&mut plan, 45_744);
        assert_eq!(
            plan.iter().map(|chunk| chunk.len).collect::<Vec<_>>(),
            vec![1024, 400, 8]
        );
        assert_eq!(
            plan.iter()
                .map(|chunk| chunk.first_position)
                .collect::<Vec<_>>(),
            vec![44_320, 45_344, 45_744]
        );
        assert_eq!(
            plan.iter().map(|chunk| chunk.start).collect::<Vec<_>>(),
            vec![0, 1024, 1424]
        );

        let mut unchanged = plan.clone();
        super::gemma4_split_body_chunk_plan_at_position(&mut unchanged, 45_344);
        assert_eq!(unchanged, plan);
    }

    #[test]
    fn test_gemma4_paged_prefill_chunk_plan_splits_decode_checkpoint_boundaries() {
        let plan = super::gemma4_paged_prefill_body_chunk_plan_with_checkpoint_interval(
            1024, 3000, 16, 16, 512, false, 1024,
        )
        .unwrap();

        assert_eq!(
            plan.iter().map(|chunk| chunk.len).collect::<Vec<_>>(),
            vec![1008, 1024, 968]
        );
        assert_eq!(
            plan.iter()
                .map(|chunk| chunk.first_position)
                .collect::<Vec<_>>(),
            vec![16, 1024, 2048]
        );
        assert_eq!(
            plan.iter().map(|chunk| chunk.start).collect::<Vec<_>>(),
            vec![0, 1008, 2032]
        );

        let capped = super::gemma4_paged_prefill_body_chunk_plan_with_checkpoint_interval(
            512, 1600, 768, 16, 512, false, 1024,
        )
        .unwrap();
        assert_eq!(
            capped.iter().map(|chunk| chunk.len).collect::<Vec<_>>(),
            vec![256, 512, 512, 320]
        );
        assert!(capped.iter().all(|chunk| chunk.len <= 512));
    }

    #[test]
    fn test_gemma4_sliding_restore_default_is_checkpoint_bounded() {
        let cfg = super::Gemma4Config {
            sliding_window: 1024,
            ..paged_tiny_config(Some(true))
        };

        assert_eq!(
            super::gemma4_large_sliding_restore_suppression_limit_for_override(
                &cfg, 16, None, 1024
            ),
            None
        );
        assert_eq!(
            super::gemma4_large_sliding_restore_suppression_limit_for_override(
                &cfg, 16, None, 24_336
            ),
            Some(super::Gemma4SlidingRestoreSuppression {
                limit: 1024,
                source: "default"
            })
        );
    }

    #[test]
    fn test_gemma4_sliding_restore_env_limit_overrides_default() {
        let cfg = super::Gemma4Config {
            sliding_window: 1024,
            ..paged_tiny_config(Some(true))
        };

        assert_eq!(
            super::parse_gemma4_sliding_restore_limit("32768"),
            Some(super::Gemma4SlidingRestoreLimitOverride::Cap(32_768))
        );
        assert_eq!(
            super::parse_gemma4_sliding_restore_limit(" 44512 "),
            Some(super::Gemma4SlidingRestoreLimitOverride::Cap(44_512))
        );
        assert_eq!(super::parse_gemma4_sliding_restore_limit(""), None);
        assert_eq!(
            super::parse_gemma4_sliding_restore_limit("off"),
            Some(super::Gemma4SlidingRestoreLimitOverride::Uncapped)
        );

        assert_eq!(
            super::gemma4_large_sliding_restore_suppression_limit_for_override(
                &cfg,
                16,
                Some(super::Gemma4SlidingRestoreLimitOverride::Cap(32_768)),
                32_768
            ),
            None
        );
        assert_eq!(
            super::gemma4_large_sliding_restore_suppression_limit_for_override(
                &cfg,
                16,
                Some(super::Gemma4SlidingRestoreLimitOverride::Cap(32_768)),
                44_512
            ),
            Some(super::Gemma4SlidingRestoreSuppression {
                limit: 32_768,
                source: "env"
            })
        );
        assert_eq!(
            super::gemma4_large_sliding_restore_suppression_limit_for_override(
                &cfg,
                16,
                Some(super::Gemma4SlidingRestoreLimitOverride::Uncapped),
                1_000_000
            ),
            None
        );
    }

    #[test]
    fn test_gemma4_chat_manual_fallback_format() {
        // When no chat template exists, manual format should:
        // 1. Start with <bos>
        // 2. Map "assistant" → "model"
        // 3. End with <|turn>model\n
        let messages = vec![
            ("system", "You are helpful."),
            ("user", "Hi"),
            ("assistant", "Hello!"),
            ("user", "Bye"),
        ];
        let mut prompt = String::from("<bos>");
        for (role, content) in &messages {
            let mapped = match *role {
                "assistant" => "model",
                other => other,
            };
            prompt.push_str(&format!("<|turn>{}\n{}<turn|>\n", mapped, content));
        }
        prompt.push_str("<|turn>model\n");

        assert!(prompt.starts_with("<bos><|turn>"), "must start with <bos>");
        assert!(prompt.contains("<|turn>system\nYou are helpful.<turn|>"));
        assert!(prompt.contains("<|turn>model\nHello!<turn|>"));
        assert!(!prompt.contains("<|turn>assistant"));
        assert!(prompt.ends_with("<|turn>model\n"));
    }

    #[test]
    fn test_gemma4_chat_role_mapping() {
        // Verify that "assistant" role gets mapped to "model" in Gemma4 format
        let messages = vec![
            ("user", "Hi"),
            ("assistant", "Hello!"),
            ("user", "How are you?"),
        ];

        let mut prompt_text = String::from("<bos>");
        for (role, content) in &messages {
            let mapped_role = match *role {
                "assistant" => "model",
                other => other,
            };
            prompt_text.push_str(&format!("<|turn>{}\n{}<turn|>\n", mapped_role, content));
        }
        prompt_text.push_str("<|turn>model\n");

        // Verify BOS is present and "assistant" was mapped to "model"
        assert!(prompt_text.starts_with("<bos>"), "must start with <bos>");
        assert!(
            !prompt_text.contains("<|turn>assistant"),
            "assistant role should be mapped to model"
        );
        assert!(
            prompt_text.contains("<|turn>model\nHello!<turn|>"),
            "assistant message should use model role"
        );

        // Verify the full format (with <bos> prefix)
        let expected = "<bos><|turn>user\nHi<turn|>\n<|turn>model\nHello!<turn|>\n<|turn>user\nHow are you?<turn|>\n<|turn>model\n";
        assert_eq!(prompt_text, expected);
    }

    #[test]
    fn test_ple_oov_masking() {
        // Simulate token IDs where some exceed PLE vocab or are negative
        let input_ids = MxArray::from_int32(&[5, 100, 262143, 0, -1], &[1, 5]).unwrap();
        let ple_vocab = 262144i32; // PLE vocab size

        let ple_vocab_arr = MxArray::scalar_int(ple_vocab).unwrap();
        let zero = MxArray::scalar_int(0).unwrap();
        let valid_mask = input_ids
            .greater_equal(&zero)
            .unwrap()
            .logical_and(&input_ids.less(&ple_vocab_arr).unwrap())
            .unwrap();
        let masked_ids = valid_mask.where_(&input_ids, &zero).unwrap();

        masked_ids.eval();
        // IDs within range: unchanged. IDs out of range (negative): mapped to 0.
        assert_eq!(masked_ids.item_at_int32(0).unwrap(), 5); // in range
        assert_eq!(masked_ids.item_at_int32(1).unwrap(), 100); // in range
        // 262143 < 262144, so it's valid
        assert_eq!(masked_ids.item_at_int32(2).unwrap(), 262143);
        assert_eq!(masked_ids.item_at_int32(3).unwrap(), 0); // in range (0 is valid)
        assert_eq!(masked_ids.item_at_int32(4).unwrap(), 0); // -1 is OOV, mapped to 0
    }

    #[test]
    fn test_gemma4_chat_tool_calls_serialization() {
        // Verify tool call args use Gemma4 DSL format (not raw JSON)
        // JSON: {"location": "Paris", "units": "celsius"}
        // DSL:  location:<|"|>Paris<|"|>,units:<|"|>celsius<|"|>  (keys sorted alphabetically)
        let args_json = r#"{"location": "Paris", "units": "celsius"}"#;
        let dsl = json_args_to_gemma4_dsl(args_json);
        assert_eq!(
            dsl, r#"location:<|"|>Paris<|"|>,units:<|"|>celsius<|"|>"#,
            "string values should be wrapped in <|\"|> delimiters, keys sorted alphabetically"
        );

        // Verify numeric and bool values are bare (no quotes)
        let args_with_number = r#"{"count": 5, "active": true}"#;
        let dsl2 = json_args_to_gemma4_dsl(args_with_number);
        assert_eq!(
            dsl2, "active:true,count:5",
            "numbers and bools should be bare (no <|\"|> wrapping), keys sorted alphabetically"
        );

        // Verify format_gemma4_value handles nested JSON objects correctly
        let nested_json = r#"{"temp": 20}"#;
        let nested_val: serde_json::Value = serde_json::from_str(nested_json).unwrap();
        let dsl3 = format_gemma4_value(&nested_val);
        assert_eq!(dsl3, "{temp:20}", "object with bare number value");

        // Build a full prompt matching the manual fallback path
        let mut prompt = String::from("<bos>");

        // user turn
        prompt.push_str("<|turn>user\nWhat's the weather?<turn|>\n");

        // model tool-call turn (assistant → model)
        let tc_dsl = json_args_to_gemma4_dsl(r#"{"location": "Paris", "units": "celsius"}"#);
        prompt.push_str(&format!(
            "<|turn>model\n<|tool_call>call:get_weather{{{}}}<tool_call|><turn|>\n",
            tc_dsl
        ));

        // tool response turn — plain <|turn>tool format (matches HF tokenizer behavior)
        prompt.push_str("<|turn>tool\n{\"temp\": 20}<turn|>\n");

        // final model answer
        prompt.push_str("<|turn>model\nIt's 20 degrees in Paris.<turn|>\n");
        prompt.push_str("<|turn>model\n");

        // Verify DSL format in tool call (no raw JSON quotes)
        assert!(
            prompt.contains(r#"<|tool_call>call:get_weather{location:<|"|>Paris<|"|>,units:<|"|>celsius<|"|>}<tool_call|>"#),
            "tool call args should use Gemma4 DSL with <|\"|> string delimiters"
        );
        assert!(
            !prompt.contains(r#""location""#),
            "tool call should NOT contain raw JSON quoted keys"
        );

        // Verify tool response uses simple <|turn>tool format (not rewritten)
        assert!(
            prompt.contains("<|turn>tool\n"),
            "tool response should use plain <|turn>tool format"
        );
        assert!(
            !prompt.contains("<|tool_response>"),
            "tool response should NOT use <|tool_response> rewriting"
        );

        // Verify assistant→model mapping
        assert!(!prompt.contains("<|turn>assistant"));
    }

    #[test]
    fn test_gemma4_chat_developer_role_mapping() {
        // "developer" role should be mapped to "system"
        let mut prompt = String::from("<bos>");
        let role = "developer";
        let mapped = match role {
            "assistant" => "model",
            "developer" => "system",
            other => other,
        };
        prompt.push_str(&format!(
            "<|turn>{}\nYou are a helpful bot.<turn|>\n",
            mapped
        ));
        prompt.push_str("<|turn>model\n");

        assert!(
            prompt.contains("<|turn>system\nYou are a helpful bot."),
            "developer role should be mapped to system"
        );
        assert!(
            !prompt.contains("<|turn>developer"),
            "developer should not appear as a raw role"
        );
    }

    /// Tiny Gemma4 config compatible with `LayerKVPool`'s validate
    /// constraints (head_size in {32, 64, 96, 128, 256}, FP8 off, etc.).
    /// `head_dim = 32`, num_kv_heads = 2, no PLE/MoE/vision/sharing.
    #[cfg(test)]
    fn paged_tiny_config(use_block_paged: Option<bool>) -> super::Gemma4Config {
        super::Gemma4Config {
            vocab_size: 100,
            hidden_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            head_dim: 32,
            intermediate_size: 64,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: true,
            max_position_embeddings: 128,
            sliding_window: 128,
            // All-global so the uniform paged pool's head_dim choice
            // matches every layer trivially.
            layer_types: vec!["full_attention".to_string(), "full_attention".to_string()],
            rope_theta: 1_000_000.0,
            rope_local_base_freq: 10_000.0,
            partial_rotary_factor: 0.25,
            global_num_key_value_heads: None,
            global_head_dim: None,
            attention_k_eq_v: false,
            is_unified: false,
            use_bidirectional_attention: None,
            final_logit_softcapping: None,
            per_layer_input_embeds: false,
            hidden_size_per_layer_input: None,
            vocab_size_per_layer_input: None,
            pad_token_id: 0,
            eos_token_ids: vec![1],
            bos_token_id: 2,
            attention_bias: false,
            use_double_wide_mlp: false,
            num_kv_shared_layers: None,
            default_temperature: None,
            default_top_k: None,
            default_top_p: None,
            enable_moe_block: false,
            num_experts: None,
            top_k_experts: None,
            moe_intermediate_size: None,
            vision_config: None,
            unified_vision_config: None,
            image_token_id: None,
            boi_token_id: None,
            eoi_token_id: None,
            vision_soft_tokens_per_image: None,
            has_audio: false,
            audio_token_id: None,
            boa_token_id: None,
            eoa_token_id: None,
            audio_samples_per_token: None,
            paged_cache_memory_mb: Some(256),
            paged_block_size: Some(16),
            use_block_paged_cache: use_block_paged,
        }
    }

    /// `use_block_paged_cache` defaults to `None` when absent from the
    /// JSON config — guards against silently switching the storage
    /// backend on existing Gemma4 checkpoints.
    ///
    /// Pure-CPU; no MLX runtime needed.
    #[test]
    fn test_use_block_paged_cache_defaults_to_none_via_serde() {
        let json = serde_json::json!({
            "vocab_size": 0,
            "hidden_size": 0,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "intermediate_size": 1,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": false,
            "max_position_embeddings": 2048,
        });
        let cfg: super::Gemma4Config =
            serde_json::from_value(json).expect("deserialize Gemma4Config");
        assert_eq!(
            cfg.use_block_paged_cache, None,
            "use_block_paged_cache must default to None on JSON without the key"
        );
        assert_eq!(cfg.paged_block_size, None);
        assert_eq!(cfg.paged_cache_memory_mb, None);
    }

    /// `use_block_paged_cache: true` round-trips through serde.
    #[test]
    fn test_use_block_paged_cache_round_trips_true() {
        let json = serde_json::json!({
            "vocab_size": 0,
            "hidden_size": 0,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "intermediate_size": 1,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": false,
            "max_position_embeddings": 2048,
            "use_block_paged_cache": true,
        });
        let cfg: super::Gemma4Config =
            serde_json::from_value(json).expect("deserialize Gemma4Config");
        assert_eq!(cfg.use_block_paged_cache, Some(true));
    }

    #[test]
    fn test_default_paged_cache_memory_covers_gemma4_full_context() {
        let memory_mb = super::gemma4_default_paged_cache_memory_mb(131_072, 16, 512, 2, 5);
        assert_eq!(
            memory_mb, 2560,
            "Gemma4 26B-A4B global KV cache needs 2560MiB to cover 128k tokens"
        );

        let cfg = mlx_paged_attn::PagedAttentionConfig {
            block_size: 16,
            gpu_memory_mb: memory_mb,
            head_size: 512,
            num_kv_heads: 2,
            num_layers: 5,
            use_fp8_cache: Some(false),
            max_seq_len: Some(131_072),
            max_batch_size: Some(32),
        };
        assert_eq!(cfg.calculate_num_blocks(), 8192);
        assert_eq!(cfg.max_cached_tokens(), 131_072);

        let undersized_cfg = mlx_paged_attn::PagedAttentionConfig {
            gpu_memory_mb: 2048,
            ..cfg
        };
        assert!(
            undersized_cfg.max_cached_tokens() < 124_920,
            "the previous fixed 2048MiB default cannot hold the failed 124,920-token prompt"
        );
    }

    #[test]
    fn test_default_paged_cache_memory_respects_minimum() {
        assert_eq!(
            super::gemma4_default_paged_cache_memory_mb(128, 16, 32, 2, 2),
            256
        );
    }

    /// Explicit opt-out (`Some(false)`) must NOT allocate the block-paged
    /// adapter. The previous "None means no adapter" assertion was removed
    /// when the default flipped from `unwrap_or(false)` to `unwrap_or(true)`
    /// — the explicit-false path is the new "no adapter" guarantee.
    #[test]
    fn test_gemma4_inner_no_paged_adapter_when_flag_is_explicit_false() {
        let cfg = paged_tiny_config(Some(false));
        let inner = match super::Gemma4Inner::new(cfg) {
            Ok(i) => i,
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("No Metal device found") {
                    eprintln!("skipping (no Metal device): {msg}");
                    return;
                }
                panic!("unexpected Gemma4Inner::new failure: {msg}");
            }
        };
        assert!(
            inner.paged_adapter.is_none(),
            "paged_adapter must be None when use_block_paged_cache is Some(false)"
        );
    }

    /// Default-flag construction (`None`) must allocate the block-paged
    /// adapter under the new default-on policy (`unwrap_or(true)`).
    /// Allocates a `LayerKVPool`, so requires Metal — gracefully skips
    /// on no-Metal sandboxes.
    #[test]
    fn test_gemma4_inner_paged_adapter_when_flag_is_none_default_on_macos() {
        // Block-paged needs the Metal backend; on a non-Metal build the
        // adapter is gated off (None) and there is nothing to exercise.
        if !crate::engine::persistence::compiled_forward_backend_available() {
            eprintln!("skipping (paged backend unavailable without Metal)");
            return;
        }
        let cfg = paged_tiny_config(None);
        match super::Gemma4Inner::new(cfg) {
            Ok(inner) => {
                assert!(
                    inner.paged_adapter.is_some(),
                    "paged_adapter must be Some when use_block_paged_cache is None \
                     (new default-on policy: unwrap_or(true))"
                );
            }
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("No Metal device found") {
                    eprintln!("skipping (no Metal device): {msg}");
                    return;
                }
                panic!("unexpected Gemma4Inner::new failure: {msg}");
            }
        }
    }

    /// Construction with `use_block_paged_cache: Some(true)` must populate
    /// `paged_adapter`. Allocates a `LayerKVPool`, so requires Metal —
    /// gracefully skips on no-Metal sandboxes.
    #[test]
    fn test_gemma4_inner_constructs_paged_adapter_when_flag_is_true() {
        // Block-paged needs the Metal backend; on a non-Metal build the
        // adapter is gated off (None) and there is nothing to exercise.
        if !crate::engine::persistence::compiled_forward_backend_available() {
            eprintln!("skipping (paged backend unavailable without Metal)");
            return;
        }
        let cfg = paged_tiny_config(Some(true));
        match super::Gemma4Inner::new(cfg) {
            Ok(inner) => {
                assert!(
                    inner.paged_adapter.is_some(),
                    "paged_adapter must be Some when use_block_paged_cache = Some(true)"
                );
            }
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("No Metal device found") {
                    eprintln!("skipping (no Metal device): {msg}");
                    return;
                }
                panic!("unexpected Gemma4Inner::new failure: {msg}");
            }
        }
    }

    /// Contract: a text delta on a media session is governed by BOTH the
    /// `media_session_continuable` marker AND a still-live paged request
    /// (`is_live_for_continue()`), not the raw media key and not the marker
    /// alone. A continuable media turn warm-continues by reading the adapter's
    /// live `block_table`; if the live request is gone (no adapter, or a
    /// shared-adapter `reset_for_new_request` from another session), there is
    /// no live block table to continue and the guard REJECTS with
    /// `IMAGE_CHANGE_RESTART_PREFIX` so the TS floor cold-restarts. When the
    /// marker is false (single-shot: unified image, `reuse_cache=false`, or a
    /// downgraded finalize), the guard also REJECTS exactly as before. The
    /// reject path is preserved for every non-continuable case.
    ///
    /// This test uses a `paged_tiny_config(Some(false))` `Gemma4Inner`, whose
    /// `paged_adapter` is `None` (see the construction-gate test), so
    /// `is_live_for_continue()` is `false`. It therefore exercises:
    ///   - clean session (no media, marker false) → ALLOW,
    ///   - media held + marker false → REJECT (both modalities),
    ///   - media held + marker true but NOT live (the cross-session-released
    ///     hazard) → REJECT, the leak-closing path.
    ///
    /// The marker-true AND live → ALLOW (warm-continue) path needs a live paged
    /// request, which requires real Metal block allocation + a finalized turn
    /// and is not cheaply constructible in a unit test; the single-session 12B
    /// media-continuation e2e proves it instead.
    ///
    /// Constructs a `Gemma4Inner` (needs Metal — gracefully skips on a
    /// no-Metal sandbox) and drives the guard directly by toggling the cached
    /// media keys + the continuable marker.
    #[test]
    fn test_text_delta_after_audio_turn_rejected_like_image_turn() {
        let cfg = paged_tiny_config(Some(false));
        let mut inner = match super::Gemma4Inner::new(cfg) {
            Ok(i) => i,
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("No Metal device found") {
                    eprintln!("skipping (no Metal device): {msg}");
                    return;
                }
                panic!("unexpected Gemma4Inner::new failure: {msg}");
            }
        };

        // `paged_tiny_config(Some(false))` builds no paged adapter, so the
        // session is never live-for-continue — the precondition for the
        // not-live reject assertions below.
        assert!(
            inner.paged_adapter.is_none(),
            "paged_tiny_config(Some(false)) must leave paged_adapter None"
        );

        // Clean session: no media held, marker false, guard passes (None).
        inner.cached_image_key = None;
        inner.cached_audio_key = None;
        inner.media_session_continuable = false;
        assert!(
            inner
                .text_delta_image_guard("chat_session_continue")
                .is_none(),
            "clean session must not reject a text delta"
        );

        // Image turn held, NOT continuable (single-shot): text delta rejected
        // with the restart prefix.
        inner.cached_image_key = Some(42);
        inner.cached_audio_key = None;
        inner.media_session_continuable = false;
        let image_reject = inner
            .text_delta_image_guard("chat_session_continue")
            .expect("text delta after non-continuable image turn must reject");
        assert!(
            image_reject.starts_with(engine::IMAGE_CHANGE_RESTART_PREFIX),
            "image-turn rejection must carry the restart prefix, got: {image_reject}"
        );
        assert!(
            image_reject.contains("image state"),
            "image-turn rejection must mention image state, got: {image_reject}"
        );

        // Audio turn held, NOT continuable: the audio branch must reject the
        // SAME way as the image branch — same restart prefix.
        inner.cached_image_key = None;
        inner.cached_audio_key = Some(7);
        inner.media_session_continuable = false;
        let audio_reject = inner
            .text_delta_image_guard("chat_session_continue")
            .expect("text delta after non-continuable audio turn must reject");
        assert!(
            audio_reject.starts_with(engine::IMAGE_CHANGE_RESTART_PREFIX),
            "audio-turn rejection must carry the restart prefix, got: {audio_reject}"
        );
        assert!(
            audio_reject.contains("audio state"),
            "audio-turn rejection must mention audio state, got: {audio_reject}"
        );

        // Marker armed but NOT live (no live paged request — here no adapter
        // at all; on a shared adapter this is the cross-session-released case):
        // a continuable AUDIO session must REJECT, not warm-continue, because
        // there is no live block_table to read. This is the leak-closing path.
        inner.cached_image_key = None;
        inner.cached_audio_key = Some(7);
        inner.media_session_continuable = true;
        let audio_not_live_reject = inner
            .text_delta_image_guard("chat_session_continue")
            .expect("continuable audio session with no live request must REJECT");
        assert!(
            audio_not_live_reject.starts_with(engine::IMAGE_CHANGE_RESTART_PREFIX),
            "not-live continuable audio rejection must carry the restart prefix, \
             got: {audio_not_live_reject}"
        );
        assert!(
            audio_not_live_reject.contains("audio state"),
            "not-live continuable audio rejection must mention audio state, \
             got: {audio_not_live_reject}"
        );

        // Same for a continuable non-unified IMAGE session with no live request.
        inner.cached_image_key = Some(42);
        inner.cached_audio_key = None;
        inner.media_session_continuable = true;
        let image_not_live_reject = inner
            .text_delta_image_guard("chat_session_continue")
            .expect("continuable image session with no live request must REJECT");
        assert!(
            image_not_live_reject.starts_with(engine::IMAGE_CHANGE_RESTART_PREFIX),
            "not-live continuable image rejection must carry the restart prefix, \
             got: {image_not_live_reject}"
        );
        assert!(
            image_not_live_reject.contains("image state"),
            "not-live continuable image rejection must mention image state, \
             got: {image_not_live_reject}"
        );
    }

    /// Paged/flat parity: a fresh (non-reuse) text-only `save_paged_history`
    /// must clear `cached_audio_key`, exactly as the flat `save_cache_state`
    /// does on a fresh turn. Without that clear, a text-only paged start over a
    /// reused model whose prior turn was audio would leave `cached_audio_key`
    /// stale, and the next text delta's `text_delta_image_guard` would wrongly
    /// force an "audio state" restart on the text-only session. This pins the
    /// fix: pre-fix the post-save key would stay `Some` and the guard would
    /// return the audio-state restart string, failing both asserts below.
    #[test]
    fn test_text_only_paged_save_clears_stale_audio_key() {
        let cfg = paged_tiny_config(Some(false));
        let mut inner = match super::Gemma4Inner::new(cfg) {
            Ok(i) => i,
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("No Metal device found") {
                    eprintln!("skipping (no Metal device): {msg}");
                    return;
                }
                panic!("unexpected Gemma4Inner::new failure: {msg}");
            }
        };

        // Simulate a completed audio turn that left the audio key set, then a
        // fresh text-only paged START (no `reset()`): image key already None,
        // session not continuable.
        inner.cached_audio_key = Some(7);
        inner.cached_image_key = None;
        inner.media_session_continuable = false;

        // Fresh (non-reuse, non-delta) text-only paged save — the same shape
        // the engine uses to persist a fresh text turn's history.
        let save_tokens: Vec<u32> = vec![10, 11, 12];
        let generated: Vec<u32> = vec![20, 21];
        inner
            .save_paged_history(&save_tokens, &generated, false, false)
            .expect("text-only paged save must succeed");

        // The fix: the stale audio key is cleared on the text-only save.
        assert!(
            inner.cached_audio_key.is_none(),
            "text-only paged save must clear the stale audio key"
        );

        // Downstream effect: the next text delta is no longer rejected with an
        // "audio state" restart — the guard returns None on the text-only
        // session. Pre-fix this would be `Some("…holds audio state")`.
        assert!(
            inner
                .text_delta_image_guard("chat_session_continue")
                .is_none(),
            "after a text-only paged save the guard must not force an audio restart"
        );
    }

    /// Image/audio symmetry in `verify_cache_prefix`: a non-continuable session
    /// that still holds a cached AUDIO key must MISS (return `0`), exactly as it
    /// already does for a cached IMAGE key, so stale media KV is reset instead
    /// of being reused as a token-id prefix hit. With an otherwise-hitting
    /// prefix (live caches + matching `cached_token_history`), the audio guard
    /// must override the would-be hit. A continuable audio session (warm-
    /// continue) must NOT be forced to miss by this guard.
    ///
    /// Pre-fix (image-only guard) this would return `cached.len()` for the
    /// non-continuable audio case — a HIT — because the audio key was ignored,
    /// so the first assertion below would fail.
    #[test]
    fn test_verify_cache_prefix_audio_key_forces_miss() {
        let cfg = paged_tiny_config(Some(false));
        let mut inner = match super::Gemma4Inner::new(cfg) {
            Ok(i) => i,
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("No Metal device found") {
                    eprintln!("skipping (no Metal device): {msg}");
                    return;
                }
                panic!("unexpected Gemma4Inner::new failure: {msg}");
            }
        };

        // Build an otherwise-hitting state: live caches + a non-empty cached
        // history that the incoming tokens match as a prefix. `init_caches_sync`
        // also clears reuse state, so the keys/marker/history are set AFTER.
        inner
            .init_caches_sync()
            .expect("init_caches_sync must succeed");
        inner.cached_token_history = vec![100, 101, 102];
        let tokens: Vec<u32> = vec![100, 101, 102, 103];

        // Non-continuable session holding only an AUDIO key: must MISS.
        inner.cached_image_key = None;
        inner.cached_audio_key = Some(7);
        inner.media_session_continuable = false;
        assert_eq!(
            inner.verify_cache_prefix(&tokens, true),
            0,
            "a non-continuable session holding audio state must force a cache miss"
        );

        // Continuable audio session (warm-continue): the guard must NOT force a
        // miss, so the otherwise-hitting prefix returns `cached.len()`.
        inner.media_session_continuable = true;
        assert_eq!(
            inner.verify_cache_prefix(&tokens, true),
            inner.cached_token_history.len(),
            "a continuable audio session must not be forced to miss by the media guard"
        );

        // Parity check: the same shape with an IMAGE key (already guarded) also
        // misses when non-continuable — the audio branch mirrors it exactly.
        inner.cached_image_key = Some(42);
        inner.cached_audio_key = None;
        inner.media_session_continuable = false;
        assert_eq!(
            inner.verify_cache_prefix(&tokens, true),
            0,
            "a non-continuable session holding image state must force a cache miss"
        );
    }

    /// Marker reset matrix: `media_session_continuable` must return to `false`
    /// at every session-reset entry point so a dropped-media session can never
    /// wrongly warm-continue. Covers `clear_reuse_state` and `reset_caches_sync`
    /// (both clear via `clear_reuse_state`).
    #[test]
    fn test_media_session_continuable_reset_matrix() {
        let cfg = paged_tiny_config(Some(false));
        let mut inner = match super::Gemma4Inner::new(cfg) {
            Ok(i) => i,
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("No Metal device found") {
                    eprintln!("skipping (no Metal device): {msg}");
                    return;
                }
                panic!("unexpected Gemma4Inner::new failure: {msg}");
            }
        };

        // Fresh construction: marker defaults to false.
        assert!(
            !inner.media_session_continuable,
            "marker must default to false on construction"
        );

        // clear_reuse_state resets the marker.
        inner.media_session_continuable = true;
        inner.clear_reuse_state();
        assert!(
            !inner.media_session_continuable,
            "clear_reuse_state must reset the continuable marker"
        );

        // reset_caches_sync (which calls clear_reuse_state) resets the marker
        // AND nulls caches → has_live_session() false → a delta cannot continue.
        inner.media_session_continuable = true;
        inner.cached_audio_key = Some(9);
        inner
            .reset_caches_sync()
            .expect("reset_caches_sync must succeed");
        assert!(
            !inner.media_session_continuable,
            "reset_caches_sync must reset the continuable marker"
        );
        assert!(
            inner.cached_audio_key.is_none(),
            "reset_caches_sync must clear the media key"
        );
        // After reset, even toggling the marker can't allow a delta: the
        // session is dead (no live caches), and the reset already cleared it.
        assert!(
            inner
                .text_delta_image_guard("chat_session_continue")
                .is_none(),
            "post-reset session holds no media key → guard returns None (no media to reject)"
        );
    }

    /// Eligibility gate (`gemma4_media_continuable`): ANY image or audio turn is
    /// eligible to warm-continue a text follow-up — audio, non-unified image, AND
    /// the unified bidirectional-vision image. A text-only turn is never a media
    /// turn. Faithfulness is enforced downstream by the `stored && live_for_continue`
    /// gate in `finalize_vision_turn_media_state`, not here.
    #[test]
    fn test_gemma4_media_continuable_gate() {
        let mut inner = match super::Gemma4Inner::new(paged_tiny_config(Some(false))) {
            Ok(i) => i,
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("No Metal device found") {
                    eprintln!("skipping (no Metal device): {msg}");
                    return;
                }
                panic!("unexpected Gemma4Inner::new failure: {msg}");
            }
        };

        let image_id = inner.config.image_token_id.unwrap_or(258880) as u32;
        let audio_id = inner.config.audio_token_id.unwrap_or(258881) as u32;
        let text_tokens: Vec<u32> = vec![10, 11, 12, 13];
        let image_tokens: Vec<u32> = vec![10, image_id, image_id, 13];
        let audio_tokens: Vec<u32> = vec![10, audio_id, audio_id, 13];

        // Text-only: never eligible as a media turn.
        assert!(!inner.gemma4_media_continuable(&text_tokens));

        // Audio is eligible regardless of is_unified / bidirectional config.
        inner.config.is_unified = false;
        inner.config.use_bidirectional_attention = None;
        assert!(inner.gemma4_media_continuable(&audio_tokens));
        inner.config.is_unified = true;
        inner.config.use_bidirectional_attention = Some("vision".to_string());
        assert!(
            inner.gemma4_media_continuable(&audio_tokens),
            "audio is eligible even on a unified ckpt"
        );

        // Non-unified image: eligible.
        inner.config.is_unified = false;
        inner.config.use_bidirectional_attention = None;
        assert!(inner.gemma4_media_continuable(&image_tokens));

        // Unified bidirectional-vision image: now ALSO eligible. The warm text
        // delta routes through the causal text path (no overlay), so it is
        // faithful; the finalize stored/live gate decides whether the checkpoint
        // can actually keep KV live (12B non-shared → warm; e2b KV-shared → cold).
        inner.config.is_unified = true;
        inner.config.use_bidirectional_attention = Some("vision".to_string());
        assert!(
            inner.gemma4_media_continuable(&image_tokens),
            "unified bidirectional-vision image is eligible; faithfulness gated downstream"
        );
    }

    /// All-global config: every layer must route through `GlobalPaged`
    /// with paged_idx == absolute index, no shared layers.
    #[test]
    fn test_compute_layer_kinds_all_global() {
        let cfg = super::Gemma4Config {
            num_hidden_layers: 4,
            layer_types: vec!["full_attention".to_string(); 4],
            ..paged_tiny_config(None)
        };
        let kinds = super::compute_layer_kinds(&cfg);
        assert_eq!(kinds.len(), 4);
        for (i, k) in kinds.iter().enumerate() {
            match k {
                super::Gemma4LayerKind::GlobalPaged { paged_idx } => {
                    assert_eq!(*paged_idx as usize, i, "layer {i} paged_idx mismatch");
                }
                other => panic!("layer {i}: expected GlobalPaged, got {other:?}"),
            }
        }
    }

    /// Hybrid sliding+global with no sharing: paged_idx counts only
    /// global layers in original order; sliding layers map to `Sliding`.
    #[test]
    fn test_compute_layer_kinds_hybrid_no_sharing() {
        // 5-layer cycle: 4 sliding + 1 global, repeated for 10 layers.
        let cycle = ["sliding_attention"; 4]
            .iter()
            .map(|s| s.to_string())
            .chain(std::iter::once("full_attention".to_string()))
            .collect::<Vec<_>>();
        let layer_types: Vec<String> = (0..10).map(|i| cycle[i % 5].clone()).collect();
        let cfg = super::Gemma4Config {
            num_hidden_layers: 10,
            layer_types,
            ..paged_tiny_config(None)
        };
        let kinds = super::compute_layer_kinds(&cfg);
        // Global layers at indices 4 and 9 -> paged_idx 0, 1.
        for (i, k) in kinds.iter().enumerate() {
            if i == 4 {
                assert!(
                    matches!(k, super::Gemma4LayerKind::GlobalPaged { paged_idx: 0 }),
                    "layer 4 must be GlobalPaged{{0}}, got {k:?}"
                );
            } else if i == 9 {
                assert!(
                    matches!(k, super::Gemma4LayerKind::GlobalPaged { paged_idx: 1 }),
                    "layer 9 must be GlobalPaged{{1}}, got {k:?}"
                );
            } else {
                assert!(
                    matches!(k, super::Gemma4LayerKind::Sliding),
                    "layer {i} must be Sliding, got {k:?}"
                );
            }
        }
    }

    /// Smoke test for `paged_turn_sync_core` via direct helper drives.
    ///
    /// Random-init weights cast to BF16 (the paged pool's expected
    /// dtype). Validates the adapter lifecycle (reset →
    /// find_cached_prefix → allocate_suffix → record_tokens →
    /// forward_paged_or_flat) and that produced logits have the
    /// expected shape, without asserting numerical equivalence to the
    /// flat path (random weights). Gracefully skipped on no-Metal.
    #[test]
    fn test_run_paged_prefill_decode_smoke() {
        // Block-paged needs the Metal backend; on a non-Metal build the
        // adapter is gated off (None) and there is nothing to exercise.
        if !crate::engine::persistence::compiled_forward_backend_available() {
            eprintln!("skipping (paged backend unavailable without Metal)");
            return;
        }
        use crate::array::{DType, MxArray};

        let cfg = paged_tiny_config(Some(true));
        let mut inner = match super::Gemma4Inner::new(cfg) {
            Ok(i) => i,
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("No Metal device found") {
                    eprintln!("skipping (no Metal device): {msg}");
                    return;
                }
                panic!("unexpected Gemma4Inner::new failure: {msg}");
            }
        };
        assert!(inner.paged_adapter.is_some());
        if let Err(e) = inner.init_caches_sync() {
            eprintln!("init_caches_sync skipped: {}", e.reason);
            return;
        }

        // Cast all weights to BF16 to match the pool dtype. Mirrors
        // LFM2's smoke-test cast pattern.
        let cast = |a: &MxArray| -> MxArray { a.astype(DType::BFloat16).expect("astype BFloat16") };
        let w = inner.embed_tokens.get_weight();
        inner.embed_tokens.set_weight(&cast(&w)).expect("embed");
        let w = inner.final_norm.get_weight();
        inner.final_norm.set_weight(&cast(&w)).expect("final_norm");
        if let Some(ref mut head) = inner.lm_head {
            let w = head.get_weight();
            head.set_weight(&cast(&w), "lm_head").expect("lm_head");
        }
        for layer in inner.layers.iter_mut() {
            // Norms.
            layer
                .set_input_layernorm_weight(&cast(&layer.input_layernorm_weight().clone()))
                .ok();
            layer
                .set_post_attention_layernorm_weight(&cast(
                    &layer.post_attention_layernorm_weight().clone(),
                ))
                .ok();
            layer
                .set_pre_feedforward_layernorm_weight(&cast(
                    &layer.pre_feedforward_layernorm_weight().clone(),
                ))
                .ok();
            layer
                .set_post_feedforward_layernorm_weight(&cast(
                    &layer.post_feedforward_layernorm_weight().clone(),
                ))
                .ok();
            // Attention projections + norms.
            let attn = &mut layer.self_attn;
            let w = attn.q_proj_weight();
            attn.set_q_proj_weight(&cast(&w)).expect("q");
            let w = attn.k_proj_weight();
            attn.set_k_proj_weight(&cast(&w)).expect("k");
            if let Some(w) = attn.v_proj_weight_opt() {
                attn.set_v_proj_weight(&cast(&w)).expect("v");
            }
            let w = attn.o_proj_weight();
            attn.set_o_proj_weight(&cast(&w)).expect("o");
            let w = attn.q_norm_weight();
            attn.set_q_norm_weight(&cast(&w)).expect("qn");
            let w = attn.k_norm_weight();
            attn.set_k_norm_weight(&cast(&w)).expect("kn");
            // MLP.
            if let crate::models::gemma4::quantized_linear::Gemma4MLPVariant::Standard(
                ref mut mlp,
            ) = layer.mlp
            {
                let w = mlp.gate_proj_weight();
                mlp.set_gate_proj_weight(&cast(&w)).expect("gate");
                let w = mlp.up_proj_weight();
                mlp.set_up_proj_weight(&cast(&w)).expect("up");
                let w = mlp.down_proj_weight();
                mlp.set_down_proj_weight(&cast(&w)).expect("down");
            }
        }

        // Adapter lifecycle.
        let prompt: Vec<u32> = vec![1, 2, 3, 4];
        if let Some(adapter) = inner.paged_adapter.as_mut() {
            if let Err(e) = adapter.reset_for_new_request(0) {
                eprintln!("skipping (adapter reset failed): {e}");
                return;
            }
            if let Err(e) = adapter.find_cached_prefix(&prompt, &[], 0, false) {
                eprintln!("skipping (find_cached_prefix failed): {e}");
                return;
            }
            if let Err(e) = adapter.allocate_suffix_blocks(16) {
                eprintln!("skipping (allocate_suffix_blocks failed): {e}");
                return;
            }
        }

        let last_logits = match inner.run_paged_prefill_chunk(&prompt, &prompt, 0, 0) {
            Ok(l) => l,
            Err(e) => {
                let msg = e.reason.to_string();
                if msg.contains("No Metal device found") || msg.contains("not supported") {
                    eprintln!("skipping smoke: {msg}");
                    return;
                }
                panic!("run_paged_prefill_chunk failed: {msg}");
            }
        };
        let vocab = last_logits.shape_at(0).expect("shape");
        assert_eq!(vocab, 100, "vocab_size from paged_tiny_config");

        let mut next_token: u32 = 5;
        for _ in 0..4 {
            match inner.run_paged_decode_step(next_token) {
                Ok(logits) => {
                    assert_eq!(logits.shape_at(0).expect("shape"), 1);
                    assert_eq!(logits.shape_at(1).expect("shape"), 1);
                    assert_eq!(logits.shape_at(2).expect("shape"), 100);
                }
                Err(e) => {
                    let msg = e.reason.to_string();
                    if msg.contains("No Metal device found") {
                        eprintln!("skipping decode (no Metal): {msg}");
                        return;
                    }
                    panic!("run_paged_decode_step failed: {msg}");
                }
            }
            next_token = next_token.wrapping_add(1);
        }

        if let Some(adapter) = inner.paged_adapter.as_mut() {
            let _ = adapter.release_request();
        }
    }

    #[test]
    fn test_gemma4_prompt_boundary_checkpoint_survives_decode_checkpoint_eviction() {
        let cfg = paged_tiny_config(Some(true));
        let mut inner = match super::Gemma4Inner::new(cfg) {
            Ok(i) => i,
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("No Metal device found") {
                    eprintln!("skipping (no Metal device): {msg}");
                    return;
                }
                panic!("unexpected Gemma4Inner::new failure: {msg}");
            }
        };

        let block_size = 16;
        let prompt: Vec<u32> = (10..26).collect();
        let prompt_hash = super::compute_gemma4_paged_prefix_block_hash(
            &prompt,
            prompt.len() as u32,
            block_size,
            0,
        )
        .expect("prompt hash");
        inner.sliding_prompt_boundary_checkpoint = Some(super::Gemma4SlidingPrefixCheckpoint {
            prefix_len: prompt.len() as u32,
            block_size,
            final_block_hash: prompt_hash,
            tokens: prompt.clone(),
            snapshots: vec![None; inner.config.num_hidden_layers as usize],
        });

        let checkpoint_limit = super::gemma4_sliding_prefix_checkpoint_limit_for_override(
            &inner.config,
            block_size,
            None,
        );
        for i in 0..(checkpoint_limit + 3) {
            let tokens: Vec<u32> = (0..16).map(|token| 100 + i as u32 + token).collect();
            inner
                .sliding_prefix_checkpoints
                .push_back(super::Gemma4SlidingPrefixCheckpoint {
                    prefix_len: tokens.len() as u32,
                    block_size,
                    final_block_hash: i as u64 + 1,
                    tokens,
                    snapshots: vec![None; inner.config.num_hidden_layers as usize],
                });
            while inner.sliding_prefix_checkpoints.len() > checkpoint_limit {
                inner.sliding_prefix_checkpoints.pop_front();
            }
        }
        assert_eq!(inner.sliding_prefix_checkpoints.len(), checkpoint_limit);

        let restored = inner
            .find_gemma4_sliding_prefix_checkpoint(&prompt, prompt.len() as u32, block_size, 0)
            .expect("prefix lookup");
        assert!(
            restored.is_some(),
            "prompt-boundary checkpoint must not be evicted by decode-boundary checkpoints"
        );
    }

    #[test]
    fn test_gemma4_decode_checkpoint_retains_recent_retokenization_drift() {
        let cfg = paged_tiny_config(Some(true));
        let mut inner = match super::Gemma4Inner::new(cfg) {
            Ok(i) => i,
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("No Metal device found") {
                    eprintln!("skipping (no Metal device): {msg}");
                    return;
                }
                panic!("unexpected Gemma4Inner::new failure: {msg}");
            }
        };

        let block_size = 16;
        let target_tokens: Vec<u32> = (1000..1016).collect();
        let target_hash = super::compute_gemma4_paged_prefix_block_hash(
            &target_tokens,
            target_tokens.len() as u32,
            block_size,
            0,
        )
        .expect("target hash");
        inner
            .sliding_prefix_checkpoints
            .push_back(super::Gemma4SlidingPrefixCheckpoint {
                prefix_len: target_tokens.len() as u32,
                block_size,
                final_block_hash: target_hash,
                tokens: target_tokens.clone(),
                snapshots: vec![None; inner.config.num_hidden_layers as usize],
            });

        // The observed Gemma4 tool-call retokenization drift needed the
        // checkpoint five block boundaries behind the final decode state:
        // 46272 was requested after 46288, 46304, 46320, and 46336 had
        // also been checkpointed.
        for i in 0..4 {
            let tokens: Vec<u32> = (0..16).map(|token| 2000 + i as u32 + token).collect();
            let hash = super::compute_gemma4_paged_prefix_block_hash(
                &tokens,
                tokens.len() as u32,
                block_size,
                0,
            )
            .expect("newer hash");
            inner
                .sliding_prefix_checkpoints
                .push_back(super::Gemma4SlidingPrefixCheckpoint {
                    prefix_len: tokens.len() as u32,
                    block_size,
                    final_block_hash: hash,
                    tokens,
                    snapshots: vec![None; inner.config.num_hidden_layers as usize],
                });
            let checkpoint_limit = super::gemma4_sliding_prefix_checkpoint_limit_for_override(
                &inner.config,
                block_size,
                None,
            );
            while inner.sliding_prefix_checkpoints.len() > checkpoint_limit {
                inner.sliding_prefix_checkpoints.pop_front();
            }
        }

        let restored = inner
            .find_gemma4_sliding_prefix_checkpoint(
                &target_tokens,
                target_tokens.len() as u32,
                block_size,
                0,
            )
            .expect("prefix lookup");
        assert!(
            restored.is_some(),
            "decode checkpoints must retain the block needed after modest retokenization drift"
        );
    }

    #[test]
    fn test_gemma4_decode_checkpoint_retains_sliding_window_drift() {
        let mut cfg = paged_tiny_config(Some(true));
        cfg.sliding_window = 512;
        let mut inner = match super::Gemma4Inner::new(cfg) {
            Ok(i) => i,
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("No Metal device found") {
                    eprintln!("skipping (no Metal device): {msg}");
                    return;
                }
                panic!("unexpected Gemma4Inner::new failure: {msg}");
            }
        };

        let block_size = 16;
        let checkpoint_limit = super::gemma4_sliding_prefix_checkpoint_limit_for_override(
            &inner.config,
            block_size,
            None,
        );
        assert_eq!(
            checkpoint_limit, 64,
            "512-token sliding window with 16-token blocks should retain two windows of decode checkpoints"
        );
        let target_tokens: Vec<u32> = (3000..3016).collect();
        let target_hash = super::compute_gemma4_paged_prefix_block_hash(
            &target_tokens,
            target_tokens.len() as u32,
            block_size,
            0,
        )
        .expect("target hash");
        inner
            .sliding_prefix_checkpoints
            .push_back(super::Gemma4SlidingPrefixCheckpoint {
                prefix_len: target_tokens.len() as u32,
                block_size,
                final_block_hash: target_hash,
                tokens: target_tokens.clone(),
                snapshots: vec![None; inner.config.num_hidden_layers as usize],
            });

        // The live 2026-05-09 Gemma4 trace needed a checkpoint eighteen
        // block boundaries behind the final decode state (57072 requested
        // after decode reached 57360). A one-window default retains that
        // level of retokenization drift instead of forcing a full replay.
        for i in 0..18 {
            let token_base = 4000 + (i as u32 * block_size);
            let tokens: Vec<u32> = (0..block_size).map(|token| token_base + token).collect();
            let hash = super::compute_gemma4_paged_prefix_block_hash(
                &tokens,
                tokens.len() as u32,
                block_size,
                0,
            )
            .expect("newer hash");
            inner
                .sliding_prefix_checkpoints
                .push_back(super::Gemma4SlidingPrefixCheckpoint {
                    prefix_len: tokens.len() as u32,
                    block_size,
                    final_block_hash: hash,
                    tokens,
                    snapshots: vec![None; inner.config.num_hidden_layers as usize],
                });
            while inner.sliding_prefix_checkpoints.len() > checkpoint_limit {
                inner.sliding_prefix_checkpoints.pop_front();
            }
        }

        let restored = inner
            .find_gemma4_sliding_prefix_checkpoint(
                &target_tokens,
                target_tokens.len() as u32,
                block_size,
                0,
            )
            .expect("prefix lookup");
        assert!(
            restored.is_some(),
            "decode checkpoints must retain one sliding-window worth of retokenization drift"
        );
    }

    #[test]
    fn test_gemma4_decode_checkpoint_retains_auxiliary_branch_interleaving() {
        let mut cfg = paged_tiny_config(Some(true));
        cfg.sliding_window = 1024;
        let mut inner = match super::Gemma4Inner::new(cfg) {
            Ok(i) => i,
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("No Metal device found") {
                    eprintln!("skipping (no Metal device): {msg}");
                    return;
                }
                panic!("unexpected Gemma4Inner::new failure: {msg}");
            }
        };

        let block_size = 16;
        let checkpoint_limit = super::gemma4_sliding_prefix_checkpoint_limit_for_override(
            &inner.config,
            block_size,
            None,
        );
        assert_eq!(
            checkpoint_limit, 128,
            "1024-token sliding window with 16-token blocks should retain two windows"
        );
        let target_tokens: Vec<u32> = (10_000..10_016).collect();
        let target_hash = super::compute_gemma4_paged_prefix_block_hash(
            &target_tokens,
            target_tokens.len() as u32,
            block_size,
            0,
        )
        .expect("target hash");
        inner
            .sliding_prefix_checkpoints
            .push_back(super::Gemma4SlidingPrefixCheckpoint {
                prefix_len: target_tokens.len() as u32,
                block_size,
                final_block_hash: target_hash,
                tokens: target_tokens.clone(),
                snapshots: vec![None; inner.config.num_hidden_layers as usize],
            });

        // The 2026-05-09 live trace stored the needed 48,416-token
        // checkpoint, then 93 checkpoints from auxiliary 29k/33k branches
        // before the main branch asked for 48,416 again. A one-window FIFO
        // cap evicted it; two windows retains it without unbounded growth.
        for i in 0..93 {
            let token_base = 20_000 + (i as u32 * block_size);
            let tokens: Vec<u32> = (0..block_size).map(|token| token_base + token).collect();
            let hash = super::compute_gemma4_paged_prefix_block_hash(
                &tokens,
                tokens.len() as u32,
                block_size,
                0,
            )
            .expect("newer hash");
            inner
                .sliding_prefix_checkpoints
                .push_back(super::Gemma4SlidingPrefixCheckpoint {
                    prefix_len: tokens.len() as u32,
                    block_size,
                    final_block_hash: hash,
                    tokens,
                    snapshots: vec![None; inner.config.num_hidden_layers as usize],
                });
            super::trim_gemma4_sliding_prefix_checkpoints(
                &mut inner.sliding_prefix_checkpoints,
                checkpoint_limit,
                false,
            );
        }

        let restored = inner
            .find_gemma4_sliding_prefix_checkpoint(
                &target_tokens,
                target_tokens.len() as u32,
                block_size,
                0,
            )
            .expect("prefix lookup");
        assert!(
            restored.is_some(),
            "decode checkpoints must survive auxiliary branch interleaving seen in live sessions"
        );
    }

    #[test]
    fn test_gemma4_sliding_decode_checkpoint_interval_uses_window_stride() {
        let mut cfg = paged_tiny_config(Some(true));
        cfg.sliding_window = 1024;
        assert_eq!(
            super::gemma4_sliding_decode_checkpoint_interval(&cfg, 16),
            1024
        );

        cfg.sliding_window = 1000;
        assert_eq!(
            super::gemma4_sliding_decode_checkpoint_interval(&cfg, 16),
            1008,
            "checkpoint interval should stay aligned to paged block boundaries"
        );
    }

    #[test]
    fn test_gemma4_sliding_prefix_checkpoint_restores_nearest_prefix() {
        let cfg = paged_tiny_config(Some(true));
        let mut inner = match super::Gemma4Inner::new(cfg) {
            Ok(i) => i,
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("No Metal device found") {
                    eprintln!("skipping (no Metal device): {msg}");
                    return;
                }
                panic!("unexpected Gemma4Inner::new failure: {msg}");
            }
        };

        let block_size = 16;
        let tokens: Vec<u32> = (0..1280).map(|token| 50_000 + token).collect();
        let checkpoint_len = 1024;
        let checkpoint_hash =
            super::compute_gemma4_paged_prefix_block_hash(&tokens, checkpoint_len, block_size, 0)
                .expect("checkpoint hash");
        inner
            .sliding_prefix_checkpoints
            .push_back(super::Gemma4SlidingPrefixCheckpoint {
                prefix_len: checkpoint_len,
                block_size,
                final_block_hash: checkpoint_hash,
                tokens: tokens[..checkpoint_len as usize].to_vec(),
                snapshots: vec![None; inner.config.num_hidden_layers as usize],
            });

        let hit = inner
            .find_gemma4_sliding_prefix_checkpoint(&tokens, tokens.len() as u32, block_size, 0)
            .expect("prefix lookup")
            .expect("nearest checkpoint hit");
        assert_eq!(hit.prefix_len, checkpoint_len);
    }

    #[test]
    fn test_gemma4_mid_prompt_prefix_hit_uses_near_prefill_checkpoint() {
        let mut cfg = paged_tiny_config(Some(true));
        cfg.sliding_window = 1024;
        let mut inner = match super::Gemma4Inner::new(cfg) {
            Ok(i) => i,
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("No Metal device found") {
                    eprintln!("skipping (no Metal device): {msg}");
                    return;
                }
                panic!("unexpected Gemma4Inner::new failure: {msg}");
            }
        };

        let block_size = 16;
        let cached_prefix_len = 24_352;
        let checkpoint_len = 23_552;
        let tokens: Vec<u32> = (0..cached_prefix_len).map(|token| 90_000 + token).collect();
        let checkpoint_hash =
            super::compute_gemma4_paged_prefix_block_hash(&tokens, checkpoint_len, block_size, 0)
                .expect("checkpoint hash");
        inner
            .sliding_prefix_checkpoints
            .push_back(super::Gemma4SlidingPrefixCheckpoint {
                prefix_len: checkpoint_len,
                block_size,
                final_block_hash: checkpoint_hash,
                tokens: tokens[..checkpoint_len as usize].to_vec(),
                snapshots: vec![None; inner.config.num_hidden_layers as usize],
            });

        let hit = inner
            .find_gemma4_sliding_prefix_checkpoint(&tokens, cached_prefix_len, block_size, 0)
            .expect("prefix lookup")
            .expect("near checkpoint hit");
        assert_eq!(hit.prefix_len, checkpoint_len);
        assert_eq!(cached_prefix_len - hit.prefix_len, 800);
        assert_eq!(
            super::gemma4_large_sliding_restore_suppression_limit(
                &inner.config,
                block_size,
                cached_prefix_len - hit.prefix_len
            ),
            None,
            "a one-window prefill checkpoint should prevent cold-prefill suppression"
        );
    }

    #[test]
    fn test_gemma4_window_stride_checkpoints_retain_old_branch_prefix() {
        let mut cfg = paged_tiny_config(Some(true));
        cfg.sliding_window = 1024;
        let mut inner = match super::Gemma4Inner::new(cfg) {
            Ok(i) => i,
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("No Metal device found") {
                    eprintln!("skipping (no Metal device): {msg}");
                    return;
                }
                panic!("unexpected Gemma4Inner::new failure: {msg}");
            }
        };

        let block_size = 16;
        let target_len = 36_096;
        let target_tokens: Vec<u32> = (0..target_len).map(|token| 70_000 + token).collect();
        let target_hash = super::compute_gemma4_paged_prefix_block_hash(
            &target_tokens,
            target_len,
            block_size,
            0,
        )
        .expect("target hash");
        inner
            .sliding_prefix_checkpoints
            .push_back(super::Gemma4SlidingPrefixCheckpoint {
                prefix_len: target_len,
                block_size,
                final_block_hash: target_hash,
                tokens: target_tokens.clone(),
                snapshots: vec![None; inner.config.num_hidden_layers as usize],
            });

        let checkpoint_limit = super::gemma4_sliding_prefix_checkpoint_limit_for_override(
            &inner.config,
            block_size,
            None,
        );
        let interval = super::gemma4_sliding_decode_checkpoint_interval(&inner.config, block_size);
        assert_eq!(interval, 1024);
        assert_eq!(checkpoint_limit, 128);

        for i in 0..96 {
            let prefix_len = 80_000 + i as u32 * interval;
            let tokens: Vec<u32> = (0..prefix_len).map(|token| 200_000 + token).collect();
            let hash =
                super::compute_gemma4_paged_prefix_block_hash(&tokens, prefix_len, block_size, 0)
                    .expect("newer hash");
            inner
                .sliding_prefix_checkpoints
                .push_back(super::Gemma4SlidingPrefixCheckpoint {
                    prefix_len,
                    block_size,
                    final_block_hash: hash,
                    tokens,
                    snapshots: vec![None; inner.config.num_hidden_layers as usize],
                });
            super::trim_gemma4_sliding_prefix_checkpoints(
                &mut inner.sliding_prefix_checkpoints,
                checkpoint_limit,
                false,
            );
        }

        let hit = inner
            .find_gemma4_sliding_prefix_checkpoint(
                &target_tokens,
                target_tokens.len() as u32,
                block_size,
                0,
            )
            .expect("prefix lookup")
            .expect("old branch checkpoint hit");
        assert_eq!(hit.prefix_len, target_len);
    }

    /// KV-shared layers must resolve their anchor's pool slot
    /// (SharedOnGlobal) or absolute index (SharedOnSliding).
    #[test]
    fn test_compute_layer_kinds_kv_sharing_resolves_anchors() {
        // 8 layers: pattern S G S G S G S G (4 global @ 1, 3, 5, 7).
        // num_kv_shared_layers = 4 → last 4 (indices 4, 5, 6, 7) reuse anchors.
        // Anchor for shared global at i=5 should be the last non-shared
        // global before first_kv_shared_layer (=4): that's i=3 → paged_idx=1.
        // Anchor for shared sliding at i=4 should be sliding at i=2.
        let layer_types: Vec<String> = (0..8)
            .map(|i| {
                if i % 2 == 1 {
                    "full_attention".to_string()
                } else {
                    "sliding_attention".to_string()
                }
            })
            .collect();
        let cfg = super::Gemma4Config {
            num_hidden_layers: 8,
            layer_types,
            num_kv_shared_layers: Some(4),
            ..paged_tiny_config(None)
        };
        let kinds = super::compute_layer_kinds(&cfg);
        // Non-shared layers: 0=Sliding, 1=GlobalPaged{0}, 2=Sliding, 3=GlobalPaged{1}.
        assert!(matches!(kinds[0], super::Gemma4LayerKind::Sliding));
        assert!(matches!(
            kinds[1],
            super::Gemma4LayerKind::GlobalPaged { paged_idx: 0 }
        ));
        assert!(matches!(kinds[2], super::Gemma4LayerKind::Sliding));
        assert!(matches!(
            kinds[3],
            super::Gemma4LayerKind::GlobalPaged { paged_idx: 1 }
        ));
        // Shared layers 4..8 are aliases. They do not consume paged slots;
        // SharedOnGlobal carries the ANCHOR's pool slot, and
        // SharedOnSliding carries the anchor's absolute layer index.
        match kinds[4] {
            super::Gemma4LayerKind::SharedOnSliding { anchor_layer_idx } => {
                assert_eq!(anchor_layer_idx, 2, "anchor for sliding-shared layer 4");
            }
            ref other => panic!("layer 4: expected SharedOnSliding, got {other:?}"),
        }
        match kinds[5] {
            super::Gemma4LayerKind::SharedOnGlobal { anchor_paged_idx } => {
                // Anchor at layer 3 → paged_idx 1.
                assert_eq!(anchor_paged_idx, 1, "anchor paged_idx for global-shared 5");
            }
            ref other => panic!("layer 5: expected SharedOnGlobal, got {other:?}"),
        }
        match kinds[6] {
            super::Gemma4LayerKind::SharedOnSliding { anchor_layer_idx } => {
                assert_eq!(anchor_layer_idx, 2, "anchor for sliding-shared layer 6");
            }
            ref other => panic!("layer 6: expected SharedOnSliding, got {other:?}"),
        }
        match kinds[7] {
            super::Gemma4LayerKind::SharedOnGlobal { anchor_paged_idx } => {
                assert_eq!(anchor_paged_idx, 1, "anchor paged_idx for global-shared 7");
            }
            ref other => panic!("layer 7: expected SharedOnGlobal, got {other:?}"),
        }
    }

    #[test]
    fn test_compute_layer_kv_cache_specs_group_full_sliding_and_shared_aliases() {
        let layer_types: Vec<String> = (0..8)
            .map(|i| {
                if i % 2 == 1 {
                    "full_attention".to_string()
                } else {
                    "sliding_attention".to_string()
                }
            })
            .collect();
        let cfg = super::Gemma4Config {
            num_hidden_layers: 8,
            layer_types,
            num_kv_shared_layers: Some(4),
            sliding_window: 17,
            max_position_embeddings: 128,
            ..paged_tiny_config(None)
        };

        let specs =
            super::compute_layer_kv_cache_specs(&cfg, 8, super::KVCacheDType::BFloat16).unwrap();
        assert_eq!(specs.len(), 8);
        assert_eq!(specs[4].shared_kv_anchor, Some(2));
        assert_eq!(specs[5].shared_kv_anchor, Some(3));
        assert_eq!(super::physical_full_attention_layer_count(&specs), 2);

        let groups =
            super::compute_layer_kv_cache_groups(&cfg, 8, super::KVCacheDType::BFloat16, 32)
                .unwrap();
        let full_group = groups
            .iter()
            .find(|group| matches!(group.attention_kind, super::AttentionKind::Full))
            .expect("full group");
        assert_eq!(full_group.layer_indices, vec![1, 3, 5, 7]);
        assert_eq!(full_group.physical_layer_indices, vec![1, 3]);

        let sliding_group = groups
            .iter()
            .find(|group| {
                matches!(
                    group.attention_kind,
                    super::AttentionKind::SlidingWindow { sliding_window: 17 }
                )
            })
            .expect("sliding group");
        assert_eq!(sliding_group.layer_indices, vec![0, 2, 4, 6]);
        assert_eq!(sliding_group.physical_layer_indices, vec![0, 2]);
        assert_eq!(
            sliding_group.max_admission_blocks, 7,
            "ceil((17 - 1 + 32) / 8) + one partial block"
        );
    }
}

#[cfg(test)]
mod prefix_cache_reuse_integration_tests {
    //! End-to-end tests for the prefix KV cache reuse refactor on Gemma4.
    //! These verify that `chat_session_start_sync` no longer
    //! unconditionally wipes the cache — stateless agent clients that
    //! resend the full transcript on every turn should hit the
    //! `verify_cache_prefix` exact-append path and skip redundant
    //! prefill work.
    //!
    //! The Gemma4 variant additionally locks in the exact-match policy:
    //! when the new prompt equals the cached one
    //! (`cached_prefix_len == tokens.len()`), we fall through to the
    //! miss branch and do a full reset + re-prefill. Gemma4 has no
    //! snapshot of final-step logits and no safe rewind-by-1 primitive
    //! over its sliding-window cache; reprefilling the last cached token
    //! on top of the live caches would advance cache state to
    //! `prompt + last_token` (duplicated) while the history write-back
    //! block only persists `tokens + generated`, corrupting the next
    //! warm-hit turn.
    //!
    //! These tests are `#[ignore]`-marked because they require loading a
    //! real Gemma4 model file and a tokenizer. Run them with:
    //!
    //!     cargo test -p mlx-core --test '*' -- --ignored prefix_cache_reuse_integration
    //!
    //! with `MLX_NODE_GEMMA4_MODEL_DIR` set to a local Gemma4 model dir.

    /// Append hit: two back-to-back session-start calls where the second
    /// extends the first by exactly one user turn. Must report
    /// `cached_tokens > 0` and only prefill the delta.
    #[ignore = "requires a real Gemma4 model directory; run with --ignored"]
    #[test]
    fn append_hit_reuses_cached_prefix() {
        // Pseudocode (same shape as the Qwen3.5 Dense stubs):
        //
        //   let p = vec![ChatMessage::user("Hi")];
        //   let r1 = model.chat_session_start_sync(p.clone(), cfg())?;
        //   let mut p2 = p.clone();
        //   p2.push(ChatMessage::assistant(&r1.text));
        //   p2.push(ChatMessage::user("Follow-up"));
        //   let r2 = model.chat_session_start_sync(p2, cfg())?;
        //   assert!(r2.cached_tokens > 0);
    }

    /// Divergence miss: second call's history is unrelated. Must report
    /// `cached_tokens == 0` and do a full-history prefill.
    #[ignore = "requires a real Gemma4 model directory; run with --ignored"]
    #[test]
    fn divergence_miss_resets_and_full_prefills() {
        // Pseudocode:
        //
        //   let p1 = vec![ChatMessage::user("Ping")];
        //   let p2 = vec![ChatMessage::user("Totally unrelated")];
        //   let _ = model.chat_session_start_sync(p1, cfg())?;
        //   let r2 = model.chat_session_start_sync(p2, cfg())?;
        //   assert_eq!(r2.cached_tokens, 0);
    }

    /// Exact-match: the new prompt is byte-equal to the cached one.
    /// With the exact-match-as-miss fix, the second call must report
    /// `cached_tokens == 0` (full reset + full re-prefill). A subsequent
    /// strict-extension must then hit the warm path.
    #[ignore = "requires a real Gemma4 model directory; run with --ignored"]
    #[test]
    fn exact_match_falls_through_to_cache_miss() {
        // Pseudocode:
        //
        //   let p = vec![ChatMessage::user("Ping")];
        //   let _ = model.chat_session_start_sync(p.clone(), cfg())?;
        //   let r2 = model.chat_session_start_sync(p.clone(), cfg())?;
        //   assert_eq!(r2.cached_tokens, 0); // miss, not exact-match reuse
        //
        //   // After the miss, the caches represent `p` cleanly. A strict
        //   // extension should warm-hit against that fresh state.
        //   let prompt_token_count_p = r2.prompt_token_count;
        //   let mut p3 = p.clone();
        //   p3.push(ChatMessage::assistant(&r2.text));
        //   p3.push(ChatMessage::user("Follow-up"));
        //   let r3 = model.chat_session_start_sync(p3, cfg())?;
        //   assert!(r3.cached_tokens >= prompt_token_count_p);
    }
}

#[cfg(test)]
mod prefix_cache_decision_tests {
    //! Pure-logic coverage of the prefix-cache decision tree — no model
    //! load required. The verifier `Gemma4Inner::verify_cache_prefix`
    //! returns either `0` (miss) or `cached_token_history.len()` (exact
    //! prefix relation). The engine session core
    //! (`engine::session::chat_turn_core`) then classifies that
    //! value plus the incoming prompt length into
    //! [`PrefixCacheDecision::StrictExtendHit`] (warm-reuse, skip the
    //! cached prefix, prefill only the tail) vs
    //! [`PrefixCacheDecision::Miss`] (reset caches + re-init + full
    //! prefill).
    //!
    //! The four cases covered below pin the invariant: exact-match MUST
    //! route to `Miss`, not to `StrictExtendHit`. Treating exact-match as a
    //! shortcut would corrupt the next warm-hit turn by advancing cache
    //! state to `prompt + last_token` while the history write-back only
    //! persists `tokens + generated`. The `#[ignore]`-gated integration
    //! tests above exercise the end-to-end behaviour against a loaded
    //! Gemma4 model; this module guarantees the decision logic stays
    //! correct in every CI run without a model dependency.

    use super::{PrefixCacheDecision, classify_prefix_cache_decision};

    #[test]
    fn empty_cache_is_miss() {
        // verify_cache_prefix returned 0 (cached_token_history empty,
        // reuse_cache disabled, has_images guard, or prefix mismatch).
        // Regardless of tokens.len(), the classifier routes to Miss so
        // the caller runs reset_caches_sync + init_caches_sync + full
        // prefill.
        assert_eq!(
            classify_prefix_cache_decision(0, 0),
            PrefixCacheDecision::Miss,
            "empty cache + empty tokens must be Miss"
        );
        assert_eq!(
            classify_prefix_cache_decision(0, 10),
            PrefixCacheDecision::Miss,
            "empty cache + non-empty tokens must be Miss"
        );
    }

    #[test]
    fn strict_extend_is_hit() {
        // verify_cache_prefix returned cached_token_history.len() AND
        // tokens.len() > cached_token_history.len() — the new prompt
        // strictly extends the cached one. This is the only case that
        // takes the warm-reuse path: prefill_offset = cached_prefix_len,
        // so only the tail delta is prefilled.
        assert_eq!(
            classify_prefix_cache_decision(5, 8),
            PrefixCacheDecision::StrictExtendHit,
            "cached.len() < tokens.len() must be StrictExtendHit"
        );
        assert_eq!(
            classify_prefix_cache_decision(1, 2),
            PrefixCacheDecision::StrictExtendHit,
            "cached.len() = 1, tokens.len() = 2 must be StrictExtendHit (smallest hit)"
        );
    }

    #[test]
    fn divergence_is_miss() {
        // verify_cache_prefix returned 0 because tokens[..cached.len()]
        // != cached[..] — semantically a divergence even though we only
        // observe the 0 return here. Same code path as `empty_cache_is_miss`
        // — both flavours of Miss fall into the same branch.
        assert_eq!(
            classify_prefix_cache_decision(0, 20),
            PrefixCacheDecision::Miss,
            "divergence (verifier returned 0) must be Miss"
        );
    }

    #[test]
    fn exact_match_is_miss() {
        // verify_cache_prefix returned cached_token_history.len() AND
        // tokens.len() == cached_token_history.len() — byte-equal
        // prompt. The classifier routes to Miss because Gemma4 has no
        // snapshot of final-step logits and no safe "rewind by 1"
        // primitive over the sliding-window cache. Reprefilling the
        // last cached token over the live caches would advance cache
        // state to `prompt + last_token` (duplicated) while the
        // history write-back persists `tokens + generated`, desyncing
        // cache and history for the next warm-hit turn.
        //
        // This invariant guards against silently corrupting multi-turn
        // correctness.
        assert_eq!(
            classify_prefix_cache_decision(5, 5),
            PrefixCacheDecision::Miss,
            "exact-match (cached.len() == tokens.len()) must be Miss, not StrictExtendHit"
        );
        assert_eq!(
            classify_prefix_cache_decision(1, 1),
            PrefixCacheDecision::Miss,
            "exact-match single token must be Miss"
        );
        assert_eq!(
            classify_prefix_cache_decision(1000, 1000),
            PrefixCacheDecision::Miss,
            "exact-match long prompts must still be Miss"
        );
    }

    #[test]
    fn invariant_cached_len_never_exceeds_tokens_len_in_hit() {
        // Belt-and-braces: the verifier itself returns 0 when
        // tokens.len() < cached.len() (no partial-cache reuse), so
        // `cached_prefix_len > tokens_len` should never be observed by
        // the classifier in practice. But if it ever was, the branch
        // routes it to Miss (cached_prefix_len < tokens_len is false),
        // which is the safe fallthrough.
        assert_eq!(
            classify_prefix_cache_decision(10, 5),
            PrefixCacheDecision::Miss,
            "cached_prefix_len > tokens_len must be Miss (defensive fallthrough)"
        );
    }
}

#[cfg(test)]
mod tool_delta_marker_tests {
    //! Guard the structured `is_error` channel on Gemma4's tool-result
    //! wire format. The shared
    //! [`crate::tokenizer::TOOL_ERROR_MARKER`] must be injected inside
    //! the `<|turn>tool` block only when the caller passes
    //! `Some(true)`. `None` and `Some(false)` keep the output
    //! byte-equal to the pre-feature behavior — guarding both the hot
    //! (successful) path and the explicit-false path against
    //! accidental drift. The marker text contains no Gemma4 delimiter
    //! tokens so the downstream `escape_gemma4_content` step is a
    //! no-op on it.

    use super::build_gemma4_tool_delta_text;
    use crate::tokenizer::TOOL_ERROR_MARKER;

    #[test]
    fn injects_marker_when_is_error_true() {
        let payload = "boom: connection refused";
        let rendered = build_gemma4_tool_delta_text("call_fail", payload, None, Some(true));
        let expected_inner = format!("{TOOL_ERROR_MARKER}{payload}");
        assert!(
            rendered.contains(&expected_inner),
            "expected error marker inside <|turn>tool block; got:\n{rendered}",
        );
        // Wrapper integrity stays correct on the marked path.
        assert!(
            rendered.contains("<|turn>tool\n"),
            "tool block opener missing"
        );
        assert!(rendered.contains("<turn|>"), "turn closer missing");
        assert!(rendered.contains("<|turn>model\n"), "model opener missing");
    }

    #[test]
    fn skips_marker_when_is_error_none() {
        let payload = "{\"temperature\": 72}";
        let rendered = build_gemma4_tool_delta_text("call_ok", payload, None, None);
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
    fn skips_marker_when_is_error_some_false() {
        let payload = "ok";
        let rendered = build_gemma4_tool_delta_text("call_ok", payload, None, Some(false));
        assert!(
            !rendered.contains(TOOL_ERROR_MARKER),
            "marker leaked into Some(false) delta:\n{rendered}",
        );
    }

    #[test]
    fn does_not_remark_content_that_resembles_marker() {
        // The structured channel removes the collision concern: a
        // successful tool result whose literal content begins with the
        // marker text must NOT double-prefix the marker on its way
        // through the renderer.
        let suspicious = format!("{TOOL_ERROR_MARKER}this is a successful payload");
        let rendered = build_gemma4_tool_delta_text("call_ok", &suspicious, None, None);
        let occurrences = rendered.matches(TOOL_ERROR_MARKER).count();
        assert_eq!(
            occurrences, 1,
            "marker count should be 1 (the original literal); got {occurrences} in:\n{rendered}",
        );
    }
}
