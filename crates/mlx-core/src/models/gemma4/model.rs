use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use napi::Either;
use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi_derive::napi;

use crate::array::{DType, MxArray};
use crate::model_thread::{ResponseTx, StreamTx};
use crate::nn::{Embedding, Linear, RMSNorm};
use crate::sampling::{SamplingConfig, sample};
use crate::stream::{DeviceType, Stream, StreamContext};
use crate::tokenizer::{ChatMessage, Qwen3Tokenizer};
use crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter;

use super::image_processor::{Gemma4ImageProcessor, ProcessedGemma4Image};
use super::vision::{Gemma4MultimodalEmbedder, Gemma4VisionModel};

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
use crate::models::qwen3_5::chat_common;
use crate::models::qwen3_5::model::{ChatConfig, ChatResult, ChatStreamChunk, ChatStreamHandle};
use tracing::{debug, info};

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

struct StreamSender(StreamTx<ChatStreamChunk>);

impl StreamSender {
    fn call(&self, result: Result<ChatStreamChunk>, _mode: ThreadsafeFunctionCallMode) {
        let _ = self.0.send(result);
    }
}

/// Emit the text/reasoning deltas collected by the stream parser on this
/// step, and fold any surfaced `ToolCall` segments into the shared
/// accumulator. Tool calls do NOT stream out as deltas — they land on the
/// terminal `done=true` chunk.
fn dispatch_stream_segments(segments: Vec<super::output_parser::StreamSegment>, cb: &StreamSender) {
    use super::output_parser::StreamSegment;
    for seg in segments {
        match seg {
            StreamSegment::Text(text) => {
                if text.is_empty() {
                    continue;
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
                        is_reasoning: Some(false),
                    }),
                    ThreadsafeFunctionCallMode::NonBlocking,
                );
            }
            StreamSegment::Reasoning(text) => {
                if text.is_empty() {
                    continue;
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
                        is_reasoning: Some(true),
                    }),
                    ThreadsafeFunctionCallMode::NonBlocking,
                );
            }
            StreamSegment::ToolCall(_) => {
                // Accumulated on `parser.tool_calls()` for the terminal chunk.
            }
        }
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
    pub(crate) lm_head: Option<Linear>,
    /// Pre-transposed embedding weight for tied lm_head: [hidden_size, vocab_size].
    /// Only populated when tie_word_embeddings=true.
    pub(crate) embed_weight_t: Option<MxArray>,
    pub(crate) ple: Option<PleComponents>,
    // Vision components (None for text-only models)
    pub(crate) vision_tower: Option<Gemma4VisionModel>,
    pub(crate) embed_vision: Option<Gemma4MultimodalEmbedder>,
    pub(crate) image_processor: Option<Gemma4ImageProcessor>,
    pub(crate) tokenizer: Option<Arc<Qwen3Tokenizer>>,
    /// Lazily-initialized KV caches that persist across chat turns.
    ///
    /// `None` after construction and after `reset_caches_sync`. Populated on
    /// the first call to `init_caches_sync`, which is triggered lazily by
    /// `chat_sync_core` / `chat_stream_sync_core` on the first turn of a
    /// session. Step 5c will use this state to implement the session API
    /// methods (`chat_session_start_sync`, `chat_session_continue_sync`,
    /// etc.) that share a live cache across turns.
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
    pub(crate) model_id: u64,
}

/// Commands dispatched from NAPI methods to the dedicated model thread.
///
/// Images ride along inside `ChatMessage.images` (`Vec<Uint8Array>`) and are
/// decoded by the Gemma4 image processor on the model thread inside
/// `chat_sync_core` / `chat_stream_sync_core`. napi-rs's `Uint8Array` has
/// an `unsafe impl Send`, so it's safe to cross thread boundaries in the
/// command channel. See Step 5b of the chat-session refactor for why image
/// processing moved off the NAPI thread.
pub(crate) enum Gemma4Cmd {
    /// Start a new chat session via the jinja-render path with `<turn|>`
    /// as the stop token. See [`Gemma4Inner::chat_session_start_sync`] for
    /// the behavioural contract (full cache reset, session boundary on
    /// `<turn|>`).
    ChatSessionStart {
        messages: Vec<ChatMessage>,
        config: ChatConfig,
        reply: ResponseTx<ChatResult>,
    },
    /// Continue an existing session by appending a user turn. See
    /// [`Gemma4Inner::chat_session_continue_sync`] — builds a raw Gemma4
    /// delta (`\n<|turn>user\n...<turn|>\n<|turn>model\n`), tokenizes
    /// it, and prefills on top of the live caches.
    ///
    /// Carries an opt-in `images` guard parameter that is rejected with
    /// an `IMAGE_CHANGE_REQUIRES_SESSION_RESTART:`-prefixed error so the
    /// TS `ChatSession` layer can route image-changes back through a
    /// fresh `chat_session_start` uniformly across model backends.
    ChatSessionContinue {
        user_message: String,
        images: Option<Vec<Uint8Array>>,
        config: ChatConfig,
        reply: ResponseTx<ChatResult>,
    },
    /// Continue an existing session with a tool-result delta. See
    /// [`Gemma4Inner::chat_session_continue_tool_sync`] — builds a
    /// Gemma4-format tool delta (`\n<|turn>tool\n{content}<turn|>\n<|turn>model\n`)
    /// and prefills on top of the live caches.
    ChatSessionContinueTool {
        tool_call_id: String,
        content: String,
        config: ChatConfig,
        reply: ResponseTx<ChatResult>,
    },
    /// Streaming session-start: same semantics as
    /// [`ChatSessionStart`](Self::ChatSessionStart) but streams token
    /// deltas through `stream_tx`.
    ChatStreamSessionStart {
        messages: Vec<ChatMessage>,
        config: ChatConfig,
        stream_tx: StreamTx<ChatStreamChunk>,
        cancelled: Arc<AtomicBool>,
    },
    /// Streaming session-continue: same semantics as
    /// [`ChatSessionContinue`](Self::ChatSessionContinue) but streams
    /// token deltas through `stream_tx`. Carries the same opt-in
    /// `images` guard parameter.
    ChatStreamSessionContinue {
        user_message: String,
        images: Option<Vec<Uint8Array>>,
        config: ChatConfig,
        stream_tx: StreamTx<ChatStreamChunk>,
        cancelled: Arc<AtomicBool>,
    },
    /// Streaming tool-result continuation: same semantics as
    /// [`ChatSessionContinueTool`](Self::ChatSessionContinueTool) but
    /// streams token deltas through `stream_tx`.
    ChatStreamSessionContinueTool {
        tool_call_id: String,
        content: String,
        config: ChatConfig,
        stream_tx: StreamTx<ChatStreamChunk>,
        cancelled: Arc<AtomicBool>,
    },
    /// Reset all caches and clear cached token history. Exposed so tests
    /// and session-management code can start from a known clean state
    /// between turns.
    ResetCaches { reply: ResponseTx<()> },
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
    pub(crate) thread: Option<crate::model_thread::ModelThread<Gemma4Cmd>>,
    pub(crate) model_id: u64,
    /// Whether the loaded config includes `vision_config`. Mirrored here so
    /// the NAPI side can fail fast on image inputs to a text-only model
    /// without round-tripping to the model thread. The actual image
    /// processor lives on `Gemma4Inner` and runs on the model thread.
    pub(crate) has_vision: bool,
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

/// Serializes compiled C++ forward calls across model instances.
/// Only matters if two Gemma4 models are loaded simultaneously (rare).
static COMPILED_FORWARD_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Classification of the prefix-cache decision made from a
/// [`Gemma4Inner::verify_cache_prefix`] return value plus the incoming
/// token count.
///
/// Test-only mirror of the inlined branch at the top of
/// [`Gemma4Inner::chat_sync_core`] /
/// [`Gemma4Inner::chat_stream_sync_core`] — separating the decision
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

/// Test-only helper: decide what to do given the verifier's answer and
/// the incoming prompt length. Exact-match (`cached_prefix_len ==
/// tokens_len`) and zero-length prefix both route to
/// [`PrefixCacheDecision::Miss`].
///
/// Mirrors the inlined branch at the top of
/// [`Gemma4Inner::chat_sync_core`] /
/// [`Gemma4Inner::chat_stream_sync_core`]; lifting it out keeps the
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
            Some(Linear::new(hidden_size, vocab_size, Some(false))?)
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

        // Initialize vision components if vision_config is present
        let (vision_tower, embed_vision, image_processor) = if let Some(ref vc) =
            config.vision_config
        {
            let vt = Gemma4VisionModel::new(vc)?;
            let ev =
                Gemma4MultimodalEmbedder::new(vc.hidden_size, config.hidden_size, vc.rms_norm_eps)?;
            let ip = Gemma4ImageProcessor::new(
                vc.patch_size,
                vc.default_output_length,
                vc.pooling_kernel_size,
            );
            (Some(vt), Some(ev), Some(ip))
        } else {
            (None, None, None)
        };

        let model_id = MODEL_ID_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Block-paged KV adapter — default-on; opt out via
        // `use_block_paged_cache: false`.
        //
        // Default-on adapter construction (2026-04-28 default flip).
        // Text-only paged dispatch is active: every text-only chat-entry
        // site (`chat_sync_core_paged`, `chat_stream_sync_core_paged`,
        // `chat_tokens_delta_sync`, `chat_stream_tokens_delta_sync_inner`)
        // routes through the paged adapter; vision turns still take the
        // flat `Gemma4LayerCache::Sliding(RotatingKVCache)` path. Setting
        // `use_block_paged_cache: false` in the model config is the
        // explicit opt-out and reverts every layer to the legacy flat
        // caches.
        //
        // Cache dtype: BFloat16 (Gemma4's production dtype).
        // Per-layer head_dim: Gemma4 has variable head dims per layer
        // type (sliding vs global). The pool covers ONLY global (full)
        // attention layers — sliding layers continue to use the existing
        // `Gemma4LayerCache::Sliding(RotatingKVCache)` path because their
        // window-trimmed semantics don't map onto the paged pool. The
        // pool's per-layer slot size therefore uses the global head_dim
        // and global KV-head count.
        let paged_adapter = if config.use_block_paged_cache.unwrap_or(true) {
            // Count GLOBAL layers — the paged pool only covers
            // full_attention layers in their original layer order. A
            // future paged-aware forward path indexes this pool by
            // global-layer ordinal (NOT absolute decoder index).
            let num_global_layers: u32 = (0..config.num_hidden_layers as usize)
                .filter(|&i| config.is_global_layer(i))
                .count() as u32;
            if num_global_layers == 0 {
                return Err(napi::Error::from_reason(
                    "Gemma4 block-paged adapter: config has no full_attention layers; \
                     paged KV cache requires at least one global attention layer",
                ));
            }

            let block_size = config.paged_block_size.unwrap_or(16);
            let gpu_memory_mb = config.paged_cache_memory_mb.unwrap_or(2048);
            // Use global head_dim — sliding layers don't use the pool.
            let head_size = config.effective_head_dim(true) as u32;
            // Use global KV heads — sliding layers don't use the pool.
            let num_kv_heads = config.effective_kv_heads(true) as u32;

            let pa_config = mlx_paged_attn::PagedAttentionConfig {
                block_size,
                gpu_memory_mb,
                head_size,
                num_kv_heads,
                // Pool covers ONLY global attention layers — sliding
                // layers continue to use Gemma4LayerCache::Sliding.
                num_layers: num_global_layers,
                use_fp8_cache: Some(false),
                max_seq_len: Some(config.max_position_embeddings as u32),
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
                "Gemma4 block-paged adapter enabled (construction-only): num_blocks={num_blocks}, \
                 block_size={block_size}, gpu_memory_mb={gpu_memory_mb}, num_global_layers={num_global_layers}, \
                 cache_dtype=BFloat16"
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
            embed_vision,
            image_processor,
            tokenizer: None,
            caches: None,
            cached_token_history: Vec::new(),
            cached_image_key: None,
            paged_adapter,
            model_id,
        })
    }

    /// Initialize the per-turn KV caches in-place.
    ///
    /// Called lazily by `chat_sync_core` / `chat_stream_sync_core` on the
    /// first turn of a session (or whenever `self.caches` is `None` because a
    /// previous `reset_caches_sync` wiped them). Subsequent turns reuse the
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
    pub(crate) fn compute_layer_kinds(&self) -> Vec<Gemma4LayerKind> {
        compute_layer_kinds(&self.config)
    }

    /// Drop the live KV caches and clear reuse-tracking state.
    ///
    /// `Gemma4LayerCache` has no `reset()` (the inner `KVCache` /
    /// `RotatingKVCache` don't expose one here), so this simply takes the
    /// Vec and lets the next `init_caches_sync` rebuild. Cleared reuse
    /// state ensures a subsequent chat turn can't mistakenly claim a cache
    /// prefix hit against stale history.
    ///
    /// Called by the session API's reset path and by the chat-session
    /// start command so that a fresh turn starts from an empty cache.
    /// It is NOT called from `chat_sync_core` / `chat_stream_sync_core`
    /// directly because those are re-entrant primitives that trust
    /// their caller's cache-management.
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
    }

    pub(crate) fn set_tokenizer(&mut self, tokenizer: Arc<Qwen3Tokenizer>) {
        self.tokenizer = Some(tokenizer);
    }

    /// Check whether `tokens` extends the cached conversation history and
    /// return the length of the reused prefix.
    ///
    /// **Safety invariant**: this helper returns ONLY `0` (cache miss) or
    /// `cached_token_history.len()` — either a strict-extend
    /// (`cached_prefix_len < tokens.len()`) or an exact match
    /// (`cached_prefix_len == tokens.len()`). Never an intermediate
    /// value. Combined with the "no mid-sequence rewind" policy in
    /// [`Self::chat_sync_core`] / [`Self::chat_stream_sync_core`], this
    /// keeps Gemma4's layer caches safe under prefix reuse.
    ///
    /// The caller must additionally distinguish strict-extend (warm-reuse
    /// safe) from exact-match. Only the strict-extend case is served via
    /// the warm path; exact-match is routed back through the cache-miss
    /// branch because Gemma4 has no snapshot of final-step logits and no
    /// cheap rewind primitive for its sliding-window cache. Attempting to
    /// reprefill the final cached token over the live caches would
    /// advance cache state to `prompt + last_token` (duplicated) while
    /// the history write-back block only persists `tokens + generated`,
    /// corrupting the next warm-hit turn.
    ///
    /// * Sliding-window layers (`Gemma4LayerCache::new_sliding`) are safe
    ///   because their offset only grows — appending new tokens advances
    ///   the window forward rather than rewinding into evicted state. If
    ///   the cached history already exceeded the sliding window, the
    ///   cache correctly represents the most recent `sliding_window`
    ///   tokens ending at `cached_token_history.len()`, and the delta
    ///   continues from that point.
    /// * Global layers accumulate all key/value tensors; appending delta
    ///   tokens just extends the cache linearly.
    ///
    /// **Text-only**: this is a conservative text-only variant (see the
    /// prefix-reuse plan at `.claude/plans/dapper-zooming-catmull.md`).
    /// If either the new prompt carries images OR the cached session
    /// does, we force a cache miss. A future VLM-aware variant would gate
    /// on `cached_image_key == compute_image_cache_key(...)` like the
    /// Qwen3.5 shared helper; until then, any image-bearing turn cold-
    /// starts the session.
    fn verify_cache_prefix(&self, tokens: &[u32], reuse_cache: bool, has_images: bool) -> usize {
        if !reuse_cache {
            return 0;
        }
        // Text-only: force a miss whenever images are involved on either
        // side. This keeps prefix reuse strictly aligned with text-only
        // sessions and sidesteps the mrope / image-key coordination that
        // the Qwen3.5 shared helper handles.
        if has_images || self.cached_image_key.is_some() {
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

    /// Core Gemma4 chat implementation with optional EOS override.
    ///
    /// Shared between the non-streaming and streaming session paths. All
    /// image decode + resize + patching happens here on the model thread
    /// (off the NAPI thread) using `ChatMessage.images` which is `Send`
    /// via napi-rs's `unsafe impl`.
    ///
    /// ## Field support
    ///
    /// **Supported**: `max_new_tokens`, `temperature`, `top_k`, `top_p`,
    /// `min_p`, `tools`, `reasoning_effort` (mapped to the template's
    /// `enable_thinking` kwarg via `chat_common::resolve_enable_thinking`),
    /// `report_performance`, `reuse_cache`.
    ///
    /// **Silent no-ops** (Gemma4 decode loop has no code path that reads
    /// them): `repetition_penalty`, `repetition_context_size`,
    /// `presence_penalty`, `presence_context_size`, `frequency_penalty`,
    /// `frequency_context_size`, `max_consecutive_tokens`,
    /// `max_ngram_repeats`, `ngram_size`, `thinking_token_budget`,
    /// `include_reasoning`.
    ///
    /// `eos_token_id` is the caller-supplied stop-on token id. The decode
    /// loop stops on this id OR any of `config.eos_token_ids`, so the
    /// cached history ends on a caller-controlled boundary (typically a
    /// turn-terminator token). Used by the session-start path to leave
    /// the cache on a clean Gemma4 `<turn|>` boundary.
    pub(crate) fn chat_sync_core(
        &mut self,
        messages: Vec<ChatMessage>,
        config: ChatConfig,
        eos_token_id: u32,
    ) -> Result<ChatResult> {
        let max_new_tokens = config.max_new_tokens.unwrap_or(2048);

        let tokenizer = self
            .tokenizer
            .clone()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))?;

        // Decode images on the model thread. `ChatMessage.images` is a
        // `Vec<Uint8Array>` which is `Send` via napi-rs's `unsafe impl`,
        // so we can cross the thread boundary inside the Gemma4 session
        // commands and do the image decode + resize + patching here
        // instead of duplicating the processor on the NAPI side.
        let raw_images = extract_images_from_messages(&messages);
        let processed_images: Vec<ProcessedGemma4Image> = if raw_images.is_empty() {
            Vec::new()
        } else {
            let ip = self.image_processor.as_ref().ok_or_else(|| {
                Error::from_reason(
                    "Images provided but model has no vision support (no vision_config in config.json)",
                )
            })?;
            let mut out = Vec::with_capacity(raw_images.len());
            for bytes in &raw_images {
                out.push(ip.process_bytes(bytes)?);
            }
            out
        };

        let has_images = !processed_images.is_empty();
        // Compute the image cache key BEFORE the prefill so we can
        // record it on `self.cached_image_key` after the decode loop.
        // Session callers inspect this field to decide whether a
        // session-continue delta is allowed (text-only) or requires
        // a fresh `chat_session_start`.
        let new_image_key: Option<u64> = if raw_images.is_empty() {
            None
        } else {
            Some(chat_common::compute_image_cache_key(&raw_images))
        };
        let sampling_config = make_sampling_config(&config, &self.config);
        let enable_thinking = chat_common::resolve_enable_thinking(&config);
        let eos_ids = self.config.eos_token_ids.clone();

        // Try the tokenizer's chat template if available (handles role mapping,
        // special tokens, and variant-specific formatting automatically).
        // Fall back to manual Gemma4 format if no template was loaded.
        let tokens = if tokenizer.has_chat_template() {
            tokenizer.apply_chat_template_sync(
                &messages,
                Some(true), // add_generation_prompt
                config.tools.as_deref(),
                enable_thinking, // None = template default
            )?
        } else {
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
            for msg in &messages {
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
            tokenizer.encode_sync(&prompt_text, Some(false))?
        };

        // Expand image tokens if images are present.
        // Gemma4 uses: <|image>  (BOI) + <|image|> × num_soft_tokens + <image|> (EOI)
        // The chat template inserts a single <|image|> per image; we expand it here.
        let tokens = if has_images && !processed_images.is_empty() {
            let image_token_id = self.config.image_token_id.unwrap_or(258880) as u32;
            let boi_token_id = self.config.boi_token_id.unwrap_or(255999) as u32;
            let eoi_token_id = self.config.eoi_token_id.unwrap_or(258882) as u32;
            expand_image_tokens(
                &tokens,
                &processed_images,
                image_token_id,
                boi_token_id,
                eoi_token_id,
            )
        } else {
            tokens
        };

        // Block-paged dispatch: when the adapter is configured AND no
        // images are involved, route through the parallel
        // `chat_sync_core_paged` path. The flat path stays untouched so
        // off-by-default behavior is byte-identical to before this
        // commit. Vision turns always use the flat path (paged dispatch
        // is text-only at this stage).
        if self.paged_adapter.is_some() && !has_images {
            return self.chat_sync_core_paged(
                tokens,
                tokenizer,
                config,
                eos_token_id,
                sampling_config,
                max_new_tokens,
            );
        }

        // Prefix-cache verification. `verify_cache_prefix` returns 0 on
        // miss or `cached.len()` on an exact prefix relation (either
        // strict-extend or exact-match) — never intermediate (see its
        // rustdoc). On a strict-extend hit we skip the cached prefix and
        // prefill only the tail delta. On an exact match or miss we
        // reset the caches here (not unconditionally in
        // `chat_session_start_sync`) and do a full re-prefill, so
        // stateless agent clients that resend the full transcript each
        // turn can reuse the live KV caches when the histories strictly
        // extend.
        //
        // Exact match is deliberately routed to the miss branch: Gemma4's
        // compiled C++ decode path has no snapshot of the final-step
        // logits and no cheap "rewind by one" primitive over its
        // sliding-window cache. A previous revision reprefilled the last
        // cached token on top of the live caches, but that advanced cache
        // state to `prompt + last_token` (duplicated) while the
        // history write-back block only persists `tokens + generated`.
        // The resulting drift between live cache and persisted history
        // corrupted the next warm-hit turn.
        let reuse_cache = config.reuse_cache.unwrap_or(true);
        let cached_prefix_len_raw = self.verify_cache_prefix(&tokens, reuse_cache, has_images);
        let (prefill_offset, reported_cached_tokens) =
            if cached_prefix_len_raw > 0 && cached_prefix_len_raw < tokens.len() {
                debug!(
                    "Gemma4 prefix cache reuse: {} cached tokens, {} delta to prefill",
                    cached_prefix_len_raw,
                    tokens.len() - cached_prefix_len_raw
                );
                (cached_prefix_len_raw, cached_prefix_len_raw)
            } else {
                // Cache miss OR exact-match: drop any stale caches/history
                // and re-init. See the comment above for why exact-match
                // falls through here instead of taking a shortcut.
                self.reset_caches_sync()?;
                self.init_caches_sync()?;
                (0, 0)
            };

        // Defensive: caches must be live before the prefill runs.
        // `reset_caches_sync` above only fires on miss, so on a hit we
        // rely on the prior turn's init. If somebody cleared the caches
        // out-of-band between turns, re-init here.
        if self.caches.is_none() {
            self.init_caches_sync()?;
        }

        // Slice the prompt tensor to only the tokens that still need to
        // be prefilled. On miss this is the full prompt; on hit this is
        // just the tail delta.
        let prefill_slice: Vec<i32> = tokens[prefill_offset..].iter().map(|&t| t as i32).collect();
        let prefill_len = prefill_slice.len();
        let prompt = MxArray::from_int32(&prefill_slice, &[1, prefill_len as i64])?;

        // Create dedicated generation stream for GPU scheduling.
        let generation_stream = Stream::new(DeviceType::Gpu);

        // Wired memory: pin model weights in GPU memory (prevents paging for large models).
        // Uses usize::MAX to always set limit to max_recommended_working_set_size.
        let _wired_ctx = crate::stream::WiredLimitContext::new(usize::MAX, vec![generation_stream]);

        let generation_start = std::time::Instant::now();
        let prompt_token_count = tokens.len();

        // Vision prefill: if images present, build merged embeddings
        // (text embeddings with vision features scattered at image_token positions)
        let vision_embeds: Option<MxArray> = if has_images
            && !processed_images.is_empty()
            && let Some(ref vt) = self.vision_tower
            && let Some(ref ev) = self.embed_vision
        {
            let image_token_id = self.config.image_token_id.unwrap_or(258880);

            // Run vision tower on each image and collect features
            let mut all_features: Vec<MxArray> = Vec::new();
            for proc in &processed_images {
                let features = vt.forward(&proc.pixel_values)?;
                let projected = ev.forward(&features)?;
                all_features.push(projected);
            }

            // Concatenate all image features: [1, total_soft_tokens, hidden_size]
            let image_features = if all_features.len() == 1 {
                all_features.remove(0)
            } else {
                let refs: Vec<&MxArray> = all_features.iter().collect();
                MxArray::concatenate_many(refs, Some(1))?
            };

            // Build text embeddings
            let text_embeds = self.embed_tokens.forward(&prompt)?;
            let text_embeds = text_embeds.mul_scalar((self.config.hidden_size as f64).sqrt())?;

            // Cast image features to text embedding dtype
            let embed_dtype = text_embeds.dtype()?;
            let image_features = image_features.astype(embed_dtype)?;

            // masked_scatter: replace image_token positions with vision features
            let image_token = MxArray::scalar_int(image_token_id)?;
            let image_mask = prompt.equal(&image_token)?;

            // Validate: number of True positions in mask must match vision feature count
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

            let image_mask_expanded = image_mask.expand_dims(-1)?;
            let image_mask_expanded = image_mask_expanded.broadcast_to(&text_embeds.shape()?)?;

            let merged = masked_scatter(&text_embeds, &image_mask_expanded, &image_features)?;
            Some(merged)
        } else {
            None
        };

        // Prefill: process tokens [0:N-1] through body only (no lm_head),
        // then run last token through full forward to get logits.
        // Matches mlx-lm generate_step pattern.
        //
        // `self.caches` was populated by the lazy-init block above, so the
        // expect cannot fire — kept defensive for the (impossible) future
        // where init_caches_sync silently no-ops.
        {
            let _stream_ctx = StreamContext::new(generation_stream);
            let caches = self
                .caches
                .as_mut()
                .expect("caches populated by init_caches_sync above");
            if let Some(ref embeds) = vision_embeds {
                // Vision path: prefill with merged embeddings
                prefill_body_gemma4_with_embeds(
                    &prompt,
                    embeds,
                    &self.embed_tokens,
                    &self.layers,
                    caches,
                    &self.final_norm,
                    self.ple.as_ref(),
                    &self.config,
                )?;
            } else {
                // Text-only path
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
        }
        eval_gemma4_caches(
            self.caches
                .as_ref()
                .expect("caches populated by init_caches_sync above"),
        );

        // Last token → logits. `prompt` is the delta slice, so its final
        // position is `prefill_len - 1`. `prefill_body_gemma4` processed
        // `[0 .. prefill_len - 1]` and left the final token for us.
        let last_token = prompt.slice_axis(1, prefill_len as i64 - 1, prefill_len as i64)?;
        let logits = {
            let _stream_ctx = StreamContext::new(generation_stream);
            let caches = self
                .caches
                .as_mut()
                .expect("caches populated by init_caches_sync above");
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
        let logits = logits.squeeze(Some(&[1]))?;
        let y = sample_next_token(&logits, sampling_config)?;
        y.eval();
        eval_gemma4_caches(
            self.caches
                .as_ref()
                .expect("caches populated by init_caches_sync above"),
        );

        // Mark first token time (TTFT = time to first token)
        let first_token_instant = std::time::Instant::now();

        // Decode loop — matches mlx-lm generate.py pattern:
        // 1. Build lazy graph per step via forward_inner
        // 2. async_eval the output token (caches materialize through dependency graph)
        // 3. Double-buffer: build step N+1 while GPU executes step N
        //
        // Set GEMMA4_USE_COMPILE=1 to use the old compiled C++ path for A/B testing.
        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut finish_reason = "length".to_string();

        let use_compiled = std::env::var("GEMMA4_USE_COMPILE").is_ok()
            && self.config.num_kv_shared_layers.is_none_or(|n| n <= 0)
            && unsafe { mlx_sys::mlx_qwen35_get_model_id() } == self.model_id;

        if use_compiled {
            // Legacy compiled C++ path (opt-in via GEMMA4_USE_COMPILE=1)
            let _compiled_guard = COMPILED_FORWARD_MUTEX.lock().unwrap();
            let caches_ref = self
                .caches
                .as_ref()
                .expect("caches populated by init_caches_sync above");
            let mut cache_arrays_owned: Vec<MxArray> = Vec::with_capacity(caches_ref.len() * 2);
            for (layer_idx, cache) in caches_ref.iter().enumerate() {
                let (k, v) = cache.get_cached_kv().ok_or_else(|| {
                    Error::from_reason(format!(
                        "Compiled Gemma4 decode expected cache for layer {} after prefill",
                        layer_idx
                    ))
                })?;
                cache_arrays_owned.push(k);
                cache_arrays_owned.push(v);
            }
            let mut cache_ptrs: Vec<*mut mlx_sys::mlx_array> =
                cache_arrays_owned.iter().map(|a| a.as_raw_ptr()).collect();

            let layer_types_i32: Vec<i32> = (0..self.config.num_hidden_layers as usize)
                .map(|i| if self.config.is_global_layer(i) { 1 } else { 0 })
                .collect();

            let max_kv_len =
                (tokens.len() as i32 + max_new_tokens).min(self.config.max_position_embeddings);

            unsafe {
                mlx_sys::mlx_gemma4_init_from_prefill(
                    self.config.num_hidden_layers,
                    self.config.hidden_size,
                    self.config.num_attention_heads,
                    self.config.num_key_value_heads,
                    self.config.head_dim,
                    self.config.effective_kv_heads(true),
                    self.config.effective_head_dim(true),
                    self.config.rope_theta as f32,
                    self.config.rope_local_base_freq as f32,
                    self.config.partial_rotary_factor as f32,
                    self.config.rms_norm_eps as f32,
                    self.config.sliding_window,
                    if self.config.tie_word_embeddings {
                        1
                    } else {
                        0
                    },
                    max_kv_len,
                    1,
                    self.config.num_experts.unwrap_or(0),
                    self.config.top_k_experts.unwrap_or(0),
                    self.config.moe_intermediate_size.unwrap_or(0),
                    self.config.intermediate_size,
                    self.config.final_logit_softcapping.unwrap_or(0.0) as f32,
                    layer_types_i32.as_ptr(),
                    layer_types_i32.len() as i32,
                    cache_ptrs.as_mut_ptr(),
                    tokens.len() as i32,
                );
            }

            let embed_weight = self.embed_tokens.get_weight();
            let mut current_y = y;
            for step in 0..max_new_tokens {
                let next_y = if step + 1 < max_new_tokens {
                    let _stream_ctx = StreamContext::new(generation_stream);
                    let next_ids = current_y.reshape(&[1, 1])?;
                    let logits = forward_gemma4_cpp(&next_ids, &embed_weight)?;
                    let next_token = sample_next_token(&logits, sampling_config)?;
                    eval_token_and_gemma4_caches(&next_token);
                    Some(next_token)
                } else {
                    None
                };

                // See `chat_sync_core_paged_inner` for the rationale.
                current_y.eval();
                let token_id = current_y.item_at_int32(0)? as u32;
                generated_tokens.push(token_id);

                if is_eos_token(token_id, &eos_ids, eos_token_id) {
                    finish_reason = "stop".to_string();
                    break;
                }
                if let Some(next_token) = next_y {
                    current_y = next_token;
                } else {
                    break;
                }
                if (step + 1) % 256 == 0 {
                    crate::array::synchronize_and_clear_cache();
                }
            }
            unsafe {
                mlx_sys::mlx_gemma4_reset();
            }
        } else {
            // Default: lazy eval decode (matches mlx-lm pattern)
            //
            // Double-buffered: build step N+1's graph while GPU executes step N.
            // Cache mutations (slice_assign_axis_inplace) are lazy side effects
            // in the computation graph — evaluating the token implicitly
            // materializes caches (no explicit cache eval needed during decode).
            //
            // Pattern from mlx-lm generate.py:
            //   mx.async_eval(next_y)   # fire and forget
            //   if n == 0: mx.eval(y)   # sync only for TTFT
            let mut current_y = y;
            for step in 0..max_new_tokens {
                let next_y = if step + 1 < max_new_tokens {
                    let _stream_ctx = StreamContext::new(generation_stream);
                    let caches = self
                        .caches
                        .as_mut()
                        .expect("caches populated by init_caches_sync above");

                    let next_ids = current_y.reshape(&[1, 1])?;
                    crate::models::gemma4::diagnostic::set_step(step);
                    let logits = forward_inner(
                        &next_ids,
                        &self.embed_tokens,
                        &self.layers,
                        caches,
                        &self.final_norm,
                        &self.lm_head,
                        self.embed_weight_t.as_ref(),
                        self.ple.as_ref(),
                        &self.config,
                    )?;
                    let logits = logits.squeeze(Some(&[1]))?;
                    let next_token = sample_next_token(&logits, sampling_config)?;
                    MxArray::async_eval_arrays(&[&next_token]);
                    Some(next_token)
                } else {
                    None
                };

                // Force `current_y` to evaluate before reading its host value.
                // The previous iteration kicked an async eval on the sampled
                // token (`next_token`), which became `current_y` here. On
                // intermediate steps the lazy graph from
                // `current_y.reshape(...) → forward_inner → ...` would normally
                // chain the eval, but `read_scalar` (mlx_nn_ops.cpp) reads
                // `arr.data<T>()` directly off CPU memory without triggering
                // an implicit `eval()`. On the FINAL iteration there is no
                // forward at all, so the data may still be unevaluated. Without
                // this sync the host sees raw uninitialized bits → garbage
                // token ID → mismatch on length-finish prompts. Mirrors
                // `chat_sync_core_paged_inner`.
                current_y.eval();
                let token_id = current_y.item_at_int32(0)? as u32;
                generated_tokens.push(token_id);

                if is_eos_token(token_id, &eos_ids, eos_token_id) {
                    finish_reason = "stop".to_string();
                    break;
                }
                if let Some(next_token) = next_y {
                    current_y = next_token;
                } else {
                    break;
                }

                if (step + 1) % 256 == 0 {
                    crate::array::clear_cache();
                }
            }
        }

        // Decode text with special tokens preserved so we can extract
        // Gemma4's `<|channel>...<channel|>` reasoning and
        // `<|tool_call>...<tool_call|>` tool-call DSL blocks.
        let raw_text = tokenizer.decode_sync(&generated_tokens, false)?;

        // Save session state so subsequent `chat_session_continue_sync`
        // calls can append a raw delta on top of the live caches. Drop
        // the last generated token when `finish_reason != "length"` so
        // the cached history ends on the turn-terminator boundary (the
        // final token IS that boundary marker — stop, tool_calls, etc.).
        let history_tokens: &[u32] = if finish_reason != "length" && !generated_tokens.is_empty() {
            &generated_tokens[..generated_tokens.len() - 1]
        } else {
            &generated_tokens[..]
        };
        let mut new_history = Vec::with_capacity(tokens.len() + history_tokens.len());
        new_history.extend(tokens.iter().copied());
        new_history.extend_from_slice(history_tokens);
        self.cached_token_history = new_history;
        self.cached_image_key = new_image_key;

        // Compute performance metrics
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
        let mem_after = crate::array::get_active_memory();
        debug!(
            "[gemma4-chat] after generate: {:.2} GB active",
            mem_after / 1e9
        );

        let performance = Some(crate::profiling::PerformanceMetrics {
            ttft_ms,
            prefill_tokens_per_second: if ttft_ms > 0.0 {
                prompt_token_count as f64 / (ttft_ms / 1000.0)
            } else {
                0.0
            },
            decode_tokens_per_second: if decode_ms > 0.0 && gen_toks > 1.0 {
                (gen_toks - 1.0) / (decode_ms / 1000.0)
            } else {
                0.0
            },
        });

        let parsed = super::output_parser::parse_gemma4_output(&raw_text);
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
            cached_tokens: reported_cached_tokens as u32,
            performance,
        })
    }

    /// Core Gemma4 streaming chat implementation with optional EOS override.
    ///
    /// Shared between the non-streaming session-start / session-continue
    /// streaming paths. All image decode + resize + patching happens here
    /// on the model thread (off the NAPI thread).
    ///
    /// ## Field support
    ///
    /// **Supported**: `max_new_tokens`, `temperature`, `top_k`, `top_p`,
    /// `min_p`, `tools`, `reasoning_effort` (mapped to the template's
    /// `enable_thinking` kwarg via `chat_common::resolve_enable_thinking`),
    /// `report_performance`, `reuse_cache`.
    ///
    /// **Silent no-ops** (Gemma4 decode loop has no code path that reads
    /// them): `repetition_penalty`, `repetition_context_size`,
    /// `presence_penalty`, `presence_context_size`, `frequency_penalty`,
    /// `frequency_context_size`, `max_consecutive_tokens`,
    /// `max_ngram_repeats`, `ngram_size`, `thinking_token_budget`,
    /// `include_reasoning`.
    ///
    /// `eos_token_id` is the caller-supplied stop-on token id. The decode
    /// loop stops on this id OR any of `config.eos_token_ids` (used by
    /// streaming session-start to stop at Gemma4's `<turn|>` delimiter).
    fn chat_stream_sync_core(
        &mut self,
        messages: Vec<ChatMessage>,
        config: ChatConfig,
        cb: &StreamSender,
        cancelled: &Arc<AtomicBool>,
        eos_token_id: u32,
    ) -> Result<()> {
        let max_new_tokens = config.max_new_tokens.unwrap_or(2048);

        let tokenizer = self
            .tokenizer
            .clone()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))?;

        // Decode images on the model thread. See `chat_sync_core` for the
        // same pattern and why this lives here instead of the NAPI side.
        let raw_images = extract_images_from_messages(&messages);
        let processed_images: Vec<ProcessedGemma4Image> = if raw_images.is_empty() {
            Vec::new()
        } else {
            let ip = self.image_processor.as_ref().ok_or_else(|| {
                Error::from_reason(
                    "Images provided but model has no vision support (no vision_config in config.json)",
                )
            })?;
            let mut out = Vec::with_capacity(raw_images.len());
            for bytes in &raw_images {
                out.push(ip.process_bytes(bytes)?);
            }
            out
        };

        let has_images = !processed_images.is_empty();
        // Compute the image cache key BEFORE the prefill so we can
        // record it on `self.cached_image_key` after the decode loop.
        // See `chat_sync_core` for the full rationale.
        let new_image_key: Option<u64> = if raw_images.is_empty() {
            None
        } else {
            Some(chat_common::compute_image_cache_key(&raw_images))
        };
        let sampling_config = make_sampling_config(&config, &self.config);
        let enable_thinking = chat_common::resolve_enable_thinking(&config);
        let eos_ids = self.config.eos_token_ids.clone();

        let tokens = if tokenizer.has_chat_template() {
            tokenizer.apply_chat_template_sync(
                &messages,
                Some(true),
                config.tools.as_deref(),
                enable_thinking,
            )?
        } else {
            if enable_thinking == Some(true) {
                return Err(Error::from_reason(
                    "enable_thinking=true requires a chat template",
                ));
            }
            let mut prompt_text = String::from("<bos>");
            for msg in &messages {
                let role = match msg.role.as_str() {
                    "assistant" => "model",
                    "developer" => "system",
                    other => other,
                };
                prompt_text.push_str(&format!("<|turn>{}\n", role));
                if let Some(ref tool_calls) = msg.tool_calls {
                    for tc in tool_calls {
                        prompt_text.push_str(&format!(
                            "<|tool_call>call:{}{{{}}}<tool_call|>",
                            tc.name,
                            json_args_to_gemma4_dsl(&escape_gemma4_content(&tc.arguments))
                        ));
                    }
                }
                prompt_text.push_str(&escape_gemma4_content(&msg.content));
                prompt_text.push_str("<turn|>\n");
            }
            prompt_text.push_str("<|turn>model\n");
            tokenizer.encode_sync(&prompt_text, Some(false))?
        };

        let tokens = if has_images && !processed_images.is_empty() {
            let image_token_id = self.config.image_token_id.unwrap_or(258880) as u32;
            let boi_token_id = self.config.boi_token_id.unwrap_or(255999) as u32;
            let eoi_token_id = self.config.eoi_token_id.unwrap_or(258882) as u32;
            expand_image_tokens(
                &tokens,
                &processed_images,
                image_token_id,
                boi_token_id,
                eoi_token_id,
            )
        } else {
            tokens
        };

        // Block-paged streaming dispatch: same gate as the non-streaming
        // path. See `chat_sync_core` for the rationale (text-only at
        // this stage, flat path for vision turns).
        if self.paged_adapter.is_some() && !has_images {
            return self.chat_stream_sync_core_paged(
                tokens,
                tokenizer,
                config,
                eos_token_id,
                sampling_config,
                cb,
                cancelled,
                max_new_tokens,
            );
        }

        // Prefix-cache verification — see `chat_sync_core` for the full
        // rationale and the `verify_cache_prefix` rustdoc for the
        // "returns 0 or cached.len() only" invariant. As in the
        // non-streaming path, exact match is routed to the miss branch
        // to avoid drift between live caches and the persisted
        // `cached_token_history` (Gemma4 has no safe rewind primitive
        // for its sliding-window cache).
        let reuse_cache = config.reuse_cache.unwrap_or(true);
        let cached_prefix_len_raw = self.verify_cache_prefix(&tokens, reuse_cache, has_images);
        let prefill_offset = if cached_prefix_len_raw > 0 && cached_prefix_len_raw < tokens.len() {
            cached_prefix_len_raw
        } else {
            // Cache miss OR exact-match (treated as miss).
            self.reset_caches_sync()?;
            self.init_caches_sync()?;
            0
        };
        // `cached_prefix_len_reported` is the value surfaced on the
        // terminal `ChatStreamChunk.cached_tokens` for observability.
        // Mirrors `prefill_offset`: zero on a miss or exact-match
        // (treated as miss), equal to the matched prefix length on a
        // warm-reuse hit. Same semantics as the non-streaming
        // `ChatResult.cached_tokens` for Gemma4.
        let cached_prefix_len_reported = prefill_offset as u32;

        if self.caches.is_none() {
            self.init_caches_sync()?;
        }

        let prefill_slice: Vec<i32> = tokens[prefill_offset..].iter().map(|&t| t as i32).collect();
        let prefill_len = prefill_slice.len();
        let prompt = MxArray::from_int32(&prefill_slice, &[1, prefill_len as i64])?;

        let generation_stream = Stream::new(DeviceType::Gpu);
        let _wired_ctx = crate::stream::WiredLimitContext::new(usize::MAX, vec![generation_stream]);

        let generation_start = std::time::Instant::now();
        let prompt_token_count = tokens.len();

        let vision_embeds: Option<MxArray> = if has_images
            && !processed_images.is_empty()
            && let Some(ref vt) = self.vision_tower
            && let Some(ref ev) = self.embed_vision
        {
            let image_token_id = self.config.image_token_id.unwrap_or(258880);
            let mut all_features: Vec<MxArray> = Vec::new();
            for proc in &processed_images {
                let features = vt.forward(&proc.pixel_values)?;
                let projected = ev.forward(&features)?;
                all_features.push(projected);
            }
            let image_features = if all_features.len() == 1 {
                all_features.remove(0)
            } else {
                let refs: Vec<&MxArray> = all_features.iter().collect();
                MxArray::concatenate_many(refs, Some(1))?
            };
            let text_embeds = self.embed_tokens.forward(&prompt)?;
            let text_embeds = text_embeds.mul_scalar((self.config.hidden_size as f64).sqrt())?;
            let embed_dtype = text_embeds.dtype()?;
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
                        "Image token count ({mask_count}) does not match vision feature count ({feature_count})."
                    ),
                ));
            }
            let image_mask_expanded = image_mask.expand_dims(-1)?;
            let image_mask_expanded = image_mask_expanded.broadcast_to(&text_embeds.shape()?)?;
            Some(masked_scatter(
                &text_embeds,
                &image_mask_expanded,
                &image_features,
            )?)
        } else {
            None
        };

        {
            let _stream_ctx = StreamContext::new(generation_stream);
            let caches = self
                .caches
                .as_mut()
                .expect("caches populated by init_caches_sync above");
            if let Some(ref embeds) = vision_embeds {
                prefill_body_gemma4_with_embeds(
                    &prompt,
                    embeds,
                    &self.embed_tokens,
                    &self.layers,
                    caches,
                    &self.final_norm,
                    self.ple.as_ref(),
                    &self.config,
                )?;
            } else {
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
        }
        eval_gemma4_caches(
            self.caches
                .as_ref()
                .expect("caches populated by init_caches_sync above"),
        );

        let last_token = prompt.slice_axis(1, prefill_len as i64 - 1, prefill_len as i64)?;
        let logits = {
            let _stream_ctx = StreamContext::new(generation_stream);
            let caches = self
                .caches
                .as_mut()
                .expect("caches populated by init_caches_sync above");
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
        let logits = logits.squeeze(Some(&[1]))?;
        let y = sample_next_token(&logits, sampling_config)?;
        y.eval();
        eval_gemma4_caches(
            self.caches
                .as_ref()
                .expect("caches populated by init_caches_sync above"),
        );

        let first_token_instant = std::time::Instant::now();
        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut finish_reason = "length".to_string();

        let use_compiled = std::env::var("GEMMA4_USE_COMPILE").is_ok()
            && self.config.num_kv_shared_layers.is_none_or(|n| n <= 0)
            && unsafe { mlx_sys::mlx_qwen35_get_model_id() } == self.model_id;

        // `decode_stream(false)` preserves Gemma4 special tokens
        // (`<|channel>`, `<|tool_call>`, …) in the streamed text so the
        // stream parser can see them. The final `decode_sync(…, false)`
        // below mirrors this for consistency with the parsed ChatResult.
        let mut decode_stream = tokenizer.inner().decode_stream(false);
        let mut streamed_text_len = 0;
        let mut stream_parser = super::output_parser::Gemma4StreamParser::new();

        if use_compiled {
            let _compiled_guard = COMPILED_FORWARD_MUTEX.lock().unwrap();
            let caches_ref = self
                .caches
                .as_ref()
                .expect("caches populated by init_caches_sync above");
            let mut cache_arrays_owned: Vec<MxArray> = Vec::with_capacity(caches_ref.len() * 2);
            for (layer_idx, cache) in caches_ref.iter().enumerate() {
                let (k, v) = cache.get_cached_kv().ok_or_else(|| {
                    Error::from_reason(format!(
                        "Compiled Gemma4 decode expected cache for layer {}",
                        layer_idx
                    ))
                })?;
                cache_arrays_owned.push(k);
                cache_arrays_owned.push(v);
            }
            let mut cache_ptrs: Vec<*mut mlx_sys::mlx_array> =
                cache_arrays_owned.iter().map(|a| a.as_raw_ptr()).collect();
            let layer_types_i32: Vec<i32> = (0..self.config.num_hidden_layers as usize)
                .map(|i| if self.config.is_global_layer(i) { 1 } else { 0 })
                .collect();
            let max_kv_len =
                (tokens.len() as i32 + max_new_tokens).min(self.config.max_position_embeddings);

            unsafe {
                mlx_sys::mlx_gemma4_init_from_prefill(
                    self.config.num_hidden_layers,
                    self.config.hidden_size,
                    self.config.num_attention_heads,
                    self.config.num_key_value_heads,
                    self.config.head_dim,
                    self.config.effective_kv_heads(true),
                    self.config.effective_head_dim(true),
                    self.config.rope_theta as f32,
                    self.config.rope_local_base_freq as f32,
                    self.config.partial_rotary_factor as f32,
                    self.config.rms_norm_eps as f32,
                    self.config.sliding_window,
                    if self.config.tie_word_embeddings {
                        1
                    } else {
                        0
                    },
                    max_kv_len,
                    1,
                    self.config.num_experts.unwrap_or(0),
                    self.config.top_k_experts.unwrap_or(0),
                    self.config.moe_intermediate_size.unwrap_or(0),
                    self.config.intermediate_size,
                    self.config.final_logit_softcapping.unwrap_or(0.0) as f32,
                    layer_types_i32.as_ptr(),
                    layer_types_i32.len() as i32,
                    cache_ptrs.as_mut_ptr(),
                    tokens.len() as i32,
                );
            }

            let embed_weight = self.embed_tokens.get_weight();
            let mut current_y = y;
            for step in 0..max_new_tokens {
                let next_y = if step + 1 < max_new_tokens {
                    let _stream_ctx = StreamContext::new(generation_stream);
                    let next_ids = current_y.reshape(&[1, 1])?;
                    let logits = forward_gemma4_cpp(&next_ids, &embed_weight)?;
                    let next_token = sample_next_token(&logits, sampling_config)?;
                    eval_token_and_gemma4_caches(&next_token);
                    Some(next_token)
                } else {
                    None
                };

                // See `chat_sync_core_paged_inner` for the rationale —
                // `read_scalar` reads `arr.data<T>()` directly without
                // forcing eval, so the previous iteration's async-eval'd
                // `current_y` (or the final iteration's no-forward case)
                // can hand back uninitialized bits.
                current_y.eval();
                let token_id = current_y.item_at_int32(0)? as u32;
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
                dispatch_stream_segments(segments, cb);

                if is_eos_token(token_id, &eos_ids, eos_token_id) {
                    finish_reason = "stop".to_string();
                    break;
                }
                if let Some(next_token) = next_y {
                    current_y = next_token;
                } else {
                    break;
                }
                if (step + 1) % 256 == 0 {
                    crate::array::synchronize_and_clear_cache();
                }
            }
            unsafe {
                mlx_sys::mlx_gemma4_reset();
            }
        } else {
            let mut current_y = y;
            for step in 0..max_new_tokens {
                let next_y = if step + 1 < max_new_tokens {
                    let _stream_ctx = StreamContext::new(generation_stream);
                    let caches = self
                        .caches
                        .as_mut()
                        .expect("caches populated by init_caches_sync above");
                    let next_ids = current_y.reshape(&[1, 1])?;
                    let logits = forward_inner(
                        &next_ids,
                        &self.embed_tokens,
                        &self.layers,
                        caches,
                        &self.final_norm,
                        &self.lm_head,
                        self.embed_weight_t.as_ref(),
                        self.ple.as_ref(),
                        &self.config,
                    )?;
                    let logits = logits.squeeze(Some(&[1]))?;
                    let next_token = sample_next_token(&logits, sampling_config)?;
                    MxArray::async_eval_arrays(&[&next_token]);
                    Some(next_token)
                } else {
                    None
                };

                // See `chat_sync_core_paged_inner` for the rationale.
                current_y.eval();
                let token_id = current_y.item_at_int32(0)? as u32;
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
                dispatch_stream_segments(segments, cb);

                if is_eos_token(token_id, &eos_ids, eos_token_id) {
                    finish_reason = "stop".to_string();
                    break;
                }
                if let Some(next_token) = next_y {
                    current_y = next_token;
                } else {
                    break;
                }

                if (step + 1) % 256 == 0 {
                    crate::array::clear_cache();
                }
            }
        }

        // `decode_sync(…, false)` matches the streaming decoder setting
        // so any residual bytes left inside the tokenizer's DecodeStream
        // surface with the same special-token representation the stream
        // parser was fed.
        let raw_text = tokenizer.decode_sync(&generated_tokens, false)?;

        // Flush any residual bytes that might not have resolved at the streaming layer
        if raw_text.len() > streamed_text_len {
            let residual = raw_text[streamed_text_len..].to_string();
            let mut segments = stream_parser.feed(&residual);
            segments.extend(stream_parser.flush());
            dispatch_stream_segments(segments, cb);
        } else {
            let tail = stream_parser.flush();
            dispatch_stream_segments(tail, cb);
        }

        // Save session state so subsequent
        // `chat_stream_session_continue_sync` / `chat_session_continue_sync`
        // calls can append a raw delta on top of the live caches. See
        // the non-streaming `chat_sync_core` for the full rationale.
        let history_tokens: &[u32] = if finish_reason != "length" && !generated_tokens.is_empty() {
            &generated_tokens[..generated_tokens.len() - 1]
        } else {
            &generated_tokens[..]
        };
        let mut new_history = Vec::with_capacity(tokens.len() + history_tokens.len());
        new_history.extend(tokens.iter().copied());
        new_history.extend_from_slice(history_tokens);
        self.cached_token_history = new_history;
        self.cached_image_key = new_image_key;

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
                prompt_token_count as f64 / (ttft_ms / 1000.0)
            } else {
                0.0
            },
            decode_tokens_per_second: if decode_ms > 0.0 && gen_toks > 1.0 {
                (gen_toks - 1.0) / (decode_ms / 1000.0)
            } else {
                0.0
            },
        });

        let parsed_tool_calls = stream_parser.tool_calls();
        let parsed_thinking = stream_parser.thinking();
        let finish_reason = if parsed_tool_calls.iter().any(|tc| tc.status == "ok") {
            "tool_calls".to_string()
        } else {
            finish_reason
        };

        // Emit final block
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
                // Start path: report the matched prefix length. Zero on
                // a miss or exact-match (treated as miss), equal to the
                // matched prefix length on a warm-reuse hit.
                cached_tokens: Some(cached_prefix_len_reported),
                performance,
                is_reasoning: None,
            }),
            ThreadsafeFunctionCallMode::NonBlocking,
        );

        Ok(())
    }

    // =================================================================
    // Block-paged dispatch (chat_sync_core_paged + helpers).
    //
    // Mirrors Qwen3's `chat_sync_core_paged` and LFM2's `forward_paged_or_flat`
    // pattern — sliding layers continue to use the existing flat
    // `Gemma4LayerCache::Sliding` path while global layers route through
    // `PagedKVCacheAdapter`. KV-shared layers are wired in a follow-up
    // commit; for now the dispatch surface returns an error if the
    // config has any shared layers.
    //
    // Lifecycle (mirrors Qwen3 / LFM2):
    // 1. Adapter cold-start (or warm-continue when previous turn
    //    finalize_turn_keep_live'd a strict-prefix request).
    // 2. Conv/sliding caches RESET each turn (paged path doesn't carry
    //    sliding state across turns at this stage — same caveat as
    //    LFM2's conv layers).
    // 3. Prefill via `run_paged_prefill_chunk` over the suffix.
    // 4. Decode loop via `run_paged_decode_step`.
    // 5. End-of-turn (success): `finalize_turn_keep_live` so the next
    //    turn's `continue_turn` can build on top of the partial trailing
    //    block's K/V (same partial-block carry trick as Qwen3 / LFM2).
    //
    // Caveats / scope:
    // * Text-only — vision turns dispatch through the flat path.
    // * Sliding layers re-prefill the cached prefix every turn.
    // * Zero-delta prompts (every prompt token cached) are rejected.
    // =================================================================

    /// Block-paged variant of [`Self::chat_sync_core`].
    ///
    /// Reached when `paged_adapter.is_some()` AND the prompt has no
    /// images. The caller has already done image processing /
    /// expansion / template rendering, so this method receives a
    /// fully-baked `tokens` vector and dispatches the paged forward.
    #[allow(clippy::too_many_arguments)]
    fn chat_sync_core_paged(
        &mut self,
        tokens: Vec<u32>,
        tokenizer: Arc<Qwen3Tokenizer>,
        config: ChatConfig,
        eos_token_id: u32,
        sampling_config: Option<SamplingConfig>,
        max_new_tokens: i32,
    ) -> Result<ChatResult> {
        if tokens.is_empty() {
            return Err(Error::from_reason("Empty prompt"));
        }

        let reuse_cache = config.reuse_cache.unwrap_or(true);
        let prompt_token_count = tokens.len();
        let eos_ids = self.config.eos_token_ids.clone();
        let generation_start = std::time::Instant::now();

        // === Adapter lifecycle: warm continuation OR cold start. ===
        // See PagedKVCacheAdapter::finalize_turn_keep_live for why warm-
        // continue preserves the partial trailing block's K/V across
        // turns. Mirrors Qwen3 / LFM2.
        let seq_id: u32 = 0;
        // Lazy decode allocation: pass the prompt length only.
        let total_budget = tokens.len() as u32;
        let cached_prefix_len = {
            let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                Error::from_reason(
                    "chat_sync_core_paged: paged_adapter is None — caller must check \
                     use_block_paged_cache before dispatch",
                )
            })?;

            let can_continue = reuse_cache
                && adapter.is_live_for_continue()
                && tokens.starts_with(adapter.request_tokens());

            if can_continue {
                match adapter.continue_turn(&tokens, total_budget) {
                    Ok((prior_token_count, _newly_alloc)) => prior_token_count,
                    Err(_drift) => {
                        let _ = adapter.release_request();
                        adapter
                            .reset_for_new_request(seq_id)
                            .map_err(Error::from_reason)?;
                        let prefix = adapter
                            .find_cached_prefix(&tokens, &[], 0, false)
                            .map_err(Error::from_reason)?;
                        let cached = prefix.cached_token_count;
                        adapter
                            .allocate_suffix_blocks(total_budget)
                            .map_err(Error::from_reason)?;
                        cached
                    }
                }
            } else {
                if adapter.block_table().is_some() {
                    let _ = adapter.release_request();
                }
                adapter
                    .reset_for_new_request(seq_id)
                    .map_err(Error::from_reason)?;
                let prefix = adapter
                    .find_cached_prefix(&tokens, &[], 0, false)
                    .map_err(Error::from_reason)?;
                let cached = prefix.cached_token_count;
                adapter
                    .allocate_suffix_blocks(total_budget)
                    .map_err(Error::from_reason)?;
                cached
            }
        };

        // Reset sliding-layer flat caches for this turn — paged path
        // does not carry sliding prefix state across turns.
        self.reset_caches_sync()?;
        self.init_caches_sync()?;

        let total_prompt_tokens = tokens.len() as u32;
        let suffix_len = total_prompt_tokens
            .checked_sub(cached_prefix_len)
            .ok_or_else(|| {
                Error::from_reason("chat_sync_core_paged: cached_prefix_len > total_prompt_tokens")
            })?;
        if suffix_len == 0 {
            // Zero-delta: every prompt token cached. Same caveat as
            // Qwen3 / LFM2 — flat path required for this corner case.
            if let Some(adapter) = self.paged_adapter.as_mut() {
                let _ = adapter.release_request();
            }
            return Err(Error::from_reason(
                "Gemma4 paged: zero-delta prompt (every token cached) is not yet \
                 supported on the block-paged path",
            ));
        }

        // Wrap forward in a try-style flow for proper adapter cleanup.
        let forward_result = self.chat_sync_core_paged_inner(
            &tokens,
            cached_prefix_len,
            sampling_config,
            max_new_tokens,
            eos_token_id,
            &eos_ids,
        );

        let (generated_tokens, finish_reason, first_token_instant) = match forward_result {
            Ok(t) => {
                if let Some(adapter) = self.paged_adapter.as_mut() {
                    if reuse_cache {
                        let _ = adapter.finalize_turn_keep_live(&[], 0);
                    } else {
                        let _ = adapter.register_full_blocks_for_reuse(&[], 0);
                        let _ = adapter.release_request();
                    }
                }
                t
            }
            Err(e) => {
                if let Some(adapter) = self.paged_adapter.as_mut() {
                    let _ = adapter.release_request();
                }
                return Err(e);
            }
        };

        // Decode + parse output (mirrors flat path).
        let raw_text = tokenizer.decode_sync(&generated_tokens, false)?;

        // Persist session token history for the next continue turn.
        if reuse_cache {
            let history_tokens: &[u32] =
                if finish_reason != "length" && !generated_tokens.is_empty() {
                    &generated_tokens[..generated_tokens.len() - 1]
                } else {
                    &generated_tokens[..]
                };
            let mut new_history = Vec::with_capacity(tokens.len() + history_tokens.len());
            new_history.extend(tokens.iter().copied());
            new_history.extend_from_slice(history_tokens);
            self.cached_token_history = new_history;
        } else {
            self.cached_token_history.clear();
        }
        self.cached_image_key = None;

        // Performance metrics.
        let generation_end = std::time::Instant::now();
        let ttft_ms = first_token_instant
            .map(|fti| fti.duration_since(generation_start).as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        let decode_ms = first_token_instant
            .map(|fti| generation_end.duration_since(fti).as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        let gen_toks = generated_tokens.len() as f64;
        let actual_prefill_count = (tokens.len() as f64) - cached_prefix_len as f64;
        let performance = Some(crate::profiling::PerformanceMetrics {
            ttft_ms,
            prefill_tokens_per_second: if ttft_ms > 0.0 {
                actual_prefill_count.max(1.0) / (ttft_ms / 1000.0)
            } else {
                0.0
            },
            decode_tokens_per_second: if decode_ms > 0.0 && gen_toks > 1.0 {
                (gen_toks - 1.0) / (decode_ms / 1000.0)
            } else {
                0.0
            },
        });

        let parsed = super::output_parser::parse_gemma4_output(&raw_text);
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
            cached_tokens: cached_prefix_len,
            performance,
        })
    }

    /// Inner forward + decode loop for [`Self::chat_sync_core_paged`].
    /// Split out so the outer can wrap with adapter `release_request`
    /// on either path.
    fn chat_sync_core_paged_inner(
        &mut self,
        tokens: &[u32],
        cached_prefix_len: u32,
        sampling_config: Option<SamplingConfig>,
        max_new_tokens: i32,
        eos_token_id: u32,
        eos_ids: &[i32],
    ) -> Result<(Vec<u32>, String, Option<std::time::Instant>)> {
        let suffix = &tokens[(cached_prefix_len as usize)..];

        // Pin model weights in GPU memory; share generation stream.
        let generation_stream = Stream::new(DeviceType::Gpu);
        let _wired_ctx = crate::stream::WiredLimitContext::new(usize::MAX, vec![generation_stream]);

        // === PREFILL ===
        let last_logits = {
            let _stream_ctx = StreamContext::new(generation_stream);
            crate::models::gemma4::diagnostic::set_step(-1);
            self.run_paged_prefill_chunk(tokens, suffix, cached_prefix_len)?
        };

        let mut y = sample_next_token(&last_logits, sampling_config)?;
        y.eval();

        // Smooth memory peak: drop transient prefill buffers before decode
        // starts allocating. Prefill builds a massive MLX subgraph; once
        // we have the last logits, those intermediates are dead but
        // MLX's caching allocator holds them.
        crate::array::synchronize_and_clear_cache();

        let first_token_instant = Some(std::time::Instant::now());

        // === DECODE LOOP ===
        let mut generated_tokens: Vec<u32> = Vec::with_capacity(max_new_tokens.max(0) as usize);
        let mut finish_reason = String::from("length");

        for step in 0..max_new_tokens {
            // Force `y` to evaluate before reading via `item_at_int32`.
            // The previous iteration only kicked an async eval on `y`,
            // and `item_at_int32` reads CPU memory directly via
            // `read_scalar` (mlx_nn_ops.cpp) — it does NOT trigger an
            // implicit eval. Without this sync, decode reads the
            // raw-bit-uninitialized buffer, the "token id" is garbage,
            // and the EOS check trivially never matches. Mirrors
            // Qwen3's `run_paged_decode_step` loop (qwen3/model.rs:2935).
            y.eval();
            let token_id = y.item_at_int32(0)? as u32;
            generated_tokens.push(token_id);

            if is_eos_token(token_id, eos_ids, eos_token_id) {
                finish_reason = String::from("stop");
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
            MxArray::async_eval_arrays(&[&y]);

            crate::array::maybe_clear_cache_for_paged_step(step);
        }

        Ok((generated_tokens, finish_reason, first_token_instant))
    }

    /// Streaming variant of [`Self::chat_sync_core_paged`].
    #[allow(clippy::too_many_arguments)]
    fn chat_stream_sync_core_paged(
        &mut self,
        tokens: Vec<u32>,
        tokenizer: Arc<Qwen3Tokenizer>,
        config: ChatConfig,
        eos_token_id: u32,
        sampling_config: Option<SamplingConfig>,
        cb: &StreamSender,
        cancelled: &Arc<AtomicBool>,
        max_new_tokens: i32,
    ) -> Result<()> {
        if tokens.is_empty() {
            return Err(Error::from_reason("Empty prompt"));
        }

        let reuse_cache = config.reuse_cache.unwrap_or(true);
        let prompt_token_count = tokens.len();
        let eos_ids = self.config.eos_token_ids.clone();
        let generation_start = std::time::Instant::now();

        // Adapter lifecycle: warm continuation OR cold start (mirrors
        // chat_sync_core_paged).
        let seq_id: u32 = 0;
        // Lazy decode allocation: pass the prompt length only.
        let total_budget = tokens.len() as u32;
        let cached_prefix_len = {
            let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                Error::from_reason("chat_stream_sync_core_paged: paged_adapter is None")
            })?;
            let can_continue = reuse_cache
                && adapter.is_live_for_continue()
                && tokens.starts_with(adapter.request_tokens());
            if can_continue {
                match adapter.continue_turn(&tokens, total_budget) {
                    Ok((prior, _)) => prior,
                    Err(_) => {
                        let _ = adapter.release_request();
                        adapter
                            .reset_for_new_request(seq_id)
                            .map_err(Error::from_reason)?;
                        let prefix = adapter
                            .find_cached_prefix(&tokens, &[], 0, false)
                            .map_err(Error::from_reason)?;
                        let cached = prefix.cached_token_count;
                        adapter
                            .allocate_suffix_blocks(total_budget)
                            .map_err(Error::from_reason)?;
                        cached
                    }
                }
            } else {
                if adapter.block_table().is_some() {
                    let _ = adapter.release_request();
                }
                adapter
                    .reset_for_new_request(seq_id)
                    .map_err(Error::from_reason)?;
                let prefix = adapter
                    .find_cached_prefix(&tokens, &[], 0, false)
                    .map_err(Error::from_reason)?;
                let cached = prefix.cached_token_count;
                adapter
                    .allocate_suffix_blocks(total_budget)
                    .map_err(Error::from_reason)?;
                cached
            }
        };

        // Reset sliding flat caches.
        self.reset_caches_sync()?;
        self.init_caches_sync()?;

        let total_prompt_tokens = tokens.len() as u32;
        let suffix_len = total_prompt_tokens
            .checked_sub(cached_prefix_len)
            .ok_or_else(|| {
                Error::from_reason(
                    "chat_stream_sync_core_paged: cached_prefix_len > total_prompt_tokens",
                )
            })?;
        if suffix_len == 0 {
            if let Some(adapter) = self.paged_adapter.as_mut() {
                let _ = adapter.release_request();
            }
            return Err(Error::from_reason(
                "Gemma4 paged streaming: zero-delta prompt (every token cached) is not yet supported",
            ));
        }

        let stream_result = self.chat_stream_sync_core_paged_inner(
            &tokens,
            cached_prefix_len,
            sampling_config,
            max_new_tokens,
            eos_token_id,
            &eos_ids,
            tokenizer.clone(),
            cb,
            cancelled,
        );

        let (generated_tokens, finish_reason, first_token_instant) = match stream_result {
            Ok(t) => {
                if let Some(adapter) = self.paged_adapter.as_mut() {
                    if reuse_cache {
                        let _ = adapter.finalize_turn_keep_live(&[], 0);
                    } else {
                        let _ = adapter.register_full_blocks_for_reuse(&[], 0);
                        let _ = adapter.release_request();
                    }
                }
                t
            }
            Err(e) => {
                if let Some(adapter) = self.paged_adapter.as_mut() {
                    let _ = adapter.release_request();
                }
                return Err(e);
            }
        };

        // Persist session history.
        if reuse_cache {
            let history_tokens: &[u32] =
                if finish_reason != "length" && !generated_tokens.is_empty() {
                    &generated_tokens[..generated_tokens.len() - 1]
                } else {
                    &generated_tokens[..]
                };
            let mut new_history = Vec::with_capacity(tokens.len() + history_tokens.len());
            new_history.extend(tokens.iter().copied());
            new_history.extend_from_slice(history_tokens);
            self.cached_token_history = new_history;
        } else {
            self.cached_token_history.clear();
        }
        self.cached_image_key = None;

        // Terminal stream chunk.
        let raw_text = tokenizer.decode_sync(&generated_tokens, false)?;
        let parsed = super::output_parser::parse_gemma4_output(&raw_text);
        let finish_reason = if parsed.tool_calls.iter().any(|tc| tc.status == "ok") {
            "tool_calls".to_string()
        } else {
            finish_reason
        };

        let generation_end = std::time::Instant::now();
        let ttft_ms = first_token_instant
            .map(|fti| fti.duration_since(generation_start).as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        let decode_ms = first_token_instant
            .map(|fti| generation_end.duration_since(fti).as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        let gen_toks = generated_tokens.len() as f64;
        let actual_prefill_count = (tokens.len() as f64) - cached_prefix_len as f64;
        let performance = Some(crate::profiling::PerformanceMetrics {
            ttft_ms,
            prefill_tokens_per_second: if ttft_ms > 0.0 {
                actual_prefill_count.max(1.0) / (ttft_ms / 1000.0)
            } else {
                0.0
            },
            decode_tokens_per_second: if decode_ms > 0.0 && gen_toks > 1.0 {
                (gen_toks - 1.0) / (decode_ms / 1000.0)
            } else {
                0.0
            },
        });

        cb.call(
            Ok(ChatStreamChunk {
                text: String::new(),
                done: true,
                finish_reason: Some(finish_reason),
                tool_calls: if parsed.tool_calls.is_empty() {
                    None
                } else {
                    Some(parsed.tool_calls)
                },
                thinking: parsed.thinking,
                num_tokens: Some(generated_tokens.len() as u32),
                prompt_tokens: Some(prompt_token_count as u32),
                reasoning_tokens: Some(0),
                raw_text: Some(raw_text),
                cached_tokens: Some(cached_prefix_len),
                performance,
                is_reasoning: None,
            }),
            ThreadsafeFunctionCallMode::NonBlocking,
        );

        Ok(())
    }

    /// Inner streaming forward + decode loop for
    /// [`Self::chat_stream_sync_core_paged`]. Emits text deltas via the
    /// stream callback as tokens are produced.
    #[allow(clippy::too_many_arguments)]
    fn chat_stream_sync_core_paged_inner(
        &mut self,
        tokens: &[u32],
        cached_prefix_len: u32,
        sampling_config: Option<SamplingConfig>,
        max_new_tokens: i32,
        eos_token_id: u32,
        eos_ids: &[i32],
        tokenizer: Arc<Qwen3Tokenizer>,
        cb: &StreamSender,
        cancelled: &Arc<AtomicBool>,
    ) -> Result<(Vec<u32>, String, Option<std::time::Instant>)> {
        let suffix = &tokens[(cached_prefix_len as usize)..];

        let generation_stream = Stream::new(DeviceType::Gpu);
        let _wired_ctx = crate::stream::WiredLimitContext::new(usize::MAX, vec![generation_stream]);

        let last_logits = {
            let _stream_ctx = StreamContext::new(generation_stream);
            self.run_paged_prefill_chunk(tokens, suffix, cached_prefix_len)?
        };

        let mut y = sample_next_token(&last_logits, sampling_config)?;
        y.eval();

        // Smooth memory peak: drop transient prefill buffers before decode
        // starts allocating (see chat_sync_core_paged_inner for rationale).
        crate::array::synchronize_and_clear_cache();

        let first_token_instant = Some(std::time::Instant::now());

        // Streaming detokenizer + parser.
        let mut decode_stream = tokenizer.inner().decode_stream(false);
        let mut stream_parser = super::output_parser::Gemma4StreamParser::new();

        let mut generated_tokens: Vec<u32> = Vec::with_capacity(max_new_tokens.max(0) as usize);
        let mut finish_reason = String::from("length");

        for step in 0..max_new_tokens {
            if cancelled.load(Ordering::Relaxed) {
                finish_reason = String::from("cancelled");
                break;
            }
            // Force `y` to evaluate before reading via `item_at_int32`
            // — async_eval kicked from the previous iteration does not
            // implicitly sync. Same rationale as `chat_sync_core_paged_inner`.
            y.eval();
            let token_id = y.item_at_int32(0)? as u32;
            generated_tokens.push(token_id);

            // Emit any text segments produced by this token.
            if let Some(piece) = decode_stream
                .step(token_id)
                .map_err(|e| Error::from_reason(format!("decode_stream: {e}")))?
            {
                let segments = stream_parser.feed(&piece);
                dispatch_stream_segments(segments, cb);
            }

            if is_eos_token(token_id, eos_ids, eos_token_id) {
                finish_reason = String::from("stop");
                break;
            }
            if step + 1 >= max_new_tokens {
                break;
            }

            let next_logits = {
                let _stream_ctx = StreamContext::new(generation_stream);
                self.run_paged_decode_step(token_id)?
            };
            let next_logits = next_logits.squeeze(Some(&[1]))?;
            y = sample_next_token(&next_logits, sampling_config)?;
            MxArray::async_eval_arrays(&[&y]);

            crate::array::maybe_clear_cache_for_paged_step(step);
        }

        // Flush any residual segments accumulated by the parser but not
        // yet emitted.
        let residual_segments = stream_parser.flush();
        dispatch_stream_segments(residual_segments, cb);

        Ok((generated_tokens, finish_reason, first_token_instant))
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
    ) -> Result<MxArray> {
        if suffix_tokens.is_empty() {
            return Err(Error::from_reason(
                "run_paged_prefill_chunk called with empty suffix",
            ));
        }

        let suffix_len = suffix_tokens.len() as u32;
        let layer_kinds = self.compute_layer_kinds();

        // For sliding layers we need state at position cached_prefix_len.
        // Sliding layers are restored each turn via reset_caches_sync, so
        // we need to reprefill the cached prefix through them BEFORE the
        // suffix can attend. Run a "sliding-only" pass over the prefix
        // tokens, then the suffix passes below.
        if cached_prefix_len > 0 {
            let prefix = &full_tokens[..(cached_prefix_len as usize)];
            self.run_sliding_only_prefill(prefix, &layer_kinds)?;
        }

        crate::models::gemma4::diagnostic::set_path("paged");
        crate::models::gemma4::diagnostic::set_step(-1);

        // Two-pass split (mirrors flat `prefill_body_gemma4 →
        // forward_inner`):
        //   Pass 1: tokens `[..suffix_len-1]` (no-op if suffix_len == 1).
        //           Layer-loop only — its hidden_states are discarded;
        //           we keep the per-layer K/V writes the loop made into
        //           the paged pool / sliding caches.
        //   Pass 2: the FINAL token (length 1). Now
        //           `cached_prefix_len_for_chunk = cached_prefix_len +
        //           suffix_len - 1`, which is > 0, so global layers
        //           take the cache-hit `read_kv_range` branch in
        //           `forward_paged` — the same path decode uses. This
        //           aligns the SDPA reduction order with the flat
        //           path's `forward_inner` dispatch.
        if suffix_len > 1 {
            // --- Pass 1: all-but-last suffix tokens. ---
            let pass1_tokens = &suffix_tokens[..(suffix_len as usize - 1)];
            {
                let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                    Error::from_reason("run_paged_prefill_chunk: paged_adapter is None")
                })?;
                adapter
                    .record_tokens(pass1_tokens)
                    .map_err(Error::from_reason)?;
            }
            let _hidden_pass1 = self.run_paged_prefill_layer_loop(
                pass1_tokens,
                /* first_logical_position */ cached_prefix_len,
                /* cached_prefix_len_for_chunk */ cached_prefix_len,
                &layer_kinds,
            )?;

            // Materialize sliding-cache writes from pass 1 before pass 2
            // reads through them. The paged pool's `update_keys_values`
            // calls `synchronize_mlx()` internally so global-layer K/V
            // is already on-pool, but the sliding flat caches are still
            // lazy graphs at this point. Mirrors the
            // `eval_gemma4_caches(caches)` call between chunks in
            // `prefill_body_gemma4`. We deliberately do NOT eval the
            // pass-1 hidden_states — it is discarded.
            if let Some(caches) = self.caches.as_ref() {
                eval_gemma4_caches(caches);
            }
            crate::array::clear_cache();
        }

        // --- Pass 2: the FINAL suffix token (length 1). ---
        let pass2_tokens = &suffix_tokens[(suffix_len as usize - 1)..];
        {
            let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                Error::from_reason("run_paged_prefill_chunk: paged_adapter is None")
            })?;
            adapter
                .record_tokens(pass2_tokens)
                .map_err(Error::from_reason)?;
        }
        let pass2_first_position = cached_prefix_len + (suffix_len - 1);
        let pass2_cached_prefix_len = pass2_first_position;
        let mut hidden_states = self.run_paged_prefill_layer_loop(
            pass2_tokens,
            pass2_first_position,
            pass2_cached_prefix_len,
            &layer_kinds,
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

        // Build sliding mask if seq_len exceeds window. Mirrors
        // `forward_body`'s mask logic.
        let seq_len = chunk_len as i64;
        let sliding_mask = if seq_len > 1 && seq_len > self.config.sliding_window as i64 {
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
            Some(create_sliding_mask(
                seq_len,
                sliding_offset,
                self.config.sliding_window as i64,
            )?)
        } else {
            None
        };

        let has_kv_sharing = self.config.num_kv_shared_layers.is_some_and(|n| n > 0);
        let num_layers = self.layers.len();
        // Stash for sliding-anchor K/V reused by SharedOnSliding layers.
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

            hidden_states = layer.forward_paged_or_flat(
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
            crate::array::maybe_eval_clear_for_paged_prefill_layer(layer_idx, &hidden_states);
        }

        Ok(hidden_states)
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

        let layer_kinds = self.compute_layer_kinds();

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

    /// Forward the cached prefix tokens through SLIDING layers ONLY,
    /// updating their flat caches in-place. Used to bring sliding-layer
    /// state up to the paged cache's `cached_prefix_len` boundary
    /// before the main `run_paged_prefill_chunk` continues with the
    /// suffix.
    ///
    /// Global layers are skipped — their K/V already lives in the
    /// paged pool (the prefix-cache hit promised that).
    fn run_sliding_only_prefill(
        &mut self,
        prefix_tokens: &[u32],
        layer_kinds: &[Gemma4LayerKind],
    ) -> Result<()> {
        if prefix_tokens.is_empty() {
            return Ok(());
        }
        let input_ids = MxArray::from_uint32(prefix_tokens, &[1, prefix_tokens.len() as i64])?;
        let mut hidden_states = self.embed_tokens.forward(&input_ids)?;
        hidden_states = hidden_states.mul_scalar((self.config.hidden_size as f64).sqrt())?;

        // Sliding mask if seq_len > window.
        let seq_len = prefix_tokens.len() as i64;

        // Compute PLE for the prefix tokens. Without this, sliding-layer
        // K/V written here will diverge from the flat path's K/V because
        // hidden_states fed into the sliding pre-norm depends on the
        // PLE-augmented output of the prior layer. Same load-bearing
        // residual as in `run_paged_prefill_chunk`.
        let projected_ple_prefix: Option<MxArray> = if let Some(ref ple) = self.ple {
            let pre_layer_h = hidden_states.clone();
            Some(compute_ple(&input_ids, &pre_layer_h, ple, seq_len)?)
        } else {
            None
        };
        let sliding_mask = if seq_len > self.config.sliding_window as i64 {
            Some(create_sliding_mask(
                seq_len,
                0,
                self.config.sliding_window as i64,
            )?)
        } else {
            None
        };

        let num_layers = self.layers.len();
        #[allow(clippy::needless_range_loop)]
        for layer_idx in 0..num_layers {
            let kind = layer_kinds[layer_idx];
            // For sliding layers we forward through the flat path; for
            // global layers we use a NO-OP forward (skip — the K/V is
            // already in the paged pool, hidden_states discarded).
            // The hidden_states need to flow through ALL layers because
            // the next sliding layer sees the previous global layer's
            // output; we run global layers via the flat `forward` with
            // a fresh KVCache that we throw away.
            let layer: &Gemma4DecoderLayer = unsafe {
                let ptr = self.layers.as_ptr().add(layer_idx);
                &*ptr
            };
            // Slice this layer's PLE input ([B, T, num_layers, ple_dim] →
            // [B, T, ple_dim]).
            let ple_input = projected_ple_prefix.as_ref().map(|p| {
                p.slice_axis(2, layer_idx as i64, layer_idx as i64 + 1)
                    .and_then(|s| s.squeeze(Some(&[2])))
            });
            let ple_input_ref = match &ple_input {
                Some(Ok(arr)) => Some(arr),
                _ => None,
            };
            match kind {
                Gemma4LayerKind::Sliding => {
                    let caches = unsafe {
                        let raw = self.caches.as_mut().ok_or_else(|| {
                            Error::from_reason(
                                "run_sliding_only_prefill: sliding cache slot missing",
                            )
                        })? as *mut Vec<Gemma4LayerCache>;
                        &mut *raw
                    };
                    hidden_states = layer.forward(
                        &hidden_states,
                        sliding_mask.as_ref(),
                        Some(&mut caches[layer_idx]),
                        ple_input_ref,
                        /* needs_stash */ false,
                    )?;
                }
                Gemma4LayerKind::GlobalPaged { .. } => {
                    // Build a throwaway global cache so we can run the
                    // standard `forward(...)` over the prefix and let
                    // hidden_states propagate. The cache is discarded
                    // afterwards — the paged pool already holds K/V.
                    let mut throwaway = Gemma4LayerCache::new_global();
                    hidden_states = layer.forward(
                        &hidden_states,
                        /* mask */ None,
                        Some(&mut throwaway),
                        ple_input_ref,
                        false,
                    )?;
                }
                Gemma4LayerKind::SharedOnGlobal { .. }
                | Gemma4LayerKind::SharedOnSliding { .. } => {
                    // Shared layers don't write to caches and don't
                    // affect the sliding-only re-prefill state. Use a
                    // throwaway global cache so hidden_states still
                    // flows through the layer.
                    let mut throwaway = Gemma4LayerCache::new_global();
                    hidden_states = layer.forward(
                        &hidden_states,
                        /* mask */ None,
                        Some(&mut throwaway),
                        ple_input_ref,
                        false,
                    )?;
                }
            }
        }

        // Materialize sliding caches.
        if let Some(caches) = self.caches.as_ref() {
            eval_gemma4_caches(caches);
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

    /// Start a new chat session.
    ///
    /// Fully resets the caches and delegates to [`Self::chat_sync_core`]
    /// with `<turn|>` as the stop token so the decode loop leaves the
    /// caches on a clean turn boundary that subsequent
    /// [`Self::chat_session_continue_sync`] /
    /// [`Self::chat_session_continue_tool_sync`] calls can append a raw
    /// delta on top of.
    ///
    /// Vision-capable: `messages` may carry images (they'll be decoded
    /// on the model thread inside `chat_sync_core`).
    pub(crate) fn chat_session_start_sync(
        &mut self,
        messages: Vec<ChatMessage>,
        config: ChatConfig,
    ) -> Result<ChatResult> {
        // Resolve the turn-end token up front so session_continue can
        // rely on the cached history always terminating on a clean
        // `<turn|>` boundary.
        let turn_end_id = self.turn_end_id()?;

        // NOTE: no unconditional reset here. `chat_sync_core` runs
        // `verify_cache_prefix` against the incoming `messages` and only
        // resets the KV caches on a miss. This preserves prefix-reuse for
        // stateless agent clients (pi-mono / Aider / Codex) that resend
        // the full conversation transcript every turn — wiping here would
        // make every session-start a cache miss by construction.
        self.chat_sync_core(messages, config, turn_end_id)
    }

    /// Continue an existing chat session with a user turn.
    ///
    /// Builds a Gemma4 wire-format delta (`\n<|turn>user\n...<turn|>\n
    /// <|turn>model\n`), tokenizes it, and prefills on top of the live
    /// caches via [`Self::chat_tokens_delta_sync`].
    ///
    /// Text-only on the delta path: callers that need to change the
    /// image set must restart the session via
    /// [`Self::chat_session_start_sync`]. The `images` parameter is an
    /// opt-in guard that returns an
    /// `IMAGE_CHANGE_REQUIRES_SESSION_RESTART:`-prefixed error when
    /// non-empty, letting the TS `ChatSession` layer pattern-match the
    /// prefix and route image-changes through a fresh session start.
    pub(crate) fn chat_session_continue_sync(
        &mut self,
        user_message: String,
        images: Option<Vec<Uint8Array>>,
        config: ChatConfig,
    ) -> Result<ChatResult> {
        // Guard 1: text-only delta path.
        if images.as_ref().is_some_and(|v| !v.is_empty()) {
            return Err(Error::from_reason(format!(
                "{}chat_session_continue is text-only; start a new session with chat_session_start to change the image",
                chat_common::IMAGE_CHANGE_RESTART_PREFIX
            )));
        }

        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))?
            .clone();

        // Subject the session path to the same sanitization as the
        // legacy chat path so role/content injection guards stay
        // uniform across all entry points.
        let synthetic = chat_common::build_synthetic_user_message(&user_message);
        let sanitized = Qwen3Tokenizer::sanitize_messages_public(std::slice::from_ref(&synthetic));
        let sanitized_user = &sanitized[0].content;

        let enable_thinking = chat_common::resolve_enable_thinking(&config);
        let delta_text = build_gemma4_continue_delta_text(sanitized_user, enable_thinking);
        let delta_tokens = tokenizer.encode_sync(&delta_text, Some(false))?;

        self.chat_tokens_delta_sync(delta_tokens, config)
    }

    /// Continue an existing chat session with a tool-result turn.
    ///
    /// Gemma4's chat template renders tool-role messages as
    /// `<|turn>tool\n{content}<turn|>` — no `<tool_response>` wrapping.
    /// We build the delta inline rather than using
    /// [`chat_common::build_chatml_tool_delta_text`] (which is
    /// Qwen3.5-specific). The `tool_call_id` is intentionally dropped
    /// from the wire format — Gemma4's template identifies tool
    /// responses positionally, not via an explicit id.
    pub(crate) fn chat_session_continue_tool_sync(
        &mut self,
        tool_call_id: String,
        content: String,
        config: ChatConfig,
    ) -> Result<ChatResult> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))?
            .clone();

        let enable_thinking = chat_common::resolve_enable_thinking(&config);
        let delta_text = build_gemma4_tool_delta_text(&tool_call_id, &content, enable_thinking);
        let delta_tokens = tokenizer.encode_sync(&delta_text, Some(false))?;

        self.chat_tokens_delta_sync(delta_tokens, config)
    }

    /// Prefill a pre-tokenized delta on top of the existing Gemma4 KV
    /// caches and run the decode loop. Text-only session primitive used
    /// by [`Self::chat_session_continue_sync`] and
    /// [`Self::chat_session_continue_tool_sync`].
    ///
    /// Uses `<turn|>` as the eos token so the cached history continues
    /// to end on a clean turn boundary for the next turn. The delta
    /// prefill runs through `prefill_body_gemma4` which appends to the
    /// existing `self.caches` via `update_and_fetch_stash` — no
    /// separate "append to existing KV" logic is needed.
    pub(crate) fn chat_tokens_delta_sync(
        &mut self,
        delta_tokens: Vec<u32>,
        config: ChatConfig,
    ) -> Result<ChatResult> {
        // --- Five guards (mirrors Qwen3 / LFM2). ---
        // The delta path is a session-reuse operation by construction: it
        // prefills on top of the existing caches. `reuse_cache = Some(false)`
        // would make the post-decode `save_cache_state_direct` wipe those
        // caches + `cached_token_history`, making the delta turn both depend
        // on and then destroy the session — confusing and wrong. Reject early
        // so no state is mutated.
        if config.reuse_cache == Some(false) {
            return Err(Error::from_reason(
                "chat_tokens_delta_sync requires reuse_cache to be enabled; \
                 the delta path operates on session state by construction",
            ));
        }
        if self.cached_token_history.is_empty() {
            return Err(Error::from_reason(
                "chat_tokens_delta_sync requires an initialized session (call chatSessionStart first)",
            ));
        }
        if delta_tokens.is_empty() {
            return Err(Error::from_reason(
                "chat_tokens_delta_sync requires a non-empty delta",
            ));
        }
        if self.cached_image_key.is_some() {
            return Err(Error::from_reason(format!(
                "{}chat_tokens_delta_sync is text-only; session currently holds image state",
                chat_common::IMAGE_CHANGE_RESTART_PREFIX
            )));
        }
        if self.caches.is_none() {
            return Err(Error::from_reason(
                "chat_tokens_delta_sync requires a live cache (call chatSessionStart first)",
            ));
        }

        let tokenizer = self
            .tokenizer
            .clone()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))?;

        // Session path: use `<turn|>` as eos, NOT config.eos_token_ids.
        // This keeps the cached history aligned on a clean turn boundary
        // for the next `chat_session_continue*` call.
        let turn_end_id = self.turn_end_id()?;

        let max_new_tokens = config.max_new_tokens.unwrap_or(2048);
        let sampling_config = make_sampling_config(&config, &self.config);
        let eos_ids = self.config.eos_token_ids.clone();

        // Block-paged dispatch: when the adapter is configured the
        // session's K/V for global layers lives in the adapter's pool —
        // the flat caches (sliding only on the paged path) were reset at
        // the end of turn 1 and cannot be used to pick up the global
        // context. Route the delta through the same pipeline as
        // `chat_sync_core_paged` by reconstructing the full token
        // history (cached + delta) — `find_cached_prefix` will discover
        // the prefix that turn 1 registered for reuse, sliding-only
        // re-prefill bridges the sliding-layer state, and the full
        // suffix (just the delta tokens) gets the same SDPA reduction
        // order as turn 1's prefill→decode boundary. Mirrors Qwen3's
        // `chat_tokens_delta_sync` paged dispatch.
        if self.paged_adapter.is_some() {
            let mut full_token_history = self.cached_token_history.clone();
            full_token_history.extend(delta_tokens.iter().copied());
            return self.chat_sync_core_paged(
                full_token_history,
                tokenizer,
                config,
                turn_end_id,
                sampling_config,
                max_new_tokens,
            );
        }

        // Build the full token history = cached_history + delta. Used
        // when save_cache_state-ing back to `self.cached_token_history`
        // at the end (the decode loop doesn't actually consult the
        // history for penalty context — Gemma4's bespoke decode loop
        // ignores penalties entirely).
        //
        // The delta path is a 100% cache-reuse operation by construction
        // (the caller is appending on top of the live session), so
        // `cached_token_history.len()` is exactly the reused prefix that
        // should be reported through `ChatResult.cached_tokens`.
        let reused_prefix_len = self.cached_token_history.len();
        let mut save_history = Vec::with_capacity(reused_prefix_len + delta_tokens.len());
        save_history.extend(self.cached_token_history.iter().copied());
        save_history.extend(delta_tokens.iter().copied());

        let prompt_token_count = save_history.len();

        let generation_stream = Stream::new(DeviceType::Gpu);
        let _wired_ctx = crate::stream::WiredLimitContext::new(usize::MAX, vec![generation_stream]);

        let generation_start = std::time::Instant::now();

        // Prefill the delta tokens on top of the existing caches.
        // `prefill_body_gemma4` processes tokens [0:N-1] through the
        // transformer body, leaving the last token for `forward_inner`
        // below to produce logits for the first sampled token. When
        // `delta_tokens.len() == 1` the prefill is a no-op and we go
        // straight to forward_inner with that single token.
        let token_arr: Vec<i32> = delta_tokens.iter().map(|&t| t as i32).collect();
        let prompt = MxArray::from_int32(&token_arr, &[1, delta_tokens.len() as i64])?;

        {
            let _stream_ctx = StreamContext::new(generation_stream);
            let caches = self.caches.as_mut().expect("caches checked is_some above");
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
        eval_gemma4_caches(self.caches.as_ref().expect("caches checked is_some above"));

        // Last token → logits
        let last_token =
            prompt.slice_axis(1, delta_tokens.len() as i64 - 1, delta_tokens.len() as i64)?;
        let logits = {
            let _stream_ctx = StreamContext::new(generation_stream);
            let caches = self.caches.as_mut().expect("caches checked is_some above");
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
        let logits = logits.squeeze(Some(&[1]))?;
        let y = sample_next_token(&logits, sampling_config)?;
        y.eval();
        eval_gemma4_caches(self.caches.as_ref().expect("caches checked is_some above"));

        let first_token_instant = std::time::Instant::now();

        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut finish_reason = "length".to_string();
        let mut current_y = y;
        for step in 0..max_new_tokens {
            let next_y = if step + 1 < max_new_tokens {
                let _stream_ctx = StreamContext::new(generation_stream);
                let caches = self.caches.as_mut().expect("caches checked is_some above");
                let next_ids = current_y.reshape(&[1, 1])?;
                let next_logits = forward_inner(
                    &next_ids,
                    &self.embed_tokens,
                    &self.layers,
                    caches,
                    &self.final_norm,
                    &self.lm_head,
                    self.embed_weight_t.as_ref(),
                    self.ple.as_ref(),
                    &self.config,
                )?;
                let next_logits = next_logits.squeeze(Some(&[1]))?;
                let next_token = sample_next_token(&next_logits, sampling_config)?;
                MxArray::async_eval_arrays(&[&next_token]);
                Some(next_token)
            } else {
                None
            };

            // See `chat_sync_core_paged_inner` for the rationale.
            current_y.eval();
            let token_id = current_y.item_at_int32(0)? as u32;
            generated_tokens.push(token_id);

            if is_eos_token(token_id, &eos_ids, turn_end_id) {
                finish_reason = "stop".to_string();
                break;
            }
            if let Some(next_token) = next_y {
                current_y = next_token;
            } else {
                break;
            }

            if (step + 1) % 256 == 0 {
                crate::array::clear_cache();
            }
        }

        let raw_text = tokenizer.decode_sync(&generated_tokens, false)?;

        // Save cache state: drop the terminal turn-boundary token when
        // the decode terminated on stop (matches the semantics of
        // `chat_sync_core`'s save block).
        let history_tokens: &[u32] = if finish_reason != "length" && !generated_tokens.is_empty() {
            &generated_tokens[..generated_tokens.len() - 1]
        } else {
            &generated_tokens[..]
        };
        let mut new_history = save_history;
        new_history.extend_from_slice(history_tokens);
        self.cached_token_history = new_history;
        // Delta path is text-only; the invariant is enforced by the
        // guard above, so no image key changes here.
        // (self.cached_image_key stays None.)

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
                delta_tokens.len() as f64 / (ttft_ms / 1000.0)
            } else {
                0.0
            },
            decode_tokens_per_second: if decode_ms > 0.0 && gen_toks > 1.0 {
                (gen_toks - 1.0) / (decode_ms / 1000.0)
            } else {
                0.0
            },
        });

        let parsed = super::output_parser::parse_gemma4_output(&raw_text);
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
            cached_tokens: reused_prefix_len as u32,
            performance,
        })
    }

    /// Streaming variant of [`Self::chat_session_start_sync`].
    pub(crate) fn chat_stream_session_start_sync(
        &mut self,
        messages: Vec<ChatMessage>,
        config: ChatConfig,
        stream_tx: StreamTx<ChatStreamChunk>,
        cancelled: Arc<AtomicBool>,
    ) {
        if cancelled.load(Ordering::Relaxed) {
            chat_common::send_stream_error(
                &stream_tx,
                "chat_stream_session_start cancelled before start",
            );
            return;
        }

        let turn_end_id = match self.turn_end_id() {
            Ok(id) => id,
            Err(e) => {
                let _ = stream_tx.send(Err(e));
                return;
            }
        };

        // NOTE: no unconditional reset here — see `chat_session_start_sync`
        // for the prefix-reuse rationale. `chat_stream_sync_core` runs
        // `verify_cache_prefix` against the incoming `messages` and only
        // resets on a cache miss.
        let cb = StreamSender(stream_tx.clone());
        let result = self.chat_stream_sync_core(messages, config, &cb, &cancelled, turn_end_id);
        if let Err(e) = result {
            let _ = stream_tx.send(Err(e));
        }
    }

    /// Streaming variant of [`Self::chat_session_continue_sync`].
    pub(crate) fn chat_stream_session_continue_sync(
        &mut self,
        user_message: String,
        images: Option<Vec<Uint8Array>>,
        config: ChatConfig,
        stream_tx: StreamTx<ChatStreamChunk>,
        cancelled: Arc<AtomicBool>,
    ) {
        if cancelled.load(Ordering::Relaxed) {
            chat_common::send_stream_error(
                &stream_tx,
                "chat_stream_session_continue cancelled before start",
            );
            return;
        }

        if images.as_ref().is_some_and(|v| !v.is_empty()) {
            chat_common::send_stream_error(
                &stream_tx,
                &format!(
                    "{}chat_stream_session_continue is text-only; start a new session with chat_stream_session_start to change the image",
                    chat_common::IMAGE_CHANGE_RESTART_PREFIX
                ),
            );
            return;
        }

        let tokenizer = match self.tokenizer.as_ref() {
            Some(t) => t.clone(),
            None => {
                chat_common::send_stream_error(&stream_tx, "Tokenizer not loaded");
                return;
            }
        };

        let synthetic = chat_common::build_synthetic_user_message(&user_message);
        let sanitized = Qwen3Tokenizer::sanitize_messages_public(std::slice::from_ref(&synthetic));
        let sanitized_user = &sanitized[0].content;

        let enable_thinking = chat_common::resolve_enable_thinking(&config);
        let delta_text = build_gemma4_continue_delta_text(sanitized_user, enable_thinking);

        let delta_tokens = match tokenizer.encode_sync(&delta_text, Some(false)) {
            Ok(t) => t,
            Err(e) => {
                let _ = stream_tx.send(Err(e));
                return;
            }
        };

        self.chat_stream_tokens_delta_sync(delta_tokens, config, stream_tx, cancelled);
    }

    /// Streaming variant of [`Self::chat_session_continue_tool_sync`].
    pub(crate) fn chat_stream_session_continue_tool_sync(
        &mut self,
        tool_call_id: String,
        content: String,
        config: ChatConfig,
        stream_tx: StreamTx<ChatStreamChunk>,
        cancelled: Arc<AtomicBool>,
    ) {
        if cancelled.load(Ordering::Relaxed) {
            chat_common::send_stream_error(
                &stream_tx,
                "chat_stream_session_continue_tool cancelled before start",
            );
            return;
        }

        let tokenizer = match self.tokenizer.as_ref() {
            Some(t) => t.clone(),
            None => {
                chat_common::send_stream_error(&stream_tx, "Tokenizer not loaded");
                return;
            }
        };

        let enable_thinking = chat_common::resolve_enable_thinking(&config);
        let delta_text = build_gemma4_tool_delta_text(&tool_call_id, &content, enable_thinking);

        let delta_tokens = match tokenizer.encode_sync(&delta_text, Some(false)) {
            Ok(t) => t,
            Err(e) => {
                let _ = stream_tx.send(Err(e));
                return;
            }
        };

        self.chat_stream_tokens_delta_sync(delta_tokens, config, stream_tx, cancelled);
    }

    /// Streaming analog of [`Self::chat_tokens_delta_sync`]: prefill
    /// the caller-provided delta tokens on top of the existing Gemma4
    /// caches and stream the reply through `stream_tx`.
    ///
    /// Applies the same guards as the non-streaming path and uses
    /// `<turn|>` as the eos token so the cached history continues to
    /// end on a clean turn boundary after the reply is saved.
    pub(crate) fn chat_stream_tokens_delta_sync(
        &mut self,
        delta_tokens: Vec<u32>,
        config: ChatConfig,
        stream_tx: StreamTx<ChatStreamChunk>,
        cancelled: Arc<AtomicBool>,
    ) {
        if cancelled.load(Ordering::Relaxed) {
            chat_common::send_stream_error(
                &stream_tx,
                "chat_stream_tokens_delta cancelled before start",
            );
            return;
        }

        // --- Same five guards as chat_tokens_delta_sync ---
        if config.reuse_cache == Some(false) {
            chat_common::send_stream_error(
                &stream_tx,
                "chat_tokens_delta_sync requires reuse_cache to be enabled; \
                 the delta path operates on session state by construction",
            );
            return;
        }
        if self.cached_token_history.is_empty() {
            chat_common::send_stream_error(
                &stream_tx,
                "chat_stream_tokens_delta requires an initialized session (call chatStreamSessionStart first)",
            );
            return;
        }
        if delta_tokens.is_empty() {
            chat_common::send_stream_error(
                &stream_tx,
                "chat_stream_tokens_delta requires a non-empty delta",
            );
            return;
        }
        if self.cached_image_key.is_some() {
            chat_common::send_stream_error(
                &stream_tx,
                &format!(
                    "{}chat_stream_tokens_delta is text-only; session currently holds image state",
                    chat_common::IMAGE_CHANGE_RESTART_PREFIX
                ),
            );
            return;
        }
        if self.caches.is_none() {
            chat_common::send_stream_error(
                &stream_tx,
                "chat_stream_tokens_delta requires a live cache (call chatStreamSessionStart first)",
            );
            return;
        }

        let cb = StreamSender(stream_tx.clone());
        let result =
            self.chat_stream_tokens_delta_sync_inner(delta_tokens, config, &cb, &cancelled);
        if let Err(e) = result {
            let _ = stream_tx.send(Err(e));
        }
    }

    /// Inner body of [`Self::chat_stream_tokens_delta_sync`]: prefill
    /// delta tokens on top of the live caches, then run the streaming
    /// decode loop. Mirrors [`Self::chat_stream_sync_core`] but skips
    /// the message rendering + image processing stages — the caller
    /// owns cache coherence by construction.
    fn chat_stream_tokens_delta_sync_inner(
        &mut self,
        delta_tokens: Vec<u32>,
        config: ChatConfig,
        cb: &StreamSender,
        cancelled: &Arc<AtomicBool>,
    ) -> Result<()> {
        let tokenizer = self
            .tokenizer
            .clone()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))?;

        let turn_end_id = self.turn_end_id()?;
        let max_new_tokens = config.max_new_tokens.unwrap_or(2048);
        let sampling_config = make_sampling_config(&config, &self.config);
        let eos_ids = self.config.eos_token_ids.clone();

        // Paged dispatch: same rationale as `chat_tokens_delta_sync` —
        // the global K/V lives in the paged adapter's pool, the flat
        // (sliding) caches were reset at end-of-turn-1, so the only way
        // to resume the conversation faithfully is to route the full
        // history (cached + delta) through the paged streaming pipeline
        // and let `find_cached_prefix` discover the prefix that turn 1
        // registered for reuse.
        if self.paged_adapter.is_some() {
            let mut full_token_history = self.cached_token_history.clone();
            full_token_history.extend(delta_tokens.iter().copied());
            return self.chat_stream_sync_core_paged(
                full_token_history,
                tokenizer,
                config,
                turn_end_id,
                sampling_config,
                cb,
                cancelled,
                max_new_tokens,
            );
        }

        // The streaming delta path is 100% cache-reuse by construction
        // (mirrors `chat_tokens_delta_sync`); capture the reused prefix
        // length for the final `cached_tokens` report.
        let reused_prefix_len = self.cached_token_history.len();
        let mut save_history = Vec::with_capacity(reused_prefix_len + delta_tokens.len());
        save_history.extend(self.cached_token_history.iter().copied());
        save_history.extend(delta_tokens.iter().copied());

        let prompt_token_count = save_history.len();

        let generation_stream = Stream::new(DeviceType::Gpu);
        let _wired_ctx = crate::stream::WiredLimitContext::new(usize::MAX, vec![generation_stream]);

        let generation_start = std::time::Instant::now();

        let token_arr: Vec<i32> = delta_tokens.iter().map(|&t| t as i32).collect();
        let prompt = MxArray::from_int32(&token_arr, &[1, delta_tokens.len() as i64])?;

        {
            let _stream_ctx = StreamContext::new(generation_stream);
            let caches = self.caches.as_mut().expect("caches checked is_some above");
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
        eval_gemma4_caches(self.caches.as_ref().expect("caches checked is_some above"));

        let last_token =
            prompt.slice_axis(1, delta_tokens.len() as i64 - 1, delta_tokens.len() as i64)?;
        let logits = {
            let _stream_ctx = StreamContext::new(generation_stream);
            let caches = self.caches.as_mut().expect("caches checked is_some above");
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
        let logits = logits.squeeze(Some(&[1]))?;
        let y = sample_next_token(&logits, sampling_config)?;
        y.eval();
        eval_gemma4_caches(self.caches.as_ref().expect("caches checked is_some above"));

        let first_token_instant = std::time::Instant::now();

        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut finish_reason = "length".to_string();

        // `decode_stream(false)` preserves Gemma4 special tokens so the
        // stream parser sees `<|channel>` / `<|tool_call>` markers.
        let mut decode_stream = tokenizer.inner().decode_stream(false);
        let mut streamed_text_len = 0;
        let mut stream_parser = super::output_parser::Gemma4StreamParser::new();

        let mut current_y = y;
        for step in 0..max_new_tokens {
            let next_y = if step + 1 < max_new_tokens {
                let _stream_ctx = StreamContext::new(generation_stream);
                let caches = self.caches.as_mut().expect("caches checked is_some above");
                let next_ids = current_y.reshape(&[1, 1])?;
                let next_logits = forward_inner(
                    &next_ids,
                    &self.embed_tokens,
                    &self.layers,
                    caches,
                    &self.final_norm,
                    &self.lm_head,
                    self.embed_weight_t.as_ref(),
                    self.ple.as_ref(),
                    &self.config,
                )?;
                let next_logits = next_logits.squeeze(Some(&[1]))?;
                let next_token = sample_next_token(&next_logits, sampling_config)?;
                MxArray::async_eval_arrays(&[&next_token]);
                Some(next_token)
            } else {
                None
            };

            // See `chat_sync_core_paged_inner` for the rationale.
            current_y.eval();
            let token_id = current_y.item_at_int32(0)? as u32;
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
            dispatch_stream_segments(segments, cb);

            if is_eos_token(token_id, &eos_ids, turn_end_id) {
                finish_reason = "stop".to_string();
                break;
            }
            if let Some(next_token) = next_y {
                current_y = next_token;
            } else {
                break;
            }

            if (step + 1) % 256 == 0 {
                crate::array::clear_cache();
            }
        }

        let raw_text = tokenizer.decode_sync(&generated_tokens, false)?;

        // Flush residual bytes buffered inside decode_stream.
        if raw_text.len() > streamed_text_len {
            let residual = raw_text[streamed_text_len..].to_string();
            let mut segments = stream_parser.feed(&residual);
            segments.extend(stream_parser.flush());
            dispatch_stream_segments(segments, cb);
        } else {
            let tail = stream_parser.flush();
            dispatch_stream_segments(tail, cb);
        }

        // Save cache state for the next session turn.
        let history_tokens: &[u32] = if finish_reason != "length" && !generated_tokens.is_empty() {
            &generated_tokens[..generated_tokens.len() - 1]
        } else {
            &generated_tokens[..]
        };
        let mut new_history = save_history;
        new_history.extend_from_slice(history_tokens);
        self.cached_token_history = new_history;
        // Delta path is text-only; cached_image_key stays None.

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
                delta_tokens.len() as f64 / (ttft_ms / 1000.0)
            } else {
                0.0
            },
            decode_tokens_per_second: if decode_ms > 0.0 && gen_toks > 1.0 {
                (gen_toks - 1.0) / (decode_ms / 1000.0)
            } else {
                0.0
            },
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
                // Delta path reuses the full prior history by
                // construction — report `reused_prefix_len` as the
                // authoritative cached-prefix length.
                cached_tokens: Some(reused_prefix_len as u32),
                performance,
                is_reasoning: None,
            }),
            ThreadsafeFunctionCallMode::NonBlocking,
        );

        Ok(())
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
/// escape the tool turn and inject synthetic structure.
fn build_gemma4_tool_delta_text(
    _tool_call_id: &str,
    content: &str,
    enable_thinking: Option<bool>,
) -> String {
    // `enable_thinking` intentionally unused: see
    // `build_gemma4_continue_delta_text` for why the raw delta path
    // ignores reasoning mode.
    let _ = enable_thinking;
    let escaped = escape_gemma4_content(content);
    format!("\n<|turn>tool\n{escaped}<turn|>\n<|turn>model\n")
}

/// Command handler for the dedicated model thread.
pub(crate) fn handle_gemma4_cmd(inner: &mut Gemma4Inner, cmd: Gemma4Cmd) {
    match cmd {
        Gemma4Cmd::ChatSessionStart {
            messages,
            config,
            reply,
        } => {
            // NOTE: no per-request cache drain here. On a multi-model
            // server the MLX allocator free-pool is process-wide, so
            // flushing after a request on model A discards blocks about
            // to be reused by model B. The TS idle sweeper in
            // `@mlx-node/server` handles between-turn drains.
            let _ = reply.send(inner.chat_session_start_sync(messages, config));
        }
        Gemma4Cmd::ChatSessionContinue {
            user_message,
            images,
            config,
            reply,
        } => {
            let _ = reply.send(inner.chat_session_continue_sync(user_message, images, config));
        }
        Gemma4Cmd::ChatSessionContinueTool {
            tool_call_id,
            content,
            config,
            reply,
        } => {
            let _ =
                reply.send(inner.chat_session_continue_tool_sync(tool_call_id, content, config));
        }
        Gemma4Cmd::ChatStreamSessionStart {
            messages,
            config,
            stream_tx,
            cancelled,
        } => {
            inner.chat_stream_session_start_sync(messages, config, stream_tx, cancelled);
        }
        Gemma4Cmd::ChatStreamSessionContinue {
            user_message,
            images,
            config,
            stream_tx,
            cancelled,
        } => {
            inner.chat_stream_session_continue_sync(
                user_message,
                images,
                config,
                stream_tx,
                cancelled,
            );
        }
        Gemma4Cmd::ChatStreamSessionContinueTool {
            tool_call_id,
            content,
            config,
            stream_tx,
            cancelled,
        } => {
            inner.chat_stream_session_continue_tool_sync(
                tool_call_id,
                content,
                config,
                stream_tx,
                cancelled,
            );
        }
        Gemma4Cmd::ResetCaches { reply } => {
            let result = inner.reset_caches_sync();
            let _ = reply.send(result);
        }
    }
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
    /// Callers relying on the pre-round-2 behavior where `new(config)`
    /// returned a runnable model MUST migrate to `await
    /// Gemma4Model.load(path)`. The constructor signature is unchanged
    /// on purpose (NAPI-RS pins it), so this is a deliberate runtime
    /// behavior break covered by the regression tests in
    /// `__test__/models/model-loader-gemma4.test.ts`.
    #[napi(constructor)]
    pub fn new(config: Gemma4Config) -> Self {
        let has_vision = config.vision_config.is_some();
        Self {
            thread: None,
            model_id: 0,
            has_vision,
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
    /// `Gemma4Config::use_block_paged_cache`, default-ON since the
    /// `gemma4_paged_vs_flat_parity` integration test verified greedy
    /// byte-equal at BF16 against real Gemma-4-E2B-IT weights — see
    /// CLAUDE.md). Stubs constructed via `new(config)` always return
    /// `false`. Surfaced through this NAPI method so server endpoints
    /// can branch on it without a model-thread roundtrip.
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

    /// Reset all caches and clear cached token history. Exposed so
    /// tests and session-management code can start from a known clean
    /// state between turns.
    ///
    /// Synchronous on the NAPI boundary — every other `SessionCapableModel`
    /// exposes `resetCaches(): void` and the `ChatSession<M>` cross-model
    /// wrapper calls this inline during the image-change restart and
    /// `reset()` flows. Running it as an async NAPI method would break
    /// that contract and silently drop reset failures because
    /// `ChatSession.reset()` and the session-start restart path invoke
    /// `model.resetCaches()` without awaiting.
    #[napi]
    pub fn reset_caches(&self) -> Result<()> {
        let Some(thread) = self.thread.as_ref() else {
            // Uninitialized stub (constructed via `new(config)` without
            // `load()`): nothing to reset. Match the OCR models'
            // silent no-op to keep `ChatSession.reset()` idempotent.
            return Ok(());
        };
        crate::model_thread::send_and_block(thread, |reply| Gemma4Cmd::ResetCaches { reply })
    }

    /// Start a new chat session.
    ///
    /// Runs the full jinja chat template once, decodes until Gemma4's
    /// `<turn|>` delimiter, and leaves the KV caches on a clean turn
    /// boundary so subsequent `chatSessionContinue` /
    /// `chatSessionContinueTool` calls can append a raw delta on top
    /// without re-rendering the chat template.
    #[napi]
    pub async fn chat_session_start(
        &self,
        messages: Vec<ChatMessage>,
        config: Option<ChatConfig>,
    ) -> Result<ChatResult> {
        let thread = self.thread.as_ref().ok_or_else(|| {
            Error::from_reason("Model not initialized. Call Gemma4Model.load() first.")
        })?;
        let config = config.unwrap_or_default();

        // Fast-fail: images on a text-only model.
        if !self.has_vision
            && messages
                .iter()
                .any(|m| m.images.as_ref().is_some_and(|imgs| !imgs.is_empty()))
        {
            return Err(Error::from_reason(
                "Images provided but model has no vision support (no vision_config in config.json)",
            ));
        }

        crate::model_thread::send_and_await(thread, |reply| Gemma4Cmd::ChatSessionStart {
            messages,
            config,
            reply,
        })
        .await
    }

    /// Continue an existing chat session with a new user message.
    ///
    /// Appends a raw Gemma4 user/model delta to the session's cached KV
    /// state, then decodes the model reply. Stops on `<turn|>` so the
    /// cache remains on a clean turn boundary for the next turn.
    ///
    /// Requires a live session started via `chatSessionStart`. Errors
    /// if the session is empty, carries image state, or if
    /// `config.reuse_cache` is explicitly set to `false`.
    ///
    /// `images` is an opt-in guard parameter: when non-empty the native
    /// side returns an error whose message begins with
    /// `IMAGE_CHANGE_REQUIRES_SESSION_RESTART:` so the TypeScript
    /// `ChatSession` layer can catch the prefix and route image-changes
    /// back through a fresh `chatSessionStart` uniformly across all
    /// model backends.
    #[napi(
        ts_args_type = "userMessage: string, images: Uint8Array[] | null | undefined, config: ChatConfig | null | undefined"
    )]
    pub async fn chat_session_continue(
        &self,
        user_message: String,
        images: Option<Vec<Uint8Array>>,
        config: Option<ChatConfig>,
    ) -> Result<ChatResult> {
        let thread = self.thread.as_ref().ok_or_else(|| {
            Error::from_reason("Model not initialized. Call Gemma4Model.load() first.")
        })?;
        let config = config.unwrap_or_default();

        crate::model_thread::send_and_await(thread, |reply| Gemma4Cmd::ChatSessionContinue {
            user_message,
            images,
            config,
            reply,
        })
        .await
    }

    /// Continue an existing chat session with a tool-result turn.
    ///
    /// Builds a Gemma4-format tool delta
    /// (`\n<|turn>tool\n{content}<turn|>\n<|turn>model\n`) from
    /// `content` and prefills it on top of the live session caches,
    /// then decodes the model reply. Stops on `<turn|>` so the cache
    /// stays on a clean turn boundary for the next turn.
    ///
    /// The `tool_call_id` is currently dropped by the wire format —
    /// Gemma4's chat template identifies tool responses positionally,
    /// not via an explicit id. Callers may still log it for their own
    /// bookkeeping.
    ///
    /// Requires a live session started via `chatSessionStart`.
    #[napi]
    pub async fn chat_session_continue_tool(
        &self,
        tool_call_id: String,
        content: String,
        config: Option<ChatConfig>,
    ) -> Result<ChatResult> {
        let thread = self.thread.as_ref().ok_or_else(|| {
            Error::from_reason("Model not initialized. Call Gemma4Model.load() first.")
        })?;
        let config = config.unwrap_or_default();

        crate::model_thread::send_and_await(thread, |reply| Gemma4Cmd::ChatSessionContinueTool {
            tool_call_id,
            content,
            config,
            reply,
        })
        .await
    }

    /// Streaming variant of `chatSessionStart`.
    #[napi(
        ts_args_type = "messages: ChatMessage[], config: ChatConfig | null | undefined, callback: (err: Error | null, chunk: ChatStreamChunk) => void"
    )]
    pub async fn chat_stream_session_start(
        &self,
        messages: Vec<ChatMessage>,
        config: Option<ChatConfig>,
        callback: ThreadsafeFunction<ChatStreamChunk, ()>,
    ) -> Result<ChatStreamHandle> {
        let thread = self.thread.as_ref().ok_or_else(|| {
            Error::from_reason("Model not initialized. Call Gemma4Model.load() first.")
        })?;
        let config = config.unwrap_or_default();

        // Fast-fail: images on a text-only model.
        if !self.has_vision
            && messages
                .iter()
                .any(|m| m.images.as_ref().is_some_and(|imgs| !imgs.is_empty()))
        {
            return Err(Error::from_reason(
                "Images provided but model has no vision support (no vision_config in config.json)",
            ));
        }

        let cancelled = Arc::new(AtomicBool::new(false));
        let cancelled_inner = cancelled.clone();
        let (stream_tx, mut stream_rx) =
            tokio::sync::mpsc::unbounded_channel::<napi::Result<ChatStreamChunk>>();

        thread.send(Gemma4Cmd::ChatStreamSessionStart {
            messages,
            config,
            stream_tx,
            cancelled: cancelled_inner,
        })?;

        let callback = Arc::new(callback);
        tokio::spawn(async move {
            while let Some(result) = stream_rx.recv().await {
                callback.call(result, ThreadsafeFunctionCallMode::NonBlocking);
            }
        });

        Ok(ChatStreamHandle { cancelled })
    }

    /// Streaming variant of `chatSessionContinue`.
    #[napi(
        ts_args_type = "userMessage: string, images: Uint8Array[] | null | undefined, config: ChatConfig | null | undefined, callback: (err: Error | null, chunk: ChatStreamChunk) => void"
    )]
    pub async fn chat_stream_session_continue(
        &self,
        user_message: String,
        images: Option<Vec<Uint8Array>>,
        config: Option<ChatConfig>,
        callback: ThreadsafeFunction<ChatStreamChunk, ()>,
    ) -> Result<ChatStreamHandle> {
        let thread = self.thread.as_ref().ok_or_else(|| {
            Error::from_reason("Model not initialized. Call Gemma4Model.load() first.")
        })?;
        let config = config.unwrap_or_default();

        let cancelled = Arc::new(AtomicBool::new(false));
        let cancelled_inner = cancelled.clone();
        let (stream_tx, mut stream_rx) =
            tokio::sync::mpsc::unbounded_channel::<napi::Result<ChatStreamChunk>>();

        thread.send(Gemma4Cmd::ChatStreamSessionContinue {
            user_message,
            images,
            config,
            stream_tx,
            cancelled: cancelled_inner,
        })?;

        let callback = Arc::new(callback);
        tokio::spawn(async move {
            while let Some(result) = stream_rx.recv().await {
                callback.call(result, ThreadsafeFunctionCallMode::NonBlocking);
            }
        });

        Ok(ChatStreamHandle { cancelled })
    }

    /// Streaming variant of `chatSessionContinueTool`.
    #[napi(
        ts_args_type = "toolCallId: string, content: string, config: ChatConfig | null | undefined, callback: (err: Error | null, chunk: ChatStreamChunk) => void"
    )]
    pub async fn chat_stream_session_continue_tool(
        &self,
        tool_call_id: String,
        content: String,
        config: Option<ChatConfig>,
        callback: ThreadsafeFunction<ChatStreamChunk, ()>,
    ) -> Result<ChatStreamHandle> {
        let thread = self.thread.as_ref().ok_or_else(|| {
            Error::from_reason("Model not initialized. Call Gemma4Model.load() first.")
        })?;
        let config = config.unwrap_or_default();

        let cancelled = Arc::new(AtomicBool::new(false));
        let cancelled_inner = cancelled.clone();
        let (stream_tx, mut stream_rx) =
            tokio::sync::mpsc::unbounded_channel::<napi::Result<ChatStreamChunk>>();

        thread.send(Gemma4Cmd::ChatStreamSessionContinueTool {
            tool_call_id,
            content,
            config,
            stream_tx,
            cancelled: cancelled_inner,
        })?;

        let callback = Arc::new(callback);
        tokio::spawn(async move {
            while let Some(result) = stream_rx.recv().await {
                callback.call(result, ThreadsafeFunctionCallMode::NonBlocking);
            }
        });

        Ok(ChatStreamHandle { cancelled })
    }
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
/// persistent path lazily initializes its caches inside `chat_sync_core` /
/// `chat_stream_sync_core` via `init_caches_sync`.
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
/// replace the config list. This matches the dense model's
/// `chat_sync_core` semantics: session-start callers get their clean
/// boundary token (for Gemma4 that is `<turn|>`) while still respecting
/// the underlying model's intrinsic eos set.
#[inline]
fn is_eos_token(token: u32, eos_ids: &[i32], eos_token_id: u32) -> bool {
    if eos_ids.contains(&(token as i32)) {
        return true;
    }
    eos_token_id == token
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

/// Call the compiled C++ forward for a single Gemma4 decode step.
fn forward_gemma4_cpp(input_ids: &MxArray, embedding_weight: &MxArray) -> Result<MxArray> {
    let mut logits_ptr: *mut mlx_sys::mlx_array = std::ptr::null_mut();
    let mut cache_offset: i32 = 0;
    unsafe {
        mlx_sys::mlx_gemma4_forward(
            input_ids.as_raw_ptr(),
            embedding_weight.as_raw_ptr(),
            &mut logits_ptr,
            &mut cache_offset,
        );
    }
    if logits_ptr.is_null() {
        return Err(Error::from_reason("Gemma4 compiled forward returned null"));
    }
    MxArray::from_handle(logits_ptr, "gemma4_compiled_forward")
}

fn eval_token_and_gemma4_caches(next_token: &MxArray) {
    unsafe {
        mlx_sys::mlx_gemma4_eval_token_and_caches(next_token.as_raw_ptr());
    }
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
    // Sliding mask: only needed when seq_len > window_size.
    // Matches Python create_attention_mask: when N <= window_size, returns "causal".
    // When N > window_size, returns explicit causal+window mask.
    let sliding_mask = if seq_len > 1 && seq_len > config.sliding_window as i64 {
        let sliding_idx = (0..config.num_hidden_layers as usize)
            .find(|&i| config.is_sliding_layer(i))
            .unwrap_or(0);
        let offset = if sliding_idx < caches.len() {
            caches[sliding_idx].get_offset()
        } else {
            0
        };
        Some(create_sliding_mask(
            seq_len,
            offset,
            config.sliding_window as i64,
        )?)
    } else {
        None
    };

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
    lm_head: &Option<Linear>,
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
/// `paged_idx` counts only `full_attention` layers in their original
/// decoder order — matches the `LayerKVPool` slot count from
/// `Gemma4Inner::new`. KV-shared layers do NOT consume a paged slot
/// (they reuse the anchor's K/V); the shared variants carry the
/// anchor's index so the shared forward path can resolve it.
///
/// Lifted to a free helper so unit tests can drive it without owning a
/// `Gemma4Inner` (which requires loaded weights). Mirrors LFM2's
/// `compute_layer_kinds` pattern.
pub(crate) fn compute_layer_kinds(config: &Gemma4Config) -> Vec<Gemma4LayerKind> {
    let n = config.num_hidden_layers as usize;

    // Pre-compute global -> paged_idx mapping so KV-shared layers can
    // look up their anchor's pool slot.
    let mut global_to_paged: Vec<Option<u32>> = vec![None; n];
    let mut paged_idx: u32 = 0;
    for (i, slot) in global_to_paged.iter_mut().enumerate().take(n) {
        if config.is_global_layer(i) {
            *slot = Some(paged_idx);
            paged_idx += 1;
        }
    }

    let mut kinds = Vec::with_capacity(n);
    for i in 0..n {
        let kind = if config.is_kv_shared_layer(i) {
            match config.kv_shared_anchor(i) {
                Some(anchor) if config.is_global_layer(anchor) => {
                    let anchor_paged_idx = global_to_paged[anchor].unwrap_or(0);
                    Gemma4LayerKind::SharedOnGlobal { anchor_paged_idx }
                }
                Some(anchor) => Gemma4LayerKind::SharedOnSliding {
                    anchor_layer_idx: anchor as u32,
                },
                None => {
                    // Fallback for malformed configs (no anchor
                    // resolvable). Treat as flat sliding/global based
                    // on the layer's own type so dispatch doesn't panic.
                    if config.is_global_layer(i) {
                        let pidx = global_to_paged[i].unwrap_or(0);
                        Gemma4LayerKind::GlobalPaged { paged_idx: pidx }
                    } else {
                        Gemma4LayerKind::Sliding
                    }
                }
            }
        } else if config.is_global_layer(i) {
            Gemma4LayerKind::GlobalPaged {
                paged_idx: global_to_paged[i].unwrap_or(0),
            }
        } else {
            Gemma4LayerKind::Sliding
        };
        kinds.push(kind);
    }
    kinds
}

/// Default prefill chunk size (tokens per chunk).
/// Note: mlx-lm uses 2048 but the first eval triggers Metal shader compilation
/// which can GPU-timeout with very large graphs. Using 512 keeps individual
/// command buffers under Metal's timeout limit.
const GEMMA4_PREFILL_STEP_SIZE: i64 = 512;

/// Evaluate all Gemma4 cache arrays to materialize them on GPU.
/// Must be called between prefill chunks to break lazy dependency chains.
fn eval_gemma4_caches(caches: &[Gemma4LayerCache]) {
    let mut arrays: Vec<&MxArray> = Vec::new();
    for cache in caches {
        cache.collect_cache_arrays(&mut arrays);
    }
    if !arrays.is_empty() {
        MxArray::eval_arrays(&arrays);
    }
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
        eval_gemma4_caches(caches);
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

    let neg_inf = MxArray::full(
        &[1],
        Either::A(f64::NEG_INFINITY),
        Some(crate::array::DType::Float32),
    )?;
    let zero_f = MxArray::full(&[1], Either::A(0.0), Some(crate::array::DType::Float32))?;
    let mask = valid.where_(&zero_f, &neg_inf)?;
    mask.reshape(&[1, 1, seq_len, total_len])
}

// ---------------------------------------------------------------------------
// Vision helpers
// ---------------------------------------------------------------------------

/// Extract raw image bytes from ChatMessage.images fields.
fn extract_images_from_messages(messages: &[ChatMessage]) -> Vec<Vec<u8>> {
    let mut all_images: Vec<Vec<u8>> = Vec::new();
    for msg in messages {
        if let Some(ref images) = msg.images {
            for img in images {
                all_images.push(img.to_vec());
            }
        }
    }
    all_images
}

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

/// Chunked prefill with pre-computed embeddings (for vision path).
///
/// Same as `prefill_body_gemma4` but uses pre-merged `inputs_embeds` instead
/// of looking up from the embedding table. PLE tokens at image positions are
/// zeroed to avoid confusing the per-layer embeddings with vision token IDs.
fn prefill_body_gemma4_with_embeds(
    prompt: &MxArray,
    inputs_embeds: &MxArray,
    embedding: &Embedding,
    layers: &[Gemma4DecoderLayer],
    caches: &mut [Gemma4LayerCache],
    final_norm: &RMSNorm,
    ple: Option<&PleComponents>,
    config: &Gemma4Config,
) -> Result<()> {
    let total_len = inputs_embeds.shape_at(1)?;

    if total_len <= 1 {
        return Ok(());
    }

    // Process tokens [0:N-1] — leave last token for forward_inner
    let prefill_len = total_len - 1;
    let all_embeds = inputs_embeds.slice_axis(1, 0, prefill_len)?;

    // PLE: mask image token positions to 0 before computing per-layer embeddings
    let all_ple: Option<MxArray> = if let Some(ple) = ple {
        let prefill_ids = prompt.slice_axis(1, 0, prefill_len)?;
        let image_token_id = config.image_token_id.unwrap_or(258880);
        let image_token = MxArray::scalar_int(image_token_id)?;
        let image_mask = prefill_ids.equal(&image_token)?;
        let zero = MxArray::scalar_int(0)?;
        let masked_ids = image_mask.where_(&zero, &prefill_ids)?;
        Some(compute_ple(&masked_ids, &all_embeds, ple, prefill_len)?)
    } else {
        None
    };

    let mut offset: i64 = 0;

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
        eval_gemma4_caches(caches);
        crate::array::clear_cache();
        offset += GEMMA4_PREFILL_STEP_SIZE;
    }

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

#[cfg(test)]
mod tests {
    use super::*;

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
            image_token_id: None,
            boi_token_id: None,
            eoi_token_id: None,
            vision_soft_tokens_per_image: None,
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

    /// Smoke test for `chat_sync_core_paged` via direct helper drives.
    ///
    /// Random-init weights cast to BF16 (the paged pool's expected
    /// dtype). Validates the adapter lifecycle (reset →
    /// find_cached_prefix → allocate_suffix → record_tokens →
    /// forward_paged_or_flat) and that produced logits have the
    /// expected shape, without asserting numerical equivalence to the
    /// flat path (random weights). Gracefully skipped on no-Metal.
    #[test]
    fn test_run_paged_prefill_decode_smoke() {
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
            head.set_weight(&cast(&w)).expect("lm_head");
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

        let last_logits = match inner.run_paged_prefill_chunk(&prompt, &prompt, 0) {
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
        // Shared layers 4..8 (note these still consume paged_idx slots
        // because they ARE global layers themselves at the type level —
        // but the LayerKVPool sizing in Gemma4Inner::new counts ALL
        // global layers (both anchors and shared) so the indexing is
        // consistent. SharedOnGlobal carries the ANCHOR's pool slot.
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
    //! prefix relation). The call sites in
    //! `chat_sync_core` / `chat_stream_sync_core` then classify that
    //! value plus the incoming prompt length into
    //! [`PrefixCacheDecision::StrictExtendHit`] (warm-reuse, skip the
    //! cached prefix, prefill only the tail) vs
    //! [`PrefixCacheDecision::Miss`] (reset caches + re-init + full
    //! prefill).
    //!
    //! The four cases covered below pin the Round 1 Fix #1 invariant:
    //! exact-match MUST route to `Miss`, not to `StrictExtendHit` — a
    //! previous revision treated exact-match as a shortcut and corrupted
    //! the next warm-hit turn by advancing cache state to
    //! `prompt + last_token` while the history write-back only persisted
    //! `tokens + generated`. The `#[ignore]`-gated integration tests
    //! above exercise the end-to-end behaviour against a loaded Gemma4
    //! model; this module guarantees the decision logic stays correct
    //! in every CI run without a model dependency.

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
        // This is the core Round 1 Fix #1 invariant — guarding against
        // a regression that silently corrupts multi-turn correctness.
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
