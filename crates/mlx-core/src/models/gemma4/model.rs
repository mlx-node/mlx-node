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
use super::decoder_layer::Gemma4DecoderLayer;
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
    pub(crate) thread: crate::model_thread::ModelThread<Gemma4Cmd>,
    pub(crate) model_id: u64,
    /// Whether the loaded config includes `vision_config`. Mirrored here so
    /// the NAPI side can fail fast on image inputs to a text-only model
    /// without round-tripping to the model thread. The actual image
    /// processor lives on `Gemma4Inner` and runs on the model thread.
    pub(crate) has_vision: bool,
}

static MODEL_ID_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Serializes compiled C++ forward calls across model instances.
/// Only matters if two Gemma4 models are loaded simultaneously (rare).
static COMPILED_FORWARD_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

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

        // Create prompt tensor
        let token_arr: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let prompt = MxArray::from_int32(&token_arr, &[1, tokens.len() as i64])?;

        // Lazily initialize the persistent KV caches on the first turn.
        // Subsequent turns reuse `self.caches` in place. Step 5c wires
        // the session-reset and prefix-verification paths on top of this.
        if self.caches.is_none() {
            self.init_caches_sync()?;
        }

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

        // Last token → logits
        let last_token = prompt.slice_axis(1, tokens.len() as i64 - 1, tokens.len() as i64)?;
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

        // Decode text
        let text = tokenizer.decode_sync(&generated_tokens, true)?;

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

        Ok(ChatResult {
            text: text.clone(),
            tool_calls: vec![],
            thinking: None,
            num_tokens: generated_tokens.len() as u32,
            prompt_tokens: prompt_token_count as u32,
            reasoning_tokens: 0,
            finish_reason,
            raw_text: text,
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

        let token_arr: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let prompt = MxArray::from_int32(&token_arr, &[1, tokens.len() as i64])?;

        // Lazily initialize the persistent KV caches on the first turn.
        // Subsequent turns reuse `self.caches` in place.
        if self.caches.is_none() {
            self.init_caches_sync()?;
        }

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

        let last_token = prompt.slice_axis(1, tokens.len() as i64 - 1, tokens.len() as i64)?;
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

        let mut decode_stream = tokenizer.inner().decode_stream(true);
        let mut streamed_text_len = 0;

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

                cb.call(
                    Ok(ChatStreamChunk {
                        text: token_text,
                        done: false,
                        finish_reason: None,
                        tool_calls: None,
                        thinking: None,
                        num_tokens: None,
                        prompt_tokens: None,
                        reasoning_tokens: None,
                        raw_text: None,
                        performance: None,
                        is_reasoning: None,
                    }),
                    ThreadsafeFunctionCallMode::NonBlocking,
                );

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

                cb.call(
                    Ok(ChatStreamChunk {
                        text: token_text,
                        done: false,
                        finish_reason: None,
                        tool_calls: None,
                        thinking: None,
                        num_tokens: None,
                        prompt_tokens: None,
                        reasoning_tokens: None,
                        raw_text: None,
                        performance: None,
                        is_reasoning: None,
                    }),
                    ThreadsafeFunctionCallMode::NonBlocking,
                );

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

        let text = tokenizer.decode_sync(&generated_tokens, true)?;

        // Flush any residual bytes that might not have resolved at the streaming layer
        if text.len() > streamed_text_len {
            let residual = text[streamed_text_len..].to_string();
            cb.call(
                Ok(ChatStreamChunk {
                    text: residual,
                    done: false,
                    finish_reason: None,
                    tool_calls: None,
                    thinking: None,
                    num_tokens: None,
                    prompt_tokens: None,
                    reasoning_tokens: None,
                    raw_text: None,
                    performance: None,
                    is_reasoning: None,
                }),
                ThreadsafeFunctionCallMode::NonBlocking,
            );
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

        // Emit final block
        cb.call(
            Ok(ChatStreamChunk {
                text: String::new(),
                done: true,
                finish_reason: Some(finish_reason),
                tool_calls: Some(vec![]),
                thinking: None,
                num_tokens: Some(generated_tokens.len() as u32),
                prompt_tokens: Some(prompt_token_count as u32),
                reasoning_tokens: Some(0),
                raw_text: Some(text),
                performance,
                is_reasoning: None,
            }),
            ThreadsafeFunctionCallMode::NonBlocking,
        );

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

        // Full reset: the session-start path always begins from a clean
        // state. This matches the documented contract that the session
        // is owned end-to-end by the `chat_session_*` surface and
        // intentionally invalidates any prior cache.
        self.reset_caches_sync()?;

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

        // Build the full token history = cached_history + delta. Used
        // when save_cache_state-ing back to `self.cached_token_history`
        // at the end (the decode loop doesn't actually consult the
        // history for penalty context — Gemma4's bespoke decode loop
        // ignores penalties entirely).
        let mut save_history =
            Vec::with_capacity(self.cached_token_history.len() + delta_tokens.len());
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

        let text = tokenizer.decode_sync(&generated_tokens, true)?;

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

        Ok(ChatResult {
            text: text.clone(),
            tool_calls: vec![],
            thinking: None,
            num_tokens: generated_tokens.len() as u32,
            prompt_tokens: prompt_token_count as u32,
            reasoning_tokens: 0,
            finish_reason,
            raw_text: text,
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

        // Full reset: the session-start path always begins clean.
        if let Err(e) = self.reset_caches_sync() {
            let _ = stream_tx.send(Err(e));
            return;
        }

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

        let mut save_history =
            Vec::with_capacity(self.cached_token_history.len() + delta_tokens.len());
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

        let mut decode_stream = tokenizer.inner().decode_stream(true);
        let mut streamed_text_len = 0;

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

            cb.call(
                Ok(ChatStreamChunk {
                    text: token_text,
                    done: false,
                    finish_reason: None,
                    tool_calls: None,
                    thinking: None,
                    num_tokens: None,
                    prompt_tokens: None,
                    reasoning_tokens: None,
                    raw_text: None,
                    performance: None,
                    is_reasoning: None,
                }),
                ThreadsafeFunctionCallMode::NonBlocking,
            );

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

        let text = tokenizer.decode_sync(&generated_tokens, true)?;

        // Flush residual bytes buffered inside decode_stream.
        if text.len() > streamed_text_len {
            let residual = text[streamed_text_len..].to_string();
            cb.call(
                Ok(ChatStreamChunk {
                    text: residual,
                    done: false,
                    finish_reason: None,
                    tool_calls: None,
                    thinking: None,
                    num_tokens: None,
                    prompt_tokens: None,
                    reasoning_tokens: None,
                    raw_text: None,
                    performance: None,
                    is_reasoning: None,
                }),
                ThreadsafeFunctionCallMode::NonBlocking,
            );
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

        cb.call(
            Ok(ChatStreamChunk {
                text: String::new(),
                done: true,
                finish_reason: Some(finish_reason),
                tool_calls: Some(vec![]),
                thinking: None,
                num_tokens: Some(generated_tokens.len() as u32),
                prompt_tokens: Some(prompt_token_count as u32),
                reasoning_tokens: Some(0),
                raw_text: Some(text),
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
    #[napi(constructor)]
    pub fn new(config: Gemma4Config) -> Result<Self> {
        let has_vision = config.vision_config.is_some();

        let (thread, init_rx) = crate::model_thread::ModelThread::spawn_with_init(
            move || {
                let inner = Gemma4Inner::new(config)?;
                let model_id = inner.model_id;
                Ok((inner, model_id))
            },
            handle_gemma4_cmd,
        );

        let model_id = init_rx
            .blocking_recv()
            .map_err(|_| napi::Error::from_reason("Model thread exited during init"))??;

        Ok(Self {
            thread,
            model_id,
            has_vision,
        })
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
        crate::model_thread::send_and_block(&self.thread, |reply| Gemma4Cmd::ResetCaches { reply })
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

        crate::model_thread::send_and_await(&self.thread, |reply| Gemma4Cmd::ChatSessionStart {
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
        let config = config.unwrap_or_default();

        crate::model_thread::send_and_await(&self.thread, |reply| Gemma4Cmd::ChatSessionContinue {
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
        let config = config.unwrap_or_default();

        crate::model_thread::send_and_await(&self.thread, |reply| {
            Gemma4Cmd::ChatSessionContinueTool {
                tool_call_id,
                content,
                config,
                reply,
            }
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

        self.thread.send(Gemma4Cmd::ChatStreamSessionStart {
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
        let config = config.unwrap_or_default();

        let cancelled = Arc::new(AtomicBool::new(false));
        let cancelled_inner = cancelled.clone();
        let (stream_tx, mut stream_rx) =
            tokio::sync::mpsc::unbounded_channel::<napi::Result<ChatStreamChunk>>();

        self.thread.send(Gemma4Cmd::ChatStreamSessionContinue {
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
        let config = config.unwrap_or_default();

        let cancelled = Arc::new(AtomicBool::new(false));
        let cancelled_inner = cancelled.clone();
        let (stream_tx, mut stream_rx) =
            tokio::sync::mpsc::unbounded_channel::<napi::Result<ChatStreamChunk>>();

        self.thread.send(Gemma4Cmd::ChatStreamSessionContinueTool {
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

    for (i, layer) in layers.iter().enumerate() {
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

    // Logit softcapping — compiled fused kernel (matches Python's mx.compile logit_softcap)
    if let Some(cap) = config.final_logit_softcapping {
        let cap_arr = MxArray::scalar_float_like(cap, &logits)?;
        let handle = unsafe { mlx_sys::mlx_logit_softcap(logits.handle.0, cap_arr.handle.0) };
        Ok(MxArray::from_handle(handle, "logit_softcap")?)
    } else {
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
}
