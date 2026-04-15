use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi_derive::napi;
use tracing::{info, warn};

use crate::array::MxArray;
use crate::model_thread::{ResponseTx, StreamTx};
use crate::nn::{Embedding, Linear, RMSNorm};
use crate::sampling::{SamplingConfig, sample};
use crate::stream::{DeviceType, Stream, StreamContext};
use crate::tokenizer::{ChatMessage, Qwen3Tokenizer, ToolDefinition};
use crate::tools::ToolCallResult;

use super::chat_common;
use super::chat_common::{
    apply_all_penalties, build_chatml_continue_delta_text, build_synthetic_user_message,
    compute_image_cache_key, compute_performance_metrics, extract_chat_params,
    finalize_chat_result, save_cache_state_direct, send_stream_error, verify_cache_prefix_direct,
};
use super::config::Qwen3_5Config;
use super::decoder_layer::DecoderLayer;
use super::layer_cache::Qwen3_5LayerCache;
use super::persistence;
use super::processing::Qwen35VLImageProcessor;
use super::vision::Qwen3_5VisionEncoder;
use crate::models::paddleocr_vl::processing::ProcessedImages;

/// Maximum number of entries in the vision encoder cache before LRU eviction.
pub(crate) const VISION_CACHE_MAX_ENTRIES: usize = 32;

/// LRU cache for vision encoder embeddings, keyed by image content hash.
pub(crate) struct VisionCacheInner {
    pub entries: HashMap<u64, (MxArray, MxArray, u64)>,
    /// Monotonically increasing counter for LRU generation tracking.
    pub generation: u64,
}

pub(crate) type VisionCache = Arc<Mutex<VisionCacheInner>>;

/// Monotonically incrementing counter for assigning unique model IDs.
/// Shared by BOTH dense and MoE models — the C++ weight map is shared,
/// so IDs must be globally unique across all Qwen3.5 model variants.
pub(crate) static QWEN35_MODEL_ID_COUNTER: AtomicU64 = AtomicU64::new(1); // 0 = no model

/// RwLock protecting the C++ global weight map against concurrent mutation.
/// Write-locked during weight registration (model load), read-locked during
/// compiled inference. This prevents a concurrent model load from swapping
/// weights underneath an in-flight compiled decode, and eliminates the TOCTOU
/// between has_weight() / get_weight() in linear_proj().
pub(crate) static COMPILED_WEIGHTS_RWLOCK: std::sync::RwLock<()> = std::sync::RwLock::new(());

/// Acquire the compiled weight read lock and verify model ownership in one step.
/// Returns `Some(guard)` if this model owns the compiled weights, `None` otherwise.
/// The guard must be held for the lifetime of the compiled decode to prevent
/// Process-wide mutex serializing the dense compiled forward lifecycle across
/// model instances. Within a single model instance, the dedicated model thread
/// serializes calls. But with multiple model instances, compiled C++ forward
/// calls from different model threads can collide on process-wide globals.
static DENSE_COMPILED_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Internal model state owned exclusively by the dedicated model thread.
///
/// No `Arc<RwLock<>>` — the model thread has sole ownership of all inference
/// and training state. Training commands are routed via `TrainingDispatch`.
pub(crate) struct Qwen35Inner {
    pub(crate) config: Qwen3_5Config,
    pub(crate) embedding: Embedding,
    pub(crate) layers: Vec<DecoderLayer>,
    pub(crate) final_norm: RMSNorm,
    pub(crate) lm_head: Option<Linear>,
    pub(crate) caches: Option<Vec<Qwen3_5LayerCache>>,
    pub(crate) tokenizer: Option<Arc<Qwen3Tokenizer>>,
    pub(crate) vision_encoder: Option<Arc<Qwen3_5VisionEncoder>>,
    pub(crate) image_processor: Option<Arc<Qwen35VLImageProcessor>>,
    pub(crate) spatial_merge_size: Option<i32>,
    pub(crate) vision_cache: VisionCache,
    pub(crate) cached_token_history: Vec<u32>,
    pub(crate) cached_image_key: Option<u64>,
    pub(crate) cached_rope_deltas: Option<i32>,
    pub(crate) model_id: u64,
    /// Training state owned by the model thread.
    /// Created when `InitTraining` command is received, destroyed when training ends.
    pub(crate) training_state: Option<crate::training_state::ModelThreadTrainingState>,
}

/// Commands dispatched from NAPI methods to the dedicated model thread.
pub(crate) enum Qwen35Cmd {
    /// Session-based chat continuation: prefill a pre-tokenized delta on top
    /// of the existing KV caches, then decode. Text-only; requires an active
    /// session (prior `ChatSessionStart` call that initialized `self.caches`).
    ///
    /// This bypasses the jinja chat template entirely — the caller is
    /// responsible for producing the correctly-formatted delta tokens
    /// (typically `\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`).
    ///
    /// Constructed internally by `chat_session_continue_sync` after building
    /// and tokenizing the delta. Not currently wired through a NAPI method
    /// directly — external callers use `ChatSessionContinue` instead, which
    /// handles delta construction on the model thread. Kept as its own
    /// variant so the lower-level pre-tokenized entry point stays exposed
    /// for the gated integration test and future advanced use cases.
    #[allow(dead_code)]
    ChatTokensDelta {
        delta_tokens: Vec<u32>,
        config: ChatConfig,
        reply: ResponseTx<ChatResult>,
    },
    /// Start a new session via the text-only jinja-render path with
    /// `<|im_end|>` as the stop token. See
    /// [`Qwen35Inner::chat_session_start_sync`] for the behavioural
    /// contract (full cache reset, text-only, session boundary).
    ChatSessionStart {
        messages: Vec<ChatMessage>,
        config: ChatConfig,
        reply: ResponseTx<ChatResult>,
    },
    /// Continue an existing session by appending a user turn. See
    /// [`Qwen35Inner::chat_session_continue_sync`] — builds a raw ChatML
    /// delta from `user_message`, tokenizes it, and prefills on top of
    /// the live caches.
    ///
    /// `images` is an opt-in guard parameter: non-empty input is rejected
    /// with an `IMAGE_CHANGE_REQUIRES_SESSION_RESTART:`-prefixed error so
    /// the TS `ChatSession` layer can route image-changes back through a
    /// fresh `chat_session_start`.
    ChatSessionContinue {
        user_message: String,
        images: Option<Vec<Uint8Array>>,
        config: ChatConfig,
        reply: ResponseTx<ChatResult>,
    },
    /// Continue an existing session with a tool-result delta. See
    /// [`Qwen35Inner::chat_session_continue_tool_sync`] — builds a
    /// ChatML `<tool_response>` delta and prefills on top of the live
    /// caches.
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
    Generate {
        prompt_tokens: MxArray,
        config: Qwen3_5GenerationConfig,
        reply: ResponseTx<Qwen3_5GenerationResult>,
    },
    TakeCache {
        reply: ResponseTx<Option<crate::models::qwen3_5::prompt_cache::PromptCache>>,
    },
    SetCache {
        cache: crate::models::qwen3_5::prompt_cache::PromptCache,
        reply: ResponseTx<()>,
    },
    InitCaches {
        reply: ResponseTx<()>,
    },
    ResetCaches {
        reply: ResponseTx<()>,
    },
    SaveModel {
        save_path: String,
        reply: ResponseTx<()>,
    },
    // --- Training commands ---
    InitTraining {
        config: Box<crate::grpo::engine::GRPOEngineConfig>,
        model_type: crate::training_model::ModelType,
        reply: ResponseTx<()>,
    },
    GenerateForTraining {
        prompts: Vec<Vec<crate::tokenizer::ChatMessage>>,
        group_size: usize,
        gen_config: crate::models::qwen3::GenerationConfig,
        enable_thinking: Option<bool>,
        tools: Option<Vec<crate::tokenizer::ToolDefinition>>,
        reply: ResponseTx<crate::training_model::GenerationPlainData>,
    },
    TrainStepGRPO {
        rewards: Vec<f64>,
        group_size: i32,
        loss_config: crate::grpo::loss::GRPOLossConfig,
        valid_indices: Option<Vec<usize>>,
        reply: ResponseTx<crate::training_model::TrainStepPlainMetrics>,
    },
    /// Bump the training step counter without applying gradients
    /// (used by engine skip paths that abort before training).
    /// Also clears cached generation MxArrays.
    /// Returns the new step.
    BumpSkippedStep {
        reply: ResponseTx<i64>,
    },
    /// Restore the training step counter (for resume from checkpoint).
    /// Does not touch optimizer state — that's loaded via LoadOptimizerState.
    SetTrainingStep {
        step: i64,
        reply: ResponseTx<()>,
    },
    /// Drop the training state on the model thread.
    /// After this, InitTraining can be called again. No-op if no training state.
    ResetTraining {
        reply: ResponseTx<()>,
    },
    TrainStepSFT {
        input_ids: Vec<i32>,
        input_shape: Vec<i64>,
        labels: Vec<i32>,
        labels_shape: Vec<i64>,
        config: crate::sft::engine::SftEngineConfig,
        reply: ResponseTx<crate::training_model::TrainStepPlainMetrics>,
    },
    SaveOptimizerState {
        path: String,
        reply: ResponseTx<()>,
    },
    LoadOptimizerState {
        path: String,
        reply: ResponseTx<()>,
    },
}

/// Command handler for the dedicated model thread.
pub(crate) fn handle_qwen35_cmd(inner: &mut Qwen35Inner, cmd: Qwen35Cmd) {
    match cmd {
        Qwen35Cmd::ChatTokensDelta {
            delta_tokens,
            config,
            reply,
        } => {
            let _ = reply.send(inner.chat_tokens_delta_sync(delta_tokens, config));
        }
        Qwen35Cmd::ChatSessionStart {
            messages,
            config,
            reply,
        } => {
            let _ = reply.send(inner.chat_session_start_sync(messages, config));
        }
        Qwen35Cmd::ChatSessionContinue {
            user_message,
            images,
            config,
            reply,
        } => {
            let _ = reply.send(inner.chat_session_continue_sync(user_message, images, config));
        }
        Qwen35Cmd::ChatSessionContinueTool {
            tool_call_id,
            content,
            config,
            reply,
        } => {
            let _ =
                reply.send(inner.chat_session_continue_tool_sync(tool_call_id, content, config));
        }
        Qwen35Cmd::ChatStreamSessionStart {
            messages,
            config,
            stream_tx,
            cancelled,
        } => {
            inner.chat_stream_session_start_sync(messages, config, stream_tx, cancelled);
        }
        Qwen35Cmd::ChatStreamSessionContinue {
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
        Qwen35Cmd::ChatStreamSessionContinueTool {
            tool_call_id,
            content,
            stream_tx,
            config,
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
        Qwen35Cmd::Generate {
            prompt_tokens,
            config,
            reply,
        } => {
            let _ = reply.send(inner.generate_sync(prompt_tokens, config));
        }
        Qwen35Cmd::TakeCache { reply } => {
            let _ = reply.send(Ok(inner.take_cache_sync()));
        }
        Qwen35Cmd::SetCache { cache, reply } => {
            let _ = reply.send(inner.set_cache_sync(cache));
        }
        Qwen35Cmd::InitCaches { reply } => {
            let _ = reply.send(inner.init_caches_sync());
        }
        Qwen35Cmd::ResetCaches { reply } => {
            let _ = reply.send(inner.reset_caches_sync());
        }
        Qwen35Cmd::SaveModel { save_path, reply } => {
            let _ = reply.send(inner.save_model_sync(&save_path));
        }
        // --- Training commands ---
        Qwen35Cmd::InitTraining {
            config,
            model_type,
            reply,
        } => {
            let _ = reply.send(inner.init_training_sync(*config, model_type));
        }
        Qwen35Cmd::GenerateForTraining {
            prompts,
            group_size,
            gen_config,
            enable_thinking,
            tools,
            reply,
        } => {
            let _ = reply.send(inner.generate_for_training_thread_sync(
                prompts,
                group_size,
                gen_config,
                enable_thinking,
                tools,
            ));
        }
        Qwen35Cmd::TrainStepGRPO {
            rewards,
            group_size,
            loss_config,
            valid_indices,
            reply,
        } => {
            let _ = reply.send(inner.train_step_grpo_sync(
                rewards,
                group_size,
                loss_config,
                valid_indices,
            ));
        }
        Qwen35Cmd::BumpSkippedStep { reply } => {
            let result = if let Some(ref mut ts) = inner.training_state {
                ts.clear_generation_cache();
                ts.step += 1;
                Ok(ts.step)
            } else {
                Err(napi::Error::from_reason(
                    "Training state not initialized. Call InitTraining first.",
                ))
            };
            let _ = reply.send(result);
        }
        Qwen35Cmd::SetTrainingStep { step, reply } => {
            let result = if let Some(ref mut ts) = inner.training_state {
                ts.step = step;
                Ok(())
            } else {
                Err(napi::Error::from_reason(
                    "Training state not initialized. Call InitTraining first.",
                ))
            };
            let _ = reply.send(result);
        }
        Qwen35Cmd::ResetTraining { reply } => {
            inner.training_state = None;
            let _ = reply.send(Ok(()));
        }
        Qwen35Cmd::TrainStepSFT {
            input_ids,
            input_shape,
            labels,
            labels_shape,
            config,
            reply,
        } => {
            let _ = reply.send(inner.train_step_sft_sync(
                input_ids,
                input_shape,
                labels,
                labels_shape,
                config,
            ));
        }
        Qwen35Cmd::SaveOptimizerState { path, reply } => {
            let _ = reply.send(inner.save_optimizer_state_sync(path));
        }
        Qwen35Cmd::LoadOptimizerState { path, reply } => {
            let _ = reply.send(inner.load_optimizer_state_sync(path));
        }
    }
}

/// Input bundle for [`Qwen35Inner::chat_with_caches_inner`].
///
/// Packs every value the shared post-prefill pipeline needs into a single
/// named struct so callers don't have to thread 20+ positional arguments.
/// Constructed by the prefill-side of [`Qwen35Inner::chat_sync`] and
/// [`Qwen35Inner::chat_tokens_delta_sync`].
///
/// The caller is responsible for:
///   - acquiring `DENSE_COMPILED_MUTEX` and `COMPILED_WEIGHTS_RWLOCK` in
///     the correct order (when `use_compiled == true`),
///   - constructing a `WiredLimitContext` tied to `generation_stream` for
///     the lifetime of the call,
///   - running prefill and packaging the resulting `last_logits`,
///     `seq_len`, and `vlm_compiled_init_done`.
pub(crate) struct ChatDecodeInputs {
    // --- Prefill outputs -------------------------------------------------
    /// Logits for the last position of the prefill chunk. Penalties and
    /// sampling run against this to produce the first decoded token.
    pub last_logits: MxArray,
    /// Total context length after prefill (cached + newly-prefilled).
    /// Used to compute the compiled path's `max_kv_len`.
    pub seq_len: i64,
    /// `true` when the VLM prefill has already run the compiled init.
    /// `false` for text-only paths and the session delta path.
    pub vlm_compiled_init_done: bool,

    // --- Compiled-path state --------------------------------------------
    /// `true` when this model owns the compiled weights and the compiled
    /// forward path is usable for decode.
    pub use_compiled: bool,
    /// `true` when the current turn carries images.
    pub has_images: bool,
    /// Length of the cached prefix that the prefill reused. Only
    /// consulted by the VLM rope-delta replay branch.
    pub cached_prefix_len: usize,

    // --- Token bookkeeping ----------------------------------------------
    /// Full pre-decode token sequence. Seeds the decode loop's running
    /// history (mutated in place) and the penalty context.
    pub token_history_init: Vec<u32>,
    /// Token snapshot handed to `save_cache_state_direct`. For text-only
    /// this equals `token_history_init`; for VLM it's the pre-expansion
    /// tokens.
    pub save_tokens: Vec<u32>,
    /// Expanded token sequence (with image placeholders expanded) used by
    /// the VLM save path. `None` for text-only.
    pub save_expanded_tokens: Option<Vec<u32>>,
    /// Image cache key for the current turn. 0 for text-only.
    pub save_image_cache_key: u64,

    // --- Tokenizer / reasoning state ------------------------------------
    pub tokenizer: Arc<Qwen3Tokenizer>,
    pub think_end_id: Option<u32>,
    pub think_end_str: Option<String>,
    pub enable_thinking: Option<bool>,
    /// End-of-sequence token id for the decode loop. For `chat_sync` this
    /// is `config.eos_token_id`; for the session delta path it's
    /// `<|im_end|>` so cache boundaries stay clean.
    pub eos_id: u32,

    // --- Profiler / perf metrics ----------------------------------------
    pub profiler: crate::decode_profiler::DecodeProfiler,
    pub generation_start: Option<std::time::Instant>,
    pub first_token_instant: Option<std::time::Instant>,
    /// Number of tokens actually prefilled this turn (for throughput math).
    pub prefill_tokens_len: usize,
    /// Prompt token count reported on the `ChatResult`.
    pub prompt_tokens_for_result: u32,

    // --- MLX state ------------------------------------------------------
    pub embedding_weight: MxArray,
    pub embedding_weight_t: MxArray,
    pub generation_stream: Stream,
    pub params: super::chat_common::ChatParams,
}

// ========== Qwen35Inner implementation ==========
// All these methods run on the dedicated model thread (synchronous, no locks).

impl Qwen35Inner {
    /// Create a new Qwen35Inner with the given configuration.
    pub(crate) fn new(config: Qwen3_5Config) -> Result<Self> {
        let embedding = Embedding::new(config.vocab_size as u32, config.hidden_size as u32)?;

        let layers = (0..config.num_layers as usize)
            .map(|i| DecoderLayer::new(&config, i))
            .collect::<Result<Vec<_>>>()?;

        let final_norm = RMSNorm::new(config.hidden_size as u32, Some(config.rms_norm_eps))?;

        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(Linear::new(
                config.hidden_size as u32,
                config.vocab_size as u32,
                Some(false),
            )?)
        };

        let model_id = QWEN35_MODEL_ID_COUNTER.fetch_add(1, Ordering::Relaxed);

        Ok(Self {
            config,
            embedding,
            layers,
            final_norm,
            lm_head,
            caches: None,
            tokenizer: None,
            vision_encoder: None,
            image_processor: None,
            spatial_merge_size: None,
            vision_cache: Arc::new(Mutex::new(VisionCacheInner {
                entries: HashMap::new(),
                generation: 0,
            })),
            cached_token_history: Vec::new(),
            cached_image_key: None,
            cached_rope_deltas: None,
            model_id,
            training_state: None,
        })
    }

    /// Initialize KV caches.
    pub(crate) fn init_caches_sync(&mut self) -> Result<()> {
        let caches = (0..self.config.num_layers as usize)
            .map(|i| {
                if self.config.is_linear_layer(i) {
                    Qwen3_5LayerCache::new_linear()
                } else {
                    Qwen3_5LayerCache::new_full_attention()
                }
            })
            .collect();
        self.caches = Some(caches);
        self.clear_reuse_state();
        Ok(())
    }

    /// Reset all caches.
    pub(crate) fn reset_caches_sync(&mut self) -> Result<()> {
        if let Some(ref mut caches) = self.caches {
            for cache in caches.iter_mut() {
                cache.reset();
            }
        }
        self.caches = None;
        self.clear_reuse_state();
        Ok(())
    }

    /// Clear cached token history, image key, and rope deltas.
    fn clear_reuse_state(&mut self) {
        self.cached_token_history.clear();
        self.cached_image_key = None;
        self.cached_rope_deltas = None;
    }

    /// Take the KV cache from the model, returning a `PromptCache` handle.
    pub(crate) fn take_cache_sync(
        &mut self,
    ) -> Option<crate::models::qwen3_5::prompt_cache::PromptCache> {
        if self.cached_token_history.is_empty() {
            return None;
        }
        let caches = self.caches.take()?;
        Some(crate::models::qwen3_5::prompt_cache::PromptCache::new(
            caches,
            self.cached_token_history.clone(),
            "qwen3_5",
            self.config.num_layers as usize,
            self.cached_image_key,
            self.cached_rope_deltas,
            self.model_id,
        ))
    }

    /// Restore a previously taken `PromptCache` into the model.
    pub(crate) fn set_cache_sync(
        &mut self,
        mut cache: crate::models::qwen3_5::prompt_cache::PromptCache,
    ) -> Result<()> {
        let restored_caches = cache.take_caches().ok_or_else(|| {
            Error::from_reason("PromptCache is empty (already consumed or disposed)")
        })?;
        self.caches = Some(restored_caches);
        self.cached_token_history = cache.token_history().to_vec();
        self.cached_image_key = cache.image_cache_key();
        self.cached_rope_deltas = cache.rope_deltas();
        Ok(())
    }

    /// Save model weights and configuration to a directory (synchronous).
    pub(crate) fn save_model_sync(&self, save_path: &str) -> Result<()> {
        use super::decoder_layer::AttentionType;

        let mut params = HashMap::new();

        // Embedding
        params.insert("embedding.weight".to_string(), self.embedding.get_weight());

        // Layers
        for (i, layer) in self.layers.iter().enumerate() {
            let prefix = format!("layers.{}", i);
            match &layer.attn {
                AttentionType::Linear(gdn) => {
                    params.insert(
                        format!("{}.linear_attn.in_proj_qkvz.weight", prefix),
                        gdn.get_in_proj_qkvz_weight(),
                    );
                    params.insert(
                        format!("{}.linear_attn.in_proj_ba.weight", prefix),
                        gdn.get_in_proj_ba_weight(),
                    );
                    params.insert(
                        format!("{}.linear_attn.conv1d.weight", prefix),
                        gdn.get_conv1d_weight(),
                    );
                    params.insert(
                        format!("{}.linear_attn.norm.weight", prefix),
                        gdn.get_norm_weight(),
                    );
                    params.insert(
                        format!("{}.linear_attn.out_proj.weight", prefix),
                        gdn.get_out_proj_weight(),
                    );
                    params.insert(format!("{}.linear_attn.dt_bias", prefix), gdn.get_dt_bias());
                    params.insert(format!("{}.linear_attn.a_log", prefix), gdn.get_a_log());
                }
                AttentionType::Full(attn) => {
                    params.insert(
                        format!("{}.self_attn.q_proj.weight", prefix),
                        attn.get_q_proj_weight(),
                    );
                    params.insert(
                        format!("{}.self_attn.k_proj.weight", prefix),
                        attn.get_k_proj_weight(),
                    );
                    params.insert(
                        format!("{}.self_attn.v_proj.weight", prefix),
                        attn.get_v_proj_weight(),
                    );
                    params.insert(
                        format!("{}.self_attn.o_proj.weight", prefix),
                        attn.get_o_proj_weight(),
                    );
                    params.insert(
                        format!("{}.self_attn.q_norm.weight", prefix),
                        attn.get_q_norm_weight(),
                    );
                    params.insert(
                        format!("{}.self_attn.k_norm.weight", prefix),
                        attn.get_k_norm_weight(),
                    );
                }
            }
            params.insert(
                format!("{}.mlp.gate_proj.weight", prefix),
                layer.mlp.get_gate_proj_weight(),
            );
            params.insert(
                format!("{}.mlp.up_proj.weight", prefix),
                layer.mlp.get_up_proj_weight(),
            );
            params.insert(
                format!("{}.mlp.down_proj.weight", prefix),
                layer.mlp.get_down_proj_weight(),
            );
            params.insert(
                format!("{}.input_layernorm.weight", prefix),
                layer.get_input_layernorm_weight(),
            );
            params.insert(
                format!("{}.post_attention_layernorm.weight", prefix),
                layer.get_post_attention_layernorm_weight(),
            );
        }

        // Final norm
        params.insert(
            "final_norm.weight".to_string(),
            self.final_norm.get_weight(),
        );

        // LM head
        if !self.config.tie_word_embeddings
            && let Some(ref lm_head) = self.lm_head
        {
            params.insert("lm_head.weight".to_string(), lm_head.get_weight());
        }

        // Include vision encoder weights
        if let Some(ref vision_enc) = self.vision_encoder {
            let vision_params = vision_enc.get_parameters();
            params.extend(vision_params);
        }

        // Validate for NaN/Inf
        for (name, param) in params.iter() {
            let data = param.to_float32()?;
            let invalid_count = data
                .iter()
                .filter(|v| v.is_nan() || v.is_infinite())
                .count();
            if invalid_count > 0 {
                return Err(napi::Error::new(
                    napi::Status::GenericFailure,
                    format!(
                        "Cannot save model: parameter '{}' contains {} NaN/Inf values.",
                        name, invalid_count
                    ),
                ));
            }
        }

        let params_clone: HashMap<String, MxArray> =
            params.iter().map(|(k, v)| (k.clone(), v.clone())).collect();

        // Weights metadata
        let mut weights_metadata = serde_json::Map::new();
        for (key, array) in params.iter() {
            let shape_data = array.shape()?;
            let shape: Vec<i64> = shape_data.as_ref().to_vec();
            let dtype = array.dtype()?;
            let mut param_info = serde_json::Map::new();
            param_info.insert("shape".to_string(), serde_json::json!(shape));
            param_info.insert("dtype".to_string(), serde_json::json!(dtype as i32));
            weights_metadata.insert(key.clone(), serde_json::Value::Object(param_info));
        }

        // Config JSON
        let mut config_value = serde_json::to_value(&self.config).map_err(|e| {
            napi::Error::new(
                napi::Status::GenericFailure,
                format!("Failed to serialize config: {e}"),
            )
        })?;
        if let serde_json::Value::Object(ref mut map) = config_value {
            map.insert("model_type".to_string(), serde_json::json!("qwen3_5"));
        }

        let weights_json = serde_json::json!({
            "version": "1.0",
            "config": config_value,
            "weights": weights_metadata,
            "note": "Full weights are in weights.safetensors"
        });

        let path = std::path::Path::new(save_path);
        std::fs::create_dir_all(path)?;

        info!("Saving model to {}", save_path);

        let config_path = path.join("config.json");
        let config_json = serde_json::to_string_pretty(&config_value)?;
        std::fs::write(&config_path, config_json)?;
        info!("Saved config.json");

        let safetensors_path = path.join("weights.safetensors");
        let metadata = Some(serde_json::json!({
            "format": "mlx-node",
            "version": "1.0"
        }));
        crate::utils::safetensors::save_safetensors(&safetensors_path, &params_clone, metadata)?;
        info!("Saved weights.safetensors");

        let weights_str = serde_json::to_string_pretty(&weights_json)?;
        let weights_path = path.join("weights.mlx");
        std::fs::write(&weights_path, weights_str)?;
        info!("Saved weights.mlx metadata");

        Ok(())
    }

    /// Set the tokenizer.
    pub(crate) fn set_tokenizer(&mut self, tokenizer: Arc<Qwen3Tokenizer>) {
        self.tokenizer = Some(tokenizer);
    }

    /// Set the vision encoder.
    pub(crate) fn set_vision_encoder(&mut self, enc: Qwen3_5VisionEncoder) {
        self.vision_encoder = Some(Arc::new(enc));
    }

    /// Set the image processor.
    pub(crate) fn set_image_processor(&mut self, proc: Qwen35VLImageProcessor) {
        self.image_processor = Some(Arc::new(proc));
    }

    /// Set spatial merge size.
    pub(crate) fn set_spatial_merge_size(&mut self, size: i32) {
        self.spatial_merge_size = Some(size);
    }

    /// Initialize M-RoPE on all full attention layers (VLM mode).
    pub(crate) fn init_mrope_layers(
        &mut self,
        mrope_section: Vec<i32>,
        rope_theta: f64,
        max_position_embeddings: i32,
    ) -> Result<()> {
        let rope_dims = self.config.rope_dims();
        for layer in self.layers.iter_mut() {
            if let super::decoder_layer::AttentionType::Full(ref mut attn) = layer.attn {
                attn.init_mrope(
                    mrope_section.clone(),
                    rope_theta,
                    max_position_embeddings,
                    rope_dims,
                )?;
            }
        }
        Ok(())
    }

    /// Start a new chat session.
    ///
    /// Resets the caches up-front so the session is guaranteed to start
    /// from a known-clean state, then delegates to [`Self::chat_sync_core`]
    /// with `<|im_end|>` (from the tokenizer vocab) as the stop token so
    /// the cached history ends on a clean ChatML boundary that subsequent
    /// `chat_session_continue_sync` / [`Self::chat_tokens_delta_sync`]
    /// calls can append a raw delta on top of without re-rendering the
    /// jinja template.
    ///
    /// Images are accepted on session start — the downstream
    /// [`Self::chat_sync_core`] already handles the VLM prefill path
    /// (vision encoder, image cache key, expanded tokens). Subsequent
    /// turns in the same session MUST go through `chat_session_continue_sync`
    /// which is text-only; changing the image set mid-session requires
    /// starting a new session via this method again.
    pub(crate) fn chat_session_start_sync(
        &mut self,
        messages: Vec<ChatMessage>,
        config: ChatConfig,
    ) -> Result<ChatResult> {
        // Mirror the symmetric guard in `chat_tokens_delta_sync`. The session
        // API only makes sense with cache reuse enabled: if we silently accept
        // `reuse_cache = false` here, the post-decode `save_cache_state_direct`
        // path wipes the caches we just populated, and the next
        // `chat_session_continue` call fails with a cryptic "missing session"
        // error. Fail fast before mutating any state.
        if config.reuse_cache == Some(false) {
            return Err(Error::from_reason(
                "chat_session_start requires reuse_cache=true (pass ChatConfig { reuse_cache: Some(true), .. } or leave as None). The session API only makes sense with cache reuse enabled.",
            ));
        }

        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))?
            .clone();
        let im_end_id = tokenizer
            .im_end_id()
            .ok_or_else(|| Error::from_reason("Tokenizer missing <|im_end|> special token"))?;

        // Full reset: the session-start path always begins from a clean
        // state. This matches the documented contract that the session is
        // owned end-to-end by the `ChatSession<Qwen35Model>` /
        // `chat_session_*` surface and intentionally invalidates any
        // prior session cache.
        self.reset_caches_sync()?;
        self.init_caches_sync()?;

        self.chat_sync_core(messages, config, im_end_id)
    }

    /// Core synchronous chat implementation (runs on the model thread).
    ///
    /// Shared jinja rendering + prefill + decode plumbing for the session
    /// surface. `eos_token_id` is the caller-supplied stop-on token id
    /// (`<|im_end|>` for ChatML boundaries) so the cached history ends on
    /// a clean delimiter that subsequent session-delta turns can append
    /// to.
    ///
    /// Only called from [`Self::chat_session_start_sync`]; there is no
    /// longer a non-session entry point.
    fn chat_sync_core(
        &mut self,
        messages: Vec<ChatMessage>,
        config: ChatConfig,
        eos_token_id: u32,
    ) -> Result<ChatResult> {
        let reuse_cache = config.reuse_cache.unwrap_or(true);
        let report_perf = config.report_performance.unwrap_or(false);

        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))?
            .clone();

        let has_images = messages
            .iter()
            .any(|m| m.images.as_ref().is_some_and(|imgs| !imgs.is_empty()));

        let think_end_id = tokenizer.think_end_id();
        let think_end_str = tokenizer.think_end_str().map(|s| s.to_string());

        let tool_defs = config.tools.as_deref();
        let enable_thinking = chat_common::resolve_enable_thinking(&config);
        let tokens = tokenizer.apply_chat_template_sync(
            &messages,
            Some(true),
            tool_defs,
            enable_thinking,
        )?;

        let p = extract_chat_params(&config);
        let max_new_tokens = p.max_new_tokens;

        let generation_start = if report_perf {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let first_token_instant: Option<std::time::Instant> = None;

        let model_id = self.model_id;

        // Check if compiled path will be used
        let use_compiled = unsafe { mlx_sys::mlx_qwen35_get_model_id() } == model_id;

        // Serialize compiled lifecycle across model instances
        let _compiled_lock = if use_compiled {
            Some(
                DENSE_COMPILED_MUTEX
                    .lock()
                    .unwrap_or_else(|e| e.into_inner()),
            )
        } else {
            None
        };

        // Re-validate compiled path under weight lock
        let mut _weight_guard = None;
        let use_compiled = if use_compiled {
            let guard = COMPILED_WEIGHTS_RWLOCK.read().unwrap();
            if unsafe { mlx_sys::mlx_qwen35_get_model_id() } == model_id {
                _weight_guard = Some(guard);
                true
            } else {
                false
            }
        } else {
            false
        };

        let embedding_weight = self.embedding.get_weight();

        // === VLM image processing ===
        let sms = self.spatial_merge_size.unwrap_or(2);
        let (expanded_tokens, current_image_cache_key, vlm_processed) = if has_images {
            if let (Some(_vision_enc), Some(img_proc)) =
                (self.vision_encoder.as_ref(), self.image_processor.as_ref())
            {
                let all_images = extract_images_from_messages(&messages);
                let image_refs: Vec<&[u8]> = all_images.iter().map(|v| v.as_slice()).collect();
                let processed_pre = img_proc.process_many(&image_refs)?;
                let num_image_tokens = compute_num_image_tokens(&processed_pre.grid_thw(), sms)?;
                let expanded = inject_image_placeholders(&tokens, num_image_tokens);
                let cache_key = compute_image_cache_key(&all_images);
                (expanded, cache_key, Some(processed_pre))
            } else {
                (tokens.clone(), 0u64, None)
            }
        } else {
            (tokens.clone(), 0u64, None)
        };

        // === Cache reuse: prefix verification ===
        let cached_prefix_len = verify_cache_prefix_direct(
            reuse_cache,
            has_images,
            &tokens,
            &expanded_tokens,
            current_image_cache_key,
            &self.cached_token_history,
            &self.cached_image_key,
            self.caches.is_some(),
        );

        let prefill_tokens = if cached_prefix_len > 0 {
            if has_images {
                info!(
                    "VLM cache reuse: {} cached tokens, {} new tokens to prefill",
                    cached_prefix_len,
                    expanded_tokens.len() - cached_prefix_len
                );
                expanded_tokens[cached_prefix_len..].to_vec()
            } else {
                info!(
                    "Cache reuse: {} cached tokens, {} new tokens to prefill",
                    cached_prefix_len,
                    tokens.len() - cached_prefix_len
                );
                tokens[cached_prefix_len..].to_vec()
            }
        } else {
            // Full reset
            if let Some(ref mut caches) = self.caches {
                for cache in caches.iter_mut() {
                    cache.reset();
                }
            }
            let new_caches = (0..self.config.num_layers as usize)
                .map(|i| {
                    if self.config.is_linear_layer(i) {
                        Qwen3_5LayerCache::new_linear()
                    } else {
                        Qwen3_5LayerCache::new_full_attention()
                    }
                })
                .collect();
            self.caches = Some(new_caches);
            tokens.clone()
        };

        // Zero-delta guard
        let (prefill_tokens, cached_prefix_len) = if prefill_tokens.is_empty() {
            info!("Zero-delta cache hit: resetting caches for full re-prefill");
            if let Some(ref mut caches) = self.caches {
                for cache in caches.iter_mut() {
                    cache.reset();
                }
            }
            let new_caches = (0..self.config.num_layers as usize)
                .map(|i| {
                    if self.config.is_linear_layer(i) {
                        Qwen3_5LayerCache::new_linear()
                    } else {
                        Qwen3_5LayerCache::new_full_attention()
                    }
                })
                .collect();
            self.caches = Some(new_caches);
            let tokens = if has_images {
                expanded_tokens.clone()
            } else {
                tokens.clone()
            };
            (tokens, 0)
        } else {
            (prefill_tokens, cached_prefix_len)
        };

        let eos_id = eos_token_id;

        let embedding_weight_t = embedding_weight.transpose(Some(&[1, 0]))?;
        let generation_stream = Stream::new(DeviceType::Gpu);
        let model_size_bytes = self.config.estimate_memory_bytes() as usize;
        let _wired_ctx =
            crate::stream::WiredLimitContext::new(model_size_bytes, vec![generation_stream]);

        let mut profiler = crate::decode_profiler::DecodeProfiler::new("chat", "qwen3_5");
        profiler.set_prompt_tokens(prefill_tokens.len() as u32);
        profiler.snapshot_memory_before();

        // === VLM or text prefill branching ===
        profiler.begin_prefill();
        let (last_logits, seq_len, vlm_compiled_init_done) = if has_images && cached_prefix_len == 0
        {
            if let Some(vision_enc) = self.vision_encoder.clone() {
                let final_tokens = &expanded_tokens;
                let processed = vlm_processed
                    .as_ref()
                    .ok_or_else(|| Error::from_reason("VLM processed images missing"))?;

                let input_ids =
                    MxArray::from_uint32(final_tokens, &[1, final_tokens.len() as i64])?;

                let (logits, rope_deltas, vlm_compiled) = vlm_prefill(
                    &input_ids,
                    current_image_cache_key,
                    processed,
                    &vision_enc,
                    sms,
                    &embedding_weight,
                    &mut self.layers,
                    &mut self.caches,
                    &self.final_norm,
                    &self.lm_head,
                    &self.config,
                    max_new_tokens,
                    generation_stream,
                    &self.vision_cache,
                    model_id,
                )?;

                self.cached_rope_deltas = Some(rope_deltas as i32);

                let vlm_seq_len = final_tokens.len() as i64;
                (logits, vlm_seq_len, vlm_compiled)
            } else {
                return Err(Error::from_reason(
                    "VLM prefill requested but vision encoder/processor not loaded",
                ));
            }
        } else {
            let prompt = MxArray::from_uint32(&prefill_tokens, &[1, prefill_tokens.len() as i64])?;
            let logits = chunked_prefill(
                &prompt,
                &embedding_weight,
                &mut self.layers,
                &mut self.caches,
                &self.final_norm,
                &self.lm_head,
                Some(&embedding_weight_t),
                generation_stream,
            )?;

            let seq_len = logits.shape_at(1)?;
            let last_logits = logits.slice_axis(1, seq_len - 1, seq_len)?;
            let last_logits = last_logits.squeeze(Some(&[1]))?;
            let total_seq_len = if has_images {
                expanded_tokens.len() as i64
            } else {
                tokens.len() as i64
            };
            (last_logits, total_seq_len, false)
        };
        profiler.end_prefill();

        let prompt_tokens_for_result = if has_images {
            expanded_tokens.len() as u32
        } else {
            tokens.len() as u32
        };

        let save_expanded_tokens = if has_images {
            Some(expanded_tokens.clone())
        } else {
            None
        };

        self.chat_with_caches_inner(ChatDecodeInputs {
            last_logits,
            seq_len,
            vlm_compiled_init_done,
            use_compiled,
            has_images,
            cached_prefix_len,
            token_history_init: tokens.clone(),
            save_tokens: tokens,
            save_expanded_tokens,
            save_image_cache_key: current_image_cache_key,
            tokenizer,
            think_end_id,
            think_end_str,
            enable_thinking,
            eos_id,
            profiler,
            generation_start,
            first_token_instant,
            prefill_tokens_len: prefill_tokens.len(),
            prompt_tokens_for_result,
            embedding_weight,
            embedding_weight_t,
            generation_stream,
            params: p,
        })
    }

    /// Session-based chat continuation via a pre-tokenized delta.
    ///
    /// Runs a text-only prefill of `delta_tokens` on top of the existing KV
    /// caches and decodes the next reply. This path:
    /// - skips the jinja chat template entirely (caller produces the delta),
    /// - skips prefix verification (caller owns cache coherence by construction),
    /// - uses `<|im_end|>` (from the tokenizer vocab) as its stop token instead
    ///   of `config.eos_token_id`, yielding clean cache boundaries for the next
    ///   turn's delta,
    /// - resolves `enable_thinking` from `config.reasoning_effort` via
    ///   `chat_common::resolve_enable_thinking`,
    /// - is text-only: errors if the session has images.
    ///
    /// Requires a live session: `self.caches` must have been initialized by a
    /// prior [`Self::chat_session_start_sync`] call. Errors otherwise.
    pub(crate) fn chat_tokens_delta_sync(
        &mut self,
        delta_tokens: Vec<u32>,
        config: ChatConfig,
    ) -> Result<ChatResult> {
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
        if self.caches.is_none() {
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
            return Err(Error::from_reason(
                "chat_tokens_delta_sync is text-only; session currently holds image state",
            ));
        }

        let report_perf = config.report_performance.unwrap_or(false);

        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))?
            .clone();

        // Session path: use <|im_end|> as eos, NOT config.eos_token_id.
        // This yields clean cache boundaries (see Phase 0 validation notes).
        let eos_id = tokenizer
            .im_end_id()
            .ok_or_else(|| Error::from_reason("Tokenizer missing <|im_end|> special token"))?;

        let think_end_id = tokenizer.think_end_id();
        let think_end_str = tokenizer.think_end_str().map(|s| s.to_string());

        // Build full token history = cached_history + delta. Used for
        // penalty context AND as the running token history in the decode loop.
        // Also used as the snapshot we hand to `save_cache_state_direct` so
        // the saved `cached_token_history` correctly reflects the appended
        // delta plus the generated tokens.
        let mut full_token_history = self.cached_token_history.clone();
        full_token_history.extend(delta_tokens.iter().copied());

        let p = extract_chat_params(&config);
        let enable_thinking = chat_common::resolve_enable_thinking(&config);

        let generation_start = if report_perf {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let first_token_instant: Option<std::time::Instant> = None;

        let model_id = self.model_id;

        // Check compiled path availability (same contract as chat_sync).
        let use_compiled = unsafe { mlx_sys::mlx_qwen35_get_model_id() } == model_id;
        let _compiled_lock = if use_compiled {
            Some(
                DENSE_COMPILED_MUTEX
                    .lock()
                    .unwrap_or_else(|e| e.into_inner()),
            )
        } else {
            None
        };

        let mut _weight_guard = None;
        let use_compiled = if use_compiled {
            let guard = COMPILED_WEIGHTS_RWLOCK.read().unwrap();
            if unsafe { mlx_sys::mlx_qwen35_get_model_id() } == model_id {
                _weight_guard = Some(guard);
                true
            } else {
                false
            }
        } else {
            false
        };

        let embedding_weight = self.embedding.get_weight();
        let embedding_weight_t = embedding_weight.transpose(Some(&[1, 0]))?;
        let generation_stream = Stream::new(DeviceType::Gpu);
        let model_size_bytes = self.config.estimate_memory_bytes() as usize;
        let _wired_ctx =
            crate::stream::WiredLimitContext::new(model_size_bytes, vec![generation_stream]);

        let mut profiler = crate::decode_profiler::DecodeProfiler::new("chat_delta", "qwen3_5");
        profiler.set_prompt_tokens(delta_tokens.len() as u32);
        profiler.snapshot_memory_before();

        // Text-only prefill of the delta on top of the existing caches.
        profiler.begin_prefill();
        let prompt = MxArray::from_uint32(&delta_tokens, &[1, delta_tokens.len() as i64])?;
        let logits = chunked_prefill(
            &prompt,
            &embedding_weight,
            &mut self.layers,
            &mut self.caches,
            &self.final_norm,
            &self.lm_head,
            Some(&embedding_weight_t),
            generation_stream,
        )?;
        let prefill_out_seq_len = logits.shape_at(1)?;
        let last_logits = logits.slice_axis(1, prefill_out_seq_len - 1, prefill_out_seq_len)?;
        let last_logits = last_logits.squeeze(Some(&[1]))?;
        // Total context length post-prefill = full history length.
        let total_seq_len = full_token_history.len() as i64;
        profiler.end_prefill();

        let prompt_tokens_for_result = full_token_history.len() as u32;

        // For the delta path there is no cached_prefix_len distinction — the
        // caches already reflect the entire prior history. Pass 0 so the
        // rope-deltas branch inside the helper is skipped (text-only anyway).
        let cached_prefix_len = 0usize;

        // For cache save, pass the full token history (cached + delta) as
        // `save_tokens`; the helper / `save_cache_state_direct` will append
        // the generated tokens.
        let save_tokens = full_token_history.clone();

        self.chat_with_caches_inner(ChatDecodeInputs {
            last_logits,
            seq_len: total_seq_len,
            vlm_compiled_init_done: false,
            use_compiled,
            has_images: false,
            cached_prefix_len,
            token_history_init: full_token_history,
            save_tokens,
            save_expanded_tokens: None,
            save_image_cache_key: 0,
            tokenizer,
            think_end_id,
            think_end_str,
            enable_thinking,
            eos_id,
            profiler,
            generation_start,
            first_token_instant,
            prefill_tokens_len: delta_tokens.len(),
            prompt_tokens_for_result,
            embedding_weight,
            embedding_weight_t,
            generation_stream,
            params: p,
        })
    }

    /// Session-based chat continuation via a plain user message string.
    ///
    /// Convenience entry point on top of `chat_tokens_delta_sync`: builds the
    /// ChatML delta that closes the previous assistant turn (the cache ended
    /// on `<|im_end|>` courtesy of `chat_session_start_sync`), opens a new
    /// user turn with `user_message`, and opens a fresh assistant turn.
    /// Then tokenizes the delta and delegates to `chat_tokens_delta_sync`.
    ///
    /// The delta is built manually (NOT via jinja) to keep prefix stability
    /// against the cached state: re-rendering the full conversation through
    /// jinja would tokenize differently than the accumulated cache and break
    /// the prefix match that makes session reuse correct.
    ///
    /// Text-only; errors propagate from `chat_tokens_delta_sync`.
    ///
    /// `images` is an opt-in guard parameter: non-empty input is rejected
    /// with an `IMAGE_CHANGE_REQUIRES_SESSION_RESTART:`-prefixed error so
    /// the TS `ChatSession` layer can catch the prefix and route
    /// image-changes back through a fresh `chat_session_start_sync`. A
    /// `None`/empty vector takes the normal text-only delta path.
    pub(crate) fn chat_session_continue_sync(
        &mut self,
        user_message: String,
        images: Option<Vec<Uint8Array>>,
        config: ChatConfig,
    ) -> Result<ChatResult> {
        if images.as_ref().is_some_and(|v| !v.is_empty()) {
            return Err(Error::from_reason(format!(
                "{} chat_session_continue is text-only; start a new session with chat_session_start to change the image",
                chat_common::IMAGE_CHANGE_RESTART_PREFIX
            )));
        }

        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))?
            .clone();

        // Match `chat_sync`'s sanitization so the session path is subject to
        // the same role/content injection protection as the legacy path.
        // The delta is text-only — images are stripped here anyway because
        // they are never valid on the session continue path.
        let synthetic = build_synthetic_user_message(&user_message);
        let sanitized = Qwen3Tokenizer::sanitize_messages_public(std::slice::from_ref(&synthetic));
        let sanitized_user = &sanitized[0].content;

        // Build the delta in ChatML wire format. See
        // `chat_common::build_chatml_continue_delta_text` for the exact
        // wire format and thinking-prefix semantics.
        let enable_thinking = chat_common::resolve_enable_thinking(&config);
        let delta_text = build_chatml_continue_delta_text(sanitized_user, enable_thinking);

        // `add_special_tokens: Some(false)` — we do NOT want the tokenizer
        // auto-prepending BOS. The delta is already a raw ChatML snippet.
        let delta_tokens = tokenizer.encode_sync(&delta_text, Some(false))?;

        self.chat_tokens_delta_sync(delta_tokens, config)
    }

    /// Session-based chat continuation via a tool-result turn.
    ///
    /// Builds a ChatML `<tool_response>`-wrapped delta from `content`
    /// (see [`chat_common::build_chatml_tool_delta_text`] for the exact
    /// wire format) and prefills it on top of the live session caches.
    /// The `tool_call_id` is currently ignored by the wire format —
    /// Qwen3.5's chat template identifies tool responses by the
    /// surrounding `<tool_response>` tags + position, not an explicit
    /// id. Callers may still log it for their own bookkeeping.
    ///
    /// Text-only; delegates to [`Self::chat_tokens_delta_sync`] which
    /// inherits the same text-only-delta invariant (errors if the
    /// session currently holds image state).
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
        let delta_text =
            chat_common::build_chatml_tool_delta_text(&tool_call_id, &content, enable_thinking);

        let delta_tokens = tokenizer.encode_sync(&delta_text, Some(false))?;

        self.chat_tokens_delta_sync(delta_tokens, config)
    }

    /// Shared post-prefill pipeline: penalty → sample → compiled init (if needed)
    /// → decode loop → cache export → save cache state → finalize result.
    ///
    /// Extracted from `chat_sync` so it can also be driven by the text-only
    /// session path (`chat_tokens_delta_sync`). Preserves the exact semantics
    /// of `chat_sync` for the existing caller — `token_history_init` is the
    /// full pre-decode token sequence (used for penalty context and the decode
    /// loop's running history), and the decode loop mutates it in place.
    ///
    /// The caller is responsible for:
    /// - Holding the `DENSE_COMPILED_MUTEX` / `COMPILED_WEIGHTS_RWLOCK` guards
    ///   (when `inputs.use_compiled == true`) for the lifetime of this call.
    /// - Creating a `WiredLimitContext` tied to `inputs.generation_stream` for
    ///   the lifetime of this call.
    /// - Running prefill and populating the resulting `last_logits`, `seq_len`,
    ///   and `vlm_compiled_init_done` fields of `ChatDecodeInputs`.
    /// - Pre-starting the profiler (`set_prompt_tokens`, `snapshot_memory_before`,
    ///   `begin_prefill`, `end_prefill`).
    fn chat_with_caches_inner(&mut self, inputs: ChatDecodeInputs) -> Result<ChatResult> {
        let ChatDecodeInputs {
            last_logits,
            seq_len,
            vlm_compiled_init_done,
            use_compiled,
            has_images,
            cached_prefix_len,
            token_history_init,
            save_tokens,
            save_expanded_tokens,
            save_image_cache_key,
            tokenizer,
            think_end_id,
            think_end_str,
            enable_thinking,
            eos_id,
            mut profiler,
            generation_start,
            mut first_token_instant,
            prefill_tokens_len,
            prompt_tokens_for_result,
            embedding_weight,
            embedding_weight_t,
            generation_stream,
            params: p,
        } = inputs;

        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut finish_reason = String::from("length");
        let max_new_tokens = p.max_new_tokens;

        let last_logits = apply_all_penalties(last_logits, &token_history_init, &p)?;
        let mut y = sample(&last_logits, p.sampling_config)?;
        MxArray::async_eval_arrays(&[&y]);

        let _compiled_guard = if use_compiled {
            Some(CompiledResetGuard)
        } else {
            None
        };

        let mut token_history: Vec<u32> = token_history_init;

        let mut reasoning_tracker = chat_common::ReasoningTracker::new(
            enable_thinking.unwrap_or(true),
            p.thinking_token_budget,
            think_end_id,
        );

        if use_compiled {
            if vlm_compiled_init_done {
                // VLM prefill already initialized compiled state
            } else {
                use mlx_sys as sys;
                let prefill_len = seq_len as i32;
                let max_kv_len = ((prefill_len + max_new_tokens + 255) / 256) * 256;
                let num_layers = self.config.num_layers as usize;
                let mut cache_ptrs: Vec<*mut sys::mlx_array> =
                    vec![std::ptr::null_mut(); num_layers * 2];
                if let Some(ref caches) = self.caches {
                    for (i, cache) in caches.iter().enumerate() {
                        let (p0, p1) = cache.export_ptrs();
                        cache_ptrs[i * 2] = p0;
                        cache_ptrs[i * 2 + 1] = p1;
                    }
                }
                unsafe {
                    sys::mlx_qwen35_compiled_init_from_prefill(
                        self.config.num_layers,
                        self.config.hidden_size,
                        self.config.num_heads,
                        self.config.num_kv_heads,
                        self.config.head_dim,
                        self.config.rope_theta as f32,
                        self.config.rope_dims(),
                        self.config.rms_norm_eps as f32,
                        self.config.full_attention_interval,
                        self.config.linear_num_key_heads,
                        self.config.linear_num_value_heads,
                        self.config.linear_key_head_dim,
                        self.config.linear_value_head_dim,
                        self.config.linear_conv_kernel_dim,
                        if self.config.tie_word_embeddings {
                            1
                        } else {
                            0
                        },
                        max_kv_len,
                        1,
                        cache_ptrs.as_mut_ptr(),
                        prefill_len,
                    );
                }

                // VLM cache reuse: apply saved rope_deltas
                if has_images
                    && cached_prefix_len > 0
                    && let Some(delta) = self.cached_rope_deltas
                {
                    unsafe {
                        mlx_sys::mlx_qwen35_compiled_adjust_offset(delta);
                    }
                }
            }

            // For text-only, clear stale rope deltas
            if !has_images {
                self.cached_rope_deltas = None;
            }

            profiler.set_label("chat_compiled");

            let mut ops = chat_common::DecodeOps {
                forward: |ids: &MxArray, emb: &MxArray| -> Result<(MxArray, bool)> {
                    Ok((forward_compiled(ids, emb)?, false))
                },
                eval_step: |token: &MxArray, logits: &MxArray, budget_forced: bool| {
                    eval_token_and_compiled_caches(token);
                    if budget_forced {
                        logits.eval();
                    }
                },
            };
            chat_common::decode_loop!(
                ops: ops,
                y: y,
                embedding_weight: embedding_weight,
                params: p,
                reasoning_tracker: reasoning_tracker,
                profiler: profiler,
                max_new_tokens: max_new_tokens,
                eos_id: eos_id,
                generated_tokens: generated_tokens,
                token_history: token_history,
                finish_reason: finish_reason,
                first_token_instant: first_token_instant,
                report_perf: p.report_performance,
                generation_stream: generation_stream
            );

            // Export caches from C++ before CompiledResetGuard drops
            if p.reuse_cache {
                let num_layers = self.config.num_layers as usize;
                let mut export_ptrs: Vec<*mut mlx_sys::mlx_array> =
                    vec![std::ptr::null_mut(); num_layers * 2];
                let exported = unsafe {
                    mlx_sys::mlx_qwen35_export_caches(
                        export_ptrs.as_mut_ptr(),
                        (num_layers * 2) as i32,
                    )
                };
                if exported > 0 {
                    let cache_offset = unsafe { mlx_sys::mlx_qwen35_get_cache_offset() };
                    let mut new_caches = Vec::with_capacity(num_layers);
                    for i in 0..num_layers {
                        let p0 = export_ptrs[i * 2];
                        let p1 = export_ptrs[i * 2 + 1];
                        let mut lc = if self.config.is_linear_layer(i) {
                            Qwen3_5LayerCache::new_linear()
                        } else {
                            Qwen3_5LayerCache::new_full_attention()
                        };
                        lc.import_ptrs(p0, p1, cache_offset);
                        new_caches.push(lc);
                    }
                    self.caches = Some(new_caches);
                    // Force-materialize the exported cache arrays before
                    // `CompiledResetGuard` drops at end of scope and tears
                    // down `g_compiled_caches`. `mlx_qwen35_export_caches`
                    // hands back lazy `array` copies whose compute graph
                    // still references compiled-graph nodes; without this
                    // eval those handles point at buffers that get freed
                    // when the compile cache resets, and the next turn's
                    // compile init would feed stale handles to the GPU —
                    // triggering Metal page-faults / innocent-victim hangs
                    // on the first forward of the next turn.
                    eval_layer_caches(&self.caches);
                }
            }
        } else {
            profiler.set_label("chat_rust");

            MxArray::async_eval_arrays(&[&y]);

            let mut ops = chat_common::DecodeOps {
                forward: |ids: &MxArray, emb: &MxArray| -> Result<(MxArray, bool)> {
                    let logits = forward_inner(
                        ids,
                        emb,
                        &mut self.layers,
                        &mut self.caches,
                        &self.final_norm,
                        &self.lm_head,
                        Some(&embedding_weight_t),
                    )?;
                    Ok((logits, true))
                },
                eval_step: |token: &MxArray, logits: &MxArray, _budget_forced: bool| {
                    MxArray::async_eval_arrays(&[token, logits]);
                },
            };
            chat_common::decode_loop!(
                ops: ops,
                y: y,
                embedding_weight: embedding_weight,
                params: p,
                reasoning_tracker: reasoning_tracker,
                profiler: profiler,
                max_new_tokens: max_new_tokens,
                eos_id: eos_id,
                generated_tokens: generated_tokens,
                token_history: token_history,
                finish_reason: finish_reason,
                first_token_instant: first_token_instant,
                report_perf: p.report_performance,
                generation_stream: generation_stream
            );
        }

        // Save cache state
        save_cache_state_direct(
            p.reuse_cache,
            has_images,
            &generated_tokens,
            &finish_reason,
            &save_tokens,
            save_expanded_tokens.as_deref(),
            save_image_cache_key,
            &mut self.cached_token_history,
            &mut self.cached_image_key,
            &mut self.cached_rope_deltas,
            &mut self.caches,
        );

        let performance = compute_performance_metrics(
            generation_start,
            first_token_instant,
            prefill_tokens_len,
            generated_tokens.len(),
        );

        // `y` is the last sampled token from the decode loop. The
        // `decode_loop!` macro assigns to `y` each iteration and the final
        // assignment in the last iteration is never observed, which without
        // this explicit discard trips `clippy::unused_assignments` (the
        // macro repetition hides the usage pattern from the lint). Binding
        // here is cleaner than spraying `#[allow]` inside the macro body.
        let _final_sampled_token = y;

        finalize_chat_result(
            &tokenizer,
            &generated_tokens,
            finish_reason,
            think_end_id,
            think_end_str.as_deref(),
            performance,
            p.include_reasoning,
            enable_thinking.unwrap_or(true),
            prompt_tokens_for_result,
            reasoning_tracker.reasoning_token_count(),
        )
    }

    /// Streaming chat (session-start variant): same semantics as
    /// [`Self::chat_session_start_sync`] but streams token deltas through
    /// `stream_tx` rather than returning a `ChatResult`. Stops on
    /// `<|im_end|>` and resets caches before prefill.
    ///
    /// Images are accepted on session start — the downstream
    /// `chat_stream_sync_inner` already handles VLM prefill. Subsequent
    /// turns in the same session go through
    /// `chat_stream_session_continue_sync` which is text-only; changing
    /// the image set mid-session requires starting a new session.
    pub(crate) fn chat_stream_session_start_sync(
        &mut self,
        messages: Vec<ChatMessage>,
        config: ChatConfig,
        stream_tx: StreamTx<ChatStreamChunk>,
        cancelled: Arc<AtomicBool>,
    ) {
        let cb = StreamSender(stream_tx.clone());

        // Guard: respect cancellation before doing any work.
        if cancelled.load(Ordering::Relaxed) {
            send_stream_error(
                &stream_tx,
                "chat_stream_session_start cancelled before start",
            );
            return;
        }

        // Guard: reuse_cache must not be explicitly disabled.
        if config.reuse_cache == Some(false) {
            send_stream_error(
                &stream_tx,
                "chat_stream_session_start requires reuse_cache=true (leave as None or set to true). \
                 The session API only makes sense with cache reuse enabled.",
            );
            return;
        }

        // Resolve <|im_end|> for the eos override.
        let im_end_id = match self.tokenizer.as_ref().and_then(|t| t.im_end_id()) {
            Some(id) => id,
            None => {
                send_stream_error(
                    &stream_tx,
                    "chat_stream_session_start requires a tokenizer with an <|im_end|> special token",
                );
                return;
            }
        };

        // Full reset: the session always starts clean.
        if let Err(e) = self.reset_caches_sync() {
            let _ = stream_tx.send(Err(e));
            return;
        }
        if let Err(e) = self.init_caches_sync() {
            let _ = stream_tx.send(Err(e));
            return;
        }

        let result = self.chat_stream_sync_inner(messages, config, im_end_id, &cb, &cancelled);
        if let Err(e) = result {
            let _ = stream_tx.send(Err(e));
        }
    }

    /// Streaming chat (session-continue variant): same semantics as
    /// [`Self::chat_session_continue_sync`] but streams token deltas
    /// through `stream_tx`. Builds the ChatML delta, tokenizes it, and
    /// delegates to [`Self::chat_stream_tokens_delta_sync`].
    ///
    /// `images` is an opt-in guard parameter: non-empty input is
    /// rejected via `send_stream_error` with an
    /// `IMAGE_CHANGE_REQUIRES_SESSION_RESTART:`-prefixed message so
    /// the TS `ChatSession` layer can catch the prefix and route
    /// image-changes back through a fresh session start.
    pub(crate) fn chat_stream_session_continue_sync(
        &mut self,
        user_message: String,
        images: Option<Vec<Uint8Array>>,
        config: ChatConfig,
        stream_tx: StreamTx<ChatStreamChunk>,
        cancelled: Arc<AtomicBool>,
    ) {
        if cancelled.load(Ordering::Relaxed) {
            send_stream_error(
                &stream_tx,
                "chat_stream_session_continue cancelled before start",
            );
            return;
        }

        if images.as_ref().is_some_and(|v| !v.is_empty()) {
            send_stream_error(
                &stream_tx,
                &format!(
                    "{} chat_stream_session_continue is text-only; start a new session with chat_stream_session_start to change the image",
                    chat_common::IMAGE_CHANGE_RESTART_PREFIX
                ),
            );
            return;
        }

        let tokenizer = match self.tokenizer.as_ref() {
            Some(t) => t.clone(),
            None => {
                send_stream_error(&stream_tx, "Tokenizer not loaded");
                return;
            }
        };

        // Sanitize the user message the same way chat_session_continue_sync
        // does so the streaming path is subject to the same role/content
        // injection protection.
        let synthetic = build_synthetic_user_message(&user_message);
        let sanitized = Qwen3Tokenizer::sanitize_messages_public(std::slice::from_ref(&synthetic));
        let sanitized_user = &sanitized[0].content;

        let enable_thinking = chat_common::resolve_enable_thinking(&config);
        let delta_text = build_chatml_continue_delta_text(sanitized_user, enable_thinking);

        let delta_tokens = match tokenizer.encode_sync(&delta_text, Some(false)) {
            Ok(t) => t,
            Err(e) => {
                // Forward the tokenizer error as an mpsc send — this path
                // signals generic errors via an error-result send rather
                // than an error chunk, matching the `chat_stream_sync`
                // final-catch behavior.
                let _ = stream_tx.send(Err(e));
                return;
            }
        };

        self.chat_stream_tokens_delta_sync(delta_tokens, config, stream_tx, cancelled);
    }

    /// Streaming analog of [`Self::chat_session_continue_tool_sync`].
    ///
    /// Builds a ChatML tool-response delta, tokenizes it, and delegates
    /// to [`Self::chat_stream_tokens_delta_sync`]. Inherits the same
    /// text-only-delta invariant.
    pub(crate) fn chat_stream_session_continue_tool_sync(
        &mut self,
        tool_call_id: String,
        content: String,
        config: ChatConfig,
        stream_tx: StreamTx<ChatStreamChunk>,
        cancelled: Arc<AtomicBool>,
    ) {
        if cancelled.load(Ordering::Relaxed) {
            send_stream_error(
                &stream_tx,
                "chat_stream_session_continue_tool cancelled before start",
            );
            return;
        }

        let tokenizer = match self.tokenizer.as_ref() {
            Some(t) => t.clone(),
            None => {
                send_stream_error(&stream_tx, "Tokenizer not loaded");
                return;
            }
        };

        let enable_thinking = chat_common::resolve_enable_thinking(&config);
        let delta_text =
            chat_common::build_chatml_tool_delta_text(&tool_call_id, &content, enable_thinking);

        let delta_tokens = match tokenizer.encode_sync(&delta_text, Some(false)) {
            Ok(t) => t,
            Err(e) => {
                let _ = stream_tx.send(Err(e));
                return;
            }
        };

        self.chat_stream_tokens_delta_sync(delta_tokens, config, stream_tx, cancelled);
    }

    /// Streaming analog of [`Self::chat_tokens_delta_sync`]: prefill the
    /// caller-provided delta tokens on top of the existing KV caches and
    /// stream the reply through `stream_tx`.
    ///
    /// Applies the same four guards as the non-streaming path and —
    /// critically — still calls `save_cache_state_direct` at the end
    /// regardless of whether cancellation fired, so the cache stays
    /// consistent for the next turn even on an early abort.
    pub(crate) fn chat_stream_tokens_delta_sync(
        &mut self,
        delta_tokens: Vec<u32>,
        config: ChatConfig,
        stream_tx: StreamTx<ChatStreamChunk>,
        cancelled: Arc<AtomicBool>,
    ) {
        // Respect cancellation before any work.
        if cancelled.load(Ordering::Relaxed) {
            send_stream_error(
                &stream_tx,
                "chat_stream_tokens_delta cancelled before start",
            );
            return;
        }

        // --- Same four guards as chat_tokens_delta_sync ---
        if config.reuse_cache == Some(false) {
            send_stream_error(
                &stream_tx,
                "chat_stream_tokens_delta requires reuse_cache to be enabled; \
                 the delta path operates on session state by construction",
            );
            return;
        }
        if self.caches.is_none() {
            send_stream_error(
                &stream_tx,
                "chat_stream_tokens_delta requires an initialized session (call chatStreamSessionStart first)",
            );
            return;
        }
        if delta_tokens.is_empty() {
            send_stream_error(
                &stream_tx,
                "chat_stream_tokens_delta requires a non-empty delta",
            );
            return;
        }
        if self.cached_image_key.is_some() {
            send_stream_error(
                &stream_tx,
                "chat_stream_tokens_delta is text-only; session currently holds image state",
            );
            return;
        }

        // All guards passed — enter the prefill+decode helper. Any error
        // returned from here propagates as an mpsc error, same as
        // `chat_stream_sync`.
        let cb = StreamSender(stream_tx.clone());
        let result =
            self.chat_stream_tokens_delta_sync_inner(delta_tokens, config, &cb, &cancelled);
        if let Err(e) = result {
            let _ = stream_tx.send(Err(e));
        }
    }

    /// Prefill the delta tokens and run the streaming decode loop.
    ///
    /// This mirrors [`Self::chat_stream_sync_inner`] but skips the
    /// message rendering + prefix verification stages — the caller owns
    /// cache coherence by construction. Uses `<|im_end|>` as eos so the
    /// cached history continues to end on a clean ChatML boundary after
    /// the reply is saved.
    fn chat_stream_tokens_delta_sync_inner(
        &mut self,
        delta_tokens: Vec<u32>,
        config: ChatConfig,
        cb: &StreamSender,
        cancelled: &Arc<AtomicBool>,
    ) -> Result<()> {
        let reuse_cache = config.reuse_cache.unwrap_or(true);
        let report_perf = config.report_performance.unwrap_or(false);

        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))?
            .clone();

        // Session path: use <|im_end|> as eos, NOT config.eos_token_id.
        let eos_id = tokenizer
            .im_end_id()
            .ok_or_else(|| Error::from_reason("Tokenizer missing <|im_end|> special token"))?;

        let think_end_id = tokenizer.think_end_id();
        let think_end_str = tokenizer.think_end_str().map(|s| s.to_string());
        let tokenizer_for_decode = tokenizer.clone();

        // Build full token history = cached_history + delta.
        let mut full_token_history = self.cached_token_history.clone();
        full_token_history.extend(delta_tokens.iter().copied());

        let p = extract_chat_params(&config);
        let enable_thinking = chat_common::resolve_enable_thinking(&config);

        let generation_start = if report_perf {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut first_token_instant: Option<std::time::Instant> = None;

        let model_id = self.model_id;

        // Compiled path availability check, same pattern as chat_tokens_delta_sync.
        let use_compiled = unsafe { mlx_sys::mlx_qwen35_get_model_id() } == model_id;
        let _compiled_lock = if use_compiled {
            Some(
                DENSE_COMPILED_MUTEX
                    .lock()
                    .unwrap_or_else(|e| e.into_inner()),
            )
        } else {
            None
        };

        let mut _weight_guard = None;
        let use_compiled = if use_compiled {
            let guard = COMPILED_WEIGHTS_RWLOCK.read().unwrap();
            if unsafe { mlx_sys::mlx_qwen35_get_model_id() } == model_id {
                _weight_guard = Some(guard);
                true
            } else {
                false
            }
        } else {
            false
        };

        let embedding_weight = self.embedding.get_weight();
        let embedding_weight_t = embedding_weight.transpose(Some(&[1, 0]))?;
        let generation_stream = Stream::new(DeviceType::Gpu);
        let model_size_bytes = self.config.estimate_memory_bytes() as usize;
        let _wired_ctx =
            crate::stream::WiredLimitContext::new(model_size_bytes, vec![generation_stream]);

        let mut profiler =
            crate::decode_profiler::DecodeProfiler::new("chat_stream_delta", "qwen3_5");
        profiler.set_prompt_tokens(delta_tokens.len() as u32);
        profiler.snapshot_memory_before();

        // Text-only prefill of the delta on top of the existing caches.
        profiler.begin_prefill();
        let prompt = MxArray::from_uint32(&delta_tokens, &[1, delta_tokens.len() as i64])?;
        let logits = chunked_prefill(
            &prompt,
            &embedding_weight,
            &mut self.layers,
            &mut self.caches,
            &self.final_norm,
            &self.lm_head,
            Some(&embedding_weight_t),
            generation_stream,
        )?;
        let prefill_out_seq_len = logits.shape_at(1)?;
        let mut last_logits = logits.slice_axis(1, prefill_out_seq_len - 1, prefill_out_seq_len)?;
        last_logits = last_logits.squeeze(Some(&[1]))?;
        let seq_len = full_token_history.len() as i64;
        profiler.end_prefill();

        // Save snapshot for save_cache_state_direct (prior history + delta).
        let save_tokens = full_token_history.clone();

        // Decode setup
        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut finish_reason = String::from("length");
        let mut decode_stream = tokenizer_for_decode.inner().decode_stream(true);
        let mut streamed_text_len: usize = 0;

        let mut token_history: Vec<u32> = full_token_history;
        last_logits = apply_all_penalties(last_logits, &token_history, &p)?;
        let mut y = sample(&last_logits, p.sampling_config)?;
        MxArray::async_eval_arrays(&[&y]);

        let _compiled_guard = if use_compiled {
            Some(CompiledResetGuard)
        } else {
            None
        };

        let starts_in_thinking = enable_thinking.unwrap_or(true);
        let mut last_is_reasoning = starts_in_thinking;
        let mut reasoning_tracker = chat_common::ReasoningTracker::new(
            starts_in_thinking,
            p.thinking_token_budget,
            think_end_id,
        );

        if use_compiled {
            // Initialize compiled state from prefill — text-only path, no
            // VLM adjustments.
            use mlx_sys as sys;
            let prefill_len = seq_len as i32;
            let max_kv_len = ((prefill_len + p.max_new_tokens + 255) / 256) * 256;
            let num_layers = self.config.num_layers as usize;
            let mut cache_ptrs: Vec<*mut sys::mlx_array> =
                vec![std::ptr::null_mut(); num_layers * 2];
            if let Some(ref caches) = self.caches {
                for (i, cache) in caches.iter().enumerate() {
                    let (p0, p1) = cache.export_ptrs();
                    cache_ptrs[i * 2] = p0;
                    cache_ptrs[i * 2 + 1] = p1;
                }
            }
            unsafe {
                sys::mlx_qwen35_compiled_init_from_prefill(
                    self.config.num_layers,
                    self.config.hidden_size,
                    self.config.num_heads,
                    self.config.num_kv_heads,
                    self.config.head_dim,
                    self.config.rope_theta as f32,
                    self.config.rope_dims(),
                    self.config.rms_norm_eps as f32,
                    self.config.full_attention_interval,
                    self.config.linear_num_key_heads,
                    self.config.linear_num_value_heads,
                    self.config.linear_key_head_dim,
                    self.config.linear_value_head_dim,
                    self.config.linear_conv_kernel_dim,
                    if self.config.tie_word_embeddings {
                        1
                    } else {
                        0
                    },
                    max_kv_len,
                    1,
                    cache_ptrs.as_mut_ptr(),
                    prefill_len,
                );
            }
            // Text-only path: clear stale rope deltas.
            self.cached_rope_deltas = None;

            profiler.set_label("chat_stream_delta_compiled");

            let mut ops = chat_common::DecodeOps {
                forward: |ids: &MxArray, emb: &MxArray| -> Result<(MxArray, bool)> {
                    Ok((forward_compiled(ids, emb)?, false))
                },
                eval_step: |token: &MxArray, logits: &MxArray, budget_forced: bool| {
                    eval_token_and_compiled_caches(token);
                    if budget_forced {
                        logits.eval();
                    }
                },
            };
            chat_common::decode_loop!(
                ops: ops,
                y: y,
                embedding_weight: embedding_weight,
                params: p,
                reasoning_tracker: reasoning_tracker,
                profiler: profiler,
                max_new_tokens: p.max_new_tokens,
                eos_id: eos_id,
                generated_tokens: generated_tokens,
                token_history: token_history,
                finish_reason: finish_reason,
                first_token_instant: first_token_instant,
                report_perf: p.report_performance,
                generation_stream: generation_stream,
                streaming: {
                    callback: cb,
                    cancelled: cancelled,
                    decode_stream: decode_stream,
                    tokenizer: tokenizer_for_decode,
                    streamed_text_len: streamed_text_len,
                    last_is_reasoning: last_is_reasoning
                }
            );

            // Export caches from C++ so the next turn's Rust-side cache
            // state is consistent with the compiled forward's view.
            if reuse_cache {
                let num_layers = self.config.num_layers as usize;
                let mut export_ptrs: Vec<*mut mlx_sys::mlx_array> =
                    vec![std::ptr::null_mut(); num_layers * 2];
                let exported = unsafe {
                    mlx_sys::mlx_qwen35_export_caches(
                        export_ptrs.as_mut_ptr(),
                        (num_layers * 2) as i32,
                    )
                };
                if exported > 0 {
                    let cache_offset = unsafe { mlx_sys::mlx_qwen35_get_cache_offset() };
                    let mut new_caches = Vec::with_capacity(num_layers);
                    for i in 0..num_layers {
                        let p0 = export_ptrs[i * 2];
                        let p1 = export_ptrs[i * 2 + 1];
                        let mut lc = if self.config.is_linear_layer(i) {
                            Qwen3_5LayerCache::new_linear()
                        } else {
                            Qwen3_5LayerCache::new_full_attention()
                        };
                        lc.import_ptrs(p0, p1, cache_offset);
                        new_caches.push(lc);
                    }
                    self.caches = Some(new_caches);
                    // See `chat_with_caches_inner` for rationale: force-eval
                    // the exported lazy handles before `CompiledResetGuard`
                    // clears `g_compiled_caches` at end of scope.
                    eval_layer_caches(&self.caches);
                }
            }
        } else {
            profiler.set_label("chat_stream_delta_rust");

            let mut ops = chat_common::DecodeOps {
                forward: |ids: &MxArray, emb: &MxArray| -> Result<(MxArray, bool)> {
                    let logits = forward_inner(
                        ids,
                        emb,
                        &mut self.layers,
                        &mut self.caches,
                        &self.final_norm,
                        &self.lm_head,
                        Some(&embedding_weight_t),
                    )?;
                    Ok((logits, true))
                },
                eval_step: |token: &MxArray, logits: &MxArray, _budget_forced: bool| {
                    MxArray::async_eval_arrays(&[token, logits]);
                },
            };
            chat_common::decode_loop!(
                ops: ops,
                y: y,
                embedding_weight: embedding_weight,
                params: p,
                reasoning_tracker: reasoning_tracker,
                profiler: profiler,
                max_new_tokens: p.max_new_tokens,
                eos_id: eos_id,
                generated_tokens: generated_tokens,
                token_history: token_history,
                finish_reason: finish_reason,
                first_token_instant: first_token_instant,
                report_perf: p.report_performance,
                generation_stream: generation_stream,
                streaming: {
                    callback: cb,
                    cancelled: cancelled,
                    decode_stream: decode_stream,
                    tokenizer: tokenizer_for_decode,
                    streamed_text_len: streamed_text_len,
                    last_is_reasoning: last_is_reasoning
                }
            );
        }

        // Save cache state unconditionally — even on cancellation, the
        // partial generated_tokens must be appended so the session stays
        // consistent for the next turn. `has_images` is always false on
        // the delta path and we pass the full pre-decode snapshot as the
        // text-only `save_tokens`.
        save_cache_state_direct(
            p.reuse_cache,
            false,
            &generated_tokens,
            &finish_reason,
            &save_tokens,
            None,
            0,
            &mut self.cached_token_history,
            &mut self.cached_image_key,
            &mut self.cached_rope_deltas,
            &mut self.caches,
        );

        // Decode the full reply text and emit the final done chunk.
        let text = tokenizer_for_decode
            .decode_sync(&generated_tokens, true)
            .unwrap_or_else(|e| {
                warn!("Failed to decode generated tokens: {}", e);
                String::new()
            });

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
                    is_reasoning: Some(last_is_reasoning),
                }),
                ThreadsafeFunctionCallMode::NonBlocking,
            );
        }

        let num_tokens = generated_tokens.len() as u32;
        let prompt_token_count = delta_tokens.len() as u32;

        let (clean_text, tool_calls, thinking) = chat_common::parse_thinking_and_tools(
            &text,
            &generated_tokens,
            enable_thinking.unwrap_or(true),
            think_end_id,
            think_end_str.as_deref(),
            p.include_reasoning,
        );

        let finish_reason = if tool_calls.iter().any(|tc| tc.status == "ok") {
            "tool_calls".to_string()
        } else {
            finish_reason
        };

        let perf_metrics = compute_performance_metrics(
            generation_start,
            first_token_instant,
            delta_tokens.len(),
            generated_tokens.len(),
        );

        cb.call(
            Ok(ChatStreamChunk {
                text: clean_text,
                done: true,
                finish_reason: Some(finish_reason),
                tool_calls: Some(tool_calls),
                thinking,
                num_tokens: Some(num_tokens),
                prompt_tokens: Some(prompt_token_count),
                reasoning_tokens: Some(reasoning_tracker.reasoning_token_count()),
                raw_text: Some(text),
                performance: perf_metrics,
                is_reasoning: None,
            }),
            ThreadsafeFunctionCallMode::NonBlocking,
        );

        Ok(())
    }

    fn chat_stream_sync_inner(
        &mut self,
        messages: Vec<ChatMessage>,
        config: ChatConfig,
        eos_token_id: u32,
        cb: &StreamSender,
        cancelled: &Arc<AtomicBool>,
    ) -> Result<()> {
        let reuse_cache = config.reuse_cache.unwrap_or(true);
        let report_perf = config.report_performance.unwrap_or(false);

        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))?
            .clone();

        let has_images = messages
            .iter()
            .any(|m| m.images.as_ref().is_some_and(|imgs| !imgs.is_empty()));

        let think_end_id = tokenizer.think_end_id();
        let think_end_str = tokenizer.think_end_str().map(|s| s.to_string());
        let tokenizer_for_decode = tokenizer.clone();

        let tool_defs = config.tools.as_deref();
        let enable_thinking = chat_common::resolve_enable_thinking(&config);
        let tokens = tokenizer.apply_chat_template_sync(
            &messages,
            Some(true),
            tool_defs,
            enable_thinking,
        )?;

        let p = chat_common::extract_chat_params(&config);
        let model_id = self.model_id;

        let generation_start = if report_perf {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut first_token_instant: Option<std::time::Instant> = None;

        // Check compiled path
        let use_compiled = unsafe { mlx_sys::mlx_qwen35_get_model_id() } == model_id;
        let _compiled_lock = if use_compiled {
            Some(
                DENSE_COMPILED_MUTEX
                    .lock()
                    .unwrap_or_else(|e| e.into_inner()),
            )
        } else {
            None
        };

        let mut _weight_guard = None;
        let use_compiled = if use_compiled {
            let guard = COMPILED_WEIGHTS_RWLOCK.read().unwrap();
            if unsafe { mlx_sys::mlx_qwen35_get_model_id() } == model_id {
                _weight_guard = Some(guard);
                true
            } else {
                false
            }
        } else {
            false
        };

        let embedding_weight = self.embedding.get_weight();

        // VLM image processing
        let sms = self.spatial_merge_size.unwrap_or(2);
        let (expanded_tokens, current_image_cache_key, vlm_processed) = if has_images {
            if let (Some(_vision_enc), Some(img_proc)) =
                (self.vision_encoder.as_ref(), self.image_processor.as_ref())
            {
                let all_images = extract_images_from_messages(&messages);
                let image_refs: Vec<&[u8]> = all_images.iter().map(|v| v.as_slice()).collect();
                let processed_pre = img_proc.process_many(&image_refs)?;
                let num_image_tokens = compute_num_image_tokens(&processed_pre.grid_thw(), sms)?;
                let expanded = inject_image_placeholders(&tokens, num_image_tokens);
                let cache_key = compute_image_cache_key(&all_images);
                (expanded, cache_key, Some(processed_pre))
            } else {
                (tokens.clone(), 0u64, None)
            }
        } else {
            (tokens.clone(), 0u64, None)
        };

        // Cache reuse
        let cached_prefix_len = verify_cache_prefix_direct(
            reuse_cache,
            has_images,
            &tokens,
            &expanded_tokens,
            current_image_cache_key,
            &self.cached_token_history,
            &self.cached_image_key,
            self.caches.is_some(),
        );

        let prefill_tokens = if cached_prefix_len > 0 {
            if has_images {
                info!(
                    "VLM cache reuse: {} cached tokens, {} new tokens to prefill",
                    cached_prefix_len,
                    expanded_tokens.len() - cached_prefix_len
                );
                expanded_tokens[cached_prefix_len..].to_vec()
            } else {
                info!(
                    "Cache reuse: {} cached tokens, {} new tokens to prefill",
                    cached_prefix_len,
                    tokens.len() - cached_prefix_len
                );
                tokens[cached_prefix_len..].to_vec()
            }
        } else {
            if let Some(ref mut caches) = self.caches {
                for cache in caches.iter_mut() {
                    cache.reset();
                }
            }
            let new_caches = (0..self.config.num_layers as usize)
                .map(|i| {
                    if self.config.is_linear_layer(i) {
                        Qwen3_5LayerCache::new_linear()
                    } else {
                        Qwen3_5LayerCache::new_full_attention()
                    }
                })
                .collect();
            self.caches = Some(new_caches);
            tokens.clone()
        };

        // Zero-delta guard
        let (prefill_tokens, cached_prefix_len) = if prefill_tokens.is_empty() {
            info!("Zero-delta cache hit: resetting caches for full re-prefill");
            if let Some(ref mut caches) = self.caches {
                for cache in caches.iter_mut() {
                    cache.reset();
                }
            }
            let new_caches = (0..self.config.num_layers as usize)
                .map(|i| {
                    if self.config.is_linear_layer(i) {
                        Qwen3_5LayerCache::new_linear()
                    } else {
                        Qwen3_5LayerCache::new_full_attention()
                    }
                })
                .collect();
            self.caches = Some(new_caches);
            let tokens = if has_images {
                expanded_tokens.clone()
            } else {
                tokens.clone()
            };
            (tokens, 0)
        } else {
            (prefill_tokens, cached_prefix_len)
        };

        let eos_id = eos_token_id;
        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut finish_reason = String::from("length");
        let mut decode_stream = tokenizer_for_decode.inner().decode_stream(true);
        let mut streamed_text_len: usize = 0;

        let embedding_weight_t = embedding_weight.transpose(Some(&[1, 0]))?;
        let generation_stream = Stream::new(DeviceType::Gpu);
        let model_size_bytes = self.config.estimate_memory_bytes() as usize;
        let _wired_ctx =
            crate::stream::WiredLimitContext::new(model_size_bytes, vec![generation_stream]);

        let mut profiler = crate::decode_profiler::DecodeProfiler::new("chat_stream", "qwen3_5");
        profiler.set_prompt_tokens(prefill_tokens.len() as u32);
        profiler.snapshot_memory_before();

        // VLM or text prefill
        profiler.begin_prefill();
        let (mut last_logits, seq_len, vlm_compiled_init_done) = if has_images
            && cached_prefix_len == 0
        {
            if let Some(vision_enc) = self.vision_encoder.clone() {
                let final_tokens = &expanded_tokens;
                let processed = vlm_processed
                    .as_ref()
                    .ok_or_else(|| Error::from_reason("VLM processed images missing"))?;

                let input_ids =
                    MxArray::from_uint32(final_tokens, &[1, final_tokens.len() as i64])?;

                let (logits, rope_deltas, vlm_compiled) = vlm_prefill(
                    &input_ids,
                    current_image_cache_key,
                    processed,
                    &vision_enc,
                    sms,
                    &embedding_weight,
                    &mut self.layers,
                    &mut self.caches,
                    &self.final_norm,
                    &self.lm_head,
                    &self.config,
                    p.max_new_tokens,
                    generation_stream,
                    &self.vision_cache,
                    model_id,
                )?;

                self.cached_rope_deltas = Some(rope_deltas as i32);
                let vlm_seq_len = final_tokens.len() as i64;
                (logits, vlm_seq_len, vlm_compiled)
            } else {
                return Err(Error::from_reason(
                    "VLM prefill requested but vision encoder/processor not loaded",
                ));
            }
        } else {
            let prompt = MxArray::from_uint32(&prefill_tokens, &[1, prefill_tokens.len() as i64])?;
            let logits = chunked_prefill(
                &prompt,
                &embedding_weight,
                &mut self.layers,
                &mut self.caches,
                &self.final_norm,
                &self.lm_head,
                Some(&embedding_weight_t),
                generation_stream,
            )?;

            let seq_len = logits.shape_at(1)?;
            let last_logits = logits.slice_axis(1, seq_len - 1, seq_len)?;
            let last_logits = last_logits.squeeze(Some(&[1]))?;
            let total_seq_len = if has_images {
                expanded_tokens.len() as i64
            } else {
                tokens.len() as i64
            };
            (last_logits, total_seq_len, false)
        };
        profiler.end_prefill();

        let mut token_history: Vec<u32> = tokens.clone();
        last_logits = apply_all_penalties(last_logits, &token_history, &p)?;
        let mut y = sample(&last_logits, p.sampling_config)?;
        MxArray::async_eval_arrays(&[&y]);

        let _compiled_guard = if use_compiled {
            Some(CompiledResetGuard)
        } else {
            None
        };

        let starts_in_thinking = enable_thinking.unwrap_or(true);
        let mut last_is_reasoning = starts_in_thinking;
        let mut reasoning_tracker = chat_common::ReasoningTracker::new(
            starts_in_thinking,
            p.thinking_token_budget,
            think_end_id,
        );

        if use_compiled {
            if vlm_compiled_init_done {
                // VLM prefill already initialized compiled state
            } else {
                use mlx_sys as sys;
                let prefill_len = seq_len as i32;
                let max_kv_len = ((prefill_len + p.max_new_tokens + 255) / 256) * 256;
                let num_layers = self.config.num_layers as usize;
                let mut cache_ptrs: Vec<*mut sys::mlx_array> =
                    vec![std::ptr::null_mut(); num_layers * 2];
                if let Some(ref caches) = self.caches {
                    for (i, cache) in caches.iter().enumerate() {
                        let (p0, p1) = cache.export_ptrs();
                        cache_ptrs[i * 2] = p0;
                        cache_ptrs[i * 2 + 1] = p1;
                    }
                }
                unsafe {
                    sys::mlx_qwen35_compiled_init_from_prefill(
                        self.config.num_layers,
                        self.config.hidden_size,
                        self.config.num_heads,
                        self.config.num_kv_heads,
                        self.config.head_dim,
                        self.config.rope_theta as f32,
                        self.config.rope_dims(),
                        self.config.rms_norm_eps as f32,
                        self.config.full_attention_interval,
                        self.config.linear_num_key_heads,
                        self.config.linear_num_value_heads,
                        self.config.linear_key_head_dim,
                        self.config.linear_value_head_dim,
                        self.config.linear_conv_kernel_dim,
                        if self.config.tie_word_embeddings {
                            1
                        } else {
                            0
                        },
                        max_kv_len,
                        1,
                        cache_ptrs.as_mut_ptr(),
                        prefill_len,
                    );
                }

                if has_images
                    && cached_prefix_len > 0
                    && let Some(delta) = self.cached_rope_deltas
                {
                    unsafe {
                        mlx_sys::mlx_qwen35_compiled_adjust_offset(delta);
                    }
                }
            }

            if !has_images {
                self.cached_rope_deltas = None;
            }

            profiler.set_label("chat_stream_compiled");

            let mut ops = chat_common::DecodeOps {
                forward: |ids: &MxArray, emb: &MxArray| -> Result<(MxArray, bool)> {
                    Ok((forward_compiled(ids, emb)?, false))
                },
                eval_step: |token: &MxArray, logits: &MxArray, budget_forced: bool| {
                    eval_token_and_compiled_caches(token);
                    if budget_forced {
                        logits.eval();
                    }
                },
            };
            chat_common::decode_loop!(
                ops: ops,
                y: y,
                embedding_weight: embedding_weight,
                params: p,
                reasoning_tracker: reasoning_tracker,
                profiler: profiler,
                max_new_tokens: p.max_new_tokens,
                eos_id: eos_id,
                generated_tokens: generated_tokens,
                token_history: token_history,
                finish_reason: finish_reason,
                first_token_instant: first_token_instant,
                report_perf: p.report_performance,
                generation_stream: generation_stream,
                streaming: {
                    callback: cb,
                    cancelled: cancelled,
                    decode_stream: decode_stream,
                    tokenizer: tokenizer_for_decode,
                    streamed_text_len: streamed_text_len,
                    last_is_reasoning: last_is_reasoning
                }
            );

            // Export caches
            if reuse_cache {
                let num_layers = self.config.num_layers as usize;
                let mut export_ptrs: Vec<*mut mlx_sys::mlx_array> =
                    vec![std::ptr::null_mut(); num_layers * 2];
                let exported = unsafe {
                    mlx_sys::mlx_qwen35_export_caches(
                        export_ptrs.as_mut_ptr(),
                        (num_layers * 2) as i32,
                    )
                };
                if exported > 0 {
                    let cache_offset = unsafe { mlx_sys::mlx_qwen35_get_cache_offset() };
                    let mut new_caches = Vec::with_capacity(num_layers);
                    for i in 0..num_layers {
                        let p0 = export_ptrs[i * 2];
                        let p1 = export_ptrs[i * 2 + 1];
                        let mut lc = if self.config.is_linear_layer(i) {
                            Qwen3_5LayerCache::new_linear()
                        } else {
                            Qwen3_5LayerCache::new_full_attention()
                        };
                        lc.import_ptrs(p0, p1, cache_offset);
                        new_caches.push(lc);
                    }
                    self.caches = Some(new_caches);
                    // See `chat_with_caches_inner` for rationale: force-eval
                    // the exported lazy handles before `CompiledResetGuard`
                    // clears `g_compiled_caches` at end of scope.
                    eval_layer_caches(&self.caches);
                }
            }
        } else {
            profiler.set_label("chat_stream_rust");

            let mut ops = chat_common::DecodeOps {
                forward: |ids: &MxArray, emb: &MxArray| -> Result<(MxArray, bool)> {
                    let logits = forward_inner(
                        ids,
                        emb,
                        &mut self.layers,
                        &mut self.caches,
                        &self.final_norm,
                        &self.lm_head,
                        Some(&embedding_weight_t),
                    )?;
                    Ok((logits, true))
                },
                eval_step: |token: &MxArray, logits: &MxArray, _budget_forced: bool| {
                    MxArray::async_eval_arrays(&[token, logits]);
                },
            };
            chat_common::decode_loop!(
                ops: ops,
                y: y,
                embedding_weight: embedding_weight,
                params: p,
                reasoning_tracker: reasoning_tracker,
                profiler: profiler,
                max_new_tokens: p.max_new_tokens,
                eos_id: eos_id,
                generated_tokens: generated_tokens,
                token_history: token_history,
                finish_reason: finish_reason,
                first_token_instant: first_token_instant,
                report_perf: p.report_performance,
                generation_stream: generation_stream,
                streaming: {
                    callback: cb,
                    cancelled: cancelled,
                    decode_stream: decode_stream,
                    tokenizer: tokenizer_for_decode,
                    streamed_text_len: streamed_text_len,
                    last_is_reasoning: last_is_reasoning
                }
            );
        }

        // Save cache state
        save_cache_state_direct(
            p.reuse_cache,
            has_images,
            &generated_tokens,
            &finish_reason,
            &tokens,
            Some(&expanded_tokens),
            current_image_cache_key,
            &mut self.cached_token_history,
            &mut self.cached_image_key,
            &mut self.cached_rope_deltas,
            &mut self.caches,
        );

        let text = tokenizer_for_decode
            .decode_sync(&generated_tokens, true)
            .unwrap_or_else(|e| {
                warn!("Failed to decode generated tokens: {}", e);
                String::new()
            });

        // Flush residual bytes
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
                    is_reasoning: Some(last_is_reasoning),
                }),
                ThreadsafeFunctionCallMode::NonBlocking,
            );
        }

        let num_tokens = generated_tokens.len() as u32;
        let prompt_token_count = if has_images {
            expanded_tokens.len() as u32
        } else {
            tokens.len() as u32
        };

        let (clean_text, tool_calls, thinking) = chat_common::parse_thinking_and_tools(
            &text,
            &generated_tokens,
            enable_thinking.unwrap_or(true),
            think_end_id,
            think_end_str.as_deref(),
            p.include_reasoning,
        );

        let finish_reason = if tool_calls.iter().any(|tc| tc.status == "ok") {
            "tool_calls".to_string()
        } else {
            finish_reason
        };

        let perf_metrics = compute_performance_metrics(
            generation_start,
            first_token_instant,
            prefill_tokens.len(),
            generated_tokens.len(),
        );

        // Send final done chunk
        cb.call(
            Ok(ChatStreamChunk {
                text: clean_text,
                done: true,
                finish_reason: Some(finish_reason),
                tool_calls: Some(tool_calls),
                thinking,
                num_tokens: Some(num_tokens),
                prompt_tokens: Some(prompt_token_count),
                reasoning_tokens: Some(reasoning_tracker.reasoning_token_count()),
                raw_text: Some(text),
                performance: perf_metrics,
                is_reasoning: None,
            }),
            ThreadsafeFunctionCallMode::NonBlocking,
        );

        Ok(())
    }

    /// Generate text from prompt tokens (synchronous, runs on model thread).
    pub(crate) fn generate_sync(
        &mut self,
        prompt_tokens: MxArray,
        config: Qwen3_5GenerationConfig,
    ) -> Result<Qwen3_5GenerationResult> {
        let tokenizer = self.tokenizer.clone();

        // Init caches
        self.init_caches_sync()?;

        let embedding_weight = self.embedding.get_weight();
        let embedding_weight_t = embedding_weight.transpose(Some(&[1, 0]))?;
        let generation_stream = Stream::new(DeviceType::Gpu);

        // Prefill
        let prompt = prompt_tokens.reshape(&[1, -1])?;
        let logits = {
            let _stream_ctx = StreamContext::new(generation_stream);
            forward_inner(
                &prompt,
                &embedding_weight,
                &mut self.layers,
                &mut self.caches,
                &self.final_norm,
                &self.lm_head,
                Some(&embedding_weight_t),
            )?
        };

        let seq_len = logits.shape_at(1)?;
        let last_logits = logits.slice_axis(1, seq_len - 1, seq_len)?;
        let last_logits = last_logits.squeeze(Some(&[1]))?;

        let sampling_config = Some(SamplingConfig {
            temperature: config.temperature,
            top_k: config.top_k,
            top_p: config.top_p,
            min_p: config.min_p,
        });

        let eos_id = self.config.eos_token_id as u32;
        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut y = sample(&last_logits, sampling_config)?;

        for _step in 0..config.max_new_tokens {
            y.eval();
            let token_id = y.item_at_int32(0)? as u32;
            generated_tokens.push(token_id);

            if token_id == eos_id {
                break;
            }

            let next_ids = y.reshape(&[1, 1])?;
            let logits = {
                let _stream_ctx = StreamContext::new(generation_stream);
                forward_inner(
                    &next_ids,
                    &embedding_weight,
                    &mut self.layers,
                    &mut self.caches,
                    &self.final_norm,
                    &self.lm_head,
                    Some(&embedding_weight_t),
                )?
            };

            let logits = logits.squeeze(Some(&[1]))?;
            y = sample(&logits, sampling_config)?;
            MxArray::async_eval_arrays(&[&y]);

            if (_step + 1) % 256 == 0 {
                crate::array::synchronize_and_clear_cache();
            }
        }

        self.reset_caches_sync()?;

        let finish_reason = if generated_tokens.last().is_some_and(|&t| t == eos_id) {
            "stop"
        } else {
            "length"
        };

        let text = if let Some(ref tok) = tokenizer {
            tok.decode_sync(&generated_tokens, true).unwrap_or_default()
        } else {
            String::new()
        };

        Ok(Qwen3_5GenerationResult {
            tokens: generated_tokens.clone(),
            text,
            num_tokens: generated_tokens.len() as u32,
            finish_reason: finish_reason.to_string(),
        })
    }

    // ========== Training methods (run on model thread) ==========

    /// Initialize training state with optimizer and configuration.
    fn init_training_sync(
        &mut self,
        config: crate::grpo::engine::GRPOEngineConfig,
        _model_type: crate::training_model::ModelType,
    ) -> Result<()> {
        if self.training_state.is_some() {
            return Err(napi::Error::from_reason(
                "Training state already initialized. A single model thread can host only one active training run.",
            ));
        }
        let optimizer = if config.optimizer_type.as_deref().unwrap_or("adamw") == "adamw" {
            Some(crate::optimizers::AdamW::new(
                config.learning_rate,
                config.adamw_beta1,
                config.adamw_beta2,
                config.adamw_eps,
                config.weight_decay,
                Some(true), // bias correction
            ))
        } else {
            None
        };

        self.training_state = Some(crate::training_state::ModelThreadTrainingState::new(
            config.learning_rate.unwrap_or(1e-6),
            config.gradient_accumulation_steps.unwrap_or(1),
            config.gradient_clip_norm,
            config.gradient_clip_value,
            config.max_nan_gradients.unwrap_or(100),
            config.emergency_save_threshold.unwrap_or(5),
            config.verbose_nan_detection.unwrap_or(false),
            config.gradient_checkpointing.unwrap_or(true),
            optimizer,
        ));
        info!("Training state initialized on model thread (Qwen3.5 Dense)");
        Ok(())
    }

    fn save_optimizer_state_sync(&self, path: String) -> Result<()> {
        let ts = self.training_state.as_ref().ok_or_else(|| {
            napi::Error::from_reason("Training state not initialized. Call InitTraining first.")
        })?;
        ts.save_optimizer_state_sync(&path)
    }

    fn load_optimizer_state_sync(&mut self, path: String) -> Result<()> {
        let ts = self.training_state.as_mut().ok_or_else(|| {
            napi::Error::from_reason("Training state not initialized. Call InitTraining first.")
        })?;
        ts.load_optimizer_state_sync(&path)
    }

    /// Generate completions for training.
    ///
    /// Tokenizes prompts using Jinja2 chat template, generates completions,
    /// caches MxArray results in training_state for the subsequent training step,
    /// and returns plain data across the thread boundary.
    fn generate_for_training_thread_sync(
        &mut self,
        prompts: Vec<Vec<ChatMessage>>,
        group_size: usize,
        gen_config: crate::models::qwen3::GenerationConfig,
        enable_thinking: Option<bool>,
        tools: Option<Vec<ToolDefinition>>,
    ) -> Result<crate::training_model::GenerationPlainData> {
        use crate::array::heavy_cleanup;

        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason("Tokenizer not available."))?
            .clone();

        let num_prompts = prompts.len();
        let total_completions = num_prompts * group_size;

        let mut completion_texts = Vec::with_capacity(total_completions);
        let mut prompt_texts = Vec::with_capacity(total_completions);
        let mut completion_tokens_plain = Vec::with_capacity(total_completions);
        let mut completion_logprobs_plain = Vec::with_capacity(total_completions);
        let mut token_counts = Vec::with_capacity(total_completions);
        let mut finish_reasons = Vec::with_capacity(total_completions);

        // Cache MxArrays for the training step (prompt-major layout)
        let mut cached_prompt_tokens: Vec<MxArray> = Vec::with_capacity(num_prompts);
        let mut cached_completion_tokens: Vec<MxArray> = Vec::with_capacity(total_completions);
        let mut cached_completion_logprobs: Vec<MxArray> = Vec::with_capacity(total_completions);

        for prompt_messages in prompts.iter() {
            // Tokenize the prompt using Jinja2 chat template (supports tools + thinking)
            let prompt_token_ids = tokenizer.apply_chat_template_sync(
                prompt_messages,
                Some(true),
                tools.as_deref(),
                enable_thinking,
            )?;

            let prompt_array =
                MxArray::from_uint32(&prompt_token_ids, &[1, prompt_token_ids.len() as i64])?;
            let prompt_array_1d = prompt_array.squeeze(Some(&[0]))?;
            let prompt_text = tokenizer.decode_sync(&prompt_token_ids, true)?;

            // Generate group_size completions for this prompt
            for _g in 0..group_size {
                let result = self
                    .generate_single_for_training_sync(&prompt_array, Some(gen_config.clone()))?;

                // Extract plain data for crossing thread boundary
                let tok_ids: Vec<i32> = result
                    .tokens
                    .to_uint32()?
                    .iter()
                    .map(|&t| t as i32)
                    .collect();
                let lp_data: Vec<f32> = result.logprobs.to_float32()?.to_vec();
                let decoded = tokenizer
                    .decode_sync(&tok_ids.iter().map(|&t| t as u32).collect::<Vec<_>>(), true)?;

                completion_texts.push(decoded);
                prompt_texts.push(prompt_text.clone());
                completion_tokens_plain.push(tok_ids);
                completion_logprobs_plain.push(lp_data);
                token_counts.push(result.num_tokens as u32);
                finish_reasons.push(result.finish_reason.clone());

                // Cache MxArrays (these stay on the model thread)
                cached_completion_tokens.push(result.tokens);
                cached_completion_logprobs.push(result.logprobs);

                // Clean up between completions to prevent Metal context accumulation
                heavy_cleanup();
            }

            cached_prompt_tokens.push(prompt_array_1d);
        }

        // Store cached MxArrays in training_state (prompt-major layout)
        if let Some(ref mut ts) = self.training_state {
            ts.cached_prompt_tokens = Some(cached_prompt_tokens);
            ts.cached_completion_tokens = Some(cached_completion_tokens);
            ts.cached_completion_logprobs = Some(cached_completion_logprobs);
        }

        Ok(crate::training_model::GenerationPlainData {
            completion_texts,
            prompt_texts,
            completion_tokens: completion_tokens_plain,
            completion_logprobs: completion_logprobs_plain,
            token_counts,
            finish_reasons,
        })
    }

    /// Generate a single completion for training purposes.
    ///
    /// Uses fresh local KV caches (not the shared inference caches).
    /// Returns GenerationResult with MxArray tokens and logprobs.
    fn generate_single_for_training_sync(
        &mut self,
        input_ids: &MxArray,
        config: Option<crate::models::qwen3::GenerationConfig>,
    ) -> Result<crate::models::qwen3::GenerationResult> {
        use crate::array::synchronize_and_clear_cache;
        use crate::sampling::{
            apply_frequency_penalty, apply_presence_penalty, apply_repetition_penalty,
            check_repetition_cutoff, sample_and_logprobs,
        };

        let config = config.unwrap_or_default();
        let max_new_tokens = config.max_new_tokens.unwrap_or(2048);
        let temperature = config.temperature.unwrap_or(1.0);
        let top_k = config.top_k.unwrap_or(0);
        let top_p = config.top_p.unwrap_or(1.0);
        let min_p = config.min_p.unwrap_or(0.0);
        let repetition_penalty = config.repetition_penalty.unwrap_or(1.0);
        let repetition_context_size = config.repetition_context_size.unwrap_or(256);
        let presence_penalty = config.presence_penalty.unwrap_or(0.0);
        let presence_context_size = config.presence_context_size.unwrap_or(20);
        let frequency_penalty = config.frequency_penalty.unwrap_or(0.0);
        let frequency_context_size = config.frequency_context_size.unwrap_or(20);
        let max_consecutive_tokens = config.max_consecutive_tokens.unwrap_or(16);
        let max_ngram_repeats = config.max_ngram_repeats.unwrap_or(3);
        let ngram_size = config.ngram_size.unwrap_or(64);
        let eos_token_id = config.eos_token_id.or(Some(self.config.eos_token_id));
        let return_logprobs = config.return_logprobs.unwrap_or(true);

        let embedding_weight = self.embedding.get_weight();
        let embedding_weight_t = embedding_weight.transpose(Some(&[1, 0]))?;
        let generation_stream = Stream::new(DeviceType::Gpu);

        // Use fresh caches for training (not shared inference caches)
        let mut training_caches: Option<Vec<Qwen3_5LayerCache>> = Some(
            (0..self.config.num_layers as usize)
                .map(|i| {
                    if self.config.is_linear_layer(i) {
                        Qwen3_5LayerCache::new_linear()
                    } else {
                        Qwen3_5LayerCache::new_full_attention()
                    }
                })
                .collect(),
        );

        let input_tokens = input_ids.to_uint32()?;
        let current_ids = input_ids.clone();
        let mut generated_tokens: Vec<u32> = Vec::with_capacity(max_new_tokens as usize);
        let mut generated_logprobs: Vec<f32> = if return_logprobs {
            Vec::with_capacity(max_new_tokens as usize)
        } else {
            Vec::new()
        };
        let mut finish_reason = "length";

        let sampling_config = SamplingConfig {
            temperature: Some(temperature),
            top_k: Some(top_k),
            top_p: Some(top_p),
            min_p: Some(min_p),
        };

        // PREFILL
        let mut last_logits = {
            let logits = {
                let _stream_ctx = StreamContext::new(generation_stream);
                forward_inner(
                    &current_ids,
                    &embedding_weight,
                    &mut self.layers,
                    &mut training_caches,
                    &self.final_norm,
                    &self.lm_head,
                    Some(&embedding_weight_t),
                )?
            };
            let seq_len = logits.shape_at(1)?;
            logits
                .slice_axis(1, seq_len - 1, seq_len)?
                .squeeze(Some(&[0, 1]))?
        };

        if repetition_penalty != 1.0 && !input_tokens.is_empty() {
            last_logits = apply_repetition_penalty(
                &last_logits,
                &input_tokens,
                repetition_penalty,
                Some(repetition_context_size),
            )?;
        }
        if presence_penalty != 0.0 {
            last_logits = apply_presence_penalty(
                &last_logits,
                &input_tokens,
                presence_penalty,
                Some(presence_context_size),
            )?;
        }
        if frequency_penalty != 0.0 {
            last_logits = apply_frequency_penalty(
                &last_logits,
                &input_tokens,
                frequency_penalty,
                Some(frequency_context_size),
            )?;
        }

        let (mut token, mut logprobs) = if return_logprobs {
            let (tok, lp) = sample_and_logprobs(&last_logits, Some(sampling_config))?;
            (tok, Some(lp))
        } else {
            (sample(&last_logits, Some(sampling_config))?, None)
        };

        // DECODE
        const DECODE_CLEANUP_INTERVAL: i32 = 256;
        for step in 0..max_new_tokens {
            let _stream_ctx = StreamContext::new(generation_stream);
            token.eval();
            if step > 0 && step % DECODE_CLEANUP_INTERVAL == 0 {
                synchronize_and_clear_cache();
            }
            let token_value = token.item_at_int32(0)? as u32;
            generated_tokens.push(token_value);
            if return_logprobs && let Some(ref lp) = logprobs {
                lp.eval();
                let lp_value = lp.item_at_float32(0)?;
                generated_logprobs.push(lp_value);
            }
            if let Some(eos) = eos_token_id
                && token_value == eos as u32
            {
                finish_reason = "stop";
                break;
            }
            if let Some(reason) = check_repetition_cutoff(
                &generated_tokens,
                max_consecutive_tokens,
                max_ngram_repeats,
                ngram_size,
            ) {
                finish_reason = reason;
                break;
            }
            let next_ids = MxArray::from_uint32(&[token_value], &[1, 1])?;
            let next_logits = forward_inner(
                &next_ids,
                &embedding_weight,
                &mut self.layers,
                &mut training_caches,
                &self.final_norm,
                &self.lm_head,
                Some(&embedding_weight_t),
            )?;
            let next_last_logits = next_logits.slice_axis(1, 0, 1)?.squeeze(Some(&[0, 1]))?;
            last_logits = next_last_logits;
            if repetition_penalty != 1.0 || presence_penalty != 0.0 || frequency_penalty != 0.0 {
                let context_tokens: Vec<u32> = input_tokens
                    .iter()
                    .copied()
                    .chain(generated_tokens.iter().copied())
                    .collect();
                if repetition_penalty != 1.0 {
                    last_logits = apply_repetition_penalty(
                        &last_logits,
                        &context_tokens,
                        repetition_penalty,
                        Some(repetition_context_size),
                    )?;
                }
                if presence_penalty != 0.0 {
                    last_logits = apply_presence_penalty(
                        &last_logits,
                        &context_tokens,
                        presence_penalty,
                        Some(presence_context_size),
                    )?;
                }
                if frequency_penalty != 0.0 {
                    last_logits = apply_frequency_penalty(
                        &last_logits,
                        &context_tokens,
                        frequency_penalty,
                        Some(frequency_context_size),
                    )?;
                }
            }
            let (next_tok, next_lp) = if return_logprobs {
                let (tok, lp) = sample_and_logprobs(&last_logits, Some(sampling_config))?;
                (tok, Some(lp))
            } else {
                (sample(&last_logits, Some(sampling_config))?, None)
            };
            token = next_tok;
            logprobs = next_lp;
        }

        let tokens_array =
            MxArray::from_uint32(&generated_tokens, &[generated_tokens.len() as i64])?;
        let logprobs_array = if return_logprobs {
            MxArray::from_float32(&generated_logprobs, &[generated_logprobs.len() as i64])?
        } else {
            MxArray::from_float32(&[], &[0])?
        };

        Ok(crate::models::qwen3::GenerationResult {
            text: String::new(), // Text decoding done by caller
            tokens: tokens_array,
            logprobs: logprobs_array,
            finish_reason: finish_reason.to_string(),
            num_tokens: generated_tokens.len(),
        })
    }

    /// GRPO training step: compute loss, gradients, and apply optimizer.
    ///
    /// Consumes cached MxArrays from the generation phase, computes loss and
    /// gradients via autograd, validates and clips gradients, accumulates them,
    /// and applies the optimizer step when accumulation is complete.
    fn train_step_grpo_sync(
        &mut self,
        rewards: Vec<f64>,
        group_size: i32,
        loss_config: crate::grpo::loss::GRPOLossConfig,
        valid_indices: Option<Vec<usize>>,
    ) -> Result<crate::training_model::TrainStepPlainMetrics> {
        use crate::array::memory::{get_active_memory, get_peak_memory, reset_peak_memory};
        use crate::array::{heavy_cleanup, synchronize_and_clear_cache};
        use crate::grpo::advantages::compute_advantages;
        use crate::grpo::autograd::compute_loss_and_gradients_autograd;
        use crate::optimizers::GradientUtils;
        use crate::training_model::ModelType;

        reset_peak_memory();

        // Get cached generation results from training_state
        let ts = self.training_state.as_ref().ok_or_else(|| {
            napi::Error::from_reason("Training state not initialized. Call InitTraining first.")
        })?;

        let prompt_tokens = ts.cached_prompt_tokens.as_ref().ok_or_else(|| {
            napi::Error::from_reason("No cached prompt tokens. Call GenerateForTraining first.")
        })?;
        let completion_tokens = ts.cached_completion_tokens.as_ref().ok_or_else(|| {
            napi::Error::from_reason("No cached completion tokens. Call GenerateForTraining first.")
        })?;
        let completion_logprobs = ts.cached_completion_logprobs.as_ref().ok_or_else(|| {
            napi::Error::from_reason(
                "No cached completion logprobs. Call GenerateForTraining first.",
            )
        })?;

        let use_checkpointing = ts.gradient_checkpointing;
        let gradient_clip_value = ts.gradient_clip_value;
        let gradient_clip_norm = ts.gradient_clip_norm;
        let verbose_nan = ts.verbose_nan_detection;
        let learning_rate = ts.learning_rate;
        let max_nan_gradients = ts.max_nan_gradients;
        let emergency_save_threshold = ts.emergency_save_threshold;

        // Get model parameters
        let params = self.get_parameters_sync()?;
        let model_type = ModelType::Qwen35Dense(self.config.clone());

        // Build completion/logprob refs, optionally filtering by valid_indices from
        // the engine's degenerate-completion filter. prompt_refs always has one
        // entry per prompt — the autograd function expands them to one per
        // completion via repeat_n(group_size), so the `group_size` passed here
        // must be the effective group size after filtering (the engine computes
        // effective_group_size = valid_indices.len() / num_prompts).
        let prompt_refs: Vec<&MxArray> = prompt_tokens.iter().collect();
        let (completion_refs, logprob_refs): (Vec<&MxArray>, Vec<&MxArray>) =
            if let Some(ref indices) = valid_indices {
                let n = completion_tokens.len();
                for &i in indices {
                    if i >= n {
                        return Err(napi::Error::from_reason(format!(
                            "valid_indices contains out-of-range index {} (completion count = {})",
                            i, n
                        )));
                    }
                }
                let c: Vec<&MxArray> = indices.iter().map(|&i| &completion_tokens[i]).collect();
                let l: Vec<&MxArray> = indices.iter().map(|&i| &completion_logprobs[i]).collect();
                (c, l)
            } else {
                (
                    completion_tokens.iter().collect(),
                    completion_logprobs.iter().collect(),
                )
            };

        let (loss_value, gradients) = compute_loss_and_gradients_autograd(
            &model_type,
            &params,
            &prompt_refs,
            &completion_refs,
            &logprob_refs,
            &rewards,
            group_size,
            loss_config,
            use_checkpointing,
        )?;

        // Check for NaN/Inf loss
        if loss_value.is_nan() || loss_value.is_infinite() {
            warn!("Skipping step due to invalid loss: {}", loss_value);
            synchronize_and_clear_cache();
            // Skipped steps must still advance the authoritative step counter
            // (H1) and drop the cached generation so the next cycle starts
            // clean.
            let ts = self.training_state.as_mut().unwrap();
            ts.clear_generation_cache();
            ts.step += 1;
            let new_step = ts.step;
            let nan_count = ts.nan_gradient_count;
            return Ok(crate::training_model::TrainStepPlainMetrics {
                loss: loss_value,
                gradients_applied: false,
                mean_advantage: 0.0,
                std_advantage: 0.0,
                nan_gradient_count: nan_count,
                peak_memory_mb: get_peak_memory() / 1e6,
                active_memory_mb: get_active_memory() / 1e6,
                total_tokens: 0,
                step: new_step,
            });
        }

        // Validate ALL gradients — skip entire step if ANY has NaN/Inf
        for (name, grad) in gradients.iter() {
            grad.eval();
            let has_invalid = grad.has_nan_or_inf()?;
            if has_invalid {
                if verbose_nan {
                    let data = grad.to_float32()?;
                    let invalid_count = data
                        .iter()
                        .filter(|v| v.is_nan() || v.is_infinite())
                        .count();
                    warn!(
                        "Gradient '{}' contains {} invalid values - SKIPPING STEP",
                        name, invalid_count
                    );
                } else {
                    warn!("Gradient '{}' contains NaN/Inf - SKIPPING STEP", name);
                }

                let ts = self.training_state.as_mut().unwrap();
                ts.nan_gradient_count += 1;
                ts.consecutive_nan_count += 1;

                if ts.nan_gradient_count >= max_nan_gradients as u64 {
                    return Err(napi::Error::from_reason(format!(
                        "Training stopped: exceeded max NaN gradient count ({}/{})",
                        ts.nan_gradient_count, max_nan_gradients
                    )));
                }

                if ts.consecutive_nan_count >= emergency_save_threshold as u32 {
                    warn!(
                        "Emergency save triggered: {} consecutive NaN gradients",
                        ts.consecutive_nan_count
                    );
                }

                // Advance the authoritative step counter (H1) and clear the
                // cached generation data so the next cycle starts clean.
                ts.clear_generation_cache();
                ts.step += 1;
                let new_step = ts.step;
                let nan_count = ts.nan_gradient_count;
                synchronize_and_clear_cache();
                return Ok(crate::training_model::TrainStepPlainMetrics {
                    loss: loss_value,
                    gradients_applied: false,
                    mean_advantage: 0.0,
                    std_advantage: 0.0,
                    nan_gradient_count: nan_count,
                    peak_memory_mb: get_peak_memory() / 1e6,
                    active_memory_mb: get_active_memory() / 1e6,
                    total_tokens: 0,
                    step: new_step,
                });
            }
        }

        // Element-wise gradient clipping
        let grad_clip_val = gradient_clip_value.unwrap_or(1.0);
        let mut clamped_gradients: HashMap<String, MxArray> = HashMap::new();
        for (name, grad) in gradients.iter() {
            let clamped = grad.clip(Some(-grad_clip_val), Some(grad_clip_val))?;
            clamped.eval();
            clamped_gradients.insert(name.clone(), clamped);
        }

        // Gradient norm clipping
        let clipped_gradients = if let Some(max_norm) = gradient_clip_norm {
            let grad_refs: HashMap<String, &MxArray> = clamped_gradients
                .iter()
                .map(|(k, v)| (k.clone(), v))
                .collect();
            GradientUtils::clip_grad_norm(grad_refs, max_norm)?
        } else {
            clamped_gradients
        };

        // Accumulate gradients
        let ts = self.training_state.as_mut().unwrap();
        // Reset consecutive NaN count on successful gradient computation
        ts.consecutive_nan_count = 0;

        Self::accumulate_gradients_inner(ts, clipped_gradients)?;
        ts.micro_step += 1;

        let grad_acc_steps = ts.grad_accumulation_steps;
        let gradients_applied = if ts.micro_step >= grad_acc_steps {
            let grads = ts
                .accumulated_gradients
                .take()
                .ok_or_else(|| napi::Error::from_reason("No accumulated gradients"))?;

            // Apply optimizer step
            if let Some(ref mut optimizer) = ts.optimizer {
                // AdamW path
                let mut param_names_vec: Vec<String> = Vec::new();
                let mut param_refs: Vec<&MxArray> = Vec::new();
                let mut grad_refs: Vec<&MxArray> = Vec::new();

                // Scale gradients if using accumulation
                let scaled_grads: HashMap<String, MxArray>;
                let grads_to_use = if grad_acc_steps > 1 {
                    let scale = 1.0 / grad_acc_steps as f32;
                    let scale_arr = MxArray::from_float32(&[scale], &[])?;
                    scaled_grads = grads
                        .iter()
                        .map(|(name, grad)| (name.clone(), grad.mul(&scale_arr).unwrap()))
                        .collect();
                    &scaled_grads
                } else {
                    &grads
                };

                for (name, grad) in grads_to_use {
                    if let Some(param) = params.get(name) {
                        param_names_vec.push(name.clone());
                        param_refs.push(param);
                        grad_refs.push(grad);
                    }
                }

                let updated = optimizer.update_batch(
                    param_names_vec.clone(),
                    param_refs.clone(),
                    grad_refs,
                )?;

                // Create deltas: delta = param - updated (so param - 1.0 * delta = updated)
                let delta_map: HashMap<String, MxArray> = param_names_vec
                    .iter()
                    .enumerate()
                    .map(|(i, name)| {
                        let delta = param_refs[i].sub(&updated[i]).unwrap();
                        (name.clone(), delta)
                    })
                    .collect();

                let delta_refs: HashMap<String, &MxArray> =
                    delta_map.iter().map(|(k, v)| (k.clone(), v)).collect();
                self.apply_gradients_inner(delta_refs, 1.0, &params)?;

                tracing::debug!(
                    "Applied AdamW update (step={})",
                    self.training_state.as_ref().unwrap().step
                );
            } else {
                // SGD path
                let lr = learning_rate / grad_acc_steps as f64;
                let grads_refs: HashMap<String, &MxArray> =
                    grads.iter().map(|(k, v)| (k.clone(), v)).collect();
                self.apply_gradients_inner(grads_refs, lr, &params)?;
                tracing::debug!("Applied SGD gradients with lr: {}", lr);
            }

            let ts = self.training_state.as_mut().unwrap();
            ts.accumulated_gradients = None;
            ts.micro_step = 0;
            ts.step += 1;
            true
        } else {
            ts.step += 1;
            false
        };

        // Compute advantage statistics
        let rewards_f32: Vec<f32> = rewards.iter().map(|&r| r as f32).collect();
        let rewards_array = MxArray::from_float32(&rewards_f32, &[rewards.len() as i64])?;
        let advantages = compute_advantages(&rewards_array, group_size, "group".to_string())?;
        let adv_data = advantages.to_float32()?;
        let mean_advantage =
            adv_data.iter().map(|&a| a as f64).sum::<f64>() / adv_data.len().max(1) as f64;
        let std_advantage = {
            let variance = adv_data
                .iter()
                .map(|&a| {
                    let diff = a as f64 - mean_advantage;
                    diff * diff
                })
                .sum::<f64>()
                / adv_data.len().max(1) as f64;
            variance.sqrt()
        };

        // Count tokens BEFORE clearing the cache — otherwise total_tokens is
        // always zero on the success path.
        let ts = self.training_state.as_ref().unwrap();
        let total_tokens: i32 = if let Some(ref ct) = ts.cached_completion_tokens {
            ct.iter()
                .filter_map(|t| t.shape_at(0).ok())
                .map(|n| n as i32)
                .sum()
        } else {
            0
        };

        // Clear cached generation data
        if let Some(ref mut ts) = self.training_state {
            ts.clear_generation_cache();
        }

        // CRITICAL: heavy_cleanup after autograd to clear compiled graph cache
        heavy_cleanup();

        let ts = self.training_state.as_ref().unwrap();
        Ok(crate::training_model::TrainStepPlainMetrics {
            loss: loss_value,
            gradients_applied,
            mean_advantage,
            std_advantage,
            nan_gradient_count: ts.nan_gradient_count,
            peak_memory_mb: get_peak_memory() / 1e6,
            active_memory_mb: get_active_memory() / 1e6,
            total_tokens,
            step: ts.step,
        })
    }

    /// SFT training step: compute loss, gradients, and apply optimizer.
    ///
    /// Receives plain data (Vec<i32> + shape) from the SFT engine, reconstructs
    /// MxArrays on the model thread, computes SFT loss + gradients, validates,
    /// clips, accumulates, and applies optimizer step when accumulation is complete.
    fn train_step_sft_sync(
        &mut self,
        input_ids: Vec<i32>,
        input_shape: Vec<i64>,
        labels: Vec<i32>,
        labels_shape: Vec<i64>,
        config: crate::sft::engine::SftEngineConfig,
    ) -> Result<crate::training_model::TrainStepPlainMetrics> {
        use crate::array::memory::{get_active_memory, get_peak_memory, reset_peak_memory};
        use crate::array::{heavy_cleanup, synchronize_and_clear_cache};
        use crate::optimizers::GradientUtils;

        reset_peak_memory();

        // Ensure training state is initialized
        let ts = self.training_state.as_ref().ok_or_else(|| {
            napi::Error::from_reason("Training state not initialized. Call InitTraining first.")
        })?;
        let _ = ts;

        // Reconstruct MxArrays from plain data
        let input_ids_arr = MxArray::from_int32(&input_ids, &input_shape)?;
        let labels_arr = MxArray::from_int32(&labels, &labels_shape)?;

        // Get model parameters
        let params = self.get_parameters_sync()?;
        let model_type = crate::training_model::ModelType::Qwen35Dense(self.config.clone());

        // Build loss config from SftEngineConfig
        let loss_config = crate::sft::SftLossConfig {
            ignore_index: Some(-100),
            label_smoothing: config.label_smoothing,
        };

        let use_checkpointing = config.gradient_checkpointing.unwrap_or(true);
        let verbose_nan = config.verbose_nan_detection.unwrap_or(false);
        let max_nan_gradients = config.max_nan_gradients.unwrap_or(100);
        let emergency_save_threshold = config.emergency_save_threshold.unwrap_or(5);

        // Compute loss and gradients
        let (loss_value, gradients) = crate::sft::autograd::compute_sft_loss_and_gradients(
            &model_type,
            &params,
            &input_ids_arr,
            &labels_arr,
            loss_config,
            use_checkpointing,
        )?;

        // Check for NaN/Inf loss
        if loss_value.is_nan() || loss_value.is_infinite() {
            warn!("SFT: Skipping step due to invalid loss: {}", loss_value);
            synchronize_and_clear_cache();
            let ts = self.training_state.as_mut().unwrap();
            ts.nan_gradient_count += 1;
            ts.consecutive_nan_count += 1;

            if ts.nan_gradient_count >= max_nan_gradients as u64 {
                return Err(napi::Error::from_reason(format!(
                    "Training stopped: exceeded max NaN gradient count ({}/{})",
                    ts.nan_gradient_count, max_nan_gradients
                )));
            }

            if ts.consecutive_nan_count >= emergency_save_threshold as u32 {
                warn!(
                    "Emergency save triggered: {} consecutive NaN losses",
                    ts.consecutive_nan_count
                );
            }

            return Ok(crate::training_model::TrainStepPlainMetrics {
                loss: 0.0,
                gradients_applied: false,
                mean_advantage: 0.0,
                std_advantage: 0.0,
                nan_gradient_count: ts.nan_gradient_count,
                peak_memory_mb: get_peak_memory() / 1e6,
                active_memory_mb: get_active_memory() / 1e6,
                total_tokens: 0,
                step: ts.step,
            });
        }

        // Validate ALL gradients — skip entire step if ANY has NaN/Inf
        for (name, grad) in gradients.iter() {
            grad.eval();
            let has_invalid = grad.has_nan_or_inf()?;
            if has_invalid {
                if verbose_nan {
                    let data = grad.to_float32()?;
                    let invalid_count = data
                        .iter()
                        .filter(|v| v.is_nan() || v.is_infinite())
                        .count();
                    warn!(
                        "SFT: Gradient '{}' contains {} invalid values - SKIPPING STEP",
                        name, invalid_count
                    );
                } else {
                    warn!("SFT: Gradient '{}' contains NaN/Inf - SKIPPING STEP", name);
                }

                let ts = self.training_state.as_mut().unwrap();
                ts.nan_gradient_count += 1;
                ts.consecutive_nan_count += 1;

                if ts.nan_gradient_count >= max_nan_gradients as u64 {
                    return Err(napi::Error::from_reason(format!(
                        "Training stopped: exceeded max NaN gradient count ({}/{})",
                        ts.nan_gradient_count, max_nan_gradients
                    )));
                }

                if ts.consecutive_nan_count >= emergency_save_threshold as u32 {
                    warn!(
                        "Emergency save triggered: {} consecutive NaN gradients",
                        ts.consecutive_nan_count
                    );
                }

                synchronize_and_clear_cache();
                return Ok(crate::training_model::TrainStepPlainMetrics {
                    loss: loss_value,
                    gradients_applied: false,
                    mean_advantage: 0.0,
                    std_advantage: 0.0,
                    nan_gradient_count: ts.nan_gradient_count,
                    peak_memory_mb: get_peak_memory() / 1e6,
                    active_memory_mb: get_active_memory() / 1e6,
                    total_tokens: 0,
                    step: ts.step,
                });
            }
        }

        // Element-wise gradient clipping (if configured)
        let clipped_gradients = if let Some(clip_val) = config.gradient_clip_value {
            let mut clamped: HashMap<String, MxArray> = HashMap::new();
            for (name, grad) in gradients.iter() {
                let c = grad.clip(Some(-clip_val), Some(clip_val))?;
                c.eval();
                clamped.insert(name.clone(), c);
            }
            clamped
        } else {
            gradients.clone()
        };

        // Gradient norm clipping (if configured)
        let final_gradients = if let Some(clip_norm) = config.gradient_clip_norm {
            let grad_refs: HashMap<String, &MxArray> = clipped_gradients
                .iter()
                .map(|(k, v)| (k.clone(), v))
                .collect();
            GradientUtils::clip_grad_norm(grad_refs, clip_norm)?
        } else {
            clipped_gradients
        };

        // Accumulate gradients
        let ts = self.training_state.as_mut().unwrap();
        ts.consecutive_nan_count = 0;

        Self::accumulate_gradients_inner(ts, final_gradients)?;
        ts.micro_step += 1;

        let grad_acc_steps = config.gradient_accumulation_steps.unwrap_or(1);
        let learning_rate = config.learning_rate.unwrap_or(2e-5);
        let weight_decay = config.weight_decay.unwrap_or(0.01);

        let gradients_applied = if ts.micro_step >= grad_acc_steps {
            let grads = ts
                .accumulated_gradients
                .take()
                .ok_or_else(|| napi::Error::from_reason("No accumulated gradients"))?;

            if let Some(ref mut optimizer) = ts.optimizer {
                let mut param_names_vec: Vec<String> = Vec::new();
                let mut param_refs: Vec<&MxArray> = Vec::new();
                let mut grad_refs: Vec<&MxArray> = Vec::new();

                let scaled_grads: HashMap<String, MxArray>;
                let grads_to_use = if grad_acc_steps > 1 {
                    let scale = 1.0 / grad_acc_steps as f32;
                    let scale_arr = MxArray::from_float32(&[scale], &[])?;
                    scaled_grads = grads
                        .iter()
                        .map(|(name, grad)| (name.clone(), grad.mul(&scale_arr).unwrap()))
                        .collect();
                    &scaled_grads
                } else {
                    &grads
                };

                for (name, grad) in grads_to_use {
                    if let Some(param) = params.get(name) {
                        param_names_vec.push(name.clone());
                        param_refs.push(param);
                        grad_refs.push(grad);
                    }
                }

                let updated = optimizer.update_batch(
                    param_names_vec.clone(),
                    param_refs.clone(),
                    grad_refs,
                )?;

                let delta_map: HashMap<String, MxArray> = param_names_vec
                    .iter()
                    .enumerate()
                    .map(|(i, name)| {
                        let delta = param_refs[i].sub(&updated[i]).unwrap();
                        (name.clone(), delta)
                    })
                    .collect();

                let delta_refs: HashMap<String, &MxArray> =
                    delta_map.iter().map(|(k, v)| (k.clone(), v)).collect();
                self.apply_gradients_inner(delta_refs, 1.0, &params)?;

                tracing::debug!(
                    "SFT: Applied AdamW update (step={})",
                    self.training_state.as_ref().unwrap().step
                );
            } else {
                let lr = learning_rate / grad_acc_steps as f64;

                let grads_with_decay = if weight_decay > 0.0 {
                    grads
                        .into_iter()
                        .map(|(name, grad)| {
                            if let Some(param) = params.get(&name) {
                                if let Ok(decay_term) = param.mul_scalar(weight_decay)
                                    && let Ok(new_grad) = grad.add(&decay_term)
                                {
                                    return (name, new_grad);
                                }
                                (name, grad)
                            } else {
                                (name, grad)
                            }
                        })
                        .collect::<HashMap<_, _>>()
                } else {
                    grads
                };

                let grads_refs: HashMap<String, &MxArray> = grads_with_decay
                    .iter()
                    .map(|(k, v)| (k.clone(), v))
                    .collect();
                self.apply_gradients_inner(grads_refs, lr, &params)?;
                tracing::debug!("SFT: Applied SGD gradients with lr: {}", lr);
            }

            let ts = self.training_state.as_mut().unwrap();
            ts.accumulated_gradients = None;
            ts.micro_step = 0;
            ts.step += 1;
            true
        } else {
            ts.step += 1;
            false
        };

        // Count valid tokens from the labels
        let total_tokens = {
            let ignore_val = MxArray::scalar_int(-100)?;
            let valid_mask = labels_arr.not_equal(&ignore_val)?;
            let count = valid_mask.sum(None, Some(false))?;
            count.eval();
            count.item_at_int32(0).unwrap_or(0)
        };

        // CRITICAL: heavy_cleanup after autograd to clear compiled graph cache
        heavy_cleanup();

        let ts = self.training_state.as_ref().unwrap();
        Ok(crate::training_model::TrainStepPlainMetrics {
            loss: loss_value,
            gradients_applied,
            mean_advantage: 0.0,
            std_advantage: 0.0,
            nan_gradient_count: ts.nan_gradient_count,
            peak_memory_mb: get_peak_memory() / 1e6,
            active_memory_mb: get_active_memory() / 1e6,
            total_tokens,
            step: ts.step,
        })
    }

    /// Accumulate gradients into training state.
    fn accumulate_gradients_inner(
        ts: &mut crate::training_state::ModelThreadTrainingState,
        new_grads: HashMap<String, MxArray>,
    ) -> Result<()> {
        match &mut ts.accumulated_gradients {
            Some(acc) => {
                for (name, grad) in new_grads {
                    grad.eval();
                    if grad.has_nan_or_inf()? {
                        warn!(
                            "Skipping gradient accumulation for '{}' due to NaN/Inf",
                            name
                        );
                        continue;
                    }
                    if let Some(existing) = acc.get_mut(&name) {
                        let summed = existing.add(&grad)?;
                        summed.eval();
                        *existing = summed;
                    } else {
                        acc.insert(name, grad);
                    }
                }
            }
            None => {
                let mut evaluated_grads = HashMap::with_capacity(new_grads.len());
                for (name, grad) in new_grads {
                    grad.eval();
                    if grad.has_nan_or_inf()? {
                        warn!("Skipping initial gradient for '{}' due to NaN/Inf", name);
                        continue;
                    }
                    evaluated_grads.insert(name, grad);
                }
                ts.accumulated_gradients = Some(evaluated_grads);
            }
        }
        Ok(())
    }

    /// Apply gradients to model weights (SGD or AdamW delta application).
    ///
    /// Direct field access on Qwen35Inner — no locks needed.
    fn apply_gradients_inner(
        &mut self,
        gradients: HashMap<String, &MxArray>,
        learning_rate: f64,
        current_params: &HashMap<String, MxArray>,
    ) -> Result<()> {
        use super::decoder_layer::AttentionType;

        let updated_params =
            crate::training_model::compute_sgd_updates(&gradients, learning_rate, current_params)?;

        // Apply updated parameters directly to model fields
        for (name, updated_param) in updated_params.iter() {
            if name == "lm_head.weight" {
                if let Some(ref mut lm) = self.lm_head {
                    lm.set_weight(updated_param)?;
                }
            } else if name == "final_norm.weight" {
                self.final_norm.set_weight(updated_param)?;
            } else if name == "embedding.weight" {
                self.embedding.set_weight(updated_param)?;
            } else if name.starts_with("layers.") {
                let parts: Vec<&str> = name.split('.').collect();
                if parts.len() >= 3
                    && let Ok(layer_idx) = parts[1].parse::<usize>()
                    && layer_idx < self.layers.len()
                {
                    let layer = &mut self.layers[layer_idx];
                    if name.contains(".linear_attn.") {
                        if let AttentionType::Linear(ref mut gdn) = layer.attn {
                            if name.ends_with(".in_proj_qkvz.weight") {
                                gdn.set_in_proj_qkvz_weight(updated_param)?;
                            } else if name.ends_with(".in_proj_ba.weight") {
                                gdn.set_in_proj_ba_weight(updated_param)?;
                            } else if name.ends_with(".conv1d.weight") {
                                gdn.set_conv1d_weight(updated_param)?;
                            } else if name.ends_with(".norm.weight") {
                                gdn.set_norm_weight(updated_param)?;
                            } else if name.ends_with(".out_proj.weight") {
                                gdn.set_out_proj_weight(updated_param)?;
                            } else if name.ends_with(".dt_bias") {
                                gdn.set_dt_bias(updated_param);
                            } else if name.ends_with(".a_log") {
                                gdn.set_a_log(updated_param)?;
                            }
                        }
                    } else if name.contains(".self_attn.") {
                        if let AttentionType::Full(ref mut attn) = layer.attn {
                            if name.ends_with(".q_proj.weight") {
                                attn.set_q_proj_weight(updated_param)?;
                            } else if name.ends_with(".k_proj.weight") {
                                attn.set_k_proj_weight(updated_param)?;
                            } else if name.ends_with(".v_proj.weight") {
                                attn.set_v_proj_weight(updated_param)?;
                            } else if name.ends_with(".o_proj.weight") {
                                attn.set_o_proj_weight(updated_param)?;
                            } else if name.ends_with(".q_norm.weight") {
                                attn.set_q_norm_weight(updated_param)?;
                            } else if name.ends_with(".k_norm.weight") {
                                attn.set_k_norm_weight(updated_param)?;
                            }
                        }
                    } else if name.contains(".mlp.") {
                        if name.ends_with(".gate_proj.weight") {
                            layer.mlp.set_gate_proj_weight(updated_param)?;
                        } else if name.ends_with(".up_proj.weight") {
                            layer.mlp.set_up_proj_weight(updated_param)?;
                        } else if name.ends_with(".down_proj.weight") {
                            layer.mlp.set_down_proj_weight(updated_param)?;
                        }
                    } else if name.ends_with(".input_layernorm.weight") {
                        layer.set_input_layernorm_weight(updated_param)?;
                    } else if name.ends_with(".post_attention_layernorm.weight") {
                        layer.set_post_attention_layernorm_weight(updated_param)?;
                    }
                }
            }
        }

        Ok(())
    }

    /// Extract all trainable parameters from the model.
    /// Direct field access — no locks needed on model thread.
    fn get_parameters_sync(&self) -> Result<HashMap<String, MxArray>> {
        use super::decoder_layer::AttentionType;

        let mut params = HashMap::new();

        // Embedding
        params.insert("embedding.weight".to_string(), self.embedding.get_weight());

        // Transformer layers
        for (i, layer) in self.layers.iter().enumerate() {
            let prefix = format!("layers.{}", i);

            match &layer.attn {
                AttentionType::Linear(gdn) => {
                    params.insert(
                        format!("{}.linear_attn.in_proj_qkvz.weight", prefix),
                        gdn.get_in_proj_qkvz_weight(),
                    );
                    params.insert(
                        format!("{}.linear_attn.in_proj_ba.weight", prefix),
                        gdn.get_in_proj_ba_weight(),
                    );
                    params.insert(
                        format!("{}.linear_attn.conv1d.weight", prefix),
                        gdn.get_conv1d_weight(),
                    );
                    params.insert(
                        format!("{}.linear_attn.norm.weight", prefix),
                        gdn.get_norm_weight(),
                    );
                    params.insert(
                        format!("{}.linear_attn.out_proj.weight", prefix),
                        gdn.get_out_proj_weight(),
                    );
                    params.insert(format!("{}.linear_attn.dt_bias", prefix), gdn.get_dt_bias());
                    params.insert(format!("{}.linear_attn.a_log", prefix), gdn.get_a_log());
                }
                AttentionType::Full(attn) => {
                    params.insert(
                        format!("{}.self_attn.q_proj.weight", prefix),
                        attn.get_q_proj_weight(),
                    );
                    params.insert(
                        format!("{}.self_attn.k_proj.weight", prefix),
                        attn.get_k_proj_weight(),
                    );
                    params.insert(
                        format!("{}.self_attn.v_proj.weight", prefix),
                        attn.get_v_proj_weight(),
                    );
                    params.insert(
                        format!("{}.self_attn.o_proj.weight", prefix),
                        attn.get_o_proj_weight(),
                    );
                    params.insert(
                        format!("{}.self_attn.q_norm.weight", prefix),
                        attn.get_q_norm_weight(),
                    );
                    params.insert(
                        format!("{}.self_attn.k_norm.weight", prefix),
                        attn.get_k_norm_weight(),
                    );
                }
            }

            // MLP (all layers have dense MLP)
            params.insert(
                format!("{}.mlp.gate_proj.weight", prefix),
                layer.mlp.get_gate_proj_weight(),
            );
            params.insert(
                format!("{}.mlp.up_proj.weight", prefix),
                layer.mlp.get_up_proj_weight(),
            );
            params.insert(
                format!("{}.mlp.down_proj.weight", prefix),
                layer.mlp.get_down_proj_weight(),
            );

            // Layer norms
            params.insert(
                format!("{}.input_layernorm.weight", prefix),
                layer.get_input_layernorm_weight(),
            );
            params.insert(
                format!("{}.post_attention_layernorm.weight", prefix),
                layer.get_post_attention_layernorm_weight(),
            );
        }

        // Final norm
        params.insert(
            "final_norm.weight".to_string(),
            self.final_norm.get_weight(),
        );

        // LM head (only if not tied to embeddings)
        if !self.config.tie_word_embeddings
            && let Some(ref lm_head) = self.lm_head
        {
            params.insert("lm_head.weight".to_string(), lm_head.get_weight());
        }

        Ok(params)
    }
}

/// Wrapper around `StreamTx` that provides a `.call()` method matching the
/// `ThreadsafeFunction` interface expected by the `decode_loop!` macro.
///
/// This allows the macro to work unchanged for both:
/// - MoE model: passes a real `ThreadsafeFunction` (old path, until Phase 4)
/// - Dense model: passes this `StreamSender` (new dedicated-thread path)
struct StreamSender(StreamTx<ChatStreamChunk>);

impl StreamSender {
    fn call(&self, result: napi::Result<ChatStreamChunk>, _mode: ThreadsafeFunctionCallMode) {
        let _ = self.0.send(result);
    }
}

/// RAII guard that calls `mlx_qwen35_compiled_reset()` on drop.
///
/// Ensures C++ compiled state is always cleaned up, even if the decode
/// loop returns early via `?` operator. Without this, an error during decode
/// would leave stale compiled state that corrupts the next generation call.
struct CompiledResetGuard;

impl Drop for CompiledResetGuard {
    fn drop(&mut self) {
        unsafe {
            mlx_sys::mlx_qwen35_compiled_reset();
        }
    }
}

/// Generation configuration for Qwen3.5
#[napi(object)]
#[derive(Debug, Clone)]
pub struct Qwen3_5GenerationConfig {
    pub max_new_tokens: i32,
    #[napi(ts_type = "number | undefined")]
    pub temperature: Option<f64>,
    #[napi(ts_type = "number | undefined")]
    pub top_k: Option<i32>,
    #[napi(ts_type = "number | undefined")]
    pub top_p: Option<f64>,
    #[napi(ts_type = "number | undefined")]
    pub min_p: Option<f64>,
}

/// Generation result
#[napi(object)]
#[derive(Debug, Clone)]
pub struct Qwen3_5GenerationResult {
    pub tokens: Vec<u32>,
    pub text: String,
    pub num_tokens: u32,
    pub finish_reason: String,
}

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
    /// Max consecutive identical tokens before stopping (default: 16, 0 = disabled)
    #[napi(ts_type = "number | undefined")]
    pub max_consecutive_tokens: Option<i32>,
    /// Max n-gram repetitions before stopping (default: 3, 0 = disabled)
    #[napi(ts_type = "number | undefined")]
    pub max_ngram_repeats: Option<i32>,
    /// Max pattern size for n-gram repetition detection (default: 64)
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

/// Qwen3.5 Model -- hybrid linear/full attention with optional MoE.
///
/// All inference and training state lives on a dedicated OS thread. NAPI methods
/// dispatch commands via channels and await responses. Training commands are
/// routed through `TrainingDispatch` to the model thread.
#[napi]
pub struct Qwen3_5Model {
    /// Dedicated model thread for inference and training.
    pub(crate) thread: crate::model_thread::ModelThread<Qwen35Cmd>,
    /// Cloned from inner for pure-getter NAPI methods (no command dispatch needed).
    pub(crate) config: Qwen3_5Config,
    pub(crate) model_id: u64,
}

#[napi]
impl Qwen3_5Model {
    /// Initialize caches for incremental generation.
    #[napi]
    pub fn init_caches(&self) -> Result<()> {
        crate::model_thread::send_and_block(&self.thread, |reply| Qwen35Cmd::InitCaches { reply })
    }

    /// Reset all caches.
    #[napi]
    pub fn reset_caches(&self) -> Result<()> {
        crate::model_thread::send_and_block(&self.thread, |reply| Qwen35Cmd::ResetCaches { reply })
    }

    /// Take the KV cache from the model, returning a `PromptCache` handle.
    ///
    /// The cache is moved out of the model — calling `takeCache()` twice
    /// returns `null` the second time. Pass the cache back via `setCache()`
    /// before the next `chatSessionStart` / `chatSessionContinue` call for
    /// incremental prefill.
    #[napi]
    pub fn take_cache(&self) -> Option<crate::models::qwen3_5::prompt_cache::PromptCache> {
        crate::model_thread::send_and_block(&self.thread, |reply| Qwen35Cmd::TakeCache { reply })
            .ok()?
    }

    /// Restore a previously taken `PromptCache` into the model.
    ///
    /// On the next `chatSessionStart` / `chatSessionContinue` call with
    /// `reuseCache: true`, the model will prefix-match the new tokens against
    /// the cache and only prefill the delta.
    #[napi]
    pub fn set_cache(
        &self,
        cache: &mut crate::models::qwen3_5::prompt_cache::PromptCache,
    ) -> Result<()> {
        // Validate before sending (these checks don't need model-thread state)
        if cache.model_type() != "qwen3_5" {
            return Err(Error::from_reason(format!(
                "Cache type '{}' doesn't match model type 'qwen3_5'",
                cache.model_type()
            )));
        }
        if cache.num_layers() != self.config.num_layers as usize {
            return Err(Error::from_reason(format!(
                "Cache has {} layers but model has {} layers",
                cache.num_layers(),
                self.config.num_layers
            )));
        }
        if cache.model_id() != self.model_id {
            return Err(Error::from_reason(
                "Cache was created by a different model instance (different checkpoint or config)",
            ));
        }
        // Extract the cache data to send to model thread
        let owned_cache = crate::models::qwen3_5::prompt_cache::PromptCache::new(
            cache.take_caches().ok_or_else(|| {
                Error::from_reason("PromptCache is empty (already consumed or disposed)")
            })?,
            cache.token_history().to_vec(),
            "qwen3_5",
            cache.num_layers(),
            cache.image_cache_key(),
            cache.rope_deltas(),
            cache.model_id(),
        );
        crate::model_thread::send_and_block(&self.thread, |reply| Qwen35Cmd::SetCache {
            cache: owned_cache,
            reply,
        })
    }

    /// Load a pretrained model from a directory.
    ///
    /// Expects the directory to contain:
    /// - config.json
    /// - model.safetensors (or model-*.safetensors)
    /// - tokenizer.json + tokenizer_config.json
    #[napi]
    pub async fn load(path: String) -> Result<Qwen3_5Model> {
        persistence::load_with_thread(&path).await
    }

    /// Generate text from a prompt token sequence.
    #[napi]
    pub async fn generate(
        &self,
        prompt_tokens: &MxArray,
        config: Qwen3_5GenerationConfig,
    ) -> Result<Qwen3_5GenerationResult> {
        if config.max_new_tokens <= 0 {
            return Err(Error::from_reason(format!(
                "max_new_tokens must be > 0, got {}",
                config.max_new_tokens
            )));
        }
        let batch_size = prompt_tokens.shape_at(0)?;
        if batch_size != 1 {
            return Err(Error::from_reason(format!(
                "generate() only supports batch_size=1, got batch_size={}",
                batch_size
            )));
        }
        crate::model_thread::send_and_await(&self.thread, |reply| Qwen35Cmd::Generate {
            prompt_tokens: prompt_tokens.clone(),
            config,
            reply,
        })
        .await
    }

    /// Start a new chat session.
    ///
    /// Runs the full jinja chat template once and uses `<|im_end|>` as
    /// its stop token so the cached KV state ends on a clean ChatML
    /// boundary. Image support is conditional on the loaded
    /// checkpoint: a Qwen3.5-VL dense model loaded with vision weights
    /// accepts images in `messages` (the vision encoder handles
    /// prefill), while a plain text Qwen3.5 checkpoint rejects them
    /// with a runtime error. Subsequent turns in the same session MUST
    /// go through `chatSessionContinue` so the caller appends raw
    /// ChatML deltas on top of the live caches without rerunning the
    /// jinja template; a mid-session image change requires a fresh
    /// `chatSessionStart` call. The session is owned end-to-end by
    /// the `chatSession*` surface.
    ///
    /// This method is the production entry point used by the TypeScript
    /// `ChatSession` wrapper for turn 1 of a multi-round conversation.
    #[napi]
    pub async fn chat_session_start(
        &self,
        messages: Vec<ChatMessage>,
        config: Option<ChatConfig>,
    ) -> Result<ChatResult> {
        let config = config.unwrap_or(ChatConfig {
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
            thinking_token_budget: None,
            include_reasoning: None,
            reasoning_effort: None,
            report_performance: None,
            reuse_cache: None,
        });

        crate::model_thread::send_and_await(&self.thread, |reply| Qwen35Cmd::ChatSessionStart {
            messages,
            config,
            reply,
        })
        .await
    }

    /// Continue an existing chat session with a new user message.
    ///
    /// Appends a raw ChatML user/assistant delta to the session's cached
    /// KV state, then decodes the assistant reply. Stops on `<|im_end|>`
    /// so the cache remains on a clean boundary for the next turn.
    ///
    /// Requires a live session started via `chatSessionStart`. Errors
    /// if the session is empty, carries image state, or if
    /// `config.reuse_cache` is explicitly set to `false`.
    ///
    /// `images` is an opt-in guard parameter: when non-empty, the native
    /// side returns an error whose message begins with
    /// `IMAGE_CHANGE_REQUIRES_SESSION_RESTART:` so the TypeScript
    /// `ChatSession` layer can catch the prefix and route image-changes
    /// back through a fresh `chatSessionStart`.
    #[napi(
        ts_args_type = "userMessage: string, images: Uint8Array[] | null | undefined, config: ChatConfig | null | undefined"
    )]
    pub async fn chat_session_continue(
        &self,
        user_message: String,
        images: Option<Vec<Uint8Array>>,
        config: Option<ChatConfig>,
    ) -> Result<ChatResult> {
        let config = config.unwrap_or(ChatConfig {
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
            thinking_token_budget: None,
            include_reasoning: None,
            reasoning_effort: None,
            report_performance: None,
            reuse_cache: None,
        });

        crate::model_thread::send_and_await(&self.thread, |reply| Qwen35Cmd::ChatSessionContinue {
            user_message,
            images,
            config,
            reply,
        })
        .await
    }

    /// Continue an existing chat session with a tool-result turn.
    ///
    /// Builds a ChatML `<tool_response>`-wrapped delta from `content` and
    /// prefills it on top of the live session caches, then decodes the
    /// assistant reply. Stops on `<|im_end|>` so the cache stays on a
    /// clean boundary for the next turn.
    ///
    /// The `tool_call_id` is currently dropped by the wire format —
    /// Qwen3.5's chat template identifies tool responses by position +
    /// wrapper tags, not an explicit id. Callers may still log it for
    /// their own bookkeeping.
    ///
    /// Requires a live session started via `chatSessionStart`.
    #[napi]
    pub async fn chat_session_continue_tool(
        &self,
        tool_call_id: String,
        content: String,
        config: Option<ChatConfig>,
    ) -> Result<ChatResult> {
        let config = config.unwrap_or(ChatConfig {
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
            thinking_token_budget: None,
            include_reasoning: None,
            reasoning_effort: None,
            report_performance: None,
            reuse_cache: None,
        });

        crate::model_thread::send_and_await(&self.thread, |reply| {
            Qwen35Cmd::ChatSessionContinueTool {
                tool_call_id,
                content,
                config,
                reply,
            }
        })
        .await
    }

    /// Streaming variant of `chatSessionStart`.
    ///
    /// Dispatches to the dedicated model thread. Behaviourally identical
    /// to `chatSessionStart` (resets caches, uses `<|im_end|>` as
    /// eos, inherits the same VLM-vs-text image-support contract) but
    /// streams token deltas through the JS callback instead of
    /// returning a `ChatResult`. Used by the TypeScript
    /// `ChatSession.sendStream()` for turn 1 of a multi-round streaming
    /// conversation.
    #[napi(
        ts_args_type = "messages: ChatMessage[], config: ChatConfig | null, callback: (err: Error | null, chunk: ChatStreamChunk) => void"
    )]
    pub async fn chat_stream_session_start(
        &self,
        messages: Vec<ChatMessage>,
        config: Option<ChatConfig>,
        callback: ThreadsafeFunction<ChatStreamChunk, ()>,
    ) -> Result<ChatStreamHandle> {
        let config = config.unwrap_or(ChatConfig {
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
            thinking_token_budget: None,
            include_reasoning: None,
            reasoning_effort: None,
            report_performance: None,
            reuse_cache: None,
        });

        let cancelled = Arc::new(AtomicBool::new(false));
        let cancelled_inner = cancelled.clone();
        let (stream_tx, mut stream_rx) =
            tokio::sync::mpsc::unbounded_channel::<napi::Result<ChatStreamChunk>>();

        self.thread.send(Qwen35Cmd::ChatStreamSessionStart {
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
    ///
    /// Appends a ChatML user/assistant delta on top of the live session
    /// caches and streams the decoded reply. Requires a live session
    /// started via `chatStreamSessionStart` (or the non-streaming
    /// `chatSessionStart`). Used by the TypeScript
    /// `ChatSession.sendStream()` for turns 2..N of a multi-round
    /// streaming conversation.
    ///
    /// `images` is an opt-in guard parameter: when non-empty, the
    /// streaming path emits an error chunk whose message begins with
    /// `IMAGE_CHANGE_REQUIRES_SESSION_RESTART:` so the TypeScript
    /// `ChatSession` layer can route image-changes through a fresh
    /// session start.
    #[napi(
        ts_args_type = "userMessage: string, images: Uint8Array[] | null | undefined, config: ChatConfig | null, callback: (err: Error | null, chunk: ChatStreamChunk) => void"
    )]
    pub async fn chat_stream_session_continue(
        &self,
        user_message: String,
        images: Option<Vec<Uint8Array>>,
        config: Option<ChatConfig>,
        callback: ThreadsafeFunction<ChatStreamChunk, ()>,
    ) -> Result<ChatStreamHandle> {
        let config = config.unwrap_or(ChatConfig {
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
            thinking_token_budget: None,
            include_reasoning: None,
            reasoning_effort: None,
            report_performance: None,
            reuse_cache: None,
        });

        let cancelled = Arc::new(AtomicBool::new(false));
        let cancelled_inner = cancelled.clone();
        let (stream_tx, mut stream_rx) =
            tokio::sync::mpsc::unbounded_channel::<napi::Result<ChatStreamChunk>>();

        self.thread.send(Qwen35Cmd::ChatStreamSessionContinue {
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
    ///
    /// Builds a ChatML tool-response delta on top of the live session
    /// caches and streams the decoded reply. Requires a live session
    /// started via `chatSessionStart` / `chatStreamSessionStart`.
    #[napi(
        ts_args_type = "toolCallId: string, content: string, config: ChatConfig | null, callback: (err: Error | null, chunk: ChatStreamChunk) => void"
    )]
    pub async fn chat_stream_session_continue_tool(
        &self,
        tool_call_id: String,
        content: String,
        config: Option<ChatConfig>,
        callback: ThreadsafeFunction<ChatStreamChunk, ()>,
    ) -> Result<ChatStreamHandle> {
        let config = config.unwrap_or(ChatConfig {
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
            thinking_token_budget: None,
            include_reasoning: None,
            reasoning_effort: None,
            report_performance: None,
            reuse_cache: None,
        });

        let cancelled = Arc::new(AtomicBool::new(false));
        let cancelled_inner = cancelled.clone();
        let (stream_tx, mut stream_rx) =
            tokio::sync::mpsc::unbounded_channel::<napi::Result<ChatStreamChunk>>();

        self.thread.send(Qwen35Cmd::ChatStreamSessionContinueTool {
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

    // ---------------------------------------------------------------
    // Test-only helpers: streaming session entry points that bypass
    // ThreadsafeFunction and expose the mpsc receiver directly. Used
    // by `crates/mlx-core/tests/qwen3_5_delta_chat.rs` to exercise the
    // streaming path from a pure-Rust integration test without a NAPI
    // host. Marked `#[doc(hidden)]` because they're not part of the
    // public API surface.
    // ---------------------------------------------------------------

    /// Test-only entry point that dispatches `ChatStreamSessionStart`
    /// and returns the raw mpsc receiver the model thread writes into.
    /// Callers can iterate the receiver directly rather than going
    /// through a NAPI callback.
    #[doc(hidden)]
    pub fn chat_stream_session_start_for_test(
        &self,
        messages: Vec<ChatMessage>,
        config: Option<ChatConfig>,
    ) -> Result<(
        ChatStreamHandle,
        tokio::sync::mpsc::UnboundedReceiver<Result<ChatStreamChunk>>,
    )> {
        let config = config.unwrap_or_default();
        let cancelled = Arc::new(AtomicBool::new(false));
        let cancelled_inner = cancelled.clone();
        let (stream_tx, stream_rx) =
            tokio::sync::mpsc::unbounded_channel::<Result<ChatStreamChunk>>();
        self.thread.send(Qwen35Cmd::ChatStreamSessionStart {
            messages,
            config,
            stream_tx,
            cancelled: cancelled_inner,
        })?;
        Ok((ChatStreamHandle { cancelled }, stream_rx))
    }

    /// Test-only entry point that dispatches `ChatStreamSessionContinue`
    /// and returns the raw mpsc receiver the model thread writes into.
    #[doc(hidden)]
    pub fn chat_stream_session_continue_for_test(
        &self,
        user_message: String,
        images: Option<Vec<Uint8Array>>,
        config: Option<ChatConfig>,
    ) -> Result<(
        ChatStreamHandle,
        tokio::sync::mpsc::UnboundedReceiver<Result<ChatStreamChunk>>,
    )> {
        let config = config.unwrap_or_default();
        let cancelled = Arc::new(AtomicBool::new(false));
        let cancelled_inner = cancelled.clone();
        let (stream_tx, stream_rx) =
            tokio::sync::mpsc::unbounded_channel::<Result<ChatStreamChunk>>();
        self.thread.send(Qwen35Cmd::ChatStreamSessionContinue {
            user_message,
            images,
            config,
            stream_tx,
            cancelled: cancelled_inner,
        })?;
        Ok((ChatStreamHandle { cancelled }, stream_rx))
    }

    /// Test-only entry point that dispatches
    /// `ChatStreamSessionContinueTool` and returns the raw mpsc
    /// receiver the model thread writes into.
    #[doc(hidden)]
    pub fn chat_stream_session_continue_tool_for_test(
        &self,
        tool_call_id: String,
        content: String,
        config: Option<ChatConfig>,
    ) -> Result<(
        ChatStreamHandle,
        tokio::sync::mpsc::UnboundedReceiver<Result<ChatStreamChunk>>,
    )> {
        let config = config.unwrap_or_default();
        let cancelled = Arc::new(AtomicBool::new(false));
        let cancelled_inner = cancelled.clone();
        let (stream_tx, stream_rx) =
            tokio::sync::mpsc::unbounded_channel::<Result<ChatStreamChunk>>();
        self.thread.send(Qwen35Cmd::ChatStreamSessionContinueTool {
            tool_call_id,
            content,
            config,
            stream_tx,
            cancelled: cancelled_inner,
        })?;
        Ok((ChatStreamHandle { cancelled }, stream_rx))
    }

    /// Get the number of parameters in the model.
    ///
    /// Pure config computation — no model-thread dispatch needed.
    #[napi]
    pub fn num_parameters(&self) -> i64 {
        let h = self.config.hidden_size as i64;
        let v = self.config.vocab_size as i64;
        let n = self.config.num_layers as usize;
        let dense_i = self.config.intermediate_size as i64;

        let mut total = v * h;
        if !self.config.tie_word_embeddings {
            total += v * h;
        }

        let kd = self.config.linear_key_dim() as i64;
        let vd = self.config.linear_value_dim() as i64;

        for layer_idx in 0..n {
            let is_linear = self.config.is_linear_layer(layer_idx);
            if is_linear {
                let num_vh = self.config.linear_num_value_heads as i64;
                let vhd = self.config.linear_value_head_dim as i64;
                total += h * (kd * 2 + vd * 2)
                    + h * (num_vh * 2)
                    + (kd * 2 + vd) * self.config.linear_conv_kernel_dim as i64
                    + vd * h
                    + num_vh
                    + num_vh
                    + vhd;
            } else {
                let d = self.config.head_dim as i64;
                total += h * h * 2 + h * (self.config.num_kv_heads as i64 * d) * 2 + h * h + d * 2;
            }
            total += 3 * h * dense_i;
            total += h * 2;
        }
        total += h;
        total
    }

    /// Save the model weights and configuration to a directory.
    ///
    /// Dispatches to model thread.
    #[napi]
    pub fn save_model<'env>(
        &self,
        env: &'env Env,
        save_path: String,
    ) -> Result<PromiseRaw<'env, ()>> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.thread.send(Qwen35Cmd::SaveModel {
            save_path,
            reply: tx,
        })?;
        let promise = env.spawn_future(async move {
            rx.await
                .map_err(|_| napi::Error::from_reason("Model thread exited unexpectedly"))?
        })?;
        Ok(promise)
    }
}

/// Default prefill chunk size (tokens per chunk).
/// Matches Python mlx-lm's `prefill_step_size` default of 2048.
const PREFILL_STEP_SIZE: i64 = 2048;

/// Evaluate all cache arrays across all layers to materialize them on GPU.
/// Must be called between prefill chunks to break lazy dependency chains.
pub(crate) fn eval_layer_caches(caches: &Option<Vec<Qwen3_5LayerCache>>) {
    if let Some(caches) = caches {
        let mut arrays: Vec<&MxArray> = Vec::new();
        for cache in caches.iter() {
            cache.collect_arrays(&mut arrays);
        }
        MxArray::eval_arrays(&arrays);
    }
}

/// Chunked prefill: process prompt in chunks of `PREFILL_STEP_SIZE`, evaluating
/// caches and clearing compute cache between chunks to bound peak memory.
///
/// Accepts `&MxArray` shaped `[1, seq_len]`. Slices on GPU — no data roundtrip.
/// For `&[u32]` inputs (from tokenizer), callers convert with `MxArray::from_uint32` first.
fn chunked_prefill(
    prompt: &MxArray,
    embedding_weight: &MxArray,
    layers: &mut [DecoderLayer],
    caches: &mut Option<Vec<Qwen3_5LayerCache>>,
    final_norm: &RMSNorm,
    lm_head: &Option<Linear>,
    embedding_weight_t: Option<&MxArray>,
    generation_stream: crate::stream::Stream,
) -> Result<MxArray> {
    let total_len = prompt.shape_at(1)?;
    let mut offset: i64 = 0;

    while total_len - offset > PREFILL_STEP_SIZE {
        let chunk = prompt.slice_axis(1, offset, offset + PREFILL_STEP_SIZE)?;
        {
            let _stream_ctx = StreamContext::new(generation_stream);
            let _logits = forward_inner(
                &chunk,
                embedding_weight,
                layers,
                caches,
                final_norm,
                lm_head,
                embedding_weight_t,
            )?;
        }
        eval_layer_caches(caches);
        crate::array::clear_cache();
        offset += PREFILL_STEP_SIZE;
    }

    let remaining = prompt.slice_axis(1, offset, total_len)?;
    let logits = {
        let _stream_ctx = StreamContext::new(generation_stream);
        forward_inner(
            &remaining,
            embedding_weight,
            layers,
            caches,
            final_norm,
            lm_head,
            embedding_weight_t,
        )?
    };
    Ok(logits)
}

/// Lock-free forward pass through all layers.
/// Attention layer handles causal masking internally via "causal" SDPA mode.
fn forward_inner(
    input_ids: &MxArray,
    embedding_weight: &MxArray,
    layers: &mut [DecoderLayer],
    caches: &mut Option<Vec<Qwen3_5LayerCache>>,
    final_norm: &RMSNorm,
    lm_head: &Option<Linear>,
    embedding_weight_t: Option<&MxArray>,
) -> Result<MxArray> {
    let embedding = Embedding::from_weight(embedding_weight)?;
    let hidden_states = embedding.forward(input_ids)?;
    let mut h = hidden_states.clone();

    let num_layers = layers.len();
    for i in 0..num_layers {
        let cache = caches.as_mut().map(|c| &mut c[i]);
        h = layers[i].forward(&h, None, cache, None, true)?;
    }

    let h = final_norm.forward(&h)?;
    match lm_head {
        Some(head) => head.forward(&h),
        None => match embedding_weight_t {
            Some(wt) => h.matmul(wt),
            None => {
                let wt = embedding_weight.transpose(Some(&[1, 0]))?;
                h.matmul(&wt)
            }
        },
    }
}

/// Compiled single-token decode step using mlx::core::compile().
///
/// On the first call, MLX traces the 64-layer forward pass and caches the graph.
/// All subsequent calls reuse the cached graph via compile_replace — no re-tracing.
/// This eliminates per-step graph reconstruction overhead (~5358 nodes).
///
/// State is held in C++ globals (g_compiled_caches, g_compiled_offset).
/// Must call `mlx_qwen35_compiled_init_from_prefill` before the decode loop.
fn forward_compiled(input_ids: &MxArray, embedding_weight: &MxArray) -> Result<MxArray> {
    use mlx_sys as sys;

    let mut output_ptr: *mut sys::mlx_array = std::ptr::null_mut();
    unsafe {
        sys::mlx_qwen35_forward_compiled(
            input_ids.as_raw_ptr(),
            embedding_weight.as_raw_ptr(),
            &mut output_ptr,
            std::ptr::null_mut(),
        );
    }

    if output_ptr.is_null() {
        return Err(Error::from_reason(
            "C++ compiled forward step returned null — check stderr for exception details",
        ));
    }

    MxArray::from_handle(output_ptr, "compiled_forward_logits")
}

/// Evaluate next_token and all compiled cache arrays to prevent graph accumulation.
///
/// Called after each compiled decode step. mlx::core::compile reuses the graph
/// structure across steps, but we still need to eval to materialize output arrays
/// and break lazy dependency chains (preventing O(N²) graph growth).
fn eval_token_and_compiled_caches(next_token: &MxArray) {
    unsafe {
        mlx_sys::mlx_qwen35_eval_token_and_compiled_caches(next_token.as_raw_ptr());
    }
}

// ============================================================================
// VLM helper functions (moved from vl_model.rs for unification)
// ============================================================================

/// Image token ID used by Qwen3.5-VL
pub(crate) const IMAGE_TOKEN_ID: i32 = 248056;

/// Extract all raw image bytes from chat messages.
pub(crate) fn extract_images_from_messages(messages: &[ChatMessage]) -> Vec<Vec<u8>> {
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

/// Compute the number of merged image tokens from a processed grid_thw array.
pub(crate) fn compute_num_image_tokens(grid: &MxArray, spatial_merge_size: i32) -> Result<usize> {
    grid.eval();
    let grid_data = grid.to_int32()?;
    let merge_factor = spatial_merge_size * spatial_merge_size;
    let mut num_tokens = 0usize;
    for i in 0..(grid_data.len() / 3) {
        let t = grid_data[i * 3];
        let h = grid_data[i * 3 + 1];
        let w = grid_data[i * 3 + 2];
        num_tokens += ((t * h * w) / merge_factor) as usize;
    }
    Ok(num_tokens)
}

/// Ensure image token placeholders are present in the tokenized output.
///
/// If the chat template didn't inject `IMAGE_TOKEN_ID` placeholders,
/// splice `num_image_tokens` of them after position 0 (after BOS).
/// Always returns an owned Vec.
pub(crate) fn inject_image_placeholders(tokens: &[u32], num_image_tokens: usize) -> Vec<u32> {
    let existing = tokens
        .iter()
        .filter(|&&t| t == IMAGE_TOKEN_ID as u32)
        .count();
    if num_image_tokens > 0 && existing == 0 {
        let mut new_tokens = tokens.to_vec();
        let placeholders: Vec<u32> = vec![IMAGE_TOKEN_ID as u32; num_image_tokens];
        new_tokens.splice(1..1, placeholders);
        new_tokens
    } else {
        tokens.to_vec()
    }
}

/// Compute M-RoPE position IDs for VLM
///
/// Text tokens get sequential positions [0, 1, 2, ...].
/// Image tokens get 2D spatial positions based on grid_thw.
///
/// Returns (position_ids [3, B, T], rope_deltas)
pub(crate) fn get_rope_index(
    input_ids: &MxArray,
    image_grid_thw: Option<&MxArray>,
    spatial_merge_size: i32,
    image_token_id: i32,
) -> Result<(MxArray, i64)> {
    let shape = input_ids.shape()?;
    let batch_size = shape[0];
    let seq_len = shape[1];

    // If no images, use simple sequential positions
    if image_grid_thw.is_none() {
        let pos = MxArray::arange(0.0, seq_len as f64, Some(1.0), None)?;
        let pos = pos.reshape(&[1, 1, seq_len])?;
        let position_ids = MxArray::tile(&pos, &[3, batch_size as i32, 1])?;
        return Ok((position_ids, 0));
    }

    let grid_thw = image_grid_thw.unwrap();
    let input_ids_data = input_ids.to_int32()?;
    grid_thw.eval();
    let grid_data = grid_thw.to_int32()?;

    let mut all_position_ids: Vec<Vec<i64>> = vec![Vec::new(); 3];

    for batch_idx in 0..batch_size as usize {
        let start = batch_idx * seq_len as usize;
        let end = start + seq_len as usize;
        let batch_tokens: Vec<i32> = input_ids_data[start..end].to_vec();

        let mut image_positions: Vec<usize> = Vec::new();
        for (i, &token) in batch_tokens.iter().enumerate() {
            if token == image_token_id {
                image_positions.push(i);
            }
        }

        if image_positions.is_empty() {
            for i in 0..seq_len {
                all_position_ids[0].push(i);
                all_position_ids[1].push(i);
                all_position_ids[2].push(i);
            }
            continue;
        }

        let num_images = grid_data.len() / 3;
        if num_images == 0 || grid_data.len() % 3 != 0 {
            return Err(Error::new(
                Status::InvalidArg,
                format!("grid_data must have 3N elements, got {}", grid_data.len()),
            ));
        }

        // Calculate token info for each image
        let mut image_token_info: Vec<(i64, i64, i64, usize)> = Vec::new();
        let mut total_expected_tokens = 0usize;

        for img_idx in 0..num_images {
            let t = grid_data[img_idx * 3] as i64;
            let h = grid_data[img_idx * 3 + 1] as i64;
            let w = grid_data[img_idx * 3 + 2] as i64;

            let llm_grid_t = t;
            let llm_grid_h = h / spatial_merge_size as i64;
            let llm_grid_w = w / spatial_merge_size as i64;
            let num_tokens = (llm_grid_t * llm_grid_h * llm_grid_w) as usize;

            image_token_info.push((llm_grid_t, llm_grid_h, llm_grid_w, num_tokens));
            total_expected_tokens += num_tokens;
        }

        if total_expected_tokens != image_positions.len() {
            return Err(Error::new(
                Status::GenericFailure,
                format!(
                    "Image token count mismatch: expected {} from grid, found {} in prompt",
                    total_expected_tokens,
                    image_positions.len()
                ),
            ));
        }

        // Build position IDs
        let image_start = image_positions[0];
        let image_end = image_positions[image_positions.len() - 1] + 1;

        // Text before images: sequential
        for i in 0..image_start {
            all_position_ids[0].push(i as i64);
            all_position_ids[1].push(i as i64);
            all_position_ids[2].push(i as i64);
        }

        // Image tokens: 2D spatial positions
        let mut current_pos = image_start as i64;
        let mut max_pos = image_start as i64;

        for (llm_grid_t, llm_grid_h, llm_grid_w, _) in &image_token_info {
            for t_idx in 0..*llm_grid_t {
                for h_idx in 0..*llm_grid_h {
                    for w_idx in 0..*llm_grid_w {
                        all_position_ids[0].push(current_pos + t_idx);
                        all_position_ids[1].push(current_pos + h_idx);
                        all_position_ids[2].push(current_pos + w_idx);
                    }
                }
            }
            let img_max = current_pos
                + std::cmp::max(
                    *llm_grid_t - 1,
                    std::cmp::max(*llm_grid_h - 1, *llm_grid_w - 1),
                );
            max_pos = std::cmp::max(max_pos, img_max);
            current_pos = img_max + 1;
        }

        // Text after images: continue from max
        let next_pos = max_pos + 1;
        for i in image_end..seq_len as usize {
            let pos = next_pos + (i - image_end) as i64;
            all_position_ids[0].push(pos);
            all_position_ids[1].push(pos);
            all_position_ids[2].push(pos);
        }
    }

    // Convert to MxArray [3, batch, seq_len]
    let t_positions: Vec<i32> = all_position_ids[0].iter().map(|&x| x as i32).collect();
    let h_positions: Vec<i32> = all_position_ids[1].iter().map(|&x| x as i32).collect();
    let w_positions: Vec<i32> = all_position_ids[2].iter().map(|&x| x as i32).collect();

    let t_arr = MxArray::from_int32(&t_positions, &[batch_size, seq_len])?;
    let h_arr = MxArray::from_int32(&h_positions, &[batch_size, seq_len])?;
    let w_arr = MxArray::from_int32(&w_positions, &[batch_size, seq_len])?;

    let position_ids = MxArray::stack(vec![&t_arr, &h_arr, &w_arr], Some(0))?;

    let max_position = *all_position_ids[0].iter().max().unwrap_or(&0);
    let rope_deltas = max_position + 1 - seq_len;

    Ok((position_ids, rope_deltas))
}

/// Merge image features into input embeddings at image token positions
pub(crate) fn merge_input_ids_with_image_features(
    image_token_id: i32,
    image_features: &MxArray,
    inputs_embeds: &MxArray,
    input_ids: &MxArray,
) -> Result<MxArray> {
    let input_shape = input_ids.shape()?;
    let batch_size = input_shape[0];

    let image_token = MxArray::scalar_int(image_token_id)?;
    let image_positions = input_ids.equal(&image_token)?;
    let inputs_embeds_shape = inputs_embeds.shape()?;
    let hidden_dim = inputs_embeds_shape[2];

    let mut batch_outputs: Vec<MxArray> = Vec::new();
    let mut feature_start_idx = 0i64;

    for batch_idx in 0..batch_size {
        let batch_mask = image_positions.slice_axis(0, batch_idx, batch_idx + 1)?;
        let batch_mask = batch_mask.squeeze(Some(&[0]))?;

        let mask_sum = batch_mask.sum(None, None)?;
        let num_positions = mask_sum.to_int32()?[0] as i64;

        if num_positions > 0 {
            let batch_features = image_features.slice_axis(
                0,
                feature_start_idx,
                feature_start_idx + num_positions,
            )?;

            let batch_embeds = inputs_embeds.slice_axis(0, batch_idx, batch_idx + 1)?;
            let batch_embeds = batch_embeds.squeeze(Some(&[0]))?;

            let mask_int = batch_mask.astype(crate::array::DType::Int32)?;
            let cumsum = mask_int.cumsum(0)?;

            let ones = MxArray::scalar_int(1)?;
            let feature_indices = cumsum.sub(&ones)?;
            let zeros =
                MxArray::zeros(&feature_indices.shape()?, Some(crate::array::DType::Int32))?;
            let feature_indices = batch_mask.where_(&feature_indices, &zeros)?;

            let gathered_features = batch_features.take(&feature_indices, 0)?;

            let mask_expanded = batch_mask.reshape(&[-1, 1])?;
            let mask_expanded =
                MxArray::broadcast_to(&mask_expanded, &[batch_mask.shape()?[0], hidden_dim])?;

            let batch_output = mask_expanded.where_(&gathered_features, &batch_embeds)?;
            batch_outputs.push(batch_output);
            feature_start_idx += num_positions;
        } else {
            let batch_embeds = inputs_embeds.slice_axis(0, batch_idx, batch_idx + 1)?;
            batch_outputs.push(batch_embeds.squeeze(Some(&[0]))?);
        }
    }

    let refs: Vec<&MxArray> = batch_outputs.iter().collect();
    MxArray::stack(refs, Some(0))
}

/// VLM prefill: processes images through vision encoder, merges with text embeddings,
/// and runs the prefill forward pass with M-RoPE position IDs.
///
/// Returns (first_logits [1, vocab], rope_deltas).
///
/// This is a free function (not a method) since it needs mutable lock guards passed in,
/// matching the pattern of `forward_inner()`.
#[allow(clippy::too_many_arguments)]
fn vlm_prefill(
    input_ids: &MxArray,
    image_cache_key: u64,
    pre_processed: &ProcessedImages,
    vision_encoder: &Qwen3_5VisionEncoder,
    spatial_merge_size: i32,
    text_model_embedding: &MxArray,
    layers_guard: &mut [DecoderLayer],
    caches_guard: &mut Option<Vec<Qwen3_5LayerCache>>,
    final_norm_guard: &RMSNorm,
    lm_head_guard: &Option<Linear>,
    model_config: &Qwen3_5Config,
    max_new_tokens: i32,
    generation_stream: Stream,
    vision_cache: &VisionCache,
    model_id: u64,
) -> Result<(MxArray, i64, bool)> {
    use crate::array::clear_cache;

    let (inputs_embeds, position_ids, rope_deltas) = vlm_prepare_vision_features(
        input_ids,
        image_cache_key,
        pre_processed,
        vision_encoder,
        spatial_merge_size,
        text_model_embedding,
        generation_stream,
        vision_cache,
    )?;

    // === STEP 4: Prefill with M-RoPE ===
    let use_compiled = unsafe { mlx_sys::mlx_qwen35_get_model_id() } == model_id;

    let (last_logits, _seq_len, compiled_init_done) = if use_compiled {
        // C++ VLM prefill: runs all layers in one FFI call with M-RoPE
        use mlx_sys as sys;

        let seq_len_i32 = inputs_embeds.shape_at(1)? as i32;
        let max_kv_len = ((seq_len_i32 + max_new_tokens + 255) / 256) * 256;
        let mrope_section: [i32; 3] = [11, 11, 10]; // Qwen3.5-VL

        let mut output_ptr: *mut sys::mlx_array = std::ptr::null_mut();
        unsafe {
            sys::mlx_qwen35_vlm_prefill(
                inputs_embeds.as_raw_ptr(),
                position_ids.as_raw_ptr(),
                model_config.num_layers,
                model_config.hidden_size,
                model_config.num_heads,
                model_config.num_kv_heads,
                model_config.head_dim,
                model_config.rope_theta as f32,
                model_config.rope_dims(),
                model_config.rms_norm_eps as f32,
                model_config.full_attention_interval,
                model_config.linear_num_key_heads,
                model_config.linear_num_value_heads,
                model_config.linear_key_head_dim,
                model_config.linear_value_head_dim,
                model_config.linear_conv_kernel_dim,
                if model_config.tie_word_embeddings {
                    1
                } else {
                    0
                },
                max_kv_len,
                1, // batch_size
                mrope_section.as_ptr(),
                rope_deltas as i32,
                &mut output_ptr,
            );
        }

        if output_ptr.is_null() {
            return Err(Error::from_reason(
                "C++ VLM prefill returned null — check stderr for details",
            ));
        }

        // Transfer VLM caches to compiled decode path
        // vlm_get_cache returns heap-allocated copies — we must delete them after use
        let num_caches = unsafe { sys::mlx_qwen35_vlm_cache_count() };
        let mut cache_ptrs: Vec<*mut sys::mlx_array> = Vec::with_capacity(num_caches as usize);
        for idx in 0..num_caches {
            cache_ptrs.push(unsafe { sys::mlx_qwen35_vlm_get_cache(idx) });
        }

        unsafe {
            sys::mlx_qwen35_compiled_init_from_prefill(
                model_config.num_layers,
                model_config.hidden_size,
                model_config.num_heads,
                model_config.num_kv_heads,
                model_config.head_dim,
                model_config.rope_theta as f32,
                model_config.rope_dims(),
                model_config.rms_norm_eps as f32,
                model_config.full_attention_interval,
                model_config.linear_num_key_heads,
                model_config.linear_num_value_heads,
                model_config.linear_key_head_dim,
                model_config.linear_value_head_dim,
                model_config.linear_conv_kernel_dim,
                if model_config.tie_word_embeddings {
                    1
                } else {
                    0
                },
                max_kv_len,
                1,
                cache_ptrs.as_mut_ptr(),
                seq_len_i32,
            );

            // Adjust offset for rope_deltas (VLM positions differ from sequential)
            if rope_deltas != 0 {
                sys::mlx_qwen35_compiled_adjust_offset(rope_deltas as i32);
            }

            // Clean up heap-allocated cache copies from vlm_get_cache
            for ptr in &cache_ptrs {
                if !ptr.is_null() {
                    sys::mlx_array_delete(*ptr);
                }
            }

            // Clean up VLM prefill state (caches now owned by compiled decode)
            sys::mlx_qwen35_vlm_reset();
        }

        let logits = MxArray::from_handle(output_ptr, "vlm_cpp_prefill")?;
        // logits is already [1, vocab] from C++ prefill
        (logits, seq_len_i32 as i64, true) // compiled init done
    } else {
        // Rust fallback prefill (when C++ weights not loaded, e.g. test models)
        // Init fresh caches
        *caches_guard = Some(
            (0..model_config.num_layers as usize)
                .map(|i| {
                    if model_config.is_linear_layer(i) {
                        super::layer_cache::Qwen3_5LayerCache::new_linear()
                    } else {
                        super::layer_cache::Qwen3_5LayerCache::new_full_attention()
                    }
                })
                .collect(),
        );

        let logits = {
            let _stream_ctx = StreamContext::new(generation_stream);

            let mut h = inputs_embeds.clone();

            // No explicit mask — Qwen3_5Attention uses "causal" SDPA mode
            let num_layers = layers_guard.len();
            for i in 0..num_layers {
                let cache = caches_guard.as_mut().map(|c| &mut c[i]);
                let layer_pos = if layers_guard[i].is_linear() {
                    None
                } else {
                    Some(&position_ids)
                };
                h = layers_guard[i].forward(&h, None, cache, layer_pos, true)?;
            }

            let h = final_norm_guard.forward(&h)?;
            let logits = match lm_head_guard {
                Some(head) => head.forward(&h)?,
                None => {
                    let weight_t = text_model_embedding.transpose(Some(&[1, 0]))?;
                    h.matmul(&weight_t)?
                }
            };

            if let Some(ref caches) = *caches_guard {
                let mut cache_arrays: Vec<&MxArray> = Vec::new();
                for cache in caches.iter() {
                    cache.collect_arrays(&mut cache_arrays);
                }
                if !cache_arrays.is_empty() {
                    MxArray::async_eval_arrays(&cache_arrays);
                }
            }
            clear_cache();

            logits
        };

        let seq_len = logits.shape_at(1)?;
        let last_logits = logits.slice_axis(1, seq_len - 1, seq_len)?;
        let last_logits = last_logits.squeeze(Some(&[1]))?;
        (last_logits, seq_len, false) // compiled init NOT done
    };

    Ok((last_logits, rope_deltas, compiled_init_done))
}

/// Shared VLM prefill steps 1-3: vision cache lookup, vision encoder,
/// embedding merge, and M-RoPE position computation.
///
/// Returns (inputs_embeds, position_ids, rope_deltas) ready for the
/// language model forward pass. Used by both dense and MoE VLM prefill.
#[allow(clippy::too_many_arguments)]
pub(crate) fn vlm_prepare_vision_features(
    input_ids: &MxArray,
    image_cache_key: u64,
    pre_processed: &ProcessedImages,
    vision_encoder: &Qwen3_5VisionEncoder,
    spatial_merge_size: i32,
    text_model_embedding: &MxArray,
    generation_stream: Stream,
    vision_cache: &VisionCache,
) -> Result<(MxArray, MxArray, i64)> {
    // === STEP 1: Compute vision features (with hash cache) ===
    let combined_hash = image_cache_key;

    let cached = {
        let mut cache = vision_cache
            .lock()
            .map_err(|_| Error::from_reason("Vision cache mutex poisoned"))?;
        cache.generation += 1;
        let lru_gen = cache.generation;
        if let Some((features, grid, lru)) = cache.entries.get_mut(&combined_hash) {
            *lru = lru_gen;
            tracing::debug!("Vision cache HIT for hash {:016x}", combined_hash);
            Some((features.clone(), grid.clone()))
        } else {
            None
        }
    };

    let (vision_features, grid) = if let Some((features, grid)) = cached {
        (features, grid)
    } else {
        let grid = pre_processed.grid_thw();
        let pv = pre_processed.pixel_values();
        let pv_shape = pv.shape()?;
        let pv_5d = pv.reshape(&[1, pv_shape[0], pv_shape[1], pv_shape[2], pv_shape[3]])?;

        let features = {
            let _stream_ctx = StreamContext::new(generation_stream);
            vision_encoder.forward(&pv_5d, &grid)?
        };

        {
            let mut cache = vision_cache
                .lock()
                .map_err(|_| Error::from_reason("Vision cache mutex poisoned"))?;
            if cache.entries.len() >= VISION_CACHE_MAX_ENTRIES
                && let Some((&oldest_key, _)) =
                    cache.entries.iter().min_by_key(|(_, (_, _, lru))| *lru)
            {
                cache.entries.remove(&oldest_key);
            }
            cache.generation += 1;
            let lru_gen = cache.generation;
            cache
                .entries
                .insert(combined_hash, (features.clone(), grid.clone(), lru_gen));
        }
        tracing::debug!("Vision cache MISS for hash {:016x}", combined_hash);

        (features, grid)
    };

    // === STEP 2: Get text embeddings and merge with vision features ===
    let text_embeds = {
        let _stream_ctx = StreamContext::new(generation_stream);
        let embedding = Embedding::from_weight(text_model_embedding)?;
        embedding.forward(input_ids)?
    };

    let inputs_embeds = {
        let _stream_ctx = StreamContext::new(generation_stream);
        let embed_dtype = text_embeds.dtype()?;
        let vf_cast = if vision_features.dtype()? != embed_dtype {
            vision_features.astype(embed_dtype)?
        } else {
            vision_features
        };
        merge_input_ids_with_image_features(IMAGE_TOKEN_ID, &vf_cast, &text_embeds, input_ids)?
    };

    // === STEP 3: Compute M-RoPE position IDs ===
    let (position_ids, rope_deltas) =
        get_rope_index(input_ids, Some(&grid), spatial_merge_size, IMAGE_TOKEN_ID)?;

    tracing::debug!(
        "VLM prefill: seq_len={}, rope_deltas={}",
        inputs_embeds.shape_at(1)?,
        rope_deltas
    );

    Ok((inputs_embeds, position_ids, rope_deltas))
}
