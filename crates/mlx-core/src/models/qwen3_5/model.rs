use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi_derive::napi;
use tracing::{info, warn};

use crate::array::MxArray;
use crate::inference_trace::{
    elapsed_ms, enabled as inference_trace_enabled, write as write_inference_trace,
};
use crate::model_thread::{ResponseTx, StreamTx};
use crate::nn::{Embedding, Linear, RMSNorm};
use crate::sampling::{SamplingConfig, sample};
use crate::stream::{DeviceType, Stream, StreamContext};
use crate::tokenizer::{ChatMessage, Qwen3Tokenizer, ToolDefinition};
use crate::tools::ToolCallResult;
use crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter;

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

/// Process-wide mutex serializing the compiled forward LIFECYCLE (per-turn
/// init / decode / reset) across model instances AND model families.
///
/// Within a single model instance the dedicated model thread serializes calls,
/// but distinct models run on distinct OS threads (one per model, see
/// `model_thread.rs`), so a qwen3.5 compiled decode and an lfm2 compiled decode
/// genuinely run in parallel. They collide on the SAME process-global C++
/// globals: the `g_weights()` map (read by the NOT id-aware `get_weight` /
/// `get_weight_t`), the shared `g_active_model_id` atom, and each family's
/// compiled decode state (`g_*_caches` / `g_*_offset_int`).
///
/// `pub(crate)` and family-agnostic by design: EVERY model that drives a
/// compiled path over the shared registry (qwen3.5 dense + MoE, lfm2, …) MUST
/// serialize its compiled lifecycle on THIS one instance — never a private
/// per-family mutex, which would provide zero mutual exclusion against an
/// in-flight compiled decode from another family.
pub(crate) static COMPILED_LIFECYCLE_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn fresh_dense_layer_caches(config: &Qwen3_5Config) -> Vec<Qwen3_5LayerCache> {
    (0..config.num_layers as usize)
        .map(|i| {
            if config.is_linear_layer(i) {
                Qwen3_5LayerCache::new_linear()
            } else {
                Qwen3_5LayerCache::new_full_attention()
            }
        })
        .collect()
}

const DENSE_GDN_PREFIX_CHECKPOINT_LIMIT: usize = 8;

struct DenseGdnPrefixCheckpoint {
    prefix_len: u32,
    block_size: u32,
    final_block_hash: u64,
    tokens: Vec<u32>,
    caches: Vec<Qwen3_5LayerCache>,
}

struct DenseGdnHistoryCheckpoint {
    tokens: Vec<u32>,
    caches: Vec<Qwen3_5LayerCache>,
}

struct DenseGdnPrefixPreparation {
    state: &'static str,
    already_primed: bool,
}

#[derive(Default)]
struct DenseGdnCheckpointStoreTrace {
    stored: bool,
    hash_ms: f64,
    eval_ms: f64,
    clone_ms: f64,
    token_clone_ms: f64,
    update_ms: f64,
    total_ms: f64,
}

impl DenseGdnCheckpointStoreTrace {
    fn finish(mut self, start: Option<std::time::Instant>) -> Self {
        self.total_ms = start.map(elapsed_ms).unwrap_or(0.0);
        self
    }
}

fn dense_gdn_store_replayed_prefix_checkpoint_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        crate::inference_trace::env_flag_enabled("MLX_DENSE_GDN_REPLAY_PREFIX_CHECKPOINT")
    })
}

#[derive(Clone, Copy)]
struct TokenPrefixMismatchTrace {
    index: i64,
    prompt_token: i64,
    cached_token: i64,
}

impl Default for TokenPrefixMismatchTrace {
    fn default() -> Self {
        Self {
            index: -1,
            prompt_token: -1,
            cached_token: -1,
        }
    }
}

fn token_prefix_mismatch_trace(prompt: &[u32], cached: &[u32]) -> TokenPrefixMismatchTrace {
    let common_len = prompt.len().min(cached.len());
    for i in 0..common_len {
        if prompt[i] != cached[i] {
            return TokenPrefixMismatchTrace {
                index: i as i64,
                prompt_token: prompt[i] as i64,
                cached_token: cached[i] as i64,
            };
        }
    }

    TokenPrefixMismatchTrace {
        index: common_len as i64,
        prompt_token: prompt.get(common_len).map_or(-1, |token| *token as i64),
        cached_token: cached.get(common_len).map_or(-1, |token| *token as i64),
    }
}

fn dense_paged_linear_caches_ready(
    config: &Qwen3_5Config,
    caches: Option<&[Qwen3_5LayerCache]>,
) -> bool {
    let Some(caches) = caches else {
        return false;
    };
    if caches.len() != config.num_layers as usize {
        return false;
    }
    for (i, cache) in caches.iter().enumerate() {
        if !config.is_linear_layer(i) {
            continue;
        }
        let Qwen3_5LayerCache::Linear(arrays) = cache else {
            return false;
        };
        if arrays.get(0).is_none() || arrays.get(1).is_none() {
            return false;
        }
    }
    true
}

fn clone_dense_linear_layer_caches(
    config: &Qwen3_5Config,
    caches: &[Qwen3_5LayerCache],
) -> Option<Vec<Qwen3_5LayerCache>> {
    if !dense_paged_linear_caches_ready(config, Some(caches)) {
        return None;
    }

    let mut cloned = fresh_dense_layer_caches(config);
    for i in 0..config.num_layers as usize {
        if !config.is_linear_layer(i) {
            continue;
        }
        let Qwen3_5LayerCache::Linear(arrays) = &caches[i] else {
            return None;
        };
        cloned[i] = Qwen3_5LayerCache::Linear(arrays.clone());
    }
    Some(cloned)
}

fn compute_paged_prefix_block_hash(
    tokens: &[u32],
    prefix_len: u32,
    block_size: u32,
    extra_keys_per_block: &[Vec<u64>],
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
        let extra_keys = extra_keys_per_block.get(block_idx)?;
        let start = block_idx * block_size;
        let end = start + block_size;
        parent_hash = if block_idx == 0 && cache_salt != 0 {
            let mut salted_keys = Vec::with_capacity(extra_keys.len() + 1);
            salted_keys.extend_from_slice(extra_keys);
            salted_keys.push(cache_salt);
            mlx_paged_attn::hash_tokens(&tokens[start..end], parent_hash, &salted_keys)
        } else {
            mlx_paged_attn::hash_tokens(&tokens[start..end], parent_hash, extra_keys)
        };
    }

    Some(parent_hash)
}

fn export_paged_dense_linear_caches(
    config: &Qwen3_5Config,
) -> Result<Option<Vec<Qwen3_5LayerCache>>> {
    let num_layers = config.num_layers as usize;
    let expected = num_layers
        .checked_mul(2)
        .ok_or_else(|| Error::from_reason("paged dense cache export size overflow"))?;
    let mut export_ptrs: Vec<*mut mlx_sys::mlx_array> = vec![std::ptr::null_mut(); expected];
    let exported = unsafe {
        mlx_sys::mlx_qwen35_export_paged_linear_caches(export_ptrs.as_mut_ptr(), expected as i32)
    };
    if exported == 0 {
        return Ok(None);
    }
    if exported != expected as i32 {
        return Err(Error::from_reason(format!(
            "paged dense linear cache export returned {exported} arrays; expected {expected}"
        )));
    }

    let cache_offset = unsafe { mlx_sys::mlx_qwen35_get_paged_cache_offset() };
    let mut new_caches = fresh_dense_layer_caches(config);
    for i in 0..num_layers {
        if !config.is_linear_layer(i) {
            continue;
        }
        let p0 = export_ptrs[i * 2];
        let p1 = export_ptrs[i * 2 + 1];
        if p0.is_null() || p1.is_null() {
            return Err(Error::from_reason(format!(
                "paged dense linear cache export missing layer {i}"
            )));
        }
        new_caches[i].import_ptrs(p0, p1, cache_offset);
    }

    Ok(Some(new_caches))
}

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
    gdn_prefix_checkpoints: VecDeque<DenseGdnPrefixCheckpoint>,
    gdn_last_history_checkpoint: Option<DenseGdnHistoryCheckpoint>,
    /// Block-paged KV adapter (vLLM-style refcounted prefix cache) for
    /// full-attention layers.
    ///
    /// **Opt-in via `Qwen3_5Config::use_block_paged_cache`** — see the
    /// flag's rustdoc for the full architectural rationale. When
    /// `Some(...)`, full-attention layers route through this adapter
    /// while linear-attention (GDN) layers stay on
    /// `Qwen3_5LayerCache::Linear` with no cross-request prefix reuse.
    /// The compiled C++ forward path is bypassed entirely on the paged
    /// path.
    pub(crate) paged_adapter: Option<PagedKVCacheAdapter>,
    /// Training state owned by the model thread.
    /// Created when `InitTraining` command is received, destroyed when training ends.
    pub(crate) training_state: Option<crate::training_state::ModelThreadTrainingState>,
}

/// Commands dispatched from NAPI methods to the dedicated model thread.
pub(crate) enum Qwen35Cmd {
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
    ///
    /// `is_error` is the structured tool-error signal threaded through
    /// from the NAPI surface (`chatSessionContinueTool(..., isError)`).
    /// When `Some(true)`, the renderer prepends the shared
    /// [`crate::tokenizer::TOOL_ERROR_MARKER`] inside the
    /// `<tool_response>` wrapper. `None` / `Some(false)` produce the
    /// pre-feature byte-equal output.
    ChatSessionContinueTool {
        tool_call_id: String,
        content: String,
        is_error: Option<bool>,
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
    /// streams token deltas through `stream_tx`. Carries the same
    /// structured `is_error` signal.
    ChatStreamSessionContinueTool {
        tool_call_id: String,
        content: String,
        is_error: Option<bool>,
        config: ChatConfig,
        stream_tx: StreamTx<ChatStreamChunk>,
        cancelled: Arc<AtomicBool>,
    },
    Generate {
        prompt_tokens: MxArray,
        config: Qwen3_5GenerationConfig,
        reply: ResponseTx<Qwen3_5GenerationResult>,
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
        Qwen35Cmd::ChatSessionStart {
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
            is_error,
            config,
            reply,
        } => {
            let _ = reply.send(inner.chat_session_continue_tool_sync(
                tool_call_id,
                content,
                is_error,
                config,
            ));
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
            is_error,
            stream_tx,
            config,
            cancelled,
        } => {
            inner.chat_stream_session_continue_tool_sync(
                tool_call_id,
                content,
                is_error,
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
/// Constructed by the prefill-side of [`Qwen35Inner::chat_sync_core`] and
/// [`Qwen35Inner::chat_tokens_delta_sync`].
///
/// The caller is responsible for:
///   - acquiring `COMPILED_LIFECYCLE_MUTEX` and `COMPILED_WEIGHTS_RWLOCK` in
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
    /// `true` when this invocation is a session DELTA continuation
    /// (text-only append on top of the live KV cache). Drives the
    /// post-decode save pathway: deltas keep `cached_image_key` sticky
    /// so image attention state baked into the KV caches by a prior
    /// prefill stays addressable; prefills (re)set the key based on
    /// the fresh turn's `has_images`.
    pub is_delta: bool,

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
    /// End-of-sequence token id for the decode loop. For `chat_sync_core` this
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
    /// Length of the reused cached prefix to report on `ChatResult.cached_tokens`.
    ///
    /// For fresh prefills this is `cached_prefix_len` (0 on a miss, full
    /// cached length on an exact-append hit). For the session delta path
    /// this is the full prior-history length because the delta is
    /// appended on top of the existing caches — we skip the `cached_prefix_len`
    /// driver (which only gates the VLM rope-delta replay branch) while
    /// still reporting the reused prefix accurately for observability.
    pub cached_tokens_for_result: u32,

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

        // Block-paged KV adapter — opt-in via `use_block_paged_cache`.
        //
        // VLM checkpoints can co-exist with paged dispatch for
        // text-only inference: the M-RoPE / vision-feature path is
        // only invoked when an input message carries images, and the
        // chat-entry sites reject `has_images && paged_adapter` at
        // runtime. Text-only forward (`Qwen3_5Attention::forward` with
        // `position_ids = None`) and the paged forward
        // (`Qwen3_5Attention::forward_paged`) both go through standard
        // `self.rope`, so byte-equal parity holds on text-only inputs
        // even on VLM weights.
        let paged_adapter = if config.use_block_paged_cache.unwrap_or(false) {
            let attn_layer_count = config.full_attention_layer_count() as u32;
            if attn_layer_count == 0 {
                return Err(Error::from_reason(
                    "Qwen3.5 block-paged adapter: config has no full_attention layers; \
                     paged KV cache requires at least one attention layer. Check \
                     full_attention_interval.",
                ));
            }

            let block_size = config.paged_block_size.unwrap_or(16);
            let gpu_memory_mb = config.paged_cache_memory_mb.unwrap_or(2048);
            let head_size = config.head_dim as u32;
            let num_kv_heads = config.num_kv_heads as u32;

            let pa_config = mlx_paged_attn::PagedAttentionConfig {
                block_size,
                gpu_memory_mb,
                head_size,
                num_kv_heads,
                // Pool covers only the full-attention layers — GDN
                // (linear-attention) layers continue to use
                // `Qwen3_5LayerCache::Linear`.
                num_layers: attn_layer_count,
                use_fp8_cache: Some(false),
                max_seq_len: Some(config.max_position_embeddings as u32),
                max_batch_size: Some(32),
            };

            let num_blocks = pa_config.calculate_num_blocks();
            if num_blocks == 0 {
                return Err(Error::from_reason(format!(
                    "Qwen3.5 block-paged adapter: gpu_memory_mb={gpu_memory_mb} too small \
                     (head_size={head_size}, num_kv_heads={num_kv_heads}, \
                     block_size={block_size}, num_attn_layers={attn_layer_count})"
                )));
            }

            let allocator = Arc::new(std::sync::Mutex::new(mlx_paged_attn::BlockAllocator::new(
                num_blocks, block_size,
            )));

            let cache_dtype = mlx_paged_attn::metal::MetalDtype::BFloat16;
            let pool = mlx_paged_attn::LayerKVPool::new(pa_config, num_blocks, cache_dtype)
                .map_err(|e| {
                    Error::from_reason(format!(
                        "Failed to construct LayerKVPool for Qwen3.5 block-paged adapter: {e}"
                    ))
                })?;

            let adapter =
                PagedKVCacheAdapter::new(allocator, Arc::new(pool), block_size).map_err(|e| {
                    Error::from_reason(format!(
                        "Failed to construct Qwen3.5 PagedKVCacheAdapter: {e}"
                    ))
                })?;

            info!(
                "Qwen3.5 block-paged adapter enabled: num_blocks={}, block_size={}, \
                 gpu_memory_mb={}, num_attn_layers={}, cache_dtype=BFloat16",
                num_blocks, block_size, gpu_memory_mb, attn_layer_count
            );
            Some(adapter)
        } else {
            None
        };

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
            gdn_prefix_checkpoints: VecDeque::new(),
            gdn_last_history_checkpoint: None,
            paged_adapter,
            training_state: None,
        })
    }

    /// Initialize KV caches.
    pub(crate) fn init_caches_sync(&mut self) -> Result<()> {
        self.caches = Some(fresh_dense_layer_caches(&self.config));
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
        self.gdn_prefix_checkpoints.clear();
        self.gdn_last_history_checkpoint = None;
    }

    fn find_dense_gdn_history_checkpoint(
        &self,
        tokens: &[u32],
        prefix_len: u32,
    ) -> Option<Vec<Qwen3_5LayerCache>> {
        let prefix_tokens = tokens.get(..prefix_len as usize)?;
        let checkpoint = self.gdn_last_history_checkpoint.as_ref()?;
        if checkpoint.tokens.as_slice() != prefix_tokens {
            return None;
        }
        clone_dense_linear_layer_caches(&self.config, &checkpoint.caches)
    }

    fn remember_dense_gdn_history_checkpoint(&mut self) -> Result<DenseGdnCheckpointStoreTrace> {
        let trace_enabled = inference_trace_enabled();
        let total_start = trace_enabled.then(std::time::Instant::now);
        let mut trace = DenseGdnCheckpointStoreTrace::default();
        if self.cached_token_history.is_empty() {
            self.gdn_last_history_checkpoint = None;
            return Ok(trace.finish(total_start));
        }

        let eval_start = trace_enabled.then(std::time::Instant::now);
        eval_layer_caches(&self.caches)?;
        trace.eval_ms = eval_start.map(elapsed_ms).unwrap_or(0.0);
        let clone_start = trace_enabled.then(std::time::Instant::now);
        let Some(caches) = self
            .caches
            .as_ref()
            .and_then(|caches| clone_dense_linear_layer_caches(&self.config, caches))
        else {
            self.gdn_last_history_checkpoint = None;
            trace.clone_ms = clone_start.map(elapsed_ms).unwrap_or(0.0);
            return Ok(trace.finish(total_start));
        };
        trace.clone_ms = clone_start.map(elapsed_ms).unwrap_or(0.0);
        let token_clone_start = trace_enabled.then(std::time::Instant::now);
        let tokens = self.cached_token_history.clone();
        trace.token_clone_ms = token_clone_start.map(elapsed_ms).unwrap_or(0.0);

        let update_start = trace_enabled.then(std::time::Instant::now);
        self.gdn_last_history_checkpoint = Some(DenseGdnHistoryCheckpoint { tokens, caches });
        trace.update_ms = update_start.map(elapsed_ms).unwrap_or(0.0);
        trace.stored = true;
        Ok(trace.finish(total_start))
    }

    fn find_dense_gdn_prefix_checkpoint(
        &self,
        tokens: &[u32],
        prefix_len: u32,
        block_size: u32,
        extra_keys_per_block: &[Vec<u64>],
        cache_salt: u64,
    ) -> Option<Vec<Qwen3_5LayerCache>> {
        let final_block_hash = compute_paged_prefix_block_hash(
            tokens,
            prefix_len,
            block_size,
            extra_keys_per_block,
            cache_salt,
        )?;
        let prefix_tokens = tokens.get(..prefix_len as usize)?;

        self.gdn_prefix_checkpoints
            .iter()
            .rev()
            .find(|checkpoint| {
                checkpoint.prefix_len == prefix_len
                    && checkpoint.block_size == block_size
                    && checkpoint.final_block_hash == final_block_hash
                    && checkpoint.tokens.as_slice() == prefix_tokens
                    && dense_paged_linear_caches_ready(&self.config, Some(&checkpoint.caches))
            })
            .and_then(|checkpoint| {
                clone_dense_linear_layer_caches(&self.config, &checkpoint.caches)
            })
    }

    fn remember_dense_gdn_prefix_checkpoint(
        &mut self,
        tokens: &[u32],
        prefix_len: u32,
        block_size: u32,
        extra_keys_per_block: &[Vec<u64>],
        cache_salt: u64,
    ) -> Result<DenseGdnCheckpointStoreTrace> {
        let trace_enabled = inference_trace_enabled();
        let total_start = trace_enabled.then(std::time::Instant::now);
        let mut trace = DenseGdnCheckpointStoreTrace::default();
        let hash_start = trace_enabled.then(std::time::Instant::now);
        let Some(final_block_hash) = compute_paged_prefix_block_hash(
            tokens,
            prefix_len,
            block_size,
            extra_keys_per_block,
            cache_salt,
        ) else {
            trace.hash_ms = hash_start.map(elapsed_ms).unwrap_or(0.0);
            return Ok(trace.finish(total_start));
        };
        trace.hash_ms = hash_start.map(elapsed_ms).unwrap_or(0.0);
        let Some(prefix_tokens) = tokens.get(..prefix_len as usize) else {
            return Ok(trace.finish(total_start));
        };

        let eval_start = trace_enabled.then(std::time::Instant::now);
        eval_layer_caches(&self.caches)?;
        trace.eval_ms = eval_start.map(elapsed_ms).unwrap_or(0.0);
        let clone_start = trace_enabled.then(std::time::Instant::now);
        let Some(caches) = self
            .caches
            .as_ref()
            .and_then(|caches| clone_dense_linear_layer_caches(&self.config, caches))
        else {
            trace.clone_ms = clone_start.map(elapsed_ms).unwrap_or(0.0);
            return Ok(trace.finish(total_start));
        };
        trace.clone_ms = clone_start.map(elapsed_ms).unwrap_or(0.0);
        let token_clone_start = trace_enabled.then(std::time::Instant::now);
        let prefix_tokens = prefix_tokens.to_vec();
        trace.token_clone_ms = token_clone_start.map(elapsed_ms).unwrap_or(0.0);

        let update_start = trace_enabled.then(std::time::Instant::now);
        self.gdn_prefix_checkpoints.retain(|checkpoint| {
            !(checkpoint.prefix_len == prefix_len
                && checkpoint.block_size == block_size
                && checkpoint.final_block_hash == final_block_hash
                && checkpoint.tokens == prefix_tokens)
        });
        self.gdn_prefix_checkpoints
            .push_back(DenseGdnPrefixCheckpoint {
                prefix_len,
                block_size,
                final_block_hash,
                tokens: prefix_tokens,
                caches,
            });
        while self.gdn_prefix_checkpoints.len() > DENSE_GDN_PREFIX_CHECKPOINT_LIMIT {
            self.gdn_prefix_checkpoints.pop_front();
        }
        trace.update_ms = update_start.map(elapsed_ms).unwrap_or(0.0);
        trace.stored = true;

        Ok(trace.finish(total_start))
    }

    fn prepare_dense_gdn_prefix_state(
        &mut self,
        tokens: &[u32],
        cached_prefix_len: u32,
        block_size: u32,
        extra_keys_per_block: &[Vec<u64>],
        cache_salt: u64,
        continued_live_prefix: bool,
    ) -> Result<DenseGdnPrefixPreparation> {
        let trace_enabled = inference_trace_enabled();
        let prepare_trace_start = trace_enabled.then(std::time::Instant::now);
        let gdn_caches_ready =
            dense_paged_linear_caches_ready(&self.config, self.caches.as_deref());
        if gdn_caches_ready && continued_live_prefix {
            if let Some(start) = prepare_trace_start {
                write_inference_trace(format_args!(
                    "[MLX_TRACE] qwen3.5-dense gdn_prefix_prepare_done state=live \
                     cached_prefix_tokens={} elapsed_ms={:.1}",
                    cached_prefix_len,
                    elapsed_ms(start)
                ));
            }
            return Ok(DenseGdnPrefixPreparation {
                state: "live",
                already_primed: true,
            });
        }

        let gdn_prefix_from_history = cached_prefix_len > 0
            && self.cached_token_history.len() == cached_prefix_len as usize
            && tokens.starts_with(&self.cached_token_history);
        if gdn_caches_ready && gdn_prefix_from_history {
            if let Some(start) = prepare_trace_start {
                write_inference_trace(format_args!(
                    "[MLX_TRACE] qwen3.5-dense gdn_prefix_prepare_done state=last_history \
                     cached_prefix_tokens={} elapsed_ms={:.1}",
                    cached_prefix_len,
                    elapsed_ms(start)
                ));
            }
            return Ok(DenseGdnPrefixPreparation {
                state: "last_history",
                already_primed: true,
            });
        }

        if cached_prefix_len > 0 {
            let history_lookup_start = trace_enabled.then(std::time::Instant::now);
            let history_checkpoint =
                self.find_dense_gdn_history_checkpoint(tokens, cached_prefix_len);
            let history_lookup_ms = history_lookup_start.map(elapsed_ms);
            if let Some(checkpoint) = history_checkpoint {
                self.caches = Some(checkpoint);
                if let Some(start) = prepare_trace_start {
                    write_inference_trace(format_args!(
                        "[MLX_TRACE] qwen3.5-dense gdn_prefix_prepare_done \
                         state=last_history_checkpoint cached_prefix_tokens={} \
                         history_lookup_ms={:.1} elapsed_ms={:.1}",
                        cached_prefix_len,
                        history_lookup_ms.unwrap_or(0.0),
                        elapsed_ms(start)
                    ));
                }
                return Ok(DenseGdnPrefixPreparation {
                    state: "last_history_checkpoint",
                    already_primed: true,
                });
            } else if trace_enabled {
                let history_checkpoint_len = self
                    .gdn_last_history_checkpoint
                    .as_ref()
                    .map_or(0, |checkpoint| checkpoint.tokens.len());
                let history_mismatch =
                    token_prefix_mismatch_trace(tokens, &self.cached_token_history);
                write_inference_trace(format_args!(
                    "[MLX_TRACE] qwen3.5-dense gdn_history_checkpoint_miss \
                     cached_prefix_tokens={} history_len={} checkpoint_len={} \
                     history_match={} history_mismatch_at={} prompt_token={} \
                     history_token={} history_lookup_ms={:.1}",
                    cached_prefix_len,
                    self.cached_token_history.len(),
                    history_checkpoint_len,
                    gdn_prefix_from_history,
                    history_mismatch.index,
                    history_mismatch.prompt_token,
                    history_mismatch.cached_token,
                    history_lookup_ms.unwrap_or(0.0)
                ));
            }
        }

        let prefix_lookup_start = trace_enabled.then(std::time::Instant::now);
        let prefix_checkpoint = self.find_dense_gdn_prefix_checkpoint(
            tokens,
            cached_prefix_len,
            block_size,
            extra_keys_per_block,
            cache_salt,
        );
        let prefix_lookup_ms = prefix_lookup_start.map(elapsed_ms);
        if let Some(checkpoint) = prefix_checkpoint {
            self.caches = Some(checkpoint);
            if let Some(start) = prepare_trace_start {
                write_inference_trace(format_args!(
                    "[MLX_TRACE] qwen3.5-dense gdn_prefix_prepare_done state=checkpoint \
                     cached_prefix_tokens={} prefix_lookup_ms={:.1} elapsed_ms={:.1}",
                    cached_prefix_len,
                    prefix_lookup_ms.unwrap_or(0.0),
                    elapsed_ms(start)
                ));
            }
            return Ok(DenseGdnPrefixPreparation {
                state: "checkpoint",
                already_primed: true,
            });
        }

        self.caches = Some(fresh_dense_layer_caches(&self.config));
        if cached_prefix_len == 0 {
            if let Some(start) = prepare_trace_start {
                write_inference_trace(format_args!(
                    "[MLX_TRACE] qwen3.5-dense gdn_prefix_prepare_done state=replay \
                     cached_prefix_tokens=0 prefix_lookup_ms={:.1} elapsed_ms={:.1}",
                    prefix_lookup_ms.unwrap_or(0.0),
                    elapsed_ms(start)
                ));
            }
            return Ok(DenseGdnPrefixPreparation {
                state: "replay",
                already_primed: false,
            });
        }

        let prefix = tokens.get(..cached_prefix_len as usize).ok_or_else(|| {
            Error::from_reason("dense paged GDN prefix replay length exceeds prompt length")
        })?;
        let embed = self.embedding.clone();
        let caches_ref = self
            .caches
            .as_mut()
            .ok_or_else(|| Error::from_reason("dense paged GDN prefix caches not initialized"))?;
        let replay_trace_start = trace_enabled.then(std::time::Instant::now);
        super::paged_forward::run_gdn_only_prefill(prefix, &embed, &mut self.layers, caches_ref)?;
        let replay_ms = replay_trace_start.map(elapsed_ms);
        let store_trace = if dense_gdn_store_replayed_prefix_checkpoint_enabled() {
            self.remember_dense_gdn_prefix_checkpoint(
                tokens,
                cached_prefix_len,
                block_size,
                extra_keys_per_block,
                cache_salt,
            )?
        } else {
            DenseGdnCheckpointStoreTrace::default()
        };
        if let Some(start) = prepare_trace_start {
            write_inference_trace(format_args!(
                "[MLX_TRACE] qwen3.5-dense gdn_prefix_prepare_done state={} \
                 cached_prefix_tokens={} prefix_lookup_ms={:.1} replay_ms={:.1} stored={} \
                 store_hash_ms={:.1} store_eval_ms={:.1} store_clone_ms={:.1} \
                 store_token_clone_ms={:.1} store_update_ms={:.1} store_ms={:.1} \
                 elapsed_ms={:.1}",
                if store_trace.stored {
                    "replay_store"
                } else {
                    "replay"
                },
                cached_prefix_len,
                prefix_lookup_ms.unwrap_or(0.0),
                replay_ms.unwrap_or(0.0),
                store_trace.stored,
                store_trace.hash_ms,
                store_trace.eval_ms,
                store_trace.clone_ms,
                store_trace.token_clone_ms,
                store_trace.update_ms,
                store_trace.total_ms,
                elapsed_ms(start)
            ));
        }

        Ok(DenseGdnPrefixPreparation {
            state: if store_trace.stored {
                "replay_store"
            } else {
                "replay"
            },
            already_primed: true,
        })
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

        let mut params_clone: HashMap<String, MxArray> =
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
        crate::utils::safetensors::save_safetensors(
            &safetensors_path,
            &mut params_clone,
            metadata,
        )?;
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
    ///
    /// Permits loading the vision encoder even when `paged_adapter` is
    /// active so VLM checkpoints can run text-only inference through
    /// the paged dispatch. The actual incompatibility is the
    /// M-RoPE / vision-feature plumbing on the paged forward path,
    /// which only fires when an input message carries images. The
    /// chat-entry sites (`chat_sync_core`, `chat_stream_sync_inner`,
    /// and the MoE counterparts) reject `has_images && paged_adapter`
    /// before dispatching, so text-only paged turns proceed normally
    /// while image turns surface a clear runtime error.
    ///
    /// For text-only inputs M-RoPE collapses to standard scalar-offset
    /// RoPE — `Qwen3_5Attention::forward` uses `self.rope` whenever
    /// `position_ids` is `None`, which is the case for every text-only
    /// flat call. The paged forward (`Qwen3_5Attention::forward_paged`)
    /// also goes through `self.rope` unconditionally. Both paths share
    /// the same RoPE on text-only inputs, so byte-equal parity holds
    /// on VLM checkpoints provided no images are passed.
    pub(crate) fn set_vision_encoder(&mut self, enc: Qwen3_5VisionEncoder) -> Result<()> {
        self.vision_encoder = Some(Arc::new(enc));
        Ok(())
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
    /// Delegates to [`Self::chat_sync_core`] with `<|im_end|>` (from the
    /// tokenizer vocab) as the stop token so the cached history ends on a
    /// clean ChatML boundary that subsequent `chat_session_continue_sync`
    /// / [`Self::chat_tokens_delta_sync`] calls can append a raw delta on
    /// top of without re-rendering the jinja template.
    ///
    /// Unlike the pre-refactor contract, this method no longer resets the
    /// caches up-front. The core path runs `verify_cache_prefix_direct`
    /// against the freshly-tokenized prompt and reuses the cached prefix
    /// on an exact-append hit or resets + fully prefills on a miss. This
    /// is what enables prefix-cache reuse for stateless agent clients
    /// that resend the full transcript on every turn. See the matching
    /// block comment in the method body for the GDN-safety rationale.
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

        // Prefix-cache reuse contract: the caches may carry state from a
        // prior session-start turn. `chat_sync_core` runs
        // `verify_cache_prefix_direct` against the freshly-tokenized prompt
        // and either (a) reuses the cached prefix and prefills only the
        // trailing delta (exact-append hit) or (b) resets + fully prefills
        // from scratch on a cache miss. Driving the reset from inside the
        // core — rather than wiping up-front here — is what lets
        // stateless-agent clients (Aider, Codex CLI, pi-mono, etc.) that
        // resend the full transcript on every turn avoid paying an O(N)
        // prefill cost on every turn.
        //
        // Safety: the invariant on `verify_cache_prefix_direct` (returns
        // either `0` or the full cached length — never an intermediate
        // value) guarantees a non-zero hit means the new tokens are a
        // pure *append* on the live caches, so Qwen3.5's GDN linear-
        // attention layers (whose recurrent state cannot be rewound
        // mid-sequence) stay consistent without any snapshot
        // machinery. See the rustdoc on `verify_cache_prefix_direct`.

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

        // Block-paged dispatch — early-return BEFORE the compile lock so
        // the paged path never acquires `COMPILED_LIFECYCLE_MUTEX` /
        // `COMPILED_WEIGHTS_RWLOCK` and never calls
        // `mlx_qwen35_compiled_init_from_prefill`. The compiled C++
        // forward path is incompatible with per-layer paged dispatch;
        // mixing the two on the same turn would corrupt subsequent
        // flat-path turns' compiled state.
        if self.paged_adapter.is_some() {
            if has_images {
                return Err(Error::from_reason(
                    "Qwen3.5 paged dispatch is text-only; image-bearing turns require \
                     use_block_paged_cache=false (text-only turns continue to work).",
                ));
            }
            return self.chat_sync_core_paged(tokens, tokenizer, eos_token_id, p, report_perf);
        }

        // Check if compiled path will be used
        let use_compiled = unsafe { mlx_sys::mlx_qwen35_get_model_id() } == model_id;

        // Serialize compiled lifecycle across model instances
        let _compiled_lock = if use_compiled {
            Some(
                COMPILED_LIFECYCLE_MUTEX
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
                let per_image_token_counts =
                    compute_image_token_counts_per_image(&processed_pre.grid_thw(), sms)?;
                let expanded = inject_image_placeholders(&tokens, &per_image_token_counts);
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

        // Zero-delta guard.
        //
        // Triggers when `cached_prefix_len == (expanded_)tokens.len()`, i.e.
        // the new prompt is byte-for-byte identical to the cached history
        // and there is literally no delta to prefill. The decode loop still
        // needs a `last_logits` to sample from, so we must run *some*
        // forward pass. Two options were considered:
        //   1. Trim every layer cache by one token and reprefill the final
        //      token. **Infeasible** for Qwen3.5 — the GDN linear-attention
        //      layers store a recurrent state (`conv_state`,
        //      `recurrent_state`) that cannot be rewound mid-sequence
        //      without corrupting the hidden representation. Only the
        //      full-attention layers support `KVCache::trim`, and applying
        //      trim to the hybrid stack would produce silent miscompiles.
        //   2. Full reset + full re-prefill. Wasteful when it triggers but
        //      always correct, and this branch is a cold edge case —
        //      real-world turns always append at least a user message, so
        //      the cached prefix is strictly shorter than the new prompt.
        //
        // We take option 2. The full-reset is intentional, not a "wrong
        // force-reset" to be patched out. See the invariant doc on
        // `verify_cache_prefix_direct` for the underlying linear-attention
        // rewind constraint.
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
            is_delta: false,
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
            // Fresh prefill: report the matched prefix length.
            cached_tokens_for_result: cached_prefix_len as u32,
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
        // A populated `cached_image_key` means the live KV cache carries
        // attention state for images seen on the preceding prefill. The
        // delta path appends a text delta on top of that — the image
        // context stays intact and the model can keep reasoning about
        // it. We do NOT reject here; the outer `chat_session_continue_*`
        // gate already rejects IMAGE-SET CHANGES (non-empty new images
        // that don't match the cached key) with a prefixed error the TS
        // `ChatSession` can catch and route through `chatSessionStart`.

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

        // Block-paged dispatch — early-return BEFORE the compile lock.
        // The delta path treats `cached_token_history + delta` as the
        // FULL prompt for paged purposes; the paged adapter's
        // warm-continue path picks up the matching prefix
        // automatically.
        if self.paged_adapter.is_some() {
            return self.chat_sync_core_paged(
                full_token_history.clone(),
                tokenizer.clone(),
                eos_id,
                p,
                report_perf,
            );
        }

        // Check compiled path availability (same contract as chat_sync_core).
        let use_compiled = unsafe { mlx_sys::mlx_qwen35_get_model_id() } == model_id;
        let _compiled_lock = if use_compiled {
            Some(
                COMPILED_LIFECYCLE_MUTEX
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

        // For the delta path there is no cached_prefix_len distinction for
        // the prefill branching — the caches already reflect the entire
        // prior history. However, for ChatResult observability we report
        // the cached-prefix length so clients can see the session delta
        // reused the full history. `prior_cached_len` feeds the reported
        // `cached_tokens`; `cached_prefix_len` stays 0 so the VLM
        // rope-deltas re-apply branch inside the decode helper is
        // skipped (text-only delta anyway).
        let prior_cached_len = full_token_history.len().saturating_sub(delta_tokens.len());
        let cached_prefix_len = 0usize;

        // For cache save, pass the full token history (cached + delta) as
        // `save_tokens`; the helper / `save_cache_state_direct` will append
        // the generated tokens.
        let save_tokens = full_token_history.clone();

        self.chat_with_caches_inner(ChatDecodeInputs {
            last_logits,
            seq_len: total_seq_len,
            vlm_compiled_init_done: false,
            is_delta: true,
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
            // Delta path reuses the full prior history by construction.
            cached_tokens_for_result: prior_cached_len as u32,
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

        // Match `chat_sync_core`'s sanitization so the session path is subject to
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
    ///
    /// `is_error` is forwarded verbatim to
    /// [`chat_common::build_chatml_tool_delta_text`]: `Some(true)` injects
    /// the shared [`crate::tokenizer::TOOL_ERROR_MARKER`] inside the
    /// `<tool_response>` wrapper; `None` / `Some(false)` keep the
    /// pre-feature byte-equal output.
    pub(crate) fn chat_session_continue_tool_sync(
        &mut self,
        tool_call_id: String,
        content: String,
        is_error: Option<bool>,
        config: ChatConfig,
    ) -> Result<ChatResult> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))?
            .clone();

        let enable_thinking = chat_common::resolve_enable_thinking(&config);
        let delta_text = chat_common::build_chatml_tool_delta_text(
            &tool_call_id,
            &content,
            enable_thinking,
            is_error,
        );

        let delta_tokens = tokenizer.encode_sync(&delta_text, Some(false))?;

        self.chat_tokens_delta_sync(delta_tokens, config)
    }

    /// Shared post-prefill pipeline: penalty → sample → compiled init (if needed)
    /// → decode loop → cache export → save cache state → finalize result.
    ///
    /// Extracted from `chat_sync_core` so it can also be driven by the text-only
    /// session path (`chat_tokens_delta_sync`). Preserves the exact semantics
    /// of `chat_sync_core` for the existing caller — `token_history_init` is the
    /// full pre-decode token sequence (used for penalty context and the decode
    /// loop's running history), and the decode loop mutates it in place.
    ///
    /// The caller is responsible for:
    /// - Holding the `COMPILED_LIFECYCLE_MUTEX` / `COMPILED_WEIGHTS_RWLOCK` guards
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
            is_delta,
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
            cached_tokens_for_result,
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

                // Re-apply the saved M-RoPE offset when the compiled
                // state is being (re)initialized from a KV cache that
                // encodes image attention — see
                // `chat_common::should_reapply_rope_delta`.
                if let Some(delta) = self.cached_rope_deltas
                    && chat_common::should_reapply_rope_delta(
                        true,
                        is_delta,
                        has_images,
                        cached_prefix_len,
                    )
                {
                    unsafe {
                        mlx_sys::mlx_qwen35_compiled_adjust_offset(delta);
                    }
                }
            }

            // Clear stale rope deltas only on fresh text-only prefills —
            // see `chat_common::should_clear_rope_delta`.
            if chat_common::should_clear_rope_delta(is_delta, has_images) {
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
                    eval_layer_caches(&self.caches)?;
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

        // Save cache state. Delta continuations preserve
        // `cached_image_key` — the KV cache still holds the prior
        // prefill's image attention state even though this turn was
        // text-only. Prefill paths (re)set the key based on the fresh
        // turn's `has_images`.
        if is_delta {
            chat_common::save_cache_state_after_delta(
                p.reuse_cache,
                &generated_tokens,
                &finish_reason,
                &save_tokens,
                &mut self.cached_token_history,
                &mut self.cached_image_key,
                &mut self.cached_rope_deltas,
                &mut self.caches,
            );
        } else {
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
        }

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

        let mut result = finalize_chat_result(
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
        )?;
        // Report the length of the reused cached prefix for observability.
        // Driven by the caller-supplied `cached_tokens_for_result`:
        //   - fresh prefill: equals `cached_prefix_len` (0 miss / full hit)
        //   - session delta: equals the prior-history length (full reuse)
        // See the invariant doc on `verify_cache_prefix_direct`.
        result.cached_tokens = cached_tokens_for_result;
        Ok(result)
    }

    /// Block-paged variant of [`Self::chat_sync_core`].
    ///
    /// Mirrors the flat path's control flow (penalty stack, decode
    /// loop, EOS / repetition cutoff, performance timing, output
    /// post-processing) but routes full-attention layers through
    /// `forward_paged_or_flat` against the paged KV adapter. GDN
    /// (linear-attention) layers continue to use their existing
    /// `Qwen3_5LayerCache::Linear(ArraysCache)` storage and are
    /// reset+re-prefilled every turn (no cross-request prefix reuse —
    /// vLLM's `MambaManager` stance).
    ///
    /// Per-turn lifecycle:
    /// 1. Adapter lifecycle: warm-continue when the prior turn ended
    ///    via `finalize_turn_keep_live`; cold-start (reset →
    ///    find_cached_prefix → allocate_suffix) otherwise.
    /// 2. Prepare GDN prefix state from live/session checkpoints when
    ///    available; otherwise replay the cached prefix through GDN.
    /// 3. Prefill via `paged_forward::run_paged_prefill_chunk`.
    /// 4. Decode loop via `paged_forward::run_paged_decode_step`.
    /// 5. End-of-turn: `finalize_turn_keep_live` keeps the partial
    ///    trailing block live for the next turn's warm
    ///    `continue_turn` (mirrors LFM2 / Qwen3).
    ///
    /// Limitations:
    /// * VLM is rejected upstream — paged dispatch is text-only.
    /// * Cross-turn GDN prefix reuse is limited to live/history/prefix
    ///   checkpoints whose identity matches the paged KV prefix. Misses
    ///   fall back to GDN replay from token 0.
    /// * Pure-cache prompt (every prompt token already in the paged
    ///   pool) is rejected — same caveat as LFM2 / Qwen3 paged paths.
    /// * The compiled C++ forward path is bypassed — paged turns run
    ///   the pure-Rust `DecoderLayer::forward_paged_or_flat`.
    fn chat_sync_core_paged(
        &mut self,
        tokens: Vec<u32>,
        tokenizer: Arc<Qwen3Tokenizer>,
        eos_token_id: u32,
        p: chat_common::ChatParams,
        report_perf: bool,
    ) -> Result<ChatResult> {
        if tokens.is_empty() {
            return Err(Error::from_reason("Empty prompt"));
        }

        let prompt_token_count = tokens.len() as u32;
        let sampling_config = p.sampling_config;

        let think_end_id = tokenizer.think_end_id();
        let think_end_str = tokenizer.think_end_str().map(|s| s.to_string());
        let thinking_enabled = true;
        let mut reasoning_tracker = chat_common::ReasoningTracker::new(
            thinking_enabled,
            p.thinking_token_budget,
            think_end_id,
        );

        let generation_start = if report_perf {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut first_token_instant: Option<std::time::Instant> = None;
        let trace_enabled = inference_trace_enabled();

        // Detect availability of the C++ compiled paged decode path.
        // Mirrors the MoE paged path: if the weights for this model are
        // still registered (no other model swapped `g_active_model_id`),
        // we can run `mlx_qwen35_init_paged` + `mlx_qwen35_forward_paged`.
        // Both the legacy flat compiled path and the new paged compiled
        // path share `COMPILED_LIFECYCLE_MUTEX`/`COMPILED_WEIGHTS_RWLOCK` for
        // process-wide serialization.
        let model_id = self.model_id;
        let use_cpp_paged = unsafe { mlx_sys::mlx_qwen35_get_model_id() } == model_id;
        let _dense_lock = if use_cpp_paged {
            Some(
                COMPILED_LIFECYCLE_MUTEX
                    .lock()
                    .unwrap_or_else(|e| e.into_inner()),
            )
        } else {
            None
        };
        let mut _weight_guard = None;
        let use_cpp_paged = if use_cpp_paged {
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

        // === Adapter lifecycle: warm continuation OR cold start ===
        let seq_id: u32 = 0;
        // Lazy decode allocation: pass the prompt length only.
        let total_budget = tokens.len() as u32;
        // Phase 6: per-block extra_keys for prefix-cache lookup. Text-only
        // dispatch (current state — image-bearing turns route to the flat
        // path) yields all-empty per-block vecs; the resulting hashes are
        // bit-equal to passing `&[]` to the uniform API. When VLM-paged
        // forward integration lands, this call site swaps in real image-
        // position pairs to enable image-aware cache isolation.
        let block_size = {
            let adapter = self
                .paged_adapter
                .as_ref()
                .ok_or_else(|| Error::from_reason("chat_sync_core_paged: paged_adapter is None"))?;
            adapter.block_size()
        };
        let lookup_extra_keys = chat_common::build_paged_extra_keys(tokens.len(), block_size, &[]);
        let cache_salt = 0;
        // vLLM exact-prefix cap — see qwen3/model.rs:chat_sync_core_paged.
        // Ensures every paged turn has at least one suffix token to prefill,
        // even when the live cache (or a prior request's residue) already
        // covers the entire new prompt.
        let max_cache_hit_tokens = total_budget.saturating_sub(1);
        let live_ready;
        let live_prefix_match;
        let live_tokens_len;
        let mut live_mismatch = TokenPrefixMismatchTrace::default();
        let (cached_prefix_len, continued_live_prefix) = {
            let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                Error::from_reason(
                    "chat_sync_core_paged: paged_adapter is None — caller must check \
                     use_block_paged_cache before dispatch",
                )
            })?;

            live_ready = adapter.is_live_for_continue();
            let live_tokens = adapter.request_tokens();
            live_tokens_len = live_tokens.len();
            live_prefix_match = tokens.starts_with(live_tokens);
            if trace_enabled && live_ready && !live_prefix_match {
                live_mismatch = token_prefix_mismatch_trace(&tokens, live_tokens);
            }
            let can_continue =
                live_ready && live_prefix_match && live_tokens_len <= max_cache_hit_tokens as usize;

            if can_continue {
                match adapter.continue_turn(&tokens, total_budget) {
                    Ok((prior_token_count, _newly_alloc)) => (prior_token_count, true),
                    Err(_drift) => {
                        let _ = adapter.release_request();
                        adapter
                            .reset_for_new_request(seq_id)
                            .map_err(Error::from_reason)?;
                        let prefix = adapter
                            .find_cached_prefix_per_block_with_max_tokens(
                                &tokens,
                                &lookup_extra_keys,
                                cache_salt,
                                false,
                                max_cache_hit_tokens,
                            )
                            .map_err(Error::from_reason)?;
                        let cached = prefix.cached_token_count;
                        adapter
                            .allocate_suffix_blocks(total_budget)
                            .map_err(Error::from_reason)?;
                        (cached, false)
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
                    .find_cached_prefix_per_block_with_max_tokens(
                        &tokens,
                        &lookup_extra_keys,
                        cache_salt,
                        false,
                        max_cache_hit_tokens,
                    )
                    .map_err(Error::from_reason)?;
                let cached = prefix.cached_token_count;
                adapter
                    .allocate_suffix_blocks(total_budget)
                    .map_err(Error::from_reason)?;
                (cached, false)
            }
        };
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] qwen3.5-dense paged_prefix_lookup prompt_tokens={} \
                 cached_prefix_tokens={} continued_live_prefix={} live_ready={} \
                 live_match={} live_tokens={} live_mismatch_at={} prompt_token={} live_token={}",
                tokens.len(),
                cached_prefix_len,
                continued_live_prefix,
                live_ready,
                live_prefix_match,
                live_tokens_len,
                live_mismatch.index,
                live_mismatch.prompt_token,
                live_mismatch.cached_token
            ));
        }

        let gdn_prefix_preparation = self.prepare_dense_gdn_prefix_state(
            &tokens,
            cached_prefix_len,
            block_size,
            &lookup_extra_keys,
            cache_salt,
            continued_live_prefix,
        )?;
        let gdn_prefix_already_primed = gdn_prefix_preparation.already_primed;
        self.cached_token_history.clear();
        self.cached_image_key = None;
        self.cached_rope_deltas = None;

        let suffix_len = prompt_token_count
            .checked_sub(cached_prefix_len)
            .ok_or_else(|| {
                Error::from_reason("chat_sync_core_paged: cached_prefix_len > total_prompt_tokens")
            })?;

        let forward_result = self.chat_sync_core_paged_inner(
            &tokens,
            cached_prefix_len,
            suffix_len,
            &p,
            eos_token_id,
            &sampling_config,
            &mut reasoning_tracker,
            report_perf,
            &mut first_token_instant,
            use_cpp_paged,
            gdn_prefix_already_primed,
        );

        let (generated_tokens, finish_reason) = match forward_result {
            Ok(t) => {
                if let Some(adapter) = self.paged_adapter.as_mut() {
                    // Build per-block extra_keys covering the FULL request
                    // (prompt + decoded tokens) for finalize. Text-only
                    // path produces all-empty vecs → bit-equal to the
                    // uniform `&[]` finalize.
                    let total_for_finalize = adapter.request_tokens().len();
                    let finalize_extra_keys =
                        chat_common::build_paged_extra_keys(total_for_finalize, block_size, &[]);
                    let _ = adapter.finalize_turn_keep_live_per_block(&finalize_extra_keys, 0);
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

        // Persist the full token history so subsequent
        // `chat_session_continue` /
        // `chat_tokens_delta_sync` calls find an initialized session
        // to extend. The paged decode loop never feeds the LAST
        // sampled token through the model, so drop it from the
        // saved history (mirrors LFM2 / Qwen3 paged path).
        let last_token_in_cache = false;
        let mut full_history = tokens.clone();
        if !generated_tokens.is_empty() {
            let upto = if last_token_in_cache {
                generated_tokens.len()
            } else {
                generated_tokens.len().saturating_sub(1)
            };
            full_history.extend_from_slice(&generated_tokens[..upto]);
        }
        self.cached_token_history = full_history;
        let gdn_history_checkpoint_store = self.remember_dense_gdn_history_checkpoint()?;
        if inference_trace_enabled() {
            write_inference_trace(format_args!(
                "[MLX_TRACE] qwen3.5-dense gdn_history_checkpoint stored={} tokens={} \
                 eval_ms={:.1} clone_ms={:.1} token_clone_ms={:.1} update_ms={:.1} total_ms={:.1}",
                gdn_history_checkpoint_store.stored,
                self.cached_token_history.len(),
                gdn_history_checkpoint_store.eval_ms,
                gdn_history_checkpoint_store.clone_ms,
                gdn_history_checkpoint_store.token_clone_ms,
                gdn_history_checkpoint_store.update_ms,
                gdn_history_checkpoint_store.total_ms
            ));
        }

        let performance = if report_perf {
            compute_performance_metrics(
                generation_start,
                first_token_instant,
                tokens.len() - cached_prefix_len as usize,
                generated_tokens.len(),
            )
        } else {
            None
        };

        let mut result = finalize_chat_result(
            &tokenizer,
            &generated_tokens,
            finish_reason,
            think_end_id,
            think_end_str.as_deref(),
            performance,
            p.include_reasoning,
            thinking_enabled,
            prompt_token_count,
            reasoning_tracker.reasoning_token_count(),
        )?;
        result.cached_tokens = cached_prefix_len;
        Ok(result)
    }

    /// Inner forward + decode loop for `chat_sync_core_paged`. Split
    /// out so the caller can wrap it with `release_request` on either
    /// path.
    ///
    /// Phase 5 piece 1 adds a C++ compiled paged decode dispatcher when
    /// `use_cpp_paged` is true. The pure-Rust paged prefill always runs
    /// (it populates the GDN linear caches and writes K/V into the
    /// adapter pool); after prefill, decode steps choose between
    /// `mlx_qwen35_forward_paged` (fast) and
    /// `paged_forward::run_paged_decode_step` (fallback) based on
    /// adapter block size and `init_paged_dense_compiled_session`
    /// success.
    #[allow(clippy::too_many_arguments)]
    fn chat_sync_core_paged_inner(
        &mut self,
        tokens: &[u32],
        cached_prefix_len: u32,
        suffix_len: u32,
        p: &chat_common::ChatParams,
        eos_token_id: u32,
        sampling_config: &Option<crate::sampling::SamplingConfig>,
        reasoning_tracker: &mut chat_common::ReasoningTracker,
        report_perf: bool,
        first_token_instant: &mut Option<std::time::Instant>,
        use_cpp_paged: bool,
        gdn_prefix_already_primed: bool,
    ) -> Result<(Vec<u32>, String)> {
        // Invariant: caller-applied vLLM cap guarantees suffix_len > 0.
        debug_assert!(
            suffix_len > 0,
            "chat_sync_core_paged_inner: caller must cap max_cache_hit_tokens at prompt.len() - 1"
        );

        let suffix = &tokens[(cached_prefix_len as usize)..];
        let layer_kinds =
            super::decoder_layer::compute_layer_kinds(self.config.num_layers as usize, |i| {
                self.config.is_linear_layer(i)
            });

        // === PREFILL ===
        let last_logits = {
            let embed = self.embedding.clone();
            let embedding_weight = embed.get_weight();
            let caches_ref = self.caches.as_mut().ok_or_else(|| {
                Error::from_reason("chat_sync_core_paged_inner: caches not initialized")
            })?;
            let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                Error::from_reason("chat_sync_core_paged_inner: paged_adapter dropped")
            })?;
            super::paged_forward::run_paged_prefill_chunk(
                tokens,
                suffix,
                cached_prefix_len,
                gdn_prefix_already_primed,
                &embed,
                &mut self.layers,
                caches_ref,
                &self.final_norm,
                &self.lm_head,
                &embedding_weight,
                &layer_kinds,
                adapter,
            )?
        };

        // First-token sample.
        let mut token_history: Vec<u32> = tokens.to_vec();
        let last_logits = apply_all_penalties(last_logits, &token_history, p)?;
        let mut y = sample(&last_logits, *sampling_config)?;
        y.eval();

        // Smooth memory peak: drop transient prefill buffers before decode
        // starts allocating. Prefill builds a massive MLX subgraph; once
        // we have the last logits, those intermediates are dead but
        // MLX's caching allocator holds them.
        crate::array::synchronize_and_clear_cache();

        if report_perf {
            *first_token_instant = Some(std::time::Instant::now());
        }

        // Decide between C++ compiled paged decode (fast) and pure-Rust
        // paged decode (fallback). C++ paged needs:
        // 1. `use_cpp_paged` (weights still registered for our model_id —
        //    re-validated under `COMPILED_WEIGHTS_RWLOCK` in the caller).
        // 2. `adapter.block_size() == CPP_PAGED_REQUIRED_BLOCK_SIZE`.
        //    The compiled C++ paged graph hard-codes block_size = 16.
        // 3. `init_paged_dense_compiled_session` to succeed (every linear
        //    layer must have populated conv/recurrent state from the
        //    pure-Rust GDN forward above; every full-attn layer must
        //    have a usable `LayerKVPool` slot, and the C++
        //    `g_dense_paged_inited` must be set after init catches no
        //    exceptions).
        // The `CompiledResetGuard` ensures `mlx_qwen35_compiled_reset()`
        // runs on any exit path so the next session starts with cleared
        // C++ globals.
        let mut cpp_session_ready = if use_cpp_paged {
            let caches_ref = self.caches.as_ref().ok_or_else(|| {
                Error::from_reason("chat_sync_core_paged_inner: caches dropped post-prefill")
            })?;
            let adapter_ref = self.paged_adapter.as_ref().ok_or_else(|| {
                Error::from_reason("chat_sync_core_paged_inner: paged_adapter dropped post-prefill")
            })?;
            if adapter_ref.block_size() != CPP_PAGED_REQUIRED_BLOCK_SIZE {
                eprintln!(
                    "[MLX] Qwen3.5 Dense: skipping C++ compiled paged decode — \
                     adapter block_size={} but compiled graph requires {}; \
                     falling back to pure-Rust paged decode",
                    adapter_ref.block_size(),
                    CPP_PAGED_REQUIRED_BLOCK_SIZE
                );
                false
            } else {
                let prefill_offset = adapter_ref.current_token_count() as i32;
                init_paged_dense_compiled_session(
                    &self.config,
                    caches_ref,
                    adapter_ref,
                    prefill_offset,
                )
                .is_ok()
            }
        } else {
            false
        };

        // RAII guard: resets BOTH g_compiled_* and g_dense_paged_*
        // globals (single `mlx_qwen35_compiled_reset` symbol clears
        // both per Phase 5 piece 1) when this scope ends. Always armed
        // when init succeeded — even if a later forward fails and we
        // flip `cpp_session_ready=false`, the guard still runs at scope
        // exit so stale C++ globals don't leak into the next request.
        let _dense_paged_guard = cpp_session_ready.then_some(CompiledResetGuard);

        // Tracks whether ANY compiled C++ paged step has succeeded
        // during this turn. After a successful compiled step the C++
        // side has advanced its per-layer GDN linear-cache globals
        // (conv_state / recurrent_state) but those updates never get
        // imported back into `self.caches`. Falling back to pure-Rust
        // decode after that point would run from stale pre-step-N GDN
        // state while `paged_adapter` and `token_history` have already
        // advanced — silently corrupting the rest of the request. The
        // mid-turn fallback below is therefore only safe BEFORE the
        // first successful compiled step (the only failure mode we
        // really need to defend against is init/configuration mismatch
        // on the very first forward call).
        let mut cpp_compiled_step_completed = false;

        // === DECODE LOOP ===
        let max_new_tokens = p.max_new_tokens;
        let mut generated_tokens: Vec<u32> = Vec::with_capacity(max_new_tokens.max(0) as usize);
        let mut finish_reason = String::from("length");

        // Compile-cached `max_blocks_per_seq` shape — picking the
        // adapter's max_seq_len divided by block_size keeps the compile
        // key stable across all decode steps within one turn.
        let max_blocks_per_seq: u32 = {
            let adapter = self.paged_adapter.as_ref().ok_or_else(|| {
                Error::from_reason("chat_sync_core_paged_inner: paged_adapter dropped pre-decode")
            })?;
            let max_seq = self.config.max_position_embeddings as u32;
            max_seq.div_ceil(adapter.block_size())
        };

        for step in 0..max_new_tokens {
            let token_id = y.item_at_int32(0)? as u32;
            generated_tokens.push(token_id);
            token_history.push(token_id);
            reasoning_tracker.observe_token(token_id);

            if token_id == eos_token_id {
                finish_reason = String::from("stop");
                break;
            }
            if let Some(reason) = crate::sampling::check_repetition_cutoff(
                &generated_tokens,
                p.max_consecutive_tokens,
                p.max_ngram_repeats,
                p.ngram_size,
            ) {
                finish_reason = reason.to_string();
                break;
            }
            if step + 1 >= max_new_tokens {
                break;
            }

            // Decode forward.
            //
            // Defense-in-depth: if the C++ compiled paged forward returns
            // null on the FIRST step, we rollback the `record_tokens`
            // cursor advance, mark `cpp_session_ready = false`, and
            // re-run this token through pure-Rust `run_paged_decode_step`
            // (which re-calls `record_tokens` on the now-rolled-back
            // cursor). After ANY compiled step has succeeded the C++ GDN
            // linear-cache globals have advanced but we never copy them
            // back into `self.caches`, so a Rust fallback would read
            // stale pre-step state — propagate the error as fatal
            // instead of silently corrupting the response.
            let next_logits = if cpp_session_ready {
                let embedding_weight = self.embedding.get_weight();
                let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                    Error::from_reason(
                        "chat_sync_core_paged_inner: paged_adapter dropped mid-decode (cpp)",
                    )
                })?;
                adapter
                    .record_tokens(&[token_id])
                    .map_err(Error::from_reason)?;
                let inputs = adapter
                    .build_paged_attention_inputs(1, 1, max_blocks_per_seq)
                    .map_err(Error::from_reason)?;
                let input_ids = MxArray::from_uint32(&[token_id], &[1, 1])?;
                match forward_dense_cpp_paged(&input_ids, &embedding_weight, &inputs) {
                    Ok(logits) => {
                        cpp_compiled_step_completed = true;
                        logits
                    }
                    Err(e) => {
                        if chat_common::should_propagate_compiled_paged_error(
                            cpp_compiled_step_completed,
                        ) {
                            eprintln!(
                                "[MLX] Qwen3.5 Dense: C++ compiled paged forward failed \
                                 mid-decode (step={step}) AFTER an earlier compiled step \
                                 succeeded. The C++ GDN linear-cache globals have advanced \
                                 but those updates are not imported back into self.caches, \
                                 so a pure-Rust fallback would run from stale pre-step state \
                                 and silently corrupt the response. Propagating as fatal. \
                                 cause: {e}"
                            );
                            adapter
                                .rollback_last_tokens(1)
                                .map_err(Error::from_reason)?;
                            return Err(e);
                        }
                        eprintln!(
                            "[MLX] Qwen3.5 Dense: C++ compiled paged forward failed on \
                             first decode step (step={step}); rolling back token cursor \
                             and falling back to pure-Rust paged decode for the rest of \
                             this request. cause: {e}"
                        );
                        adapter
                            .rollback_last_tokens(1)
                            .map_err(Error::from_reason)?;
                        cpp_session_ready = false;
                        let embed = self.embedding.clone();
                        let embedding_weight_pure = embed.get_weight();
                        let caches_ref = self.caches.as_mut().ok_or_else(|| {
                            Error::from_reason(
                                "chat_sync_core_paged_inner: caches dropped during cpp fallback",
                            )
                        })?;
                        let adapter_mut = self.paged_adapter.as_mut().ok_or_else(|| {
                            Error::from_reason(
                                "chat_sync_core_paged_inner: paged_adapter dropped during cpp fallback",
                            )
                        })?;
                        let logits = super::paged_forward::run_paged_decode_step(
                            token_id,
                            &embed,
                            &mut self.layers,
                            caches_ref,
                            &self.final_norm,
                            &self.lm_head,
                            &embedding_weight_pure,
                            &layer_kinds,
                            adapter_mut,
                        )?;
                        logits.squeeze(Some(&[1]))?
                    }
                }
            } else {
                // Pure-Rust paged decode fallback.
                let embed = self.embedding.clone();
                let embedding_weight = embed.get_weight();
                let caches_ref = self.caches.as_mut().ok_or_else(|| {
                    Error::from_reason("chat_sync_core_paged_inner: caches dropped mid-decode")
                })?;
                let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                    Error::from_reason(
                        "chat_sync_core_paged_inner: paged_adapter dropped mid-decode",
                    )
                })?;
                let logits = super::paged_forward::run_paged_decode_step(
                    token_id,
                    &embed,
                    &mut self.layers,
                    caches_ref,
                    &self.final_norm,
                    &self.lm_head,
                    &embedding_weight,
                    &layer_kinds,
                    adapter,
                )?;
                logits.squeeze(Some(&[1]))?
            };

            let next_logits = if reasoning_tracker.should_force_think_end() {
                let forced_id = reasoning_tracker.forced_token_id() as i32;
                y = MxArray::from_int32(&[forced_id], &[1])?;
                y.eval();
                continue;
            } else {
                apply_all_penalties(next_logits, &token_history, p)?
            };

            y = sample(&next_logits, *sampling_config)?;
            y.eval();

            crate::array::maybe_clear_cache_for_paged_step(step);
        }

        if cpp_compiled_step_completed {
            match export_paged_dense_linear_caches(&self.config) {
                Ok(Some(new_caches)) => {
                    self.caches = Some(new_caches);
                    eval_layer_caches(&self.caches)?;
                    write_inference_trace(format_args!(
                        "[MLX_TRACE] qwen3.5-dense paged_linear_cache_export ok=true"
                    ));
                }
                Ok(None) => {
                    self.caches = None;
                    write_inference_trace(format_args!(
                        "[MLX_TRACE] qwen3.5-dense paged_linear_cache_export ok=false reason=not_initialized"
                    ));
                }
                Err(err) => {
                    write_inference_trace(format_args!(
                        "[MLX_TRACE] qwen3.5-dense paged_linear_cache_export ok=false error={}",
                        err
                    ));
                    self.caches = None;
                }
            }
        }

        Ok((generated_tokens, finish_reason))
    }

    /// Block-paged streaming variant of [`Self::chat_stream_sync_inner`].
    ///
    /// Mirrors `chat_sync_core_paged`'s adapter lifecycle and
    /// per-layer dispatch but emits each generated token through the
    /// streaming callback as it is produced.
    #[allow(clippy::too_many_arguments)]
    fn chat_stream_sync_core_paged(
        &mut self,
        tokens: Vec<u32>,
        tokenizer: Arc<Qwen3Tokenizer>,
        eos_token_id: u32,
        p: chat_common::ChatParams,
        report_perf: bool,
        cb: &StreamSender,
        cancelled: &Arc<AtomicBool>,
    ) -> Result<()> {
        if tokens.is_empty() {
            return Err(Error::from_reason("Empty prompt"));
        }

        let prompt_token_count = tokens.len() as u32;
        let sampling_config = p.sampling_config;
        let include_reasoning = p.include_reasoning;

        let think_end_id = tokenizer.think_end_id();
        let think_end_str = tokenizer.think_end_str().map(|s| s.to_string());
        let thinking_enabled = true;
        let mut reasoning_tracker = chat_common::ReasoningTracker::new(
            thinking_enabled,
            p.thinking_token_budget,
            think_end_id,
        );

        let generation_start = if report_perf {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut first_token_instant: Option<std::time::Instant> = None;
        let trace_enabled = inference_trace_enabled();

        // Streaming decode state.
        let mut decode_stream = tokenizer.inner().decode_stream(true);
        let mut streamed_text_len = 0usize;
        let mut last_is_reasoning = thinking_enabled;

        // C++ paged-decode availability + compile-lifecycle locks. See
        // `chat_sync_core_paged` for the full rationale; this is the
        // streaming twin.
        let model_id = self.model_id;
        let use_cpp_paged = unsafe { mlx_sys::mlx_qwen35_get_model_id() } == model_id;
        let _dense_lock = if use_cpp_paged {
            Some(
                COMPILED_LIFECYCLE_MUTEX
                    .lock()
                    .unwrap_or_else(|e| e.into_inner()),
            )
        } else {
            None
        };
        let mut _weight_guard = None;
        let use_cpp_paged = if use_cpp_paged {
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

        // === Adapter lifecycle: warm continue OR cold start ===
        let seq_id: u32 = 0;
        // Lazy decode allocation: pass the prompt length only.
        let total_budget = tokens.len() as u32;
        // Phase 6: per-block extra_keys for prefix-cache lookup. See the
        // matching comment in `chat_sync_core_paged`.
        let block_size = {
            let adapter = self.paged_adapter.as_ref().ok_or_else(|| {
                Error::from_reason("chat_stream_sync_core_paged: paged_adapter is None")
            })?;
            adapter.block_size()
        };
        let lookup_extra_keys = chat_common::build_paged_extra_keys(tokens.len(), block_size, &[]);
        let cache_salt = 0;
        // See `chat_sync_core_paged` for the vLLM exact-prefix cap rationale.
        let max_cache_hit_tokens = total_budget.saturating_sub(1);
        let live_ready;
        let live_prefix_match;
        let live_tokens_len;
        let mut live_mismatch = TokenPrefixMismatchTrace::default();
        let (cached_prefix_len, continued_live_prefix) = {
            let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                Error::from_reason(
                    "chat_stream_sync_core_paged: paged_adapter is None — caller must check \
                     use_block_paged_cache before dispatch",
                )
            })?;

            live_ready = adapter.is_live_for_continue();
            let live_tokens = adapter.request_tokens();
            live_tokens_len = live_tokens.len();
            live_prefix_match = tokens.starts_with(live_tokens);
            if trace_enabled && live_ready && !live_prefix_match {
                live_mismatch = token_prefix_mismatch_trace(&tokens, live_tokens);
            }
            let can_continue =
                live_ready && live_prefix_match && live_tokens_len <= max_cache_hit_tokens as usize;

            if can_continue {
                match adapter.continue_turn(&tokens, total_budget) {
                    Ok((prior_token_count, _newly_alloc)) => (prior_token_count, true),
                    Err(_drift) => {
                        let _ = adapter.release_request();
                        adapter
                            .reset_for_new_request(seq_id)
                            .map_err(Error::from_reason)?;
                        let prefix = adapter
                            .find_cached_prefix_per_block_with_max_tokens(
                                &tokens,
                                &lookup_extra_keys,
                                cache_salt,
                                false,
                                max_cache_hit_tokens,
                            )
                            .map_err(Error::from_reason)?;
                        let cached = prefix.cached_token_count;
                        adapter
                            .allocate_suffix_blocks(total_budget)
                            .map_err(Error::from_reason)?;
                        (cached, false)
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
                    .find_cached_prefix_per_block_with_max_tokens(
                        &tokens,
                        &lookup_extra_keys,
                        cache_salt,
                        false,
                        max_cache_hit_tokens,
                    )
                    .map_err(Error::from_reason)?;
                let cached = prefix.cached_token_count;
                adapter
                    .allocate_suffix_blocks(total_budget)
                    .map_err(Error::from_reason)?;
                (cached, false)
            }
        };
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] qwen3.5-dense paged_prefix_lookup prompt_tokens={} \
                 cached_prefix_tokens={} continued_live_prefix={} live_ready={} \
                 live_match={} live_tokens={} live_mismatch_at={} prompt_token={} live_token={}",
                tokens.len(),
                cached_prefix_len,
                continued_live_prefix,
                live_ready,
                live_prefix_match,
                live_tokens_len,
                live_mismatch.index,
                live_mismatch.prompt_token,
                live_mismatch.cached_token
            ));
        }

        let gdn_prefix_preparation = self.prepare_dense_gdn_prefix_state(
            &tokens,
            cached_prefix_len,
            block_size,
            &lookup_extra_keys,
            cache_salt,
            continued_live_prefix,
        )?;
        let gdn_prefix_already_primed = gdn_prefix_preparation.already_primed;
        let gdn_prefix_state = gdn_prefix_preparation.state;
        self.cached_token_history.clear();
        self.cached_image_key = None;
        self.cached_rope_deltas = None;

        let suffix_len = prompt_token_count
            .checked_sub(cached_prefix_len)
            .ok_or_else(|| {
                Error::from_reason(
                    "chat_stream_sync_core_paged: cached_prefix_len > total_prompt_tokens",
                )
            })?;

        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] qwen3.5-dense stream_paged_start prompt_tokens={} \
                 cached_prefix_tokens={} suffix_tokens={} block_size={} \
                 prefill_chunk_size={} prefill_eval_interval={} decode_clear_interval={} \
                 cpp_paged_candidate={} gdn_prefix_state={}",
                prompt_token_count,
                cached_prefix_len,
                suffix_len,
                block_size,
                crate::array::paged_prefill_chunk_size(),
                crate::array::paged_prefill_eval_interval(),
                crate::array::paged_decode_cache_clear_interval(),
                use_cpp_paged,
                gdn_prefix_state
            ));
        }

        let result = self.chat_stream_sync_core_paged_inner(
            &tokens,
            cached_prefix_len,
            suffix_len,
            &p,
            sampling_config,
            eos_token_id,
            &mut reasoning_tracker,
            report_perf,
            &mut first_token_instant,
            &tokenizer,
            &mut decode_stream,
            &mut streamed_text_len,
            &mut last_is_reasoning,
            cb,
            cancelled,
            use_cpp_paged,
            gdn_prefix_already_primed,
        );

        let (generated_tokens, finish_reason) = match result {
            Ok(t) => {
                if let Some(adapter) = self.paged_adapter.as_mut() {
                    let total_for_finalize = adapter.request_tokens().len();
                    let finalize_extra_keys =
                        chat_common::build_paged_extra_keys(total_for_finalize, block_size, &[]);
                    let _ = adapter.finalize_turn_keep_live_per_block(&finalize_extra_keys, 0);
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

        // Persist token history for subsequent session-continue calls.
        let last_token_in_cache = false;
        let mut full_history = tokens.clone();
        if !generated_tokens.is_empty() {
            let upto = if last_token_in_cache {
                generated_tokens.len()
            } else {
                generated_tokens.len().saturating_sub(1)
            };
            full_history.extend_from_slice(&generated_tokens[..upto]);
        }
        self.cached_token_history = full_history;
        let gdn_history_checkpoint_store = self.remember_dense_gdn_history_checkpoint()?;
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] qwen3.5-dense gdn_history_checkpoint stored={} tokens={} \
                 eval_ms={:.1} clone_ms={:.1} token_clone_ms={:.1} update_ms={:.1} total_ms={:.1}",
                gdn_history_checkpoint_store.stored,
                self.cached_token_history.len(),
                gdn_history_checkpoint_store.eval_ms,
                gdn_history_checkpoint_store.clone_ms,
                gdn_history_checkpoint_store.token_clone_ms,
                gdn_history_checkpoint_store.update_ms,
                gdn_history_checkpoint_store.total_ms
            ));
        }

        // Flush residual buffered bytes (mirrors flat streaming).
        let full_text = tokenizer
            .decode_sync(&generated_tokens, true)
            .unwrap_or_else(|e| {
                tracing::warn!("Failed to decode generated tokens: {}", e);
                String::new()
            });
        if full_text.len() > streamed_text_len {
            let residual = full_text[streamed_text_len..].to_string();
            // Suppress residual when it is reasoning text and
            // include_reasoning == false.
            if include_reasoning || !last_is_reasoning {
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
                        cached_tokens: None,
                        performance: None,
                        is_reasoning: Some(last_is_reasoning),
                    }),
                    ThreadsafeFunctionCallMode::NonBlocking,
                );
            }
        }

        let performance = if report_perf {
            compute_performance_metrics(
                generation_start,
                first_token_instant,
                tokens.len() - cached_prefix_len as usize,
                generated_tokens.len(),
            )
        } else {
            None
        };

        let reasoning_tokens = reasoning_tracker.reasoning_token_count();

        let mut result = finalize_chat_result(
            &tokenizer,
            &generated_tokens,
            finish_reason,
            think_end_id,
            think_end_str.as_deref(),
            performance,
            include_reasoning,
            thinking_enabled,
            prompt_token_count,
            reasoning_tokens,
        )?;
        result.cached_tokens = cached_prefix_len;

        // Terminal chunk.
        cb.call(
            Ok(ChatStreamChunk {
                text: result.text.clone(),
                done: true,
                finish_reason: Some(result.finish_reason.clone()),
                tool_calls: Some(result.tool_calls.clone()),
                thinking: result.thinking.clone(),
                num_tokens: Some(result.num_tokens),
                prompt_tokens: Some(result.prompt_tokens),
                reasoning_tokens: Some(result.reasoning_tokens),
                raw_text: Some(result.raw_text.clone()),
                cached_tokens: Some(cached_prefix_len),
                performance: result.performance.clone(),
                is_reasoning: None,
            }),
            ThreadsafeFunctionCallMode::NonBlocking,
        );

        Ok(())
    }

    /// Inner forward + streaming decode loop for
    /// [`Self::chat_stream_sync_core_paged`]. Mirrors LFM2's
    /// `chat_stream_sync_core_paged_inner`.
    ///
    /// Phase 5 piece 1 adds a C++ compiled paged decode dispatcher when
    /// `use_cpp_paged` is true. See the sync sibling
    /// `chat_sync_core_paged_inner` for the cpp_session_ready gate
    /// rationale.
    #[allow(clippy::too_many_arguments)]
    fn chat_stream_sync_core_paged_inner<'a>(
        &mut self,
        tokens: &[u32],
        cached_prefix_len: u32,
        suffix_len: u32,
        p: &chat_common::ChatParams,
        sampling_config: Option<crate::sampling::SamplingConfig>,
        eos_token_id: u32,
        reasoning_tracker: &mut chat_common::ReasoningTracker,
        report_perf: bool,
        first_token_instant: &mut Option<std::time::Instant>,
        tokenizer: &'a Arc<Qwen3Tokenizer>,
        decode_stream: &mut tokenizers::DecodeStream<
            'a,
            tokenizers::ModelWrapper,
            tokenizers::NormalizerWrapper,
            tokenizers::PreTokenizerWrapper,
            tokenizers::PostProcessorWrapper,
            tokenizers::DecoderWrapper,
        >,
        streamed_text_len: &mut usize,
        last_is_reasoning: &mut bool,
        cb: &StreamSender,
        cancelled: &Arc<AtomicBool>,
        use_cpp_paged: bool,
        gdn_prefix_already_primed: bool,
    ) -> Result<(Vec<u32>, String)> {
        // Invariant: caller-applied vLLM cap guarantees suffix_len > 0.
        debug_assert!(
            suffix_len > 0,
            "chat_stream_sync_core_paged_inner: caller must cap max_cache_hit_tokens at prompt.len() - 1"
        );

        let suffix = &tokens[(cached_prefix_len as usize)..];
        let layer_kinds =
            super::decoder_layer::compute_layer_kinds(self.config.num_layers as usize, |i| {
                self.config.is_linear_layer(i)
            });

        let last_logits = {
            let embed = self.embedding.clone();
            let embedding_weight = embed.get_weight();
            let caches_ref = self.caches.as_mut().ok_or_else(|| {
                Error::from_reason("chat_stream_sync_core_paged_inner: caches not initialized")
            })?;
            let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                Error::from_reason("chat_stream_sync_core_paged_inner: paged_adapter dropped")
            })?;
            super::paged_forward::run_paged_prefill_chunk(
                tokens,
                suffix,
                cached_prefix_len,
                gdn_prefix_already_primed,
                &embed,
                &mut self.layers,
                caches_ref,
                &self.final_norm,
                &self.lm_head,
                &embedding_weight,
                &layer_kinds,
                adapter,
            )?
        };

        let mut token_history: Vec<u32> = tokens.to_vec();
        let last_logits = apply_all_penalties(last_logits, &token_history, p)?;
        let mut y = sample(&last_logits, sampling_config)?;
        y.eval();

        // Smooth memory peak: drop transient prefill buffers before decode
        // starts allocating (see chat_sync_core_paged_inner for rationale).
        crate::array::synchronize_and_clear_cache();

        if report_perf {
            *first_token_instant = Some(std::time::Instant::now());
        }

        // Decide between C++ compiled paged decode (fast) and pure-Rust
        // paged decode (fallback). See `chat_sync_core_paged_inner` for
        // the cpp_session_ready gate rationale.
        let mut cpp_session_ready = if use_cpp_paged {
            let caches_ref = self.caches.as_ref().ok_or_else(|| {
                Error::from_reason("chat_stream_sync_core_paged_inner: caches dropped post-prefill")
            })?;
            let adapter_ref = self.paged_adapter.as_ref().ok_or_else(|| {
                Error::from_reason(
                    "chat_stream_sync_core_paged_inner: paged_adapter dropped post-prefill",
                )
            })?;
            if adapter_ref.block_size() != CPP_PAGED_REQUIRED_BLOCK_SIZE {
                eprintln!(
                    "[MLX] Qwen3.5 Dense (stream): skipping C++ compiled paged decode — \
                     adapter block_size={} but compiled graph requires {}; \
                     falling back to pure-Rust paged decode",
                    adapter_ref.block_size(),
                    CPP_PAGED_REQUIRED_BLOCK_SIZE
                );
                false
            } else {
                let prefill_offset = adapter_ref.current_token_count() as i32;
                init_paged_dense_compiled_session(
                    &self.config,
                    caches_ref,
                    adapter_ref,
                    prefill_offset,
                )
                .is_ok()
            }
        } else {
            false
        };

        // RAII guard: even if a later forward fails and we flip
        // `cpp_session_ready=false`, the guard still runs at scope exit
        // so stale C++ globals don't leak into the next request.
        let _dense_paged_guard = cpp_session_ready.then_some(CompiledResetGuard);

        // Tracks whether ANY compiled C++ paged step has succeeded
        // during this turn. After a successful compiled step the C++
        // GDN linear-cache globals (conv_state / recurrent_state) have
        // advanced but are never imported back into `self.caches`, so a
        // pure-Rust fallback would read stale pre-step state. The
        // mid-turn fallback below is therefore only safe BEFORE the
        // first successful compiled step. See sync sibling
        // `chat_sync_core_paged_inner` for the full rationale.
        let mut cpp_compiled_step_completed = false;

        let max_new_tokens = p.max_new_tokens;
        let mut generated_tokens: Vec<u32> = Vec::with_capacity(max_new_tokens.max(0) as usize);
        let mut finish_reason = String::from("length");

        let max_blocks_per_seq: u32 = {
            let adapter = self.paged_adapter.as_ref().ok_or_else(|| {
                Error::from_reason(
                    "chat_stream_sync_core_paged_inner: paged_adapter dropped pre-decode",
                )
            })?;
            let max_seq = self.config.max_position_embeddings as u32;
            max_seq.div_ceil(adapter.block_size())
        };

        for step in 0..max_new_tokens {
            let token_id = y.item_at_int32(0)? as u32;
            generated_tokens.push(token_id);
            token_history.push(token_id);
            let is_reasoning = reasoning_tracker.observe_token(token_id);
            *last_is_reasoning = is_reasoning;

            if token_id == eos_token_id {
                finish_reason = String::from("stop");
                break;
            }
            if cancelled.load(Ordering::Relaxed) {
                finish_reason = String::from("cancelled");
                break;
            }

            // Stream delta chunk.
            let token_text = Qwen3Tokenizer::step_decode_stream(
                decode_stream,
                tokenizer.inner(),
                token_id,
                &generated_tokens,
                *streamed_text_len,
            );
            *streamed_text_len += token_text.len();
            // Suppress reasoning deltas when include_reasoning == false.
            // Detokenize + length-advance above stay OUTSIDE this gate.
            if p.include_reasoning || !is_reasoning {
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
                        cached_tokens: None,
                        performance: None,
                        is_reasoning: Some(is_reasoning),
                    }),
                    ThreadsafeFunctionCallMode::NonBlocking,
                );
            }

            if let Some(reason) = crate::sampling::check_repetition_cutoff(
                &generated_tokens,
                p.max_consecutive_tokens,
                p.max_ngram_repeats,
                p.ngram_size,
            ) {
                finish_reason = reason.to_string();
                break;
            }
            if step + 1 >= max_new_tokens {
                break;
            }

            // Decode forward. Defense-in-depth fallback: see
            // `chat_sync_core_paged_inner` for the rollback rationale —
            // mid-turn fallback only safe BEFORE the first successful
            // compiled step.
            let next_logits = if cpp_session_ready {
                let embedding_weight = self.embedding.get_weight();
                let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                    Error::from_reason(
                        "chat_stream_sync_core_paged_inner: paged_adapter dropped mid-decode (cpp)",
                    )
                })?;
                adapter
                    .record_tokens(&[token_id])
                    .map_err(Error::from_reason)?;
                let inputs = adapter
                    .build_paged_attention_inputs(1, 1, max_blocks_per_seq)
                    .map_err(Error::from_reason)?;
                let input_ids = MxArray::from_uint32(&[token_id], &[1, 1])?;
                match forward_dense_cpp_paged(&input_ids, &embedding_weight, &inputs) {
                    Ok(logits) => {
                        cpp_compiled_step_completed = true;
                        logits
                    }
                    Err(e) => {
                        if chat_common::should_propagate_compiled_paged_error(
                            cpp_compiled_step_completed,
                        ) {
                            eprintln!(
                                "[MLX] Qwen3.5 Dense (stream): C++ compiled paged forward \
                                 failed mid-decode (step={step}) AFTER an earlier compiled \
                                 step succeeded. The C++ GDN linear-cache globals have \
                                 advanced but those updates are not imported back into \
                                 self.caches, so a pure-Rust fallback would run from stale \
                                 pre-step state and silently corrupt the response. \
                                 Propagating as fatal. cause: {e}"
                            );
                            adapter
                                .rollback_last_tokens(1)
                                .map_err(Error::from_reason)?;
                            return Err(e);
                        }
                        eprintln!(
                            "[MLX] Qwen3.5 Dense (stream): C++ compiled paged forward failed \
                             on first decode step (step={step}); rolling back token cursor \
                             and falling back to pure-Rust paged decode for the rest of \
                             this request. cause: {e}"
                        );
                        adapter
                            .rollback_last_tokens(1)
                            .map_err(Error::from_reason)?;
                        cpp_session_ready = false;
                        let embed = self.embedding.clone();
                        let embedding_weight_pure = embed.get_weight();
                        let caches_ref = self.caches.as_mut().ok_or_else(|| {
                            Error::from_reason(
                                "chat_stream_sync_core_paged_inner: caches dropped during cpp fallback",
                            )
                        })?;
                        let adapter_mut = self.paged_adapter.as_mut().ok_or_else(|| {
                            Error::from_reason(
                                "chat_stream_sync_core_paged_inner: paged_adapter dropped during cpp fallback",
                            )
                        })?;
                        let logits = super::paged_forward::run_paged_decode_step(
                            token_id,
                            &embed,
                            &mut self.layers,
                            caches_ref,
                            &self.final_norm,
                            &self.lm_head,
                            &embedding_weight_pure,
                            &layer_kinds,
                            adapter_mut,
                        )?;
                        logits.squeeze(Some(&[1]))?
                    }
                }
            } else {
                let embed = self.embedding.clone();
                let embedding_weight = embed.get_weight();
                let caches_ref = self.caches.as_mut().ok_or_else(|| {
                    Error::from_reason(
                        "chat_stream_sync_core_paged_inner: caches dropped mid-decode",
                    )
                })?;
                let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                    Error::from_reason(
                        "chat_stream_sync_core_paged_inner: paged_adapter dropped mid-decode",
                    )
                })?;
                let logits = super::paged_forward::run_paged_decode_step(
                    token_id,
                    &embed,
                    &mut self.layers,
                    caches_ref,
                    &self.final_norm,
                    &self.lm_head,
                    &embedding_weight,
                    &layer_kinds,
                    adapter,
                )?;
                logits.squeeze(Some(&[1]))?
            };

            let next_logits = if reasoning_tracker.should_force_think_end() {
                let forced_id = reasoning_tracker.forced_token_id() as i32;
                y = MxArray::from_int32(&[forced_id], &[1])?;
                y.eval();
                continue;
            } else {
                apply_all_penalties(next_logits, &token_history, p)?
            };

            y = sample(&next_logits, sampling_config)?;
            y.eval();

            crate::array::maybe_clear_cache_for_paged_step(step);
        }

        if cpp_compiled_step_completed {
            match export_paged_dense_linear_caches(&self.config) {
                Ok(Some(new_caches)) => {
                    self.caches = Some(new_caches);
                    eval_layer_caches(&self.caches)?;
                    write_inference_trace(format_args!(
                        "[MLX_TRACE] qwen3.5-dense paged_linear_cache_export ok=true"
                    ));
                }
                Ok(None) => {
                    self.caches = None;
                    write_inference_trace(format_args!(
                        "[MLX_TRACE] qwen3.5-dense paged_linear_cache_export ok=false reason=not_initialized"
                    ));
                }
                Err(err) => {
                    write_inference_trace(format_args!(
                        "[MLX_TRACE] qwen3.5-dense paged_linear_cache_export ok=false error={}",
                        err
                    ));
                    self.caches = None;
                }
            }
        }

        Ok((generated_tokens, finish_reason))
    }

    /// Streaming chat (session-start variant): same semantics as
    /// [`Self::chat_session_start_sync`] but streams token deltas through
    /// `stream_tx` rather than returning a `ChatResult`. Stops on
    /// `<|im_end|>`. Inherits the prefix-cache reuse contract — the
    /// caches may carry state from a prior turn; `chat_stream_sync_inner`
    /// verifies the prefix and resets on miss.
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

        // Prefix-cache reuse contract: same as `chat_session_start_sync`.
        // Any cached state from a prior turn is intentionally preserved so
        // `chat_stream_sync_inner` can run `verify_cache_prefix_direct`
        // against the freshly-tokenized prompt. The inner path resets the
        // caches on a miss and reuses them on an exact-append hit. See
        // the rustdoc on `verify_cache_prefix_direct` for the GDN-safety
        // invariant that keeps this sound.

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
                // than an error chunk, matching the `chat_stream_sync_inner`
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
    /// text-only-delta invariant. `is_error` is forwarded verbatim to
    /// the wire-format renderer; see the non-streaming entry point for
    /// the marker semantics.
    pub(crate) fn chat_stream_session_continue_tool_sync(
        &mut self,
        tool_call_id: String,
        content: String,
        is_error: Option<bool>,
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
        let delta_text = chat_common::build_chatml_tool_delta_text(
            &tool_call_id,
            &content,
            enable_thinking,
            is_error,
        );

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
        // Text-only deltas are allowed on sessions whose KV cache carries
        // prior image attention state — see `chat_tokens_delta_sync` for
        // the full rationale. The outer `chat_stream_session_continue`
        // gate already filters IMAGE-SET CHANGES via the
        // `IMAGE_CHANGE_REQUIRES_SESSION_RESTART:` prefix path.

        // All guards passed — enter the prefill+decode helper. Any error
        // returned from here propagates as an mpsc error, same as
        // `chat_stream_sync_inner`.
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
        // Capture `prior_cached_len` BEFORE the extend — this is the
        // reused-prefix length reported on the terminal ChatStreamChunk's
        // `cached_tokens` field (mirrors the non-streaming delta path's
        // `cached_tokens_for_result`).
        let prior_cached_len = self.cached_token_history.len() as u32;
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

        // Block-paged dispatch — early-return BEFORE the compile lock.
        // Delta path: drive the paged streaming core with
        // `cached_history + delta` as the full prompt; the adapter's
        // warm-continue path matches the cached prefix automatically.
        if self.paged_adapter.is_some() {
            return self.chat_stream_sync_core_paged(
                full_token_history.clone(),
                tokenizer_for_decode,
                eos_id,
                p,
                report_perf,
                cb,
                cancelled,
            );
        }

        // Compiled path availability check, same pattern as chat_tokens_delta_sync.
        let use_compiled = unsafe { mlx_sys::mlx_qwen35_get_model_id() } == model_id;
        let _compiled_lock = if use_compiled {
            Some(
                COMPILED_LIFECYCLE_MUTEX
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
            // Re-apply the saved M-RoPE offset if the session carries
            // image state. The delta prefill just ran against the live
            // KV caches, which still encode the prior VLM prefill's
            // image attention; without re-applying the offset here, the
            // newly-built compiled graph would use a sequential M-RoPE
            // position and misposition all decoded tokens relative to
            // the cached image patches. `cached_rope_deltas` stays
            // alive across deltas so chained text-only turns on the
            // same image session keep the offset.
            if let Some(delta) = self.cached_rope_deltas {
                unsafe {
                    mlx_sys::mlx_qwen35_compiled_adjust_offset(delta);
                }
            }

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
                    eval_layer_caches(&self.caches)?;
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
        // consistent for the next turn. Delta continuations preserve
        // `cached_image_key` so the next turn's cache-prefix verify
        // still sees the prior prefill's image state.
        chat_common::save_cache_state_after_delta(
            p.reuse_cache,
            &generated_tokens,
            &finish_reason,
            &save_tokens,
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
            // Suppress residual when it is reasoning text and
            // include_reasoning == false.
            if p.include_reasoning || !last_is_reasoning {
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
                        cached_tokens: None,
                        performance: None,
                        is_reasoning: Some(last_is_reasoning),
                    }),
                    ThreadsafeFunctionCallMode::NonBlocking,
                );
            }
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
                raw_text: Some(chat_common::raw_text_with_reasoning_suppressed(
                    &text,
                    &generated_tokens,
                    enable_thinking.unwrap_or(true),
                    think_end_id,
                    think_end_str.as_deref(),
                    p.include_reasoning,
                )),
                // Delta path reuses the full prior history by construction
                // — report `prior_cached_len` (captured before the
                // `self.cached_token_history` extend above) as the
                // authoritative cached-prefix length.
                cached_tokens: Some(prior_cached_len),
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

        // Block-paged dispatch — early-return BEFORE the compile lock.
        // See `chat_sync_core` for the compile-lockout rationale.
        if self.paged_adapter.is_some() {
            if has_images {
                return Err(Error::from_reason(
                    "Qwen3.5 paged dispatch is text-only; image-bearing turns require \
                     use_block_paged_cache=false (text-only turns continue to work).",
                ));
            }
            return self.chat_stream_sync_core_paged(
                tokens,
                tokenizer_for_decode,
                eos_token_id,
                p,
                report_perf,
                cb,
                cancelled,
            );
        }

        // Check compiled path
        let use_compiled = unsafe { mlx_sys::mlx_qwen35_get_model_id() } == model_id;
        let _compiled_lock = if use_compiled {
            Some(
                COMPILED_LIFECYCLE_MUTEX
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
                let per_image_token_counts =
                    compute_image_token_counts_per_image(&processed_pre.grid_thw(), sms)?;
                let expanded = inject_image_placeholders(&tokens, &per_image_token_counts);
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

        // Zero-delta guard. See the matching `chat_sync_core` comment for
        // the design rationale — rewinding a GDN recurrent cache by one
        // token is not possible, so the only safe response to an exact-
        // match prompt is a full reset + re-prefill.
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
                    eval_layer_caches(&self.caches)?;
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
            // Suppress residual when it is reasoning text and
            // include_reasoning == false.
            if p.include_reasoning || !last_is_reasoning {
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
                        cached_tokens: None,
                        performance: None,
                        is_reasoning: Some(last_is_reasoning),
                    }),
                    ThreadsafeFunctionCallMode::NonBlocking,
                );
            }
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
                raw_text: Some(chat_common::raw_text_with_reasoning_suppressed(
                    &text,
                    &generated_tokens,
                    enable_thinking.unwrap_or(true),
                    think_end_id,
                    think_end_str.as_deref(),
                    p.include_reasoning,
                )),
                // Start path: report the matched prefix length from
                // `verify_cache_prefix_direct`. Zero on a miss, full
                // cached length on an exact-append hit.
                cached_tokens: Some(cached_prefix_len as u32),
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
    /// Snapshot of `Qwen35Inner::paged_adapter.is_some()` captured at
    /// construction time. Currently default-OFF on Qwen3.5 (parity-pending
    /// — see CLAUDE.md and `Qwen3_5Config::use_block_paged_cache`).
    /// VLM checkpoints can load with the adapter on for text-only
    /// inference; image-bearing chat turns are rejected at runtime by
    /// the chat-entry sites. Surfaced through the
    /// `hasBlockPagedCache()` NAPI method.
    pub(crate) paged_active: bool,
    /// RAII: unregisters this model's baseline from the cache-limit
    /// coordinator on drop, so the global cap can shrink once JS GCs
    /// the wrapper.
    pub(crate) _cache_limit_guard: crate::cache_limit::CacheLimitGuard,
}

#[napi]
impl Qwen3_5Model {
    /// Reset all caches.
    #[napi]
    pub fn reset_caches(&self) -> Result<()> {
        crate::model_thread::send_and_block(&self.thread, |reply| Qwen35Cmd::ResetCaches { reply })
    }

    /// Whether the block-paged KV cache adapter is active on this model
    /// instance.
    ///
    /// `true` iff `Qwen35Inner::paged_adapter` was successfully
    /// constructed at load time (driven by
    /// `Qwen3_5Config::use_block_paged_cache`, currently default-OFF
    /// because parity is pending real-weights validation). On VLM
    /// checkpoints the adapter can still be active for text-only
    /// inference; image-bearing chat turns are rejected at runtime by
    /// the chat-entry sites. Surfaced through this NAPI method so
    /// server endpoints can branch on it without round-tripping through
    /// the model thread.
    #[napi]
    pub fn has_block_paged_cache(&self) -> bool {
        self.paged_active
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
    /// `is_error` is the structured tool-error signal. When `Some(true)`,
    /// the renderer prepends the shared
    /// [`crate::tokenizer::TOOL_ERROR_MARKER`] inside the
    /// `<tool_response>` wrapper so the model receives a clear text-level
    /// cue. `None` / `Some(false)` keep the wire bytes byte-equal to the
    /// pre-feature output.
    ///
    /// Requires a live session started via `chatSessionStart`.
    #[napi]
    pub async fn chat_session_continue_tool(
        &self,
        tool_call_id: String,
        content: String,
        config: Option<ChatConfig>,
        is_error: Option<bool>,
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
                is_error,
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
    /// `is_error` mirrors the non-streaming entry point — when
    /// `Some(true)`, the renderer prepends the shared
    /// [`crate::tokenizer::TOOL_ERROR_MARKER`] inside the
    /// `<tool_response>` wrapper.
    #[napi(
        ts_args_type = "toolCallId: string, content: string, config: ChatConfig | null, callback: (err: Error | null, chunk: ChatStreamChunk) => void, isError?: boolean | null | undefined"
    )]
    pub async fn chat_stream_session_continue_tool(
        &self,
        tool_call_id: String,
        content: String,
        config: Option<ChatConfig>,
        callback: ThreadsafeFunction<ChatStreamChunk, ()>,
        is_error: Option<bool>,
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
            is_error,
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
///
/// E55: bumped 1024 → 2048 after benching against mlx-lm at 20k prompt:
/// chunk=1024 incurred 20 chunk boundaries vs mlx-lm's 10 (mlx-lm uses
/// 2048 by default); the doubled per-chunk overhead cost ~14% at 20k.
/// At 1024-prompt single-chunk the value is irrelevant — the loop is
/// guarded by `total_len - offset > PREFILL_STEP_SIZE` so any T < step
/// goes through the single `remaining` branch unchanged.
const PREFILL_STEP_SIZE: i64 = 2048;

/// Evaluate all cache arrays across all layers to materialize them on GPU.
/// Must be called between prefill chunks to break lazy dependency chains.
pub(crate) fn eval_layer_caches(caches: &Option<Vec<Qwen3_5LayerCache>>) -> Result<()> {
    if let Some(caches) = caches {
        let mut arrays: Vec<&MxArray> = Vec::new();
        for cache in caches.iter() {
            cache.collect_arrays(&mut arrays);
        }
        MxArray::eval_arrays(&arrays)?;
    }
    Ok(())
}

/// Async variant of `eval_layer_caches`: kicks GPU on cache materialization
/// but does NOT block the CPU. Used between prefill chunks so the CPU can
/// start building the next chunk's graph while the previous chunk's cache
/// writes are still in flight.
pub(crate) fn async_eval_layer_caches(caches: &Option<Vec<Qwen3_5LayerCache>>) {
    if let Some(caches) = caches {
        let mut arrays: Vec<&MxArray> = Vec::new();
        for cache in caches.iter() {
            cache.collect_arrays(&mut arrays);
        }
        MxArray::async_eval_arrays(&arrays);
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

    // E28: env-var toggle for A/B. Default: async between chunks. When set,
    // falls back to synchronous eval_layer_caches (the prior behavior).
    let chunk_async = std::env::var("MLX_PREFILL_SYNC_BETWEEN_CHUNKS").is_err();
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
        if chunk_async {
            async_eval_layer_caches(caches);
        } else {
            eval_layer_caches(caches)?;
        }
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
    // Plain layer loop. In-loop async_eval was tested (every 8 layers,
    // including h + all cache arrays) and found neutral-to-negative at
    // single-chunk prefill on M3 (back-to-back A/B in the same binary
    // showed deltas inside the run-to-run noise band). The CPU/GPU
    // overlap benefit only materializes across the inter-chunk barrier
    // in chunked_prefill, which now uses async_eval_layer_caches.
    // E37: slice h to last token before final_norm + lm_head. All callers
    // of forward_inner only consume the last token's logits (chat decode is
    // already T=1; prefill samples from last token only — see e.g. line
    // ~5151's `slice_axis(1, seq_len-1, seq_len)`). For prefill with T=860
    // this skips ~99.9% of the lm_head matmul.
    // E38: additionally, in the last layer, slice h to last token BEFORE
    // the MLP. Skips ~120 GFLOPs of wasted MLP work for the discarded T-1
    // rows. Cache writes still happen on full T inside attention. For
    // decode (T=1) both slices are no-ops.
    // Env-toggle MLX_DISABLE_E37_LAST_TOKEN_SLICE=1 reverts both for A/B.
    let slice_last_token = std::env::var("MLX_DISABLE_E37_LAST_TOKEN_SLICE").is_err();
    for i in 0..num_layers {
        let cache = caches.as_mut().map(|c| &mut c[i]);
        let is_last_layer = i + 1 == num_layers;
        h = layers[i].forward_with_optional_last_slice(
            &h,
            None,
            cache,
            None,
            true,
            slice_last_token && is_last_layer,
        )?;
    }

    let h = if slice_last_token {
        // E37 may already have reduced h to [B, 1, hidden] in the last layer;
        // this is a no-op in that case but kept for decode (T=1) safety.
        let seq_len = h.shape_at(1)?;
        if seq_len > 1 {
            h.slice_axis(1, seq_len - 1, seq_len)?
        } else {
            h
        }
    } else {
        h
    };

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
// Phase 5 piece 1: C++ compiled paged-decode dispatcher (Dense).
//
// Mirrors `init_paged_moe_compiled_session` / `forward_moe_cpp_paged`
// from `crates/mlx-core/src/models/qwen3_5_moe/model.rs` but without
// MoE expert routing — the dense compiled paged graph in
// `mlx_qwen35.cpp` is structurally simpler.
// ============================================================================

/// Block size hard-coded into the compiled C++ paged graph
/// (`mlx_qwen35.cpp` — see `attn_for_compile_paged` and the
/// `mlx_qwen35_init_paged` docstring). The Rust adapter supports
/// configurable block sizes via `Qwen3_5Config::paged_block_size`,
/// but the compiled graph traces against block_size=16 baked into the
/// `paged_kv_write` / `paged_attention` kernel calls. Mismatched values
/// would have Rust encode slot/block tables at the adapter's block size
/// while C++ writes/reads at 16, corrupting KV state. The compile-branch
/// selectors gate `cpp_session_ready` on this equality and fall back to
/// the pure-Rust paged path when the adapter is configured otherwise.
pub(crate) const CPP_PAGED_REQUIRED_BLOCK_SIZE: u32 = 16;

/// Initialize the C++ paged forward graph from the live `paged_adapter`
/// pool/scale arrays AND the per-layer linear-attention recurrent caches
/// already populated by the pure-Rust paged prefill.
///
/// # Layer-index contract
///
/// The C++ FFI accepts pool/scale handle arrays of size `num_layers`
/// (absolute decoder count). For each absolute layer index `i`:
/// * Linear-attention layers: pool/scale slots are null pointers; the
///   `linear_cache_arrays` pair `[i*2, i*2+1]` holds
///   `(conv_state, recurrent_state)` from the layer's
///   `Qwen3_5LayerCache::Linear(ArraysCache)`.
/// * Full-attention layers: pool/scale slots come from the adapter's
///   `LayerKVPool` at the COMPACT (full-attention) ordinal; the linear
///   cache pair is null.
///
/// The compact-ordinal mapping is computed via
/// [`crate::models::qwen3_5::decoder_layer::compute_layer_kinds`], the
/// same helper the production Rust paged-forward dispatch uses.
///
/// # Caller contract
///
/// 1. `caches` is fully populated by a prior pure-Rust paged prefill.
/// 2. The C++ weights for this model are still registered (caller must
///    have verified `mlx_qwen35_get_model_id() == self.model_id` and
///    holds the appropriate read locks).
///
/// `prefill_offset` is the global token cursor the compiled paged
/// graph's `g_dense_paged_offset_int` will start incrementing from.
///
/// On any failure (missing linear cache, missing pool/scale handle, or
/// the C++ FFI returning a non-zero status), the helper returns `Err`
/// so the caller can fall back to the pure-Rust paged decode path.
fn init_paged_dense_compiled_session(
    config: &Qwen3_5Config,
    caches: &[Qwen3_5LayerCache],
    paged_adapter: &PagedKVCacheAdapter,
    prefill_offset: i32,
) -> Result<()> {
    use super::decoder_layer::{Qwen3_5LayerKind, compute_layer_kinds};

    let num_layers_us = config.num_layers as usize;
    if caches.len() != num_layers_us {
        return Err(Error::from_reason(format!(
            "init_paged_dense_compiled_session: caches.len()={} but config.num_layers={}",
            caches.len(),
            num_layers_us
        )));
    }

    let layer_kinds = compute_layer_kinds(num_layers_us, |i| config.is_linear_layer(i));

    let mut k_pool_handles: Vec<*mut mlx_sys::mlx_array> =
        vec![std::ptr::null_mut(); num_layers_us];
    let mut v_pool_handles: Vec<*mut mlx_sys::mlx_array> =
        vec![std::ptr::null_mut(); num_layers_us];
    let mut k_scale_handles: Vec<*mut mlx_sys::mlx_array> =
        vec![std::ptr::null_mut(); num_layers_us];
    let mut v_scale_handles: Vec<*mut mlx_sys::mlx_array> =
        vec![std::ptr::null_mut(); num_layers_us];
    let mut linear_cache_handles: Vec<*mut mlx_sys::mlx_array> =
        vec![std::ptr::null_mut(); num_layers_us * 2];

    // Hold the wrapping `MxArray`s alive across the FFI call so the C++
    // side has time to copy them into its own globals.
    let mut held_arrays: Vec<MxArray> = Vec::with_capacity(num_layers_us * 4);

    for (i, kind) in layer_kinds.iter().enumerate() {
        match kind {
            Qwen3_5LayerKind::Linear => {
                let arrays_cache = match &caches[i] {
                    Qwen3_5LayerCache::Linear(c) => c,
                    Qwen3_5LayerCache::FullAttention(_) => {
                        return Err(Error::from_reason(format!(
                            "init_paged_dense_compiled_session: layer {i} is Linear by config \
                             but cache slot is FullAttention",
                        )));
                    }
                };
                let conv = arrays_cache.get(0).ok_or_else(|| {
                    Error::from_reason(format!(
                        "init_paged_dense_compiled_session: layer {i} conv_state not populated; \
                         pure-Rust paged prefill must run before C++ paged init",
                    ))
                })?;
                let rec = arrays_cache.get(1).ok_or_else(|| {
                    Error::from_reason(format!(
                        "init_paged_dense_compiled_session: layer {i} recurrent_state not \
                         populated; pure-Rust paged prefill must run before C++ paged init",
                    ))
                })?;
                linear_cache_handles[i * 2] = conv.as_raw_ptr();
                linear_cache_handles[i * 2 + 1] = rec.as_raw_ptr();
                let _ = (conv, rec);
            }
            Qwen3_5LayerKind::FullAttentionPaged { paged_idx } => {
                let k_arr = paged_adapter.key_pool_array(*paged_idx).map_err(|e| {
                    Error::from_reason(format!(
                        "init_paged_dense_compiled_session: key_pool_array(layer={i}, \
                         paged_idx={paged_idx}): {e}",
                    ))
                })?;
                let v_arr = paged_adapter.value_pool_array(*paged_idx).map_err(|e| {
                    Error::from_reason(format!(
                        "init_paged_dense_compiled_session: value_pool_array(layer={i}, \
                         paged_idx={paged_idx}): {e}",
                    ))
                })?;
                let ks_arr = paged_adapter.k_scale_array(*paged_idx).map_err(|e| {
                    Error::from_reason(format!(
                        "init_paged_dense_compiled_session: k_scale_array(layer={i}, \
                         paged_idx={paged_idx}): {e}",
                    ))
                })?;
                let vs_arr = paged_adapter.v_scale_array(*paged_idx).map_err(|e| {
                    Error::from_reason(format!(
                        "init_paged_dense_compiled_session: v_scale_array(layer={i}, \
                         paged_idx={paged_idx}): {e}",
                    ))
                })?;
                k_pool_handles[i] = k_arr.as_raw_ptr();
                v_pool_handles[i] = v_arr.as_raw_ptr();
                k_scale_handles[i] = ks_arr.as_raw_ptr();
                v_scale_handles[i] = vs_arr.as_raw_ptr();
                held_arrays.push(k_arr);
                held_arrays.push(v_arr);
                held_arrays.push(ks_arr);
                held_arrays.push(vs_arr);
            }
        }
    }

    let status = unsafe {
        mlx_sys::mlx_qwen35_init_paged(
            config.num_layers,
            config.hidden_size,
            config.num_heads,
            config.num_kv_heads,
            config.head_dim,
            config.rope_theta as f32,
            config.rope_dims(),
            config.rms_norm_eps as f32,
            config.full_attention_interval,
            config.linear_num_key_heads,
            config.linear_num_value_heads,
            config.linear_key_head_dim,
            config.linear_value_head_dim,
            config.linear_conv_kernel_dim,
            if config.tie_word_embeddings { 1 } else { 0 },
            config.max_position_embeddings,
            1, // batch_size
            k_pool_handles.as_mut_ptr(),
            v_pool_handles.as_mut_ptr(),
            k_scale_handles.as_mut_ptr(),
            v_scale_handles.as_mut_ptr(),
            linear_cache_handles.as_mut_ptr(),
            prefill_offset,
        )
    };

    drop(held_arrays);

    // The C++ side returns 0 on success, -1 on failure. Failure paths
    // include missing pool/scale handles for full-attention layers and
    // any exception caught during graph build. On failure
    // `g_dense_paged_inited` is left cleared so subsequent
    // `mlx_qwen35_forward_paged` calls would null-out their logits;
    // surfacing the failure here lets the dispatcher fall back to the
    // pure-Rust paged path before any decode-step FFI is dispatched.
    if status != 0 {
        return Err(Error::from_reason(format!(
            "init_paged_dense_compiled_session: mlx_qwen35_init_paged returned status={status} \
             (expected 0); see stderr for the C++ diagnostic. Caller must fall back to the \
             pure-Rust paged path."
        )));
    }

    Ok(())
}

/// Single-token decode step using the C++ compiled paged forward pass.
///
/// Mirrors `forward_compiled` but threads through the paged-attention
/// inputs (offset_arr, block_table, slot_mapping, num_valid_tokens,
/// num_valid_blocks, seq_lens) so K/V is written into the adapter's
/// paged Metal pool via `paged_kv_write` and gathered via
/// `paged_attention`.
///
/// Caller contract:
/// * `init_paged_dense_compiled_session` has been called this turn
///   (sets `g_dense_paged_inited = true`).
/// * `paged_adapter.record_tokens(&[token_id])` has been called to
///   advance the cursor (and lazily allocate any new block).
/// * `inputs` was just built via
///   `paged_adapter.build_paged_attention_inputs(1, 1, max_blocks_per_seq)`.
///
/// On any FFI failure (`output_logits == null`) the helper returns an
/// `Err` so the dispatcher falls back to pure-Rust paged decode.
fn forward_dense_cpp_paged(
    input_ids: &MxArray,
    embedding_weight: &MxArray,
    inputs: &crate::transformer::paged_attention_inputs::PagedAttentionInputs,
) -> Result<MxArray> {
    use mlx_sys as sys;

    let mut output_ptr: *mut sys::mlx_array = std::ptr::null_mut();
    let mut cache_offset_out: i32 = 0;
    unsafe {
        sys::mlx_qwen35_forward_paged(
            input_ids.as_raw_ptr(),
            embedding_weight.as_raw_ptr(),
            inputs.offset_arr.as_raw_ptr(),
            inputs.block_table.as_raw_ptr(),
            inputs.slot_mapping.as_raw_ptr(),
            inputs.num_valid_tokens.as_raw_ptr(),
            inputs.num_valid_blocks.as_raw_ptr(),
            inputs.seq_lens.as_raw_ptr(),
            &mut output_ptr,
            &mut cache_offset_out,
        );
    }

    if output_ptr.is_null() {
        return Err(Error::from_reason(
            "C++ Dense paged forward step returned null — check stderr for diagnostic. \
             (Common causes: g_dense_paged_inited = false, slot_mapping shape != [1], \
             input_ids size != 1, or weights cleared by another model load.)",
        ));
    }

    MxArray::from_handle(output_ptr, "dense_paged_forward_logits")
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

/// Compute the per-image merged-token count from a processed grid_thw
/// array. Each entry is the number of `IMAGE_TOKEN_ID` slots that image
/// must occupy in the prompt so the vision embeddings align 1:1 with the
/// corresponding token positions.
pub(crate) fn compute_image_token_counts_per_image(
    grid: &MxArray,
    spatial_merge_size: i32,
) -> Result<Vec<usize>> {
    grid.eval();
    let grid_data = grid.to_int32()?;
    let merge_factor = spatial_merge_size * spatial_merge_size;
    let mut counts = Vec::with_capacity(grid_data.len() / 3);
    for i in 0..(grid_data.len() / 3) {
        let t = grid_data[i * 3];
        let h = grid_data[i * 3 + 1];
        let w = grid_data[i * 3 + 2];
        counts.push(((t * h * w) / merge_factor) as usize);
    }
    Ok(counts)
}

/// Ensure the tokenized prompt contains the right number of
/// `IMAGE_TOKEN_ID` placeholders — one per vision patch, in the order
/// produced by the chat template.
///
/// Three input shapes are accepted:
///
/// 1. **Template emitted one `<|image_pad|>` per image** (the proper
///    Qwen VLM shape, produced by
///    `tokenizer::serialize_message_for_jinja` when the user turn
///    carries images). Each placeholder is expanded in-place to its
///    image's grid count. This keeps the vision tokens inside the user
///    turn — `get_rope_index` builds correct M-RoPE positions and the
///    model attends to the image in-context.
///
/// 2. **Template already emitted the fully expanded count** (non-Qwen
///    templates that inline the full patch run). Pass through unchanged.
///
/// 3. **Template emitted zero placeholders** (non-VLM template, or a
///    VLM template that silently drops vision markers). Splice the
///    total count right after BOS as a last-resort fallback. Vision
///    tokens land outside the user turn; this usually still produces
///    sensible output for simple prompts but M-RoPE position IDs are
///    suboptimal.
pub(crate) fn inject_image_placeholders(
    tokens: &[u32],
    per_image_token_counts: &[usize],
) -> Vec<u32> {
    let total: usize = per_image_token_counts.iter().sum();
    if total == 0 {
        return tokens.to_vec();
    }
    let existing = tokens
        .iter()
        .filter(|&&t| t == IMAGE_TOKEN_ID as u32)
        .count();

    if existing == 0 {
        // Case 3 — fallback splice after BOS.
        let mut new_tokens = tokens.to_vec();
        let placeholders: Vec<u32> = vec![IMAGE_TOKEN_ID as u32; total];
        new_tokens.splice(1..1, placeholders);
        return new_tokens;
    }

    if existing == per_image_token_counts.len() {
        // Case 1 — one placeholder per image; expand each in place to
        // its grid count. Capacity pre-sized to the final length so no
        // reallocations.
        let mut new_tokens: Vec<u32> = Vec::with_capacity(tokens.len() + total - existing);
        let mut img_iter = per_image_token_counts.iter().copied();
        for &t in tokens {
            if t == IMAGE_TOKEN_ID as u32 {
                match img_iter.next() {
                    Some(count) => {
                        new_tokens.extend(std::iter::repeat_n(IMAGE_TOKEN_ID as u32, count));
                    }
                    None => {
                        // More placeholders than images — preserve as-is
                        // and let `get_rope_index` surface the mismatch.
                        new_tokens.push(t);
                    }
                }
            } else {
                new_tokens.push(t);
            }
        }
        return new_tokens;
    }

    // Case 2 (existing == total) or unknown shape — return as-is.
    // `get_rope_index` will surface any mismatch.
    tokens.to_vec()
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

        // Scan `batch_tokens` for maximal contiguous runs of
        // `image_token_id`. After the tokenizer fix that serialises
        // one `<|image_pad|>` per image inline in the user turn and
        // `inject_image_placeholders` expands each marker in place,
        // the prompt can carry MULTIPLE separated image runs when
        // history is replayed (e.g. two image-bearing user turns
        // joined by an assistant reply). Flattening the span from
        // `positions[0]` to `positions[last]` the way the old
        // single-span code did would skip every interior text token
        // between runs and blow up the downstream shape check.
        let mut image_runs: Vec<(usize, usize)> = Vec::new();
        {
            let mut i = 0;
            while i < batch_tokens.len() {
                if batch_tokens[i] == image_token_id {
                    let start = i;
                    while i < batch_tokens.len() && batch_tokens[i] == image_token_id {
                        i += 1;
                    }
                    image_runs.push((start, i));
                } else {
                    i += 1;
                }
            }
        }

        if image_runs.is_empty() {
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

        let total_image_tokens: usize = image_runs.iter().map(|(s, e)| e - s).sum();
        if total_expected_tokens != total_image_tokens {
            return Err(Error::new(
                Status::GenericFailure,
                format!(
                    "Image token count mismatch: expected {} from grid, found {} in prompt",
                    total_expected_tokens, total_image_tokens,
                ),
            ));
        }

        // Two token layouts are valid here:
        //
        //  (a) N runs, one per image — the proper Qwen VLM shape after
        //      the tokenizer serialiser emits a `{type:"image"}` part
        //      per image and `inject_image_placeholders` expands each
        //      marker in place. Per-run length must match its grid.
        //
        //  (b) 1 big run whose length equals the grids' total —
        //      the fallback layout for chat templates that emit no
        //      `<|image_pad|>` markers, where
        //      `inject_image_placeholders` crams every image's tokens
        //      into a single splice after BOS. No text gap sits between
        //      images in this layout, so the position walk collapses
        //      consecutive sub-runs into one contiguous span without
        //      emitting any interior text.
        //
        // We canonicalise both into a `per_image_offsets: Vec<(start,
        // grid_info)>` list of length `num_images` and feed it to the
        // position walk below. Any other shape is ambiguous (we'd have
        // to guess which grid goes with which run) — reject it.
        let per_image_offsets: Vec<(usize, (i64, i64, i64, usize))> = if image_runs.len()
            == num_images
        {
            // Case (a): validate per-run length, then pair by ordinal.
            for (run_idx, (run_start, run_end)) in image_runs.iter().enumerate() {
                let expected = image_token_info[run_idx].3;
                let actual = run_end - run_start;
                if expected != actual {
                    return Err(Error::new(
                        Status::GenericFailure,
                        format!(
                            "Image run {run_idx} has {actual} placeholder tokens but its grid expects {expected}",
                        ),
                    ));
                }
            }
            image_runs
                .iter()
                .zip(image_token_info.iter().copied())
                .map(|((start, _), info)| (*start, info))
                .collect()
        } else if image_runs.len() == 1 {
            // Case (b): fallback splice — synthesise per-image start
            // offsets by walking `image_token_info` lengths from the
            // single run's start. Total was already validated against
            // `total_expected_tokens` above, so this just distributes
            // the shared span across the grids.
            let big_start = image_runs[0].0;
            let mut offsets = Vec::with_capacity(num_images);
            let mut cursor = big_start;
            for info in image_token_info.iter().copied() {
                offsets.push((cursor, info));
                cursor += info.3;
            }
            offsets
        } else {
            return Err(Error::new(
                Status::GenericFailure,
                format!(
                    "Image run layout mismatch: prompt carries {} contiguous image-token runs but {} images \
                     were processed; expected either one run per image or a single contiguous fallback run \
                     containing every image's tokens.",
                    image_runs.len(),
                    num_images,
                ),
            ));
        };

        // End of the last image token in the token stream — everything
        // beyond is trailing text. For case (a) this is the last run's
        // end; for case (b) it's the shared run's end. In both cases
        // it equals `image_runs.last().unwrap().1`.
        let last_image_end = image_runs.last().expect("at least one run").1;

        // Emit positions by walking the sequence: text gap, image,
        // text gap, image, … final text gap. `current_pos` carries the
        // M-RoPE counter forward across both text and image segments so
        // every token gets a monotonically non-decreasing position id
        // in each axis. Synthesised case-(b) sub-runs sit back-to-back
        // so their text-gap loops iterate zero times between them —
        // the walk collapses naturally.
        let mut cursor: usize = 0;
        let mut current_pos: i64 = 0;

        for (run_start, info) in per_image_offsets.iter().copied() {
            // Text gap before this image run (zero-length for adjacent
            // case-(b) sub-runs after the first).
            for _ in cursor..run_start {
                all_position_ids[0].push(current_pos);
                all_position_ids[1].push(current_pos);
                all_position_ids[2].push(current_pos);
                current_pos += 1;
            }

            // Spatial positions for the image at this run
            let (llm_grid_t, llm_grid_h, llm_grid_w, count) = info;
            let image_base = current_pos;
            for t_idx in 0..llm_grid_t {
                for h_idx in 0..llm_grid_h {
                    for w_idx in 0..llm_grid_w {
                        all_position_ids[0].push(image_base + t_idx);
                        all_position_ids[1].push(image_base + h_idx);
                        all_position_ids[2].push(image_base + w_idx);
                    }
                }
            }
            let max_axis = std::cmp::max(
                llm_grid_t - 1,
                std::cmp::max(llm_grid_h - 1, llm_grid_w - 1),
            );
            current_pos = image_base + max_axis + 1;
            cursor = run_start + count;
        }

        // Trailing text after the last image (run in case (a), sub-run
        // end in case (b) — both resolve to `last_image_end`).
        debug_assert_eq!(cursor, last_image_end);
        let _ = last_image_end;
        for _ in cursor..seq_len as usize {
            all_position_ids[0].push(current_pos);
            all_position_ids[1].push(current_pos);
            all_position_ids[2].push(current_pos);
            current_pos += 1;
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

#[cfg(test)]
mod image_placeholder_tests {
    use super::*;

    const BOS: u32 = 1;
    const USER: u32 = 100;
    const TEXT: u32 = 200;
    const IMG: u32 = IMAGE_TOKEN_ID as u32;

    #[test]
    fn expands_single_placeholder_per_image_inline() {
        // Template emitted: BOS, USER, <|image_pad|>, TEXT
        // Expected: BOS, USER, <|image_pad|>×5, TEXT  (vision wrapper stays
        // INSIDE the user turn instead of getting spliced after BOS).
        let tokens = vec![BOS, USER, IMG, TEXT];
        let out = inject_image_placeholders(&tokens, &[5]);
        assert_eq!(out, vec![BOS, USER, IMG, IMG, IMG, IMG, IMG, TEXT]);
    }

    #[test]
    fn expands_distinct_grid_counts_for_multiple_images_in_order() {
        // Two images with different grid sizes — each placeholder must be
        // replaced by its own image's count, not the other way around.
        let tokens = vec![BOS, IMG, TEXT, IMG];
        let out = inject_image_placeholders(&tokens, &[2, 3]);
        assert_eq!(out, vec![BOS, IMG, IMG, TEXT, IMG, IMG, IMG]);
    }

    #[test]
    fn empty_counts_is_passthrough() {
        let tokens = vec![BOS, USER, TEXT];
        let out = inject_image_placeholders(&tokens, &[]);
        assert_eq!(out, tokens);
    }

    #[test]
    fn fallback_splices_total_after_bos_when_template_emitted_none() {
        // Case 3: template didn't emit any placeholder. Fallback splice
        // preserved for non-VLM templates / silent vision-dropping
        // templates.
        let tokens = vec![BOS, USER, TEXT];
        let out = inject_image_placeholders(&tokens, &[3]);
        assert_eq!(out, vec![BOS, IMG, IMG, IMG, USER, TEXT]);
    }

    #[test]
    fn fully_expanded_input_passes_through_unchanged() {
        // Case 2: template already emitted the full 5-token run. `existing`
        // (5) != `per_image.len()` (1) so the "one-per-image" branch
        // doesn't fire; total (5) matches so no fallback splice either.
        let tokens = vec![BOS, USER, IMG, IMG, IMG, IMG, IMG, TEXT];
        let out = inject_image_placeholders(&tokens, &[5]);
        assert_eq!(out, tokens);
    }

    #[test]
    fn preserves_relative_position_of_surrounding_tokens() {
        // Regression guard: every non-IMG token must survive in its
        // original relative order.
        let tokens = vec![BOS, USER, 10, 11, IMG, 12, 13];
        let out = inject_image_placeholders(&tokens, &[4]);
        assert_eq!(out, vec![BOS, USER, 10, 11, IMG, IMG, IMG, IMG, 12, 13]);
    }
}

#[cfg(test)]
mod rope_index_tests {
    //! `get_rope_index` builds M-RoPE position IDs for VLM prefill. The
    //! pre-fix implementation collapsed every `IMAGE_TOKEN_ID` in the
    //! prompt to a single contiguous span from `positions[0]` to
    //! `positions[last]`; a multi-turn history with two image-bearing
    //! user turns joined by an assistant reply silently skipped every
    //! interior text token, leaving `all_position_ids` shorter than
    //! `seq_len` and crashing the downstream reshape with a cryptic
    //! "length mismatch". These tests pin per-run indexing against that
    //! regression.
    use super::*;
    use crate::array::MxArray;
    use std::sync::Mutex;
    use std::sync::OnceLock;
    const IMG: i32 = IMAGE_TOKEN_ID;
    const TEXT_A: i32 = 100;
    const TEXT_B: i32 = 200;

    /// MLX's MPS backend is not re-entrant — every test that touches an
    /// `MxArray` must hold this mutex so only one such test runs at a
    /// time across the test binary.
    fn mlx_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    /// Encode a flat `Vec<i32>` token stream as a `[1, seq_len]`
    /// `MxArray` and a `[num_images, 3]` grid array (or `None`) to
    /// feed `get_rope_index`.
    fn mk_inputs(tokens: &[i32], grids: &[(i64, i64, i64)]) -> (MxArray, Option<MxArray>) {
        let seq_len = tokens.len() as i64;
        let input_ids = MxArray::from_int32(tokens, &[1, seq_len]).unwrap();
        let grid = if grids.is_empty() {
            None
        } else {
            let flat: Vec<i32> = grids
                .iter()
                .flat_map(|(t, h, w)| [*t as i32, *h as i32, *w as i32])
                .collect();
            Some(MxArray::from_int32(&flat, &[grids.len() as i64, 3]).unwrap())
        };
        (input_ids, grid)
    }

    fn extract_positions(pos: &MxArray) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
        // pos shape [3, 1, seq_len] — flatten to Vec<i32> and split.
        pos.eval();
        let flat = pos.to_int32().unwrap();
        let n = flat.len() / 3;
        (
            flat[0..n].to_vec(),
            flat[n..2 * n].to_vec(),
            flat[2 * n..3 * n].to_vec(),
        )
    }

    #[test]
    fn pure_text_prompt_gets_sequential_positions() {
        let _g = mlx_lock().lock().unwrap();
        let tokens = vec![TEXT_A, TEXT_B, TEXT_A, TEXT_B];
        let (ids, grid) = mk_inputs(&tokens, &[]);
        let (pos, rope_deltas) = get_rope_index(&ids, grid.as_ref(), 2, IMG).unwrap();
        let (t, h, w) = extract_positions(&pos);
        assert_eq!(t, vec![0, 1, 2, 3]);
        assert_eq!(h, vec![0, 1, 2, 3]);
        assert_eq!(w, vec![0, 1, 2, 3]);
        assert_eq!(rope_deltas, 0);
    }

    #[test]
    fn single_image_run_preserves_baseline_shape() {
        let _g = mlx_lock().lock().unwrap();
        // 2 text + (grid 2x2x2=8 tokens after spatial_merge=2, so t=2,h=4,w=4, split=2
        //   → llm grid 2×2×2 = 8 image tokens) + 2 text
        let tokens: Vec<i32> = [TEXT_A, TEXT_B]
            .iter()
            .chain(std::iter::repeat_n(&IMG, 8))
            .chain([TEXT_A, TEXT_B].iter())
            .copied()
            .collect();
        let (ids, grid) = mk_inputs(&tokens, &[(2, 4, 4)]);
        let (pos, _) = get_rope_index(&ids, grid.as_ref(), 2, IMG).unwrap();
        let (t, h, w) = extract_positions(&pos);
        // Leading text
        assert_eq!(&t[..2], &[0, 1]);
        assert_eq!(&h[..2], &[0, 1]);
        // Image span starts at 2; with llm_grid_t=2 h=2 w=2 → max_axis=1
        // so current_pos advances to 2 + 1 + 1 = 4 after the image.
        // Trailing text: 4, 5
        assert_eq!(&t[10..], &[4, 5]);
        assert_eq!(&h[10..], &[4, 5]);
        assert_eq!(&w[10..], &[4, 5]);
    }

    #[test]
    fn two_image_runs_separated_by_text_emits_every_position() {
        // THE regression test for the P2 bug — pre-fix this crashed the
        // downstream reshape with a length mismatch.
        let _g = mlx_lock().lock().unwrap();
        let mut tokens: Vec<i32> = Vec::new();
        tokens.push(TEXT_A); // position 0
        tokens.extend(std::iter::repeat_n(IMG, 8)); // 1 image → llm 2×2×2=8
        tokens.push(TEXT_A); // interior text between images
        tokens.push(TEXT_B);
        tokens.extend(std::iter::repeat_n(IMG, 8)); // 2nd image → same grid
        tokens.push(TEXT_A); // trailing text

        let (ids, grid) = mk_inputs(&tokens, &[(2, 4, 4), (2, 4, 4)]);
        let (pos, _) = get_rope_index(&ids, grid.as_ref(), 2, IMG).unwrap();
        let (t, _h, _w) = extract_positions(&pos);

        // seq_len == tokens.len() — every token must have a position
        // (the old single-span path dropped the interior text entries,
        // which then failed the reshape at the end of get_rope_index).
        assert_eq!(
            t.len(),
            tokens.len(),
            "position count must equal token count"
        );

        // Leading text at pos 0
        assert_eq!(t[0], 0);
        // Image 1 at base=1, max_axis=1 → current_pos after = 3
        // Interior text: 3, 4
        assert_eq!(t[9], 3);
        assert_eq!(t[10], 4);
        // Image 2 at base=5, max_axis=1 → current_pos after = 7
        // Trailing text: 7
        assert_eq!(*t.last().unwrap(), 7);
    }

    #[test]
    fn leading_image_run_no_text_prefix() {
        let _g = mlx_lock().lock().unwrap();
        let tokens: Vec<i32> = std::iter::repeat_n(IMG, 8)
            .chain([TEXT_A, TEXT_B].iter().copied())
            .collect();
        let (ids, grid) = mk_inputs(&tokens, &[(2, 4, 4)]);
        let (pos, _) = get_rope_index(&ids, grid.as_ref(), 2, IMG).unwrap();
        let (t, _, _) = extract_positions(&pos);
        assert_eq!(t.len(), tokens.len());
        // Image at base=0, max_axis=1 → current_pos=2 after, trailing text 2, 3
        assert_eq!(&t[8..], &[2, 3]);
    }

    #[test]
    fn trailing_image_run_no_text_suffix() {
        let _g = mlx_lock().lock().unwrap();
        let tokens: Vec<i32> = [TEXT_A, TEXT_B]
            .iter()
            .copied()
            .chain(std::iter::repeat_n(IMG, 8))
            .collect();
        let (ids, grid) = mk_inputs(&tokens, &[(2, 4, 4)]);
        let (pos, _) = get_rope_index(&ids, grid.as_ref(), 2, IMG).unwrap();
        let (t, _, _) = extract_positions(&pos);
        assert_eq!(t.len(), tokens.len());
        assert_eq!(&t[..2], &[0, 1]);
    }

    #[test]
    fn run_count_must_match_image_count() {
        // 2 image runs in the prompt but only 1 grid supplied — ambiguous
        // pairing; reject.
        let _g = mlx_lock().lock().unwrap();
        let mut tokens = vec![TEXT_A];
        tokens.extend(std::iter::repeat_n(IMG, 4));
        tokens.push(TEXT_A);
        tokens.extend(std::iter::repeat_n(IMG, 4));
        let (ids, grid) = mk_inputs(&tokens, &[(2, 4, 4)]);
        let err = match get_rope_index(&ids, grid.as_ref(), 2, IMG) {
            Ok(_) => panic!("expected get_rope_index to error"),
            Err(e) => e,
        };
        assert!(
            err.reason.contains("Image run layout mismatch"),
            "got: {}",
            err.reason
        );
    }

    #[test]
    fn per_run_length_must_match_its_grid_count() {
        // 2 runs, 2 grids — but run 0 has too few tokens for grid 0.
        let _g = mlx_lock().lock().unwrap();
        let mut tokens = vec![TEXT_A];
        tokens.extend(std::iter::repeat_n(IMG, 4)); // should be 8 for (2,4,4)
        tokens.push(TEXT_A);
        tokens.extend(std::iter::repeat_n(IMG, 12)); // compensates the total, but per-run wrong
        let (ids, grid) = mk_inputs(&tokens, &[(2, 4, 4), (2, 4, 4)]);
        let err = match get_rope_index(&ids, grid.as_ref(), 2, IMG) {
            Ok(_) => panic!("expected get_rope_index to error"),
            Err(e) => e,
        };
        assert!(err.reason.contains("Image run 0"), "got: {}", err.reason);
    }

    #[test]
    fn multi_image_fallback_single_contiguous_run_is_accepted() {
        // Fallback case (b): chat template emits zero `<|image_pad|>`
        // markers and `inject_image_placeholders` crams every image's
        // tokens into a single splice after BOS. For N images with
        // distinct grids, the prompt carries ONE big contiguous run of
        // `sum(per_image_counts)` image tokens. The old strict guard
        // rejected this as "run layout mismatch" even though it was
        // the legitimate fallback layout that previously worked via
        // the old single-span position walk. The new path synthesises
        // per-image sub-run offsets from the shared span and emits
        // correct M-RoPE positions for each image.
        let _g = mlx_lock().lock().unwrap();
        // Two 1×2×2 grids → 4 image tokens each, 8 total.
        let mut tokens = vec![TEXT_A];
        tokens.extend(std::iter::repeat_n(IMG, 8));
        tokens.push(TEXT_B);
        let (ids, grid) = mk_inputs(&tokens, &[(1, 4, 4), (1, 4, 4)]);
        let (pos, _) = get_rope_index(&ids, grid.as_ref(), 2, IMG)
            .expect("fallback single-run layout for two images must be accepted");
        let (t, _, _) = extract_positions(&pos);
        assert_eq!(t.len(), tokens.len(), "every token must have a position");
        // Leading text at 0.
        assert_eq!(t[0], 0);
        // First image base = 1, llm grid 1×2×2, max_axis=1 → current_pos
        // after = 3. Next image base = 3, max_axis=1 → current_pos
        // after = 5. Trailing TEXT_B at 5.
        assert_eq!(*t.last().unwrap(), 5);
    }

    #[test]
    fn multi_image_fallback_with_distinct_grids_preserves_per_image_offsets() {
        // Same fallback shape but with DIFFERENT grid sizes per image —
        // the synthesised sub-run offsets must distribute the shared
        // span correctly (image[0] consumes its own count tokens,
        // image[1] starts right after).
        let _g = mlx_lock().lock().unwrap();
        // image 0: 1×2×2 → 4 tokens. image 1: 1×4×4 → 16 tokens. Total 20.
        let mut tokens = vec![TEXT_A];
        tokens.extend(std::iter::repeat_n(IMG, 20));
        tokens.push(TEXT_B);
        let (ids, grid) = mk_inputs(&tokens, &[(1, 4, 4), (1, 8, 8)]);
        let (pos, _) = get_rope_index(&ids, grid.as_ref(), 2, IMG)
            .expect("fallback layout with distinct per-image grids must succeed");
        let (t, _, _) = extract_positions(&pos);
        assert_eq!(t.len(), tokens.len());
        assert_eq!(t[0], 0);
        // image 0 base=1, max_axis = max(0,1,1) = 1 → current_pos = 3
        // image 1 base=3, max_axis = max(0,3,3) = 3 → current_pos = 7
        // Trailing TEXT_B at 7.
        assert_eq!(*t.last().unwrap(), 7);
    }
}

#[cfg(test)]
mod prefix_cache_reuse_integration_tests {
    //! End-to-end tests for the prefix KV cache reuse refactor on
    //! Qwen3.5 Dense. These verify that `chat_session_start_sync` no
    //! longer unconditionally wipes the cache — stateless agent clients
    //! (Aider, Codex CLI, pi-mono, etc.) that resend the full transcript
    //! on every turn should hit the `verify_cache_prefix_direct`
    //! exact-append path and pay only the delta prefill cost.
    //!
    //! These tests are `#[ignore]`-marked because they require loading a
    //! real Qwen3.5 model file and a tokenizer. Run them with:
    //!
    //!     cargo test -p mlx-core --test '*' -- --ignored prefix_cache_reuse_integration
    //!
    //! with `MLX_NODE_QWEN35_MODEL_DIR` set to a local Qwen3.5 dir
    //! (e.g. `~/models/Qwen3.5-0.8B`).
    //!
    //! The test bodies are intentionally skeletal — they document what
    //! needs to hold rather than wiring up the full model-loading
    //! boilerplate. Flesh them out alongside the upcoming end-to-end
    //! harness in `serve.ts` / the pi-mono vitest-migration smoke test
    //! (plan doc §Verification).

    /// Append hit: two back-to-back session-start calls where the second's
    /// token sequence is `first_tokens + extra_tokens`. The result of the
    /// second call must report `cached_tokens > 0` and a prefill that
    /// counts only the delta tokens.
    #[ignore = "requires a real Qwen3.5 Dense model directory; run with --ignored"]
    #[test]
    fn append_hit_reuses_cached_prefix() {
        // Pseudocode for the real test body:
        //
        //   let mut model = Qwen35Model::load(env!("MLX_NODE_QWEN35_MODEL_DIR"))?;
        //   let turn1 = vec![ChatMessage::user("What is 2+2?")];
        //   let r1 = model.chat_session_start_sync(turn1, cfg())?;
        //   assert_eq!(r1.cached_tokens, 0);
        //
        //   let turn2 = vec![
        //       ChatMessage::user("What is 2+2?"),
        //       ChatMessage::assistant(&r1.text),
        //       ChatMessage::user("And 3+3?"),
        //   ];
        //   let r2 = model.chat_session_start_sync(turn2, cfg())?;
        //   assert!(r2.cached_tokens > 0);
        //   assert!(r2.performance.as_ref().unwrap().prefill_tokens
        //             < turn2_token_count);  // only the delta was prefilled
    }

    /// Divergence miss: two calls with totally different histories. The
    /// result of the second call must report `cached_tokens == 0` and a
    /// full-history prefill.
    #[ignore = "requires a real Qwen3.5 Dense model directory; run with --ignored"]
    #[test]
    fn divergence_miss_resets_and_full_prefills() {
        // Pseudocode:
        //
        //   let r1 = model.chat_session_start_sync(
        //       vec![ChatMessage::user("Hello")],
        //       cfg(),
        //   )?;
        //   let r2 = model.chat_session_start_sync(
        //       vec![ChatMessage::user("Goodbye")],
        //       cfg(),
        //   )?;
        //   assert_eq!(r2.cached_tokens, 0);
        //   // And the second call must have done a full prefill, not
        //   // attempted to decode from stale caches.
    }
}

#[cfg(test)]
mod paged_construction_tests {
    //! Smoke tests for the block-paged adapter construction on Qwen3.5
    //! dense. The forward dispatch lives in `chat_sync_core_paged`
    //! / `chat_stream_sync_core_paged`; these tests cover the
    //! Inner-construction surface in isolation.
    //!
    //! Tests that allocate a `LayerKVPool` require Metal. Construction-only
    //! cases are `#[ignore]`-marked behind `MLX_TEST_PAGED=1`; forward-path
    //! checks are also ignored because no-Metal hosts can abort inside MLX
    //! before Rust receives an `Err`.

    use super::*;
    use crate::array::DType;
    use crate::models::qwen3_5::config::Qwen3_5Config;
    use crate::models::qwen3_5::decoder_layer::{self, AttentionType};

    fn tiny_cfg(use_block_paged: bool) -> Qwen3_5Config {
        Qwen3_5Config {
            vocab_size: 1024,
            hidden_size: 64,
            num_layers: 8,
            num_heads: 4,
            num_kv_heads: 2,
            intermediate_size: 128,
            rms_norm_eps: 1e-6,
            head_dim: 16,
            tie_word_embeddings: true,
            attention_bias: false,
            max_position_embeddings: 1024,
            pad_token_id: 0,
            eos_token_id: 0,
            bos_token_id: 0,
            linear_num_value_heads: 4,
            linear_num_key_heads: 2,
            linear_key_head_dim: 16,
            linear_value_head_dim: 16,
            linear_conv_kernel_dim: 4,
            full_attention_interval: 4,
            partial_rotary_factor: 0.25,
            rope_theta: 100_000.0,
            paged_cache_memory_mb: Some(64),
            paged_block_size: Some(16),
            use_block_paged_cache: if use_block_paged { Some(true) } else { None },
        }
    }

    fn tiny_paged_forward_cfg() -> Qwen3_5Config {
        let mut cfg = tiny_cfg(true);
        // Paged attention's Metal kernels require head_dim=32+; keep the
        // production-forward tests on a separate shape so construction tests
        // preserve their smaller historical config.
        cfg.hidden_size = 128;
        cfg.intermediate_size = 256;
        cfg.head_dim = 32;
        cfg.linear_key_head_dim = 32;
        cfg.linear_value_head_dim = 32;
        cfg.paged_cache_memory_mb = Some(256);
        cfg
    }

    fn paged_inner_or_skip(test_name: &str) -> Option<(Qwen35Inner, Qwen3_5Config)> {
        let cfg = tiny_paged_forward_cfg();
        match Qwen35Inner::new(cfg.clone()) {
            Ok(inner) => Some((inner, cfg)),
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("Metal") || msg.contains("device") || msg.contains("LayerKVPool") {
                    eprintln!("skipping {test_name} (paged adapter unavailable): {msg}");
                    None
                } else {
                    panic!("unexpected Qwen35Inner::new failure in {test_name}: {msg}");
                }
            }
        }
    }

    fn cast_qwen35_inner_weights_bf16(inner: &mut Qwen35Inner) {
        let cast = |a: &MxArray| -> MxArray { a.astype(DType::BFloat16).expect("astype bf16") };

        let w = inner.embedding.get_weight();
        inner.embedding.set_weight(&cast(&w)).expect("set embed");

        let w = inner.final_norm.get_weight();
        inner
            .final_norm
            .set_weight(&cast(&w))
            .expect("set final_norm");

        if let Some(head) = inner.lm_head.as_mut() {
            let w = head.get_weight();
            head.set_weight(&cast(&w)).expect("set lm_head");
        }

        for layer in inner.layers.iter_mut() {
            let w = layer.get_input_layernorm_weight();
            layer
                .set_input_layernorm_weight(&cast(&w))
                .expect("set input_layernorm");
            let w = layer.get_post_attention_layernorm_weight();
            layer
                .set_post_attention_layernorm_weight(&cast(&w))
                .expect("set post_attention_layernorm");

            match &mut layer.attn {
                AttentionType::Linear(gdn) => {
                    let w = gdn.get_dt_bias();
                    gdn.set_dt_bias(&cast(&w));
                    let w = gdn.get_a_log();
                    gdn.set_a_log(&cast(&w)).expect("set a_log");
                    let w = gdn.get_in_proj_qkvz_weight();
                    gdn.set_in_proj_qkvz_weight(&cast(&w))
                        .expect("set in_proj_qkvz");
                    let w = gdn.get_in_proj_ba_weight();
                    gdn.set_in_proj_ba_weight(&cast(&w))
                        .expect("set in_proj_ba");
                    let w = gdn.get_conv1d_weight();
                    gdn.set_conv1d_weight(&cast(&w)).expect("set conv1d");
                    let w = gdn.get_norm_weight();
                    gdn.set_norm_weight(&cast(&w)).expect("set gdn norm");
                    let w = gdn.get_out_proj_weight();
                    gdn.set_out_proj_weight(&cast(&w)).expect("set out_proj");
                }
                AttentionType::Full(attn) => {
                    let w = attn.get_q_proj_weight();
                    attn.set_q_proj_weight(&cast(&w)).expect("set q_proj");
                    let w = attn.get_k_proj_weight();
                    attn.set_k_proj_weight(&cast(&w)).expect("set k_proj");
                    let w = attn.get_v_proj_weight();
                    attn.set_v_proj_weight(&cast(&w)).expect("set v_proj");
                    let w = attn.get_o_proj_weight();
                    attn.set_o_proj_weight(&cast(&w)).expect("set o_proj");
                    let w = attn.get_q_norm_weight();
                    attn.set_q_norm_weight(&cast(&w)).expect("set q_norm");
                    let w = attn.get_k_norm_weight();
                    attn.set_k_norm_weight(&cast(&w)).expect("set k_norm");
                }
            }

            let w = layer.mlp.get_gate_proj_weight();
            layer
                .mlp
                .set_gate_proj_weight(&cast(&w))
                .expect("set gate_proj");
            let w = layer.mlp.get_up_proj_weight();
            layer
                .mlp
                .set_up_proj_weight(&cast(&w))
                .expect("set up_proj");
            let w = layer.mlp.get_down_proj_weight();
            layer
                .mlp
                .set_down_proj_weight(&cast(&w))
                .expect("set down_proj");
        }
    }

    fn reset_paged_request(inner: &mut Qwen35Inner, prompt: &[u32]) {
        inner.caches = Some(
            (0..inner.config.num_layers as usize)
                .map(|i| {
                    if inner.config.is_linear_layer(i) {
                        Qwen3_5LayerCache::new_linear()
                    } else {
                        Qwen3_5LayerCache::new_full_attention()
                    }
                })
                .collect(),
        );

        let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");
        if adapter.block_table().is_some() {
            adapter.release_request().expect("release_request");
        }
        adapter.reset_for_new_request(0).expect("reset request");
        let prefix = adapter
            .find_cached_prefix(prompt, &[], 0, false)
            .expect("find_cached_prefix");
        assert_eq!(
            prefix.cached_token_count, 0,
            "dense chunking tests must start from a cold adapter prefix"
        );
        adapter
            .allocate_suffix_blocks(prompt.len() as u32)
            .expect("allocate suffix blocks");
    }

    fn run_dense_paged_prefill_with_size(
        inner: &mut Qwen35Inner,
        full_tokens: &[u32],
        suffix_tokens: &[u32],
        cached_prefix_len: u32,
        chunk_size: i32,
    ) -> Result<MxArray> {
        let layer_kinds =
            decoder_layer::compute_layer_kinds(inner.config.num_layers as usize, |i| {
                inner.config.is_linear_layer(i)
            });
        let embed = inner.embedding.clone();
        let embedding_weight = embed.get_weight();
        let caches = inner.caches.as_mut().expect("qwen35 caches initialized");
        let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");

        super::super::paged_forward::run_paged_prefill_chunk_with_size(
            full_tokens,
            suffix_tokens,
            cached_prefix_len,
            false,
            &embed,
            &mut inner.layers,
            caches,
            &inner.final_norm,
            &inner.lm_head,
            &embedding_weight,
            &layer_kinds,
            adapter,
            chunk_size,
        )
    }

    fn logits_to_f32_vec(logits: &MxArray) -> Vec<f32> {
        let f32_arr = logits.astype(DType::Float32).expect("astype f32");
        f32_arr.eval();
        let n = f32_arr.shape_at(0).expect("shape_at(0)") as usize;
        (0..n)
            .map(|i| f32_arr.item_at_float32(i).expect("item_at_float32"))
            .collect()
    }

    fn assert_finite_vocab_logits(logits: &MxArray, vocab_size: i32, context: &str) {
        assert_eq!(logits.ndim().expect("ndim"), 1, "{context}: logits ndim");
        assert_eq!(
            logits.shape_at(0).expect("shape_at(0)"),
            vocab_size as i64,
            "{context}: logits shape"
        );
        let values = logits_to_f32_vec(logits);
        for (i, v) in values.iter().enumerate() {
            assert!(v.is_finite(), "{context}: logits[{i}] is not finite: {v}");
        }
    }

    /// `use_block_paged_cache` defaults to `None` and round-trips
    /// through serde.
    #[test]
    fn test_use_block_paged_cache_serde_default_none() {
        let json = serde_json::json!({
            "vocab_size": 1024,
            "hidden_size": 64,
            "num_layers": 8,
            "num_heads": 4,
            "num_kv_heads": 2,
            "intermediate_size": 128,
            "rms_norm_eps": 1e-6,
            "head_dim": 16,
            "tie_word_embeddings": true,
            "max_position_embeddings": 1024,
            "pad_token_id": 0,
            "eos_token_id": 0,
            "bos_token_id": 0,
        });
        let cfg: Qwen3_5Config = serde_json::from_value(json).unwrap();
        assert_eq!(
            cfg.use_block_paged_cache, None,
            "use_block_paged_cache must default to None on JSON without the key"
        );
        assert_eq!(cfg.paged_block_size, None);
        assert_eq!(cfg.paged_cache_memory_mb, None);
    }

    #[test]
    fn test_use_block_paged_cache_serde_true_round_trip() {
        let json = serde_json::json!({
            "vocab_size": 1024,
            "hidden_size": 64,
            "num_layers": 8,
            "num_heads": 4,
            "num_kv_heads": 2,
            "intermediate_size": 128,
            "rms_norm_eps": 1e-6,
            "head_dim": 16,
            "tie_word_embeddings": true,
            "max_position_embeddings": 1024,
            "pad_token_id": 0,
            "eos_token_id": 0,
            "bos_token_id": 0,
            "use_block_paged_cache": true,
            "paged_block_size": 16,
            "paged_cache_memory_mb": 256,
        });
        let cfg: Qwen3_5Config = serde_json::from_value(json).unwrap();
        assert_eq!(cfg.use_block_paged_cache, Some(true));
        assert_eq!(cfg.paged_block_size, Some(16));
        assert_eq!(cfg.paged_cache_memory_mb, Some(256));
    }

    #[test]
    fn test_full_attention_layer_count() {
        let cfg = tiny_cfg(false);
        // 8 layers, full_attention_interval=4 → layers 3 and 7 are
        // full-attention (2 layers).
        assert_eq!(cfg.full_attention_layer_count(), 2);
    }

    /// When `use_block_paged_cache` is `None`, `paged_adapter` is None.
    #[test]
    fn test_inner_no_paged_adapter_when_flag_is_none() {
        let cfg = tiny_cfg(false);
        let inner =
            Qwen35Inner::new(cfg).expect("Qwen35Inner::new must succeed without paged adapter");
        assert!(
            inner.paged_adapter.is_none(),
            "paged_adapter must be None when use_block_paged_cache is None"
        );
    }

    #[test]
    fn test_fresh_dense_layer_caches_are_not_gdn_reuse_ready() {
        let cfg = tiny_cfg(true);
        let caches = fresh_dense_layer_caches(&cfg);
        assert_eq!(caches.len(), cfg.num_layers as usize);
        assert!(
            !dense_paged_linear_caches_ready(&cfg, Some(&caches)),
            "fresh linear caches have empty conv/recurrent slots, so a live continuation must replay GDN"
        );
        assert!(matches!(caches[0], Qwen3_5LayerCache::Linear(_)));
        assert!(matches!(caches[3], Qwen3_5LayerCache::FullAttention(_)));
    }

    #[test]
    fn test_paged_dense_linear_cache_export_uninitialized_returns_none() {
        unsafe {
            mlx_sys::mlx_qwen35_compiled_reset();
        }
        let cfg = tiny_cfg(true);
        let exported = export_paged_dense_linear_caches(&cfg)
            .expect("uninitialized paged dense export should not fail");
        assert!(exported.is_none());
    }

    #[test]
    fn test_dense_paged_prefix_block_hash_matches_allocator_chain() {
        let tokens: Vec<u32> = (1..=12).collect();
        let per_block = vec![vec![11], vec![], vec![33, 44]];

        let h0 = mlx_paged_attn::hash_tokens(&tokens[0..4], 0, &per_block[0]);
        let h1 = mlx_paged_attn::hash_tokens(&tokens[4..8], h0, &per_block[1]);
        let h2 = mlx_paged_attn::hash_tokens(&tokens[8..12], h1, &per_block[2]);

        assert_eq!(
            compute_paged_prefix_block_hash(&tokens, 12, 4, &per_block, 0),
            Some(h2)
        );
    }

    #[test]
    fn test_dense_paged_prefix_block_hash_applies_salt_to_first_block_only() {
        let tokens: Vec<u32> = (1..=8).collect();
        let per_block = vec![vec![11], vec![22]];
        let salt = 99;

        let mut first_block_keys = per_block[0].clone();
        first_block_keys.push(salt);
        let h0 = mlx_paged_attn::hash_tokens(&tokens[0..4], 0, &first_block_keys);
        let h1 = mlx_paged_attn::hash_tokens(&tokens[4..8], h0, &per_block[1]);

        assert_eq!(
            compute_paged_prefix_block_hash(&tokens, 8, 4, &per_block, salt),
            Some(h1)
        );
    }

    #[test]
    fn test_dense_paged_prefix_block_hash_rejects_non_full_or_unkeyed_prefix() {
        let tokens: Vec<u32> = (1..=8).collect();
        let per_block = vec![vec![]];

        assert_eq!(
            compute_paged_prefix_block_hash(&tokens, 6, 4, &per_block, 0),
            None
        );
        assert_eq!(
            compute_paged_prefix_block_hash(&tokens, 8, 4, &per_block, 0),
            None
        );
    }

    /// Allocates a `LayerKVPool`. Requires Metal; gate on
    /// `MLX_TEST_PAGED=1`.
    #[test]
    #[ignore = "Allocates Metal LayerKVPool; gate on MLX_TEST_PAGED=1"]
    fn test_inner_constructs_paged_adapter_when_flag_is_true() {
        if std::env::var_os("MLX_TEST_PAGED").is_none() {
            return;
        }
        let cfg = tiny_cfg(true);
        let inner = Qwen35Inner::new(cfg).expect(
            "Qwen35Inner::new with use_block_paged_cache=true must succeed on Metal-capable host",
        );
        assert!(
            inner.paged_adapter.is_some(),
            "paged_adapter must be Some when use_block_paged_cache = Some(true)"
        );
    }

    /// VLM checkpoints are accepted under paged dispatch: the vision
    /// encoder loads even with `paged_adapter` on. Text-only chat
    /// entry points reject image-bearing turns at runtime; this test
    /// verifies only the load-time wiring.
    #[test]
    #[ignore = "Allocates Metal LayerKVPool; gate on MLX_TEST_PAGED=1"]
    fn test_vlm_loads_when_paged_enabled() {
        if std::env::var_os("MLX_TEST_PAGED").is_none() {
            return;
        }
        use crate::models::qwen3_5::vision::Qwen3_5VisionConfig;
        use crate::models::qwen3_5::vision::Qwen3_5VisionEncoder;

        let cfg = tiny_cfg(true);
        let mut inner = Qwen35Inner::new(cfg).unwrap();
        let vision_cfg = Qwen3_5VisionConfig {
            hidden_size: 64,
            intermediate_size: 256,
            num_heads: 4,
            num_layers: 2,
            patch_size: 16,
            spatial_merge_size: 2,
            image_size: 256,
            out_hidden_size: 64,
        };
        let vision_enc =
            Qwen3_5VisionEncoder::new(vision_cfg).expect("vision encoder construction");
        let result = inner.set_vision_encoder(vision_enc);
        assert!(
            result.is_ok(),
            "set_vision_encoder must succeed when paged_adapter is Some so VLM \
             checkpoints can run text-only paged inference; got {result:?}"
        );
        assert!(
            inner.vision_encoder.is_some(),
            "vision_encoder field must be populated after a successful set"
        );
    }

    /// Phase 5 piece 1: hard-fails fast when caller passes a `caches`
    /// slice whose length disagrees with `config.num_layers`. The
    /// check runs BEFORE any FFI dispatch so we don't perturb the C++
    /// paged globals — making this the only branch of
    /// `init_paged_dense_compiled_session` we can exercise from a
    /// non-Metal sandbox.
    ///
    /// (The other failure branches all require populated `MxArray`
    /// handles, which need a real Metal allocation. They're covered
    /// indirectly by parity testing on real-weights checkpoints.)
    #[test]
    fn test_init_paged_dense_compiled_session_rejects_cache_length_mismatch() {
        let cfg = tiny_cfg(true);
        // `cfg` has num_layers=8; pass an empty cache slice to trigger
        // the length check before any FFI / Metal call.
        let empty_caches: Vec<Qwen3_5LayerCache> = Vec::new();
        let alloc = std::sync::Arc::new(std::sync::Mutex::new(
            mlx_paged_attn::BlockAllocator::new(2, 16),
        ));
        let pa_cfg = mlx_paged_attn::PagedAttentionConfig {
            block_size: 16,
            gpu_memory_mb: 8,
            head_size: cfg.head_dim as u32,
            num_kv_heads: cfg.num_kv_heads as u32,
            num_layers: cfg.full_attention_layer_count() as u32,
            use_fp8_cache: Some(false),
            max_seq_len: Some(64),
            max_batch_size: Some(1),
        };
        let pool = match mlx_paged_attn::LayerKVPool::new(
            pa_cfg,
            2,
            mlx_paged_attn::metal::MetalDtype::BFloat16,
        ) {
            Ok(p) => std::sync::Arc::new(p),
            Err(e) => {
                eprintln!(
                    "skipping test_init_paged_dense_compiled_session_rejects_cache_length_mismatch: {e}"
                );
                return;
            }
        };
        let adapter = PagedKVCacheAdapter::new(alloc, pool, 16)
            .expect("paged adapter construction must succeed once pool is built");
        let res = init_paged_dense_compiled_session(&cfg, &empty_caches, &adapter, 0);
        let msg = res
            .expect_err("expected Err on cache length mismatch")
            .to_string();
        assert!(
            msg.contains("caches.len()"),
            "error must reference caches length contract; got: {msg}"
        );
    }

    /// Phase 5 piece 1: the C++ compiled paged graph hard-codes
    /// block_size=16. The dispatcher MUST gate `cpp_session_ready` on
    /// `adapter.block_size() == 16` so a config with
    /// `paged_block_size: Some(8)` falls back to the pure-Rust paged
    /// path instead of corrupting KV state. Validates the constant.
    #[test]
    fn test_cpp_paged_required_block_size_is_sixteen() {
        assert_eq!(
            CPP_PAGED_REQUIRED_BLOCK_SIZE, 16,
            "C++ compiled paged graph in mlx_qwen35.cpp hard-codes block_size=16; \
             changing this constant requires re-tracing the compiled graph"
        );
    }

    /// Phase 5 piece 1: build an adapter with `block_size != 16` and
    /// verify that the dispatcher gate
    /// (`adapter.block_size() != CPP_PAGED_REQUIRED_BLOCK_SIZE`) would
    /// correctly reject it. The gate runs BEFORE
    /// `init_paged_dense_compiled_session` is called, so the FFI is
    /// never touched on the fallback path.
    #[test]
    fn test_block_size_eight_adapter_falls_back_to_pure_rust() {
        let cfg = tiny_cfg(true);
        let alloc = std::sync::Arc::new(std::sync::Mutex::new(
            mlx_paged_attn::BlockAllocator::new(2, 8),
        ));
        let pa_cfg = mlx_paged_attn::PagedAttentionConfig {
            block_size: 8,
            gpu_memory_mb: 8,
            head_size: cfg.head_dim as u32,
            num_kv_heads: cfg.num_kv_heads as u32,
            num_layers: cfg.full_attention_layer_count() as u32,
            use_fp8_cache: Some(false),
            max_seq_len: Some(64),
            max_batch_size: Some(1),
        };
        let pool = match mlx_paged_attn::LayerKVPool::new_for_test(
            pa_cfg,
            2,
            cfg.full_attention_layer_count() as u32,
            mlx_paged_attn::metal::MetalDtype::BFloat16,
        ) {
            Ok(p) => std::sync::Arc::new(p),
            Err(e) => {
                eprintln!("skipping test_block_size_eight_adapter_falls_back_to_pure_rust: {e}");
                return;
            }
        };
        let adapter = PagedKVCacheAdapter::new(alloc, pool, 8)
            .expect("PagedKVCacheAdapter::new must accept block_size=8 (validated by adapter)");
        assert_eq!(
            adapter.block_size(),
            8,
            "adapter must report the block_size it was constructed with"
        );
        // Simulate the dispatcher's gate. We do NOT call
        // `init_paged_dense_compiled_session` here — that's exactly
        // what the gate prevents.
        assert_ne!(
            adapter.block_size(),
            CPP_PAGED_REQUIRED_BLOCK_SIZE,
            "block_size=8 adapter must not match the compiled graph's hard-coded 16; \
             the dispatcher gate at chat_sync_core_paged_inner / chat_stream_sync_core_paged_inner \
             relies on this inequality to fall back to the pure-Rust paged path"
        );
    }

    /// Dense Qwen3.5 paged-prefill chunking state test. This drives the
    /// production chunk-size worker once and asserts the adapter cursor,
    /// request token log, and block table cover the whole prompt after all
    /// chunks have been recorded.
    #[test]
    #[ignore = "requires Metal GPU; run with --ignored"]
    fn test_dense_paged_prefill_chunks_advance_adapter_state() {
        let Some((mut inner, cfg)) =
            paged_inner_or_skip("test_dense_paged_prefill_chunks_advance_adapter_state")
        else {
            return;
        };
        cast_qwen35_inner_weights_bf16(&mut inner);

        let prompt: Vec<u32> = (0u32..64).map(|i| (i * 5 + 7) % 257).collect();
        reset_paged_request(&mut inner, &prompt);

        let logits = match run_dense_paged_prefill_with_size(
            &mut inner, &prompt, &prompt, 0, /* chunk_size */ 16,
        ) {
            Ok(logits) => logits,
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("Metal GPU not available") || msg.contains("No Metal device") {
                    eprintln!(
                        "skipping test_dense_paged_prefill_chunks_advance_adapter_state: {msg}"
                    );
                    return;
                }
                panic!("unexpected dense paged chunk failure: {msg}");
            }
        };

        let adapter = inner.paged_adapter.as_ref().expect("paged_adapter");
        assert_eq!(
            adapter.current_token_count() as usize,
            prompt.len(),
            "adapter cursor after chunked prefill"
        );
        assert_eq!(
            adapter.request_tokens(),
            prompt.as_slice(),
            "request token log after chunked prefill"
        );
        let block_table = adapter.block_table().expect("block_table");
        let expected_min_blocks = prompt.len().div_ceil(adapter.block_size() as usize);
        assert!(
            block_table.num_blocks() >= expected_min_blocks,
            "block table has {} blocks, expected at least {expected_min_blocks}",
            block_table.num_blocks()
        );
        assert_finite_vocab_logits(&logits, cfg.vocab_size, "final dense paged chunk prefill");

        let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");
        let _ = adapter.register_full_blocks_for_reuse(&[], 0);
        adapter.release_request().expect("release_request");
    }

    /// Uneven-tail coverage for dense Qwen3.5 paged prefill: a 33-token
    /// prompt with chunk_size=16 must record two full chunks plus a
    /// one-token tail and return valid logits for the tail chunk.
    #[test]
    #[ignore = "requires Metal GPU; run with --ignored"]
    fn test_dense_paged_prefill_chunks_handle_uneven_tail() {
        let Some((mut inner, cfg)) =
            paged_inner_or_skip("test_dense_paged_prefill_chunks_handle_uneven_tail")
        else {
            return;
        };
        cast_qwen35_inner_weights_bf16(&mut inner);

        let prompt: Vec<u32> = (0u32..33).map(|i| (i * 11 + 3) % 257).collect();
        reset_paged_request(&mut inner, &prompt);

        let final_logits = run_dense_paged_prefill_with_size(
            &mut inner, &prompt, &prompt, 0, /* chunk_size */ 16,
        )
        .expect("dense paged uneven-tail chunked prefill");

        assert_eq!(
            prompt.len(),
            33,
            "test setup must exercise a one-token tail"
        );
        let adapter = inner.paged_adapter.as_ref().expect("paged_adapter");
        assert_eq!(
            adapter.current_token_count(),
            33,
            "adapter cursor must include the uneven tail"
        );
        assert_eq!(
            adapter.request_tokens(),
            prompt.as_slice(),
            "request token log must include the uneven tail"
        );
        assert_finite_vocab_logits(
            &final_logits,
            cfg.vocab_size,
            "uneven-tail dense paged chunk prefill",
        );

        let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");
        let _ = adapter.register_full_blocks_for_reuse(&[], 0);
        adapter.release_request().expect("release_request");
    }

    /// Compatibility guard for the current/default dense Qwen3.5 paged
    /// prefill behavior: a full suffix passed in one call remains a valid
    /// single-shot prefill and is stable across a fresh adapter reset.
    #[test]
    #[ignore = "requires Metal GPU; run with --ignored"]
    fn test_dense_paged_prefill_single_shot_default_still_works() {
        let Some((mut inner, cfg)) =
            paged_inner_or_skip("test_dense_paged_prefill_single_shot_default_still_works")
        else {
            return;
        };
        cast_qwen35_inner_weights_bf16(&mut inner);

        let prompt: Vec<u32> = vec![5, 11, 21, 33, 47, 60, 71, 83];

        reset_paged_request(&mut inner, &prompt);
        let logits_a = run_dense_paged_prefill_with_size(
            &mut inner, &prompt, &prompt, 0, /* chunk_size */ 0,
        )
        .expect("single-shot A");
        assert_finite_vocab_logits(&logits_a, cfg.vocab_size, "single-shot A");
        {
            let adapter = inner.paged_adapter.as_ref().expect("paged_adapter");
            assert_eq!(
                adapter.current_token_count() as usize,
                prompt.len(),
                "single-shot cursor"
            );
            assert_eq!(adapter.request_tokens(), prompt.as_slice());
        }

        reset_paged_request(&mut inner, &prompt);
        let logits_b = run_dense_paged_prefill_with_size(
            &mut inner, &prompt, &prompt, 0, /* chunk_size */ 0,
        )
        .expect("single-shot B");
        let a = logits_to_f32_vec(&logits_a);
        let b = logits_to_f32_vec(&logits_b);
        assert_eq!(a.len(), cfg.vocab_size as usize);
        assert_eq!(b.len(), cfg.vocab_size as usize);
        for (i, (left, right)) in a.iter().zip(b.iter()).enumerate() {
            let abs_diff = (left - right).abs();
            assert!(
                abs_diff <= 1e-6,
                "single-shot dense paged prefill changed after fresh reset at index {i}: \
                 first={left}, second={right}, abs_diff={abs_diff}"
            );
        }

        let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");
        let _ = adapter.register_full_blocks_for_reuse(&[], 0);
        adapter.release_request().expect("release_request");
    }

    /// Phase 5 piece 1: the C++ FFI returns `int32_t` (0 success / -1
    /// failure). The Rust `init_paged_dense_compiled_session`
    /// propagates non-zero status as `Err` so the dispatcher's
    /// `cpp_session_ready` becomes false on init failure and falls
    /// back to the pure-Rust paged decode. This test forces the C++
    /// side's null-handle rejection branch by passing null pool
    /// handles for the full-attention layers, and asserts the FFI
    /// returns -1.
    #[test]
    fn test_mlx_qwen35_init_paged_returns_negative_on_null_handles() {
        let cfg = tiny_cfg(true);
        let num_layers = cfg.num_layers as usize;
        // All-null pool/scale handles. The C++ side iterates layers
        // [0..num_layers); for each non-linear (full-attention) layer
        // the null-handle check fires and the function returns -1.
        // Layer 0 with full_attention_interval=4 is linear, so the
        // first full-attn layer is at index 3, where the rejection
        // fires.
        let mut k_pool: Vec<*mut mlx_sys::mlx_array> = vec![std::ptr::null_mut(); num_layers];
        let mut v_pool: Vec<*mut mlx_sys::mlx_array> = vec![std::ptr::null_mut(); num_layers];
        let mut k_scale: Vec<*mut mlx_sys::mlx_array> = vec![std::ptr::null_mut(); num_layers];
        let mut v_scale: Vec<*mut mlx_sys::mlx_array> = vec![std::ptr::null_mut(); num_layers];
        let mut linear_caches: Vec<*mut mlx_sys::mlx_array> =
            vec![std::ptr::null_mut(); num_layers * 2];

        let status = unsafe {
            mlx_sys::mlx_qwen35_init_paged(
                cfg.num_layers,
                cfg.hidden_size,
                cfg.num_heads,
                cfg.num_kv_heads,
                cfg.head_dim,
                cfg.rope_theta as f32,
                cfg.rope_dims(),
                cfg.rms_norm_eps as f32,
                cfg.full_attention_interval,
                cfg.linear_num_key_heads,
                cfg.linear_num_value_heads,
                cfg.linear_key_head_dim,
                cfg.linear_value_head_dim,
                cfg.linear_conv_kernel_dim,
                if cfg.tie_word_embeddings { 1 } else { 0 },
                cfg.max_position_embeddings,
                1,
                k_pool.as_mut_ptr(),
                v_pool.as_mut_ptr(),
                k_scale.as_mut_ptr(),
                v_scale.as_mut_ptr(),
                linear_caches.as_mut_ptr(),
                0,
            )
        };
        assert_eq!(
            status, -1,
            "mlx_qwen35_init_paged MUST return -1 when full-attention pool/scale handles \
             are null. Returning 0 would let the dispatcher enter the compiled paged decode \
             against uninitialized globals."
        );
        // Reset C++ globals to a clean state so the test doesn't
        // contaminate any later tests in the same process.
        unsafe {
            mlx_sys::mlx_qwen35_compiled_reset();
        }
    }
}
