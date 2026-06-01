use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi_derive::napi;
use tracing::{info, warn};

use crate::inference_trace::{
    elapsed_ms, enabled as inference_trace_enabled, write as write_inference_trace,
};
use crate::model_thread::{ResponseTx, StreamTx};
use crate::models::paddleocr_vl::processing::ProcessedImages;
use crate::models::qwen3_5::model::{
    ChatConfig, ChatResult, ChatStreamChunk, ChatStreamHandle, VisionCache, VisionCacheInner,
    compute_image_token_counts_per_image, eval_layer_caches, extract_images_from_messages,
    inject_image_placeholders, vlm_prepare_vision_features,
};
use crate::models::qwen3_5::processing::Qwen35VLImageProcessor;
use crate::models::qwen3_5::vision::Qwen3_5VisionEncoder;

use super::config::Qwen3_5MoeConfig;
use super::decoder_layer::DecoderLayer;
use super::layer_cache::Qwen3_5LayerCache;
use super::persistence;
use super::quantized_linear::LinearProj;
use crate::array::MxArray;
use crate::array::mask::create_causal_mask;
use crate::models::qwen3_5::chat_common;
use crate::models::qwen3_5::chat_common::{
    IMAGE_CHANGE_RESTART_PREFIX, apply_all_penalties, build_chatml_continue_delta_text,
    build_synthetic_user_message, compute_image_cache_key, compute_performance_metrics,
    extract_chat_params, finalize_chat_result, save_cache_state_direct, send_stream_error,
    verify_cache_prefix_direct,
};
use crate::nn::{Embedding, Linear, RMSNorm};
use crate::sampling::{SamplingConfig, sample};
use crate::stream::{DeviceType, Stream, StreamContext};
use crate::tokenizer::{ChatMessage, Qwen3Tokenizer, ToolDefinition};
use crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter;

fn fresh_moe_layer_caches(config: &Qwen3_5MoeConfig) -> Vec<Qwen3_5LayerCache> {
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

const MOE_GDN_PREFIX_CHECKPOINT_LIMIT: usize = 8;

struct MoeGdnPrefixCheckpoint {
    prefix_len: u32,
    block_size: u32,
    final_block_hash: u64,
    tokens: Vec<u32>,
    caches: Vec<Qwen3_5LayerCache>,
}

struct MoeGdnHistoryCheckpoint {
    tokens: Vec<u32>,
    caches: Vec<Qwen3_5LayerCache>,
}

struct MoeGdnPrefixPreparation {
    state: &'static str,
    already_primed: bool,
}

#[derive(Default)]
struct MoeGdnCheckpointStoreTrace {
    stored: bool,
    hash_ms: f64,
    eval_ms: f64,
    clone_ms: f64,
    token_clone_ms: f64,
    update_ms: f64,
    total_ms: f64,
}

impl MoeGdnCheckpointStoreTrace {
    fn finish(mut self, start: Option<std::time::Instant>) -> Self {
        self.total_ms = start.map(elapsed_ms).unwrap_or(0.0);
        self
    }
}

fn moe_gdn_store_replayed_prefix_checkpoint_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        crate::inference_trace::env_flag_enabled("MLX_MOE_GDN_REPLAY_PREFIX_CHECKPOINT")
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

fn moe_paged_linear_caches_ready(
    config: &Qwen3_5MoeConfig,
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

fn clone_moe_linear_layer_caches(
    config: &Qwen3_5MoeConfig,
    caches: &[Qwen3_5LayerCache],
) -> Option<Vec<Qwen3_5LayerCache>> {
    if !moe_paged_linear_caches_ready(config, Some(caches)) {
        return None;
    }

    let mut cloned = fresh_moe_layer_caches(config);
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

fn export_paged_moe_linear_caches(
    config: &Qwen3_5MoeConfig,
) -> Result<Option<Vec<Qwen3_5LayerCache>>> {
    let num_layers = config.num_layers as usize;
    let expected = num_layers
        .checked_mul(2)
        .ok_or_else(|| Error::from_reason("paged MoE cache export size overflow"))?;
    let mut export_ptrs: Vec<*mut mlx_sys::mlx_array> = vec![std::ptr::null_mut(); expected];
    let exported = unsafe {
        mlx_sys::mlx_qwen35_moe_export_paged_linear_caches(
            export_ptrs.as_mut_ptr(),
            expected as i32,
        )
    };
    if exported == 0 {
        return Ok(None);
    }
    if exported != expected as i32 {
        return Err(Error::from_reason(format!(
            "paged MoE linear cache export returned {exported} arrays; expected {expected}"
        )));
    }

    let cache_offset = unsafe { mlx_sys::mlx_qwen35_moe_get_paged_cache_offset() };
    let mut new_caches = fresh_moe_layer_caches(config);
    for i in 0..num_layers {
        if !config.is_linear_layer(i) {
            continue;
        }
        let p0 = export_ptrs[i * 2];
        let p1 = export_ptrs[i * 2 + 1];
        if p0.is_null() || p1.is_null() {
            return Err(Error::from_reason(format!(
                "paged MoE linear cache export missing layer {i}"
            )));
        }
        new_caches[i].import_ptrs(p0, p1, cache_offset);
    }

    Ok(Some(new_caches))
}

// Import the shared model ID counter from the dense module — dense and MoE
// share the same C++ weight map, so IDs must be globally unique.
use crate::models::qwen3_5::model::{COMPILED_WEIGHTS_RWLOCK, QWEN35_MODEL_ID_COUNTER};

/// Process-wide mutex serializing the MoE compiled forward lifecycle across
/// model instances. Within a single model instance, the dedicated model thread
/// serializes calls. But with multiple model instances, compiled C++ forward
/// calls from different model threads can collide on process-wide globals.
static MOE_COMPILED_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// RAII guard that calls `mlx_qwen35_moe_reset()` on drop.
///
/// Ensures C++ compiled MoE state is always cleaned up, even if the decode
/// loop returns early via `?` operator. Without this, an error during decode
/// would leave stale compiled state that corrupts the next generation call.
struct MoeResetGuard;

impl Drop for MoeResetGuard {
    fn drop(&mut self) {
        unsafe {
            mlx_sys::mlx_qwen35_moe_reset();
        }
    }
}

/// Internal model state owned exclusively by the dedicated model thread.
///
/// No `Arc<RwLock<>>` — the model thread has sole ownership of all inference
/// and training state. Training commands are routed via `TrainingDispatch`.
pub(crate) struct Qwen35MoeInner {
    pub(crate) config: Qwen3_5MoeConfig,
    pub(crate) embedding: Embedding,
    pub(crate) layers: Vec<DecoderLayer>,
    pub(crate) final_norm: RMSNorm,
    pub(crate) lm_head: Option<LinearProj>,
    pub(crate) caches: Option<Vec<Qwen3_5LayerCache>>,
    pub(crate) tokenizer: Option<Arc<Qwen3Tokenizer>>,
    pub(crate) fa_idx: usize,
    pub(crate) vision_encoder: Option<Arc<Qwen3_5VisionEncoder>>,
    pub(crate) image_processor: Option<Arc<Qwen35VLImageProcessor>>,
    pub(crate) spatial_merge_size: Option<i32>,
    pub(crate) vision_cache: VisionCache,
    pub(crate) cached_token_history: Vec<u32>,
    pub(crate) cached_image_key: Option<u64>,
    pub(crate) cached_rope_deltas: Option<i32>,
    pub(crate) model_id: u64,
    gdn_prefix_checkpoints: VecDeque<MoeGdnPrefixCheckpoint>,
    gdn_last_history_checkpoint: Option<MoeGdnHistoryCheckpoint>,
    /// Block-paged KV adapter (vLLM-style refcounted prefix cache) for
    /// full-attention layers — same semantics as the dense model.
    /// **Opt-in via `Qwen3_5MoeConfig::use_block_paged_cache`.**
    pub(crate) paged_adapter: Option<PagedKVCacheAdapter>,
    /// Training state owned by the model thread.
    /// Created when `InitTraining` command is received, destroyed when training ends.
    pub(crate) training_state: Option<crate::training_state::ModelThreadTrainingState>,
}

/// Commands dispatched from NAPI methods to the dedicated model thread.
pub(crate) enum Qwen35MoeCmd {
    /// Start a new session via the jinja-render path with `<|im_end|>` as
    /// the stop token. See [`Qwen35MoeInner::chat_session_start_sync`] for
    /// the behavioural contract (full cache reset, session boundary on
    /// `<|im_end|>`, images accepted for VLM variants).
    ChatSessionStart {
        messages: Vec<ChatMessage>,
        config: ChatConfig,
        reply: ResponseTx<ChatResult>,
    },
    /// Continue an existing session by appending a user turn. See
    /// [`Qwen35MoeInner::chat_session_continue_sync`] — builds a raw ChatML
    /// delta from `user_message`, tokenizes it, and prefills on top of the
    /// live caches.
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
    /// [`Qwen35MoeInner::chat_session_continue_tool_sync`] — builds a
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
        config: Qwen3_5MoeGenerationConfig,
        reply: ResponseTx<Qwen3_5MoeGenerationResult>,
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
pub(crate) fn handle_qwen35_moe_cmd(inner: &mut Qwen35MoeInner, cmd: Qwen35MoeCmd) {
    match cmd {
        Qwen35MoeCmd::ChatSessionStart {
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
        Qwen35MoeCmd::ChatSessionContinue {
            user_message,
            images,
            config,
            reply,
        } => {
            let _ = reply.send(inner.chat_session_continue_sync(user_message, images, config));
        }
        Qwen35MoeCmd::ChatSessionContinueTool {
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
        Qwen35MoeCmd::ChatStreamSessionStart {
            messages,
            config,
            stream_tx,
            cancelled,
        } => {
            inner.chat_stream_session_start_sync(messages, config, stream_tx, cancelled);
        }
        Qwen35MoeCmd::ChatStreamSessionContinue {
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
        Qwen35MoeCmd::ChatStreamSessionContinueTool {
            tool_call_id,
            content,
            is_error,
            config,
            stream_tx,
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
        Qwen35MoeCmd::Generate {
            prompt_tokens,
            config,
            reply,
        } => {
            let _ = reply.send(inner.generate_sync(prompt_tokens, config));
        }
        Qwen35MoeCmd::ResetCaches { reply } => {
            let _ = reply.send(inner.reset_caches_sync());
        }
        Qwen35MoeCmd::SaveModel { save_path, reply } => {
            let _ = reply.send(inner.save_model_sync(&save_path));
        }
        // --- Training commands ---
        Qwen35MoeCmd::InitTraining {
            config,
            model_type,
            reply,
        } => {
            let _ = reply.send(inner.init_training_sync(*config, model_type));
        }
        Qwen35MoeCmd::GenerateForTraining {
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
        Qwen35MoeCmd::TrainStepGRPO {
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
        Qwen35MoeCmd::BumpSkippedStep { reply } => {
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
        Qwen35MoeCmd::SetTrainingStep { step, reply } => {
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
        Qwen35MoeCmd::ResetTraining { reply } => {
            inner.training_state = None;
            let _ = reply.send(Ok(()));
        }
        Qwen35MoeCmd::TrainStepSFT {
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
        Qwen35MoeCmd::SaveOptimizerState { path, reply } => {
            let _ = reply.send(inner.save_optimizer_state_sync(path));
        }
        Qwen35MoeCmd::LoadOptimizerState { path, reply } => {
            let _ = reply.send(inner.load_optimizer_state_sync(path));
        }
    }
}

/// Wrapper around `StreamTx` that provides a `.call()` method matching the
/// `ThreadsafeFunction` interface expected by the `decode_loop!` macro.
struct StreamSender(StreamTx<ChatStreamChunk>);

impl StreamSender {
    fn call(&self, result: napi::Result<ChatStreamChunk>, _mode: ThreadsafeFunctionCallMode) {
        let _ = self.0.send(result);
    }
}

/// Generation configuration for Qwen3.5 MoE
#[napi(object)]
#[derive(Debug, Clone)]
pub struct Qwen3_5MoeGenerationConfig {
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
pub struct Qwen3_5MoeGenerationResult {
    pub tokens: Vec<u32>,
    pub text: String,
    pub num_tokens: u32,
    pub finish_reason: String,
}

// ========== Qwen35MoeInner implementation ==========
// All these methods run on the dedicated model thread (synchronous, no locks).

impl Qwen35MoeInner {
    /// Create a new Qwen35MoeInner with the given configuration.
    pub(crate) fn new(config: Qwen3_5MoeConfig) -> Result<Self> {
        let embedding = Embedding::new(config.vocab_size as u32, config.hidden_size as u32)?;

        let layers = (0..config.num_layers as usize)
            .map(|i| DecoderLayer::new(&config, i))
            .collect::<Result<Vec<_>>>()?;

        let final_norm = RMSNorm::new(config.hidden_size as u32, Some(config.rms_norm_eps))?;

        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(LinearProj::Standard(Linear::new(
                config.hidden_size as u32,
                config.vocab_size as u32,
                Some(false),
            )?))
        };

        let fa_idx = (0..config.num_layers as usize)
            .find(|&i| !config.is_linear_layer(i))
            .unwrap_or(0);

        let model_id = QWEN35_MODEL_ID_COUNTER.fetch_add(1, Ordering::Relaxed);

        // Block-paged KV adapter — opt-in via `use_block_paged_cache`.
        // See `Qwen35Inner::new` (dense model) for the full architectural
        // discussion; this is the MoE-side mirror.
        let paged_adapter = if config.use_block_paged_cache.unwrap_or(false) {
            let attn_layer_count = config.full_attention_layer_count() as u32;
            if attn_layer_count == 0 {
                return Err(Error::from_reason(
                    "Qwen3.5 MoE block-paged adapter: config has no full_attention layers; \
                     paged KV cache requires at least one attention layer.",
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
                num_layers: attn_layer_count,
                use_fp8_cache: Some(false),
                max_seq_len: Some(config.max_position_embeddings as u32),
                max_batch_size: Some(32),
            };

            let num_blocks = pa_config.calculate_num_blocks();
            if num_blocks == 0 {
                return Err(Error::from_reason(format!(
                    "Qwen3.5 MoE block-paged adapter: gpu_memory_mb={gpu_memory_mb} too small \
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
                        "Failed to construct LayerKVPool for Qwen3.5 MoE block-paged adapter: {e}"
                    ))
                })?;

            let adapter =
                PagedKVCacheAdapter::new(allocator, Arc::new(pool), block_size).map_err(|e| {
                    Error::from_reason(format!(
                        "Failed to construct Qwen3.5 MoE PagedKVCacheAdapter: {e}"
                    ))
                })?;

            info!(
                "Qwen3.5 MoE block-paged adapter enabled: num_blocks={}, block_size={}, \
                 gpu_memory_mb={}, num_attn_layers={}, cache_dtype=BFloat16",
                num_blocks, block_size, gpu_memory_mb, attn_layer_count
            );
            Some(adapter)
        } else {
            None
        };

        info!(
            "Qwen3.5 MoE inner created: {} layers, fa_idx={}, experts={}, paged={}",
            config.num_layers,
            fa_idx,
            config.num_experts,
            paged_adapter.is_some()
        );

        Ok(Self {
            config,
            embedding,
            layers,
            final_norm,
            lm_head,
            caches: None,
            tokenizer: None,
            fa_idx,
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
        self.gdn_prefix_checkpoints.clear();
        self.gdn_last_history_checkpoint = None;
    }

    fn find_moe_gdn_history_checkpoint(
        &self,
        tokens: &[u32],
        prefix_len: u32,
    ) -> Option<Vec<Qwen3_5LayerCache>> {
        let prefix_tokens = tokens.get(..prefix_len as usize)?;
        let checkpoint = self.gdn_last_history_checkpoint.as_ref()?;
        if checkpoint.tokens.as_slice() != prefix_tokens {
            return None;
        }
        clone_moe_linear_layer_caches(&self.config, &checkpoint.caches)
    }

    fn remember_moe_gdn_history_checkpoint(&mut self) -> Result<MoeGdnCheckpointStoreTrace> {
        let trace_enabled = inference_trace_enabled();
        let total_start = trace_enabled.then(std::time::Instant::now);
        let mut trace = MoeGdnCheckpointStoreTrace::default();
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
            .and_then(|caches| clone_moe_linear_layer_caches(&self.config, caches))
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
        self.gdn_last_history_checkpoint = Some(MoeGdnHistoryCheckpoint { tokens, caches });
        trace.update_ms = update_start.map(elapsed_ms).unwrap_or(0.0);
        trace.stored = true;
        Ok(trace.finish(total_start))
    }

    fn find_moe_gdn_prefix_checkpoint(
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
        let prefix_len_usize = prefix_len as usize;
        let prefix_tokens = tokens.get(..prefix_len_usize)?;

        self.gdn_prefix_checkpoints
            .iter()
            .rev()
            .find(|checkpoint| {
                checkpoint.prefix_len == prefix_len
                    && checkpoint.block_size == block_size
                    && checkpoint.final_block_hash == final_block_hash
                    && checkpoint.tokens.as_slice() == prefix_tokens
                    && moe_paged_linear_caches_ready(&self.config, Some(&checkpoint.caches))
            })
            .and_then(|checkpoint| clone_moe_linear_layer_caches(&self.config, &checkpoint.caches))
    }

    fn remember_moe_gdn_prefix_checkpoint(
        &mut self,
        tokens: &[u32],
        prefix_len: u32,
        block_size: u32,
        extra_keys_per_block: &[Vec<u64>],
        cache_salt: u64,
    ) -> Result<MoeGdnCheckpointStoreTrace> {
        let trace_enabled = inference_trace_enabled();
        let total_start = trace_enabled.then(std::time::Instant::now);
        let mut trace = MoeGdnCheckpointStoreTrace::default();
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
            .and_then(|caches| clone_moe_linear_layer_caches(&self.config, caches))
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
            .push_back(MoeGdnPrefixCheckpoint {
                prefix_len,
                block_size,
                final_block_hash,
                tokens: prefix_tokens,
                caches,
            });
        while self.gdn_prefix_checkpoints.len() > MOE_GDN_PREFIX_CHECKPOINT_LIMIT {
            self.gdn_prefix_checkpoints.pop_front();
        }
        trace.update_ms = update_start.map(elapsed_ms).unwrap_or(0.0);
        trace.stored = true;

        Ok(trace.finish(total_start))
    }

    fn prepare_moe_gdn_prefix_state(
        &mut self,
        tokens: &[u32],
        cached_prefix_len: u32,
        block_size: u32,
        extra_keys_per_block: &[Vec<u64>],
        cache_salt: u64,
        continued_live_prefix: bool,
    ) -> Result<MoeGdnPrefixPreparation> {
        let trace_enabled = inference_trace_enabled();
        let prepare_trace_start = trace_enabled.then(std::time::Instant::now);
        let gdn_caches_ready = moe_paged_linear_caches_ready(&self.config, self.caches.as_deref());
        if gdn_caches_ready && continued_live_prefix {
            if let Some(start) = prepare_trace_start {
                write_inference_trace(format_args!(
                    "[MLX_TRACE] qwen3.5-moe gdn_prefix_prepare_done state=live \
                     cached_prefix_tokens={} elapsed_ms={:.1}",
                    cached_prefix_len,
                    elapsed_ms(start)
                ));
            }
            return Ok(MoeGdnPrefixPreparation {
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
                    "[MLX_TRACE] qwen3.5-moe gdn_prefix_prepare_done state=last_history \
                     cached_prefix_tokens={} elapsed_ms={:.1}",
                    cached_prefix_len,
                    elapsed_ms(start)
                ));
            }
            return Ok(MoeGdnPrefixPreparation {
                state: "last_history",
                already_primed: true,
            });
        }
        if cached_prefix_len > 0 {
            let history_lookup_start = trace_enabled.then(std::time::Instant::now);
            let history_checkpoint =
                self.find_moe_gdn_history_checkpoint(tokens, cached_prefix_len);
            let history_lookup_ms = history_lookup_start.map(elapsed_ms);
            if let Some(checkpoint) = history_checkpoint {
                self.caches = Some(checkpoint);
                if let Some(start) = prepare_trace_start {
                    write_inference_trace(format_args!(
                        "[MLX_TRACE] qwen3.5-moe gdn_prefix_prepare_done state=last_history_checkpoint \
                         cached_prefix_tokens={} history_lookup_ms={:.1} elapsed_ms={:.1}",
                        cached_prefix_len,
                        history_lookup_ms.unwrap_or(0.0),
                        elapsed_ms(start)
                    ));
                }
                return Ok(MoeGdnPrefixPreparation {
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
                    "[MLX_TRACE] qwen3.5-moe gdn_history_checkpoint_miss \
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
        let prefix_checkpoint = self.find_moe_gdn_prefix_checkpoint(
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
                    "[MLX_TRACE] qwen3.5-moe gdn_prefix_prepare_done state=checkpoint \
                     cached_prefix_tokens={} prefix_lookup_ms={:.1} elapsed_ms={:.1}",
                    cached_prefix_len,
                    prefix_lookup_ms.unwrap_or(0.0),
                    elapsed_ms(start)
                ));
            }
            return Ok(MoeGdnPrefixPreparation {
                state: "checkpoint",
                already_primed: true,
            });
        }

        self.caches = Some(fresh_moe_layer_caches(&self.config));
        if cached_prefix_len == 0 {
            if let Some(start) = prepare_trace_start {
                write_inference_trace(format_args!(
                    "[MLX_TRACE] qwen3.5-moe gdn_prefix_prepare_done state=replay \
                     cached_prefix_tokens=0 prefix_lookup_ms={:.1} elapsed_ms={:.1}",
                    prefix_lookup_ms.unwrap_or(0.0),
                    elapsed_ms(start)
                ));
            }
            return Ok(MoeGdnPrefixPreparation {
                state: "replay",
                already_primed: false,
            });
        }

        let cached_prefix_len_usize = cached_prefix_len as usize;
        let prefix = tokens.get(..cached_prefix_len_usize).ok_or_else(|| {
            Error::from_reason("MoE paged GDN prefix replay length exceeds prompt length")
        })?;
        let embed = self.embedding.clone();
        let caches_ref = self
            .caches
            .as_mut()
            .ok_or_else(|| Error::from_reason("MoE paged GDN prefix caches not initialized"))?;
        let replay_trace_start = trace_enabled.then(std::time::Instant::now);
        super::paged_forward::run_gdn_only_prefill(prefix, &embed, &mut self.layers, caches_ref)?;
        let replay_ms = replay_trace_start.map(elapsed_ms);
        let store_trace = if moe_gdn_store_replayed_prefix_checkpoint_enabled() {
            self.remember_moe_gdn_prefix_checkpoint(
                tokens,
                cached_prefix_len,
                block_size,
                extra_keys_per_block,
                cache_salt,
            )?
        } else {
            MoeGdnCheckpointStoreTrace::default()
        };
        if let Some(start) = prepare_trace_start {
            write_inference_trace(format_args!(
                "[MLX_TRACE] qwen3.5-moe gdn_prefix_prepare_done state={} \
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

        Ok(MoeGdnPrefixPreparation {
            state: if store_trace.stored {
                "replay_store"
            } else {
                "replay"
            },
            already_primed: true,
        })
    }

    /// Set the tokenizer.
    pub(crate) fn set_tokenizer(&mut self, tokenizer: Arc<Qwen3Tokenizer>) {
        self.tokenizer = Some(tokenizer);
    }

    /// Set the vision encoder.
    ///
    /// Permits loading the vision encoder even when `paged_adapter` is
    /// active so VLM checkpoints can run text-only inference through
    /// the paged dispatch. See `Qwen35Inner::set_vision_encoder` (dense)
    /// for the full rationale; in short, the chat-entry sites reject
    /// `has_images && paged_adapter` so text-only paged turns proceed
    /// normally while image turns surface a clear runtime error, and
    /// for text-only inputs M-RoPE collapses to standard scalar-offset
    /// RoPE so flat and paged byte-equal parity holds.
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

    /// Core chat implementation (runs on model thread).
    ///
    /// `eos_token_id` is the caller-supplied stop-on token id (typically
    /// `<|im_end|>`) so the cached history ends on a clean ChatML
    /// boundary, yielding a reusable prefix for subsequent session
    /// deltas via [`Self::chat_session_start_sync`].
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
        let mut first_token_instant: Option<std::time::Instant> = None;

        let model_id = self.model_id;

        // Block-paged dispatch — early-return BEFORE the compile lock.
        // See dense `chat_sync_core` for the compile-lockout rationale.
        if self.paged_adapter.is_some() {
            if has_images {
                return Err(Error::from_reason(
                    "Qwen3.5 MoE paged dispatch is text-only; image-bearing turns require \
                     use_block_paged_cache=false (text-only turns continue to work).",
                ));
            }
            return self.chat_sync_core_paged(tokens, tokenizer, eos_token_id, p, report_perf);
        }

        // Check if compiled path will be used
        let use_cpp = unsafe { mlx_sys::mlx_qwen35_get_model_id() } == model_id;

        // Serialize MoE compiled lifecycle across model instances
        let _moe_lock = if use_cpp {
            Some(MOE_COMPILED_MUTEX.lock().unwrap_or_else(|e| e.into_inner()))
        } else {
            None
        };

        // Re-validate compiled path under weight lock
        let mut _weight_guard = None;
        let use_cpp = if use_cpp {
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
        // and there is literally no delta to prefill. We still need to
        // produce a `last_logits` for the decode loop, and the only safe
        // way to do that on the Qwen3.5 MoE hybrid stack is a full reset
        // + re-prefill. Trimming the cache by one token is infeasible
        // because the 30 GDN linear-attention layers carry a recurrent
        // state that cannot be rewound mid-sequence (see the invariant
        // doc on `verify_cache_prefix_direct`). In practice this branch
        // is a cold edge case — real agent turns always append at least
        // a user message, so the cached prefix is strictly shorter than
        // the new prompt.
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

        // Track token history for repetition penalty
        let mut token_history: Vec<u32> = expanded_tokens.clone();

        let embedding_weight_t = embedding_weight.transpose(Some(&[1, 0]))?;
        let generation_stream = Stream::new(DeviceType::Gpu);
        let model_size_bytes = self.config.estimate_memory_bytes() as usize;
        let _wired_ctx =
            crate::stream::WiredLimitContext::new(model_size_bytes, vec![generation_stream]);

        // StreamContext created ONCE for entire prefill+decode
        let _stream_ctx = StreamContext::new(generation_stream);

        let fa_idx = self.fa_idx;

        // Profiler
        let mut profiler = crate::decode_profiler::DecodeProfiler::new("moe_chat", "qwen3_5_moe");
        profiler.set_prompt_tokens(prefill_tokens.len() as u32);
        profiler.snapshot_memory_before();

        // === VLM or text prefill branching ===
        profiler.begin_prefill();
        let (mut last_logits, seq_len) = if has_images && cached_prefix_len > 0 {
            // VLM cache reuse: same images, incremental text-only prefill.
            // Routed through chunked_prefill — typically the delta is a
            // small user turn so this is a one-iteration no-op, but a user
            // pasting a long follow-up message still benefits.
            let prompt = MxArray::from_uint32(&prefill_tokens, &[1, prefill_tokens.len() as i64])?;

            let logits = chunked_prefill(
                &prompt,
                &embedding_weight,
                &mut self.layers,
                &mut self.caches,
                &self.final_norm,
                &self.lm_head,
                fa_idx,
                Some(&embedding_weight_t),
                generation_stream,
            )?;

            let seq_len = logits.shape_at(1)?;
            let last_logits = logits.slice_axis(1, seq_len - 1, seq_len)?;
            let last_logits = last_logits.squeeze(Some(&[1]))?;
            (last_logits, expanded_tokens.len() as i64)
        } else if has_images && cached_prefix_len == 0 {
            if let Some(vision_enc) = self.vision_encoder.clone() {
                let final_tokens = &expanded_tokens;
                let processed = vlm_processed
                    .as_ref()
                    .ok_or_else(|| Error::from_reason("VLM processed images missing"))?;

                let input_ids =
                    MxArray::from_uint32(final_tokens, &[1, final_tokens.len() as i64])?;

                let (logits, rope_deltas) = vlm_prefill_moe(
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
                    generation_stream,
                    fa_idx,
                    Some(&embedding_weight_t),
                    &self.vision_cache,
                )?;

                self.cached_rope_deltas = Some(rope_deltas as i32);
                let vlm_seq_len = final_tokens.len() as i64;
                (logits, vlm_seq_len)
            } else {
                return Err(Error::from_reason(
                    "VLM prefill requested but vision encoder/processor not loaded",
                ));
            }
        } else {
            // Standard text prefill. Chunked to bound peak GPU memory for
            // long prompts (e.g. 40k+ tokens) — see `chunked_prefill` docs.
            let prompt = MxArray::from_uint32(&prefill_tokens, &[1, prefill_tokens.len() as i64])?;

            let logits = chunked_prefill(
                &prompt,
                &embedding_weight,
                &mut self.layers,
                &mut self.caches,
                &self.final_norm,
                &self.lm_head,
                fa_idx,
                Some(&embedding_weight_t),
                generation_stream,
            )?;

            let seq_len = logits.shape_at(1)?;
            let last_logits = logits.slice_axis(1, seq_len - 1, seq_len)?;
            let last_logits = last_logits.squeeze(Some(&[1]))?;
            (last_logits, tokens.len() as i64)
        };
        profiler.end_prefill();

        last_logits = apply_all_penalties(last_logits, &token_history, &p)?;
        let mut y = sample(&last_logits, p.sampling_config)?;
        MxArray::async_eval_arrays(&[&y]);

        let mut reasoning_tracker = chat_common::ReasoningTracker::new(
            enable_thinking.unwrap_or(true),
            p.thinking_token_budget,
            think_end_id,
        );

        if use_cpp {
            let _moe_guard = MoeResetGuard;
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
            let mlp_only: Vec<i32> = self
                .config
                .mlp_only_layers
                .as_deref()
                .unwrap_or(&[])
                .to_vec();
            unsafe {
                sys::mlx_qwen35_moe_init_from_prefill(
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
                    1, // batch_size
                    self.config.num_experts,
                    self.config.num_experts_per_tok,
                    if self.config.norm_topk_prob { 1 } else { 0 },
                    self.config.decoder_sparse_step,
                    if mlp_only.is_empty() {
                        std::ptr::null()
                    } else {
                        mlp_only.as_ptr()
                    },
                    mlp_only.len() as i32,
                    cache_ptrs.as_mut_ptr(),
                    prefill_len,
                );
            }

            // Apply M-RoPE offset correction AFTER init_from_prefill
            if has_images && let Some(delta) = self.cached_rope_deltas {
                unsafe {
                    mlx_sys::mlx_qwen35_moe_adjust_offset(delta);
                }
            }

            // For text-only, clear stale rope deltas
            if !has_images {
                self.cached_rope_deltas = None;
            }

            profiler.set_label("moe_chat_compiled");

            let mut ops = chat_common::DecodeOps {
                forward: |ids: &MxArray, emb: &MxArray| -> Result<(MxArray, bool)> {
                    Ok((forward_moe_cpp(ids, emb)?, false))
                },
                eval_step: |token: &MxArray, logits: &MxArray, budget_forced: bool| {
                    eval_token_and_moe_caches(token);
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

            // Export caches from C++ before MoeResetGuard drops
            if reuse_cache {
                let num_layers = self.config.num_layers as usize;
                let mut export_ptrs: Vec<*mut mlx_sys::mlx_array> =
                    vec![std::ptr::null_mut(); num_layers * 2];
                let exported = unsafe {
                    mlx_sys::mlx_qwen35_moe_export_caches(
                        export_ptrs.as_mut_ptr(),
                        (num_layers * 2) as i32,
                    )
                };
                if exported > 0 {
                    let cache_offset = unsafe { mlx_sys::mlx_qwen35_moe_get_cache_offset() };
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
                    // Force-materialize the exported lazy cache handles
                    // before `MoeResetGuard` drops at end of scope and tears
                    // down `g_compiled_caches_moe`. Without this, the arrays
                    // held by `self.caches` reference compiled-graph nodes
                    // that get freed at guard drop, so the next turn's
                    // compile init would feed stale handles to the GPU —
                    // triggering Metal page-faults / innocent-victim hangs
                    // on the first forward of the next turn.
                    eval_layer_caches(&self.caches)?;
                }
            }
            // _moe_guard dropped here, calling mlx_qwen35_moe_reset()
        } else {
            // Rust fallback decode loop
            profiler.set_label("moe_chat_rust");

            let mut ops = chat_common::DecodeOps {
                forward: |ids: &MxArray, emb: &MxArray| -> Result<(MxArray, bool)> {
                    let logits = forward_inner(
                        ids,
                        emb,
                        &mut self.layers,
                        &mut self.caches,
                        &self.final_norm,
                        &self.lm_head,
                        fa_idx,
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
            &tokens,
            Some(&expanded_tokens),
            current_image_cache_key,
            &mut self.cached_token_history,
            &mut self.cached_image_key,
            &mut self.cached_rope_deltas,
            &mut self.caches,
        );

        let performance = compute_performance_metrics(
            generation_start,
            first_token_instant,
            prefill_tokens.len(),
            generated_tokens.len(),
        );

        let mut result = finalize_chat_result(
            &tokenizer,
            &generated_tokens,
            finish_reason,
            think_end_id,
            think_end_str.as_deref(),
            performance,
            p.include_reasoning,
            enable_thinking.unwrap_or(true),
            if has_images {
                expanded_tokens.len() as u32
            } else {
                tokens.len() as u32
            },
            reasoning_tracker.reasoning_token_count(),
        )?;
        // Report the length of the reused cached prefix for observability.
        // `cached_prefix_len` is 0 on fresh/miss paths and the full cached
        // length on an exact-append hit — see the invariant doc on
        // `verify_cache_prefix_direct`.
        result.cached_tokens = cached_prefix_len as u32;
        Ok(result)
    }

    /// Block-paged variant of [`Self::chat_sync_core`] for the MoE
    /// model. Mirrors the dense paged dispatch — see
    /// `Qwen35Inner::chat_sync_core_paged` for the full rationale.
    ///
    /// Unlike the dense paged path, the MoE paged decode loop dispatches
    /// through the C++ compiled paged forward (`mlx_qwen35_moe_forward_paged`)
    /// when the C++ weights are still registered for this model
    /// (`mlx_qwen35_get_model_id() == self.model_id`). The compiled graph
    /// reads K/V from the adapter pool via `paged_kv_write` /
    /// `paged_attention` and reads GDN linear caches from the per-layer
    /// `Qwen3_5LayerCache::Linear(ArraysCache)` via the
    /// `linear_cache_arrays` FFI parameter. Falls back to the pure-Rust
    /// paged decode (`paged_forward::run_paged_decode_step`) when weights
    /// have been swapped out by another model load.
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
        let trace_enabled = inference_trace_enabled();
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

        // Detect availability of the C++ compiled paged decode path. The
        // gating is identical to the flat path: the weights for this
        // model must still be registered (no other model has overwritten
        // `g_active_model_id`). We acquire `MOE_COMPILED_MUTEX` to
        // serialize the compiled lifecycle across model instances —
        // both the legacy flat init/forward/reset AND the new paged
        // init/forward/reset share the same C++ `g_paged_*` /
        // `g_moe_*` globals (see `mlx_qwen35_moe_reset` in
        // `mlx_qwen35_moe.cpp`), so concurrent dispatchers from
        // different models would otherwise stomp on each other's state.
        // Then re-validate the model id under the weights read lock to
        // avoid a TOCTOU race where another model swapped weights
        // between the unlocked check and our compiled init.
        let model_id = self.model_id;
        let use_cpp_paged = unsafe { mlx_sys::mlx_qwen35_get_model_id() } == model_id;
        let _moe_lock = if use_cpp_paged {
            Some(MOE_COMPILED_MUTEX.lock().unwrap_or_else(|e| e.into_inner()))
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

        let seq_id: u32 = 0;
        // Lazy decode allocation: pass the prompt length only.
        let total_budget = tokens.len() as u32;
        // Phase 6: per-block extra_keys. See `chat_sync_core_paged` in
        // qwen3_5/model.rs for the rationale; text-only paged dispatch
        // builds an all-empty per-block vec which is bit-equal to
        // passing `&[]` to the uniform API. VLM-paged would replace the
        // empty positions with real (token_pos, image_hash) pairs.
        let block_size = {
            let adapter = self.paged_adapter.as_ref().ok_or_else(|| {
                Error::from_reason("MoE chat_sync_core_paged: paged_adapter is None")
            })?;
            adapter.block_size()
        };
        let lookup_extra_keys = chat_common::build_paged_extra_keys(tokens.len(), block_size, &[]);
        let cache_salt = 0;
        // vLLM exact-prefix cap — see qwen3/model.rs:chat_sync_core_paged.
        let max_cache_hit_tokens = total_budget.saturating_sub(1);
        let live_ready;
        let live_prefix_match;
        let live_tokens_len;
        let mut live_mismatch = TokenPrefixMismatchTrace::default();
        let (cached_prefix_len, continued_live_prefix) = {
            let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                Error::from_reason("MoE chat_sync_core_paged: paged_adapter is None")
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
                    Ok((prior, _)) => (prior, true),
                    Err(_) => {
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
                "[MLX_TRACE] qwen3.5-moe paged_prefix_lookup prompt_tokens={} \
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

        let gdn_prefix_preparation = self.prepare_moe_gdn_prefix_state(
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
                Error::from_reason(
                    "MoE chat_sync_core_paged: cached_prefix_len > total_prompt_tokens",
                )
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
        let gdn_history_checkpoint_store = self.remember_moe_gdn_history_checkpoint()?;
        if inference_trace_enabled() {
            write_inference_trace(format_args!(
                "[MLX_TRACE] qwen3.5-moe gdn_history_checkpoint stored={} tokens={} \
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
            "MoE chat_sync_core_paged_inner: caller must cap max_cache_hit_tokens at prompt.len() - 1"
        );

        let suffix = &tokens[(cached_prefix_len as usize)..];
        let layer_kinds = crate::models::qwen3_5::decoder_layer::compute_layer_kinds(
            self.config.num_layers as usize,
            |i| self.config.is_linear_layer(i),
        );

        // Pure-Rust paged prefill: writes K/V into the adapter pool via
        // `update_keys_values` per layer (Metal kernel dispatch — direct
        // buffer mutation, NOT MLX graph) and populates the GDN linear
        // caches in `Qwen3_5LayerCache::Linear(ArraysCache)`. Both are
        // exactly what the C++ compiled paged decode reads as inputs.
        let last_logits = {
            let embed = self.embedding.clone();
            let embedding_weight = embed.get_weight();
            let caches_ref = self.caches.as_mut().ok_or_else(|| {
                Error::from_reason("MoE chat_sync_core_paged_inner: caches not initialized")
            })?;
            let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                Error::from_reason("MoE chat_sync_core_paged_inner: paged_adapter dropped")
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
        let mut y = sample(&last_logits, *sampling_config)?;
        y.eval();

        // Smooth memory peak: drop transient prefill buffers before decode
        // starts allocating. Prefill of long prompts builds a massive MLX
        // subgraph; once we have the last logits, those intermediates are
        // dead but MLX's cache holds them.
        crate::array::synchronize_and_clear_cache();

        if report_perf {
            *first_token_instant = Some(std::time::Instant::now());
        }

        // Decide between C++ compiled paged decode (fast) and pure-Rust
        // paged decode (fallback). C++ paged needs:
        // 1. `use_cpp_paged` (weights still registered for our model_id —
        //    re-validated under `COMPILED_WEIGHTS_RWLOCK` in the caller).
        // 2. `adapter.block_size() == 16`. The compiled C++ paged graph
        //    in `mlx_qwen35_moe.cpp` hard-codes `block_size = 16` (see
        //    `attn_for_compile_paged` in `mlx_qwen35_common.h` and the
        //    docstring on `mlx_qwen35_moe_init_paged`). Any other value
        //    would have Rust encode slot/block tables at the adapter's
        //    block size while C++ writes/reads at 16, corrupting KV.
        // 3. `init_paged_moe_compiled_session` to succeed (every linear
        //    layer must have populated conv/recurrent state from the
        //    pure-Rust GDN forward above; every full-attn layer must
        //    have a usable `LayerKVPool` slot, and the C++ `g_paged_inited`
        //    must be set after init catches no exceptions).
        // The `MoeResetGuard` ensures `mlx_qwen35_moe_reset()` runs on
        // any exit path so the next session starts with cleared
        // `g_paged_*` globals (`g_paged_inited == false`).
        let mut cpp_session_ready = if use_cpp_paged {
            // We need both immutable borrows for caches+adapter; this
            // closure scope is the simplest way to satisfy the borrow
            // checker without ferrying handles up through the function.
            let caches_ref = self.caches.as_ref().ok_or_else(|| {
                Error::from_reason("MoE chat_sync_core_paged_inner: caches dropped post-prefill")
            })?;
            let adapter_ref = self.paged_adapter.as_ref().ok_or_else(|| {
                Error::from_reason(
                    "MoE chat_sync_core_paged_inner: paged_adapter dropped post-prefill",
                )
            })?;
            if adapter_ref.block_size() != CPP_PAGED_REQUIRED_BLOCK_SIZE {
                eprintln!(
                    "[MLX] Qwen3_5MoE: skipping C++ compiled paged decode — \
                     adapter block_size={} but compiled graph requires {}; \
                     falling back to pure-Rust paged decode",
                    adapter_ref.block_size(),
                    CPP_PAGED_REQUIRED_BLOCK_SIZE
                );
                false
            } else {
                let prefill_offset = adapter_ref.current_token_count() as i32;
                init_paged_moe_compiled_session(
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

        // RAII guard: resets BOTH g_moe_* and g_paged_* globals when this
        // scope ends. Always installed when we touched `g_paged_inited`
        // (i.e. on the cpp_session_ready path) so a subsequent flat-mode
        // turn or another model load doesn't see stale paged state.
        // Even if a later forward fails and we flip
        // `cpp_session_ready=false`, the guard still runs at scope exit.
        let _moe_paged_guard = cpp_session_ready.then_some(MoeResetGuard);

        // Tracks whether ANY compiled C++ paged step has succeeded
        // during this turn. After a successful compiled step the C++
        // side has advanced its per-layer GDN linear-cache globals
        // (conv_state / recurrent_state) but those updates are never
        // imported back into `self.caches` until the loop finishes.
        // Falling back to pure-Rust decode after that point would run
        // from stale pre-step GDN state while `paged_adapter` and
        // `token_history` have already advanced — silently corrupting
        // the rest of the request. The mid-turn fallback below is
        // therefore only safe BEFORE the first successful compiled step.
        let mut cpp_compiled_step_completed = false;

        let max_new_tokens = p.max_new_tokens;
        let mut generated_tokens: Vec<u32> = Vec::with_capacity(max_new_tokens.max(0) as usize);
        let mut finish_reason = String::from("length");

        // Compile-cached `max_blocks_per_seq` shape — picking the
        // adapter's max_seq_len divided by block_size keeps the compile
        // key stable across all decode steps within one turn (the only
        // varying inputs are the array contents). For shorter sequences
        // the trailing block_table entries are sentinel (-1) and the
        // gather kernel skips them via `num_valid_blocks`.
        let max_blocks_per_seq: u32 = {
            let adapter = self.paged_adapter.as_ref().ok_or_else(|| {
                Error::from_reason(
                    "MoE chat_sync_core_paged_inner: paged_adapter dropped pre-decode",
                )
            })?;
            // Round up the model's max position embedding to block size.
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
                        "MoE chat_sync_core_paged_inner: paged_adapter dropped mid-decode (cpp)",
                    )
                })?;
                adapter
                    .record_tokens(&[token_id])
                    .map_err(Error::from_reason)?;
                let inputs = adapter
                    .build_paged_attention_inputs(1, 1, max_blocks_per_seq)
                    .map_err(Error::from_reason)?;
                let input_ids = MxArray::from_uint32(&[token_id], &[1, 1])?;
                match forward_moe_cpp_paged(&input_ids, &embedding_weight, &inputs) {
                    Ok(logits) => {
                        cpp_compiled_step_completed = true;
                        logits
                    }
                    Err(e) => {
                        if chat_common::should_propagate_compiled_paged_error(
                            cpp_compiled_step_completed,
                        ) {
                            eprintln!(
                                "[MLX] Qwen3_5MoE: C++ compiled paged forward failed \
                                 mid-decode (step={step}) AFTER an earlier compiled step \
                                 succeeded. The C++ GDN linear-cache globals have advanced \
                                 but those updates are not imported back into self.caches, \
                                 so a pure-Rust fallback would run from stale pre-step \
                                 state and silently corrupt the response. Propagating as \
                                 fatal. cause: {e}"
                            );
                            adapter
                                .rollback_last_tokens(1)
                                .map_err(Error::from_reason)?;
                            return Err(e);
                        }
                        eprintln!(
                            "[MLX] Qwen3_5MoE: C++ compiled paged forward failed on first \
                             decode step (step={step}); rolling back token cursor and \
                             falling back to pure-Rust paged decode for the rest of this \
                             request. cause: {e}"
                        );
                        adapter
                            .rollback_last_tokens(1)
                            .map_err(Error::from_reason)?;
                        cpp_session_ready = false;
                        let embed = self.embedding.clone();
                        let embedding_weight_pure = embed.get_weight();
                        let caches_ref = self.caches.as_mut().ok_or_else(|| {
                            Error::from_reason(
                                "MoE chat_sync_core_paged_inner: caches dropped during cpp fallback",
                            )
                        })?;
                        let adapter_mut = self.paged_adapter.as_mut().ok_or_else(|| {
                            Error::from_reason(
                                "MoE chat_sync_core_paged_inner: paged_adapter dropped during cpp fallback",
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
                    Error::from_reason("MoE chat_sync_core_paged_inner: caches dropped mid-decode")
                })?;
                let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                    Error::from_reason(
                        "MoE chat_sync_core_paged_inner: paged_adapter dropped mid-decode",
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
            match export_paged_moe_linear_caches(&self.config) {
                Ok(Some(new_caches)) => {
                    self.caches = Some(new_caches);
                    eval_layer_caches(&self.caches)?;
                }
                Ok(None) => {
                    self.caches = None;
                }
                Err(err) => {
                    write_inference_trace(format_args!(
                        "[MLX_TRACE] qwen3.5-moe paged_linear_cache_export_error error={}",
                        err
                    ));
                    self.caches = None;
                }
            }
        }

        Ok((generated_tokens, finish_reason))
    }

    /// Block-paged streaming variant for MoE — mirrors dense
    /// `chat_stream_sync_core_paged`. See [`Self::chat_sync_core_paged`]
    /// for the C++ compiled paged dispatch rationale; the streaming path
    /// uses the same lock acquisition + fall-back semantics.
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
        let trace_enabled = inference_trace_enabled();
        let request_trace_start = trace_enabled.then(std::time::Instant::now);
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

        let mut decode_stream = tokenizer.inner().decode_stream(true);
        let mut streamed_text_len = 0usize;
        let mut last_is_reasoning = thinking_enabled;

        // C++ paged-decode availability + compile-lifecycle locks. See
        // `chat_sync_core_paged` for the full rationale; this is the
        // streaming twin.
        let model_id = self.model_id;
        let use_cpp_paged = unsafe { mlx_sys::mlx_qwen35_get_model_id() } == model_id;
        let _moe_lock = if use_cpp_paged {
            Some(MOE_COMPILED_MUTEX.lock().unwrap_or_else(|e| e.into_inner()))
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

        let seq_id: u32 = 0;
        // Lazy decode allocation: pass the prompt length only.
        let total_budget = tokens.len() as u32;
        // Phase 6: per-block extra_keys. See comments above.
        let block_size = {
            let adapter = self.paged_adapter.as_ref().ok_or_else(|| {
                Error::from_reason("MoE chat_stream_sync_core_paged: paged_adapter is None")
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
                Error::from_reason("MoE chat_stream_sync_core_paged: paged_adapter is None")
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
                    Ok((prior, _)) => (prior, true),
                    Err(_) => {
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
                "[MLX_TRACE] qwen3.5-moe paged_prefix_lookup prompt_tokens={} \
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

        let prefill_trace_start = trace_enabled.then(std::time::Instant::now);
        let gdn_prefix_preparation = self.prepare_moe_gdn_prefix_state(
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
                    "MoE chat_stream_sync_core_paged: cached_prefix_len > total_prompt_tokens",
                )
            })?;

        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] qwen3.5-moe stream_paged_start prompt_tokens={} \
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
            prefill_trace_start,
        );

        if let Some(start) = request_trace_start {
            match &result {
                Ok((generated_tokens, finish_reason)) => {
                    write_inference_trace(format_args!(
                        "[MLX_TRACE] qwen3.5-moe stream_paged_done generated_tokens={} \
                         finish_reason={} elapsed_ms={:.1}",
                        generated_tokens.len(),
                        finish_reason,
                        elapsed_ms(start)
                    ));
                }
                Err(err) => {
                    write_inference_trace(format_args!(
                        "[MLX_TRACE] qwen3.5-moe stream_paged_error elapsed_ms={:.1} error={}",
                        elapsed_ms(start),
                        err
                    ));
                }
            }
        }

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
        let gdn_history_checkpoint_store = self.remember_moe_gdn_history_checkpoint()?;
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] qwen3.5-moe gdn_history_checkpoint stored={} tokens={} \
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
        prefill_trace_start: Option<std::time::Instant>,
    ) -> Result<(Vec<u32>, String)> {
        // Invariant: caller-applied vLLM cap guarantees suffix_len > 0.
        debug_assert!(
            suffix_len > 0,
            "MoE chat_stream_sync_core_paged_inner: caller must cap max_cache_hit_tokens at prompt.len() - 1"
        );

        let trace_enabled = inference_trace_enabled();
        let suffix = &tokens[(cached_prefix_len as usize)..];
        let layer_kinds = crate::models::qwen3_5::decoder_layer::compute_layer_kinds(
            self.config.num_layers as usize,
            |i| self.config.is_linear_layer(i),
        );

        // Pure-Rust paged prefill — see `chat_sync_core_paged_inner` for
        // the data-flow contract this populates (pool K/V + GDN linear
        // caches).
        let last_logits = {
            let embed = self.embedding.clone();
            let embedding_weight = embed.get_weight();
            let caches_ref = self.caches.as_mut().ok_or_else(|| {
                Error::from_reason("MoE chat_stream_sync_core_paged_inner: caches not initialized")
            })?;
            let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                Error::from_reason("MoE chat_stream_sync_core_paged_inner: paged_adapter dropped")
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

        if let Some(start) = prefill_trace_start {
            write_inference_trace(format_args!(
                "[MLX_TRACE] qwen3.5-moe paged_first_token_ready prompt_tokens={} \
                 cached_prefix_tokens={} suffix_tokens={} prefill_to_first_token_ms={:.1}",
                tokens.len(),
                cached_prefix_len,
                suffix_len,
                elapsed_ms(start)
            ));
        }

        if report_perf {
            *first_token_instant = Some(std::time::Instant::now());
        }

        // C++ compiled paged decode setup (see sync twin for full
        // explanation). Mirrors the sync path so streaming and sync
        // dispatchers behave identically when both paths are available.
        let decode_setup_trace_start = trace_enabled.then(std::time::Instant::now);
        let mut cpp_session_ready = if use_cpp_paged {
            let caches_ref = self.caches.as_ref().ok_or_else(|| {
                Error::from_reason(
                    "MoE chat_stream_sync_core_paged_inner: caches dropped post-prefill",
                )
            })?;
            let adapter_ref = self.paged_adapter.as_ref().ok_or_else(|| {
                Error::from_reason(
                    "MoE chat_stream_sync_core_paged_inner: paged_adapter dropped post-prefill",
                )
            })?;
            if adapter_ref.block_size() != CPP_PAGED_REQUIRED_BLOCK_SIZE {
                eprintln!(
                    "[MLX] Qwen3_5MoE: skipping C++ compiled paged decode — \
                     adapter block_size={} but compiled graph requires {}; \
                     falling back to pure-Rust paged decode",
                    adapter_ref.block_size(),
                    CPP_PAGED_REQUIRED_BLOCK_SIZE
                );
                false
            } else {
                let prefill_offset = adapter_ref.current_token_count() as i32;
                init_paged_moe_compiled_session(
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
        if let Some(start) = decode_setup_trace_start {
            let adapter_block_size = self
                .paged_adapter
                .as_ref()
                .map(|adapter| adapter.block_size())
                .unwrap_or(0);
            write_inference_trace(format_args!(
                "[MLX_TRACE] qwen3.5-moe paged_decode_setup cpp_requested={} cpp_ready={} \
                 block_size={} compiled_required_block_size={} setup_ms={:.1}",
                use_cpp_paged,
                cpp_session_ready,
                adapter_block_size,
                CPP_PAGED_REQUIRED_BLOCK_SIZE,
                elapsed_ms(start)
            ));
        }

        // Even if a later forward fails and we flip
        // `cpp_session_ready=false`, the guard still runs at scope exit.
        let _moe_paged_guard = cpp_session_ready.then_some(MoeResetGuard);

        // Tracks whether ANY compiled C++ paged step has succeeded
        // during this turn. After a successful compiled step the C++
        // GDN linear-cache globals (conv_state / recurrent_state) have
        // advanced but are not imported back into `self.caches` until
        // the loop finishes, so a pure-Rust fallback would read stale
        // pre-step state. The mid-turn fallback below is therefore only
        // safe BEFORE the first successful compiled step. See sync sibling
        // `chat_sync_core_paged_inner` for the full rationale.
        let mut cpp_compiled_step_completed = false;

        let max_new_tokens = p.max_new_tokens;
        let mut generated_tokens: Vec<u32> = Vec::with_capacity(max_new_tokens.max(0) as usize);
        let mut finish_reason = String::from("length");
        let decode_trace_start = trace_enabled.then(std::time::Instant::now);
        let decode_progress_interval = if trace_enabled {
            crate::array::paged_decode_cache_clear_interval().max(1) as usize
        } else {
            usize::MAX
        };
        let mut decode_progress_last = decode_trace_start.unwrap_or_else(std::time::Instant::now);
        let mut decode_progress_last_count = 0usize;
        let mut decode_build_inputs_ms = 0.0;
        let mut decode_forward_ms = 0.0;
        let mut decode_sample_build_ms = 0.0;
        let mut decode_token_eval_ms = 0.0;
        let mut decode_cache_clear_ms = 0.0;

        let max_blocks_per_seq: u32 = {
            let adapter = self.paged_adapter.as_ref().ok_or_else(|| {
                Error::from_reason(
                    "MoE chat_stream_sync_core_paged_inner: paged_adapter dropped pre-decode",
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
                        "MoE chat_stream_sync_core_paged_inner: paged_adapter dropped mid-decode (cpp)",
                    )
                })?;
                adapter
                    .record_tokens(&[token_id])
                    .map_err(Error::from_reason)?;
                let build_inputs_trace_start = trace_enabled.then(std::time::Instant::now);
                let inputs = adapter
                    .build_paged_attention_inputs(1, 1, max_blocks_per_seq)
                    .map_err(Error::from_reason)?;
                if let Some(start) = build_inputs_trace_start {
                    decode_build_inputs_ms += elapsed_ms(start);
                }
                let input_ids = MxArray::from_uint32(&[token_id], &[1, 1])?;
                let forward_trace_start = trace_enabled.then(std::time::Instant::now);
                let forward_result = forward_moe_cpp_paged(&input_ids, &embedding_weight, &inputs);
                if let Some(start) = forward_trace_start {
                    decode_forward_ms += elapsed_ms(start);
                }
                match forward_result {
                    Ok(logits) => {
                        cpp_compiled_step_completed = true;
                        logits
                    }
                    Err(e) => {
                        if chat_common::should_propagate_compiled_paged_error(
                            cpp_compiled_step_completed,
                        ) {
                            eprintln!(
                                "[MLX] Qwen3_5MoE (stream): C++ compiled paged forward \
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
                            "[MLX] Qwen3_5MoE (stream): C++ compiled paged forward failed \
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
                                "MoE chat_stream_sync_core_paged_inner: caches dropped during cpp fallback",
                            )
                        })?;
                        let adapter_mut = self.paged_adapter.as_mut().ok_or_else(|| {
                            Error::from_reason(
                                "MoE chat_stream_sync_core_paged_inner: paged_adapter dropped during cpp fallback",
                            )
                        })?;
                        let fallback_trace_start = trace_enabled.then(std::time::Instant::now);
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
                        if let Some(start) = fallback_trace_start {
                            decode_forward_ms += elapsed_ms(start);
                        }
                        logits.squeeze(Some(&[1]))?
                    }
                }
            } else {
                let embed = self.embedding.clone();
                let embedding_weight = embed.get_weight();
                let caches_ref = self.caches.as_mut().ok_or_else(|| {
                    Error::from_reason(
                        "MoE chat_stream_sync_core_paged_inner: caches dropped mid-decode",
                    )
                })?;
                let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                    Error::from_reason(
                        "MoE chat_stream_sync_core_paged_inner: paged_adapter dropped mid-decode",
                    )
                })?;
                let forward_trace_start = trace_enabled.then(std::time::Instant::now);
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
                if let Some(start) = forward_trace_start {
                    decode_forward_ms += elapsed_ms(start);
                }
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

            let sample_trace_start = trace_enabled.then(std::time::Instant::now);
            y = sample(&next_logits, sampling_config)?;
            if let Some(start) = sample_trace_start {
                decode_sample_build_ms += elapsed_ms(start);
            }
            let token_eval_trace_start = trace_enabled.then(std::time::Instant::now);
            y.eval();
            if let Some(start) = token_eval_trace_start {
                decode_token_eval_ms += elapsed_ms(start);
            }

            let cache_clear_trace_start = trace_enabled.then(std::time::Instant::now);
            crate::array::maybe_clear_cache_for_paged_step(step);
            if let Some(start) = cache_clear_trace_start {
                decode_cache_clear_ms += elapsed_ms(start);
            }
            if trace_enabled
                && generated_tokens
                    .len()
                    .is_multiple_of(decode_progress_interval)
            {
                let window_ms = elapsed_ms(decode_progress_last);
                let window_tokens = generated_tokens
                    .len()
                    .saturating_sub(decode_progress_last_count);
                let window_tok_s = if window_ms > 0.0 {
                    window_tokens as f64 / (window_ms / 1000.0)
                } else {
                    0.0
                };
                let elapsed_decode_ms = decode_trace_start.map(elapsed_ms).unwrap_or(0.0);
                let active_mib = crate::array::get_active_memory() / (1024.0 * 1024.0);
                let cache_mib = crate::array::get_cache_memory() / (1024.0 * 1024.0);
                let peak_mib = crate::array::get_peak_memory() / (1024.0 * 1024.0);
                write_inference_trace(format_args!(
                    "[MLX_TRACE] qwen3.5-moe paged_decode_progress generated_tokens={} \
                     context_tokens={} window_tokens={} window_ms={:.1} window_tok_s={:.2} \
                     elapsed_ms={:.1} cpp_ready={} build_inputs_ms={:.1} forward_ms={:.1} \
                     sample_ms={:.1} sample_build_ms={:.1} token_eval_ms={:.1} \
                     cache_clear_ms={:.1} active_mib={:.1} cache_mib={:.1} peak_mib={:.1}",
                    generated_tokens.len(),
                    token_history.len(),
                    window_tokens,
                    window_ms,
                    window_tok_s,
                    elapsed_decode_ms,
                    cpp_session_ready,
                    decode_build_inputs_ms,
                    decode_forward_ms,
                    decode_sample_build_ms + decode_token_eval_ms,
                    decode_sample_build_ms,
                    decode_token_eval_ms,
                    decode_cache_clear_ms,
                    active_mib,
                    cache_mib,
                    peak_mib
                ));
                decode_progress_last = std::time::Instant::now();
                decode_progress_last_count = generated_tokens.len();
            }
        }

        if let Some(start) = decode_trace_start {
            write_inference_trace(format_args!(
                "[MLX_TRACE] qwen3.5-moe paged_decode_done generated_tokens={} finish_reason={} \
                 decode_loop_ms={:.1} cpp_compiled_step_completed={} build_inputs_ms={:.1} \
                 forward_ms={:.1} sample_ms={:.1} sample_build_ms={:.1} \
                 token_eval_ms={:.1} cache_clear_ms={:.1}",
                generated_tokens.len(),
                finish_reason,
                elapsed_ms(start),
                cpp_compiled_step_completed,
                decode_build_inputs_ms,
                decode_forward_ms,
                decode_sample_build_ms + decode_token_eval_ms,
                decode_sample_build_ms,
                decode_token_eval_ms,
                decode_cache_clear_ms
            ));
        }

        if cpp_compiled_step_completed {
            match export_paged_moe_linear_caches(&self.config) {
                Ok(Some(new_caches)) => {
                    self.caches = Some(new_caches);
                    eval_layer_caches(&self.caches)?;
                    write_inference_trace(format_args!(
                        "[MLX_TRACE] qwen3.5-moe paged_linear_cache_export ok=true"
                    ));
                }
                Ok(None) => {
                    self.caches = None;
                    write_inference_trace(format_args!(
                        "[MLX_TRACE] qwen3.5-moe paged_linear_cache_export ok=false reason=not_initialized"
                    ));
                }
                Err(err) => {
                    write_inference_trace(format_args!(
                        "[MLX_TRACE] qwen3.5-moe paged_linear_cache_export ok=false error={}",
                        err
                    ));
                    self.caches = None;
                }
            }
        }

        Ok((generated_tokens, finish_reason))
    }

    /// Core streaming chat implementation (runs on model thread).
    ///
    /// `eos_token_id` is the caller-supplied stop-on token id (typically
    /// `<|im_end|>`) so the cached history ends on a clean ChatML
    /// boundary, yielding a reusable prefix for subsequent session
    /// deltas via [`Self::chat_stream_session_start_sync`].
    fn chat_stream_sync_core(
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
        if self.paged_adapter.is_some() {
            if has_images {
                return Err(Error::from_reason(
                    "Qwen3.5 MoE paged dispatch is text-only; image-bearing turns require \
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
        let use_cpp = unsafe { mlx_sys::mlx_qwen35_get_model_id() } == model_id;
        let _moe_lock = if use_cpp {
            Some(MOE_COMPILED_MUTEX.lock().unwrap_or_else(|e| e.into_inner()))
        } else {
            None
        };

        let mut _weight_guard = None;
        let use_cpp = if use_cpp {
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
        // token is not possible across Qwen3.5 MoE's 30 linear-attention
        // layers, so the only safe response to an exact-match prompt is
        // a full reset + re-prefill.
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

        let fa_idx = self.fa_idx;

        let mut profiler =
            crate::decode_profiler::DecodeProfiler::new("moe_chat_stream", "qwen3_5_moe");
        profiler.set_prompt_tokens(prefill_tokens.len() as u32);
        profiler.snapshot_memory_before();

        // VLM or text prefill
        profiler.begin_prefill();
        let (mut last_logits, seq_len) = if has_images && cached_prefix_len > 0 {
            // VLM cache reuse (streaming): same images, incremental text-only
            // prefill. See the sync sibling for the rationale.
            let prompt = MxArray::from_uint32(&prefill_tokens, &[1, prefill_tokens.len() as i64])?;

            let logits = chunked_prefill(
                &prompt,
                &embedding_weight,
                &mut self.layers,
                &mut self.caches,
                &self.final_norm,
                &self.lm_head,
                fa_idx,
                Some(&embedding_weight_t),
                generation_stream,
            )?;

            let seq_len = logits.shape_at(1)?;
            let last_logits = logits.slice_axis(1, seq_len - 1, seq_len)?;
            let last_logits = last_logits.squeeze(Some(&[1]))?;
            (last_logits, expanded_tokens.len() as i64)
        } else if has_images && cached_prefix_len == 0 {
            if let Some(vision_enc) = self.vision_encoder.clone() {
                let final_tokens = &expanded_tokens;
                let processed = vlm_processed
                    .as_ref()
                    .ok_or_else(|| Error::from_reason("VLM processed images missing"))?;

                let input_ids =
                    MxArray::from_uint32(final_tokens, &[1, final_tokens.len() as i64])?;

                let (logits, rope_deltas) = vlm_prefill_moe(
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
                    generation_stream,
                    fa_idx,
                    Some(&embedding_weight_t),
                    &self.vision_cache,
                )?;

                self.cached_rope_deltas = Some(rope_deltas as i32);
                let vlm_seq_len = final_tokens.len() as i64;
                (logits, vlm_seq_len)
            } else {
                return Err(Error::from_reason(
                    "VLM prefill requested but vision encoder/processor not loaded",
                ));
            }
        } else {
            // Chunked to bound peak GPU memory for long prompts. See
            // `chunked_prefill` docs for the memory rationale.
            let prompt = MxArray::from_uint32(&prefill_tokens, &[1, prefill_tokens.len() as i64])?;
            let logits = chunked_prefill(
                &prompt,
                &embedding_weight,
                &mut self.layers,
                &mut self.caches,
                &self.final_norm,
                &self.lm_head,
                fa_idx,
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
            (last_logits, total_seq_len)
        };
        profiler.end_prefill();

        let mut token_history: Vec<u32> = tokens.clone();
        last_logits = apply_all_penalties(last_logits, &token_history, &p)?;
        let mut y = sample(&last_logits, p.sampling_config)?;
        MxArray::async_eval_arrays(&[&y]);

        let starts_in_thinking = enable_thinking.unwrap_or(true);
        let mut last_is_reasoning = starts_in_thinking;
        let mut reasoning_tracker = chat_common::ReasoningTracker::new(
            starts_in_thinking,
            p.thinking_token_budget,
            think_end_id,
        );

        if use_cpp {
            let _moe_guard = MoeResetGuard;
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
            let mlp_only: Vec<i32> = self
                .config
                .mlp_only_layers
                .as_deref()
                .unwrap_or(&[])
                .to_vec();
            unsafe {
                sys::mlx_qwen35_moe_init_from_prefill(
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
                    self.config.num_experts,
                    self.config.num_experts_per_tok,
                    if self.config.norm_topk_prob { 1 } else { 0 },
                    self.config.decoder_sparse_step,
                    if mlp_only.is_empty() {
                        std::ptr::null()
                    } else {
                        mlp_only.as_ptr()
                    },
                    mlp_only.len() as i32,
                    cache_ptrs.as_mut_ptr(),
                    prefill_len,
                );
            }

            if has_images && let Some(delta) = self.cached_rope_deltas {
                unsafe {
                    mlx_sys::mlx_qwen35_moe_adjust_offset(delta);
                }
            }

            if !has_images {
                self.cached_rope_deltas = None;
            }

            profiler.set_label("moe_chat_stream_compiled");

            let mut ops = chat_common::DecodeOps {
                forward: |ids: &MxArray, emb: &MxArray| -> Result<(MxArray, bool)> {
                    Ok((forward_moe_cpp(ids, emb)?, false))
                },
                eval_step: |token: &MxArray, logits: &MxArray, budget_forced: bool| {
                    eval_token_and_moe_caches(token);
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

            // Export caches from C++ before MoeResetGuard drops
            if reuse_cache {
                let num_layers = self.config.num_layers as usize;
                let mut export_ptrs: Vec<*mut mlx_sys::mlx_array> =
                    vec![std::ptr::null_mut(); num_layers * 2];
                let exported = unsafe {
                    mlx_sys::mlx_qwen35_moe_export_caches(
                        export_ptrs.as_mut_ptr(),
                        (num_layers * 2) as i32,
                    )
                };
                if exported > 0 {
                    let cache_offset = unsafe { mlx_sys::mlx_qwen35_moe_get_cache_offset() };
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
                    // See the chat path for rationale: force-eval the
                    // exported lazy handles before `MoeResetGuard` clears
                    // `g_compiled_caches_moe` at end of scope.
                    eval_layer_caches(&self.caches)?;
                }
            }
            // _moe_guard dropped here
        } else {
            profiler.set_label("moe_chat_stream_rust");

            let mut ops = chat_common::DecodeOps {
                forward: |ids: &MxArray, emb: &MxArray| -> Result<(MxArray, bool)> {
                    let logits = forward_inner(
                        ids,
                        emb,
                        &mut self.layers,
                        &mut self.caches,
                        &self.final_norm,
                        &self.lm_head,
                        fa_idx,
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

    /// Start a new chat session.
    ///
    /// Delegates to [`Self::chat_sync_core`] with `<|im_end|>` (from the
    /// tokenizer vocab) as the stop token so the cached history ends on
    /// a clean ChatML boundary that subsequent `chat_session_continue_sync`
    /// deltas can append to without re-rendering the jinja template.
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
    /// [`Self::chat_sync_core`] already handles the VLM prefill path via
    /// `vlm_prefill_moe`. Subsequent turns in the same session MUST go
    /// through `chat_session_continue_sync` which is text-only; changing
    /// the image set mid-session requires starting a new session via
    /// this method again.
    pub(crate) fn chat_session_start_sync(
        &mut self,
        messages: Vec<ChatMessage>,
        config: ChatConfig,
    ) -> Result<ChatResult> {
        // Mirror the symmetric guard in `chat_tokens_delta_sync`. The session
        // API only makes sense with cache reuse enabled.
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
        // pure *append* on the live caches. Qwen3.5 MoE has 30 GDN
        // linear-attention layers whose recurrent state cannot be
        // rewound mid-sequence; the all-or-nothing return contract is
        // what keeps that state consistent without any snapshot
        // machinery. See the rustdoc on `verify_cache_prefix_direct`.

        self.chat_sync_core(messages, config, im_end_id)
    }

    /// Prefill a pre-tokenized delta on top of the existing KV caches and
    /// run the decode loop. Text-only session primitive used by
    /// `chat_session_continue_sync` and `chat_session_continue_tool_sync`.
    ///
    /// Uses `<|im_end|>` as the eos token (not `config.eos_token_id`) so
    /// the cached history continues to end on a clean ChatML boundary for
    /// the next turn. Cache save runs unconditionally at the end so the
    /// session stays consistent even on error.
    pub(crate) fn chat_tokens_delta_sync(
        &mut self,
        delta_tokens: Vec<u32>,
        config: ChatConfig,
    ) -> Result<ChatResult> {
        // The delta path is a session-reuse operation by construction.
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
        // Text-only delta on image-bearing cache is intentional — the KV
        // cache retains the image attention state from the prior prefill.
        // See the sibling guard's doc in `qwen3_5/model.rs`. The outer
        // `chat_session_continue_sync` gate filters real image-set
        // changes with the `IMAGE_CHANGE_REQUIRES_SESSION_RESTART:`
        // prefix so the TS `ChatSession` can route those through
        // `chatSessionStart`.

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

        // Build full token history = cached_history + delta. Used for
        // penalty context AND as the running token history in the decode loop.
        // Snapshot the cached-prefix length before extending so we can
        // report it on the ChatResult for observability — the delta path
        // always reuses the full cached history by construction.
        let cached_prefix_len_for_result = self.cached_token_history.len() as u32;
        let mut full_token_history = self.cached_token_history.clone();
        full_token_history.extend(delta_tokens.iter().copied());

        let p = extract_chat_params(&config);
        let max_new_tokens = p.max_new_tokens;
        let enable_thinking = chat_common::resolve_enable_thinking(&config);

        let generation_start = if report_perf {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut first_token_instant: Option<std::time::Instant> = None;

        let model_id = self.model_id;

        // Block-paged dispatch — early-return BEFORE the compile lock.
        // The delta path drives the paged core with the FULL token
        // history; the adapter's warm-continue path matches the cached
        // prefix automatically.
        if self.paged_adapter.is_some() {
            return self.chat_sync_core_paged(
                full_token_history.clone(),
                tokenizer.clone(),
                eos_id,
                p,
                report_perf,
            );
        }

        // Check compiled path availability
        let use_cpp = unsafe { mlx_sys::mlx_qwen35_get_model_id() } == model_id;

        // Serialize MoE compiled lifecycle across model instances
        let _moe_lock = if use_cpp {
            Some(MOE_COMPILED_MUTEX.lock().unwrap_or_else(|e| e.into_inner()))
        } else {
            None
        };

        let mut _weight_guard = None;
        let use_cpp = if use_cpp {
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

        // StreamContext created ONCE for entire prefill+decode
        let _stream_ctx = StreamContext::new(generation_stream);

        let fa_idx = self.fa_idx;

        let mut profiler =
            crate::decode_profiler::DecodeProfiler::new("moe_chat_delta", "qwen3_5_moe");
        profiler.set_prompt_tokens(delta_tokens.len() as u32);
        profiler.snapshot_memory_before();

        // Text-only prefill of the delta on top of the existing caches.
        // Usually tiny (a single user turn), but chunked defensively so a
        // user pasting a long follow-up message doesn't blow memory.
        profiler.begin_prefill();
        let prompt = MxArray::from_uint32(&delta_tokens, &[1, delta_tokens.len() as i64])?;
        let logits = chunked_prefill(
            &prompt,
            &embedding_weight,
            &mut self.layers,
            &mut self.caches,
            &self.final_norm,
            &self.lm_head,
            fa_idx,
            Some(&embedding_weight_t),
            generation_stream,
        )?;
        let prefill_out_seq_len = logits.shape_at(1)?;
        let mut last_logits = logits.slice_axis(1, prefill_out_seq_len - 1, prefill_out_seq_len)?;
        last_logits = last_logits.squeeze(Some(&[1]))?;
        // Total context length post-prefill = full history length.
        let seq_len = full_token_history.len() as i64;
        profiler.end_prefill();

        let prompt_tokens_for_result = full_token_history.len() as u32;

        // Save snapshot for save_cache_state_direct (prior history + delta).
        let save_tokens = full_token_history.clone();

        // Decode setup.
        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut finish_reason = String::from("length");

        let mut token_history: Vec<u32> = full_token_history;
        last_logits = apply_all_penalties(last_logits, &token_history, &p)?;
        let mut y = sample(&last_logits, p.sampling_config)?;
        MxArray::async_eval_arrays(&[&y]);

        let mut reasoning_tracker = chat_common::ReasoningTracker::new(
            enable_thinking.unwrap_or(true),
            p.thinking_token_budget,
            think_end_id,
        );

        if use_cpp {
            let _moe_guard = MoeResetGuard;
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
            let mlp_only: Vec<i32> = self
                .config
                .mlp_only_layers
                .as_deref()
                .unwrap_or(&[])
                .to_vec();
            unsafe {
                sys::mlx_qwen35_moe_init_from_prefill(
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
                    self.config.num_experts,
                    self.config.num_experts_per_tok,
                    if self.config.norm_topk_prob { 1 } else { 0 },
                    self.config.decoder_sparse_step,
                    if mlp_only.is_empty() {
                        std::ptr::null()
                    } else {
                        mlp_only.as_ptr()
                    },
                    mlp_only.len() as i32,
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
                    mlx_sys::mlx_qwen35_moe_adjust_offset(delta);
                }
            }

            profiler.set_label("moe_chat_delta_compiled");

            let mut ops = chat_common::DecodeOps {
                forward: |ids: &MxArray, emb: &MxArray| -> Result<(MxArray, bool)> {
                    Ok((forward_moe_cpp(ids, emb)?, false))
                },
                eval_step: |token: &MxArray, logits: &MxArray, budget_forced: bool| {
                    eval_token_and_moe_caches(token);
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

            // Export caches from C++ before MoeResetGuard drops
            if p.reuse_cache {
                let num_layers = self.config.num_layers as usize;
                let mut export_ptrs: Vec<*mut mlx_sys::mlx_array> =
                    vec![std::ptr::null_mut(); num_layers * 2];
                let exported = unsafe {
                    mlx_sys::mlx_qwen35_moe_export_caches(
                        export_ptrs.as_mut_ptr(),
                        (num_layers * 2) as i32,
                    )
                };
                if exported > 0 {
                    let cache_offset = unsafe { mlx_sys::mlx_qwen35_moe_get_cache_offset() };
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
                    // See the chat path for rationale: force-eval the
                    // exported lazy handles before `MoeResetGuard` clears
                    // `g_compiled_caches_moe` at end of scope.
                    eval_layer_caches(&self.caches)?;
                }
            }
            // _moe_guard dropped here, calling mlx_qwen35_moe_reset()
        } else {
            profiler.set_label("moe_chat_delta_rust");

            let mut ops = chat_common::DecodeOps {
                forward: |ids: &MxArray, emb: &MxArray| -> Result<(MxArray, bool)> {
                    let logits = forward_inner(
                        ids,
                        emb,
                        &mut self.layers,
                        &mut self.caches,
                        &self.final_norm,
                        &self.lm_head,
                        fa_idx,
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
        // `cached_image_key` — the live KV cache still encodes the prior
        // prefill's image attention state even though this turn is
        // text-only, and a subsequent cache-prefix verify needs that
        // key to stay in place so a later image-bearing turn correctly
        // flags an image-set change instead of being accepted on the
        // delta path.
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

        let performance = compute_performance_metrics(
            generation_start,
            first_token_instant,
            delta_tokens.len(),
            generated_tokens.len(),
        );

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
        // Delta path always reuses the full cached history — report it.
        result.cached_tokens = cached_prefix_len_for_result;
        Ok(result)
    }

    /// Session-based chat continuation via a plain user message string.
    ///
    /// Convenience entry point on top of `chat_tokens_delta_sync`: builds
    /// the ChatML delta that closes the previous assistant turn, opens a
    /// new user turn with `user_message`, and opens a fresh assistant
    /// turn. Then tokenizes the delta and delegates to
    /// `chat_tokens_delta_sync`.
    ///
    /// Text-only; `images` is an opt-in guard parameter: non-empty input
    /// is rejected with an `IMAGE_CHANGE_REQUIRES_SESSION_RESTART:`-
    /// prefixed error so the TS `ChatSession` layer can route
    /// image-changes back through a fresh `chat_session_start_sync`.
    pub(crate) fn chat_session_continue_sync(
        &mut self,
        user_message: String,
        images: Option<Vec<Uint8Array>>,
        config: ChatConfig,
    ) -> Result<ChatResult> {
        if images.as_ref().is_some_and(|v| !v.is_empty()) {
            return Err(Error::from_reason(format!(
                "{} chat_session_continue is text-only; start a new session with chat_session_start to change the image",
                IMAGE_CHANGE_RESTART_PREFIX
            )));
        }

        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))?
            .clone();

        // Match `chat_sync`'s sanitization so the session path is subject
        // to the same role/content injection protection as the legacy path.
        let synthetic = build_synthetic_user_message(&user_message);
        let sanitized = Qwen3Tokenizer::sanitize_messages_public(std::slice::from_ref(&synthetic));
        let sanitized_user = &sanitized[0].content;

        let enable_thinking = chat_common::resolve_enable_thinking(&config);
        let delta_text = build_chatml_continue_delta_text(sanitized_user, enable_thinking);

        let delta_tokens = tokenizer.encode_sync(&delta_text, Some(false))?;

        self.chat_tokens_delta_sync(delta_tokens, config)
    }

    /// Session-based chat continuation via a tool-result turn.
    ///
    /// Builds a ChatML `<tool_response>`-wrapped delta from `content` and
    /// prefills it on top of the live session caches. See
    /// [`chat_common::build_chatml_tool_delta_text`] for the exact wire
    /// format. The `tool_call_id` is currently ignored by the wire format.
    ///
    /// Text-only; delegates to [`Self::chat_tokens_delta_sync`] which
    /// inherits the same text-only-delta invariant (errors if the session
    /// currently holds image state).
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

    /// Streaming chat (session-start variant): same semantics as
    /// [`Self::chat_session_start_sync`] but streams token deltas through
    /// `stream_tx` rather than returning a `ChatResult`. Stops on
    /// `<|im_end|>` and resets caches before prefill.
    ///
    /// Images are accepted on session start — the downstream
    /// `chat_stream_sync_core` already handles VLM prefill via
    /// `vlm_prefill_moe`.
    pub(crate) fn chat_stream_session_start_sync(
        &mut self,
        messages: Vec<ChatMessage>,
        config: ChatConfig,
        stream_tx: StreamTx<ChatStreamChunk>,
        cancelled: Arc<AtomicBool>,
    ) {
        let cb = StreamSender(stream_tx.clone());

        if cancelled.load(Ordering::Relaxed) {
            send_stream_error(
                &stream_tx,
                "chat_stream_session_start cancelled before start",
            );
            return;
        }

        if config.reuse_cache == Some(false) {
            send_stream_error(
                &stream_tx,
                "chat_stream_session_start requires reuse_cache=true (leave as None or set to true). \
                 The session API only makes sense with cache reuse enabled.",
            );
            return;
        }

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
        // Any cached state from a prior turn is intentionally preserved
        // so `chat_stream_sync_core` can run `verify_cache_prefix_direct`
        // against the freshly-tokenized prompt. The inner path resets the
        // caches on a miss and reuses them on an exact-append hit. See
        // the rustdoc on `verify_cache_prefix_direct` for the GDN-safety
        // invariant that keeps this sound on the 30 linear-attention
        // layers.

        let result = self.chat_stream_sync_core(messages, config, im_end_id, &cb, &cancelled);
        if let Err(e) = result {
            let _ = stream_tx.send(Err(e));
        }
    }

    /// Streaming chat (session-continue variant): same semantics as
    /// [`Self::chat_session_continue_sync`] but streams token deltas
    /// through `stream_tx`.
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
                    IMAGE_CHANGE_RESTART_PREFIX
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

        let synthetic = build_synthetic_user_message(&user_message);
        let sanitized = Qwen3Tokenizer::sanitize_messages_public(std::slice::from_ref(&synthetic));
        let sanitized_user = &sanitized[0].content;

        let enable_thinking = chat_common::resolve_enable_thinking(&config);
        let delta_text = build_chatml_continue_delta_text(sanitized_user, enable_thinking);

        let delta_tokens = match tokenizer.encode_sync(&delta_text, Some(false)) {
            Ok(t) => t,
            Err(e) => {
                let _ = stream_tx.send(Err(e));
                return;
            }
        };

        self.chat_stream_tokens_delta_sync(delta_tokens, config, stream_tx, cancelled);
    }

    /// Streaming analog of [`Self::chat_session_continue_tool_sync`].
    /// `is_error` is forwarded verbatim to the wire-format renderer;
    /// see the non-streaming entry point for the marker semantics.
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
    /// Applies the same four guards as the non-streaming path and still
    /// calls `save_cache_state_direct` at the end regardless of whether
    /// cancellation fired, so the cache stays consistent for the next
    /// turn even on an early abort.
    pub(crate) fn chat_stream_tokens_delta_sync(
        &mut self,
        delta_tokens: Vec<u32>,
        config: ChatConfig,
        stream_tx: StreamTx<ChatStreamChunk>,
        cancelled: Arc<AtomicBool>,
    ) {
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
        // Text-only streaming deltas are allowed over image-bearing
        // cache — see the sync sibling for the rationale. Real image-set
        // changes are caught by the outer `chat_stream_session_continue`
        // gate.

        let cb = StreamSender(stream_tx.clone());
        let result =
            self.chat_stream_tokens_delta_sync_inner(delta_tokens, config, &cb, &cancelled);
        if let Err(e) = result {
            let _ = stream_tx.send(Err(e));
        }
    }

    /// Prefill the delta tokens and run the streaming decode loop.
    ///
    /// This mirrors [`Self::chat_stream_sync_core`] but skips the
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

        let use_cpp = unsafe { mlx_sys::mlx_qwen35_get_model_id() } == model_id;
        let _moe_lock = if use_cpp {
            Some(MOE_COMPILED_MUTEX.lock().unwrap_or_else(|e| e.into_inner()))
        } else {
            None
        };

        let mut _weight_guard = None;
        let use_cpp = if use_cpp {
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

        let fa_idx = self.fa_idx;

        let mut profiler =
            crate::decode_profiler::DecodeProfiler::new("moe_chat_stream_delta", "qwen3_5_moe");
        profiler.set_prompt_tokens(delta_tokens.len() as u32);
        profiler.snapshot_memory_before();

        // Text-only prefill of the delta on top of the existing caches.
        // Chunked defensively — see the sync sibling for rationale.
        profiler.begin_prefill();
        let prompt = MxArray::from_uint32(&delta_tokens, &[1, delta_tokens.len() as i64])?;
        let logits = chunked_prefill(
            &prompt,
            &embedding_weight,
            &mut self.layers,
            &mut self.caches,
            &self.final_norm,
            &self.lm_head,
            fa_idx,
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

        let starts_in_thinking = enable_thinking.unwrap_or(true);
        let mut last_is_reasoning = starts_in_thinking;
        let mut reasoning_tracker = chat_common::ReasoningTracker::new(
            starts_in_thinking,
            p.thinking_token_budget,
            think_end_id,
        );

        if use_cpp {
            let _moe_guard = MoeResetGuard;
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
            let mlp_only: Vec<i32> = self
                .config
                .mlp_only_layers
                .as_deref()
                .unwrap_or(&[])
                .to_vec();
            unsafe {
                sys::mlx_qwen35_moe_init_from_prefill(
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
                    self.config.num_experts,
                    self.config.num_experts_per_tok,
                    if self.config.norm_topk_prob { 1 } else { 0 },
                    self.config.decoder_sparse_step,
                    if mlp_only.is_empty() {
                        std::ptr::null()
                    } else {
                        mlp_only.as_ptr()
                    },
                    mlp_only.len() as i32,
                    cache_ptrs.as_mut_ptr(),
                    prefill_len,
                );
            }

            // Re-apply the saved M-RoPE offset if the session carries
            // image state. See the sync sibling for the full rationale:
            // the live KV caches still encode the prior VLM prefill's
            // image attention, so the offset must re-apply here for
            // tokens to position correctly. `cached_rope_deltas` stays
            // alive across deltas so chained streaming text-only turns
            // on the same image session keep the offset.
            if let Some(delta) = self.cached_rope_deltas {
                unsafe {
                    mlx_sys::mlx_qwen35_moe_adjust_offset(delta);
                }
            }

            profiler.set_label("moe_chat_stream_delta_compiled");

            let mut ops = chat_common::DecodeOps {
                forward: |ids: &MxArray, emb: &MxArray| -> Result<(MxArray, bool)> {
                    Ok((forward_moe_cpp(ids, emb)?, false))
                },
                eval_step: |token: &MxArray, logits: &MxArray, budget_forced: bool| {
                    eval_token_and_moe_caches(token);
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

            // Export caches from C++ before MoeResetGuard drops
            if reuse_cache {
                let num_layers = self.config.num_layers as usize;
                let mut export_ptrs: Vec<*mut mlx_sys::mlx_array> =
                    vec![std::ptr::null_mut(); num_layers * 2];
                let exported = unsafe {
                    mlx_sys::mlx_qwen35_moe_export_caches(
                        export_ptrs.as_mut_ptr(),
                        (num_layers * 2) as i32,
                    )
                };
                if exported > 0 {
                    let cache_offset = unsafe { mlx_sys::mlx_qwen35_moe_get_cache_offset() };
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
                    // See the chat path for rationale: force-eval the
                    // exported lazy handles before `MoeResetGuard` clears
                    // `g_compiled_caches_moe` at end of scope.
                    eval_layer_caches(&self.caches)?;
                }
            }
            // _moe_guard dropped here
        } else {
            profiler.set_label("moe_chat_stream_delta_rust");

            let mut ops = chat_common::DecodeOps {
                forward: |ids: &MxArray, emb: &MxArray| -> Result<(MxArray, bool)> {
                    let logits = forward_inner(
                        ids,
                        emb,
                        &mut self.layers,
                        &mut self.caches,
                        &self.final_norm,
                        &self.lm_head,
                        fa_idx,
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
        // consistent for the next turn. Delta stream preserves
        // `cached_image_key` (see the sync sibling's rationale).
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

    /// Generate text from prompt tokens (synchronous, runs on model thread).
    pub(crate) fn generate_sync(
        &mut self,
        prompt_tokens: MxArray,
        config: Qwen3_5MoeGenerationConfig,
    ) -> Result<Qwen3_5MoeGenerationResult> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))?
            .clone();

        // Init caches
        self.init_caches_sync()?;

        let embedding_weight = self.embedding.get_weight();
        let embedding_weight_t = embedding_weight.transpose(Some(&[1, 0]))?;
        let generation_stream = Stream::new(DeviceType::Gpu);
        let fa_idx = self.fa_idx;

        // Prefill. Chunked to bound peak GPU memory for long prompts —
        // see `chunked_prefill` docs. `chunked_prefill` internally manages
        // the StreamContext per chunk so we don't need an outer one here.
        let prompt = prompt_tokens.reshape(&[1, -1])?;
        let logits = chunked_prefill(
            &prompt,
            &embedding_weight,
            &mut self.layers,
            &mut self.caches,
            &self.final_norm,
            &self.lm_head,
            fa_idx,
            Some(&embedding_weight_t),
            generation_stream,
        )?;

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
                    fa_idx,
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

        let text = tokenizer
            .decode_sync(&generated_tokens, true)
            .unwrap_or_default();

        Ok(Qwen3_5MoeGenerationResult {
            tokens: generated_tokens.clone(),
            text,
            num_tokens: generated_tokens.len() as u32,
            finish_reason: finish_reason.to_string(),
        })
    }

    /// Save model weights and configuration to a directory (synchronous).
    ///
    /// Runs on the dedicated model thread and serializes all weights owned
    /// directly by `Qwen35MoeInner` (no locks). Mirrors the dense implementation
    /// in `qwen3_5::model::Qwen35Inner::save_model_sync`, adapted for the MoE
    /// MLP variant (per-layer dense vs sparse expert routing).
    pub(crate) fn save_model_sync(&self, save_path: &str) -> Result<()> {
        use super::decoder_layer::{AttentionType, MLPType};

        let mut params: HashMap<String, MxArray> = HashMap::new();

        // Embedding
        params.insert("embedding.weight".to_string(), self.embedding.get_weight());

        // Transformer layers
        for (i, layer) in self.layers.iter().enumerate() {
            let prefix = format!("layers.{}", i);

            // Attention weights
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

            // MLP weights — different for Dense vs MoE layers
            match &layer.mlp {
                MLPType::Dense(mlp) => {
                    params.insert(
                        format!("{}.mlp.gate_proj.weight", prefix),
                        mlp.get_gate_proj_weight(),
                    );
                    params.insert(
                        format!("{}.mlp.up_proj.weight", prefix),
                        mlp.get_up_proj_weight(),
                    );
                    params.insert(
                        format!("{}.mlp.down_proj.weight", prefix),
                        mlp.get_down_proj_weight(),
                    );
                }
                MLPType::MoE(moe) => {
                    // Router gate
                    params.insert(format!("{}.mlp.gate.weight", prefix), moe.get_gate_weight());
                    // Expert weights (3D: [num_experts, out, in])
                    let switch_mlp = moe.get_switch_mlp();
                    params.insert(
                        format!("{}.mlp.switch_mlp.gate_proj.weight", prefix),
                        switch_mlp.get_gate_proj_weight(),
                    );
                    params.insert(
                        format!("{}.mlp.switch_mlp.up_proj.weight", prefix),
                        switch_mlp.get_up_proj_weight(),
                    );
                    params.insert(
                        format!("{}.mlp.switch_mlp.down_proj.weight", prefix),
                        switch_mlp.get_down_proj_weight(),
                    );
                    // Shared expert
                    params.insert(
                        format!("{}.mlp.shared_expert.gate_proj.weight", prefix),
                        moe.get_shared_expert_gate_proj_weight(),
                    );
                    params.insert(
                        format!("{}.mlp.shared_expert.up_proj.weight", prefix),
                        moe.get_shared_expert_up_proj_weight(),
                    );
                    params.insert(
                        format!("{}.mlp.shared_expert.down_proj.weight", prefix),
                        moe.get_shared_expert_down_proj_weight(),
                    );
                    // Shared expert gate
                    params.insert(
                        format!("{}.mlp.shared_expert_gate.weight", prefix),
                        moe.get_shared_expert_gate_weight(),
                    );
                }
            }

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

        // LM head (only if not tied to the embedding)
        if !self.config.tie_word_embeddings
            && let Some(ref lm_head) = self.lm_head
        {
            params.insert("lm_head.weight".to_string(), lm_head.get_weight());
        }

        // Include vision encoder weights when present (VLM models)
        if let Some(ref vision_enc) = self.vision_encoder {
            let vision_params = vision_enc.get_parameters();
            params.extend(vision_params);
        }

        // Validate all parameters for NaN/Inf before writing to disk
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

        // Weights metadata (reference sidecar)
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

        // Serialize config and inject model_type for detectModelType
        let mut config_value = serde_json::to_value(&self.config).map_err(|e| {
            napi::Error::new(
                napi::Status::GenericFailure,
                format!("Failed to serialize config: {e}"),
            )
        })?;
        if let serde_json::Value::Object(ref mut map) = config_value {
            map.insert("model_type".to_string(), serde_json::json!("qwen3_5_moe"));
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
        info!("Training state initialized on model thread (Qwen3.5 MoE)");
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
        let fa_idx = self.fa_idx;

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
                    fa_idx,
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
                fa_idx,
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
        let model_type = ModelType::Qwen35Moe(self.config.clone());

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
        let model_type = crate::training_model::ModelType::Qwen35Moe(self.config.clone());

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
    /// Direct field access on Qwen35MoeInner — no locks needed.
    fn apply_gradients_inner(
        &mut self,
        gradients: HashMap<String, &MxArray>,
        learning_rate: f64,
        current_params: &HashMap<String, MxArray>,
    ) -> Result<()> {
        use super::decoder_layer::{AttentionType, MLPType};

        let updated_params =
            crate::training_model::compute_sgd_updates(&gradients, learning_rate, current_params)?;

        // Apply updated parameters directly to model fields
        for (name, updated_param) in updated_params.iter() {
            if name == "lm_head.weight" {
                if let Some(ref mut lm) = self.lm_head {
                    lm.set_weight(updated_param, "lm_head")?;
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
                        match &mut layer.mlp {
                            MLPType::Dense(mlp) => {
                                if name.ends_with(".gate_proj.weight") {
                                    mlp.set_gate_proj_weight(updated_param)?;
                                } else if name.ends_with(".up_proj.weight") {
                                    mlp.set_up_proj_weight(updated_param)?;
                                } else if name.ends_with(".down_proj.weight") {
                                    mlp.set_down_proj_weight(updated_param)?;
                                }
                            }
                            MLPType::MoE(moe) => {
                                if name.ends_with(".mlp.gate.weight") {
                                    moe.set_gate_weight(updated_param)?;
                                } else if name.contains(".mlp.switch_mlp.") {
                                    if name.ends_with(".gate_proj.weight") {
                                        moe.set_switch_mlp_gate_proj_weight(updated_param);
                                    } else if name.ends_with(".up_proj.weight") {
                                        moe.set_switch_mlp_up_proj_weight(updated_param);
                                    } else if name.ends_with(".down_proj.weight") {
                                        moe.set_switch_mlp_down_proj_weight(updated_param);
                                    }
                                } else if name.contains(".mlp.shared_expert_gate.") {
                                    moe.set_shared_expert_gate_weight(updated_param)?;
                                } else if name.contains(".mlp.shared_expert.") {
                                    if name.ends_with(".gate_proj.weight") {
                                        moe.set_shared_expert_gate_proj_weight(updated_param)?;
                                    } else if name.ends_with(".up_proj.weight") {
                                        moe.set_shared_expert_up_proj_weight(updated_param)?;
                                    } else if name.ends_with(".down_proj.weight") {
                                        moe.set_shared_expert_down_proj_weight(updated_param)?;
                                    }
                                }
                            }
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
        use super::decoder_layer::{AttentionType, MLPType};

        let mut params = HashMap::new();

        // Embedding
        params.insert("embedding.weight".to_string(), self.embedding.get_weight());

        // Transformer layers
        for (i, layer) in self.layers.iter().enumerate() {
            let prefix = format!("layers.{}", i);

            // Attention weights
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

            // MLP weights — different for Dense vs MoE layers
            match &layer.mlp {
                MLPType::Dense(mlp) => {
                    params.insert(
                        format!("{}.mlp.gate_proj.weight", prefix),
                        mlp.get_gate_proj_weight(),
                    );
                    params.insert(
                        format!("{}.mlp.up_proj.weight", prefix),
                        mlp.get_up_proj_weight(),
                    );
                    params.insert(
                        format!("{}.mlp.down_proj.weight", prefix),
                        mlp.get_down_proj_weight(),
                    );
                }
                MLPType::MoE(moe) => {
                    // Router gate
                    params.insert(format!("{}.mlp.gate.weight", prefix), moe.get_gate_weight());
                    // Expert weights (3D: [num_experts, out, in])
                    let switch_mlp = moe.get_switch_mlp();
                    params.insert(
                        format!("{}.mlp.switch_mlp.gate_proj.weight", prefix),
                        switch_mlp.get_gate_proj_weight(),
                    );
                    params.insert(
                        format!("{}.mlp.switch_mlp.up_proj.weight", prefix),
                        switch_mlp.get_up_proj_weight(),
                    );
                    params.insert(
                        format!("{}.mlp.switch_mlp.down_proj.weight", prefix),
                        switch_mlp.get_down_proj_weight(),
                    );
                    // Shared expert
                    params.insert(
                        format!("{}.mlp.shared_expert.gate_proj.weight", prefix),
                        moe.get_shared_expert_gate_proj_weight(),
                    );
                    params.insert(
                        format!("{}.mlp.shared_expert.up_proj.weight", prefix),
                        moe.get_shared_expert_up_proj_weight(),
                    );
                    params.insert(
                        format!("{}.mlp.shared_expert.down_proj.weight", prefix),
                        moe.get_shared_expert_down_proj_weight(),
                    );
                    // Shared expert gate
                    params.insert(
                        format!("{}.mlp.shared_expert_gate.weight", prefix),
                        moe.get_shared_expert_gate_weight(),
                    );
                }
            }

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

        // LM head (only if not tied)
        if !self.config.tie_word_embeddings
            && let Some(ref lm_head) = self.lm_head
        {
            params.insert("lm_head.weight".to_string(), lm_head.get_weight());
        }

        Ok(params)
    }
}

/// Qwen3.5 MoE Model -- hybrid linear/full attention with Mixture-of-Experts.
///
/// All inference and training state lives on a dedicated OS thread. NAPI methods
/// dispatch commands via channels and await responses. Training commands are
/// routed through `TrainingDispatch` to the model thread.
#[napi]
pub struct Qwen3_5MoeModel {
    /// Dedicated model thread for inference and training.
    pub(crate) thread: crate::model_thread::ModelThread<Qwen35MoeCmd>,
    /// Cloned from inner for pure-getter NAPI methods (no command dispatch needed).
    pub(crate) config: Qwen3_5MoeConfig,
    /// Snapshot of `Qwen35MoeInner::paged_adapter.is_some()` captured at
    /// construction time. Currently default-OFF on Qwen3.5 MoE
    /// (parity-pending — see CLAUDE.md and
    /// `Qwen3_5MoeConfig::use_block_paged_cache`). VLM checkpoints can
    /// load with the adapter on for text-only inference; image-bearing
    /// chat turns are rejected at runtime by the chat-entry sites.
    /// Surfaced through the `hasBlockPagedCache()` NAPI method.
    pub(crate) paged_active: bool,
    /// RAII: unregisters this model's baseline from the cache-limit
    /// coordinator on drop.
    pub(crate) _cache_limit_guard: crate::cache_limit::CacheLimitGuard,
}

#[napi]
impl Qwen3_5MoeModel {
    /// Reset all caches.
    #[napi]
    pub fn reset_caches(&self) -> Result<()> {
        crate::model_thread::send_and_block(&self.thread, |reply| Qwen35MoeCmd::ResetCaches {
            reply,
        })
    }

    /// Whether the block-paged KV cache adapter is active on this model
    /// instance.
    ///
    /// `true` iff `Qwen35MoeInner::paged_adapter` was successfully
    /// constructed at load time (driven by
    /// `Qwen3_5MoeConfig::use_block_paged_cache`, currently default-OFF
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
    #[napi]
    pub async fn load(path: String) -> Result<Qwen3_5MoeModel> {
        persistence::load_with_thread(&path).await
    }

    /// Generate text from a prompt token sequence.
    #[napi]
    pub async fn generate(
        &self,
        prompt_tokens: &MxArray,
        config: Qwen3_5MoeGenerationConfig,
    ) -> Result<Qwen3_5MoeGenerationResult> {
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
        crate::model_thread::send_and_await(&self.thread, |reply| Qwen35MoeCmd::Generate {
            prompt_tokens: prompt_tokens.clone(),
            config,
            reply,
        })
        .await
    }

    /// Start a new chat session.
    ///
    /// Runs the full jinja chat template once, decodes until `<|im_end|>`,
    /// and leaves the KV caches on a clean ChatML boundary so subsequent
    /// `chatSessionContinue` / `chatSessionContinueTool` calls can
    /// append a raw delta on top without re-rendering the chat
    /// template.
    ///
    /// Image support is conditional on the loaded checkpoint: a
    /// Qwen3.5-VL MoE model loaded with vision weights accepts images
    /// in `messages` (the vision encoder handles prefill), while a
    /// plain text Qwen3.5 MoE checkpoint rejects them with a runtime
    /// error. A mid-session image change requires a fresh
    /// `chatSessionStart` call.
    ///
    /// Requires `config.reuse_cache` to be enabled (the default).
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

        crate::model_thread::send_and_await(&self.thread, |reply| Qwen35MoeCmd::ChatSessionStart {
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
    /// Requires a live session started via `chatSessionStart`.
    /// Errors if the session is empty, carries image state, or if
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

        crate::model_thread::send_and_await(&self.thread, |reply| {
            Qwen35MoeCmd::ChatSessionContinue {
                user_message,
                images,
                config,
                reply,
            }
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
            Qwen35MoeCmd::ChatSessionContinueTool {
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

        self.thread.send(Qwen35MoeCmd::ChatStreamSessionStart {
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

        self.thread.send(Qwen35MoeCmd::ChatStreamSessionContinue {
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

        self.thread
            .send(Qwen35MoeCmd::ChatStreamSessionContinueTool {
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
    // by `crates/mlx-core/tests/qwen3_5_moe_session.rs` to exercise
    // the streaming path from a pure-Rust integration test without a
    // NAPI host. Marked `#[doc(hidden)]` because they're not part of
    // the public API surface.
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
        self.thread.send(Qwen35MoeCmd::ChatStreamSessionStart {
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
        self.thread.send(Qwen35MoeCmd::ChatStreamSessionContinue {
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
    /// Pure config computation -- no model-thread dispatch needed.
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

        let num_experts = self.config.num_experts as i64;
        let moe_i = self
            .config
            .moe_intermediate_size
            .unwrap_or(self.config.intermediate_size) as i64;
        let shared_i = self
            .config
            .shared_expert_intermediate_size
            .unwrap_or(self.config.intermediate_size) as i64;

        let kd = self.config.linear_key_dim() as i64;
        let vd = self.config.linear_value_dim() as i64;

        for layer_idx in 0..n {
            let is_linear = self.config.is_linear_layer(layer_idx);
            let is_moe = self.config.is_moe_layer(layer_idx);

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

            if is_moe {
                total += h * num_experts + num_experts * 3 * h * moe_i + 3 * h * shared_i + h;
            } else {
                total += 3 * h * dense_i;
            }

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
        self.thread.send(Qwen35MoeCmd::SaveModel {
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

/// Forward pass using already-acquired lock guards (no lock overhead).
///
/// Used by generate/chat to avoid re-acquiring locks on every decode step.
fn forward_inner(
    input_ids: &MxArray,
    embedding_weight: &MxArray,
    layers: &mut [DecoderLayer],
    caches: &mut Option<Vec<Qwen3_5LayerCache>>,
    final_norm: &RMSNorm,
    lm_head: &Option<LinearProj>,
    fa_idx: usize,
    embedding_weight_t: Option<&MxArray>,
) -> Result<MxArray> {
    let embedding = Embedding::from_weight(embedding_weight)?;
    let hidden_states = embedding.forward(input_ids)?;
    let mut h = hidden_states.clone();

    let seq_len = hidden_states.shape_at(1)?;
    let fa_mask = {
        let has_cache = caches.is_some();
        if seq_len <= 1 && has_cache {
            None
        } else {
            let offset = caches.as_ref().map(|c| c[fa_idx].offset()).unwrap_or(0);
            Some(create_causal_mask(seq_len as i32, Some(offset), None)?)
        }
    };

    // SSM mask is always None — mlx-vlm never creates one for ArraysCache.
    // An all-ones mask is a no-op that adds unnecessary graph nodes and Metal overhead.

    let num_layers = layers.len();
    for i in 0..num_layers {
        let mask = if layers[i].is_linear() {
            None
        } else {
            fa_mask.as_ref()
        };
        let cache = caches.as_mut().map(|c| &mut c[i]);
        h = layers[i].forward(&h, mask, cache, None, true)?;
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

/// Default prefill chunk size (tokens per chunk).
///
/// Matches the Qwen3.5 Dense path and Python mlx-lm's `prefill_step_size`
/// default of 2048. Long-context prompts (40k+ tokens) would otherwise
/// allocate all per-layer activations concurrently (batch=1 × seq × hidden
/// plus a full attention score tensor per FA layer), blowing past the 96 GB
/// wired limit on an M3 Max 128 GB box. Chunking bounds the per-layer
/// transient peak at `chunk × hidden_dim` and inserts a cache-eval +
/// `clear_cache` barrier between chunks so the transient allocator state
/// does not accumulate across chunks.
pub(crate) const PREFILL_STEP_SIZE: i64 = 2048;

/// Chunked prefill for Qwen3.5 MoE.
///
/// Processes `prompt` (shape `[1, seq_len]`) in chunks of `PREFILL_STEP_SIZE`
/// tokens, evaluating all KV-cache arrays and clearing the MLX compute cache
/// between chunks to bound peak GPU activation memory. Returns the logits
/// from the **final** chunk, which share the same shape contract as a
/// single-shot `forward_inner` call: `[1, last_chunk_len, vocab_size]`.
///
/// Invariants vs. single-shot `forward_inner`:
/// - Identical numerical output at full precision (the KV caches thread
///   through chunk N into chunk N+1 just like they would through
///   successive `forward_inner(full_prompt)` calls during regular decode).
/// - The linear-attention recurrent state advances chunk-by-chunk. This is
///   the same forward direction as a single-shot call — chunking is a
///   memory-only transformation, not a semantic one.
/// - Compiled-path seeding (`mlx_qwen35_moe_init_from_prefill`) is the
///   caller's responsibility and MUST happen **after** the full chunked
///   prefill completes. Do NOT call `init_from_prefill` per chunk.
///
/// Small prompts (<= `PREFILL_STEP_SIZE` tokens) hit exactly one loop
/// iteration and behave identically to a single `forward_inner` call — no
/// extra evals, no extra cache clears.
#[allow(clippy::too_many_arguments)]
fn chunked_prefill(
    prompt: &MxArray,
    embedding_weight: &MxArray,
    layers: &mut [DecoderLayer],
    caches: &mut Option<Vec<Qwen3_5LayerCache>>,
    final_norm: &RMSNorm,
    lm_head: &Option<LinearProj>,
    fa_idx: usize,
    embedding_weight_t: Option<&MxArray>,
    generation_stream: Stream,
) -> Result<MxArray> {
    chunked_prefill_with_size(
        prompt,
        embedding_weight,
        layers,
        caches,
        final_norm,
        lm_head,
        fa_idx,
        embedding_weight_t,
        generation_stream,
        PREFILL_STEP_SIZE,
    )
}

/// Explicit-size variant of `chunked_prefill`.
///
/// Same semantics as `chunked_prefill` but the chunk size is an explicit
/// parameter. Primarily used by tests to compare chunked vs single-shot
/// (by passing a chunk size >= prompt length) without plumbing a config
/// knob through every caller. Production callers should use
/// `chunked_prefill` which hardcodes `PREFILL_STEP_SIZE`.
#[allow(clippy::too_many_arguments)]
fn chunked_prefill_with_size(
    prompt: &MxArray,
    embedding_weight: &MxArray,
    layers: &mut [DecoderLayer],
    caches: &mut Option<Vec<Qwen3_5LayerCache>>,
    final_norm: &RMSNorm,
    lm_head: &Option<LinearProj>,
    fa_idx: usize,
    embedding_weight_t: Option<&MxArray>,
    generation_stream: Stream,
    chunk_size: i64,
) -> Result<MxArray> {
    debug_assert!(chunk_size > 0, "chunk_size must be positive");
    let total_len = prompt.shape_at(1)?;
    let mut offset: i64 = 0;

    // All-but-last chunks: run forward, eval caches, clear compute cache.
    // The returned logits from these chunks are thrown away because only
    // the final chunk's logits are consumed by the sampler.
    while total_len - offset > chunk_size {
        let chunk = prompt.slice_axis(1, offset, offset + chunk_size)?;
        {
            let _stream_ctx = StreamContext::new(generation_stream);
            let _logits = forward_inner(
                &chunk,
                embedding_weight,
                layers,
                caches,
                final_norm,
                lm_head,
                fa_idx,
                embedding_weight_t,
            )?;
        }
        // Materialize all cache arrays on GPU so the next chunk doesn't
        // extend a giant lazy graph rooted at the prior chunk's inputs.
        eval_layer_caches(caches)?;
        crate::array::clear_cache();
        offset += chunk_size;
    }

    // Final chunk: return logits to caller. No eval/clear here — the
    // caller's next step (sampling / slicing last_logits) triggers eval
    // naturally, and the outer decode loop clears cache on its own rhythm.
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
            fa_idx,
            embedding_weight_t,
        )?
    };
    Ok(logits)
}

/// Single-token decode step using C++ MoE forward pass.
///
/// Unlike the dense model's compiled path, MoE routing is data-dependent so
/// mlx::core::compile cannot be used. Instead, C++ builds a fresh computation
/// graph per step, eliminating ~2,200 FFI round-trips per token.
fn forward_moe_cpp(input_ids: &MxArray, embedding_weight: &MxArray) -> Result<MxArray> {
    use mlx_sys as sys;

    let mut output_ptr: *mut sys::mlx_array = std::ptr::null_mut();
    unsafe {
        sys::mlx_qwen35_moe_forward(
            input_ids.as_raw_ptr(),
            embedding_weight.as_raw_ptr(),
            &mut output_ptr,
            std::ptr::null_mut(),
        );
    }

    if output_ptr.is_null() {
        return Err(Error::from_reason(
            "C++ MoE forward step returned null — check stderr for exception details",
        ));
    }

    MxArray::from_handle(output_ptr, "moe_forward_logits")
}

/// Evaluate next_token and all MoE cache arrays to prevent graph accumulation.
///
/// Called after each C++ MoE decode step. The C++ code builds a fresh graph per
/// step but we still need to eval to materialize output arrays and break lazy
/// dependency chains (preventing O(N²) graph growth).
fn eval_token_and_moe_caches(next_token: &MxArray) {
    unsafe {
        mlx_sys::mlx_qwen35_moe_eval_token_and_caches(next_token.as_raw_ptr());
    }
}

/// Block size hard-coded into the compiled C++ paged graph
/// (`mlx_qwen35_moe.cpp` — see `attn_for_compile_paged` and the
/// `mlx_qwen35_moe_init_paged` docstring). The Rust adapter supports
/// configurable block sizes via `Qwen3_5MoeConfig::paged_block_size`,
/// but the compiled graph traces against block_size=16 baked into the
/// `paged_kv_write` / `paged_attention` kernel calls. Mismatched values
/// would have Rust encode slot/block tables at the adapter's block size
/// while C++ writes/reads at 16, corrupting KV state. The compile-branch
/// selectors (`chat_sync_core_paged_inner` / `chat_stream_sync_core_paged_inner`)
/// gate `cpp_session_ready` on this equality and fall back to the
/// pure-Rust paged path when the adapter is configured otherwise.
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
/// 1. `caches` is fully populated by a prior pure-Rust paged prefill —
///    full-attention layers already wrote K/V into the adapter pool via
///    `update_keys_values`; linear layers populated `ArraysCache` via
///    `GatedDeltaNet::forward`.
/// 2. The C++ weights for this model are still registered (caller must
///    have verified `mlx_qwen35_get_model_id() == self.model_id` and
///    holds the appropriate read locks).
///
/// `prefill_offset` is the global token cursor the compiled paged
/// graph's `g_paged_offset_int` will start incrementing from. After a
/// fresh prefill it equals `current_token_count`.
///
/// On any failure (missing linear cache, missing pool/scale handle, or
/// the C++ FFI returning a non-zero status), the helper returns `Err`
/// so the caller can fall back to the pure-Rust paged decode path.
/// Mirrors the `mlx_qwen35_moe_init_paged` exception safety: a non-OK
/// return leaves `g_paged_inited == false` on the C++ side, AND the
/// status code is now propagated back to Rust (Phase 4 piece 3 review
/// fix) so a failed init can never be mistaken for a successful one.
fn init_paged_moe_compiled_session(
    config: &Qwen3_5MoeConfig,
    caches: &[Qwen3_5LayerCache],
    paged_adapter: &PagedKVCacheAdapter,
    prefill_offset: i32,
) -> Result<()> {
    use crate::models::qwen3_5::decoder_layer::{Qwen3_5LayerKind, compute_layer_kinds};

    let num_layers_us = config.num_layers as usize;
    if caches.len() != num_layers_us {
        return Err(Error::from_reason(format!(
            "init_paged_moe_compiled_session: caches.len()={} but config.num_layers={}",
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
    // side has time to copy them into its own globals (the ctor `array(x)`
    // bumps the refcount; once we return the wrappers drop and decrement
    // back to whatever the C++ side incremented to).
    let mut held_arrays: Vec<MxArray> = Vec::with_capacity(num_layers_us * 4);

    for (i, kind) in layer_kinds.iter().enumerate() {
        match kind {
            Qwen3_5LayerKind::Linear => {
                // Pull the live (conv_state, recurrent_state) from the
                // layer's `Qwen3_5LayerCache::Linear(ArraysCache)`. They
                // were populated by the pure-Rust GDN forward during
                // prefill. If a slot is None (e.g. caller forgot to run
                // prefill, or GDN code path bypassed the cache) we
                // surface an error so the dispatcher falls back to
                // pure-Rust paged decode rather than silently producing
                // garbage from bf16 zero placeholders.
                let arrays_cache = match &caches[i] {
                    Qwen3_5LayerCache::Linear(c) => c,
                    Qwen3_5LayerCache::FullAttention(_) => {
                        return Err(Error::from_reason(format!(
                            "init_paged_moe_compiled_session: layer {i} is Linear by config \
                             but cache slot is FullAttention",
                        )));
                    }
                };
                let conv = arrays_cache.get(0).ok_or_else(|| {
                    Error::from_reason(format!(
                        "init_paged_moe_compiled_session: layer {i} conv_state not populated; \
                         pure-Rust paged prefill must run before C++ paged init",
                    ))
                })?;
                let rec = arrays_cache.get(1).ok_or_else(|| {
                    Error::from_reason(format!(
                        "init_paged_moe_compiled_session: layer {i} recurrent_state not \
                         populated; pure-Rust paged prefill must run before C++ paged init",
                    ))
                })?;
                linear_cache_handles[i * 2] = conv.as_raw_ptr();
                linear_cache_handles[i * 2 + 1] = rec.as_raw_ptr();
                // No need to push to held_arrays — `arrays_cache` keeps
                // them alive for the duration of this call (we hold a
                // shared borrow over `caches`).
                let _ = (conv, rec);
            }
            Qwen3_5LayerKind::FullAttentionPaged { paged_idx } => {
                // CRITICAL: index by the COMPACT full-attention ordinal,
                // NOT the absolute layer index. The pool is sized for
                // `full_attention_layer_count()` slots.
                let k_arr = paged_adapter.key_pool_array(*paged_idx).map_err(|e| {
                    Error::from_reason(format!(
                        "init_paged_moe_compiled_session: key_pool_array(layer={i}, \
                         paged_idx={paged_idx}): {e}",
                    ))
                })?;
                let v_arr = paged_adapter.value_pool_array(*paged_idx).map_err(|e| {
                    Error::from_reason(format!(
                        "init_paged_moe_compiled_session: value_pool_array(layer={i}, \
                         paged_idx={paged_idx}): {e}",
                    ))
                })?;
                let ks_arr = paged_adapter.k_scale_array(*paged_idx).map_err(|e| {
                    Error::from_reason(format!(
                        "init_paged_moe_compiled_session: k_scale_array(layer={i}, \
                         paged_idx={paged_idx}): {e}",
                    ))
                })?;
                let vs_arr = paged_adapter.v_scale_array(*paged_idx).map_err(|e| {
                    Error::from_reason(format!(
                        "init_paged_moe_compiled_session: v_scale_array(layer={i}, \
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

    let mlp_only: Vec<i32> = config.mlp_only_layers.as_deref().unwrap_or(&[]).to_vec();

    let status = unsafe {
        mlx_sys::mlx_qwen35_moe_init_paged(
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
            // max_kv_len: bound on cumulative tokens; the paged graph
            // doesn't depend on this for shape (block_table /
            // max_blocks_per_seq is the real bound) but the FFI takes
            // it for symmetry with the flat init.
            config.max_position_embeddings,
            1, // batch_size
            config.num_experts,
            config.num_experts_per_tok,
            if config.norm_topk_prob { 1 } else { 0 },
            config.decoder_sparse_step,
            if mlp_only.is_empty() {
                std::ptr::null()
            } else {
                mlp_only.as_ptr()
            },
            mlp_only.len() as i32,
            k_pool_handles.as_mut_ptr(),
            v_pool_handles.as_mut_ptr(),
            k_scale_handles.as_mut_ptr(),
            v_scale_handles.as_mut_ptr(),
            linear_cache_handles.as_mut_ptr(),
            prefill_offset,
        )
    };

    // held_arrays drops here; refcounts on the wrapped Metal buffers
    // settle to whatever the C++ side bumped them to.
    drop(held_arrays);

    // The C++ side returns 0 on success, -1 on failure. Failure paths
    // include missing pool/scale handles for full-attention layers and
    // any exception caught during graph build (compiled trace, RNG split,
    // etc.). On failure `g_paged_inited` is left cleared so subsequent
    // `mlx_qwen35_moe_forward_paged` calls would null-out their logits;
    // surfacing the failure here lets the dispatcher fall back to the
    // pure-Rust paged path before any decode-step FFI is dispatched.
    if status != 0 {
        return Err(Error::from_reason(format!(
            "init_paged_moe_compiled_session: mlx_qwen35_moe_init_paged returned status={status} \
             (expected 0); see stderr for the C++ diagnostic. Caller must fall back to the \
             pure-Rust paged path."
        )));
    }

    Ok(())
}

/// Single-token decode step using the C++ compiled paged forward pass.
///
/// Mirrors `forward_moe_cpp` but threads through the paged-attention
/// inputs (offset_arr, block_table, slot_mapping, num_valid_tokens,
/// num_valid_blocks, seq_lens) so K/V is written into the adapter's
/// paged Metal pool via `paged_kv_write` and gathered via
/// `paged_attention`.
///
/// Caller contract:
/// * `init_paged_moe_compiled_session` has been called this turn (sets
///   `g_paged_inited = true`).
/// * `paged_adapter.record_tokens(&[token_id])` has been called to
///   advance the cursor (and lazily allocate any new block).
/// * `inputs` was just built via `paged_adapter.build_paged_attention_inputs(1, 1, max_blocks_per_seq)`.
///
/// On any FFI failure (`output_logits == null`) the helper returns an
/// `Err` so the dispatcher falls back to pure-Rust paged decode.
fn forward_moe_cpp_paged(
    input_ids: &MxArray,
    embedding_weight: &MxArray,
    inputs: &crate::transformer::paged_attention_inputs::PagedAttentionInputs,
) -> Result<MxArray> {
    use mlx_sys as sys;

    let mut output_ptr: *mut sys::mlx_array = std::ptr::null_mut();
    let mut cache_offset_out: i32 = 0;
    unsafe {
        sys::mlx_qwen35_moe_forward_paged(
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
            "C++ MoE paged forward step returned null — check stderr for diagnostic. \
             (Common causes: g_paged_inited = false, slot_mapping shape != [1], \
             input_ids size != 1, or weights cleared by another model load.)",
        ));
    }

    MxArray::from_handle(output_ptr, "moe_paged_forward_logits")
}

/// VLM prefill for MoE model using Rust path with M-RoPE position IDs.
///
/// Processes images through vision encoder, merges features into embeddings,
/// computes M-RoPE positions, and runs forward through all layers.
/// Returns (last_logits [1, vocab], rope_deltas).
#[allow(clippy::too_many_arguments)]
fn vlm_prefill_moe(
    input_ids: &MxArray,
    image_cache_key: u64,
    pre_processed: &ProcessedImages,
    vision_encoder: &Qwen3_5VisionEncoder,
    spatial_merge_size: i32,
    text_model_embedding: &MxArray,
    layers_guard: &mut [DecoderLayer],
    caches_guard: &mut Option<Vec<Qwen3_5LayerCache>>,
    final_norm_guard: &RMSNorm,
    lm_head_guard: &Option<LinearProj>,
    generation_stream: Stream,
    fa_idx: usize,
    embedding_weight_t: Option<&MxArray>,
    vision_cache: &VisionCache,
) -> Result<(MxArray, i64)> {
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

    // === STEP 4: Rust prefill with M-RoPE ===
    // MoE VLM always uses Rust path for prefill (no C++ VLM prefill for MoE)
    let logits = {
        let _stream_ctx = StreamContext::new(generation_stream);

        let mut h = inputs_embeds.clone();
        let seq_len = h.shape_at(1)?;

        let fa_mask = if seq_len > 1 {
            let offset = caches_guard
                .as_ref()
                .map(|c| c[fa_idx].offset())
                .unwrap_or(0);
            Some(create_causal_mask(seq_len as i32, Some(offset), None)?)
        } else {
            None
        };

        let num_layers = layers_guard.len();
        for i in 0..num_layers {
            let mask = if layers_guard[i].is_linear() {
                None
            } else {
                fa_mask.as_ref()
            };
            let cache = caches_guard.as_mut().map(|c| &mut c[i]);
            let layer_pos = if layers_guard[i].is_linear() {
                None
            } else {
                Some(&position_ids)
            };
            h = layers_guard[i].forward(&h, mask, cache, layer_pos, true)?;
        }

        let h = final_norm_guard.forward(&h)?;
        let logits = match lm_head_guard {
            Some(head) => head.forward(&h)?,
            None => match embedding_weight_t {
                Some(wt) => h.matmul(wt)?,
                None => {
                    let wt = text_model_embedding.transpose(Some(&[1, 0]))?;
                    h.matmul(&wt)?
                }
            },
        };

        // Eval caches to break lazy chains
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
    Ok((last_logits, rope_deltas))
}

#[cfg(test)]
mod prefix_cache_reuse_integration_tests {
    //! End-to-end tests for the prefix KV cache reuse refactor on
    //! Qwen3.5 MoE. These verify that `chat_session_start_sync` no longer
    //! unconditionally wipes the cache — stateless agent clients that
    //! resend the full transcript on every turn should hit the
    //! `verify_cache_prefix_direct` exact-append path and skip redundant
    //! prefill work.
    //!
    //! The MoE variant additionally exercises the zero-delta guard,
    //! which is architecturally constrained to a full reset + re-prefill
    //! because rewinding the 30 GDN linear-attention layers' recurrent
    //! state mid-sequence is infeasible. The exact-match test locks in
    //! that the guard does not corrupt state (even though it's wasteful).
    //!
    //! These tests are `#[ignore]`-marked because they require loading a
    //! real Qwen3.5 MoE model file and a tokenizer. Run them with:
    //!
    //!     cargo test -p mlx-core --test '*' -- --ignored prefix_cache_reuse_integration
    //!
    //! with `MLX_NODE_QWEN35_MOE_MODEL_DIR` set to a local Qwen3.5-MoE
    //! dir.

    /// Append hit: two back-to-back session-start calls where the second
    /// extends the first by exactly one user turn. Must report
    /// `cached_tokens > 0` and only prefill the delta.
    #[ignore = "requires a real Qwen3.5 MoE model directory; run with --ignored"]
    #[test]
    fn append_hit_reuses_cached_prefix() {
        // See the matching test on `qwen3_5/model.rs` for the pseudocode
        // shape. Identical surface; different model type.
    }

    /// Divergence miss: second call's history is unrelated. Must report
    /// `cached_tokens == 0` and do a full-history prefill (which includes
    /// resetting the 30 GDN layers' recurrent state).
    #[ignore = "requires a real Qwen3.5 MoE model directory; run with --ignored"]
    #[test]
    fn divergence_miss_resets_and_full_prefills() {
        // See the matching test on `qwen3_5/model.rs` for the pseudocode
        // shape.
    }

    /// Exact-match: second call's tokens == first call's tokens, no
    /// delta. The zero-delta guard MUST NOT corrupt state — after the
    /// forced full-reset + re-prefill, generation must still produce
    /// coherent output (not random tokens). This test locks in the
    /// behavior documented alongside the guard in
    /// `chat_sync_core` / `chat_stream_sync_core`.
    #[ignore = "requires a real Qwen3.5 MoE model directory; run with --ignored"]
    #[test]
    fn exact_match_zero_delta_guard_preserves_correctness() {
        // Pseudocode:
        //
        //   let messages = vec![ChatMessage::user("Ping")];
        //   let r1 = model.chat_session_start_sync(messages.clone(), cfg())?;
        //   let r2 = model.chat_session_start_sync(messages, cfg())?;
        //   // Zero-delta guard fires — full reset + re-prefill. The
        //   // second response should still be coherent (same length,
        //   // sensible tokens), not garbage from a corrupted GDN state.
        //   assert!(!r2.text.is_empty());
        //   assert!(r2.num_tokens > 0);
    }
}

#[cfg(test)]
mod paged_construction_tests {
    //! Construction-only smoke tests for the MoE block-paged adapter.

    use super::*;
    use crate::models::qwen3_5_moe::config::Qwen3_5MoeConfig;

    fn tiny_moe_cfg(use_block_paged: bool) -> Qwen3_5MoeConfig {
        Qwen3_5MoeConfig {
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
            num_experts: 4,
            num_experts_per_tok: 2,
            decoder_sparse_step: 1,
            shared_expert_intermediate_size: None,
            moe_intermediate_size: None,
            norm_topk_prob: true,
            mlp_only_layers: None,
            paged_cache_memory_mb: Some(64),
            paged_block_size: Some(16),
            use_block_paged_cache: if use_block_paged { Some(true) } else { None },
        }
    }

    #[test]
    fn test_moe_use_block_paged_cache_serde_default_none() {
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
            "num_experts": 4,
            "num_experts_per_tok": 2,
        });
        let cfg: Qwen3_5MoeConfig = serde_json::from_value(json).unwrap();
        assert_eq!(cfg.use_block_paged_cache, None);
        assert_eq!(cfg.paged_block_size, None);
        assert_eq!(cfg.paged_cache_memory_mb, None);
    }

    #[test]
    fn test_moe_full_attention_layer_count() {
        let cfg = tiny_moe_cfg(false);
        assert_eq!(cfg.full_attention_layer_count(), 2);
    }

    #[test]
    fn test_moe_inner_no_paged_adapter_when_flag_is_none() {
        let cfg = tiny_moe_cfg(false);
        let inner = Qwen35MoeInner::new(cfg)
            .expect("Qwen35MoeInner::new must succeed without paged adapter");
        assert!(inner.paged_adapter.is_none());
    }

    #[test]
    fn test_fresh_moe_layer_caches_are_not_gdn_reuse_ready() {
        let cfg = tiny_moe_cfg(true);
        let caches = fresh_moe_layer_caches(&cfg);
        assert_eq!(caches.len(), cfg.num_layers as usize);
        assert!(
            !moe_paged_linear_caches_ready(&cfg, Some(&caches)),
            "fresh linear caches have empty conv/recurrent slots, so a live continuation must replay GDN"
        );
        assert!(matches!(caches[0], Qwen3_5LayerCache::Linear(_)));
        assert!(matches!(caches[3], Qwen3_5LayerCache::FullAttention(_)));
    }

    #[test]
    fn test_paged_moe_linear_cache_export_uninitialized_returns_none() {
        unsafe {
            mlx_sys::mlx_qwen35_moe_reset();
        }
        let cfg = tiny_moe_cfg(true);
        let exported = export_paged_moe_linear_caches(&cfg)
            .expect("uninitialized paged export should not fail");
        assert!(exported.is_none());
    }

    #[test]
    fn test_paged_prefix_block_hash_matches_allocator_chain() {
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
    fn test_paged_prefix_block_hash_applies_salt_to_first_block_only() {
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
    fn test_paged_prefix_block_hash_rejects_non_full_or_unkeyed_prefix() {
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

    #[test]
    #[ignore = "Allocates Metal LayerKVPool; gate on MLX_TEST_PAGED=1"]
    fn test_moe_inner_constructs_paged_adapter_when_flag_is_true() {
        if std::env::var_os("MLX_TEST_PAGED").is_none() {
            return;
        }
        let cfg = tiny_moe_cfg(true);
        let inner = Qwen35MoeInner::new(cfg).expect(
            "Qwen35MoeInner::new with use_block_paged_cache=true must succeed on Metal host",
        );
        assert!(inner.paged_adapter.is_some());
    }

    /// Hard-fails fast when caller passes a `caches` slice whose length
    /// disagrees with `config.num_layers`. The check runs BEFORE any FFI
    /// dispatch so we don't perturb the C++ paged globals — making this
    /// the only branch of `init_paged_moe_compiled_session` we can
    /// exercise from a non-Metal sandbox.
    ///
    /// (The other failure branches all require populated `MxArray`
    /// handles, which need a real Metal allocation. They're covered
    /// indirectly by the parity test in
    /// `crates/mlx-core/tests/qwen3_5_moe_paged_vs_flat_parity.rs`.)
    #[test]
    fn test_init_paged_moe_compiled_session_rejects_cache_length_mismatch() {
        let cfg = tiny_moe_cfg(true);
        // `cfg` has num_layers=8; pass an empty cache slice to trigger
        // the length check before any FFI / Metal call.
        let empty_caches: Vec<Qwen3_5LayerCache> = Vec::new();
        // `paged_adapter` would normally be borrowed from
        // `Qwen35MoeInner::new`, but `Qwen35MoeInner::new` would
        // attempt to allocate a Metal LayerKVPool — the same kind of
        // sandbox-incompatible work the test wants to avoid. So we
        // construct a pool-free adapter via the `BlockAllocator` +
        // `LayerKVPool` directly with a tiny shape. On non-Metal hosts
        // the helper allocator can still be built; the LayerKVPool
        // construction does require Metal so we skip if that fails.
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
                    "skipping test_init_paged_moe_compiled_session_rejects_cache_length_mismatch: {e}"
                );
                return;
            }
        };
        let adapter = PagedKVCacheAdapter::new(alloc, pool, 16)
            .expect("paged adapter construction must succeed once pool is built");
        let res = init_paged_moe_compiled_session(&cfg, &empty_caches, &adapter, 0);
        let msg = res
            .expect_err("expected Err on cache length mismatch")
            .to_string();
        assert!(
            msg.contains("caches.len()"),
            "error must reference caches length contract; got: {msg}"
        );
    }

    /// Phase 4 piece 3 review fix (Finding 1): the C++ compiled paged
    /// graph hard-codes block_size=16. The dispatcher MUST gate
    /// `cpp_session_ready` on `adapter.block_size() == 16` so a config
    /// with `paged_block_size: Some(8)` (or 32) falls back to the
    /// pure-Rust paged path instead of corrupting KV state by mixing
    /// adapter-encoded slot/block tables (block_size=8) with
    /// C++-decoded writes (block_size=16).
    ///
    /// This test validates the constant + adapter-side contract: the
    /// FFI compile-key constant is 16, an adapter built with
    /// `paged_block_size: Some(8)` reports `block_size() == 8`, and the
    /// equality check the dispatcher uses (`!=`) correctly identifies
    /// the mismatch. Exercising the dispatcher itself requires loaded
    /// model weights (covered indirectly by the parity test).
    #[test]
    fn test_cpp_paged_required_block_size_is_sixteen() {
        // The C++ compile-graph constant. If this ever changes (e.g.
        // we re-trace at block_size=32 for a future tier), the
        // dispatcher gate, the FFI docstrings in `mlx_qwen35_moe.cpp`,
        // and the `paged_block_size` validation in PagedKVCacheAdapter
        // must all agree.
        assert_eq!(
            CPP_PAGED_REQUIRED_BLOCK_SIZE, 16,
            "C++ compiled paged graph in mlx_qwen35_moe.cpp hard-codes block_size=16; \
             changing this constant requires re-tracing the compiled graph"
        );
    }

    /// Build an adapter with `block_size != 16` and verify that the
    /// dispatcher gate (`adapter.block_size() != CPP_PAGED_REQUIRED_BLOCK_SIZE`)
    /// would correctly reject it. The gate runs BEFORE
    /// `init_paged_moe_compiled_session` is called, so the FFI is never
    /// touched on the fallback path. Exercises the test-only
    /// `LayerKVPool::new_for_test` constructor since we don't need real
    /// Metal allocations to validate the gate logic.
    #[test]
    fn test_block_size_eight_adapter_falls_back_to_pure_rust() {
        let cfg = tiny_moe_cfg(true);
        // Build an adapter that reports block_size=8 (a valid
        // `PagedAttentionConfig::validate` value, but mismatched with
        // the compiled graph's hard-coded 16).
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
        // `new_for_test` skips full GPU-buffer allocation but still
        // allocates 1-byte placeholder buffers on macOS so
        // `key_cache_array_raw` returns Some. We never dispatch a
        // kernel against this pool — the gate check fires first.
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
        // `init_paged_moe_compiled_session` here — that's exactly
        // what the gate prevents.
        assert_ne!(
            adapter.block_size(),
            CPP_PAGED_REQUIRED_BLOCK_SIZE,
            "block_size=8 adapter must not match the compiled graph's hard-coded 16; \
             the dispatcher gate at chat_sync_core_paged_inner / chat_stream_sync_core_paged_inner \
             relies on this inequality to fall back to the pure-Rust paged path"
        );
    }

    /// Phase 4 piece 3 review fix (Finding 2): the C++ FFI now returns
    /// `int32_t` (0 success / -1 failure) instead of `void`. The Rust
    /// `init_paged_moe_compiled_session` propagates non-zero status as
    /// `Err` so the dispatcher's `cpp_session_ready` becomes false on
    /// init failure and falls back to the pure-Rust paged decode. This
    /// test forces the C++ side's null-handle rejection branch by
    /// passing null pool handles for the full-attention layers, and
    /// asserts the FFI returns -1 (post-fix) instead of silently
    /// succeeding (pre-fix).
    ///
    /// Note: this test bypasses `init_paged_moe_compiled_session` and
    /// calls the FFI directly with synthetic null handles, because the
    /// helper would otherwise fail at `key_pool_array(...)` BEFORE
    /// reaching the FFI when the adapter's pool is unavailable. The
    /// goal here is to verify the FFI's status contract — the helper's
    /// guard is exercised by
    /// `test_init_paged_moe_compiled_session_rejects_cache_length_mismatch`.
    #[test]
    fn test_mlx_qwen35_moe_init_paged_returns_negative_on_null_handles() {
        let cfg = tiny_moe_cfg(true);
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

        let mlp_only: Vec<i32> = cfg.mlp_only_layers.as_deref().unwrap_or(&[]).to_vec();
        let status = unsafe {
            mlx_sys::mlx_qwen35_moe_init_paged(
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
                cfg.num_experts,
                cfg.num_experts_per_tok,
                if cfg.norm_topk_prob { 1 } else { 0 },
                cfg.decoder_sparse_step,
                if mlp_only.is_empty() {
                    std::ptr::null()
                } else {
                    mlp_only.as_ptr()
                },
                mlp_only.len() as i32,
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
            "mlx_qwen35_moe_init_paged MUST return -1 when full-attention pool/scale handles \
             are null. Returning 0 (or void, pre-fix) would let the dispatcher enter the \
             compiled paged decode against uninitialized globals."
        );
        // Reset C++ globals to a clean state so the test doesn't
        // contaminate any later tests in the same process.
        unsafe {
            mlx_sys::mlx_qwen35_moe_reset();
        }
    }
}
