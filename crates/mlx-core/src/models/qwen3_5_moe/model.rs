use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi_derive::napi;
use tracing::{info, warn};

use crate::model_thread::{ResponseTx, StreamTx};
use crate::models::paddleocr_vl::processing::ProcessedImages;
use crate::models::qwen3_5::model::{
    ChatConfig, ChatResult, ChatStreamChunk, ChatStreamHandle, VisionCache, VisionCacheInner,
    compute_image_cache_key, compute_num_image_tokens, extract_images_from_messages,
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
    apply_all_penalties, compute_performance_metrics, extract_chat_params, finalize_chat_result,
    save_cache_state_direct, verify_cache_prefix_direct,
};
use crate::nn::{Embedding, Linear, RMSNorm};
use crate::sampling::{SamplingConfig, sample};
use crate::stream::{DeviceType, Stream, StreamContext};
use crate::tokenizer::{ChatMessage, Qwen3Tokenizer, ToolDefinition};

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
    /// Training state owned by the model thread.
    /// Created when `InitTraining` command is received, destroyed when training ends.
    pub(crate) training_state: Option<crate::training_state::ModelThreadTrainingState>,
}

/// Commands dispatched from NAPI methods to the dedicated model thread.
pub(crate) enum Qwen35MoeCmd {
    Chat {
        messages: Vec<ChatMessage>,
        config: ChatConfig,
        reply: ResponseTx<ChatResult>,
    },
    ChatStream {
        messages: Vec<ChatMessage>,
        config: ChatConfig,
        stream_tx: StreamTx<ChatStreamChunk>,
        cancelled: Arc<AtomicBool>,
    },
    Generate {
        prompt_tokens: MxArray,
        config: Qwen3_5MoeGenerationConfig,
        reply: ResponseTx<Qwen3_5MoeGenerationResult>,
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
pub(crate) fn handle_qwen35_moe_cmd(inner: &mut Qwen35MoeInner, cmd: Qwen35MoeCmd) {
    match cmd {
        Qwen35MoeCmd::Chat {
            messages,
            config,
            reply,
        } => {
            let _ = reply.send(inner.chat_sync(messages, config));
        }
        Qwen35MoeCmd::ChatStream {
            messages,
            config,
            stream_tx,
            cancelled,
        } => {
            inner.chat_stream_sync(messages, config, stream_tx, cancelled);
        }
        Qwen35MoeCmd::Generate {
            prompt_tokens,
            config,
            reply,
        } => {
            let _ = reply.send(inner.generate_sync(prompt_tokens, config));
        }
        Qwen35MoeCmd::TakeCache { reply } => {
            let _ = reply.send(Ok(inner.take_cache_sync()));
        }
        Qwen35MoeCmd::SetCache { cache, reply } => {
            let _ = reply.send(inner.set_cache_sync(cache));
        }
        Qwen35MoeCmd::InitCaches { reply } => {
            let _ = reply.send(inner.init_caches_sync());
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

        info!(
            "Qwen3.5 MoE inner created: {} layers, fa_idx={}, experts={}",
            config.num_layers, fa_idx, config.num_experts
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
            "qwen3_5_moe",
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

    /// Chat synchronous (runs on model thread).
    pub(crate) fn chat_sync(
        &mut self,
        messages: Vec<ChatMessage>,
        config: ChatConfig,
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

        let eos_id = self.config.eos_token_id as u32;
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
            // VLM cache reuse: same images, incremental text-only prefill
            let prompt = MxArray::from_uint32(&prefill_tokens, &[1, prefill_tokens.len() as i64])?;

            let logits = forward_inner(
                &prompt,
                &embedding_weight,
                &mut self.layers,
                &mut self.caches,
                &self.final_norm,
                &self.lm_head,
                fa_idx,
                Some(&embedding_weight_t),
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
            // Standard text prefill
            let prompt = MxArray::from_uint32(&prefill_tokens, &[1, prefill_tokens.len() as i64])?;

            let logits = forward_inner(
                &prompt,
                &embedding_weight,
                &mut self.layers,
                &mut self.caches,
                &self.final_norm,
                &self.lm_head,
                fa_idx,
                Some(&embedding_weight_t),
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

            let starts_in_thinking = enable_thinking.unwrap_or(true);
            let mut reasoning_tracker = chat_common::ReasoningTracker::new(
                starts_in_thinking,
                p.thinking_token_budget,
                think_end_id,
            );

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
                }
            }
            // _moe_guard dropped here, calling mlx_qwen35_moe_reset()
        } else {
            // Rust fallback decode loop
            profiler.set_label("moe_chat_rust");

            let starts_in_thinking = enable_thinking.unwrap_or(true);
            let mut reasoning_tracker = chat_common::ReasoningTracker::new(
                starts_in_thinking,
                p.thinking_token_budget,
                think_end_id,
            );

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

        finalize_chat_result(
            &tokenizer,
            &generated_tokens,
            finish_reason,
            think_end_id,
            think_end_str.as_deref(),
            performance,
            p.include_reasoning,
            enable_thinking.unwrap_or(true),
        )
    }

    /// Streaming chat synchronous (runs on model thread).
    pub(crate) fn chat_stream_sync(
        &mut self,
        messages: Vec<ChatMessage>,
        config: ChatConfig,
        stream_tx: StreamTx<ChatStreamChunk>,
        cancelled: Arc<AtomicBool>,
    ) {
        let cb = StreamSender(stream_tx.clone());
        let result = self.chat_stream_sync_inner(messages, config, &cb, &cancelled);
        if let Err(e) = result {
            let _ = stream_tx.send(Err(e));
        }
    }

    fn chat_stream_sync_inner(
        &mut self,
        messages: Vec<ChatMessage>,
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

        let eos_id = self.config.eos_token_id as u32;
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
            let prompt = MxArray::from_uint32(&prefill_tokens, &[1, prefill_tokens.len() as i64])?;

            let logits = forward_inner(
                &prompt,
                &embedding_weight,
                &mut self.layers,
                &mut self.caches,
                &self.final_norm,
                &self.lm_head,
                fa_idx,
                Some(&embedding_weight_t),
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
            let prompt = MxArray::from_uint32(&prefill_tokens, &[1, prefill_tokens.len() as i64])?;
            let logits = forward_inner(
                &prompt,
                &embedding_weight,
                &mut self.layers,
                &mut self.caches,
                &self.final_norm,
                &self.lm_head,
                fa_idx,
                Some(&embedding_weight_t),
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

            let mut reasoning_tracker = chat_common::ReasoningTracker::new(
                starts_in_thinking,
                p.thinking_token_budget,
                think_end_id,
            );
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
                }
            }
            // _moe_guard dropped here
        } else {
            profiler.set_label("moe_chat_stream_rust");

            let mut reasoning_tracker = chat_common::ReasoningTracker::new(
                starts_in_thinking,
                p.thinking_token_budget,
                think_end_id,
            );
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
            cb.call(
                Ok(ChatStreamChunk {
                    text: residual,
                    done: false,
                    finish_reason: None,
                    tool_calls: None,
                    thinking: None,
                    num_tokens: None,
                    raw_text: None,
                    performance: None,
                    is_reasoning: Some(last_is_reasoning),
                }),
                ThreadsafeFunctionCallMode::NonBlocking,
            );
        }

        let num_tokens = generated_tokens.len() as u32;

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
                fa_idx,
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

        let params_clone: HashMap<String, MxArray> =
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
        crate::utils::safetensors::save_safetensors(&safetensors_path, &params_clone, metadata)?;
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
    pub(crate) model_id: u64,
}

#[napi]
impl Qwen3_5MoeModel {
    /// Initialize caches for incremental generation.
    #[napi]
    pub fn init_caches(&self) -> Result<()> {
        crate::model_thread::send_and_block(&self.thread, |reply| Qwen35MoeCmd::InitCaches {
            reply,
        })
    }

    /// Reset all caches.
    #[napi]
    pub fn reset_caches(&self) -> Result<()> {
        crate::model_thread::send_and_block(&self.thread, |reply| Qwen35MoeCmd::ResetCaches {
            reply,
        })
    }

    /// Take the KV cache from the model, returning a `PromptCache` handle.
    #[napi]
    pub fn take_cache(&self) -> Option<crate::models::qwen3_5::prompt_cache::PromptCache> {
        crate::model_thread::send_and_block(&self.thread, |reply| Qwen35MoeCmd::TakeCache { reply })
            .ok()?
    }

    /// Restore a previously taken `PromptCache` into the model.
    #[napi]
    pub fn set_cache(
        &self,
        cache: &mut crate::models::qwen3_5::prompt_cache::PromptCache,
    ) -> Result<()> {
        // Validate before sending (these checks don't need model-thread state)
        if cache.model_type() != "qwen3_5_moe" {
            return Err(Error::from_reason(format!(
                "Cache type '{}' doesn't match model type 'qwen3_5_moe'",
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
            "qwen3_5_moe",
            cache.num_layers(),
            cache.image_cache_key(),
            cache.rope_deltas(),
            cache.model_id(),
        );
        crate::model_thread::send_and_block(&self.thread, |reply| Qwen35MoeCmd::SetCache {
            cache: owned_cache,
            reply,
        })
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

    /// Chat API with tool calling support.
    ///
    /// Dispatches to the dedicated model thread and awaits the result.
    #[napi]
    pub async fn chat(
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

        crate::model_thread::send_and_await(&self.thread, |reply| Qwen35MoeCmd::Chat {
            messages,
            config,
            reply,
        })
        .await
    }

    /// Streaming chat API with tool calling support.
    ///
    /// Dispatches to the dedicated model thread. Tokens stream back via
    /// an mpsc channel bridged to the JS callback. Returns a `ChatStreamHandle`
    /// immediately; generation runs on the model thread.
    /// Call `handle.cancel()` to abort generation early.
    #[napi(
        ts_args_type = "messages: ChatMessage[], config: ChatConfig | null, callback: (err: Error | null, chunk: ChatStreamChunk) => void"
    )]
    pub async fn chat_stream(
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

        // Create mpsc channel to bridge model thread -> tokio task -> JS callback
        let (stream_tx, mut stream_rx) =
            tokio::sync::mpsc::unbounded_channel::<napi::Result<ChatStreamChunk>>();

        // Send streaming command to model thread
        self.thread.send(Qwen35MoeCmd::ChatStream {
            messages,
            config,
            stream_tx,
            cancelled: cancelled_inner,
        })?;

        // Spawn tokio task that reads from stream_rx and calls the JS callback
        let callback = Arc::new(callback);
        tokio::spawn(async move {
            while let Some(result) = stream_rx.recv().await {
                callback.call(result, ThreadsafeFunctionCallMode::NonBlocking);
            }
        });

        Ok(ChatStreamHandle { cancelled })
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
