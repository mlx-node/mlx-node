use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi_derive::napi;
use tokio::sync::Mutex as TokioMutex;
use tracing::{info, warn};

use crate::models::qwen3_5::model::{
    ChatConfig, ChatResult, ChatStreamChunk, ChatStreamHandle, IMAGE_TOKEN_ID,
    VISION_CACHE_MAX_ENTRIES, VisionCache, VisionCacheInner, combine_image_hashes,
    compute_num_image_tokens, extract_images_from_messages, get_rope_index, hash_image_bytes,
    inject_image_placeholders, merge_input_ids_with_image_features,
};
use crate::models::qwen3_5::processing::Qwen35VLImageProcessor;
use crate::models::qwen3_5::vision::Qwen3_5VisionEncoder;

use super::quantized_linear::LinearProj;
use crate::array::MxArray;
use crate::array::mask::create_causal_mask;
use crate::nn::{Embedding, Linear, RMSNorm};
use crate::sampling::{SamplingConfig, apply_repetition_penalty, check_repetition_cutoff, sample};
use crate::stream::{DeviceType, Stream, StreamContext};
use crate::tokenizer::{ChatMessage, Qwen3Tokenizer};
use crate::tools;

use super::config::Qwen3_5MoeConfig;
use super::decoder_layer::DecoderLayer;
use super::layer_cache::Qwen3_5LayerCache;
use super::persistence;

/// Maximum number of entries in the vision encoder cache before LRU eviction.
/// Process-wide mutex serializing the MoE compiled forward lifecycle.
///
/// The C++ MoE forward path uses process-wide globals (separate from dense).
/// This mutex prevents concurrent `generate()`/`chat()` calls from racing
/// on those globals when dispatched via `spawn_blocking`.
static MOE_COMPILED_MUTEX: TokioMutex<()> = TokioMutex::const_new(());

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

/// Qwen3.5 MoE Model -- hybrid linear/full attention with Mixture-of-Experts.
///
/// Supports C++ MoE forward path (non-compiled, builds fresh graph per step)
/// when weights are registered via `register_moe_weights_with_cpp`.
/// Falls back to Rust forward_inner path for test models without stored weights.
#[napi]
pub struct Qwen3_5MoeModel {
    config: Qwen3_5MoeConfig,
    pub(crate) embedding: Embedding,
    pub(crate) layers: Arc<RwLock<Vec<DecoderLayer>>>,
    pub(crate) final_norm: Arc<RwLock<RMSNorm>>,
    pub(crate) lm_head: Arc<RwLock<Option<LinearProj>>>,
    caches: Arc<RwLock<Option<Vec<Qwen3_5LayerCache>>>>,
    pub(crate) tokenizer: Option<Arc<Qwen3Tokenizer>>,
    fa_idx: usize,
    /// Optional vision encoder (set when loading a VLM)
    pub(crate) vision_encoder: Option<Arc<Qwen3_5VisionEncoder>>,
    /// Optional image processor (set when loading a VLM)
    pub(crate) image_processor: Option<Arc<Qwen35VLImageProcessor>>,
    /// Spatial merge size for VLM (typically 2)
    pub(crate) spatial_merge_size: Option<i32>,
    /// LRU cache for vision encoder embeddings, avoids re-encoding the same
    /// image in multi-turn VLM conversations.
    pub(crate) vision_cache: VisionCache,
}

#[napi]
impl Qwen3_5MoeModel {
    #[napi(constructor)]
    pub fn new(config: Qwen3_5MoeConfig) -> Result<Self> {
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

        info!(
            "Qwen3.5 MoE model created: {} layers, fa_idx={}, experts={}",
            config.num_layers, fa_idx, config.num_experts
        );

        Ok(Self {
            config,
            embedding,
            layers: Arc::new(RwLock::new(layers)),
            final_norm: Arc::new(RwLock::new(final_norm)),
            lm_head: Arc::new(RwLock::new(lm_head)),
            caches: Arc::new(RwLock::new(None)),
            tokenizer: None,
            fa_idx,
            vision_encoder: None,
            image_processor: None,
            spatial_merge_size: None,
            vision_cache: Arc::new(Mutex::new(VisionCacheInner {
                entries: HashMap::new(),
                generation: 0,
            })),
        })
    }

    #[napi]
    pub fn init_caches(&self) -> Result<()> {
        let caches = (0..self.config.num_layers as usize)
            .map(|i| {
                if self.config.is_linear_layer(i) {
                    Qwen3_5LayerCache::new_linear()
                } else {
                    Qwen3_5LayerCache::new_full_attention()
                }
            })
            .collect();
        let mut caches_guard = self
            .caches
            .write()
            .map_err(|_| Error::from_reason("Failed to acquire caches write lock"))?;
        *caches_guard = Some(caches);
        Ok(())
    }

    #[napi]
    pub fn reset_caches(&self) -> Result<()> {
        let mut caches_guard = self
            .caches
            .write()
            .map_err(|_| Error::from_reason("Failed to acquire caches write lock"))?;
        if let Some(ref mut caches) = *caches_guard {
            for cache in caches.iter_mut() {
                cache.reset();
            }
        }
        *caches_guard = None;
        Ok(())
    }

    #[napi]
    pub fn forward(&self, input_ids: &MxArray) -> Result<MxArray> {
        let hidden_states = self.embedding.forward(input_ids)?;
        self.forward_from_embeddings(&hidden_states)
    }

    #[napi]
    pub fn forward_with_cache(&self, input_ids: &MxArray) -> Result<MxArray> {
        {
            let caches_guard = self
                .caches
                .read()
                .map_err(|_| Error::from_reason("Failed to acquire caches read lock"))?;
            if caches_guard.is_none() {
                drop(caches_guard);
                self.init_caches()?;
            }
        }

        let hidden_states = self.embedding.forward(input_ids)?;
        self.forward_from_embeddings(&hidden_states)
    }

    #[napi]
    pub async fn load_pretrained(path: String) -> Result<Qwen3_5MoeModel> {
        persistence::load_pretrained(&path).await
    }

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

        // generate() assumes batch_size=1 (reshape to [1,1], item_at_int32(0)).
        // Reject multi-batch prompts early to avoid silently wrong results.
        let batch_size = prompt_tokens.shape_at(0)?;
        if batch_size != 1 {
            return Err(Error::from_reason(format!(
                "generate() only supports batch_size=1, got batch_size={}",
                batch_size
            )));
        }

        let embedding_weight = self.embedding.get_weight();
        let layers_arc = self.layers.clone();
        let final_norm_arc = self.final_norm.clone();
        let lm_head_arc = self.lm_head.clone();
        let caches_arc = self.caches.clone();
        let model_config = self.config.clone();
        let tokenizer = self.tokenizer.clone();
        let fa_idx = self.fa_idx;
        let prompt_tokens = prompt_tokens.clone();

        // Check if C++ MoE path will be used (weights loaded from safetensors).
        let use_cpp = unsafe { mlx_sys::mlx_qwen35_weight_count() } > 0;

        // Serialize MoE compiled lifecycle — prevents concurrent C++ global corruption
        let _moe_lock = if use_cpp {
            Some(MOE_COMPILED_MUTEX.lock().await)
        } else {
            None
        };

        napi::bindgen_prelude::spawn_blocking(move || {
            // Acquire all locks ONCE for the entire prefill+decode sequence
            let mut layers_guard = layers_arc
                .write()
                .map_err(|_| Error::from_reason("Failed to acquire layers write lock"))?;
            let mut caches_guard = caches_arc
                .write()
                .map_err(|_| Error::from_reason("Failed to acquire caches write lock"))?;
            let final_norm_guard = final_norm_arc
                .read()
                .map_err(|_| Error::from_reason("Failed to acquire final_norm read lock"))?;
            let lm_head_guard = lm_head_arc
                .read()
                .map_err(|_| Error::from_reason("Failed to acquire lm_head read lock"))?;

            // Reset and init caches (already holding write lock)
            if let Some(ref mut caches) = *caches_guard {
                for cache in caches.iter_mut() {
                    cache.reset();
                }
            }
            let new_caches = (0..model_config.num_layers as usize)
                .map(|i| {
                    if model_config.is_linear_layer(i) {
                        Qwen3_5LayerCache::new_linear()
                    } else {
                        Qwen3_5LayerCache::new_full_attention()
                    }
                })
                .collect();
            *caches_guard = Some(new_caches);

            let max_tokens = config.max_new_tokens;
            let sampling_config = Some(SamplingConfig {
                temperature: config.temperature,
                top_k: config.top_k,
                top_p: config.top_p,
                min_p: config.min_p,
            });

            let eos_id = model_config.eos_token_id as u32;
            let mut generated_tokens: Vec<u32> = Vec::new();
            let mut finish_reason = String::from("length");

            let embedding_weight_t = embedding_weight.transpose(Some(&[1, 0]))?;
            let generation_stream = Stream::new(DeviceType::Gpu);
            let model_size_bytes = model_config.estimate_memory_bytes() as usize;
            let _wired_ctx =
                crate::stream::WiredLimitContext::new(model_size_bytes, vec![generation_stream]);

            // StreamContext created ONCE for entire prefill+decode
            let _stream_ctx = StreamContext::new(generation_stream);

            // Profiler — covers both compiled and rust decode paths
            let mut profiler =
                crate::decode_profiler::DecodeProfiler::new("moe_generate", "qwen3_5_moe");
            profiler.set_prompt_tokens(prompt_tokens.shape_at(1).unwrap_or(0) as u32);
            profiler.snapshot_memory_before();

            // Prefill
            profiler.begin_prefill();
            let logits = forward_inner(
                &prompt_tokens,
                &embedding_weight,
                &mut layers_guard,
                &mut caches_guard,
                &final_norm_guard,
                &lm_head_guard,
                fa_idx,
                Some(&embedding_weight_t),
            )?;

            let seq_len = logits.shape_at(1)?;
            let last_logits = logits.slice_axis(1, seq_len - 1, seq_len)?;
            let last_logits = last_logits.squeeze(Some(&[1]))?;

            profiler.end_prefill();

            let mut y = sample(&last_logits, sampling_config)?;
            MxArray::async_eval_arrays(&[&y]);

            if use_cpp {
                // Guard ensures mlx_qwen35_moe_reset() is called even if `?` returns early.
                let _moe_guard = MoeResetGuard;
                // Initialize C++ MoE forward pass from prefill caches
                use mlx_sys as sys;
                let prefill_len = seq_len as i32;
                let max_kv_len = ((prefill_len + max_tokens + 255) / 256) * 256;
                let num_layers = model_config.num_layers as usize;
                let mut cache_ptrs: Vec<*mut sys::mlx_array> =
                    vec![std::ptr::null_mut(); num_layers * 2];
                if let Some(ref caches) = *caches_guard {
                    for (i, cache) in caches.iter().enumerate() {
                        let (p0, p1) = cache.export_ptrs();
                        cache_ptrs[i * 2] = p0;
                        cache_ptrs[i * 2 + 1] = p1;
                    }
                }
                let mlp_only: Vec<i32> = model_config
                    .mlp_only_layers
                    .as_deref()
                    .unwrap_or(&[])
                    .to_vec();
                // Drop non-cache locks — not needed during C++ MoE decode
                drop(layers_guard);
                drop(final_norm_guard);
                drop(lm_head_guard);
                // Keep caches_guard alive through init_from_prefill so cache_ptrs
                // (raw pointers into the cache MxArrays) remain valid.
                unsafe {
                    sys::mlx_qwen35_moe_init_from_prefill(
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
                        model_config.num_experts,
                        model_config.num_experts_per_tok,
                        if model_config.norm_topk_prob { 1 } else { 0 },
                        model_config.decoder_sparse_step,
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
                // C++ has copied arrays into its own globals — safe to release
                drop(caches_guard);

                // C++ decode loop (all locks dropped — C++ owns the state)
                profiler.set_label("moe_generate_compiled");

                for step in 0..max_tokens {
                    let next_y = if step + 1 < max_tokens {
                        profiler.begin("forward");
                        let next_ids = y.reshape(&[1, 1])?;
                        let logits = forward_moe_cpp(&next_ids, &embedding_weight)?;
                        profiler.end();

                        profiler.begin("sample");
                        let next_token = sample(&logits, sampling_config)?;
                        profiler.end();

                        profiler.begin("eval_caches");
                        eval_token_and_moe_caches(&next_token);
                        profiler.end();

                        Some(next_token)
                    } else {
                        None
                    };

                    profiler.begin("eval_token");
                    y.eval();
                    profiler.end();

                    profiler.begin("extract");
                    let token_id = y.item_at_int32(0)? as u32;
                    profiler.end();
                    profiler.mark_first_token();

                    generated_tokens.push(token_id);

                    if token_id == eos_id {
                        finish_reason = String::from("eos");
                        break;
                    }

                    match next_y {
                        Some(next) => y = next,
                        None => break,
                    }

                    profiler.step();

                    if (step + 1) % 256 == 0 {
                        crate::array::clear_cache();
                    }
                }

                profiler.snapshot_memory_after();
                profiler.report();
                // _moe_guard dropped here, calling mlx_qwen35_moe_reset()
            } else {
                // Rust fallback decode loop (outer StreamContext already active)
                profiler.set_label("moe_generate_rust");

                for step in 0..max_tokens {
                    let next_y = if step + 1 < max_tokens {
                        profiler.begin("forward");
                        let next_ids = y.reshape(&[1, 1])?;
                        let logits = forward_inner(
                            &next_ids,
                            &embedding_weight,
                            &mut layers_guard,
                            &mut caches_guard,
                            &final_norm_guard,
                            &lm_head_guard,
                            fa_idx,
                            Some(&embedding_weight_t),
                        )?;
                        let logits = logits.squeeze(Some(&[1]))?;
                        profiler.end();

                        profiler.begin("sample");
                        let next_token = sample(&logits, sampling_config)?;
                        profiler.end();

                        profiler.begin("async_eval");
                        MxArray::async_eval_arrays(&[&next_token]);
                        profiler.end();

                        Some(next_token)
                    } else {
                        None
                    };

                    profiler.begin("eval_token");
                    y.eval();
                    profiler.end();

                    profiler.begin("extract");
                    let token_id = y.item_at_int32(0)? as u32;
                    profiler.end();
                    profiler.mark_first_token();

                    generated_tokens.push(token_id);

                    if token_id == eos_id {
                        finish_reason = String::from("eos");
                        break;
                    }

                    match next_y {
                        Some(next) => y = next,
                        None => break,
                    }

                    profiler.step();

                    if (step + 1) % 256 == 0 {
                        crate::array::clear_cache();
                    }
                }

                profiler.snapshot_memory_after();
                profiler.report();

                drop(layers_guard);
                drop(caches_guard);
                drop(final_norm_guard);
                drop(lm_head_guard);
            }

            let text = if let Some(ref tok) = tokenizer {
                tok.decode_sync(&generated_tokens, true)
                    .unwrap_or_else(|e| {
                        warn!("Failed to decode generated tokens: {}", e);
                        String::new()
                    })
            } else {
                warn!("No tokenizer loaded - text decoding unavailable");
                String::new()
            };

            let num_tokens = generated_tokens.len() as u32;

            Ok(Qwen3_5MoeGenerationResult {
                tokens: generated_tokens,
                text,
                num_tokens,
                finish_reason,
            })
        })
        .await
        .map_err(|e| Error::from_reason(format!("Generation task failed: {}", e)))?
    }

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
            max_consecutive_tokens: None,
            max_ngram_repeats: None,
            ngram_size: None,
            tools: None,
            report_performance: None,
        });

        let tokenizer = self
            .tokenizer
            .clone()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))?;

        // Detect images in messages
        let has_images = messages
            .iter()
            .any(|m| m.images.as_ref().is_some_and(|imgs| !imgs.is_empty()));

        let embedding_weight = self.embedding.get_weight();
        let layers_arc = self.layers.clone();
        let final_norm_arc = self.final_norm.clone();
        let lm_head_arc = self.lm_head.clone();
        let caches_arc = self.caches.clone();
        let model_config = self.config.clone();
        let fa_idx = self.fa_idx;
        let tokenizer_for_decode = tokenizer.clone();

        // Clone vision fields if images present
        let vision_encoder_arc = if has_images {
            self.vision_encoder.clone()
        } else {
            None
        };
        let image_processor_arc = if has_images {
            self.image_processor.clone()
        } else {
            None
        };
        let spatial_merge_size = self.spatial_merge_size;
        let vision_cache = self.vision_cache.clone();

        // Check if C++ MoE path will be used (weights loaded from safetensors).
        let use_cpp = unsafe { mlx_sys::mlx_qwen35_weight_count() } > 0;

        // Capture start time BEFORE mutex + spawn_blocking so TTFT
        // reflects the full user-perceived latency.
        let report_perf = config.report_performance.unwrap_or(false);
        let generation_start = if report_perf {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // Serialize MoE compiled lifecycle — prevents concurrent C++ global corruption
        let _moe_lock = if use_cpp {
            Some(MOE_COMPILED_MUTEX.lock().await)
        } else {
            None
        };

        napi::bindgen_prelude::spawn_blocking(move || {
            let tool_defs = config.tools.as_deref();
            let tokens =
                tokenizer.apply_chat_template_sync(&messages, Some(true), tool_defs, None)?;

            let max_new_tokens = config.max_new_tokens.unwrap_or(2048);
            let repetition_penalty = config.repetition_penalty.unwrap_or(1.0);
            let repetition_context_size = config.repetition_context_size.unwrap_or(256);
            let max_consecutive_tokens = config.max_consecutive_tokens.unwrap_or(16);
            let max_ngram_repeats = config.max_ngram_repeats.unwrap_or(3);
            let ngram_size = config.ngram_size.unwrap_or(64);
            let sampling_config = Some(SamplingConfig {
                temperature: config.temperature,
                top_k: config.top_k.or(Some(20)), // Qwen3.5 recommends top_k=20
                top_p: config.top_p,
                min_p: config.min_p,
            });

            // Acquire all locks ONCE for the entire prefill+decode sequence
            let mut layers_guard = layers_arc
                .write()
                .map_err(|_| Error::from_reason("Failed to acquire layers write lock"))?;
            let mut caches_guard = caches_arc
                .write()
                .map_err(|_| Error::from_reason("Failed to acquire caches write lock"))?;
            let final_norm_guard = final_norm_arc
                .read()
                .map_err(|_| Error::from_reason("Failed to acquire final_norm read lock"))?;
            let lm_head_guard = lm_head_arc
                .read()
                .map_err(|_| Error::from_reason("Failed to acquire lm_head read lock"))?;

            // Reset and init caches (already holding write lock)
            if let Some(ref mut caches) = *caches_guard {
                for cache in caches.iter_mut() {
                    cache.reset();
                }
            }
            let new_caches = (0..model_config.num_layers as usize)
                .map(|i| {
                    if model_config.is_linear_layer(i) {
                        Qwen3_5LayerCache::new_linear()
                    } else {
                        Qwen3_5LayerCache::new_full_attention()
                    }
                })
                .collect();
            *caches_guard = Some(new_caches);

            let eos_id = model_config.eos_token_id as u32;
            let mut generated_tokens: Vec<u32> = Vec::new();
            let mut finish_reason = String::from("length");

            // Track token history for repetition penalty
            let mut token_history: Vec<u32> = tokens.clone();

            let embedding_weight_t = embedding_weight.transpose(Some(&[1, 0]))?;
            let generation_stream = Stream::new(DeviceType::Gpu);
            let model_size_bytes = model_config.estimate_memory_bytes() as usize;
            let _wired_ctx =
                crate::stream::WiredLimitContext::new(model_size_bytes, vec![generation_stream]);

            // StreamContext created ONCE for entire prefill+decode
            let _stream_ctx = StreamContext::new(generation_stream);

            let mut first_token_instant: Option<std::time::Instant> = None;

            // Profiler — covers both compiled and rust chat decode paths
            let mut profiler =
                crate::decode_profiler::DecodeProfiler::new("moe_chat", "qwen3_5_moe");
            profiler.set_prompt_tokens(tokens.len() as u32);
            profiler.snapshot_memory_before();

            // === VLM or text prefill branching ===
            profiler.begin_prefill();
            let (mut last_logits, seq_len) = if let (true, Some(vision_enc), Some(img_proc)) = (
                has_images,
                vision_encoder_arc.as_ref(),
                image_processor_arc.as_ref(),
            ) {
                // --- VLM path: process images and run VLM prefill ---
                let sms = spatial_merge_size.unwrap_or(2);
                let all_images = extract_images_from_messages(&messages);

                // Process images and inject placeholders
                let image_refs: Vec<&[u8]> = all_images.iter().map(|v| v.as_slice()).collect();
                let processed = img_proc.process_many(&image_refs)?;
                let num_image_tokens = compute_num_image_tokens(&processed.grid_thw(), sms)?;
                let final_tokens = inject_image_placeholders(&tokens, num_image_tokens);

                let input_ids =
                    MxArray::from_uint32(&final_tokens, &[1, final_tokens.len() as i64])?;

                // Update token history with final tokens (including image placeholders)
                token_history = final_tokens.clone();

                // VLM prefill using Rust path with M-RoPE position IDs
                let (logits, _rope_deltas) = vlm_prefill_moe(
                    &input_ids,
                    &all_images,
                    img_proc,
                    vision_enc,
                    sms,
                    &embedding_weight,
                    &mut layers_guard,
                    &mut caches_guard,
                    &final_norm_guard,
                    &lm_head_guard,
                    &model_config,
                    generation_stream,
                    fa_idx,
                    Some(&embedding_weight_t),
                    &vision_cache,
                )?;

                let vlm_seq_len = final_tokens.len() as i64;
                (logits, vlm_seq_len)
            } else {
                // --- Standard text prefill path ---
                let prompt = MxArray::from_uint32(&tokens, &[1, tokens.len() as i64])?;

                let logits = forward_inner(
                    &prompt,
                    &embedding_weight,
                    &mut layers_guard,
                    &mut caches_guard,
                    &final_norm_guard,
                    &lm_head_guard,
                    fa_idx,
                    Some(&embedding_weight_t),
                )?;

                let seq_len = logits.shape_at(1)?;
                let last_logits = logits.slice_axis(1, seq_len - 1, seq_len)?;
                let last_logits = last_logits.squeeze(Some(&[1]))?;
                (last_logits, seq_len)
            };
            profiler.end_prefill();

            // Apply repetition penalty to prefill logits
            if repetition_penalty != 1.0 && !token_history.is_empty() {
                last_logits = apply_repetition_penalty(
                    &last_logits,
                    &token_history,
                    repetition_penalty,
                    Some(repetition_context_size),
                )?;
            }

            let mut y = sample(&last_logits, sampling_config)?;
            MxArray::async_eval_arrays(&[&y]);

            if use_cpp {
                // Guard ensures mlx_qwen35_moe_reset() is called even if `?` returns early.
                let _moe_guard = MoeResetGuard;
                // Initialize C++ MoE forward pass from prefill caches
                use mlx_sys as sys;
                let prefill_len = seq_len as i32;
                let max_kv_len = ((prefill_len + max_new_tokens + 255) / 256) * 256;
                let num_layers = model_config.num_layers as usize;
                let mut cache_ptrs: Vec<*mut sys::mlx_array> =
                    vec![std::ptr::null_mut(); num_layers * 2];
                if let Some(ref caches) = *caches_guard {
                    for (i, cache) in caches.iter().enumerate() {
                        let (p0, p1) = cache.export_ptrs();
                        cache_ptrs[i * 2] = p0;
                        cache_ptrs[i * 2 + 1] = p1;
                    }
                }
                let mlp_only: Vec<i32> = model_config
                    .mlp_only_layers
                    .as_deref()
                    .unwrap_or(&[])
                    .to_vec();
                // Drop non-cache locks — not needed during C++ MoE decode
                drop(layers_guard);
                drop(final_norm_guard);
                drop(lm_head_guard);
                // Keep caches_guard alive through init_from_prefill so cache_ptrs
                // (raw pointers into the cache MxArrays) remain valid.
                unsafe {
                    sys::mlx_qwen35_moe_init_from_prefill(
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
                        model_config.num_experts,
                        model_config.num_experts_per_tok,
                        if model_config.norm_topk_prob { 1 } else { 0 },
                        model_config.decoder_sparse_step,
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
                // C++ has copied arrays into its own globals — safe to release
                drop(caches_guard);

                // C++ decode loop (all locks dropped — C++ owns the state)
                profiler.set_label("moe_chat_compiled");

                for step in 0..max_new_tokens {
                    profiler.begin("eval_token");
                    y.eval();
                    profiler.end();

                    profiler.begin("extract");
                    let token_id = y.item_at_int32(0)? as u32;
                    profiler.end();
                    profiler.mark_first_token();
                    if report_perf && first_token_instant.is_none() {
                        first_token_instant = Some(std::time::Instant::now());
                    }

                    generated_tokens.push(token_id);
                    token_history.push(token_id);

                    if token_id == eos_id {
                        finish_reason = String::from("eos");
                        break;
                    }

                    if let Some(reason) = check_repetition_cutoff(
                        &generated_tokens,
                        max_consecutive_tokens,
                        max_ngram_repeats,
                        ngram_size,
                    ) {
                        finish_reason = reason.to_string();
                        break;
                    }

                    if step + 1 < max_new_tokens {
                        profiler.begin("forward");
                        let next_ids = y.reshape(&[1, 1])?;
                        let mut logits = forward_moe_cpp(&next_ids, &embedding_weight)?;
                        profiler.end();

                        profiler.begin("rep_penalty");
                        if repetition_penalty != 1.0 {
                            logits = apply_repetition_penalty(
                                &logits,
                                &token_history,
                                repetition_penalty,
                                Some(repetition_context_size),
                            )?;
                        }
                        profiler.end();

                        profiler.begin("sample");
                        let next_token = sample(&logits, sampling_config)?;
                        profiler.end();

                        profiler.begin("eval_caches");
                        eval_token_and_moe_caches(&next_token);
                        profiler.end();

                        y = next_token;
                    } else {
                        break;
                    }

                    profiler.step();

                    if (step + 1) % 256 == 0 {
                        crate::array::clear_cache();
                    }
                }

                profiler.snapshot_memory_after();
                profiler.report();
                // _moe_guard dropped here, calling mlx_qwen35_moe_reset()
            } else {
                // Rust fallback decode loop
                profiler.set_label("moe_chat_rust");

                for step in 0..max_new_tokens {
                    profiler.begin("eval_token");
                    y.eval();
                    profiler.end();

                    profiler.begin("extract");
                    let token_id = y.item_at_int32(0)? as u32;
                    profiler.end();
                    profiler.mark_first_token();
                    if report_perf && first_token_instant.is_none() {
                        first_token_instant = Some(std::time::Instant::now());
                    }

                    generated_tokens.push(token_id);
                    token_history.push(token_id);

                    if token_id == eos_id {
                        finish_reason = String::from("eos");
                        break;
                    }

                    if let Some(reason) = check_repetition_cutoff(
                        &generated_tokens,
                        max_consecutive_tokens,
                        max_ngram_repeats,
                        ngram_size,
                    ) {
                        finish_reason = reason.to_string();
                        break;
                    }

                    if step + 1 < max_new_tokens {
                        profiler.begin("forward");
                        let next_ids = y.reshape(&[1, 1])?;
                        let logits = forward_inner(
                            &next_ids,
                            &embedding_weight,
                            &mut layers_guard,
                            &mut caches_guard,
                            &final_norm_guard,
                            &lm_head_guard,
                            fa_idx,
                            Some(&embedding_weight_t),
                        )?;
                        let mut logits = logits.squeeze(Some(&[1]))?;
                        profiler.end();

                        profiler.begin("rep_penalty");
                        if repetition_penalty != 1.0 {
                            logits = apply_repetition_penalty(
                                &logits,
                                &token_history,
                                repetition_penalty,
                                Some(repetition_context_size),
                            )?;
                        }
                        profiler.end();

                        profiler.begin("sample");
                        let next_token = sample(&logits, sampling_config)?;
                        profiler.end();

                        profiler.begin("async_eval");
                        MxArray::async_eval_arrays(&[&next_token]);
                        profiler.end();

                        y = next_token;
                    } else {
                        break;
                    }

                    profiler.step();

                    if (step + 1) % 256 == 0 {
                        crate::array::clear_cache();
                    }
                }

                profiler.snapshot_memory_after();
                profiler.report();

                drop(layers_guard);
                drop(caches_guard);
                drop(final_norm_guard);
                drop(lm_head_guard);
            }

            // Compute performance metrics if requested
            let performance = if let (Some(gen_start), Some(first_tok)) =
                (generation_start, first_token_instant)
            {
                let generation_end = std::time::Instant::now();
                let prompt_tokens = tokens.len() as f64;
                let gen_tokens = generated_tokens.len() as f64;
                let ttft_ms = first_tok.duration_since(gen_start).as_secs_f64() * 1000.0;
                let decode_ms = generation_end.duration_since(first_tok).as_secs_f64() * 1000.0;
                Some(crate::profiling::PerformanceMetrics {
                    ttft_ms,
                    prefill_tokens_per_second: if ttft_ms > 0.0 {
                        prompt_tokens / (ttft_ms / 1000.0)
                    } else {
                        0.0
                    },
                    decode_tokens_per_second: if decode_ms > 0.0 && gen_tokens > 1.0 {
                        (gen_tokens - 1.0) / (decode_ms / 1000.0)
                    } else {
                        0.0
                    },
                })
            } else {
                None
            };

            let text = tokenizer_for_decode
                .decode_sync(&generated_tokens, true)
                .unwrap_or_else(|e| {
                    warn!("Failed to decode generated tokens: {}", e);
                    String::new()
                });

            let num_tokens = generated_tokens.len() as u32;

            // Parse tool calls and thinking from the generated text
            let (clean_text, tool_calls, thinking) = tools::parse_generation_output(&text);

            // If we have valid tool calls, override finish reason
            let finish_reason = if tool_calls.iter().any(|tc| tc.status == "ok") {
                "tool_calls".to_string()
            } else {
                finish_reason
            };

            Ok(ChatResult {
                text: clean_text,
                tool_calls,
                thinking,
                num_tokens,
                finish_reason,
                raw_text: text,
                performance,
            })
        })
        .await
        .map_err(|e| Error::from_reason(format!("Chat task failed: {}", e)))?
    }

    /// Streaming chat API with tool calling support.
    ///
    /// Same as `chat()` but streams tokens one-by-one via the callback.
    /// Returns a `ChatStreamHandle` immediately; generation runs in background.
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
            max_consecutive_tokens: None,
            max_ngram_repeats: None,
            ngram_size: None,
            tools: None,
            report_performance: None,
        });

        let report_perf = config.report_performance.unwrap_or(false);

        let tokenizer = self
            .tokenizer
            .clone()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))?;

        // Detect images in messages
        let has_images = messages
            .iter()
            .any(|m| m.images.as_ref().is_some_and(|imgs| !imgs.is_empty()));

        let embedding_weight = self.embedding.get_weight();
        let layers_arc = self.layers.clone();
        let final_norm_arc = self.final_norm.clone();
        let lm_head_arc = self.lm_head.clone();
        let caches_arc = self.caches.clone();
        let model_config = self.config.clone();
        let fa_idx = self.fa_idx;
        let tokenizer_for_decode = tokenizer.clone();

        // Clone vision fields if images present
        let vision_encoder_arc = if has_images {
            self.vision_encoder.clone()
        } else {
            None
        };
        let image_processor_arc = if has_images {
            self.image_processor.clone()
        } else {
            None
        };
        let spatial_merge_size = self.spatial_merge_size;
        let vision_cache_stream = self.vision_cache.clone();

        // Check if C++ MoE path will be used (weights loaded from safetensors).
        let use_cpp = unsafe { mlx_sys::mlx_qwen35_weight_count() } > 0;

        // Capture start time BEFORE mutex + spawn_blocking so TTFT
        // reflects the full user-perceived latency.
        let generation_start = if report_perf {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // Serialize MoE compiled lifecycle — prevents concurrent C++ global corruption.
        // Acquire in async context, move into tokio::spawn so it stays held during generation.
        let moe_lock = if use_cpp {
            Some(MOE_COMPILED_MUTEX.lock().await)
        } else {
            None
        };

        let cancelled = Arc::new(AtomicBool::new(false));
        let cancelled_inner = cancelled.clone();

        let callback = Arc::new(callback);

        tokio::spawn(async move {
            // Hold the MoE lock for the duration of generation
            let _moe_lock = moe_lock;

            let callback_err = callback.clone();
            let result =
                napi::bindgen_prelude::spawn_blocking(move || -> std::result::Result<(), Error> {
                    let tool_defs = config.tools.as_deref();
                    let tokens = tokenizer.apply_chat_template_sync(
                        &messages,
                        Some(true),
                        tool_defs,
                        None,
                    )?;

                    let max_new_tokens = config.max_new_tokens.unwrap_or(2048);
                    let repetition_penalty = config.repetition_penalty.unwrap_or(1.0);
                    let repetition_context_size = config.repetition_context_size.unwrap_or(256);
                    let max_consecutive_tokens = config.max_consecutive_tokens.unwrap_or(16);
                    let max_ngram_repeats = config.max_ngram_repeats.unwrap_or(3);
                    let ngram_size = config.ngram_size.unwrap_or(64);
                    let sampling_config = Some(SamplingConfig {
                        temperature: config.temperature,
                        top_k: config.top_k.or(Some(20)),
                        top_p: config.top_p,
                        min_p: config.min_p,
                    });

                    // Acquire all locks ONCE for the entire prefill+decode sequence
                    let mut layers_guard = layers_arc
                        .write()
                        .map_err(|_| Error::from_reason("Failed to acquire layers write lock"))?;
                    let mut caches_guard = caches_arc
                        .write()
                        .map_err(|_| Error::from_reason("Failed to acquire caches write lock"))?;
                    let final_norm_guard = final_norm_arc.read().map_err(|_| {
                        Error::from_reason("Failed to acquire final_norm read lock")
                    })?;
                    let lm_head_guard = lm_head_arc
                        .read()
                        .map_err(|_| Error::from_reason("Failed to acquire lm_head read lock"))?;

                    // Reset and init caches
                    if let Some(ref mut caches) = *caches_guard {
                        for cache in caches.iter_mut() {
                            cache.reset();
                        }
                    }
                    let new_caches = (0..model_config.num_layers as usize)
                        .map(|i| {
                            if model_config.is_linear_layer(i) {
                                Qwen3_5LayerCache::new_linear()
                            } else {
                                Qwen3_5LayerCache::new_full_attention()
                            }
                        })
                        .collect();
                    *caches_guard = Some(new_caches);

                    let eos_id = model_config.eos_token_id as u32;
                    let mut generated_tokens: Vec<u32> = Vec::new();
                    let mut finish_reason = String::from("length");

                    // Track token history for repetition penalty
                    let mut token_history: Vec<u32> = tokens.clone();

                    let embedding_weight_t = embedding_weight.transpose(Some(&[1, 0]))?;
                    let generation_stream = Stream::new(DeviceType::Gpu);
                    let model_size_bytes = model_config.estimate_memory_bytes() as usize;
                    let _wired_ctx = crate::stream::WiredLimitContext::new(
                        model_size_bytes,
                        vec![generation_stream],
                    );

                    // StreamContext created ONCE for entire prefill+decode (MoE pattern)
                    let _stream_ctx = StreamContext::new(generation_stream);

                    let mut first_token_instant: Option<std::time::Instant> = None;

                    // Profiler — covers both compiled and rust chat_stream decode paths
                    let mut profiler = crate::decode_profiler::DecodeProfiler::new(
                        "moe_chat_stream",
                        "qwen3_5_moe",
                    );
                    profiler.set_prompt_tokens(tokens.len() as u32);
                    profiler.snapshot_memory_before();

                    // === VLM or text prefill branching ===
                    profiler.begin_prefill();
                    let (mut last_logits, seq_len) =
                        if let (true, Some(vision_enc), Some(img_proc)) = (
                            has_images,
                            vision_encoder_arc.as_ref(),
                            image_processor_arc.as_ref(),
                        ) {
                            // --- VLM path: process images and run VLM prefill ---
                            let sms = spatial_merge_size.unwrap_or(2);
                            let all_images = extract_images_from_messages(&messages);

                            // Process images and inject placeholders
                            let image_refs: Vec<&[u8]> =
                                all_images.iter().map(|v| v.as_slice()).collect();
                            let processed = img_proc.process_many(&image_refs)?;
                            let num_image_tokens =
                                compute_num_image_tokens(&processed.grid_thw(), sms)?;
                            let final_tokens = inject_image_placeholders(&tokens, num_image_tokens);

                            let input_ids = MxArray::from_uint32(
                                &final_tokens,
                                &[1, final_tokens.len() as i64],
                            )?;

                            // Update token history with final tokens (including image placeholders)
                            token_history = final_tokens.clone();

                            // VLM prefill using Rust path with M-RoPE position IDs
                            let (logits, _rope_deltas) = vlm_prefill_moe(
                                &input_ids,
                                &all_images,
                                img_proc,
                                vision_enc,
                                sms,
                                &embedding_weight,
                                &mut layers_guard,
                                &mut caches_guard,
                                &final_norm_guard,
                                &lm_head_guard,
                                &model_config,
                                generation_stream,
                                fa_idx,
                                Some(&embedding_weight_t),
                                &vision_cache_stream,
                            )?;

                            let vlm_seq_len = final_tokens.len() as i64;
                            (logits, vlm_seq_len)
                        } else {
                            // --- Standard text prefill path ---
                            let prompt = MxArray::from_uint32(&tokens, &[1, tokens.len() as i64])?;

                            let logits = forward_inner(
                                &prompt,
                                &embedding_weight,
                                &mut layers_guard,
                                &mut caches_guard,
                                &final_norm_guard,
                                &lm_head_guard,
                                fa_idx,
                                Some(&embedding_weight_t),
                            )?;

                            let seq_len = logits.shape_at(1)?;
                            let last_logits = logits.slice_axis(1, seq_len - 1, seq_len)?;
                            let last_logits = last_logits.squeeze(Some(&[1]))?;
                            (last_logits, seq_len)
                        };
                    profiler.end_prefill();

                    // Apply repetition penalty to prefill logits
                    if repetition_penalty != 1.0 && !token_history.is_empty() {
                        last_logits = apply_repetition_penalty(
                            &last_logits,
                            &token_history,
                            repetition_penalty,
                            Some(repetition_context_size),
                        )?;
                    }

                    let mut y = sample(&last_logits, sampling_config)?;
                    MxArray::async_eval_arrays(&[&y]);

                    if use_cpp {
                        // Guard ensures mlx_qwen35_moe_reset() is called even if `?` returns early.
                        let _moe_guard = MoeResetGuard;
                        // Initialize C++ MoE forward pass from prefill caches
                        use mlx_sys as sys;
                        let prefill_len = seq_len as i32;
                        let max_kv_len = ((prefill_len + max_new_tokens + 255) / 256) * 256;
                        let num_layers = model_config.num_layers as usize;
                        let mut cache_ptrs: Vec<*mut sys::mlx_array> =
                            vec![std::ptr::null_mut(); num_layers * 2];
                        if let Some(ref caches) = *caches_guard {
                            for (i, cache) in caches.iter().enumerate() {
                                let (p0, p1) = cache.export_ptrs();
                                cache_ptrs[i * 2] = p0;
                                cache_ptrs[i * 2 + 1] = p1;
                            }
                        }
                        let mlp_only: Vec<i32> = model_config
                            .mlp_only_layers
                            .as_deref()
                            .unwrap_or(&[])
                            .to_vec();
                        // Drop non-cache locks — not needed during C++ MoE decode
                        drop(layers_guard);
                        drop(final_norm_guard);
                        drop(lm_head_guard);
                        // Keep caches_guard alive through init_from_prefill so cache_ptrs
                        // (raw pointers into the cache MxArrays) remain valid.
                        unsafe {
                            sys::mlx_qwen35_moe_init_from_prefill(
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
                                model_config.num_experts,
                                model_config.num_experts_per_tok,
                                if model_config.norm_topk_prob { 1 } else { 0 },
                                model_config.decoder_sparse_step,
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
                        // C++ has copied arrays into its own globals — safe to release
                        drop(caches_guard);

                        // C++ decode loop (all locks dropped — C++ owns the state)
                        profiler.set_label("moe_chat_stream_compiled");
                        for step in 0..max_new_tokens {
                            y.eval();
                            let token_id = y.item_at_int32(0)? as u32;
                            profiler.mark_first_token();
                            if report_perf && first_token_instant.is_none() {
                                first_token_instant = Some(std::time::Instant::now());
                            }
                            generated_tokens.push(token_id);
                            token_history.push(token_id);

                            if cancelled_inner.load(Ordering::Relaxed) {
                                finish_reason = String::from("cancelled");
                                break;
                            }

                            // Decode and stream this token
                            let token_text = tokenizer_for_decode
                                .decode_sync(&[token_id], true)
                                .unwrap_or_default();
                            callback.call(
                                Ok(ChatStreamChunk {
                                    text: token_text,
                                    done: false,
                                    finish_reason: None,
                                    tool_calls: None,
                                    thinking: None,
                                    num_tokens: None,
                                    raw_text: None,
                                    performance: None,
                                }),
                                ThreadsafeFunctionCallMode::NonBlocking,
                            );

                            if token_id == eos_id {
                                finish_reason = String::from("eos");
                                break;
                            }

                            if let Some(reason) = check_repetition_cutoff(
                                &generated_tokens,
                                max_consecutive_tokens,
                                max_ngram_repeats,
                                ngram_size,
                            ) {
                                finish_reason = reason.to_string();
                                break;
                            }

                            if step + 1 >= max_new_tokens {
                                break;
                            }

                            let next_ids = y.reshape(&[1, 1])?;
                            let mut logits = forward_moe_cpp(&next_ids, &embedding_weight)?;
                            if repetition_penalty != 1.0 {
                                logits = apply_repetition_penalty(
                                    &logits,
                                    &token_history,
                                    repetition_penalty,
                                    Some(repetition_context_size),
                                )?;
                            }
                            let next_token = sample(&logits, sampling_config)?;
                            eval_token_and_moe_caches(&next_token);
                            y = next_token;

                            profiler.step();

                            if (step + 1) % 256 == 0 {
                                crate::array::clear_cache();
                            }
                        }
                        profiler.snapshot_memory_after();
                        profiler.report();
                        // _moe_guard dropped here, calling mlx_qwen35_moe_reset()
                    } else {
                        // Rust fallback decode loop (locks held for entire loop)
                        profiler.set_label("moe_chat_stream_rust");
                        for step in 0..max_new_tokens {
                            y.eval();
                            let token_id = y.item_at_int32(0)? as u32;
                            profiler.mark_first_token();
                            if report_perf && first_token_instant.is_none() {
                                first_token_instant = Some(std::time::Instant::now());
                            }
                            generated_tokens.push(token_id);
                            token_history.push(token_id);

                            if cancelled_inner.load(Ordering::Relaxed) {
                                finish_reason = String::from("cancelled");
                                break;
                            }

                            // Decode and stream this token
                            let token_text = tokenizer_for_decode
                                .decode_sync(&[token_id], true)
                                .unwrap_or_default();
                            callback.call(
                                Ok(ChatStreamChunk {
                                    text: token_text,
                                    done: false,
                                    finish_reason: None,
                                    tool_calls: None,
                                    thinking: None,
                                    num_tokens: None,
                                    raw_text: None,
                                    performance: None,
                                }),
                                ThreadsafeFunctionCallMode::NonBlocking,
                            );

                            if token_id == eos_id {
                                finish_reason = String::from("eos");
                                break;
                            }

                            if let Some(reason) = check_repetition_cutoff(
                                &generated_tokens,
                                max_consecutive_tokens,
                                max_ngram_repeats,
                                ngram_size,
                            ) {
                                finish_reason = reason.to_string();
                                break;
                            }

                            if step + 1 >= max_new_tokens {
                                break;
                            }

                            let next_ids = y.reshape(&[1, 1])?;
                            let logits = forward_inner(
                                &next_ids,
                                &embedding_weight,
                                &mut layers_guard,
                                &mut caches_guard,
                                &final_norm_guard,
                                &lm_head_guard,
                                fa_idx,
                                Some(&embedding_weight_t),
                            )?;
                            let mut logits = logits.squeeze(Some(&[1]))?;
                            if repetition_penalty != 1.0 {
                                logits = apply_repetition_penalty(
                                    &logits,
                                    &token_history,
                                    repetition_penalty,
                                    Some(repetition_context_size),
                                )?;
                            }
                            let next_token = sample(&logits, sampling_config)?;
                            MxArray::async_eval_arrays(&[&next_token]);
                            y = next_token;

                            profiler.step();

                            if (step + 1) % 256 == 0 {
                                crate::array::clear_cache();
                            }
                        }

                        profiler.snapshot_memory_after();
                        profiler.report();

                        drop(layers_guard);
                        drop(caches_guard);
                        drop(final_norm_guard);
                        drop(lm_head_guard);
                    }

                    // Compute performance metrics if requested
                    let performance = if let (Some(gen_start), Some(first_tok)) =
                        (generation_start, first_token_instant)
                    {
                        let generation_end = std::time::Instant::now();
                        let prompt_tokens = tokens.len() as f64;
                        let gen_tokens = generated_tokens.len() as f64;
                        let ttft_ms = first_tok.duration_since(gen_start).as_secs_f64() * 1000.0;
                        let decode_ms =
                            generation_end.duration_since(first_tok).as_secs_f64() * 1000.0;
                        Some(crate::profiling::PerformanceMetrics {
                            ttft_ms,
                            prefill_tokens_per_second: if ttft_ms > 0.0 {
                                prompt_tokens / (ttft_ms / 1000.0)
                            } else {
                                0.0
                            },
                            decode_tokens_per_second: if decode_ms > 0.0 && gen_tokens > 1.0 {
                                (gen_tokens - 1.0) / (decode_ms / 1000.0)
                            } else {
                                0.0
                            },
                        })
                    } else {
                        None
                    };

                    // Decode full text for final chunk
                    let text = tokenizer_for_decode
                        .decode_sync(&generated_tokens, true)
                        .unwrap_or_else(|e| {
                            warn!("Failed to decode generated tokens: {}", e);
                            String::new()
                        });

                    let num_tokens = generated_tokens.len() as u32;

                    // Parse tool calls and thinking from the generated text
                    let (clean_text, tool_calls, thinking) = tools::parse_generation_output(&text);

                    // If we have valid tool calls, override finish reason
                    let finish_reason = if tool_calls.iter().any(|tc| tc.status == "ok") {
                        "tool_calls".to_string()
                    } else {
                        finish_reason
                    };

                    // Send final done chunk
                    callback.call(
                        Ok(ChatStreamChunk {
                            text: clean_text,
                            done: true,
                            finish_reason: Some(finish_reason),
                            tool_calls: Some(tool_calls),
                            thinking,
                            num_tokens: Some(num_tokens),
                            raw_text: Some(text),
                            performance,
                        }),
                        ThreadsafeFunctionCallMode::NonBlocking,
                    );

                    Ok(())
                })
                .await;

            match result {
                Ok(Ok(())) => {} // Success — final chunk already sent via callback
                Ok(Err(e)) => {
                    // Inner closure error (tokenization, lock, array ops, etc.)
                    callback_err.call(Err(e), ThreadsafeFunctionCallMode::NonBlocking);
                }
                Err(e) => {
                    // JoinError (panic in spawn_blocking)
                    callback_err.call(
                        Err(Error::from_reason(format!(
                            "Chat stream task panicked: {}",
                            e
                        ))),
                        ThreadsafeFunctionCallMode::NonBlocking,
                    );
                }
            }
        });

        Ok(ChatStreamHandle { cancelled })
    }

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
        h = layers[i].forward(&h, mask, cache, None)?;
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
    all_images: &[Vec<u8>],
    image_processor: &Qwen35VLImageProcessor,
    vision_encoder: &Qwen3_5VisionEncoder,
    spatial_merge_size: i32,
    text_model_embedding: &MxArray,
    layers_guard: &mut [DecoderLayer],
    caches_guard: &mut Option<Vec<Qwen3_5LayerCache>>,
    final_norm_guard: &RMSNorm,
    lm_head_guard: &Option<LinearProj>,
    _model_config: &Qwen3_5MoeConfig,
    generation_stream: Stream,
    fa_idx: usize,
    embedding_weight_t: Option<&MxArray>,
    vision_cache: &VisionCache,
) -> Result<(MxArray, i64)> {
    use crate::array::clear_cache;

    // === STEP 1: Compute vision features (with hash cache) ===
    let individual_hashes: Vec<u64> = all_images.iter().map(|img| hash_image_bytes(img)).collect();
    let combined_hash = combine_image_hashes(&individual_hashes);

    // Check cache for pre-computed vision features + grid_thw
    let cached = {
        let mut cache = vision_cache.lock().unwrap();
        cache.generation += 1;
        let lru_gen = cache.generation;
        if let Some((features, grid, lru)) = cache.entries.get_mut(&combined_hash) {
            *lru = lru_gen;
            tracing::debug!("MoE vision cache HIT for hash {:016x}", combined_hash);
            Some((features.clone(), grid.clone()))
        } else {
            None
        }
    };

    let (vision_features, grid) = if let Some((features, grid)) = cached {
        (features, grid)
    } else {
        let image_refs: Vec<&[u8]> = all_images.iter().map(|v| v.as_slice()).collect();
        let processed = image_processor.process_many(&image_refs)?;
        let grid = processed.grid_thw();
        let pv = processed.pixel_values();
        let pv_shape = pv.shape()?;
        let pv_5d = pv.reshape(&[1, pv_shape[0], pv_shape[1], pv_shape[2], pv_shape[3]])?;

        let features = {
            let _stream_ctx = StreamContext::new(generation_stream);
            vision_encoder.forward(&pv_5d, &grid)?
        };

        {
            let mut cache = vision_cache.lock().unwrap();
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
        tracing::debug!("MoE vision cache MISS for hash {:016x}", combined_hash);

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
        "MoE VLM prefill: seq_len={}, rope_deltas={}",
        inputs_embeds.shape_at(1)?,
        rope_deltas
    );

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
            h = layers_guard[i].forward(&h, mask, cache, layer_pos)?;
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

impl Qwen3_5MoeModel {
    fn forward_from_embeddings(&self, hidden_states: &MxArray) -> Result<MxArray> {
        self.forward_from_embeddings_with_positions(hidden_states, None)
    }

    fn create_fa_mask(
        &self,
        hidden_states: &MxArray,
        caches: &Option<Vec<Qwen3_5LayerCache>>,
    ) -> Result<Option<MxArray>> {
        let seq_len = hidden_states.shape_at(1)?;
        if seq_len <= 1 && caches.is_some() {
            return Ok(None);
        }

        let offset = caches
            .as_ref()
            .map(|c| c[self.fa_idx].offset())
            .unwrap_or(0);

        create_causal_mask(seq_len as i32, Some(offset), None).map(Some)
    }

    /// Get embeddings for input IDs (used by VLM).
    pub fn get_embeddings(&self, input_ids: &MxArray) -> Result<MxArray> {
        self.embedding.forward(input_ids)
    }

    /// Forward pass from pre-computed embeddings with M-RoPE position IDs (VLM mode).
    pub fn forward_from_embeddings_with_positions(
        &self,
        hidden_states: &MxArray,
        position_ids: Option<&MxArray>,
    ) -> Result<MxArray> {
        let mut h = hidden_states.clone();

        let mut layers_guard = self
            .layers
            .write()
            .map_err(|_| Error::from_reason("Failed to acquire layers write lock"))?;
        let mut caches_guard = self
            .caches
            .write()
            .map_err(|_| Error::from_reason("Failed to acquire caches write lock"))?;

        let fa_mask = self.create_fa_mask(hidden_states, &caches_guard)?;

        let num_layers = layers_guard.len();
        for i in 0..num_layers {
            let mask = if layers_guard[i].is_linear() {
                None
            } else {
                fa_mask.as_ref()
            };

            let cache = caches_guard.as_mut().map(|c| &mut c[i]);
            // Only full attention layers receive position_ids
            let layer_pos = if layers_guard[i].is_linear() {
                None
            } else {
                position_ids
            };
            h = layers_guard[i].forward(&h, mask, cache, layer_pos)?;
        }

        drop(layers_guard);
        drop(caches_guard);

        let final_norm_guard = self
            .final_norm
            .read()
            .map_err(|_| Error::from_reason("Failed to acquire final_norm read lock"))?;
        let h = final_norm_guard.forward(&h)?;
        drop(final_norm_guard);

        let lm_head_guard = self
            .lm_head
            .read()
            .map_err(|_| Error::from_reason("Failed to acquire lm_head read lock"))?;
        match &*lm_head_guard {
            Some(head) => head.forward(&h),
            None => {
                let weight = self.embedding.get_weight();
                let weight_t = weight.transpose(Some(&[1, 0]))?;
                h.matmul(&weight_t)
            }
        }
    }

    /// Initialize M-RoPE on all full attention layers (VLM mode).
    pub fn init_mrope_layers(
        &self,
        mrope_section: Vec<i32>,
        rope_theta: f64,
        max_position_embeddings: i32,
    ) -> Result<()> {
        let rope_dims = self.config.rope_dims();
        let mut layers_guard = self
            .layers
            .write()
            .map_err(|_| Error::from_reason("Failed to acquire layers write lock"))?;
        for layer in layers_guard.iter_mut() {
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

    /// Set the vision encoder (wraps in Arc).
    pub(crate) fn set_vision_encoder(&mut self, enc: Qwen3_5VisionEncoder) {
        self.vision_encoder = Some(Arc::new(enc));
    }

    /// Set the image processor.
    pub(crate) fn set_image_processor(&mut self, proc: Qwen35VLImageProcessor) {
        self.image_processor = Some(Arc::new(proc));
    }

    /// Set the spatial merge size.
    pub(crate) fn set_spatial_merge_size(&mut self, size: i32) {
        self.spatial_merge_size = Some(size);
    }
}
