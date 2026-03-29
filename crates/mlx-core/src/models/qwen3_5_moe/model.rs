use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use futures::TryFutureExt;
use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi_derive::napi;
use tokio::sync::Mutex as TokioMutex;
use tracing::{info, warn};

use crate::models::paddleocr_vl::processing::ProcessedImages;
use crate::models::qwen3_5::model::{
    ChatConfig, ChatResult, ChatStreamChunk, ChatStreamHandle, VisionCache, VisionCacheInner,
    compute_image_cache_key, compute_num_image_tokens, extract_images_from_messages,
    inject_image_placeholders, vlm_prepare_vision_features,
};
use crate::models::qwen3_5::processing::Qwen35VLImageProcessor;
use crate::models::qwen3_5::vision::Qwen3_5VisionEncoder;

use super::quantized_linear::LinearProj;
use crate::array::MxArray;
use crate::array::mask::create_causal_mask;
use crate::models::qwen3::{BatchGenerationResult, GenerationConfig, GenerationResult};
use crate::nn::{Embedding, Linear, RMSNorm};
use crate::sampling::{
    SamplingConfig, apply_frequency_penalty, apply_presence_penalty, apply_repetition_penalty,
    check_repetition_cutoff, sample,
};
use crate::stream::{DeviceType, Stream, StreamContext};
use crate::tokenizer::{ChatMessage, Qwen3Tokenizer, ToolDefinition};
use crate::tools;

use super::config::Qwen3_5MoeConfig;
use super::decoder_layer::DecoderLayer;
use super::layer_cache::Qwen3_5LayerCache;
use super::persistence;

// Import the shared model ID counter from the dense module — dense and MoE
// share the same C++ weight map, so IDs must be globally unique.
use crate::models::qwen3_5::model::{QWEN35_MODEL_ID_COUNTER, acquire_compiled_weight_guard};

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
    /// Token history for KV cache reuse across chat() calls.
    /// Stores the full token sequence (template + generated) from the last call.
    cached_token_history: Arc<RwLock<Vec<u32>>>,
    /// Image cache key for VLM cache reuse (None for text-only conversations).
    cached_image_key: Arc<RwLock<Option<u64>>>,
    /// Rope deltas from VLM prefill, for cache reuse M-RoPE correction
    cached_rope_deltas: Arc<RwLock<Option<i32>>>,
    /// Unique model instance ID for compiled path ownership.
    pub(crate) model_id: u64,
    /// Serializes cache state access during generation.
    generation_lock: Arc<TokioMutex<()>>,
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
            cached_token_history: Arc::new(RwLock::new(Vec::new())),
            cached_image_key: Arc::new(RwLock::new(None)),
            cached_rope_deltas: Arc::new(RwLock::new(None)),
            model_id: QWEN35_MODEL_ID_COUNTER.fetch_add(1, Ordering::Relaxed),
            generation_lock: Arc::new(TokioMutex::new(())),
        })
    }

    /// Take the KV cache from the model, returning a `PromptCache` handle.
    ///
    /// The cache is moved out of the model — calling `takeCache()` twice
    /// returns `null` the second time. Pass the cache back via `setCache()`
    /// before the next `chat()` call for incremental prefill.
    #[napi]
    pub fn take_cache(&self) -> Option<crate::models::qwen3_5::prompt_cache::PromptCache> {
        let _guard = self.generation_lock.try_lock().ok()?;
        let mut caches_guard = self.caches.write().ok()?;
        let token_history_guard = self.cached_token_history.read().ok()?;
        let caches = caches_guard.take()?;
        if token_history_guard.is_empty() {
            // No generation has happened yet — put caches back
            *caches_guard = Some(caches);
            return None;
        }
        let image_key = self.cached_image_key.read().ok().and_then(|g| *g);
        let rope_deltas = self.cached_rope_deltas.read().ok().and_then(|g| *g);
        Some(crate::models::qwen3_5::prompt_cache::PromptCache::new(
            caches,
            token_history_guard.clone(),
            "qwen3_5_moe",
            self.config.num_layers as usize,
            image_key,
            rope_deltas,
            self.model_id,
        ))
    }

    /// Restore a previously taken `PromptCache` into the model.
    ///
    /// On the next `chat()` call with `reuseCache: true`, the model will
    /// prefix-match the new tokens against the cache and only prefill the delta.
    #[napi]
    pub fn set_cache(
        &self,
        cache: &mut crate::models::qwen3_5::prompt_cache::PromptCache,
    ) -> Result<()> {
        let _guard = self
            .generation_lock
            .try_lock()
            .map_err(|_| Error::from_reason("Cannot set cache while generation is in progress"))?;
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
        let restored_caches = cache.take_caches().ok_or_else(|| {
            Error::from_reason("PromptCache is empty (already consumed or disposed)")
        })?;
        let mut caches_guard = self
            .caches
            .write()
            .map_err(|_| Error::from_reason("Failed to acquire caches write lock"))?;
        let mut token_history_guard = self
            .cached_token_history
            .write()
            .map_err(|_| Error::from_reason("Failed to acquire token history write lock"))?;
        *caches_guard = Some(restored_caches);
        *token_history_guard = cache.token_history().to_vec();
        if let Ok(mut ik) = self.cached_image_key.write() {
            *ik = cache.image_cache_key();
        }
        if let Ok(mut rd) = self.cached_rope_deltas.write() {
            *rd = cache.rope_deltas();
        }
        Ok(())
    }

    #[napi]
    pub fn init_caches(&self) -> Result<()> {
        let _guard = self.generation_lock.try_lock().map_err(|_| {
            Error::from_reason("Cannot init caches while generation is in progress")
        })?;
        self.init_caches_inner()
    }

    fn init_caches_inner(&self) -> Result<()> {
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
        self.clear_reuse_state();
        Ok(())
    }

    #[napi]
    pub fn reset_caches(&self) -> Result<()> {
        let _guard = self.generation_lock.try_lock().map_err(|_| {
            Error::from_reason("Cannot reset caches while generation is in progress")
        })?;
        self.reset_caches_inner()
    }

    fn reset_caches_inner(&self) -> Result<()> {
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
        self.clear_reuse_state();
        Ok(())
    }

    fn clear_reuse_state(&self) {
        if let Ok(mut th) = self.cached_token_history.write() {
            th.clear();
        }
        if let Ok(mut ik) = self.cached_image_key.write() {
            *ik = None;
        }
        if let Ok(mut rd) = self.cached_rope_deltas.write() {
            *rd = None;
        }
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
    pub async fn load(path: String) -> Result<Qwen3_5MoeModel> {
        persistence::load(&path).await
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

        // Hold generation lock for the entire lifecycle.
        let gen_lock = self.generation_lock.clone();
        let _gen_guard = gen_lock.lock().await;

        let embedding_weight = self.embedding.get_weight();
        let layers_arc = self.layers.clone();
        let final_norm_arc = self.final_norm.clone();
        let lm_head_arc = self.lm_head.clone();
        let caches_arc = self.caches.clone();
        let model_config = self.config.clone();
        let tokenizer = self.tokenizer.clone();
        let fa_idx = self.fa_idx;
        let prompt_tokens = prompt_tokens.clone();
        let model_id = self.model_id;

        // Check if C++ MoE path will be used (weights belong to this model).
        let use_cpp = unsafe { mlx_sys::mlx_qwen35_get_model_id() } == model_id;

        // Serialize MoE compiled lifecycle — prevents concurrent C++ global corruption
        let _moe_lock = if use_cpp {
            Some(MOE_COMPILED_MUTEX.lock().await)
        } else {
            None
        };

        napi::bindgen_prelude::spawn_blocking(move || {
            let _weight_guard = if use_cpp {
                acquire_compiled_weight_guard(model_id)
            } else {
                None
            };
            let use_cpp = _weight_guard.is_some();

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
                        finish_reason = String::from("stop");
                        break;
                    }

                    match next_y {
                        Some(next) => y = next,
                        None => break,
                    }

                    profiler.step();

                    if (step + 1) % 256 == 0 {
                        crate::array::synchronize_and_clear_cache();
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
                        finish_reason = String::from("stop");
                        break;
                    }

                    match next_y {
                        Some(next) => y = next,
                        None => break,
                    }

                    profiler.step();

                    if (step + 1) % 256 == 0 {
                        crate::array::synchronize_and_clear_cache();
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
            presence_penalty: None,
            presence_context_size: None,
            frequency_penalty: None,
            frequency_context_size: None,
            max_consecutive_tokens: None,
            max_ngram_repeats: None,
            ngram_size: None,
            tools: None,
            enable_thinking: None,
            report_performance: None,
            reuse_cache: None,
        });

        let reuse_cache = config.reuse_cache.unwrap_or(true);

        let gen_lock = self.generation_lock.clone();
        let _gen_guard = gen_lock.lock().await;

        let tokenizer = self
            .tokenizer
            .clone()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))?;

        let has_images = messages
            .iter()
            .any(|m| m.images.as_ref().is_some_and(|imgs| !imgs.is_empty()));

        let embedding_weight = self.embedding.get_weight();
        let layers_arc = self.layers.clone();
        let final_norm_arc = self.final_norm.clone();
        let lm_head_arc = self.lm_head.clone();
        let caches_arc = self.caches.clone();
        let cached_token_history_arc = self.cached_token_history.clone();
        let cached_image_key_arc = self.cached_image_key.clone();
        let cached_rope_deltas_arc = self.cached_rope_deltas.clone();
        let model_config = self.config.clone();
        let fa_idx = self.fa_idx;
        let think_end_id = tokenizer.think_end_id();
        let think_end_str = tokenizer.think_end_str().map(|s| s.to_string());
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
        let model_id = self.model_id;

        // Check if C++ MoE path will be used (weights belong to this model).
        let use_cpp = unsafe { mlx_sys::mlx_qwen35_get_model_id() } == model_id;

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
            let _weight_guard = if use_cpp {
                acquire_compiled_weight_guard(model_id)
            } else {
                None
            };
            let use_cpp = _weight_guard.is_some();

            let tool_defs = config.tools.as_deref();
            let enable_thinking = config.enable_thinking;
            let tokens = tokenizer.apply_chat_template_sync(
                &messages,
                Some(true),
                tool_defs,
                enable_thinking,
            )?;

            let max_new_tokens = config.max_new_tokens.unwrap_or(2048);
            let repetition_penalty = config.repetition_penalty.unwrap_or(1.0);
            let repetition_context_size = config.repetition_context_size.unwrap_or(256);
            let presence_penalty = config.presence_penalty.unwrap_or(0.0);
            let presence_context_size = config.presence_context_size.unwrap_or(20);
            let frequency_penalty = config.frequency_penalty.unwrap_or(0.0);
            let frequency_context_size = config.frequency_context_size.unwrap_or(20);
            let max_consecutive_tokens = config.max_consecutive_tokens.unwrap_or(16);
            let max_ngram_repeats = config.max_ngram_repeats.unwrap_or(3);
            let ngram_size = config.ngram_size.unwrap_or(64);
            let sampling_config = Some(SamplingConfig {
                temperature: config.temperature,
                top_k: config.top_k,
                top_p: config.top_p,
                min_p: config.min_p,
            });

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

            // === VLM: pre-compute expanded tokens and image cache key for prefix matching ===
            let (expanded_tokens, image_cache_key, vlm_processed) =
                if let (true, Some(_), Some(img_proc)) = (
                    has_images,
                    vision_encoder_arc.as_ref(),
                    image_processor_arc.as_ref(),
                ) {
                    let sms = spatial_merge_size.unwrap_or(2);
                    let all_images = extract_images_from_messages(&messages);
                    let image_refs: Vec<&[u8]> = all_images.iter().map(|v| v.as_slice()).collect();
                    let processed = img_proc.process_many(&image_refs)?;
                    let num_image_tokens = compute_num_image_tokens(&processed.grid_thw(), sms)?;
                    let final_tokens = inject_image_placeholders(&tokens, num_image_tokens);
                    let key = compute_image_cache_key(&all_images);
                    (Some(final_tokens), key, Some(processed))
                } else {
                    (None, 0u64, None)
                };

            // For prefix matching, use expanded tokens (with image placeholders) for VLM
            let tokens_for_matching = expanded_tokens.as_deref().unwrap_or(&tokens);

            // === Cache reuse: prefix verification ===
            let cached_token_history_guard = cached_token_history_arc
                .read()
                .map_err(|_| Error::from_reason("Failed to read cached token history"))?;
            let cached_prefix_len = if reuse_cache {
                let cached = &*cached_token_history_guard;
                if has_images {
                    // VLM: also check that image_cache_key matches
                    let cached_img_key = cached_image_key_arc
                        .read()
                        .map_err(|_| Error::from_reason("Failed to read cached image key"))?;
                    if let Some(cached_key) = *cached_img_key {
                        if cached_key == image_cache_key
                            && !cached.is_empty()
                            && tokens_for_matching.len() >= cached.len()
                            && tokens_for_matching[..cached.len()] == cached[..]
                            && caches_guard.is_some()
                        {
                            cached.len()
                        } else {
                            0
                        }
                    } else {
                        0
                    }
                } else {
                    // Text-only: existing logic
                    if !cached.is_empty()
                        && tokens.len() >= cached.len()
                        && tokens[..cached.len()] == cached[..]
                        && caches_guard.is_some()
                    {
                        cached.len()
                    } else {
                        0
                    }
                }
            } else {
                0
            };
            drop(cached_token_history_guard);

            let prefill_tokens = if cached_prefix_len > 0 {
                // Prefix matches — incremental prefill (only new tokens)
                info!(
                    "Cache reuse: {} cached tokens, {} new tokens to prefill (vlm={})",
                    cached_prefix_len,
                    tokens_for_matching.len() - cached_prefix_len,
                    has_images
                );
                tokens_for_matching[cached_prefix_len..].to_vec()
            } else {
                // No match — full reset + full prefill
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
                tokens.clone()
            };

            // Zero-delta guard: also reset cached_prefix_len for VLM routing.
            let (prefill_tokens, cached_prefix_len) = if prefill_tokens.is_empty() {
                info!("Zero-delta cache hit: resetting caches for full re-prefill");
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
                let tokens = if has_images {
                    expanded_tokens.as_ref().unwrap_or(&tokens).clone()
                } else {
                    tokens.clone()
                };
                (tokens, 0)
            } else {
                (prefill_tokens, cached_prefix_len)
            };

            let eos_id = model_config.eos_token_id as u32;
            let mut generated_tokens: Vec<u32> = Vec::new();
            let mut finish_reason = String::from("length");

            // Track token history for repetition penalty
            let mut token_history: Vec<u32> = if let Some(ref et) = expanded_tokens {
                et.clone()
            } else {
                tokens.clone()
            };

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
            profiler.set_prompt_tokens(prefill_tokens.len() as u32);
            profiler.snapshot_memory_before();

            // === VLM or text prefill branching ===
            profiler.begin_prefill();
            let (mut last_logits, seq_len) = if has_images && cached_prefix_len > 0 {
                // --- VLM cache reuse: same images, incremental text-only prefill ---
                let expanded = expanded_tokens.as_ref().unwrap();
                let prompt =
                    MxArray::from_uint32(&prefill_tokens, &[1, prefill_tokens.len() as i64])?;

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
                // seq_len is the TOTAL expanded tokens (cached + new)
                (last_logits, expanded.len() as i64)
            } else if let (true, Some(vision_enc), Some(_)) = (
                has_images,
                vision_encoder_arc.as_ref(),
                image_processor_arc.as_ref(),
            ) {
                // --- VLM path: full VLM prefill (no cache reuse) ---
                let sms = spatial_merge_size.unwrap_or(2);
                let final_tokens = expanded_tokens.as_ref().unwrap();
                let processed = vlm_processed.as_ref().unwrap();

                let input_ids =
                    MxArray::from_uint32(final_tokens, &[1, final_tokens.len() as i64])?;

                // VLM prefill using Rust path with M-RoPE position IDs
                let (logits, rope_deltas) = vlm_prefill_moe(
                    &input_ids,
                    image_cache_key,
                    processed,
                    vision_enc,
                    sms,
                    &embedding_weight,
                    &mut layers_guard,
                    &mut caches_guard,
                    &final_norm_guard,
                    &lm_head_guard,
                    generation_stream,
                    fa_idx,
                    Some(&embedding_weight_t),
                    &vision_cache,
                )?;

                // Save rope_deltas for cache reuse on subsequent turns
                if let Ok(mut rd) = cached_rope_deltas_arc.write() {
                    *rd = Some(rope_deltas as i32);
                }

                let vlm_seq_len = final_tokens.len() as i64;
                (logits, vlm_seq_len)
            } else {
                // --- Standard text prefill path ---
                let prompt =
                    MxArray::from_uint32(&prefill_tokens, &[1, prefill_tokens.len() as i64])?;

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
                // seq_len for the C++ init is the TOTAL tokens (cached + new)
                (last_logits, tokens.len() as i64)
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
            if presence_penalty != 0.0 {
                last_logits = apply_presence_penalty(
                    &last_logits,
                    &token_history,
                    presence_penalty,
                    Some(presence_context_size),
                )?;
            }
            if frequency_penalty != 0.0 {
                last_logits = apply_frequency_penalty(
                    &last_logits,
                    &token_history,
                    frequency_penalty,
                    Some(frequency_context_size),
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

                // Apply M-RoPE offset correction AFTER init_from_prefill (which sets
                // g_moe_offset_int = prefill_len). Must come after, not before, or the
                // correction gets overwritten.
                if has_images
                    && let Ok(rd) = cached_rope_deltas_arc.read()
                    && let Some(delta) = *rd
                {
                    unsafe {
                        mlx_sys::mlx_qwen35_moe_adjust_offset(delta);
                    }
                }

                // For text-only conversations, clear any stale cached rope deltas
                if !has_images && let Ok(mut rd) = cached_rope_deltas_arc.write() {
                    *rd = None;
                }

                // C++ decode loop (all locks dropped — C++ owns the state)
                // Pipelined: submit forward(N+1) BEFORE eval(N) so GPU
                // computes the next step while CPU extracts the current token.
                // Rep penalty uses token_history as-of graph build time (one
                // token behind), matching Python mlx-lm's pipelining behavior.
                profiler.set_label("moe_chat_compiled");
                for step in 0..max_new_tokens {
                    // Build and submit graph for step N+1 before waiting for N
                    let next_y = if step + 1 < max_new_tokens {
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
                        if presence_penalty != 0.0 {
                            logits = apply_presence_penalty(
                                &logits,
                                &token_history,
                                presence_penalty,
                                Some(presence_context_size),
                            )?;
                        }
                        if frequency_penalty != 0.0 {
                            logits = apply_frequency_penalty(
                                &logits,
                                &token_history,
                                frequency_penalty,
                                Some(frequency_context_size),
                            )?;
                        }
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

                    // Wait for step N (GPU already computing N+1)
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
                        finish_reason = String::from("stop");
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

                    match next_y {
                        Some(next) => y = next,
                        None => break,
                    }

                    profiler.step();

                    if (step + 1) % 256 == 0 {
                        crate::array::synchronize_and_clear_cache();
                    }
                }

                profiler.snapshot_memory_after();
                profiler.report();

                // === Export caches from C++ before MoeResetGuard drops ===
                if reuse_cache {
                    let num_layers = model_config.num_layers as usize;
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
                            let mut lc = if model_config.is_linear_layer(i) {
                                Qwen3_5LayerCache::new_linear()
                            } else {
                                Qwen3_5LayerCache::new_full_attention()
                            };
                            lc.import_ptrs(p0, p1, cache_offset);
                            new_caches.push(lc);
                        }
                        let mut cg = caches_arc.write().map_err(|_| {
                            Error::from_reason("Failed to acquire caches lock for cache export")
                        })?;
                        *cg = Some(new_caches);
                    }
                }
                // _moe_guard dropped here, calling mlx_qwen35_moe_reset()
            } else {
                // Rust fallback decode loop (pipelined)
                profiler.set_label("moe_chat_rust");
                for step in 0..max_new_tokens {
                    // Build and submit graph for step N+1 before waiting for N
                    let next_y = if step + 1 < max_new_tokens {
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
                        if presence_penalty != 0.0 {
                            logits = apply_presence_penalty(
                                &logits,
                                &token_history,
                                presence_penalty,
                                Some(presence_context_size),
                            )?;
                        }
                        if frequency_penalty != 0.0 {
                            logits = apply_frequency_penalty(
                                &logits,
                                &token_history,
                                frequency_penalty,
                                Some(frequency_context_size),
                            )?;
                        }
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

                    // Wait for step N (GPU already computing N+1)
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
                        finish_reason = String::from("stop");
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

                    match next_y {
                        Some(next) => y = next,
                        None => break,
                    }

                    profiler.step();

                    if (step + 1) % 256 == 0 {
                        crate::array::synchronize_and_clear_cache();
                    }
                }

                profiler.snapshot_memory_after();
                profiler.report();

                drop(layers_guard);
                drop(caches_guard);
                drop(final_norm_guard);
                drop(lm_head_guard);
            }

            // === Save token history and image key for cache reuse on next call ===
            if reuse_cache {
                // For VLM, save the expanded token history (with image placeholders)
                let mut full_history = if let Some(ref et) = expanded_tokens {
                    et.clone()
                } else {
                    tokens.clone()
                };
                // Only include tokens that were actually forwarded through the model.
                // When stopped at max_tokens ("length"), the last token was never forwarded
                // (the pipelined loop skips forward on the final step).
                let history_tokens = if finish_reason == "length" && !generated_tokens.is_empty() {
                    &generated_tokens[..generated_tokens.len() - 1]
                } else {
                    &generated_tokens
                };
                full_history.extend_from_slice(history_tokens);
                if let Ok(mut th) = cached_token_history_arc.write() {
                    *th = full_history;
                }
                // Save image cache key (Some for VLM, None for text-only)
                if let Ok(mut ik) = cached_image_key_arc.write() {
                    *ik = if has_images {
                        Some(image_cache_key)
                    } else {
                        None
                    };
                }
            } else {
                // reuseCache: false — clear all cache state to free GPU memory
                if let Ok(mut cg) = caches_arc.write() {
                    *cg = None;
                }
                if let Ok(mut th) = cached_token_history_arc.write() {
                    th.clear();
                }
                if let Ok(mut ik) = cached_image_key_arc.write() {
                    *ik = None;
                }
                if let Ok(mut rd) = cached_rope_deltas_arc.write() {
                    *rd = None;
                }
            }

            // Compute performance metrics if requested
            let performance = if let (Some(gen_start), Some(first_tok)) =
                (generation_start, first_token_instant)
            {
                let generation_end = std::time::Instant::now();
                let actual_prefill_toks = prefill_tokens.len() as f64;
                let gen_tokens = generated_tokens.len() as f64;
                let ttft_ms = first_tok.duration_since(gen_start).as_secs_f64() * 1000.0;
                let decode_ms = generation_end.duration_since(first_tok).as_secs_f64() * 1000.0;
                Some(crate::profiling::PerformanceMetrics {
                    ttft_ms,
                    prefill_tokens_per_second: if ttft_ms > 0.0 {
                        actual_prefill_toks / (ttft_ms / 1000.0)
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

            let think_tag = if tools::has_think_end_token(&generated_tokens, think_end_id) {
                think_end_str.as_deref()
            } else {
                None
            };
            let (clean_text, tool_calls, thinking) = tools::split_at_think_end(&text, think_tag);

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
            presence_penalty: None,
            presence_context_size: None,
            frequency_penalty: None,
            frequency_context_size: None,
            max_consecutive_tokens: None,
            max_ngram_repeats: None,
            ngram_size: None,
            tools: None,
            enable_thinking: None,
            report_performance: None,
            reuse_cache: None,
        });

        let reuse_cache = config.reuse_cache.unwrap_or(true);
        let report_perf = config.report_performance.unwrap_or(false);

        let gen_guard = Arc::clone(&self.generation_lock).lock_owned().await;

        let tokenizer = self
            .tokenizer
            .clone()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))?;

        let has_images = messages
            .iter()
            .any(|m| m.images.as_ref().is_some_and(|imgs| !imgs.is_empty()));

        let embedding_weight = self.embedding.get_weight();
        let layers_arc = self.layers.clone();
        let final_norm_arc = self.final_norm.clone();
        let lm_head_arc = self.lm_head.clone();
        let caches_arc = self.caches.clone();
        let cached_token_history_arc = self.cached_token_history.clone();
        let cached_image_key_arc = self.cached_image_key.clone();
        let cached_rope_deltas_arc = self.cached_rope_deltas.clone();
        let model_config = self.config.clone();
        let fa_idx = self.fa_idx;
        let think_end_id_stream = tokenizer.think_end_id();
        let think_end_str_stream = tokenizer.think_end_str().map(|s| s.to_string());
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
        let model_id = self.model_id;

        // Check if C++ MoE path will be used (weights belong to this model).
        let use_cpp = unsafe { mlx_sys::mlx_qwen35_get_model_id() } == model_id;

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
            let _gen_guard = gen_guard;
            let _moe_lock = moe_lock;

            let callback_err = callback.clone();
            let result =
                napi::bindgen_prelude::spawn_blocking(move || -> std::result::Result<(), Error> {
                    let _weight_guard = if use_cpp {
                        acquire_compiled_weight_guard(model_id)
                    } else {
                        None
                    };
                    let use_cpp = _weight_guard.is_some();

                    let tool_defs = config.tools.as_deref();
                    let enable_thinking = config.enable_thinking;
                    let tokens = tokenizer.apply_chat_template_sync(
                        &messages,
                        Some(true),
                        tool_defs,
                        enable_thinking,
                    )?;

                    let max_new_tokens = config.max_new_tokens.unwrap_or(2048);
                    let repetition_penalty = config.repetition_penalty.unwrap_or(1.0);
                    let repetition_context_size = config.repetition_context_size.unwrap_or(256);
                    let presence_penalty = config.presence_penalty.unwrap_or(0.0);
                    let presence_context_size = config.presence_context_size.unwrap_or(20);
                    let frequency_penalty = config.frequency_penalty.unwrap_or(0.0);
                    let frequency_context_size = config.frequency_context_size.unwrap_or(20);
                    let max_consecutive_tokens = config.max_consecutive_tokens.unwrap_or(16);
                    let max_ngram_repeats = config.max_ngram_repeats.unwrap_or(3);
                    let ngram_size = config.ngram_size.unwrap_or(64);
                    let sampling_config = Some(SamplingConfig {
                        temperature: config.temperature,
                        top_k: config.top_k,
                        top_p: config.top_p,
                        min_p: config.min_p,
                    });

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

                    // === VLM: pre-compute expanded tokens and image cache key for prefix matching ===
                    let (expanded_tokens, image_cache_key, vlm_processed) =
                        if let (true, Some(_), Some(img_proc)) = (
                            has_images,
                            vision_encoder_arc.as_ref(),
                            image_processor_arc.as_ref(),
                        ) {
                            let sms = spatial_merge_size.unwrap_or(2);
                            let all_images = extract_images_from_messages(&messages);
                            let image_refs: Vec<&[u8]> =
                                all_images.iter().map(|v| v.as_slice()).collect();
                            let processed = img_proc.process_many(&image_refs)?;
                            let num_image_tokens =
                                compute_num_image_tokens(&processed.grid_thw(), sms)?;
                            let final_tokens = inject_image_placeholders(&tokens, num_image_tokens);
                            let key = compute_image_cache_key(&all_images);
                            (Some(final_tokens), key, Some(processed))
                        } else {
                            (None, 0u64, None)
                        };

                    // For prefix matching, use expanded tokens (with image placeholders) for VLM
                    let tokens_for_matching = expanded_tokens.as_deref().unwrap_or(&tokens);

                    // === Cache reuse: prefix verification ===
                    let cached_token_history_guard = cached_token_history_arc
                        .read()
                        .map_err(|_| Error::from_reason("Failed to read cached token history"))?;
                    let cached_prefix_len = if reuse_cache {
                        let cached = &*cached_token_history_guard;
                        if has_images {
                            // VLM: also check that image_cache_key matches
                            let cached_img_key = cached_image_key_arc.read().map_err(|_| {
                                Error::from_reason("Failed to read cached image key")
                            })?;
                            if let Some(cached_key) = *cached_img_key {
                                if cached_key == image_cache_key
                                    && !cached.is_empty()
                                    && tokens_for_matching.len() >= cached.len()
                                    && tokens_for_matching[..cached.len()] == cached[..]
                                    && caches_guard.is_some()
                                {
                                    cached.len()
                                } else {
                                    0
                                }
                            } else {
                                0
                            }
                        } else {
                            // Text-only: existing logic
                            if !cached.is_empty()
                                && tokens.len() >= cached.len()
                                && tokens[..cached.len()] == cached[..]
                                && caches_guard.is_some()
                            {
                                cached.len()
                            } else {
                                0
                            }
                        }
                    } else {
                        0
                    };
                    drop(cached_token_history_guard);

                    let prefill_tokens = if cached_prefix_len > 0 {
                        // Prefix matches — incremental prefill (only new tokens)
                        info!(
                            "Cache reuse: {} cached tokens, {} new tokens to prefill (vlm={})",
                            cached_prefix_len,
                            tokens_for_matching.len() - cached_prefix_len,
                            has_images
                        );
                        tokens_for_matching[cached_prefix_len..].to_vec()
                    } else {
                        // No match — full reset + full prefill
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
                        tokens.clone()
                    };

                    // Zero-delta guard: also reset cached_prefix_len for VLM routing.
                    let (prefill_tokens, cached_prefix_len) = if prefill_tokens.is_empty() {
                        info!("Zero-delta cache hit: resetting caches for full re-prefill");
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
                        let tokens = if has_images {
                            expanded_tokens.as_ref().unwrap_or(&tokens).clone()
                        } else {
                            tokens.clone()
                        };
                        (tokens, 0)
                    } else {
                        (prefill_tokens, cached_prefix_len)
                    };

                    let eos_id = model_config.eos_token_id as u32;
                    let mut generated_tokens: Vec<u32> = Vec::new();
                    let mut finish_reason = String::from("length");
                    let mut decode_stream = tokenizer_for_decode.inner().decode_stream(true);
                    let mut streamed_text_len: usize = 0;

                    // Track token history for repetition penalty
                    let mut token_history: Vec<u32> = if let Some(ref et) = expanded_tokens {
                        et.clone()
                    } else {
                        tokens.clone()
                    };

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
                    profiler.set_prompt_tokens(prefill_tokens.len() as u32);
                    profiler.snapshot_memory_before();

                    // === VLM or text prefill branching ===
                    profiler.begin_prefill();
                    let (mut last_logits, seq_len) = if has_images && cached_prefix_len > 0 {
                        // --- VLM cache reuse: same images, incremental text-only prefill ---
                        let expanded = expanded_tokens.as_ref().unwrap();
                        let prompt = MxArray::from_uint32(
                            &prefill_tokens,
                            &[1, prefill_tokens.len() as i64],
                        )?;

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
                        // seq_len is the TOTAL expanded tokens (cached + new)
                        (last_logits, expanded.len() as i64)
                    } else if let (true, Some(vision_enc), Some(_)) = (
                        has_images,
                        vision_encoder_arc.as_ref(),
                        image_processor_arc.as_ref(),
                    ) {
                        // --- VLM path: full VLM prefill (no cache reuse) ---
                        let sms = spatial_merge_size.unwrap_or(2);
                        let final_tokens = expanded_tokens.as_ref().unwrap();
                        let processed = vlm_processed.as_ref().unwrap();

                        let input_ids =
                            MxArray::from_uint32(final_tokens, &[1, final_tokens.len() as i64])?;

                        // VLM prefill using Rust path with M-RoPE position IDs
                        let (logits, rope_deltas) = vlm_prefill_moe(
                            &input_ids,
                            image_cache_key,
                            processed,
                            vision_enc,
                            sms,
                            &embedding_weight,
                            &mut layers_guard,
                            &mut caches_guard,
                            &final_norm_guard,
                            &lm_head_guard,
                            generation_stream,
                            fa_idx,
                            Some(&embedding_weight_t),
                            &vision_cache_stream,
                        )?;

                        // Save rope_deltas for cache reuse on subsequent turns
                        if let Ok(mut rd) = cached_rope_deltas_arc.write() {
                            *rd = Some(rope_deltas as i32);
                        }

                        let vlm_seq_len = final_tokens.len() as i64;
                        (logits, vlm_seq_len)
                    } else {
                        // --- Standard text prefill path ---
                        let prompt = MxArray::from_uint32(
                            &prefill_tokens,
                            &[1, prefill_tokens.len() as i64],
                        )?;

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
                        // seq_len for the C++ init is the TOTAL tokens (cached + new)
                        (last_logits, tokens.len() as i64)
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
                    if presence_penalty != 0.0 {
                        last_logits = apply_presence_penalty(
                            &last_logits,
                            &token_history,
                            presence_penalty,
                            Some(presence_context_size),
                        )?;
                    }
                    if frequency_penalty != 0.0 {
                        last_logits = apply_frequency_penalty(
                            &last_logits,
                            &token_history,
                            frequency_penalty,
                            Some(frequency_context_size),
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

                        // Apply M-RoPE offset correction AFTER init_from_prefill.
                        if has_images
                            && let Ok(rd) = cached_rope_deltas_arc.read()
                            && let Some(delta) = *rd
                        {
                            unsafe {
                                mlx_sys::mlx_qwen35_moe_adjust_offset(delta);
                            }
                        }

                        // For text-only conversations, clear any stale cached rope deltas
                        if !has_images && let Ok(mut rd) = cached_rope_deltas_arc.write() {
                            *rd = None;
                        }

                        // C++ decode loop (pipelined — submit N+1 before eval N)
                        profiler.set_label("moe_chat_stream_compiled");
                        for step in 0..max_new_tokens {
                            // Build and submit graph for step N+1
                            let next_y = if step + 1 < max_new_tokens {
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
                                if presence_penalty != 0.0 {
                                    logits = apply_presence_penalty(
                                        &logits,
                                        &token_history,
                                        presence_penalty,
                                        Some(presence_context_size),
                                    )?;
                                }
                                if frequency_penalty != 0.0 {
                                    logits = apply_frequency_penalty(
                                        &logits,
                                        &token_history,
                                        frequency_penalty,
                                        Some(frequency_context_size),
                                    )?;
                                }
                                let next_token = sample(&logits, sampling_config)?;
                                eval_token_and_moe_caches(&next_token);
                                Some(next_token)
                            } else {
                                None
                            };

                            // Wait for step N (GPU already computing N+1)
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

                            let token_text = crate::tokenizer::Qwen3Tokenizer::step_decode_stream(
                                &mut decode_stream,
                                tokenizer_for_decode.inner(),
                                token_id,
                                &generated_tokens,
                                streamed_text_len,
                            );
                            streamed_text_len += token_text.len();
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
                                finish_reason = String::from("stop");
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

                            match next_y {
                                Some(next) => y = next,
                                None => break,
                            }

                            profiler.step();

                            if (step + 1) % 256 == 0 {
                                crate::array::synchronize_and_clear_cache();
                            }
                        }
                        profiler.snapshot_memory_after();
                        profiler.report();

                        // === Export caches from C++ before MoeResetGuard drops ===
                        if reuse_cache {
                            let num_layers = model_config.num_layers as usize;
                            let mut export_ptrs: Vec<*mut mlx_sys::mlx_array> =
                                vec![std::ptr::null_mut(); num_layers * 2];
                            let exported = unsafe {
                                mlx_sys::mlx_qwen35_moe_export_caches(
                                    export_ptrs.as_mut_ptr(),
                                    (num_layers * 2) as i32,
                                )
                            };
                            if exported > 0 {
                                let cache_offset =
                                    unsafe { mlx_sys::mlx_qwen35_moe_get_cache_offset() };
                                let mut new_caches = Vec::with_capacity(num_layers);
                                for i in 0..num_layers {
                                    let p0 = export_ptrs[i * 2];
                                    let p1 = export_ptrs[i * 2 + 1];
                                    let mut lc = if model_config.is_linear_layer(i) {
                                        Qwen3_5LayerCache::new_linear()
                                    } else {
                                        Qwen3_5LayerCache::new_full_attention()
                                    };
                                    lc.import_ptrs(p0, p1, cache_offset);
                                    new_caches.push(lc);
                                }
                                let mut cg = caches_arc.write().map_err(|_| {
                                    Error::from_reason(
                                        "Failed to acquire caches lock for cache export",
                                    )
                                })?;
                                *cg = Some(new_caches);
                            }
                        }
                        // _moe_guard dropped here, calling mlx_qwen35_moe_reset()
                    } else {
                        // Rust fallback decode loop (pipelined)
                        profiler.set_label("moe_chat_stream_rust");
                        for step in 0..max_new_tokens {
                            // Build and submit graph for step N+1
                            let next_y = if step + 1 < max_new_tokens {
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
                                if presence_penalty != 0.0 {
                                    logits = apply_presence_penalty(
                                        &logits,
                                        &token_history,
                                        presence_penalty,
                                        Some(presence_context_size),
                                    )?;
                                }
                                if frequency_penalty != 0.0 {
                                    logits = apply_frequency_penalty(
                                        &logits,
                                        &token_history,
                                        frequency_penalty,
                                        Some(frequency_context_size),
                                    )?;
                                }
                                let next_token = sample(&logits, sampling_config)?;
                                MxArray::async_eval_arrays(&[&next_token]);
                                Some(next_token)
                            } else {
                                None
                            };

                            // Wait for step N (GPU already computing N+1)
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

                            let token_text = crate::tokenizer::Qwen3Tokenizer::step_decode_stream(
                                &mut decode_stream,
                                tokenizer_for_decode.inner(),
                                token_id,
                                &generated_tokens,
                                streamed_text_len,
                            );
                            streamed_text_len += token_text.len();
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
                                finish_reason = String::from("stop");
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

                            match next_y {
                                Some(next) => y = next,
                                None => break,
                            }

                            profiler.step();

                            if (step + 1) % 256 == 0 {
                                crate::array::synchronize_and_clear_cache();
                            }
                        }

                        profiler.snapshot_memory_after();
                        profiler.report();

                        drop(layers_guard);
                        drop(caches_guard);
                        drop(final_norm_guard);
                        drop(lm_head_guard);
                    }

                    // === Save token history and image key for cache reuse on next call ===
                    if reuse_cache {
                        // For VLM, save the expanded token history (with image placeholders)
                        let mut full_history = if let Some(ref et) = expanded_tokens {
                            et.clone()
                        } else {
                            tokens.clone()
                        };
                        // Only include tokens that were actually forwarded through the model.
                        // When stopped at max_tokens ("length"), the last token was never forwarded
                        // (the pipelined loop skips forward on the final step).
                        let history_tokens =
                            if finish_reason == "length" && !generated_tokens.is_empty() {
                                &generated_tokens[..generated_tokens.len() - 1]
                            } else {
                                &generated_tokens
                            };
                        full_history.extend_from_slice(history_tokens);
                        if let Ok(mut th) = cached_token_history_arc.write() {
                            *th = full_history;
                        }
                        // Save image cache key (Some for VLM, None for text-only)
                        if let Ok(mut ik) = cached_image_key_arc.write() {
                            *ik = if has_images {
                                Some(image_cache_key)
                            } else {
                                None
                            };
                        }
                    } else {
                        if let Ok(mut cg) = caches_arc.write() {
                            *cg = None;
                        }
                        if let Ok(mut th) = cached_token_history_arc.write() {
                            th.clear();
                        }
                        if let Ok(mut ik) = cached_image_key_arc.write() {
                            *ik = None;
                        }
                        if let Ok(mut rd) = cached_rope_deltas_arc.write() {
                            *rd = None;
                        }
                    }

                    // Compute performance metrics if requested
                    let performance = if let (Some(gen_start), Some(first_tok)) =
                        (generation_start, first_token_instant)
                    {
                        let generation_end = std::time::Instant::now();
                        let actual_prefill_toks = prefill_tokens.len() as f64;
                        let gen_tokens = generated_tokens.len() as f64;
                        let ttft_ms = first_tok.duration_since(gen_start).as_secs_f64() * 1000.0;
                        let decode_ms =
                            generation_end.duration_since(first_tok).as_secs_f64() * 1000.0;
                        Some(crate::profiling::PerformanceMetrics {
                            ttft_ms,
                            prefill_tokens_per_second: if ttft_ms > 0.0 {
                                actual_prefill_toks / (ttft_ms / 1000.0)
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

                    // Flush residual bytes buffered by DecodeStream
                    if text.len() > streamed_text_len {
                        let residual = text[streamed_text_len..].to_string();
                        callback.call(
                            Ok(ChatStreamChunk {
                                text: residual,
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
                    }

                    let num_tokens = generated_tokens.len() as u32;

                    let think_tag =
                        if tools::has_think_end_token(&generated_tokens, think_end_id_stream) {
                            think_end_str_stream.as_deref()
                        } else {
                            None
                        };
                    let (clean_text, tool_calls, thinking) =
                        tools::split_at_think_end(&text, think_tag);

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

    /// Save the model weights and configuration to a directory.
    ///
    /// This saves:
    /// - config.json: Model configuration (with model_type for detectModelType)
    /// - weights.safetensors: Full model weights in SafeTensors format
    /// - weights.mlx: Parameter metadata (for reference)
    ///
    /// # Arguments
    /// * `save_path` - Directory to save the model
    #[napi]
    pub fn save_model<'env>(
        &self,
        env: &'env Env,
        save_path: String,
    ) -> Result<PromiseRaw<'env, ()>> {
        let mut params = self.get_parameters_for_training()?;

        // Include vision encoder weights when present (VLM models)
        if let Some(ref vision_enc) = self.vision_encoder {
            let vision_params = vision_enc.get_parameters();
            params.extend(vision_params);
        }

        // Validate all parameters for NaN/Inf before saving
        for (name, param) in params.iter() {
            let data = param.to_float32()?;
            let invalid_count = data
                .iter()
                .filter(|v| v.is_nan() || v.is_infinite())
                .count();
            if invalid_count > 0 {
                return Err(napi::Error::new(
                    Status::GenericFailure,
                    format!(
                        "Cannot save model: parameter '{}' contains {} NaN/Inf values. \
                        Model weights are corrupted, likely due to training instability. \
                        Consider reducing learning rate or using an earlier checkpoint.",
                        name, invalid_count
                    ),
                ));
            }
        }

        let params_clone: HashMap<String, MxArray> =
            params.iter().map(|(k, v)| (k.clone(), v.clone())).collect();

        // Create weights metadata (for reference)
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
        let config = self.get_config();
        let mut config_value = serde_json::to_value(&config).map_err(|e| {
            napi::Error::new(
                Status::GenericFailure,
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

        let promise = env.spawn_future(async move {
            tokio::task::spawn_blocking(move || {
                let path = std::path::Path::new(&save_path);
                std::fs::create_dir_all(path)?;

                info!("Saving model to {}", save_path);

                // 1. Save configuration as JSON
                let config_path = path.join("config.json");
                let config_json = serde_json::to_string_pretty(&config_value)?;
                std::fs::write(&config_path, config_json)?;
                info!("Saved config.json");

                // 2. Save full weights in SafeTensors format
                let safetensors_path = path.join("weights.safetensors");
                let metadata = Some(serde_json::json!({
                    "format": "mlx-node",
                    "version": "1.0"
                }));
                crate::utils::safetensors::save_safetensors(
                    &safetensors_path,
                    &params_clone,
                    metadata,
                )?;
                info!("Saved weights.safetensors");

                // 3. Save weights metadata (for reference)
                let weights_str = serde_json::to_string_pretty(&weights_json)?;
                let weights_path = path.join("weights.mlx");
                std::fs::write(&weights_path, weights_str)?;
                info!("Saved weights.mlx metadata");

                Ok::<_, Error>(())
            })
            .map_err(|err| {
                napi::Error::new(
                    Status::GenericFailure,
                    format!("Failed to save model: {}", err),
                )
            })
            .await
            .flatten()?;
            Ok(())
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
            h = layers_guard[i].forward(&h, mask, cache, layer_pos, true)?;
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

// ========== Training Support Methods ==========
// These methods are Rust-internal only (not exposed via NAPI).
// They implement the TrainableModel trait interface for MoE.

impl Qwen3_5MoeModel {
    /// Get model configuration.
    pub(crate) fn get_config(&self) -> Qwen3_5MoeConfig {
        self.config.clone()
    }

    /// Create a cheap clone for training sessions.
    /// Arc-clones all shared components, no deep copy. No VLM components.
    pub(crate) fn clone_for_training(&self) -> Result<Self> {
        Ok(Self {
            config: self.config.clone(),
            embedding: Embedding::from_weight(&self.embedding.get_weight())?,
            layers: Arc::clone(&self.layers),
            final_norm: Arc::clone(&self.final_norm),
            lm_head: Arc::clone(&self.lm_head),
            caches: Arc::new(RwLock::new(None)), // Fresh empty caches
            tokenizer: self.tokenizer.clone(),
            fa_idx: self.fa_idx,
            vision_encoder: None, // Not needed for training
            image_processor: None,
            spatial_merge_size: None,
            vision_cache: Arc::new(Mutex::new(VisionCacheInner {
                entries: HashMap::new(),
                generation: 0,
            })),
            cached_token_history: Arc::new(RwLock::new(Vec::new())),
            cached_image_key: Arc::new(RwLock::new(None)),
            cached_rope_deltas: Arc::new(RwLock::new(None)),
            model_id: QWEN35_MODEL_ID_COUNTER.fetch_add(1, Ordering::Relaxed),
            generation_lock: Arc::new(TokioMutex::new(())),
        })
    }

    /// Extract all trainable parameters as a name->array map.
    ///
    /// Parameter naming convention matches HuggingFace format:
    /// - `embedding.weight`
    /// - Linear attention layers: `layers.{i}.linear_attn.{in_proj_qkvz,in_proj_ba,conv1d,norm,out_proj}.weight`
    /// - Linear attention learnable: `layers.{i}.linear_attn.{a_log,dt_bias}`
    /// - Full attention layers: `layers.{i}.self_attn.{q_proj,k_proj,v_proj,o_proj}.weight`
    /// - Full attention norms: `layers.{i}.self_attn.{q_norm,k_norm}.weight`
    /// - Dense MLP layers: `layers.{i}.mlp.{gate_proj,up_proj,down_proj}.weight`
    /// - MoE layers:
    ///   - `layers.{i}.mlp.gate.weight` (router)
    ///   - `layers.{i}.mlp.switch_mlp.{gate_proj,up_proj,down_proj}.weight` (expert weights 3D)
    ///   - `layers.{i}.mlp.shared_expert.{gate_proj,up_proj,down_proj}.weight`
    ///   - `layers.{i}.mlp.shared_expert_gate.weight`
    /// - All layers: `layers.{i}.{input_layernorm,post_attention_layernorm}.weight`
    /// - `final_norm.weight`
    /// - `lm_head.weight` (if not tied)
    pub(crate) fn get_parameters_for_training(&self) -> Result<HashMap<String, MxArray>> {
        use super::decoder_layer::{AttentionType, MLPType};

        let mut params = HashMap::new();

        let layers_guard = self
            .layers
            .read()
            .map_err(|_| Error::from_reason("Failed to acquire layers read lock"))?;
        let final_norm_guard = self
            .final_norm
            .read()
            .map_err(|_| Error::from_reason("Failed to acquire final_norm read lock"))?;
        let lm_head_guard = self
            .lm_head
            .read()
            .map_err(|_| Error::from_reason("Failed to acquire lm_head read lock"))?;

        // Embedding
        params.insert("embedding.weight".to_string(), self.embedding.get_weight());

        // Transformer layers
        for (i, layer) in layers_guard.iter().enumerate() {
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
            final_norm_guard.get_weight(),
        );

        // LM head (only if not tied)
        if !self.config.tie_word_embeddings
            && let Some(ref lm_head) = *lm_head_guard
        {
            params.insert("lm_head.weight".to_string(), lm_head.get_weight());
        }

        Ok(params)
    }

    /// Apply gradients to model parameters using pre-fetched params.
    /// SGD update: param = param - lr * grad.
    pub(crate) fn apply_gradients_with_params(
        &mut self,
        gradients: HashMap<String, &MxArray>,
        learning_rate: f64,
        current_params: &HashMap<String, MxArray>,
    ) -> Result<()> {
        use crate::training_model::compute_sgd_updates;

        // Compute updated parameters using shared SGD helper
        let updated_params = compute_sgd_updates(&gradients, learning_rate, current_params)?;

        // Acquire write locks
        let mut layers = self.layers.write().map_err(|_| {
            Error::new(
                Status::GenericFailure,
                "Failed to acquire layers write lock",
            )
        })?;
        let mut final_norm = self.final_norm.write().map_err(|_| {
            Error::new(
                Status::GenericFailure,
                "Failed to acquire final_norm write lock",
            )
        })?;
        let mut lm_head = self.lm_head.write().map_err(|_| {
            Error::new(
                Status::GenericFailure,
                "Failed to acquire lm_head write lock",
            )
        })?;

        // Apply updates
        for (name, updated_param) in updated_params.iter() {
            if name == "lm_head.weight" {
                if let Some(ref mut lm) = *lm_head {
                    lm.set_weight(updated_param, "lm_head")?;
                }
            } else if name == "final_norm.weight" {
                final_norm.set_weight(updated_param)?;
            } else if name == "embedding.weight" {
                self.embedding.set_weight(updated_param)?;
            } else if name.starts_with("layers.") {
                let parts: Vec<&str> = name.split('.').collect();
                if parts.len() >= 3
                    && let Ok(layer_idx) = parts[1].parse::<usize>()
                    && layer_idx < layers.len()
                {
                    let layer = &mut layers[layer_idx];
                    self.apply_layer_gradient(layer, name, updated_param)?;
                }
            }
        }

        Ok(())
    }

    fn apply_layer_gradient(
        &self,
        layer: &mut super::decoder_layer::DecoderLayer,
        name: &str,
        updated_param: &MxArray,
    ) -> Result<()> {
        use super::decoder_layer::{AttentionType, MLPType};

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
        } else {
            tracing::warn!(
                "Unrecognized parameter name in apply_layer_gradient: {}",
                name
            );
        }

        Ok(())
    }

    /// Tokenize messages using the model's chat template.
    pub(crate) fn apply_chat_template_sync(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: Option<bool>,
        tools: Option<&[ToolDefinition]>,
        enable_thinking: Option<bool>,
    ) -> Result<Vec<u32>> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded - call load() first"))?;
        tokenizer.apply_chat_template_sync(messages, add_generation_prompt, tools, enable_thinking)
    }

    /// Generate a single completion with logprob tracking (for GRPO training).
    ///
    /// Uses the C++ MoE forward path when available (~10x faster than Rust).
    /// Generation does NOT need differentiability — gradients are computed separately
    /// via the functional forward path in autograd Phase 2.
    pub(crate) fn generate_for_training_sync(
        &self,
        input_ids: &MxArray,
        config: Option<GenerationConfig>,
    ) -> Result<GenerationResult> {
        let config = config.unwrap_or_default();
        let eos_token_id = config.eos_token_id.or(Some(self.config.eos_token_id));
        let model_id = self.model_id;

        // Hold generation lock (blocking — called from spawn_blocking context).
        let _gen_guard = self.generation_lock.blocking_lock();

        // Check if C++ MoE path is available (weights belong to this model)
        let use_cpp = unsafe { mlx_sys::mlx_qwen35_get_model_id() } == model_id;

        // Try to acquire MoE compiled mutex (non-blocking, safe from sync context).
        // If locked (concurrent generate() call), fall back to Rust path.
        let compiled_lock = if use_cpp {
            MOE_COMPILED_MUTEX.try_lock().ok()
        } else {
            None
        };
        let use_cpp = compiled_lock.is_some();

        let _weight_guard = if use_cpp {
            acquire_compiled_weight_guard(model_id)
        } else {
            None
        };
        let use_cpp = _weight_guard.is_some();

        self.init_caches_inner()?;

        // Acquire locks
        let embedding_weight = self.embedding.get_weight();
        let mut layers_guard = self
            .layers
            .write()
            .map_err(|_| Error::from_reason("Failed to acquire layers write lock"))?;
        let final_norm_guard = self
            .final_norm
            .read()
            .map_err(|_| Error::from_reason("Failed to acquire final_norm read lock"))?;
        let lm_head_guard = self
            .lm_head
            .read()
            .map_err(|_| Error::from_reason("Failed to acquire lm_head read lock"))?;
        let mut caches_guard = self
            .caches
            .write()
            .map_err(|_| Error::from_reason("Failed to acquire caches write lock"))?;

        let fa_idx = self.fa_idx;

        // === Prefill (always uses Rust forward — runs once) ===
        let logits = forward_inner(
            input_ids,
            &embedding_weight,
            &mut layers_guard,
            &mut caches_guard,
            &final_norm_guard,
            &lm_head_guard,
            fa_idx,
            None,
        )?;
        let seq_len = logits.shape_at(1)?;
        let last_logits = logits.slice_axis(1, seq_len - 1, seq_len)?;
        let last_logits = last_logits.squeeze(Some(&[1]))?;
        let input_tokens = input_ids.to_uint32()?;

        let result = if use_cpp {
            // === C++ MoE decode path ===
            let _moe_guard = MoeResetGuard;
            let max_new_tokens = config.max_new_tokens.unwrap_or(100);
            let prefill_len = seq_len as i32;
            let max_kv_len = ((prefill_len + max_new_tokens + 255) / 256) * 256;
            let num_layers = self.config.num_layers as usize;
            let mut cache_ptrs: Vec<*mut mlx_sys::mlx_array> =
                vec![std::ptr::null_mut(); num_layers * 2];
            if let Some(ref caches) = *caches_guard {
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
            // Drop locks not needed during C++ MoE decode
            drop(layers_guard);
            drop(final_norm_guard);
            drop(lm_head_guard);
            // Keep caches_guard alive through init_from_prefill so cache_ptrs remain valid
            unsafe {
                mlx_sys::mlx_qwen35_moe_init_from_prefill(
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
            // C++ has copied arrays into its own globals — safe to release
            drop(caches_guard);

            // Decode using C++ MoE forward with synchronous cache eval.
            // Caches are eval'd BEFORE each forward call to ensure the previous step's
            // caches are materialized, preventing O(N²) graph growth.
            let mut forward_fn = |ids: &MxArray| -> Result<MxArray> {
                unsafe { mlx_sys::mlx_qwen35_moe_sync_eval_caches() };
                let logits = forward_moe_cpp(ids, &embedding_weight)?;
                // forward_moe_cpp returns [1, vocab] but training loop expects [1, 1, vocab]
                let logits = logits.reshape(&[1, 1, -1])?;
                Ok(logits)
            };

            crate::models::training_generate::generate_decode_loop_for_training(
                &last_logits,
                &input_tokens,
                &config,
                eos_token_id,
                &mut forward_fn,
            )?
            // _moe_guard dropped here → mlx_qwen35_moe_reset()
        } else {
            // === Rust fallback decode path ===
            let mut forward_fn = |ids: &MxArray| -> Result<MxArray> {
                forward_inner(
                    ids,
                    &embedding_weight,
                    &mut layers_guard,
                    &mut caches_guard,
                    &final_norm_guard,
                    &lm_head_guard,
                    fa_idx,
                    None,
                )
            };

            let result = crate::models::training_generate::generate_decode_loop_for_training(
                &last_logits,
                &input_tokens,
                &config,
                eos_token_id,
                &mut forward_fn,
            )?;

            drop(layers_guard);
            drop(final_norm_guard);
            drop(lm_head_guard);
            drop(caches_guard);

            result
        };

        drop(compiled_lock);
        self.reset_caches_inner()?;

        Ok(result)
    }

    /// Generate a batch of completions for GRPO training.
    pub(crate) fn generate_batch_for_training_sync(
        &self,
        prompt_arrays: &[MxArray],
        group_size: usize,
        config: Option<GenerationConfig>,
    ) -> Result<BatchGenerationResult> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))?;

        crate::models::training_generate::generate_batch_for_training_loop(
            prompt_arrays,
            group_size,
            config,
            tokenizer,
            |prompt, cfg| self.generate_for_training_sync(prompt, cfg),
        )
    }

    /// Decode token IDs to text.
    pub(crate) fn decode_tokens_sync(&self, tokens: &MxArray) -> Result<String> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))?;
        let token_ids = tokens.to_uint32()?;
        tokenizer.decode_sync(&token_ids, true)
    }
}
