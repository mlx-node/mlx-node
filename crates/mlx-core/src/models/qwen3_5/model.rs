use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi_derive::napi;
use tokio::sync::Mutex as TokioMutex;
use tracing::{info, warn};

use crate::array::MxArray;
use crate::array::mask::create_causal_mask;
use crate::nn::{Embedding, Linear, RMSNorm};
use crate::sampling::{SamplingConfig, apply_repetition_penalty, check_repetition_cutoff, sample};
use crate::stream::{DeviceType, Stream, StreamContext};
use crate::tokenizer::{ChatMessage, Qwen3Tokenizer, ToolDefinition};
use crate::tools;
use crate::tools::ToolCallResult;

use super::config::Qwen3_5Config;
use super::decoder_layer::DecoderLayer;
use super::layer_cache::Qwen3_5LayerCache;
use super::persistence;
use super::processing::Qwen35VLImageProcessor;
use super::vision::Qwen3_5VisionEncoder;

/// Maximum number of entries in the vision encoder cache before LRU eviction.
pub(crate) const VISION_CACHE_MAX_ENTRIES: usize = 32;

/// LRU cache for vision encoder embeddings, keyed by image content hash.
pub(crate) struct VisionCacheInner {
    pub entries: HashMap<u64, (MxArray, MxArray, u64)>,
    /// Monotonically increasing counter for LRU generation tracking.
    pub generation: u64,
}

pub(crate) type VisionCache = Arc<Mutex<VisionCacheInner>>;

/// Hash raw image bytes to a u64 key for cache lookup.
pub(crate) fn hash_image_bytes(bytes: &[u8]) -> u64 {
    let mut hasher = DefaultHasher::new();
    bytes.hash(&mut hasher);
    hasher.finish()
}

/// Combine individual image hashes into a single cache key.
/// Order matters: different orderings of the same images produce different keys.
pub(crate) fn combine_image_hashes(hashes: &[u64]) -> u64 {
    let mut hasher = DefaultHasher::new();
    for h in hashes {
        h.hash(&mut hasher);
    }
    hasher.finish()
}

/// Process-wide mutex serializing the dense compiled forward lifecycle.
///
/// The C++ compiled decode path uses process-wide globals (`g_compiled_caches`,
/// `g_offset_int`, etc.). Concurrent `generate()`/`chat()` calls via
/// `Promise.all()` would race on these globals since `spawn_blocking` dispatches
/// to separate threads. This mutex is acquired in the async context *before*
/// `spawn_blocking`, ensuring only one compiled lifecycle runs at a time.
static DENSE_COMPILED_MUTEX: TokioMutex<()> = TokioMutex::const_new(());

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
#[derive(Debug, Clone)]
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
    /// Max consecutive identical tokens before stopping (default: 16, 0 = disabled)
    #[napi(ts_type = "number | undefined")]
    pub max_consecutive_tokens: Option<i32>,
    /// Max n-gram repetitions before stopping (default: 8, 0 = disabled)
    #[napi(ts_type = "number | undefined")]
    pub max_ngram_repeats: Option<i32>,
    /// N-gram size for repetition detection (default: 3)
    #[napi(ts_type = "number | undefined")]
    pub ngram_size: Option<i32>,
    #[napi(ts_type = "Array<ToolDefinition>")]
    pub tools: Option<Vec<ToolDefinition>>,
    /// When true, include performance metrics (TTFT, prefill tok/s, decode tok/s) in the result
    #[napi(ts_type = "boolean | undefined")]
    pub report_performance: Option<bool>,
}

/// Unified chat result shared by all model variants (Qwen3, Qwen3.5, Qwen3.5 MoE).
#[napi(object)]
#[derive(Debug, Clone)]
pub struct ChatResult {
    pub text: String,
    pub tool_calls: Vec<ToolCallResult>,
    pub thinking: Option<String>,
    pub num_tokens: u32,
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
    pub raw_text: Option<String>,
    /// Performance metrics (only present in the final chunk when `reportPerformance: true`)
    pub performance: Option<crate::profiling::PerformanceMetrics>,
}

/// Handle returned by `chat_stream()` to control an in-progress streaming generation.
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
/// Uses interior mutability (RwLock) for layers, final_norm, lm_head, and caches
/// to allow async generation via spawn_blocking without blocking the Node.js event loop.
/// This matches the pattern used by Qwen3Model.
#[napi]
pub struct Qwen3_5Model {
    pub(crate) config: Qwen3_5Config,
    pub(crate) embedding: Embedding,
    /// Decoder layers wrapped in RwLock for interior mutability during generation.
    pub(crate) layers: Arc<RwLock<Vec<DecoderLayer>>>,
    /// Final layer norm wrapped in RwLock for interior mutability.
    pub(crate) final_norm: Arc<RwLock<RMSNorm>>,
    /// LM head wrapped in RwLock for interior mutability.
    pub(crate) lm_head: Arc<RwLock<Option<Linear>>>, // None when tie_word_embeddings
    /// KV/SSM caches wrapped in RwLock for interior mutability during generation.
    pub(crate) caches: Arc<RwLock<Option<Vec<Qwen3_5LayerCache>>>>,
    pub(crate) tokenizer: Option<Arc<Qwen3Tokenizer>>,
    pub(crate) fa_idx: usize, // Index of first full attention layer
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
impl Qwen3_5Model {
    /// Create a new Qwen3.5 model with the given configuration.
    #[napi(constructor)]
    pub fn new(config: Qwen3_5Config) -> Result<Self> {
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

        // Find first full attention layer index
        let fa_idx = (0..config.num_layers as usize)
            .find(|&i| !config.is_linear_layer(i))
            .unwrap_or(0);

        info!(
            "Qwen3.5 model created: {} layers, fa_idx={}",
            config.num_layers, fa_idx,
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

    /// Initialize caches for incremental generation.
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

    /// Reset all caches.
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

    /// Forward pass through the model.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs [B, T]
    ///
    /// # Returns
    /// Logits [B, T, vocab_size]
    #[napi]
    pub fn forward(&self, input_ids: &MxArray) -> Result<MxArray> {
        let hidden_states = self.embedding.forward(input_ids)?;
        self.forward_from_embeddings(&hidden_states)
    }

    /// Forward pass with cache for incremental generation.
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

    /// Load a pretrained model from a directory.
    ///
    /// Expects the directory to contain:
    /// - config.json
    /// - model.safetensors (or model-*.safetensors)
    /// - tokenizer.json + tokenizer_config.json
    #[napi]
    pub async fn load_pretrained(path: String) -> Result<Qwen3_5Model> {
        persistence::load_pretrained(&path).await
    }

    /// Generate text from a prompt token sequence.
    ///
    /// Runs generation on a worker thread via spawn_blocking to avoid
    /// blocking the Node.js event loop.
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

        // generate() assumes batch_size=1 (reshape to [1,1], item_at_int32(0)).
        // Reject multi-batch prompts early to avoid silently wrong results.
        let batch_size = prompt_tokens.shape_at(0)?;
        if batch_size != 1 {
            return Err(Error::from_reason(format!(
                "generate() only supports batch_size=1, got batch_size={}",
                batch_size
            )));
        }

        // Clone Arcs and data needed for the closure
        let embedding_weight = self.embedding.get_weight();
        let layers_arc = self.layers.clone();
        let final_norm_arc = self.final_norm.clone();
        let lm_head_arc = self.lm_head.clone();
        let caches_arc = self.caches.clone();
        let model_config = self.config.clone();
        let tokenizer = self.tokenizer.clone();
        let fa_idx = self.fa_idx;
        let prompt_tokens = prompt_tokens.clone();

        // Check if compiled path will be used (C++ weights loaded from safetensors).
        // Must be checked before spawn_blocking so we can acquire the mutex in async context.
        let use_compiled = unsafe { mlx_sys::mlx_qwen35_weight_count() } > 0;

        // Serialize compiled lifecycle — prevents concurrent C++ global corruption
        let _compiled_lock = if use_compiled {
            Some(DENSE_COMPILED_MUTEX.lock().await)
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

            // Init fresh caches (old ones dropped on overwrite)
            *caches_guard = Some(
                (0..model_config.num_layers as usize)
                    .map(|i| {
                        if model_config.is_linear_layer(i) {
                            Qwen3_5LayerCache::new_linear()
                        } else {
                            Qwen3_5LayerCache::new_full_attention()
                        }
                    })
                    .collect(),
            );

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

            // Pre-compute embedding weight transpose once (avoids recomputing per step)
            let embedding_weight_t = embedding_weight.transpose(Some(&[1, 0]))?;

            // Create dedicated generation stream for GPU scheduling
            let generation_stream = Stream::new(DeviceType::Gpu);

            // Pin model weights in Metal memory for the duration of generation
            let model_size_bytes = model_config.estimate_memory_bytes() as usize;
            let _wired_ctx =
                crate::stream::WiredLimitContext::new(model_size_bytes, vec![generation_stream]);

            // Profiler — covers both compiled and rust decode paths
            let mut profiler = crate::decode_profiler::DecodeProfiler::new("generate", "qwen3_5");
            profiler.set_prompt_tokens(prompt_tokens.shape_at(1).unwrap_or(0) as u32);
            profiler.snapshot_memory_before();

            // Prefill: forward pass on entire prompt
            profiler.begin_prefill();
            let logits = {
                let _stream_ctx = StreamContext::new(generation_stream);
                forward_inner(
                    &prompt_tokens,
                    &embedding_weight,
                    &mut layers_guard,
                    &mut caches_guard,
                    &final_norm_guard,
                    &lm_head_guard,
                    fa_idx,
                    Some(&embedding_weight_t),
                )?
            };
            profiler.end_prefill();

            // Get last token logits: [1, vocab]
            let seq_len = logits.shape_at(1)?;
            let last_logits = logits.slice_axis(1, seq_len - 1, seq_len)?;
            let last_logits = last_logits.squeeze(Some(&[1]))?; // [1, vocab]

            // Sample first token (lazy — not evaluated yet)
            let mut y = sample(&last_logits, sampling_config)?;
            MxArray::async_eval_arrays(&[&y]);

            // Guard ensures mlx_qwen35_compiled_reset() is called even if `?` returns early.
            let _compiled_guard = if use_compiled {
                Some(CompiledResetGuard)
            } else {
                None
            };

            if use_compiled {
                // Initialize compiled forward pass from prefill caches.
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
                // Drop non-cache locks — not needed during compiled decode
                drop(layers_guard);
                drop(final_norm_guard);
                drop(lm_head_guard);
                // Keep caches_guard alive through init_from_prefill so cache_ptrs
                // (raw pointers into the cache MxArrays) remain valid.
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
                        1, // batch_size
                        cache_ptrs.as_mut_ptr(),
                        prefill_len,
                    );
                }
                // C++ has copied arrays into g_compiled_caches — safe to release
                drop(caches_guard);

                // Compiled C++ decode loop (all locks dropped — C++ owns the state)
                profiler.set_label("generate_compiled");

                for step in 0..max_tokens {
                    let next_y = {
                        let _stream_ctx = StreamContext::new(generation_stream);
                        if step + 1 < max_tokens {
                            profiler.begin("forward");
                            let next_ids = y.reshape(&[1, 1])?;
                            let logits = forward_compiled(&next_ids, &embedding_weight)?;
                            profiler.end();

                            profiler.begin("sample");
                            let next_token = sample(&logits, sampling_config)?;
                            profiler.end();

                            profiler.begin("eval_caches");
                            eval_token_and_compiled_caches(&next_token);
                            profiler.end();

                            Some(next_token)
                        } else {
                            None
                        }
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
                        crate::array::synchronize_and_clear_cache();
                    }
                }

                profiler.snapshot_memory_after();
                profiler.report();
            } else {
                // Rust fallback decode loop (locks held for entire loop)
                profiler.set_label("generate_rust");

                for step in 0..max_tokens {
                    let next_y = {
                        let _stream_ctx = StreamContext::new(generation_stream);
                        if step + 1 < max_tokens {
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
                        }
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
                        crate::array::synchronize_and_clear_cache();
                    }
                }

                profiler.snapshot_memory_after();
                profiler.report();
            }

            // _compiled_guard dropped here (if Some), calling mlx_qwen35_compiled_reset()

            // Decode text if tokenizer available
            let text = if let Some(ref tok) = tokenizer {
                tok.decode_sync(&generated_tokens, true)
                    .unwrap_or_else(|e| {
                        warn!("Failed to decode generated tokens: {}", e);
                        String::new()
                    })
            } else {
                warn!("No tokenizer loaded - text decoding unavailable, only token IDs returned");
                String::new()
            };

            let num_tokens = generated_tokens.len() as u32;

            Ok(Qwen3_5GenerationResult {
                tokens: generated_tokens,
                text,
                num_tokens,
                finish_reason,
            })
        })
        .await
        .map_err(|e| Error::from_reason(format!("Generation task failed: {}", e)))?
    }

    /// Chat API with tool calling support.
    ///
    /// Runs tokenization + generation on a worker thread via spawn_blocking
    /// to avoid blocking the Node.js event loop.
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

        // Tokenize messages using chat template
        let tokenizer = self
            .tokenizer
            .clone()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))?;

        // Detect images in messages
        let has_images = messages
            .iter()
            .any(|m| m.images.as_ref().is_some_and(|imgs| !imgs.is_empty()));

        // Clone Arcs and data needed for the closure
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

        // Check if compiled path will be used (C++ weights loaded from safetensors).
        let use_compiled = unsafe { mlx_sys::mlx_qwen35_weight_count() } > 0;

        // Capture start time BEFORE compiled mutex + spawn_blocking so TTFT
        // reflects the full user-perceived latency (mutex wait + thread dispatch
        // + tokenization + prefill + first GPU eval).
        let report_perf = config.report_performance.unwrap_or(false);
        let generation_start = if report_perf {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // Serialize compiled lifecycle — prevents concurrent C++ global corruption
        let _compiled_lock = if use_compiled {
            Some(DENSE_COMPILED_MUTEX.lock().await)
        } else {
            None
        };

        napi::bindgen_prelude::spawn_blocking(move || {
            let mut first_token_instant: Option<std::time::Instant> = None;

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

            let eos_id = model_config.eos_token_id as u32;
            let mut generated_tokens: Vec<u32> = Vec::new();
            let mut finish_reason = String::from("length");

            let embedding_weight_t = embedding_weight.transpose(Some(&[1, 0]))?;
            let generation_stream = Stream::new(DeviceType::Gpu);
            let model_size_bytes = model_config.estimate_memory_bytes() as usize;
            let _wired_ctx =
                crate::stream::WiredLimitContext::new(model_size_bytes, vec![generation_stream]);

            // Profiler — covers both compiled and rust chat decode paths
            let mut profiler = crate::decode_profiler::DecodeProfiler::new("chat", "qwen3_5");
            profiler.set_prompt_tokens(tokens.len() as u32);
            profiler.snapshot_memory_before();

            // === VLM or text prefill branching ===
            // vlm_compiled_init_done: true if vlm_prefill already called compiled_init_from_prefill
            profiler.begin_prefill();
            let (mut last_logits, seq_len, vlm_compiled_init_done) =
                if let (true, Some(vision_enc), Some(img_proc)) = (
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

                    let (logits, _rope_deltas) = vlm_prefill(
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
                        max_new_tokens,
                        generation_stream,
                        &vision_cache,
                    )?;

                    // VLM prefill already did compiled_init_from_prefill for compiled path
                    let vlm_seq_len = final_tokens.len() as i64;
                    (logits, vlm_seq_len, use_compiled)
                } else {
                    // --- Standard text prefill path ---
                    let prompt = MxArray::from_uint32(&tokens, &[1, tokens.len() as i64])?;

                    // Init fresh caches (old ones dropped on overwrite)
                    *caches_guard = Some(
                        (0..model_config.num_layers as usize)
                            .map(|i| {
                                if model_config.is_linear_layer(i) {
                                    Qwen3_5LayerCache::new_linear()
                                } else {
                                    Qwen3_5LayerCache::new_full_attention()
                                }
                            })
                            .collect(),
                    );

                    let logits = {
                        let _stream_ctx = StreamContext::new(generation_stream);
                        forward_inner(
                            &prompt,
                            &embedding_weight,
                            &mut layers_guard,
                            &mut caches_guard,
                            &final_norm_guard,
                            &lm_head_guard,
                            fa_idx,
                            Some(&embedding_weight_t),
                        )?
                    };

                    let seq_len = logits.shape_at(1)?;
                    let last_logits = logits.slice_axis(1, seq_len - 1, seq_len)?;
                    let last_logits = last_logits.squeeze(Some(&[1]))?;
                    (last_logits, seq_len, false)
                };
            profiler.end_prefill();

            // Track token history for repetition penalty
            let mut token_history: Vec<u32> = tokens.clone();

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

            let _compiled_guard = if use_compiled {
                Some(CompiledResetGuard)
            } else {
                None
            };

            if use_compiled {
                if vlm_compiled_init_done {
                    // VLM prefill already called compiled_init_from_prefill and
                    // transferred caches — just drop the locks we don't need.
                    drop(layers_guard);
                    drop(final_norm_guard);
                    drop(lm_head_guard);
                    drop(caches_guard);
                } else {
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
                    // Drop non-cache locks — not needed during compiled decode
                    drop(layers_guard);
                    drop(final_norm_guard);
                    drop(lm_head_guard);
                    // Keep caches_guard alive through init_from_prefill so cache_ptrs
                    // (raw pointers into the cache MxArrays) remain valid.
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
                            prefill_len,
                        );
                    }
                    // C++ has copied arrays into g_compiled_caches — safe to release
                    drop(caches_guard);
                }

                // Compiled C++ decode loop (all locks dropped — C++ owns the state)
                profiler.set_label("chat_compiled");

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

                    if step + 1 >= max_new_tokens {
                        break;
                    }
                    {
                        let _stream_ctx = StreamContext::new(generation_stream);

                        profiler.begin("forward");
                        let next_ids = y.reshape(&[1, 1])?;
                        let mut logits = forward_compiled(&next_ids, &embedding_weight)?;
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
                        eval_token_and_compiled_caches(&next_token);
                        profiler.end();

                        y = next_token;
                    }

                    profiler.step();

                    if (step + 1) % 256 == 0 {
                        crate::array::synchronize_and_clear_cache();
                    }
                }

                profiler.snapshot_memory_after();
                profiler.report();
            } else {
                // Rust fallback decode loop — pipelined like mlx-lm:
                // Build next step's graph before blocking on current token.
                profiler.set_label("chat_rust");

                // Kick off first token's async eval
                MxArray::async_eval_arrays(&[&y]);

                for step in 0..max_new_tokens {
                    // Build NEXT step's graph BEFORE blocking on current token.
                    let mut next_y_opt: Option<MxArray> = None;
                    if step + 1 < max_new_tokens {
                        let _stream_ctx = StreamContext::new(generation_stream);

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

                        next_y_opt = Some(next_token);
                    }

                    // Block on the CURRENT token
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

                    profiler.step();

                    if let Some(next_y) = next_y_opt {
                        y = next_y;
                    } else {
                        break;
                    }

                    if (step + 1) % 256 == 0 {
                        crate::array::synchronize_and_clear_cache();
                    }
                }

                profiler.snapshot_memory_after();
                profiler.report();
            }

            // _compiled_guard dropped here (if Some), calling mlx_qwen35_compiled_reset()

            // Decode text
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

            // Compute performance metrics if requested
            let performance = if let (Some(gen_start), Some(first_tok)) =
                (generation_start, first_token_instant)
            {
                let generation_end = std::time::Instant::now();
                let prompt_toks = tokens.len() as f64;
                let gen_toks = generated_tokens.len() as f64;
                let ttft_ms = first_tok.duration_since(gen_start).as_secs_f64() * 1000.0;
                let decode_ms = generation_end.duration_since(first_tok).as_secs_f64() * 1000.0;
                Some(crate::profiling::PerformanceMetrics {
                    ttft_ms,
                    prefill_tokens_per_second: if ttft_ms > 0.0 {
                        prompt_toks / (ttft_ms / 1000.0)
                    } else {
                        0.0
                    },
                    decode_tokens_per_second: if decode_ms > 0.0 && gen_toks > 1.0 {
                        (gen_toks - 1.0) / (decode_ms / 1000.0)
                    } else {
                        0.0
                    },
                })
            } else {
                None
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

        // Tokenize messages using chat template
        let tokenizer = self
            .tokenizer
            .clone()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))?;

        // Detect images in messages
        let has_images = messages
            .iter()
            .any(|m| m.images.as_ref().is_some_and(|imgs| !imgs.is_empty()));

        // Clone Arcs and data needed for the closure
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

        // Check if compiled path will be used (C++ weights loaded from safetensors).
        let use_compiled = unsafe { mlx_sys::mlx_qwen35_weight_count() } > 0;

        // Capture start time BEFORE compiled mutex + spawn_blocking so TTFT
        // reflects the full user-perceived latency.
        let report_perf = config.report_performance.unwrap_or(false);
        let generation_start = if report_perf {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // Serialize compiled lifecycle — prevents concurrent C++ global corruption.
        // Acquire in async context, move into tokio::spawn so it stays held during generation.
        let compiled_lock = if use_compiled {
            Some(DENSE_COMPILED_MUTEX.lock().await)
        } else {
            None
        };

        let cancelled = Arc::new(AtomicBool::new(false));
        let cancelled_inner = cancelled.clone();

        let callback = Arc::new(callback);

        tokio::spawn(async move {
            // Hold the compiled lock for the duration of generation
            let _compiled_lock = compiled_lock;

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

                    let mut first_token_instant: Option<std::time::Instant> = None;

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

                    let eos_id = model_config.eos_token_id as u32;
                    let mut generated_tokens: Vec<u32> = Vec::new();
                    let mut finish_reason = String::from("length");

                    let embedding_weight_t = embedding_weight.transpose(Some(&[1, 0]))?;
                    let generation_stream = Stream::new(DeviceType::Gpu);
                    let model_size_bytes = model_config.estimate_memory_bytes() as usize;
                    let _wired_ctx = crate::stream::WiredLimitContext::new(
                        model_size_bytes,
                        vec![generation_stream],
                    );

                    // Profiler — covers both compiled and rust chat_stream decode paths
                    let mut profiler =
                        crate::decode_profiler::DecodeProfiler::new("chat_stream", "qwen3_5");
                    profiler.set_prompt_tokens(tokens.len() as u32);
                    profiler.snapshot_memory_before();

                    // === VLM or text prefill branching ===
                    // vlm_compiled_init_done: true if vlm_prefill already called compiled_init_from_prefill
                    profiler.begin_prefill();
                    let (mut last_logits, seq_len, vlm_compiled_init_done) =
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

                            let (logits, _rope_deltas) = vlm_prefill(
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
                                max_new_tokens,
                                generation_stream,
                                &vision_cache_stream,
                            )?;

                            // VLM prefill already did compiled_init_from_prefill for compiled path
                            let vlm_seq_len = final_tokens.len() as i64;
                            (logits, vlm_seq_len, use_compiled)
                        } else {
                            // --- Standard text prefill path ---
                            let prompt = MxArray::from_uint32(&tokens, &[1, tokens.len() as i64])?;

                            // Init fresh caches (old ones dropped on overwrite)
                            *caches_guard = Some(
                                (0..model_config.num_layers as usize)
                                    .map(|i| {
                                        if model_config.is_linear_layer(i) {
                                            Qwen3_5LayerCache::new_linear()
                                        } else {
                                            Qwen3_5LayerCache::new_full_attention()
                                        }
                                    })
                                    .collect(),
                            );

                            let logits = {
                                let _stream_ctx = StreamContext::new(generation_stream);
                                forward_inner(
                                    &prompt,
                                    &embedding_weight,
                                    &mut layers_guard,
                                    &mut caches_guard,
                                    &final_norm_guard,
                                    &lm_head_guard,
                                    fa_idx,
                                    Some(&embedding_weight_t),
                                )?
                            };

                            let seq_len = logits.shape_at(1)?;
                            let last_logits = logits.slice_axis(1, seq_len - 1, seq_len)?;
                            let last_logits = last_logits.squeeze(Some(&[1]))?;
                            (last_logits, seq_len, false)
                        };
                    profiler.end_prefill();

                    // Track token history for repetition penalty
                    let mut token_history: Vec<u32> = tokens.clone();

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

                    let _compiled_guard = if use_compiled {
                        Some(CompiledResetGuard)
                    } else {
                        None
                    };

                    if use_compiled {
                        if vlm_compiled_init_done {
                            // VLM prefill already called compiled_init_from_prefill and
                            // transferred caches — just drop the locks we don't need.
                            drop(layers_guard);
                            drop(final_norm_guard);
                            drop(lm_head_guard);
                            drop(caches_guard);
                        } else {
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
                            // Drop non-cache locks — not needed during compiled decode
                            drop(layers_guard);
                            drop(final_norm_guard);
                            drop(lm_head_guard);
                            // Keep caches_guard alive through init_from_prefill so cache_ptrs
                            // (raw pointers into the cache MxArrays) remain valid.
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
                                    prefill_len,
                                );
                            }
                            // C++ has copied arrays into g_compiled_caches — safe to release
                            drop(caches_guard);
                        }

                        // Compiled C++ decode loop
                        profiler.set_label("chat_stream_compiled");
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
                            {
                                let _stream_ctx = StreamContext::new(generation_stream);
                                let next_ids = y.reshape(&[1, 1])?;
                                let mut logits = forward_compiled(&next_ids, &embedding_weight)?;
                                if repetition_penalty != 1.0 {
                                    logits = apply_repetition_penalty(
                                        &logits,
                                        &token_history,
                                        repetition_penalty,
                                        Some(repetition_context_size),
                                    )?;
                                }
                                let next_token = sample(&logits, sampling_config)?;
                                eval_token_and_compiled_caches(&next_token);
                                y = next_token;
                            }

                            profiler.step();

                            if (step + 1) % 256 == 0 {
                                crate::array::synchronize_and_clear_cache();
                            }
                        }
                        profiler.snapshot_memory_after();
                        profiler.report();
                    } else {
                        // Rust fallback decode loop (locks held for entire loop)
                        profiler.set_label("chat_stream_rust");
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
                            {
                                let _stream_ctx = StreamContext::new(generation_stream);
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
                            }

                            profiler.step();

                            if (step + 1) % 256 == 0 {
                                crate::array::synchronize_and_clear_cache();
                            }
                        }
                        profiler.snapshot_memory_after();
                        profiler.report();
                    }

                    // _compiled_guard dropped here (if Some), calling mlx_qwen35_compiled_reset()

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

                    // Compute performance metrics if requested
                    let perf_metrics = if let (Some(gen_start), Some(first_tok)) =
                        (generation_start, first_token_instant)
                    {
                        let generation_end = std::time::Instant::now();
                        let prompt_toks = tokens.len() as f64;
                        let gen_toks = generated_tokens.len() as f64;
                        let ttft_ms = first_tok.duration_since(gen_start).as_secs_f64() * 1000.0;
                        let decode_ms =
                            generation_end.duration_since(first_tok).as_secs_f64() * 1000.0;
                        Some(crate::profiling::PerformanceMetrics {
                            ttft_ms,
                            prefill_tokens_per_second: if ttft_ms > 0.0 {
                                prompt_toks / (ttft_ms / 1000.0)
                            } else {
                                0.0
                            },
                            decode_tokens_per_second: if decode_ms > 0.0 && gen_toks > 1.0 {
                                (gen_toks - 1.0) / (decode_ms / 1000.0)
                            } else {
                                0.0
                            },
                        })
                    } else {
                        None
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
                            performance: perf_metrics,
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

    /// Get the number of parameters in the model.
    #[napi]
    pub fn num_parameters(&self) -> i64 {
        let h = self.config.hidden_size as i64;
        let v = self.config.vocab_size as i64;
        let n = self.config.num_layers as usize;
        let dense_i = self.config.intermediate_size as i64;

        // Embedding + LM head
        let mut total = v * h;
        if !self.config.tie_word_embeddings {
            total += v * h;
        }

        let kd = self.config.linear_key_dim() as i64;
        let vd = self.config.linear_value_dim() as i64;

        for layer_idx in 0..n {
            let is_linear = self.config.is_linear_layer(layer_idx);

            // Attention params
            if is_linear {
                let num_vh = self.config.linear_num_value_heads as i64;
                let vhd = self.config.linear_value_head_dim as i64;
                total += h * (kd * 2 + vd * 2) // in_proj_qkvz
                    + h * (num_vh * 2) // in_proj_ba
                    + (kd * 2 + vd) * self.config.linear_conv_kernel_dim as i64 // conv1d
                    + vd * h // out_proj
                    + num_vh // dt_bias
                    + num_vh // a_log
                    + vhd; // norm (RMSNormGated weight)
            } else {
                let d = self.config.head_dim as i64;
                total += h * h * 2 // q_proj (2x for gate)
                    + h * (self.config.num_kv_heads as i64 * d) * 2 // k, v
                    + h * h // o_proj
                    + d * 2; // q_norm + k_norm (each [head_dim])
            }

            // Dense MLP params
            total += 3 * h * dense_i;

            // Norms (2 per layer)
            total += h * 2;
        }

        // Final norm
        total += h;

        total
    }
}

/// Forward pass through the model, acquiring all necessary locks.
///
/// This is a free function (not a method) so it can be called from
/// within spawn_blocking closures that have cloned the Arc fields.
///
/// `embedding_weight_t` is an optional pre-transposed embedding weight for
/// tied embeddings. When provided, avoids recomputing the transpose every step.
/// Lock-free forward pass that takes direct mutable refs instead of Arcs.
/// Caller must acquire locks once and hold them for the entire prefill+decode sequence.
fn forward_inner(
    input_ids: &MxArray,
    embedding_weight: &MxArray,
    layers: &mut [DecoderLayer],
    caches: &mut Option<Vec<Qwen3_5LayerCache>>,
    final_norm: &RMSNorm,
    lm_head: &Option<Linear>,
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

/// Evaluate the sampled token AND all cache arrays together in a single async_eval call.
///
/// This is the key fix for performance: without evaluating the cache arrays, they remain
/// as lazy computation nodes. Each decode step's graph then includes the entire lazy chain
/// from all previous steps (O(N^2) graph growth), causing GPU execution time to grow
/// linearly with the number of decode steps.
///
/// By evaluating cache arrays here (alongside next_token), MLX materializes them and
/// breaks the dependency chain (via arr.detach() internally). The next step then starts
/// with a clean graph bounded to exactly one decode step.
///
/// This matches Python mlx-lm's behavior: async_eval(y, logprobs) evaluates y (the token)
/// whose graph transitively causes all cache state INPUTS to be detached/materialized.
// Internal methods (not NAPI-exported)
impl Qwen3_5Model {
    /// Forward pass from embeddings through all layers (internal, acquires locks).
    fn forward_from_embeddings(&self, hidden_states: &MxArray) -> Result<MxArray> {
        let mut h = hidden_states.clone();

        // Acquire write locks for layers and caches (forward mutates caches)
        let mut layers_guard = self
            .layers
            .write()
            .map_err(|_| Error::from_reason("Failed to acquire layers write lock"))?;
        let mut caches_guard = self
            .caches
            .write()
            .map_err(|_| Error::from_reason("Failed to acquire caches write lock"))?;

        // Create masks
        let fa_mask = self.create_fa_mask(hidden_states, &caches_guard)?;
        let ssm_mask = self.create_ssm_mask(hidden_states)?;

        // Forward through layers
        let num_layers = layers_guard.len();
        for i in 0..num_layers {
            let mask = if layers_guard[i].is_linear() {
                ssm_mask.as_ref()
            } else {
                fa_mask.as_ref()
            };

            let cache = caches_guard.as_mut().map(|c| &mut c[i]);
            h = layers_guard[i].forward(&h, mask, cache, None)?;
        }

        // Drop locks early -- no longer needed
        drop(layers_guard);
        drop(caches_guard);

        // Final norm
        let final_norm_guard = self
            .final_norm
            .read()
            .map_err(|_| Error::from_reason("Failed to acquire final_norm read lock"))?;
        let h = final_norm_guard.forward(&h)?;
        drop(final_norm_guard);

        // LM head
        let lm_head_guard = self
            .lm_head
            .read()
            .map_err(|_| Error::from_reason("Failed to acquire lm_head read lock"))?;
        match &*lm_head_guard {
            Some(head) => head.forward(&h),
            None => {
                // tie_word_embeddings: use embedding weight as linear
                let weight = self.embedding.get_weight();
                let weight_t = weight.transpose(Some(&[1, 0]))?;
                h.matmul(&weight_t)
            }
        }
    }

    /// Forward pass from pre-computed embeddings with M-RoPE position IDs (VLM mode).
    ///
    /// Used by `Qwen3_5VLModel` to inject vision features into the text stream.
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
        let ssm_mask = self.create_ssm_mask(hidden_states)?;

        let num_layers = layers_guard.len();
        for i in 0..num_layers {
            let mask = if layers_guard[i].is_linear() {
                ssm_mask.as_ref()
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

    /// Get embeddings for input IDs (used by VLModel).
    pub fn get_embeddings(&self, input_ids: &MxArray) -> Result<MxArray> {
        self.embedding.forward(input_ids)
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

    /// Create causal attention mask for full attention layers.
    fn create_fa_mask(
        &self,
        hidden_states: &MxArray,
        caches: &Option<Vec<Qwen3_5LayerCache>>,
    ) -> Result<Option<MxArray>> {
        let seq_len = hidden_states.shape_at(1)?;
        if seq_len <= 1 && caches.is_some() {
            // Single-token decode step with cache -- no mask needed
            return Ok(None);
        }

        // Get cache offset from the first full attention layer
        let offset = caches
            .as_ref()
            .map(|c| c[self.fa_idx].offset())
            .unwrap_or(0);

        // Create causal mask using existing utility
        create_causal_mask(
            seq_len as i32,
            Some(offset),
            None, // no sliding window
        )
        .map(Some)
    }

    /// Create mask for linear attention (SSM) layers.
    /// Always returns None — mlx-vlm never creates an SSM mask for ArraysCache.
    /// An all-ones mask is a no-op that adds unnecessary graph nodes and Metal overhead.
    fn create_ssm_mask(&self, _hidden_states: &MxArray) -> Result<Option<MxArray>> {
        Ok(None)
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
    all_images: &[Vec<u8>],
    image_processor: &Qwen35VLImageProcessor,
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
            *lru = lru_gen; // bump LRU
            tracing::debug!("Vision cache HIT for hash {:016x}", combined_hash);
            Some((features.clone(), grid.clone()))
        } else {
            None
        }
    };

    let (vision_features, grid) = if let Some((features, grid)) = cached {
        // Cache hit — skip image processing AND vision encoder
        (features, grid)
    } else {
        // Cache miss — process images and run vision encoder
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

        // Store in cache with LRU eviction
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

    // === STEP 4: Prefill with M-RoPE ===
    let use_compiled = unsafe { mlx_sys::mlx_qwen35_weight_count() } > 0;

    let (last_logits, _seq_len) = if use_compiled {
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

            // Clean up VLM prefill state (caches now owned by compiled decode)
            sys::mlx_qwen35_vlm_reset();
        }

        let logits = MxArray::from_handle(output_ptr, "vlm_cpp_prefill")?;
        // logits is already [1, vocab] from C++ prefill
        (logits, seq_len_i32 as i64)
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
            let seq_len = h.shape_at(1)?;

            let fa_mask = if seq_len > 1 {
                Some(crate::array::mask::create_causal_mask(
                    seq_len as i32,
                    None,
                    None,
                )?)
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
        (last_logits, seq_len)
    };

    Ok((last_logits, rope_deltas))
}
