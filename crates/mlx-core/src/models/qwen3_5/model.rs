use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use futures::TryFutureExt;
use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi_derive::napi;
use tokio::sync::Mutex as TokioMutex;
use tracing::{info, warn};

use crate::array::MxArray;
use crate::models::qwen3::{BatchGenerationResult, GenerationConfig, GenerationResult};
use crate::nn::{Embedding, Linear, RMSNorm};
use crate::sampling::{SamplingConfig, sample};
use crate::stream::{DeviceType, Stream, StreamContext};
use crate::tokenizer::{ChatMessage, Qwen3Tokenizer, ToolDefinition};
use crate::tools::ToolCallResult;

use super::chat_common;
use super::chat_common::{
    apply_all_penalties, compute_performance_metrics, extract_chat_params, finalize_chat_result,
    save_cache_state, verify_cache_prefix,
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

/// Compute a combined cache key from raw image bytes.
pub(crate) fn compute_image_cache_key(all_images: &[Vec<u8>]) -> u64 {
    let individual_hashes: Vec<u64> = all_images.iter().map(|img| hash_image_bytes(img)).collect();
    combine_image_hashes(&individual_hashes)
}

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
/// concurrent model loads from swapping weights mid-generation.
pub(crate) fn acquire_compiled_weight_guard(
    model_id: u64,
) -> Option<std::sync::RwLockReadGuard<'static, ()>> {
    let guard = COMPILED_WEIGHTS_RWLOCK.read().unwrap();
    if unsafe { mlx_sys::mlx_qwen35_get_model_id() } == model_id {
        Some(guard)
    } else {
        None
    }
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
    /// Reuse KV cache across chat() calls for incremental prefill. Default: true.
    /// When true, the model preserves its KV cache after generation. On the next
    /// chat() call, it prefix-matches the new token sequence against the cached
    /// tokens and only prefills the delta — avoiding redundant computation for
    /// multi-turn conversations.
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
    /// Whether this delta chunk contains reasoning/thinking content.
    /// true = reasoning (inside <think>...</think>), false = content (after </think>).
    /// Only present on intermediate (non-final) chunks.
    #[napi(ts_type = "boolean | undefined")]
    pub is_reasoning: Option<bool>,
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
    /// Token history for KV cache reuse across chat() calls.
    /// Stores the full token sequence (template + generated) from the last call.
    cached_token_history: Arc<RwLock<Vec<u32>>>,
    /// Image cache key for VLM cache reuse (None for text-only conversations).
    cached_image_key: Arc<RwLock<Option<u64>>>,
    /// Rope deltas from VLM prefill, needed for cache reuse on subsequent turns.
    /// Without this, the compiled decode path starts with wrong RoPE positions
    /// when VLM prefill is skipped on Turn 2+.
    cached_rope_deltas: Arc<RwLock<Option<i32>>>,
    /// Unique model instance ID for compiled path ownership.
    /// The C++ global weight map is shared across all models — this ID ensures
    /// inference only uses the compiled path when the weights belong to this model.
    pub(crate) model_id: u64,
    /// Serializes cache state access: held during the entire chat()/chatStream()/generate()
    /// lifecycle. Cache API methods (reset_caches, take_cache, set_cache) use try_lock
    /// and return an error if generation is in-flight.
    generation_lock: Arc<TokioMutex<()>>,
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
            cached_token_history: Arc::new(RwLock::new(Vec::new())),
            cached_image_key: Arc::new(RwLock::new(None)),
            cached_rope_deltas: Arc::new(RwLock::new(None)),
            model_id: QWEN35_MODEL_ID_COUNTER.fetch_add(1, Ordering::Relaxed),
            generation_lock: Arc::new(TokioMutex::new(())),
        })
    }

    /// Initialize caches for incremental generation.
    #[napi]
    pub fn init_caches(&self) -> Result<()> {
        let _guard = self.generation_lock.try_lock().map_err(|_| {
            Error::from_reason("Cannot init caches while generation is in progress")
        })?;
        self.init_caches_inner()
    }

    /// Init caches without checking the generation lock (for internal use
    /// by generate_for_training_sync which already holds the lock).
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

    /// Reset all caches.
    #[napi]
    pub fn reset_caches(&self) -> Result<()> {
        let _guard = self.generation_lock.try_lock().map_err(|_| {
            Error::from_reason("Cannot reset caches while generation is in progress")
        })?;
        self.reset_caches_inner()
    }

    /// Reset caches without checking the generation lock.
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

    /// Clear cached token history, image key, and rope deltas.
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
            "qwen3_5",
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
    pub async fn load(path: String) -> Result<Qwen3_5Model> {
        persistence::load(&path).await
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

        // Hold generation lock for the entire cache-read + generation + cache-write lifecycle.
        let gen_lock = self.generation_lock.clone();
        let _gen_guard = gen_lock.lock().await;

        // Clone Arcs and data needed for the closure
        let embedding_weight = self.embedding.get_weight();
        let layers_arc = self.layers.clone();
        let final_norm_arc = self.final_norm.clone();
        let lm_head_arc = self.lm_head.clone();
        let caches_arc = self.caches.clone();
        let model_config = self.config.clone();
        let tokenizer = self.tokenizer.clone();

        let prompt_tokens = prompt_tokens.clone();
        let model_id = self.model_id;

        // Check if compiled path will be used (C++ weights belong to this model).
        // Must be checked before spawn_blocking so we can acquire the mutex in async context.
        let use_compiled = unsafe { mlx_sys::mlx_qwen35_get_model_id() } == model_id;

        // Serialize compiled lifecycle — prevents concurrent C++ global corruption
        let _compiled_lock = if use_compiled {
            Some(DENSE_COMPILED_MUTEX.lock().await)
        } else {
            None
        };

        napi::bindgen_prelude::spawn_blocking(move || {
            let _weight_guard = if use_compiled {
                acquire_compiled_weight_guard(model_id)
            } else {
                None
            };
            let use_compiled = _weight_guard.is_some();

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

            // Prefill: chunked forward pass on prompt (stays on GPU, no roundtrip)
            profiler.begin_prefill();
            let logits = chunked_prefill(
                &prompt_tokens,
                &embedding_weight,
                &mut layers_guard,
                &mut caches_guard,
                &final_norm_guard,
                &lm_head_guard,
                Some(&embedding_weight_t),
                generation_stream,
            )?;
            profiler.end_prefill();

            // Get last token logits: [1, vocab]
            let last_chunk_len = logits.shape_at(1)?;
            let last_logits = logits.slice_axis(1, last_chunk_len - 1, last_chunk_len)?;
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
                // Use TOTAL prompt length (not last chunk length) for correct
                // RoPE offset and KV capacity pre-allocation.
                use mlx_sys as sys;
                let prefill_len = prompt_tokens.shape_at(1)? as i32;
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

        let reuse_cache = config.reuse_cache.unwrap_or(true);

        let gen_lock = self.generation_lock.clone();
        let _gen_guard = gen_lock.lock().await;

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
        let cached_token_history_arc = self.cached_token_history.clone();
        let cached_image_key_arc = self.cached_image_key.clone();
        let cached_rope_deltas_arc = self.cached_rope_deltas.clone();
        let model_config = self.config.clone();

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

        // Check if compiled path will be used (C++ weights belong to this model).
        let use_compiled = unsafe { mlx_sys::mlx_qwen35_get_model_id() } == model_id;

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
            // Re-validate compiled path under weight lock.
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

            let mut first_token_instant: Option<std::time::Instant> = None;

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

            // === VLM image processing (before cache check, needed for expanded tokens) ===
            // Save processed images for reuse in VLM prefill to avoid double processing.
            let sms = spatial_merge_size.unwrap_or(2);
            let (expanded_tokens, current_image_cache_key, vlm_processed) = if has_images {
                if let (Some(_vision_enc), Some(img_proc)) =
                    (vision_encoder_arc.as_ref(), image_processor_arc.as_ref())
                {
                    let all_images = extract_images_from_messages(&messages);
                    let image_refs: Vec<&[u8]> = all_images.iter().map(|v| v.as_slice()).collect();
                    let processed_pre = img_proc.process_many(&image_refs)?;
                    let num_image_tokens =
                        compute_num_image_tokens(&processed_pre.grid_thw(), sms)?;
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
            let cached_token_history_guard = cached_token_history_arc
                .read()
                .map_err(|_| Error::from_reason("Failed to read cached token history"))?;
            let cached_prefix_len = verify_cache_prefix(
                reuse_cache,
                has_images,
                &tokens,
                &expanded_tokens,
                current_image_cache_key,
                &cached_token_history_guard,
                &cached_image_key_arc,
                caches_guard.is_some(),
            )?;
            drop(cached_token_history_guard);

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

            // Zero-delta guard: if entire prompt was cached (exact same input repeated),
            // reset caches and do a full re-prefill. GDN recurrence state cannot be
            // rewound, so full re-prefill is the only correct approach for Qwen3.5.
            // Also reset cached_prefix_len so VLM routing correctly triggers vlm_prefill.
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
                    expanded_tokens.clone()
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

            let embedding_weight_t = embedding_weight.transpose(Some(&[1, 0]))?;
            let generation_stream = Stream::new(DeviceType::Gpu);
            let model_size_bytes = model_config.estimate_memory_bytes() as usize;
            let _wired_ctx =
                crate::stream::WiredLimitContext::new(model_size_bytes, vec![generation_stream]);

            // Profiler — covers both compiled and rust chat decode paths
            let mut profiler = crate::decode_profiler::DecodeProfiler::new("chat", "qwen3_5");
            profiler.set_prompt_tokens(prefill_tokens.len() as u32);
            profiler.snapshot_memory_before();

            // === VLM or text prefill branching ===
            // vlm_compiled_init_done: true if vlm_prefill already called compiled_init_from_prefill
            profiler.begin_prefill();
            let (mut last_logits, seq_len, vlm_compiled_init_done) =
                if has_images && cached_prefix_len == 0 {
                    // --- VLM full prefill (first call or different images) ---
                    // Reuse expanded_tokens and processed images from pre-cache-check step.
                    if let Some(vision_enc) = vision_encoder_arc.as_ref() {
                        let final_tokens = &expanded_tokens;
                        let processed = vlm_processed
                            .as_ref()
                            .ok_or_else(|| Error::from_reason("VLM processed images missing"))?;
                        let image_cache_key = current_image_cache_key;

                        let input_ids =
                            MxArray::from_uint32(final_tokens, &[1, final_tokens.len() as i64])?;

                        let (logits, rope_deltas, vlm_compiled) = vlm_prefill(
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
                            &model_config,
                            max_new_tokens,
                            generation_stream,
                            &vision_cache,
                            model_id,
                        )?;

                        // Save rope_deltas for cache reuse on subsequent turns
                        if let Ok(mut rd) = cached_rope_deltas_arc.write() {
                            *rd = Some(rope_deltas as i32);
                        }

                        let vlm_seq_len = final_tokens.len() as i64;
                        (logits, vlm_seq_len, vlm_compiled)
                    } else {
                        return Err(Error::from_reason(
                            "VLM prefill requested but vision encoder/processor not loaded",
                        ));
                    }
                } else {
                    // --- Text prefill path (text-only OR VLM cache reuse with same images) ---
                    let prompt =
                        MxArray::from_uint32(&prefill_tokens, &[1, prefill_tokens.len() as i64])?;
                    let logits = chunked_prefill(
                        &prompt,
                        &embedding_weight,
                        &mut layers_guard,
                        &mut caches_guard,
                        &final_norm_guard,
                        &lm_head_guard,
                        Some(&embedding_weight_t),
                        generation_stream,
                    )?;

                    let seq_len = logits.shape_at(1)?;
                    let last_logits = logits.slice_axis(1, seq_len - 1, seq_len)?;
                    let last_logits = last_logits.squeeze(Some(&[1]))?;
                    // seq_len for the C++ init is the TOTAL tokens (cached + new)
                    // For VLM cache reuse, use expanded_tokens length; for text-only, use tokens length
                    let total_seq_len = if has_images {
                        expanded_tokens.len() as i64
                    } else {
                        tokens.len() as i64
                    };
                    (last_logits, total_seq_len, false)
                };
            profiler.end_prefill();

            // Track token history for repetition penalty
            let mut token_history: Vec<u32> = tokens.clone();

            // Apply repetition penalty to prefill logits
            last_logits = apply_all_penalties(last_logits, &token_history, &p)?;

            let mut y = sample(&last_logits, p.sampling_config)?;
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

                    // VLM cache reuse: apply saved rope_deltas so compiled decode
                    // uses correct M-RoPE positions (vlm_prefill was skipped).
                    if has_images
                        && cached_prefix_len > 0
                        && let Ok(rd) = cached_rope_deltas_arc.read()
                        && let Some(delta) = *rd
                    {
                        unsafe {
                            mlx_sys::mlx_qwen35_compiled_adjust_offset(delta);
                        }
                    }
                }

                // For text-only conversations, clear any stale cached rope deltas
                if !has_images && let Ok(mut rd) = cached_rope_deltas_arc.write() {
                    *rd = None;
                }

                // Compiled C++ decode loop (pipelined — submit N+1 before eval N)
                profiler.set_label("chat_compiled");

                let starts_in_thinking = enable_thinking.unwrap_or(true);
                let mut reasoning_tracker = chat_common::ReasoningTracker::new(
                    starts_in_thinking,
                    p.thinking_token_budget,
                    think_end_id,
                );

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

                // === Export caches from C++ before CompiledResetGuard drops ===
                if reuse_cache {
                    let num_layers = model_config.num_layers as usize;
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
            } else {
                // Rust fallback decode loop — pipelined like mlx-lm:
                // Build next step's graph before blocking on current token.
                profiler.set_label("chat_rust");

                let starts_in_thinking = enable_thinking.unwrap_or(true);
                let mut reasoning_tracker = chat_common::ReasoningTracker::new(
                    starts_in_thinking,
                    p.thinking_token_budget,
                    think_end_id,
                );

                // Kick off first token's async eval
                MxArray::async_eval_arrays(&[&y]);

                let mut ops = chat_common::DecodeOps {
                    forward: |ids: &MxArray, emb: &MxArray| -> Result<(MxArray, bool)> {
                        let logits = forward_inner(
                            ids,
                            emb,
                            &mut layers_guard,
                            &mut caches_guard,
                            &final_norm_guard,
                            &lm_head_guard,
                            Some(&embedding_weight_t),
                        )?;
                        Ok((logits, true)) // needs squeeze
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

            // _compiled_guard dropped here (if Some), calling mlx_qwen35_compiled_reset()

            // === Save token history and image key for cache reuse on next call ===
            save_cache_state(
                p.reuse_cache,
                has_images,
                &generated_tokens,
                &finish_reason,
                &tokens,
                Some(&expanded_tokens),
                current_image_cache_key,
                &cached_token_history_arc,
                &cached_image_key_arc,
                &cached_rope_deltas_arc,
                &caches_arc,
            )?;

            // Compute performance metrics
            let performance = compute_performance_metrics(
                generation_start,
                first_token_instant,
                prefill_tokens.len(),
                generated_tokens.len(),
            );

            finalize_chat_result(
                &tokenizer_for_decode,
                &generated_tokens,
                finish_reason,
                think_end_id,
                think_end_str.as_deref(),
                performance,
                p.include_reasoning,
                enable_thinking.unwrap_or(true),
            )
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
            thinking_token_budget: None,
            include_reasoning: None,
            reasoning_effort: None,
            report_performance: None,
            reuse_cache: None,
        });

        let reuse_cache = config.reuse_cache.unwrap_or(true);
        let report_perf = config.report_performance.unwrap_or(false);

        // Use lock_owned() so the guard is 'static and can be moved into tokio::spawn.
        let gen_guard = Arc::clone(&self.generation_lock).lock_owned().await;

        let tokenizer = self
            .tokenizer
            .clone()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))?;

        let has_images = messages
            .iter()
            .any(|m| m.images.as_ref().is_some_and(|imgs| !imgs.is_empty()));

        // Clone Arcs and data needed for the closure
        let embedding_weight = self.embedding.get_weight();
        let layers_arc = self.layers.clone();
        let final_norm_arc = self.final_norm.clone();
        let lm_head_arc = self.lm_head.clone();
        let caches_arc = self.caches.clone();
        let cached_token_history_arc = self.cached_token_history.clone();
        let cached_image_key_arc = self.cached_image_key.clone();
        let cached_rope_deltas_arc = self.cached_rope_deltas.clone();
        let model_config = self.config.clone();

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
        let vision_cache_stream = self.vision_cache.clone();
        let model_id = self.model_id;

        // Check if compiled path will be used (C++ weights belong to this model).
        let use_compiled = unsafe { mlx_sys::mlx_qwen35_get_model_id() } == model_id;

        // Capture start time BEFORE compiled mutex + spawn_blocking so TTFT
        // reflects the full user-perceived latency.
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
            let _gen_guard = gen_guard;
            let _compiled_lock = compiled_lock;

            let callback_err = callback.clone();
            let result =
                napi::bindgen_prelude::spawn_blocking(move || -> std::result::Result<(), Error> {
                    // Re-validate compiled path under weight lock.
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

                    let tool_defs = config.tools.as_deref();
                    let enable_thinking = chat_common::resolve_enable_thinking(&config);
                    let tokens = tokenizer.apply_chat_template_sync(
                        &messages,
                        Some(true),
                        tool_defs,
                        enable_thinking,
                    )?;

                    let mut first_token_instant: Option<std::time::Instant> = None;

                    let p = chat_common::extract_chat_params(&config);

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

                    // === VLM image processing (before cache check, needed for expanded tokens) ===
                    let sms = spatial_merge_size.unwrap_or(2);
                    let (expanded_tokens, current_image_cache_key, vlm_processed) = if has_images {
                        if let (Some(_vision_enc), Some(img_proc)) =
                            (vision_encoder_arc.as_ref(), image_processor_arc.as_ref())
                        {
                            let all_images = extract_images_from_messages(&messages);
                            let image_refs: Vec<&[u8]> =
                                all_images.iter().map(|v| v.as_slice()).collect();
                            let processed_pre = img_proc.process_many(&image_refs)?;
                            let num_image_tokens =
                                compute_num_image_tokens(&processed_pre.grid_thw(), sms)?;
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
                    let cached_token_history_guard = cached_token_history_arc
                        .read()
                        .map_err(|_| Error::from_reason("Failed to read cached token history"))?;
                    let cached_prefix_len = if reuse_cache {
                        let cached = &*cached_token_history_guard;
                        if has_images {
                            // VLM: check image_cache_key matches AND expanded token prefix matches
                            let cached_img_key = cached_image_key_arc.read().map_err(|_| {
                                Error::from_reason("Failed to read cached image key")
                            })?;
                            if let Some(cached_key) = *cached_img_key {
                                if cached_key == current_image_cache_key
                                    && !cached.is_empty()
                                    && expanded_tokens.len() >= cached.len()
                                    && expanded_tokens[..cached.len()] == cached[..]
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
                            // Text-only: existing logic unchanged
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
                            expanded_tokens.clone()
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
                    profiler.set_prompt_tokens(prefill_tokens.len() as u32);
                    profiler.snapshot_memory_before();

                    // === VLM or text prefill branching ===
                    // vlm_compiled_init_done: true if vlm_prefill already called compiled_init_from_prefill
                    profiler.begin_prefill();
                    let (mut last_logits, seq_len, vlm_compiled_init_done) =
                        if has_images && cached_prefix_len == 0 {
                            // --- VLM full prefill (first call or different images) ---
                            // Reuse expanded_tokens and processed images from pre-cache-check step.
                            if let Some(vision_enc) = vision_encoder_arc.as_ref() {
                                let final_tokens = &expanded_tokens;
                                let processed = vlm_processed.as_ref().ok_or_else(|| {
                                    Error::from_reason("VLM processed images missing")
                                })?;
                                let image_cache_key = current_image_cache_key;

                                let input_ids = MxArray::from_uint32(
                                    final_tokens,
                                    &[1, final_tokens.len() as i64],
                                )?;

                                let (logits, rope_deltas, vlm_compiled) = vlm_prefill(
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
                                    &model_config,
                                    p.max_new_tokens,
                                    generation_stream,
                                    &vision_cache_stream,
                                    model_id,
                                )?;

                                // Save rope_deltas for cache reuse on subsequent turns
                                if let Ok(mut rd) = cached_rope_deltas_arc.write() {
                                    *rd = Some(rope_deltas as i32);
                                }

                                let vlm_seq_len = final_tokens.len() as i64;
                                (logits, vlm_seq_len, vlm_compiled)
                            } else {
                                return Err(Error::from_reason(
                                    "VLM prefill requested but vision encoder/processor not loaded",
                                ));
                            }
                        } else {
                            // --- Text prefill path (text-only OR VLM cache reuse with same images) ---
                            let prompt = MxArray::from_uint32(
                                &prefill_tokens,
                                &[1, prefill_tokens.len() as i64],
                            )?;
                            let logits = chunked_prefill(
                                &prompt,
                                &embedding_weight,
                                &mut layers_guard,
                                &mut caches_guard,
                                &final_norm_guard,
                                &lm_head_guard,
                                Some(&embedding_weight_t),
                                generation_stream,
                            )?;

                            let seq_len = logits.shape_at(1)?;
                            let last_logits = logits.slice_axis(1, seq_len - 1, seq_len)?;
                            let last_logits = last_logits.squeeze(Some(&[1]))?;
                            // seq_len for the C++ init is the TOTAL tokens (cached + new)
                            // For VLM cache reuse, use expanded_tokens length; for text-only, use tokens length
                            let total_seq_len = if has_images {
                                expanded_tokens.len() as i64
                            } else {
                                tokens.len() as i64
                            };
                            (last_logits, total_seq_len, false)
                        };
                    profiler.end_prefill();

                    // Track token history for repetition penalty
                    let mut token_history: Vec<u32> = tokens.clone();

                    // Apply penalties to prefill logits
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
                            let max_kv_len = ((prefill_len + p.max_new_tokens + 255) / 256) * 256;
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

                            // VLM cache reuse: apply saved rope_deltas so compiled decode
                            // uses correct M-RoPE positions (vlm_prefill was skipped).
                            if has_images
                                && cached_prefix_len > 0
                                && let Ok(rd) = cached_rope_deltas_arc.read()
                                && let Some(delta) = *rd
                            {
                                unsafe {
                                    mlx_sys::mlx_qwen35_compiled_adjust_offset(delta);
                                }
                            }
                        }

                        // For text-only conversations, clear any stale cached rope deltas
                        if !has_images && let Ok(mut rd) = cached_rope_deltas_arc.write() {
                            *rd = None;
                        }

                        // Compiled C++ decode loop (pipelined — submit N+1 before eval N)
                        profiler.set_label("chat_stream_compiled");

                        let mut reasoning_tracker = chat_common::ReasoningTracker::new(
                            starts_in_thinking,
                            p.thinking_token_budget,
                            think_end_id,
                        );

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
                                callback: callback,
                                cancelled: cancelled_inner,
                                decode_stream: decode_stream,
                                tokenizer: tokenizer_for_decode,
                                streamed_text_len: streamed_text_len,
                                last_is_reasoning: last_is_reasoning
                            }
                        );

                        // === Export caches from C++ before CompiledResetGuard drops ===
                        if reuse_cache {
                            let num_layers = model_config.num_layers as usize;
                            let mut export_ptrs: Vec<*mut mlx_sys::mlx_array> =
                                vec![std::ptr::null_mut(); num_layers * 2];
                            let exported = unsafe {
                                mlx_sys::mlx_qwen35_export_caches(
                                    export_ptrs.as_mut_ptr(),
                                    (num_layers * 2) as i32,
                                )
                            };
                            if exported > 0 {
                                let cache_offset =
                                    unsafe { mlx_sys::mlx_qwen35_get_cache_offset() };
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
                    } else {
                        // Rust fallback decode loop (pipelined)
                        profiler.set_label("chat_stream_rust");

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
                                    &mut layers_guard,
                                    &mut caches_guard,
                                    &final_norm_guard,
                                    &lm_head_guard,
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
                                callback: callback,
                                cancelled: cancelled_inner,
                                decode_stream: decode_stream,
                                tokenizer: tokenizer_for_decode,
                                streamed_text_len: streamed_text_len,
                                last_is_reasoning: last_is_reasoning
                            }
                        );
                    }

                    // _compiled_guard dropped here (if Some), calling mlx_qwen35_compiled_reset()

                    // === Save token history and image key for cache reuse on next call ===
                    if reuse_cache {
                        let mut full_history = if has_images {
                            expanded_tokens.clone()
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
                        if let Ok(mut ik) = cached_image_key_arc.write() {
                            *ik = if has_images {
                                Some(current_image_cache_key)
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

                    let text = tokenizer_for_decode
                        .decode_sync(&generated_tokens, true)
                        .unwrap_or_else(|e| {
                            warn!("Failed to decode generated tokens: {}", e);
                            String::new()
                        });

                    // Flush any residual bytes buffered by DecodeStream that
                    // weren't emitted as intermediate chunks.
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
                        let actual_prefill_toks = prefill_tokens.len() as f64;
                        let gen_toks = generated_tokens.len() as f64;
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
                            is_reasoning: None,
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
            map.insert("model_type".to_string(), serde_json::json!("qwen3_5"));
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

/// Default prefill chunk size (tokens per chunk).
/// Matches Python mlx-lm's `prefill_step_size` default of 2048.
const PREFILL_STEP_SIZE: i64 = 2048;

/// Evaluate all cache arrays across all layers to materialize them on GPU.
/// Must be called between prefill chunks to break lazy dependency chains.
fn eval_layer_caches(caches: &Option<Vec<Qwen3_5LayerCache>>) {
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

        // Forward through layers — attention handles causal masking internally
        let num_layers = layers_guard.len();
        for i in 0..num_layers {
            let cache = caches_guard.as_mut().map(|c| &mut c[i]);
            h = layers_guard[i].forward(&h, None, cache, None, true)?;
        }

        drop(layers_guard);
        drop(caches_guard);

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

        let num_layers = layers_guard.len();
        for i in 0..num_layers {
            let cache = caches_guard.as_mut().map(|c| &mut c[i]);
            let layer_pos = if layers_guard[i].is_linear() {
                None
            } else {
                position_ids
            };
            h = layers_guard[i].forward(&h, None, cache, layer_pos, true)?;
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
}

// ========== Training Support Methods ==========
// These methods are Rust-internal only (not exposed via NAPI).
// They implement the TrainableModel trait interface.

impl Qwen3_5Model {
    /// Get model configuration.
    pub(crate) fn get_config(&self) -> Qwen3_5Config {
        self.config.clone()
    }

    /// Create a cheap clone for training sessions.
    /// Arc-clones all shared components, no deep copy.
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

    /// Extract all trainable parameters as a name→array map.
    ///
    /// Parameter naming convention matches HuggingFace format:
    /// - `embedding.weight`
    /// - Linear attention layers: `layers.{i}.linear_attn.{in_proj_qkvz,in_proj_ba,conv1d,norm,out_proj}.weight`
    /// - Linear attention learnable: `layers.{i}.linear_attn.{a_log,dt_bias}`
    /// - Full attention layers: `layers.{i}.self_attn.{q_proj,k_proj,v_proj,o_proj}.weight`
    /// - Full attention norms: `layers.{i}.self_attn.{q_norm,k_norm}.weight`
    /// - All layers: `layers.{i}.mlp.{gate_proj,up_proj,down_proj}.weight`
    /// - All layers: `layers.{i}.{input_layernorm,post_attention_layernorm}.weight`
    /// - `final_norm.weight`
    /// - `lm_head.weight` (if not tied)
    pub(crate) fn get_parameters_for_training(&self) -> Result<HashMap<String, MxArray>> {
        use super::decoder_layer::AttentionType;

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
                    lm.set_weight(updated_param)?;
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
        use super::decoder_layer::AttentionType;

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
    /// Uses the compiled C++ forward path when available (~10x faster than Rust).
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

        // Check if compiled path is available (C++ weights belong to this model)
        let use_compiled = unsafe { mlx_sys::mlx_qwen35_get_model_id() } == model_id;

        // Try to acquire compiled mutex (non-blocking, safe from sync context).
        // If locked (concurrent generate() call), fall back to Rust path.
        let compiled_lock = if use_compiled {
            DENSE_COMPILED_MUTEX.try_lock().ok()
        } else {
            None
        };
        let use_compiled = compiled_lock.is_some();

        let _weight_guard = if use_compiled {
            acquire_compiled_weight_guard(model_id)
        } else {
            None
        };
        let use_compiled = _weight_guard.is_some();

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

        // === Prefill (always uses Rust forward — runs once) ===
        let logits = forward_inner(
            input_ids,
            &embedding_weight,
            &mut layers_guard,
            &mut caches_guard,
            &final_norm_guard,
            &lm_head_guard,
            None,
        )?;
        let seq_len = logits.shape_at(1)?;
        let last_logits = logits.slice_axis(1, seq_len - 1, seq_len)?;
        let last_logits = last_logits.squeeze(Some(&[1]))?;
        let input_tokens = input_ids.to_uint32()?;

        let result = if use_compiled {
            // === Compiled C++ decode path ===
            let _compiled_guard = CompiledResetGuard;
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
            // Drop locks not needed during compiled decode
            drop(layers_guard);
            drop(final_norm_guard);
            drop(lm_head_guard);
            // Keep caches_guard alive through init_from_prefill so cache_ptrs remain valid
            unsafe {
                mlx_sys::mlx_qwen35_compiled_init_from_prefill(
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
                    cache_ptrs.as_mut_ptr(),
                    prefill_len,
                );
            }
            // C++ has copied arrays into g_compiled_caches — safe to release
            drop(caches_guard);

            // Decode using compiled forward with synchronous cache eval.
            // Caches are eval'd BEFORE each forward call to ensure the previous step's
            // caches are materialized, breaking lazy dependency chains that would otherwise
            // cause O(N²) graph growth and 100+GB memory.
            let mut forward_fn = |ids: &MxArray| -> Result<MxArray> {
                // Sync-eval compiled caches from the previous step before building new graph.
                // Uses synchronous eval (not async_eval) to avoid interaction issues with
                // the training loop's own synchronous eval calls.
                unsafe { mlx_sys::mlx_qwen35_sync_eval_compiled_caches() };
                let logits = forward_compiled(ids, &embedding_weight)?;
                // forward_compiled returns [1, vocab] but training loop expects [1, 1, vocab]
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
            // _compiled_guard dropped here → mlx_qwen35_compiled_reset()
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
        // Use inner variant — we already hold the generation lock
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
