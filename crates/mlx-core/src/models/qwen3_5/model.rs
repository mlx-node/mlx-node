use std::sync::{Arc, RwLock};

use napi::bindgen_prelude::*;
use napi_derive::napi;
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

/// Chat configuration for Qwen3.5
#[napi(object)]
#[derive(Debug, Clone)]
pub struct Qwen3_5ChatConfig {
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
}

/// Chat result
#[napi(object)]
#[derive(Debug, Clone)]
pub struct Qwen3_5ChatResult {
    pub text: String,
    pub tool_calls: Vec<ToolCallResult>,
    pub thinking: Option<String>,
    pub num_tokens: u32,
    pub finish_reason: String,
    pub raw_text: String,
}

/// Qwen3.5 Model -- hybrid linear/full attention with optional MoE.
///
/// Uses interior mutability (RwLock) for layers, final_norm, lm_head, and caches
/// to allow async generation via spawn_blocking without blocking the Node.js event loop.
/// This matches the pattern used by Qwen3Model.
#[napi]
pub struct Qwen3_5Model {
    config: Qwen3_5Config,
    pub(crate) embedding: Embedding,
    /// Decoder layers wrapped in RwLock for interior mutability during generation.
    pub(crate) layers: Arc<RwLock<Vec<DecoderLayer>>>,
    /// Final layer norm wrapped in RwLock for interior mutability.
    pub(crate) final_norm: Arc<RwLock<RMSNorm>>,
    /// LM head wrapped in RwLock for interior mutability.
    pub(crate) lm_head: Arc<RwLock<Option<Linear>>>, // None when tie_word_embeddings
    /// KV/SSM caches wrapped in RwLock for interior mutability during generation.
    caches: Arc<RwLock<Option<Vec<Qwen3_5LayerCache>>>>,
    pub(crate) tokenizer: Option<Arc<Qwen3Tokenizer>>,
    fa_idx: usize, // Index of first full attention layer
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

        napi::bindgen_prelude::spawn_blocking(move || {
            // Reset and init caches
            {
                let mut caches_guard = caches_arc
                    .write()
                    .map_err(|_| Error::from_reason("Failed to acquire caches write lock"))?;
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
            }

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

            // Prefill: forward pass on entire prompt
            let logits = {
                let _stream_ctx = StreamContext::new(generation_stream);
                forward_with_locks(
                    &prompt_tokens,
                    &embedding_weight,
                    &layers_arc,
                    &final_norm_arc,
                    &lm_head_arc,
                    &caches_arc,
                    fa_idx,
                    Some(&embedding_weight_t),
                )?
            };

            // Get last token logits: [1, vocab]
            let seq_len = logits.shape_at(1)?;
            let last_logits = logits.slice_axis(1, seq_len - 1, seq_len)?;
            let last_logits = last_logits.squeeze(Some(&[1]))?; // [1, vocab]

            // Sample first token (lazy — not evaluated yet)
            let mut y = sample(&last_logits, sampling_config)?;
            MxArray::async_eval_arrays(&[&y]);

            // Decide whether to use the compiled forward path.
            // The compiled path requires C++ weights to be loaded (only true for
            // safetensors-loaded models). Test models have no stored weights and
            // must fall back to the pure Rust forward_with_locks path.
            let use_compiled = unsafe { mlx_sys::mlx_qwen35_weight_count() } > 0;

            // Guard ensures mlx_qwen35_compiled_reset() is called even if `?` returns early.
            // Created before the decode loop so it outlives the entire loop + init block.
            let _compiled_guard = if use_compiled {
                Some(CompiledResetGuard)
            } else {
                None
            };

            if use_compiled {
                // Initialize compiled forward pass from prefill caches.
                // Imports post-prefill cache arrays into C++ global state and sets
                // up mlx::core::compile() to cache the graph across decode steps.
                // max_kv_len is rounded up to handle prompt + max_tokens.
                use mlx_sys as sys;
                let prefill_len = seq_len as i32;
                let max_kv_len = ((prefill_len + max_tokens + 255) / 256) * 256;
                let num_layers = model_config.num_layers as usize;
                let mut cache_ptrs: Vec<*mut sys::mlx_array> =
                    vec![std::ptr::null_mut(); num_layers * 2];
                let caches_guard = caches_arc
                    .read()
                    .map_err(|_| Error::from_reason("Failed to acquire caches read lock"))?;
                if let Some(ref caches) = *caches_guard {
                    for (i, cache) in caches.iter().enumerate() {
                        let (p0, p1) = cache.export_ptrs();
                        cache_ptrs[i * 2] = p0;
                        cache_ptrs[i * 2 + 1] = p1;
                    }
                }
                drop(caches_guard);
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
            }

            // Decode loop: matches Python mlx-lm's generate_step pattern.
            // Key: compute NEXT token's forward pass BEFORE extracting current token.
            // This overlaps GPU computation with CPU token extraction.
            // Uses compiled C++ forward pass for real models (graph cached after step 1).
            // Falls back to pure Rust forward_with_locks for test/unweighted models.
            for step in 0..max_tokens {
                // 1. Compute NEXT token (GPU work starts immediately)
                // Wrap entire step in generation stream (matches Python mlx-lm)
                let next_y = {
                    let _stream_ctx = StreamContext::new(generation_stream);
                    if step + 1 < max_tokens {
                        let next_ids = y.reshape(&[1, 1])?;
                        let next_token = if use_compiled {
                            let logits = forward_compiled(&next_ids, &embedding_weight)?;
                            // Compiled path returns 2D [B, vocab] — no squeeze needed.
                            let next_token = sample(&logits, sampling_config)?;
                            // Eval token + compiled caches to break lazy graph chains.
                            eval_token_and_compiled_caches(&next_token);
                            next_token
                        } else {
                            // Fallback: pure Rust path (test models, no C++ weights)
                            let logits = forward_with_locks(
                                &next_ids,
                                &embedding_weight,
                                &layers_arc,
                                &final_norm_arc,
                                &lm_head_arc,
                                &caches_arc,
                                fa_idx,
                                Some(&embedding_weight_t),
                            )?;
                            let logits = logits.squeeze(Some(&[1]))?;
                            let next_token = sample(&logits, sampling_config)?;
                            eval_token_and_caches(&next_token, &caches_arc);
                            next_token
                        };
                        Some(next_token)
                    } else {
                        None
                    }
                };

                // 2. Extract CURRENT token (GPU is already working on next)
                // eval() is required every step: read_scalar (used by item_at_int32)
                // accesses data<T>() which does not block on async_eval completion.
                y.eval();
                let token_id = y.item_at_int32(0)? as u32;
                generated_tokens.push(token_id);

                if token_id == eos_id {
                    finish_reason = String::from("eos");
                    break;
                }

                // 3. Advance to next token
                match next_y {
                    Some(next) => y = next,
                    None => break,
                }

                // Periodic cleanup to prevent computation graph memory accumulation
                if (step + 1) % 256 == 0 {
                    crate::array::synchronize_and_clear_cache();
                }
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
        config: Option<Qwen3_5ChatConfig>,
    ) -> Result<Qwen3_5ChatResult> {
        let config = config.unwrap_or(Qwen3_5ChatConfig {
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
        });

        // Tokenize messages using chat template
        let tokenizer = self
            .tokenizer
            .clone()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))?;

        // Clone Arcs and data needed for the closure
        let embedding_weight = self.embedding.get_weight();
        let layers_arc = self.layers.clone();
        let final_norm_arc = self.final_norm.clone();
        let lm_head_arc = self.lm_head.clone();
        let caches_arc = self.caches.clone();
        let model_config = self.config.clone();
        let fa_idx = self.fa_idx;
        let tokenizer_for_decode = tokenizer.clone();

        napi::bindgen_prelude::spawn_blocking(move || {
            let tool_defs = config.tools.as_deref();
            let tokens =
                tokenizer.apply_chat_template_sync(&messages, Some(true), tool_defs, None)?;

            // Create prompt tensor
            let prompt = MxArray::from_uint32(&tokens, &[1, tokens.len() as i64])?;

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

            // Reset and init caches
            {
                let mut caches_guard = caches_arc
                    .write()
                    .map_err(|_| Error::from_reason("Failed to acquire caches write lock"))?;
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
            }

            let eos_id = model_config.eos_token_id as u32;
            let mut generated_tokens: Vec<u32> = Vec::new();
            let mut finish_reason = String::from("length");

            // Track token history for repetition penalty
            let mut token_history: Vec<u32> = tokens.clone();

            // Pre-compute embedding weight transpose once (avoids recomputing per step)
            let embedding_weight_t = embedding_weight.transpose(Some(&[1, 0]))?;

            // Create dedicated generation stream for GPU scheduling
            let generation_stream = Stream::new(DeviceType::Gpu);

            // Pin model weights in Metal memory for the duration of generation
            let model_size_bytes = model_config.estimate_memory_bytes() as usize;
            let _wired_ctx =
                crate::stream::WiredLimitContext::new(model_size_bytes, vec![generation_stream]);

            // Prefill
            let logits = {
                let _stream_ctx = StreamContext::new(generation_stream);
                forward_with_locks(
                    &prompt,
                    &embedding_weight,
                    &layers_arc,
                    &final_norm_arc,
                    &lm_head_arc,
                    &caches_arc,
                    fa_idx,
                    Some(&embedding_weight_t),
                )?
            };

            let seq_len = logits.shape_at(1)?;
            let last_logits = logits.slice_axis(1, seq_len - 1, seq_len)?;
            let mut last_logits = last_logits.squeeze(Some(&[1]))?;

            // Apply repetition penalty to prefill logits
            if repetition_penalty != 1.0 && !token_history.is_empty() {
                last_logits = apply_repetition_penalty(
                    &last_logits,
                    &token_history,
                    repetition_penalty,
                    Some(repetition_context_size),
                )?;
            }

            // Sample first token (lazy — not evaluated yet)
            let mut y = sample(&last_logits, sampling_config)?;
            MxArray::async_eval_arrays(&[&y]);

            // Initialize compiled forward pass (same as generate() path).
            let use_compiled = unsafe { mlx_sys::mlx_qwen35_weight_count() } > 0;

            // Guard ensures mlx_qwen35_compiled_reset() is called even if `?` returns early.
            let _compiled_guard = if use_compiled {
                Some(CompiledResetGuard)
            } else {
                None
            };

            if use_compiled {
                use mlx_sys as sys;
                let prefill_len = seq_len as i32;
                let max_kv_len = ((prefill_len + max_new_tokens + 255) / 256) * 256;
                let num_layers = model_config.num_layers as usize;
                let mut cache_ptrs: Vec<*mut sys::mlx_array> =
                    vec![std::ptr::null_mut(); num_layers * 2];
                let caches_guard = caches_arc
                    .read()
                    .map_err(|_| Error::from_reason("Failed to acquire caches read lock"))?;
                if let Some(ref caches) = *caches_guard {
                    for (i, cache) in caches.iter().enumerate() {
                        let (p0, p1) = cache.export_ptrs();
                        cache_ptrs[i * 2] = p0;
                        cache_ptrs[i * 2 + 1] = p1;
                    }
                }
                drop(caches_guard);
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
            }

            // Decode loop: compiled C++ path for real models, Rust fallback for tests.
            for step in 0..max_new_tokens {
                // 1. Extract CURRENT token FIRST so repetition penalty includes it.
                // Without this, the penalty is one token behind (off-by-one).
                y.eval();
                let token_id = y.item_at_int32(0)? as u32;
                generated_tokens.push(token_id);
                token_history.push(token_id);

                if token_id == eos_id {
                    finish_reason = String::from("eos");
                    break;
                }

                // Check repetition cutoff
                if let Some(reason) = check_repetition_cutoff(
                    &generated_tokens,
                    max_consecutive_tokens,
                    max_ngram_repeats,
                    ngram_size,
                ) {
                    finish_reason = reason.to_string();
                    break;
                }

                // 2. Compute NEXT token (with complete token_history including current)
                if step + 1 >= max_new_tokens {
                    break;
                }
                {
                    let _stream_ctx = StreamContext::new(generation_stream);
                    let next_ids = y.reshape(&[1, 1])?;
                    y = if use_compiled {
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
                        next_token
                    } else {
                        let logits = forward_with_locks(
                            &next_ids,
                            &embedding_weight,
                            &layers_arc,
                            &final_norm_arc,
                            &lm_head_arc,
                            &caches_arc,
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
                        eval_token_and_caches(&next_token, &caches_arc);
                        next_token
                    };
                }

                // Periodic cleanup to prevent computation graph memory accumulation
                if (step + 1) % 256 == 0 {
                    crate::array::synchronize_and_clear_cache();
                }
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

            Ok(Qwen3_5ChatResult {
                text: clean_text,
                tool_calls,
                thinking,
                num_tokens,
                finish_reason,
                raw_text: text,
            })
        })
        .await
        .map_err(|e| Error::from_reason(format!("Chat task failed: {}", e)))?
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
fn forward_with_locks(
    input_ids: &MxArray,
    embedding_weight: &MxArray,
    layers_arc: &Arc<RwLock<Vec<DecoderLayer>>>,
    final_norm_arc: &Arc<RwLock<RMSNorm>>,
    lm_head_arc: &Arc<RwLock<Option<Linear>>>,
    caches_arc: &Arc<RwLock<Option<Vec<Qwen3_5LayerCache>>>>,
    fa_idx: usize,
    embedding_weight_t: Option<&MxArray>,
) -> Result<MxArray> {
    // Compute embeddings (embedding is immutable, no lock needed)
    let embedding = Embedding::from_weight(embedding_weight)?;
    let hidden_states = embedding.forward(input_ids)?;

    let mut h = hidden_states.clone();

    // Acquire write locks for layers and caches (forward mutates caches)
    let mut layers_guard = layers_arc
        .write()
        .map_err(|_| Error::from_reason("Failed to acquire layers write lock"))?;
    let mut caches_guard = caches_arc
        .write()
        .map_err(|_| Error::from_reason("Failed to acquire caches write lock"))?;

    // Create masks
    let seq_len = hidden_states.shape_at(1)?;
    let fa_mask = {
        let has_cache = caches_guard.is_some();
        if seq_len <= 1 && has_cache {
            None
        } else {
            let offset = caches_guard
                .as_ref()
                .map(|c| c[fa_idx].offset())
                .unwrap_or(0);
            Some(create_causal_mask(seq_len as i32, Some(offset), None)?)
        }
    };

    // SSM mask is always None — mlx-vlm never creates one for ArraysCache.
    // An all-ones mask is a no-op that adds unnecessary graph nodes and Metal overhead.

    // Forward through layers
    let num_layers = layers_guard.len();
    for i in 0..num_layers {
        let mask = if layers_guard[i].is_linear() {
            None
        } else {
            fa_mask.as_ref()
        };

        let cache = caches_guard.as_mut().map(|c| &mut c[i]);
        h = layers_guard[i].forward(&h, mask, cache)?;
    }

    // Drop layers lock early -- no longer needed
    drop(layers_guard);

    // Final norm
    let final_norm_guard = final_norm_arc
        .read()
        .map_err(|_| Error::from_reason("Failed to acquire final_norm read lock"))?;
    let h = final_norm_guard.forward(&h)?;
    drop(final_norm_guard);

    // LM head
    let lm_head_guard = lm_head_arc
        .read()
        .map_err(|_| Error::from_reason("Failed to acquire lm_head read lock"))?;
    match &*lm_head_guard {
        Some(head) => head.forward(&h),
        None => {
            // tie_word_embeddings: use pre-transposed weight or compute on the fly
            match embedding_weight_t {
                Some(wt) => h.matmul(wt),
                None => {
                    let wt = embedding_weight.transpose(Some(&[1, 0]))?;
                    h.matmul(&wt)
                }
            }
        }
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
fn eval_token_and_caches(
    next_token: &MxArray,
    caches_arc: &Arc<RwLock<Option<Vec<Qwen3_5LayerCache>>>>,
) {
    // Collect all cache array handles while holding read lock.
    // We build a Vec<*mut mlx_array> directly to avoid lifetime issues.
    let mut handles: Vec<*mut mlx_sys::mlx_array> = vec![next_token.as_raw_ptr()];

    if let Ok(caches_guard) = caches_arc.read()
        && let Some(ref caches) = *caches_guard
    {
        let mut arr_refs: Vec<&MxArray> = Vec::with_capacity(caches.len() * 2);
        for cache in caches.iter() {
            cache.collect_arrays(&mut arr_refs);
        }
        for arr in &arr_refs {
            handles.push(arr.as_raw_ptr());
        }
    }

    // Single async_eval call for token + all cache arrays.
    // MLX will evaluate and detach all of them, preventing graph accumulation.
    unsafe {
        mlx_sys::mlx_async_eval(handles.as_mut_ptr(), handles.len());
    }
}

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
            h = layers_guard[i].forward(&h, mask, cache)?;
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
