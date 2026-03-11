use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};

use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi_derive::napi;
use tokio::sync::Mutex as TokioMutex;
use tracing::{info, warn};

use crate::models::qwen3_5::model::{ChatStreamChunk, ChatStreamHandle};

use super::quantized_linear::LinearProj;
use crate::array::MxArray;
use crate::array::mask::create_causal_mask;
use crate::nn::{Embedding, Linear, RMSNorm};
use crate::sampling::{SamplingConfig, apply_repetition_penalty, check_repetition_cutoff, sample};
use crate::stream::{DeviceType, Stream, StreamContext};
use crate::tokenizer::{ChatMessage, Qwen3Tokenizer, ToolDefinition};
use crate::tools;
use crate::tools::ToolCallResult;

use super::config::Qwen3_5MoeConfig;
use super::decoder_layer::DecoderLayer;
use super::layer_cache::Qwen3_5LayerCache;
use super::persistence;

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

/// Chat configuration for Qwen3.5 MoE
#[napi(object)]
#[derive(Debug, Clone)]
pub struct Qwen3_5MoeChatConfig {
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
pub struct Qwen3_5MoeChatResult {
    pub text: String,
    pub tool_calls: Vec<ToolCallResult>,
    pub thinking: Option<String>,
    pub num_tokens: u32,
    pub finish_reason: String,
    pub raw_text: String,
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

            // Prefill
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
                for step in 0..max_tokens {
                    let next_y = if step + 1 < max_tokens {
                        let next_ids = y.reshape(&[1, 1])?;
                        let logits = forward_moe_cpp(&next_ids, &embedding_weight)?;
                        let next_token = sample(&logits, sampling_config)?;
                        eval_token_and_moe_caches(&next_token);
                        Some(next_token)
                    } else {
                        None
                    };

                    y.eval();
                    let token_id = y.item_at_int32(0)? as u32;
                    generated_tokens.push(token_id);

                    if token_id == eos_id {
                        finish_reason = String::from("eos");
                        break;
                    }

                    match next_y {
                        Some(next) => y = next,
                        None => break,
                    }

                    if (step + 1) % 256 == 0 {
                        crate::array::clear_cache();
                    }
                }
                // _moe_guard dropped here, calling mlx_qwen35_moe_reset()
            } else {
                // Rust fallback decode loop (outer StreamContext already active)
                for step in 0..max_tokens {
                    let next_y = if step + 1 < max_tokens {
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
                        let next_token = sample(&logits, sampling_config)?;
                        eval_token_and_caches_inner(&next_token, &caches_guard);
                        Some(next_token)
                    } else {
                        None
                    };

                    y.eval();
                    let token_id = y.item_at_int32(0)? as u32;
                    generated_tokens.push(token_id);

                    if token_id == eos_id {
                        finish_reason = String::from("eos");
                        break;
                    }

                    match next_y {
                        Some(next) => y = next,
                        None => break,
                    }

                    if (step + 1) % 256 == 0 {
                        crate::array::clear_cache();
                    }
                }

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
        config: Option<Qwen3_5MoeChatConfig>,
    ) -> Result<Qwen3_5MoeChatResult> {
        let config = config.unwrap_or(Qwen3_5MoeChatConfig {
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

        let tokenizer = self
            .tokenizer
            .clone()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))?;

        let embedding_weight = self.embedding.get_weight();
        let layers_arc = self.layers.clone();
        let final_norm_arc = self.final_norm.clone();
        let lm_head_arc = self.lm_head.clone();
        let caches_arc = self.caches.clone();
        let model_config = self.config.clone();
        let fa_idx = self.fa_idx;
        let tokenizer_for_decode = tokenizer.clone();

        // Check if C++ MoE path will be used (weights loaded from safetensors).
        let use_cpp = unsafe { mlx_sys::mlx_qwen35_weight_count() } > 0;

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

            // Prefill
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
                for step in 0..max_new_tokens {
                    // Extract CURRENT token FIRST so repetition penalty includes it
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

                    // Compute NEXT token (with complete token_history including current)
                    if step + 1 < max_new_tokens {
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
                    } else {
                        break;
                    }

                    if (step + 1) % 256 == 0 {
                        crate::array::clear_cache();
                    }
                }
                // _moe_guard dropped here, calling mlx_qwen35_moe_reset()
            } else {
                // Rust fallback decode loop
                for step in 0..max_new_tokens {
                    // Extract CURRENT token FIRST so repetition penalty includes it
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

                    // Compute NEXT token (with complete token_history including current)
                    if step + 1 < max_new_tokens {
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
                        eval_token_and_caches_inner(&next_token, &caches_guard);
                        y = next_token;
                    } else {
                        break;
                    }

                    if (step + 1) % 256 == 0 {
                        crate::array::clear_cache();
                    }
                }

                drop(layers_guard);
                drop(caches_guard);
                drop(final_norm_guard);
                drop(lm_head_guard);
            }

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

            Ok(Qwen3_5MoeChatResult {
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

    /// Streaming chat API with tool calling support.
    ///
    /// Same as `chat()` but streams tokens one-by-one via the callback.
    /// Returns a `ChatStreamHandle` immediately; generation runs in background.
    /// Call `handle.cancel()` to abort generation early.
    #[napi(
        ts_args_type = "messages: ChatMessage[], config: Qwen35MoeChatConfig | null, callback: (err: Error | null, chunk: ChatStreamChunk) => void"
    )]
    pub async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        config: Option<Qwen3_5MoeChatConfig>,
        callback: ThreadsafeFunction<ChatStreamChunk, ()>,
    ) -> Result<ChatStreamHandle> {
        let config = config.unwrap_or(Qwen3_5MoeChatConfig {
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

        let tokenizer = self
            .tokenizer
            .clone()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))?;

        let embedding_weight = self.embedding.get_weight();
        let layers_arc = self.layers.clone();
        let final_norm_arc = self.final_norm.clone();
        let lm_head_arc = self.lm_head.clone();
        let caches_arc = self.caches.clone();
        let model_config = self.config.clone();
        let fa_idx = self.fa_idx;
        let tokenizer_for_decode = tokenizer.clone();

        // Check if C++ MoE path will be used (weights loaded from safetensors).
        let use_cpp = unsafe { mlx_sys::mlx_qwen35_weight_count() } > 0;

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

                    let prompt = MxArray::from_uint32(&tokens, &[1, tokens.len() as i64])?;

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

                    // Prefill
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
                        for step in 0..max_new_tokens {
                            y.eval();
                            let token_id = y.item_at_int32(0)? as u32;
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

                            if (step + 1) % 256 == 0 {
                                crate::array::clear_cache();
                            }
                        }
                        // _moe_guard dropped here, calling mlx_qwen35_moe_reset()
                    } else {
                        // Rust fallback decode loop (locks held for entire loop)
                        for step in 0..max_new_tokens {
                            y.eval();
                            let token_id = y.item_at_int32(0)? as u32;
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
                            eval_token_and_caches_inner(&next_token, &caches_guard);
                            y = next_token;

                            if (step + 1) % 256 == 0 {
                                crate::array::clear_cache();
                            }
                        }

                        drop(layers_guard);
                        drop(caches_guard);
                        drop(final_norm_guard);
                        drop(lm_head_guard);
                    }

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
        h = layers[i].forward(&h, mask, cache)?;
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

/// Evaluate the sampled token AND all cache arrays together (lock-free version).
///
/// Takes caches directly instead of acquiring a read lock.
fn eval_token_and_caches_inner(next_token: &MxArray, caches: &Option<Vec<Qwen3_5LayerCache>>) {
    let mut handles: Vec<*mut mlx_sys::mlx_array> = vec![next_token.as_raw_ptr()];

    if let Some(ref caches) = *caches {
        let mut arr_refs: Vec<&MxArray> = Vec::with_capacity(caches.len() * 2);
        for cache in caches.iter() {
            cache.collect_arrays(&mut arr_refs);
        }
        for arr in &arr_refs {
            handles.push(arr.as_raw_ptr());
        }
    }

    unsafe {
        mlx_sys::mlx_async_eval(handles.as_mut_ptr(), handles.len());
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

impl Qwen3_5MoeModel {
    fn forward_from_embeddings(&self, hidden_states: &MxArray) -> Result<MxArray> {
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
            h = layers_guard[i].forward(&h, mask, cache)?;
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

    fn create_ssm_mask(&self, _hidden_states: &MxArray) -> Result<Option<MxArray>> {
        // SSM mask is always None — mlx-vlm never creates one for ArraysCache.
        // An all-ones mask is a no-op that adds unnecessary graph nodes and Metal overhead.
        Ok(None)
    }
}
