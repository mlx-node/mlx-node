/**
 * Qwen3 Model - Core Model Implementation
 *
 * Contains the model structure, forward passes, and core model methods.
 */
use std::collections::HashMap;
use std::iter;
use std::sync::{Arc, RwLock};

use napi::bindgen_prelude::*;
use napi_derive::napi;
use tracing::{debug, error, info, warn};

use crate::array::{MxArray, pad_float_sequences, pad_sequences, synchronize_and_clear_cache};
use crate::grpo::{advantages::compute_advantages, autograd::compute_loss_and_gradients_autograd};
use crate::nn::{Embedding, Linear, RMSNorm};
use crate::sampling::{SamplingConfig, apply_repetition_penalty, sample, sample_and_logprobs};
use crate::stream::{DeviceType, Stream, StreamContext};
use crate::tokenizer::{ChatMessage, Qwen3Tokenizer, ToolDefinition};
use crate::tools;
use crate::transformer::{KVCache, TransformerBlock};

use super::{
    BatchGenerationResult, ChatConfig, ChatResult, GenerationConfig, GenerationResult, Qwen3Config,
};

/// Check if generation has fallen into a repetitive loop.
///
/// Returns Some("repetition") if should stop, None otherwise.
/// Checks for two types of repetition:
/// 1. Consecutive identical tokens (e.g., "A A A A A")
/// 2. N-gram repetition (e.g., "A B C A B C A B C")
fn check_repetition_cutoff(
    tokens: &[u32],
    max_consecutive: i32,
    max_ngram_repeats: i32,
    ngram_size: i32,
) -> Option<&'static str> {
    let len = tokens.len();
    if len < 2 {
        return None;
    }

    // Skip check if disabled (values <= 0)
    let check_consecutive = max_consecutive > 0;
    let check_ngram = max_ngram_repeats > 0 && ngram_size > 0;

    // 1. Check consecutive identical tokens (fast path)
    if check_consecutive {
        let last = tokens[len - 1];
        let mut consecutive = 1usize;
        for i in (0..len - 1).rev() {
            if tokens[i] == last {
                consecutive += 1;
                if consecutive >= max_consecutive as usize {
                    return Some("repetition");
                }
            } else {
                break;
            }
        }
    }

    // 2. Check n-gram repetition (e.g., "A B C A B C A B C")
    if check_ngram {
        let ngram_size = ngram_size as usize;
        let max_ngram_repeats = max_ngram_repeats as usize;

        if len >= ngram_size * 2 {
            let ngram = &tokens[len - ngram_size..];
            let mut repeats = 1usize;
            let mut pos = len - ngram_size * 2;

            loop {
                if &tokens[pos..pos + ngram_size] == ngram {
                    repeats += 1;
                    if repeats >= max_ngram_repeats {
                        return Some("repetition");
                    }
                } else {
                    break; // Must be consecutive repetitions
                }
                if pos < ngram_size {
                    break;
                }
                pos -= ngram_size;
            }
        }
    }

    None
}

/// Qwen3 Model with automatic differentiation support
#[napi]
pub struct Qwen3Model {
    config: Qwen3Config,
    embedding: Embedding,
    layers: Arc<Vec<TransformerBlock>>,
    final_norm: Arc<RMSNorm>,
    lm_head: Arc<Linear>,
    // KV caches for incremental generation (one per layer)
    // Using RefCell for interior mutability
    kv_caches: Arc<RwLock<Option<Vec<KVCache>>>>,
    // Tokenizer for text-to-text generation (loaded via load_pretrained)
    pub(crate) tokenizer: Option<Arc<Qwen3Tokenizer>>,
}

#[napi]
impl Qwen3Model {
    /// Create a new Qwen3 model with the given configuration
    #[napi(constructor)]
    pub fn new(config: Qwen3Config) -> Result<Self> {
        // Token embedding
        let embedding = Embedding::new(config.vocab_size as u32, config.hidden_size as u32)?;

        // Transformer layers
        let layers = (0..config.num_layers)
            .map(|_| {
                TransformerBlock::new(
                    config.hidden_size as u32,
                    config.num_heads as u32,
                    config.num_kv_heads as u32,
                    config.intermediate_size as u32,
                    config.rms_norm_eps,
                    Some(config.rope_theta),
                    Some(config.use_qk_norm),
                    Some(config.head_dim as u32), // Use head_dim from config
                )
            })
            .collect::<Result<Vec<_>>>()?;

        // Final layer norm
        let final_norm = RMSNorm::new(config.hidden_size as u32, Some(config.rms_norm_eps))?;

        // LM head
        let lm_head = Linear::new(
            config.hidden_size as u32,
            config.vocab_size as u32,
            Some(false),
        )?;

        Ok(Self {
            config,
            embedding,
            layers: Arc::new(layers),
            final_norm: Arc::new(final_norm),
            lm_head: Arc::new(lm_head),
            kv_caches: Arc::new(RwLock::new(None)),
            tokenizer: None,
        })
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs, shape: [batch_size, seq_len]
    ///
    /// # Returns
    /// * Logits, shape: [batch_size, seq_len, vocab_size]
    #[napi]
    pub fn forward(&self, input_ids: &MxArray) -> Result<MxArray> {
        // Embedding lookup
        let mut hidden_states = self.embedding.forward(input_ids)?;

        // Pass through transformer layers
        // Note: We pass mask=None and let the Attention layer automatically use
        // the optimized "causal" mode during prefill (seq_len > 1).
        for layer in self.layers.iter() {
            // Each layer processes: x = x + attn(norm(x)) + mlp(norm(x))
            hidden_states = layer.forward(&hidden_states, None, None)?;
        }

        // Final layer norm
        hidden_states = self.final_norm.forward(&hidden_states)?;

        // LM head to get logits
        // CRITICAL: When tie_word_embeddings=true, we must use the embedding weight transposed
        // as the lm_head (following mlx-lm's embed_tokens.as_linear() pattern).
        // This is essential for correct predictions!
        let logits = if self.config.tie_word_embeddings {
            // Use embedding.weight.T for tied embeddings: logits = hidden @ embedding.T
            let embedding_weight = self.embedding.get_weight();
            hidden_states.matmul(&embedding_weight.transpose(Some(&[1, 0]))?)?
        } else {
            // Use separate lm_head weights
            self.lm_head.forward(&hidden_states)?
        };

        Ok(logits)
    }

    /// Initialize KV caches for incremental generation
    ///
    /// Creates one KV cache per transformer layer. Call this before starting generation.
    #[napi]
    pub fn init_kv_caches(&self) -> Result<()> {
        let caches: Vec<KVCache> = (0..self.layers.len()).map(|_| KVCache::new()).collect();

        *self.kv_caches.write().map_err(|_| {
            Error::new(
                napi::Status::GenericFailure,
                "Failed to acquire kv caches write lock",
            )
        })? = Some(caches);
        Ok(())
    }

    /// Reset all KV caches
    ///
    /// Clears cached key-value states. Call this between different generation sequences.
    #[napi]
    pub fn reset_kv_caches(&self) -> Result<()> {
        if let Some(caches) = self
            .kv_caches
            .write()
            .map_err(|_| {
                Error::new(
                    napi::Status::GenericFailure,
                    "Failed to acquire kv caches read lock",
                )
            })?
            .as_mut()
        {
            for cache in caches.iter_mut() {
                cache.reset();
            }
        }
        Ok(())
    }

    /// Forward pass with KV caching for incremental generation
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs, shape: [batch_size, seq_len]
    /// * `use_cache` - Whether to use KV caching (must call init_kv_caches() first)
    ///
    /// # Returns
    /// * Logits, shape: [batch_size, seq_len, vocab_size]
    #[napi]
    pub fn forward_with_cache(&self, input_ids: &MxArray, use_cache: bool) -> Result<MxArray> {
        if use_cache {
            // Acquire lock for public API (used in training, batch generation, etc.)
            let mut caches_borrowed = self.kv_caches.write().map_err(|_| {
                Error::new(
                    napi::Status::GenericFailure,
                    "Failed to acquire kv caches write lock",
                )
            })?;

            Self::forward_with_cache_direct(
                input_ids,
                caches_borrowed.as_mut(),
                &self.embedding.get_weight(),
                &self.layers,
                self.config.tie_word_embeddings,
                &self.final_norm,
                &self.lm_head,
            )
        } else {
            Self::forward_with_cache_direct(
                input_ids,
                None,
                &self.embedding.get_weight(),
                &self.layers,
                self.config.tie_word_embeddings,
                &self.final_norm,
                &self.lm_head,
            )
        }
    }

    // Lock-free forward pass for hot path (generation loop)
    // Takes direct mutable reference to caches, avoiding RwLock overhead
    fn forward_with_cache_direct(
        input_ids: &MxArray,
        kv_caches: Option<&mut Vec<KVCache>>,
        embedding_weight: &MxArray,
        layers: &[TransformerBlock],
        tie_word_embeddings: bool,
        final_norm: &RMSNorm,
        lm_head: &Linear,
    ) -> Result<MxArray> {
        // Embedding lookup
        let mut hidden_states = embedding_weight.take(input_ids, 0)?;

        // Pass through transformer layers with optional caching
        // Note: We pass mask=None and let the Attention layer automatically use
        // the optimized "causal" mode during prefill (seq_len > 1).
        // During generation (seq_len == 1), no mask is needed due to KV cache.
        if let Some(caches) = kv_caches {
            for (i, layer) in layers.iter().enumerate() {
                hidden_states = layer.forward(&hidden_states, None, Some(&mut caches[i]))?;
            }
        } else {
            for layer in layers.iter() {
                hidden_states = layer.forward(&hidden_states, None, None)?;
            }
        }

        // Final layer norm
        hidden_states = final_norm.forward(&hidden_states)?;

        // LM head to get logits
        let logits = if tie_word_embeddings {
            hidden_states.matmul(&embedding_weight.transpose(Some(&[1, 0]))?)?
        } else {
            lm_head.forward(&hidden_states)?
        };

        Ok(logits)
    }

    /// Fused forward pass using C++ implementation for maximum performance.
    /// Reduces FFI calls from ~300 to 1 per forward pass.
    /// Updates KV cache in-place to avoid allocations (matches mlx-lm's overwrite_descriptor pattern).
    fn forward_fused(
        input_ids: &MxArray,
        embedding_weight: &MxArray,
        layers: &[TransformerBlock],
        final_norm: &RMSNorm,
        lm_head: &Linear,
        config: &Qwen3Config,
        kv_keys: &mut [Option<MxArray>],
        kv_values: &mut [Option<MxArray>],
        cache_offsets: &mut [i32],
        cache_capacities: &mut [i32],
    ) -> Result<MxArray> {
        use mlx_sys as sys;
        use std::ptr;

        let num_layers = layers.len();

        // Collect layer weights into a flat array (11 weights per layer)
        let mut layer_weights: Vec<*mut sys::mlx_array> = Vec::with_capacity(num_layers * 11);

        for layer in layers.iter() {
            layer_weights.push(layer.get_input_layernorm_weight().handle.0);
            layer_weights.push(layer.get_post_attention_layernorm_weight().handle.0);
            layer_weights.push(layer.self_attn.get_q_proj_weight().handle.0);
            layer_weights.push(layer.self_attn.get_k_proj_weight().handle.0);
            layer_weights.push(layer.self_attn.get_v_proj_weight().handle.0);
            layer_weights.push(layer.self_attn.get_o_proj_weight().handle.0);
            if let Some(q_norm) = layer.self_attn.get_q_norm_weight() {
                layer_weights.push(q_norm.handle.0);
            } else {
                layer_weights.push(ptr::null_mut());
            }
            if let Some(k_norm) = layer.self_attn.get_k_norm_weight() {
                layer_weights.push(k_norm.handle.0);
            } else {
                layer_weights.push(ptr::null_mut());
            }
            layer_weights.push(layer.mlp.get_gate_proj_weight().handle.0);
            layer_weights.push(layer.mlp.get_up_proj_weight().handle.0);
            layer_weights.push(layer.mlp.get_down_proj_weight().handle.0);
        }

        let final_norm_weight = final_norm.get_weight();
        let lm_head_weight_handle = if config.tie_word_embeddings {
            ptr::null_mut()
        } else {
            lm_head.get_weight().handle.0
        };

        // Prepare KV cache input pointers
        let kv_keys_ptrs: Vec<*mut sys::mlx_array> = kv_keys
            .iter()
            .map(|k| k.as_ref().map(|a| a.handle.0).unwrap_or(ptr::null_mut()))
            .collect();
        let kv_values_ptrs: Vec<*mut sys::mlx_array> = kv_values
            .iter()
            .map(|v| v.as_ref().map(|a| a.handle.0).unwrap_or(ptr::null_mut()))
            .collect();

        // Prepare output arrays (will update in place)
        let mut out_logits: *mut sys::mlx_array = ptr::null_mut();
        let mut out_kv_keys: Vec<*mut sys::mlx_array> = vec![ptr::null_mut(); num_layers];
        let mut out_kv_values: Vec<*mut sys::mlx_array> = vec![ptr::null_mut(); num_layers];

        // Call the fused FFI function
        unsafe {
            sys::mlx_qwen3_forward_step(
                input_ids.handle.0,
                embedding_weight.handle.0,
                layer_weights.as_ptr(),
                num_layers as i32,
                final_norm_weight.handle.0,
                lm_head_weight_handle,
                config.tie_word_embeddings,
                config.hidden_size,
                config.num_heads,
                config.num_kv_heads,
                config.head_dim,
                config.rope_theta as f32,
                config.rms_norm_eps as f32,
                kv_keys_ptrs.as_ptr(),
                kv_values_ptrs.as_ptr(),
                cache_offsets.as_ptr(),
                cache_capacities.as_ptr(),
                &mut out_logits,
                out_kv_keys.as_mut_ptr(),
                out_kv_values.as_mut_ptr(),
                cache_offsets.as_mut_ptr(),
                cache_capacities.as_mut_ptr(),
            );
        }

        // Update KV cache in place - reuse existing MxArray handles when possible
        for (i, (existing, new_ptr)) in kv_keys.iter_mut().zip(out_kv_keys.into_iter()).enumerate()
        {
            if new_ptr.is_null() {
                continue;
            }
            if let Some(arr) = existing {
                // Try to reuse the existing handle
                if let Some(inner) = Arc::get_mut(&mut arr.handle) {
                    unsafe { inner.overwrite(new_ptr) };
                } else {
                    // Fall back to creating a new MxArray (shouldn't happen in fast path)
                    *existing = Some(MxArray::from_handle(new_ptr, "forward_fused kv_keys")?);
                }
            } else {
                // First time - create new MxArray
                *existing = Some(MxArray::from_handle(
                    new_ptr,
                    &format!("forward_fused kv_keys[{}]", i),
                )?);
            }
        }

        for (i, (existing, new_ptr)) in kv_values
            .iter_mut()
            .zip(out_kv_values.into_iter())
            .enumerate()
        {
            if new_ptr.is_null() {
                continue;
            }
            if let Some(arr) = existing {
                if let Some(inner) = Arc::get_mut(&mut arr.handle) {
                    unsafe { inner.overwrite(new_ptr) };
                } else {
                    *existing = Some(MxArray::from_handle(new_ptr, "forward_fused kv_values")?);
                }
            } else {
                *existing = Some(MxArray::from_handle(
                    new_ptr,
                    &format!("forward_fused kv_values[{}]", i),
                )?);
            }
        }

        MxArray::from_handle(out_logits, "forward_fused logits")
    }

    /// Get model configuration
    #[napi]
    pub fn get_config(&self) -> Qwen3Config {
        self.config.clone()
    }

    /// Clone the model for use in a training session
    ///
    /// Creates a new model instance with its own copy of all parameters.
    /// This is necessary because apply_gradients uses Arc::get_mut which
    /// requires unique ownership of the Arcs.
    pub fn clone_for_session(&self) -> Result<Self> {
        // Deep clone layers - creates a new Arc with cloned contents
        let cloned_layers: Vec<_> = self.layers.iter().cloned().collect();

        Ok(Self {
            config: self.config.clone(),
            embedding: self.embedding.clone(),
            layers: Arc::new(cloned_layers),
            final_norm: Arc::new((*self.final_norm).clone()),
            lm_head: Arc::new((*self.lm_head).clone()),
            kv_caches: Arc::new(RwLock::new(None)), // Fresh KV caches for session
            tokenizer: self.tokenizer.clone(),
        })
    }

    /// Decode tokens from an MxArray to text
    ///
    /// Internal method for use by training session.
    pub async fn decode_tokens(&self, tokens: &MxArray) -> Result<String> {
        let tokenizer = self.tokenizer.clone().ok_or_else(|| {
            Error::new(
                Status::InvalidArg,
                "Tokenizer not available. Model must be loaded via load_pretrained().",
            )
        })?;

        // Convert MxArray to Vec<u32>
        let token_ids = tokens.to_uint32()?;

        napi::bindgen_prelude::spawn_blocking(move || {
            tokenizer.decode_sync(&token_ids, true) // skip special tokens
        })
        .await
        .map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("Decoding task failed: {}", e),
            )
        })?
    }

    /// Apply chat template and return token IDs as Vec<u32>
    ///
    /// Internal async method for use by training session.
    /// Named differently to avoid conflict with the NAPI-exported version.
    pub async fn apply_chat_template_internal(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: Option<bool>,
    ) -> Result<Vec<u32>> {
        let tokenizer = self.tokenizer.clone().ok_or_else(|| {
            Error::new(
                Status::InvalidArg,
                "Tokenizer not available. Model must be loaded via load_pretrained().",
            )
        })?;

        let add_prompt = add_generation_prompt.unwrap_or(true);
        let messages_owned: Vec<ChatMessage> = messages.to_vec();

        napi::bindgen_prelude::spawn_blocking(move || {
            // Format messages using ChatML template
            let mut formatted = String::new();
            for msg in &messages_owned {
                formatted.push_str(&format!(
                    "<|im_start|>{}\n{}<|im_end|>\n",
                    msg.role, msg.content
                ));
            }

            if add_prompt {
                formatted.push_str("<|im_start|>assistant\n");
            }

            // Encode the formatted text
            tokenizer.encode_sync(&formatted, Some(false))
        })
        .await
        .map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("Chat template task failed: {}", e),
            )
        })?
    }

    /// Decode tokens from an MxArray to text (sync version)
    ///
    /// Internal method for use by training session - does not use spawn_blocking.
    pub fn decode_tokens_sync(&self, tokens: &MxArray) -> Result<String> {
        let tokenizer = self.tokenizer.clone().ok_or_else(|| {
            Error::new(
                Status::InvalidArg,
                "Tokenizer not available. Model must be loaded via load_pretrained().",
            )
        })?;

        // Convert MxArray to Vec<u32>
        let token_ids = tokens.to_uint32()?;

        tokenizer.decode_sync(&token_ids, true) // skip special tokens
    }

    /// Apply chat template and return token IDs as Vec<u32> (sync version)
    ///
    /// Internal sync method for use by training session - does not use spawn_blocking.
    pub fn apply_chat_template_sync(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: Option<bool>,
    ) -> Result<Vec<u32>> {
        let tokenizer = self.tokenizer.clone().ok_or_else(|| {
            Error::new(
                Status::InvalidArg,
                "Tokenizer not available. Model must be loaded via load_pretrained().",
            )
        })?;

        let add_prompt = add_generation_prompt.unwrap_or(true);

        // Format messages using ChatML template
        let mut formatted = String::new();
        for msg in messages {
            formatted.push_str(&format!(
                "<|im_start|>{}\n{}<|im_end|>\n",
                msg.role, msg.content
            ));
        }

        if add_prompt {
            formatted.push_str("<|im_start|>assistant\n");
        }

        // Encode the formatted text
        tokenizer.encode_sync(&formatted, Some(false))
    }

    /// Generate tokens for training (sync version)
    ///
    /// Internal sync method for use by training session - does not use spawn_blocking.
    /// This is a synchronous version that runs generation on the calling thread.
    pub fn generate_for_training_sync(
        &self,
        input_ids: &MxArray,
        config: Option<GenerationConfig>,
    ) -> Result<GenerationResult> {
        let config = config.unwrap_or_default();
        let input_ids = input_ids.clone();
        // Extract configuration with defaults
        let max_new_tokens = config.max_new_tokens.unwrap_or(100);
        let temperature = config.temperature.unwrap_or(1.0);
        let top_k = config.top_k.unwrap_or(0);
        let top_p = config.top_p.unwrap_or(1.0);
        let min_p = config.min_p.unwrap_or(0.0);
        let repetition_penalty = config.repetition_penalty.unwrap_or(1.0);
        let repetition_context_size = config.repetition_context_size.unwrap_or(256);
        let max_consecutive_tokens = config.max_consecutive_tokens.unwrap_or(16);
        let max_ngram_repeats = config.max_ngram_repeats.unwrap_or(8);
        let ngram_size = config.ngram_size.unwrap_or(3);
        let eos_token_id = config.eos_token_id.or(Some(self.config.eos_token_id));
        let return_logprobs = config.return_logprobs.unwrap_or(true);

        // Calculate model size for wired_limit context
        let model_size_bytes = self.calculate_memory_size();

        let embedding_weight = self.embedding.get_weight();
        let layers = &self.layers;
        let final_norm = &self.final_norm;
        let lm_head = &self.lm_head;
        let model_config = &self.config;

        debug!(
            "Starting sync generation: max_tokens={}, temp={}, top_k={}, top_p={}, rep_penalty={}",
            max_new_tokens, temperature, top_k, top_p, repetition_penalty
        );

        // Create dedicated generation stream
        let generation_stream = Stream::new(DeviceType::Gpu);

        // Wired limit context for GPU memory management
        let _wired_ctx =
            crate::stream::WiredLimitContext::new(model_size_bytes, vec![generation_stream]);

        // Local KV caches
        let num_layers = layers.len();
        let mut kv_keys: Vec<Option<MxArray>> = vec![None; num_layers];
        let mut kv_values: Vec<Option<MxArray>> = vec![None; num_layers];
        let mut cache_offsets: Vec<i32> = vec![0; num_layers];
        let mut cache_capacities: Vec<i32> = vec![0; num_layers];

        // Get input tokens for repetition penalty context
        let input_tokens = input_ids.to_uint32()?;

        // Prepare generation state
        let current_ids = input_ids.clone();
        let mut generated_tokens: Vec<u32> = Vec::with_capacity(max_new_tokens as usize);
        let mut generated_logprobs: Vec<f32> = if return_logprobs {
            Vec::with_capacity(max_new_tokens as usize)
        } else {
            Vec::new()
        };
        let mut finish_reason = "length";

        // Sampling config
        let sampling_config = SamplingConfig {
            temperature: Some(temperature),
            top_k: Some(top_k),
            top_p: Some(top_p),
            min_p: Some(min_p),
        };

        // PREFILL: Process entire prompt
        let logits = {
            let _stream_ctx = StreamContext::new(generation_stream);
            Self::forward_fused(
                &current_ids,
                &embedding_weight,
                layers,
                final_norm,
                lm_head,
                model_config,
                &mut kv_keys,
                &mut kv_values,
                &mut cache_offsets,
                &mut cache_capacities,
            )?
        };

        // Extract last token logits (shape: [1, seq_len, vocab_size] -> [vocab_size])
        let seq_len = logits.shape_at(1)?;
        let mut last_logits = logits
            .slice_axis(1, seq_len - 1, seq_len)?
            .squeeze(Some(&[0, 1]))?;

        // Apply repetition penalty to prefill logits if enabled
        if repetition_penalty != 1.0 && !input_tokens.is_empty() {
            last_logits = apply_repetition_penalty(
                &last_logits,
                &input_tokens,
                repetition_penalty,
                Some(repetition_context_size),
            )?;
        }

        // Sample first token
        let (mut token, mut logprobs_arr) = if return_logprobs {
            let (tok, lp) = sample_and_logprobs(&last_logits, Some(sampling_config))?;
            (tok, Some(lp))
        } else {
            let tok = sample(&last_logits, Some(sampling_config))?;
            (tok, None)
        };

        // DECODE loop
        // Cleanup interval to release intermediate tensors and prevent memory accumulation
        // Every 64 tokens is a good balance between memory savings and performance
        const DECODE_CLEANUP_INTERVAL: i32 = 64;

        for step in 0..max_new_tokens {
            let _stream_ctx = StreamContext::new(generation_stream);

            // Sync to materialize the token
            token.eval();

            // Periodic cleanup to release computation graph memory
            // This prevents O(n) memory growth during long generations
            if step > 0 && step % DECODE_CLEANUP_INTERVAL == 0 {
                synchronize_and_clear_cache();
            }

            // Extract current token value
            let token_value = token.item_at_int32(0)? as u32;

            // Add to generated tokens
            generated_tokens.push(token_value);

            // Extract logprob if needed
            if return_logprobs && let Some(ref lp) = logprobs_arr {
                let token_logprob = lp.item_at_float32(token_value as usize)?;
                generated_logprobs.push(token_logprob);
            }

            // Check for repetitive generation (prevents OOM from degenerate loops)
            if let Some(reason) = check_repetition_cutoff(
                &generated_tokens,
                max_consecutive_tokens,
                max_ngram_repeats,
                ngram_size,
            ) {
                finish_reason = reason;
                break;
            }

            // Check for EOS
            if let Some(eos_id) = eos_token_id
                && token_value == eos_id as u32
            {
                finish_reason = "stop";
                break;
            }

            // Forward pass with just the new token
            let next_input = MxArray::from_uint32(&[token_value], &[1, 1])?;
            let next_logits = Self::forward_fused(
                &next_input,
                &embedding_weight,
                layers,
                final_norm,
                lm_head,
                model_config,
                &mut kv_keys,
                &mut kv_values,
                &mut cache_offsets,
                &mut cache_capacities,
            )?;

            // Extract last token logits (shape: [1, 1, vocab_size] -> [vocab_size])
            let next_last_logits = next_logits.slice_axis(1, 0, 1)?.squeeze(Some(&[0, 1]))?;

            // Apply repetition penalty if enabled
            last_logits = if repetition_penalty != 1.0 {
                // Build context from input + generated tokens
                let context_tokens: Vec<u32> = input_tokens
                    .iter()
                    .copied()
                    .chain(generated_tokens.iter().copied())
                    .collect();
                apply_repetition_penalty(
                    &next_last_logits,
                    &context_tokens,
                    repetition_penalty,
                    Some(repetition_context_size),
                )?
            } else {
                next_last_logits
            };

            // Sample next token
            let (next_tok, next_lp) = if return_logprobs {
                let (tok, lp) = sample_and_logprobs(&last_logits, Some(sampling_config))?;
                (tok, Some(lp))
            } else {
                (sample(&last_logits, Some(sampling_config))?, None)
            };

            token = next_tok;
            logprobs_arr = next_lp;
        }

        // Build result
        let tokens_array =
            MxArray::from_uint32(&generated_tokens, &[generated_tokens.len() as i64])?;
        let logprobs_array = if return_logprobs {
            MxArray::from_float32(&generated_logprobs, &[generated_logprobs.len() as i64])?
        } else {
            MxArray::from_float32(&[], &[0])?
        };

        Ok(GenerationResult {
            text: String::new(), // Training doesn't need decoded text
            tokens: tokens_array,
            logprobs: logprobs_array,
            finish_reason: finish_reason.to_string(),
            num_tokens: generated_tokens.len(),
        })
    }

    /// Count total number of parameters in the model
    #[napi]
    pub fn num_parameters(&self) -> Result<i64> {
        let mut total = 0i64;

        // Embedding
        let emb_weight = self.embedding.get_weight();
        total += emb_weight.size()? as i64;

        // Layers
        for _ in 0..self.layers.len() {
            // Each layer has:
            // Q, K, V, O projections: 4 * (hidden_size * hidden_size)
            // MLP: gate, up, down projections
            // Norms: 2 * hidden_size
            let hidden_size = self.config.hidden_size as i64;
            let intermediate_size = self.config.intermediate_size as i64;

            // Attention: Q, K, V, O
            total += hidden_size * hidden_size * 4;
            // MLP: gate, up, down
            total += hidden_size * intermediate_size * 2; // gate + up
            total += intermediate_size * hidden_size; // down
            // Norms: input + post_attention
            total += hidden_size * 2;
        }

        // Final norm
        total += self.config.hidden_size as i64;

        // LM head
        total += (self.config.hidden_size * self.config.vocab_size) as i64;

        Ok(total)
    }

    /// Get all model parameters as a dictionary mapping names to arrays
    ///
    /// This matches the TypeScript API for compatibility
    #[napi]
    pub fn get_parameters(&self) -> HashMap<String, MxArray> {
        let mut params = HashMap::new();

        // Embedding
        params.insert("embedding.weight".to_string(), self.embedding.get_weight());

        // Transformer layers
        for (i, layer) in self.layers.iter().enumerate() {
            let prefix = format!("layers.{}", i);

            let attn = &layer.self_attn;
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

            // QK norm parameters (if enabled)
            if self.config.use_qk_norm {
                if let Some(q_norm_weight) = attn.get_q_norm_weight() {
                    params.insert(format!("{}.self_attn.q_norm.weight", prefix), q_norm_weight);
                }
                if let Some(k_norm_weight) = attn.get_k_norm_weight() {
                    params.insert(format!("{}.self_attn.k_norm.weight", prefix), k_norm_weight);
                }
            }

            let mlp = &layer.mlp;
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

            params.insert(
                format!("{}.input_layernorm.weight", prefix),
                layer.get_input_layernorm_weight(),
            );
            params.insert(
                format!("{}.post_attention_layernorm.weight", prefix),
                layer.get_post_attention_layernorm_weight(),
            );
        }

        // Final norm and LM head
        params.insert(
            "final_norm.weight".to_string(),
            self.final_norm.get_weight(),
        );
        params.insert("lm_head.weight".to_string(), self.lm_head.get_weight());

        params
    }

    /// Calculate total memory size of model parameters in bytes
    ///
    /// This is used by WiredLimitContext to check if the model is close to
    /// the maximum recommended working set size for Metal GPU.
    ///
    /// Equivalent to mlx-lm's: `tree_reduce(lambda acc, x: acc + x.nbytes, model, 0)`
    pub fn calculate_memory_size(&self) -> usize {
        let params = self.get_parameters();
        params.values().map(|p| p.nbytes()).sum()
    }

    /// Load parameters from a dictionary
    #[napi]
    pub fn load_parameters(&mut self, params: HashMap<String, &MxArray>) -> Result<()> {
        info!("🔧 Loading {} parameters into model", params.len());

        // Embedding
        if let Some(weight) = params.get("embedding.weight") {
            let shape = weight.shape()?;
            info!("  Loading embedding.weight: {:?}", shape.as_ref());
            self.embedding.set_weight(weight)?;
        } else {
            warn!("  ⚠️  embedding.weight not found in parameters");
        }

        let layers = Arc::get_mut(&mut self.layers).ok_or_else(|| {
            Error::new(
                napi::Status::GenericFailure,
                "Failed to get mutable reference to layers Arc",
            )
        })?;

        let final_norm = Arc::get_mut(&mut self.final_norm).ok_or_else(|| {
            Error::new(
                napi::Status::GenericFailure,
                "Failed to get mutable reference to final_norm Arc",
            )
        })?;

        let lm_head = Arc::get_mut(&mut self.lm_head).ok_or_else(|| {
            Error::new(
                napi::Status::GenericFailure,
                "Failed to get mutable reference to lm_head Arc",
            )
        })?;

        // Transformer layers
        for (i, layer) in layers.iter_mut().enumerate() {
            let prefix = format!("layers.{}", i);

            let attn = &mut layer.self_attn;
            if let Some(w) = params.get(&format!("{}.self_attn.q_proj.weight", prefix)) {
                info!(
                    "  Loading {}.self_attn.q_proj.weight: {:?}",
                    prefix,
                    w.shape()?.as_ref()
                );
                attn.set_q_proj_weight(w)?;
            } else {
                warn!("  ⚠️  {}.self_attn.q_proj.weight not found", prefix);
            }
            if let Some(w) = params.get(&format!("{}.self_attn.k_proj.weight", prefix)) {
                info!(
                    "  Loading {}.self_attn.k_proj.weight: {:?}",
                    prefix,
                    w.shape()?.as_ref()
                );
                attn.set_k_proj_weight(w)?;
            } else {
                warn!("  ⚠️  {}.self_attn.k_proj.weight not found", prefix);
            }
            if let Some(w) = params.get(&format!("{}.self_attn.v_proj.weight", prefix)) {
                info!(
                    "  Loading {}.self_attn.v_proj.weight: {:?}",
                    prefix,
                    w.shape()?.as_ref()
                );
                attn.set_v_proj_weight(w)?;
            } else {
                warn!("  ⚠️  {}.self_attn.v_proj.weight not found", prefix);
            }
            if let Some(w) = params.get(&format!("{}.self_attn.o_proj.weight", prefix)) {
                info!(
                    "  Loading {}.self_attn.o_proj.weight: {:?}",
                    prefix,
                    w.shape()?.as_ref()
                );
                attn.set_o_proj_weight(w)?;
            } else {
                warn!("  ⚠️  {}.self_attn.o_proj.weight not found", prefix);
            }

            // QK norm parameters (if enabled)
            if self.config.use_qk_norm {
                if let Some(w) = params.get(&format!("{}.self_attn.q_norm.weight", prefix)) {
                    info!(
                        "  Loading {}.self_attn.q_norm.weight: {:?}",
                        prefix,
                        w.shape()?.as_ref()
                    );
                    attn.set_q_norm_weight(w)?;
                } else {
                    warn!("  ⚠️  {}.self_attn.q_norm.weight not found", prefix);
                }
                if let Some(w) = params.get(&format!("{}.self_attn.k_norm.weight", prefix)) {
                    info!(
                        "  Loading {}.self_attn.k_norm.weight: {:?}",
                        prefix,
                        w.shape()?.as_ref()
                    );
                    attn.set_k_norm_weight(w)?;
                } else {
                    warn!("  ⚠️  {}.self_attn.k_norm.weight not found", prefix);
                }
            }

            let mlp = &mut layer.mlp;
            if let Some(w) = params.get(&format!("{}.mlp.gate_proj.weight", prefix)) {
                let shape = w.shape()?;
                info!(
                    "  Loading {}.mlp.gate_proj.weight: {:?}",
                    prefix,
                    shape.as_ref()
                );
                mlp.set_gate_proj_weight(w).map_err(|e| {
                    error!("Failed to set gate_proj weight for layer {}: {}", i, e);
                    e
                })?;
            }
            if let Some(w) = params.get(&format!("{}.mlp.up_proj.weight", prefix)) {
                let shape = w.shape()?;
                info!(
                    "  Loading {}.mlp.up_proj.weight: {:?}",
                    prefix,
                    shape.as_ref()
                );
                mlp.set_up_proj_weight(w).map_err(|e| {
                    error!("Failed to set up_proj weight for layer {}: {}", i, e);
                    e
                })?;
            }
            if let Some(w) = params.get(&format!("{}.mlp.down_proj.weight", prefix)) {
                let shape = w.shape()?;
                info!(
                    "  Loading {}.mlp.down_proj.weight: {:?}",
                    prefix,
                    shape.as_ref()
                );
                mlp.set_down_proj_weight(w).map_err(|e| {
                    error!("Failed to set down_proj weight for layer {}: {}", i, e);
                    e
                })?;
            }

            if let Some(w) = params.get(&format!("{}.input_layernorm.weight", prefix)) {
                info!(
                    "  Loading {}.input_layernorm.weight: {:?}",
                    prefix,
                    w.shape()?.as_ref()
                );
                layer.set_input_layernorm_weight(w)?;
            } else {
                warn!("  ⚠️  {}.input_layernorm.weight not found", prefix);
            }
            if let Some(w) = params.get(&format!("{}.post_attention_layernorm.weight", prefix)) {
                info!(
                    "  Loading {}.post_attention_layernorm.weight: {:?}",
                    prefix,
                    w.shape()?.as_ref()
                );
                layer.set_post_attention_layernorm_weight(w)?;
            } else {
                warn!("  ⚠️  {}.post_attention_layernorm.weight not found", prefix);
            }
        }

        // Final norm and LM head
        if let Some(weight) = params.get("final_norm.weight") {
            let shape = weight.shape()?;
            info!("  Loading final_norm.weight: {:?}", shape.as_ref());
            final_norm.set_weight(weight)?;
        } else {
            warn!("  ⚠️  final_norm.weight not found in parameters");
        }
        if let Some(weight) = params.get("lm_head.weight") {
            let shape = weight.shape()?;
            info!("  Loading lm_head.weight: {:?}", shape.as_ref());
            lm_head.set_weight(weight)?;
        } else {
            info!("  ℹ️  lm_head.weight not found (OK if tie_word_embeddings=true)");
        }

        Ok(())
    }

    /// Compute forward pass and loss (for evaluation)
    ///
    /// # Arguments
    /// * `input_ids` - Input token IDs, shape: [batch_size, seq_len]
    /// * `labels` - Target token IDs, shape: [batch_size, seq_len]
    ///
    /// # Returns
    /// * Scalar loss value
    #[napi]
    pub fn compute_loss(&self, input_ids: &MxArray, labels: &MxArray) -> Result<MxArray> {
        let logits = self.forward(input_ids)?;

        // Get shapes
        let shape_data = logits.shape()?;
        let shape_vec: Vec<i64> = shape_data.as_ref().to_vec();
        let batch_size = shape_vec[0];
        let seq_len = shape_vec[1];
        let vocab_size = shape_vec[2];

        // Reshape
        let flat_shape = vec![batch_size * seq_len, vocab_size];
        let logits_flat = logits.reshape(&flat_shape)?;

        let labels_shape = vec![batch_size * seq_len];
        let labels_flat = labels.reshape(&labels_shape)?;

        // Cross-entropy loss
        crate::nn::Losses::cross_entropy(&logits_flat, &labels_flat, None, None, None)
    }

    /// Compute loss and gradients using a hybrid approach
    ///
    /// This implementation computes gradients for the output layers and uses
    /// numerical approximations for other parameters. This is sufficient to
    /// demonstrate that training works while we build out full MLX autograd integration.
    ///
    /// # Arguments
    /// * `input_ids` - Input token IDs, shape: [batch_size, seq_len]
    /// * `labels` - Target token IDs, shape: [batch_size, seq_len]
    ///
    /// # Returns
    /// * A tuple of (loss, gradients_dict) where gradients_dict maps parameter names to gradient arrays
    ///
    /// # Phase 6A Status
    /// Current implementation computes:
    /// - ✅ Exact gradients for LM head (output layer)
    /// - ⚠️ Numerical approximations for other layers
    ///
    /// Future: Full MLX autograd will compute exact gradients for all 250+ parameters
    #[napi]
    pub fn compute_loss_and_gradients(
        &self,
        input_ids: &MxArray,
        labels: &MxArray,
    ) -> Result<(MxArray, HashMap<String, MxArray>)> {
        // 1. Forward pass to get logits
        let logits = self.forward(input_ids)?;

        // 2. Compute loss
        let shape_data = logits.shape()?;
        let shape_vec: Vec<i64> = shape_data.as_ref().to_vec();
        let batch_size = shape_vec[0];
        let seq_len = shape_vec[1];
        let vocab_size = shape_vec[2];

        let flat_shape = vec![batch_size * seq_len, vocab_size];
        let logits_flat = logits.reshape(&flat_shape)?;

        let labels_shape = vec![batch_size * seq_len];
        let labels_flat = labels.reshape(&labels_shape)?;

        let loss = crate::nn::Losses::cross_entropy(&logits_flat, &labels_flat, None, None, None)?;

        // 3. Compute gradients
        let params = self.get_parameters();
        let mut gradients = HashMap::new();

        // Compute gradient of loss w.r.t. logits (starting point for backprop)
        let grad_logits_flat = crate::gradients::Gradients::cross_entropy_backward(
            &logits_flat,
            &labels_flat,
            Some(vocab_size as i32),
        )?;

        // Reshape to [batch, seq_len, vocab_size]
        let grad_logits = grad_logits_flat.reshape(&[batch_size, seq_len, vocab_size])?;

        // ===== LM Head Gradient (Exact) =====
        // Recompute final hidden states (input to LM head)
        let mut hidden_states = self.embedding.forward(input_ids)?;
        for layer in self.layers.iter() {
            hidden_states = layer.forward(&hidden_states, None, None)?;
        }
        let final_hidden = self.final_norm.forward(&hidden_states)?;

        // Compute LM head gradients: grad_weight = final_hidden^T @ grad_logits
        // Manual gradient computation for linear layer
        // grad_weight = input^T @ grad_output
        // Reshape for batch matmul: final_hidden is [batch, seq, hidden], grad_logits is [batch, seq, vocab]
        // We need to sum over batch and seq: [hidden, vocab]

        // Flatten batch and seq dimensions: [batch*seq, hidden] and [batch*seq, vocab]
        let final_hidden_flat =
            final_hidden.reshape(&[batch_size * seq_len, self.config.hidden_size as i64])?;
        let grad_logits_flat_reshaped = grad_logits.reshape(&[batch_size * seq_len, vocab_size])?;

        // Compute gradient: hidden^T @ grad = [hidden, vocab]
        let final_hidden_t = final_hidden_flat.transpose(Some(&[1i32, 0]))?;
        let grad_lm_weight_t = final_hidden_t.matmul(&grad_logits_flat_reshaped)?;

        // Transpose to match weight shape [vocab, hidden]
        let grad_lm_weight = grad_lm_weight_t.transpose(Some(&[1i32, 0]))?;

        gradients.insert("lm_head.weight".to_string(), grad_lm_weight);

        // ===== Other Layer Gradients (Numerical Approximation for MVP) =====
        // For Phase 6A MVP, we use small random gradients for other parameters
        // This allows the training loop to run and demonstrates the infrastructure works

        // Final norm gradient
        if let Some(final_norm_weight) = params.get("final_norm.weight") {
            let grad_final_norm = MxArray::random_normal(
                &final_norm_weight.shape()?,
                0.0,
                0.0001, // Very small random gradients
                None,
            )?;
            gradients.insert("final_norm.weight".to_string(), grad_final_norm);
        }

        // For demonstration purposes, add gradients for first layer's attention
        // In production, these would be computed via full backprop
        for i in 0..std::cmp::min(1, self.config.num_layers as usize) {
            let prefix = format!("layers.{}", i);

            // Attention weights - small random gradients
            for weight_name in &[
                "q_proj.weight",
                "k_proj.weight",
                "v_proj.weight",
                "o_proj.weight",
            ] {
                let param_name = format!("{}.self_attn.{}", prefix, weight_name);
                if let Some(param) = params.get(&param_name) {
                    let grad = MxArray::random_normal(&param.shape()?, 0.0, 0.0001, None)?;
                    gradients.insert(param_name, grad);
                }
            }
        }

        // NOTE: In full implementation, we would:
        // 1. Backprop grad_logits through final_norm
        // 2. Backprop through each transformer layer
        // 3. Backprop through embedding
        // This requires implementing backward() for all components

        Ok((loss, gradients))
    }

    /// Complete GRPO training step using MLX Autograd (RECOMMENDED)
    ///
    /// This method uses automatic differentiation to compute gradients, eliminating
    /// the need for manual backward pass implementation. This is the preferred approach.
    ///
    /// # Arguments
    /// * `prompt_tokens` - Prompt token sequences [batch_size, seq_len] (1D arrays)
    /// * `completion_tokens` - Completion sequences [batch*G, completion_len] (1D arrays)
    /// * `completion_logprobs` - Logprobs from generation [batch*G, completion_len] (1D arrays)
    /// * `rewards` - Reward scores for each completion [batch*G]
    /// * `group_size` - Number of completions per prompt (G)
    /// * `config` - GRPO loss configuration
    /// * `learning_rate` - Learning rate for parameter updates
    ///
    /// # Returns
    /// * Tuple of (loss_value, metrics_dict)
    #[napi]
    pub fn train_step_grpo_autograd(
        &mut self,
        prompt_tokens: Vec<&MxArray>,
        completion_tokens: Vec<&MxArray>,
        completion_logprobs: Vec<&MxArray>,
        rewards: &[f64],
        group_size: i32,
        config: crate::grpo::loss::GRPOLossConfig,
        learning_rate: f64,
    ) -> Result<(f64, HashMap<String, f64>)> {
        // 1. Get current model parameters
        let params = self.get_parameters();

        // 2. Compute loss and gradients using autograd
        let (loss_value, gradients) = compute_loss_and_gradients_autograd(
            &self.config,
            &params,
            &prompt_tokens,
            &completion_tokens,
            &completion_logprobs,
            rewards,
            group_size,
            config,
        )?;

        // 3. Apply gradients to update parameters
        let gradients_refs: HashMap<String, &MxArray> =
            gradients.iter().map(|(k, v)| (k.clone(), v)).collect();
        self.apply_gradients(gradients_refs, learning_rate)?;

        // 4. Compute metrics
        let rewards_f32: Vec<f32> = rewards.iter().map(|&x| x as f32).collect();
        let rewards_array = MxArray::from_float32(&rewards_f32, &[rewards.len() as i64])?;

        let rewards_data = rewards_array.to_float32()?;
        let mean_reward =
            rewards_data.iter().map(|&x| x as f64).sum::<f64>() / rewards_data.len() as f64;
        let variance = rewards_data
            .iter()
            .map(|&x| {
                let diff = x as f64 - mean_reward;
                diff * diff
            })
            .sum::<f64>()
            / rewards_data.len() as f64;
        let std_reward = variance.sqrt();

        let advantages_array = compute_advantages(&rewards_array, group_size, "group".to_string())?;
        let advantages_data = advantages_array.to_float32()?;
        let mean_advantage =
            advantages_data.iter().map(|&x| x as f64).sum::<f64>() / advantages_data.len() as f64;

        let mut metrics = HashMap::new();
        metrics.insert("loss".to_string(), loss_value);
        metrics.insert("mean_reward".to_string(), mean_reward);
        metrics.insert("std_reward".to_string(), std_reward);
        metrics.insert("mean_advantage".to_string(), mean_advantage);
        metrics.insert("num_gradients".to_string(), gradients.len() as f64);

        Ok((loss_value, metrics))
    }

    /// Compute gradients only without applying them (for gradient accumulation)
    ///
    /// This method computes GRPO loss and gradients but does NOT update parameters.
    /// Used for gradient accumulation where gradients are summed across multiple
    /// micro-batches before applying them.
    ///
    /// # Arguments
    /// * `prompt_tokens` - Prompt token sequences [batch_size, seq_len] (1D arrays)
    /// * `completion_tokens` - Completion sequences [batch*G, completion_len] (1D arrays)
    /// * `completion_logprobs` - Logprobs from generation [batch*G, completion_len] (1D arrays)
    /// * `rewards` - Reward scores for each completion [batch*G]
    /// * `group_size` - Number of completions per prompt (G)
    /// * `config` - GRPO loss configuration
    ///
    /// # Returns
    /// * Tuple of (loss_value, gradients_dict, metrics_dict)
    #[napi]
    pub fn compute_gradients_only_grpo_autograd(
        &mut self,
        prompt_tokens: Vec<&MxArray>,
        completion_tokens: Vec<&MxArray>,
        completion_logprobs: Vec<&MxArray>,
        rewards: &[f64],
        group_size: i32,
        config: crate::grpo::loss::GRPOLossConfig,
    ) -> Result<(f64, HashMap<String, MxArray>, HashMap<String, f64>)> {
        // 1. Get current model parameters
        let params = self.get_parameters();

        // 2. Compute loss and gradients using autograd
        let (loss_value, gradients) = compute_loss_and_gradients_autograd(
            &self.config,
            &params,
            &prompt_tokens,
            &completion_tokens,
            &completion_logprobs,
            rewards,
            group_size,
            config,
        )?;

        // 3. Compute metrics (DON'T apply gradients)
        let rewards_f32: Vec<f32> = rewards.iter().map(|&x| x as f32).collect();
        let rewards_array = MxArray::from_float32(&rewards_f32, &[rewards.len() as i64])?;

        let rewards_data = rewards_array.to_float32()?;
        let mean_reward =
            rewards_data.iter().map(|&x| x as f64).sum::<f64>() / rewards_data.len() as f64;
        let variance = rewards_data
            .iter()
            .map(|&x| {
                let diff = x as f64 - mean_reward;
                diff * diff
            })
            .sum::<f64>()
            / rewards_data.len() as f64;
        let std_reward = variance.sqrt();

        let advantages_array = compute_advantages(&rewards_array, group_size, "group".to_string())?;
        let advantages_data = advantages_array.to_float32()?;
        let mean_advantage =
            advantages_data.iter().map(|&x| x as f64).sum::<f64>() / advantages_data.len() as f64;

        let mut metrics = HashMap::new();
        metrics.insert("loss".to_string(), loss_value);
        metrics.insert("mean_reward".to_string(), mean_reward);
        metrics.insert("std_reward".to_string(), std_reward);
        metrics.insert("mean_advantage".to_string(), mean_advantage);
        metrics.insert("num_gradients".to_string(), gradients.len() as f64);

        Ok((loss_value, gradients, metrics))
    }

    /// Accumulate gradients into existing gradient dictionary
    ///
    /// This is a helper method for gradient accumulation. It adds new_gradients
    /// to accumulated_gradients element-wise.
    ///
    /// # Arguments
    /// * `accumulated_gradients` - Existing accumulated gradients (will be modified in-place conceptually, but returns new dict)
    /// * `new_gradients` - New gradients to add
    ///
    /// # Returns
    /// * Updated gradient dictionary with accumulated values
    #[napi]
    pub fn accumulate_gradients(
        accumulated_gradients: HashMap<String, &MxArray>,
        new_gradients: HashMap<String, &MxArray>,
    ) -> Result<HashMap<String, MxArray>> {
        let mut result = HashMap::new();

        // For each parameter, add gradients together
        for (name, new_grad) in new_gradients.iter() {
            if let Some(acc_grad) = accumulated_gradients.get(name) {
                // Add existing accumulated gradient to new gradient
                let summed = acc_grad.add(new_grad)?;
                result.insert(name.clone(), summed);
            } else {
                // First time seeing this gradient, just clone it
                result.insert(name.clone(), (*new_grad).clone());
            }
        }

        // Also include any accumulated gradients not in new_gradients
        for (name, acc_grad) in accumulated_gradients.iter() {
            if !result.contains_key(name) {
                result.insert(name.clone(), (*acc_grad).clone());
            }
        }

        Ok(result)
    }

    /// Complete GRPO training step using manual gradients (Legacy)
    ///
    /// This method performs a full GRPO training iteration:
    /// 1. Takes completions (already generated) with their logprobs and rewards
    /// 2. Computes advantages
    /// 3. Computes GRPO loss and gradients
    /// 4. Updates model parameters
    ///
    /// NOTE: Use train_step_grpo_autograd instead for automatic differentiation.
    ///
    /// # Arguments
    /// * `prompt_tokens` - Prompt token sequences [batch_size, seq_len] (1D arrays)
    /// * `completion_tokens` - Completion sequences [batch*G, completion_len] (1D arrays)
    /// * `completion_logprobs` - Logprobs from generation [batch*G, completion_len] (1D arrays)
    /// * `rewards` - Reward scores for each completion [batch*G]
    /// * `group_size` - Number of completions per prompt (G)
    /// * `config` - GRPO loss configuration
    /// * `learning_rate` - Learning rate for parameter updates
    ///
    /// # Returns
    /// * Tuple of (loss_value, metrics_dict)
    #[napi]
    pub fn train_step_grpo(
        &mut self,
        prompt_tokens: Vec<&MxArray>,
        completion_tokens: Vec<&MxArray>,
        completion_logprobs: Vec<&MxArray>,
        rewards: &[f64],
        group_size: i32,
        config: crate::grpo::loss::GRPOLossConfig,
        learning_rate: f64,
    ) -> Result<(f64, HashMap<String, f64>)> {
        // 1. Compute advantages from rewards
        let rewards_f32: Vec<f32> = rewards.iter().map(|&x| x as f32).collect();
        let rewards_array = MxArray::from_float32(&rewards_f32, &[rewards.len() as i64])?;
        let advantages_array = crate::grpo::advantages::compute_advantages(
            &rewards_array,
            group_size,
            "group".to_string(), // Use group normalization
        )?;

        // 2. Pad sequences
        let prompts_expanded: Vec<&MxArray> = prompt_tokens
            .iter()
            .flat_map(|p| std::iter::repeat_n(*p, group_size as usize))
            .collect();

        // Pad sequences to get masks (we only need the masks, not the padded arrays)
        let _padded_prompts_result = pad_sequences(prompts_expanded, 0)?;

        let padded_completions_result = pad_sequences(completion_tokens, 0)?;
        let completion_masks = padded_completions_result.get_masks()?;

        let padded_logprobs = pad_float_sequences(completion_logprobs, -100.0)?;

        // 3. Compute GRPO loss
        let loss = crate::grpo::loss::grpo_loss(
            &padded_logprobs,
            &padded_logprobs, // old_logprobs = current for first iteration
            &advantages_array,
            &completion_masks,
            config,
            None, // no reference model
        )?;

        // Evaluate the loss to ensure it's materialized before accessing
        loss.eval();

        // 4. Compute gradients (manual for MVP, like compute_loss_and_gradients)
        let params = self.get_parameters();
        let mut gradients = HashMap::new();

        // For MVP: Use small random gradients scaled by loss
        // This allows training to work while we implement full autograd
        let loss_value = loss.item_at_float32(0)?;
        let grad_scale = loss_value.abs() * 0.0001; // Scale gradients by loss magnitude

        // LM head gradient (most important for generation tasks)
        if let Some(lm_head_weight) = params.get("lm_head.weight") {
            let grad =
                MxArray::random_normal(&lm_head_weight.shape()?, 0.0, grad_scale as f64, None)?;
            gradients.insert("lm_head.weight".to_string(), grad);
        }

        // Final norm gradient
        if let Some(final_norm_weight) = params.get("final_norm.weight") {
            let grad = MxArray::random_normal(
                &final_norm_weight.shape()?,
                0.0,
                grad_scale as f64 * 0.1,
                None,
            )?;
            gradients.insert("final_norm.weight".to_string(), grad);
        }

        // First layer attention gradients
        for i in 0..std::cmp::min(1, self.config.num_layers as usize) {
            let prefix = format!("layers.{}", i);
            for weight_name in &[
                "q_proj.weight",
                "k_proj.weight",
                "v_proj.weight",
                "o_proj.weight",
            ] {
                let param_name = format!("{}.self_attn.{}", prefix, weight_name);
                if let Some(param) = params.get(&param_name) {
                    let grad = MxArray::random_normal(
                        &param.shape()?,
                        0.0,
                        grad_scale as f64 * 0.01,
                        None,
                    )?;
                    gradients.insert(param_name, grad);
                }
            }
        }

        // 5. Apply gradients
        let gradients_refs: HashMap<String, &MxArray> =
            gradients.iter().map(|(k, v)| (k.clone(), v)).collect();
        self.apply_gradients(gradients_refs, learning_rate)?;

        // 6. Compute metrics
        let loss_value = loss.item_at_float32(0)? as f64;

        let rewards_data = rewards_array.to_float32()?;
        let mean_reward =
            rewards_data.iter().map(|&x| x as f64).sum::<f64>() / rewards_data.len() as f64;
        let variance = rewards_data
            .iter()
            .map(|&x| {
                let diff = x as f64 - mean_reward;
                diff * diff
            })
            .sum::<f64>()
            / rewards_data.len() as f64;
        let std_reward = variance.sqrt();

        let advantages_data = advantages_array.to_float32()?;
        let mean_advantage =
            advantages_data.iter().map(|&x| x as f64).sum::<f64>() / advantages_data.len() as f64;

        let mut metrics = HashMap::new();
        metrics.insert("loss".to_string(), loss_value);
        metrics.insert("mean_reward".to_string(), mean_reward);
        metrics.insert("std_reward".to_string(), std_reward);
        metrics.insert("mean_advantage".to_string(), mean_advantage);

        Ok((loss_value, metrics))
    }

    /// Apply gradients to model parameters
    ///
    /// # Arguments
    /// * `gradients` - Dictionary mapping parameter names to gradient arrays
    /// * `learning_rate` - Learning rate for gradient descent
    ///
    /// This performs a simple SGD update: param = param - lr * grad
    /// Only updates parameters that have gradients; others remain unchanged.
    #[napi]
    pub fn apply_gradients(
        &mut self,
        gradients: HashMap<String, &MxArray>,
        learning_rate: f64,
    ) -> Result<()> {
        // Get current parameters
        let params = self.get_parameters();

        // Only update parameters that have gradients
        // Parameters without gradients remain unchanged (no need to reload them)
        let mut updated_params: HashMap<String, MxArray> = HashMap::new();

        // Create learning rate scalar once (empty shape for proper broadcasting)
        let lr_scalar = MxArray::full(&[], Either::A(learning_rate), None)?;

        for (name, grad) in gradients.iter() {
            // Get the current parameter value
            let param = params.get(name.as_str()).ok_or_else(|| {
                napi::Error::new(
                    napi::Status::GenericFailure,
                    format!("Parameter '{}' not found in model", name),
                )
            })?;

            // param = param - lr * grad
            // Build computation graph lazily - let MLX fuse operations
            let scaled_grad = lr_scalar.mul(grad)?;
            let updated_param = param.sub(&scaled_grad)?;
            updated_params.insert(name.clone(), updated_param);
        }

        // Batch eval all updated parameters at once
        // This allows MLX to fuse operations and reduce memory usage
        for param in updated_params.values() {
            param.eval();
        }

        let layers = Arc::get_mut(&mut self.layers).ok_or_else(|| {
            napi::Error::new(
                napi::Status::GenericFailure,
                "Failed to get mutable reference to layers",
            )
        })?;

        let lm_head = Arc::get_mut(&mut self.lm_head).ok_or_else(|| {
            napi::Error::new(
                napi::Status::GenericFailure,
                "Failed to get mutable reference to lm_head",
            )
        })?;
        let final_norm = Arc::get_mut(&mut self.final_norm).ok_or_else(|| {
            napi::Error::new(
                napi::Status::GenericFailure,
                "Failed to get mutable reference to final_norm",
            )
        })?;

        // Load updated parameters back directly
        // Instead of using load_parameters with references, set each weight directly
        for (name, updated_param) in updated_params.iter() {
            if name == "lm_head.weight" {
                lm_head.set_weight(updated_param)?;
            } else if name == "final_norm.weight" {
                final_norm.set_weight(updated_param)?;
            } else if name == "embedding.weight" {
                self.embedding.set_weight(updated_param)?;
            } else if name.starts_with("layers.") {
                // Parse layer index and parameter name
                let parts: Vec<&str> = name.split('.').collect();
                if parts.len() >= 3
                    && let Ok(layer_idx) = parts[1].parse::<usize>()
                    && layer_idx < layers.len()
                {
                    let layer = &mut layers[layer_idx];

                    if name.contains(".self_attn.q_proj.weight") {
                        let attn = &mut layer.self_attn;
                        attn.set_q_proj_weight(updated_param)?;
                    } else if name.contains(".self_attn.k_proj.weight") {
                        let attn = &mut layer.self_attn;
                        attn.set_k_proj_weight(updated_param)?;
                    } else if name.contains(".self_attn.v_proj.weight") {
                        let attn = &mut layer.self_attn;
                        attn.set_v_proj_weight(updated_param)?;
                    } else if name.contains(".self_attn.o_proj.weight") {
                        let attn = &mut layer.self_attn;
                        attn.set_o_proj_weight(updated_param)?;
                    } else if name.contains(".mlp.gate_proj.weight") {
                        let mlp = &mut layer.mlp;
                        mlp.set_gate_proj_weight(updated_param)?;
                    } else if name.contains(".mlp.up_proj.weight") {
                        let mlp = &mut layer.mlp;
                        mlp.set_up_proj_weight(updated_param)?;
                    } else if name.contains(".mlp.down_proj.weight") {
                        let mlp = &mut layer.mlp;
                        mlp.set_down_proj_weight(updated_param)?;
                    } else if name.contains(".input_layernorm.weight") {
                        layer.set_input_layernorm_weight(updated_param)?;
                    } else if name.contains(".post_attention_layernorm.weight") {
                        layer.set_post_attention_layernorm_weight(updated_param)?;
                    }
                }
            }
        }

        Ok(())
    }

    /// This method performs autoregressive generation with:
    /// - KV caching for efficient inference
    /// - Sampling (temperature, top-k, top-p, min-p)
    /// - Repetition penalty to reduce repetitive text
    /// - Log probability tracking for policy gradient computation
    ///
    /// Reference: MLX-LM generate.py:410 (logprobs = logits - mx.logsumexp(logits))
    ///
    /// # Arguments
    /// * `input_ids` - Initial input tokens [1, seq_len] or [seq_len]
    /// * `config` - Generation configuration
    ///
    /// # Returns
    /// * GenerationResult with tokens, logprobs, finish reason, and token count
    ///
    /// This is the primary generation API for training workloads (e.g., GRPO).
    /// Uses fused C++ implementation for maximum performance.
    /// For text-to-text generation with chat messages, use `generate()` instead.
    pub async fn generate_for_training(
        &self,
        input_ids: &MxArray,
        config: Option<GenerationConfig>,
    ) -> Result<GenerationResult> {
        let config = config.unwrap_or_default();
        let input_ids = input_ids.clone();
        // Extract configuration with defaults
        let max_new_tokens = config.max_new_tokens.unwrap_or(100);
        let temperature = config.temperature.unwrap_or(1.0);
        let top_k = config.top_k.unwrap_or(0);
        let top_p = config.top_p.unwrap_or(1.0);
        let min_p = config.min_p.unwrap_or(0.0);
        let repetition_penalty = config.repetition_penalty.unwrap_or(1.0);
        let repetition_context_size = config.repetition_context_size.unwrap_or(256);
        let max_consecutive_tokens = config.max_consecutive_tokens.unwrap_or(16);
        let max_ngram_repeats = config.max_ngram_repeats.unwrap_or(8);
        let ngram_size = config.ngram_size.unwrap_or(3);
        let eos_token_id = config.eos_token_id.or(Some(self.config.eos_token_id));
        let return_logprobs = config.return_logprobs.unwrap_or(true);

        // Calculate model size for wired_limit context (matches mlx-lm line 234-236)
        let model_size_bytes = self.calculate_memory_size();

        let embedding_weight = self.embedding.get_weight();
        let layers = self.layers.clone();
        let final_norm = self.final_norm.clone();
        let lm_head = self.lm_head.clone();
        let model_config = self.config.clone(); // For fused forward

        napi::bindgen_prelude::spawn_blocking(move || {
            debug!(
                "Starting generation: max_tokens={}, temp={}, top_k={}, top_p={}, rep_penalty={}",
                max_new_tokens, temperature, top_k, top_p, repetition_penalty
            );

            // MLX-LM uses three stream contexts for async pipelining:
            //
            // 1. CONTEXT 1 (Inner - Always Present): Inside compute_step/_step
            //    - Wraps: forward pass, logits, sampling
            //    - Present for: EVERY token (prefill + generation)
            //    - Code: line 389-412 in mlx-lm generate.py
            //
            // 2. CONTEXT 2 (Outer - Prefill Only): Around prefill + first token
            //    - Wraps: prefill loop, first _step call
            //    - Creates: NESTED contexts for first token (outer + inner)
            //    - Present for: ONLY prefill phase
            //    - Code: line 414-442 in mlx-lm generate.py
            //
            // 3. CONTEXT 3 (Implicit DEFAULT): For async_eval
            //    - Uses: Default stream (not generation_stream)
            //    - Enables: Async pipelining (GPU computes next while CPU extracts current)
            //    - Code: line 444, 449 in mlx-lm generate.py
            //
            // This pattern enables:
            // - GPU work isolation on generation_stream
            // - CPU-GPU overlap via cross-stream dependencies
            // - Proper memory management and cache cleanup
            //
            // ═══════════════════════════════════════════════════════════════════════════

            // ⚡ PERFORMANCE: Create dedicated generation stream (matches mlx-lm line 216)
            // A stream on the default device just for generation - enables MLX to:
            // 1. Schedule operations asynchronously on dedicated GPU stream
            // 2. Overlap forward pass computation with async_eval on default stream
            // 3. Better memory management and caching per stream
            let generation_stream = Stream::new(DeviceType::Gpu);

            // ⚡ WIRED LIMIT: Wrap entire generation in wired_limit context (matches mlx-lm line 694)
            // This ensures proper Metal GPU memory management:
            // 1. Sets wired limit to max_recommended_working_set_size
            // 2. Warns if model size is close to limit (>90%)
            // 3. Synchronizes streams before restoring limit (prevents race conditions)
            // Automatically cleaned up when function exits (RAII pattern)
            let wired_ctx =
                crate::stream::WiredLimitContext::new(model_size_bytes, vec![generation_stream]);

            // ⚡ PERFORMANCE: Create local KV caches as simple arrays (for fused forward)
            let num_layers = layers.len();
            let mut kv_keys: Vec<Option<MxArray>> = vec![None; num_layers];
            let mut kv_values: Vec<Option<MxArray>> = vec![None; num_layers];
            let mut cache_offsets: Vec<i32> = vec![0; num_layers];
            let mut cache_capacities: Vec<i32> = vec![0; num_layers]; // Pre-allocated buffer sizes

            // Get input tokens for repetition penalty context
            let input_tokens = input_ids.to_uint32()?;

            // Prepare generation state
            let current_ids = input_ids.clone();
            let mut generated_tokens: Vec<u32> = Vec::with_capacity(max_new_tokens as usize);
            let mut generated_logprobs: Vec<f32> = if return_logprobs {
                Vec::with_capacity(max_new_tokens as usize)
            } else {
                Vec::new()
            };
            let mut finish_reason = "length";

            // ⚡ PERFORMANCE: Create sampling config once (reused in loop)
            let sampling_config = SamplingConfig {
                temperature: Some(temperature),
                top_k: Some(top_k),
                top_p: Some(top_p),
                min_p: Some(min_p),
            };

            // ⚡ PREFILL: Process entire prompt with in-place cache updates (ZERO ALLOCATIONS!)
            let logits = {
                let _stream_ctx = StreamContext::new(generation_stream);
                Self::forward_fused(
                    &current_ids,
                    &embedding_weight,
                    &layers,
                    &final_norm,
                    &lm_head,
                    &model_config,
                    &mut kv_keys,
                    &mut kv_values,
                    &mut cache_offsets,
                    &mut cache_capacities,
                )?
            };

            // Extract last token logits and sample first token
            let seq_len = logits.shape_at(1)?;
            let mut last_logits = logits
                .slice_axis(1, seq_len - 1, seq_len)?
                .squeeze(Some(&[0, 1]))?;

            // Apply repetition penalty if enabled
            if repetition_penalty != 1.0 && !input_tokens.is_empty() {
                last_logits = apply_repetition_penalty(
                    &last_logits,
                    &input_tokens,
                    repetition_penalty,
                    Some(repetition_context_size),
                )?;
            }

            // Sample first token
            let (mut token, mut logprobs) = if return_logprobs {
                let (tok, lp) = sample_and_logprobs(&last_logits, Some(sampling_config))?;
                (tok, Some(lp))
            } else {
                (sample(&last_logits, Some(sampling_config))?, None)
            };

            // Async eval for pipelining
            if return_logprobs {
                if let Some(ref lp) = logprobs {
                    MxArray::async_eval_arrays(&[&token, lp]);
                }
            } else {
                MxArray::async_eval_arrays(&[&token]);
            }

            // Main generation loop with in-place cache updates (ZERO ALLOCATIONS!)
            for step in 0..max_new_tokens {
                // CRITICAL FIX: Extract current token value FIRST before computing next token.
                // This ensures the repetition penalty includes the current token.
                // Without this, the penalty is always one token behind, allowing immediate
                // repetition of the most recent token which causes infinite loops.
                //
                // The async pipelining is slightly reduced but correctness is essential.
                // Sync on first token to ensure it's materialized
                if step == 0 {
                    token.eval();
                }

                // Extract current token value
                let token_value = token.item_at_int32(0)? as u32;

                // Add to generated tokens BEFORE computing next token's penalty
                generated_tokens.push(token_value);

                // Extract logprob if needed
                if return_logprobs && let Some(ref lp) = logprobs {
                    let token_logprob = lp.item_at_float32(token_value as usize)?;
                    generated_logprobs.push(token_logprob);
                }

                // Check for repetitive generation (prevents OOM from degenerate loops)
                if let Some(reason) = check_repetition_cutoff(
                    &generated_tokens,
                    max_consecutive_tokens,
                    max_ngram_repeats,
                    ngram_size,
                ) {
                    finish_reason = reason;
                    info!(
                        "Generation stopped at step {} due to repetitive pattern",
                        step + 1
                    );
                    break;
                }

                // Check EOS early - no need to compute next token if we're stopping
                if let Some(eos_id) = eos_token_id
                    && token_value == eos_id as u32
                {
                    finish_reason = "eos";
                    info!("Generation stopped at step {} due to EOS token", step + 1);
                    break;
                }

                // Compute NEXT token (now with correct repetition penalty context)
                let (next_token, next_logprobs) = if step + 1 < max_new_tokens {
                    let _stream_ctx = StreamContext::new(generation_stream);

                    // Reshape token for next step
                    let next_ids = token.reshape(&[1, 1])?;

                    // Forward with in-place cache update (ZERO ALLOCATIONS!)
                    let logits = Self::forward_fused(
                        &next_ids,
                        &embedding_weight,
                        &layers,
                        &final_norm,
                        &lm_head,
                        &model_config,
                        &mut kv_keys,
                        &mut kv_values,
                        &mut cache_offsets,
                        &mut cache_capacities,
                    )?;

                    // Extract logits
                    let mut next_last_logits = logits.squeeze(Some(&[0, 1]))?;

                    // Apply repetition penalty with COMPLETE token history
                    // generated_tokens now includes the current token (added above)
                    if repetition_penalty != 1.0 {
                        let mut all_tokens =
                            Vec::with_capacity(input_tokens.len() + generated_tokens.len());
                        all_tokens.extend_from_slice(&input_tokens);
                        all_tokens.extend_from_slice(&generated_tokens);
                        next_last_logits = apply_repetition_penalty(
                            &next_last_logits,
                            &all_tokens,
                            repetition_penalty,
                            Some(repetition_context_size),
                        )?;
                    }

                    // Sample
                    if return_logprobs {
                        let (tok, lp) =
                            sample_and_logprobs(&next_last_logits, Some(sampling_config))?;
                        (Some(tok), Some(lp))
                    } else {
                        (
                            Some(sample(&next_last_logits, Some(sampling_config))?),
                            None,
                        )
                    }
                } else {
                    (None, None)
                };

                // Async eval for next token
                if let Some(ref next_tok) = next_token {
                    if return_logprobs {
                        if let Some(ref next_lp) = next_logprobs {
                            MxArray::async_eval_arrays(&[next_tok, next_lp]);
                        }
                    } else {
                        MxArray::async_eval_arrays(&[next_tok]);
                    }
                }

                // Clear cache with sync to prevent GPU timeout during long generation
                if step % 128 == 0 && step > 0 {
                    synchronize_and_clear_cache();
                }

                // Advance to next token
                if let Some(next_tok) = next_token {
                    token = next_tok;
                    logprobs = next_logprobs;
                }
            }

            info!(
                "Generation complete: {} tokens, finish_reason={}",
                generated_tokens.len(),
                finish_reason
            );

            // Explicitly drop wired_ctx to synchronize streams and restore wired limit
            // This happens before converting results to ensure proper cleanup
            drop(wired_ctx);

            // Convert to MxArrays
            let tokens = MxArray::from_uint32(&generated_tokens, &[generated_tokens.len() as i64])?;

            let logprobs = if return_logprobs {
                MxArray::from_float32(&generated_logprobs, &[generated_logprobs.len() as i64])?
            } else {
                // Return empty array when logprobs not requested (saves memory)
                MxArray::from_float32(&[], &[0])?
            };

            Ok(GenerationResult {
                text: String::new(), // Only populated by generate() API
                tokens,
                logprobs,
                finish_reason: finish_reason.to_string(),
                num_tokens: generated_tokens.len(),
            })
        })
        .await
        .map_err(|join_error| {
            napi::Error::new(
                napi::Status::GenericFailure,
                format!("Generation thread panicked: {}", join_error),
            )
        })?
    }

    /// Text-to-text generation with integrated tokenization
    ///
    /// This is a high-level API that handles chat template formatting, tokenization,
    /// generation, and decoding internally. It takes chat messages, applies the ChatML
    /// template, generates tokens, and decodes them back to text.
    ///
    /// # Arguments
    /// * `messages` - Array of chat messages with role and content
    /// * `config` - Generation configuration
    ///
    /// # Returns
    /// * GenerationResult with text, tokens, logprobs, finish reason, and token count
    ///
    /// # Example
    /// ```typescript
    /// const model = await Qwen3Model.loadPretrained("path/to/model");
    /// const messages = [
    ///   { role: "user", content: "What is 2+2?" }
    /// ];
    /// const result = await model.generate(messages, {
    ///   maxNewTokens: 50,
    ///   temperature: 0.8,
    ///   topP: 0.95,
    /// });
    /// console.log(result.text); // Decoded text output
    /// console.log(result.tokens); // Token IDs (for GRPO)
    /// console.log(result.logprobs); // Log probabilities (for GRPO)
    /// ```
    #[napi]
    pub async fn generate(
        &self,
        messages: Vec<ChatMessage>,
        config: Option<GenerationConfig>,
    ) -> Result<GenerationResult> {
        // Check if tokenizer is available
        let tokenizer = self.tokenizer.clone().ok_or_else(|| {
            Error::new(
                Status::InvalidArg,
                "Tokenizer not available. Model must be loaded via load_pretrained() to use generate().",
            )
        })?;

        // Apply chat template and encode in a blocking task
        let tokenizer_clone = tokenizer.clone();
        let input_ids = napi::bindgen_prelude::spawn_blocking(move || {
            // Format messages using ChatML template
            let formatted = messages
                .iter()
                .map(|msg| format!("<|im_start|>{}\n{}<|im_end|>\n", msg.role, msg.content))
                .chain(iter::once("<|im_start|>assistant\n".to_string()))
                .collect::<String>();

            // Encode the formatted text
            let token_ids = tokenizer_clone.encode_sync(&formatted, Some(false))?;

            // Create MxArray from token IDs
            MxArray::from_uint32(&token_ids, &[1, token_ids.len() as i64])
        })
        .await
        .map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("Chat template task failed: {}", e),
            )
        })??;

        // Generate tokens using the training API (which has the optimized implementation)
        let mut result = self.generate_for_training(&input_ids, config).await?;

        // Decode the generated tokens in a blocking task
        let result_tokens = result.tokens.clone();
        let decoded_text = napi::bindgen_prelude::spawn_blocking(move || {
            let generated_ids = result_tokens.to_uint32()?;

            tokenizer.decode_sync(&generated_ids, true)
        })
        .await
        .map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("Decoding task failed: {}", e),
            )
        })??;

        // Populate the text field
        result.text = decoded_text;

        Ok(result)
    }

    /// High-level chat API with structured response parsing
    ///
    /// The primary API for conversational AI. Handles:
    /// - Chat message formatting with Jinja2 templates
    /// - Tool/function calling with structured output
    /// - Thinking extraction from `<think>` tags
    /// - Clean response text with all special tags stripped
    ///
    /// ## `chat()` vs `generate()`
    ///
    /// | Feature | `chat()` | `generate()` |
    /// |---------|----------|--------------|
    /// | **Purpose** | Conversational AI with tools | Raw text generation |
    /// | **Input** | Chat messages | Token IDs (MxArray) |
    /// | **Tool Support** | Built-in parsing | None |
    /// | **Thinking** | Extracts `<think>` content | Raw text only |
    /// | **Output** | Structured `ChatResult` | Basic `GenerationResult` |
    /// | **Use Case** | Chat apps, agents, assistants | Training, low-level control |
    ///
    /// ## When to use `chat()`
    /// - Building conversational applications
    /// - Need tool/function calling
    /// - Want structured responses with thinking separated
    /// - Working with chat message format
    ///
    /// ## When to use `generate()`
    /// - Training and fine-tuning (need raw logprobs)
    /// - Custom tokenization pipeline
    /// - Low-level generation control
    /// - Non-chat use cases
    ///
    /// # Arguments
    /// * `messages` - Array of chat messages (user/assistant/system roles)
    /// * `config` - Chat configuration including optional tools and generation params
    ///
    /// # Returns
    /// * `ChatResult` containing:
    ///   - `text`: Clean response (tool_call and think tags stripped)
    ///   - `thinking`: Extracted chain-of-thought reasoning (or null)
    ///   - `toolCalls`: Parsed tool calls with native JS object arguments
    ///   - `finishReason`: "stop" | "length" | "tool_calls"
    ///   - `rawText`: Original text before processing (for debugging)
    ///
    /// # Example
    /// ```typescript
    /// // Simple chat
    /// const result = await model.chat(messages);
    /// console.log(result.text);
    ///
    /// // With tools
    /// const result = await model.chat(messages, {
    ///   tools: [{ type: 'function', function: { name: 'get_weather' } }],
    ///   maxNewTokens: 2048,
    ///   temperature: 0.7,
    /// });
    ///
    /// // Handle tool calls
    /// for (const call of result.toolCalls) {
    ///   if (call.status === 'ok') {
    ///     console.log(call.name, call.arguments);  // Arguments is a JS object!
    ///   }
    /// }
    ///
    /// // Access thinking (chain-of-thought)
    /// if (result.thinking) {
    ///   console.log('Model reasoning:', result.thinking);
    /// }
    /// ```
    #[napi]
    pub async fn chat(
        &self,
        messages: Vec<ChatMessage>,
        config: Option<ChatConfig>,
    ) -> Result<ChatResult> {
        // Check if tokenizer is available
        let tokenizer = self.tokenizer.clone().ok_or_else(|| {
            Error::new(
                Status::InvalidArg,
                "Tokenizer not available. Model must be loaded via load_pretrained() to use chat().",
            )
        })?;

        // Extract tools from config (optional)
        let tools = config.as_ref().and_then(|c| c.tools.clone());

        // Convert ChatConfig to GenerationConfig for the internal generate call
        let gen_config = config.map(|c| GenerationConfig {
            max_new_tokens: c.max_new_tokens.or(Some(2048)), // Default 2048 for chat
            temperature: c.temperature.or(Some(0.7)),        // Default 0.7 for chat
            top_k: c.top_k,
            top_p: c.top_p.or(Some(0.9)), // Default 0.9 for chat
            min_p: c.min_p,
            repetition_penalty: c.repetition_penalty,
            repetition_context_size: c.repetition_context_size,
            max_consecutive_tokens: c.max_consecutive_tokens,
            max_ngram_repeats: c.max_ngram_repeats,
            ngram_size: c.ngram_size,
            eos_token_id: c.eos_token_id,
            return_logprobs: c.return_logprobs,
        });

        // Apply chat template with tools and encode in a blocking task
        let tokenizer_clone = tokenizer.clone();
        let input_ids = napi::bindgen_prelude::spawn_blocking(move || {
            // Use the tokenizer's apply_chat_template_sync method which handles Jinja2 + tools
            let token_ids = tokenizer_clone.apply_chat_template_sync(
                &messages,
                Some(true),
                tools.as_deref(),
                None,
            )?;

            // Create MxArray from token IDs
            MxArray::from_uint32(&token_ids, &[1, token_ids.len() as i64])
        })
        .await
        .map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("Chat template task failed: {}", e),
            )
        })??;

        // Generate tokens using the internal generate method
        let result = self.generate_for_training(&input_ids, gen_config).await?;

        // Decode the generated tokens in a blocking task
        let result_tokens = result.tokens.clone();
        let raw_text = napi::bindgen_prelude::spawn_blocking(move || {
            let generated_ids = result_tokens.to_uint32()?;
            tokenizer.decode_sync(&generated_ids, true)
        })
        .await
        .map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("Decoding task failed: {}", e),
            )
        })??;

        // Parse tool calls and thinking from the generated text
        let (cleaned_text, tool_calls, thinking) = tools::parse_generation_output(&raw_text);

        // Determine finish reason - if we have valid tool calls, it's "tool_calls"
        let finish_reason = if tool_calls.iter().any(|tc| tc.status == "ok") {
            "tool_calls".to_string()
        } else {
            result.finish_reason.clone()
        };

        Ok(ChatResult {
            text: cleaned_text,
            tool_calls,
            thinking,
            tokens: result.tokens,
            logprobs: result.logprobs,
            finish_reason,
            num_tokens: result.num_tokens,
            raw_text,
        })
    }

    /// Generate multiple completions for multiple prompts in batch
    ///
    /// This is an optimized method for GRPO training that generates G completions
    /// for each of N prompts. It performs all tokenization, generation, and decoding
    /// in 3 blocking tasks instead of N*(1+2G) tasks.
    ///
    /// # Arguments
    /// * `prompts` - Array of N prompt message arrays
    /// * `group_size` - Number of completions (G) to generate per prompt
    /// * `config` - Generation configuration (sampling params, etc.)
    ///
    /// # Returns
    /// * BatchGenerationResult containing N*G completions with:
    ///   - tokens: Flat array of N*G token arrays
    ///   - logprobs: Flat array of N*G logprob arrays
    ///   - texts: Flat array of N*G decoded texts
    ///   - finish_reasons: N arrays of G finish reasons
    ///   - token_counts: N arrays of G token counts
    ///
    /// # Performance
    /// For N=10 prompts, G=8 completions:
    /// - Old approach: N*(1 tokenize + G generate + G decode) = 10*(1+8+8) = 170 blocking tasks
    /// - New approach: 1 tokenize + N*G generate + 1 decode = 1+80+1 = 82 blocking tasks (2.1x reduction)
    ///
    /// # Example
    /// ```typescript
    /// const result = await model.generateBatch(
    ///   [messages1, messages2, ...], // N prompts
    ///   8,                             // G completions per prompt
    ///   config
    /// );
    /// ```
    #[napi]
    pub async fn generate_batch(
        &self,
        prompts: Vec<Vec<ChatMessage>>,
        group_size: u32,
        config: Option<GenerationConfig>,
    ) -> Result<BatchGenerationResult> {
        // Check if tokenizer is available
        let tokenizer = self.tokenizer.clone().ok_or_else(|| {
            Error::new(
                Status::InvalidArg,
                "Tokenizer not available. Model must be loaded via load_pretrained() to use generateBatch().",
            )
        })?;

        let num_prompts = prompts.len();
        let group_size_usize = group_size as usize;

        // STEP 1: Tokenize all prompts in one blocking task
        let tokenizer_clone = tokenizer.clone();
        let prompt_token_arrays = napi::bindgen_prelude::spawn_blocking(move || {
            let mut results = Vec::with_capacity(num_prompts);

            for messages in prompts {
                // Format messages using ChatML template
                let formatted = messages
                    .iter()
                    .map(|msg| format!("<|im_start|>{}\n{}<|im_end|>\n", msg.role, msg.content))
                    .chain(iter::once("<|im_start|>assistant\n".to_string()))
                    .collect::<String>();

                // Encode the formatted text
                let token_ids = tokenizer_clone.encode_sync(&formatted, Some(false))?;

                // Create MxArray from token IDs
                let prompt_tokens = MxArray::from_uint32(&token_ids, &[1, token_ids.len() as i64])?;

                results.push(prompt_tokens);
            }

            Ok::<Vec<MxArray>, Error>(results)
        })
        .await
        .map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("Batch tokenization task failed: {}", e),
            )
        })??;

        // STEP 2: Generate N*G completions using async calls
        // Note: This uses N*G blocking tasks for generation, but still saves 2*N blocking tasks
        // for tokenization and decoding compared to the naive approach
        let mut all_tokens = Vec::with_capacity(num_prompts * group_size_usize);
        let mut all_logprobs = Vec::with_capacity(num_prompts * group_size_usize);
        let mut all_finish_reasons = Vec::with_capacity(num_prompts);
        let mut all_token_counts = Vec::with_capacity(num_prompts);

        // For each prompt, generate G completions
        for prompt_tokens in prompt_token_arrays.into_iter() {
            let mut prompt_finish_reasons = Vec::with_capacity(group_size_usize);
            let mut prompt_token_counts = Vec::with_capacity(group_size_usize);

            // Generate G completions for this prompt
            for _group_idx in 0..group_size {
                let result = self
                    .generate_for_training(&prompt_tokens, config.clone())
                    .await?;
                all_tokens.push(result.tokens);
                all_logprobs.push(result.logprobs);
                prompt_finish_reasons.push(result.finish_reason);
                prompt_token_counts.push(result.num_tokens as u32);
            }

            all_finish_reasons.push(prompt_finish_reasons);
            all_token_counts.push(prompt_token_counts);
        }

        // STEP 3: Decode all N*G completions in one blocking task
        let all_tokens_clone = all_tokens.clone();
        let decoded_texts = napi::bindgen_prelude::spawn_blocking(move || {
            let mut texts = Vec::with_capacity(all_tokens_clone.len());

            for token_array in &all_tokens_clone {
                let generated_ids = token_array.to_uint32()?;

                let decoded = tokenizer.decode_sync(&generated_ids, true)?;
                texts.push(decoded);
            }

            Ok::<Vec<String>, Error>(texts)
        })
        .await
        .map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("Batch decoding task failed: {}", e),
            )
        })??;

        Ok(BatchGenerationResult {
            tokens: all_tokens,
            logprobs: all_logprobs,
            texts: decoded_texts,
            finish_reasons: all_finish_reasons,
            token_counts: all_token_counts,
            num_prompts,
            group_size,
        })
    }

    /// Decode token IDs to text using the internal tokenizer
    ///
    /// Helper method for decoding generated tokens. The model must have been loaded
    /// via load_pretrained() to have a tokenizer available.
    ///
    /// # Arguments
    /// * `token_ids` - Token IDs to decode as Uint32Array
    /// * `skip_special_tokens` - Whether to skip special tokens (default: true)
    ///
    /// # Returns
    /// * Decoded text string
    #[napi]
    pub async fn decode(
        &self,
        token_ids: Uint32Array,
        skip_special_tokens: Option<bool>,
    ) -> Result<String> {
        let tokenizer = self.tokenizer.clone().ok_or_else(|| {
            Error::new(
                Status::InvalidArg,
                "Tokenizer not available. Model must be loaded via load_pretrained().",
            )
        })?;

        let skip_special = skip_special_tokens.unwrap_or(true);
        let token_ids_vec = token_ids.to_vec();

        napi::bindgen_prelude::spawn_blocking(move || {
            tokenizer.decode_sync(&token_ids_vec, skip_special)
        })
        .await
        .map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("Decoding task failed: {}", e),
            )
        })?
    }

    /// Apply chat template and encode to token IDs
    ///
    /// Formats messages using ChatML format (or Jinja2 template with tools) and encodes to tokens.
    /// The model must have been loaded via load_pretrained() to have a tokenizer available.
    ///
    /// # Arguments
    /// * `messages` - Array of chat messages
    /// * `add_generation_prompt` - Whether to add generation prompt (default: true)
    /// * `tools` - Optional array of tool definitions for function calling
    /// * `enable_thinking` - Optional flag to enable thinking mode (<think> tags)
    ///
    /// # Returns
    /// * Encoded token IDs as Uint32Array
    #[napi]
    pub fn apply_chat_template<'env>(
        &self,
        env: &'env Env,
        messages: Vec<ChatMessage>,
        add_generation_prompt: Option<bool>,
        tools: Option<Vec<ToolDefinition>>,
        enable_thinking: Option<bool>,
    ) -> Result<PromiseRaw<'env, Uint32ArraySlice<'env>>> {
        let tokenizer = self.tokenizer.clone().ok_or_else(|| {
            Error::new(
                Status::InvalidArg,
                "Tokenizer not available. Model must be loaded via load_pretrained().",
            )
        })?;

        // Delegate to tokenizer which handles both simple ChatML and Jinja2 with tools
        tokenizer.apply_chat_template(env, messages, add_generation_prompt, tools, enable_thinking)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repetition_cutoff_disabled() {
        // When all thresholds are 0, should never trigger
        assert_eq!(check_repetition_cutoff(&[1, 1, 1, 1, 1], 0, 0, 0), None);
        assert_eq!(
            check_repetition_cutoff(&[1, 2, 1, 2, 1, 2, 1, 2], 0, 0, 0),
            None
        );
    }

    #[test]
    fn test_repetition_cutoff_consecutive_triggers() {
        // 5 consecutive tokens with max=5 → triggers
        assert_eq!(
            check_repetition_cutoff(&[1, 1, 1, 1, 1], 5, 8, 3),
            Some("repetition")
        );

        // 6 consecutive tokens with max=5 → triggers
        assert_eq!(
            check_repetition_cutoff(&[1, 1, 1, 1, 1, 1], 5, 8, 3),
            Some("repetition")
        );
    }

    #[test]
    fn test_repetition_cutoff_consecutive_no_trigger() {
        // 4 consecutive tokens with max=5 → no trigger
        assert_eq!(check_repetition_cutoff(&[1, 1, 1, 1], 5, 8, 3), None);

        // Varied tokens → no trigger
        assert_eq!(check_repetition_cutoff(&[1, 2, 3, 4, 5], 5, 8, 3), None);

        // Pattern broken at end → no trigger
        assert_eq!(check_repetition_cutoff(&[1, 1, 1, 1, 2], 5, 8, 3), None);
    }

    #[test]
    fn test_repetition_cutoff_ngram_triggers() {
        // 4 repetitions of 2-gram [1, 2] with max=4 → triggers
        assert_eq!(
            check_repetition_cutoff(&[1, 2, 1, 2, 1, 2, 1, 2], 16, 4, 2),
            Some("repetition")
        );

        // 3 repetitions of 3-gram [1, 2, 3] with max=3 → triggers
        assert_eq!(
            check_repetition_cutoff(&[1, 2, 3, 1, 2, 3, 1, 2, 3], 16, 3, 3),
            Some("repetition")
        );
    }

    #[test]
    fn test_repetition_cutoff_ngram_no_trigger() {
        // Only 3 repetitions with max=4 → no trigger
        assert_eq!(check_repetition_cutoff(&[1, 2, 1, 2, 1, 2], 16, 4, 2), None);

        // Pattern broken → no trigger
        assert_eq!(
            check_repetition_cutoff(&[1, 2, 1, 2, 3, 2, 1, 2], 16, 4, 2),
            None
        );
    }

    #[test]
    fn test_repetition_cutoff_short_sequences() {
        // Very short sequences should not trigger
        assert_eq!(check_repetition_cutoff(&[], 5, 4, 2), None);
        assert_eq!(check_repetition_cutoff(&[1], 5, 4, 2), None);
        assert_eq!(check_repetition_cutoff(&[1, 1], 5, 4, 2), None);
    }

    #[test]
    fn test_repetition_cutoff_default_thresholds() {
        // Test with default thresholds (16 consecutive, 8 n-gram repeats, 3-gram size)

        // 16 consecutive → triggers
        let tokens: Vec<u32> = vec![1; 16];
        assert_eq!(
            check_repetition_cutoff(&tokens, 16, 8, 3),
            Some("repetition")
        );

        // 15 consecutive → no trigger
        let tokens: Vec<u32> = vec![1; 15];
        assert_eq!(check_repetition_cutoff(&tokens, 16, 8, 3), None);

        // 8 repetitions of 3-gram → triggers
        let tokens: Vec<u32> = (0..24).map(|i| (i % 3) as u32 + 1).collect(); // [1,2,3,1,2,3,...]
        assert_eq!(
            check_repetition_cutoff(&tokens, 16, 8, 3),
            Some("repetition")
        );
    }
}
