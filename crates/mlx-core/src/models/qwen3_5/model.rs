use std::sync::{Arc, RwLock};

use napi::bindgen_prelude::*;
use napi_derive::napi;
use tracing::{info, warn};

use crate::array::MxArray;
use crate::array::mask::create_causal_mask;
use crate::nn::{Embedding, Linear, RMSNorm};
use crate::sampling::{SamplingConfig, sample};
use crate::tokenizer::{ChatMessage, Qwen3Tokenizer, ToolDefinition};
use crate::tools;

use super::config::Qwen3_5Config;
use super::decoder_layer::DecoderLayer;
use super::layer_cache::Qwen3_5LayerCache;
use super::persistence;

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
    #[napi(ts_type = "Array<ToolDefinition>")]
    pub tools: Option<Vec<ToolDefinition>>,
}

/// Chat result
#[napi(object)]
#[derive(Debug, Clone)]
pub struct Qwen3_5ChatResult {
    pub text: String,
    pub thinking: Option<String>,
    pub num_tokens: u32,
    pub finish_reason: String,
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
            "Qwen3.5 model created: {} layers, fa_idx={}, moe={}",
            config.num_layers,
            fa_idx,
            config.is_moe()
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

            // Prefill: forward pass on entire prompt
            let logits = forward_with_locks(
                &prompt_tokens,
                &embedding_weight,
                &layers_arc,
                &final_norm_arc,
                &lm_head_arc,
                &caches_arc,
                fa_idx,
            )?;

            // Get last token logits: [1, vocab]
            let seq_len = logits.shape_at(1)?;
            let last_logits = logits.slice_axis(1, seq_len - 1, seq_len)?;
            let last_logits = last_logits.squeeze(Some(&[1]))?; // [1, vocab]

            // Sample first token
            let next_token = sample(&last_logits, sampling_config)?;
            let token_data = next_token.to_int32()?;
            let mut token_id = *token_data.first().ok_or_else(|| {
                Error::from_reason("Sampling returned empty token array - logits may contain NaN")
            })? as u32;
            generated_tokens.push(token_id);

            if token_id == eos_id {
                finish_reason = String::from("eos");
            }

            // Decode loop
            for _ in 1..max_tokens {
                if finish_reason == "eos" {
                    break;
                }

                // Forward single token
                let input = MxArray::from_int32(&[token_id as i32], &[1, 1])?;
                let logits = forward_with_locks(
                    &input,
                    &embedding_weight,
                    &layers_arc,
                    &final_norm_arc,
                    &lm_head_arc,
                    &caches_arc,
                    fa_idx,
                )?;
                let logits = logits.squeeze(Some(&[1]))?; // [1, vocab]

                // Sample
                let token_arr = sample(&logits, sampling_config)?;
                let token_data = token_arr.to_int32()?;
                let token_id_new = *token_data.first().ok_or_else(|| {
                    Error::from_reason(
                        "Sampling returned empty token array - logits may contain NaN",
                    )
                })? as u32;
                generated_tokens.push(token_id_new);

                if token_id_new == eos_id {
                    finish_reason = String::from("eos");
                }

                token_id = token_id_new;
            }

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
            let sampling_config = Some(SamplingConfig {
                temperature: config.temperature,
                top_k: config.top_k,
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

            // Prefill
            let logits = forward_with_locks(
                &prompt,
                &embedding_weight,
                &layers_arc,
                &final_norm_arc,
                &lm_head_arc,
                &caches_arc,
                fa_idx,
            )?;

            let seq_len = logits.shape_at(1)?;
            let last_logits = logits.slice_axis(1, seq_len - 1, seq_len)?;
            let last_logits = last_logits.squeeze(Some(&[1]))?;

            let next_token = sample(&last_logits, sampling_config)?;
            let token_data = next_token.to_int32()?;
            let mut token_id = *token_data.first().ok_or_else(|| {
                Error::from_reason("Sampling returned empty token array - logits may contain NaN")
            })? as u32;
            generated_tokens.push(token_id);

            if token_id == eos_id {
                finish_reason = String::from("eos");
            }

            // Decode loop
            for _ in 1..max_new_tokens {
                if finish_reason == "eos" {
                    break;
                }

                let input = MxArray::from_int32(&[token_id as i32], &[1, 1])?;
                let logits = forward_with_locks(
                    &input,
                    &embedding_weight,
                    &layers_arc,
                    &final_norm_arc,
                    &lm_head_arc,
                    &caches_arc,
                    fa_idx,
                )?;
                let logits = logits.squeeze(Some(&[1]))?;

                let token_arr = sample(&logits, sampling_config)?;
                let token_data = token_arr.to_int32()?;
                let token_id_new = *token_data.first().ok_or_else(|| {
                    Error::from_reason(
                        "Sampling returned empty token array - logits may contain NaN",
                    )
                })? as u32;
                generated_tokens.push(token_id_new);

                if token_id_new == eos_id {
                    finish_reason = String::from("eos");
                }

                token_id = token_id_new;
            }

            // Decode text
            let text = tokenizer_for_decode
                .decode_sync(&generated_tokens, true)
                .unwrap_or_else(|e| {
                    warn!("Failed to decode generated tokens: {}", e);
                    String::new()
                });

            let num_tokens = generated_tokens.len() as u32;

            // Extract thinking and clean text
            let (clean_text, thinking) = tools::parse_thinking(&text);

            Ok(Qwen3_5ChatResult {
                text: clean_text,
                thinking,
                num_tokens,
                finish_reason,
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

        // MoE params
        let num_experts = self.config.num_experts.unwrap_or(0) as i64;
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
                // Full attention layer parameters
                // Note: assumes num_heads * head_dim == hidden_size (true for standard configs)
                total += h * h * 2 // q_proj (2x for gate)
                    + h * (self.config.num_kv_heads as i64 * self.config.head_dim as i64) * 2 // k, v
                    + h * h; // o_proj
            }

            // MLP params
            if is_moe {
                total += h * num_experts // router gate
                    + num_experts * 3 * h * moe_i // expert gate/up/down projections
                    + 3 * h * shared_i // shared expert
                    + h; // shared expert gate
            } else {
                total += 3 * h * dense_i; // dense MLP gate/up/down
            }

            // Norms (2 per layer)
            total += h * 2;
        }

        total
    }
}

/// Forward pass through the model, acquiring all necessary locks.
///
/// This is a free function (not a method) so it can be called from
/// within spawn_blocking closures that have cloned the Arc fields.
fn forward_with_locks(
    input_ids: &MxArray,
    embedding_weight: &MxArray,
    layers_arc: &Arc<RwLock<Vec<DecoderLayer>>>,
    final_norm_arc: &Arc<RwLock<RMSNorm>>,
    lm_head_arc: &Arc<RwLock<Option<Linear>>>,
    caches_arc: &Arc<RwLock<Option<Vec<Qwen3_5LayerCache>>>>,
    fa_idx: usize,
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

    let ssm_mask = {
        let batch = hidden_states.shape_at(0)?;
        let mask = MxArray::ones(&[batch, seq_len], Some(hidden_states.dtype()?))?;
        Some(mask)
    };

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
            // tie_word_embeddings: use embedding weight as linear
            let weight_t = embedding_weight.transpose(Some(&[1, 0]))?;
            h.matmul(&weight_t)
        }
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
    /// Currently returns an all-ones mask (no masking applied).
    /// TODO: Implement left-padding support for batched generation.
    fn create_ssm_mask(&self, hidden_states: &MxArray) -> Result<Option<MxArray>> {
        let batch = hidden_states.shape_at(0)?;
        let seq_len = hidden_states.shape_at(1)?;

        // For now, return all-ones mask (no masking)
        // Use hidden_states' dtype to avoid f32 promotion for bf16/f16 models
        // TODO: Support left-padding mask for batched generation
        let mask = MxArray::ones(&[batch, seq_len], Some(hidden_states.dtype()?))?;
        Ok(Some(mask))
    }
}
