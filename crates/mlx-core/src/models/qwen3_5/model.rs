use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi_derive::napi;
use tracing::info;

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
    #[napi(ts_type = "object[] | undefined")]
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

/// Qwen3.5 Model — hybrid linear/full attention with optional MoE.
///
/// Inference-only implementation. Supports both dense and MoE variants.
#[napi]
pub struct Qwen3_5Model {
    config: Qwen3_5Config,
    pub(crate) embedding: Embedding,
    pub(crate) layers: Vec<DecoderLayer>,
    pub(crate) final_norm: RMSNorm,
    pub(crate) lm_head: Option<Linear>, // None when tie_word_embeddings
    caches: Option<Vec<Qwen3_5LayerCache>>,
    pub(crate) tokenizer: Option<Arc<Qwen3Tokenizer>>,
    #[allow(dead_code)]
    ssm_idx: usize,  // Index of first linear attention layer
    fa_idx: usize,   // Index of first full attention layer
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

        // Find first linear and first full attention layer indices
        let ssm_idx = (0..config.num_layers as usize)
            .find(|&i| config.is_linear_layer(i))
            .unwrap_or(0);
        let fa_idx = (0..config.num_layers as usize)
            .find(|&i| !config.is_linear_layer(i))
            .unwrap_or(config.full_attention_interval as usize - 1);

        info!(
            "Qwen3.5 model created: {} layers, ssm_idx={}, fa_idx={}, moe={}",
            config.num_layers, ssm_idx, fa_idx, config.is_moe()
        );

        Ok(Self {
            config,
            embedding,
            layers,
            final_norm,
            lm_head,
            caches: None,
            tokenizer: None,
            ssm_idx,
            fa_idx,
        })
    }

    /// Initialize caches for incremental generation.
    #[napi]
    pub fn init_caches(&mut self) {
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
    }

    /// Reset all caches.
    #[napi]
    pub fn reset_caches(&mut self) {
        if let Some(ref mut caches) = self.caches {
            for cache in caches.iter_mut() {
                cache.reset();
            }
        }
        self.caches = None;
    }

    /// Forward pass through the model.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs [B, T]
    ///
    /// # Returns
    /// Logits [B, T, vocab_size]
    #[napi]
    pub fn forward(&mut self, input_ids: &MxArray) -> Result<MxArray> {
        let hidden_states = self.embedding.forward(input_ids)?;
        self.forward_from_embeddings(&hidden_states)
    }

    /// Forward pass with cache for incremental generation.
    #[napi]
    pub fn forward_with_cache(&mut self, input_ids: &MxArray) -> Result<MxArray> {
        if self.caches.is_none() {
            self.init_caches();
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
    #[napi]
    pub fn generate(
        &mut self,
        prompt_tokens: &MxArray,
        config: Qwen3_5GenerationConfig,
    ) -> Result<Qwen3_5GenerationResult> {
        self.reset_caches();
        self.init_caches();

        let max_tokens = config.max_new_tokens;
        let sampling_config = Some(SamplingConfig {
            temperature: config.temperature,
            top_k: config.top_k,
            top_p: config.top_p,
            min_p: config.min_p,
        });

        let eos_id = self.config.eos_token_id as u32;
        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut finish_reason = String::from("length");

        // Prefill: forward pass on entire prompt
        let logits = self.forward_with_cache(prompt_tokens)?;

        // Get last token logits: [1, vocab]
        let seq_len = logits.shape_at(1)?;
        let last_logits = logits.slice_axis(1, seq_len - 1, seq_len)?;
        let last_logits = last_logits.squeeze(Some(&[1]))?; // [1, vocab]

        // Sample first token
        let next_token = sample(&last_logits, sampling_config.clone())?;
        let mut token_id = next_token.to_int32()?[0] as u32;
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
            let logits = self.forward_with_cache(&input)?;
            let logits = logits.squeeze(Some(&[1]))?; // [1, vocab]

            // Sample
            let token_arr = sample(&logits, sampling_config.clone())?;
            let token_id_new = token_arr.to_int32()?[0] as u32;
            generated_tokens.push(token_id_new);

            if token_id_new == eos_id {
                finish_reason = String::from("eos");
            }

            token_id = token_id_new;
        }

        // Decode text if tokenizer available
        let text = if let Some(ref tok) = self.tokenizer {
            tok.decode_sync(&generated_tokens, true)
                .unwrap_or_default()
        } else {
            String::new()
        };

        let num_tokens = generated_tokens.len() as u32;

        Ok(Qwen3_5GenerationResult {
            tokens: generated_tokens,
            text,
            num_tokens,
            finish_reason,
        })
    }

    /// Chat API with tool calling support.
    #[napi]
    pub fn chat(
        &mut self,
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
            .as_ref()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))?;

        let tool_defs = config.tools.as_deref();
        let tokens = tokenizer.apply_chat_template_sync(
            &messages,
            Some(true),
            tool_defs,
            None,
        )?;

        // Create prompt tensor
        let prompt = MxArray::from_uint32(&tokens, &[1, tokens.len() as i64])?;

        // Generate
        let gen_config = Qwen3_5GenerationConfig {
            max_new_tokens: config.max_new_tokens.unwrap_or(2048),
            temperature: config.temperature,
            top_k: config.top_k,
            top_p: config.top_p,
            min_p: config.min_p,
        };

        let result = self.generate(&prompt, gen_config)?;

        // Extract thinking and clean text
        let (clean_text, thinking) = tools::parse_thinking(&result.text);

        Ok(Qwen3_5ChatResult {
            text: clean_text,
            thinking,
            num_tokens: result.num_tokens,
            finish_reason: result.finish_reason,
        })
    }

    /// Get the number of parameters in the model.
    #[napi]
    pub fn num_parameters(&self) -> i64 {
        // Approximate based on config
        let h = self.config.hidden_size as i64;
        let v = self.config.vocab_size as i64;
        let n = self.config.num_layers as i64;
        let i = self.config.intermediate_size as i64;

        // Embedding + LM head
        let mut total = v * h;
        if !self.config.tie_word_embeddings {
            total += v * h;
        }

        // Per-layer params (rough estimate)
        // Full attention layers
        let fa_layers = n / self.config.full_attention_interval as i64;
        let linear_layers = n - fa_layers;

        // Full attention: q(2x), k, v, o projections + norms + MLP
        let fa_params = h * h * 2 // q_proj (2x for gate)
            + h * (self.config.num_kv_heads as i64 * self.config.head_dim as i64) * 2 // k, v
            + h * h // o_proj
            + 3 * h * i // MLP
            + h * 4; // norms
        total += fa_layers * fa_params;

        // Linear attention: projections + conv + delta params + MLP
        let kd = self.config.linear_key_dim() as i64;
        let vd = self.config.linear_value_dim() as i64;
        let la_params = h * (kd * 2 + vd * 2) // in_proj_qkvz
            + h * (self.config.linear_num_value_heads as i64 * 2) // in_proj_ba
            + (kd + vd) * self.config.linear_conv_kernel_dim as i64 // conv1d
            + vd * h // out_proj
            + 3 * h * i // MLP
            + h * 4; // norms + misc
        total += linear_layers * la_params;

        total
    }
}

// Internal methods (not NAPI-exported)
impl Qwen3_5Model {
    /// Forward pass from embeddings through all layers.
    fn forward_from_embeddings(&mut self, hidden_states: &MxArray) -> Result<MxArray> {
        let mut h = hidden_states.clone();

        // Create masks
        let fa_mask = self.create_fa_mask(&h)?;
        let ssm_mask = self.create_ssm_mask(&h)?;

        // Forward through layers
        let num_layers = self.layers.len();
        for i in 0..num_layers {
            let mask = if self.layers[i].is_linear {
                ssm_mask.as_ref()
            } else {
                fa_mask.as_ref()
            };

            let cache = self.caches.as_mut().map(|c| &mut c[i]);
            h = self.layers[i].forward(&h, mask, cache)?;
        }

        // Final norm
        let h = self.final_norm.forward(&h)?;

        // LM head
        match &self.lm_head {
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
    fn create_fa_mask(&self, hidden_states: &MxArray) -> Result<Option<MxArray>> {
        let seq_len = hidden_states.shape_at(1)?;
        if seq_len <= 1 && self.caches.is_some() {
            // Single-token decode step with cache — no mask needed
            return Ok(None);
        }

        // Get cache offset from the first full attention layer
        let offset = self
            .caches
            .as_ref()
            .map(|c| c[self.fa_idx].offset())
            .unwrap_or(0);

        // Create causal mask using existing utility
        create_causal_mask(
            seq_len as i32,
            Some(offset as i32),
            None,  // no sliding window
        )
        .map(Some)
    }

    /// Create boolean mask for linear attention (SSM) layers.
    ///
    /// For SSM layers, the mask is simpler — just ones for valid positions.
    fn create_ssm_mask(&self, hidden_states: &MxArray) -> Result<Option<MxArray>> {
        let batch = hidden_states.shape_at(0)?;
        let seq_len = hidden_states.shape_at(1)?;

        // For now, return all-ones mask (no masking)
        // TODO: Support left-padding mask for batched generation
        let mask = MxArray::ones(&[batch, seq_len], None)?;
        Ok(Some(mask))
    }
}
