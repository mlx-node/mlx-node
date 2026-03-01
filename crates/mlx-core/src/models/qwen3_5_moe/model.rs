use std::sync::{Arc, RwLock};

use napi::bindgen_prelude::*;
use napi_derive::napi;
use tracing::{info, warn};

use crate::array::MxArray;
use crate::array::mask::create_causal_mask;
use crate::nn::{Embedding, Linear, RMSNorm};
use crate::sampling::{SamplingConfig, sample};
use crate::stream::{DeviceType, Stream, StreamContext};
use crate::tokenizer::{ChatMessage, Qwen3Tokenizer, ToolDefinition};
use crate::tools;

use super::config::Qwen3_5MoeConfig;
use super::decoder_layer::DecoderLayer;
use super::layer_cache::Qwen3_5LayerCache;
use super::persistence;

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
    #[napi(ts_type = "Array<ToolDefinition>")]
    pub tools: Option<Vec<ToolDefinition>>,
}

/// Chat result
#[napi(object)]
#[derive(Debug, Clone)]
pub struct Qwen3_5MoeChatResult {
    pub text: String,
    pub thinking: Option<String>,
    pub num_tokens: u32,
    pub finish_reason: String,
}

/// Qwen3.5 MoE Model -- hybrid linear/full attention with Mixture-of-Experts.
///
/// No compiled C++ forward path — MoE models use the Rust forward_with_locks path
/// since the C++ compiled forward doesn't support sparse expert routing.
#[napi]
pub struct Qwen3_5MoeModel {
    config: Qwen3_5MoeConfig,
    pub(crate) embedding: Embedding,
    pub(crate) layers: Arc<RwLock<Vec<DecoderLayer>>>,
    pub(crate) final_norm: Arc<RwLock<RMSNorm>>,
    pub(crate) lm_head: Arc<RwLock<Option<Linear>>>,
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
            Some(Linear::new(
                config.hidden_size as u32,
                config.vocab_size as u32,
                Some(false),
            )?)
        };

        let fa_idx = (0..config.num_layers as usize)
            .find(|&i| !config.is_linear_layer(i))
            .unwrap_or(0);

        info!(
            "Qwen3.5 MoE model created: {} layers, fa_idx={}, experts={}",
            config.num_layers,
            fa_idx,
            config.num_experts
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

            let embedding_weight_t = embedding_weight.transpose(Some(&[1, 0]))?;
            let generation_stream = Stream::new(DeviceType::Gpu);
            let model_size_bytes = model_config.estimate_memory_bytes() as usize;
            let _wired_ctx =
                crate::stream::WiredLimitContext::new(model_size_bytes, vec![generation_stream]);

            // Prefill
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

            let seq_len = logits.shape_at(1)?;
            let last_logits = logits.slice_axis(1, seq_len - 1, seq_len)?;
            let last_logits = last_logits.squeeze(Some(&[1]))?;

            let mut y = sample(&last_logits, sampling_config)?;
            MxArray::async_eval_arrays(&[&y]);

            // Decode loop: Rust-only path (no compiled C++ for MoE)
            for step in 0..max_tokens {
                let next_y = {
                    let _stream_ctx = StreamContext::new(generation_stream);
                    if step + 1 < max_tokens {
                        let next_ids = y.reshape(&[1, 1])?;
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
                        Some(next_token)
                    } else {
                        None
                    }
                };

                if step == 0 {
                    y.eval();
                }
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
                    crate::array::synchronize_and_clear_cache();
                }
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

        napi::bindgen_prelude::spawn_blocking(move || {
            let tool_defs = config.tools.as_deref();
            let tokens =
                tokenizer.apply_chat_template_sync(&messages, Some(true), tool_defs, None)?;

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

            let embedding_weight_t = embedding_weight.transpose(Some(&[1, 0]))?;
            let generation_stream = Stream::new(DeviceType::Gpu);
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
            let last_logits = last_logits.squeeze(Some(&[1]))?;

            let mut y = sample(&last_logits, sampling_config)?;
            MxArray::async_eval_arrays(&[&y]);

            // Decode loop: Rust-only path
            for step in 0..max_new_tokens {
                let next_y = {
                    let _stream_ctx = StreamContext::new(generation_stream);
                    if step + 1 < max_new_tokens {
                        let next_ids = y.reshape(&[1, 1])?;
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
                        Some(next_token)
                    } else {
                        None
                    }
                };

                if step == 0 {
                    y.eval();
                }
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
                    crate::array::synchronize_and_clear_cache();
                }
            }

            let text = tokenizer_for_decode
                .decode_sync(&generated_tokens, true)
                .unwrap_or_else(|e| {
                    warn!("Failed to decode generated tokens: {}", e);
                    String::new()
                });

            let num_tokens = generated_tokens.len() as u32;
            let (clean_text, thinking) = tools::parse_thinking(&text);

            Ok(Qwen3_5MoeChatResult {
                text: clean_text,
                thinking,
                num_tokens,
                finish_reason,
            })
        })
        .await
        .map_err(|e| Error::from_reason(format!("Chat task failed: {}", e)))?
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
                total += h * h * 2
                    + h * (self.config.num_kv_heads as i64 * d) * 2
                    + h * h
                    + d * 2;
            }

            if is_moe {
                total += h * num_experts
                    + num_experts * 3 * h * moe_i
                    + 3 * h * shared_i
                    + h;
            } else {
                total += 3 * h * dense_i;
            }

            total += h * 2;
        }

        total += h;
        total
    }
}

/// Forward pass through the model, acquiring all necessary locks.
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
    let embedding = Embedding::from_weight(embedding_weight)?;
    let hidden_states = embedding.forward(input_ids)?;

    let mut h = hidden_states.clone();

    let mut layers_guard = layers_arc
        .write()
        .map_err(|_| Error::from_reason("Failed to acquire layers write lock"))?;
    let mut caches_guard = caches_arc
        .write()
        .map_err(|_| Error::from_reason("Failed to acquire caches write lock"))?;

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

    let ssm_mask = if seq_len > 1 {
        let batch = hidden_states.shape_at(0)?;
        let mask = MxArray::ones(&[batch, seq_len], Some(hidden_states.dtype()?))?;
        Some(mask)
    } else {
        None
    };

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

    let final_norm_guard = final_norm_arc
        .read()
        .map_err(|_| Error::from_reason("Failed to acquire final_norm read lock"))?;
    let h = final_norm_guard.forward(&h)?;
    drop(final_norm_guard);

    let lm_head_guard = lm_head_arc
        .read()
        .map_err(|_| Error::from_reason("Failed to acquire lm_head read lock"))?;
    match &*lm_head_guard {
        Some(head) => head.forward(&h),
        None => {
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

/// Evaluate the sampled token AND all cache arrays together.
fn eval_token_and_caches(
    next_token: &MxArray,
    caches_arc: &Arc<RwLock<Option<Vec<Qwen3_5LayerCache>>>>,
) {
    let mut handles: Vec<*mut mlx_sys::mlx_array> = vec![next_token.as_raw_ptr()];

    if let Ok(caches_guard) = caches_arc.read() {
        if let Some(ref caches) = *caches_guard {
            let mut arr_refs: Vec<&MxArray> = Vec::with_capacity(caches.len() * 2);
            for cache in caches.iter() {
                cache.collect_arrays(&mut arr_refs);
            }
            for arr in &arr_refs {
                handles.push(arr.as_raw_ptr());
            }
        }
    }

    unsafe {
        mlx_sys::mlx_async_eval(handles.as_mut_ptr(), handles.len());
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

    fn create_ssm_mask(&self, hidden_states: &MxArray) -> Result<Option<MxArray>> {
        let batch = hidden_states.shape_at(0)?;
        let seq_len = hidden_states.shape_at(1)?;
        let mask = MxArray::ones(&[batch, seq_len], Some(hidden_states.dtype()?))?;
        Ok(Some(mask))
    }
}
