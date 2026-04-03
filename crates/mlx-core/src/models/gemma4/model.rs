use std::collections::HashMap;
use std::sync::Arc;

use napi::Either;
use napi::bindgen_prelude::*;
use napi_derive::napi;
use tokio::sync::RwLock;

use crate::array::MxArray;
use crate::nn::{Embedding, Linear, RMSNorm};
use crate::sampling::{SamplingConfig, sample};
use crate::tokenizer::{ChatMessage, Qwen3Tokenizer};

use super::config::Gemma4Config;
use super::decoder_layer::Gemma4DecoderLayer;
use super::layer_cache::Gemma4LayerCache;

/// Gemma4 generation configuration.
#[napi(object)]
pub struct Gemma4ChatConfig {
    pub max_new_tokens: Option<i32>,
    pub temperature: Option<f64>,
    pub top_k: Option<i32>,
    pub top_p: Option<f64>,
    pub min_p: Option<f64>,
}

/// Gemma4 chat result.
#[napi(object)]
pub struct Gemma4ChatResult {
    pub text: String,
    pub num_tokens: u32,
    pub finish_reason: String,
}

/// PLE (Per-Layer Embeddings) model-level components.
///
/// Provides per-layer token-level information to each decoder layer.
/// Present in E2B (2.3B) and E4B (4.5B) models.
pub(crate) struct PleComponents {
    /// Embedding table: [vocab_size_per_layer_input, num_layers * ple_dim]
    pub embed_tokens_per_layer: Embedding,
    /// Projection: [hidden_size, num_layers * ple_dim]
    pub per_layer_model_projection: Linear,
    /// Norm applied per ple_dim slice: weight shape [ple_dim]
    pub per_layer_projection_norm: RMSNorm,
    /// Scale factor: 2.0^(-0.5) = 1/sqrt(2) for per_layer_input_scale
    pub per_layer_input_scale: f64,
    /// Scale factor: hidden_size^(-0.5) for per_layer_model_projection_scale
    pub per_layer_model_projection_scale: f64,
    /// Dimension of per-layer embeddings
    pub ple_dim: i32,
    /// Number of layers
    pub num_layers: i32,
    /// PLE vocab size (may be smaller than main vocab_size)
    pub vocab_size_per_layer_input: i32,
}

/// Gemma 4 dense language model.
///
/// Supports E2B (2.3B), E4B (4.5B), and 31B variants.
/// Features: hybrid attention (sliding + global), GeGLU MLP, logit softcapping,
/// embedding scaling, and optional per-layer embeddings.
#[napi]
pub struct Gemma4Model {
    config: Gemma4Config,
    pub(crate) embed_tokens: Arc<RwLock<Embedding>>,
    pub(crate) layers: Arc<RwLock<Vec<Gemma4DecoderLayer>>>,
    pub(crate) final_norm: Arc<RwLock<RMSNorm>>,
    pub(crate) lm_head: Arc<RwLock<Option<Linear>>>,
    pub(crate) ple: Arc<RwLock<Option<PleComponents>>>,
    tokenizer: Option<Arc<Qwen3Tokenizer>>,
    model_id: u64,
}

static MODEL_ID_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[napi]
impl Gemma4Model {
    #[napi(constructor)]
    pub fn new(config: Gemma4Config) -> Result<Self> {
        let num_layers = config.num_hidden_layers as usize;
        let hidden_size = config.hidden_size as u32;
        let vocab_size = config.vocab_size as u32;

        let embed_tokens = Embedding::new(vocab_size, hidden_size)?;
        let final_norm = RMSNorm::new(hidden_size, Some(config.rms_norm_eps))?;

        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(Linear::new(hidden_size, vocab_size, Some(false))?)
        };

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(Gemma4DecoderLayer::new(&config, i)?);
        }

        // Initialize PLE model-level components if enabled
        let ple = if config.per_layer_input_embeds {
            let ple_dim = config.ple_dim();
            let vocab_ple = config.vocab_size_per_layer_input.unwrap_or(0);
            if ple_dim > 0 && vocab_ple > 0 {
                let total_ple_dim = (num_layers as i32) * ple_dim;
                Some(PleComponents {
                    embed_tokens_per_layer: Embedding::new(vocab_ple as u32, total_ple_dim as u32)?,
                    per_layer_model_projection: Linear::new(
                        hidden_size,
                        total_ple_dim as u32,
                        Some(false),
                    )?,
                    per_layer_projection_norm: RMSNorm::new(
                        ple_dim as u32,
                        Some(config.rms_norm_eps),
                    )?,
                    per_layer_input_scale: 2.0_f64.powf(-0.5),
                    per_layer_model_projection_scale: (config.hidden_size as f64).powf(-0.5),
                    ple_dim,
                    num_layers: num_layers as i32,
                    vocab_size_per_layer_input: vocab_ple,
                })
            } else {
                None
            }
        } else {
            None
        };

        let model_id = MODEL_ID_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        Ok(Self {
            config,
            embed_tokens: Arc::new(RwLock::new(embed_tokens)),
            layers: Arc::new(RwLock::new(layers)),
            final_norm: Arc::new(RwLock::new(final_norm)),
            lm_head: Arc::new(RwLock::new(lm_head)),
            ple: Arc::new(RwLock::new(ple)),
            tokenizer: None,
            model_id,
        })
    }

    #[napi]
    pub fn model_id(&self) -> u32 {
        self.model_id as u32
    }

    /// Load a Gemma4 model from a directory.
    #[napi]
    pub async fn load(model_path: String) -> Result<Gemma4Model> {
        Self::load_from_dir(&model_path).await
    }

    /// Chat with the model using a list of messages.
    #[napi]
    pub async fn chat(
        &self,
        messages: Vec<ChatMessage>,
        config: Option<Gemma4ChatConfig>,
    ) -> Result<Gemma4ChatResult> {
        let config = config.unwrap_or(Gemma4ChatConfig {
            max_new_tokens: None,
            temperature: None,
            top_k: None,
            top_p: None,
            min_p: None,
        });

        let max_new_tokens = config.max_new_tokens.unwrap_or(2048);

        let tokenizer = self
            .tokenizer
            .clone()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))?;

        // Clone Arcs for spawn_blocking
        let embed_tokens = self.embed_tokens.clone();
        let layers = self.layers.clone();
        let final_norm = self.final_norm.clone();
        let lm_head = self.lm_head.clone();
        let ple = self.ple.clone();
        let model_config = self.config.clone();

        let sampling_config = make_sampling_config(&config);
        let eos_ids = model_config.eos_token_ids.clone();

        tokio::task::spawn_blocking(move || {
            // Construct Gemma 4 chat format manually:
            //   <bos><|turn>user\n{content}<turn|>\n<|turn>model\n
            // No Jinja2 template needed — Gemma 4 tokenizer doesn't include one.
            let mut prompt_text = String::new();
            for msg in &messages {
                let role = &msg.role;
                let content = &msg.content;
                prompt_text.push_str(&format!("<|turn>{}\n{}<turn|>\n", role, content));
            }
            // Add generation prompt for the model's response
            prompt_text.push_str("<|turn>model\n");

            // Encode with BOS token
            let tokens = tokenizer.encode_sync(&prompt_text, Some(true))?;

            // Create prompt tensor
            let token_arr: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
            let prompt = MxArray::from_int32(&token_arr, &[1, tokens.len() as i64])?;

            // Initialize caches
            let mut caches = init_caches_for_config(&model_config);

            // Acquire locks
            let embed_guard = embed_tokens.blocking_read();
            let layers_guard = layers.blocking_read();
            let norm_guard = final_norm.blocking_read();
            let lm_head_guard = lm_head.blocking_read();
            let ple_guard = ple.blocking_read();

            // Prefill
            let logits = forward_inner(
                &prompt,
                &embed_guard,
                &layers_guard,
                &mut caches,
                &norm_guard,
                &lm_head_guard,
                ple_guard.as_ref(),
                &model_config,
            )?;

            // Sample first token from last position
            let last_logits = logits.slice_axis(1, tokens.len() as i64 - 1, tokens.len() as i64)?;
            let last_logits = last_logits.squeeze(Some(&[1]))?;

            let mut y = sample(&last_logits, sampling_config)?;
            MxArray::async_eval_arrays(&[&y]);

            // Decode loop
            let mut generated_tokens: Vec<u32> = Vec::new();
            let mut finish_reason = "length".to_string();

            for step in 0..max_new_tokens {
                y.eval();
                let token_id = y.item_at_int32(0)? as u32;
                generated_tokens.push(token_id);

                // Check EOS
                if eos_ids.contains(&(token_id as i32)) {
                    finish_reason = "stop".to_string();
                    break;
                }

                // Forward single token
                let next_ids = y.reshape(&[1, 1])?;
                let logits = forward_inner(
                    &next_ids,
                    &embed_guard,
                    &layers_guard,
                    &mut caches,
                    &norm_guard,
                    &lm_head_guard,
                    ple_guard.as_ref(),
                    &model_config,
                )?;
                let logits = logits.squeeze(Some(&[1]))?;

                // Sample next token
                let next_token = sample(&logits, sampling_config)?;
                MxArray::async_eval_arrays(&[&next_token]);

                y = next_token;

                // Periodic cache clear to prevent memory accumulation
                if (step + 1) % 256 == 0 {
                    crate::array::synchronize_and_clear_cache();
                }
            }

            // Decode text
            let text = tokenizer.decode_sync(&generated_tokens, true)?;

            Ok(Gemma4ChatResult {
                text,
                num_tokens: generated_tokens.len() as u32,
                finish_reason,
            })
        })
        .await
        .map_err(|e| Error::from_reason(format!("Chat task failed: {}", e)))?
    }
}

impl Gemma4Model {
    pub fn set_tokenizer(&mut self, tokenizer: Arc<Qwen3Tokenizer>) {
        self.tokenizer = Some(tokenizer);
    }
}

fn init_caches_for_config(config: &Gemma4Config) -> Vec<Gemma4LayerCache> {
    let num_layers = config.num_hidden_layers as usize;
    let mut caches = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        if config.is_global_layer(i) {
            caches.push(Gemma4LayerCache::new_global());
        } else {
            caches.push(Gemma4LayerCache::new_sliding(config.sliding_window));
        }
    }
    caches
}

fn make_sampling_config(config: &Gemma4ChatConfig) -> Option<SamplingConfig> {
    let temp = config.temperature.unwrap_or(0.0);
    if temp <= 0.0 {
        // Greedy: use a near-zero temperature for argmax-like behavior.
        // Cannot pass None because sample() defaults to temperature=1.0.
        return Some(SamplingConfig {
            temperature: Some(0.0),
            top_k: None,
            top_p: None,
            min_p: None,
        });
    }
    Some(SamplingConfig {
        temperature: Some(temp),
        top_k: config.top_k,
        top_p: config.top_p,
        min_p: config.min_p,
    })
}

/// Synchronous forward pass (used inside spawn_blocking).
fn forward_inner(
    input_ids: &MxArray,
    embedding: &Embedding,
    layers: &[Gemma4DecoderLayer],
    caches: &mut [Gemma4LayerCache],
    final_norm: &RMSNorm,
    lm_head: &Option<Linear>,
    ple: Option<&PleComponents>,
    config: &Gemma4Config,
) -> Result<MxArray> {
    let mut h = embedding.forward(input_ids)?;

    // Embedding scaling: multiply by sqrt(hidden_size)
    let scale = (config.hidden_size as f64).sqrt();
    let scale_arr = MxArray::scalar_float(scale)?;
    let scale_arr = scale_arr.astype(h.dtype()?)?;
    h = h.mul(&scale_arr)?;

    let seq_len = h.shape_at(1)?;

    // Compute PLE (per-layer embeddings) if enabled.
    // Result: per_layer_inputs[i] = combined[:, :, i, :] for each layer, shape [B, T, ple_dim]
    let per_layer_inputs: Option<Vec<MxArray>> = if let Some(ple) = ple {
        let ple_dim = ple.ple_dim as i64;
        let num_layers = ple.num_layers as i64;

        // Clamp token IDs to PLE vocab range (may be smaller than main vocab)
        let max_valid = MxArray::scalar_int(ple.vocab_size_per_layer_input - 1)?;
        let zero = MxArray::scalar_int(0)?;
        let clamped_ids = input_ids.maximum(&zero)?.minimum(&max_valid)?;

        // per_layer_embeds: [B, T, num_layers * ple_dim]
        let per_layer_embeds = ple.embed_tokens_per_layer.forward(&clamped_ids)?;
        // HF's Gemma4TextScaledWordEmbedding scales by sqrt(ple_dim)
        let embed_scale = MxArray::scalar_float((ple.ple_dim as f64).sqrt())?;
        let embed_scale = embed_scale.astype(per_layer_embeds.dtype()?)?;
        let per_layer_embeds = per_layer_embeds.mul(&embed_scale)?;
        // Reshape to [B, T, num_layers, ple_dim]
        let batch = per_layer_embeds.shape_at(0)?;
        let per_layer_embeds = per_layer_embeds.reshape(&[batch, seq_len, num_layers, ple_dim])?;

        // projected: [B, T, num_layers * ple_dim]
        let projected = ple.per_layer_model_projection.forward(&h)?;
        let proj_scale = MxArray::scalar_float(ple.per_layer_model_projection_scale)?;
        let proj_scale = proj_scale.astype(projected.dtype()?)?;
        let projected = projected.mul(&proj_scale)?;
        // Reshape to [B, T, num_layers, ple_dim]
        let projected = projected.reshape(&[batch, seq_len, num_layers, ple_dim])?;

        // Normalize the projection ONLY (before combining with embeds).
        // vLLM: per_layer_projection_norm(projection), then combine.
        let projected = ple.per_layer_projection_norm.forward(&projected)?;

        // Combine: (normed_projection + per_layer_embeds) * 1/sqrt(2)
        let combined = projected.add(&per_layer_embeds)?;
        let input_scale = MxArray::scalar_float(ple.per_layer_input_scale)?;
        let input_scale = input_scale.astype(combined.dtype()?)?;
        let combined = combined.mul(&input_scale)?;

        // Slice per layer: combined[:, :, i, :] -> [B, T, ple_dim]
        let mut inputs = Vec::with_capacity(num_layers as usize);
        for i in 0..num_layers {
            let layer_ple = combined.slice_axis(2, i, i + 1)?;
            let layer_ple = layer_ple.squeeze(Some(&[2]))?;
            inputs.push(layer_ple);
        }
        Some(inputs)
    } else {
        None
    };

    // Build masks for prefill (seq_len > 1) only
    // During decode (seq_len == 1), pass None to let SDPA handle it optimally
    let global_mask = if seq_len > 1 {
        let global_idx = (0..config.num_hidden_layers as usize)
            .find(|&i| config.is_global_layer(i))
            .unwrap_or(0);
        Some(create_causal_mask(
            seq_len,
            caches[global_idx].get_offset(),
        )?)
    } else {
        None
    };

    let sliding_mask = if seq_len > 1 {
        let sliding_idx = (0..config.num_hidden_layers as usize)
            .find(|&i| config.is_sliding_layer(i))
            .unwrap_or(0);
        Some(create_sliding_mask(
            seq_len,
            caches[sliding_idx].get_offset(),
            config.sliding_window as i64,
        )?)
    } else {
        None
    };

    // Forward through layers with KV cache sharing.
    //
    // Layers are split into two groups:
    // - Non-shared (0..first_kv_shared): compute Q, K, V normally; update own cache
    // - Shared (first_kv_shared..num_layers): compute Q only; reuse K/V from anchor cache
    //
    // The anchor for a shared layer is the last non-shared layer of the same attention
    // type (sliding or global). After running the anchor layer, we read its K/V cache
    // contents and provide them to all shared layers of that type.
    let has_kv_sharing = config.num_kv_shared_layers.is_some_and(|n| n > 0);

    // Shared K/V storage: anchor_layer_idx -> (keys, values)
    // Only populated when KV sharing is active.
    let mut shared_kv: HashMap<usize, (MxArray, MxArray)> = HashMap::new();

    for (i, layer) in layers.iter().enumerate() {
        let is_global = config.is_global_layer(i);
        let mask = if is_global {
            global_mask.as_ref()
        } else {
            sliding_mask.as_ref()
        };
        let ple_input = per_layer_inputs.as_ref().and_then(|inputs| inputs.get(i));

        if has_kv_sharing && config.is_kv_shared_layer(i) {
            // Shared layer: compute Q only, reuse K/V from anchor
            let anchor_idx = config.kv_shared_anchor(i).ok_or_else(|| {
                Error::from_reason(format!(
                    "Layer {} is shared but has no anchor (missing layer type match)",
                    i
                ))
            })?;

            let (shared_keys, shared_values) = shared_kv.get(&anchor_idx).ok_or_else(|| {
                Error::from_reason(format!(
                    "Anchor layer {} K/V not found for shared layer {}",
                    anchor_idx, i
                ))
            })?;

            // RoPE offset for shared layer queries.
            // The anchor cache's offset has already been incremented by update_and_fetch
            // (offset = total tokens in cache AFTER the anchor processed the current input).
            // Shared layer queries need RoPE at the same positions as the anchor's queries,
            // which were applied BEFORE update. So subtract seq_len to get the pre-update offset.
            let cache_offset = caches[anchor_idx].get_offset() - seq_len as i32;

            // For shared layers, the mask may need to match the anchor's K/V length.
            // The anchor's K/V sequence length may differ from what the original mask
            // was built for (e.g., sliding anchor has window-sized K/V).
            let kv_seq_len = shared_keys.shape_at(2)?;
            let adjusted_mask = if let Some(m) = mask {
                let mask_kv_len = m.shape_at(3)?;
                if mask_kv_len > kv_seq_len {
                    // Narrow the mask's last dim to match shared K/V length
                    Some(m.slice_axis(3, mask_kv_len - kv_seq_len, mask_kv_len)?)
                } else {
                    Some(m.clone())
                }
            } else {
                None
            };

            h = layer.forward_shared(
                &h,
                adjusted_mask.as_ref(),
                shared_keys,
                shared_values,
                cache_offset,
                ple_input,
            )?;
        } else {
            h = layer.forward(&h, mask, Some(&mut caches[i]), ple_input)?;

            if has_kv_sharing
                && config.should_store_shared_kv(i)
                && let Some((keys, values)) = caches[i].get_cached_kv()
            {
                shared_kv.insert(i, (keys, values));
            }
        }
    }

    // Final norm
    h = final_norm.forward(&h)?;

    // LM head or tied embeddings
    let logits = if let Some(head) = lm_head {
        head.forward(&h)?
    } else {
        let weight = embedding.get_weight();
        let weight_t = weight.transpose(Some(&[1, 0]))?;
        h.matmul(&weight_t)?
    };

    // Logit softcapping
    if let Some(cap) = config.final_logit_softcapping {
        let cap_arr = MxArray::scalar_float(cap)?;
        let cap_arr = cap_arr.astype(logits.dtype()?)?;
        let scaled = logits.div(&cap_arr)?;
        let capped = scaled.tanh()?;
        Ok(capped.mul(&cap_arr)?)
    } else {
        Ok(logits)
    }
}

fn create_causal_mask(seq_len: i64, offset: i32) -> Result<MxArray> {
    let total_len = seq_len + offset as i64;
    let rows = MxArray::arange(offset as f64, (offset as i64 + seq_len) as f64, None, None)?;
    let cols = MxArray::arange(0.0, total_len as f64, None, None)?;
    let rows = rows.reshape(&[seq_len, 1])?;
    let cols = cols.reshape(&[1, total_len])?;
    let valid = rows.greater_equal(&cols)?;

    let neg_inf = MxArray::full(
        &[1],
        Either::A(f64::NEG_INFINITY),
        Some(crate::array::DType::Float32),
    )?;
    let zero_f = MxArray::full(&[1], Either::A(0.0), Some(crate::array::DType::Float32))?;
    let mask = valid.where_(&zero_f, &neg_inf)?;
    mask.reshape(&[1, 1, seq_len, total_len])
}

fn create_sliding_mask(seq_len: i64, offset: i32, window_size: i64) -> Result<MxArray> {
    let total_len = seq_len + offset as i64;
    let rows = MxArray::arange(offset as f64, (offset as i64 + seq_len) as f64, None, None)?;
    let cols = MxArray::arange(0.0, total_len as f64, None, None)?;
    let rows = rows.reshape(&[seq_len, 1])?;
    let cols = cols.reshape(&[1, total_len])?;
    let distance = rows.sub(&cols)?;

    let zero = MxArray::scalar_int(0)?;
    let window = MxArray::scalar_int(window_size as i32)?;
    let causal = distance.greater_equal(&zero)?;
    let in_window = distance.less(&window)?;
    let valid = causal.logical_and(&in_window)?;

    let neg_inf = MxArray::full(
        &[1],
        Either::A(f64::NEG_INFINITY),
        Some(crate::array::DType::Float32),
    )?;
    let zero_f = MxArray::full(&[1], Either::A(0.0), Some(crate::array::DType::Float32))?;
    let mask = valid.where_(&zero_f, &neg_inf)?;
    mask.reshape(&[1, 1, seq_len, total_len])
}
