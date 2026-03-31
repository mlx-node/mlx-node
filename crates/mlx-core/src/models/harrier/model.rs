use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi_derive::napi;
use tracing::info;

use crate::array::MxArray;
use crate::nn::{Embedding, RMSNorm};
use crate::tokenizer::Qwen3Tokenizer;
use crate::transformer::TransformerBlock;

use super::HarrierConfig;

/// Harrier embedding model (Qwen3 backbone for text embeddings).
///
/// Uses last-token pooling and L2 normalization to produce fixed-size
/// embedding vectors from variable-length text inputs.
#[napi]
pub struct HarrierModel {
    pub(crate) config: HarrierConfig,
    pub(crate) embedding: Embedding,
    pub(crate) layers: Vec<TransformerBlock>,
    pub(crate) final_norm: RMSNorm,
    pub(crate) tokenizer: Option<Arc<Qwen3Tokenizer>>,
}

#[napi]
impl HarrierModel {
    #[napi(constructor)]
    pub fn new(config: HarrierConfig) -> Result<Self> {
        let embedding = Embedding::new(config.vocab_size as u32, config.hidden_size as u32)?;

        let layers = (0..config.num_layers)
            .map(|_| {
                TransformerBlock::new(
                    config.hidden_size as u32,
                    config.num_heads as u32,
                    config.num_key_value_heads as u32,
                    config.intermediate_size as u32,
                    config.rms_norm_eps,
                    Some(config.rope_theta),
                    Some(config.use_qk_norm),
                    Some(config.head_dim as u32),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let final_norm = RMSNorm::new(config.hidden_size as u32, Some(config.rms_norm_eps))?;

        Ok(Self {
            config,
            embedding,
            layers,
            final_norm,
            tokenizer: None,
        })
    }

    /// Forward pass returning hidden states (no lm_head projection).
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs, shape: [batch_size, seq_len]
    ///
    /// # Returns
    /// * Hidden states, shape: [batch_size, seq_len, hidden_size]
    #[napi]
    pub fn forward(&self, input_ids: &MxArray) -> Result<MxArray> {
        forward_inner(&self.embedding, &self.layers, &self.final_norm, input_ids)
    }

    /// Encode a single text into a normalized embedding vector.
    ///
    /// Tokenizes the text, runs the forward pass, applies last-token pooling,
    /// and L2-normalizes the result.
    ///
    /// # Arguments
    /// * `text` - Input text to encode
    /// * `instruction` - Optional task instruction to prepend (for queries)
    ///
    /// # Returns
    /// * Embedding vector, shape: [hidden_size]
    #[napi]
    pub async fn encode(&self, text: String, instruction: Option<String>) -> Result<MxArray> {
        let tokenizer = self.require_tokenizer()?.clone();
        let config_hidden = self.config.hidden_size;

        let embedding = self.embedding.clone();
        let layers: Vec<_> = self.layers.iter().cloned().collect();
        let final_norm = self.final_norm.clone();

        napi::bindgen_prelude::spawn_blocking(move || {
            let full_text = match instruction {
                Some(instr) => format!("{}{}", instr, text),
                None => text,
            };

            let token_ids = tokenizer.encode_sync(&full_text, Some(true))?;
            let seq_len = token_ids.len();
            let input = MxArray::from_uint32(&token_ids, &[1, seq_len as i64])?;

            let hidden_states = forward_inner(&embedding, &layers, &final_norm, &input)?;

            let pooled = last_token_pool(&hidden_states, seq_len, config_hidden)?;
            let pooled = pooled.reshape(&[config_hidden as i64])?;

            let result = l2_normalize(&pooled)?;
            result.eval();
            Ok(result)
        })
        .await
        .map_err(|e| Error::from_reason(format!("encode failed: {}", e)))?
    }

    /// Encode a batch of texts into normalized embedding vectors.
    ///
    /// Each text is independently tokenized and encoded (no padding needed
    /// since each goes through its own forward pass).
    ///
    /// # Arguments
    /// * `texts` - Input texts to encode
    /// * `instruction` - Optional task instruction to prepend to each text
    ///
    /// # Returns
    /// * Embedding matrix, shape: [batch_size, hidden_size]
    #[napi]
    pub async fn encode_batch(
        &self,
        texts: Vec<String>,
        instruction: Option<String>,
    ) -> Result<MxArray> {
        let tokenizer = self.require_tokenizer()?.clone();
        let config_hidden = self.config.hidden_size;

        let embedding = self.embedding.clone();
        let layers: Vec<_> = self.layers.iter().cloned().collect();
        let final_norm = self.final_norm.clone();

        napi::bindgen_prelude::spawn_blocking(move || {
            let mut all_embeddings: Vec<MxArray> = Vec::with_capacity(texts.len());

            for text in texts {
                let full_text = match &instruction {
                    Some(instr) => format!("{}{}", instr, text),
                    None => text,
                };

                let token_ids = tokenizer.encode_sync(&full_text, Some(true))?;
                let seq_len = token_ids.len();
                let input = MxArray::from_uint32(&token_ids, &[1, seq_len as i64])?;

                let hidden_states = forward_inner(&embedding, &layers, &final_norm, &input)?;

                let pooled = last_token_pool(&hidden_states, seq_len, config_hidden)?;
                let pooled = pooled.reshape(&[1, config_hidden as i64])?;

                all_embeddings.push(l2_normalize(&pooled)?);
            }

            if all_embeddings.is_empty() {
                return Err(Error::from_reason("Cannot encode empty batch"));
            }

            let refs: Vec<&MxArray> = all_embeddings.iter().collect();
            let result = MxArray::concatenate_many(refs, Some(0))?;
            result.eval();
            Ok(result)
        })
        .await
        .map_err(|e| Error::from_reason(format!("encode_batch failed: {}", e)))?
    }

    /// Get the model configuration.
    #[napi]
    pub fn get_config(&self) -> HarrierConfig {
        self.config.clone()
    }

    /// Get the total number of model parameters.
    #[napi]
    pub fn num_parameters(&self) -> i64 {
        let vocab = self.config.vocab_size as i64;
        let hidden = self.config.hidden_size as i64;
        let inter = self.config.intermediate_size as i64;
        let heads = self.config.num_heads as i64;
        let kv_heads = self.config.num_key_value_heads as i64;
        let head_dim = self.config.head_dim as i64;
        let n_layers = self.config.num_layers as i64;

        let embedding_params = vocab * hidden;
        let final_norm_params = hidden;

        let attn_params = (heads * head_dim * hidden)
            + (kv_heads * head_dim * hidden)
            + (kv_heads * head_dim * hidden)
            + (hidden * heads * head_dim);
        let mlp_params = inter * hidden * 3;
        let norm_params = hidden * 2;
        let qk_norm_params = if self.config.use_qk_norm {
            head_dim * 2
        } else {
            0
        };
        let layer_params = attn_params + mlp_params + norm_params + qk_norm_params;

        embedding_params + final_norm_params + n_layers * layer_params
    }

    /// Apply loaded parameters to the model.
    pub(crate) fn load_parameters(
        &mut self,
        params: &std::collections::HashMap<String, MxArray>,
    ) -> Result<()> {
        if let Some(w) = params.get("embedding.weight") {
            self.embedding.set_weight(w)?;
        } else {
            return Err(Error::from_reason("embedding.weight not found"));
        }

        for (i, layer) in self.layers.iter_mut().enumerate() {
            let prefix = format!("layers.{}", i);

            let attn = &mut layer.self_attn;
            set_required(
                params,
                &format!("{}.self_attn.q_proj.weight", prefix),
                |w| attn.set_q_proj_weight(w),
            )?;
            set_required(
                params,
                &format!("{}.self_attn.k_proj.weight", prefix),
                |w| attn.set_k_proj_weight(w),
            )?;
            set_required(
                params,
                &format!("{}.self_attn.v_proj.weight", prefix),
                |w| attn.set_v_proj_weight(w),
            )?;
            set_required(
                params,
                &format!("{}.self_attn.o_proj.weight", prefix),
                |w| attn.set_o_proj_weight(w),
            )?;

            if self.config.use_qk_norm {
                set_required(
                    params,
                    &format!("{}.self_attn.q_norm.weight", prefix),
                    |w| attn.set_q_norm_weight(w),
                )?;
                set_required(
                    params,
                    &format!("{}.self_attn.k_norm.weight", prefix),
                    |w| attn.set_k_norm_weight(w),
                )?;
            }

            let mlp = &mut layer.mlp;
            set_required(params, &format!("{}.mlp.gate_proj.weight", prefix), |w| {
                mlp.set_gate_proj_weight(w)
            })?;
            set_required(params, &format!("{}.mlp.up_proj.weight", prefix), |w| {
                mlp.set_up_proj_weight(w)
            })?;
            set_required(params, &format!("{}.mlp.down_proj.weight", prefix), |w| {
                mlp.set_down_proj_weight(w)
            })?;

            set_required(params, &format!("{}.input_layernorm.weight", prefix), |w| {
                layer.set_input_layernorm_weight(w)
            })?;
            set_required(
                params,
                &format!("{}.post_attention_layernorm.weight", prefix),
                |w| layer.set_post_attention_layernorm_weight(w),
            )?;
        }

        if let Some(w) = params.get("final_norm.weight") {
            self.final_norm.set_weight(w)?;
        } else {
            return Err(Error::from_reason("final_norm.weight not found"));
        }

        info!(
            "Loaded {} layers into HarrierModel ({} hidden)",
            self.config.num_layers, self.config.hidden_size
        );
        Ok(())
    }

    fn require_tokenizer(&self) -> Result<&Arc<Qwen3Tokenizer>> {
        self.tokenizer.as_ref().ok_or_else(|| {
            Error::from_reason(
                "Tokenizer not loaded. Use HarrierModel.load() to load a model with tokenizer.",
            )
        })
    }
}

/// Shared forward pass: embedding -> transformer layers -> final norm.
fn forward_inner(
    embedding: &Embedding,
    layers: &[TransformerBlock],
    final_norm: &RMSNorm,
    input_ids: &MxArray,
) -> Result<MxArray> {
    let mut hidden_states = embedding.forward(input_ids)?;
    for layer in layers {
        hidden_states = layer.forward(&hidden_states, None, None)?;
    }
    final_norm.forward(&hidden_states)
}

/// Extract the last token's hidden state from the full sequence output.
fn last_token_pool(hidden_states: &MxArray, seq_len: usize, hidden_size: i32) -> Result<MxArray> {
    hidden_states.slice(
        &[0, seq_len as i64 - 1, 0],
        &[1, seq_len as i64, hidden_size as i64],
    )
}

/// L2-normalize an array along the last axis.
fn l2_normalize(x: &MxArray) -> Result<MxArray> {
    let norm = x.square()?.sum(Some(&[-1]), Some(true))?.sqrt()?;
    let norm = norm.clip(Some(1e-12), None)?;
    x.div(&norm)
}

fn set_required(
    params: &std::collections::HashMap<String, MxArray>,
    name: &str,
    setter: impl FnOnce(&MxArray) -> Result<()>,
) -> Result<()> {
    match params.get(name) {
        Some(w) => setter(w),
        None => Err(Error::from_reason(format!("{} not found", name))),
    }
}
