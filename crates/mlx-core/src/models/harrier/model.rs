use std::collections::HashMap;
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
    /// Named prompt presets loaded from config_sentence_transformers.json.
    /// Keys are task names (e.g. "web_search_query"), values are full prefix strings.
    pub(crate) prompts: HashMap<String, String>,
}

#[napi]
impl HarrierModel {
    #[napi(constructor)]
    pub fn new(mut config: HarrierConfig) -> Result<Self> {
        // Resolve use_qk_norm default — Qwen3 always uses QK normalization
        let use_qk_norm = config.use_qk_norm.unwrap_or(true);
        config.use_qk_norm = Some(use_qk_norm);

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
                    Some(use_qk_norm),
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
            prompts: HashMap::new(),
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
    /// and L2-normalizes the result. Truncates to `max_position_embeddings`.
    ///
    /// # Arguments
    /// * `text` - Input text to encode
    /// * `instruction` - Optional task instruction prefix or preset name
    ///   (e.g. `"web_search_query"` resolves to the full Harrier prompt).
    ///   Pass `null` for documents/passages that need no instruction.
    ///
    /// # Returns
    /// * Embedding vector, shape: [hidden_size]
    #[napi]
    pub async fn encode(&self, text: String, instruction: Option<String>) -> Result<MxArray> {
        let tokenizer = self.require_tokenizer()?.clone();
        let instruction = instruction.map(|i| self.resolve_instruction(i));
        let max_tokens = self.config.max_position_embeddings as usize;
        let config_hidden = self.config.hidden_size;

        let embedding = self.embedding.clone();
        let layers: Vec<_> = self.layers.to_vec();
        let final_norm = self.final_norm.clone();

        napi::bindgen_prelude::spawn_blocking(move || {
            let full_text = match instruction {
                Some(instr) => format!("{}{}", instr, text),
                None => text,
            };

            let mut token_ids = tokenizer.encode_sync(&full_text, Some(true))?;
            truncate_preserving_tail(&mut token_ids, max_tokens);
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
    /// since each goes through its own forward pass). Truncates each text
    /// to `max_position_embeddings`.
    ///
    /// # Arguments
    /// * `texts` - Input texts to encode
    /// * `instruction` - Optional task instruction prefix or preset name
    ///   (e.g. `"web_search_query"` resolves to the full Harrier prompt).
    ///   Pass `null` for documents/passages that need no instruction.
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
        let instruction = instruction.map(|i| self.resolve_instruction(i));
        let max_tokens = self.config.max_position_embeddings as usize;
        let config_hidden = self.config.hidden_size;

        let embedding = self.embedding.clone();
        let layers: Vec<_> = self.layers.to_vec();
        let final_norm = self.final_norm.clone();

        napi::bindgen_prelude::spawn_blocking(move || {
            let mut all_embeddings: Vec<MxArray> = Vec::with_capacity(texts.len());

            for text in texts {
                let full_text = match &instruction {
                    Some(instr) => format!("{}{}", instr, text),
                    None => text,
                };

                let mut token_ids = tokenizer.encode_sync(&full_text, Some(true))?;
                truncate_preserving_tail(&mut token_ids, max_tokens);
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

    /// Get available prompt presets loaded from config_sentence_transformers.json.
    ///
    /// Returns a map of task name -> full instruction prefix.
    /// Pass a task name to `encode()`/`encodeBatch()` as the `instruction` parameter
    /// to use a preset instead of a raw prefix string.
    #[napi]
    pub fn get_prompts(&self) -> HashMap<String, String> {
        self.prompts.clone()
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
        let qk_norm_params = if self.config.use_qk_norm.unwrap_or(true) {
            head_dim * 2
        } else {
            0
        };
        let layer_params = attn_params + mlp_params + norm_params + qk_norm_params;

        embedding_params + final_norm_params + n_layers * layer_params
    }

    /// Apply loaded parameters to the model.
    pub(crate) fn load_parameters(&mut self, params: &HashMap<String, MxArray>) -> Result<()> {
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

            if self.config.use_qk_norm.unwrap_or(true) {
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

    /// If `instruction` matches a loaded prompt name, return the full prefix.
    /// Otherwise return it as-is (the caller passed a raw prefix string).
    fn resolve_instruction(&self, instruction: String) -> String {
        self.prompts
            .get(&instruction)
            .cloned()
            .unwrap_or(instruction)
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

/// Truncate token sequence to `max_len` while preserving the trailing token.
///
/// Harrier pools the final token, which is the special end-of-sequence token
/// added by `add_special_tokens=true`. Naive truncation would drop it, causing
/// the model to pool a content token instead — deviating from the training recipe.
/// This keeps the first `max_len - 1` tokens plus the original tail token.
fn truncate_preserving_tail(token_ids: &mut Vec<u32>, max_len: usize) {
    if token_ids.len() > max_len && max_len > 0 {
        let tail = *token_ids.last().unwrap();
        token_ids.truncate(max_len);
        *token_ids.last_mut().unwrap() = tail;
    }
}

/// L2-normalize an array along the last axis.
fn l2_normalize(x: &MxArray) -> Result<MxArray> {
    let norm = x.square()?.sum(Some(&[-1]), Some(true))?.sqrt()?;
    let norm = norm.clip(Some(1e-12), None)?;
    x.div(&norm)
}

fn set_required(
    params: &HashMap<String, MxArray>,
    name: &str,
    setter: impl FnOnce(&MxArray) -> Result<()>,
) -> Result<()> {
    match params.get(name) {
        Some(w) => setter(w),
        None => Err(Error::from_reason(format!("{} not found", name))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_values(arr: &MxArray) -> Vec<f32> {
        arr.eval();
        arr.to_float32().unwrap().to_vec()
    }

    fn get_shape(arr: &MxArray) -> Vec<i64> {
        arr.shape().unwrap().to_vec()
    }

    #[test]
    fn test_l2_normalize_produces_unit_vector() {
        // [3, 4, 0] has L2 norm = 5, so normalized = [0.6, 0.8, 0.0]
        let x = MxArray::from_float32(&[3.0, 4.0, 0.0], &[3]).unwrap();
        let normed = l2_normalize(&x).unwrap();
        let vals = get_values(&normed);

        assert!((vals[0] - 0.6).abs() < 1e-5);
        assert!((vals[1] - 0.8).abs() < 1e-5);
        assert!(vals[2].abs() < 1e-5);

        // Check the result actually has unit norm
        let norm_sq: f32 = vals.iter().map(|v| v * v).sum();
        assert!((norm_sq - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_l2_normalize_handles_zero_vector() {
        // Zero vector should not produce NaN (clip(1e-12) prevents div-by-zero)
        let x = MxArray::from_float32(&[0.0, 0.0, 0.0], &[3]).unwrap();
        let normed = l2_normalize(&x).unwrap();
        let vals = get_values(&normed);

        for v in &vals {
            assert!(!v.is_nan(), "L2 normalize of zero vector produced NaN");
        }
    }

    #[test]
    fn test_l2_normalize_2d_normalizes_per_row() {
        // Two rows: [3,4] (norm=5) and [0,1] (norm=1)
        let x = MxArray::from_float32(&[3.0, 4.0, 0.0, 1.0], &[2, 2]).unwrap();
        let normed = l2_normalize(&x).unwrap();
        let vals = get_values(&normed);

        // Row 0: [0.6, 0.8]
        assert!((vals[0] - 0.6).abs() < 1e-5);
        assert!((vals[1] - 0.8).abs() < 1e-5);
        // Row 1: [0.0, 1.0]
        assert!(vals[2].abs() < 1e-5);
        assert!((vals[3] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_last_token_pool_extracts_final_position() {
        // [1, 3, 4] tensor: 3 tokens, 4-dim hidden states
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let x = MxArray::from_float32(&data, &[1, 3, 4]).unwrap();
        let pooled = last_token_pool(&x, 3, 4).unwrap();

        assert_eq!(get_shape(&pooled), vec![1, 1, 4]);
        // Last token (index 2): values [8, 9, 10, 11]
        let vals = get_values(&pooled);
        assert_eq!(vals, vec![8.0, 9.0, 10.0, 11.0]);
    }

    #[test]
    fn test_last_token_pool_single_token() {
        let x = MxArray::from_float32(&[1.0, 2.0], &[1, 1, 2]).unwrap();
        let pooled = last_token_pool(&x, 1, 2).unwrap();

        assert_eq!(get_shape(&pooled), vec![1, 1, 2]);
        assert_eq!(get_values(&pooled), vec![1.0, 2.0]);
    }

    #[test]
    fn test_truncate_preserving_tail_keeps_special_token() {
        // Reproduces the reviewer's scenario: content tokens + trailing EOS (151643)
        let mut ids = vec![14990, 23811, 23811, 23811, 151643];
        truncate_preserving_tail(&mut ids, 3);
        // First 2 content tokens + EOS preserved at end
        assert_eq!(ids, vec![14990, 23811, 151643]);
    }

    #[test]
    fn test_truncate_preserving_tail_noop_under_limit() {
        let mut ids = vec![1, 2, 3];
        truncate_preserving_tail(&mut ids, 5);
        assert_eq!(ids, vec![1, 2, 3]);
    }

    #[test]
    fn test_truncate_preserving_tail_noop_at_exact_limit() {
        let mut ids = vec![1, 2, 3];
        truncate_preserving_tail(&mut ids, 3);
        assert_eq!(ids, vec![1, 2, 3]);
    }

    #[test]
    fn test_truncate_preserving_tail_to_single_token() {
        // Even truncating to 1 should preserve the tail special token
        let mut ids = vec![1, 2, 3, 151643];
        truncate_preserving_tail(&mut ids, 1);
        assert_eq!(ids, vec![151643]);
    }

    #[test]
    fn test_truncate_preserving_tail_to_two_tokens() {
        let mut ids = vec![100, 200, 300, 400, 151643];
        truncate_preserving_tail(&mut ids, 2);
        // First content token + tail special token
        assert_eq!(ids, vec![100, 151643]);
    }

    #[test]
    fn test_resolve_instruction_with_known_prompt() {
        let mut prompts = HashMap::new();
        prompts.insert(
            "web_search_query".to_string(),
            "Instruct: search\nQuery: ".to_string(),
        );

        let config = HarrierConfig {
            hidden_size: 64,
            num_layers: 1,
            num_heads: 2,
            num_key_value_heads: 1,
            intermediate_size: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            max_position_embeddings: 512,
            head_dim: 32,
            use_qk_norm: None,
            vocab_size: 100,
        };

        let mut model = HarrierModel::new(config).unwrap();
        model.prompts = prompts;

        // Named prompt resolves to full prefix
        assert_eq!(
            model.resolve_instruction("web_search_query".to_string()),
            "Instruct: search\nQuery: "
        );
        // Unknown name passes through as-is
        assert_eq!(
            model.resolve_instruction("Instruct: custom\nQuery: ".to_string()),
            "Instruct: custom\nQuery: "
        );
    }
}
