/**
 * InternVL Language Model for Qianfan-OCR
 *
 * A Qwen3-4B language model wrapper using TransformerBlock.
 * Structurally identical to Qwen3 (GQA, SiLU MLP, RMSNorm, 1D RoPE, QK norm)
 * but accepts pre-computed embeddings for vision-text merging.
 */
use std::collections::HashMap;

use napi::bindgen_prelude::*;

use crate::array::MxArray;
use crate::models::pp_doclayout_v3::persistence::get_tensor;
use crate::models::qianfan_ocr::config::Qwen3LMConfig;
use crate::nn::{Embedding, Linear, RMSNorm};
use crate::transformer::TransformerBlock;
use crate::transformer::kv_cache::KVCache;

/// InternVL Language Model (Qwen3 architecture)
///
/// Standard transformer with GQA, SiLU MLP, RMSNorm, 1D RoPE, and QK norm.
/// Differs from the standalone Qwen3Model in that it accepts pre-computed
/// embeddings from the vision-text merge step.
pub(crate) struct InternVLLanguageModel {
    embedding: Embedding,
    layers: Vec<TransformerBlock>,
    final_norm: RMSNorm,
    lm_head: Option<Linear>,
    tie_word_embeddings: bool,
    kv_caches: Option<Vec<KVCache>>,
}

impl InternVLLanguageModel {
    /// Build the language model from pre-loaded weights.
    ///
    /// # Arguments
    /// * `weights` - HashMap of weight name to MxArray
    /// * `prefix` - Weight key prefix (e.g., "lm")
    /// * `config` - Qwen3 language model configuration
    pub fn build(
        weights: &HashMap<String, MxArray>,
        prefix: &str,
        config: &Qwen3LMConfig,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size as u32;
        let num_heads = config.num_attention_heads as u32;
        let num_kv_heads = config.num_key_value_heads as u32;
        let intermediate_size = config.intermediate_size as u32;
        let head_dim = config.head_dim as u32;
        let num_layers = config.num_hidden_layers as usize;

        // Load embedding
        let embed_weight = get_tensor(weights, &format!("{prefix}.embedding.weight"))?;
        let embedding = Embedding::from_weight(&embed_weight)?;

        // Build transformer layers
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let layer_prefix = format!("{prefix}.layers.{i}");

            let mut block = TransformerBlock::new(
                hidden_size,
                num_heads,
                num_kv_heads,
                intermediate_size,
                config.rms_norm_eps,
                Some(config.rope_theta),
                Some(config.use_qk_norm),
                Some(head_dim),
            )?;

            // Load attention weights
            let attn = &mut block.self_attn;
            attn.set_q_proj_weight(&get_tensor(
                weights,
                &format!("{layer_prefix}.self_attn.q_proj.weight"),
            )?)?;
            attn.set_k_proj_weight(&get_tensor(
                weights,
                &format!("{layer_prefix}.self_attn.k_proj.weight"),
            )?)?;
            attn.set_v_proj_weight(&get_tensor(
                weights,
                &format!("{layer_prefix}.self_attn.v_proj.weight"),
            )?)?;
            attn.set_o_proj_weight(&get_tensor(
                weights,
                &format!("{layer_prefix}.self_attn.o_proj.weight"),
            )?)?;

            // QK norm weights (if enabled)
            if config.use_qk_norm {
                attn.set_q_norm_weight(&get_tensor(
                    weights,
                    &format!("{layer_prefix}.self_attn.q_norm.weight"),
                )?)?;
                attn.set_k_norm_weight(&get_tensor(
                    weights,
                    &format!("{layer_prefix}.self_attn.k_norm.weight"),
                )?)?;
            }

            // Load MLP weights
            let mlp = &mut block.mlp;
            mlp.set_gate_proj_weight(&get_tensor(
                weights,
                &format!("{layer_prefix}.mlp.gate_proj.weight"),
            )?)?;
            mlp.set_up_proj_weight(&get_tensor(
                weights,
                &format!("{layer_prefix}.mlp.up_proj.weight"),
            )?)?;
            mlp.set_down_proj_weight(&get_tensor(
                weights,
                &format!("{layer_prefix}.mlp.down_proj.weight"),
            )?)?;

            // Load layer norm weights
            block.set_input_layernorm_weight(&get_tensor(
                weights,
                &format!("{layer_prefix}.input_layernorm.weight"),
            )?)?;
            block.set_post_attention_layernorm_weight(&get_tensor(
                weights,
                &format!("{layer_prefix}.post_attention_layernorm.weight"),
            )?)?;

            layers.push(block);
        }

        // Load final norm
        let final_norm_weight = get_tensor(weights, &format!("{prefix}.final_norm.weight"))?;
        let final_norm = RMSNorm::from_weight(&final_norm_weight, Some(config.rms_norm_eps))?;

        // Load lm_head (if not tied)
        let tie_word_embeddings = config.tie_word_embeddings;
        let lm_head = if tie_word_embeddings {
            None
        } else {
            let lm_head_weight = get_tensor(weights, &format!("{prefix}.lm_head.weight"))?;
            Some(Linear::from_weights(&lm_head_weight, None)?)
        };

        Ok(Self {
            embedding,
            layers,
            final_norm,
            lm_head,
            tie_word_embeddings,
            kv_caches: None,
        })
    }

    /// Get token embeddings (for the vision-text merge step).
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs [B, seq_len]
    ///
    /// # Returns
    /// Embeddings [B, seq_len, hidden_size]
    pub fn get_embeddings(&self, input_ids: &MxArray) -> Result<MxArray> {
        self.embedding.forward(input_ids)
    }

    /// Forward pass from pre-computed embeddings (used after vision-text merge).
    ///
    /// # Arguments
    /// * `hidden_states` - Pre-computed embeddings [B, seq_len, hidden_size]
    /// * `cache` - Optional KV caches, one per layer
    ///
    /// # Returns
    /// Logits [B, seq_len, vocab_size]
    pub fn forward_from_embeddings(
        &mut self,
        hidden_states: &MxArray,
        cache: &mut Option<Vec<KVCache>>,
    ) -> Result<MxArray> {
        let mut h = hidden_states.clone();

        // The TransformerBlock handles causal masking internally:
        // - For prefill (seq_len > 1): fused C++ path uses causal SDPA
        // - For decode (seq_len == 1): no mask needed
        // We pass None as the mask and let the block handle it.

        for (i, layer) in self.layers.iter().enumerate() {
            let layer_cache = cache.as_mut().map(|c| &mut c[i]);
            h = layer.forward(&h, None, layer_cache)?;
        }

        // Final norm
        h = self.final_norm.forward(&h)?;

        // LM head
        if self.tie_word_embeddings {
            let embed_weight = self.embedding.get_weight();
            h = h.matmul(&embed_weight.transpose(Some(&[1, 0]))?)?;
        } else if let Some(ref lm_head) = self.lm_head {
            h = lm_head.forward(&h)?;
        } else {
            return Err(Error::from_reason(
                "LM head is None but tie_word_embeddings is false",
            ));
        }

        Ok(h)
    }

    /// Standard forward pass: embed tokens, then run forward_from_embeddings.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs [B, seq_len]
    /// * `cache` - Optional KV caches, one per layer
    ///
    /// # Returns
    /// Logits [B, seq_len, vocab_size]
    pub fn forward(
        &mut self,
        input_ids: &MxArray,
        cache: &mut Option<Vec<KVCache>>,
    ) -> Result<MxArray> {
        let embeddings = self.get_embeddings(input_ids)?;
        self.forward_from_embeddings(&embeddings, cache)
    }

    /// Initialize KV caches for all layers.
    ///
    /// Must be called before starting generation with caching.
    pub fn init_kv_caches(&mut self) {
        self.kv_caches = Some((0..self.layers.len()).map(|_| KVCache::new()).collect());
    }

    /// Reset KV caches, clearing all cached key-value states.
    ///
    /// Call this between different generation sequences.
    pub fn reset_kv_caches(&mut self) {
        if let Some(ref mut caches) = self.kv_caches {
            for cache in caches.iter_mut() {
                cache.reset();
            }
        }
    }

    /// Get a mutable reference to the internal KV caches.
    pub fn kv_caches_mut(&mut self) -> &mut Option<Vec<KVCache>> {
        &mut self.kv_caches
    }

    /// Get the number of transformer layers.
    #[cfg(test)]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get the current cache offset (number of cached tokens).
    pub fn get_cache_offset(&self) -> i32 {
        self.kv_caches
            .as_ref()
            .and_then(|caches| caches.first())
            .map(|cache| cache.get_offset())
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create minimal dummy weights for testing.
    fn make_dummy_weights(prefix: &str, config: &Qwen3LMConfig) -> HashMap<String, MxArray> {
        let mut w = HashMap::new();
        let h = config.hidden_size as i64;
        let inter = config.intermediate_size as i64;
        let n_heads = config.num_attention_heads as i64;
        let n_kv = config.num_key_value_heads as i64;
        let head_dim = config.head_dim as i64;
        let vocab = config.vocab_size as i64;

        // Embedding
        w.insert(
            format!("{prefix}.embedding.weight"),
            MxArray::zeros(&[vocab, h], None).unwrap(),
        );

        // Layers
        for i in 0..config.num_hidden_layers {
            let lp = format!("{prefix}.layers.{i}");
            w.insert(
                format!("{lp}.self_attn.q_proj.weight"),
                MxArray::zeros(&[n_heads * head_dim, h], None).unwrap(),
            );
            w.insert(
                format!("{lp}.self_attn.k_proj.weight"),
                MxArray::zeros(&[n_kv * head_dim, h], None).unwrap(),
            );
            w.insert(
                format!("{lp}.self_attn.v_proj.weight"),
                MxArray::zeros(&[n_kv * head_dim, h], None).unwrap(),
            );
            w.insert(
                format!("{lp}.self_attn.o_proj.weight"),
                MxArray::zeros(&[h, n_heads * head_dim], None).unwrap(),
            );
            if config.use_qk_norm {
                w.insert(
                    format!("{lp}.self_attn.q_norm.weight"),
                    MxArray::ones(&[head_dim], None).unwrap(),
                );
                w.insert(
                    format!("{lp}.self_attn.k_norm.weight"),
                    MxArray::ones(&[head_dim], None).unwrap(),
                );
            }
            w.insert(
                format!("{lp}.mlp.gate_proj.weight"),
                MxArray::zeros(&[inter, h], None).unwrap(),
            );
            w.insert(
                format!("{lp}.mlp.up_proj.weight"),
                MxArray::zeros(&[inter, h], None).unwrap(),
            );
            w.insert(
                format!("{lp}.mlp.down_proj.weight"),
                MxArray::zeros(&[h, inter], None).unwrap(),
            );
            w.insert(
                format!("{lp}.input_layernorm.weight"),
                MxArray::ones(&[h], None).unwrap(),
            );
            w.insert(
                format!("{lp}.post_attention_layernorm.weight"),
                MxArray::ones(&[h], None).unwrap(),
            );
        }

        // Final norm
        w.insert(
            format!("{prefix}.final_norm.weight"),
            MxArray::ones(&[h], None).unwrap(),
        );

        // LM head (only if not tied)
        if !config.tie_word_embeddings {
            w.insert(
                format!("{prefix}.lm_head.weight"),
                MxArray::zeros(&[vocab, h], None).unwrap(),
            );
        }

        w
    }

    /// Small config for fast tests (2 layers, small dimensions).
    fn small_config() -> Qwen3LMConfig {
        Qwen3LMConfig {
            hidden_size: 64,
            num_hidden_layers: 2,
            intermediate_size: 128,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 16,
            rms_norm_eps: 1e-6,
            vocab_size: 32,
            max_position_embeddings: 128,
            rope_theta: 5_000_000.0,
            use_qk_norm: true,
            tie_word_embeddings: false,
        }
    }

    #[test]
    fn test_build_layer_count() {
        let config = small_config();
        let weights = make_dummy_weights("lm", &config);
        let model = InternVLLanguageModel::build(&weights, "lm", &config).unwrap();
        assert_eq!(model.num_layers(), 2);
    }

    #[test]
    fn test_build_tied_embeddings() {
        let mut config = small_config();
        config.tie_word_embeddings = true;
        let weights = make_dummy_weights("lm", &config);
        let model = InternVLLanguageModel::build(&weights, "lm", &config).unwrap();
        assert!(model.lm_head.is_none());
        assert!(model.tie_word_embeddings);
    }

    #[test]
    fn test_get_embeddings_shape() {
        let config = small_config();
        let weights = make_dummy_weights("lm", &config);
        let model = InternVLLanguageModel::build(&weights, "lm", &config).unwrap();

        let input_ids = MxArray::zeros(&[1, 4], Some(crate::array::DType::Int32)).unwrap();
        let embeds = model.get_embeddings(&input_ids).unwrap();
        embeds.eval();

        let shape = embeds.shape().unwrap();
        assert_eq!(shape.as_ref(), &[1, 4, 64]);
    }

    #[test]
    fn test_forward_output_shape() {
        let config = small_config();
        let weights = make_dummy_weights("lm", &config);
        let mut model = InternVLLanguageModel::build(&weights, "lm", &config).unwrap();

        let input_ids = MxArray::zeros(&[1, 4], Some(crate::array::DType::Int32)).unwrap();
        let mut cache = None;
        let logits = model.forward(&input_ids, &mut cache).unwrap();
        logits.eval();

        let shape = logits.shape().unwrap();
        assert_eq!(shape.as_ref(), &[1, 4, 32]); // [B, seq_len, vocab_size]
    }

    #[test]
    fn test_forward_from_embeddings_output_shape() {
        let config = small_config();
        let weights = make_dummy_weights("lm", &config);
        let mut model = InternVLLanguageModel::build(&weights, "lm", &config).unwrap();

        let hidden = MxArray::zeros(&[1, 4, 64], None).unwrap();
        let mut cache = None;
        let logits = model.forward_from_embeddings(&hidden, &mut cache).unwrap();
        logits.eval();

        let shape = logits.shape().unwrap();
        assert_eq!(shape.as_ref(), &[1, 4, 32]); // [B, seq_len, vocab_size]
    }

    #[test]
    fn test_forward_tied_embeddings() {
        let mut config = small_config();
        config.tie_word_embeddings = true;
        let weights = make_dummy_weights("lm", &config);
        let mut model = InternVLLanguageModel::build(&weights, "lm", &config).unwrap();

        let hidden = MxArray::zeros(&[1, 4, 64], None).unwrap();
        let mut cache = None;
        let logits = model.forward_from_embeddings(&hidden, &mut cache).unwrap();
        logits.eval();

        let shape = logits.shape().unwrap();
        assert_eq!(shape.as_ref(), &[1, 4, 32]); // [B, seq_len, vocab_size]
    }

    #[test]
    fn test_kv_cache_init_and_reset() {
        let config = small_config();
        let weights = make_dummy_weights("lm", &config);
        let mut model = InternVLLanguageModel::build(&weights, "lm", &config).unwrap();

        // Initially no caches
        assert!(model.kv_caches.is_none());
        assert_eq!(model.get_cache_offset(), 0);

        // Init caches
        model.init_kv_caches();
        assert!(model.kv_caches.is_some());
        assert_eq!(model.kv_caches.as_ref().unwrap().len(), 2);
        assert_eq!(model.get_cache_offset(), 0);

        // Reset caches
        model.reset_kv_caches();
        assert!(model.kv_caches.is_some());
        assert_eq!(model.get_cache_offset(), 0);
    }

    #[test]
    fn test_forward_with_kv_cache() {
        let config = small_config();
        let weights = make_dummy_weights("lm", &config);
        let mut model = InternVLLanguageModel::build(&weights, "lm", &config).unwrap();

        // Create separate caches to pass into forward
        let mut caches: Option<Vec<KVCache>> = Some(
            (0..config.num_hidden_layers)
                .map(|_| KVCache::new())
                .collect(),
        );

        // Prefill with 4 tokens
        let input_ids = MxArray::zeros(&[1, 4], Some(crate::array::DType::Int32)).unwrap();
        let logits = model.forward(&input_ids, &mut caches).unwrap();
        logits.eval();

        let shape = logits.shape().unwrap();
        assert_eq!(shape.as_ref(), &[1, 4, 32]);

        // Verify cache was populated
        let offset = caches.as_ref().unwrap()[0].get_offset();
        assert_eq!(offset, 4);
    }

    #[test]
    fn test_build_missing_weight_fails() {
        let config = small_config();
        let weights: HashMap<String, MxArray> = HashMap::new(); // empty
        let result = InternVLLanguageModel::build(&weights, "lm", &config);
        assert!(result.is_err());
    }
}
