use napi::{Error, Result};
use serde::{Deserialize, Serialize};

/// Qwen3.8-Flash-Next's Qwen4 experimental decoder. PLE layer ids are one-based
/// in Hugging Face config and zero-based in GGUF metadata.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Config {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    pub hc_count: usize,
    pub hc_lowrank: usize,
    pub rms_norm_eps: f64,
    pub max_position_embeddings: usize,
    pub full_attention_interval: usize,
    pub indexer_n_heads: usize,
    #[serde(default = "one")]
    pub indexer_kv_heads: usize,
    pub indexer_head_dim: usize,
    pub indexer_budget: usize,
    pub indexer_compress_ratio: usize,
    pub ple_layer_ids: Vec<usize>,
    pub ple_embed_dim: usize,
    pub ple_conv_kernel_size: usize,
    pub ngram_size: usize,
    pub heads_per_ngram: usize,
    pub ngram_vocab_size_base: u64,
    pub split_ngram_parts: usize,
    pub eos_token_id: u32,
    #[serde(default = "seed")]
    pub seed: u64,
    #[serde(default)]
    pub layer_types: Vec<String>,
    #[serde(default = "gate")]
    pub output_gate_type: String,
    #[serde(default = "silu")]
    pub hidden_act: String,
    #[serde(default = "yes")]
    pub norm_topk_prob: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub rope_parameters: serde_json::Value,
}
fn seed() -> u64 {
    1234
}
fn gate() -> String {
    "sigmoid".into()
}
fn one() -> usize {
    1
}
fn yes() -> bool {
    true
}
fn silu() -> String {
    "silu".into()
}

impl Config {
    pub fn parse(raw: &serde_json::Value) -> Result<Self> {
        let model_type = raw["model_type"].as_str().unwrap_or("");
        if !matches!(model_type, "qwen4_exp" | "qwen4_exp_text") {
            return Err(Error::from_reason(
                "Expected qwen4_exp or qwen4_exp_text config",
            ));
        }
        let c: Self = serde_json::from_value(raw.get("text_config").unwrap_or(raw).clone())
            .map_err(|e| Error::from_reason(format!("Invalid qwen4_exp config: {e}")))?;
        c.validate()?;
        Ok(c)
    }
    pub fn validate(&self) -> Result<()> {
        if self.indexer_kv_heads != 1
            || self.hidden_act != "silu"
            || !self.norm_topk_prob
            || self.tie_word_embeddings
            || self.attention_bias
            || self
                .rope_parameters
                .get("type")
                .or_else(|| self.rope_parameters.get("rope_type"))
                .is_some_and(|v| v.as_str() != Some("default"))
        {
            return Err(Error::from_reason(
                "Unsupported qwen4_exp indexer, MLP, attention bias, tied head, or scaled RoPE configuration",
            ));
        }
        let positive = [
            self.hidden_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.vocab_size,
            self.num_experts,
            self.num_experts_per_tok,
            self.moe_intermediate_size,
            self.shared_expert_intermediate_size,
            self.linear_num_key_heads,
            self.linear_num_value_heads,
            self.linear_key_head_dim,
            self.linear_value_head_dim,
            self.linear_conv_kernel_dim,
            self.hc_lowrank,
            self.max_position_embeddings,
            self.full_attention_interval,
            self.indexer_n_heads,
            self.indexer_head_dim,
            self.indexer_budget,
            self.indexer_compress_ratio,
            self.ple_embed_dim,
            self.ple_conv_kernel_size,
            self.heads_per_ngram,
            self.split_ngram_parts,
        ];
        if positive.contains(&0)
            || positive.iter().any(|&x| x > 1_048_576)
            || !(2..=16).contains(&self.hc_count)
            || !(2..=8).contains(&self.ngram_size)
            || self.num_experts_per_tok > self.num_experts
            || !self
                .linear_num_value_heads
                .is_multiple_of(self.linear_num_key_heads)
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
            || !self
                .indexer_budget
                .is_multiple_of(self.indexer_compress_ratio)
            || !self
                .ple_embed_dim
                .is_multiple_of((self.ngram_size - 1) * self.heads_per_ngram)
            || self
                .ple_layer_ids
                .iter()
                .any(|&i| i == 0 || i > self.num_hidden_layers)
            || self.ple_layer_ids.windows(2).any(|w| w[0] >= w[1])
            || !self.rms_norm_eps.is_finite()
            || self.rms_norm_eps <= 0.0
            || !matches!(self.output_gate_type.as_str(), "sigmoid" | "silu")
            || self.eos_token_id as usize >= self.vocab_size
            || self.ngram_vocab_size_base < 2
            || (!self.layer_types.is_empty()
                && (self.layer_types.len() != self.num_hidden_layers
                    || self.layer_types.iter().any(|s| {
                        !matches!(
                            s.as_str(),
                            "linear_attention" | "full_attention" | "qwen_sparse_attention"
                        )
                    })))
        {
            return Err(Error::from_reason(
                "Invalid qwen4_exp dimensions, layer types, or PLE configuration",
            ));
        }
        let recurrent_bytes = self.num_hidden_layers as u128
            * self.linear_num_value_heads as u128
            * self.linear_key_head_dim as u128
            * self.linear_value_head_dim as u128
            * 4;
        if self.num_hidden_layers > 512
            || recurrent_bytes > (1u128 << 30)
            || self.ple_layer_ids.iter().any(|&i| !self.linear(i - 1))
            || self.effective_context_limit() == 0
        {
            return Err(Error::from_reason(
                "qwen4_exp configuration exceeds the recurrent/cache safety budget or places PLE on a non-linear layer",
            ));
        }
        let dims = self.rope_dims();
        if dims == 0
            || !dims.is_multiple_of(2)
            || dims > self.head_dim
            || dims > self.indexer_head_dim
            || !self.rope_theta().is_finite()
            || self.rope_theta() <= 0.0
        {
            return Err(Error::from_reason(
                "Invalid qwen4_exp rotary dimensions/base",
            ));
        }
        Ok(())
    }
    pub fn linear(&self, layer: usize) -> bool {
        self.layer_types.get(layer).map_or(
            !(layer + 1).is_multiple_of(self.full_attention_interval),
            |kind| kind == "linear_attention",
        )
    }
    /// Reserve at most 2 GiB for attention/index caches, conservatively sizing
    /// every value as float32 and reserving room for the append copy.
    pub fn effective_context_limit(&self) -> usize {
        let full = (0..self.num_hidden_layers)
            .filter(|&i| !self.linear(i))
            .count();
        let per_token = full as u128
            * (2 * self.num_key_value_heads as u128 * self.head_dim as u128 * 4
                + self.indexer_head_dim as u128 * 4);
        self.max_position_embeddings
            .min(((2u128 << 30) / per_token.max(1)).min(32_768) as usize)
    }
    pub fn rope_dims(&self) -> usize {
        (self.head_dim as f64
            * self.rope_parameters["partial_rotary_factor"]
                .as_f64()
                .unwrap_or(0.25)) as usize
    }
    pub fn rope_theta(&self) -> f64 {
        self.rope_parameters["rope_theta"]
            .as_f64()
            .unwrap_or(10_000_000.0)
    }
}
