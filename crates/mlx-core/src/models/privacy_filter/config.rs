//! OpenAI Privacy Filter configuration.
//!
//! Deserializes `config.json` from `openai/privacy-filter` style checkpoints.
//! This config is **internal** to the Rust/native layer — the user-facing TS
//! API is `@mlx-node/privacy`, not the raw config — so this struct does not
//! carry `#[napi(object)]`.

use serde::Deserialize;
use std::collections::BTreeMap;

/// Token-classification config for the OpenAI Privacy Filter family.
///
/// The architecture is an MoE transformer with sliding-window attention
/// and YaRN RoPE, terminating in a per-token classifier head. Only the
/// fields consumed by the privacy-filter implementation are deserialized
/// here — the upstream `config.json` includes additional metadata
/// (`architectures`, `transformers_version`, `transformers.js_config`,
/// etc.) which we intentionally ignore.
#[derive(Debug, Clone, Deserialize)]
pub struct PrivacyFilterConfig {
    /// Architecture identifier — `"openai_privacy_filter"`.
    pub model_type: String,
    /// Model hidden dimension.
    pub hidden_size: usize,
    /// Per-head attention dimension.
    pub head_dim: usize,
    /// Number of attention (query) heads.
    pub num_attention_heads: usize,
    /// Number of key/value heads (GQA).
    pub num_key_value_heads: usize,
    /// Number of transformer decoder layers.
    pub num_hidden_layers: usize,
    /// Total number of MoE experts per MoE layer.
    pub num_local_experts: usize,
    /// Top-k experts routed per token.
    pub num_experts_per_tok: usize,
    /// MoE expert intermediate (FFN) dimension.
    pub intermediate_size: usize,
    /// Sliding-window attention span (tokens). `-1` would disable;
    /// shipped configs use a positive value.
    pub sliding_window: i32,
    /// Whether attention QKV / output projections include a bias term.
    pub attention_bias: bool,
    /// Epsilon for RMSNorm.
    pub rms_norm_eps: f32,
    /// Tokenizer vocabulary size.
    pub vocab_size: usize,
    /// Maximum supported position index.
    pub max_position_embeddings: usize,
    /// Rotary positional embedding (YaRN) parameters.
    pub rope_parameters: RopeParameters,
    /// Map of stringified class index → label name (e.g. `"0" → "O"`,
    /// `"13" → "B-private_email"`). Stored as `BTreeMap` so iteration is
    /// deterministic for downstream consumers.
    pub id2label: BTreeMap<String, String>,
    /// Inverse of [`Self::id2label`] — label name → class index.
    pub label2id: BTreeMap<String, u32>,
    /// Whether the LM head shares weights with the input embedding.
    /// Privacy-filter checkpoints publish `false`, but the field is
    /// optional in upstream configs so we default to `false`.
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

/// YaRN RoPE parameters as stored in the privacy-filter `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    /// RoPE scaling type — `"yarn"` for privacy-filter checkpoints.
    pub rope_type: String,
    /// Base period for rotary embeddings.
    pub rope_theta: f32,
    /// YaRN extrapolation factor.
    pub factor: f32,
    /// YaRN beta_fast (high-frequency boundary).
    pub beta_fast: f32,
    /// YaRN beta_slow (low-frequency boundary).
    pub beta_slow: f32,
    /// Position count at which the model was originally pretrained,
    /// before YaRN extrapolation widened the effective context.
    pub original_max_position_embeddings: usize,
    /// Truncate ramp values outside `[beta_slow, beta_fast]`. Optional
    /// in upstream configs; defaults to `false`.
    #[serde(default)]
    pub truncate: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_real_config_json() {
        let json = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../.cache/models/privacy-filter/config.json"
        ));
        let cfg: PrivacyFilterConfig = serde_json::from_str(json).expect("config.json");

        assert_eq!(cfg.model_type, "openai_privacy_filter");
        assert_eq!(cfg.hidden_size, 640);
        assert_eq!(cfg.head_dim, 64);
        assert_eq!(cfg.num_attention_heads, 14);
        assert_eq!(cfg.num_key_value_heads, 2);
        assert_eq!(cfg.num_hidden_layers, 8);
        assert_eq!(cfg.num_local_experts, 128);
        assert_eq!(cfg.num_experts_per_tok, 4);
        assert_eq!(cfg.intermediate_size, 640);
        assert_eq!(cfg.sliding_window, 128);
        assert!(cfg.attention_bias);
        assert_eq!(cfg.rope_parameters.rope_type, "yarn");
        assert_eq!(cfg.rope_parameters.factor, 32.0);
        assert_eq!(cfg.rope_parameters.original_max_position_embeddings, 4096);
        assert_eq!(cfg.rope_parameters.rope_theta, 150000.0);
        assert_eq!(cfg.id2label.len(), 33);
        assert_eq!(cfg.id2label.get("0").map(String::as_str), Some("O"));
        assert_eq!(
            cfg.id2label.get("13").map(String::as_str),
            Some("B-private_email")
        );
        assert_eq!(cfg.id2label.get("32").map(String::as_str), Some("S-secret"));
    }
}
