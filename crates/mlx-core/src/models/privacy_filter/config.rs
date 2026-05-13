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
    /// Optional explicit per-layer attention type. When absent (as for
    /// privacy-filter `config.json`), the gpt-oss default of alternating
    /// `"sliding_attention"` / `"full_attention"` is used — starting with
    /// `"sliding_attention"` at layer 0. Mirrors
    /// `transformers/models/gpt_oss/configuration_gpt_oss.py:102-105` and
    /// `mlx-lm/mlx_lm/models/gpt_oss.py:197-204`.
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
}

impl PrivacyFilterConfig {
    /// Per-layer attention type for layer `idx`.
    ///
    /// If `config.layer_types` is set, returns the entry at `idx` (or
    /// `"sliding_attention"` as a fallback if the index is out of range —
    /// which would itself be a config bug we surface elsewhere).
    /// Otherwise applies the gpt-oss default: even indices →
    /// `"sliding_attention"`, odd indices → `"full_attention"`.
    ///
    /// Verified against both references:
    /// - `mlx-lm/mlx_lm/models/gpt_oss.py:197-204` —
    ///   `["sliding_attention", "full_attention"] * (num_hidden_layers // 2)`
    ///   ⇒ idx 0 sliding, idx 1 full, ...
    /// - `transformers/models/gpt_oss/configuration_gpt_oss.py:102-105` —
    ///   `"sliding_attention" if bool((i + 1) % 2) else "full_attention"`
    ///   ⇒ i=0 sliding, i=1 full, ...
    pub fn attention_type_for_layer(&self, idx: usize) -> &str {
        if let Some(types) = &self.layer_types {
            return types
                .get(idx)
                .map(String::as_str)
                .unwrap_or("sliding_attention");
        }
        if idx.is_multiple_of(2) {
            "sliding_attention"
        } else {
            "full_attention"
        }
    }

    /// Effective attention band for layer `idx`.
    ///
    /// - `sliding_attention` ⇒ `self.sliding_window`
    /// - `full_attention`    ⇒ an effectively-unbounded band
    ///   (`i32::MAX / 2` — large enough that
    ///   `|q_pos - k_pos| <= band` admits every pair for any plausible
    ///   sequence length while leaving headroom against overflow in
    ///   downstream arithmetic).
    pub fn band_for_layer(&self, idx: usize) -> i32 {
        match self.attention_type_for_layer(idx) {
            "sliding_attention" => self.sliding_window,
            _ => i32::MAX / 2,
        }
    }
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
    /// YaRN `mscale` (numerator of the attention factor). Defaults to
    /// `1.0` — matches HF transformers and mlx-lm when the field is
    /// absent from `config.json` (as it is for privacy-filter).
    #[serde(default = "default_mscale")]
    pub mscale: f32,
    /// YaRN `mscale_all_dim` (denominator of the attention factor).
    /// Defaults to `0.0` — matches HF transformers and mlx-lm when the
    /// field is absent. With this default `_get_mscale(scale, 0.0)` is
    /// always `1.0`, so the denominator vanishes.
    #[serde(default = "default_mscale_all_dim")]
    pub mscale_all_dim: f32,
}

fn default_mscale() -> f32 {
    1.0
}

fn default_mscale_all_dim() -> f32 {
    0.0
}

impl RopeParameters {
    /// YaRN `attention_factor` (a.k.a. `mscale`) — the multiplier applied
    /// to Q and K immediately before the RoPE rotation.
    ///
    /// Mirrors mlx-lm (`YarnRoPE.mscale`, see
    /// `mlx-lm/mlx_lm/models/rope_utils.py:171`) and HF transformers
    /// (`_compute_yarn_parameters`):
    ///
    /// ```text
    /// _get_mscale(scale, m) = 0.1 * m * ln(scale) + 1     if scale > 1
    ///                       = 1                            otherwise
    ///
    /// attention_factor       = _get_mscale(factor, mscale)
    ///                          / _get_mscale(factor, mscale_all_dim)
    /// ```
    ///
    /// For privacy-filter the config sets only `factor = 32`, so the
    /// defaults `mscale = 1.0`, `mscale_all_dim = 0.0` apply and the
    /// result is `0.1 * ln(32) + 1 ≈ 1.3465735902`.
    pub fn attention_factor(&self) -> f32 {
        fn get_mscale(scale: f32, m: f32) -> f32 {
            if scale > 1.0 {
                0.1 * m * scale.ln() + 1.0
            } else {
                1.0
            }
        }
        get_mscale(self.factor, self.mscale) / get_mscale(self.factor, self.mscale_all_dim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal config wired up with the privacy-filter shipping
    /// shape — only the helpers under test need to be meaningful.
    fn dummy_config(layer_types: Option<Vec<String>>) -> PrivacyFilterConfig {
        PrivacyFilterConfig {
            model_type: "openai_privacy_filter".into(),
            hidden_size: 640,
            head_dim: 64,
            num_attention_heads: 14,
            num_key_value_heads: 2,
            num_hidden_layers: 8,
            num_local_experts: 128,
            num_experts_per_tok: 4,
            intermediate_size: 640,
            sliding_window: 128,
            attention_bias: true,
            rms_norm_eps: 1e-5,
            vocab_size: 200_064,
            max_position_embeddings: 4096,
            rope_parameters: RopeParameters {
                rope_type: "yarn".into(),
                rope_theta: 150000.0,
                factor: 32.0,
                beta_fast: 32.0,
                beta_slow: 1.0,
                original_max_position_embeddings: 4096,
                truncate: false,
                mscale: 1.0,
                mscale_all_dim: 0.0,
            },
            id2label: BTreeMap::new(),
            label2id: BTreeMap::new(),
            tie_word_embeddings: false,
            layer_types,
        }
    }

    /// Without an explicit override, layers must alternate starting from
    /// `sliding_attention` at idx 0 — matches both mlx-lm and HF
    /// transformers gpt-oss reference (see [`PrivacyFilterConfig::
    /// attention_type_for_layer`] for citations).
    #[test]
    fn default_layer_types_alternate_sliding_full() {
        let cfg = dummy_config(None);
        for idx in 0usize..8 {
            let expected = if idx.is_multiple_of(2) {
                "sliding_attention"
            } else {
                "full_attention"
            };
            assert_eq!(
                cfg.attention_type_for_layer(idx),
                expected,
                "layer {idx} should be {expected}"
            );
        }
    }

    /// An explicit `layer_types` array wins over the default alternation.
    #[test]
    fn explicit_layer_types_overrides_default() {
        let cfg = dummy_config(Some(vec!["full_attention".into(); 8]));
        for idx in 0..8 {
            assert_eq!(
                cfg.attention_type_for_layer(idx),
                "full_attention",
                "layer {idx} should be full_attention when overridden"
            );
        }
    }

    /// Sliding layers report `sliding_window`; full layers report a band
    /// large enough to admit every (q, k) pair for any plausible
    /// sequence length.
    #[test]
    fn band_for_full_attention_layer_is_unbounded() {
        let cfg = dummy_config(None);
        let band_sliding = cfg.band_for_layer(0);
        let band_full = cfg.band_for_layer(1);
        assert_eq!(band_sliding, 128);
        assert!(
            band_full > 100_000,
            "full-attention band should be effectively unbounded, got {band_full}"
        );
    }

    #[test]
    #[ignore = "requires .cache/models/privacy-filter/config.json — run with --ignored"]
    fn parses_real_config_json() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(".cache/models/privacy-filter/config.json");
        let json = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read config.json from {path:?}: {e}"));
        let cfg: PrivacyFilterConfig = serde_json::from_str(&json).expect("parse config.json");

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

    /// privacy-filter ships `factor=32` and no `mscale` / `mscale_all_dim`
    /// fields, so defaults (`1.0`, `0.0`) apply. Expected attention
    /// factor: `0.1 * 1.0 * ln(32) + 1 ≈ 1.3465735902`. The denominator
    /// `_get_mscale(32, 0.0) = 1.0`, so it has no effect.
    #[test]
    fn attention_factor_for_privacy_filter_defaults() {
        let params = RopeParameters {
            rope_type: "yarn".into(),
            rope_theta: 150000.0,
            factor: 32.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            original_max_position_embeddings: 4096,
            truncate: false,
            mscale: 1.0,
            mscale_all_dim: 0.0,
        };
        let af = params.attention_factor();
        let expected = 0.1f32 * 1.0 * 32.0_f32.ln() + 1.0;
        assert!(
            (af - expected).abs() < 1e-6,
            "got {af}, expected {expected}"
        );
    }

    /// When the extrapolation factor is `1.0` (no YaRN scaling),
    /// `_get_mscale` short-circuits to `1.0` for both numerator and
    /// denominator, so the attention factor must be exactly `1.0`.
    #[test]
    fn attention_factor_returns_1_when_factor_is_1() {
        let params = RopeParameters {
            rope_type: "yarn".into(),
            rope_theta: 150000.0,
            factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            original_max_position_embeddings: 4096,
            truncate: false,
            mscale: 1.0,
            mscale_all_dim: 0.0,
        };
        assert!((params.attention_factor() - 1.0).abs() < 1e-7);
    }
}
