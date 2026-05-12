use napi_derive::napi;

fn default_true() -> bool {
    true
}

fn default_rope_theta() -> f64 {
    1_000_000.0
}

fn default_block_multiple_of() -> i32 {
    256
}

fn default_block_ffn_dim_multiplier() -> f64 {
    1.0
}

/// LFM2 model configuration.
///
/// Supports LiquidAI's LFM2.5 hybrid conv+attention architecture.
/// 16 layers total: 10 conv + 6 full_attention, defined by `layer_types` array.
#[napi(object)]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Lfm2Config {
    pub vocab_size: i32,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub max_position_embeddings: i32,
    pub norm_eps: f64,
    pub conv_bias: bool,
    #[serde(rename = "conv_L_cache")]
    pub conv_l_cache: i32,
    #[serde(default)]
    pub block_dim: i32,
    #[serde(default)]
    pub block_ff_dim: i32,
    #[serde(default = "default_block_multiple_of")]
    pub block_multiple_of: i32,
    #[serde(default = "default_block_ffn_dim_multiplier")]
    pub block_ffn_dim_multiplier: f64,
    #[serde(default)]
    pub block_auto_adjust_ff_dim: bool,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    pub layer_types: Vec<String>,
    #[serde(default = "default_true")]
    pub tie_embedding: bool,
    #[serde(default)]
    pub eos_token_id: i32,
    #[serde(default)]
    pub bos_token_id: i32,
    #[serde(default)]
    pub pad_token_id: i32,

    // Paged attention options (opt-in, mirror Qwen3/Gemma4 knobs).
    /// GPU memory budget for paged KV cache in megabytes.
    /// Only used when `use_block_paged_cache` is true.
    /// Default: 2048 (2GB).
    #[serde(default)]
    #[napi(ts_type = "number | undefined")]
    pub paged_cache_memory_mb: Option<u32>,

    /// Block size for paged attention (tokens per block).
    /// Only used when `use_block_paged_cache` is true.
    /// Default: 16.
    #[serde(default)]
    #[napi(ts_type = "number | undefined")]
    pub paged_block_size: Option<u32>,

    /// Use the new block-paged KV cache adapter (`PagedKVCacheAdapter`).
    ///
    /// Default: `true` since 2026-04-28 (parity-verified via
    /// `crates/mlx-core/tests/lfm2_paged_vs_flat_parity.rs` against real
    /// LFM2.5-1.2B weights: byte-equal greedy decode + prefix-reuse
    /// byte-equal at BF16). Wired through
    /// `Lfm2DecoderLayer::forward_paged_or_flat`.
    ///
    /// Per-layer routing: LFM2's hybrid architecture means only
    /// `full_attention` layers go through the paged adapter; conv layers
    /// stay on the existing flat `Lfm2LayerCache::Conv(ArraysCache)`
    /// storage regardless of this flag. The `LayerKVPool` is sized to
    /// the count of `full_attention` layers and indexed by
    /// attention-ordinal (via `config.full_attn_idxs()`), not by absolute
    /// layer index.
    ///
    /// Opt out with `use_block_paged_cache: Some(false)` to revert to the
    /// fully flat `Lfm2LayerCache` path on all layers.
    #[serde(default)]
    #[napi(ts_type = "boolean | undefined")]
    pub use_block_paged_cache: Option<bool>,
}

impl Lfm2Config {
    /// Whether the layer at `idx` is a full_attention layer.
    pub fn is_attention_layer(&self, idx: usize) -> bool {
        self.layer_types
            .get(idx)
            .is_some_and(|t| t == "full_attention")
    }

    /// Compute the effective feed-forward dimension.
    ///
    /// Matches Python MLP.__init__:
    /// ```python
    /// if auto_adjust_ff_dim:
    ///     ff_dim = int(2 * ff_dim / 3)
    ///     if ffn_dim_multiplier is not None:
    ///         ff_dim = int(ffn_dim_multiplier * ff_dim)
    ///     ff_dim = multiple_of * ((ff_dim + multiple_of - 1) // multiple_of)
    /// ```
    pub fn computed_ff_dim(&self) -> i32 {
        if self.block_auto_adjust_ff_dim {
            let mut ff = (2 * self.block_ff_dim) / 3;
            ff = (self.block_ffn_dim_multiplier * ff as f64) as i32;
            let m = self.block_multiple_of;
            m * ((ff + m - 1) / m)
        } else {
            self.block_ff_dim
        }
    }

    /// Head dimension: hidden_size / num_attention_heads.
    pub fn head_dim(&self) -> i32 {
        self.hidden_size / self.num_attention_heads
    }

    /// Indices of full_attention layers.
    pub fn full_attn_idxs(&self) -> Vec<usize> {
        self.layer_types
            .iter()
            .enumerate()
            .filter_map(|(i, t)| if t == "full_attention" { Some(i) } else { None })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> Lfm2Config {
        Lfm2Config {
            vocab_size: 65536,
            hidden_size: 2048,
            num_hidden_layers: 16,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            max_position_embeddings: 128000,
            norm_eps: 1e-5,
            conv_bias: false,
            conv_l_cache: 3,
            block_dim: 2048,
            block_ff_dim: 12288,
            block_multiple_of: 256,
            block_ffn_dim_multiplier: 1.0,
            block_auto_adjust_ff_dim: true,
            rope_theta: 1_000_000.0,
            layer_types: vec![
                "conv".into(),
                "conv".into(),
                "full_attention".into(),
                "conv".into(),
                "conv".into(),
                "full_attention".into(),
                "conv".into(),
                "conv".into(),
                "full_attention".into(),
                "conv".into(),
                "full_attention".into(),
                "conv".into(),
                "full_attention".into(),
                "conv".into(),
                "full_attention".into(),
                "conv".into(),
            ],
            tie_embedding: true,
            eos_token_id: 7,
            bos_token_id: 1,
            pad_token_id: 0,
            paged_cache_memory_mb: None,
            paged_block_size: None,
            use_block_paged_cache: None,
        }
    }

    #[test]
    fn test_is_attention_layer() {
        let cfg = test_config();
        assert!(!cfg.is_attention_layer(0)); // conv
        assert!(!cfg.is_attention_layer(1)); // conv
        assert!(cfg.is_attention_layer(2)); // full_attention
        assert!(!cfg.is_attention_layer(3)); // conv
        assert!(cfg.is_attention_layer(5)); // full_attention
    }

    #[test]
    fn test_computed_ff_dim() {
        let cfg = test_config();
        // int(2 * 12288 / 3) = 8192
        // 8192 * 1.0 = 8192
        // 256 * ((8192 + 255) / 256) = 256 * 33 = 8448? No:
        // (8192 + 256 - 1) / 256 = 8447 / 256 = 32 (integer division)
        // 256 * 32 = 8192
        assert_eq!(cfg.computed_ff_dim(), 8192);
    }

    #[test]
    fn test_head_dim() {
        let cfg = test_config();
        assert_eq!(cfg.head_dim(), 64); // 2048 / 32
    }

    #[test]
    fn test_full_attn_idxs() {
        let cfg = test_config();
        assert_eq!(cfg.full_attn_idxs(), vec![2, 5, 8, 10, 12, 14]);
    }

    #[test]
    fn test_deserialize_config() {
        let json = r#"{
            "vocab_size": 65536,
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "max_position_embeddings": 128000,
            "norm_eps": 1e-5,
            "conv_bias": false,
            "conv_L_cache": 3,
            "block_dim": 2048,
            "block_ff_dim": 12288,
            "block_multiple_of": 256,
            "block_ffn_dim_multiplier": 1.0,
            "block_auto_adjust_ff_dim": true,
            "rope_theta": 1000000.0,
            "layer_types": ["conv","conv","full_attention","conv","conv","full_attention","conv","conv","full_attention","conv","full_attention","conv","full_attention","conv","full_attention","conv"],
            "tie_embedding": true,
            "eos_token_id": 7,
            "bos_token_id": 1,
            "pad_token_id": 0
        }"#;
        let cfg: Lfm2Config = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.conv_l_cache, 3);
        assert_eq!(cfg.vocab_size, 65536);
        assert_eq!(cfg.layer_types.len(), 16);
    }

    /// `use_block_paged_cache` defaults to `None` when absent from the JSON
    /// config — guards against silently switching the storage backend on
    /// existing LFM2 checkpoints.
    #[test]
    fn test_use_block_paged_cache_defaults_to_none_via_serde() {
        let json = r#"{
            "vocab_size": 100,
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "max_position_embeddings": 128,
            "norm_eps": 1e-5,
            "conv_bias": false,
            "conv_L_cache": 3,
            "block_dim": 64,
            "block_ff_dim": 64,
            "layer_types": ["conv", "full_attention"]
        }"#;
        let cfg: Lfm2Config = serde_json::from_str(json).unwrap();
        assert_eq!(
            cfg.use_block_paged_cache, None,
            "use_block_paged_cache must default to None on JSON without the key"
        );
        assert_eq!(cfg.paged_block_size, None);
        assert_eq!(cfg.paged_cache_memory_mb, None);
    }

    /// `use_block_paged_cache: true` round-trips through serde.
    #[test]
    fn test_use_block_paged_cache_round_trips_true() {
        let json = r#"{
            "vocab_size": 100,
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "max_position_embeddings": 128,
            "norm_eps": 1e-5,
            "conv_bias": false,
            "conv_L_cache": 3,
            "block_dim": 64,
            "block_ff_dim": 64,
            "layer_types": ["conv", "full_attention"],
            "use_block_paged_cache": true,
            "paged_block_size": 16,
            "paged_cache_memory_mb": 256
        }"#;
        let cfg: Lfm2Config = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.use_block_paged_cache, Some(true));
        assert_eq!(cfg.paged_block_size, Some(16));
        assert_eq!(cfg.paged_cache_memory_mb, Some(256));
    }
}
