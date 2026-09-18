use napi_derive::napi;

use crate::models::paged_config::PagedCacheConfig;

fn default_rms_norm_eps() -> f64 {
    1e-6
}

fn default_layernorm_num_groups() -> i32 {
    4
}

fn default_rope_theta() -> f64 {
    10_000_000.0
}

/// Nested RoPE parameter block (`config.json["rope_parameters"]`).
///
/// K2-Horizon nests `rope_theta` under `rope_parameters` instead of the
/// flat `rope_theta` key the other families use:
/// `"rope_parameters": {"rope_theta": 10000000.0, "rope_type": "default"}`.
#[napi(object)]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct K2RopeParameters {
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    /// `default` (full RoPE) is the only supported type; anything else is
    /// ignored here and rejected by `K2HorizonConfig::validate`.
    #[serde(default)]
    pub rope_type: Option<String>,
}

/// K2-Horizon model configuration (`model_type: "k2_horizon"`).
///
/// Dense decoder-only transformer (IFM K2-Horizon-7B): GQA attention,
/// SwiGLU MLP, grouped RMSNorm (`layernorm_num_groups`), full RoPE,
/// untied embeddings. No MoE, no conv/recurrent state, no MTP head —
/// a pure standard-KV transformer.
#[napi(object)]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct K2HorizonConfig {
    pub vocab_size: i32,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub intermediate_size: i32,
    pub max_position_embeddings: i32,
    /// Per-head dimension. K2 ships `head_dim` explicitly; fall back to
    /// `hidden_size / num_attention_heads` when absent.
    #[serde(default)]
    pub head_dim: Option<i32>,
    /// RMSNorm epsilon (`rms_norm_eps` in config.json).
    #[serde(default = "default_rms_norm_eps", rename = "rms_norm_eps")]
    pub norm_eps: f64,
    /// Number of groups the grouped RMSNorm partitions `hidden_size` into
    /// (variance computed per `hidden_size / layernorm_num_groups` group,
    /// not over the whole row). K2 ships `layernorm_num_groups: 4`.
    /// `1` degenerates to plain RMSNorm.
    #[serde(default = "default_layernorm_num_groups")]
    pub layernorm_num_groups: i32,
    /// Flat `rope_theta` key (other families' convention). K2 nests it
    /// under `rope_parameters`; both are accepted, nested wins.
    #[serde(default)]
    pub rope_theta: Option<f64>,
    /// Nested RoPE block; `rope_theta` resolved via [`Self::rope_theta`].
    #[serde(default)]
    pub rope_parameters: Option<K2RopeParameters>,
    /// Whether `lm_head` shares `embed_tokens` weights. K2 ships `false`
    /// (a separate `lm_head.weight` tensor). Defaults false rather than
    /// the HF `PretrainedConfig` convention of true: an omitted flag on
    /// a checkpoint that still ships `lm_head.weight` must stay untied,
    /// and one that omits BOTH flag and tensor fails the mandatory-weight
    /// check loudly instead of silently binding random-init logits.
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub eos_token_id: serde_json::Value,
    #[serde(default)]
    pub bos_token_id: i32,

    // Paged attention options (default on; mirror qwen3/lfm2 knobs).
    /// GPU memory budget for paged KV cache in megabytes.
    /// Only used when `use_block_paged_cache` is true.
    /// Default: adaptive (weights-aware) sizing.
    #[serde(default)]
    #[napi(ts_type = "number | undefined")]
    pub paged_cache_memory_mb: Option<u32>,

    /// Block size for paged attention (tokens per block).
    /// Only used when `use_block_paged_cache` is true.
    /// Default: 16.
    #[serde(default)]
    #[napi(ts_type = "number | undefined")]
    pub paged_block_size: Option<u32>,

    /// Use the block-paged KV cache adapter (`PagedKVCacheAdapter`).
    ///
    /// Default: `true` — K2 is a pure standard-KV transformer, so every
    /// layer routes through the adapter. Opt out with
    /// `use_block_paged_cache: Some(false)` for the flat `KVCache` path.
    #[serde(default)]
    #[napi(ts_type = "boolean | undefined")]
    pub use_block_paged_cache: Option<bool>,

    /// Persist block-paged attention state to the SSD cold tier. Explicit
    /// config overrides the process-wide `MLX_PERSIST_PAGED_CACHE` default.
    /// Currently inert: `k2_horizon` is not in `COLD_RESTORE_FAMILIES`, so
    /// `resolve_persist_cold` fails closed and no state is written until the
    /// family is allowlisted (parity-gated — see `cold_tier.rs`).
    #[serde(default)]
    #[napi(ts_type = "boolean | undefined")]
    pub persist_paged_cache: Option<bool>,
}

impl K2HorizonConfig {
    /// Effective head dimension: explicit `head_dim` wins, else
    /// `hidden_size / num_attention_heads`.
    pub fn head_dim(&self) -> i32 {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Effective RoPE base frequency: nested `rope_parameters.rope_theta`
    /// wins, then the flat `rope_theta` key, then the K2 default (1e7).
    pub fn rope_theta(&self) -> f64 {
        self.rope_parameters
            .as_ref()
            .map(|p| p.rope_theta)
            .or(self.rope_theta)
            .unwrap_or_else(default_rope_theta)
    }

    /// Resolve the load-time default for `use_block_paged_cache`.
    ///
    /// Policy (pure, no I/O — isolated here for unit testing): an explicit
    /// `Some(_)` from config.json always wins; absent (`None`) defaults to
    /// `Some(true)` — K2 is pure standard-KV, so the paged path is the
    /// production architecture for both dense and quantized checkpoints.
    pub fn resolve_use_block_paged_default(explicit: Option<bool>) -> Option<bool> {
        PagedCacheConfig::resolve_use_paged_default(explicit, true)
    }

    /// Structural validation beyond serde shape checks. Called once at load
    /// so malformed geometry fails before any weight is touched.
    pub fn validate(&self) -> napi::bindgen_prelude::Result<()> {
        if self.hidden_size <= 0
            || self.num_hidden_layers <= 0
            || self.num_attention_heads <= 0
            || self.num_key_value_heads <= 0
            || self.intermediate_size <= 0
            || self.vocab_size <= 0
        {
            return Err(napi::Error::from_reason(format!(
                "k2_horizon config has non-positive geometry: hidden_size={}, layers={}, \
                 heads={}, kv_heads={}, intermediate={}, vocab={}",
                self.hidden_size,
                self.num_hidden_layers,
                self.num_attention_heads,
                self.num_key_value_heads,
                self.intermediate_size,
                self.vocab_size,
            )));
        }
        let head_dim = self.head_dim();
        if head_dim <= 0 {
            return Err(napi::Error::from_reason(format!(
                "k2_horizon config: head_dim ({head_dim}) must be positive"
            )));
        }
        if self.num_attention_heads % self.num_key_value_heads != 0 {
            return Err(napi::Error::from_reason(format!(
                "k2_horizon config: num_attention_heads ({}) must be a multiple of \
                 num_key_value_heads ({})",
                self.num_attention_heads, self.num_key_value_heads,
            )));
        }
        if self.layernorm_num_groups <= 0 || self.hidden_size % self.layernorm_num_groups != 0 {
            return Err(napi::Error::from_reason(format!(
                "k2_horizon config: hidden_size ({}) must be divisible by \
                 layernorm_num_groups ({})",
                self.hidden_size, self.layernorm_num_groups,
            )));
        }
        if let Some(rope_type) = self
            .rope_parameters
            .as_ref()
            .and_then(|p| p.rope_type.as_deref())
            && rope_type != "default"
        {
            return Err(napi::Error::from_reason(format!(
                "k2_horizon config: unsupported rope_type '{rope_type}' (only 'default' is implemented)"
            )));
        }
        if self.use_block_paged_cache.unwrap_or(true)
            && let Some(block_size) = self.paged_block_size
            && ![8, 16, 32].contains(&block_size)
        {
            return Err(napi::Error::from_reason(format!(
                "k2_horizon config: paged_block_size ({block_size}) must be 8, 16, or 32"
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mirror of the shipped `k2-horizon-7b-fp8` config.json (geometry +
    /// grouped-norm fields; quant block omitted — it is ignored by serde).
    #[test]
    fn test_deserialize_shipped_config() {
        let json = r#"{
            "model_type": "k2_horizon",
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 250624,
            "max_position_embeddings": 524288,
            "layernorm_num_groups": 4,
            "rms_norm_eps": 1e-6,
            "rope_head_dim": 128,
            "rope_parameters": {"rope_theta": 10000000.0, "rope_type": "default"},
            "query_key_norm": false,
            "tie_word_embeddings": false,
            "eos_token_id": 1,
            "bos_token_id": 0
        }"#;
        let cfg: K2HorizonConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_hidden_layers, 36);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.head_dim(), 128);
        assert_eq!(cfg.layernorm_num_groups, 4);
        assert!((cfg.norm_eps - 1e-6).abs() < 1e-12);
        assert!((cfg.rope_theta() - 10_000_000.0).abs() < 1e-3);
        assert!(!cfg.tie_word_embeddings);
        cfg.validate().unwrap();
    }

    #[test]
    fn test_rope_theta_resolution_order() {
        // Nested wins over flat; flat wins over the 1e7 default.
        let mut cfg: K2HorizonConfig = serde_json::from_str(
            r#"{"vocab_size": 8, "hidden_size": 8, "num_hidden_layers": 1,
                "num_attention_heads": 1, "num_key_value_heads": 1,
                "intermediate_size": 8, "max_position_embeddings": 8,
                "rope_theta": 500000.0}"#,
        )
        .unwrap();
        assert!((cfg.rope_theta() - 500_000.0).abs() < 1e-3);
        cfg.rope_parameters = Some(K2RopeParameters {
            rope_theta: 10_000_000.0,
            rope_type: None,
        });
        assert!((cfg.rope_theta() - 10_000_000.0).abs() < 1e-3);
    }

    #[test]
    fn test_validate_rejects_indivisible_groups() {
        let cfg: K2HorizonConfig = serde_json::from_str(
            r#"{"vocab_size": 8, "hidden_size": 10, "num_hidden_layers": 1,
                "num_attention_heads": 1, "num_key_value_heads": 1,
                "intermediate_size": 8, "max_position_embeddings": 8,
                "layernorm_num_groups": 4}"#,
        )
        .unwrap();
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_non_positive_head_dim() {
        let mut cfg: K2HorizonConfig = serde_json::from_str(
            r#"{"vocab_size": 8, "hidden_size": 8, "num_hidden_layers": 1,
                "num_attention_heads": 1, "num_key_value_heads": 1,
                "intermediate_size": 8, "max_position_embeddings": 8}"#,
        )
        .unwrap();
        for enabled in [None, Some(true), Some(false)] {
            cfg.use_block_paged_cache = enabled;
            for head_dim in [0, -1, i32::MIN] {
                cfg.head_dim = Some(head_dim);
                assert_eq!(
                    cfg.validate().unwrap_err().reason,
                    format!("k2_horizon config: head_dim ({head_dim}) must be positive")
                );
            }
        }
    }

    #[test]
    fn test_validate_rejects_zero_inferred_head_dim() {
        let mut cfg: K2HorizonConfig = serde_json::from_str(
            r#"{"vocab_size": 8, "hidden_size": 8, "num_hidden_layers": 1,
                "num_attention_heads": 16, "num_key_value_heads": 1,
                "intermediate_size": 8, "max_position_embeddings": 8}"#,
        )
        .unwrap();
        assert_eq!(cfg.head_dim(), 0);
        for enabled in [None, Some(true), Some(false)] {
            cfg.use_block_paged_cache = enabled;
            assert_eq!(
                cfg.validate().unwrap_err().reason,
                "k2_horizon config: head_dim (0) must be positive"
            );
        }
    }

    #[test]
    fn test_validate_preserves_head_dim_resolution_and_geometry_guard() {
        let mut cfg: K2HorizonConfig = serde_json::from_str(
            r#"{"vocab_size": 8, "hidden_size": 8, "num_hidden_layers": 1,
                "num_attention_heads": 1, "num_key_value_heads": 1,
                "intermediate_size": 8, "max_position_embeddings": 8}"#,
        )
        .unwrap();
        for enabled in [None, Some(true), Some(false)] {
            cfg.use_block_paged_cache = enabled;
            cfg.num_attention_heads = 1;
            cfg.head_dim = None;
            assert_eq!(cfg.head_dim(), 8);
            cfg.validate().unwrap();
            cfg.num_attention_heads = 16;
            cfg.head_dim = Some(128);
            assert_eq!(cfg.head_dim(), 128);
            cfg.validate().unwrap();
            for head_dim in [None, Some(128)] {
                cfg.head_dim = head_dim;
                for heads in [0, -1] {
                    cfg.num_attention_heads = heads;
                    assert!(
                        cfg.validate()
                            .unwrap_err()
                            .reason
                            .contains("non-positive geometry")
                    );
                }
            }
        }
    }

    #[test]
    fn test_validate_paged_block_size_before_pool_sizing() {
        let mut cfg: K2HorizonConfig = serde_json::from_str(
            r#"{"vocab_size": 8, "hidden_size": 8, "num_hidden_layers": 1,
                "num_attention_heads": 1, "num_key_value_heads": 1,
                "intermediate_size": 8, "max_position_embeddings": 8}"#,
        )
        .unwrap();
        for enabled in [None, Some(true), Some(false)] {
            cfg.use_block_paged_cache = enabled;
            for size in [None, Some(8), Some(16), Some(32)] {
                cfg.paged_block_size = size;
                cfg.validate().unwrap();
            }
            for size in [0, 1, 7, 64] {
                cfg.paged_block_size = Some(size);
                if enabled == Some(false) {
                    cfg.validate().unwrap();
                } else {
                    assert!(
                        cfg.validate()
                            .unwrap_err()
                            .reason
                            .contains("paged_block_size")
                    );
                }
            }
        }
    }

    #[test]
    fn test_resolve_use_block_paged_default() {
        assert_eq!(
            K2HorizonConfig::resolve_use_block_paged_default(None),
            Some(true)
        );
        assert_eq!(
            K2HorizonConfig::resolve_use_block_paged_default(Some(false)),
            Some(false)
        );
    }
}
