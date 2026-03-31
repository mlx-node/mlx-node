use napi_derive::napi;

fn default_rope_theta() -> f64 {
    1_000_000.0
}

fn default_rms_norm_eps() -> f64 {
    1e-6
}

fn default_max_position_embeddings() -> i32 {
    32768
}

fn default_use_qk_norm() -> Option<bool> {
    Some(true)
}

/// Configuration for Harrier embedding model (Qwen3 backbone).
///
/// Only includes backbone dimensions needed for encoding.
/// No generation fields (token IDs, paged attention, etc.).
#[napi(object)]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HarrierConfig {
    pub hidden_size: i32,
    #[serde(alias = "num_hidden_layers")]
    pub num_layers: i32,
    #[serde(alias = "num_attention_heads")]
    pub num_heads: i32,
    pub num_key_value_heads: i32,
    pub intermediate_size: i32,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: i32,
    pub head_dim: i32,
    /// Qwen3 always uses QK normalization. Omit to use the default (true).
    /// Explicitly passing false is allowed but produces a model incompatible
    /// with published Harrier weights.
    #[serde(default = "default_use_qk_norm")]
    pub use_qk_norm: Option<bool>,
    pub vocab_size: i32,
}
