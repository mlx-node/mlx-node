/**
 * Qwen3 Model Configuration
 */
use napi_derive::napi;

/// Qwen3 model configuration
#[napi(object)]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Qwen3Config {
    pub vocab_size: i32,
    pub hidden_size: i32,
    pub num_layers: i32,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub intermediate_size: i32,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: i32,
    pub head_dim: i32, // Dimension per attention head (e.g., 128 for Qwen3-0.6B)
    pub use_qk_norm: bool,
    pub tie_word_embeddings: bool,
    pub pad_token_id: i32,
    pub eos_token_id: i32,
    pub bos_token_id: i32,

    // Paged attention options (opt-in)
    /// Enable paged attention for memory-efficient inference.
    /// Default: false (use standard KVCache)
    #[serde(default)]
    #[napi(ts_type = "boolean | undefined")]
    pub use_paged_attention: Option<bool>,

    /// GPU memory budget for paged KV cache in megabytes.
    /// Only used when use_paged_attention is true.
    /// Default: 2048 (2GB)
    #[serde(default)]
    #[napi(ts_type = "number | undefined")]
    pub paged_cache_memory_mb: Option<u32>,

    /// Block size for paged attention (tokens per block).
    /// Only used when use_paged_attention is true.
    /// Default: 16
    #[serde(default)]
    #[napi(ts_type = "number | undefined")]
    pub paged_block_size: Option<u32>,

    /// Use FP8 cache for 2x memory reduction (experimental).
    /// Only used when use_paged_attention is true.
    /// Default: false
    #[serde(default)]
    #[napi(ts_type = "boolean | undefined")]
    pub use_fp8_cache: Option<bool>,
}
