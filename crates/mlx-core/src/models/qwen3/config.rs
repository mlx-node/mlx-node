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
}
