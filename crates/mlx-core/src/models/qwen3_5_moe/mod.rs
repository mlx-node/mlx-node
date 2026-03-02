// Re-export shared types from qwen3_5 (identical between dense and MoE)
pub use crate::models::qwen3_5::arrays_cache;
pub use crate::models::qwen3_5::attention;
pub use crate::models::qwen3_5::gated_delta;
pub use crate::models::qwen3_5::gated_delta_net;
pub use crate::models::qwen3_5::layer_cache;
pub use crate::models::qwen3_5::rms_norm_gated;

// MoE-specific modules
pub mod config;
pub mod decoder_layer;
pub mod model;
pub mod persistence;
pub mod quantized_linear;
pub mod sparse_moe;
pub mod switch_glu;
pub mod switch_linear;

pub use config::Qwen3_5MoeConfig;
pub use model::Qwen3_5MoeModel;
