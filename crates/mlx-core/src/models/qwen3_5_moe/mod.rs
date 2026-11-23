// Re-export shared types from qwen3_5 (identical between dense and MoE)
pub use crate::models::qwen3_5::arrays_cache;
pub use crate::models::qwen3_5::attention;
pub use crate::models::qwen3_5::gated_delta;
pub use crate::models::qwen3_5::gated_delta_net;
pub use crate::models::qwen3_5::layer_cache;
pub use crate::models::qwen3_5::rms_norm_gated;
pub use crate::models::qwen3_5::sparse_moe;
pub use crate::models::qwen3_5::switch_glu;
pub(crate) use crate::models::qwen3_5::verify_budget;

// MoE-specific modules
pub mod config;
pub mod decoder_layer;
pub(crate) mod dflash2_decode;
pub mod model;
pub mod mtp;
pub(crate) mod paged_forward;
pub mod persistence;
pub mod quantized_linear;
pub mod switch_linear;

pub use config::Qwen3_5MoeConfig;
pub use model::Qwen3_5MoeModel;
