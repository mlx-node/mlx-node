// Re-export shared types from qwen3_5 (identical between dense and MoE)
pub use crate::models::qwen3_5::arrays_cache;
pub use crate::models::qwen3_5::attention;
pub use crate::models::qwen3_5::gated_delta;
pub use crate::models::qwen3_5::gated_delta_net;
pub use crate::models::qwen3_5::layer_cache;
pub use crate::models::qwen3_5::rms_norm_gated;

// MoE-specific modules.
//
// The browser runs Qwen3.5-0.8B **dense** only; the MoE runtime modules pull in
// `crate::moe` (native-only) via `sparse_moe` / `switch_glu`. On wasm we keep
// just `config` — the sole piece wasm code needs (`Qwen3_5MoeConfig`, used by
// the functional MoE forward in `utils::functional`). Native is unaffected.
pub mod config;
#[cfg(not(target_family = "wasm"))]
pub mod decoder_layer;
#[cfg(not(target_family = "wasm"))]
pub mod model;
#[cfg(feature = "full")]
pub(crate) mod paged_forward;
#[cfg(not(target_family = "wasm"))]
pub mod persistence;
#[cfg(not(target_family = "wasm"))]
pub mod quantized_linear;
#[cfg(not(target_family = "wasm"))]
pub mod sparse_moe;
#[cfg(not(target_family = "wasm"))]
pub mod switch_glu;
#[cfg(not(target_family = "wasm"))]
pub mod switch_linear;

pub use config::Qwen3_5MoeConfig;
#[cfg(not(target_family = "wasm"))]
pub use model::Qwen3_5MoeModel;
