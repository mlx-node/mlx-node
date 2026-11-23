// Re-export shared modules from `qwen3_5` (unified dense + MoE implementation).
pub use crate::models::qwen3_5::arrays_cache;
pub use crate::models::qwen3_5::attention;
pub use crate::models::qwen3_5::gated_delta;
pub use crate::models::qwen3_5::gated_delta_net;
pub use crate::models::qwen3_5::layer_cache;
pub use crate::models::qwen3_5::rms_norm_gated;
pub use crate::models::qwen3_5::sparse_moe;
pub use crate::models::qwen3_5::switch_glu;

pub use crate::models::quantized_linear;
pub mod switch_linear {
    pub use crate::moe::switch::SwitchLinear;
}

/// Persistence shim delegating to the unified `qwen3_5::persistence` loader.
pub mod persistence {
    pub(crate) use crate::models::qwen3_5::persistence::load_moe_with_source as load_with_source;
    pub use crate::models::qwen3_5::persistence::{
        create_random_qwen35_moe_checkpoint, create_random_qwen35_moe_checkpoint_sync,
        load_moe_for_tool as load_for_tool, load_moe_with_thread as load_with_thread,
    };
    #[cfg(test)]
    pub(crate) use crate::models::qwen3_5::persistence::{
        load_moe_tool_source as load_tool_source, parse_moe_config as parse_config,
        sanitize_weights_moe as sanitize_weights,
    };
}

// MoE-specific config and NAPI facade.
pub mod config;
pub mod model;

pub use config::Qwen3_5MoeConfig;
pub use model::Qwen3_5MoeModel;
