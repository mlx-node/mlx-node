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

// MoE-specific config and NAPI facade.
// Keep `config` declared before `persistence` so `napi-derive` resolves
// `Qwen3_5MoeConfig` -> `Qwen35MoeConfig` when emitting `index.d.cts`.
pub mod config;
pub mod model;

/// Persistence shim delegating to the unified `qwen3_5::persistence` loader.
pub mod persistence {
    use napi::bindgen_prelude::*;
    use napi_derive::napi;

    use super::config::Qwen3_5MoeConfig;

    pub(crate) use crate::models::qwen3_5::persistence::load_moe_with_source as load_with_source;
    pub use crate::models::qwen3_5::persistence::{
        create_random_qwen35_moe_checkpoint_sync, load_moe_for_tool as load_for_tool,
        load_moe_with_thread as load_with_thread,
    };
    #[cfg(test)]
    pub(crate) use crate::models::qwen3_5::persistence::{
        load_moe_tool_source as load_tool_source, parse_moe_config as parse_config,
        sanitize_weights_moe as sanitize_weights,
    };

    /// Create a random-init Qwen3.5 MoE model and save it to disk.
    ///
    /// Spawns a dedicated model thread whose init runs
    /// [`create_random_qwen35_moe_checkpoint_sync`] (random-init inner + save);
    /// the thread holds no state and is dropped once the promise resolves, so
    /// the in-memory model is released as soon as the checkpoint has been
    /// written. Used by TypeScript test fixtures that need an on-disk
    /// checkpoint without keeping a NAPI model instance alive.
    #[napi]
    pub fn create_random_qwen35_moe_checkpoint<'env>(
        env: &'env Env,
        config: Qwen3_5MoeConfig,
        save_path: String,
    ) -> Result<PromiseRaw<'env, ()>> {
        let (thread, init_rx) = crate::model_thread::ModelThread::<()>::spawn_with_init(
            move || {
                create_random_qwen35_moe_checkpoint_sync(config, &save_path)?;
                Ok(((), ()))
            },
            |_state, _cmd| {},
        );

        env.spawn_future(async move {
            init_rx
                .await
                .map_err(|_| napi::Error::from_reason("Model thread exited during init"))??;

            drop(thread);
            Ok(())
        })
    }
}

pub use config::Qwen3_5MoeConfig;
pub use model::Qwen3_5MoeModel;
