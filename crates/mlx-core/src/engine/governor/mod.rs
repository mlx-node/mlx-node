//! One process memory governor for the qwen3_5 dense and MoE models, after
//! Splash `MemoryGovernor` and `MemoryLedger`.

use std::sync::{Arc, OnceLock};

pub(crate) mod allocate;
#[allow(clippy::module_inception)]
pub(crate) mod governor;
pub(crate) mod host;
pub(crate) mod ledger;
pub(crate) mod metrics;
pub(crate) mod monitor;
mod oom;
pub(crate) mod status;
pub(crate) mod wait;

pub(crate) use oom::{is_gpu_out_of_memory, metal_out_of_memory};

/// The one governor of the process: one Metal device, one MLX allocator.
pub(crate) fn governor() -> &'static governor::Governor {
    static GOVERNOR: OnceLock<governor::Governor> = OnceLock::new();
    GOVERNOR.get_or_init(|| governor::Governor::new(Arc::new(governor::SystemIo::new())))
}
