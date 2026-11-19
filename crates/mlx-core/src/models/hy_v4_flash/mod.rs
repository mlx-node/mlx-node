//! Hy4-flash (`hy_v4_flash`) text model from the llama.cpp PR #32517 GGUF
//! (AngelSlim Q4_K_M: Q4_K/Q6_K experts, Q8_0 / F32 / BF16 dense, F8_E4M3
//! Engram table stored as GGUF type 43).

pub mod attention;
pub mod bank;
pub mod batch;
pub mod block;
pub mod cache;
pub mod chat;
pub mod config;
pub mod engram;
pub mod f8;
pub mod finalize;
pub mod header;
pub mod indexer;
pub mod ladder;
pub mod model;
pub mod moe;
pub mod mtp;
pub mod mtp_weights;
pub mod ops;
pub mod owner;
pub mod paged;
pub(crate) mod persist;
pub(crate) mod persist_codec;
pub(crate) mod persist_writer;
pub mod prefill;
pub mod prefix_state;
pub mod rows;
pub mod scan;
pub mod sched;
pub mod sparse;
pub mod spec;
pub mod stats;
pub mod store;
pub mod weights;

#[cfg(test)]
mod tests_batch;
#[cfg(test)]
mod tests_chat;
#[cfg(test)]
mod tests_config;
#[cfg(test)]
mod tests_model;
#[cfg(test)]
mod tests_mtp;
#[cfg(test)]
mod tests_owner;
#[cfg(test)]
mod tests_paged;
#[cfg(test)]
mod tests_persist;
#[cfg(all(test, target_os = "macos"))]
mod tests_pool;
#[cfg(test)]
mod tests_prefill;
#[cfg(test)]
mod tests_prefix_state;
#[cfg(test)]
mod tests_rows;
#[cfg(test)]
mod tests_sched;
#[cfg(test)]
mod tests_sched_mtp;
#[cfg(test)]
mod tests_weights;

pub use config::{
    Attn, DEFAULT_CONTEXT_CAP, HcConfig, HyV4FlashConfig, IndexerConfig, LayerConfig, MlaConfig,
    MoeConfig,
};

/// True f32 GEMMs on the calling thread, as `Model::new` sets for the model
/// thread. Component tests that never build a `Model` call it: on NAX GPUs a
/// thread that keeps MLX's default runs f32 GEMMs with M > 1 in TF32.
#[cfg(test)]
pub(crate) fn exact_f32() {
    crate::array::set_thread_tf32(Some(false));
}
