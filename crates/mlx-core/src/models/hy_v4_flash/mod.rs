//! Hy4-flash (`hy_v4_flash`) text model from the llama.cpp PR #32517 GGUF
//! (AngelSlim Q4_K_M: Q4_K/Q6_K experts, Q8_0 / F32 / BF16 dense, F8_E4M3
//! Engram table stored as GGUF type 43).

pub mod attention;
pub mod bank;
pub mod block;
pub mod cache;
pub mod config;
pub mod engram;
pub mod f8;
pub mod finalize;
pub mod header;
pub mod indexer;
pub mod model;
pub mod moe;
pub mod ops;
pub mod paged;
pub mod rows;
pub mod scan;
pub mod sparse;
pub mod stats;
pub mod store;
pub mod weights;

#[cfg(test)]
mod tests_config;
#[cfg(test)]
mod tests_model;
#[cfg(test)]
mod tests_paged;
#[cfg(all(test, target_os = "macos"))]
mod tests_pool;
#[cfg(test)]
mod tests_rows;
#[cfg(test)]
mod tests_weights;

pub use config::{
    Attn, DEFAULT_CONTEXT_CAP, HcConfig, HyV4FlashConfig, IndexerConfig, LayerConfig, MlaConfig,
    MoeConfig,
};
