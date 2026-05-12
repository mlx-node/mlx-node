//! PagedAttention for MLX-Node
//!
//! This crate provides efficient KV cache management using PagedAttention,
//! ported from the HuggingFace kernels-community Metal implementation.
//!
//! ## Features
//! - Block-based KV cache allocation (reduces memory waste from 60-80% to <4%)
//! - Copy-on-write semantics for beam search
//! - Prefix caching for shared system prompts
//! - Continuous batching support
//! - GPU-accelerated Metal kernel dispatch (macOS only)
//!
//! ## Platform Support
//! - The `metal` module and GPU kernel dispatch are only available on macOS
//! - Core PagedAttention logic (block allocation, scheduling) works on all platforms
//!
//! ## References
//! - [PagedAttention Paper](https://arxiv.org/abs/2309.06180)
//! - [HuggingFace kernels-community](https://huggingface.co/kernels-community/paged-attention)

mod block_allocator;
mod block_table;
mod config;
#[cfg(target_os = "macos")]
mod extern_c;
mod layer_kv_pool;
pub mod profile;

#[cfg(target_os = "macos")]
pub mod metal;

// Re-export the extern "C" shim symbols so the C++ Custom primitives
// in `crates/mlx-sys/src/mlx_paged_ops.cpp` can link against them.
#[cfg(target_os = "macos")]
pub use extern_c::{
    mlx_paged_attn_paged_attention_dispatch, mlx_paged_attn_reshape_and_cache_dispatch,
};

pub use block_allocator::*;
pub use block_table::*;
pub use config::*;
pub use layer_kv_pool::LayerKVPool;

/// Path to the compiled Metal library (set at build time)
/// Only valid on macOS; empty string on other platforms
#[cfg(target_os = "macos")]
pub const METALLIB_PATH: &str = env!("PAGED_ATTN_METALLIB");

/// Placeholder for non-macOS platforms
#[cfg(not(target_os = "macos"))]
pub const METALLIB_PATH: &str = "";
