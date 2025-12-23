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
mod cache_engine;
mod config;
mod paged_kv_cache;
mod scheduler;

#[cfg(target_os = "macos")]
pub mod metal;

pub use block_allocator::*;
pub use block_table::*;
pub use cache_engine::*;
pub use config::*;
pub use paged_kv_cache::*;
pub use scheduler::*;

/// Path to the compiled Metal library (set at build time)
/// Only valid on macOS; empty string on other platforms
#[cfg(target_os = "macos")]
pub const METALLIB_PATH: &str = env!("PAGED_ATTN_METALLIB");

/// Placeholder for non-macOS platforms
#[cfg(not(target_os = "macos"))]
pub const METALLIB_PATH: &str = "";
