//! Metal kernel dispatch for paged attention
//!
//! This module provides GPU-accelerated paged attention using Metal compute kernels.
//! Based on patterns from mistral.rs Metal implementation.
//!
//! # Architecture
//!
//! The paged attention Metal dispatch works in two parts:
//!
//! 1. **Pre-compiled Kernels**: Metal shaders are compiled to a metallib at build time.
//!    The kernels use template parameters (not function constants) for MLX compatibility.
//!
//! 2. **MLX Buffer Integration**: Metal buffer pointers are extracted from MLX arrays
//!    via FFI, then used with the Rust `metal` crate for kernel dispatch.
//!
//! # Kernel Naming Convention
//!
//! - `reshape_and_cache_kv_{type}_cache_{type}[_fp8]`
//! - `paged_attention_{type}_cache_{type}_hs{head}_bs{block}_nt256_nsl32_ps{partition}[_alibi]`

mod mlx_integration;
mod paged_attention;
mod reshape_and_cache;
mod state;

pub use mlx_integration::{MlxMetalBuffer, is_metal_extraction_supported, synchronize_mlx};
pub use paged_attention::{
    PagedAttentionParams, dispatch_paged_attention_v1, dispatch_paged_attention_v2,
};
pub use reshape_and_cache::{ReshapeAndCacheParams, dispatch_reshape_and_cache};
pub use state::{MetalDtype, MetalState};
