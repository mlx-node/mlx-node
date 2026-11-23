//! Metal state and MLX buffer integration for paged KV pools.
//!
//! Paged-attention and `reshape_and_cache` compute dispatch lives in `mlx-sys`'s
//! graph-native C++ Custom primitives (`paged_attn.metallib`). This module
//! provides the shared `MetalState` (device, command queue, residency kick
//! metallib), `MetalDtype`, command-buffer observation, and MLX Metal buffer
//! extraction.

pub(crate) mod command_buffer;
mod mlx_integration;
mod state;

pub use mlx_integration::{MlxMetalBuffer, is_metal_extraction_supported, synchronize_mlx};
pub use state::{MetalDtype, MetalState};
