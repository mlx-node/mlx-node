//! Kernel facades for Qwen3.8-Flash-Next (`qwen4_exp`), split by backend:
//!
//! - [`common`]: shared weight descriptors and cross-backend compiled FFI ops
//!   (`common/mlx_qwen4_common.cpp`).
//! - [`cuda`]: CUDA kernel facades (`cuda/qwen4/*.cpp`), dispatching only when
//!   `active_backend() == ActiveBackend::CudaGpu`.
//! - [`metal`]: Metal kernel facades (`metal/mlx_qwen4_*.cpp`,
//!   `metal/mlx_gated_delta_metal.cpp`).

pub(crate) mod common;
pub(crate) mod cuda;
pub(crate) mod metal;

#[cfg(test)]
mod tests;

pub(crate) use common::*;
pub(crate) use cuda::*;
