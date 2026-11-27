//! CUDA kernel facades for Hy4-flash (`models::hy_v4_flash`).
//!
//! All primitives live in `crates/mlx-sys/src/cuda/hy4/{nvfp4,moe,hc_engram,mla,indexer}.cpp`
//! and dispatch only when CUDA GPU is active (`available()`). Every wrapper
//! returns `Result<Option<T>>` (`Ok(None)` off CUDA or on off-contract shapes/dtypes)
//! so callers can fall back cleanly without panicking.

pub mod cuda;

pub use cuda::*;
