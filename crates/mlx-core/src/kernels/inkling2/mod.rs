//! CUDA kernel facades for Inkling-2-Mini (`models::inkling2`).
//!
//! All primitives live in `crates/mlx-sys/src/cuda/inkling2/{ops,sdpa,moe}.cpp`
//! and dispatch only when CUDA GPU is active (`available()`). Every wrapper
//! returns `Result<Option<T>>` (`Ok(None)` off CUDA or on off-contract shapes/dtypes)
//! so model code can fall back cleanly without panicking.

pub mod cuda;

pub use cuda::*;
