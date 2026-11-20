//! Zero-overhead static-dispatch kernel facades for `mlx-core`.
//!
//! Model and layer code calls backend-neutral helpers here instead of branching
//! directly on Metal vs CUDA FFI pairs. Every wrapper dispatches via
//! `#[inline]` `match crate::accelerator::active_backend()` with zero heap
//! allocations and returns `Result<Option<T>>` (`Ok(Some(v))` = kernel ran,
//! `Ok(None)` = unsupported backend/shape/dtype decline, `Err(e)` = fail-closed
//! execution error).

pub(crate) mod attention;
pub(crate) mod dflash2;
pub(crate) mod gdn;
pub(crate) mod kv;
pub(crate) mod linear;
