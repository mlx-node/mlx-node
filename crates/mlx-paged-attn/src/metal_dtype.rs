//! Non-macOS definition of `MetalDtype`.
//!
//! On macOS this enum lives in `metal/state.rs` (inside the Metal-gated
//! `metal` module). It is a pure-data enum with no Apple/Metal dependency,
//! but it leaks into the cross-platform `LayerKVPool` / `profile` APIs and
//! into mlx-core's per-model paged-cache bring-up (`MetalDtype::BFloat16`).
//! This file provides a byte-identical copy for non-macOS builds so that
//! cross-platform code compiles, re-exported as `metal::MetalDtype` from
//! `lib.rs`. It is compiled ONLY on non-macOS hosts; macOS keeps the
//! original definition in `metal/state.rs` unchanged.

/// Data types supported by Metal kernels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetalDtype {
    Float32,
    Float16,
    BFloat16,
    /// FP8 E4M3 format (1 byte per element)
    UChar,
}

impl MetalDtype {
    /// Get the Metal kernel type string
    pub fn type_string(&self) -> &'static str {
        match self {
            MetalDtype::Float32 => "float",
            MetalDtype::Float16 => "half",
            MetalDtype::BFloat16 => "bfloat16_t",
            MetalDtype::UChar => "uchar",
        }
    }

    /// Get size in bytes
    pub fn size(&self) -> usize {
        match self {
            MetalDtype::Float32 => 4,
            MetalDtype::Float16 | MetalDtype::BFloat16 => 2,
            MetalDtype::UChar => 1,
        }
    }

    /// Check if this is an FP8 type
    pub fn is_fp8(&self) -> bool {
        matches!(self, MetalDtype::UChar)
    }
}
