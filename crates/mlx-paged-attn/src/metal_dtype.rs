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
    /// Splash `kv::Format::Int8`: symmetric int8 with one f32 scale per
    /// token, head and side (`splash_store_kv_row`).
    Int8,
}

impl MetalDtype {
    /// Get the Metal kernel type string
    pub fn type_string(&self) -> &'static str {
        match self {
            MetalDtype::Float32 => "float",
            MetalDtype::Float16 => "half",
            MetalDtype::BFloat16 => "bfloat16_t",
            MetalDtype::Int8 => "char",
        }
    }

    /// Get size in bytes
    pub fn size(&self) -> usize {
        match self {
            MetalDtype::Float32 => 4,
            MetalDtype::Float16 | MetalDtype::BFloat16 => 2,
            MetalDtype::Int8 => 1,
        }
    }

    /// Dtype of the rows written into and read out of a cache of this
    /// dtype. Splash io is always bf16 for an int8 cache.
    pub fn io_dtype(&self) -> MetalDtype {
        match self {
            MetalDtype::Int8 => MetalDtype::BFloat16,
            other => *other,
        }
    }

    /// Scale bytes per token, head and side (Splash `kv::Layout`
    /// `bytesPerLayerPage`: an f32 per row for int8, none otherwise).
    pub fn scale_bytes(&self) -> usize {
        match self {
            MetalDtype::Int8 => 4,
            _ => 0,
        }
    }
}
