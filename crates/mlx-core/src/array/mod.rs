// Existing submodules
pub mod attention;
mod handle;
pub mod mask;
pub mod padding;

// New submodules (decomposed from this file)
mod comparison;
mod creation;
mod data;
pub mod memory;
mod ops;
mod random;
mod reduction;
mod shape;

// Re-exports from existing submodules
pub use attention::{scaled_dot_product_attention, scaled_dot_product_attention_causal};
pub(crate) use handle::{MxHandle, check_handle};
pub use padding::{
    LeftPaddedSequences, PaddedSequences, left_pad_sequences, pad_float_sequences, pad_sequences,
};

// Re-exports from memory submodule (used across the crate)
pub use memory::{
    check_memory_safety, clear_cache, compile_clear_cache, get_active_memory, get_cache_memory,
    get_memory_limit, get_peak_memory, heavy_cleanup, reset_peak_memory, set_cache_limit,
    set_memory_limit, synchronize, synchronize_and_clear_cache,
};

use mlx_sys as sys;
use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::sync::Arc;

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[napi]
pub enum DType {
    Float32 = 0,
    Int32 = 1,
    Float16 = 2,
    BFloat16 = 3,
    Uint32 = 4,
    Uint8 = 5,
}

impl DType {
    fn code(self) -> i32 {
        self as i32
    }

    /// Byte size per element for this dtype.
    /// Used to pre-compute SafeTensors data offsets without materializing tensors.
    pub(crate) fn byte_size(self) -> usize {
        match self {
            DType::Float32 | DType::Int32 | DType::Uint32 => 4,
            DType::Float16 | DType::BFloat16 => 2,
            DType::Uint8 => 1,
        }
    }
}

impl TryFrom<i32> for DType {
    type Error = Error;

    fn try_from(value: i32) -> Result<Self> {
        match value {
            0 => Ok(DType::Float32),
            1 => Ok(DType::Int32),
            2 => Ok(DType::Float16),
            3 => Ok(DType::BFloat16),
            4 => Ok(DType::Uint32),
            5 => Ok(DType::Uint8),
            other => Err(Error::from_reason(format!(
                "Unsupported dtype code {other}"
            ))),
        }
    }
}

#[napi]
pub struct MxArray {
    pub(crate) handle: Arc<MxHandle>,
}

impl MxArray {
    pub(crate) fn from_handle(handle: *mut sys::mlx_array, context: &str) -> Result<Self> {
        Ok(Self {
            handle: Arc::new(MxHandle(check_handle(handle, context)?)),
        })
    }

    /// Get the raw MLX array pointer for FFI operations
    ///
    /// This is primarily used for Metal buffer extraction to enable
    /// GPU kernel dispatch with external Metal infrastructure.
    ///
    /// # Safety
    /// The returned pointer is only valid as long as this MxArray exists.
    /// Do not use the pointer after the MxArray is dropped.
    pub fn as_raw_ptr(&self) -> *mut sys::mlx_array {
        self.handle.0
    }
}

impl Clone for MxArray {
    fn clone(&self) -> Self {
        Self {
            handle: Arc::clone(&self.handle),
        }
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod array_ops_tests;
