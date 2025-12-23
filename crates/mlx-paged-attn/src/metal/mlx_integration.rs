//! MLX Array to Metal Buffer Integration
//!
//! This module provides the bridge between MLX arrays and our Metal kernel dispatch.
//! It extracts Metal buffer pointers from MLX arrays and dispatches kernels using
//! our pre-compiled metallib.
//!
//! # Architecture
//!
//! ```text
//! MxArray (Rust) → mlx_array* (C++ FFI) → buffer().ptr() → MTLBuffer*
//!                                                              ↓
//!                                       Rust metal crate ← Buffer::from_ptr()
//!                                                              ↓
//!                                       dispatch_reshape_and_cache()
//! ```
//!
//! # Safety
//!
//! The extracted buffer pointers are only valid:
//! - After MLX has evaluated the array (ensured by calling mlx_metal_synchronize)
//! - Before any MLX operation that could reallocate the buffer
//! - When Metal/GPU backend is available (checked via mlx_metal_is_available)
//!
//! We ensure safety by:
//! 1. Checking mlx_metal_is_available() before extracting buffers
//! 2. Calling mlx_metal_synchronize() before extracting buffers
//! 3. Completing our Metal dispatch before returning control to MLX
//! 4. Using Metal's waitUntilCompleted() to ensure our kernels finish

use mlx_sys::{
    mlx_array, mlx_array_get_buffer_offset, mlx_array_get_data_size, mlx_array_get_itemsize,
    mlx_array_get_metal_buffer, mlx_metal_is_available, mlx_metal_synchronize,
};
use std::ffi::c_void;

/// Information about an MLX array's Metal buffer
#[derive(Debug)]
pub struct MlxMetalBuffer {
    /// Raw MTLBuffer pointer (as void* for FFI)
    /// This is only valid when Metal/GPU is available
    pub buffer_ptr: *mut c_void,
    /// Byte offset into the buffer (for sliced arrays)
    pub offset: usize,
    /// Number of elements in the array (NOT bytes)
    pub data_size: usize,
    /// Size of each element in bytes
    pub itemsize: usize,
}

impl MlxMetalBuffer {
    /// Get the total data size in bytes
    pub fn data_size_bytes(&self) -> usize {
        self.data_size * self.itemsize
    }

    /// Extract Metal buffer info from an MLX array handle
    ///
    /// Returns `None` if:
    /// - handle is null
    /// - Metal/GPU is not available
    /// - array has no data
    ///
    /// # Safety
    /// The handle must be a valid MLX array pointer
    pub unsafe fn from_mlx_array(handle: *mut mlx_array) -> Option<Self> {
        if handle.is_null() {
            return None;
        }

        // SAFETY: handle is checked non-null above, caller guarantees validity
        // mlx_array_get_metal_buffer returns nullptr if Metal is not available
        let buffer_ptr = unsafe { mlx_array_get_metal_buffer(handle) };
        if buffer_ptr.is_null() {
            return None;
        }

        // SAFETY: handle is valid MLX array
        let offset = unsafe { mlx_array_get_buffer_offset(handle) };
        let data_size = unsafe { mlx_array_get_data_size(handle) };
        let itemsize = unsafe { mlx_array_get_itemsize(handle) };

        Some(MlxMetalBuffer {
            buffer_ptr,
            offset,
            data_size,
            itemsize,
        })
    }
}

/// Synchronize MLX Metal operations
///
/// Call this before extracting Metal buffers to ensure all MLX
/// operations are complete and buffers are valid.
pub fn synchronize_mlx() {
    unsafe {
        mlx_metal_synchronize();
    }
}

/// Check if Metal buffer extraction is supported
///
/// Returns true if we can extract Metal buffers from MLX arrays.
/// This checks both platform (macOS) and GPU availability.
pub fn is_metal_extraction_supported() -> bool {
    // Check both compile-time platform and runtime GPU availability
    cfg!(target_os = "macos") && unsafe { mlx_metal_is_available() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synchronize() {
        // Should not panic
        synchronize_mlx();
    }

    #[test]
    fn test_extraction_supported() {
        // This test verifies the function returns a valid boolean.
        // On macOS with GPU, this is typically true, but may be false
        // on headless CI, VMs, or when Metal is unavailable.
        let supported = is_metal_extraction_supported();

        #[cfg(not(target_os = "macos"))]
        assert!(!supported, "Metal should not be available on non-macOS");

        // On macOS, we just verify the function runs without panicking.
        // We don't assert it's true because headless/CI environments
        // may not have Metal available.
        #[cfg(target_os = "macos")]
        {
            // Log the result for debugging CI failures
            eprintln!("Metal extraction supported: {}", supported);
            // The function should work regardless of result
        }
    }

    #[test]
    fn test_null_handle() {
        let result = unsafe { MlxMetalBuffer::from_mlx_array(std::ptr::null_mut()) };
        assert!(result.is_none());
    }
}
