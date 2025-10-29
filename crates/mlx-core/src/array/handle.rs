use mlx_sys as sys;
use napi::bindgen_prelude::*;

pub(crate) fn check_handle(
    handle: *mut sys::mlx_array,
    context: &str,
) -> Result<*mut sys::mlx_array> {
    if handle.is_null() {
        Err(Error::from_reason(format!(
            "null handle returned: {}",
            context
        )))
    } else {
        Ok(handle)
    }
}

/// Internal handle wrapper that owns the MLX C++ array pointer
/// and ensures proper cleanup via Drop
pub(crate) struct MxHandle(pub(crate) *mut sys::mlx_array);

unsafe impl Send for MxHandle {}
unsafe impl Sync for MxHandle {}

impl MxHandle {
    /// Overwrite this handle's pointer with a new one, cleaning up the old pointer.
    /// This matches mlx-lm's `overwrite_descriptor` pattern for zero-allocation cache updates.
    ///
    /// # Safety
    /// The new_handle must be a valid MLX array pointer that this MxHandle will now own.
    pub(crate) unsafe fn overwrite(&mut self, new_handle: *mut sys::mlx_array) {
        // Delete old array if not null
        // SAFETY: caller guarantees self.0 is a valid array pointer (or null)
        // and new_handle is a valid array pointer
        unsafe {
            if !self.0.is_null() {
                sys::mlx_array_delete(self.0);
            }
            // Take ownership of new array
            self.0 = new_handle;
        }
    }
}

impl Drop for MxHandle {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { sys::mlx_array_delete(self.0) };
        }
    }
}
