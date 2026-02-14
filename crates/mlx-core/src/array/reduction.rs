use super::MxArray;
use mlx_sys as sys;
use napi::bindgen_prelude::*;
use napi_derive::napi;

/// Validate that all axes are within bounds for the array's dimensions.
/// Negative axes are normalized (axis + ndim) before checking bounds [0, ndim).
/// This prevents invalid axes from reaching the C++ FFI boundary where an
/// uncaught exception would be undefined behavior.
fn validate_axes(arr: &MxArray, axes: &[i32], context: &str) -> Result<()> {
    let ndim = arr.ndim()? as i32;
    for &axis in axes {
        let normalized = if axis < 0 { axis + ndim } else { axis };
        if normalized < 0 || normalized >= ndim {
            return Err(Error::from_reason(format!(
                "{}: axis {} is out of bounds for array with {} dimension{}",
                context,
                axis,
                ndim,
                if ndim == 1 { "" } else { "s" },
            )));
        }
    }
    Ok(())
}

#[napi]
impl MxArray {
    #[napi]
    pub fn sum(&self, axes: Option<&[i32]>, keepdims: Option<bool>) -> Result<MxArray> {
        let axes_vec = axes.unwrap_or_default();
        if !axes_vec.is_empty() {
            validate_axes(self, axes_vec, "sum")?;
        }
        let handle = unsafe {
            sys::mlx_array_sum(
                self.handle.0,
                axes_vec.as_ptr(),
                axes_vec.len(),
                keepdims.unwrap_or(false),
            )
        };
        MxArray::from_handle(handle, "array_sum")
    }

    #[napi]
    pub fn mean(&self, axes: Option<&[i32]>, keepdims: Option<bool>) -> Result<MxArray> {
        let axes_vec = axes.unwrap_or_default();
        if !axes_vec.is_empty() {
            validate_axes(self, axes_vec, "mean")?;
        }
        let handle = unsafe {
            sys::mlx_array_mean(
                self.handle.0,
                axes_vec.as_ptr(),
                axes_vec.len(),
                keepdims.unwrap_or(false),
            )
        };
        MxArray::from_handle(handle, "array_mean")
    }

    #[napi]
    pub fn argmax(&self, axis: i32, keepdims: Option<bool>) -> Result<MxArray> {
        validate_axes(self, &[axis], "argmax")?;
        let handle =
            unsafe { sys::mlx_array_argmax(self.handle.0, axis, keepdims.unwrap_or(false)) };
        MxArray::from_handle(handle, "argmax")
    }

    #[napi]
    pub fn argmin(&self, axis: i32, keepdims: Option<bool>) -> Result<MxArray> {
        validate_axes(self, &[axis], "argmin")?;
        let handle =
            unsafe { sys::mlx_array_argmin(self.handle.0, axis, keepdims.unwrap_or(false)) };
        MxArray::from_handle(handle, "argmin")
    }

    #[napi]
    pub fn max(&self, axes: Option<&[i32]>, keepdims: Option<bool>) -> Result<MxArray> {
        let axes_vec = axes.unwrap_or_default();
        if !axes_vec.is_empty() {
            validate_axes(self, axes_vec, "max")?;
        }
        let handle = unsafe {
            sys::mlx_array_max(
                self.handle.0,
                axes_vec.as_ptr(),
                axes_vec.len(),
                keepdims.unwrap_or(false),
            )
        };
        MxArray::from_handle(handle, "max")
    }

    #[napi]
    pub fn min(&self, axes: Option<&[i32]>, keepdims: Option<bool>) -> Result<MxArray> {
        let axes_vec = axes.unwrap_or_default();
        if !axes_vec.is_empty() {
            validate_axes(self, axes_vec, "min")?;
        }
        let handle = unsafe {
            sys::mlx_array_min(
                self.handle.0,
                axes_vec.as_ptr(),
                axes_vec.len(),
                keepdims.unwrap_or(false),
            )
        };
        MxArray::from_handle(handle, "min")
    }

    #[napi]
    pub fn prod(&self, axes: Option<&[i32]>, keepdims: Option<bool>) -> Result<MxArray> {
        let axes_vec = axes.unwrap_or_default();
        if !axes_vec.is_empty() {
            validate_axes(self, axes_vec, "prod")?;
        }
        let handle = unsafe {
            sys::mlx_array_prod(
                self.handle.0,
                axes_vec.as_ptr(),
                axes_vec.len(),
                keepdims.unwrap_or(false),
            )
        };
        MxArray::from_handle(handle, "prod")
    }

    #[napi]
    pub fn var(
        &self,
        axes: Option<&[i32]>,
        keepdims: Option<bool>,
        ddof: Option<i32>,
    ) -> Result<MxArray> {
        let axes_vec = axes.unwrap_or_default();
        if !axes_vec.is_empty() {
            validate_axes(self, axes_vec, "var")?;
        }
        let handle = unsafe {
            sys::mlx_array_var(
                self.handle.0,
                axes_vec.as_ptr(),
                axes_vec.len(),
                keepdims.unwrap_or(false),
                ddof.unwrap_or(0),
            )
        };
        MxArray::from_handle(handle, "var")
    }

    #[napi]
    pub fn std(
        &self,
        axes: Option<&[i32]>,
        keepdims: Option<bool>,
        ddof: Option<i32>,
    ) -> Result<MxArray> {
        let axes_vec = axes.unwrap_or_default();
        if !axes_vec.is_empty() {
            validate_axes(self, axes_vec, "std")?;
        }
        let handle = unsafe {
            sys::mlx_array_std(
                self.handle.0,
                axes_vec.as_ptr(),
                axes_vec.len(),
                keepdims.unwrap_or(false),
                ddof.unwrap_or(0),
            )
        };
        MxArray::from_handle(handle, "std")
    }

    #[napi]
    pub fn logsumexp(&self, axes: Option<&[i32]>, keepdims: Option<bool>) -> Result<MxArray> {
        let axes_vec = axes.unwrap_or_default();
        if !axes_vec.is_empty() {
            validate_axes(self, axes_vec, "logsumexp")?;
        }
        let handle = unsafe {
            sys::mlx_array_logsumexp(
                self.handle.0,
                axes_vec.as_ptr(),
                axes_vec.len(),
                keepdims.unwrap_or(false),
            )
        };
        MxArray::from_handle(handle, "logsumexp")
    }

    #[napi]
    pub fn cumsum(&self, axis: i32) -> Result<MxArray> {
        validate_axes(self, &[axis], "cumsum")?;
        let handle = unsafe { sys::mlx_array_cumsum(self.handle.0, axis) };
        MxArray::from_handle(handle, "cumsum")
    }

    #[napi]
    pub fn cumprod(&self, axis: i32) -> Result<MxArray> {
        validate_axes(self, &[axis], "cumprod")?;
        let handle = unsafe { sys::mlx_array_cumprod(self.handle.0, axis) };
        MxArray::from_handle(handle, "cumprod")
    }
}
