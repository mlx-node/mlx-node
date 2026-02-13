use super::MxArray;
use mlx_sys as sys;
use napi::bindgen_prelude::*;
use napi_derive::napi;

#[napi]
impl MxArray {
    #[napi]
    pub fn sum(&self, axes: Option<&[i32]>, keepdims: Option<bool>) -> Result<MxArray> {
        let axes_vec = axes.unwrap_or_default();
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
        let handle =
            unsafe { sys::mlx_array_argmax(self.handle.0, axis, keepdims.unwrap_or(false)) };
        MxArray::from_handle(handle, "argmax")
    }

    #[napi]
    pub fn argmin(&self, axis: i32, keepdims: Option<bool>) -> Result<MxArray> {
        let handle =
            unsafe { sys::mlx_array_argmin(self.handle.0, axis, keepdims.unwrap_or(false)) };
        MxArray::from_handle(handle, "argmin")
    }

    #[napi]
    pub fn max(&self, axes: Option<&[i32]>, keepdims: Option<bool>) -> Result<MxArray> {
        let axes_vec = axes.unwrap_or_default();
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
        let handle = unsafe { sys::mlx_array_cumsum(self.handle.0, axis) };
        MxArray::from_handle(handle, "cumsum")
    }

    #[napi]
    pub fn cumprod(&self, axis: i32) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_cumprod(self.handle.0, axis) };
        MxArray::from_handle(handle, "cumprod")
    }
}
