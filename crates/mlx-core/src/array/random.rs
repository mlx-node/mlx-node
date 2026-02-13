use super::{DType, MxArray};
use mlx_sys as sys;
use napi::bindgen_prelude::*;
use napi_derive::napi;

#[napi]
impl MxArray {
    #[napi]
    pub fn random_uniform(
        shape: &[i64],
        low: f64,
        high: f64,
        dtype: Option<DType>,
    ) -> Result<Self> {
        let dt = dtype.unwrap_or(DType::Float32);
        let handle = unsafe {
            sys::mlx_array_random_uniform(
                shape.as_ptr(),
                shape.len(),
                low as f32,
                high as f32,
                dt.code(),
            )
        };
        MxArray::from_handle(handle, "random_uniform")
    }

    #[napi]
    pub fn random_normal(shape: &[i64], mean: f64, std: f64, dtype: Option<DType>) -> Result<Self> {
        let dt = dtype.unwrap_or(DType::Float32);
        let handle = unsafe {
            sys::mlx_array_random_normal(
                shape.as_ptr(),
                shape.len(),
                mean as f32,
                std as f32,
                dt.code(),
            )
        };
        MxArray::from_handle(handle, "random_normal")
    }

    #[napi]
    pub fn random_bernoulli(shape: &[i64], prob: f64) -> Result<Self> {
        let handle =
            unsafe { sys::mlx_array_random_bernoulli(shape.as_ptr(), shape.len(), prob as f32) };
        MxArray::from_handle(handle, "random_bernoulli")
    }

    #[napi]
    pub fn randint(shape: &[i64], low: i32, high: i32) -> Result<Self> {
        let handle = unsafe { sys::mlx_array_randint(shape.as_ptr(), shape.len(), low, high) };
        MxArray::from_handle(handle, "randint")
    }

    /// Sample from categorical distribution
    /// Takes logits and returns sampled indices
    #[napi]
    pub fn categorical(&self, axis: Option<i32>) -> Result<MxArray> {
        let axis_val = axis.unwrap_or(-1);
        let handle = unsafe { sys::mlx_array_categorical(self.handle.0, axis_val) };
        MxArray::from_handle(handle, "categorical")
    }
}
