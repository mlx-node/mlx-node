use super::{DType, MxArray};
use mlx_sys as sys;
use napi::bindgen_prelude::*;
use napi_derive::napi;

fn validate_data_shape(data_len: usize, shape: &[i64], context: &str) -> Result<()> {
    let mut expected: usize = 1;
    for (i, &d) in shape.iter().enumerate() {
        if d < 0 {
            return Err(Error::from_reason(format!(
                "{}: negative dimension {} at axis {}",
                context, d, i
            )));
        }
        expected = expected.checked_mul(d as usize).ok_or_else(|| {
            Error::from_reason(format!(
                "{}: shape {:?} overflows usize at axis {}",
                context, shape, i
            ))
        })?;
    }
    if data_len != expected {
        return Err(Error::from_reason(format!(
            "{}: data length {} does not match shape {:?} (expected {})",
            context, data_len, shape, expected
        )));
    }
    Ok(())
}

#[napi]
impl MxArray {
    #[napi]
    pub fn from_int32(data: &[i32], shape: &[i64]) -> Result<Self> {
        validate_data_shape(data.len(), shape, "from_int32")?;
        let handle =
            unsafe { sys::mlx_array_from_int32(data.as_ptr(), shape.as_ptr(), shape.len()) };
        MxArray::from_handle(handle, "array_from_int32")
    }

    #[napi]
    pub fn from_int64(data: &[i64], shape: &[i64]) -> Result<Self> {
        validate_data_shape(data.len(), shape, "from_int64")?;
        let handle =
            unsafe { sys::mlx_array_from_int64(data.as_ptr(), shape.as_ptr(), shape.len()) };
        MxArray::from_handle(handle, "array_from_int64")
    }

    #[napi]
    pub fn from_uint32(data: &[u32], shape: &[i64]) -> Result<Self> {
        validate_data_shape(data.len(), shape, "from_uint32")?;
        let handle =
            unsafe { sys::mlx_array_from_uint32(data.as_ptr(), shape.as_ptr(), shape.len()) };
        MxArray::from_handle(handle, "array_from_uint32")
    }

    #[napi]
    pub fn from_float32(data: &[f32], shape: &[i64]) -> Result<Self> {
        validate_data_shape(data.len(), shape, "from_float32")?;
        let handle =
            unsafe { sys::mlx_array_from_float32(data.as_ptr(), shape.as_ptr(), shape.len()) };
        MxArray::from_handle(handle, "array_from_float32")
    }

    /// Create an MxArray from raw uint8 bytes.
    /// Used for loading FP8 E4M3 weights (1 byte per element).
    pub fn from_uint8(data: &[u8], shape: &[i64]) -> Result<Self> {
        validate_data_shape(data.len(), shape, "from_uint8")?;
        let handle =
            unsafe { sys::mlx_array_from_uint8(data.as_ptr(), shape.as_ptr(), shape.len()) };
        MxArray::from_handle(handle, "array_from_uint8")
    }

    /// Convert FP8 E4M3 array to target dtype using MLX's from_fp8.
    /// Input must be a uint8 array containing FP8 E4M3 encoded values.
    pub fn from_fp8(&self, target_dtype: DType) -> Result<Self> {
        let handle = unsafe { sys::mlx_from_fp8(self.as_raw_ptr(), target_dtype.code()) };
        MxArray::from_handle(handle, "from_fp8")
    }

    /// Create an MxArray from raw bfloat16 bytes (as u16 values).
    /// This enables zero-copy loading of bf16 weights from safetensors.
    /// The input is the raw bytes reinterpreted as u16 (2 bytes per element).
    pub fn from_bfloat16(data: &[u16], shape: &[i64]) -> Result<Self> {
        validate_data_shape(data.len(), shape, "from_bfloat16")?;
        let handle =
            unsafe { sys::mlx_array_from_bfloat16(data.as_ptr(), shape.as_ptr(), shape.len()) };
        MxArray::from_handle(handle, "array_from_bfloat16")
    }

    /// Create an MxArray from raw float16 bytes (as u16 values).
    /// This enables zero-copy loading of f16 weights from safetensors.
    /// The input is the raw bytes reinterpreted as u16 (2 bytes per element).
    pub fn from_float16(data: &[u16], shape: &[i64]) -> Result<Self> {
        validate_data_shape(data.len(), shape, "from_float16")?;
        let handle =
            unsafe { sys::mlx_array_from_float16(data.as_ptr(), shape.as_ptr(), shape.len()) };
        MxArray::from_handle(handle, "array_from_float16")
    }

    #[napi]
    pub fn zeros(shape: &[i64], dtype: Option<DType>) -> Result<Self> {
        let dt = dtype.unwrap_or(DType::Float32);
        let handle = unsafe { sys::mlx_array_zeros(shape.as_ptr(), shape.len(), dt.code()) };
        MxArray::from_handle(handle, "array_zeros")
    }

    #[napi]
    pub fn scalar_float(value: f64) -> Result<Self> {
        let handle = unsafe { sys::mlx_array_scalar_float(value) };
        MxArray::from_handle(handle, "array_scalar_float")
    }

    /// Create a scalar with a specific dtype (no AsType node in the graph).
    /// Matches Python's `mx.array(value, dtype=dtype)`.
    pub(crate) fn scalar_float_like(value: f64, like: &MxArray) -> Result<Self> {
        let dt = like.dtype()?;
        let handle = unsafe { sys::mlx_array_scalar_float_dtype(value, dt.code()) };
        MxArray::from_handle(handle, "array_scalar_float_dtype")
    }

    #[napi]
    pub fn scalar_int(value: i32) -> Result<Self> {
        let handle = unsafe { sys::mlx_array_scalar_int(value) };
        MxArray::from_handle(handle, "array_scalar_int")
    }

    #[napi]
    pub fn ones(shape: &[i64], dtype: Option<DType>) -> Result<Self> {
        let dt = dtype.unwrap_or(DType::Float32);
        let handle = unsafe { sys::mlx_array_ones(shape.as_ptr(), shape.len(), dt.code()) };
        MxArray::from_handle(handle, "array_ones")
    }

    #[napi]
    pub fn full(
        shape: &[i64],
        fill_value: Either<f64, &MxArray>,
        dtype: Option<DType>,
    ) -> Result<Self> {
        let (dtype_value, has_dtype) = match dtype {
            Some(dt) => (dt, true),
            None => (DType::Float32, false),
        };

        let mut scalar_holder: Option<MxArray> = None;
        let value_handle = match fill_value {
            Either::A(number) => {
                let scalar = if dtype_value == DType::Int32 {
                    MxArray::scalar_int(number as i32)?
                } else {
                    MxArray::scalar_float(number)?
                };
                let handle = scalar.handle.0;
                scalar_holder = Some(scalar);
                handle
            }
            Either::B(array) => array.handle.0,
        };

        let handle = unsafe {
            sys::mlx_array_full(
                shape.as_ptr(),
                shape.len(),
                value_handle,
                dtype_value.code(),
                has_dtype,
            )
        };

        // Drop temporary scalar after native call
        drop(scalar_holder);

        MxArray::from_handle(handle, "array_full")
    }

    #[napi]
    pub fn linspace(start: f64, stop: f64, num: Option<i32>, dtype: Option<DType>) -> Result<Self> {
        let samples = num.unwrap_or(50);
        if samples < 0 {
            return Err(Error::from_reason(format!(
                "linspace requires non-negative num, got {}",
                samples
            )));
        }

        let (dtype_value, has_dtype) = match dtype {
            Some(dt) => (dt, true),
            None => (DType::Float32, false),
        };

        let handle =
            unsafe { sys::mlx_array_linspace(start, stop, samples, dtype_value.code(), has_dtype) };
        MxArray::from_handle(handle, "array_linspace")
    }

    #[napi]
    pub fn eye(n: i32, m: Option<i32>, k: Option<i32>, dtype: Option<DType>) -> Result<Self> {
        if n <= 0 {
            return Err(Error::from_reason(format!(
                "eye requires positive n, got {}",
                n
            )));
        }
        let columns = m.unwrap_or(n);
        if columns <= 0 {
            return Err(Error::from_reason(format!(
                "eye requires positive m, got {}",
                columns
            )));
        }

        let (dtype_value, has_dtype) = match dtype {
            Some(dt) => (dt, true),
            None => (DType::Float32, false),
        };

        let handle = unsafe {
            sys::mlx_array_eye(n, columns, k.unwrap_or(0), dtype_value.code(), has_dtype)
        };
        MxArray::from_handle(handle, "array_eye")
    }

    #[napi]
    pub fn arange(start: f64, stop: f64, step: Option<f64>, dtype: Option<DType>) -> Result<Self> {
        let dt = dtype.unwrap_or(DType::Float32);
        let handle = unsafe { sys::mlx_array_arange(start, stop, step.unwrap_or(1.0), dt.code()) };
        MxArray::from_handle(handle, "array_arange")
    }
}
