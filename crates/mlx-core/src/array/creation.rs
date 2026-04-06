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

    /// Create an MxArray from raw int8 bytes (bit-reinterpret, no numeric
    /// conversion). Used for loading sym8 per-channel symmetric int8 weights
    /// (1 byte per element). NOT from_uint8 + astype — that would numerically
    /// convert (e.g. 255 -> 255i32 saturated) instead of reinterpreting -1.
    pub fn from_int8(data: &[i8], shape: &[i64]) -> Result<Self> {
        validate_data_shape(data.len(), shape, "from_int8")?;
        let handle =
            unsafe { sys::mlx_array_from_int8(data.as_ptr(), shape.as_ptr(), shape.len()) };
        MxArray::from_handle(handle, "array_from_int8")
    }

    /// Convert FP8 E4M3 array to target dtype using MLX's from_fp8.
    /// Input must be a uint8 array containing FP8 E4M3 encoded values.
    pub fn from_fp8(&self, target_dtype: DType) -> Result<Self> {
        let handle = unsafe { sys::mlx_from_fp8(self.as_raw_ptr(), target_dtype.code()) };
        MxArray::from_handle(handle, "from_fp8")
    }

    /// Convert a float array to FP8 E4M3 using MLX's to_fp8.
    /// Output is a uint8 array holding raw E4M3 bytes (no scale applied);
    /// round-trips back to a float dtype via `from_fp8`.
    pub fn to_fp8(&self) -> Result<Self> {
        let handle = unsafe { sys::mlx_to_fp8(self.as_raw_ptr()) };
        MxArray::from_handle(handle, "to_fp8")
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

    // --- Compile tests (exercise mlx::core::compile on WebGPU) ---

    #[napi]
    pub fn test_compile_basic() -> bool {
        unsafe { sys::mlx_test_compile_basic() }
    }

    #[napi]
    pub fn test_compile_matmul() -> bool {
        unsafe { sys::mlx_test_compile_matmul() }
    }

    #[napi]
    pub fn test_compile_repeated() -> bool {
        unsafe { sys::mlx_test_compile_repeated() }
    }

    #[napi]
    pub fn test_gpu_buffer_arrays() -> bool {
        unsafe { sys::mlx_test_gpu_buffer_arrays() }
    }

    /// Run single layer forward using model weights, return first 5 values
    #[napi]
    pub fn test_single_layer_forward() -> Vec<f64> {
        let mut buf = [0f32; 5];
        let ok = unsafe { sys::mlx_test_single_layer_forward(buf.as_mut_ptr(), 5) };
        if !ok { return vec![-999.0]; }
        buf.iter().map(|v| *v as f64).collect()
    }

    /// Read first N float values from a C++ weight map entry
    #[napi]
    pub fn read_cpp_weight(name: String, count: i32) -> Vec<f64> {
        if count <= 0 { return vec![-999.0]; }
        let c_name = match std::ffi::CString::new(name) {
            Ok(s) => s,
            Err(_) => return vec![-999.0],
        };
        let mut buf = vec![0f32; count as usize];
        let n = unsafe { sys::mlx_qwen35_read_weight(c_name.as_ptr(), buf.as_mut_ptr(), count) };
        if n <= 0 { return vec![-999.0]; }
        buf[..n as usize].iter().map(|v| *v as f64).collect()
    }

    /// Step-by-step GDN test: returns intermediate values at the given checkpoint.
    /// checkpoint 0-8: various stages of GDN pipeline. 9: conv1d weight info.
    #[napi]
    pub fn test_gdn_step(checkpoint: i32, max_count: i32) -> Vec<f64> {
        let mut buf = vec![0f32; max_count as usize];
        let n = unsafe { sys::mlx_test_gdn_step_by_step(checkpoint, buf.as_mut_ptr(), max_count) };
        if n <= 0 { return vec![-999.0]; }
        buf[..n as usize].iter().map(|v| *v as f64).collect()
    }

    /// Isolated GDN recurrence test with small known inputs (2x2 dims).
    /// Returns y_out values followed by state_out values.
    #[napi]
    pub fn test_gdn_recurrence_small() -> Vec<f64> {
        let mut buf = vec![0f32; 20];
        let n = unsafe { sys::mlx_test_gdn_recurrence_small(buf.as_mut_ptr(), 20) };
        if n <= 0 { return vec![-999.0]; }
        buf[..n as usize].iter().map(|v| *v as f64).collect()
    }

    /// Run full-attention layer 0 forward using model weights, return first N values.
    /// Exercises: RMSNorm + Q/K/V proj + QK norm + RoPE + SDPA + gate + output proj.
    #[napi]
    pub fn test_sdpa_causal(max_count: Option<i32>) -> Vec<f64> {
        let count = max_count.unwrap_or(20) as usize;
        let mut buf = vec![0f32; count];
        let n = unsafe { sys::mlx_test_sdpa_causal(buf.as_mut_ptr(), count as i32) };
        if n <= 0 { return vec![-999.0]; }
        buf[..n as usize].iter().map(|v| *v as f64).collect()
    }

    #[napi]
    pub fn test_sdpa_gqa(max_count: Option<i32>) -> Vec<f64> {
        let count = max_count.unwrap_or(20) as usize;
        let mut buf = vec![0f32; count];
        let n = unsafe { sys::mlx_test_sdpa_gqa(buf.as_mut_ptr(), count as i32) };
        if n <= 0 { return vec![-999.0]; }
        buf[..n as usize].iter().map(|v| *v as f64).collect()
    }

    #[napi]
    pub fn test_attention_layer_forward(max_count: Option<i32>) -> Vec<f64> {
        let count = max_count.unwrap_or(10) as usize;
        let mut buf = vec![0f32; count];
        let n = unsafe { sys::mlx_test_attention_layer_forward(buf.as_mut_ptr(), count as i32) };
        if n <= 0 { return vec![-999.0]; }
        buf[..n as usize].iter().map(|v| *v as f64).collect()
    }

    // ===== Phase 1+2 inference step tests =====

    #[napi]
    pub fn test_rope_bf16(max_count: Option<i32>) -> Vec<f64> {
        let c = max_count.unwrap_or(20) as usize;
        let mut buf = vec![0f32; c];
        let n = unsafe { sys::mlx_test_rope_bf16(buf.as_mut_ptr(), c as i32) };
        if n <= 0 { return vec![-999.0]; }
        buf[..n as usize].iter().map(|v| *v as f64).collect()
    }

    #[napi]
    pub fn test_qk_norm_rope(max_count: Option<i32>) -> Vec<f64> {
        let c = max_count.unwrap_or(20) as usize;
        let mut buf = vec![0f32; c];
        let n = unsafe { sys::mlx_test_qk_norm_rope(buf.as_mut_ptr(), c as i32) };
        if n <= 0 { return vec![-999.0]; }
        buf[..n as usize].iter().map(|v| *v as f64).collect()
    }

    #[napi]
    pub fn test_sdpa_additive_mask(max_count: Option<i32>) -> Vec<f64> {
        let c = max_count.unwrap_or(20) as usize;
        let mut buf = vec![0f32; c];
        let n = unsafe { sys::mlx_test_sdpa_additive_mask(buf.as_mut_ptr(), c as i32) };
        if n <= 0 { return vec![-999.0]; }
        buf[..n as usize].iter().map(|v| *v as f64).collect()
    }

    #[napi]
    pub fn test_sdpa_decode_gqa(max_count: Option<i32>) -> Vec<f64> {
        let c = max_count.unwrap_or(20) as usize;
        let mut buf = vec![0f32; c];
        let n = unsafe { sys::mlx_test_sdpa_decode_gqa(buf.as_mut_ptr(), c as i32) };
        if n <= 0 { return vec![-999.0]; }
        buf[..n as usize].iter().map(|v| *v as f64).collect()
    }

    #[napi]
    pub fn test_full_attn_layer_bf16(max_count: Option<i32>) -> Vec<f64> {
        let c = max_count.unwrap_or(20) as usize;
        let mut buf = vec![0f32; c];
        let n = unsafe { sys::mlx_test_full_attn_layer_bf16(buf.as_mut_ptr(), c as i32) };
        if n <= 0 { return vec![-999.0]; }
        buf[..n as usize].iter().map(|v| *v as f64).collect()
    }

    #[napi]
    pub fn test_rms_norm_bf16(max_count: Option<i32>) -> Vec<f64> {
        let c = max_count.unwrap_or(20) as usize;
        let mut buf = vec![0f32; c];
        let n = unsafe { sys::mlx_test_rms_norm_bf16(buf.as_mut_ptr(), c as i32) };
        if n <= 0 { return vec![-999.0]; }
        buf[..n as usize].iter().map(|v| *v as f64).collect()
    }

    #[napi]
    pub fn test_swiglu_mlp_bf16(max_count: Option<i32>) -> Vec<f64> {
        let c = max_count.unwrap_or(20) as usize;
        let mut buf = vec![0f32; c];
        let n = unsafe { sys::mlx_test_swiglu_mlp_bf16(buf.as_mut_ptr(), c as i32) };
        if n <= 0 { return vec![-999.0]; }
        buf[..n as usize].iter().map(|v| *v as f64).collect()
    }

    #[napi]
    pub fn test_decode_step_with_cache(max_count: Option<i32>) -> Vec<f64> {
        let c = max_count.unwrap_or(20) as usize;
        let mut buf = vec![0f32; c];
        let n = unsafe { sys::mlx_test_decode_step_with_cache(buf.as_mut_ptr(), c as i32) };
        if n <= 0 { return vec![-999.0]; }
        buf[..n as usize].iter().map(|v| *v as f64).collect()
    }

    #[napi]
    pub fn test_attn_layer_bf16(max_count: Option<i32>) -> Vec<f64> {
        let c = max_count.unwrap_or(20) as usize;
        let mut buf = vec![0f32; c];
        let n = unsafe { sys::mlx_test_attn_layer_bf16(buf.as_mut_ptr(), c as i32) };
        if n <= 0 { return vec![-999.0]; }
        buf[..n as usize].iter().map(|v| *v as f64).collect()
    }

    #[napi]
    pub fn test_first_4_layers_bf16(max_count: Option<i32>) -> Vec<f64> {
        let c = max_count.unwrap_or(20) as usize;
        let mut buf = vec![0f32; c];
        let n = unsafe { sys::mlx_test_first_4_layers_bf16(buf.as_mut_ptr(), c as i32) };
        if n <= 0 { return vec![-999.0]; }
        buf[..n as usize].iter().map(|v| *v as f64).collect()
    }

    #[napi]
    pub fn test_gdn_full_bf16(max_count: Option<i32>) -> Vec<f64> {
        let c = max_count.unwrap_or(20) as usize;
        let mut buf = vec![0f32; c];
        let n = unsafe { sys::mlx_test_gdn_full_bf16(buf.as_mut_ptr(), c as i32) };
        if n <= 0 { return vec![-999.0]; }
        buf[..n as usize].iter().map(|v| *v as f64).collect()
    }

    #[napi]
    pub fn test_gdn_multi_step_bf16(max_count: Option<i32>) -> Vec<f64> {
        let c = max_count.unwrap_or(20) as usize;
        let mut buf = vec![0f32; c];
        let n = unsafe { sys::mlx_test_gdn_multi_step_bf16(buf.as_mut_ptr(), c as i32) };
        if n <= 0 { return vec![-999.0]; }
        buf[..n as usize].iter().map(|v| *v as f64).collect()
    }

    #[napi]
    pub fn test_categorical_sampling_bf16(max_count: Option<i32>) -> Vec<f64> {
        let c = max_count.unwrap_or(10) as usize;
        let mut buf = vec![0f32; c];
        let n = unsafe { sys::mlx_test_categorical_sampling_bf16(buf.as_mut_ptr(), c as i32) };
        if n <= 0 { return vec![-999.0]; }
        buf[..n as usize].iter().map(|v| *v as f64).collect()
    }

    #[napi]
    pub fn test_gdn_layer_bf16(max_count: Option<i32>) -> Vec<f64> {
        let c = max_count.unwrap_or(20) as usize;
        let mut buf = vec![0f32; c];
        let n = unsafe { sys::mlx_test_gdn_layer_bf16(buf.as_mut_ptr(), c as i32) };
        if n <= 0 { return vec![-999.0]; }
        buf[..n as usize].iter().map(|v| *v as f64).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_fp8_roundtrips_representable_values() {
        // 1.5, -2.0, and 0.0 are exact E4M3 grid points → lossless roundtrip.
        let x = MxArray::from_float32(&[1.5, -2.0, 0.0], &[3]).unwrap();
        let q = x.to_fp8().unwrap();
        assert_eq!(q.dtype().unwrap(), DType::Uint8);
        let back = q.from_fp8(DType::Float32).unwrap();
        let v = back.to_float32().unwrap().as_ref().to_vec();
        assert_eq!(v, vec![1.5, -2.0, 0.0]);
    }
}
