use super::{DType, MxArray};
use mlx_sys as sys;
use napi::bindgen_prelude::*;
use napi_derive::napi;
#[cfg(target_family = "wasm")]
use std::ffi::CString;

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
}

#[cfg(target_family = "wasm")]
#[napi]
impl MxArray {
    /// NAPI-exposed variant that accepts the raw bf16 bytes as a `Uint8Array`.
    /// The input length must be `elements * 2`. This is the only path to build
    /// a CPU-resident bf16 weight buffer from JavaScript - the regular
    /// `fromFloat32(...).astype(BF16)` idiom produces a *GPU output*, which the
    /// WebGPU backend does not treat as an "upload-pending" buffer and so
    /// cannot be placed in the packed-bf16 storage mode. When the WebGPU
    /// backend has the packed-bf16 flag enabled AND the buffer is large
    /// enough (>= `min_elements`, default `PACKED_BF16_DEFAULT_MIN_ELEMENTS`),
    /// the underlying WebGPUBuffer is flipped to `StorageMode::PackedBf16`
    /// here so the first GPU upload stores the weights 2-per-u32 and downstream
    /// GEMV dispatches the packed kernel.
    ///
    /// The optional `min_elements` override exists strictly for the browser
    /// test suite, which needs to exercise the small-norm packed path at the
    /// production `NORM_PACKED_MIN_ELEMENTS = 256` threshold used by
    /// `gpu-worker.ts` (D=1024 norm weights are far below the default 4096
    /// GEMV-tuned floor). Callers outside tests MUST NOT pass this override.
    #[napi(js_name = "fromBfloat16Bytes")]
    pub fn from_bfloat16_bytes(
        data: &[u8],
        shape: &[i64],
        min_elements: Option<u32>,
    ) -> Result<Self> {
        if data.len() % 2 != 0 {
            return Err(napi::Error::from_reason(format!(
                "from_bfloat16_bytes: byte length {} is not even",
                data.len()
            )));
        }
        let n_elem = data.len() / 2;
        validate_data_shape(n_elem, shape, "from_bfloat16_bytes")?;
        // SAFETY: the caller-owned Uint8Array has the same lifetime as the
        // NAPI call; u16 alignment isn't required here because the underlying
        // FFI just memcpy's `n_elem * 2` bytes into the mlx_array allocation.
        let u16_ptr = data.as_ptr() as *const u16;
        let handle = unsafe { sys::mlx_array_from_bfloat16(u16_ptr, shape.as_ptr(), shape.len()) };
        let arr = MxArray::from_handle(handle, "array_from_bfloat16_bytes")?;
        // Opt into packed-bf16 storage when the runtime flag is enabled and
        // the buffer is weight-sized. A no-op if the flag is off or if the
        // buffer is too small for the packed GEMV to be worthwhile.
        const PACKED_BF16_DEFAULT_MIN_ELEMENTS: usize = 4096;
        let threshold = min_elements
            .map(|m| m as usize)
            .unwrap_or(PACKED_BF16_DEFAULT_MIN_ELEMENTS);
        unsafe {
            sys::mlx_wgpu_try_opt_in_packed_bf16(arr.as_raw_ptr(), threshold);
        }
        Ok(arr)
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

    /// Browser/WebGPU regression test for affine quantized kernels.
    ///
    /// Returns [
    ///   dequantize_max_error,
    ///   qmatmul_max_error,
    ///   qgemv_max_error,
    ///   gather_qmm_direct_max_error,
    ///   gather_qmm_explicit_lhs_max_error,
    ///   gather_qmm_sorted_max_error,
    ///   gather_qmm_sorted_vs_direct_max_error,
    /// ].
    #[napi]
    pub fn test_quantized_affine_ops(bits: i32) -> Result<Vec<f64>> {
        if !matches!(bits, 2 | 3 | 4 | 5 | 6 | 8) {
            return Err(Error::from_reason(format!(
                "unsupported affine quantization bits: {bits}"
            )));
        }

        fn pack_value(
            packed: &mut [u32],
            row: usize,
            row_words: usize,
            k: usize,
            bits: usize,
            value: u32,
        ) {
            fn write_byte(
                packed: &mut [u32],
                row: usize,
                row_words: usize,
                byte_offset: usize,
                byte: u32,
            ) {
                let word = row * row_words + byte_offset / 4;
                let shift = (byte_offset % 4) * 8;
                packed[word] |= (byte & 0xff) << shift;
            }

            let value = value & ((1u32 << bits) - 1);
            if bits == 3 {
                let byte_base = (k / 8) * 3;
                match k & 7 {
                    0 => write_byte(packed, row, row_words, byte_base, value),
                    1 => write_byte(packed, row, row_words, byte_base, value << 3),
                    2 => {
                        write_byte(packed, row, row_words, byte_base, (value & 0x3) << 6);
                        write_byte(packed, row, row_words, byte_base + 1, value >> 2);
                    }
                    3 => write_byte(packed, row, row_words, byte_base + 1, value << 1),
                    4 => write_byte(packed, row, row_words, byte_base + 1, value << 4),
                    5 => {
                        write_byte(packed, row, row_words, byte_base + 1, (value & 0x1) << 7);
                        write_byte(packed, row, row_words, byte_base + 2, value >> 1);
                    }
                    6 => write_byte(packed, row, row_words, byte_base + 2, value << 2),
                    _ => write_byte(packed, row, row_words, byte_base + 2, value << 5),
                }
                return;
            }

            if bits == 5 {
                let byte_base = (k / 8) * 5;
                match k & 7 {
                    0 => write_byte(packed, row, row_words, byte_base, value),
                    1 => {
                        write_byte(packed, row, row_words, byte_base, (value & 0x7) << 5);
                        write_byte(packed, row, row_words, byte_base + 1, value >> 3);
                    }
                    2 => write_byte(packed, row, row_words, byte_base + 1, value << 2),
                    3 => {
                        write_byte(packed, row, row_words, byte_base + 1, (value & 0x1) << 7);
                        write_byte(packed, row, row_words, byte_base + 2, value >> 1);
                    }
                    4 => {
                        write_byte(packed, row, row_words, byte_base + 2, (value & 0xf) << 4);
                        write_byte(packed, row, row_words, byte_base + 3, value >> 4);
                    }
                    5 => write_byte(packed, row, row_words, byte_base + 3, value << 1),
                    6 => {
                        write_byte(packed, row, row_words, byte_base + 3, (value & 0x3) << 6);
                        write_byte(packed, row, row_words, byte_base + 4, value >> 2);
                    }
                    _ => write_byte(packed, row, row_words, byte_base + 4, value << 3),
                }
                return;
            }

            if bits == 6 {
                let byte_base = (k / 4) * 3;
                match k & 3 {
                    0 => write_byte(packed, row, row_words, byte_base, value),
                    1 => {
                        write_byte(packed, row, row_words, byte_base, (value & 0x3) << 6);
                        write_byte(packed, row, row_words, byte_base + 1, value >> 2);
                    }
                    2 => {
                        write_byte(packed, row, row_words, byte_base + 1, (value & 0xf) << 4);
                        write_byte(packed, row, row_words, byte_base + 2, value >> 4);
                    }
                    _ => write_byte(packed, row, row_words, byte_base + 2, value << 2),
                }
                return;
            }

            let bit_offset = k * bits;
            let word = bit_offset / 32;
            let shift = bit_offset % 32;
            packed[row * row_words + word] |= value << shift;
            if shift + bits > 32 {
                packed[row * row_words + word + 1] |= value >> (32 - shift);
            }
        }

        fn quant_value(seed: usize, mask: u32) -> u32 {
            ((seed * 13 + seed / 3 + 7) as u32) & mask
        }

        fn qmatmul(
            x: &MxArray,
            w: &MxArray,
            scales: &MxArray,
            biases: &MxArray,
            group_size: i32,
            bits: i32,
        ) -> Result<MxArray> {
            let mode = CString::new("affine").unwrap();
            let handle = unsafe {
                sys::mlx_quantized_matmul(
                    x.handle.0,
                    w.handle.0,
                    scales.handle.0,
                    biases.handle.0,
                    true,
                    group_size,
                    bits,
                    mode.as_ptr(),
                )
            };
            MxArray::from_handle(handle, "test_quantized_matmul")
        }

        fn dequantize(
            w: &MxArray,
            scales: &MxArray,
            biases: &MxArray,
            group_size: i32,
            bits: i32,
        ) -> Result<MxArray> {
            let mode = CString::new("affine").unwrap();
            let handle = unsafe {
                sys::mlx_dequantize(
                    w.handle.0,
                    scales.handle.0,
                    biases.handle.0,
                    group_size,
                    bits,
                    -1,
                    mode.as_ptr(),
                )
            };
            MxArray::from_handle(handle, "test_dequantize")
        }

        fn gather_qmm(
            x: &MxArray,
            w: &MxArray,
            scales: &MxArray,
            biases: &MxArray,
            lhs_indices: Option<&MxArray>,
            rhs_indices: &MxArray,
            group_size: i32,
            bits: i32,
            sorted: bool,
        ) -> Result<MxArray> {
            let mode = CString::new("affine").unwrap();
            let handle = unsafe {
                sys::mlx_gather_qmm(
                    x.handle.0,
                    w.handle.0,
                    scales.handle.0,
                    biases.handle.0,
                    lhs_indices.map_or(std::ptr::null_mut(), |idx| idx.handle.0),
                    rhs_indices.handle.0,
                    true,
                    group_size,
                    bits,
                    mode.as_ptr(),
                    sorted,
                )
            };
            MxArray::from_handle(handle, "test_gather_qmm")
        }

        fn max_abs_diff(actual: &[f32], expected: &[f32]) -> f64 {
            actual
                .iter()
                .zip(expected.iter())
                .map(|(a, e)| (*a - *e).abs() as f64)
                .fold(0.0, f64::max)
        }

        let bits_usize = bits as usize;
        let group_size = 64i32;
        let group_size_usize = group_size as usize;
        let k_dim = 256usize;
        let num_groups = k_dim / group_size_usize;
        let w_cols = k_dim * bits_usize / 32;
        let mask = (1u32 << bits_usize) - 1;

        let m_dim = 3usize;
        let n_dim = 5usize;
        let mut x_data = vec![0.0f32; m_dim * k_dim];
        for (i, v) in x_data.iter_mut().enumerate() {
            *v = ((i % 29) as f32 - 14.0) / 17.0;
        }

        let mut w_data = vec![0u32; n_dim * w_cols];
        let mut scales_data = vec![0.0f32; n_dim * num_groups];
        let mut biases_data = vec![0.0f32; n_dim * num_groups];
        for n in 0..n_dim {
            for group in 0..num_groups {
                let idx = n * num_groups + group;
                scales_data[idx] = 0.015625 * ((idx % 7) as f32 + 1.0);
                biases_data[idx] = -0.375 + (idx as f32) * 0.03125;
            }
            for k in 0..k_dim {
                let q = quant_value(n * 97 + k, mask);
                pack_value(&mut w_data, n, w_cols, k, bits_usize, q);
            }
        }

        let x = MxArray::from_float32(&x_data, &[m_dim as i64, k_dim as i64])?;
        let x_gemv = MxArray::from_float32(&x_data[..k_dim], &[1, k_dim as i64])?;
        let w = MxArray::from_uint32(&w_data, &[n_dim as i64, w_cols as i64])?;
        let scales = MxArray::from_float32(&scales_data, &[n_dim as i64, num_groups as i64])?;
        let biases = MxArray::from_float32(&biases_data, &[n_dim as i64, num_groups as i64])?;
        let q_out = qmatmul(&x, &w, &scales, &biases, group_size, bits)?;
        let q_gemv_out = qmatmul(&x_gemv, &w, &scales, &biases, group_size, bits)?;
        let dequantized = dequantize(&w, &scales, &biases, group_size, bits)?;
        q_out.eval();
        q_gemv_out.eval();
        dequantized.eval();
        let q_actual = q_out.to_float32_vec()?;
        let q_gemv_actual = q_gemv_out.to_float32_vec()?;
        let deq_actual = dequantized.to_float32_vec()?;
        let mut deq_expected = vec![0.0f32; n_dim * k_dim];
        let mut q_expected = vec![0.0f32; m_dim * n_dim];
        for n in 0..n_dim {
            for k in 0..k_dim {
                let q = quant_value(n * 97 + k, mask) as f32;
                let group = k / group_size_usize;
                let sb_idx = n * num_groups + group;
                deq_expected[n * k_dim + k] = q * scales_data[sb_idx] + biases_data[sb_idx];
            }
        }
        for m in 0..m_dim {
            for n in 0..n_dim {
                let mut acc = 0.0f32;
                for k in 0..k_dim {
                    let q = quant_value(n * 97 + k, mask) as f32;
                    let group = k / group_size_usize;
                    let sb_idx = n * num_groups + group;
                    let deq = q * scales_data[sb_idx] + biases_data[sb_idx];
                    acc += x_data[m * k_dim + k] * deq;
                }
                q_expected[m * n_dim + n] = acc;
            }
        }
        let deq_err = max_abs_diff(&deq_actual, &deq_expected);
        let q_err = max_abs_diff(&q_actual, &q_expected);
        let q_gemv_err = max_abs_diff(&q_gemv_actual, &q_expected[..n_dim]);

        let num_experts = 3usize;
        let tokens = 20usize;
        let top_k = 4usize;
        let routed = tokens * top_k;
        let mut gx_data = vec![0.0f32; tokens * k_dim];
        for (i, v) in gx_data.iter_mut().enumerate() {
            *v = ((i * 5 % 37) as f32 - 18.0) / 23.0;
        }
        let mut gw_data = vec![0u32; num_experts * n_dim * w_cols];
        let mut gs_data = vec![0.0f32; num_experts * n_dim * num_groups];
        let mut gb_data = vec![0.0f32; num_experts * n_dim * num_groups];
        for e in 0..num_experts {
            for n in 0..n_dim {
                let row = e * n_dim + n;
                for group in 0..num_groups {
                    let idx = row * num_groups + group;
                    gs_data[idx] = 0.0125 * ((idx % 11) as f32 + 1.0);
                    gb_data[idx] = -0.25 + (idx as f32) * 0.015625;
                }
                for k in 0..k_dim {
                    let q = quant_value(e * 211 + n * 97 + k, mask);
                    pack_value(&mut gw_data, row, w_cols, k, bits_usize, q);
                }
            }
        }
        let mut idx_data = vec![0u32; routed];
        let mut lhs_data = vec![0u32; routed];
        for t in 0..tokens {
            for j in 0..top_k {
                idx_data[t * top_k + j] = ((t * 2 + j * 5 + 1) % num_experts) as u32;
                lhs_data[t * top_k + j] = t as u32;
            }
        }

        let gx = MxArray::from_float32(&gx_data, &[tokens as i64, k_dim as i64])?;
        let gw =
            MxArray::from_uint32(&gw_data, &[num_experts as i64, n_dim as i64, w_cols as i64])?;
        let gs = MxArray::from_float32(
            &gs_data,
            &[num_experts as i64, n_dim as i64, num_groups as i64],
        )?;
        let gb = MxArray::from_float32(
            &gb_data,
            &[num_experts as i64, n_dim as i64, num_groups as i64],
        )?;
        let indices = MxArray::from_uint32(&idx_data, &[tokens as i64, top_k as i64])?;
        let lhs_indices = MxArray::from_uint32(&lhs_data, &[tokens as i64, top_k as i64])?;

        let x_expanded = gx.reshape(&[tokens as i64, 1, 1, k_dim as i64])?;
        let direct = gather_qmm(
            &x_expanded,
            &gw,
            &gs,
            &gb,
            None,
            &indices,
            group_size,
            bits,
            false,
        )?;
        let direct_explicit_lhs = gather_qmm(
            &x_expanded,
            &gw,
            &gs,
            &gb,
            Some(&lhs_indices),
            &indices,
            group_size,
            bits,
            false,
        )?;

        let flat_indices = indices.reshape(&[routed as i64])?;
        let order = flat_indices.argsort(Some(-1))?;
        let inv_order = order.argsort(Some(-1))?;
        let idx_sorted = flat_indices.take(&order, 0)?;
        let x_flat = x_expanded.reshape(&[-1, 1, k_dim as i64])?;
        let token_indices = order.floor_divide(&MxArray::scalar_int(top_k as i32)?)?;
        let x_sorted = x_flat.take(&token_indices, 0)?;
        let sorted = gather_qmm(
            &x_sorted,
            &gw,
            &gs,
            &gb,
            None,
            &idx_sorted,
            group_size,
            bits,
            true,
        )?;
        let unsorted =
            sorted
                .take(&inv_order, 0)?
                .reshape(&[tokens as i64, top_k as i64, 1, n_dim as i64])?;

        let mut gather_expected = vec![0.0f32; tokens * top_k * n_dim];
        for t in 0..tokens {
            for j in 0..top_k {
                let expert = idx_data[t * top_k + j] as usize;
                for n in 0..n_dim {
                    let row = expert * n_dim + n;
                    let mut acc = 0.0f32;
                    for k in 0..k_dim {
                        let q = quant_value(expert * 211 + n * 97 + k, mask) as f32;
                        let group = k / group_size_usize;
                        let sb_idx = row * num_groups + group;
                        let deq = q * gs_data[sb_idx] + gb_data[sb_idx];
                        acc += gx_data[t * k_dim + k] * deq;
                    }
                    gather_expected[(t * top_k + j) * n_dim + n] = acc;
                }
            }
        }

        direct.eval();
        direct_explicit_lhs.eval();
        unsorted.eval();
        let direct_vals = direct.to_float32_vec()?;
        let direct_explicit_lhs_vals = direct_explicit_lhs.to_float32_vec()?;
        let unsorted_vals = unsorted.to_float32_vec()?;
        let direct_err = max_abs_diff(&direct_vals, &gather_expected);
        let direct_explicit_lhs_err = max_abs_diff(&direct_explicit_lhs_vals, &gather_expected);
        let sorted_err = max_abs_diff(&unsorted_vals, &gather_expected);
        let gather_err = max_abs_diff(&unsorted_vals, &direct_vals);

        // Keep the shape/group constants used above live in this test; changing
        // them accidentally can make non-power-of-two packing stop crossing word
        // boundaries.
        debug_assert_eq!(k_dim % group_size_usize, 0);
        Ok(vec![
            deq_err,
            q_err,
            q_gemv_err,
            direct_err,
            direct_explicit_lhs_err,
            sorted_err,
            gather_err,
        ])
    }

    /// Browser/WebGPU regression test for model-scale affine quantized decode
    /// shapes. Qwen3.6 MoE quantized checkpoints use K=2048, group_size=64,
    /// bf16 activations/scales/biases, and 3/4/5/6/8-bit packed rows; the smaller
    /// smoke test above does not exercise those dispatch and packing paths.
    #[napi]
    pub fn test_quantized_affine_model_shape_ops(bits: i32) -> Result<Vec<f64>> {
        if !matches!(bits, 3 | 4 | 5 | 6 | 8) {
            return Err(Error::from_reason(format!(
                "model-shape affine test supports bits 3, 4, 5, 6, and 8; got {bits}"
            )));
        }

        fn pack_value(
            packed: &mut [u32],
            row: usize,
            row_words: usize,
            k: usize,
            bits: usize,
            value: u32,
        ) {
            fn write_byte(
                packed: &mut [u32],
                row: usize,
                row_words: usize,
                byte_offset: usize,
                byte: u32,
            ) {
                let word = row * row_words + byte_offset / 4;
                let shift = (byte_offset % 4) * 8;
                packed[word] |= (byte & 0xff) << shift;
            }

            let value = value & ((1u32 << bits) - 1);
            if bits == 3 {
                let byte_base = (k / 8) * 3;
                match k & 7 {
                    0 => write_byte(packed, row, row_words, byte_base, value),
                    1 => write_byte(packed, row, row_words, byte_base, value << 3),
                    2 => {
                        write_byte(packed, row, row_words, byte_base, (value & 0x3) << 6);
                        write_byte(packed, row, row_words, byte_base + 1, value >> 2);
                    }
                    3 => write_byte(packed, row, row_words, byte_base + 1, value << 1),
                    4 => write_byte(packed, row, row_words, byte_base + 1, value << 4),
                    5 => {
                        write_byte(packed, row, row_words, byte_base + 1, (value & 0x1) << 7);
                        write_byte(packed, row, row_words, byte_base + 2, value >> 1);
                    }
                    6 => write_byte(packed, row, row_words, byte_base + 2, value << 2),
                    _ => write_byte(packed, row, row_words, byte_base + 2, value << 5),
                }
                return;
            }

            if bits == 5 {
                let byte_base = (k / 8) * 5;
                match k & 7 {
                    0 => write_byte(packed, row, row_words, byte_base, value),
                    1 => {
                        write_byte(packed, row, row_words, byte_base, (value & 0x7) << 5);
                        write_byte(packed, row, row_words, byte_base + 1, value >> 3);
                    }
                    2 => write_byte(packed, row, row_words, byte_base + 1, value << 2),
                    3 => {
                        write_byte(packed, row, row_words, byte_base + 1, (value & 0x1) << 7);
                        write_byte(packed, row, row_words, byte_base + 2, value >> 1);
                    }
                    4 => {
                        write_byte(packed, row, row_words, byte_base + 2, (value & 0xf) << 4);
                        write_byte(packed, row, row_words, byte_base + 3, value >> 4);
                    }
                    5 => write_byte(packed, row, row_words, byte_base + 3, value << 1),
                    6 => {
                        write_byte(packed, row, row_words, byte_base + 3, (value & 0x3) << 6);
                        write_byte(packed, row, row_words, byte_base + 4, value >> 2);
                    }
                    _ => write_byte(packed, row, row_words, byte_base + 4, value << 3),
                }
                return;
            }

            if bits == 6 {
                let byte_base = (k / 4) * 3;
                match k & 3 {
                    0 => write_byte(packed, row, row_words, byte_base, value),
                    1 => {
                        write_byte(packed, row, row_words, byte_base, (value & 0x3) << 6);
                        write_byte(packed, row, row_words, byte_base + 1, value >> 2);
                    }
                    2 => {
                        write_byte(packed, row, row_words, byte_base + 1, (value & 0xf) << 4);
                        write_byte(packed, row, row_words, byte_base + 2, value >> 4);
                    }
                    _ => write_byte(packed, row, row_words, byte_base + 2, value << 2),
                }
                return;
            }

            let bit_offset = k * bits;
            let word = bit_offset / 32;
            let shift = bit_offset % 32;
            packed[row * row_words + word] |= value << shift;
            if shift + bits > 32 {
                packed[row * row_words + word + 1] |= value >> (32 - shift);
            }
        }

        fn quant_value(seed: usize, mask: u32) -> u32 {
            ((seed
                .wrapping_mul(1103515245)
                .wrapping_add(seed >> 2)
                .wrapping_add(12345)) as u32)
                & mask
        }

        fn qmatmul(
            x: &MxArray,
            w: &MxArray,
            scales: &MxArray,
            biases: &MxArray,
            group_size: i32,
            bits: i32,
        ) -> Result<MxArray> {
            let mode = CString::new("affine").unwrap();
            let handle = unsafe {
                sys::mlx_quantized_matmul(
                    x.handle.0,
                    w.handle.0,
                    scales.handle.0,
                    biases.handle.0,
                    true,
                    group_size,
                    bits,
                    mode.as_ptr(),
                )
            };
            MxArray::from_handle(handle, "test_model_shape_quantized_matmul")
        }

        fn gather_qmm(
            x: &MxArray,
            w: &MxArray,
            scales: &MxArray,
            biases: &MxArray,
            rhs_indices: &MxArray,
            group_size: i32,
            bits: i32,
            sorted: bool,
        ) -> Result<MxArray> {
            let mode = CString::new("affine").unwrap();
            let handle = unsafe {
                sys::mlx_gather_qmm(
                    x.handle.0,
                    w.handle.0,
                    scales.handle.0,
                    biases.handle.0,
                    std::ptr::null_mut(),
                    rhs_indices.handle.0,
                    true,
                    group_size,
                    bits,
                    mode.as_ptr(),
                    sorted,
                )
            };
            MxArray::from_handle(handle, "test_model_shape_gather_qmm")
        }

        fn dequantize(
            w: &MxArray,
            scales: &MxArray,
            biases: &MxArray,
            group_size: i32,
            bits: i32,
        ) -> Result<MxArray> {
            let mode = CString::new("affine").unwrap();
            let handle = unsafe {
                sys::mlx_dequantize(
                    w.handle.0,
                    scales.handle.0,
                    biases.handle.0,
                    group_size,
                    bits,
                    -1,
                    mode.as_ptr(),
                )
            };
            MxArray::from_handle(handle, "test_model_shape_dequantize")
        }

        fn max_abs_diff(actual: &[f32], expected: &[f32]) -> f64 {
            actual
                .iter()
                .zip(expected.iter())
                .map(|(a, e)| (*a - *e).abs() as f64)
                .fold(0.0, f64::max)
        }

        let bits_usize = bits as usize;
        let group_size = 64i32;
        let group_size_usize = group_size as usize;
        let k_dim = 2048usize;
        let num_groups = k_dim / group_size_usize;
        let w_cols = k_dim * bits_usize / 32;
        let mask = (1u32 << bits_usize) - 1;

        // GEMV/GEMM path: large enough to use many column workgroups, but
        // still cheap enough to run in a browser worker during regression.
        let m_dim = 2usize;
        let n_dim = if bits == 6 { 4096usize } else { 2048usize };
        let mut x_data = vec![0.0f32; m_dim * k_dim];
        for (i, v) in x_data.iter_mut().enumerate() {
            *v = ((i * 17 % 97) as f32 - 48.0) / 97.0;
        }

        let mut w_data = vec![0u32; n_dim * w_cols];
        let mut scales_data = vec![0.0f32; n_dim * num_groups];
        let mut biases_data = vec![0.0f32; n_dim * num_groups];
        for n in 0..n_dim {
            for group in 0..num_groups {
                let idx = n * num_groups + group;
                scales_data[idx] = 0.001953125 * ((idx % 13) as f32 + 1.0);
                biases_data[idx] = -0.0625 + ((idx % 17) as f32) * 0.0078125;
            }
            for k in 0..k_dim {
                let q = quant_value(n * 4099 + k, mask);
                pack_value(&mut w_data, n, w_cols, k, bits_usize, q);
            }
        }

        let x = MxArray::from_float32(&x_data, &[m_dim as i64, k_dim as i64])?
            .astype(DType::BFloat16)?;
        let x_gemv =
            MxArray::from_float32(&x_data[..k_dim], &[1, k_dim as i64])?.astype(DType::BFloat16)?;
        let w = MxArray::from_uint32(&w_data, &[n_dim as i64, w_cols as i64])?;
        let scales = MxArray::from_float32(&scales_data, &[n_dim as i64, num_groups as i64])?
            .astype(DType::BFloat16)?;
        let biases = MxArray::from_float32(&biases_data, &[n_dim as i64, num_groups as i64])?
            .astype(DType::BFloat16)?;

        let q_out = qmatmul(&x, &w, &scales, &biases, group_size, bits)?;
        let q_gemv_out = qmatmul(&x_gemv, &w, &scales, &biases, group_size, bits)?;
        q_out.eval();
        q_gemv_out.eval();
        let q_actual = q_out.to_float32_vec()?;
        let q_gemv_actual = q_gemv_out.to_float32_vec()?;

        let mut q_expected = vec![0.0f32; m_dim * n_dim];
        for m in 0..m_dim {
            for n in 0..n_dim {
                let mut acc = 0.0f32;
                for k in 0..k_dim {
                    let q = quant_value(n * 4099 + k, mask) as f32;
                    let group = k / group_size_usize;
                    let sb_idx = n * num_groups + group;
                    let deq = q * scales_data[sb_idx] + biases_data[sb_idx];
                    acc += x_data[m * k_dim + k] * deq;
                }
                q_expected[m * n_dim + n] = acc;
            }
        }
        let q_err = max_abs_diff(&q_actual, &q_expected);
        let q_gemv_err = max_abs_diff(&q_gemv_actual, &q_expected[..n_dim]);

        let (large_vocab_gemv_err, large_vocab_argmax_ok) = if bits == 6 {
            // Qwen3.6 MoE uses a large 6-bit lm_head
            // [vocab_size, hidden_size]. Keep N just above WebGPU's minimum
            // 65535 workgroups-per-dimension boundary so the 2-D GEMV tile
            // addressing path is exercised without allocating the full 248K
            // vocab in the smoke test.
            let vocab_dim = 70_000usize;
            let selected_ks = [0usize, 63, 64, 127, 1024, 2047];
            let mut vx_data = vec![0.0f32; k_dim];
            for (i, &k) in selected_ks.iter().enumerate() {
                vx_data[k] = 0.125 + i as f32 * 0.03125;
            }
            let mut vw_data = vec![0u32; vocab_dim * w_cols];
            let mut vs_data = vec![0.0f32; vocab_dim * num_groups];
            let mut vb_data = vec![0.0f32; vocab_dim * num_groups];
            for n in 0..vocab_dim {
                for group in 0..num_groups {
                    let idx = n * num_groups + group;
                    vs_data[idx] = 0.001953125 * ((idx % 13) as f32 + 1.0);
                    vb_data[idx] = -0.03125 + ((idx % 7) as f32) * 0.00390625;
                }
                for &k in &selected_ks {
                    let q = quant_value(n * 32771 + k, mask);
                    pack_value(&mut vw_data, n, w_cols, k, bits_usize, q);
                }
            }

            let vx =
                MxArray::from_float32(&vx_data, &[1, k_dim as i64])?.astype(DType::BFloat16)?;
            let vw = MxArray::from_uint32(&vw_data, &[vocab_dim as i64, w_cols as i64])?;
            let vs = MxArray::from_float32(&vs_data, &[vocab_dim as i64, num_groups as i64])?
                .astype(DType::BFloat16)?;
            let vb = MxArray::from_float32(&vb_data, &[vocab_dim as i64, num_groups as i64])?
                .astype(DType::BFloat16)?;
            let out = qmatmul(&vx, &vw, &vs, &vb, group_size, bits)?;
            out.eval();
            let actual = out.to_float32_vec()?;
            let mut expected = vec![0.0f32; vocab_dim];
            for n in 0..vocab_dim {
                let mut acc = 0.0f32;
                for &k in &selected_ks {
                    let q = quant_value(n * 32771 + k, mask) as f32;
                    let group = k / group_size_usize;
                    let sb_idx = n * num_groups + group;
                    acc += vx_data[k] * (q * vs_data[sb_idx] + vb_data[sb_idx]);
                }
                expected[n] = acc;
            }
            let actual_argmax = actual
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(usize::MAX);
            let expected_argmax = expected
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(usize::MAX);
            (
                max_abs_diff(&actual, &expected),
                if actual_argmax == expected_argmax {
                    0.0
                } else {
                    1.0
                },
            )
        } else {
            (0.0, 0.0)
        };

        // Batched activations with unbatched quantized weights must broadcast
        // the weight/scales/biases batch stride as zero. This is the common
        // projection shape for transformer batches.
        let batch_dim = 2usize;
        let batched_m_dim = 3usize;
        let batched_n_dim = 512usize;
        let mut bx_data = vec![0.0f32; batch_dim * batched_m_dim * k_dim];
        for (i, v) in bx_data.iter_mut().enumerate() {
            *v = ((i * 23 % 101) as f32 - 50.0) / 101.0;
        }
        let mut bw_data = vec![0u32; batched_n_dim * w_cols];
        let mut bs_data = vec![0.0f32; batched_n_dim * num_groups];
        let mut bb_data = vec![0.0f32; batched_n_dim * num_groups];
        for n in 0..batched_n_dim {
            for group in 0..num_groups {
                let idx = n * num_groups + group;
                bs_data[idx] = 0.001953125 * ((idx % 7) as f32 + 1.0);
                bb_data[idx] = -0.046875 + ((idx % 11) as f32) * 0.005859375;
            }
            for k in 0..k_dim {
                let q = quant_value(n * 8191 + k, mask);
                pack_value(&mut bw_data, n, w_cols, k, bits_usize, q);
            }
        }
        let bx = MxArray::from_float32(
            &bx_data,
            &[batch_dim as i64, batched_m_dim as i64, k_dim as i64],
        )?
        .astype(DType::BFloat16)?;
        let bw = MxArray::from_uint32(&bw_data, &[batched_n_dim as i64, w_cols as i64])?;
        let bs = MxArray::from_float32(&bs_data, &[batched_n_dim as i64, num_groups as i64])?
            .astype(DType::BFloat16)?;
        let bb = MxArray::from_float32(&bb_data, &[batched_n_dim as i64, num_groups as i64])?
            .astype(DType::BFloat16)?;
        let batched_out = qmatmul(&bx, &bw, &bs, &bb, group_size, bits)?;
        batched_out.eval();
        let batched_actual = batched_out.to_float32_vec()?;
        let mut batched_expected = vec![0.0f32; batch_dim * batched_m_dim * batched_n_dim];
        for b in 0..batch_dim {
            for m in 0..batched_m_dim {
                for n in 0..batched_n_dim {
                    let mut acc = 0.0f32;
                    for k in 0..k_dim {
                        let q = quant_value(n * 8191 + k, mask) as f32;
                        let group = k / group_size_usize;
                        let sb_idx = n * num_groups + group;
                        let deq = q * bs_data[sb_idx] + bb_data[sb_idx];
                        let x_idx = (b * batched_m_dim + m) * k_dim + k;
                        acc += bx_data[x_idx] * deq;
                    }
                    batched_expected[(b * batched_m_dim + m) * batched_n_dim + n] = acc;
                }
            }
        }
        let batched_err = max_abs_diff(&batched_actual, &batched_expected);

        // MoE gather path for Q3 expert weights. Keep the expert count at the
        // real 256 so index arithmetic reaches the same address range pattern.
        let num_experts = 256usize;
        let expert_out = 32usize;
        let top_k = 8usize;
        let routed = top_k;
        let mut gx_data = vec![0.0f32; k_dim];
        for (i, v) in gx_data.iter_mut().enumerate() {
            *v = ((i * 19 % 89) as f32 - 44.0) / 89.0;
        }
        let mut gw_data = vec![0u32; num_experts * expert_out * w_cols];
        let mut gs_data = vec![0.0f32; num_experts * expert_out * num_groups];
        let mut gb_data = vec![0.0f32; num_experts * expert_out * num_groups];
        for e in 0..num_experts {
            for n in 0..expert_out {
                let row = e * expert_out + n;
                for group in 0..num_groups {
                    let idx = row * num_groups + group;
                    gs_data[idx] = 0.001953125 * ((idx % 11) as f32 + 1.0);
                    gb_data[idx] = -0.03125 + ((idx % 7) as f32) * 0.0078125;
                }
                for k in 0..k_dim {
                    let q = quant_value(e * 65537 + n * 4099 + k, mask);
                    pack_value(&mut gw_data, row, w_cols, k, bits_usize, q);
                }
            }
        }
        let rhs_data = [255u32, 0, 127, 1, 254, 64, 192, 5];
        let gx = MxArray::from_float32(&gx_data, &[1, 1, k_dim as i64])?.astype(DType::BFloat16)?;
        let gw = MxArray::from_uint32(
            &gw_data,
            &[num_experts as i64, expert_out as i64, w_cols as i64],
        )?;
        let gs = MxArray::from_float32(
            &gs_data,
            &[num_experts as i64, expert_out as i64, num_groups as i64],
        )?
        .astype(DType::BFloat16)?;
        let gb = MxArray::from_float32(
            &gb_data,
            &[num_experts as i64, expert_out as i64, num_groups as i64],
        )?
        .astype(DType::BFloat16)?;
        let rhs = MxArray::from_uint32(&rhs_data, &[routed as i64])?;
        let gather_out = gather_qmm(&gx, &gw, &gs, &gb, &rhs, group_size, bits, true)?;
        gather_out.eval();
        let gather_actual = gather_out.to_float32_vec()?;
        let mut gather_expected = vec![0.0f32; routed * expert_out];
        for slot in 0..routed {
            let expert = rhs_data[slot] as usize;
            for n in 0..expert_out {
                let row = expert * expert_out + n;
                let mut acc = 0.0f32;
                for k in 0..k_dim {
                    let q = quant_value(expert * 65537 + n * 4099 + k, mask) as f32;
                    let group = k / group_size_usize;
                    let sb_idx = row * num_groups + group;
                    let deq = q * gs_data[sb_idx] + gb_data[sb_idx];
                    acc += gx_data[k] * deq;
                }
                gather_expected[slot * expert_out + n] = acc;
            }
        }
        let gather_err = max_abs_diff(&gather_actual, &gather_expected);

        let exact_sorted_gather_err = if bits == 3 {
            // Real Qwen3.6 MoE decode/prefill sorts routed tokens before
            // calling gather_qmm. This covers the exact [routes, 1, K] x-shape
            // and [routes] rhs-index shape with the real 512-column expert
            // projection size used by Q3 gate/up expert weights.
            let exact_expert_out = 512usize;
            let mut ex_data = vec![0.0f32; routed * k_dim];
            for slot in 0..routed {
                for k in 0..k_dim {
                    ex_data[slot * k_dim + k] =
                        (((slot + 3) * (k + 17) % 127) as f32 - 63.0) / 127.0;
                }
            }
            let mut ew_data = vec![0u32; num_experts * exact_expert_out * w_cols];
            let mut es_data = vec![0.0f32; num_experts * exact_expert_out * num_groups];
            let mut eb_data = vec![0.0f32; num_experts * exact_expert_out * num_groups];
            for e in 0..num_experts {
                for n in 0..exact_expert_out {
                    let row = e * exact_expert_out + n;
                    for group in 0..num_groups {
                        let idx = row * num_groups + group;
                        es_data[idx] = 0.001953125 * ((idx % 13) as f32 + 1.0);
                        eb_data[idx] = -0.0234375 + ((idx % 11) as f32) * 0.00390625;
                    }
                    for k in 0..k_dim {
                        let q = quant_value(e * 131071 + n * 4099 + k, mask);
                        pack_value(&mut ew_data, row, w_cols, k, bits_usize, q);
                    }
                }
            }
            let ex = MxArray::from_float32(&ex_data, &[routed as i64, 1, k_dim as i64])?
                .astype(DType::BFloat16)?;
            let ew = MxArray::from_uint32(
                &ew_data,
                &[num_experts as i64, exact_expert_out as i64, w_cols as i64],
            )?;
            let es = MxArray::from_float32(
                &es_data,
                &[
                    num_experts as i64,
                    exact_expert_out as i64,
                    num_groups as i64,
                ],
            )?
            .astype(DType::BFloat16)?;
            let eb = MxArray::from_float32(
                &eb_data,
                &[
                    num_experts as i64,
                    exact_expert_out as i64,
                    num_groups as i64,
                ],
            )?
            .astype(DType::BFloat16)?;
            let exact_out = gather_qmm(&ex, &ew, &es, &eb, &rhs, group_size, bits, true)?;
            exact_out.eval();
            let actual = exact_out.to_float32_vec()?;
            let mut expected = vec![0.0f32; routed * exact_expert_out];
            for slot in 0..routed {
                let expert = rhs_data[slot] as usize;
                for n in 0..exact_expert_out {
                    let row = expert * exact_expert_out + n;
                    let mut acc = 0.0f32;
                    for k in 0..k_dim {
                        let q = quant_value(expert * 131071 + n * 4099 + k, mask) as f32;
                        let group = k / group_size_usize;
                        let sb_idx = row * num_groups + group;
                        let deq = q * es_data[sb_idx] + eb_data[sb_idx];
                        acc += ex_data[slot * k_dim + k] * deq;
                    }
                    expected[slot * exact_expert_out + n] = acc;
                }
            }
            max_abs_diff(&actual, &expected)
        } else {
            0.0
        };

        let large_dequant_err = if bits == 5 {
            // Embedding dequantization for Qwen3.6 uses bits=5, K=2048 and a
            // very large row count. This row count is enough to force the
            // WebGPU dequantize kernel into its 2-D dispatch addressing path.
            let rows = 10_000usize;
            let mut lw_data = vec![0u32; rows * w_cols];
            let mut ls_data = vec![0.0f32; rows * num_groups];
            let mut lb_data = vec![0.0f32; rows * num_groups];
            for row in 0..rows {
                for group in 0..num_groups {
                    let idx = row * num_groups + group;
                    ls_data[idx] = 0.001953125 * ((idx % 9) as f32 + 1.0);
                    lb_data[idx] = -0.046875 + ((idx % 5) as f32) * 0.0078125;
                }
                for k in 0..k_dim {
                    let q = quant_value(row * 4099 + k, mask);
                    pack_value(&mut lw_data, row, w_cols, k, bits_usize, q);
                }
            }
            let lw = MxArray::from_uint32(&lw_data, &[rows as i64, w_cols as i64])?;
            let ls = MxArray::from_float32(&ls_data, &[rows as i64, num_groups as i64])?
                .astype(DType::BFloat16)?;
            let lb = MxArray::from_float32(&lb_data, &[rows as i64, num_groups as i64])?
                .astype(DType::BFloat16)?;
            let deq = dequantize(&lw, &ls, &lb, group_size, bits)?;
            let sample_rows = [0usize, 1, 4096, 8191, 9999];
            let idx = MxArray::from_uint32(
                &sample_rows.iter().map(|v| *v as u32).collect::<Vec<_>>(),
                &[sample_rows.len() as i64],
            )?;
            let sampled = deq.take(&idx, 0)?;
            sampled.eval();
            let actual = sampled.to_float32_vec()?;
            let mut expected = vec![0.0f32; sample_rows.len() * k_dim];
            for (sr, row) in sample_rows.iter().copied().enumerate() {
                for k in 0..k_dim {
                    let q = quant_value(row * 4099 + k, mask) as f32;
                    let group = k / group_size_usize;
                    let sb_idx = row * num_groups + group;
                    expected[sr * k_dim + k] = q * ls_data[sb_idx] + lb_data[sb_idx];
                }
            }
            max_abs_diff(&actual, &expected)
        } else {
            0.0
        };

        let split_projection_concat_err = if bits == 5 {
            // Qwen3.6-35B-A3B stores linear_attn.in_proj_qkv and in_proj_z as
            // separate 5-bit affine tensors, then sanitize_weights merges them
            // with MLX concatenate before QuantizedLinear construction. This
            // exercises that exact WebGPU path without reading packed U32
            // weights back through the test-only to_uint32 materialization.
            let qkv_rows = 8192usize;
            let z_rows = 4096usize;
            let total_rows = qkv_rows + z_rows;
            let mut qkv_w = vec![0u32; qkv_rows * w_cols];
            let mut z_w = vec![0u32; z_rows * w_cols];
            let mut qkv_scales = vec![0.0f32; qkv_rows * num_groups];
            let mut z_scales = vec![0.0f32; z_rows * num_groups];
            let mut qkv_biases = vec![0.0f32; qkv_rows * num_groups];
            let mut z_biases = vec![0.0f32; z_rows * num_groups];
            for row in 0..qkv_rows {
                for group in 0..num_groups {
                    let idx = row * num_groups + group;
                    qkv_scales[idx] = 0.001953125 * ((idx % 17) as f32 + 1.0);
                    qkv_biases[idx] = -0.0390625 + ((idx % 19) as f32) * 0.00390625;
                }
                for k in 0..k_dim {
                    let q = quant_value(row * 4099 + k, mask);
                    pack_value(&mut qkv_w, row, w_cols, k, bits_usize, q);
                }
            }
            for row in 0..z_rows {
                for group in 0..num_groups {
                    let idx = row * num_groups + group;
                    z_scales[idx] = 0.001953125 * ((idx % 23) as f32 + 1.0);
                    z_biases[idx] = -0.03125 + ((idx % 13) as f32) * 0.00390625;
                }
                for k in 0..k_dim {
                    let q = quant_value((row + qkv_rows) * 4099 + k, mask);
                    pack_value(&mut z_w, row, w_cols, k, bits_usize, q);
                }
            }
            let mut sx_data = vec![0.0f32; k_dim];
            for (i, v) in sx_data.iter_mut().enumerate() {
                *v = ((i * 29 % 113) as f32 - 56.0) / 113.0;
            }

            let sx =
                MxArray::from_float32(&sx_data, &[1, k_dim as i64])?.astype(DType::BFloat16)?;
            let qkv_w_arr = MxArray::from_uint32(&qkv_w, &[qkv_rows as i64, w_cols as i64])?;
            let z_w_arr = MxArray::from_uint32(&z_w, &[z_rows as i64, w_cols as i64])?;
            let qkv_s_arr =
                MxArray::from_float32(&qkv_scales, &[qkv_rows as i64, num_groups as i64])?
                    .astype(DType::BFloat16)?;
            let z_s_arr = MxArray::from_float32(&z_scales, &[z_rows as i64, num_groups as i64])?
                .astype(DType::BFloat16)?;
            let qkv_b_arr =
                MxArray::from_float32(&qkv_biases, &[qkv_rows as i64, num_groups as i64])?
                    .astype(DType::BFloat16)?;
            let z_b_arr = MxArray::from_float32(&z_biases, &[z_rows as i64, num_groups as i64])?
                .astype(DType::BFloat16)?;

            let merged_w = MxArray::concatenate(&qkv_w_arr, &z_w_arr, 0)?;
            let merged_s = MxArray::concatenate(&qkv_s_arr, &z_s_arr, 0)?;
            let merged_b = MxArray::concatenate(&qkv_b_arr, &z_b_arr, 0)?;
            let out = qmatmul(&sx, &merged_w, &merged_s, &merged_b, group_size, bits)?;
            out.eval();
            let actual = out.to_float32_vec()?;
            let mut expected = vec![0.0f32; total_rows];
            for row in 0..total_rows {
                let (scales_ref, biases_ref, local_row) = if row < qkv_rows {
                    (&qkv_scales, &qkv_biases, row)
                } else {
                    (&z_scales, &z_biases, row - qkv_rows)
                };
                let mut acc = 0.0f32;
                for k in 0..k_dim {
                    let q = quant_value(row * 4099 + k, mask) as f32;
                    let group = k / group_size_usize;
                    let sb_idx = local_row * num_groups + group;
                    let deq = q * scales_ref[sb_idx] + biases_ref[sb_idx];
                    acc += sx_data[k] * deq;
                }
                expected[row] = acc;
            }
            max_abs_diff(&actual, &expected)
        } else {
            0.0
        };

        Ok(vec![
            q_err,
            q_gemv_err,
            gather_err,
            large_dequant_err,
            batched_err,
            large_vocab_gemv_err,
            large_vocab_argmax_ok,
            exact_sorted_gather_err,
            split_projection_concat_err,
        ])
    }

    /// WebGPU diagnostic: softmax over a contiguous slice view with non-zero offset.
    /// Returns [max_abs_error, output...].
    #[cfg(target_family = "wasm")]
    #[napi]
    pub fn test_webgpu_softmax_offset_view() -> Result<Vec<f64>> {
        let input = MxArray::from_float32(
            &[
                -5.0, -4.0, -3.0, -2.0, //
                1.0, 2.0, 3.0, 4.0, //
                7.0, 8.0, 9.0, 10.0,
            ],
            &[3, 4],
        )?;
        let view = input.slice(&[1, 0], &[2, 4])?;
        let handle = unsafe { sys::mlx_array_softmax_precise(view.handle.0, -1) };
        let out = MxArray::from_handle(handle, "test_webgpu_softmax_offset_view")?;
        out.eval();
        let actual = out.to_float32_vec()?;

        let row = [1.0f32, 2.0, 3.0, 4.0];
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = row.iter().map(|v| (*v - max).exp()).collect();
        let denom: f32 = exp.iter().sum();
        let expected: Vec<f32> = exp.iter().map(|v| *v / denom).collect();
        let err = actual
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (*a - *b).abs())
            .fold(0.0f32, f32::max);

        let mut result = Vec::with_capacity(actual.len() + 1);
        result.push(err as f64);
        result.extend(actual.iter().map(|v| *v as f64));
        Ok(result)
    }

    /// Run single layer forward using model weights, return first 5 values
    #[napi]
    pub fn test_single_layer_forward() -> Vec<f64> {
        let mut buf = [0f32; 5];
        let ok = unsafe { sys::mlx_test_single_layer_forward(buf.as_mut_ptr(), 5) };
        if !ok {
            return vec![-999.0];
        }
        buf.iter().map(|v| *v as f64).collect()
    }

    /// Read first N float values from a C++ weight map entry
    #[napi]
    pub fn read_cpp_weight(name: String, count: i32) -> Vec<f64> {
        if count <= 0 {
            return vec![-999.0];
        }
        let c_name = match std::ffi::CString::new(name) {
            Ok(s) => s,
            Err(_) => return vec![-999.0],
        };
        let mut buf = vec![0f32; count as usize];
        let n = unsafe { sys::mlx_qwen35_read_weight(c_name.as_ptr(), buf.as_mut_ptr(), count) };
        if n <= 0 {
            return vec![-999.0];
        }
        buf[..n as usize].iter().map(|v| *v as f64).collect()
    }

    /// Step-by-step GDN test: returns intermediate values at the given checkpoint.
    /// checkpoint 0-8: various stages of GDN pipeline. 9: conv1d weight info.
    #[napi]
    pub fn test_gdn_step(checkpoint: i32, max_count: i32) -> Vec<f64> {
        let mut buf = vec![0f32; max_count as usize];
        let n = unsafe { sys::mlx_test_gdn_step_by_step(checkpoint, buf.as_mut_ptr(), max_count) };
        if n <= 0 {
            return vec![-999.0];
        }
        buf[..n as usize].iter().map(|v| *v as f64).collect()
    }

    /// Isolated GDN recurrence test with small known inputs (2x2 dims).
    /// Returns y_out values followed by state_out values.
    #[napi]
    pub fn test_gdn_recurrence_small() -> Vec<f64> {
        let mut buf = vec![0f32; 20];
        let n = unsafe { sys::mlx_test_gdn_recurrence_small(buf.as_mut_ptr(), 20) };
        if n <= 0 {
            return vec![-999.0];
        }
        buf[..n as usize].iter().map(|v| *v as f64).collect()
    }

    /// Run full-attention layer 0 forward using model weights, return first N values.
    /// Exercises: RMSNorm + Q/K/V proj + QK norm + RoPE + SDPA + gate + output proj.
    #[napi]
    pub fn test_sdpa_causal(max_count: Option<i32>) -> Vec<f64> {
        let count = max_count.unwrap_or(20) as usize;
        let mut buf = vec![0f32; count];
        let n = unsafe { sys::mlx_test_sdpa_causal(buf.as_mut_ptr(), count as i32) };
        if n <= 0 {
            return vec![-999.0];
        }
        buf[..n as usize].iter().map(|v| *v as f64).collect()
    }

    #[napi]
    pub fn test_sdpa_gqa(max_count: Option<i32>) -> Vec<f64> {
        let count = max_count.unwrap_or(20) as usize;
        let mut buf = vec![0f32; count];
        let n = unsafe { sys::mlx_test_sdpa_gqa(buf.as_mut_ptr(), count as i32) };
        if n <= 0 {
            return vec![-999.0];
        }
        buf[..n as usize].iter().map(|v| *v as f64).collect()
    }

    #[napi]
    pub fn test_attention_layer_forward(max_count: Option<i32>) -> Vec<f64> {
        let count = max_count.unwrap_or(10) as usize;
        let mut buf = vec![0f32; count];
        let n = unsafe { sys::mlx_test_attention_layer_forward(buf.as_mut_ptr(), count as i32) };
        if n <= 0 {
            return vec![-999.0];
        }
        buf[..n as usize].iter().map(|v| *v as f64).collect()
    }

    // ===== Phase 1+2 inference step tests =====

    #[napi]
    pub fn test_rope_bf16(max_count: Option<i32>) -> Vec<f64> {
        let c = max_count.unwrap_or(20) as usize;
        let mut buf = vec![0f32; c];
        let n = unsafe { sys::mlx_test_rope_bf16(buf.as_mut_ptr(), c as i32) };
        if n <= 0 {
            return vec![-999.0];
        }
        buf[..n as usize].iter().map(|v| *v as f64).collect()
    }

    #[napi]
    pub fn test_qk_norm_rope(max_count: Option<i32>) -> Vec<f64> {
        let c = max_count.unwrap_or(20) as usize;
        let mut buf = vec![0f32; c];
        let n = unsafe { sys::mlx_test_qk_norm_rope(buf.as_mut_ptr(), c as i32) };
        if n <= 0 {
            return vec![-999.0];
        }
        buf[..n as usize].iter().map(|v| *v as f64).collect()
    }

    #[napi]
    pub fn test_sdpa_additive_mask(max_count: Option<i32>) -> Vec<f64> {
        let c = max_count.unwrap_or(20) as usize;
        let mut buf = vec![0f32; c];
        let n = unsafe { sys::mlx_test_sdpa_additive_mask(buf.as_mut_ptr(), c as i32) };
        if n <= 0 {
            return vec![-999.0];
        }
        buf[..n as usize].iter().map(|v| *v as f64).collect()
    }

    #[napi]
    pub fn test_sdpa_decode_gqa(max_count: Option<i32>) -> Vec<f64> {
        let c = max_count.unwrap_or(20) as usize;
        let mut buf = vec![0f32; c];
        let n = unsafe { sys::mlx_test_sdpa_decode_gqa(buf.as_mut_ptr(), c as i32) };
        if n <= 0 {
            return vec![-999.0];
        }
        buf[..n as usize].iter().map(|v| *v as f64).collect()
    }

    // ===== Tile (prefill) SDPA parity tests =====

    /// Tq=2, D=64, f32. Minimal tile shape — exercises zero-padding of
    /// non-live Q rows inside a single 16-row Q tile.
    #[napi]
    pub fn test_sdpa_tile_tq2_d64(max_count: Option<i32>) -> Vec<f64> {
        let c = max_count.unwrap_or(32) as usize;
        let mut buf = vec![0f32; c];
        let n = unsafe { sys::mlx_test_sdpa_tile_tq2_d64(buf.as_mut_ptr(), c as i32) };
        if n <= 0 {
            return vec![-999.0];
        }
        buf[..n as usize].iter().map(|v| *v as f64).collect()
    }

    /// Tq=8, D=128, Hq=8, Hkv=2, causal. Single tile + GQA 4:1 + causal.
    #[napi]
    pub fn test_sdpa_tile_tq8_d128_causal_gqa(max_count: Option<i32>) -> Vec<f64> {
        let c = max_count.unwrap_or(32) as usize;
        let mut buf = vec![0f32; c];
        let n = unsafe { sys::mlx_test_sdpa_tile_tq8_d128_causal_gqa(buf.as_mut_ptr(), c as i32) };
        if n <= 0 {
            return vec![-999.0];
        }
        buf[..n as usize].iter().map(|v| *v as f64).collect()
    }

    /// Tq=32, D=128, additive float mask. Two Q tiles + row-contiguous
    /// [1,1,Tq,L] additive mask. Exercises the mask binding path.
    #[napi]
    pub fn test_sdpa_tile_tq32_d128_addmask(max_count: Option<i32>) -> Vec<f64> {
        let c = max_count.unwrap_or(32) as usize;
        let mut buf = vec![0f32; c];
        let n = unsafe { sys::mlx_test_sdpa_tile_tq32_d128_addmask(buf.as_mut_ptr(), c as i32) };
        if n <= 0 {
            return vec![-999.0];
        }
        buf[..n as usize].iter().map(|v| *v as f64).collect()
    }

    /// Tq=33, D=128, causal. Partial last tile (1 live row out of 16) —
    /// exercises the `q_row < Tq` store guard.
    #[napi]
    pub fn test_sdpa_tile_tq33_d128_tailtile(max_count: Option<i32>) -> Vec<f64> {
        let c = max_count.unwrap_or(32) as usize;
        let mut buf = vec![0f32; c];
        let n = unsafe { sys::mlx_test_sdpa_tile_tq33_d128_tailtile(buf.as_mut_ptr(), c as i32) };
        if n <= 0 {
            return vec![-999.0];
        }
        buf[..n as usize].iter().map(|v| *v as f64).collect()
    }

    /// Tq=128, D=128, L=4096 causal. Large prefill-like workload — 8 Q tiles,
    /// 512 KV blocks. Main bandwidth stress + online softmax running-state
    /// consistency test.
    #[napi]
    pub fn test_sdpa_tile_tq128_d128_l4096(max_count: Option<i32>) -> Vec<f64> {
        let c = max_count.unwrap_or(32) as usize;
        let mut buf = vec![0f32; c];
        let n = unsafe { sys::mlx_test_sdpa_tile_tq128_d128_l4096(buf.as_mut_ptr(), c as i32) };
        if n <= 0 {
            return vec![-999.0];
        }
        buf[..n as usize].iter().map(|v| *v as f64).collect()
    }

    /// Tq=8, D=256, Hq=8, Hkv=2 causal GQA (Qwen3.5-0.8B shape). Exercises
    /// the D=256 tile path with DPT=32 and GQA 4:1 head reindexing.
    #[napi]
    pub fn test_sdpa_tile_d256_gqa(max_count: Option<i32>) -> Vec<f64> {
        let c = max_count.unwrap_or(32) as usize;
        let mut buf = vec![0f32; c];
        let n = unsafe { sys::mlx_test_sdpa_tile_d256_gqa(buf.as_mut_ptr(), c as i32) };
        if n <= 0 {
            return vec![-999.0];
        }
        buf[..n as usize].iter().map(|v| *v as f64).collect()
    }

    /// Diagnostic: simplest D=256 tile case. No causal, no GQA, minimal dims.
    #[napi]
    pub fn test_sdpa_tile_d256_simple(max_count: Option<i32>) -> Vec<f64> {
        let c = max_count.unwrap_or(32) as usize;
        let mut buf = vec![0f32; c];
        let n = unsafe { sys::mlx_test_sdpa_tile_d256_simple(buf.as_mut_ptr(), c as i32) };
        if n <= 0 {
            return vec![-999.0];
        }
        buf[..n as usize].iter().map(|v| *v as f64).collect()
    }

    /// Diagnostic: D=256 with causal, no GQA.
    #[napi]
    pub fn test_sdpa_tile_d256_causal(max_count: Option<i32>) -> Vec<f64> {
        let c = max_count.unwrap_or(32) as usize;
        let mut buf = vec![0f32; c];
        let n = unsafe { sys::mlx_test_sdpa_tile_d256_causal(buf.as_mut_ptr(), c as i32) };
        if n <= 0 {
            return vec![-999.0];
        }
        buf[..n as usize].iter().map(|v| *v as f64).collect()
    }

    /// Diagnostic: D=256 with GQA, no causal.
    #[napi]
    pub fn test_sdpa_tile_d256_gqa_nocausal(max_count: Option<i32>) -> Vec<f64> {
        let c = max_count.unwrap_or(32) as usize;
        let mut buf = vec![0f32; c];
        let n = unsafe { sys::mlx_test_sdpa_tile_d256_gqa_nocausal(buf.as_mut_ptr(), c as i32) };
        if n <= 0 {
            return vec![-999.0];
        }
        buf[..n as usize].iter().map(|v| *v as f64).collect()
    }

    /// Minimal causal+GQA at D=256: H=2 Hkv=1 gqa=2 Tq=2 L=4.
    #[napi]
    pub fn test_sdpa_tile_d256_causal_gqa_minimal(max_count: Option<i32>) -> Vec<f64> {
        let c = max_count.unwrap_or(32) as usize;
        let mut buf = vec![0f32; c];
        let n =
            unsafe { sys::mlx_test_sdpa_tile_d256_causal_gqa_minimal(buf.as_mut_ptr(), c as i32) };
        if n <= 0 {
            return vec![-999.0];
        }
        buf[..n as usize].iter().map(|v| *v as f64).collect()
    }

    /// D=256 vector kernel with GQA: Qwen3.5-0.8B decode shape (Tq=1, Hq=8, Hkv=2, L=32).
    #[napi]
    pub fn test_sdpa_vector_d256(max_count: Option<i32>) -> Vec<f64> {
        let c = max_count.unwrap_or(32) as usize;
        let mut buf = vec![0f32; c];
        let n = unsafe { sys::mlx_test_sdpa_vector_d256(buf.as_mut_ptr(), c as i32) };
        if n <= 0 {
            return vec![-999.0];
        }
        buf[..n as usize].iter().map(|v| *v as f64).collect()
    }

    /// Simplest D=256 vector case: no GQA, Tq=1, H=Hkv=2, L=16.
    #[napi]
    pub fn test_sdpa_vector_d256_simple(max_count: Option<i32>) -> Vec<f64> {
        let c = max_count.unwrap_or(32) as usize;
        let mut buf = vec![0f32; c];
        let n = unsafe { sys::mlx_test_sdpa_vector_d256_simple(buf.as_mut_ptr(), c as i32) };
        if n <= 0 {
            return vec![-999.0];
        }
        buf[..n as usize].iter().map(|v| *v as f64).collect()
    }

    #[napi]
    pub fn test_full_attn_layer_bf16(max_count: Option<i32>) -> Vec<f64> {
        let c = max_count.unwrap_or(20) as usize;
        let mut buf = vec![0f32; c];
        let n = unsafe { sys::mlx_test_full_attn_layer_bf16(buf.as_mut_ptr(), c as i32) };
        if n <= 0 {
            return vec![-999.0];
        }
        buf[..n as usize].iter().map(|v| *v as f64).collect()
    }

    #[napi]
    pub fn test_rms_norm_bf16(max_count: Option<i32>) -> Vec<f64> {
        let c = max_count.unwrap_or(20) as usize;
        let mut buf = vec![0f32; c];
        let n = unsafe { sys::mlx_test_rms_norm_bf16(buf.as_mut_ptr(), c as i32) };
        if n <= 0 {
            return vec![-999.0];
        }
        buf[..n as usize].iter().map(|v| *v as f64).collect()
    }

    /// Call `mlx::fast::rms_norm(self, weight, eps)` directly. This is a
    /// thin NAPI wrapper around the FFI signature so browser tests can
    /// exercise the fused RMSNorm GPU kernel against arbitrary weight
    /// tensors (the C++ test helpers only expose fixed-dim bench cases).
    /// Intended for kernel parity testing — production code uses the
    /// `RMSNorm` layer in `nn/normalization.rs`.
    #[napi]
    pub fn fast_rms_norm(&self, weight: &MxArray, eps: f64) -> Result<Self> {
        let handle =
            unsafe { sys::mlx_fast_rms_norm(self.as_raw_ptr(), weight.as_raw_ptr(), eps as f32) };
        MxArray::from_handle(handle, "fast_rms_norm")
    }

    #[napi]
    pub fn test_swiglu_mlp_bf16(max_count: Option<i32>) -> Vec<f64> {
        let c = max_count.unwrap_or(20) as usize;
        let mut buf = vec![0f32; c];
        let n = unsafe { sys::mlx_test_swiglu_mlp_bf16(buf.as_mut_ptr(), c as i32) };
        if n <= 0 {
            return vec![-999.0];
        }
        buf[..n as usize].iter().map(|v| *v as f64).collect()
    }

    #[napi]
    pub fn test_decode_step_with_cache(max_count: Option<i32>) -> Vec<f64> {
        let c = max_count.unwrap_or(20) as usize;
        let mut buf = vec![0f32; c];
        let n = unsafe { sys::mlx_test_decode_step_with_cache(buf.as_mut_ptr(), c as i32) };
        if n <= 0 {
            return vec![-999.0];
        }
        buf[..n as usize].iter().map(|v| *v as f64).collect()
    }

    #[napi]
    pub fn test_attn_layer_bf16(max_count: Option<i32>) -> Vec<f64> {
        let c = max_count.unwrap_or(20) as usize;
        let mut buf = vec![0f32; c];
        let n = unsafe { sys::mlx_test_attn_layer_bf16(buf.as_mut_ptr(), c as i32) };
        if n <= 0 {
            return vec![-999.0];
        }
        buf[..n as usize].iter().map(|v| *v as f64).collect()
    }

    #[napi]
    pub fn test_first_4_layers_bf16(max_count: Option<i32>) -> Vec<f64> {
        let c = max_count.unwrap_or(20) as usize;
        let mut buf = vec![0f32; c];
        let n = unsafe { sys::mlx_test_first_4_layers_bf16(buf.as_mut_ptr(), c as i32) };
        if n <= 0 {
            return vec![-999.0];
        }
        buf[..n as usize].iter().map(|v| *v as f64).collect()
    }

    #[napi]
    pub fn test_gdn_full_bf16(max_count: Option<i32>) -> Vec<f64> {
        let c = max_count.unwrap_or(20) as usize;
        let mut buf = vec![0f32; c];
        let n = unsafe { sys::mlx_test_gdn_full_bf16(buf.as_mut_ptr(), c as i32) };
        if n <= 0 {
            return vec![-999.0];
        }
        buf[..n as usize].iter().map(|v| *v as f64).collect()
    }

    #[napi]
    pub fn test_gdn_multi_step_bf16(max_count: Option<i32>) -> Vec<f64> {
        let c = max_count.unwrap_or(20) as usize;
        let mut buf = vec![0f32; c];
        let n = unsafe { sys::mlx_test_gdn_multi_step_bf16(buf.as_mut_ptr(), c as i32) };
        if n <= 0 {
            return vec![-999.0];
        }
        buf[..n as usize].iter().map(|v| *v as f64).collect()
    }

    #[napi]
    pub fn test_categorical_sampling_bf16(max_count: Option<i32>) -> Vec<f64> {
        let c = max_count.unwrap_or(10) as usize;
        let mut buf = vec![0f32; c];
        let n = unsafe { sys::mlx_test_categorical_sampling_bf16(buf.as_mut_ptr(), c as i32) };
        if n <= 0 {
            return vec![-999.0];
        }
        buf[..n as usize].iter().map(|v| *v as f64).collect()
    }

    #[napi]
    pub fn test_gdn_layer_bf16(max_count: Option<i32>) -> Vec<f64> {
        let c = max_count.unwrap_or(20) as usize;
        let mut buf = vec![0f32; c];
        let n = unsafe { sys::mlx_test_gdn_layer_bf16(buf.as_mut_ptr(), c as i32) };
        if n <= 0 {
            return vec![-999.0];
        }
        buf[..n as usize].iter().map(|v| *v as f64).collect()
    }

    /// Multi-dim batch matmul with broadcasting (GQA pattern).
    /// Returns [gpu_output..., cpu_reference...] to verify multi-dim batch
    /// stride decomposition in the WebGPU matmul kernel.
    #[napi]
    pub fn test_matmul_broadcast_batch(max_count: Option<i32>) -> Vec<f64> {
        let c = max_count.unwrap_or(128) as usize;
        let mut buf = vec![0f32; c];
        let n = unsafe { sys::mlx_test_matmul_broadcast_batch(buf.as_mut_ptr(), c as i32) };
        if n <= 0 {
            return vec![-999.0];
        }
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
