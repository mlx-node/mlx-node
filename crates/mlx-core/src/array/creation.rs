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
