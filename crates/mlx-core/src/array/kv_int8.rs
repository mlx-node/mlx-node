//! Per-row symmetric int8 KV rows (Splash's target KV format): one int8 row
//! of `head_dim = 256` per (batch, head, token) with one fp32 scale,
//! `scale = max|x| / 127`, `q = clamp(rint(x * 127 / max), -127, 127)`.
//!
//! The flat Qwen3.5-family `KVCache` stores its full-attention rows this way
//! when loaded with `kvFormat: 'int8'`; the segmented SDPA kernels read the
//! int8 rows and their scales directly, and prefill chunks after the first
//! dequantize the cached prefix back to BF16 for MLX's fused causal SDPA.

use super::MxArray;
use mlx_sys as sys;
use napi::bindgen_prelude::*;

/// Quantized rows: `keys` / `values` int8 `[B, H, N, 256]`, `key_scales` /
/// `value_scales` float32 `[B, H, N]`.
#[derive(Clone)]
pub struct Int8KvRows {
    pub keys: MxArray,
    pub values: MxArray,
    pub key_scales: MxArray,
    pub value_scales: MxArray,
}

fn quantize_with(
    x: &MxArray,
    entry: unsafe extern "C-unwind" fn(
        *mut sys::mlx_array,
        *mut *mut sys::mlx_array,
        *mut *mut sys::mlx_array,
    ) -> i32,
    context: &str,
) -> Result<(MxArray, MxArray)> {
    let mut q: *mut sys::mlx_array = std::ptr::null_mut();
    let mut s: *mut sys::mlx_array = std::ptr::null_mut();
    // SAFETY: `x` is a live array handle; the out pointers receive owned
    // handles on success and stay null on failure.
    let status = unsafe { entry(x.as_raw_ptr(), &mut q, &mut s) };
    if status != 0 {
        return Err(Error::from_reason(format!(
            "{context}: int8 row quantization failed (details on stderr)"
        )));
    }
    Ok((
        MxArray::from_handle(q, context)?,
        MxArray::from_handle(s, context)?,
    ))
}

/// BF16 rows `[B, H, N, 256]` -> `(int8 rows, fp32 scales [B, H, N])`.
pub fn quantize_kv_rows(x: &MxArray) -> Result<(MxArray, MxArray)> {
    quantize_with(x, sys::mlx_kv_int8_quantize_rows, "quantize_kv_rows")
}

/// The MLX-op reference of [`quantize_kv_rows`] (tests).
#[cfg(test)]
pub(crate) fn quantize_kv_rows_reference(x: &MxArray) -> Result<(MxArray, MxArray)> {
    quantize_with(
        x,
        sys::mlx_kv_int8_quantize_rows_reference,
        "quantize_kv_rows_reference",
    )
}

/// int8 rows `[B, H, N, 256]` x fp32 scales `[B, H, N]` -> BF16 rows.
pub fn dequantize_kv_rows(q: &MxArray, s: &MxArray) -> Result<MxArray> {
    // SAFETY: both handles are live arrays; the result is an owned handle or
    // null on failure.
    let handle = unsafe { sys::mlx_kv_int8_dequantize_rows(q.as_raw_ptr(), s.as_raw_ptr()) };
    MxArray::from_handle(handle, "dequantize_kv_rows")
}

impl Int8KvRows {
    /// Quantize a BF16 `(keys, values)` pair.
    pub fn quantize(keys: &MxArray, values: &MxArray) -> Result<Self> {
        let (k, ks) = quantize_kv_rows(keys)?;
        let (v, vs) = quantize_kv_rows(values)?;
        Ok(Self {
            keys: k,
            values: v,
            key_scales: ks,
            value_scales: vs,
        })
    }

    /// BF16 `(keys, values)` of these rows.
    pub fn dequantize(&self) -> Result<(MxArray, MxArray)> {
        Ok((
            dequantize_kv_rows(&self.keys, &self.key_scales)?,
            dequantize_kv_rows(&self.values, &self.value_scales)?,
        ))
    }

    /// Rows `[start, end)` along the token axis of every array (views).
    pub fn slice_tokens(&self, start: i64, end: i64) -> Result<Self> {
        Ok(Self {
            keys: self.keys.slice_axis(2, start, end)?,
            values: self.values.slice_axis(2, start, end)?,
            key_scales: self.key_scales.slice_axis(2, start, end)?,
            value_scales: self.value_scales.slice_axis(2, start, end)?,
        })
    }

    /// Token rows held.
    pub fn len(&self) -> Result<i64> {
        self.keys.shape_at(2)
    }

    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    /// The four arrays in the order the segmented int8 primitive and the
    /// compiled verify tape use: keys, values, key scales, value scales.
    pub fn arrays(&self) -> [&MxArray; 4] {
        [
            &self.keys,
            &self.values,
            &self.key_scales,
            &self.value_scales,
        ]
    }
}

/// Segmented BF16-query attention over an int8 prefix followed by int8 new
/// rows (the verify block or the single decode row). Never falls back to the
/// dequantizing concat: an unsupported launch is an error.
pub fn segmented_sdpa_int8(
    queries: &MxArray,
    prefix: &Int8KvRows,
    new: &Int8KvRows,
    scale: f32,
    causal: bool,
) -> Result<MxArray> {
    // SAFETY: every handle is a live array; the result is an owned handle or
    // null (message on stderr).
    let handle = unsafe {
        sys::mlx_segmented_sdpa_int8_forward(
            queries.as_raw_ptr(),
            prefix.keys.as_raw_ptr(),
            prefix.values.as_raw_ptr(),
            prefix.key_scales.as_raw_ptr(),
            prefix.value_scales.as_raw_ptr(),
            new.keys.as_raw_ptr(),
            new.values.as_raw_ptr(),
            new.key_scales.as_raw_ptr(),
            new.value_scales.as_raw_ptr(),
            scale,
            causal,
        )
    };
    MxArray::from_handle(handle, "segmented_sdpa_int8")
}

/// Forced-route form of [`segmented_sdpa_int8`] (tests): `mode` -1
/// automatic, 0 vector, 1 tile, 2 nax.
#[cfg(test)]
pub(crate) fn segmented_sdpa_int8_with_route(
    queries: &MxArray,
    prefix: &Int8KvRows,
    new: &Int8KvRows,
    scale: f32,
    causal: bool,
    mode: i32,
) -> Result<MxArray> {
    // SAFETY: as `segmented_sdpa_int8`.
    let handle = unsafe {
        sys::mlx_segmented_sdpa_int8_test_forward(
            queries.as_raw_ptr(),
            prefix.keys.as_raw_ptr(),
            prefix.values.as_raw_ptr(),
            prefix.key_scales.as_raw_ptr(),
            prefix.value_scales.as_raw_ptr(),
            new.keys.as_raw_ptr(),
            new.values.as_raw_ptr(),
            new.key_scales.as_raw_ptr(),
            new.value_scales.as_raw_ptr(),
            scale,
            causal,
            mode,
        )
    };
    MxArray::from_handle(handle, "segmented_sdpa_int8_with_route")
}
