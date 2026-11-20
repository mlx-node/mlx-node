//! Backend-neutral small-row dense & quantized linear kernel dispatch.

use std::ffi::{CStr, CString};

use mlx_sys as sys;
use napi::bindgen_prelude::*;

use crate::accelerator::{ActiveBackend, active_backend};
use crate::array::{DType, MxArray};

/// Pre-interned static C-strings for every CUDA K-quant mode so hot-path
/// `QuantizedLinear::forward` performs zero per-op `CString` heap allocations.
#[inline]
pub(crate) fn intern_kquant_mode(mode: &str) -> Option<&'static CStr> {
    match mode {
        "q3k" => Some(c"q3k"),
        "q4k" => Some(c"q4k"),
        "q5k" => Some(c"q5k"),
        "q6k" => Some(c"q6k"),
        "iq4nl" => Some(c"iq4nl"),
        "iq4xs" => Some(c"iq4xs"),
        "q3k@t64" => Some(c"q3k@t64"),
        "q4k@t64" => Some(c"q4k@t64"),
        "q5k@t64" => Some(c"q5k@t64"),
        "q6k@t64" => Some(c"q6k@t64"),
        "iq4nl@t64" => Some(c"iq4nl@t64"),
        "iq4xs@t64" => Some(c"iq4xs@t64"),
        _ => None,
    }
}

/// Small-row (`M in 1..=8`) BF16 GEMM (`X @ W^T`) fast path.
#[inline]
pub(crate) fn bf16_matmul_few(
    input: &MxArray,
    weight: &MxArray,
    out_features: i64,
) -> Result<Option<MxArray>> {
    match active_backend() {
        ActiveBackend::CudaGpu => {
            if input.dtype()? != DType::BFloat16 || weight.dtype()? != DType::BFloat16 {
                return Ok(None);
            }
            let ndim = input.ndim()?;
            if ndim < 2 {
                return Ok(None);
            }
            let k = input.shape_at(ndim - 1)?;
            let rows = if k > 0 { input.size()? as i64 / k } else { 0 };
            if !(1..=8).contains(&rows) || k < 1024 || k % 8 != 0 || out_features < 1024 {
                return Ok(None);
            }
            let handle =
                unsafe { sys::mlx_bf16_gemm_few_cuda(input.as_raw_ptr(), weight.as_raw_ptr()) };
            if handle.is_null() {
                Ok(None)
            } else {
                Ok(Some(MxArray::from_handle(handle, "bf16_gemm_few_cuda")?))
            }
        }
        ActiveBackend::MetalGpu | ActiveBackend::Cpu => Ok(None),
    }
}

/// Small-row (`M in 2..=8`) BF16-in / BF16-out affine quantized matmul with
/// F32 sidecars (Metal `mlx_quantized_matmul_affine_bf16` or CUDA `Q8_0` fast path).
#[inline]
pub(crate) fn affine_bf16_matmul_few(
    x: &MxArray,
    weight: &MxArray,
    scales: &MxArray,
    biases: &MxArray,
    group_size: i32,
    bits: i32,
) -> Result<Option<MxArray>> {
    match active_backend() {
        ActiveBackend::MetalGpu => {
            let handle = unsafe {
                sys::mlx_quantized_matmul_affine_bf16(
                    x.handle.0,
                    weight.handle.0,
                    scales.handle.0,
                    biases.handle.0,
                    group_size,
                    bits,
                )
            };
            if handle.is_null() {
                Ok(None)
            } else {
                Ok(Some(MxArray::from_handle(
                    handle,
                    "quantized_matmul_affine_bf16",
                )?))
            }
        }
        ActiveBackend::CudaGpu if bits == 8 && group_size == 32 => {
            let handle = unsafe {
                sys::mlx_affine_q8_few_cuda(
                    x.as_raw_ptr(),
                    weight.as_raw_ptr(),
                    scales.as_raw_ptr(),
                    biases.as_raw_ptr(),
                )
            };
            if handle.is_null() {
                Ok(None)
            } else {
                Ok(Some(MxArray::from_handle(handle, "affine_q8_few_cuda")?))
            }
        }
        ActiveBackend::CudaGpu | ActiveBackend::Cpu => Ok(None),
    }
}

/// Small-row (`M in 2..=8`) packed K-quant INT8-MMA matmul on CUDA.
#[inline]
pub(crate) fn kquant_matmul_few(
    x: &MxArray,
    weight: &MxArray,
    scales: &MxArray,
    biases: &MxArray,
    mode: &str,
) -> Result<Option<MxArray>> {
    if active_backend() != ActiveBackend::CudaGpu {
        return Ok(None);
    }
    let handle = if let Some(mode_c) = intern_kquant_mode(mode) {
        unsafe {
            sys::mlx_kquant_few(
                x.as_raw_ptr(),
                weight.as_raw_ptr(),
                scales.as_raw_ptr(),
                biases.as_raw_ptr(),
                mode_c.as_ptr(),
            )
        }
    } else {
        let mode_c = CString::new(mode)
            .map_err(|e| Error::from_reason(format!("Invalid mode string: {e}")))?;
        unsafe {
            sys::mlx_kquant_few(
                x.as_raw_ptr(),
                weight.as_raw_ptr(),
                scales.as_raw_ptr(),
                biases.as_raw_ptr(),
                mode_c.as_ptr(),
            )
        }
    };
    if handle.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(handle, "kquant_few")?))
    }
}
