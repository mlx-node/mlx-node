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

/// Small-row (`M in 1..=16`) BF16 GEMM (`X @ W^T`) fast path.
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
            if !(1..=16).contains(&rows) || k < 1024 || k % 8 != 0 || out_features < 256 {
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

/// Small-row BF16-in / BF16-out affine quantized matmul (Metal `M in 2..=8`
/// `mlx_quantized_matmul_affine_bf16` with F32 sidecars, or CUDA `M in 1..=16`
/// `Q8_0` fast path with F32/BF16/F16 sidecars).
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

/// Small-row (`M in 1..=16`) packed K-quant INT8-MMA matmul on CUDA.
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

/// Large-row (`M >= 17`) in-register W4A8 INT8-MMA prefill matmul on CUDA for
/// `Tiled64` K-quants (eliminates 48+ GB BF16 RetainedDenseWeight cache).
#[inline]
pub(crate) fn kquant_matmul_prefill(
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
            sys::mlx_kquant_prefill(
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
            sys::mlx_kquant_prefill(
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
        Ok(Some(MxArray::from_handle(handle, "kquant_prefill")?))
    }
}

/// Multi-projection shared-input K-quant matmul on CUDA (quantizes `x` once,
/// runs 2..=4 projections in parallel on the same packed activation buffer).
/// Uses `KQuantFewMulti` for `M in 1..=16` and `KQuantPrefillMulti` for `M >= 17`.
#[inline]
pub(crate) fn kquant_matmul_multi(
    x: &MxArray,
    rows: i64,
    projections: &[(&MxArray, &MxArray, &MxArray, &str)],
) -> Result<Option<Vec<MxArray>>> {
    if active_backend() != ActiveBackend::CudaGpu || !(2..=4).contains(&projections.len()) {
        return Ok(None);
    }
    let count = projections.len();
    let mut weights = [std::ptr::null_mut(); 4];
    let mut scales = [std::ptr::null_mut(); 4];
    let mut biases = [std::ptr::null_mut(); 4];
    let mut mode_ptrs = [std::ptr::null(); 4];
    let mut owned_modes: Vec<CString> = Vec::new();
    for (i, &(w, sc, bi, mode)) in projections.iter().enumerate() {
        weights[i] = w.as_raw_ptr();
        scales[i] = sc.as_raw_ptr();
        biases[i] = bi.as_raw_ptr();
        if let Some(interned) = intern_kquant_mode(mode) {
            mode_ptrs[i] = interned.as_ptr();
        } else {
            let cs = CString::new(mode)
                .map_err(|e| Error::from_reason(format!("Invalid mode string: {e}")))?;
            owned_modes.push(cs);
            mode_ptrs[i] = owned_modes[owned_modes.len() - 1].as_ptr();
        }
    }
    let mut out_handles = [std::ptr::null_mut(); 4];
    let ok = if (1..=16).contains(&rows) {
        unsafe {
            sys::mlx_kquant_few_multi(
                x.as_raw_ptr(),
                count as u32,
                weights.as_ptr(),
                scales.as_ptr(),
                biases.as_ptr(),
                mode_ptrs.as_ptr(),
                out_handles.as_mut_ptr(),
            )
        }
    } else if rows >= 17 {
        unsafe {
            sys::mlx_kquant_prefill_multi(
                x.as_raw_ptr(),
                count as u32,
                weights.as_ptr(),
                scales.as_ptr(),
                biases.as_ptr(),
                mode_ptrs.as_ptr(),
                out_handles.as_mut_ptr(),
            )
        }
    } else {
        false
    };
    // Keep any dynamically allocated CStrings alive across the FFI call.
    drop(owned_modes);
    if !ok {
        return Ok(None);
    }
    let mut result = Vec::with_capacity(count);
    for handle in out_handles.iter().take(count) {
        if handle.is_null() {
            return Ok(None);
        }
        result.push(MxArray::from_handle(*handle, "kquant_multi")?);
    }
    Ok(Some(result))
}

/// High-bandwidth 8-warp split-K BF16 Tensor-Core matmul for row-major affine
/// 4-bit group_size=64 (BF16 scales/biases) on CUDA (`M in 1..=8`).
#[inline]
pub(crate) fn affine_q4_g64_matmul_few(
    x: &MxArray,
    weight: &MxArray,
    scales: &MxArray,
    biases: &MxArray,
) -> Result<Option<MxArray>> {
    if active_backend() != ActiveBackend::CudaGpu {
        return Ok(None);
    }
    let handle = unsafe {
        sys::mlx_affine_q4_g64_few_cuda(
            x.as_raw_ptr(),
            weight.as_raw_ptr(),
            scales.as_raw_ptr(),
            biases.as_raw_ptr(),
        )
    };
    if handle.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(handle, "affine_q4_g64_few")?))
    }
}
