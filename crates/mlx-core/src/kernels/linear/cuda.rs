//! CUDA dense & quantized linear kernel wrappers.

use std::ffi::CString;

use mlx_sys as sys;
use napi::bindgen_prelude::*;

use super::intern_kquant_mode;
use crate::array::{DType, MxArray};

#[inline]
pub(super) fn bf16_matmul_few(
    input: &MxArray,
    weight: &MxArray,
    out_features: i64,
) -> Result<Option<MxArray>> {
    if input.dtype()? != DType::BFloat16 || weight.dtype()? != DType::BFloat16 {
        return Ok(None);
    }
    let ndim = input.ndim()?;
    if ndim < 2 {
        return Ok(None);
    }
    let k = input.shape_at(ndim - 1)?;
    let rows = if k > 0 { input.size()? as i64 / k } else { 0 };
    if !(1..=16).contains(&rows) || k < 1024 || k % 8 != 0 || out_features < 1 {
        return Ok(None);
    }
    let handle = unsafe { sys::mlx_bf16_gemm_few_cuda(input.as_raw_ptr(), weight.as_raw_ptr()) };
    if handle.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(handle, "bf16_gemm_few_cuda")?))
    }
}

#[inline]
pub(super) fn affine_q8_matmul_few(
    x: &MxArray,
    weight: &MxArray,
    scales: &MxArray,
    biases: &MxArray,
) -> Result<Option<MxArray>> {
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

#[inline]
pub(super) fn kquant_matmul_few(
    x: &MxArray,
    weight: &MxArray,
    scales: &MxArray,
    biases: &MxArray,
    mode: &str,
) -> Result<Option<MxArray>> {
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

#[inline]
pub(super) fn kquant_matmul_prefill(
    x: &MxArray,
    weight: &MxArray,
    scales: &MxArray,
    biases: &MxArray,
    mode: &str,
) -> Result<Option<MxArray>> {
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

#[inline]
pub(super) fn kquant_matmul_multi(
    x: &MxArray,
    rows: i64,
    projections: &[(&MxArray, &MxArray, &MxArray, &str)],
) -> Result<Option<Vec<MxArray>>> {
    if !(2..=4).contains(&projections.len()) {
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

#[inline]
pub(super) fn affine_q4_g64_matmul_few(
    x: &MxArray,
    weight: &MxArray,
    scales: &MxArray,
    biases: &MxArray,
) -> Result<Option<MxArray>> {
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
