//! Metal quantized linear kernel wrapper.

use mlx_sys as sys;
use napi::bindgen_prelude::*;

use crate::array::MxArray;

#[inline]
pub(super) fn affine_bf16_matmul_few(
    x: &MxArray,
    weight: &MxArray,
    scales: &MxArray,
    biases: &MxArray,
    group_size: i32,
    bits: i32,
) -> Result<Option<MxArray>> {
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
