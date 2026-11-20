//! Backend-neutral Tensor Core / Metal attention dispatch facades.

use mlx_sys as sys;
use napi::bindgen_prelude::*;

use crate::accelerator::{ActiveBackend, active_backend};
use crate::array::{DType, MxArray};

/// Small-query (`Q in 2..=8`) causal Flash-Decoding SDPA fast path on CUDA
/// (`D=256`, `GQA > 1`, `gqa * Q <= 48`, BF16).
#[inline]
pub(crate) fn sdpa_causal_few(
    queries: &MxArray,
    keys: &MxArray,
    values: &MxArray,
    head_dim: i64,
    gqa: i64,
    seq_len: i64,
    scale: f32,
) -> Result<Option<MxArray>> {
    match active_backend() {
        ActiveBackend::CudaGpu => {
            if !(2..=8).contains(&seq_len)
                || head_dim != 256
                || gqa <= 1
                || gqa * seq_len > 48
                || queries.dtype()? != DType::BFloat16
                || keys.dtype()? != DType::BFloat16
                || values.dtype()? != DType::BFloat16
            {
                return Ok(None);
            }
            let handle = unsafe {
                sys::mlx_sdpa_tc_few_cuda(
                    queries.as_raw_ptr(),
                    keys.as_raw_ptr(),
                    values.as_raw_ptr(),
                    scale,
                )
            };
            if handle.is_null() {
                Ok(None)
            } else {
                Ok(Some(MxArray::from_handle(handle, "sdpa_tc_few_cuda")?))
            }
        }
        ActiveBackend::MetalGpu | ActiveBackend::Cpu => Ok(None),
    }
}

/// DFlash2 masked/unmasked small-query attention fast path:
/// - Metal: `mlx_dflash2_attention_with_policy` (fail-closed on negative status).
/// - CUDA: `mlx_sdpa_tc_few_masked_cuda` for `D in {128, 256}`, `Q in 1..=8`, BF16.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn sdpa_draft_masked(
    queries: &MxArray,
    keys: &MxArray,
    values: &MxArray,
    mask: Option<&MxArray>,
    head_dim: i64,
    num_heads: i64,
    num_kv_heads: i64,
    seq_len: i64,
    scale: f32,
    tensor_attention_enabled: bool,
) -> Result<Option<MxArray>> {
    match active_backend() {
        ActiveBackend::MetalGpu => {
            let mut tensor_op = std::ptr::null_mut();
            let status = unsafe {
                sys::mlx_dflash2_attention_with_policy(
                    queries.as_raw_ptr(),
                    keys.as_raw_ptr(),
                    values.as_raw_ptr(),
                    mask.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr),
                    scale,
                    tensor_attention_enabled,
                    &mut tensor_op,
                )
            };
            match status {
                1 => Ok(Some(MxArray::from_handle(
                    tensor_op,
                    "dflash2 tensor-op attention",
                )?)),
                0 => Ok(None),
                _ => Err(Error::from_reason(
                    "dflash2 tensor-op attention failed after admission",
                )),
            }
        }
        ActiveBackend::CudaGpu => {
            let m = (num_heads / num_kv_heads) * seq_len;
            if !(1..=8).contains(&seq_len)
                || !(head_dim == 128 || head_dim == 256)
                || m > 64
                || (head_dim == 256 && m > 48)
                || queries.dtype()? != DType::BFloat16
                || keys.dtype()? != DType::BFloat16
                || values.dtype()? != DType::BFloat16
            {
                return Ok(None);
            }
            let mask_ptr = match mask {
                Some(m) if m.ndim()? == 4 && m.shape_at(0)? == 1 && m.shape_at(1)? == 1 => {
                    m.as_raw_ptr()
                }
                None => std::ptr::null_mut(),
                _ => return Ok(None),
            };
            let handle = unsafe {
                sys::mlx_sdpa_tc_few_masked_cuda(
                    queries.as_raw_ptr(),
                    keys.as_raw_ptr(),
                    values.as_raw_ptr(),
                    mask_ptr,
                    scale,
                    false,
                )
            };
            if handle.is_null() {
                Ok(None)
            } else {
                Ok(Some(MxArray::from_handle(
                    handle,
                    "dflash2 sdpa_tc_few_cuda",
                )?))
            }
        }
        ActiveBackend::Cpu => Ok(None),
    }
}
