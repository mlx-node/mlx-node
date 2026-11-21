//! Backend-neutral Tensor Core / Metal attention dispatch facades.

use mlx_sys as sys;
use napi::bindgen_prelude::*;

use crate::accelerator::{ActiveBackend, active_backend};
use crate::array::{DType, MxArray};

/// Small-query (`Q in 2..=16`) causal Flash-Decoding SDPA fast path on CUDA
/// (`D=256`, `GQA > 1`, `M = gqa * Q <= 96` or two-tile `Q > 8 && gqa * 8 <= 64`, BF16).
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
            let m = gqa * seq_len;
            let ok_single = (1..=96).contains(&m);
            let ok_two_tile = seq_len > 8 && gqa * 8 <= 64;
            if !(2..=16).contains(&seq_len)
                || head_dim != 256
                || gqa <= 1
                || (!ok_single && !ok_two_tile)
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

/// Causal FlashAttention-2 prefill fast path on CUDA (`D=256`, `Q_len >= 9`, BF16).
#[inline]
pub fn sdpa_prefill_d256(
    queries: &MxArray,
    keys: &MxArray,
    values: &MxArray,
    head_dim: i64,
    seq_len: i64,
    scale: f32,
) -> Result<Option<MxArray>> {
    match active_backend() {
        ActiveBackend::CudaGpu => {
            if seq_len < 9
                || head_dim != 256
                || queries.dtype()? != DType::BFloat16
                || keys.dtype()? != DType::BFloat16
                || values.dtype()? != DType::BFloat16
            {
                return Ok(None);
            }
            let handle = unsafe {
                sys::mlx_sdpa_prefill_d256_cuda(
                    queries.as_raw_ptr(),
                    keys.as_raw_ptr(),
                    values.as_raw_ptr(),
                    scale,
                )
            };
            if handle.is_null() {
                Ok(None)
            } else {
                Ok(Some(MxArray::from_handle(
                    handle,
                    "sdpa_prefill_d256_cuda",
                )?))
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
            let gqa = num_heads / num_kv_heads;
            let m = gqa * seq_len;
            let ok_single = (1..=96).contains(&m);
            let ok_two_tile = seq_len > 8 && gqa * 8 <= 64;
            if !(1..=16).contains(&seq_len)
                || !(head_dim == 128 || head_dim == 256)
                || (!ok_single && !ok_two_tile)
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

/// Fused Q/K RMSNorm + non-traditional partial RoPE with per-batch `offsets`
/// (`int32[B]`, or scalar for `B == 1`).
///
/// Inputs: `queries` `[B, T, HQ, D]`, `keys` `[B, T, HK, D]`, `q_weight`/`k_weight` `[D]`.
/// Outputs: `([B, HQ, T, D], [B, HK, T, D])`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn qk_norm_rope_offsets(
    queries: &MxArray,
    keys: &MxArray,
    q_weight: &MxArray,
    k_weight: &MxArray,
    offsets: &MxArray,
    eps: f32,
    rope_base: f32,
    rope_scale: f32,
    rope_dims: i32,
) -> Result<Option<(MxArray, MxArray)>> {
    let mut out_q = std::ptr::null_mut();
    let mut out_k = std::ptr::null_mut();
    let ok = match active_backend() {
        ActiveBackend::MetalGpu => unsafe {
            sys::mlx_qk_norm_rope(
                queries.as_raw_ptr(),
                keys.as_raw_ptr(),
                q_weight.as_raw_ptr(),
                k_weight.as_raw_ptr(),
                offsets.as_raw_ptr(),
                eps,
                rope_base,
                rope_scale,
                rope_dims,
                &mut out_q,
                &mut out_k,
            )
        },
        ActiveBackend::CudaGpu => unsafe {
            sys::mlx_qk_norm_rope_cuda(
                queries.as_raw_ptr(),
                keys.as_raw_ptr(),
                q_weight.as_raw_ptr(),
                k_weight.as_raw_ptr(),
                offsets.as_raw_ptr(),
                eps,
                rope_base,
                rope_scale,
                rope_dims,
                &mut out_q,
                &mut out_k,
            )
        },
        ActiveBackend::Cpu => false,
    };
    if !ok || out_q.is_null() || out_k.is_null() {
        return Ok(None);
    }
    Ok(Some((
        MxArray::from_handle(out_q, "qk_norm_rope:q")?,
        MxArray::from_handle(out_k, "qk_norm_rope:k")?,
    )))
}

/// Tree-attention small-query (`Q in 1..=16`) Flash-Decoding SDPA on CUDA
/// (`D in {128, 256}`, BF16, `tree_mask` uint32 `[Q]` bitmask over speculative
/// tree nodes past `cache_len`).
#[inline]
#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn sdpa_tree_few(
    queries: &MxArray,
    keys: &MxArray,
    values: &MxArray,
    tree_mask: &MxArray,
    cache_len: i32,
    head_dim: i64,
    gqa: i64,
    seq_len: i64,
    scale: f32,
) -> Result<Option<MxArray>> {
    if active_backend() != ActiveBackend::CudaGpu {
        return Ok(None);
    }
    let m = gqa * seq_len;
    let ok_single = (1..=96).contains(&m);
    let ok_two_tile = seq_len > 8 && gqa * 8 <= 64;
    if !(1..=16).contains(&seq_len)
        || !(head_dim == 128 || head_dim == 256)
        || cache_len < 0
        || (!ok_single && !ok_two_tile)
        || queries.dtype()? != DType::BFloat16
        || keys.dtype()? != DType::BFloat16
        || values.dtype()? != DType::BFloat16
        || tree_mask.dtype()? != DType::Uint32
    {
        return Ok(None);
    }
    let handle = unsafe {
        sys::mlx_sdpa_tc_tree_few_cuda(
            queries.as_raw_ptr(),
            keys.as_raw_ptr(),
            values.as_raw_ptr(),
            tree_mask.as_raw_ptr(),
            cache_len,
            scale,
        )
    };
    if handle.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(handle, "sdpa_tc_tree_few_cuda")?))
    }
}
