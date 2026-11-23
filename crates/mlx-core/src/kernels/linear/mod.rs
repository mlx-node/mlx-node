//! Backend-neutral small-row dense & quantized linear kernel dispatch.

mod common;
mod cuda;
mod metal;

use napi::bindgen_prelude::*;

use crate::accelerator::{ActiveBackend, active_backend};
use crate::array::MxArray;

pub(crate) use common::intern_kquant_mode;

/// Small-row (`M in 1..=16`) BF16 GEMM (`X @ W^T`) fast path.
#[inline]
pub(crate) fn bf16_matmul_few(
    input: &MxArray,
    weight: &MxArray,
    out_features: i64,
) -> Result<Option<MxArray>> {
    match active_backend() {
        ActiveBackend::CudaGpu => cuda::bf16_matmul_few(input, weight, out_features),
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
            metal::affine_bf16_matmul_few(x, weight, scales, biases, group_size, bits)
        }
        ActiveBackend::CudaGpu if bits == 8 && group_size == 32 => {
            cuda::affine_q8_matmul_few(x, weight, scales, biases)
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
    match active_backend() {
        ActiveBackend::CudaGpu => cuda::kquant_matmul_few(x, weight, scales, biases, mode),
        ActiveBackend::MetalGpu | ActiveBackend::Cpu => Ok(None),
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
    match active_backend() {
        ActiveBackend::CudaGpu => cuda::kquant_matmul_prefill(x, weight, scales, biases, mode),
        ActiveBackend::MetalGpu | ActiveBackend::Cpu => Ok(None),
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
    match active_backend() {
        ActiveBackend::CudaGpu => cuda::kquant_matmul_multi(x, rows, projections),
        ActiveBackend::MetalGpu | ActiveBackend::Cpu => Ok(None),
    }
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
    match active_backend() {
        ActiveBackend::CudaGpu => cuda::affine_q4_g64_matmul_few(x, weight, scales, biases),
        ActiveBackend::MetalGpu | ActiveBackend::Cpu => Ok(None),
    }
}

/// 1-pass affine Q8_0 (`bits=8, group_size=32`) -> BF16 dequantization on CUDA.
#[inline]
#[allow(dead_code)]
pub(crate) fn dequant_q8_bf16_cuda(
    w: &MxArray,
    scales: &MxArray,
    biases: &MxArray,
) -> Result<Option<MxArray>> {
    match active_backend() {
        ActiveBackend::CudaGpu => cuda::dequant_q8_bf16_cuda(w, scales, biases),
        ActiveBackend::MetalGpu | ActiveBackend::Cpu => Ok(None),
    }
}

/// Direct fused `Q8_0` (`bits=8, group_size=32, f16 scales/biases`) Tensor-Core
/// prefill GEMM on CUDA (eliminates BF16 dense weight cache allocation).
///
/// `gs_shift`: `0` = per-row INT8, `7` = GS=128 INT8, `8` = GS=256 INT8,
/// `-1` = auto-grouped (`8` if `K%256==0`, else `7` if `K%128==0`, else `0`),
/// `-2` = bit-exact BF16 MMA (zero X quantization).
#[inline]
#[allow(dead_code)]
pub(crate) fn q8_prefill_cuda(
    x: &MxArray,
    w: &MxArray,
    scales: &MxArray,
    biases: &MxArray,
    gs_shift: i32,
    general_bias: bool,
) -> Result<Option<MxArray>> {
    match active_backend() {
        ActiveBackend::CudaGpu => {
            cuda::q8_prefill_cuda(x, w, scales, biases, gs_shift, general_bias)
        }
        ActiveBackend::MetalGpu | ActiveBackend::Cpu => Ok(None),
    }
}
