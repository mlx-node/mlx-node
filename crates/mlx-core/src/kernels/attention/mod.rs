//! Backend-neutral Tensor Core / Metal attention dispatch facades.

mod cuda;
mod metal;

use napi::bindgen_prelude::*;

use crate::accelerator::{ActiveBackend, active_backend};
use crate::array::MxArray;

/// Small-query (`Q in 2..=16`) causal Flash-Decoding SDPA fast path on CUDA
/// (`D=256`, `GQA > 1`, `M = gqa * Q <= 96` or two-tile `Q > 8 && gqa * 8 <= 96`, BF16).
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
    if !(2..=16).contains(&seq_len) {
        return Ok(None);
    }
    sdpa_causal_few_q1_to_16(queries, keys, values, head_dim, gqa, seq_len, scale)
}

/// Small-query (`Q in 1..=16`) causal Flash-Decoding SDPA fast path on CUDA
/// (`D=256`, `GQA > 1`, `M = gqa * Q <= 96` or two-tile `Q > 8 && gqa * 8 <= 96`, BF16).
#[inline]
#[allow(dead_code)]
pub(crate) fn sdpa_causal_few_q1_to_16(
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
            cuda::sdpa_causal_few_q1_to_16(queries, keys, values, head_dim, gqa, seq_len, scale)
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
            cuda::sdpa_prefill_d256(queries, keys, values, head_dim, seq_len, scale)
        }
        ActiveBackend::MetalGpu | ActiveBackend::Cpu => Ok(None),
    }
}

/// Causal FlashAttention-2 prefill fast path on CUDA (`D=256`, `Q_len >= 9`, BF16)
/// with fused `[B, Q_len, Hq * 256]` layout and `sigmoid(gate)` epilogue.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn sdpa_prefill_d256_gated(
    queries: &MxArray,
    keys: &MxArray,
    values: &MxArray,
    gate: &MxArray,
    head_dim: i64,
    seq_len: i64,
    scale: f32,
) -> Result<Option<MxArray>> {
    match active_backend() {
        ActiveBackend::CudaGpu => {
            cuda::sdpa_prefill_d256_gated(queries, keys, values, gate, head_dim, seq_len, scale)
        }
        ActiveBackend::MetalGpu | ActiveBackend::Cpu => Ok(None),
    }
}

/// Unmasked small-query (`Q in 1..=16`) Flash-Decoding SDPA fast path on CUDA
/// (`D in {128, 256}`, `GQA >= 1`, `M = gqa * Q <= 96` or two-tile `Q > 8 && gqa * 8 <= 96`, BF16).
#[inline]
pub(crate) fn sdpa_unmasked_few(
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
            cuda::sdpa_unmasked_few(queries, keys, values, head_dim, gqa, seq_len, scale)
        }
        ActiveBackend::MetalGpu | ActiveBackend::Cpu => Ok(None),
    }
}

/// Masked small-query (`Q in 1..=16`) Flash-Decoding SDPA fast path on CUDA
/// (`D in {128, 256}`, `GQA >= 1`, `M = gqa * Q <= 96` or two-tile `Q > 8 && gqa * 8 <= 96`,
/// BF16, boolean mask).
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn sdpa_masked_few(
    queries: &MxArray,
    keys: &MxArray,
    values: &MxArray,
    mask: &MxArray,
    head_dim: i64,
    gqa: i64,
    seq_len: i64,
    scale: f32,
) -> Result<Option<MxArray>> {
    match active_backend() {
        ActiveBackend::CudaGpu => {
            cuda::sdpa_masked_few(queries, keys, values, mask, head_dim, gqa, seq_len, scale)
        }
        ActiveBackend::MetalGpu | ActiveBackend::Cpu => Ok(None),
    }
}

/// Masked FlashAttention-2 prefill fast path on CUDA (`D=256`, `Q_len >= 9`, BF16, boolean mask).
#[inline]
pub(crate) fn sdpa_prefill_d256_masked(
    queries: &MxArray,
    keys: &MxArray,
    values: &MxArray,
    mask: &MxArray,
    head_dim: i64,
    seq_len: i64,
    scale: f32,
) -> Result<Option<MxArray>> {
    match active_backend() {
        ActiveBackend::CudaGpu => {
            cuda::sdpa_prefill_d256_masked(queries, keys, values, mask, head_dim, seq_len, scale)
        }
        ActiveBackend::MetalGpu | ActiveBackend::Cpu => Ok(None),
    }
}

/// Causal + per-key validity `[B|1, 1, 1, K]` FlashAttention-2 prefill fast
/// path on CUDA (`D=256`, `Q_len >= 9`, `K_len >= Q_len`, BF16, boolean
/// validity mask with diagonal self-attend for padded query rows).
#[inline]
pub(crate) fn sdpa_prefill_d256_causal_valid(
    queries: &MxArray,
    keys: &MxArray,
    values: &MxArray,
    valid_mask: &MxArray,
    scale: f32,
) -> Result<Option<MxArray>> {
    match active_backend() {
        ActiveBackend::CudaGpu => {
            cuda::sdpa_prefill_d256_causal_valid(queries, keys, values, valid_mask, scale)
        }
        ActiveBackend::MetalGpu | ActiveBackend::Cpu => Ok(None),
    }
}

/// Masked FlashAttention-2 prefill fast path on CUDA (`D=256`, `Q_len >= 9`,
/// BF16, boolean mask) with fused `[B, Q_len, Hq * 256]` layout and
/// `sigmoid(gate)` epilogue.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn sdpa_prefill_d256_masked_gated(
    queries: &MxArray,
    keys: &MxArray,
    values: &MxArray,
    gate: &MxArray,
    mask: &MxArray,
    head_dim: i64,
    seq_len: i64,
    scale: f32,
) -> Result<Option<MxArray>> {
    match active_backend() {
        ActiveBackend::CudaGpu => cuda::sdpa_prefill_d256_masked_gated(
            queries, keys, values, gate, mask, head_dim, seq_len, scale,
        ),
        ActiveBackend::MetalGpu | ActiveBackend::Cpu => Ok(None),
    }
}

/// Causal + per-key validity `[B|1, 1, 1, K]` FlashAttention-2 prefill fast
/// path on CUDA (`D=256`, `Q_len >= 9`, `K_len >= Q_len`, BF16, boolean
/// validity mask with diagonal self-attend for padded query rows) with fused
/// `[B, Q_len, Hq * 256]` layout and `sigmoid(gate)` epilogue.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn sdpa_prefill_d256_causal_valid_gated(
    queries: &MxArray,
    keys: &MxArray,
    values: &MxArray,
    gate: &MxArray,
    valid_mask: &MxArray,
    scale: f32,
) -> Result<Option<MxArray>> {
    match active_backend() {
        ActiveBackend::CudaGpu => cuda::sdpa_prefill_d256_causal_valid_gated(
            queries, keys, values, gate, valid_mask, scale,
        ),
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
            metal::sdpa_draft_masked(queries, keys, values, mask, scale, tensor_attention_enabled)
        }
        ActiveBackend::CudaGpu => cuda::sdpa_draft_masked(
            queries,
            keys,
            values,
            mask,
            head_dim,
            num_heads,
            num_kv_heads,
            seq_len,
            scale,
        ),
        ActiveBackend::Cpu => Ok(None),
    }
}

/// DFlash2 masked/unmasked small-query attention fast path outputting `[B, T, Hq * D]`
/// (BTHD) directly on CUDA via `mlx_sdpa_tc_few_masked_bthd_cuda`, eliminating the
/// post-SDPA `[B, Hq, T, D] -> [B, T, Hq, D]` transpose copy.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn sdpa_draft_masked_bthd(
    queries: &MxArray,
    keys: &MxArray,
    values: &MxArray,
    mask: Option<&MxArray>,
    head_dim: i64,
    num_heads: i64,
    num_kv_heads: i64,
    seq_len: i64,
    scale: f32,
) -> Result<Option<MxArray>> {
    match active_backend() {
        ActiveBackend::CudaGpu => cuda::sdpa_draft_masked_bthd(
            queries,
            keys,
            values,
            mask,
            head_dim,
            num_heads,
            num_kv_heads,
            seq_len,
            scale,
        ),
        ActiveBackend::MetalGpu | ActiveBackend::Cpu => Ok(None),
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
    match active_backend() {
        ActiveBackend::MetalGpu => metal::qk_norm_rope_offsets(
            queries, keys, q_weight, k_weight, offsets, eps, rope_base, rope_scale, rope_dims,
        ),
        ActiveBackend::CudaGpu => cuda::qk_norm_rope_offsets(
            queries, keys, q_weight, k_weight, offsets, eps, rope_base, rope_scale, rope_dims,
        ),
        ActiveBackend::Cpu => Ok(None),
    }
}

/// Fused CUDA Q/K RMSNorm + non-traditional partial RoPE with explicit per-token
/// `positions` (`int32[B, T]`) and optional FP32 `PreciseF32` norm.
///
/// Inputs: `queries` `[B, T, HQ, D]`, `keys` `[B, T, HK, D]`, `q_norm_w`/`k_norm_w` `[D]`,
/// `positions` `int32[B, T]`. Outputs: `([B, HQ, T, D], [B, HK, T, D])`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn qk_norm_rope_positions(
    queries: &MxArray,
    keys: &MxArray,
    q_norm_w: &MxArray,
    k_norm_w: &MxArray,
    positions: &MxArray,
    eps: f32,
    rope_dims: i32,
    rope_base: f32,
    rope_scale: f32,
    precise_f32: bool,
) -> Result<Option<(MxArray, MxArray)>> {
    match active_backend() {
        ActiveBackend::CudaGpu => cuda::qk_norm_rope_positions(
            queries,
            keys,
            q_norm_w,
            k_norm_w,
            positions,
            eps,
            rope_dims,
            rope_base,
            rope_scale,
            precise_f32,
        ),
        ActiveBackend::MetalGpu | ActiveBackend::Cpu => Ok(None),
    }
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
    match active_backend() {
        ActiveBackend::CudaGpu => cuda::sdpa_tree_few(
            queries, keys, values, tree_mask, cache_len, head_dim, gqa, seq_len, scale,
        ),
        ActiveBackend::MetalGpu | ActiveBackend::Cpu => Ok(None),
    }
}
