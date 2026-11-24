//! Backend-neutral Gated Delta Net (GDN) kernel dispatch facades.

mod cuda;
mod metal;

use napi::bindgen_prelude::*;

use crate::accelerator::{ActiveBackend, active_backend};
use crate::array::MxArray;

/// Fused GDN prepare (`conv1d + SiLU + q/k RMSNorm + decay/beta + next_history`).
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare(
    qkv: &MxArray,
    a: &MxArray,
    b: &MxArray,
    conv_w: &MxArray,
    history: &MxArray,
    scale: &MxArray,
    dt: &MxArray,
    mean_eps: bool,
    beta_input_dtype: bool,
) -> Result<Option<[MxArray; 6]>> {
    match active_backend() {
        ActiveBackend::MetalGpu => metal::prepare(
            qkv,
            a,
            b,
            conv_w,
            history,
            scale,
            dt,
            mean_eps,
            beta_input_dtype,
        ),
        ActiveBackend::CudaGpu => cuda::prepare(
            qkv,
            a,
            b,
            conv_w,
            history,
            scale,
            dt,
            mean_eps,
            beta_input_dtype,
        ),
        ActiveBackend::Cpu => Ok(None),
    }
}

/// Batched / general-geometry CUDA fused GDN prepare
/// (`conv1d + SiLU + q/k RMSNorm + decay/beta + next_history`, with optional
/// per-row `valid_lengths` for right-padded batches).
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_batched(
    qkv: &MxArray,
    a: &MxArray,
    b: &MxArray,
    conv_w: &MxArray,
    history: &MxArray,
    scale: &MxArray,
    dt: &MxArray,
    valid_lengths: Option<&MxArray>,
    mean_eps: bool,
    bf16_beta: bool,
) -> Result<Option<[MxArray; 6]>> {
    match active_backend() {
        ActiveBackend::CudaGpu => cuda::prepare_batched(
            qkv,
            a,
            b,
            conv_w,
            history,
            scale,
            dt,
            valid_lengths,
            mean_eps,
            bf16_beta,
        ),
        ActiveBackend::MetalGpu | ActiveBackend::Cpu => Ok(None),
    }
}

/// Fused GDN recurrence + gated RMSNorm + z-gate in one GPU dispatch (SiLU gate).
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn fused_step(
    q: &MxArray,
    k: &MxArray,
    v: &MxArray,
    g: &MxArray,
    beta: &MxArray,
    state: &MxArray,
    z: &MxArray,
    norm_weight: &MxArray,
    eps: f32,
    tiled_heads: bool,
    state_dst: Option<&MxArray>,
) -> Result<Option<(MxArray, MxArray)>> {
    fused_step_with_gate(
        q,
        k,
        v,
        g,
        beta,
        state,
        z,
        norm_weight,
        eps,
        tiled_heads,
        false,
        state_dst,
    )
}

/// Fused GDN recurrence + gated RMSNorm + z-gate with explicit gate selector:
/// `sigmoid_gate = false` uses SiLU (`z * sigmoid(z)`, Qwen3.5),
/// `sigmoid_gate = true` uses Sigmoid (`sigmoid(z)`, Qwen3.8-Flash-Next / `qwen4_exp`, CUDA).
/// On CUDA, both `beta` and `norm_weight` may be BF16 or FP32.
#[inline]
#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn fused_step_with_gate(
    q: &MxArray,
    k: &MxArray,
    v: &MxArray,
    g: &MxArray,
    beta: &MxArray,
    state: &MxArray,
    z: &MxArray,
    norm_weight: &MxArray,
    eps: f32,
    tiled_heads: bool,
    sigmoid_gate: bool,
    state_dst: Option<&MxArray>,
) -> Result<Option<(MxArray, MxArray)>> {
    match active_backend() {
        ActiveBackend::CudaGpu => cuda::fused_step_with_gate(
            q,
            k,
            v,
            g,
            beta,
            state,
            z,
            norm_weight,
            eps,
            tiled_heads,
            sigmoid_gate,
            state_dst,
        ),
        ActiveBackend::MetalGpu if !sigmoid_gate => {
            metal::fused_step(q, k, v, g, beta, state, z, norm_weight, eps, state_dst)
        }
        ActiveBackend::MetalGpu | ActiveBackend::Cpu => Ok(None),
    }
}

/// Deferred-fold CUDA fused GDN step (SiLU gate): folds `prev_keep` accepted rows from the
/// prior cycle's `(prev_k, prev_v, prev_g, prev_beta)` into `state_base` before
/// stepping the current verify window and applying gated RMSNorm + z-gate.
///
/// When `pending` is `None` (cycle 0), passes null pointers and `prev_keep = 0`,
/// reading `state_base` read-only and returning `(out_y, state_base_alias)`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn fused_step_deferred(
    q: &MxArray,
    k: &MxArray,
    v: &MxArray,
    g: &MxArray,
    beta: &MxArray,
    state_base: &MxArray,
    pending: Option<(&MxArray, &MxArray, &MxArray, &MxArray, i32)>,
    z: &MxArray,
    norm_weight: &MxArray,
    eps: f32,
    tiled_heads: bool,
) -> Result<Option<(MxArray, MxArray)>> {
    fused_step_deferred_with_gate(
        q,
        k,
        v,
        g,
        beta,
        state_base,
        pending,
        z,
        norm_weight,
        eps,
        tiled_heads,
        false,
    )
}

/// Deferred-fold CUDA fused GDN step with explicit `sigmoid_gate` selector
/// (`false` = SiLU, `true` = Qwen4 Sigmoid; BF16/FP32 `beta` and `norm_weight`).
#[inline]
#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn fused_step_deferred_with_gate(
    q: &MxArray,
    k: &MxArray,
    v: &MxArray,
    g: &MxArray,
    beta: &MxArray,
    state_base: &MxArray,
    pending: Option<(&MxArray, &MxArray, &MxArray, &MxArray, i32)>,
    z: &MxArray,
    norm_weight: &MxArray,
    eps: f32,
    tiled_heads: bool,
    sigmoid_gate: bool,
) -> Result<Option<(MxArray, MxArray)>> {
    match active_backend() {
        ActiveBackend::CudaGpu => cuda::fused_step_deferred_with_gate(
            q,
            k,
            v,
            g,
            beta,
            state_base,
            pending,
            z,
            norm_weight,
            eps,
            tiled_heads,
            sigmoid_gate,
        ),
        ActiveBackend::MetalGpu | ActiveBackend::Cpu => Ok(None),
    }
}

/// Tree-verify CUDA fused GDN step with deferred fold from the prior cycle (SiLU gate).
/// `parent` is `int32[T]` (`parent[0] = -1`, `parent[t] < t`).
/// `pending` optionally carries `(prev_k, prev_v, prev_g, prev_beta, prev_keep, prev_indices)`.
#[inline]
#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn tree_step_deferred(
    q: &MxArray,
    k: &MxArray,
    v: &MxArray,
    g: &MxArray,
    beta: &MxArray,
    state_base: &MxArray,
    pending: Option<(
        &MxArray,
        &MxArray,
        &MxArray,
        &MxArray,
        i32,
        Option<&MxArray>,
    )>,
    parent: &MxArray,
    z: &MxArray,
    norm_weight: &MxArray,
    eps: f32,
    tiled_heads: bool,
) -> Result<Option<(MxArray, MxArray)>> {
    tree_step_deferred_with_gate(
        q,
        k,
        v,
        g,
        beta,
        state_base,
        pending,
        parent,
        z,
        norm_weight,
        eps,
        tiled_heads,
        false,
    )
}

/// Tree-verify CUDA fused GDN step with deferred fold and explicit `sigmoid_gate`
/// selector (`false` = SiLU, `true` = Qwen4 Sigmoid; BF16/FP32 `beta` and `norm_weight`).
#[inline]
#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn tree_step_deferred_with_gate(
    q: &MxArray,
    k: &MxArray,
    v: &MxArray,
    g: &MxArray,
    beta: &MxArray,
    state_base: &MxArray,
    pending: Option<(
        &MxArray,
        &MxArray,
        &MxArray,
        &MxArray,
        i32,
        Option<&MxArray>,
    )>,
    parent: &MxArray,
    z: &MxArray,
    norm_weight: &MxArray,
    eps: f32,
    tiled_heads: bool,
    sigmoid_gate: bool,
) -> Result<Option<(MxArray, MxArray)>> {
    match active_backend() {
        ActiveBackend::CudaGpu => cuda::tree_step_deferred_with_gate(
            q,
            k,
            v,
            g,
            beta,
            state_base,
            pending,
            parent,
            z,
            norm_weight,
            eps,
            tiled_heads,
            sigmoid_gate,
        ),
        ActiveBackend::MetalGpu | ActiveBackend::Cpu => Ok(None),
    }
}

/// Tree-conv1d variant of fused CUDA GDN preparation (`conv_taps` is `int32[T, 4]`).
#[inline]
#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn prepare_tree(
    qkv: &MxArray,
    a: &MxArray,
    b: &MxArray,
    conv_w: &MxArray,
    history: &MxArray,
    scale: &MxArray,
    dt: &MxArray,
    conv_taps: &MxArray,
    mean_eps: bool,
    beta_input_dtype: bool,
) -> Result<Option<[MxArray; 6]>> {
    match active_backend() {
        ActiveBackend::CudaGpu => cuda::prepare_tree(
            qkv,
            a,
            b,
            conv_w,
            history,
            scale,
            dt,
            conv_taps,
            mean_eps,
            beta_input_dtype,
        ),
        ActiveBackend::MetalGpu | ActiveBackend::Cpu => Ok(None),
    }
}

/// Register-resident CUDA GDN prefill recurrence (B=1, Dk=Dv=128, BF16 q/k/v).
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn prefill_chain(
    q: &MxArray,
    k: &MxArray,
    v: &MxArray,
    g: &MxArray,
    beta: &MxArray,
    state: &MxArray,
    tiled_heads: bool,
) -> Result<Option<(MxArray, MxArray)>> {
    match active_backend() {
        ActiveBackend::CudaGpu => cuda::prefill_chain(q, k, v, g, beta, state, tiled_heads),
        ActiveBackend::MetalGpu | ActiveBackend::Cpu => Ok(None),
    }
}

/// Register-resident single-block-per-head CUDA GDN prefill recurrence with
/// fused strided-`z` `RMSNormGated` epilogue (`[1, T, hv * 128]` BF16 output, SiLU gate).
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn prefill_chain_gated(
    q: &MxArray,
    k: &MxArray,
    v: &MxArray,
    g: &MxArray,
    beta: &MxArray,
    state: &MxArray,
    z: &MxArray,
    norm_weight: &MxArray,
    eps: f32,
    tiled_heads: bool,
) -> Result<Option<(MxArray, MxArray)>> {
    prefill_chain_gated_with_gate(
        q,
        k,
        v,
        g,
        beta,
        state,
        z,
        norm_weight,
        eps,
        tiled_heads,
        false,
    )
}

/// Register-resident single-block-per-head CUDA GDN prefill recurrence with
/// fused strided-`z` `RMSNormGated` epilogue and explicit `sigmoid_gate` selector
/// (`false` = SiLU, `true` = Qwen4 Sigmoid; BF16/FP32 `beta` and `norm_weight`).
#[inline]
#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn prefill_chain_gated_with_gate(
    q: &MxArray,
    k: &MxArray,
    v: &MxArray,
    g: &MxArray,
    beta: &MxArray,
    state: &MxArray,
    z: &MxArray,
    norm_weight: &MxArray,
    eps: f32,
    tiled_heads: bool,
    sigmoid_gate: bool,
) -> Result<Option<(MxArray, MxArray)>> {
    match active_backend() {
        ActiveBackend::CudaGpu => cuda::prefill_chain_gated_with_gate(
            q,
            k,
            v,
            g,
            beta,
            state,
            z,
            norm_weight,
            eps,
            tiled_heads,
            sigmoid_gate,
        ),
        ActiveBackend::MetalGpu | ActiveBackend::Cpu => Ok(None),
    }
}

/// Fused accepted-prefix recurrent state replay for eager-MTP GDN rollback.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn replay(
    k: &MxArray,
    v: &MxArray,
    g: &MxArray,
    beta: &MxArray,
    start_state: &MxArray,
    accepted_steps: i32,
    tiled_heads: bool,
) -> Result<Option<MxArray>> {
    match active_backend() {
        ActiveBackend::MetalGpu => metal::replay(k, v, g, beta, start_state, accepted_steps),
        ActiveBackend::CudaGpu => {
            cuda::replay(k, v, g, beta, start_state, accepted_steps, tiled_heads)
        }
        ActiveBackend::Cpu => Ok(None),
    }
}

/// CUDA 64x64 strictly-lower-triangular `(I + A)^-1` fast path (non-differentiable).
#[inline]
pub(crate) fn invert64(a: &MxArray) -> Result<Option<MxArray>> {
    match active_backend() {
        ActiveBackend::CudaGpu => cuda::invert64(a),
        ActiveBackend::MetalGpu | ActiveBackend::Cpu => Ok(None),
    }
}
