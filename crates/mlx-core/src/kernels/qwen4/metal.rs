//! Metal kernel facades for Qwen3.8-Flash-Next (`qwen4_exp`), backed by
//! `metal/mlx_qwen4_*.cpp` and `metal/mlx_gated_delta_metal.cpp`.
//!
//! Every wrapper takes only `&MxArray` / `&[&MxArray]` and primitive scalars
//! (zero dependency on `models::qwen4_exp` types). On non-Metal builds the C++
//! entry points return null/false/0 cleanly; callers keep their existing guards
//! and fallback/error handling unchanged.

use mlx_sys as sys;
use napi::bindgen_prelude::*;

use crate::array::MxArray;

// ---------------------------------------------------------------------------
// Hyper-connections, residual norm, and dense quantized projections
// ---------------------------------------------------------------------------

/// Fused prefill hyper-connection combine (`sigmoid(up) * normed` + 4-stream mean).
#[inline]
pub(crate) fn prefill_hc_mix(up: &MxArray, normed: &MxArray) -> Result<Option<MxArray>> {
    let handle = unsafe { sys::mlx_qwen4_prefill_hc_mix(up.as_raw_ptr(), normed.as_raw_ptr()) };
    if handle.is_null() {
        Ok(None)
    } else {
        MxArray::from_handle(handle, "Qwen4 prefill mixer combine").map(Some)
    }
}

/// Fused residual injection + per-stream RMSNorm (`mlx_qwen4_inject_norm`).
#[inline]
pub(crate) fn inject_norm(
    x: &MxArray,
    branch: &MxArray,
    gate: &MxArray,
    weight: &MxArray,
    eps: f64,
) -> Result<Option<(MxArray, MxArray)>> {
    let (mut stream, mut normed) = (std::ptr::null_mut(), std::ptr::null_mut());
    let ok = unsafe {
        sys::mlx_qwen4_inject_norm(
            x.as_raw_ptr(),
            branch.as_raw_ptr(),
            gate.as_raw_ptr(),
            weight.as_raw_ptr(),
            eps,
            &mut stream,
            &mut normed,
        )
    };
    if !ok {
        return Ok(None);
    }
    Ok(Some((
        MxArray::from_handle(stream, "Qwen4 fused injection stream")?,
        MxArray::from_handle(normed, "Qwen4 fused injection normalization")?,
    )))
}

/// Fused single-row Q8 hyper-connection down projection + mixer activation.
#[inline]
pub(crate) fn decode_mixer_act(
    x: &MxArray,
    weight: &MxArray,
    scales: &MxArray,
    biases: &MxArray,
) -> Result<Option<MxArray>> {
    let raw = unsafe {
        sys::mlx_qwen4_decode_mixer_act(
            x.as_raw_ptr(),
            weight.as_raw_ptr(),
            scales.as_raw_ptr(),
            biases.as_raw_ptr(),
        )
    };
    if raw.is_null() {
        Ok(None)
    } else {
        MxArray::from_handle(raw, "Qwen4 reference mixer activation").map(Some)
    }
}

/// Fused prefill Q8 hyper-connection down projection + mixer activation.
#[inline]
pub(crate) fn prefill_mixer_act(
    x: &MxArray,
    weight: &MxArray,
    scales: &MxArray,
    biases: &MxArray,
) -> Result<Option<MxArray>> {
    let raw = unsafe {
        sys::mlx_qwen4_prefill_mixer_act(
            x.as_raw_ptr(),
            weight.as_raw_ptr(),
            scales.as_raw_ptr(),
            biases.as_raw_ptr(),
        )
    };
    if raw.is_null() {
        Ok(None)
    } else {
        MxArray::from_handle(raw, "Qwen4 reference mixer activation").map(Some)
    }
}

/// Fused Q8 hyper-connection up projection + sigmoid + 4-stream weighted mean.
#[inline]
pub(crate) fn hyper_up(
    x: &MxArray,
    weight: &MxArray,
    scales: &MxArray,
    biases: &MxArray,
    normed: &MxArray,
) -> Result<Option<MxArray>> {
    let raw = unsafe {
        sys::mlx_qwen4_hyper_up(
            x.as_raw_ptr(),
            weight.as_raw_ptr(),
            scales.as_raw_ptr(),
            biases.as_raw_ptr(),
            normed.as_raw_ptr(),
        )
    };
    if raw.is_null() {
        Ok(None)
    } else {
        MxArray::from_handle(raw, "Qwen4 hyper up").map(Some)
    }
}

/// Compact Q8 dense decode GEMV (`mlx_qwen4_dense_decode`).
#[inline]
pub(crate) fn dense_decode(
    x: &MxArray,
    weight: &MxArray,
    scales: &MxArray,
    biases: &MxArray,
) -> Result<Option<MxArray>> {
    let raw = unsafe {
        sys::mlx_qwen4_dense_decode(
            x.as_raw_ptr(),
            weight.as_raw_ptr(),
            scales.as_raw_ptr(),
            biases.as_raw_ptr(),
        )
    };
    if raw.is_null() {
        Ok(None)
    } else {
        MxArray::from_handle(raw, "Qwen4 compact dense decode").map(Some)
    }
}

/// Compact Q8 dense prefill GEMM (`mlx_qwen4_dense_prefill`).
#[inline]
pub(crate) fn dense_prefill(
    x: &MxArray,
    weight: &MxArray,
    scales: &MxArray,
    biases: &MxArray,
) -> Result<Option<MxArray>> {
    let raw = unsafe {
        sys::mlx_qwen4_dense_prefill(
            x.as_raw_ptr(),
            weight.as_raw_ptr(),
            scales.as_raw_ptr(),
            biases.as_raw_ptr(),
        )
    };
    if raw.is_null() {
        Ok(None)
    } else {
        MxArray::from_handle(raw, "Qwen4 compact dense prefill").map(Some)
    }
}

// ---------------------------------------------------------------------------
// Row-batched decode kernels (1..=4 rows sharing one weight stream)
// ---------------------------------------------------------------------------

/// Row-batched compact Q8 dense decode (`mlx_qwen4_dense_decode_rows`).
#[inline]
pub(crate) fn dense_decode_rows(
    x: &MxArray,
    weight: &MxArray,
    scales: &MxArray,
    biases: &MxArray,
) -> Result<Option<MxArray>> {
    let raw = unsafe {
        sys::mlx_qwen4_dense_decode_rows(
            x.as_raw_ptr(),
            weight.as_raw_ptr(),
            scales.as_raw_ptr(),
            biases.as_raw_ptr(),
        )
    };
    if raw.is_null() {
        Ok(None)
    } else {
        MxArray::from_handle(raw, "Qwen4 row-batched dense decode").map(Some)
    }
}

/// Row-batched Q8 mixer down projection + activation (`mlx_qwen4_decode_mixer_act_rows`).
#[inline]
pub(crate) fn decode_mixer_act_rows(
    x: &MxArray,
    weight: &MxArray,
    scales: &MxArray,
    biases: &MxArray,
) -> Result<Option<MxArray>> {
    let raw = unsafe {
        sys::mlx_qwen4_decode_mixer_act_rows(
            x.as_raw_ptr(),
            weight.as_raw_ptr(),
            scales.as_raw_ptr(),
            biases.as_raw_ptr(),
        )
    };
    if raw.is_null() {
        Ok(None)
    } else {
        MxArray::from_handle(raw, "Qwen4 row-batched mixer activation").map(Some)
    }
}

/// Row-batched Q8 hyper-connection up projection + mix (`mlx_qwen4_hyper_up_rows`).
#[inline]
pub(crate) fn hyper_up_rows(
    x: &MxArray,
    weight: &MxArray,
    scales: &MxArray,
    biases: &MxArray,
    normed: &MxArray,
) -> Result<Option<MxArray>> {
    let raw = unsafe {
        sys::mlx_qwen4_hyper_up_rows(
            x.as_raw_ptr(),
            weight.as_raw_ptr(),
            scales.as_raw_ptr(),
            biases.as_raw_ptr(),
            normed.as_raw_ptr(),
        )
    };
    if raw.is_null() {
        Ok(None)
    } else {
        MxArray::from_handle(raw, "Qwen4 row-batched hyper up").map(Some)
    }
}

/// Row-batched fused routed + shared MoE decode (`mlx_qwen4_routed_shared_experts_rows`).
///
/// Expects the 22-element ABI input array: `[x, ids, scores, routed(gate,up,down * 3),
/// shared(gate,up,down * 3), shared_gate]`.
#[inline]
pub(crate) fn routed_shared_experts_rows(inputs: &[&MxArray; 22]) -> Result<Option<MxArray>> {
    let ptrs: [*mut sys::mlx_array; 22] = std::array::from_fn(|i| inputs[i].as_raw_ptr());
    let raw = unsafe { sys::mlx_qwen4_routed_shared_experts_rows(ptrs.as_ptr(), ptrs.len()) };
    if raw.is_null() {
        Ok(None)
    } else {
        MxArray::from_handle(raw, "Qwen4 row-batched routed/shared experts").map(Some)
    }
}

// ---------------------------------------------------------------------------
// MoE routing and expert projections
// ---------------------------------------------------------------------------

/// Fused routed-expert MoE decode (`mlx_qwen4_routed_experts`).
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn routed_experts(
    x: &MxArray,
    ids: &MxArray,
    scores: &MxArray,
    gate_w: &MxArray,
    gate_s: &MxArray,
    gate_b: &MxArray,
    up_w: &MxArray,
    up_s: &MxArray,
    up_b: &MxArray,
    down_w: &MxArray,
    down_s: &MxArray,
    down_b: &MxArray,
) -> Result<Option<MxArray>> {
    let raw = unsafe {
        sys::mlx_qwen4_routed_experts(
            x.as_raw_ptr(),
            ids.as_raw_ptr(),
            scores.as_raw_ptr(),
            gate_w.as_raw_ptr(),
            gate_s.as_raw_ptr(),
            gate_b.as_raw_ptr(),
            up_w.as_raw_ptr(),
            up_s.as_raw_ptr(),
            up_b.as_raw_ptr(),
            down_w.as_raw_ptr(),
            down_s.as_raw_ptr(),
            down_b.as_raw_ptr(),
        )
    };
    if raw.is_null() {
        Ok(None)
    } else {
        MxArray::from_handle(raw, "Qwen4 fused routed experts").map(Some)
    }
}

/// Fused routed + shared MoE decode (`mlx_qwen4_routed_shared_experts`).
///
/// Expects the 22-element ABI input array: `[x, ids, scores, routed(gate,up,down * 3),
/// shared(gate,up,down * 3), shared_gate]`.
#[inline]
pub(crate) fn routed_shared_experts(inputs: &[&MxArray; 22]) -> Result<Option<MxArray>> {
    let ptrs: [*mut sys::mlx_array; 22] = std::array::from_fn(|i| inputs[i].as_raw_ptr());
    let raw = unsafe { sys::mlx_qwen4_routed_shared_experts(ptrs.as_ptr(), ptrs.len()) };
    if raw.is_null() {
        Ok(None)
    } else {
        MxArray::from_handle(raw, "Qwen4 fused routed/shared experts").map(Some)
    }
}

/// Indirect sorted-route expert prefill (`mlx_qwen4_prefill_indirect`).
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn prefill_indirect(
    x: &MxArray,
    ids: &MxArray,
    token_rows: &MxArray,
    gate_w: &MxArray,
    gate_s: &MxArray,
    gate_b: &MxArray,
    up_w: &MxArray,
    up_s: &MxArray,
    up_b: &MxArray,
    down_w: &MxArray,
    down_s: &MxArray,
    down_b: &MxArray,
    experts: usize,
) -> Result<Option<MxArray>> {
    let raw = unsafe {
        sys::mlx_qwen4_prefill_indirect(
            x.as_raw_ptr(),
            ids.as_raw_ptr(),
            token_rows.as_raw_ptr(),
            gate_w.as_raw_ptr(),
            gate_s.as_raw_ptr(),
            gate_b.as_raw_ptr(),
            up_w.as_raw_ptr(),
            up_s.as_raw_ptr(),
            up_b.as_raw_ptr(),
            down_w.as_raw_ptr(),
            down_s.as_raw_ptr(),
            down_b.as_raw_ptr(),
            experts as i32,
        )
    };
    if raw.is_null() {
        Ok(None)
    } else {
        MxArray::from_handle(raw, "Qwen4 indirect expert prefill").map(Some)
    }
}

/// Expert tile table builder for NAX tiled expert prefill (`mlx_qwen4_expert_tiles`).
#[inline]
pub(crate) fn expert_tiles(ids: &MxArray, experts: usize) -> Result<Option<MxArray>> {
    let raw = unsafe { sys::mlx_qwen4_expert_tiles(ids.as_raw_ptr(), experts as i32) };
    if raw.is_null() {
        Ok(None)
    } else {
        MxArray::from_handle(raw, "Qwen4 expert tile table").map(Some)
    }
}

/// Sorted expert output weighted combine (`mlx_qwen4_sorted_combine`).
#[inline]
pub(crate) fn sorted_combine(
    values: &MxArray,
    scores: &MxArray,
    inverse: &MxArray,
    top: usize,
) -> Result<Option<MxArray>> {
    let raw = unsafe {
        sys::mlx_qwen4_sorted_combine(
            values.as_raw_ptr(),
            scores.as_raw_ptr(),
            inverse.as_raw_ptr(),
            top as i32,
        )
    };
    if raw.is_null() {
        Ok(None)
    } else {
        MxArray::from_handle(raw, "Qwen4 sorted expert combine").map(Some)
    }
}

/// Singleton router softmax + top-10 + normalize (`mlx_qwen4_singleton_routes`).
#[inline]
pub(crate) fn singleton_routes(logits: &MxArray) -> Result<(MxArray, MxArray)> {
    let (mut ids, mut scores) = (std::ptr::null_mut(), std::ptr::null_mut());
    let ok = unsafe { sys::mlx_qwen4_singleton_routes(logits.as_raw_ptr(), &mut ids, &mut scores) };
    if !ok {
        return Err(Error::from_reason("Qwen4 singleton routing kernel failed"));
    }
    Ok((
        MxArray::from_handle(ids, "singleton route IDs")?,
        MxArray::from_handle(scores, "singleton route scores")?,
    ))
}

/// Multi-row router softmax + top-10 + normalize (`mlx_qwen4_prefill_routes`).
///
/// Returns `Ok(None)` when the FFI call returns `false` so callers can emit
/// their context-specific error message.
#[inline]
pub(crate) fn prefill_routes(logits: &MxArray) -> Result<Option<(MxArray, MxArray)>> {
    let (mut ids, mut scores) = (std::ptr::null_mut(), std::ptr::null_mut());
    let ok = unsafe { sys::mlx_qwen4_prefill_routes(logits.as_raw_ptr(), &mut ids, &mut scores) };
    if !ok {
        return Ok(None);
    }
    Ok(Some((
        MxArray::from_handle(ids, "prefill route IDs")?,
        MxArray::from_handle(scores, "prefill route scores")?,
    )))
}

/// Stable GPU route sort (`mlx_qwen4_route_sort`).
#[inline]
pub(crate) fn route_sort(ids: &MxArray, experts: usize) -> Result<(MxArray, MxArray, MxArray)> {
    let (mut order, mut inverse, mut sorted) = (
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        std::ptr::null_mut(),
    );
    let ok = unsafe {
        sys::mlx_qwen4_route_sort(
            ids.as_raw_ptr(),
            experts as i32,
            &mut order,
            &mut inverse,
            &mut sorted,
        )
    };
    if !ok {
        return Err(Error::from_reason("Qwen4 stable routing kernel failed"));
    }
    Ok((
        MxArray::from_handle(order, "route order")?,
        MxArray::from_handle(inverse, "route inverse")?,
        MxArray::from_handle(sorted, "sorted routes")?,
    ))
}

/// Expert-aligned tiled prefill GEMM (`mlx_qwen4_expert_prefill`).
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn expert_prefill(
    x: &MxArray,
    ids: &MxArray,
    tiles: &MxArray,
    weight: &MxArray,
    scales: &MxArray,
    biases: &MxArray,
    experts: usize,
    group: i32,
    bits: i32,
    mode: &str,
) -> Result<Option<MxArray>> {
    let mode = std::ffi::CString::new(mode).map_err(|e| Error::from_reason(e.to_string()))?;
    let out = unsafe {
        sys::mlx_qwen4_expert_prefill(
            x.as_raw_ptr(),
            ids.as_raw_ptr(),
            tiles.as_raw_ptr(),
            weight.as_raw_ptr(),
            scales.as_raw_ptr(),
            biases.as_raw_ptr(),
            experts as i32,
            group,
            bits,
            mode.as_ptr(),
        )
    };
    if out.is_null() {
        Ok(None)
    } else {
        MxArray::from_handle(out, "Qwen4 expert-aligned prefill").map(Some)
    }
}

/// Direct affine quantized expert GEMV (`mlx_qwen4_affine_expert_gemv`).
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn affine_expert_gemv(
    x: &MxArray,
    ids: &MxArray,
    weight: &MxArray,
    scales: &MxArray,
    biases: &MxArray,
    experts: usize,
    group: i32,
    bits: i32,
) -> Result<MxArray> {
    let out = unsafe {
        sys::mlx_qwen4_affine_expert_gemv(
            x.as_raw_ptr(),
            ids.as_raw_ptr(),
            weight.as_raw_ptr(),
            scales.as_raw_ptr(),
            biases.as_raw_ptr(),
            experts as i32,
            group,
            bits,
        )
    };
    MxArray::from_handle(out, "Qwen4 affine expert GEMV")
}

/// Direct K-quant expert GEMV (`mlx_qwen4_expert_gemv`).
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn expert_gemv(
    x: &MxArray,
    ids: &MxArray,
    weight: &MxArray,
    scales: &MxArray,
    biases: &MxArray,
    experts: usize,
    group: i32,
    bits: i32,
    mode: &str,
) -> Result<MxArray> {
    let mode = std::ffi::CString::new(mode).map_err(|e| Error::from_reason(e.to_string()))?;
    let out = unsafe {
        sys::mlx_qwen4_expert_gemv(
            x.as_raw_ptr(),
            ids.as_raw_ptr(),
            weight.as_raw_ptr(),
            scales.as_raw_ptr(),
            biases.as_raw_ptr(),
            experts as i32,
            group,
            bits,
            mode.as_ptr(),
        )
    };
    MxArray::from_handle(out, "Qwen4 direct expert GEMV")
}

// ---------------------------------------------------------------------------
// Gated Delta Net (GDN) and convolutions
// ---------------------------------------------------------------------------

/// Complete singleton GGUF GDN graph (`mlx_qwen4_complete_gdn`).
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn complete_gdn(
    qkv: &MxArray,
    z: &MxArray,
    a: &MxArray,
    b: &MxArray,
    conv: &MxArray,
    history: &MxArray,
    scale: &MxArray,
    dt: &MxArray,
    state: &MxArray,
    norm: &MxArray,
    eps: f64,
) -> Result<(MxArray, MxArray, MxArray)> {
    let (mut out, mut next, mut next_history) = (
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        std::ptr::null_mut(),
    );
    let ok = unsafe {
        sys::mlx_qwen4_complete_gdn(
            qkv.as_raw_ptr(),
            z.as_raw_ptr(),
            a.as_raw_ptr(),
            b.as_raw_ptr(),
            conv.as_raw_ptr(),
            history.as_raw_ptr(),
            scale.as_raw_ptr(),
            dt.as_raw_ptr(),
            state.as_raw_ptr(),
            norm.as_raw_ptr(),
            eps,
            &mut out,
            &mut next,
            &mut next_history,
        )
    };
    if !ok {
        return Err(Error::from_reason("Qwen4 complete GDN graph failed"));
    }
    Ok((
        MxArray::from_handle(out, "Qwen4 complete GDN output")?,
        MxArray::from_handle(next, "Qwen4 complete GDN state")?,
        MxArray::from_handle(next_history, "Qwen4 complete GDN history")?,
    ))
}

/// Fused Metal gated-delta recurrence kernel (`mlx_qwen4_gated_delta_kernel`).
///
/// Returns `Ok(None)` when the FFI call returns `false` so callers can emit
/// their context-specific error message.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gated_delta_kernel(
    q: &MxArray,
    k: &MxArray,
    v: &MxArray,
    g: &MxArray,
    beta: &MxArray,
    state: &MxArray,
    mask: Option<&MxArray>,
) -> Result<Option<(MxArray, MxArray)>> {
    let (mut out, mut next) = (std::ptr::null_mut(), std::ptr::null_mut());
    let mask_ptr = mask.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    let ok = unsafe {
        sys::mlx_qwen4_gated_delta_kernel(
            q.as_raw_ptr(),
            k.as_raw_ptr(),
            v.as_raw_ptr(),
            g.as_raw_ptr(),
            beta.as_raw_ptr(),
            state.as_raw_ptr(),
            mask_ptr,
            &mut out,
            &mut next,
        )
    };
    if !ok {
        return Ok(None);
    }
    Ok(Some((
        MxArray::from_handle(out, "Qwen4 recurrent output")?,
        MxArray::from_handle(next, "Qwen4 recurrent state")?,
    )))
}

/// Fused Metal GDN preparation (`mlx_qwen4_gdn_prepare`):
/// returns `[q, k, v, decay, beta, next_history]`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gdn_prepare(
    qkv: &MxArray,
    a: &MxArray,
    b: &MxArray,
    conv: &MxArray,
    history: &MxArray,
    scale: &MxArray,
    dt: &MxArray,
    mean_eps: bool,
    beta_input_dtype: bool,
) -> Result<Option<[MxArray; 6]>> {
    let mut outputs = [std::ptr::null_mut(); 6];
    let ok = unsafe {
        sys::mlx_qwen4_gdn_prepare(
            qkv.as_raw_ptr(),
            a.as_raw_ptr(),
            b.as_raw_ptr(),
            conv.as_raw_ptr(),
            history.as_raw_ptr(),
            scale.as_raw_ptr(),
            dt.as_raw_ptr(),
            mean_eps,
            beta_input_dtype,
            outputs.as_mut_ptr(),
        )
    };
    if !ok {
        return Ok(None);
    }
    let [q, k, v, decay, beta, next_history] = outputs;
    Ok(Some([
        MxArray::from_handle(q, "GDN prepared Q")?,
        MxArray::from_handle(k, "GDN prepared K")?,
        MxArray::from_handle(v, "GDN prepared V")?,
        MxArray::from_handle(decay, "GDN prepared decay")?,
        MxArray::from_handle(beta, "GDN prepared beta")?,
        MxArray::from_handle(next_history, "GDN prepared history")?,
    ]))
}

/// Fused causal depthwise window convolution + SiLU (`mlx_qwen4_window_conv`).
#[inline]
pub(crate) fn window_conv(
    x: &MxArray,
    history: &MxArray,
    weight: &MxArray,
) -> Result<Option<(MxArray, MxArray)>> {
    let (mut out, mut next_history) = (std::ptr::null_mut(), std::ptr::null_mut());
    let ok = unsafe {
        sys::mlx_qwen4_window_conv(
            x.as_raw_ptr(),
            history.as_raw_ptr(),
            weight.as_raw_ptr(),
            &mut out,
            &mut next_history,
        )
    };
    if !ok {
        return Ok(None);
    }
    Ok(Some((
        MxArray::from_handle(out, "Qwen4 window convolution")?,
        MxArray::from_handle(next_history, "Qwen4 convolution history")?,
    )))
}

/// Prefill dilated PLE depthwise convolution (`mlx_qwen4_prefill_ple_conv`).
#[inline]
pub(crate) fn prefill_ple_conv(full: &MxArray, weight: &MxArray) -> Result<Option<MxArray>> {
    let raw = unsafe { sys::mlx_qwen4_prefill_ple_conv(full.as_raw_ptr(), weight.as_raw_ptr()) };
    if raw.is_null() {
        Ok(None)
    } else {
        MxArray::from_handle(raw, "Qwen4 reference PLE convolution").map(Some)
    }
}

// ---------------------------------------------------------------------------
// Qwen4 Sparse Attention (QSA) and paged KV gather
// ---------------------------------------------------------------------------

/// Indexed QSA sparse attention kernel (`mlx_qwen4_qsa_attention`).
///
/// Returns `(output, heads_per_kv_read)` where `heads_per_kv_read` is `1` or `2`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn qsa_attention(
    queries: &MxArray,
    keys: &MxArray,
    values: &MxArray,
    selected: &MxArray,
    query_ends: &MxArray,
    scale: f32,
    block_size: usize,
    heads_per_group: i32,
) -> Result<(MxArray, i32)> {
    let mut out = std::ptr::null_mut();
    let heads = unsafe {
        sys::mlx_qwen4_qsa_attention(
            queries.as_raw_ptr(),
            keys.as_raw_ptr(),
            values.as_raw_ptr(),
            selected.as_raw_ptr(),
            query_ends.as_raw_ptr(),
            scale,
            block_size as i32,
            heads_per_group,
            &mut out,
        )
    };
    if heads == 0 {
        return Err(Error::from_reason("Qwen4 sparse attention kernel failed"));
    }
    Ok((MxArray::from_handle(out, "Qwen4 sparse attention")?, heads))
}

/// Test-only probe returning `maxTotalThreadsPerThreadgroup` for a QSA pipeline.
#[cfg(test)]
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn qsa_attention_pipeline_threads(
    queries: &MxArray,
    keys: &MxArray,
    values: &MxArray,
    selected: &MxArray,
    query_ends: &MxArray,
    block_size: usize,
    heads_per_group: i32,
) -> i32 {
    unsafe {
        sys::mlx_qwen4_qsa_attention_pipeline_threads(
            queries.as_raw_ptr(),
            keys.as_raw_ptr(),
            values.as_raw_ptr(),
            selected.as_raw_ptr(),
            query_ends.as_raw_ptr(),
            block_size as i32,
            heads_per_group,
        )
    }
}

/// GPU gather from paged KV cache pools by absolute slot index (`mlx_qwen4_gather_pages`).
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gather_pages(
    keys: &MxArray,
    values: &MxArray,
    key_scales: &MxArray,
    value_scales: &MxArray,
    slots: &MxArray,
    heads: usize,
    dim: usize,
    block_size: usize,
) -> Result<(MxArray, MxArray)> {
    let (mut ko, mut vo) = (std::ptr::null_mut(), std::ptr::null_mut());
    let ok = unsafe {
        sys::mlx_qwen4_gather_pages(
            keys.as_raw_ptr(),
            values.as_raw_ptr(),
            key_scales.as_raw_ptr(),
            value_scales.as_raw_ptr(),
            slots.as_raw_ptr(),
            heads as i32,
            dim as i32,
            block_size as i32,
            &mut ko,
            &mut vo,
        )
    };
    if !ok {
        return Err(Error::from_reason("Qwen4 native page gather failed"));
    }
    Ok((
        MxArray::from_handle(ko, "Qwen4 gathered keys")?,
        MxArray::from_handle(vo, "Qwen4 gathered values")?,
    ))
}

/// GPU sparse window gather combining paged KV pools and fresh chunk rows (`mlx_qwen4_gather_window`).
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gather_window(
    keys: &MxArray,
    values: &MxArray,
    key_scales: &MxArray,
    value_scales: &MxArray,
    table: &MxArray,
    tokens: &MxArray,
    fresh_k: &MxArray,
    fresh_v: &MxArray,
    base: usize,
    block_size: usize,
) -> Result<(MxArray, MxArray)> {
    let (mut k, mut v) = (std::ptr::null_mut(), std::ptr::null_mut());
    let ok = unsafe {
        sys::mlx_qwen4_gather_window(
            keys.as_raw_ptr(),
            values.as_raw_ptr(),
            key_scales.as_raw_ptr(),
            value_scales.as_raw_ptr(),
            table.as_raw_ptr(),
            tokens.as_raw_ptr(),
            fresh_k.as_raw_ptr(),
            fresh_v.as_raw_ptr(),
            base as i32,
            block_size as i32,
            &mut k,
            &mut v,
        )
    };
    if !ok {
        return Err(Error::from_reason("Qwen4 GPU sparse page gather failed"));
    }
    Ok((
        MxArray::from_handle(k, "sparse page keys")?,
        MxArray::from_handle(v, "sparse page values")?,
    ))
}
