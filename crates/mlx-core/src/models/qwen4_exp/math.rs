use crate::array::{DType, MxArray};
use crate::models::qwen4_exp::runtime_flags;
use crate::nn::Activations;
use napi::{Error, Result};

/// Two opaque launches over the existing K-quant gate/up and affine down banks.
/// The caller owns slot leases; every current bank and mapping is a graph input.
pub(super) fn routed_experts(
    x: &MxArray,
    ids: &MxArray,
    scores: &MxArray,
    banks: &[std::sync::Arc<super::weights::Weight>; 3],
) -> Result<Option<MxArray>> {
    let shape = x.shape()?;
    if runtime_flags::is_zero(c"MLX_QWEN4_FUSED_EXPERTS")
        || runtime_flags::is_zero(c"MLX_QWEN4_DIRECT_GEMV")
        || runtime_flags::is_zero(c"MLX_QWEN4_AFFINE_GEMV")
        || !crate::engine::persistence::compiled_forward_backend_available()
        || x.dtype()? != DType::BFloat16
        || shape.first() != Some(&1)
        || shape.len() != 3
        || !(1..=8).contains(&shape[1])
        || shape[2] != 2560
    {
        return Ok(None);
    }
    let [gate, up, down] = banks;
    if !matches!(gate.mode.as_str(), "q4k" | "q5k")
        || gate.mode != up.mode
        || gate.bits != up.bits
        || down.mode != "affine"
        || !matches!(down.bits, 5 | 8)
        || banks.iter().any(|w| w.group != 32)
    {
        return Ok(None);
    }
    let (
        Some(gate_scales),
        Some(gate_biases),
        Some(up_scales),
        Some(up_biases),
        Some(down_scales),
        Some(down_biases),
    ) = (
        &gate.scales,
        &gate.biases,
        &up.scales,
        &up.biases,
        &down.scales,
        &down.biases,
    )
    else {
        return Ok(None);
    };
    let indices = ids.reshape(&[-1])?;
    let raw = unsafe {
        mlx_sys::mlx_qwen4_routed_experts(
            x.as_raw_ptr(),
            indices.as_raw_ptr(),
            scores.as_raw_ptr(),
            gate.values.as_raw_ptr(),
            gate_scales.as_raw_ptr(),
            gate_biases.as_raw_ptr(),
            up.values.as_raw_ptr(),
            up_scales.as_raw_ptr(),
            up_biases.as_raw_ptr(),
            down.values.as_raw_ptr(),
            down_scales.as_raw_ptr(),
            down_biases.as_raw_ptr(),
        )
    };
    if raw.is_null() {
        Ok(None)
    } else {
        MxArray::from_handle(raw, "Qwen4 fused routed experts").map(Some)
    }
}

/// The shared Q8 branch joins the two routed kernels without changing either
/// branch's GEMV accumulation, activation rounding or final addition order.
pub(super) fn routed_shared_experts(
    x: &MxArray,
    ids: &MxArray,
    scores: &MxArray,
    banks: &[std::sync::Arc<super::weights::Weight>; 3],
    shared: &[std::sync::Arc<super::weights::Weight>; 3],
    shared_gate: &MxArray,
) -> Result<Option<MxArray>> {
    if !crate::engine::persistence::compiled_forward_backend_available()
        || *x.shape()? != [1, 1, 2560]
        || x.dtype()? != DType::BFloat16
        || *shared_gate.shape()? != [1, 1, 1]
        || shared_gate.dtype()? != DType::BFloat16
        || runtime_flags::is_zero(c"MLX_QWEN4_FUSED_EXPERTS")
        || runtime_flags::is_zero(c"MLX_QWEN4_DIRECT_GEMV")
        || runtime_flags::is_zero(c"MLX_QWEN4_AFFINE_GEMV")
        || !matches!(banks[0].mode.as_str(), "q4k" | "q5k")
        || banks[0].mode != banks[1].mode
        || banks[0].bits != banks[1].bits
        || banks[2].mode != "affine"
        || !matches!(banks[2].bits, 5 | 8)
        || shared.iter().any(|w| w.mode != "affine" || w.bits != 8)
        || banks
            .iter()
            .chain(shared)
            .any(|w| w.group != 32 || w.scales.is_none() || w.biases.is_none())
    {
        return Ok(None);
    }
    let ids = ids.reshape(&[-1])?;
    let mut inputs = vec![x.as_raw_ptr(), ids.as_raw_ptr(), scores.as_raw_ptr()];
    for bank in banks.iter().chain(shared) {
        let (Some(scales), Some(biases)) = (&bank.scales, &bank.biases) else {
            return Ok(None);
        };
        inputs.extend([
            bank.values.as_raw_ptr(),
            scales.as_raw_ptr(),
            biases.as_raw_ptr(),
        ]);
    }
    inputs.push(shared_gate.as_raw_ptr());
    let raw = unsafe { mlx_sys::mlx_qwen4_routed_shared_experts(inputs.as_ptr(), inputs.len()) };
    if raw.is_null() {
        return Ok(None);
    }
    if runtime_flags::is_one(c"MLX_QWEN4_TRACE_SHARED_EXPERTS") {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| eprintln!("QWEN4_SHARED_EXPERTS singleton mixed K-quant/Q8 path"));
    }
    MxArray::from_handle(raw, "Qwen4 fused routed/shared experts").map(Some)
}

/// Wide sorted assignments read the original token matrix. Every bank and row
/// map remains an explicit graph input, including mutable partial-residency banks.
pub(super) fn prefill_indirect(
    x: &MxArray,
    ids: &MxArray,
    token_rows: &MxArray,
    banks: &[std::sync::Arc<super::weights::Weight>; 3],
    experts: usize,
) -> Result<Option<MxArray>> {
    let [gate, up, down] = banks;
    if !crate::engine::persistence::compiled_forward_backend_available()
        || !matches!(gate.mode.as_str(), "q4k" | "q5k")
        || gate.mode != up.mode
        || gate.bits != up.bits
        || down.mode != "affine"
        || !matches!(down.bits, 5 | 8)
        || banks.iter().any(|w| w.group != 32)
    {
        return Ok(None);
    }
    let (
        Some(gate_scales),
        Some(gate_biases),
        Some(up_scales),
        Some(up_biases),
        Some(down_scales),
        Some(down_biases),
    ) = (
        &gate.scales,
        &gate.biases,
        &up.scales,
        &up.biases,
        &down.scales,
        &down.biases,
    )
    else {
        return Ok(None);
    };
    let raw = unsafe {
        mlx_sys::mlx_qwen4_prefill_indirect(
            x.as_raw_ptr(),
            ids.as_raw_ptr(),
            token_rows.as_raw_ptr(),
            gate.values.as_raw_ptr(),
            gate_scales.as_raw_ptr(),
            gate_biases.as_raw_ptr(),
            up.values.as_raw_ptr(),
            up_scales.as_raw_ptr(),
            up_biases.as_raw_ptr(),
            down.values.as_raw_ptr(),
            down_scales.as_raw_ptr(),
            down_biases.as_raw_ptr(),
            experts as i32,
        )
    };
    if raw.is_null() {
        Ok(None)
    } else {
        if runtime_flags::is_one(c"MLX_QWEN4_TRACE_PREFILL_INDIRECT") {
            static TRACE: std::sync::Once = std::sync::Once::new();
            TRACE.call_once(|| {
                eprintln!(
                    "QWEN4_PREFILL_INDIRECT tile_rows=32 assignments={} bank_slots={experts}",
                    ids.size().unwrap_or(0)
                )
            });
        }
        MxArray::from_handle(raw, "Qwen4 indirect expert prefill").map(Some)
    }
}

pub(super) fn expert_tiles(ids: &MxArray, experts: usize) -> Result<Option<MxArray>> {
    if !crate::engine::persistence::compiled_forward_backend_available() {
        return Ok(None);
    }
    let raw = unsafe { mlx_sys::mlx_qwen4_expert_tiles(ids.as_raw_ptr(), experts as i32) };
    if raw.is_null() {
        Ok(None)
    } else {
        MxArray::from_handle(raw, "Qwen4 expert tile table").map(Some)
    }
}

pub(super) fn combine_expert_rows(
    values: &MxArray,
    scores: &MxArray,
    inverse: Option<&MxArray>,
    top: usize,
) -> Result<MxArray> {
    let shape = values.shape()?;
    let tokens = shape[0] / top as i64;
    if let Some(inverse) = inverse
        && tokens > 8
        && crate::engine::persistence::compiled_forward_backend_available()
        && !runtime_flags::is_zero(c"MLX_QWEN4_SORTED_COMBINE")
    {
        let raw = unsafe {
            mlx_sys::mlx_qwen4_sorted_combine(
                values.as_raw_ptr(),
                scores.as_raw_ptr(),
                inverse.as_raw_ptr(),
                top as i32,
            )
        };
        if !raw.is_null() {
            return MxArray::from_handle(raw, "Qwen4 sorted expert combine");
        }
    }
    let values = match inverse {
        Some(inverse) => values.take(inverse, 0)?,
        None => values.clone(),
    };
    values
        .reshape(&[1, tokens, top as i64, shape[1]])?
        .mul(&scores.reshape(&[1, tokens, top as i64, 1])?)?
        .sum(Some(&[2]), Some(false))
}

pub(super) fn combine_shared_expert_rows(
    values: &MxArray,
    scores: &MxArray,
    inverse: &MxArray,
    shared: &MxArray,
    gate: &MxArray,
    top: usize,
) -> Result<MxArray> {
    let raw = if crate::engine::persistence::compiled_forward_backend_available()
        && !runtime_flags::is_zero(c"MLX_QWEN4_SORTED_COMBINE")
        && !runtime_flags::is_zero(c"MLX_QWEN4_FUSED_POINTWISE")
    {
        unsafe {
            mlx_sys::mlx_qwen4_sorted_shared_combine(
                values.as_raw_ptr(),
                scores.as_raw_ptr(),
                inverse.as_raw_ptr(),
                shared.as_raw_ptr(),
                gate.as_raw_ptr(),
                top as i32,
            )
        }
    } else {
        std::ptr::null_mut()
    };
    if !raw.is_null() {
        return MxArray::from_handle(raw, "Qwen4 sorted routed and shared experts");
    }
    combine_expert_rows(values, scores, Some(inverse), top)?
        .astype(shared.dtype()?)?
        .add(&sigmoid_mul(gate, shared)?)
}

/// A complete singleton GGUF GDN graph. Histories are explicit inputs and fresh
/// outputs so compiled replay cannot retain a previous layer or cache owner.
#[allow(clippy::too_many_arguments)]
pub(super) fn complete_gdn(
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
        mlx_sys::mlx_qwen4_complete_gdn(
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

/// F32 recurrent state is retained across steps and speculative snapshots.
/// The shared Metal kernel writes a fresh state; it never mutates its input.
pub(super) fn recurrent_step(
    q: &MxArray,
    k: &MxArray,
    v: &MxArray,
    decay: &MxArray,
    beta: &MxArray,
    state: &MxArray,
) -> Result<(MxArray, MxArray)> {
    let shape = q.shape()?;
    let [1, query_heads, 1, kd] = &*shape else {
        return Err(Error::from_reason(
            "Qwen4 recurrent query must have shape [1, heads, 1, key_dim]",
        ));
    };
    if *query_heads <= 0 || *kd <= 0 {
        return Err(Error::from_reason(
            "Qwen4 recurrent query heads and key dimension must be positive",
        ));
    }
    let (query_heads, kd) = (*query_heads, *kd);
    if kd >= 32
        && kd % 32 == 0
        && crate::engine::persistence::compiled_forward_backend_available()
        && !runtime_flags::is_zero(c"MLX_QWEN4_FUSED_GDN")
    {
        let q = q.reshape(&[1, 1, query_heads, kd])?;
        let k = k.reshape(&[1, 1, query_heads, kd])?;
        let heads = state.shape_at(1)?;
        let v = v.reshape(&[1, 1, heads, -1])?;
        let decay = decay.reshape(&[1, 1, heads])?;
        let beta = beta.reshape(&[1, 1, heads])?;
        let mut out = std::ptr::null_mut();
        let mut next = std::ptr::null_mut();
        let ok = unsafe {
            mlx_sys::mlx_qwen4_gated_delta_kernel(
                q.as_raw_ptr(),
                k.as_raw_ptr(),
                v.as_raw_ptr(),
                decay.as_raw_ptr(),
                beta.as_raw_ptr(),
                state.as_raw_ptr(),
                std::ptr::null_mut(),
                &mut out,
                &mut next,
            )
        };
        if !ok {
            return Err(Error::from_reason("Qwen4 fused recurrent kernel failed"));
        }
        return Ok((
            MxArray::from_handle(out, "Qwen4 recurrent output")?,
            MxArray::from_handle(next, "Qwen4 recurrent state")?,
        ));
    }
    let repeats = state.shape_at(1)? / query_heads;
    recurrent_step_reference(
        &MxArray::tile(q, &[1, repeats as i32, 1, 1])?,
        &MxArray::tile(k, &[1, repeats as i32, 1, 1])?,
        v,
        decay,
        beta,
        state,
    )
}

pub(super) fn recurrent_step_reference(
    q: &MxArray,
    k: &MxArray,
    v: &MxArray,
    decay: &MxArray,
    beta: &MxArray,
    state: &MxArray,
) -> Result<(MxArray, MxArray)> {
    let state = state.mul(decay)?;
    let memory = state.mul(k)?.sum(Some(&[-1]), Some(false))?;
    let delta = v.sub(&memory)?.mul(beta)?;
    let state = state.add(&delta.expand_dims(-1)?.mul(k)?)?;
    let out = state
        .mul(q)?
        .sum(Some(&[-1]), Some(false))?
        .expand_dims(1)?;
    Ok((out, state))
}

pub(super) fn recurrent_sequence(
    q: &MxArray,
    k: &MxArray,
    v: &MxArray,
    decay: &MxArray,
    beta: &MxArray,
    state: &MxArray,
) -> Result<(MxArray, MxArray)> {
    let shape = q.shape()?;
    let (tokens, heads, kd) = (shape[1], shape[2], shape[3]);
    if kd >= 32
        && kd % 32 == 0
        && crate::engine::persistence::compiled_forward_backend_available()
        && !runtime_flags::is_zero(c"MLX_QWEN4_FUSED_GDN")
    {
        let mut out = std::ptr::null_mut();
        let mut next = std::ptr::null_mut();
        let ok = unsafe {
            mlx_sys::mlx_qwen4_gated_delta_kernel(
                q.as_raw_ptr(),
                k.as_raw_ptr(),
                v.as_raw_ptr(),
                decay.as_raw_ptr(),
                beta.as_raw_ptr(),
                state.as_raw_ptr(),
                std::ptr::null_mut(),
                &mut out,
                &mut next,
            )
        };
        if !ok {
            return Err(Error::from_reason("Qwen4 fused recurrent sequence failed"));
        }
        return Ok((
            MxArray::from_handle(out, "Qwen4 recurrent sequence")?,
            MxArray::from_handle(next, "Qwen4 recurrent sequence state")?,
        ));
    }
    let repeats = state.shape()?[1] / heads;
    let q = MxArray::tile(q, &[1, 1, repeats as i32, 1])?;
    let k = MxArray::tile(k, &[1, 1, repeats as i32, 1])?;
    let heads = state.shape()?[1];
    let mut next = state.clone();
    let mut outputs = Vec::with_capacity(tokens as usize);
    for t in 0..tokens {
        let (out, s) = recurrent_step_reference(
            &q.slice_axis(1, t, t + 1)?.reshape(&[1, heads, 1, kd])?,
            &k.slice_axis(1, t, t + 1)?.reshape(&[1, heads, 1, kd])?,
            &v.slice_axis(1, t, t + 1)?.reshape(&[1, heads, -1])?,
            &decay.slice_axis(1, t, t + 1)?.reshape(&[1, heads, 1, 1])?,
            &beta.slice_axis(1, t, t + 1)?.reshape(&[1, heads, 1])?,
            &next,
        )?;
        outputs.push(out);
        next = s;
    }
    Ok((
        MxArray::concatenate_many(outputs.iter().collect(), Some(1))?,
        next,
    ))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn gdn_prepare(
    qkv: &MxArray,
    a: &MxArray,
    b: &MxArray,
    conv: &MxArray,
    state: &mut Option<MxArray>,
    scale: &MxArray,
    dt: &MxArray,
) -> Result<Option<[MxArray; 5]>> {
    if !crate::engine::persistence::compiled_forward_backend_available() {
        return Ok(None);
    }
    let old = match state {
        Some(s) => s.clone(),
        None => MxArray::zeros(&[3, 10240], Some(DType::BFloat16))?,
    };
    let mut outputs = [std::ptr::null_mut(); 6];
    if !unsafe {
        // qwen4 semantics: L2-style norm (eps on the sum) and f32 beta —
        // matches this family's `math::l2` reference exactly.
        mlx_sys::mlx_qwen4_gdn_prepare(
            qkv.as_raw_ptr(),
            a.as_raw_ptr(),
            b.as_raw_ptr(),
            conv.as_raw_ptr(),
            old.as_raw_ptr(),
            scale.as_raw_ptr(),
            dt.as_raw_ptr(),
            false,
            false,
            outputs.as_mut_ptr(),
        )
    } {
        return Ok(None);
    }
    let [q, k, v, decay, beta, history] = outputs;
    let values = [
        MxArray::from_handle(q, "GDN prepared Q")?,
        MxArray::from_handle(k, "GDN prepared K")?,
        MxArray::from_handle(v, "GDN prepared V")?,
        MxArray::from_handle(decay, "GDN prepared decay")?,
        MxArray::from_handle(beta, "GDN prepared beta")?,
    ];
    *state = Some(MxArray::from_handle(history, "GDN prepared history")?);
    Ok(Some(values))
}

pub(super) fn conv_sequence(
    x: &MxArray,
    weight: &MxArray,
    state: &mut Option<MxArray>,
    kernel: usize,
) -> Result<MxArray> {
    if kernel == 4
        && x.dtype()? == DType::BFloat16
        && weight.dtype()? == DType::Float32
        && crate::engine::persistence::compiled_forward_backend_available()
        && !runtime_flags::is_zero(c"MLX_QWEN4_WINDOW_CONV")
    {
        let old = match state {
            Some(s) => s.clone(),
            None => MxArray::zeros(&[3, x.shape_at(2)?], Some(x.dtype()?))?,
        };
        let mut out = std::ptr::null_mut();
        let mut history = std::ptr::null_mut();
        let ok = unsafe {
            mlx_sys::mlx_qwen4_window_conv(
                x.as_raw_ptr(),
                old.as_raw_ptr(),
                weight.as_raw_ptr(),
                &mut out,
                &mut history,
            )
        };
        if ok {
            let out = MxArray::from_handle(out, "Qwen4 window convolution")?;
            let history = MxArray::from_handle(history, "Qwen4 convolution history")?;
            MxArray::eval_arrays_with_context(&[&history], "qwen4::batch::conv_state")?;
            *state = Some(history);
            return Ok(out);
        }
    }
    conv_sequence_reference(x, weight, state, kernel)
}

fn conv_sequence_reference(
    x: &MxArray,
    weight: &MxArray,
    state: &mut Option<MxArray>,
    kernel: usize,
) -> Result<MxArray> {
    let shape = x.shape()?;
    let (tokens, width) = (shape[1], shape[2]);
    let keep = kernel as i64 - 1;
    let old = match state {
        Some(s) => s.clone(),
        None => MxArray::zeros(&[keep, width], Some(x.dtype()?))?,
    };
    let input = MxArray::concatenate(&old, &x.reshape(&[tokens, width])?, 0)?;
    let indices = MxArray::from_int32(
        &(0..tokens)
            .flat_map(|t| (0..kernel).map(move |j| t as i32 + j as i32))
            .collect::<Vec<_>>(),
        &[tokens * kernel as i64],
    )?;
    let selected = input
        .take(&indices, 0)?
        .reshape(&[tokens, kernel as i64, width])?
        .astype(DType::Float32)?;
    let weight = weight
        .reshape(&[width, kernel as i64])?
        .transpose(None)?
        .astype(DType::Float32)?;
    let out = selected
        .mul(&weight)?
        .sum(Some(&[1]), Some(false))?
        .astype(x.dtype()?)?
        .reshape(&[1, tokens, width])?;
    // A tiny history must not keep the entire prefill allocation alive.
    let next = input.slice_axis(0, tokens, tokens + keep)?.deep_copy()?;
    MxArray::eval_arrays_with_context(&[&next], "qwen4::batch::conv_state")?;
    *state = Some(next);
    Activations::silu(&out)
}

pub(super) fn gdn_epilogue(
    out: &MxArray,
    z: &MxArray,
    weight: &MxArray,
    eps: f64,
) -> Result<Option<MxArray>> {
    if !crate::engine::persistence::compiled_forward_backend_available() {
        return Ok(None);
    }
    let handle = unsafe {
        mlx_sys::mlx_qwen4_gdn_epilogue(out.as_raw_ptr(), z.as_raw_ptr(), weight.as_raw_ptr(), eps)
    };
    if handle.is_null() {
        return Ok(None);
    }
    MxArray::from_handle(handle, "Qwen4 prefill GDN epilogue").map(Some)
}

pub fn norm(x: &MxArray, w: &MxArray, group: usize, eps: f64, centered: bool) -> Result<MxArray> {
    if !runtime_flags::is_zero(c"MLX_QWEN4_FUSED_POINTWISE")
        && crate::engine::persistence::compiled_forward_backend_available()
    {
        let out = unsafe {
            mlx_sys::mlx_qwen4_norm(x.as_raw_ptr(), w.as_raw_ptr(), group as i32, eps, centered)
        };
        return MxArray::from_handle(out, "Qwen4 compiled normalization");
    }
    let shape = x.shape()?;
    let y = x.astype(DType::Float32)?.reshape(&[-1, group as i64])?;
    let y = y
        .div(
            &y.square()?
                .mean(Some(&[-1]), Some(true))?
                .add_scalar(eps)?
                .sqrt()?,
        )?
        .reshape(&shape)?;
    let w = w.astype(DType::Float32)?;
    let w = if centered { w.add_scalar(1.0)? } else { w };
    y.mul(&w)?.astype(x.dtype()?)
}

/// Split-half partial RoPE. Text positions coincide on the three MRoPE axes.
pub fn rope(x: &MxArray, position: usize, dims: usize, theta: f64) -> Result<MxArray> {
    mrope(x, [position as i64; 3], dims, theta, [0; 3], true)
}

pub fn mrope(
    x: &MxArray,
    positions: [i64; 3],
    dims: usize,
    theta: f64,
    sections: [usize; 3],
    interleaved: bool,
) -> Result<MxArray> {
    let shape = x.shape()?;
    let (&width, axes) = shape
        .split_last()
        .ok_or_else(|| Error::from_reason("Qwen4 rotary input must have a feature dimension"))?;
    let axis = axes.len();
    let half = dims as i64 / 2;
    let angles = rotary_angles(positions, dims, theta, sections, interleaved);
    let angles = MxArray::from_float32(&angles, &[half])?;
    let cos = angles.cos()?.astype(x.dtype()?)?;
    let sin = angles.sin()?.astype(x.dtype()?)?;
    let a = x.slice_axis(axis, 0, half)?;
    let b = x.slice_axis(axis, half, dims as i64)?;
    let left = a.mul(&cos)?.sub(&b.mul(&sin)?)?;
    let right = b.mul(&cos)?.add(&a.mul(&sin)?)?;
    let tail = x.slice_axis(axis, dims as i64, width)?;
    MxArray::concatenate_many(vec![&left, &right, &tail], Some(-1))
}

fn rotary_angles(
    positions: [i64; 3],
    dims: usize,
    theta: f64,
    sections: [usize; 3],
    interleaved: bool,
) -> Vec<f32> {
    let half = dims as i64 / 2;
    let angles: Vec<f32> = (0..half)
        .map(|i| {
            let j = i as usize;
            let axis = if interleaved {
                if j % 3 == 1 && j < sections[1] * 3 {
                    1
                } else if j % 3 == 2 && j < sections[2] * 3 {
                    2
                } else {
                    0
                }
            } else if j < sections[0] {
                0
            } else if j < sections[0] + sections[1] {
                1
            } else {
                2
            };
            (positions[axis] as f64 / theta.powf((2 * i) as f64 / dims as f64)) as f32
        })
        .collect();
    angles
}

/// Position tables for a complete [B,H,T,D] window, preserving the singleton
/// angle computation and intermediate dtype. Media axes may differ per token.
pub(super) fn mrope_window(
    x: &MxArray,
    positions: &[[i64; 3]],
    dims: usize,
    theta: f64,
    sections: [usize; 3],
    interleaved: bool,
) -> Result<MxArray> {
    let (cos, sin) = rotary_tables(positions, dims, theta, sections, interleaved, x.dtype()?)?;
    apply_rotary_window(x, dims, &cos, &sin)
}

#[derive(PartialEq)]
struct RotaryKey {
    positions: Vec<[i64; 3]>,
    dims: usize,
    theta: u64,
    sections: [usize; 3],
    interleaved: bool,
    dtype: DType,
}

/// At most the attention and compressed-indexer tables for one admitted window.
/// These are temporary graph inputs, never part of an owner's persistent state.
#[derive(Default)]
pub(super) struct RotaryWindowCache(Vec<(RotaryKey, MxArray, MxArray)>);

impl RotaryWindowCache {
    pub fn clear(&mut self) {
        self.0.clear();
    }

    /// Reuse the same forward's tables for singleton attention and indexer
    /// heads. The original shape can be [H,D] or [1,H,1,D].
    #[allow(clippy::too_many_arguments)]
    pub fn apply_singleton(
        &mut self,
        x: &MxArray,
        position: [i64; 3],
        dims: usize,
        theta: f64,
        sections: [usize; 3],
        interleaved: bool,
    ) -> Result<MxArray> {
        let shape = x.shape()?;
        let width = shape.last().copied().ok_or_else(|| {
            Error::from_reason("Qwen4 cached rotary input must have a feature dimension")
        })?;
        self.apply(
            &x.reshape(&[1, -1, 1, width])?,
            vec![position],
            dims,
            theta,
            sections,
            interleaved,
        )?
        .reshape(&shape)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn apply(
        &mut self,
        x: &MxArray,
        positions: Vec<[i64; 3]>,
        dims: usize,
        theta: f64,
        sections: [usize; 3],
        interleaved: bool,
    ) -> Result<MxArray> {
        let key = RotaryKey {
            positions,
            dims,
            theta: theta.to_bits(),
            sections,
            interleaved,
            dtype: x.dtype()?,
        };
        let (cos, sin) = self.tables(key)?;
        apply_rotary_window(x, dims, &cos, &sin)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn apply_normalized(
        &mut self,
        x: &MxArray,
        weight: &MxArray,
        positions: Vec<[i64; 3]>,
        dims: usize,
        theta: f64,
        sections: [usize; 3],
        interleaved: bool,
        eps: f64,
    ) -> Result<Option<MxArray>> {
        let key = RotaryKey {
            positions,
            dims,
            theta: theta.to_bits(),
            sections,
            interleaved,
            dtype: x.dtype()?,
        };
        let (cos, sin) = self.tables(key)?;
        fused_attention_norm_rotary(x, weight, &cos, &sin, eps)
    }

    fn tables(&mut self, key: RotaryKey) -> Result<(MxArray, MxArray)> {
        if let Some((_, cos, sin)) = self.0.iter().find(|(old, _, _)| *old == key) {
            return Ok((cos.clone(), sin.clone()));
        }
        let (cos, sin) = rotary_tables(
            &key.positions,
            key.dims,
            f64::from_bits(key.theta),
            key.sections,
            key.interleaved,
            key.dtype,
        )?;
        if self.0.len() == 2 {
            self.0.remove(0);
        }
        self.0.push((key, cos.clone(), sin.clone()));
        Ok((cos, sin))
    }
}

fn rotary_tables(
    positions: &[[i64; 3]],
    dims: usize,
    theta: f64,
    sections: [usize; 3],
    interleaved: bool,
    dtype: DType,
) -> Result<(MxArray, MxArray)> {
    let half = dims as i64 / 2;
    let angles: Vec<f32> = positions
        .iter()
        .flat_map(|&p| rotary_angles(p, dims, theta, sections, interleaved))
        .collect();
    let angles = MxArray::from_float32(&angles, &[1, 1, positions.len() as i64, half])?;
    Ok((angles.cos()?.astype(dtype)?, angles.sin()?.astype(dtype)?))
}

fn fused_attention_norm_rotary(
    x: &MxArray,
    weight: &MxArray,
    cos: &MxArray,
    sin: &MxArray,
    eps: f64,
) -> Result<Option<MxArray>> {
    if !crate::engine::persistence::compiled_forward_backend_available() {
        return Ok(None);
    }
    let raw = unsafe {
        mlx_sys::mlx_qwen4_attention_norm_rotary(
            x.as_raw_ptr(),
            weight.as_raw_ptr(),
            cos.as_raw_ptr(),
            sin.as_raw_ptr(),
            eps,
        )
    };
    if raw.is_null() {
        Ok(None)
    } else {
        MxArray::from_handle(raw, "Qwen4 attention normalization and rotary").map(Some)
    }
}

fn fused_rotary_window(x: &MxArray, cos: &MxArray, sin: &MxArray) -> Result<Option<MxArray>> {
    if !crate::engine::persistence::compiled_forward_backend_available() {
        return Ok(None);
    }
    let raw = unsafe {
        mlx_sys::mlx_qwen4_rotary_window(x.as_raw_ptr(), cos.as_raw_ptr(), sin.as_raw_ptr())
    };
    if raw.is_null() {
        Ok(None)
    } else {
        MxArray::from_handle(raw, "Qwen4 fused rotary window").map(Some)
    }
}

fn apply_rotary_window(x: &MxArray, dims: usize, cos: &MxArray, sin: &MxArray) -> Result<MxArray> {
    let shape = x.shape()?;
    if shape.len() != 4
        || shape[2] != cos.shape()?[2]
        || !dims.is_multiple_of(2)
        || dims as i64 > shape[3]
    {
        return Err(Error::from_reason("Qwen4 rotary window shape mismatch"));
    }
    if runtime_flags::is_one(c"MLX_QWEN4_FUSED_ROTARY")
        && let Some(out) = fused_rotary_window(x, cos, sin)?
    {
        return Ok(out);
    }
    let half = dims as i64 / 2;
    let a = x.slice_axis(3, 0, half)?;
    let b = x.slice_axis(3, half, dims as i64)?;
    let left = a.mul(cos)?.sub(&b.mul(sin)?)?;
    let right = b.mul(cos)?.add(&a.mul(sin)?)?;
    let tail = x.slice_axis(3, dims as i64, shape[3])?;
    MxArray::concatenate_many(vec![&left, &right, &tail], Some(-1))
}

pub fn l2(x: &MxArray) -> Result<MxArray> {
    let y = x.astype(DType::Float32)?;
    y.div(
        &y.square()?
            .sum(Some(&[-1]), Some(true))?
            .add_scalar(1e-6)?
            .sqrt()?,
    )?
    .astype(x.dtype()?)
}

#[cfg(test)]
mod input_validation_tests {
    use super::*;

    #[test]
    fn scalar_inputs_return_shape_errors_without_changing_convolution_history() {
        let scalar = MxArray::from_float32(&[1.], &[]).unwrap();
        let rotary_error = mrope(&scalar, [0; 3], 2, 10_000., [0; 3], true)
            .err()
            .expect("scalar rotary input must be rejected");
        assert_eq!(
            rotary_error.reason,
            "Qwen4 rotary input must have a feature dimension"
        );
        let mut tables = RotaryWindowCache::default();
        let cached_error = tables
            .apply_singleton(&scalar, [0; 3], 2, 10_000., [0; 3], true)
            .err()
            .expect("scalar cached rotary input must be rejected");
        assert_eq!(
            cached_error.reason,
            "Qwen4 cached rotary input must have a feature dimension"
        );
        assert!(tables.0.is_empty());

        let recurrent_error = recurrent_step(&scalar, &scalar, &scalar, &scalar, &scalar, &scalar)
            .err()
            .expect("scalar recurrent input must be rejected");
        assert_eq!(
            recurrent_error.reason,
            "Qwen4 recurrent query must have shape [1, heads, 1, key_dim]"
        );

        let mut history = Some(MxArray::from_float32(&[2., 3.], &[2, 1]).unwrap());
        let convolution_error = conv(&scalar, &scalar, &mut history, 3, 1)
            .err()
            .expect("scalar convolution input must be rejected");
        assert_eq!(
            convolution_error.reason,
            "Qwen4 convolution input must have a feature dimension"
        );
        assert_eq!(&*history.as_ref().unwrap().to_float32().unwrap(), &[2., 3.]);
    }
}

#[cfg(test)]
mod compact_recurrence_tests {
    use super::*;

    fn input(shape: &[i64], phase: f32) -> MxArray {
        let n: i64 = shape.iter().product();
        let values: Vec<_> = (0..n)
            .map(|i| (i as f32 * 0.017 + phase).sin() * 0.03)
            .collect();
        MxArray::from_float32(&values, shape)
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap()
    }

    #[test]
    fn prepared_gdn_matches_separate_ops_across_short_and_wide_history() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        let conv = input(&[10240, 4], 0.2).astype(DType::Float32).unwrap();
        let scale = input(&[48], 0.8)
            .astype(DType::Float32)
            .unwrap()
            .sub_scalar(0.5)
            .unwrap();
        let dt = input(&[48], 1.3).astype(DType::Float32).unwrap();
        let mut expected_state = Some(input(&[3, 10240], 2.4));
        let mut actual_state = expected_state.clone();
        let mut compact_state = expected_state.clone();
        for tokens in [1, 2, 7, 1024, 3] {
            let x = input(&[1, tokens, 10240], 0.1);
            let a = input(&[1, tokens, 48], 0.7).astype(DType::Float32).unwrap();
            let b = input(&[1, tokens, 48], 1.1).astype(DType::Float32).unwrap();
            let y = conv_sequence(&x, &conv, &mut expected_state, 4).unwrap();
            let q = l2(&y
                .slice_axis(2, 0, 2048)
                .unwrap()
                .reshape(&[1, tokens, 16, 128])
                .unwrap())
            .unwrap()
            .mul_scalar(128f64.powf(-0.5))
            .unwrap();
            let k = l2(&y
                .slice_axis(2, 2048, 4096)
                .unwrap()
                .reshape(&[1, tokens, 16, 128])
                .unwrap())
            .unwrap();
            let v = y
                .slice_axis(2, 4096, 10240)
                .unwrap()
                .reshape(&[1, tokens, 48, 128])
                .unwrap();
            let (decay, beta) = gdn_gates(&a, &b, &scale, &dt).unwrap();
            let actual = gdn_prepare(&x, &a, &b, &conv, &mut actual_state, &scale, &dt)
                .unwrap()
                .unwrap();
            let compact = gdn_prepare(
                &x,
                &a.astype(DType::BFloat16).unwrap(),
                &b.astype(DType::BFloat16).unwrap(),
                &conv,
                &mut compact_state,
                &scale,
                &dt,
            )
            .unwrap()
            .unwrap();
            for (field, (want, got)) in actual.iter().zip(&compact).enumerate() {
                assert_eq!(want.dtype().unwrap(), got.dtype().unwrap());
                assert_eq!(
                    want.to_float32().unwrap().to_vec(),
                    got.to_float32().unwrap().to_vec(),
                    "compact gate preparation field={field}, tokens={tokens}"
                );
            }
            assert_eq!(
                actual_state
                    .as_ref()
                    .unwrap()
                    .to_float32()
                    .unwrap()
                    .to_vec(),
                compact_state
                    .as_ref()
                    .unwrap()
                    .to_float32()
                    .unwrap()
                    .to_vec()
            );
            for (field, (want, got)) in [q, k, v, decay, beta].iter().zip(&actual).enumerate() {
                assert_eq!(want.dtype().unwrap(), got.dtype().unwrap());
                assert_eq!(
                    &*want.astype(DType::Float32).unwrap().to_float32().unwrap(),
                    &*got.astype(DType::Float32).unwrap().to_float32().unwrap(),
                    "GDN preparation field={field}, tokens={tokens}"
                );
            }
            assert_eq!(
                &*expected_state
                    .as_ref()
                    .unwrap()
                    .astype(DType::Float32)
                    .unwrap()
                    .to_float32()
                    .unwrap(),
                &*actual_state
                    .as_ref()
                    .unwrap()
                    .astype(DType::Float32)
                    .unwrap()
                    .to_float32()
                    .unwrap()
            );
        }
    }

    #[test]
    fn bf16_inputs_preserve_fp32_recurrence_and_continuation() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        // Compact GGUF head modulo, expanded heads, and the smaller 2-row
        // dispatch all retain the same arithmetic and FP32 output/state.
        for (kh, nh, d) in [(16, 48, 128), (48, 48, 128), (2, 4, 32)] {
            let mut expected = input(&[1, nh, d, d], 0.3).astype(DType::Float32).unwrap();
            let mut actual = expected.clone();
            for tokens in [7, 1024, 1] {
                let q = input(&[1, tokens, kh, d], 0.1);
                let k = input(&[1, tokens, kh, d], 0.7);
                let v = input(&[1, tokens, nh, d], 1.3);
                let decay = input(&[1, tokens, nh], 2.1)
                    .astype(DType::Float32)
                    .unwrap()
                    .add_scalar(0.95)
                    .unwrap();
                let beta = input(&[1, tokens, nh], 3.4)
                    .astype(DType::Float32)
                    .unwrap()
                    .add_scalar(0.5)
                    .unwrap();
                let (want, next) = recurrent_sequence(
                    &q.astype(DType::Float32).unwrap(),
                    &k.astype(DType::Float32).unwrap(),
                    &v.astype(DType::Float32).unwrap(),
                    &decay,
                    &beta,
                    &expected,
                )
                .unwrap();
                let (got, state) = recurrent_sequence(&q, &k, &v, &decay, &beta, &actual).unwrap();
                assert_eq!(got.dtype().unwrap(), DType::Float32);
                assert_eq!(state.dtype().unwrap(), DType::Float32);
                assert_eq!(
                    &*got.to_float32().unwrap(),
                    &*want.to_float32().unwrap(),
                    "output kh={kh}, nh={nh}, d={d}, tokens={tokens}"
                );
                assert_eq!(
                    &*state.to_float32().unwrap(),
                    &*next.to_float32().unwrap(),
                    "state kh={kh}, nh={nh}, d={d}, tokens={tokens}"
                );
                expected = next;
                actual = state;
            }
        }
    }
}

#[cfg(test)]
mod rotary_window_tests {
    use super::*;

    #[test]
    fn attention_norm_rotary_matches_separate_gguf_ops_and_changing_inputs() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        let mut cache = RotaryWindowCache::default();
        for seed in [1., 7., 37., 113.] {
            for count in [1_i64, 8, 1024] {
                for heads in [1_i64, 2, 8, 24] {
                    let data: Vec<_> = (0..heads * (count + 1) * 256)
                        .map(|i| (i as f32 * 0.013 * seed).sin() * 3.1)
                        .collect();
                    let x = MxArray::from_float32(&data, &[1, heads, count + 1, 256])
                        .unwrap()
                        .astype(DType::BFloat16)
                        .unwrap()
                        .slice_axis(2, 1, count + 1)
                        .unwrap();
                    let x = if heads == 2 || heads == 24 {
                        // Real Q/gate layout: gate lanes separate head rows,
                        // and token/head axes are transposed before rotation.
                        let data: Vec<_> = (0..(count + 1) * heads * 512)
                            .map(|i| (i as f32 * 0.013 * seed).sin() * 3.1)
                            .collect();
                        MxArray::from_float32(&data, &[1, count + 1, heads, 512])
                            .unwrap()
                            .astype(DType::BFloat16)
                            .unwrap()
                            .slice_axis(1, 1, count + 1)
                            .unwrap()
                            .slice_axis(3, 0, 256)
                            .unwrap()
                            .transpose(Some(&[0, 2, 1, 3]))
                            .unwrap()
                    } else {
                        x
                    };
                    let weights: Vec<_> = (0..256)
                        .map(|i| (i as f32 * 0.071 * seed).sin() * 0.35 + 1.)
                        .collect();
                    let w = MxArray::from_float32(&weights, &[256]).unwrap();
                    let positions: Vec<_> = (0..count)
                        .map(|i| [1023 + i, 11 + i / 7, i / 3 - 5])
                        .collect();
                    for (dims, interleaved) in [(64, true), (256, false)] {
                        let n = norm(&x, &w, 256, 1e-6, false).unwrap();
                        let expected = mrope_window(
                            &n,
                            &positions,
                            dims,
                            10_000_000.,
                            [16, 8, 8],
                            interleaved,
                        )
                        .unwrap();
                        let got = cache
                            .apply_normalized(
                                &x,
                                &w,
                                positions.clone(),
                                dims,
                                10_000_000.,
                                [16, 8, 8],
                                interleaved,
                                1e-6,
                            )
                            .unwrap()
                            .unwrap();
                        assert_eq!(
                            &*got.astype(DType::Float32).unwrap().to_float32().unwrap(),
                            &*expected
                                .astype(DType::Float32)
                                .unwrap()
                                .to_float32()
                                .unwrap(),
                            "seed={seed} count={count} heads={heads} dims={dims}",
                        );
                        assert!(cache.0.len() <= 2);
                    }
                }
            }
        }
        let x = MxArray::from_float32(&[1.; 256], &[1, 1, 1, 256]).unwrap();
        let w = MxArray::from_float32(&[1.; 256], &[256]).unwrap();
        let (cos, sin) =
            rotary_tables(&[[0; 3]], 64, 10_000_000., [0; 3], true, DType::BFloat16).unwrap();
        assert!(
            fused_attention_norm_rotary(&x, &w, &cos, &sin, 1e-6)
                .unwrap()
                .is_none()
        );
        let x = x.astype(DType::BFloat16).unwrap();
        assert!(
            fused_attention_norm_rotary(&x, &w, &cos, &sin, 0.)
                .unwrap()
                .is_none()
        );
        assert!(
            fused_attention_norm_rotary(&x, &w.astype(DType::BFloat16).unwrap(), &cos, &sin, 1e-6)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn singleton_tables_preserve_head_views_media_axes_and_positions() {
        let mut cache = RotaryWindowCache::default();
        for dtype in [DType::BFloat16, DType::Float32] {
            for interleaved in [true, false] {
                for position in [[0, 0, 0], [1023, 89, -4], [2048, 2048, 2048]] {
                    for shape in [vec![1, 256], vec![4, 128], vec![1, 8, 1, 256]] {
                        let n: i64 = shape.iter().product();
                        let values: Vec<_> = (0..n + shape.last().unwrap())
                            .map(|i| (i as f32 * 0.13).sin() * 4.1)
                            .collect();
                        let x = MxArray::from_float32(&values, &[values.len() as i64])
                            .unwrap()
                            .slice_axis(0, *shape.last().unwrap(), values.len() as i64)
                            .unwrap()
                            .reshape(&shape)
                            .unwrap()
                            .astype(dtype)
                            .unwrap();
                        let expected =
                            mrope(&x, position, 64, 10_000_000., [16, 8, 8], interleaved).unwrap();
                        for _ in 0..2 {
                            let got = cache
                                .apply_singleton(
                                    &x,
                                    position,
                                    64,
                                    10_000_000.,
                                    [16, 8, 8],
                                    interleaved,
                                )
                                .unwrap();
                            assert_eq!(&*got.shape().unwrap(), shape.as_slice());
                            assert_eq!(
                                &*got.astype(DType::Float32).unwrap().to_float32().unwrap(),
                                &*expected
                                    .astype(DType::Float32)
                                    .unwrap()
                                    .to_float32()
                                    .unwrap(),
                                "{dtype:?}, {interleaved}, {position:?}, {shape:?}"
                            );
                        }
                        assert!(cache.0.len() <= 2);
                    }
                }
            }
        }
        cache.clear();
        assert!(cache.0.is_empty());
    }

    #[test]
    fn shared_tables_preserve_scalar_rounding_and_media_axes() {
        let mut cache = RotaryWindowCache::default();
        for dtype in [DType::BFloat16, DType::Float32] {
            for interleaved in [true, false] {
                for (count, stride, base) in [(1, 1, 1023), (7, 4, 1), (256, 4, 2045), (1024, 1, 0)]
                {
                    let positions: Vec<_> = (0..count)
                        .map(|i| {
                            let p = base + i * stride;
                            [p, p / 7 + 11, p / 3 - 5]
                        })
                        .collect();
                    for width in [128, 256] {
                        let data: Vec<_> = (0..count * width)
                            .map(|i| (i as f32 * 0.17).sin() * 7.1)
                            .collect();
                        let x = MxArray::from_float32(&data, &[1, 1, count, width])
                            .unwrap()
                            .astype(dtype)
                            .unwrap();
                        let got = cache
                            .apply(
                                &x,
                                positions.clone(),
                                64,
                                10_000_000.,
                                [16, 8, 8],
                                interleaved,
                            )
                            .unwrap();
                        let rows: Vec<_> = positions
                            .iter()
                            .enumerate()
                            .map(|(i, &p)| {
                                mrope(
                                    &x.slice_axis(2, i as i64, i as i64 + 1).unwrap(),
                                    p,
                                    64,
                                    10_000_000.,
                                    [16, 8, 8],
                                    interleaved,
                                )
                                .unwrap()
                            })
                            .collect();
                        let expected =
                            MxArray::concatenate_many(rows.iter().collect(), Some(2)).unwrap();
                        let (cos, sin) = rotary_tables(
                            &positions,
                            64,
                            10_000_000.,
                            [16, 8, 8],
                            interleaved,
                            dtype,
                        )
                        .unwrap();
                        if dtype == DType::BFloat16
                            && crate::engine::persistence::compiled_forward_backend_available()
                        {
                            let fused = fused_rotary_window(&x, &cos, &sin).unwrap().unwrap();
                            assert_eq!(
                                &*fused.astype(DType::Float32).unwrap().to_float32().unwrap(),
                                &*expected
                                    .astype(DType::Float32)
                                    .unwrap()
                                    .to_float32()
                                    .unwrap(),
                                "fused rotary {interleaved}, {count}, {width}"
                            );
                        }
                        assert_eq!(
                            &*got.astype(DType::Float32).unwrap().to_float32().unwrap(),
                            &*expected
                                .astype(DType::Float32)
                                .unwrap()
                                .to_float32()
                                .unwrap(),
                            "{dtype:?}, {interleaved}, {count}, {width}"
                        );
                        assert!(cache.0.len() <= 2);
                    }
                }
            }
        }
        cache.clear();
        assert!(cache.0.is_empty());
    }
}

/// Stateful depthwise causal convolution. Complete the owned finite history
/// before returning the lazy activation to its caller.
pub fn conv(
    x: &MxArray,
    weight: &MxArray,
    state: &mut Option<MxArray>,
    kernel: usize,
    dilation: usize,
) -> Result<MxArray> {
    conv_with_completion(x, weight, state, kernel, dilation, false)
}

fn conv_with_completion(
    x: &MxArray,
    weight: &MxArray,
    state: &mut Option<MxArray>,
    kernel: usize,
    dilation: usize,
    defer_state: bool,
) -> Result<MxArray> {
    let width = x.shape()?.last().copied().ok_or_else(|| {
        Error::from_reason("Qwen4 convolution input must have a feature dimension")
    })?;
    let keep = (kernel - 1) * dilation;
    let old = match state {
        Some(s) => s.clone(),
        None => MxArray::zeros(&[keep as i64, width], Some(x.dtype()?))?,
    };
    let input = MxArray::concatenate(&old, &x.reshape(&[1, width])?, 0)?;
    let indices = MxArray::from_int32(
        &(0..kernel)
            .map(|i| (i * dilation) as i32)
            .collect::<Vec<_>>(),
        &[kernel as i64],
    )?;
    let selected = input.take(&indices, 0)?.astype(DType::Float32)?;
    let weights = weight
        .reshape(&[width, kernel as i64])?
        .transpose(None)?
        .astype(DType::Float32)?;
    let output = selected
        .mul(&weights)?
        .sum(Some(&[0]), Some(false))?
        .astype(x.dtype()?)?
        .reshape(&[1, 1, width])?;
    let next = input.slice_axis(0, 1, keep as i64 + 1)?.deep_copy()?;
    if !defer_state {
        MxArray::eval_arrays_with_context(&[&next], "qwen4::math::next")?;
    }
    *state = Some(next);
    Activations::silu(&output)
}

/// Dilated causal convolution over a prompt window. Keep the same tap order
/// and F32 products as the singleton operation, with an owned finite tail.
pub(super) fn conv_window(
    x: &MxArray,
    weight: &MxArray,
    state: &mut Option<MxArray>,
    kernel: usize,
    dilation: usize,
) -> Result<MxArray> {
    conv_window_with_completion(x, weight, state, kernel, dilation, false)
}

// A tentative window owns immutable banks until its final output/state join.
// Keep the tiny history copy lazy within that window, as TrackFastModel does.
pub(super) fn conv_window_with_completion(
    x: &MxArray,
    weight: &MxArray,
    state: &mut Option<MxArray>,
    kernel: usize,
    dilation: usize,
    defer_state: bool,
) -> Result<MxArray> {
    let shape = x.shape()?;
    let (tokens, width) = (shape[1], shape[2]);
    if tokens == 1 {
        return conv_with_completion(x, weight, state, kernel, dilation, defer_state);
    }
    let keep = ((kernel - 1) * dilation) as i64;
    let old = match state {
        Some(s) => s.clone(),
        None => MxArray::zeros(&[keep, width], Some(x.dtype()?))?,
    };
    let input = MxArray::concatenate(&old, &x.reshape(&[tokens, width])?, 0)?;
    let ported = if kernel == 4
        && dilation == 3
        && width == 10240
        && (9..=1024).contains(&tokens)
        && x.dtype()? == DType::BFloat16
        && weight.dtype()? == DType::Float32
        && crate::engine::persistence::compiled_forward_backend_available()
        && !runtime_flags::is_zero(c"MLX_QWEN4_PREFILL_PLE_CONV")
    {
        let raw = unsafe {
            mlx_sys::mlx_qwen4_prefill_ple_conv(
                input.as_raw_ptr(),
                weight.reshape(&[width, 4])?.as_raw_ptr(),
            )
        };
        if raw.is_null() {
            None
        } else {
            Some(MxArray::from_handle(
                raw,
                "Qwen4 reference PLE convolution",
            )?)
        }
    } else {
        None
    };
    let out = if let Some(out) = ported {
        out
    } else {
        let indices = (0..tokens)
            .flat_map(|t| (0..kernel).map(move |k| t as i32 + (k * dilation) as i32))
            .collect::<Vec<_>>();
        let selected = input
            .take(&MxArray::from_int32(&indices, &[indices.len() as i64])?, 0)?
            .reshape(&[tokens, kernel as i64, width])?
            .astype(DType::Float32)?;
        let weights = weight
            .reshape(&[width, kernel as i64])?
            .transpose(None)?
            .astype(DType::Float32)?;
        selected
            .mul(&weights)?
            .sum(Some(&[1]), Some(false))?
            .astype(x.dtype()?)?
            .reshape(&shape)?
    };
    let tail = input.slice_axis(0, tokens, tokens + keep)?.deep_copy()?;
    if !defer_state {
        MxArray::eval_arrays_with_context(&[&tail], "qwen4::ple::conv_tail")?;
    }
    *state = Some(tail);
    Activations::silu(&out)
}

pub fn hash_ids(
    token: u32,
    previous: &[u32],
    eos: u32,
    ngram: usize,
    heads: usize,
    multipliers: &[u64],
    sizes: &[u64],
    offsets: &[u64],
) -> Result<Vec<u64>> {
    if multipliers.len() != ngram
        || sizes.len() != (ngram - 1) * heads
        || offsets.len() != sizes.len()
        || sizes.contains(&0)
    {
        return Err(Error::from_reason("Invalid PLE hash constants"));
    }
    let mut history = vec![token as u64];
    let mut stop = false;
    for back in 0..ngram - 1 {
        let v = previous.iter().rev().nth(back).copied().unwrap_or(eos);
        stop |= v == eos;
        history.push(if stop { eos } else { v } as u64);
    }
    let mut mixed = history[0].wrapping_mul(multipliers[0]);
    let mut ids = Vec::new();
    for n in 1..ngram {
        mixed ^= history[n].wrapping_mul(multipliers[n]);
        for h in 0..heads {
            let i = (n - 1) * heads + h;
            // The released constants keep products below i64::MAX. Reject bad
            // constants instead of silently changing signed remainder semantics.
            if mixed > i64::MAX as u64 {
                return Err(Error::from_reason("PLE hash product exceeds signed int64"));
            }
            ids.push(
                (mixed % sizes[i])
                    .checked_add(offsets[i])
                    .ok_or_else(|| Error::from_reason("PLE index overflow"))?,
            );
        }
    }
    Ok(ids)
}

pub(super) fn routes_shared_gate(
    logits: &MxArray,
    x: &MxArray,
    weight: &MxArray,
    top: usize,
) -> Result<Option<(MxArray, MxArray, MxArray)>> {
    if top != 10
        || !runtime_flags::is_one(c"MLX_QWEN4_ROUTE_SHARED_GATE")
        || runtime_flags::is_zero(c"MLX_QWEN4_SINGLETON_ROUTER")
        || !crate::engine::persistence::compiled_forward_backend_available()
    {
        return Ok(None);
    }
    let (mut ids, mut scores, mut gate) = (
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        std::ptr::null_mut(),
    );
    if !unsafe {
        mlx_sys::mlx_qwen4_routes_shared_gate(
            logits.as_raw_ptr(),
            x.as_raw_ptr(),
            weight.as_raw_ptr(),
            &mut ids,
            &mut scores,
            &mut gate,
        )
    } {
        return Ok(None);
    }
    Ok(Some((
        MxArray::from_handle(ids, "combined route IDs")?,
        MxArray::from_handle(scores, "combined route scores")?,
        MxArray::from_handle(gate, "combined shared gate")?,
    )))
}

pub(super) fn singleton_routes(logits: &MxArray, top: usize) -> Result<Option<(MxArray, MxArray)>> {
    if top != 10
        || *logits.shape()? != [1, 1, 512]
        || !matches!(logits.dtype()?, DType::BFloat16 | DType::Float32)
        || !crate::engine::persistence::compiled_forward_backend_available()
        || runtime_flags::is_zero(c"MLX_QWEN4_SINGLETON_ROUTER")
    {
        return Ok(None);
    }
    let (mut ids, mut scores) = (std::ptr::null_mut(), std::ptr::null_mut());
    let ok =
        unsafe { mlx_sys::mlx_qwen4_singleton_routes(logits.as_raw_ptr(), &mut ids, &mut scores) };
    if !ok {
        return Err(Error::from_reason("Qwen4 singleton routing kernel failed"));
    }
    Ok(Some((
        MxArray::from_handle(ids, "singleton route IDs")?,
        MxArray::from_handle(scores, "singleton route scores")?,
    )))
}

pub(super) fn prefill_routes(logits: &MxArray, top: usize) -> Result<Option<(MxArray, MxArray)>> {
    if top != 10
        || logits.ndim()? != 3
        || logits.shape_at(0)? != 1
        || !(9..=1024).contains(&logits.shape_at(1)?)
        || logits.shape_at(2)? != 512
        || !matches!(logits.dtype()?, DType::BFloat16 | DType::Float32)
        || !crate::engine::persistence::compiled_forward_backend_available()
        || runtime_flags::is_zero(c"MLX_QWEN4_PREFILL_ROUTER")
    {
        return Ok(None);
    }
    let (mut ids, mut scores) = (std::ptr::null_mut(), std::ptr::null_mut());
    let ok =
        unsafe { mlx_sys::mlx_qwen4_prefill_routes(logits.as_raw_ptr(), &mut ids, &mut scores) };
    if !ok {
        return Err(Error::from_reason("Qwen4 prefill routing kernel failed"));
    }
    Ok(Some((
        MxArray::from_handle(ids, "prefill route IDs")?,
        MxArray::from_handle(scores, "prefill route scores")?,
    )))
}

pub(super) fn route_sort(
    ids: &MxArray,
    experts: usize,
) -> Result<Option<(MxArray, MxArray, MxArray)>> {
    if !(256..=65536).contains(&ids.size()?)
        || experts == 0
        || experts > 1024
        || ids.dtype()? != DType::Uint32
        || ids.ndim()? != 1
        || !crate::engine::persistence::compiled_forward_backend_available()
        || runtime_flags::is_zero(c"MLX_QWEN4_COUNTING_SORT")
    {
        return Ok(None);
    }
    let (mut order, mut inverse, mut sorted) = (
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        std::ptr::null_mut(),
    );
    let ok = unsafe {
        mlx_sys::mlx_qwen4_route_sort(
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
    Ok(Some((
        MxArray::from_handle(order, "route order")?,
        MxArray::from_handle(inverse, "route inverse")?,
        MxArray::from_handle(sorted, "sorted routes")?,
    )))
}

pub(super) fn gdn_gates(
    a: &MxArray,
    b: &MxArray,
    scale: &MxArray,
    dt: &MxArray,
) -> Result<(MxArray, MxArray)> {
    // Fallback consumers retain the original F32 gate arithmetic even when
    // complete GDN/preparation receives compact projection storage.
    let a = a.astype(DType::Float32)?;
    let b = b.astype(DType::Float32)?;
    if !runtime_flags::is_zero(c"MLX_QWEN4_FUSED_POINTWISE")
        && crate::engine::persistence::compiled_forward_backend_available()
    {
        let (mut decay, mut beta) = (std::ptr::null_mut(), std::ptr::null_mut());
        let ok = unsafe {
            mlx_sys::mlx_qwen4_gdn_gates(
                a.as_raw_ptr(),
                b.as_raw_ptr(),
                scale.as_raw_ptr(),
                dt.as_raw_ptr(),
                &mut decay,
                &mut beta,
            )
        };
        if !ok {
            return Err(Error::from_reason("Qwen4 compiled GDN gates failed"));
        }
        return Ok((
            MxArray::from_handle(decay, "GDN decay")?,
            MxArray::from_handle(beta, "GDN beta")?,
        ));
    }
    Ok((
        Activations::softplus(&a.add(dt)?)?.mul(scale)?.exp()?,
        Activations::sigmoid(&b)?,
    ))
}

pub(super) fn swiglu(gate: &MxArray, up: &MxArray) -> Result<MxArray> {
    if runtime_flags::is_zero(c"MLX_QWEN4_FUSED_POINTWISE") {
        Activations::silu(gate)?.mul(up)
    } else {
        Activations::swiglu_compiled(gate, up)
    }
}
fn fused_attention_gate(attention: &MxArray, projection: &MxArray) -> Result<Option<MxArray>> {
    if !crate::engine::persistence::compiled_forward_backend_available() {
        return Ok(None);
    }
    let raw = unsafe {
        mlx_sys::mlx_qwen4_attention_gate(attention.as_raw_ptr(), projection.as_raw_ptr())
    };
    if raw.is_null() {
        Ok(None)
    } else {
        MxArray::from_handle(raw, "Qwen4 attention output gate").map(Some)
    }
}

pub(super) fn attention_output(attention: &MxArray, projection: &MxArray) -> Result<MxArray> {
    if runtime_flags::is_one(c"MLX_QWEN4_ATTENTION_GATE")
        && !runtime_flags::is_zero(c"MLX_QWEN4_FUSED_POINTWISE")
        && let Some(output) = fused_attention_gate(attention, projection)?
    {
        return Ok(output);
    }
    let shape = attention.shape()?;
    let (tokens, heads, width) = (shape[2], shape[1], shape[3]);
    let gate = projection
        .slice_axis(3, width, 2 * width)?
        .reshape(&[1, tokens, heads * width])?;
    let value = attention
        .transpose(Some(&[0, 2, 1, 3]))?
        .reshape(&[1, tokens, heads * width])?;
    sigmoid_mul(&gate, &value)
}

pub(super) fn sigmoid_mul(gate: &MxArray, value: &MxArray) -> Result<MxArray> {
    if runtime_flags::is_zero(c"MLX_QWEN4_FUSED_POINTWISE") {
        value.mul(&Activations::sigmoid(gate)?)
    } else {
        Activations::sigmoid_mul_compiled(gate, value)
    }
}

#[cfg(test)]
mod attention_gate_tests {
    use super::*;

    #[test]
    fn attention_gate_preserves_layouts_native_sigmoid_and_changing_inputs() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        for seed in [0_u32, 197] {
            for (tokens, heads) in [(1_i64, 1_i64), (7, 8), (64, 8), (1024, 24)] {
                let attention = values(
                    (tokens + 1) * heads * 256,
                    &[1, tokens + 1, heads, 256],
                    seed as f32,
                )
                .astype(DType::BFloat16)
                .unwrap()
                .slice_axis(1, 1, tokens + 1)
                .unwrap()
                .transpose(Some(&[0, 2, 1, 3]))
                .unwrap();
                let data: Vec<_> = (0..(tokens + 1) * heads * 512)
                    .map(|i| {
                        // All finite BF16 inputs, signed zeros and infinities;
                        // includes native sigmoid exponential halfway cases.
                        let bits = ((i / 512 * 256 + i % 256) as u32 + seed) & 65535;
                        let x = f32::from_bits(bits << 16);
                        if x.is_nan() { -6.84375 } else { x }
                    })
                    .collect();
                let projection = MxArray::from_float32(&data, &[1, tokens + 1, heads, 512])
                    .unwrap()
                    .astype(DType::BFloat16)
                    .unwrap()
                    .slice_axis(1, 1, tokens + 1)
                    .unwrap();
                let gate = projection
                    .slice_axis(3, 256, 512)
                    .unwrap()
                    .reshape(&[1, tokens, heads * 256])
                    .unwrap();
                let value = attention
                    .transpose(Some(&[0, 2, 1, 3]))
                    .unwrap()
                    .reshape(&[1, tokens, heads * 256])
                    .unwrap();
                let expected = Activations::sigmoid_mul_compiled(&gate, &value).unwrap();
                let actual = fused_attention_gate(&attention, &projection)
                    .unwrap()
                    .expect("eligible direct-address attention gate");
                let actual = actual.to_float32().unwrap();
                let expected = expected.to_float32().unwrap();
                assert!(
                    actual
                        .iter()
                        .zip(expected.iter())
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "seed={seed} tokens={tokens} heads={heads}",
                );
                assert!(
                    fused_attention_gate(&attention.astype(DType::Float32).unwrap(), &projection)
                        .unwrap()
                        .is_none()
                );
            }
        }
    }

    fn values(n: i64, shape: &[i64], phase: f32) -> MxArray {
        MxArray::from_float32(
            &(0..n)
                .map(|i| (i as f32 * 0.137 + phase).sin() * 3.1)
                .collect::<Vec<_>>(),
            shape,
        )
        .unwrap()
    }

    fn equal(a: &MxArray, b: &MxArray) {
        assert_eq!(a.dtype().unwrap(), b.dtype().unwrap());
        assert_eq!(a.shape().unwrap().to_vec(), b.shape().unwrap().to_vec());
        assert_eq!(
            a.astype(DType::Float32)
                .unwrap()
                .to_float32()
                .unwrap()
                .to_vec(),
            b.astype(DType::Float32)
                .unwrap()
                .to_float32()
                .unwrap()
                .to_vec()
        );
    }

    #[test]
    fn window_convolution_preserves_rounding_and_chunk_history() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        for width in [17, 256, 10240] {
            // Transposed weights and offset activation views also exercise
            // the custom primitive's contiguous-input preparation.
            let w = values(4 * width, &[4, width], 0.7).transpose(None).unwrap();
            let x = values(73 * width, &[1, 73, width], 1.3)
                .astype(DType::BFloat16)
                .unwrap();
            let old = values(6 * width, &[6, width], 2.1)
                .astype(DType::BFloat16)
                .unwrap()
                .slice_axis(0, 3, 6)
                .unwrap();
            let mut expected = Some(old.clone());
            let mut actual = Some(old);
            let mut offset = 0;
            for count in [1, 2, 3, 7, 60] {
                let input = x.slice_axis(1, offset, offset + count).unwrap();
                let want = conv_sequence_reference(&input, &w, &mut expected, 4).unwrap();
                // Call the primitive directly so an unsupported/fallback
                // dispatch cannot make a kernel regression pass silently.
                let mut out = std::ptr::null_mut();
                let mut history = std::ptr::null_mut();
                assert!(unsafe {
                    mlx_sys::mlx_qwen4_window_conv(
                        input.as_raw_ptr(),
                        actual.as_ref().unwrap().as_raw_ptr(),
                        w.as_raw_ptr(),
                        &mut out,
                        &mut history,
                    )
                });
                let got = MxArray::from_handle(out, "test window conv").unwrap();
                let next = MxArray::from_handle(history, "test window history").unwrap();
                equal(&got, &want);
                equal(&next, expected.as_ref().unwrap());
                actual = Some(next);
                offset += count;
            }
            let mut whole_state = Some(
                values(6 * width, &[6, width], 2.1)
                    .astype(DType::BFloat16)
                    .unwrap()
                    .slice_axis(0, 3, 6)
                    .unwrap(),
            );
            conv_sequence(&x, &w, &mut whole_state, 4).unwrap();
            equal(whole_state.as_ref().unwrap(), actual.as_ref().unwrap());
        }
    }
}

#[cfg(test)]
mod prefill_epilogue_tests {
    use super::*;

    #[test]
    fn replay_keeps_bf16_norm_and_output_gate_boundaries() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        for tokens in [1i64, 7, 128, 512, 692, 1024] {
            let shape = [1, tokens, 2, 128];
            let n = (tokens * 2 * 128) as usize;
            let values: Vec<_> = (0..n).map(|i| (i as f32 * 0.013).sin() * 5.3).collect();
            let gates: Vec<_> = (0..n).map(|i| (i as f32 * 0.177).cos() * 9.1).collect();
            let scales: Vec<_> = (0..128).map(|i| (i as f32 * 0.103).cos() * 1.3).collect();
            let out = MxArray::from_float32(&values, &shape).unwrap();
            let z = MxArray::from_float32(&gates, &shape)
                .unwrap()
                .astype(DType::BFloat16)
                .unwrap();
            let w = MxArray::from_float32(&scales, &[128]).unwrap();
            let normalized =
                norm(&out.astype(DType::BFloat16).unwrap(), &w, 128, 1e-6, false).unwrap();
            let expected = sigmoid_mul(
                &z.astype(DType::Float32).unwrap(),
                &normalized.astype(DType::Float32).unwrap(),
            )
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
            let actual = gdn_epilogue(&out, &z, &w, 1e-6).unwrap().unwrap();
            assert_eq!(
                &*actual.to_float32().unwrap(),
                &*expected.to_float32().unwrap(),
                "tokens={tokens}"
            );
        }
    }
}
