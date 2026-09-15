use crate::array::{DType, MxArray};
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
    if std::env::var("MLX_QWEN4_FUSED_EXPERTS").as_deref() == Ok("0")
        || std::env::var("MLX_QWEN4_DIRECT_GEMV").as_deref() == Ok("0")
        || std::env::var("MLX_QWEN4_AFFINE_GEMV").as_deref() == Ok("0")
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
        || banks
            .iter()
            .any(|w| w.group != 32 || w.scales.is_none() || w.biases.is_none())
    {
        return Ok(None);
    }
    let indices = ids.reshape(&[-1])?;
    let raw = unsafe {
        mlx_sys::mlx_qwen4_routed_experts(
            x.as_raw_ptr(),
            indices.as_raw_ptr(),
            scores.as_raw_ptr(),
            gate.values.as_raw_ptr(),
            gate.scales.as_ref().unwrap().as_raw_ptr(),
            gate.biases.as_ref().unwrap().as_raw_ptr(),
            up.values.as_raw_ptr(),
            up.scales.as_ref().unwrap().as_raw_ptr(),
            up.biases.as_ref().unwrap().as_raw_ptr(),
            down.values.as_raw_ptr(),
            down.scales.as_ref().unwrap().as_raw_ptr(),
            down.biases.as_ref().unwrap().as_raw_ptr(),
        )
    };
    if raw.is_null() {
        Ok(None)
    } else {
        MxArray::from_handle(raw, "Qwen4 fused routed experts").map(Some)
    }
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
        || banks
            .iter()
            .any(|w| w.group != 32 || w.scales.is_none() || w.biases.is_none())
    {
        return Ok(None);
    }
    let raw = unsafe {
        mlx_sys::mlx_qwen4_prefill_indirect(
            x.as_raw_ptr(),
            ids.as_raw_ptr(),
            token_rows.as_raw_ptr(),
            gate.values.as_raw_ptr(),
            gate.scales.as_ref().unwrap().as_raw_ptr(),
            gate.biases.as_ref().unwrap().as_raw_ptr(),
            up.values.as_raw_ptr(),
            up.scales.as_ref().unwrap().as_raw_ptr(),
            up.biases.as_ref().unwrap().as_raw_ptr(),
            down.values.as_raw_ptr(),
            down.scales.as_ref().unwrap().as_raw_ptr(),
            down.biases.as_ref().unwrap().as_raw_ptr(),
            experts as i32,
        )
    };
    if raw.is_null() {
        Ok(None)
    } else {
        if std::env::var("MLX_QWEN4_TRACE_PREFILL_INDIRECT").as_deref() == Ok("1") {
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
        && std::env::var("MLX_QWEN4_SORTED_COMBINE").as_deref() != Ok("0")
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
    let kd = *shape.last().unwrap();
    if kd >= 32
        && kd % 32 == 0
        && crate::engine::persistence::compiled_forward_backend_available()
        && std::env::var("MLX_QWEN4_FUSED_GDN").as_deref() != Ok("0")
    {
        let q = q.reshape(&[1, 1, shape[1], kd])?;
        let k = k.reshape(&[1, 1, shape[1], kd])?;
        let heads = state.shape()?[1];
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
    let repeats = state.shape()?[1] / shape[1];
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
        && std::env::var("MLX_QWEN4_FUSED_GDN").as_deref() != Ok("0")
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
        && std::env::var("MLX_QWEN4_WINDOW_CONV").as_deref() != Ok("0")
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
    if std::env::var("MLX_QWEN4_FUSED_POINTWISE").as_deref() != Ok("0")
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
    let width = *x.shape()?.last().unwrap();
    let half = dims as i64 / 2;
    let angles = rotary_angles(positions, dims, theta, sections, interleaved);
    let angles = MxArray::from_float32(&angles, &[half])?;
    let cos = angles.cos()?.astype(x.dtype()?)?;
    let sin = angles.sin()?.astype(x.dtype()?)?;
    let a = x.slice_axis(x.shape()?.len() - 1, 0, half)?;
    let b = x.slice_axis(x.shape()?.len() - 1, half, dims as i64)?;
    let left = a.mul(&cos)?.sub(&b.mul(&sin)?)?;
    let right = b.mul(&cos)?.add(&a.mul(&sin)?)?;
    let tail = x.slice_axis(x.shape()?.len() - 1, dims as i64, width)?;
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
        if let Some((_, cos, sin)) = self.0.iter().find(|(old, _, _)| *old == key) {
            return apply_rotary_window(x, dims, cos, sin);
        }
        let (cos, sin) = rotary_tables(
            &key.positions,
            dims,
            theta,
            sections,
            interleaved,
            key.dtype,
        )?;
        let out = apply_rotary_window(x, dims, &cos, &sin)?;
        if self.0.len() == 2 {
            self.0.remove(0);
        }
        self.0.push((key, cos, sin));
        Ok(out)
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

fn apply_rotary_window(x: &MxArray, dims: usize, cos: &MxArray, sin: &MxArray) -> Result<MxArray> {
    let shape = x.shape()?;
    if shape.len() != 4
        || shape[2] != cos.shape()?[2]
        || !dims.is_multiple_of(2)
        || dims as i64 > shape[3]
    {
        return Err(Error::from_reason("Qwen4 rotary window shape mismatch"));
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
    fn shared_tables_preserve_scalar_rounding_and_media_axes() {
        let mut cache = RotaryWindowCache::default();
        for dtype in [DType::BFloat16, DType::Float32] {
            for interleaved in [true, false] {
                for (count, stride, base) in [(7, 4, 1), (256, 4, 2045), (1024, 1, 0)] {
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

/// Stateful depthwise causal convolution; the state contains only the finite
/// input window, and each output is evaluated before the next SSD weight read.
pub fn conv(
    x: &MxArray,
    weight: &MxArray,
    state: &mut Option<MxArray>,
    kernel: usize,
    dilation: usize,
) -> Result<MxArray> {
    let width = *x.shape()?.last().unwrap();
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
    MxArray::eval_arrays_with_context(&[&next], "qwen4::math::next")?;
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
    let shape = x.shape()?;
    let (tokens, width) = (shape[1], shape[2]);
    if tokens == 1 {
        return conv(x, weight, state, kernel, dilation);
    }
    let keep = ((kernel - 1) * dilation) as i64;
    let old = match state {
        Some(s) => s.clone(),
        None => MxArray::zeros(&[keep, width], Some(x.dtype()?))?,
    };
    let input = MxArray::concatenate(&old, &x.reshape(&[tokens, width])?, 0)?;
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
    let out = selected
        .mul(&weights)?
        .sum(Some(&[1]), Some(false))?
        .astype(x.dtype()?)?
        .reshape(&shape)?;
    let tail = input.slice_axis(0, tokens, tokens + keep)?.deep_copy()?;
    MxArray::eval_arrays_with_context(&[&tail], "qwen4::ple::conv_tail")?;
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
        || std::env::var("MLX_QWEN4_COUNTING_SORT").as_deref() == Ok("0")
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
    if std::env::var("MLX_QWEN4_FUSED_POINTWISE").as_deref() != Ok("0")
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
        Activations::sigmoid(b)?,
    ))
}

pub(super) fn swiglu(gate: &MxArray, up: &MxArray) -> Result<MxArray> {
    if std::env::var("MLX_QWEN4_FUSED_POINTWISE").as_deref() == Ok("0") {
        Activations::silu(gate)?.mul(up)
    } else {
        Activations::swiglu_compiled(gate, up)
    }
}
pub(super) fn sigmoid_mul(gate: &MxArray, value: &MxArray) -> Result<MxArray> {
    if std::env::var("MLX_QWEN4_FUSED_POINTWISE").as_deref() == Ok("0") {
        value.mul(&Activations::sigmoid(gate)?)
    } else {
        Activations::sigmoid_mul_compiled(gate, value)
    }
}

#[cfg(test)]
mod window_conv_tests {
    use super::*;

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
