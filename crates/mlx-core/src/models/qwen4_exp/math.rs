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
    if !crate::engine::persistence::compiled_forward_backend_available()
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
    if kd >= 32 && kd % 32 == 0 && crate::engine::persistence::compiled_forward_backend_available()
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
    if kd >= 32 && kd % 32 == 0 && crate::engine::persistence::compiled_forward_backend_available()
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
    if crate::engine::persistence::compiled_forward_backend_available() {
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

/// The attention and compressed-indexer tables of the windows one pass
/// holds: two per window, so a layer-major group of W windows keeps 2W.
/// These are temporary graph inputs, never part of an owner's persistent state.
pub(super) struct RotaryWindowCache {
    tables: Vec<(RotaryKey, MxArray, MxArray)>,
    capacity: usize,
}

impl Default for RotaryWindowCache {
    fn default() -> Self {
        Self {
            tables: Vec::new(),
            capacity: 2,
        }
    }
}

impl RotaryWindowCache {
    pub fn clear(&mut self) {
        self.tables.clear();
        self.capacity = 2;
    }

    /// Hold the tables of `windows` windows for the pass that starts.
    pub fn reserve(&mut self, windows: usize) {
        self.capacity = 2 * windows.max(1);
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

    fn tables(&mut self, key: RotaryKey) -> Result<(MxArray, MxArray)> {
        if let Some((_, cos, sin)) = self.tables.iter().find(|(old, _, _)| *old == key) {
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
        if self.tables.len() >= self.capacity {
            self.tables.remove(0);
        }
        self.tables.push((key, cos.clone(), sin.clone()));
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
        let recurrent_error = recurrent_step(&scalar, &scalar, &scalar, &scalar, &scalar, &scalar)
            .err()
            .expect("scalar recurrent input must be rejected");
        assert_eq!(
            recurrent_error.reason,
            "Qwen4 recurrent query must have shape [1, heads, 1, key_dim]"
        );

        let mut history = Some(MxArray::from_float32(&[2., 3.], &[2, 1]).unwrap());
        let convolution_error = conv_with_completion(&scalar, &scalar, &mut history, 3, 1, false)
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
                        assert_eq!(
                            &*got.astype(DType::Float32).unwrap().to_float32().unwrap(),
                            &*expected
                                .astype(DType::Float32)
                                .unwrap()
                                .to_float32()
                                .unwrap(),
                            "{dtype:?}, {interleaved}, {count}, {width}"
                        );
                        assert!(cache.tables.len() <= 2);
                    }
                }
            }
        }
        cache.clear();
        assert!(cache.tables.is_empty());
    }
}

/// Stateful depthwise causal convolution. Unless `defer_state`, complete the
/// owned finite history before returning the lazy activation to its caller;
/// a deferred history must be completed before its frontier is published.
pub(super) fn conv_with_completion(
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

/// Dilated causal convolution over a PLE window. Keep the same tap order
/// and F32 products as the singleton operation, with an owned finite tail.
/// The tail stays lazy: it is a slice of an input the window's own output
/// materializes, so the next window or the route commit completes it
/// without reading any expert bank, and the token start does not drain
/// the GPU for a 10 KB copy.
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
        return conv_with_completion(x, weight, state, kernel, dilation, true);
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
    *state = Some(input.slice_axis(0, tokens, tokens + keep)?.deep_copy()?);
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

pub(super) fn singleton_routes(logits: &MxArray, top: usize) -> Result<Option<(MxArray, MxArray)>> {
    if top != 10
        || *logits.shape()? != [1, 1, 512]
        || !matches!(logits.dtype()?, DType::BFloat16 | DType::Float32)
        || !crate::engine::persistence::compiled_forward_backend_available()
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
    if crate::engine::persistence::compiled_forward_backend_available() {
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
    Activations::swiglu_compiled(gate, up)
}
pub(super) fn attention_output(attention: &MxArray, projection: &MxArray) -> Result<MxArray> {
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

/// mlx-vlm's batched QSA block choice (`select_from_projected`) for rows
/// `base..base + t`, from the rotated indexer queries `[1, heads, t, dim]` and
/// the rotated block keys `[blocks, dim]`. Returns the sorted `[1, t, topk]`
/// picks (-1 where unused) and the exclusive `[1, t]` query ends.
pub(super) fn qsa_select(
    query: &MxArray,
    index_keys: &MxArray,
    base: usize,
    ratio: usize,
    topk: usize,
) -> Result<(MxArray, MxArray)> {
    let shape = query.shape()?;
    let (t, dim) = (shape[2], shape[3]);
    let blocks = index_keys.shape_at(0)?;
    let keys = index_keys
        .astype(DType::Float32)?
        .reshape(&[1, 1, blocks, dim])?
        .transpose(Some(&[0, 1, 3, 2]))?;
    let scores = query
        .astype(DType::Float32)?
        .matmul(&keys)?
        .maximum(&MxArray::scalar_float(0.0)?)?
        .sum(Some(&[1]), Some(false))?
        .div_scalar((dim as f64).sqrt())?;
    let complete: Vec<i32> = (0..t as usize)
        .map(|j| ((base + j + 1) / ratio) as i32)
        .collect();
    let complete = MxArray::from_int32(&complete, &[1, t, 1])?;
    let ids =
        MxArray::arange(0., blocks as f64, None, Some(DType::Int32))?.reshape(&[1, 1, blocks])?;
    let negative = MxArray::scalar_float(f64::NEG_INFINITY)?;
    let mut scores = ids.less(&complete)?.where_(&scores, &negative)?;
    let topk = topk as i64;
    if blocks < topk {
        scores = scores.pad(&[0, 0, 0, 0, 0, (topk - blocks) as i32], f64::NEG_INFINITY)?;
    }
    let width = blocks.max(topk);
    let selected = scores
        .argpartition(-(topk as i32), Some(-1))?
        .slice_axis(2, width - topk, width)?
        .astype(DType::Int32)?;
    let selected = selected
        .less(&complete)?
        .where_(&selected, &MxArray::scalar_int(-1)?)?
        .sort(Some(-1))?;
    let query_ends = MxArray::arange(
        (base + 1) as f64,
        (base + t as usize + 1) as f64,
        None,
        Some(DType::Int32),
    )?
    .reshape(&[1, t])?;
    Ok((selected, query_ends))
}

/// Which indexed QSA kernel runs a window. Both write the same bits; the
/// paired kernel reads each K/V row once for two heads of the kv group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum QsaKernel {
    /// The paired kernel when its pipeline holds the 1024-thread threadgroup
    /// on this GPU, the single-head kernel otherwise.
    ByCapability,
    #[cfg(test)]
    SingleHead,
    #[cfg(test)]
    PairedHeads,
}

impl QsaKernel {
    fn heads_per_group(self) -> i32 {
        match self {
            Self::ByCapability => 0,
            #[cfg(test)]
            Self::SingleHead => 1,
            #[cfg(test)]
            Self::PairedHeads => 2,
        }
    }
}

/// mlx-vlm's indexed QSA kernel. `selected` is `[b, t, topk]` sorted block ids
/// with -1 for unused picks; each row also attends its incomplete block up to
/// `query_ends` (`[b, t]`, exclusive).
#[allow(clippy::too_many_arguments)]
pub(super) fn qsa_attention(
    q: &MxArray,
    keys: &MxArray,
    values: &MxArray,
    selected: &MxArray,
    query_ends: &MxArray,
    scale: f64,
    block_size: usize,
    kernel: QsaKernel,
) -> Result<MxArray> {
    qsa_dispatch(
        q, keys, values, selected, query_ends, scale, block_size, kernel,
    )
    .map(|(out, _)| out)
}

/// `qsa_attention` and how many heads of the kv group each K/V row read of the
/// kernel that ran serves (1 or 2).
#[allow(clippy::too_many_arguments)]
pub(super) fn qsa_dispatch(
    q: &MxArray,
    keys: &MxArray,
    values: &MxArray,
    selected: &MxArray,
    query_ends: &MxArray,
    scale: f64,
    block_size: usize,
    kernel: QsaKernel,
) -> Result<(MxArray, i32)> {
    let mut out = std::ptr::null_mut();
    let heads = unsafe {
        mlx_sys::mlx_qwen4_qsa_attention(
            q.as_raw_ptr(),
            keys.as_raw_ptr(),
            values.as_raw_ptr(),
            selected.as_raw_ptr(),
            query_ends.as_raw_ptr(),
            scale as f32,
            block_size as i32,
            kernel.heads_per_group(),
            &mut out,
        )
    };
    if heads == 0 {
        return Err(Error::from_reason("Qwen4 sparse attention kernel failed"));
    }
    tracing::debug!(heads_per_kv_read = heads, "Qwen4 sparse attention kernel");
    Ok((MxArray::from_handle(out, "Qwen4 sparse attention")?, heads))
}

/// `maxTotalThreadsPerThreadgroup` of `kernel`'s pipeline (`SingleHead` or
/// `PairedHeads`) for these arrays on this GPU; 0 when it cannot be built.
#[cfg(test)]
pub(super) fn qsa_pipeline_threads(
    q: &MxArray,
    keys: &MxArray,
    values: &MxArray,
    selected: &MxArray,
    query_ends: &MxArray,
    block_size: usize,
    kernel: QsaKernel,
) -> i32 {
    unsafe {
        mlx_sys::mlx_qwen4_qsa_attention_pipeline_threads(
            q.as_raw_ptr(),
            keys.as_raw_ptr(),
            values.as_raw_ptr(),
            selected.as_raw_ptr(),
            query_ends.as_raw_ptr(),
            block_size as i32,
            kernel.heads_per_group(),
        )
    }
}

/// Makes this thread's next `count` QSA pipeline probes that miss the cache
/// fail as a Metal compile error would; returns the count armed before.
#[cfg(test)]
pub(super) fn qsa_fail_pipeline_probes(count: i32) -> i32 {
    unsafe { mlx_sys::mlx_test_qwen4_qsa_fail_probes(count) }
}

pub(super) fn sigmoid_mul(gate: &MxArray, value: &MxArray) -> Result<MxArray> {
    Activations::sigmoid_mul_compiled(gate, value)
}

#[cfg(test)]
mod attention_gate_tests {
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

/// Row-batched twins of the single-row decode kernels. Up to four activation
/// rows share one weight stream, and row `j` of every output carries the bits
/// of the single-row kernel applied to row `j` alone.
pub(super) mod rows {
    use super::super::weights::Weight;
    use crate::array::{DType, MxArray};
    use napi::{Error, Result};
    use std::sync::Arc;

    pub(crate) const MAX_ROWS: i64 = 4;

    fn backend() -> bool {
        crate::engine::persistence::compiled_forward_backend_available()
    }

    fn affine_q8(w: &Weight) -> Option<(&MxArray, &MxArray)> {
        if w.mode != "affine" || w.bits != 8 || w.group != 32 {
            return None;
        }
        match (&w.scales, &w.biases) {
            (Some(scales), Some(biases)) => Some((scales, biases)),
            _ => None,
        }
    }

    fn row_count(x: &MxArray, width: Option<i64>) -> Result<Option<i64>> {
        let shape = x.shape()?;
        Ok(match &*shape {
            [1, rows, k]
                if (1..=MAX_ROWS).contains(rows) && width.is_none_or(|width| width == *k) =>
            {
                Some(*rows)
            }
            _ => None,
        })
    }

    /// `x` is `[1, R, K]`; the result is `[1, R, N]` for an admitted Q8 bank.
    pub(crate) fn dense_decode(w: &Weight, x: &MxArray) -> Result<Option<MxArray>> {
        let Some((scales, biases)) = affine_q8(w) else {
            return Ok(None);
        };
        if row_count(x, None)?.is_none() || !backend() {
            return Ok(None);
        }
        let raw = unsafe {
            mlx_sys::mlx_qwen4_dense_decode_rows(
                x.as_raw_ptr(),
                w.values.as_raw_ptr(),
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

    /// `x` is `[1, R, 10240]`; the result is the `[1, R, 320]` mixer activation.
    pub(crate) fn mixer_act(w: &Weight, x: &MxArray) -> Result<Option<MxArray>> {
        let Some((scales, biases)) = affine_q8(w) else {
            return Ok(None);
        };
        if row_count(x, Some(10240))?.is_none() || !backend() {
            return Ok(None);
        }
        let raw = unsafe {
            mlx_sys::mlx_qwen4_decode_mixer_act_rows(
                x.as_raw_ptr(),
                w.values.as_raw_ptr(),
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

    /// `x` is `[1, R, 320]`, `normed` is `[1, R, 10240]`; the result is `[1, R, 2560]`.
    pub(crate) fn hyper_up(w: &Weight, x: &MxArray, normed: &MxArray) -> Result<Option<MxArray>> {
        let Some((scales, biases)) = affine_q8(w) else {
            return Ok(None);
        };
        if row_count(x, Some(320))?.is_none() || !backend() {
            return Ok(None);
        }
        let raw = unsafe {
            mlx_sys::mlx_qwen4_hyper_up_rows(
                x.as_raw_ptr(),
                w.values.as_raw_ptr(),
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

    /// `x` is `[1, R, 2560]`, `ids` holds `10 R` routes, `scores` `10 R`
    /// weights and `shared_gate` is `[1, R, 1]`; the result is `[1, R, 2560]`.
    pub(crate) fn routed_shared_experts(
        x: &MxArray,
        ids: &MxArray,
        scores: &MxArray,
        banks: &[Arc<Weight>; 3],
        shared: &[Arc<Weight>; 3],
        shared_gate: &MxArray,
    ) -> Result<Option<MxArray>> {
        if !backend()
            || row_count(x, Some(2560))?.is_none()
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
        let raw =
            unsafe { mlx_sys::mlx_qwen4_routed_shared_experts_rows(inputs.as_ptr(), inputs.len()) };
        if raw.is_null() {
            Ok(None)
        } else {
            MxArray::from_handle(raw, "Qwen4 row-batched routed/shared experts").map(Some)
        }
    }

    /// `logits` is `[1, R, 512]`; returns (`[1, R, 10]` ids, `[1, R, 10]` scores)
    /// with the single-row kernel's probability rounding, ties and normalization.
    pub(crate) fn singleton_routes(
        logits: &MxArray,
        top: usize,
    ) -> Result<Option<(MxArray, MxArray)>> {
        if top != 10
            || row_count(logits, Some(512))?.is_none()
            || !matches!(logits.dtype()?, DType::BFloat16 | DType::Float32)
            || !backend()
        {
            return Ok(None);
        }
        let (mut ids, mut scores) = (std::ptr::null_mut(), std::ptr::null_mut());
        let ok = unsafe {
            mlx_sys::mlx_qwen4_prefill_routes(logits.as_raw_ptr(), &mut ids, &mut scores)
        };
        if !ok {
            return Err(Error::from_reason(
                "Qwen4 row-batched routing kernel failed",
            ));
        }
        Ok(Some((
            MxArray::from_handle(ids, "row-batched route IDs")?,
            MxArray::from_handle(scores, "row-batched route scores")?,
        )))
    }
}
