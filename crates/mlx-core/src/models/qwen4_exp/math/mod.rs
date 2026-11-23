//! Math primitives, reference fallbacks, and backend dispatch for `qwen4_exp`.

mod cuda;
mod metal;
#[cfg(test)]
mod tests;

use crate::array::{DType, MxArray};
use crate::nn::Activations;
use napi::{Error, Result};

pub(super) use cuda::{cuda_moe_prefill, cuda_moe_routed_shared_few};
pub(super) use metal::{
    QsaKernel, complete_gdn, expert_tiles, prefill_indirect, qsa_attention, route_sort,
    routed_experts, routed_shared_experts, rows,
};
#[cfg(test)]
pub(super) use metal::{qsa_dispatch, qsa_fail_pipeline_probes, qsa_pipeline_threads};

/// The expert kernel a prompt window runs, chosen from the window's
/// assignment count and the model's expert count alone. Every slot group of
/// the window runs the window's kernel, whatever the slot bank's row count
/// or the group's size, so a window split over slot banks gives the bits of
/// the whole-bank run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ExpertKernel {
    /// Sorted assignments through the fused NAX expert tiles.
    Tiles,
    /// Sorted assignments through MLX's gather matrix-vector kernel.
    Gather,
    /// Router-order assignments through the per-row GEMV.
    Rows,
}

impl ExpertKernel {
    pub(super) fn for_window(rows: usize, experts: usize) -> Self {
        if experts > 0 && (256..=65536).contains(&rows) && rows / experts >= 4 {
            Self::Tiles
        } else if rows > 80 {
            Self::Gather
        } else {
            Self::Rows
        }
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
        && let Some(out) = metal::sorted_combine(values, scores, inverse, top)?
    {
        return Ok(out);
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
        return metal::gated_delta_kernel(&q, &k, &v, &decay, &beta, state)?
            .ok_or_else(|| Error::from_reason("Qwen4 fused recurrent kernel failed"));
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
        return metal::gated_delta_kernel(q, k, v, decay, beta, state)?
            .ok_or_else(|| Error::from_reason("Qwen4 fused recurrent sequence failed"));
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
    let use_metal = crate::engine::persistence::compiled_forward_backend_available();
    let use_cuda = super::cuda_qwen4_fast_enabled();
    if !use_metal && !use_cuda {
        return Ok(None);
    }
    let old = match state {
        Some(s) => s.clone(),
        None => MxArray::zeros(&[3, 10240], Some(DType::BFloat16))?,
    };
    // qwen4 semantics: L2-style norm (`mean_eps = false`) and FP32 `beta`
    // (`beta_input_dtype = false`), matching `math::l2` + `math::gdn_gates`.
    let prepared = crate::kernels::gdn::prepare(qkv, a, b, conv, &old, scale, dt, false, false)?;
    let Some([q, k, v, decay, beta, history]) = prepared else {
        return Ok(None);
    };
    *state = Some(history);
    Ok(Some([q, k, v, decay, beta]))
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
        if let Some((out, history)) = metal::window_conv(x, &old, weight)? {
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
    crate::kernels::qwen4::common::gdn_epilogue(out, z, weight, eps)
}

pub fn norm(x: &MxArray, w: &MxArray, group: usize, eps: f64, centered: bool) -> Result<MxArray> {
    if crate::engine::persistence::compiled_forward_backend_available() {
        return crate::kernels::qwen4::common::norm(x, w, group, eps, centered);
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

/// YaRN of a target past its native window: Splash
/// `rotaryInverseFrequencies` and the `attentionFactor` cos and sin take.
#[derive(Clone, Debug, PartialEq)]
pub struct Yarn {
    inverse: std::sync::Arc<[f32]>,
    factor: f32,
}

impl Yarn {
    /// `None` without YaRN.
    pub fn of(config: &super::config::Config) -> Result<Option<Self>> {
        let scaling = &config.rope_scaling;
        if !scaling.yarn() {
            return Ok(None);
        }
        let inverse = crate::nn::rope_scaling::rotary_inverse_frequencies(
            config.rope_theta() as f32,
            (config.rope_dims() / 2) as u32,
            scaling,
        )?;
        Ok(Some(Self {
            inverse: inverse.into(),
            factor: scaling.attention_factor(),
        }))
    }
}

/// `cos(angles)` and `sin(angles)` in f32, each times the YaRN attention
/// factor (Splash `target_rope_pair`), cast to `dtype`.
fn cos_sin(angles: &MxArray, yarn: Option<&Yarn>, dtype: DType) -> Result<(MxArray, MxArray)> {
    let (cos, sin) = (angles.cos()?, angles.sin()?);
    let (cos, sin) = match yarn {
        Some(yarn) => (
            cos.mul_scalar(f64::from(yarn.factor))?,
            sin.mul_scalar(f64::from(yarn.factor))?,
        ),
        None => (cos, sin),
    };
    Ok((cos.astype(dtype)?, sin.astype(dtype)?))
}

/// Split-half partial RoPE. Text positions coincide on the three MRoPE axes.
pub fn rope(
    x: &MxArray,
    position: usize,
    dims: usize,
    theta: f64,
    yarn: Option<&Yarn>,
) -> Result<MxArray> {
    if yarn.is_none()
        && let Some(out) = cuda::rope_fast(x, position, dims, theta)?
    {
        return Ok(out);
    }
    mrope(x, [position as i64; 3], dims, theta, [0; 3], true, yarn)
}

pub fn mrope(
    x: &MxArray,
    positions: [i64; 3],
    dims: usize,
    theta: f64,
    sections: [usize; 3],
    interleaved: bool,
    yarn: Option<&Yarn>,
) -> Result<MxArray> {
    let shape = x.shape()?;
    let (&width, axes) = shape
        .split_last()
        .ok_or_else(|| Error::from_reason("Qwen4 rotary input must have a feature dimension"))?;
    let axis = axes.len();
    let half = dims as i64 / 2;
    let angles = rotary_angles(positions, dims, theta, sections, interleaved, yarn);
    let angles = MxArray::from_float32(&angles, &[half])?;
    let (cos, sin) = cos_sin(&angles, yarn, x.dtype()?)?;
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
    yarn: Option<&Yarn>,
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
            match yarn {
                Some(yarn) => (positions[axis] as f64 * f64::from(yarn.inverse[j])) as f32,
                None => (positions[axis] as f64 / theta.powf((2 * i) as f64 / dims as f64)) as f32,
            }
        })
        .collect();
    angles
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
        yarn: Option<&Yarn>,
    ) -> Result<MxArray> {
        let key = RotaryKey {
            positions,
            dims,
            theta: theta.to_bits(),
            sections,
            interleaved,
            dtype: x.dtype()?,
        };
        let (cos, sin) = self.tables(key, yarn)?;
        apply_rotary_window(x, dims, &cos, &sin)
    }

    /// The YaRN of a decoder is fixed for its life, so it is not in the key.
    fn tables(&mut self, key: RotaryKey, yarn: Option<&Yarn>) -> Result<(MxArray, MxArray)> {
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
            yarn,
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
    yarn: Option<&Yarn>,
) -> Result<(MxArray, MxArray)> {
    let half = dims as i64 / 2;
    let angles: Vec<f32> = positions
        .iter()
        .flat_map(|&p| rotary_angles(p, dims, theta, sections, interleaved, yarn))
        .collect();
    let angles = MxArray::from_float32(&angles, &[1, 1, positions.len() as i64, half])?;
    cos_sin(&angles, yarn, dtype)
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
        metal::prefill_ple_conv(&input, &weight.reshape(&[width, 4])?)?
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
    if let Some(routes) = cuda::singleton_routes(logits, top)? {
        return Ok(Some(routes));
    }
    metal::singleton_routes(logits, top)
}

pub(super) fn prefill_routes(logits: &MxArray, top: usize) -> Result<Option<(MxArray, MxArray)>> {
    if let Some(routes) = cuda::prefill_routes(logits, top)? {
        return Ok(Some(routes));
    }
    metal::prefill_routes(logits, top)
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
        return crate::kernels::qwen4::common::gdn_gates(&a, &b, scale, dt);
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

/// Vectorized flat-KV sparse QSA attention for non-Metal hosts (Windows CUDA
/// Phase 0; Phase 2 hook point: replace with fused `mlx_qwen4_qsa_attention_cuda`
/// once the CUDA kernel lands).
///
/// `q` is `[1, nh, t, hd]`, `keys`/`values` are `[1, kh, total_kv, hd]`,
/// `selected` is `[1, t, topk]` (sorted block ids with `-1` for unused picks),
/// and `query_ends` is `[1, t]` (exclusive end `base + j + 1`). Each query row
/// attends to all tokens in its selected complete blocks plus its incomplete
/// tail `[complete * block_size, query_end)`, matching `attention_projected`.
#[allow(clippy::too_many_arguments)]
pub(super) fn qsa_attention_flat(
    q: &MxArray,
    keys: &MxArray,
    values: &MxArray,
    selected: &MxArray,
    query_ends: &MxArray,
    base: usize,
    scale: f64,
    block_size: usize,
) -> Result<MxArray> {
    const Q_CHUNK: usize = 32;
    let q_shape = q.shape()?;
    let k_shape = keys.shape()?;
    let sel_shape = selected.shape()?;
    let (nh, t, hd) = (q_shape[1], q_shape[2] as usize, q_shape[3]);
    let kh = k_shape[1];
    let topk = sel_shape[2];
    let ratio = block_size.max(1);
    let ratio_i64 = ratio as i64;
    let tail_slots = ratio - 1;
    let s_len = topk * ratio_i64 + tail_slots as i64;

    let k_t = keys.squeeze(Some(&[0]))?.transpose(Some(&[1, 0, 2]))?;
    let v_t = values.squeeze(Some(&[0]))?.transpose(Some(&[1, 0, 2]))?;
    let zero = MxArray::scalar_int(0)?;
    let ratio_scalar = MxArray::scalar_int(ratio as i32)?;
    let block_offsets = MxArray::arange(0.0, ratio as f64, None, Some(DType::Int32))?
        .reshape(&[1, 1, ratio_i64])?;
    let tail_offsets = if tail_slots > 0 {
        Some(
            MxArray::arange(0.0, tail_slots as f64, None, Some(DType::Int32))?
                .reshape(&[1, tail_slots as i64])?,
        )
    } else {
        None
    };

    let mut chunks = Vec::with_capacity(t.div_ceil(Q_CHUNK));
    for start in (0..t).step_by(Q_CHUNK) {
        let len = Q_CHUNK.min(t - start);
        let c_len = len as i64;
        let sel_c = selected
            .slice_axis(1, start as i64, (start + len) as i64)?
            .squeeze(Some(&[0]))?;
        let block_valid = sel_c
            .greater_equal(&zero)?
            .expand_dims(-1)?
            .broadcast_to(&[c_len, topk, ratio_i64])?
            .reshape(&[c_len, topk * ratio_i64])?;
        let block_safe = sel_c.maximum(&zero)?;
        let block_toks_raw = block_safe
            .mul(&ratio_scalar)?
            .expand_dims(-1)?
            .add(&block_offsets)?
            .reshape(&[c_len, topk * ratio_i64])?;
        let block_toks = block_valid.where_(&block_toks_raw, &zero)?;

        let (tok_ids, valid) = if let Some(tail_off) = &tail_offsets {
            let ends_c = query_ends
                .slice_axis(1, start as i64, (start + len) as i64)?
                .squeeze(Some(&[0]))?
                .expand_dims(-1)?;
            let tail_starts: Vec<i32> = (start..start + len)
                .map(|j| (((base + j + 1) / ratio) * ratio) as i32)
                .collect();
            let tail_starts = MxArray::from_int32(&tail_starts, &[c_len, 1])?;
            let tail_toks_raw = tail_starts.add(tail_off)?;
            let tail_valid = tail_toks_raw.less(&ends_c)?;
            let tail_toks = tail_valid.where_(&tail_toks_raw, &zero)?;
            (
                MxArray::concatenate(&block_toks, &tail_toks, 1)?,
                MxArray::concatenate(&block_valid, &tail_valid, 1)?,
            )
        } else {
            (block_toks, block_valid)
        };

        let flat_ids = tok_ids.reshape(&[-1])?;
        let k_sel = k_t
            .take(&flat_ids, 0)?
            .reshape(&[c_len, s_len, kh, hd])?
            .transpose(Some(&[0, 2, 1, 3]))?;
        let v_sel = v_t
            .take(&flat_ids, 0)?
            .reshape(&[c_len, s_len, kh, hd])?
            .transpose(Some(&[0, 2, 1, 3]))?;
        let q_c = q
            .slice_axis(2, start as i64, (start + len) as i64)?
            .squeeze(Some(&[0]))?
            .transpose(Some(&[1, 0, 2]))?
            .expand_dims(2)?;
        let mask = valid.reshape(&[c_len, 1, 1, s_len])?;
        let out_c =
            crate::array::scaled_dot_product_attention(&q_c, &k_sel, &v_sel, scale, Some(&mask))?;
        let out_c = out_c
            .squeeze(Some(&[2]))?
            .transpose(Some(&[1, 0, 2]))?
            .expand_dims(0)?;
        chunks.push(out_c);
    }
    if chunks.len() == 1 {
        Ok(chunks.remove(0))
    } else {
        let out = MxArray::concatenate_many(chunks.iter().collect(), Some(2))?;
        debug_assert_eq!(out.shape_at(1).ok(), Some(nh));
        Ok(out)
    }
}

pub(super) fn sigmoid_mul(gate: &MxArray, value: &MxArray) -> Result<MxArray> {
    Activations::sigmoid_mul_compiled(gate, value)
}
