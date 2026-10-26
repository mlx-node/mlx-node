//! Small DeepSeek-V4.1 (1017) ops, each a transcription of the reference
//! `model.py` / `kernel.py` math (pinned by the fixture's `ops.json`).

use napi::{Error, Result};

use super::config::YarnConfig;
use crate::array::{DType, MxArray};
use crate::nn::Activations;

fn dims(x: &MxArray) -> Result<Vec<i64>> {
    Ok(x.shape()?.as_ref().to_vec())
}

fn scalar(v: f32) -> Result<MxArray> {
    MxArray::from_float32(&[v], &[1])
}

fn int_scalar(v: i32) -> Result<MxArray> {
    MxArray::from_int32(&[v], &[1])
}

/// `x[..., lo..hi]` on the last axis.
fn last(x: &MxArray, lo: i64, hi: i64) -> Result<MxArray> {
    let d = dims(x)?;
    let mut starts = vec![0; d.len()];
    let mut stops = d.clone();
    starts[d.len() - 1] = lo;
    stops[d.len() - 1] = hi;
    x.slice(&starts, &stops)
}

/// Rotary inverse frequencies `f_i` in f32, `model.py:370-388`; `yarn` None is
/// the ratio-0 (window-only) path: base theta, no YaRN.
pub fn inv_freqs(dim: usize, theta: f32, yarn: Option<&YarnConfig>) -> Vec<f32> {
    let half = dim / 2;
    let mut f: Vec<f32> = (0..half)
        // torch's f32 `base ** e` is the correctly rounded power (checked on
        // both thetas); libm powf is not, so round the f64 power
        .map(|i| {
            let e = (2 * i) as f32 / dim as f32;
            1.0 / f64::from(theta).powf(f64::from(e)) as f32
        })
        .collect();
    if let Some(y) = yarn.filter(|y| y.original_seq_len > 0) {
        let base = f64::from(theta);
        let corrected = |rot: f32| {
            dim as f64
                * (y.original_seq_len as f64 / (f64::from(rot) * 2.0 * std::f64::consts::PI)).ln()
                / (2.0 * base.ln())
        };
        let low = corrected(y.beta_fast).floor().max(0.0);
        let high = corrected(y.beta_slow).ceil().min((dim - 1) as f64);
        let span = (high - low).max(1e-3) as f32;
        for (i, fi) in f.iter_mut().enumerate() {
            let ramp = ((i as f32 - low as f32) / span).clamp(0.0, 1.0);
            let smooth = 1.0 - ramp;
            *fi = *fi / y.factor * (1.0 - smooth) + *fi * smooth;
        }
    }
    f
}

/// The reference angle `fl32(p * f_i)` (`torch.outer(arange, freqs)` in f32).
pub fn rope_angle(pos: u64, f: f32) -> f32 {
    pos as f32 * f
}

/// cos / sin tables `[positions, dim/2]` (f32) of the reference angles, each
/// value the correctly rounded cos / sin of the f32 angle. A compressed row
/// `j` of ratio `r` takes position `j * r`.
pub fn rope_tables(inv: &[f32], positions: &[u64]) -> Result<(MxArray, MxArray)> {
    let mut cos = Vec::with_capacity(positions.len() * inv.len());
    let mut sin = Vec::with_capacity(positions.len() * inv.len());
    for &p in positions {
        for &f in inv {
            let a = f64::from(rope_angle(p, f));
            cos.push(a.cos() as f32);
            sin.push(a.sin() as f32);
        }
    }
    let shape = [positions.len() as i64, inv.len() as i64];
    Ok((
        MxArray::from_float32(&cos, &shape)?,
        MxArray::from_float32(&sin, &shape)?,
    ))
}

/// Interleaved RoPE on the last `rd` dims of `x` (`[s, d]` or `[s, h, d]`),
/// adjacent pairs as complex numbers, f32 math, result in `x`'s dtype;
/// `inverse` conjugates the rotation (`apply_rotary_emb`).
pub fn apply_partial_rope(
    x: &MxArray,
    rd: i64,
    cos: &MxArray,
    sin: &MxArray,
    inverse: bool,
) -> Result<MxArray> {
    let d = dims(x)?;
    let (s, width) = (d[0], d[d.len() - 1]);
    let half = rd / 2;
    let mut tshape = vec![s];
    tshape.extend(std::iter::repeat_n(1, d.len() - 2));
    tshape.extend([half, 1]);
    let c = cos.reshape(&tshape)?;
    let mut sn = sin.reshape(&tshape)?;
    if inverse {
        sn = sn.negative()?;
    }
    let mut pshape = d[..d.len() - 1].to_vec();
    pshape.extend([half, 2]);
    let r = last(x, width - rd, width)?
        .astype(DType::Float32)?
        .reshape(&pshape)?;
    let (a, b) = (last(&r, 0, 1)?, last(&r, 1, 2)?);
    let re = a.mul(&c)?.sub(&b.mul(&sn)?)?;
    let im = a.mul(&sn)?.add(&b.mul(&c)?)?;
    let mut rshape = d[..d.len() - 1].to_vec();
    rshape.push(rd);
    let rot = MxArray::concatenate(&re, &im, -1)?
        .reshape(&rshape)?
        .astype(x.dtype()?)?;
    if rd == width {
        return Ok(rot);
    }
    MxArray::concatenate(&last(x, 0, width - rd)?, &rot, -1)
}

/// `2^ceil(log2(r))` for positive f32 `r`, exact (`kernel.py:22-38`
/// `fast_round_scale`): a log2 guess corrected by exact compares against a
/// table of powers of two.
pub(crate) fn pow2_ceil(r: &MxArray) -> Result<MxArray> {
    // index k + 149 holds 2^k, k = -149..=128 (2^128 = inf as in the bit form)
    let table: Vec<f32> = (0u32..=277)
        .map(|i| {
            if i < 23 {
                f32::from_bits(1 << i)
            } else {
                f32::from_bits((i - 22) << 23)
            }
        })
        .collect();
    let table = MxArray::from_float32(&table, &[278])?;
    let g = r
        .log2()?
        .ceil()?
        .clip(Some(-148.0), Some(127.0))?
        .add_scalar(149.0)?
        .astype(DType::Int32)?;
    let below = table.take(&g, 0)?.less(r)?.astype(DType::Int32)?;
    let above = table
        .take(&g.sub(&int_scalar(1)?)?, 0)?
        .greater_equal(r)?
        .astype(DType::Int32)?;
    table.take(&g.add(&below)?.sub(&above)?, 0)
}

/// Round to the e2m1 grid, ties to even, sign kept (`kernel_stub.e2m1_round`;
/// `-0.0` gives `+0.0` like `torch.sign`).
pub(crate) fn e2m1_round(v: &MxArray) -> Result<MxArray> {
    let grid = MxArray::from_float32(&[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], &[8])?;
    let mag = v.abs()?;
    let mut idx = int_scalar(0)?;
    for t in [0.25, 1.25, 2.5, 5.0] {
        idx = idx.add(&mag.greater(&scalar(t)?)?.astype(DType::Int32)?)?;
    }
    for t in [0.75, 1.75, 3.5] {
        idx = idx.add(&mag.greater_equal(&scalar(t)?)?.astype(DType::Int32)?)?;
    }
    let q = grid.take(&idx, 0)?;
    v.less(&scalar(0.0)?)?.where_(&q.negative()?, &q)
}

/// `x` as f32 blocks `[..., n / bs, bs]` and the shape to restore.
fn blocks(x: &MxArray, bs: i64) -> Result<(MxArray, Vec<i64>)> {
    let d = dims(x)?;
    let n = d[d.len() - 1];
    if n % bs != 0 {
        return Err(Error::from_reason(format!(
            "fake quant: last dim {n} is not a multiple of {bs}"
        )));
    }
    let mut b = d[..d.len() - 1].to_vec();
    b.extend([n / bs, bs]);
    Ok((x.astype(DType::Float32)?.reshape(&b)?, d))
}

/// FP8 e4m3 activation fake quant with ue8m0 (power-of-two) scales per `bs`
/// (`act_quant(..., scale_fmt="ue8m0", inplace=True)`); result in `x`'s dtype.
pub fn fp8_roundtrip(x: &MxArray, bs: i64) -> Result<MxArray> {
    let (xb, d) = blocks(x, bs)?;
    let amax = xb
        .abs()?
        .max(Some(&[-1]), Some(true))?
        .clip(Some(1e-4), None)?;
    let s = pow2_ceil(&amax.mul_scalar(f64::from((1.0f64 / 448.0) as f32))?)?;
    xb.div(&s)?
        .clip(Some(-448.0), Some(448.0))?
        .to_fp8()?
        .from_fp8(DType::Float32)?
        .mul(&s)?
        .reshape(&d)?
        .astype(x.dtype()?)
}

/// FP4 e2m1 activation fake quant per `bs`: e4m3 scales (`amax / 6`,
/// saturating at 448) or ue8m0 scales (`fp4_act_quant`); result in `x`'s dtype.
pub fn fp4_roundtrip(x: &MxArray, bs: i64, e4m3_scale: bool) -> Result<MxArray> {
    let (xb, d) = blocks(x, bs)?;
    let amax = xb.abs()?.max(Some(&[-1]), Some(true))?;
    let s = if e4m3_scale {
        amax.clip(Some(6.0 * 2f64.powi(-9)), None)?
            .div_scalar(6.0)?
            .clip(None, Some(448.0))?
            .to_fp8()?
            .from_fp8(DType::Float32)?
    } else {
        let amax = amax.clip(Some(6.0 * 2f64.powi(-126)), None)?;
        pow2_ceil(&amax.mul_scalar(f64::from((1.0f64 / 6.0) as f32))?)?
    };
    e2m1_round(&xb.div(&s)?.clip(Some(-6.0), Some(6.0))?)?
        .mul(&s)?
        .reshape(&d)?
        .astype(x.dtype()?)
}

/// Router scores `sqrt(softplus(logits))` in f32 (torch softplus: `x` above
/// 20). `gate_temp` is 1 in the 1017 config, so it is not applied.
pub fn router_scores(logits: &MxArray) -> Result<MxArray> {
    let x = logits.astype(DType::Float32)?;
    let soft = x.exp()?.log1p()?;
    x.greater(&scalar(20.0)?)?.where_(&x, &soft)?.sqrt()
}

/// Top-k by `biased` (scores + bias), ties to the lowest index, weights from
/// the unbiased `scores`: `/ (sum + 1e-20) * route_scale` when `norm` and
/// k > 1. Row-major `[rows, n]` in, `[rows, k]` ids and weights out.
pub fn route_select(
    biased: &[f32],
    scores: &[f32],
    n: usize,
    k: usize,
    route_scale: f32,
    norm: bool,
) -> (Vec<u32>, Vec<f32>) {
    let rows = biased.len() / n;
    let (mut ids, mut weights) = (Vec::with_capacity(rows * k), Vec::with_capacity(rows * k));
    let mut order: Vec<u32> = Vec::with_capacity(n);
    for r in 0..rows {
        let b = &biased[r * n..(r + 1) * n];
        order.clear();
        order.extend(0..n as u32);
        order.sort_by(|&i, &j| b[j as usize].total_cmp(&b[i as usize]).then(i.cmp(&j)));
        let w: Vec<f32> = order[..k]
            .iter()
            .map(|&e| scores[r * n + e as usize])
            .collect();
        let sum = w.iter().fold(0.0f32, |a, &x| a + x) + 1e-20;
        ids.extend_from_slice(&order[..k]);
        weights.extend(
            w.iter()
                .map(|&x| (if norm && k > 1 { x / sum } else { x }) * route_scale),
        );
    }
    (ids, weights)
}

/// `hc_split_sinkhorn` (`kernel.py:421-460`): mixes `[n, (2 + hc) * hc]` ->
/// pre `[n, hc]`, post `[n, hc]`, comb `[n, hc, hc]` (`comb[j, k]` = from copy
/// j to copy k), exactly `iters` normalization rounds, no early exit.
pub fn hc_split_sinkhorn(
    mixes: &MxArray,
    scale: &MxArray,
    base: &MxArray,
    hc: i64,
    iters: usize,
    eps: f32,
) -> Result<(MxArray, MxArray, MxArray)> {
    let m = mixes.astype(DType::Float32)?;
    let n = dims(&m)?[0];
    let part = |i: i64, lo: i64, hi: i64| -> Result<MxArray> {
        last(&m, lo, hi)?
            .mul(&last(scale, i, i + 1)?)?
            .add(&last(base, lo, hi)?)
    };
    let pre = Activations::sigmoid(&part(0, 0, hc)?)?.add_scalar(f64::from(eps))?;
    let post = Activations::sigmoid(&part(1, hc, 2 * hc)?)?.mul_scalar(2.0)?;
    let c = part(2, 2 * hc, (2 + hc) * hc)?.reshape(&[n, hc, hc])?;
    let e = c.sub(&c.max(Some(&[-1]), Some(true))?)?.exp()?;
    let mut comb = e
        .div(&e.sum(Some(&[-1]), Some(true))?)?
        .add_scalar(f64::from(eps))?;
    let norm = |x: &MxArray, axis: i32| -> Result<MxArray> {
        x.div(
            &x.sum(Some(&[axis]), Some(true))?
                .add_scalar(f64::from(eps))?,
        )
    };
    comb = norm(&comb, -2)?;
    for _ in 1..iters {
        comb = norm(&norm(&comb, -1)?, -2)?;
    }
    Ok((pre, post, comb))
}

/// `Block.hc_mixes`: x `[n, hc, d]`, hc_fn `[(2 + hc) * hc, hc * d]` (f32);
/// one RMS statistic over the flattened `hc * d` stream per row.
#[allow(clippy::too_many_arguments)]
pub fn hc_mixes(
    x: &MxArray,
    hc_fn: &MxArray,
    scale: &MxArray,
    base: &MxArray,
    norm_eps: f32,
    hc: i64,
    iters: usize,
    hc_eps: f32,
) -> Result<(MxArray, MxArray, MxArray)> {
    let d = dims(x)?;
    let xf = x.astype(DType::Float32)?.reshape(&[d[0], d[1] * d[2]])?;
    let rs = xf
        .square()?
        .mean(Some(&[-1]), Some(true))?
        .add_scalar(f64::from(norm_eps))?
        .sqrt()?
        .reciprocal()?;
    let mixes = xf
        .matmul(&hc_fn.astype(DType::Float32)?.transpose(Some(&[1, 0]))?)?
        .mul(&rs)?;
    hc_split_sinkhorn(&mixes, scale, base, hc, iters, hc_eps)
}

/// `Block.hc_pre`: `sum_j pre[j] * x[j]` in f32, back to `x`'s dtype.
pub fn hc_pre(x: &MxArray, pre: &MxArray) -> Result<MxArray> {
    let d = dims(x)?;
    pre.reshape(&[d[0], 1, d[1]])?
        .matmul(&x.astype(DType::Float32)?)?
        .reshape(&[d[0], d[2]])?
        .astype(x.dtype()?)
}

/// `Block.hc_post`: `out[k] = post[k] * x + sum_j comb[j, k] * residual[j]`
/// in f32, in `x`'s dtype. x `[n, d]`, residual `[n, hc, d]`.
pub fn hc_post(x: &MxArray, residual: &MxArray, post: &MxArray, comb: &MxArray) -> Result<MxArray> {
    let d = dims(residual)?;
    let xf = x.astype(DType::Float32)?.reshape(&[d[0], 1, d[2]])?;
    let mixed = comb
        .transpose(Some(&[0, 2, 1]))?
        .matmul(&residual.astype(DType::Float32)?)?;
    post.reshape(&[d[0], d[1], 1])?
        .mul(&xf)?
        .add(&mixed)?
        .astype(x.dtype()?)
}

/// SwiGLU clamps in f32: up to `[-limit, limit]`, gate to `<= limit`; a limit
/// of 0 disables both (`Expert.forward`).
pub fn swiglu_clamp(gate: &MxArray, up: &MxArray, limit: f32) -> Result<(MxArray, MxArray)> {
    let (g, u) = (gate.astype(DType::Float32)?, up.astype(DType::Float32)?);
    if limit <= 0.0 {
        return Ok((g, u));
    }
    let l = f64::from(limit);
    Ok((g.clip(None, Some(l))?, u.clip(Some(-l), Some(l))?))
}

/// `silu(gate) * up` after the clamps, times the routing `weights` when given
/// (applied before `w2`, in f32), in `dtype`.
pub fn swiglu(
    gate: &MxArray,
    up: &MxArray,
    limit: f32,
    weights: Option<&MxArray>,
    dtype: DType,
) -> Result<MxArray> {
    let (g, u) = swiglu_clamp(gate, up, limit)?;
    let mut y = Activations::silu(&g)?.mul(&u)?;
    if let Some(w) = weights {
        y = w.astype(DType::Float32)?.mul(&y)?;
    }
    y.astype(dtype)
}

/// `Engram.forward` after `wkv`: x `[n, hc, d]`, key `[n, hc * d]`, value
/// `[n, d]`, weight `[hc, d]` (the q * k product stored as `engram_gate`).
/// Gate per (row, hc copy) = sigmoid(signed sqrt of the normalized dot).
pub fn engram_gate(
    x: &MxArray,
    key: &MxArray,
    value: &MxArray,
    weight: &MxArray,
    eps: f32,
) -> Result<MxArray> {
    let d = dims(x)?;
    let h = x.astype(DType::Float32)?;
    let k = key.astype(DType::Float32)?.reshape(&d)?;
    let rms = |t: &MxArray| -> Result<MxArray> {
        t.square()?
            .mean(Some(&[-1]), None)?
            .add_scalar(f64::from(eps))?
            .sqrt()?
            .reciprocal()
    };
    let rstd = rms(&h)?.mul(&rms(&k)?)?;
    let dot = h
        .mul(&weight.astype(DType::Float32)?)?
        .mul(&k)?
        .sum(Some(&[-1]), None)?
        .mul(&rstd)?
        .mul_scalar((d[2] as f64).powf(-0.5))?;
    let a = dot.abs()?.clip(Some(1e-6), None)?.sqrt()?;
    let gate = Activations::sigmoid(&dot.less(&scalar(0.0)?)?.where_(&a.negative()?, &a)?)?;
    let v = value.astype(DType::Float32)?.reshape(&[d[0], 1, d[2]])?;
    h.add(&gate.reshape(&[d[0], d[1], 1])?.mul(&v)?)?
        .astype(x.dtype()?)
}

#[cfg(test)]
mod tests;
