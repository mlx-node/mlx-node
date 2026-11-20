//! Hy4 small ops, all math in f32 (llama.cpp PR #32517 graph):
//! hyper-connection mix / post / head with no comb term, the sigmoid router
//! on DS `route_select`, and the vocab head in row chunks.

use napi::{Error, Result};

use super::config::{HcConfig, MoeConfig};
use super::weights::HcWeights;
use crate::array::{DType, MxArray};
use crate::models::deepseek_v41::attention::linear;
use crate::models::deepseek_v41::ops::{hc_pre, route_select};
use crate::models::deepseek_v41::store::{Dense, Weight};
use crate::nn::Activations;

/// Vocab rows per f32 head chunk (16384 x 3072 f32 = 192 MiB).
pub const HEAD_CHUNK: i64 = 16384;

/// Calls of at most this many rows evaluate two head chunks per eval.
pub const HEAD_PAIR_ROWS: i64 = 16;

fn bad(msg: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("Hy4 ops: {msg}"))
}

fn dims(x: &MxArray) -> Result<Vec<i64>> {
    Ok(x.shape()?.as_ref().to_vec())
}

/// `[n, hc, d]` of a stream tensor.
fn streams(x: &MxArray) -> Result<[i64; 3]> {
    match dims(x)?.as_slice() {
        &[n, hc, d] => Ok([n, hc, d]),
        s => Err(bad(format!("streams must be [n, hc, d], got {s:?}"))),
    }
}

/// `x[.., lo..hi]` on the last axis.
fn last(x: &MxArray, lo: i64, hi: i64) -> Result<MxArray> {
    let d = dims(x)?;
    let mut starts = vec![0; d.len()];
    let mut stops = d.clone();
    if let (Some(s), Some(e)) = (starts.last_mut(), stops.last_mut()) {
        *s = lo;
        *e = hi;
    }
    x.slice(&starts, &stops)
}

fn check_len(a: &MxArray, want: i64, what: &str) -> Result<()> {
    let got = dims(a)?;
    if got.as_slice() != [want] {
        return Err(bad(format!("{what} must be [{want}], got {got:?}")));
    }
    Ok(())
}

/// `(flat · fnᵀ) · rsqrt(mean(flat²) + eps)` with `flat = x [n, hc·d]`:
/// one RMS statistic over all streams of a row. Out `[n, fn rows]` f32.
pub fn hc_mixes(x: &MxArray, weight: &MxArray, norm_eps: f32) -> Result<MxArray> {
    let [n, hc, d] = streams(x)?;
    if weight.ndim()? != 2 || weight.shape_at(1)? != hc * d {
        return Err(bad(format!(
            "hc fn {:?} does not take {hc} streams of {d}",
            dims(weight)?
        )));
    }
    let xf = x.astype(DType::Float32)?.reshape(&[n, hc * d])?;
    let rs = xf
        .square()?
        .mean(Some(&[-1]), Some(true))?
        .add_scalar(f64::from(norm_eps))?
        .sqrt()?
        .reciprocal()?;
    xf.matmul(&weight.astype(DType::Float32)?.transpose(Some(&[1, 0]))?)?
        .mul(&rs)
}

/// `σ(m[.., lo..lo+hc] · scale[si] + base[lo..lo+hc])`.
fn sigmoid_part(m: &MxArray, w: &HcWeights, lo: i64, hc: i64, si: i64) -> Result<MxArray> {
    let s = last(&w.scale.astype(DType::Float32)?, si, si + 1)?;
    let b = last(&w.base.astype(DType::Float32)?, lo, lo + hc)?;
    Activations::sigmoid(&last(m, lo, lo + hc)?.mul(&s)?.add(&b)?)
}

/// Per-row stream weights `[n, hc]` of one sublayer.
pub struct HcGates {
    /// `σ(·) + eps`: how the streams are summed into the sublayer input.
    pub pre: MxArray,
    /// `magnitude · σ(·) + eps`: how the sublayer output joins each stream.
    pub post: MxArray,
}

/// Sublayer gates from `{prefix}_fn [2·hc, hc·d]`, `_scale [2]`,
/// `_base [2·hc]`: mixes `0..hc` give `pre`, `hc..2·hc` give `post`.
pub fn hc_mix(x: &MxArray, w: &HcWeights, norm_eps: f32, cfg: &HcConfig) -> Result<HcGates> {
    let hc = streams(x)?[1];
    if hc != cfg.count as i64 || w.weight.shape_at(0)? != 2 * hc {
        return Err(bad(format!(
            "hc mix: {hc} streams, config {}, fn rows {}",
            cfg.count,
            w.weight.shape_at(0)?
        )));
    }
    check_len(&w.scale, 2, "hc scale")?;
    check_len(&w.base, 2 * hc, "hc base")?;
    let m = hc_mixes(x, &w.weight, norm_eps)?;
    let eps = f64::from(cfg.eps);
    Ok(HcGates {
        pre: sigmoid_part(&m, w, 0, hc, 0)?.add_scalar(eps)?,
        post: sigmoid_part(&m, w, hc, hc, 1)?
            .mul_scalar(f64::from(cfg.magnitude))?
            .add_scalar(eps)?,
    })
}

/// `Σ_c pre_c · x_c`: x `[n, hc, d]`, pre `[n, hc]` -> `[n, d]` in x's dtype.
pub fn hc_reduce(x: &MxArray, pre: &MxArray) -> Result<MxArray> {
    let [n, hc, _] = streams(x)?;
    if dims(pre)?.as_slice() != [n, hc] {
        return Err(bad(format!("hc pre {:?} for [{n}, {hc}]", dims(pre)?)));
    }
    hc_pre(x, &pre.astype(DType::Float32)?)
}

/// `res_c + post_c · y` (the comb is the identity): y `[n, d]`, res
/// `[n, hc, d]`, post `[n, hc]` -> `[n, hc, d]` in res's dtype.
pub fn hc_post_identity(y: &MxArray, res: &MxArray, post: &MxArray) -> Result<MxArray> {
    let [n, hc, d] = streams(res)?;
    if dims(y)?.as_slice() != [n, d] || dims(post)?.as_slice() != [n, hc] {
        return Err(bad(format!(
            "hc post: y {:?}, post {:?} for streams [{n}, {hc}, {d}]",
            dims(y)?,
            dims(post)?
        )));
    }
    let p = post.astype(DType::Float32)?.reshape(&[n, hc, 1])?;
    let yf = y.astype(DType::Float32)?.reshape(&[n, 1, d])?;
    res.astype(DType::Float32)?
        .add(&p.mul(&yf)?)?
        .astype(res.dtype()?)
}

/// Head stream weights `σ(mixes · scale + base) + eps` `[n, hc]` from
/// `output_hc_fn [hc, hc·d]`, `_scale [1]`, `_base [hc]`.
pub fn hc_head_pre(x: &MxArray, w: &HcWeights, norm_eps: f32, hc_eps: f32) -> Result<MxArray> {
    let hc = streams(x)?[1];
    if w.weight.shape_at(0)? != hc {
        return Err(bad(format!(
            "hc head fn has {} rows for {hc} streams",
            w.weight.shape_at(0)?
        )));
    }
    check_len(&w.scale, 1, "hc head scale")?;
    check_len(&w.base, hc, "hc head base")?;
    let m = hc_mixes(x, &w.weight, norm_eps)?;
    sigmoid_part(&m, w, 0, hc, 0)?.add_scalar(f64::from(hc_eps))
}

/// The streams collapsed for the head: `Σ_c pre_c · x_c` `[n, d]`.
pub fn hc_head(x: &MxArray, w: &HcWeights, norm_eps: f32, hc_eps: f32) -> Result<MxArray> {
    hc_reduce(x, &hc_head_pre(x, w, norm_eps, hc_eps)?)
}

/// Router scores `σ(x · gate_inpᵀ)` in f32, `[rows, n_experts]`.
pub fn router_scores(x: &MxArray, router: &Dense) -> Result<MxArray> {
    Activations::sigmoid(&linear(&x.astype(DType::Float32)?, router)?)
}

/// Top `n_used` experts per row by `score + bias` (ties to the lowest id),
/// weights from the unbiased scores `/ (Σ + 1e-20) · route_scale`.
/// Row-major `scores [rows, n_experts]`; `[rows, n_used]` ids and weights.
pub fn route_host(
    scores: &[f32],
    bias: Option<&[f32]>,
    moe: &MoeConfig,
) -> Result<(Vec<u32>, Vec<f32>)> {
    let n = moe.n_experts;
    if n == 0 || !scores.len().is_multiple_of(n) || moe.n_used == 0 || moe.n_used > n {
        return Err(bad(format!(
            "router: {} scores, {n} experts, {} used",
            scores.len(),
            moe.n_used
        )));
    }
    let biased: Vec<f32> = match bias {
        None => scores.to_vec(),
        Some(b) if b.len() == n => scores
            .iter()
            .zip(b.iter().cycle())
            .map(|(&s, &e)| s + e)
            .collect(),
        Some(b) => {
            return Err(bad(format!("router bias has {} entries for {n}", b.len())));
        }
    };
    Ok(route_select(
        &biased,
        scores,
        n,
        moe.n_used,
        moe.route_scale,
        moe.norm_topk,
    ))
}

/// `route_host` on scores read back from the GPU (one sync, which also
/// evaluates what `stats::defer` holds).
pub fn route(
    scores: &MxArray,
    bias: Option<&[f32]>,
    moe: &MoeConfig,
) -> Result<(Vec<u32>, Vec<f32>)> {
    if scores.ndim()? != 2 || scores.shape_at(1)? != moe.n_experts as i64 {
        return Err(bad(format!(
            "router scores {:?} for {} experts",
            dims(scores)?,
            moe.n_experts
        )));
    }
    let s = scores.astype(DType::Float32)?;
    super::stats::eval(&[&s], "hy_v4_flash router")?;
    let s = s.to_float32()?;
    route_host(s.as_ref(), bias, moe)
}

/// f32 logits `[s, vocab]` of `x [s, d]`; see `head_logits_chunked`.
pub fn head_logits(x: &MxArray, head: &Dense) -> Result<MxArray> {
    head_logits_chunked(x, head, HEAD_CHUNK)
}

/// f32 logits of `x [s, d]`. A `Qmm` head is one `quantized_matmul`; any
/// other head is cast to f32 `chunk` vocab rows at a time (`head_chunks`).
pub fn head_logits_chunked(x: &MxArray, head: &Dense, chunk: i64) -> Result<MxArray> {
    if chunk <= 0 {
        return Err(bad(format!("head chunk {chunk} must be > 0")));
    }
    let xf = x.astype(DType::Float32)?;
    if let Dense::Qmm { .. } = head {
        return head.mm_t(&xf);
    }
    head_chunks(head, chunk, xf.shape_at(0)?, |w| {
        xf.matmul(&w.transpose(Some(&[1, 0]))?)
    })
}

/// `part(f32 rows lo..hi of head)` per `chunk` vocab rows, concatenated
/// on axis 1. The parts are evaluated as they come, two per eval for calls
/// of at most `HEAD_PAIR_ROWS` rows, so at most two f32 chunks are alive.
pub fn head_chunks(
    head: &Dense,
    chunk: i64,
    rows: i64,
    part: impl Fn(&MxArray) -> Result<MxArray>,
) -> Result<MxArray> {
    if chunk <= 0 {
        return Err(bad(format!("head chunk {chunk} must be > 0")));
    }
    let per = if rows <= HEAD_PAIR_ROWS { 2 } else { 1 };
    let vocab = head.shape_at(0)?;
    let mut parts = Vec::new();
    let mut lo = 0;
    while lo < vocab {
        let hi = (lo + chunk).min(vocab);
        parts.push(part(&head.row_range(lo, hi)?.f32()?)?);
        let n = parts.len();
        if n % per == 0 || hi == vocab {
            let open = if n % per == 0 { per } else { n % per };
            let live: Vec<&MxArray> = parts[n - open..].iter().collect();
            super::stats::eval(&live, "hy_v4_flash head chunk")?;
        }
        lo = hi;
    }
    match parts.len() {
        0 => Err(bad("head has no vocab rows")),
        1 => Ok(parts.swap_remove(0)),
        _ => MxArray::concatenate_many(parts.iter().collect(), Some(1)),
    }
}

#[cfg(test)]
mod tests;
