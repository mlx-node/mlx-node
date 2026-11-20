//! Sparse sink attention over cached latent rows, gather-then-cast: each
//! query row reads only the key rows its index row lists, and only those
//! gathered rows are cast to f32. One key row is both K (all columns) and
//! V (its first `v_width` columns), as in the absorbed MLA form.
//!
//! Scores are f32, the row max is floored at -1e30 and the per-head sink
//! enters the denominator only (DS `sparse_sink_attention` rule,
//! `deepseek_v41/attention.rs:316-331`).

use napi::{Error, Result};

use crate::array::{DType, MxArray};

/// Byte budget of one row chunk (gathered keys + scores), DS
/// `SPARSE_CHUNK_BYTES` (`deepseek_v41/attention.rs:262`).
pub const SPARSE_CHUNK_BYTES: usize = 1 << 30;
/// Most query rows in one chunk.
const MAX_CHUNK_ROWS: usize = 256;
/// Floor of the score row max: a row with no listed key gives zeros, not NaN.
const MAX_FLOOR: f32 = -1e30;

fn bad(msg: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("Hy4 sparse attention: {msg}"))
}

fn dims(x: &MxArray) -> Result<Vec<i64>> {
    Ok(x.shape()?.as_ref().to_vec())
}

/// Bytes one query row needs: `kmax` gathered rows of `d` in the cache dtype
/// (2 B) and in f32, three `[h, kmax]` f32 score temporaries, q and o.
pub fn row_bytes(kmax: usize, h: usize, d: usize) -> usize {
    kmax * d * (2 + 4) + 4 * (3 * h * kmax + 2 * h * d)
}

/// Query rows per chunk so that `per_row` bytes per row stay within `budget`.
pub fn chunk_rows(per_row: usize, budget: usize) -> usize {
    (budget / per_row.max(1)).clamp(1, MAX_CHUNK_ROWS)
}

/// The cached rows a query gathers from.
#[derive(Clone, Copy)]
pub enum KeyRows<'a> {
    /// `[n, d]` rows, read by row id.
    Whole(&'a MxArray),
    /// Each row split over two planes `[n, d0]` and `[n, d - d0]` (the paged
    /// pool's K and V planes), read by slot id and joined.
    Split(&'a MxArray, &'a MxArray),
}

impl KeyRows<'_> {
    fn width(&self) -> Result<i64> {
        match self {
            Self::Whole(k) => match dims(k)?.as_slice() {
                &[_, d] => Ok(d),
                s => Err(bad(format!("keys {s:?} must be [n, d]"))),
            },
            Self::Split(k, v) => match (dims(k)?.as_slice(), dims(v)?.as_slice()) {
                (&[n, a], &[m, b]) if n == m => Ok(a + b),
                (a, b) => Err(bad(format!(
                    "key planes {a:?} and {b:?} must be [n, d0], [n, d1]"
                ))),
            },
        }
    }

    /// The rows listed by `gid [r]`, `[r, d]` in the cache dtype.
    fn gather(&self, gid: &MxArray) -> Result<MxArray> {
        match self {
            Self::Whole(k) => k.take(gid, 0),
            Self::Split(k, v) => MxArray::concatenate(&k.take(gid, 0)?, &v.take(gid, 0)?, 1),
        }
    }
}

/// `q [s, h, d]` attends the rows of `keys [n, d]` listed by `idx [s, kmax]`
/// (i32, `-1` = no key); out `[s, h, v_width]` f32. `sinks [h]`.
pub fn gather_sink_attention(
    q: &MxArray,
    keys: &MxArray,
    idx: &MxArray,
    v_width: i64,
    sinks: &MxArray,
    scale: f32,
) -> Result<MxArray> {
    gather_sink_attention_in(q, keys, idx, v_width, sinks, scale, SPARSE_CHUNK_BYTES)
}

/// `gather_sink_attention` over `keys` from either source; `idx` lists row
/// ids of `Whole` rows or slot ids of `Split` planes.
pub fn gather_sink_attention_rows(
    q: &MxArray,
    keys: KeyRows<'_>,
    idx: &MxArray,
    v_width: i64,
    sinks: &MxArray,
    scale: f32,
) -> Result<MxArray> {
    gather_rows_in(q, keys, idx, v_width, sinks, scale, SPARSE_CHUNK_BYTES)
}

/// `gather_sink_attention` with an explicit chunk budget.
pub(crate) fn gather_sink_attention_in(
    q: &MxArray,
    keys: &MxArray,
    idx: &MxArray,
    v_width: i64,
    sinks: &MxArray,
    scale: f32,
    budget: usize,
) -> Result<MxArray> {
    gather_rows_in(q, KeyRows::Whole(keys), idx, v_width, sinks, scale, budget)
}

/// The core of every entry above. With more than one chunk, every chunk is
/// evaluated before the next one is built.
fn gather_rows_in(
    q: &MxArray,
    keys: KeyRows<'_>,
    idx: &MxArray,
    v_width: i64,
    sinks: &MxArray,
    scale: f32,
    budget: usize,
) -> Result<MxArray> {
    let &[s, h, d] = dims(q)?.as_slice() else {
        return Err(bad(format!(
            "queries must be [s, h, d], got {:?}",
            dims(q)?
        )));
    };
    let kd = keys.width()?;
    if kd != d {
        return Err(bad(format!("keys of width {kd} for queries of width {d}")));
    }
    let &[is, kmax] = dims(idx)?.as_slice() else {
        return Err(bad(format!(
            "index must be [s, kmax], got {:?}",
            dims(idx)?
        )));
    };
    if is != s || kmax <= 0 || s <= 0 {
        return Err(bad(format!("index [{is}, {kmax}] for {s} query rows")));
    }
    if v_width <= 0 || v_width > d {
        return Err(bad(format!("value width {v_width} of key width {d}")));
    }
    if dims(sinks)?.as_slice() != [h] {
        return Err(bad(format!("sinks {:?} for {h} heads", dims(sinks)?)));
    }
    let idx = idx.astype(DType::Int32)?;
    let rows = chunk_rows(row_bytes(kmax as usize, h as usize, d as usize), budget) as i64;
    let neg_inf = MxArray::from_float32(&[f32::NEG_INFINITY], &[1])?;
    let floor = MxArray::from_float32(&[MAX_FLOOR], &[1])?;
    let zero = MxArray::from_int32(&[0], &[1])?;
    let zf = MxArray::from_float32(&[0.0], &[1])?;
    let sink = sinks.astype(DType::Float32)?.reshape(&[1, h, 1])?;
    let mut outs = Vec::new();
    let mut lo = 0;
    while lo < s {
        let hi = (lo + rows).min(s);
        let mut ic = idx.slice(&[lo, 0], &[hi, kmax])?;
        let mut qc = q.slice(&[lo, 0, 0], &[hi, h, d])?.astype(DType::Float32)?;
        // a 1-row chunk would take split-K (matmul.cpp:972), which rounds a
        // row unlike the same row of a taller chunk; pad it with a keyless row
        let pad = hi - lo == 1;
        if pad {
            let none = MxArray::from_int32(&vec![-1; kmax as usize], &[1, kmax])?;
            ic = MxArray::concatenate(&ic, &none, 0)?;
            let zq = MxArray::zeros(&[1, h, d], Some(DType::Float32))?;
            qc = MxArray::concatenate(&qc, &zq, 0)?;
        }
        let c = if pad { 2 } else { hi - lo };
        let gid = ic.maximum(&zero)?.reshape(&[c * kmax])?;
        let listed = ic.greater_equal(&zero)?;
        let valid = listed.reshape(&[c, 1, kmax])?;
        // unlisted ids read slot 0, which may hold another request's
        // non-finite row; zero it so `0 · NaN` never reaches the output
        let g = listed.reshape(&[c, kmax, 1])?.where_(
            &keys
                .gather(&gid)?
                .astype(DType::Float32)?
                .reshape(&[c, kmax, d])?,
            &zf,
        )?;
        let sc = qc.matmul(&g.transpose(Some(&[0, 2, 1]))?)?;
        let sc = valid.where_(&sc, &neg_inf)?.mul_scalar(f64::from(scale))?;
        let m = sc.max(Some(&[-1]), Some(true))?.maximum(&floor)?;
        let p = sc.sub(&m)?.exp()?;
        let den = p.sum(Some(&[-1]), Some(true))?.add(&sink.sub(&m)?.exp()?)?;
        let v = g.slice(&[0, 0, 0], &[c, kmax, v_width])?;
        let o = p.matmul(&v)?.div(&den)?;
        let o = if pad {
            o.slice(&[0, 0, 0], &[1, h, v_width])?
        } else {
            o
        };
        if rows < s {
            super::stats::eval(&[&o], "hy_v4_flash sparse attention chunk")?;
        }
        outs.push(o);
        lo = hi;
    }
    match outs.len() {
        1 => Ok(outs.swap_remove(0)),
        _ => MxArray::concatenate_many(outs.iter().collect(), Some(0)),
    }
}
