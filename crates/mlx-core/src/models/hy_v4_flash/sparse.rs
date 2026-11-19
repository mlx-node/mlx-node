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

/// `gather_sink_attention` with an explicit chunk budget. Every chunk is
/// evaluated before the next one is built.
pub(crate) fn gather_sink_attention_in(
    q: &MxArray,
    keys: &MxArray,
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
    let kd = dims(keys)?;
    if kd.len() != 2 || kd[1] != d {
        return Err(bad(format!("keys {kd:?} for queries of width {d}")));
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
    let sink = sinks.astype(DType::Float32)?.reshape(&[1, h, 1])?;
    let mut outs = Vec::new();
    let mut lo = 0;
    while lo < s {
        let hi = (lo + rows).min(s);
        let c = hi - lo;
        let ic = idx.slice(&[lo, 0], &[hi, kmax])?;
        let gid = ic.maximum(&zero)?.reshape(&[c * kmax])?;
        let valid = ic.greater_equal(&zero)?.reshape(&[c, 1, kmax])?;
        let g = keys
            .take(&gid, 0)?
            .astype(DType::Float32)?
            .reshape(&[c, kmax, d])?;
        let qc = q.slice(&[lo, 0, 0], &[hi, h, d])?.astype(DType::Float32)?;
        let sc = qc.matmul(&g.transpose(Some(&[0, 2, 1]))?)?;
        let sc = valid.where_(&sc, &neg_inf)?.mul_scalar(f64::from(scale))?;
        let m = sc.max(Some(&[-1]), Some(true))?.maximum(&floor)?;
        let p = sc.sub(&m)?.exp()?;
        let den = p.sum(Some(&[-1]), Some(true))?.add(&sink.sub(&m)?.exp()?)?;
        let v = g.slice(&[0, 0, 0], &[c, kmax, v_width])?;
        let o = p.matmul(&v)?.div(&den)?;
        MxArray::eval_arrays_with_context(&[&o], "hy_v4_flash sparse attention chunk")?;
        outs.push(o);
        lo = hi;
    }
    match outs.len() {
        1 => Ok(outs.swap_remove(0)),
        _ => MxArray::concatenate_many(outs.iter().collect(), Some(0)),
    }
}
