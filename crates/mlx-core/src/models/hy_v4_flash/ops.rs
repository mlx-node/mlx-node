//! Hy4 small ops, all math in f32 (llama.cpp PR #32517 graph):
//! hyper-connection mix / post / head with no comb term, the sigmoid router,
//! and the vocab head in row chunks.

use napi::{Error, Result};

use super::config::{HcConfig, MoeConfig};
use super::weights::{Dense, HcWeights};
use crate::array::{DType, MxArray};
use crate::models::deepseek_v41::attention::linear;
use crate::models::deepseek_v41::ops::hc_pre;
use crate::models::deepseek_v41::store::Weight;
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

fn as_f32(x: &MxArray) -> Result<MxArray> {
    if x.dtype()? == DType::Float32 {
        Ok(x.clone())
    } else {
        x.astype(DType::Float32)
    }
}

/// Fused RMSNorm with 1-D weight `w`, computed in f32 and cast back to `x`'s dtype.
pub fn rms_norm(x: &MxArray, w: &MxArray, eps: f32) -> Result<MxArray> {
    if w.ndim()? != 1 {
        return Err(bad(format!(
            "rms_norm weight must be 1-D, got {:?}",
            dims(w)?
        )));
    }
    let dt = x.dtype()?;
    let xf = if dt == DType::Float32 {
        x.clone()
    } else {
        x.astype(DType::Float32)?
    };
    let wf = if w.dtype()? == DType::Float32 {
        w.clone()
    } else {
        w.astype(DType::Float32)?
    };
    let y = crate::nn::rms_norm_scaled(&xf, &wf, eps)?;
    if dt == DType::Float32 {
        Ok(y)
    } else {
        y.astype(dt)
    }
}

/// Fused unweighted RMSNorm, computed in f32 and cast back to `x`'s dtype.
pub fn rms_norm_unweighted(x: &MxArray, eps: f32) -> Result<MxArray> {
    let dt = x.dtype()?;
    let xf = if dt == DType::Float32 {
        x.clone()
    } else {
        x.astype(DType::Float32)?
    };
    let y = crate::nn::rms_norm_unscaled(&xf, eps)?;
    if dt == DType::Float32 {
        Ok(y)
    } else {
        y.astype(dt)
    }
}

/// Precomputed inverse frequencies `[rd/2]` f32 for compiled precise RoPE.
pub fn rope_inv_freqs(inv_freqs: &[f32]) -> Result<MxArray> {
    if inv_freqs.is_empty() {
        return Err(bad("rope_inv_freqs: empty inv_freqs"));
    }
    MxArray::from_float32(inv_freqs, &[inv_freqs.len() as i64])
}

/// Converts `positions` to a cached 1-D f32 `MxArray` `[n]` for RoPE calls,
/// reusing the allocation across layers within the same wave.
pub fn rope_positions_f32(positions: &[u64]) -> Result<MxArray> {
    let n = positions.len();
    if n == 0 {
        return MxArray::from_float32(&[], &[0]);
    }
    let is_contig = n > 32
        && positions.last().copied() == Some(positions[0].saturating_add((n - 1) as u64))
        && positions
            .iter()
            .enumerate()
            .all(|(i, &p)| p == positions[0] + i as u64);

    thread_local! {
        static SMALL_CACHE: std::cell::RefCell<Vec<(Vec<u64>, MxArray)>> =
            const { std::cell::RefCell::new(Vec::new()) };
        static CONTIG_CACHE: std::cell::RefCell<Option<(u64, usize, MxArray)>> =
            const { std::cell::RefCell::new(None) };
    }

    if n <= 32 {
        if let Some(hit) = SMALL_CACHE.with(|c| {
            let c = c.borrow();
            c.iter()
                .find(|(k, _)| k.as_slice() == positions)
                .map(|(_, a)| a.clone())
        }) {
            return Ok(hit);
        }
    } else if is_contig
        && let Some(hit) = CONTIG_CACHE.with(|c| {
            c.borrow()
                .as_ref()
                .filter(|(s, len, _)| *s == positions[0] && *len == n)
                .map(|(_, _, a)| a.clone())
        })
    {
        return Ok(hit);
    }

    let pos_f32: Vec<f32> = positions
        .iter()
        .map(|&p| {
            i32::try_from(p)
                .map(|v| v as f32)
                .map_err(|_| bad(format!("rope position {p} exceeds i32")))
        })
        .collect::<Result<_>>()?;
    let arr = MxArray::from_float32(&pos_f32, &[n as i64])?;

    if n <= 32 {
        SMALL_CACHE.with(|c| {
            let mut c = c.borrow_mut();
            if c.len() >= 8 {
                c.remove(0);
            }
            c.push((positions.to_vec(), arr.clone()));
        });
    } else if is_contig {
        CONTIG_CACHE.with(|c| {
            *c.borrow_mut() = Some((positions[0], n, arr.clone()));
        });
    }
    Ok(arr)
}

/// Fused precise-f32 RoPE using a pre-built 1-D f32 `pos_arr [n]`.
pub fn fast_rope_with_pos(x: &MxArray, inv_freqs: &MxArray, pos_arr: &MxArray) -> Result<MxArray> {
    if inv_freqs.ndim()? != 1 {
        return Err(bad(format!(
            "rope inv_freqs must be 1-D, got {:?}",
            dims(inv_freqs)?
        )));
    }
    let d = dims(x)?;
    let rd = 2 * inv_freqs.shape_at(0)?;
    let n = match d.as_slice() {
        &[n, w] | &[n, _, w] if w == rd => n,
        _ => {
            return Err(bad(format!("fast_rope got shape {d:?} for rope_dim {rd}")));
        }
    };
    if pos_arr.ndim()? != 1 || pos_arr.shape_at(0)? != n {
        return Err(bad(format!(
            "fast_rope_with_pos got pos shape {:?} for {n} rows",
            dims(pos_arr)?
        )));
    }
    if n == 0 {
        return Ok(x.clone());
    }
    let dt = x.dtype()?;
    let xf = if dt == DType::Float32 {
        x.clone()
    } else {
        x.astype(DType::Float32)?
    };
    let handle = unsafe {
        mlx_sys::mlx_rope_precise_compiled(
            xf.as_raw_ptr(),
            pos_arr.as_raw_ptr(),
            inv_freqs.as_raw_ptr(),
        )
    };
    let rot = MxArray::from_handle(handle, "hy_v4_flash fast_rope")?;
    if dt == DType::Float32 {
        Ok(rot)
    } else {
        rot.astype(dt)
    }
}

/// Fused precise-f32 RoPE on `x` (`[n, rd]` or `[n, H, rd]`) at `positions`.
pub fn fast_rope(x: &MxArray, inv_freqs: &MxArray, positions: &[u64]) -> Result<MxArray> {
    let d = dims(x)?;
    let n = *d.first().ok_or_else(|| bad("fast_rope: scalar input"))?;
    if positions.len() as i64 != n {
        return Err(bad(format!(
            "fast_rope got {} positions for {n} rows",
            positions.len()
        )));
    }
    if n == 0 {
        return Ok(x.clone());
    }
    let pos_arr = rope_positions_f32(positions)?;
    fast_rope_with_pos(x, inv_freqs, &pos_arr)
}

/// Fused RoPE on the trailing `rd` dims of `x` (`[n, d]` or `[n, H, d]`) using
/// a pre-built 1-D f32 `pos_arr [n]`.
pub fn fast_partial_rope_tail_with_pos(
    x: &MxArray,
    inv_freqs: &MxArray,
    pos_arr: &MxArray,
) -> Result<MxArray> {
    if inv_freqs.ndim()? != 1 {
        return Err(bad(format!(
            "rope inv_freqs must be 1-D, got {:?}",
            dims(inv_freqs)?
        )));
    }
    let rd = 2 * inv_freqs.shape_at(0)?;
    let d = dims(x)?;
    let width = *d
        .last()
        .ok_or_else(|| bad("fast_partial_rope_tail: scalar input"))?;
    if rd <= 0 || rd > width {
        return Err(bad(format!(
            "fast_partial_rope_tail: rope dim {rd} outside 1..={width}"
        )));
    }
    if rd == width {
        return fast_rope_with_pos(x, inv_freqs, pos_arr);
    }
    let dt = x.dtype()?;
    if let Some(rot) = try_partial_rope_tail(x, pos_arr, inv_freqs)? {
        return if dt == DType::Float32 {
            Ok(rot)
        } else {
            rot.astype(dt)
        };
    }
    let head = last(x, 0, width - rd)?;
    let tail = last(x, width - rd, width)?;
    let rot = fast_rope_with_pos(&tail, inv_freqs, pos_arr)?;
    MxArray::concatenate(&head, &rot, -1)
}

/// Fused RoPE on the trailing `rd` dims of `x` (`[n, d]` or `[n, H, d]`).
pub fn fast_partial_rope_tail(
    x: &MxArray,
    inv_freqs: &MxArray,
    positions: &[u64],
) -> Result<MxArray> {
    let pos_arr = rope_positions_f32(positions)?;
    fast_partial_rope_tail_with_pos(x, inv_freqs, &pos_arr)
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
    let wt = weight.astype(DType::Float32)?.transpose(Some(&[1, 0]))?;
    // Above 256 rows, scaling after the matmul avoids materializing a second [n, hc*d] tensor.
    if n <= 256 {
        return rms_norm_unweighted(&xf, norm_eps)?.matmul(&wt);
    }
    let rs = xf
        .square()?
        .mean(Some(&[-1]), Some(true))?
        .add_scalar(f64::from(norm_eps))?
        .sqrt()?
        .reciprocal()?;
    xf.matmul(&wt)?.mul(&rs)
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

/// Fused `hc_mix` + `hc_reduce`: returns `(h [n, d], post [n, hc])` with `h`
/// in `x`'s dtype and `post` in f32.
pub fn hc_mix_reduce(
    x: &MxArray,
    w: &HcWeights,
    norm_eps: f32,
    cfg: &HcConfig,
) -> Result<(MxArray, MxArray)> {
    let [n, hc, d] = streams(x)?;
    if hc != cfg.count as i64
        || w.weight.ndim()? != 2
        || w.weight.shape_at(0)? != 2 * hc
        || w.weight.shape_at(1)? != hc * d
    {
        return Err(bad(format!(
            "hc mix: {hc} streams of {d}, config {}, fn {:?}",
            cfg.count,
            dims(&w.weight)?
        )));
    }
    check_len(&w.scale, 2, "hc scale")?;
    check_len(&w.base, 2 * hc, "hc base")?;
    if n > 0 {
        let dt = x.dtype()?;
        let xf = as_f32(x)?;
        let wf = as_f32(&w.weight)?;
        let sf = as_f32(&w.scale)?;
        let bf = as_f32(&w.base)?;
        let mut out_h = std::ptr::null_mut();
        let mut out_post = std::ptr::null_mut();
        // SAFETY: all input handles are live for the call; out pointers receive new arrays on true.
        let ok = unsafe {
            mlx_sys::mlx_hy4_hc_mix_reduce(
                xf.as_raw_ptr(),
                wf.as_raw_ptr(),
                sf.as_raw_ptr(),
                bf.as_raw_ptr(),
                norm_eps,
                cfg.eps,
                cfg.magnitude,
                true,
                &mut out_h,
                &mut out_post,
            )
        };
        if ok {
            let h = MxArray::from_handle(out_h, "hy_v4_flash hc_mix_reduce h")?;
            let post = MxArray::from_handle(out_post, "hy_v4_flash hc_mix_reduce post")?;
            let h = if dt == DType::Float32 {
                h
            } else {
                h.astype(dt)?
            };
            return Ok((h, post));
        }
    }
    let g = hc_mix(x, w, norm_eps, cfg)?;
    let h = hc_reduce(x, &g.pre)?;
    Ok((h, g.post))
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
    let dt = res.dtype()?;
    if n > 0 {
        let yf = as_f32(y)?;
        let rf = as_f32(res)?;
        let pf = as_f32(post)?;
        // SAFETY: all input handles are live for the call; returns a new array or null.
        let h = unsafe {
            mlx_sys::mlx_hy4_hc_post_identity(yf.as_raw_ptr(), rf.as_raw_ptr(), pf.as_raw_ptr())
        };
        if !h.is_null() {
            let out = MxArray::from_handle(h, "hy_v4_flash hc_post_identity")?;
            return if dt == DType::Float32 {
                Ok(out)
            } else {
                out.astype(dt)
            };
        }
    }
    let p = post.astype(DType::Float32)?.reshape(&[n, hc, 1])?;
    let yf = y.astype(DType::Float32)?.reshape(&[n, 1, d])?;
    res.astype(DType::Float32)?.add(&p.mul(&yf)?)?.astype(dt)
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
    let [n, hc, d] = streams(x)?;
    if w.weight.ndim()? != 2 || w.weight.shape_at(0)? != hc || w.weight.shape_at(1)? != hc * d {
        return Err(bad(format!(
            "hc head fn {:?} for {hc} streams of {d}",
            dims(&w.weight)?
        )));
    }
    check_len(&w.scale, 1, "hc head scale")?;
    check_len(&w.base, hc, "hc head base")?;
    if n > 0 {
        let dt = x.dtype()?;
        let xf = as_f32(x)?;
        let wf = as_f32(&w.weight)?;
        let sf = as_f32(&w.scale)?;
        let bf = as_f32(&w.base)?;
        let mut out_h = std::ptr::null_mut();
        // SAFETY: all input handles are live for the call; out_h receives a new array on true.
        let ok = unsafe {
            mlx_sys::mlx_hy4_hc_mix_reduce(
                xf.as_raw_ptr(),
                wf.as_raw_ptr(),
                sf.as_raw_ptr(),
                bf.as_raw_ptr(),
                norm_eps,
                hc_eps,
                0.0,
                false,
                &mut out_h,
                std::ptr::null_mut(),
            )
        };
        if ok {
            let h = MxArray::from_handle(out_h, "hy_v4_flash hc_head h")?;
            return if dt == DType::Float32 {
                Ok(h)
            } else {
                h.astype(dt)
            };
        }
    }
    hc_reduce(x, &hc_head_pre(x, w, norm_eps, hc_eps)?)
}

/// Fused `hc_mix_reduce` + `rms_norm(norm_w)`: returns `(h_norm [n, d], post [n, hc])`
/// with `h_norm` in `x`'s dtype and `post` in f32.
pub fn hc_mix_reduce_norm(
    x: &MxArray,
    w: &HcWeights,
    norm_w: &MxArray,
    norm_eps: f32,
    cfg: &HcConfig,
) -> Result<(MxArray, MxArray)> {
    let [n, hc, d] = streams(x)?;
    if hc != cfg.count as i64
        || w.weight.ndim()? != 2
        || w.weight.shape_at(0)? != 2 * hc
        || w.weight.shape_at(1)? != hc * d
    {
        return Err(bad(format!(
            "hc mix norm: {hc} streams of {d}, config {}, fn {:?}",
            cfg.count,
            dims(&w.weight)?
        )));
    }
    check_len(&w.scale, 2, "hc scale")?;
    check_len(&w.base, 2 * hc, "hc base")?;
    check_len(norm_w, d, "hc norm_w")?;
    if n > 0 {
        let dt = x.dtype()?;
        let xf = as_f32(x)?;
        let wf = as_f32(&w.weight)?;
        let sf = as_f32(&w.scale)?;
        let bf = as_f32(&w.base)?;
        let nwf = as_f32(norm_w)?;
        let mut out_h = std::ptr::null_mut();
        let mut out_post = std::ptr::null_mut();
        // SAFETY: all input handles are live for the call; out pointers receive new arrays on true.
        let ok = unsafe {
            mlx_sys::mlx_hy4_hc_mix_reduce_norm(
                xf.as_raw_ptr(),
                wf.as_raw_ptr(),
                sf.as_raw_ptr(),
                bf.as_raw_ptr(),
                nwf.as_raw_ptr(),
                norm_eps,
                cfg.eps,
                cfg.magnitude,
                norm_eps,
                false,
                &mut out_h,
                &mut out_post,
            )
        };
        if ok {
            let h = MxArray::from_handle(out_h, "hy_v4_flash hc_mix_reduce_norm h")?;
            let post = MxArray::from_handle(out_post, "hy_v4_flash hc_mix_reduce_norm post")?;
            let h = if dt == DType::Float32 {
                h
            } else {
                h.astype(dt)?
            };
            return Ok((h, post));
        }
    }
    let (h, post) = hc_mix_reduce(x, w, norm_eps, cfg)?;
    let hn = rms_norm(&h, norm_w, norm_eps)?;
    Ok((hn, post))
}

/// Fused `hc_head` + `rms_norm(norm_w)`: returns `h_norm [n, d]` in `x`'s dtype.
pub fn hc_head_norm(
    x: &MxArray,
    w: &HcWeights,
    norm_w: &MxArray,
    norm_eps: f32,
    hc_eps: f32,
) -> Result<MxArray> {
    let [n, hc, d] = streams(x)?;
    if w.weight.ndim()? != 2 || w.weight.shape_at(0)? != hc || w.weight.shape_at(1)? != hc * d {
        return Err(bad(format!(
            "hc head norm fn {:?} for {hc} streams of {d}",
            dims(&w.weight)?
        )));
    }
    check_len(&w.scale, 1, "hc head scale")?;
    check_len(&w.base, hc, "hc head base")?;
    check_len(norm_w, d, "hc head norm_w")?;
    if n > 0 {
        let dt = x.dtype()?;
        let xf = as_f32(x)?;
        let wf = as_f32(&w.weight)?;
        let sf = as_f32(&w.scale)?;
        let bf = as_f32(&w.base)?;
        let nwf = as_f32(norm_w)?;
        let mut out_h = std::ptr::null_mut();
        // SAFETY: all input handles are live for the call; out_h receives a new array on true.
        let ok = unsafe {
            mlx_sys::mlx_hy4_hc_mix_reduce_norm(
                xf.as_raw_ptr(),
                wf.as_raw_ptr(),
                sf.as_raw_ptr(),
                bf.as_raw_ptr(),
                nwf.as_raw_ptr(),
                norm_eps,
                hc_eps,
                0.0,
                norm_eps,
                true,
                &mut out_h,
                std::ptr::null_mut(),
            )
        };
        if ok {
            let h = MxArray::from_handle(out_h, "hy_v4_flash hc_head_norm h")?;
            return if dt == DType::Float32 {
                Ok(h)
            } else {
                h.astype(dt)
            };
        }
    }
    rms_norm(&hc_head(x, w, norm_eps, hc_eps)?, norm_w, norm_eps)
}

/// Fused `rms_norm(y, post_norm_w)` + `hc_post_identity`: returns `[n, hc, d]` in `res`'s dtype.
pub fn post_norm_hc_post(
    y: &MxArray,
    res: &MxArray,
    post: &MxArray,
    post_norm_w: &MxArray,
    norm_eps: f32,
) -> Result<MxArray> {
    let [n, hc, d] = streams(res)?;
    if dims(y)?.as_slice() != [n, d] || dims(post)?.as_slice() != [n, hc] {
        return Err(bad(format!(
            "post_norm_hc_post: y {:?}, post {:?} for streams [{n}, {hc}, {d}]",
            dims(y)?,
            dims(post)?
        )));
    }
    check_len(post_norm_w, d, "post_norm_w")?;
    let dt = res.dtype()?;
    if n > 0 {
        let yf = as_f32(y)?;
        let rf = as_f32(res)?;
        let pf = as_f32(post)?;
        let pnwf = as_f32(post_norm_w)?;
        // SAFETY: all input handles are live for the call; returns a new array or null.
        let h = unsafe {
            mlx_sys::mlx_hy4_post_norm_hc_post(
                yf.as_raw_ptr(),
                rf.as_raw_ptr(),
                pf.as_raw_ptr(),
                pnwf.as_raw_ptr(),
                norm_eps,
            )
        };
        if !h.is_null() {
            let out = MxArray::from_handle(h, "hy_v4_flash post_norm_hc_post")?;
            return if dt == DType::Float32 {
                Ok(out)
            } else {
                out.astype(dt)
            };
        }
    }
    let yn = rms_norm(&as_f32(y)?, post_norm_w, norm_eps)?;
    hc_post_identity(&yn, res, post)
}

/// Megafused `post_norm(y) + hc_post_identity(res, post_prev) + hc_mix_reduce(next_hc) [+ rms_norm(next_norm_w)]`.
/// Returns `(x_out [n, hc, d], h_next [n, d], post_next [n, hc])`.
#[allow(clippy::too_many_arguments)]
pub fn hc_bridge(
    y: &MxArray,
    res: &MxArray,
    post_prev: &MxArray,
    post_norm_w: &MxArray,
    next_hc: &HcWeights,
    next_norm_w: Option<&MxArray>,
    norm_eps: f32,
    cfg: &HcConfig,
) -> Result<(MxArray, MxArray, MxArray)> {
    let [n, hc, d] = streams(res)?;
    if dims(y)?.as_slice() != [n, d] || dims(post_prev)?.as_slice() != [n, hc] {
        return Err(bad(format!(
            "hc_bridge: y {:?}, post_prev {:?} for streams [{n}, {hc}, {d}]",
            dims(y)?,
            dims(post_prev)?
        )));
    }
    if hc != cfg.count as i64
        || next_hc.weight.ndim()? != 2
        || next_hc.weight.shape_at(0)? != 2 * hc
        || next_hc.weight.shape_at(1)? != hc * d
    {
        return Err(bad(format!(
            "hc_bridge: {hc} streams of {d}, config {}, fn {:?}",
            cfg.count,
            dims(&next_hc.weight)?
        )));
    }
    check_len(post_norm_w, d, "post_norm_w")?;
    check_len(&next_hc.scale, 2, "next hc scale")?;
    check_len(&next_hc.base, 2 * hc, "next hc base")?;
    if let Some(nnw) = next_norm_w {
        check_len(nnw, d, "next_norm_w")?;
    }
    let dt = res.dtype()?;
    if n > 0 {
        let yf = as_f32(y)?;
        let rf = as_f32(res)?;
        let ppf = as_f32(post_prev)?;
        let pnwf = as_f32(post_norm_w)?;
        let nwf = as_f32(&next_hc.weight)?;
        let nsf = as_f32(&next_hc.scale)?;
        let nbf = as_f32(&next_hc.base)?;
        let nnwf = next_norm_w.map(as_f32).transpose()?;
        let nnw_ptr = nnwf
            .as_ref()
            .map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
        let mut out_x = std::ptr::null_mut();
        let mut out_h = std::ptr::null_mut();
        let mut out_post = std::ptr::null_mut();
        // SAFETY: all input handles are live for the call; out pointers receive new arrays on true.
        let ok = unsafe {
            mlx_sys::mlx_hy4_hc_bridge(
                yf.as_raw_ptr(),
                rf.as_raw_ptr(),
                ppf.as_raw_ptr(),
                pnwf.as_raw_ptr(),
                nwf.as_raw_ptr(),
                nsf.as_raw_ptr(),
                nbf.as_raw_ptr(),
                nnw_ptr,
                norm_eps,
                norm_eps,
                cfg.eps,
                cfg.magnitude,
                norm_eps,
                next_norm_w.is_none(),
                false,
                &mut out_x,
                &mut out_h,
                &mut out_post,
            )
        };
        if ok {
            let x_out = MxArray::from_handle(out_x, "hy_v4_flash hc_bridge x_out")?;
            let h_out = MxArray::from_handle(out_h, "hy_v4_flash hc_bridge h_out")?;
            let p_out = MxArray::from_handle(out_post, "hy_v4_flash hc_bridge post_next")?;
            let x_out = if dt == DType::Float32 {
                x_out
            } else {
                x_out.astype(dt)?
            };
            let h_out = if dt == DType::Float32 {
                h_out
            } else {
                h_out.astype(dt)?
            };
            return Ok((x_out, h_out, p_out));
        }
    }
    let x_out = post_norm_hc_post(y, res, post_prev, post_norm_w, norm_eps)?;
    let (h_out, p_out) = match next_norm_w {
        Some(nnw) => hc_mix_reduce_norm(&x_out, next_hc, nnw, norm_eps, cfg)?,
        None => hc_mix_reduce(&x_out, next_hc, norm_eps, cfg)?,
    };
    Ok((x_out, h_out, p_out))
}

/// Megafused `post_norm(y) + hc_post_identity(res, post_prev) + hc_head(output_hc) [+ rms_norm(output_norm)]`.
/// Avoids materializing the final `[n, hc, d]` stream tensor.
#[allow(clippy::too_many_arguments)]
pub fn hc_bridge_head(
    y: &MxArray,
    res: &MxArray,
    post_prev: &MxArray,
    post_norm_w: &MxArray,
    output_hc: &HcWeights,
    output_norm_w: Option<&MxArray>,
    norm_eps: f32,
    hc_eps: f32,
) -> Result<MxArray> {
    let [n, hc, d] = streams(res)?;
    if dims(y)?.as_slice() != [n, d] || dims(post_prev)?.as_slice() != [n, hc] {
        return Err(bad(format!(
            "hc_bridge_head: y {:?}, post_prev {:?} for streams [{n}, {hc}, {d}]",
            dims(y)?,
            dims(post_prev)?
        )));
    }
    if output_hc.weight.ndim()? != 2
        || output_hc.weight.shape_at(0)? != hc
        || output_hc.weight.shape_at(1)? != hc * d
    {
        return Err(bad(format!(
            "hc_bridge_head: fn {:?} for {hc} streams of {d}",
            dims(&output_hc.weight)?
        )));
    }
    check_len(post_norm_w, d, "post_norm_w")?;
    check_len(&output_hc.scale, 1, "hc head scale")?;
    check_len(&output_hc.base, hc, "hc head base")?;
    if let Some(nnw) = output_norm_w {
        check_len(nnw, d, "output_norm_w")?;
    }
    let dt = res.dtype()?;
    if n > 0 {
        let yf = as_f32(y)?;
        let rf = as_f32(res)?;
        let ppf = as_f32(post_prev)?;
        let pnwf = as_f32(post_norm_w)?;
        let nwf = as_f32(&output_hc.weight)?;
        let nsf = as_f32(&output_hc.scale)?;
        let nbf = as_f32(&output_hc.base)?;
        let nnwf = output_norm_w.map(as_f32).transpose()?;
        let nnw_ptr = nnwf
            .as_ref()
            .map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
        let mut out_h = std::ptr::null_mut();
        // SAFETY: all input handles are live for the call; out_h receives a new array on true.
        let ok = unsafe {
            mlx_sys::mlx_hy4_hc_bridge(
                yf.as_raw_ptr(),
                rf.as_raw_ptr(),
                ppf.as_raw_ptr(),
                pnwf.as_raw_ptr(),
                nwf.as_raw_ptr(),
                nsf.as_raw_ptr(),
                nbf.as_raw_ptr(),
                nnw_ptr,
                norm_eps,
                norm_eps,
                hc_eps,
                0.0,
                norm_eps,
                output_norm_w.is_none(),
                true,
                std::ptr::null_mut(),
                &mut out_h,
                std::ptr::null_mut(),
            )
        };
        if ok {
            let h_out = MxArray::from_handle(out_h, "hy_v4_flash hc_bridge_head h_out")?;
            return if dt == DType::Float32 {
                Ok(h_out)
            } else {
                h_out.astype(dt)
            };
        }
    }
    let x_out = post_norm_hc_post(y, res, post_prev, post_norm_w, norm_eps)?;
    match output_norm_w {
        Some(nnw) => hc_head_norm(&x_out, output_hc, nnw, norm_eps, hc_eps),
        None => hc_head(&x_out, output_hc, norm_eps, hc_eps),
    }
}

/// Row-invariant f32/f16 router GEMV for `1..=16` rows when `router` is a 2-D
/// f32 or f16 weight with `K % 32 == 0`. Returns `Ok(None)` when Metal or shape
/// is unsupported.
pub(crate) fn try_router_mv(xf: &MxArray, router: &Dense) -> Result<Option<MxArray>> {
    let Dense::Array(w) = router else {
        return Ok(None);
    };
    if xf.ndim()? != 2 || w.ndim()? != 2 || !matches!(w.dtype()?, DType::Float32 | DType::Float16) {
        return Ok(None);
    }
    let m = xf.shape_at(0)?;
    let k = xf.shape_at(1)?;
    if !(1..=16).contains(&m) || k < 32 || k % 32 != 0 || w.shape_at(1)? != k {
        return Ok(None);
    }
    // SAFETY: live array handles; returns a new array or null on unsupported/non-Metal.
    let h = unsafe { mlx_sys::mlx_hy4_router_mv(xf.as_raw_ptr(), w.as_raw_ptr()) };
    if h.is_null() {
        return Ok(None);
    }
    Ok(Some(MxArray::from_handle(h, "hy_v4_flash router_mv")?))
}

/// Raw router matmul logits `x · gate_inpᵀ` in f32, `[rows, n_experts]`.
pub fn router_logits(x: &MxArray, router: &Dense) -> Result<MxArray> {
    let xf = as_f32(x)?;
    if xf.ndim()? == 2
        && (1..=16).contains(&xf.shape_at(0)?)
        && let Some(logits) = try_router_mv(&xf, router)?
    {
        return Ok(logits);
    }
    linear(&xf, router)
}

/// Fused `routed_up(routed_latent) + shared_down(swiglu(shared_gu, limit))` for
/// SymQ8 decode (`1..=8` rows). Returns `Ok(None)` when unsupported.
pub(crate) fn try_q8_dual_down_add(
    routed_latent: &MxArray,
    shared_gu: &MxArray,
    routed_up: &Dense,
    shared_down: &Dense,
    limit: f32,
) -> Result<Option<MxArray>> {
    let (
        Dense::SymQ8 {
            w: w_up,
            scales: s_up,
        },
        Dense::SymQ8 {
            w: w_sd,
            scales: s_sd,
        },
    ) = (routed_up, shared_down)
    else {
        return Ok(None);
    };
    if routed_latent.ndim()? != 2 || shared_gu.ndim()? != 2 {
        return Ok(None);
    }
    let m = routed_latent.shape_at(0)?;
    let k = routed_latent.shape_at(1)?;
    if !(1..=8).contains(&m)
        || k < 32
        || k % 32 != 0
        || shared_gu.shape_at(0)? != m
        || shared_gu.shape_at(1)? != 2 * k
    {
        return Ok(None);
    }
    let rlf = as_f32(routed_latent)?;
    let sguf = as_f32(shared_gu)?;
    // SAFETY: live array handles; returns a new array or null on unsupported/non-Metal.
    let h = unsafe {
        mlx_sys::mlx_hy4_q8_dual_down_add(
            rlf.as_raw_ptr(),
            sguf.as_raw_ptr(),
            w_up.as_raw_ptr(),
            s_up.as_raw_ptr(),
            w_sd.as_raw_ptr(),
            s_sd.as_raw_ptr(),
            limit,
        )
    };
    if h.is_null() {
        return Ok(None);
    }
    Ok(Some(MxArray::from_handle(
        h,
        "hy_v4_flash q8_dual_down_add",
    )?))
}

/// Router scores `σ(x · gate_inpᵀ)` in f32, `[rows, n_experts]`.
pub fn router_scores(x: &MxArray, router: &Dense) -> Result<MxArray> {
    Activations::sigmoid(&router_logits(x, router)?)
}

/// Fused GPU sigmoid + bias + top-k + ascending-id sort + renormalization.
/// Returns `(ids [rows, k] u32, weights [rows, k] f32)` in ascending expert ID
/// order per row, or `Ok(None)` when Metal or the shape is unsupported.
pub fn router_topk_gpu(
    logits: &MxArray,
    bias: Option<&MxArray>,
    moe: &MoeConfig,
) -> Result<Option<(MxArray, MxArray)>> {
    if logits.ndim()? != 2 || logits.shape_at(1)? != moe.n_experts as i64 {
        return Err(bad(format!(
            "router logits {:?} for {} experts",
            dims(logits)?,
            moe.n_experts
        )));
    }
    if moe.n_used == 0 || moe.n_used > moe.n_experts {
        return Err(bad(format!(
            "router: {} experts, {} used",
            moe.n_experts, moe.n_used
        )));
    }
    let lf = as_f32(logits)?;
    let bf = match bias {
        Some(b) => {
            check_len(b, moe.n_experts as i64, "router bias")?;
            Some(as_f32(b)?)
        }
        None => None,
    };
    let bias_ptr = bf
        .as_ref()
        .map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    let mut out_ids = std::ptr::null_mut();
    let mut out_weights = std::ptr::null_mut();
    // SAFETY: input handles are live for the call; out pointers receive new arrays on true.
    let ok = unsafe {
        mlx_sys::mlx_hy4_router_topk(
            lf.as_raw_ptr(),
            bias_ptr,
            moe.route_scale,
            1e-20,
            moe.n_used as i32,
            moe.norm_topk,
            &mut out_ids,
            &mut out_weights,
        )
    };
    if !ok {
        return Ok(None);
    }
    let ids = MxArray::from_handle(out_ids, "hy_v4_flash router_topk ids")?;
    let weights = MxArray::from_handle(out_weights, "hy_v4_flash router_topk weights")?;
    Ok(Some((ids, weights)))
}

fn route_rows(
    scores: &[f32],
    bias: Option<&[f32]>,
    n: usize,
    k: usize,
    route_scale: f32,
    norm: bool,
    out_ids: &mut [u32],
    out_weights: &mut [f32],
) {
    let cmp = |a: &(f32, u32), b: &(f32, u32)| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1));
    let mut pairs: Vec<(f32, u32)> = Vec::with_capacity(n);
    for ((row_scores, row_ids), row_weights) in scores
        .chunks_exact(n)
        .zip(out_ids.chunks_exact_mut(k))
        .zip(out_weights.chunks_exact_mut(k))
    {
        pairs.clear();
        match bias {
            Some(b) => pairs.extend(
                row_scores
                    .iter()
                    .zip(b)
                    .enumerate()
                    .map(|(e, (&s, &bi))| (s + bi, e as u32)),
            ),
            None => pairs.extend(row_scores.iter().enumerate().map(|(e, &s)| (s, e as u32))),
        }
        if k < n {
            pairs.select_nth_unstable_by(k - 1, cmp);
        }
        let top = &mut pairs[..k];
        top.sort_unstable_by(cmp);
        let sum = top
            .iter()
            .fold(0.0f32, |a, &(_, e)| a + row_scores[e as usize])
            + 1e-20;
        for (j, &(_, e)) in top.iter().enumerate() {
            let x = row_scores[e as usize];
            row_ids[j] = e;
            row_weights[j] = (if norm && k > 1 { x / sum } else { x }) * route_scale;
        }
    }
}

/// O(N) per-row top-k selection matching `deepseek_v41::ops::route_select`
/// bit-for-bit, parallelized across rows when `rows >= 256`.
pub fn route_select_fast(
    scores: &[f32],
    bias: Option<&[f32]>,
    n: usize,
    k: usize,
    route_scale: f32,
    norm: bool,
) -> (Vec<u32>, Vec<f32>) {
    let rows = scores.len() / n;
    if rows == 0 {
        return (Vec::new(), Vec::new());
    }
    let mut ids = vec![0u32; rows * k];
    let mut weights = vec![0.0f32; rows * k];
    if rows < 256 {
        route_rows(
            scores,
            bias,
            n,
            k,
            route_scale,
            norm,
            &mut ids,
            &mut weights,
        );
        return (ids, weights);
    }
    let threads = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(8)
        .min(8)
        .min(rows / 32)
        .max(1);
    let chunk_rows = rows.div_ceil(threads);
    std::thread::scope(|scope| {
        for ((sc, id), wt) in scores
            .chunks(chunk_rows * n)
            .zip(ids.chunks_mut(chunk_rows * k))
            .zip(weights.chunks_mut(chunk_rows * k))
        {
            scope.spawn(move || {
                route_rows(sc, bias, n, k, route_scale, norm, id, wt);
            });
        }
    });
    (ids, weights)
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
    if let Some(b) = bias
        && b.len() != n
    {
        return Err(bad(format!("router bias has {} entries for {n}", b.len())));
    }
    Ok(route_select_fast(
        scores,
        bias,
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

/// Like `route`, evaluating `scores` and `next_scores` in one host sync.
pub fn route_pair(
    scores: &MxArray,
    bias: Option<&[f32]>,
    moe: &MoeConfig,
    next_scores: &MxArray,
    next_bias: Option<&[f32]>,
    next_moe: &MoeConfig,
) -> Result<((Vec<u32>, Vec<f32>), (Vec<u32>, Vec<f32>))> {
    if scores.ndim()? != 2 || scores.shape_at(1)? != moe.n_experts as i64 {
        return Err(bad(format!(
            "router scores {:?} for {} experts",
            dims(scores)?,
            moe.n_experts
        )));
    }
    if next_scores.ndim()? != 2 || next_scores.shape_at(1)? != next_moe.n_experts as i64 {
        return Err(bad(format!(
            "router scores {:?} for {} experts",
            dims(next_scores)?,
            next_moe.n_experts
        )));
    }
    let s = as_f32(scores)?;
    let ns = as_f32(next_scores)?;
    super::stats::eval(&[&s, &ns], "hy_v4_flash router")?;
    let s = s.to_float32()?;
    let ns = ns.to_float32()?;
    Ok((
        route_host(s.as_ref(), bias, moe)?,
        route_host(ns.as_ref(), next_bias, next_moe)?,
    ))
}

/// Single-dispatch BF16-weight / F32-accumulation LM head for 1..=16 rows.
pub(crate) fn try_bf16_lm_head(xf: &MxArray, w: &MxArray) -> Result<Option<MxArray>> {
    if xf.ndim()? != 2 || w.ndim()? != 2 || w.dtype()? != DType::BFloat16 {
        return Ok(None);
    }
    let rows = xf.shape_at(0)?;
    let k = xf.shape_at(1)?;
    if !(1..=16).contains(&rows) || k <= 0 || k % 4 != 0 || w.shape_at(1)? != k {
        return Ok(None);
    }
    // SAFETY: live array handles; returns a new array or null on unsupported/non-Metal.
    let h = unsafe { mlx_sys::mlx_hy4_bf16_lm_head(xf.as_raw_ptr(), w.as_raw_ptr()) };
    if h.is_null() {
        return Ok(None);
    }
    Ok(Some(MxArray::from_handle(h, "hy_v4_flash bf16_lm_head")?))
}

/// f32 logits `[s, vocab]` of `x [s, d]`; see `head_logits_chunked`.
pub fn head_logits(x: &MxArray, head: &Dense) -> Result<MxArray> {
    head_logits_chunked(x, head, HEAD_CHUNK)
}

/// f32 logits of `x [s, d]`. A `Qmm` head is one `quantized_matmul`; a BF16
/// `Dense::Array` with 1..=16 rows runs through the native BF16 LM head kernel;
/// any other head is cast to f32 `chunk` vocab rows at a time (`head_chunks`).
pub fn head_logits_chunked(x: &MxArray, head: &Dense, chunk: i64) -> Result<MxArray> {
    if chunk <= 0 {
        return Err(bad(format!("head chunk {chunk} must be > 0")));
    }
    let xf = x.astype(DType::Float32)?;
    if head.is_qmm() {
        return head.mm_t(&xf);
    }
    if let Dense::Array(w) = head
        && let Some(logits) = try_bf16_lm_head(&xf, w)?
    {
        return Ok(logits);
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
mod gguf_tests;
#[cfg(test)]
mod tests;

/// Routed expert decode reading raw GGUF `block_q4_K` (gate/up) and
/// `block_q4_K` or `block_q6_K` (down) bytes directly from `uint8` buffers.
/// Returns `Ok(None)` when Metal is unavailable or the arguments are unsupported.
#[allow(clippy::too_many_arguments)]
pub fn gguf_routed_decode(
    x: &MxArray,
    slots: &MxArray,
    weights: &MxArray,
    gate_buf: &MxArray,
    gate_off: u64,
    up_buf: &MxArray,
    up_off: u64,
    down_buf: &MxArray,
    down_off: u64,
    n_experts: usize,
    m: usize,
    h: usize,
    down_bits: i32,
    limit: f32,
) -> Result<Option<MxArray>> {
    let (Ok(n_experts_i32), Ok(m_i32), Ok(h_i32)) =
        (i32::try_from(n_experts), i32::try_from(m), i32::try_from(h))
    else {
        return Ok(None);
    };
    let xf = as_f32(x)?;
    let wf = as_f32(weights)?;
    // SAFETY: all handles are live arrays owned by this call.
    let handle = unsafe {
        mlx_sys::mlx_hy4_gguf_routed_decode(
            xf.as_raw_ptr(),
            slots.as_raw_ptr(),
            wf.as_raw_ptr(),
            gate_buf.as_raw_ptr(),
            gate_off,
            up_buf.as_raw_ptr(),
            up_off,
            down_buf.as_raw_ptr(),
            down_off,
            n_experts_i32,
            m_i32,
            h_i32,
            down_bits,
            limit,
        )
    };
    if handle.is_null() {
        return Ok(None);
    }
    Ok(Some(MxArray::from_handle(
        handle,
        "hy_v4_flash gguf_routed_decode",
    )?))
}

/// Fused 2-kernel routed expert decode for resident MLX affine-4 g64 bf16
/// scales/biases (`rows <= 8`, `top <= 32`, asymmetric clamp `limit`).
/// Returns `Ok(None)` when Metal is unavailable or shapes/dtypes are unsupported.
#[allow(clippy::too_many_arguments)]
pub fn affine4_routed_decode(
    x: &MxArray,
    slots: &MxArray,
    weights: &MxArray,
    wg: &MxArray,
    sg: &MxArray,
    bg: &MxArray,
    wu: &MxArray,
    su: &MxArray,
    bu: &MxArray,
    wd: &MxArray,
    sd: &MxArray,
    bd: &MxArray,
    limit: f32,
) -> Result<Option<MxArray>> {
    let xf = as_f32(x)?;
    let wf = as_f32(weights)?;
    // SAFETY: all handles are live arrays owned by this call.
    let handle = unsafe {
        mlx_sys::mlx_hy4_affine4_routed_decode(
            xf.as_raw_ptr(),
            slots.as_raw_ptr(),
            wf.as_raw_ptr(),
            wg.as_raw_ptr(),
            sg.as_raw_ptr(),
            bg.as_raw_ptr(),
            wu.as_raw_ptr(),
            su.as_raw_ptr(),
            bu.as_raw_ptr(),
            wd.as_raw_ptr(),
            sd.as_raw_ptr(),
            bd.as_raw_ptr(),
            limit,
        )
    };
    if handle.is_null() {
        return Ok(None);
    }
    Ok(Some(MxArray::from_handle(
        handle,
        "hy_v4_flash affine4_routed_decode",
    )?))
}

/// Routed expert prefill gate+up+asymmetric-SwiGLU reading raw GGUF `block_q4_K`
/// directly via NAX TF32 MMA. Returns `Ok(None)` when NAX/Metal is unavailable
/// or disabled (`MLX_HY4_PREFILL_NAX=0`).
#[allow(clippy::too_many_arguments)]
pub fn gguf_prefill_indirect_gu(
    x: &MxArray,
    token_rows: &MxArray,
    tiles: &MxArray,
    gate_buf: &MxArray,
    gate_off: u64,
    up_buf: &MxArray,
    up_off: u64,
    n_experts: usize,
    m: usize,
    limit: f32,
) -> Result<Option<MxArray>> {
    let (Ok(n_experts_i32), Ok(m_i32)) = (i32::try_from(n_experts), i32::try_from(m)) else {
        return Ok(None);
    };
    let xf = as_f32(x)?;
    // SAFETY: all handles are live arrays owned by this call.
    let handle = unsafe {
        mlx_sys::mlx_hy4_gguf_prefill_indirect_gu(
            xf.as_raw_ptr(),
            token_rows.as_raw_ptr(),
            tiles.as_raw_ptr(),
            gate_buf.as_raw_ptr(),
            gate_off,
            up_buf.as_raw_ptr(),
            up_off,
            n_experts_i32,
            m_i32,
            limit,
        )
    };
    if handle.is_null() {
        return Ok(None);
    }
    Ok(Some(MxArray::from_handle(
        handle,
        "hy_v4_flash gguf_prefill_indirect_gu",
    )?))
}

/// Routed expert prefill down projection reading raw GGUF `block_q4_K` or
/// `block_q6_K` directly via FP16 NAX MMA. Accepts `Float16` or `Float32`
/// `hidden` without casting `Float16` back to `Float32`. Returns `Ok(None)`
/// when NAX/Metal is unavailable or disabled (`MLX_HY4_PREFILL_NAX=0`).
#[allow(clippy::too_many_arguments)]
pub fn gguf_prefill_indirect_down(
    hidden: &MxArray,
    tiles: &MxArray,
    down_buf: &MxArray,
    down_off: u64,
    n_experts: usize,
    h: usize,
    down_bits: i32,
) -> Result<Option<MxArray>> {
    let (Ok(n_experts_i32), Ok(h_i32)) = (i32::try_from(n_experts), i32::try_from(h)) else {
        return Ok(None);
    };
    let hf = match hidden.dtype()? {
        DType::Float16 | DType::Float32 => hidden.clone(),
        _ => hidden.astype(DType::Float32)?,
    };
    // SAFETY: all handles are live arrays owned by this call.
    let handle = unsafe {
        mlx_sys::mlx_hy4_gguf_prefill_indirect_down(
            hf.as_raw_ptr(),
            tiles.as_raw_ptr(),
            down_buf.as_raw_ptr(),
            down_off,
            n_experts_i32,
            h_i32,
            down_bits,
        )
    };
    if handle.is_null() {
        return Ok(None);
    }
    Ok(Some(MxArray::from_handle(
        handle,
        "hy_v4_flash gguf_prefill_indirect_down",
    )?))
}

/// GPU histogram, prefix sum, BM=32 tile emission, and scatter for MoE prefill
/// routing (`ids [rows, k]` u32 -> `(token_rows [R], tiles [max_tiles, 3],
/// inverse [rows, k], sorted_slots [R])` u32) with zero host synchronization.
pub fn moe_prefill_sort_tiles(
    ids: &MxArray,
    n_experts: usize,
) -> Result<Option<(MxArray, MxArray, MxArray, MxArray)>> {
    let Ok(ne_i32) = i32::try_from(n_experts) else {
        return Ok(None);
    };
    let mut out_tr = std::ptr::null_mut();
    let mut out_tl = std::ptr::null_mut();
    let mut out_inv = std::ptr::null_mut();
    let mut out_sl = std::ptr::null_mut();
    // SAFETY: `ids` is live for the call; output handles are initialized on true.
    let ok = unsafe {
        mlx_sys::mlx_hy4_moe_prefill_sort_tiles(
            ids.as_raw_ptr(),
            ne_i32,
            &mut out_tr,
            &mut out_tl,
            &mut out_inv,
            &mut out_sl,
        )
    };
    if !ok {
        return Ok(None);
    }
    Ok(Some((
        MxArray::from_handle(out_tr, "hy_v4_flash moe_prefill_sort_tiles token_rows")?,
        MxArray::from_handle(out_tl, "hy_v4_flash moe_prefill_sort_tiles tiles")?,
        MxArray::from_handle(out_inv, "hy_v4_flash moe_prefill_sort_tiles inverse")?,
        MxArray::from_handle(out_sl, "hy_v4_flash moe_prefill_sort_tiles sorted_slots")?,
    )))
}

/// Circular row write of `src [s, width]` into `buf [cap, width]` starting at
/// `start_slot`, wrapping modulo `cap`. Donates `buf` in place when it has no
/// other live references.
pub fn ring_write(buf: &MxArray, src: &MxArray, start_slot: usize, cap: usize) -> Result<MxArray> {
    let start_i32 = i32::try_from(start_slot)
        .map_err(|_| bad(format!("ring_write start_slot {start_slot} exceeds i32")))?;
    let cap_i32 =
        i32::try_from(cap).map_err(|_| bad(format!("ring_write cap {cap} exceeds i32")))?;
    // SAFETY: buf and src are live array handles for the call.
    let handle = unsafe {
        mlx_sys::mlx_hy4_ring_write(buf.as_raw_ptr(), src.as_raw_ptr(), start_i32, cap_i32)
    };
    if handle.is_null() {
        return Err(bad(format!(
            "ring_write rejected buf {:?}, src {:?}, start_slot {start_slot}, cap {cap}",
            dims(buf)?,
            dims(src)?,
        )));
    }
    MxArray::from_handle(handle, "hy_v4_flash ring_write")
}

/// Fused indexer group scoring directly from a flat `[cap_g, 128]` f16 or
/// paged `[n_blocks, index_rows, 128]` f16 pool. Returns `[S, n_groups]` f32,
/// or `Ok(None)` when Metal or shape is unsupported.
pub(crate) fn try_indexer_score(
    q: &MxArray,
    w: &MxArray,
    k_pool: &MxArray,
    page_ids: Option<&MxArray>,
    n_groups: usize,
    start_g: usize,
) -> Result<Option<MxArray>> {
    let (Ok(ng_i32), Ok(sg_i32)) = (i32::try_from(n_groups), i32::try_from(start_g)) else {
        return Ok(None);
    };
    if ng_i32 < 1 || sg_i32 < 0 {
        return Ok(None);
    }
    let qf = as_f32(q)?;
    let wf = as_f32(w)?;
    let pages_ptr = page_ids.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    // SAFETY: all handles are live for the call; returns a new array or null.
    let handle = unsafe {
        mlx_sys::mlx_hy4_indexer_score(
            qf.as_raw_ptr(),
            wf.as_raw_ptr(),
            k_pool.as_raw_ptr(),
            pages_ptr,
            ng_i32,
            sg_i32,
        )
    };
    if handle.is_null() {
        return Ok(None);
    }
    Ok(Some(MxArray::from_handle(
        handle,
        "hy_v4_flash indexer_score",
    )?))
}

/// Fused radix top-k group selection + cell expansion + causal tail + optional
/// paged slot mapping (`page_ids[c / block_size] * block_size + c % block_size`)
/// padded to `kmax` with `-1`. Returns `[S, kmax]` i32, or `Ok(None)` when
/// Metal or shape is unsupported.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_indexer_topk_cells(
    scores: &MxArray,
    visible_groups: &MxArray,
    tail_starts: &MxArray,
    tail_lens: &MxArray,
    page_ids: Option<&MxArray>,
    n_top: usize,
    kpool: usize,
    kmax: usize,
    block_size: usize,
) -> Result<Option<MxArray>> {
    let (Ok(nt_i32), Ok(kp_i32), Ok(km_i32), Ok(bs_i32)) = (
        i32::try_from(n_top),
        i32::try_from(kpool),
        i32::try_from(kmax),
        i32::try_from(block_size),
    ) else {
        return Ok(None);
    };
    let sf = as_f32(scores)?;
    let pages_ptr = page_ids.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    // SAFETY: all handles are live for the call; returns a new array or null.
    let handle = unsafe {
        mlx_sys::mlx_hy4_indexer_topk_cells(
            sf.as_raw_ptr(),
            visible_groups.as_raw_ptr(),
            tail_starts.as_raw_ptr(),
            tail_lens.as_raw_ptr(),
            pages_ptr,
            nt_i32,
            kp_i32,
            km_i32,
            bs_i32,
        )
    };
    if handle.is_null() {
        return Ok(None);
    }
    Ok(Some(MxArray::from_handle(
        handle,
        "hy_v4_flash indexer_topk_cells",
    )?))
}

/// Fused `rms_norm(q_a)`, `rms_norm(kv_lora)`, interleaved precise `rope(k_pe)`,
/// and optional f16 cast / split `[M, half]` × 2 output directly from strided
/// `q_kv_gate` / `q_kv` `[M, total]`. Returns `(qr, kv0, kv1_opt)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_mla_prep_q_kv(
    all: &MxArray,
    q_norm_w: &MxArray,
    kv_norm_w: &MxArray,
    pos: &MxArray,
    inv_freq: &MxArray,
    q_lora: i64,
    kv_lora: i64,
    rope_dim: i64,
    norm_eps: f32,
    out_f16: bool,
    split_out: bool,
) -> Result<Option<(MxArray, MxArray, Option<MxArray>)>> {
    let (Ok(ql), Ok(kvl), Ok(rd)) = (
        i32::try_from(q_lora),
        i32::try_from(kv_lora),
        i32::try_from(rope_dim),
    ) else {
        return Ok(None);
    };
    let allf = as_f32(all)?;
    let qnwf = as_f32(q_norm_w)?;
    let kvnwf = as_f32(kv_norm_w)?;
    let invf = as_f32(inv_freq)?;
    let pos_arr = if matches!(pos.dtype()?, DType::Float32 | DType::Int32) {
        pos.clone()
    } else {
        pos.astype(DType::Float32)?
    };
    let mut out_qr = std::ptr::null_mut();
    let mut out_kv0 = std::ptr::null_mut();
    let mut out_kv1 = std::ptr::null_mut();
    // SAFETY: input handles are live for the call; out pointers receive new arrays on true.
    let ok = unsafe {
        mlx_sys::mlx_hy4_mla_prep_q_kv(
            allf.as_raw_ptr(),
            qnwf.as_raw_ptr(),
            kvnwf.as_raw_ptr(),
            pos_arr.as_raw_ptr(),
            invf.as_raw_ptr(),
            ql,
            kvl,
            rd,
            norm_eps,
            out_f16,
            split_out,
            &mut out_qr,
            &mut out_kv0,
            if split_out {
                &mut out_kv1
            } else {
                std::ptr::null_mut()
            },
        )
    };
    if !ok {
        return Ok(None);
    }
    let qr = MxArray::from_handle(out_qr, "hy_v4_flash mla_prep_q_kv qr")?;
    let kv0 = MxArray::from_handle(out_kv0, "hy_v4_flash mla_prep_q_kv kv0")?;
    let kv1 = if split_out {
        Some(MxArray::from_handle(
            out_kv1,
            "hy_v4_flash mla_prep_q_kv kv1",
        )?)
    } else {
        None
    };
    Ok(Some((qr, kv0, kv1)))
}

/// Fused `k_b` grouped Q8 (`K = 192`) + strided `q_pe` precise interleaved RoPE
/// directly into `[M, H, N + rope_dim]` f32 (`[M, H, 576]`).
pub(crate) fn try_q8_grouped_k192_rope(
    q_b: &MxArray,
    k_b: &Dense,
    heads: i64,
    pos: &MxArray,
    inv_freq: &MxArray,
) -> Result<Option<MxArray>> {
    let Some((qw, scales, biases)) = k_b.qmm_parts() else {
        return Ok(None);
    };
    if q_b.ndim()? != 3 || q_b.shape_at(1)? != heads || !(1..=16).contains(&q_b.shape_at(0)?) {
        return Ok(None);
    }
    let qbf = as_f32(q_b)?;
    let invf = as_f32(inv_freq)?;
    let pos_arr = if matches!(pos.dtype()?, DType::Float32 | DType::Int32) {
        pos.clone()
    } else {
        pos.astype(DType::Float32)?
    };
    let biases_ptr = biases.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    // SAFETY: all handles are live for the call; returns a new array or null.
    let h = unsafe {
        mlx_sys::mlx_hy4_q8_grouped_k192_rope(
            qbf.as_raw_ptr(),
            qw.as_raw_ptr(),
            scales.as_raw_ptr(),
            biases_ptr,
            pos_arr.as_raw_ptr(),
            invf.as_raw_ptr(),
        )
    };
    if h.is_null() {
        return Ok(None);
    }
    Ok(Some(MxArray::from_handle(
        h,
        "hy_v4_flash q8_grouped_k192_rope",
    )?))
}

/// Fused indexer kpool-4 compression (`d = 128`): gathers open rows from the
/// circular `tail_buf` + `tok`, applies `w_val`, `w_gate + ape`, per-channel
/// softmax weighted sum, RMSNorm, and trailing precise interleaved RoPE,
/// writing `[n_groups, 128]` directly in `out_dtype`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_kpool_compress(
    tail_buf: Option<&MxArray>,
    tok: &MxArray,
    w_val: &MxArray,
    w_gate: &MxArray,
    ape: &MxArray,
    comp_norm: &MxArray,
    inv_freq: &MxArray,
    n_groups: usize,
    first_group: usize,
    open_len: usize,
    tail_start_slot: usize,
    tail_cap: usize,
    norm_eps: f32,
    out_dtype: DType,
) -> Result<Option<MxArray>> {
    let (Ok(ng), Ok(fg), Ok(ol), Ok(ts), Ok(tc)) = (
        i32::try_from(n_groups),
        i32::try_from(first_group),
        i32::try_from(open_len),
        i32::try_from(tail_start_slot),
        i32::try_from(tail_cap),
    ) else {
        return Ok(None);
    };
    let out_code = match out_dtype {
        DType::Float32 => 0,
        DType::Float16 => 1,
        DType::BFloat16 => 2,
        _ => return Ok(None),
    };
    let tail_f = tail_buf.map(as_f32).transpose()?;
    let tok_f = as_f32(tok)?;
    let wv_f = as_f32(w_val)?;
    let wg_f = as_f32(w_gate)?;
    let ape_f = as_f32(ape)?;
    let cn_f = as_f32(comp_norm)?;
    let inv_f = as_f32(inv_freq)?;
    let tail_ptr = tail_f
        .as_ref()
        .map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    // SAFETY: all handles are live for the call; returns a new array or null.
    let h = unsafe {
        mlx_sys::mlx_hy4_kpool_compress(
            tail_ptr,
            tok_f.as_raw_ptr(),
            wv_f.as_raw_ptr(),
            wg_f.as_raw_ptr(),
            ape_f.as_raw_ptr(),
            cn_f.as_raw_ptr(),
            inv_f.as_raw_ptr(),
            ng,
            fg,
            ol,
            ts,
            tc,
            norm_eps,
            out_code,
        )
    };
    if h.is_null() {
        return Ok(None);
    }
    Ok(Some(MxArray::from_handle(h, "hy_v4_flash kpool_compress")?))
}

/// Single-dispatch partial interleaved precise RoPE on the trailing `2 * inv_freq.len()`
/// dims of `x` (`[M, D]` or `[M, H, D]`), copying leading dims untouched.
pub(crate) fn try_partial_rope_tail(
    x: &MxArray,
    pos: &MxArray,
    inv_freq: &MxArray,
) -> Result<Option<MxArray>> {
    let nd = x.ndim()?;
    if nd != 2 && nd != 3 {
        return Ok(None);
    }
    let xf = as_f32(x)?;
    let invf = as_f32(inv_freq)?;
    let pos_arr = if matches!(pos.dtype()?, DType::Float32 | DType::Int32) {
        pos.clone()
    } else {
        pos.astype(DType::Float32)?
    };
    // SAFETY: all handles are live for the call; returns a new array or null.
    let h = unsafe {
        mlx_sys::mlx_hy4_partial_rope_tail(xf.as_raw_ptr(), pos_arr.as_raw_ptr(), invf.as_raw_ptr())
    };
    if h.is_null() {
        return Ok(None);
    }
    Ok(Some(MxArray::from_handle(
        h,
        "hy_v4_flash partial_rope_tail",
    )?))
}

/// Fused Engram `rms(rows) * lookup_norm` flattened to `[M, cols * hd]` f32.
pub(crate) fn try_engram_norm_rows(
    rows: &MxArray,
    lookup_norm: &MxArray,
    eps: f32,
) -> Result<Option<MxArray>> {
    let rf = as_f32(rows)?;
    let nf = as_f32(lookup_norm)?;
    // SAFETY: all handles are live for the call; returns a new array or null.
    let h = unsafe { mlx_sys::mlx_hy4_engram_norm_rows(rf.as_raw_ptr(), nf.as_raw_ptr(), eps) };
    if h.is_null() {
        return Ok(None);
    }
    Ok(Some(MxArray::from_handle(
        h,
        "hy_v4_flash engram_norm_rows",
    )?))
}

/// Fused Engram tail: `kv_b` add + `rms(x)` + `rms(key)` + dot + `stream_gate`
/// + `hc_scale` + `signal` + optional masked residual add into `[M, HC, D]` f32.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_engram_gate_add(
    x: &MxArray,
    kv_raw: &MxArray,
    kv_b: &MxArray,
    gate_w: &MxArray,
    pre_gamma: &MxArray,
    hc_scale: &MxArray,
    keep_mask: Option<&MxArray>,
    eps: f32,
) -> Result<Option<MxArray>> {
    let xf = as_f32(x)?;
    let kvf = as_f32(kv_raw)?;
    let kvbf = as_f32(kv_b)?;
    let gwf = as_f32(gate_w)?;
    let pgf = as_f32(pre_gamma)?;
    let hsf = as_f32(hc_scale)?;
    let mask_ptr = keep_mask.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    // SAFETY: all handles are live for the call; returns a new array or null.
    let h = unsafe {
        mlx_sys::mlx_hy4_engram_gate_add(
            xf.as_raw_ptr(),
            kvf.as_raw_ptr(),
            kvbf.as_raw_ptr(),
            gwf.as_raw_ptr(),
            pgf.as_raw_ptr(),
            hsf.as_raw_ptr(),
            mask_ptr,
            eps,
        )
    };
    if h.is_null() {
        return Ok(None);
    }
    Ok(Some(MxArray::from_handle(
        h,
        "hy_v4_flash engram_gate_add",
    )?))
}
