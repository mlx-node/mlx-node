//! Engram block (llama.cpp PR #32517 `build_engram`), f32 math: a token's
//! F8 table rows -> per-row RMS · `lookup_norm` -> `engram_kv` + `kv_b`
//! -> one key per HC stream and one value; stream c gains
//! `rms(value) · pre_gamma · (gate_c + hc_scale_c)`. Positions holding
//! the pad, image or video id get no delta (HF `build_engram_inputs`,
//! `modeling_hy_v4_flash.py:1894-1898`, applied at `:800`); they still
//! hash into the n-grams of the positions after them.

use napi::{Error, Result};

use super::moe::qlinear;
use super::weights::EngramWeights;
use crate::array::{DType, MxArray};
use crate::models::engram_hash::{EngramConfig, EngramHasher};
use crate::nn::Activations;

/// `HY_V4_FLASH_ENGRAM_EPS` (f32 machine epsilon, 2^-23).
pub const ENGRAM_EPS: f32 = f32::EPSILON;
/// HF `image_token_id` / `video_token_id` (`config.json`; no GGUF key).
pub const IMAGE_TOKEN: u32 = 120_120;
pub const VIDEO_TOKEN: u32 = 120_124;
/// Floor of `|dot|` under the gate's square root.
const DOT_FLOOR: f64 = 1e-6;

fn bad(msg: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("Hy4 Engram: {msg}"))
}

fn dims(x: &MxArray) -> Result<Vec<i64>> {
    Ok(x.shape()?.as_ref().to_vec())
}

fn dims2(a: &MxArray, what: &str) -> Result<[i64; 2]> {
    match dims(a)?.as_slice() {
        &[r, c] => Ok([r, c]),
        s => Err(bad(format!("{what} must be 2-D, got {s:?}"))),
    }
}

/// `x · rsqrt(mean(x²) + eps)` over the last axis, in f32.
fn rms(x: &MxArray) -> Result<MxArray> {
    super::ops::rms_norm_unweighted(&x.astype(DType::Float32)?, ENGRAM_EPS)
}

/// Table rows `[n, cols, head_dim]` (row-major f32) -> `rms(row) ·
/// lookup_norm`, flattened to `[n, cols · head_dim]`.
pub fn normed_rows(rows: &[f32], n: i64, w: &EngramWeights) -> Result<MxArray> {
    let [cols, hd] = dims2(&w.lookup_norm, "engram lookup_norm")?;
    if n <= 0 || rows.len() as i64 != n * cols * hd {
        return Err(bad(format!(
            "{} row values for {n} positions of {cols} x {hd}",
            rows.len()
        )));
    }
    let r = MxArray::from_float32(rows, &[n, cols, hd])?;
    if let Some(out) = super::ops::try_engram_norm_rows(&r, &w.lookup_norm, ENGRAM_EPS)? {
        return Ok(out);
    }
    rms(&r)?
        .mul(&w.lookup_norm.astype(DType::Float32)?)?
        .reshape(&[n, cols * hd])
}

/// `engram_kv · r + kv_b` -> `[n, hc + 1, d]`: the hc keys, then the value.
pub fn key_value(r: &MxArray, w: &EngramWeights) -> Result<MxArray> {
    let [heads, d] = dims2(&w.kv_b, "engram kv_b")?;
    let n = r.shape_at(0)?;
    if w.kv.shape_at(0)? != heads * d {
        return Err(bad(format!(
            "engram_kv has {} rows, kv_b is [{heads}, {d}]",
            w.kv.shape_at(0)?
        )));
    }
    qlinear(&r.astype(DType::Float32)?, &w.kv)?
        .reshape(&[n, heads, d])?
        .add(&w.kv_b.astype(DType::Float32)?)
}

/// `⟨rms(x_c), rms(k_c) · gate_c⟩ / √d`: x, key `[n, hc, d]`, gate
/// `[hc, d]` -> `[n, hc]`.
pub fn stream_dot(x: &MxArray, key: &MxArray, gate: &MxArray) -> Result<MxArray> {
    let d = x.shape_at(2)?;
    rms(x)?
        .mul(&rms(key)?)?
        .mul(&gate.astype(DType::Float32)?)?
        .sum(Some(&[-1]), None)?
        .mul_scalar((d as f64).powf(-0.5))
}

/// `σ(sign(dot) · √max(|dot|, 1e-6))` with sign(0) = +1.
pub fn stream_gate(dot: &MxArray) -> Result<MxArray> {
    let a = dot.abs()?.clip(Some(DOT_FLOOR), None)?.sqrt()?;
    let zero = MxArray::from_float32(&[0.0], &[1])?;
    Activations::sigmoid(&dot.less(&zero)?.where_(&a.negative()?, &a)?)
}

/// `rms(value) · pre_gamma`: value `[n, d]`.
pub fn signal(value: &MxArray, pre_gamma: &MxArray) -> Result<MxArray> {
    super::ops::rms_norm(&value.astype(DType::Float32)?, pre_gamma, ENGRAM_EPS)
}

/// Whether each id's position gets the Engram delta: not the Engram pad
/// id (`engram.pad_token_id`), the image or the video id.
pub fn keeps(ids: &[u32], pad: usize) -> Vec<bool> {
    ids.iter()
        .map(|&t| t as usize != pad && t != IMAGE_TOKEN && t != VIDEO_TOKEN)
        .collect()
}

/// x `[n, hc, d]` plus this layer's Engram delta from its table rows
/// `[n, cols, head_dim]` on the rows `keep` marks (the others stay x);
/// in x's dtype.
pub fn forward(x: &MxArray, rows: &[f32], keep: &[bool], w: &EngramWeights) -> Result<MxArray> {
    let &[n, hc, d] = dims(x)?.as_slice() else {
        return Err(bad(format!(
            "streams must be [n, hc, d], got {:?}",
            dims(x)?
        )));
    };
    if dims2(&w.gate, "engram gate")? != [hc, d]
        || dims2(&w.hc_scale, "engram hc_scale")? != [hc, d]
    {
        return Err(bad(format!(
            "gate {:?} / hc_scale {:?} for streams [{n}, {hc}, {d}]",
            dims(&w.gate)?,
            dims(&w.hc_scale)?
        )));
    }
    if dims2(&w.kv_b, "engram kv_b")? != [hc + 1, d] {
        return Err(bad(format!(
            "kv_b {:?} for {hc} streams of {d}",
            dims(&w.kv_b)?
        )));
    }
    if keep.len() as i64 != n {
        return Err(bad(format!(
            "{} mask entries for {n} positions",
            keep.len()
        )));
    }
    let nr = normed_rows(rows, n, w)?;
    let [heads, _] = dims2(&w.kv_b, "engram kv_b")?;
    if w.kv.shape_at(0)? != heads * d {
        return Err(bad(format!(
            "engram_kv has {} rows, kv_b is [{heads}, {d}]",
            w.kv.shape_at(0)?
        )));
    }
    let kv_raw = qlinear(&nr, &w.kv)?;
    let xf = if x.dtype()? == DType::Float32 {
        x.clone()
    } else {
        x.astype(DType::Float32)?
    };
    let all_keep = keep.iter().all(|&k| k);
    let mask_arr = if all_keep {
        None
    } else {
        let mask: Vec<u8> = keep.iter().map(|&k| u8::from(k)).collect();
        Some(MxArray::from_uint8(&mask, &[n])?)
    };
    if let Some(y) = super::ops::try_engram_gate_add(
        &xf,
        &kv_raw,
        &w.kv_b,
        &w.gate,
        &w.pre_gamma,
        &w.hc_scale,
        mask_arr.as_ref(),
        ENGRAM_EPS,
    )? {
        return if x.dtype()? == DType::Float32 {
            Ok(y)
        } else {
            y.astype(x.dtype()?)
        };
    }
    let kv = kv_raw
        .reshape(&[n, heads, d])?
        .add(&w.kv_b.astype(DType::Float32)?)?;
    let key = kv.slice(&[0, 0, 0], &[n, hc, d])?;
    let value = kv.slice(&[0, hc, 0], &[n, hc + 1, d])?.reshape(&[n, d])?;
    let gate = stream_gate(&stream_dot(&xf, &key, &w.gate)?)?;
    let scale = gate
        .reshape(&[n, hc, 1])?
        .add(&w.hc_scale.astype(DType::Float32)?)?;
    let s = signal(&value, &w.pre_gamma)?.reshape(&[n, 1, d])?;
    let y = xf.add(&s.mul(&scale)?)?;
    let y = if all_keep {
        y
    } else {
        let mask: Vec<u8> = keep.iter().map(|&k| u8::from(k)).collect();
        MxArray::from_uint8(&mask, &[n, 1, 1])?
            .not_equal(&MxArray::from_uint8(&[0], &[1])?)?
            .where_(&y, &xf)?
    };
    y.astype(x.dtype()?)
}

/// Row ids of the n-gram hash for one call, hashing only the ids the
/// n-grams reach back to.
pub struct Hasher {
    inner: EngramHasher,
    back: usize,
    n_layers: usize,
    n_cols: usize,
}

impl Hasher {
    pub fn new(cfg: &EngramConfig) -> Result<Self> {
        if cfg.max_ngram < 2 || cfg.n_heads == 0 || cfg.layers.is_empty() {
            return Err(bad("n-gram size < 2, no heads or no layers"));
        }
        let want = cfg.layers.len() * cfg.n_cols();
        if cfg.multipliers.len() != cfg.layers.len() * cfg.max_ngram
            || cfg.head_offsets.len() != want
            || cfg.head_vocab_sizes.len() != want
            || cfg.head_vocab_sizes.contains(&0)
        {
            return Err(bad(
                "multipliers / head offsets / bucket sizes do not fit the layers",
            ));
        }
        Ok(Self {
            inner: EngramHasher::new(cfg)?,
            back: cfg.max_ngram - 1,
            n_layers: cfg.layers.len(),
            n_cols: cfg.n_cols(),
        })
    }

    pub fn n_cols(&self) -> usize {
        self.n_cols
    }

    pub fn n_layers(&self) -> usize {
        self.n_layers
    }

    /// Global table rows `[pos][layer][col]` of positions
    /// `start..tokens.len()`; `tokens` is the sequence history.
    pub fn call_rows(&self, tokens: &[u32], start: usize) -> Result<Vec<u32>> {
        if start > tokens.len() {
            return Err(bad(format!(
                "call starts at {start}, history has {} tokens",
                tokens.len()
            )));
        }
        let lo = start.saturating_sub(self.back);
        Ok(self.inner.rows(&tokens[lo..], start - lo))
    }

    /// Engram layer `e`'s ids `[pos][col]` out of `call_rows`.
    pub fn layer_ids(&self, rows: &[u32], e: usize) -> Result<Vec<u32>> {
        let per_pos = self.n_layers * self.n_cols;
        if e >= self.n_layers || !rows.len().is_multiple_of(per_pos) {
            return Err(bad(format!(
                "layer {e} of {}, {} ids for {per_pos} per position",
                self.n_layers,
                rows.len()
            )));
        }
        Ok(rows
            .chunks_exact(per_pos)
            .flat_map(|p| &p[e * self.n_cols..(e + 1) * self.n_cols])
            .copied()
            .collect())
    }
}
