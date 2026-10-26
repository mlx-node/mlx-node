//! KV compressor of the source layers (`model.py` `Compressor`): ratio 1 is
//! `kv_norm(wkv x)`; ratio 2 pools each group of 2 rows with a softmax gate
//! in f32, carrying an incomplete group across calls.

use napi::{Error, Result};

use super::attention::{linear, rms_norm};
use crate::array::{DType, MxArray};

pub struct CompressorWeights {
    pub wkv: MxArray,
    /// ratio > 1 only
    pub wgate: Option<MxArray>,
    pub norm: MxArray,
}

impl CompressorWeights {
    pub fn from_gguf(
        layer: usize,
        ratio: u32,
        mut get: impl FnMut(&str) -> Result<MxArray>,
    ) -> Result<Self> {
        let mut t = |s: &str| get(&format!("blk.{layer}.attn_compressor_{s}.weight"));
        Ok(Self {
            wkv: t("kv")?,
            wgate: if ratio > 1 { Some(t("gate")?) } else { None },
            norm: t("norm")?,
        })
    }
}

/// The rows of an incomplete group (`kv_state` / `score_state`), f32.
#[derive(Default, Clone)]
pub struct CompressorTail {
    pub kv: Option<MxArray>,
    pub score: Option<MxArray>,
}

pub struct Compressor {
    ratio: i64,
    eps: f32,
    w: CompressorWeights,
}

impl Compressor {
    pub fn new(ratio: u32, eps: f32, w: CompressorWeights) -> Result<Self> {
        if ratio == 0 || (ratio > 1) != w.wgate.is_some() {
            return Err(Error::from_reason(format!(
                "compressor: ratio {ratio} with gate {}",
                w.wgate.is_some()
            )));
        }
        Ok(Self {
            ratio: i64::from(ratio),
            eps,
            w,
        })
    }

    /// `x` `[s, dim]` at positions `start..start + s` -> the latents of the
    /// groups completed by this chunk, `[g, head_dim]` before RoPE in `x`'s
    /// dtype, or `None`. Group `j` covers positions `j*ratio..(j+1)*ratio`.
    pub fn forward(
        &self,
        x: &MxArray,
        start: usize,
        tail: &mut CompressorTail,
    ) -> Result<Option<MxArray>> {
        let r = self.ratio;
        let s = x.shape_at(0)?;
        if r == 1 {
            return Ok(Some(rms_norm(
                &linear(x, &self.w.wkv)?,
                &self.w.norm,
                self.eps,
            )?));
        }
        let Some(gate) = &self.w.wgate else {
            return Err(Error::from_reason("compressor: ratio > 1 without a gate"));
        };
        let xf = x.astype(DType::Float32)?;
        let (mut kv, mut score) = (linear(&xf, &self.w.wkv)?, linear(&xf, gate)?);
        let held = (start as i64) % r;
        let have = tail.kv.as_ref().map_or(Ok(0), |t| t.shape_at(0))?;
        if have != held {
            return Err(Error::from_reason(format!(
                "compressor: {have} tail rows at position {start}, expected {held}"
            )));
        }
        if let (Some(tk), Some(ts)) = (&tail.kv, &tail.score) {
            kv = MxArray::concatenate(tk, &kv, 0)?;
            score = MxArray::concatenate(ts, &score, 0)?;
        }
        let total = held + s;
        let (groups, rem) = (total / r, total % r);
        let hd = kv.shape_at(1)?;
        let rows = |a: &MxArray, lo: i64, hi: i64| a.slice(&[lo, 0], &[hi, hd]);
        let (new_kv, new_score) = if rem > 0 {
            // independent copies: a slice view would pin the whole projections
            let (k, sc) = (
                rows(&kv, total - rem, total)?.deep_copy()?,
                rows(&score, total - rem, total)?.deep_copy()?,
            );
            MxArray::eval_arrays_with_context(&[&k, &sc], "deepseek_v41 compressor tail")?;
            (Some(k), Some(sc))
        } else {
            (None, None)
        };
        *tail = CompressorTail {
            kv: new_kv,
            score: new_score,
        };
        if groups == 0 {
            return Ok(None);
        }
        let shape = [groups, r, hd];
        let kv = rows(&kv, 0, groups * r)?.reshape(&shape)?;
        let score = rows(&score, 0, groups * r)?.reshape(&shape)?;
        // softmax over the group axis: exp(s - max) / sum
        let e = score.sub(&score.max(Some(&[1]), Some(true))?)?.exp()?;
        let p = e.div(&e.sum(Some(&[1]), Some(true))?)?;
        let pooled = kv.mul(&p)?.sum(Some(&[1]), Some(false))?;
        Ok(Some(rms_norm(
            &pooled.astype(x.dtype()?)?,
            &self.w.norm,
            self.eps,
        )?))
    }
}
