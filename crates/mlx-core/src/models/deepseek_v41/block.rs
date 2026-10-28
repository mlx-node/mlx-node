//! One transformer block (`model.py` `Block`): hyper-connection mixes around
//! attention and the MoE, with the staggered pre-mix (attention collapses
//! the copies with the previous FFN's `pre`, the FFN with this attention's).

use napi::Result;

use super::attention::{CompressedAttention, FakeQuant, WindowAttention, rms_norm};
use super::cache::{CompressedCache, ForwardShared, WindowCache};
use super::config::DeepseekV41Config;
use super::model::fp32_site;
use super::moe::{Moe, Routes};
use super::ops::{hc_mixes, hc_post, hc_pre};
use super::stats::{Phase, prof_end, prof_start};
use super::store::Dense;
use crate::array::MxArray;

/// FFN rows per chunk: the routed scratch is `rows x k x (2 inter + dim)`.
pub const FFN_CHUNK_ROWS: i64 = 256;

#[cfg(test)]
thread_local! {
    static TEST_CHUNK_ROWS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// `rows`, or the test cap on every row-chunk loop (FFN, sparse attention,
/// indexer) of this thread: tiny prompts fit one chunk otherwise, which
/// leaves the stage-P5 chunk skip nothing to skip.
pub(crate) fn chunk_cap(rows: usize) -> usize {
    #[cfg(test)]
    {
        let c = TEST_CHUNK_ROWS.with(|c| c.get());
        if c > 0 {
            return rows.min(c);
        }
    }
    rows
}

/// Test hook for `chunk_cap` (this thread; 0 = off).
#[cfg(test)]
pub(crate) fn set_test_chunk_rows(n: usize) {
    TEST_CHUNK_ROWS.with(|c| c.set(n));
}

pub enum Attn {
    Window(Box<WindowAttention>),
    Compressed(Box<CompressedAttention>),
}

/// `hc_{attn,ffn}_{fn,base,scale}` (f32).
pub struct HcWeights {
    pub fn_: MxArray,
    pub base: MxArray,
    pub scale: MxArray,
}

impl HcWeights {
    fn from_gguf(
        layer: usize,
        tag: &str,
        get: &mut impl FnMut(&str) -> Result<Dense>,
    ) -> Result<Self> {
        let mut t = |s: &str| get(&format!("blk.{layer}.hc_{tag}_{s}.weight"))?.array();
        Ok(Self {
            fn_: t("fn")?,
            base: t("base")?,
            scale: t("scale")?,
        })
    }
}

pub struct Block {
    pub layer: usize,
    pub attn: Attn,
    pub moe: Moe,
    attn_norm: MxArray,
    ffn_norm: MxArray,
    hc_attn: HcWeights,
    hc_ffn: HcWeights,
    eps: f32,
    hc: i64,
    iters: usize,
    hc_eps: f32,
}

impl Block {
    pub fn from_gguf(
        cfg: &DeepseekV41Config,
        layer: usize,
        fq: FakeQuant,
        moe: Moe,
        mut get: impl FnMut(&str) -> Result<Dense>,
    ) -> Result<Self> {
        let attn = if cfg.compress_ratios[layer] == 0 {
            let w = super::attention::AttentionWeights::from_gguf(layer, &mut get)?;
            Attn::Window(Box::new(WindowAttention::new(cfg, layer, w, fq)?))
        } else {
            Attn::Compressed(Box::new(CompressedAttention::from_gguf(
                cfg, layer, fq, &mut get,
            )?))
        };
        Ok(Self {
            layer,
            attn,
            moe,
            attn_norm: get(&format!("blk.{layer}.attn_norm.weight"))?.array()?,
            ffn_norm: get(&format!("blk.{layer}.ffn_norm.weight"))?.array()?,
            hc_attn: HcWeights::from_gguf(layer, "attn", &mut get)?,
            hc_ffn: HcWeights::from_gguf(layer, "ffn", &mut get)?,
            eps: cfg.norm_eps,
            hc: cfg.hc_mult as i64,
            iters: cfg.hc_sinkhorn_iters,
            hc_eps: cfg.hc_eps,
        })
    }

    fn mixes(&self, x: &MxArray, w: &HcWeights) -> Result<(MxArray, MxArray, MxArray)> {
        let m = hc_mixes(
            x,
            &w.fn_,
            &w.scale,
            &w.base,
            self.eps,
            self.hc,
            self.iters,
            self.hc_eps,
        )?;
        for a in [&m.0, &m.1, &m.2] {
            fp32_site("hc_mixes", self.layer, a)?;
        }
        Ok(m)
    }

    /// `x` `[s, hc, dim]` at `start..start + s` with `pre_mix` `[s, hc]` ->
    /// (`x`, this FFN's `pre` for the next block, the routes, the index
    /// scores when the attention exports them). `src` is the
    /// layer's KV source cache (compressed layers only). Stage P5: output
    /// rows below `from` are not needed by anyone; the attention and FFN
    /// chunks that hold only such rows are skipped (their output rows are
    /// stale, never read), every other op still runs on all `s` rows so a
    /// needed row's bits do not depend on `from`. `next` is the next
    /// layer's FFN and its own `from` (read-ahead prediction rows).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &MxArray,
        start: usize,
        pre_mix: &MxArray,
        win: &mut WindowCache,
        src: Option<&mut CompressedCache>,
        shared: &mut ForwardShared,
        next: Option<(&Moe, usize)>,
        from: usize,
    ) -> Result<(MxArray, MxArray, Routes, Option<Vec<f32>>)> {
        let tp = prof_start();
        let (attn_pre, attn_post, attn_comb) = self.mixes(x, &self.hc_attn)?;
        let a = rms_norm(&hc_pre(x, pre_mix)?, &self.attn_norm, self.eps)?;
        prof_end(Phase::Hc, tp, &[&attn_pre, &attn_post, &attn_comb, &a])?;
        let (a, scores) = match (&self.attn, src) {
            (Attn::Window(w), _) => (w.forward_from(&a, start, win, from)?, None),
            (Attn::Compressed(c), Some(src)) => {
                c.forward_from(&a, start, win, src, shared, from)?
            }
            (Attn::Compressed(_), None) => {
                return Err(napi::Error::from_reason(format!(
                    "layer {}: compressed attention without its source cache",
                    self.layer
                )));
            }
        };
        let tp = prof_start();
        let x = hc_post(&a, x, &attn_post, &attn_comb)?;

        let (ffn_pre, ffn_post, ffn_comb) = self.mixes(&x, &self.hc_ffn)?;
        let f = rms_norm(&hc_pre(&x, &attn_pre)?, &self.ffn_norm, self.eps)?;
        prof_end(Phase::Hc, tp, &[&x, &ffn_pre, &ffn_post, &ffn_comb, &f])?;
        let (s, dim) = (f.shape_at(0)?, f.shape_at(1)?);
        // stage P4: the next layer's experts are read while this one computes
        if let Some((nx, nfrom)) = next
            && s > 1
        {
            // from the first FFN chunk the next layer will run
            let chunk = chunk_cap(FFN_CHUNK_ROWS as usize) as i64;
            let lo = (nfrom as i64 / chunk * chunk).min(s - 1);
            nx.prefetch_from(&f.slice(&[lo, 0], &[s, dim])?)?;
        }
        let mut outs = Vec::new();
        let mut routes = Routes {
            k: self.moe.k(),
            ids: Vec::new(),
            weights: Vec::new(),
            biased: Vec::new(),
        };
        let chunk = chunk_cap(FFN_CHUNK_ROWS as usize) as i64;
        let mut lo = 0;
        while lo < s {
            let hi = (lo + chunk).min(s);
            if hi as usize <= from {
                outs.push(MxArray::zeros(&[hi - lo, dim], Some(f.dtype()?))?);
                lo = hi;
                continue;
            }
            let (y, r) = self.moe.forward(&f.slice(&[lo, 0], &[hi, dim])?)?;
            MxArray::eval_arrays_with_context(&[&y], "deepseek_v41 ffn chunk")?;
            routes.ids.extend(r.ids);
            routes.weights.extend(r.weights);
            routes.biased.extend(r.biased);
            outs.push(y);
            lo = hi;
        }
        self.moe.end_layer();
        let f = match outs.len() {
            1 => outs.swap_remove(0),
            _ => MxArray::concatenate_many(outs.iter().collect(), Some(0))?,
        };
        let tp = prof_start();
        let x = hc_post(&f, &x, &ffn_post, &ffn_comb)?;
        prof_end(Phase::Hc, tp, &[&x])?;
        Ok((x, ffn_pre, routes, scores))
    }
}
