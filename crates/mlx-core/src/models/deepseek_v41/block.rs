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
use crate::array::MxArray;

/// FFN rows per chunk: the routed scratch is `rows x k x (2 inter + dim)`.
pub const FFN_CHUNK_ROWS: i64 = 256;

pub enum Attn {
    Window(WindowAttention),
    Compressed(CompressedAttention),
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
        get: &mut impl FnMut(&str) -> Result<MxArray>,
    ) -> Result<Self> {
        let mut t = |s: &str| get(&format!("blk.{layer}.hc_{tag}_{s}.weight"));
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
        mut get: impl FnMut(&str) -> Result<MxArray>,
    ) -> Result<Self> {
        let attn = if cfg.compress_ratios[layer] == 0 {
            let w = super::attention::AttentionWeights::from_gguf(layer, &mut get)?;
            Attn::Window(WindowAttention::new(cfg, layer, w, fq)?)
        } else {
            Attn::Compressed(CompressedAttention::from_gguf(cfg, layer, fq, &mut get)?)
        };
        Ok(Self {
            layer,
            attn,
            moe,
            attn_norm: get(&format!("blk.{layer}.attn_norm.weight"))?,
            ffn_norm: get(&format!("blk.{layer}.ffn_norm.weight"))?,
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
    /// layer's KV source cache (compressed layers only).
    pub fn forward(
        &self,
        x: &MxArray,
        start: usize,
        pre_mix: &MxArray,
        win: &mut WindowCache,
        src: Option<&mut CompressedCache>,
        shared: &mut ForwardShared,
    ) -> Result<(MxArray, MxArray, Routes, Option<Vec<f32>>)> {
        let (attn_pre, attn_post, attn_comb) = self.mixes(x, &self.hc_attn)?;
        let a = rms_norm(&hc_pre(x, pre_mix)?, &self.attn_norm, self.eps)?;
        let (a, scores) = match (&self.attn, src) {
            (Attn::Window(w), _) => (w.forward(&a, start, win)?, None),
            (Attn::Compressed(c), Some(src)) => c.forward(&a, start, win, src, shared)?,
            (Attn::Compressed(_), None) => {
                return Err(napi::Error::from_reason(format!(
                    "layer {}: compressed attention without its source cache",
                    self.layer
                )));
            }
        };
        let x = hc_post(&a, x, &attn_post, &attn_comb)?;

        let (ffn_pre, ffn_post, ffn_comb) = self.mixes(&x, &self.hc_ffn)?;
        let f = rms_norm(&hc_pre(&x, &attn_pre)?, &self.ffn_norm, self.eps)?;
        let (s, dim) = (f.shape_at(0)?, f.shape_at(1)?);
        let mut outs = Vec::new();
        let mut routes = Routes {
            k: self.moe.k(),
            ids: Vec::new(),
            weights: Vec::new(),
            biased: Vec::new(),
        };
        let mut lo = 0;
        while lo < s {
            let hi = (lo + FFN_CHUNK_ROWS).min(s);
            let (y, r) = self.moe.forward(&f.slice(&[lo, 0], &[hi, dim])?)?;
            MxArray::eval_arrays_with_context(&[&y], "deepseek_v41 ffn chunk")?;
            routes.ids.extend(r.ids);
            routes.weights.extend(r.weights);
            routes.biased.extend(r.biased);
            outs.push(y);
            lo = hi;
        }
        let f = match outs.len() {
            1 => outs.swap_remove(0),
            _ => MxArray::concatenate_many(outs.iter().collect(), Some(0))?,
        };
        let x = hc_post(&f, &x, &ffn_post, &ffn_comb)?;
        Ok((x, ffn_pre, routes, scores))
    }
}
