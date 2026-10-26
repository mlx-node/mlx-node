//! DeepSeek-V4.1 attention, window path (`model.py` `Attention` with
//! `compress_ratio == 0`): q-LoRA, one shared K = V row per token, partial
//! RoPE, sliding window with an attention sink, grouped low-rank output.

use napi::{Error, Result};

use super::cache::WindowCache;
use super::config::DeepseekV41Config;
use super::ops::{apply_partial_rope, fp8_roundtrip, inv_freqs, rope_tables};
use crate::array::{DType, MxArray};

/// Block size of the fp8 activation round trips (`model.py` `fp8_block_size`).
const FP8_BLOCK: i64 = 32;

/// The 1017 activation fake-quant sites; each can be switched off (tests).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FakeQuant {
    pub kv: bool,
    pub latent: bool,
    pub idx_q: bool,
    pub idx_k: bool,
    pub o: bool,
}

impl FakeQuant {
    pub const ON: Self = Self {
        kv: true,
        latent: true,
        idx_q: true,
        idx_k: true,
        o: true,
    };
    pub const OFF: Self = Self {
        kv: false,
        latent: false,
        idx_q: false,
        idx_k: false,
        o: false,
    };
}

/// One layer's attention weights, f32, torch layout (`[out, in]`).
pub struct AttentionWeights {
    pub wq_a: MxArray,
    pub q_norm: MxArray,
    pub wq_b: MxArray,
    pub wkv: MxArray,
    pub kv_norm: MxArray,
    /// `[o_groups * o_lora_rank, n_heads * head_dim / o_groups]`
    pub wo_a: MxArray,
    pub wo_b: MxArray,
    /// `[n_heads]`
    pub sink: MxArray,
}

impl AttentionWeights {
    /// Reads `blk.<layer>.attn_*` through `get` (GGUF name -> dense f32).
    pub fn from_gguf(layer: usize, mut get: impl FnMut(&str) -> Result<MxArray>) -> Result<Self> {
        let mut t = |s: &str| get(&format!("blk.{layer}.{s}.weight"));
        Ok(Self {
            wq_a: t("attn_q_a")?,
            q_norm: t("attn_q_a_norm")?,
            wq_b: t("attn_q_b")?,
            wkv: t("attn_kv")?,
            kv_norm: t("attn_kv_a_norm")?,
            wo_a: t("attn_output_a")?,
            wo_b: t("attn_output_b")?,
            sink: t("attn_sinks")?,
        })
    }
}

/// `F.linear(x.float(), W).to(x.dtype)` (oracle A8).
pub(crate) fn linear(x: &MxArray, w: &MxArray) -> Result<MxArray> {
    x.astype(DType::Float32)?
        .matmul(&w.astype(DType::Float32)?.transpose(Some(&[1, 0]))?)?
        .astype(x.dtype()?)
}

/// `RMSNorm.forward`: f32 statistics, `weight * (x * rsqrt(var + eps))`.
pub(crate) fn rms_norm(x: &MxArray, w: &MxArray, eps: f32) -> Result<MxArray> {
    let xf = x.astype(DType::Float32)?;
    let r = xf
        .square()?
        .mean(Some(&[-1]), Some(true))?
        .add_scalar(f64::from(eps))?
        .sqrt()?
        .reciprocal()?;
    w.astype(DType::Float32)?
        .mul(&xf.mul(&r)?)?
        .astype(x.dtype()?)
}

/// `sparse_attn` over dense keys (K = V) with a validity mask: f32 scores,
/// running max floored at -1e30, the sink in the denominator only, P cast
/// to the activation dtype before `P @ V`. q `[s, h, d]`, keys `[k, d]`,
/// valid `[s, k]` bool, sink `[h]`; output `[s, h, d]` in q's dtype.
pub(crate) fn sink_attention(
    q: &MxArray,
    keys: &MxArray,
    valid: &MxArray,
    sink: &MxArray,
    scale: f32,
) -> Result<MxArray> {
    let act = q.dtype()?;
    let h = q.shape_at(1)?;
    let qf = q.astype(DType::Float32)?.transpose(Some(&[1, 0, 2]))?;
    let kf = keys.astype(DType::Float32)?;
    let s = qf.matmul(&kf.transpose(Some(&[1, 0]))?)?;
    let neg_inf = MxArray::from_float32(&[f32::NEG_INFINITY], &[1])?;
    let s = valid.where_(&s, &neg_inf)?.mul_scalar(f64::from(scale))?;
    let m = s
        .max(Some(&[-1]), Some(true))?
        .maximum(&MxArray::from_float32(&[-1e30], &[1])?)?;
    let p = s.sub(&m)?.exp()?;
    let den = p.sum(Some(&[-1]), Some(true))?.add(
        &sink
            .astype(DType::Float32)?
            .reshape(&[h, 1, 1])?
            .sub(&m)?
            .exp()?,
    )?;
    p.astype(act)?
        .astype(DType::Float32)?
        .matmul(&kf)?
        .div(&den)?
        .transpose(Some(&[1, 0, 2]))?
        .astype(act)
}

/// Window mask `[s, k]`: query `start + i` sees key `first + j` iff it is
/// at most `window - 1` positions back and not in the future.
pub(crate) fn window_mask(
    start: usize,
    s: usize,
    first: usize,
    k: usize,
    window: usize,
) -> Result<MxArray> {
    let mut v = Vec::with_capacity(s * k);
    for i in 0..s {
        let qp = start + i;
        for j in 0..k {
            let kp = first + j;
            v.push(i32::from(kp <= qp && kp + window > qp));
        }
    }
    MxArray::from_int32(&v, &[s as i64, k as i64])?.greater(&MxArray::from_int32(&[0], &[1])?)
}

/// Attention of one ratio-0 layer.
pub struct WindowAttention {
    n_heads: i64,
    head_dim: i64,
    rope_dim: i64,
    o_groups: i64,
    o_lora: i64,
    window: usize,
    eps: f32,
    scale: f32,
    inv: Vec<f32>,
    w: AttentionWeights,
    fq: FakeQuant,
}

impl WindowAttention {
    pub fn new(
        cfg: &DeepseekV41Config,
        layer: usize,
        w: AttentionWeights,
        fq: FakeQuant,
    ) -> Result<Self> {
        let ratio = cfg.compress_ratios.get(layer).copied();
        if ratio != Some(0) {
            return Err(Error::from_reason(format!(
                "layer {layer}: compress ratio {ratio:?}, the window path takes ratio 0 only"
            )));
        }
        Ok(Self {
            n_heads: cfg.n_heads as i64,
            head_dim: cfg.head_dim as i64,
            rope_dim: cfg.rope_head_dim as i64,
            o_groups: cfg.o_groups as i64,
            o_lora: cfg.o_lora_rank as i64,
            window: cfg.window_size,
            eps: cfg.norm_eps,
            scale: (cfg.head_dim as f64).powf(-0.5) as f32,
            // ratio-0 layers: base theta, no YaRN
            inv: inv_freqs(cfg.rope_head_dim, cfg.rope_theta, None),
            w,
            fq,
        })
    }

    fn rope(&self, start: usize, s: usize) -> Result<(MxArray, MxArray)> {
        let pos: Vec<u64> = (start..start + s).map(|p| p as u64).collect();
        rope_tables(&self.inv, &pos)
    }

    /// KV fp8 round trip over the whole post-RoPE row (when on).
    pub(crate) fn quant_kv(&self, kv: &MxArray) -> Result<MxArray> {
        if self.fq.kv {
            fp8_roundtrip(kv, FP8_BLOCK)
        } else {
            Ok(kv.clone())
        }
    }

    /// `o` `[s, groups, heads * head_dim / groups]` fp8 round trip (when on).
    pub(crate) fn quant_o(&self, o: &MxArray) -> Result<MxArray> {
        if self.fq.o {
            fp8_roundtrip(o, FP8_BLOCK)
        } else {
            Ok(o.clone())
        }
    }

    /// `kv_norm(wkv x)` with RoPE on the tail, before the fp8 round trip.
    pub(crate) fn kv_rotated(&self, x: &MxArray, cos: &MxArray, sin: &MxArray) -> Result<MxArray> {
        let kv = rms_norm(&linear(x, &self.w.wkv)?, &self.w.kv_norm, self.eps)?;
        apply_partial_rope(&kv, self.rope_dim, cos, sin, false)
    }

    /// `x` `[s, dim]` at positions `start..start + s` -> `[s, dim]`; appends
    /// this chunk's KV to `cache` (any chunk size, any `start`).
    pub fn forward(&self, x: &MxArray, start: usize, cache: &mut WindowCache) -> Result<MxArray> {
        let s = x.shape_at(0)? as usize;
        let (cos, sin) = self.rope(start, s)?;
        let qr = rms_norm(&linear(x, &self.w.wq_a)?, &self.w.q_norm, self.eps)?;
        let q = linear(&qr, &self.w.wq_b)?.reshape(&[s as i64, self.n_heads, self.head_dim])?;
        let q = apply_partial_rope(&q, self.rope_dim, &cos, &sin, false)?;
        let kv = self.quant_kv(&self.kv_rotated(x, &cos, &sin)?)?;
        let (keys, first) = cache.append(&kv, start)?;
        let k = keys.shape_at(0)? as usize;
        let valid = window_mask(start, s, first, k, self.window)?;
        let o = sink_attention(&q, &keys, &valid, &self.w.sink, self.scale)?;
        let o = apply_partial_rope(&o, self.rope_dim, &cos, &sin, true)?;
        self.output(&o)
    }

    /// Grouped `wo_a` (block-diagonal, held in the activation dtype as the
    /// reference does) after the `o` round trip, then `wo_b`.
    pub(crate) fn output(&self, o: &MxArray) -> Result<MxArray> {
        let act = o.dtype()?;
        let s = o.shape_at(0)?;
        let g = self.o_groups;
        let width = self.n_heads * self.head_dim / g;
        let o = self.quant_o(&o.reshape(&[s, g, width])?)?;
        let wa = self
            .w
            .wo_a
            .astype(act)?
            .astype(DType::Float32)?
            .reshape(&[g, self.o_lora, width])?
            .transpose(Some(&[0, 2, 1]))?;
        let y = o
            .astype(DType::Float32)?
            .transpose(Some(&[1, 0, 2]))?
            .matmul(&wa)?
            .transpose(Some(&[1, 0, 2]))?
            .reshape(&[s, g * self.o_lora])?
            .astype(act)?;
        linear(&y, &self.w.wo_b)
    }
}

#[cfg(test)]
mod tests;
