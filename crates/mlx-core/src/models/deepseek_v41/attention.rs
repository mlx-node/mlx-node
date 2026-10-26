//! DeepSeek-V4.1 attention, window path (`model.py` `Attention` with
//! `compress_ratio == 0`): q-LoRA, one shared K = V row per token, partial
//! RoPE, sliding window with an attention sink, grouped low-rank output.

use napi::{Error, Result};

use std::sync::atomic::{AtomicUsize, Ordering};

use super::cache::{CompressedCache, ForwardShared, TopK, WindowCache};
use super::compressor::{Compressor, CompressorWeights};
use super::config::DeepseekV41Config;
use super::indexer::{CandidateRole, Indexer, IndexerWeights, chunk_rows};
use super::ops::{apply_partial_rope, fp4_roundtrip, fp8_roundtrip, inv_freqs, rope_tables};
use crate::array::{DType, MxArray};

/// Block size of the fp8 activation round trips (`model.py` `fp8_block_size`).
const FP8_BLOCK: i64 = 32;
/// Compressed-KV latent round trip: fp4 with e4m3 scales per 16.
const LATENT_FP4_BLOCK: i64 = 16;

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
    super::model::fp32_site("rms_norm", usize::MAX, &r)?;
    w.astype(DType::Float32)?
        .mul(&xf.mul(&r)?)?
        .astype(x.dtype()?)
}

/// Appends query position `p`'s window entries (`window` slots, `-1` when
/// empty) as indices into keys that start at position `first`.
fn push_window_row(idx: &mut Vec<i32>, p: usize, first: usize, window: usize) {
    let lo = (p + 1).saturating_sub(window).max(first);
    idx.extend((lo..=p).map(|kp| (kp - first) as i32));
    idx.extend(std::iter::repeat_n(-1, window - (p + 1 - lo)));
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
        // ratio-0 layers: base theta, no YaRN
        let inv = inv_freqs(cfg.rope_head_dim, cfg.rope_theta, None);
        Ok(Self::build(cfg, w, fq, inv))
    }

    fn build(cfg: &DeepseekV41Config, w: AttentionWeights, fq: FakeQuant, inv: Vec<f32>) -> Self {
        Self {
            n_heads: cfg.n_heads as i64,
            head_dim: cfg.head_dim as i64,
            rope_dim: cfg.rope_head_dim as i64,
            o_groups: cfg.o_groups as i64,
            o_lora: cfg.o_lora_rank as i64,
            window: cfg.window_size,
            eps: cfg.norm_eps,
            scale: (cfg.head_dim as f64).powf(-0.5) as f32,
            inv,
            w,
            fq,
        }
    }

    /// `qr = q_norm(wq_a x)` and the RoPE'd query heads `[s, h, head_dim]`.
    fn query(&self, x: &MxArray, cos: &MxArray, sin: &MxArray) -> Result<(MxArray, MxArray)> {
        let s = x.shape_at(0)?;
        let qr = rms_norm(&linear(x, &self.w.wq_a)?, &self.w.q_norm, self.eps)?;
        let q = linear(&qr, &self.w.wq_b)?.reshape(&[s, self.n_heads, self.head_dim])?;
        let q = apply_partial_rope(&q, self.rope_dim, cos, sin, false)?;
        Ok((qr, q))
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
        let (_, q) = self.query(x, &cos, &sin)?;
        let kv = self.quant_kv(&self.kv_rotated(x, &cos, &sin)?)?;
        let (keys, first) = cache.append(&kv, start)?;
        let mut idx = Vec::with_capacity(s * self.window);
        for p in start..start + s {
            push_window_row(&mut idx, p, first, self.window);
        }
        let mut peak = 0;
        let o = sparse_sink_attention(
            &q,
            &keys,
            &idx,
            self.window,
            &self.w.sink,
            self.scale,
            &mut peak,
        )?;
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

/// Byte budget of one sparse-attention row chunk (gathered keys + scores).
pub const SPARSE_CHUNK_BYTES: usize = 1 << 30;

/// Bytes one query row needs in `sparse_sink_attention`: gathered keys
/// `[kmax, d]`, three `[h, kmax]` score temporaries, q and o `[h, d]`, f32.
pub fn sparse_row_bytes(kmax: usize, h: usize, d: usize) -> usize {
    4 * (kmax * d + 3 * h * kmax + 2 * h * d)
}

/// `sparse_attn` (K = V): each query row attends to the key rows `idx`
/// `[s * kmax]` lists (`-1` = empty), f32 scores, max floored at -1e30, sink
/// in the denominator only, P cast to the activation dtype before `P @ V`.
/// Rows go in chunks of at most 256 within `SPARSE_CHUNK_BYTES`, each
/// evaluated before the next; `peak` records the largest chunk's bytes.
pub(crate) fn sparse_sink_attention(
    q: &MxArray,
    keys: &MxArray,
    idx: &[i32],
    kmax: usize,
    sink: &MxArray,
    scale: f32,
    peak: &mut usize,
) -> Result<MxArray> {
    let act = q.dtype()?;
    let (s, h, d) = (q.shape_at(0)? as usize, q.shape_at(1)?, q.shape_at(2)?);
    if idx.len() != s * kmax || kmax == 0 {
        return Err(Error::from_reason(
            "sparse attention: index rows do not match the queries",
        ));
    }
    let kf = keys.astype(DType::Float32)?;
    let row = sparse_row_bytes(kmax, h as usize, d as usize);
    let rows = chunk_rows(row, SPARSE_CHUNK_BYTES);
    let neg_inf = MxArray::from_float32(&[f32::NEG_INFINITY], &[1])?;
    let floor = MxArray::from_float32(&[-1e30], &[1])?;
    let zero = MxArray::from_int32(&[0], &[1])?;
    let sink = sink.astype(DType::Float32)?.reshape(&[1, h, 1])?;
    let mut outs = Vec::new();
    let mut lo = 0;
    while lo < s {
        let hi = (lo + rows).min(s);
        let c = (hi - lo) as i64;
        *peak = (*peak).max((hi - lo) * row);
        let part = &idx[lo * kmax..hi * kmax];
        let gid: Vec<i32> = part.iter().map(|&j| j.max(0)).collect();
        let ok: Vec<i32> = part.iter().map(|&j| i32::from(j >= 0)).collect();
        let g = kf
            .take(&MxArray::from_int32(&gid, &[c * kmax as i64])?, 0)?
            .reshape(&[c, kmax as i64, d])?;
        let valid = MxArray::from_int32(&ok, &[c, 1, kmax as i64])?.greater(&zero)?;
        let qc = q
            .slice(&[lo as i64, 0, 0], &[hi as i64, h, d])?
            .astype(DType::Float32)?;
        let sc = qc.matmul(&g.transpose(Some(&[0, 2, 1]))?)?;
        let sc = valid.where_(&sc, &neg_inf)?.mul_scalar(f64::from(scale))?;
        super::model::fp32_site("attn_scores", usize::MAX, &sc)?;
        let m = sc.max(Some(&[-1]), Some(true))?.maximum(&floor)?;
        let p = sc.sub(&m)?.exp()?;
        let den = p.sum(Some(&[-1]), Some(true))?.add(&sink.sub(&m)?.exp()?)?;
        let o = p
            .astype(act)?
            .astype(DType::Float32)?
            .matmul(&g)?
            .div(&den)?
            .astype(act)?;
        MxArray::eval_arrays_with_context(&[&o], "deepseek_v41 sparse attention chunk")?;
        outs.push(o);
        lo = hi;
    }
    MxArray::concatenate_many(outs.iter().collect(), Some(0))
}

/// Attention of a compressed layer (ratio 1 or 2): the window path plus the
/// shared compressed KV of its source layer, at the rows its index source
/// selected. KV sources also run the compressor; index sources the indexer.
pub struct CompressedAttention {
    core: WindowAttention,
    layer: usize,
    ratio: usize,
    rope_dim: i64,
    compressor: Option<Compressor>,
    indexer: Option<Indexer>,
    /// Largest scratch chunk so far (indexer scores, sparse attention).
    scratch_peak: AtomicUsize,
    /// Return the index scores from `forward` (diagnostics and tests only:
    /// `s * T` host floats per call).
    pub export_scores: bool,
}

impl CompressedAttention {
    pub fn from_gguf(
        cfg: &DeepseekV41Config,
        layer: usize,
        fq: FakeQuant,
        mut get: impl FnMut(&str) -> Result<MxArray>,
    ) -> Result<Self> {
        let ratio = cfg.compress_ratios.get(layer).copied().unwrap_or(0);
        if ratio == 0 {
            return Err(Error::from_reason(format!(
                "layer {layer}: ratio 0 takes the window path"
            )));
        }
        let kv_source = cfg.kv_src[layer] == Some(layer);
        let index_source = cfg.topk_src[layer] == Some(layer);
        let w = AttentionWeights::from_gguf(layer, &mut get)?;
        let compressor = if kv_source {
            let cw = CompressorWeights::from_gguf(layer, ratio, &mut get)?;
            Some(Compressor::new(ratio, cfg.norm_eps, cw)?)
        } else {
            None
        };
        let indexer = if index_source {
            let role = match &cfg.candidate {
                Some(c) if c.source_layer == layer => CandidateRole::Source {
                    topk_blocks: c.topk_blocks,
                    block_size: c.block_size,
                },
                Some(c) if c.source_layer < layer => CandidateRole::Consumer,
                _ => CandidateRole::None,
            };
            let iw = IndexerWeights::from_gguf(layer, kv_source, &mut get)?;
            Some(Indexer::new(
                cfg.index_n_heads,
                cfg.index_head_dim,
                cfg.rope_head_dim,
                cfg.index_topk,
                ratio,
                cfg.norm_eps,
                role,
                fq,
                iw,
            )?)
        } else {
            None
        };
        let inv = inv_freqs(
            cfg.rope_head_dim,
            cfg.compress_rope_theta,
            cfg.yarn.as_ref(),
        );
        Ok(Self {
            core: WindowAttention::build(cfg, w, fq, inv),
            layer,
            ratio: ratio as usize,
            rope_dim: cfg.rope_head_dim as i64,
            compressor,
            indexer,
            scratch_peak: AtomicUsize::new(0),
            export_scores: false,
        })
    }

    pub fn scratch_peak(&self) -> usize {
        self.scratch_peak.load(Ordering::Relaxed)
    }

    pub fn indexer(&self) -> Option<&Indexer> {
        self.indexer.as_ref()
    }

    /// RoPE tables of compressed rows `g0..g0 + g` (position `j * ratio`).
    fn group_rope(&self, g0: usize, g: usize) -> Result<(MxArray, MxArray)> {
        let pos: Vec<u64> = (g0..g0 + g).map(|j| (j * self.ratio) as u64).collect();
        rope_tables(&self.core.inv, &pos)
    }

    /// `x` `[s, dim]` at `start..start + s`. `src` is the KV source's cache
    /// (this layer's own when it is the source, appended here; else already
    /// advanced by the source in this forward). `shared` carries the index
    /// source's top-k and the candidate blocks down the stack. Returns the
    /// output and, at an index source with `export_scores`, the scores before
    /// the final top-k. Source positions are checked before any cache moves;
    /// a later failure can leave the window advanced (reset the session).
    pub fn forward(
        &self,
        x: &MxArray,
        start: usize,
        win: &mut WindowCache,
        src: &mut CompressedCache,
        shared: &mut ForwardShared,
    ) -> Result<(MxArray, Option<Vec<f32>>)> {
        let s = x.shape_at(0)? as usize;
        let end = start + s;
        let t = end / self.ratio;
        let bad = |m: String| Error::from_reason(format!("layer {}: {m}", self.layer));
        if src.ratio != self.ratio {
            return Err(bad(format!("source ratio {} != {}", src.ratio, self.ratio)));
        }
        let want = if self.compressor.is_some() {
            start
        } else {
            end
        };
        if src.len != want {
            return Err(bad(format!(
                "source cache at {}, call at {start}+{s}",
                src.len
            )));
        }
        let (cos, sin) = self.core.rope(start, s)?;
        let (qr, q) = self.core.query(x, &cos, &sin)?;
        let kv = self.core.quant_kv(&self.core.kv_rotated(x, &cos, &sin)?)?;
        let (keys, first) = win.append(&kv, start)?;

        // the source's new state is staged and committed only once this call
        // has produced its output
        let mut tail = src.tail.clone();
        let mut index_k = src.index_k.clone();
        let mut ckv = src.kv.clone();
        let mut latent = None;
        if let Some(comp) = &self.compressor {
            let g0 = src.rows();
            latent = comp.forward(x, start, &mut tail)?.map(|l| (g0, l));
            if let (Some(ix), Some((g0, l))) = (&self.indexer, &latent)
                && ix.owns_k()
            {
                let (c, sn) = self.group_rope(*g0, l.shape_at(0)? as usize)?;
                index_k = Some(CompressedCache::appended(
                    &index_k,
                    &ix.index_keys(l, &c, &sn)?,
                )?);
            }
        }

        let mut scores = None;
        if let Some(ix) = &self.indexer {
            let topk = if t == 0 {
                TopK {
                    rows: s,
                    k: 0,
                    ids: Vec::new(),
                }
            } else {
                let ik = index_k
                    .as_ref()
                    .ok_or_else(|| bad("no index keys in the source cache".into()))?;
                let mut peak = 0;
                let (sc, tk) = ix.select(
                    x,
                    &qr,
                    start,
                    &cos,
                    &sin,
                    ik,
                    &mut shared.candidates,
                    &mut peak,
                    self.export_scores,
                )?;
                self.scratch_peak.fetch_max(peak, Ordering::Relaxed);
                scores = sc;
                tk
            };
            shared.topk = Some(topk);
        }

        if let Some((g0, l)) = latent {
            let (c, sn) = self.group_rope(g0, l.shape_at(0)? as usize)?;
            let l = apply_partial_rope(&l, self.rope_dim, &c, &sn, false)?;
            let l = if self.core.fq.latent {
                fp4_roundtrip(&l, LATENT_FP4_BLOCK, true)?
            } else {
                l
            };
            ckv = Some(CompressedCache::appended(&ckv, &l)?);
        }

        let topk = shared
            .topk
            .as_ref()
            .filter(|tk| tk.rows == s)
            .ok_or_else(|| bad("no top-k from an index source for this call".into()))?;
        let kw = keys.shape_at(0)? as usize;
        let win_n = self.core.window;
        let kmax = win_n + topk.k;
        let mut idx = Vec::with_capacity(s * kmax);
        for i in 0..s {
            push_window_row(&mut idx, start + i, first, win_n);
            idx.extend(
                topk.ids[i * topk.k..(i + 1) * topk.k]
                    .iter()
                    .map(|&j| if j >= 0 { j + kw as i32 } else { -1 }),
            );
        }
        let all = match (&ckv, t) {
            (Some(ckv), t) if t > 0 => {
                let hd = ckv.shape_at(1)?;
                MxArray::concatenate(&keys, &ckv.slice(&[0, 0], &[t as i64, hd])?, 0)?
            }
            _ => keys,
        };
        let mut peak = 0;
        let o = sparse_sink_attention(
            &q,
            &all,
            &idx,
            kmax,
            &self.core.w.sink,
            self.core.scale,
            &mut peak,
        )?;
        self.scratch_peak.fetch_max(peak, Ordering::Relaxed);
        let o = apply_partial_rope(&o, self.rope_dim, &cos, &sin, true)?;
        let out = self.core.output(&o)?;
        if self.compressor.is_some() {
            src.tail = tail;
            src.index_k = index_k;
            src.kv = ckv;
            src.len = end;
        }
        Ok((out, scores))
    }
}

#[cfg(test)]
mod tests;
