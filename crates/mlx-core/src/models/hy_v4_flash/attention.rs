//! Hy4 MLA attention (llama.cpp PR #32517 `build_attention`), absorbed
//! form, f32 math over f16 cached latent rows:
//!
//! ```text
//! x    = rms(h)·attn_norm                       qr = rms(q_a·x)·q_a_norm
//! q    = q_b·qr  [n, H, nope | rope]            ckv = kv_a·x  [n, kv_lora | rope]
//! K    = [rms(ckv[:kv_lora])·kv_a_norm | rope(ckv[kv_lora:])]       one row per token
//! Q    = [k_b_h·q_nope_h | rope(q_pe)]          o_lat = attn(Q, K, V = K[:kv_lora])
//! o    = (v_b_h·o_lat_h) · σ(attn_gate·x)       out = rms(attn_output·o)·post_attention_norm
//! ```
//!
//! RoPE rotates consecutive pairs (`LLAMA_ROPE_TYPE_NORM`) with the layer's
//! base; scale `1/√key_length_mla`; the sink is a softmax denominator term.
//! Sliding layers attend the last `window` rows incl. self; indexed layers
//! attend the cells the k-pool indexer picks (`indexer.rs`).

use napi::{Error, Result};

use super::cache::{LatentBuf, LayerCache, WindowRing};
use super::config::{Attn, HyV4FlashConfig};
use super::indexer::{Indexing, PagedIndexing};
use super::paged::{SlotMap, planes, write_latent};
use super::sparse::{KeyRows, gather_sink_attention, gather_sink_attention_rows};
use super::stats::{Phase, prof_end, prof_start};
use super::weights::AttnWeights;
use crate::array::{DType, MxArray};
use crate::models::deepseek_v41::attention::{linear, rms_norm};
use crate::models::deepseek_v41::ops::{apply_partial_rope, inv_freqs, rope_tables};
use crate::models::deepseek_v41::store::{Dense, Weight};
use crate::nn::Activations;
use crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter;

fn bad(msg: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("Hy4 attention: {msg}"))
}

/// The sizes one layer's attention runs with.
#[derive(Debug, Clone, PartialEq)]
pub struct Geometry {
    pub attn: Attn,
    pub n_heads: usize,
    pub kv_lora: usize,
    pub nope: usize,
    pub rope_dim: usize,
    pub head_v: usize,
    pub eps: f32,
    pub scale: f32,
    pub rope_theta: f32,
    pub window: usize,
}

impl Geometry {
    /// Layer `layer` of `cfg` (trunk or MTP).
    pub fn of(cfg: &HyV4FlashConfig, layer: usize) -> Result<Self> {
        let l = cfg
            .layers
            .get(layer)
            .ok_or_else(|| bad(format!("layer {layer} of {}", cfg.layers.len())))?;
        Ok(Self {
            attn: l.attn,
            n_heads: l.n_heads,
            kv_lora: cfg.mla.kv_lora_rank,
            nope: cfg.mla.qk_nope_dim,
            rope_dim: cfg.mla.rope_dim,
            head_v: cfg.mla.head_dim_v,
            eps: cfg.norm_eps,
            scale: cfg.mla.scale,
            rope_theta: l.rope_theta,
            window: cfg.window,
        })
    }

    /// One cached row: `kv_lora + rope_dim`.
    pub fn row(&self) -> usize {
        self.kv_lora + self.rope_dim
    }

    pub fn head_qk(&self) -> usize {
        self.nope + self.rope_dim
    }
}

/// Key ids (relative to `first`) of the window of each query `start..start
/// + n`: `window` slots per row, `-1` when empty. DS `push_window_row`
/// (`deepseek_v41/attention.rs:103-107`).
pub fn window_rows(start: usize, n: usize, first: usize, window: usize) -> Vec<i32> {
    let mut idx = Vec::with_capacity(n * window);
    for p in start..start + n {
        let lo = (p + 1).saturating_sub(window).max(first);
        idx.extend((lo..=p).map(|kp| (kp - first) as i32));
        idx.extend(std::iter::repeat_n(-1, window - (p + 1 - lo)));
    }
    idx
}

/// `x [g, s, k] · W_iᵀ` for the `g` row groups `W_i` of `w [g·r, k]`
/// -> `[g, s, r]`.
fn grouped_mm_t(w: &Dense, x: &MxArray, g: i64) -> Result<MxArray> {
    if let Some(y) = w.grouped_qmm_t(x, g)? {
        return Ok(y);
    }
    let wf = w.f32()?;
    let wf = wf.reshape(&[g, wf.shape_at(0)? / g, wf.shape_at(1)?])?;
    x.astype(DType::Float32)?
        .matmul(&wf.transpose(Some(&[0, 2, 1]))?)
}

/// One call's projections.
pub struct Projected {
    /// `rms(h)·attn_norm` `[n, dim]`: the gate's and the indexer's input.
    pub x: MxArray,
    /// `rms(q_a·x)·q_a_norm` `[n, q_lora]`.
    pub qr: MxArray,
    /// Absorbed queries `[n, H, kv_lora + rope]` f32.
    pub q: MxArray,
    /// Latent rows `[n, kv_lora + rope]` f32, before the cache's f16 cast.
    pub kv: MxArray,
}

/// The paged pool for one call of the adapter's active request.
pub struct PagedPool<'a> {
    pub adapter: &'a mut PagedKVCacheAdapter,
    /// Tokens recorded before this pass; no write reaches below them.
    pub floor: u32,
    /// The active request's pages, mapped after the call's rows were
    /// recorded.
    pub slots: &'a SlotMap,
}

/// The attention of one layer; the weights stay in `Resident`.
pub struct Mla {
    g: Geometry,
    inv: Vec<f32>,
}

fn want(d: &Dense, axis: u32, n: usize, what: &str) -> Result<()> {
    let got = d.shape_at(axis)?;
    if got != n as i64 {
        return Err(bad(format!("{what} axis {axis} is {got}, need {n}")));
    }
    Ok(())
}

impl Mla {
    /// Checks `w` against `g` (row counts and widths of every projection).
    pub fn new(g: Geometry, w: &AttnWeights) -> Result<Self> {
        if g.n_heads == 0
            || g.kv_lora == 0
            || g.nope == 0
            || g.head_v == 0
            || g.rope_dim == 0
            || !g.rope_dim.is_multiple_of(2)
            || g.window == 0
        {
            return Err(bad(format!("degenerate geometry {g:?}")));
        }
        let h = g.n_heads;
        want(&w.q_b, 0, h * g.head_qk(), "attn_q_b")?;
        want(&w.kv_a, 0, g.row(), "attn_kv_a_mqa")?;
        want(&w.k_b, 0, h * g.kv_lora, "attn_k_b")?;
        want(&w.k_b, 1, g.nope, "attn_k_b")?;
        want(&w.v_b, 0, h * g.head_v, "attn_v_b")?;
        want(&w.v_b, 1, g.kv_lora, "attn_v_b")?;
        want(&w.gate, 0, h * g.head_v, "attn_gate")?;
        want(&w.output, 1, h * g.head_v, "attn_output")?;
        let sinks = w.sinks.shape()?.as_ref().to_vec();
        if sinks.as_slice() != [h as i64] {
            return Err(bad(format!("attn_sinks {sinks:?} for {h} heads")));
        }
        let inv = inv_freqs(g.rope_dim, g.rope_theta, None);
        Ok(Self { g, inv })
    }

    pub fn geometry(&self) -> &Geometry {
        &self.g
    }

    /// Projections of `h [n, dim]` at positions `start..start + n`.
    pub fn project(&self, w: &AttnWeights, h: &MxArray, start: usize) -> Result<Projected> {
        let g = &self.g;
        let n = h.shape_at(0)?;
        let (heads, nope, rd) = (g.n_heads as i64, g.nope as i64, g.rope_dim as i64);
        let qk = g.head_qk() as i64;
        let kv_lora = g.kv_lora as i64;
        let pos: Vec<u64> = (start..start + n as usize).map(|p| p as u64).collect();
        let (cos, sin) = rope_tables(&self.inv, &pos)?;

        let x = rms_norm(&h.astype(DType::Float32)?, &w.norm, g.eps)?;
        let qr = rms_norm(&linear(&x, &w.q_a)?, &w.q_a_norm, g.eps)?;
        let q = linear(&qr, &w.q_b)?.reshape(&[n, heads, qk])?;
        let q = apply_partial_rope(&q, rd, &cos, &sin, false)?;
        let q_nope = q
            .slice(&[0, 0, 0], &[n, heads, nope])?
            .transpose(Some(&[1, 0, 2]))?;
        let q_pe = q.slice(&[0, 0, nope], &[n, heads, qk])?;
        let q_abs = grouped_mm_t(&w.k_b, &q_nope, heads)?.transpose(Some(&[1, 0, 2]))?;
        let q = MxArray::concatenate(&q_abs, &q_pe, -1)?;

        let ckv = linear(&x, &w.kv_a)?;
        let kv = rms_norm(&ckv.slice(&[0, 0], &[n, kv_lora])?, &w.kv_a_norm, g.eps)?;
        let k_pe = apply_partial_rope(
            &ckv.slice(&[0, kv_lora], &[n, kv_lora + rd])?,
            rd,
            &cos,
            &sin,
            false,
        )?;
        let kv = MxArray::concatenate(&kv, &k_pe, -1)?;
        Ok(Projected { x, qr, q, kv })
    }

    /// `o_lat [n, H, kv_lora]` -> the sublayer output `[n, dim]` f32.
    pub fn finish(&self, w: &AttnWeights, x: &MxArray, o_lat: &MxArray) -> Result<MxArray> {
        let g = &self.g;
        let n = o_lat.shape_at(0)?;
        let heads = g.n_heads as i64;
        let width = heads * g.head_v as i64;
        let o = grouped_mm_t(&w.v_b, &o_lat.transpose(Some(&[1, 0, 2]))?, heads)?
            .transpose(Some(&[1, 0, 2]))?
            .reshape(&[n, width])?;
        let gate = Activations::sigmoid(&linear(x, &w.gate)?)?;
        let y = linear(&o.mul(&gate)?, &w.output)?;
        rms_norm(&y, &w.post_norm, g.eps)
    }

    /// `finish` under the attention-projection clock.
    fn finish_timed(&self, w: &AttnWeights, x: &MxArray, o_lat: &MxArray) -> Result<MxArray> {
        let tp = prof_start();
        let y = self.finish(w, x, o_lat)?;
        prof_end(Phase::AttnProj, tp, &[&y])?;
        Ok(y)
    }

    /// The sublayer output for `h [n, dim]` (the HC-reduced stream, before
    /// `attn_norm`) at positions `start..start + n`; appends this call's
    /// rows to `cache` (and to the indexer state of an indexed layer, which
    /// needs `index`). On error both hold what they held before.
    pub fn forward(
        &self,
        w: &AttnWeights,
        h: &MxArray,
        start: usize,
        cache: &mut LayerCache,
        index: Option<Indexing<'_>>,
    ) -> Result<MxArray> {
        match (self.g.attn, cache, index) {
            (Attn::Sliding, LayerCache::Window(ring), None) => {
                self.forward_sliding(w, h, start, ring)
            }
            (Attn::Indexed, LayerCache::Latent(buf), Some(ix)) => {
                self.forward_indexed(w, h, start, buf, ix)
            }
            (Attn::Indexed, LayerCache::Latent(_), None) => {
                Err(bad("indexed layer called without its indexer"))
            }
            (Attn::Sliding, LayerCache::Window(_), Some(_)) => {
                Err(bad("sliding layer given an indexer"))
            }
            (attn, _, _) => Err(bad(format!("{attn:?} layer given the other cache kind"))),
        }
    }

    pub fn forward_sliding(
        &self,
        w: &AttnWeights,
        h: &MxArray,
        start: usize,
        ring: &mut WindowRing,
    ) -> Result<MxArray> {
        let n = h.shape_at(0)? as usize;
        let tp = prof_start();
        let p = self.project(w, h, start)?;
        let before = ring.clone();
        let out = ring.append(&p.kv, start).and_then(|(keys, first)| {
            prof_end(Phase::AttnProj, tp, &[&p.q, &p.x])?;
            let tp = prof_start();
            let window = self.g.window;
            let idx = MxArray::from_int32(
                &window_rows(start, n, first, window),
                &[n as i64, window as i64],
            )?;
            let o = gather_sink_attention(
                &p.q,
                &keys,
                &idx,
                self.g.kv_lora as i64,
                &w.sinks,
                self.g.scale,
            )?;
            prof_end(Phase::Sparse, tp, &[&o])?;
            self.finish_timed(w, &p.x, &o)
        });
        if out.is_err() {
            *ring = before;
        }
        out
    }

    pub fn forward_indexed(
        &self,
        w: &AttnWeights,
        h: &MxArray,
        start: usize,
        buf: &mut LatentBuf,
        ix: Indexing<'_>,
    ) -> Result<MxArray> {
        let tp = prof_start();
        let p = self.project(w, h, start)?;
        let mark = ix.state.mark();
        let out = buf.write(&p.kv, start).and_then(|()| {
            prof_end(Phase::AttnProj, tp, &[&p.q, &p.x, &p.qr])?;
            let tp = prof_start();
            let idx = ix
                .indexer
                .select(ix.weights, &p.x, &p.qr, start, ix.state)?;
            prof_end(Phase::Indexer, tp, &[&idx])?;
            let tp = prof_start();
            let o = gather_sink_attention(
                &p.q,
                buf.rows()?,
                &idx,
                self.g.kv_lora as i64,
                &w.sinks,
                self.g.scale,
            )?;
            prof_end(Phase::Sparse, tp, &[&o])?;
            self.finish_timed(w, &p.x, &o)
        });
        if out.is_err() {
            if buf.len() > start {
                buf.truncate(start)?;
            }
            ix.state.restore(mark)?;
        }
        out
    }

    /// `forward_indexed` of a paged sequence: the call's latent rows go to
    /// the pool layer of `ix.state` as two half rows, its pooled keys to that
    /// layer's index plane, and the attention reads both planes by slot id.
    /// On error the indexer state holds what it held before; the pool rows
    /// of the call stay dead until written again.
    pub fn forward_indexed_paged(
        &self,
        w: &AttnWeights,
        h: &MxArray,
        start: usize,
        pool: PagedPool<'_>,
        ix: PagedIndexing<'_>,
    ) -> Result<MxArray> {
        if self.g.attn != Attn::Indexed {
            return Err(bad("sliding layer given the paged pool"));
        }
        let layer = ix.state.layer();
        let base = u32::try_from(start).map_err(|_| bad("position exceeds u32"))?;
        let tp = prof_start();
        let p = self.project(w, h, start)?;
        let mark = ix.state.mark();
        let out = write_latent(pool.adapter, layer, &p.kv, base, pool.floor).and_then(|()| {
            prof_end(Phase::AttnProj, tp, &[&p.q, &p.x, &p.qr])?;
            let tp = prof_start();
            let idx =
                ix.indexer
                    .select_paged(ix.weights, &p.x, &p.qr, start, ix.state, pool.adapter)?;
            prof_end(Phase::Indexer, tp, &[&idx])?;
            let tp = prof_start();
            let slots = pool.slots.slots(pool.adapter, &idx)?;
            let (k, v) = planes(pool.adapter, layer)?;
            let o = gather_sink_attention_rows(
                &p.q,
                KeyRows::Split(&k, &v),
                &slots,
                self.g.kv_lora as i64,
                &w.sinks,
                self.g.scale,
            )?;
            prof_end(Phase::Sparse, tp, &[&o])?;
            self.finish_timed(w, &p.x, &o)
        });
        if out.is_err() {
            ix.state.restore(mark)?;
        }
        out
    }
}

#[cfg(test)]
mod tests;
