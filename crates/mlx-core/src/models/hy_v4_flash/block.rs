//! One trunk layer (llama.cpp PR #32517 graph loop), f32 streams `[n, hc, d]`:
//!
//! ```text
//! X = X + engram(X)                               Engram layers only, before the attention mix
//! (pre, post) = hc_mix(X, hc_attn)   X = X + post · attn(Σ pre_c X_c)
//! (pre, post) = hc_mix(X, hc_ffn)    X = X + post · ffn(Σ pre_c X_c)
//! ```
//!
//! Each sublayer reduces the streams with the `pre` of its own mix.
//!
//! A step runs the rows of one or more owners (a `Wave`) at once: the
//! Engram block, the HC streams, the attention projections and the FFN
//! over all rows, the attention core per owner on its own state
//! (`Owners::core`). Ops group as `batch.rs` lists them.

use napi::{Error, Result};

use super::attention::{Geometry, Mla, Projected};
use super::batch::{Core, Group, Op, Owners, Wave, group};
use super::cache::LayerCache;
use super::config::{Attn, HcConfig, HyV4FlashConfig};
use super::engram;
use super::indexer::{Indexer, IndexerGeometry, IndexerState, Indexing};
use super::moe::{ExpertSource, FfnConfig, LayerFfn, Routes, RowKind};
use super::ops::{HcGates, hc_mix, hc_post_identity, hc_reduce};
use super::stats::{Phase, prof_end, prof_start};
use super::weights::{HcWeights, LayerWeights, Residual};
use crate::array::MxArray;

fn bad(msg: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("Hy4 block: {msg}"))
}

/// Fewest prefill rows that trigger next-layer expert read-ahead.
pub const READ_AHEAD_MIN_ROWS: usize = 256;

/// One layer's per-sequence state.
pub struct LayerState {
    pub cache: LayerCache,
    /// Indexed layers only.
    pub index: Option<IndexerState>,
}

impl LayerState {
    /// Positions the attention cache holds.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    pub fn nbytes(&self) -> usize {
        self.cache.nbytes() + self.index.as_ref().map_or(0, IndexerState::nbytes)
    }
}

/// Engram inputs of a step's rows: table rows `[n, cols, head_dim]` and
/// which rows get the delta (`engram::keeps`).
#[derive(Clone, Copy)]
pub struct EngramIn<'a> {
    pub rows: &'a [f32],
    pub keep: &'a [bool],
}

/// One trunk layer; the weights stay in the model's weight set.
pub struct Block {
    layer: usize,
    mla: Mla,
    indexer: Option<Indexer>,
    ffn: LayerFfn,
    engram: Option<usize>,
    hc: HcConfig,
    eps: f32,
}

impl Block {
    /// Checks `w` against trunk layer `layer` of `cfg`; `experts` is
    /// required for a MoE layer and refused for the dense one.
    pub fn new(
        cfg: &HyV4FlashConfig,
        layer: usize,
        w: &LayerWeights,
        experts: Option<ExpertSource>,
    ) -> Result<Self> {
        let l = cfg.trunk().get(layer).ok_or_else(|| {
            bad(format!(
                "layer {layer} is not a trunk layer ({})",
                cfg.n_layers
            ))
        })?;
        if !matches!(w.residual, Residual::Hc { .. }) {
            return Err(bad(format!(
                "layer {layer}: MTP weights where a trunk layer is expected"
            )));
        }
        let indexer = match (l.attn, &w.indexer) {
            (Attn::Indexed, Some(iw)) => Some(Indexer::new(IndexerGeometry::of(cfg, layer)?, iw)?),
            (Attn::Sliding, None) => None,
            (attn, iw) => {
                return Err(bad(format!(
                    "layer {layer}: {attn:?} attention with indexer weights = {}",
                    iw.is_some()
                )));
            }
        };
        if l.engram.is_some() != w.engram.is_some() {
            return Err(bad(format!(
                "layer {layer}: Engram layer = {}, Engram weights = {}",
                l.engram.is_some(),
                w.engram.is_some()
            )));
        }
        Ok(Self {
            layer,
            mla: Mla::new(Geometry::of(cfg, layer)?, &w.attn)?,
            indexer,
            ffn: LayerFfn::new(&FfnConfig::of(cfg, layer)?, layer, &w.ffn, experts)?,
            engram: l.engram,
            hc: cfg.hc.clone(),
            eps: cfg.norm_eps,
        })
    }

    pub fn layer(&self) -> usize {
        self.layer
    }

    /// Index into the Engram layer list.
    pub fn engram(&self) -> Option<usize> {
        self.engram
    }

    pub fn is_moe(&self) -> bool {
        self.ffn.is_moe()
    }

    pub fn attn(&self) -> Attn {
        self.mla.geometry().attn
    }

    pub fn indexer(&self) -> Option<&Indexer> {
        self.indexer.as_ref()
    }

    pub fn mla(&self) -> &Mla {
        &self.mla
    }

    /// Empty state for a context of `cap` positions.
    pub fn new_state(&self, cap: usize) -> LayerState {
        let g = self.mla.geometry();
        LayerState {
            cache: LayerCache::new(g.attn, g.window, cap, g.row() as i64),
            index: self
                .indexer
                .as_ref()
                .map(|ix| IndexerState::new(cap, ix.geometry())),
        }
    }

    pub fn prefetch_from(&self, w: &LayerWeights, h: &MxArray) -> Result<usize> {
        self.ffn.prefetch_from(w, h)
    }

    pub fn end_layer(&self) {
        self.ffn.end_layer();
    }

    /// The streams `x [n, hc, d]` (f32) after this layer for the rows of
    /// `wave` (`n` of them), whose owners' states take their rows through
    /// `owners`; `engram` is required on an Engram layer and refused
    /// elsewhere. On error the caller puts every owner's state back.
    pub fn forward(
        &self,
        w: &LayerWeights,
        x: &MxArray,
        wave: &Wave,
        owners: &mut dyn Owners,
        engram: Option<EngramIn<'_>>,
        next: Option<(&Block, &LayerWeights)>,
    ) -> Result<(MxArray, Option<Routes>)> {
        let Residual::Hc {
            attn: hc_attn,
            ffn: hc_ffn,
        } = &w.residual
        else {
            return Err(bad(format!("layer {}: no HC weights", self.layer)));
        };
        let n = x.shape_at(0)?;
        if wave.rows() as i64 != n || owners.count() != wave.owners() {
            return Err(bad(format!(
                "layer {}: a wave of {} rows / {} owners for {n} rows / {} owners",
                self.layer,
                wave.rows(),
                wave.owners(),
                owners.count()
            )));
        }
        let kind = wave.kind();
        let x = match (self.engram, &w.engram, engram) {
            (Some(_), Some(ew), Some(e)) => {
                let tp = prof_start();
                let y = engram::forward(x, e.rows, e.keep, ew)?;
                prof_end(Phase::Engram, tp, &[&y])?;
                y
            }
            (None, None, None) => x.clone(),
            (e, _, r) => {
                return Err(bad(format!(
                    "layer {}: Engram layer = {}, table rows given = {}",
                    self.layer,
                    e.is_some(),
                    r.is_some()
                )));
            }
        };

        let (g, h) = self.mix(&x, hc_attn, kind)?;
        let a = self.attend(w, &h, wave, owners)?;
        let x = post(&a, &x, &g.post)?;

        let (g, h) = self.mix(&x, hc_ffn, kind)?;
        if let Some((next_blk, next_w)) = next {
            next_blk.prefetch_from(next_w, &h)?;
        }
        let (f, routes) = self.ffn.forward(w, &h, kind)?;
        self.end_layer();
        Ok((post(&f, &x, &g.post)?, routes))
    }

    /// The attention sublayer output `[n, dim]` for `h [n, dim]`: the
    /// projections over every row, the core per owner.
    fn attend(
        &self,
        w: &LayerWeights,
        h: &MxArray,
        wave: &Wave,
        owners: &mut dyn Owners,
    ) -> Result<MxArray> {
        let p = self.mla.project_timed(&w.attn, h, &wave.positions())?;
        let mut outs = Vec::with_capacity(wave.owners());
        for (i, s) in wave.spans().iter().enumerate() {
            let po = p.owner(wave, i)?;
            let call = Core {
                blk: self,
                w,
                p: &po,
                start: s.start,
                kind: wave.kind(),
            };
            outs.push(owners.core(i, call)?);
        }
        let o = match outs.len() {
            1 => outs.swap_remove(0),
            _ => MxArray::concatenate_many(outs.iter().collect(), Some(0))?,
        };
        self.mla
            .finish_timed_with_gate(&w.attn, &p.x, p.gate_lin.as_ref(), &o)
    }

    /// The attention core of one owner's rows `p` at positions `start..`
    /// on its own flat state `st` of this layer (`Mla::core`).
    pub fn core_flat(
        &self,
        w: &LayerWeights,
        p: &Projected,
        start: usize,
        st: &mut LayerState,
        kind: RowKind,
    ) -> Result<MxArray> {
        let index = match (&self.indexer, &w.indexer, st.index.as_mut()) {
            (Some(indexer), Some(weights), Some(state)) => Some(Indexing {
                indexer,
                weights,
                state,
                kind,
            }),
            (None, None, None) => None,
            _ => {
                return Err(bad(format!(
                    "layer {}: indexer, its weights and its state do not match",
                    self.layer
                )));
            }
        };
        self.mla.core(&w.attn, p, start, &mut st.cache, index)
    }

    /// A sublayer's gates and its reduced input, under the HC clock.
    fn mix(&self, x: &MxArray, hw: &HcWeights, kind: RowKind) -> Result<(HcGates, MxArray)> {
        let tp = prof_start();
        let g = match group(Op::HcMix, kind) {
            Group::Row => self.mix_per_row(x, hw)?,
            Group::Fused | Group::Owner => hc_mix(x, hw, self.eps, &self.hc)?,
        };
        let h = hc_reduce(x, &g.pre)?;
        prof_end(Phase::Hc, tp, &[&g.post, &h])?;
        Ok((g, h))
    }

    fn mix_per_row(&self, x: &MxArray, hw: &HcWeights) -> Result<HcGates> {
        let n = x.shape_at(0)?;
        if n == 1 {
            return hc_mix(x, hw, self.eps, &self.hc);
        }
        let (mut pre, mut post) = (Vec::new(), Vec::new());
        for r in 0..n {
            let row = x.slice(&[r, 0, 0], &[r + 1, x.shape_at(1)?, x.shape_at(2)?])?;
            let g = hc_mix(&row, hw, self.eps, &self.hc)?;
            pre.push(g.pre);
            post.push(g.post);
        }
        Ok(HcGates {
            pre: MxArray::concatenate_many(pre.iter().collect(), Some(0))?,
            post: MxArray::concatenate_many(post.iter().collect(), Some(0))?,
        })
    }
}

/// `hc_post_identity` under the HC clock.
fn post(y: &MxArray, x: &MxArray, gate: &MxArray) -> Result<MxArray> {
    let tp = prof_start();
    let out = hc_post_identity(y, x, gate)?;
    prof_end(Phase::Hc, tp, &[&out])?;
    Ok(out)
}
