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
use super::ops::{hc_bridge, hc_mix_reduce_norm, post_norm_hc_post};
use super::stats::{Phase, prof_end, prof_start};
use super::weights::{HcWeights, LayerWeights, Residual};
use crate::array::MxArray;

fn bad(msg: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("Hy4 block: {msg}"))
}

/// Fewest prefill rows that trigger next-layer expert read-ahead.
pub const READ_AHEAD_MIN_ROWS: usize = 256;

/// Most decode rows that trigger next-layer expert read-ahead.
pub const DECODE_READ_AHEAD_MAX_ROWS: usize = 4;

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

/// Stream state threaded across trunk layers so `post_norm + hc_post_identity`
/// can fuse directly into the next sublayer's `hc_mix_reduce + input_norm`.
pub(crate) enum HcCarry {
    /// Materialized streams `[n, hc, d]`.
    Streams(MxArray),
    /// Previous sublayer's raw output `y [n, d]` waiting to be post-normed and
    /// added into `res [n, hc, d]` with gate `post [n, hc]` and `post_norm [d]`.
    Pending {
        y: MxArray,
        res: MxArray,
        post: MxArray,
        post_norm: MxArray,
    },
}

impl HcCarry {
    pub(crate) fn into_streams(self, eps: f32) -> Result<MxArray> {
        match self {
            Self::Streams(x) => Ok(x),
            Self::Pending {
                y,
                res,
                post,
                post_norm,
            } => {
                let tp = prof_start();
                let out = post_norm_hc_post(&y, &res, &post, &post_norm, eps)?;
                prof_end(Phase::Hc, tp, &[&out])?;
                Ok(out)
            }
        }
    }
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
        self.new_state_with_slack(cap, super::cache::WINDOW_SLACK)
    }

    pub(crate) fn new_state_with_slack(&self, cap: usize, window_slack: usize) -> LayerState {
        let g = self.mla.geometry();
        LayerState {
            cache: LayerCache::new_with_slack(g.attn, g.window, window_slack, cap, g.row() as i64),
            index: self
                .indexer
                .as_ref()
                .map(|ix| IndexerState::new(cap, ix.geometry())),
        }
    }

    pub fn prefetch_from(&self, w: &LayerWeights, h: &MxArray) -> Result<usize> {
        self.ffn.prefetch_from(w, h)
    }

    pub fn speculative_router_scores(
        &self,
        w: &LayerWeights,
        h: &MxArray,
    ) -> Result<Option<MxArray>> {
        self.ffn.speculative_router_scores(w, h)
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
        self.forward_traced(w, x, wave, owners, engram, next, true)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_traced(
        &self,
        w: &LayerWeights,
        x: &MxArray,
        wave: &Wave,
        owners: &mut dyn Owners,
        engram: Option<EngramIn<'_>>,
        next: Option<(&Block, &LayerWeights)>,
        need_routes: bool,
    ) -> Result<(MxArray, Option<Routes>)> {
        let (carry, routes) = self.forward_carry(
            w,
            HcCarry::Streams(x.clone()),
            wave,
            owners,
            engram,
            next,
            need_routes,
            None,
        )?;
        Ok((carry.into_streams(self.eps)?, routes))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_carry(
        &self,
        w: &LayerWeights,
        carry: HcCarry,
        wave: &Wave,
        owners: &mut dyn Owners,
        engram: Option<EngramIn<'_>>,
        next: Option<(&Block, &LayerWeights)>,
        need_routes: bool,
        prev_out: Option<&mut Option<MxArray>>,
    ) -> Result<(HcCarry, Option<Routes>)> {
        self.forward_carry_with_pos(
            w,
            carry,
            wave,
            None,
            owners,
            engram,
            next,
            need_routes,
            prev_out,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_carry_with_pos(
        &self,
        w: &LayerWeights,
        carry: HcCarry,
        wave: &Wave,
        pos_f32: Option<&MxArray>,
        owners: &mut dyn Owners,
        engram: Option<EngramIn<'_>>,
        next: Option<(&Block, &LayerWeights)>,
        need_routes: bool,
        prev_out: Option<&mut Option<MxArray>>,
    ) -> Result<(HcCarry, Option<Routes>)> {
        let Residual::Hc {
            attn: hc_attn,
            ffn: hc_ffn,
        } = &w.residual
        else {
            return Err(bad(format!("layer {}: no HC weights", self.layer)));
        };
        let n = match &carry {
            HcCarry::Streams(x) => x.shape_at(0)?,
            HcCarry::Pending { res, .. } => res.shape_at(0)?,
        };
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

        let (x, h_attn_norm, post_attn) = match (self.engram, &w.engram, engram) {
            (Some(_), Some(ew), Some(e)) => {
                let x_in = carry.into_streams(self.eps)?;
                if let Some(slot) = prev_out {
                    *slot = Some(x_in.clone());
                }
                let tp = prof_start();
                let x_eng = engram::forward(&x_in, e.rows, e.keep, ew)?;
                prof_end(Phase::Engram, tp, &[&x_eng])?;
                let (h_norm, post_a) = self.mix_norm(&x_eng, hc_attn, &w.attn.norm, kind)?;
                (x_eng, h_norm, post_a)
            }
            (None, None, None) => match carry {
                HcCarry::Streams(x_in) => {
                    let (h_norm, post_a) = self.mix_norm(&x_in, hc_attn, &w.attn.norm, kind)?;
                    (x_in, h_norm, post_a)
                }
                HcCarry::Pending {
                    y,
                    res,
                    post: prev_post,
                    post_norm,
                } => {
                    let (x_in, h_norm, post_a) = self.bridge(
                        &y,
                        &res,
                        &prev_post,
                        &post_norm,
                        hc_attn,
                        Some(&w.attn.norm),
                        kind,
                    )?;
                    if let Some(slot) = prev_out {
                        *slot = Some(x_in.clone());
                    }
                    (x_in, h_norm, post_a)
                }
            },
            (e, _, r) => {
                return Err(bad(format!(
                    "layer {}: Engram layer = {}, table rows given = {}",
                    self.layer,
                    e.is_some(),
                    r.is_some()
                )));
            }
        };

        let a_raw = self.attend_normed_raw(w, &h_attn_norm, wave, pos_f32, owners)?;

        let (x_mid, h_ffn_norm, post_ffn) = self.bridge(
            &a_raw,
            &x,
            &post_attn,
            &w.attn.post_norm,
            hc_ffn,
            Some(&w.ffn_norm),
            kind,
        )?;
        let decode_next = if kind.is_decode() {
            next.map(|(next_blk, next_w)| (&next_blk.ffn, next_w))
        } else {
            if let Some((next_blk, next_w)) = next {
                next_blk.prefetch_from(next_w, &h_ffn_norm)?;
            }
            None
        };
        let (f_raw, routes) =
            self.ffn
                .forward_normed_raw(w, &h_ffn_norm, kind, need_routes, decode_next)?;
        self.end_layer();
        Ok((
            HcCarry::Pending {
                y: f_raw,
                res: x_mid,
                post: post_ffn,
                post_norm: w.post_ffw_norm.clone(),
            },
            routes,
        ))
    }

    /// Attention sublayer on pre-normalized `x_norm [n, dim]`, returning the
    /// raw output before `post_attention_norm`.
    fn attend_normed_raw(
        &self,
        w: &LayerWeights,
        x_norm: &MxArray,
        wave: &Wave,
        pos_f32: Option<&MxArray>,
        owners: &mut dyn Owners,
    ) -> Result<MxArray> {
        let split_kv = owners.is_paged() && self.mla.geometry().attn == Attn::Indexed;
        let split_q = wave.kind() == RowKind::Prefill
            && wave.rows() >= 16
            && self.mla.geometry().kv_lora == 512
            && self.mla.geometry().rope_dim == 64;
        let p = match pos_f32 {
            Some(pos) => self
                .mla
                .project_normed_timed_with_pos_split_q(&w.attn, x_norm, pos, split_kv, split_q)?,
            None => {
                let pos = super::ops::rope_positions_f32(&wave.positions())?;
                self.mla.project_normed_timed_with_pos_split_q(
                    &w.attn, x_norm, &pos, split_kv, split_q,
                )?
            }
        };
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
            .finish_raw_timed_with_gate(&w.attn, &p.x, p.gate_lin.as_ref(), &o)
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
        self.mla
            .core_for(&w.attn, p, start, &mut st.cache, index, kind)
    }

    /// Fused HC mix + reduce + input RMSNorm under the HC clock.
    fn mix_norm(
        &self,
        x: &MxArray,
        hw: &HcWeights,
        norm_w: &MxArray,
        kind: RowKind,
    ) -> Result<(MxArray, MxArray)> {
        let tp = prof_start();
        let (h, post) = match group(Op::HcMix, kind) {
            Group::Row => self.mix_norm_per_row(x, hw, norm_w)?,
            Group::Fused | Group::Owner => hc_mix_reduce_norm(x, hw, norm_w, self.eps, &self.hc)?,
        };
        prof_end(Phase::Hc, tp, &[&post, &h])?;
        Ok((h, post))
    }

    fn mix_norm_per_row(
        &self,
        x: &MxArray,
        hw: &HcWeights,
        norm_w: &MxArray,
    ) -> Result<(MxArray, MxArray)> {
        let n = x.shape_at(0)?;
        if n == 1 {
            return hc_mix_reduce_norm(x, hw, norm_w, self.eps, &self.hc);
        }
        let (mut hs, mut posts) = (Vec::new(), Vec::new());
        for r in 0..n {
            let row = x.slice(&[r, 0, 0], &[r + 1, x.shape_at(1)?, x.shape_at(2)?])?;
            let (h, post) = hc_mix_reduce_norm(&row, hw, norm_w, self.eps, &self.hc)?;
            hs.push(h);
            posts.push(post);
        }
        Ok((
            MxArray::concatenate_many(hs.iter().collect(), Some(0))?,
            MxArray::concatenate_many(posts.iter().collect(), Some(0))?,
        ))
    }

    /// Megafused sublayer boundary bridge under the HC clock.
    #[allow(clippy::too_many_arguments)]
    fn bridge(
        &self,
        y: &MxArray,
        res: &MxArray,
        post_prev: &MxArray,
        post_norm_w: &MxArray,
        next_hc: &HcWeights,
        next_norm_w: Option<&MxArray>,
        kind: RowKind,
    ) -> Result<(MxArray, MxArray, MxArray)> {
        let tp = prof_start();
        let (x_out, h_out, post_next) = match group(Op::HcMix, kind) {
            Group::Row => {
                let n = res.shape_at(0)?;
                if n == 1 {
                    hc_bridge(
                        y,
                        res,
                        post_prev,
                        post_norm_w,
                        next_hc,
                        next_norm_w,
                        self.eps,
                        &self.hc,
                    )?
                } else {
                    let (hc, d) = (res.shape_at(1)?, res.shape_at(2)?);
                    let (mut xs, mut hs, mut ps) = (Vec::new(), Vec::new(), Vec::new());
                    for r in 0..n {
                        let yr = y.slice(&[r, 0], &[r + 1, d])?;
                        let rr = res.slice(&[r, 0, 0], &[r + 1, hc, d])?;
                        let pr = post_prev.slice(&[r, 0], &[r + 1, hc])?;
                        let (xo, ho, po) = hc_bridge(
                            &yr,
                            &rr,
                            &pr,
                            post_norm_w,
                            next_hc,
                            next_norm_w,
                            self.eps,
                            &self.hc,
                        )?;
                        xs.push(xo);
                        hs.push(ho);
                        ps.push(po);
                    }
                    (
                        MxArray::concatenate_many(xs.iter().collect(), Some(0))?,
                        MxArray::concatenate_many(hs.iter().collect(), Some(0))?,
                        MxArray::concatenate_many(ps.iter().collect(), Some(0))?,
                    )
                }
            }
            Group::Fused | Group::Owner => hc_bridge(
                y,
                res,
                post_prev,
                post_norm_w,
                next_hc,
                next_norm_w,
                self.eps,
                &self.hc,
            )?,
        };
        prof_end(Phase::Hc, tp, &[&x_out, &h_out, &post_next])?;
        Ok((x_out, h_out, post_next))
    }
}
