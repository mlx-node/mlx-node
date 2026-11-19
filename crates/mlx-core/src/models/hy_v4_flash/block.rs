//! One trunk layer (llama.cpp PR #32517 graph loop), f32 streams `[n, hc, d]`:
//!
//! ```text
//! X = X + engram(X)                               Engram layers only, before the attention mix
//! (pre, post) = hc_mix(X, hc_attn)   X = X + post · attn(Σ pre_c X_c)
//! (pre, post) = hc_mix(X, hc_ffn)    X = X + post · ffn(Σ pre_c X_c)
//! ```
//!
//! Each sublayer reduces the streams with the `pre` of its own mix.

use napi::{Error, Result};

use super::attention::{Geometry, Mla};
use super::cache::LayerCache;
use super::config::{Attn, HcConfig, HyV4FlashConfig};
use super::engram;
use super::indexer::{Indexer, IndexerGeometry, IndexerState, Indexing};
use super::moe::{ExpertSource, FfnConfig, LayerFfn, Routes};
use super::ops::{hc_mix, hc_post_identity, hc_reduce};
use super::weights::{LayerWeights, Residual};
use crate::array::MxArray;

fn bad(msg: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("Hy4 block: {msg}"))
}

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

    /// The streams `x [n, hc, d]` (f32) after this layer, for positions
    /// `start..start + n`; `rows` = this layer's Engram table rows `[n,
    /// cols, head_dim]` (Engram layers only). Appends the call to `st`; on
    /// error the attention state holds what it held before, the caller
    /// restores the rest.
    pub fn forward(
        &self,
        w: &LayerWeights,
        x: &MxArray,
        start: usize,
        st: &mut LayerState,
        rows: Option<&[f32]>,
    ) -> Result<(MxArray, Option<Routes>)> {
        let Residual::Hc {
            attn: hc_attn,
            ffn: hc_ffn,
        } = &w.residual
        else {
            return Err(bad(format!("layer {}: no HC weights", self.layer)));
        };
        let x = match (self.engram, &w.engram, rows) {
            (Some(_), Some(ew), Some(r)) => engram::forward(x, r, ew)?,
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

        let g = hc_mix(&x, hc_attn, self.eps, &self.hc)?;
        let h = hc_reduce(&x, &g.pre)?;
        let index = match (&self.indexer, &w.indexer, st.index.as_mut()) {
            (Some(indexer), Some(weights), Some(state)) => Some(Indexing {
                indexer,
                weights,
                state,
            }),
            (None, None, None) => None,
            _ => {
                return Err(bad(format!(
                    "layer {}: indexer, its weights and its state do not match",
                    self.layer
                )));
            }
        };
        let a = self.mla.forward(&w.attn, &h, start, &mut st.cache, index)?;
        let x = hc_post_identity(&a, &x, &g.post)?;

        let g = hc_mix(&x, hc_ffn, self.eps, &self.hc)?;
        let h = hc_reduce(&x, &g.pre)?;
        let (f, routes) = self.ffn.forward(w, &h)?;
        Ok((hc_post_identity(&f, &x, &g.post)?, routes))
    }
}
