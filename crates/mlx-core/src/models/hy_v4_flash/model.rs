//! The Hy4 trunk (llama.cpp PR #32517 graph), f32 activations:
//!
//! ```text
//! ids ─► token_embd rows ─► hc copies ─► 42 × Block ─► hc_head ─► rms·output_norm = h_trunk ─► output ─► logits
//! ```
//!
//! A step runs layer by layer over the rows of one or more owners (a
//! `Wave`): decode rows of any number of sequences, or one sequence's
//! prefill slice, which is never fused with other rows. Ops group as
//! `batch.rs` lists them, so an owner's rows give the same bits alone and
//! among others. `step` runs flat sessions; `wave` runs any `Owners`.
//!
//! A `Session` is one sequence; its latent and pooled buffers are written
//! in place, so a session is never cloned: snapshots are lengths plus
//! window clones (`Mark`), and a failed or cancelled step puts every owner
//! back at its start.

use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use napi::{Error, Result};

use super::bank::KqBank;
use super::batch::{Core, Group, Op, Owners, Wave, group};
use super::block::{Block, EngramIn, HcCarry, LayerState};
use super::cache::{LatentBuf, LayerCache, WINDOW_SLACK, WindowRing, WindowRingSnapshot};
use super::config::HyV4FlashConfig;
use super::engram::{Hasher, keeps};
use super::indexer::{IndexerMark, IndexerState};
use super::moe::{ExpertSource, Routes, RowKind, per_row, qlinear};
use super::ops::{
    HEAD_CHUNK, hc_bridge_head, hc_head_norm, head_chunks, head_logits, rope_positions_f32,
    try_bf16_lm_head,
};
use super::prefix_state::StateLayer;
use super::rows::{EngramRows, PrefetchHandle};
use super::stats::{DeferScope, Phase, prof_end, prof_start};
use super::weights::{Dense, HcWeights, LayerWeights, Resident};
use crate::array::{DType, MxArray};

fn bad(msg: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("Hy4 model: {msg}"))
}

/// The trunk tensors a `Model` runs on.
pub trait TrunkWeights {
    fn token_embd(&self) -> &Dense;
    fn output(&self) -> &Dense;
    fn output_norm(&self) -> &MxArray;
    fn output_hc(&self) -> &HcWeights;
    fn layers(&self) -> &[LayerWeights];
}

impl TrunkWeights for Resident {
    fn token_embd(&self) -> &Dense {
        &self.token_embd
    }
    fn output(&self) -> &Dense {
        &self.output
    }
    fn output_norm(&self) -> &MxArray {
        &self.output_norm
    }
    fn output_hc(&self) -> &HcWeights {
        &self.output_hc
    }
    fn layers(&self) -> &[LayerWeights] {
        &self.layers
    }
}

/// Which rows of a call get logits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Logits {
    None,
    Last,
    All,
}

/// What a call returns besides the state update.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Want {
    pub logits: Logits,
    /// Every row's `h_trunk = rms(hc_head(X))·output_norm` `[n, d]` f32
    /// (the MTP layer's input).
    pub hidden: bool,
}

impl Want {
    pub const LAST: Self = Self {
        logits: Logits::Last,
        hidden: false,
    };
    pub const ALL: Self = Self {
        logits: Logits::All,
        hidden: false,
    };
    pub const STATE: Self = Self {
        logits: Logits::None,
        hidden: false,
    };
}

pub struct Output {
    /// f32 `[1 or n, vocab]`.
    pub logits: Option<MxArray>,
    /// f32 `[n, d]`.
    pub hidden: Option<MxArray>,
}

/// Per-layer debug dump of one step (owners' rows in order).
#[derive(Default)]
pub struct Trace {
    /// Engram table rows `[pos][engram layer][col]` (global ids).
    pub engram_rows: Vec<u32>,
    /// One entry per layer (`None` on the dense one), every row of the step.
    pub routes: Vec<Option<Routes>>,
    /// Keep each layer's output streams `[n, hc, d]` in `out`.
    pub keep_out: bool,
    pub out: Vec<MxArray>,
    /// Layer `l`'s input streams `[n, hc, d]` replaced by `inputs[l]` when
    /// it is `Some`.
    #[cfg(test)]
    pub inputs: Vec<Option<MxArray>>,
}

/// One owner's part of a step: `ids` appended to `sess`, and what the step
/// returns for them.
pub struct OwnerRows<'a> {
    pub sess: &'a mut Session,
    pub ids: &'a [u32],
    pub want: Want,
}

/// `Owners` over flat sessions; the sessions' tokens are appended by
/// `Model::step` after the wave.
struct FlatOwners<'s, 'a> {
    owners: &'s mut [OwnerRows<'a>],
}

impl Owners for FlatOwners<'_, '_> {
    fn count(&self) -> usize {
        self.owners.len()
    }

    fn ids(&self, owner: usize) -> &[u32] {
        match self.owners.get(owner) {
            Some(o) => o.ids,
            None => &[],
        }
    }

    fn history(&self, owner: usize) -> &[u32] {
        match self.owners.get(owner) {
            Some(o) => o.sess.tokens(),
            None => &[],
        }
    }

    fn want(&self, owner: usize) -> Want {
        self.owners.get(owner).map_or(Want::STATE, |o| o.want)
    }

    fn core(&mut self, owner: usize, call: Core<'_>) -> Result<MxArray> {
        let l = call.blk.layer();
        let st = self
            .owners
            .get_mut(owner)
            .ok_or_else(|| bad(format!("owner {owner} of the wave")))?
            .sess
            .layers
            .get_mut(l)
            .ok_or_else(|| bad(format!("session without layer {l}")))?;
        call.blk
            .core_flat(call.w, call.p, call.start, st, call.kind)
    }
}

/// f32 logits `[s, vocab]` of `x [s, d]`: each row is `head_logits` of
/// that row alone, the f32 head chunks cast once for all rows. A Q8 head
/// is one `qlinear` call.
pub fn head_rows(x: &MxArray, head: &Dense) -> Result<MxArray> {
    let xf = x.astype(DType::Float32)?;
    if head.is_qmm() {
        return qlinear(&xf, head);
    }
    if let Some(logits) = try_bf16_lm_head(&xf, head)? {
        return Ok(logits);
    }
    if xf.shape_at(0)? == 1 {
        return head_logits(&xf, head);
    }
    head_chunks(head, HEAD_CHUNK, xf.shape_at(0)?, |w| {
        let wt = w.transpose(Some(&[1, 0]))?;
        per_row(&xf, |r| r.matmul(&wt))
    })
}

/// Engram table rows `[pos][layer][col]` of the call `ids` after `history`,
/// hashing only the last `back` ids of the history (the n-gram reach).
pub fn call_engram_rows(
    hasher: &Hasher,
    back: usize,
    history: &[u32],
    ids: &[u32],
) -> Result<Vec<u32>> {
    let k = back.min(history.len());
    let mut ctx = Vec::with_capacity(k + ids.len());
    ctx.extend_from_slice(&history[history.len() - k..]);
    ctx.extend_from_slice(ids);
    hasher.call_rows(&ctx, k)
}

/// The latent rows and the indexer of indexed layer `l`.
fn latent_of(st: &mut LayerState, l: usize) -> Result<(&mut LatentBuf, &mut IndexerState)> {
    match (&mut st.cache, st.index.as_mut()) {
        (LayerCache::Latent(buf), Some(ix)) => Ok((buf, ix)),
        _ => Err(bad(format!("layer {l} is not indexed"))),
    }
}

/// Whether `IndexerState::truncate(len)` keeps the open `tok_k` rows it
/// needs (the refusal rule of `indexer.rs` `truncate`).
fn index_keeps(st: &IndexerState, len: usize, kpool: usize) -> bool {
    let closed = len / kpool * kpool;
    let first = st.len() - st.held();
    len <= closed || closed >= first
}

#[derive(Clone)]
enum LayerMark {
    Window(Option<WindowRingSnapshot>),
    Latent {
        len: usize,
        index: Option<IndexerMark>,
    },
}

enum StepMark {
    Short {
        len: usize,
        index: Vec<Option<IndexerMark>>,
    },
    Full(Mark),
}

/// A point of a session to go back to: valid while no write lands below
/// its length (`Session::restore` refuses it after one).
#[derive(Clone)]
pub struct Mark {
    len: usize,
    /// The call that wrote position `len - 1`.
    last: Option<u64>,
    layers: Vec<LayerMark>,
}

impl Mark {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Each trunk layer's part outside the latent and pooled buffers at
    /// the mark: a sliding layer's ring, an indexed layer's `tok_k` tail.
    pub fn state_layers(&self) -> Result<Vec<StateLayer>> {
        self.layers
            .iter()
            .enumerate()
            .map(|(l, m)| match m {
                LayerMark::Window(Some(r)) => Ok(StateLayer::Window(r.clone())),
                LayerMark::Window(None) => {
                    Err(bad(format!("layer {l}: window snapshot unavailable")))
                }
                LayerMark::Latent {
                    index: Some(im), ..
                } => Ok(StateLayer::Tail(im.tail().clone())),
                LayerMark::Latent { index: None, .. } => Err(bad(format!(
                    "layer {l}: an indexed layer without an indexer"
                ))),
            })
            .collect()
    }
}

/// One sequence's state: its token ids and every layer's attention state.
pub struct Session {
    tokens: Vec<u32>,
    /// Per position, the call that wrote it (`calls` at that call).
    written: Vec<u64>,
    calls: u64,
    cap: usize,
    kpool: usize,
    layers: Vec<LayerState>,
}

impl Session {
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Most positions the session may hold.
    pub fn cap(&self) -> usize {
        self.cap
    }

    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    pub fn layers(&self) -> &[LayerState] {
        &self.layers
    }

    pub fn nbytes(&self) -> usize {
        self.layers.iter().map(LayerState::nbytes).sum::<usize>()
            + self.tokens.len() * std::mem::size_of::<u32>()
            + self.written.len() * std::mem::size_of::<u64>()
    }

    pub fn mark(&self) -> Mark {
        let mut layers: Vec<LayerMark> = self
            .layers
            .iter()
            .map(|st| match &st.cache {
                LayerCache::Window(r) => LayerMark::Window(r.snapshot_lazy().ok()),
                LayerCache::Latent(b) => LayerMark::Latent {
                    len: b.len(),
                    index: st.index.as_ref().map(IndexerState::mark),
                },
            })
            .collect();
        let live: Vec<&MxArray> = layers
            .iter()
            .filter_map(|lm| match lm {
                LayerMark::Window(Some(s)) => s.rows(),
                _ => None,
            })
            .collect();
        if !live.is_empty()
            && MxArray::eval_arrays_with_context(&live, "hy_v4_flash session mark").is_err()
        {
            for lm in &mut layers {
                if let LayerMark::Window(w) = lm {
                    *w = None;
                }
            }
        }
        Mark {
            len: self.tokens.len(),
            last: self.written.last().copied(),
            layers,
        }
    }

    fn step_mark(&self, rows: usize) -> StepMark {
        if rows <= WINDOW_SLACK {
            StepMark::Short {
                len: self.tokens.len(),
                index: self
                    .layers
                    .iter()
                    .map(|st| st.index.as_ref().map(IndexerState::mark))
                    .collect(),
            }
        } else {
            StepMark::Full(self.mark())
        }
    }

    fn restore_step(&mut self, m: &StepMark) -> Result<()> {
        match m {
            StepMark::Full(mark) => self.restore(mark),
            StepMark::Short { len, index } => {
                for (st, im) in self.layers.iter_mut().zip(index) {
                    if let (Some(ix), Some(saved)) = (st.index.as_mut(), im) {
                        ix.restore(saved.clone())?;
                    }
                }
                for st in &mut self.layers {
                    match &mut st.cache {
                        LayerCache::Window(r) if r.len() > *len => r.truncate(*len)?,
                        LayerCache::Latent(b) if b.len() > *len => b.truncate(*len)?,
                        _ => {}
                    }
                }
                self.tokens.truncate(*len);
                self.written.truncate(*len);
                Ok(())
            }
        }
    }

    /// Goes back to `m` (taken from this session at a length <= now).
    /// Refused, with nothing changed, when a position below `m`'s length was
    /// rewritten since `m` was taken.
    pub fn restore(&mut self, m: &Mark) -> Result<()> {
        self.fits(m)?;
        let rebuilt: Vec<Option<WindowRing>> = self
            .layers
            .iter()
            .zip(&m.layers)
            .map(|(st, lm)| match (&st.cache, lm) {
                (LayerCache::Window(_), LayerMark::Window(Some(saved))) => {
                    WindowRing::from_snapshot(saved, WINDOW_SLACK).map(Some)
                }
                (LayerCache::Latent(_), LayerMark::Latent { .. }) => Ok(None),
                _ => Err(bad("layer kinds changed during restore")),
            })
            .collect::<Result<_>>()?;
        for (st, lm) in self.layers.iter_mut().zip(&m.layers) {
            if let (
                Some(ix),
                LayerMark::Latent {
                    index: Some(im), ..
                },
            ) = (st.index.as_mut(), lm)
            {
                ix.restore(im.clone())?;
            }
        }
        for ((st, lm), ring) in self.layers.iter_mut().zip(&m.layers).zip(rebuilt) {
            match (&mut st.cache, lm, ring) {
                (LayerCache::Window(r), LayerMark::Window(_), Some(fresh)) => *r = fresh,
                (LayerCache::Latent(b), LayerMark::Latent { len, .. }, None) => b.truncate(*len)?,
                _ => return Err(bad("layer kinds changed during restore")),
            }
        }
        self.tokens.truncate(m.len);
        self.written.truncate(m.len);
        Ok(())
    }

    /// Whether `restore(m)` would succeed.
    pub fn can_restore(&self, m: &Mark) -> bool {
        self.fits(m).is_ok()
    }

    /// Fills an empty session with the `tokens` of a stored state: every
    /// sliding layer gets its ring of `layers`, every indexed layer (in
    /// trunk order) its latent rows and pooled keys from `runs`, one
    /// `(rows [n, width], keys [n / kpool, d])` per indexed layer each, and
    /// its tail. On error the session stays empty.
    pub fn fill(
        &mut self,
        tokens: &[u32],
        layers: &[StateLayer],
        runs: &mut dyn Iterator<Item = Vec<(MxArray, MxArray)>>,
    ) -> Result<()> {
        let b = tokens.len();
        if !self.tokens.is_empty() || b == 0 || b > self.cap {
            return Err(bad(format!(
                "a fill of {b} positions into a session of {} (context {})",
                self.tokens.len(),
                self.cap
            )));
        }
        if layers.len() != self.layers.len() {
            return Err(bad(format!(
                "a state of {} layers for a trunk of {}",
                layers.len(),
                self.layers.len()
            )));
        }
        let out = self.fill_layers(b, layers, runs);
        if out.is_err() {
            for st in &mut self.layers {
                let _ = st.cache.truncate(0);
                if let Some(ix) = st.index.as_mut() {
                    let _ = ix.truncate(0);
                }
            }
            return out;
        }
        self.calls += 1;
        self.tokens.extend_from_slice(tokens);
        self.written = vec![self.calls; b];
        Ok(())
    }

    fn fill_layers(
        &mut self,
        b: usize,
        layers: &[StateLayer],
        runs: &mut dyn Iterator<Item = Vec<(MxArray, MxArray)>>,
    ) -> Result<()> {
        let mut indexed = Vec::new();
        for (l, (st, part)) in self.layers.iter_mut().zip(layers).enumerate() {
            match (&mut st.cache, st.index.as_ref(), part) {
                (LayerCache::Window(r), None, StateLayer::Window(s)) if s.len() == b => {
                    *r = WindowRing::from_snapshot(s, WINDOW_SLACK)?;
                }
                (LayerCache::Latent(_), Some(_), StateLayer::Tail(t)) if t.len() == b => {
                    indexed.push((l, t));
                }
                _ => return Err(bad(format!("layer {l}: the state is not of this model"))),
            }
        }
        for run in runs {
            if run.len() != indexed.len() {
                return Err(bad(format!(
                    "{} latent row sets for {} indexed layers",
                    run.len(),
                    indexed.len()
                )));
            }
            for (&(l, _), (rows, keys)) in indexed.iter().zip(&run) {
                let (buf, ix) = latent_of(&mut self.layers[l], l)?;
                let at = buf.len();
                buf.write(rows, at)?;
                ix.append_pooled(keys)?;
            }
        }
        for (l, t) in indexed {
            let (buf, ix) = latent_of(&mut self.layers[l], l)?;
            if buf.len() != b {
                return Err(bad(format!(
                    "layer {l}: latent rows end at {} of {b}",
                    buf.len()
                )));
            }
            ix.close_fill(t.clone())?;
        }
        Ok(())
    }

    fn fits(&self, m: &Mark) -> Result<()> {
        if m.layers.len() != self.layers.len() || m.len > self.tokens.len() {
            return Err(bad(format!(
                "mark of {} positions / {} layers for a session of {} / {}",
                m.len,
                m.layers.len(),
                self.tokens.len(),
                self.layers.len()
            )));
        }
        let last = m
            .len
            .checked_sub(1)
            .and_then(|i| self.written.get(i).copied());
        if last != m.last {
            return Err(bad(format!(
                "mark of {} positions: its rows were rewritten since",
                m.len
            )));
        }
        for (l, (st, lm)) in self.layers.iter().zip(&m.layers).enumerate() {
            let fits = match (&st.cache, lm) {
                (LayerCache::Window(_), LayerMark::Window(Some(_))) => st.index.is_none(),
                (LayerCache::Latent(b), LayerMark::Latent { len, index }) => {
                    *len <= b.len() && st.index.is_some() == index.is_some()
                }
                _ => false,
            };
            if !fits {
                return Err(bad(format!("layer {l}: the mark is not of this session")));
            }
        }
        Ok(())
    }

    /// Drops the positions past `len`. Refused, with nothing changed, when
    /// a window or an indexer no longer holds the rows `len` needs.
    pub fn truncate(&mut self, len: usize) -> Result<()> {
        let now = self.tokens.len();
        if len > now {
            return Err(bad(format!("truncate to {len}, {now} positions held")));
        }
        if len == now {
            return Ok(());
        }
        if let Some(l) = self.layers.iter().position(|st| {
            st.index
                .as_ref()
                .is_some_and(|ix| !index_keeps(ix, len, self.kpool))
        }) {
            return Err(bad(format!(
                "truncate {now} -> {len}: layer {l}'s indexer no longer holds the open rows"
            )));
        }
        if let Some(l) = self.layers.iter().position(|st| match &st.cache {
            LayerCache::Window(r) => !r.can_truncate(len),
            LayerCache::Latent(_) => false,
        }) {
            return Err(bad(format!(
                "truncate {now} -> {len}: layer {l}'s window no longer holds the rows"
            )));
        }
        for st in &mut self.layers {
            if let Some(ix) = st.index.as_mut() {
                ix.truncate(len)?;
            }
            match &mut st.cache {
                LayerCache::Window(r) => r.truncate(len)?,
                LayerCache::Latent(b) => b.truncate(len)?,
            }
        }
        self.tokens.truncate(len);
        self.written.truncate(len);
        Ok(())
    }
}

struct PrefillChunkPlan {
    start: usize,
    end: usize,
    /// Pre-hashed Engram table row IDs per Engram layer (`[n_engram][rows * n_cols]`).
    layer_ids: Vec<Vec<u32>>,
    /// Handles pre-submitted during the prior chunk (`[n_engram]`).
    handles: Vec<Option<PrefetchHandle>>,
}

struct EngramLookahead {
    prompt: Vec<u32>,
    chunks: Vec<PrefillChunkPlan>,
    cursor: usize,
}

pub struct Model {
    cfg: HyV4FlashConfig,
    weights: Box<dyn TrunkWeights>,
    blocks: Vec<Block>,
    hasher: Hasher,
    table: Box<dyn EngramRows>,
    /// The distinct banks the blocks gather from.
    banks: Vec<Arc<KqBank>>,
    engram_lookahead: Mutex<Option<EngramLookahead>>,
}

/// Refuses an empty call or one past the session's context.
fn check_room(o: &OwnerRows<'_>) -> Result<()> {
    let (n, start) = (o.ids.len(), o.sess.tokens.len());
    if n == 0 {
        return Err(bad("empty call"));
    }
    if start + n > o.sess.cap {
        return Err(bad(format!(
            "{start} + {n} tokens exceed the context of {}",
            o.sess.cap
        )));
    }
    Ok(())
}

fn want_dims(d: &Dense, rows: usize, cols: usize, what: &str) -> Result<()> {
    let got = (d.shape_at(0)?, d.shape_at(1)?);
    if got != (rows as i64, cols as i64) {
        return Err(bad(format!("{what} is {got:?}, need ({rows}, {cols})")));
    }
    Ok(())
}

impl Model {
    /// `experts(layer)` gives the routed experts of each MoE trunk layer;
    /// `table` serves the Engram rows. The calling thread runs every later
    /// call of this model: it is set to true f32 GEMMs here.
    pub fn new(
        cfg: HyV4FlashConfig,
        weights: Box<dyn TrunkWeights>,
        mut experts: impl FnMut(usize) -> Result<ExpertSource>,
        table: Box<dyn EngramRows>,
    ) -> Result<Self> {
        // NAX GPUs run f32 GEMMs in TF32 under MLX's default unless the
        // thread opts out
        crate::array::set_thread_tf32(Some(false));
        let layers = weights.layers();
        if layers.len() != cfg.n_layers {
            return Err(bad(format!(
                "{} layer weight sets for {} trunk layers",
                layers.len(),
                cfg.n_layers
            )));
        }
        want_dims(weights.token_embd(), cfg.vocab_size, cfg.dim, "token_embd")?;
        want_dims(weights.output(), cfg.vocab_size, cfg.dim, "output")?;
        let mut banks: Vec<Arc<KqBank>> = Vec::new();
        let blocks = layers
            .iter()
            .enumerate()
            .map(|(l, w)| {
                let src = if cfg.layers[l].dense_ffn {
                    None
                } else {
                    Some(experts(l)?)
                };
                if let Some(ExpertSource::Bank(b)) = &src
                    && !banks.iter().any(|k| Arc::ptr_eq(k, b))
                {
                    banks.push(Arc::clone(b));
                }
                Block::new(&cfg, l, w, src)
            })
            .collect::<Result<Vec<_>>>()?;
        let hasher = Hasher::new(&cfg.engram)?;
        Ok(Self {
            cfg,
            weights,
            blocks,
            hasher,
            table,
            banks,
            engram_lookahead: Mutex::new(None),
        })
    }

    /// Pre-hashes Engram row IDs for all chunks of a multi-chunk prefill so
    /// chunk `k+1`'s Engram layers can be prefetched asynchronously during
    /// chunk `k`.
    pub fn begin_prefill_engram(&self, prompt: &[u32], from: usize, ends: &[usize]) {
        let n_engram = self.hasher.n_layers();
        let n_cols = self.hasher.n_cols();
        if ends.len() < 2 || from >= prompt.len() || n_engram == 0 || n_cols == 0 {
            self.clear_prefill_engram();
            return;
        }
        let Ok(all_rows) = self.hasher.call_rows(prompt, from) else {
            self.clear_prefill_engram();
            return;
        };
        let stride = n_engram * n_cols;
        let mut chunks = Vec::with_capacity(ends.len());
        let mut start = from;
        for &end in ends {
            if end <= start || end > prompt.len() {
                self.clear_prefill_engram();
                return;
            }
            let pos_lo = start - from;
            let pos_hi = end - from;
            let n_pos = pos_hi - pos_lo;
            let mut layer_ids = Vec::with_capacity(n_engram);
            for e in 0..n_engram {
                let mut ids_e = Vec::with_capacity(n_pos * n_cols);
                for p in pos_lo..pos_hi {
                    let base = p * stride + e * n_cols;
                    ids_e.extend_from_slice(&all_rows[base..base + n_cols]);
                }
                layer_ids.push(ids_e);
            }
            chunks.push(PrefillChunkPlan {
                start,
                end,
                layer_ids,
                handles: (0..n_engram).map(|_| None).collect(),
            });
            start = end;
        }
        *self
            .engram_lookahead
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = Some(EngramLookahead {
            prompt: prompt.to_vec(),
            chunks,
            cursor: 0,
        });
    }

    /// Clears any active prefill Engram lookahead plan and cancels unused handles.
    pub fn clear_prefill_engram(&self) {
        *self
            .engram_lookahead
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = None;
    }

    pub(crate) fn clear_prefill_engram_for_prompt(&self, prompt: &[u32]) {
        let mut g = self
            .engram_lookahead
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if g.as_ref().is_some_and(|la| la.prompt == prompt) {
            *g = None;
        }
    }

    pub(crate) fn has_prefill_engram(&self, prompt: &[u32], start: usize) -> bool {
        let g = self
            .engram_lookahead
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        g.as_ref().is_some_and(|la| {
            la.prompt.len() >= prompt.len()
                && la.prompt[..prompt.len()] == *prompt
                && la.chunks.get(la.cursor).is_some_and(|c| c.start == start)
        })
    }

    fn take_engram_chunk(
        &self,
        start: usize,
        ids: &[u32],
        history: &[u32],
    ) -> Option<(Vec<Vec<u32>>, Vec<Option<PrefetchHandle>>)> {
        let mut g = self
            .engram_lookahead
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let la = g.as_mut()?;
        let cur = la.chunks.get_mut(la.cursor)?;
        if cur.start != start
            || cur.end != start + ids.len()
            || la.prompt.len() < cur.end
            || la.prompt[cur.start..cur.end] != *ids
        {
            return None;
        }
        let back = self.cfg.engram.max_ngram.saturating_sub(1);
        let lo = start.saturating_sub(back);
        if history.len() < start || la.prompt[lo..start] != history[lo..start] {
            return None;
        }
        let layer_ids = std::mem::take(&mut cur.layer_ids);
        let handles = std::mem::take(&mut cur.handles);
        la.cursor += 1;
        Some((layer_ids, handles))
    }

    fn prefetch_next_engram_layer(&self, e: usize) {
        let next_req = {
            let g = self
                .engram_lookahead
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            g.as_ref().and_then(|la| {
                let next = la.chunks.get(la.cursor)?;
                if e < next.handles.len() && next.handles[e].is_none() {
                    Some((next.start, next.end, next.layer_ids[e].clone()))
                } else {
                    None
                }
            })
        };
        let Some((start, end, ids)) = next_req else {
            return;
        };
        if let Ok(Some(handle)) = self.table.prefetch(e, &ids) {
            let mut g = self
                .engram_lookahead
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            if let Some(la) = g.as_mut()
                && let Some(next) = la.chunks.get_mut(la.cursor)
                && next.start == start
                && next.end == end
                && e < next.handles.len()
                && next.handles[e].is_none()
            {
                next.handles[e] = Some(handle);
            }
        }
    }

    pub fn cfg(&self) -> &HyV4FlashConfig {
        &self.cfg
    }

    pub fn weights(&self) -> &dyn TrunkWeights {
        self.weights.as_ref()
    }

    pub fn blocks(&self) -> &[Block] {
        &self.blocks
    }

    pub fn bank_is_direct(&self) -> bool {
        !self.banks.is_empty() && self.banks.iter().all(|b| b.is_direct())
    }

    fn decode_prefetch_active(&self) -> bool {
        self.banks.iter().any(|b| {
            !b.is_direct() && b.prefetch() && b.decode_prefetch() && b.staging_layers() >= 2
        })
    }

    /// An empty sequence of at most `cap` positions (the admitted context);
    /// its indexed-layer buffers are made at the first write and kept. A new
    /// conversation reuses a session through `truncate(0)`: a second live
    /// session holds a second set of buffers the admission did not charge.
    pub fn new_session(&self, cap: usize) -> Result<Session> {
        if cap == 0 || cap > self.cfg.context_length {
            return Err(bad(format!(
                "context {cap} outside 1..={}",
                self.cfg.context_length
            )));
        }
        Ok(Session {
            tokens: Vec::new(),
            written: Vec::new(),
            calls: 0,
            cap,
            kpool: self.cfg.indexer.kpool,
            layers: self.blocks.iter().map(|b| b.new_state(cap)).collect(),
        })
    }

    /// Runs `ids` at the end of `sess` as one owner's `kind` rows; see
    /// `step`.
    pub fn forward(
        &self,
        sess: &mut Session,
        ids: &[u32],
        kind: RowKind,
        want: Want,
        cancel: Option<&dyn Fn(usize) -> bool>,
        trace: Option<&mut Trace>,
    ) -> Result<Output> {
        let mut owners = [OwnerRows { sess, ids, want }];
        self.step(&mut owners, kind, cancel, trace)?
            .pop()
            .ok_or_else(|| bad("no output"))
    }

    /// `forward` with the last row's f32 logits `[1, vocab]` only.
    pub fn forward_last(
        &self,
        sess: &mut Session,
        ids: &[u32],
        kind: RowKind,
        cancel: Option<&dyn Fn(usize) -> bool>,
    ) -> Result<MxArray> {
        self.forward(sess, ids, kind, Want::LAST, cancel, None)?
            .logits
            .ok_or_else(|| bad("no logits"))
    }

    /// Runs every owner's ids at the end of its session in one step of
    /// `kind`; a prefill step takes exactly one owner. `cancel(layer)` is
    /// asked before each layer; a cancelled or failed step leaves every
    /// session as it was. One output per owner, in order.
    pub fn step(
        &self,
        owners: &mut [OwnerRows<'_>],
        kind: RowKind,
        cancel: Option<&dyn Fn(usize) -> bool>,
        trace: Option<&mut Trace>,
    ) -> Result<Vec<Output>> {
        if owners.iter().any(|o| {
            o.sess.layers.len() != self.blocks.len() || o.sess.kpool != self.cfg.indexer.kpool
        }) {
            return Err(bad("a session belongs to another model"));
        }
        for o in owners.iter() {
            check_room(o)?;
        }
        let spans: Vec<(usize, usize)> = owners
            .iter()
            .map(|o| (o.sess.tokens.len(), o.ids.len()))
            .collect();
        let wave = Wave::new(kind, &spans, None)?;
        let marks: Vec<StepMark> = owners
            .iter()
            .map(|o| o.sess.step_mark(o.ids.len()))
            .collect();
        let out = self.wave(
            &mut FlatOwners {
                owners: &mut *owners,
            },
            &wave,
            cancel,
            trace,
        );
        match &out {
            Ok(_) => {
                for o in owners.iter_mut() {
                    let sess = &mut *o.sess;
                    sess.calls += 1;
                    sess.written
                        .extend(std::iter::repeat_n(sess.calls, o.ids.len()));
                    sess.tokens.extend_from_slice(o.ids);
                }
            }
            Err(e) => {
                let lost: Vec<String> = owners
                    .iter_mut()
                    .zip(&marks)
                    .filter_map(|(o, m)| o.sess.restore_step(m).err().map(|r| r.reason))
                    .collect();
                if !lost.is_empty() {
                    return Err(bad(format!(
                        "{}; sessions could not be restored: {}",
                        e.reason,
                        lost.join("; ")
                    )));
                }
            }
        }
        out
    }

    /// Runs the rows of `wave` layer by layer, owner `i`'s ids from
    /// `owners.ids(i)` at its span's start, its attention state through
    /// `owners.core`. `cancel(layer)` is asked before each layer. On error
    /// the caller puts every owner's state back; on success the caller
    /// records the owners' new tokens. One output per owner, in order.
    /// A decode wave commits each layer without waiting and syncs only at
    /// its router reads and its head; on return no GPU work of the wave
    /// is in flight and every bank step it planned is settled.
    pub fn wave(
        &self,
        owners: &mut dyn Owners,
        wave: &Wave,
        cancel: Option<&dyn Fn(usize) -> bool>,
        trace: Option<&mut Trace>,
    ) -> Result<Vec<Output>> {
        let scope = DeferScope::enter();
        let out = self.wave_layers(owners, wave, cancel, trace);
        if out.is_err() {
            crate::array::synchronize();
        }
        // the head eval waited for every gather (each one feeds the final
        // streams); a failed wave was drained above
        for b in &self.banks {
            b.settle_all();
        }
        drop(scope);
        out
    }

    fn wave_layers(
        &self,
        owners: &mut dyn Owners,
        wave: &Wave,
        cancel: Option<&dyn Fn(usize) -> bool>,
        mut trace: Option<&mut Trace>,
    ) -> Result<Vec<Output>> {
        self.check_wave(owners, wave)?;
        let total = wave.rows();
        let mut ids = Vec::with_capacity(total);
        let mut keep = Vec::with_capacity(total);
        let mut rows = Vec::new();
        let n_engram = self.hasher.n_layers();
        let mut handles: Vec<Option<PrefetchHandle>> = Vec::with_capacity(n_engram);
        let mut lookahead_layer_ids: Option<Vec<Vec<u32>>> = None;
        let tp = prof_start();
        for i in 0..wave.owners() {
            let call = owners.ids(i);
            keep.extend(keeps(call, self.cfg.engram.pad_token_id));
            ids.extend_from_slice(call);
        }
        let matched_lookahead = if wave.kind() == RowKind::Prefill && wave.owners() == 1 {
            let span = wave.spans()[0];
            if let Some((layer_ids, mut pre_handles)) =
                self.take_engram_chunk(span.start, owners.ids(0), owners.history(0))
            {
                for (e, lids) in layer_ids.iter().enumerate() {
                    let h = match pre_handles.get_mut(e).and_then(Option::take) {
                        Some(h) => Some(h),
                        None => self.table.prefetch(e, lids)?,
                    };
                    handles.push(h);
                }
                if trace.is_some() {
                    let n_cols = self.hasher.n_cols();
                    let n_pos = span.rows;
                    rows.reserve(n_pos * n_engram * n_cols);
                    for p in 0..n_pos {
                        for lids in &layer_ids {
                            let base = p * n_cols;
                            rows.extend_from_slice(&lids[base..base + n_cols]);
                        }
                    }
                }
                lookahead_layer_ids = Some(layer_ids);
                true
            } else {
                false
            }
        } else {
            false
        };
        if !matched_lookahead {
            for i in 0..wave.owners() {
                rows.extend(call_engram_rows(
                    &self.hasher,
                    self.cfg.engram.max_ngram - 1,
                    owners.history(i),
                    owners.ids(i),
                )?);
            }
            for e in 0..n_engram {
                handles.push(self.table.prefetch(e, &self.hasher.layer_ids(&rows, e)?)?);
            }
        }
        prof_end(Phase::Engram, tp, &[])?;
        if let Some(t) = trace.as_deref_mut() {
            t.engram_rows = rows.clone();
        }
        let w = self.weights.as_ref();
        let (ni, hc, dim) = (total as i64, self.cfg.hc.count as i64, self.cfg.dim as i64);
        let tok = MxArray::from_uint32(&ids, &[ni])?;
        let mut carry = HcCarry::Streams(
            w.token_embd()
                .rows(&tok)?
                .astype(DType::Float32)?
                .reshape(&[ni, 1, dim])?
                .broadcast_to(&[ni, hc, dim])?,
        );
        let direct = self.bank_is_direct();
        if direct && wave.kind() == RowKind::Decode {
            let mut from_prefill = false;
            let mut stale = false;
            for b in &self.banks {
                from_prefill |= b.take_prefill_ran();
                stale |= b.residency_stale(Duration::from_secs(1));
            }
            if from_prefill {
                crate::array::clear_cache();
            }
            if from_prefill || stale {
                for b in &self.banks {
                    b.keep_residency_alive();
                }
            }
        }
        let prefill_ahead = !direct
            && wave.kind() == RowKind::Prefill
            && total >= super::block::READ_AHEAD_MIN_ROWS;
        let decode_ahead = !direct
            && wave.kind().is_decode()
            && total <= super::block::DECODE_READ_AHEAD_MAX_ROWS
            && self.decode_prefetch_active();
        let keep_out = trace.as_deref().is_some_and(|t| t.keep_out);
        let mut pending_trace_out = false;
        let wave_pos_f32 = rope_positions_f32(&wave.positions())?;
        for (l, blk) in self.blocks.iter().enumerate() {
            if cancel.is_some_and(|c| c(l)) {
                return Err(bad(format!("cancelled before layer {l}")));
            }
            let table_rows = match blk.engram() {
                Some(e) => {
                    let tp = prof_start();
                    let r = match handles.get_mut(e).and_then(Option::take) {
                        Some(h) => self.table.take_prefetched(h)?,
                        None => match lookahead_layer_ids.as_ref().and_then(|v| v.get(e)) {
                            Some(lids) => self.table.rows(e, lids)?,
                            None => self.table.rows(e, &self.hasher.layer_ids(&rows, e)?)?,
                        },
                    };
                    prof_end(Phase::Engram, tp, &[])?;
                    if matched_lookahead {
                        self.prefetch_next_engram_layer(e);
                    }
                    Some(r)
                }
                None => None,
            };
            let engram = table_rows.as_deref().map(|r| EngramIn {
                rows: r,
                keep: &keep,
            });
            #[cfg(test)]
            if let Some(Some(xi)) = trace.as_deref().and_then(|t| t.inputs.get(l)) {
                let injected = xi.clone();
                if pending_trace_out {
                    let prev_x = carry.into_streams(self.cfg.norm_eps)?;
                    if let Some(t) = trace.as_deref_mut() {
                        t.out.push(prev_x);
                    }
                    pending_trace_out = false;
                }
                carry = HcCarry::Streams(injected);
            }
            let need_routes = trace.is_some();
            let next = ((prefill_ahead || decode_ahead) && l + 1 < self.blocks.len())
                .then(|| (&self.blocks[l + 1], &w.layers()[l + 1]));
            let mut prev_out_slot = None;
            let (next_carry, routes) = blk.forward_carry_with_pos(
                &w.layers()[l],
                carry,
                wave,
                Some(&wave_pos_f32),
                owners,
                engram,
                next,
                need_routes,
                pending_trace_out.then_some(&mut prev_out_slot),
            )?;
            if pending_trace_out {
                if let (Some(t), Some(prev_x)) = (trace.as_deref_mut(), prev_out_slot) {
                    t.out.push(prev_x);
                }
                pending_trace_out = false;
            }
            match wave.kind() {
                RowKind::Prefill if !direct => {
                    match &next_carry {
                        HcCarry::Streams(y) => super::stats::eval(&[y], "hy_v4_flash layer")?,
                        HcCarry::Pending { y, res, post, .. } => {
                            super::stats::eval(&[y, res, post], "hy_v4_flash layer")?
                        }
                    }
                    for b in &self.banks {
                        b.settle_all();
                    }
                }
                RowKind::Prefill | RowKind::Decode => {
                    // Flush at layer 0 in direct decode so GPU layer 0 overlaps
                    // layer 1's Engram SSD read, then every 10 layers. For large
                    // prefill chunks (> 2048 rows), flush every layer so peak
                    // Metal scratch stays bounded to one layer.
                    let flush = if !direct {
                        true
                    } else if wave.kind() == RowKind::Decode {
                        l == 0 || l == 10 || l == 20 || l == 30
                    } else if total > 2048 {
                        true
                    } else {
                        l >= 2 && (l - 2).is_multiple_of(3)
                    };
                    if flush {
                        match &next_carry {
                            HcCarry::Streams(y) => MxArray::async_eval_arrays(&[y]),
                            HcCarry::Pending { y, res, post, .. } => {
                                MxArray::async_eval_arrays(&[y, res, post])
                            }
                        }
                        if direct && wave.kind() == RowKind::Prefill {
                            super::stats::async_flush();
                        }
                    }
                }
            }
            carry = next_carry;
            if let Some(t) = trace.as_deref_mut() {
                t.routes.push(routes);
                if keep_out {
                    match &carry {
                        HcCarry::Streams(y) => t.out.push(y.clone()),
                        HcCarry::Pending { .. } => pending_trace_out = true,
                    }
                }
            }
        }
        if direct && wave.kind() == RowKind::Prefill {
            for b in &self.banks {
                b.mark_prefill_wave();
                b.keep_residency_alive();
            }
        }
        if pending_trace_out {
            let x_last = carry.into_streams(self.cfg.norm_eps)?;
            if let Some(t) = trace {
                t.out.push(x_last.clone());
            }
            carry = HcCarry::Streams(x_last);
        }
        let tp = prof_start();
        let outs = self.heads_carry(carry, owners, wave)?;
        prof_end(Phase::Head, tp, &[])?;
        if direct
            && wave.kind() == RowKind::Prefill
            && (0..wave.owners()).any(|i| owners.want(i).logits != Logits::None)
        {
            crate::array::clear_cache();
        }
        Ok(outs)
    }

    /// Refuses a wave whose owners do not match it: count, rows, starts
    /// (each owner's history is as long as its start) and vocab ids.
    fn check_wave(&self, owners: &dyn Owners, wave: &Wave) -> Result<()> {
        if owners.count() != wave.owners() {
            return Err(bad(format!(
                "{} owners for a wave of {}",
                owners.count(),
                wave.owners()
            )));
        }
        for (i, s) in wave.spans().iter().enumerate() {
            let (ids, history) = (owners.ids(i), owners.history(i));
            if ids.len() != s.rows || history.len() != s.start {
                return Err(bad(format!(
                    "owner {i}: {} ids after {} tokens for {} rows at {}",
                    ids.len(),
                    history.len(),
                    s.rows,
                    s.start
                )));
            }
            if let Some(id) = ids.iter().find(|&&id| id as usize >= self.cfg.vocab_size) {
                return Err(bad(format!(
                    "token id {id} outside the vocab of {}",
                    self.cfg.vocab_size
                )));
            }
        }
        Ok(())
    }

    /// `rms(hc_head(x))·output_norm` of the streams `x [r, hc, d]`.
    fn trunk_out(&self, x: &MxArray) -> Result<MxArray> {
        let w = self.weights.as_ref();
        let eps = self.cfg.norm_eps;
        hc_head_norm(x, w.output_hc(), w.output_norm(), eps, self.cfg.hc.eps)
    }

    /// `trunk_out` as `Op::TrunkOut` groups for `kind`.
    fn trunk_rows(&self, x: &MxArray, kind: RowKind) -> Result<MxArray> {
        match group(Op::TrunkOut, kind) {
            Group::Row => per_row(x, |r| self.trunk_out(r)),
            Group::Fused | Group::Owner => self.trunk_out(x),
        }
    }

    /// `rms(hc_head(x))·output_norm` directly from `carry` (fuses the final
    /// `post_norm + hc_post_identity + hc_head + output_norm` when `Pending`).
    fn trunk_out_carry(&self, carry: &HcCarry) -> Result<MxArray> {
        let w = self.weights.as_ref();
        let eps = self.cfg.norm_eps;
        match carry {
            HcCarry::Streams(x) => self.trunk_out(x),
            HcCarry::Pending {
                y,
                res,
                post,
                post_norm,
            } => hc_bridge_head(
                y,
                res,
                post,
                post_norm,
                w.output_hc(),
                Some(w.output_norm()),
                eps,
                self.cfg.hc.eps,
            ),
        }
    }

    /// Slices rows `lo..hi` of `carry` before applying `trunk_out_carry`.
    fn trunk_out_carry_slice(&self, carry: &HcCarry, lo: i64, hi: i64) -> Result<MxArray> {
        let w = self.weights.as_ref();
        let eps = self.cfg.norm_eps;
        match carry {
            HcCarry::Streams(x) => {
                let (hc, d) = (x.shape_at(1)?, x.shape_at(2)?);
                let sub = x.slice(&[lo, 0, 0], &[hi, hc, d])?;
                self.trunk_out(&sub)
            }
            HcCarry::Pending {
                y,
                res,
                post,
                post_norm,
            } => {
                let (hc, d) = (res.shape_at(1)?, res.shape_at(2)?);
                let ys = y.slice(&[lo, 0], &[hi, d])?;
                let rs = res.slice(&[lo, 0, 0], &[hi, hc, d])?;
                let ps = post.slice(&[lo, 0], &[hi, hc])?;
                hc_bridge_head(
                    &ys,
                    &rs,
                    &ps,
                    post_norm,
                    w.output_hc(),
                    Some(w.output_norm()),
                    eps,
                    self.cfg.hc.eps,
                )
            }
        }
    }

    /// Every owner's outputs from `carry`, evaluated together with the trunk
    /// output and what `stats::defer` holds.
    fn heads_carry(&self, carry: HcCarry, owners: &dyn Owners, wave: &Wave) -> Result<Vec<Output>> {
        let kind = wave.kind();
        let n_owners = wave.owners();
        let any_wanted = (0..n_owners).any(|i| {
            let want = owners.want(i);
            want.hidden || want.logits != Logits::None
        });
        if !any_wanted {
            let anchor = match &carry {
                HcCarry::Streams(x) => x,
                HcCarry::Pending { y, .. } => y,
            };
            super::stats::eval(&[anchor], "hy_v4_flash head")?;
            return Ok((0..n_owners)
                .map(|_| Output {
                    logits: None,
                    hidden: None,
                })
                .collect());
        }

        // Fast path for single-owner prefill needing only the final row's logits:
        // avoid computing hc_head + output_norm on the other S-1 rows.
        if n_owners == 1
            && !owners.want(0).hidden
            && owners.want(0).logits == Logits::Last
            && wave.rows() > 1
        {
            let last = (wave.rows() - 1) as i64;
            let h = self.trunk_out_carry_slice(&carry, last, last + 1)?;
            let mut outs = vec![Output {
                logits: None,
                hidden: None,
            }];
            let anchor = match &carry {
                HcCarry::Streams(x) => x,
                HcCarry::Pending { y, .. } => y,
            };
            self.eval_head_logits(anchor, &mut outs, vec![h], &[(0, 0, 1)], owners, wave)?;
            return Ok(outs);
        }

        let all_trunk = match group(Op::TrunkOut, kind) {
            Group::Row => {
                let x = carry.into_streams(self.cfg.norm_eps)?;
                let h = self.trunk_rows(&x, kind)?;
                return self.heads_from_trunk(&x, &h, owners, wave);
            }
            Group::Fused | Group::Owner => self.trunk_out_carry(&carry)?,
        };

        let mut outs = Vec::with_capacity(n_owners);
        let mut logit_hs: Vec<MxArray> = Vec::new();
        let mut logit_spans: Vec<(usize, i64, i64)> = Vec::new();
        let mut row_off = 0i64;
        for i in 0..n_owners {
            let want = owners.want(i);
            let span = wave.spans()[i];
            let hidden = if want.hidden {
                Some(wave.owner_rows(&all_trunk, i)?)
            } else {
                None
            };
            let h = match want.logits {
                Logits::None => None,
                Logits::All => Some(match &hidden {
                    Some(h) => h.clone(),
                    None => wave.owner_rows(&all_trunk, i)?,
                }),
                Logits::Last => {
                    let n = span.rows as i64;
                    let h = if let Some(h) = &hidden {
                        if n == 1 {
                            h.clone()
                        } else if kind == RowKind::Decode {
                            h.slice(&[n - 1, 0], &[n, h.shape_at(1)?])?
                        } else {
                            let last = (span.lo + span.rows - 1) as i64;
                            self.trunk_out_carry_slice(&carry, last, last + 1)?
                        }
                    } else if kind == RowKind::Decode || n == 1 {
                        let ho = wave.owner_rows(&all_trunk, i)?;
                        if n == 1 {
                            ho
                        } else {
                            ho.slice(&[n - 1, 0], &[n, ho.shape_at(1)?])?
                        }
                    } else {
                        let last = (span.lo + span.rows - 1) as i64;
                        self.trunk_out_carry_slice(&carry, last, last + 1)?
                    };
                    Some(h)
                }
            };
            if let Some(h) = h {
                let rows = h.shape_at(0)?;
                logit_spans.push((i, row_off, rows));
                row_off += rows;
                logit_hs.push(h);
            }
            outs.push(Output {
                logits: None,
                hidden,
            });
        }
        self.eval_head_logits(
            match &carry {
                HcCarry::Streams(x) => x,
                HcCarry::Pending { y, .. } => y,
            },
            &mut outs,
            logit_hs,
            &logit_spans,
            owners,
            wave,
        )?;
        Ok(outs)
    }

    fn heads_from_trunk(
        &self,
        anchor: &MxArray,
        all_trunk: &MxArray,
        owners: &dyn Owners,
        wave: &Wave,
    ) -> Result<Vec<Output>> {
        let n_owners = wave.owners();
        let mut outs = Vec::with_capacity(n_owners);
        let mut logit_hs: Vec<MxArray> = Vec::new();
        let mut logit_spans: Vec<(usize, i64, i64)> = Vec::new();
        let mut row_off = 0i64;
        for i in 0..n_owners {
            let want = owners.want(i);
            let ho = wave.owner_rows(all_trunk, i)?;
            let hidden = want.hidden.then(|| ho.clone());
            let h = match want.logits {
                Logits::None => None,
                Logits::All => Some(ho),
                Logits::Last => {
                    let n = ho.shape_at(0)?;
                    Some(if n == 1 {
                        ho
                    } else {
                        ho.slice(&[n - 1, 0], &[n, ho.shape_at(1)?])?
                    })
                }
            };
            if let Some(h) = h {
                let rows = h.shape_at(0)?;
                logit_spans.push((i, row_off, rows));
                row_off += rows;
                logit_hs.push(h);
            }
            outs.push(Output {
                logits: None,
                hidden,
            });
        }
        self.eval_head_logits(anchor, &mut outs, logit_hs, &logit_spans, owners, wave)?;
        Ok(outs)
    }

    fn eval_head_logits(
        &self,
        anchor: &MxArray,
        outs: &mut [Output],
        mut logit_hs: Vec<MxArray>,
        logit_spans: &[(usize, i64, i64)],
        owners: &dyn Owners,
        wave: &Wave,
    ) -> Result<()> {
        let kind = wave.kind();
        let n_owners = wave.owners();
        if !logit_hs.is_empty() {
            let head = self.weights.output();
            let all_h = if logit_hs.len() == 1 {
                logit_hs.swap_remove(0)
            } else {
                MxArray::concatenate_many(logit_hs.iter().collect(), Some(0))?
            };
            let all_logits = if group(Op::VocabHead, kind) != Group::Row
                && kind == RowKind::Prefill
                && n_owners == 1
                && owners.want(0).logits == Logits::All
            {
                head_logits(&all_h, head)?
            } else {
                head_rows(&all_h, head)?
            };
            if logit_spans.len() == 1 {
                outs[logit_spans[0].0].logits = Some(all_logits);
            } else {
                let vocab = all_logits.shape_at(1)?;
                for &(owner, lo, rows) in logit_spans {
                    outs[owner].logits = Some(all_logits.slice(&[lo, 0], &[lo + rows, vocab])?);
                }
            }
        }
        let live: Vec<&MxArray> = std::iter::once(anchor)
            .chain(
                outs.iter()
                    .flat_map(|o| o.logits.iter().chain(o.hidden.iter())),
            )
            .collect();
        super::stats::eval(&live, "hy_v4_flash head")?;
        Ok(())
    }
}
