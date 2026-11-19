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

use napi::{Error, Result};

use super::batch::{Core, Group, Op, Owners, Wave, group};
use super::block::{Block, EngramIn, LayerState};
use super::cache::{LayerCache, WindowRing};
use super::config::HyV4FlashConfig;
use super::engram::{Hasher, keeps};
use super::indexer::{IndexerMark, IndexerState};
use super::moe::{ExpertSource, Routes, RowKind, per_row, qlinear};
use super::ops::{HEAD_CHUNK, hc_head, head_logits};
use super::prefix_state::StateLayer;
use super::rows::EngramRows;
use super::stats::{Phase, prof_end, prof_start};
use super::weights::{HcWeights, LayerWeights, Resident};
use crate::array::{DType, MxArray};
use crate::models::deepseek_v41::attention::rms_norm;
use crate::models::deepseek_v41::store::{Dense, Weight};

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
    if let Dense::Qmm { .. } = head {
        return qlinear(&xf, head);
    }
    if xf.shape_at(0)? == 1 {
        return head_logits(&xf, head);
    }
    let vocab = head.shape_at(0)?;
    let mut parts = Vec::new();
    let mut lo = 0;
    while lo < vocab {
        let hi = (lo + HEAD_CHUNK).min(vocab);
        let wt = head.row_range(lo, hi)?.f32()?.transpose(Some(&[1, 0]))?;
        let p = per_row(&xf, |r| r.matmul(&wt))?;
        super::stats::eval(&[&p], "hy_v4_flash head chunk")?;
        parts.push(p);
        lo = hi;
    }
    match parts.len() {
        0 => Err(bad("head has no vocab rows")),
        1 => Ok(parts.swap_remove(0)),
        _ => MxArray::concatenate_many(parts.iter().collect(), Some(1)),
    }
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

/// Whether `IndexerState::truncate(len)` keeps the open `tok_k` rows it
/// needs (the refusal rule of `indexer.rs` `truncate`).
fn index_keeps(st: &IndexerState, len: usize, kpool: usize) -> bool {
    let closed = len / kpool * kpool;
    let first = st.len() - st.held();
    len <= closed || closed >= first
}

#[derive(Clone)]
enum LayerMark {
    Window(WindowRing),
    Latent {
        len: usize,
        index: Option<IndexerMark>,
    },
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
                LayerMark::Window(r) => Ok(StateLayer::Window(r.clone())),
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
        Mark {
            len: self.tokens.len(),
            last: self.written.last().copied(),
            layers: self
                .layers
                .iter()
                .map(|st| match &st.cache {
                    LayerCache::Window(r) => LayerMark::Window(r.clone()),
                    LayerCache::Latent(b) => LayerMark::Latent {
                        len: b.len(),
                        index: st.index.as_ref().map(IndexerState::mark),
                    },
                })
                .collect(),
        }
    }

    /// Goes back to `m` (taken from this session at a length <= now).
    /// Refused, with nothing changed, when a position below `m`'s length was
    /// rewritten since `m` was taken.
    pub fn restore(&mut self, m: &Mark) -> Result<()> {
        self.fits(m)?;
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
        for (st, lm) in self.layers.iter_mut().zip(&m.layers) {
            match (&mut st.cache, lm) {
                (LayerCache::Window(r), LayerMark::Window(saved)) => *r = saved.clone(),
                (LayerCache::Latent(b), LayerMark::Latent { len, .. }) => b.truncate(*len)?,
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
    /// trunk order) its latent rows `latent[i] [b, width]`, its pooled keys
    /// `pooled[i] [b / kpool, d]` and its tail. On error the session stays
    /// empty.
    pub fn fill(
        &mut self,
        tokens: &[u32],
        layers: &[StateLayer],
        latent: &[MxArray],
        pooled: &[MxArray],
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
        let out = self.fill_layers(b, layers, latent, pooled);
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
        latent: &[MxArray],
        pooled: &[MxArray],
    ) -> Result<()> {
        let mut at = 0;
        for (l, (st, part)) in self.layers.iter_mut().zip(layers).enumerate() {
            match (&mut st.cache, st.index.as_mut(), part) {
                (LayerCache::Window(r), None, StateLayer::Window(s)) if s.len() == b => {
                    *r = s.clone();
                }
                (LayerCache::Latent(buf), Some(ix), StateLayer::Tail(t)) if t.len() == b => {
                    let (rows, keys) = match (latent.get(at), pooled.get(at)) {
                        (Some(r), Some(k)) => (r, k),
                        _ => return Err(bad(format!("layer {l}: no latent rows to fill"))),
                    };
                    at += 1;
                    if rows.shape_at(0)? as usize != b {
                        return Err(bad(format!("layer {l}: latent rows for another row")));
                    }
                    buf.write(rows, 0)?;
                    let keys = (keys.shape_at(0)? > 0).then_some(keys);
                    ix.fill(keys, t.clone())?;
                }
                _ => return Err(bad(format!("layer {l}: the state is not of this model"))),
            }
        }
        if at != latent.len() || at != pooled.len() {
            return Err(bad(format!(
                "{} latent row sets for {at} indexed layers",
                latent.len()
            )));
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
                (LayerCache::Window(_), LayerMark::Window(_)) => st.index.is_none(),
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
        let rings: Vec<Option<WindowRing>> = self
            .layers
            .iter()
            .map(|st| match &st.cache {
                LayerCache::Window(r) => Some(r.clone()),
                LayerCache::Latent(_) => None,
            })
            .collect();
        let windows = self
            .layers
            .iter_mut()
            .try_for_each(|st| match &mut st.cache {
                LayerCache::Window(r) => r.truncate(len),
                LayerCache::Latent(_) => Ok(()),
            });
        if let Err(e) = windows {
            for (st, saved) in self.layers.iter_mut().zip(rings) {
                if let (LayerCache::Window(r), Some(s)) = (&mut st.cache, saved) {
                    *r = s;
                }
            }
            return Err(e);
        }
        for st in &mut self.layers {
            if let Some(ix) = st.index.as_mut() {
                ix.truncate(len)?;
            }
            if let LayerCache::Latent(b) = &mut st.cache {
                b.truncate(len)?;
            }
        }
        self.tokens.truncate(len);
        self.written.truncate(len);
        Ok(())
    }
}

pub struct Model {
    cfg: HyV4FlashConfig,
    weights: Box<dyn TrunkWeights>,
    blocks: Vec<Block>,
    hasher: Hasher,
    table: Box<dyn EngramRows>,
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
        let blocks = layers
            .iter()
            .enumerate()
            .map(|(l, w)| {
                let src = if cfg.layers[l].dense_ffn {
                    None
                } else {
                    Some(experts(l)?)
                };
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
        })
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
        let marks: Vec<Mark> = owners.iter().map(|o| o.sess.mark()).collect();
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
                    .filter_map(|(o, m)| o.sess.restore(m).err().map(|r| r.reason))
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
    pub fn wave(
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
        let tp = prof_start();
        for i in 0..wave.owners() {
            let call = owners.ids(i);
            rows.extend(call_engram_rows(
                &self.hasher,
                self.cfg.engram.max_ngram - 1,
                owners.history(i),
                call,
            )?);
            keep.extend(keeps(call, self.cfg.engram.pad_token_id));
            ids.extend_from_slice(call);
        }
        prof_end(Phase::Engram, tp, &[])?;
        if let Some(t) = trace.as_deref_mut() {
            t.engram_rows = rows.clone();
        }
        let w = self.weights.as_ref();
        let (ni, hc, dim) = (total as i64, self.cfg.hc.count as i64, self.cfg.dim as i64);
        let tok = MxArray::from_uint32(&ids, &[ni])?;
        let mut x = w
            .token_embd()
            .rows(&tok)?
            .astype(DType::Float32)?
            .reshape(&[ni, 1, dim])?
            .broadcast_to(&[ni, hc, dim])?;
        for (l, blk) in self.blocks.iter().enumerate() {
            if cancel.is_some_and(|c| c(l)) {
                return Err(bad(format!("cancelled before layer {l}")));
            }
            let table_rows = match blk.engram() {
                Some(e) => {
                    let tp = prof_start();
                    let r = self.table.rows(e, &self.hasher.layer_ids(&rows, e)?)?;
                    prof_end(Phase::Engram, tp, &[])?;
                    Some(r)
                }
                None => None,
            };
            let engram = table_rows.as_deref().map(|r| EngramIn {
                rows: r,
                keep: &keep,
            });
            let (y, routes) = blk.forward(&w.layers()[l], &x, wave, owners, engram)?;
            super::stats::eval(&[&y], "hy_v4_flash layer")?;
            x = y;
            if let Some(t) = trace.as_deref_mut() {
                t.routes.push(routes);
                if t.keep_out {
                    t.out.push(x.clone());
                }
            }
        }
        let tp = prof_start();
        let outs = self.heads(&x, owners, wave)?;
        prof_end(Phase::Head, tp, &[])?;
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
        rms_norm(
            &hc_head(x, w.output_hc(), eps, self.cfg.hc.eps)?,
            w.output_norm(),
            eps,
        )
    }

    /// `trunk_out` as `Op::TrunkOut` groups for `kind`.
    fn trunk_rows(&self, x: &MxArray, kind: RowKind) -> Result<MxArray> {
        match group(Op::TrunkOut, kind) {
            Group::Row => per_row(x, |r| self.trunk_out(r)),
            Group::Fused | Group::Owner => self.trunk_out(x),
        }
    }

    /// Every owner's outputs from its rows of `x [r, hc, d]`, evaluated
    /// together.
    fn heads(&self, x: &MxArray, owners: &dyn Owners, wave: &Wave) -> Result<Vec<Output>> {
        let mut outs = Vec::with_capacity(wave.owners());
        for i in 0..wave.owners() {
            let xo = wave.owner_rows(x, i)?;
            outs.push(self.head(&xo, owners.want(i), wave.kind())?);
        }
        let live: Vec<&MxArray> = outs
            .iter()
            .flat_map(|o| o.logits.iter().chain(o.hidden.iter()))
            .collect();
        if !live.is_empty() {
            super::stats::eval(&live, "hy_v4_flash head")?;
        }
        Ok(outs)
    }

    /// One owner's outputs. Last-row logits come from the last row's
    /// streams alone, so they do not depend on `want.hidden` or `kind`.
    fn head(&self, x: &MxArray, want: Want, kind: RowKind) -> Result<Output> {
        let head = self.weights.output();
        let hidden = if want.hidden {
            Some(self.trunk_rows(x, kind)?)
        } else {
            None
        };
        let logits = match want.logits {
            Logits::None => None,
            Logits::All => {
                let h = match &hidden {
                    Some(h) => h.clone(),
                    None => self.trunk_rows(x, kind)?,
                };
                Some(match group(Op::VocabHead, kind) {
                    Group::Row => head_rows(&h, head)?,
                    Group::Fused | Group::Owner => head_logits(&h, head)?,
                })
            }
            Logits::Last => {
                let n = x.shape_at(0)?;
                let (hc, d) = (x.shape_at(1)?, x.shape_at(2)?);
                let last = x.slice(&[n - 1, 0, 0], &[n, hc, d])?;
                Some(head_rows(&self.trunk_out(&last)?, head)?)
            }
        };
        Ok(Output { logits, hidden })
    }
}
