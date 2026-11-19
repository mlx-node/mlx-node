//! The Hy4 trunk (llama.cpp PR #32517 graph), f32 activations:
//!
//! ```text
//! ids ─► token_embd rows ─► hc copies ─► 42 × Block ─► hc_head ─► rms·output_norm = h_trunk ─► output ─► logits
//! ```
//!
//! One call runs layer by layer over all its rows. A `Session` is one
//! sequence; its latent and pooled buffers are written in place, so a
//! session is never cloned: snapshots are lengths plus window clones
//! (`Mark`), and a failed or cancelled call goes back to its start.

use napi::{Error, Result};

use super::block::{Block, LayerState};
use super::cache::{LayerCache, WindowRing};
use super::config::HyV4FlashConfig;
use super::engram::Hasher;
use super::indexer::{IndexerMark, IndexerState};
use super::moe::{ExpertSource, Routes};
use super::ops::{hc_head, head_logits};
use super::rows::EngramRows;
use super::stats::{Phase, prof_end, prof_start};
use super::weights::{HcWeights, LayerWeights, Resident};
use crate::array::{DType, MxArray};
use crate::models::deepseek_v41::attention::rms_norm;
use crate::models::deepseek_v41::store::Dense;

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

/// Per-layer debug dump of one call.
#[derive(Default)]
pub struct Trace {
    /// Engram table rows `[pos][engram layer][col]` (global ids).
    pub engram_rows: Vec<u32>,
    /// One entry per layer (`None` on the dense one).
    pub routes: Vec<Option<Routes>>,
    /// Keep each layer's output streams `[n, hc, d]` in `out`.
    pub keep_out: bool,
    pub out: Vec<MxArray>,
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

    /// Runs `ids` at the end of `sess`. `cancel(layer)` is asked before
    /// each layer; a cancelled or failed call leaves `sess` as it was.
    pub fn forward(
        &self,
        sess: &mut Session,
        ids: &[u32],
        want: Want,
        cancel: Option<&dyn Fn(usize) -> bool>,
        trace: Option<&mut Trace>,
    ) -> Result<Output> {
        if sess.layers.len() != self.blocks.len() || sess.kpool != self.cfg.indexer.kpool {
            return Err(bad("the session belongs to another model"));
        }
        let mark = sess.mark();
        let out = self.run(sess, ids, want, cancel, trace);
        if let Err(e) = &out
            && let Err(r) = sess.restore(&mark)
        {
            return Err(bad(format!(
                "{}; the session could not be restored: {}",
                e.reason, r.reason
            )));
        }
        out
    }

    /// `forward` with the last row's f32 logits `[1, vocab]` only.
    pub fn forward_last(
        &self,
        sess: &mut Session,
        ids: &[u32],
        cancel: Option<&dyn Fn(usize) -> bool>,
    ) -> Result<MxArray> {
        self.forward(sess, ids, Want::LAST, cancel, None)?
            .logits
            .ok_or_else(|| bad("no logits"))
    }

    fn run(
        &self,
        sess: &mut Session,
        ids: &[u32],
        want: Want,
        cancel: Option<&dyn Fn(usize) -> bool>,
        mut trace: Option<&mut Trace>,
    ) -> Result<Output> {
        let (n, start) = (ids.len(), sess.tokens.len());
        if n == 0 {
            return Err(bad("empty call"));
        }
        if start + n > sess.cap {
            return Err(bad(format!(
                "{start} + {n} tokens exceed the context of {}",
                sess.cap
            )));
        }
        if let Some(id) = ids.iter().find(|&&id| id as usize >= self.cfg.vocab_size) {
            return Err(bad(format!(
                "token id {id} outside the vocab of {}",
                self.cfg.vocab_size
            )));
        }
        let tp = prof_start();
        let rows = call_engram_rows(
            &self.hasher,
            self.cfg.engram.max_ngram - 1,
            &sess.tokens,
            ids,
        )?;
        prof_end(Phase::Engram, tp, &[])?;
        if let Some(t) = trace.as_deref_mut() {
            t.engram_rows = rows.clone();
        }
        let w = self.weights.as_ref();
        let (ni, hc, dim) = (n as i64, self.cfg.hc.count as i64, self.cfg.dim as i64);
        let tok = MxArray::from_uint32(ids, &[ni])?;
        let mut x = w
            .token_embd()
            .rows(&tok)?
            .astype(DType::Float32)?
            .reshape(&[ni, 1, dim])?
            .broadcast_to(&[ni, hc, dim])?;
        for (l, (blk, st)) in self.blocks.iter().zip(sess.layers.iter_mut()).enumerate() {
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
            let (y, routes) = blk.forward(&w.layers()[l], &x, start, st, table_rows.as_deref())?;
            MxArray::eval_arrays_with_context(&[&y], "hy_v4_flash layer")?;
            x = y;
            if let Some(t) = trace.as_deref_mut() {
                t.routes.push(routes);
                if t.keep_out {
                    t.out.push(x.clone());
                }
            }
        }
        let tp = prof_start();
        let out = self.head(&x, want)?;
        prof_end(Phase::Head, tp, &[])?;
        sess.calls += 1;
        sess.written.extend(std::iter::repeat_n(sess.calls, n));
        sess.tokens.extend_from_slice(ids);
        Ok(out)
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

    /// Last-row logits come from the last row's streams alone, so they do
    /// not depend on `want.hidden`.
    fn head(&self, x: &MxArray, want: Want) -> Result<Output> {
        let head = self.weights.output();
        let hidden = if want.hidden {
            Some(self.trunk_out(x)?)
        } else {
            None
        };
        let logits = match want.logits {
            Logits::None => None,
            Logits::All => {
                let h = match &hidden {
                    Some(h) => h.clone(),
                    None => self.trunk_out(x)?,
                };
                Some(head_logits(&h, head)?)
            }
            Logits::Last => {
                let n = x.shape_at(0)?;
                let (hc, d) = (x.shape_at(1)?, x.shape_at(2)?);
                let last = x.slice(&[n - 1, 0, 0], &[n, hc, d])?;
                Some(head_logits(&self.trunk_out(&last)?, head)?)
            }
        };
        let live: Vec<&MxArray> = logits.iter().chain(hidden.iter()).collect();
        if !live.is_empty() {
            MxArray::eval_arrays_with_context(&live, "hy_v4_flash head")?;
        }
        Ok(Output { logits, hidden })
    }
}
