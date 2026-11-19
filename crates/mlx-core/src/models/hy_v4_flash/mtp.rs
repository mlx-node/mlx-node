//! The MTP head `blk.{n_layers}` (llama.cpp PR #32517 `graph_mtp`) and its
//! per-sequence state, f32 math:
//!
//! ```text
//! x = eh_proj · [ rms(e)·enorm ; rms(h)·hnorm ]     e = token_embd[t(p+1)]   h = h_trunk(p)
//! x = x + attn(x)                                   one indexed layer, own K/V rows and indexer
//! x = x + ffn(x)                                    latent MoE, resident affine-4 experts
//! h_mtp = rms(x)·shared_head_norm  ──► logits = output · h_mtp   (h_mtp feeds the next draft)
//! ```
//!
//! MTP row `p` pairs `h_trunk(p)` with token `p + 1` (llama's pair
//! `(h_p, x_{p+1})`), so it lags the trunk by one row: a trunk call of rows
//! `[s, e)` writes MTP rows `s - 1 .. e - 1` from the carried seed
//! `h_trunk(s - 1)` and its own rows. Rows, groups and indexer positions
//! count from MTP row 0; the attention RoPE runs at `p + 1`, the position
//! llama gives the row (scores depend on differences only).
//!
//! The K/V rows live in one buffer that grows with them (`KvRows`), never
//! one sized for the whole context up front. A settled sequence (`Lagged`)
//! holds `trunk - 1` MTP rows and the seed `h_trunk(trunk - 1)`.
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

use std::collections::VecDeque;
use std::path::PathBuf;

use napi::{Error, Result};

use super::attention::{Geometry, Mla};
use super::cache::ROW_DTYPE;
use super::indexer::{Call, Indexer, IndexerGeometry, IndexerMark, IndexerState};
use super::ladder::Snapshot;
use super::model::{Mark, Model, Output, Session, Want, head_rows};
use super::moe::{ExpertSource, FfnConfig, LayerFfn, RowKind, qlinear};
use super::mtp_weights::{self, MtpWeights};
use super::prefill::{Runner, Seq, index_keeps, session_keeps};
use super::scan::Scan;
use super::sparse::gather_sink_attention;
use super::weights::{LayerWeights, NextnWeights, Residual};
use crate::array::{DType, MxArray};
use crate::cache_limit::CacheLimitGuard;
use crate::models::deepseek_v41::attention::rms_norm;
use crate::models::deepseek_v41::ops::{apply_partial_rope, inv_freqs, rope_tables};
use crate::models::deepseek_v41::store::Dense;
use crate::moe_stream::io::ShardSet;
use crate::nn::Embedding;

/// The K/V buffer's rows are a multiple of this.
const BLOCK_ROWS: usize = 32;
/// Trunk output rows kept as seeds: a rollback of up to `WINDOW_SLACK` rows
/// still finds the seed of the row it lands on.
pub const SEEDS: usize = 8;
/// MTP rows a draft may hold past the trunk.
pub const DRAFT_SLACK: usize = 4;

fn bad(msg: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("Hy4 MTP: {msg}"))
}

/// The MTP rows a trunk call of `n` rows at `start` writes: `rows` rows from
/// row `at`, the seed `h_trunk(start - 1)` first when `seed`; the call's
/// own rows `0..n - 1` follow, paired with its last `rows` token ids.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Lag {
    pub at: usize,
    pub seed: bool,
    pub rows: usize,
}

pub fn lag(start: usize, n: usize) -> Lag {
    match start.checked_sub(1) {
        Some(at) => Lag {
            at,
            seed: true,
            rows: n,
        },
        None => Lag {
            at: 0,
            seed: false,
            rows: n.saturating_sub(1),
        },
    }
}

fn round_up(n: usize) -> usize {
    n.div_ceil(BLOCK_ROWS) * BLOCK_ROWS
}

fn rows_of(x: &MxArray, width: i64) -> Result<usize> {
    match x.shape()?.as_ref() {
        &[n, w] if w == width && n > 0 => Ok(n as usize),
        s => Err(bad(format!("rows {s:?}, need [n > 0, {width}]"))),
    }
}

/// f16 K/V rows in one buffer written in place; rows past `len` are dead.
/// A write past the buffer moves the rows to one of twice the rows (a
/// multiple of `BLOCK_ROWS`, at most `cap`); a cut that leaves a quarter or
/// less live moves them to one of twice the live rows. The array must have
/// exactly one handle (this struct): a second one turns every in-place
/// write into a copy of the whole buffer.
pub struct KvRows {
    width: i64,
    cap: usize,
    buf: Option<MxArray>,
    held: usize,
    len: usize,
}

impl KvRows {
    pub fn new(width: i64, cap: usize) -> Self {
        Self {
            width,
            cap,
            buf: None,
            held: 0,
            len: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Rows the buffer holds, live or dead.
    pub fn capacity(&self) -> usize {
        self.held
    }

    /// The buffer `[capacity, width]`; rows `0..len` are live. A caller must
    /// not keep the handle (or a view of it) past its call.
    pub fn rows(&self) -> Result<&MxArray> {
        self.buf
            .as_ref()
            .ok_or_else(|| bad("K/V read before the first write"))
    }

    /// Moves the live rows to a buffer of `rows` rows (`>= len`).
    fn resize(&mut self, rows: usize) -> Result<()> {
        let r = i64::try_from(rows).map_err(|_| bad("K/V rows exceed i64"))?;
        let mut b = MxArray::zeros(&[r, self.width], Some(ROW_DTYPE))?;
        if self.len > 0
            && let Some(old) = &self.buf
        {
            let live = old.slice(&[0, 0], &[self.len as i64, self.width])?;
            b.slice_assign_axis_inplace(0, 0, self.len as i64, &live)?;
        }
        MxArray::eval_arrays_with_context(&[&b], "hy_v4_flash MTP K/V resize")?;
        self.buf = Some(b);
        self.held = rows;
        Ok(())
    }

    /// Writes `kv [n, width]` (stored as f16) at `start` (== `len()`).
    pub fn write(&mut self, kv: &MxArray, start: usize) -> Result<()> {
        if start != self.len {
            return Err(bad(format!("K/V write at {start}, {} rows held", self.len)));
        }
        let n = rows_of(kv, self.width)?;
        let end = start + n;
        if end > self.cap {
            return Err(bad(format!(
                "K/V write to {end} past the {} rows a draft holds",
                self.cap
            )));
        }
        let kv = kv.astype(ROW_DTYPE)?;
        if end > self.held {
            self.resize(round_up(end).max(2 * self.held).min(self.cap))?;
        }
        let buf = self.buf.as_mut().ok_or_else(|| bad("K/V buffer missing"))?;
        buf.slice_assign_axis_inplace(0, start as i64, end as i64, &kv)?;
        MxArray::eval_arrays_with_context(&[&*buf], "hy_v4_flash MTP K/V write")?;
        self.len = end;
        Ok(())
    }

    /// Forgets the rows past `len`; a cut to 0 frees the buffer.
    pub fn truncate(&mut self, len: usize) -> Result<()> {
        if len > self.len {
            return Err(bad(format!(
                "K/V truncate to {len}, {} rows held",
                self.len
            )));
        }
        self.len = len;
        if len == 0 {
            self.buf = None;
            self.held = 0;
            return Ok(());
        }
        let keep = round_up(len);
        if 4 * keep <= self.held {
            self.resize((2 * keep).min(self.cap))?;
        }
        Ok(())
    }

    pub fn nbytes(&self) -> usize {
        self.buf.as_ref().map_or(0, MxArray::nbytes)
    }
}

/// The last `SEEDS` trunk output rows by position, copies of their own.
#[derive(Default)]
pub struct Seeds {
    rows: VecDeque<(usize, MxArray)>,
}

impl Seeds {
    /// Copies of the last rows of `h [n, d]` (positions `start..start + n`).
    pub fn copies(h: &MxArray, start: usize) -> Result<Vec<(usize, MxArray)>> {
        let (n, d) = (h.shape_at(0)? as usize, h.shape_at(1)?);
        let k = n.min(SEEDS);
        let mut out = Vec::with_capacity(k);
        for r in n - k..n {
            let row = h
                .slice(&[r as i64, 0], &[r as i64 + 1, d])?
                .astype(DType::Float32)?
                .deep_copy()?;
            out.push((start + r, row));
        }
        let live: Vec<&MxArray> = out.iter().map(|(_, a)| a).collect();
        MxArray::eval_arrays_with_context(&live, "hy_v4_flash MTP seed")?;
        Ok(out)
    }

    /// Adds `rows` (ascending positions) in place of any at or past the first.
    pub fn extend(&mut self, rows: Vec<(usize, MxArray)>) {
        if let Some(&(first, _)) = rows.first() {
            self.cut(first);
        }
        self.rows.extend(rows);
        while self.rows.len() > SEEDS {
            self.rows.pop_front();
        }
    }

    pub fn get(&self, pos: usize) -> Option<&MxArray> {
        self.rows.iter().find(|(p, _)| *p == pos).map(|(_, a)| a)
    }

    /// Drops the rows at or past `len`.
    pub fn cut(&mut self, len: usize) {
        self.rows.retain(|(p, _)| *p < len);
    }

    pub fn clear(&mut self) {
        self.rows.clear();
    }

    pub fn nbytes(&self) -> usize {
        self.rows.iter().map(|(_, a)| a.nbytes()).sum()
    }
}

/// One sequence's MTP state: its K/V rows, its indexer and the seeds.
pub struct Draft {
    kv: KvRows,
    index: IndexerState,
    seeds: Seeds,
    kpool: usize,
    cap: usize,
}

/// A settled draft to go back to: its rows, indexer and seed.
#[derive(Clone)]
pub struct DraftMark {
    len: usize,
    index: IndexerMark,
    seed: Option<(usize, MxArray)>,
}

impl Draft {
    /// MTP rows held.
    pub fn len(&self) -> usize {
        self.kv.len()
    }

    pub fn is_empty(&self) -> bool {
        self.kv.is_empty()
    }

    pub fn cap(&self) -> usize {
        self.cap
    }

    pub fn kv(&self) -> &KvRows {
        &self.kv
    }

    pub fn seed(&self, pos: usize) -> Option<&MxArray> {
        self.seeds.get(pos)
    }

    pub fn seeds_mut(&mut self) -> &mut Seeds {
        &mut self.seeds
    }

    /// Whether `cut(len)` would succeed.
    pub fn can_cut(&self, len: usize) -> bool {
        len <= self.len() && index_keeps(self.index.len(), self.index.held(), self.kpool, len)
    }

    /// Drops the rows past `len`; refused with nothing changed.
    pub fn cut(&mut self, len: usize) -> Result<()> {
        if len > self.len() {
            return Err(bad(format!("cut to {len}, {} rows held", self.len())));
        }
        self.index.truncate(len)?;
        self.kv.truncate(len)
    }

    fn mark(&self, trunk: usize) -> Option<DraftMark> {
        let len = trunk.checked_sub(1);
        let seed = match len {
            Some(at) => Some((at, self.seeds.get(at)?.clone())),
            None => None,
        };
        (self.len() == len.unwrap_or(0)).then(|| DraftMark {
            len: self.len(),
            index: self.index.mark(),
            seed,
        })
    }

    fn can_restore(&self, m: &DraftMark) -> bool {
        m.len <= self.len()
            && m.index.len() == m.len
            && m.len / self.kpool.max(1) <= self.index.pooled_len()
    }

    fn restore(&mut self, m: &DraftMark) -> Result<()> {
        self.index.restore(m.index.clone())?;
        self.kv.truncate(m.len)?;
        self.seeds.clear();
        if let Some(s) = &m.seed {
            self.seeds.extend(vec![s.clone()]);
        }
        Ok(())
    }

    pub fn nbytes(&self) -> usize {
        self.kv.nbytes() + self.index.nbytes() + self.seeds.nbytes()
    }
}

/// The MTP layer; built once at load.
pub struct MtpLayer {
    layer: usize,
    dim: i64,
    eps: f32,
    w: LayerWeights,
    mla: Mla,
    indexer: Indexer,
    ffn: LayerFfn,
    /// MLA RoPE inverse frequencies (the commit path's own K rows).
    inv: Vec<f32>,
    /// `token_embd` as a packed embedding: the draft inputs.
    embedding: Embedding,
    bytes: u64,
    _guard: CacheLimitGuard,
}

/// The draft-input embedding over `token_embd`'s codes: rows equal to
/// `Dense::rows`' (f16 for Q8_0 codes).
fn draft_embedding(token_embd: &Dense, vocab: usize, dim: usize) -> Result<Embedding> {
    let (v, d) = (
        u32::try_from(vocab).map_err(|_| bad("vocab exceeds u32"))?,
        u32::try_from(dim).map_err(|_| bad("dim exceeds u32"))?,
    );
    match token_embd {
        Dense::Array(a) => Embedding::from_weight(a),
        Dense::Q8 { w, scales } => {
            let b = scales
                .astype(DType::Float32)?
                .mul_scalar(-128.0)?
                .astype(scales.dtype()?)?;
            MxArray::eval_arrays_with_context(&[&b], "hy_v4_flash MTP embedding")?;
            let mut e = Embedding::new_uninitialized(v, d)?;
            e.load_quantized_packed(w, scales, Some(&b), 32, 8, "affine")?;
            Ok(e)
        }
        Dense::Qmm { w, scales, biases } => {
            let mut e = Embedding::new_uninitialized(v, d)?;
            e.load_quantized_packed(w, scales, Some(biases), 32, 8, "affine")?;
            Ok(e)
        }
    }
}

impl MtpLayer {
    /// Reads blk.{n_layers} of the shards `paths` and builds the layer
    /// beside `model`. Model thread only.
    pub fn load(paths: &[PathBuf], model: &Model) -> Result<Self> {
        let scan = Scan::open(paths)?;
        let shards = ShardSet::open(&scan.paths)?;
        let w = mtp_weights::load(&scan, &shards)?;
        let tok = model.weights().token_embd();
        Self::new(model.cfg(), w, tok)
    }

    pub fn new(
        cfg: &super::config::HyV4FlashConfig,
        mw: MtpWeights,
        token_embd: &Dense,
    ) -> Result<Self> {
        let MtpWeights {
            layer,
            weights: w,
            experts,
            bytes,
            guard,
            ..
        } = mw;
        let l = cfg
            .layers
            .get(layer)
            .ok_or_else(|| bad(format!("layer {layer} of {}", cfg.layers.len())))?;
        if !l.mtp || layer != cfg.n_layers {
            return Err(bad(format!("layer {layer} is not the MTP layer")));
        }
        if !matches!(w.residual, Residual::Mtp(_)) || w.engram.is_some() {
            return Err(bad("trunk weights where the MTP layer is expected"));
        }
        let iw = w
            .indexer
            .as_ref()
            .ok_or_else(|| bad("the MTP layer has no indexer weights"))?;
        let g = Geometry::of(cfg, layer)?;
        let inv = inv_freqs(g.rope_dim, g.rope_theta, None);
        let indexer = Indexer::new(IndexerGeometry::of(cfg, layer)?, iw)?;
        let mla = Mla::new(g, &w.attn)?;
        let ffn = LayerFfn::new(
            &FfnConfig::of(cfg, layer)?,
            layer,
            &w.ffn,
            Some(ExpertSource::Resident(experts)),
        )?;
        let embedding = draft_embedding(token_embd, cfg.vocab_size, cfg.dim)?;
        Ok(Self {
            layer,
            dim: cfg.dim as i64,
            eps: cfg.norm_eps,
            w,
            mla,
            indexer,
            ffn,
            inv,
            embedding,
            bytes,
            _guard: guard,
        })
    }

    pub fn layer(&self) -> usize {
        self.layer
    }

    /// Bytes of the held weights.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    pub fn embedding(&self) -> &Embedding {
        &self.embedding
    }

    /// An empty draft for a sequence of at most `cap` trunk positions.
    pub fn new_draft(&self, cap: usize) -> Draft {
        let g = self.indexer.geometry();
        let cap = cap + DRAFT_SLACK;
        Draft {
            kv: KvRows::new(self.mla.geometry().row() as i64, cap),
            index: IndexerState::new(cap, g),
            seeds: Seeds::default(),
            kpool: g.kpool,
            cap,
        }
    }

    fn nextn(&self) -> Result<&NextnWeights> {
        match &self.w.residual {
            Residual::Mtp(n) => Ok(n),
            Residual::Hc { .. } => Err(bad("the MTP layer has HC weights")),
        }
    }

    /// `eh_proj · [rms(e)·enorm ; rms(h)·hnorm]` `[m, d]` f32.
    fn eh(&self, e: &MxArray, h: &MxArray) -> Result<MxArray> {
        let n = self.nextn()?;
        let a = rms_norm(&e.astype(DType::Float32)?, &n.enorm, self.eps)?;
        let b = rms_norm(&h.astype(DType::Float32)?, &n.hnorm, self.eps)?;
        qlinear(&MxArray::concatenate(&a, &b, 1)?, &n.eh_proj)
    }

    /// `rms(x)·attn_norm` and the K/V rows of `x [m, d]` at MTP rows
    /// `at..at + m` (`Mla::project` at RoPE `at + 1`, without the queries).
    fn latent(&self, x: &MxArray, at: usize) -> Result<(MxArray, MxArray)> {
        let g = self.mla.geometry();
        let w = &self.w.attn;
        let m = x.shape_at(0)?;
        let (kv_lora, rd) = (g.kv_lora as i64, g.rope_dim as i64);
        let pos: Vec<u64> = (at + 1..at + 1 + m as usize).map(|p| p as u64).collect();
        let (cos, sin) = rope_tables(&self.inv, &pos)?;
        let xn = rms_norm(&x.astype(DType::Float32)?, &w.norm, g.eps)?;
        let ckv = qlinear(&xn, &w.kv_a)?;
        let kv = rms_norm(&ckv.slice(&[0, 0], &[m, kv_lora])?, &w.kv_a_norm, g.eps)?;
        let k_pe = apply_partial_rope(
            &ckv.slice(&[0, kv_lora], &[m, kv_lora + rd])?,
            rd,
            &cos,
            &sin,
            false,
        )?;
        Ok((xn, MxArray::concatenate(&kv, &k_pe, -1)?))
    }

    fn indexer_weights(&self) -> Result<&super::weights::IndexerWeights> {
        self.w
            .indexer
            .as_ref()
            .ok_or_else(|| bad("the MTP layer has no indexer weights"))
    }

    /// Appends the exact rows pairing `h [m, d]` (trunk outputs) with the
    /// next tokens `ids` (m) at the end of `d`; their K/V and indexer keys
    /// only. On error `d` is as it was.
    pub fn commit(&self, model: &Model, d: &mut Draft, h: &MxArray, ids: &[u32]) -> Result<()> {
        let at = d.len();
        let m = ids.len();
        if m == 0 || h.shape_at(0)? as usize != m {
            return Err(bad(format!(
                "commit of {m} ids with {:?} hiddens",
                h.shape()?.as_ref()
            )));
        }
        if at + m > d.cap {
            return Err(bad(format!(
                "commit to {} past the {} rows a draft holds",
                at + m,
                d.cap
            )));
        }
        let tok = MxArray::from_uint32(ids, &[m as i64])?;
        let e = model.weights().token_embd().rows(&tok)?;
        let x = self.eh(&e, h)?;
        let (xn, kv) = self.latent(&x, at)?;
        let mark = d.index.mark();
        self.indexer
            .update(self.indexer_weights()?, &xn, at, &mut d.index)?;
        if let Err(e) = d.kv.write(&kv, at) {
            d.index.restore(mark)?;
            return Err(e);
        }
        Ok(())
    }

    /// One draft row at the end of `d` from `h [1, 1, d]` (a trunk or MTP
    /// output) and `emb [1, 1, d]` (the next token's embedding):
    /// `(h_mtp [1, 1, d], logits [1, vocab])`. On error `d` is as it was.
    pub fn draft(
        &self,
        model: &Model,
        d: &mut Draft,
        h: &MxArray,
        emb: &MxArray,
    ) -> Result<(MxArray, MxArray)> {
        let at = d.len();
        if at >= d.cap {
            return Err(bad(format!("draft row {at} past the {} rows held", d.cap)));
        }
        let x = self.eh(&emb.reshape(&[1, self.dim])?, &h.reshape(&[1, self.dim])?)?;
        let w = &self.w.attn;
        let p = self.mla.project(w, &x, at + 1)?;
        let mark = d.index.mark();
        let call = Call {
            x: &p.x,
            qr: &p.qr,
            start: at,
            kind: RowKind::Decode,
        };
        let idx = self
            .indexer
            .select(self.indexer_weights()?, call, &mut d.index)?;
        if let Err(e) = d.kv.write(&p.kv, at) {
            d.index.restore(mark)?;
            return Err(e);
        }
        let out = self.finish_draft(model, d, &x, &p.q, &p.x, &idx);
        if out.is_err() {
            d.kv.truncate(at)?;
            d.index.restore(mark)?;
        }
        out
    }

    fn finish_draft(
        &self,
        model: &Model,
        d: &Draft,
        x: &MxArray,
        q: &MxArray,
        xn: &MxArray,
        idx: &MxArray,
    ) -> Result<(MxArray, MxArray)> {
        let g = self.mla.geometry();
        let w = &self.w.attn;
        let o = gather_sink_attention(q, d.kv.rows()?, idx, g.kv_lora as i64, &w.sinks, g.scale)?;
        let x = x.add(&self.mla.finish(w, xn, &o)?)?;
        let (f, _) = self.ffn.forward(&self.w, &x, RowKind::Decode)?;
        let x = x.add(&f)?;
        let h = rms_norm(&x, &self.nextn()?.shared_head_norm, self.eps)?;
        let logits = head_rows(&h, model.weights().output())?;
        MxArray::eval_arrays_with_context(&[&h, &logits], "hy_v4_flash MTP draft")?;
        Ok((h.reshape(&[1, 1, self.dim])?, logits))
    }
}

/// A sequence: the trunk session and, with an MTP layer, its draft one row
/// behind (`lag`).
pub struct Lagged {
    trunk: Session,
    draft: Option<Draft>,
}

#[derive(Clone)]
enum DraftSnap {
    Off,
    At(DraftMark),
    /// Taken while the draft was not settled: never restored.
    Unsettled,
}

/// A point of a `Lagged` to go back to.
#[derive(Clone)]
pub struct LagMark {
    trunk: Mark,
    draft: DraftSnap,
}

impl Snapshot for LagMark {
    fn len(&self) -> usize {
        self.trunk.len()
    }
}

impl Lagged {
    pub fn new(trunk: Session, draft: Option<Draft>) -> Self {
        Self { trunk, draft }
    }

    pub fn trunk(&self) -> &Session {
        &self.trunk
    }

    pub fn draft(&self) -> Option<&Draft> {
        self.draft.as_ref()
    }

    pub fn parts(&mut self) -> (&mut Session, Option<&mut Draft>) {
        (&mut self.trunk, self.draft.as_mut())
    }

    pub fn len(&self) -> usize {
        self.trunk.len()
    }

    pub fn is_empty(&self) -> bool {
        self.trunk.is_empty()
    }

    pub fn tokens(&self) -> &[u32] {
        self.trunk.tokens()
    }

    pub fn nbytes(&self) -> usize {
        self.trunk.nbytes() + self.draft.as_ref().map_or(0, Draft::nbytes)
    }

    /// The draft length `truncate(len)` leaves, when it can.
    fn draft_fits(&self, len: usize) -> Result<usize> {
        let Some(d) = &self.draft else {
            return Ok(0);
        };
        let keep = len.saturating_sub(1);
        if d.len() < keep {
            return Err(bad(format!(
                "the draft holds {} rows, a trunk of {len} needs {keep}",
                d.len()
            )));
        }
        if len > 0 && d.seed(len - 1).is_none() {
            return Err(bad(format!("no seed for trunk row {}", len - 1)));
        }
        if !d.can_cut(keep) {
            return Err(bad(format!("the draft indexer cannot go back to {keep}")));
        }
        Ok(keep)
    }

    /// Whether `truncate(len)` would succeed.
    pub fn keeps(&self, len: usize, kpool: usize) -> bool {
        session_keeps(&self.trunk, len, kpool) && self.draft_fits(len).is_ok()
    }

    /// The draft cut to `trunk - 1` rows.
    pub fn settle(&mut self) -> Result<()> {
        let keep = self.trunk.len().saturating_sub(1);
        match self.draft.as_mut() {
            Some(d) if d.len() > keep => d.cut(keep),
            _ => Ok(()),
        }
    }

    fn mismatch() -> Error {
        bad("an MTP layer without a draft, or a draft without the layer")
    }

    /// One prefill chunk: the trunk rows and the MTP rows they close. On
    /// error the sequence is as it was.
    pub fn prefill(
        &mut self,
        model: &Model,
        mtp: Option<&MtpLayer>,
        ids: &[u32],
        want: Want,
        cancel: &dyn Fn(usize) -> bool,
    ) -> Result<Output> {
        let (m, d) = match (mtp, self.draft.as_mut()) {
            (None, None) => {
                return model.forward(
                    &mut self.trunk,
                    ids,
                    RowKind::Prefill,
                    want,
                    Some(cancel),
                    None,
                );
            }
            (Some(m), Some(d)) => (m, d),
            _ => return Err(Self::mismatch()),
        };
        let start = self.trunk.len();
        if d.len() != start.saturating_sub(1) {
            return Err(bad(format!(
                "a chunk at {start} with {} draft rows",
                d.len()
            )));
        }
        let mark = self.trunk.mark();
        let all = Want {
            hidden: true,
            ..want
        };
        let mut out = model.forward(
            &mut self.trunk,
            ids,
            RowKind::Prefill,
            all,
            Some(cancel),
            None,
        )?;
        let h = out
            .hidden
            .take()
            .ok_or_else(|| bad("the trunk returned no hidden rows"))?;
        if let Err(e) = lag_chunk(m, model, d, start, ids, &h) {
            return Err(match self.trunk.restore(&mark) {
                Ok(()) => e,
                Err(r) => bad(format!(
                    "{}; the trunk could not go back: {}",
                    e.reason, r.reason
                )),
            });
        }
        if want.hidden {
            out.hidden = Some(h);
        }
        Ok(out)
    }

    /// One decode row of `token`: the MTP row it closes (none when the
    /// draft already holds it), then the trunk row. Returns the last-row
    /// logits and, with a draft, the trunk output row `[1, d]`.
    pub fn step(
        &mut self,
        model: &Model,
        mtp: Option<&MtpLayer>,
        token: u32,
    ) -> Result<(MxArray, Option<MxArray>)> {
        let (m, d) = match (mtp, self.draft.as_mut()) {
            (None, None) => {
                let l = model.forward_last(&mut self.trunk, &[token], RowKind::Decode, None)?;
                return Ok((l, None));
            }
            (Some(m), Some(d)) => (m, d),
            _ => return Err(Self::mismatch()),
        };
        let p = self.trunk.len();
        let at = d.len();
        if at + 1 == p {
            let seed = d
                .seed(at)
                .cloned()
                .ok_or_else(|| bad(format!("no seed for trunk row {at}")))?;
            m.commit(model, d, &seed, &[token])?;
        } else if at != p {
            return Err(bad(format!("a decode row at {p} with {at} draft rows")));
        }
        let want = Want {
            logits: super::model::Logits::Last,
            hidden: true,
        };
        let out = model
            .forward(&mut self.trunk, &[token], RowKind::Decode, want, None, None)
            .and_then(|o| {
                let h = o
                    .hidden
                    .ok_or_else(|| bad("the trunk returned no hidden row"))?;
                let l = o
                    .logits
                    .ok_or_else(|| bad("the trunk returned no logits"))?;
                Ok((l, h))
            });
        let (logits, h) = match out {
            Ok(v) => v,
            Err(e) => {
                d.cut(at)?;
                return Err(e);
            }
        };
        match Seeds::copies(&h, p) {
            Ok(rows) => d.seeds.extend(rows),
            Err(e) => {
                self.trunk.truncate(p)?;
                d.cut(at)?;
                return Err(e);
            }
        }
        Ok((logits, Some(h)))
    }

    fn clear(&mut self) -> Result<()> {
        self.trunk.truncate(0)?;
        if let Some(d) = self.draft.as_mut() {
            d.cut(0)?;
            d.seeds.clear();
        }
        Ok(())
    }
}

/// The MTP rows a prefill chunk of `ids` at `start` closes, from the
/// chunk's trunk outputs `h [n, d]`; the chunk's last rows become seeds.
fn lag_chunk(
    m: &MtpLayer,
    model: &Model,
    d: &mut Draft,
    start: usize,
    ids: &[u32],
    h: &MxArray,
) -> Result<()> {
    let n = ids.len();
    let l = lag(start, n);
    let seeds = Seeds::copies(h, start)?;
    if l.rows > 0 {
        let dim = h.shape_at(1)?;
        let body = (n > 1)
            .then(|| h.slice(&[0, 0], &[n as i64 - 1, dim]))
            .transpose()?;
        let hs = match (l.seed, body) {
            (true, body) => {
                let seed = d
                    .seed(l.at)
                    .ok_or_else(|| bad(format!("no seed for trunk row {}", l.at)))?;
                match body {
                    Some(b) => MxArray::concatenate(seed, &b, 0)?,
                    None => seed.clone(),
                }
            }
            (false, Some(b)) => b,
            (false, None) => return Err(bad("a lag of rows without hidden rows")),
        };
        m.commit(model, d, &hs, &ids[n - l.rows..])?;
    }
    d.seeds.extend(seeds);
    Ok(())
}

impl Seq for Lagged {
    type Snap = LagMark;

    fn len(&self) -> usize {
        self.trunk.len()
    }

    fn tokens(&self) -> &[u32] {
        self.trunk.tokens()
    }

    fn cap(&self) -> usize {
        self.trunk.cap()
    }

    fn truncate(&mut self, len: usize) -> Result<()> {
        if len == 0 {
            return self.clear();
        }
        let keep = self.draft_fits(len)?;
        self.trunk.truncate(len)?;
        if let Some(d) = self.draft.as_mut() {
            d.cut(keep)?;
            d.seeds.cut(len);
        }
        Ok(())
    }

    fn snap(&self) -> LagMark {
        let len = self.trunk.len();
        LagMark {
            trunk: self.trunk.mark(),
            draft: match &self.draft {
                None => DraftSnap::Off,
                Some(d) => d.mark(len).map_or(DraftSnap::Unsettled, DraftSnap::At),
            },
        }
    }

    fn restore(&mut self, s: &LagMark) -> Result<()> {
        let dm = match (&self.draft, &s.draft) {
            (None, DraftSnap::Off) => return self.trunk.restore(&s.trunk),
            (Some(d), DraftSnap::At(dm)) if d.can_restore(dm) => dm,
            _ => return Err(bad("the mark does not fit this sequence's draft")),
        };
        self.trunk.restore(&s.trunk)?;
        let out = match self.draft.as_mut() {
            Some(d) => d.restore(dm),
            None => Err(Self::mismatch()),
        };
        if let Err(e) = out {
            return Err(match self.clear() {
                Ok(()) => e,
                Err(c) => bad(format!(
                    "{}; the sequence could not be cleared: {}",
                    e.reason, c.reason
                )),
            });
        }
        Ok(())
    }

    fn can_restore(&self, s: &LagMark) -> bool {
        let draft = match (&self.draft, &s.draft) {
            (None, DraftSnap::Off) => true,
            (Some(d), DraftSnap::At(dm)) => d.can_restore(dm),
            _ => false,
        };
        draft && self.trunk.can_restore(&s.trunk)
    }
}

/// Runs prefill chunks on a `Lagged`: the trunk model and its MTP layer.
pub struct LagRunner<'a> {
    pub model: &'a Model,
    pub mtp: Option<&'a MtpLayer>,
}

impl Runner<Lagged> for LagRunner<'_> {
    type Out = Output;

    fn can_truncate(&self, seq: &Lagged, len: usize) -> bool {
        seq.keeps(len, self.model.cfg().indexer.kpool)
    }

    fn run(
        &self,
        seq: &mut Lagged,
        ids: &[u32],
        want: Want,
        cancel: &dyn Fn(usize) -> bool,
    ) -> Result<Output> {
        seq.prefill(self.model, self.mtp, ids, want, cancel)
    }
}
